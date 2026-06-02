"""
単勝モデル 市場エッジ診断スクリプト
===================================
既存のバックテスト成果物だけを使い、以下を診断する:
1. edge_ratio (p_ai / p_market) 分位別 ROI
2. pred_ev (p_ai × odds) 帯別 ROI
3. オッズ帯別 ROI
4. turf/dirt 別分析
5. 補正前後比較 (p_win_pred → p_win_corrected → p_win_final)
6. p_ai と p_market の相関

使用データ:
  - data/backtest/bt_{year}_horse_diagnostics.csv  (2024, 2025)
  - data/backtest/bt_{year}_horse_features.parquet  (2024, 2025)
  - data/backtest/shadow_mawc_selective_best/shadow_horse_diff.parquet (参考)

出力:
  - data/analysis/win_market_edge_diagnostic.json
  - data/analysis/win_market_edge_diagnostic.md
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
BACKTEST_DIR = DATA_DIR / "backtest"

# ── ユーティリティ ──────────────────────────────────────────────

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def calc_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary logloss."""
    eps = 1e-15
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def calc_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def calc_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_pred = y_pred[mask].mean()
        avg_true = y_true[mask].mean()
        ece += abs(avg_pred - avg_true) * mask.sum() / n
    return float(ece)


def roi_summary(
    df: pd.DataFrame,
    odds_col: str = "tanodds",
    win_col: str = "is_win",
) -> dict:
    """ROI・的中率などの基本統計を返す。"""
    n = len(df)
    if n == 0:
        return {
            "count": 0, "hit_rate": 0.0, "avg_odds": 0.0,
            "roi": 0.0, "profit": 0.0,
        }
    hits = df[win_col].sum()
    total_payout = (df[odds_col] * df[win_col]).sum()  # 的中時払戻額合計
    total_stake = n  # 100円固定 = n * 100 → ROI = payout / stake * 100
    hit_rate = float(hits / n)
    avg_odds = float(df[odds_col].mean())
    roi = float(total_payout / total_stake)  # 1.0 = 100%
    profit = float(total_payout - total_stake)  # 単位: 100円あたり
    return {
        "count": int(n),
        "hit_rate": round(hit_rate, 4),
        "avg_odds": round(avg_odds, 2),
        "roi": round(roi, 4),
        "profit": round(profit, 1),
    }


def edge_ratio_quantile_summary(
    df: pd.DataFrame,
    quantile_edges: list[float],
    odds_col: str = "tanodds",
    edge_col: str = "edge_ratio",
    win_col: str = "is_win",
    p_ai_col: str = "p_ai",
    p_market_col: str = "p_market",
    ev_col: str = "pred_ev",
) -> list[dict]:
    """edge_ratio を分位で区切って ROI サマリを返す。"""
    results = []
    labels = [
        "下位 (0-20%)", "中下位 (20-40%)", "中央 (40-60%)",
        "中上位 (60-80%)", "上位 (80-95%)", "最上位 (95-99%)", "極上位 (99-100%)",
    ]
    for i in range(len(quantile_edges) - 1):
        lo = quantile_edges[i]
        hi = quantile_edges[i + 1]
        sub = df[(df[edge_col] >= lo) & (df[edge_col] < hi)]
        base = roi_summary(sub, odds_col, win_col)
        base["label"] = labels[i] if i < len(labels) else f"{lo:.3f}-{hi:.3f}"
        base["edge_range"] = f"{lo:.3f} - {hi:.3f}"
        if len(sub) > 0:
            base["avg_p_ai"] = round(float(sub[p_ai_col].mean()), 4)
            base["avg_p_market"] = round(float(sub[p_market_col].mean()), 4)
            base["avg_pred_ev"] = round(float(sub[ev_col].mean()), 4)
        results.append(base)
    return results


def pred_ev_band_summary(
    df: pd.DataFrame,
    bands: list[tuple[float, float | None]],
    ev_col: str = "pred_ev",
    odds_col: str = "tanodds",
    win_col: str = "is_win",
) -> list[dict]:
    """pred_ev 帯別 ROI サマリ。"""
    results = []
    for lo, hi in bands:
        if hi is None:
            sub = df[df[ev_col] >= lo]
            label = f"EV {lo:.2f}+"
        else:
            sub = df[(df[ev_col] >= lo) & (df[ev_col] < hi)]
            label = f"EV {lo:.2f}-{hi:.2f}"
        base = roi_summary(sub, odds_col, win_col)
        base["label"] = label
        if len(sub) > 0:
            base["avg_pred_ev"] = round(float(sub[ev_col].mean()), 4)
        results.append(base)
    return results


def odds_band_summary(
    df: pd.DataFrame,
    bands: list[tuple[float, float | None]],
    odds_col: str = "tanodds",
    win_col: str = "is_win",
    p_ai_col: str = "p_ai",
    p_market_col: str = "p_market",
    ev_col: str = "pred_ev",
) -> list[dict]:
    """オッズ帯別 ROI サマリ。"""
    results = []
    for lo, hi in bands:
        if hi is None:
            sub = df[df[odds_col] >= lo]
            label = f"{lo:.1f}倍+"
        else:
            sub = df[(df[odds_col] >= lo) & (df[odds_col] < hi)]
            label = f"{lo:.1f}-{hi:.1f}倍"
        base = roi_summary(sub, odds_col, win_col)
        base["label"] = label
        if len(sub) > 0:
            base["avg_p_ai"] = round(float(sub[p_ai_col].mean()), 4)
            base["avg_p_market"] = round(float(sub[p_market_col].mean()), 4)
            base["avg_pred_ev"] = round(float(sub[ev_col].mean()), 4)
        results.append(base)
    return results


def correction_comparison(
    df: pd.DataFrame,
    prob_cols: list[str],
    win_col: str = "is_win",
) -> list[dict]:
    """補正段階ごとに Logloss / Brier / ECE / APR を計算。"""
    y_true = df[win_col].values.astype(float)
    results = []
    for col in prob_cols:
        if col not in df.columns:
            continue
        y_pred = df[col].values.astype(float)
        valid = ~(np.isnan(y_pred) | np.isnan(y_true))
        if valid.sum() < 100:
            continue
        y_pred_v = y_pred[valid]
        y_true_v = y_true[valid]
        apr = safe_div(float(y_true_v.sum()), float(y_pred_v.sum()))
        results.append({
            "stage": col,
            "logloss": round(calc_logloss(y_true_v, y_pred_v), 4),
            "brier": round(calc_brier(y_true_v, y_pred_v), 4),
            "ece": round(calc_ece(y_true_v, y_pred_v), 4),
            "apr": round(apr, 4),
            "count": int(valid.sum()),
        })
    return results


# ── メイン処理 ──────────────────────────────────────────────────

def load_and_merge_data() -> pd.DataFrame:
    """horse_diagnostics (CSV) + horse_features (Parquet) を結合。"""
    frames = []
    for year in [2024, 2025]:
        diag_path = BACKTEST_DIR / f"bt_{year}_horse_diagnostics.csv"
        feat_path = BACKTEST_DIR / f"bt_{year}_horse_features.parquet"
        if not diag_path.exists() or not feat_path.exists():
            print(f"  [skip] {year}: file not found")
            continue

        diag = pd.read_csv(diag_path)
        feat = pd.read_parquet(feat_path, columns=[
            "race_id", "umaban", "kakuteijyuni", "surface",
            "tanodds", "confirmed_odds", "popularity_rank",
        ])

        # race_id の型を揃える (CSV=object, Parquet=object/int64 の場合がある)
        diag["race_id"] = diag["race_id"].astype(str)
        feat["race_id"] = feat["race_id"].astype(str)
        diag["umaban"] = diag["umaban"].astype(int)
        feat["umaban"] = feat["umaban"].astype(int)

        merged = diag.merge(feat, on=["race_id", "umaban"], how="inner", suffixes=("", "_feat"))
        merged["year"] = year
        frames.append(merged)
        print(f"  {year}: diagnostics={len(diag)}, features={len(feat)}, merged={len(merged)}")

    if not frames:
        raise FileNotFoundError("No data files found for 2024/2025")

    df = pd.concat(frames, ignore_index=True)
    print(f"  Total merged: {len(df)} rows")
    return df


def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """p_market, edge_ratio, pred_ev, is_win 等を計算。"""
    # ── 市場確率 (p_market) を計算 ──
    # tanodds が欠損/ゼロの行を除外
    df = df[df["tanodds"] > 0].copy()

    # レース内正規化された市場確率
    # 既に p_market_win_norm があればそれを使う、なければ自前計算
    if "p_market_win_norm" in df.columns and df["p_market_win_norm"].notna().sum() > 0:
        df["p_market"] = df["p_market_win_norm"].astype(float)
    else:
        p_raw = 1.0 / df["tanodds"]
        p_sum = p_raw.groupby(df["race_id"]).transform("sum")
        df["p_market"] = (p_raw / p_sum).astype(float)

    # p_ai: p_win_final > p_win_corrected > p_win_pred の優先順位
    for col in ["p_win_final", "p_win_corrected", "p_win_pred"]:
        if col in df.columns:
            df["p_ai"] = df[col].astype(float)
            break

    # is_win (1着フラグ)
    df["is_win"] = (df["kakuteijyuni"] == 1).astype(int)

    # edge 指標
    df["edge_ratio"] = (df["p_ai"] / df["p_market"]).clip(0.01, 100.0)
    df["edge_diff"] = df["p_ai"] - df["p_market"]
    df["pred_ev"] = df["p_ai"] * df["tanodds"]

    # 無効な行を除外
    df = df.dropna(subset=["p_ai", "p_market", "tanodds", "kakuteijyuni"])
    df = df[df["tanodds"] > 0]

    return df


def run_analysis() -> dict:
    """全分析を実行して結果 dict を返す。"""
    print("=== 単勝モデル 市場エッジ診断 ===\n")

    # 1) データ読み込み
    print("[1] データ読み込み・結合...")
    df = load_and_merge_data()

    print("\n[2] 派生列を計算...")
    df = compute_derived_columns(df)
    print(f"  Valid rows: {len(df)}")
    print(f"  Surface: {df['surface'].value_counts().to_dict()}")
    print(f"  p_ai range: [{df['p_ai'].min():.4f}, {df['p_ai'].max():.4f}]")
    print(f"  p_market range: [{df['p_market'].min():.4f}, {df['p_market'].max():.4f}]")
    print(f"  tanodds range: [{df['tanodds'].min():.1f}, {df['tanodds'].max():.1f}]")

    results: dict = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "total_rows": len(df),
            "years": sorted(df["year"].unique().tolist()),
            "surfaces": df["surface"].value_counts().to_dict(),
            "input_files": [
                f"bt_{y}_horse_diagnostics.csv" for y in [2024, 2025]
            ] + [
                f"bt_{y}_horse_features.parquet" for y in [2024, 2025]
            ],
        },
    }

    # ── 3) edge_ratio 分位別 ROI ──
    print("\n[3] edge_ratio 分位別 ROI...")
    quantiles = [0.0, 0.20, 0.40, 0.60, 0.80, 0.95, 0.99, 1.0]
    q_edges = df["edge_ratio"].quantile(quantiles).tolist()
    print(f"  Quantile edges: {[round(q, 3) for q in q_edges]}")
    results["edge_ratio_quantile_roi"] = edge_ratio_quantile_summary(df, q_edges)
    for r in results["edge_ratio_quantile_roi"]:
        print(f"    {r['label']}: n={r['count']}, ROI={r['roi']:.4f}, hit={r['hit_rate']:.4f}")

    # ── 4) pred_ev 帯別 ROI ──
    print("\n[4] pred_ev 帯別 ROI...")
    ev_bands = [
        (0.0, 0.90), (0.90, 1.00), (1.00, 1.10), (1.10, 1.20),
        (1.20, 1.30), (1.30, 1.50), (1.50, None),
    ]
    results["pred_ev_band_roi"] = pred_ev_band_summary(df, ev_bands)
    for r in results["pred_ev_band_roi"]:
        print(f"    {r['label']}: n={r['count']}, ROI={r['roi']:.4f}, hit={r['hit_rate']:.4f}")

    # ── 5) オッズ帯別 ROI ──
    print("\n[5] オッズ帯別 ROI...")
    odds_bands = [
        (1.0, 3.0), (3.0, 5.0), (5.0, 10.0),
        (10.0, 20.0), (20.0, 50.0), (50.0, None),
    ]
    results["odds_band_roi"] = odds_band_summary(df, odds_bands)
    for r in results["odds_band_roi"]:
        print(f"    {r['label']}: n={r['count']}, ROI={r['roi']:.4f}, hit={r['hit_rate']:.4f}")

    # ── 6) Surface 別分析 ──
    print("\n[6] Surface 別分析...")
    results["surface_analysis"] = {}
    for surface in ["turf", "dirt"]:
        sub = df[df["surface"] == surface]
        if len(sub) == 0:
            continue
        print(f"\n  === {surface.upper()} ({len(sub)} rows) ===")
        surf_data: dict = {"total": roi_summary(sub)}

        # edge_ratio 分位
        q_edges_s = sub["edge_ratio"].quantile(quantiles).tolist()
        surf_data["edge_ratio_quantile"] = edge_ratio_quantile_summary(sub, q_edges_s)
        for r in surf_data["edge_ratio_quantile"]:
            print(f"    {r['label']}: n={r['count']}, ROI={r['roi']:.4f}")

        # オッズ帯
        surf_data["odds_band"] = odds_band_summary(sub, odds_bands)

        # EV帯
        surf_data["pred_ev_band"] = pred_ev_band_summary(sub, ev_bands)

        results["surface_analysis"][surface] = surf_data

    # ── 7) 補正前後比較 ──
    print("\n[7] 補正前後比較...")
    correction_cols = []
    for col in ["p_win_pred", "p_win_corrected", "p_win_final"]:
        if col in df.columns:
            correction_cols.append(col)
    print(f"  Available correction stages: {correction_cols}")
    results["correction_comparison"] = correction_comparison(df, correction_cols)
    for r in results["correction_comparison"]:
        print(f"    {r['stage']}: logloss={r['logloss']:.4f}, brier={r['brier']:.4f}, "
              f"ECE={r['ece']:.4f}, APR={r['apr']:.4f}")

    # 補正段階別に edge_ratio 分位 ROI も比較
    if len(correction_cols) >= 2:
        print("\n  補正段階別 edge_ratio 上位5% ROI:")
        results["correction_edge_roi"] = []
        for col in correction_cols:
            p_ai_backup = df["p_ai"].copy()
            df["p_ai"] = df[col].astype(float)
            df["edge_ratio"] = (df["p_ai"] / df["p_market"]).clip(0.01, 100.0)
            df["pred_ev"] = df["p_ai"] * df["tanodds"]

            q_edges_c = df["edge_ratio"].quantile(quantiles).tolist()
            # 上位5% = quantile[5] (95%) 以上
            top5 = df[df["edge_ratio"] >= q_edges_c[5]]
            base = roi_summary(top5)
            base["stage"] = col
            base["top5_edge_threshold"] = round(q_edges_c[5], 3)
            results["correction_edge_roi"].append(base)
            print(f"    {col}: n={base['count']}, ROI={base['roi']:.4f}, "
                  f"hit={base['hit_rate']:.4f}, avg_odds={base['avg_odds']:.2f}")

            # EV>1.2 の ROI も
            ev_high = df[df["pred_ev"] > 1.2]
            ev_high_stats = roi_summary(ev_high)
            ev_high_stats["stage"] = col
            ev_high_stats["threshold"] = "EV>1.2"
            results["correction_edge_roi"].append(ev_high_stats)
            print(f"    {col} (EV>1.2): n={ev_high_stats['count']}, "
                  f"ROI={ev_high_stats['roi']:.4f}")

        # 復元
        df["p_ai"] = p_ai_backup
        df["edge_ratio"] = (df["p_ai"] / df["p_market"]).clip(0.01, 100.0)
        df["pred_ev"] = df["p_ai"] * df["tanodds"]

    # ── 8) p_ai と p_market の相関 ──
    print("\n[8] p_ai と p_market の相関...")
    corr_data: dict = {}
    # 全体
    corr_all = float(df["p_ai"].corr(df["p_market"]))
    corr_data["overall"] = round(corr_all, 4)
    print(f"  全体相関: {corr_all:.4f}")

    # Surface 別
    for surface in ["turf", "dirt"]:
        sub = df[df["surface"] == surface]
        if len(sub) > 0:
            c = float(sub["p_ai"].corr(sub["p_market"]))
            corr_data[surface] = round(c, 4)
            print(f"  {surface}: {c:.4f}")

    # オッズ帯別
    corr_data["by_odds_band"] = {}
    for lo, hi in odds_bands:
        if hi is None:
            sub = df[df["tanodds"] >= lo]
            label = f"{lo:.0f}倍+"
        else:
            sub = df[(df["tanodds"] >= lo) & (df["tanodds"] < hi)]
            label = f"{lo:.0f}-{hi:.0f}倍"
        if len(sub) > 10:
            c = float(sub["p_ai"].corr(sub["p_market"]))
            corr_data["by_odds_band"][label] = round(c, 4)
            print(f"  {label}: {c:.4f}")

    results["correlation"] = corr_data

    # ── 9) 市場ベースライン ROI ──
    print("\n[9] 市場ベースライン (全馬一律買い) ROI...")
    total_stake = len(df)
    total_payout = (df["tanodds"] * df["is_win"]).sum()
    baseline_roi = safe_div(total_payout, total_stake)
    results["market_baseline"] = {
        "total_rows": total_stake,
        "total_payout": round(float(total_payout), 1),
        "roi": round(baseline_roi, 4),
        "note": "全馬100円買いした場合のROI。JRA控除率約25%なので理論値は0.75前後",
    }
    print(f"  全馬買い ROI: {baseline_roi:.4f}")

    # ── 10) Shadow comparison 参考 ──
    shadow_path = BACKTEST_DIR / "shadow_mawc_selective_best" / "shadow_horse_diff.parquet"
    if shadow_path.exists():
        print("\n[10] Shadow comparison 読み込み...")
        sh = pd.read_parquet(shadow_path)
        print(f"  shadow_horse_diff: {len(sh)} rows")
        results["meta"]["shadow_file"] = str(shadow_path.name)
        results["meta"]["shadow_rows"] = len(sh)

    return results, df


# ── Markdown レポート生成 ────────────────────────────────────────

def generate_markdown(results: dict) -> str:
    """分析結果を初心者向け Markdown レポートにする。"""
    lines: list[str] = []

    lines.append("# 単勝モデル 市場エッジ診断レポート")
    lines.append(f"\n生成日時: {results['meta']['generated_at']}")
    lines.append(f"分析対象: {results['meta']['years']}年 / {results['meta']['total_rows']}レコード")
    lines.append(f"馬場: {', '.join(f'{k}={v}' for k, v in results['meta']['surfaces'].items())}")

    # ── 1. 結論 ──
    lines.append("\n---\n")
    lines.append("## 1. 結論")

    # 判定ロジック
    edge_roi = results["edge_ratio_quantile_roi"]
    top_roi = edge_roi[-1]["roi"] if edge_roi else 0  # 最上位 (99-100%)
    high_roi = edge_roi[-2]["roi"] if len(edge_roi) >= 2 else 0  # 上位 (95-99%)
    mid_roi = edge_roi[3]["roi"] if len(edge_roi) >= 4 else 0  # 中上位 (60-80%)
    low_roi = edge_roi[0]["roi"] if edge_roi else 0  # 下位

    ev_roi = results["pred_ev_band_roi"]
    ev_high_roi = next((r for r in ev_roi if "1.50" in r["label"]), {}).get("roi", 0)
    ev_mid_roi = next((r for r in ev_roi if "1.10-1.20" in r["label"]), {}).get("roi", 0)

    corr = results["correlation"]["overall"]
    corr_surface = results["correlation"].get("turf", 0)

    # 判定
    verdict = "D"  # default: データ不足
    verdict_reason = ""

    if edge_roi and ev_roi:
        if top_roi > 1.05 and high_roi > 1.0 and high_roi > mid_roi:
            verdict = "A"
            verdict_reason = (
                f"edge_ratio 最上位のROI={top_roi:.2f}, 上位5%のROI={high_roi:.2f} であり、\n"
                "AIは市場より過小評価された馬を正しく見つけている。"
            )
        elif ev_high_roi < 0.90 and high_roi < 0.95:
            verdict = "B"
            verdict_reason = (
                f"EV 1.50+のROI={ev_high_roi:.2f}, edge上位5%のROI={high_roi:.2f} であり、\n"
                "EV指標が買い指標として壊れている可能性が高い。"
            )
        elif corr > 0.95:
            verdict = "B"
            verdict_reason = (
                f"p_aiとp_marketの相関が {corr:.3f} と極めて高い。\n"
                "AIは市場を正確に再現しているが、市場を出し抜く差分を捉えられていない。"
            )
        elif high_roi > 1.0 and mid_roi > high_roi:
            verdict = "C"
            verdict_reason = (
                f"edge上位5%のROI={high_roi:.2f} は100%を超えているが、\n"
                f"中位帯のROI={mid_roi:.2f} の方が高いなど、帯によって有効/無効が分かれている。"
            )
        elif high_roi < 0.95:
            verdict = "B"
            verdict_reason = (
                f"edge上位5%のROI={high_roi:.2f} と低く、\n"
                "AIは「市場より過小評価された馬」を見つけられていない。"
            )
        else:
            verdict = "C"
            verdict_reason = "オッズ帯・条件によって有効/無効が分かれており、条件別閾値が必要。"

    lines.append(f"\n**判定: {verdict}**\n")
    lines.append(verdict_reason)

    lines.append("\n### 判定基準")
    lines.append("| 判定 | 意味 |")
    lines.append("|------|------|")
    lines.append("| A | EVは買い指標として機能している |")
    lines.append("| B | EV上位帯のROIが低く、EV指標が壊れている可能性 |")
    lines.append("| C | 条件によって有効/無効が分かれている |")
    lines.append("| D | 必要な列が不足 |")

    # ── 2. 使用した入力ファイル ──
    lines.append("\n---\n")
    lines.append("## 2. 使用した入力ファイル")
    for f in results["meta"]["input_files"]:
        lines.append(f"- `data/backtest/{f}`")
    lines.append(f"- 結合キー: `race_id`, `umaban`")
    lines.append(f"- 分析対象年: {results['meta']['years']}")
    lines.append("")
    lines.append("### 使用した主要列")
    lines.append("| 列名 | 用途 |")
    lines.append("|------|------|")
    lines.append("| `p_win_final` | AI推定勝率（最終版） |")
    lines.append("| `p_market_win_norm` | 市場確率（正規化済み） |")
    lines.append("| `tanodds` | 購入時点単勝オッズ（t-5min） |")
    lines.append("| `kakuteijyuni` | 確定着順（1=勝利） |")
    lines.append("| `surface` | 馬場（turf/dirt） |")

    missing_notes = []
    if "p_ability_win" not in results.get("meta", {}):
        missing_notes.append("`p_ability_win` (Stage1能力確率) — horse_featuresに存在するが diagnostics にないため未使用")
    if missing_notes:
        lines.append("\n### 不足・未使用列")
        for n in missing_notes:
            lines.append(f"- {n}")

    # ── 3. 主要テーブル ──
    lines.append("\n---\n")
    lines.append("## 3. 主要テーブル")

    # 市場ベースライン
    bl = results["market_baseline"]
    lines.append(f"\n### 市場ベースライン（全馬100円買い）")
    lines.append(f"- 対象: {bl['total_rows']}件")
    lines.append(f"- ROI: **{bl['roi']:.2%}** (= 100円あたり {bl['roi']*100:.1f}円戻り)")
    lines.append(f"- {bl['note']}")

    # edge_ratio 分位別 ROI
    lines.append("\n### edge_ratio 分位別 ROI")
    lines.append("> edge_ratio = p_ai / p_market。1.0より大きい = AIが市場より高く評価")
    lines.append("")
    lines.append("| 分位 | edge_ratio範囲 | 件数 | 的中率 | 平均オッズ | ROI | 平均p_ai | 平均p_market | 平均pred_ev |")
    lines.append("|------|---------------|------|--------|-----------|-----|---------|-------------|------------|")
    for r in results["edge_ratio_quantile_roi"]:
        lines.append(
            f"| {r['label']} | {r['edge_range']} | {r['count']:,} | "
            f"{r['hit_rate']:.1%} | {r['avg_odds']:.1f} | **{r['roi']:.2%}** | "
            f"{r.get('avg_p_ai', 0):.3f} | {r.get('avg_p_market', 0):.3f} | "
            f"{r.get('avg_pred_ev', 0):.3f} |"
        )

    # pred_ev 帯別 ROI
    lines.append("\n### pred_ev 帯別 ROI")
    lines.append("> pred_ev = p_ai × 単勝オッズ。1.0より大きい = プラス期待値")
    lines.append("")
    lines.append("| EV帯 | 件数 | 的中率 | 平均オッズ | ROI | 平均pred_ev |")
    lines.append("|------|------|--------|-----------|-----|------------|")
    for r in results["pred_ev_band_roi"]:
        lines.append(
            f"| {r['label']} | {r['count']:,} | {r['hit_rate']:.1%} | "
            f"{r['avg_odds']:.1f} | **{r['roi']:.2%}** | "
            f"{r.get('avg_pred_ev', 0):.3f} |"
        )

    # オッズ帯別 ROI
    lines.append("\n### オッズ帯別 ROI")
    lines.append("")
    lines.append("| オッズ帯 | 件数 | 的中率 | 平均オッズ | ROI | 平均p_ai | 平均p_market | 平均pred_ev |")
    lines.append("|---------|------|--------|-----------|-----|---------|-------------|------------|")
    for r in results["odds_band_roi"]:
        lines.append(
            f"| {r['label']} | {r['count']:,} | {r['hit_rate']:.1%} | "
            f"{r['avg_odds']:.1f} | **{r['roi']:.2%}** | "
            f"{r.get('avg_p_ai', 0):.3f} | {r.get('avg_p_market', 0):.3f} | "
            f"{r.get('avg_pred_ev', 0):.3f} |"
        )

    # Surface 別
    lines.append("\n### Surface 別サマリ")
    for surface, surf_data in results["surface_analysis"].items():
        total = surf_data["total"]
        lines.append(f"\n#### {surface.upper()}")
        lines.append(f"- 件数: {total['count']:,}, 的中率: {total['hit_rate']:.1%}, ROI: **{total['roi']:.2%}**")

        # edge_ratio 分位
        if "edge_ratio_quantile" in surf_data:
            lines.append("")
            lines.append("| edge_ratio分位 | 件数 | ROI | 的中率 |")
            lines.append("|---------------|------|-----|--------|")
            for r in surf_data["edge_ratio_quantile"]:
                lines.append(f"| {r['label']} | {r['count']:,} | **{r['roi']:.2%}** | {r['hit_rate']:.1%} |")

        # オッズ帯
        if "odds_band" in surf_data:
            lines.append("")
            lines.append("| オッズ帯 | 件数 | ROI | 的中率 |")
            lines.append("|---------|------|-----|--------|")
            for r in surf_data["odds_band"]:
                lines.append(f"| {r['label']} | {r['count']:,} | **{r['roi']:.2%}** | {r['hit_rate']:.1%} |")

    # 補正前後比較
    lines.append("\n### 補正前後比較")
    lines.append("> 各補正段階の確率品質と、edge上位5%でのROI比較")
    lines.append("")
    lines.append("| 段階 | Logloss | Brier | ECE | APR | 件数 |")
    lines.append("|------|---------|-------|-----|-----|------|")
    for r in results["correction_comparison"]:
        lines.append(
            f"| `{r['stage']}` | {r['logloss']:.4f} | {r['brier']:.4f} | "
            f"{r['ece']:.4f} | {r['apr']:.4f} | {r['count']:,} |"
        )

    if "correction_edge_roi" in results:
        lines.append("")
        lines.append("| 段階 | 条件 | 件数 | ROI | 的中率 | 平均オッズ |")
        lines.append("|------|------|------|-----|--------|-----------|")
        for r in results["correction_edge_roi"]:
            cond = r.get("threshold", "edge上位5%")
            lines.append(
                f"| `{r['stage']}` | {cond} | {r['count']:,} | "
                f"**{r['roi']:.2%}** | {r['hit_rate']:.1%} | {r['avg_odds']:.1f} |"
            )

    # 相関
    lines.append("\n### p_ai と p_market の相関")
    lines.append("> 1.0に近い = AIが市場を忠実に再現している（＝エッジが小さい）")
    lines.append("")
    lines.append(f"- **全体相関: {corr:.4f}**")
    for surface in ["turf", "dirt"]:
        if surface in results["correlation"]:
            lines.append(f"- {surface}: {results['correlation'][surface]:.4f}")
    if "by_odds_band" in results["correlation"]:
        lines.append("")
        lines.append("| オッズ帯 | 相関 |")
        lines.append("|---------|------|")
        for band, c in results["correlation"]["by_odds_band"].items():
            lines.append(f"| {band} | {c:.4f} |")

    # ── 4. 判断 ──
    lines.append("\n---\n")
    lines.append("## 4. 判断")
    lines.append(f"\n**判定: {verdict}**\n")
    lines.append(verdict_reason)

    # 追加の定性的判断
    lines.append("\n### 詳細判断")

    if edge_roi:
        top_r = edge_roi[-1]
        mid_r = edge_roi[3]
        lines.append(f"\n1. **edge_ratio 最上位(99-100%)のROI = {top_r['roi']:.2%}**")
        if top_r["roi"] > 1.0:
            lines.append("   - ✅ 市場より大幅に高く評価した馬は、実際に利益を出している")
        else:
            lines.append("   - ❌ 市場より大幅に高く評価した馬が、利益を出せていない")

        lines.append(f"\n2. **edge_ratio 中位(60-80%)のROI = {mid_r['roi']:.2%}**")
        lines.append(f"   - AIが「市場並み」と判断した馬のROI")

    if ev_roi:
        ev_high = next((r for r in ev_roi if "1.50" in r["label"]), None)
        if ev_high:
            lines.append(f"\n3. **EV 1.50+ のROI = {ev_high['roi']:.2%}** (n={ev_high['count']:,})")
            if ev_high["roi"] > 1.1:
                lines.append("   - ✅ 高EV馬は実際に高い回収率を達成している")
            elif ev_high["roi"] > 1.0:
                lines.append("   - ⚠️ 高EV馬はギリギリ黒字だが、マージンが薄い")
            else:
                lines.append("   - ❌ 高EV馬が赤字。EVは信頼できる買い指標ではない")

    lines.append(f"\n4. **p_aiとp_marketの相関 = {corr:.4f}**")
    if corr > 0.95:
        lines.append("   - ❌ 相関が極めて高い。AIは市場を「再現」しているが、「出し抜く」差分が小さい")
    elif corr > 0.90:
        lines.append("   - ⚠️ 相関が高い。エッジは存在するが薄い")
    else:
        lines.append("   - ✅ 相関が中程度。AIは市場と異なる確率を出している")

    for surface in ["turf", "dirt"]:
        surf_c = results["correlation"].get(surface, 0)
        if surf_c:
            lines.append(f"   - {surface}: {surf_c:.4f}")

    # ── 5. 次の推奨アクション ──
    lines.append("\n---\n")
    lines.append("## 5. 次の推奨アクション")
    lines.append("")

    if verdict == "B":
        lines.append("1. **Win Return モデルを外した simple win EV 版の作成**")
        lines.append("   - `EV = p_win × tanodds` のみで評価")
        lines.append("   - Returnモデルによる選択バイアス（1着馬のみ学習）の影響を確認")
        lines.append("")
        lines.append("2. **補正前 p_win_pred で EV 評価する**")
        lines.append("   - EVCorrection / Isotonic / OddsBand 補正がエッジを消していないか確認")
        lines.append("")
        lines.append("3. **オッズ帯別買い閾値の導入**")
        lines.append("   - オッズ帯ごとに最適な EV閾値 を設定")
        lines.append("")
        lines.append("4. **edge_ratio 上位帯だけに絞る戦略の検討**")
        lines.append("   - 全条件で買わず、AIが市場より高く評価した馬だけに集中")
        lines.append("")
    elif verdict == "C":
        lines.append("1. **オッズ帯別の買い閾値を最適化する**")
        lines.append("   - 全帯で共通の EV>1.1 ではなく、帯ごとに最適な閾値を設定")
        lines.append("")
        lines.append("2. **Surface 別の分析を深める**")
        lines.append("   - Turf と Dirt でエッジの分布が異なる可能性")
        lines.append("")
        lines.append("3. **Win Return 外しアブレーション**")
        lines.append("   - 影響の程度を定量的に確認")
        lines.append("")
    elif verdict == "A":
        lines.append("1. **現行構造をベースに微調整**")
        lines.append("   - EVは機能しているので、閾値や資金配分の最適化に注力")
        lines.append("")

    lines.append("5. **Turf MAWC interaction の見直し**")
    lines.append("   - Turfの確率品質が悪い場合、MarketAwareWinCalibrator のinteraction項を減らす")

    return "\n".join(lines)


def main() -> None:
    """エントリポイント。"""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    results, df = run_analysis()

    # JSON 出力
    json_path = ANALYSIS_DIR / "win_market_edge_diagnostic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON saved: {json_path}")

    # Markdown 出力
    md_text = generate_markdown(results)
    md_path = ANALYSIS_DIR / "win_market_edge_diagnostic.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Markdown saved: {md_path}")

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
