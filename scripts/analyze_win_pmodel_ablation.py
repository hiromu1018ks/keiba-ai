"""
単勝Pモデル アブレーション診断スクリプト
=========================================
補正パイプラインの各段階 (p_win_pred → p_win_corrected → p_win_final)
と市場ベースライン (p_market) を横断比較し、エッジ消失の原因を特定する。

候補:
  B: p_win_pred          (補正前P: 純粋ファンダメンタル)
  C: p_win_corrected     (P補正後, MAWC前)
  A: p_win_final         (MAWC後: 現行本番)
  D: p_market_win_norm   (市場ベースライン: 参考指標)

ROI定義:
  各グループ内の全馬を単勝100円で購入した場合の仮想回収率。

出力:
  - data/analysis/win_pmodel_ablation_diagnostic.json
  - data/analysis/win_pmodel_ablation_diagnostic.md
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
BACKTEST_DIR = DATA_DIR / "backtest"

# ── ROI 定義 ─────────────────────────────────────────────────────
# 各グループ内の全馬を単勝100円で購入した場合の仮想回収率 (payout / stake)
ROI_NOTE = (
    "ROIは各グループ内の全馬を単勝100円で購入した場合の仮想回収率とする。"
    "市場ベースラインDも pred_ev = p_market_win_norm × tanodds を計算するが、"
    "AI候補とは別の参考指標として扱う。"
)

# ── 候補定義 ─────────────────────────────────────────────────────

CANDIDATES: list[dict[str, str]] = [
    {"id": "B", "col": "p_win_pred", "label": "補正前P (p_win_pred)",
     "desc": "純粋ファンダメンタル確率。WinTwoStageModel の直接出力。"},
    {"id": "C", "col": "p_win_corrected", "label": "P補正後 (p_win_corrected)",
     "desc": "EV補正P-correction後。MAWC適用前。"},
    {"id": "A", "col": "p_win_final", "label": "現行本番 (p_win_final)",
     "desc": "MAWC適用後。現行の最終確率。"},
    {"id": "D", "col": "p_market_win_norm", "label": "市場BL (p_market_win_norm)",
     "desc": "市場正規化確率。AIなしベースライン。参考指標。"},
]

# 分位・帯の定義
QUANTILE_CUTS = [0.0, 0.20, 0.40, 0.60, 0.80, 0.95, 0.99, 1.0]
QUANTILE_LABELS = [
    "下位 (0-20%)", "中下位 (20-40%)", "中央 (40-60%)",
    "中上位 (60-80%)", "上位 (80-95%)", "最上位 (95-99%)", "極上位 (99-100%)",
]

EV_BANDS: list[tuple[float, float | None]] = [
    (0.0, 0.90), (0.90, 1.00), (1.00, 1.10), (1.10, 1.20),
    (1.20, 1.30), (1.30, 1.50), (1.50, None),
]

ODDS_BANDS: list[tuple[float, float | None]] = [
    (1.0, 3.0), (3.0, 5.0), (5.0, 10.0),
    (10.0, 20.0), (20.0, 50.0), (50.0, None),
]

EV_TOPN_PCTS = [1, 5, 10]
EV_TOPN_THRESHOLDS = [1.1, 1.2, 1.3]


# ── ユーティリティ ──────────────────────────────────────────────

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def calc_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def calc_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def calc_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
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
) -> dict[str, Any]:
    """ROI・的中率などの基本統計。各馬100円買いの仮想ROI。"""
    n = len(df)
    if n == 0:
        return {"count": 0, "hit_rate": 0.0, "avg_odds": 0.0,
                "roi": 0.0, "profit": 0.0}
    hits = df[win_col].sum()
    total_payout = float((df[odds_col] * df[win_col]).sum())
    roi = safe_div(total_payout, float(n))
    return {
        "count": int(n),
        "hit_rate": round(float(hits / n), 4),
        "avg_odds": round(float(df[odds_col].mean()), 2),
        "roi": round(roi, 4),
        "profit": round(total_payout - float(n), 1),
    }


# ── データ読み込み ──────────────────────────────────────────────

def load_horse_features() -> pd.DataFrame:
    """bt_{year}_horse_features.parquet を読み込んで結合。"""
    need_cols = [
        "race_id", "umaban", "kakuteijyuni", "surface", "tanodds",
        "p_win_pred", "p_win_corrected", "p_win_final",
        "p_market_win_norm",
    ]
    frames: list[pd.DataFrame] = []
    for year in [2024, 2025]:
        path = BACKTEST_DIR / f"bt_{year}_horse_features.parquet"
        if not path.exists():
            print(f"  [skip] {year}: {path} not found")
            continue
        # 必要列だけ読み込み
        available = pd.read_parquet(path).columns.tolist()
        cols = [c for c in need_cols if c in available]
        df = pd.read_parquet(path, columns=cols)
        df["year"] = year
        df["race_id"] = df["race_id"].astype(str)
        frames.append(df)
        print(f"  {year}: {len(df)} rows, {len(cols)} cols")

    if not frames:
        raise FileNotFoundError("No backtest horse_features files found")

    df = pd.concat(frames, ignore_index=True)
    # 前処理
    df = df[df["tanodds"] > 0].copy()
    df["is_win"] = (df["kakuteijyuni"] == 1).astype(int)
    # p_market フォールバック: p_market_win_norm がなければ 1/tanodds から計算
    if "p_market_win_norm" not in df.columns or df["p_market_win_norm"].isna().all():
        p_raw = 1.0 / df["tanodds"]
        p_sum = p_raw.groupby(df["race_id"]).transform("sum")
        df["p_market_win_norm"] = (p_raw / p_sum).astype(float)

    # 無効行除外
    df = df.dropna(subset=["p_win_pred", "p_win_corrected", "p_win_final",
                            "p_market_win_norm", "tanodds", "kakuteijyuni"])
    print(f"  Total valid: {len(df)} rows")
    return df


# ── 各候補の分析 ──────────────────────────────────────────────

def compute_correlation(
    df: pd.DataFrame,
    p_col: str,
    p_market_col: str = "p_market_win_norm",
) -> dict[str, Any]:
    """p_candidate と p_market の相関を計算。"""
    corr_data: dict[str, Any] = {}
    corr_data["overall"] = round(float(df[p_col].corr(df[p_market_col])), 4)

    for surface in ["turf", "dirt"]:
        sub = df[df["surface"] == surface]
        if len(sub) > 10:
            corr_data[surface] = round(float(sub[p_col].corr(sub[p_market_col])), 4)

    corr_data["by_odds_band"] = {}
    for lo, hi in ODDS_BANDS:
        if hi is None:
            sub = df[df["tanodds"] >= lo]
            label = f"{lo:.0f}-{hi}倍+" if hi else f"{lo:.0f}倍+"
        else:
            sub = df[(df["tanodds"] >= lo) & (df["tanodds"] < hi)]
            label = f"{lo:.0f}-{hi:.0f}倍"
        if len(sub) > 10:
            corr_data["by_odds_band"][label] = round(
                float(sub[p_col].corr(sub[p_market_col])), 4
            )

    return corr_data


def compute_edge_ratio_quantile_roi(
    df: pd.DataFrame,
    p_col: str,
    p_market_col: str = "p_market_win_norm",
) -> list[dict[str, Any]]:
    """edge_ratio = p_candidate / p_market を分位で区切ってROI計算。

    候補D (p_market同士) の場合、edge_ratio = 1.0 で全行同じ値になり、
    分位分析が無意味なので空リストを返す。
    """
    edge_ratio = (df[p_col] / df[p_market_col]).clip(0.01, 100.0)

    # 全行が同じ値（=候補D）の場合はスキップ
    if edge_ratio.nunique() <= 1:
        return [{"label": "N/A (市場同士のためedge=1.0)",
                 "edge_range": "1.000 - 1.000",
                 "count": len(df), "hit_rate": round(float(df["is_win"].mean()), 4),
                 "avg_odds": round(float(df["tanodds"].mean()), 2),
                 "roi": round(float((df["tanodds"] * df["is_win"]).sum() / len(df)), 4),
                 "profit": round(float((df["tanodds"] * df["is_win"]).sum() - len(df)), 1),
                 "avg_p_candidate": round(float(df[p_col].mean()), 4),
                 "avg_p_market": round(float(df[p_market_col].mean()), 4)}]

    q_edges = edge_ratio.quantile(QUANTILE_CUTS).tolist()

    results: list[dict[str, Any]] = []
    for i in range(len(q_edges) - 1):
        lo, hi = q_edges[i], q_edges[i + 1]
        sub = df[(edge_ratio >= lo) & (edge_ratio < hi)]
        base = roi_summary(sub)
        base["label"] = QUANTILE_LABELS[i]
        base["edge_range"] = f"{lo:.3f} - {hi:.3f}"
        if len(sub) > 0:
            base["avg_p_candidate"] = round(float(sub[p_col].mean()), 4)
            base["avg_p_market"] = round(float(sub[p_market_col].mean()), 4)
        results.append(base)
    return results


def compute_pred_ev_band_roi(
    df: pd.DataFrame,
    p_col: str,
    odds_col: str = "tanodds",
    win_col: str = "is_win",
) -> list[dict[str, Any]]:
    """pred_ev = p_candidate × tanodds 帯別ROI。"""
    pred_ev = df[p_col] * df[odds_col]
    results: list[dict[str, Any]] = []
    for lo, hi in EV_BANDS:
        if hi is None:
            sub = df[pred_ev >= lo]
            label = f"EV {lo:.2f}+"
        else:
            sub = df[(pred_ev >= lo) & (pred_ev < hi)]
            label = f"EV {lo:.2f}-{hi:.2f}"
        base = roi_summary(sub, odds_col, win_col)
        base["label"] = label
        if len(sub) > 0:
            base["avg_pred_ev"] = round(float(pred_ev[sub.index].mean()), 4)
        results.append(base)
    return results


def compute_odds_band_roi(
    df: pd.DataFrame,
    p_col: str,
    p_market_col: str = "p_market_win_norm",
    odds_col: str = "tanodds",
) -> list[dict[str, Any]]:
    """オッズ帯別ROI。"""
    results: list[dict[str, Any]] = []
    for lo, hi in ODDS_BANDS:
        if hi is None:
            sub = df[df[odds_col] >= lo]
            label = f"{lo:.1f}倍+"
        else:
            sub = df[(df[odds_col] >= lo) & (df[odds_col] < hi)]
            label = f"{lo:.1f}-{hi:.1f}倍"
        base = roi_summary(sub)
        base["label"] = label
        if len(sub) > 0:
            pred_ev = sub[p_col] * sub[odds_col]
            base["avg_p_candidate"] = round(float(sub[p_col].mean()), 4)
            base["avg_p_market"] = round(float(sub[p_market_col].mean()), 4)
            base["avg_pred_ev"] = round(float(pred_ev.mean()), 4)
        results.append(base)
    return results


def compute_ev_topn_roi(
    df: pd.DataFrame,
    p_col: str,
    odds_col: str = "tanodds",
) -> list[dict[str, Any]]:
    """EV上位N% と EV>閾値 のROI。"""
    pred_ev = df[p_col] * df[odds_col]
    results: list[dict[str, Any]] = []

    # EV上位 N%
    for pct in EV_TOPN_PCTS:
        threshold = pred_ev.quantile(1.0 - pct / 100.0)
        sub = df[pred_ev >= threshold]
        base = roi_summary(sub)
        base["label"] = f"EV上位{pct}%"
        base["ev_threshold"] = round(float(threshold), 4)
        results.append(base)

    # EV > X
    for ev_thresh in EV_TOPN_THRESHOLDS:
        sub = df[pred_ev > ev_thresh]
        base = roi_summary(sub)
        base["label"] = f"EV>{ev_thresh:.1f}"
        base["ev_threshold"] = ev_thresh
        results.append(base)

    return results


def compute_calibration(
    df: pd.DataFrame,
    p_col: str,
    win_col: str = "is_win",
) -> dict[str, Any]:
    """確率品質 (Logloss, Brier, ECE, APR)。"""
    y_true = df[win_col].values.astype(float)
    y_pred = df[p_col].values.astype(float)
    valid = ~(np.isnan(y_pred) | np.isnan(y_true))
    if valid.sum() < 100:
        return {"logloss": None, "brier": None, "ece": None, "apr": None,
                "count": int(valid.sum())}
    y_pred_v = y_pred[valid]
    y_true_v = y_true[valid]
    apr = safe_div(float(y_true_v.sum()), float(y_pred_v.sum()))
    return {
        "logloss": round(calc_logloss(y_true_v, y_pred_v), 4),
        "brier": round(calc_brier(y_true_v, y_pred_v), 4),
        "ece": round(calc_ece(y_true_v, y_pred_v), 4),
        "apr": round(apr, 4),
        "count": int(valid.sum()),
    }


def check_monotonicity(
    edge_quantile_roi: list[dict[str, Any]],
) -> dict[str, Any]:
    """edge_ratio分位ROIの単調性をチェック。

    下位→上位に向かってROIが単調改善しているか。
    最上位(99-100%)が直前(95-99%)より大きく崩落している場合、
    edge指標が不安定とみなす。
    """
    rois = [r["roi"] for r in edge_quantile_roi]
    n = len(rois)

    # 概ね単調増加かチェック（隣接ペアで低下が2回以下なら「概ね単調」）
    decreases = sum(1 for i in range(1, n) if rois[i] < rois[i - 1])
    is_mostly_monotone = decreases <= 2

    # 最上位崩落チェック (99-100% < 95-99%)
    top_collapse = False
    collapse_pct = 0.0
    if n >= 2:
        diff = rois[-1] - rois[-2]
        collapse_pct = round(diff, 4)
        if rois[-1] < rois[-2]:
            top_collapse = True

    return {
        "is_mostly_monotone": is_mostly_monotone,
        "decrease_count": decreases,
        "top_collapse": top_collapse,
        "top_collapse_pct": collapse_pct,
        "roi_sequence": rois,
    }


def analyze_single_candidate(
    df: pd.DataFrame,
    cand: dict[str, str],
    p_market_col: str = "p_market_win_norm",
) -> dict[str, Any]:
    """1候補について全指標を計算。"""
    p_col = cand["col"]
    cand_id = cand["id"]

    result: dict[str, Any] = {
        "id": cand_id,
        "col": p_col,
        "label": cand["label"],
        "desc": cand["desc"],
    }

    print(f"\n  === 候補{cand_id}: {cand['label']} ===")

    # 1) 相関
    result["correlation"] = compute_correlation(df, p_col, p_market_col)
    print(f"    相関: overall={result['correlation']['overall']}")

    # 2) edge_ratio 分位別ROI
    result["edge_ratio_quantile_roi"] = compute_edge_ratio_quantile_roi(
        df, p_col, p_market_col
    )
    result["monotonicity"] = check_monotonicity(result["edge_ratio_quantile_roi"])
    for r in result["edge_ratio_quantile_roi"]:
        print(f"    {r['label']}: n={r['count']}, ROI={r['roi']}")

    # 3) pred_ev 帯別ROI
    result["pred_ev_band_roi"] = compute_pred_ev_band_roi(df, p_col)
    for r in result["pred_ev_band_roi"]:
        print(f"    {r['label']}: n={r['count']}, ROI={r['roi']}")

    # 4) オッズ帯別ROI
    result["odds_band_roi"] = compute_odds_band_roi(df, p_col, p_market_col)

    # 5) EV上位ROI
    result["ev_topn_roi"] = compute_ev_topn_roi(df, p_col)
    for r in result["ev_topn_roi"]:
        ref = " ※参考値" if r["count"] < 30 else ""
        print(f"    {r['label']}: n={r['count']}, ROI={r['roi']}{ref}")

    # 6) 確率品質 (A/B/Cのみ。Dは市場確率なので品質比較に含めない)
    if cand_id in ("A", "B", "C"):
        result["calibration"] = compute_calibration(df, p_col)
        cal = result["calibration"]
        print(f"    Logloss={cal['logloss']}, Brier={cal['brier']}, "
              f"ECE={cal['ece']}, APR={cal['apr']}")

    return result


# ── Surface別サマリ ──────────────────────────────────────────────

def compute_surface_summary(
    df: pd.DataFrame,
    candidates: list[dict[str, str]],
) -> dict[str, Any]:
    """turf/dirt別に各候補のedge上位5% ROIとEV>1.2 ROIを計算。"""
    summary: dict[str, Any] = {}
    for surface in ["turf", "dirt"]:
        sub = df[df["surface"] == surface]
        if len(sub) == 0:
            continue
        surf_data: dict[str, Any] = {"total_rows": len(sub)}
        surf_data["total_roi"] = roi_summary(sub)

        cand_results: dict[str, Any] = {}
        for cand in candidates:
            p_col = cand["col"]
            cand_id = cand["id"]

            # edge上位5% ROI
            edge_ratio = (sub[p_col] / sub["p_market_win_norm"]).clip(0.01, 100.0)
            top5_threshold = edge_ratio.quantile(0.95)
            top5 = sub[edge_ratio >= top5_threshold]
            top5_stats = roi_summary(top5)

            # EV>1.2 ROI
            pred_ev = sub[p_col] * sub["tanodds"]
            ev_high = sub[pred_ev > 1.2]
            ev_high_stats = roi_summary(ev_high)

            cand_results[cand_id] = {
                "edge_top5_roi": top5_stats,
                "ev_gt_1_2_roi": ev_high_stats,
            }

        surf_data["candidates"] = cand_results
        summary[surface] = surf_data

    return summary


# ── 原因判定 ──────────────────────────────────────────────────────

def determine_verdict(
    candidates_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """原因判定ロジック。

    判定には2軸を組み合わせる:
    - edge上位ROIの絶対値 (95-99%分位 + 99-100%分位のROI)
    - edge_ratio分位の単調性 (下位→上位に向かってROIが単調改善しているか)
    """
    # 候補別データに素早くアクセス
    by_id: dict[str, dict[str, Any]] = {c["id"]: c for c in candidates_data}

    def get_edge_top_roi(cand_id: str) -> float:
        """95-99%分位 (index 5) のROIを返す。"""
        eq = by_id[cand_id].get("edge_ratio_quantile_roi", [])
        return eq[5]["roi"] if len(eq) >= 6 else 0.0

    def get_edge_top1_roi(cand_id: str) -> float:
        """99-100%分位 (index 6) のROIを返す。"""
        eq = by_id[cand_id].get("edge_ratio_quantile_roi", [])
        return eq[6]["roi"] if len(eq) >= 7 else 0.0

    def get_ev_top5_roi(cand_id: str) -> float:
        """EV上位5%のROIを返す。"""
        for r in by_id[cand_id].get("ev_topn_roi", []):
            if "上位5%" in r["label"]:
                return r["roi"]
        return 0.0

    def get_ev_gt12_roi(cand_id: str) -> float:
        """EV>1.2のROIを返す。"""
        for r in by_id[cand_id].get("ev_topn_roi", []):
            if "EV>1.2" in r["label"]:
                return r["roi"]
        return 0.0

    def get_correlation(cand_id: str) -> float:
        return by_id[cand_id].get("correlation", {}).get("overall", 0.0)

    def is_monotone(cand_id: str) -> bool:
        return by_id[cand_id].get("monotonicity", {}).get("is_mostly_monotone", False)

    def has_top_collapse(cand_id: str) -> bool:
        return by_id[cand_id].get("monotonicity", {}).get("top_collapse", False)

    # ── 判定ロジック ──
    verdict_code = "MIXED"
    verdict_reason = ""
    verdict_detail: dict[str, Any] = {}

    b_corr = get_correlation("B")
    b_edge_top = get_edge_top_roi("B")
    b_monotone = is_monotone("B")
    b_top_collapse = has_top_collapse("B")

    c_edge_top = get_edge_top_roi("C")
    a_edge_top = get_edge_top_roi("A")

    b_ev_gt12 = get_ev_gt12_roi("B")
    c_ev_gt12 = get_ev_gt12_roi("C")
    a_ev_gt12 = get_ev_gt12_roi("A")

    verdict_detail = {
        "B_correlation": b_corr,
        "B_edge_95_99_roi": b_edge_top,
        "B_edge_99_100_roi": get_edge_top1_roi("B"),
        "B_monotone": b_monotone,
        "B_top_collapse": b_top_collapse,
        "C_edge_95_99_roi": c_edge_top,
        "A_edge_95_99_roi": a_edge_top,
        "B_ev_gt_1_2_roi": b_ev_gt12,
        "C_ev_gt_1_2_roi": c_ev_gt12,
        "A_ev_gt_1_2_roi": a_ev_gt12,
    }

    # 判定2: P_MODEL_WEAK
    if b_corr > 0.95 and (b_edge_top < 0.85 or not b_monotone):
        verdict_code = "P_MODEL_WEAK"
        verdict_reason = (
            f"候補B (p_win_pred) の市場相関が {b_corr:.3f} と極めて高く、"
            f"edge上位95-99%分位ROIが {b_edge_top:.2%}。"
        )
        if not b_monotone:
            verdict_reason += "edge分位ROIが単調改善していない。"
        if b_top_collapse:
            verdict_reason += "最上位(99-100%)が崩落している。"
        verdict_reason += "→ Pモデル自体が市場追随しており、エッジが存在しない。"

    # 判定3: CORRECTION_KILLS_EDGE
    elif b_edge_top >= 0.85 and c_edge_top < b_edge_top - 0.05:
        verdict_code = "CORRECTION_KILLS_EDGE"
        verdict_reason = (
            f"候補Bのedge上位ROI {b_edge_top:.2%} に対して "
            f"候補C (p_win_corrected) は {c_edge_top:.2%} に低下（5pt以上）。"
            "→ EV補正がエッジを消している。"
        )

    # 判定4: MAWC_KILLS_EDGE
    elif c_edge_top >= 0.85 and a_edge_top < c_edge_top - 0.05:
        verdict_code = "MAWC_KILLS_EDGE"
        verdict_reason = (
            f"候補Cのedge上位ROI {c_edge_top:.2%} に対して "
            f"候補A (p_win_final) は {a_edge_top:.2%} に低下（5pt以上）。"
            "→ MAWCがエッジを消している。"
        )

    # 判定5: EV_BROKEN
    elif (b_ev_gt12 < 0.85 and c_ev_gt12 < 0.85 and a_ev_gt12 < 0.85) or (
        has_top_collapse("B") and has_top_collapse("A")
    ):
        verdict_code = "EV_BROKEN"
        verdict_reason = (
            f"全候補でEV>1.2のROIが低い "
            f"(B={b_ev_gt12:.2%}, C={c_ev_gt12:.2%}, A={a_ev_gt12:.2%})。"
            "またはedge最上位が大幅崩落。"
            "→ EV指標自体が選択指標として機能していない。"
        )

    else:
        verdict_code = "MIXED"
        verdict_reason = (
            "候補・条件別に有効/無効が混在しており、単一の原因に絞り込めない。"
        )

    return {
        "code": verdict_code,
        "reason": verdict_reason,
        "detail": verdict_detail,
    }


# ── 比較サマリ表 ──────────────────────────────────────────────

def build_comparison_table(
    candidates_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """全候補の主要指標を1行にまとめた比較表を生成。"""
    rows: list[dict[str, Any]] = []
    for c in candidates_data:
        corr = c.get("correlation", {}).get("overall", 0.0)
        eq = c.get("edge_ratio_quantile_roi", [])
        edge_top = eq[5]["roi"] if len(eq) >= 6 else 0.0
        edge_top1 = eq[6]["roi"] if len(eq) >= 7 else 0.0
        ev_roi = c.get("ev_topn_roi", [])
        ev_gt12 = next((r["roi"] for r in ev_roi if "EV>1.2" in r.get("label", "")), 0.0)
        ev_gt12_n = next((r["count"] for r in ev_roi if "EV>1.2" in r.get("label", "")), 0)
        ev_top5 = next((r["roi"] for r in ev_roi if "上位5%" in r.get("label", "")), 0.0)
        mono = c.get("monotonicity", {})
        rows.append({
            "candidate": c["id"],
            "label": c["label"],
            "correlation": corr,
            "edge_95_99_roi": edge_top,
            "edge_99_100_roi": edge_top1,
            "ev_top5_roi": ev_top5,
            "ev_gt_1_2_roi": ev_gt12,
            "ev_gt_1_2_n": ev_gt12_n,
            "monotone": mono.get("is_mostly_monotone", False),
            "top_collapse": mono.get("top_collapse", False),
        })
    return rows


# ── Markdown レポート ──────────────────────────────────────────────

def generate_ablation_markdown(
    results: dict[str, Any],
) -> str:
    lines: list[str] = []

    meta = results["meta"]
    verdict = results["verdict"]
    comp = results["comparison_table"]
    cands = list(results["candidates"].values())

    # ── ヘッダ ──
    lines.append("# 単勝Pモデル アブレーション診断レポート")
    lines.append(f"\n生成日時: {meta['generated_at']}")
    lines.append(f"分析対象: {meta['years']}年 / {meta['total_rows']}レコード")
    lines.append(f"馬場: {', '.join(f'{k}={v}' for k, v in meta['surfaces'].items())}")

    # ── 1. 結論 ──
    lines.append("\n---\n")
    lines.append("## 1. 結論")
    code = verdict["code"]
    code_labels = {
        "P_MODEL_WEAK": "Pモデル自体が弱い",
        "CORRECTION_KILLS_EDGE": "補正がエッジを消している",
        "MAWC_KILLS_EDGE": "MAWCがエッジを消している",
        "EV_BROKEN": "EV指標が機能していない",
        "MIXED": "原因が混在",
    }
    lines.append(f"\n**判定: {code} — {code_labels.get(code, code)}**\n")
    lines.append(verdict["reason"])

    # ── 2. ROI定義 ──
    lines.append("\n---\n")
    lines.append("## 2. 分析方法")
    lines.append(f"\n> {ROI_NOTE}")

    lines.append("\n### 比較候補")
    lines.append("")
    lines.append("| ID | 確率列 | 説明 |")
    lines.append("|----|--------|------|")
    for c in CANDIDATES:
        lines.append(f"| {c['id']} | `{c['col']}` | {c['desc']} |")

    # ── 3. 比較サマリ表 ──
    lines.append("\n---\n")
    lines.append("## 3. 比較サマリ表")
    lines.append("")
    lines.append("| 候補 | 相関 | edge上位5% ROI | edge最上位ROI | EV上位5% ROI | EV>1.2 ROI (n) | 単調性 | 最上位崩落 |")
    lines.append("|------|------|---------------|-------------|------------|---------------|--------|-----------|")
    for row in comp:
        mono_str = "✅" if row["monotone"] else "❌"
        collapse_str = "❌崩落" if row["top_collapse"] else "OK"
        lines.append(
            f"| {row['candidate']} ({row['label'].split(' ')[0]}) "
            f"| {row['correlation']:.4f} "
            f"| {row['edge_95_99_roi']:.2%} "
            f"| {row['edge_99_100_roi']:.2%} "
            f"| {row['ev_top5_roi']:.2%} "
            f"| {row['ev_gt_1_2_roi']:.2%} (n={row['ev_gt_1_2_n']:,}) "
            f"| {mono_str} "
            f"| {collapse_str} |"
        )

    # ── 4. edge_ratio 分位別ROI ──
    lines.append("\n---\n")
    lines.append("## 4. edge_ratio 分位別ROI")
    lines.append("> edge_ratio = p_candidate / p_market。1.0より大きい = AIが市場より高く評価")
    lines.append("")

    # ヘッダー
    header = "| 分位 |"
    sep = "|------|"
    for c in cands:
        header += f" {c['id']} ROI (n) |"
        sep += "---------------|"
    lines.append(header)
    lines.append(sep)

    # 各分位行
    n_quantiles = 7
    for qi in range(n_quantiles):
        row_label = QUANTILE_LABELS[qi]
        row = f"| {row_label} |"
        for c in cands:
            eq = c.get("edge_ratio_quantile_roi", [])
            if qi < len(eq):
                r = eq[qi]
                ref = " ※" if r["count"] < 30 else ""
                row += f" {r['roi']:.2%} ({r['count']:,}){ref} |"
            else:
                row += " — |"
        lines.append(row)

    # 単調性チェック結果
    lines.append("\n### 単調性チェック結果")
    lines.append("")
    for c in cands:
        mono = c.get("monotonicity", {})
        status = "✅ 概ね単調" if mono.get("is_mostly_monotone") else "❌ 非単調"
        collapse = "❌ 最上位崩落あり" if mono.get("top_collapse") else ""
        lines.append(f"- 候補{c['id']}: {status} (低下回数: {mono.get('decrease_count', '?')}) {collapse}")

    # ── 5. pred_ev 帯別ROI ──
    lines.append("\n---\n")
    lines.append("## 5. pred_ev 帯別ROI")
    lines.append("> pred_ev = p_candidate × tanodds。1.0より大きい = プラス期待値")
    lines.append("")

    header = "| EV帯 |"
    sep = "|------|"
    for c in cands:
        header += f" {c['id']} ROI (n) |"
        sep += "---------------|"
    lines.append(header)
    lines.append(sep)

    for bi, (lo, hi) in enumerate(EV_BANDS):
        label = f"EV {lo:.2f}+" if hi is None else f"EV {lo:.2f}-{hi:.2f}"
        row = f"| {label} |"
        for c in cands:
            evb = c.get("pred_ev_band_roi", [])
            if bi < len(evb):
                r = evb[bi]
                ref = " ※" if r["count"] < 30 else ""
                row += f" {r['roi']:.2%} ({r['count']:,}){ref} |"
            else:
                row += " — |"
        lines.append(row)

    # ── 6. オッズ帯別ROI ──
    lines.append("\n---\n")
    lines.append("## 6. オッズ帯別ROI")
    lines.append("")

    header = "| オッズ帯 |"
    sep = "|---------|"
    for c in cands:
        header += f" {c['id']} ROI (n) |"
        sep += "---------------|"
    lines.append(header)
    lines.append(sep)

    for oi, (lo, hi) in enumerate(ODDS_BANDS):
        label = f"{lo:.1f}倍+" if hi is None else f"{lo:.1f}-{hi:.1f}倍"
        row = f"| {label} |"
        for c in cands:
            ob = c.get("odds_band_roi", [])
            if oi < len(ob):
                r = ob[oi]
                ref = " ※" if r["count"] < 30 else ""
                row += f" {r['roi']:.2%} ({r['count']:,}){ref} |"
            else:
                row += " — |"
        lines.append(row)

    # ── 7. EV上位ROI ──
    lines.append("\n---\n")
    lines.append("## 7. EV上位ROI")
    lines.append("")

    header = "| 条件 |"
    sep = "|------|"
    for c in cands:
        header += f" {c['id']} ROI (n) |"
        sep += "---------------|"
    lines.append(header)
    lines.append(sep)

    all_labels: list[str] = []
    for pct in EV_TOPN_PCTS:
        all_labels.append(f"EV上位{pct}%")
    for ev_t in EV_TOPN_THRESHOLDS:
        all_labels.append(f"EV>{ev_t:.1f}")

    for li, target_label in enumerate(all_labels):
        row = f"| {target_label} |"
        for c in cands:
            evn = c.get("ev_topn_roi", [])
            match = next((r for r in evn if r["label"] == target_label), None)
            if match:
                ref = " ※参考値" if match["count"] < 30 else ""
                row += f" {match['roi']:.2%} ({match['count']:,}){ref} |"
            else:
                row += " — |"
        lines.append(row)

    # ── 8. Surface別サマリ ──
    lines.append("\n---\n")
    lines.append("## 8. Surface別サマリ")

    surf_summary = results.get("surface_summary", {})
    for surface, surf_data in surf_summary.items():
        lines.append(f"\n### {surface.upper()}")
        total = surf_data.get("total_roi", {})
        lines.append(f"- 件数: {total.get('count', 0):,}, ROI: **{total.get('roi', 0):.2%}**")
        lines.append("")
        lines.append("| 候補 | edge上位5% ROI (n) | EV>1.2 ROI (n) |")
        lines.append("|------|-------------------|----------------|")
        for cand_id, cand_results in surf_data.get("candidates", {}).items():
            e5 = cand_results.get("edge_top5_roi", {})
            ev12 = cand_results.get("ev_gt_1_2_roi", {})
            lines.append(
                f"| {cand_id} "
                f"| {e5.get('roi', 0):.2%} ({e5.get('count', 0):,}) "
                f"| {ev12.get('roi', 0):.2%} ({ev12.get('count', 0):,}) |"
            )

    # ── 9. 確率品質 ──
    lines.append("\n---\n")
    lines.append("## 9. 確率品質比較 (候補B/C/A)")
    lines.append("")
    lines.append("| 段階 | Logloss | Brier | ECE | APR | 件数 |")
    lines.append("|------|---------|-------|-----|-----|------|")
    for c in cands:
        cal = c.get("calibration")
        if cal and cal.get("logloss") is not None:
            lines.append(
                f"| 候補{c['id']} ({c['label'].split(' ')[0]}) "
                f"| {cal['logloss']:.4f} "
                f"| {cal['brier']:.4f} "
                f"| {cal['ece']:.4f} "
                f"| {cal['apr']:.4f} "
                f"| {cal['count']:,} |"
            )

    # ── 10. 解釈 ──
    lines.append("\n---\n")
    lines.append("## 10. 解釈（初心者向け）")

    corr_b = verdict["detail"].get("B_correlation", 0)
    lines.append(f"\n- **相関{corr_b:.3f}** = AIの予測確率と市場オッズがほぼ同じ方向を向いている")
    lines.append("  - 1.0に近いほど「AIが市場を真似している」ことを意味する")
    lines.append("  - 0.8以下なら「AIが市場と異なる予測を出している」= エッジが大きい")

    lines.append("\n- **edge上位ROIが低い** = AIが「市場より強い」と言った馬が実際には勝てていない")
    lines.append("  - 100%を超えれば「AIの評価が市場より正確」")
    lines.append("  - 80%未満なら「AIの評価は市場より悪い」")

    lines.append("\n- **EV上位の件数が少ない** = 条件を満たす馬が少なく、統計的に信頼できない")
    lines.append("  - n<30の場合は「参考値」として扱い、結論の根拠にしない")

    lines.append("\n- **最上位崩落** = edge_ratioが最も高い馬(上位1%)のROIが、上位5%より悪い")
    lines.append("  - AIが最も自信を持った馬が負けている = 過信の兆候")

    # 候補別の読み方
    lines.append("\n### 候補の読み方")
    lines.append("- **候補B (p_win_pred)** が悪い → Pモデルそのものの問題")
    lines.append("- **候補B→Cで悪化** → EV補正がエッジを消している")
    lines.append("- **候補C→Aで悪化** → MAWC(市場ブレンド)がエッジを消している")
    lines.append("- **候補D (市場BL)** よりAI候補が悪い → AIが市場より劣っている")

    # ── 11. 次の段階の提案 ──
    lines.append("\n---\n")
    lines.append("## 11. 次の段階の提案")

    next_steps = {
        "P_MODEL_WEAK": (
            "### 推奨: オッズ系特徴量を抜いたPモデルの小規模OOF分析設計\n\n"
            "Pモデル自体が市場追随している場合、補正の調整では根本解決しない。\n"
            "WinTwoStageModelの特徴量からオッズ系（log_error, odds_dynamics 等）を\n"
            "段階的に除外し、OOF予測の市場相関がどこまで下がるかを確認する実験を設計する。\n\n"
            "理由: 既存の特徴量の多くが市場情報を間接的に取り込んでおり、\n"
            "これがPモデルを「賢い市場コピー」にしている可能性が高い。"
        ),
        "CORRECTION_KILLS_EDGE": (
            "### 推奨: 補正なしPを使った買い判定の軽量検証\n\n"
            "p_win_corrected が p_win_pred よりROIを下げている場合、\n"
            "EV補正のP-correctionをスキップする構成でバックテストを実行し、\n"
            "ROIが改善するかを確認する。"
        ),
        "MAWC_KILLS_EDGE": (
            "### 推奨: MAWCをスキップした p_win_corrected 直接評価\n\n"
            "MAWC(市場ブレンド)がエッジを消している場合、\n"
            "p_win_corrected × tanodds を直接EVとして評価する軽量検証を行う。\n"
            "MAWCのBETA_MARKET_FLOOR (20%) を下げるか、セグメント条件を見直す。"
        ),
        "EV_BROKEN": (
            "### 推奨: Win Returnなし simple EV 版の実装設計\n\n"
            "全段階でEVが選択指標として機能していない場合、\n"
            "EV = p_win × tanodds の単純構成で評価する設計を作る。\n"
            "WinReturnモデル（1着馬のみ学習）がEV計算に悪影響を与えている可能性。"
        ),
        "MIXED": (
            "### 推奨: 追加データ列の出力を先に整備\n\n"
            "原因が混在している場合、より細かい切り分けが必要。\n"
            "p_ability_win (Stage1) やオッズ帯別の予測をバックテストに出力する\n"
            "仕組みを整備してから再分析する。"
        ),
    }
    lines.append(next_steps.get(code, "個別に検討が必要。"))

    return "\n".join(lines)


# ── メイン処理 ──────────────────────────────────────────────────

def run_ablation() -> dict[str, Any]:
    """アブレーション分析を実行。"""
    print("=== 単勝Pモデル アブレーション診断 ===\n")

    # 1) データ読み込み
    print("[1] データ読み込み...")
    df = load_horse_features()
    print(f"  Surface: {df['surface'].value_counts().to_dict()}")

    meta: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "total_rows": len(df),
        "years": sorted(df["year"].unique().tolist()),
        "surfaces": df["surface"].value_counts().to_dict(),
        "roi_note": ROI_NOTE,
    }

    # 2) 各候補を分析
    print("\n[2] 候補別分析...")
    candidates_data: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        cdata = analyze_single_candidate(df, cand)
        candidates_data.append(cdata)

    # 3) Surface別サマリ
    print("\n[3] Surface別サマリ...")
    surface_summary = compute_surface_summary(df, CANDIDATES)

    # 4) 比較サマリ表
    print("\n[4] 比較サマリ表...")
    comp_table = build_comparison_table(candidates_data)

    # 5) 原因判定
    print("\n[5] 原因判定...")
    verdict = determine_verdict(candidates_data)
    print(f"  判定: {verdict['code']}")
    print(f"  理由: {verdict['reason']}")

    results = {
        "meta": meta,
        "candidates": {f"{c['id']}_{c['col']}": c for c in candidates_data},
        "surface_summary": surface_summary,
        "comparison_table": comp_table,
        "verdict": verdict,
    }

    return results


def main() -> None:
    """エントリポイント。"""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    results = run_ablation()

    # JSON 出力
    json_path = ANALYSIS_DIR / "win_pmodel_ablation_diagnostic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")

    # Markdown 出力
    md_text = generate_ablation_markdown(results)
    md_path = ANALYSIS_DIR / "win_pmodel_ablation_diagnostic.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Markdown saved: {md_path}")

    print(f"\n=== 完了: 判定={results['verdict']['code']} ===")


if __name__ == "__main__":
    main()
