#!/usr/bin/env python
"""Win Rule Stability Grid Analysis.

p_win_final × tanodds を使ったルール候補をグリッドで軽量安定性検証する。
本番パイプラインへの実装は行わない。

出力:
  - data/analysis/win_rule_stability.json
  - data/analysis/win_rule_stability.md
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# 定数
# ══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
BACKTEST_DIR = DATA_DIR / "backtest"

# グリッド定義
ODDS_BANDS: list[tuple[int, int]] = [
    (1, 3),
    (3, 10),
    (3, 15),
    (3, 20),
    (5, 15),
    (5, 20),
    (10, 30),
    (1, 20),
    (1, 50),
]

EV_THRESHOLDS: list[float] = [1.00, 1.05, 1.10, 1.15, 1.20, 1.30]

EDGE_RATIOS: list[float | None] = [None, 1.00, 1.05, 1.10, 1.15]

SURFACES: list[str] = ["all", "turf", "dirt"]

# スコア重み
ROI_WEIGHT = 1.0
COUNT_WEIGHT = 3.0
YEAR_GAP_PENALTY = 0.5
SURFACE_GAP_PENALTY = 0.3
DRAWDOWN_PENALTY = 0.2

# 重点比較定義
KEY_COMPARISONS: list[dict[str, Any]] = [
    {"id": "orig", "odds": (3, 20), "ev": 1.10, "edge": None,
     "label": "元ルール: odds 3-20, EV>1.10"},
    {"id": "nbr1", "odds": (3, 20), "ev": 1.05, "edge": None,
     "label": "近傍: odds 3-20, EV>1.05"},
    {"id": "nbr2", "odds": (3, 20), "ev": 1.15, "edge": None,
     "label": "近傍: odds 3-20, EV>1.15"},
    {"id": "nbr3", "odds": (3, 15), "ev": 1.10, "edge": None,
     "label": "近傍: odds 3-15, EV>1.10"},
    {"id": "nbr4", "odds": (5, 20), "ev": 1.10, "edge": None,
     "label": "近傍: odds 5-20, EV>1.10"},
    {"id": "edge1", "odds": (3, 20), "ev": 1.10, "edge": 1.05,
     "label": "Edge追加: odds 3-20, EV>1.10, edge>1.05"},
    {"id": "edge2", "odds": (3, 20), "ev": 1.10, "edge": 1.10,
     "label": "Edge追加: odds 3-20, EV>1.10, edge>1.10"},
]


# ══════════════════════════════════════════════════════════════════
# ユーティリティ
# ══════════════════════════════════════════════════════════════════

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """ゼロ除算安全な割り算。"""
    return a / b if b != 0 else default


def roi_summary(
    df: pd.DataFrame,
    odds_col: str = "payout_odds",
    win_col: str = "is_win",
) -> dict[str, Any]:
    """ROI・的中率などの基本統計。各馬100円買いの仮想ROI。

    ROIは%表記 (例: 105.24 = 105.24%)。
    """
    n = len(df)
    if n == 0:
        return {"count": 0, "hit_rate": 0.0, "avg_odds": 0.0,
                "roi_pct": 0.0, "profit": 0.0}
    hits = int(df[win_col].sum())
    total_payout = float((df[odds_col] * df[win_col]).sum())
    roi_pct = safe_div(total_payout, float(n)) * 100.0  # %表記
    return {
        "count": n,
        "hit_rate": round(float(hits / n), 4),
        "avg_odds": round(float(df[odds_col].mean()), 2),
        "roi_pct": round(roi_pct, 2),
        "profit": round(total_payout - float(n), 1),
    }


# ══════════════════════════════════════════════════════════════════
# データ読込
# ══════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """bt_{year}_horse_features.parquet を読み込んで結合。"""
    need_cols = [
        "race_id", "race_date", "kakuteijyuni", "tanodds",
        "confirmed_odds", "surface", "popularity_rank",
        "p_win_final", "p_market_win_norm",
        "field_size", "distance_bin", "track_condition_code", "grade_code",
    ]
    frames: list[pd.DataFrame] = []
    found_cols: set[str] = set()

    for year in [2024, 2025]:
        path = BACKTEST_DIR / f"bt_{year}_horse_features.parquet"
        if not path.exists():
            print(f"  [skip] {year}: {path} not found")
            continue
        available = pd.read_parquet(path).columns.tolist()
        cols = [c for c in need_cols if c in available]
        df = pd.read_parquet(path, columns=cols)
        df["year"] = year
        df["race_id"] = df["race_id"].astype(str)
        frames.append(df)
        found_cols.update(cols)
        print(f"  {year}: {len(df)} rows, {len(cols)} cols loaded")

    if not frames:
        raise FileNotFoundError("No backtest horse_features files found")

    df = pd.concat(frames, ignore_index=True)

    # 前処理
    df = df[df["tanodds"] > 0].copy()
    df["is_win"] = (df["kakuteijyuni"] == 1).astype(int)

    # p_market_win_norm: 既存列を最優先。欠損時のみフォールバック。
    if "p_market_win_norm" not in df.columns or df["p_market_win_norm"].isna().all():
        p_raw = 1.0 / df["tanodds"]
        p_sum = p_raw.groupby(df["race_id"]).transform("sum")
        df["p_market_win_norm"] = (p_raw / p_sum).astype(float)
        print("  [info] p_market_win_norm: computed from 1/tanodds (fallback)")
    else:
        # 既存列がある場合はそのまま使う。欠損行のみ補完。
        na_count = df["p_market_win_norm"].isna().sum()
        if na_count > 0:
            mask_na = df["p_market_win_norm"].isna()
            p_raw = 1.0 / df.loc[mask_na, "tanodds"]
            p_sum = p_raw.groupby(df.loc[mask_na, "race_id"]).transform("sum")
            df.loc[mask_na, "p_market_win_norm"] = (p_raw / p_sum).astype(float)
            print(f"  [info] p_market_win_norm: {na_count} NaN rows filled from 1/tanodds")

    # 無効行除外
    essential = ["p_win_final", "tanodds", "kakuteijyuni", "surface"]
    before = len(df)
    df = df.dropna(subset=essential)
    print(f"  Dropped {before - len(df)} rows with NaN in essential cols")

    missing = set(need_cols) - found_cols
    if missing:
        print(f"  [warn] Missing columns: {sorted(missing)}")

    print(f"  Total valid: {len(df)} rows")
    return df


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """派生列を追加。"""
    df["pred_ev_final"] = df["p_win_final"] * df["tanodds"]
    df["edge_ratio_final"] = (
        df["p_win_final"] / df["p_market_win_norm"]
    ).clip(0.01, 100.0)
    df["edge_diff_final"] = df["p_win_final"] - df["p_market_win_norm"]
    # confirmed_odds 優先、なければ tanodds
    df["payout_odds"] = df["confirmed_odds"].fillna(df["tanodds"])
    # race_date を datetime に確実に変換
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


# ══════════════════════════════════════════════════════════════════
# 最大ドローダウン
# ══════════════════════════════════════════════════════════════════

def compute_max_drawdown(df_sub: pd.DataFrame) -> float:
    """race_date順に100円betした累積損益の最大落ち込み (円単位)。"""
    if len(df_sub) < 2:
        return 0.0
    df_sorted = df_sub.sort_values(["race_date", "race_id"])
    pnl = np.where(
        df_sorted["is_win"].values == 1,
        df_sorted["payout_odds"].values * 100.0 - 100.0,
        -100.0,
    )
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    drawdowns = running_max - cumulative
    return float(np.max(drawdowns))


# ══════════════════════════════════════════════════════════════════
# グリッド評価
# ══════════════════════════════════════════════════════════════════

def evaluate_grid_cell(
    df: pd.DataFrame,
    odds_lo: int,
    odds_hi: int,
    ev_thr: float,
    edge_thr: float | None,
    surface: str,
) -> dict[str, Any]:
    """1グリッドセルの全指標を計算。"""
    mask = (
        (df["tanodds"] >= odds_lo)
        & (df["tanodds"] <= odds_hi)
        & (df["pred_ev_final"] > ev_thr)
    )
    if edge_thr is not None:
        mask &= df["edge_ratio_final"] > edge_thr
    if surface != "all":
        mask &= df["surface"] == surface

    sub = df[mask]
    overall = roi_summary(sub)

    # 年別
    sub_2024 = sub[sub["year"] == 2024]
    sub_2025 = sub[sub["year"] == 2025]
    year_2024 = roi_summary(sub_2024)
    year_2025 = roi_summary(sub_2025)

    # Surface別 (常に両方計算)
    sub_turf = sub[sub["surface"] == "turf"]
    sub_dirt = sub[sub["surface"] == "dirt"]
    turf = roi_summary(sub_turf)
    dirt = roi_summary(sub_dirt)

    # 最大ドローダウン
    max_dd = compute_max_drawdown(sub)

    # 平均人気
    avg_pop = round(float(sub["popularity_rank"].mean()), 2) if len(sub) > 0 else 0.0

    # 条件名
    edge_str = f" & edge>{edge_thr:.2f}" if edge_thr is not None else ""
    surf_str = f" [{surface}]" if surface != "all" else ""
    condition_name = f"odds {odds_lo}-{odds_hi} & EV>{ev_thr:.2f}{edge_str}{surf_str}"

    return {
        "condition_name": condition_name,
        "odds_band": [odds_lo, odds_hi],
        "ev_threshold": ev_thr,
        "edge_threshold": edge_thr,
        "surface": surface,
        **overall,
        "avg_popularity": avg_pop,
        "year_2024": year_2024,
        "year_2025": year_2025,
        "turf": turf,
        "dirt": dirt,
        "max_drawdown": round(max_dd, 1),
    }


def run_grid_search(df: pd.DataFrame) -> list[dict[str, Any]]:
    """全810セルを評価。"""
    results: list[dict[str, Any]] = []
    total = len(ODDS_BANDS) * len(EV_THRESHOLDS) * len(EDGE_RATIOS) * len(SURFACES)
    print(f"Grid search: {total} cells ...")
    for odds_lo, odds_hi in ODDS_BANDS:
        for ev_thr in EV_THRESHOLDS:
            for edge_thr in EDGE_RATIOS:
                for surface in SURFACES:
                    cell = evaluate_grid_cell(
                        df, odds_lo, odds_hi, ev_thr, edge_thr, surface,
                    )
                    results.append(cell)
    print(f"  Done: {len(results)} cells evaluated")
    return results


# ══════════════════════════════════════════════════════════════════
# 安定性フラグ
# ══════════════════════════════════════════════════════════════════

def apply_stability_flags(results: list[dict[str, Any]]) -> None:
    """各セルに stable_candidate / reference_only / stability_flags を付与。"""
    for cell in results:
        count = cell["count"]
        roi_total = cell["roi_pct"]
        roi_2024 = cell["year_2024"]["roi_pct"]
        roi_2025 = cell["year_2025"]["roi_pct"]
        cnt_2024 = cell["year_2024"]["count"]
        cnt_2025 = cell["year_2025"]["count"]
        roi_turf = cell["turf"]["roi_pct"]
        roi_dirt = cell["dirt"]["roi_pct"]
        cnt_turf = cell["turf"]["count"]
        cnt_dirt = cell["dirt"]["count"]
        dd = cell["max_drawdown"]
        profit_yen = cell["profit"] * 100.0  # profit単位を円に変換

        stable = True
        flags: list[str] = []

        if count < 100:
            stable = False
            flags.append("count < 100")
        if cnt_2024 < 30 or cnt_2025 < 30:
            stable = False
            flags.append("year count < 30")
        if roi_total < 100.0:
            stable = False
            flags.append("total ROI < 100%")
        if roi_2024 < 90.0 or roi_2025 < 90.0:
            stable = False
            flags.append("year ROI < 90%")
        # turf/dirt両方に件数がある場合は両方チェック
        if cnt_turf >= 30 and cnt_dirt >= 30:
            if roi_turf < 80.0 or roi_dirt < 80.0:
                stable = False
                flags.append("surface ROI < 80% (both present)")
        dd_limit = max(20000.0, profit_yen * 3.0)
        if dd > dd_limit:
            stable = False
            flags.append("max_drawdown exceeds limit")

        reference_only = count < 50

        cell["stable_candidate"] = stable
        cell["reference_only"] = reference_only
        cell["stability_flags"] = flags


# ══════════════════════════════════════════════════════════════════
# スコアリング
# ══════════════════════════════════════════════════════════════════

def compute_score(cell: dict[str, Any]) -> float:
    """総合スコアを計算。

    score = ROI_WEIGHT * min(roi_pct, 150)
          + COUNT_WEIGHT * log1p(count)
          - YEAR_GAP_PENALTY * |roi_2024 - roi_2025| / 100
          - SURFACE_GAP_PENALTY * |roi_turf - roi_dirt| / 100
          - DRAWDOWN_PENALTY * max(0, max_drawdown - profit_yen) / 10000

    roi_pct は%表記 (例: 105.24)。
    """
    if cell["count"] == 0:
        return -999.0

    roi_pct = min(cell["roi_pct"], 150.0)
    count = cell["count"]
    roi_2024 = cell["year_2024"]["roi_pct"]
    roi_2025 = cell["year_2025"]["roi_pct"]
    roi_turf = cell["turf"]["roi_pct"]
    roi_dirt = cell["dirt"]["roi_pct"]
    cnt_turf = cell["turf"]["count"]
    cnt_dirt = cell["dirt"]["count"]
    dd = cell["max_drawdown"]
    profit_yen = cell["profit"] * 100.0

    year_gap = abs(roi_2024 - roi_2025) / 100.0
    surface_gap = (
        abs(roi_turf - roi_dirt) / 100.0
        if cnt_turf >= 30 and cnt_dirt >= 30
        else 0.0
    )
    dd_excess = max(0.0, dd - profit_yen) / 10000.0

    score = (
        ROI_WEIGHT * roi_pct
        + COUNT_WEIGHT * math.log1p(count)
        - YEAR_GAP_PENALTY * year_gap
        - SURFACE_GAP_PENALTY * surface_gap
        - DRAWDOWN_PENALTY * dd_excess
    )
    return round(score, 4)


# ══════════════════════════════════════════════════════════════════
# 重点比較
# ══════════════════════════════════════════════════════════════════

def run_key_comparisons(df: pd.DataFrame) -> list[dict[str, Any]]:
    """元ルール近傍の重点比較。"""
    results: list[dict[str, Any]] = []
    for comp in KEY_COMPARISONS:
        cell = evaluate_grid_cell(
            df, comp["odds"][0], comp["odds"][1],
            comp["ev"], comp["edge"], "all",
        )
        cell["id"] = comp["id"]
        cell["label"] = comp["label"]
        # 元ルールは turf/dirt も個別に評価
        if comp["id"] == "orig":
            cell["turf_detail"] = evaluate_grid_cell(
                df, comp["odds"][0], comp["odds"][1],
                comp["ev"], comp["edge"], "turf",
            )
            cell["dirt_detail"] = evaluate_grid_cell(
                df, comp["odds"][0], comp["odds"][1],
                comp["ev"], comp["edge"], "dirt",
            )
        results.append(cell)
    return results


# ══════════════════════════════════════════════════════════════════
# JSON出力構築
# ══════════════════════════════════════════════════════════════════

def build_json_output(
    df: pd.DataFrame,
    grid_results: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """JSON出力のトップレベル構造を構築。"""
    # スコア計算
    for cell in grid_results:
        cell["score"] = compute_score(cell)

    # 安定性でソート
    stable = [c for c in grid_results if c["stable_candidate"] and c["count"] > 0]
    stable.sort(key=lambda c: c["score"], reverse=True)

    # 高ROI低件数
    high_roi_low = [
        c for c in grid_results
        if c["roi_pct"] >= 100.0 and c["count"] < 100 and c["count"] > 0
    ]
    high_roi_low.sort(key=lambda c: c["roi_pct"], reverse=True)

    # 近傍条件 (元ルール: odds 3-20, EV>1.10, edge=None)
    near_original = [
        c for c in grid_results
        if c["odds_band"] == [3, 20]
        and c["surface"] == "all"
        and c["edge_threshold"] is None
        and c["count"] > 0
    ]
    near_original.sort(key=lambda c: c["ev_threshold"])

    # 元ルール
    orig = next(c for c in comparisons if c["id"] == "orig")

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "script": "scripts/analyze_win_rule_stability.py",
            "total_rows": len(df),
            "years": sorted(df["year"].unique().tolist()),
            "surface_counts": df["surface"].value_counts().to_dict(),
            "grid_dimensions": {
                "odds_bands": len(ODDS_BANDS),
                "ev_thresholds": len(EV_THRESHOLDS),
                "edge_ratios": len(EDGE_RATIOS),
                "surfaces": len(SURFACES),
                "total_cells": len(ODDS_BANDS) * len(EV_THRESHOLDS)
                               * len(EDGE_RATIOS) * len(SURFACES),
            },
            "scoring_formula": {
                "description": (
                    "score = ROI_W * min(roi_pct, 150)"
                    " + COUNT_W * log1p(count)"
                    " - YEAR_GAP_W * |roi_2024 - roi_2025| / 100"
                    " - SURF_GAP_W * |roi_turf - roi_dirt| / 100"
                    " - DD_W * max(0, max_drawdown - profit_yen) / 10000"
                ),
                "weights": {
                    "roi_weight": ROI_WEIGHT,
                    "count_weight": COUNT_WEIGHT,
                    "year_gap_penalty": YEAR_GAP_PENALTY,
                    "surface_gap_penalty": SURFACE_GAP_PENALTY,
                    "drawdown_penalty": DRAWDOWN_PENALTY,
                },
                "roi_unit": "percent (105.24 = 105.24%)",
            },
            "payout_note": (
                "ROI計算は confirmed_odds 優先 (欠損時 tanodds フォールバック)。"
                " profit単位は100円ベットの倍率差分。"
            ),
        },
        "original_rule": orig,
        "key_comparisons": comparisons,
        "grid_search": {
            "total_cells": len(grid_results),
            "stable_candidates": stable,
            "high_roi_low_count": high_roi_low[:20],
            "near_original": near_original,
            "all_cells": grid_results,
        },
    }


# ══════════════════════════════════════════════════════════════════
# Markdownレポート
# ══════════════════════════════════════════════════════════════════

def _fmt_roi(val: float) -> str:
    """ROIを%表記の文字列に。"""
    return f"{val:.1f}%"


def _fmt_cell_row(c: dict[str, Any]) -> str:
    """1セルをテーブル行に。"""
    return (
        f"| {c['condition_name']} "
        f"| {c['count']} "
        f"| {_fmt_roi(c['roi_pct'])} "
        f"| {c['year_2024']['count']}/{_fmt_roi(c['year_2024']['roi_pct'])} "
        f"| {c['year_2025']['count']}/{_fmt_roi(c['year_2025']['roi_pct'])} "
        f"| {c['turf']['count']}/{_fmt_roi(c['turf']['roi_pct'])} "
        f"| {c['dirt']['count']}/{_fmt_roi(c['dirt']['roi_pct'])} "
        f"| {c['max_drawdown']:.0f} "
        f"| {c.get('score', 0):.2f} |"
    )


def generate_markdown(
    data: dict[str, Any],
    grid_results: list[dict[str, Any]],
) -> str:
    """10セクションのMarkdownレポートを生成。"""
    lines: list[str] = []
    w = lines.append

    orig = data["original_rule"]
    comparisons = data["key_comparisons"]
    stable = data["grid_search"]["stable_candidates"]
    high_roi_low = data["grid_search"]["high_roi_low_count"]
    near_original = data["grid_search"]["near_original"]

    # ── ヘッダ ──
    w("# Win Rule Stability Analysis")
    w("")
    w(f"生成日時: {data['meta']['generated_at']}  ")
    w(f"データ件数: {data['meta']['total_rows']}行  ")
    w(f"グリッドセル数: {data['meta']['grid_dimensions']['total_cells']}  ")
    w("")

    # ── 1. 結論 ──
    w("## 1. 結論")
    w("")
    verdict, verdict_reason = _determine_verdict(orig, stable, comparisons)
    w(f"**判定: {verdict}**")
    w("")
    w(verdict_reason)
    w("")

    # ── 2. 元ルール詳細 ──
    w("## 2. 元ルール詳細 (odds 3-20, EV>1.10)")
    w("")
    w(_detail_table(orig))
    w("")

    # ── 3. 上位候補表 ──
    w("## 3. 上位候補表")
    w("")
    if stable:
        w("### stable_candidate=true (スコア順上位20)")
        w("")
        w("| # | Condition | Count | ROI | 2024 (n/ROI) | 2025 (n/ROI) | Turf (n/ROI) | Dirt (n/ROI) | Max DD | Score |")
        w("|--:|-----------|------:|-----:|-------------|-------------|-------------|-------------|-------:|------:|")
        for i, c in enumerate(stable[:20], 1):
            w(f"| {i} {_fmt_cell_row(c)}")
        w("")
    else:
        w("**stable_candidate=true の候補なし。** 基準を緩和するか、追加診断を検討。")
        w("")

    if high_roi_low:
        w("### 高ROI・低件数 (ROI>=100%, n<100, 上位10)")
        w("")
        w("| # | Condition | Count | ROI | 2024 (n/ROI) | 2025 (n/ROI) | Turf (n/ROI) | Dirt (n/ROI) | Max DD | Score |")
        w("|--:|-----------|------:|-----:|-------------|-------------|-------------|-------------|-------:|------:|")
        for i, c in enumerate(high_roi_low[:10], 1):
            w(f"| {i} {_fmt_cell_row(c)}")
        w("")

    if near_original:
        w("### 元ルール近傍 (odds 3-20, edgeなし, surface=all)")
        w("")
        w("| Condition | Count | ROI | 2024 (n/ROI) | 2025 (n/ROI) | Turf (n/ROI) | Dirt (n/ROI) | Max DD | Score |")
        w("|-----------|------:|-----:|-------------|-------------|-------------|-------------|-------:|------:|")
        for c in near_original:
            w(_fmt_cell_row(c))
        w("")

    # ── 4. 年別安定性 ──
    w("## 4. 年別安定性")
    w("")
    w("| ルール | 2024 Count | 2024 ROI | 2025 Count | 2025 ROI | 年間差 | 備考 |")
    w("|--------|----------:|---------:|----------:|---------:|-------:|------|")
    for c in comparisons:
        gap = abs(c["year_2024"]["roi_pct"] - c["year_2025"]["roi_pct"])
        note = ""
        if c["year_2024"]["count"] < 30:
            note += "2024件数不足 "
        if c["year_2025"]["count"] < 30:
            note += "2025件数不足 "
        if gap > 20:
            note += "年間差大 "
        w(
            f"| {c['label']} "
            f"| {c['year_2024']['count']} "
            f"| {_fmt_roi(c['year_2024']['roi_pct'])} "
            f"| {c['year_2025']['count']} "
            f"| {_fmt_roi(c['year_2025']['roi_pct'])} "
            f"| {gap:.1f}pp "
            f"| {note.strip()} |"
        )
    w("")

    # ── 5. Surface別安定性 ──
    w("## 5. Surface別安定性")
    w("")
    w("| ルール | Turf Count | Turf ROI | Dirt Count | Dirt ROI | Surface差 | 備考 |")
    w("|--------|----------:|---------:|----------:|---------:|----------:|------|")
    for c in comparisons:
        gap = abs(c["turf"]["roi_pct"] - c["dirt"]["roi_pct"])
        note = ""
        if c["turf"]["count"] < 30:
            note += "turf件数不足 "
        if c["dirt"]["count"] < 30:
            note += "dirt件数不足 "
        if c["turf"]["count"] >= 30 and c["dirt"]["count"] >= 30 and gap > 20:
            note += "surface差大 "
        w(
            f"| {c['label']} "
            f"| {c['turf']['count']} "
            f"| {_fmt_roi(c['turf']['roi_pct'])} "
            f"| {c['dirt']['count']} "
            f"| {_fmt_roi(c['dirt']['roi_pct'])} "
            f"| {gap:.1f}pp "
            f"| {note.strip()} |"
        )
    w("")

    # ── 6. edge_ratio追加の効果 ──
    w("## 6. edge_ratio追加の効果")
    w("")
    orig_comp = next(c for c in comparisons if c["id"] == "orig")
    edge1 = next(c for c in comparisons if c["id"] == "edge1")
    edge2 = next(c for c in comparisons if c["id"] == "edge2")
    w("| 条件 | Count | ROI | 2024 ROI | 2025 ROI | Turf ROI | Dirt ROI |")
    w("|------|------:|----:|---------:|---------:|---------:|---------:|")
    for label, c in [("edgeなし (元ルール)", orig_comp),
                     ("edge>1.05", edge1),
                     ("edge>1.10", edge2)]:
        w(
            f"| {label} "
            f"| {c['count']} "
            f"| {_fmt_roi(c['roi_pct'])} "
            f"| {_fmt_roi(c['year_2024']['roi_pct'])} "
            f"| {_fmt_roi(c['year_2025']['roi_pct'])} "
            f"| {_fmt_roi(c['turf']['roi_pct'])} "
            f"| {_fmt_roi(c['dirt']['roi_pct'])} |"
        )
    w("")

    delta1 = edge1["roi_pct"] - orig_comp["roi_pct"]
    delta2 = edge2["roi_pct"] - orig_comp["roi_pct"]
    loss1 = orig_comp["count"] - edge1["count"]
    loss2 = orig_comp["count"] - edge2["count"]
    w(f"- edge>1.05 追加: ROI差 {delta1:+.1f}pp, 件数減 {loss1}")
    w(f"- edge>1.10 追加: ROI差 {delta2:+.1f}pp, 件数減 {loss2}")
    if delta1 > 0 and loss1 < 50:
        w("- **edge>1.05はROI改善と件数維持のバランスが良い**")
    elif delta1 > 0:
        w("- edge>1.05はROI改善するが件数減が大きい")
    else:
        w("- edge>1.05はROIを改善しない（件数を減らすだけ）")
    w("")

    # ── 7. 推奨アクション ──
    w("## 7. 推奨アクション")
    w("")
    w(f"**{verdict}**")
    w("")
    if verdict.startswith("A"):
        w("ルールを無効化可能なフィルタとして実装する。")
        w("- win_strategy にトグル可能な条件として追加")
        w("- デフォルトOFF、設定で有効化")
        w("- Paper tradingで監視")
    elif verdict.startswith("B"):
        w("ルールは不安定なので実装せず、メタモデル設計に進む。")
        w("- 1条件だけ黒字 → 過剰適合の可能性高い")
        w("- 近傍条件でROIが崩壊する場合は特に注意")
        w("- 構造的な改善（メタモデル、MAWC再設計等）を検討")
    elif verdict.startswith("C"):
        w("追加の軽量診断が必要。")
        w("- シグナルは存在するが確証不足")
        w("- 対象期間の拡大や別年のデータで再検証を推奨")
        w("- turf/dirt別ルールの検討")
    else:
        w("単勝EV設計は一旦止める。")
        w("- confirmed_odds補正でROI<100%の場合はシグナル自体が疑わしい")
    w("")

    # ── 8. 実行コマンド ──
    w("## 8. 実行コマンド")
    w("")
    w("```bash")
    w("python scripts/analyze_win_rule_stability.py")
    w("```")
    w("")

    # ── 9. 生成ファイル ──
    w("## 9. 生成ファイル")
    w("")
    w("- `data/analysis/win_rule_stability.json`")
    w("- `data/analysis/win_rule_stability.md`")
    w("")

    # ── 10. 元に戻す方法 ──
    w("## 10. 元に戻す方法")
    w("")
    w("```powershell")
    w("Remove-Item scripts\\analyze_win_rule_stability.py")
    w("Remove-Item data\\analysis\\win_rule_stability.json")
    w("Remove-Item data\\analysis\\win_rule_stability.md")
    w("```")
    w("")

    return "\n".join(lines)


def _detail_table(cell: dict[str, Any]) -> str:
    """元ルールの詳細テーブル。"""
    lines: list[str] = []
    w = lines.append
    w("| 指標 | 値 |")
    w("|------|---:|")
    w(f"| 件数 | {cell['count']} |")
    w(f"| 的中数 | {int(cell['count'] * cell['hit_rate'])} |")
    w(f"| 的中率 | {cell['hit_rate']:.1%} |")
    w(f"| 平均オッズ | {cell['avg_odds']:.2f} |")
    w(f"| 平均人気 | {cell['avg_popularity']:.1f} |")
    w(f"| ROI | {_fmt_roi(cell['roi_pct'])} |")
    w(f"| 利益 (100円bet倍率差分) | {cell['profit']:.1f} |")
    w(f"| 2024 件数/ROI | {cell['year_2024']['count']} / {_fmt_roi(cell['year_2024']['roi_pct'])} |")
    w(f"| 2025 件数/ROI | {cell['year_2025']['count']} / {_fmt_roi(cell['year_2025']['roi_pct'])} |")
    w(f"| Turf 件数/ROI | {cell['turf']['count']} / {_fmt_roi(cell['turf']['roi_pct'])} |")
    w(f"| Dirt 件数/ROI | {cell['dirt']['count']} / {_fmt_roi(cell['dirt']['roi_pct'])} |")
    w(f"| 最大ドローダウン (円) | {cell['max_drawdown']:.0f} |")
    if cell.get("turf_detail"):
        td = cell["turf_detail"]
        w(f"| **Turf詳細** 件数/ROI/DD | {td['count']} / {_fmt_roi(td['roi_pct'])} / {td['max_drawdown']:.0f}円 |")
    if cell.get("dirt_detail"):
        dd = cell["dirt_detail"]
        w(f"| **Dirt詳細** 件数/ROI/DD | {dd['count']} / {_fmt_roi(dd['roi_pct'])} / {dd['max_drawdown']:.0f}円 |")
    return "\n".join(lines)


def _determine_verdict(
    orig: dict[str, Any],
    stable: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> tuple[str, str]:
    """A/B/C/D判定を決定。"""
    orig_roi = orig["roi_pct"]
    orig_count = orig["count"]
    n_stable = len(stable)

    # 近傍で黒字のものを数える
    neighbors_black = sum(
        1 for c in comparisons
        if c["id"].startswith("nbr") and c["roi_pct"] >= 100.0 and c["count"] >= 50
    )
    neighbors_total = sum(1 for c in comparisons if c["id"].startswith("nbr"))

    # 年別チェック
    y2024_roi = orig["year_2024"]["roi_pct"]
    y2025_roi = orig["year_2025"]["roi_pct"]
    year_gap = abs(y2024_roi - y2025_roi)

    # Surface別チェック
    turf_roi = orig["turf"]["roi_pct"]
    dirt_roi = orig["dirt"]["roi_pct"]
    surface_gap = abs(turf_roi - dirt_roi)

    reasons: list[str] = []
    reasons.append(f"元ルール ROI={_fmt_roi(orig_roi)}, n={orig_count}")

    if orig_roi < 100.0:
        verdict = "D: 破棄"
        reasons.append("confirmed_odds使用でROI<100%。シグナル自体が疑わしい。")
        return verdict, "\n".join(reasons)

    # stable_candidateが複数あり、近傍も黒字
    if n_stable >= 3 and neighbors_black >= 2:
        verdict = "A: 実装推奨"
        reasons.append(f"stable_candidate={n_stable}件。近傍{neighbors_black}/{neighbors_total}件が黒字。")
        reasons.append(f"年間差: {year_gap:.1f}pp, Surface差: {surface_gap:.1f}pp。")
        return verdict, "\n".join(reasons)

    # stable_candidateはあるが少数、または近傍が不安定
    if n_stable >= 1:
        verdict = "C: 追加診断"
        reasons.append(f"stable_candidate={n_stable}件だが、近傍は{neighbors_black}/{neighbors_total}件のみ黒字。")
        reasons.append(f"年間差: {year_gap:.1f}pp, Surface差: {surface_gap:.1f}pp。")
        reasons.append("シグナルはあるが確証不足。別データや期間拡大で再検証を推奨。")
        return verdict, "\n".join(reasons)

    # stable_candidate=0
    if orig_roi >= 100.0:
        verdict = "C: 追加診断"
        reasons.append("stable_candidate=0だが、元ルール自体は黒字。")
        reasons.append(f"年間差: {year_gap:.1f}pp, Surface差: {surface_gap:.1f}pp。")
        reasons.append("条件が厳しいだけでシグナルが完全にないわけではない。")
        reasons.append("turf/dirt別ルールや期間拡大を検討。")
        return verdict, "\n".join(reasons)

    verdict = "B: 過剰適合"
    reasons.append("元ルールのみ黒字、近傍で崩壊。過剰適合の可能性高い。")
    reasons.append("メタモデルや構造的改善を優先。")
    return verdict, "\n".join(reasons)


# ══════════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    """エントリポイント。"""
    print("=" * 60)
    print("Win Rule Stability Grid Analysis")
    print("=" * 60)

    # データ読込
    print("\n[1/5] Loading data ...")
    df = load_data()
    df = compute_derived(df)

    # グリッド探索
    print("\n[2/5] Running grid search ...")
    grid_results = run_grid_search(df)

    # 安定性フラグ
    print("\n[3/5] Applying stability flags ...")
    apply_stability_flags(grid_results)
    n_stable = sum(1 for c in grid_results if c["stable_candidate"])
    n_nonzero = sum(1 for c in grid_results if c["count"] > 0)
    print(f"  stable_candidate=true: {n_stable} / {n_nonzero} non-empty cells")

    # 重点比較
    print("\n[4/5] Running key comparisons ...")
    comparisons = run_key_comparisons(df)
    apply_stability_flags(comparisons)
    for c in comparisons:
        c["score"] = compute_score(c)

    # JSON出力
    print("\n[5/5] Generating output ...")
    json_data = build_json_output(df, grid_results, comparisons)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = ANALYSIS_DIR / "win_rule_stability.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  JSON: {json_path}")

    # Markdown出力
    md_text = generate_markdown(json_data, grid_results)
    md_path = ANALYSIS_DIR / "win_rule_stability.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  MD:   {md_path}")

    # コンソールサマリ
    orig = json_data["original_rule"]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"元ルール (odds 3-20, EV>1.10):")
    print(f"  count={orig['count']}, ROI={_fmt_roi(orig['roi_pct'])}, "
          f"profit={orig['profit']}")
    print(f"  2024: n={orig['year_2024']['count']}, "
          f"ROI={_fmt_roi(orig['year_2024']['roi_pct'])}")
    print(f"  2025: n={orig['year_2025']['count']}, "
          f"ROI={_fmt_roi(orig['year_2025']['roi_pct'])}")
    print(f"  Turf: n={orig['turf']['count']}, "
          f"ROI={_fmt_roi(orig['turf']['roi_pct'])}")
    print(f"  Dirt: n={orig['dirt']['count']}, "
          f"ROI={_fmt_roi(orig['dirt']['roi_pct'])}")
    print(f"  MaxDD: {orig['max_drawdown']:.0f}円")
    print(f"\nstable_candidate=true: {n_stable}件")
    if stable := json_data["grid_search"]["stable_candidates"]:
        best = stable[0]
        print(f"ベスト候補: {best['condition_name']}")
        print(f"  count={best['count']}, ROI={_fmt_roi(best['roi_pct'])}, "
              f"score={best['score']:.2f}")
    verdict, reason = _determine_verdict(
        orig, json_data["grid_search"]["stable_candidates"], comparisons,
    )
    print(f"\n判定: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
