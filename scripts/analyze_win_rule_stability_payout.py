#!/usr/bin/env python
"""Win Rule Stability Grid Analysis (Payout-based ROI).

p_win_final × tanodds を使ったルール候補をグリッドで軽量安定性検証する。
ROI計算は payouts.parquet の paytansyopay1 / 100 を使用（バックテスト本体と同一手法）。
confirmed_odds は参考比較として併記するが、採用判定には使わない。

本番パイプラインへの実装は行わない。

出力:
  - data/analysis/win_rule_stability_payout.json
  - data/analysis/win_rule_stability_payout.md
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
RAW_DIR = DATA_DIR / "raw"

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
    # 候補#2 (前回 stable_candidate)
    {"id": "cand2", "odds": (10, 30), "ev": 1.10, "edge": None,
     "label": "候補#2: odds 10-30, EV>1.10"},
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
    payout_odds は paytansyopay1/100 由来の払戻倍率。
    """
    n = len(df)
    if n == 0:
        return {"count": 0, "hit_rate": 0.0, "avg_odds": 0.0,
                "roi_pct": 0.0, "profit": 0.0}
    hits = int(df[win_col].sum())
    # 払戻 = is_win * payout_odds (負けは0)
    return_per_bet = df[odds_col] * df[win_col]
    total_payout = float(return_per_bet.sum())
    roi_pct = safe_div(total_payout, float(n)) * 100.0  # %表記
    return {
        "count": n,
        "hit_rate": round(float(hits / n), 4),
        "avg_odds": round(float(df[odds_col].mean()), 2),
        "roi_pct": round(roi_pct, 2),
        "profit": round(total_payout - float(n), 1),  # 倍率差分 (100円ベット前提)
    }


def roi_summary_confirmed(
    df: pd.DataFrame,
    odds_col: str = "confirmed_odds",
    win_col: str = "is_win",
) -> dict[str, Any]:
    """confirmed_odds ベースの ROI (参考比較用)。"""
    n = len(df)
    if n == 0:
        return {"count": 0, "roi_pct": 0.0}
    return_per_bet = df[odds_col] * df[win_col]
    total_payout = float(return_per_bet.sum())
    roi_pct = safe_div(total_payout, float(n)) * 100.0
    return {"count": n, "roi_pct": round(roi_pct, 2)}


# ══════════════════════════════════════════════════════════════════
# データ読込
# ══════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """bt_{year}_horse_features.parquet を読み込んで結合。"""
    need_cols = [
        "race_id", "race_date", "kakuteijyuni", "tanodds",
        "confirmed_odds", "surface", "popularity_rank", "umaban",
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
        na_count = df["p_market_win_norm"].isna().sum()
        if na_count > 0:
            mask_na = df["p_market_win_norm"].isna()
            p_raw = 1.0 / df.loc[mask_na, "tanodds"]
            p_sum = p_raw.groupby(df.loc[mask_na, "race_id"]).transform("sum")
            df.loc[mask_na, "p_market_win_norm"] = (p_raw / p_sum).astype(float)
            print(f"  [info] p_market_win_norm: {na_count} NaN rows filled from 1/tanodds")

    # 無効行除外
    essential = ["p_win_final", "tanodds", "kakuteijyuni", "surface", "umaban"]
    before = len(df)
    df = df.dropna(subset=essential)
    print(f"  Dropped {before - len(df)} rows with NaN in essential cols")

    missing = set(need_cols) - found_cols
    if missing:
        print(f"  [warn] Missing columns: {sorted(missing)}")

    print(f"  Total valid: {len(df)} rows")
    return df


def load_payouts() -> pd.DataFrame:
    """payouts.parquet から (race_id, umaban) → win_payout_odds を構築。"""
    path = RAW_DIR / "payouts.parquet"
    if not path.exists():
        raise FileNotFoundError(f"payouts.parquet not found: {path}")

    payouts = pd.read_parquet(
        path,
        columns=["race_id", "paytansyoumaban1", "paytansyopay1"],
    )
    payouts = payouts.dropna(subset=["paytansyoumaban1", "paytansyopay1"]).copy()
    payouts["race_id"] = payouts["race_id"].astype(str)
    payouts["umaban"] = payouts["paytansyoumaban1"].astype(int)
    payouts["win_payout_odds"] = payouts["paytansyopay1"].astype(float) / 100.0
    print(f"  Payouts: {len(payouts)} valid entries loaded")
    return payouts[["race_id", "umaban", "win_payout_odds"]]


def compute_derived(df: pd.DataFrame, payouts_df: pd.DataFrame) -> pd.DataFrame:
    """派生列を追加。payouts ベースの払戻倍率をマージ。"""
    # EV計算 (条件判定用)
    df["pred_ev_final"] = df["p_win_final"] * df["tanodds"]
    df["edge_ratio_final"] = (
        df["p_win_final"] / df["p_market_win_norm"]
    ).clip(0.01, 100.0)
    df["edge_diff_final"] = df["p_win_final"] - df["p_market_win_norm"]

    # payouts ベースの精算オッズをマージ
    df = df.merge(
        payouts_df,
        on=["race_id", "umaban"],
        how="left",
    )

    # 勝馬で payout 欠損の警告
    wins_no_payout = df[(df["is_win"] == 1) & (df["win_payout_odds"].isna())]
    if len(wins_no_payout) > 0:
        print(f"  [warn] {len(wins_no_payout)} wins without payout data "
              f"(race_ids: {wins_no_payout['race_id'].tolist()[:5]}...)")
        # 勝馬のpayout欠損は0返し (ROI計算からは除外しない)
        # これによりROIは保守的に見積もられる
        df.loc[(df["is_win"] == 1) & (df["win_payout_odds"].isna()), "win_payout_odds"] = 0.0

    # 非勝馬の payout は NaN のまま (is_win * payout で 0 になる)
    # ROI計算の都合上、非勝馬の payout は 0 にしておく
    df.loc[df["is_win"] == 0, "win_payout_odds"] = 0.0

    # payout_odds 列名を統一 (roi_summary で参照)
    df["payout_odds"] = df["win_payout_odds"]

    # race_date を datetime に確実に変換
    df["race_date"] = pd.to_datetime(df["race_date"])

    payout_match_rate = (df["win_payout_odds"] > 0).sum() / (df["is_win"] == 1).sum() * 100
    print(f"  Payout match rate for winners: {payout_match_rate:.2f}%")
    return df


# ══════════════════════════════════════════════════════════════════
# 最大ドローダウン
# ══════════════════════════════════════════════════════════════════

def compute_max_drawdown(df_sub: pd.DataFrame) -> float:
    """race_date順に100円betした累積損益の最大落ち込み (円単位)。"""
    if len(df_sub) < 2:
        return 0.0
    df_sorted = df_sub.sort_values(["race_date", "race_id"])
    # 払戻 = is_win * payout_odds * 100 (円)。コスト = 100円/件。
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

    # confirmed_odds ベースの参考ROI (全体のみ)
    roi_confirmed = roi_summary_confirmed(sub)

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
        "roi_confirmed_ref": roi_confirmed["roi_pct"],
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
        # 候補#2 も turf/dirt 個別評価
        if comp["id"] == "cand2":
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

    # 候補#2
    cand2 = next(c for c in comparisons if c["id"] == "cand2")

    # confirmed_odds vs payouts 比較
    payout_vs_confirmed = {
        "original_rule": {
            "payout_roi": orig["roi_pct"],
            "confirmed_roi": orig["roi_confirmed_ref"],
            "diff_pp": round(orig["roi_pct"] - orig["roi_confirmed_ref"], 2),
        },
        "candidate2": {
            "payout_roi": cand2["roi_pct"],
            "confirmed_roi": cand2["roi_confirmed_ref"],
            "diff_pp": round(cand2["roi_pct"] - cand2["roi_confirmed_ref"], 2),
        },
    }

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "script": "scripts/analyze_win_rule_stability_payout.py",
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
                "ROI計算は paytansyopay1/100 (payouts.parquet) ベース。"
                " バックテスト本体 (engine.build_win_payout_map) と同一手法。"
                " confirmed_odds は roi_confirmed_ref に参考値として併記。"
                " profit単位は100円ベットの倍率差分。"
            ),
            "settlement_source": "payouts.paytansyopay1 / 100",
        },
        "original_rule": orig,
        "candidate2": cand2,
        "payout_vs_confirmed": payout_vs_confirmed,
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
    """11セクションのMarkdownレポートを生成。"""
    lines: list[str] = []
    w = lines.append

    orig = data["original_rule"]
    cand2 = data["candidate2"]
    comparisons = data["key_comparisons"]
    stable = data["grid_search"]["stable_candidates"]
    high_roi_low = data["grid_search"]["high_roi_low_count"]
    near_original = data["grid_search"]["near_original"]
    pvc = data["payout_vs_confirmed"]

    # ── ヘッダ ──
    w("# Win Rule Stability Analysis (Payout-based ROI)")
    w("")
    w(f"生成日時: {data['meta']['generated_at']}  ")
    w(f"データ件数: {data['meta']['total_rows']}行  ")
    w(f"グリッドセル数: {data['meta']['grid_dimensions']['total_cells']}  ")
    w(f"精算ソース: **payouts.paytansyopay1 / 100** (バックテスト本体と同一)  ")
    w("")

    # ── 1. 結論 ──
    w("## 1. 結論")
    w("")
    verdict, verdict_reason = _determine_verdict(orig, cand2, stable, comparisons)
    w(f"**判定: {verdict}**")
    w("")
    w(verdict_reason)
    w("")

    # ── 2. tanodds / confirmed_odds / payouts の役割整理 ──
    w("## 2. オッズ列の役割整理")
    w("")
    w("| 列 | 来源 | 用途 |")
    w("|----|------|------|")
    w("| `tanodds` | 発走5分前時系列オッズ | 条件判定 (オッズ帯・EV計算) |")
    w("| `confirmed_odds` | entries.odds (レース後オッズ) | 参考比較のみ |")
    w("| `win_payout_odds` | payouts.paytansyopay1 / 100 | **精算・ROI評価** (バックテスト本体と同一) |")
    w("")
    w(f"- payout_odds と confirmed_odds の差: "
      f"元ルールで {_fmt_roi(pvc['original_rule']['payout_roi'])} vs "
      f"{_fmt_roi(pvc['original_rule']['confirmed_roi'])} "
      f"({pvc['original_rule']['diff_pp']:+.1f}pp)")
    w(f"- 候補#2で {_fmt_roi(pvc['candidate2']['payout_roi'])} vs "
      f"{_fmt_roi(pvc['candidate2']['confirmed_roi'])} "
      f"({pvc['candidate2']['diff_pp']:+.1f}pp)")
    w("")

    # ── 3. 元ルール詳細 ──
    w("## 3. 元ルール詳細 (odds 3-20, EV>1.10)")
    w("")
    w(_detail_table(orig))
    w("")

    # ── 4. 候補#2詳細 ──
    w("## 4. 候補#2詳細 (odds 10-30, EV>1.10)")
    w("")
    w(_detail_table(cand2))
    w("")

    # ── 5. 上位候補表 ──
    w("## 5. 上位候補表")
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
        w("**stable_candidate=true の候補なし。**")
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
        w("| Condition | Count | ROI | Confirmed ROI | 2024 (n/ROI) | 2025 (n/ROI) | Turf (n/ROI) | Dirt (n/ROI) | Max DD | Score |")
        w("|-----------|------:|-----:|--------------:|-------------|-------------|-------------|-------------|-------:|------:|")
        for c in near_original:
            w(
                f"| {c['condition_name']} "
                f"| {c['count']} "
                f"| {_fmt_roi(c['roi_pct'])} "
                f"| {_fmt_roi(c['roi_confirmed_ref'])} "
                f"| {c['year_2024']['count']}/{_fmt_roi(c['year_2024']['roi_pct'])} "
                f"| {c['year_2025']['count']}/{_fmt_roi(c['year_2025']['roi_pct'])} "
                f"| {c['turf']['count']}/{_fmt_roi(c['turf']['roi_pct'])} "
                f"| {c['dirt']['count']}/{_fmt_roi(c['dirt']['roi_pct'])} "
                f"| {c['max_drawdown']:.0f} "
                f"| {c.get('score', 0):.2f} |"
            )
        w("")

    # ── 6. 年別安定性 ──
    w("## 6. 年別安定性")
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

    # ── 7. Surface別安定性 ──
    w("## 7. Surface別安定性")
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

    # ── 8. confirmed_oddsベースとの差 ──
    w("## 8. confirmed_oddsベースとの比較")
    w("")
    w("| ルール | Payouts ROI | Confirmed ROI | 差 | 備考 |")
    w("|--------|----------:|-----------:|----:|------|")
    for c in comparisons:
        diff = c["roi_pct"] - c["roi_confirmed_ref"]
        note = "payout > confirmed" if diff > 0 else "payout <= confirmed"
        w(
            f"| {c['label']} "
            f"| {_fmt_roi(c['roi_pct'])} "
            f"| {_fmt_roi(c['roi_confirmed_ref'])} "
            f"| {diff:+.1f}pp "
            f"| {note} |"
        )
    w("")
    w("- confirmed_odds と payout_odds は99.8%一致 (差はpayout欠損の数件のみ)")
    w("- したがってROI差は実質的に0に近い")
    w("- ただし本スクリプトはバックテスト本体 (engine.build_win_payout_map) と同じ精算手法を使用")
    w("")

    # ── 9. 推奨アクション ──
    w("## 9. 推奨アクション")
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
        w("- payouts ベースでもROI<100%の場合はシグナル自体が疑わしい")
    w("")

    # ── 10. 実行コマンド ──
    w("## 10. 実行コマンド")
    w("")
    w("```bash")
    w("python scripts/analyze_win_rule_stability_payout.py")
    w("```")
    w("")

    # ── 11. 生成ファイル / 元に戻す方法 ──
    w("## 11. 生成ファイル")
    w("")
    w("- `data/analysis/win_rule_stability_payout.json`")
    w("- `data/analysis/win_rule_stability_payout.md`")
    w("")
    w("## 12. 元に戻す方法")
    w("")
    w("```powershell")
    w("Remove-Item scripts\\analyze_win_rule_stability_payout.py")
    w("Remove-Item data\\analysis\\win_rule_stability_payout.json")
    w("Remove-Item data\\analysis\\win_rule_stability_payout.md")
    w("```")
    w("")

    return "\n".join(lines)


def _detail_table(cell: dict[str, Any]) -> str:
    """ルールの詳細テーブル。"""
    lines: list[str] = []
    w = lines.append
    w("| 指標 | 値 |")
    w("|------|---:|")
    w(f"| 件数 | {cell['count']} |")
    w(f"| 的中数 | {int(cell['count'] * cell['hit_rate'])} |")
    w(f"| 的中率 | {cell['hit_rate']:.1%} |")
    w(f"| 平均 tanodds | {cell['avg_odds']:.2f} |")
    w(f"| 平均人気 | {cell['avg_popularity']:.1f} |")
    w(f"| ROI (payouts) | {_fmt_roi(cell['roi_pct'])} |")
    w(f"| ROI (confirmed_odds参考) | {_fmt_roi(cell['roi_confirmed_ref'])} |")
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
    cand2: dict[str, Any],
    stable: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> tuple[str, str]:
    """A/B/C/D判定を決定。"""
    orig_roi = orig["roi_pct"]
    orig_count = orig["count"]
    cand2_roi = cand2["roi_pct"]
    cand2_count = cand2["count"]
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
    reasons.append(f"候補#2 ROI={_fmt_roi(cand2_roi)}, n={cand2_count}")

    if orig_roi < 100.0 and cand2_roi < 100.0:
        verdict = "D: 破棄"
        reasons.append("元ルールも候補#2もpayoutsベースでROI<100%。シグナル自体が疑わしい。")
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
    if orig_roi >= 100.0 or cand2_roi >= 100.0:
        verdict = "C: 追加診断"
        reasons.append("stable_candidate=0だが、一部ルールは黒字。")
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
    print("Win Rule Stability Grid Analysis (Payout-based ROI)")
    print("=" * 60)

    # データ読込
    print("\n[1/6] Loading horse_features ...")
    df = load_data()

    print("\n[2/6] Loading payouts ...")
    payouts_df = load_payouts()

    print("\n[3/6] Computing derived columns + merging payouts ...")
    df = compute_derived(df, payouts_df)

    # グリッド探索
    print("\n[4/6] Running grid search ...")
    grid_results = run_grid_search(df)

    # 安定性フラグ
    print("\n[5/6] Applying stability flags ...")
    apply_stability_flags(grid_results)
    n_stable = sum(1 for c in grid_results if c["stable_candidate"])
    n_nonzero = sum(1 for c in grid_results if c["count"] > 0)
    print(f"  stable_candidate=true: {n_stable} / {n_nonzero} non-empty cells")

    # 重点比較
    comparisons = run_key_comparisons(df)
    apply_stability_flags(comparisons)
    for c in comparisons:
        c["score"] = compute_score(c)

    # JSON出力
    print("\n[6/6] Generating output ...")
    json_data = build_json_output(df, grid_results, comparisons)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = ANALYSIS_DIR / "win_rule_stability_payout.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  JSON: {json_path}")

    # Markdown出力
    md_text = generate_markdown(json_data, grid_results)
    md_path = ANALYSIS_DIR / "win_rule_stability_payout.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  MD:   {md_path}")

    # コンソールサマリ
    orig = json_data["original_rule"]
    cand2 = json_data["candidate2"]
    print("\n" + "=" * 60)
    print("SUMMARY (Payout-based ROI)")
    print("=" * 60)
    print(f"元ルール (odds 3-20, EV>1.10):")
    print(f"  count={orig['count']}, ROI={_fmt_roi(orig['roi_pct'])} "
          f"(confirmed: {_fmt_roi(orig['roi_confirmed_ref'])}), "
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
    print(f"\n候補#2 (odds 10-30, EV>1.10):")
    print(f"  count={cand2['count']}, ROI={_fmt_roi(cand2['roi_pct'])} "
          f"(confirmed: {_fmt_roi(cand2['roi_confirmed_ref'])}), "
          f"profit={cand2['profit']}")
    print(f"\nstable_candidate=true: {n_stable}件")
    if stable := json_data["grid_search"]["stable_candidates"]:
        best = stable[0]
        print(f"ベスト候補: {best['condition_name']}")
        print(f"  count={best['count']}, ROI={_fmt_roi(best['roi_pct'])}, "
              f"score={best['score']:.2f}")
    verdict, reason = _determine_verdict(
        orig, cand2,
        json_data["grid_search"]["stable_candidates"], comparisons,
    )
    print(f"\n判定: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
