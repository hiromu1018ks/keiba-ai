"""ドリフト診断モジュール.

ks_2samp/wasserstein_distanceによるOOF確率分布ドリフト診断。
GATE-02要件: 単一モデル/アンサンブル間の分布差異の定量化。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

logger = logging.getLogger("models.drift_diagnostics")

# ドリフト診断対象列
DRIFT_COLUMNS = [
    "p_win_pred",
    "ev_win",
    "p_win_corrected",
    "ev_win_corrected",
    "win_selection_prob",
    "win_selection_edge",
    "win_selection_ev",
]

# ドリフト検出閾値
KS_PVALUE_THRESHOLD = 0.05
WASSERSTEIN_WARN_THRESHOLD = 0.05
MIN_SAMPLE_SIZE = 30


def _compute_column_stats(series: pd.Series) -> dict[str, float]:
    """単一列の基本統計量を計算する."""
    clean = series.dropna()
    n = len(clean)
    if n < MIN_SAMPLE_SIZE:
        return {}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "q25": float(clean.quantile(0.25)),
        "q50": float(clean.quantile(0.50)),
        "q75": float(clean.quantile(0.75)),
        "n": n,
    }


def _compare_columns(
    df_oof: pd.DataFrame,
    df_baseline: pd.DataFrame,
    col: str,
) -> dict[str, float] | None:
    """2つのDataFrame間で単一列のドリフト比較を行う."""
    oof_vals = df_oof[col].dropna() if col in df_oof.columns else pd.Series(dtype=float)
    base_vals = df_baseline[col].dropna() if col in df_baseline.columns else pd.Series(dtype=float)

    if len(oof_vals) < MIN_SAMPLE_SIZE or len(base_vals) < MIN_SAMPLE_SIZE:
        return None

    ks_stat, ks_pvalue = ks_2samp(oof_vals, base_vals)
    wd = wasserstein_distance(oof_vals, base_vals)

    return {
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "wasserstein_distance": float(wd),
    }


def _compute_leaf_stats(
    df_oof: pd.DataFrame,
    df_baseline: pd.DataFrame | None = None,
) -> dict:
    """再帰しない最小単位のドリフト診断（列stats + baseline比較）."""
    result: dict = {
        "drift_detected": False,
        "columns": {},
    }

    drift_columns_count = 0

    for col in DRIFT_COLUMNS:
        if col not in df_oof.columns:
            continue

        stats = _compute_column_stats(df_oof[col])
        if not stats:
            continue

        col_entry: dict = {"stats": stats}

        if df_baseline is not None and col in df_baseline.columns:
            comparison = _compare_columns(df_oof, df_baseline, col)
            if comparison is not None:
                col_entry["comparison"] = comparison
                if comparison["ks_pvalue"] < KS_PVALUE_THRESHOLD:
                    drift_columns_count += 1
                if comparison["wasserstein_distance"] > WASSERSTEIN_WARN_THRESHOLD:
                    drift_columns_count += 1

        result["columns"][col] = col_entry

    if df_baseline is not None and drift_columns_count > 0:
        result["drift_detected"] = True

    return result


def compute_drift_diagnostics(
    df_oof: pd.DataFrame,
    df_baseline: pd.DataFrame | None = None,
    *,
    output_path: Path | None = None,
    surface: str = "unknown",
) -> dict:
    """OOF確率分布のドリフト診断を実行する.

    Parameters
    ----------
    df_oof : pd.DataFrame
        現在のOOF予測DataFrame。
    df_baseline : pd.DataFrame | None
        比較対象のbaseline DataFrame。None時はcharacterization mode。
    output_path : Path | None
        JSON出力パス。None時はファイル出力なし。
    surface : str
        サーフェス名（ログ用）。

    Returns
    -------
    dict
        ドリフト診断結果。
    """
    result: dict = {
        "surface": surface,
        "drift_detected": False,
        "columns": {},
        "surfaces": {},
        "years": {},
    }

    # 列stats + baseline比較
    leaf = _compute_leaf_stats(df_oof, df_baseline)
    result["columns"] = leaf["columns"]
    result["drift_detected"] = leaf["drift_detected"]

    # ドリフト検出判定 + ログ
    if df_baseline is not None and result["drift_detected"]:
        result["recommendations"] = [
            "WinSelectionGateを現在のOOF予測で再学習してください",
            "JSONレポートで分布シフトの大きさを確認してください",
        ]
        logger.warning(
            "Drift detected in %d column(s) for surface=%s. Gate retraining recommended.",
            sum(
                1
                for col_data in result["columns"].values()
                if "comparison" in col_data
                and (
                    col_data["comparison"]["ks_pvalue"] < KS_PVALUE_THRESHOLD
                    or col_data["comparison"]["wasserstein_distance"] > WASSERSTEIN_WARN_THRESHOLD
                )
            ),
            surface,
        )

    # サーフェス別breakdown (再帰しない、leafのみ)
    if "surface" in df_oof.columns:
        surfaces_in_df = df_oof["surface"].dropna().unique()
        for surf in surfaces_in_df:
            surf_str = str(surf)
            if surf_str not in ("turf", "dirt"):
                continue
            surf_df = df_oof[df_oof["surface"] == surf]
            if len(surf_df) < MIN_SAMPLE_SIZE:
                continue
            surf_baseline = None
            if df_baseline is not None and "surface" in df_baseline.columns:
                surf_baseline = df_baseline[df_baseline["surface"] == surf]
            surf_leaf = _compute_leaf_stats(surf_df, surf_baseline)
            result["surfaces"][surf_str] = {
                "columns": surf_leaf["columns"],
                "drift_detected": surf_leaf["drift_detected"],
            }

    # 年度別breakdown (再帰しない、leafのみ)
    if "race_date" in df_oof.columns:
        dates = pd.to_datetime(df_oof["race_date"], errors="coerce")
        years = dates.dt.year.dropna().unique()
        for year in sorted(years):
            year_str = str(int(year))
            year_df = df_oof[dates.dt.year == year]
            if len(year_df) < MIN_SAMPLE_SIZE:
                continue
            year_baseline = None
            if df_baseline is not None and "race_date" in df_baseline.columns:
                base_dates = pd.to_datetime(df_baseline["race_date"], errors="coerce")
                year_baseline = df_baseline[base_dates.dt.year == year]
            year_leaf = _compute_leaf_stats(year_df, year_baseline)
            result["years"][year_str] = {
                "columns": year_leaf["columns"],
                "drift_detected": year_leaf["drift_detected"],
            }

    # JSON出力
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=_json_default)
        logger.info("Drift diagnostics saved to %s", output_path)

    return result


def _json_default(obj: object) -> object:
    """JSON非対応型のフォールバック."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def console_summary(result: dict) -> None:
    """ドリフト診断結果のフォーマット済みサマリをログ出力する.

    Parameters
    ----------
    result : dict
        compute_drift_diagnostics()の戻り値。
    """
    surface = result.get("surface", "unknown")
    logger.info("=== Drift Diagnostics Summary (surface=%s) ===", surface)

    for col, data in result.get("columns", {}).items():
        stats = data.get("stats", {})
        if "comparison" in data:
            comp = data["comparison"]
            logger.info(
                "  %s: KS=%.4f p=%.4f WD=%.4f (mean=%.4f std=%.4f n=%d)",
                col,
                comp["ks_stat"],
                comp["ks_pvalue"],
                comp["wasserstein_distance"],
                stats.get("mean", float("nan")),
                stats.get("std", float("nan")),
                stats.get("n", 0),
            )
        else:
            logger.info(
                "  %s: mean=%.4f std=%.4f q50=%.4f n=%d",
                col,
                stats.get("mean", float("nan")),
                stats.get("std", float("nan")),
                stats.get("q50", float("nan")),
                stats.get("n", 0),
            )

    # サーフェス別サマリ
    for surf, surf_data in result.get("surfaces", {}).items():
        drift_flag = "DRIFT DETECTED" if surf_data.get("drift_detected") else "OK"
        n_cols = len(surf_data.get("columns", {}))
        logger.info("  Surface %s: %s (%d columns)", surf, drift_flag, n_cols)

    # 年度別サマリ
    for year, year_data in result.get("years", {}).items():
        drift_flag = "DRIFT DETECTED" if year_data.get("drift_detected") else "OK"
        n_cols = len(year_data.get("columns", {}))
        logger.info("  Year %s: %s (%d columns)", year, drift_flag, n_cols)

    if result.get("drift_detected"):
        logger.info(">>> DRIFT DETECTED — Gate retraining recommended <<<")
    else:
        logger.info("No significant drift detected.")
