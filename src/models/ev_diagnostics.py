"""EV推定精度診断モジュール.

OOF EV推定値と実際の払戻額を比較し、EV推定精度を評価する。
EVF-02要件: ECE + Brier score分解 + Reliability diagram + 時系列ドリフト追跡。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logger = logging.getLogger("models.ev_diagnostics")

# 診断対象列
EV_PRED_COLUMN = "ev_win_calibrated"
EV_ACTUAL_COLUMN = "actual_ev_win"
EV_LOWER_COLUMN = "EV_lower_win_corrected"
EDGE_COLUMN = "win_selection_edge"
WIN_COLUMN = "kakuteijyuni"
DATE_COLUMN = "race_date"

# 設定
N_BINS = 10
MIN_SAMPLE_SIZE = 30
N_BINS_RELIABILITY = 10


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error (Guo et al. 2017).

    quantile binning (equal-frequency) を使用。EV予測は右裾が長いため。
    """
    bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bin_boundaries[0] = -np.inf
    bin_boundaries[-1] = np.inf

    ece = 0.0
    n_total = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        avg_confidence = float(y_prob[mask].mean())
        avg_accuracy = float(y_true[mask].mean())
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)
    return ece


def _brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Brier score decomposition (Murphy 1973).

    Returns: brier_score, reliability (lower=better), resolution (higher=better), uncertainty.
    """
    brier = float(np.mean((y_prob - y_true) ** 2))
    bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bin_boundaries[0] = -np.inf
    bin_boundaries[-1] = np.inf

    n = len(y_true)
    o_bar = float(y_true.mean())

    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        o_k = float(y_true[mask].mean())
        f_k = float(y_prob[mask].mean())
        reliability += (n_k / n) * (o_k - f_k) ** 2
        resolution += (n_k / n) * (o_k - o_bar) ** 2

    uncertainty = o_bar * (1 - o_bar)
    return {
        "brier_score": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


def _reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = N_BINS_RELIABILITY,
) -> dict:
    """Reliability diagram data using sklearn.calibration_curve."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true,
        y_prob,
        n_bins=n_bins,
        strategy="quantile",
    )
    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
        "n_bins": n_bins,
    }


def _temporal_drift(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
) -> list[dict]:
    """年度別の時系列ドリフト追跡.

    各年度のサンプル数、平均EV予測、平均実際払戻、相関を計算。
    """
    if DATE_COLUMN not in df.columns:
        return []
    df_copy = df.copy()
    df_copy["year"] = pd.to_datetime(df_copy[DATE_COLUMN]).dt.year

    results: list[dict] = []
    for year, group in df_copy.groupby("year"):
        pred = pd.to_numeric(group[pred_col], errors="coerce").dropna()
        actual = pd.to_numeric(group[actual_col], errors="coerce").dropna()
        # 共通index
        common = pred.index.intersection(actual.index)
        if len(common) < MIN_SAMPLE_SIZE:
            results.append(
                {
                    "year": int(year),
                    "n": len(common),
                    "warning": "insufficient_samples",
                }
            )
            continue
        pred_vals = pred.loc[common].values
        actual_vals = actual.loc[common].values
        corr_r = float("nan")
        if len(common) >= 2:
            try:
                corr_r, _ = pearsonr(pred_vals, actual_vals)
            except (ValueError, FloatingPointError):
                pass
        results.append(
            {
                "year": int(year),
                "n": int(len(common)),
                "mean_predicted_ev": float(pred_vals.mean()),
                "mean_actual_ev": float(actual_vals.mean()),
                "ev_bias": float(pred_vals.mean() - actual_vals.mean()),
                "correlation": float(corr_r),
            }
        )
    return results


def compute_ev_diagnostics(
    df_oof: pd.DataFrame,
    output_path: Path | None = None,
    surface: str | None = None,
) -> dict:
    """EV推定精度の深度診断を実行する (EVF-02, D-04).

    Parameters
    ----------
    df_oof : pd.DataFrame
        OOF予測DataFrame。ev_win_corrected, confirmed_odds, kakuteijyuni, race_dateを含む。
    output_path : Path | None
        JSON出力先パス。Noneならファイル出力なし。
    surface : str | None
        サーフェス名(ログ表示用)。

    Returns
    -------
    dict
        診断結果。
    """
    surface_label = surface or "all"
    result: dict = {"surface": surface_label}

    # Phase 19: ev_win_calibrated がなければ ev_win_corrected にフォールバック
    if EV_PRED_COLUMN not in df_oof.columns and "ev_win_corrected" in df_oof.columns:
        ev_pred_col = "ev_win_corrected"
    else:
        ev_pred_col = EV_PRED_COLUMN

    # Phase 19: EV_lower フォールバック
    if EV_LOWER_COLUMN not in df_oof.columns and "EV_lower_win" in df_oof.columns:
        ev_lower_col = "EV_lower_win"
    else:
        ev_lower_col = EV_LOWER_COLUMN

    # Phase 19: edge フォールバック
    if EDGE_COLUMN not in df_oof.columns and "edge_win" in df_oof.columns:
        edge_col = "edge_win"
    else:
        edge_col = EDGE_COLUMN

    # actual_ev_win列がなければ計算
    if EV_ACTUAL_COLUMN not in df_oof.columns:
        if "confirmed_odds" in df_oof.columns and WIN_COLUMN in df_oof.columns:
            df_oof = df_oof.copy()
            df_oof[EV_ACTUAL_COLUMN] = df_oof["confirmed_odds"] * (df_oof[WIN_COLUMN] == 1).astype(
                float
            )
        else:
            logger.warning("EV diagnostics: cannot compute actual EV -- missing columns")
            result["error"] = "missing_actual_ev_columns"
            return result

    # 有効データ抽出
    pred = pd.to_numeric(df_oof[ev_pred_col], errors="coerce")
    actual = pd.to_numeric(df_oof[EV_ACTUAL_COLUMN], errors="coerce")
    valid_mask = pred.notna() & actual.notna()
    n_valid = int(valid_mask.sum())

    result["n_total"] = len(df_oof)
    result["n_valid"] = n_valid

    if n_valid < MIN_SAMPLE_SIZE:
        logger.warning(
            "EV diagnostics (%s): insufficient samples (%d < %d)",
            surface_label,
            n_valid,
            MIN_SAMPLE_SIZE,
        )
        result["warning"] = "insufficient_samples"
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    pred_vals = pred[valid_mask].values
    actual_vals = actual[valid_mask].values

    # 1. 相関 + RMSE
    corr_r, corr_p = pearsonr(pred_vals, actual_vals)
    rmse = float(np.sqrt(np.mean((pred_vals - actual_vals) ** 2)))
    result["correlation"] = {"r": float(corr_r), "p_value": float(corr_p)}
    result["rmse"] = rmse

    # 2. Win確率版のECE/Brier (kakuteijyuni==1 をバイナリターゲット)
    win_binary = (df_oof.loc[valid_mask, WIN_COLUMN] == 1).astype(float).values
    prob_col = "p_win_corrected" if "p_win_corrected" in df_oof.columns else "p_win_pred"
    if prob_col in df_oof.columns:
        prob_vals = (
            pd.to_numeric(df_oof.loc[valid_mask, prob_col], errors="coerce").fillna(0.0).values
        )
        prob_vals = np.clip(prob_vals, 0.0, 1.0)
        result["ece"] = _compute_ece(win_binary, prob_vals)
        result["brier"] = _brier_decomposition(win_binary, prob_vals)
        result["brier_sklearn"] = float(brier_score_loss(win_binary, prob_vals))
        result["reliability_diagram"] = _reliability_diagram_data(win_binary, prob_vals)
    else:
        result["ece"] = None
        result["brier"] = None
        result["reliability_diagram"] = None
        logger.warning("EV diagnostics: no probability column found for ECE/Brier")

    # 3. EV過大/過小評価バイアス
    ev_bias = float(pred_vals.mean() - actual_vals.mean())
    result["ev_bias"] = ev_bias
    result["mean_predicted_ev"] = float(pred_vals.mean())
    result["mean_actual_ev"] = float(actual_vals.mean())

    # 4. EV分布統計
    result["ev_predicted_stats"] = {
        "mean": float(pred_vals.mean()),
        "std": float(pred_vals.std()),
        "q25": float(np.percentile(pred_vals, 25)),
        "q50": float(np.percentile(pred_vals, 50)),
        "q75": float(np.percentile(pred_vals, 75)),
    }
    result["ev_actual_stats"] = {
        "mean": float(actual_vals.mean()),
        "std": float(actual_vals.std()),
        "q25": float(np.percentile(actual_vals, 25)),
        "q50": float(np.percentile(actual_vals, 50)),
        "q75": float(np.percentile(actual_vals, 75)),
    }

    # 5. 時系列ドリフト (年度別)
    result["temporal_drift"] = _temporal_drift(
        df_oof.loc[valid_mask].copy(),
        ev_pred_col,
        EV_ACTUAL_COLUMN,
    )

    # JSON出力
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("EV diagnostics written to %s", output_path)

    return result


def console_summary(result: dict) -> None:
    """EV診断結果のコンソールサマリを出力する (D-05).

    Parameters
    ----------
    result : dict
        compute_ev_diagnostics()の戻り値。
    """
    surface = result.get("surface", "?")
    n = result.get("n_valid", 0)
    logger.info("=== EV Diagnostics (%s) ===", surface)
    logger.info("  Samples: %d / %d", n, result.get("n_total", 0))

    if "warning" in result:
        logger.info("  WARNING: %s", result["warning"])
        return

    corr = result.get("correlation", {})
    logger.info(
        "  Correlation: r=%.4f (p=%.4f)",
        corr.get("r", float("nan")),
        corr.get("p_value", float("nan")),
    )
    logger.info("  RMSE: %.4f", result.get("rmse", float("nan")))
    logger.info(
        "  EV Bias: %+.4f (predicted %.4f vs actual %.4f)",
        result.get("ev_bias", float("nan")),
        result.get("mean_predicted_ev", float("nan")),
        result.get("mean_actual_ev", float("nan")),
    )

    ece = result.get("ece")
    if ece is not None:
        logger.info("  ECE: %.4f", ece)

    brier = result.get("brier", {})
    if brier:
        logger.info(
            "  Brier: %.4f (reliability=%.4f, resolution=%.4f, uncertainty=%.4f)",
            brier.get("brier_score", float("nan")),
            brier.get("reliability", float("nan")),
            brier.get("resolution", float("nan")),
            brier.get("uncertainty", float("nan")),
        )

    drift = result.get("temporal_drift", [])
    if drift:
        logger.info("  Temporal drift (%d years):", len(drift))
        for entry in drift:
            if "warning" in entry:
                logger.info(
                    "    %s: %s (n=%d)",
                    entry.get("year"),
                    entry["warning"],
                    entry.get("n", 0),
                )
            else:
                logger.info(
                    "    %s: n=%d, bias=%+.4f, corr=%.3f",
                    entry.get("year"),
                    entry.get("n", 0),
                    entry.get("ev_bias", float("nan")),
                    entry.get("correlation", float("nan")),
                )
