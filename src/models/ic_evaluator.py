"""IC (Information Coefficient) 評価モジュール.

OOF予測に対する市場独立予測力を4定式化で定量的に測定する。
RIC-01〜06要件: B差分IC/C直交IC/E Incremental IC/Per-race IC + 方向一致性検証。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger("models.ic_evaluator")

# モジュール定数
MIN_SAMPLE_SIZE = 30
IC_TARGET_COLUMN = "kakuteijyuni"
MODEL_PRED_COLUMN = "p_win_corrected"
MARKET_ODDS_COLUMN = "tanodds"
MIN_HORSES_PER_RACE = 5


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


def _get_market_probability(df: pd.DataFrame) -> np.ndarray:
    """市場確率をDataFrameから取得する (D-05).

    implied_prob列があれば使用、なければ1/tanoddsから計算。
    """
    if "implied_prob" in df.columns:
        return pd.to_numeric(df["implied_prob"], errors="coerce").values
    odds = pd.to_numeric(df[MARKET_ODDS_COLUMN], errors="coerce").replace(0, np.nan)
    return np.clip(1.0 / odds.values, 0.01, 0.99)


def _compute_b_difference_ic(
    model_pred: np.ndarray,
    market_prob: np.ndarray,
    y: np.ndarray,
) -> dict:
    """B-difference IC: Spearman(delta, y) where delta = model - market (RIC-01)."""
    delta = model_pred - market_prob
    valid = np.isfinite(delta) & np.isfinite(y)
    n = int(valid.sum())
    if n < MIN_SAMPLE_SIZE:
        return {"rho": float("nan"), "p_value": float("nan"), "n": n}
    rho, p_value = spearmanr(delta[valid], y[valid])
    return {"rho": float(rho), "p_value": float(p_value), "n": n}


def _compute_c_orthogonal_ic(
    model_pred: np.ndarray,
    market_prob: np.ndarray,
    y: np.ndarray,
) -> dict:
    """C-orthogonal IC: Spearman(resid, y) where resid = model - OLS(model|market) (RIC-02)."""
    valid = np.isfinite(model_pred) & np.isfinite(market_prob) & np.isfinite(y)
    n = int(valid.sum())
    if n < MIN_SAMPLE_SIZE:
        return {"rho": float("nan"), "p_value": float("nan"), "n": n}
    x = market_prob[valid]
    y_pred = model_pred[valid]
    x_with_intercept = np.column_stack([np.ones(len(x)), x])
    coeffs, _, _, _ = np.linalg.lstsq(x_with_intercept, y_pred, rcond=None)
    residuals = y_pred - x_with_intercept @ coeffs
    rho, p_value = spearmanr(residuals, y[valid])
    return {"rho": float(rho), "p_value": float(p_value), "n": n}


def _compute_e_incremental_ic(
    model_pred: np.ndarray,
    market_prob: np.ndarray,
    y: np.ndarray,
) -> dict:
    """E-incremental IC: IC(model, y) - IC(market, y) (RIC-03)."""
    valid = np.isfinite(model_pred) & np.isfinite(market_prob) & np.isfinite(y)
    n = int(valid.sum())
    if n < MIN_SAMPLE_SIZE:
        return {
            "ic_model": float("nan"),
            "ic_market": float("nan"),
            "delta_ic": float("nan"),
            "n": n,
        }
    ic_model, _ = spearmanr(model_pred[valid], y[valid])
    ic_market, _ = spearmanr(market_prob[valid], y[valid])
    return {
        "ic_model": float(ic_model),
        "ic_market": float(ic_market),
        "delta_ic": float(ic_model - ic_market),
        "n": n,
    }


def _compute_per_race_ic(
    df: pd.DataFrame,
    pred_col: str,
    y_col: str,
    group_col: str = "race_id",
    min_horses: int = MIN_HORSES_PER_RACE,
) -> dict:
    """Per-race IC: レース内Spearmanの平均 (RIC-04)."""
    results: list[float] = []
    skipped = 0
    for _, group in df.groupby(group_col, observed=True):
        pred = pd.to_numeric(group[pred_col], errors="coerce").dropna()
        actual = pd.to_numeric(group[y_col], errors="coerce").dropna()
        common = pred.index.intersection(actual.index)
        if len(common) < min_horses:
            skipped += 1
            continue
        rho, _ = spearmanr(pred.loc[common].values, actual.loc[common].values)
        if np.isfinite(rho):
            results.append(float(rho))
    if not results:
        return {
            "mean_rho": float("nan"),
            "std_rho": float("nan"),
            "median_rho": float("nan"),
            "n_races": 0,
            "skipped_races": skipped,
        }
    return {
        "mean_rho": float(np.mean(results)),
        "std_rho": float(np.std(results)),
        "median_rho": float(np.median(results)),
        "n_races": len(results),
        "skipped_races": skipped,
    }


def _check_direction_consistency(ic_results: dict) -> dict:
    """4種IC指標の方向一致性を検証する (RIC-06)."""
    ic_values: list[tuple[str, float]] = []
    for key in ["b_difference", "c_orthogonal", "e_incremental", "per_race"]:
        metric = ic_results.get(key, {})
        rho = metric.get("rho")
        if rho is None:
            rho = metric.get("delta_ic")
        if rho is None:
            rho = metric.get("mean_rho")
        if rho is not None and np.isfinite(rho):
            ic_values.append((key, float(rho)))

    if len(ic_values) < 2:
        return {"consistent": True, "n_metrics_checked": len(ic_values), "details": {}}

    signs = [1 if v > 0 else -1 if v < 0 else 0 for _, v in ic_values]
    non_zero_signs = [s for s in signs if s != 0]

    consistent = len(set(non_zero_signs)) <= 1
    result: dict = {
        "consistent": consistent,
        "n_metrics_checked": len(ic_values),
        "details": {k: v for k, v in ic_values},
    }
    if not consistent:
        result["warning"] = "IC direction inconsistency detected -- possible computation error"
        logger.warning("IC direction inconsistency: %s", {k: v for k, v in ic_values})
    return result


def run_ic_evaluation(
    df_oof: pd.DataFrame,
    output_path: Path | None = None,
    mlflow_log: bool = False,
) -> dict:
    """IC評価のメインエントリポイント.

    Parameters
    ----------
    df_oof : pd.DataFrame
        OOF予測DataFrame。
    output_path : Path | None
        JSON出力パス。None時はファイル出力なし。
    mlflow_log : bool
        MLflowへのメトリクス記録を有効化。

    Returns
    -------
    dict
        IC評価結果。
    """
    # 列検証
    pred_col = MODEL_PRED_COLUMN if MODEL_PRED_COLUMN in df_oof.columns else "p_win_pred"
    if pred_col not in df_oof.columns:
        raise ValueError(f"Required column '{pred_col}' not found in DataFrame")

    odds_cols = [MARKET_ODDS_COLUMN, "implied_prob"]
    if not any(c in df_oof.columns for c in odds_cols):
        raise ValueError(f"Required column ({MARKET_ODDS_COLUMN} or implied_prob) not found")

    for col in [IC_TARGET_COLUMN, "surface", "race_id"]:
        if col not in df_oof.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # データ抽出
    model_pred = pd.to_numeric(df_oof[pred_col], errors="coerce").values
    market_prob = _get_market_probability(df_oof)
    y = (pd.to_numeric(df_oof[IC_TARGET_COLUMN], errors="coerce") == 1).astype(float).values

    result: dict = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_total": len(df_oof),
        "n_valid": int(np.isfinite(model_pred).sum()),
    }

    # Surface別IC計算
    for surface_key, surface_filter in [
        ("turf", df_oof["surface"] == "turf"),
        ("dirt", df_oof["surface"] == "dirt"),
        ("all", pd.Series(True, index=df_oof.index)),
    ]:
        if surface_key != "all" and surface_filter.sum() < MIN_SAMPLE_SIZE:
            result[surface_key] = {"warning": "insufficient_samples"}
            continue

        sub_df = df_oof[surface_filter]
        sub_pred = model_prob_filter(model_pred, market_prob, y, surface_filter)
        if sub_pred is None:
            result[surface_key] = {"warning": "insufficient_samples"}
            continue

        sp, sm, sy = sub_pred
        surface_result = {
            "b_difference": _compute_b_difference_ic(sp, sm, sy),
            "c_orthogonal": _compute_c_orthogonal_ic(sp, sm, sy),
            "e_incremental": _compute_e_incremental_ic(sp, sm, sy),
            "per_race": _compute_per_race_ic(sub_df, pred_col, IC_TARGET_COLUMN),
        }
        result[surface_key] = surface_result

    # 方向一致性チェック
    consistency: dict = {}
    for surface_key in ["turf", "dirt", "all"]:
        if surface_key in result and "warning" not in result[surface_key]:
            consistency[surface_key] = _check_direction_consistency(result[surface_key])
    result["consistency_check"] = consistency

    # JSON出力
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=_json_default)
        logger.info("IC baseline written to %s", output_path)

    # MLflow記録
    if mlflow_log:
        _log_mlflow(result)

    return result


def model_prob_filter(
    model_pred: np.ndarray,
    market_prob: np.ndarray,
    y: np.ndarray,
    mask: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """マスクでフィルタリングした配列を返す。サンプル不足時はNone."""
    sp = model_pred[mask.values]
    sm = market_prob[mask.values]
    sy = y[mask.values]
    valid = np.isfinite(sp) & np.isfinite(sm) & np.isfinite(sy)
    n = int(valid.sum())
    if n < MIN_SAMPLE_SIZE:
        return None
    return sp[valid], sm[valid], sy[valid]


def _log_mlflow(result: dict) -> None:
    """MLflowへIC評価メトリクスを記録する (D-06)."""
    import mlflow

    for surface_key in ["turf", "dirt", "all"]:
        if surface_key not in result or "warning" in result[surface_key]:
            continue
        sr = result[surface_key]
        prefix = f"ic_{surface_key}_"

        b = sr.get("b_difference", {})
        if "rho" in b and np.isfinite(b["rho"]):
            mlflow.log_metric(f"{prefix}b_diff_rho", b["rho"])

        c = sr.get("c_orthogonal", {})
        if "rho" in c and np.isfinite(c["rho"]):
            mlflow.log_metric(f"{prefix}c_orth_rho", c["rho"])

        e = sr.get("e_incremental", {})
        if "delta_ic" in e and np.isfinite(e["delta_ic"]):
            mlflow.log_metric(f"{prefix}e_incr_delta", e["delta_ic"])

        p = sr.get("per_race", {})
        if "mean_rho" in p and np.isfinite(p["mean_rho"]):
            mlflow.log_metric(f"{prefix}per_race_mean", p["mean_rho"])

    consistency = result.get("consistency_check", {})
    for surface_key, check in consistency.items():
        mlflow.set_tag(f"ic_consistency_{surface_key}", str(check.get("consistent", "unknown")))


def console_summary(result: dict) -> None:
    """IC評価結果のコンソールサマリを出力する."""
    logger.info("=== IC Evaluation Summary ===")
    n_total = result.get("n_total", 0)
    n_valid = result.get("n_valid", 0)
    logger.info("  Total samples: %d (valid: %d)", n_total, n_valid)

    for surface_key in ["turf", "dirt", "all"]:
        if surface_key not in result:
            continue
        sr = result[surface_key]
        if "warning" in sr:
            logger.info("  [%s] WARNING: %s", surface_key.upper(), sr["warning"])
            continue

        b = sr.get("b_difference", {})
        c = sr.get("c_orthogonal", {})
        e = sr.get("e_incremental", {})
        p = sr.get("per_race", {})
        sk = surface_key.upper()

        logger.info(
            "  [%s] B-diff rho=%.4f p=%.4f n=%d",
            sk, b.get("rho", float("nan")), b.get("p_value", float("nan")), b.get("n", 0),
        )
        logger.info(
            "  [%s] C-orth rho=%.4f p=%.4f n=%d",
            sk, c.get("rho", float("nan")), c.get("p_value", float("nan")), c.get("n", 0),
        )
        logger.info(
            "  [%s] E-incr delta=%.4f (model=%.4f market=%.4f)",
            sk,
            e.get("delta_ic", float("nan")),
            e.get("ic_model", float("nan")),
            e.get("ic_market", float("nan")),
        )
        logger.info(
            "  [%s] Per-race mean=%.4f median=%.4f n_races=%d",
            sk,
            p.get("mean_rho", float("nan")),
            p.get("median_rho", float("nan")),
            p.get("n_races", 0),
        )

    consistency = result.get("consistency_check", {})
    for surface_key, check in consistency.items():
        status = "CONSISTENT" if check.get("consistent") else "INCONSISTENT"
        n_metrics = check.get("n_metrics_checked", 0)
        logger.info(
            "  [%s] Direction: %s (%d metrics)", surface_key, status, n_metrics,
        )
