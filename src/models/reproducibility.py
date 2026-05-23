"""Shared reproducibility settings for tree-based models."""

from __future__ import annotations

from typing import Any, Literal

DEFAULT_RANDOM_SEED = 42
HistogramMode = Literal["row", "col"]


def lightgbm_native_params(
    seed: int = DEFAULT_RANDOM_SEED,
    *,
    histogram_mode: HistogramMode = "row",
) -> dict[str, Any]:
    """Return LightGBM native API parameters for repeatable CPU training."""
    params: dict[str, Any] = {
        "seed": seed,
        "data_random_seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "extra_seed": seed,
        "drop_seed": seed,
        "deterministic": True,
    }
    if histogram_mode == "row":
        params["force_row_wise"] = True
    else:
        params["force_col_wise"] = True
    return params


def lightgbm_sklearn_params(
    seed: int = DEFAULT_RANDOM_SEED,
    *,
    histogram_mode: HistogramMode = "row",
) -> dict[str, Any]:
    """Return LightGBM sklearn API parameters for repeatable CPU training."""
    params = lightgbm_native_params(seed=seed, histogram_mode=histogram_mode)
    params["random_state"] = seed
    params.pop("seed", None)
    return params


def xgboost_params(seed: int = DEFAULT_RANDOM_SEED) -> dict[str, Any]:
    """Return XGBoost parameters for repeatable single-node CPU training."""
    return {
        "seed": seed,
        "tree_method": "hist",
        "device": "cpu",
    }


def catboost_params(seed: int = DEFAULT_RANDOM_SEED) -> dict[str, Any]:
    """Return CatBoost parameters for repeatable CPU training."""
    return {"random_seed": seed}
