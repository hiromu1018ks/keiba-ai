from __future__ import annotations

from models.reproducibility import (
    catboost_params,
    lightgbm_native_params,
    lightgbm_sklearn_params,
    xgboost_params,
)


def test_lightgbm_native_params_enable_deterministic_row_wise() -> None:
    params = lightgbm_native_params(seed=123)

    assert params["seed"] == 123
    assert params["data_random_seed"] == 123
    assert params["feature_fraction_seed"] == 123
    assert params["bagging_seed"] == 123
    assert params["extra_seed"] == 123
    assert params["drop_seed"] == 123
    assert params["deterministic"] is True
    assert params["force_row_wise"] is True
    assert "force_col_wise" not in params


def test_lightgbm_native_params_can_force_col_wise() -> None:
    params = lightgbm_native_params(seed=123, histogram_mode="col")

    assert params["deterministic"] is True
    assert params["force_col_wise"] is True
    assert "force_row_wise" not in params


def test_lightgbm_sklearn_params_use_random_state_alias() -> None:
    params = lightgbm_sklearn_params(seed=123)

    assert params["random_state"] == 123
    assert "seed" not in params
    assert params["feature_fraction_seed"] == 123
    assert params["deterministic"] is True


def test_xgboost_and_catboost_params_pin_cpu_seed() -> None:
    assert xgboost_params(seed=123) == {
        "seed": 123,
        "tree_method": "hist",
        "device": "cpu",
    }
    assert catboost_params(seed=123) == {"random_seed": 123}
