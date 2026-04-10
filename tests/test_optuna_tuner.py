import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from tuning.optuna_tuner import OptunaTuner, SEARCH_SPACES


def _make_train_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "race_id": [f"R{i//10:04d}" for i in range(n)],
        "race_date": pd.date_range("2020-01-01", periods=n, freq="h"),
        "surface": np.where(rng.rand(n) > 0.5, "turf", "dirt"),
        "distance_bin": "mile",
        "track_condition_code": 2,
        "grade_code": "C",
        "field_size": 14,
        "kakuteijyuni": rng.randint(1, 18, n),
        "p_ability_win": rng.rand(n),
        "odds": rng.uniform(2, 50, n),
        "fukuoddslow": rng.uniform(1.5, 10, n),
    })


class TestOptunaTuner:
    def test_search_spaces_valid(self):
        """検索空間が妥当な範囲"""
        for model_name, space in SEARCH_SPACES.items():
            assert "num_leaves" in space or "learning_rate" in space or "alpha" in space
            if "num_leaves" in space:
                lo, hi = space["num_leaves"]
                assert 7 <= lo <= 127
                assert lo <= hi <= 127

    def test_objective_returns_float(self):
        """目的関数が float を返す"""
        df = _make_train_data()
        tuner = OptunaTuner(model_type="win_hit")
        trial = MagicMock()
        trial.suggest_int = MagicMock(return_value=31)
        trial.suggest_float = MagicMock(return_value=0.03)
        score = tuner.objective(trial, df)
        assert isinstance(score, float)

    def test_tune_runs(self):
        """チューニングが完了する (n_trials=3)"""
        df = _make_train_data()
        tuner = OptunaTuner(model_type="win_hit")
        result = tuner.tune(df, n_trials=3)
        assert "best_params" in result
        assert "best_value" in result
        assert isinstance(result["best_params"], dict)
