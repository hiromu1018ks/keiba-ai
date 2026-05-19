"""IC評価モジュールのテスト (Phase 30)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.ic_evaluator import (
    _check_direction_consistency,
    _compute_b_difference_ic,
    _compute_c_orthogonal_ic,
    _compute_e_incremental_ic,
    _compute_per_race_ic,
    _get_market_probability,
    console_summary,
    run_ic_evaluation,
)


def _make_arrays(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """テスト用の合成配列を生成する."""
    rng = np.random.RandomState(seed)
    market_prob = np.clip(rng.beta(2, 5, n), 0.01, 0.99)
    signal = rng.normal(0, 0.05, n)
    model_pred = np.clip(market_prob + signal, 0.01, 0.99)
    y = (rng.rand(n) < model_pred).astype(float)
    return model_pred, market_prob, y


def _make_oof_df(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """テスト用のOOF DataFrameを生成する."""
    rng = np.random.RandomState(seed)
    n_half = n // 2
    market_prob = np.clip(rng.beta(2, 5, n), 0.01, 0.99)
    signal = rng.normal(0, 0.05, n)
    model_pred = np.clip(market_prob + signal, 0.01, 0.99)
    y = (rng.rand(n) < model_pred).astype(float)

    race_ids = np.repeat([f"R{i:04d}" for i in range(n // 8)], 8)[:n]
    surfaces = ["turf"] * n_half + ["dirt"] * (n - n_half)
    tanodds = 1.0 / market_prob

    return pd.DataFrame({
        "p_win_corrected": model_pred,
        "tanodds": tanodds,
        "kakuteijyuni": y,
        "surface": surfaces,
        "race_id": race_ids,
    })


class TestBDifferenceIC:
    """Test 1: B-difference IC (RIC-01)."""

    def test_positive_rho_when_model_adds_value(self) -> None:
        model_pred, market_prob, y = _make_arrays()
        result = _compute_b_difference_ic(model_pred, market_prob, y)
        assert "rho" in result
        assert "p_value" in result
        assert "n" in result
        assert result["n"] > 0

    def test_nan_below_threshold(self) -> None:
        model_pred = np.array([0.3, 0.5])
        market_prob = np.array([0.2, 0.4])
        y = np.array([1.0, 0.0])
        result = _compute_b_difference_ic(model_pred, market_prob, y)
        assert np.isnan(result["rho"])
        assert result["n"] == 2


class TestCOrthogonalIC:
    """Test 2: C-orthogonal IC (RIC-02)."""

    def test_positive_rho_with_signal(self) -> None:
        rng = np.random.RandomState(42)
        market_prob = np.clip(rng.beta(2, 5, 200), 0.01, 0.99)
        signal = rng.normal(0, 0.1, 200)
        model_pred = 0.5 * market_prob + 0.5 * signal
        y = (rng.rand(200) < model_pred).astype(float)
        result = _compute_c_orthogonal_ic(model_pred, market_prob, y)
        assert "rho" in result
        assert "n" in result
        assert result["n"] > 0

    def test_nan_below_threshold(self) -> None:
        model_pred = np.array([0.3, 0.5])
        market_prob = np.array([0.2, 0.4])
        y = np.array([1.0, 0.0])
        result = _compute_c_orthogonal_ic(model_pred, market_prob, y)
        assert np.isnan(result["rho"])


class TestEIncrementalIC:
    """Test 3: E-incremental IC (RIC-03)."""

    def test_returns_expected_keys(self) -> None:
        model_pred, market_prob, y = _make_arrays()
        result = _compute_e_incremental_ic(model_pred, market_prob, y)
        assert "ic_model" in result
        assert "ic_market" in result
        assert "delta_ic" in result
        assert "n" in result

    def test_delta_ic_is_difference(self) -> None:
        model_pred, market_prob, y = _make_arrays()
        result = _compute_e_incremental_ic(model_pred, market_prob, y)
        np.testing.assert_almost_equal(
            result["delta_ic"],
            result["ic_model"] - result["ic_market"],
        )


class TestPerRaceIC:
    """Test 4: Per-race IC (RIC-04)."""

    def test_returns_expected_keys(self) -> None:
        rng = np.random.RandomState(42)
        n_races = 20
        records = []
        for i in range(n_races):
            for j in range(8):
                records.append({
                    "race_id": f"R{i:04d}",
                    "pred": rng.rand(),
                    "kakuteijyuni": float(rng.rand() < 0.2),
                })
        df = pd.DataFrame(records)
        result = _compute_per_race_ic(df, "pred", "kakuteijyuni")
        assert "mean_rho" in result
        assert "n_races" in result
        assert "skipped_races" in result
        assert result["n_races"] > 0

    def test_skips_small_races(self) -> None:
        df = pd.DataFrame({
            "race_id": ["R0001"] * 3 + ["R0002"] * 8,
            "pred": np.random.rand(11),
            "kakuteijyuni": np.random.randint(0, 2, 11).astype(float),
        })
        result = _compute_per_race_ic(df, "pred", "kakuteijyuni", min_horses=5)
        assert result["skipped_races"] == 1
        assert result["n_races"] == 1


class TestDirectionConsistency:
    """Test 5: Direction consistency check (RIC-06)."""

    def test_consistent_when_all_positive(self) -> None:
        ic_results = {
            "b_difference": {"rho": 0.05},
            "c_orthogonal": {"rho": 0.03},
            "e_incremental": {"delta_ic": 0.04},
            "per_race": {"mean_rho": 0.02},
        }
        result = _check_direction_consistency(ic_results)
        assert result["consistent"] is True
        assert result["n_metrics_checked"] == 4

    def test_inconsistent_when_mixed_signs(self) -> None:
        ic_results = {
            "b_difference": {"rho": 0.05},
            "c_orthogonal": {"rho": -0.03},
            "e_incremental": {"delta_ic": 0.04},
            "per_race": {"mean_rho": 0.02},
        }
        result = _check_direction_consistency(ic_results)
        assert result["consistent"] is False
        assert "warning" in result


class TestRunICEvaluation:
    """Test 6-9: Full evaluation with synthetic DataFrame."""

    def test_returns_all_ic_metrics_per_surface(self) -> None:
        df = _make_oof_df()
        result = run_ic_evaluation(df)
        for surface_key in ["turf", "dirt", "all"]:
            assert surface_key in result
            sr = result[surface_key]
            if "warning" in sr:
                continue
            for ic_key in ["b_difference", "c_orthogonal", "e_incremental", "per_race"]:
                assert ic_key in sr

    def test_json_output(self, tmp_path: Path) -> None:
        df = _make_oof_df()
        output_path = tmp_path / "ic_baseline.json"
        result = run_ic_evaluation(df, output_path=output_path)
        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["n_total"] == result["n_total"]

    def test_fallback_from_missing_implied_prob(self) -> None:
        df = _make_oof_df()
        assert "implied_prob" not in df.columns
        result = run_ic_evaluation(df)
        assert "all" in result

    def test_min_sample_size_returns_warning(self) -> None:
        df = pd.DataFrame({
            "p_win_corrected": [0.1, 0.2, 0.3],
            "tanodds": [10.0, 5.0, 3.3],
            "kakuteijyuni": [1.0, 0.0, 0.0],
            "surface": ["turf", "dirt", "turf"],
            "race_id": ["R001", "R001", "R002"],
        })
        result = run_ic_evaluation(df)
        assert result["n_total"] == 3
        for surface_key in ["turf", "dirt", "all"]:
            assert "warning" in result[surface_key]


class TestConsoleSummary:
    """Test 10: console_summary logs without error."""

    def test_logs_without_error(self) -> None:
        df = _make_oof_df()
        result = run_ic_evaluation(df)
        with patch("models.ic_evaluator.logger") as mock_logger:
            console_summary(result)
            assert mock_logger.info.called


class TestGetMarketProbability:
    """Test: _get_market_probability fallback."""

    def test_uses_implied_prob_when_present(self) -> None:
        df = pd.DataFrame({
            "implied_prob": [0.1, 0.2, 0.3],
            "tanodds": [10.0, 5.0, 3.3],
        })
        result = _get_market_probability(df)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

    def test_computes_from_tanodds(self) -> None:
        df = pd.DataFrame({"tanodds": [10.0, 5.0, 3.3]})
        result = _get_market_probability(df)
        expected = np.clip(1.0 / np.array([10.0, 5.0, 3.3]), 0.01, 0.99)
        np.testing.assert_array_almost_equal(result, expected)
