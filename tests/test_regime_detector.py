"""src/models/regime_detector.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from domain.models import RegimeConfig
from domain.types import RegimeState
from models.regime_detector import RegimeDetector


@pytest.fixture
def regime_stats_df() -> pd.DataFrame:
    """レジーム検知用のレース集計データ (5行) — 新 FEATURE_COLS"""
    return pd.DataFrame(
        {
            "market_error_std": [0.4, 0.2, 0.6, 0.15, 0.5],
            "market_error_mean": [0.1, 0.05, 0.2, 0.03, 0.15],
            "overround_rolling": [0.24, 0.20, 0.28, 0.18, 0.26],
            "entropy_rolling": [2.8, 2.2, 3.2, 2.0, 3.0],
            "favorite_implied_prob_rolling": [0.35, 0.42, 0.28, 0.50, 0.30],
            "odds_skewness_rolling": [0.8, 0.3, 1.2, 0.2, 1.0],
            "odds_volatility_mean": [0.15, 0.08, 0.25, 0.05, 0.20],
            "field_size_mean": [14, 12, 16, 10, 15],
        }
    )


class TestRegimeDetector:
    def test_initial_state_is_conservative(self) -> None:
        detector = RegimeDetector()
        assert detector.current_regime == RegimeState.CONSERVATIVE

    def test_detect_returns_regime_state(
        self,
        regime_stats_df: pd.DataFrame,
    ) -> None:
        cfg = RegimeConfig(min_samples=5)
        detector = RegimeDetector(cfg=cfg)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.6, 0.3, 0.1]])
        detector.model = mock_model

        result = detector.detect(regime_stats_df)
        assert isinstance(result, RegimeState)

    def test_hysteresis_prevents_frequent_switching(
        self,
        regime_stats_df: pd.DataFrame,
    ) -> None:
        """ヒステリシス: 連続N回同じ状態で初めて遷移"""
        cfg = RegimeConfig(min_samples=5)
        detector = RegimeDetector(cfg=cfg)
        detector._transition_hysteresis = 3

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.6, 0.3, 0.1]])
        detector.model = mock_model

        for _ in range(2):
            result = detector.detect(regime_stats_df)
            assert result == RegimeState.CONSERVATIVE

        result = detector.detect(regime_stats_df)
        assert result == RegimeState.CONSERVATIVE  # counter=2, threshold=3
        result = detector.detect(regime_stats_df)
        assert result == RegimeState.AGGRESSIVE  # counter=3, 遷移発生

    def test_get_strategy_params_aggressive(self) -> None:
        detector = RegimeDetector()
        params = detector.get_strategy_params(RegimeState.AGGRESSIVE)
        assert params["ev_threshold"] < 1.20
        assert params["max_bets_per_race"] == 1
        assert params["quality_second_margin"] > params["soft_gate_second_margin"]
        assert params["runner_up_rescue_min_prob"] >= 0.25
        assert params["runner_up_rerank_market_condition_max"] <= 0.20
        assert params["runner_up_rerank_entropy_min"] < params["runner_up_rerank_entropy_max"]

    def test_get_strategy_params_conservative(self) -> None:
        detector = RegimeDetector()
        params = detector.get_strategy_params(RegimeState.CONSERVATIVE)
        assert params["ev_threshold"] > 1.20
        assert params["max_bets_per_race"] == 1

    def test_get_strategy_params_collapsed(self) -> None:
        detector = RegimeDetector()
        params = detector.get_strategy_params(RegimeState.COLLAPSED)
        assert params["ev_threshold"] >= 1.50
        assert params["max_bets_per_race"] == 1

    def test_should_retrain_false_by_default(self) -> None:
        detector = RegimeDetector()
        assert detector.should_retrain() is False

    def test_should_retrain_after_consecutive_collapsed(self) -> None:
        detector = RegimeDetector()
        detector._collapsed_consecutive = 100
        assert detector.should_retrain() is True

    def test_min_samples_returns_conservative(self) -> None:
        detector = RegimeDetector()
        small_df = pd.DataFrame(
            {
                "market_error_std": [0.1],
                "market_error_mean": [0.0],
                "overround_rolling": [0.20],
                "entropy_rolling": [2.0],
                "favorite_implied_prob_rolling": [0.30],
                "odds_skewness_rolling": [0.3],
                "odds_volatility_mean": [0.05],
                "field_size_mean": [12],
            }
        )
        result = detector.detect(small_df)
        assert result == RegimeState.CONSERVATIVE

    def test_feature_cols_contain_only_pre_race_indicators(self) -> None:
        """FEATURE_COLS に結果依存 (POST_RACE) 指標が含まれないことを確認"""
        assert "favorite_win_rate" not in RegimeDetector.FEATURE_COLS
        assert "flb_slope" not in RegimeDetector.FEATURE_COLS
        assert "favorite_roi_ema" not in RegimeDetector.FEATURE_COLS
        assert "mid_roi_ema" not in RegimeDetector.FEATURE_COLS
        assert "longshot_roi_ema" not in RegimeDetector.FEATURE_COLS
        # PRE_RACE 指標が含まれる
        assert "overround_rolling" in RegimeDetector.FEATURE_COLS
        assert "entropy_rolling" in RegimeDetector.FEATURE_COLS
        assert "favorite_implied_prob_rolling" in RegimeDetector.FEATURE_COLS
        assert "odds_skewness_rolling" in RegimeDetector.FEATURE_COLS
        assert "odds_volatility_mean" in RegimeDetector.FEATURE_COLS
        assert "market_error_std" in RegimeDetector.FEATURE_COLS
        assert "market_error_mean" in RegimeDetector.FEATURE_COLS
        assert "field_size_mean" in RegimeDetector.FEATURE_COLS

    def test_get_strategy_params_contains_edge_threshold(self) -> None:
        """RegimeDetector should return edge_threshold for Value Betting."""
        detector = RegimeDetector()
        for regime in [RegimeState.AGGRESSIVE, RegimeState.CONSERVATIVE, RegimeState.COLLAPSED]:
            params = detector.get_strategy_params(regime)
            assert "edge_threshold" in params
            assert isinstance(params["edge_threshold"], float)
            assert params["edge_threshold"] > 0

    def test_edge_threshold_values_by_regime(self) -> None:
        """Edge thresholds should be non-decreasing from AGGRESSIVE to COLLAPSED."""
        detector = RegimeDetector()
        agg = detector.get_strategy_params(RegimeState.AGGRESSIVE)
        con = detector.get_strategy_params(RegimeState.CONSERVATIVE)
        col = detector.get_strategy_params(RegimeState.COLLAPSED)
        assert agg["edge_threshold"] <= con["edge_threshold"] <= col["edge_threshold"]

    def test_collapsed_strategy_has_skip_true(self) -> None:
        """Test 11: COLLAPSED params に skip=True が含まれる (D-11)"""
        detector = RegimeDetector()
        params = detector.get_strategy_params(RegimeState.COLLAPSED)
        assert params.get("skip") is True

    def test_aggressive_strategy_no_skip(self) -> None:
        """Test 12: AGGRESSIVE params に skip キーがない (または False)"""
        detector = RegimeDetector()
        params = detector.get_strategy_params(RegimeState.AGGRESSIVE)
        assert params.get("skip") is not True

    def test_override_params_injects_values(self) -> None:
        """override_params で主要3パラメータが上書きされる"""
        detector = RegimeDetector(
            override_params={
                "aggressive": {"fractional_kelly": 0.8, "ev_threshold": 1.5, "edge_threshold": 0.12},
                "conservative": {"fractional_kelly": 0.3},
            }
        )
        agg = detector.get_strategy_params(RegimeState.AGGRESSIVE)
        assert agg["fractional_kelly"] == 0.8
        assert agg["ev_threshold"] == 1.5
        assert agg["edge_threshold"] == 0.12

        con = detector.get_strategy_params(RegimeState.CONSERVATIVE)
        assert con["fractional_kelly"] == 0.3

    def test_override_params_does_not_affect_unoverridden_regime(self) -> None:
        """override_params がないレジームはデフォルト値のまま"""
        detector = RegimeDetector(
            override_params={"aggressive": {"fractional_kelly": 0.8}}
        )
        col = detector.get_strategy_params(RegimeState.COLLAPSED)
        detector_default = RegimeDetector()
        col_default = detector_default.get_strategy_params(RegimeState.COLLAPSED)
        assert col["fractional_kelly"] == col_default["fractional_kelly"]

    def test_train_uses_pre_race_features_for_labels(self) -> None:
        """train() の教師ラベルが PRE_RACE 指標のみで計算される"""
        detector = RegimeDetector()
        np.random.seed(42)
        n = 200
        df_race = pd.DataFrame(
            {
                "market_error_std": np.random.uniform(0.1, 0.5, n),
                "market_error_mean": np.random.uniform(0.0, 0.2, n),
                "overround_rolling": np.random.uniform(0.15, 0.30, n),
                "entropy_rolling": np.random.uniform(1.5, 3.5, n),
                "favorite_implied_prob_rolling": np.random.uniform(0.20, 0.50, n),
                "odds_skewness_rolling": np.random.uniform(0.1, 1.5, n),
                "odds_volatility_mean": np.random.uniform(0.05, 0.25, n),
                "field_size_mean": np.random.choice([10, 12, 14, 16], n),
            }
        )
        detector.train(df_race, num_threads=1)
        assert hasattr(detector, "model")
