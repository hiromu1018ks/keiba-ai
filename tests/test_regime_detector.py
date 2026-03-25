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
    """レジーム検知用のレース集計データ (5行)"""
    return pd.DataFrame(
        {
            "market_error_std": [0.4, 0.2, 0.6, 0.15, 0.5],
            "market_error_mean": [0.1, 0.05, 0.2, 0.03, 0.15],
            "market_entropy_mean": [2.8, 2.2, 3.2, 2.0, 3.0],
            "overround_mean": [0.24, 0.20, 0.28, 0.18, 0.26],
            "favorite_win_rate": [0.22, 0.35, 0.15, 0.40, 0.18],
            "flb_slope": [0.8, 0.3, 1.2, 0.2, 1.0],
            "odds_volatility_mean": [0.15, 0.08, 0.25, 0.05, 0.20],
            "rolling_roi_200": [0.95, 1.10, 0.80, 1.20, 0.85],
            "hit_rate_top3_mean": [0.25, 0.35, 0.15, 0.40, 0.20],
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
        # 3番目の行: entropy高 + fav_rate低 → AGGRESSIVE
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
        detector._transition_hysteresis = 3  # テスト用に短縮

        mock_model = MagicMock()
        # 常に AGGRESSIVE を予測
        mock_model.predict.return_value = np.array([[0.6, 0.3, 0.1]])
        detector.model = mock_model

        # ヒステリシス未満では遷移しない
        for _ in range(2):
            result = detector.detect(regime_stats_df)
            assert result == RegimeState.CONSERVATIVE

        # ヒステリシス到達で遷移
        result = detector.detect(regime_stats_df)
        assert result == RegimeState.CONSERVATIVE  # counter=2, threshold=3
        result = detector.detect(regime_stats_df)
        assert result == RegimeState.AGGRESSIVE  # counter=3, 遷移発生

    def test_get_strategy_params_aggressive(self) -> None:
        detector = RegimeDetector()
        params = detector.get_strategy_params(RegimeState.AGGRESSIVE)
        assert params["ev_threshold"] < 1.20
        assert params["max_bets_per_race"] == 3

    def test_get_strategy_params_conservative(self) -> None:
        detector = RegimeDetector()
        params = detector.get_strategy_params(RegimeState.CONSERVATIVE)
        assert params["ev_threshold"] > 1.20
        assert params["max_bets_per_race"] == 2

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
        detector._current_regime = RegimeState.COLLAPSED
        detector._regime_counter = 100
        assert detector.should_retrain() is True

    def test_min_samples_returns_conservative(self) -> None:
        detector = RegimeDetector()
        small_df = pd.DataFrame(
            {
                "market_error_std": [0.1],
                "market_error_mean": [0.0],
                "market_entropy_mean": [2.0],
                "overround_mean": [0.20],
                "favorite_win_rate": [0.30],
                "flb_slope": [0.3],
                "odds_volatility_mean": [0.05],
                "rolling_roi_200": [1.0],
                "hit_rate_top3_mean": [0.30],
                "field_size_mean": [12],
            }
        )
        result = detector.detect(small_df)
        assert result == RegimeState.CONSERVATIVE

    def test_feature_cols_no_strategy_dependent_in_label(self) -> None:
        """教師ラベルに戦略依存指標を使用しない (Rule 19)"""
        assert "favorite_win_rate" in RegimeDetector.FEATURE_COLS
        assert "flb_slope" in RegimeDetector.FEATURE_COLS
