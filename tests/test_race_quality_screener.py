"""src/models/race_quality_screener.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.race_quality_screener import RaceQualityScreener


@pytest.fixture
def race_features_df() -> pd.DataFrame:
    """レースレベル特徴量のテストデータ (4レース)"""
    return pd.DataFrame(
        {
            "market_log_error_max_abs": [0.5, 0.2, 0.8, 0.1],
            "market_log_error_std": [0.3, 0.1, 0.5, 0.05],
            "market_log_error_top_q75": [0.4, 0.15, 0.6, 0.08],
            "n_positive_errors": [3, 1, 5, 0],
            "top_k_error_sum": [1.0, 0.3, 2.0, 0.1],
            "positive_error_ratio": [0.375, 0.125, 0.625, 0.0],
            "hist_hit_rate_topk": [0.25, 0.15, 0.30, 0.10],
            "hist_roi_topk": [1.5, 0.8, 2.0, 0.5],
            "hist_positive_return_ratio": [0.6, 0.4, 0.8, 0.2],
            "market_entropy": [2.8, 2.2, 3.0, 1.8],
            "overround": [0.22, 0.20, 0.25, 0.18],
            "overround_deviation": [0.02, 0.0, 0.05, -0.02],
            "field_size": [8, 12, 16, 10],
            "surface": ["turf", "dirt", "turf", "dirt"],
            "distance_bin": ["mile", "sprint", "long", "mile"],
            "track_condition_code": [1, 2, 3, 1],
            "grade_code": ["_", "_", "C", "_"],
            "difficulty_score": [0.5, 0.4, 0.7, 0.3],
            "hist_win_rate_same_condition": [0.20, 0.15, 0.25, 0.10],
            "hist_market_entropy_avg": [2.7, 2.1, 2.9, 1.9],
        }
    )


class TestRaceQualityScreener:
    def test_target_uses_result_based_proxy(
        self,
        race_features_df: pd.DataFrame,
    ) -> None:
        """_build_target が結果ベースproxyを使用する (Rule 16)"""
        screener = RaceQualityScreener()
        target = screener._build_target(race_features_df)
        assert len(target) == len(race_features_df)
        # hist_roi_topk が高いレースほど target が高い
        assert target.iloc[2] > target.iloc[1]  # ROI 2.0 vs 0.8

    def test_target_does_not_use_ev_dependent_features(self) -> None:
        """FEATURE_COLS にEV依存特徴量が含まれない (Rule 16)"""
        ev_dependent = ["ev_win", "ev_place", "p_win_pred", "edge", "actual_bet_roi"]
        for f in ev_dependent:
            assert f not in RaceQualityScreener.FEATURE_COLS

    def test_should_bet_returns_bool(
        self,
        race_features_df: pd.DataFrame,
    ) -> None:
        screener = RaceQualityScreener()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.3, 0.8, 0.1])
        screener.model = mock_model
        screener.threshold = 0.4

        features = race_features_df.iloc[0].to_dict()
        result = screener.should_bet(features)
        assert isinstance(result, bool)

    def test_should_bet_above_threshold(
        self,
        race_features_df: pd.DataFrame,
    ) -> None:
        screener = RaceQualityScreener()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])  # 高スコア
        screener.model = mock_model
        screener.threshold = 0.4

        features = race_features_df.iloc[0].to_dict()
        assert screener.should_bet(features) is True

    def test_should_bet_below_threshold(
        self,
        race_features_df: pd.DataFrame,
    ) -> None:
        screener = RaceQualityScreener()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1])  # 低スコア
        screener.model = mock_model
        screener.threshold = 0.4

        features = race_features_df.iloc[0].to_dict()
        assert screener.should_bet(features) is False

    def test_has_distribution_features(self) -> None:
        """分布特徴量が含まれる (v5.1追加)"""
        assert "n_positive_errors" in RaceQualityScreener.FEATURE_COLS
        assert "top_k_error_sum" in RaceQualityScreener.FEATURE_COLS
        assert "positive_error_ratio" in RaceQualityScreener.FEATURE_COLS

    def test_has_result_based_profit_proxy(self) -> None:
        """結果ベース利益proxyが含まれる (v5.4追加, Rule 16)"""
        assert "hist_hit_rate_topk" in RaceQualityScreener.FEATURE_COLS
        assert "hist_roi_topk" in RaceQualityScreener.FEATURE_COLS
        assert "hist_positive_return_ratio" in RaceQualityScreener.FEATURE_COLS
