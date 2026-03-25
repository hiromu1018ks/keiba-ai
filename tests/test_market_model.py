"""src/models/market_model.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.market_model import MarketModel


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """4頭立てのテストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "surface": ["turf"] * 4,
            "distance_bin": ["mile"] * 4,
            "track_condition_code": [1] * 4,
            "grade_code": ["_"] * 4,
            "field_size": [4] * 4,
            "weight_diff_from_mean": [0.0, -2.0, 1.0, 5.0],
            "difficulty_score": [0.5, 0.5, 0.5, 0.5],
            "p_market_win_adj": [0.40, 0.30, 0.20, 0.10],
        }
    )


@pytest.fixture
def trained_market_model(sample_df: pd.DataFrame) -> MarketModel:
    """学習済みMarketModel (mock)"""
    model = MarketModel()
    mock_lgb = MagicMock()
    # 予測値: 市場確率に近いが少しズレた値
    mock_lgb.predict.return_value = np.array([0.35, 0.28, 0.22, 0.15])
    model.model = mock_lgb
    return model


class TestMarketModelPredict:
    def test_log_error_computation(
        self, trained_market_model: MarketModel, sample_df: pd.DataFrame
    ) -> None:
        """log(p_market / p_pred) が正しく計算される"""
        result = trained_market_model.predict_and_calc_error(sample_df)
        # log(0.40 / 0.35) ≈ 0.1335
        expected = np.log(0.40 / 0.35)
        assert abs(result["market_log_error_win"].iloc[0] - expected) < 1e-10

    def test_signed_and_abs_log_error(
        self, trained_market_model: MarketModel, sample_df: pd.DataFrame
    ) -> None:
        """signed_log_error_win と abs_log_error_win が正しく分離される"""
        result = trained_market_model.predict_and_calc_error(sample_df)
        for i in range(len(result)):
            signed = result["signed_log_error_win"].iloc[i]
            log_err = result["market_log_error_win"].iloc[i]
            assert signed == log_err
            assert result["abs_log_error_win"].iloc[i] == abs(log_err)

    def test_p_market_pred_dropped(
        self, trained_market_model: MarketModel, sample_df: pd.DataFrame
    ) -> None:
        """p_market_pred_win は出力から削除される (Rule 11)"""
        result = trained_market_model.predict_and_calc_error(sample_df)
        assert "_p_market_pred_win" not in result.columns
        assert "p_market_pred_win" not in result.columns

    def test_market_error_rank_in_race(
        self, trained_market_model: MarketModel, sample_df: pd.DataFrame
    ) -> None:
        """レース内相対ランクが正しく計算される"""
        result = trained_market_model.predict_and_calc_error(sample_df)
        ranks = result["market_error_rank_in_race"].values
        assert sorted(ranks) == [1, 2, 3, 4]

    def test_raw_error_preserved(
        self, trained_market_model: MarketModel, sample_df: pd.DataFrame
    ) -> None:
        """生の差分 market_pred_error_win も保持される"""
        result = trained_market_model.predict_and_calc_error(sample_df)
        expected = 0.40 - 0.35
        assert abs(result["market_pred_error_win"].iloc[0] - expected) < 1e-10

    def test_clipping_prevents_divergence(self) -> None:
        """極端な p_pred がクリップされる (Rule 13)"""
        model = MarketModel()
        mock_lgb = MagicMock()
        mock_lgb.predict.return_value = np.array([0.001, 0.999, 0.5, 0.5])
        model.model = mock_lgb

        df = pd.DataFrame(
            {
                "race_id": ["R1"] * 4,
                "surface": ["turf"] * 4,
                "distance_bin": ["mile"] * 4,
                "track_condition_code": [1] * 4,
                "grade_code": ["_"] * 4,
                "field_size": [4] * 4,
                "weight_diff_from_mean": [0.0, 0.0, 0.0, 0.0],
                "difficulty_score": [0.5, 0.5, 0.5, 0.5],
                "p_market_win_adj": [0.40, 0.10, 0.25, 0.25],
            }
        )
        result = model.predict_and_calc_error(df)
        # クリップ後のlog_errorは発散しない
        assert result["market_log_error_win"].notna().all()
        assert np.isfinite(result["market_log_error_win"].values).all()


class TestMarketModelStage2Features:
    def test_no_p_market_pred_in_features(self) -> None:
        """get_stage2_features() に p_market_pred は含まれない (Rule 11)"""
        model = MarketModel()
        features = model.get_stage2_features()
        assert "p_market_pred_win" not in features

    def test_returns_expected_features(self) -> None:
        model = MarketModel()
        features = model.get_stage2_features()
        assert "signed_log_error_win" in features
        assert "abs_log_error_win" in features
        assert "market_error_rank_in_race" in features
