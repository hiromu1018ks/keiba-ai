"""src/models/two_stage_return_model.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """WinTwoStageModel FEATURE_COLS を満たす8頭テストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 8,
            "p_ability_win": [0.30, 0.25, 0.20, 0.10, 0.08, 0.04, 0.02, 0.01],
            "signed_log_error_win": [0.1, -0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.3],
            "abs_log_error_win": [0.1, 0.1, 0.2, 0.3, 0.0, 0.5, 0.2, 0.3],
            "odds_drop_rate_60_10": [0.0, -0.1, 0.1, -0.2, 0.0, 0.1, -0.1, 0.0],
            "odds_drop_rate_30_10": [0.0, -0.05, 0.05, -0.1, 0.0, 0.05, -0.05, 0.0],
            "odds_velocity": [0.0, -0.02, 0.02, -0.03, 0.0, 0.01, -0.01, 0.0],
            "odds_volatility": [0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01, 0.01],
            "popularity_change_30_10": [0, -1, 1, -2, 0, 1, -1, 0],
            "market_entropy": [2.5] * 8,
            "popularity_rank": [1, 2, 3, 4, 5, 6, 7, 8],
            "overround": [0.22] * 8,
            "surface": ["turf"] * 8,
            "distance_bin": ["mile"] * 8,
            "track_condition_code": [1] * 8,
            "grade_code": ["_"] * 8,
            "field_size": [8] * 8,
            "finish_pos": [1, 2, 3, 4, 5, 6, 7, 8],
            "win_odds_actual": [3.5, 5.0, 8.0, 15.0, 25.0, 40.0, 80.0, 150.0],
            "place_odds_actual": [1.3, 1.6, 2.1, 3.5, 5.0, 8.0, 15.0, 30.0],
        }
    )


@pytest.fixture
def trained_win_model(feature_df: pd.DataFrame) -> WinTwoStageModel:
    """学習済みWinTwoStageModel (mock)"""
    model = WinTwoStageModel()
    mock_hit = MagicMock()
    mock_hit.predict.return_value = np.array(
        [
            0.28,
            0.22,
            0.18,
            0.12,
            0.08,
            0.06,
            0.04,
            0.02,
        ]
    )
    mock_return = MagicMock()
    mock_return.predict.return_value = np.array(
        [
            4.0,
            6.0,
            9.0,
            16.0,
            28.0,
            45.0,
            90.0,
            160.0,
        ]
    )
    model.hit_model = mock_hit
    model.return_model = mock_return
    return model


class TestWinTwoStageModel:
    def test_ev_equals_p_times_e(
        self,
        trained_win_model: WinTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        """EV = P(win) × E(win_odds | win)"""
        result = trained_win_model.predict_ev(feature_df)
        expected_ev = result["p_win_pred"] * result["e_return_win_pred"]
        assert np.allclose(result["ev_win"].values, expected_ev.values)

    def test_output_columns(
        self,
        trained_win_model: WinTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        result = trained_win_model.predict_ev(feature_df)
        assert "p_win_pred" in result.columns
        assert "e_return_win_pred" in result.columns
        assert "ev_win" in result.columns

    def test_stage_b_no_zeros_in_label(self, feature_df: pd.DataFrame) -> None:
        """Stage B の学習データにゼロが含まれないことを確認するテスト"""
        hit_df = feature_df[feature_df["finish_pos"] == 1]
        assert (hit_df["win_odds_actual"] > 0).all()

    def test_feature_cols_no_p_market_pred(self) -> None:
        """p_market_pred が特徴量に含まれない (Rule 11)"""
        assert "p_market_pred_win" not in WinTwoStageModel.FEATURE_COLS


class TestPlaceTwoStageModel:
    @pytest.fixture
    def trained_place_model(self, feature_df: pd.DataFrame) -> PlaceTwoStageModel:
        model = PlaceTwoStageModel()
        mock_hit = MagicMock()
        mock_hit.predict.return_value = np.array(
            [
                0.40,
                0.35,
                0.30,
                0.15,
                0.10,
                0.05,
                0.03,
                0.01,
            ]
        )
        mock_return = MagicMock()
        mock_return.predict.return_value = np.array(
            [
                1.4,
                1.7,
                2.2,
                3.8,
                5.5,
                9.0,
                16.0,
                32.0,
            ]
        )
        model.hit_model = mock_hit
        model.return_model = mock_return
        return model

    def test_ev_equals_p_times_e(
        self,
        trained_place_model: PlaceTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        result = trained_place_model.predict_ev(feature_df)
        expected_ev = result["p_place_pred"] * result["e_return_place_pred"]
        assert np.allclose(result["ev_place"].values, expected_ev.values)

    def test_output_columns(
        self,
        trained_place_model: PlaceTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        result = trained_place_model.predict_ev(feature_df)
        assert "p_place_pred" in result.columns
        assert "e_return_place_pred" in result.columns
        assert "ev_place" in result.columns

    def test_shared_feature_cols(self) -> None:
        assert PlaceTwoStageModel.FEATURE_COLS == WinTwoStageModel.FEATURE_COLS
