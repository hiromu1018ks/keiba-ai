"""src/models/ev_correction_model.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.ev_correction_model import EVCorrectionModel


@pytest.fixture
def pre_ev_df() -> pd.DataFrame:
    """WinTwoStageModel.predict_ev() 出力後のテストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 8,
            "finish_pos": [1, 2, 3, 4, 5, 6, 7, 8],
            "win_odds_actual": [4.0, 6.0, 9.0, 16.0, 28.0, 45.0, 90.0, 160.0],
            "p_win_pred": [0.28, 0.22, 0.18, 0.12, 0.08, 0.06, 0.04, 0.02],
            "e_return_win_pred": [4.0, 6.0, 9.0, 16.0, 28.0, 45.0, 90.0, 160.0],
            "ev_win": [1.12, 1.32, 1.62, 1.92, 2.24, 2.70, 3.60, 3.20],
            "signed_log_error_win": [0.1, -0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.3],
            "abs_log_error_win": [0.1, 0.1, 0.2, 0.3, 0.0, 0.5, 0.2, 0.3],
            "market_entropy": [2.5] * 8,
            "popularity_rank": [1, 2, 3, 4, 5, 6, 7, 8],
            "surface": ["turf"] * 8,
            "distance_bin": ["mile"] * 8,
            "track_condition_code": [1] * 8,
            "field_size": [8] * 8,
        }
    )


@pytest.fixture
def trained_ev_model(pre_ev_df: pd.DataFrame) -> EVCorrectionModel:
    """学習済みEVCorrectionModel (mock)"""
    model = EVCorrectionModel()
    # P補正: 小さい補正値 (correction_logit)
    mock_p = MagicMock()
    mock_p.predict.return_value = np.array(
        [
            0.01,
            -0.01,
            0.02,
            -0.02,
            0.0,
            0.03,
            -0.03,
            0.01,
        ]
    )
    # E補正: log residual
    mock_e = MagicMock()
    mock_e.predict.return_value = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    model.p_correction_model = mock_p
    model.e_correction_model = mock_e
    return model


class TestEVCorrectionModel:
    def test_p_corrected_in_0_1(
        self,
        trained_ev_model: EVCorrectionModel,
        pre_ev_df: pd.DataFrame,
    ) -> None:
        """P_corrected が [0, 1] に制約される"""
        result = trained_ev_model.correct_ev(pre_ev_df)
        assert (result["p_win_corrected"] >= 0).all()
        assert (result["p_win_corrected"] <= 1).all()

    def test_ev_corrected_equals_p_times_e(
        self,
        trained_ev_model: EVCorrectionModel,
        pre_ev_df: pd.DataFrame,
    ) -> None:
        """EV_corrected = P_corrected × E_corrected"""
        result = trained_ev_model.correct_ev(pre_ev_df)
        expected = result["p_win_corrected"] * result["e_return_win_corrected"]
        assert np.allclose(result["ev_win_corrected"].values, expected.values, atol=1e-10)

    def test_e_corrected_positive(
        self,
        trained_ev_model: EVCorrectionModel,
        pre_ev_df: pd.DataFrame,
    ) -> None:
        """E_corrected は正値 (オッズは1.0以上)"""
        result = trained_ev_model.correct_ev(pre_ev_df)
        assert (result["e_return_win_corrected"] > 0).all()

    def test_feature_cols_no_p_win_pred(self) -> None:
        """p_win_pred は特徴量から除外される (init_scoreで代替)"""
        assert "p_win_pred" not in EVCorrectionModel.FEATURE_COLS

    def test_has_interaction_features(self) -> None:
        """交互作用特徴量が含まれる"""
        assert "p_x_e_interaction" in EVCorrectionModel.FEATURE_COLS
        assert "p_minus_e_gap" in EVCorrectionModel.FEATURE_COLS

    def test_train_asserts_ev_win(self, pre_ev_df: pd.DataFrame) -> None:
        """train() は ev_win 列を要求する"""
        model = EVCorrectionModel()
        bad_df = pre_ev_df.drop(columns=["ev_win"])
        with pytest.raises(AssertionError, match="ev_win"):
            model.train(bad_df)

    def test_e_clip_floor(self) -> None:
        assert EVCorrectionModel.E_CLIP_FLOOR == 1.0
