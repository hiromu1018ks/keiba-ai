"""src/models/wide_two_stage_model.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.wide_two_stage_model import WideTwoStageModel


@pytest.fixture
def pair_df() -> pd.DataFrame:
    """ワイド馬券ペアのテストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 3,
            "umaban_a": [1, 1, 2],
            "umaban_b": [2, 3, 3],
            "popularity_sum": [3, 5, 9],
            "running_style_combo": [3, 0, 5],
            "p_hit": [0.30, 0.10, 0.05],
            "e_return_given_hit": [4.0, 12.0, 24.0],
            "surface": ["turf"] * 3,
            "distance_bin": ["mile"] * 3,
            "track_condition_code": [1] * 3,
            "grade_code": ["_"] * 3,
            "field_size": [3] * 3,
        }
    )


@pytest.fixture
def trained_wide_model(pair_df: pd.DataFrame) -> WideTwoStageModel:
    """学習済みWideTwoStageModel (mock)"""
    model = WideTwoStageModel()
    mock_hit = MagicMock()
    mock_hit.predict.return_value = np.array([0.30, 0.10, 0.05])
    mock_return = MagicMock()
    mock_return.predict.return_value = np.array([4.0, 12.0, 24.0])
    model.hit_model = mock_hit
    model.return_model = mock_return
    return model


class TestWideTwoStageModel:
    def test_score_is_ev_over_e_sqrt_p(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        """score = EV / (E * sqrt(P)) (Rule 3, Rule 15)"""
        result = trained_wide_model.predict_score(pair_df)
        for i in range(len(result)):
            ev = result["ev_wide"].iloc[i]
            e = result["e_return_given_hit"].iloc[i]
            p = result["p_hit"].iloc[i]
            expected_score = ev / (e * np.sqrt(max(p, 0.001)))
            assert abs(result["wide_score_adj"].iloc[i] - expected_score) < 1e-10

    def test_ev_equals_p_times_e(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        result = trained_wide_model.predict_score(pair_df)
        expected_ev = result["p_hit"] * result["e_return_given_hit"]
        assert np.allclose(result["ev_wide"].values, expected_ev.values)

    def test_select_bets_dual_filter(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        """2段階フィルタ: ev_threshold + score_threshold"""
        bets = trained_wide_model.select_bets(
            pair_df,
            ev_threshold=1.20,
            score_threshold=0.015,
            max_bets=3,
        )
        for bet in bets:
            assert bet["ev_wide"] >= 1.20
            assert bet["wide_score_adj"] >= 0.015

    def test_select_bets_max_bets(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        """max_bets で返す数が制限される"""
        bets = trained_wide_model.select_bets(pair_df, max_bets=2)
        assert len(bets) <= 2

    def test_select_bets_excludes_zero_style(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        """running_style_combo == 0 のペアは除外される"""
        bets = trained_wide_model.select_bets(pair_df, ev_threshold=0.0, score_threshold=0.0)
        for bet in bets:
            assert bet["running_style_combo"] != 0

    def test_select_bets_popularity_filter(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        """popularity_sum >= 6 のペアのみ"""
        bets = trained_wide_model.select_bets(pair_df, ev_threshold=0.0, score_threshold=0.0)
        for bet in bets:
            assert bet["popularity_sum"] >= 6

    def test_sharpe_ratio_approximation(self) -> None:
        """異なるP/Eペアでスコアがシャープレシオ的になることを確認"""
        # ペアA: P=0.30, E=4.0 -> score = 1.20 / (4.0 * sqrt(0.30)) = 0.548
        # ペアB: P=0.05, E=24.0 -> score = 1.20 / (24.0 * sqrt(0.05)) = 0.224
        p_a, e_a = 0.30, 4.0
        p_b, e_b = 0.05, 24.0
        score_a = (p_a * e_a) / (e_a * np.sqrt(p_a))
        score_b = (p_b * e_b) / (e_b * np.sqrt(p_b))
        # 高P低E (A) の方が低P高E (B) よりスコアが高い
        assert score_a > score_b
