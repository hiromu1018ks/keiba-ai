"""src/models/stage1_ability_model.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.stage1_ability_model import AbilityModel


@pytest.fixture
def train_df() -> pd.DataFrame:
    """2レース x 4頭の学習データ"""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 4 + ["R2"] * 4,
            "surface": ["turf"] * 4 + ["dirt"] * 4,
            "distance_bin": ["mile"] * 4 + ["sprint"] * 4,
            "track_condition_code": [1] * 8,
            "grade_code": ["_"] * 8,
            "field_size": [4] * 8,
            "weight_diff_from_mean": [0.0, -2.0, 1.0, 5.0, -1.0, 0.0, 2.0, -3.0],
            "difficulty_score": [0.5] * 8,
            # Phase 1: 馬の過去成績
            "norm_finish_logit_avg": rng.uniform(-2, 2, 8),
            "jockey_surprise": rng.uniform(-1, 1, 8),
            "haron_time_zscore_avg": rng.uniform(-3, 3, 8),
            # Phase 1: レース内z-score
            "norm_finish_logit_avg_race_z": rng.uniform(-2, 2, 8),
            "jockey_surprise_race_z": rng.uniform(-2, 2, 8),
            "haron_time_zscore_avg_race_z": rng.uniform(-2, 2, 8),
            # Phase 1: レース内pct
            "norm_finish_logit_avg_race_pct": rng.uniform(0, 1, 8),
            "jockey_surprise_race_pct": rng.uniform(0, 1, 8),
            "haron_time_zscore_avg_race_pct": rng.uniform(0, 1, 8),
            # Phase 2 (4)
            "jockey_cond_wr": rng.uniform(0, 0.3, 8),
            "jockey_cond_wr_race_z": rng.uniform(-2, 2, 8),
            "jockey_cond_wr_race_pct": rng.uniform(0, 1, 8),
            "weight_absolute": rng.uniform(400, 550, 8),
            "distance": [1600] * 4 + [1200] * 4,
            "finish_pos": [1, 2, 3, 4, 1, 3, 2, 4],
        }
    )


@pytest.fixture
def trained_ability_model(train_df: pd.DataFrame) -> AbilityModel:
    """学習済みAbilityModel (mock)"""
    model = AbilityModel()
    mock_turf = MagicMock()
    mock_turf.predict.return_value = np.array([0.8, 0.5, 0.3, 0.1])
    mock_dirt = MagicMock()
    mock_dirt.predict.return_value = np.array([0.7, 0.3, 0.5, 0.1])
    model.models = {"turf": mock_turf, "dirt": mock_dirt}
    return model


class TestAbilityModelFeatures:
    def test_no_odds_features(self) -> None:
        """オッズ特徴量が含まれない (Rule 1)"""
        odds_features = [
            "win_odds",
            "tan_odds",
            "fuku_odds",
            "odds_drop_rate_60_10",
            "odds_drop_rate_30_10",
            "odds_velocity",
            "odds_volatility",
            "popularity_change_30_10",
            "p_market_win_adj",
            "market_entropy",
            "overround",
            "popularity_rank",
        ]
        for f in odds_features:
            assert f not in AbilityModel.FEATURE_COLS, f"{f} should not be in Stage1 features"

    def test_has_required_features(self) -> None:
        assert "surface" in AbilityModel.FEATURE_COLS
        assert "distance_bin" in AbilityModel.FEATURE_COLS
        assert "field_size" in AbilityModel.FEATURE_COLS


class TestAbilityModelPredict:
    def test_add_ability_probs_returns_probs(
        self,
        trained_ability_model: AbilityModel,
        train_df: pd.DataFrame,
    ) -> None:
        """add_ability_probs が確率列を追加する"""
        result = trained_ability_model.add_ability_probs(train_df)
        assert "p_ability_win" in result.columns
        # p_ability_place は PlaceAbilityModel が担当 (AbilityModel では出力しない)
        assert "p_ability_place" not in result.columns

    def test_probs_sum_to_one_within_race(
        self,
        trained_ability_model: AbilityModel,
        train_df: pd.DataFrame,
    ) -> None:
        """レース内で確率の合計が1に近い"""
        result = trained_ability_model.add_ability_probs(train_df)
        for race_id in result["race_id"].unique():
            race_probs = result[result["race_id"] == race_id]["p_ability_win"]
            assert abs(race_probs.sum() - 1.0) < 0.05

    def test_probs_are_positive(
        self,
        trained_ability_model: AbilityModel,
        train_df: pd.DataFrame,
    ) -> None:
        result = trained_ability_model.add_ability_probs(train_df)
        assert (result["p_ability_win"] > 0).all()

    def test_returns_new_dataframe(
        self,
        trained_ability_model: AbilityModel,
        train_df: pd.DataFrame,
    ) -> None:
        result = trained_ability_model.add_ability_probs(train_df)
        assert result is not train_df
