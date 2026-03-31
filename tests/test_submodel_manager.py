"""src/models/submodel_manager.py のテスト"""

from __future__ import annotations

import pandas as pd
import pytest

from domain.models import Race
from models.submodel_manager import SubModelManager


@pytest.fixture
def turf_race() -> Race:
    return Race(
        year=2024,
        month_day="0324",
        jyo_cd="05",
        kaiji="03",
        nichiji="02",
        race_num="08",
        track_cd=11,
        distance=1600,
        tenko_cd=1,
        baba_cd=1,
        syubetu_cd="13",
        jyoken_cd="999",
        grade_cd="_",
        field_size=18,
    )


@pytest.fixture
def dirt_race() -> Race:
    return Race(
        year=2024,
        month_day="0324",
        jyo_cd="05",
        kaiji="03",
        nichiji="02",
        race_num="08",
        track_cd=23,
        distance=1400,
        tenko_cd=1,
        baba_cd=1,
        syubetu_cd="13",
        jyoken_cd="999",
        grade_cd="_",
        field_size=16,
    )


class TestSubModelManager:
    def test_get_key_turf(self, turf_race: Race) -> None:
        mgr = SubModelManager()
        assert mgr.get_key(turf_race) == "turf"

    def test_get_key_dirt(self, dirt_race: Race) -> None:
        mgr = SubModelManager()
        assert mgr.get_key(dirt_race) == "dirt"

    def test_valid_keys(self) -> None:
        assert SubModelManager.VALID_KEYS == ["turf", "dirt"]

    def test_min_samples(self) -> None:
        assert SubModelManager.MIN_SAMPLES == 20_000

    def test_should_split_further_enough_samples(self) -> None:
        mgr = SubModelManager()
        assert mgr.should_split_further("turf", "sprint", 25_000) is True

    def test_should_split_further_insufficient_samples(self) -> None:
        mgr = SubModelManager()
        assert mgr.should_split_further("dirt", "sprint", 15_000) is False


class TestAddDistanceBandFeatures:
    @pytest.fixture
    def mixed_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "surface": ["turf", "turf", "dirt", "dirt", "turf"],
                "kyori": [1200, 1600, 1400, 1800, 2200],
                "track_condition_code": [1, 3, 2, 4, 1],
            }
        )

    def test_turf_sprint(self, mixed_df: pd.DataFrame) -> None:
        mgr = SubModelManager()
        result = mgr.add_distance_band_features(mixed_df)
        assert result["is_turf_sprint"].iloc[0] == 1
        assert result["is_turf_sprint"].iloc[1] == 0
        assert result["is_dirt_sprint"].iloc[2] == 1

    def test_turf_mile(self, mixed_df: pd.DataFrame) -> None:
        mgr = SubModelManager()
        result = mgr.add_distance_band_features(mixed_df)
        assert result["is_turf_mile"].iloc[1] == 1

    def test_turf_long(self, mixed_df: pd.DataFrame) -> None:
        mgr = SubModelManager()
        result = mgr.add_distance_band_features(mixed_df)
        assert result["is_turf_long"].iloc[4] == 1

    def test_dirt_intermediate(self, mixed_df: pd.DataFrame) -> None:
        mgr = SubModelManager()
        result = mgr.add_distance_band_features(mixed_df)
        assert result["is_dirt_intermediate"].iloc[3] == 1

    def test_good_soft_track(self, mixed_df: pd.DataFrame) -> None:
        mgr = SubModelManager()
        result = mgr.add_distance_band_features(mixed_df)
        assert result["is_good_track"].iloc[0] == 1
        assert result["is_good_track"].iloc[1] == 0
        assert result["is_soft_track"].iloc[1] == 1

    def test_returns_new_dataframe(self, mixed_df: pd.DataFrame) -> None:
        mgr = SubModelManager()
        result = mgr.add_distance_band_features(mixed_df)
        assert result is not mixed_df

    def test_preserves_existing_columns(self, mixed_df: pd.DataFrame) -> None:
        mgr = SubModelManager()
        result = mgr.add_distance_band_features(mixed_df)
        original_cols = ["surface", "kyori", "track_condition_code"]
        assert list(mixed_df.columns) == original_cols
        assert all(c in result.columns for c in original_cols)
