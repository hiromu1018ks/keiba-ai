"""test_trf_features.py — TRF-01 race_rank + TRF-02 weighted_recent_form のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from db.parquet_store import ParquetStore
from features.horse_history_features import HorseHistoryFeatures


def _make_store_with_history() -> MagicMock:
    """3走の過去成績を持つモックストア"""
    store = MagicMock(spec=ParquetStore)
    entries_hist = pd.DataFrame({
        "race_id": ["p1", "p2", "p3"],
        "race_date": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-04-01"]),
        "kettonum": ["K1", "K1", "K1"],
        "kisyucode": ["J1", "J1", "J1"],
        "umaban": [1, 1, 1],
        "kakuteijyuni": [3, 5, 2],
        "odds": [5.0, 8.0, 3.0],
        "harontimel3": [35.0, 36.0, 34.5],
        "distance_bin": ["mile", "mile", "sprint"],
        "timediff": [0.3, -0.2, 0.5],
        "jyuni1c": [5, 8, 3],
        "jyuni4c": [4, 6, 2],
        "kyakusitukubun": [2, 2, 1],
        "bataijyu": [480.0, 482.0, 484.0],
    })
    races_hist = pd.DataFrame({
        "race_id": ["p1", "p2", "p3"],
        "syussotosu": [16, 16, 18],
        "race_date": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-04-01"]),
        "trackcd": [11, 11, 11],
        "kyori": [1600, 1600, 1200],
        "surface": ["turf", "turf", "turf"],
        "track_condition_code": [1, 2, 1],
    })

    def mock_read(category, name, **kwargs):
        if name == "entries":
            return entries_hist
        elif name == "races":
            return races_hist
        return pd.DataFrame()

    store.read = MagicMock(side_effect=mock_read)
    return store


def _make_empty_store() -> MagicMock:
    """空の過去成績を持つモックストア (ヘッダーのみDataFrame)"""
    store = MagicMock(spec=ParquetStore)

    def mock_read(category, name, **kwargs):
        if name == "entries":
            return pd.DataFrame({
                "race_id": pd.Series([], dtype=str),
                "race_date": pd.Series([], dtype="datetime64[ns]"),
                "kettonum": pd.Series([], dtype=str),
                "kisyucode": pd.Series([], dtype=str),
                "umaban": pd.Series([], dtype=int),
                "kakuteijyuni": pd.Series([], dtype=float),
                "odds": pd.Series([], dtype=float),
                "harontimel3": pd.Series([], dtype=float),
                "bataijyu": pd.Series([], dtype=float),
            })
        elif name == "races":
            return pd.DataFrame({
                "race_id": pd.Series([], dtype=str),
                "race_date": pd.Series([], dtype="datetime64[ns]"),
                "syussotosu": pd.Series([], dtype=float),
                "trackcd": pd.Series([], dtype=float),
                "kyori": pd.Series([], dtype=float),
                "surface": pd.Series([], dtype=str),
            })
        return pd.DataFrame()

    store.read = MagicMock(side_effect=mock_read)
    return store


class TestWeightedRecentForm:
    """TRF-02: weighted_recent_form_finish / weighted_recent_form_time"""

    def test_returns_values_with_3_past_races(self):
        """3走以上の過去成績がある場合、weighted_recent_form_* が非NaNで返る"""
        store = _make_store_with_history()
        hhf = HorseHistoryFeatures(store=store)

        race_df = pd.DataFrame({
            "race_id": ["R0"],
            "race_date": pd.to_datetime(["2024-06-01"]),
            "surface": ["turf"],
            "kyori": [1600],
        })
        entry_df = pd.DataFrame({
            "race_id": ["R0"],
            "umaban": [1],
            "kettonum": ["K1"],
            "kisyucode": ["J1"],
            "bataijyu": [486.0],
            "kakuteijyuni": [1],
            "syussotosu": [10],
        })

        result = hhf.compute(race_df, entry_df)
        assert len(result) == 1
        row = result.iloc[0]

        # weighted_recent_form_finish should be non-NaN (3 past races with valid data)
        assert not np.isnan(row["weighted_recent_form_finish"]), \
            f"weighted_recent_form_finish should not be NaN, got {row['weighted_recent_form_finish']}"

        # weighted_recent_form_time should be non-NaN
        assert not np.isnan(row["weighted_recent_form_time"]), \
            f"weighted_recent_form_time should not be NaN, got {row['weighted_recent_form_time']}"

    def test_returns_nan_when_no_past_races(self):
        """過去0走の場合、weighted_recent_form_* は NaN

        Note: compute() has an early return for empty entries_filtered.
        To test a horse with no past races, we provide history for another horse
        so entries_filtered is non-empty, but the target horse K_NEW has no history.
        """
        store = MagicMock(spec=ParquetStore)
        # Provide history for OTHER horses so entries_filtered is non-empty
        entries_hist = pd.DataFrame({
            "race_id": ["p1"],
            "race_date": pd.to_datetime(["2024-01-01"]),
            "kettonum": ["K_OTHER"],
            "kisyucode": ["J_OTHER"],
            "umaban": [1],
            "kakuteijyuni": [3],
            "odds": [5.0],
            "harontimel3": [35.0],
            "bataijyu": [480.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1"],
            "syussotosu": [16],
            "race_date": pd.to_datetime(["2024-01-01"]),
            "trackcd": [11],
            "kyori": [1600],
            "surface": ["turf"],
            "track_condition_code": [1],
        })

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            elif name == "races":
                return races_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)
        hhf = HorseHistoryFeatures(store=store)

        race_df = pd.DataFrame({
            "race_id": ["R0"],
            "race_date": pd.to_datetime(["2024-06-01"]),
            "surface": ["turf"],
            "kyori": [1600],
        })
        entry_df = pd.DataFrame({
            "race_id": ["R0"],
            "umaban": [1],
            "kettonum": ["K_NEW"],
            "kisyucode": ["J_OTHER"],  # same jockey as history → passes kisyu filter
        })

        result = hhf.compute(race_df, entry_df)
        # K_NEW has no past races but appears in entries, so should get a row with NaN
        assert len(result) == 1
        row = result.iloc[0]
        assert np.isnan(row["weighted_recent_form_finish"])
        assert np.isnan(row["weighted_recent_form_time"])


class TestRaceRankTRF:
    """TRF-01: add_race_transforms の3新規race_rank列"""

    def test_race_rank_columns_appear(self):
        """form_trend_race_rank, blood_total_wr_race_rank, blood_surface_wr_race_rank が生成される"""
        df = pd.DataFrame({
            "race_id": ["R1", "R1", "R1", "R1"],
            "form_trend": [0.1, -0.2, 0.3, 0.0],
            "blood_total_wr": [0.15, 0.20, 0.10, 0.25],
            "blood_surface_wr": [0.18, 0.22, 0.12, 0.28],
            "norm_finish_logit_avg": [1.0, 0.5, -0.5, 0.0],
        })
        result = HorseHistoryFeatures.add_race_transforms(df)

        assert "form_trend_race_rank" in result.columns
        assert "blood_total_wr_race_rank" in result.columns
        assert "blood_surface_wr_race_rank" in result.columns

    def test_race_rank_values_in_range(self):
        """race_rank値が (0, 1] の範囲にある"""
        df = pd.DataFrame({
            "race_id": ["R1", "R1", "R1", "R1"],
            "form_trend": [0.1, -0.2, 0.3, 0.0],
            "blood_total_wr": [0.15, 0.20, 0.10, 0.25],
            "blood_surface_wr": [0.18, 0.22, 0.12, 0.28],
        })
        result = HorseHistoryFeatures.add_race_transforms(df)

        for col in ["form_trend_race_rank", "blood_total_wr_race_rank", "blood_surface_wr_race_rank"]:
            vals = result[col].dropna()
            assert (vals > 0).all() and (vals <= 1).all(), \
                f"{col} values out of (0, 1]: {vals.tolist()}"


class TestBaseCols:
    """BASE_COLS に新特徴量が含まれることを検証"""

    def test_weighted_recent_form_in_base_cols(self):
        """weighted_recent_form_finish と weighted_recent_form_time が BASE_COLS に含まれる"""
        assert "weighted_recent_form_finish" in HorseHistoryFeatures.BASE_COLS
        assert "weighted_recent_form_time" in HorseHistoryFeatures.BASE_COLS
