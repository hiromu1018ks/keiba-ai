"""BloodlineFeatures PIT (point-in-time) 版のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from features.bloodline_features import ALPHA_PRIOR, TOTAL_OFFSET, BloodlineFeatures


def _make_store_mock(career_stats_df: pd.DataFrame | None = None) -> MagicMock:
    """ParquetStore モックを作成。"""
    store = MagicMock()

    def read_side_effect(group, name, **kwargs):
        if name == "horse_career_stats" and career_stats_df is not None:
            return career_stats_df
        return pd.DataFrame()

    def exists_side_effect(group, name):
        return group == "raw" and name == "horse_career_stats" and career_stats_df is not None

    store.read = MagicMock(side_effect=read_side_effect)
    store.exists = MagicMock(side_effect=exists_side_effect)
    return store


def test_pit_debut_horse_gets_nan_blood_total_wr():
    """デビュー馬は blood_total_wr = NaN"""
    career = pd.DataFrame(
        {
            "race_id": ["20250101A01"],
            "kettonum": ["H001"],
            "race_date": pd.to_datetime(["2025-01-01"]),
            "cum_starts": [0],
            "cum_wins": [0],
            "cum_prize": [0.0],
            "cum_turf_starts": [0],
            "cum_turf_wins": [0],
            "cum_dirt_starts": [0],
            "cum_dirt_wins": [0],
            "cum_short_starts": [0],
            "cum_short_wins": [0],
        }
    )
    entry_df = pd.DataFrame(
        {
            "race_id": ["20250101A01"],
            "umaban": [1],
            "kettonum": ["H001"],
        }
    )

    store = _make_store_mock(career)
    bf = BloodlineFeatures(store)
    result = bf.compute(entry_df)

    assert result.iloc[0]["blood_total_wr"] is np.nan or pd.isna(result.iloc[0]["blood_total_wr"])


def test_pit_experienced_horse_gets_correct_wr():
    """既出走馬は正しい point-in-time 勝率を得る"""
    career = pd.DataFrame(
        {
            "race_id": ["20250115A01"],
            "kettonum": ["H001"],
            "race_date": pd.to_datetime(["2025-01-15"]),
            "cum_starts": [5],
            "cum_wins": [2],
            "cum_prize": [100000.0],
            "cum_turf_starts": [3],
            "cum_turf_wins": [1],
            "cum_dirt_starts": [2],
            "cum_dirt_wins": [1],
            "cum_short_starts": [2],
            "cum_short_wins": [1],
        }
    )
    entry_df = pd.DataFrame(
        {
            "race_id": ["20250115A01"],
            "umaban": [1],
            "kettonum": ["H001"],
        }
    )

    store = _make_store_mock(career)
    bf = BloodlineFeatures(store)
    result = bf.compute(entry_df)

    # blood_total_wr = (2 + 1) / (5 + 11) = 3/16 = 0.1875
    assert abs(result.iloc[0]["blood_total_wr"] - (2 + ALPHA_PRIOR) / (5 + TOTAL_OFFSET)) < 1e-6


def test_pit_prize_log():
    """累積賞金の log 変換が正しいこと"""
    career = pd.DataFrame(
        {
            "race_id": ["20250115A01"],
            "kettonum": ["H001"],
            "race_date": pd.to_datetime(["2025-01-15"]),
            "cum_starts": [3],
            "cum_wins": [1],
            "cum_prize": [50000.0],
            "cum_turf_starts": [3],
            "cum_turf_wins": [1],
            "cum_dirt_starts": [0],
            "cum_dirt_wins": [0],
            "cum_short_starts": [0],
            "cum_short_wins": [0],
        }
    )
    entry_df = pd.DataFrame(
        {
            "race_id": ["20250115A01"],
            "umaban": [1],
            "kettonum": ["H001"],
        }
    )

    store = _make_store_mock(career)
    bf = BloodlineFeatures(store)
    result = bf.compute(entry_df)

    # blood_prize_log = log(1 + 50000) = log(50001)
    assert abs(result.iloc[0]["blood_prize_log"] - np.log1p(50000)) < 1e-6


def test_pit_surface_wr():
    """芝別勝率が正しいこと"""
    career = pd.DataFrame(
        {
            "race_id": ["20250115A01"],
            "kettonum": ["H001"],
            "race_date": pd.to_datetime(["2025-01-15"]),
            "cum_starts": [5],
            "cum_wins": [2],
            "cum_prize": [0.0],
            "cum_turf_starts": [3],
            "cum_turf_wins": [1],
            "cum_dirt_starts": [2],
            "cum_dirt_wins": [1],
            "cum_short_starts": [0],
            "cum_short_wins": [0],
        }
    )
    entry_df = pd.DataFrame(
        {
            "race_id": ["20250115A01"],
            "umaban": [1],
            "kettonum": ["H001"],
        }
    )

    store = _make_store_mock(career)
    bf = BloodlineFeatures(store)
    result = bf.compute(entry_df)

    # blood_surface_wr = (1 + 1) / (3 + 11) = 2/14 ≈ 0.1429
    expected = (1 + ALPHA_PRIOR) / (3 + TOTAL_OFFSET)
    assert abs(result.iloc[0]["blood_surface_wr"] - expected) < 1e-6


def test_pit_surface_wr_zero_turf_starts_is_nan():
    """芝出走が0の場合は blood_surface_wr = NaN"""
    career = pd.DataFrame(
        {
            "race_id": ["20250115A01"],
            "kettonum": ["H001"],
            "race_date": pd.to_datetime(["2025-01-15"]),
            "cum_starts": [3],
            "cum_wins": [1],
            "cum_prize": [0.0],
            "cum_turf_starts": [0],
            "cum_turf_wins": [0],
            "cum_dirt_starts": [3],
            "cum_dirt_wins": [1],
            "cum_short_starts": [0],
            "cum_short_wins": [0],
        }
    )
    entry_df = pd.DataFrame(
        {
            "race_id": ["20250115A01"],
            "umaban": [1],
            "kettonum": ["H001"],
        }
    )

    store = _make_store_mock(career)
    bf = BloodlineFeatures(store)
    result = bf.compute(entry_df)

    assert pd.isna(result.iloc[0]["blood_surface_wr"])
