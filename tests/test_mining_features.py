"""tests/test_mining_features.py -- MiningFeatures の mock-based テスト (DB不要)"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from features.mining_features import FEATURE_COLS, MiningFeatures, _pivot_mining_to_long


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_race_id(year: str = "2024", monthday: str = "0101",
                  jyocd: str = "05", kaiji: str = "01",
                  nichiji: str = "01", racenum: str = "01") -> str:
    """16桁 race_id を生成 (year+monthday+jyocd+kaiji+nichiji+racenum)"""
    return (
        year.zfill(4) + monthday.zfill(4)
        + jyocd.zfill(2) + kaiji.zfill(2)
        + nichiji.zfill(2) + racenum.zfill(2)
    )


def _make_wide_mining_df() -> pd.DataFrame:
    """2 race, 各レースに3頭 (race1) / 2頭 (race2) のwide-format n_miningを生成。

    Race 1: 3頭 (umaban 01,02,03), DataKubun=3
      DMTime: 95.00 (=1:35.00), 96.50, 98.00 (秒換算)
      DMGosaP: 0.50, 0.60, 0.70
      DMGosaM: 0.30, 0.40, 0.50

    Race 2: 2頭 (umaban 01,02), DataKubun=3
      DMTime: 100.00, 101.50
      DMGosaP: 0.80, 0.90
      DMGosaM: 0.60, 0.70
    """
    rows = []
    # --- Race 1 ---
    race1_id_parts = {
        "year": "2024", "monthday": "0615", "jyocd": "05",
        "kaiji": "01", "nichiji": "01", "racenum": "11",
    }
    base = {
        "recordspec": "DM",
        "datakubun": "3",
        "makedate": "20240615",
        "makehm": "1000",
        **race1_id_parts,
    }
    horses_r1 = [
        ("01", "95.00", "0.50", "0.30"),
        ("02", "96.50", "0.60", "0.40"),
        ("03", "98.00", "0.70", "0.50"),
    ]
    row1 = dict(base)
    for i in range(1, 19):
        idx = i - 1
        if idx < len(horses_r1):
            row1[f"umaban{i}"] = horses_r1[idx][0]
            row1[f"dmtime{i}"] = horses_r1[idx][1]
            row1[f"dmgosap{i}"] = horses_r1[idx][2]
            row1[f"dmgosam{i}"] = horses_r1[idx][3]
        else:
            row1[f"umaban{i}"] = "sp"
            row1[f"dmtime{i}"] = "sp"
            row1[f"dmgosap{i}"] = "sp"
            row1[f"dmgosam{i}"] = "sp"
    rows.append(row1)

    # --- Race 2 ---
    race2_id_parts = {
        "year": "2024", "monthday": "0616", "jyocd": "08",
        "kaiji": "02", "nichiji": "01", "racenum": "05",
    }
    base2 = {
        "recordspec": "DM",
        "datakubun": "3",
        "makedate": "20240616",
        "makehm": "1100",
        **race2_id_parts,
    }
    horses_r2 = [
        ("01", "100.00", "0.80", "0.60"),
        ("02", "101.50", "0.90", "0.70"),
    ]
    row2 = dict(base2)
    for i in range(1, 19):
        idx = i - 1
        if idx < len(horses_r2):
            row2[f"umaban{i}"] = horses_r2[idx][0]
            row2[f"dmtime{i}"] = horses_r2[idx][1]
            row2[f"dmgosap{i}"] = horses_r2[idx][2]
            row2[f"dmgosam{i}"] = horses_r2[idx][3]
        else:
            row2[f"umaban{i}"] = "sp"
            row2[f"dmtime{i}"] = "sp"
            row2[f"dmgosap{i}"] = "sp"
            row2[f"dmgosam{i}"] = "sp"
    rows.append(row2)

    return pd.DataFrame(rows)


def _make_mock_store(mining_df: pd.DataFrame | None = None) -> MagicMock:
    """ParquetStore mock。mining_df=None で空振り"""
    store = MagicMock()
    if mining_df is not None:
        store.exists.return_value = True
        store.read.return_value = mining_df
    else:
        store.exists.return_value = False
    return store


def _make_entry_df(n_races: int = 2) -> pd.DataFrame:
    """テスト用 entry_df (race_id, umaban)。Race1=3頭, Race2=2頭"""
    r1 = _make_race_id("2024", "0615", "05", "01", "01", "11")
    r2 = _make_race_id("2024", "0616", "08", "02", "01", "05")
    rows = []
    for umaban in [1, 2, 3]:
        rows.append({"race_id": r1, "umaban": umaban})
    for umaban in [1, 2]:
        rows.append({"race_id": r2, "umaban": umaban})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPivotMiningToLong:
    """_pivot_mining_to_long: wide -> long 変換のテスト"""

    def test_pivot_produces_correct_row_count(self) -> None:
        """Test 1: 2 races x (3+2 horses) = 5 long rows"""
        wide_df = _make_wide_mining_df()
        # race_id列を計算するために必要な列を追加
        wide_df["race_id"] = (
            wide_df["year"].astype(str).str.zfill(4)
            + wide_df["monthday"].astype(str).str.zfill(4)
            + wide_df["jyocd"].astype(str).str.zfill(2)
            + wide_df["kaiji"].astype(str).str.zfill(2)
            + wide_df["nichiji"].astype(str).str.zfill(2)
            + wide_df["racenum"].astype(str).str.zfill(2)
        )
        long_df = _pivot_mining_to_long(wide_df)
        assert len(long_df) == 5  # 3 + 2 (sp slots filtered)

    def test_pivot_filters_sp_initial_values(self) -> None:
        """Test 2: 'sp' initial values (empty slots) are filtered out"""
        wide_df = _make_wide_mining_df()
        wide_df["race_id"] = (
            wide_df["year"].astype(str).str.zfill(4)
            + wide_df["monthday"].astype(str).str.zfill(4)
            + wide_df["jyocd"].astype(str).str.zfill(2)
            + wide_df["kaiji"].astype(str).str.zfill(2)
            + wide_df["nichiji"].astype(str).str.zfill(2)
            + wide_df["racenum"].astype(str).str.zfill(2)
        )
        long_df = _pivot_mining_to_long(wide_df)
        # 全18スロット x 2レース = 36 から有効5行のみ抽出
        assert "sp" not in long_df["umaban"].values
        # umabanはNaNでない
        assert long_df["umaban"].notna().all()

    def test_pivot_preserves_dm_time_values(self) -> None:
        """DMTimeが正しくfloatに変換される"""
        wide_df = _make_wide_mining_df()
        wide_df["race_id"] = (
            wide_df["year"].astype(str).str.zfill(4)
            + wide_df["monthday"].astype(str).str.zfill(4)
            + wide_df["jyocd"].astype(str).str.zfill(2)
            + wide_df["kaiji"].astype(str).str.zfill(2)
            + wide_df["nichiji"].astype(str).str.zfill(2)
            + wide_df["racenum"].astype(str).str.zfill(2)
        )
        long_df = _pivot_mining_to_long(wide_df)
        r1 = _make_race_id("2024", "0615", "05", "01", "01", "11")
        race1 = long_df[long_df["race_id"] == r1]
        # Race1, umaban=1, DMTime=95.00
        horse1 = race1[race1["umaban"] == 1]
        assert len(horse1) == 1
        assert np.isclose(horse1["dm_time"].iloc[0], 95.00)


class TestMiningFeaturesCompute:
    """MiningFeatures.compute() のテスト"""

    def test_compute_produces_all_feature_cols(self) -> None:
        """Test 3: compute() produces all 4 FEATURE_COLS columns"""
        wide_df = _make_wide_mining_df()
        store = _make_mock_store(wide_df)
        entry_df = _make_entry_df()

        feat = MiningFeatures(store)
        result = feat.compute(entry_df)

        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_dm_time_rank_lowest_time_is_one(self) -> None:
        """Test 4: dm_time_rank=1 for the horse with lowest DMTime in each race"""
        wide_df = _make_wide_mining_df()
        store = _make_mock_store(wide_df)
        entry_df = _make_entry_df()

        feat = MiningFeatures(store)
        result = feat.compute(entry_df)

        # Race 1: DMTime 95.00 (umaban=1) が最小
        r1 = _make_race_id("2024", "0615", "05", "01", "01", "11")
        race1 = result[result["race_id"] == r1]
        horse1_rank = race1[race1["umaban"] == 1]["dm_time_rank"].iloc[0]
        assert horse1_rank == 1.0

        # Race 2: DMTime 100.00 (umaban=1) が最小
        r2 = _make_race_id("2024", "0616", "08", "02", "01", "05")
        race2 = result[result["race_id"] == r2]
        horse1_r2_rank = race2[race2["umaban"] == 1]["dm_time_rank"].iloc[0]
        assert horse1_r2_rank == 1.0

    def test_dm_time_zscore_zero_when_all_same(self) -> None:
        """Test 5: dm_time_zscore is 0 when all horses have same DMTime (std=0 fallback)"""
        # 全馬同じDMTimeのレースを作成
        wide_df = _make_wide_mining_df()
        # Race 1の全馬を同じタイムに変更
        for i in range(1, 4):
            wide_df.loc[0, f"dmtime{i}"] = "95.00"
        store = _make_mock_store(wide_df)
        entry_df = _make_entry_df()

        feat = MiningFeatures(store)
        result = feat.compute(entry_df)

        r1 = _make_race_id("2024", "0615", "05", "01", "01", "11")
        race1 = result[result["race_id"] == r1]
        for _, row in race1.iterrows():
            assert np.isclose(row["dm_time_zscore"], 0.0)

    def test_dm_confidence_range_equals_gosap_plus_gosam(self) -> None:
        """Test 6: dm_confidence_range = DMGosaP + DMGosaM"""
        wide_df = _make_wide_mining_df()
        store = _make_mock_store(wide_df)
        entry_df = _make_entry_df()

        feat = MiningFeatures(store)
        result = feat.compute(entry_df)

        # Race 1, umaban=1: GosaP=0.50, GosaM=0.30 -> range=0.80
        r1 = _make_race_id("2024", "0615", "05", "01", "01", "11")
        race1 = result[result["race_id"] == r1]
        horse1 = race1[race1["umaban"] == 1]
        assert np.isclose(horse1["dm_confidence_range"].iloc[0], 0.80)

        # Race 1, umaban=2: GosaP=0.60, GosaM=0.40 -> range=1.00
        horse2 = race1[race1["umaban"] == 2]
        assert np.isclose(horse2["dm_confidence_range"].iloc[0], 1.00)

    def test_empty_mining_returns_nan(self) -> None:
        """Test 7: empty mining data returns NaN for all features"""
        store = _make_mock_store(None)
        entry_df = _make_entry_df()

        feat = MiningFeatures(store)
        result = feat.compute(entry_df)

        for col in FEATURE_COLS:
            assert result[col].isna().all(), f"Expected all NaN for {col}, got {result[col].values}"

    def test_feature_cols_has_four_entries_no_duplicates(self) -> None:
        """Test 8: FEATURE_COLS list has exactly 4 entries with no duplicates"""
        assert len(FEATURE_COLS) == 4
        assert len(set(FEATURE_COLS)) == 4

    def test_compute_batch_returns_same_format(self) -> None:
        """compute_batch() は compute() と同じフォーマットを返す"""
        wide_df = _make_wide_mining_df()
        store = _make_mock_store(wide_df)
        entry_df = _make_entry_df()

        feat = MiningFeatures(store)
        result = feat.compute_batch(entry_df)

        assert "race_id" in result.columns
        assert "umaban" in result.columns
        for col in FEATURE_COLS:
            assert col in result.columns

    def test_dm_time_margin_to_fav(self) -> None:
        """dm_time_margin_to_fav: 自馬DMTime - レース最小DMTime"""
        wide_df = _make_wide_mining_df()
        store = _make_mock_store(wide_df)
        entry_df = _make_entry_df()

        feat = MiningFeatures(store)
        result = feat.compute(entry_df)

        r1 = _make_race_id("2024", "0615", "05", "01", "01", "11")
        race1 = result[result["race_id"] == r1]

        # umaban=1: DMTime=95.00 (min), margin=0.0
        horse1 = race1[race1["umaban"] == 1]
        assert np.isclose(horse1["dm_time_margin_to_fav"].iloc[0], 0.0)

        # umaban=2: DMTime=96.50, margin=96.50-95.00=1.50
        horse2 = race1[race1["umaban"] == 2]
        assert np.isclose(horse2["dm_time_margin_to_fav"].iloc[0], 1.50)

        # umaban=3: DMTime=98.00, margin=98.00-95.00=3.00
        horse3 = race1[race1["umaban"] == 3]
        assert np.isclose(horse3["dm_time_margin_to_fav"].iloc[0], 3.00)
