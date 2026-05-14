"""RecordFeatures + SireFeatures BMS拡張 の mock-based テスト (DB不要)"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from features.record_features import FEATURE_COLS as RECORD_FEATURE_COLS
from features.record_features import RecordFeatures

# ---------------------------------------------------------------------------
# RecordFeatures helpers
# ---------------------------------------------------------------------------


def _make_record_store(records: pd.DataFrame | None = None) -> MagicMock:
    """RecordFeatures 用 ParquetStore mock"""
    store = MagicMock()

    def exists(category: str, name: str) -> bool:
        if category == "raw" and name == "record":
            return records is not None and not records.empty
        return False

    store.exists.side_effect = exists

    def read(category: str, name: str, **kwargs):  # type: ignore[misc]
        if category == "raw" and name == "record":
            return records if records is not None else pd.DataFrame()
        return pd.DataFrame()

    store.read.side_effect = read
    return store


@pytest.fixture
def multi_record_df() -> pd.DataFrame:
    """2コース x 各2レコード (異なる年) のコースレコードデータ"""
    return pd.DataFrame(
        {
            "recinfokubun": ["1", "1", "1", "1"],
            "jyocd": ["05", "05", "06", "06"],
            "trackcd": ["01", "01", "01", "01"],
            "kyori": ["1600", "1600", "2000", "2000"],
            "rectime": ["1553", "1548", "2105", "2058"],
            "makedate": ["20200101", "20230601", "20200101", "20230601"],
        }
    )


@pytest.fixture
def race_df() -> pd.DataFrame:
    """テスト用レースデータ: 2レース"""
    return pd.DataFrame(
        {
            "race_id": ["R001", "R002"],
            "jyocd": ["05", "06"],
            "trackcd": ["01", "01"],
            "kyori": ["1600", "2000"],
            "umaban": [1, 1],
        }
    )


class TestRecordFeatures:
    """RecordFeatures テストスイート"""

    def test_compute_returns_course_record_time(
        self, multi_record_df: pd.DataFrame, race_df: pd.DataFrame
    ) -> None:
        """Test 1: compute() は course_record_time 列を返す"""
        store = _make_record_store(multi_record_df)
        feat = RecordFeatures(store)
        result = feat.compute(race_df)
        assert "course_record_time" in result.columns

    def test_recinfokubun_filter(
        self, race_df: pd.DataFrame
    ) -> None:
        """Test 2: RecInfoKubun=1 フィルタが適用される"""
        records = pd.DataFrame(
            {
                "recinfokubun": ["1", "2", "1"],
                "jyocd": ["05", "05", "06"],
                "trackcd": ["01", "01", "01"],
                "kyori": ["1600", "1600", "2000"],
                "rectime": ["1553", "1400", "2105"],
                "makedate": ["20230601", "20230601", "20230601"],
            }
        )
        store = _make_record_store(records)
        feat = RecordFeatures(store)
        result = feat.compute(race_df)

        # jyocd=05 に RecInfoKubun=2 (rectime=1400) が含まれないことを確認
        # 1553 → 1*60+55.3 = 115.3
        np.testing.assert_allclose(
            result.loc[result.index[0], "course_record_time"],
            115.3,
            rtol=1e-3,
        )

    def test_most_recent_record_selected(
        self, multi_record_df: pd.DataFrame, race_df: pd.DataFrame
    ) -> None:
        """Test 3: 同一コースの最新レコードが選択される"""
        store = _make_record_store(multi_record_df)
        feat = RecordFeatures(store)
        result = feat.compute(race_df)

        # jyocd=05, trackcd=01, kyori=1600:
        # 2 records: 2020 (1553→115.3s) and 2023 (1548→1*60+54.8=114.8s)
        # most recent (2023) should be selected → 114.8s
        np.testing.assert_allclose(
            result.loc[result.index[0], "course_record_time"],
            114.8,
            rtol=1e-3,
        )

    def test_feature_cols_has_1_entry(self) -> None:
        """Test 4: FEATURE_COLS はちょうど1要素"""
        assert len(RECORD_FEATURE_COLS) == 1
        assert RECORD_FEATURE_COLS == ["course_record_time"]

    def test_empty_store_returns_nan(
        self, race_df: pd.DataFrame
    ) -> None:
        """Test 5: 空 store は全特徴量 NaN"""
        store = _make_record_store()  # no records
        feat = RecordFeatures(store)
        result = feat.compute(race_df)

        assert result["course_record_time"].isna().all()

    def test_rectime_parsing(self) -> None:
        """Test 6: RecTime '1553' が 1*60 + 55.3 = 115.3 秒に正しくパースされる"""
        records = pd.DataFrame(
            {
                "recinfokubun": ["1"],
                "jyocd": ["05"],
                "trackcd": ["01"],
                "kyori": ["1600"],
                "rectime": ["1553"],
                "makedate": ["20230601"],
            }
        )
        store = _make_record_store(records)
        feat = RecordFeatures(store)
        race_df_single = pd.DataFrame(
            {
                "race_id": ["R001"],
                "jyocd": ["05"],
                "trackcd": ["01"],
                "kyori": ["1600"],
            }
        )
        result = feat.compute(race_df_single)
        np.testing.assert_allclose(
            result["course_record_time"].iloc[0], 115.3, rtol=1e-3
        )

    def test_multiple_records_same_course_different_years(
        self, multi_record_df: pd.DataFrame, race_df: pd.DataFrame
    ) -> None:
        """Test 7: 同一コースで異なる年の複数レコード -- 最新が使用される"""
        store = _make_record_store(multi_record_df)
        feat = RecordFeatures(store)
        result = feat.compute(race_df)

        # jyocd=06, trackcd=01, kyori=2000:
        # 2020 (2105→2*60+10.5=130.5s) and 2023 (2058→2*60+05.8=125.8s)
        # most recent (2023) → 125.8s
        np.testing.assert_allclose(
            result.loc[result.index[1], "course_record_time"],
            125.8,
            rtol=1e-3,
        )


# ---------------------------------------------------------------------------
# SireFeatures BMS extension tests
# ---------------------------------------------------------------------------


class TestSireFeaturesBMSExtension:
    """SireFeatures BMS拡張テストスイート"""

    def test_compute_batch_returns_bms_distance_wr(self) -> None:
        """Test 8: compute_batch() は bms_distance_wr 列を返す"""
        from features.sire_features import SireFeatures

        stats = pd.DataFrame(
            {
                "sire_id": ["S001"],
                "race_date": pd.to_datetime(["2020-01-01"]),
                "sire_wins": [10],
                "sire_starts": [100],
                "sire_places": [30],
                "sire_turf_wins": [6],
                "sire_turf_starts": [60],
                "sire_dirt_wins": [4],
                "sire_dirt_starts": [40],
                "sire_short_wins": [5],
                "sire_short_starts": [50],
                "sire_long_wins": [5],
                "sire_long_starts": [50],
                "sire_prize_total": [1000000],
            }
        )
        feat = SireFeatures(stats)
        df = pd.DataFrame(
            {
                "race_id": ["R001"],
                "race_date": pd.to_datetime(["2020-06-01"]),
                "surface": ["turf"],
                "kyori": [1400],
                "sire_id": ["S001"],
                "bms_id": ["S001"],
            }
        )
        result = feat.compute_batch(df)
        assert "bms_distance_wr" in result.columns

    def test_bms_distance_wr_uses_short_for_short(
        self,
    ) -> None:
        """Test 9: bms_distance_wr は kyori<=1600 で short distance stats を使用"""
        from features.sire_features import SireFeatures

        stats = pd.DataFrame(
            {
                "sire_id": ["S001"],
                "race_date": pd.to_datetime(["2020-01-01"]),
                "sire_wins": [10],
                "sire_starts": [100],
                "sire_places": [30],
                "sire_turf_wins": [6],
                "sire_turf_starts": [60],
                "sire_dirt_wins": [4],
                "sire_dirt_starts": [40],
                "sire_short_wins": [8],
                "sire_short_starts": [40],
                "sire_long_wins": [2],
                "sire_long_starts": [60],
                "sire_prize_total": [1000000],
            }
        )
        feat = SireFeatures(stats)
        df = pd.DataFrame(
            {
                "race_id": ["R001"],
                "race_date": pd.to_datetime(["2020-06-01"]),
                "surface": ["turf"],
                "kyori": [1400],
                "sire_id": ["S001"],
                "bms_id": ["S001"],
            }
        )
        result = feat.compute_batch(df)
        # short: (8+1)/(40+11) = 9/51 ≈ 0.1765
        expected = (8 + 1) / (40 + 11)
        np.testing.assert_allclose(
            result["bms_distance_wr"].iloc[0], expected, rtol=1e-3
        )

    def test_bms_surface_wr_uses_turf_for_turf(
        self,
    ) -> None:
        """Test 10: bms_surface_wr は surface=turf で turf stats を使用"""
        from features.sire_features import SireFeatures

        stats = pd.DataFrame(
            {
                "sire_id": ["S001"],
                "race_date": pd.to_datetime(["2020-01-01"]),
                "sire_wins": [10],
                "sire_starts": [100],
                "sire_places": [30],
                "sire_turf_wins": [7],
                "sire_turf_starts": [50],
                "sire_dirt_wins": [3],
                "sire_dirt_starts": [50],
                "sire_short_wins": [5],
                "sire_short_starts": [50],
                "sire_long_wins": [5],
                "sire_long_starts": [50],
                "sire_prize_total": [1000000],
            }
        )
        feat = SireFeatures(stats)
        df = pd.DataFrame(
            {
                "race_id": ["R001"],
                "race_date": pd.to_datetime(["2020-06-01"]),
                "surface": ["turf"],
                "kyori": [1800],
                "sire_id": ["S001"],
                "bms_id": ["S001"],
            }
        )
        result = feat.compute_batch(df)
        # turf: (7+1)/(50+11) = 8/61 ≈ 0.1311
        expected = (7 + 1) / (50 + 11)
        np.testing.assert_allclose(
            result["bms_surface_wr"].iloc[0], expected, rtol=1e-3
        )
