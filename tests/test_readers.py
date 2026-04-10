"""readers.py のテスト"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from db.readers import (
    load_entries,
    load_horses,
    load_jockey_stats,
    load_odds_snapshots,
    load_odds_time_series,
    load_odds_time_series_range,
    load_payouts,
    load_races,
    load_trainer_stats,
    load_wide_odds,
)


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.read.return_value = MagicMock()
    return store


class TestLoadRaces:
    def test_calls_store_with_correct_args(self, mock_store):
        load_races(mock_store, "20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("raw", "races")
        assert kwargs["filters"] is not None


class TestLoadEntries:
    def test_calls_store_with_correct_args(self, mock_store):
        load_entries(mock_store, "20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("raw", "entries")


class TestLoadOddsSnapshots:
    def test_calls_store_with_correct_args(self, mock_store):
        load_odds_snapshots(mock_store, "20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("odds", "odds_tanpuku")


class TestLoadWideOdds:
    def test_calls_store_with_correct_args(self, mock_store):
        load_wide_odds(mock_store, "20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("odds", "odds_wide")


class TestLoadPayouts:
    def test_calls_store_with_correct_args(self, mock_store):
        load_payouts(mock_store, "20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("raw", "payouts")


class TestLoadOddsTimeSeries:
    def test_calls_store_with_race_id_filter(self, mock_store):
        mock_store.exists.return_value = True
        mock_store.read.return_value = pd.DataFrame({"race_id": ["20240101010101"]})
        load_odds_time_series(mock_store, "20240101010101")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("odds", "time_series")

    def test_falls_back_to_jodds_tanpuku_when_empty(self):
        """time_series が空の場合、jodds_tanpuku にフォールバックする。"""
        store = MagicMock()
        store.exists.side_effect = lambda cat, name: name in ("time_series", "jodds_tanpuku")
        empty_df = pd.DataFrame()
        fallback_df = pd.DataFrame(
            {
                "race_id": ["20260401010101"],
                "happyotime": ["03241000"],
                "umaban": [1],
                "tanodds": [3.0],
            }
        )
        store.read.side_effect = [empty_df, fallback_df]
        result = load_odds_time_series(store, "20260401010101")
        assert store.read.call_count == 2
        assert len(result) == 1


class TestLoadOddsTimeSeriesRange:
    def test_calls_store_with_date_range(self, mock_store):
        mock_store.exists.return_value = True
        mock_store.read.return_value = pd.DataFrame({"race_id": ["20240101010101"]})
        load_odds_time_series_range(mock_store, "20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("odds", "time_series")

    def test_falls_back_to_jodds_tanpuku_when_time_series_empty(self):
        """time_series が空の場合、jodds_tanpuku にフォールバックする。"""
        store = MagicMock()
        store.exists.side_effect = lambda cat, name: name in ("time_series", "jodds_tanpuku")
        empty_df = pd.DataFrame()
        fallback_df = pd.DataFrame(
            {
                "race_id": ["20260401010101"],
                "happyotime": ["03241000"],
                "umaban": [1],
                "tanodds": [3.0],
            }
        )
        store.read.side_effect = [empty_df, fallback_df]
        result = load_odds_time_series_range(store, "20260401", "20260401")
        assert store.read.call_count == 2
        assert len(result) == 1

    def test_no_fallback_when_time_series_has_data(self):
        """time_series にデータがある場合、フォールバックしない。"""
        store = MagicMock()
        store.exists.return_value = True
        valid_df = pd.DataFrame(
            {
                "race_id": ["20240701010101"],
                "happyotime": ["03241000"],
                "umaban": [1],
                "tanodds": [5.4],
            }
        )
        store.read.return_value = valid_df
        result = load_odds_time_series_range(store, "20240701", "20240701")
        assert store.read.call_count == 1
        assert len(result) == 1


class TestLoadStaticTables:
    def test_load_horses(self, mock_store):
        load_horses(mock_store)
        mock_store.read.assert_called_once_with("raw", "horses")

    def test_load_jockey_stats(self, mock_store):
        load_jockey_stats(mock_store)
        mock_store.read.assert_called_once_with("raw", "kisyu_seiseki")

    def test_load_trainer_stats(self, mock_store):
        load_trainer_stats(mock_store)
        mock_store.read.assert_called_once_with("raw", "chokyo_seiseki")
