"""readers.py のテスト"""

from unittest.mock import MagicMock

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
        load_odds_time_series(mock_store, "20240101010101")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("odds", "jodds_tanpuku")


class TestLoadOddsTimeSeriesRange:
    def test_calls_store_with_date_range(self, mock_store):
        load_odds_time_series_range(mock_store, "20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("odds", "jodds_tanpuku")


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
