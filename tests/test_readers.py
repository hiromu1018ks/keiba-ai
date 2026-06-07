"""readers.py のテスト"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from db.readers import (
    load_career_stats,
    load_entries,
    load_horses,
    load_jockey_stats,
    load_keito,
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
        assert args == ("odds", "jodds_tanpuku")

    def test_falls_back_to_time_series_when_empty(self):
        """jodds_tanpuku が空の場合、time_series にフォールバックする。"""
        store = MagicMock()
        store.exists.side_effect = lambda cat, name: name in ("time_series", "jodds_tanpuku")
        empty_df = pd.DataFrame()
        fallback_df = pd.DataFrame(
            {
                "race_id": ["20260401010101"],
                "happyo_time": ["03241000"],
                "umaban": [1],
                "tan_odds": [3.0],
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
        assert args == ("odds", "jodds_tanpuku")

    def test_falls_back_to_time_series_when_jodds_empty(self):
        """jodds_tanpuku が空の場合、time_series にフォールバックする。"""
        store = MagicMock()
        store.exists.side_effect = lambda cat, name: name in ("time_series", "jodds_tanpuku")
        empty_df = pd.DataFrame()
        fallback_df = pd.DataFrame(
            {
                "race_id": ["20260401010101"],
                "happyo_time": ["03241000"],
                "umaban": [1],
                "tan_odds": [3.0],
            }
        )
        store.read.side_effect = [empty_df, fallback_df]
        result = load_odds_time_series_range(store, "20260401", "20260401")
        assert store.read.call_count == 2
        assert len(result) == 1

    def test_no_fallback_when_jodds_tanpuku_has_data(self):
        """jodds_tanpuku にデータがある場合、フォールバックしない。"""
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


class TestLoadCareerStats:
    def test_returns_empty_dataframe_when_not_exists(self, mock_store):
        """horse_career_stats.parquet が存在しない場合は空 DataFrame を返す。"""
        mock_store.exists.return_value = False
        result = load_career_stats(mock_store)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_calls_store_with_correct_args(self, mock_store):
        """存在する場合は正しい引数で store.read を呼ぶ。"""
        mock_store.exists.return_value = True
        mock_store.read.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "kettonum": ["1234"], "cum_starts": [5]}
        )
        result = load_career_stats(mock_store)
        assert not result.empty
        mock_store.read.assert_called_once_with("raw", "horse_career_stats")


class TestLoadKeito:
    def test_returns_empty_dataframe_when_not_exists(self, mock_store):
        """keito.parquet が存在しない場合は空 DataFrame を返す。"""
        mock_store.exists.return_value = False
        result = load_keito(mock_store)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_calls_store_with_correct_args(self, mock_store):
        """存在する場合は正しい引数で store.read を呼ぶ。"""
        mock_store.exists.return_value = True
        mock_store.read.return_value = pd.DataFrame(
            {"keitoucode": ["1234"], "keitousystemcd": ["SS"]}
        )
        result = load_keito(mock_store)
        assert not result.empty
        mock_store.read.assert_called_once_with("raw", "keito")

    def test_preserves_current_etl_string_columns(self, mock_store):
        """現行keitoスキーマの識別子・名称・説明を文字列のまま保持する。"""
        mock_store.exists.return_value = True
        mock_store.read.return_value = pd.DataFrame(
            {
                "hansyokunum": ["0110000576"],
                "keitoid": ["02010201"],
                "keitoname": ["パーソロン"],
                "keitoex": ["系統説明"],
            }
        )

        result = load_keito(mock_store)

        assert result.loc[0, "hansyokunum"] == "0110000576"
        assert result.loc[0, "keitoid"] == "02010201"
        assert result.loc[0, "keitoname"] == "パーソロン"
        assert result.loc[0, "keitoex"] == "系統説明"
