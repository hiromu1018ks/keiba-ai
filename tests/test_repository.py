"""DataRepository のテスト — 日付フィルタ・障害除外・pyarrowプッシュダウン"""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from db.parquet_store import ParquetStore
from db.repository import DataRepository, _date_filters, _exclude_steeple, _to_dt


class TestHelpers:
    def test_to_dt(self) -> None:
        assert _to_dt("20200101") == datetime(2020, 1, 1)

    def test_date_filters(self) -> None:
        filters = _date_filters("20200101", "20201231")
        assert len(filters) == 2
        assert filters[0] == ("race_date", ">=", datetime(2020, 1, 1))
        assert filters[1] == ("race_date", "<=", datetime(2020, 12, 31))

    def test_exclude_steeple(self) -> None:
        df = pd.DataFrame({"track_cd": [30, 51, 55, 59, 10]})
        result = _exclude_steeple(df)
        assert list(result["track_cd"]) == [30, 10]

    def test_exclude_steeple_no_track_cd_column(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(KeyError):
            _exclude_steeple(df)


@pytest.fixture
def mock_store() -> MagicMock:
    return MagicMock(spec=ParquetStore)


@pytest.fixture
def repo(mock_store: MagicMock) -> DataRepository:
    return DataRepository(store=mock_store)


class TestDataRepositoryLoadRaces:
    def test_calls_store_with_date_filters(
        self, repo: DataRepository, mock_store: MagicMock
    ) -> None:
        mock_store.read.return_value = pd.DataFrame(
            {"race_date": [datetime(2020, 6, 1)], "track_cd": [10]}
        )
        repo.load_races("20200101", "20201231")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("raw", "races")
        filters = call_args[1].get("filters") or call_args[0][2]
        assert filters is not None

    def test_excludes_steeple(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame(
            {
                "race_date": [datetime(2020, 6, 1)] * 3,
                "track_cd": [10, 51, 55],
            }
        )
        result = repo.load_races("20200101", "20201231")
        assert len(result) == 1
        assert result["track_cd"].iloc[0] == 10


class TestDataRepositoryLoadEntries:
    def test_calls_store_correctly(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame(
            {"race_date": [datetime(2020, 6, 1)], "track_cd": [10]}
        )
        repo.load_entries("20200101", "20201231")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("raw", "entries")

    def test_excludes_steeple(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame(
            {
                "race_date": [datetime(2020, 6, 1)] * 3,
                "track_cd": [10, 51, 55],
            }
        )
        result = repo.load_entries("20200101", "20201231")
        assert len(result) == 1
        assert result["track_cd"].iloc[0] == 10


class TestDataRepositoryLoadOddsSnapshots:
    def test_calls_store_correctly(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame({"race_date": [datetime(2020, 6, 1)]})
        repo.load_odds_snapshots("20200101", "20201231")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("odds", "snapshots")


class TestDataRepositoryLoadWideOdds:
    def test_calls_store_correctly(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame({"race_date": [datetime(2020, 6, 1)]})
        repo.load_wide_odds("20200101", "20201231")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("odds", "wide")


class TestDataRepositoryLoadPayouts:
    def test_calls_store_correctly(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame({"race_date": [datetime(2020, 6, 1)]})
        repo.load_payouts("20200101", "20201231")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("raw", "payouts")


class TestDataRepositoryLoadOddsTimeSeries:
    def test_range_calls_partitioned_table(
        self, repo: DataRepository, mock_store: MagicMock
    ) -> None:
        mock_store.read.return_value = pd.DataFrame({"race_date": [datetime(2020, 6, 1)]})
        repo.load_odds_time_series_range("20200101", "20201231")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("odds", "time_series")

    def test_single_race_filters_by_race_id(
        self, repo: DataRepository, mock_store: MagicMock
    ) -> None:
        mock_store.read.return_value = pd.DataFrame({"race_id": ["abc", "def"]})
        repo.load_odds_time_series("abc")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("odds", "time_series")
        filters = call_args[1].get("filters") or call_args[0][2]
        assert any(f[0] == "race_id" for f in filters)


class TestDataRepositoryLoadHistory:
    def test_load_history_entries_uses_lookback(
        self, repo: DataRepository, mock_store: MagicMock
    ) -> None:
        mock_store.read.return_value = pd.DataFrame({"race_date": [datetime(2020, 1, 1)]})
        repo.load_history_entries(lookback_years=3)
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("raw", "entries")
        filters = call_args[1].get("filters") or call_args[0][2]
        assert len(filters) == 1
        assert filters[0][0] == "race_date"

    def test_load_history_races_uses_lookback(
        self, repo: DataRepository, mock_store: MagicMock
    ) -> None:
        mock_store.read.return_value = pd.DataFrame({"race_date": [datetime(2020, 1, 1)]})
        repo.load_history_races(lookback_years=5)
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("raw", "races")


class TestDataRepositoryFeatures:
    def test_load_features_returns_none_when_missing(
        self, repo: DataRepository, mock_store: MagicMock
    ) -> None:
        mock_store.exists.return_value = False
        result = repo.load_features("20200101", "20201231")
        assert result is None

    def test_load_features_returns_df_when_exists(
        self, repo: DataRepository, mock_store: MagicMock
    ) -> None:
        mock_store.exists.return_value = True
        mock_store.read.return_value = pd.DataFrame({"race_date": [datetime(2020, 6, 1)]})
        result = repo.load_features("20200101", "20201231")
        assert result is not None


class TestDataRepositorySave:
    def test_save_features(self, repo: DataRepository, mock_store: MagicMock) -> None:
        df = pd.DataFrame({"a": [1]})
        repo.save_features(df)
        mock_store.write.assert_called_once_with("features", "horse_features", df)

    def test_save_predictions(self, repo: DataRepository, mock_store: MagicMock) -> None:
        df = pd.DataFrame({"a": [1]})
        repo.save_predictions(df)
        mock_store.write.assert_called_once_with("predictions", "predictions", df)

    def test_save_bets(self, repo: DataRepository, mock_store: MagicMock) -> None:
        df = pd.DataFrame({"a": [1]})
        repo.save_bets(df)
        mock_store.write.assert_called_once_with("bets", "bets", df)
