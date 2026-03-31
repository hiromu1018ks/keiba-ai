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
        """track_cd列がない場合はそのまま返す（entriesテーブル等）"""
        df = pd.DataFrame({"a": [1, 2]})
        result = _exclude_steeple(df)
        assert len(result) == 2


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
            {"race_date": [datetime(2020, 6, 1)], "trackcd": ["10"]}
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
                "trackcd": ["10", "51", "55"],
            }
        )
        result = repo.load_races("20200101", "20201231")
        assert len(result) == 1
        assert result["track_cd"].iloc[0] == 10


class TestDataRepositoryLoadEntries:
    def test_calls_store_correctly(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame(
            {"race_date": [datetime(2020, 6, 1)], "trackcd": ["10"]}
        )
        repo.load_entries("20200101", "20201231")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("raw", "entries")

    def test_excludes_steeple(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame(
            {
                "race_date": [datetime(2020, 6, 1)] * 3,
                "trackcd": ["10", "51", "55"],
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


class TestDataRepositoryLoadHorses:
    def test_calls_store_correctly(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame({"horse_id": [1]})
        result = repo.load_horses()
        mock_store.read.assert_called_once_with("raw", "horses")
        assert result is not None

    def test_no_date_filters(self, repo: DataRepository, mock_store: MagicMock) -> None:
        """静的マスターデータには日付フィルタが不要"""
        mock_store.read.return_value = pd.DataFrame({"horse_id": [1]})
        repo.load_horses()
        call_args = mock_store.read.call_args
        # filters が渡されていない（位置引数は2つのみ、キーワードもなし）
        assert call_args[0] == ("raw", "horses")
        assert "filters" not in call_args[1]


class TestDataRepositoryLoadJockeyStats:
    def test_calls_store_correctly(
        self, repo: DataRepository, mock_store: MagicMock
    ) -> None:
        mock_store.read.return_value = pd.DataFrame({"jockey_id": [1]})
        result = repo.load_jockey_stats()
        mock_store.read.assert_called_once_with("raw", "jockey_stats")
        assert result is not None


class TestDataRepositoryLoadTrainerStats:
    def test_calls_store_correctly(
        self, repo: DataRepository, mock_store: MagicMock
    ) -> None:
        mock_store.read.return_value = pd.DataFrame({"trainer_id": [1]})
        result = repo.load_trainer_stats()
        mock_store.read.assert_called_once_with("raw", "trainer_stats")
        assert result is not None


class TestTransformLayer:
    def test_load_races_renames_raw_columns(self, repo: DataRepository, mock_store: MagicMock):
        """Repository が生カラム名をML既存名にリネームする"""
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "year": [2020], "monthday": ["0601"],
            "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
            "trackcd": ["11"], "kyori": ["1600"], "tenkocd": ["1"],
            "syubetucd": ["13"], "jyokencd1": ["999"],
            "gradecd": [""], "syussotosu": ["18"],
        })
        result = repo.load_races("20200101", "20201231")
        assert "month_day" in result.columns
        assert "jyo_cd" in result.columns
        assert "race_num" in result.columns
        assert "track_cd" in result.columns
        assert "distance" in result.columns
        assert "monthday" not in result.columns
        assert "jyocd" not in result.columns
        assert "trackcd" not in result.columns

    def test_load_entries_renames_raw_columns(self, repo: DataRepository, mock_store: MagicMock):
        """entries の生カラム名がリネームされる"""
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "year": [2020], "monthday": ["0601"],
            "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
            "umaban": ["1"], "kettonum": ["0001234567"],
            "kakuteijyuni": ["3"], "time": ["95.3"],
            "odds": ["0054"], "ninki": ["3"],
            "bataijyu": ["480"], "zogenfugo": [""], "zogensa": [""],
            "kisyucode": ["01056"], "chokyosicode": ["01023"],
            "harontimel3": ["33.5"], "timediff": ["0.3"],
            "jyuni1c": ["2"], "jyuni4c": ["3"],
            "honsyokin": ["0"], "kyakusitukubun": ["0"],
        })
        result = repo.load_entries("20200101", "20201231")
        assert "ketto_num" in result.columns
        assert "finish_pos" in result.columns
        assert "win_odds" in result.columns
        assert "kisyu_code" in result.columns
        assert "haron_time_l3" in result.columns
        assert "kettonum" not in result.columns
        assert "kakuteijyuni" not in result.columns

    def test_load_races_computes_race_id(self, repo: DataRepository, mock_store: MagicMock):
        """race_id が生カラムから計算される"""
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "year": [2020], "monthday": ["0601"],
            "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
            "trackcd": ["11"], "kyori": ["1600"],
        })
        result = repo.load_races("20200101", "20201231")
        assert "race_id" in result.columns
        assert result["race_id"].iloc[0] == "2020060105010101"

    def test_load_races_steeple_exclusion_still_works(self, repo: DataRepository, mock_store: MagicMock):
        """障害除外が track_cd (変換後int) で動作"""
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)] * 3,
            "trackcd": ["11", "51", "55"],
            "year": [2020] * 3, "monthday": ["0601"] * 3,
            "jyocd": ["05"] * 3, "kaiji": ["01"] * 3,
            "nichiji": ["01"] * 3, "racenum": ["01"] * 3,
            "kyori": ["1600"] * 3,
        })
        result = repo.load_races("20200101", "20201231")
        assert len(result) == 1
        assert result["track_cd"].iloc[0] == 11


class TestTransformPayoutsColumns:
    def test_payouts_columns_renamed(self, repo: DataRepository, mock_store: MagicMock):
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "paytansyoumaban1": ["3"], "paytansyopay1": ["540"],
            "payfukusyoumaban1": ["3"], "payfukusyopay1": ["140"],
        })
        result = repo.load_payouts("20200101", "20201231")
        assert "tan_umaban" in result.columns
        assert "tan_pay" in result.columns
        assert "fuku_umaban1" in result.columns

    def test_payouts_types_converted(self, repo: DataRepository, mock_store: MagicMock):
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "paytansyoumaban1": ["3"], "paytansyopay1": ["540"],
        })
        result = repo.load_payouts("20200101", "20201231")
        assert result["tan_umaban"].iloc[0] == 3
        assert result["tan_pay"].iloc[0] == 540.0


class TestTransformOddsColumns:
    def test_odds_snapshots_renamed(self, repo: DataRepository, mock_store: MagicMock):
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "umaban": ["1"], "tanodds": ["0032"], "fukuoddslow": ["0013"],
        })
        result = repo.load_odds_snapshots("20200101", "20201231")
        assert "tan_odds" in result.columns
        assert "fuku_odds" in result.columns

    def test_odds_snapshots_types_converted(self, repo: DataRepository, mock_store: MagicMock):
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "umaban": ["1"], "tanodds": ["0032"], "fukuoddslow": ["0013"],
        })
        result = repo.load_odds_snapshots("20200101", "20201231")
        assert result["tan_odds"].iloc[0] == 3.2
        assert result["fuku_odds"].iloc[0] == 1.3

    def test_wide_odds_renamed(self, repo: DataRepository, mock_store: MagicMock):
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "kumi": ["1-2"], "oddslow": ["00320"], "oddshigh": ["00450"],
        })
        result = repo.load_wide_odds("20200101", "20201231")
        assert "odds_low" in result.columns
        assert "odds_high" in result.columns

    def test_wide_odds_types_converted(self, repo: DataRepository, mock_store: MagicMock):
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "kumi": ["1-2"], "oddslow": ["00320"], "oddshigh": ["00450"],
        })
        result = repo.load_wide_odds("20200101", "20201231")
        assert result["odds_low"].iloc[0] == 3.20
        assert result["odds_high"].iloc[0] == 4.50
