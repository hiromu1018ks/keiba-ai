"""readers.py の DB 版ローダーのテスト"""

from unittest.mock import MagicMock

import pandas as pd

from db.readers import (
    load_entries_from_db,
    load_odds_snapshots_from_db,
    load_odds_time_series_from_db,
    load_races_from_db,
)


class TestLoadRacesFromDb:
    """load_races_from_db のテスト"""

    def test_applies_type_conversions_and_derives_race_id(self) -> None:
        """型変換 → race_date → race_id → _coerce_types → _exclude_steeple が適用される"""
        mock_db = MagicMock()
        mock_db.get_races.return_value = pd.DataFrame(
            {
                "year": ["2026"],
                "monthday": ["0405"],
                "jyocd": ["05"],
                "kaiji": ["01"],
                "nichiji": ["01"],
                "racenum": ["01"],
                "trackcd": ["11"],
                "kyori": ["1200"],
                "tenkocd": ["2"],
                "syussotosu": ["18"],
                "honsyokin": ["10000000"],
            }
        )

        result = load_races_from_db(mock_db, "20260405")

        assert not result.empty
        assert "race_date" in result.columns
        assert "race_id" in result.columns
        assert result["race_id"].iloc[0] == "2026040505010101"
        # trackcd が Int64 に変換される
        assert result["trackcd"].dtype.name == "Int64"
        assert result["trackcd"].iloc[0] == 11

    def test_excludes_steeple_races(self) -> None:
        """障害レース (trackcd 51-59) が除外される"""
        mock_db = MagicMock()
        mock_db.get_races.return_value = pd.DataFrame(
            {
                "year": ["2026", "2026"],
                "monthday": ["0405", "0405"],
                "jyocd": ["05", "05"],
                "kaiji": ["01", "01"],
                "nichiji": ["01", "01"],
                "racenum": ["01", "02"],
                "trackcd": ["11", "55"],  # 芝 + 障害
            }
        )

        result = load_races_from_db(mock_db, "20260405")

        assert len(result) == 1
        assert result["trackcd"].iloc[0] == 11

    def test_empty_result_returns_empty_dataframe(self) -> None:
        mock_db = MagicMock()
        mock_db.get_races.return_value = pd.DataFrame()

        result = load_races_from_db(mock_db, "20260405")
        assert result.empty


class TestLoadEntriesFromDb:
    """load_entries_from_db のテスト"""

    def test_applies_type_conversions_and_derives_race_id(self) -> None:
        mock_db = MagicMock()
        mock_db.get_entries.return_value = pd.DataFrame(
            {
                "year": ["2026", "2026"],
                "monthday": ["0405", "0405"],
                "jyocd": ["05", "05"],
                "kaiji": ["01", "01"],
                "nichiji": ["01", "01"],
                "racenum": ["01", "01"],
                "umaban": ["1", "2"],
                "kettonum": ["0012345678", "0012345679"],
                "kakuteijyuni": ["1", ""],
                "ninki": ["1", "3"],
            }
        )

        result = load_entries_from_db(mock_db, "20260405")

        assert not result.empty
        assert result["race_id"].iloc[0] == "2026040505010101"
        assert result["umaban"].dtype.name == "Int64"
        assert result["umaban"].iloc[0] == 1
        # 空文字は NA になる
        assert pd.isna(result["kakuteijyuni"].iloc[1])

    def test_excludes_steeple_entries(self) -> None:
        mock_db = MagicMock()
        mock_db.get_entries.return_value = pd.DataFrame(
            {
                "year": ["2026", "2026"],
                "monthday": ["0405", "0405"],
                "jyocd": ["05", "05"],
                "kaiji": ["01", "01"],
                "nichiji": ["01", "01"],
                "racenum": ["01", "01"],
                "umaban": ["1", "2"],
                "trackcd": ["11", "55"],
            }
        )

        result = load_entries_from_db(mock_db, "20260405")
        assert len(result) == 1


class TestLoadOddsSnapshotsFromDb:
    """load_odds_snapshots_from_db のテスト"""

    def test_converts_odds_with_divisor_10(self) -> None:
        """tanodds, fukuoddslow を /10 で変換: "150" -> 15.0"""
        mock_db = MagicMock()
        mock_db.get_odds_snapshots.return_value = pd.DataFrame(
            {
                "year": ["2026"],
                "monthday": ["0405"],
                "jyocd": ["05"],
                "kaiji": ["01"],
                "nichiji": ["01"],
                "racenum": ["01"],
                "umaban": ["1"],
                "tanodds": ["150"],
                "fukuoddslow": ["80"],
            }
        )

        result = load_odds_snapshots_from_db(mock_db, "20260405")

        assert result["tanodds"].iloc[0] == 15.0
        assert result["fukuoddslow"].iloc[0] == 8.0
        assert result["umaban"].dtype.name == "Int64"
        assert result["umaban"].iloc[0] == 1

    def test_empty_result_returns_empty_dataframe(self) -> None:
        mock_db = MagicMock()
        mock_db.get_odds_snapshots.return_value = pd.DataFrame()

        result = load_odds_snapshots_from_db(mock_db, "20260405")
        assert result.empty


class TestLoadOddsTimeSeriesFromDb:
    """load_odds_time_series_from_db のテスト"""

    def test_happyotime_preserved_as_string(self) -> None:
        """happyotime が _coerce_types で数値変換されない"""
        mock_db = MagicMock()
        mock_db.get_odds_time_series.return_value = pd.DataFrame(
            {
                "year": ["2026"],
                "monthday": ["0405"],
                "jyocd": ["05"],
                "kaiji": ["01"],
                "nichiji": ["01"],
                "racenum": ["01"],
                "umaban": ["1"],
                "tanodds": ["150"],
                "fukuoddslow": ["80"],
                "tanninki": ["1"],
                "happyotime": ["03101500"],
            }
        )

        result = load_odds_time_series_from_db(mock_db, "20260405")

        assert result["happyotime"].dtype == object
        assert result["happyotime"].iloc[0] == "03101500"
        assert result["tanninki"].dtype.name == "Int64"
        assert result["tanninki"].iloc[0] == 1

    def test_converts_odds_and_derives_race_id(self) -> None:
        mock_db = MagicMock()
        mock_db.get_odds_time_series.return_value = pd.DataFrame(
            {
                "year": ["2026"],
                "monthday": ["0405"],
                "jyocd": ["05"],
                "kaiji": ["01"],
                "nichiji": ["01"],
                "racenum": ["01"],
                "umaban": ["1"],
                "tanodds": ["55"],
                "fukuoddslow": ["22"],
                "tanninki": ["2"],
                "happyotime": ["03101500"],
            }
        )

        result = load_odds_time_series_from_db(mock_db, "20260405")

        assert result["race_id"].iloc[0] == "2026040505010101"
        assert result["tanodds"].iloc[0] == 5.5
        assert result["tanninki"].iloc[0] == 2
