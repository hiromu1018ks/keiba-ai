"""EveryDB2Queries のテスト"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd


class TestEveryDB2Queries:
    """EveryDB2Queries の各メソッドをテスト。

    pd.read_sql_query をモックして DB 接続なしでテストする。
    """

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_race_schedule_returns_list(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "venue": ["中山"],
                "race_num": [1],
                "post_time": ["10:05"],
                "surface": ["turf"],
                "distance": [1200],
            }
        )

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        schedule = queries.get_race_schedule(date(2026, 4, 5))

        assert len(schedule) == 1
        assert schedule[0]["race_id"] == "2026040510010101"
        assert schedule[0]["venue"] == "中山"
        mock_read_sql.assert_called_once()
        # SQL に日付パラメータが渡されていることを確認
        call_args = mock_read_sql.call_args
        assert call_args[1]["params"] == ("20260405",)

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_race_schedule_returns_empty_on_non_racing_day(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame()

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        schedule = queries.get_race_schedule(date(2026, 4, 7))  # 月曜
        assert schedule == []

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_race_results(self, mock_connect: MagicMock, mock_read_sql: MagicMock) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "umaban": [3],
                "finish_pos": [1],
                "place_pay": [240.0],
                "place_odds": [2.4],
                "horse_name": ["テスト馬"],
            }
        )

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        results = queries.get_race_results(date(2026, 4, 5))

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        assert results.iloc[0]["umaban"] == 3

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_horse_weights_returns_dataframe(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame(
            {
                "umaban": [1, 2, 3],
                "weight": [468, 502, 480],
            }
        )

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_horse_weights("2026040510010101")

        assert result is not None
        assert len(result) == 3
        assert result.iloc[0]["weight"] == 468

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_horse_weights_returns_none_when_empty(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame()

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_horse_weights("2026040510010101")
        assert result is None

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_latest_odds_returns_dataframe(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame(
            {
                "umaban": [1, 2, 3],
                "tan_odds": [3.2, 5.1, 12.4],
                "fuku_odds": [1.3, 2.1, 4.5],
            }
        )

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_latest_odds("2026040510010101")

        assert result is not None
        assert len(result) == 3
        assert result.iloc[0]["tan_odds"] == 3.2

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_latest_odds_returns_none_when_empty(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame()

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_latest_odds("2026040510010101")
        assert result is None

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_track_condition_returns_value(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame(
            {
                "baba_cd": ["3"],
                "tenko_cd": ["2"],
            }
        )

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_track_condition("2026040510010101")

        assert result == "3"

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_track_condition_returns_none_when_empty(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame()

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_track_condition("2026040510010101")
        assert result is None

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_race_schedule_returns_empty_on_db_error(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = Exception("Connection refused")

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        # エラー時は空リストを返す（例外は伝播しない）
        schedule = queries.get_race_schedule(date(2026, 4, 5))
        assert schedule == []

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_race_results_returns_empty_df_on_db_error(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = Exception("Connection refused")

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        results = queries.get_race_results(date(2026, 4, 5))
        assert isinstance(results, pd.DataFrame)
        assert results.empty

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_horse_weights_returns_none_on_db_error(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = Exception("Connection refused")

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_horse_weights("2026040510010101")
        assert result is None

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_latest_odds_returns_none_on_db_error(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = Exception("Connection refused")

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_latest_odds("2026040510010101")
        assert result is None

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_track_condition_returns_none_on_db_error(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = Exception("Connection refused")

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_track_condition("2026040510010101")
        assert result is None

    # --- get_races / get_entries テスト ---

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_races_returns_raw_dataframe(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame(
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
            }
        )

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_races("20260405")

        assert not result.empty
        assert len(result) == 1
        # 型変換は行わない — 全列 object のまま
        assert result["trackcd"].dtype == object
        assert result.iloc[0]["trackcd"] == "11"
        # s_ テーブルが使われることを確認
        sql_called = mock_read_sql.call_args[0][0]
        assert "s_race" in sql_called

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_races_falls_back_to_n_table(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = [
            pd.DataFrame(),  # s_race 空
            pd.DataFrame({"year": ["2026"], "trackcd": ["11"]}),
        ]

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_races("20260405")

        assert not result.empty
        assert mock_read_sql.call_count == 2
        second_sql = mock_read_sql.call_args_list[1][0][0]
        assert "n_race" in second_sql

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_races_returns_empty_when_both_empty(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame()

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_races("20260405")

        assert result.empty

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_entries_returns_raw_dataframe(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame(
            {
                "year": ["2026", "2026", "2026"],
                "monthday": ["0405", "0405", "0405"],
                "jyocd": ["05", "05", "05"],
                "kaiji": ["01", "01", "01"],
                "nichiji": ["01", "01", "01"],
                "racenum": ["01", "01", "01"],
                "umaban": ["1", "2", "3"],
                "kettonum": ["0012345678", "0012345679", "0012345680"],
            }
        )

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_entries("20260405")

        assert not result.empty
        assert len(result) == 3
        assert result["umaban"].dtype == object
        sql_called = mock_read_sql.call_args[0][0]
        assert "s_uma_race" in sql_called

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_entries_falls_back_to_n_table(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = [
            pd.DataFrame(),
            pd.DataFrame({"year": ["2026"], "umaban": ["1"]}),
        ]

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_entries("20260405")

        assert not result.empty
        assert mock_read_sql.call_count == 2
        second_sql = mock_read_sql.call_args_list[1][0][0]
        assert "n_uma_race" in second_sql

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_races_returns_empty_on_db_error(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = Exception("Connection refused")

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_races("20260405")
        assert result.empty

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_entries_returns_empty_on_db_error(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = Exception("Connection refused")

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_entries("20260405")
        assert result.empty

    # --- get_odds_snapshots / get_odds_time_series テスト ---

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_odds_snapshots_returns_raw_dataframe(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame(
            {
                "year": ["2026", "2026", "2026"],
                "monthday": ["0405", "0405", "0405"],
                "jyocd": ["05", "05", "05"],
                "kaiji": ["01", "01", "01"],
                "nichiji": ["01", "01", "01"],
                "racenum": ["01", "01", "01"],
                "umaban": ["1", "2", "3"],
                "tanodds": ["32", "51", "124"],
                "fukuoddslow": ["13", "21", "45"],
            }
        )

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_odds_snapshots("20260405")

        assert not result.empty
        assert len(result) == 3
        assert result["umaban"].dtype == object
        assert result.iloc[0]["tanodds"] == "32"
        sql_called = mock_read_sql.call_args[0][0]
        assert "s_jodds_tanpuku" in sql_called

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_odds_snapshots_falls_back_to_n_table(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = [
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "year": ["2026"],
                    "monthday": ["0405"],
                    "jyocd": ["05"],
                    "kaiji": ["01"],
                    "nichiji": ["01"],
                    "racenum": ["01"],
                    "umaban": ["1"],
                    "happyotime": ["03101500"],
                    "tanodds": ["50"],
                    "fukuoddslow": ["20"],
                }
            ),
        ]

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_odds_snapshots("20260405")

        assert not result.empty
        assert mock_read_sql.call_count == 2
        second_sql = mock_read_sql.call_args_list[1][0][0]
        assert "n_jodds_tanpuku" in second_sql

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_odds_time_series_returns_raw_dataframe(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.return_value = pd.DataFrame(
            {
                "year": ["2026", "2026"],
                "monthday": ["0405", "0405"],
                "jyocd": ["05", "05"],
                "kaiji": ["01", "01"],
                "nichiji": ["01", "01"],
                "racenum": ["01", "01"],
                "umaban": ["1", "1"],
                "happyotime": ["03101500", "03101530"],
                "tanodds": ["35", "33"],
                "fukuoddslow": ["14", "12"],
                "tanninki": ["1", "1"],
            }
        )

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_odds_time_series("20260405")

        assert not result.empty
        assert len(result) == 2
        assert result["happyotime"].dtype == object
        sql_called = mock_read_sql.call_args[0][0]
        assert "s_jodds_tanpuku" in sql_called

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_odds_time_series_falls_back_to_n_table(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = [
            pd.DataFrame(),
            pd.DataFrame(
                {
                    "year": ["2026"],
                    "monthday": ["0405"],
                    "jyocd": ["05"],
                    "kaiji": ["01"],
                    "nichiji": ["01"],
                    "racenum": ["01"],
                    "umaban": ["1"],
                    "happyotime": ["03101500"],
                    "tanodds": ["35"],
                    "fukuoddslow": ["14"],
                    "tanninki": ["1"],
                }
            ),
        ]

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_odds_time_series("20260405")

        assert not result.empty
        assert mock_read_sql.call_count == 2
        second_sql = mock_read_sql.call_args_list[1][0][0]
        assert "n_jodds_tanpuku" in second_sql

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_odds_snapshots_returns_empty_on_db_error(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = Exception("Connection refused")

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_odds_snapshots("20260405")
        assert result.empty

    @patch("db.everydb2_queries.pd.read_sql_query")
    @patch("db.everydb2_queries.psycopg2.connect")
    def test_get_odds_time_series_returns_empty_on_db_error(
        self, mock_connect: MagicMock, mock_read_sql: MagicMock
    ) -> None:
        from db.everydb2_queries import EveryDB2Queries

        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_read_sql.side_effect = Exception("Connection refused")

        queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
        result = queries.get_odds_time_series("20260405")
        assert result.empty
