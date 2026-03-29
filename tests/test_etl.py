"""src/db/etl モジュールのテスト（モックDB使用・実際のDB接続不要）"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from db.etl import (
    _insert_on_conflict,
    _make_race_id,
    _select_baba_cd,
    _to_float,
    _to_int,
    _to_odds,
    etl_entries,
    etl_odds_snapshots,
    etl_odds_timeseries,
    etl_payouts,
    etl_races,
    etl_wide_odds,
    run_full_etl,
)

# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestToInt:
    def test_empty_string_returns_none(self):
        assert _to_int("") is None

    def test_none_returns_none(self):
        assert _to_int(None) is None

    def test_valid_integer(self):
        assert _to_int("123") == 123

    def test_negative_integer(self):
        assert _to_int("-5") == -5

    def test_zero(self):
        assert _to_int("0") == 0


class TestToFloat:
    def test_empty_string_returns_none(self):
        assert _to_float("") is None

    def test_none_returns_none(self):
        assert _to_float(None) is None

    def test_valid_float(self):
        assert _to_float("3.14") == pytest.approx(3.14)

    def test_integer_string(self):
        assert _to_float("42") == pytest.approx(42.0)

    def test_negative_float(self):
        assert _to_float("-1.5") == pytest.approx(-1.5)

    def test_zero(self):
        assert _to_float("0") == pytest.approx(0.0)


class TestToOdds:
    """EveryDB2 ゼロ埋め整数オッズ → float 変換のテスト"""

    def test_favorite_odds(self):
        """1番人気: "0014" → 1.4"""
        assert _to_odds("0014") == pytest.approx(1.4)

    def test_mid_odds(self):
        """中間オッズ: "0325" → 32.5"""
        assert _to_odds("0325") == pytest.approx(32.5)

    def test_longshot_odds(self):
        """大穴: "6510" → 651.0"""
        assert _to_odds("6510") == pytest.approx(651.0)

    def test_wide_odds_divisor_100(self):
        """ワイドオッズ (÷100): "03783" → 37.83"""
        assert _to_odds("03783", divisor=100) == pytest.approx(37.83)

    def test_wide_odds_low_divisor_100(self):
        """ワイドオッズ低額 (÷100): "00591" → 5.91"""
        assert _to_odds("00591", divisor=100) == pytest.approx(5.91)

    def test_empty_string_returns_none(self):
        assert _to_odds("") is None

    def test_none_returns_none(self):
        assert _to_odds(None) is None

    def test_non_numeric_returns_none(self):
        assert _to_odds("****") is None

    def test_default_divisor_is_10(self):
        """デフォルトの divisor は 10"""
        assert _to_odds("0050") == pytest.approx(5.0)


class TestMakeRaceId:
    def test_standard_race_id(self):
        result = _make_race_id("2024", "0324", "05", "03", "02", "08")
        assert result == "2024032405030208"

    def test_single_digit_components(self):
        result = _make_race_id("2024", "0101", "01", "01", "01", "01")
        assert result == "2024010101010101"


class TestSelectBabaCd:
    def test_turf_track_uses_siba(self):
        assert _select_baba_cd(11, "2", "3") == 2

    def test_dirt_track_uses_dirt(self):
        assert _select_baba_cd(23, "2", "3") == 3

    def test_turf_boundary_low(self):
        assert _select_baba_cd(10, "1", "4") == 1

    def test_turf_boundary_high(self):
        assert _select_baba_cd(22, "3", "2") == 3

    def test_dirt_boundary_low(self):
        assert _select_baba_cd(23, "1", "4") == 4

    def test_dirt_boundary_high(self):
        assert _select_baba_cd(29, "1", "5") == 5

    def test_out_of_range_returns_none(self):
        assert _select_baba_cd(51, "1", "2") is None

    def test_empty_strings(self):
        assert _select_baba_cd(11, "", "3") is None
        assert _select_baba_cd(23, "2", "") is None


# ---------------------------------------------------------------------------
# _insert_on_conflict tests
# ---------------------------------------------------------------------------


class TestInsertOnConflict:
    def test_empty_df_returns_zero(self):
        mock_engine = MagicMock()
        df = pd.DataFrame()
        result = _insert_on_conflict(mock_engine, df, "races", "raw", ["id"])
        assert result == 0
        mock_engine.begin.assert_not_called()

    @patch.object(pd.DataFrame, "to_sql")
    def test_staging_table_pattern(self, mock_to_sql):
        """ステージングテーブル → ON CONFLICT DO NOTHING → DROP の流れを検証"""
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_engine.begin.return_value = mock_cm

        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        result = _insert_on_conflict(mock_engine, df, "test_table", "test_schema", ["id"])

        assert result == 3

        # to_sql がステージングテーブルに書き込み（if_exists="replace"）
        mock_to_sql.assert_called_once_with(
            "_etl_staging_test_table", mock_engine, if_exists="replace", index=False
        )

        # engine.begin() が2回呼ばれる（INSERT + DROP）
        assert mock_engine.begin.call_count == 2

        # 実行されたSQLを検証
        sql_calls = [c[0][0].text for c in mock_conn.execute.call_args_list]
        insert_sql = sql_calls[0]
        assert "ON CONFLICT" in insert_sql
        assert "DO NOTHING" in insert_sql
        assert "test_schema.test_table" in insert_sql

        drop_sql = sql_calls[1]
        assert "DROP TABLE" in drop_sql
        assert "_etl_staging_test_table" in drop_sql


# ---------------------------------------------------------------------------
# ETL function tests (mock-based)
# ---------------------------------------------------------------------------


class TestEtlRaces:
    @patch("db.etl._insert_on_conflict")
    @patch("db.etl.pd.read_sql")
    def test_empty_result(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame()

        result = etl_races(mock_engine, "20240101", "20241231")

        assert result == 0
        mock_insert.assert_not_called()

    @patch("db.etl._insert_on_conflict", return_value=10)
    @patch("db.etl.pd.read_sql")
    def test_calls_insert_with_correct_data(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "year": ["2024"],
                "monthday": ["0324"],
                "jyocd": ["05"],
                "kaiji": ["03"],
                "nichiji": ["02"],
                "racenum": ["08"],
                "trackcd": ["11"],
                "kyori": ["1600"],
                "tenkocd": ["1"],
                "sibababacd": ["2"],
                "dirtbabacd": ["3"],
                "syubetucd": ["13"],
                "jyokencd1": ["999"],
                "gradecd": ["_"],
                "syussotosu": ["18"],
            }
        )

        result = etl_races(mock_engine, "20240101", "20241231")

        assert result == 10
        mock_insert.assert_called_once()
        call_args = mock_insert.call_args
        # _insert_on_conflict(engine, df, table, schema, pk_columns)
        assert call_args[0][2] == "races"
        assert call_args[0][3] == "raw"
        # Check baba_cd is from siba (turf track_cd=11)
        inserted_df = call_args[0][1]
        assert inserted_df["baba_cd"].iloc[0] == 2
        assert inserted_df["track_cd"].iloc[0] == 11
        assert inserted_df["distance"].iloc[0] == 1600

    @patch("db.etl._insert_on_conflict", return_value=5)
    @patch("db.etl.pd.read_sql")
    def test_empty_grade_cd_defaults_to_underscore(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "year": ["2024"],
                "monthday": ["0324"],
                "jyocd": ["05"],
                "kaiji": ["03"],
                "nichiji": ["02"],
                "racenum": ["08"],
                "trackcd": ["23"],
                "kyori": ["1200"],
                "tenkocd": ["1"],
                "sibababacd": ["2"],
                "dirtbabacd": ["3"],
                "syubetucd": ["13"],
                "jyokencd1": ["999"],
                "gradecd": [""],
                "syussotosu": ["14"],
            }
        )

        result = etl_races(mock_engine, "20240101", "20241231")

        assert result == 5
        inserted_df = mock_insert.call_args[0][1]
        assert inserted_df["grade_cd"].iloc[0] == "_"
        # dirt track_cd=23 -> uses dirtbabacd
        assert inserted_df["baba_cd"].iloc[0] == 3


class TestEtlEntries:
    @patch("db.etl._insert_on_conflict")
    @patch("db.etl.pd.read_sql")
    def test_empty_result(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame()

        result = etl_entries(mock_engine, "20240101", "20241231")

        assert result == 0
        mock_insert.assert_not_called()

    @patch("db.etl._insert_on_conflict", return_value=5)
    @patch("db.etl.pd.read_sql")
    def test_column_mapping(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "umaban": ["1"],
                "kettonum": ["0001234567"],
                "kakuteijyuni": ["3"],
                "time": ["95.3"],
                "odds": ["0054"],
                "ninki": ["3"],
                "bataijyu": ["480"],
                "zogenfugo": ["2"],
                "zogensa": ["-4"],
                "kisyucode": ["01056"],
                "chokyosicode": ["01023"],
                "harontimel3": ["33.5"],
                "timedifn": ["0.3"],
                "jyuni1c": ["2"],
                "jyuni4c": ["3"],
                "honsyokin": ["0"],
                "kyakusitukubun": ["0"],
                "race_id": ["2024032405030208"],
            }
        )

        result = etl_entries(mock_engine, "20240101", "20241231")

        assert result == 5
        call_args = mock_insert.call_args
        assert call_args[0][2] == "entries"
        assert call_args[0][3] == "raw"
        inserted_df = call_args[0][1]
        assert "ketto_num" in inserted_df.columns
        assert "finish_pos" in inserted_df.columns
        assert "win_odds" in inserted_df.columns
        assert "race_id" in inserted_df.columns
        assert "time_diff" in inserted_df.columns
        assert "corner_1c" in inserted_df.columns
        assert "corner_4c" in inserted_df.columns
        assert inserted_df["win_odds"].iloc[0] == pytest.approx(5.4)
        assert inserted_df["time_diff"].iloc[0] == pytest.approx(0.3)
        assert inserted_df["corner_1c"].iloc[0] == 2
        assert inserted_df["corner_4c"].iloc[0] == 3


class TestEtlPayouts:
    @patch("db.etl._insert_on_conflict")
    @patch("db.etl.pd.read_sql")
    def test_empty_result(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame()

        result = etl_payouts(mock_engine, "20240101", "20241231")

        assert result == 0
        mock_insert.assert_not_called()

    @patch("db.etl._insert_on_conflict", return_value=2)
    @patch("db.etl.pd.read_sql")
    def test_column_mapping(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "paytansyoumaban1": ["3"],
                "paytansyopay1": ["540"],
                "payfukusyoumaban1": ["3"],
                "payfukusyopay1": ["140"],
                "payfukusyoumaban2": ["7"],
                "payfukusyopay2": ["240"],
                "payfukusyoumaban3": ["1"],
                "payfukusyopay3": ["120"],
                "payfukusyoumaban4": ["5"],
                "payfukusyopay4": ["360"],
                "payfukusyoumaban5": ["9"],
                "payfukusyopay5": ["890"],
                "race_id": ["2024032405030208"],
            }
        )

        result = etl_payouts(mock_engine, "20240101", "20241231")

        assert result == 2
        call_args = mock_insert.call_args
        assert call_args[0][2] == "payouts"
        inserted_df = call_args[0][1]
        assert "tan_umaban" in inserted_df.columns
        assert "tan_pay" in inserted_df.columns
        assert "fuku_umaban1" in inserted_df.columns
        assert "fuku_pay5" in inserted_df.columns


class TestEtlOddsSnapshots:
    @patch("db.etl._insert_on_conflict")
    @patch("db.etl.pd.read_sql")
    def test_empty_result(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame()

        result = etl_odds_snapshots(mock_engine, "20240101", "20241231")

        assert result == 0
        mock_insert.assert_not_called()

    @patch("db.etl._insert_on_conflict", return_value=8)
    @patch("db.etl.pd.read_sql")
    def test_column_mapping(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "umaban": ["1", "2"],
                "tanodds": ["0032", "0054"],
                "fukuoddslow": ["0013", "0021"],
                "race_id": ["2024032405030208", "2024032405030208"],
            }
        )

        result = etl_odds_snapshots(mock_engine, "20240101", "20241231")

        assert result == 8
        call_args = mock_insert.call_args
        inserted_df = call_args[0][1]
        assert list(inserted_df.columns) == ["race_id", "umaban", "tan_odds", "fuku_odds"]
        assert inserted_df["tan_odds"].iloc[0] == pytest.approx(3.2)
        assert inserted_df["fuku_odds"].iloc[0] == pytest.approx(1.3)


class TestEtlWideOdds:
    @patch("db.etl._insert_on_conflict")
    @patch("db.etl.pd.read_sql")
    def test_empty_result(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame()

        result = etl_wide_odds(mock_engine, "20240101", "20241231")

        assert result == 0
        mock_insert.assert_not_called()

    @patch("db.etl._insert_on_conflict", return_value=6)
    @patch("db.etl.pd.read_sql")
    def test_column_mapping(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "kumi": ["1-2", "1-3"],
                "oddslow": ["00320", "00510"],
                "oddshigh": ["00450", "00780"],
                "race_id": ["2024032405030208", "2024032405030208"],
            }
        )

        result = etl_wide_odds(mock_engine, "20240101", "20241231")

        assert result == 6
        call_args = mock_insert.call_args
        inserted_df = call_args[0][1]
        assert list(inserted_df.columns) == ["race_id", "kumi", "odds_low", "odds_high"]
        assert inserted_df["odds_low"].iloc[0] == pytest.approx(3.20)
        assert inserted_df["odds_high"].iloc[0] == pytest.approx(4.50)


class TestEtlOddsTimeseries:
    @patch("db.etl._insert_on_conflict")
    @patch("db.etl.pd.read_sql")
    def test_empty_result(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame()

        result = etl_odds_timeseries(mock_engine, "20240101", "20241231")

        assert result == 0
        mock_insert.assert_not_called()

    @patch("db.etl._insert_on_conflict", return_value=100)
    @patch("db.etl.pd.read_sql")
    def test_year_by_year_loading(self, mock_read_sql, mock_insert):
        """複数年にまたがる場合、yearごとにクエリが発行される"""
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "happyotime": ["03241505"],
                "umaban": ["1"],
                "tanodds": ["0032"],
                "fukuoddslow": ["0013"],
                "tanninki": ["1"],
                "race_id": ["2024032405030208"],
            }
        )

        # 2023-01-01 ~ 2024-12-31 -> 2年分
        result = etl_odds_timeseries(mock_engine, "20230101", "20241231")

        assert result == 200  # 2年 x 100件
        assert mock_read_sql.call_count == 2
        assert mock_insert.call_count == 2

    @patch("db.etl._insert_on_conflict", return_value=50)
    @patch("db.etl.pd.read_sql")
    def test_column_mapping(self, mock_read_sql, mock_insert):
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "happyotime": ["03241505", "03241600"],
                "umaban": ["1", "2"],
                "tanodds": ["0032", "0054"],
                "fukuoddslow": ["0013", "0021"],
                "tanninki": ["1", "2"],
                "race_id": ["2024032405030208", "2024032405030208"],
            }
        )

        result = etl_odds_timeseries(mock_engine, "20240101", "20241231")

        assert result == 50
        call_args = mock_insert.call_args
        inserted_df = call_args[0][1]
        assert "ninki" in inserted_df.columns
        assert "happyo_time" in inserted_df.columns
        assert "tan_odds" in inserted_df.columns


# ---------------------------------------------------------------------------
# run_full_etl integration test (mock-based)
# ---------------------------------------------------------------------------


class TestRunFullEtl:
    @patch("db.etl.etl_odds_timeseries", return_value=100)
    @patch("db.etl.etl_wide_odds", return_value=80)
    @patch("db.etl.etl_odds_snapshots", return_value=70)
    @patch("db.etl.etl_payouts", return_value=60)
    @patch("db.etl.etl_entries", return_value=50)
    @patch("db.etl.etl_races", return_value=40)
    @patch("db.etl.create_project_schemas")
    def test_calls_all_etl_functions(
        self,
        mock_create_schemas,
        mock_races,
        mock_entries,
        mock_payouts,
        mock_snapshots,
        mock_wide,
        mock_timeseries,
    ):
        mock_engine = MagicMock()

        result = run_full_etl(mock_engine, "20240101", "20241231")

        mock_create_schemas.assert_called_once_with(mock_engine)
        mock_races.assert_called_once_with(mock_engine, "20240101", "20241231")
        mock_entries.assert_called_once_with(mock_engine, "20240101", "20241231")
        mock_payouts.assert_called_once_with(mock_engine, "20240101", "20241231")
        mock_snapshots.assert_called_once_with(mock_engine, "20240101", "20241231")
        mock_wide.assert_called_once_with(mock_engine, "20240101", "20241231")
        mock_timeseries.assert_called_once_with(mock_engine, "20240101", "20241231")

        assert result == {
            "raw.races": 40,
            "raw.entries": 50,
            "raw.payouts": 60,
            "odds_history.odds_snapshots": 70,
            "odds_history.wide_odds": 80,
            "odds_history.odds_time_series": 100,
        }

    @patch("db.etl.etl_odds_timeseries", return_value=0)
    @patch("db.etl.etl_wide_odds", return_value=0)
    @patch("db.etl.etl_odds_snapshots", return_value=0)
    @patch("db.etl.etl_payouts", return_value=0)
    @patch("db.etl.etl_entries", return_value=0)
    @patch("db.etl.etl_races", return_value=0)
    @patch("db.etl.create_project_schemas")
    def test_returns_zero_counts_for_no_data(
        self,
        mock_create_schemas,
        mock_races,
        mock_entries,
        mock_payouts,
        mock_snapshots,
        mock_wide,
        mock_timeseries,
    ):
        mock_engine = MagicMock()

        result = run_full_etl(mock_engine, "20990101", "20991231")

        assert all(v == 0 for v in result.values())


# ---------------------------------------------------------------------------
# run_full_etl_to_parquet tests
# ---------------------------------------------------------------------------


class TestRunFullEtlToParquet:
    @patch("db.etl.pd.read_sql")
    def test_writes_to_parquet_store(self, mock_read_sql) -> None:
        """Verify run_full_etl_to_parquet writes to ParquetStore, not PostgreSQL"""
        from db.etl import run_full_etl_to_parquet

        mock_engine = MagicMock()
        mock_store = MagicMock()

        # Return empty for all SQL queries
        mock_read_sql.return_value = pd.DataFrame()

        result = run_full_etl_to_parquet(mock_engine, mock_store, "20240101", "20241231")

        # Verify store.write was NOT called (empty data)
        mock_store.write.assert_not_called()
        assert result["races"] == 0
        assert result["entries"] == 0
