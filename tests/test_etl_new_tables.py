"""src/db/etl 新規Parquet ETL関数のテスト (horses, jockey_stats, trainer_stats)"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from db.etl import (
    _etl_horses_to_parquet,
    _etl_jockey_stats_to_parquet,
    _etl_trainer_stats_to_parquet,
)


# ---------------------------------------------------------------------------
# _etl_horses_to_parquet
# ---------------------------------------------------------------------------


class TestEtlHorsesToParquet:
    @patch("db.etl.pd.read_sql")
    def test_empty_result_returns_zero(self, mock_read_sql) -> None:
        mock_engine = MagicMock()
        mock_store = MagicMock()
        mock_read_sql.return_value = pd.DataFrame()

        result = _etl_horses_to_parquet(mock_engine, mock_store)

        assert result == 0
        mock_store.write.assert_not_called()

    @patch("db.etl.pd.read_sql")
    def test_returns_row_count_and_writes_to_store(self, mock_read_sql) -> None:
        mock_engine = MagicMock()
        mock_store = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "kettonum": ["0001234567", "0001234568"],
                "ketto3infohansyokunum1": ["A001", "A002"],
                "ba1chakukaisu1": ["10", "5"],
                "ba1chakukaisu2": ["8", "3"],
                "ruikeihonsyoheiti": ["1500.5", "800.0"],
                "kyakusitu1": ["1", "2"],
                "chuochakukaisu1": ["4", "6"],
            }
        )

        result = _etl_horses_to_parquet(mock_engine, mock_store)

        assert result == 2
        assert isinstance(result, int)
        mock_store.write.assert_called_once()
        call_args = mock_store.write.call_args
        assert call_args[0][0] == "raw"
        assert call_args[0][1] == "horses"

    @patch("db.etl.pd.read_sql")
    def test_type_conversions(self, mock_read_sql) -> None:
        mock_engine = MagicMock()
        mock_store = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "kettonum": ["0001234567"],
                "ketto3infohansyokunum1": ["A001"],
                "ba1chakukaisu1": ["10"],
                "ruikeihonsyoheiti": ["1500.5"],
                "kyakusitu1": ["1"],
                "chuochakukaisu1": ["4"],
            }
        )

        _etl_horses_to_parquet(mock_engine, mock_store)

        written_df = mock_store.write.call_args[0][2]
        # chakukaisu columns should be int
        assert written_df["ba1chakukaisu1"].iloc[0] == 10
        # ruikeihonsyoheiti should be float
        assert written_df["ruikeihonsyoheiti"].iloc[0] == pytest.approx(1500.5)
        # bloodline columns should be str
        assert isinstance(written_df["ketto3infohansyokunum1"].iloc[0], str)
        # kyakusitu should be int
        assert written_df["kyakusitu1"].iloc[0] == 1

    @patch("db.etl.pd.read_sql")
    def test_empty_values_converted_to_none(self, mock_read_sql) -> None:
        mock_engine = MagicMock()
        mock_store = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "kettonum": ["0001234567"],
                "ketto3infohansyokunum1": ["A001"],
                "ba1chakukaisu1": [""],
                "ruikeihonsyoheiti": [""],
                "kyakusitu1": [""],
                "chuochakukaisu1": [""],
            }
        )

        _etl_horses_to_parquet(mock_engine, mock_store)

        written_df = mock_store.write.call_args[0][2]
        assert pd.isna(written_df["ba1chakukaisu1"].iloc[0])
        assert pd.isna(written_df["ruikeihonsyoheiti"].iloc[0])
        assert pd.isna(written_df["kyakusitu1"].iloc[0])


# ---------------------------------------------------------------------------
# _etl_jockey_stats_to_parquet
# ---------------------------------------------------------------------------


class TestEtlJockeyStatsToParquet:
    @patch("db.etl.pd.read_sql")
    def test_empty_result_returns_zero(self, mock_read_sql) -> None:
        mock_engine = MagicMock()
        mock_store = MagicMock()
        mock_read_sql.return_value = pd.DataFrame()

        result = _etl_jockey_stats_to_parquet(mock_engine, mock_store)

        assert result == 0
        mock_store.write.assert_not_called()

    @patch("db.etl.pd.read_sql")
    def test_returns_row_count_and_writes_to_store(self, mock_read_sql) -> None:
        mock_engine = MagicMock()
        mock_store = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "setyear": ["2024", "2024"],
                "kisyucode": ["01056", "01111"],
                "heichichakukaisu1": ["50", "30"],
                "jyo1chakukaisu1": ["10", "5"],
                "kyori1chakukaisu1": ["8", "4"],
                "honsyokinheichi": ["50000.0", "30000.0"],
            }
        )

        result = _etl_jockey_stats_to_parquet(mock_engine, mock_store)

        assert result == 2
        assert isinstance(result, int)
        mock_store.write.assert_called_once()
        call_args = mock_store.write.call_args
        assert call_args[0][0] == "raw"
        assert call_args[0][1] == "jockey_stats"

    @patch("db.etl.pd.read_sql")
    def test_type_conversions(self, mock_read_sql) -> None:
        mock_engine = MagicMock()
        mock_store = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "setyear": ["2024"],
                "kisyucode": ["01056"],
                "heichichakukaisu1": ["50"],
                "honsyokinheichi": ["50000.0"],
            }
        )

        _etl_jockey_stats_to_parquet(mock_engine, mock_store)

        written_df = mock_store.write.call_args[0][2]
        # setyear should be int
        assert written_df["setyear"].iloc[0] == 2024
        # chakukaisu should be int
        assert written_df["heichichakukaisu1"].iloc[0] == 50
        # honsyokinheichi should be float
        assert written_df["honsyokinheichi"].iloc[0] == pytest.approx(50000.0)


# ---------------------------------------------------------------------------
# _etl_trainer_stats_to_parquet
# ---------------------------------------------------------------------------


class TestEtlTrainerStatsToParquet:
    @patch("db.etl.pd.read_sql")
    def test_empty_result_returns_zero(self, mock_read_sql) -> None:
        mock_engine = MagicMock()
        mock_store = MagicMock()
        mock_read_sql.return_value = pd.DataFrame()

        result = _etl_trainer_stats_to_parquet(mock_engine, mock_store)

        assert result == 0
        mock_store.write.assert_not_called()

    @patch("db.etl.pd.read_sql")
    def test_returns_row_count_and_writes_to_store(self, mock_read_sql) -> None:
        mock_engine = MagicMock()
        mock_store = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "setyear": ["2024", "2023"],
                "chokyosicode": ["01023", "01024"],
                "heichichakukaisu1": ["40", "25"],
                "jyo1chakukaisu1": ["8", "5"],
                "kyori1chakukaisu1": ["6", "3"],
                "honsyokinheichi": ["40000.0", "20000.0"],
            }
        )

        result = _etl_trainer_stats_to_parquet(mock_engine, mock_store)

        assert result == 2
        assert isinstance(result, int)
        mock_store.write.assert_called_once()
        call_args = mock_store.write.call_args
        assert call_args[0][0] == "raw"
        assert call_args[0][1] == "trainer_stats"

    @patch("db.etl.pd.read_sql")
    def test_type_conversions(self, mock_read_sql) -> None:
        mock_engine = MagicMock()
        mock_store = MagicMock()
        mock_read_sql.return_value = pd.DataFrame(
            {
                "setyear": ["2024"],
                "chokyosicode": ["01023"],
                "heichichakukaisu1": ["40"],
                "honsyokinheichi": ["40000.0"],
            }
        )

        _etl_trainer_stats_to_parquet(mock_engine, mock_store)

        written_df = mock_store.write.call_args[0][2]
        assert written_df["setyear"].iloc[0] == 2024
        assert written_df["heichichakukaisu1"].iloc[0] == 40
        assert written_df["honsyokinheichi"].iloc[0] == pytest.approx(40000.0)


# ---------------------------------------------------------------------------
# Integration: run_full_etl_to_parquet calls new functions
# ---------------------------------------------------------------------------


class TestRunFullEtlToParquetNewTables:
    @patch("db.etl.pd.read_sql")
    def test_includes_new_tables_in_counts(self, mock_read_sql) -> None:
        """Verify run_full_etl_to_parquet includes horses, jockey_stats, trainer_stats"""
        from db.etl import run_full_etl_to_parquet

        mock_engine = MagicMock()
        mock_store = MagicMock()

        # Return empty for all SQL queries
        mock_read_sql.return_value = pd.DataFrame()

        result = run_full_etl_to_parquet(mock_engine, mock_store, "20240101", "20241231")

        assert "horses" in result
        assert "jockey_stats" in result
        assert "trainer_stats" in result
