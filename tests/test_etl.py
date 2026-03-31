"""Generic ETL engine tests (mock-based, no DB required)"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml


class TestLoadTableConfig:
    def test_loads_yaml_and_returns_list(self, tmp_path: Path):
        config_file = tmp_path / "etl_tables.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "tables": [
                        {
                            "db_table": "n_race",
                            "parquet_key": "races",
                            "category": "raw",
                            "type": "raced",
                            "pk": ["year", "monthday", "jyocd"],
                        },
                    ]
                }
            )
        )
        from db.etl import load_table_config

        result = load_table_config(str(config_file))
        assert len(result) == 1
        assert result[0]["db_table"] == "n_race"
        assert result[0]["type"] == "raced"

    def test_raises_on_missing_file(self):
        from db.etl import load_table_config

        with pytest.raises(FileNotFoundError):
            load_table_config("/nonexistent/path.yaml")


class TestReadDbTable:
    @patch("db.etl.pd.read_sql")
    def test_raced_type_adds_date_filter(self, mock_read_sql):
        """type=raced のテーブルは WHERE (year||monthday)::int BETWEEN :start AND :end を付ける"""
        from db.etl import _read_db_table

        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame({"year": ["2024"], "monthday": ["0101"]})

        cfg = {"db_table": "n_race", "type": "raced"}
        _read_db_table(mock_engine, cfg, start="20240101", end="20241231")

        sql_text = mock_read_sql.call_args[0][0].text
        assert "BETWEEN" in sql_text
        assert ":start" in sql_text
        assert ":end" in sql_text
        params = mock_read_sql.call_args[1].get("params", {})
        assert params.get("start") == 20240101
        assert params.get("end") == 20241231

    @patch("db.etl.pd.read_sql")
    def test_master_type_no_date_filter(self, mock_read_sql):
        """type=master のテーブルは SELECT * FROM table のみ (日付フィルタなし)"""
        from db.etl import _read_db_table

        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame({"kettonum": ["123"]})

        cfg = {"db_table": "n_uma", "type": "master"}
        _read_db_table(mock_engine, cfg)

        sql_text = mock_read_sql.call_args[0][0].text
        assert "BETWEEN" not in sql_text
        assert "FROM n_uma" in sql_text

    @patch("db.etl.pd.read_sql")
    def test_delta_type_no_date_filter(self, mock_read_sql):
        """type=delta のテーブルも SELECT * FROM table のみ"""
        from db.etl import _read_db_table

        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame({"id": ["1"], "datakubun": ["1"]})

        cfg = {"db_table": "s_race", "type": "delta"}
        _read_db_table(mock_engine, cfg)

        sql_text = mock_read_sql.call_args[0][0].text
        assert "BETWEEN" not in sql_text


class TestComputeRaceDate:
    def test_adds_race_date_column(self):
        from db.etl import _compute_race_date

        df = pd.DataFrame({"year": ["2024"], "monthday": ["0324"]})
        result = _compute_race_date(df)
        assert "race_date" in result.columns
        assert result["race_date"].iloc[0] == pd.Timestamp("2024-03-24")

    def test_preserves_existing_columns(self):
        from db.etl import _compute_race_date

        df = pd.DataFrame({"year": ["2024"], "monthday": ["0101"], "jyocd": ["05"]})
        result = _compute_race_date(df)
        assert "jyocd" in result.columns

    def test_skips_when_no_year_monthday(self):
        from db.etl import _compute_race_date

        df = pd.DataFrame({"kettonum": ["123"]})
        result = _compute_race_date(df)
        assert "race_date" not in result.columns


class TestComputeRaceId:
    def test_computes_16_digit_race_id(self):
        from db.etl import _compute_race_id

        df = pd.DataFrame(
            {
                "year": ["2024"],
                "monthday": ["0324"],
                "jyocd": ["05"],
                "kaiji": ["03"],
                "nichiji": ["02"],
                "racenum": ["08"],
            }
        )
        result = _compute_race_id(df)
        assert result["race_id"].iloc[0] == "2024032405030208"

    def test_zfill_padding(self):
        from db.etl import _compute_race_id

        df = pd.DataFrame(
            {
                "year": ["2024"],
                "monthday": ["101"],
                "jyocd": ["1"],
                "kaiji": ["1"],
                "nichiji": ["1"],
                "racenum": ["1"],
            }
        )
        result = _compute_race_id(df)
        assert result["race_id"].iloc[0] == "2024010101010101"


class TestMergeDelta:
    def test_upsert_replaces_existing_row(self):
        from db.etl import _merge_delta

        existing = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        delta = pd.DataFrame({"id": [2], "val": ["B"], "datakubun": ["1"]})
        result = _merge_delta(existing, delta, pk=["id"])
        assert len(result) == 2
        assert result[result["id"] == 2]["val"].iloc[0] == "B"

    def test_upsert_inserts_new_row(self):
        from db.etl import _merge_delta

        existing = pd.DataFrame({"id": [1], "val": ["a"]})
        delta = pd.DataFrame({"id": [2], "val": ["b"], "datakubun": ["1"]})
        result = _merge_delta(existing, delta, pk=["id"])
        assert len(result) == 2

    def test_delete_removes_row(self):
        from db.etl import _merge_delta

        existing = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        delta = pd.DataFrame({"id": [2], "val": ["x"], "datakubun": ["0"]})
        result = _merge_delta(existing, delta, pk=["id"])
        assert len(result) == 2
        assert 2 not in result["id"].values

    def test_composite_pk(self):
        from db.etl import _merge_delta

        existing = pd.DataFrame(
            {
                "year": ["2024", "2024"],
                "monthday": ["0101", "0102"],
                "val": ["a", "b"],
            }
        )
        delta = pd.DataFrame(
            {
                "year": ["2024"],
                "monthday": ["0101"],
                "val": ["A"],
                "datakubun": ["1"],
            }
        )
        result = _merge_delta(existing, delta, pk=["year", "monthday"])
        assert len(result) == 2
        assert result[(result["year"] == "2024") & (result["monthday"] == "0101")]["val"].iloc[0] == "A"

    def test_empty_delta_returns_existing(self):
        from db.etl import _merge_delta

        existing = pd.DataFrame({"id": [1], "val": ["a"]})
        delta = pd.DataFrame({"id": [], "val": [], "datakubun": []})
        result = _merge_delta(existing, delta, pk=["id"])
        assert len(result) == 1

    def test_datakubun_stripped_from_upserts(self):
        from db.etl import _merge_delta

        existing = pd.DataFrame({"id": [1], "val": ["a"]})
        delta = pd.DataFrame({"id": [2], "val": ["b"], "datakubun": ["1"]})
        result = _merge_delta(existing, delta, pk=["id"])
        assert "datakubun" not in result.columns


class TestRunFullLoad:
    @patch("db.etl._read_db_table")
    def test_processes_all_raced_and_master_tables(self, mock_read):
        """Full load processes raced + master tables, skips delta tables"""
        from db.etl import run_full_load

        mock_store = MagicMock()
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame(
            {"year": ["2024"], "monthday": ["0101"], "col1": ["val"]}
        )

        config = [
            {
                "db_table": "n_race",
                "parquet_key": "races",
                "category": "raw",
                "type": "raced",
                "pk": ["year", "monthday"],
            },
            {
                "db_table": "n_uma",
                "parquet_key": "horses",
                "category": "raw",
                "type": "master",
                "pk": ["kettonum"],
            },
            {
                "db_table": "s_race",
                "parquet_key": "races",
                "category": "raw",
                "type": "delta",
                "pk": ["year", "monthday"],
            },
        ]

        result = run_full_load(mock_store, mock_engine, config, "20240101", "20241231")

        # Should have processed 2 tables (races + horses), skipped 1 (delta)
        assert "races" in result
        assert "horses" in result
        assert mock_store.write.call_count == 2

    @patch("db.etl._read_db_table")
    def test_table_filter_limits_scope(self, mock_read):
        """--tables filter limits which tables are processed"""
        from db.etl import run_full_load

        mock_store = MagicMock()
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame({"year": ["2024"], "monthday": ["0101"]})

        config = [
            {
                "db_table": "n_race",
                "parquet_key": "races",
                "category": "raw",
                "type": "raced",
                "pk": ["year"],
            },
            {
                "db_table": "n_uma",
                "parquet_key": "horses",
                "category": "raw",
                "type": "master",
                "pk": ["kettonum"],
            },
        ]

        result = run_full_load(
            mock_store, mock_engine, config, "20240101", "20241231", tables=["races"]
        )

        assert "races" in result
        assert "horses" not in result

    @patch("db.etl._read_db_table")
    def test_raced_table_gets_race_date_and_race_id(self, mock_read):
        """type=raced テーブルは race_date と race_id を付与"""
        from db.etl import run_full_load

        mock_store = MagicMock()
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame(
            {
                "year": ["2024"],
                "monthday": ["0101"],
                "jyocd": ["05"],
                "kaiji": ["01"],
                "nichiji": ["01"],
                "racenum": ["01"],
            }
        )

        config = [
            {
                "db_table": "n_race",
                "parquet_key": "races",
                "category": "raw",
                "type": "raced",
                "pk": ["year", "monthday"],
            },
        ]

        result = run_full_load(mock_store, mock_engine, config, "20240101", "20241231")

        written_df = mock_store.write.call_args[0][2]
        assert "race_date" in written_df.columns
        assert "race_id" in written_df.columns
        assert written_df["race_id"].iloc[0] == "2024010105010101"

    @patch("db.etl._read_db_table")
    def test_master_table_no_race_date(self, mock_read):
        """type=master テーブルは race_date / race_id を付与しない"""
        from db.etl import run_full_load

        mock_store = MagicMock()
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame({"kettonum": ["123"]})

        config = [
            {
                "db_table": "n_uma",
                "parquet_key": "horses",
                "category": "raw",
                "type": "master",
                "pk": ["kettonum"],
            },
        ]

        run_full_load(mock_store, mock_engine, config, "20240101", "20241231")

        written_df = mock_store.write.call_args[0][2]
        assert "race_date" not in written_df.columns
        assert "race_id" not in written_df.columns


class TestRunDeltaUpdate:
    @patch("db.etl._read_db_table")
    def test_skips_when_no_existing_parquet(self, mock_read):
        """Delta skips tables with no existing Parquet file"""
        from db.etl import run_delta_update

        mock_store = MagicMock()
        mock_store.exists.return_value = False
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame({"id": [1], "datakubun": ["1"]})

        config = [
            {
                "db_table": "s_race",
                "parquet_key": "races",
                "category": "raw",
                "type": "delta",
                "pk": ["id"],
            },
        ]

        result = run_delta_update(mock_store, mock_engine, config)
        assert result["races"] == -1  # skipped

    @patch("db.etl._read_db_table")
    def test_merges_delta_into_existing(self, mock_read):
        """Delta merges s_ data into existing Parquet"""
        from db.etl import run_delta_update

        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_store.read.return_value = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame({"id": [2], "val": ["B"], "datakubun": ["1"]})

        config = [
            {
                "db_table": "s_race",
                "parquet_key": "races",
                "category": "raw",
                "type": "delta",
                "pk": ["id"],
            },
        ]

        result = run_delta_update(mock_store, mock_engine, config)
        assert result["races"] == 1  # 1 delta row processed
        mock_store.write.assert_called_once()
