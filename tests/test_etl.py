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
