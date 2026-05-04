"""src/db モジュールのテスト（モックDB使用・実際のDB接続不要）"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd

from db.connection import DatabaseConnection, _compute_race_date, _compute_race_id
from db.schema import (
    ALL_CREATE_STATEMENTS,
    SCHEMA_BETTING,
    SCHEMA_FEATURE,
    SCHEMA_ODDS_HISTORY,
    SCHEMA_PREDICTION,
    SCHEMA_RAW,
)


class TestSchemaDefinitions:
    def test_raw_schema_contains_races_table(self):
        assert "CREATE TABLE IF NOT EXISTS raw.races" in SCHEMA_RAW

    def test_raw_schema_contains_entries_table(self):
        assert "CREATE TABLE IF NOT EXISTS raw.entries" in SCHEMA_RAW

    def test_odds_history_schema_contains_snapshots_table(self):
        assert "CREATE TABLE IF NOT EXISTS odds_history.odds_snapshots" in SCHEMA_ODDS_HISTORY

    def test_odds_history_schema_contains_time_series_table(self):
        assert "CREATE TABLE IF NOT EXISTS odds_history.odds_time_series" in SCHEMA_ODDS_HISTORY

    def test_feature_schema_exists(self):
        assert "CREATE TABLE IF NOT EXISTS feature.features" in SCHEMA_FEATURE

    def test_prediction_schema_exists(self):
        assert "CREATE TABLE IF NOT EXISTS prediction.predictions" in SCHEMA_PREDICTION

    def test_betting_schema_contains_bets_table(self):
        assert "CREATE TABLE IF NOT EXISTS betting.bets" in SCHEMA_BETTING

    def test_all_schemas_list(self):
        assert len(ALL_CREATE_STATEMENTS) == 5

    def test_race_primary_key(self):
        """複合主キーの確認"""
        assert "PRIMARY KEY (year, month_day, jyo_cd, kaiji, nichiji, race_num)" in SCHEMA_RAW

    def test_entries_foreign_key(self):
        assert "REFERENCES raw.races" in SCHEMA_RAW

    def test_raw_schema_race_id_generated(self):
        """race_id は GENERATED ALWAYS AS で複合PKから自動生成"""
        assert "race_id" in SCHEMA_RAW
        assert "GENERATED ALWAYS AS" in SCHEMA_RAW
        assert "UNIQUE" in SCHEMA_RAW

    def test_raw_schema_surface_computed_column(self):
        """surface は GENERATED COLUMN で計算"""
        assert "surface" in SCHEMA_RAW


class TestDatabaseConnection:
    def test_connection_string_from_settings(self):
        """settings.yaml から接続文字列を正しく生成"""
        mock_settings = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "dbname": "everydb2",
                "user": "postgres",
                "password": "",
            }
        }
        env_without_pg = {k: v for k, v in os.environ.items() if k != "PGPASSWORD"}
        with (
            patch.dict(os.environ, env_without_pg, clear=True),
            patch("db.connection._load_settings", return_value=mock_settings),
        ):
            conn = DatabaseConnection()
            expected = "postgresql+psycopg2://postgres@localhost:5432/everydb2"
            assert conn._connection_url == expected

    def test_connection_string_with_password(self):
        mock_settings = {
            "database": {
                "host": "db.example.com",
                "port": 5433,
                "dbname": "everydb2",
                "user": "app_user",
                "password": "secret",
            }
        }
        env_without_pg = {k: v for k, v in os.environ.items() if k != "PGPASSWORD"}
        with (
            patch.dict(os.environ, env_without_pg, clear=True),
            patch("db.connection._load_settings", return_value=mock_settings),
        ):
            conn = DatabaseConnection()
            expected = "postgresql+psycopg2://app_user:secret@db.example.com:5433/everydb2"
            assert conn._connection_url == expected

    def test_connection_string_uses_env_password(self):
        """環境変数 PGPASSWORD で password を上書き"""
        mock_settings = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "dbname": "everydb2",
                "user": "postgres",
                "password": "",
            }
        }
        with (
            patch.dict(os.environ, {"PGPASSWORD": "env_secret"}),
            patch("db.connection._load_settings", return_value=mock_settings),
        ):
            conn = DatabaseConnection()
            assert "env_secret" in conn._connection_url
            assert conn._connection_url.startswith("postgresql+psycopg2://postgres:env_secret@")

    def test_get_engine_returns_engine(self):
        """engine はキャッシュされる"""
        mock_settings = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "dbname": "everydb2",
                "user": "postgres",
                "password": "",
            }
        }
        with (
            patch("db.connection._load_settings", return_value=mock_settings),
            patch("db.connection.create_engine") as mock_create_engine,
        ):
            conn = DatabaseConnection()
            engine1 = conn.get_engine()
            engine2 = conn.get_engine()
            mock_create_engine.assert_called_once()
            assert engine1 is engine2

    def test_create_schemas_executes_all(self):
        """全ステートメントが個別に実行される（DDL分割対応）

        SCHEMA_RAW: 4文, SCHEMA_ODDS_HISTORY: 4文, SCHEMA_FEATURE: 2文,
        SCHEMA_PREDICTION: 2文, SCHEMA_BETTING: 5文 = 合計17文
        """
        mock_settings = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "dbname": "everydb2",
                "user": "postgres",
                "password": "",
            }
        }
        with (
            patch("db.connection._load_settings", return_value=mock_settings),
            patch("db.connection.create_engine") as mock_create_engine,
        ):
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            conn = DatabaseConnection()
            conn.create_schemas()

            # 17個の個別SQLステートメントが実行される
            assert mock_engine.begin.call_count == 17


class TestComputeHelpers:
    def test_compute_race_id(self) -> None:
        df = pd.DataFrame(
            {
                "year": ["2020"],
                "monthday": ["0101"],
                "jyocd": ["5"],
                "kaiji": ["1"],
                "nichiji": ["1"],
                "racenum": ["11"],
            }
        )
        result = _compute_race_id(df)
        assert result["race_id"].iloc[0] == "2020010105010111"

    def test_compute_race_date(self) -> None:
        df = pd.DataFrame({"year": ["2020"], "monthday": ["0315"]})
        result = _compute_race_date(df)
        assert result["race_date"].iloc[0] == pd.Timestamp("2020-03-15")
