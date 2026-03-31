"""PostgreSQL DB接続モジュール（SQLAlchemy Core 使用）"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from db.etl import _compute_race_date, _compute_race_id  # noqa: F401
from db.schema import ALL_CREATE_STATEMENTS

if TYPE_CHECKING:

    from db.parquet_store import ParquetStore

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_settings(settings_path: Optional[Path] = None) -> dict:
    """config/settings.yaml をロード"""
    path = settings_path or _DEFAULT_SETTINGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class DatabaseConnection:
    """データベース接続を管理するクラス（シングルトンエンジン）"""

    def __init__(self, settings_path: Optional[str] = None):
        path = Path(settings_path) if settings_path else None
        settings = _load_settings(path)
        db = settings["database"]

        password = os.environ.get("PGPASSWORD", db.get("password", ""))
        if password:
            self._connection_url = (
                f"postgresql+psycopg2://{db['user']}:{password}"
                f"@{db['host']}:{db['port']}/{db['dbname']}"
            )
        else:
            self._connection_url = (
                f"postgresql+psycopg2://{db['user']}@{db['host']}:{db['port']}/{db['dbname']}"
            )
        self._engine: Optional[Engine] = None

    def get_engine(self) -> Engine:
        """SQLAlchemy エンジンを取得（キャッシュ）"""
        if self._engine is None:
            self._engine = create_engine(
                self._connection_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
        return self._engine

    def create_schemas(self) -> None:
        """全スキーマとテーブルを作成（冪等）

        SQLAlchemy text() は単一SQL文のみ実行可能なため、
        セミコロンで分割して個別に実行する。
        """
        engine = self.get_engine()
        for ddl in ALL_CREATE_STATEMENTS:
            for statement in ddl.split(";"):
                stmt = statement.strip()
                if stmt:
                    with engine.begin() as conn:
                        conn.execute(text(stmt))

    def etl_to_parquet(self, store: "ParquetStore", start: str, end: str) -> dict[str, int]:
        """EveryDB2外部テーブル → Parquet にETL。"""
        from db.etl import load_table_config, run_full_load

        config = load_table_config()
        return run_full_load(store, self.get_engine(), config, start, end)
