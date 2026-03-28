"""PostgreSQL DB接続モジュール（SQLAlchemy Core 使用）"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from db.schema import ALL_CREATE_STATEMENTS

if TYPE_CHECKING:
    import pandas as pd

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


def _compute_race_id(df: "pd.DataFrame") -> "pd.DataFrame":
    """year + month_day + jyo_cd + kaiji + nichiji + race_num → race_id (16桁)"""
    df["race_id"] = (
        df["year"].astype(str).str.zfill(4)
        + df["month_day"].astype(str).str.zfill(4)
        + df["jyo_cd"].astype(str).str.zfill(2)
        + df["kaiji"].astype(str).str.zfill(2)
        + df["nichiji"].astype(str).str.zfill(2)
        + df["race_num"].astype(str).str.zfill(2)
    )
    return df


def _compute_race_date(df: "pd.DataFrame") -> "pd.DataFrame":
    """year + month_day → race_date (datetime64)

    注意: month_day は int (例: 101) または str (例: "0101") の両方に対応。
    ETL直後は int (101) → zfill で4桁に。
    """
    import pandas as pd

    month_day_str = df["month_day"].astype(str).str.zfill(4)
    year_str = df["year"].astype(str).str.zfill(4)
    df["race_date"] = pd.to_datetime(year_str + month_day_str, format="%Y%m%d")
    return df


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

    def load_races(self, start_date: str, end_date: str) -> "pd.DataFrame":
        """指定期間のレースデータをDataFrameで取得"""
        import pandas as pd

        engine = self.get_engine()
        sql = text("""
            SELECT * FROM raw.races
            WHERE (year || month_day)::int BETWEEN :start AND :end
            AND track_cd NOT BETWEEN 51 AND 59
            ORDER BY year, month_day, jyo_cd, kaiji, nichiji, race_num
        """)
        df = pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})
        df["race_date"] = pd.to_datetime(df["year"].astype(str) + df["month_day"], format="%Y%m%d")
        return df

    def load_entries_with_results(self, start_date: str, end_date: str) -> "pd.DataFrame":
        """指定期間の出走馬データをDataFrameで取得"""
        import pandas as pd

        engine = self.get_engine()
        sql = text("""
            SELECT e.*, r.year, r.month_day
            FROM raw.entries e
            JOIN raw.races r ON e.race_id = r.race_id
            WHERE (r.year || r.month_day)::int BETWEEN :start AND :end
            AND r.track_cd NOT BETWEEN 51 AND 59
            AND e.finish_pos > 0
            ORDER BY e.race_id, e.umaban
        """)
        df = pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})
        df["race_date"] = pd.to_datetime(df["year"].astype(str) + df["month_day"], format="%Y%m%d")
        return df

    def load_odds_snapshots(self, start_date: str, end_date: str) -> "pd.DataFrame":
        """指定期間のオッズスナップショットをDataFrameで取得"""
        import pandas as pd

        engine = self.get_engine()
        sql = text("""
            SELECT o.* FROM odds_history.odds_snapshots o
            JOIN raw.races r ON o.race_id = r.race_id
            WHERE (r.year || r.month_day)::int BETWEEN :start AND :end
            AND r.track_cd NOT BETWEEN 51 AND 59
            ORDER BY o.race_id, o.umaban
        """)
        return pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})

    def load_odds_time_series(self, race_id: str) -> "pd.DataFrame":
        """特定レースの時系列オッズを取得"""
        import pandas as pd

        engine = self.get_engine()
        sql = text("""
            SELECT * FROM odds_history.odds_time_series
            WHERE race_id = :race_id
            ORDER BY happyo_time, umaban
        """)
        return pd.read_sql(sql, engine, params={"race_id": race_id})

    def load_odds_time_series_range(self, start_date: str, end_date: str) -> "pd.DataFrame":
        """指定期間の時系列オッズをDataFrameで取得"""
        import pandas as pd

        engine = self.get_engine()
        sql = text("""
            SELECT o.* FROM odds_history.odds_time_series o
            JOIN raw.races r ON o.race_id = r.race_id
            WHERE (r.year || r.month_day)::int BETWEEN :start AND :end
            AND r.track_cd NOT BETWEEN 51 AND 59
            ORDER BY o.race_id, o.happyo_time, o.umaban
        """)
        return pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})

    def load_wide_odds(self, start_date: str, end_date: str) -> "pd.DataFrame":
        """指定期間のワイドオッズをDataFrameで取得"""
        import pandas as pd

        engine = self.get_engine()
        sql = text("""
            SELECT w.* FROM odds_history.wide_odds w
            JOIN raw.races r ON w.race_id = r.race_id
            WHERE (r.year || r.month_day)::int BETWEEN :start AND :end
            AND r.track_cd NOT BETWEEN 51 AND 59
            ORDER BY w.race_id, w.kumi
        """)
        return pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})

    def save_predictions(self, df: "pd.DataFrame") -> None:
        """予測結果を prediction.predictions に保存"""
        engine = self.get_engine()
        df.to_sql("predictions", engine, schema="prediction", if_exists="append", index=False)

    def save_bets(self, df: "pd.DataFrame") -> None:
        """投票記録を betting.bets に保存"""
        engine = self.get_engine()
        df.to_sql("bets", engine, schema="betting", if_exists="append", index=False)

    def etl_to_parquet(self, store: "ParquetStore", start: str, end: str) -> dict[str, int]:
        """EveryDB2外部テーブル → Parquet にETL。"""
        from db.etl import run_full_etl_to_parquet

        return run_full_etl_to_parquet(self.get_engine(), store, start, end)
