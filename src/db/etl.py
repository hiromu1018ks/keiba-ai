"""Generic ETL engine: EveryDB2 → Parquet (YAML-driven)"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

_SAFE_TABLE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "etl_tables.yaml"
_STATE_PATH = _PROJECT_ROOT / "data" / "etl_state.json"


def load_table_config(path: str = str(_DEFAULT_CONFIG_PATH)) -> list[dict]:
    """Load table definitions from YAML config."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["tables"]


def _read_db_table(
    engine: Engine,
    cfg: dict,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Read a table from PostgreSQL. Adds date filter for type=raced."""
    table = cfg["db_table"]
    if not _SAFE_TABLE.match(table):
        raise ValueError(f"Invalid table name: {table}")
    table_type = cfg.get("type", "master")

    if table_type == "raced" and start and end:
        sql = text(f"SELECT * FROM {table} WHERE (year || monthday)::int BETWEEN :start AND :end")
        return pd.read_sql(sql, engine, params={"start": int(start), "end": int(end)})

    return pd.read_sql(text(f"SELECT * FROM {table}"), engine)


def _compute_race_date(df: pd.DataFrame) -> pd.DataFrame:
    """Compute race_date from year + monthday columns."""
    if "year" in df.columns and "monthday" in df.columns:
        year_str = df["year"].astype(str).str.zfill(4)
        monthday_str = df["monthday"].astype(str).str.zfill(4)
        df["race_date"] = pd.to_datetime(year_str + monthday_str, format="%Y%m%d")
    return df


def _compute_race_id(df: pd.DataFrame) -> pd.DataFrame:
    """year + monthday + jyocd + kaiji + nichiji + racenum → 16桁 race_id"""
    required = ["year", "monthday", "jyocd", "kaiji", "nichiji", "racenum"]
    if all(c in df.columns for c in required):
        df["race_id"] = (
            df["year"].astype(str).str.zfill(4)
            + df["monthday"].astype(str).str.zfill(4)
            + df["jyocd"].astype(str).str.zfill(2)
            + df["kaiji"].astype(str).str.zfill(2)
            + df["nichiji"].astype(str).str.zfill(2)
            + df["racenum"].astype(str).str.zfill(2)
        )
    return df


_TABLE_TYPE_RULES: dict[str, dict[str, list[str] | dict | list[dict]]] = {
    "races": {
        "int": ["trackcd", "kyori", "tenkocd", "syussotosu", "honsyokin"],
        "sentinel_float": [
            # RA table HaronTimeL3/L4: race-level, sentinels 000/999, no divisor
            {"columns": ["harontimel3", "harontimel4"], "sentinels": ["000", "999"]},
            # RA table LapTime1~25: varchar(3), sentinels 000, divisor=10
            {
                "columns": [f"laptime{i}" for i in range(1, 26)],
                "sentinels": ["000"],
                "divisor": 10,
            },
        ],
    },
    "entries": {
        "int": [
            "umaban",
            "kakuteijyuni",
            "ninki",
            "kyakusitukubun",
            "jyuni1c",
            "jyuni4c",
            "zogenfugo",
        ],
        "float": ["time", "bataijyu", "zogensa", "timediff"],
        "odds10": ["odds"],
        "sentinel_float": {
            "columns": ["harontimel3", "harontimel4", "jyuni2c", "jyuni3c"],
            "sentinels": ["000", "999", "00"],
        },
    },
    "odds_tanpuku": {
        "int": ["umaban"],
        "odds10": ["tanodds", "fukuoddslow"],
    },
    "odds_wide": {
        "odds100": ["oddslow", "oddshigh"],
    },
    "jodds_tanpuku": {
        "int": ["umaban", "tanninki"],
        "odds10": ["tanodds", "fukuoddslow"],
    },
    "payouts": {
        "int": ["paytansyoumaban1"] + [f"payfukusyoumaban{i}" for i in range(1, 6)],
        "float": ["paytansyopay1"] + [f"payfukusyopay{i}" for i in range(1, 6)],
    },
}


def _apply_type_conversions(df: pd.DataFrame, table_key: str) -> pd.DataFrame:
    """Apply type conversions based on table key rules."""
    rules = _TABLE_TYPE_RULES.get(table_key)
    if rules is None:
        return df

    def _to_int(val: object) -> int | None:
        if val is None or val == "":
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _to_float(val: object) -> float | None:
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _to_odds(val: object, divisor: int) -> float | None:
        f = _to_float(val)
        return f / divisor if f is not None else None

    for col in rules.get("int", []):
        if col in df.columns:
            df[col] = df[col].apply(_to_int).astype("Int64")

    for col in rules.get("float", []):
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    for col in rules.get("odds10", []):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _to_odds(v, 10))

    for col in rules.get("odds100", []):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _to_odds(v, 100))

    # Sentinel float: replace sentinel strings with NaN, convert to float64
    _sentinel_float_rule = rules.get("sentinel_float")
    if _sentinel_float_rule is not None:
        _rule_list: list[dict] = (
            list(_sentinel_float_rule)
            if isinstance(_sentinel_float_rule, list)
            else [_sentinel_float_rule]  # type: ignore[list-item]
        )
        for _rule in _rule_list:
            _cols = _rule.get("columns", [])
            _sentinels = _rule.get("sentinels", [])
            _divisor = _rule.get("divisor", 1)
            for _col in _cols:
                if _col in df.columns:
                    df[_col] = df[_col].replace(_sentinels, float("nan"))
                    df[_col] = pd.to_numeric(df[_col], errors="coerce")
                    if _divisor != 1:
                        df[_col] = df[_col] / _divisor

    # Sentinel int: replace sentinel strings with NaN, convert to Int64
    _sentinel_int_rule = rules.get("sentinel_int")
    if _sentinel_int_rule is not None:
        _int_rules: list[dict] = (
            list(_sentinel_int_rule)
            if isinstance(_sentinel_int_rule, list)
            else [_sentinel_int_rule]  # type: ignore[list-item]
        )
        for _rule in _int_rules:
            _cols = _rule.get("columns", [])
            _sentinels = _rule.get("sentinels", [])
            for _col in _cols:
                if _col in df.columns:
                    df[_col] = df[_col].replace(_sentinels, float("nan"))
                    df[_col] = pd.to_numeric(df[_col], errors="coerce")
                    df[_col] = df[_col].astype("Int64")

    return df


def _compute_surface(df: pd.DataFrame) -> pd.DataFrame:
    """trackcd -> surface (turf/dirt/other)."""
    if "trackcd" in df.columns:
        df["surface"] = df["trackcd"].apply(
            lambda x: "turf" if 10 <= x <= 22 else "dirt" if 23 <= x <= 29 else "other"
        )
    return df


def _compute_track_condition_code(df: pd.DataFrame) -> pd.DataFrame:
    """sibababacd/dirtbabacd + trackcd -> track_condition_code.

    Turf(trackcd 10-22) uses sibababacd, dirt(23-29) uses dirtbabacd.
    """
    if "sibababacd" in df.columns and "dirtbabacd" in df.columns and "trackcd" in df.columns:
        import numpy as np

        is_turf = df["trackcd"].between(10, 22)
        df["track_condition_code"] = np.where(is_turf, df["sibababacd"], df["dirtbabacd"])
    return df


def _load_state() -> dict:
    """Load ETL state from JSON file."""
    if _STATE_PATH.exists():
        with open(_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"tables": {}}


def _save_state(state: dict) -> None:
    """Save ETL state to JSON file."""
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def run_full_load(
    store: ParquetStore,
    engine: Engine,
    config: list[dict],
    start: str,
    end: str,
    tables: list[str] | None = None,
) -> dict[str, int]:
    """Full load: read all n_ tables and write to Parquet."""
    state = _load_state()
    counts: dict[str, int] = {}

    # Filter to non-delta tables only
    full_configs = [c for c in config if c.get("type") != "delta"]

    # Optional table filter
    if tables:
        full_configs = [c for c in full_configs if c["parquet_key"] in tables]

    for cfg in tqdm(full_configs, desc="Full ETL"):
        key = cfg["parquet_key"]
        category = cfg["category"]
        table_type = cfg.get("type", "master")
        partition_cols = cfg.get("partition_cols")

        try:
            if partition_cols and table_type == "raced":
                # Year-by-year streaming write for large partitioned tables
                import pyarrow as pa
                import pyarrow.parquet as pq

                start_year = int(start) // 10000
                end_year = int(end) // 10000
                partition_path = store.data_dir / category / key
                # Clear existing partitioned data to avoid stale year partitions
                if partition_path.is_dir():
                    shutil.rmtree(partition_path)
                partition_path.mkdir(parents=True, exist_ok=True)

                total_rows = 0
                for year in range(start_year, end_year + 1):
                    sql = text(f"SELECT * FROM {cfg['db_table']} WHERE year = :year")
                    df = pd.read_sql(sql, engine, params={"year": str(year)})
                    if not df.empty:
                        _compute_race_date(df)
                        _compute_race_id(df)
                        df = _apply_type_conversions(df, key)
                        if table_type == "raced":
                            df = _compute_surface(df)
                            df = _compute_track_condition_code(df)
                        # Add partition columns from race_date
                        if "race_date" in df.columns:
                            df["year"] = df["race_date"].dt.year
                            df["month"] = df["race_date"].dt.month
                        table = pa.Table.from_pandas(df)
                        pq.write_to_dataset(
                            table, root_path=str(partition_path), partition_cols=partition_cols
                        )
                        n = len(df)
                        total_rows += n
                        logger.info("  %s year=%d: %d rows", key, year, n)
                        del df, table
                counts[key] = total_rows
            else:
                df = _read_db_table(
                    engine,
                    cfg,
                    start=start if table_type == "raced" else None,
                    end=end if table_type == "raced" else None,
                )
                if not df.empty:
                    if table_type == "raced":
                        _compute_race_date(df)
                        _compute_race_id(df)
                    df = _apply_type_conversions(df, key)
                    if table_type == "raced":
                        df = _compute_surface(df)
                        df = _compute_track_condition_code(df)
                    store.write(category, key, df)
                    counts[key] = len(df)
                else:
                    counts[key] = 0

            logger.info("Full load %s: %d rows", key, counts.get(key, 0))

        except Exception as e:
            logger.error("Full load failed for %s: %s", key, e)
            counts[key] = -1

        # Update state
        state["tables"][key] = {
            "rows": counts.get(key, 0),
            "last_full": pd.Timestamp.now().isoformat(),
        }

    _save_state(state)
    return counts


def _merge_delta(existing: pd.DataFrame, delta: pd.DataFrame, pk: list[str]) -> pd.DataFrame:
    """Merge delta records into existing DataFrame using PK-based upsert/delete.

    datakubun='0' → delete row (remove from existing)
    datakubun!='0' → upsert row (replace existing or insert new)
    If datakubun column is absent, treat all rows as upserts.
    """
    # Validate PK columns exist in both DataFrames
    missing_in_existing = [c for c in pk if c not in existing.columns]
    missing_in_delta = [c for c in pk if c not in delta.columns]
    if missing_in_existing or missing_in_delta:
        raise ValueError(
            f"PK columns mismatch: missing in existing={missing_in_existing}, "
            f"missing in delta={missing_in_delta}"
        )

    # Normalize PK columns to string for merge compatibility
    # (existing Parquet may have Int64 PKs while delta has string PKs from EveryDB2)
    for col in pk:
        existing = existing.copy()
        existing[col] = existing[col].astype(str)
        delta = delta.copy()
        delta[col] = delta[col].astype(str)

    # Split into deletes and upserts based on datakubun
    if "datakubun" in delta.columns:
        deletes = delta[delta["datakubun"] == "0"]
        upserts = delta[delta["datakubun"] != "0"].drop(columns=["datakubun"], errors="ignore")
    else:
        # No datakubun column (e.g., s_odds_tanpuku) — treat all as upserts
        deletes = pd.DataFrame()
        upserts = delta

    # Start with existing data
    result = existing.copy()

    # Remove rows matching delete PKs
    if not deletes.empty:
        delete_keys = deletes[pk].drop_duplicates()
        merge = result.merge(delete_keys.assign(_delete=True), on=pk, how="left", indicator=False)
        result = result[merge["_delete"] != True].copy()  # noqa: E712

    # Remove rows matching upsert PKs (old versions)
    if not upserts.empty:
        upsert_keys = upserts[pk].drop_duplicates()
        merge = result.merge(upsert_keys.assign(_upsert=True), on=pk, how="left", indicator=False)
        result = result[merge["_upsert"] != True].copy()  # noqa: E712

    # Append upsert rows
    if not upserts.empty:
        result = pd.concat([result, upserts], ignore_index=True)

    return result


def run_delta_update(
    store: ParquetStore,
    engine: Engine,
    config: list[dict],
) -> dict[str, int]:
    """Delta update: read s_ tables and merge into existing Parquet files."""
    state = _load_state()
    counts: dict[str, int] = {}

    # Filter to delta tables only
    delta_configs = [c for c in config if c.get("type") == "delta"]

    for cfg in tqdm(delta_configs, desc="Delta ETL"):
        key = cfg["parquet_key"]
        category = cfg["category"]
        pk = cfg["pk"]

        try:
            # Read delta data
            delta_df = _read_db_table(engine, cfg)
            if delta_df.empty:
                counts[key] = 0
                continue

            # Check existing Parquet exists
            if not store.exists(category, key):
                logger.warning(
                    "Delta skipped for %s: no existing Parquet. Run --mode full first.", key
                )
                counts[key] = -1
                continue

            # Type conversions: merge前にdelta行を変換
            # EveryDB2は全列character varyingのため、raw deltaは全て文字列
            # 既存Parquetは型変換済みなので、merge前に型を合わせる必要がある
            delta_df = _apply_type_conversions(delta_df, key)

            # Read existing data
            existing_df = store.read(category, key)

            # Merge (deltaは既に型変換済み)
            merged = _merge_delta(existing_df, delta_df, pk)

            # Re-compute derived columns for raced tables
            is_raced = any(
                c["parquet_key"] == key and c.get("type") == "raced"
                for c in config
                if c.get("type") != "delta"
            )
            if is_raced:
                _compute_race_date(merged)
                _compute_race_id(merged)
                merged = _compute_surface(merged)
                merged = _compute_track_condition_code(merged)

            store.write(category, key, merged)
            counts[key] = len(delta_df)

            logger.info(
                "Delta merge %s: %d delta rows -> %d total rows",
                key,
                len(delta_df),
                len(merged),
            )

        except Exception as e:
            logger.error("Delta merge failed for %s: %s", key, e)
            counts[key] = -1

        # Update state
        state["tables"][key] = {
            "rows": counts.get(key, 0),
            "last_delta": pd.Timestamp.now().isoformat(),
        }

    state["last_delta_applied"] = pd.Timestamp.now().isoformat()
    _save_state(state)
    return counts
