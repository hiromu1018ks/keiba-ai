"""Generic ETL engine: EveryDB2 → Parquet (YAML-driven)"""

from __future__ import annotations

import json
import logging
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
                # Year-by-year chunked loading for large partitioned tables
                start_year = int(start) // 10000
                end_year = int(end) // 10000
                frames = []
                for year in range(start_year, end_year + 1):
                    year_start = f"{year}0101"
                    year_end = f"{year}1231"
                    df = _read_db_table(engine, cfg, start=year_start, end=year_end)
                    if not df.empty:
                        _compute_race_date(df)
                        _compute_race_id(df)
                        frames.append(df)
                if frames:
                    combined = pd.concat(frames, ignore_index=True)
                    # Add partition columns from race_date
                    if "race_date" in combined.columns:
                        combined["year"] = combined["race_date"].dt.year
                        combined["month"] = combined["race_date"].dt.month
                    # Clear existing partitioned data to avoid stale year partitions
                    partition_path = store.data_dir / category / key
                    if partition_path.is_dir():
                        shutil.rmtree(partition_path)
                    store.write(category, key, combined, partition_cols=partition_cols)
                    counts[key] = len(combined)
                else:
                    counts[key] = 0
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
    """
    deletes = delta[delta["datakubun"] == "0"]
    upserts = delta[delta["datakubun"] != "0"].drop(columns=["datakubun"], errors="ignore")

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

            # Read existing data
            existing_df = store.read(category, key)

            # Merge
            merged = _merge_delta(existing_df, delta_df, pk)

            # Re-add race_date if needed
            is_raced = any(
                c["parquet_key"] == key and c.get("type") == "raced"
                for c in config
                if c.get("type") != "delta"
            )
            if is_raced:
                if "race_date" not in merged.columns:
                    _compute_race_date(merged)
                if "race_id" not in merged.columns:
                    _compute_race_id(merged)

            # Write back
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
