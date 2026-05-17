"""Generic ETL: EveryDB2 → Parquet

使い方:
  python scripts/run_etl.py --mode full --start 20140101 --end 20231231
  python scripts/run_etl.py --mode delta
  python scripts/run_etl.py --mode full --tables races entries --start 20140101 --end 20231231
"""

import argparse
import logging
import os
import sys
import time
from typing import TYPE_CHECKING

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _verify_coverage(
    store: "ParquetStore",
    tables: list[dict],
    start_year: int,
    end_year: int,
) -> None:
    """Post-ETL coverage verification: row counts, year coverage, missing rate."""
    from db.readers import date_filters

    start_str = f"{start_year}0101"
    end_str = f"{end_year}1231"

    for cfg in tables:
        table = cfg["parquet_key"]
        category = cfg["category"]

        if not store.exists(category, table):
            logger.warning("Coverage SKIP: %s (file not found)", table)
            continue

        df = store.read(category, table, filters=date_filters(start_str, end_str))
        n_rows = len(df)

        # Year coverage from race_date
        if "race_date" in df.columns and n_rows > 0:
            years_present = sorted(df["race_date"].dt.year.unique().tolist())
        else:
            years_present = []

        # Max missing rate across all columns
        if n_rows > 0:
            max_missing = df.isnull().mean().max() * 100
        else:
            max_missing = 0.0

        logger.info(
            "Coverage %s: %d rows, years=%s, max_missing=%.1f%%",
            table, n_rows, years_present, max_missing,
        )

        # Check missing years
        expected_years = set(range(start_year, end_year + 1))
        actual_years = set(years_present)
        missing_years = sorted(expected_years - actual_years)
        if missing_years:
            logger.warning(
                "Coverage WARN: %s missing years %s", table, missing_years,
            )

        # Check missing rate threshold
        if max_missing > 30:
            logger.warning(
                "Coverage WARN: %s missing rate %.1f%% exceeds 30%%",
                table, max_missing,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic ETL: EveryDB2 → Parquet")
    parser.add_argument("--mode", choices=["full", "delta"], required=True,
                        help="full: n_テーブル全件出力, delta: s_テーブル差分マージ")
    parser.add_argument("--start", help="開始日 YYYYMMDD (full mode)")
    parser.add_argument("--end", help="終了日 YYYYMMDD (full mode)")
    parser.add_argument("--tables", nargs="*", help="対象テーブル (parquet_key指定)")
    args = parser.parse_args()

    if args.mode == "full" and (not args.start or not args.end):
        parser.error("--mode full requires --start and --end")

    from db.connection import DatabaseConnection
    from db.etl import load_table_config, run_delta_update, run_full_load
    from db.parquet_store import ParquetStore

    config = load_table_config()
    store = ParquetStore()
    db = DatabaseConnection()
    engine = db.get_engine()

    logger.info("ETL開始: mode=%s", args.mode)
    t0 = time.time()

    try:
        if args.mode == "full":
            counts = run_full_load(store, engine, config, args.start, args.end, args.tables)
        else:
            counts = run_delta_update(store, engine, config)
    except KeyboardInterrupt:
        logger.warning("ETLが中断されました")
        sys.exit(1)
    except Exception as e:
        if "could not connect" in str(e).lower() or "connection refused" in str(e).lower():
            logger.error("PostgreSQLに接続できません。localhost:5432 が実行中か確認してください。")
        else:
            logger.error("ETL失敗: %s", e)
        sys.exit(1)
    finally:
        engine.dispose()

    elapsed = time.time() - t0
    logger.info("ETL完了 (%.0f秒)", elapsed)

    for table, n in counts.items():
        logger.info("  %s: %d行", table, n)

    if args.mode == "full":
        active_configs = [c for c in config if c["parquet_key"] in counts]
        _verify_coverage(store, active_configs, int(args.start[:4]), int(args.end[:4]))


if __name__ == "__main__":
    main()
