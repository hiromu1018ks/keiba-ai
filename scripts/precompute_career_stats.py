"""Point-in-Time キャリア統計の事前計算

ETL 実行後に実行する:
  python scripts/run_etl.py --mode full --start 20150101 --end 20260412
  python scripts/precompute_career_stats.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def main() -> None:
    from db.parquet_store import ParquetStore
    from features.horse_career_stats import precompute_career_stats

    store = ParquetStore()

    logger.info("Loading entries.parquet...")
    t0 = time.time()
    entries_df = store.read("raw", "entries")
    logger.info("  %d rows (%.1fs)", len(entries_df), time.time() - t0)

    logger.info("Loading races.parquet...")
    races_df = store.read("raw", "races")

    logger.info("Computing career stats...")
    t0 = time.time()
    stats = precompute_career_stats(entries_df, races_df)
    logger.info("  %d rows (%.1fs)", len(stats), time.time() - t0)

    logger.info("Saving to data/raw/horse_career_stats.parquet...")
    store.write("raw", "horse_career_stats", stats)

    # 検証
    debut_rate = (stats["cum_starts"] == 0).mean()
    logger.info(
        "Debut rate: %.1f%% (%d / %d)",
        debut_rate * 100,
        (stats["cum_starts"] == 0).sum(),
        len(stats),
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
