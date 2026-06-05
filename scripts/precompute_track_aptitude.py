"""Point-in-Time 馬場条件適性統計の事前計算

ETL + track_conditions precompute 実行後に実行する:
  python scripts/run_etl.py --mode full --start 20150101 --end 20260412
  python scripts/precompute_career_stats.py
  python scripts/precompute_track_aptitude.py
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
    from features.horse_track_aptitude import precompute_track_aptitude

    store = ParquetStore()

    logger.info("Loading entries.parquet...")
    t0 = time.time()
    entries_df = store.read("raw", "entries")
    logger.info("  %d rows (%.1fs)", len(entries_df), time.time() - t0)

    logger.info("Loading races.parquet...")
    races_df = store.read("raw", "races")

    logger.info("Loading track_conditions.parquet...")
    t0 = time.time()
    tc_df = store.read("raw", "track_conditions")
    logger.info("  %d rows (%.1fs)", len(tc_df), time.time() - t0)

    logger.info("Computing horse track aptitude...")
    t0 = time.time()
    stats = precompute_track_aptitude(entries_df, races_df, tc_df)
    logger.info("  %d rows (%.1fs)", len(stats), time.time() - t0)

    if stats.empty:
        logger.warning("No aptitude data computed. Check input data.")
        return

    logger.info("Saving to data/raw/horse_track_aptitude.parquet...")
    store.write("raw", "horse_track_aptitude", stats)

    # 検証
    debut_rate = (
        (stats["horse_dirt_wet_starts_count"] == 0)
        & (stats["horse_dirt_dry_starts_count"] == 0)
        & (stats["horse_cushion_hard_starts_count"] == 0)
        & (stats["horse_cushion_soft_starts_count"] == 0)
    ).mean()
    logger.info(
        "Debut rate: %.1f%% (%d / %d)",
        debut_rate * 100,
        int(debut_rate * len(stats)),
        len(stats),
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
