"""ダート含水率・芝クッション値CSV → track_conditions.parquet 変換スクリプト

ETL 実行後に実行する:
  python scripts/run_etl.py --mode full --start 20150101 --end 20260412
  python scripts/precompute_track_condition.py

CSVファイルは data/ ディレクトリ内をglobで自動検出する:
  - *含水率*.csv → ダート含水率
  - *クッション*.csv → 芝クッション値
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def _resolve_csv(pattern: str, label: str) -> str:
    """glob パターンでCSVファイルを検索。見つからない場合はエラー終了。"""
    matches = glob.glob(os.path.join(ROOT, "data", pattern))
    if not matches:
        logger.error("No %s CSV found (pattern: data/%s)", label, pattern)
        sys.exit(1)
    if len(matches) > 1:
        logger.warning("Multiple %s CSVs found, using first: %s", label, matches[0])
    return matches[0]


def main() -> None:
    from db.parquet_store import ParquetStore
    from features.track_condition_data import convert_track_conditions

    store = ParquetStore()

    # CSVパスの解決
    dirt_csv = _resolve_csv("*含水率*.csv", "dirt moisture")
    cushion_csv = _resolve_csv("*クッション*.csv", "turf cushion")
    logger.info("Dirt CSV:    %s", dirt_csv)
    logger.info("Cushion CSV: %s", cushion_csv)

    # races.parquetがあれば交差検証に使用
    races_df = None
    if store.exists("raw", "races"):
        logger.info("Loading races.parquet for cross-validation...")
        races_df = store.read("raw", "races")
        logger.info("  %d races loaded", len(races_df))
    else:
        logger.warning("races.parquet not found, skipping cross-validation")

    # 変換実行
    logger.info("Converting track condition data...")
    t0 = time.time()
    result = convert_track_conditions(dirt_csv, cushion_csv, store, races_df=races_df)
    elapsed = time.time() - t0

    # 結果サマリ
    n_total = len(result)
    n_dirt = result["dirt_moisture"].notna().sum()
    n_turf = result["turf_cushion"].notna().sum()
    n_both = (result["dirt_moisture"].notna() & result["turf_cushion"].notna()).sum()

    logger.info("Summary (%.1fs):", elapsed)
    logger.info("  Total races:          %d", n_total)
    logger.info("  With dirt_moisture:   %d", int(n_dirt))
    logger.info("  With turf_cushion:    %d", int(n_turf))
    logger.info("  With both:            %d", int(n_both))

    # 日付範囲
    if not result.empty and result["race_date"].notna().any():
        dates = result["race_date"].dropna()
        logger.info("  Date range: %s ~ %s", dates.min().date(), dates.max().date())

    logger.info("Done.")


if __name__ == "__main__":
    main()
