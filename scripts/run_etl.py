"""EveryDB2外部テーブル → Parquetエクスポート

使い方:
  python scripts/run_etl.py --start 20150101 --end 20241231
"""

import argparse
import logging
import os
import sys
import time

# プロジェクトルートをパスに追加
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="EveryDB2 → Parquet ETL")
    parser.add_argument("--start", required=True, help="開始日 (YYYYMMDD)")
    parser.add_argument("--end", required=True, help="終了日 (YYYYMMDD)")
    args = parser.parse_args()

    # DB接続
    from db.connection import DatabaseConnection
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    db = DatabaseConnection()

    logger.info("ETL開始: %s ~ %s", args.start, args.end)
    t0 = time.time()

    try:
        counts = db.etl_to_parquet(store, args.start, args.end)
    except KeyboardInterrupt:
        logger.warning("ETLが中断されました")
        sys.exit(1)
    except Exception as e:
        if "could not connect" in str(e).lower() or "connection refused" in str(e).lower():
            logger.error("PostgreSQLに接続できません。localhost:5432 が実行中か確認してください。")
        else:
            logger.error("ETL失敗: %s", e)
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info("ETL完了 (%.0f秒)", elapsed)

    for table, n in counts.items():
        logger.info("  %s: %d行", table, n)


if __name__ == "__main__":
    main()
