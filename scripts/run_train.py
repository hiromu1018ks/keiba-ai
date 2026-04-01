"""モデル学習スクリプト

使い方:
  python scripts/run_train.py --start 20200101 --end 20231231
  python scripts/run_train.py --start 20200101 --end 20231231 --experiment keiba-v5.5
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


def to_dash_date(yyyymmdd: str) -> str:
    """YYYYMMDD → YYYY-MM-DD"""
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="モデル学習")
    parser.add_argument("--start", required=True, help="学習開始日 (YYYYMMDD)")
    parser.add_argument("--end", required=True, help="学習終了日 (YYYYMMDD)")
    parser.add_argument("--experiment", default="keiba-v5", help="MLflow実験名 (default: keiba-v5)")
    args = parser.parse_args()

    train_start = to_dash_date(args.start)
    train_end = to_dash_date(args.end)

    # データ検証
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    # MLflow実験名設定
    import mlflow

    mlflow.set_experiment(args.experiment)

    # 学習
    from pipelines.training_pipeline import TrainingPipelineV5

    pipeline = TrainingPipelineV5(store=store)

    logger.info("学習開始: %s ~ %s", train_start, train_end)
    t0 = time.time()

    try:
        models = pipeline.run(train_start, train_end)
    except KeyboardInterrupt:
        logger.warning("学習が中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error("学習失敗: %s", e)
        sys.exit(1)

    elapsed = time.time() - t0
    n_surfaces = len(models.submodels)
    logger.info("学習完了 (%.0f秒) — %dサーフェスモデル", elapsed, n_surfaces)


if __name__ == "__main__":
    main()
