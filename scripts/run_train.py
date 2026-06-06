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
    parser.add_argument(
        "--ensemble", action="store_true",
        help="StackedEnsemble (LGBM+XGB+CatBoost→Ridge) を有効化",
    )
    parser.add_argument(
        "--betting-target",
        choices=["win", "place", "wide"],
        default="place",
        help="学習スコープ (win=共通+Win, place=共通+Win+Place, wide=拒否) (default: place)",
    )
    args = parser.parse_args()

    # wide は v2.4 で対象外 (D-13)
    if args.betting_target == "wide":
        logger.error("--betting-target wide は v2.4 で未対応です。win または place を指定してください。")
        sys.exit(1)

    train_start = to_dash_date(args.start)
    train_end = to_dash_date(args.end)

    # データ検証
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    # Pre-training Parquet 検証 (TRN-02)
    import pandas as pd

    races_df = store.read("raw", "races")
    if "race_date" in races_df.columns:
        rd = pd.to_datetime(races_df["race_date"], errors="coerce").dropna()
        if len(rd) >= 1:
            logger.info("races.parquet 日付範囲: %s ~ %s (%d件)", rd.min().date(), rd.max().date(), len(rd))

    # track_conditions.parquet 存在確認
    if store.exists("raw", "track_conditions"):
        tc_df = store.read("raw", "track_conditions")
        logger.info("track_conditions.parquet: %d件", len(tc_df))
        if "race_date" in tc_df.columns:
            tc_rd = pd.to_datetime(tc_df["race_date"], errors="coerce").dropna()
            if len(tc_rd) >= 1:
                logger.info("  日付範囲: %s ~ %s", tc_rd.min().date(), tc_rd.max().date())
    else:
        logger.warning("track_conditions.parquet が見つかりません。トラック特徴量はデフォルト値を使用します。")

    # horse_track_aptitude.parquet 存在確認
    if store.exists("raw", "horse_track_aptitude"):
        hta_df = store.read("raw", "horse_track_aptitude")
        logger.info("horse_track_aptitude.parquet: %d件", len(hta_df))
    else:
        logger.warning("horse_track_aptitude.parquet が見つかりません。馬場適性特徴量は使用されません。")

    # キー列のNaN率確認
    if store.exists("raw", "entries"):
        entries_df = store.read("raw", "entries")
        for col in ["umaban", "horse_id"]:
            if col in entries_df.columns:
                nan_rate = entries_df[col].isna().mean() * 100
                if nan_rate > 50:
                    logger.warning("entries.%s NaN率: %.1f%% (高すぎる可能性)", col, nan_rate)
                else:
                    logger.info("entries.%s NaN率: %.1f%%", col, nan_rate)

    # MLflow実験名設定
    import mlflow

    mlflow.set_experiment(args.experiment)

    # 学習
    from pipelines.training_pipeline import TrainingPipelineV5

    pipeline = TrainingPipelineV5(store=store)

    logger.info("学習開始: %s ~ %s (betting_target=%s)", train_start, train_end, args.betting_target)
    t0 = time.time()

    try:
        models = pipeline.run(train_start, train_end, use_ensemble=args.ensemble, betting_target=args.betting_target)
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
