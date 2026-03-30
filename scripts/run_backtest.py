"""バックテスト計測スクリプト

使い方:
  python scripts/run_backtest.py \
    --train-start 20200101 --train-end 20231231 \
    --test-start 20240101 --test-end 20241231
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

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
    parser = argparse.ArgumentParser(description="バックテスト")
    parser.add_argument("--train-start", required=True, help="学習開始日 (YYYYMMDD)")
    parser.add_argument("--train-end", required=True, help="学習終了日 (YYYYMMDD)")
    parser.add_argument("--test-start", required=True, help="テスト開始日 (YYYYMMDD)")
    parser.add_argument("--test-end", required=True, help="テスト終了日 (YYYYMMDD)")
    parser.add_argument("--report", action="store_true", help="HTMLレポートを生成")
    args = parser.parse_args()

    train_start = to_dash_date(args.train_start)
    train_end = to_dash_date(args.train_end)
    test_start = to_dash_date(args.test_start)
    test_end = to_dash_date(args.test_end)

    # データ検証
    from db.parquet_store import ParquetStore

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    # データリポジトリ
    from db.repository import DataRepository

    repo = DataRepository(store)
    logger.info("DataRepository OK")

    # 学習
    logger.info("=" * 50)
    logger.info("  学習期間: %s ~ %s", train_start, train_end)
    logger.info("=" * 50)
    t0 = time.time()

    from pipelines.training_pipeline import TrainingPipelineV5

    pipeline = TrainingPipelineV5(repo=repo)
    try:
        models = pipeline.run(train_start, train_end)
    except KeyboardInterrupt:
        logger.warning("学習が中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error("学習失敗: %s", e)
        sys.exit(1)

    elapsed_train = time.time() - t0
    logger.info("学習完了 (%.0f秒)", elapsed_train)

    # バックテスト
    logger.info("=" * 50)
    logger.info("  テスト期間: %s ~ %s", test_start, test_end)
    logger.info("=" * 50)
    t1 = time.time()

    from backtest.engine import BacktestEngine

    engine = BacktestEngine(models=models, repo=repo)
    result = engine.run(test_start, test_end)
    elapsed_test = time.time() - t1
    logger.info("バックテスト完了 (%.0f秒)", elapsed_test)

    # 結果表示
    print()
    print("=" * 50)
    print("  結果")
    print("=" * 50)
    print(f"  レース数:       {result.total_bets:>8,}")
    print(f"  投資額:         {result.total_stake:>10,.0f} 円")
    print(f"  払戻額:         {result.total_return:>10,.0f} 円")
    print(f"  利益:           {result.profit:>10,.0f} 円")
    print(f"  ROI:            {result.total_roi:>9.1%}")
    print(f"  最大DD:         {result.max_drawdown:>9.1%}")
    print(f"  最終資金:       {result.final_bankroll:>10,.0f} 円")
    print(f"  学習時間:       {elapsed_train:>7.0f} 秒")
    print(f"  テスト時間:     {elapsed_test:>7.0f} 秒")

    # 改善前との比較
    before_roi = 0.638
    diff = result.total_roi - before_roi
    status = "目標達成!" if result.total_roi >= 1.01 else "未達"
    print()
    print("=" * 50)
    print("  Before vs After")
    print("=" * 50)
    print(f"  改善前 ROI:     {before_roi:.1%}")
    print(f"  改善後 ROI:     {result.total_roi:.1%}")
    print(f"  差分:           {diff:+.1%}")
    print(f"  判定:           {status}")

    # JSON出力
    out = {
        "before_roi": before_roi,
        "total_roi": result.total_roi,
        "total_bets": result.total_bets,
        "total_stake": result.total_stake,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "final_bankroll": result.final_bankroll,
        "train_period": [train_start, train_end],
        "test_period": [test_start, test_end],
        "train_seconds": round(elapsed_train),
        "test_seconds": round(elapsed_test),
    }

    # --report フラグ: 全出力を data/backtest/ に集約
    if args.report:
        from backtest.report import BacktestReportGenerator

        output_dir = os.path.join(ROOT, "data", "backtest")
        os.makedirs(output_dir, exist_ok=True)

        gen = BacktestReportGenerator(output_dir=Path(output_dir))
        bet_history_path = gen.save_bet_history(result.bet_history)
        print(f"\nbet_history保存: {bet_history_path}")

        result_path = os.path.join(output_dir, "backtest_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"結果保存: {result_path}")

        train_period_str = f"{train_start} ~ {train_end}"
        test_period_str = f"{test_start} ~ {test_end}"
        report_path = gen.generate(
            result, result.bet_history,
            train_period=train_period_str, test_period=test_period_str,
        )
        print(f"レポート生成: {report_path}")
    else:
        # 従来通りプロジェクトルートに保存
        outpath = os.path.join(ROOT, "backtest_result.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n結果保存: {outpath}")


if __name__ == "__main__":
    main()
