"""
ROI改善のバックテスト計測スクリプト
===========================
使い方:
  python scripts/run_backtest.py

やること:
  1. 2020-2023年のデータで学習
  2. 2024年のデータでバックテスト
  3. ROI等の結果を表示・保存

改善前のROI: 63.8% / 目標: 101%+
"""

import os
import sys
import time
import json
import warnings

warnings.filterwarnings("ignore")

# DBパスワード（環境変数に設定）
if not os.environ.get("PGPASSWORD"):
    os.environ["PGPASSWORD"] = "aa8940aa"

# プロジェクトルートをパスに追加
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    # ── 期間設定 ──
    TRAIN_START = "2020-01-01"
    TRAIN_END = "2023-12-31"
    TEST_START = "2024-01-01"
    TEST_END = "2024-12-31"

    # ── データリポジトリ ──
    from db.parquet_store import ParquetStore
    from db.repository import DataRepository

    store = ParquetStore()
    repo = DataRepository(store)
    print("DataRepository OK")

    # ── 学習 ──
    print(f"\n{'=' * 50}")
    print(f"  学習期間: {TRAIN_START} ~ {TRAIN_END}")
    print(f"{'=' * 50}")
    t0 = time.time()
    from pipelines.training_pipeline import TrainingPipelineV5

    pipeline = TrainingPipelineV5(repo=repo)
    models = pipeline.run(TRAIN_START, TRAIN_END)
    elapsed_train = time.time() - t0
    print(f"\n学習完了 ({elapsed_train:.0f}秒)")

    # ── バックテスト ──
    print(f"\n{'=' * 50}")
    print(f"  テスト期間: {TEST_START} ~ {TEST_END}")
    print(f"{'=' * 50}")
    t1 = time.time()
    from backtest.engine import BacktestEngine

    engine = BacktestEngine(repo=repo, models=models)
    result = engine.run(TEST_START, TEST_END)
    elapsed_test = time.time() - t1
    print(f"\nバックテスト完了 ({elapsed_test:.0f}秒)")

    # ── 結果表示 ──
    print(f"\n{'=' * 50}")
    print(f"  結果")
    print(f"{'=' * 50}")
    print(f"  レース数:       {result.total_bets:>8,}")
    print(f"  投資額:         {result.total_stake:>10,.0f} 円")
    print(f"  払戻額:         {result.total_return:>10,.0f} 円")
    print(f"  利益:           {result.profit:>10,.0f} 円")
    print(f"  ROI:            {result.roi:>9.1%}")
    print(f"  最大DD:         {result.max_drawdown:>9.1%}")
    print(f"  最終資金:       {result.final_bankroll:>10,.0f} 円")
    print(f"  学習時間:       {elapsed_train:>7.0f} 秒")
    print(f"  テスト時間:     {elapsed_test:>7.0f} 秒")

    # ── 改善前との比較 ──
    BEFORE_ROI = 0.638
    diff = result.roi - BEFORE_ROI
    status = "目標達成!" if result.roi >= 1.01 else "未達"
    print(f"\n{'=' * 50}")
    print(f"  Before vs After")
    print(f"{'=' * 50}")
    print(f"  改善前 ROI:     {BEFORE_ROI:.1%}")
    print(f"  改善後 ROI:     {result.roi:.1%}")
    print(f"  差分:           {diff:+.1%}")
    print(f"  判定:           {status}")

    # ── JSON保存 ──
    out = {
        "before_roi": BEFORE_ROI,
        "roi": result.roi,
        "total_bets": result.total_bets,
        "total_stake": result.total_stake,
        "total_return": result.total_return,
        "max_drawdown": result.max_drawdown,
        "final_bankroll": result.final_bankroll,
        "train_period": [TRAIN_START, TRAIN_END],
        "test_period": [TEST_START, TEST_END],
        "train_seconds": round(elapsed_train),
        "test_seconds": round(elapsed_test),
    }
    outpath = os.path.join(ROOT, "backtest_result.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n結果保存: {outpath}")


if __name__ == "__main__":
    main()
