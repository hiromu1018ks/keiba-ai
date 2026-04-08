"""マルチ年度バックテストスクリプト

使い方:
  python scripts/run_multi_year_backtest.py
  python scripts/run_multi_year_backtest.py --years 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# Windows cp932 環境で ¥ が表示できない問題を回避
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    parser = argparse.ArgumentParser(description="マルチ年度バックテスト")
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2023, 2024, 2025],
        help="テスト年度 (デフォルト: 2023 2024 2025)",
    )
    args = parser.parse_args()

    from db.parquet_store import ParquetStore

    store = ParquetStore()
    if not store.exists("raw", "races"):
        logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
        sys.exit(1)

    logger.info("ParquetStore OK")

    all_results: dict[int, Any] = {}
    all_metadata: dict[int, dict[str, str]] = {}

    for test_year in args.years:
        train_start = f"{test_year - 5}-01-01"
        train_end = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"

        print()
        print("=" * 50)
        print(f"  {test_year}年 (学習: {train_start[:4]}-{train_end[:4]})")
        print("=" * 50)

        # 学習
        t0 = time.time()
        try:
            from pipelines.training_pipeline import TrainingPipelineV5

            pipeline = TrainingPipelineV5(store=store)
            models = pipeline.run(train_start, train_end)
        except KeyboardInterrupt:
            logger.warning("中断されました")
            sys.exit(1)
        except Exception as e:
            logger.error("%d年 学習失敗: %s — スキップ", test_year, e)
            continue
        elapsed_train = time.time() - t0

        # バックテスト
        t1 = time.time()
        try:
            from backtest.engine import BacktestEngine

            engine = BacktestEngine(models=models, store=store)
            result = engine.run(test_start, test_end)
        except Exception as e:
            logger.error("%d年 テスト失敗: %s — スキップ", test_year, e)
            continue
        elapsed_test = time.time() - t1

        all_results[test_year] = result
        all_metadata[test_year] = {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_seconds": str(round(elapsed_train)),
            "test_seconds": str(round(elapsed_test)),
        }

        # 年度サマリー
        profit = result.profit
        print(f"  学習完了 ({elapsed_train:.0f}秒)")
        print(f"  テスト完了 ({elapsed_test:.0f}秒)")
        print(
            f"  ベット数: {result.total_bets:>8,} | "
            f"投資額: ¥{result.total_stake:>10,.0f} | "
            f"払戻: ¥{result.total_return:>10,.0f}"
        )
        print(
            f"  ROI: {result.total_roi:>8.1%} | "
            f"利益: ¥{profit:>+10,.0f} | "
            f"最大DD: {result.max_drawdown:>6.1%}"
        )

    # 全体サマリー
    if not all_results:
        logger.error("全年度失敗。レポートは生成しません。")
        sys.exit(1)

    print()
    print("=" * 50)
    print("  全体サマリー")
    print("=" * 50)
    total_bets = sum(r.total_bets for r in all_results.values())
    total_stake = sum(r.total_stake for r in all_results.values())
    total_return = sum(r.total_return for r in all_results.values())
    total_profit = total_return - total_stake
    total_roi = total_return / total_stake if total_stake > 0 else 0.0
    best_year = max(all_results, key=lambda y: all_results[y].total_roi)
    worst_year = min(all_results, key=lambda y: all_results[y].total_roi)

    print(f"  総ベット数:  {total_bets:>10,}")
    print(f"  総投資額:  ¥{total_stake:>12,.0f}")
    print(f"  総払戻額:  ¥{total_return:>12,.0f}")
    print(f"  総利益:    ¥{total_profit:>+12,.0f}")
    print(f"  合計 ROI:   {total_roi:>10.1%}")
    print(f"  最良年度:  {best_year} ({all_results[best_year].total_roi:.1%})")
    print(f"  最悪年度:  {worst_year} ({all_results[worst_year].total_roi:.1%})")

    # レポート生成
    output_dir = Path(os.path.join(ROOT, "data", "backtest"))
    output_dir.mkdir(parents=True, exist_ok=True)

    from backtest.report import MultiYearReportGenerator

    gen = MultiYearReportGenerator(output_dir=output_dir)
    report_path = gen.generate(all_results, all_metadata)
    print(f"\n  レポート生成: {report_path}")

    # JSON保存
    json_data: dict[str, Any] = {
        "overall": {
            "total_bets": total_bets,
            "total_stake": total_stake,
            "total_return": total_return,
            "profit": total_profit,
            "roi": total_roi,
            "best_year": best_year,
            "worst_year": worst_year,
        },
        "years": {},
    }
    for year, result in all_results.items():
        json_data["years"][str(year)] = {
            "total_bets": result.total_bets,
            "total_stake": result.total_stake,
            "total_return": result.total_return,
            "roi": result.total_roi,
            "profit": result.profit,
            "max_drawdown": result.max_drawdown,
            "metadata": all_metadata[year],
        }
    json_path = output_dir / "multi_year_result.json"
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON保存: {json_path}")

    # bet_history 保存
    all_bets: list[dict[str, Any]] = []
    for year, result in all_results.items():
        for bet in result.bet_history:
            all_bets.append({**bet, "_test_year": year})
    bets_path = output_dir / "multi_year_bet_history.json"
    bets_path.write_text(json.dumps(all_bets, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  bet_history保存: {bets_path}")


if __name__ == "__main__":
    main()
