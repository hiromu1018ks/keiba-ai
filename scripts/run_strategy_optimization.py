"""戦略パラメータOptuna最適化CLI

Usage:
    python scripts/run_strategy_optimization.py \
        --n-trials 100 \
        --models-dir data/models \
        --output data/strategy_manifest.json
"""
import argparse
import logging
import sys
from pathlib import Path

# src/ を pythonpath に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tuning.strategy_optimizer import StrategyOptimizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna戦略パラメータ最適化")
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Optuna試行回数 (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="TPESampler乱数シード (default: 42)",
    )
    parser.add_argument(
        "--models-dir", type=str, default="data/models",
        help="学習済みモデルディレクトリ",
    )
    parser.add_argument(
        "--output", type=str, default="data/strategy_manifest.json",
        help="出力manifestファイルパス",
    )
    parser.add_argument(
        "--min-bets", type=int, default=1000,
        help="1foldあたりの最低ベット数 (default: 1000)",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Multi-seed安定性検証用のカンマ区切りseedリスト (例: 42,43,44)",
    )
    args = parser.parse_args()

    optimizer = StrategyOptimizer(
        models_dir=args.models_dir,
        min_bets_per_fold=args.min_bets,
    )

    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        logger.info("Multi-seed stability optimization: seeds=%s", seeds)
        result = optimizer.optimize_multi_seed(
            n_trials=args.n_trials,
            seeds=seeds,
            output_dir=Path(args.output).parent,
        )
        logger.info(
            "Stability report generated. Mean best ROI: %.4f",
            result.get("mean_best_roi", 0.0),
        )
    else:
        logger.info(f"Starting strategy optimization: n_trials={args.n_trials}")
        result = optimizer.optimize(
            n_trials=args.n_trials,
            seed=args.seed,
            output_path=Path(args.output),
        )

        logger.info(
            f"Best ROI: {result['best_value']:.4f}, "
            f"Best params: {result['best_params']}, "
            f"Trials: {result['n_trials']}, "
            f"Pruned: {result['n_pruned']}"
        )


if __name__ == "__main__":
    main()
