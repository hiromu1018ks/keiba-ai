#!/usr/bin/env python3
"""OOF予測のIC評価を実行するCLIスクリプト (Phase 30).

使い方:
  python scripts/run_ic_eval.py data/oof/oof_predictions.parquet
  python scripts/run_ic_eval.py data/oof/oof_predictions.parquet --mlflow
"""

import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from models.ic_evaluator import console_summary, run_ic_evaluation  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="IC evaluation for OOF predictions")
    parser.add_argument("oof_path", help="Path to OOF predictions Parquet file")
    parser.add_argument(
        "--output",
        default="data/baseline/ic_baseline.json",
        help="JSON output path (default: data/baseline/ic_baseline.json)",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow logging for IC baseline",
    )
    parser.add_argument(
        "--experiment",
        default="keiba-v5-ic-eval",
        help="MLflow experiment name (default: keiba-v5-ic-eval)",
    )
    args = parser.parse_args()

    logger.info("Loading OOF predictions from %s", args.oof_path)
    df_oof = pd.read_parquet(args.oof_path)
    logger.info("Loaded %d rows, %d columns", len(df_oof), len(df_oof.columns))

    if args.mlflow:
        import mlflow

        mlflow.set_experiment(args.experiment)
        mlflow.start_run(run_name="ic_baseline_eval")

    output_path = Path(args.output)
    result = run_ic_evaluation(
        df_oof, output_path=output_path, mlflow_log=args.mlflow,
    )
    console_summary(result)

    if args.mlflow:
        import mlflow

        mlflow.end_run()

    logger.info("IC evaluation complete. Output: %s", args.output)


if __name__ == "__main__":
    main()
