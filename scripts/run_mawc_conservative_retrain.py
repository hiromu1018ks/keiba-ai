"""MAWC Conservative Retrain CLI -- retrain MAWC with conservative regularization.

Phase 45 Plan 02: CLI entry point for conservative MAWC retraining.
Produces 3 output files in target directory:
  1. manifest.json -- Machine-readable manifest for Phase 46 consumption
  2. retrain_summary.md -- Human-readable summary with C grid results
  3. mawc_conservative_report.html -- (optional with --report) HTML report

Usage:
  python scripts/run_mawc_conservative_retrain.py \\
    --oof-path data/oof/oof_predictions.parquet \\
    --source-model-dir data/models-backtest \\
    --target-root data/models-backtest-mawc-conservative \\
    --years 2024,2025

  # With HTML report
  python scripts/run_mawc_conservative_retrain.py \\
    --oof-path data/oof/oof_predictions.parquet \\
    --source-model-dir data/models-backtest \\
    --target-root data/models-backtest-mawc-conservative \\
    --years 2024,2025 --report
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Windows cp932 環境でエンコーディング問題を回避
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


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for conservative MAWC retrain CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "MAWC Conservative Retrain -- retrain MarketAwareWinCalibrator with "
            "reduced features and strong regularization"
        ),
    )
    parser.add_argument(
        "--oof-path",
        type=Path,
        default=Path("data/oof/oof_predictions.parquet"),
        help="Path to OOF predictions parquet (default: data/oof/oof_predictions.parquet)",
    )
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        default=Path("data/models-backtest"),
        help="Source model directory (default: data/models-backtest)",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/models-backtest-mawc-conservative"),
        help="Output directory for conservative variants "
             "(default: data/models-backtest-mawc-conservative)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2024,2025",
        help="Comma-separated test years (default: 2024,2025)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Main entry point for conservative MAWC retrain."""
    from models.mawc_conservative_retrainer import (
        MawcConservativeRetrainer,
        save_retrain_results,
    )

    # Parse years
    years = [int(y.strip()) for y in args.years.split(",")]

    logger.info(
        "MAWC Conservative Retrain: oof=%s source=%s target=%s years=%s",
        args.oof_path, args.source_model_dir, args.target_root, years,
    )

    # Run full pipeline
    trainer = MawcConservativeRetrainer()
    manifest, all_results = trainer.run_full_pipeline(
        oof_path=args.oof_path,
        source_model_dir=args.source_model_dir,
        target_root=args.target_root,
        years=years,
    )

    # Save manifest + summary
    manifest_path, summary_path = save_retrain_results(
        manifest, all_results, args.target_root,
    )

    # HTML report (optional)
    if args.report:
        from models.mawc_conservative_report import MawcConservativeReportGenerator

        report_gen = MawcConservativeReportGenerator(args.target_root)
        report_path = report_gen.generate(manifest, all_results)
        logger.info("HTML report: %s", report_path)

    # Summary to stdout
    print()
    print("MAWC Conservative Retrain Complete")
    print("=" * 60)
    print(f"Target: {args.target_root}")
    print(f"Years: {years}")

    per_year_surface = manifest.get("per_year_surface", {})
    # Show last year's metrics for each surface as summary
    last_year = str(years[-1]) if years else ""
    per_surface = per_year_surface.get(last_year, {})
    for surface, data in per_surface.items():
        best_c = data.get("best_c")
        deployed = data.get("deployed", False)
        c_str = f"{best_c:.4f}" if best_c is not None else "N/A"
        dep_str = "DEPLOYED" if deployed else "not_deployed"
        beta = data.get("beta_market_contribution")
        beta_str = f"{beta:.4f}" if beta is not None else "N/A"
        gate = data.get("quality_gate_summary", {})
        ece_cons = gate.get("overall_ece")
        ece_str = f"{ece_cons:.4f}" if ece_cons is not None else "N/A"
        print(
            f"  {surface}: {dep_str} | C={c_str} | beta_market={beta_str}"
            f" | ECE={ece_str}"
        )

    print(f"\nManifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    if args.report:
        print(f"Report:   {args.target_root / 'mawc_conservative_report.html'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
