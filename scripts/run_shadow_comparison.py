"""Shadow Comparison CLI スクリプト (D-07).

使い方:
  python scripts/run_shadow_comparison.py \\
    --baseline-root data/models-backtest \\
    --shadow-root data/models-backtest \\
    --folds 2024 2025

  # HTMLレポート付き
  python scripts/run_shadow_comparison.py \\
    --baseline-root data/models-backtest \\
    --shadow-root data/models-backtest \\
    --folds 2024 2025 --report
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
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
    """引数パーサーを構築."""
    parser = argparse.ArgumentParser(
        description="Shadow Comparison — baseline vs shadow model comparison",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="Root directory containing baseline model year subdirectories",
    )
    parser.add_argument(
        "--shadow-root",
        type=Path,
        required=True,
        help="Root directory containing shadow model year subdirectories",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[2024, 2025],
        help="Fold years to run (default: 2024 2025)",
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=4,
        help="Training window in years (default: 4)",
    )
    parser.add_argument(
        "--betting-target",
        type=str,
        default="win",
        help="Betting target (default: win)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/backtest/shadow"),
        help="Output directory (default: data/backtest/shadow)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="baseline",
        help="Name for baseline variant (default: baseline)",
    )
    parser.add_argument(
        "--shadow-name",
        type=str,
        default="ridge_shadow",
        help="Name for shadow variant (default: ridge_shadow)",
    )
    parser.add_argument(
        "--betting-mode",
        type=str,
        default="flat",
        help="Betting mode: flat or kelly (default: flat)",
    )
    parser.add_argument(
        "--calibration-bt",
        action="store_true",
        default=False,
        help="Run calibration BT on training period (default: skip, matching run_backtest.py)",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """メイン処理."""
    from backtest.shadow_comparison import (
        FoldDefinition,
        ShadowComparisonFramework,
        VariantConfig,
        save_manifest,
        save_results,
    )

    # D-18: Baseline has MAWC/ranker disabled
    variant_configs = [
        VariantConfig(
            variant_name=args.baseline_name,
            model_dir=args.baseline_root,
            enable_market_aware_calibrator=False,
            enable_race_level_ranker=False,
        ),
        VariantConfig(
            variant_name=args.shadow_name,
            model_dir=args.shadow_root,
            enable_market_aware_calibrator=True,
            enable_race_level_ranker=True,
        ),
    ]

    # D-05: Build folds
    folds = FoldDefinition.create_folds(args.folds, train_window=args.train_window)

    logger.info(
        "Shadow Comparison: %d fold(s), variants=%s",
        len(folds),
        [v.variant_name for v in variant_configs],
    )

    # Create framework and run
    logger.info(
        "Shadow calibration BT: %s",
        "enabled" if args.calibration_bt else "disabled",
    )
    framework = ShadowComparisonFramework(
        variants=variant_configs,
        betting_target=args.betting_target,
        betting_mode=args.betting_mode,
        run_calibration_bt=args.calibration_bt,
    )

    t0 = time.time()
    results = framework.run(folds)
    elapsed = time.time() - t0
    logger.info("Comparison completed in %.1f seconds", elapsed)

    # Save artifacts
    artifact_paths = save_results(results, args.output_dir)
    save_manifest(
        results, variant_configs, args.output_dir, artifact_paths,
        calibration_bt=args.calibration_bt,
    )
    logger.info("Artifacts saved to %s", args.output_dir)

    # Generate HTML report if requested
    if args.report:
        import json

        from backtest.shadow_report import ShadowComparisonReportGenerator

        report_gen = ShadowComparisonReportGenerator(args.output_dir)
        metrics_data = json.loads(
            artifact_paths["metrics_json"].read_text(encoding="utf-8"),
        )
        report_path = report_gen.generate(
            comparison_results=results,
            variant_configs=variant_configs,
            metrics_json=metrics_data,
        )
        logger.info("HTML report: %s", report_path)

    # Print summary
    print()
    print("Shadow Comparison Complete")
    print("=" * 40)

    for cr in results:
        year = cr.fold.year
        print(f"Fold {year}:")
        for vname in sorted(cr.metrics.keys()):
            m = cr.metrics[vname]
            print(
                f"  {vname:12s}: ROI={m.roi * 100:.1f}%, "
                f"HR={m.hit_rate * 100:.1f}%, "
                f"Bets={m.bet_count}"
            )

        # Delta ROI
        vnames = sorted(cr.metrics.keys())
        if len(vnames) >= 2:
            delta = cr.metrics[vnames[1]].roi - cr.metrics[vnames[0]].roi
            agreement = cr.metrics[vnames[0]].selection_agreement
            agreement_str = f"{agreement * 100:.1f}%" if agreement is not None else "N/A"
            print(
                f"  Delta ROI: {delta * 100:+.1f}pp | "
                f"Selection Agreement: {agreement_str}"
            )

    print("=" * 40)

    # Overall summary
    if results:
        all_variant_names = sorted(set(vn for cr in results for vn in cr.metrics))
        for vname in all_variant_names:
            pooled_stake = sum(
                cr.variants[vname].backtest_result.total_stake
                for cr in results
                if vname in cr.variants
            )
            pooled_return = sum(
                cr.variants[vname].backtest_result.total_return
                for cr in results
                if vname in cr.variants
            )
            overall_roi = pooled_return / pooled_stake - 1.0 if pooled_stake > 0 else 0.0
            print(f"  {vname}: Overall ROI={overall_roi * 100:.1f}%")

    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
