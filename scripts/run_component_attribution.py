"""Component Attribution CLI -- bisect DeploymentGate FAIL causes by component.

Usage:
  python scripts/run_component_attribution.py --input-dir data/backtest/shadow

  # With HTML report
  python scripts/run_component_attribution.py --input-dir data/backtest/shadow --report

  # With model directory and custom output
  python scripts/run_component_attribution.py \\
    --input-dir data/backtest/shadow \\
    --output-dir data/backtest/shadow/bisect \\
    --model-dir data/models-backtest \\
    --report
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
    """Build argument parser for component attribution CLI."""
    parser = argparse.ArgumentParser(
        description="Component Attribution -- bisect DeploymentGate FAIL causes",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/backtest/shadow"),
        help="Directory containing Phase 41/43/42 shadow artifacts (default: data/backtest/shadow)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/backtest/shadow/bisect"),
        help="Output directory (default: data/backtest/shadow/bisect)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("data/models-backtest"),
        help="Directory containing trained model artifacts (default: data/models-backtest)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Main entry point for component attribution."""
    from backtest.component_attribution import (
        ComponentAttribution,
        save_attribution_results,
    )

    logger.info("Component Attribution: input_dir=%s", args.input_dir)

    # Run full attribution
    ca = ComponentAttribution(
        input_dir=args.input_dir,
        models_dir=args.model_dir,
    )
    attribution_result = ca.run_full_attribution()

    # Optional historical bisect for auxiliary context
    historical_result = None
    try:
        from backtest.historical_bisect import HistoricalBisect

        hb = HistoricalBisect(input_dir=args.input_dir)
        historical_result = hb.run_historical_comparison()
        logger.info("Historical bisect completed (auxiliary)")
    except Exception as e:
        logger.warning("Historical bisect skipped: %s", e)

    # Save JSON + Markdown results
    save_attribution_results(
        attribution_result,
        args.output_dir,
        historical_result=historical_result,
    )
    logger.info("Results saved to %s", args.output_dir)

    # HTML report (optional)
    if args.report:
        from backtest.component_attribution_report import (
            ComponentAttributionReportGenerator,
        )

        report_gen = ComponentAttributionReportGenerator(args.output_dir)
        report_path = report_gen.generate(attribution_result, historical_result)
        logger.info("HTML report: %s", report_path)

    # Summary to stdout
    print()
    print("Component Attribution Complete")
    print("=" * 60)

    # ECE
    ece = attribution_result.ece_attribution
    ece_segs = ece.get("segments", [])
    if ece_segs:
        worst = sorted(ece_segs, key=lambda s: s.get("delta_ece", 0), reverse=True)[0]
        print(
            f"ECE: worst segment {worst.get('segment_name', '')}="
            f"{worst.get('segment_value', '')}:"
            f" delta_ece={worst.get('delta_ece', 0):+.4f}"
        )
    else:
        print("ECE: no degradation detected")

    # APR
    apr = attribution_result.apr_attribution
    all_apr_delta = apr.get("all_horse_apr", {}).get("delta_apr", 0)
    sel_apr_delta = apr.get("selected_horse_apr", {}).get("delta_apr", 0)
    print(f"APR: all-horse delta={all_apr_delta:+.4f} | selected delta={sel_apr_delta:+.4f}")

    # Bet count
    bc = attribution_result.bet_count_attribution
    print(
        f"Bet count: baseline={bc.get('baseline_bet_count', 0)}"
        f" shadow={bc.get('shadow_bet_count', 0)}"
        f" gap={bc.get('gap', 0)}"
    )

    # Upstream anomaly check
    print(f"Upstream anomaly: {attribution_result.upstream_anomaly_check}")

    # Recommendations
    if attribution_result.recommendations:
        print("\nRecommendations for Phase 45:")
        for i, rec in enumerate(attribution_result.recommendations, 1):
            print(f"  {i}. {rec}")

    # Historical context
    if historical_result is not None:
        finding = (
            historical_result.auxiliary_findings[0]
            if historical_result.auxiliary_findings
            else "N/A"
        )
        print(f"\nHistorical: {finding}")

    print("=" * 60)
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
