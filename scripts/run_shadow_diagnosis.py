"""Shadow Diagnosis CLI スクリプト (D-01, D-04).

使い方:
  python scripts/run_shadow_diagnosis.py --input-dir data/backtest/shadow/2024

  # HTMLレポート付き
  python scripts/run_shadow_diagnosis.py --input-dir data/backtest/shadow/2024 --report

  # 出力先指定
  python scripts/run_shadow_diagnosis.py \\
    --input-dir data/backtest/shadow/2024 \\
    --output-dir data/backtest/shadow/diagnosis
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
    """引数パーサーを構築."""
    parser = argparse.ArgumentParser(
        description="Shadow Diagnosis — baseline vs shadow diagnostic analysis",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing Phase 41 shadow comparison artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/backtest/shadow/diagnosis"),
        help="Output directory (default: data/backtest/shadow/diagnosis)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """メイン処理."""
    from backtest.shadow_diagnosis import (
        ShadowDiagnosis,
        ShadowDiagnosisReportGenerator,
        save_diagnosis_results,
    )

    logger.info("Shadow Diagnosis: input_dir=%s", args.input_dir)

    # 診断実行
    sd = ShadowDiagnosis(args.input_dir)
    diagnosis_result = sd.run()

    # JSON + Markdown 出力
    save_diagnosis_results(diagnosis_result, args.output_dir)
    logger.info("Results saved to %s", args.output_dir)

    # HTML レポート (オプション)
    if args.report:
        report_gen = ShadowDiagnosisReportGenerator(args.output_dir)
        report_path = report_gen.generate(diagnosis_result)
        logger.info("HTML report: %s", report_path)

    # サマリーを stdout に印刷
    print()
    print("Shadow Diagnosis Complete")
    print("=" * 50)

    # Step 1: Probability Quality
    s1 = diagnosis_result.step1
    print("Step 1: Probability Quality")
    print(
        f"  Brier:   baseline={s1.baseline_brier:.4f}  shadow={s1.shadow_brier:.4f}"
        f"  delta={s1.delta_brier:+.4f}"
    )
    print(
        f"  Logloss: baseline={s1.baseline_logloss:.4f}  shadow={s1.shadow_logloss:.4f}"
        f"  delta={s1.delta_logloss:+.4f}"
    )
    print(
        f"  ECE:     baseline={s1.baseline_ece:.4f}  shadow={s1.shadow_ece:.4f}"
        f"  delta={s1.delta_ece:+.4f}"
    )
    print(
        f"  APR:     baseline={s1.baseline_apr:.4f}  shadow={s1.shadow_apr:.4f}"
        f"  delta={s1.delta_apr:+.4f}"
    )

    # Step 2: Selection Pattern
    s2 = diagnosis_result.step2
    print("Step 2: Selection Pattern")
    print(
        f"  Changed:   {s2.n_changed_races} races, ROI={s2.changed.roi:.4f},"
        f" HR={s2.changed.hit_rate:.4f}"
    )
    print(
        f"  Unchanged: {s2.n_unchanged_races} races, ROI={s2.unchanged.roi:.4f},"
        f" HR={s2.unchanged.hit_rate:.4f}"
    )
    print(f"  Delta ROI: {s2.delta_roi:+.4f}")

    # Step 3: Top calibration gaps
    s3 = diagnosis_result.step3
    top_segs = sorted(
        s3.segments,
        key=lambda s: abs(s.delta_apr) + abs(s.delta_ece),
        reverse=True,
    )[:3]
    if top_segs:
        print("Step 3: Top 3 Calibration Gaps")
        for seg in top_segs:
            print(
                f"  {seg.segment_name}={seg.segment_value}:"
                f" delta_apr={seg.delta_apr:+.4f}, delta_ece={seg.delta_ece:+.4f}"
            )

    # Missing inputs
    if diagnosis_result.missing_inputs:
        print(f"Missing Inputs: {', '.join(diagnosis_result.missing_inputs)}")
    else:
        print("Missing Inputs: None")

    print("=" * 50)
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
