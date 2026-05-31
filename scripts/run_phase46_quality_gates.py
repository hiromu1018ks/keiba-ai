"""Phase 46: Quality Gate Verification -- orchestrate 2-stage quality gate flow.

Stage 1: MAWC Conservative Retrain (subprocess)
Stage 2: 5 quality checks in sequence
  1. FeatureRoutingAudit
  2. OOFHealthValidator
  3. Shadow Comparison (~82 min)
  4. Shadow Diagnosis
  5. DeploymentGateEvaluator

Results aggregate into phase46_quality_gate_result.json with 3-label framework:
  - quality_gate: PASS/FAIL
  - roi_trend: recovered/weak_recovery/not_recovered
  - deployment: deployable/not_deployable/manual_review

Usage:
  python scripts/run_phase46_quality_gates.py --stage 1
  python scripts/run_phase46_quality_gates.py --stage 2
  python scripts/run_phase46_quality_gates.py --force
  python scripts/run_phase46_quality_gates.py --report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from audit.feature_routing_registry import run_feature_audit
from backtest.deployment_gates import run_deployment_gates
from backtest.shadow_diagnosis import ShadowDiagnosis, save_diagnosis_results
from validation.oof_health_validator import OOF_PREDICTIONS_PROFILE, OOFHealthValidator

warnings.filterwarnings("ignore")

# Windows cp932 workaround
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


# ---------------------------------------------------------------------------
# QualityGateOrchestrator
# ---------------------------------------------------------------------------


class QualityGateOrchestrator:
    """Orchestrates 2-stage quality gate flow for MAWC conservative variant.

    Per D-01: 2-stage execution. Per D-02: wraps existing components.
    Per D-03: 3-label framework. Per D-04: no retry.
    """

    BASELINE_ROI_THRESHOLD: float = 87.8
    RECOVERED_THRESHOLD: float = 90.0
    BET_COUNT_RATIO_THRESHOLD: float = 0.95

    def _should_run(self, output_path: Path, force: bool) -> bool:
        """Check if a step should run based on artifact existence."""
        if force:
            return True
        if output_path.exists():
            logger.info("SKIP: %s already exists", output_path)
            return False
        return True

    def _check_manifest_deployed(self, manifest_path: Path) -> bool:
        """Check if at least one surface is deployed in manifest.json."""
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        per_year_surface = manifest.get("per_year_surface", {})
        for year_data in per_year_surface.values():
            for surface_data in year_data.values():
                if surface_data.get("deployed", False):
                    return True

        logger.error("No surfaces deployed in conservative variant")
        return False

    def _run_stage1(self, args: argparse.Namespace) -> Path:
        """Stage 1: Run MAWC Conservative Retrain via subprocess."""
        manifest_path = args.conservative_root / "manifest.json"

        if not self._should_run(manifest_path, args.force):
            logger.info("Stage 1 SKIP: manifest already exists")
            return manifest_path

        cmd = [
            sys.executable,
            "scripts/run_mawc_conservative_retrain.py",
            "--oof-path", str(args.oof_path),
            "--source-model-dir", str(args.source_model_dir),
            "--target-root", str(args.conservative_root),
            "--years", args.years,
        ]
        if args.report:
            cmd.append("--report")

        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, cwd=ROOT,
        )
        if result.returncode != 0:
            logger.error("Stage 1 FAILED: %s", result.stderr)
            sys.exit(1)

        if not manifest_path.exists():
            logger.error("Stage 1 FAILED: manifest.json not created")
            sys.exit(1)

        if not self._check_manifest_deployed(manifest_path):
            logger.error("Stage 1 FAILED: No surfaces deployed")
            sys.exit(1)

        logger.info("Stage 1 COMPLETE: %s", manifest_path)
        return manifest_path

    def _run_oof_validation(self, args: argparse.Namespace) -> dict[str, Any]:
        """Stage 2 step: Run OOFHealthValidator."""
        df = pd.read_parquet(args.oof_path)
        validator = OOFHealthValidator()
        result = validator.validate(df, OOF_PREDICTIONS_PROFILE)

        return {
            "status": result["status"],
            "failures": result.get("failures", []),
            "warnings": result.get("warnings", []),
        }

    def _run_feature_audit(self) -> dict[str, Any]:
        """Stage 2 step: Run FeatureRoutingAudit."""
        results = run_feature_audit()
        return {
            "status": results["overall_status"],
            "critical_models": results.get("critical_models", []),
            "advisory_models": results.get("advisory_models", []),
        }

    def _run_shadow_comparison(self, args: argparse.Namespace) -> Path:
        """Stage 2 step: Run Shadow Comparison via subprocess (~82 min)."""
        output_dir = args.shadow_output_dir
        result_path = output_dir / "shadow_comparison_result.json"

        if not self._should_run(result_path, args.force):
            return result_path

        cmd = [
            sys.executable,
            "scripts/run_shadow_comparison.py",
            "--baseline-root", str(args.source_model_dir),
            "--shadow-root", str(args.conservative_root),
            "--folds",
        ] + args.years.split(",") + [
            "--output-dir", str(output_dir),
            "--baseline-name", "baseline",
            "--shadow-name", "mawc_conservative",
        ]
        if args.report:
            cmd.append("--report")

        # Long-running: do NOT capture output so user sees progress
        subprocess.run(cmd, cwd=ROOT)  # noqa: S603

        if not result_path.exists():
            logger.error("Shadow Comparison FAILED: result not created")
            sys.exit(1)

        return result_path

    def _run_shadow_diagnosis(self, args: argparse.Namespace) -> dict[str, Any]:
        """Stage 2 step: Run ShadowDiagnosis."""
        input_dir = args.shadow_output_dir
        diagnosis_dir = input_dir / "diagnosis"
        diagnosis_result_path = diagnosis_dir / "shadow_diagnosis_result.json"

        if not self._should_run(diagnosis_result_path, args.force):
            return {"status": "SKIP", "path": str(diagnosis_result_path)}

        sd = ShadowDiagnosis(input_dir)
        result = sd.run()
        save_diagnosis_results(result, diagnosis_dir)

        return {"status": "PASS", "path": str(diagnosis_result_path)}

    def _run_deployment_gates(self, args: argparse.Namespace) -> dict[str, Any]:
        """Stage 2 step: Run DeploymentGateEvaluator."""
        result_path = str(args.shadow_output_dir / "shadow_comparison_result.json")
        manifest_path = str(args.shadow_output_dir / "shadow_manifest.json")
        gates_dir = str(args.shadow_output_dir / "gates")

        gate_result = run_deployment_gates(result_path, manifest_path, gates_dir)

        return {
            "status": gate_result.overall_status,
            "conditions": [
                {
                    "name": c.condition_name,
                    "status": c.status,
                    "message": c.message,
                }
                for c in gate_result.conditions
            ],
        }

    def _compute_roi_trend(self, shadow_result: dict[str, Any]) -> str:
        """Compute ROI trend label from shadow comparison result."""
        try:
            overall = shadow_result["overall"]["metrics"]
            # Try shadow variant name first, then generic
            shadow_roi = None
            for key in ("mawc_conservative", "shadow"):
                if key in overall:
                    shadow_roi = overall[key].get("roi")
                    break
            if shadow_roi is None:
                logger.warning("Could not extract shadow ROI from result")
                return "unknown"

            if shadow_roi >= self.RECOVERED_THRESHOLD:
                return "recovered"
            if shadow_roi >= self.BASELINE_ROI_THRESHOLD:
                return "weak_recovery"
            return "not_recovered"
        except (KeyError, TypeError) as e:
            logger.warning("ROI extraction failed: %s", e)
            return "unknown"

    def _compute_deployment_verdict(self, quality_gate: str, roi_trend: str) -> str:
        """Compute deployment verdict from quality gate and ROI trend."""
        if quality_gate == "FAIL":
            return "not_deployable"
        if roi_trend in ("recovered", "weak_recovery"):
            return "deployable"
        if roi_trend == "not_recovered":
            return "manual_review"
        return "manual_review"

    def _extract_roi(self, shadow_result: dict[str, Any], variant: str) -> float | None:
        """Extract ROI for a specific variant from shadow result."""
        try:
            overall = shadow_result["overall"]["metrics"]
            # Try exact variant name
            if variant in overall:
                return overall[variant].get("roi")
            # Try standard names
            for key in ("baseline", "mawc_conservative", "shadow"):
                if key in overall and key == variant:
                    return overall[key].get("roi")
            return None
        except (KeyError, TypeError):
            return None

    def _run_stage2(self, args: argparse.Namespace) -> dict[str, Any]:
        """Run Stage 2: 5 quality checks in sequence, stopping on first FAIL."""
        stage2: dict[str, Any] = {}

        # Step 1: FeatureRoutingAudit
        feature_audit = self._run_feature_audit()
        stage2["feature_audit"] = feature_audit
        if feature_audit["status"] == "FAIL":
            return {"stage2": stage2}

        # Step 2: OOFHealthValidator
        oof_validation = self._run_oof_validation(args)
        stage2["oof_validation"] = oof_validation
        if oof_validation["status"] == "FAIL":
            return {"stage2": stage2}

        # Step 3: Shadow Comparison
        shadow_result_path = self._run_shadow_comparison(args)
        stage2["shadow_comparison"] = {
            "status": "PASS",
            "path": str(shadow_result_path),
        }

        # Step 4: Shadow Diagnosis
        diagnosis = self._run_shadow_diagnosis(args)
        stage2["shadow_diagnosis"] = diagnosis
        if diagnosis.get("status") == "FAIL":
            return {"stage2": stage2}

        # Step 5: DeploymentGateEvaluator
        gates = self._run_deployment_gates(args)
        stage2["deployment_gates"] = gates
        if gates["status"] == "FAIL":
            return {"stage2": stage2}

        return {"stage2": stage2}

    def _aggregate_results(
        self,
        stage_results: dict[str, Any],
        shadow_result_path: Path,
    ) -> dict[str, Any]:
        """Aggregate all results into 3-label framework."""
        # Load shadow comparison result for ROI extraction
        shadow_result: dict[str, Any] = {}
        if shadow_result_path.exists():
            with open(shadow_result_path, encoding="utf-8") as f:
                shadow_result = json.load(f)

        # Compute quality_gate from stage2 results
        stage2 = stage_results.get("stage2", {})
        all_pass = True
        for step_name, step_result in stage2.items():
            if isinstance(step_result, dict) and step_result.get("status") in ("FAIL",):
                all_pass = False
                break
        quality_gate = "PASS" if all_pass else "FAIL"

        # Compute roi_trend
        roi_trend = self._compute_roi_trend(shadow_result)

        # Extract ROI values
        baseline_roi = self._extract_roi(shadow_result, "baseline")
        shadow_roi = self._extract_roi(shadow_result, "mawc_conservative")

        # Compute deployment verdict
        deployment = self._compute_deployment_verdict(quality_gate, roi_trend)

        # Collect artifacts
        artifacts: dict[str, str] = {}
        if stage_results.get("stage1", {}).get("manifest_path"):
            artifacts["manifest"] = stage_results["stage1"]["manifest_path"]
        for step_name, step_result in stage2.items():
            if isinstance(step_result, dict) and step_result.get("path"):
                artifacts[step_name] = step_result["path"]

        return {
            "phase": "46-quality-gate-verification",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stage1": stage_results.get("stage1", {}),
            "stage2": stage2,
            "quality_gate": quality_gate,
            "roi_trend": roi_trend,
            "deployment": deployment,
            "baseline_roi": baseline_roi,
            "shadow_roi": shadow_roi,
            "artifacts": artifacts,
        }

    def _write_results(self, result: dict[str, Any], output_dir: Path) -> None:
        """Write JSON result and Markdown summary."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = output_dir / "phase46_quality_gate_result.json"
        json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        # Markdown summary
        md_lines = self._build_markdown_summary(result)
        md_path = output_dir / "phase46_quality_gate_summary.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        logger.info("Results written to %s", output_dir)

    def _build_markdown_summary(self, result: dict[str, Any]) -> list[str]:
        """Build Markdown summary lines."""
        lines: list[str] = [
            "# Phase 46: Quality Gate Verification Summary",
            "",
            f"**Generated:** {result.get('generated_at', 'N/A')}",
            "",
            "## Final Verdict",
            "",
            "| Label | Value |",
            "|-------|-------|",
            f"| Quality Gate | {result.get('quality_gate', 'N/A')} |",
            f"| ROI Trend | {result.get('roi_trend', 'N/A')} |",
            f"| Deployment | {result.get('deployment', 'N/A')} |",
            "",
            "## ROI Comparison",
            "",
            "| Metric | Baseline | Shadow (Conservative) |",
            "|--------|----------|----------------------|",
            f"| ROI | {result.get('baseline_roi', 'N/A')}% | {result.get('shadow_roi', 'N/A')}% |",
            "",
            "## Stage 1: MAWC Conservative Retrain",
            "",
        ]

        stage1 = result.get("stage1", {})
        if stage1:
            lines.append(f"- Manifest: {stage1.get('manifest_path', 'N/A')}")
            deployed = stage1.get("deployed_surfaces", [])
            lines.append(f"- Deployed surfaces: {', '.join(deployed) if deployed else 'none'}")
            lines.append(f"- Status: {stage1.get('status', 'N/A')}")
        else:
            lines.append("- (skipped)")
        lines.append("")

        # Stage 2 steps
        lines.append("## Stage 2: Quality Gate Steps")
        lines.append("")
        lines.append("| Step | Status | Details |")
        lines.append("|------|--------|---------|")

        stage2 = result.get("stage2", {})
        for step_name, step_result in stage2.items():
            if isinstance(step_result, dict):
                status = step_result.get("status", "N/A")
                if step_name == "feature_audit":
                    n_crit = len(step_result.get("critical_models", []))
                    n_adv = len(step_result.get("advisory_models", []))
                    detail = f"{n_crit} critical, {n_adv} advisory"
                elif step_name == "oof_validation":
                    failures = step_result.get("failures", [])
                    detail = f"{len(failures)} failures" if failures else "OK"
                elif step_name == "deployment_gates":
                    conditions = step_result.get("conditions", [])
                    passed = sum(1 for c in conditions if c.get("status") == "PASS")
                    detail = f"{passed}/{len(conditions)} conditions passed"
                else:
                    detail = step_result.get("path", "")
                lines.append(f"| {step_name} | {status} | {detail} |")
        lines.append("")

        # Artifacts
        lines.append("## Artifacts")
        lines.append("")
        artifacts = result.get("artifacts", {})
        for name, path in artifacts.items():
            lines.append(f"- {name}: {path}")
        if not artifacts:
            lines.append("- (none)")
        lines.append("")

        # Notes
        lines.append("## Notes")
        lines.append("")
        if result.get("quality_gate") == "FAIL":
            for step_name, step_result in stage2.items():
                if isinstance(step_result, dict) and step_result.get("status") == "FAIL":
                    failures = step_result.get("failures", [])
                    lines.append(
                        f"**FAILED at {step_name}**: "
                        + ("; ".join(failures) if failures else "See logs for details")
                    )
                    break
        else:
            lines.append("All quality gate checks passed.")
        lines.append("")

        return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for Phase 46 Quality Gate CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 46: Quality Gate Verification -- orchestrate 2-stage "
            "quality gate flow for MAWC conservative variant"
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
        "--conservative-root",
        type=Path,
        default=Path("data/models-backtest-mawc-conservative"),
        help="Conservative variant directory (default: data/models-backtest-mawc-conservative)",
    )
    parser.add_argument(
        "--shadow-output-dir",
        type=Path,
        default=Path("data/backtest/shadow_mawc_conservative"),
        help="Shadow Comparison output directory (default: data/backtest/shadow_mawc_conservative)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/backtest/phase46_quality_gates"),
        help="Phase 46 result output directory (default: data/backtest/phase46_quality_gates)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2024,2025",
        help="Comma-separated test years (default: 2024,2025)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run specific stage only (1 or 2). Default: auto-detect",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run all steps even if artifacts exist",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate HTML report for Shadow Comparison",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """Main entry point for Phase 46 Quality Gate Verification."""
    orch = QualityGateOrchestrator()

    stage_results: dict[str, Any] = {}

    # Determine stage execution plan
    if args.stage == 1:
        # Run only Stage 1
        manifest_path = orch._run_stage1(args)
        stage_results["stage1"] = {
            "status": "COMPLETE",
            "manifest_path": str(manifest_path),
        }
        print(f"\nStage 1 complete. Manifest: {manifest_path}")
        return

    if args.stage is None:
        # Auto-detect: skip Stage 1 if manifest exists
        manifest_path = args.conservative_root / "manifest.json"
        if manifest_path.exists() and not args.force:
            logger.info("Stage 1 SKIP: manifest already exists")
            stage_results["stage1"] = {
                "status": "SKIP",
                "manifest_path": str(manifest_path),
            }
        else:
            manifest_path = orch._run_stage1(args)
            stage_results["stage1"] = {
                "status": "COMPLETE",
                "manifest_path": str(manifest_path),
            }

    # Stage 2
    stage2_results = orch._run_stage2(args)
    stage_results.update(stage2_results)

    # Aggregate
    shadow_result_path = args.shadow_output_dir / "shadow_comparison_result.json"
    result = orch._aggregate_results(stage_results, shadow_result_path)

    # Write results
    orch._write_results(result, args.output_dir)

    # Print summary
    print()
    print("Phase 46 Quality Gate Verification")
    print("=" * 60)
    print(f"Quality Gate: {result['quality_gate']}")
    print(
        f"ROI Trend: {result['roi_trend']} "
        f"(baseline={result.get('baseline_roi', 'N/A')}%, "
        f"shadow={result.get('shadow_roi', 'N/A')}%)"
    )
    print(f"Deployment: {result['deployment']}")
    print(f"Results: {args.output_dir}/phase46_quality_gate_result.json")
    print("=" * 60)

    if result["quality_gate"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
