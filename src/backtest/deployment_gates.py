"""Deployment Gate Evaluator (SAF-03).

Reads Phase 41 shadow comparison artifacts (shadow_comparison_result.json,
shadow_manifest.json) and produces a structured PASS/FAIL/WARN report.

D-09: Independent evaluator -- no RacePredictor integration.
D-10: GatePolicy frozen dataclass with explicit thresholds.
D-11: Specific gate conditions with tolerances.
D-12: Report only -- never modifies deployment_status or models.
D-13: Auto-deploy deferred to v2.2.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatePolicy:
    """Gate evaluation policy with explicit thresholds (D-10, D-11).

    Frozen to prevent accidental mutation at runtime.
    """

    brier_tolerance: float = 1e-6
    logloss_tolerance: float = 1e-6
    ece_tolerance: float = 1e-6
    bet_count_ratio_threshold: float = 0.95
    require_oof_pass: bool = True
    require_audit_pass: bool = True
    require_manifest_complete: bool = True


DEFAULT_GATE_POLICY = GatePolicy()


@dataclass
class GateConditionResult:
    """Single gate condition evaluation result."""

    condition_name: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateEvaluationResult:
    """Overall gate evaluation result."""

    overall_status: str  # "PASS", "FAIL", "WARN"
    policy: GatePolicy
    conditions: list[GateConditionResult]
    report_metrics: dict[str, Any]
    generated_at: str


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class DeploymentGateEvaluator:
    """Evaluates deployment gates against shadow comparison artifacts (D-09).

    Reads shadow_comparison_result.json and optionally shadow_manifest.json,
    applies GatePolicy conditions, and returns a GateEvaluationResult.

    This evaluator produces a report only and does NOT modify deployment_status
    or models (D-12).
    """

    def __init__(self, policy: GatePolicy | None = None) -> None:
        self.policy = policy or DEFAULT_GATE_POLICY

    def evaluate(
        self,
        result_path: str | Path,
        manifest_path: str | Path | None = None,
    ) -> GateEvaluationResult:
        """Evaluate all deployment gates.

        Args:
            result_path: Path to shadow_comparison_result.json.
            manifest_path: Optional path to shadow_manifest.json.

        Returns:
            GateEvaluationResult with overall verdict and per-condition details.

        Raises:
            FileNotFoundError: If result_path does not exist.
        """
        result_path = Path(result_path)
        if not result_path.exists():
            raise FileNotFoundError(
                f"Shadow comparison result not found: {result_path}"
            )

        # Load result JSON
        with open(result_path, encoding="utf-8") as f:
            result_data = json.load(f)

        # Load manifest if provided
        manifest_data: dict[str, Any] | None = None
        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as f:
                    manifest_data = json.load(f)

        conditions: list[GateConditionResult] = []
        report_metrics: dict[str, Any] = {}

        # Identify variant names from manifest flag_states (Pitfall 4)
        baseline_name, shadow_name = self._identify_variants(
            result_data, manifest_data
        )

        # Evaluate probability quality gates per fold and overall
        self._evaluate_probability_quality(
            result_data, baseline_name, shadow_name, conditions
        )

        # Evaluate bet count preservation
        self._evaluate_bet_count(
            result_data, baseline_name, shadow_name, conditions
        )

        # Evaluate actual/predicted ratio (WARN only per D-11)
        self._evaluate_actual_predicted_ratio(
            result_data, baseline_name, shadow_name, conditions
        )

        # Evaluate artifact reproducibility
        self._evaluate_artifact_reproducibility(
            result_path, manifest_data, conditions
        )

        # Diagnostic gates (OOF, audit) -- SKIP placeholders per D-05
        self._evaluate_diagnostics(conditions)

        # Collect non-gate report metrics (D-11)
        self._collect_report_metrics(
            result_data, baseline_name, shadow_name, report_metrics
        )

        # Determine overall status
        has_fail = any(c.status == "FAIL" for c in conditions)
        has_warn = any(c.status == "WARN" for c in conditions)

        if has_fail:
            overall_status = "FAIL"
        elif has_warn:
            overall_status = "WARN"
        else:
            overall_status = "PASS"

        return GateEvaluationResult(
            overall_status=overall_status,
            policy=self.policy,
            conditions=conditions,
            report_metrics=report_metrics,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Variant identification (Pitfall 4)
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_variants(
        result_data: dict[str, Any],
        manifest_data: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Identify baseline and shadow variant names from manifest or result.

        Uses manifest flag_states: variant with enable_market_aware_calibrator=False
        is baseline, variant with enable_market_aware_calibrator=True is shadow.
        Falls back to first/second variant in result if manifest is unavailable.
        """
        if manifest_data and "variants" in manifest_data:
            variants = manifest_data["variants"]
            baseline_name = ""
            shadow_name = ""
            for v in variants:
                flags = v.get("flag_states", {})
                if not flags.get("enable_market_aware_calibrator", True):
                    baseline_name = v.get("variant_name", "")
                if flags.get("enable_market_aware_calibrator", False):
                    shadow_name = v.get("variant_name", "")
            if baseline_name and shadow_name:
                return baseline_name, shadow_name

        # Fallback: use variant names from result data
        folds = result_data.get("folds", {})
        if folds:
            first_fold = next(iter(folds.values()))
            metrics = first_fold.get("metrics", {})
            variant_names = list(metrics.keys())
            if len(variant_names) >= 2:
                return variant_names[0], variant_names[1]

        return "baseline", "shadow"

    # ------------------------------------------------------------------
    # Probability quality gates (D-11)
    # ------------------------------------------------------------------

    def _evaluate_probability_quality(
        self,
        result_data: dict[str, Any],
        baseline_name: str,
        shadow_name: str,
        conditions: list[GateConditionResult],
    ) -> None:
        """Evaluate Brier, logloss, ECE gates per fold and overall."""
        metrics_to_check = [
            ("brier", self.policy.brier_tolerance),
            ("logloss", self.policy.logloss_tolerance),
            ("ece", self.policy.ece_tolerance),
        ]

        # Per-fold evaluation
        folds = result_data.get("folds", {})
        for year, fold_data in folds.items():
            fold_metrics = fold_data.get("metrics", {})
            bl = fold_metrics.get(baseline_name, {})
            sh = fold_metrics.get(shadow_name, {})

            if not bl or not sh:
                conditions.append(GateConditionResult(
                    condition_name=f"probability_quality_fold_{year}",
                    status="WARN",
                    message=f"Fold {year} missing baseline or shadow metrics",
                    details={"year": year},
                ))
                continue

            for metric_name, tolerance in metrics_to_check:
                bl_val = bl.get(metric_name)
                sh_val = sh.get(metric_name)
                if bl_val is None or sh_val is None:
                    continue
                if sh_val > bl_val + tolerance:
                    conditions.append(GateConditionResult(
                        condition_name=f"{metric_name}_fold_{year}",
                        status="FAIL",
                        message=(
                            f"Fold {year}: shadow {metric_name}={sh_val:.6f} "
                            f"> baseline {bl_val:.6f} + tolerance {tolerance}"
                        ),
                        details={
                            "year": year,
                            "metric": metric_name,
                            "shadow_value": sh_val,
                            "baseline_value": bl_val,
                            "tolerance": tolerance,
                        },
                    ))
                else:
                    conditions.append(GateConditionResult(
                        condition_name=f"{metric_name}_fold_{year}",
                        status="PASS",
                        message=(
                            f"Fold {year}: shadow {metric_name}={sh_val:.6f} "
                            f"<= baseline {bl_val:.6f} + tolerance {tolerance}"
                        ),
                        details={
                            "year": year,
                            "metric": metric_name,
                            "shadow_value": sh_val,
                            "baseline_value": bl_val,
                        },
                    ))

        # Overall evaluation
        overall_metrics = result_data.get("overall", {}).get("metrics", {})
        bl_overall = overall_metrics.get(baseline_name, {})
        sh_overall = overall_metrics.get(shadow_name, {})

        if bl_overall and sh_overall:
            for metric_name, tolerance in metrics_to_check:
                bl_val = bl_overall.get(metric_name)
                sh_val = sh_overall.get(metric_name)
                if bl_val is None or sh_val is None:
                    continue
                if sh_val > bl_val + tolerance:
                    conditions.append(GateConditionResult(
                        condition_name=f"{metric_name}_overall",
                        status="FAIL",
                        message=(
                            f"Overall: shadow {metric_name}={sh_val:.6f} "
                            f"> baseline {bl_val:.6f} + tolerance {tolerance}"
                        ),
                        details={
                            "metric": metric_name,
                            "shadow_value": sh_val,
                            "baseline_value": bl_val,
                            "tolerance": tolerance,
                        },
                    ))
                else:
                    conditions.append(GateConditionResult(
                        condition_name=f"{metric_name}_overall",
                        status="PASS",
                        message=(
                            f"Overall: shadow {metric_name}={sh_val:.6f} "
                            f"<= baseline {bl_val:.6f} + tolerance {tolerance}"
                        ),
                        details={
                            "metric": metric_name,
                            "shadow_value": sh_val,
                            "baseline_value": bl_val,
                        },
                    ))

    # ------------------------------------------------------------------
    # Bet count preservation (D-11)
    # ------------------------------------------------------------------

    def _evaluate_bet_count(
        self,
        result_data: dict[str, Any],
        baseline_name: str,
        shadow_name: str,
        conditions: list[GateConditionResult],
    ) -> None:
        """Evaluate bet count preservation: shadow >= baseline * threshold."""
        folds = result_data.get("folds", {})
        for year, fold_data in folds.items():
            bet_counts = fold_data.get("bet_counts", {})
            bl_count = bet_counts.get(baseline_name)
            sh_count = bet_counts.get(shadow_name)

            # Fallback to metrics bet_count
            if bl_count is None or sh_count is None:
                fold_metrics = fold_data.get("metrics", {})
                if bl_count is None:
                    bl_count = fold_metrics.get(baseline_name, {}).get("bet_count")
                if sh_count is None:
                    sh_count = fold_metrics.get(shadow_name, {}).get("bet_count")

            if bl_count is None or sh_count is None:
                continue

            threshold_count = bl_count * self.policy.bet_count_ratio_threshold
            if sh_count < threshold_count:
                conditions.append(GateConditionResult(
                    condition_name=f"bet_count_preservation_fold_{year}",
                    status="FAIL",
                    message=(
                        f"Fold {year}: shadow bet_count={sh_count} "
                        f"< baseline {bl_count} * {self.policy.bet_count_ratio_threshold} "
                        f"= {threshold_count:.0f}"
                    ),
                    details={
                        "year": year,
                        "shadow_bet_count": sh_count,
                        "baseline_bet_count": bl_count,
                        "threshold": threshold_count,
                    },
                ))
            else:
                conditions.append(GateConditionResult(
                    condition_name=f"bet_count_preservation_fold_{year}",
                    status="PASS",
                    message=(
                        f"Fold {year}: shadow bet_count={sh_count} "
                        f">= baseline {bl_count} * {self.policy.bet_count_ratio_threshold} "
                        f"= {threshold_count:.0f}"
                    ),
                    details={
                        "year": year,
                        "shadow_bet_count": sh_count,
                        "baseline_bet_count": bl_count,
                    },
                ))

    # ------------------------------------------------------------------
    # Actual/predicted ratio (WARN only per D-11)
    # ------------------------------------------------------------------

    def _evaluate_actual_predicted_ratio(
        self,
        result_data: dict[str, Any],
        baseline_name: str,
        shadow_name: str,
        conditions: list[GateConditionResult],
    ) -> None:
        """Evaluate actual/predicted ratio -- WARN if shadow worse than baseline."""
        folds = result_data.get("folds", {})

        # Per-fold
        for year, fold_data in folds.items():
            fold_metrics = fold_data.get("metrics", {})
            bl = fold_metrics.get(baseline_name, {})
            sh = fold_metrics.get(shadow_name, {})

            bl_ratio = bl.get("actual_predicted_ratio")
            sh_ratio = sh.get("actual_predicted_ratio")

            if bl_ratio is None or sh_ratio is None:
                continue

            # "Worse" = further from 1.0 than baseline
            bl_deviation = abs(bl_ratio - 1.0)
            sh_deviation = abs(sh_ratio - 1.0)

            if sh_deviation > bl_deviation:
                conditions.append(GateConditionResult(
                    condition_name=f"actual_predicted_ratio_fold_{year}",
                    status="WARN",
                    message=(
                        f"Fold {year}: shadow ratio={sh_ratio:.3f} worse than "
                        f"baseline ratio={bl_ratio:.3f} (further from 1.0)"
                    ),
                    details={
                        "year": year,
                        "shadow_ratio": sh_ratio,
                        "baseline_ratio": bl_ratio,
                    },
                ))
            else:
                conditions.append(GateConditionResult(
                    condition_name=f"actual_predicted_ratio_fold_{year}",
                    status="PASS",
                    message=(
                        f"Fold {year}: shadow ratio={sh_ratio:.3f} "
                        f"<= baseline ratio={bl_ratio:.3f}"
                    ),
                    details={
                        "year": year,
                        "shadow_ratio": sh_ratio,
                        "baseline_ratio": bl_ratio,
                    },
                ))

        # Overall
        overall_metrics = result_data.get("overall", {}).get("metrics", {})
        bl_overall = overall_metrics.get(baseline_name, {})
        sh_overall = overall_metrics.get(shadow_name, {})

        bl_ratio_overall = bl_overall.get("actual_predicted_ratio")
        sh_ratio_overall = sh_overall.get("actual_predicted_ratio")

        if bl_ratio_overall is not None and sh_ratio_overall is not None:
            bl_dev = abs(bl_ratio_overall - 1.0)
            sh_dev = abs(sh_ratio_overall - 1.0)

            if sh_dev > bl_dev:
                conditions.append(GateConditionResult(
                    condition_name="actual_predicted_ratio_overall",
                    status="WARN",
                    message=(
                        f"Overall: shadow ratio={sh_ratio_overall:.3f} worse than "
                        f"baseline ratio={bl_ratio_overall:.3f} (further from 1.0)"
                    ),
                    details={
                        "shadow_ratio": sh_ratio_overall,
                        "baseline_ratio": bl_ratio_overall,
                    },
                ))
            else:
                conditions.append(GateConditionResult(
                    condition_name="actual_predicted_ratio_overall",
                    status="PASS",
                    message=(
                        f"Overall: shadow ratio={sh_ratio_overall:.3f} "
                        f"<= baseline ratio={bl_ratio_overall:.3f}"
                    ),
                    details={
                        "shadow_ratio": sh_ratio_overall,
                        "baseline_ratio": bl_ratio_overall,
                    },
                ))

    # ------------------------------------------------------------------
    # Artifact reproducibility (D-11)
    # ------------------------------------------------------------------

    def _evaluate_artifact_reproducibility(
        self,
        result_path: Path,
        manifest_data: dict[str, Any] | None,
        conditions: list[GateConditionResult],
    ) -> None:
        """Evaluate artifact reproducibility: manifest SHA256 and completeness."""
        if manifest_data is None:
            if self.policy.require_manifest_complete:
                conditions.append(GateConditionResult(
                    condition_name="artifact_reproducibility",
                    status="FAIL",
                    message="Manifest not provided or not found",
                    details={"manifest_found": False},
                ))
            return

        # Check artifacts section exists
        artifacts = manifest_data.get("artifacts")
        if not artifacts:
            conditions.append(GateConditionResult(
                condition_name="artifact_reproducibility_manifest_completeness",
                status="FAIL",
                message="Manifest missing 'artifacts' section or it is empty",
                details={"artifacts_present": False},
            ))
            return

        # Verify SHA256 of result file against manifest
        metrics_entry = artifacts.get("metrics_json")
        if metrics_entry and "sha256" in metrics_entry:
            expected_sha = metrics_entry["sha256"]
            if result_path.exists():
                actual_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
                if actual_sha != expected_sha:
                    conditions.append(GateConditionResult(
                        condition_name="artifact_reproducibility_sha256",
                        status="FAIL",
                        message=(
                            f"SHA256 mismatch for {result_path.name}: "
                            f"expected={expected_sha}, actual={actual_sha}"
                        ),
                        details={
                            "file": result_path.name,
                            "expected_sha256": expected_sha,
                            "actual_sha256": actual_sha,
                        },
                    ))
                else:
                    conditions.append(GateConditionResult(
                        condition_name="artifact_reproducibility_sha256",
                        status="PASS",
                        message=f"SHA256 verified for {result_path.name}",
                        details={
                            "file": result_path.name,
                            "sha256": actual_sha,
                        },
                    ))

        # Check all artifacts have path and sha256
        missing_fields = []
        for key, entry in artifacts.items():
            if not isinstance(entry, dict):
                missing_fields.append(f"{key}: not a dict")
                continue
            if "path" not in entry:
                missing_fields.append(f"{key}: missing 'path'")
            if "sha256" not in entry:
                missing_fields.append(f"{key}: missing 'sha256'")

        if missing_fields:
            conditions.append(GateConditionResult(
                condition_name="artifact_reproducibility_completeness",
                status="FAIL",
                message=f"Artifact entries incomplete: {'; '.join(missing_fields)}",
                details={"missing_fields": missing_fields},
            ))
        else:
            conditions.append(GateConditionResult(
                condition_name="artifact_reproducibility_completeness",
                status="PASS",
                message="All artifact entries have path and sha256",
                details={"artifact_count": len(artifacts)},
            ))

    # ------------------------------------------------------------------
    # Diagnostic gates (D-05: SKIP placeholders)
    # ------------------------------------------------------------------

    def _evaluate_diagnostics(
        self,
        conditions: list[GateConditionResult],
    ) -> None:
        """Add diagnostic gate conditions (SKIP placeholders per D-05)."""
        if self.policy.require_oof_pass:
            conditions.append(GateConditionResult(
                condition_name="diagnostic_oof_health",
                status="SKIP",
                message="OOF validation requires manual run (D-05)",
                details={"reason": "CI-independent per design decision D-05"},
            ))

        if self.policy.require_audit_pass:
            conditions.append(GateConditionResult(
                condition_name="diagnostic_feature_routing_audit",
                status="SKIP",
                message="Feature routing audit requires manual run (D-05)",
                details={"reason": "CI-independent per design decision D-05"},
            ))

    # ------------------------------------------------------------------
    # Report metrics (non-gate, D-11)
    # ------------------------------------------------------------------

    def _collect_report_metrics(
        self,
        result_data: dict[str, Any],
        baseline_name: str,
        shadow_name: str,
        report_metrics: dict[str, Any],
    ) -> None:
        """Collect non-gate metrics for reporting only (D-11).

        Selection agreement and ROI are NOT gates -- they are diagnostic
        report metrics.
        """
        report_metrics["generated_at"] = result_data.get("generated_at", "")

        # Per-fold report metrics
        folds_report: dict[str, Any] = {}
        folds = result_data.get("folds", {})
        for year, fold_data in folds.items():
            fold_report: dict[str, Any] = {}

            # Selection agreement
            sa = fold_data.get("selection_agreement")
            if sa is not None:
                fold_report["selection_agreement"] = sa

            # ROI from metrics
            fold_metrics = fold_data.get("metrics", {})
            for vname in [baseline_name, shadow_name]:
                vm = fold_metrics.get(vname, {})
                roi = vm.get("roi")
                if roi is not None:
                    fold_report[f"{vname}_roi"] = roi

            folds_report[year] = fold_report

        report_metrics["folds"] = folds_report

        # Overall report metrics
        overall_metrics = result_data.get("overall", {}).get("metrics", {})
        for vname in [baseline_name, shadow_name]:
            vm = overall_metrics.get(vname, {})
            roi = vm.get("roi")
            if roi is not None:
                report_metrics[f"{vname}_roi"] = roi

        # Selection agreement from overall is not typically present,
        # but include any available
        report_metrics["baseline_name"] = baseline_name
        report_metrics["shadow_name"] = shadow_name


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def to_json(
    result: GateEvaluationResult,
    output_path: Path | None = None,
) -> str:
    """Serialize GateEvaluationResult to JSON string.

    Args:
        result: The evaluation result to serialize.
        output_path: Optional file path to write JSON to.

    Returns:
        JSON string of the evaluation result.
    """
    data = {
        "overall_status": result.overall_status,
        "generated_at": result.generated_at,
        "policy": {
            "brier_tolerance": result.policy.brier_tolerance,
            "logloss_tolerance": result.policy.logloss_tolerance,
            "ece_tolerance": result.policy.ece_tolerance,
            "bet_count_ratio_threshold": result.policy.bet_count_ratio_threshold,
            "require_oof_pass": result.policy.require_oof_pass,
            "require_audit_pass": result.policy.require_audit_pass,
            "require_manifest_complete": result.policy.require_manifest_complete,
        },
        "conditions": [
            {
                "condition_name": c.condition_name,
                "status": c.status,
                "message": c.message,
                "details": c.details,
            }
            for c in result.conditions
        ],
        "report_metrics": result.report_metrics,
    }

    json_str = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str, encoding="utf-8")

    return json_str


def to_markdown(result: GateEvaluationResult) -> str:
    """Format GateEvaluationResult as Markdown."""
    lines: list[str] = []
    lines.append("# Deployment Gate Evaluation Report")
    lines.append("")
    lines.append(f"**Overall Status:** {result.overall_status}")
    lines.append(f"**Generated At:** {result.generated_at}")
    lines.append("")

    # Policy section
    lines.append("## Policy Configuration")
    lines.append("")
    lines.append(f"- Brier tolerance: {result.policy.brier_tolerance}")
    lines.append(f"- Logloss tolerance: {result.policy.logloss_tolerance}")
    lines.append(f"- ECE tolerance: {result.policy.ece_tolerance}")
    lines.append(f"- Bet count ratio threshold: {result.policy.bet_count_ratio_threshold}")
    lines.append(f"- Require OOF pass: {result.policy.require_oof_pass}")
    lines.append(f"- Require audit pass: {result.policy.require_audit_pass}")
    lines.append(f"- Require manifest complete: {result.policy.require_manifest_complete}")
    lines.append("")

    # Conditions
    lines.append("## Gate Conditions")
    lines.append("")

    pass_conds = [c for c in result.conditions if c.status == "PASS"]
    fail_conds = [c for c in result.conditions if c.status == "FAIL"]
    warn_conds = [c for c in result.conditions if c.status == "WARN"]
    skip_conds = [c for c in result.conditions if c.status == "SKIP"]

    if fail_conds:
        lines.append("### FAIL")
        for c in fail_conds:
            lines.append(f"- **{c.condition_name}**: {c.message}")
        lines.append("")

    if warn_conds:
        lines.append("### WARN")
        for c in warn_conds:
            lines.append(f"- **{c.condition_name}**: {c.message}")
        lines.append("")

    if pass_conds:
        lines.append("### PASS")
        for c in pass_conds:
            lines.append(f"- **{c.condition_name}**: {c.message}")
        lines.append("")

    if skip_conds:
        lines.append("### SKIP")
        for c in skip_conds:
            lines.append(f"- **{c.condition_name}**: {c.message}")
        lines.append("")

    # Report metrics
    if result.report_metrics:
        lines.append("## Report Metrics (non-gate, informational)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result.report_metrics, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_deployment_gates(
    result_path: str,
    manifest_path: str | None = None,
    output_dir: str | None = None,
) -> GateEvaluationResult:
    """CLI entry function for deployment gate evaluation.

    Evaluates gates, writes JSON + Markdown reports, and exits non-zero on FAIL.

    Args:
        result_path: Path to shadow_comparison_result.json.
        manifest_path: Optional path to shadow_manifest.json.
        output_dir: Optional directory for output reports.

    Returns:
        GateEvaluationResult.
    """
    evaluator = DeploymentGateEvaluator()
    result = evaluator.evaluate(result_path, manifest_path)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        to_json(result, out / "deployment_gate_result.json")
        md_path = out / "deployment_gate_report.md"
        md_path.write_text(to_markdown(result), encoding="utf-8")

        logger.info("Gate evaluation reports written to %s", out)

    # Log summary
    logger.info(
        "Deployment gate evaluation: %s (%d conditions: %d PASS, %d FAIL, %d WARN, %d SKIP)",
        result.overall_status,
        len(result.conditions),
        sum(1 for c in result.conditions if c.status == "PASS"),
        sum(1 for c in result.conditions if c.status == "FAIL"),
        sum(1 for c in result.conditions if c.status == "WARN"),
        sum(1 for c in result.conditions if c.status == "SKIP"),
    )

    # D-12: Exit non-zero on FAIL
    if result.overall_status == "FAIL":
        logger.error("Deployment gate evaluation FAILED -- see report for details")
        sys.exit(1)

    return result
