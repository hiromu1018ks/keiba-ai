"""DeploymentGateEvaluator のユニットテスト (SAF-03)

DeploymentGateEvaluator が Phase 41 の shadow_comparison_result.json と
shadow_manifest.json を読み取り、GatePolicy に基づいて PASS/FAIL/WARN を
判定することを検証する。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shadow_result_json(
    *,
    folds: dict[str, dict[str, Any]] | None = None,
    overall: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """shadow_comparison_result.json の最小構造を生成.

    デフォルトでは2fold (2024/2025) で baseline/shadow のメトリクスが同一.
    """
    default_metrics: dict[str, dict[str, Any]] = {
        "baseline": {
            "brier": 0.15,
            "logloss": 0.45,
            "ece": 0.05,
            "roi": 0.10,
            "bet_count": 1500,
            "hit_rate": 0.30,
            "avg_odds": 8.0,
            "max_drawdown": 0.15,
            "clv": 0.03,
            "clv_available": True,
            "selection_agreement": None,
            "avg_investment_score": 0.55,
            "actual_predicted_ratio": 1.02,
        },
        "shadow": {
            "brier": 0.15,
            "logloss": 0.45,
            "ece": 0.05,
            "roi": 0.12,
            "bet_count": 1500,
            "hit_rate": 0.31,
            "avg_odds": 7.8,
            "max_drawdown": 0.12,
            "clv": 0.04,
            "clv_available": True,
            "selection_agreement": None,
            "avg_investment_score": 0.60,
            "actual_predicted_ratio": 1.01,
        },
    }

    default_folds: dict[str, dict[str, Any]] = {}
    for year in ["2024", "2025"]:
        default_folds[year] = {
            "metrics": {k: dict(v) for k, v in default_metrics.items()},
            "selection_agreement": 0.85,
            "bet_counts": {"baseline": 1500, "shadow": 1500},
        }

    if overall is None:
        overall = {"metrics": {k: dict(v) for k, v in default_metrics.items()}}

    return {
        "generated_at": "2026-05-28T00:00:00+00:00",
        "folds": folds if folds is not None else default_folds,
        "overall": overall,
    }


def _make_manifest_json(
    *,
    artifacts: dict[str, Any] | None = None,
    variants: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """shadow_manifest.json の最小構造を生成."""
    default_variants = [
        {
            "variant_name": "baseline",
            "flag_states": {
                "enable_market_aware_calibrator": False,
                "enable_race_level_ranker": False,
            },
            "baseline_definition": "MAWC/ranker disabled",
        },
        {
            "variant_name": "shadow",
            "flag_states": {
                "enable_market_aware_calibrator": True,
                "enable_race_level_ranker": True,
            },
        },
    ]
    return {
        "generated_at": "2026-05-28T00:00:00+00:00",
        "framework_version": "1.0",
        "variants": variants if variants is not None else default_variants,
        "folds": [
            {"year": 2024, "train_start": "2020-01-01", "train_end": "2023-12-31",
             "test_start": "2024-01-01", "test_end": "2024-12-31"},
            {"year": 2025, "train_start": "2021-01-01", "train_end": "2024-12-31",
             "test_start": "2025-01-01", "test_end": "2025-12-31"},
        ],
        "artifacts": artifacts if artifacts is not None else {
            "metrics_json": {
                "path": "shadow_comparison_result.json",
                "sha256": "abc123",
            },
            "race_diff_parquet": {
                "path": "shadow_race_diff.parquet",
                "sha256": "def456",
            },
        },
    }


def _write_json_files(
    tmp_path: Path,
    result: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[Path, Path | None]:
    """Write result and optional manifest to tmp_path."""
    result_path = tmp_path / "shadow_comparison_result.json"
    result_data = result if result is not None else _make_shadow_result_json()
    result_path.write_text(
        json.dumps(result_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest_path: Path | None = None
    if manifest is not None:
        manifest_path = tmp_path / "shadow_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return result_path, manifest_path


# ---------------------------------------------------------------------------
# Import target
# ---------------------------------------------------------------------------

# Import after helpers so the module existence check is meaningful
from backtest.deployment_gates import (  # noqa: E402
    DEFAULT_GATE_POLICY,
    DeploymentGateEvaluator,
    GateEvaluationResult,
    GateConditionResult,
    GatePolicy,
)


# ===========================================================================
# Test 1-2: GatePolicy
# ===========================================================================


class TestGatePolicy:
    """GatePolicy frozen dataclass tests."""

    def test_frozen(self) -> None:
        """Test 1: GatePolicy is frozen (raises FrozenInstanceError)."""
        policy = GatePolicy()
        with pytest.raises(dataclasses.FrozenInstanceError):
            policy.brier_tolerance = 0.01  # type: ignore[misc]

    def test_default_values(self) -> None:
        """Test 2: DEFAULT_GATE_POLICY has expected threshold values."""
        assert DEFAULT_GATE_POLICY.brier_tolerance == 1e-6
        assert DEFAULT_GATE_POLICY.logloss_tolerance == 1e-6
        assert DEFAULT_GATE_POLICY.ece_tolerance == 1e-6
        assert DEFAULT_GATE_POLICY.bet_count_ratio_threshold == 0.95
        assert DEFAULT_GATE_POLICY.require_oof_pass is True
        assert DEFAULT_GATE_POLICY.require_audit_pass is True
        assert DEFAULT_GATE_POLICY.require_manifest_complete is True


# ===========================================================================
# Test 3-18: DeploymentGateEvaluator
# ===========================================================================


class TestDeploymentGateEvaluator:
    """DeploymentGateEvaluator comprehensive tests."""

    # ------------------------------------------------------------------
    # Test 3: All gates PASS with identical metrics
    # ------------------------------------------------------------------
    def test_all_gates_pass_with_identical_metrics(self, tmp_path: Path) -> None:
        """Test 3: Identical metrics produce overall PASS."""
        result_path, manifest_path = _write_json_files(tmp_path)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        assert isinstance(result, GateEvaluationResult)
        assert result.overall_status == "PASS"

        # All conditions should be PASS or SKIP
        for cond in result.conditions:
            assert cond.status in ("PASS", "SKIP"), (
                f"Condition '{cond.condition_name}' has status {cond.status}: {cond.message}"
            )

    # ------------------------------------------------------------------
    # Test 4: Brier shadow > baseline + tolerance produces FAIL
    # ------------------------------------------------------------------
    def test_brier_worse_produces_fail(self, tmp_path: Path) -> None:
        """Test 4: Brier shadow > baseline + tolerance -> FAIL."""
        result_data = _make_shadow_result_json()
        # Make shadow brier worse in fold 2024
        result_data["folds"]["2024"]["metrics"]["shadow"]["brier"] = 0.20
        result_data["overall"]["metrics"]["shadow"]["brier"] = 0.20

        result_path, manifest_path = _write_json_files(tmp_path, result=result_data)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        assert result.overall_status == "FAIL"
        brier_conditions = [
            c for c in result.conditions
            if "brier" in c.condition_name.lower() and c.status == "FAIL"
        ]
        assert len(brier_conditions) > 0, "Expected FAIL for brier gate"

    # ------------------------------------------------------------------
    # Test 5: Logloss shadow > baseline + tolerance produces FAIL
    # ------------------------------------------------------------------
    def test_logloss_worse_produces_fail(self, tmp_path: Path) -> None:
        """Test 5: Logloss shadow > baseline + tolerance -> FAIL."""
        result_data = _make_shadow_result_json()
        result_data["folds"]["2024"]["metrics"]["shadow"]["logloss"] = 0.55
        result_data["overall"]["metrics"]["shadow"]["logloss"] = 0.55

        result_path, manifest_path = _write_json_files(tmp_path, result=result_data)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        assert result.overall_status == "FAIL"
        logloss_conditions = [
            c for c in result.conditions
            if "logloss" in c.condition_name.lower() and c.status == "FAIL"
        ]
        assert len(logloss_conditions) > 0, "Expected FAIL for logloss gate"

    # ------------------------------------------------------------------
    # Test 6: ECE shadow > baseline + tolerance produces FAIL
    # ------------------------------------------------------------------
    def test_ece_worse_produces_fail(self, tmp_path: Path) -> None:
        """Test 6: ECE shadow > baseline + tolerance -> FAIL."""
        result_data = _make_shadow_result_json()
        result_data["folds"]["2024"]["metrics"]["shadow"]["ece"] = 0.10
        result_data["overall"]["metrics"]["shadow"]["ece"] = 0.10

        result_path, manifest_path = _write_json_files(tmp_path, result=result_data)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        assert result.overall_status == "FAIL"
        ece_conditions = [
            c for c in result.conditions
            if "ece" in c.condition_name.lower() and c.status == "FAIL"
        ]
        assert len(ece_conditions) > 0, "Expected FAIL for ECE gate"

    # ------------------------------------------------------------------
    # Test 7: Bet count shadow < baseline * 0.95 produces FAIL
    # ------------------------------------------------------------------
    def test_bet_count_below_threshold_produces_fail(self, tmp_path: Path) -> None:
        """Test 7: shadow bet_count < baseline * 0.95 -> FAIL."""
        result_data = _make_shadow_result_json()
        # 1400 < 1500 * 0.95 = 1425
        result_data["folds"]["2024"]["metrics"]["shadow"]["bet_count"] = 1400
        result_data["folds"]["2024"]["bet_counts"]["shadow"] = 1400

        result_path, manifest_path = _write_json_files(tmp_path, result=result_data)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        assert result.overall_status == "FAIL"
        bet_conditions = [
            c for c in result.conditions
            if "bet_count" in c.condition_name.lower() and c.status == "FAIL"
        ]
        assert len(bet_conditions) > 0, "Expected FAIL for bet count gate"

    # ------------------------------------------------------------------
    # Test 8: Bet count shadow >= baseline * 0.95 produces PASS
    # ------------------------------------------------------------------
    def test_bet_count_at_threshold_produces_pass(self, tmp_path: Path) -> None:
        """Test 8: shadow bet_count >= baseline * 0.95 -> PASS for bet count."""
        result_data = _make_shadow_result_json()
        # 1430 >= 1500 * 0.95 = 1425
        result_data["folds"]["2024"]["metrics"]["shadow"]["bet_count"] = 1430
        result_data["folds"]["2024"]["bet_counts"]["shadow"] = 1430

        result_path, manifest_path = _write_json_files(tmp_path, result=result_data)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        # Check bet count conditions are all PASS
        bet_conditions = [
            c for c in result.conditions
            if "bet_count" in c.condition_name.lower()
        ]
        for bc in bet_conditions:
            assert bc.status == "PASS", (
                f"Bet count condition '{bc.condition_name}' should be PASS: {bc.message}"
            )

    # ------------------------------------------------------------------
    # Test 9: Missing fold year in metrics produces WARN
    # ------------------------------------------------------------------
    def test_missing_fold_year_produces_warn(self, tmp_path: Path) -> None:
        """Test 9: Fold year present in result but missing metrics -> WARN."""
        result_data = _make_shadow_result_json()
        # Remove metrics for one fold but keep the fold entry
        result_data["folds"]["2024"]["metrics"] = {}

        result_path, manifest_path = _write_json_files(tmp_path, result=result_data)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        warn_conditions = [c for c in result.conditions if c.status == "WARN"]
        assert len(warn_conditions) > 0, "Expected WARN conditions for missing fold data"

    # ------------------------------------------------------------------
    # Test 10: actual/predicted ratio worse than baseline produces WARN
    # ------------------------------------------------------------------
    def test_actual_predicted_ratio_worse_produces_warn(self, tmp_path: Path) -> None:
        """Test 10: actual/predicted ratio worse than baseline -> WARN (not FAIL)."""
        result_data = _make_shadow_result_json()
        # baseline ratio is 1.02 (close to 1.0 is good), shadow worse
        result_data["folds"]["2024"]["metrics"]["shadow"]["actual_predicted_ratio"] = 1.20
        result_data["overall"]["metrics"]["shadow"]["actual_predicted_ratio"] = 1.20

        result_path, manifest_path = _write_json_files(tmp_path, result=result_data)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        # Should produce WARN, not FAIL
        ratio_conditions = [
            c for c in result.conditions
            if "actual_predicted_ratio" in c.condition_name.lower()
            or "ratio" in c.condition_name.lower()
        ]
        for rc in ratio_conditions:
            if "actual_predicted" in rc.condition_name.lower():
                assert rc.status == "WARN", (
                    f"Expected WARN for ratio condition, got {rc.status}: {rc.message}"
                )

        # Overall should be WARN (not FAIL) if only ratio warnings exist
        # (unless other gates fail too)
        # At minimum, we should not have FAIL solely from ratio
        has_other_fail = any(
            c.status == "FAIL"
            and "actual_predicted" not in c.condition_name.lower()
            for c in result.conditions
        )
        if not has_other_fail:
            assert result.overall_status == "WARN"

    # ------------------------------------------------------------------
    # Test 11: Missing manifest produces FAIL for artifact reproducibility
    # ------------------------------------------------------------------
    def test_missing_manifest_produces_fail(self, tmp_path: Path) -> None:
        """Test 11: No manifest -> FAIL for artifact reproducibility."""
        result_data = _make_shadow_result_json()
        result_path, _ = _write_json_files(tmp_path, result=result_data, manifest=None)

        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, None)

        artifact_conditions = [
            c for c in result.conditions
            if "artifact" in c.condition_name.lower() or "manifest" in c.condition_name.lower()
        ]
        assert any(c.status == "FAIL" for c in artifact_conditions), (
            "Expected FAIL for artifact reproducibility when manifest is missing"
        )
        assert result.overall_status == "FAIL"

    # ------------------------------------------------------------------
    # Test 12: Manifest SHA256 mismatch produces FAIL
    # ------------------------------------------------------------------
    def test_sha256_mismatch_produces_fail(self, tmp_path: Path) -> None:
        """Test 12: SHA256 mismatch in manifest -> FAIL."""
        result_data = _make_shadow_result_json()

        # Create the actual result file so we can compute its SHA256
        result_path = tmp_path / "shadow_comparison_result.json"
        result_path.write_text(
            json.dumps(result_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Manifest has wrong SHA256
        manifest_data = _make_manifest_json(artifacts={
            "metrics_json": {
                "path": "shadow_comparison_result.json",
                "sha256": "wrong_hash_value",
            },
        })
        manifest_path = tmp_path / "shadow_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        sha_conditions = [
            c for c in result.conditions
            if "sha256" in c.condition_name.lower() or "reproducibility" in c.condition_name.lower()
        ]
        assert any(c.status == "FAIL" for c in sha_conditions), (
            "Expected FAIL for SHA256 mismatch"
        )

    # ------------------------------------------------------------------
    # Test 13: Missing artifacts section in manifest produces FAIL
    # ------------------------------------------------------------------
    def test_missing_artifacts_section_produces_fail(self, tmp_path: Path) -> None:
        """Test 13: No artifacts section in manifest -> FAIL."""
        result_data = _make_shadow_result_json()
        manifest_data = _make_manifest_json()
        del manifest_data["artifacts"]

        result_path, manifest_path = _write_json_files(
            tmp_path, result=result_data, manifest=manifest_data
        )
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        artifact_conditions = [
            c for c in result.conditions
            if "artifact" in c.condition_name.lower() or "manifest" in c.condition_name.lower()
        ]
        assert any(c.status == "FAIL" for c in artifact_conditions), (
            "Expected FAIL for missing artifacts section"
        )

    # ------------------------------------------------------------------
    # Test 14: Evaluation result includes per-condition details
    # ------------------------------------------------------------------
    def test_evaluation_result_includes_per_condition_details(self, tmp_path: Path) -> None:
        """Test 14: GateEvaluationResult has conditions with status and message."""
        result_path, manifest_path = _write_json_files(tmp_path)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        assert isinstance(result.conditions, list)
        assert len(result.conditions) > 0
        for cond in result.conditions:
            assert isinstance(cond, GateConditionResult)
            assert cond.status in ("PASS", "FAIL", "WARN", "SKIP")
            assert isinstance(cond.condition_name, str)
            assert len(cond.condition_name) > 0
            assert isinstance(cond.message, str)

    # ------------------------------------------------------------------
    # Test 15: Overall verdict is FAIL if any required gate FAILs
    # ------------------------------------------------------------------
    def test_overall_fail_if_any_gate_fails(self, tmp_path: Path) -> None:
        """Test 15: Single FAIL condition -> overall FAIL."""
        result_data = _make_shadow_result_json()
        # Make brier worse in just one fold
        result_data["folds"]["2024"]["metrics"]["shadow"]["brier"] = 0.20
        result_data["overall"]["metrics"]["shadow"]["brier"] = 0.20

        result_path, manifest_path = _write_json_files(tmp_path, result=result_data)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        assert result.overall_status == "FAIL"

    # ------------------------------------------------------------------
    # Test 16: Overall verdict is WARN if only WARN conditions exist
    # ------------------------------------------------------------------
    def test_overall_warn_if_only_warnings(self, tmp_path: Path) -> None:
        """Test 16: Only WARN conditions -> overall WARN."""
        result_data = _make_shadow_result_json()
        # Make actual/predicted ratio worse (WARN only per D-11)
        result_data["folds"]["2024"]["metrics"]["shadow"]["actual_predicted_ratio"] = 1.50
        result_data["folds"]["2025"]["metrics"]["shadow"]["actual_predicted_ratio"] = 1.50
        result_data["overall"]["metrics"]["shadow"]["actual_predicted_ratio"] = 1.50

        result_path, manifest_path = _write_json_files(tmp_path, result=result_data)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        # Should be WARN because ratio degradation is WARN, not FAIL
        assert result.overall_status == "WARN"

    # ------------------------------------------------------------------
    # Test 17: evaluate() with missing result file raises FileNotFoundError
    # ------------------------------------------------------------------
    def test_missing_result_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Test 17: Non-existent result file -> FileNotFoundError."""
        evaluator = DeploymentGateEvaluator()
        with pytest.raises(FileNotFoundError):
            evaluator.evaluate(tmp_path / "nonexistent.json", None)

    # ------------------------------------------------------------------
    # Test 18: Non-gate metrics appear in report section only
    # ------------------------------------------------------------------
    def test_non_gate_metrics_in_report_only(self, tmp_path: Path) -> None:
        """Test 18: selection_agreement and ROI are in report_metrics, not conditions."""
        result_path, manifest_path = _write_json_files(tmp_path)
        evaluator = DeploymentGateEvaluator()
        result = evaluator.evaluate(result_path, manifest_path)

        # selection_agreement and ROI should be in report_metrics
        assert isinstance(result.report_metrics, dict)

        # Conditions should NOT contain gates for selection_agreement or ROI
        condition_names = [c.condition_name.lower() for c in result.conditions]
        gate_metrics_in_conditions = [
            name for name in condition_names
            if "selection_agreement" in name or "roi" in name
        ]
        assert len(gate_metrics_in_conditions) == 0, (
            f"selection_agreement/ROI should not be gate conditions: {gate_metrics_in_conditions}"
        )

        # report_metrics should contain these values
        assert "selection_agreement" in result.report_metrics or "roi" in result.report_metrics
