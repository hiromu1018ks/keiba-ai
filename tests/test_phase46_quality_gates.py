"""Tests for Phase 46 Quality Gate Orchestrator CLI.

Covers QualityGateOrchestrator Stage 1/2 orchestration, skip/resume,
3-label aggregation, CLI argument parsing, and end-to-end smoke tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(
    tmp_path: Path,
    per_year_surface: dict | None = None,
    deployed: bool = True,
) -> Path:
    """Create a synthetic manifest.json for testing."""
    if per_year_surface is None:
        per_year_surface = {
            "2024": {
                "turf": {"best_c": 0.01, "deployed": deployed},
                "dirt": {"best_c": 0.03, "deployed": deployed},
            },
            "2025": {
                "turf": {"best_c": 0.005, "deployed": deployed},
                "dirt": {"best_c": 0.01, "deployed": deployed},
            },
        }
    manifest = {
        "mawc_fix_version": "45-conservative",
        "source_model_dir": "data/models-backtest",
        "target_variant_dir": str(tmp_path),
        "per_year_surface": per_year_surface,
        "years": ["2024", "2025"],
        "generated_at": "2026-06-01T00:00:00Z",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _make_shadow_result(
    tmp_path: Path,
    shadow_roi: float = 92.0,
    baseline_roi: float = 87.8,
) -> Path:
    """Create a synthetic shadow_comparison_result.json for testing."""
    result = {
        "overall": {
            "metrics": {
                "baseline": {"roi": baseline_roi, "brier": 0.15},
                "mawc_conservative": {"roi": shadow_roi, "brier": 0.14},
            },
        },
        "folds": {
            "2024": {
                "metrics": {
                    "baseline": {"roi": baseline_roi, "brier": 0.15, "bet_count": 500},
                    "mawc_conservative": {"roi": shadow_roi, "brier": 0.14, "bet_count": 490},
                },
                "bet_counts": {"baseline": 500, "mawc_conservative": 490},
            },
            "2025": {
                "metrics": {
                    "baseline": {"roi": baseline_roi, "brier": 0.15, "bet_count": 480},
                    "mawc_conservative": {"roi": shadow_roi, "brier": 0.14, "bet_count": 470},
                },
                "bet_counts": {"baseline": 480, "mawc_conservative": 470},
            },
        },
        "generated_at": "2026-06-01T00:00:00Z",
    }
    output_dir = tmp_path / "shadow_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "shadow_comparison_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Also create manifest
    manifest = {
        "artifacts": {
            "metrics_json": {
                "path": str(result_path),
                "sha256": "abc123",
            },
        },
        "variants": [
            {
                "variant_name": "baseline",
                "flag_states": {"enable_market_aware_calibrator": False},
            },
            {
                "variant_name": "mawc_conservative",
                "flag_states": {"enable_market_aware_calibrator": True},
            },
        ],
    }
    manifest_json = json.dumps(manifest, indent=2)
    (output_dir / "shadow_manifest.json").write_text(manifest_json, encoding="utf-8")

    return output_dir


def _make_args(**overrides: object) -> argparse.Namespace:
    """Create default args namespace with overrides."""
    defaults = {
        "oof_path": Path("data/oof/oof_predictions.parquet"),
        "source_model_dir": Path("data/models-backtest"),
        "conservative_root": Path("data/models-backtest-mawc-conservative"),
        "shadow_output_dir": Path("data/backtest/shadow_mawc_conservative"),
        "output_dir": Path("data/backtest/phase46_quality_gates"),
        "years": "2024,2025",
        "stage": None,
        "force": False,
        "report": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Task 1 Tests: QualityGateOrchestrator core methods
# ---------------------------------------------------------------------------


class TestShouldRun:
    """Tests for _should_run() skip/resume logic."""

    def test_returns_false_when_artifact_exists_and_no_force(self, tmp_path: Path) -> None:
        """Test 1: _should_run() returns False when output artifact exists and force=False."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        artifact = tmp_path / "existing.json"
        artifact.write_text("{}", encoding="utf-8")
        assert orch._should_run(artifact, force=False) is False

    def test_returns_true_when_artifact_exists_and_force(self, tmp_path: Path) -> None:
        """Test 2: _should_run() returns True when output artifact exists and force=True."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        artifact = tmp_path / "existing.json"
        artifact.write_text("{}", encoding="utf-8")
        assert orch._should_run(artifact, force=True) is True

    def test_returns_true_when_artifact_missing(self, tmp_path: Path) -> None:
        """Test 3: _should_run() returns True when output artifact does not exist."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        artifact = tmp_path / "nonexistent.json"
        assert orch._should_run(artifact, force=False) is True


class TestCheckManifestDeployed:
    """Tests for _check_manifest_deployed()."""

    def test_returns_true_when_at_least_one_surface_deployed(self, tmp_path: Path) -> None:
        """Test 4: _check_manifest_deployed() returns True when surfaces deployed."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        manifest_path = _make_manifest(tmp_path, deployed=True)
        assert orch._check_manifest_deployed(manifest_path) is True

    def test_returns_false_when_no_surfaces_deployed(self, tmp_path: Path) -> None:
        """Test 5: _check_manifest_deployed() returns False when no surfaces deployed."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        manifest_path = _make_manifest(tmp_path, deployed=False)
        assert orch._check_manifest_deployed(manifest_path) is False


class TestRunStage1:
    """Tests for _run_stage1()."""

    @patch("run_phase46_quality_gates.subprocess.run")
    def test_invokes_subprocess_and_returns_manifest_path(
        self, mock_run: MagicMock, tmp_path: Path,
    ) -> None:
        """Test 6: _run_stage1() invokes subprocess with correct arguments."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        conservative_root = tmp_path / "conservative"
        conservative_root.mkdir()
        args = _make_args(
            conservative_root=conservative_root,
            oof_path=Path("data/oof/oof.parquet"),
        )

        # Create manifest inside mock side_effect (simulates subprocess creating it)
        def _create_manifest(*a: object, **kw: object) -> MagicMock:
            _make_manifest(conservative_root, deployed=True)
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = _create_manifest

        orch = QualityGateOrchestrator()

        result = orch._run_stage1(args)

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "scripts/run_mawc_conservative_retrain.py" in cmd
        assert str(conservative_root) in cmd
        assert result == conservative_root / "manifest.json"

    def test_skips_execution_when_manifest_exists(self, tmp_path: Path) -> None:
        """Test 7: _run_stage1() skips when manifest exists and force=False."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        conservative_root = tmp_path / "conservative"
        conservative_root.mkdir()
        _make_manifest(conservative_root, deployed=True)
        args = _make_args(conservative_root=conservative_root)

        orch = QualityGateOrchestrator()
        with patch("run_phase46_quality_gates.subprocess.run") as mock_run:
            result = orch._run_stage1(args)
            mock_run.assert_not_called()
            assert result == conservative_root / "manifest.json"


class TestRunOofValidation:
    """Tests for _run_oof_validation()."""

    @patch("run_phase46_quality_gates.pd")
    def test_calls_validator_and_returns_pass(self, mock_pd: MagicMock, tmp_path: Path) -> None:
        """Test 8: _run_oof_validation() calls OOFHealthValidator.validate()."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        mock_df = MagicMock()
        mock_pd.read_parquet.return_value = mock_df

        args = _make_args(oof_path=tmp_path / "oof.parquet")
        orch = QualityGateOrchestrator()

        with patch("run_phase46_quality_gates.OOFHealthValidator") as mock_cls:
            mock_validator = MagicMock()
            mock_validator.validate.return_value = {
                "status": "PASS", "failures": [], "warnings": [],
            }
            mock_cls.return_value = mock_validator

            result = orch._run_oof_validation(args)

        assert result["status"] == "PASS"
        mock_validator.validate.assert_called_once()


class TestRunFeatureAudit:
    """Tests for _run_feature_audit()."""

    def test_calls_run_feature_audit_and_returns_pass(self) -> None:
        """Test 9: _run_feature_audit() calls run_feature_audit() and returns PASS/FAIL."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        with patch("run_phase46_quality_gates.run_feature_audit") as mock_audit:
            mock_audit.return_value = {
                "overall_status": "PASS",
                "critical_models": [],
                "advisory_models": [],
            }
            result = orch._run_feature_audit()

        assert result["status"] == "PASS"


class TestRunShadowComparison:
    """Tests for _run_shadow_comparison()."""

    @patch("run_phase46_quality_gates.subprocess.run")
    def test_invokes_subprocess_with_correct_args(
        self, mock_run: MagicMock, tmp_path: Path,
    ) -> None:
        """Test 10: _run_shadow_comparison() invokes subprocess with correct arguments."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        shadow_dir = tmp_path / "shadow_output"
        args = _make_args(
            shadow_output_dir=shadow_dir,
            source_model_dir=Path("data/models-backtest"),
            conservative_root=Path("data/models-backtest-mawc-conservative"),
            years="2024,2025",
        )

        # Create the result file after subprocess completes
        def _create_result(*a: object, **kw: object) -> None:
            shadow_dir.mkdir(parents=True, exist_ok=True)
            (shadow_dir / "shadow_comparison_result.json").write_text("{}", encoding="utf-8")

        mock_run.side_effect = _create_result

        orch = QualityGateOrchestrator()
        result = orch._run_shadow_comparison(args)

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "scripts/run_shadow_comparison.py" in cmd
        assert "--shadow-name" in cmd
        assert "mawc_conservative" in cmd
        assert result == shadow_dir / "shadow_comparison_result.json"


class TestRunShadowDiagnosis:
    """Tests for _run_shadow_diagnosis()."""

    def test_calls_shadow_diagnosis_and_save(self, tmp_path: Path) -> None:
        """Test 11: _run_shadow_diagnosis() calls ShadowDiagnosis.run() and save."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        shadow_dir = tmp_path / "shadow_output"
        shadow_dir.mkdir()
        args = _make_args(shadow_output_dir=shadow_dir)

        # Pre-create diagnosis dir and result to test skip logic
        # (we test the actual call when force=True or result doesn't exist)
        orch = QualityGateOrchestrator()

        with patch("run_phase46_quality_gates.ShadowDiagnosis") as mock_cls, \
             patch("run_phase46_quality_gates.save_diagnosis_results") as mock_save:
            mock_sd = MagicMock()
            mock_result = MagicMock()
            mock_sd.run.return_value = mock_result
            mock_cls.return_value = mock_sd

            result = orch._run_shadow_diagnosis(args)

        assert result["status"] == "PASS"
        mock_sd.run.assert_called_once()
        mock_save.assert_called_once()


class TestRunDeploymentGates:
    """Tests for _run_deployment_gates()."""

    def test_calls_run_deployment_gates_and_returns_pass(self, tmp_path: Path) -> None:
        """Test 12: _run_deployment_gates() calls run_deployment_gates() with correct paths."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        from backtest.deployment_gates import GateEvaluationResult, GatePolicy

        shadow_dir = tmp_path / "shadow_output"
        shadow_dir.mkdir()
        args = _make_args(shadow_output_dir=shadow_dir)

        gate_result = GateEvaluationResult(
            overall_status="PASS",
            policy=GatePolicy(),
            generated_at="2026-06-01T00:00:00Z",
            conditions=[],
            report_metrics={},
        )

        orch = QualityGateOrchestrator()
        with patch("run_phase46_quality_gates.run_deployment_gates") as mock_gates:
            mock_gates.return_value = gate_result
            result = orch._run_deployment_gates(args)

        assert result["status"] == "PASS"
        mock_gates.assert_called_once()


class TestComputeRoiTrend:
    """Tests for _compute_roi_trend()."""

    def test_returns_recovered_when_roi_ge_90(self) -> None:
        """Test 13: _compute_roi_trend() returns 'recovered' when shadow ROI >= 90%."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        shadow_result = {
            "overall": {"metrics": {"mawc_conservative": {"roi": 90.0}}},
        }
        assert orch._compute_roi_trend(shadow_result) == "recovered"

    def test_returns_weak_recovery_when_roi_between_87_8_and_90(self) -> None:
        """Test 14: _compute_roi_trend() returns 'weak_recovery' when 87.8% <= ROI < 90%."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        shadow_result = {
            "overall": {"metrics": {"mawc_conservative": {"roi": 89.9}}},
        }
        assert orch._compute_roi_trend(shadow_result) == "weak_recovery"

    def test_returns_not_recovered_when_roi_lt_87_8(self) -> None:
        """Test 15: _compute_roi_trend() returns 'not_recovered' when ROI < 87.8%."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        shadow_result = {
            "overall": {"metrics": {"mawc_conservative": {"roi": 85.0}}},
        }
        assert orch._compute_roi_trend(shadow_result) == "not_recovered"


class TestComputeDeploymentVerdict:
    """Tests for _compute_deployment_verdict()."""

    def test_returns_deployable_when_pass_and_recovered(self) -> None:
        """Test 16: _compute_deployment_verdict() returns 'deployable' when PASS + recovered."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        assert orch._compute_deployment_verdict("PASS", "recovered") == "deployable"

    def test_returns_manual_review_when_pass_and_not_recovered(self) -> None:
        """Test 17: manual_review when PASS + not_recovered."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        assert orch._compute_deployment_verdict("PASS", "not_recovered") == "manual_review"

    def test_returns_not_deployable_when_fail(self) -> None:
        """Test 18: _compute_deployment_verdict() returns 'not_deployable' when FAIL."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        assert orch._compute_deployment_verdict("FAIL", "recovered") == "not_deployable"


class TestStage2Orchestration:
    """Tests for Stage 2 full orchestration flow."""

    def test_stops_on_first_fail_and_records_partial(self, tmp_path: Path) -> None:
        """Test 19: Full Stage 2 orchestration stops on first FAIL."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        shadow_dir = _make_shadow_result(tmp_path, shadow_roi=92.0)
        args = _make_args(shadow_output_dir=shadow_dir)

        orch = QualityGateOrchestrator()

        with patch("run_phase46_quality_gates.run_feature_audit") as mock_audit:
            mock_audit.return_value = {
                "overall_status": "FAIL",
                "critical_models": [{"model_name": "X", "status": "FAIL"}],
                "advisory_models": [],
            }

            stage_results = orch._run_stage2(args)

        assert stage_results["stage2"]["feature_audit"]["status"] == "FAIL"
        # Steps after audit should not have been executed
        assert "oof_validation" not in stage_results["stage2"] or \
               stage_results["stage2"].get("oof_validation", {}).get("status") is None

    def test_completes_all_steps_and_aggregates(self, tmp_path: Path) -> None:
        """Test 20: Full Stage 2 completes all steps and aggregates."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        shadow_dir = _make_shadow_result(tmp_path, shadow_roi=92.0)
        output_dir = tmp_path / "output"
        args = _make_args(shadow_output_dir=shadow_dir, output_dir=output_dir)

        orch = QualityGateOrchestrator()

        with patch("run_phase46_quality_gates.run_feature_audit") as mock_audit, \
             patch("run_phase46_quality_gates.OOFHealthValidator") as mock_oof_cls, \
             patch("run_phase46_quality_gates.subprocess.run") as mock_sub, \
             patch("run_phase46_quality_gates.ShadowDiagnosis") as mock_sd_cls, \
             patch("run_phase46_quality_gates.save_diagnosis_results"), \
             patch("run_phase46_quality_gates.run_deployment_gates") as mock_gates, \
             patch("run_phase46_quality_gates.pd"):

            mock_audit.return_value = {
                "overall_status": "PASS",
                "critical_models": [],
                "advisory_models": [],
            }

            mock_validator = MagicMock()
            mock_validator.validate.return_value = {
                "status": "PASS", "failures": [], "warnings": [],
            }
            mock_oof_cls.return_value = mock_validator

            mock_sub.return_value = MagicMock(returncode=0)

            mock_sd = MagicMock()
            mock_sd.run.return_value = MagicMock()
            mock_sd_cls.return_value = mock_sd

            from backtest.deployment_gates import GateEvaluationResult, GatePolicy
            mock_gates.return_value = GateEvaluationResult(
                overall_status="PASS",
                policy=GatePolicy(),
                generated_at="2026-06-01T00:00:00Z",
                conditions=[],
                report_metrics={},
            )

            stage_results = orch._run_stage2(args)

        assert stage_results["stage2"]["feature_audit"]["status"] == "PASS"
        assert stage_results["stage2"]["oof_validation"]["status"] == "PASS"
        assert stage_results["stage2"]["deployment_gates"]["status"] == "PASS"

        # Aggregate results
        shadow_result_path = shadow_dir / "shadow_comparison_result.json"
        result = orch._aggregate_results(stage_results, shadow_result_path)
        assert result["quality_gate"] == "PASS"
        assert result["roi_trend"] == "recovered"
        assert result["deployment"] == "deployable"


# ---------------------------------------------------------------------------
# Task 2 Tests: CLI entry point + build_parser + end-to-end
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Tests for build_parser()."""

    def test_creates_parser_with_all_9_arguments(self) -> None:
        """Test 1 (Task 2): build_parser() creates parser with all 9 arguments."""
        from run_phase46_quality_gates import build_parser

        parser = build_parser()
        # Parse with defaults
        args = parser.parse_args([])
        arg_names = dir(args)
        assert "oof_path" in arg_names
        assert "source_model_dir" in arg_names
        assert "conservative_root" in arg_names
        assert "shadow_output_dir" in arg_names
        assert "output_dir" in arg_names
        assert "years" in arg_names
        assert "stage" in arg_names
        assert "force" in arg_names
        assert "report" in arg_names

    def test_defaults_match_specification(self) -> None:
        """Test 2 (Task 2): build_parser() defaults match specification."""
        from run_phase46_quality_gates import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.oof_path == Path("data/oof/oof_predictions.parquet")
        assert args.source_model_dir == Path("data/models-backtest")
        assert args.conservative_root == Path("data/models-backtest-mawc-conservative")
        assert args.shadow_output_dir == Path("data/backtest/shadow_mawc_conservative")
        assert args.output_dir == Path("data/backtest/phase46_quality_gates")
        assert args.years == "2024,2025"
        assert args.stage is None
        assert args.force is False
        assert args.report is False

    def test_help_exits_zero_with_description(self) -> None:
        """Test 3 (Task 2): CLI --help exits with code 0 and contains description."""
        from run_phase46_quality_gates import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0


class TestMainStageSelection:
    """Tests for main() stage selection logic."""

    @patch("run_phase46_quality_gates.QualityGateOrchestrator")
    def test_stage1_only(self, mock_cls: MagicMock, tmp_path: Path) -> None:
        """Test 4 (Task 2): main() with --stage 1 runs only Stage 1."""
        from run_phase46_quality_gates import main

        mock_orch = MagicMock()
        mock_orch._run_stage1.return_value = tmp_path / "manifest.json"
        mock_cls.return_value = mock_orch

        args = _make_args(stage=1)
        main(args)

        mock_orch._run_stage1.assert_called_once_with(args)
        mock_orch._run_stage2.assert_not_called()

    @patch("run_phase46_quality_gates.QualityGateOrchestrator")
    def test_stage2_only(self, mock_cls: MagicMock, tmp_path: Path) -> None:
        """Test 5 (Task 2): main() with --stage 2 skips Stage 1."""
        from run_phase46_quality_gates import main

        mock_orch = MagicMock()
        mock_orch._run_stage2.return_value = {
            "stage2": {
                "feature_audit": {"status": "PASS"},
                "oof_validation": {"status": "PASS"},
                "shadow_comparison": {"status": "PASS"},
                "shadow_diagnosis": {"status": "PASS"},
                "deployment_gates": {"status": "PASS"},
            },
        }
        mock_orch._aggregate_results.return_value = {
            "quality_gate": "PASS",
            "roi_trend": "recovered",
            "deployment": "deployable",
            "baseline_roi": 87.8,
            "shadow_roi": 92.0,
            "artifacts": {},
        }
        mock_cls.return_value = mock_orch

        output_dir = tmp_path / "output"
        args = _make_args(stage=2, output_dir=output_dir)
        main(args)

        mock_orch._run_stage1.assert_not_called()
        mock_orch._run_stage2.assert_called_once_with(args)

    @patch("run_phase46_quality_gates.QualityGateOrchestrator")
    def test_auto_detect_manifest_exists(self, mock_cls: MagicMock, tmp_path: Path) -> None:
        """Test 6 (Task 2): main() auto-detects Stage 1 completion when manifest exists."""
        from run_phase46_quality_gates import main

        conservative_root = tmp_path / "conservative"
        conservative_root.mkdir()
        _make_manifest(conservative_root, deployed=True)

        mock_orch = MagicMock()
        mock_orch._run_stage2.return_value = {
            "stage2": {
                "feature_audit": {"status": "PASS"},
                "oof_validation": {"status": "PASS"},
                "shadow_comparison": {"status": "PASS"},
                "shadow_diagnosis": {"status": "PASS"},
                "deployment_gates": {"status": "PASS"},
            },
        }
        mock_orch._aggregate_results.return_value = {
            "quality_gate": "PASS",
            "roi_trend": "recovered",
            "deployment": "deployable",
            "baseline_roi": 87.8,
            "shadow_roi": 92.0,
            "artifacts": {},
        }
        mock_cls.return_value = mock_orch

        output_dir = tmp_path / "output"
        args = _make_args(stage=None, conservative_root=conservative_root, output_dir=output_dir)
        main(args)

        # Should NOT call _run_stage1 since manifest exists
        mock_orch._run_stage1.assert_not_called()
        mock_orch._run_stage2.assert_called_once_with(args)


class TestMainExitCodes:
    """Tests for main() exit codes."""

    @patch("run_phase46_quality_gates.QualityGateOrchestrator")
    def test_exits_zero_on_pass(self, mock_cls: MagicMock, tmp_path: Path) -> None:
        """Test 7 (Task 2): main() exits 0 when quality_gate is PASS."""
        from run_phase46_quality_gates import main

        mock_orch = MagicMock()
        mock_orch._run_stage2.return_value = {
            "stage2": {
                "feature_audit": {"status": "PASS"},
                "oof_validation": {"status": "PASS"},
                "shadow_comparison": {"status": "PASS"},
                "shadow_diagnosis": {"status": "PASS"},
                "deployment_gates": {"status": "PASS"},
            },
        }
        mock_orch._aggregate_results.return_value = {
            "quality_gate": "PASS",
            "roi_trend": "recovered",
            "deployment": "deployable",
            "baseline_roi": 87.8,
            "shadow_roi": 92.0,
            "artifacts": {},
        }
        mock_cls.return_value = mock_orch

        output_dir = tmp_path / "output"
        args = _make_args(stage=2, output_dir=output_dir)
        main(args)  # Should not raise SystemExit

    @patch("run_phase46_quality_gates.QualityGateOrchestrator")
    def test_exits_one_on_fail(self, mock_cls: MagicMock, tmp_path: Path) -> None:
        """Test 8 (Task 2): main() exits 1 when quality_gate is FAIL."""
        from run_phase46_quality_gates import main

        mock_orch = MagicMock()
        mock_orch._run_stage2.return_value = {
            "stage2": {
                "feature_audit": {"status": "FAIL"},
            },
        }
        mock_orch._aggregate_results.return_value = {
            "quality_gate": "FAIL",
            "roi_trend": "not_recovered",
            "deployment": "not_deployable",
            "baseline_roi": 87.8,
            "shadow_roi": 85.0,
            "artifacts": {},
        }
        mock_cls.return_value = mock_orch

        output_dir = tmp_path / "output"
        args = _make_args(stage=2, output_dir=output_dir)
        with pytest.raises(SystemExit) as exc_info:
            main(args)
        assert exc_info.value.code == 1


class TestMarkdownSummary:
    """Tests for Markdown summary content."""

    def test_summary_contains_all_sections(self, tmp_path: Path) -> None:
        """Test 9 (Task 2): Markdown summary contains all required sections."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        output_dir = tmp_path / "output"

        result = {
            "phase": "46-quality-gate-verification",
            "generated_at": "2026-06-01T00:00:00Z",
            "quality_gate": "PASS",
            "roi_trend": "recovered",
            "deployment": "deployable",
            "baseline_roi": 87.8,
            "shadow_roi": 92.0,
            "stage1": {
                "status": "COMPLETE",
                "manifest_path": "/path/manifest.json",
                "deployed_surfaces": ["turf", "dirt"],
            },
            "stage2": {
                "feature_audit": {"status": "PASS"},
                "oof_validation": {"status": "PASS"},
                "shadow_comparison": {"status": "PASS", "path": "/path/shadow"},
                "shadow_diagnosis": {"status": "PASS", "path": "/path/diag"},
                "deployment_gates": {"status": "PASS", "conditions": []},
            },
            "artifacts": {"result_json": "/path/result.json"},
        }

        orch._write_results(result, output_dir)
        md_path = output_dir / "phase46_quality_gate_summary.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "Quality Gate" in content
        assert "ROI Trend" in content or "ROI Comparison" in content
        assert "Deployment" in content
        assert "Stage 1" in content
        assert "Stage 2" in content or "Quality Gate Steps" in content

    def test_summary_records_fail_step_clearly(self, tmp_path: Path) -> None:
        """Test 10 (Task 2): Markdown summary records FAIL step with reason."""
        from run_phase46_quality_gates import QualityGateOrchestrator

        orch = QualityGateOrchestrator()
        output_dir = tmp_path / "output"

        result = {
            "phase": "46-quality-gate-verification",
            "generated_at": "2026-06-01T00:00:00Z",
            "quality_gate": "FAIL",
            "roi_trend": "not_recovered",
            "deployment": "not_deployable",
            "baseline_roi": 87.8,
            "shadow_roi": 85.0,
            "stage1": {
                "status": "COMPLETE",
                "manifest_path": "/path/manifest.json",
                "deployed_surfaces": [],
            },
            "stage2": {
                "feature_audit": {
                    "status": "FAIL",
                    "failures": ["MarketModel has forbidden features"],
                },
            },
            "artifacts": {"result_json": "/path/result.json"},
        }

        orch._write_results(result, output_dir)
        md_path = output_dir / "phase46_quality_gate_summary.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "FAIL" in content
        assert "feature_audit" in content
