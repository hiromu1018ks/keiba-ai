"""DataCutoffManifest, PFPVerifier, SessionManifest のユニットテスト."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from features.data_cutoff_manifest import DataCutoffManifest
from features.pipeline_consistency import PFPVerifier
from features.session_manifest import SessionManifest, get_code_version, write_session_manifest


# ── DataCutoffManifest ──────────────────────────────────────────


class TestDataCutoffManifest:
    """DataCutoffManifest の検証ロジックをテスト."""

    def test_verify_returns_empty_when_all_dates_before_prediction(self) -> None:
        manifest = DataCutoffManifest(
            model_train_end="2024-12-31",
            stats_fit_end="2024-12-31",
            odds_band_calibration_end="2024-12-31",
            strategy_optimization_end="2024-12-31",
            prediction_date="2025-06-01",
        )
        actual = {
            "model_train_end": "2024-12-31",
            "stats_fit_end": "2024-12-31",
            "odds_band_calibration_end": "2024-12-31",
            "strategy_optimization_end": "2024-12-31",
        }
        assert manifest.verify(actual) == []

    def test_verify_returns_violations_when_model_train_exceeds_prediction(self) -> None:
        manifest = DataCutoffManifest(
            model_train_end="2024-12-31",
            stats_fit_end="2024-12-31",
            odds_band_calibration_end="2024-12-31",
            strategy_optimization_end="2024-12-31",
            prediction_date="2025-06-01",
        )
        actual = {
            "model_train_end": "2025-07-01",
            "stats_fit_end": "2024-12-31",
            "odds_band_calibration_end": "2024-12-31",
            "strategy_optimization_end": "2024-12-31",
        }
        violations = manifest.verify(actual)
        assert len(violations) == 1
        assert "model_train_end" in violations[0]

    def test_verify_returns_multiple_violations(self) -> None:
        manifest = DataCutoffManifest(
            model_train_end="2024-12-31",
            stats_fit_end="2024-12-31",
            odds_band_calibration_end="2024-12-31",
            strategy_optimization_end="2024-12-31",
            prediction_date="2025-06-01",
        )
        actual = {
            "model_train_end": "2025-07-01",
            "stats_fit_end": "2025-08-01",
            "odds_band_calibration_end": "2024-12-31",
            "strategy_optimization_end": "2025-09-01",
        }
        violations = manifest.verify(actual)
        assert len(violations) == 3

    def test_verify_returns_violation_when_date_not_provided(self) -> None:
        manifest = DataCutoffManifest(
            model_train_end="2024-12-31",
            stats_fit_end="2024-12-31",
            odds_band_calibration_end="2024-12-31",
            strategy_optimization_end="2024-12-31",
            prediction_date="2025-06-01",
        )
        actual: dict[str, str] = {}
        violations = manifest.verify(actual)
        assert len(violations) == 4

    def test_verify_strict_raises_on_violation(self) -> None:
        manifest = DataCutoffManifest(
            model_train_end="2024-12-31",
            stats_fit_end="2024-12-31",
            odds_band_calibration_end="2024-12-31",
            strategy_optimization_end="2024-12-31",
            prediction_date="2025-06-01",
        )
        actual = {
            "model_train_end": "2025-07-01",
            "stats_fit_end": "2024-12-31",
            "odds_band_calibration_end": "2024-12-31",
            "strategy_optimization_end": "2024-12-31",
        }
        with pytest.raises(ValueError, match="Data cutoff violations"):
            manifest.verify_strict(actual)

    def test_verify_strict_passes_with_no_violations(self) -> None:
        manifest = DataCutoffManifest(
            model_train_end="2024-12-31",
            stats_fit_end="2024-12-31",
            odds_band_calibration_end="2024-12-31",
            strategy_optimization_end="2024-12-31",
            prediction_date="2025-06-01",
        )
        actual = {
            "model_train_end": "2024-12-31",
            "stats_fit_end": "2024-12-31",
            "odds_band_calibration_end": "2024-12-31",
            "strategy_optimization_end": "2024-12-31",
        }
        # Should not raise
        manifest.verify_strict(actual)

    def test_from_config_extracts_from_models(self) -> None:
        models = MagicMock()
        models.train_period = ("2020-01-01", "2023-12-31")
        manifest = DataCutoffManifest.from_config(
            prediction_date="2025-06-01",
            models=models,
        )
        assert manifest.model_train_end == "2023-12-31"
        assert manifest.prediction_date == "2025-06-01"

    def test_from_config_with_strategy_manifest(self) -> None:
        models = MagicMock()
        models.train_period = ("2020-01-01", "2023-12-31")
        tmp_path = Path(tempfile.mkdtemp()) / "strategy_manifest.json"
        try:
            tmp_path.write_text(
                json.dumps(
                    {"params": {"optimization_end": "2024-06-30"}, "sha256": "abc"},
                ),
                encoding="utf-8",
            )
            manifest = DataCutoffManifest.from_config(
                prediction_date="2025-06-01",
                models=models,
                strategy_manifest_path=tmp_path,
            )
            assert manifest.strategy_optimization_end == "2024-06-30"
        finally:
            tmp_path.unlink(missing_ok=True)
            try:
                tmp_path.parent.rmdir()
            except OSError:
                pass


# ── PFPVerifier ─────────────────────────────────────────────────


def _make_mock_manifest(hash_val: str = "abcdef1234567890") -> MagicMock:
    """モック FeatureManifest を生成."""
    manifest = MagicMock()
    manifest.compute_hash.return_value = hash_val
    return manifest


def _make_mock_state(hash_val: str = "state_hash_1234") -> MagicMock:
    """モック FeatureState を生成."""
    state = MagicMock()
    state.compute_hash.return_value = hash_val
    return state


def _make_mock_models() -> MagicMock:
    """モック TrainedModelsV5 を生成."""
    return MagicMock()


class TestPFPVerifier:
    """PFPVerifier の凍結/検証ロジックをテスト."""

    def test_freeze_and_verify_passes_when_nothing_changed(self) -> None:
        models = _make_mock_models()
        manifest = _make_mock_manifest()
        state = _make_mock_state()
        verifier = PFPVerifier(models, manifest, state, "win", "flat")
        verifier.freeze()
        result = verifier.verify()
        assert result["passed"] is True
        assert result["checks"]["model_hp"] is True
        assert result["checks"]["feature_manifest"] is True

    def test_verify_fails_when_manifest_hash_changes(self) -> None:
        models = _make_mock_models()
        manifest = _make_mock_manifest("hash_original")
        state = _make_mock_state()
        verifier = PFPVerifier(models, manifest, state, "win", "flat")
        verifier.freeze()
        # Simulate manifest hash change
        manifest.compute_hash.return_value = "hash_modified"
        result = verifier.verify()
        assert result["passed"] is False
        assert result["checks"]["feature_manifest"] is False

    def test_verify_fails_when_betting_target_changes(self) -> None:
        models = _make_mock_models()
        manifest = _make_mock_manifest()
        state = _make_mock_state()
        verifier = PFPVerifier(models, manifest, state, "win", "flat")
        verifier.freeze()
        # Mutate internal state to simulate change
        verifier._betting_target = "place"
        result = verifier.verify()
        assert result["passed"] is False
        assert result["checks"]["betting_target"] is False

    def test_verify_excludes_runtime_state(self) -> None:
        """RegimeDetector / DDController が checks に含まれないことを確認 (D-08)."""
        models = _make_mock_models()
        manifest = _make_mock_manifest()
        state = _make_mock_state()
        verifier = PFPVerifier(models, manifest, state, "win", "flat")
        verifier.freeze()
        result = verifier.verify()
        assert "regime_detector" not in result["checks"]
        assert "dd_controller" not in result["checks"]
        assert "drawdown_controller" not in result["checks"]

    def test_get_frozen_state_returns_expected_keys(self) -> None:
        models = _make_mock_models()
        manifest = _make_mock_manifest("manifest_hash_val")
        state = _make_mock_state("state_hash_val")
        verifier = PFPVerifier(models, manifest, state, "win", "flat")
        verifier.freeze()
        frozen = verifier.get_frozen_state()
        assert "manifest_hash" in frozen
        assert "state_hash" in frozen
        assert "betting_target" in frozen
        assert "betting_mode" in frozen
        assert frozen["manifest_hash"] == "manifest_hash_val"
        assert frozen["betting_target"] == "win"

    def test_verify_before_freeze_returns_not_passed(self) -> None:
        models = _make_mock_models()
        manifest = _make_mock_manifest()
        state = _make_mock_state()
        verifier = PFPVerifier(models, manifest, state, "win", "flat")
        # freeze() なしで verify() を呼び出し
        result = verifier.verify()
        assert result["passed"] is False
        assert "freeze" in result["message"].lower() or "not" in result["message"].lower()

    def test_verify_fails_when_state_hash_changes(self) -> None:
        models = _make_mock_models()
        manifest = _make_mock_manifest()
        state = _make_mock_state("original_state_hash")
        verifier = PFPVerifier(models, manifest, state, "win", "flat")
        verifier.freeze()
        state.compute_hash.return_value = "modified_state_hash"
        result = verifier.verify()
        assert result["passed"] is False
        assert result["checks"]["feature_state"] is False


# ── SessionManifest ─────────────────────────────────────────────


class TestSessionManifest:
    """SessionManifest の記録・シリアライズをテスト."""

    def test_to_dict_contains_all_required_keys(self) -> None:
        manifest = SessionManifest(session_id="abc123", prediction_date="2025-06-01")
        d = manifest.to_dict()
        required_keys = {
            "session_id", "prediction_date", "code_version",
            "model_run_id", "manifest_hash", "pfp_result",
            "status", "exit_code", "training_start", "training_end",
        }
        assert required_keys.issubset(set(d.keys()))

    def test_is_dirty_returns_true_when_git_dirty(self) -> None:
        manifest = SessionManifest(
            session_id="abc123",
            prediction_date="2025-06-01",
            code_version={"commit_sha": "deadbeef", "git_dirty": True},
        )
        assert manifest.is_dirty is True

    def test_is_dirty_returns_false_when_clean(self) -> None:
        manifest = SessionManifest(
            session_id="abc123",
            prediction_date="2025-06-01",
            code_version={"commit_sha": "deadbeef", "git_dirty": False},
        )
        assert manifest.is_dirty is False

    def test_set_model_identity_stores_all_fields(self) -> None:
        manifest = SessionManifest(session_id="abc123", prediction_date="2025-06-01")
        manifest.set_model_identity(
            run_id="mlflow-run-42",
            training_start="2020-01-01",
            training_end="2023-12-31",
            manifest_hash="hash123",
        )
        assert manifest.model_run_id == "mlflow-run-42"
        assert manifest.training_start == "2020-01-01"
        assert manifest.training_end == "2023-12-31"
        assert manifest.manifest_hash == "hash123"

    def test_set_pfp_result_stores_result_dict(self) -> None:
        manifest = SessionManifest(session_id="abc123", prediction_date="2025-06-01")
        pfp_data = {"passed": True, "checks": {"model_hp": True}}
        manifest.set_pfp_result(pfp_data)
        assert manifest.pfp_result == pfp_data


# ── get_code_version ────────────────────────────────────────────


class TestCodeVersion:
    """get_code_version() の git 状態検出をテスト."""

    def test_returns_commit_sha_and_git_dirty_false_for_clean_repo(self) -> None:
        """git status --porcelain が空なら git_dirty=False."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="abc123def456\n", returncode=0),  # rev-parse
                MagicMock(stdout="", returncode=0),  # status --porcelain
            ]
            result = get_code_version()
        assert result["commit_sha"] == "abc123def456"
        assert result["git_dirty"] is False
        assert result["dirty_diff_hash"] is None

    def test_returns_git_dirty_true_for_dirty_repo(self) -> None:
        """git status --porcelain が空でなければ git_dirty=True."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="abc123def456\n", returncode=0),  # rev-parse
                MagicMock(stdout=" M src/foo.py\n?? src/bar.py\n", returncode=0),
                MagicMock(stdout="diff content", returncode=0),  # git diff
            ]
            result = get_code_version()
        assert result["git_dirty"] is True
        assert result["dirty_diff_hash"] is not None
        assert len(result["untracked_files"]) == 1

    def test_raises_runtime_error_when_git_unavailable(self) -> None:
        """git コマンドが見つからない場合は RuntimeError."""
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            with pytest.raises(RuntimeError, match="git rev-parse"):
                get_code_version()


# ── write_session_manifest ──────────────────────────────────────


class TestWriteSessionManifest:
    """write_session_manifest() のファイル書き込みをテスト."""

    def test_file_created_with_valid_json(self) -> None:
        manifest = SessionManifest(session_id="test123", prediction_date="2025-06-01")
        manifest.set_model_identity("run-1", "2020-01-01", "2023-12-31", "hash")
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            out_path = tmp_dir / "session_manifest.json"
            write_session_manifest(manifest, out_path)
            assert out_path.exists()
            data = json.loads(out_path.read_text(encoding="utf-8"))
            assert data["session_id"] == "test123"
            assert data["model_run_id"] == "run-1"
        finally:
            out_path.unlink(missing_ok=True)
            try:
                tmp_dir.rmdir()
            except OSError:
                pass

    def test_atomic_write_creates_readable_file(self) -> None:
        """アトミック書き込み後、ファイルが読み取り可能であることを確認."""
        manifest = SessionManifest(session_id="atomic_test", prediction_date="2025-06-01")
        manifest.set_status("completed", exit_code=0)
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            out_path = tmp_dir / "session_manifest.json"
            write_session_manifest(manifest, out_path)
            # 読み取り可能か確認
            content = out_path.read_text(encoding="utf-8")
            parsed = json.loads(content)
            assert parsed["status"] == "completed"
            assert parsed["exit_code"] == 0
        finally:
            out_path.unlink(missing_ok=True)
            try:
                tmp_dir.rmdir()
            except OSError:
                pass
