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
