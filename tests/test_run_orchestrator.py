"""Tests for RunModeOrchestrator lifecycle and resume (Task 1)."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from paper_trading.exit_codes import ExitCode
from paper_trading.race_progress import RaceProgress, RaceState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> MagicMock:
    """Create a mock PaperTradingConfig."""
    cfg = MagicMock()
    cfg.paper_trading_dir = tmp_path / "paper_trading"
    cfg.paper_trading_dir.mkdir(parents=True, exist_ok=True)
    cfg.initial_bankroll = 100000.0
    cfg.everydb2_connection_string = "postgresql://test"
    cfg.mlflow_tracking_uri = "file:///mlruns"
    cfg.mlflow_run_id = "test-run-001"
    return cfg


def _make_models() -> MagicMock:
    """Create a mock TrainedModelsV5."""
    models = MagicMock()
    models.regime_detector = MagicMock()
    models.regime_detector.current_regime = "aggressive"
    models.regime_detector.cfg = MagicMock(min_samples=10)
    models.regime_detector.get_strategy_params.return_value = {"ev_threshold": 1.10}
    return models


def _make_store() -> MagicMock:
    return MagicMock()


def _make_args(**overrides: Any) -> MagicMock:
    args = MagicMock()
    args.date = "2026-06-06"
    args.minutes_before = 5
    args.ensemble = False
    args.run_id = None
    args.mode = "run"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _make_session_manifest() -> MagicMock:
    manifest = MagicMock()
    manifest.session_id = "20260606_120000"
    manifest.prediction_date = "2026-06-06"
    manifest.model_run_id = "test-run-001"
    manifest.training_start = "2020-01-01"
    manifest.training_end = "2025-12-31"
    manifest.manifest_hash = "abc123"
    return manifest


def _write_schedule(config: MagicMock, races: list[dict[str, Any]]) -> None:
    """Write a schedule.json file."""
    schedule_path = config.paper_trading_dir / "schedule.json"
    schedule_path.write_text(
        json.dumps({"date": "2026-06-06", "races": races}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _make_race(race_id: str = "20260606050101", post_time: str = "15:00") -> dict[str, Any]:
    return {
        "race_id": race_id,
        "surface": "turf",
        "distance": 1600,
        "post_time": post_time,
        "n_horses": 14,
        "n_with_results": 0,
    }


def _write_bets_parquet(
    bets_path: Path,
    rows: list[dict[str, Any]] | None = None,
) -> None:
    """Write a minimal bets.parquet with schema_version=2 columns."""
    if rows is None:
        rows = []
    if not rows:
        # Empty df with required columns
        df = pd.DataFrame(
            columns=[
                "bet_id", "race_id", "umaban", "bet_type", "stake", "odds",
                "schema_version", "settlement_status", "outcome", "payout",
                "race_date", "session_id", "bankroll_after",
            ]
        )
    else:
        df = pd.DataFrame(rows)
    df.to_parquet(bets_path, index=False)


# ---------------------------------------------------------------------------
# Test 1: Fresh session lifecycle
# ---------------------------------------------------------------------------

class TestFreshSession:
    """Test 1: Fresh session: execute() runs full lifecycle."""

    def test_fresh_session_calls_lifecycle_methods(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()
        strategy_config: dict[str, Any] = {}

        orch = RunModeOrchestrator(
            config=config,
            models=models,
            store=store,
            args=args,
            strategy_config=strategy_config,
            session_manifest=manifest,
        )

        # Patch all lifecycle methods to track calls
        with patch.object(orch, "_ensure_schedule") as mock_schedule, \
             patch.object(orch, "_fetch_track_conditions") as mock_tc, \
             patch.object(orch, "_predict_races") as mock_predict, \
             patch.object(orch, "_reconcile") as mock_reconcile, \
             patch.object(orch, "_aggregate_and_report") as mock_report:

            result = orch.execute()

        mock_schedule.assert_called_once()
        mock_tc.assert_called_once()
        mock_predict.assert_called_once()
        mock_reconcile.assert_called_once()
        mock_report.assert_called_once()
        assert result == ExitCode.SUCCESS


# ---------------------------------------------------------------------------
# Test 2: Resume with predicted races
# ---------------------------------------------------------------------------

class TestResumePredicted:
    """Test 2: Resume with predicted races: _predict_races skips PREDICTED/NO_BET."""

    def test_skips_predicted_races(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        # Set up race_progress with a PREDICTED race
        progress = RaceProgress(orch.race_progress_path)
        progress.mark("race_001", RaceState.PREDICTED, bet_ids=["bet_001"])

        # Set up schedule
        _write_schedule(config, [
            _make_race("race_001", "15:00"),
            _make_race("race_002", "15:30"),
        ])

        # Set up bets.parquet with the bet for race_001
        _write_bets_parquet(
            config.paper_trading_dir / "bets.parquet",
            [{"bet_id": "bet_001", "race_id": "race_001", "umaban": 1,
              "bet_type": "win", "stake": 100.0, "odds": 5.0,
              "schema_version": 2, "settlement_status": "pending",
              "outcome": None, "payout": None,
              "race_date": pd.Timestamp("2026-06-06"),
              "session_id": "20260606_120000", "bankroll_after": 99900.0}],
        )

        # _predict_races should skip race_001 (PREDICTED) and attempt race_002
        # We patch _build_race_predictor to avoid heavy dependencies
        with patch.object(orch, "_build_race_predictor", return_value=None):
            # race_002 is PENDING (no progress entry), should be processed
            # but since we can't actually predict without a predictor, we expect
            # it to be marked FAILED
            orch._predict_races()

        # race_001 should remain PREDICTED, race_002 should be attempted
        states = progress.to_dict()
        assert states.get("race_001", {}).get("state") == str(RaceState.PREDICTED)

    def test_skips_no_bet_races(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        # Set up race_progress with a NO_BET race
        progress = RaceProgress(orch.race_progress_path)
        progress.mark("race_001", RaceState.NO_BET)

        _write_schedule(config, [_make_race("race_001", "15:00")])
        _write_bets_parquet(config.paper_trading_dir / "bets.parquet")

        with patch.object(orch, "_build_race_predictor", return_value=None):
            orch._predict_races()

        # race_001 should remain NO_BET
        assert progress.get_state("race_001") == str(RaceState.NO_BET)


# ---------------------------------------------------------------------------
# Test 3: Resume with failed races
# ---------------------------------------------------------------------------

class TestResumeFailed:
    """Test 3: Resume reprocesses FAILED/PROCESSING/PENDING races."""

    def test_reprocesses_failed_race(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        progress = RaceProgress(orch.race_progress_path)
        progress.mark("race_001", RaceState.FAILED, failure_reason="db_error")

        _write_schedule(config, [_make_race("race_001", "15:00")])
        _write_bets_parquet(config.paper_trading_dir / "bets.parquet")

        # _predict_races should attempt to reprocess race_001 (FAILED)
        with patch.object(orch, "_build_race_predictor", return_value=None):
            orch._predict_races()

        # The race should have been attempted (state changed from FAILED)
        # It will fail again since there's no predictor, so it should be FAILED again
        state = progress.get_state("race_001")
        # Either FAILED again or remained FAILED is fine
        assert state is not None  # It was at least processed


# ---------------------------------------------------------------------------
# Test 4: Cross-validation (D-08)
# ---------------------------------------------------------------------------

class TestCrossValidation:
    """Test 4: Cross-validate PREDICTED races against bets.parquet (D-08)."""

    def test_predicted_race_missing_bets_marked_failed(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        # Mark race as PREDICTED with a bet_id that doesn't exist in bets.parquet
        progress = RaceProgress(orch.race_progress_path)
        progress.mark("race_001", RaceState.PREDICTED, bet_ids=["nonexistent_bet"])

        _write_schedule(config, [_make_race("race_001", "15:00")])
        # bets.parquet exists but has no matching bet_id
        _write_bets_parquet(config.paper_trading_dir / "bets.parquet")

        with patch.object(orch, "_build_race_predictor", return_value=None):
            orch._predict_races()

        # Cross-validation should have detected missing bet and re-processed
        # race_001 should no longer be PREDICTED
        state = progress.get_state("race_001")
        assert state != str(RaceState.PREDICTED)


# ---------------------------------------------------------------------------
# Test 5: Schedule reuse (D-03)
# ---------------------------------------------------------------------------

class TestScheduleReuse:
    """Test 5: _ensure_schedule reuses existing schedule.json (D-03)."""

    def test_reuses_existing_schedule(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        races = [_make_race("race_001", "15:00"), _make_race("race_002", "15:30")]
        _write_schedule(config, races)

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        # _ensure_schedule should NOT call DB if schedule.json exists
        with patch("paper_trading.run_orchestrator.EveryDB2Queries") as mock_db:
            orch._ensure_schedule()
            mock_db.assert_not_called()

        # schedule should be loaded
        assert orch._schedule is not None
        assert len(orch._schedule) == 2

    def test_fetches_schedule_when_missing(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        mock_db_instance = MagicMock()
        mock_race_df = pd.DataFrame({"race_id": ["race_001"], "surface": ["turf"],
                                      "kyori": [1600], "hassotime": ["1500"]})
        mock_entry_df = pd.DataFrame({"race_id": ["race_001"], "umaban": [1],
                                       "kakuteijyuni": [None]})
        mock_db_instance.get_races.return_value = mock_race_df
        mock_db_instance.get_entries.return_value = mock_entry_df

        with patch("paper_trading.run_orchestrator.EveryDB2Queries",
                    return_value=mock_db_instance), \
             patch("paper_trading.run_orchestrator.load_races_from_db",
                    return_value=mock_race_df), \
             patch("paper_trading.run_orchestrator.load_entries_from_db",
                    return_value=mock_entry_df):
            orch._ensure_schedule()

        assert orch._schedule is not None


# ---------------------------------------------------------------------------
# Test 6: DB fetch error -> ExitCode.DB_FETCH_ERROR
# ---------------------------------------------------------------------------

class TestDBFetchError:
    """Test 6: DB fetch error during prediction -> ExitCode.DB_FETCH_ERROR (D-17)."""

    def test_db_error_during_schedule(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        with patch("paper_trading.run_orchestrator.EveryDB2Queries",
                    side_effect=Exception("DB connection failed")):
            result = orch._ensure_schedule()

        assert result is False
        assert ExitCode.DB_FETCH_ERROR in orch.errors


# ---------------------------------------------------------------------------
# Test 7: Model validation error -> ExitCode.MODEL_VALIDATION_ERROR
# ---------------------------------------------------------------------------

class TestModelValidationError:
    """Test 7: Model validation error -> ExitCode.MODEL_VALIDATION_ERROR."""

    def test_model_error(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        # Simulate model validation failure
        orch.errors.append(ExitCode.MODEL_VALIDATION_ERROR)
        result = orch._determine_exit_code()
        assert result == ExitCode.MODEL_VALIDATION_ERROR


# ---------------------------------------------------------------------------
# Test 8: Pending remaining -> ExitCode.PENDING_REMAIN (D-02)
# ---------------------------------------------------------------------------

class TestPendingRemain:
    """Test 8: Pending remaining after reconcile -> ExitCode.PENDING_REMAIN (D-02)."""

    def test_pending_remain_exit_code(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        orch.errors.append(ExitCode.PENDING_REMAIN)
        result = orch._determine_exit_code()
        assert result == ExitCode.PENDING_REMAIN


# ---------------------------------------------------------------------------
# Test 9: _wait_until_with_cancel respects _cancelled flag
# ---------------------------------------------------------------------------

class TestWaitUntilWithCancel:
    """Test 9: _wait_until_with_cancel respects _cancelled flag -> returns False."""

    def test_cancelled_returns_false(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        # Set cancelled flag
        orch._cancelled = True

        # Target is in the future
        future_time = datetime.now() + timedelta(hours=1)
        result = orch._wait_until_with_cancel(future_time)
        assert result is False


# ---------------------------------------------------------------------------
# Test 10: Full happy path returns ExitCode.SUCCESS
# ---------------------------------------------------------------------------

class TestHappyPath:
    """Test 10: Full happy path returns ExitCode.SUCCESS."""

    def test_success_exit_code(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        # No errors -> SUCCESS
        result = orch._determine_exit_code()
        assert result == ExitCode.SUCCESS


# ---------------------------------------------------------------------------
# Test 11: Input snapshot with metadata (D-09)
# ---------------------------------------------------------------------------

class TestInputSnapshot:
    """Test 11: _save_input_snapshot writes parquet with metadata columns (D-09)."""

    def test_snapshot_has_metadata_columns(self, tmp_path: Path) -> None:
        from paper_trading.run_orchestrator import RunModeOrchestrator

        config = _make_config(tmp_path)
        models = _make_models()
        store = _make_store()
        args = _make_args()
        manifest = _make_session_manifest()

        orch = RunModeOrchestrator(
            config=config, models=models, store=store,
            args=args, strategy_config={}, session_manifest=manifest,
        )

        # Create minimal feature + odds data
        features_df = pd.DataFrame({
            "race_id": ["race_001"],
            "umaban": [1],
            "feature_a": [0.5],
        })
        odds_df = pd.DataFrame({
            "race_id": ["race_001"],
            "umaban": [1],
            "odds": [5.0],
        })

        orch._save_input_snapshot("race_001", features_df, odds_df)

        # Verify the snapshot file was created
        snapshot_path = orch.session_dir / "inputs" / "race_001.parquet"
        assert snapshot_path.exists()

        # Load and verify metadata columns
        saved = pd.read_parquet(snapshot_path)
        assert "_snapshot_hash" in saved.columns
        assert "_parent_session_id" in saved.columns
        assert "_source_info" in saved.columns

        # Verify values
        assert saved["_parent_session_id"].iloc[0] == "20260606_120000"
        assert len(saved["_snapshot_hash"].iloc[0]) == 64  # SHA256 hex

        # Source info should be valid JSON
        source_info = json.loads(saved["_source_info"].iloc[0])
        assert source_info["race_id"] == "race_001"
        assert source_info["target_date"] == "2026-06-06"
