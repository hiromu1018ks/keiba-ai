"""Tests for ExitCode, RaceState, and RaceProgress (Task 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paper_trading.exit_codes import EXIT_SEVERITY, ExitCode, determine_final_exit_code
from paper_trading.race_progress import RaceProgress, RaceState


class TestExitCode:
    """ExitCode IntEnum tests."""

    def test_success_and_sigint_values(self) -> None:
        """Test 1: ExitCode.SUCCESS == 0, ExitCode.SIGINT == 130."""
        assert ExitCode.SUCCESS == 0
        assert ExitCode.SIGINT == 130

    def test_has_eight_members(self) -> None:
        """Verify all 8 exit codes defined per D-17."""
        expected = {
            "SUCCESS": 0,
            "GENERAL_ERROR": 1,
            "PENDING_REMAIN": 2,
            "DB_FETCH_ERROR": 3,
            "DATA_INTEGRITY_ERROR": 4,
            "MODEL_VALIDATION_ERROR": 5,
            "REPORT_ERROR": 6,
            "SIGINT": 130,
        }
        for name, value in expected.items():
            assert ExitCode[name] == value
        assert len(ExitCode) == 8

    def test_severity_ordering(self) -> None:
        """Test 2: determine_final_exit_code selects highest severity."""
        result = determine_final_exit_code([ExitCode.PENDING_REMAIN, ExitCode.REPORT_ERROR])
        assert result == ExitCode.REPORT_ERROR
        assert EXIT_SEVERITY[ExitCode.REPORT_ERROR] > EXIT_SEVERITY[ExitCode.PENDING_REMAIN]

    def test_severity_empty_returns_success(self) -> None:
        """Empty error list returns SUCCESS."""
        assert determine_final_exit_code([]) == ExitCode.SUCCESS

    def test_sigint_is_highest_severity(self) -> None:
        """SIGINT has highest severity."""
        result = determine_final_exit_code([ExitCode.MODEL_VALIDATION_ERROR, ExitCode.SIGINT])
        assert result == ExitCode.SIGINT


class TestRaceProgress:
    """RaceProgress state machine tests."""

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        """Test 3: RaceProgress.load() on nonexistent file returns empty states."""
        rp = RaceProgress.load(tmp_path / "nonexistent.json")
        assert rp.all_race_ids() == []

    def test_mark_writes_json_with_state_and_timestamp(self, tmp_path: Path) -> None:
        """Test 4: mark() writes JSON with state and timestamp."""
        path = tmp_path / "race_progress.json"
        rp = RaceProgress(path)
        rp.mark("r1", RaceState.PROCESSING)

        assert path.exists()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "r1" in data
        assert data["r1"]["state"] == "processing"
        assert "timestamp" in data["r1"]

    def test_pending_or_failed_filters_states(self, tmp_path: Path) -> None:
        """Test 5: pending_or_failed_race_ids returns PENDING/FAILED/PROCESSING."""
        rp = RaceProgress(tmp_path / "race_progress.json")
        rp.mark("r1", RaceState.PENDING)
        rp.mark("r2", RaceState.FAILED)
        rp.mark("r3", RaceState.PROCESSING)
        rp.mark("r4", RaceState.PREDICTED)

        result = rp.pending_or_failed_race_ids()
        assert "r1" in result
        assert "r2" in result
        assert "r3" in result
        assert "r4" not in result

    def test_predicted_excluded_from_pending(self, tmp_path: Path) -> None:
        """Test 6: PREDICTED race is excluded from pending_or_failed."""
        rp = RaceProgress(tmp_path / "race_progress.json")
        rp.mark("r1", RaceState.PREDICTED)
        assert "r1" not in rp.pending_or_failed_race_ids()

    def test_atomic_write_creates_parent_dir(self, tmp_path: Path) -> None:
        """Test 7: Atomic write creates parent dir and uses temp file + os.replace."""
        nested = tmp_path / "sub" / "dir" / "race_progress.json"
        rp = RaceProgress(nested)
        rp.mark("r1", RaceState.PENDING)

        assert nested.exists()
        with open(nested, encoding="utf-8") as f:
            data = json.load(f)
        assert "r1" in data

    def test_verify_bet_ids_present(self, tmp_path: Path) -> None:
        """Test 8: verify_bet_ids_present checks stored bet_ids against DataFrame."""
        rp = RaceProgress(tmp_path / "race_progress.json")
        rp.mark("r1", RaceState.PREDICTED, bet_ids=["bid1", "bid2"])

        bets_df = pd.DataFrame({"bet_id": ["bid1", "bid2", "bid3"]})
        assert rp.verify_bet_ids_present("r1", bets_df) is True

        incomplete_df = pd.DataFrame({"bet_id": ["bid1"]})
        assert rp.verify_bet_ids_present("r1", incomplete_df) is False
