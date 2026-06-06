"""Race progress state machine with atomic JSON writes (D-06)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RaceState(StrEnum):
    """Per-race processing states (D-06)."""

    PENDING = "pending"
    PROCESSING = "processing"
    PREDICTED = "predicted"
    NO_BET = "no_bet"
    FAILED = "failed"


class RaceProgress:
    """Track per-race state with atomic JSON writes and resume support (D-06).

    States:
        PENDING -> PROCESSING -> PREDICTED | NO_BET | FAILED

    On resume, PENDING/FAILED/PROCESSING races are re-processed.
    PREDICTED/NO_BET races are skipped.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._states: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        """Load progress from JSON file, or return empty dict if not found."""
        if not self._path.exists():
            return {}
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            logger.warning("Invalid race_progress format, starting fresh")
            return {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load race_progress: %s, starting fresh", e)
            return {}

    @classmethod
    def load(cls, path: Path) -> RaceProgress:
        """Factory: load existing progress or create empty."""
        return cls(path)

    def mark(
        self,
        race_id: str,
        state: RaceState,
        **metadata: Any,
    ) -> None:
        """Record state transition for a race with atomic write."""
        entry: dict[str, Any] = {
            "state": str(state),
            "timestamp": datetime.now().isoformat(),
        }
        entry.update(metadata)
        self._states[race_id] = entry
        self._atomic_write()

    def _atomic_write(self) -> None:
        """Atomic JSON write via tempfile.mkstemp + os.replace with Windows retry."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self._states, indent=2, ensure_ascii=False, default=str)

        fd, tmp_path = tempfile.mkstemp(
            suffix=".json",
            prefix=".race_progress_",
            dir=str(self._path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            # Windows PermissionError retry (from reconciler.py pattern)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    os.replace(tmp_path, str(self._path))
                    return
                except PermissionError:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.1)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def pending_or_failed_race_ids(self) -> list[str]:
        """Return race_ids in PENDING, FAILED, or PROCESSING states."""
        resumable = {RaceState.PENDING, RaceState.FAILED, RaceState.PROCESSING}
        return [
            race_id for race_id, entry in self._states.items() if entry.get("state") in resumable
        ]

    def verify_bet_ids_present(self, race_id: str, bets_df: Any) -> bool:
        """Check that bet_ids stored for race_id all exist in bets_df['bet_id']."""
        entry = self._states.get(race_id)
        if entry is None:
            return False
        bet_ids = entry.get("bet_ids", [])
        if not bet_ids:
            return True
        if not hasattr(bets_df, "columns"):
            return False
        if "bet_id" not in bets_df.columns:
            return False
        existing = set(bets_df["bet_id"].tolist())
        return all(bid in existing for bid in bet_ids)

    def get_state(self, race_id: str) -> str | None:
        """Return the state string for a race, or None if not tracked."""
        entry = self._states.get(race_id)
        if entry is None:
            return None
        return entry.get("state")

    def all_race_ids(self) -> list[str]:
        """Return list of all tracked race_ids."""
        return list(self._states.keys())

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Return internal _states for serialization."""
        return dict(self._states)
