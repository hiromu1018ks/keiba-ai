# Phase 54: Automation & Reporting - Research

**Researched:** 2026-06-06
**Domain:** Paper Trading pipeline orchestration, report aggregation
**Confidence:** HIGH

## Summary

Phase 54 adds `--mode run` as a 6th mode to `scripts/run_paper_trading.py` that orchestrates the full PT lifecycle (setup -> TC fetch -> sequential per-race prediction -> reconcile -> aggregate -> report) in a single command. It also introduces `PaperTradingReportAggregator` as the single source of truth for all statistical aggregation (daily, weekly, per-target), shrinking `PaperTradingReport` to a pure HTML renderer.

The existing codebase provides strong foundations: `PaperReconciler` (3-column state model, atomic writes, idempotent bet_id), `SessionManifest` (PFP, code version, model identity), `RaceWatcher` (post_time parsing, skip-already-processed), and `_build_race_predictor()` (shared across predict/diagnose/dry-run). The `--mode run` orchestrator must weave these together with a new `race_progress.json` progress tracker and graceful Ctrl+C handling.

**Primary recommendation:** Build a `RunModeOrchestrator` class in `src/paper_trading/run_orchestrator.py` that encapsulates the 6-phase lifecycle. Add `race_progress.json` for crash-resume state tracking. Create `PaperTradingReportAggregator` as the single aggregation engine consumed by both JSON output and HTML rendering.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `--mode run` is smart resume. Restores state from session_manifest / schedule.json / bets.parquet / race_progress.json. Auto-detects incomplete steps and resumes.
- **D-02:** Single-day session only. `--date` required. Each day has independent session_id / schedule / live TC snapshot / session_manifest. No cross-day. Pending at midnight -> save + exit code 2.
- **D-03:** Setup is embedded. schedule.json is validated and reused if exists; otherwise auto-setup.
- **D-04:** Live track condition fetched once before first race prediction (batch for all venues). Validate measurement time, fetch time, race date, venue. Fail-fast on stale/missing/partial. Fetched HTML and normalized data are immutable within session.
- **D-05:** Sequential per-race prediction. N minutes before each race: fetch latest odds + horse weight -> predict -> record. TC fixed at morning but odds/weight per-race. Reconcile retry starts after last race post time.
- **D-06:** `race_progress.json` with atomic write. States: `pending -> processing -> predicted | no_bet | failed`. Records: state + timestamp + input snapshot hash + bet_id list + failure reason. On resume: skip predicted/no_bet, reprocess pending/failed/processing.
- **D-07:** Input snapshots saved to `sessions/{session_id}/inputs/{race_id}.parquet` for each race.
- **D-08:** Cross-validate on resume: race_progress / bets.parquet / input snapshot 3-file consistency.
- **D-09:** Replay feature: create new replay session from saved inputs, re-predict. Original session/bets unchanged.
- **D-10:** bets.parquet = bet records, race_progress.json = processing progress, input snapshot = reproducibility. Cumulative history from bets.parquet only, no duplicate copies.
- **D-11:** PaperTradingReportAggregator is the single aggregation implementation. New schema (settlement_status/outcome/payout) bets.parquet -> daily/weekly/monthly/target stats. CLI JSON, HTML, future notifications share same aggregated results.
- **D-12:** PaperTradingReport shrinks to HTML renderer. Receives Aggregator results only.
- **D-13:** Output structure: `daily_summary/YYYY/YYYY-MM-DD.json`, `weekly_summary/{iso_year}/W{iso_week:02d}.json`, `target_summary/YYYY-MM-DD.json` + `target_summary/latest.json`, `report.html`. Common fields: schema_version, period, generated_at, session_ids.
- **D-14:** Aggregator auto-generates all report types at run end. After reconcile retry completes, even with pending remaining. Aggregation uses settled only for ROI; reports pending count, unsettled stake, data completeness status.
- **D-15:** Existing reconcile mode also calls Aggregator post-reconciliation for report updates.
- **D-16:** Report generation failure does not roll back bets/reconciliation. Returns exit code 6.
- **D-17:** IntEnum ExitCode: 0=SUCCESS, 1=GENERAL_ERROR, 2=PENDING_REMAIN, 3=DB_FETCH_ERROR, 4=DATA_INTEGRITY_ERROR, 5=MODEL_VALIDATION_ERROR, 6=REPORT_ERROR, 130=SIGINT.
- **D-18:** Multi-error severity priority table. Highest severity determines final exit code. All error details saved to session_manifest as array.
- **D-19:** Model identity (MLflow run ID, training period, manifest hash) from session_manifest included in all reports (RPT-04).

### Claude's Discretion
- RaceWatcher and run mode integration details (sleep intervals, post_time detection)
- Minutes-before N configuration (CLI arg vs config file)
- race_progress.json atomic write implementation (temp file naming)
- sessions/{session_id}/ directory structure details
- Aggregator weekly aggregation timing (ISO week boundary)
- Severity priority table definition
- Replay session CLI argument design
- PaperTradingReport HTML template new-schema adaptation
- Input snapshot parquet schema (which columns to save)

### Deferred Ideas (OUT OF SCOPE)
- SafetyGuard integration (v2.5+)
- Wide bet support (v2.5+)
- Conservative MAWC redesign (v2.5+)
- WinSegmentCalibrator dead code removal (v2.5+)
- Auto deployment gate (DEP-01, v2.5+)
- Optuna 19-dim optimization (DEP-02, v2.5+)
- Daemon/period-run mode
- RaceWatcher real-time monitoring
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AUT-01 | One-command run mode -- `--mode run` for full PT lifecycle | RunModeOrchestrator design (see Architecture Patterns) |
| AUT-02 | Restart resumption -- idempotent, crash-recovery via race_progress.json | race_progress.json state machine + 3-file cross-validation |
| AUT-03 | DB failure exit codes -- non-zero exit for DB/model/integrity errors | ExitCode IntEnum (D-17) with severity priority (D-18) |
| RPT-01 | Weekly aggregation -- ISO week ROI/hit-rate/bet-count JSON | PaperTradingReportAggregator weekly aggregation logic |
| RPT-02 | Cumulative history with losses -- pending/settled/won/lost in cumulative record | Aggregator reads bets.parquet (source of truth, D-10) |
| RPT-03 | Per-target aggregation -- Win/Place separate ROI/hit-rate | Aggregator group-by bet_type with target_summary output |
| RPT-04 | Model identity in reports -- MLflow run ID, training period, manifest hash | SessionManifest.to_dict() provides all fields (D-19) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Run mode orchestration | CLI script (composition root) | src/paper_trading/ | `run_paper_trading.py` is the composition root; RunModeOrchestrator encapsulates lifecycle |
| Race scheduling / progress | src/paper_trading/ | CLI | race_progress.json I/O + state machine logic belongs in paper_trading package |
| Sequential per-race prediction | src/paper_trading/ | src/backtest/ | RacePredictor does inference; orchestrator calls it per-race |
| Track condition fetch | src/ingestion/ | -- | JRATrackConditionFetcher already exists from Phase 53 |
| Reconciliation | src/paper_trading/ | -- | PaperReconciler already complete from Phase 51 |
| Statistical aggregation | src/paper_trading/ | -- | New PaperTradingReportAggregator class |
| HTML rendering | src/paper_trading/ | -- | PaperTradingReport shrinks to renderer (D-12) |
| Exit code management | domain/ or paper_trading/ | CLI | ExitCode IntEnum used by orchestrator and main() |
| Signal handling (Ctrl+C) | CLI script | -- | signal.signal() at main() level |
| Session manifest | src/features/ | -- | SessionManifest already exists, extended for errors |

## Standard Stack

### Core (all already installed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | 2.3.3 | DataFrame I/O, aggregation | Existing data layer |
| jinja2 | 3.1.6 | HTML report rendering | Already used by PaperTradingReport |
| Python stdlib `signal` | 3.11 | SIGINT handling | No external dependency needed |
| Python stdlib `enum.IntEnum` | 3.11 | ExitCode enum | No external dependency needed |
| Python stdlib `json` | 3.11 | JSON I/O for race_progress, reports | No external dependency needed |
| Python stdlib `tempfile` + `os.replace` | 3.11 | Atomic writes | Already used in PaperReconciler |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib | stdlib | SHA256 for bet_id, input snapshot hash | Deterministic identity |
| time | stdlib | Sleep in polling loops, retry | Sequential per-race wait |
| datetime | stdlib | ISO week calculation, timestamp | Weekly aggregation |

### New Packages Required

**None.** All functionality uses existing dependencies.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual signal handler | `asyncio` event loop | asyncio overkill for sequential blocking flow; signal.signal() is simpler |
| JSON race_progress | SQLite for progress | SQLite adds complexity for a single-session progress file; JSON is simpler and human-readable |
| IntEnum exit codes | Plain int constants | IntEnum gives type safety, string representation, and extensibility |

## Package Legitimacy Audit

> No new external packages are required for this phase. All functionality uses Python standard library or already-installed packages (pandas, jinja2). slopcheck was unavailable, but no new packages need verification.

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                          run_paper_trading.py --mode run --date YYYY-MM-DD
                                      |
                            [1. Parse args + load config]
                                      |
                            [2. Load models (ModelLoader)]
                                      |
                            [3. Restore or create session]
                              session_manifest.json
                              schedule.json
                              race_progress.json
                              bets.parquet
                                      |
                     +--[resume?]------+
                     |                 |
                [fresh session]   [resume from state]
                     |                 |
                     +--------+--------+
                              |
                    [4. Fetch live TC (once)]
                     JRATrackConditionFetcher
                              |
                    [5. Sequential per-race loop]
                     +--- for each race_id ---+
                     |                        |
              [5a. Wait until N min    [Skip if predicted/
                   before post_time]    no_bet in progress]
                     |                        |
              [5b. Fetch latest odds]         |
              [5c. Build features]            |
              [5d. Predict -> bets]           |
              [5e. Save input snapshot]       |
              [5f. Update race_progress]      |
                     |                        |
                     +---------+--------------+
                               |
                    [6. Wait for last race post_time]
                               |
                    [7. Reconcile (PaperReconciler)]
                               |
                    [8. Aggregate (ReportAggregator)]
                     daily / weekly / target JSON
                               |
                    [9. Render HTML (PaperTradingReport)]
                               |
                    [10. Write session_manifest final]
                               |
                    [Exit with ExitCode]
```

### Recommended Project Structure

```
src/paper_trading/
  __init__.py              # (exists, empty)
  config.py                # PaperTradingConfig (exists)
  predictor.py             # PaperPredictor (exists, unchanged)
  reconciler.py            # PaperReconciler (exists, reused)
  report.py                # PaperTradingReport -> shrunk to HTML renderer
  watcher.py               # RaceWatcher (exists, patterns reused)
  run_orchestrator.py      # NEW: RunModeOrchestrator
  report_aggregator.py     # NEW: PaperTradingReportAggregator
  race_progress.py         # NEW: RaceProgress state machine
  exit_codes.py            # NEW: ExitCode IntEnum

scripts/
  run_paper_trading.py     # MODIFIED: add --mode run, signal handler

data/paper_trading/
  schedule.json            # (existing)
  session_manifest.json    # (existing)
  bets.parquet             # (existing, source of truth)
  race_progress.json       # NEW: per-race processing state
  sessions/{session_id}/   # NEW: session artifacts
    inputs/{race_id}.parquet  # input snapshots
    race_progress.json     # (or shared with top-level)
  daily_summary/YYYY/YYYY-MM-DD.json  # NEW structure (year subdirectory)
  weekly_summary/{iso_year}/W{iso_week:02d}.json  # NEW
  target_summary/YYYY-MM-DD.json      # NEW
  target_summary/latest.json          # NEW
  report.html              # (existing, updated schema)
```

### Pattern 1: RunModeOrchestrator Lifecycle

**What:** Encapsulates the 6-phase run mode lifecycle (setup -> TC -> predict -> reconcile -> aggregate -> report).
**When to use:** `--mode run` in `run_paper_trading.py`.
**Example:**

```python
# src/paper_trading/run_orchestrator.py
class RunModeOrchestrator:
    def __init__(self, config, models, store, args, strategy_config, session_manifest):
        self.config = config
        self.models = models
        self.store = store
        self.args = args
        self.strategy_config = strategy_config
        self.session_manifest = session_manifest
        self.errors: list[dict[str, Any]] = []

    def execute(self) -> ExitCode:
        """Full lifecycle. Returns final ExitCode."""
        try:
            self._ensure_schedule()       # D-03
            self._fetch_track_conditions() # D-04
            self._predict_races()          # D-05, D-06
            self._reconcile()              # via PaperReconciler
            self._aggregate_and_report()   # D-11, D-14
        except KeyboardInterrupt:
            return ExitCode.SIGINT
        except DBFetchError:
            return ExitCode.DB_FETCH_ERROR
        # ... etc
        return self._determine_exit_code()  # D-18
```

### Pattern 2: RaceProgress State Machine

**What:** Atomic-write JSON tracking per-race processing state.
**When to use:** Sequential per-race prediction with crash recovery.
**Example:**

```python
# src/paper_trading/race_progress.py
from enum import StrEnum
from pathlib import Path
import json, tempfile, os

class RaceState(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    PREDICTED = "predicted"
    NO_BET = "no_bet"
    FAILED = "failed"

class RaceProgress:
    """Atomic-write progress tracker for per-race state."""

    def __init__(self, path: Path):
        self.path = path
        self._states: dict[str, dict] = {}  # race_id -> {state, timestamp, ...}

    def load(self) -> None:
        """Load from disk (resume case)."""
        if self.path.exists():
            self._states = json.loads(self.path.read_text(encoding="utf-8"))

    def mark(self, race_id: str, state: RaceState, **metadata) -> None:
        """Update state and atomic-write."""
        self._states[race_id] = {"state": state, "timestamp": now_iso(), **metadata}
        self._atomic_write()

    def _atomic_write(self) -> None:
        """Same pattern as PaperReconciler._atomic_write_parquet."""
        # temp file + os.replace

    def pending_or_failed_race_ids(self) -> list[str]:
        """Return race_ids that need (re)processing."""
        return [
            rid for rid, info in self._states.items()
            if info["state"] in (RaceState.PENDING, RaceState.FAILED, RaceState.PROCESSING)
        ]
```

### Pattern 3: PaperTradingReportAggregator

**What:** Single aggregation engine for daily/weekly/target statistics from bets.parquet.
**When to use:** All report generation (run mode end, reconcile mode, future notifications).
**Example:**

```python
# src/paper_trading/report_aggregator.py
class PaperTradingReportAggregator:
    def __init__(self, bets_path: Path, session_manifest: SessionManifest):
        self.bets_path = bets_path
        self.session_manifest = session_manifest

    def aggregate_daily(self, target_date: date) -> dict:
        """Daily summary with new schema fields."""
        # Read bets.parquet, filter by date, compute ROI (settled only)
        # Include pending_count, unsettled_stake, data_completeness

    def aggregate_weekly(self, iso_year: int, iso_week: int) -> dict:
        """Weekly summary using ISO week (Monday start, JST)."""
        # Filter bets by ISO week, aggregate stats

    def aggregate_by_target(self, target_date: date | None = None) -> dict:
        """Per bet_type (win/place) breakdown."""
        # Group by bet_type, compute per-target ROI/hit-rate

    def aggregate_all(self, target_date: date) -> dict:
        """Run all aggregation types. Returns dict of results."""
        daily = self.aggregate_daily(target_date)
        iso_year, iso_week, _ = target_date.isocalendar()
        weekly = self.aggregate_weekly(iso_year, iso_week)
        target = self.aggregate_by_target(target_date)
        return {"daily": daily, "weekly": weekly, "target": target}
```

### Pattern 4: Exit Code Management

**What:** IntEnum for exit codes with severity-based resolution.
**When to use:** All `--mode run` error paths.

```python
# src/paper_trading/exit_codes.py
from enum import IntEnum

class ExitCode(IntEnum):
    SUCCESS = 0
    GENERAL_ERROR = 1
    PENDING_REMAIN = 2
    DB_FETCH_ERROR = 3
    DATA_INTEGRITY_ERROR = 4
    MODEL_VALIDATION_ERROR = 5
    REPORT_ERROR = 6
    SIGINT = 130

# Severity: higher = more severe. Used when multiple errors occur.
EXIT_SEVERITY: dict[ExitCode, int] = {
    ExitCode.SUCCESS: 0,
    ExitCode.PENDING_REMAIN: 1,
    ExitCode.REPORT_ERROR: 2,
    ExitCode.MODEL_VALIDATION_ERROR: 3,
    ExitCode.DATA_INTEGRITY_ERROR: 4,
    ExitCode.DB_FETCH_ERROR: 5,
    ExitCode.GENERAL_ERROR: 6,
    ExitCode.SIGINT: 7,
}
```

### Anti-Patterns to Avoid

- **Duplicating bets.parquet data into separate cumulative history files:** Use bets.parquet as the single source of truth (D-10). The Aggregator reads from bets.parquet directly; no copies.
- **Putting aggregation logic in PaperTradingReport:** Report must be a pure HTML renderer (D-12). All stats come from the Aggregator.
- **Modifying existing modes:** `--mode run` is the 6th mode. setup/predict/reconcile/dry-run/diagnose must remain unchanged.
- **Race-by-race TC fetching:** TC values are fixed once at morning (D-04). Only odds and horse weight are per-race.
- **Real-time polling loop:** Use schedule-based wait_until() from RaceWatcher, not busy-polling.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parquet atomic write | Custom file locking | `PaperReconciler._atomic_write_parquet()` | Already battle-tested with Windows PermissionError retry |
| JSON atomic write | Custom rename logic | `write_session_manifest()` pattern | Same temp+replace pattern already proven |
| bet_id generation | New hash function | `PaperReconciler.compute_bet_id()` | Deterministic SHA256, already tested |
| ROI calculation | Custom accumulator | `PaperReconciler._compute_roi()` | Handles effective_stake (won+lost only), max DD |
| Post-time parsing | New datetime logic | `RaceWatcher._parse_post_time()` | Already handles HH:MM -> datetime conversion |
| wait_until pattern | New sleep loop | `RaceWatcher.wait_until()` (adapted) | Already handles target_time calculation |
| Model loading | Custom MLflow logic | `_load_models()` in run_paper_trading.py | Handles ModelLoader, model_info.json save |
| RacePredictor construction | Manual kwargs | `_build_race_predictor()` | Shared across all modes (WR-01) |
| Payout map construction | Custom join logic | `build_payout_map()`, `build_win_payout_map()` | Vectorized, tested, handles edge cases |

**Key insight:** Phase 54 is primarily an orchestration/integration phase. Nearly all the "hard" logic (reconciliation, prediction, feature building, TC fetching) already exists. The new code is lifecycle management (RunModeOrchestrator), state tracking (RaceProgress), aggregation (ReportAggregator), and exit code management.

## Runtime State Inventory

> Phase 54 is greenfield (new mode + new classes), not a rename/refactor. However, the existing data files need compatibility analysis.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data (bets.parquet) | 117 rows in old v1 schema (`result` column, no `settlement_status`/`outcome`/`payout`) | CONTEXT.md says v2.4 starts fresh; old data incompatible with new reconciler (already raises ValueError). Planner must decide: clear old data or dual-read path |
| Stored data (daily_summary) | 4 JSON files in `data/paper_trading/daily_summary/YYYYMMDD.json` (flat, no year subdirectory) | D-13 requires `daily_summary/YYYY/YYYY-MM-DD.json` -- migration or new-format only |
| Stored data (schedule.json) | Not currently present on disk | Will be created by embedded setup in run mode |
| Stored data (session_manifest.json) | Not currently present on disk | Will be created by run mode |
| Live service config | None | -- |
| OS-registered state | None | -- |
| Secrets/env vars | `PGPASSWORD`, `SLACK_WEBHOOK_URL` in .env | Read-only, no changes needed |
| Build artifacts | None relevant | -- |

## Common Pitfalls

### Pitfall 1: Old vs New bets.parquet Schema Mismatch

**What goes wrong:** The existing `bets.parquet` uses v1 schema (`result` column, no `settlement_status`/`outcome`/`payout`). The Phase 51 `PaperReconciler` rejects this with ValueError: "Old schema detected". The Phase 53 `_run_predict()` writes v2 schema rows. But the file on disk is v1.
**Why it happens:** Phases 51-53 modified the code but existing data was generated with pre-Phase-51 code.
**How to avoid:** The planner must include a task to either (a) clear the old bets.parquet and daily_summary before first run-mode execution, or (b) document that v2.4 PT starts from a clean state. CONTEXT.md's "v2.4以前の PT レコード移行" is out of scope, so clearing is the expected path.
**Warning signs:** `ValueError: Old schema detected in bets.parquet` on first reconcile.

### Pitfall 2: Windows Atomic Write PermissionError

**What goes wrong:** `os.replace()` can fail with `PermissionError` if the target file is open by another process (antivirus, Parquet reader).
**Why it happens:** Windows file locking is stricter than POSIX.
**How to avoid:** Use the retry loop pattern from `PaperReconciler._atomic_write_parquet()` (3 retries, 100ms sleep). Apply the same pattern to `race_progress.json` atomic writes.
**Warning signs:** Intermittent `PermissionError` on `os.replace()`.

### Pitfall 3: Ctrl+C During Parquet Write

**What goes wrong:** If SIGINT arrives during a Parquet write, the file may be partially written (corrupt).
**Why it happens:** signal handlers run between Python bytecode instructions; file I/O is not atomic.
**How to avoid:** The signal handler should set a `threading.Event` or flag, not raise immediately. The orchestrator checks the flag between race processing steps. Atomic writes via temp+replace ensure the target file is never in a partial state.
**Warning signs:** Corrupt parquet files after Ctrl+C.

### Pitfall 4: race_progress.json Out of Sync with bets.parquet

**What goes wrong:** After crash, `race_progress.json` says `predicted` for a race, but the corresponding bets were not actually appended to `bets.parquet`.
**Why it happens:** Progress is updated before bets.parquet write completes, or crash between the two operations.
**How to avoid:** D-08 cross-validation: on resume, check that every `predicted` race has matching bet_ids in bets.parquet. If mismatch, treat as `failed` and reprocess.
**Warning signs:** Missing bets for races marked as `predicted`.

### Pitfall 5: ISO Week Boundary at Year End

**What goes wrong:** ISO week 1 of year N+1 may include days from December of year N. A December 30 race could belong to ISO week 1 of the next year.
**Why it happens:** ISO 8601 defines weeks as Monday-Sunday, and week 1 contains the year's first Thursday.
**How to avoid:** Use `date.isocalendar()` which returns `(iso_year, iso_week, iso_weekday)`. Always use `iso_year` (not calendar year) for the weekly_summary directory structure.
**Warning signs:** December races appearing in wrong year's weekly_summary.

### Pitfall 6: PaperTradingReport Still Uses Old `result` Column

**What goes wrong:** `PaperTradingReport._derive_fields()` and `_compute_monthly_stats()` reference `b["result"]` which exists in v1 schema but not in v2 schema (v2 uses `outcome` + `payout`).
**Why it happens:** Report code was not updated when Phase 51 introduced the 3-column state model.
**How to avoid:** When shrinking PaperTradingReport to HTML renderer (D-12), the Aggregator provides pre-computed fields. The renderer should never directly access `result` or `outcome` columns -- it receives structured data from the Aggregator.
**Warning signs:** KeyError on `result` when rendering HTML.

## Code Examples

### RaceWatcher wait_until Adaptation for Run Mode

```python
# Reuse RaceWatcher._parse_post_time() for post_time parsing
# Adapt wait_until() for run mode: add cancellation check
import signal

class RunModeOrchestrator:
    _cancelled = False

    def _wait_until_with_cancel(self, target_time: datetime) -> bool:
        """Wait until target_time, checking for cancellation.
        Returns True if reached target, False if cancelled."""
        now = datetime.now()
        if target_time <= now:
            return True
        remaining = (target_time - now).total_seconds()
        while remaining > 0:
            if self._cancelled:
                return False
            sleep_secs = min(remaining, 1.0)  # Check cancel every 1s
            time.sleep(sleep_secs)
            remaining = (target_time - datetime.now()).total_seconds()
        return True
```
[VERIFIED: RaceWatcher._parse_post_time() and wait_until() patterns examined from src/paper_trading/watcher.py]

### Signal Handler Setup

```python
# In run_paper_trading.py main()
import signal

def _handle_sigint(signum, frame):
    """Set cancellation flag (D-17: exit code 130)."""
    # Do NOT raise immediately -- let orchestrator finish current operation
    if hasattr(main, '_orchestrator') and main._orchestrator:
        main._orchestrator._cancelled = True
    else:
        sys.exit(130)

# In the "run" branch of main():
signal.signal(signal.SIGINT, _handle_sigint)
```
[ASSUMED: Standard Python signal handling pattern; no existing signal handling in the codebase]

### Aggregator Weekly Calculation with ISO Week

```python
from datetime import date, timedelta

def _iso_week_range(iso_year: int, iso_week: int) -> tuple[date, date]:
    """Get Monday-Sunday date range for an ISO week."""
    # ISO week 1 always contains January 4th
    jan4 = date(iso_year, 1, 4)
    start_of_week1 = jan4 - timedelta(days=jan4.weekday())  # Monday
    week_start = start_of_week1 + timedelta(weeks=iso_week - 1)
    week_end = week_start + timedelta(days=6)  # Sunday
    return week_start, week_end

# Usage: filter bets.parquet by race_date in [week_start, week_end]
```
[VERIFIED: Python 3.11 `date.isocalendar()` returns `(iso_year, iso_week, weekday)`]

### Atomic JSON Write (Reuse SessionManifest Pattern)

```python
def _atomic_write_json(data: dict, target: Path) -> None:
    """Atomic JSON write via temp file + os.replace (same as write_session_manifest)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".json", prefix=".tmp_", dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, str(target))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```
[VERIFIED: Pattern from src/features/session_manifest.py:239-267]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 5-mode PT (separate setup/predict/reconcile) | 6-mode with `--mode run` (all-in-one) | Phase 54 | Single-command lifecycle |
| Old schema bets.parquet (`result` column) | v2 schema (`settlement_status/outcome/payout`) | Phase 51 | Correct ROI with losses |
| PaperTradingReport computes stats | PaperTradingReportAggregator computes stats | Phase 54 | Single aggregation engine |
| daily_summary/YYYYMMDD.json (flat) | daily_summary/YYYY/YYYY-MM-DD.json (year subdirectory) | Phase 54 | Better directory organization |
| No crash recovery | race_progress.json + cross-validation | Phase 54 | Idempotent resume |
| No exit code taxonomy | ExitCode IntEnum with severity | Phase 54 | Structured error handling |
| No Ctrl+C handling | SIGINT handler + graceful shutdown | Phase 54 | Clean interruption |

**Deprecated/outdated:**
- `PaperTradingReport._derive_fields()`: References old `result` column. Must be replaced with Aggregator-provided data.
- `PaperTradingReport._compute_monthly_stats()`: References old `result` column. Aggregator replaces this.
- `PaperPredictor.predict_race()`: Old `result: 0.0` field in bet dicts. Phase 53 `_run_predict()` writes new schema directly.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Signal handling pattern (SIGINT -> flag -> graceful) works on Windows Python | Pattern 2 | Windows Python signal support is limited; SIGINT may not be catchable in all scenarios. Fallback: KeyboardInterrupt try/except |
| A2 | No new external packages needed for this phase | Standard Stack | If an unexpected need arises, planner should flag for human verification |
| A3 | `os.replace()` on Windows works for JSON files (same as parquet) | Pattern 4 | Already proven in session_manifest.py, but race_progress.json may have different access patterns |
| A4 | Existing bets.parquet (117 rows, v1 schema) should be cleared before first v2.4 run | Pitfall 1 | If wrong, reconciler will reject with ValueError and need manual cleanup |
| A5 | `RaceWatcher.wait_until()` can be adapted without importing the class directly | Pattern 1 | If RaceWatcher has tightly coupled dependencies, may need to extract the pure wait logic |

## Open Questions

1. **bets.parquet migration strategy**
   - What we know: Old file has 117 rows in v1 schema. PaperReconciler rejects it. CONTEXT.md defers migration.
   - What's unclear: Should the planner include a task to rename/remove the old file, or just document the incompatibility?
   - Recommendation: Include a Wave 0 task to archive old bets.parquet (rename to `bets_v1.parquet.bak`) before first run-mode execution. This preserves data without breaking new code.

2. **RaceProgress storage location**
   - What we know: D-06 says race_progress.json. D-07 says `sessions/{session_id}/inputs/`.
   - What's unclear: Should race_progress.json live in `sessions/{session_id}/race_progress.json` or at `data/paper_trading/race_progress.json` (top level)?
   - Recommendation: Place in `sessions/{session_id}/race_progress.json` since it's session-scoped. The resume logic looks up the session by date first.

3. **Minutes-before N default in run mode**
   - What we know: `--minutes-before` defaults to 5 (matches BT). Claude has discretion on this.
   - What's unclear: Whether run mode should use the same default or allow per-race configuration.
   - Recommendation: Use same `--minutes-before 5` default. Single value for the session.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | All | Yes | 3.11.15 | -- |
| pandas | Data I/O, aggregation | Yes | 2.3.3 | -- |
| jinja2 | HTML rendering | Yes | 3.1.6 | -- |
| PostgreSQL (EveryDB2) | setup/reconcile data | Yes (assumed) | localhost:5432 | run mode fails without DB |
| Playwright | TC fetch (Phase 53) | Yes (Phase 53 installed) | -- | TC fetch fails -> exit 3 |
| pytest | Testing | Yes | -- | -- |

**Missing dependencies with no fallback:**
- None identified

**Missing dependencies with fallback:**
- None identified

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A (no auth in PT pipeline) |
| V3 Session Management | no | N/A (single-user CLI) |
| V4 Access Control | no | N/A (no multi-user) |
| V5 Input Validation | yes | argparse for CLI, pydantic-style manual validation for JSON |
| V6 Cryptography | no | SHA256 used for identity hashing, not security |

### Known Threat Patterns for CLI Data Pipeline

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Data tampering (bets.parquet) | Tampering | Atomic writes, schema validation, bet_id dedup |
| Path traversal in session_id | Tampering | session_id is UUID hex, not user-controlled |
| Sensitive data exposure (DB password) | Information Disclosure | PGPASSWORD env var, not in code |

## Sources

### Primary (HIGH confidence)
- `scripts/run_paper_trading.py` — Full 1446-line source read. parse_args(), _run_predict(), _run_reconcile(), _run_setup() signatures and flows verified.
- `src/paper_trading/reconciler.py` — Full source read. PaperReconciler.reconcile(), retry_pending(), _compute_roi() signatures verified.
- `src/paper_trading/report.py` — Full source read. PaperTradingReport.generate() consumes bets list + summary dict.
- `src/paper_trading/watcher.py` — Full source read. RaceWatcher.watch() pattern, wait_until(), _parse_post_time().
- `src/paper_trading/config.py` — Full source read. PaperTradingConfig dataclass.
- `src/features/session_manifest.py` — Full source read. SessionManifest dataclass, write_session_manifest() atomic write pattern.
- `src/automation/scheduler.py` — Full source read. RaceScheduler Protocol-based DI pattern.
- `src/betting/payout_maps.py` — Full source read. build_payout_map(), build_win_payout_map() signatures.
- `data/paper_trading/bets.parquet` — Schema inspected: 15 columns, 117 rows, v1 schema (no settlement_status/outcome/payout).
- `data/paper_trading/daily_summary/*.json` — 4 files inspected, flat YYYYMMDD.json format, old result format.
- `config/settings.yaml` — Full file read. No paper_trading specific section.
- `tests/test_paper_reconciler.py` — Read first 80 lines. 36 tests all pass.

### Secondary (MEDIUM confidence)
- Phase 53 CONTEXT.md — TrackConditionFetcherProtocol pattern, DD shadow mode, regime AGGRESSIVE fixed.
- Phase 51 CONTEXT.md (referenced via STATE.md) — bet_id, 3-column state model, ROI formula.

### Tertiary (LOW confidence)
- Signal handling pattern (ASSUMED) — Standard Python SIGINT handling; not verified against Windows quirks in this specific Python build.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — No new packages; all existing dependencies verified via code read and test execution.
- Architecture: HIGH — All integration points (reconciler, watcher, report, session_manifest) read and understood. RaceProgress and Aggregator are new but follow established patterns.
- Pitfalls: HIGH — Old/new schema mismatch confirmed by inspecting actual bets.parquet on disk. Windows atomic write issue already handled in existing code.

**Research date:** 2026-06-06
**Valid until:** 2026-07-06 (stable -- no fast-moving external dependencies)
