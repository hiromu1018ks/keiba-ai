---
phase: 54-automation-reporting
plan: 02
subsystem: paper_trading
tags: [run-mode, orchestrator, cli, exit-codes, crash-resume, cross-validation, signal-handler]
requires:
  - phase: 54-01
    provides: [exit_codes, race_progress, report_aggregator]
provides:
  - RunModeOrchestrator class with 6-phase lifecycle
  - "--mode run CLI integration with SIGINT handler"
  - "D-15 Aggregator call in reconcile mode"
  - Input snapshot with SHA256 hash and metadata (D-09)
affects: [paper_trading, run_paper_trading CLI]

tech_stack:
  added: []
  patterns: [sequential per-race prediction, crash resume via RaceProgress, signal-based cancellation]

key_files:
  created:
    - src/paper_trading/run_orchestrator.py
    - tests/test_run_orchestrator.py
    - tests/test_cli_run_mode.py
  modified:
    - scripts/run_paper_trading.py

key-decisions:
  - D-01 Smart resume: re-running skips predicted/no_bet races
  - D-02 --date required for run mode, pending at end = exit code 2
  - D-03 Schedule reuse: existing schedule.json is validated and reused
  - D-08 Cross-validation: PREDICTED races with missing bets marked FAILED for reprocessing
  - D-09 Input snapshots include _snapshot_hash, _parent_session_id, _source_info
  - D-15 Reconcile mode calls PaperTradingReportAggregator.save_outputs()
  - D-16 Report failure does NOT roll back bets/reconciliation
  - D-17 ExitCode IntEnum mapped to all error scenarios

patterns-established:
  - "RunModeOrchestrator: composition root pattern for PT lifecycle"
  - "SIGINT handler: sets _cancelled flag on orchestrator via module attribute"
  - "Input snapshots: SHA256 hash + parent session + source info for replay readiness"

requirements-completed: [AUT-01, AUT-02, AUT-03]

metrics:
  duration: 15m
  completed: "2026-06-06"
  tasks: 2
  tests: 17
  files: 4
---

# Phase 54 Plan 02: Run Mode Orchestrator Summary

RunModeOrchestrator with 6-phase PT lifecycle, crash resume via RaceProgress cross-validation, --mode run CLI integration with SIGINT handler, D-15 Aggregator wiring in reconcile mode

## One-liner

RunModeOrchestrator class executing schedule->TC->predict->reconcile->report with crash resume, cross-validation (D-08), structured exit codes, and PaperTradingReportAggregator integration in both run and reconcile modes

## Performance

- **Duration:** 15 min
- **Started:** 2026-06-06T11:45:26Z
- **Completed:** 2026-06-06T12:00:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- RunModeOrchestrator class with 6-phase lifecycle (ensure_schedule, fetch_TC, predict_races, reconcile, aggregate_and_report)
- Crash resume: PREDICTED/NO_BET races skipped, FAILED/PROCESSING/PENDING reprocessed
- Cross-validation (D-08): detects PREDICTED races with missing bets in bets.parquet, marks FAILED
- Input snapshots with SHA256 hash, parent session ID, and source info metadata (D-09)
- --mode run as 6th CLI mode with SIGINT handler and ExitCode-based exit
- _run_reconcile() calls PaperTradingReportAggregator.save_outputs() (D-15)
- 17 tests passing (13 orchestrator + 4 CLI structure)

## Task Commits

Each task was committed atomically:

1. **Task 1 (TDD RED): Failing tests for RunModeOrchestrator** - `31d183a` (test)
2. **Task 1 (TDD GREEN): RunModeOrchestrator implementation** - `fce0633` (feat)
3. **Task 2: CLI integration and D-15 Aggregator wiring** - `362fa2f` (feat)

## Files Created/Modified
- `src/paper_trading/run_orchestrator.py` - RunModeOrchestrator class (650+ lines): 6-phase lifecycle, crash resume, cross-validation, input snapshots, cancellation
- `tests/test_run_orchestrator.py` - 13 tests: fresh session, resume predicted/no_bet/failed, cross-validation, schedule reuse, DB error, model error, pending remain, cancellation, happy path, input snapshots
- `tests/test_cli_run_mode.py` - 4 tests: parse_args "run" choice, _run_run_mode callable, invalid mode rejection, Aggregator reference in _run_reconcile
- `scripts/run_paper_trading.py` - Modified: parse_args adds "run" choice, _handle_sigint(), _run_run_mode(), main() run branch, _run_reconcile() D-15 Aggregator call

## Decisions Made
- Tests reload RaceProgress from disk after _predict_races() because the method creates its own RaceProgress instance internally
- _run_reconcile() constructs SessionManifest from JSON file (from_json classmethod does not exist on dataclass) rather than deserializing directly
- Schedule reuse is validated by checking both date match and non-empty race list
- TC fetch failure is non-fatal (some venues/days have no TC data)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Initial test failures due to _schedule not being set when calling _predict_races directly; fixed by calling _ensure_schedule() in tests
- ruff lint: unused imports and line-length issues auto-fixed; long lines in _save_input_snapshot refactored

## Next Phase Readiness
- RunModeOrchestrator ready for production use via --mode run
- Plan 03 can extend with HTML report integration and Slack notifications for run mode
- No blockers

## Self-Check: PASSED

- src/paper_trading/run_orchestrator.py: EXISTS
- tests/test_run_orchestrator.py: EXISTS
- tests/test_cli_run_mode.py: EXISTS
- scripts/run_paper_trading.py: EXISTS (modified)
- Commit 31d183a: FOUND
- Commit fce0633: FOUND
- Commit 362fa2f: FOUND
- 17/17 tests passing

---
*Phase: 54-automation-reporting*
*Completed: 2026-06-06*
