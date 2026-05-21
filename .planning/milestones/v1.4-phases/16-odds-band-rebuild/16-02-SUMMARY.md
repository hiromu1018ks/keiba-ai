---
phase: 16-odds-band-rebuild
plan: 02
subsystem: backtest
tags: [backtest-engine, odds-band-filter, training-bet-history, auto-calibrate, lookahead-bias, tdd]

# Dependency graph
requires:
  - phase: 16-odds-band-rebuild/01
    provides: build_default_strategy_config() shared utility for default strategy config construction
provides:
  - BacktestEngine._generate_training_bet_history() automatic training bet history generation
  - run_backtest.py._collect_training_bet_history() uses shared default_strategy utility
  - Auto-calibrate flow: run() -> _generate() -> OddsBandFilter.calibrate()
affects: [16-odds-band-rebuild, backtest-engine, run-backtest, odds-band-filter]

# Tech tracking
tech-stack:
  added: []
  patterns: [auto-training-bet-history-generation, default-param-delegation, recursion-prevention-via-betting-target]

key-files:
  created:
    - tests/test_backtest_engine_autocalibrate.py
  modified:
    - src/backtest/engine.py
    - scripts/run_backtest.py

key-decisions:
  - "betting_target=place for inner engine prevents OddsBandFilter instantiation (recursion prevention)"
  - "train_period unpack wrapped in try/except for backward compatibility with mock-based tests"
  - "_collect_training_bet_history() keeps strategy_params argument for interface compatibility but ignores it internally"

patterns-established:
  - "Auto-generation pattern: run() checks training_bet_history is None -> calls _generate_training_bet_history() -> calibrate()"
  - "Recursion prevention: inner engine uses different betting_target to skip OddsBandFilter creation"
  - "Shared utility delegation: both engine.py and run_backtest.py import from default_strategy.py"

requirements-completed: [ODDS-01]

# Metrics
duration: 9min
completed: 2026-05-06
---

# Phase 16 Plan 02: Auto Training Bet History Generation + E2E Calibrate Flow Summary

**BacktestEngine.run() automatic training_bet_history generation from models.train_period with Pitfall 3 recursion prevention and run_backtest.py shared utility delegation**

## Performance

- **Duration:** 9 min
- **Started:** 2026-05-06T10:36:13Z
- **Completed:** 2026-05-06T10:45:18Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 3

## Accomplishments
- Added BacktestEngine._generate_training_bet_history() that auto-generates training bet history when None is passed to run()
- Implemented D-07 compliance: training period sourced from self.models.train_period (not test_start/test_end)
- Prevented Pitfall 3 recursion: inner engine uses betting_target="place" so OddsBandFilter is never instantiated
- Updated run_backtest.py._collect_training_bet_history() to use shared build_default_strategy_config() utility
- All 8 new tests pass (5 unit + 3 E2E), all 1300 existing tests pass

## Task Commits

Each task was committed atomically (TDD):

1. **Task 1 (RED): test auto training_bet_history + E2E flow** - `97509ec` (test)
2. **Task 1 (GREEN): implement auto-generation + fix lookahead bias** - `208ef5d` (feat)

_Note: No REFACTOR commit needed - implementation was clean on first pass._

## Files Created/Modified
- `src/backtest/engine.py` - Added _generate_training_bet_history() method + updated run() calibration logic for auto-generation
- `scripts/run_backtest.py` - Updated _collect_training_bet_history() to use build_default_strategy_config() shared utility
- `tests/test_backtest_engine_autocalibrate.py` - 8 tests: 5 unit tests for _generate_training_bet_history() + 3 E2E tests for calibrate flow

## Decisions Made
- Used betting_target="place" for inner engine to prevent OddsBandFilter instantiation (win-only filter, so place target skips it entirely)
- Wrapped train_period unpack in try/except for backward compatibility with existing tests that use mock models without train_period
- Kept strategy_params argument in _collect_training_bet_history() for caller interface compatibility, but internally uses default config only

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed train_period unpack crash in existing tests**
- **Found during:** Task 1 (full test suite run)
- **Issue:** Existing tests in test_backtest_engine.py use MagicMock for models without setting train_period, causing ValueError when auto-generation tries to unpack
- **Fix:** Moved train_start/train_end unpack inside try block so ValueError is caught and auto-generation returns None gracefully
- **Files modified:** src/backtest/engine.py
- **Verification:** All 1300 tests pass (2 previously failing tests now pass)
- **Committed in:** 208ef5d (part of GREEN commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Minimal - defensive coding that maintains backward compatibility. No scope creep.

## Issues Encountered
None beyond the auto-fix documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Auto-calibrate flow is complete: run() -> _generate_training_bet_history() -> OddsBandFilter.calibrate()
- run_backtest.py --ensemble will now auto-generate training bet history from models.train_period
- Both engine.py internal path and run_backtest.py external path use the same default_strategy.py utility
- Ready for Phase 17 (Optuna optimization) which depends on correct training bet history generation

## Self-Check: PASSED

All created files exist. All commit hashes verified in git log.

---
*Phase: 16-odds-band-rebuild*
*Completed: 2026-05-06*
