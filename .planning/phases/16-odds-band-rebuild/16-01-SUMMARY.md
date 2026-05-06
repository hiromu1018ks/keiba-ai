---
phase: 16-odds-band-rebuild
plan: 01
subsystem: betting
tags: [optuna, strategy-optimizer, regime-detector, lookahead-bias-fix, tdd]

# Dependency graph
requires:
  - phase: 13-risk-calibration-parameter-optimization
    provides: RegimeDetector._get_base_params() hardcoded defaults, DDConfig dataclass
provides:
  - build_default_strategy_config() shared utility for default strategy config construction
  - _build_default_config() delegation method on StrategyOptimizer
  - Lookahead bias fix in _run_single_backtest() step 3
affects: [16-02, strategy_optimizer, backtest_engine, run_backtest]

# Tech tracking
tech-stack:
  added: []
  patterns: [default-strategy-single-source-of-truth, train-test-config-separation]

key-files:
  created:
    - src/betting/default_strategy.py
    - tests/test_default_strategy.py
  modified:
    - src/tuning/strategy_optimizer.py
    - tests/test_strategy_optimizer.py

key-decisions:
  - "CONSERVATIVE fractional_kelly=0.25 as top-level default (most neutral state per RESEARCH Pitfall 1)"
  - "default_strategy.py extracts config from RegimeDetector._get_base_params() hardcoded values, not Optuna search space"

patterns-established:
  - "Default config delegation: _build_default_config() delegates to build_default_strategy_config() to avoid duplication"
  - "Train/test config separation: training backtest uses default_config, test backtest uses Optuna strategy_config"

requirements-completed: [ODDS-02]

# Metrics
duration: 4min
completed: 2026-05-06
---

# Phase 16 Plan 01: Default Strategy Utility + Lookahead Bias Fix Summary

**Shared default_strategy.py utility from RegimeDetector hardcoded defaults + lookahead bias fix separating training/test strategy configs in _run_single_backtest()**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-06T10:27:38Z
- **Completed:** 2026-05-06T10:32:01Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 4

## Accomplishments
- Created build_default_strategy_config() shared utility that builds strategy config from RegimeDetector._get_base_params() hardcoded defaults
- Added _build_default_config() delegation method to StrategyOptimizer, eliminating future duplication
- Fixed lookahead bias: _run_single_backtest() step 3 now uses default_config (not Optuna's strategy_config) for training bet history generation
- Regime overrides are correctly switched: default values injected during training, then overwritten with Optuna values for test phase
- All 23 tests pass (6 new default_strategy tests + 3 new optimizer tests + 14 existing tests)

## Task Commits

Each task was committed atomically (TDD):

1. **Task 1 (RED): test default_strategy + lookahead bias tests** - `d493b65` (test)
2. **Task 1 (GREEN): implement default_strategy utility + fix lookahead bias** - `0fd3df8` (feat)

_Note: No REFACTOR commit needed - implementation was clean on first pass._

## Files Created/Modified
- `src/betting/default_strategy.py` - Shared utility: build_default_strategy_config() constructs strategy config from RegimeDetector hardcoded defaults
- `src/tuning/strategy_optimizer.py` - Added _build_default_config() delegation + fixed _run_single_backtest() step 2-3 to use default_config for training
- `tests/test_default_strategy.py` - 6 tests for build_default_strategy_config() (required keys, DDConfig defaults, regime overrides, stake calculator defaults)
- `tests/test_strategy_optimizer.py` - Added TestBuildDefaultConfig + 2 new TestRunSingleBacktest tests verifying train/test config separation

## Decisions Made
- Used CONSERVATIVE fractional_kelly=0.25 as top-level default (most neutral state, prevents aggressive betting by default)
- default_strategy.py reads from RegimeDetector._get_base_params() at call time (no caching needed since values are hardcoded)
- Regime override injection order: default during training -> Optuna values for test (clean two-phase pattern)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- default_strategy.py is ready for Plan 02 to import (BacktestEngine._generate_training_bet_history, run_backtest.py._collect_training_bet_history)
- _build_default_config() is the single entry point for default config in strategy_optimizer.py
- Lookahead bias fully resolved in _run_single_backtest()

---
*Phase: 16-odds-band-rebuild*
*Completed: 2026-05-06*
