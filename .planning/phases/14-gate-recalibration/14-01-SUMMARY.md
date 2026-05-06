---
phase: 14-gate-recalibration
plan: 01
subsystem: ml-pipeline
tags: [drift-diagnostics, ks-test, wasserstein-distance, scipy, gate-retraining, ensemble]

# Dependency graph
requires:
  - phase: 13-ensemble-enhancement
    provides: StackedEnsemble OOF predictions in _train_submodel()
provides:
  - compute_drift_diagnostics() module with KS/Wasserstein drift detection
  - Pipeline-integrated drift diagnostics (ensemble mode only)
  - Gate retraining verification test (D-08 Part 1 + Part 2)
affects: [15-ev-lower-threshold, 16-oddsband-filter, 17-optuna]

# Tech tracking
tech-stack:
  added: [scipy.stats.ks_2samp, scipy.stats.wasserstein_distance]
  patterns: [pipeline-integrated drift diagnostics with TimingContext isolation]

key-files:
  created:
    - src/models/drift_diagnostics.py
    - tests/test_drift_diagnostics.py
  modified:
    - src/pipelines/training_pipeline.py
    - tests/test_win_selection_gate.py

key-decisions:
  - "drift diagnostics uses _compute_leaf_stats() for non-recursive column stats, avoiding infinite recursion in surface/year splits"
  - "pipeline integration guarded by use_ensemble=True with own TimingContext"

patterns-established:
  - "Pipeline-integrated diagnostics: TimingContext-isolated, ensemble-only, JSON output to data/backtest/"
  - "D-08 two-tier verification: unit test with fixture data + runtime assertion in pipeline"

requirements-completed: [GATE-01, GATE-02]

# Metrics
duration: 19min
completed: 2026-05-06
---

# Phase 14 Plan 01: Drift Diagnostics Module + Gate Retraining Tests Summary

**ks_2samp/wasserstein_distance drift diagnostics module with surface/year splits, integrated into ensemble backtest pipeline with gate retraining verification**

## Performance

- **Duration:** 19 min
- **Started:** 2026-05-06T00:04:23Z
- **Completed:** 2026-05-06T00:23:18Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created drift_diagnostics.py with compute_drift_diagnostics() and console_summary() functions
- 8 unit tests for drift diagnostics covering basic stats, drift detection, surface/year splits, JSON output, missing columns, and logging
- Pipeline integration in _train_submodel() with TimingContext isolation, ensemble-only guard
- Runtime assertions (D-08 Part 2) verify gate trains with non-empty edges in ensemble mode
- Gate retraining test (D-08 Part 1) proves edges differ between single-model and ensemble fixture data
- Full regression suite: 1266 passed, 1 skipped

## Task Commits

Each task was committed atomically:

1. **Task 1: Create drift diagnostics module with tests (TDD)** - `59729e8` (test) + `bfe99fc` (feat)
2. **Task 2: Integrate drift diagnostics into pipeline + gate retraining test** - `bcab171` (feat)

_Note: Task 1 followed TDD: RED commit (failing tests) then GREEN commit (implementation)._

## Files Created/Modified
- `src/models/drift_diagnostics.py` - Drift diagnostics module: compute_drift_diagnostics(), console_summary(), DRIFT_COLUMNS
- `tests/test_drift_diagnostics.py` - 8 unit tests for drift diagnostics
- `src/pipelines/training_pipeline.py` - Pipeline integration of drift diagnostics in _train_submodel()
- `tests/test_win_selection_gate.py` - Gate retraining verification test (test_gate_edges_differ_between_single_and_ensemble_oof)

## Decisions Made
- Extracted `_compute_leaf_stats()` helper to avoid infinite recursion when surface/year splits recursively call compute_drift_diagnostics()
- Pipeline uses its own `TimingContext(f"{surface}/drift_diagnostics")` separate from `TimingContext(f"{surface}/win_selection_gate_train")` to keep timing measurements accurate
- Gate training wrapped in its own `TimingContext(f"{surface}/win_selection_gate_train")` instead of the original combined block

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed infinite recursion in year-split breakdown**
- **Found during:** Task 1 (drift diagnostics implementation)
- **Issue:** compute_drift_diagnostics() recursively called itself for year splits, which then attempted year splits again, causing infinite recursion and RecursionError
- **Fix:** Extracted `_compute_leaf_stats()` internal helper that computes column stats + baseline comparison without recursive splitting. Year and surface breakdowns call this leaf function instead of the full function.
- **Files modified:** src/models/drift_diagnostics.py
- **Verification:** All 8 tests pass, no recursion errors
- **Committed in:** bfe99fc (Task 1 commit)

**2. [Rule 1 - Bug] Fixed pytest.warns(None) TypeError in test 6**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** pytest.warns(None) raises TypeError in pytest 8.x (NoneType is not a Warning subclass)
- **Fix:** Removed pytest.warns(None) wrapper; the test just calls the function directly and checks the result
- **Files modified:** tests/test_drift_diagnostics.py
- **Verification:** test_recommendations_on_drift passes
- **Committed in:** bfe99fc (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Drift diagnostics module ready for use in ensemble backtests (run_backtest.py --ensemble)
- Gate retraining verification ensures ensemble OOF produces different edges than single-model
- Phase 14-02 (use_ensemble propagation test) can proceed independently

---
*Phase: 14-gate-recalibration*
*Completed: 2026-05-06*

## Self-Check: PASSED

- All 4 created/modified files verified on disk
- All 3 commit hashes verified in git log
- 1266 tests passed, 1 skipped (full regression)
