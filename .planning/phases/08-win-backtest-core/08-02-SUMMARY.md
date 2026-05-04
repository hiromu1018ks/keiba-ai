---
phase: 08-win-backtest-core
plan: 02
subsystem: backtest
tags: [win-candidates, conformal-confidence, select-bets, wf-validation, argparse]

# Dependency graph
requires:
  - phase: 08-01
    provides: build_win_payout_map, _settle_bet WIN branch, betting_target dispatch, --betting-target CLI
provides:
  - get_win_candidates() method in RacePredictor
  - select_bets() win path with BetType.WIN generation
  - conformal_confidence_score as tertiary ranking signal
  - --betting-target in run_wf_validation.py
affects: [engine.py, race_predictor.py, run_wf_validation.py]

# Tech tracking
tech-stack:
  added: []
  patterns: [win-candidate-selection, symmetric-place-win-design, soft-ranking-signal]

key-files:
  created: []
  modified:
    - src/backtest/race_predictor.py
    - src/backtest/engine.py
    - scripts/run_wf_validation.py
    - tests/test_race_predictor.py
    - tests/test_backtest_engine.py

key-decisions:
  - "get_win_candidates() is symmetric to get_place_candidates() but simplified (no regime_params, no gate model)"
  - "conformal_confidence_score is tertiary sort key only, not hard filter (D-08)"
  - "win_gate_pass is logged but never used as filter (D-08)"
  - "select_bets() gets betting_target param, dispatches to win/place path"
  - "engine.py guards place_selection_reason merge for win mode"

patterns-established:
  - "Win candidate selection: edge>0 AND odds>=1.0, gate_score DESC, max 2 per race"
  - "Soft signal pattern: conformal_confidence_score as tertiary sort, never filter"

requirements-completed: [WIN-03, WIN-05]

# Metrics
duration: 19min
completed: 2026-05-04
---

# Phase 8 Plan 02: Win Candidate Selection Summary

get_win_candidates() + select_bets() win path + conformal_confidence_score soft ranking + --betting-target in run_wf_validation.py

## Performance

- **Duration:** 19 min
- **Started:** 2026-05-04T00:43:09Z
- **Completed:** 2026-05-04T01:02:32Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- get_win_candidates() filters by win_selection_edge>0 AND tanodds>=1.0, ranks by win_gate_score DESC with conformal_confidence_score as tertiary sort, returns max 2 candidates
- select_bets() accepts betting_target parameter and generates BetType.WIN Bets in kelly/flat modes
- run_wf_validation.py accepts --betting-target CLI flag (default=win) and passes to both BacktestEngine instances
- All 1134 tests pass (11 new, 0 regressions)

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): test get_win_candidates + select_bets win path** - `33dc37e` (test)
2. **Task 1 (GREEN): implement get_win_candidates + select_bets win path** - `cf0a537` (feat)
3. **Task 2: add --betting-target to run_wf_validation.py** - `a184789` (feat)

_Note: TDD RED/GREEN cycle for Task 1._

## Files Created/Modified
- `src/backtest/race_predictor.py` - Added get_win_candidates(), added betting_target param + win branch to select_bets()
- `src/backtest/engine.py` - Pass betting_target to select_bets(), guard place_selection_reason column merge for win mode
- `scripts/run_wf_validation.py` - Added argparse --betting-target, pass to both BacktestEngine constructions
- `tests/test_race_predictor.py` - 11 new tests (7 get_win_candidates + 4 select_bets win path)
- `tests/test_backtest_engine.py` - Fixed 3 existing tests to pass betting_target="place"

## Decisions Made
- get_win_candidates() uses no regime_params (simpler than get_place_candidates)
- conformal_confidence_score is tertiary sort key only, never hard filter (prevents zero candidates when uncalibrated)
- win_gate_pass is logged for debugging but never used as filter (D-08)
- select_bets() dispatches via betting_target param rather than creating separate method

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed place_selection_reason KeyError in win mode**
- **Found during:** Task 1 (engine.py regression after adding betting_target dispatch)
- **Issue:** candidate_df from get_win_candidates() lacks place_selection_reason column, causing KeyError in merge
- **Fix:** Guard the column merge with `if "place_selection_reason" in candidate_df.columns`
- **Files modified:** src/backtest/engine.py
- **Verification:** All 3 previously-passing engine integration tests pass again
- **Committed in:** cf0a537 (Task 1 GREEN commit)

**2. [Rule 1 - Bug] Fixed existing engine tests for new default betting_target=win**
- **Found during:** Task 1 (full test suite regression)
- **Issue:** 3 engine integration tests assumed default betting_target=place but engine now defaults to "win"
- **Fix:** Pass explicit betting_target="place" to BacktestEngine in tests that test place-mode behavior
- **Files modified:** tests/test_backtest_engine.py
- **Verification:** All tests pass
- **Committed in:** cf0a537 (Task 1 GREEN commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
- None beyond the deviations documented above

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Win backtest core is complete (both 08-01 and 08-02 done)
- Phase 9 (Win Reporting) can proceed: bet_history already contains win bet data, reporting pipeline needs single-win adaptation
- Phase 10 (Pipeline Performance) can proceed: depends on Phase 8 only

## Self-Check: PASSED

- [x] src/backtest/race_predictor.py -- FOUND
- [x] src/backtest/engine.py -- FOUND
- [x] scripts/run_wf_validation.py -- FOUND
- [x] tests/test_race_predictor.py -- FOUND
- [x] 33dc37e (test RED) -- FOUND
- [x] cf0a537 (feat GREEN) -- FOUND
- [x] a184789 (feat Task 2) -- FOUND
- [x] get_win_candidates method count: 1
- [x] betting_target in race_predictor.py: 2 occurrences
- [x] betting-target in run_wf_validation.py: 3 occurrences

---
*Phase: 08-win-backtest-core*
*Completed: 2026-05-04*
