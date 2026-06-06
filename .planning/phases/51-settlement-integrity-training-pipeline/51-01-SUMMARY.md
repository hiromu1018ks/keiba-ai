---
phase: 51
plan: 01
subsystem: betting
tags: [payout-maps, pure-functions, extraction, D-09, D-12, D-10]
dependency_graph:
  requires: []
  provides: [build_win_payout_map, build_payout_map, build_wide_payout_map, build_place_payout_map]
  affects: [src/backtest/engine.py]
tech_stack:
  added: []
  patterns: [pure-function-module, verbatim-extraction, alias-pattern]
key_files:
  created:
    - src/betting/payout_maps.py
    - tests/test_payout_maps.py
  modified:
    - src/backtest/engine.py
decisions:
  - D-09 shared pure functions for payout map construction
  - D-10 payout maps return odds multipliers (pay/100), not raw yen
  - D-12 no I/O or EveryDB2 access in helper module
metrics:
  duration: 9m 23s
  completed: 2026-06-06
  tasks_total: 1
  tasks_completed: 1
  files_created: 2
  files_modified: 1
  tests_added: 20
  tests_passed: 20
---

# Phase 51 Plan 01: Extract Payout Maps Pure Functions Summary

Payout map construction functions extracted from BacktestEngine into shared `src/betting/payout_maps.py` module with zero I/O dependencies, enabling identical payout logic for both BT and PT pipelines.

## Tasks Completed

| Task | Name | Status | Commit |
|------|------|--------|--------|
| 1 | Create payout_maps.py with extracted pure functions and tests | Done | 47e5ac1 |

## Key Changes

### src/betting/payout_maps.py (NEW)
- `build_payout_map(payouts_df) -> dict[(str, int), float]` -- place/fuku payout builder using melt + groupby
- `build_win_payout_map(payouts_df) -> dict[(str, int), float]` -- win payout builder using dropna + dict comprehension
- `build_wide_payout_map(payouts_df) -> dict[(str, int, int), float]` -- wide payout builder with kumi string parsing (lengths 2-5)
- `build_place_payout_map` alias for CONTEXT.md reference clarity
- Zero imports from db/, backtest/, paper_trading/ (D-12 compliant)
- Only imports: `numpy`, `pandas`

### src/backtest/engine.py (MODIFIED)
- Added `from betting.payout_maps import build_payout_map, build_wide_payout_map, build_win_payout_map`
- Deleted ~200 lines of local function definitions (lines 163-358 of original)
- All callers unchanged (identical signatures preserved)

### tests/test_payout_maps.py (NEW)
- `TestBuildWinPayoutMap`: 5 tests (empty DF, single payout, NaN skip, multiple races, all-NaN)
- `TestBuildPayoutMap`: 6 tests (empty DF, multiple positions, NaN skip, same-key max, multiple races, alias verification)
- `TestBuildWidePayoutMap`: 9 tests (empty DF, valid pairs, length-3/4 kumi split, float kumi, multiple pairs, NaN skip, lo/hi ordering)
- All 20 tests use helper functions `_make_place_df` / `_make_wide_df` to construct complete-column DataFrames

## Verification Results

| Check | Result |
|-------|--------|
| `python -m pytest tests/test_payout_maps.py -v` | 20/20 PASSED |
| `python -m pytest tests/test_backtest_engine.py -v` | 79/80 PASSED (1 pre-existing failure in unrelated `test_observed_true_on_all_groupby`) |
| `from betting.payout_maps import ...` | OK |
| `grep -c "def build_payout_map" src/backtest/engine.py` | 0 (functions removed) |
| `grep "from betting.payout_maps import" src/backtest/engine.py` | Match found (line 24) |

## Decisions Made

1. **Verbatim extraction** -- Copied function implementations exactly from engine.py to preserve behavior. No logic changes.
2. **Test data completeness** -- Test DataFrames include all expected columns (e.g., `payfukusyoumaban1-5`, `paywidekumi1-7`) with NaN for unused slots, matching real Parquet schema.
3. **build_place_payout_map alias** -- Simple assignment `build_place_payout_map = build_payout_map` for CONTEXT.md reference resolution.

## Deviations from Plan

None -- plan executed exactly as written.

## Self-Check: PASSED

- `src/betting/payout_maps.py` exists
- `tests/test_payout_maps.py` exists
- `src/backtest/engine.py` imports from `betting.payout_maps`
- Commit `47e5ac1` exists in git log
