---
phase: 10-pipeline-performance
plan: 01
subsystem: backtest-engine
tags: [performance, vectorization, pandas, groupby]
dependency_graph:
  requires: []
  provides: [vectorized-payout-maps, groupby-dict-lookups]
  affects: [src/backtest/engine.py, tests/test_backtest_engine.py]
tech_stack:
  added: []
  patterns: [melt+groupby vectorization, str.slice vectorized kumi parsing, set_index+items, itertuples, groupby dict preprocessing]
key_files:
  created: []
  modified:
    - src/backtest/engine.py
    - tests/test_backtest_engine.py
decisions:
  - melt+groupby for build_payout_map (D-04)
  - str.len/str.slice vectorized kumi parsing for build_wide_payout_map (D-02)
  - set_index+items for final_odds_map (D-06)
  - itertuples for top3, diag_logger, _generate_bets (D-01, D-03)
  - build_race_groups() helper with str-key + logging for groupby dicts (D-07, D-08)
  - groupby dict .get() returns None for missing groups (RacePredictor handles None)
metrics:
  duration: 553s
  completed: 2026-05-04
  tasks_completed: 2
  files_modified: 2
  tests_added: 4
  tests_total: 36
  iterrows_eliminated: 7
---

# Phase 10 Plan 01: Vectorize Payout Maps + Groupby Dict Lookups Summary

Vectorized all 7 iterrows() calls in engine.py with pandas melt/groupby/str-ops/set_index/itertuples, and replaced 5 per-race DataFrame filterings with O(1) groupby dict lookups.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Vectorize payout map functions + odds maps + diag loops | b43555a | src/backtest/engine.py, tests/test_backtest_engine.py |
| 2 | build_race_groups() helper + groupby dict O(1) lookups | 1c96a50 | src/backtest/engine.py |

## Changes Summary

### Task 1: Vectorize all iterrows() calls

**build_payout_map()** -- Replaced iterrows() with melt + groupby. Two separate melts for maban and pay columns, combined by aligned rows, dropna, dedup by idxmax on (race_id, umaban) groups.

**build_wide_payout_map()** -- Replaced iterrows() + _parse_kumi() with vectorized str.len()/str.slice() operations. Handles kumi lengths 2, 3, 4, 5+ with length-3 ambiguous split resolved by checking if first two digits <= 18.

**final_odds_map** -- Replaced iterrows() with set_index(["race_id", "umaban"]) + .items() comprehension.

**top3 extraction** -- Replaced iterrows() on 3-row nsmallest result with itertuples(). nsmallest(3, "kakuteijyuni") preserved as-is.

**diag_logger loops (2x)** -- Replaced iterrows() with itertuples(index=False), all hr["col"] replaced with getattr(hr, "col", default), hr.to_dict() replaced with hr._asdict().

**_generate_bets()** -- Replaced iterrows() on candidates with itertuples(index=False).

**Tests added:** TestVectorizedPayoutMaps with 4 methods:
- test_build_payout_map_vectorized_matches_original
- test_build_wide_payout_map_vectorized_kumi_formats
- test_build_payout_map_keeps_max_per_key
- test_final_odds_map_vectorized

### Task 2: build_race_groups() + groupby dict lookups

New module-level helper `build_race_groups()` with str-key conversion, empty group logging, and memory usage logging. Called 5 times for feat_df, hist_df_all, jockey_df_all, trainer_df_all, jt_df_all.

Race loop now uses dict.get(race_id) returning None for missing groups (RacePredictor handles None via `is not None` checks). race_id cast to str at loop top for type consistency with str-keyed dicts.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] groupby dict .get() fallback changed from pd.DataFrame() to None**
- **Found during:** Task 2
- **Issue:** pd.DataFrame() fallback has no columns, causing KeyError when RacePredictor tries to merge on "race_id"
- **Fix:** Changed to return None (RacePredictor.predict() already handles None via `if hist_features is not None:` checks)
- **Files modified:** src/backtest/engine.py
- **Commit:** 1c96a50

**2. [Rule 2 - Missing functionality] final_win_odds_map not in codebase**
- **Found during:** Task 1
- **Issue:** Plan step 5 references final_win_odds_map but it does not exist in current engine.py
- **Fix:** Skipped final_win_odds_map creation -- not needed for current codebase. Only 7 iterrows() found (not 8)
- **Files modified:** None (no change needed)

## Verification Results

- iterrows() count in engine.py: **0**
- melt() calls: **4** (2 in build_payout_map, 2 in build_wide_payout_map)
- str.len()/str.slice() calls: **14** in build_wide_payout_map
- set_index in final_odds_map: **confirmed**
- nsmallest(3, "kakuteijyuni"): **preserved**
- itertuples() calls: **4** (top3 + 2 diag_logger + _generate_bets)
- build_race_groups() calls: **5** in run()
- All 36 tests pass: **confirmed**

## Known Stubs

None.

## Threat Flags

None. No new security-relevant surface introduced.

## Self-Check: PASSED

- FOUND: src/backtest/engine.py
- FOUND: tests/test_backtest_engine.py
- FOUND: .planning/phases/10-pipeline-performance/10-01-SUMMARY.md
- FOUND: commit b43555a (Task 1)
- FOUND: commit 1c96a50 (Task 2)
