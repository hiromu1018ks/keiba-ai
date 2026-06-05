---
phase: "47"
plan: "02"
subsystem: db
tags: [etl, data-repository, track-conditions, ci]
dependency_graph:
  requires: [47-01]
  provides: [DataRepository.load_track_conditions]
  affects: [src/db/repository.py, tests/test_etl_type_conversion.py]
tech_stack:
  added: []
  patterns: [parquet-date-filter, coerce_types-gate]
key_files:
  created:
    - tests/test_repository_track_conditions.py
  modified:
    - src/db/repository.py
    - tests/test_etl_type_conversion.py
    - src/features/track_condition_data.py
decisions:
  - D-10: load_track_conditions follows existing load_* pattern with exists-check gate
  - D-11: POST_RACE_COLS CI test ensures track condition columns are safe for ML features
metrics:
  duration: 12m
  completed: "2026-06-04"
---

# Phase 47 Plan 02: DataRepository Track Condition Integration Summary

DataRepository.load_track_conditions() provides date-filtered access to track_conditions.parquet with exists-check gate, plus CI verification that dirt_moisture/turf_cushion are not in POST_RACE_COLS.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add DataRepository.load_track_conditions + POST_RACE CI test | 0cd90b1 | repository.py, test_etl_type_conversion.py, test_repository_track_conditions.py, track_condition_data.py |
| 2 | Integration verification | (verified) | -- |

## Changes Made

### src/db/repository.py
- Added `load_track_conditions(self, start: str, end: str) -> pd.DataFrame` method
- Follows existing `load_*` pattern: `exists()` check gate, `date_filters()` for range, `coerce_types()` for type safety
- Returns empty `pd.DataFrame()` when parquet file does not exist

### tests/test_etl_type_conversion.py
- Added `TestPostRaceCols` class with 2 tests (ETL-04)
- `test_dirt_moisture_not_in_post_race_cols`: asserts dirt_moisture is not in POST_RACE_COLS
- `test_turf_cushion_not_in_post_race_cols`: asserts turf_cushion is not in POST_RACE_COLS

### tests/test_repository_track_conditions.py (new)
- 4 unit tests with `unittest.mock`:
  - `test_returns_dataframe_when_parquet_exists`: verifies data returned when file present
  - `test_passes_date_filters_to_read`: verifies date_filters(start, end) passed correctly
  - `test_returns_empty_dataframe_when_parquet_missing`: verifies empty DataFrame on missing file
  - `test_coerce_types_applied`: verifies race_date string-to-datetime conversion

### src/features/track_condition_data.py
- Added `observed=True` to `groupby()` call at line 96 (Rule 1 fix for pre-existing lint violation)

## Integration Verification Results

- ParquetStore + DataRepository round-trip: 20,949 rows for date range 20200101-20261231
- Column schema: `{race_id, race_date, dirt_moisture, turf_cushion}` -- confirmed
- Non-NaN values: dirt_moisture=11,013, turf_cushion=9,936
- Date range filtering: 3,455 rows for 2024 only (correct subset of 20,949)
- ruff: All checks passed
- All targeted tests: 24/24 passed (test_etl_type_conversion + test_repository_track_conditions)
- Core tests: 50/50 passed (including test_domain)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added observed=True to track_condition_data.py groupby**
- **Found during:** Full test suite run (Task 2)
- **Issue:** `groupby(["race_id", "race_date"], sort=False)` missing `observed=True` flag, flagged by existing CI test `test_observed_true_on_all_groupby`
- **Fix:** Added `observed=True` to the groupby call
- **Files modified:** src/features/track_condition_data.py
- **Commit:** 0cd90b1

## Pre-existing Test Failures (Out of Scope)

4 test failures exist on the clean tree (confirmed by stashing changes and running):
1. `test_observed_true_on_all_groupby` -- 12 violations in `src/investment/feature_frame.py` (pre-existing, not introduced by this plan)
2. `test_blood_keito_cd_from_sire` -- bloodline keito code lookup returns 'unknown' instead of 'SS'
3. `test_generate_ev_oof_uses_walk_forward_split` -- TimeSeriesSplit not found in training_pipeline module
4. `test_race_predictor_uses_profit_selector_candidate_set` -- candidate set assertion mismatch

These are documented in CLAUDE.md known issues or prior phases. None are caused by 47-02 changes.

## Self-Check: PASSED

- [x] `src/db/repository.py` exists and contains `load_track_conditions` method
- [x] `tests/test_repository_track_conditions.py` exists with 4 tests
- [x] `tests/test_etl_type_conversion.py` contains `TestPostRaceCols` class
- [x] Commit `0cd90b1` exists in git log
