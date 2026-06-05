---
phase: "49-derived-higher-order-features"
plan: "01"
subsystem: features
tags: [track-aptitude, pit-safe, precompute-parquet, tdd, t3]
dependency_graph:
  requires: [48-01]
  provides: [precompute_track_aptitude, APTITUDE_COLS, horse_track_aptitude.parquet, load_horse_track_aptitude]
  affects: [src/features/horse_track_aptitude.py, scripts/precompute_track_aptitude.py, src/db/readers.py, src/db/repository.py, src/features/feature_engine.py]
tech_stack:
  added: []
  patterns: [expanding-window-shift1, precompute-parquet, pit-safe-cumulative, condition-classification]
key_files:
  created:
    - src/features/horse_track_aptitude.py
    - scripts/precompute_track_aptitude.py
    - tests/test_horse_track_aptitude.py
  modified:
    - src/db/readers.py
    - src/db/repository.py
    - src/features/feature_engine.py
decisions:
  - D-01: PIT-safe expanding window + shift(1) pattern following horse_career_stats.py
  - D-03: Condition thresholds dirt_wet>=12%, dirt_dry<3%, turf_hard>=10, turf_soft<8
  - D-04: Hit=kakuteijyuni<=3, excluded kakuteijyuni<=0 or NaN
  - D-05: horse_condition_type with min_starts=3, hit_rate_threshold=0.3
  - D-06: versatility = mean(wet_rate, dry_rate) * (1 - |wet_rate - dry_rate|)
  - D-07: 14-column output schema with 2 keys + 4 rates + 4 counts + 1 versatility + 1 type + 2 prev
metrics:
  duration: 11m
  completed: "2026-06-05"
---

# Phase 49 Plan 01: Horse Track Condition Aptitude Precompute Summary

PIT-safe horse track condition aptitude precompute with 14-column parquet output, expanding window + shift(1) pattern, condition classification, and FeatureEngine build_all() integration.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create horse_track_aptitude.py precompute module with PIT-safe logic and tests | 0544f94 | horse_track_aptitude.py, test_horse_track_aptitude.py |
| 2 | Create precompute script + repository loader + FeatureEngine merge + integration tests | 80ae5fc | precompute_track_aptitude.py, readers.py, repository.py, feature_engine.py, test_horse_track_aptitude.py |

## Changes Made

### src/features/horse_track_aptitude.py (new)
- `precompute_track_aptitude(entries_df, races_df, track_conditions_df) -> pd.DataFrame`: 14-column PIT-safe precompute
- `APTITUDE_COLS`: 14 output column names constant
- Condition classification: dirt_wet (moisture>=12), dirt_dry (moisture<3), turf_hard (cushion>=10), turf_soft (cushion<8)
- Hit definition: kakuteijyuni <= 3, excluded <= 0 or NaN
- `horse_condition_type`: wet_good/dry_good/balanced/unknown (min_starts=3, threshold=0.3)
- `horse_condition_versatility`: mean(wet_rate, dry_rate) * (1 - |wet_rate - dry_rate|)
- `prev_dirt_moisture` / `prev_turf_cushion`: shift(1) on sorted per-horse group

### scripts/precompute_track_aptitude.py (new)
- CLI precompute script following precompute_career_stats.py pattern
- Loads entries.parquet, races.parquet, track_conditions.parquet
- Writes to data/raw/horse_track_aptitude.parquet
- Prints debut rate verification

### src/db/readers.py
- `load_horse_track_aptitude(store: ParquetStore) -> pd.DataFrame`: standalone loader with exists-check

### src/db/repository.py
- `DataRepository.load_horse_track_aptitude(start, end) -> pd.DataFrame`: date-filtered loader

### src/features/feature_engine.py
- T3 merge block in `build_all()` after track_conditions merge
- Left join on race_id + kettonum
- Guarded by store, date range, and empty DataFrame checks
- Wrapped in TimingContext("build_all/horse_track_aptitude")

### tests/test_horse_track_aptitude.py (new)
- 19 tests: 16 unit + 3 integration
- Unit: column count, PIT-safety, hit definition, exclusion, classification (4 types), versatility, prev values, APTITUDE_COLS, empty input, turf classification, multi-horse isolation
- Integration: readers missing parquet, repository missing, FeatureEngine merge row count

## Verification Results

- `python -m pytest tests/test_horse_track_aptitude.py -v`: 19/19 passed
- `ruff check src/features/horse_track_aptitude.py src/db/readers.py src/db/repository.py src/features/feature_engine.py`: All checks passed
- `python -c "from features.horse_track_aptitude import APTITUDE_COLS; print(len(APTITUDE_COLS))"`: prints 14
- `python -m pytest tests/test_track_condition_features.py -v`: 22/22 passed (no regression)
- `python -m pytest tests/test_domain.py -v`: 26/26 passed (no regression)

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- [x] `src/features/horse_track_aptitude.py` exists and exports `precompute_track_aptitude` and `APTITUDE_COLS`
- [x] `scripts/precompute_track_aptitude.py` exists and follows precompute_career_stats.py pattern
- [x] `src/db/readers.py` contains `load_horse_track_aptitude` function
- [x] `src/db/repository.py` contains `load_horse_track_aptitude` method
- [x] `src/features/feature_engine.py` contains `horse_track_aptitude` merge in build_all()
- [x] `tests/test_horse_track_aptitude.py` exists with 19 tests
- [x] Both commits exist: 0544f94, 80ae5fc
