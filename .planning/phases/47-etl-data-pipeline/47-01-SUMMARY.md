---
phase: 47
plan: 01
subsystem: etl-data-pipeline
tags: [etl, track-conditions, parquet, csv-conversion]
dependency_graph:
  requires: [races.parquet, dirt-moisture-csv, turf-cushion-csv]
  provides: [track_conditions.parquet]
  affects: [src/features/track_condition_data.py, scripts/precompute_track_condition.py]
tech_stack:
  added: [pandas-csv-parsing, parquet-io]
  patterns: [thin-orchestrator, race-level-aggregation, physical-validation]
key_files:
  created:
    - src/features/track_condition_data.py
    - tests/test_track_condition_data.py
    - scripts/precompute_track_condition.py
  modified: []
decisions:
  - D-04: entry_id(18-digit) = race_id(first 16) + umaban(last 2)
  - D-05: ValueError on multiple distinct non-NaN values per race_id
  - D-06: NaN/non-NaN mix resolves to non-NaN
  - D-07: Cross-validation against races.parquet logs only (no raise)
  - D-08: NaN values preserved as-is (no statistical imputation)
  - D-09: Physical outliers NaN-ified (dirt: 0<x<100, turf: x>0)
metrics:
  duration_seconds: 294
  completed_date: 2026-06-04
  task_count: 2
  file_count: 3
  test_count: 16
  output_rows: 23259
  runtime_seconds: 2.6
---

# Phase 47 Plan 01: Track Condition ETL Pipeline Summary

CSV-to-Parquet conversion pipeline for dirt moisture (189K rows) and turf cushion (133K rows) data, producing a single race-level track_conditions.parquet (23,259 races).

## Results

| Metric | Value |
|--------|-------|
| Input rows (dirt) | 189,334 entry-level |
| Input rows (turf) | 133,672 entry-level |
| Output rows | 23,259 race-level |
| Races with dirt_moisture | 13,323 |
| Races with turf_cushion | 9,936 |
| Date range | 2018-07-28 to 2026-05-31 |
| Runtime | 2.6s |
| Tests | 16/16 passed |
| Ruff | All checks passed |

## Commits

| Hash | Message |
|------|---------|
| 3246fd0 | feat(47): add track condition data module + tests (CSV parsing, aggregation, validation) |
| ba59c81 | feat(47): add precompute_track_condition.py thin orchestrator script |

## Files Created

| File | Description |
|------|-------------|
| `src/features/track_condition_data.py` | Core logic: parse CSV, aggregate to race-level, validate physical ranges, end-to-end conversion |
| `tests/test_track_condition_data.py` | 16 tests covering all functions and edge cases |
| `scripts/precompute_track_condition.py` | Thin orchestrator: glob-resolve CSVs, call conversion, log summary |
| `data/raw/track_conditions.parquet` | Output (gitignored): 23,259 rows, columns: race_id, race_date, dirt_moisture, turf_cushion |

## Public API

```python
from features.track_condition_data import (
    parse_track_condition_csv,    # CSV -> entry-level DataFrame
    aggregate_to_race_level,      # entry-level -> race-level with conflict detection
    validate_physical_range,      # physical outlier -> NaN replacement
    convert_track_conditions,     # end-to-end: CSV -> ParquetStore.write()
)
```

## Cross-Validation Results

- 485 race_ids in track_conditions not found in races.parquet (expected: 2018 races before ETL range, 2026 races after ETL range)
- 46,319 races.parquet races missing from track_conditions (expected: many races have neither measurement)

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check

- [x] `src/features/track_condition_data.py` exists
- [x] `tests/test_track_condition_data.py` exists
- [x] `scripts/precompute_track_condition.py` exists
- [x] `data/raw/track_conditions.parquet` exists (23,259 rows)
- [x] Commit 3246fd0 exists (task 1)
- [x] Commit ba59c81 exists (task 2)
- [x] All 16 tests passing
- [x] Ruff check passed
