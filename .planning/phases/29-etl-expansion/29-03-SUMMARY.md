---
phase: 29-etl-expansion
plan: 03
subsystem: etl
tags: [parquet, coverage-verification, data-quality, etl]

# Dependency graph
requires:
  - phase: 29-etl-expansion
    provides: "ParquetStore.exists/read API for coverage checks"
provides:
  - "_verify_coverage function for post-ETL data quality checks"
  - "Coverage logging: row count, year range, max missing rate per table"
  - "Warning thresholds: missing years and >30% missing rate"
affects: [run_etl, etl, data-quality]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Post-ETL coverage verification pattern"]

key-files:
  created:
    - tests/test_etl_coverage.py
  modified:
    - scripts/run_etl.py

key-decisions:
  - "Used TYPE_CHECKING guard for ParquetStore import to avoid E402 at module level"
  - "Coverage check runs only in full mode, using start/end year from CLI args"

patterns-established:
  - "Post-ETL verification: read extracted Parquet, log coverage metrics, warn on gaps"

requirements-completed: [ETL-04]

# Metrics
duration: 10min
completed: 2026-05-17
---

# Phase 29 Plan 03: Post-ETL Coverage Verification Summary

**_verify_coverage function validates ETL output with row counts, year coverage (start_year-end_year), and missing-rate thresholds per table**

## Performance

- **Duration:** 10 min
- **Started:** 2026-05-17T13:01:21Z
- **Completed:** 2026-05-17T13:11:04Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- Added `_verify_coverage()` to run_etl.py: checks row count, year coverage from race_date, max missing rate per table
- Coverage warnings for missing years and >30% missing rate; graceful skip for nonexistent files
- 5 mock-based tests covering all behavioral cases (full coverage, missing years, high missing rate, nonexistent, empty)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add coverage verification to run_etl.py and create tests** - `064148e` (feat)

## Files Created/Modified
- `scripts/run_etl.py` - Added `_verify_coverage()` function + integration in `main()` for full mode
- `tests/test_etl_coverage.py` - NEW: 5 mock-based tests for coverage verification

## Decisions Made
- Used `TYPE_CHECKING` guard for `ParquetStore` type hint to avoid E402 lint error (module-level import after sys.path manipulation)
- Coverage check runs only in full mode, deriving start/end year from `args.start[:4]`/`args.end[:4]`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 29 complete (all 3 plans executed)
- Coverage verification is ready for use when ETL runs with `--mode full`
- Next: Phase 30 (Residual IC) or Phase 31 (depends on Phase 29 trio odds data)

---
*Phase: 29-etl-expansion*
*Completed: 2026-05-17*

## Self-Check: PASSED

- FOUND: scripts/run_etl.py
- FOUND: tests/test_etl_coverage.py
- FOUND: .planning/phases/29-etl-expansion/29-03-SUMMARY.md
- FOUND: commit 064148e
