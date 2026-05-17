---
phase: 29-etl-expansion
plan: 01
subsystem: database
tags: [etl, yaml, parquet, type-conversion, pk-fix]

# Dependency graph
requires:
  - phase: prior
    provides: existing etl_tables.yaml and etl.py
provides:
  - Correct PK definitions (kumi) for 10 odds table entries in etl_tables.yaml
  - Type conversion rules (int/odds10) for odds_sanren, odds_umaren, odds_sanrentan in etl.py
affects: [29-02, 29-03, phase-31, phase-32]

# Tech tracking
tech-stack:
  added: []
  patterns: [kumi-based PK for combination odds tables, odds10 type conversion pattern]

key-files:
  created: []
  modified:
    - config/etl_tables.yaml
    - src/db/etl.py

key-decisions:
  - "umatan tables also use kumi (same JRA-VAN encoding pattern as umaren/wide/sanren/sanrentan)"
  - "_head tables excluded from _TABLE_TYPE_RULES (no odds column)"

patterns-established:
  - "Combination odds tables use kumi as PK component (not individual umaban columns)"
  - "odds10 type conversion pattern applies to all new odds tables (varchar/10 -> float)"

requirements-completed: [ETL-01, ETL-02, ETL-03]

# Metrics
duration: 8min
completed: 2026-05-17
---

# Phase 29 Plan 01: ETL PK Fix & Type Rules Summary

**10 PK definitions corrected from umaban to kumi, 3 type conversion rules added for trio/quinella/trifecta odds tables**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-17T12:39:32Z
- **Completed:** 2026-05-17T12:47:05Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Fixed 10 PK entries in etl_tables.yaml (5 odds types x 2 variants n_/s_) from umaban1/2[/3] to kumi
- Added _TABLE_TYPE_RULES entries for odds_sanren, odds_umaren, odds_sanrentan with int and odds10 conversions
- All 1,526 existing tests pass with zero failures

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix PK definitions in etl_tables.yaml** - `0e9fd6c` (fix)
2. **Task 2: Add type conversion rules for 3 new odds tables in etl.py** - `28cba0f` (feat)

## Files Created/Modified
- `config/etl_tables.yaml` - PK columns for 10 odds table entries changed from umaban to kumi
- `src/db/etl.py` - Added 3 entries to _TABLE_TYPE_RULES dict

## Decisions Made
- Included odds_umatan tables in the fix (same JRA-VAN kumi encoding pattern as umaren/wide/sanren/sanrentan) -- plan suggested checking them and fixing if applicable
- Excluded _head tables from _TABLE_TYPE_RULES as specified (no odds column per D-07/RESEARCH Pattern 2)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ETL pipeline ready to extract odds_sanren, odds_umaren, odds_sanrentan, odds_umatan, odds_wide with correct PKs and types
- Plan 29-02 (DataRepository pattern) can proceed
- Plan 29-03 (full pipeline integration) can proceed after 29-02

## Self-Check: PASSED
- config/etl_tables.yaml: FOUND
- src/db/etl.py: FOUND
- 29-01-SUMMARY.md: FOUND
- 0e9fd6c (Task 1): FOUND
- 28cba0f (Task 2): FOUND

---
*Phase: 29-etl-expansion*
*Completed: 2026-05-17*
