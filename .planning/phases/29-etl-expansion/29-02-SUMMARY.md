---
phase: 29-etl-expansion
plan: 02
subsystem: database
tags: [parquet, data-access, repository-pattern, odds]

# Dependency graph
requires:
  - phase: 29-01
    provides: PK definitions + type rules for odds_sanren/umaren/sanrentan
provides:
  - DataRepository class with load_trio_odds, load_exacta_odds, load_trifecta_odds
affects: [31-race-level-aggregation, 32-market-cross-consistency]

# Tech tracking
tech-stack:
  added: []
  patterns: [repository-pattern, delegated-parquet-read]

key-files:
  created:
    - src/db/repository.py
    - tests/test_repository.py
  modified: []

key-decisions:
  - "ParquetStore injection pattern: optional constructor arg defaults to ParquetStore()"
  - "Reused _coerce_types and _date_filters from db.readers — no duplication"

patterns-established:
  - "DataRepository load methods: store.read(category, name, filters=_date_filters(start, end)) then _coerce_types(df)"

requirements-completed: [ETL-01, ETL-02, ETL-03]

# Metrics
duration: 8min
completed: 2026-05-17
---

# Phase 29 Plan 02: DataRepository Class Summary

**DataRepository with 3 delegated load methods (trio/exacta/trifecta odds) following the established load_wide_odds pattern from db.readers**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-17T12:49:52Z
- **Completed:** 2026-05-17T12:58:00Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- DataRepository class with optional ParquetStore injection (DI pattern)
- load_trio_odds reads "odds_sanren", load_exacta_odds reads "odds_umaren", load_trifecta_odds reads "odds_sanrentan"
- 11 mock-based tests covering init, correct delegation, date filters, and return types
- Full test suite 1537 passed with 0 regressions

## Task Commits

Each task was committed atomically (TDD RED/GREEN):

1. **Task 1 (RED): Failing tests for DataRepository** - `91b73f4` (test)
2. **Task 1 (GREEN): Implement DataRepository with 3 loaders** - `1a26c04` (feat)

## Files Created/Modified
- `src/db/repository.py` - DataRepository class with 3 load methods, ParquetStore injection, full type annotations
- `tests/test_repository.py` - 11 mock-based tests (TestInit, TestLoadTrioOdds, TestLoadExactaOdds, TestLoadTrifectaOdds)

## Decisions Made
- ParquetStore injection: `__init__(store: ParquetStore | None = None)` enables both DI for tests and default construction for production use
- Reused `_coerce_types` and `_date_filters` from `db.readers` instead of duplicating logic
- Followed existing `load_wide_odds` pattern exactly for consistency with codebase conventions

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- ruff N806 (variable naming) on `MockPS` in test — fixed to lowercase `mock_ps`
- mypy errors are pre-existing project-wide issues (pandas-stubs not installed, py.typed missing) — not introduced by this plan

## Next Phase Readiness
- DataRepository is ready for Phase 31 (Race-Level Aggregation) and Phase 32 (Market Cross-Consistency) to consume trio/exacta/trifecta odds
- Plan 29-03 (coverage verification) is next in this phase

## Self-Check: PASSED

- src/db/repository.py: FOUND
- tests/test_repository.py: FOUND
- .planning/phases/29-etl-expansion/29-02-SUMMARY.md: FOUND
- Commit 91b73f4 (test/RED): FOUND
- Commit 1a26c04 (feat/GREEN): FOUND

---
*Phase: 29-etl-expansion*
*Completed: 2026-05-17*
