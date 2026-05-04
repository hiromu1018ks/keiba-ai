---
phase: 10-pipeline-performance
plan: 02
subsystem: pipeline-perf
tags: [parquet, caching, pyinstrument, profiling, sha256]

# Dependency graph
requires:
  - phase: 10-pipeline-performance/01
    provides: Research context and design decisions for caching and profiling
provides:
  - Parquet feature cache with hybrid invalidation in FeatureEngine.build_all()
  - ProfileContext context manager for pyinstrument profiling
  - --profile CLI flag on run_backtest.py and run_wf_validation.py
affects: [training-pipeline, backtest-engine, wf-validation]

# Tech tracking
tech-stack:
  added: [pyinstrument]
  patterns: [hybrid-cache-invalidation, single-return-point-cache-write, lazy-import-profiling]

key-files:
  created:
    - src/utils/profiling.py
  modified:
    - src/features/feature_engine.py
    - scripts/run_backtest.py
    - scripts/run_wf_validation.py
    - tests/test_feature_engine.py

key-decisions:
  - "Single-file caching for build_all() output rather than per-module (per RESEARCH.md Q3)"
  - "Cache key via SHA-256 hash of input paths + date range + feature type (16 hex chars = 64 bits)"
  - "Hybrid invalidation: fast timestamp check, no content hash fallback needed for typical use"
  - "ProfileContext with lazy pyinstrument import and graceful ImportError degradation"

patterns-established:
  - "Single-return-point pattern for guaranteed cache write in build_all()"
  - "ProfileContext context manager wraps entire execution for profiling without overhead when disabled"

requirements-completed: [PERF-03, PERF-04]

# Metrics
duration: 7min
completed: 2026-05-04
---

# Phase 10 Plan 02: Pipeline Performance Summary

**Parquet feature cache with SHA-256 hybrid invalidation and pyinstrument profiling integration**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-04T05:00:02Z
- **Completed:** 2026-05-04T05:07:30Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- FeatureEngine.build_all() now caches output to Parquet with hybrid invalidation (timestamp check)
- Cache key derived from input Parquet paths + date range + feature type via SHA-256 (16 hex chars)
- Single-return-point pattern guarantees cache write at every exit path
- ProfileContext context manager created with lazy pyinstrument import and graceful degradation
- Both CLI scripts (run_backtest.py, run_wf_validation.py) have --profile flag
- 10 new cache tests added, all 40 feature engine tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Create ProfileContext utility and add --profile flag to both CLI scripts** - `19e1489` (feat)
2. **Task 2: Add Parquet feature cache with hybrid invalidation to FeatureEngine.build_all()** - `db7f3e9` (feat)

## Files Created/Modified
- `src/utils/profiling.py` - ProfileContext context manager with lazy pyinstrument import
- `src/features/feature_engine.py` - Added compute_cache_key, is_cache_valid functions and cache logic to build_all()
- `scripts/run_backtest.py` - Added --profile flag and ProfileContext wrapping
- `scripts/run_wf_validation.py` - Added --profile flag and ProfileContext wrapping
- `tests/test_feature_engine.py` - Added TestFeatureCache class with 10 test methods

## Decisions Made
- Single-file caching for build_all() output rather than per-module caching (per RESEARCH.md Q3 resolution). Per-module caching deferred as future optimization.
- Cache key uses SHA-256 hash truncated to 16 hex characters (64 bits), making collision impractical per threat model T-10-03.
- isinstance(store, ParquetStore) guard prevents mock objects from triggering cache code paths, which is the correct behavior for non-ParquetStore stores.
- ProfileContext only imports pyinstrument inside __enter__, avoiding ImportError when profiling is disabled.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Mock compatibility with isinstance check**
- **Found during:** Task 2 (cache tests)
- **Issue:** MagicMock objects fail isinstance(store, ParquetStore) check, preventing cache hit/miss tests
- **Fix:** Used MagicMock(spec=ParquetStore) in tests to pass isinstance check
- **Files modified:** tests/test_feature_engine.py
- **Verification:** All 10 cache tests pass including cache hit and miss scenarios
- **Committed in:** db7f3e9 (Task 2 commit)

**2. [Rule 2 - Missing Critical] Guard against non-ParquetStore store objects in cache path**
- **Found during:** Task 2 (cache tests)
- **Issue:** assert isinstance(store, ParquetStore) would crash on non-ParquetStore objects passed as store
- **Fix:** Changed assert to if isinstance() guard, skipping cache when store is not ParquetStore
- **Files modified:** src/features/feature_engine.py
- **Verification:** Tests with both mock and None store pass without errors
- **Committed in:** db7f3e9 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 missing critical)
**Impact on plan:** Both auto-fixes necessary for test correctness and robustness. No scope creep.

## Issues Encountered
None beyond the auto-fixes documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Feature cache is ready for use in TrainingPipelineV5 and BacktestEngine
- Profiling infrastructure is ready for identifying bottlenecks in backtest runs
- pyinstrument must be installed separately (`pip install pyinstrument`) to use --profile flag

---
*Phase: 10-pipeline-performance*
*Completed: 2026-05-04*

## Self-Check: PASSED

All files verified present:
- src/utils/profiling.py
- src/features/feature_engine.py
- scripts/run_backtest.py
- scripts/run_wf_validation.py
- tests/test_feature_engine.py
- .planning/phases/10-pipeline-performance/10-02-SUMMARY.md

All commits verified:
- 19e1489 (feat: ProfileContext + --profile flags)
- db7f3e9 (feat: Parquet feature cache)
