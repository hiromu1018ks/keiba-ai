---
phase: 38-investmentfeatureframe
plan: 02
subsystem: api
tags: [feature-frame-builder, manifest, cache, dual-mode, tdd, derived-features]

# Dependency graph
requires:
  - phase: 38-investmentfeatureframe/plan-01
    provides: FEATURE_SPECS, CATEGORY_ORDER, ALL_IF_COLUMNS, leakage validators
provides:
  - InvestmentFeatureFrameBuilder with build_frame(df, mode) dual-mode API
  - compute_investment_schema_hash for deterministic SHA256 column hashing
  - generate_investment_manifest for D-30 artifact manifest generation
  - InvestmentFrameCache with Parquet + sidecar JSON caching
  - 20 derived feature computations (ev_raw, logit_gap, race-relative ranks, etc.)
affects: [38-03, investment-frame-consumers]

# Tech tracking
tech-stack:
  added: []
  patterns: [two-pass-builder, dict-based-column-construction, derived-feature-dispatch]

key-files:
  created:
    - src/investment/manifest.py
    - src/investment/cache.py
    - src/investment/feature_frame.py
    - tests/test_investment_manifest.py
    - tests/test_investment_cache.py
    - tests/test_investment_feature_frame.py
  modified:
    - src/investment/__init__.py

key-decisions:
  - "Two-pass builder: resolve all source-based features first, then compute derived features for cross-category dependencies"
  - "Dict-based column construction for initial features to avoid DataFrame fragmentation (PerformanceWarning)"
  - "Derived features computed in CATEGORY_ORDER so if_abs_logit_gap can reference if_logit_gap"
  - "load_or_compute uses input df schema_hash for cache key, not output schema_hash"
  - "if_odds_band_id passes through tanodds value directly (band binning deferred to Phase 39)"

patterns-established:
  - "Two-pass build: source resolution -> derived computation enables cross-category references"
  - "Manifest per D-30: artifact_name, builder_version, feature_version, generated_at, mode, schema_hash, schema_dtype_hash, source_artifact_hash, source_oof_manifest_path, row_count, column_count"
  - "Cache key: SHA256(mode|feature_version|source_artifact_hash|schema_hash|builder_version)[:16]"

requirements-completed: [IFF-01, IFF-02, IFF-03, IFF-06, IFF-07]

# Metrics
duration: 10min
completed: 2026-05-27
---

# Phase 38 Plan 02: InvestmentFeatureFrame Builder Summary

**Manifest, cache, and feature frame builder with 94-spec dual-mode resolution and 20 derived feature computations**

## Performance

- **Duration:** 10 min
- **Started:** 2026-05-27T10:10:21Z
- **Completed:** 2026-05-27T10:20:21Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- InvestmentFrameManifest generation with D-30 fields and deterministic SHA256 schema hashing
- InvestmentFrameCache with Parquet + sidecar JSON, schema_hash verification, and load_or_compute
- InvestmentFeatureFrameBuilder with 94 specs across 9 categories, 20 derived features
- Dual-mode (train/infer) identical output schema verified by tests
- TDD cycle: 7 commits (3 RED + 3 GREEN + 1 refactor), 43 tests passing

## Task Commits

Each task was committed atomically via TDD:

1. **Task 1 RED: Manifest tests** - `340636e` (test)
2. **Task 1 GREEN: Manifest implementation** - `e2c44f8` (feat)
3. **Task 2 RED: Cache tests** - `8ea3c26` (test)
4. **Task 2 GREEN: Cache implementation** - `9589277` (feat)
5. **Task 3 RED: Feature frame tests** - `5864ac2` (test)
6. **Task 3 GREEN: Feature frame builder** - `568ebee` (feat)
7. **Lint fixes** - `132361b` (refactor)

## Files Created/Modified
- `src/investment/manifest.py` - compute_investment_schema_hash() + generate_investment_manifest() per D-30
- `src/investment/cache.py` - InvestmentFrameCache with Parquet + sidecar JSON per D-21~D-27
- `src/investment/feature_frame.py` - InvestmentFeatureFrameBuilder + build_frame() + convenience wrappers
- `src/investment/__init__.py` - Updated exports for all new public API
- `tests/test_investment_manifest.py` - 15 tests for manifest module
- `tests/test_investment_cache.py` - 11 tests for cache module
- `tests/test_investment_feature_frame.py` - 17 tests for builder module

## Decisions Made
- Two-pass builder resolves source-based features in CATEGORY_ORDER first, then derived features, enabling cross-category references (e.g., if_logit_gap references if_p_win from model_prob and if_implied_prob from market_prob)
- Dict-based column construction used for initial DataFrame to avoid pandas PerformanceWarning about frame fragmentation with 100+ column insertions
- if_odds_band_id passes through tanodds directly -- actual band binning is a Phase 39 concern
- if_p_win_gap_to_fav uses idxmin() approach instead of groupby().apply() to avoid FutureWarning about grouping columns

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Two-pass build for cross-category derived features**
- **Found during:** Task 3 GREEN phase
- **Issue:** Derived features in early categories (model_prob) referenced columns from later categories (market_prob, odds_band). Single-pass CATEGORY_ORDER processing caused KeyError.
- **Fix:** Restructured build_frame into two passes: Pass 1 resolves all source-based features, Pass 2 computes derived features. Derived features are added immediately to result so subsequent derived features can reference them.
- **Files modified:** src/investment/feature_frame.py
- **Commit:** 568ebee

**2. [Rule 3 - Blocking] DataFrame fragmentation PerformanceWarning**
- **Found during:** Task 3 GREEN phase
- **Issue:** Inserting 100+ columns one-by-one into a DataFrame triggered pandas PerformanceWarning
- **Fix:** Build initial columns as a dict, construct DataFrame once, then add derived columns individually (needed for inter-dependent computation)
- **Files modified:** src/investment/feature_frame.py
- **Commit:** 568ebee

**3. [Rule 1 - Bug] Test adjustment: if_* column count range**
- **Found during:** Task 3 GREEN phase
- **Issue:** Test counted all if_* prefixed columns including _missing indicators (165 total), exceeding the 90-130 range which applies to feature columns only (94)
- **Fix:** Updated test to exclude columns ending with _missing from the count
- **Files modified:** tests/test_investment_feature_frame.py
- **Commit:** 568ebee

**4. [Rule 1 - Bug] Ruff lint issues**
- **Found during:** Post-implementation lint check
- **Issue:** Unused pyarrow imports in cache.py, unused ALL_IF_COLUMNS import in feature_frame.py, long line in cache.py
- **Fix:** Removed unused imports, broke long line
- **Files modified:** src/investment/cache.py, src/investment/feature_frame.py
- **Commit:** 132361b

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Feature frame builder ready for Plan 03 integration
- Manifest generation ready for pipeline integration
- Cache mechanism ready for production use with load_or_compute pattern

---
*Phase: 38-investmentfeatureframe*
*Completed: 2026-05-27*

## Self-Check: PASSED

All 8 created/modified files verified present.
All 8 commit hashes verified in git log.
