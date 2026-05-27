---
phase: 38-investmentfeatureframe
plan: 03
subsystem: testing
tags: [integration-tests, leakage-audit, e2e-pipeline, tdd-verification]

# Dependency graph
requires:
  - phase: 38-investmentfeatureframe/plan-01
    provides: InvestmentFeatureSpec, FEATURE_SPECS, leakage validators
  - phase: 38-investmentfeatureframe/plan-02
    provides: InvestmentFeatureFrameBuilder, manifest, cache
provides:
  - End-to-end integration test suite validating full InvestmentFeatureFrame pipeline
  - Leakage audit across all 94 specs (train_sources, infer_sources, spec names)
  - Required/optional behavior verification with fail-fast and graceful degradation
  - Column count per category validation against D-05 ranges
  - Schema determinism and train/infer identity verification
affects: [38-phase-completion, investment-feature-frame-consumers]

# Tech tracking
tech-stack:
  added: []
  patterns: [e2e-pipeline-test, spec-wide-leakage-audit, category-range-validation]

key-files:
  created:
    - tests/test_investment_integration.py
  modified: []

key-decisions:
  - "Integration tests use multi-race DataFrames (4 races, 5 horses each) for groupby operation validity"
  - "Leakage audit covers 5 dimensions: train_sources OOF-safe, spec names vs POST_RACE, train_sources vs POST_RACE, infer_sources vs POST_RACE, leakage_class validation"

patterns-established:
  - "Integration test pattern: build_frame(train) -> build_frame(infer) -> validate_schema_identity -> generate_investment_manifest"
  - "Leakage audit pattern: validate_oof_safe_sources(FEATURE_SPECS) + per-spec source-level POST_RACE check"

requirements-completed: [IFF-01, IFF-02, IFF-03, IFF-04, IFF-05, IFF-06, IFF-07, VAL-01]

# Metrics
duration: 8min
completed: 2026-05-27
---

# Phase 38 Plan 03: InvestmentFeatureFrame Integration Tests Summary

**18 end-to-end integration tests covering full pipeline (train/infer/manifest), 5-dimension leakage audit across all 94 specs, required/optional behavior, and schema determinism**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-27T10:24:39Z
- **Completed:** 2026-05-27T10:32:34Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments
- 18 new integration tests in test_investment_integration.py, all passing
- Full suite of 85 investment tests (24 schema/leakage + 43 manifest/cache/feature_frame + 18 integration) passing
- 5-dimension leakage audit validates all 94 specs across OOF-safe, POST_RACE, and leakage_class
- End-to-end pipeline test validates train -> infer -> schema identity -> manifest generation
- No regression in full test suite (2012 passed, 1 pre-existing failure in test_backtest_engine.py)

## Task Commits

Each task was committed atomically:

1. **Task 1: Schema registry + leakage test verification** - Verified (tests existed from Plan 01, 24 tests passing)
2. **Task 2: Feature frame + manifest + cache test verification** - Verified (tests existed from Plan 02, 43 tests passing)
3. **Task 3: End-to-end integration test + full suite verification** - `8ab9d2b` (test)

## Files Created/Modified
- `tests/test_investment_integration.py` - 18 integration tests: TestEndToEnd (4), TestLeakageAudit (5), TestRequiredOptionalBehavior (3), TestColumnCountPerCategory (2), TestDeterminismAndSchema (4)

## Decisions Made
- Used multi-race DataFrames (4 races, 5 horses each = 20 entries) in integration tests to ensure groupby-based derived features (if_edge_rank_in_race, if_field_ev_dispersion, etc.) work correctly
- Added 5 separate leakage audit tests for comprehensive coverage: OOF-safe sources, spec names vs POST_RACE, train_sources vs POST_RACE, infer_sources vs POST_RACE, and leakage_class validation
- Reused helper pattern from test_investment_feature_frame.py for _make_train_source_df/_make_infer_source_df

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 38 test coverage complete: 85 investment tests across 6 test files
- All IFF-01~07 + VAL-01 requirements verified by automated tests
- InvestmentFeatureFrame ready for pipeline integration (Phase 39+)

---
*Phase: 38-investmentfeatureframe*
*Completed: 2026-05-27*
