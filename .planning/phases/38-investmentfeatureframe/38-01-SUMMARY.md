---
phase: 38-investmentfeatureframe
plan: 01
subsystem: api
tags: [schema-registry, frozen-dataclass, leakage-detection, dual-mode, oof-safe]

# Dependency graph
requires:
  - phase: 37-ev-calibration-layers
    provides: OOFHealthValidator, POST_RACE_COLS, 3-layer leakage test pattern
provides:
  - InvestmentFeatureSpec frozen dataclass with 94 specs across 9 categories
  - FEATURE_SPECS dict for schema-driven dual-mode feature resolution
  - CATEGORY_ORDER tuple for stable column ordering
  - ALL_IF_COLUMNS ordered list for output column names
  - validate_no_post_race_leakage() for POST_RACE exclusion (IFF-05)
  - validate_oof_safe_sources() for in-sample-only source detection (D-13)
  - validate_schema_identity() for train/infer schema equivalence (IFF-03)
  - IN_SAMPLE_ONLY_COLS frozenset constant
affects: [38-02, 38-03, investment-feature-frame-builder, investment-cache]

# Tech tracking
tech-stack:
  added: []
  patterns: [frozen-dataclass-schema-registry, dual-mode-source-resolution, leakage-validation-layer]

key-files:
  created:
    - src/investment/__init__.py
    - src/investment/schema_registry.py
    - src/investment/leakage.py
    - tests/test_investment_schema.py
    - tests/test_investment_leakage.py
  modified: []

key-decisions:
  - "94 specs defined across 9 categories within D-05 column ranges"
  - "Derived features (if_ev_raw, if_logit_gap, etc.) use empty train/infer_sources tuples to signal builder computation"
  - "Optional features use float('nan') as default_value with if_*_missing indicator columns"
  - "if_ev_calibrated uses ev_win_corrected for train mode (Isotonic unavailable in OOF) and ev_win_calibrated for infer"
  - "if_p_win_final uses p_win_corrected for train mode (Benter blend unavailable in OOF) and p_win_final for infer"

patterns-established:
  - "Schema-driven dual mode: InvestmentFeatureSpec.train_sources/infer_sources for OOF-safe vs production resolution"
  - "Category-ordered output: CATEGORY_ORDER determines column order, ALL_IF_COLUMNS provides the ordered list"
  - "Required/optional split: required=True means fail-fast on source absence; optional means default_value + missing indicator"

requirements-completed: [IFF-04, IFF-05]

# Metrics
duration: 5min
completed: 2026-05-27
---

# Phase 38 Plan 01: InvestmentFeatureFrame Foundation Summary

**94-feature schema registry with frozen dataclass + 3-function leakage detection module for OOF-safe dual-mode investment features**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-27T10:02:11Z
- **Completed:** 2026-05-27T10:07:01Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- InvestmentFeatureSpec frozen dataclass (10 fields per D-16) with 94 specs across 9 categories
- All spec names use "if_" prefix, no overlap with POST_RACE_COLS
- 3 leakage validators: POST_RACE exclusion, OOF-safe source detection, schema identity assertion
- TDD cycle: 4 commits (2 RED + 2 GREEN), 24 tests passing

## Task Commits

Each task was committed atomically via TDD:

1. **Task 1 RED: InvestmentFeatureSpec tests** - `7be7712` (test)
2. **Task 1 GREEN: Schema registry implementation** - `0345ecf` (feat)
3. **Task 2 RED: Leakage detection tests** - `1b23056` (test)
4. **Task 2 GREEN: Leakage module implementation** - `fc7bb35` (feat)

_Note: TDD tasks have RED (test) + GREEN (feat) commits per plan tdd="true"_

## Files Created/Modified
- `src/investment/__init__.py` - Package init with public API exports (InvestmentFeatureSpec, FEATURE_SPECS, CATEGORY_ORDER, ALL_IF_COLUMNS, leakage validators)
- `src/investment/schema_registry.py` - InvestmentFeatureSpec frozen dataclass + FEATURE_SPECS dict (94 specs) + CATEGORY_ORDER + ALL_IF_COLUMNS
- `src/investment/leakage.py` - validate_no_post_race_leakage(), validate_oof_safe_sources(), validate_schema_identity(), IN_SAMPLE_ONLY_COLS
- `tests/test_investment_schema.py` - 15 tests for schema registry (frozen, categories, unique names, column counts, metadata)
- `tests/test_investment_leakage.py` - 9 tests for leakage detection (POST_RACE, OOF-safe, schema identity)

## Decisions Made
- if_ev_calibrated uses ev_win_corrected for train (OOF) mode since Isotonic calibration is not available within OOF folds
- if_p_win_final uses p_win_corrected for train mode as the most OOF-safe pre-Benter probability
- Derived features (if_ev_raw, if_logit_gap, if_edge_win, etc.) have empty train_sources/infer_sources tuples to indicate builder-side computation
- if_odds_band_id uses tanodds as source (band assignment happens in builder)
- Added if_odds_band_ev_rank as 6th odds_band spec to reach 6-spec minimum for D-05 range

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Schema registry and leakage detection ready for Plan 02 (feature_frame.py builder)
- FEATURE_SPECS provides the source resolution contract for build_frame(df, mode)
- leakage.py validators ready for builder output validation

---
*Phase: 38-investmentfeatureframe*
*Completed: 2026-05-27*
