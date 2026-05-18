---
phase: 33-gain-per-depth-diagnostic
plan: 01
subsystem: models/diagnostics
tags: [gpd, diagnostics, lightgbm, feature-importance, depth-analysis]
dependency_graph:
  requires: [TrainedModelsV5, lgb.Booster, FEATURE_COLS from all model classes]
  provides: [compute_gpd_diagnostics, console_summary, FEATURE_CATEGORY_MAP, gpd_report.json]
  affects: [src/models/gpd_diagnostics.py, tests/test_gpd_diagnostics.py]
tech_stack:
  added: []
  patterns: [function-based diagnostic module, duck-type booster detection, depth-gain aggregation]
key_files:
  created:
    - src/models/gpd_diagnostics.py
    - tests/test_gpd_diagnostics.py
  modified: []
decisions:
  - Duck-type _is_booster() instead of strict isinstance for test compatibility
  - Unknown features in trees_to_dataframe() default to "fundamental" category
  - FEATURE_CATEGORY_MAP has 41 market + 119 fundamental + 19 categorical = 179 total
  - Difficulty_score, field_size, weight_diff_from_mean classified as fundamental (race/horse context)
metrics:
  duration: 12m
  completed: 2026-05-18
  tasks_completed: 1
  tests_added: 19
  tests_passed: 19
  files_created: 2
  loc_added: 821
---

# Phase 33 Plan 01: Gain per Depth Diagnostic Module Summary

Core GPD diagnostic module: FEATURE_CATEGORY_MAP (179 features), Booster extraction, depth-by-category gain computation, MDR/FAD metrics, JSON output, console_summary(), and 19 comprehensive tests.

## What Was Built

### src/models/gpd_diagnostics.py
- **FEATURE_CATEGORY_MAP**: Explicit dict mapping 179 unique features to market/fundamental/categorical categories, validated against all 9 model FEATURE_COLS lists
- **_is_booster()**: Duck-type detection for lgb.Booster compatibility (isinstance + trees_to_dataframe/feature_importance check)
- **_extract_boosters()**: Iterates TrainedModelsV5.submodels to extract all LightGBM Boosters with tier labels (primary/detailed), handles StackedEnsemble unwrapping, optional models, PlaceAbilityModel LGBMClassifier access
- **_compute_depth_gains()**: Calls trees_to_dataframe(), filters leaf nodes (Pitfall 3), fills NaN split_gain (Pitfall 7), maps features via FEATURE_CATEGORY_MAP, groups gain by (depth, category)
- **_compute_market_dominance_ratio()**: MDR = Market_share(depth 1-3) - Market_share(depth 4+), returns None when total gain is zero
- **_compute_fundamental_activation_depth()**: FAD = min depth where Fundamental > Market, returns None when Market always dominates
- **compute_gpd_diagnostics()**: Orchestrator iterating all Boosters, computing metrics, writing JSON to data/gpd/gpd_report.json
- **console_summary()**: Formatted logging output with tier labels, MDR, FAD, tree stats, shallow depth gain breakdown (no PASS/FAIL per D-12)

### tests/test_gpd_diagnostics.py
- Test 1 (3 cases): FEATURE_CATEGORY_MAP completeness -- all 179 features registered, valid categories, no extras
- Test 2 (4 cases): Booster extraction -- primary tier, detailed tier, StackedEnsemble unwrapping, None optionals
- Test 3 (4 cases): Depth-gain computation -- grouping accuracy, leaf node exclusion, NaN handling, summary statistics
- Test 4 (3 cases): MDR computation -- positive/negative/None edge cases
- Test 5 (2 cases): FAD computation -- correct depth detection, None when Market always dominates
- Test 6 (3 cases): Full pipeline -- result structure, JSON output, console_summary()

## Commits

| Commit | Type | Description |
|--------|------|-------------|
| 3ec3239 | test(33-01) | TDD RED: 19 failing tests for GPD diagnostics |
| be25bd8 | feat(33-01) | GREEN: implementation with 179-feature category map, all 19 tests pass |

## TDD Gate Compliance

- [x] RED gate: `test(33-01)` commit with 19 failing tests
- [x] GREEN gate: `feat(33-01)` commit with implementation, all 19 tests pass
- [ ] REFACTOR gate: Not needed -- clean implementation, no refactoring required

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Duck-type _is_booster() instead of strict isinstance**
- **Found during:** Task 1 GREEN phase
- **Issue:** MagicMock objects in tests fail isinstance(obj, lgb.Booster), blocking Booster extraction tests
- **Fix:** Added _is_booster() function with duck-type fallback (hasattr trees_to_dataframe + feature_importance), matching the plan's intent of checking isinstance first
- **Files modified:** src/models/gpd_diagnostics.py
- **Commit:** be25bd8

**2. [Rule 3 - Blocking] Test depth-gain tests passed DataFrame instead of Booster**
- **Found during:** Task 1 GREEN phase
- **Issue:** _compute_depth_gains() expects a Booster (calls .trees_to_dataframe()), but tests passed raw DataFrames
- **Fix:** Wrapped test DataFrames in mock Booster objects via _make_mock_booster() helper
- **Files modified:** tests/test_gpd_diagnostics.py
- **Commit:** be25bd8

### Planned Decisions

None - plan executed exactly as written.

## Verification Results

```
python -m pytest tests/test_gpd_diagnostics.py -v: 19 passed, 0 failed
FEATURE_CATEGORY_MAP size: 179 features (41 market + 119 fundamental + 19 categorical)
ruff check: All checks passed
mypy: No new errors (pre-existing import-untyped stub issues only)
```

## Threat Flags

No new threat surface introduced. All operations are read-only analysis on model Booster objects.

## Self-Check: PASSED

- FOUND: src/models/gpd_diagnostics.py
- FOUND: tests/test_gpd_diagnostics.py
- FOUND: .planning/phases/33-gain-per-depth-diagnostic/33-01-SUMMARY.md
- 5c6f242: docs(33-01) summary commit
- be25bd8: feat(33-01) GREEN commit
- 3ec3239: test(33-01) RED commit
