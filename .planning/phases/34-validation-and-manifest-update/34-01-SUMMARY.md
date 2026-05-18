---
phase: 34-validation-and-manifest-update
plan: 01
subsystem: models
tags: [feature-registration, rl-features, gpd-diagnostics, leakage-test]
dependency_graph:
  requires: [Phase 31 race-level features]
  provides: [rl_* features registered in all model FEATURE_COLS, GPD FEATURE_CATEGORY_MAP updated]
  affects: [all ML models, gpd_diagnostics, test_post_race_leakage]
tech_stack:
  added: []
  patterns: [TDD RED/GREEN for GPD map, FEATURE_COLS registration pattern]
key_files:
  created: []
  modified:
    - src/models/stage1_ability_model.py
    - src/models/two_stage_return_model.py
    - src/models/ev_correction_model.py
    - src/models/conformal_ev_model.py
    - src/models/market_model.py
    - src/models/place_ability_model.py
    - src/models/race_quality_screener.py
    - src/models/regime_detector.py
    - src/models/wide_two_stage_model.py
    - src/models/gpd_diagnostics.py
    - tests/test_post_race_leakage.py
decisions:
  - rl_* 6 features appended after rl_wide_harville_ratio (last MCF entry) in all 12 model FEATURE_COLS lists
  - GPD FEATURE_CATEGORY_MAP classifies all 6 rl_* as "market" (derived from pre-race tanodds)
  - test_ev_correction_odds_col_uses_pre_race_odds test data updated with rl_* columns (existing test broke due to new FEATURE_COLS entries)
metrics:
  duration_minutes: 4
  completed: "2026-05-18"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 11
  tests_added: 2
  tests_passing: 13
---

# Phase 34 Plan 01: rl_* Feature Registration Summary

6 rl_* race-level features (RLF-01~06) registered in all 12 model FEATURE_COLS lists and GPD FEATURE_CATEGORY_MAP, with POST_RACE leakage test verification.

## Changes Made

### Task 1: Add 6 rl_* features to all 9 model files (12 lists total)

Added the following 6 features to the end of each model's FEATURE_COLS list, after the existing MCF entries:
- `rl_log_odds_entropy` (RLF-01: Shannon entropy of implied probabilities)
- `rl_odds_dispersion` (RLF-02: std deviation of tanodds)
- `rl_top3_odds_gap` (RLF-03: odds gap between 1st and 3rd favorite)
- `rl_top1_odds` (RLF-04: 1st favorite's tanodds)
- `rl_favorite_rank_gap` (RLF-05: log odds ratio between 1st and 2nd favorite)
- `rl_n_horses` (RLF-06: number of runners)

Target 12 lists across 9 files:
1. `AbilityModel.FEATURE_COLS`
2. `WinTwoStageModel.FEATURE_COLS`
3. `PlaceTwoStageModel.HIT_FEATURE_COLS`
4. `PlaceTwoStageModel.RETURN_FEATURE_COLS`
5. `EVCorrectionModel.FEATURE_COLS`
6. `PlaceEVCorrectionModel.FEATURE_COLS`
7. `ConformalEVModel.FEATURE_COLS`
8. `MarketModel.FEATURE_COLS`
9. `PlaceAbilityModel.FEATURE_COLS`
10. `RaceQualityScreener.FEATURE_COLS`
11. `RegimeDetector.FEATURE_COLS`
12. `WideTwoStageModel.SHARED_FEATURE_COLS`

### Task 2: GPD FEATURE_CATEGORY_MAP + leakage test (TDD)

**RED phase:** Added 2 new tests to `TestMarketCrossFeatures`:
- `test_all_models_have_rl_features`: verifies all 12 model lists contain all 6 rl_* features
- `test_gpd_category_map_has_rl_features`: verifies GPD FEATURE_CATEGORY_MAP has all 6 rl_* mapped to "market"

**GREEN phase:** Added 6 rl_* entries to `FEATURE_CATEGORY_MAP` in `src/models/gpd_diagnostics.py` as "market" category.

**Auto-fix (Rule 3):** Updated `test_ev_correction_odds_col_uses_pre_race_odds` test data to include the 6 new rl_* columns. The test DataFrame was missing columns that are now in `EVCorrectionModel.FEATURE_COLS`, causing a KeyError.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed test data missing rl_* columns**
- **Found during:** Task 2 GREEN phase
- **Issue:** `test_ev_correction_odds_col_uses_pre_race_odds` failed with KeyError because its test DataFrame didn't include the 6 new rl_* columns that were added to `EVCorrectionModel.FEATURE_COLS` in Task 1
- **Fix:** Added 6 rl_* columns to the test DataFrame
- **Files modified:** tests/test_post_race_leakage.py
- **Commit:** 3b96812

## Verification Results

- All 12 model FEATURE_COLS lists contain all 6 rl_* features: PASS
- GPD FEATURE_CATEGORY_MAP has 6 new "market" entries for rl_*: PASS
- All 13 POST_RACE leakage tests pass: PASS
- Full test suite collects without errors (1627 tests): PASS

## Self-Check: PASSED

- All 11 modified files exist in git history
- 3 commits verified: 963a862, 079033c, 3b96812
- No accidental file deletions in any commit
