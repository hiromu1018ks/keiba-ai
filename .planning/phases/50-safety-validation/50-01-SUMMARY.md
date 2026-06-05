---
phase: "50-safety-validation"
plan: "01"
subsystem: validation
tags: [ci-safety, feature-routing-audit, post-race-leakage, surface-aware-nan, track-condition, tdd]
dependency_graph:
  requires: [48-01, 49-02]
  provides: [track-condition-routing-ci, track-condition-post-race-ci, surface-aware-nan-ci, nan-diagnostic-script]
  affects: [tests/test_feature_routing_audit.py, tests/test_post_race_leakage.py, tests/test_track_condition_routing.py, tests/test_track_condition_nan.py, scripts/validate_track_condition_nan.py]
tech_stack:
  added: []
  patterns: [surgical-routing-verification, surface-aware-denominator, 3-tier-nan-verdict, cause-separation-reporting]
key_files:
  created:
    - tests/test_track_condition_routing.py
    - tests/test_track_condition_nan.py
    - scripts/validate_track_condition_nan.py
  modified:
    - tests/test_feature_routing_audit.py
    - tests/test_post_race_leakage.py
decisions:
  - Verified Phase 48/49 surgical routing: 4 excluded models have 0 TC features, 7 included have all 23
  - Surface-aware NaN verification: dirt features NaN on turf, turf features NaN on dirt, cross-surface available on both
  - POST_RACE CI: 23 TC features not in POST_RACE_COLS, all registered in included models, raw values excluded
  - NaN diagnostic script uses surface-aware denominator with 3-tier verdict (PASS/WARN/FAIL)
  - NaN cause separation distinguishes raw data missing from derived processing NaN
metrics:
  duration: 17m
  completed: "2026-06-05"
---

# Phase 50 Plan 01: Track Condition Feature Safety CI Tests Summary

CI safety validation of all 23 track condition features through Feature Routing Audit (4 excluded, 7 included), surface-aware NaN verification, POST_RACE 3-layer CI, and WF Fold0 NaN diagnostic script. 17 new CI tests + 1 diagnostic script, all passing.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Extend Feature Routing Audit CI tests + POST_RACE 3-layer CI for track condition features | 2c30f2e | test_feature_routing_audit.py, test_post_race_leakage.py |
| 2 | Create surface-aware NaN CI test + WF Fold0 NaN diagnostic script | 320aa0a | test_track_condition_routing.py, test_track_condition_nan.py, validate_track_condition_nan.py |

## Changes Made

### tests/test_feature_routing_audit.py
- Added `TestTrackConditionRouting` class with 4 tests:
  - `test_excluded_models_no_track_condition_features`: 4 excluded models (MarketModel, RaceQualityScreener, RegimeDetector, ConformalEVModel) have zero intersection with ALL_TRACK_CONDITION_COLS
  - `test_included_models_have_track_condition_features`: 7 included models (AbilityModel, WinTwoStageModel, PlaceTwoStageModel HIT/RETURN, EVCorrectionModel, PlaceEVCorrectionModel, PlaceAbilityModel) have all 23 features
  - `test_wide_two_stage_has_track_condition_features`: WideTwoStageModel.SHARED_FEATURE_COLS has all 23
  - `test_audit_still_passes`: run_feature_audit() returns overall_status="PASS"
- Added imports: ConformalEVModel, EVCorrectionModel, PlaceEVCorrectionModel, RegimeDetector, PlaceAbilityModel, AbilityModel, PlaceTwoStageModel, WinTwoStageModel, track condition feature lists
- Added `ALL_TRACK_CONDITION_COLS` union list (23 features)

### tests/test_post_race_leakage.py
- Added `TestTrackConditionPostRace` class with 3 tests:
  - `test_track_condition_not_post_race`: All 23 TC features NOT in POST_RACE_COLS
  - `test_track_condition_features_registered_in_models`: All 23 TC features present in at least one model's FEATURE_COLS (not orphaned)
  - `test_raw_track_values_not_in_feature_cols`: Raw dirt_moisture and turf_cushion are NOT in any model's FEATURE_COLS

### tests/test_track_condition_routing.py (new)
- `TestSurfaceAwareNaN` class with 5 surface-aware NaN CI tests:
  - `test_dirt_features_nan_on_turf_rows`: dirt_moisture_x_kyakusitu, dirt_moisture_x_barrier_pos, dirt_moisture_high_flag, dirt_moisture_dry_flag are all NaN on turf rows
  - `test_turf_features_nan_on_dirt_rows`: turf_cushion_track_relative, turf_cushion_track_zscore, turf_cushion_x_kyakusitu are all NaN on dirt rows
  - `test_cross_surface_features_available`: track_front_bias_score and kickback_risk_score are NOT NaN on both turf and dirt rows
  - `test_sire_x_cushion_band_nan_on_dirt`: sire_x_cushion_band is NaN on dirt rows
  - `test_expected_pace_class_available_both_surfaces`: expected_pace_class is non-NaN on both surfaces

### tests/test_track_condition_nan.py (new)
- `TestWFold0NaNRate` class with 5 NaN rate threshold logic tests:
  - `test_nan_rate_thresholds_applied_correctly`: PASS (< 30%), WARN (30-50%), FAIL (>= 50%) verdicts
  - `test_nan_rate_warn_threshold`: Turf feature with 35% NaN rate on turf-only denominator -> WARN
  - `test_nan_rate_fail_threshold`: Dirt feature with 55% NaN rate on dirt-only denominator -> FAIL
  - `test_surface_aware_denominator`: Turf feature uses turf rows only (100) not total (200)
  - `test_cross_surface_denominator_uses_all_rows`: Cross-surface features use total rows

### scripts/validate_track_condition_nan.py (new)
- CLI diagnostic script for VLD-03 NaN rate measurement
- Arguments: --features-path, --start, --end, --output
- Surface-aware NaN rate computation per D-10:
  - dirt_* features: denominator = dirt rows only
  - turf_* / cushion_* features: denominator = turf rows only
  - cross-surface features: denominator = all rows
- 3-tier verdict per D-11: < 30% PASS, 30-50% WARN, >= 50% FAIL
- NaN cause separation per D-12: raw_cause_pct vs derived_cause_pct
- FAIL detail report per D-13: prints raw/derived cause breakdown
- Training start date NOT modified per D-14 (report only)
- Output JSON: fold0_period, total/turf/dirt rows, per-feature nan_rate/verdict/cause_separation, overall_verdict

## Verification Results

- `python -m pytest tests/test_feature_routing_audit.py tests/test_post_race_leakage.py tests/test_track_condition_routing.py tests/test_track_condition_nan.py -v`: 42/42 passed
- `python -m pytest tests/test_feature_routing_audit.py tests/test_post_race_leakage.py tests/test_track_condition_routing.py tests/test_track_condition_nan.py tests/test_track_condition_features.py tests/test_domain.py -v`: 123/123 passed
- `ruff check` on all new/modified files: all checks passed
- `python scripts/validate_track_condition_nan.py --help`: shows usage with 4 arguments
- Full test suite (2472 passed, 14 failed): all failures pre-existing, none introduced by Phase 50

## Deviations from Plan

None - plan executed exactly as written.

## Deferred Issues

Pre-existing test failures (out of scope, not introduced by this plan):
- `test_backtest_engine.py::test_observed_true_on_all_groupby` (un-observed groupby in investment/feature_frame.py)
- `test_feature_engine.py::test_mcf_odds_load_uses_yyyymmdd_dates` (FakeRepository missing load_horse_track_aptitude)
- 12 other pre-existing failures in bloodline, component_attribution, ev_isotonic, gpd_diagnostics, phase46, place_ability, win_feature_analysis, win_profit_selector tests

## Self-Check: PASSED

- [x] tests/test_feature_routing_audit.py exists with TestTrackConditionRouting (4 tests)
- [x] tests/test_post_race_leakage.py exists with TestTrackConditionPostRace (3 tests)
- [x] tests/test_track_condition_routing.py exists with TestSurfaceAwareNaN (5 tests)
- [x] tests/test_track_condition_nan.py exists with TestWFold0NaNRate (5 tests)
- [x] scripts/validate_track_condition_nan.py exists with --help output
- [x] Commit 2c30f2e exists: test(50-01): extend Feature Routing Audit + POST_RACE CI
- [x] Commit 320aa0a exists: test(50-01): add surface-aware NaN CI tests + WF Fold0 NaN diagnostic script
