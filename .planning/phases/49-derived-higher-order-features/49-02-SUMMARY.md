---
phase: "49-derived-higher-order-features"
plan: "02"
subsystem: features
tags: [track-condition, derived-features, race-level, season-deviation, anomaly, pace-bias, surgical-routing, tdd]
dependency_graph:
  requires: [49-01, 48-01]
  provides: [TRACK_DERIVED_COLS, RACE_CONDITION_COLS, compute_race_condition_features, _compute_track_month_stats, track_month_stats_on_SubmodelSet]
  affects: [src/features/track_condition_features.py, src/pipelines/training_pipeline.py, src/backtest/race_predictor.py, src/domain/models.py, src/models/stage1_ability_model.py, src/models/two_stage_return_model.py, src/models/ev_correction_model.py, src/models/place_ability_model.py, src/models/wide_two_stage_model.py, config/settings.yaml]
tech_stack:
  added: []
  patterns: [linear-interpolation-bias-score, track-month-stats-lookup, zscore-season-deviation, race-level-aggregation, NaN-safe-product, surgical-routing]
key_files:
  created: []
  modified:
    - src/features/track_condition_features.py
    - src/pipelines/training_pipeline.py
    - src/backtest/race_predictor.py
    - src/domain/models.py
    - src/models/stage1_ability_model.py
    - src/models/two_stage_return_model.py
    - src/models/ev_correction_model.py
    - src/models/place_ability_model.py
    - src/models/wide_two_stage_model.py
    - config/settings.yaml
    - tests/test_track_condition_features.py
decisions:
  - D-08/D-09: Linear interpolation bias/kickback scores with NaN propagation
  - D-10: expected_pace_class 3-level numeric (0=slow, 1=neutral, 2=fast)
  - D-18: trackcd x month zscore for season deviation
  - D-15: |deviation| > 2 anomaly flags with NaN propagation
  - D-16: 3 numeric interaction products (cushion_x_distance, moisture_x_weight, cushion_x_age)
  - D-17: surface_condition_transition (current - previous race condition)
  - D-11/D-12/D-13: Race-level condition match score/max/ratio from T3 aptitude rates
  - D-14: race_field_front_bias = front_runner_ratio x track_front_bias_score
  - D-23/D-26: Surgical routing to 6 included models, 4 excluded (safety net)
metrics:
  duration: 17m
  completed: "2026-06-05"
---

# Phase 49 Plan 02: Derived and Higher-Order Track Condition Features Summary

15 derived/higher-order track condition features (T3-04: 2, T4-01: 3, T4-03: 2, T4-04: 4, T4-02: 4) with pipeline integration, track_month_stats on SubmodelSet, and surgical FEATURE_COLS routing to 6 included + 4 excluded models.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Extend track_condition_features.py with T4-01/T4-03/T4-04 row features + T3-04 season deviation + T4-02 race features + tests | 1bbc612 | track_condition_features.py, test_track_condition_features.py, settings.yaml |
| 2 | Pipeline integration + surgical FEATURE_COLS routing | e056f1e | training_pipeline.py, race_predictor.py, models.py, 5 model files, test_track_condition_features.py |

## Changes Made

### src/features/track_condition_features.py
- TRACK_DERIVED_COLS: 11 new feature names (T4-01 + T3-04 + T4-03 + T4-04)
- RACE_CONDITION_COLS: 4 new race-level feature names (T4-02)
- `_compute_track_month_stats(df)`: trackcd x month mean/std statistics for T3-04
- `compute_track_condition_features(df, *, track_stats, track_month_stats)`:
  - T4-01: track_front_bias_score (linear interpolation), kickback_risk_score (inverse), expected_pace_class (3-level)
  - T3-04: cushion_season_deviation, moisture_season_deviation (zscore using track_month_stats)
  - T4-03: cushion_anomaly_flag, moisture_extreme_flag (|deviation| > 2)
  - T4-04: cushion_x_distance, moisture_x_weight, cushion_x_age, surface_condition_transition
- `compute_race_condition_features(df)`: T4-02 race-level aggregation
  - race_condition_match_score: mean of matching aptitude rate per race
  - race_condition_match_max: max of matching aptitude rate per race
  - race_condition_match_ratio: count(qualified) / valid entries per race
  - race_field_front_bias: front_runner_ratio x track_front_bias_score

### src/pipelines/training_pipeline.py
- `_train_submodel()`: computes _track_month_stats via _compute_track_month_stats(df)
- Passes track_month_stats to compute_track_condition_features()
- Calls compute_race_condition_features(df) after compute_track_condition_features()
- Saves track_month_stats on SubmodelSet

### src/backtest/race_predictor.py
- `predict()`: reads track_month_stats from SubmodelSet via getattr
- Passes track_month_stats to compute_track_condition_features()
- Calls compute_race_condition_features(df) after compute_track_condition_features()

### src/domain/models.py
- Added `track_month_stats: dict | None = None` field to SubmodelSet dataclass

### Surgical FEATURE_COLS routing (6 included models)
- AbilityModel.FEATURE_COLS: +15 features
- WinTwoStageModel.FEATURE_COLS: +15 features
- PlaceTwoStageModel (HIT/RETURN/FEATURE): +15 features each
- EVCorrectionModel.FEATURE_COLS: +15 features
- PlaceEVCorrectionModel.FEATURE_COLS: +15 features
- PlaceAbilityModel.FEATURE_COLS: +15 features
- WideTwoStageModel.SHARED_FEATURE_COLS: +15 features

### Excluded models (D-26 safety net)
- MarketModel, RaceQualityScreener, RegimeDetector, ConformalEVModel: no new features

### config/settings.yaml
- Added track_condition section: dirt_wet_threshold (12.0), dirt_dry_threshold (3.0), turf_hard_threshold (10.0), turf_soft_threshold (8.0), hit_rate_threshold (0.3), min_starts (3)

### tests/test_track_condition_features.py
- 55 total tests (22 existing + 33 new)
- T4-01: 11 tests (bias, kickback, pace_class for dirt/turf, NaN propagation)
- T3-04: 4 tests (track_month_stats, season_deviation, std_zero, no_stats)
- T4-03: 3 tests (anomaly triggered, normal, NaN propagation)
- T4-04: 9 tests (3 products + transition for dirt/turf, NaN propagation)
- T4-02: 4 tests (match_score, match_max, match_ratio, field_front_bias)
- Constants: 2 tests (TRACK_DERIVED_COLS count, RACE_CONDITION_COLS count)
- Updated surgical routing tests to verify TRACK_DERIVED_COLS + RACE_CONDITION_COLS

## Verification Results

- `python -m pytest tests/test_track_condition_features.py -v`: 55/55 passed
- `python -m pytest tests/test_domain.py -v`: 26/26 passed
- `python -m pytest tests/test_interaction_features.py -v`: 32/32 passed
- `python -c "from features.track_condition_features import TRACK_DERIVED_COLS, RACE_CONDITION_COLS; print(len(TRACK_DERIVED_COLS), len(RACE_CONDITION_COLS))"`: 11 4
- `python -m ruff check` on all modified files: all checks passed (1 pre-existing import sort warning in training_pipeline.py)

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- [x] `src/features/track_condition_features.py` exports TRACK_DERIVED_COLS (11), RACE_CONDITION_COLS (4), compute_race_condition_features, _compute_track_month_stats
- [x] `src/pipelines/training_pipeline.py` calls _compute_track_month_stats and compute_race_condition_features
- [x] `src/backtest/race_predictor.py` calls compute_race_condition_features with track_month_stats from SubmodelSet
- [x] `src/domain/models.py` SubmodelSet has track_month_stats field
- [x] 6 included models have TRACK_DERIVED_COLS + RACE_CONDITION_COLS in FEATURE_COLS
- [x] 4 excluded models do NOT have new features
- [x] `config/settings.yaml` has track_condition section
- [x] Both commits exist: 1bbc612, e056f1e
