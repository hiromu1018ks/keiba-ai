---
phase: "48-core-edge-features"
plan: "01"
subsystem: features
tags: [track-conditions, interaction-features, surgical-routing, tdd]
dependency_graph:
  requires: [47-01, 47-02]
  provides: [compute_track_condition_features, TRACK_CONDITION_COLS, track_stats_on_SubmodelSet]
  affects: [src/features/track_condition_features.py, src/features/feature_engine.py, src/pipelines/training_pipeline.py, src/backtest/race_predictor.py, src/domain/models.py, src/models/stage1_ability_model.py, src/models/two_stage_return_model.py, src/models/ev_correction_model.py, src/models/place_ability_model.py, src/models/wide_two_stage_model.py]
tech_stack:
  added: []
  patterns: [parquet-merge, track_stats-lookup, NaN-safe-product, category-interaction, surgical-routing]
key_files:
  created:
    - src/features/track_condition_features.py
    - tests/test_track_condition_features.py
  modified:
    - src/features/feature_engine.py
    - src/pipelines/training_pipeline.py
    - src/backtest/race_predictor.py
    - src/domain/models.py
    - src/models/stage1_ability_model.py
    - src/models/two_stage_return_model.py
    - src/models/ev_correction_model.py
    - src/models/place_ability_model.py
    - src/models/wide_two_stage_model.py
decisions:
  - D-07: track_condition_features.py as dedicated module with compute_track_condition_features() + TRACK_CONDITION_COLS
  - D-10: turf_cushion_track_relative uses training-period-only trackcd-level mean (no lookahead)
  - D-12: Fixed 5-bin cushion boundaries [0,7,8,9,10,inf] (very_soft/soft/standard/firm/very_firm)
  - D-04: Surgical routing: 8 models included, 4 excluded (MarketModel, RaceQualityScreener, RegimeDetector, ConformalEVModel)
  - track_stats stored on SubmodelSet for inference-time reuse (not recomputed from single race)
metrics:
  duration: 15m
  completed: "2026-06-04"
---

# Phase 48 Plan 01: Track Condition T1/T2 Features Summary

8 track condition interaction features (T1: 3, T2: 5) implemented in track_condition_features.py with NaN-safe surface-aware design, integrated into FeatureEngine/TrainingPipeline/RacePredictor, and surgically routed to 6 model FEATURE_COLS.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create track_condition_features.py module with compute function and tests | 789495f | track_condition_features.py, test_track_condition_features.py |
| 2 | Integrate track conditions into FeatureEngine.build_all() and pipeline insertion points | 2e907d2 | feature_engine.py, training_pipeline.py, race_predictor.py, models.py |
| 3 | Surgical FEATURE_COLS registration for target models | 2a9b161 | stage1_ability_model.py, two_stage_return_model.py, ev_correction_model.py, place_ability_model.py, wide_two_stage_model.py, test_track_condition_features.py |

## Changes Made

### src/features/track_condition_features.py (new)
- `compute_track_condition_features(df, *, track_stats=None) -> pd.DataFrame`: 8 features with NaN-safe, surface-aware computation
- `_compute_track_stats(df) -> dict[str, dict[str, float]]`: trackcd-level mean/std from training data
- `TRACK_CONDITION_COLS`: 8 feature names constant
- T1-01: `dirt_moisture_x_kyakusitu` -- numeric product, NaN-safe via `.where()`
- T1-02: `turf_cushion_track_relative` / `turf_cushion_track_zscore` -- track_stats lookup, lookahead-free
- T2-01: `dirt_moisture_x_barrier_pos` + `dirt_moisture_high_flag` (>12) + `dirt_moisture_dry_flag` (<3)
- T2-02: `turf_cushion_x_kyakusitu` -- numeric product, NaN-safe
- T2-03: `sire_x_cushion_band` -- fixed 5-bin category interaction

### src/features/feature_engine.py
- Added track_conditions raw value merge in `build_all()` after bloodline features
- Uses `DataRepository(store).load_track_conditions(start, end)` with date range from race_date
- Follows existing BloodlineFeatures(store) pattern

### src/pipelines/training_pipeline.py
- `_train_submodel()`: computes `_track_stats` via `_compute_track_stats(df)` and calls `compute_track_condition_features(df, track_stats=_track_stats)`
- Insertion point: after HorseHistoryFeatures, before interaction_features (per D-09)
- Saves `_track_stats` on SubmodelSet for inference-time access

### src/backtest/race_predictor.py
- `predict()`: reads `track_stats` from SubmodelSet via `getattr(submodel, "track_stats", None)`
- Calls `compute_track_condition_features()` before `compute_interaction_features()`

### src/domain/models.py
- Added `track_stats: dict | None = None` field to SubmodelSet dataclass

### Model FEATURE_COLS (surgical routing)
- AbilityModel.FEATURE_COLS: +8 features
- WinTwoStageModel.FEATURE_COLS: +8 features
- PlaceTwoStageModel.HIT_FEATURE_COLS: +8 features
- PlaceTwoStageModel.RETURN_FEATURE_COLS: +8 features
- EVCorrectionModel.FEATURE_COLS: +8 features
- PlaceEVCorrectionModel.FEATURE_COLS: +8 features
- PlaceAbilityModel.FEATURE_COLS: +8 features
- WideTwoStageModel.SHARED_FEATURE_COLS: +8 features

### tests/test_track_condition_features.py (new)
- 22 tests: per-feature, NaN propagation, missing columns, constant count, input immutability, _compute_track_stats, surgical routing (included + excluded)

## Verification Results

- `python -m pytest tests/test_track_condition_features.py -v`: 22/22 passed
- `python -m pytest tests/test_interaction_features.py -v`: 32/32 passed
- `python -m pytest tests/test_domain.py -v`: 26/26 passed
- `ruff check src/features/track_condition_features.py`: All checks passed
- `python -c "from features.track_condition_features import TRACK_CONDITION_COLS; print(len(TRACK_CONDITION_COLS))"`: prints 8

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- [x] `src/features/track_condition_features.py` exists and exports `compute_track_condition_features` and `TRACK_CONDITION_COLS`
- [x] `tests/test_track_condition_features.py` exists with 22 tests
- [x] `src/features/feature_engine.py` contains `load_track_conditions` in build_all()
- [x] `src/pipelines/training_pipeline.py` contains `compute_track_condition_features` call
- [x] `src/backtest/race_predictor.py` contains `compute_track_condition_features` call
- [x] `src/domain/models.py` SubmodelSet has `track_stats` field
- [x] All 3 commits exist: 789495f, 2e907d2, 2a9b161
