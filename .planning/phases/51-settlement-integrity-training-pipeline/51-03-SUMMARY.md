---
phase: 51
plan: 03
subsystem: training-pipeline
tags: [training, model-loader, track-stats, betting-target, cache-deps]
dependency_graph:
  requires: [TRN-01, TRN-02, TRN-03, TRN-04, TRN-05]
  provides: [--betting-target, track_stats_persistence, model_loader_priority, cache_deps]
  affects: [run_train.py, training_pipeline.py, feature_engine.py, model_loader.py, test_model_loader.py]
tech_stack:
  added: []
  patterns: [explicit-source-selection, sha256-checksum, pre-training-validation]
key_files:
  created: []
  modified:
    - scripts/run_train.py
    - src/pipelines/training_pipeline.py
    - src/features/feature_engine.py
    - src/db/model_loader.py
    - tests/test_model_loader.py
decisions: [D-13, D-14, D-15, D-16, D-17]
metrics:
  duration: 15min
  completed: 2026-06-06
  tasks: 2
  files: 5
---

# Phase 51 Plan 03: Training Pipeline Fixes Summary

Training pipeline --betting-target support, track_stats persistence, pre-training Parquet validation, feature cache deps, and ModelLoader priority fix.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | run_train.py + TrainingPipeline + FeatureEngine modifications | 962217d | scripts/run_train.py, src/pipelines/training_pipeline.py, src/features/feature_engine.py |
| 2 | ModelLoader priority fix + track_stats restore with tests | 962217d | src/db/model_loader.py, tests/test_model_loader.py |

## Key Changes

### scripts/run_train.py
- Added `--betting-target` CLI argument (choices: win|place|wide, default: place)
- Wide is rejected with sys.exit(1) per D-13
- Added pre-training Parquet validation: date range logging, track_conditions.parquet existence check, horse_track_aptitude.parquet check, NaN rate reporting on key columns
- Passes `betting_target` to `pipeline.run()`

### src/pipelines/training_pipeline.py
- `run()` already accepted `betting_target` parameter (verified at line 305)
- `_save_models_local()`: writes `track_stats_{surface}.json` and `track_month_stats_{surface}.json` with SHA256 checksum logging
- MLflow logging: added `mlflow.log_artifact()` for track_stats JSONs and `mlflow.log_param()` for SHA256 checksums
- `meta.json`: includes `betting_target` field

### src/features/feature_engine.py
- Added `("raw", "track_conditions")` and `("raw", "horse_track_aptitude")` to source_paths cache dependency list

### src/db/model_loader.py
- `load()` now requires explicit `run_id` OR `models_dir` (mutually exclusive per D-16)
- Both specified raises ValueError; neither specified raises ValueError
- Removed implicit `_find_latest_run()` auto-selection from `load()`
- `models_dir` specified routes directly to `load_from_dir()`
- `run_id` specified uses MLflow only, no local fallback
- track_stats/track_month_stats restored from JSON in both `load()` (MLflow) and `load_from_dir()` (local)
- `ModelInfo` dataclass includes `betting_target` field (default "place")
- `load_from_dir()` reads `betting_target` from `meta.json`

### tests/test_model_loader.py
- Added `test_load_with_no_args_raises_value_error` (D-16: neither specified)
- Added `test_load_with_both_args_raises_value_error` (D-16: both specified)
- Added `test_load_run_id_does_not_check_local_dir` (D-16: no local fallback)
- Added `test_load_models_dir_does_not_call_mlflow` (D-16: routes to load_from_dir)
- Added `test_model_info_has_betting_target_field` (D-14)
- Added `test_model_info_betting_target_defaults_to_place` (D-14)
- Removed `test_load_uses_latest_run_when_no_run_id` (behavior removed per D-16)

## Verification Results

- `run_train.py --help` shows `--betting-target {win,place,wide}` -- PASSED
- `python -m pytest tests/test_model_loader.py -v` -- 12/12 PASSED
- Feature cache deps include `track_conditions` and `horse_track_aptitude` -- PASSED
- `meta.json` includes `betting_target` field -- PASSED
- track_stats JSON persistence present in `_save_models_local()` -- PASSED
- ModelLoader mutual exclusivity (both/neither raises ValueError) -- PASSED

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None -- all data paths are wired; track_stats restore gracefully handles missing files (backward compatible).

## Threat Flags

No new threat surface beyond what was documented in the plan's threat model.

## Self-Check: PASSED

All 5 modified files exist. Commit 962217d confirmed. SUMMARY.md created at expected path.
