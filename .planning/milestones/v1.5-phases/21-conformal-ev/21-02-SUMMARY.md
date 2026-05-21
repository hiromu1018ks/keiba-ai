---
phase: 21-conformal-ev
plan: 02
subsystem: ml-pipeline
tags: [cqr, conformal-prediction, pipeline-integration, pfp]
dependency_graph:
  requires: [21-01]
  provides: [cqr-training-in-pipeline, cqr-model-loading, cqr-diagnostics, pfp-sha256]
  affects: [training_pipeline, model_loader, race_predictor, ev_diagnostics]
tech_stack:
  added: [ConformalEVModel.train-in-pipeline, _compute_cqr_coverage]
  patterns: [CQR-per-surface, SHA256-tamper-detection, legacy-fallback-loading]
key_files:
  created: []
  modified:
    - src/pipelines/training_pipeline.py
    - src/db/model_loader.py
    - src/backtest/race_predictor.py
    - src/models/ev_diagnostics.py
    - src/models/conformal_ev_model.py
    - tests/test_conformal_ev_model.py
decisions:
  - CQR feature columns filtered to numeric-only dtypes to prevent LightGBM errors
  - MLflow logging logs CQR models directly without local intermediate save
  - Legacy confidence_params.json loading preserved as fallback in ModelLoader
metrics:
  duration: 46min
  completed: 2026-05-09
  tests_passed: 1392
  tests_added: 4
---

# Phase 21 Plan 02: Pipeline Integration Summary

CQR-based ConformalEVModel integrated into TrainingPipelineV5, ModelLoader, and RacePredictor with PFP SHA256 tamper detection and CQR coverage diagnostics.

## Changes Made

### Task 1: TrainingPipelineV5 + ModelLoader + RacePredictor + PFP

**src/pipelines/training_pipeline.py:**
- Replaced `RobustConfidenceEstimator` alias with direct `ConformalEVModel` import
- Replaced section 7 (confidence calibration shim) with CQR training block that:
  - Computes `actual_ev_win` from confirmed_odds and kakuteijyuni
  - Filters feature columns to numeric dtypes only (excludes object columns)
  - Creates ConformalEVModel with alpha=0.1 and calls train()
- Updated PlaceSelectionGate/WinSelectionGate training data generation to use `conformal_ev.predict_interval()`
- Updated SubmodelSet construction: `conformal_ev_model=conformal_ev`
- Replaced `_save_models_local()` RobustConfidenceEstimator params with CQR model save + SHA256 hash computation
- Added CQR checksums to strategy_manifest.json (if exists)
- Updated `_log_to_mlflow()` to log CQR LightGBM models and params JSON to MLflow

**src/db/model_loader.py:**
- Updated `_load_from_mlflow()` to load CQR models (per-surface files) with legacy confidence_params.json fallback
- Updated `load_from_dir()` to use `ConformalEVModel.load()` with legacy fallback
- Updated SubmodelSet construction in both methods: `conformal_ev_model=conformal_ev`

**src/backtest/race_predictor.py:**
- Added `conformal_ev_model is not None` check before calling predict_interval()
- Added fallback when conformal_ev_model is None: uses raw EV values as both lower and upper bounds

**src/models/conformal_ev_model.py:**
- Removed backward-compat shim attributes (rolling_window, _win_cp_quantile, etc.)

### Task 2: ev_diagnostics extension + tests

**src/models/ev_diagnostics.py:**
- Added `_compute_cqr_coverage()` function: computes coverage rate, interval width stats (mean/median/min/max), target coverage comparison
- Integrated CQR coverage into `compute_ev_diagnostics()` with per-surface breakdown
- Added CQR coverage to `console_summary()` output

**tests/test_conformal_ev_model.py:**
- Added `TestCQRCoverageDiagnostics` class with 4 tests:
  - `test_cqr_coverage_calculation`: normal data coverage rate computation
  - `test_cqr_coverage_insufficient_samples`: <30 samples returns warning
  - `test_cqr_coverage_no_columns`: missing CQR columns returns warning
  - `test_cqr_coverage_in_compute_ev_diagnostics`: integration test with surface breakdown

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Object dtype columns in CQR features**
- **Found during:** Task 1 verification (test_training_pipeline tests failed)
- **Issue:** `distance_bin` (object) and `grade_code` (object) columns passed to LightGBM causing ValueError
- **Fix:** Added dtype filter to feature column extraction: only numeric dtypes (float64, int64, etc.) are included
- **Files modified:** src/pipelines/training_pipeline.py
- **Commit:** 9e93ac2

**2. [Rule 3 - Blocking] Undefined models_dir in _log_to_mlflow()**
- **Found during:** Task 1 verification (test_mlflow_logging tests failed)
- **Issue:** CQR model save to models_dir called from _log_to_mlflow() where models_dir is not defined
- **Fix:** Removed local save from _log_to_mlflow(); models are saved locally in _save_models_local() only
- **Files modified:** src/pipelines/training_pipeline.py
- **Commit:** 9e93ac2

## Verification Results

- All 1392 tests pass (1388 baseline + 4 new CQR coverage tests)
- No regressions
- All acceptance criteria met

## Commits

- `1c7f1ce`: feat(21-02): integrate ConformalEVModel into training/inference pipeline
- `006b465`: feat(21-02): add CQR coverage diagnostics and tests
- `9e93ac2`: fix(21-02): filter object dtype columns from CQR feature set
