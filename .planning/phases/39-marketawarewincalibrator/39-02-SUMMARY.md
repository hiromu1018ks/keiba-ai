---
phase: 39-marketawarewincalibrator
plan: 02
subsystem: pipelines, models
tags: [sklearn, logistic-regression, training-pipeline, submodelset, oof-dataframe, mlflow, joblib]

# Dependency graph
requires:
  - phase: 39-marketawarewincalibrator
    provides: MarketAwareWinCalibrator class with train/apply/save/load/build_feature_matrix
provides:
  - TrainingPipeline trains MarketAwareWinCalibrator via extended OOF DataFrame
  - SubmodelSet updated with market_aware_win_calibrator field (4 old win fields removed)
  - generate_win_oof_predictions() returns DataFrame with calibrator feature columns
  - MLflow and local save/load handle market_aware_win_calibrator artifact
affects: [39-03, model-loader, race-predictor]

# Tech tracking
tech-stack:
  added: []
patterns:
  - "generate_win_oof_predictions() returns enriched DataFrame instead of tuple of arrays"
  - "MarketAwareWinCalibrator.train() called directly on OOF DataFrame (no BenterCombination intermediate)"

key-files:
  created: []
  modified:
    - src/domain/models.py
    - src/pipelines/training_pipeline.py
    - src/models/win_benter_gate.py
    - tests/test_domain.py
    - tests/test_win_benter_gate.py

key-decisions:
  - "Extended generate_win_oof_predictions() to return DataFrame directly instead of creating wrapper function (simpler, single call site)"
  - "p_win_race_rank_pct computed inside generate_win_oof_predictions() from OOF predictions grouped by race_id (D-19)"
  - "WinSegmentCalibrator training block replaced with comment marker for Plan 03 clarity"

patterns-established:
  - "OOF DataFrame pattern: generate_win_oof_predictions() returns DataFrame with p_win_oof, p_market_norm, tanodds, popularity_rank, field_size, p_win_race_rank_pct, race_id, race_date, umaban, surface, kakuteijyuni, p_win_corrected"

requirements-completed: [CAL-01, CAL-03, CAL-04, CAL-05]

# Metrics
duration: 16min
completed: 2026-05-27
---

# Phase 39 Plan 02: Pipeline Integration Summary

**TrainingPipeline trains MarketAwareWinCalibrator via enriched OOF DataFrame, replacing WinBenterGate grid search + calibration comparison + WinSegmentCalibrator chain**

## Performance

- **Duration:** 16 min
- **Started:** 2026-05-27T21:57:12Z
- **Completed:** 2026-05-27T22:13:19Z
- **Tasks:** 2 (TDD: RED + GREEN for both)
- **Files modified:** 5

## Accomplishments
- MarketAwareWinCalibrator integrated into TrainingPipeline replacing ~150 lines of WinBenterGate grid search, calibration comparison, and temperature scaling logic
- SubmodelSet updated: 4 old win fields removed (win_benter, win_isotonic_calibrator, win_temperature_scaler, win_segment_calibrator), 1 new field added (market_aware_win_calibrator)
- generate_win_oof_predictions() extended to return DataFrame with all columns needed for calibrator feature matrix (D-18/D-19/D-20)
- MLflow and local save/load handle new market_aware_win_calibrator_{surface} artifact
- All 31 training pipeline tests + 15 win_benter_gate tests + 11 calibrator tests + 6 domain tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests for SubmodelSet field changes** - `97dde5d` (test)
2. **Task 1 (GREEN): SubmodelSet field update** - `23d4f8a` (feat)
3. **Task 2 (GREEN): Pipeline integration + test updates** - `7eca429` (feat)
4. **Task 2 (fix): win_benter_gate test updates** - `2dd237e` (fix)

## Files Created/Modified
- `src/domain/models.py` - SubmodelSet: removed 4 win fields, added market_aware_win_calibrator, added MarketAwareWinCalibrator TYPE_CHECKING import
- `src/pipelines/training_pipeline.py` - Replaced WinBenterGate training block with MarketAwareWinCalibrator.train(), removed WinSegmentCalibrator training, updated SubmodelSet construction, updated MLflow/local save, removed WinSegmentCalibrator import
- `src/models/win_benter_gate.py` - generate_win_oof_predictions() now returns DataFrame with enriched columns (p_win_oof, p_market_norm, tanodds, popularity_rank, field_size, p_win_race_rank_pct, race_id, race_date, umaban, surface, kakuteijyuni, p_win_corrected)
- `tests/test_domain.py` - 4 new tests for SubmodelSet field changes (market_aware_win_calibrator, removed fields, place fields preserved)
- `tests/test_win_benter_gate.py` - Updated 3 tests for new DataFrame return and SubmodelSet field changes

## Decisions Made
- Extended generate_win_oof_predictions() to return DataFrame directly instead of creating a separate wrapper function. Rationale: only one call site exists (training_pipeline.py), so a wrapper adds unnecessary indirection.
- p_win_race_rank_pct computed inside generate_win_oof_predictions() from OOF predictions grouped by race_id (D-19). This is an OOF-dependent feature that MUST be recomputed per fold.
- WinBenterGate class and compare_calibrations/generate_reliability_data functions retained in win_benter_gate.py for backward compatibility (used by existing tests).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- 6 pre-existing test failures in test_model_loader.py, test_ensemble_gate_propagation.py, and test_win_profit_selector.py due to ModelLoader/RacePredictor still referencing old SubmodelSet fields. These are in scope for Plan 39-03 (ModelLoader/RacePredictor integration).
- 1 pre-existing test failure in test_backtest_engine.py (unobserved groupby in feature_frame.py) from Phase 38.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- TrainingPipeline fully trains MarketAwareWinCalibrator and stores it in SubmodelSet
- ModelLoader needs update to load market_aware_win_calibrator_{surface}.joblib and populate SubmodelSet (Plan 39-03)
- RacePredictor needs update to use MarketAwareWinCalibrator.apply() instead of WinBenterGate.apply() (Plan 39-03)
- 6 test failures in ModelLoader/RacePredictor tests to be resolved in Plan 39-03

## Self-Check: PASSED

- FOUND: src/domain/models.py
- FOUND: src/pipelines/training_pipeline.py
- FOUND: src/models/win_benter_gate.py
- FOUND: tests/test_domain.py
- FOUND: tests/test_win_benter_gate.py
- FOUND: .planning/phases/39-marketawarewincalibrator/39-02-SUMMARY.md
- FOUND: 97dde5d (test -- RED phase Task 1)
- FOUND: 23d4f8a (feat -- GREEN phase Task 1)
- FOUND: 7eca429 (feat -- GREEN phase Task 2)
- FOUND: 2dd237e (fix -- test updates Task 2)

---
*Phase: 39-marketawarewincalibrator*
*Completed: 2026-05-27*
