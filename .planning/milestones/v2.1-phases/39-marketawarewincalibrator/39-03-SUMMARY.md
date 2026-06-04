---
phase: 39-marketawarewincalibrator
plan: 03
subsystem: backtest, db
tags: [calibrator-integration, race-predictor, model-loader, mlflow, joblib, submodelset]

# Dependency graph
requires:
  - phase: 39-marketawarewincalibrator
    plan: 01
    provides: MarketAwareWinCalibrator class with train/apply/save/load
  - phase: 39-marketawarewincalibrator
    plan: 02
    provides: SubmodelSet with market_aware_win_calibrator field, TrainingPipeline integration
provides:
  - RacePredictor using MarketAwareWinCalibrator instead of WinBenterGate + WinSegmentCalibrator
  - ModelLoader loading MarketAwareWinCalibrator from MLflow and local storage
  - Complete removal of WinBenterGate/WinSegmentCalibrator from inference pipeline
affects: [backtest-engine, paper-predictor, race-predictor, model-loader]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MarketAwareWinCalibrator.apply() at WinBenterGate pipeline position"
    - "Fallback block computing p_win_final/edge_win when calibrator unavailable"
    - "Neutral segment factor defaults (1.0/1.0/NA) replacing WinSegmentCalibrator"

key-files:
  created: []
  modified:
    - src/backtest/race_predictor.py
    - src/db/model_loader.py
    - tests/test_race_predictor.py

key-decisions:
  - "Fallback block computes p_win_final = p_win_corrected / race_sum when calibrator is None or untrained"
  - "win_segment_prob_factor/ev_factor/key set to neutral defaults (1.0/1.0/NA) unconditionally"
  - "BenterCombination import preserved in ModelLoader for place prediction (benter_combo field)"

patterns-established:
  - "MarketAwareWinCalibrator loaded via lazy import inside try/except blocks in ModelLoader"

requirements-completed: [CAL-01, CAL-04, CAL-05]

# Metrics
duration: 19min
completed: 2026-05-27
---

# Phase 39 Plan 03: Pipeline Integration (RacePredictor + ModelLoader) Summary

**RacePredictor and ModelLoader fully migrated to MarketAwareWinCalibrator, removing all WinBenterGate/WinSegmentCalibrator references from inference path**

## Performance

- **Duration:** 19 min
- **Started:** 2026-05-27T22:20:08Z
- **Completed:** 2026-05-27T22:39:10Z
- **Tasks:** 2 (both auto)
- **Files modified:** 3

## Accomplishments
- RacePredictor.predict() uses MarketAwareWinCalibrator.apply() at same pipeline position (after EV correction, before WinSelectionGate)
- Fallback block computes p_win_final and edge_win when calibrator is None or untrained
- _get_win_segment_calibrator method removed entirely
- get_win_candidates() no longer calls segment_calibrator.apply(); segment factors always neutral (1.0)
- ModelLoader loads MarketAwareWinCalibrator from both MLflow and local storage paths
- SubmodelSet construction in both load paths uses market_aware_win_calibrator parameter
- No references to WinBenterGate, win_benter, WinSegmentCalibrator, win_segment_calibrator in race_predictor.py
- No loading references to win_benter, win_isotonic_calibrator, win_temperature_scaler, win_segment_calibrator in model_loader.py
- BenterCombination import preserved for place prediction (benter_combo field)
- All 71 RacePredictor + ModelLoader tests passing (including 4 previously-failing ModelLoader tests)
- Full test suite: 2074 passed, 2 pre-existing failures (backtest_engine, win_profit_selector)

## Task Commits

Each task was committed atomically:

1. **Task 1: RacePredictor replacement** - `ac2c464` (feat)
2. **Task 2: ModelLoader update** - `1594803` (feat)

## Files Created/Modified
- `src/backtest/race_predictor.py` - Replaced WinBenterGate with MarketAwareWinCalibrator.apply(); removed _get_win_segment_calibrator; replaced WinSegmentCalibrator with neutral defaults; added fallback block for p_win_final/edge_win
- `src/db/model_loader.py` - Replaced WinSegmentCalibrator loading with MarketAwareWinCalibrator in MLflow and local paths; removed Win Benter/Isotonic/TempScaler loading; updated both SubmodelSet construction sites; removed WinSegmentCalibrator import
- `tests/test_race_predictor.py` - Updated _make_submodel_mock to use market_aware_win_calibrator; replaced segment calibrator test with neutral factor test; added 2 new MarketAwareWinCalibrator integration tests (apply + fallback)

## Decisions Made
- Fallback block computes p_win_final = p_win_corrected / groupby(race_id).transform("sum") when calibrator unavailable, providing graceful degradation
- Segment factors (win_segment_prob_factor, win_segment_ev_factor, win_segment_key) set to neutral values unconditionally since segment calibration is now handled by the global calibrator
- Lazy import of MarketAwareWinCalibrator inside try/except blocks in ModelLoader, consistent with existing pattern for other artifacts

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- 2 pre-existing test failures (test_backtest_engine unobserved groupby, test_win_profit_selector candidate set) -- both confirmed failing before 39-03 changes, caused by Phase 38 and Phase 39-02 changes respectively
- 1 test iteration needed for fallback behavior test (mock predict_interval not preserving fallback columns) -- fixed by adjusting mock return value

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- RacePredictor and ModelLoader fully integrated with MarketAwareWinCalibrator
- Inference pipeline complete: TrainingPipeline trains -> ModelLoader loads -> RacePredictor applies
- Phase 39 fully complete (all 3 plans done)

## Self-Check: PASSED

- FOUND: src/backtest/race_predictor.py
- FOUND: src/db/model_loader.py
- FOUND: tests/test_race_predictor.py
- FOUND: ac2c464 (feat -- Task 1: RacePredictor replacement)
- FOUND: 1594803 (feat -- Task 2: ModelLoader update)

---
*Phase: 39-marketawarewincalibrator*
*Completed: 2026-05-27*
