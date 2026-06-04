---
phase: 39-marketawarewincalibrator
plan: 01
subsystem: models
tags: [sklearn, logistic-regression, l2-regularization, calibration, benter-blend, one-hot-encoding]

# Dependency graph
requires:
  - phase: 38-investmentfeatureframe
    provides: InvestmentFeatureFrame segment keys (popularity_rank, odds, p_win_race_rank_pct)
provides:
  - MarketAwareWinCalibrator class with train/apply/save/load/build_feature_matrix
  - 51-dim feature matrix construction (6 main + 15 one-hot + 30 interactions)
  - C-selection WF grid search with beta_market guard
affects: [39-02, 39-03, race-predictor, training-pipeline, model-loader]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "LogisticRegression + L2 logit-blend calibrator (sklearn 1.8, no penalty param)"
    - "Fixed one-hot schema with guaranteed columns across folds"
    - "C-selection WF grid search: logloss primary, smaller C tie-breaker"

key-files:
  created:
    - src/models/market_aware_win_calibrator.py
    - tests/test_market_aware_win_calibrator.py
  modified: []

key-decisions:
  - "sklearn 1.8 LogisticRegression without penalty param (default l1_ratio=0.0 = L2)"
  - "beta_market guard: abs(coef_market)/(abs(coef_model)+abs(coef_market)) >= 0.20"
  - "No standardization of features -- L2 handles ~51 dims with known ranges"
  - "joblib serialization consistent with WinSegmentCalibrator pattern"

patterns-established:
  - "Feature matrix: logit(p_model) + logit(p_market) + continuous + one-hot + logit-segment interactions"
  - "WF C-selection grid with year/surface ratio gates and beta_market floor guard"

requirements-completed: [CAL-01, CAL-02, CAL-03, CAL-05]

# Metrics
duration: 6min
completed: 2026-05-27
---

# Phase 39 Plan 01: MarketAwareWinCalibrator Summary

**LogisticRegression L2 logit-blend calibrator with 51-dim feature matrix replacing WinBenterGate + WinSegmentCalibrator**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-27T21:48:33Z
- **Completed:** 2026-05-27T21:54:47Z
- **Tasks:** 1 (TDD: RED + GREEN + REFACTOR)
- **Files modified:** 2

## Accomplishments
- MarketAwareWinCalibrator class with train/apply/save/load/build_feature_matrix methods
- 51-dim feature matrix: 6 main effects + 7 odds band one-hot + 5 pop bucket one-hot + 3 p_rank one-hot + 30 logit-segment interactions
- C-selection WF grid search over [0.03, 0.1, 0.3, 1.0, 3.0] with logloss primary metric and smaller-C tie-breaker
- beta_market >= 0.20 guard via coefficient relative contribution ratio
- D-22 guard rejecting train-mode p_win_pred leakage
- Race-level sum-to-1.0 normalization after predict_proba
- All 11 unit tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Test suite for MarketAwareWinCalibrator** - `3d01530` (test)
2. **Task 1 (GREEN+REFACTOR): Full implementation with lint fixes** - `e02e5dd` (feat)

## Files Created/Modified
- `src/models/market_aware_win_calibrator.py` - MarketAwareWinCalibrator class: LogisticRegression L2 logit-blend calibrator with 51-dim feature matrix, C-selection WF grid search, beta_market guard, race normalization
- `tests/test_market_aware_win_calibrator.py` - 11 test cases covering feature encoding, training, inference, guards, save/load roundtrip, interaction structure

## Decisions Made
- Used sklearn 1.8 default LogisticRegression without `penalty` param (l1_ratio=0.0 default = L2) to avoid deprecation warning
- beta_market guard implemented as abs(coef_market) / (abs(coef_model) + abs(coef_market)) matching BenterCombination alpha/beta ratio semantics
- No feature standardization -- L2 regularization handles ~51 dims with known ranges (logit ~[-5,5], log_odds ~[0,5], pct ~[0,1], field_size ~[8,18])
- Used file-level `# ruff: noqa: N803,N806` for ML convention (X for feature matrix), consistent with stacked_ensemble.py and conformal_ev_model.py

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- MarketAwareWinCalibrator class ready for pipeline integration (Plan 39-02)
- Train/apply/save/load interface matches existing WinBenterGate/WinSegmentCalibrator patterns
- Feature matrix schema (51 dims) fixed and tested across all edge cases
- Ready for training_pipeline.py integration to replace WinBenterGate training block

## Self-Check: PASSED

- FOUND: src/models/market_aware_win_calibrator.py
- FOUND: tests/test_market_aware_win_calibrator.py
- FOUND: .planning/phases/39-marketawarewincalibrator/39-01-SUMMARY.md
- FOUND: 3d01530 (test -- RED phase)
- FOUND: e02e5dd (feat -- GREEN+REFACTOR phase)

---
*Phase: 39-marketawarewincalibrator*
*Completed: 2026-05-27*
