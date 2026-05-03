---
phase: 06-odds-deviation
plan: 01
subsystem: ml-pipeline
tags: [conformal-prediction, ev-interval, odds-deviation, lightgbm, win-selection-gate]

# Dependency graph
requires:
  - phase: 05-foundation-features
    provides: odds_dynamics_features, haron_zscore_trend, actual_pace_fit
provides:
  - compute_odds_deviation_features() standalone function for deviation_rank and deviation_zscore
  - RobustConfidenceEstimator.predict_interval() with 2-level alpha (80%/90%) EV intervals
  - conformal_confidence_score integrated into WinSelectionGate pair scoring
affects: [07-ensemble-stacking, backtest-roi]

# Tech tracking
tech-stack:
  added: []
  patterns: [conformal-interval-scaling, race-relative-features, pair-score-integration]

key-files:
  created:
    - src/features/odds_deviation_features.py
    - tests/test_odds_deviation.py
  modified:
    - src/models/two_stage_return_model.py
    - src/models/robust_confidence_estimator.py
    - src/models/win_selection_gate.py
    - src/pipelines/training_pipeline.py
    - src/backtest/race_predictor.py
    - tests/test_robust_confidence_estimator.py
    - tests/test_race_predictor.py
    - tests/test_backtest_engine.py
    - tests/test_win_feature_analysis.py

key-decisions:
  - "conformal_confidence_score = EV_lower_80 * (1 - normalized_width_90) -- higher score = more confident bet"
  - "predict_lower_bound refactored as wrapper calling predict_interval for code reuse (Pitfall 2 prevention)"
  - "confidence added as pair_scores only (not 4th combo dimension) to avoid sample sparsity (Pitfall 3)"
  - "alpha scaling formula: sqrt(calibrated_alpha / requested_alpha) for interval width adjustment"
  - "deviation features added to PlaceTwoStageModel.RETURN_FEATURE_COLS to maintain Win/Place feature parity"

patterns-established:
  - "Post-model feature computation: compute_odds_deviation_features() called after AbilityModel in both training and inference"
  - "Conformal interval expansion: predict_interval() provides upper/lower bounds with confidence scoring"
  - "Lazy feature computation: _prepare_features() fallback for deviation features at inference time"

requirements-completed: [ODDS-01, ODDS-02, ODDS-03]

# Metrics
duration: 19min
completed: 2026-05-03
---

# Phase 6 Plan 01: Odds Deviation EV Summary

**Odds deviation features (deviation_rank, deviation_zscore) as Stage2 inputs + conformal EV interval with 2-level alpha (80%/90%) + conformal_confidence_score in WinSelectionGate**

## Performance

- **Duration:** 19 min
- **Started:** 2026-05-03T09:18:20Z
- **Completed:** 2026-05-03T09:37:36Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Standalone compute_odds_deviation_features() with race-internal rank and z-score from odds_to_ability_ratio
- RobustConfidenceEstimator extended with predict_interval() for upper/lower EV bounds at 80%/90% confidence
- conformal_confidence_score (EV_lower_80 * (1 - normalized_width_90)) integrated into WinSelectionGate pair scoring
- predict_lower_bound() refactored as backward-compatible wrapper (3 existing call sites unchanged)
- Three-layer test suite: 12 unit tests + 2 integration tests + 3 numerical consistency tests

## Task Commits

Each task was committed atomically (TDD with RED/GREEN):

1. **Task 1 RED: Odds deviation tests** - `191a6d7` (test)
2. **Task 1 GREEN: Odds deviation features + pipeline integration** - `89a49aa` (feat)
3. **Task 2 RED: predict_interval tests** - `4eb3af3` (test)
4. **Task 2 GREEN: Conformal EV interval + WinSelectionGate integration** - `9bede2d` (feat)

## Files Created/Modified
- `src/features/odds_deviation_features.py` - compute_odds_deviation_features() standalone function
- `src/models/two_stage_return_model.py` - Added deviation_rank, deviation_zscore to FEATURE_COLS + lazy computation
- `src/models/robust_confidence_estimator.py` - Added predict_interval(), refactored predict_lower_bound()
- `src/models/win_selection_gate.py` - Added confidence_edges, confidence_prob/confidence_edge pair_scores
- `src/pipelines/training_pipeline.py` - Added compute_odds_deviation_features() call after odds_to_ability_ratio
- `src/backtest/race_predictor.py` - Added compute_odds_deviation_features() in inference chain + predict_interval
- `tests/test_odds_deviation.py` - 9 tests (6 unit + 3 numerical consistency)
- `tests/test_robust_confidence_estimator.py` - 6 new tests for predict_interval
- `tests/test_race_predictor.py` - 2 pipeline integration tests
- `tests/test_backtest_engine.py` - Updated mocks for predict_interval migration
- `tests/test_win_feature_analysis.py` - Updated feature baseline for deviation features

## Decisions Made
- conformal_confidence_score formula chosen as EV_lower_80 * (1 - normalized_width_90) per D-06/D-08: balances EV lower bound (profitability) with interval width (confidence)
- Confidence added as pair_scores (confidence_prob, confidence_edge) not 4th combo dimension to avoid combination explosion per Pitfall 3 in RESEARCH.md
- predict_lower_bound() refactored as thin wrapper calling predict_interval() then dropping upper/confidence columns, preventing dual maintenance per Pitfall 2
- deviation features added to PlaceTwoStageModel.RETURN_FEATURE_COLS to maintain existing Win/Place feature parity invariant
- race_id guard added to compute_odds_deviation_features() for robustness with test DataFrames that lack race_id

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added race_id guard to compute_odds_deviation_features()**
- **Found during:** Task 2 (running full test suite)
- **Issue:** Test DataFrames for _prepare_features() don't contain race_id, causing KeyError in groupby
- **Fix:** Added early return with NaN columns when race_id is missing from DataFrame
- **Files modified:** src/features/odds_deviation_features.py
- **Verification:** All 1095 tests pass, including test_two_stage_return_model.py
- **Committed in:** 9bede2d (Task 2 GREEN commit)

**2. [Rule 1 - Bug] Fixed alpha scaling direction for conformal intervals**
- **Found during:** Task 2 (TDD GREEN phase - test_narrower_alpha_narrower_interval failed)
- **Issue:** sqrt(primary_alpha / self.alpha) caused 80% interval to be wider than 90% (wrong direction)
- **Fix:** Changed to sqrt(self.alpha / primary_alpha) so higher alpha produces narrower interval
- **Files modified:** src/models/robust_confidence_estimator.py
- **Verification:** test_narrower_alpha_narrower_interval passes
- **Committed in:** 9bede2d (Task 2 GREEN commit)

**3. [Rule 3 - Blocking] Updated test mocks for predict_interval migration**
- **Found during:** Task 2 (running full test suite)
- **Issue:** Existing tests in test_backtest_engine.py mock predict_lower_bound but RacePredictor now calls predict_interval
- **Fix:** Added predict_interval.return_value to all mock setups alongside predict_lower_bound
- **Files modified:** tests/test_backtest_engine.py, tests/test_race_predictor.py
- **Verification:** All 1095 tests pass with zero failures
- **Committed in:** 9bede2d (Task 2 GREEN commit)

**4. [Rule 2 - Missing Critical] Added deviation features to PlaceTwoStageModel.RETURN_FEATURE_COLS**
- **Found during:** Task 2 (running full test suite)
- **Issue:** Existing test asserts all Win FEATURE_COLS are in Place RETURN_FEATURE_COLS; deviation features broke this invariant
- **Fix:** Added deviation_rank, deviation_zscore to PlaceTwoStageModel.RETURN_FEATURE_COLS
- **Files modified:** src/models/two_stage_return_model.py, tests/test_win_feature_analysis.py
- **Verification:** All 1095 tests pass
- **Committed in:** 9bede2d (Task 2 GREEN commit)

---

**Total deviations:** 4 auto-fixed (2 missing critical, 1 bug, 1 blocking)
**Impact on plan:** All auto-fixes necessary for correctness and backward compatibility. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Odds deviation features and EV interval fully integrated in both training and inference pipelines
- Ready for Phase 7 (ensemble stacking) which can leverage deviation features and confidence scores
- conformal_confidence_score available in WinSelectionGate for potential future confidence-weighted betting strategies
- Backward compatibility maintained: predict_lower_bound() still works at 3 existing call sites

---
*Phase: 06-odds-deviation*
*Completed: 2026-05-03*
