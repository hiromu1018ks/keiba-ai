---
phase: 44-roi-bisect
plan: 01
subsystem: backtest
tags: [attribution, bisect, mawc, ranker, coefficients, ece, apr, bet-count, obf]

# Dependency graph
requires:
  - phase: 43-shadow-diagnosis
    provides: "shadow_diagnosis artifacts (diagnosis result, segment constants)"
  - phase: 41-shadow-comparison-framework
    provides: "shadow_horse_diff, shadow_race_diff, _compute_ece"
  - phase: 42-feature-routing-audit-safety-gates
    provides: "deployment_gate_result.json"
  - phase: 39-market-aware-calibration
    provides: "MAWC LogisticRegression calibrator with 51-dim coef_"
  - phase: 40-race-level-ranker
    provides: "Ranker Ridge scorers with coef_"
provides:
  - "ComponentAttribution class: 4-step sequential attribution (ECE->APR->bet_count->OBF)"
  - "CoefficientAnalysisResult: MAWC 51-dim + Ranker Ridge coefficient analysis"
  - "HistoricalBisect class: auxiliary v1.7->v2.0 artifact comparison"
  - "Recommendations for Phase 45: which 1-2 components to fix"
affects: [45-structural-fix]

# Tech tracking
tech-stack:
  added: []
  patterns: ["4-step sequential attribution (ECE->APR->bet_count->OBF per D-02)", "conditional upstream anomaly check per D-03 clause 4"]

key-files:
  created:
    - src/backtest/component_attribution.py
    - src/backtest/historical_bisect.py
    - tests/test_component_attribution.py
    - tests/test_historical_bisect.py
  modified: []

key-decisions:
  - "Reuse POPULARITY_BAND_EDGES/ODDS_BAND_EDGES/PROB_RANK_BAND_EDGES from shadow_diagnosis.py per D-02"
  - "OBF analysis integrated into bet_count step per D-04, not independent"
  - "Upstream SHAP/gain only triggered when coefficient analysis detects anomalies (D-03 clause 4)"
  - "v1.7 reference ROI=0.978 sourced from CLAUDE.md known issues section"

patterns-established:
  - "Post-hoc attribution pattern: load artifacts -> segment analysis -> coefficient extraction -> recommendations"
  - "Conditional upstream check: anomaly detection in coefficients triggers targeted gain analysis"

requirements-completed: [BISECT-01, BISECT-02]

# Metrics
duration: 9min
completed: 2026-05-30
---

# Phase 44 Plan 01: Component Attribution + Historical Bisect Summary

**4-step sequential ECE/APR/bet_count/OBF attribution engine with MAWC 51-dim and Ranker Ridge coefficient analysis, plus auxiliary v1.7-to-v2.0 historical degradation estimate**

## Performance

- **Duration:** 9 min
- **Started:** 2026-05-30T13:02:39Z
- **Completed:** 2026-05-30T13:11:22Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- ComponentAttribution class loads Phase 41/43/42 artifacts and runs 4-step sequential attribution
- MAWC coefficient analysis extracts and interprets 51-dim LogisticRegression coef_ with per-segment contributions
- Ranker coefficient analysis extracts Ridge relevance (15-dim) and value (15-dim) scorer weights
- HistoricalBisect provides auxiliary v1.7->v2.0 context estimating Phase 35-36 as degradation source

## Task Commits

Each task was committed atomically (TDD RED->GREEN):

1. **Task 1: ComponentAttribution tests** - `a8b2842` (test) -- RED phase
2. **Task 1: ComponentAttribution implementation** - `3d0cd19` (feat) -- GREEN phase
3. **Task 2: HistoricalBisect tests** - `c657036` (test) -- RED phase
4. **Task 2: HistoricalBisect implementation** - `88ffbd6` (feat) -- GREEN phase

## Files Created/Modified
- `src/backtest/component_attribution.py` - Post-hoc 4-step attribution engine with coefficient analysis
- `src/backtest/historical_bisect.py` - Lightweight v1.7->v2.0 artifact comparison
- `tests/test_component_attribution.py` - 19 unit tests (8 test cases) for attribution logic
- `tests/test_historical_bisect.py` - 7 unit tests (4 test cases) for historical bisect

## Decisions Made
- Reused segment constants (POPULARITY_BAND_EDGES etc.) from shadow_diagnosis.py rather than redefining
- OBF analysis integrated into bet_count step per D-04, using post-hoc stake/odds distribution analysis
- Conditional upstream anomaly check uses coefficient magnitude thresholds (>1.5 for MAWC, >1.0 for Ranker relevance dominance, >0.95 for beta_market)
- v1.7 reference ROI (0.978) sourced from CLAUDE.md known issues rather than git tag artifacts
- Phase 35-36 estimated as degradation source based on CLAUDE.md documentation of "Phase 36 side effects"

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- ComponentAttribution ready for Phase 45 to consume recommendation output
- MAWC coefficient analysis identifies logit_market dominance (beta=0.90) as primary ECE degradation driver
- Ranker coefficient analysis identifies if_p_win_final (0.80) and if_ev_calibrated (0.83) as dominant weights
- Phase 35-36 estimated as historical degradation source -- Phase 45 should focus on MAWC segment coefficient adjustment and Ranker threshold relaxation

## Self-Check: PASSED

- src/backtest/component_attribution.py: FOUND
- src/backtest/historical_bisect.py: FOUND
- tests/test_component_attribution.py: FOUND
- tests/test_historical_bisect.py: FOUND
- Commit a8b2842: FOUND
- Commit 3d0cd19: FOUND
- Commit c657036: FOUND
- Commit 88ffbd6: FOUND

---
*Phase: 44-roi-bisect*
*Completed: 2026-05-30*
