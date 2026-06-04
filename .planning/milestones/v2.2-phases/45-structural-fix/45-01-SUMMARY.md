---
phase: 45-structural-fix
plan: 01
subsystem: ml-models
tags: [mawc, logistic-regression, calibration, oof, quality-gates, conservative-retraining]

# Dependency graph
requires:
  - phase: 44-roi-bisect
    provides: MAWC beta_market=0.90 dominance finding, ECE 3x degradation in odds 1-3
  - phase: 39-marketawarewincalibrator
    provides: MarketAwareWinCalibrator class with 51-dim feature matrix + encoding helpers
provides:
  - MawcConservativeRetrainer class with OOF retraining + quality gates + variant creation + manifest generation
  - 36-dim conservative feature matrix (15 logit_model_x_* interactions removed)
  - Conservative C grid [0.003, 0.005, 0.01, 0.03] search with quality gate evaluation
  - Favorite band guard (odds 1-3): ECE non-degradation + p compression >= 0.90 + EV pass rate check
  - Conservative variant directory creation (copytree + MAWC joblib replacement)
  - Manifest JSON generation for Phase 46 consumption
affects: [46-quality-gate-verification, shadow-comparison]

# Tech tracking
tech-stack:
  added: []
  patterns: [conservative-variant-creation, quality-gate-c-grid-search, favorite-band-guard]

key-files:
  created:
    - src/models/mawc_conservative_retrainer.py
    - tests/test_mawc_conservative_retrainer.py
  modified: []

key-decisions:
  - "Conservative retraining uses 36-dim features (removed all 15 logit_model_x_* interactions)"
  - "C grid [0.003, 0.005, 0.01, 0.03] -- stronger regularization than original [0.03, 0.1, 0.3, 1.0, 3.0]"
  - "Minimum C selected among gate-passing candidates; not_deployed if all fail"
  - "ECE_DEGRADATION_TOLERANCE=0.10, P_COMPRESSION_FLOOR=0.90, BET_COUNT_TOLERANCE=0.10"
  - "Standalone module independent of TrainingPipelineV5 (per CONTEXT D-02)"

patterns-established:
  - "Conservative variant pattern: copytree source year dir + replace only MAWC joblib"
  - "Quality gate pattern: overall metrics + favorite band guard + year-level non-degradation"
  - "Self-contained ECE computation (no import from shadow_comparison)"

requirements-completed: [FIX-01, FIX-02]

# Metrics
duration: 15min
completed: 2026-05-31
---

# Phase 45 Plan 01: Conservative MAWC Retrainer Summary

**MawcConservativeRetrainer with 36-dim feature matrix, C grid quality gates, favorite band guard, and variant directory creation for Phase 46 Shadow Comparison**

## Performance

- **Duration:** 15 min
- **Started:** 2026-05-31T12:55:55Z
- **Completed:** 2026-05-31T13:10:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- MawcConservativeRetrainer class with 6 methods: prepare_oof_data, build_conservative_feature_matrix, retrain_with_c_grid, evaluate_quality_gates, select_best_c, run_retrain
- Conservative variant creation via shutil.copytree + MAWC joblib replacement, manifest JSON generation for Phase 46
- 27 unit tests covering all methods, edge cases, and data flow, lint clean

## Task Commits

Each task was committed atomically:

1. **Task 1: MawcConservativeRetrainer -- OOF data + feature matrix + C grid + quality gates** - `21ae2ac` (feat)
2. **Task 2: Conservative variant directory + manifest generation** - `1d671b5` (feat)

_Note: Both tasks followed TDD flow with tests written alongside implementation._

## Files Created/Modified
- `src/models/mawc_conservative_retrainer.py` - MawcConservativeRetrainer class with OOF retraining, quality gates, variant creation, manifest generation (720 lines)
- `tests/test_mawc_conservative_retrainer.py` - Comprehensive unit tests (660 lines)

## Decisions Made
- Reused MAWC encoding helpers (_encode_odds_band, _encode_pop_bucket, _encode_p_rank) via class-level helper instance instead of duplicating encoding logic
- Implemented self-contained _compute_ece function (mirrors ShadowComparisonFramework algorithm) to avoid circular dependency
- Conservative feature matrix reuses MAWC main effects + segment one-hot + logit_market_x_* interactions, only removing logit_model_x_* (15 terms)
- Year-level non-degradation checks skip years with < 10 samples to avoid unreliable metrics

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Test test_passes_when_non_degrade initially failed because baseline metrics were hardcoded rather than computed from the actual prediction arrays. Fixed by computing actual_brier/logloss/ece from the predictions before passing as baselines.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- MawcConservativeRetrainer is ready for Phase 46 (Quality Gate Verification)
- run_full_pipeline() can be called with data/oof/oof_predictions.parquet and data/models-backtest/ to produce conservative variants
- Phase 46 should run ShadowComparisonFramework with --shadow-root data/models-backtest-mawc-conservative

---
*Phase: 45-structural-fix*
*Completed: 2026-05-31*

## Self-Check: PASSED

- FOUND: src/models/mawc_conservative_retrainer.py
- FOUND: tests/test_mawc_conservative_retrainer.py
- FOUND: .planning/phases/45-structural-fix/45-01-SUMMARY.md
- FOUND: 21ae2ac (Task 1 commit)
- FOUND: 1d671b5 (Task 2 commit)
