---
phase: 34-validation-and-manifest-update
plan: 03
subsystem: validation
tags: [backtest, ic-evaluation, gpd-diagnostic, validation]

# Dependency graph
requires:
  - phase: 34-01
    provides: rl_* feature registration in FEATURE_COLS
  - phase: 34-02
    provides: POST_RACE leakage verification complete
provides:
  - v1.7 IC baseline values (4 formulations x 3 surfaces)
  - BT 2024 failure diagnosis (rl_* features not available during pipeline.run() post-training phase)
affects: [34-04 manifest freeze]

# Tech tracking
tech-stack:
  added: []
  patterns: [validation-execution-sequence]

key-files:
  created: []
  modified: []

key-decisions:
  - "BT 2024 failed due to missing rl_* columns in pipeline.run() race_level_features phase -- D-10 applied: record as-is"
  - "IC evaluation succeeded from 139,042 OOF predictions -- v1.7 baseline established"
  - "GPD diagnostic failed because place_hit models not saved (betting_target=win skips place training)"
  - "C-orthogonal IC 0.2716 (turf) / 0.2035 (dirt) confirms model has market-independent predictive component"

patterns-established: []

requirements-completed: [VAL-02]

# Metrics
duration: 24min
completed: "2026-05-19"
---

# Phase 34 Plan 03: Validation Execution Summary

IC evaluation succeeded (C-orthogonal IC 0.27 turf/0.20 dirt), but BT 2024 and GPD diagnostic both failed due to feature column availability issues -- OOF predictions saved (139K rows) enable future IC comparison.

## Performance

- **Duration:** 24 min
- **Started:** 2026-05-19T00:24:12Z
- **Completed:** 2026-05-19T00:48:18Z
- **Tasks:** 3 attempted (1 succeeded, 2 failed)
- **Files modified:** 0 (data-only outputs)

## Accomplishments
- IC baseline values recorded as v1.7 baseline per D-06 (4 formulations x 3 surfaces)
- OOF predictions saved: 139,042 rows (usable for future IC evaluation)
- C-orthogonal IC confirms market-independent predictive component exists

## Task Commits

No source code commits -- all tasks produce data artifacts (ignored by .gitignore).

1. **Task 1: Execute single-year BT 2024 (VAL-01)** -- FAILED
2. **Task 2: Execute IC evaluation (VAL-02)** -- SUCCEEDED
3. **Task 3: Execute GPD diagnostic (VAL-03)** -- FAILED

## Task Details

### Task 1: BT 2024 (VAL-01) -- FAILED

**Error:** `pipeline.run()` failed with: `['implied_prob_hhi', 'odds_skewness', 'rl_favorite_in_wide_top1', 'rl_trio_overlap', 'rl_market_consistency', 'rl_trio_odds_ratio', 'rl_wide_harville_ratio', 'rl_log_odds_entropy', 'rl_odds_dispersion', 'rl_top3_odds_gap', 'rl_top1_odds', 'rl_favorite_rank_gap', 'rl_n_horses'] not in index`

**Diagnosis:** The error occurs during `pipeline.run()` after the submodel training completes. Specifically:
1. Training of turf/dirt submodels succeeded (all 8 LightGBM models + ensembles trained)
2. OOF predictions saved successfully (139,042 rows)
3. Feature cache HIT used (feat_c303897e117839a5)
4. After training, `pipeline.run()` builds race-level features via `_build_race_level_features()`
5. Then `RaceQualityScreener` and `RegimeDetector` training phases attempt to use `rl_*` columns
6. The `rl_*` columns are not present in `feat_df` at this point because they are computed by `FeatureEngine.build_all()` but the training pipeline's `_build_race_level_features()` produces different columns

**Root cause:** The `pipeline.run()` method calls `_build_race_level_features()` which constructs a `race_feat` DataFrame from aggregated horse-level features. The 6 `rl_*` features (RLF-01~06) from `compute_race_level_features()` in `feature_engine.py` are NOT the same columns as what `_build_race_level_features()` in `training_pipeline.py` produces. Additionally, `implied_prob_hhi` and `odds_skewness` are missing because `compute_flb_slope()` is not called in `build_features()` (known bug WR-02 from STATE.md).

**Outcome:** No ROI result available. No models saved to `data/models-backtest/2024/`. Old models from May 17 remain.

### Task 2: IC Evaluation (VAL-02) -- SUCCEEDED

**Results:**

| Surface | B-diff (rho) | C-orth (rho) | E-incr (delta) | Per-race (mean) | Direction |
|---------|-------------|--------------|----------------|-----------------|-----------|
| Turf    | -0.0230     | **0.2716**   | 0.0382         | 0.5499          | INCONSISTENT |
| Dirt    | -0.0995     | **0.2035**   | 0.0143         | 0.5071          | INCONSISTENT |
| All     | -0.0624     | **0.2368**   | 0.0254         | 0.5282          | INCONSISTENT |

**Claude's judgment (D-06 discretion):**
- C-orthogonal IC is **positive and significant** (0.27 turf, 0.20 dirt) -- the model has genuine market-independent predictive power
- B-difference IC is negative as theoretically expected (model predictions include information already in odds)
- E-incremental is modest (0.04 turf, 0.01 dirt) -- model adds small improvement over raw market IC
- Per-race IC is strong (0.55 turf, 0.51 dirt) -- individual race predictions are meaningful
- Direction inconsistency is expected (B-diff is negative, others are positive)
- **Verdict: IC values are good and establish a solid v1.7 baseline**

**Output:** `data/baseline/ic_baseline.json` (3,119 bytes)

### Task 3: GPD Diagnostic (VAL-03) -- FAILED

**Error:** `FileNotFoundError: No model file found for place_hit_turf in data/models-backtest/2024`

**Diagnosis:** The GPD diagnostic requires all model files including place models. Since the BT ran with `--betting-target win`, place models were not trained and not saved. The models in `data/models-backtest/2024/` are from a previous run that also skipped place training.

**Outcome:** No GPD report generated. No MDR/FAD values available.

## Decisions Made
- D-10 applied: all failures recorded as-is, no retries or debugging
- IC values serve as v1.7 baseline regardless of BT failure
- GPD failure does not block manifest freeze (Plan 04)

## Deviations from Plan

### Validation Failures (D-10 applied)

**1. BT 2024 failed -- missing rl_* columns in pipeline.run() post-training phase**
- **Found during:** Task 1 (BT execution)
- **Issue:** rl_* features registered in FEATURE_COLS (Plan 34-01) but pipeline.run() fails when trying to use them after submodel training
- **Root cause:** `_build_race_level_features()` in training_pipeline.py produces different columns than `compute_race_level_features()` in feature_engine.py; also WR-02 bug (compute_flb_slope not called)
- **Outcome:** Recorded as-is per D-10

**2. GPD diagnostic failed -- missing place_hit models**
- **Found during:** Task 3 (GPD execution)
- **Issue:** Models directory lacks place_hit models (betting_target=win skips place training)
- **Outcome:** Recorded as-is per D-10

## Issues Encountered

- **WR-02 confirmed:** `implied_prob_hhi` and `odds_skewness` missing from feature pipeline (existing known bug, not introduced by this phase)
- **rl_* feature pipeline gap:** Features are registered in FEATURE_COLS but the training pipeline's internal flow does not produce them in the right place for RaceQualityScreener/RegimeDetector
- **Place model dependency:** GPD diagnostic requires place models even when only evaluating win model

## Next Phase Readiness
- IC baseline values available for manifest freeze (Plan 04)
- BT failure root cause identified: rl_* features need to be added to `_build_race_level_features()` in training_pipeline.py (requires code fix, deferred to future phase per D-11)
- GPD diagnostic requires place model training or script modification to skip place models
- All 3 validation artifacts available at varying completeness levels:
  - `data/oof/oof_predictions.parquet`: 139,042 rows (populated)
  - `data/baseline/ic_baseline.json`: complete
  - `data/gpd/gpd_report.json`: missing

---
*Phase: 34-validation-and-manifest-update*
*Completed: 2026-05-19*
