---
phase: 23-safety-gate
plan: 01
subsystem: features, models, pipelines
tags: [safe-01, leakage-prevention, post-race-cols, cqr-whitelist, ev-correction]
dependency_graph:
  requires: []
  provides: [post_race_leak_free_pipeline, cqr_whitelist_features, ev_odds_fix]
  affects: [feature_engine, conformal_ev_model, ev_correction_model, training_pipeline]
tech_stack:
  added: []
  patterns: [whitelist-feature-selection, post-race-drop-guard]
key_files:
  created:
    - tests/test_post_race_leakage.py
  modified:
    - src/features/feature_engine.py
    - src/models/conformal_ev_model.py
    - src/models/ev_correction_model.py
    - src/pipelines/training_pipeline.py
    - tests/test_feature_engine.py
    - tests/test_conformal_ev_model.py
decisions:
  - CQR blacklist replaced with explicit FEATURE_COLS whitelist for safety
  - ninki fallback removed from popularity_rank chain (POST_RACE data)
  - EV correction odds-band scaling fixed to always use pre-race odds
  - POST_RACE_COLS drop placed before cache write to prevent cache contamination
metrics:
  duration: 29min
  completed: "2026-05-11"
  tasks: 3
  files: 7
  tests_added: 4
  tests_total: 1396
---

# Phase 23 Plan 01: POST_RACE Leakage Prevention Summary

POST_RACE_COLS (kakuteijyuni, confirmed_odds, ninki, etc.) completely removed from feature pipeline output, CQR model switched to whitelist feature selection, and EV correction odds-band scaling fixed to use pre-race odds consistently.

## Changes

### Task 1: build_all() POST_RACE drop + popularity_rank fallback fix
- Added `POST_RACE_COLS` drop block before cache write in `build_all()` (M1 fix)
- Removed ninki fallback from `popularity_rank` chain in `_map_basic_features()` (M6 fix)
- Updated 3 tests to reflect new behavior (ninki no longer used, confirmed_odds now dropped)

### Task 2: CQR whitelist + EV correction odds fix
- Added `FEATURE_COLS: list[str]` whitelist to `ConformalEVModel` class (M2 fix)
- Replaced blacklist `_NON_FEATURE_COLS` logic in `train()` and `predict_interval()` with whitelist lookup
- Changed `ev_correction_model.py:370` from conditional `confirmed_odds` to fixed `"odds"` (M3 fix)
- Removed inline feature computation from `training_pipeline.py`, delegating to CQR's whitelist
- Updated test fixtures to use real whitelist column names

### Task 3: 3-layer CI leakage detection tests
- Created `tests/test_post_race_leakage.py` with `TestPostRaceLeakage` class
- Layer 1: `test_build_all_output_no_post_race_cols` verifies build_all() output is clean
- Layer 2: `test_model_feature_cols_no_post_race` verifies 7+ model FEATURE_COLS have no POST_RACE overlap
- Layer 3: `test_ev_correction_odds_col_uses_pre_race_odds` verifies correct_ev() uses pre-race odds
- Bonus: `test_conformal_ev_feature_cols_whitelist` verifies CQR whitelist is clean

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Test fixtures used synthetic column names incompatible with whitelist**
- Found during: Task 2
- Issue: `test_conformal_ev_model.py` used `feature_a`/`feature_b` which are not in FEATURE_COLS whitelist
- Fix: Updated fixtures to use real whitelist names (`popularity_rank`, `field_size`)
- Files modified: `tests/test_conformal_ev_model.py`
- Commit: `51c444a`

**2. [Rule 1 - Bug] Duplicate entry in FEATURE_COLS whitelist**
- Found during: Task 3 verification
- Issue: `norm_finish_logit_avg_race_rank` appeared twice in the whitelist
- Fix: Removed duplicate entry
- Files modified: `src/models/conformal_ev_model.py`
- Commit: `e8e6671`

**3. [Rule 2 - Missing] Unused POST_RACE_COLS import in training_pipeline.py**
- Found during: ruff verification
- Issue: After removing inline feature computation, `POST_RACE_COLS` import became unused
- Fix: Removed unused import
- Files modified: `src/pipelines/training_pipeline.py`
- Commit: `e48fd40`

### Pre-existing Issues (Out of Scope)

- N806 ruff warnings (X_train, BANDS, etc.) — pre-existing, not introduced by this plan
- E501 line too long in training_pipeline.py — pre-existing, not introduced by this plan
- F821 undefined IsotonicRegression — pre-existing, type annotation only

## Commits

| Commit | Message |
|--------|---------|
| `70d89e3` | fix(23-01): build_all() POST_RACE drop + ninki fallback removal (SAFE-01) |
| `51c444a` | fix(23-01): CQR whitelist FEATURE_COLS + EV correction odds fix (SAFE-01) |
| `e8e6671` | test(23-01): 3-layer CI leakage detection tests (SAFE-01) |
| `e48fd40` | style(23-01): remove unused POST_RACE_COLS import from training_pipeline |

## Self-Check: PASSED

All 7 modified/created files verified present. All 4 commits verified in git log. 1396 tests pass with 0 failures.
