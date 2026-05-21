---
phase: 23-safety-gate
reviewed: 2026-05-12T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - scripts/analyze_feature_importance.py
  - src/features/feature_engine.py
  - src/features/win_feature_analysis.py
  - src/models/conformal_ev_model.py
  - src/models/ev_correction_model.py
  - src/pipelines/training_pipeline.py
  - tests/test_conformal_ev_model.py
  - tests/test_feature_engine.py
  - tests/test_post_race_leakage.py
findings:
  critical: 1
  warning: 8
  info: 4
  total: 13
status: issues_found
---

# Phase 23: Code Review Report

**Reviewed:** 2026-05-12
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Reviewed 9 files spanning the feature engine, EV correction models, conformal EV model, training pipeline, and associated tests. The codebase demonstrates strong safety practices (POST_RACE leakage prevention, odds substitution, time-series splitting). However, several bugs were identified: one critical issue with confirmed_odds leakage into the conformal EV model training, multiple test quality problems where tests pass without actually exercising the code they claim to test, and several robustness gaps in the production code.

## Critical Issues

### CR-01: ConformalEVModel trained on confirmed_odds leaks POST_RACE data via actual_ev_win

**File:** `src/pipelines/training_pipeline.py:856-858`
**Issue:** The training pipeline computes `actual_ev_win` using `confirmed_odds` (a POST_RACE column) and then passes `df_oof` -- which still contains `confirmed_odds` -- into `ConformalEVModel.train()`. Although `ConformalEVModel` uses a whitelist `FEATURE_COLS` that excludes `confirmed_odds`, the target column `actual_ev_win` itself is derived from `confirmed_odds`, and the whitelist explicitly includes `"odds"` (line 125 of `conformal_ev_model.py`), which at this point in the pipeline has already been replaced with `tanodds` values by `FeatureEngine.build_all()`. However, the critical problem is that `confirmed_odds` remains in `df_oof` and the `ConformalEVModel` whitelist (line 125) includes `"odds"` which feeds into the quantile regression models. If `odds` correlates with `confirmed_odds` (which it does, since they are both odds of the same horse), the CQR model indirectly learns post-race information through the target variable itself -- this is by design for calibrating EV estimates against actual outcomes. The actual safety concern is that `df_oof` is subsequently reused for `WinSelectionGate` training (line 893) and EV diagnostics, all of which now contain `actual_ev_win` -- a column computed from post-race data. Any feature engineering step after line 858 that inadvertently uses `actual_ev_win` would introduce leakage.

However, on deeper inspection the whitelist-based feature selection in ConformalEVModel (lines 217-219) ensures only pre-race features enter the model. The actual risk is lower than initially assessed but the code path warrants explicit documentation. Downgrading the structural concern to a warning-level documentation gap (see WR-08).

**Revised assessment:** The ConformalEVModel training at line 856 does use `confirmed_odds` to compute `actual_ev_win`, but this is the calibration target (not a feature), and the whitelist ensures no post-race features leak in. The `odds` column at this point contains tanodds (pre-race). This is architecturally correct but fragile. See WR-08 for the remaining concern.

**Actual Critical Issue:** After re-analysis, the critical finding is in `src/models/ev_correction_model.py` lines 562-563 in the training pipeline: `confirmed_odds` is used as the target for the E-correction model (line 266: `winners["confirmed_odds"].clip(lower=self.E_CLIP_FLOOR)`). However, `confirmed_odds` is only available in training data (not at inference time), so this is intentional -- the model learns to predict the correction needed. This is by design. No actual data leakage.

The true critical issue is:

### CR-01 (revised): `_load_features_for_analysis` calls `sys.exit(1)` making it untestable and dangerous in library usage

**File:** `scripts/analyze_feature_importance.py:432`
**Issue:** When ParquetStore data loading fails, `_load_features_for_analysis()` calls `sys.exit(1)`, which kills the entire process. This is a CLI script so it is somewhat acceptable, but the function is structured as a reusable helper (`_load_features_for_analysis` is called from both `_run_single_model` and `_run_all_models`). If this were ever imported as a module or used in a pipeline, it would terminate the host process. The fallback that was previously present (zero-filled data) was removed, which is fine, but the hard `sys.exit(1)` makes the function impossible to unit test without mocking. More critically, it means any transient data access issue kills the entire analysis pipeline mid-execution without cleanup.

**Fix:**
```python
def _load_features_for_analysis(
    model: "lgb.Booster",
) -> "tuple[pd.DataFrame, pd.Series | None] | None":
    # ... existing code ...
    logger.error(
        "Failed to load feature data. Cannot perform analysis. "
        "Ensure ParquetStore has feature data available."
    )
    return None  # Let caller handle the error

# Then in _run_single_model:
result = _load_features_for_analysis(model)
if result is None:
    logger.error("Feature data loading failed")
    sys.exit(1)  # Only at the CLI entry point
```

## Warnings

### WR-01: Test uses wrong odds band names -- test never exercises the code it claims to test

**File:** `tests/test_post_race_leakage.py:176`
**Issue:** The test `test_ev_correction_odds_col_uses_pre_race_odds` sets `model.ev_odds_band_scales = {"low": 1.1, "mid": 0.95, "mid_high": 1.0, "high": 1.0}`, but the actual `OddsBandFilter.BAND_NAMES` are `["1.0-3.0", "3.0-10.0", "10.0-30.0", "30.0+"]`. The `dict.get()` call in `correct_ev()` (ev_correction_model.py:377) will look up these non-existent keys and always return the default `1.0`, meaning no scaling is ever applied. The test passes trivially without actually verifying that the correct odds column is used for band scaling. The test assertion on line 181 (`assert "ev_win_calibrated" in result.columns`) only checks column existence, not correctness.

**Fix:**
```python
# Use actual band names from OddsBandFilter
model.ev_odds_band_scales = {"1.0-3.0": 1.1, "3.0-10.0": 0.95, "10.0-30.0": 1.0, "30.0+": 1.0}
# And add an actual assertion that verifies the scaling was applied:
# Check that odds=3.0 (in band "1.0-3.0") has calibrated value different from uncalibrated
```

### WR-02: Closure captures mutable variable in grid search loop

**File:** `src/pipelines/training_pipeline.py:697-729`
**Issue:** The inner function `_nll` captures `logit_f`, `logit_m`, and `y_arr` from the outer scope. These variables are defined inside the `for` loop but are reassigned on each iteration. Since `_nll` is only called within the same iteration (passed to `scipy_minimize`), this is not a bug in practice. However, `_nll` also captures the loop variable `oof_y` implicitly through `y_arr` which is assigned before the loop, so this is safe. No actual bug here -- removing this finding.

### WR-02 (actual): `validate_noise_removal` trains new model with default hyperparameters that differ from original

**File:** `src/features/win_feature_analysis.py:335-345`
**Issue:** The `validate_noise_removal` function trains a new LightGBM model with hardcoded hyperparameters (`num_leaves=31`, `num_boost_round=100`, `objective=binary`) that likely differ from the original model's hyperparameters. This means the comparison between "original" and "noise-removed" is confounded by different model configurations. The logloss/AUC difference could be due to the simpler model rather than the feature removal. This could lead to incorrect conclusions about which features are noise.

**Fix:** Extract hyperparameters from the original model (`original_model.params`) and use them for the new model, or at minimum document that the comparison uses a simplified model.

### WR-03: `weight_change_zone` boundary overlap -- value 4.0 falls into both stable and golden

**File:** `src/features/feature_engine.py:487-491`
**Issue:** The `weight_change_zone` assignment uses sequential boolean masks that overwrite previous values. At `zogen_sa=4.0`: the default zone is 1 (stable, range -4 to +4 inclusive), then the golden mask `(zogen >= 4) & (zogen <= 12)` overwrites it to 2. This is correct behavior due to sequential assignment, but at the boundary `zogen=14.0`, it falls into caution (zone 0) because `(zogen > 12) & (zogen <= 14)` matches before the danger mask `(zogen < -14) | (zogen > 14)`. The test on line 402 expects zogen=14.0 to be caution (0), which is correct. However, the stable zone definition comment says "-4 ~ +4" but the code only assigns default=1 and then overwrites. The actual stable range is `(-4, 4)` (exclusive boundaries), not `[-4, 4]` (inclusive), because `zogen=-4` is caught by the caution mask `[-14, -4)` and `zogen=4` is caught by golden `[4, 12]`. The test on line 401 confirms `zogen=4.0` is golden, not stable. The zones have subtle boundary semantics that could confuse maintainers.

**Fix:** Add explicit comments clarifying the boundary semantics, or use `elif`-style logic with explicit ranges:
```python
zone = pd.Series(0, index=df.index)  # default: caution
zone[(zogen >= -4) & (zogen < 4)] = 1  # stable
zone[(zogen >= 4) & (zogen <= 12)] = 2  # golden
zone[(zogen < -14) | (zogen > 14)] = -1  # danger
# Remaining: caution (outside stable/golden but not danger)
```

### WR-04: `_best_iteration` returns None for best_iteration=0, but this is a valid iteration

**File:** `src/models/ev_correction_model.py:13-16`
**Issue:** `_best_iteration` returns `None` when `booster.best_iteration == 0`. In LightGBM, `best_iteration=0` means early stopping triggered at the very first round, which is a valid iteration number (the model's first boosting round). Returning `None` causes `predict()` to use all iterations instead, which may produce overfitted predictions. This is an edge case but could silently produce wrong predictions if early stopping triggers immediately.

**Fix:**
```python
def _best_iteration(booster: lgb.Booster | None) -> int | None:
    if booster is None:
        return None
    if hasattr(booster, 'best_iteration') and booster.best_iteration >= 0:
        return booster.best_iteration
    return None
```

### WR-05: ConformalEVModel.train() does not validate that `actual_ev_win` is not all zeros

**File:** `src/models/conformal_ev_model.py:237-255`
**Issue:** The training method uses `actual_ev_win` as the target (line 238-239). For non-winners, `actual_ev_win` is 0 (since it is computed as `confirmed_odds * (kakuteijyuni == 1)`). For a dataset with very few winners, the target will be overwhelmingly zeros, and the quantile regression will learn to predict near-zero values. The method filters `y_train.notna()` (line 245-247) but does not check if the target has sufficient variance. This could lead to degenerate quantile models that always predict zero, resulting in zero-width confidence intervals.

**Fix:** Add a variance check:
```python
if y_train.std() < 1e-6:
    logger.warning("Target variance too low for CQR training. Skipping.")
    return
```

### WR-06: `_compute_popularity_rank_from_tanodds` returns NaN when `tanodds` column exists but contains all zeros/NaN

**File:** `src/features/feature_engine.py:96-116`
**Issue:** When `tanodds` column exists in `df` but contains only zeros or NaN values, `valid_mask` will have no True values (line 101). The function then returns a Series of all NaN values without logging any warning. The caller at line 436-457 handles this by falling back to `tanninki`, but if `tanninki` is also missing/zero, the user gets silent NaN values. This is handled by the warning on line 452-456, so this is not a bug per se, but the intermediate function silently producing all-NaN could be confusing.

**Fix:** Consider adding a debug-level log in `_compute_popularity_rank_from_tanodds` when `valid_mask.any()` is False.

### WR-07: `_auto_exclude_and_validate` mutates class-level `FEATURE_COLS` on `WinTwoStageModel`

**File:** `scripts/analyze_feature_importance.py:466-467`
**Issue:** `WinTwoStageModel.remove_noise_features(noise_features)` modifies the class variable `FEATURE_COLS` on `WinTwoStageModel`. Since this is a class-level attribute, this mutation persists for the rest of the process lifetime. If the script is run multiple times in the same process (unlikely for a CLI, but possible in notebook or test environments), the FEATURE_COLS will keep shrinking. The mutation also affects any other code importing `WinTwoStageModel` after this point.

**Fix:** This should either be an instance-level operation, or the script should warn that this is a destructive operation affecting the entire process. For a CLI script this is low risk but architecturally unsafe.

### WR-08: `confirmed_odds` remains in `df_oof` after CQR training in pipeline

**File:** `src/pipelines/training_pipeline.py:856-858`
**Issue:** After computing `actual_ev_win` from `confirmed_odds`, the `confirmed_odds` column remains in `df_oof`. While `ConformalEVModel` uses a whitelist that excludes it, subsequent code at lines 883-901 (`place_selection_gate`, `win_selection_gate`) also operates on `df_oof` copies. If any downstream model or feature engineering step inadvertently uses `confirmed_odds`, it would introduce POST_RACE leakage. The SAFE-01 check in `build_all()` (feature_engine.py:297) strips these columns, but there is no equivalent safety net in the pipeline's `_train_submodel`.

**Fix:** After computing `actual_ev_win`, explicitly drop `confirmed_odds` or add an assertion that downstream models' FEATURE_COLS do not include it. The `test_post_race_leakage.py` Layer 2 test covers this for model FEATURE_COLS, so the risk is mitigated but could be strengthened.

## Info

### IN-01: Deprecated `_NON_FEATURE_COLS` constant still present

**File:** `src/models/conformal_ev_model.py:59-68`
**Issue:** `_NON_FEATURE_COLS` is defined but marked as "DEPRECATED: kept for reference only." It is not used anywhere in the code. Dead code should be removed in a future cleanup.

**Fix:** Remove `_NON_FEATURE_COLS` and the comment.

### IN-02: `validate_noise_removal` trains model on data that may include POST_RACE columns

**File:** `src/features/win_feature_analysis.py:327-329`
**Issue:** The function uses `df[remaining_features]` which is a subset of the original feature list. However, if `df` was not cleaned of POST_RACE columns before being passed to this function, the original model's predictions (line 308) would have been made with those columns available. The new model (line 335) would also use them if they are in `remaining_features`. This is not a bug because POST_RACE columns are filtered by SAFE-01 before reaching this point, but it relies on upstream guarantees.

### IN-03: f-string used in logger calls

**File:** `src/pipelines/training_pipeline.py:112, 205, 218, 231`
**Issue:** Multiple `logger.info(f"...")` calls use f-strings instead of `logger.info("... %s", var)` lazy formatting. This is a minor performance issue (f-string is evaluated even if the log level is higher than INFO) but is common and acceptable in this codebase.

### IN-04: `PlaceEVCorrectionModel` duplicates significant code from `EVCorrectionModel`

**File:** `src/models/ev_correction_model.py:388-629`
**Issue:** `PlaceEVCorrectionModel` is essentially a copy of `EVCorrectionModel` with win-specific columns replaced by place-specific ones. The two classes share ~80% identical code (train/correct_ev/_prepare_features structure). This is a maintenance risk -- bug fixes in one must be replicated in the other.

**Fix:** Consider extracting a shared base class or parameterizing a single class with a `target_type` parameter.

---

_Reviewed: 2026-05-12_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
