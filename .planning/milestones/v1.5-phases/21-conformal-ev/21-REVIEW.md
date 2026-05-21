---
phase: 21-conformal-ev
reviewed: 2026-05-09T12:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - src/models/conformal_ev_model.py
  - src/models/__init__.py
  - src/domain/models.py
  - src/pipelines/training_pipeline.py
  - src/db/model_loader.py
  - src/backtest/race_predictor.py
  - src/models/ev_diagnostics.py
findings:
  critical: 2
  warning: 6
  info: 3
  total: 11
status: issues_found
---

# Phase 21: Code Review Report

**Reviewed:** 2026-05-09T12:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Phase 21 replaces RobustConfidenceEstimator with a CQR-based ConformalEVModel for EV prediction intervals. The core CQR implementation is mathematically sound and well-tested. However, the review found two critical bugs: (1) dead code after `return` in `_train_submodel()` that silently drops EV Isotonic Calibration wiring, and (2) an overly strict dtype filter that silently excludes `Int64`/`UInt32`/`Int32` pandas nullable-integer feature columns from CQR training. Several warnings relate to calibration data leakage, inconsistent feature filtering between training and inference, and uncalibrated legacy model fallback.

## Critical Issues

### CR-01: Dead code after `return` silently drops EV Isotonic Calibration wiring

**File:** `src/pipelines/training_pipeline.py:970-998`
**Issue:** The `_train_submodel()` method has a `return SubmodelSet(...)` statement at line 970-993, followed by lines 995-998 that attempt to wire `ev_isotonic_calibrator` and `ev_odds_band_scales` into the `ev_corrector`. These lines are **unreachable dead code** -- the `SubmodelSet` is returned before the wiring executes. This means the EV Isotonic Calibration model (`ev_isotonic_calibrator`) and odds band scales (`ev_odds_band_scales`) are stored on the `SubmodelSet` but **never wired into the `ev_corrector`** instance that actually applies them during `correct_ev()`.

The result: EV Isotonic Calibration (Phase 19, EVC-01/EVC-02) is silently non-functional in the training pipeline. Every EV correction passes through the un-calibrated path, producing less accurate EV estimates. This affects all downstream betting decisions.

Note: The local save path (`_save_models_local`) separately wires these into `ev_corr` at lines 726-728 in `load_from_dir()`, so loading from disk partially mitigates this, but the in-memory `TrainedModelsV5` returned by `run()` is broken.

**Fix:**
```python
        # At line 970, replace:
        return SubmodelSet(...)

        # Wire Isotonic + band scales into ev_corrector for correct_ev() to apply
        sub.ev_corrector.ev_isotonic_calibrator = ev_isotonic_calibrator
        sub.ev_corrector.ev_odds_band_scales = ev_odds_band_scales
        return sub

        # With:
        sub = SubmodelSet(
            market=market,
            stage1=stage1,
            # ... all existing fields ...
        )
        # Wire Isotonic + band scales into ev_corrector for correct_ev() to apply
        sub.ev_corrector.ev_isotonic_calibrator = ev_isotonic_calibrator
        sub.ev_corrector.ev_odds_band_scales = ev_odds_band_scales
        return sub
```

### CR-02: CQR feature dtype filter silently excludes pandas nullable-integer columns

**File:** `src/pipelines/training_pipeline.py:874-878`
**Issue:** The CQR feature column selection uses an explicit dtype whitelist:
```python
feature_cols = [
    c for c in df_oof.columns
    if c not in _non_feature_cols
    and df_oof[c].dtype in (np.float64, np.int64, float, int, np.float32, np.int32)
]
```

The pipeline itself converts nullable-integer (`Int64`) columns to `float64` at line 460-463 **before** the submodel training block, but this conversion only covers columns that exist at that point. Subsequent merge operations (jockey context at line 534, trainer context at line 539, jt_combo at line 547) can re-introduce `Int64`/`UInt32`/`Int32` dtype columns (e.g., `Int64Dtype()` from pandas). These columns pass through the later pipeline but are silently excluded from CQR feature columns by the strict dtype check.

In contrast, `ConformalEVModel.train()`'s internal fallback (line 113-118) uses a blacklist approach (`_NON_FEATURE_COLS` exclusion) which would include these columns. This creates a **feature inconsistency**: the CQR model trains with fewer features than the `ConformalEVModel`'s own default would select, and at inference time (`predict_interval()`), if `self.feature_cols` is not set (loaded model), the inference path may use a different feature set.

This can cause `KeyError` or silent wrong-dimension prediction at inference if feature columns mismatch.

**Fix:**
```python
# Replace the strict dtype whitelist with pandas numeric check:
feature_cols = [
    c for c in df_oof.columns
    if c not in _non_feature_cols
    and pd.api.types.is_numeric_dtype(df_oof[c])
]
```

## Warnings

### WR-01: CQR calibration on training data when train_ratio=1.0 (data leakage)

**File:** `src/models/conformal_ev_model.py:121-123`
**Issue:** When `train_ratio >= 1.0`, `df_train` and `df_val` are set to the same data (`df_calib`). The CQR nonconformity scores are then computed on the training data itself, which violates the exchangeability assumption of conformal prediction. The calibration quantiles will be overly optimistic (narrower intervals), potentially failing to achieve the target coverage rate in production.

The current pipeline caller at `training_pipeline.py:884` uses the default `train_ratio=0.8`, so this code path is not triggered in production. However, the docstring explicitly documents `train_ratio=1.0` as a valid option, and the fallback logic inside `train()` makes no warning about this case.

**Fix:** Either remove the `train_ratio=1.0` support, or add a warning log and document that it produces miscalibrated intervals:
```python
if train_ratio >= 1.0:
    logger.warning(
        "train_ratio=1.0: using same data for training and calibration. "
        "CQR intervals will be overfitted and may not achieve target coverage."
    )
```

### WR-02: Inconsistent feature filtering between pipeline training and ConformalEVModel default

**File:** `src/models/conformal_ev_model.py:113-118` vs `src/pipelines/training_pipeline.py:863-878`
**Issue:** The training pipeline builds `_non_feature_cols` (line 863-873) that includes columns like `"win_selection_edge"`, `"p_hit"`, `"e_return"`, `"p_corrected"`, `"e_corrected"`, `"grade_code"`, `"track_condition_code"`, `"kettonum"`. But `ConformalEVModel._NON_FEATURE_COLS` (line 21-38) does **not** include these columns. If `ConformalEVModel` is ever used without explicit `feature_cols` (e.g., the `load()` classmethod restores `feature_cols` from JSON, but if that file is missing or corrupted), the model would fall back to including these target-leakage columns as features.

Similarly, `"surface"` is in `_NON_FEATURE_COLS` but the pipeline also excludes it via `_non_feature_cols`. The pipeline additionally excludes `"kettonum"` which the model's `_NON_FEATURE_COLS` does not exclude.

**Fix:** Synchronize `_NON_FEATURE_COLS` in `conformal_ev_model.py` with the pipeline's `_non_feature_cols`, or document that `feature_cols` must always be explicitly provided during training.

### WR-03: Legacy uncalibrated ConformalEVModel used as passthrough in model_loader

**File:** `src/db/model_loader.py:641-647`
**Issue:** When loading from a legacy `confidence_params.json` file (fallback path), the code creates a `ConformalEVModel` with only `alpha` and `_calibrated=True` set, but no `q_low_model`, `q_high_model`, or calibration quantiles. When `predict_interval()` is called on this model, the check at line 248 (`not self._calibrated or self.q_low_model is None or self.q_high_model is None`) correctly falls through to the fallback path. However, `_calibrated=True` is misleading because the model is not actually calibrated. If any future code checks only `self._calibrated`, it would incorrectly assume the model is ready.

**Fix:** Set `_calibrated=False` for the legacy fallback, or add a comment explaining that `_calibrated` is intentionally `True` only for backward-compat shim `calibrate()` to work:
```python
conformal_ev = ConformalEVModel()
conformal_ev.alpha = conf_data["alpha"]
conformal_ev._calibrated = False  # No actual CQR models; will use fallback
```

### WR-04: ConformalEVModel.load() ignores file-not-found for individual CQR model files

**File:** `src/db/model_loader.py:196-197`
**Issue:** In the MLflow loading path, the code attempts to load CQR models with:
```python
obj.q_low_model = lgb.Booster(model_file=q_low_path) if Path(q_low_path).is_file() else mlflow.lightgbm.load_model(q_low_path)
```
If neither the local path nor the MLflow URI contains the model, this raises an exception (not caught by the outer try/except because the variable assignment itself fails before the `except Exception` block). The outer `except Exception` at line 207 does catch this, but then it falls through to the legacy `confidence_params.json` fallback, which may also fail, resulting in `conformal_ev = None`. This silent cascade is acceptable behavior but should be logged at a higher level than `logger.info`.

**Fix:** Add a `logger.warning` when the primary CQR loading fails, before attempting the legacy fallback:
```python
except Exception as e:
    logger.warning("CQR model files not found for %s (%s), trying legacy format", surface, e)
```

### WR-05: train() silently returns without setting models when samples < 200

**File:** `src/models/conformal_ev_model.py:142-148`
**Issue:** When `y_train` has fewer than 200 samples, `train()` logs a warning and returns early. The model remains uncalibrated (`self._calibrated = False`, `self.q_low_model = None`, `self.q_high_model = None`). The calling code in `training_pipeline.py:884` does not check whether `train()` succeeded before logging `_calibration_quantile_90/80` values at line 887-889. If `train()` silently skips, the log will show the initial values of `0.0` for both quantiles, which is misleading.

Similarly, if the calibration set has < 10 samples (line 195-201), `train()` returns after training the models but without setting `_calibrated = True`, leaving the models trained but unusable.

**Fix:** After `conformal_ev.train(...)`, add a guard:
```python
if not conformal_ev._calibrated:
    logger.warning("Conformal EV training incomplete for %s", surface)
    conformal_ev = None
```

### WR-06: place_df parameter in predict_interval() is unused for CQR computation

**File:** `src/models/conformal_ev_model.py:226-231` and `src/backtest/race_predictor.py:170`
**Issue:** The `predict_interval()` method accepts a `place_df` parameter but only applies CQR quantile prediction to the `win_df`. Place data is simply passed through with `EV_lower_place = EV_upper_place = place_ev` (line 324-329). The caller in `race_predictor.py:170` passes the same `df` as both arguments: `submodel.conformal_ev_model.predict_interval(df, df)`. The place interval is therefore a degenerate zero-width interval, which means place betting gets no uncertainty quantification. This is not a bug per se (it's documented as "simplified processing"), but it should be called out as a known limitation that could affect downstream betting quality.

**Fix:** Add a docstring note or comment that CQR is win-only and place intervals are identity pass-through. Consider adding `TODO: implement CQR for place EV` if planned.

## Info

### IN-01: Backward-compat shim methods add dead interface surface

**File:** `src/models/conformal_ev_model.py:63-94`
**Issue:** The `calibrate()` and `predict_lower_bound()` methods exist solely for backward compatibility with the old `RobustConfidenceEstimator` interface. Since no callers of these methods were found in the current codebase (grep confirmed zero external call sites), these methods add dead interface surface. If Plan 02 is complete, these shims should be removed in a future cleanup phase.
**Fix:** Add `# TODO: remove in Phase 22+ when all callers migrated` comments, or schedule removal.

### IN-02: Comment typo "モノトonicity" (mixed Japanese/English)

**File:** `src/models/conformal_ev_model.py:291`
**Issue:** The comment reads `# モノトonicity保証: q_low <= q_high` which mixes Japanese katakana "モノト" with English "onicity". Should be either `# モノトニシティ` or `# Monotonicity`.
**Fix:** Change to `# Monotonicity guarantee: q_low <= q_high`.

### IN-03: strategy_manifest.json modified as side-effect of model training

**File:** `src/pipelines/training_pipeline.py:1610-1623`
**Issue:** The `_save_models_local()` method modifies `data/strategy_manifest.json` as a side-effect of model saving, injecting CQR checksums into the manifest. This couples model training to strategy optimization artifacts and could cause unexpected behavior if the manifest has a different structure than expected. The broad `except Exception` at line 1620 silently swallows any parsing errors.
**Fix:** Consider separating manifest updates from model saving, or at minimum log at warning level when manifest update fails.

---

_Reviewed: 2026-05-09T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
