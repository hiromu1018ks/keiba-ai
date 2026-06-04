---
phase: 39-marketawarewincalibrator
reviewed: 2026-05-28T12:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - src/backtest/race_predictor.py
  - src/db/model_loader.py
  - src/domain/models.py
  - src/models/market_aware_win_calibrator.py
  - src/models/win_benter_gate.py
  - src/pipelines/training_pipeline.py
  - tests/test_domain.py
  - tests/test_market_aware_win_calibrator.py
  - tests/test_race_predictor.py
  - tests/test_win_benter_gate.py
findings:
  critical: 2
  warning: 5
  info: 3
  total: 10
status: issues_found
---

# Phase 39: Code Review Report

**Reviewed:** 2026-05-28T12:00:00Z
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Reviewed 10 files in scope for Phase 39 (MarketAwareWinCalibrator replacing WinBenterGate + WinSegmentCalibrator). The new `MarketAwareWinCalibrator` class is well-structured with solid design decisions (fixed schema one-hot encoding, WF C-selection, beta market guard, D-22 train-mode rejection). However, two critical bugs were found: (1) a potential ZeroDivisionError in `apply()` when all calibrated probabilities per race collapse to zero, and (2) a mutation of the input DataFrame in `train()` that violates the function's documented contract and causes silent data corruption to the caller's DataFrame. Additionally, the broad exception catch in `generate_win_oof_predictions` silently swallows real training failures, and three copies of `_walk_forward_race_splits` represent a maintenance hazard.

## Critical Issues

### CR-01: ZeroDivisionError in apply() when race probabilities sum to zero

**File:** `src/models/market_aware_win_calibrator.py:473-474`
**Issue:** The `apply()` method normalizes `p_win_combined` per race by dividing by the group sum. If `LogisticRegression.predict_proba` returns near-zero probabilities for all horses in a race (e.g., all logits are extremely negative), `race_sums` can be exactly 0.0, causing division by zero and producing `inf`/`nan` in `p_win_final`. Unlike the fallback in `race_predictor.py` line 276 which uses `.clip(lower=1e-10)`, this code has no such guard.
**Fix:**
```python
# Line 473-474: add clip to prevent ZeroDivisionError
race_sums = df.groupby("race_id", observed=True)["p_win_combined"].transform("sum")
df["p_win_final"] = df["p_win_combined"] / race_sums.clip(lower=1e-10)
```

### CR-02: train() mutates caller's DataFrame via conditional df["p_model"] assignment

**File:** `src/models/market_aware_win_calibrator.py:267-269`
**Issue:** The `train()` method conditionally assigns `df["p_model"] = df["p_win_oof"]` on the input DataFrame. Although the method copies `df` at line 268, the copy only happens *inside* the conditional branch `if "p_model" not in df.columns`. If the caller's DataFrame happens to have a `p_model` column already, no copy is made and the method proceeds to modify the caller's DataFrame (reading from it directly at line 275 via `build_feature_matrix(df)` and at line 272 extracting `y`). More critically, this conditional copy means the behavior is inconsistent depending on input columns. The `df.copy()` should be unconditional and placed at the top of the method, before any column assignment.

In the actual call chain from `training_pipeline.py:1344`, the `oof_cal_df` from `generate_win_oof_predictions` does include a `p_model` column (since it has `p_win_oof` and no `p_model`), so the copy *does* occur today. However, this is fragile and violates the principle that `train()` should not mutate its input.
**Fix:**
```python
# Line 266-269: move copy before the conditional
df = df.copy()  # Unconditional copy at top
if "p_model" not in df.columns and "p_win_oof" in df.columns:
    df["p_model"] = df["p_win_oof"]
```

## Warnings

### WR-01: Broad exception catch silently swallows OOF fold training failures

**File:** `src/models/win_benter_gate.py:182-183`
**Issue:** `generate_win_oof_predictions` catches all `Exception` instances with only a `logger.warning` and `continue`. This silently swallows real errors like `MemoryError`, data corruption, or LightGBM training failures. If all folds fail, the function returns a DataFrame with zero valid rows, and the caller (`training_pipeline.py:1342`) checks `len(oof_cal_df) >= 500` and silently skips calibrator training. The result is that the entire `MarketAwareWinCalibrator` is silently disabled without any clear indication of why. At minimum, the exception type should be narrowed, or the function should raise after N consecutive failures.
**Fix:** Narrow the exception type or add a failure counter:
```python
n_failed = 0
for train_idx, val_idx in splits:
    try:
        # ... training logic ...
    except (ValueError, RuntimeError) as exc:
        n_failed += 1
        logger.warning("Skipping Win OOF fold: %s", exc)
        if n_failed >= len(splits):
            raise RuntimeError(
                f"All {len(splits)} Win OOF folds failed; "
                "check input data quality"
            ) from exc
        continue
```

### WR-02: Duplicated _walk_forward_race_splits function across three modules

**File:** `src/models/market_aware_win_calibrator.py:31-72`, `src/models/win_benter_gate.py:27-64`, `src/pipelines/training_pipeline.py:202-239`
**Issue:** The identical `_walk_forward_race_splits` function is copy-pasted in three separate modules. Any bug fix or behavioral change must be applied to all three copies. This is a significant maintenance hazard. If one copy diverges (e.g., different `min_train_races` default), it creates subtle behavioral differences that are hard to diagnose.
**Fix:** Extract to a shared utility module (e.g., `src/utils/wf_splits.py`) and import from all three locations.

### WR-03: load_from_dir does not load target_encoder for MLflow path

**File:** `src/db/model_loader.py:97-400`
**Issue:** The `load()` method (MLflow path, lines 97-400) does not attempt to load `target_encoder`. It only loads `target_encoder` in the `load_from_dir()` path (lines 823-826). This means models loaded via MLflow will have `target_encoder=None` even when the encoder was trained and saved. The `training_pipeline.py` does save it to MLflow (line 2403-2407), so the data exists but is never loaded.
**Fix:** Add target_encoder loading in the MLflow `load()` method, similar to the local `load_from_dir()` pattern:
```python
# After line 399, before submodels[surface] = SubmodelSet(...)
target_encoder = None
te_path_candidate = f"runs:/{run_id}/target_encoder_{surface}.joblib"
try:
    te_path = mlflow.artifacts.download_artifacts(te_path_candidate)
    target_encoder = joblib.load(te_path)
except Exception:
    pass
```

### WR-04: _check_ratio_gates only logs, never acts on detected imbalances

**File:** `src/models/market_aware_win_calibrator.py:371-414`
**Issue:** The `_check_ratio_gates` method (D-05) detects year/surface actual-to-predicted ratio deviations exceeding 10%, but only logs at INFO level. There is no mechanism to prevent deployment when ratios are severely imbalanced. The method name suggests a "gate" (which implies it should block), but it is purely diagnostic. This creates a false sense of safety -- the D-05 gate check appears in the design but provides no actual protection.
**Fix:** Either rename to `_check_ratio_diagnostics` to clarify intent, or add a gate that sets `deployment_status = "shadow_only"` when deviation exceeds a threshold (e.g., 0.20).

### WR-05: compare_calibrations imports betacal but always falls through to manual

**File:** `src/models/win_benter_gate.py:362-377`
**Issue:** In `compare_calibrations`, when `betacal` is importable, the code fits `_BetaCal` and then assigns `p_beta` from it (line 367), but immediately overwrites `beta_cal` with a new `BetaCalibrationManual()` instance and fits that too (lines 369-370). The `p_beta` prediction from the original betacal `_BetaCal` is then overwritten on line 376 by the manual fallback's transform. So even when betacal is available, its predictions are never used -- only the manual version. This is dead code that wastes computation.
**Fix:** Either use the betacal prediction when available:
```python
if _raw is not None:
    p_beta = np.asarray(_raw.predict(p_val), dtype=float)
    beta_cal = _raw  # Keep the fitted betacal instance
    has_beta = True
```
Or remove the betacal import attempt entirely if the manual version is preferred.

## Info

### IN-01: TODO comments in race_predictor.py for regime reactivation

**File:** `src/backtest/race_predictor.py:991,996,1132`
**Issue:** Three `TODO` comments about re-enabling dynamic regime detection (`# TODO: Regime動的に戻す場合はコメントアウト解除`). Currently, `RegimeState.AGGRESSIVE` is hardcoded, which means all races use the aggressive strategy regardless of market conditions. The `RegimeDetector` model is trained but never consulted at inference time.
**Fix:** Track as a known technical debt item. When re-enabling dynamic regime, uncomment the lines and remove the hardcoded `AGGRESSIVE` assignment.

### IN-02: prob_source_after_segment dead variable in get_win_candidates

**File:** `src/backtest/race_predictor.py:645-670`
**Issue:** `prob_source_after_segment` is initialized to `None` (line 645) and never reassigned. The condition at line 669 `if prob_source_after_segment is not None:` is therefore always `False`, making the `prob_source = prob_source_after_segment` assignment dead code. This is a leftover from the WinSegmentCalibrator removal (CAL-04).
**Fix:** Remove the dead variable and the unreachable branch:
```python
# Remove line 645 and lines 669-670
# prob_source_after_segment = None  # DELETE
# if prob_source_after_segment is not None:  # DELETE
#     prob_source = prob_source_after_segment  # DELETE
```

### IN-03: Unused import of joblib in model_loader.py load() method

**File:** `src/db/model_loader.py:13`
**Issue:** `joblib` is imported at the module level but within the `load()` method (MLflow path), it is only used for PlaceAbilityModel loading (line 253). Most other joblib usage is in `load_from_dir()`. The import is not unused, but it is worth noting that the `load()` method does not use joblib for loading `isotonic_calibrator` (line 337), `ev_isotonic_calibrator` (line 358), or `temperature_scaler` -- these all use `joblib.load` but are imported at module level. This is fine architecturally but the load() method is 300+ lines long and could benefit from extraction into smaller methods.

---

_Reviewed: 2026-05-28T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
