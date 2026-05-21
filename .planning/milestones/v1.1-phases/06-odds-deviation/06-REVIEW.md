---
phase: 06-odds-deviation
reviewed: 2026-05-03T19:30:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - src/features/odds_deviation_features.py
  - src/models/two_stage_return_model.py
  - src/models/robust_confidence_estimator.py
  - src/models/win_selection_gate.py
  - src/pipelines/training_pipeline.py
  - src/backtest/race_predictor.py
  - tests/test_odds_deviation.py
  - tests/test_robust_confidence_estimator.py
  - tests/test_race_predictor.py
  - tests/test_backtest_engine.py
  - tests/test_win_feature_analysis.py
findings:
  critical: 1
  warning: 5
  info: 3
  total: 9
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-05-03T19:30:00Z
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Reviewed 11 source files for Phase 6 (odds deviation EV features). The implementation adds odds deviation features (deviation_rank, deviation_zscore) to the ML pipeline and a conformal confidence scoring system for EV interval estimation. One critical class-level shared mutable alias bug was found in `PlaceTwoStageModel`, along with several warnings including a `DataFrame.get()` misuse pattern in fallback paths and inconsistent upper bound logic in the confidence estimator.

## Critical Issues

### CR-01: PlaceTwoStageModel.FEATURE_COLS is a shared mutable alias of RETURN_FEATURE_COLS

**File:** `src/models/two_stage_return_model.py:395`
**Issue:** `FEATURE_COLS` is assigned directly to `RETURN_FEATURE_COLS` (same list object). Any mutation to one list mutates the other. If code elsewhere appends or removes items from `FEATURE_COLS`, `RETURN_FEATURE_COLS` is silently corrupted and vice versa. This is especially dangerous because `WinTwoStageModel.remove_noise_features()` mutates `cls.FEATURE_COLS` in-place (line 139) -- if `PlaceTwoStageModel` ever calls a similar method or if the lists are accidentally modified, both the hit model and return model feature lists become corrupted silently.

Verified at runtime:
```python
PlaceTwoStageModel.FEATURE_COLS is PlaceTwoStageModel.RETURN_FEATURE_COLS  # True
PlaceTwoStageModel.FEATURE_COLS.append('test_mutated')
'test_mutated' in PlaceTwoStageModel.RETURN_FEATURE_COLS  # True
```

**Fix:**
```python
# Line 394-395: Change from alias to independent copy
    # Backward compat: FEATURE_COLS returns a copy of the return model list
    FEATURE_COLS: list[str] = list(RETURN_FEATURE_COLS)
```

## Warnings

### WR-01: RobustConfidenceEstimator fallback uses DataFrame.get() incorrectly for scalar default

**File:** `src/models/robust_confidence_estimator.py:116-120`
**Issue:** In the uncalibrated fallback path, `win_df.get("ev_win_corrected", 0.0)` returns the scalar `0.0` when the column does not exist. While pandas silently broadcasts the scalar during assignment (resulting in a column of `0.0`s), this is fragile and inconsistent with the rest of the codebase which uses `pd.to_numeric` + `fillna`. If `ev_win_corrected` exists but contains NaN, `DataFrame.get()` returns the entire Series (not the scalar default), so the NaN values propagate silently rather than being filled with 0.0.

**Fix:**
```python
        if not self._calibrated:
            logger.warning("RobustConfidenceEstimator not calibrated, using EV as bounds")
            win_df = win_df.copy()
            place_df = place_df.copy()
            win_df["EV_lower_win_corrected"] = pd.to_numeric(
                win_df.get("ev_win_corrected", pd.Series(0.0, index=win_df.index)),
                errors="coerce",
            ).fillna(0.0)
            win_df["EV_upper_win_corrected"] = win_df["EV_lower_win_corrected"]
            win_df["conformal_confidence_score"] = 0.0
            place_df["EV_lower_place"] = pd.to_numeric(
                place_df.get("ev_place_corrected", pd.Series(0.0, index=place_df.index)),
                errors="coerce",
            ).fillna(0.0)
            place_df["EV_upper_place"] = place_df["EV_lower_place"]
            return win_df, place_df
```

### WR-02: Upper bound uses np.maximum instead of np.minimum -- inconsistent conservatism with lower bound

**File:** `src/models/robust_confidence_estimator.py:166`
**Issue:** The lower bound uses `np.minimum(cp_lower, rolling_lower)` (conservative: picks the more pessimistic/lower bound). The upper bound uses `np.maximum(cp_upper, rolling_upper)` (picks the higher upper bound). If the design intent is "conservative intervals" (Rule 4), the upper bound should also use `np.minimum` to produce narrower, more conservative intervals. The current asymmetric logic means the interval is wide (not conservative on the upper end), which contradicts the docstring "min(CP, Rolling_Quantile) for lower bound" philosophy. While this may be intentional for capture rate, it is inconsistent with the stated Rule 4 approach.

**Fix:** If conservative intervals are intended:
```python
        upper = np.minimum(cp_upper, rolling_upper)
```
Or document that the upper bound intentionally uses the wider estimate.

### WR-03: race_predictor.assigns place_df columns using .values without index alignment check

**File:** `src/backtest/race_predictor.py:153`
**Issue:** `df["EV_lower_place"] = place_df["EV_lower_place"].values` uses `.values` which strips the index and relies on positional alignment. If `predict_interval` ever returns a DataFrame with different row ordering (e.g., due to internal sorting or filtering), the values would be misaligned silently. The `predict_interval` method currently preserves ordering via `.copy()`, so this is not an active bug but is fragile.

**Fix:**
```python
        if "EV_lower_place" in place_df.columns:
            # Use index-aligned assignment instead of positional .values
            df = df.assign(EV_lower_place=place_df["EV_lower_place"].reindex(df.index).values)
```
Or use merge on index if `predict_interval` ever reorders.

### WR-04: WinTwoStageModel.remove_noise_features mutates class-level list -- thread-safety hazard

**File:** `src/models/two_stage_return_model.py:123-148`
**Issue:** `remove_noise_features()` mutates `cls.FEATURE_COLS` in-place. The docstring warns about thread safety, but the method is still accessible and mutates global state. In the training pipeline, `ThreadPoolExecutor` is used for parallel surface training (line 208). If both surfaces call `remove_noise_features()` concurrently, they will race on the same class variable. The code provides a thread-safe alternative (`get_filtered_feature_cols`) but the dangerous method remains the default pattern.

**Fix:** Consider making `remove_noise_features()` deprecated or having it raise if called from within a thread context. At minimum, the pipeline should be audited to confirm `remove_noise_features` is never called from within the `ThreadPoolExecutor` block.

### WR-05: RobustConfidenceEstimator alpha-scaling uses sqrt heuristic without theoretical justification

**File:** `src/models/robust_confidence_estimator.py:151-157`
**Issue:** When `primary_alpha != self.alpha`, the quantile is scaled by `sqrt(self.alpha / primary_alpha)`. This is a Gaussian approximation heuristic that does not hold for Conformal Prediction intervals in general. For non-Gaussian residual distributions (common in betting EV), this scaling can produce miscalibrated intervals. The code does not validate or warn when this scaling is applied, and no test exercises the scaling path (all tests use `alpha=0.1` which matches the default `self.alpha`).

**Fix:** Add a warning when the scaling is applied, and consider recalculating from raw residuals instead:
```python
            if abs(primary_alpha - self.alpha) > 1e-9:
                logger.warning(
                    "Scaling CP quantile by sqrt(%.3f/%.3f) -- Gaussian approximation may be invalid",
                    self.alpha, primary_alpha,
                )
```

## Info

### IN-01: Unused seed parameter in _train_valid_split

**File:** `src/models/two_stage_return_model.py:20`
**Issue:** The `seed` parameter in `_train_valid_split()` is declared but never used (`# noqa: ARG001`). The function performs time-series split (first 80% / last 20%) so a seed is not needed. The parameter exists only for API compatibility but may confuse callers into thinking the split is randomized.
**Fix:** Remove the `seed` parameter or rename it to `_seed` to signal it is intentionally unused.

### IN-02: compute_odds_deviation_features has redundant early-return paths

**File:** `src/features/odds_deviation_features.py:22-33`
**Issue:** The function checks `ratio is None` (line 23) and then checks `"race_id" not in df.columns` (line 30) with nearly identical NaN-filling early returns. These could be consolidated into a single check for readability.
**Fix:** Combine the two early-return blocks into one.

### IN-03: Test file test_race_predictor.py re-assigns submodel.confidence.predict_interval return value on line 95-98

**File:** `tests/test_race_predictor.py:93-98`
**Issue:** `submodel.confidence.predict_interval.return_value` is assigned twice in `test_predict_returns_dataframe_with_ev_columns` (lines 91-94 then 95-98). The second assignment silently overwrites the first. This is dead code that suggests a copy-paste artifact.
**Fix:** Remove the duplicate assignment (lines 95-98).

---

_Reviewed: 2026-05-03T19:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
