---
phase: 15-ev-filter-enhancement
reviewed: 2026-05-06T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - src/domain/models.py
  - src/pipelines/training_pipeline.py
  - src/backtest/race_predictor.py
  - tests/test_race_predictor.py
  - src/models/ev_diagnostics.py
  - tests/test_ev_diagnostics.py
findings:
  critical: 1
  warning: 5
  info: 3
  total: 9
status: issues_found
---

# Phase 15: Code Review Report

**Reviewed:** 2026-05-06T00:00:00Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Reviewed 6 source and test files for the EV Filter Enhancement phase. Found one critical bug where the dynamic EV threshold computation always uses a fallback value for the surface not being trained, meaning the cross-surface threshold is always the hardcoded default rather than data-driven. Additionally found several warnings including a closure-over-loop-variable issue, an unprotected division by zero path, and input mutation in a diagnostic function.

## Critical Issues

### CR-01: Dynamic EV threshold always uses fallback for the non-trained surface

**File:** `src/pipelines/training_pipeline.py:875-880`
**Issue:** `_train_submodel` is called once per surface with only that surface's data in `wsg_train_df`. Yet lines 875-880 unconditionally compute both `ev_threshold_turf` and `ev_threshold_dirt` from the same single-surface `wsg_train_df`. When training the turf submodel, `_compute_ev_threshold(wsg_train_df, surface="dirt", ...)` filters for `df_oof["surface"] == "dirt"` on a DataFrame that contains only turf rows, gets zero matches, and returns the fallback value 0.7. The same happens in reverse for dirt training. This means `ev_lower_threshold_dirt` on the turf `SubmodelSet` is always 0.7 (fallback), and `ev_lower_threshold_turf` on the dirt `SubmodelSet` is always 0.8 (fallback). At inference time in `get_win_candidates` (race_predictor.py:444-448), the threshold is read from the surface-specific submodel -- so turf races use the correctly computed turf threshold but the dirt submodel's turf threshold is always the fallback. This defeats the purpose of data-driven dynamic thresholds.

**Fix:**
```python
# Only compute the threshold for the surface being trained.
# Store only the relevant threshold; leave the other as a default
# that will be set correctly when the other surface trains.
ev_threshold_turf = fallback_turf if surface != "turf" else self._compute_ev_threshold(
    wsg_train_df, surface="turf", fallback=0.8,
)
ev_threshold_dirt = fallback_dirt if surface != "dirt" else self._compute_ev_threshold(
    wsg_train_df, surface="dirt", fallback=0.7,
)
```

Alternatively, compute thresholds after both submodels are trained using the combined OOF data, and assign to both `SubmodelSet` instances.

## Warnings

### WR-01: Closure captures loop-variable references in _nll grid search

**File:** `src/pipelines/training_pipeline.py:652-668`
**Issue:** Inside the `for a0, b0, g0 in iter_product(...)` loop, variables `logit_f`, `logit_m`, and `y_arr` are reassigned on each iteration. The `_nll` closure captures these by reference. Since `scipy_minimize` calls `_nll` synchronously within the same iteration, this works correctly in practice. However, the pattern is fragile -- if `scipy_minimize` were ever changed to defer execution, or if the loop body became async, all closures would share the final iteration's values. Additionally, `logit_f`, `logit_m`, and `y_arr` are identical across all iterations and should be computed once before the loop for clarity and efficiency.

**Fix:**
```python
logit_f = BenterCombination._logit(oof_p_fund)
logit_m = BenterCombination._logit(oof_p_market)
y_arr = oof_y.astype(float)

for a0, b0, g0 in iter_product(alpha_grid, beta_grid, gamma_grid):
    def _nll(params: np.ndarray, _lf=logit_f, _lm=logit_m, _y=y_arr) -> float:
        alpha, beta, gamma = params
        logit_c = alpha * _lf + beta * _lm + gamma
        ...
    ...
```

### WR-02: Wide bet probability can divide by zero

**File:** `src/backtest/race_predictor.py:864-867`
**Issue:** After the `fuku_a <= 0 or fuku_b <= 0` guard on line 861, the code computes `p_a = (edge_a + 1.0) / fuku_a`. While `fuku_a > 0` is guaranteed at this point, `edge_a` comes from `float(row_a[edge_col])` on line 864 without any NaN check. If `edge_a` is NaN (which happens when `place_selection_edge` could not be computed), the resulting `p_a` is NaN, `ev_wide` becomes NaN, and a Bet with `edge=NaN` is created. The downstream `sort(key=lambda b: b.edge)` on line 885 will raise TypeError when comparing NaN with float.

**Fix:**
```python
edge_a = float(row_a[edge_col])
edge_b = float(row_b[edge_col])
if np.isnan(edge_a) or np.isnan(edge_b):
    continue
p_a = (edge_a + 1.0) / fuku_a
p_b = (edge_b + 1.0) / fuku_b
```

### WR-03: compute_ev_diagnostics mutates input DataFrame

**File:** `src/models/ev_diagnostics.py:187`
**Issue:** When `EV_ACTUAL_COLUMN` is missing but `confirmed_odds` and `WIN_COLUMN` are present, the function does `df_oof = df_oof.copy()` and then adds `EV_ACTUAL_COLUMN`. However, the guard check `if EV_ACTUAL_COLUMN not in df_oof.columns` on line 185 uses the *original* `df_oof`. The `.copy()` on line 187 correctly prevents mutating the caller's DataFrame. However, the `valid_mask` on line 199 references `EV_PRED_COLUMN` ("ev_win_corrected"), but `_build_ev_df` in tests does not always include this column -- the test helper sets `"ev_win_corrected"` which does match. This is fine, but the function lacks defensive validation that `EV_PRED_COLUMN` exists in the DataFrame. If the caller passes a DataFrame without `ev_win_corrected`, `pd.to_numeric(df_oof[EV_PRED_COLUMN], ...)` on line 197 raises KeyError rather than a clear error message.

**Fix:**
```python
if EV_PRED_COLUMN not in df_oof.columns:
    logger.warning("EV diagnostics: missing prediction column %s", EV_PRED_COLUMN)
    result["error"] = "missing_prediction_column"
    return result
```

### WR-04: Edge computation produces NaN when fukuoddslow is NaN or p_place_combined is NaN

**File:** `src/backtest/race_predictor.py:192`
**Issue:** `df["edge_place"] = p_combined * df["fukuoddslow"] - 1.0` will produce NaN for the entire row when either `p_combined` or `fukuoddslow` is NaN. While NaN edges are handled downstream (e.g., `selection_edge.fillna(0.0)` in `get_place_candidates`), the `edge_place` column propagates into the returned DataFrame and could confuse downstream consumers that do not expect NaN. The same applies to `df["ev_place_direct"]` on line 193.

**Fix:**
```python
df["edge_place"] = p_combined.fillna(0.0) * df["fukuoddslow"].fillna(0.0) - 1.0
df["ev_place_direct"] = p_combined.fillna(0.0) * df["fukuoddslow"].fillna(0.0)
```

### WR-05: compute_ev_diagnostics missing validation for EV_PRED_COLUMN

**File:** `src/models/ev_diagnostics.py:197`
**Issue:** The function checks for `EV_ACTUAL_COLUMN` (line 185) and creates it if possible, but never checks whether `EV_PRED_COLUMN` ("ev_win_corrected") exists in the DataFrame. If the prediction column is missing, `pd.to_numeric(df_oof[EV_PRED_COLUMN], errors="coerce")` raises a `KeyError`. The function should handle this gracefully with a clear error message, matching the pattern used for `EV_ACTUAL_COLUMN`.

**Fix:** Add a guard before line 197:
```python
if EV_PRED_COLUMN not in df_oof.columns:
    logger.warning("EV diagnostics: missing prediction column %s", EV_PRED_COLUMN)
    result["error"] = "missing_prediction_column"
    return result
```

## Info

### IN-01: Redundant logit computation inside grid search loop

**File:** `src/pipelines/training_pipeline.py:654-656`
**Issue:** `BenterCombination._logit(oof_p_fund)` and `BenterCombination._logit(oof_p_market)` are computed identically on every iteration of the grid search loop (12 iterations with the current grid). These values are constant and should be hoisted before the loop.

**Fix:** Move lines 654-656 before the `for a0, b0, g0 in iter_product(...)` loop on line 652.

### IN-02: Test helper _build_ev_df uses hardcoded column names

**File:** `tests/test_ev_diagnostics.py:34-42`
**Issue:** `_build_ev_df` hardcodes column names like `"p_win_corrected"`, `"ev_win_corrected"`, `"confirmed_odds"` that must match the constants in `ev_diagnostics.py`. If the production constants change, tests will silently test wrong columns. Consider importing and using the constants from the module under test.

**Fix:** Import and use the module-level constants:
```python
from models.ev_diagnostics import EV_PRED_COLUMN, EV_ACTUAL_COLUMN, WIN_COLUMN, DATE_COLUMN
```

### IN-03: Broad except in win Benter grid search silently swallows errors

**File:** `src/pipelines/training_pipeline.py:683-684`
**Issue:** `except Exception: continue` silently swallows all exceptions from the optimization, including potential bugs (e.g., shape mismatches, memory errors). At minimum, this should log the exception at debug level.

**Fix:**
```python
except Exception as e:
    logger.debug("Win Benter grid point (%.1f, %.1f, %.1f) failed: %s", a0, b0, g0, e)
    continue
```

---

_Reviewed: 2026-05-06T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
