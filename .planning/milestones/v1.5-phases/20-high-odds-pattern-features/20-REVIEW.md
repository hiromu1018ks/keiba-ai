---
phase: 20-high-odds-pattern-features
reviewed: 2026-05-09T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - scripts/analyze_high_odds.py
  - src/features/high_odds_features.py
  - src/features/horse_history_features.py
  - src/models/stage1_ability_model.py
  - tests/test_high_odds_features.py
  - tests/test_horse_history_features.py
findings:
  critical: 0
  warning: 4
  info: 4
  total: 8
status: issues_found
---

# Phase 20: Code Review Report

**Reviewed:** 2026-05-09
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Reviewed 6 files implementing HODDS-02/03/04 high-odds pattern features (class trajectory, form improvement rate, environment adaptability). The core feature computation logic in `high_odds_features.py` is well-structured with proper NaN handling. The integration into `horse_history_features.py` and `stage1_ability_model.py` is consistent. However, there are two no-op test methods that provide zero assertion coverage, a fragile variable scoping dependency, duplicated NaN-checking logic between modules, and an overly broad exception handler in the analysis script.

## Warnings

### WR-01: Two test methods are no-ops (assert True / bare pass)

**File:** `tests/test_high_odds_features.py:386-402` and `tests/test_high_odds_features.py:404-419`
**Issue:** `test_exp_count_accuracy` (line 402) ends with `assert True  # Redesign test below` -- this test always passes regardless of the function under test. `test_exp_count_three_matches` (line 419) ends with `pass` -- this test also always passes with no assertions. Both methods compute results but discard them, providing zero coverage for the `exp_count` accuracy scenario described in their docstrings. These are left-over scaffolding from test design iterations.

**Fix:**
Remove `test_exp_count_accuracy` entirely (it is superseded by `test_dist_change_exp_count` at line 421). Remove `test_exp_count_three_matches` or replace its `pass` with actual assertions:
```python
# test_exp_count_accuracy: delete entirely (dead test)

# test_exp_count_three_matches: either delete or fix:
def test_exp_count_three_matches(self):
    """Remove this test -- superseded by test_dist_change_exp_count."""
    pytest.skip("Superseded by test_dist_change_exp_count")
```

### WR-02: current_db variable defined in sibling if-block, used in dependent block

**File:** `src/features/horse_history_features.py:1147-1188`
**Issue:** `current_db` is assigned inside the `if hist_idx > 0 and horse_arrs is not None and "distance_bin" in horse_arrs:` block (line 1149-1154), but is referenced 41 lines later at line 1188 inside a separate `if` block for environment adaptability (`compute_env_adaptability`). While the env-adaptability condition is a strict superset of the distance_change condition (so `current_db` is guaranteed to be defined when the env block executes), this is a fragile coupling. If either condition changes independently (e.g., env adaptability drops the `"distance_bin" in horse_arrs` check), a `NameError` would occur at runtime.

**Fix:**
Extract `current_db` computation to before both if-blocks so it is unconditionally available:
```python
# Compute current_db before the distance_change block
if hasattr(row, "distance_bin") and not pd.isna(getattr(row, "distance_bin", None)):
    current_db = str(getattr(row, "distance_bin"))
else:
    current_db = _compute_distance_bin(
        getattr(row, "kyori", None), getattr(row, "surface", "")
    )

# Now distance_change block just uses current_db
if hist_idx > 0 and horse_arrs is not None and "distance_bin" in horse_arrs:
    last_db = str(horse_arrs["distance_bin"][history_mask][hist_start:hist_idx][-1])
    distance_change: float = 1.0 if current_db != last_db else 0.0
else:
    distance_change = float("nan")
```

### WR-03: Divergent _is_nan vs pd.notna in duplicated _class_level_from_values

**File:** `src/features/high_odds_features.py:62-77` vs `src/features/horse_history_features.py:64-68`
**Issue:** `_class_level_from_values` is duplicated across two modules. The `high_odds_features.py` version uses a custom `_is_nan()` that only checks `isinstance(value, float) and np.isnan(value)`, missing `None`, `pd.NA`, and numpy masked values. The `horse_history_features.py` version uses `pd.notna(grade_code)` which handles all of these. If a numpy object array contains `None` (not `np.nan`) in a `grade_code` slot, `high_odds_features._class_level_from_values` will call `str(None).strip()` producing `"None"`, which is not in `_CLASS_LEVEL_MAP`, and falls through to the jyoken_code path. The behavior is functionally similar but inconsistent.

**Fix:**
Replace the custom `_is_nan` check with `pd.notna` for consistency, or better, extract `_class_level_from_values` into a shared utility module:
```python
# Option A: Fix _is_nan to match pd.notna behavior
def _is_nan(value: object) -> bool:
    try:
        if isinstance(value, float) and np.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    return value is None  # Also handle None

# Option B (preferred): Import from a shared location to avoid duplication
```

### WR-04: Overly broad exception handler swallows SHAP analysis errors

**File:** `scripts/analyze_high_odds.py:242-243`
**Issue:** `except Exception as e:` catches all exceptions during SHAP analysis and silently continues with a warning log. This means any programming error (TypeError, KeyError, AttributeError from incorrect feature alignment, etc.) will be hidden, and the script will produce partial results without the user realizing the SHAP section is broken. For a one-off analysis script this is somewhat acceptable, but it risks producing misleading output where SHAP columns are all NaN.

**Fix:**
Narrow the exception type or add more specific error logging:
```python
except (lgb.LightGBMError, ValueError, FileNotFoundError) as e:
    logger.warning("SHAP analysis skipped due to model/data error: %s", e)
except Exception as e:
    logger.error("SHAP analysis failed with unexpected error: %s", e, exc_info=True)
    # Optionally: re-raise or set a flag in the output indicating SHAP failure
```

## Info

### IN-01: Redundant _to_float return logic

**File:** `src/features/high_odds_features.py:80-88`
**Issue:** `_to_float` converts to float, then checks `if np.isnan(result): return result` and then `return result` on the next line -- the NaN check is a no-op since both branches return the same value.

**Fix:**
Simplify to:
```python
def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
```

### IN-02: _to_float called twice per element in cond_match computation

**File:** `src/features/high_odds_features.py:390-392`
**Issue:** In the `cond_match` array comprehension, `_to_float(tc)` is called twice per element -- once for the NaN check and once for the comparison. For large arrays this doubles the conversion overhead.

**Fix:**
Use a single conversion:
```python
cond_match = np.array([
    (lambda v: not np.isnan(v) and v == current_track_condition)(_to_float(tc))
    for tc in track_condition_arr
])
```

### IN-03: Misleading docstring for compute_form_improvement_rate zscore_arr parameter

**File:** `src/features/high_odds_features.py:178`
**Issue:** The `zscore_arr` parameter is documented as "過去N走のタイムz-score配列 (低いほど良いタイム)" but the actual data passed from `horse_history_features.py:1041` is raw `harontimel3` values (not z-scores): `_fi_ht = horse_arrs["harontimel3"][valid_mask][start:idx].astype(float)`. The function works correctly either way since it just computes EMA improvement, but the parameter name and docstring are misleading.

**Fix:**
Either rename the parameter to `time_arr` and update the docstring, or actually pass z-score data:
```python
# Option: Fix the docstring to match reality
# In compute_form_improvement_rate:
Args:
    zscore_arr: 過去N走のハロンタイム値配列 (低いほど良いタイム)
```

### IN-04: Unused import in test_horse_history_features.py

**File:** `tests/test_horse_history_features.py:1054-1055`
**Issue:** Inside `test_harontimel5_avg_uses_5_races`, `import pandas as pd` and `import numpy as np` are re-imported at the method level even though they are already imported at the top of the file (lines 8-9). Same pattern at lines 1123-1128 in `test_harontime_late_trend_improving`.

**Fix:**
Remove the redundant local imports -- the module-level imports at lines 8-9 are sufficient.

---

_Reviewed: 2026-05-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
