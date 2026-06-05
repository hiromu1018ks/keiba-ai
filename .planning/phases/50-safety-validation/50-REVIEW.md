---
phase: 50-safety-validation
reviewed: 2026-06-05T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - scripts/run_track_condition_ic_eval.py
  - scripts/validate_track_condition_nan.py
  - tests/test_feature_routing_audit.py
  - tests/test_post_race_leakage.py
  - tests/test_track_condition_ic.py
  - tests/test_track_condition_nan.py
  - tests/test_track_condition_routing.py
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 50: Code Review Report

**Reviewed:** 2026-06-05
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Reviewed 7 files from Phase 50 (safety-validation): 2 diagnostic scripts and 5 test files. The overall test structure is sound with 17+16 new tests covering track condition feature routing, NaN diagnostics, and IC evaluation. However, one critical bug exists in `test_track_condition_nan.py` where the test's NaN rate calculation diverges from the production script's surface-aware logic, meaning the tests validate a different algorithm than what runs in production. Several warnings relate to script robustness and test assertion completeness.

## Critical Issues

### CR-01: Test NaN rate uses global NaN count instead of surface-filtered NaN count

**File:** `tests/test_track_condition_nan.py:63`
**Issue:** The test helper `_compute_nan_rate_with_thresholds` computes `nan_count = int(df[col].isna().sum())` across ALL rows (turf + dirt), then divides by a surface-specific denominator. The production script `validate_track_condition_nan.py:144` correctly uses `int(feature_series.isna().sum())` where `feature_series` is already filtered to surface-relevant rows (e.g., `df.loc[dirt_mask, col]` for dirt features).

This means the test is validating a different formula than the script implements. For a dirt feature like `dirt_moisture_x_kyakusitu`, the test counts NaN on both turf and dirt rows but divides only by dirt rows. The script counts NaN only on dirt rows and divides by dirt rows. If turf rows happen to contain NaN for the dirt feature (expected by design -- dirt features are NaN on turf), the test inflates the NaN count.

The tests pass currently because the test data is constructed to avoid NaN in the cross-surface rows (e.g., dirt features get `[1.0] * 100` on turf rows). This masks the divergence from production logic.

**Fix:**
```python
# tests/test_track_condition_nan.py, around line 59-64
# Replace global NaN count with surface-filtered NaN count:

if col_lower.startswith("dirt_") or "moisture" in col_lower:
    if "track_front_bias" in col_lower or "kickback" in col_lower:
        denominator = total_rows
        nan_count = int(df[col].isna().sum())
    else:
        denominator = dirt_rows
        nan_count = int(df.loc[df[surface_col] == "dirt", col].isna().sum())
elif col_lower.startswith("turf_") or col_lower.startswith("cushion_"):
    denominator = turf_rows
    nan_count = int(df.loc[df[surface_col] == "turf", col].isna().sum())
else:
    denominator = total_rows
    nan_count = int(df[col].isna().sum())
```

## Warnings

### WR-01: Test feature classification uses substring match `"moisture" in col_lower` instead of prefix match

**File:** `tests/test_track_condition_nan.py:41`
**Issue:** The test classifies features as dirt-type using `"moisture" in col_lower`, which would match any feature name containing the word "moisture" anywhere (e.g., a hypothetical `no_moisture_flag`). The production script uses prefix-based classification with `_DIRT_PREFIXES = ("dirt_moisture_", "moisture_")`. While these happen to produce the same results for the current feature set, the test's logic is semantically different and could diverge if new features are added.

**Fix:** Use the same `_classify_feature()` function from the production script, or replicate the prefix-based logic:
```python
# Instead of:
if col_lower.startswith("dirt_") or "moisture" in col_lower:
# Use:
if col_lower.startswith("dirt_moisture_") or col_lower.startswith("moisture_"):
```

### WR-02: `run_track_condition_ic_eval.py` injects sys.path at module level, executed on import

**File:** `scripts/run_track_condition_ic_eval.py:25-26`
**Issue:** The script modifies `sys.path` at module level:
```python
ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(ROOT) / "src"))
```
When `test_track_condition_ic.py` imports from this script (`from run_track_condition_ic_eval import ...`), this path manipulation executes as a side effect. Since the package is installed in editable mode, this is harmless but fragile -- it duplicates the path setup and could cause import ordering issues in edge cases. The `validate_track_condition_nan.py` script does not do this and works correctly via the editable install.

**Fix:** Remove the `sys.path.insert` from `run_track_condition_ic_eval.py` since the editable install already provides the correct import path. If the script must also work without the editable install, guard it with a try/except import.

### WR-03: Tier aggregation test does not verify `mean_abs_c_ic` value

**File:** `tests/test_track_condition_ic.py:219-222`
**Issue:** The test `test_tier_aggregation_basic` constructs a `per_feature` dict with three numeric features having C-orthogonal IC values of 0.010, 0.003, and -0.007. The expected `mean_abs_c_ic` should be `mean(0.010, 0.003, 0.007) = 0.00667`, and `signal_count` should be 2. The test only asserts `signal_count == 2` but does not verify `mean_abs_c_ic`. A bug in the mean calculation would go undetected.

**Fix:** Add assertion for `mean_abs_c_ic`:
```python
assert t1_t2["mean_abs_c_ic"] == pytest.approx(0.006667, abs=1e-4), (
    f"Expected mean_abs_c_ic ~0.00667, got {t1_t2['mean_abs_c_ic']}"
)
```

### WR-04: Surface stratification IC computes NaN count from `np.isfinite` but uses `spearmanr` on filtered data without checking target NaN

**File:** `scripts/run_track_condition_ic_eval.py:342-348`
**Issue:** The surface-stratified IC calculation filters only by `np.isfinite(surf_feature)` when selecting rows for `spearmanr`. The target (`surf_target`) is derived from a binary column (0.0 or 1.0) and should never be NaN, but if the merge produces unexpected NaN in `kakuteijyuni`, the `pd.to_numeric(..., errors="coerce") == 1` would produce `False` (0.0) for NaN values rather than NaN, so this is actually safe. However, the same pattern in `_compute_c_orthogonal_ic` and `_compute_univariate_ic` uses a joint validity mask (`np.isfinite(feature) & np.isfinite(target)`), while the surface stratification code does not. This inconsistency suggests the surface stratification code was written with a different assumption.

**Fix:** Use consistent validity filtering in the surface stratification block:
```python
valid_mask = np.isfinite(surf_feature) & np.isfinite(surf_target)
n_valid = int(valid_mask.sum())
if n_valid >= 30:
    rho, p_val = spearmanr(
        surf_feature[valid_mask],
        surf_target[valid_mask],
    )
```

## Info

### IN-01: Redundant `total_nan > 0` guard in `_compute_cause_separation`

**File:** `scripts/validate_track_condition_nan.py:189-190`
**Issue:** The condition `if total_nan > 0 else 0.0` is redundant because line 182 already returns early when `total_nan == 0`. The ternary always evaluates the true branch at lines 189-190.

**Fix:** Remove the redundant ternary:
```python
return {
    "raw_cause_pct": round(raw_cause / total_nan, 4),
    "derived_cause_pct": round(derived_cause / total_nan, 4),
}
```

### IN-02: `test_track_condition_routing.py` silently skips features not in output columns

**File:** `tests/test_track_condition_routing.py:62-63`
**Issue:** In `test_dirt_features_nan_on_turf_rows`, the check `if feat in mixed_result.columns:` silently passes if a feature is missing from the output entirely (not computed). This means the test would pass even if a dirt feature column was never created. The test should either assert the column exists or explicitly skip with `pytest.skip` to make the omission visible.

**Fix:** Replace the silent pass with an explicit skip or assertion:
```python
for feat in dirt_features:
    assert feat in mixed_result.columns, f"Missing dirt feature column: {feat}"
    turf_vals = mixed_result.loc[turf_mask, feat]
    assert turf_vals.isna().all(), (...)
```

### IN-03: `test_track_condition_ic.py` test `test_category_insufficient_data` lacks meaningful assertions

**File:** `tests/test_track_condition_ic.py:143-150`
**Issue:** The test creates a category evaluation with only 2 samples and asserts `result["n"] == 2`. This is below the minimum threshold of 30, so the function should return NaN statistics, but the test only checks the sample count without verifying that the returned statistics are actually NaN or gracefully handled.

**Fix:** Add assertions for the NaN/empty statistics:
```python
result = _compute_category_evaluation(series, target)
assert result["n"] == 2
assert np.isnan(result["kruskal_wallis"]["H"]), "H should be NaN with < 30 samples"
```

---

_Reviewed: 2026-06-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
