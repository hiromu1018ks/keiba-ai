---
phase: 36-feature-computation
reviewed: 2026-05-20T12:00:00Z
depth: deep
files_reviewed: 16
files_reviewed_list:
  - src/features/horse_history_features.py
  - src/features/interaction_features.py
  - src/backtest/race_predictor.py
  - src/models/stage1_ability_model.py
  - src/models/two_stage_return_model.py
  - src/models/ev_correction_model.py
  - src/models/conformal_ev_model.py
  - src/models/market_model.py
  - src/models/place_ability_model.py
  - src/models/race_quality_screener.py
  - src/models/regime_detector.py
  - src/models/wide_two_stage_model.py
  - tests/test_trf_features.py
  - tests/test_hlf_features.py
  - tests/test_horse_history_features.py
  - tests/test_interaction_features.py
findings:
  critical: 2
  warning: 3
  info: 2
  total: 7
  fixed: 3
status: fixed
---

# Phase 36: Code Review Report

**Reviewed:** 2026-05-20T12:00:00Z
**Depth:** deep
**Files Reviewed:** 16
**Status:** issues_found

## Summary

Reviewed all 16 source files from Phase 36 (TRF/INT/HLF feature computation). The implementation is generally solid with correct PIT-safety patterns, proper NaN handling, and consistent model FEATURE_COLS registration across all 12 models. However, I found 2 critical issues: (1) duplicate feature entries in RaceQualityScreener.FEATURE_COLS that will cause LightGBM training failure, and (2) a scope-leak bug in the `unified_raw` variable within the per-horse loop that can produce stale data across iterations. Additionally, there are 3 warnings and 2 info items.

## Critical Issues

### CR-01: Duplicate feature names in RaceQualityScreener.FEATURE_COLS

**File:** `src/models/race_quality_screener.py:84-112`
**Issue:** The FEATURE_COLS list contains duplicate entries for `grade_x_form_trend` (lines 90 and 109), `distance_x_closing_index` (lines 91 and 110), and `grade_x_blood_prize_log` (lines 92 and 111). When LightGBM receives a DataFrame with duplicate column names, it will raise an error or silently use only one column, causing training/prediction failures. The duplicates appear to be from a copy-paste error where the INT-01/02/03 block was appended twice -- once at lines 84-92 ("TRF-01/02/03 + INT-01/02/03/04: Phase 36") and again at lines 109-111 at the end of the list.

**Fix:** Remove the duplicate entries at lines 109-111:

```python
# Remove these three lines (109-111):
        "grade_x_form_trend",
        "distance_x_closing_index",
        "grade_x_blood_prize_log",
```

### CR-02: Stale `unified_raw` variable across per-horse loop iterations

**File:** `src/features/horse_history_features.py:1136-1142`
**Issue:** The variable `unified_raw` is conditionally defined inside the `if n_past > 0:` block (line 1117), but the z-score computation block (starting at line 1137) checks `"unified_raw" not in dir()` to determine if data is available. The `dir()` function in Python checks the local scope, and `unified_raw` from a **previous iteration** of the outer `for i, row in enumerate(horses.itertuples())` loop persists as a local variable. This means:
- On iteration 1 (first horse with `n_past == 0`), `unified_raw` is not defined. The `dir()` check correctly skips.
- On iteration 2 (second horse with `n_past == 0`), `unified_raw` from iteration 1 (if that horse had data) **still exists** in local scope, and the `dir()` check passes, potentially using stale data from the previous horse.

While the subsequent `np.all(np.isnan(unified_raw))` guard may catch this in most cases, if the previous horse had valid data, `unified_raw` would contain non-NaN values from the wrong horse, leading to incorrect z-scores.

**Fix:** Initialize `unified_raw = None` at the start of each loop iteration (e.g., after line 1091 `harontime_last3f_trend: float = float("nan")`), or restructure to use a separate flag variable:

```python
# After line 1091, add:
unified_raw = None

# Then in the n_past > 0 block (line 1117), just assign:
# unified_raw = np.where(...)

# In the z-score block (line 1139), simplify the check:
if unified_raw is not None and not np.all(np.isnan(unified_raw)):
```

## Warnings

### WR-01: EMA weights computed on sorted-valid array lose temporal order correspondence

**File:** `src/features/horse_history_features.py:909-919`
**Issue:** The EMA weight computation for `harontimel5_avg` extracts `ht_valid = ht_raw[~np.isnan(ht_raw)]`, which filters out NaN values but preserves chronological order of the remaining valid entries. The weights are then applied as `(1 - decay) ** np.arange(n_ht)` and reversed so index 0 = newest. This is correct. However, the same pattern is replicated for `harontimel4_avg` (line 1030-1033), `harontime_last3f_avg` (line 1124-1128), and `weighted_recent_form_finish/time` (lines 1614-1618, 1631-1635). Each replication duplicates the weight computation logic. If the valid-filtering logic ever changes in one location but not others, inconsistencies will arise. Consider extracting a shared helper function for EMA-weighted average.

**Fix:** Extract a helper function:

```python
def _ema_weighted_avg(values: np.ndarray, halflife: int = 3) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return float("nan")
    decay = np.log(2) / halflife
    n = len(valid)
    weights = (1 - decay) ** np.arange(n)
    weights = weights[::-1] / weights.sum()
    return float(np.sum(valid * weights))
```

### WR-02: `weight_x_class` interaction uses grade_num even when grade_code column is absent

**File:** `src/features/interaction_features.py:133-141`
**Issue:** The `grade_num` variable is computed only when `grade_code` is in `df.columns` (line 135-136), defaulting to an empty `pd.Series(dtype=float)` otherwise. But then `weight_x_class` is computed when both `weight_col` and `grade_code` exist (line 137). Inside `INT-01` (line 147), `grade_x_form_trend` reuses `grade_num` which was computed earlier. If `grade_code` is absent, `grade_num` is an empty Series, and the INT-01 block at line 146 checks `grade_code in df.columns` before using it, so it would be skipped. This is technically safe but fragile -- the reliance on a variable defined 14 lines above and only conditionally populated is a maintenance hazard. A reader could easily add a new interaction using `grade_num` without realizing it may be an empty Series.

**Fix:** Move the `grade_num` computation closer to its consumers or recompute it inline within each interaction block for clarity.

### WR-03: `_pace_lookup` built from `entries_filtered` without PIT filtering per horse

**File:** `src/features/horse_history_features.py:781-815`
**Issue:** The `_pace_lookup` dict is built by iterating over `entries_filtered.groupby("kettonum")` and collecting all race pace ratios for each horse. At query time (line 1660), searchsorted correctly filters to past dates, ensuring PIT safety. However, the lookup itself includes ALL entries for a horse (including future races relative to other horses' target dates). This is not a PIT leak for the current horse (since searchsorted filters), but it means the `_pace_lookup` dict is larger than necessary, consuming extra memory. More importantly, the `entries_filtered` DataFrame was filtered to only include horses relevant to the current batch (`ketto_set`), but if two horses share a jockey, `entries_filtered` may include extra entries. This is a minor efficiency concern, not a correctness bug.

**Fix:** No code change required for correctness. The PIT safety is ensured by the searchsorted at query time.

## Info

### IN-01: EMA decay formula comment is slightly misleading

**File:** `src/features/horse_history_features.py:912`
**Issue:** The comment says `# w[i] = (1-decay)^i where i=0 is oldest, i=n-1 is newest`, but the `decay = np.log(2) / halflife` formula does not produce a standard EMA halflife. A standard EMA with halflife `h` has decay factor `alpha = 1 - exp(-ln(2)/h)` or equivalently `alpha = 0.5^(1/h)`. The code uses `(1 - ln(2)/h)^i` which is a geometric approximation. For halflife=3, the standard EMA alpha would be `1 - 0.5^(1/3) = 0.2063`, while the code uses `ln(2)/3 = 0.2310`. The weights are slightly different from a true EMA halflife=3, but since the model trains with these exact weights, the approximation is consistent across train/test and not a correctness issue.

**Fix:** Update the comment to accurately describe the formula used: `# Geometric decay approximation: w[i] = (1 - ln(2)/halflife)^i`.

### IN-02: Test `test_base_cols_count` hardcoded to 62 may break on next feature addition

**File:** `tests/test_horse_history_features.py:1862`
**Issue:** The test `TestNewFeaturesInBaseCols.test_base_cols_count` asserts `len(HorseHistoryFeatures.BASE_COLS) == 62`. This hard-coded count will fail whenever a new feature is added to BASE_COLS, requiring manual updating. A more robust test would verify that all expected feature groups are present without checking the exact count.

**Fix:** Replace the exact count assertion with a presence check for each expected feature group, or assert a minimum count instead.

---

_Reviewed: 2026-05-20T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
