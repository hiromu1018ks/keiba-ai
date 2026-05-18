---
phase: 32-market-cross-consistency-features
reviewed: 2026-05-18T12:00:00Z
depth: standard
files_reviewed: 14
files_reviewed_list:
  - src/features/market_cross_features.py
  - src/db/repository.py
  - tests/test_market_cross_features.py
  - src/features/feature_engine.py
  - src/models/stage1_ability_model.py
  - src/models/market_model.py
  - src/models/regime_detector.py
  - src/models/place_ability_model.py
  - src/models/race_quality_screener.py
  - src/models/wide_two_stage_model.py
  - src/models/two_stage_return_model.py
  - src/models/ev_correction_model.py
  - src/models/conformal_ev_model.py
  - tests/test_post_race_leakage.py
findings:
  critical: 0
  warning: 3
  info: 3
  total: 6
status: issues_found
---

# Phase 32: Code Review Report

**Reviewed:** 2026-05-18T12:00:00Z
**Depth:** standard
**Files Reviewed:** 14
**Status:** issues_found

## Summary

Phase 32 adds Market Cross-Consistency Features (MCF-01~06): 5 new features derived from cross-market odds consistency (win x wide x trio). The primary new module `market_cross_features.py` is well-structured with proper NaN fallbacks, Harville formula numerical guards, and single/multi-race path handling. The feature is correctly wired into `feature_engine.py` (build_all + build_features paths) and all 12 downstream model FEATURE_COLS lists are updated consistently.

The integration is clean overall. No POST_RACE leakage vectors found. No security issues. Three warnings were identified: dead/redundant code in the trio feature computation, an unreachable code path in the single-race fallback, and a minor efficiency concern in the multi-race broadcast logic.

## Warnings

### WR-01: Redundant _get_prob_for_umaban call in _compute_trio_features

**File:** `src/features/market_cross_features.py:347`
**Issue:** Line 347 calls `_get_prob_for_umaban(h3, h3, ...)` to assign `_, p_c`, but `p_c` is never used. Lines 349-356 immediately recompute the same value as `p_h3` with an identical loop, and line 358-361 use `p_h3` (not `p_c`). The line 347 call is dead code that wastes computation and confuses readers about which variable holds h3's probability.

**Fix:** Remove line 347 entirely. The `p_h3` computation on lines 349-356 is the canonical path. Alternatively, replace lines 349-356 with a single call:
```python
p_a, _ = _get_prob_for_umaban(h1, h1, tanodds_valid, umaban, p_norm)
_, p_b = _get_prob_for_umaban(h2, h2, tanodds_valid, umaban, p_norm)
_, p_h3 = _get_prob_for_umaban(h3, h3, tanodds_valid, umaban, p_norm)
```
This would be more consistent but requires verifying `_get_prob_for_umaban` returns the same value for both slots when h1==h2.

### WR-02: Unreachable else branch in compute_market_cross_features

**File:** `src/features/market_cross_features.py:493-494`
**Issue:** The `else` branch at line 493 (`return _compute_for_single_race(df, tanodds, wide_df, trio_df)`) is intended for single-race inference when `race_id` is absent. However, in the `build_features()` path (`feature_engine.py:491`), `compute_market_cross_features(df)` is called with `wide_df=None, trio_df=None`, which means the function returns at line 479 (NaN fallback) before ever reaching line 493. For the single-race path to execute, a caller must pass a non-None `wide_df` and `trio_df` but omit the `race_id` column from `df`. No such caller currently exists in the codebase, making this branch effectively unreachable. While not a correctness bug, it represents untested code that may rot.

**Fix:** Either (a) add a test case that exercises `_compute_for_single_race` with actual wide/trio data (not None), or (b) document in the docstring that the single-race branch is a parity path for future use and note the current lack of callers.

### WR-03: Double .map() chain creates unnecessary intermediate Series in multi-race broadcast

**File:** `src/features/market_cross_features.py:441-445`
**Issue:** The broadcast logic `race_ids.map(result_series.map(lambda x: x[0]))` creates an intermediate Series from `result_series.map(lambda x: x[0])` and then maps it again via `race_ids.map(...)`. This is functionally correct but creates 5 unnecessary intermediate Series objects (one per MCF column). With thousands of races, each containing 10-18 horses, this doubles the mapping operations.

**Fix:** Pre-extract the tuple elements into separate Series once, then map from those:
```python
result_series = pd.Series(results)
s_fav = result_series.map(lambda x: x[0])
s_overlap = result_series.map(lambda x: x[1])
s_consistency = result_series.map(lambda x: x[2])
s_trio_ratio = result_series.map(lambda x: x[3])
s_wide_ratio = result_series.map(lambda x: x[4])

df["rl_favorite_in_wide_top1"] = race_ids.map(s_fav)
df["rl_trio_overlap"] = race_ids.map(s_overlap)
df["rl_market_consistency"] = race_ids.map(s_consistency)
df["rl_trio_odds_ratio"] = race_ids.map(s_trio_ratio)
df["rl_wide_harville_ratio"] = race_ids.map(s_wide_ratio)
```
This eliminates the double-map by separating the tuple extraction from the race_id lookup.

## Info

### IN-01: build_features() inference path produces all-NaN MCF columns

**File:** `src/features/feature_engine.py:491`
**Issue:** The `build_features()` method calls `compute_market_cross_features(df)` without passing `wide_df` or `trio_df`. This means all 5 MCF features will be NaN at inference time. While LightGBM handles NaN natively and the design is intentional (noted as "MCF-07 parity"), the MCF features effectively contribute nothing to real-time predictions. This is a known limitation documented in the module docstring but worth noting for model performance expectations.
**Fix:** Consider logging an informational message when MCF features fall back to NaN during inference, so operators can distinguish between "feature unavailable" and "feature computed as NaN due to data issue."

### IN-02: _compute_for_single_race assigns scalar to entire DataFrame column

**File:** `src/features/market_cross_features.py:218-222`
**Issue:** Each MCF feature value (e.g., `fav_in_wide = 1.0`) is a scalar assigned to `df["rl_favorite_in_wide_top1"]`. Pandas broadcasts the scalar to all rows. This works correctly but relies on implicit broadcasting behavior. If the function were ever called with an empty DataFrame, the column assignment would still succeed but produce an empty column -- which is benign.
**Fix:** No action needed. Document that `_compute_for_single_race` assumes the entire DataFrame represents one race.

### IN-03: Harville trio formula enumerates all 6 permutations explicitly

**File:** `src/features/market_cross_features.py:92-102`
**Issue:** The Harville trio probability calculation manually enumerates all 6 permutations of (a, b, c) in a hardcoded list. While mathematically correct and readable, this approach does not generalize if the formula is ever extended to more than 3 horses (e.g., exacta). The current scope (trio = 3 horses) makes this acceptable.
**Fix:** No action needed for the current scope. If extending to larger combination sizes, consider using `itertools.permutations`.

---

_Reviewed: 2026-05-18T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
