---
phase: 32-market-cross-consistency-features
fixed_at: 2026-05-18T13:00:00Z
review_path: .planning/phases/32-market-cross-consistency-features/32-REVIEW.md
iteration: 1
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 32: Code Review Fix Report

**Fixed at:** 2026-05-18T13:00:00Z
**Source review:** .planning/phases/32-market-cross-consistency-features/32-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 3
- Fixed: 3
- Skipped: 0

## Fixed Issues

### WR-01: Redundant _get_prob_for_umaban call in _compute_trio_features

**Files modified:** `src/features/market_cross_features.py`
**Commit:** 763e3d0
**Applied fix:** Removed the dead `_get_prob_for_umaban(h3, h3, ...)` call that assigned to unused `p_c`, and replaced the manual loop for `p_h3` with clean per-horse calls: `p_a, _ = _get_prob_for_umaban(h1, h1, ...)`, `_, p_b = _get_prob_for_umaban(h2, h2, ...)`, `_, p_h3 = _get_prob_for_umaban(h3, h3, ...)`. Reduced 11 lines to 3 lines with identical semantics.

### WR-02: Unreachable else branch in compute_market_cross_features

**Files modified:** `src/features/market_cross_features.py`
**Commit:** ac1f345
**Applied fix:** Added a Note to the `_compute_for_single_race` docstring documenting that the function is currently unreachable because `build_features()` passes `wide_df=None, trio_df=None` which triggers the NaN fallback before reaching the single-race branch. Documented that this is a parity implementation reserved for future single-race inference when wide/trio data becomes available.

### WR-03: Double .map() chain creates unnecessary intermediate Series

**Files modified:** `src/features/market_cross_features.py`
**Commit:** 6852e54
**Applied fix:** Replaced the double-map pattern `race_ids.map(result_series.map(lambda x: x[i]))` with pre-extracted Series: first extract tuple elements into 5 separate Series (`s_fav`, `s_overlap`, `s_consistency`, `s_trio_ratio`, `s_wide_ratio`), then assign each column via a single `race_ids.map(s_*)` call. Eliminates 5 unnecessary intermediate Series objects.

## Skipped Issues

None -- all findings were fixed.

---

_Fixed: 2026-05-18T13:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
