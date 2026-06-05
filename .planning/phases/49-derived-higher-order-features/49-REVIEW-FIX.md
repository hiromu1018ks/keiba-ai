---
phase: 49-derived-higher-order-features
fixed_at: 2026-06-05T00:00:00Z
review_path: .planning/phases/49-derived-higher-order-features/49-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 5
skipped: 1
status: partial
---

# Phase 49: Code Review Fix Report

**Fixed at:** 2026-06-05
**Source review:** .planning/phases/49-derived-higher-order-features/49-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 6
- Fixed: 5
- Skipped: 1

## Fixed Issues

### CR-01: Train-inference feature mismatch -- horse_track_aptitude not merged in inference path

**Files modified:** `src/backtest/race_predictor.py`
**Commit:** 317c7a7
**Applied fix:** Added `apt_df` parameter to `RacePredictor.__init__()` and merge logic in `predict()` before `compute_track_condition_features()` is called. When aptitude columns are already present (backtest path via build_all()), the merge is skipped. At inference time (paper trading, live), the preloaded apt_df is merged on (race_id, kettonum).

### CR-02: Conditional column creation in compute_track_condition_features causes inconsistent column sets

**Files modified:** `src/features/track_condition_features.py`
**Commit:** e649da6
**Applied fix:** Replaced `pass` in the `else` block (when track_month_stats is None) with explicit NaN assignment for `cushion_season_deviation` and `moisture_season_deviation`. The existing TRACK_DERIVED_COLS fallback loop at end-of-function provides additional safety, but this makes intent explicit at the point of conditional logic.

### WR-01: horse_condition_type uses only dirt metrics -- turf-only horses always classified as "unknown"

**Files modified:** `src/features/horse_track_aptitude.py`
**Commit:** cc7685b
**Applied fix:** Added documentation to module docstring and inline comment at D-05 classification block noting that horse_condition_type and horse_condition_versatility use only dirt metrics and that turf-only horses are always "unknown" with NaN versatility.

### WR-02: precompute_track_aptitude.py uses race_id as date ordering fallback without warning

**Files modified:** `src/features/horse_track_aptitude.py`
**Commit:** 28cdc1e
**Applied fix:** Added `logger.warning()` call when the race_id fallback is activated, alerting operators that race_date was not found and race_id is being used as a proxy.

### WR-03: compute_race_condition_features mutates caller expectations for mid-range moisture/cushion

**Files modified:** `src/features/track_condition_features.py`
**Commit:** 62e9a26
**Applied fix:** Replaced `fillna(0)` averaging with conditional logic: when both rates are available, use their mean; when only one rate is available, use that rate directly; when neither is available, use NaN. Applied to both dirt (wet/dry) and turf (hard/soft) mid-range paths.

## Skipped Issues

### WR-04: TRAIN_DERIVED_COLS / RACE_CONDITION_COLS not fully resilient to missing feature columns

**File:** `src/features/track_condition_features.py:391-443`
**Reason:** Already fixed in current codebase. Both `compute_track_condition_features()` (line 448-451) and `compute_race_condition_features()` (line 641-644) have NaN fallback loops at end-of-function that ensure all TRACK_DERIVED_COLS and RACE_CONDITION_COLS always exist. No code change needed.
**Original issue:** The T4-04 numeric interactions and surface_condition_transition may not be created when raw columns are absent.

---

_Fixed: 2026-06-05_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
