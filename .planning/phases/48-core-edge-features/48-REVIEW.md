---
phase: 48-core-edge-features
reviewed: 2026-06-04T13:50:58Z
depth: standard
files_reviewed: 15
files_reviewed_list:
  - src/features/track_condition_features.py
  - src/features/feature_engine.py
  - src/pipelines/training_pipeline.py
  - src/backtest/race_predictor.py
  - src/domain/models.py
  - src/models/stage1_ability_model.py
  - src/models/two_stage_return_model.py
  - src/models/ev_correction_model.py
  - src/models/place_ability_model.py
  - src/models/wide_two_stage_model.py
  - src/models/gpd_diagnostics.py
  - tests/test_track_condition_features.py
  - tests/test_feature_engine.py
  - tests/test_place_ability_model.py
  - tests/test_win_feature_analysis.py
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 48: Code Review Report

**Reviewed:** 2026-06-04T13:50:58Z
**Depth:** standard
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Phase 48 adds 8 track condition interaction features (dirt_moisture, turf_cushion) to the ML pipeline. The implementation is well-structured with proper NaN propagation, column existence guards, and surgical routing. Tests are comprehensive (22 tests covering per-feature, NaN propagation, missing columns, bin boundaries, and surgical routing). All 22 tests pass.

The core logic in `track_condition_features.py` is correct and defensive. NaN propagation uses `.where()` consistently -- no silent 0.0 values are produced. The `track_stats` lifecycle (training -> SubmodelSet -> inference) is correctly implemented.

Three warnings were identified: (1) `FeatureEngine.build_features()` does not merge track_conditions raw values, creating an asymmetry with `build_all()` that will cause track condition features to be silently NaN in the single-race inference path if it is ever used; (2) `pd.cut` with `bins=[0,...]` silently drops `turf_cushion=0.0` values to NaN, which is correct behavior but has no explicit test coverage; (3) `_compute_track_stats` requires only 2 samples per trackcd to compute statistics, which may produce unstable statistics for low-frequency courses.

No critical bugs or security issues were found.

## Warnings

### WR-01: FeatureEngine.build_features() does not merge track_conditions raw values

**File:** `src/features/feature_engine.py:449-534`
**Issue:** `build_features()` (single-race inference path) does not merge `dirt_moisture`/`turf_cushion` from `track_conditions.parquet`. The `build_all()` method (lines 394-414) correctly performs this merge. If any future caller uses `build_features()` followed by `compute_track_condition_features()`, all 8 track condition features will be silently NaN rather than computed.

Currently, `build_features()` is not called by any production code path (PaperPredictor uses `build_all()` in setup), so this is a latent defect, not an active bug. However, the docstring at line 459 states "BettingOrchestrator" as a caller, indicating the method is intended for active use.

**Fix:** Add the same `load_track_conditions` merge pattern from `build_all()` lines 394-414 into `build_features()`, or add a comment documenting that track condition features are not available in this path and must be added by the caller.

### WR-02: pd.cut silently drops turf_cushion=0.0 values (no test coverage for zero boundary)

**File:** `src/features/track_condition_features.py:161-166`
**Issue:** `pd.cut(cushion, bins=[0, 7, 8, 9, 10, float("inf")], right=True)` defines bins as `(0, 7]`, `(7, 8]`, etc. A `turf_cushion` value of exactly 0.0 falls outside all bins and is silently converted to NaN. This is mathematically correct (0.0 cushion is physically implausible), but the test suite only checks values in the range [0.1, 11.0] and does not verify that turf_cushion=0.0 produces NaN for `sire_x_cushion_band`.

If a data quality issue produces 0.0 cushion values, this NaN will propagate correctly through the interaction feature. However, without a test, this edge case behavior is undocumented.

**Fix:** Add a test case:
```python
def test_sire_x_cushion_band_zero_cushion():
    """turf_cushion=0.0 falls outside [0,inf) bins -> NaN"""
    df = pd.DataFrame({"sire_id": ["S1"], "turf_cushion": [0.0]})
    result = compute_track_condition_features(df)
    assert pd.isna(result["sire_x_cushion_band"].iloc[0])
```

### WR-03: _compute_track_stats minimum sample count is 2 (potentially unstable statistics)

**File:** `src/features/track_condition_features.py:57`
**Issue:** The minimum sample count to compute track-level statistics is `len(cushion_vals) >= 2`. With only 2 samples, the standard deviation is highly unstable and could produce extreme z-scores. For low-frequency courses (e.g., small regional tracks with few turf races), this could create noisy features.

This is a design trade-off rather than a bug -- increasing the threshold would exclude more tracks, reducing coverage.

**Fix:** Consider raising the minimum to 5 or 10 samples for more stable statistics, and adding a warning log for tracks with low sample counts.

## Info

### IN-01: Surgical routing test does not verify WinTwoStageModel.HIT_FEATURE_COLS separately

**File:** `tests/test_track_condition_features.py:339-361`
**Issue:** The `test_surgical_routing_included_models_have_track_condition_features` test verifies `WinTwoStageModel.FEATURE_COLS` (which is `list(RETURN_FEATURE_COLS)`) but does not explicitly verify `WinTwoStageModel.HIT_FEATURE_COLS`. The PLAN requires HIT_FEATURE_COLS registration, and the code does register it (confirmed at line 218 of `two_stage_return_model.py`), but the test only checks `FEATURE_COLS`.

This is not a bug because the test verifies the most comprehensive list (`RETURN_FEATURE_COLS` via `FEATURE_COLS`), and the actual registration is correct. However, explicit verification of `HIT_FEATURE_COLS` would provide stronger coverage.

**Fix:** Add `"WinTwoStageModel.HIT": WinTwoStageModel.HIT_FEATURE_COLS` to the `included_models` dict in the test.

### IN-02: FeatureEngine test FakeRepository.load_track_conditions returns empty DataFrame

**File:** `tests/test_feature_engine.py:215-216`
**Issue:** The `FakeRepository.load_track_conditions()` method was added to prevent `AttributeError` during tests. It correctly returns an empty DataFrame, which causes the track_conditions merge block to be skipped. This is the right approach, but it is worth noting that no test verifies the track_conditions merge produces the expected `dirt_moisture`/`turf_cushion` columns in the `build_all()` output.

**Fix:** Consider adding a test that provides fake track_conditions data and verifies the merge adds the expected columns.

---

_Reviewed: 2026-06-04T13:50:58Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
