---
phase: 49-derived-higher-order-features
reviewed: 2026-06-05T00:00:00Z
depth: standard
files_reviewed: 16
files_reviewed_list:
  - src/features/horse_track_aptitude.py
  - scripts/precompute_track_aptitude.py
  - tests/test_horse_track_aptitude.py
  - src/db/readers.py
  - src/db/repository.py
  - src/features/feature_engine.py
  - src/features/track_condition_features.py
  - src/pipelines/training_pipeline.py
  - src/backtest/race_predictor.py
  - src/domain/models.py
  - src/models/stage1_ability_model.py
  - src/models/two_stage_return_model.py
  - src/models/ev_correction_model.py
  - src/models/place_ability_model.py
  - src/models/wide_two_stage_model.py
  - config/settings.yaml
  - tests/test_track_condition_features.py
findings:
  critical: 2
  warning: 4
  info: 2
  total: 8
status: issues_found
---

# Phase 49: Code Review Report

**Reviewed:** 2026-06-05
**Depth:** standard
**Files Reviewed:** 16
**Status:** issues_found

## Summary

Reviewed Phase 49 (Derived & Higher-Order Features) implementation covering: horse track aptitude precomputation (horse_track_aptitude.py), 11 derived/higher-order track condition features (track_condition_features.py), 4 race-level aggregation features, reader/repository integration, FeatureEngine merge, and training pipeline routing.

The PIT-safe expanding window + shift(1) pattern in horse_track_aptitude.py is correctly implemented. NaN propagation is handled consistently throughout. Division-by-zero is guarded (starts > 0 check for hit rates, std > 0 check for zscore/season deviation). Feature routing to model FEATURE_COLS lists is correctly surgical -- excluded models (MarketModel, RaceQualityScreener, RegimeDetector, ConformalEVModel) do not contain any track condition features.

However, two critical issues were found: (1) the inference path (RacePredictor.predict + FeatureEngine.build_features) does not merge the horse_track_aptitude parquet, causing race_condition_match_* features to be NaN at inference time while being real values during training, and (2) moisture_season_deviation and cushion_season_deviation may not be created at all when the corresponding raw columns are absent, leading to inconsistent column sets between train and inference. Additionally, several warnings around incomplete inference-path coverage and conditional column creation patterns.

## Critical Issues

### CR-01: Train-inference feature mismatch -- horse_track_aptitude not merged in inference path

**File:** `src/backtest/race_predictor.py:263-266` and `src/features/feature_engine.py:484-569`
**Issue:** The training pipeline (via `FeatureEngine.build_all()` at line 417-449) merges `horse_track_aptitude` parquet columns (`horse_dirt_wet_hit_rate`, `horse_dirt_dry_hit_rate`, `horse_cushion_hard_hit_rate`, `horse_cushion_soft_hit_rate`, `prev_dirt_moisture`, `prev_turf_cushion`, etc.) into the feature DataFrame. These columns are then consumed by `compute_race_condition_features()` (which reads `horse_dirt_wet_hit_rate` to compute `race_condition_match_score/max/ratio`) and by `compute_track_condition_features()` (which reads `prev_dirt_moisture` and `prev_turf_cushion` to compute `surface_condition_transition`).

Neither `RacePredictor.predict()` nor `FeatureEngine.build_features()` merges this parquet. At inference time:
- `race_condition_match_score/max/ratio` will always be NaN (the `horse_dirt_wet_hit_rate` etc. columns are absent, so `compute_race_condition_features` uses `pd.Series(np.nan)` fallbacks, making match_rate all NaN, making the groupby transform produce NaN).
- `surface_condition_transition` will always be NaN (because `prev_dirt_moisture` and `prev_turf_cushion` are absent from the DataFrame).

Since `race_condition_match_score`, `race_condition_match_max`, `race_condition_match_ratio`, `race_field_front_bias`, and `surface_condition_transition` are all in model FEATURE_COLS (AbilityModel, WinTwoStageModel, PlaceTwoStageModel, EVCorrectionModel, etc.), the models will receive NaN for these features at inference time while having learned real signal during training. This is a train-inference distribution mismatch.

**Fix:** Add horse_track_aptitude merge to `RacePredictor.predict()` before `compute_race_condition_features()` is called, and/or add it to `FeatureEngine.build_features()`. The inference path needs to load the precomputed parquet (or receive it as a parameter) and merge on `(race_id, kettonum)`.

```python
# In RacePredictor.predict(), after df = race_df.copy() and before compute calls:
if hasattr(self, '_apt_df') and self._apt_df is not None:
    if "kettonum" in df.columns:
        apt_cols = [c for c in self._apt_df.columns
                    if c in {"race_id", "kettonum"} or c.startswith("horse_") or c.startswith("prev_")]
        df = df.merge(self._apt_df[apt_cols], on=["race_id", "kettonum"], how="left")
```

### CR-02: Conditional column creation in compute_track_condition_features causes inconsistent column sets

**File:** `src/features/track_condition_features.py:308-377`
**Issue:** The T3-04 season deviation section creates `cushion_season_deviation` and `moisture_season_deviation` only under specific conditions:
- If `track_month_stats is None` or empty, neither column is created (lines 375-377 just `pass`).
- If `has_turf_cushion` is False (e.g., dirt-only DataFrame), `cushion_season_deviation` is never created.
- If `has_dirt_moisture` is False (e.g., turf-only DataFrame), `moisture_season_deviation` is never created.

When these columns are absent, the downstream T4-03 anomaly flags (lines 379-389) correctly check with `if "cushion_season_deviation" in df.columns`, so no KeyError occurs. However, all 11 TRACK_DERIVED_COLS are listed in model FEATURE_COLS. If `cushion_season_deviation` is absent, `_prepare_features()` will simply not include it, and LightGBM fills it with NaN internally. But this means the model expects a column that doesn't exist in the DataFrame, and the behavior depends on `_prepare_features` implementation tolerating missing columns.

The real problem occurs when `track_month_stats` is None (e.g., inference with no stored stats on SubmodelSet). In this case, NONE of the T3-04/T4-03 features are created, yet they're expected by the model. While `_prepare_features` handles missing columns gracefully, this creates an inconsistent contract: the feature constants say 11 derived features exist, but in practice fewer may be present.

**Fix:** Always create all TRACK_DERIVED_COLS columns, filling with NaN when prerequisites are absent:

```python
# After the T3-04 block, ensure columns always exist:
if "cushion_season_deviation" not in df.columns:
    df["cushion_season_deviation"] = float("nan")
if "moisture_season_deviation" not in df.columns:
    df["moisture_season_deviation"] = float("nan")
```

## Warnings

### WR-01: horse_condition_type uses only dirt metrics -- turf-only horses always classified as "unknown"

**File:** `src/features/horse_track_aptitude.py:205-237`
**Issue:** The `horse_condition_type` classification (wet_good/dry_good/balanced) uses only `horse_dirt_wet_hit_rate` and `horse_dirt_dry_hit_rate`. For turf-only horses (no dirt starts), both `wet_rate` and `dry_rate` will be NaN, so `wet_sufficient` and `dry_sufficient` are both False, and the classification is always "unknown". Similarly, `horse_condition_versatility` (line 239-245) will be NaN for turf-only horses. This means the two output columns `horse_condition_type` and `horse_condition_versatility` carry no signal for turf-only horses, reducing their utility.

This is a design limitation rather than a bug, but it should be documented or the feature should be extended with turf-specific metrics (hard/soft rates) to be useful for the majority of races (turf is more common than dirt in JRA).

**Fix:** Consider adding turf-based classification (hard_good/soft_good/balanced) or a combined metric. At minimum, document that these two columns are dirt-only.

### WR-02: precompute_track_aptitude.py uses `race_id` as date ordering fallback

**File:** `src/features/horse_track_aptitude.py:159-163`
**Issue:** When `race_date` column is not present in entries_df, the code falls back to sorting by `race_id`. While `race_id` format (`YYYYMMDDJyoKaiNiRace`) is lexicographically sortable and correlates with chronological order, this fallback means a missing `race_date` column will silently produce potentially incorrect ordering if the race_id format ever changes. The comment says "Fallback: use race_id as proxy for date ordering" but there is no warning logged when this fallback is activated.

**Fix:** Log a warning when the fallback is used:

```python
if "race_date" not in ent.columns and "race_date" not in entries_df.columns:
    logger.warning("race_date not found; using race_id as date proxy (may be incorrect)")
    ent = ent.sort_values(["kettonum", "race_id"]).reset_index(drop=True)
```

### WR-03: `compute_race_condition_features` mutates caller expectations for mid-range moisture/cushion

**File:** `src/features/track_condition_features.py:522-530`
**Issue:** For "middle" dirt conditions (moisture between dry_threshold and wet_threshold, i.e., 3.0 <= moisture < 12.0), the match_rate is computed as `(wet_rate.fillna(0) + dry_rate.fillna(0)) / 2`. The `.fillna(0)` means a horse with only wet starts (dry_rate=NaN) gets treated as having 0 dry hit rate, pulling the average down. Similarly for turf middle conditions (8.0 <= cushion < 10.0). This creates an asymmetric bias: horses with only one-sided experience get penalized more than warranted.

**Fix:** Consider using only the available rate (not averaging with 0), or using the horse's overall hit rate as a fallback for the missing side.

### WR-04: TRAIN_DERIVED_COLS / RACE_CONDITION_COLS not fully resilient to missing feature columns

**File:** `src/features/track_condition_features.py:391-443`
**Issue:** The T4-04 numeric interactions (`cushion_x_distance`, `moisture_x_weight`, `cushion_x_age`) are only created when the raw column exists (`has_turf_cushion`, `has_dirt_moisture`). For the `surface_condition_transition`, it requires `prev_dirt_moisture` or `prev_turf_cushion` to exist. If neither is present, `surface_condition_transition` is never created. Since this column is in TRACK_DERIVED_COLS and model FEATURE_COLS, its absence means the feature silently drops out at inference time.

Unlike the season deviation features, there's no explicit NaN fallback assignment when prerequisites are missing.

**Fix:** Add explicit NaN assignment at the end of `compute_track_condition_features` for any TRACK_DERIVED_COLS not yet created:

```python
for col in TRACK_DERIVED_COLS:
    if col not in df.columns:
        df[col] = float("nan")
```

## Info

### IN-01: Duplicate threshold constants between horse_track_aptitude.py and track_condition_features.py

**File:** `src/features/horse_track_aptitude.py:22-29` and `src/features/track_condition_features.py:472-477`
**Issue:** The same threshold constants (`_DIRT_WET_THRESHOLD=12.0`, `_DIRT_DRY_THRESHOLD=3.0`, etc.) are defined independently in both `horse_track_aptitude.py` (module-level constants) and `compute_race_condition_features()` (local variables). If one is updated without the other, the classification logic becomes inconsistent. The thresholds are also duplicated in `config/settings.yaml` under `track_condition:`.

**Fix:** Import from a shared constant module or from the config:

```python
from features.horse_track_aptitude import (
    _DIRT_WET_THRESHOLD, _DIRT_DRY_THRESHOLD,
    _TURF_HARD_THRESHOLD, _TURF_SOFT_THRESHOLD,
    _MIN_STARTS, _HIT_RATE_THRESHOLD,
)
```

### IN-02: test_horse_track_aptitude.py test_condition_type_wet_good assertion may be fragile

**File:** `tests/test_horse_track_aptitude.py:209-213`
**Issue:** The test asserts `last_row["horse_condition_type"] == "wet_good"` but the "last row" depends on sort order. The test constructs 4 wet races (W1-W4) followed by 4 dry races (D1-D4), and after sorting by race_date, the last entry is D4. At D4, the cumulated wet stats are 4 wet starts with 4 hits (rate=1.0, starts=4 >= 3), and cumulated dry stats are 3 dry starts with 0 hits (rate=0.0, starts=3 >= 3). The logic correctly classifies this as wet_good, but the test's correctness depends on understanding the sort order and shift(1) behavior. A comment explaining why last_row is D4 would improve clarity.

**Fix:** Add a comment:
```python
# Last row is D4 (Jan-Aug chronological): after 4 wet hits + 3 dry non-hits
# wet_rate=4/4=1.0 >= 0.3, dry_rate=0/3=0.0 < 0.3, both sufficient -> wet_good
```

---

_Reviewed: 2026-06-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
