---
phase: 27-feature-interactions
reviewed: 2026-05-15T12:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - src/features/interaction_features.py
  - src/features/relative_features.py
  - src/features/target_encoding.py
  - src/models/stage1_ability_model.py
  - src/models/two_stage_return_model.py
  - src/pipelines/training_pipeline.py
  - tests/test_interaction_features.py
  - tests/test_relative_features.py
  - tests/test_target_encoding.py
  - tests/test_two_stage_return_model.py
  - tests/test_win_feature_analysis.py
findings:
  critical: 2
  warning: 5
  info: 4
  total: 11
status: issues_found
---

# Phase 27: Code Review Report

**Reviewed:** 2026-05-15T12:00:00Z
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Reviewed 11 files from Phase 27 (feature interactions, relative features, target encoding, and their integration into models/pipeline). Found 2 critical issues and 5 warnings. The critical issues are: (1) TargetEncoder is not stored in SubmodelSet, making trained TE mappings unavailable at inference time, causing train/test feature mismatch; and (2) relative features (`compute_relative_features`, `compute_stage2_relative_features`) are not called in the inference path (`race_predictor.py`), creating another train/test discrepancy for the new INTER-01 relative feature columns that are listed in Stage2 FEATURE_COLS.

## Critical Issues

### CR-01: TargetEncoder not persisted -- train/test feature mismatch at inference

**File:** `src/pipelines/training_pipeline.py:557-569`
**Issue:** The `TargetEncoder` is created, fitted via `fit_transform_oof()`, and then discarded. The fitted `te_encoder` object (which holds `encoding_maps_` and `global_mean_`) is never stored in `SubmodelSet`, never saved to disk via `_save_models_local()`, and never logged to MLflow. At inference time, the three TE columns (`te_blood_keito_cd`, `te_kisyucode`, `te_chokyosicode`) listed in `WinTwoStageModel.FEATURE_COLS` and `PlaceTwoStageModel.RETURN_FEATURE_COLS`/`HIT_FEATURE_COLS` will be absent from the DataFrame. LightGBM silently trains on these columns during training (they are present in `df_oof`), but at inference, `_prepare_features()` filters to `available_cols = [c for c in cols if c in df.columns]`, so the model sees fewer features than it was trained with. This causes silent model degradation (the booster internally handles missing features by using default values, but the learned splits on TE features are dead at inference).

Additionally, `race_predictor.py` has no call to `TargetEncoder.transform()` anywhere.

**Fix:**
1. Add a `te_encoder: TargetEncoder | None = None` field to `SubmodelSet` (in `src/domain/models.py`).
2. Store the fitted encoder: `sub.te_encoder = te_encoder` after `fit_transform_oof()`.
3. Save/load the encoder in `_save_models_local()` and the corresponding load function (e.g., via joblib pickle).
4. In `race_predictor.py`, call `submodel.te_encoder.transform(df)` before `submodel.win.predict_ev(df)`.

### CR-02: Relative features not computed in inference path -- train/test mismatch

**File:** `src/pipelines/training_pipeline.py:516-519,585-586` vs `src/backtest/race_predictor.py`
**Issue:** The training pipeline calls `compute_relative_features(df)` (line 519) and `compute_stage2_relative_features(df_oof)` (line 586) to generate 9+3=12 relative feature columns. These columns are listed in `AbilityModel.FEATURE_COLS`, `WinTwoStageModel.FEATURE_COLS`, `PlaceTwoStageModel.HIT_FEATURE_COLS`, and `PlaceTwoStageModel.RETURN_FEATURE_COLS`. However, the inference path in `race_predictor.py` does not call either function. The `_prepare_features()` method in both model classes silently drops missing columns via `available_cols = [c for c in self.FEATURE_COLS if c in df.columns]`, so the models train on features they never see at inference time. This is a systematic train/test skew affecting 12 features.

**Fix:**
In `race_predictor.py`, add calls to `compute_relative_features(df)` and `compute_stage2_relative_features(df)` at the appropriate points in the inference chain -- the former before market model, the latter after `submodel.stage1.add_ability_probs(df)` and after `odds_to_ability_ratio` is computed.

## Warnings

### WR-01: TargetEncoder.min_samples parameter declared but never used

**File:** `src/features/target_encoding.py:51,57`
**Issue:** The `min_samples` parameter is accepted in `__init__()` and stored as `self.min_samples`, but it is never referenced anywhere in `fit_transform_oof()`, `transform()`, or `_compute_cat_stats()`. The docstring (line 42-43) describes a smoothing formula that uses only `self.smoothing`, not `self.min_samples`. This is dead configuration that gives a false impression of controlling the minimum category frequency threshold.

**Fix:** Either implement the intended `min_samples` logic (skip categories with count < min_samples and fall back to global mean for them), or remove the parameter entirely.

### WR-02: `closer_share` feature computed but never used by any model

**File:** `src/features/interaction_features.py:213`
**Issue:** The feature `closer_share` is computed and added to the DataFrame, but it does not appear in `INTERACTION_COLS` (which has exactly 12 entries and does not include it), and it is not listed in any model's `FEATURE_COLS` across the codebase. This is dead computation that wastes memory and processing time.

**Fix:** Either add `closer_share` to `INTERACTION_COLS` and the relevant model `FEATURE_COLS`, or remove the computation.

### WR-03: `blood_keito_x_surface` NaN gate checks wrong variable

**File:** `src/features/interaction_features.py:73-78`
**Issue:** Line 73 computes `keito = pd.to_numeric(df["blood_keito_cd"], errors="coerce")`, and line 75 checks `keito.notna().any()`. However, if `keito` has *some* valid values but also NaN rows, the gate passes (`.any()` returns True), and then line 77 uses `df["blood_keito_cd"]` (not `keito`) for the string concatenation. This means rows where `blood_keito_cd` could not be converted to numeric (e.g., string values like "NaN" or other non-numeric codes) will produce interaction strings like "NaN_turf" rather than being excluded. The NaN gate is effectively a global skip-or-include decision rather than a per-row NaN policy.

**Fix:** If the intent is to generate the interaction for all rows when at least some are valid (consistent with other interaction features), the current behavior is acceptable. However, if per-row NaN exclusion is desired, the `.where()` pattern used for numeric interactions should be applied. Clarify the intent.

### WR-04: `surface_x_past_perf` produces 0.0 for unknown surface values instead of NaN

**File:** `src/features/interaction_features.py:122-126`
**Issue:** Line 122 maps `surface` via `{"turf": 1, "dirt": 2}` with `.fillna(0)`. The `.where()` on line 123-126 only guards against `norm_finish_logit_avg` being NaN. If `surface` is neither "turf" nor "dirt" (e.g., a value that was not anticipated), `surface_code` becomes 0, and the product is `norm_finish_logit_avg * 0 = 0.0`. This silently creates a misleading feature value rather than NaN. Other numeric interaction features (e.g., `sire_wr_x_distance`) use `.where()` on both operands.

**Fix:** Add `surface_code` to the `.where()` condition:
```python
df["surface_x_past_perf"] = (df["norm_finish_logit_avg"] * surface_code).where(
    df["norm_finish_logit_avg"].notna() & (surface_code != 0),
    other=float("nan"),
)
```

### WR-05: `odds_gap_fav12` reindex may misalign when `df` index is not default RangeIndex

**File:** `src/features/interaction_features.py:174-175`
**Issue:** `odds_gap = (pop1_odds - pop2_odds).reindex(df["race_id"]).values` passes a Series (`df["race_id"]`) to `.reindex()`. When `.reindex()` receives a list-like, it uses it as the new index, which is correct for label-based lookup. However, if `df` has a non-default index (e.g., after filtering), the `.values` array will be aligned to the order of `df["race_id"]` (row-by-row), which is correct. The subsequent `pd.Series(odds_gap, index=df.index)` then correctly re-aligns. This is actually correct but fragile -- if someone later changes the reindex logic, it could break silently. No immediate bug, but worth noting.

**Fix:** No immediate fix required, but consider adding a comment explaining the index alignment contract.

## Info

### IN-01: `_GRADE_MAP` duplicate values for J-grades and regular grades

**File:** `src/features/interaction_features.py:129`
**Issue:** `J.G1` and `G1` both map to 5, `J.G2` and `G2` both map to 4, etc. This is intentional (J-graded stakes are equivalent to their non-J counterparts in this encoding), but the duplicate entries could be confusing.

**Fix:** Consider adding a comment explaining the equivalence: `# J.G1/J.G2/J.G3 map to same values as G1/G2/G3 (jump races use same grade weight)`.

### IN-02: Test `test_feature_cols_minimum_length` asserts outdated minimum

**File:** `tests/test_two_stage_return_model.py:629-631`
**Issue:** The test asserts `len(cols) >= 31` with a comment saying "25 existing + 6 new", but the actual `WinTwoStageModel.FEATURE_COLS` now has many more features (60+). The test passes but its documentation and minimum are stale.

**Fix:** Update the comment and minimum to reflect the current feature count.

### IN-03: Test `test_minimum_count` uses hardcoded minimums that may become stale

**File:** `tests/test_two_stage_return_model.py:686-690`
**Issue:** The assertions `>= 50`, `>= 54`, `>= 55` are hardcoded minimums that are already far below the actual counts. While they still pass, they provide minimal regression protection.

**Fix:** Consider asserting exact expected counts to catch unintended additions/removals, or update the minimums closer to actual values.

### IN-04: `distance_bin` value inconsistency between test data and production data

**File:** `tests/test_interaction_features.py:13`
**Issue:** Test uses `distance_bin` values like `"sprint"`, `"mile"`, `"intermediate"`, while the pipeline may use different bin labels. This is fine for unit tests (they test the string concatenation logic, not domain correctness), but worth noting for integration testing.

**Fix:** No immediate action needed for unit tests, but integration tests should use production-consistent bin labels.

---

_Reviewed: 2026-05-15T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
