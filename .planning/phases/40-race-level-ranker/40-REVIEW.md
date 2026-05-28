---
phase: 40-race-level-ranker
reviewed: 2026-05-28T12:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - src/models/race_level_ranker.py
  - tests/test_race_level_ranker.py
  - src/domain/models.py
  - src/models/win_benter_gate.py
  - src/pipelines/training_pipeline.py
  - tests/test_win_benter_gate.py
  - src/backtest/race_predictor.py
  - src/db/model_loader.py
  - tests/test_race_predictor.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 40: Code Review Report

**Reviewed:** 2026-05-28T12:00:00Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Reviewed the Race-Level Ranker (Phase 40) implementation across 9 files. The core `RaceLevelRanker` class is well-structured with proper shadow mode guards, OOF-safe target construction, and per-surface Ridge training. However, there are two critical bugs in the inference pipeline: (1) the ranker receives raw feature columns at scoring time instead of the IFF-resolved `if_*` columns it was trained on, causing a feature name mismatch that produces meaningless predictions; and (2) diagnostics are computed on training data rather than held-out validation data, inflating reported metrics. Four warnings cover edge cases in alpha selection, data leakage from skipped surfaces, missing IFF inference integration in RacePredictor, and a test fixture `surface` type mismatch.

## Critical Issues

### CR-01: Feature name mismatch between training and inference -- ranker receives raw columns, not IFF features

**File:** `src/backtest/race_predictor.py:284-285`
**Issue:** At training time (in `training_pipeline.py:1336-1337`), the `InvestmentFeatureFrameBuilder.build_frame(mode="train")` converts raw DataFrame columns into canonical `if_*` feature names (e.g., `if_p_win_final`, `if_ev_calibrated`, `if_odds_log`). The ranker's `RELEVANCE_FEATURES` and `VALUE_FEATURES` lists reference these canonical `if_*` names. However, at inference time in `RacePredictor.predict()`, the ranker's `score()` method receives the raw pipeline DataFrame directly -- without running `build_frame(mode="infer")` first. This means the ranker looks for columns like `if_p_win_final` which may not exist in the raw DataFrame (the actual column might be `p_win_final` or `p_win_corrected`). When features are missing, `_build_relevance_features()` and `_build_value_features()` silently fill with zeros (line 186-188, 218-220), producing degenerate predictions from a model trained on real features but scoring zero-filled vectors.

**Fix:** In `RacePredictor.predict()`, after the MarketAwareWinCalibrator applies but before `ranker.score()`, build the IFF inference frame and merge:

```python
# In RacePredictor.predict(), around line 283:
ranker = getattr(submodel, "win_race_level_ranker", None)
if ranker is not None and ranker.is_trained:
    from investment.feature_frame import InvestmentFeatureFrameBuilder
    iff_builder = InvestmentFeatureFrameBuilder()
    iff_df = iff_builder.build_frame(df, mode="infer")
    df_with_iff = df.merge(iff_df, on=["race_id", "umaban"], how="left", suffixes=("", "_iff"))
    df = ranker.score(df_with_iff)
```

### CR-02: Diagnostics computed on training data (in-sample), not on held-out validation

**File:** `src/models/race_level_ranker.py:316-320`
**Issue:** The `_compute_diagnostics()` method is called immediately after `ridge_rel.fit(rel_X, rel_y)` and `ridge_val.fit(val_X, val_y)`, using the same `rel_X`/`rel_y`/`val_X`/`val_y` that were just used for training. Lines 446-447: `rel_scores = ridge_rel.predict(rel_X)` predicts on the training set, then lines 462-470 compute NDCG@3 on those in-sample predictions. This produces optimistically biased diagnostic metrics (top1_win_rate, ndcg_at_3, rank_of_actual_winner) that do not reflect generalization performance. These metrics flow into `training_summary` and are logged as D-11 diagnostics, giving a misleading view of ranker quality.

**Fix:** Either (a) compute diagnostics on a held-out split (e.g., last 20% of races by date), or (b) restructure `_select_alpha_*` to save the best fold's validation predictions and use those for diagnostics. Minimal fix using last fold:

```python
# After the alpha selection loop, split off last 20% for diagnostics:
diag_split = int(len(df_surf) * 0.8)
df_diag_train = df_surf.iloc[:diag_split]
df_diag_val = df_surf.iloc[diag_split:]
diag_rel_X, _ = self._build_relevance_features(df_diag_val)
diag_rel_y = self._compute_relevance_target(df_diag_val["kakuteijyuni"])
# Use ridge_rel (trained on full df_surf) but evaluate only on df_diag_val
self._compute_diagnostics(df_diag_val, surface_name, ridge_rel, ridge_val,
                          diag_rel_X, diag_rel_y, ...)
```

## Warnings

### WR-01: Alpha selection falls back to strongest regularization when splits are insufficient

**File:** `src/models/race_level_ranker.py:347-348, 403`
**Issue:** When `_walk_forward_race_splits` returns fewer than 2 splits, `_select_alpha_relevance()` and `_select_alpha_value()` return `self.ALPHA_GRID[-1]` (alpha=10.0, the strongest regularization). This is a conservative default, but it happens silently even when the surface has >= 20 rows (the training guard) but too few unique races for WF splitting. No warning is logged, so the operator has no visibility that the ranker fell back to a default alpha.

**Fix:** Log a warning when falling back to the default alpha:

```python
if len(splits) < 2:
    logger.warning(
        "Insufficient WF splits for alpha selection (%s, %d rows, %d splits), "
        "using default alpha=%.1f",
        surface_name, len(df_surf), len(splits), self.ALPHA_GRID[-1],
    )
    return self.ALPHA_GRID[-1]
```

### WR-02: `_trained = True` set even when one surface is skipped

**File:** `src/models/race_level_ranker.py:322`
**Issue:** `self._trained = True` is set unconditionally after the surface loop (line 322), even if one or both surfaces were skipped due to insufficient data (`< 20 rows`, line 271-276). The `is_trained` property (line 106) only checks `self.relevance_scorer_turf is not None`, so if turf trains but dirt does not, `is_trained` returns True. However, `score()` (line 523-537) iterates over all surfaces in the inference data, and if dirt inference data arrives, it silently skips scoring those rows (logging a warning), leaving their `relevance_score`/`value_score` as NaN. This is correct shadow-mode behavior but the user gets `_trained=True` with a partially-functional ranker.

**Fix:** Either (a) set `_trained = True` only when both surface models exist, or (b) add a `surfaces_trained` key to `training_summary` so the operator knows which surfaces are available.

```python
trained_surfaces = []
# In the loop, after successful training:
trained_surfaces.append(surface_name)
# After the loop:
self._trained = len(trained_surfaces) > 0
self.training_summary["trained_surfaces"] = trained_surfaces
```

### WR-03: OOF+IFF join in training_pipeline may produce row-order misalignment

**File:** `src/pipelines/training_pipeline.py:1337-1342`
**Issue:** The ranker training data is built by merging `oof_cal_df` with `iff_df` on `["race_id", "umaban"]`:

```python
ranker_train_df = oof_cal_df.merge(
    iff_df,
    on=["race_id", "umaban"],
    how="left",
    suffixes=("", "_iff"),
)
```

The `iff_df` is produced by `InvestmentFeatureFrameBuilder.build_frame()` which sorts by `["race_id", "umaban"]` and resets the index (feature_frame.py:329-331). However, `oof_cal_df` is NOT sorted in any particular order -- it retains the order produced by `generate_win_oof_predictions()`, which is sorted by `["race_date", "race_id", "umaban"]` within folds but concatenated across folds. The merge operation itself preserves the left frame's order, so `ranker_train_df` keeps `oof_cal_df`'s order. The `RaceLevelRanker.train()` then does `df_surf.reset_index(drop=True)` (line 269). The `_walk_forward_race_splits` function splits by `race_id` order (using positional indexing on the reset index). This means the WF splits may not be chronological if `oof_cal_df` was not sorted by `race_date` before the merge.

**Fix:** Sort `ranker_train_df` by `race_date` before passing to `ranker.train()`:

```python
ranker_train_df = oof_cal_df.merge(iff_df, on=["race_id", "umaban"], how="left", suffixes=("", "_iff"))
if "race_date" in ranker_train_df.columns:
    ranker_train_df = ranker_train_df.sort_values("race_date").reset_index(drop=True)
win_race_level_ranker = RaceLevelRanker()
win_race_level_ranker.train(ranker_train_df)
```

### WR-04: `generate_win_oof_predictions` drops `surface` column before copying to result

**File:** `src/models/win_benter_gate.py:204-206`
**Issue:** The `core_cols` used for NaN-based row filtering at line 204 is `["p_win_oof", "p_market_norm", "kakuteijyuni"]`. The `surface` column is copied from `df` at line 190, but only if it exists in `df.columns`. If any fold fails (lines 145-153), rows in `val_idx` positions retain NaN in `oof_p_win_corrected`, `oof_p_market_norm`, etc. The final NaN drop (line 205-206) removes these rows. The `surface` column is then preserved in the output. However, the `surface` column values come from the *original* `df` (line 190: `result[col] = df[col].values`), not from `oof_kakuteijyuni` which was populated from fold values. Since both `df` and the result share the same index structure, this is actually correct, but it relies on the implicit assumption that `df[col].values` at position `i` corresponds to the same row. After the `dropna` + `reset_index` at line 206, the alignment could break if `surface` has NaN values that differ from the core_cols NaN pattern.

**Fix:** No immediate code change needed, but add a defensive assertion after the drop:

```python
result = result[valid].reset_index(drop=True)
if "surface" in result.columns:
    assert result["surface"].notna().all(), "surface column has unexpected NaN after OOF filtering"
```

## Info

### IN-01: Test fixture `surface` type mismatch between training and scoring paths

**File:** `tests/test_race_level_ranker.py:69, 220`
**Issue:** In `sample_oof_df`, `surface` is set as `0` (integer) at line 69. This is correct for training (the ranker checks `df["surface"] == surface_val` with `surface_val` being an int). However, in the scoring tests (e.g., line 220), `if_surface` is set as `0.0` (float). The ranker's `score()` method compares `df["if_surface"] == surface_val` at line 526, where `surface_val` is derived from `float(surface_val)`. This works because `0 == 0.0` in Python/numpy, but it would be cleaner to use consistent types across test fixtures.

**Fix:** Use `0` (int) consistently, or `0.0` (float) consistently, in both training and scoring test fixtures.

### IN-02: `generate_win_oof_predictions` catches broad `(ValueError, RuntimeError)` but folds can fail for other reasons

**File:** `src/models/win_benter_gate.py:145`
**Issue:** The except clause catches `(ValueError, RuntimeError)` but LightGBM or sklearn could raise other exceptions (e.g., `TypeError` from bad column types). If a fold raises an unexpected exception type, the entire OOF generation fails rather than skipping the fold.

**Fix:** Consider catching `Exception` with logging, similar to how `generate_win_selection_oof_frame` (line 1596 in training_pipeline.py) catches `Exception`:

```python
except Exception as exc:
    n_failed += 1
    logger.warning("Skipping Win OOF fold: %s", exc)
```

### IN-03: `compute_ece` first bin is half-open on left, excluding probability=0.0

**File:** `src/models/win_benter_gate.py:231`
**Issue:** The bin mask is `(y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])`. For the first bin (i=0), this becomes `y_prob > 0.0 & y_prob <= 0.1`, which excludes probabilities exactly equal to 0.0. Since `extract_market_probability` clips to `[0.01, 0.99]`, this is unlikely to cause issues in practice, but it deviates from standard ECE implementations that use `>=` for the first bin.

**Fix:** For the first bin, use `y_prob >= bin_boundaries[i]`:

```python
if i == 0:
    mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
else:
    mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
```

---

_Reviewed: 2026-05-28T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
