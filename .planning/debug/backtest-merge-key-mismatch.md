---
slug: backtest-merge-key-mismatch
status: resolved
trigger: BacktestEngine merge logic mismatch with TrainingPipeline
created: 2026-05-16
resolved: 2026-05-16
---

# Debug Session: BacktestEngine Merge Logic Mismatch

## Symptoms

BacktestEngine (`src/backtest/engine.py` lines 744-763) was merging DamPedigreeFeatures,
RecordFeatures, and MiningFeatures using `on="race_id"` only, while TrainingPipeline
(`src/pipelines/training_pipeline.py` lines 462-513) correctly uses:
- DamPedigreeFeatures: `on=["race_id", "umaban"]` (horse-level)
- RecordFeatures: `on="race_id"` (race-level, correct for both)
- MiningFeatures: `on=["race_id", "umaban"]` (horse-level)

Additionally, the chained `.drop(columns=...).merge(...).drop_duplicates()` pattern in
BacktestEngine was dropping `umaban` from `feat_df` during the `.drop()` step (because
`dam_df.columns.difference(["race_id"])` includes `umaban`), then performing a
cross-product join on `race_id` only, masked by `.drop_duplicates(subset=["race_id", "umaban"])`.

## Root Cause

**Two bugs in BacktestEngine's feature merge logic (lines 744-763):**

1. **Wrong merge key**: DamPedigreeFeatures and MiningFeatures are horse-level features
   (one row per `(race_id, umaban)`), but BacktestEngine merged them with `on="race_id"`
   instead of `on=["race_id", "umaban"]`. This created cross-product joins within each race.

2. **Chained drop-merge pattern**: The `feat_df.drop(columns=dam_df.columns.difference(["race_id"]))`
   pattern accidentally dropped `umaban` from `feat_df` before merging, since `umaban` is
   part of the feature DataFrame's columns but not excluded from the drop.

The `.drop_duplicates(subset=["race_id", "umaban"])` at the end was a band-aid that masked
the cross-product but silently produced incorrect feature values -- each horse in a race
could receive the wrong horse's dam/mining features.

## Resolution

**Root Cause**: BacktestEngine used `on="race_id"` for horse-level features (DamPedigree,
Mining) and a chained drop-merge pattern that inadvertently dropped `umaban`.

**Fix**: Replaced the 3 chained merge blocks (lines 738-765) with the exact same pattern
used by TrainingPipeline:
- Import `FEATURE_COLS` aliases for each feature module
- Drop only `FEATURE_COLS` from `feat_df` (not all non-race_id columns)
- Merge with correct keys: `on=["race_id", "umaban"]` for horse-level features,
  `on="race_id"` for race-level features (RecordFeatures)
- Remove `.drop_duplicates()` calls (no longer needed with correct merge keys)
- Add NaN fallback for empty feature DataFrames

**Files Changed**: `src/backtest/engine.py` (lines 738-783)

**Verification**: All 1526 tests pass. Ruff lint shows no new errors.

## Evidence

- 2026-05-16: Compared TrainingPipeline._train_submodel (lines 462-513) with
  BacktestEngine (lines 744-763). Found merge key mismatch for DamPedigreeFeatures
  and MiningFeatures.
- 2026-05-16: Verified DamPedigreeFeatures.compute() returns `["race_id", "umaban"] + FEATURE_COLS`
  and MiningFeatures.compute() returns `["race_id", "umaban"] + FEATURE_COLS` -- both
  horse-level, requiring `on=["race_id", "umaban"]`.
- 2026-05-16: Verified RecordFeatures.compute() returns `["race_id"] + FEATURE_COLS` --
  race-level, `on="race_id"` is correct.
- 2026-05-16: Applied fix, verified import passes, all 1526 tests pass.
