---
status: resolved
trigger: "train and valid dataset categorical_feature do not match — LightGBM error during turf training after surface constant column removal"
created: 2026-05-20
updated: 2026-05-21
---

# Debug: categorical_feature mismatch

## Error
```
train and valid dataset categorical_feature do not match.
```

## Context
- Occurs during `turf/win_predict` timing in training phase
- After commits: ensemble correlation penalty, tanoddslow fix, surface constant col removal, kakuteijyuni=0 exclusion
- Ensemble training succeeded (turf LGB-CAT corr dropped to 0.5462 from 0.9226)
- Error happens AFTER ensemble, during win_return or subsequent model training

## Suspected commits
1. `62e8412` — drop constant surface columns in per-surface ensemble training
2. `3679385` — add Optuna correlation penalty (changes model hyperparams)

## Current Focus

hypothesis: CONFIRMED - Commit 62e8412 drops surface column from ensemble training features, but WinTwoStageModel._prepare_features still includes surface as a categorical column. When StackedEnsemble.predict() calls lgbm_model.predict(X), LightGBM internally runs _data_from_pandas() which detects surface as a categorical column (2 cat cols: surface + distance_bin), but the model was trained with only distance_bin (1 cat col) because surface was dropped. This triggers len(cat_cols) != len(pandas_categorical).
test: Reproduced with synthetic data - model trained without surface column fails when predicting DataFrame with surface column as category
expecting: Error reproduced and fix verified
next_action: Commit fix

## Symptoms

expected: Training pipeline completes without error
actual: LightGBM raises "train and valid dataset categorical_feature do not match" after turf/win_predict
errors: train and valid dataset categorical_feature do not match.
reproduction: Run python scripts/run_train.py --start 20200101 --end 20231231 --ensemble
started: After commits 62e8412 and 3679385

## Eliminated

- hypothesis: WinTwoStageModel.train_return_model() has categorical mismatch
  evidence: win_return timing logged successfully (0.294s), indicating no error in this step
  timestamp: 2026-05-21

- hypothesis: StackedEnsemble.train() has categorical mismatch in K-fold OOF
  evidence: win_hit_ensemble timing logged successfully (368.676s), indicating ensemble completed
  timestamp: 2026-05-21

- hypothesis: EVCorrectionModel has different category sets between train/valid split within same DataFrame
  evidence: Simple reproduction with synthetic data does not trigger the error; same DataFrame split always produces same category dtypes
  timestamp: 2026-05-21

## Evidence

- timestamp: 2026-05-21
  checked: LightGBM basic.py line 850 - error condition
  found: Error occurs when len(cat_cols) != len(pandas_categorical) — i.e., when the NUMBER of categorical columns differs between train and valid datasets
  implication: The categorical column count must differ between training and prediction DataFrames

- timestamp: 2026-05-21
  checked: LightGBM Booster.predict() code path (basic.py line 1157-1163)
  found: Booster.predict(data) calls _data_from_pandas() with self.pandas_categorical from training. _data_from_pandas counts categorical columns in input data and compares with training pandas_categorical.
  implication: If prediction DataFrame has more categorical columns than training, mismatch error occurs

- timestamp: 2026-05-21
  checked: Root cause chain: commit 62e8412 drops "surface" from ensemble features → StackedEnsemble trains LightGBM without surface column → pandas_categorical has 1 entry (distance_bin only) → WinTwoStageModel.predict_ev() calls _prepare_features() which includes surface as category → StackedEnsemble.predict(X) passes X with surface to lgbm_model.predict(X) → _data_from_pandas sees 2 cat cols but pandas_categorical has 1 → ERROR
  found: Exact reproduction confirmed: model.predict(df_with_surface) fails, model.predict(df_without_surface) succeeds
  implication: Fix must ensure predict() only passes columns the model was trained on

- timestamp: 2026-05-21
  checked: Synthetic reproduction
  found: model trained on features without surface column, then predict with features including surface (as category) → "train and valid dataset categorical_feature do not match." → predict with filtered features (matching training columns) → succeeds
  implication: Fix confirmed

## Resolution

root_cause: Commit 62e8412 dropped "surface" column from StackedEnsemble training features but WinTwoStageModel.predict_ev() still passes a DataFrame with "surface" (as category dtype) to StackedEnsemble.predict(). LightGBM's internal _data_from_pandas() detects 2 categorical columns (surface + distance_bin) in prediction data but the model was trained with only 1 (distance_bin), triggering the "train and valid dataset categorical_feature do not match" error.
fix: StackedEnsemble now stores _train_feature_names during train() and filters input columns to match training features in predict(). This ensures only the columns present during training are passed to lgbm_model.predict().
verification: Synthetic reproduction confirms fix works; 74 related tests pass; full suite: 64 failed (pre-existing) vs 75 failed (before fix), 1864 passed vs 1853 passed.
files_changed: [src/models/stacked_ensemble.py]
