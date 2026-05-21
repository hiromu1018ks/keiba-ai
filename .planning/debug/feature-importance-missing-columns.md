---
status: resolved
trigger: "analyze_feature_importance.py fails: only 34/122 model features found in horse_features.parquet"
created: 2026-05-12T00:00:00
updated: 2026-05-12T00:10:00
---

## Current Focus

hypothesis: CONFIRMED - training_pipeline never saves the full feature set after _train_submodel adds extended features
test: Modified _train_submodel to return df_oof alongside SubmodelSet, save combined features via save_features()
expecting: After re-training, horse_features.parquet will contain all model feature columns
next_action: await human verification that re-training produces the correct file

## Symptoms

expected: analyze_feature_importance.py --tier-report loads all 122 model features from horse_features.parquet
actual: "特徴量データに一部列が欠落: 34/122" -- only 34 of 122 features found
errors: warning log about missing columns, analysis fails to produce real feature importance
reproduction: run python scripts/analyze_feature_importance.py --tier-report --model-dir data/models
started: always been broken -- horse_features.parquet never contained full feature set

## Eliminated

## Evidence

- timestamp: 2026-05-12T00:01
  checked: analyze_feature_importance.py lines 454-490 (_load_features_for_analysis)
  found: Loads data/features/horse_features.parquet via ParquetStore, checks model.feature_name() against columns, logs "特徴量データに一部列が欠落" when columns missing
  implication: The file exists but is missing columns that models need

- timestamp: 2026-05-12T00:02
  checked: training_pipeline.py run() flow (lines 164-288)
  found: build_all() creates feat_df with basic features, then add_distance_band_features adds distance_band. But _train_submodel() works on copies (subset_df = feat_df[...].copy()) and adds horse_history, pace_aptitude, course_features, sire_features, interaction_features to LOCAL copies. These are never propagated back to feat_df.
  implication: feat_df never contains the full feature set that models are trained on

- timestamp: 2026-05-12T00:03
  checked: FeatureEngine.build_all() cache mechanism (feature_engine.py lines 373-381)
  found: build_all() writes its result_df to a feature cache under data/.feature_cache/ but never writes to data/features/horse_features.parquet. The horse_features.parquet file must have been created by some other mechanism or manually.
  implication: There is NO code that saves the full feature set to horse_features.parquet

- timestamp: 2026-05-12T00:04
  checked: src/db/readers.py save_features() (line 319-320) and run_train.py
  found: save_features(store, df) exists but is never called anywhere in run_train.py or training_pipeline.py. There is no code path that saves features after training.
  implication: The save_features function exists but was never wired into the training pipeline

- timestamp: 2026-05-12T00:05
  checked: Fix implementation in training_pipeline.py
  found: Modified _train_submodel return type from SubmodelSet to tuple[SubmodelSet, pd.DataFrame]. Added df_oof_for_save copy before confirmed_odds deletion. In run(), collect oof_dfs from all surfaces, concat, and save via save_features(). All 23 tests pass.
  implication: After re-training, horse_features.parquet will contain the complete feature set

## Resolution

root_cause: The training pipeline builds features in stages. build_all() produces basic features (cached), then _train_submodel() adds horse_history/pace/course/sire/interaction features to LOCAL copies only. The complete feature DataFrame (df_oof) is never saved. horse_features.parquet (if it exists) contains only build_all() output, missing ~88 columns.
fix: Changed _train_submodel to return (SubmodelSet, df_oof_for_save) tuple. Added code in run() to collect all df_oof DataFrames, concatenate them, and save to data/features/horse_features.parquet via save_features().
verification: All 23 tests pass (test_training_pipeline.py + test_ensemble_gate_propagation.py). Re-training required to generate the actual file.
files_changed: [src/pipelines/training_pipeline.py, tests/test_training_pipeline.py]
