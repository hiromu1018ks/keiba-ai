---
status: resolved
trigger: "BacktestEngine test prediction: LightGBM expects 95 features but gets 80 (15 missing)"
created: "2026-05-16T12:15:00+09:00"
updated: "2026-05-16T12:45:00+09:00"
---

# Debug: feature-mismatch-80-vs-95

## Symptoms

- **Expected:** AbilityModel predict receives 95 features (matching FEATURE_COLS)
- **Actual:** `LightGBMError: The number of features in data (80) is not the same as it was in training data (95)`
- **Error location:** `src/models/stage1_ability_model.py:270` -> `booster.predict()` -> LightGBM fatal
- **When:** BacktestEngine test phase (after training succeeds). Training phase works with 95 features.
- **Context:** Phase 26-27 added ~15 new features (bloodline, record, mining, interaction, TE, relative). Training pipeline includes them, but BacktestEngine prediction path may not.

## Current Focus

hypothesis: "BacktestEngine.run() prediction path doesn't generate all Phase 26-27 features -- training pipeline adds them via TrainingPipelineV5._train_submodel() but engine.py only calls build_all() + a few explicit feature modules, missing 15 features"
next_action: "root cause confirmed"

## Evidence

- timestamp: 2026-05-16T12:30:00
  source: code comparison
  finding: |
    Training pipeline (_train_submodel in training_pipeline.py) generates ALL features
    before AbilityModel.train_oof() is called. BacktestEngine.run() (engine.py) + RacePredictor.predict()
    together generate only a subset.

    MISSING from BacktestEngine + RacePredictor (15 columns total):

    1. DamPedigreeFeatures (4 columns):
       - dam_wr, dam_surface_wr, dam_prize_log, breeder_strength
       - Training: _train_submodel lines 458-472
       - BacktestEngine: NOT called anywhere

    2. RecordFeatures (1 column):
       - course_record_time
       - Training: _train_submodel lines 474-491
       - BacktestEngine: NOT called anywhere

    3. MiningFeatures (3 columns):
       - dm_time_rank, dm_time_zscore, dm_confidence_range
       - Training: _train_submodel lines 499-513
       - BacktestEngine: NOT called anywhere

    4. compute_relative_features (5 columns):
       - rel_norm_finish_zscore, rel_haron_vs_mean, rel_blood_quality_rank,
         rel_sire_quality_rank, rel_weight_zscore
       - Training: _train_submodel line 517-519
       - BacktestEngine: NOT called anywhere

    5. BMS extended features (2 columns):
       - bms_distance_wr, bms_surface_wr
       - Training: _train_submodel sire_cols_needed includes these (line 449)
       - BacktestEngine: sire_cols_needed at engine.py line 691 does NOT include them

    Total missing: 4 + 1 + 3 + 5 + 2 = 15 features (matches the 95 - 80 = 15 gap)

## Eliminated

- FeatureEngine.build_all() -- it correctly generates bloodline, intra_race, odds_dynamics, market_bias, difficulty features. The gap is NOT in build_all().
- RacePredictor interaction features -- compute_interaction_features IS called in RacePredictor (line 99), so pace_pressure, pace_scenario_fit, actual_pace_fit etc. are generated.
- HorseHistoryFeatures -- correctly called in BacktestEngine.run() (line 662-663) and merged in RacePredictor (line 81).

## Resolution

root_cause: |
  BacktestEngine.run() in engine.py does not call 4 feature modules that the training pipeline
  (_train_submodel) calls before AbilityModel training:
  (1) DamPedigreeFeatures, (2) RecordFeatures, (3) MiningFeatures, (4) compute_relative_features.
  Additionally, the sire feature columns in engine.py (line 691) exclude bms_distance_wr and
  bms_surface_wr that training includes (training_pipeline.py line 449).
  When AbilityModel.add_ability_probs() calls _prepare_features() during backtest prediction,
  the DataFrame has only 80 of the 95 FEATURE_COLS, causing LightGBM to fail.

fix: |
  In engine.py BacktestEngine.run(), add the missing feature generation steps after the existing
  pace/course features block (after line 734), before the per-race loop (line 738):

  1. Add bms_distance_wr and bms_surface_wr to _sire_cols_needed (engine.py line 691)
  2. Call DamPedigreeFeatures.compute() and merge results onto feat_df
  3. Call RecordFeatures.compute() and merge results onto feat_df
  4. Call MiningFeatures.compute() and merge results onto feat_df
  5. Call compute_relative_features(feat_df) to add the 5 relative comparison columns

  These must be added in engine.py AFTER feat_df is fully built from build_all() + hist + sire +
  pace + course features, but BEFORE the per-race loop feeds data to RacePredictor (which calls
  AbilityModel.add_ability_probs internally).

specialist_hint: python
