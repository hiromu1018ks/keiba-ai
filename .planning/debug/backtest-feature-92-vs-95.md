---
status: resolved
trigger: "LightGBMError: 92 vs 95 features in BacktestEngine inference after previous 80-vs-95 fix"
created: 2026-05-16T14:00:00+09:00
updated: 2026-05-16T14:45:00+09:00
---

## Current Focus

reasoning_checkpoint:
  hypothesis: "compute_relative_features is called in BacktestEngine on feat_df BEFORE HorseHistoryFeatures columns are merged, so 3 relative features that depend on HorseHistory base columns cannot be generated"
  confirming_evidence:
    - "compute_relative_features(feat_df) at engine.py line 783 runs after build_all(), sire, pace, course, dam, record, mining features -- but HorseHistoryFeatures (hist_df_all) is kept separate and only merged inside RacePredictor.predict() at line 81"
    - "The 3 missing relative features are: rel_norm_finish_zscore (base: norm_finish_logit_avg), rel_haron_vs_mean (base: harontimel5_avg), rel_weight_zscore (base: weight_zscore) -- all 3 bases come from HorseHistoryFeatures"
    - "Training pipeline (_train_submodel line 369) merges HorseHistoryFeatures BEFORE compute_relative_features (line 519), so bases are available"
  falsification_test: "If compute_relative_features is moved to after hist_features merge in RacePredictor, the 3 missing features would be generated and the 95-feature count would match"
  fix_rationale: "Move compute_relative_features call from BacktestEngine.run() (line 783) into RacePredictor.predict() after hist_features merge (after line 81), where the HorseHistory base columns are available"
  blind_spots: "RacePredictor is also used by PaperPredictor -- need to verify that PaperPredictor also provides hist_features before calling predict()"

## Symptoms

expected: BacktestEngine inference should provide all 95 features that LightGBM model was trained on
actual: "LightGBMError: The number of features in data (92) is not the same as it was in training data (95)"
errors: "lightgbm.basic.LightGBMError: The number of features in data (92) is not the same as it was in training data (95)"
reproduction: "Run backtest: training succeeds, calibration BT fails with 92 vs 95"
started: "After commits a0c6e96 and ef776dc fixed 12 of 15 missing features"

## Eliminated

## Evidence

- timestamp: 2026-05-16T14:15:00
  checked: "feature_freeze_manifest.json AbilityModel features vs stage1_ability_model.py FEATURE_COLS"
  found: "Both list exactly 95 features, matching perfectly"
  implication: "The manifest is correct, the gap is in the inference pipeline"

- timestamp: 2026-05-16T14:20:00
  checked: "BacktestEngine.run() feature assembly flow vs TrainingPipeline._train_submodel() flow"
  found: |
    Training: HorseHistoryFeatures.merge -> add_race_transforms -> sire/pace/course -> dam/record -> interaction -> mining -> compute_relative_features
    BacktestEngine: build_all -> sire/pace/course -> dam/record -> mining -> compute_relative_features -> (loop: RacePredictor does hist_features merge + interaction)
    KEY DIFFERENCE: compute_relative_features runs AFTER HorseHistoryFeatures.merge in training but BEFORE it in BacktestEngine
  implication: "Relative features that depend on HorseHistory base columns are silently skipped"

- timestamp: 2026-05-16T14:25:00
  checked: "compute_relative_features base columns -- which come from HorseHistoryFeatures vs build_all()"
  found: |
    HorseHistoryFeatures provides: norm_finish_logit_avg, harontimel5_avg, timediff_avg, closing_index_avg, weight_zscore, kyakusitukubun_cd
    build_all() provides: blood_total_wr, sire_wr, fukuoddslow, popularity_rank
    Relative features that fail due to missing bases (not in feat_df at compute_relative_features time):
    1. rel_norm_finish_zscore (base: norm_finish_logit_avg -- from HorseHistory, MISSING)
    2. rel_haron_vs_mean (base: harontimel5_avg -- from HorseHistory, MISSING)
    3. rel_weight_zscore (base: weight_zscore -- from HorseHistory, MISSING)
    Relative features that succeed (bases from build_all or sire):
    4. rel_blood_quality_rank (base: blood_total_wr -- from build_all, PRESENT)
    5. rel_sire_quality_rank (base: sire_wr -- from SireFeatures, PRESENT)
  implication: "Exactly 3 features missing -- matches the 95-92=3 gap"

## Resolution

root_cause: |
  compute_relative_features() is called in BacktestEngine.run() (engine.py line 783) on feat_df
  BEFORE HorseHistoryFeatures columns are merged into the per-race DataFrames inside
  RacePredictor.predict(). Three relative features depend on HorseHistoryFeatures base columns:
  rel_norm_finish_zscore (needs norm_finish_logit_avg), rel_haron_vs_mean (needs harontimel5_avg),
  and rel_weight_zscore (needs weight_zscore). In the training pipeline, HorseHistoryFeatures
  is merged first, so compute_relative_features has all bases available.

fix: |
  1. Remove compute_relative_features(feat_df) from BacktestEngine.run() (engine.py line 783)
  2. Add compute_relative_features(df) to RacePredictor.predict() after hist_features merge
     (after line 81), before add_ability_probs is called
  This ensures the HorseHistory base columns are available when relative features are computed.

verification: |
  - All 1526 tests pass (255 backtest/engine/race_predictor related, 1271 others)
  - No new lint errors introduced
  - PaperPredictor verified: pre-computes HorseHistoryFeatures into feat_df before predict(),
    so base columns are available for compute_relative_features
files_changed: [src/backtest/engine.py, src/backtest/race_predictor.py]
