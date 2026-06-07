---
status: resolved
trigger: "PT predict 失敗 — LightGBM categorical_feature do not match。学習時(ParquetStore)と推論時(DB直接)の特徴量カラムが不一致"
created: 2026-06-07
updated: 2026-06-07
---

# Debug Session: pt-cat-feature-mismatch

## Symptoms

- **Expected behavior:** PT predict モードで推論時に学習済みモデルと同じ特徴量カラムが渡され、正常に予測が完了する
- **Actual behavior:** ValueError: train and valid dataset categorical_feature do not match. が発生
- **Error messages:** ValueError at src/models/stage1_ability_model.py:358 → booster.predict()
- **Timeline:** run_train.py (ParquetStore経由) で学習 → run_paper_trading.py predict (DB直接経由) で推論時に発生
- **Reproduction:** 学習後に PT predict を実行

## Root Cause Hypothesis (user-provided)

- 学習時（run_train.py）は ParquetStore から特徴量を構築
- PT predict 時は DB 直接読み取りから特徴量を構築
- 両者の特徴量カラム（特にカテゴリカル系）が一致していない
- Parquet → FeatureEngine vs DB → FeatureBuilder の経路違いで、特徴量の欠落や順序の差異が生じている可能性

## Current Focus

- **hypothesis:** Parquet経路とDB経路の FeatureEngine/FeatureBuilder で生成される特徴量カラムが不一致
- **test:** 推論データのカラム数・カラム名を学習時の FEATURE_COLS と比較する診断
- **expecting:** 両経路で同一の FEATURE_COLS が生成されること
- **next_action:** 推論パイプラインのコードを追跡し、特徴量生成経路の差異を特定して修正する
- **reasoning_checkpoint:** LightGBM の categorical_feature mismatch は特徴量の過不足またはカテゴリカル指定の不一致が原因。推論経路の特徴量生成を学習経路と一致させる必要がある

## Evidence

- 2026-06-07: Reproduced root cause via controlled LightGBM test -- predicting with fewer columns than training triggers `ValueError: train and valid dataset categorical_feature do not match`
- 2026-06-07: AbilityModel._prepare_features (line 241) uses `available_cols = [c for c in self.FEATURE_COLS if c in df.columns]` which silently drops missing columns instead of filling with NaN
- 2026-06-07: EVCorrectionModel._prepare_features (line 271-284) already has the correct pattern: fills missing columns with `float("nan")` then selects `df[self.FEATURE_COLS]`
- 2026-06-07: WinTwoStageModel._prepare_features (line 319) has the same bug as AbilityModel
- 2026-06-07: AbilityModel has 167 FEATURE_COLS, 8 of which are categorical. If any single column is missing at inference time, LightGBM receives fewer columns and the categorical feature set differs from training

## Eliminated

- (none yet)

## Resolution

- **root_cause:** AbilityModel._prepare_features silently drops missing FEATURE_COLS instead of filling them with NaN. When PT inference data lacks some feature columns (e.g., enrichment module output missing), the resulting DataFrame has fewer columns and a different categorical feature set than what LightGBM saw during training, causing `categorical_feature do not match` error.
- **fix:** Adopt the same NaN-fill pattern from EVCorrectionModel._prepare_features: detect missing columns, fill with `float("nan")`, then select `df[self.FEATURE_COLS]` to guarantee the same column count and order as training. Apply same fix to WinTwoStageModel._prepare_features which has the identical bug.
- **verification:** All 53 tests in test_stage1_ability, test_two_stage_return_model, test_place_ability_model pass. NaN-fill for missing columns confirmed working via LightGBM synthetic test.
- **files_changed:** src/models/stage1_ability_model.py, src/models/two_stage_return_model.py, src/models/place_ability_model.py, tests/test_place_ability_model.py
