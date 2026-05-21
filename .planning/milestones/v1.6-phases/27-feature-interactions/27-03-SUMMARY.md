---
phase: 27-feature-interactions
plan: 03
subsystem: features,models,pipeline
tags: [target-encoding, oof-safe, high-cardinality, tdd]
dependency_graph:
  requires: [27-01, 27-02]
  provides: [INTER-03]
  affects: [target_encoding, two_stage_return_model, training_pipeline]
tech_stack:
  added: []
  patterns: [expanding-window-fold, beta-smoothing, cold-start-global-mean]
key_files:
  created:
    - src/features/target_encoding.py
  modified:
    - src/models/two_stage_return_model.py
    - src/pipelines/training_pipeline.py
    - tests/test_target_encoding.py
    - tests/test_two_stage_return_model.py
    - tests/test_win_feature_analysis.py
decisions:
  - Stage1にはTE追加せず (TE target == Stage1 targetでOOFリークの可能性)
  - Expanding window最初foldのtrain-only行にfull-data encoding mapでTE補完
  - TE cat_colsはdf_oofに存在する列のみを使用 (欠損時安全にスキップ)
metrics:
  duration: 9min
  completed: "2026-05-15"
  tasks: 2
  files: 6
  tests_added: 12
  tests_passed: 120
---

# Phase 27 Plan 03: INTER-03 Summary

高カーディナリティカテゴリ変数(blood_keito_cd, kisyucode, chokyosicode)をOOF-safe target encodingで数値化。3-fold expanding window (AbilityModel.train_oofと同一境界) + Beta(1,10) smoothing + cold start global mean補完。Stage2 (Win/Place) のみにTE特徴量を追加し、Stage1には追加しない安全性設計。

## Changes

### Task 1: target_encoding.py新規作成 -- TargetEncoderクラス (TDD)

**RED (commit 549bb77):** 12個のテストを追加 (全て失敗確認)。

**GREEN (commit b107a56):** TargetEncoderクラスを実装。

- `fit_transform_oof()`: 3-fold expanding windowでOOF安全なTEを計算
  - `AbilityModel.train_oof()`と同一のfold境界 (race_dateベース)
  - 各fold: train_mask = race_date < boundary, test_mask = boundary <= race_date < next
  - Smoothing: `(cat_sum + smoothing * fold_global_mean) / (cat_count + smoothing)`
  - 未知カテゴリ (cold start): fold_global_meanで補完
  - 最初foldのtrain-only行: full-data encoding mapで補完
- `transform()`: 学習済みencoding_maps_を使用して新規データにTE適用
  - 未知カテゴリ: global_mean_で補完
- 定数:
  - `TE_FEATURE_COLS`: ["te_blood_keito_cd"] (Stage1用 -- 実際はStage1に未追加)
  - `TE_STAGE2_FEATURE_COLS`: ["te_blood_keito_cd", "te_kisyucode", "te_chokyosicode"]
- テスト: 12通過

### Task 2: FEATURE_COLS更新 + _train_submodel()統合 (commit 3c05748)

- `WinTwoStageModel.FEATURE_COLS`: 78 -> 81 (+3 TE特徴量)
- `PlaceTwoStageModel.HIT_FEATURE_COLS`: 81 -> 84 (+3)
- `PlaceTwoStageModel.RETURN_FEATURE_COLS`: 83 -> 86 (+3)
- `Stage1AbilityModel.FEATURE_COLS`: 変更なし (94のまま、安全性優先)
- `_train_submodel()`: `TargetEncoder.fit_transform_oof()`をStage1 OOF直後に挿入
  - `blood_keito_cd`, `kisyucode`, `chokyosicode`がdf_oofに存在する場合のみ実行
  - TimingContext付きでログ出力
- テストfixtureに3個のTE特徴量を追加
- `test_win_feature_analysis.py` original_allリストに3個追加
- テスト: 120通過 (POST_RACE漏洩テスト含む)

## Feature Counts

| Model | Before | After | Delta |
|-------|--------|-------|-------|
| Win FEATURE_COLS | 78 | 81 | +3 |
| Place HIT_FEATURE_COLS | 81 | 84 | +3 |
| Place RETURN_FEATURE_COLS | 83 | 86 | +3 |
| Stage1 FEATURE_COLS | 94 | 94 | +0 (安全性) |
| TE_FEATURE_COLS | - | 1 | +1 (new constant) |
| TE_STAGE2_FEATURE_COLS | - | 3 | +3 (new constant) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Expanding window最初foldのtrain-only行にNaNが残る問題**
- **Found during:** TDD GREEN phase (test_unsorted_dataframe失敗)
- **Issue:** 3-fold expanding windowで最初のfoldのtrainデータ行(境界より前の行)がどのfoldのtestにも含まれず、TE値がNaNのまま残る
- **Fix:** 全fold処理後、NaNが残っている行にfull-data encoding mapからTE値を設定
- **Files modified:** src/features/target_encoding.py
- **Commit:** b107a56

**2. [Rule 2 - Missing functionality] test_win_feature_analysis.py original_all更新**
- **Found during:** Task 2完了後の回帰テスト
- **Issue:** test_remaining_features_are_subset_of_originalが新規3特徴量を認識せず失敗
- **Fix:** original_allリストにte_blood_keito_cd, te_kisyucode, te_chokyosicodeを追加
- **Files modified:** tests/test_win_feature_analysis.py
- **Commit:** 3c05748

### Pre-existing Issues (Out of Scope)

3 pipeline tests fail with `record_df has duplicate race_ids: 3600` (RecordFeatures.compute mock issue). Not related to INTER-03 changes. Logged in deferred-items.md.

## Known Stubs

None -- all features are fully wired with real computations.

## Threat Flags

None -- no new network endpoints, auth paths, or schema changes at trust boundaries.

## Self-Check

- [x] src/features/target_encoding.py exists and exports TargetEncoder, TE_FEATURE_COLS (1), TE_STAGE2_FEATURE_COLS (3)
- [x] src/models/two_stage_return_model.py has 81 Win / 84 Place HIT / 86 Place RETURN features
- [x] Stage1AbilityModel.FEATURE_COLS unchanged at 94 (safety decision)
- [x] src/pipelines/training_pipeline.py calls TargetEncoder.fit_transform_oof() after Stage1 OOF
- [x] All test files exist and pass (120 tests, excluding pre-existing pipeline failures)
- [x] Commits 549bb77, b107a56, 3c05748 exist in git log

## Self-Check: PASSED
