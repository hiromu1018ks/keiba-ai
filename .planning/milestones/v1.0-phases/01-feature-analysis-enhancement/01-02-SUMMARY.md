---
phase: 01-feature-analysis-enhancement
plan: 02
subsystem: features
tags: [lightgbm, feature-engineering, horse-racing, win-prediction, tdd]
dependency_graph:
  requires:
    - phase: 01-feature-analysis-enhancement/01
      provides: "WinFeatureAnalysis (SHAP/gain分析インフラ), noise feature removal from FEATURE_COLS"
  provides:
    - "5 new HorseHistoryFeatures (distance_change, surface_change, class_drop_bounce, win_dominance, freshness_score)"
    - "odds_to_ability_ratio computation in training and inference paths"
    - "6 new features integrated into WinTwoStageModel.FEATURE_COLS and PlaceTwoStageModel.RETURN_FEATURE_COLS"
  affects: [WinTwoStageModel, PlaceTwoStageModel, HorseHistoryFeatures, training_pipeline]
tech_stack:
  added: []
  patterns: [distance_bin computation helper, dual-path odds_to_ability_ratio (training via _train_submodel + inference via _prepare_features)]
key_files:
  created: []
  modified:
    - src/features/horse_history_features.py
    - src/models/two_stage_return_model.py
    - src/pipelines/training_pipeline.py
    - tests/test_horse_history_features.py
    - tests/test_two_stage_return_model.py
    - tests/test_win_feature_analysis.py
decisions:
  - "distance_bin推論: rowにdistance_binがあれば使用、なければkyori+surfaceから_compute_distance_bin()で計算"
  - "race_context_colsにdistance_bin, kyoriを追加し推論パスからアクセス可能にした"
  - "PlaceTwoStageModel.RETURN_FEATURE_COLSにも6特徴量を追加しWin/Place一貫性を維持"
  - "odds_to_ability_ratio訓練/推論デュアルパス: _train_submodel()で訓練時計算、_prepare_features()で推論時自動計算"

requirements-completed: [FEAT-02]

metrics:
  duration: 20m
  completed: "2026-05-02"
  tasks: 2
  files: 6
  tests: 45
  commits: 4
---

# Phase 1 Plan 02: 新特徴量実装・モデル統合 Summary

6つの単勝特化新特徴量(5履歴ベース + 1確率比)をHorseHistoryFeaturesとWinTwoStageModelに統合し、訓練・推論両パスで計算可能にした。

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | HorseHistoryFeaturesに5つの新特徴量を追加 | `0030f41` (RED), `5da7c9c` (GREEN) | src/features/horse_history_features.py, tests/test_horse_history_features.py |
| 2 | 6新特徴量のWinTwoStageModel統合と推論パス対応 | `164d590` (RED), `bcf6b57` (GREEN) | src/models/two_stage_return_model.py, src/pipelines/training_pipeline.py, tests/test_two_stage_return_model.py, tests/test_win_feature_analysis.py |

## Key Deliverables

### 5 New HorseHistoryFeatures (horse_history_features.py)
- **distance_change**: 距離変更要検知 (1.0=変更, 0.0=同じ, NaN=履歴なし)
- **surface_change**: 芝ダート変更要検知 (1.0=変更, 0.0=同じ, NaN=履歴なし)
- **class_drop_bounce**: クラス降級後リバウンド期待値 (降級+不調時に高い値, 上限10.0)
- **win_dominance**: 勝利時平均フィールドサイズ (勝利なし=0.0, 履歴なし=NaN)
- **freshness_score**: 休息品質x直近フォーム品質 (30-60日最適, 範囲[0.0, 1.0])

### odds_to_ability_ratio (two_stage_return_model.py + training_pipeline.py)
- 市場確率/能力確率比 (>1.0=過大評価, <1.0=過小評価)
- 訓練パス: _train_submodel()でp_ability_win生成後に計算
- 推論パス: _prepare_features()でodds_to_ability_ratio未計算時に自動計算
- 値域: [0.1, 10.0]にクリップ

### _compute_distance_bin() helper
- kyori + surface → distance_bin計算 (FeatureEngine._map_basic_features()と同じロジック)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] entries_histの重複列によるmerge失敗**
- **Found during:** Task 1 GREEN phase
- **Issue:** テストのentries_histにsurface/track_condition_code/distance_bin列が含まれており、races_histとのmergeでsurface_x/surface_yに分裂
- **Fix:** テストのentries_histからraces_hist由来の列を削除
- **Files modified:** tests/test_horse_history_features.py
- **Commit:** `5da7c9c`

**2. [Rule 1 - Bug] rowにdistance_bin/kyoriが含まれない**
- **Found during:** Task 1 GREEN phase
- **Issue:** distance_change計算でgetattr(row, "distance_bin")がNaNになる。race_context_colsにdistance_bin/kyoriが含まれていないため
- **Fix:** (a) race_context_colsにdistance_bin, kyoriを追加 (b) _compute_distance_bin()ヘルパーを追加
- **Files modified:** src/features/horse_history_features.py
- **Commit:** `5da7c9c`

**3. [Rule 1 - Bug] class_drop_bounceのテストデータ閾値問題**
- **Found during:** Task 1 GREEN phase
- **Issue:** kakuteijyuni=[8,10,7] + syussotosu=16の組み合わせでavg_recent=0.5ちょうどとなり、>0.5条件を満たさず
- **Fix:** kakuteijyuniを[8,12,9]に変更しavg_recentを明確に0.5超に
- **Files modified:** tests/test_horse_history_features.py
- **Commit:** `5da7c9c`

**4. [Rule 1 - Bug] Wave 1テストのFEATURE_COLS参照リスト更新漏れ**
- **Found during:** Task 2 GREEN phase
- **Issue:** TestRemoveNoiseFeatures::test_remaining_features_are_subset_of_originalが6新特徴量を認識せず失敗
- **Fix:** original_27リストに6新特徴量を追加
- **Files modified:** tests/test_win_feature_analysis.py
- **Commit:** `bcf6b57`

**5. [Rule 2 - Missing Critical] PlaceTwoStageModel.RETURN_FEATURE_COLSへ6特徴量追加**
- **Found during:** Task 2 GREEN phase
- **Issue:** 既存テスト(TestPlaceTwoStageModel::test_place_return_feature_cols_include_place_specific)がWin FEATURE_COLSの全列がPlace RETURN_FEATURE_COLSに含まれることを検証
- **Fix:** PlaceTwoStageModel.RETURN_FEATURE_COLSに6新特徴量を追加
- **Files modified:** src/models/two_stage_return_model.py
- **Commit:** `bcf6b57`

---

**Total deviations:** 5 auto-fixed (4 bugs, 1 missing critical)
**Impact on plan:** 全て実装の正確性と後方互換性に必須。スコープクリープなし。

## TDD Gate Compliance

| Gate | Task 1 | Task 2 |
|------|--------|--------|
| RED (test failing) | `0030f41` - 12 tests | `164d590` - 5 tests |
| GREEN (all pass) | `5da7c9c` - 58/58 passed | `bcf6b57` - 103/103 passed |
| REFACTOR | Not needed | Not needed |

TDD gate sequence validated: RED -> GREEN commits exist in git log for both tasks.

## Test Results

```
103 passed, 2 skipped in 2.04s (related tests)
1019 passed, 2 skipped in 88.16s (full suite)
```

- TestDistanceChange: 4 tests
- TestSurfaceChange: 3 tests
- TestClassDropBounce: 3 tests
- TestWinDominance: 3 tests
- TestFreshnessScore: 3 tests
- TestNewFeaturesInBaseCols: 2 tests
- TestOddsToAbilityRatio: 4 tests
- TestInferencePathComputation: 3 tests
- TestHistoryFeaturesInFeatureCols: 3 tests

## Verification Results

1. `python -m pytest tests/test_horse_history_features.py -v` -- 58/58 passed
2. `python -m pytest tests/test_two_stage_return_model.py -v` -- 27/27 passed
3. Feature name occurrences in horse_history_features.py: 29 (>= 5)
4. Feature name occurrences in two_stage_return_model.py: 16 (>= 6)
5. odds_to_ability_ratio in training_pipeline.py: 2 (>= 1)
6. odds_to_ability_ratio in two_stage_return_model.py: 6 (>= 2)
7. Full test suite: 1019 passed, 0 failures

## Self-Check: PASSED

- All 6 modified files exist and are tracked
- All 4 commits exist in git log: 0030f41, 5da7c9c, 164d590, bcf6b57
- All 103 related tests pass (0 failures)
- All 1019 full suite tests pass (0 failures)
