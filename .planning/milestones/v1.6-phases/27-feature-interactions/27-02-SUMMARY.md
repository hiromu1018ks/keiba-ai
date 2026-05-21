---
phase: 27-feature-interactions
plan: 02
subsystem: features,models
tags: [interaction-features, tdd, domain-knowledge]
dependency_graph:
  requires: [27-01]
  provides: [INTER-02]
  affects: [interaction_features, two_stage_return_model]
tech_stack:
  added: []
  patterns: [category-product, numeric-product-nan-safe, grade-mapping]
key_files:
  created: []
  modified:
    - src/features/interaction_features.py
    - src/models/two_stage_return_model.py
    - tests/test_interaction_features.py
    - tests/test_two_stage_return_model.py
    - tests/test_win_feature_analysis.py
decisions:
  - 9 new interactions (not 10) to avoid odds_to_ability_ratio dependency issues
  - blood_keito_x_surface uses pd.to_numeric NaN check before category creation
  - weight_x_class uses grade_code mapping (G1=5, G2=4, G3=3, OP=2, default=1.0)
metrics:
  duration: 6min
  completed: "2026-05-15"
  tasks: 2
  files: 5
  tests_added: 13
  tests_passed: 93
---

# Phase 27 Plan 02: INTER-02 Summary

ドメイン知識交互作用項12個 (既存3 + 新規9) をTDD RED->GREENで実装し、Win/Place FEATURE_COLS統合を完了。カテゴリ積3個 + 数値積6個の混合アプローチで、LightGBMが自動発見しにくい非線形関係を明示的に表現。

## Changes

### Task 1: interaction_features.pyに9個の新規交互作用項を追加 (TDD)

**RED (commit 9a7b4f6):** 13個の新規テストを追加 (全て失敗確認)。

**GREEN (commit 1c29f2a):** compute_interaction_features()に9個の新規交互作用を実装。

- カテゴリ積 (3個):
  - `surface_x_distance_bin`: 馬場x距離bin (category型)
  - `blood_keito_x_surface`: 血統系統x馬場 (category型、pd.to_numeric NaNチェック付き)
  - `grade_code_x_distance_bin`: グレードx距離bin (category型)
- 数値積 (6個、.where() NaN安全パターン):
  - `sire_wr_x_distance`: 種牡馬成績x距離
  - `blood_surface_wr_x_condition`: 血統馬場勝率x馬場状態
  - `pace_pressure_x_closing_index`: ペース圧力x追込指数
  - `haron_x_distance`: 末脚x距離
  - `surface_x_past_perf`: 馬場コードx過去成績
  - `weight_x_class`: 馬体xクラス (grade_code数値マッピング)
- `INTERACTION_COLS` 定数を定義 (12個の交互作用名)
- テスト: 25通過 (既存12 + 新規13)

### Task 2: Win/Place FEATURE_COLS更新 (commit d25c84d)

- `WinTwoStageModel.FEATURE_COLS`: 66 -> 78 (+12交互作用)
- `PlaceTwoStageModel.HIT_FEATURE_COLS`: 69 -> 81 (+12)
- `PlaceTwoStageModel.RETURN_FEATURE_COLS`: 71 -> 83 (+12)
- `_prepare_features()` (Win/Place): category変換リストに6個のカテゴリ積列を追加
- `test_two_stage_return_model.py`: feature_df fixtureに12個の交互作用列を追加
- `test_win_feature_analysis.py`: original_allリストに12個を追加

## Feature Counts

| Model | Before | After | Delta |
|-------|--------|-------|-------|
| Win FEATURE_COLS | 66 | 78 | +12 |
| Place HIT_FEATURE_COLS | 69 | 81 | +12 |
| Place RETURN_FEATURE_COLS | 71 | 83 | +12 |
| INTERACTION_COLS | - | 12 | +12 (new constant) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing functionality] test_win_feature_analysis.py original_all更新**
- **Found during:** post-implementation regression check
- **Issue:** test_remaining_features_are_subset_of_original が新規12特徴量を認識せず失敗
- **Fix:** original_allリストに12個の交互作用特徴量を追加
- **Files modified:** tests/test_win_feature_analysis.py
- **Commit:** d25c84d

**2. [Rule 1 - Bug] test_blood_keito_x_surfaceテストデータ修正**
- **Found during:** GREEN phase
- **Issue:** テストのblood_keito_cdに文字列"A","B","C"を使用しpd.to_numericでNaNになりスキップ
- **Fix:** 数値コード 1.0, 2.0, 3.0 に変更し期待値も更新
- **Files modified:** tests/test_interaction_features.py
- **Commit:** 1c29f2a

### Pre-existing Issues (Out of Scope)

3 pipeline tests fail with `record_df has duplicate race_ids: 3600` (RecordFeatures.compute mock issue). Not related to INTER-02 changes. Logged in deferred-items.md.

## Known Stubs

None -- all features are fully wired with real computations.

## Threat Flags

None -- no new network endpoints, auth paths, or schema changes at trust boundaries.

## Self-Check

- [x] src/features/interaction_features.py exists and exports INTERACTION_COLS (12), compute_interaction_features
- [x] src/models/two_stage_return_model.py has 78 Win / 81 Place HIT / 83 Place RETURN features
- [x] _prepare_features() category lists include kyakusitu_x_distance, surface_x_distance_bin, etc.
- [x] All test files exist and pass (excluding pre-existing pipeline failures)
- [x] Commits 9a7b4f6, 1c29f2a, d25c84d exist in git log

## Self-Check: PASSED
