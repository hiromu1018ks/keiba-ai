---
phase: 27-feature-interactions
plan: 01
subsystem: features,models,pipeline
tags: [relative-features, interaction-features, tdd]
dependency_graph:
  requires: [26-03]
  provides: [INTER-01]
  affects: [stage1_ability_model, two_stage_return_model, training_pipeline]
tech_stack:
  added: [numpy, pandas groupby transforms]
  patterns: [zscore, rank descending, vs_mean]
key_files:
  created: []
  modified:
    - src/features/relative_features.py
    - src/models/stage1_ability_model.py
    - src/models/two_stage_return_model.py
    - src/pipelines/training_pipeline.py
    - tests/test_relative_features.py
    - tests/test_two_stage_return_model.py
    - tests/test_win_feature_analysis.py
decisions:
  - D-04: _BASE_FEATURES spec-list pattern for odds features (same zscore transform)
  - D-01: Stage1 FEATURE_COLS gets rel_weight_zscore (Phase 26 remainder)
  - compute_stage2_relative_features() is a separate function because p_ability_win depends on Stage1 OOF output
metrics:
  duration: 12min
  completed: "2026-05-15"
  tasks: 2
  files: 7
  tests_added: 25
  tests_passed: 95
---

# Phase 27 Plan 01: INTER-01 Summary

オッズ相対特徴量(rel_fuku_odds_zscore, rel_popularity_rank_zscore) + Stage2能力値相対特徴量(rel_p_ability_win_zscore/rank, rel_odds_ability_deviation) を追加。TDD RED->GREENで実装し、FEATURE_COLS統合とパイプライン統合を完了。

## Changes

### Task 1: relative_features.py拡張 (commit 9fe94b3)

- `_BASE_FEATURES`: 7->9 (fukuoddslow, popularity_rankをzscoreで追加)
- `RELATIVE_FEATURE_COLS`: 7->9
- `STAGE2_RELATIVE_FEATURE_COLS`: 新規定数 (3特徴量)
- `compute_stage2_relative_features()`: 新規関数
  - p_ability_winのzscore + descending rank
  - odds_to_ability_ratioのzscore
  - base列不在時はNaN列を生成 (エラーなし、odds_deviation_features.pyパターン踏襲)
- テスト: 38テスト通過 (既存21 + 新規17)

### Task 2: FEATURE_COLS更新 + パイプライン統合 (commit 685618a)

- `Stage1AbilityModel.FEATURE_COLS`: 94->95 (rel_weight_zscore追加, per D-01)
- `WinTwoStageModel.FEATURE_COLS`: 61->66 (5新規相対特徴量)
- `PlaceTwoStageModel.HIT_FEATURE_COLS`: 64->69 (5新規相対特徴量)
- `PlaceTwoStageModel.RETURN_FEATURE_COLS`: 66->71 (5新規相対特徴量)
- `_train_submodel()`: compute_stage2_relative_features()をodds_to_ability_ratio計算直後に挿入
- テスト: 95テスト通過 (POST_RACE漏洩テスト含む)

### 追加修正 (commit 59276f4)

- `test_win_feature_analysis.py`のoriginal_allリストに5新規特徴量を追加

## Feature Counts

| Model | Before | After | Delta |
|-------|--------|-------|-------|
| Stage1 FEATURE_COLS | 94 | 95 | +1 (rel_weight_zscore) |
| Win FEATURE_COLS | 61 | 66 | +5 |
| Place HIT_FEATURE_COLS | 64 | 69 | +5 |
| Place RETURN_FEATURE_COLS | 66 | 71 | +5 |
| RELATIVE_FEATURE_COLS | 7 | 9 | +2 |
| STAGE2_RELATIVE_FEATURE_COLS | - | 3 | +3 (new) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing functionality] test_win_feature_analysis.py original_all更新**
- **Found during:** post-commit regression check
- **Issue:** test_remaining_features_are_subset_of_original が新しい5特徴量を認識せず失敗
- **Fix:** original_allリストに5新規特徴量を追加
- **Files modified:** tests/test_win_feature_analysis.py
- **Commit:** 59276f4

### Pre-existing Issues (Out of Scope)

3 pipeline tests fail with `record_df has duplicate race_ids: 3600` (RecordFeatures.compute mock issue). Not related to INTER-01 changes. Logged in deferred-items.md.

## Known Stubs

None -- all features are fully wired with real computations.

## Threat Flags

None -- no new network endpoints, auth paths, or schema changes at trust boundaries.

## Self-Check

- [x] src/features/relative_features.py exists and exports RELATIVE_FEATURE_COLS (9), STAGE2_RELATIVE_FEATURE_COLS (3), compute_relative_features, compute_stage2_relative_features
- [x] src/models/stage1_ability_model.py has 95 FEATURE_COLS including rel_weight_zscore
- [x] src/models/two_stage_return_model.py has 66 Win / 69 Place HIT / 71 Place RETURN features
- [x] src/pipelines/training_pipeline.py calls compute_stage2_relative_features
- [x] All test files exist and pass (excluding pre-existing pipeline failures)
- [x] Commits 9fe94b3, 685618a, 59276f4 exist in git log

## Self-Check: PASSED
