---
phase: 26-everydb2-new-features
plan: 03
subsystem: features
tags: [relative_features, groupby_race_id, z-score, rank, vs_mean, LightGBM]

# Dependency graph
requires:
  - phase: 25-quick-win-wire-existing
    provides: _train_submodel() integration pattern
  - phase: 26-01
    provides: MiningFeatures, _train_submodel() Group F block
  - phase: 26-02
    provides: DamPedigreeFeatures, RecordFeatures, SireFeatures BMS extension
provides:
  - compute_relative_features(): 7レース内相対比較特徴量を生成するモジュール
  - FEATURE_COLS更新: Stage1(94), Win(64), Place HIT(64), Place RETURN(68)
affects: [feature-engineering]

# Tech tracking
tech-stack:
  added: []
patterns:
  - "groupby('race_id') z-score/rank/vs_mean: intra-race relative transforms for per-horse features"
  - "std=0 fallback: replace(0,1) to prevent NaN/inf in z-score when all race values identical"
  - "_train_submodel() Group G integration: after mining, before market model"

key-files:
  created:
    - src/features/relative_features.py
    - tests/test_relative_features.py
  modified:
    - src/pipelines/training_pipeline.py
    - src/models/stage1_ability_model.py
    - src/models/two_stage_return_model.py
    - tests/test_two_stage_return_model.py
    - tests/test_win_feature_analysis.py

key-decisions:
  - "3 transform types: z-score (2 features), vs_mean (1), rank (4) for 7 relative features"
  - "Place RETURN_FEATURE_COLSにWin全relative featuresを追加 (既存テスト制約: Win全列がPlace RETURNに含まれる必要)"

patterns-established:
  - "relative features: groupby('race_id') + transform per base feature, skip if base missing"
  - "std=0 safety: fillna(0).replace(0,1) for z-score denominator"

requirements-completed: [DATA-03]

# Metrics
duration: 19min
completed: 2026-05-14
---

# Phase 26 Plan 03: レース内相対比較特徴量 Summary

**groupby("race_id")で7相対特徴量 (z-score/rank/vs_mean) を生成し、_train_submodel()に統合**

## Performance

- **Duration:** 19 min
- **Started:** 2026-05-14T14:22:36Z
- **Completed:** 2026-05-14T14:41:09Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- relative_features.py実装: 7特徴量 (2 z-score + 1 vs_mean + 4 rank) をgroupby("race_id")で生成
- std=0フォールバック: レース内全馬同値時にz-score=0.0 (NaN/inf防止)
- 欠損base特徴量はスキップ (エラーなし)、NaNは伝播
- _train_submodel() Group G block追加: mining features後、market model前に実行
- Stage1 90->94, Win 61->64, Place HIT 61->64, Place RETURN 63->68
- 22新テスト追加, 89関連テスト全通過

## Task Commits

Each task was committed atomically:

1. **Task 1: relative_features.py + テスト (TDD RED->GREEN)** - `c87c9d0` (feat)
2. **Task 2: _train_submodel()統合 + FEATURE_COLS更新** - `26b036c` (feat)

## Files Created/Modified
- `src/features/relative_features.py` - compute_relative_features() + RELATIVE_FEATURE_COLS (7特徴量)
- `tests/test_relative_features.py` - 22テスト (mock-based, DB不要)
- `src/pipelines/training_pipeline.py` - Group G relative features block追加
- `src/models/stage1_ability_model.py` - 4特徴量追加 (90->94)
- `src/models/two_stage_return_model.py` - Win +3 (64), HIT +3 (64), RETURN +5 (68)
- `tests/test_two_stage_return_model.py` - feature_df fixtureに7列追加
- `tests/test_win_feature_analysis.py` - original_allリストに3特徴量追加

## Decisions Made
- 3 transform types: z-score (norm_finish, weight), vs_mean (harontime), rank (timediff/blood/sire/closing)
- Place RETURN_FEATURE_COLSにWin全relative featuresを含めた (既存テストが「Win全列がPlace RETURNに含まれる」ことを検証)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Place RETURN_FEATURE_COLS不足でテスト失敗**
- **Found during:** Task 2 (テスト実行時)
- **Issue:** Winにrel_closing_index_rankを追加したがPlace RETURNに未追加。test_place_return_feature_cols_include_place_specificが失敗
- **Fix:** Place RETURN_FEATURE_COLSにもrel_closing_index_rankを追加
- **Files modified:** src/models/two_stage_return_model.py
- **Verification:** test_two_stage_return_model.py全テスト通過
- **Committed in:** 26b036c (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** 既存テスト制約 (Win全列がPlace RETURNに含まれる) への対応。スコープクリープなし。

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- relative_featuresモジュール完成、_train_submodel()統合済み
- POST_RACE漏洩テスト自動検証済み (4/4通過)
- Phase 26全3プラン完了 (MiningFeatures + DamPedigreeFeatures/RecordFeatures + RelativeFeatures)

---
*Phase: 26-everydb2-new-features*
*Completed: 2026-05-14*

## Self-Check: PASSED

All 7 files verified. Both commits (c87c9d0, 26b036c) verified in git log.
