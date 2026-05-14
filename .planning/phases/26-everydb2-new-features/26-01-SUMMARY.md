---
phase: 26-everydb2-new-features
plan: 01
subsystem: features
tags: [n_mining, PIT-audit, DataKubun, wide-to-long, dm_time_rank, LightGBM]

# Dependency graph
requires:
  - phase: 23-safety-gate
    provides: POST_RACE漏洩検出CIテスト
  - phase: 25-quick-win-wire-existing
    provides: _train_submodel()統合パターン
provides:
  - MiningFeatures: n_mining DataKubun=3から4特徴量を生成するモジュール
  - PIT監査ドキュメント: n_mining 82列全PRE分類
  - FEATURE_COLS更新: Stage1(83), Win(54), Place HIT(57), Place RETURN(59)
affects: [26-02, 26-03, feature-engineering]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "wide-to-long pivot: n_mining 18頭wide formatをlongに変換するパターン"
    - "DataKubun=3フィルタ + DataKubun=2フォールバックパターン"
    - "_train_submodel() Group F: interaction features後、market model前にmerging"

key-files:
  created:
    - src/features/mining_features.py
    - docs/everydb2/26-mining-pit-audit.md
    - tests/test_mining_features.py
  modified:
    - src/pipelines/training_pipeline.py
    - src/models/stage1_ability_model.py
    - src/models/two_stage_return_model.py
    - tests/test_two_stage_return_model.py
    - tests/test_win_feature_analysis.py

key-decisions:
  - "DataKubun=3 (直前予想) を主ソースとし、不可の場合はDataKubun=2にフォールバック"
  - "Place RETURN_FEATURE_COLSにWin全mining featuresを追加 (既存テスト制約: Win全列がPlace RETURNに含まれる必要)"
  - "FEATURE_COLSはモジュールレベル定数 (クラス属性ではない) -- importで別名を使用"

patterns-established:
  - "wide-to-long pivot: 18頭分の列(Umaban1..18, DMTime1..18等)をlong形式に変換"
  - "_train_submodel() Group F integration: compute() -> drop duplicates -> merge"

requirements-completed: [DATA-04]

# Metrics
duration: 18min
completed: 2026-05-14
---

# Phase 26 Plan 01: n_mining PIT監査 + MiningFeatures実装 Summary

**n_mining DataKubun=3からdm_time_rank/zscore/confidence_range/margin_to_favの4特徴量を生成し、_train_submodel()に統合**

## Performance

- **Duration:** 18 min
- **Started:** 2026-05-14T13:18:12Z
- **Completed:** 2026-05-14T13:36:32Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- n_mining 82列のPIT監査完了: 全列がPRE (レース前予想データ) であることを文書化
- MiningFeatures実装: wide-to-long変換でDataKubun=3から4特徴量を生成
- _train_submodel() Group F block追加: interaction features後、market model前に実行
- Stage1/Win/Place全モデルのFEATURE_COLS更新、1435テスト全通過

## Task Commits

Each task was committed atomically:

1. **Task 1: PIT監査 + MiningFeatures + テスト** - `019fe5b` (feat) [TDD: RED->GREEN]
2. **Task 2: _train_submodel()統合 + FEATURE_COLS更新** - `3ed1ae7` (feat)

## Files Created/Modified
- `src/features/mining_features.py` - n_mining wide-to-long変換 + 4特徴量計算
- `docs/everydb2/26-mining-pit-audit.md` - 82列PRE/POST分類ドキュメント
- `tests/test_mining_features.py` - 11テスト (mock-based, DB不要)
- `src/pipelines/training_pipeline.py` - Group F mining features block追加
- `src/models/stage1_ability_model.py` - 3特徴量追加 (80->83)
- `src/models/two_stage_return_model.py` - Win 4 (54), HIT 3 (57), RETURN 4 (59) 追加
- `tests/test_two_stage_return_model.py` - feature_df fixtureに4列追加
- `tests/test_win_feature_analysis.py` - original_allリストに4特徴量追加

## Decisions Made
- DataKubun=3 (直前予想、馬体重発表後) を主ソース。情報量最大のため
- Place RETURN_FEATURE_COLSにWin全mining featuresを含めた。既存テストが「Win全列がPlace RETURNに含まれる」ことを検証しているため
- FEATURE_COLS参照にモジュールレベルimport (MiningFeatures.FEATURE_COLSではなくfrom import) を使用

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FEATURE_COLS参照方法の修正**
- **Found during:** Task 2 (テスト実行時)
- **Issue:** `MiningFeatures.FEATURE_COLS` でアクセスするとAttributeError (FEATURE_COLSはモジュール定数でクラス属性ではない)
- **Fix:** `from features.mining_features import FEATURE_COLS as MINING_FEATURE_COLS` に変更
- **Files modified:** src/pipelines/training_pipeline.py
- **Verification:** test_training_pipeline.py 3テスト通過
- **Committed in:** 3ed1ae7 (Task 2 commit)

**2. [Rule 1 - Bug] Place RETURN_FEATURE_COLSの不足**
- **Found during:** Task 2 (テスト実行時)
- **Issue:** Winに4特徴量を追加したがPlace RETURNに2つしか追加せず、test_place_return_feature_cols_include_place_specificが失敗
- **Fix:** Place RETURN_FEATURE_COLSにも4特徴量全てを追加
- **Files modified:** src/models/two_stage_return_model.py
- **Verification:** test_two_stage_return_model.py全テスト通過
- **Committed in:** 3ed1ae7 (Task 2 commit)

**3. [Rule 3 - Blocking] test_win_feature_analysis.pyのハードコードリスト更新**
- **Found during:** Task 2 (全テスト回帰確認時)
- **Issue:** original_allリストにmining featuresが含まれておらず失敗
- **Fix:** 4特徴量をoriginal_allリストに追加
- **Files modified:** tests/test_win_feature_analysis.py
- **Verification:** test_remaining_features_are_subset_of_original通過
- **Committed in:** 3ed1ae7 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** 全て既存テストの整合性維持のため必要な修正。スコープクリープなし。

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- MiningFeaturesモジュール完成、_train_submodel()統合済み
- Plan 02 (dam_pedigree_features) も同じ_train_submodel()統合パターンを使用可能
- POST_RACE漏洩テスト自動検証済み
- ETL実行 (`run_etl.py --tables n_mining`) はユーザーがローカルで実行必要

---
*Phase: 26-everydb2-new-features*
*Completed: 2026-05-14*

## Self-Check: PASSED

All 7 files verified. Both commits (019fe5b, 3ed1ae7) verified in git log.
