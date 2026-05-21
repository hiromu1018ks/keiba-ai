---
phase: 20-high-odds-pattern-features
plan: 02
subsystem: features
tags: [high-odds, env-adaptability, class-trajectory, form-improvement, integration, tdd]

# Dependency graph
requires:
  - phase: 20-01
    provides: "compute_class_trajectory, compute_form_improvement_rate, high_odds_features.py"
provides:
  - "compute_env_adaptability: 3変化(距離/サーフェス/馬場) x 3サブ特徴量(平均着順/勝率/経験回数)"
  - "18新特徴量のHorseHistoryFeatures.compute()統合"
  - "BASE_COLS 48特徴量への拡張"
affects: [20-03]

# Tech tracking
tech-stack:
  added: []
  patterns: [env-change-detection, match-mask-filtering, _compute_change_stats helper]

key-files:
  created: []
  modified:
    - src/features/high_odds_features.py
    - src/features/horse_history_features.py
    - tests/test_high_odds_features.py
    - tests/test_horse_history_features.py

key-decisions:
  - "テスト1の距離変更データで最後の過去走=異条件に修正(変更検出ロジックの正確性)"
  - "環境変化適性でhistory_mask[hist_start:hist_idx]を使用(valid_maskでなく)してtarget_date以前を厳密に参照"

patterns-established:
  - "env-change-detection: current != last_past → change detected → filter matching past → compute stats"
  - "_compute_change_stats: 汎用ヘルパーで3変化を統一的に処理"

requirements-completed: [HODDS-04, HODDS-05]

# Metrics
duration: 7min
completed: "2026-05-09"
---

# Phase 20 Plan 02: 環境変化適性特徴量 + パイプライン統合 Summary

環境変化適性9特徴量(compute_env_adaptability)をTDDで実装し、Plan 20-01の9特徴量と合わせて全18新特徴量をHorseHistoryFeatures.compute()のper-horseループに統合。BASE_COLS 48特徴量に拡張完了。

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-09T00:22:59Z
- **Completed:** 2026-05-09T00:30:12Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- compute_env_adaptability実装: 3変化(距離/サーフェス/馬場状態) x 3サブ特徴量(平均着順/勝率/経験回数) = 9特徴量
- HorseHistoryFeatures.compute()に18新特徴量(クラストラジェクトリ7+フォーム改善率2+環境変化適性9)を統合
- BASE_COLS 30→48特徴量に拡張、results.append()辞書と整合性確認
- 92テスト全通過(回帰なし)

## Task Commits

1. **Task 1 (RED): test(20-02): add failing tests for compute_env_adaptability** - `5ae9cdf` (test)
2. **Task 1 (GREEN): feat(20-02): 環境変化適性9特徴量を実装** - `e3bf64a` (feat)
3. **Task 2: feat(20-02): HorseHistoryFeaturesに18新特徴量を統合 (HODDS-05)** - `3261150` (feat)

## Files Created/Modified
- `src/features/high_odds_features.py` - compute_env_adaptability, _compute_change_stats追加、FEATURE_COLS 9→18
- `src/features/horse_history_features.py` - import, BASE_COLS, per-horse計算, results.append()更新
- `tests/test_high_odds_features.py` - TestComputeEnvAdaptability 8テストケース、FEATURE_COLS更新
- `tests/test_horse_history_features.py` - BASE_COLS count 30→48

## Decisions Made
- 環境変化適性の計算にhistory_mask(histor_start:hist_idx)を使用し、valid_mask(start:idx)と使い分け。クラストラジェクトリ・フォーム改善率は直近走(valid_mask)、環境変化適性は全過去走(history_mask)から条件マッチング
- _compute_change_statsヘルパーで距離/サーフェス/馬場の3変化を統一的に処理

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] テスト1の距離変更検出データが不正**
- **Found during:** Task 1 GREEN phase
- **Issue:** dist_bins=["sprint","sprint","mile"] + cur_db="mile" では最後の過去走=mile=現在と同じため距離変更が検出されずNaNになる
- **Fix:** dist_bins=["mile","sprint","sprint"] に変更(最後の過去走=sprint、現在=mileで変更検出)
- **Files modified:** tests/test_high_odds_features.py
- **Verification:** test_dist_change_with_experience PASSED
- **Committed in:** e3bf64a (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** テストデータの修正のみ。実装ロジックへの影響なし。

## TDD Gate Compliance

- RED commit: 5ae9cdf (ImportError confirmed - compute_env_adaptability does not exist)
- GREEN commit: e3bf64a (27/27 tests passed including 10 new env adaptability tests)
- No REFACTOR needed (clean implementation)

## Self-Check: PASSED

- src/features/high_odds_features.py: FOUND
- src/features/horse_history_features.py: FOUND
- tests/test_high_odds_features.py: FOUND
- tests/test_horse_history_features.py: FOUND
- 20-02-SUMMARY.md: FOUND
- Commit 5ae9cdf: FOUND
- Commit e3bf64a: FOUND
- Commit 3261150: FOUND

## Next Phase Readiness
- 18新特徴量がパイプラインに統合済み
- Plan 20-03のFeature importance分析が可能
- ルックアヘッドバイアス防止: valid_mask + history_mask + searchsortedで検証済み

---
*Phase: 20-high-odds-pattern-features*
*Completed: 2026-05-09*
