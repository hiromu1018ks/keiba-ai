---
phase: 15-ev-filter-enhancement
plan: 01
subsystem: [ml-pipeline, backtest, models]
tags: [ev-lower, dynamic-threshold, percentile, conformal, lightgbm]

# Dependency graph
requires:
  - phase: 14-gate-recalibration
    provides: WinSelectionGate再学習 + drift diagnostics パターン
provides:
  - SubmodelSet.ev_lower_threshold_turf/dirt 動的閾値フィールド
  - TrainingPipelineV5._compute_ev_threshold() OOF分布からの閾値計算
  - get_win_candidates() のサーフェス別動的閾値フィルター
affects: [16-oddsband-recalibration, 17-optuna-optimization, 18-validation]

# Tech tracking
tech-stack:
  added: []
  patterns: [OOF percentile threshold, surface-specific EV filter]

key-files:
  created: []
  modified:
    - src/domain/models.py
    - src/pipelines/training_pipeline.py
    - src/backtest/race_predictor.py
    - tests/test_race_predictor.py

key-decisions:
  - "OOF positive-edge winnersの25th percentileをサーフェス別閾値として採用 (D-01/D-02)"
  - "NaNフォールバックをfillna(1.0)からサーフェス別デフォルト(芝0.8/ダート0.7)に変更 (D-03)"
  - "最小サンプル数30件で閾値計算を保護、不足時はフォールバック値を使用 (T-15-01)"

patterns-established:
  - "OOF percentile threshold: 学習時に分布から閾値を計算しSubmodelSetに格納、推論時に参照"
  - "Surface-specific EV filter: サーフェスごとに独立した閾値をgetattrで安全に取得"

requirements-completed: [EVF-01]

# Metrics
duration: 6min
completed: 2026-05-06
---

# Phase 15 Plan 01: EV Lower Dynamic Threshold Summary

**EV_lower固定閾値1.0をOOF positive-edge winnersの25th percentileベース動的閾値に置き換え、過剰除外3,594件を根本解決**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-06T05:44:27Z
- **Completed:** 2026-05-06T05:50:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- SubmodelSetにev_lower_threshold_turf/dirtフィールドを追加し、アンサンブルOOF分布から自動計算されるサーフェス別閾値を格納
- get_win_candidates()のEV_lowerフィルターを固定1.0から動的閾値に変更し、NaNフォールバックもサーフェス別に最適化
- 新規テスト3件追加(turf動的閾値/dirt動的閾値/NaNフォールバック)、全48テスト通過

## Task Commits

Each task was committed atomically:

1. **Task 1: Add dynamic threshold fields to SubmodelSet and compute in pipeline** - `7f1d420` (feat)
2. **Task 2: Replace fixed EV_lower filter with dynamic threshold in get_win_candidates()** - `0971365` (feat)

## Files Created/Modified
- `src/domain/models.py` - SubmodelSetにev_lower_threshold_turf/dirtフィールド(default=1.0)を追加
- `src/pipelines/training_pipeline.py` - _compute_ev_threshold()静的メソッド追加 + _train_submodel()で閾値計算→SubmodelSet格納
- `src/backtest/race_predictor.py` - get_win_candidates()のEV_lowerフィルターを動的閾値に変更
- `tests/test_race_predictor.py` - _make_submodel_mock()に新フィールド追加 + テスト3件追加

## Decisions Made
- OOF positive-edge winners (kakuteijyuni==1 AND win_selection_edge>0) の25th percentileを閾値計算に使用
- サーフェス別フォールバック値: 芝0.8、ダート0.7 (Phase 17 Optunaで最適化予定)
- 閾値計算の最小サンプル数30件 (少ない場合はフォールバック値を使用)
- wsg_train_df (EV_lower_win_corrected含む) を閾値計算の入力に使用

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] test_ev_lower_dynamic_threshold_turfがsurface列なしで失敗**
- **Found during:** Task 2 (テスト実行)
- **Issue:** _make_win_race_df()にsurface列が含まれておらず、動的閾値がデフォルト1.0で判定されテストが失敗
- **Fix:** テストデータに`surface=["turf"]*3`を追加
- **Files modified:** tests/test_race_predictor.py
- **Verification:** 全14 TestGetWinCandidatesテスト通過
- **Committed in:** 0971365 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** テストヘルパーの欠落対応のみ。スコープ変更なし。

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- EV_lower動的閾値の基盤が完成。Phase 17 Optunaで閾値を15-16次元目として最適化可能
- Phase 15 Plan 02 (EVF-02): EV診断モジュール(ev_diagnostics.py)の新規作成が次のステップ
- OddsBandFilter再キャリブレーション (Phase 16) の前提条件として動的閾値が利用可能

---
*Phase: 15-ev-filter-enhancement*
*Completed: 2026-05-06*
