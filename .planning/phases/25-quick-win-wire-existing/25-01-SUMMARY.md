---
phase: 25-quick-win-wire-existing
plan: 01
subsystem: ml-features
tags: [lightgbm, feature-engineering, jockey, trainer, combo, parquet, paper-trading]

# Dependency graph
requires:
  - phase: 22-early-money
    provides: JockeyContextFeatures, TrainerContextFeatures (計算ロジック)
  - phase: 24-feature-audit-pruning
    provides: 特徴量監査基盤、FEATURE_COLS管理パターン
provides:
  - WinTwoStageModel.FEATURE_COLS 50特徴量 (38+12)
  - PlaceTwoStageModel.HIT_FEATURE_COLS 54特徴量 (45+9)
  - PlaceTwoStageModel.RETURN_FEATURE_COLS 55特徴量 (43+12)
  - paper_trading/predictor.py での JockeyTrainerComboFeatures 計算・マージ
affects: [backtest, paper-trading, feature-importance-analysis]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FEATURE_COLS group comments: 騎手コンテキスト(Group C)/調教師コンテキスト(Group D)/コンビ(Stage2)のグループコメント付き追加"

key-files:
  created: []
  modified:
    - src/models/two_stage_return_model.py
    - src/paper_trading/predictor.py
    - tests/test_two_stage_return_model.py
    - tests/test_win_feature_analysis.py

key-decisions:
  - "12特徴量すべてをWin/Place FEATURE_COLSに追加 (LightGBMが不要特徴量をgain=0にするため安全)"
  - "Place HIT_FEATURE_COLSに既存3(jockey_wr_overall, trainer_wr_overall, jt_combo_place_rate)を残し残り9を追加"
  - "AbilityModel.FEATURE_COLSは変更なし (Stage2 only設計方針を維持)"

patterns-established:
  - "FEATURE_COLSへのグループコメント付き特徴量追加パターン"

requirements-completed: [WIRE-01, WIRE-02, WIRE-03]

# Metrics
duration: 9min
completed: 2026-05-12
---

# Phase 25 Plan 01: Quick Win Wire Existing Summary

**騎手(4)/調教師(4)/コンビ(4)合計12特徴量をWin/PlaceモデルFEATURE_COLSに配線しpaper_tradingパスにJockeyTrainerComboFeatures計算を追加**

## Performance

- **Duration:** 9 min
- **Started:** 2026-05-12T13:43:25Z
- **Completed:** 2026-05-12T13:52:47Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- WinTwoStageModel.FEATURE_COLS: 38 -> 50特徴量 (騎手4+調教師4+コンビ4の12特徴量を追加)
- PlaceTwoStageModel.HIT_FEATURE_COLS: 45 -> 54特徴量 (既存3+残り9を追加)
- PlaceTwoStageModel.RETURN_FEATURE_COLS: 43 -> 55特徴量 (12全てを追加)
- paper_trading/predictor.py: JockeyTrainerComboFeatures import/compute/merge を追加
- テスト38テスト全通過、POST_RACE漏洩テスト4テスト全通過

## Task Commits

Each task was committed atomically:

1. **Task 1: WinTwoStageModel + PlaceTwoStageModel FEATURE_COLSに12特徴量を追加** - `f14ba15` (feat)
2. **Task 2: paper_trading/predictor.pyにJockeyTrainerComboFeatures計算を追加 + テスト更新** - `657815e` (feat)
3. **(Rule 3 Auto-fix) test_win_feature_analysisのoriginal_allリストに12特徴量を追加** - `e4e40f4` (fix)

## Files Created/Modified

- `src/models/two_stage_return_model.py` - Win/Place両モデルのFEATURE_COLSに12特徴量をグループコメント付きで追加
- `src/paper_trading/predictor.py` - JockeyTrainerComboFeatures import/compute/merge をsetup()に追加
- `tests/test_two_stage_return_model.py` - feature_dfフィクスチャに9新カラム追加、TestJockeyTrainerComboInFeatureCols新設
- `tests/test_win_feature_analysis.py` - original_allリストにPhase 25の12特徴量を追加

## Decisions Made

- 12特徴量すべてを追加: LightGBMが不要特徴量を自動的にgain=0にするため、全追加は安全なアプローチ (D-01準拠)
- Place HITは既存3特徴量の直後に残り9を追加: コードの差分最小化と論理的なグループ化
- AbilityModelは変更なし: Stage2 onlyの設計方針を維持 (D-02準拠)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] test_win_feature_analysisのoriginal_allリスト更新**
- **Found during:** Task 2 (全体テスト検証時)
- **Issue:** `test_remaining_features_are_subset_of_original` がハードコードされたoriginal_allリストを持っており、12特徴量追加後に不一致で失敗
- **Fix:** original_allリストにPhase 25の12特徴量を追加
- **Files modified:** tests/test_win_feature_analysis.py
- **Verification:** テスト通過確認
- **Committed in:** e4e40f4

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** 最小限。テストのハードコードリストをFEATURE_COLSの実態に合わせただけ。スコープクリープなし。

## Issues Encountered

None - 全て予定通りに完了。

## User Setup Required

None - 外部サービス設定不要。

## Next Phase Readiness

- 12特徴量の配線完了。モデル学習時に自動的に利用可能
- 次フェーズ(Plan 02)のバックテストでROI改善を確認可能
- training_pipeline.py と backtest/engine.py は既に3モジュール計算済みのため、FEATURE_COLS追加のみでモデルが特徴量を利用開始

---
*Phase: 25-quick-win-wire-existing*
*Completed: 2026-05-12*

## Self-Check: PASSED

- [x] src/models/two_stage_return_model.py FOUND
- [x] src/paper_trading/predictor.py FOUND
- [x] tests/test_two_stage_return_model.py FOUND
- [x] tests/test_win_feature_analysis.py FOUND
- [x] 25-01-SUMMARY.md FOUND
- [x] Commit f14ba15 FOUND
- [x] Commit 657815e FOUND
- [x] Commit e4e40f4 FOUND
