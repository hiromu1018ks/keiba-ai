---
phase: 05-foundation-features
plan: 01
subsystem: features
tags: [lightgbm, numpy, pandas, time-series, ema, pace-features, feature-engineering]

# Dependency graph
requires:
  - phase: 01-feature-analysis-enhancement
    provides: horse_history_features.py, pace_aptitude_features.py, interaction_features.py
provides:
  - EMA-weighted harontimel5_avg (TSER-01)
  - class_adj_formetric feature (TSER-02)
  - haron_zscore_trend feature (TSER-03)
  - pace_corner_stability, pace_closing_power, pace_position_consistency (PACE-01)
  - actual_pace_fit feature (PACE-02)
affects: [06-odds-deviation, 07-ensemble]

# Tech tracking
tech-stack:
  added: []
  patterns: [EMA-exponential-decay-weighting, class-adjusted-formetric, z-score-trend-polyfit, pace-sub-feature-decomposition]

key-files:
  created: []
  modified:
    - src/features/horse_history_features.py
    - src/features/pace_aptitude_features.py
    - src/features/interaction_features.py
    - src/models/stage1_ability_model.py
    - src/models/two_stage_return_model.py
    - src/pipelines/training_pipeline.py
    - tests/test_horse_history_features.py
    - tests/test_pace_aptitude_features.py
    - tests/test_interaction_features.py

key-decisions:
  - "EMA halflife=3 (decay=ln(2)/3) adopted for harontimel5_avg — financial time-series standard"
  - "class_adj_formetric uses sum(norm_finish * class_level) / sum(class_level) — high-class wins weighted higher"
  - "haron_zscore_trend requires minimum 3 valid z-scores — avoids noisy regression on sparse data"
  - "PACE-01 outputs 3 separate sub-features rather than single composite — LightGBM learns nonlinear combinations"
  - "actual_pace_fit uses empirical front_pace_wr/closing_pace_wr based on declared running style"

patterns-established:
  - "EMA manual weighting: weights = (1-decay)**np.arange(n), reversed for newest-first, then normalized"
  - "Sub-feature decomposition: multiple correlated features from same data source, model decides importance"
  - "Running-style conditional: np.where(is_front_runner, front_pace_wr, np.where(is_closer, closing_pace_wr, np.nan))"

requirements-completed: [TSER-01, TSER-02, TSER-03, PACE-01, PACE-02]

# Metrics
duration: 9min
completed: 2026-05-03
---

# Phase 5 Plan 01: Time-Series and Pace Features Summary

**EMA重み付けハロンタイム・クラス調整フォーメトリック・z-score改善トラジェクトリ・ペースフィグア3サブ特徴量・実績ペース適性を追加 (7新特徴量)**

## Performance

- **Duration:** 9 min
- **Started:** 2026-05-03T07:25:20Z
- **Completed:** 2026-05-03T07:34:25Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- harontimel5_avg を単純平均から EMA 重み付け (halflife=3, 全過去走) に置き換え、直近成績を強調
- class_adj_formetric 新特徴量: 高クラス好走を高く評価する重み付き着順指標
- haron_zscore_trend 新特徴量: 過去走 z-score の線形回帰傾きで改善/悪化傾向を定量化
- ペースフィグア3サブ特徴量 (pace_corner_stability, pace_closing_power, pace_position_consistency) を PaceAptitudeFeatures に追加
- actual_pace_fit を脚質ベースで front_pace_wr/closing_pace_wr を選択して生成
- AbilityModel と WinTwoStageModel の FEATURE_COLS に全新特徴量を追加
- training_pipeline.py の pace_df マージ列を更新
- 94テスト全通過 (65 horse_history + 17 pace_aptitude + 12 interaction)

## Task Commits

Each task was committed atomically (TDD: test -> feat):

1. **Task 1: TSER-01~03** - `2a4da36` (test), `46bd59b` (feat)
2. **Task 2: PACE-01~02** - `e7c1fc1` (test), `74b0fc5` (feat)

## Files Created/Modified

- `src/features/horse_history_features.py` - EMA harontimel5_avg, class_adj_formetric, haron_zscore_trend 追加
- `src/features/pace_aptitude_features.py` - 3ペースフィグアサブ特徴量追加 (compute_batch拡張)
- `src/features/interaction_features.py` - actual_pace_fit 追加 (脚質ベース条件分岐)
- `src/models/stage1_ability_model.py` - FEATURE_COLS に7新特徴量追加
- `src/models/two_stage_return_model.py` - FEATURE_COLS に3新特徴量追加
- `src/pipelines/training_pipeline.py` - pace_df マージ列更新 (6列→9列)
- `tests/test_horse_history_features.py` - TSER-01~03 テスト追加 (EMA, class_adj, zscore_trend)
- `tests/test_pace_aptitude_features.py` - PACE-01 テスト追加 (corner_stability, position_consistency)
- `tests/test_interaction_features.py` - PACE-02 テスト追加 (actual_pace_fit 4テスト)

## Decisions Made

- EMA halflife=3 採用: 金融時系列解析の標準値。3走前の重みは直近の50%、5走前は25%
- class_adj_formetric の計算式: Σ(norm_finish * class_level) / Σ(class_level)。高クラス好走の価値を反映
- haron_zscore_trend の最小3走要件: 少ないデータ点での回帰はノイズが大きいため
- PACE-01 を3サブ特徴量に分割: LightGBMが非線形組み合わせを自動学習。単一スコアより情報量が多い
- actual_pace_fit は実績ベース (front_pace_wr/closing_pace_wr) を使用: 宣言脚質と実際の走法乖離を補完

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed pace_aptitude_features `start` variable reference**
- **Found during:** Task 2 (PACE-01 implementation)
- **Issue:** Plan referenced `base + start : base + c` pattern from HorseHistoryFeatures, but PaceAptitudeFeatures compute_batch uses different loop structure without `start` variable
- **Fix:** Changed to `base : base + c` since in PaceAptitudeFeatures, `c` (cutoff) directly represents past race count and all past races start from base offset
- **Files modified:** src/features/pace_aptitude_features.py
- **Verification:** All 29 PACE tests pass (17 pace_aptitude + 12 interaction)
- **Committed in:** 74b0fc5 (Task 2 feat commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor fix for code pattern mismatch between plan and actual codebase structure. No scope creep.

## Issues Encountered

None - all implementations followed established codebase patterns.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 7 new features ready for Phase 6 (Odds Deviation) and Phase 7 (Ensemble) to leverage
- All features follow existing NaN-safe patterns (LightGBM native NaN handling)
- AbilityModel.FEATURE_COLS now has 105+ features, WinTwoStageModel has 35+ features
- training_pipeline.py pace merge updated to carry all 9 pace-related columns

## Self-Check: PASSED

All 10 files verified as existing. All 4 task commits verified in git log.

---
*Phase: 05-foundation-features*
*Completed: 2026-05-03*
