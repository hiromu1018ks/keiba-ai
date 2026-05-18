---
phase: 32-market-cross-consistency-features
plan: 02
subsystem: features
tags: [feature-engine, FEATURE_COLS, market-cross-consistency, build-all, build-features, manifest]

# Dependency graph
requires:
  - phase: 32-01
    provides: "compute_market_cross_features() + DataRepository.load_wide_odds()"
provides:
  - "build_all() と build_features() のMCF統合"
  - "全12モデルFEATURE_COLSに5 MCF特徴量追加"
  - "POST_RACE安全性テスト (TestMarketCrossFeatures)"
  - "feature_freeze_manifest.json 再生成"
affects: [feature_engine, models, manifest]

# Tech tracking
tech-stack:
  added: []
  patterns: [feature-engine-submodule-integration, feature-cols-evolution]

key-files:
  created: []
  modified:
    - src/features/feature_engine.py
    - src/models/stage1_ability_model.py
    - src/models/market_model.py
    - src/models/regime_detector.py
    - src/models/place_ability_model.py
    - src/models/race_quality_screener.py
    - src/models/wide_two_stage_model.py
    - src/models/two_stage_return_model.py
    - src/models/ev_correction_model.py
    - src/models/conformal_ev_model.py
    - tests/test_post_race_leakage.py
    - data/feature_freeze_manifest.json

key-decisions:
  - "build_all()でDataRepository(store)経由でwide/trioオッズをロードし、store=None時はNaNフォールバック"
  - "build_features()はwide_df/trio_dfなしでcompute_market_cross_features()を呼び出しNaNフォールバック"
  - "EVCorrectionModelテストデータにMCF列を追加してFEATURE_COLS拡張に対応"

patterns-established:
  - "TimingContext付きsubmodule統合パターン: build_allに新しい特徴量モジュールを追加する際の標準パターン"

requirements-completed: [MCF-07]

# Metrics
duration: 6min
completed: 2026-05-18
---

# Phase 32 Plan 02: Feature Engine Integration + FEATURE_COLS Update Summary

**feature_engineにMCF統合、全12モデルFEATURE_COLSに5特徴量追加、manifest再生成**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-18T10:35:29Z
- **Completed:** 2026-05-18T10:41:16Z
- **Tasks:** 2 (TDD: Task 1 RED + GREEN, Task 2 manifest)
- **Files modified:** 12

## Accomplishments
- build_all()にDataRepository経由のwide/trioオッズロード + compute_market_cross_features()統合を実装
- build_features()にcompute_market_cross_features()のNaNフォールバック呼び出しを追加
- 全12モデルのFEATURE_COLSに5つのMCF特徴量を追加 (AbilityModel 97->102、他11モデルも+5)
- TestMarketCrossFeatures 4テストを追加 (AST scan, overlap check, build_all output, FEATURE_COLS)
- feature_freeze_manifest.jsonを再生成し全12モデルのSHA256ハッシュを更新

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): failing tests for MCF integration** - `5f8fb69` (test)
2. **Task 1 (GREEN): integrate MCF into feature_engine + 12 models** - `db09202` (feat)
3. **Task 2: regenerate feature freeze manifest** - `ec8b395` (chore)

## TDD Gate Compliance

- RED gate: `5f8fb69` - test(32-02): add failing tests for MCF integration into feature_engine and models (2 tests failing)
- GREEN gate: `db09202` - feat(32-02): integrate MCF features into feature_engine and all 12 models (all 35 tests passing)
- REFACTOR gate: Not needed - implementation was clean after GREEN

## Files Created/Modified
- `src/features/feature_engine.py` - build_all()とbuild_features()にMCF統合 (22行追加)
- `src/models/stage1_ability_model.py` - FEATURE_COLSに5 MCF追加 (102列)
- `src/models/market_model.py` - FEATURE_COLSに5 MCF追加 (14列)
- `src/models/regime_detector.py` - FEATURE_COLSに5 MCF追加 (15列)
- `src/models/place_ability_model.py` - FEATURE_COLSに5 MCF追加 (68列)
- `src/models/race_quality_screener.py` - FEATURE_COLSに5 MCF追加 (29列)
- `src/models/wide_two_stage_model.py` - SHARED_FEATURE_COLSに5 MCF追加 (12列)
- `src/models/two_stage_return_model.py` - WinTwoStageModel/PlaceTwoStageModel各FEATURE_COLSに5 MCF追加
- `src/models/ev_correction_model.py` - EVCorrectionModel/PlaceEVCorrectionModel各FEATURE_COLSに5 MCF追加
- `src/models/conformal_ev_model.py` - FEATURE_COLSに5 MCF追加 (136列)
- `tests/test_post_race_leakage.py` - TestMarketCrossFeatures 4テスト + EVCorrectionModelテスト修正
- `data/feature_freeze_manifest.json` - 全12モデルSHA256更新

## Decisions Made
- build_all()でDataRepository(store)経由でwide/trioオッズをロード -- storeパラメータを再利用し、None時はNaNフォールバック
- build_features()はwide_df/trio_dfなしでcompute_market_cross_features()を呼び出す -- 推論パスではwide/trioデータが利用不可のためNaNフォールバック
- EVCorrectionModelテストのDataFrameにMCF列を追加 -- FEATURE_COLS拡張によるKeyErrorを防止

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed missing WinTwoStageModel/PlaceTwoStageModel imports in test**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** test_all_models_have_mcf_featuresでWinTwoStageModelとPlaceTwoStageModelがインポートされておらずNameError
- **Fix:** from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModelを追加
- **Files modified:** tests/test_post_race_leakage.py
- **Verification:** テスト通過
- **Committed in:** db09202 (Task 1 GREEN commit)

**2. [Rule 1 - Bug] Fixed EVCorrectionModel test missing MCF columns**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** EVCorrectionModel.FEATURE_COLSに5つのMCF列を追加したため、テストのDataFrameにもこれらの列が必要。不足していると_prepare_features()でKeyError
- **Fix:** テストデータにrl_favorite_in_wide_top1等5列を追加
- **Files modified:** tests/test_post_race_leakage.py
- **Verification:** テスト通過
- **Committed in:** db09202 (Task 1 GREEN commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs)
**Impact on plan:** テスト修正のみ。スコープクリープなし。

## Issues Encountered
- なし

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- build_all()とbuild_features()の両方でMCF特徴量が自動計算される
- 全12モデルがMCF特徴量を利用可能 (FEATURE_COLSに含まれる)
- 次フェーズ(Phase 33, Gain per Depth診断)への準備完了

## Self-Check: PASSED

- [x] src/features/feature_engine.py: FOUND
- [x] src/models/stage1_ability_model.py: FOUND
- [x] src/models/market_model.py: FOUND
- [x] src/models/regime_detector.py: FOUND
- [x] src/models/place_ability_model.py: FOUND
- [x] src/models/race_quality_screener.py: FOUND
- [x] src/models/wide_two_stage_model.py: FOUND
- [x] src/models/two_stage_return_model.py: FOUND
- [x] src/models/ev_correction_model.py: FOUND
- [x] src/models/conformal_ev_model.py: FOUND
- [x] tests/test_post_race_leakage.py: FOUND
- [x] Commit 5f8fb69 (test): FOUND
- [x] Commit db09202 (feat): FOUND
- [x] Commit ec8b395 (chore): FOUND

---
*Phase: 32-market-cross-consistency-features*
*Completed: 2026-05-18*
