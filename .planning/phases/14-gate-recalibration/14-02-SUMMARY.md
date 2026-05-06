---
phase: 14-gate-recalibration
plan: 02
subsystem: testing
tags: [ensemble, model-loader, win-selection-gate, mock, integration-test]

# Dependency graph
requires:
  - phase: 14-gate-recalibration
    provides: "CONTEXT.md/RESEARCH.md の診断判断 D-05~D-08"
provides:
  - "use_ensemble フラグ伝播の統合テスト3件 (D-05, D-06, D-07)"
  - "ModelLoader.load_from_dir の use_ensemble 伝播バグ修正"
affects: [14-gate-recalibration, model-loader, ensemble-loading]

# Tech tracking
tech-stack:
  added: []
  patterns: ["mock-based flag propagation test", "sys.modules patch for local imports"]

key-files:
  created:
    - tests/test_ensemble_gate_propagation.py
  modified:
    - src/db/model_loader.py

key-decisions:
  - "load_from_dir() の SubmodelSet 構築に use_ensemble パラメータが欠落していたバグを修正"

patterns-established:
  - "フラグ伝播テスト: sys.modules パッチで関数内インポートをモック化するパターン"

requirements-completed: [GATE-03]

# Metrics
duration: 16min
completed: 2026-05-06
---

# Phase 14 Plan 02: use_ensemble Gate Propagation Test Summary

**use_ensemble フラグが TrainingPipeline と ModelLoader の2解決ポイントで正しく伝播されることを mock ベース統合テスト3件で検証、ModelLoader.load_from_dir の use_ensemble 未伝播バグを修正**

## Performance

- **Duration:** 16 min
- **Started:** 2026-05-06T00:05:29Z
- **Completed:** 2026-05-06T00:21:25Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- TestEnsembleFlagPropagation クラスに3テストメソッドを作成し、全1260テスト回帰なし
- ModelLoader.load_from_dir() で SubmodelSet 構築時に use_ensemble が未伝播だったバグを発見・修正
- D-05/D-06/D-07 の診断判断に基づくテストカバレッジを確保

## Task Commits

Each task was committed atomically:

1. **Task 1: Create use_ensemble propagation integration tests** - `0f13578` (feat)

## Files Created/Modified
- `tests/test_ensemble_gate_propagation.py` - TestEnsembleFlagPropagation クラス (3テストメソッド): StackedEnsemble hit_model 代入検証, ModelLoader _load_hit_model .joblib 優先ロード検証, SubmodelSet 学習済み gate 包含検証
- `src/db/model_loader.py` - load_from_dir() の SubmodelSet() 呼び出しに use_ensemble=use_ensemble を追加 (line 695)

## Decisions Made
- テストは use_ensemble=True パスのみテスト (D-07)
- Test 1 は sys.modules パッチで StackedEnsemble の関数内インポートをモック化
- Test 2 は _load_hit_model の直接テスト + load_from_dir の統合テストの2段構え
- Test 3 は実際の WinSelectionGateModel.train() を使用して gate が正しく学習されることを検証

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ModelLoader.load_from_dir() の SubmodelSet に use_ensemble が未伝播**
- **Found during:** Task 1 (test_model_loader_ensemble_override_loads_joblib テスト失敗)
- **Issue:** load_from_dir() で use_ensemble 変数を計算し _load_hit_model に渡していたが、SubmodelSet() 構築時には渡していなかった。結果としてロード後の TrainedModelsV5 で常に use_ensemble=False になっていた
- **Fix:** SubmodelSet() 呼び出しに use_ensemble=use_ensemble パラメータを追加
- **Files modified:** src/db/model_loader.py (line 695)
- **Verification:** テスト3件全て PASSED、回帰テスト1260件全て PASSED
- **Committed in:** 0f13578 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** バグ修正は計画外だが、テストが意図した検証を行うために必須。モデルロード後のフラグ一貫性に影響する重要な修正。

## Issues Encountered
- なし

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- use_ensemble フラグ伝播の正しさがテストで担保された
- ModelLoader のバグ修正により、load_from_dir(use_ensemble_override=True) でロードした SubmodelSet が正しく use_ensemble=True を持つようになった
- 次の Plan 03 以降でアンサンブルバックテストの EV_lower/OddsBand 再構築に進める準備完了

## Self-Check: PASSED

- tests/test_ensemble_gate_propagation.py: FOUND
- src/db/model_loader.py: FOUND
- 14-02-SUMMARY.md: FOUND
- Commit 0f13578: FOUND
- Test methods: 3 (expected 3)
- Test class: 1 (expected 1)
- Full regression: 1260 passed, 0 failed

---
*Phase: 14-gate-recalibration*
*Completed: 2026-05-06*
