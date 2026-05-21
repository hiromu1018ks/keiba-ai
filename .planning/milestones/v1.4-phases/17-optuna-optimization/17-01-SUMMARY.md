---
phase: 17-optuna-optimization
plan: 01
subsystem: tuning
tags: [optuna, tpe, walk-forward, 4fold, ev-lower, median-pruner]

# Dependency graph
requires:
  - phase: 15-ev-filter-enhancement
    provides: EV_lower dynamic threshold infrastructure
  - phase: 16-odds-band-rebuild
    provides: OddsBandFilter with training_bet_history
provides:
  - StrategyOptimizer 4fold対応 + 16次元最適化 + モデルロード最適化
  - _generate_training_bet_history() ヘルパーメソッド
  - _run_single_backtest_with_models() モデル共有版バックテスト
affects: [18-validation, run_strategy_optimization.py]

# Tech tracking
tech-stack:
  added: []
  patterns: [CR-01 regime reset per fold, trial-scoped model load, EV_lower submodel attribute injection]

key-files:
  created: []
  modified:
    - src/tuning/strategy_optimizer.py
    - tests/test_strategy_optimizer.py

key-decisions:
  - "n_folds デフォルト2→4変更 (D-02過学習防止)"
  - "fold_start_year コンストラクタ引数で動的fold生成"
  - "EV_lower閾値をOptuna探索空間に追加(15-16次元目)"
  - "モデルロード最適化: _objective()内1回ロード + 1回training_bet_historyキャッシュ"
  - "MedianPruner n_startup_trials=10, n_warmup_steps=0, interval_steps=1, n_min_trials=1"

patterns-established:
  - "Trial-scoped model loading: ModelLoader.load_from_dir() once per trial, shared across folds"
  - "CR-01 reset pattern: RegimeDetector 4 attributes reset at each fold start"
  - "EV_lower injection: SubmodelSet attributes set from Optuna params before fold evaluation"

requirements-completed: [OPT-01, OPT-02]

# Metrics
duration: 12min
completed: 2026-05-06
---

# Phase 17 Plan 01: StrategyOptimizer 4fold + EV_lower + Model Load Optimization Summary

**4fold動的生成対応StrategyOptimizer + EV_lower 2次元追加(16次元化) + trial内モデルロード最適化**

## Performance

- **Duration:** 12 min
- **Started:** 2026-05-06T13:07:33Z
- **Completed:** 2026-05-06T13:20:05Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- _suggest_params() を14次元から16次元に拡張 (ev_lower_threshold_turf/dirt追加)
- _generate_folds() をハードコード2foldからn_folds/fold_start_year動的生成に変更
- _objective() をモデルロード最適化(1回ロード + 1回training_bet_historyキャッシュ)にリファクタリング
- CR-01パターン: RegimeDetector 4属性を各fold開始時にリセット
- MedianPruner を4fold環境に最適化 (n_startup_trials=10)
- 全25テストPASS (8新規テスト追加、既存17テスト更新なし)

## Task Commits

Each task was committed atomically:

1. **Task 1 (TDD RED): failing tests** - `1843e50` (test)
2. **Task 1 (TDD GREEN): implementation** - `ba3f6df` (feat)

## Files Created/Modified
- `src/tuning/strategy_optimizer.py` - 4fold対応 + 16次元 + モデルロード最適化 + 新メソッド2つ
- `tests/test_strategy_optimizer.py` - 25テスト (TestGenerateFolds新規 + TestObjective 7テスト更新)

## Decisions Made
- n_foldsデフォルトを2→4に変更 (D-02: 4fold以上で過学習防止)
- fold_start_year=2022をコンストラクタ引数に追加 (2022-2025年の4年fold)
- EV_lower閾値の探索範囲を[0.5, 1.5]に設定 (Phase 15 OOF分布に基づく合理的範囲)
- _run_single_backtest()は後方互換のため残存 (CLIからの単一fold実行で使用可能性)
- TestOptimize既存テストを新_objective()構造に更新 (mock対象を_run_single_backtestから_run_single_backtest_with_modelsに変更)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] TestOptimizeテストを新_objective()構造に更新**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** _objective()が_run_single_backtest()を呼び出さなくなったため、TestOptimizeの3テストがModelLoader.load_from_dir()を実際に呼び出してFileNotFoundErrorで失敗
- **Fix:** TestOptimizeのmock対象を_run_single_backtestから_run_single_backtest_with_modelsに変更し、ModelLoaderと_generate_training_bet_historyのmockを追加
- **Files modified:** tests/test_strategy_optimizer.py
- **Verification:** 全25テストPASS
- **Committed in:** ba3f6df (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical - test compatibility)
**Impact on plan:** TestOptimizeテストの更新は新_objective()構造への追従であり、機能変更なし。

## Self-Check: PASSED

All claimed files verified present. All commit hashes verified in git log.

## Issues Encountered
- None

## Next Phase Readiness
- StrategyOptimizer 16次元4fold最適化が実行可能
- Plan 02 (multi-seed安定性検証) は本Planの成果物に依存
- run_strategy_optimization.pyのn_folds/fold_start_year引数対応が推奨

---
*Phase: 17-optuna-optimization*
*Completed: 2026-05-06*
