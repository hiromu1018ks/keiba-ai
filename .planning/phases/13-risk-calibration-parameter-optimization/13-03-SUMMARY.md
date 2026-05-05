---
phase: 13-risk-calibration-parameter-optimization
plan: 03
subsystem: tuning
tags: [optuna, tpe, walk-forward, strategy-optimization, parameter-search, manifest]

# Dependency graph
requires:
  - phase: 13-01
    provides: DDConfig dataclass, BacktestEngine strategy_params injection
  - phase: 13-02
    provides: RegimeDetector override_params, save_strategy_manifest/verify_strategy_manifest
provides:
  - StrategyOptimizer class (Optuna TPE ~16-dim parameter optimization)
  - _run_single_backtest concrete impl (ModelLoader + BacktestEngine + RegimeDetector override)
  - Lightweight WF 2fold evaluation loop with MedianPruner
  - CLI script run_strategy_optimization.py (end-to-end)
  - 13 mock-based tests (DB-free)
affects: [backtest validation pipeline, production deployment]

# Tech tracking
tech-stack:
  added: []
patterns: [optuna-tpe-16dim-search, walk-forward-lightweight-loop, dd-threshold-constraint-enforcement]

key-files:
  created:
    - src/tuning/strategy_optimizer.py
    - scripts/run_strategy_optimization.py
    - tests/test_strategy_optimizer.py
  modified: []

key-decisions:
  - "WalkForwardCV不使用: pipeline.run()が必須で変更リスクが高いため独自軽量WFループ"
  - "_run_single_backtest()は具象実装(NotImplementedErrorではなくModelLoader+BacktestEngine)"
  - "dd_threshold_2 <= dd_threshold_1の場合+0.01自動補正(DDConfig.__post_init__制約回避)"
  - "Optuna 4.x互換: optuna.trial.TrialStateを明示インポート(optuna.TrialStateは4.xで非公開)"

patterns-established:
  - "StrategyOptimizer: _suggest_params() -> _build_strategy_config() -> _run_single_backtest() pipeline"
  - "Lazy import pattern: _run_single_backtest内でfrom backtest.engine import BacktestEngine等(model_loader依存分離)"
  - "WF fold定義: _generate_folds()で[(start, end), ...]返却(将来的にCLI引数で上書き可能)"

requirements-completed: [VAL-02]

# Metrics
duration: 6min
completed: 2026-05-05
---

# Phase 13 Plan 03: StrategyOptimizer Summary

**Optuna TPEサンプラーで14次元戦略パラメータ(レジーム別fk/ev/edge + DD制御 + EVスケーリング + OddsBandFilter)を同時最適化するStrategyOptimizerを実装し、Walk-forward 2fold評価でルックアヘッドバイアスを防止、JSON manifest自動生成付きCLIスクリプトと13テストを追加**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-05T02:06:06Z
- **Completed:** 2026-05-05T02:12:06Z
- **Tasks:** 2
- **Files modified:** 3 (created)

## Accomplishments
- StrategyOptimizerクラス実装: Optuna TPE ~16次元探索空間 (レジーム別6 + DD制御5 + EVスケーリング2 + OddsBandFilter1 = 14パラメータ)
- _run_single_backtest()具象実装: ModelLoader.load_from_dir() -> RegimeDetector override -> BacktestEngine(strategy_params=...) -> engine.run()
- Walk-forward 2fold評価: 独自軽量ループ(pipeline.run回避) + MedianPruner中間報告 + ROI主 + ベット数制約
- CLIスクリプト run_strategy_optimization.py: --n-trials, --seed, --models-dir, --output, --min-bets
- 13テスト追加: TestSuggestParams(2), TestBuildStrategyConfig(3), TestObjective(3), TestOptimize(3), TestRunSingleBacktest(2)
- 回帰なし: Plan 01(31テスト) + Plan 02(14テスト) 全PASS

## Task Commits

Each task was committed atomically:

1. **Task 1: StrategyOptimizer実装 -- 具象_run_single_backtest + 軽量WFループ + Optuna TPE最適化** - `1725b8b` (feat)
2. **Task 2: CLIスクリプト + テスト** - `9601aa3` (feat)

## Files Created/Modified
- `src/tuning/strategy_optimizer.py` - StrategyOptimizerクラス (Optuna TPE最適化 + 具象_run_single_backtest + 軽量WFループ)
- `scripts/run_strategy_optimization.py` - CLIスクリプト (--n-trials, --seed, --models-dir, --output, --min-bets)
- `tests/test_strategy_optimizer.py` - 13テスト (mockベース、DB不要)

## Decisions Made
- WalkForwardCV不使用: pipeline.run()が必須で変更リスクが高いため独自軽量WFループでfold定義のみ_generate_folds()で管理
- _run_single_backtest()は具象実装(NotImplementedErrorではなくModelLoader+BacktestEngine直接呼び出し)
- dd_threshold_2 <= dd_threshold_1 の場合+0.01自動補正(DDConfig.__post_init__のValueError回避)
- Optuna 4.x互換対応: optuna.trial.TrialStateを明示的インポート

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Optuna TrialState import incompatibility**
- **Found during:** Task 2 (テスト実行時)
- **Issue:** `optuna.TrialState` が Optuna 4.8.0 でAttributeError。4.xでは `optuna.trial.TrialState` に移動
- **Fix:** `from optuna.trial import TrialState` を追加し `TrialState.PRUNED` に変更
- **Files modified:** src/tuning/strategy_optimizer.py
- **Verification:** 13テスト全PASS
- **Committed in:** 9601aa3 (Task 2 commit)

**2. [Rule 1 - Bug] dd_threshold_2 <= dd_threshold_1 causes DDConfig ValueError**
- **Found during:** Task 2 (テスト実行時: Optunaサンプラーが独立範囲提案で閾値逆転)
- **Issue:** _suggest_params()のdd_threshold_1(0.05-0.20)とdd_threshold_2(0.15-0.35)が独立範囲のため、サンプラーがdd_t2 < dd_t1となる組合せを生成 -> DDConfig.__post_init__でValueError
- **Fix:** _build_strategy_config()内でdd_t2 <= dd_t1の場合 dd_t2 = dd_t1 + 0.01 に自動補正
- **Files modified:** src/tuning/strategy_optimizer.py
- **Verification:** Trial 5のValueError解消、全テストPASS
- **Committed in:** 9601aa3 (Task 2 commit)

**3. [Rule 1 - Bug] Mock patch paths for lazy imports**
- **Found during:** Task 2 (テスト実行時)
- **Issue:** `patch("tuning.strategy_optimizer.ModelLoader")` がAttributeError。ModelLoaderは関数内lazy importのためモジュール属性として存在しない
- **Fix:** patch対象を実際のソースモジュールに変更: `db.model_loader.ModelLoader`, `backtest.engine.BacktestEngine`, `models.regime_detector.RegimeDetector`
- **Files modified:** tests/test_strategy_optimizer.py
- **Verification:** TestRunSingleBacktest 2テストPASS
- **Comitted in:** 9601aa3 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (3 bugs)
**Impact on plan:** 全てOptuna 4.x互換性とテスト品質の修正。スコープ変更なし。

## Issues Encountered
None - 計画通りに実装完了。3つのバグは全てテスト実行中に発見・修正。

## User Setup Required
None - no external service configuration required.

## Self-Check: PASSED

- [x] src/tuning/strategy_optimizer.py EXISTS
- [x] scripts/run_strategy_optimization.py EXISTS
- [x] tests/test_strategy_optimizer.py EXISTS
- [x] 13-03-SUMMARY.md EXISTS
- [x] Commit 1725b8b FOUND
- [x] Commit 9601aa3 FOUND

## Next Phase Readiness
- StrategyOptimizer + CLI完了 -> `python scripts/run_strategy_optimization.py --n-trials 100` でend-to-end最適化実行可能(PostgreSQL + 学習済みモデル必要)
- 14パラメータ最適化結果がJSON manifestとしてSHA256付き保存される
- Phase 13全3プラン完了: DD再設計(13-01) + パラメータ外部化(13-02) + Optuna最適化(13-03)

---
*Phase: 13-risk-calibration-parameter-optimization*
*Completed: 2026-05-05*
