---
phase: 17-optuna-optimization
plan: 02
subsystem: tuning
tags: [optuna, tpe, multi-seed, cv, stability-report, reoptimization]

# Dependency graph
requires:
  - phase: 17-optuna-optimization/01
    provides: StrategyOptimizer 4fold + 16dim + model load optimization
provides:
  - optimize_multi_seed() 3seed安定性検証エントリポイント
  - _compute_stability_report() CV(変動係数)による安定性定量化
  - _optimize_with_fixed_dims() 不安定次元固定再最適化
  - 安定性レポートJSON出力 (stability_report.json)
  - CLI --seeds拡張
affects: [18-validation]

# Tech tracking
tech-stack:
  added: []
  patterns: [multi-seed CV stability analysis, dimension-fixed reoptimization, trial allocation asymmetry]

key-files:
  created: []
  modified:
    - src/tuning/strategy_optimizer.py
    - scripts/run_strategy_optimization.py
    - tests/test_strategy_optimizer.py

key-decisions:
  - "CV閾値0.20(20%)で不安定判定 (D-09, Phase 17 RESEARCH A3準拠)"
  - "不安定次元のデフォルト値ハードコード (T-17-07 mitigate)"
  - "主seed 100 trials + 追加seed 50 trials (D-08非対称割り当て)"
  - "_optimize_with_fixed_dims()は1回のみ再実行 (Pitfall 5回避)"
  - "optimize_multi_seed()内で最良パラメータのmanifestも自動保存"

patterns-established:
  - "Multi-seed stability pattern: 3 seeds with asymmetric trial allocation, CV analysis, dimension fixing"
  - "Dimension-fixed reoptimization: _suggest_params temporary monkey-patch for fixed dimensions"

requirements-completed: [OPT-03]

# Metrics
duration: 4min
completed: 2026-05-06
---

# Phase 17 Plan 02: Multi-Seed Stability Verification Summary

**Multi-seed(42/43/44) Optuna安定性検証 + CV(変動係数)による不安定次元自動検出 + デフォルト値固定再最適化 + CLI --seeds拡張**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-06T13:26:16Z
- **Completed:** 2026-05-06T13:30:18Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 3

## Accomplishments
- optimize_multi_seed()が3 seed(42/43/44)で非対称試行割り当て(主100+追加50)で最適化を実行
- _compute_stability_report()が各次元のCV(変動係数)を計算し、CV>0.20の次元を不安定と判定
- _optimize_with_fixed_dims()が不安定次元をデフォルト値に固定し、探索空間を縮小して再最適化(1回のみ)
- 安定性レポートJSON(stability_report.json) + 最良パラメータmanifest自動保存
- CLI --seeds引数でmulti-seed実行をサポート
- 全31テストPASS (7新規TestMultiSeedStability追加、既存24テスト更新なし)

## Task Commits

Each task was committed atomically:

1. **Task 1 (TDD RED): failing tests** - `cb99de4` (test)
2. **Task 1 (TDD GREEN): implementation** - `4a8d310` (feat)

## Files Created/Modified
- `src/tuning/strategy_optimizer.py` - 3新規メソッド(optimize_multi_seed, _compute_stability_report, _optimize_with_fixed_dims)
- `scripts/run_strategy_optimization.py` - --seeds引数追加 + main()分岐ロジック
- `tests/test_strategy_optimizer.py` - TestMultiSeedStability 7テスト追加

## Decisions Made
- CV閾値を0.20(20%)に設定 (D-09 Claude's discretion, Phase 17 RESEARCH A3準拠)
- 不安定次元のデフォルト値をハードコード (T-17-07 mitigate: RegimeDetector._get_base_params()と一致)
- 主seed 100 trials + 追加seed 50 trials (D-08非対称割り当てで計算コスト抑制)
- _optimize_with_fixed_dims()は_suggest_paramsの一時差し替えで実装 (monkey-patch, finallyで復元)
- 安定性レポートにversion/timestamp/seeds/dimensions/best_roi_by_seed/mean_best_roi/reoptimizationスキーマ

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 未使用変数reportを削除**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** test_saves_stability_report_jsonでreport変数がassign-onlyで未使用 (ruff F841)
- **Fix:** 代文を文に変更(report変数を除去)
- **Files modified:** tests/test_strategy_optimizer.py
- **Verification:** ruff check src/ scripts/ PASS、全31テストPASS
- **Committed in:** 4a8d310 (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug - unused variable)
**Impact on plan:** 微小修正。機能変更なし。

## Issues Encountered
- None

## Next Phase Readiness
- multi-seed安定性検証が実行可能 (CLI: --seeds 42,43,44)
- Phase 18 (validation)は安定性レポートJSONを入力として使用
- 安定したパラメータセットの実データ検証が次ステップ

---
*Phase: 17-optuna-optimization*
*Completed: 2026-05-06*

## Self-Check: PASSED

All claimed files verified present. All commit hashes verified in git log.
