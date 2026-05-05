---
phase: 13-risk-calibration-parameter-optimization
plan: 01
subsystem: betting
tags: [drawdown-controller, kelly, risk-management, dataclass, hysteresis]

# Dependency graph
requires:
  - phase: 12-stake-sizing-enhancement
    provides: StakeCalculator, RacePredictor DD/Kelly integration
provides:
  - DDConfig dataclass with __post_init__ validation
  - DD%-only 3-tier DrawdownController (NORMAL/REDUCED/STOP)
  - RecoveryState.STOP enum (replaces RECOVERING)
  - DDState without rolling_roi
  - BacktestEngine strategy_params injection
affects: [13-02, 13-03]

# Tech tracking
tech-stack:
  added: []
  patterns: [constructor-injection-config, dd-percent-only-control, hysteresis-state-machine, gradual-recovery]

key-files:
  created: []
  modified:
    - src/betting/drawdown_controller.py
    - src/domain/types.py
    - src/domain/models.py
    - src/backtest/engine.py
    - tests/test_drawdown_controller.py
    - tests/test_domain.py

key-decisions:
  - "DD% only for DD control: ROI is too noisy at 10% win-rate environment"
  - "Hysteresis counter increments even on blocked transitions to avoid permanent blockage"
  - "STOP->NORMAL forced through REDUCED for gradual recovery"
  - "strategy_params dict injection for Optuna optimization compatibility"

patterns-established:
  - "DDConfig dataclass: __post_init__ validation for threshold consistency (dd_threshold_2 > dd_threshold_1)"
  - "Constructor injection: all DD parameters configurable via DDConfig cfg parameter"
  - "3-tier state machine: NORMAL/REDUCED/STOP with hysteresis min_stay_races"

requirements-completed: [RISK-01]

# Metrics
duration: 16min
completed: 2026-05-05
---

# Phase 13 Plan 01: DD再設計・パラメータ外部化 Summary

**DrawdownControllerをROI依存から完全に解放し、DD%のみの3段階制御(NORMAL/REDUCED/STOP)に再設計。DDConfig dataclassで全パラメータを外部注入可能にし、BacktestEngineにstrategy_params注入を追加**

## Performance

- **Duration:** 16 min
- **Started:** 2026-05-05T01:38:18Z
- **Completed:** 2026-05-05T01:54:29Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- DrawdownController完全再設計: ROI計算コード(numpy依存含む)を完全除去
- DDConfig dataclass: 9パラメータ + __post_init__閾値整合性検証
- RecoveryState enum: RECOVERING -> STOP に変更(3値のみ)
- DDState: rolling_roi フィールド除去
- ヒステリシス(min_stay_races)による状態遷移発振防止
- 段階的リカバリ: STOP -> REDUCED -> NORMAL (即時復帰禁止)
- BacktestEngine: strategy_params dict注入でOptuna最適化対応
- 全31テスト書き直し(DDConfig 10件 + Core 10件 + Multiplier 8件 + GetState 3件)

## Task Commits

Each task was committed atomically:

1. **Task 1: DrawdownController再設計** - `35e883c` (feat)
2. **Task 2: BacktestEngine strategy_params注入 + 全テスト書き直し** - `bd752aa` (feat)

## Files Created/Modified
- `src/betting/drawdown_controller.py` - DDConfig dataclass + ROI除去済みDrawdownController
- `src/domain/types.py` - RecoveryState enum (NORMAL/REDUCED/STOP)
- `src/domain/models.py` - DDState dataclass (rolling_roi除去)
- `src/backtest/engine.py` - strategy_params注入 + DDConfig import + update() sig変更
- `tests/test_drawdown_controller.py` - 全31テスト書き直し
- `tests/test_domain.py` - RecoveryState/ DDState更新追従

## Decisions Made
- DD% only for DD control: WIN的中率10%環境ではROIがノイジーすぎてDD制御信号として不適切
- ヒステリシスblocked時も_races_in_stateをインクリメント(永久ブロック回避)
- STOP->NORMAL即時遷移を禁止し強制的にREDUCED経由(段階的リカバリ)
- strategy_params dictでDDConfig/StakeCalculatorを動的生成(Plan 03 Optuna対応)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ヒステリシス_transitionでblocked時の_races_in_state非インクリメント**
- **Found during:** Task 2 (テスト実行時)
- **Issue:** _transition()のヒステリシスblocked return時に_races_in_stateがインクリメントされず、min_stay_races到達後に遷移不可能になる永久ブロックバグ
- **Fix:** `return`前に`self._races_in_state += 1`を追加
- **Files modified:** src/betting/drawdown_controller.py
- **Verification:** 全31テストPASS + 全1198テスト回帰なし
- **Committed in:** bd752aa (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** バグ修正は計画範囲内。スコープクリープなし。

## Issues Encountered
- ヒステリシスのオフバイワン: min_stay_races=N のとき実際の遷移にN+1回updateが必要(設計通り、テストで正しく検証)

## User Setup Required
None - no external service configuration required.

## Self-Check: PASSED

All files verified present. Both task commits (35e883c, bd752aa) confirmed in git log.

## Next Phase Readiness
- DDConfig + strategy_params注入によりPlan 03のOptuna最適化がengineパラメータを直接制御可能
- 全1198テストPASSでリグレッションなし
- Plan 02 (パラメータ外部化YAML) の前提条件完了

---
*Phase: 13-risk-calibration-parameter-optimization*
*Completed: 2026-05-05*
