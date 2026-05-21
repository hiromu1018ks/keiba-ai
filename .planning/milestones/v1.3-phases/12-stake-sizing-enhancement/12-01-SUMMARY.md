---
phase: 12-stake-sizing-enhancement
plan: 01
subsystem: betting/stake_calculator
tags: [kelly-criterion, ev-scaling, constructor-injection, tdd]
dependency_graph:
  requires: [StakeCalculator existing]
  provides: [constructor-injected StakeCalculator, apply_ev_scaling()]
  affects: [stake_calculator.py, test_stake_calculator.py]
tech_stack:
  added: []
  patterns: [constructor injection, EV-proportional scaling, regime-based Kelly]
key_files:
  created: []
  modified:
    - src/betting/stake_calculator.py
    - tests/test_stake_calculator.py
decisions:
  - "FRACTIONAL_KELLY/KELLY_FRACTION_CAP をクラス属性から __init__ 引数に移行 (後方互換デフォルト値維持)"
  - "effective_cap = kelly_fraction_cap * fractional_kelly でレジーム別キャップを実現"
  - "apply_ev_scaling は NaN/zero/negative EV をガードして stake をそのまま返す"
metrics:
  duration: 2min
  completed: "2026-05-05"
  tasks: 1
  tests_added: 17
  tests_total: 38
---

# Phase 12 Plan 01: StakeCalculator Constructor Injection + EV Scaling Summary

コンストラクタ注入によるレジーム別Kelly分数(fractional_kelly)とEV比例乗算器(apply_ev_scaling)を実装。TDD手法でRED/GREEN両ゲート通過。

## Changes

### src/betting/stake_calculator.py
- `FRACTIONAL_KELLY`, `KELLY_FRACTION_CAP` クラス属性を `__init__` インスタンス変数に移行
- `__init__(fractional_kelly=0.5, kelly_fraction_cap=0.25, target_ev=1.10, max_scale=2.0)` 追加
- `calc_stake()` 内参照を `self.FRACTIONAL_KELLY` -> `self.fractional_kelly`, `self.KELLY_FRACTION_CAP` -> `self.kelly_fraction_cap` に変更
- `effective_cap = self.kelly_fraction_cap * self.fractional_kelly` に変更
- `apply_ev_scaling(stake, ev)` メソッド追加: `scale = min(ev / target_ev, max_scale)`

### tests/test_stake_calculator.py
- 既存テスト 2件の属性参照を更新 (`calc.FRACTIONAL_KELLY` -> `calc.fractional_kelly`, etc.)
- TestConstructorInjection: 3テスト (デフォルト値, カスタムfk, 後方互換性)
- TestRegimeBasedKelly: 6テスト (AGGRESSIVE/CONSERVATIVE/COLLAPSED + キャップ計算)
- TestEvScaling: 8テスト (境界値, 中間値, 縮小, ガードケース)

## TDD Gate Compliance

| Gate | Commit | Hash |
|------|--------|------|
| RED | test(12-01): add failing tests | 5fcd4f6 |
| GREEN | feat(12-01): constructor injection + apply_ev_scaling | 4aef72f |
| REFACTOR | Not needed - clean implementation | - |

## Verification Results

- `python -m pytest tests/test_stake_calculator.py -v`: 38 passed in 1.16s
- `python -m ruff check src/betting/stake_calculator.py`: All checks passed
- `StakeCalculator(fractional_kelly=0.25).calc_stake(0.06, 5.0, 100000, BetType.PLACE)` = 300.0 (confirmed)

## Key Behaviors Verified

| Regime | fractional_kelly | stake (edge=0.06, odds=5.0, bankroll=100K) |
|--------|-----------------|---------------------------------------------|
| AGGRESSIVE | 0.50 | 700.0 |
| CONSERVATIVE | 0.25 | 300.0 |
| COLLAPSED | 0.00 | 0.0 |

| EV Scaling | ev | scale | result (stake=1000) |
|------------|-----|-------|---------------------|
| Boundary | 1.10 | 1.0 | 1000.0 |
| Max cap | 2.20 | 2.0 | 2000.0 |
| Mid | 1.50 | 1.3636 | 1363.64 |
| Shrink | 0.80 | 0.7273 | 727.27 |
| Guard (NaN) | NaN | 1.0 | 1000.0 |
| Guard (<=0) | 0/-1 | 1.0 | 1000.0 |

## Deviations from Plan

None - plan executed exactly as written.

## Threat Surface

No new threat surface introduced. All changes are pure computation with no external inputs.

## Self-Check: PASSED

- src/betting/stake_calculator.py: FOUND
- tests/test_stake_calculator.py: FOUND
- 12-01-SUMMARY.md: FOUND
- 5fcd4f6 (RED gate): FOUND
- 4aef72f (GREEN gate): FOUND
- fa01fb7 (DOCS): FOUND
- No unexpected file deletions
- No untracked files
