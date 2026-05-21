---
phase: 12-stake-sizing-enhancement
plan: 02
subsystem: betting/stake_calculator, backtest/engine
tags: [kelly-criterion, ev-scaling, regime-based-sizing, pipeline-integration]
dependency_graph:
  requires: [StakeCalculator constructor injection (12-01)]
  provides: [regime-based fractional_kelly injection, Kelly->EV->DD pipeline]
  affects: [regime_detector.py, meta_switcher.py, engine.py, race_predictor.py, settings.yaml]
tech_stack:
  added: []
  patterns: [regime parameter injection, EV-proportional pipeline]
key_files:
  created: []
  modified:
    - src/models/regime_detector.py
    - src/betting/meta_switcher.py
    - src/backtest/engine.py
    - src/backtest/race_predictor.py
    - config/settings.yaml
    - tests/test_backtest_engine.py
decisions:
  - "fractional_kelly を regime_params dict に格納し engine.py で毎レース注入 (コンストラクタ再生成不要)"
  - "EV乗算を Kelly直後・DD直前の位置に挿入 (Kelly->EV->DD 順序)"
  - "COLLAPSED は skip flag + fractional_kelly=0.00 の二重ガード"
metrics:
  duration: 3min
  completed: "2026-05-05"
  tasks: 2
  tests_added: 3
  tests_total: 1204
---

# Phase 12 Plan 02: Regime-based Kelly + EV Pipeline Integration Summary

RegimeDetector/MetaSwitcher に fractional_kelly を追加し、engine.py から RacePredictor まで Kelly->EV乗算->DD パイプラインを統合。レジーム状態に応じた動的サイジングがバックテストパスで動作する。

## Changes

### src/models/regime_detector.py
- `get_strategy_params()` の3レジーム dict 全てに `fractional_kelly` を追加
  - AGGRESSIVE: 0.50 (half-Kelly)
  - CONSERVATIVE: 0.25 (quarter-Kelly)
  - COLLAPSED: 0.00 (no betting)

### src/betting/meta_switcher.py
- `_default_params()` の3レジーム dict 全てに `fractional_kelly` を追加 (同値)

### config/settings.yaml
- `betting_strategy` section を追加: default_fractional_kelly, kelly_fraction_cap, target_ev, max_scale, regime_fractions

### src/backtest/engine.py
- レースループ内の regime_params 取得直後に fractional_kelly 注入を追加
- `regime_params.get("fractional_kelly", 0.5)` で StakeCalculator に注入

### src/backtest/race_predictor.py
- `select_bets()` winパスで Kelly->EV乗算->DD パイプラインに変更
- `ev_val = float(row.get(ev_col, 0))` で EV 値を取得し `apply_ev_scaling(stake, ev=ev_val)` に渡す

### tests/test_backtest_engine.py
- `TestStakeSizingIntegration` クラス: 3テスト追加
  - `test_regime_injects_fractional_kelly`: CONSERVATIVE 0.25 / COLLAPSED 0.00 注入確認
  - `test_ev_scaling_in_select_bets`: Kelly stake 700 -> EV scaled 954.54 -> floor 900 確認
  - `test_collapsed_regime_zero_stake`: fractional_kelly=0.00 で stake=0, bets=[] 確認

## Verification Results

- `python -m pytest tests/test_backtest_engine.py tests/test_stake_calculator.py -v`: 92 passed in 2.93s
- `python -m ruff check src/models/regime_detector.py src/betting/meta_switcher.py`: All checks passed
- `grep "fractional_kelly" src/models/regime_detector.py src/betting/meta_switcher.py src/backtest/engine.py`: 9 matches
- `grep "apply_ev_scaling" src/backtest/race_predictor.py`: 1 match (line 662)
- `grep "betting_strategy:" config/settings.yaml`: 1 match

## Key Pipeline Behavior

| Regime | fractional_kelly | Kelly (edge=0.06, odds=5.0, bankroll=100K) | EV=1.50 scaled | After DD (floor/100) |
|--------|-----------------|---------------------------------------------|----------------|----------------------|
| AGGRESSIVE | 0.50 | 700.0 | 954.54 | 900 |
| CONSERVATIVE | 0.25 | 300.0 | 409.09 | 400 |
| COLLAPSED | 0.00 | 0.0 | 0.0 | 0 (no bet) |

## Deviations from Plan

None - plan executed exactly as written.

## Threat Surface

No new threat surface beyond plan's threat_model. T-12-03 mitigated by `.get("fractional_kelly", 0.5)` default. T-12-04 mitigated by `apply_ev_scaling` NaN/0/negative guard (implemented in 12-01).

## Self-Check: PASSED

- src/models/regime_detector.py: FOUND
- src/betting/meta_switcher.py: FOUND
- src/backtest/engine.py: FOUND
- src/backtest/race_predictor.py: FOUND
- config/settings.yaml: FOUND
- tests/test_backtest_engine.py: FOUND
- 485b371 (Task 1): FOUND
- 6d5def3 (Task 2): FOUND
- No unexpected file deletions
- No untracked files
