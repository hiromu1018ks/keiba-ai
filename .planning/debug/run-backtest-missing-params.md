---
status: resolved
trigger: "run_backtest.py が strategy_params と training_bet_history を渡さないため、Phase 11-13の機能が標準バックテストで使われない"
created: 2026-05-05
updated: 2026-05-05
---

# Debug Session: run-backtest-missing-params

## Symptoms

**Expected behavior:**
- run_backtest.py のバックテストで、Phase 11-13 で導入されたフィルタ（OddsBandFilter 等）が strategy_params 経由で有効化される
- training_bet_history を用いたキャリブレーションも正しく動作する
- 最適化中（Optuna）の _run_single_backtest() でも OddsBandFilter がキャリブレーションデータを使える

**Actual behavior:**
- run_backtest.py が strategy_params を BacktestEngine に渡さない
- run_backtest.py が training_bet_history を BacktestEngine に渡さない
- _run_single_backtest() が training_bet_history を渡さないため、最適化中は OddsBandFilter が実質無効
- コード解析で発見（実行時エラーはなし）

**Error messages:**
- 実行時エラーなし。静的コード解析で発見。

**Timeline:**
- Phase 11-13 の機能導入時から存在。run_backtest.py への配線が漏れていた。

**Reproduction:**
- コード解析: run_backtest.py の _run_single_backtest() と BacktestEngine の呼び出しを確認

## Current Focus

**hypothesis:** CONFIRMED
**next_action:** fix applied
**reasoning_checkpoint:** 2つの関連問題（#1 配線漏れ、#2 最適化中キャリブレーション無効）は同じ根本原因

## Evidence

- 2026-05-05: `scripts/run_backtest.py` lines 349-356: `_run_single_year()` constructs BacktestEngine without `strategy_params`:
  ```python
  engine = BacktestEngine(
      models=models,
      store=store,
      betting_mode=args.betting_mode,
      diag_prefix=f"bt_{test_year}",
      betting_target=args.betting_target,
  )
  result = engine.run(test_start, test_end)  # no training_bet_history
  ```
  BacktestEngine.__init__ accepts `strategy_params` (line 367) but it is not passed.
  BacktestEngine.run accepts `training_bet_history` (line 419) but it is not passed.

- 2026-05-05: `scripts/run_backtest.py` lines 474-481: `_run_multi_year()` has the same missing wiring.

- 2026-05-05: `src/tuning/strategy_optimizer.py` lines 151-161: `_run_single_backtest()` correctly passes `strategy_params=strategy_config` to BacktestEngine, but does NOT pass `training_bet_history` to `engine.run()`.

- 2026-05-05: `src/backtest/engine.py`: BacktestEngine correctly accepts and uses both `strategy_params` and `training_bet_history`. OddsBandFilter is created with `roi_threshold` from strategy_params and calibrated via training_bet_history.

- 2026-05-05: `src/betting/odds_band_filter.py`: OddsBandFilter.calibrate() computes per-band ROI from training data. Without training_bet_history, filter() is a no-op.

## Eliminated

- Not a bug in BacktestEngine itself (it correctly accepts and uses strategy_params and training_bet_history).
- Not a bug in OddsBandFilter (it correctly implements calibrate/filter).
- Not a missing import (all classes are properly imported where needed).

## Resolution

**root_cause:** Two wiring gaps:
1. `scripts/run_backtest.py` does not pass `strategy_params` or `training_bet_history` to BacktestEngine in either `_run_single_year()` or `_run_multi_year()`.
2. `src/tuning/strategy_optimizer.py` `_run_single_backtest()` passes `strategy_params` but does not pass `training_bet_history` to `engine.run()`.

**fix:** Applied to 3 files:

1. `scripts/run_backtest.py`:
   - Added `--strategy-manifest` CLI option to load Optuna-optimized strategy parameters
   - Added `_load_strategy_params()` to load and verify manifest JSON
   - Added `_build_strategy_config_from_manifest()` to convert manifest params to BacktestEngine format
   - Added `_collect_training_bet_history()` to run a training-period backtest for OddsBandFilter calibration
   - Both `_run_single_year()` and `_run_multi_year()` now pass `strategy_params` and `training_bet_history`

2. `src/tuning/strategy_optimizer.py`:
   - `_run_single_backtest()` now runs a training-period backtest to collect `training_bet_history`
   - Passes `training_bet_history` to the test-period `engine.run()` call
   - Wrapped in try/except so calibration failure does not abort optimization

3. `tests/test_strategy_optimizer.py`:
   - Updated `test_calls_model_loader` to expect 2 BacktestEngine constructions (train + test)
   - Added assertions verifying `training_bet_history` is passed to the final `engine.run()` call
   - Added `bet_history` to mock_result objects for realistic testing

**tests:** 14/14 strategy_optimizer tests pass, 1235/1235 full suite passes.

**specialist_hint:** python
