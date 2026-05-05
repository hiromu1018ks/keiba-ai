---
status: resolved
trigger: "roi_threshold 死んだパラメータの修正 — Optunaが14次元のうち1次元を無駄に探索している。OddsBandFilterに接続するか、探索空間から削除する"
created: 2026-05-05
updated: 2026-05-05
---

## Symptoms

- **Expected**: roi_threshold が Optuna で最適化され、OddsBandFilter 等のフィルタで実際に使用される
- **Actual**: roi_threshold は Optuna の探索空間に存在するが、どこにも接続されていない（デッドパラメータ）
- **Errors**: 実行時エラーなし。コード解析で発見
- **Timeline**: パラメータ導入時から存在
- **Reproduction**: Optuna 最適化の search space 定義と roi_threshold の使用箇所をコード照合

## Evidence

### 1. Optuna Search Space Definition
**File**: `src/tuning/strategy_optimizer.py:79`
```python
params["roi_threshold"] = trial.suggest_float("roi_threshold", 0.8, 1.2)
```
✅ `roi_threshold` is defined in 14-dimensional search space

### 2. StrategyOptimizer to BacktestEngine
**File**: `src/tuning/strategy_optimizer.py:115`
```python
return {
    "dd_config": dd_config,
    "regime_overrides": regime_overrides,
    "fractional_kelly": params.get("fk_aggressive", 0.5),
    "target_ev": params["target_ev"],
    "max_scale": params["max_scale"],
    "roi_threshold": params["roi_threshold"],  # ← Added to strategy_params
}
```
✅ `roi_threshold` is included in strategy_params dict

**File**: `src/tuning/strategy_optimizer.py:157`
```python
engine = BacktestEngine(
    models=models,
    initial_bankroll=self.initial_bankroll,
    betting_mode="kelly",
    diag_prefix=f"opt_fold{fold_idx}",
    betting_target="win",
    strategy_params=strategy_config,  # ← Passed to BacktestEngine
)
```
✅ strategy_params is passed to BacktestEngine

### 3. BacktestEngine Stores but Never Uses strategy_params
**File**: `src/backtest/engine.py:379`
```python
self.strategy_params = strategy_params
```
❌ strategy_params is stored but NEVER accessed again in entire file

**File**: `src/backtest/engine.py:384`
```python
self._odds_band_filter: OddsBandFilter | None = None
if betting_target == "win":
    self._odds_band_filter = OddsBandFilter()  # ← No params passed!
```
❌ OddsBandFilter is instantiated WITHOUT roi_threshold parameter

**File**: `src/backtest/engine.py:662`
```python
if self._odds_band_filter is not None and training_bet_history:
    self._odds_band_filter.calibrate(training_bet_history)  # ← No roi_threshold!
```
❌ calibrate() called WITHOUT roi_threshold parameter

### 4. OddsBandFilter Implementation
**File**: `src/betting/odds_band_filter.py:37`
```python
def calibrate(self, bet_history: list[dict[str, Any]]) -> None:
    # ... method has NO roi_threshold parameter
```
❌ calibrate() method signature does NOT accept roi_threshold

**File**: `src/betting/odds_band_filter.py:69`
```python
if roi < 1.0:  # D-07: ROI < 100% → exclude
    self._excluded_bands.add(name)
```
❌ Hard-coded threshold `1.0` instead of using configurable parameter

### 5. Design Document Specification
**File**: `.planning/research/ARCHITECTURE.md:223`
```python
def calibrate(
    bet_history: list[dict],
    n_bins: int = 10,
    min_samples: int = 50,
    roi_threshold: float = 0.95,  # ← SPECIFIED in design!
) -> OddsBandFilter:
```
✅ Design document SPECIFIES roi_threshold parameter

## Root Cause

**SPECIFICATION-IMPLEMENTATION GAP**:
1. Architecture design (ARCHITECTURE.md) specified `roi_threshold` as a parameter to `OddsBandFilter.calibrate()`
2. Implementation (odds_band_filter.py) NEVER added this parameter to the method signature
3. Hard-coded `roi < 1.0` threshold used instead
4. No one noticed during implementation that the parameter was missing
5. Optuna search space included the parameter anyway, creating a dead dimension

## Current Focus

- hypothesis: CONFIRMED - OddsBandFilter.calibrate() missing roi_threshold parameter per ARCHITECTURE.md spec
- next_action: FIXED and verified

## Resolution

- root_cause: OddsBandFilter に roi_threshold パラメータが未実装。ハードコード 1.0 で代替されていた。
- fix: __init__() に roi_threshold パラメータ追加、engine.py で strategy_params から配線、テスト2件追加
- files_changed: src/betting/odds_band_filter.py, src/backtest/engine.py, tests/test_odds_band_filter.py
- verification: 1249 passed, 1 skipped, 0 failed
