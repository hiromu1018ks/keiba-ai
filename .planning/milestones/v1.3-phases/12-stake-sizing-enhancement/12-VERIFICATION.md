---
phase: 12-stake-sizing-enhancement
verified: 2026-05-05T15:30:00Z
status: human_needed
score: 2/3 must-haves verified
overrides_applied: 0
deferred:
  - truth: "フィルター+サイジング変更後のバックテストROIがベースライン(89.0%)を上回る"
    addressed_in: "Phase 13"
    evidence: "Phase 13 SC-3: 'Optuna TPEで全戦略パラメータの同時最適化が実行され、最適設定のバックテストROIがベースライン(89.0%)を上回る (VAL-02)'"
---

# Phase 12: Stake Sizing Enhancement Verification Report

**Phase Goal:** レジーム状態に応じたKelly分数とEV比例乗算器により、高確信ベットに重点配分された賭け金が算出される
**Verified:** 2026-05-05T15:30:00Z
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | レジーム状態別にKelly分数が異なり、AGGRESSIVE > CONSERVATIVE > COLLAPSED(=0)の順で賭け金が計算される (SIZE-01) | VERIFIED | StakeCalculator: fk=0.50 -> 700, fk=0.25 -> 300, fk=0.00 -> 0. RegimeDetector.get_strategy_params() returns fractional_kelly per regime. engine.py line 713 injects into StakeCalculator. 6 tests pass (TestRegimeBasedKelly + TestStakeSizingIntegration). |
| 2 | 高EVベットの賭け金にEV比例乗算器(min(ev/target_ev, max_scale))が適用され、同一レジーム内でEVが高いほど賭け金が大きくなる (SIZE-02) | VERIFIED | apply_ev_scaling() in stake_calculator.py line 134-144. Win path in race_predictor.py line 660-662: ev_val = row.get(ev_col, 0); stake = apply_ev_scaling(stake, ev=ev_val). Kelly->EV->DD pipeline confirmed. EV=1.50 scaled 700->954.54->900. test_ev_scaling_in_select_bets passes. |
| 3 | フィルター+サイジング変更後のバックテストROIがベースライン(89.0%)を上回る | DEFERRED | Requires full backtest execution with real data + parameter optimization. Deferred to Phase 13 (VAL-02: Optuna TPE optimization targeting ROI > 89.0%). |

**Score:** 2/3 truths verified (1 deferred to Phase 13)

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | バックテストROIがベースライン(89.0%)を上回る | Phase 13 | Phase 13 SC-3: "Optuna TPEで全戦略パラメータの同時最適化が実行され、最適設定のバックテストROIがベースライン(89.0%)を上回る (VAL-02)" |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/betting/stake_calculator.py` | Constructor injection + apply_ev_scaling() | VERIFIED | __init__ with fractional_kelly, kelly_fraction_cap, target_ev, max_scale. calc_stake uses self.fractional_kelly. apply_ev_scaling() at line 134. |
| `tests/test_stake_calculator.py` | Regime Kelly + EV scaling tests | VERIFIED | 38 tests: TestConstructorInjection (3), TestRegimeBasedKelly (6), TestEvScaling (8), plus existing tests. All pass. |
| `src/models/regime_detector.py` | fractional_kelly in get_strategy_params() | VERIFIED | All 3 regime dicts contain fractional_kelly: AGGRESSIVE=0.50, CONSERVATIVE=0.25, COLLAPSED=0.00. |
| `src/betting/meta_switcher.py` | fractional_kelly in _default_params() | VERIFIED | All 3 regime dicts contain fractional_kelly: AGGRESSIVE=0.50, CONSERVATIVE=0.25, COLLAPSED=0.00. |
| `src/backtest/engine.py` | fractional_kelly injection from regime_params | VERIFIED | Lines 711-714: regime_params.get("fractional_kelly", 0.5) injected into stake_calc.fractional_kelly each race. |
| `src/backtest/race_predictor.py` | Kelly->EV->DD pipeline in win path | VERIFIED | Lines 660-662: ev_val = row.get(ev_col, 0); stake = apply_ev_scaling(stake, ev=ev_val) between calc_stake and DD. |
| `config/settings.yaml` | betting_strategy section | VERIFIED | Section present at end of file with default_fractional_kelly, kelly_fraction_cap, target_ev, max_scale, regime_fractions. |
| `tests/test_backtest_engine.py` | Integration tests | VERIFIED | TestStakeSizingIntegration class with 3 tests: test_regime_injects_fractional_kelly, test_ev_scaling_in_select_bets, test_collapsed_regime_zero_stake. All pass. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/backtest/engine.py` | `src/betting/stake_calculator.py` | regime_params['fractional_kelly'] -> stake_calc.fractional_kelly | WIRED | Line 713: fk = float(regime_params.get("fractional_kelly", 0.5)); line 714: self._race_predictor.stake_calc.fractional_kelly = fk |
| `src/backtest/race_predictor.py` | `src/betting/stake_calculator.py` | apply_ev_scaling(stake, ev=ev_val) | WIRED | Line 662: stake = self.stake_calc.apply_ev_scaling(stake, ev=ev_val) |
| `src/models/regime_detector.py` | `src/backtest/engine.py` | get_strategy_params(regime) | WIRED | Line 710: regime_params = self.models.regime_detector.get_strategy_params(regime); returns dict with fractional_kelly key |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `src/backtest/engine.py` | fractional_kelly | regime_params from get_strategy_params() | RegimeDetector.get_strategy_params() returns hardcoded values per regime | FLOWING |
| `src/backtest/race_predictor.py` (win path) | ev_val | row.get(ev_col, 0) where ev_col="win_selection_ev" | Comes from WinSelectionGate pipeline in predict() | FLOWING |
| `src/backtest/race_predictor.py` (win path) | stake | calc_stake -> apply_ev_scaling -> dd_ctrl.adjust_stake | Full pipeline: Kelly stake scaled by EV ratio, capped by DD | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Regime ordering (AGGRESSIVE > CONSERVATIVE > COLLAPSED) | python -c "from betting.stake_calculator import StakeCalculator; ..." | 700 > 300 > 0 (True) | PASS |
| EV scaling: mid-EV > boundary > low-EV | python -c "calc.apply_ev_scaling(...)" | 1363.64 > 1000 > 727.27 (True) | PASS |
| All Phase 12 tests pass | python -m pytest tests/test_stake_calculator.py tests/test_backtest_engine.py -v | 92 passed in 2.90s | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SIZE-01 | 12-01, 12-02 | レジーム状態に応じたKelly分数 (AGGRESSIVE/CONSERVATIVE/COLLAPSED) | SATISFIED | StakeCalculator constructor injection + RegimeDetector fractional_kelly + engine.py injection + tests |
| SIZE-02 | 12-01, 12-02 | 高EV機会にEV比例乗算器 (min(ev/target_ev, max_scale)) で重点配分 | SATISFIED | apply_ev_scaling() + win pipeline integration + test_ev_scaling_in_select_bets |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No anti-patterns found in Phase 12 files |

Note: engine.py has pre-existing lint issues (F841 unused variables in build_wide_payout_map, E501 line length, and race_predictor.py has F401 unused import). These are NOT from Phase 12 changes.

### Human Verification Required

### 1. Full Backtest ROI Validation

**Test:** Run backtest with Phase 11 + 12 changes: `python scripts/run_backtest.py --train-start 20200101 --train-end 20231231 --test-start 20240101 --test-end 20241231`
**Expected:** ROI should show measurable change from baseline 89.0%. Phase 13 Optuna optimization will target > 89.0%.
**Why human:** Full backtest requires PostgreSQL with EveryDB2 data and ~57 minutes execution time. Cannot be verified programmatically in this context.

### Gaps Summary

No blocking gaps found. The two SIZE requirements (SIZE-01, SIZE-02) are fully implemented and wired:
- StakeCalculator accepts fractional_kelly via constructor injection with backward-compatible defaults
- RegimeDetector and MetaSwitcher both return regime-specific fractional_kelly values
- engine.py injects fractional_kelly per-race from regime_params
- Win betting path follows Kelly -> EV scaling -> DD pipeline order
- All 92 tests pass including 3 new integration tests

The third success criterion (ROI > 89.0%) is deferred to Phase 13 where Optuna parameter optimization will target this goal.

---

_Verified: 2026-05-05T15:30:00Z_
_Verifier: Claude (gsd-verifier)_
