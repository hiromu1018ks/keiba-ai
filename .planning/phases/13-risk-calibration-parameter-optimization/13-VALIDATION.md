---
phase: 13-risk-calibration-parameter-optimization
slug: risk-calibration-parameter-optimization
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-05
---

# Phase 13 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `python -m pytest tests/test_drawdown_controller.py tests/test_parameter_freeze.py tests/test_strategy_optimizer.py tests/test_regime_detector.py tests/test_meta_switcher.py -v --tb=short` |
| **Full suite command** | `python -m pytest tests/ -v --tb=short` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick command (5 test files)
- **After every plan wave:** Run full suite
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 13-01-01 | 01 | 1 | RISK-01 | T-13-01 / T-13-02 | DDConfig __post_init__ rejects invalid thresholds, negative multipliers, zero rolling_window/min_stay | unit | `pytest tests/test_drawdown_controller.py -v` | ✅ | ✅ |
| 13-01-02 | 01 | 1 | RISK-01 | T-13-02 | DrawdownController rejects peak_bankroll <= 0 (WR-02 fix) | unit | `pytest tests/test_drawdown_controller.py::TestDrawdownControllerInit -v` | ✅ | ✅ |
| 13-01-03 | 01 | 1 | RISK-01 | — | DD% only 3-tier state transitions (NORMAL/REDUCED/STOP) with hysteresis | unit | `pytest tests/test_drawdown_controller.py::TestDrawdownControllerCore -v` | ✅ | ✅ |
| 13-01-04 | 01 | 1 | RISK-01 | — | Multiplier control (normal/reduced/stop) + rate limiting | unit | `pytest tests/test_drawdown_controller.py::TestDrawdownControllerMultiplier -v` | ✅ | ✅ |
| 13-01-05 | 01 | 1 | RISK-01 | — | DDState reporting + peak bankroll tracking | unit | `pytest tests/test_drawdown_controller.py::TestDrawdownControllerGetState -v` | ✅ | ✅ |
| 13-01-06 | 01 | 1 | RISK-01 | — | BacktestEngine strategy_params injection + DDConfig import | unit | `pytest tests/test_strategy_optimizer.py::TestRunSingleBacktest -v` | ✅ | ✅ |
| 13-02-01 | 02 | 1 | VAL-01 | — | RegimeDetector override_params constructor injection (CR-01 fix verified) | unit | `pytest tests/test_regime_detector.py::TestRegimeDetector::test_override_params_injects_values -v` | ✅ | ✅ |
| 13-02-02 | 02 | 1 | VAL-01 | — | MetaSwitcher._default_params() matches RegimeDetector.get_strategy_params() | unit | `pytest tests/test_meta_switcher.py::TestMetaSwitcher::test_default_params_match_regime_detector -v` | ✅ | ✅ |
| 13-02-03 | 02 | 1 | VAL-01 | T-13-03 / T-13-04 | JSON manifest save/verify/load + SHA256 tamper detection | unit | `pytest tests/test_parameter_freeze.py::TestStrategyManifest -v` | ✅ | ✅ |
| 13-02-04 | 02 | 1 | VAL-01 | — | ParameterFreezeProtocol freeze/verify/context manager | unit | `pytest tests/test_parameter_freeze.py::TestParameterFreezeProtocol -v` | ✅ | ✅ |
| 13-03-01 | 03 | 2 | VAL-02 | T-13-06 | StrategyOptimizer ~14-dim parameter space via Optuna TPE | unit | `pytest tests/test_strategy_optimizer.py::TestSuggestParams -v` | ✅ | ✅ |
| 13-03-02 | 03 | 2 | VAL-02 | — | _build_strategy_config DDConfig construction + dd_threshold auto-correction | unit | `pytest tests/test_strategy_optimizer.py::TestBuildStrategyConfig -v` | ✅ | ✅ |
| 13-03-03 | 03 | 2 | VAL-02 | — | Objective function: ROI primary + bet count constraint + MedianPruner | unit | `pytest tests/test_strategy_optimizer.py::TestObjective -v` | ✅ | ✅ |
| 13-03-04 | 03 | 2 | VAL-02 | — | Optuna optimize loop: best_params, manifest save, pruner stats | unit | `pytest tests/test_strategy_optimizer.py::TestOptimize -v` | ✅ | ✅ |
| 13-03-05 | 03 | 2 | VAL-02 | T-13-07 | _run_single_backtest: ModelLoader + BacktestEngine + regime_overrides update (not replace) | unit | `pytest tests/test_strategy_optimizer.py::TestRunSingleBacktest -v` | ✅ | ✅ |

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. No Wave 0 needed.

---

## Manual-Only Verifications

All phase behaviors have automated verification.

---

## Validation Audit 2026-05-05

| Metric | Count |
|--------|-------|
| Gaps found | 4 |
| Resolved | 4 |
| Escalated | 0 |

### Gaps Resolved

1. **peak_bankroll <= 0 validation** — Added `TestDrawdownControllerInit` (2 tests) to `test_drawdown_controller.py`
2. **RegimeDetector override_params injection** — Added 2 tests to `test_regime_detector.py`
3. **MetaSwitcher-RegimeDetector value alignment** — Added parametrized cross-validation (3 tests) to `test_meta_switcher.py`
4. **_build_strategy_config dd_threshold auto-correction** — Added 1 test to `test_strategy_optimizer.py`

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-05
