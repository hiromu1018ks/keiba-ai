# Technology Stack: Betting Strategy Optimization (v1.3)

**Project:** keiba-ai v1.3 Betting Strategy Optimization
**Researched:** 2026-05-04
**Scope:** Kelly criterion, EV-proportional sizing, dynamic drawdown control, multi-criteria bet filtering, parameter sweep for threshold tuning
**Supersedes:** v1.2 STACK.md (pipeline optimization -- all still current)

## Current Installed Stack

| Package | Installed Version | pyproject.toml Minimum | Status |
|---------|-------------------|----------------------|--------|
| Python | 3.11 | >=3.11 | Pinned via mise |
| LightGBM | 4.6.0 | >=4.3 | Up to date |
| XGBoost | 3.2.0 | >=2.0 | Up to date |
| CatBoost | 1.2.10 | >=1.2 | Up to date |
| scikit-learn | 1.8.0 | >=1.4 | Up to date |
| scipy | 1.17.1 | (transitive via sklearn) | Available, not in pyproject.toml |
| pandas | 2.3.3 | >=2.2 | Up to date |
| numpy | 2.4.3 | >=1.26 | Up to date |
| pyarrow | (installed) | >=14.0 | Up to date |
| mlflow | 3.10.1 | >=2.12 | Up to date |
| optuna | 4.8.0 | >=3.5 | Up to date |
| joblib | (installed, transitive) | (transitive via sklearn) | Used for model persistence |

## Recommended Stack

### Production Dependencies: Zero New Additions

| Technology | Version | Purpose | Why No Change |
|------------|---------|---------|----------------|
| numpy | >=1.26 (2.4.3) | Kelly criterion formula | `f* = p - (1-p)/(odds-1)` is a one-line vectorized numpy expression. No library needed for elementary arithmetic. The existing `StakeCalculator._calc_stake()` already implements the Kelly formula (`kelly_fraction = edge / (odds - 1.0)`), which is mathematically equivalent. |
| numpy | >=1.26 (2.4.3) | EV-proportional sizing | `stake = base_fraction * bankroll * (ev - 1.0)` is trivial numpy. No library warranted. |
| numpy | >=1.26 (2.4.3) | Drawdown calculation | `dd = (peak_bankroll - bankroll) / peak_bankroll` is a running-max operation via `np.maximum.accumulate()`. Already implemented in `DrawdownController`. |
| pandas | >=2.2 (2.3.3) | Multi-criteria filtering | `df[(mask1) & (mask2) & (mask3)]` is native pandas boolean indexing. The existing `GateKeeper.filter_bets()` and `WinSelectionGateModel._pass_mask()` demonstrate this pattern. |
| itertools.product | stdlib | Parameter grid enumeration | Already used in `training_pipeline.py:600` for hyperparameter grid search. Same pattern applies to bet-filter threshold sweeps. |
| optuna | >=3.5 (4.8.0) | Smart parameter search | Already installed for model HP tuning. Can be reused for betting threshold optimization when grid search space is large. Uses TPE sampler to avoid exhaustive enumeration. |
| scipy | 1.17.1 (transitive) | Optional constrained optimization | Available as transitive dependency of scikit-learn. `scipy.optimize.minimize_scalar` could optimize Kelly fraction or DD multiplier parameters if grid search proves insufficient. NOT added to pyproject.toml because it is already guaranteed available via sklearn. |

### Existing Code That Already Implements Target Features

| Feature | Existing Implementation | Gap to Fill |
|---------|------------------------|-------------|
| Kelly criterion | `StakeCalculator.calc_stake()` -- half-Kelly with 12.5% cap, 100/10000 yen bounds | Already correct formula. Gap: expose fractional kelly, kelly cap, and EV-proportional as configurable parameters rather than class constants. |
| Drawdown control | `DrawdownController` -- 3-state (NORMAL/REDUCED/RECOVERING), EWMA hybrid, 150-bet rolling window, max 15% adjustment per 20 bets | Already comprehensive. Gap: make DD table thresholds and recovery parameters configurable for sweep tuning. |
| Regime-based filtering | `RegimeDetector` + `MetaSwitcher` -- 3-state (AGGRESSIVE/CONSERVATIVE/COLLAPSED) with hysteresis, per-regime `ev_threshold`, `edge_threshold`, `max_bets_per_race` | Already functional. Gap: allow `COLLAPSED` state to optionally skip all bets (currently just raises thresholds). |
| Edge filtering | `GateKeeper.filter_bets()` -- filters on `bet.edge >= edge_threshold` | Already correct. Gap: edge_threshold is static per regime; needs to be sweepable. |
| Conformal confidence | `RobustConfidenceEstimator` in `SubmodelSet.confidence` | Produces `confidence_score`. Gap: NOT wired into `WinStrategy.generate()` or `GateKeeper` as a filter criterion. Need a `confidence_score >= min_confidence` filter. |
| Odds band analysis | `WinSelectionGateModel` has `_quantile_edges()` and `_bucketize()` | Generates odds bin edges and bucket scores. Gap: NOT used to exclude negative-ROI odds bands in the betting path. Need post-hoc ROI analysis per odds band + exclusion filter. |

### No New Dev Dependencies Needed

The v1.2 dev dependency (pyinstrument) remains relevant if profiling the betting strategy sweep performance.

## NOT Recommended

| Technology | Why Not |
|------------|---------|
| keeks (PyPI) | Provides `KellyCriterion`, `FractionalKellyCriterion`, `DrawdownAdjustedKelly` classes -- but all are 5-10 line functions our `StakeCalculator` already implements. Adds a dependency for trivial math. Version 0.2.0, small community (low GitHub stars), educational focus. Our domain has JRA-specific constraints (25% deduction rate, 100-yen units, 2% race exposure cap) that require custom logic anyway. |
| bet-optimizer (PyPI) | Version 0.0.2, last updated Nov 2023. Provides only `kelly_criterion_bet(prob, odds, bankroll)` and `get_positive_odds(prob)` -- both one-liners. No drawdown control, no fractional Kelly, no regime awareness. Far too minimal. |
| kelly-criterion (PyPI) | PyPI page fails to load properly. Small/niche package. The formula is `f* = (b*p - q) / b` which is identical to our existing implementation. |
| scipy.optimize (direct) | Could use `minimize_scalar` or `minimize(method='trust-constr')` for optimizing Kelly fraction or DD parameters, but this is overkill. The parameter space is small (3-6 threshold values), grid search with `itertools.product` over ~100-500 combinations is fast enough and transparent. Optuna TPE is already available for larger spaces. |
| vectorbt | High-performance backtesting library with native parameter sweeps. Excellent for financial strategies, but would require rewriting our entire backtest loop (1500+ lines in `engine.py` + `race_predictor.py`). Negative ROI for adding a single filtering/sizing layer. |
| backtrader | Event-driven backtesting framework. Same rewrite problem as vectorbt. Our engine is already vectorized and batched. |
| Cython/Numba for Kelly | The Kelly formula is 3 arithmetic operations. Speed is irrelevant -- the bottleneck is the per-race ML inference loop, not the stake calculation. |

## Recommended Implementation Approach

### Kelly Criterion Enhancement (Modify Existing)

The existing `StakeCalculator` already has the correct Kelly formula. Enhancement is parameterization, not new math.

```python
# Current: hardcoded constants
class StakeCalculator:
    FRACTIONAL_KELLY: float = 0.5
    KELLY_FRACTION_CAP: float = 0.25

# Target: configurable via config or sweep
@dataclass
class KellyConfig:
    fractional_kelly: float = 0.5      # 0.25 (quarter) to 1.0 (full)
    kelly_cap: float = 0.25            # max fraction of bankroll
    min_edge: float = 0.005            # skip bets below this edge
    race_exposure_cap: float = 0.02    # max 2% per race
    ev_proportional_weight: float = 0.0  # 0 = pure Kelly, 1 = pure EV-proportional
```

The `ev_proportional_weight` parameter blends Kelly with EV-proportional:
```python
kelly_stake = bankroll * kelly_fraction
ev_prop_stake = bankroll * ev_proportional_fraction * (ev - 1.0)
blended = (1 - w) * kelly_stake + w * ev_prop_stake
```

### Dynamic Drawdown Enhancement (Modify Existing)

The existing `DrawdownController.MULTIPLIER_TABLE` is hardcoded. Make it configurable:

```python
@dataclass
class DDConfig:
    multiplier_table: list[tuple[float, float, float, float, float]]
    rolling_window: int = 150
    ewma_alpha: float = 0.1
    recovery_roi_threshold: float = 0.98
    max_adjustment_per_n_bets: int = 20
    max_adjustment_amount: float = 0.15
```

### Multi-Criteria Filter Chain (New Component)

A new `BetFilterChain` that composes existing filters:

```python
class BetFilterChain:
    """Composable multi-criteria bet filter for v1.3."""

    def __init__(self, filters: list[BetFilter]) -> None:
        self.filters = filters

    def apply(self, candidates: pd.DataFrame, bankroll: float) -> pd.DataFrame:
        df = candidates
        for f in self.filters:
            df = f(df, bankroll)
            if df.empty:
                break
        return df
```

Each filter is a simple callable:
- `ConformalConfidenceFilter(min_confidence=0.1)` -- uses existing `EV_lower_win` at alpha=0.1
- `OddsBandFilter(excluded_bands=[(0, 2.0), (30.0, inf)])` -- post-hoc ROI analysis
- `RegimeFilter(regime_detector)` -- skip bets when COLLAPSED
- `EdgeFilter(min_edge=0.04)` -- existing `GateKeeper` logic

### Parameter Sweep (Reuse Existing Patterns)

The existing `itertools.product` pattern from `training_pipeline.py:600-611`:

```python
from itertools import product as iter_product

# Sweep grid for v1.3 threshold tuning
alpha_grid = [0.05, 0.10, 0.15]           # conformal alpha
edge_grid = [0.03, 0.04, 0.05, 0.06, 0.07]  # min edge
kelly_frac_grid = [0.25, 0.375, 0.5]       # fractional kelly
dd_cap_grid = [0.10, 0.15, 0.20, 0.25]     # DD multiplier cap

for alpha, edge, kelly_frac, dd_cap in iter_product(
    alpha_grid, edge_grid, kelly_frac_grid, dd_cap_grid
):
    result = backtest_with_params(alpha, edge, kelly_frac, dd_cap)
    # ... record results ...
```

For larger search spaces, Optuna TPE is already available:
```python
import optuna

def objective(trial: optuna.Trial) -> float:
    kelly_frac = trial.suggest_float("kelly_frac", 0.2, 0.75)
    min_edge = trial.suggest_float("min_edge", 0.02, 0.10)
    confidence_alpha = trial.suggest_float("confidence_alpha", 0.05, 0.20)
    dd_cap = trial.suggest_float("dd_cap", 0.10, 0.30)
    result = backtest_with_params(kelly_frac, min_edge, confidence_alpha, dd_cap)
    return result.total_roi

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
```

## Integration Points with Existing Stack

| Component | Integration Point | Change Type |
|-----------|-------------------|-------------|
| `StakeCalculator` | Add `KellyConfig` parameter | Modify (parameterize constants) |
| `DrawdownController` | Add `DDConfig` parameter | Modify (parameterize table) |
| `WinStrategy.generate()` | Add confidence filter after EV filter | Modify (add filter step) |
| `GateKeeper.filter_bets()` | Add confidence + odds-band + regime filters | Modify (compose filters) |
| `BacktestEngine.__init__()` | Accept `KellyConfig`, `DDConfig`, `FilterConfig` | Modify (add config params) |
| `RacePredictor.select_bets()` | Wire new filters into bet selection | Modify (add filter chain) |
| `config/settings.yaml` | Add `betting:` section with threshold defaults | Extend (new config section) |
| `BacktestResult` | Add `filter_stats` field (bets filtered per criterion) | Extend (new field) |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Kelly formula | Custom numpy (already implemented) | keeks library | Adds dependency for 3-line math; our JRA-specific constraints (100-yen units, 25% deduction) need custom logic anyway |
| Kelly formula | Custom numpy | bet-optimizer PyPI | Version 0.0.2, trivial functionality, no fractional Kelly or cap support |
| Kelly formula | Custom numpy | kelly-criterion PyPI | Niche package, same formula we already have |
| Parameter sweep | itertools.product + Optuna | scipy.optimize | Overkill for 3-6 parameters; grid search is transparent and fast |
| Parameter sweep | itertools.product + Optuna | vectorbt native sweeps | Would require rewriting 1500+ lines of backtest engine |
| Drawdown control | Parameterize existing DrawdownController | keeks DrawdownAdjustedKelly | keeks implementation is simpler than our existing 3-state hysteresis system |
| Filtering | pandas boolean indexing | Custom DSL / rule engine | 3-5 filter criteria is simple enough for direct pandas expressions |
| Filter orchestration | New BetFilterChain class | No design pattern (inline) | Inline filtering in `WinStrategy.generate()` works but makes sweep harder; a composable chain lets each filter be independently toggled during sweep |

## Installation

```bash
# No new production dependencies needed.
# All required packages are already installed:
#   numpy >= 1.26    -- Kelly formula, vectorization
#   pandas >= 2.2    -- filtering
#   optuna >= 3.5    -- parameter search (already used for model HP)
#   scipy 1.17.1     -- available as sklearn transitive dependency

# Dev dependencies (from v1.2, still relevant):
pip install pyinstrument  # profiling sweep performance
```

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Kelly formula (numpy sufficient) | HIGH | Formula is `f* = p - (1-p)/(odds-1)`, verified in existing `StakeCalculator` code. Mathematically proven, no library risk. |
| EV-proportional sizing | HIGH | Linear scaling `stake = k * bankroll * (ev - 1)`, trivial arithmetic. |
| Drawdown control | HIGH | Existing `DrawdownController` already implements EWMA + hysteresis + N-bet rate limiting. Only parameterization needed. |
| No need for external Kelly libraries | HIGH | Inspected keeks (0.2.0), bet-optimizer (0.0.2), kelly-criterion (PyPI broken page). All provide trivial wrappers around basic arithmetic our code already implements. JRA-specific constraints (deduction rate, minimum bet units, exposure caps) require custom logic regardless. |
| scipy for constrained optimization | MEDIUM | Available as transitive dependency. Not needed for current 3-6 parameter grid, but could be useful if Kelly fraction optimization requires non-linear constraints. Flag as optional. |
| Optuna reuse for threshold sweep | HIGH | Already installed and used for model HP tuning. Same TPE sampler applies to betting threshold optimization. |
| Filter chain pattern | HIGH | Standard composition pattern; `WinSelectionGateModel._pass_mask()` already demonstrates multi-condition filtering. |

## Sources

- [keeks GitHub](https://github.com/wdm0006/keeks) -- bankroll allocation library, v0.2.0, MIT license, implements Kelly/FractionalKelly/DrawdownAdjustedKelly. Reviewed: too minimal for our needs.
- [bet-optimizer PyPI](https://pypi.org/project/bet-optimizer/) -- v0.0.2, last updated Nov 2023. Provides `kelly_criterion_bet()` and `get_positive_odds()` only. Reviewed: trivial, no JRA constraints.
- [kelly-criterion PyPI](https://pypi.org/project/kelly-criterion/) -- PyPI page failed to load. Low confidence in package maintenance.
- [Stack Overflow: scipy.optimize for Kelly](https://stackoverflow.com/questions/63617933/maximize-objective-using-scipy-by-kelly-criterium) -- demonstrates `scipy.optimize.minimize` for Kelly maximization with bounds. Not needed for our grid search approach.
- [Emir's Blog: Kelly Fractions for Simultaneous Bets (Jan 2025)](https://emiruz.com/post/2025-01-05-sim-kelly/) -- independent simultaneous bet sizing. Relevant concept but our system bets max 1-2 per race, not truly simultaneous portfolio optimization.
- Code analysis: `src/betting/stake_calculator.py`, `src/betting/drawdown_controller.py`, `src/betting/win_strategy.py`, `src/betting/orchestrator.py`, `src/betting/gate_keeper.py`, `src/betting/meta_switcher.py`, `src/models/regime_detector.py`, `src/models/win_selection_gate.py`, `src/backtest/engine.py`, `src/backtest/race_predictor.py`, `src/domain/models.py`, `pyproject.toml`
