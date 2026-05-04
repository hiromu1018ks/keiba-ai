# Architecture: Betting Strategy Optimization Integration

**Project:** keiba-ai v1.3 -- Betting Strategy Optimization
**Researched:** 2026-05-04
**Scope:** How Kelly criterion sizing, EV-proportional sizing, dynamic drawdown control, and multi-criteria bet filtering integrate with the existing Strategy/Orchestrator/StakeCalculator/DDController architecture.

## Current Architecture Map

The betting layer has a well-defined protocol-based architecture with clear dependency injection boundaries. The following diagram shows the current data flow during backtest (the primary integration point):

```
DataRepository
    |
    v
FeatureEngine.build_all()          (14 modules, 100+ features)
    |
    v
BacktestEngine.run()               <-- MAIN LOOP (per-race)
    |
    +---> RacePredictor.predict()   (inference: stage1 -> market -> win -> EV correction -> benter -> gate)
    |         |
    |         +---> RegimeDetector.detect()  (3-state: aggressive/conservative/collapsed)
    |
    +---> RacePredictor.get_win_candidates()   OR   get_place_candidates()
    |         |
    |         +---> WinSelectionGateModel.score() / PlaceSelectionGateModel.score()
    |         +---> Candidate filtering + pruning (regime-dependent)
    |
    +---> RacePredictor.select_bets()
    |         |
    |         +---> StakeCalculator.calc_stake(edge, odds, bankroll, bet_type)
    |         |         |
    |         |         +---> Kelly fraction = edge / (odds - 1)
    |         |         +---> x FRACTIONAL_KELLY (0.5)
    |         |         +---> cap at KELLY_FRACTION_CAP (0.25)
    |         |
    |         +---> DrawdownController.adjust_stake(base_stake, bankroll)
    |                   |
    |                   +---> Multiplier from DD table (DD% x ROI)
    |                   +---> Recovery state machine (NORMAL/REDUCED/RECOVERING)
    |                   +---> N-bet rate limiter
    |
    +---> _settle_bet()              (final odds from payout map)
    +---> DDController.update()      (feedback: bankroll + bet_return)
```

### Existing Component Inventory

| Component | File | Role | Protocol |
|-----------|------|------|----------|
| `BettingOrchestrator` | `betting/orchestrator.py` | 12-step race flow (NOT used in backtest) | Main entry |
| `RacePredictor` | `backtest/race_predictor.py` | Per-race inference + bet selection | Core in backtest |
| `StakeCalculator` | `betting/stake_calculator.py` | Kelly stake + race exposure cap | `StakeCalculatorProtocol` |
| `DrawdownController` | `betting/drawdown_controller.py` | DD multiplier + recovery FSM | `DrawdownControllerProtocol` |
| `GateKeeper` | `betting/gate_keeper.py` | Edge-based final filter | `GateKeeperProtocol` |
| `MetaSwitcher` | `betting/meta_switcher.py` | Regime -> strategy params mapping | `MetaSwitcherProtocol` |
| `WinStrategy` | `betting/win_strategy.py` | Win candidate generation (unused in backtest path) | `BetStrategyProtocol` |
| `RegimeDetector` | `models/regime_detector.py` | 3-state market regime classifier | Used directly |
| `WinSelectionGateModel` | `models/win_selection_gate.py` | OOF-learned score + threshold gate | Used via RacePredictor |
| `RobustConfidenceEstimator` | `models/robust_confidence_estimator.py` | Conformal prediction intervals | Used via RacePredictor |

### Critical Architecture Observation

The `BettingOrchestrator` is **not used in the backtest path**. The backtest flow goes directly through `BacktestEngine` -> `RacePredictor`. The `BettingOrchestrator` is designed for the live/paper-trading path (12-step flow with t-3min cancellation). This means:

- **Modifications for v1.3 must target `RacePredictor.select_bets()`** for backtest impact.
- `BettingOrchestrator.process_race()` can be updated later for live path parity.
- `WinStrategy` is also **not used in the backtest path** -- win candidates go through `RacePredictor.get_win_candidates()` directly.

---

## Target Features (v1.3) and Integration Points

### Feature 1: Conformal Confidence Filter

**What:** Exclude bets where `conformal_confidence_score` at alpha=0.1 lower bound is below a threshold.

**Integration point:** `RacePredictor.get_win_candidates()` and `RacePredictor.get_place_candidates()`.

**Current state:** `conformal_confidence_score` is already computed by `RobustConfidenceEstimator.predict_interval()` during `RacePredictor.predict()`. It is used as a tertiary sort key in `get_win_candidates()`. But it is **never used as a filter**.

**Approach:** Modify the candidate filter in `get_win_candidates()` and `get_place_candidates()` to add a conformal confidence floor.

**Component to modify:** `RacePredictor` (add filter condition) + `MetaSwitcher._default_params()` (add `min_conformal_confidence` parameter per regime).

**No new components needed.**

### Feature 2: Odds Band ROI Filter

**What:** Exclude bets in odds bands with historically negative ROI.

**Integration point:** Same candidate filtering in `RacePredictor.get_win_candidates()` / `RacePredictor.get_place_candidates()`.

**Current state:** `WinSelectionGateModel` already builds odds-bucket score tables (quantile-binned) with smoothed ROI estimates per (prob_bin, edge_bin, odds_bin) combination. The gate model's `score()` method returns a composite score, and `min_prob`, `min_edge`, `max_odds` thresholds are OOF-optimized. However, the **explicit "exclude this odds band entirely" logic** is not present -- it is implicit in the gate threshold.

**Approach:** Two options:
- (A) Add an explicit `odds_band_exclusion_ranges` parameter to `RacePredictor` that hard-excludes specific odds ranges. Simple, transparent, backtestable.
- (B) Rely on `WinSelectionGateModel` threshold optimization to achieve the same effect implicitly.

**Recommendation:** Option A is safer and more auditable. The gate model's threshold is a global cutoff, but we may want to exclude specific mid-range bands (e.g., odds 5-7 with negative ROI) while keeping higher odds that have positive ROI. This is not achievable with a single `max_odds` threshold.

**New component needed:** `OddsBandFilter` -- a lightweight filter that takes learned exclusion ranges and applies them during candidate selection.

### Feature 3: Regime-Dependent Bet Toggle

**What:** In COLLAPSED regime, skip all bets (not just raise thresholds).

**Integration point:** `BacktestEngine.run()` per-race loop, after `RegimeDetector.detect()`.

**Current state:** COLLAPSED regime already sets very high thresholds (`ev_threshold=1.55`, `edge_threshold=0.10`, `max_bets_per_race=1`). This effectively stops most betting, but not all -- a horse with edge >= 0.10 and EV >= 1.55 could still get through.

**Approach:** Add a `betting_enabled` boolean to `MetaSwitcher._default_params()` per regime. When `False`, skip the entire candidate selection and return empty bets.

**Component to modify:** `MetaSwitcher._default_params()` (add `betting_enabled` field), `BacktestEngine.run()` (check `betting_enabled` before calling `get_win_candidates`).

**No new components needed.**

### Feature 4: Kelly Criterion Sizing (Enhanced)

**What:** The standard Kelly formula `f* = p - (1-p)/(odds-1)` with proper probability input.

**Integration point:** `StakeCalculator.calc_stake()` and `RacePredictor.select_bets()`.

**Current state:** `StakeCalculator` already implements a value-betting Kelly variant: `kelly_fraction = edge / (odds - 1)`, with `FRACTIONAL_KELLY = 0.5` (half-Kelly) and `KELLY_FRACTION_CAP = 0.25`. The edge is `p_model * odds - 1`, so `edge / (odds - 1) = (p*odds - 1)/(odds - 1)` which IS mathematically equivalent to the standard Kelly formula.

**The Kelly criterion is already implemented correctly.** The formula `f* = p - (1-p)/(odds-1)` simplifies to `(p*odds - 1)/(odds - 1)` which is exactly what `edge / (odds - 1)` computes.

**What might need tuning:**
- The `FRACTIONAL_KELLY = 0.5` and `KELLY_FRACTION_CAP = 0.25` are conservative. These could be regime-dependent.
- The `RACE_EXPOSURE_CAP = 0.02` (2% per race) is very tight. For win bets with high edge, this may be too conservative.

**Component to modify:** `StakeCalculator` (make fractional Kelly and exposure cap configurable per regime). `MetaSwitcher._default_params()` (add `fractional_kelly`, `race_exposure_cap` per regime).

**No new components needed.** This is parameter tuning, not architectural change.

### Feature 5: EV-Proportional Sizing

**What:** Scale stake proportionally to expected value (higher EV = larger stake).

**Integration point:** Between `StakeCalculator.calc_stake()` (base Kelly stake) and `DrawdownController.adjust_stake()` (DD adjustment).

**Current state:** The current pipeline is:
```
base_stake = StakeCalculator.calc_stake(edge, odds, bankroll, bet_type)
final_stake = DDController.adjust_stake(base_stake, bankroll)
```

Kelly sizing already scales with edge (higher edge -> larger Kelly fraction). But EV-proportional is a separate concept: it scales the FINAL stake by a factor proportional to `(ev - 1.0) / max_ev_range`. This allows differentiation among bets that all pass the Kelly threshold -- a bet with EV=1.8 gets a larger EV-multiplier than one with EV=1.2.

**Approach:** Add an EV-scaling step between Kelly and DD:
```
kelly_stake = StakeCalculator.calc_stake(edge, odds, bankroll, bet_type)
ev_scaled_stake = EVScaler.scale(kelly_stake, ev, min_ev, max_ev)
final_stake = DDController.adjust_stake(ev_scaled_stake, bankroll)
```

**New component needed:** `EVScaler` -- simple proportional scaler that maps EV to [ev_scale_min, ev_scale_max] range. This could be a method on `StakeCalculator` rather than a new class, but extracting it keeps responsibilities clean.

**Recommendation:** Add as a method on `StakeCalculator` (`apply_ev_scaling`) rather than a separate class. It is a simple linear mapping, not complex enough for its own class.

### Feature 6: Dynamic Drawdown Control (Enhanced)

**What:** More responsive DD control with bankroll-varying risk.

**Integration point:** `DrawdownController.adjust_stake()`.

**Current state:** `DrawdownController` already implements:
- DD% x ROI multiplier table (8 entries)
- 3-state recovery FSM (NORMAL -> REDUCED -> RECOVERING -> NORMAL)
- EWMA + SMA hybrid rolling ROI
- N-bet rate limiter (max 0.15 change per 20 bets)
- Hysteresis transitions

This is already sophisticated. What the v1.3 feature requests is making it **dynamic** -- responding to bankroll changes more smoothly and adapting risk parameters.

**Specific enhancements:**
1. Regime-dependent DD thresholds (aggressive regime tolerates deeper DD)
2. Bankroll-proportional base unit (as bankroll grows, base unit scales)
3. Sharpe-adjusted multiplier (factor in risk-adjusted return, not just raw ROI)

**Component to modify:** `DrawdownController` (add regime-aware multiplier table selection, bankroll-proportional base unit).

**No new components needed.** This is enhancement of an existing component.

---

## Recommended New Components

Only one truly new component is needed:

### OddsBandFilter

```python
# betting/odds_band_filter.py

class OddsBandFilter:
    """Exclude specific odds bands with historically negative ROI."""

    def __init__(self, exclusion_ranges: list[tuple[float, float]] | None = None) -> None:
        self.exclusion_ranges = exclusion_ranges or []

    def filter_candidates(
        self,
        candidates: pd.DataFrame,
        odds_col: str = "tanodds",
    ) -> pd.DataFrame:
        """Remove candidates whose odds fall within exclusion ranges."""
        if not self.exclusion_ranges or candidates.empty:
            return candidates

        odds = pd.to_numeric(candidates[odds_col], errors="coerce")
        mask = pd.Series(True, index=candidates.index, dtype=bool)
        for low, high in self.exclusion_ranges:
            mask &= ~(odds.between(low, high))
        return candidates.loc[mask].copy()

    @classmethod
    def from_backtest_analysis(
        cls,
        bet_history: list[dict],
        n_bins: int = 10,
        min_samples: int = 50,
        roi_threshold: float = 0.95,
    ) -> OddsBandFilter:
        """Learn exclusion ranges from backtest bet history."""
        # Bin by odds, compute per-band ROI, exclude bands below threshold
        ...
```

---

## Modified Components Summary

| Component | Change Type | What Changes |
|-----------|-------------|-------------|
| `RacePredictor.get_win_candidates()` | Filter addition | Add conformal confidence floor + odds band exclusion |
| `RacePredictor.get_place_candidates()` | Filter addition | Add conformal confidence floor + odds band exclusion |
| `RacePredictor.select_bets()` | Sizing pipeline | Insert EV-proportional scaling step |
| `StakeCalculator` | Method addition | Add `apply_ev_scaling()` method; make fractional Kelly configurable |
| `DrawdownController` | Enhancement | Regime-aware multiplier tables; bankroll-proportional base unit |
| `MetaSwitcher._default_params()` | Parameter expansion | Add `betting_enabled`, `min_conformal_confidence`, `fractional_kelly`, `race_exposure_cap` per regime |
| `BacktestEngine.run()` | Guard addition | Check `betting_enabled` before candidate selection |
| `BacktestEngine.run()` | Wiring | Pass `OddsBandFilter` instance to `RacePredictor` |
| `RacePredictor.__init__()` | Dependency addition | Accept optional `OddsBandFilter` |

---

## Data Flow: After v1.3 Changes

```
DataRepository -> FeatureEngine -> BacktestEngine.run() [per race]
    |
    +---> RacePredictor.predict()
    |         (unchanged: stage1 -> market -> win -> EV correction -> benter -> gate -> conformal)
    |         Output: result_df with conformal_confidence_score, win_selection_edge, etc.
    |
    +---> RegimeDetector.detect() -> regime_params
    |         NEW: regime_params includes betting_enabled, min_conformal_confidence,
    |              fractional_kelly, race_exposure_cap
    |
    +---> [GUARD] if not regime_params["betting_enabled"]: skip race  <-- NEW
    |
    +---> RacePredictor.get_win_candidates()
    |         NEW filters applied:
    |         1. conformal_confidence_score >= min_conformal_confidence  <-- NEW
    |         2. OddsBandFilter.filter_candidates()                      <-- NEW
    |         3. (existing: edge > 0, odds >= 1.0)
    |
    +---> RacePredictor.select_bets(candidates)
    |         Per candidate:
    |         1. kelly_stake = StakeCalculator.calc_stake(edge, odds, bankroll)
    |            (uses regime-dependent fractional_kelly)               <-- ENHANCED
    |         2. ev_scaled = StakeCalculator.apply_ev_scaling(kelly_stake, ev) <-- NEW
    |         3. final_stake = DDController.adjust_stake(ev_scaled, bankroll)
    |            (uses regime-aware multiplier table)                   <-- ENHANCED
    |         4. Race exposure cap (regime-dependent cap)              <-- ENHANCED
    |
    +---> _settle_bet() (unchanged)
    +---> DDController.update() (unchanged feedback loop)
```

---

## Build Order (Dependency-Driven)

### Phase 1: Filter Enhancements (No sizing changes, safe to test independently)

These changes affect WHICH bets are placed, not HOW MUCH is wagered. They are low-risk and independently testable.

1. **Regime-dependent bet toggle** (MetaSwitcher + BacktestEngine guard)
   - Modifies: `MetaSwitcher._default_params()`, `BacktestEngine.run()`
   - Depends on: Nothing new
   - Test: Verify COLLAPSED regime produces 0 bets

2. **Conformal confidence filter** (RacePredictor candidate methods)
   - Modifies: `RacePredictor.get_win_candidates()`, `RacePredictor.get_place_candidates()`
   - Depends on: `conformal_confidence_score` already in result_df
   - Test: Verify low-confidence candidates are excluded

3. **OddsBandFilter** (new component + wiring)
   - Creates: `betting/odds_band_filter.py`
   - Modifies: `RacePredictor.__init__()` (accept filter), `RacePredictor.get_win_candidates()`
   - Depends on: Nothing new
   - Test: Verify excluded odds ranges produce no bets

**Milestone checkpoint:** Run backtest after Phase 1. Measure bet count reduction and ROI impact.

### Phase 2: Sizing Enhancements (Changes HOW MUCH is wagered)

These changes affect stake amounts. They should be built after filters are stable, because sizing changes are meaningless if we are betting on the wrong horses.

4. **Configurable Kelly parameters** (StakeCalculator + MetaSwitcher)
   - Modifies: `StakeCalculator` (accept config params), `MetaSwitcher._default_params()`
   - Depends on: Phase 1 complete (correct bet pool)
   - Test: Verify regime-dependent Kelly fractions produce expected stake ranges

5. **EV-proportional scaling** (StakeCalculator new method + RacePredictor wiring)
   - Modifies: `StakeCalculator.apply_ev_scaling()`, `RacePredictor.select_bets()`
   - Depends on: Step 4 (Kelly params must be correct first)
   - Test: Verify high-EV bets get proportionally larger stakes

6. **Enhanced DD control** (DrawdownController)
   - Modifies: `DrawdownController` (regime-aware tables, bankroll-proportional unit)
   - Depends on: Steps 4-5 (sizing pipeline must be correct first)
   - Test: Verify DD multiplier changes with regime transitions

**Milestone checkpoint:** Run full backtest. Compare ROI with Phase 1 baseline.

### Phase 3: Threshold Tuning (Grid search for optimal parameters)

7. **Parameter sweep** -- Run backtests across parameter grid:
   - `min_conformal_confidence`: [0.0, 0.3, 0.5, 0.7]
   - `fractional_kelly`: [0.25, 0.375, 0.5, 0.625]
   - `race_exposure_cap`: [0.02, 0.03, 0.04]
   - `odds_band_exclusion_ranges`: [from ROI analysis]
   - DD multiplier adjustments

**This is search, not code.** But the architecture must support parameter injection via `MetaSwitcher`.

---

## Key Design Decisions

### Decision 1: Modify RacePredictor, Not BettingOrchestrator

The backtest path goes through `RacePredictor`, not `BettingOrchestrator`. All v1.3 changes target the backtest ROI metric. The `BettingOrchestrator` path (live/paper trading) can be updated later by mirroring the same filters and sizing logic.

### Decision 2: OddsBandFilter as Separate Component, Not Inside Gate Model

The `WinSelectionGateModel` already has odds-bucket scoring. But it uses quantile bins internally and the exclusion is implicit (via threshold). An explicit `OddsBandFilter` gives:
- Transparency: "we exclude odds 5-7" is auditable
- Backtestability: easy to measure impact of specific exclusions
- Separation of concerns: gate model scores, filter excludes

### Decision 3: EV-Scaling as StakeCalculator Method, Not Separate Class

EV-proportional scaling is a simple linear mapping: `stake * (ev - min_ev) / (max_ev - min_ev)`. This is 5 lines of code. A separate class would be over-engineering. Adding it as a method on `StakeCalculator` keeps the sizing pipeline in one place.

### Decision 4: Keep DDController as Single Component (Enhance, Don't Replace)

The existing `DrawdownController` is well-tested with a sophisticated recovery FSM. Enhancing it with regime-dependent tables is safer than replacing it with a new component. The risk of introducing bugs in DD control is high -- the current implementation handles edge cases (deadlock prevention at mult=0.05, rate limiting, hysteresis).

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Modifying BettingOrchestrator First

The `BettingOrchestrator` is for the live path and is not exercised by backtests. Changes there would be untested. Always implement and validate in the `RacePredictor` path first.

### Anti-Pattern 2: Parameter Hardcoding

All new thresholds must be in `MetaSwitcher._default_params()` (regime-dependent) or configurable via `BacktestEngine.__init__()`. Hardcoded constants in `RacePredictor` would prevent grid search.

### Anti-Pattern 3: Overlapping Filters

The `WinSelectionGateModel` already filters by prob/edge/odds thresholds. Adding conformal confidence and odds band filters must not duplicate what the gate model already does. The conformal filter should be applied BEFORE gate scoring (pre-filter), while odds band exclusion should be AFTER candidate selection (post-filter).

### Anti-Pattern 4: Sizing Before Filtering

Changing stake sizes before fixing the bet pool is premature optimization. A 200-yen bet on a losing horse loses more than a 100-yen bet on the same horse. Fix selection first, then optimize sizing.

---

## Scalability Considerations

| Concern | Current Scale | v1.3 Impact | Mitigation |
|---------|--------------|-------------|------------|
| Filter complexity | O(n) per race (n=18 horses max) | +2 O(n) passes (conformal, odds band) | Negligible. 18 horses x 2 passes = constant time |
| OddsBandFilter memory | None | One list of (float, float) tuples | Negligible. <100 ranges max |
| Parameter grid search | N/A | Backtest runs per parameter combo | ~200 combos x 57 min/backtest = significant. Use coarse grid first, then refine |
| DD state complexity | 3-state FSM | Same 3 states, regime-dependent tables | No complexity increase |

---

## Testing Strategy

### Unit Tests (All mock-based, no DB required)

For each modified component:

1. **OddsBandFilter**: Test with various exclusion ranges, edge cases (empty, overlapping, single range)
2. **StakeCalculator.apply_ev_scaling()**: Test EV mapping boundaries (min EV, max EV, out of range)
3. **RacePredictor.get_win_candidates()**: Test conformal confidence filter, verify candidates excluded below threshold
4. **DrawdownController**: Test regime-dependent multiplier selection
5. **MetaSwitcher._default_params()**: Verify `betting_enabled=False` for COLLAPSED, `min_conformal_confidence` values

### Integration Test (Full backtest)

Run `scripts/run_backtest.py --years 2024` with:
- Baseline (current params) -> record ROI, bet count, max DD
- Each new filter enabled individually -> measure delta
- All filters + sizing enabled -> compare to baseline
- Parameter sweep -> find optimal combination

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/betting/odds_band_filter.py` | Odds band exclusion filter |

### Modified Files

| File | Changes |
|------|---------|
| `src/betting/stake_calculator.py` | Add `apply_ev_scaling()` method; accept fractional_kelly/race_exposure_cap as init params |
| `src/betting/drawdown_controller.py` | Add regime-aware multiplier table selection; bankroll-proportional base unit |
| `src/betting/meta_switcher.py` | Add `betting_enabled`, `min_conformal_confidence`, `fractional_kelly`, `race_exposure_cap` to params |
| `src/betting/__init__.py` | Export `OddsBandFilter` |
| `src/backtest/race_predictor.py` | Accept `OddsBandFilter`; add conformal filter in `get_win_candidates()` / `get_place_candidates()`; wire EV scaling in `select_bets()` |
| `src/backtest/engine.py` | Add `betting_enabled` guard; pass `OddsBandFilter` to `RacePredictor` |
| `tests/test_odds_band_filter.py` | New test file |
| `tests/test_stake_calculator.py` | New test for `apply_ev_scaling()` |

---

## Sources

- Direct code analysis of `src/betting/*.py`, `src/backtest/race_predictor.py`, `src/backtest/engine.py`, `src/models/win_selection_gate.py`, `src/models/regime_detector.py`, `src/models/robust_confidence_estimator.py`
- Kelly criterion equivalence: `edge/(odds-1) = (p*odds-1)/(odds-1) = p - (1-p)/(odds-1)` -- standard identity
- Confidence level: HIGH (all analysis based on direct code reading, no external dependencies)
