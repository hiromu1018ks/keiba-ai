# Feature Landscape: v1.3 Betting Strategy Optimization

**Domain:** Horse racing win-betting -- stake sizing and bet selection optimization to push backtest ROI from 91.6% to 100%+
**Researched:** 2026-05-04
**Confidence:** HIGH (codebase audit + professional betting strategy literature)

## Context

This document covers ONLY features for the v1.3 milestone. The system already has working prediction models (3-model GBM stacking), conformal confidence scores, WinSelectionGate with edge-based filtering, RegimeDetector (aggressive/conservative/collapsed), WinStrategy with basic Kelly staking, StakeCalculator with half-Kelly + 2% race exposure cap, DrawdownController with DD x ROI multiplier table and 3-state recovery logic, and BacktestEngine with vectorized operations. The current ROI is 91.6% with 9,074 bets over a 2024 test year.

The six target features are:
1. Conformal confidence interval filter (alpha=0.1 lower bound)
2. Odds band ROI-based exclusion of unprofitable bands
3. RegimeDetector state-based bet on/off switching
4. Kelly criterion optimal stake sizing
5. EV-proportional stake sizing
6. Dynamic drawdown control

These fall into three capability areas:
- **Bet Selection** (features 1-3): Which bets to take
- **Stake Sizing** (features 4-5): How much to bet
- **Risk Control** (feature 6): When to reduce exposure

---

## Table Stakes

Features users expect in any professional betting strategy system. Missing any of these means the system cannot be considered a serious betting strategy optimizer.

### A. Bet Selection (Which Bets to Take)

| Feature | Why Expected | Complexity | Current Status | Change Required |
|---------|--------------|------------|----------------|-----------------|
| **Conformal confidence interval filter** | Any system that produces EV estimates must validate that those estimates are reliable before committing capital. The `RobustConfidenceEstimator` already computes `EV_lower_win_corrected` at alpha=0.1, but the `WinSelectionGate` and `select_bets()` do not use it as a hard filter. Without this, bets with low-confidence EV estimates (where the 90% CI lower bound is below 1.0) pass through, creating negative-EV bets. | Low | `RobustConfidenceEstimator` produces `EV_lower_win_corrected` and `conformal_confidence_score`. `WinSelectionGate` uses `win_selection_ev` which already incorporates the lower bound via `build_win_selection_ev()`. But the backtest `select_bets()` does not filter on a minimum confidence threshold. | Add a `confidence_threshold` parameter to `select_bets()` (default 0.0 for backward compat). When >0, filter out candidates where `conformal_confidence_score < threshold`. Also add a `min_ev_lower` filter: reject candidates where `EV_lower_win_corrected < 1.0` (i.e., the 90% CI lower bound does not clear breakeven). |
| **Odds band ROI-based exclusion** | Professional betting systems always analyze performance by odds range because edge is not uniformly distributed. The system already computes popularity bands (1-3, 4-6, 7+) and odds multiplier bands (1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+) in the report. But these are diagnostic-only; no band is actually excluded during betting. JRA parimutuel has ~25% takeout, so certain odds ranges may be structurally unprofitable. | Low | `BacktestReportGenerator` computes odds band ROI in `_band_stats()`. `WinSelectionGate` has a `max_odds` threshold learned from OOF data. But no explicit "exclude bands with negative ROI" logic exists during bet selection. | After the initial backtest run, compute ROI by odds band. Then in subsequent runs, exclude bands with ROI < threshold (e.g., < 0.80). This is a two-pass approach: (1) run backtest to identify bad bands, (2) re-run with exclusions. Alternatively, integrate into `WinSelectionGate.train()` which already optimizes `min_prob`, `min_edge`, `max_odds` via walk-forward grid search. |
| **RegimeDetector state-based bet on/off** | A system that detects market regimes but does not act on them is incomplete. The `RegimeDetector` already classifies 3 states and `get_strategy_params()` returns different `ev_threshold`, `edge_threshold`, and `max_bets_per_race` per state. In COLLAPSED mode, `ev_threshold=1.50` and `edge_threshold=0.09` (extremely strict). But the backtest engine does not skip bets entirely in COLLAPSED state -- it just uses stricter thresholds. For the 91.6% ROI problem, skipping entirely during COLLAPSED regimes is likely the single highest-impact change. | Low | `RegimeDetector.get_strategy_params()` returns 3 different parameter sets. `BacktestEngine` applies `edge_threshold` from regime params. But never returns 0 bets regardless of regime. | Add a `skip_collapsed` flag (default True for win mode). When regime is COLLAPSED and `skip_collapsed=True`, `select_bets()` returns empty list. This removes the weakest bets (those that only pass during high-threshold states) and should improve ROI at the cost of fewer total bets. Also consider making `AGGRESSIVE` state increase max_bets to 2 (currently 1 for all states). |

### B. Stake Sizing (How Much to Bet)

| Feature | Why Expected | Complexity | Current Status | Change Required |
|---------|--------------|------------|----------------|-----------------|
| **Fractional Kelly criterion** | Kelly criterion is the mathematical optimum for long-term bankroll growth. Any system without it is leaving money on the table. The `StakeCalculator` already implements half-Kelly (`FRACTIONAL_KELLY=0.5`) with edge-based formula: `kelly = edge / (odds - 1.0)`. However, the current implementation has issues: (1) `FRACTIONAL_KELLY` is hardcoded, not configurable per regime; (2) the effective cap (`KELLY_FRACTION_CAP=0.25 * FRACTIONAL_KELLY=0.5 = 0.125`) is very conservative; (3) there is no mechanism to use different fractions for different confidence levels. Professional systems typically use 0.25-0.50 Kelly to balance growth vs variance. | Low | `StakeCalculator.calc_stake()` implements `kelly = edge / (odds - 1.0) * FRACTIONAL_KELLY` with cap at 0.125 of bankroll. `WinStrategy._calc_stake()` has a simpler version with `KELLY_FRACTION_CAP=0.25`. Both produce stakes rounded to 100 yen. The backtest uses `StakeCalculator` when `betting_mode="kelly"`. | Make `FRACTIONAL_KELLY` configurable (not hardcoded). Allow regime-dependent fractions: AGGRESSIVE=0.50, CONSERVATIVE=0.25, COLLAPSED=0.0. Raise the effective cap from 0.125 to at least 0.15-0.20 for AGGRESSIVE regime. Add `max_kelly_fraction` to `get_strategy_params()` return dict. |
| **EV-proportional sizing** | Beyond Kelly (which is edge-proportional), EV-proportional sizing scales stake by the absolute EV value, not just the edge. This captures the insight that a bet with EV=1.30 deserves more capital than EV=1.10, even if both have the same edge percentage. The current system uses Kelly (edge-based) only. EV-proportional is a complementary dimension: Kelly handles the probability-odds relationship, EV-proportional handles the magnitude of the opportunity. | Medium | No EV-proportional sizing exists. `StakeCalculator` uses `edge / (odds - 1)` which is Kelly. `WinStrategy` uses `edge / (odds - 1)` which is also Kelly. Neither uses EV as a multiplier. | Implement as an optional scaling factor on top of Kelly: `final_stake = kelly_stake * ev_scale`, where `ev_scale = min(ev / target_ev, max_ev_scale)`. For example, if `target_ev=1.10` and `max_ev_scale=2.0`, a bet with EV=1.30 gets `1.30/1.10=1.18x` the base Kelly stake (capped at 2.0x). This rewards high-EV opportunities with more capital while keeping Kelly as the base. The parameters should be configurable and regime-dependent. |

### C. Risk Control (When to Reduce Exposure)

| Feature | Why Expected | Complexity | Current Status | Change Required |
|---------|--------------|------------|----------------|-----------------|
| **Dynamic drawdown control** | Any system that bets real money must have drawdown protection. The `DrawdownController` already implements DD x ROI multiplier table with 3-state recovery (NORMAL -> REDUCED -> RECOVERING -> NORMAL), EWMA hybrid rolling ROI, and N-bet adjustment rate limiting. This is already a sophisticated implementation. The gap is that the controller is not tuned for win betting: the multiplier table thresholds are designed for place betting (which has ~25% hit rate), not win betting (~7-10% hit rate). Win betting has much higher variance, so the DD controller needs different thresholds. | Medium | `DrawdownController` has a 7-row multiplier table keyed on DD percentage and rolling ROI. Recovery thresholds: ROI >= 0.98 and DD < 0.15 for REDUCED->RECOVERING, DD < 0.05 for RECOVERING->NORMAL. These are place-tuned. For win betting with ~7% hit rate, the rolling ROI is much noisier (individual wins create large spikes), and the DD can swing more violently. | Create a win-specific multiplier table with wider DD bands (e.g., 0-15% at 1.0, 15-25% at 0.70, 25-35% at 0.40, 35-50% at 0.15, 50%+ at 0.05). Adjust recovery thresholds: require longer recovery periods (more bets) before transitioning back to NORMAL. Also consider using `conformal_confidence_score` as an additional DD factor -- when confidence is low AND DD is rising, reduce stakes more aggressively. |

---

## Differentiators

Features that go beyond basic bet selection and sizing. These create genuine competitive advantage.

### HIGH IMPACT -- Adaptive Strategy Selection

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Regime-adaptive Kelly fraction** | Instead of a static fractional Kelly, dynamically adjust the Kelly fraction based on regime state AND recent performance. During AGGRESSIVE regime with good recent ROI, use 0.50 Kelly. During CONSERVATIVE regime with declining ROI, drop to 0.25 Kelly. During COLLAPSED, use 0.0 Kelly (skip all bets). This creates a continuously adapting risk posture rather than a fixed one. Professional syndicates like Benter's Hong Kong operation used exactly this approach. | Medium | RegimeDetector, StakeCalculator, DrawdownController | Requires adding `kelly_fraction` to `get_strategy_params()` output. The StakeCalculator needs to accept this as a parameter. The regime-Kelly mapping should be configurable in `config/settings.yaml`. |
| **Confidence-weighted EV filter** | Instead of a binary confidence threshold (pass/fail), use `conformal_confidence_score` as a continuous weight on EV. A bet with `ev_win=1.30` and `confidence=0.9` gets `weighted_ev = 1.30 * 0.9 = 1.17`. A bet with `ev_win=1.30` and `confidence=0.5` gets `weighted_ev = 1.30 * 0.5 = 0.65`. This rejects low-confidence high-EV bets (likely noise) while accepting high-confidence moderate-EV bets. This is more nuanced than a hard threshold and better accounts for the uncertainty in EV estimates. | Medium | RobustConfidenceEstimator, WinSelectionGate | Add `confidence_weighted_ev` column to prediction output. Use this instead of raw `win_selection_ev` in `select_bets()`. The weighting formula should be `ev * max(confidence, confidence_floor)` to prevent zeroing out bets with moderate confidence. |
| **Two-pass odds band exclusion with cross-validation** | Instead of a simple single-pass "exclude bands with ROI < X", use walk-forward cross-validation to determine which odds bands to exclude. Train on years 1-3, identify bad bands on year 4, validate on year 5. This prevents overfitting to a specific test period. The existing `WinSelectionGate` walk-forward framework can be extended to include odds band as a parameter in the grid search. | Medium | WinSelectionGate.walk_forward grid search, backtest report | The `WinSelectionGate._build_threshold_grid()` already includes odds thresholds (4, 6, 8, 10, 12, 15, 18 + quantile-based). Extending to explicit band exclusion is straightforward: add a "banned_bands" list parameter that skips candidates in excluded ranges. |

### MEDIUM IMPACT -- Enhanced Risk Control

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Bankroll-relative stop-loss** | Stop all betting for the day/week when cumulative loss exceeds a threshold (e.g., 5% of starting bankroll per day, 10% per week). The `SafetyGuard` already has `max_daily_loss=10000` and `max_weekly_loss=30000` but these are absolute yen amounts, not bankroll-relative. For a system where bankroll grows/shrinks over time, percentage-based limits are more appropriate. | Low | SafetyGuard (already exists) | Change SafetyGuard thresholds from absolute to percentage-based. Add `max_daily_loss_pct=0.05` and `max_weekly_loss_pct=0.10` parameters. |
| **Correlation-aware race exposure** | The current 2% race exposure cap treats all bets in a race as independent. In reality, bets on the same race are highly correlated -- if one horse loses, others likely win. For win bets, this correlation is perfect negative (only one horse can win). Adjust the exposure cap to account for the fact that multiple win bets in the same race are anti-correlated diversification, not independent risks. | Medium | StakeCalculator.check_race_exposure | The 2% cap is appropriate for a single win bet per race (max_bets=1). If max_bets is relaxed to 2 in AGGRESSIVE regime, the per-race cap should be raised proportionally (e.g., 3-4% for 2 bets). |
| **EV-smoothing over rolling window** | Instead of using the instantaneous EV estimate for each bet, use a rolling average of EV estimates over the last N bets for sizing. This reduces the impact of any single noisy EV estimate and creates smoother stake progression. | Low | StakeCalculator, bet_history | Implement as an optional EV smoother in StakeCalculator: `smoothed_ev = (1-alpha) * previous_ev + alpha * current_ev`. Use `alpha=0.3` for moderate smoothing. |

### LOW IMPACT -- Nice to Have

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Bet history-based threshold adaptation** | After every N bets, re-evaluate filter thresholds based on recent bet history. If the last 100 bets have ROI < 0.90, tighten the EV threshold by 0.05. If ROI > 1.10, loosen by 0.05. This creates a self-tuning system. | Medium | BacktestEngine, RegimeDetector | Risk of overfitting to recent data. The RegimeDetector already provides regime-level adaptation. Adding bet-level adaptation may create conflicting signals. |
| **Surface/condition-specific thresholds** | Different thresholds for turf/dirt, good/soft track conditions. The `RobustConfidenceEstimator` already has surface+distance-conditioned CP quantiles, but the bet selection and sizing do not differentiate. | Low | WinSelectionGate, config | Add surface/condition dimensions to `get_strategy_params()` or use separate thresholds per surface in `select_bets()`. |
| **Stake volatility target** | Instead of Kelly-based sizing (which can produce wildly varying stakes), target a constant stake volatility. If recent stakes have been too volatile (std/mean > threshold), scale down all stakes to reduce volatility. | Medium | StakeCalculator, DrawdownController | Adds complexity for marginal benefit. The DD controller already reduces stakes during drawdowns, which naturally reduces volatility. |

---

## Anti-Features

Features to explicitly NOT build for this milestone.

### Not In Scope

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Complex ML-based stake sizing** (neural network or gradient boosting model that predicts optimal stake from features) | Massive overfitting risk. With only ~9K bets, training a model to predict optimal stake size will memorize noise. Kelly criterion is provably optimal under the assumptions, and fractional Kelly is the standard professional approach. Adding ML on top adds complexity without theoretical justification. | Use fractional Kelly with regime-dependent fractions. This is provably near-optimal and well-understood. |
| **Monte Carlo simulation for stake optimization** | Running Monte Carlo simulations to find the "optimal" stake sizing for each bet based on simulated outcomes adds enormous computational cost and provides only marginal improvement over analytical Kelly. The backtest itself is already a simulation. Adding another simulation layer is wasteful. | Use the analytical Kelly formula with configurable fractions. Validate via backtest. |
| **Real-time odds re-evaluation during backtest** | The backtest uses a single odds snapshot per bet. Adding real-time odds re-evaluation (e.g., re-checking edge at t-5, t-3, t-1 before "placing" the bet) would require significant changes to the backtest engine and odds data pipeline. This is a production feature, not a strategy optimization feature. | Use the existing pre-race odds snapshot for backtest decisions. Late money filtering is already implemented in the orchestrator. |
| **Multi-race parlay/combination betting** | PROJECT.md explicitly states focus is on single-race win betting. Parlay betting (combining bets across races) is a fundamentally different optimization problem requiring joint probability estimation across races. | Keep single-race win betting focus. Cross-race optimization is for a future milestone. |
| **Optuna/grid-search for filter threshold optimization** | While threshold tuning is important, using Optuna or exhaustive grid search to find "optimal" thresholds over the test period is textbook overfitting. The test data must remain unseen for honest evaluation. | Use the existing `WinSelectionGate` walk-forward threshold search (which trains on past, evaluates on future). For new thresholds, use the same walk-forward approach, never optimize on test data. |
| **Changes to the prediction model itself** | The 91.6% ROI is produced by the current 3-model stacking + conformal prediction pipeline. Changing the model architecture during strategy optimization makes it impossible to isolate whether ROI improvements came from better strategy or better predictions. | Lock the model architecture. Only change bet selection, stake sizing, and risk control parameters. |
| **Arbitrage or hedging strategies** | JRA parimutuel pools do not support arbitrage in the traditional sense (all odds are derived from the same pool). Hedging across bet types (win/place/wide) is theoretically possible but adds complexity without clear ROI benefit at current edge levels. | Focus on win-only strategy optimization. |
| **Dynamic bankroll reset during backtest** | Some backtest systems reset the bankroll to initial when it drops below a threshold. This hides drawdown and overstates ROI. The current system correctly tracks continuous bankroll. | Keep the existing continuous bankroll tracking. Never reset during a backtest run. |

---

## Feature Dependencies

```
Bet Selection Filters (can be implemented independently of each other):
  confidence_filter: depends on RobustConfidenceEstimator (DONE)
                     depends on EV_lower_win_corrected column (DONE)
                     depends on conformal_confidence_score column (DONE)
                     NEW: add min_ev_lower threshold to select_bets()
                     NEW: add confidence_threshold to select_bets()

  odds_band_exclusion: depends on backtest report odds_band stats (DONE)
                       depends on WinSelectionGate max_odds (DONE)
                       NEW: add excluded_odds_bands parameter to select_bets()
                       ALTERNATIVE: extend WinSelectionGate threshold grid

  regime_skip: depends on RegimeDetector.detect() (DONE)
               depends on get_strategy_params() (DONE)
               NEW: add skip_collapsed logic in BacktestEngine loop

Stake Sizing (builds on existing StakeCalculator):
  kelly_fraction_config: depends on StakeCalculator (DONE)
                         depends on RegimeDetector params (DONE)
                         NEW: make FRACTIONAL_KELLY configurable per regime
                         NEW: add kelly_fraction to get_strategy_params()

  ev_proportional: depends on StakeCalculator (DONE)
                   depends on ev_win_corrected column (DONE)
                   NEW: add ev_scale multiplier in calc_stake()
                   NEW: add ev_proportional params to get_strategy_params()

Risk Control (builds on existing DrawdownController):
  dynamic_dd: depends on DrawdownController (DONE)
              depends on StakeCalculator (DONE)
              NEW: create win-specific multiplier table
              NEW: tune recovery thresholds for win betting

Integration order:
  1. regime_skip (standalone, highest impact, lowest risk)
  2. confidence_filter (standalone, medium impact)
  3. kelly_fraction_config (builds on existing StakeCalculator)
  4. ev_proportional (builds on kelly_fraction_config)
  5. odds_band_exclusion (needs baseline backtest to identify bands)
  6. dynamic_dd (needs win-specific tuning data)
```

## Dependency on Existing Code

| Existing Component | v1.3 Feature That Uses It | Nature of Dependency |
|--------------------|--------------------------|---------------------|
| `StakeCalculator` | Kelly fraction config, EV-proportional | Modify `calc_stake()` to accept configurable `fractional_kelly` and `ev_scale` parameters |
| `DrawdownController` | Dynamic DD control | Add win-specific multiplier table; adjust recovery thresholds |
| `RegimeDetector.get_strategy_params()` | Regime skip, Kelly fraction config | Add `kelly_fraction` and `skip_betting` fields to returned dict |
| `WinSelectionGate` | Confidence filter, odds band exclusion | Gate already filters on prob/edge/odds. Extend with confidence and explicit band exclusion |
| `RobustConfidenceEstimator` | Confidence filter | Already produces `EV_lower_win_corrected` and `conformal_confidence_score`. Just wire into selection |
| `BacktestEngine.select_bets()` path | All features | The `select_bets()` call in the race loop is where all filtering and sizing decisions are made |
| `config/settings.yaml` | All features | Add new configurable parameters for thresholds, fractions, and DD table |
| `BacktestReportGenerator` | Odds band exclusion | Already computes band stats. Feed results back into exclusion logic |

## MVP Recommendation

### Phase 1: Quick Wins -- Bet Selection (Highest Impact, Lowest Complexity)

These three changes can be implemented and tested independently. Each should move ROI by 2-4 points.

1. **Regime COLLAPSED skip** -- When `RegimeDetector.current_regime == COLLAPSED`, skip the race entirely. This is 3 lines of code in the backtest engine's race loop and should eliminate the weakest bets that drag down ROI. The COLLAPSED regime's `ev_threshold=1.50` already filters most bets, but skipping entirely removes any that squeak through.

2. **Conformal EV lower bound filter** -- In `select_bets()` (or `get_win_candidates()`), add a filter: `EV_lower_win_corrected >= 1.0`. This rejects any bet where the 90% confidence interval for EV includes negative territory. The data is already computed by `RobustConfidenceEstimator`.

3. **Configurable Kelly fraction** -- Make `StakeCalculator.FRACTIONAL_KELLY` a parameter instead of a class constant. Add it to `get_strategy_params()` output so it varies by regime. AGGRESSIVE=0.50, CONSERVATIVE=0.25, COLLAPSED=0.0.

Estimated effort: 2 sessions.

### Phase 2: Enhanced Sizing (After Phase 1 Validates)

4. **EV-proportional scaling** -- Add an optional `ev_scale` multiplier on top of Kelly stake. `final_stake = kelly_stake * min(ev / 1.10, 2.0)`. This rewards high-EV opportunities with more capital.

5. **Odds band exclusion** -- After running the baseline backtest (Phase 1), analyze ROI by odds band from the report. Identify consistently negative-ROI bands. Add `excluded_odds_bands` parameter to `select_bets()` that rejects candidates in those ranges. Use walk-forward validation to prevent overfitting.

Estimated effort: 2 sessions.

### Phase 3: Risk Control Tuning (After Phase 2 Shows Positive ROI)

6. **Win-specific DD control** -- Create a separate multiplier table for win betting with wider DD bands and slower recovery. Tune recovery thresholds based on win betting variance characteristics.

7. **Confidence-weighted EV** -- Replace binary confidence filter with continuous weighting: `weighted_ev = ev * max(confidence, 0.3)`. Use this weighted EV for both selection and sizing.

Estimated effort: 1-2 sessions.

### Defer (Future Milestones)

- Bankroll-relative stop-loss (SafetyGuard percentage-based limits)
- Correlation-aware race exposure (only needed if max_bets > 1)
- Bet history-based threshold adaptation (overfitting risk)
- Surface/condition-specific thresholds (low impact at current sample size)
- Stake volatility target (complexity exceeds benefit)

---

## Key Design Decisions for This Milestone

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Kelly fraction: static or regime-dependent? | Regime-dependent | AGGRESSIVE regime has more opportunities and higher hit rates; should bet more. CONSERVATIVE has fewer opportunities; should bet less. COLLAPSED should not bet at all. Static fractions leave money on the table. |
| EV-proportional: separate from Kelly or integrated? | Integrated as a multiplier on Kelly stake | Kelly handles the probability-odds math optimally. EV-proportional adds a "conviction bonus" on top. Keeping them separate (e.g., choosing one or the other) loses information. |
| Confidence filter: binary threshold or continuous weight? | Start binary, upgrade to continuous later | Binary threshold (EV_lower >= 1.0) is simpler to implement, easier to reason about, and clearly correct. Continuous weighting adds nuance but also adds tuning surface. |
| Odds band exclusion: hardcoded or learned? | Learned via WinSelectionGate walk-forward | The existing walk-forward framework already searches over odds thresholds. Extending it to handle explicit band exclusion is natural. Hardcoded bands risk overfitting to the test period. |
| DD control: reuse place table or create win-specific table? | Create win-specific table | Win betting has ~7% hit rate vs ~25% for place. The variance profile is fundamentally different. Using place-tuned thresholds on win bets will either be too aggressive (accepting too much DD) or too conservative (cutting bets during normal variance). |
| Where to implement filters: in select_bets() or WinSelectionGate? | Both -- WinSelectionGate for learned thresholds, select_bets() for hard filters | WinSelectionGate is the right place for data-driven thresholds (learned from walk-forward). Hard filters (regime skip, min confidence) belong in the backtest engine's selection logic where they are transparent and debuggable. |

## Code Change Map

| File | Changes Required | LOC Affected (est.) |
|------|-----------------|---------------------|
| `src/betting/stake_calculator.py` | Make `FRACTIONAL_KELLY` a parameter; add `ev_scale` multiplier | ~25 lines |
| `src/betting/drawdown_controller.py` | Add win-specific multiplier table factory; adjust recovery params | ~40 lines |
| `src/models/regime_detector.py` | Add `kelly_fraction`, `skip_betting` to `get_strategy_params()` return | ~15 lines |
| `src/backtest/race_predictor.py` | Add confidence filter, odds band exclusion to `select_bets()` / `get_win_candidates()` | ~30 lines |
| `src/backtest/engine.py` | Add regime skip logic in race loop | ~10 lines |
| `config/settings.yaml` | Add betting_strategy section with configurable thresholds | ~20 lines |
| `tests/test_betting_strategy.py` | New test file for strategy parameter combinations | ~80 lines |

Total estimated new/modified code: ~220 lines.

---

## Sources

### Primary (HIGH confidence)
- Full codebase audit: `src/betting/stake_calculator.py` (123 LOC), `src/betting/drawdown_controller.py` (168 LOC), `src/betting/win_strategy.py` (78 LOC), `src/models/regime_detector.py` (240 LOC), `src/models/robust_confidence_estimator.py` (~200 LOC), `src/backtest/race_predictor.py` (~800 LOC), `src/models/win_selection_gate.py` (1113 LOC)
- PROJECT.md v1.3 milestone definition (6 target features specified)
- Kelly Criterion literature: Thorp (2006), Benter (1994) -- foundational for stake sizing design

### Secondary (MEDIUM confidence)
- [Pinnacle: Revisiting Kelly Criterion Part 2 - Fractional Kelly](https://www.pinnacle.com/betting-resources/en/betting-strategy/revisiting-the-kelly-criterion-part-2-fractional-kelly/gbd27z9nljvgflgg) -- half Kelly delivers ~75% of full Kelly growth with dramatically lower variance; quarter Kelly delivers ~50%
- [Reddit r/quant: Applying Kelly to Sports Betting 18-Month Backtest](https://www.reddit.com/r/quant/comments/1o2wzfh/applying_kelly_criterion_to_sports_betting_18/) -- real-world data showing 25% Kelly provides better risk-adjusted returns than full Kelly
- [Matthew Downey: Why Fractional Kelly?](https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html) -- fractional Kelly is optimal under probability estimation uncertainty
- [Benter 1994 via Medium: Computer Based Horse Race Handicapping](https://medium.com/parimutuel-racetrack-analysis/bill-benter-1994-computer-based-horse-race-handicapping-and-wagering-systems-a-report-db747c250e77) -- foundational model for regime-adaptive Kelly sizing
- [JournalPlus: Kelly Criterion](https://journalplus.co/metrics/kelly-criterion) -- full Kelly produces 50-80%+ drawdowns; half Kelly delivers 75% growth with much lower variance
- [PLOS ONE: Statistical Theory of Optimal Decision-Making in Sports Betting](https://pmc.ncbi.nlm.nih.gov/articles/PMC10306238/) -- confidence intervals for evaluating betting model performance

### Tertiary (LOW confidence, needs validation)
- Exact ROI impact of each feature on this specific dataset -- requires running backtests
- Optimal Kelly fractions for JRA win betting specifically -- theoretical guidance only
- Win betting hit rate variance profile on JRA data -- estimated at ~7-10% based on model characteristics

---
*Research completed: 2026-05-04*
*Ready for roadmap: yes*
