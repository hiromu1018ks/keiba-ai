# Project Research Summary

**Project:** keiba-ai v1.3 -- Betting Strategy Optimization
**Domain:** Horse racing ML prediction system -- bet selection and stake sizing optimization
**Researched:** 2026-05-04
**Confidence:** HIGH

## Executive Summary

keiba-ai v1.3 is not a greenfield project. It is a targeted optimization of an existing betting layer built on a mature ML pipeline (3-model GBM stacking, conformal prediction, regime detection, walk-forward gate model). The current system produces 89.0% ROI on 9,074 bets in the 2024 test year. The goal is to push past 100% ROI through better bet selection (which bets to take), stake sizing (how much to bet), and risk control (when to reduce exposure) -- without modifying the frozen ML model pipeline.

The research consensus across all four domains is clear: the existing codebase already implements the correct foundations. `StakeCalculator` has a mathematically correct Kelly formula. `DrawdownController` has a sophisticated 3-state recovery FSM. `RegimeDetector` classifies market states. `RobustConfidenceEstimator` produces conformal confidence intervals. The gaps are not missing algorithms -- they are missing wiring. The conformal confidence score is computed but never used as a filter. The regime detector raises thresholds in COLLAPSED mode but never fully skips betting. The Kelly fraction is hardcoded rather than regime-dependent. The DD controller is calibrated for place betting (30% hit rate) rather than win betting (7-10% hit rate). Zero new production dependencies are needed.

The primary risk is overfitting. The 9,074-bet test year provides limited statistical power, and the natural temptation to grid-search strategy parameters on backtest ROI will produce fictitious improvements. All four research files converge on the same mitigation: use walk-forward validation, freeze strategy parameters before out-of-sample evaluation, and treat parameter tuning with extreme conservatism. The recommended build order is filters-first (fix which bets to take), then sizing (optimize how much to bet), then parameter tuning (search for optimal thresholds on a held-out validation period).

## Key Findings

### Recommended Stack

No new production dependencies required. The existing numpy, pandas, optuna, and scipy (transitive via sklearn) stack covers every v1.3 feature. External Kelly criterion libraries (keeks, bet-optimizer, kelly-criterion) were evaluated and rejected: all provide trivial wrappers around arithmetic the codebase already implements, and none handle JRA-specific constraints (25% deduction rate, 100-yen minimum units, 2% race exposure cap).

**Core technologies:**
- **numpy >=1.26:** Kelly formula, EV-proportional scaling, drawdown calculations -- already installed and used in `StakeCalculator`
- **pandas >=2.2:** Multi-criteria boolean filtering for bet selection -- already used throughout
- **optuna >=3.5:** Threshold parameter search using TPE sampler -- already installed for model HP tuning, reusable for strategy parameters
- **itertools.product (stdlib):** Exhaustive grid enumeration for small parameter spaces -- already used in `training_pipeline.py`

### Expected Features

**Must have (table stakes):**
- Conformal confidence filter -- reject bets where `EV_lower_win_corrected < 1.0` (90% CI does not clear breakeven). Data already computed by `RobustConfidenceEstimator`.
- Regime COLLAPSED bet skip -- skip all betting in COLLAPSED regime rather than just raising thresholds. Likely the single highest-impact change.
- Configurable Kelly fraction -- make `FRACTIONAL_KELLY` a parameter instead of hardcoded constant, vary by regime (AGGRESSIVE=0.50, CONSERVATIVE=0.25, COLLAPSED=0.0).
- EV-proportional scaling -- optional multiplier on Kelly stake proportional to absolute EV. `final_stake = kelly_stake * min(ev / target_ev, max_scale)`.
- Odds band ROI exclusion -- learn negative-ROI odds bands from backtest history and exclude them during candidate selection.
- Win-specific DD control -- recalibrate `DrawdownController` for win betting's ~10% hit rate (vs current PLACE-tuned 30%).

**Should have (competitive):**
- Confidence-weighted EV -- replace binary confidence threshold with continuous weighting: `weighted_ev = ev * max(confidence, 0.3)`.
- Bankroll-relative stop-loss -- convert `SafetyGuard` from absolute yen limits to percentage-based limits.
- Correlation-aware race exposure -- adjust per-race cap when multiple win bets are placed (anti-correlated diversification).

**Defer (v2+):**
- ML-based stake sizing, Monte Carlo simulation, real-time odds re-evaluation, multi-race parlay betting -- all explicitly out of scope for v1.3.

### Architecture Approach

The backtest data flow goes through `BacktestEngine.run()` -> `RacePredictor` per race, not through `BettingOrchestrator` (which is for the live path). All v1.3 changes must target the `RacePredictor` path. The architecture is protocol-based with dependency injection: `StakeCalculatorProtocol`, `DrawdownControllerProtocol`, `GateKeeperProtocol`, `MetaSwitcherProtocol`. New components should follow the same pattern.

**Major components to modify:**
1. **RacePredictor** (`backtest/race_predictor.py`) -- add conformal confidence filter and odds band filter in `get_win_candidates()` / `get_place_candidates()`; wire EV scaling in `select_bets()`
2. **StakeCalculator** (`betting/stake_calculator.py`) -- make Kelly fraction configurable per regime; add `apply_ev_scaling()` method
3. **DrawdownController** (`betting/drawdown_controller.py`) -- add win-specific multiplier table; increase `ROLLING_WINDOW` from 150 to 400+; lower recovery thresholds
4. **MetaSwitcher** (`betting/meta_switcher.py`) -- add `betting_enabled`, `min_conformal_confidence`, `fractional_kelly`, `race_exposure_cap` to per-regime params

**One new component:**
5. **OddsBandFilter** (`betting/odds_band_filter.py`) -- lightweight filter that takes learned exclusion ranges and applies them during candidate selection

### Critical Pitfalls

1. **Kelly overbetting from overconfident edge estimates** -- ML models systematically overestimate probabilities for win bets (low base rate). Use half-Kelly or quarter-Kelly consistently. Never increase to full Kelly. Verify predicted vs actual win rate per decile on OOF data before trusting edge estimates.

2. **Look-ahead bias in parameter optimization** -- Grid-searching strategy parameters on the test year and reporting the best ROI is textbook overfitting. Use nested walk-forward: tune on first half of test period, validate on second half. Or reserve 2023 as validation, develop on 2024, report 2023 results.

3. **Regime detector state oscillation** -- The 5-race hysteresis counter allows ~1000 regime transitions per year, each changing bet selection. Increase hysteresis to 20-50 races. Count transitions after backtest; if >50, the detector is oscillating. Consider disabling regime switching for MVP and running in CONSERVATIVE-only mode.

4. **Odds band survivorship bias** -- High-odds bands have few bets, making ROI estimates unreliable. Require 200+ bets per band before excluding. Use shrinkage toward global mean. Validate on 2+ OOS periods. Do not exclude bands; downweight instead.

5. **DD controller feedback loop trapping** -- The 150-bet rolling window contains only ~15 wins at 10% hit rate, making ROI estimates too noisy. The 0.98 recovery threshold may never be reached, trapping the system in REDUCED state permanently. Increase window to 400-500 bets. Lower recovery threshold to 0.92-0.95.

6. **Filter cascade interaction** -- Multiple filters applied sequentially may remove 50%+ of bets, reducing statistical power below minimum. Test filters in combination, not individually. Fix filter order: regime -> gate model -> conformal confidence -> odds band -> Kelly -> DD. Set minimum bet count guard at 1,000/year.

## Implications for Roadmap

Based on combined research, the recommended phase structure follows the architectural build order from ARCHITECTURE.md: filters before sizing, sizing before tuning.

### Phase 1: Bet Selection Filters

**Rationale:** Changing which bets to take is lower risk than changing how much to bet. A bad filter removes bets (reducing volume); bad sizing loses money. Filters are independently testable. The research consensus is that COLLAPSED-regime skipping and conformal confidence filtering are likely the highest-impact, lowest-risk changes.
**Delivers:** Fewer but higher-quality bets. Measurable ROI improvement from bet pool refinement.
**Addresses:** Conformal confidence filter, regime COLLAPSED skip, odds band exclusion.
**Avoids:** Pitfall 1 (Kelly overbetting -- not touching sizing yet), Pitfall 2 (look-ahead bias -- filters use fixed thresholds from domain knowledge, not optimized), Pitfall 3 (regime oscillation -- adding skip logic is simpler than tuning regime detector), Pitfall 4 (odds band bias -- implementing minimum sample counts and shrinkage), Pitfall 6 (conformal precision -- using fixed alpha=0.1, not tuning it), Pitfall 8 (filter interaction -- testing combined filters against baseline).
**Estimated scope:** ~60-80 LOC new/modified. New file: `odds_band_filter.py`. Modifications: `race_predictor.py`, `engine.py`, `meta_switcher.py`.

### Phase 2: Stake Sizing Enhancement

**Rationale:** After the bet pool is refined, optimize stake amounts. Sizing changes depend on having a correct bet pool -- optimizing sizing on the wrong bets wastes effort. Kelly fraction configuration and EV-proportional scaling are the two sizing dimensions.
**Delivers:** Regime-dependent Kelly fractions (AGGRESSIVE bets more, CONSERVATIVE bets less). EV-proportional scaling that rewards high-conviction bets with more capital.
**Uses:** numpy for Kelly and EV arithmetic, existing `StakeCalculator` framework, `MetaSwitcher` params.
**Implements:** Configurable Kelly parameters, EV-proportional scaling as `StakeCalculator.apply_ev_scaling()`.
**Avoids:** Pitfall 1 (overbetting -- capping EV scaling at 2.0x, enforcing per-bet exposure cap), Pitfall 7 (tail risk -- log-EV scaling option, Herfindahl index monitoring).
**Estimated scope:** ~40-50 LOC modified. No new files. Modifications: `stake_calculator.py`, `race_predictor.py`, `meta_switcher.py`.

### Phase 3: Risk Control Calibration

**Rationale:** DD controller must be calibrated after sizing is correct, because DD reacts to absolute stake amounts. Win-specific calibration (wider DD bands, longer rolling window, lower recovery threshold) is the final structural change.
**Delivers:** Win-specific DD multiplier table, appropriate rolling window for 10% hit rate, calibrated recovery thresholds.
**Uses:** Existing `DrawdownController` framework, `config/settings.yaml` for thresholds.
**Implements:** Win-specific multiplier table, increased `ROLLING_WINDOW`, adjusted recovery thresholds.
**Avoids:** Pitfall 5 (DD trapping -- longer window, lower threshold, maximum REDUCED duration guard).
**Estimated scope:** ~40 LOC modified. No new files. Modifications: `drawdown_controller.py`, `config/settings.yaml`.

### Phase 4: Parameter Sweep and Validation

**Rationale:** Only after all structural changes are in place should parameter tuning begin. This phase is search, not code. It must use walk-forward validation on a held-out period to avoid look-ahead bias.
**Delivers:** Optimal parameter combination for production use. Quantified ROI improvement vs baseline.
**Uses:** optuna for TPE search, `itertools.product` for exhaustive grid on small spaces, `run_backtest.py` as evaluation harness.
**Avoids:** Pitfall 2 (look-ahead bias -- nested walk-forward, held-out validation year), all pitfalls verified through OOS testing.
**Estimated scope:** ~60 LOC for sweep script, ~80 LOC for test file. No production code changes.

### Phase Ordering Rationale

- Filters before sizing: Fixing bet selection first means sizing optimizations operate on a correct bet pool. The architectural analysis in ARCHITECTURE.md identifies this as "anti-pattern 4: sizing before filtering."
- Sizing before DD calibration: DD controller reacts to absolute stake amounts, so sizing must be finalized first.
- All structural changes before parameter tuning: Tuning before structural correctness gives false optima. The pitfalls research emphasizes that parameter optimization on incomplete implementations embeds structural deficiencies into "optimal" parameters.
- Config-driven thresholds throughout: Every new threshold goes into `MetaSwitcher._default_params()` or `config/settings.yaml`, enabling Phase 4 sweep without code changes.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (Odds band filter):** Needs analysis of historical backtest report to identify which bands have negative ROI with sufficient sample sizes. This is data-dependent and cannot be determined from code analysis alone.
- **Phase 3 (DD calibration):** Win-specific DD thresholds need empirical calibration on actual backtest data. The research provides theoretical guidance (400-500 bet window, 0.92-0.95 recovery threshold) but optimal values require experimentation.
- **Phase 4 (Parameter sweep):** The search space boundaries and validation methodology need careful design to avoid overfitting. Consider `/gsd-research-phase` for walk-forward validation best practices.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Conformal filter, regime skip):** Both are straightforward filter additions with well-documented patterns in the existing codebase.
- **Phase 2 (Kelly config, EV scaling):** Both are parameterization of existing arithmetic. No new algorithms needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new dependencies. All required packages verified installed. Kelly formula mathematically proven equivalent to existing code. External libraries evaluated and rejected with specific rationale. |
| Features | HIGH | Full codebase audit of 7 key files (2000+ LOC total). Every feature mapped to specific existing component with identified gap. Anti-features explicitly scoped out. Dependency map complete. |
| Architecture | HIGH | Data flow traced through actual backtest path. Protocol boundaries identified. Critical discovery: `BettingOrchestrator` is NOT in the backtest path. Build order derived from dependency analysis. |
| Pitfalls | HIGH | 8 pitfalls identified from code analysis + domain expertise. Each mapped to specific code locations (line numbers). Prevention strategies and warning signs specified. Recovery strategies provided. |

**Overall confidence:** HIGH

All research was grounded in direct codebase analysis with line-level references. No findings depend on assumptions or external documentation. The main uncertainty is empirical: the actual ROI impact of each feature on JRA data requires running backtests, which is the purpose of the implementation phases.

### Gaps to Address

- **Empirical ROI impact of each feature:** The research identifies what to build but cannot predict how much each feature improves ROI. Phase 4 addresses this, but Phases 1-3 should measure ROI delta after each change to validate or invalidate feature hypotheses.
- **Optimal regime parameters for WIN betting:** Current regime parameters (`ev_threshold`, `edge_threshold`) were set for PLACE betting. The research recommends OOF-percentile-based calibration, but the actual percentile values need to be computed from training data during implementation.
- **Conformal estimator coverage rate on WIN OOS data:** The research warns that conformal prediction assumes exchangeability, which may not hold for high-odds horses. The coverage rate (fraction of actual EVs above the lower bound) must be verified on OOS data before trusting the filter.
- **DD controller WIN-specific multiplier table values:** The research recommends wider DD bands but the exact thresholds (0-15%, 15-25%, 25-35%, etc.) are theoretical. Calibration requires running the DD controller in isolation on WIN backtest data and analyzing the multiplier distribution.

## Sources

### Primary (HIGH confidence)
- Full codebase audit: `src/betting/stake_calculator.py`, `src/betting/drawdown_controller.py`, `src/betting/win_strategy.py`, `src/betting/orchestrator.py`, `src/betting/gate_keeper.py`, `src/betting/meta_switcher.py`, `src/models/regime_detector.py`, `src/models/win_selection_gate.py`, `src/models/robust_confidence_estimator.py`, `src/backtest/engine.py`, `src/backtest/race_predictor.py`, `pyproject.toml`
- Kelly criterion equivalence proof: `edge/(odds-1) = (p*odds-1)/(odds-1) = p - (1-p)/(odds-1)` -- standard algebraic identity
- PROJECT.md v1.3 milestone definition (6 target features specified)
- Kelly Criterion literature: Thorp (2006), Benter (1994) -- foundational for stake sizing design

### Secondary (MEDIUM confidence)
- Pinnacle: Fractional Kelly -- half Kelly delivers ~75% of full Kelly growth with dramatically lower variance
- Matthew Downey: Why Fractional Kelly -- fractional Kelly is optimal under probability estimation uncertainty
- PLOS ONE: Statistical Theory of Optimal Decision-Making in Sports Betting -- confidence intervals for evaluating betting model performance
- Reddit r/quant: Applying Kelly to Sports Betting -- real-world data showing 25% Kelly provides better risk-adjusted returns
- arXiv: On Kelly Betting Limitations -- drawdown control feedback loop analysis
- Stanford: Risk-Constrained Kelly -- formal treatment of drawdown-aware Kelly sizing

### Tertiary (LOW confidence)
- Exact ROI impact of each feature on JRA data -- requires running backtests
- Optimal Kelly fractions for JRA win betting specifically -- theoretical guidance only
- Win betting hit rate variance profile on JRA data -- estimated at ~7-10% based on model characteristics
- Odds band survivorship bias in horse racing -- general statistical principle applied to this specific domain

---
*Research completed: 2026-05-04*
*Ready for roadmap: yes*
