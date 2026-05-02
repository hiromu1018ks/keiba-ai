# Project Research Summary

**Project:** keiba-ai win (tansho) model improvement
**Domain:** Parimutuel horse racing prediction -- JRA Japan
**Researched:** 2026-05-02
**Confidence:** HIGH

## Executive Summary

This is an incremental improvement project on a mature horse racing prediction system (keiba-ai v5.5) that already has a 2-stage LightGBM model (P(win) x E(odds|win)), 14 feature modules, and a sophisticated place betting pipeline. The win (tansho) model currently produces 89% ROI on 2024 backtest data -- a loss. Research across four dimensions converges on a clear diagnosis: the win pipeline is architecturally incomplete compared to the place pipeline. The place path has Benter combination (fundamental + market probability blending), isotonic calibration, temperature scaling, a learned selection gate, and regime-adaptive betting. The win path has none of these. It outputs raw 2-stage model predictions directly into a simple threshold-based bet selector.

The recommended approach is to close this gap through four sequential phases: (1) win-specific feature analysis and enhancement, (2) win Benter combination with calibration, (3) win selection gate with confidence estimation, and (4) win betting strategy refinement with proper Kelly sizing. No new ML frameworks are needed. The stack additions are minimal (betacal for beta calibration, optionally mapie for conformal prediction). The highest-leverage single change is implementing the Benter combination for win predictions, which would give the win model access to the market's efficiency signal -- something it currently lacks entirely.

The key risks are: overfitting to the 2024 test year (mitigated by walk-forward validation), calibration overconfidence amplified by Kelly staking (mitigated by fractional Kelly and calibration diagnostics), and the JRA 25% parimutuel takeout requiring higher edge thresholds than the current 4-8% range. Each phase produces a measurable backtest ROI change, enabling independent evaluation and rollback.

## Key Findings

### Recommended Stack

The existing stack (Python 3.11, LightGBM, XGBoost, CatBoost, scikit-learn, MLflow) is correct for this domain. Tabular GBDT models dominate horse racing prediction at this data scale (~9K bets/year). Neural networks would add complexity without signal. The only new dependency needed is `betacal>=1.1.0` for beta calibration as an alternative to isotonic regression. MAPIE for conformal prediction is optional -- CQR can be implemented manually. Version bumps are low priority; LightGBM 4.3 to 4.6 is the only one worth considering. No Python version change.

**Core technologies:**
- LightGBM + XGBoost + CatBoost stack with Ridge meta-learner: correct for tabular racing data. The gap is hyperparameter tuning (currently hardcoded), not model architecture.
- betacal (new): 3-parametric beta calibration that avoids the step-function artifacts and overcorrection of isotonic regression. Strictly superior to Platt scaling per Kull et al. (2017).
- scikit-learn IsotonicRegression + custom TemperatureScaling: keep as primary calibration path; add beta calibration as a secondary option to compare.
- scipy.optimize: already used for Kelly optimization; extend for constrained pool-size-aware sizing.
- Optuna: already available; needs to be applied to StackedEnsemble base model hyperparameters (currently hardcoded lr=0.03, leaves=31, rounds=300).

### Expected Features

The system already has 100+ features across 14 modules covering all table-stakes handicapping factors (speed, class, form, jockey, market, surface/distance, weight, field size). The gap is not in missing fundamental features but in (a) win-specific features that the place-optimized feature set lacks, and (b) derived features that directly measure the model's edge over the market.

**Must have (table stakes -- already implemented):**
- Speed/ability rating (norm_finish_logit_avg, harontimel5_zscore)
- Class level and transitions (class_level, class_move)
- Recent form cycle (form_trend, form_consistency, form_peak_flag)
- Jockey quality (jockey_wr_overall, jockey_surprise)
- Market probability (p_market_win_adj, popularity_rank)
- Surface/distance suitability (blood_*, sire_*, pace_aptitude)
- Odds dynamics (odds_drop_rate, odds_velocity, odds_volatility)

**Should build (high-impact win-specific features):**
- Odds-to-ability ratio (p_market / p_ability): the single most important ROI signal -- directly measures betting edge
- Distance change delta: horses switching distance categories show predictable performance changes
- Surface change flag: turf/dirt switch is a strong negative signal
- Class drop bounce: horses dropping in class after poor higher-class results
- Win/losing streak: consecutive result counter capturing hot/cold form
- Trainer recent form (30/60/90 day): captures trainer hot streaks masked by annual stats
- Jockey-horse pairing history: familiarity between specific jockey-horse combinations

**Defer (lower priority for win):**
- Expected pace figure per horse: high complexity, better suited for place
- Layoff return performance: more useful for place prediction
- Grade race debut flag: low signal, niche scenario
- Seasonal pattern: mostly captured by track_condition_code

### Architecture Approach

The target architecture adds a win-specific enhancement layer that mirrors the existing place pipeline's downstream refinements. The key pattern is "dual pipeline coexistence": shared upstream models (AbilityModel, MarketModel) diverge at the 2-stage model level into independent win and place paths. The win path needs five new components that the place path already has.

**Major components (in data flow order):**
1. **WinTwoStageModel** (exists, enhance): P(win) x E(odds|win) with expanded win-specific feature set
2. **WinEVCorrectionModel** (exists, minor changes): P/E decomposition correction for win probabilities
3. **WinBenterGate** (NEW): Combine fundamental P(win) with market-implied P(win) via logit-space weighting; unified combination + selection component
4. **WinCalibrationPipeline** (NEW): IsotonicRegression + TemperatureScaling fitted on win Benter-combined probabilities
5. **WinConfidenceEstimator** (NEW): EV lower bound for conservative bet sizing, using conformal prediction
6. **WinSelectionGate** (NEW): Learned binary gate for win bet pass/reject decisions
7. **WinStrategy** (exists, enhance): Kelly-based stake sizing with regime-adaptive parameters, pool-size-aware capping, quarter-Kelly fraction

### Critical Pitfalls

1. **Win model ignores market signal entirely (Pitfall 3)** -- The Benter combination is only applied to place. Win predictions use raw 2-stage output without any market probability blending. This is the single highest-impact gap. Prevent by implementing WinBenterGate as first priority.

2. **Edge thresholds ignore JRA 25% takeout (Pitfall 1)** -- Current 4-8% edge thresholds are insufficient given the 25% house take. The model bets on many "positive EV" horses that are actually negative EV. Prevent by calibrating edge thresholds against actual historical ROI and computing edge relative to fair odds.

3. **Calibration overconfidence amplified by Kelly staking (Pitfall 5)** -- Walsh & Joshi (2023) show calibration-optimized models yield +37% ROI vs accuracy-optimized yielding -76%. Uncalibrated LightGBM probabilities fed into Kelly sizing will systematically overbet. Prevent by implementing win-specific calibration diagnostics and using quarter-Kelly (0.25) for win bets.

4. **Overfitting to single 2024 test year (Pitfall 6)** -- Every design decision is made with knowledge of 2024 performance. No holdout beyond 2024. Prevent by walk-forward validation across multiple years and reserving 2025 as final holdout.

5. **2-stage PxE independence assumption breaks at tails (Pitfall 2)** -- The decomposition assumes winning probability is independent of payout odds. In parimutuel systems this is violated, causing systematic EV bias for longshots. Prevent by evaluating calibration by odds bucket and applying Benter combination before EV calculation.

## Implications for Roadmap

### Phase 1: Data Validation and Feature Analysis
**Rationale:** Before building any new components, validate data integrity and understand which existing features drive win prediction. This phase has zero architectural risk -- it only analyzes and adds features to the existing model.
**Delivers:** Win-specific feature importance ranking, odds snapshot timing audit, new high-impact features (odds-to-ability ratio, distance change, surface change, class drop bounce, win/losing streak).
**Addresses:** Pitfall 8 (place features may not transfer to win), Pitfall 4 (odds timing audit), Pitfall 12 (odds dynamics look-ahead check).
**Avoids:** Pitfall 9 (feature pre-computation duplication -- any new features must be added to shared computation path).

### Phase 2: Win Benter Combination and Calibration
**Rationale:** The single highest-leverage change. The win model currently has no mechanism to incorporate market efficiency. Benter combination (fundamental + market probability blending) must come before any betting strategy work because it changes the probability estimates that drive all downstream decisions.
**Delivers:** WinBenterGate component, win-specific IsotonicRegression + TemperatureScaling, win calibration diagnostics (reliability diagram by odds bucket).
**Uses:** betacal (for comparison with isotonic), existing BenterCombination class (reused with win-specific fit).
**Implements:** Architecture components 3 (WinBenterGate) and 4 (WinCalibrationPipeline).
**Addresses:** Pitfall 3 (missing Benter for win), Pitfall 2 (2-stage independence assumption partially mitigated by Benter blending), Pitfall 7 (favorite-longshot bias addressed by Benter's market component).

### Phase 3: Win Selection Gate and Confidence Estimation
**Rationale:** With calibrated win probabilities in hand, the system needs a learned filter to separate genuine betting opportunities from noise. This mirrors the PlaceSelectionGate pattern that proved effective for place betting.
**Delivers:** WinConfidenceEstimator (EV lower bound via conformal prediction), WinSelectionGate (learned binary filter), integration into RacePredictor and TrainingPipeline.
**Implements:** Architecture components 5 (WinConfidenceEstimator) and 6 (WinSelectionGate).
**Addresses:** Pitfall 5 (calibration diagnostics now feed directly into confidence bounds), Pitfall 11 (E model R-squared monitoring added).

### Phase 4: Win Betting Strategy
**Rationale:** With a complete prediction pipeline (features, Benter combination, calibration, selection gate), the final phase optimizes the betting decisions themselves. This must come last because stake sizing depends on calibrated edge estimates.
**Delivers:** Regime-adaptive win parameters, pool-size-aware Kelly sizing with quarter-Kelly fraction, edge thresholds calibrated for JRA 25% takeout, win bet generation in BacktestEngine.
**Implements:** Architecture component 7 (WinStrategy enhanced).
**Addresses:** Pitfall 1 (takeout-adjusted thresholds), Pitfall 10 (regime detector evaluation for win), Pitfall 15 (stake discretization).

### Phase 5: Validation and Hardening
**Rationale:** Multi-year walk-forward validation to confirm the model generalizes beyond 2024. This is a guardrail, not an optional step.
**Delivers:** Walk-forward CV results across 2022-2025, sensitivity analysis on edge thresholds, train-vs-test ROI gap analysis, concept drift monitoring.
**Addresses:** Pitfall 6 (overfitting to single year), Pitfall 13 (concept drift detection).

### Phase Ordering Rationale

- Phase 1 comes first because all subsequent phases depend on having correct data and understanding which features matter for win prediction. Feature additions are low-risk and can be measured independently.
- Phase 2 is the critical path item. Without Benter combination, the win model cannot access the market's efficiency signal. This alone may push ROI above 100%.
- Phase 3 depends on Phase 2 because the selection gate needs calibrated probabilities as input. Without calibration, the gate cannot distinguish genuine edges from noise.
- Phase 4 depends on Phases 2-3 because betting strategy needs calibrated edge estimates to size bets correctly.
- Phase 5 runs as a final validation across all phases, confirming the complete system works on unseen years.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Benter + Calibration):** The interaction between Benter combination and the 2-stage PxE decomposition needs careful design. The Benter paper uses single-stage probabilities; applying it after 2-stage decomposition requires validating that the combined probability still produces valid EV estimates.
- **Phase 4 (Betting Strategy):** JRA pool-size data is needed for pool-size-aware Kelly. The exact pool sizes and their effect on dividends require domain research specific to JRA.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Feature Analysis):** Standard LightGBM feature importance + SHAP. Well-documented, established patterns.
- **Phase 3 (Selection Gate):** Mirrors the existing PlaceSelectionGate pattern exactly.
- **Phase 5 (Validation):** Standard walk-forward CV for time-series data.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All technologies are proven in the codebase. Only betacal is new, and it is a small, well-documented library. No architectural risk from stack changes. |
| Features | HIGH | All proposed new features use existing data and simple computations. Feature importance analysis is standard ML practice. Anti-features (leakage sources) are well-identified. |
| Architecture | HIGH | The target architecture mirrors the existing place pipeline, which is already working. New components follow established patterns (BenterCombination, PlaceSelectionGate). The codebase integration points (RacePredictor, TrainingPipeline, BacktestEngine) are clearly identified. |
| Pitfalls | HIGH | Pitfalls are cross-validated against academic literature (Benter 1994, Walsh & Joshi 2023), official JRA documentation, and direct codebase analysis. The critical pitfalls (missing Benter, takeout thresholds, calibration) are directly observable in the current 89% ROI. |

**Overall confidence:** HIGH

### Gaps to Address

- **Benter-after-2-stage validity:** The Benter paper uses single-stage probabilities. Applying Benter combination after the PxE 2-stage decomposition needs validation during Phase 2 planning. If the 2-stage independence assumption (Pitfall 2) proves severely violated, a single-stage win model may be needed instead.
- **JRA pool-size data for Kelly cap:** Pool-size-aware Kelly betting requires knowing typical win pool sizes for different race grades. This data is not currently in the feature set and may need to be estimated or sourced during Phase 4.
- **Optimal win Kelly fraction:** Research recommends 0.25-0.50 fractional Kelly. The exact optimal fraction depends on the calibration quality achieved in Phase 2-3. This needs empirical tuning during Phase 4.
- **E(odds|win) model effectiveness:** With only ~7% positive samples, the E sub-model may have insufficient data (Pitfall 11). Phase 3 should include an evaluation of whether the 2-stage decomposition actually adds value over using raw odds as the payout estimate.
- **Odds snapshot timing:** The exact timing of odds snapshots relative to post time needs auditing (Pitfall 4). This is a data-level concern that may affect all phases.

## Sources

### Primary (HIGH confidence)
- Benter (1994), "Computer Based Horse Race Handicapping and Wagering Systems" -- foundational architecture and Benter combination pattern
- Walsh & Joshi (2023), arXiv:2303.06021 -- calibration vs accuracy ROI impact (+37% vs -76%)
- JRA Official Guide -- 25% takeout rate documentation
- Codebase analysis: `src/models/`, `src/backtest/race_predictor.py`, `src/pipelines/training_pipeline.py` -- directly observed architectural gaps

### Secondary (MEDIUM confidence)
- Kull et al. (2017) -- beta calibration superiority over Platt scaling
- Bolton & Chapman (1986) -- multinomial logit model for horse racing
- StableBet and seven-seas-punter -- LightGBM + calibration patterns for racing
- ResearchGate: ensemble methods for racing prediction
- Walk-forward validation literature for time-series ML

### Tertiary (LOW confidence)
- Reddit ML practitioner anecdote on concept drift frequency
- Individual blog posts on odds timing in backtesting

---
*Research completed: 2026-05-02*
*Ready for roadmap: yes*
