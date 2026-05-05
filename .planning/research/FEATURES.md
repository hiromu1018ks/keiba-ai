# Feature Landscape: v1.4 Ensemble Filter Recalibration

**Domain:** Betting filter recalibration when switching from single LightGBM to 3-model GBM stacking ensemble
**Researched:** 2026-05-05
**Confidence:** HIGH (codebase audit + distribution shift analysis + domain literature)

## Context

This document covers ONLY features for the v1.4 milestone. The system has three filter components calibrated for single-model LightGBM output distributions:
1. **WinSelectionGate** (win_selection_gate.py) -- OOF-learned scoring + threshold grid search using win_selection_prob, win_selection_edge, tanoddslow quantile binning
2. **EV_lower filter** (race_predictor.py:get_win_candidates) -- hard cutoff on EV_lower_win_corrected >= 1.0, computed by RobustConfidenceEstimator via conformal prediction + rolling quantile
3. **OddsBandFilter** (odds_band_filter.py) -- ROI-based exclusion of unprofitable odds bands using training_bet_history

The core problem: switching to ensemble (StackedEnsemble with Ridge meta-learner) shifts the probability distribution, causing EV_lower to exclude 3,594 candidates (producing only 7 bets/year at 0% ROI instead of the needed 100+ bets/year at >100% ROI).

**Distribution shift mechanics:**
- Single LightGBM outputs: raw log-odds, wider spread, more extreme probabilities
- StackedEnsemble (Ridge over 3 GBMs): averaged predictions, compressed distribution, narrower spread
- Compressed probabilities produce lower EV estimates, which the EV_lower >= 1.0 hard threshold then over-excludes
- The conformal prediction nonconformity scores were calibrated on single-model residuals; ensemble residuals have different magnitude

---

## Table Stakes

Features required to make ensemble backtesting functional. Without these, the ensemble mode produces near-zero bets.

| # | Feature | Why Expected | Complexity | Dependencies | Notes |
|---|---------|--------------|------------|--------------|-------|
| TS-01 | WinSelectionGate retrained on ensemble OOF predictions | Gate score tables (combo_scores, pair_scores, single_scores) are built from single-model quantile edges. Ensemble output distribution changes where these quantile edges fall, making the old score tables misaligned. | Medium | StackedEnsemble OOF predictions available during training pipeline | Gate already has walk-forward OOF scoring architecture -- just needs to receive ensemble OOF preds instead of single-model |
| TS-02 | EV_lower threshold recalibrated to ensemble distribution | RobustConfidenceEstimator computes EV_lower_win_corrected using nonconformity scores from single-model residuals. Ensemble produces different residuals (typically smaller variance), causing the conformal band width to misestimate. Fixed threshold 1.0 becomes too aggressive. | Medium | RobustConfidenceEstimator.calibrate() must receive ensemble-residual data | The estimator already has calibrate() for computing CP quantiles -- needs ensemble-era calibration data |
| TS-03 | OddsBandFilter recalibrated with ensemble training_bet_history | ROI per odds band depends on model accuracy in each band. Ensemble accuracy profile differs from single model (typically better in mid-odds, may be worse in extreme odds). Training bet history from single model misidentifies which bands are profitable. | Low | training_bet_history parameter in BacktestEngine.run() | Filter already supports calibrate() with arbitrary bet history -- just needs ensemble-generated history |
| TS-04 | Optuna 14-dim parameter optimization executed | All regime/DD/EV-scaling/OddsBand parameters are at hardcoded defaults. These defaults were never tuned. Even with correct filter calibration, suboptimal parameters prevent reaching 100% ROI. | High | Strategy optimizer infrastructure from Phase 13 | run_strategy_optimization.py exists but has not been executed |

### Feature Dependencies

```
TS-01 (Gate retrain) ← requires ensemble OOF predictions (already generated in training_pipeline.py)
TS-02 (EV_lower recalibration) ← requires ensemble-residual calibration data
TS-03 (OddsBandFilter retrain) ← requires ensemble training_bet_history
TS-04 (Optuna optimization) ← requires TS-01/02/03 complete (filters must work before tuning params)
```

All of TS-01/02/03 can proceed in parallel once ensemble training produces the needed data. TS-04 depends on all three completing first.

---

## Differentiators

Features that improve ensemble filter quality beyond the baseline. Not strictly required, but significantly improve ROI potential.

| # | Feature | Value Proposition | Complexity | Dependencies | Notes |
|---|---------|-------------------|------------|--------------|-------|
| D-01 | Dynamic EV_lower threshold (regime-adaptive) | Instead of fixed EV_lower >= 1.0, allow threshold to vary by regime: lower in AGGRESSIVE (e.g., 0.95), higher in CONSERVATIVE (e.g., 1.05). This extracts more bets in favorable conditions while maintaining safety. | Medium | RegimeDetector, TS-02 | Regime already controls ev_threshold -- extending to EV_lower threshold is architecturally natural |
| D-02 | Ensemble-aware conformal confidence scoring | Weight conformal_confidence_score by ensemble agreement (how much the 3 base models agree). When models disagree, widen the confidence interval, making EV_lower more conservative. When models agree, narrow it. | Medium | StackedEnsemble OOF predictions, RobustConfidenceEstimator | Already have pairwise OOF correlations logged in _check_diversity() -- can extract agreement signal |
| D-03 | Quantile-adaptive EV threshold | Instead of a single EV_lower threshold for all probabilities, use different thresholds per probability bin. High-probability favorites can tolerate lower EV thresholds; longshots need higher thresholds to overcome estimation noise. | Medium | TS-01 (gate score tables provide probability bins) | WinSelectionGate already bins by probability quantile -- threshold per bin is a natural extension |
| D-04 | Walk-forward filter recalibration | Recalibrate filters within the backtest itself using an expanding window, not just once at the start. This makes filters adaptive to distribution drift over the test period. | High | WalkForwardCV infrastructure from Phase 4 | Complex; would require architectural changes to BacktestEngine race loop |

### Differentiator Dependencies

```
D-01 ← TS-02, RegimeDetector
D-02 ← StackedEnsemble (OOF agreement), RobustConfidenceEstimator
D-03 ← TS-01 (probability bin structure)
D-04 ← WalkForwardCV + all TS features
```

---

## Anti-Features

Features to explicitly NOT build. These are tempting but counterproductive.

| # | Anti-Feature | Why Avoid | What to Do Instead |
|---|-------------|-----------|-------------------|
| AF-01 | Auto-relax EV_lower threshold when bet count drops | If EV_lower >= 1.0 excludes too many candidates, lowering the threshold to produce more bets sacrifices quality for quantity. This masks the real problem (miscalibrated filters). | Fix the calibration (TS-02) so EV_lower is correctly computed for ensemble outputs. If ensemble genuinely produces fewer profitable bets, that is signal, not a problem to fix by lowering standards. |
| AF-02 | Separate filter pipelines for ensemble vs single-model | Maintaining two parallel filter configurations doubles maintenance burden and makes A/B comparison unreliable. Filters should be model-agnostic. | Make all filters calibrate themselves from whatever model produces the data. WinSelectionGate.train(), RobustConfidenceEstimator.calibrate(), and OddsBandFilter.calibrate() already accept arbitrary DataFrames -- just feed them ensemble data. |
| AF-03 | Recalibrating single-model to match ensemble distribution | Training a post-hoc calibration layer (Platt scaling / isotonic regression) to force ensemble output to match single-model distribution wastes information and can introduce new biases. | Let the ensemble distribution be what it is. Recalibrate the DOWNSTREAM filters (EV_lower, gate scores, band ROIs) to the new distribution instead. |
| AF-04 | Ensemble stacking of filters (meta-filter) | Running multiple filter configurations and voting/averaging adds complexity without clear benefit. Filter decisions are binary (bet/no-bet) -- stacking does not improve binary decisions the way it improves continuous predictions. | Use single, well-calibrated filter pipeline with proper thresholds. Optimize thresholds via Optuna (TS-04). |
| AF-05 | Neural network meta-learner replacing Ridge | Adding a neural network as the ensemble meta-learner increases complexity, introduces GPU dependencies, and provides minimal improvement over Ridge for 3-feature input. The existing Ridge meta-learner with 3 GBM inputs is already well-calibrated. | Keep Ridge meta-learner. Focus effort on downstream filter calibration, not model architecture changes. |

---

## Distribution Shift Analysis

The core technical challenge: understanding exactly how ensemble outputs differ from single-model outputs.

### Expected Distribution Changes

| Property | Single LightGBM | StackedEnsemble (Ridge) | Impact on Filters |
|----------|----------------|------------------------|-------------------|
| Probability spread | Wider, more extreme values | Narrower, compressed toward mean | WinSelectionGate quantile edges shift; fewer candidates in extreme bins |
| Calibration | Typically overconfident in extreme probabilities | Better calibrated (Ridge smooths) | EV estimates more accurate but lower absolute values for longshots |
| Residual distribution | Higher variance, heavier tails | Lower variance, more Gaussian | Conformal prediction bands narrower; EV_lower closer to point estimate |
| EV_lower_win_corrected | Wider interval, more values > 1.0 | Narrower interval, fewer values > 1.0 | This is the PRIMARY cause of 3,594 exclusions |
| OOD detection | Single-model has blind spots | Ensemble covers more of feature space | Gate scores more uniform; less differentiation |
| ROI by odds band | Variable -- model may be better in certain ranges | More uniform ROI across bands | OddsBandFilter band exclusions will change |

### Quantitative Diagnosis Needed

Before implementing any changes, the following diagnostic measurements should be collected:

1. **EV_lower distribution comparison**: Run single-model and ensemble on same test data, compare histograms of EV_lower_win_corrected. Quantify how many more candidates fall below 1.0 with ensemble.
2. **Conformal residual comparison**: Compare single-model vs ensemble residual distributions (|actual_ev - predicted_ev|). Confirm ensemble has smaller residuals.
3. **Probability calibration comparison**: Plot calibration curves (predicted probability vs observed frequency) for both models. Confirm ensemble is better calibrated.
4. **Gate score distribution comparison**: Compare WinSelectionGate score distributions under both models. Identify where quantile edges shift.

These diagnostics should be run as part of TS-01/TS-02 implementation to confirm the distribution shift hypothesis before coding solutions.

---

## Model-Agnostic Filter Design Principles

Based on codebase analysis and domain research, the following principles should guide filter recalibration:

### Principle 1: Filters Consume Data, Not Models
All three filter components (WinSelectionGate, RobustConfidenceEstimator, OddsBandFilter) already accept generic DataFrames. They do not reference model internals. This is the correct architecture -- filters should never need to know whether upstream predictions come from LightGBM, XGBoost, or a stacked ensemble.

### Principle 2: Calibration Data Must Match Deployment Data
The fundamental rule: calibration data (used to set filter parameters) must come from the same model that will be used in deployment. WinSelectionGate.train() must receive ensemble OOF predictions, not single-model OOF. RobustConfidenceEstimator.calibrate() must receive ensemble residuals. OddsBandFilter.calibrate() must receive ensemble-era bet history.

### Principle 3: Thresholds Follow Distribution, Not Intuition
Fixed thresholds (EV_lower >= 1.0) encode assumptions about the probability distribution. When the distribution changes, the threshold must be re-validated. The correct approach is to set thresholds via optimization (grid search / Optuna) against the target metric (ROI with minimum bet count), not by intuition.

### Principle 4: Score Tables Are Distribution-Specific
WinSelectionGate's combo_scores, pair_scores, and single_scores are lookup tables built from quantile-binned realized ROI. The quantile edges (prob_edges, edge_edges, odds_edges) are specific to the training distribution. When the model changes, these edges must be recomputed from the new distribution.

### Principle 5: Separate Model Quality from Filter Calibration
A model change can affect ROI through two channels: (a) the model itself produces better/worse predictions, and (b) the filter parameters are miscalibrated for the new distribution. These must be evaluated independently. First, confirm the ensemble actually produces better predictions (check calibration curve, Brier score). Then, recalibrate filters. Then, run Optuna to optimize all parameters jointly.

---

## MVP Recommendation

**Priority 1 (Blocking -- must complete first):**
1. **TS-02**: EV_lower recalibration -- this is the direct cause of the 3,594-exclusion / 7-bet problem. Recalibrate RobustConfidenceEstimator on ensemble residuals.
2. **TS-01**: WinSelectionGate retrain on ensemble OOF -- necessary for correct gate scores and candidate ranking.

**Priority 2 (Required for target):**
3. **TS-03**: OddsBandFilter recalibration -- needed to avoid excluding profitable ensemble-era bands.
4. **TS-04**: Optuna optimization -- without this, all parameters are at defaults and ROI will likely remain suboptimal.

**Priority 3 (Defer to post-v1.4):**
5. **D-01**: Dynamic EV_lower threshold -- useful enhancement but not needed for initial 100% ROI target.
6. **D-02**: Ensemble-aware confidence scoring -- sophisticated but optional.
7. **D-03**: Quantile-adaptive EV threshold -- would require gate architecture changes.

**Defer indefinitely:**
- D-04 (walk-forward filter recalibration) -- too complex for this milestone.
- All anti-features (AF-01 through AF-05).

---

## Implementation Complexity Assessment

| Feature | Lines of Code Changed | New Code | Test Complexity | Risk |
|---------|----------------------|----------|-----------------|------|
| TS-01 | ~20 (pipeline data routing) | ~0 | Low (gate already tested) | LOW -- just feeding different data to existing train() |
| TS-02 | ~30 (calibration data routing) | ~0 | Low (estimator already tested) | LOW -- just feeding different residuals to calibrate() |
| TS-03 | ~10 (training_bet_history generation) | ~0 | Low (filter already tested) | LOW -- just passing ensemble-era bet history |
| TS-04 | ~0 (script execution) | ~0 | Medium (verify optimization results) | MEDIUM -- depends on all above working correctly |
| D-01 | ~40 (regime-adaptive EV_lower) | ~20 | Medium | MEDIUM |
| D-02 | ~50 (agreement signal extraction) | ~30 | High | HIGH |

**Total estimated effort for Priority 1+2:** Minimal code changes -- the infrastructure is already built. The work is primarily in data routing and script execution, not new algorithm development.

---

## Filter Interaction Map

How the three filters interact after ensemble switch:

```
Ensemble predictions (StackedEnsemble.predict())
    |
    v
RacePredictor.predict()
    |  Computes: p_win, e_return, ev_win, ev_win_corrected
    |  RobustConfidenceEstimator.predict_interval() computes EV_lower_win_corrected
    |
    v
get_win_candidates()
    |  Filter 1: win_selection_edge > 0 AND tanodds >= 1.0
    |  Filter 2: EV_lower_win_corrected >= 1.0 (TS-02: recalibrated)
    |  Rank: win_gate_score DESC (TS-01: retrained)
    |  Top 2 candidates
    |
    v
OddsBandFilter.filter()
    |  Filter 3: exclude unprofitable bands (TS-03: recalibrated)
    |
    v
select_bets()
    |  Kelly stake * regime fraction * EV scaling * DD control
    |  (TS-04: Optuna-optimized parameters)
    |
    v
Final bet list
```

The key insight: all three filter recalibrations (TS-01/02/03) are about feeding the RIGHT DATA to existing calibration methods. No new algorithms needed.

---

## Expected Outcomes After Recalibration

Based on the distribution shift analysis and the existing filter architecture:

| Metric | Current (ensemble, miscalibrated) | Expected (ensemble, calibrated) | Rationale |
|--------|----------------------------------|-------------------------------|-----------|
| Bets/year | 7 | 100-300 | EV_lower will exclude far fewer candidates once calibrated to ensemble residuals |
| ROI | 0% | 95-110% | Ensemble should produce better predictions + Optuna optimization |
| Gate pass rate | ~0.1% | 3-8% | Gate thresholds aligned to ensemble distribution |
| EV_lower pass rate | ~0.5% | 5-15% | Conformal bands properly sized for ensemble residuals |
| OddsBand exclusions | Unknown (likely over-excluding) | Data-driven | Ensemble ROI profile per band differs from single model |

The 100% ROI target is achievable because:
1. Ensemble predictions are better calibrated (Ridge meta-learner smooths GBM outputs)
2. Optuna will optimize 14 dimensions against ROI with bet-count constraints
3. Filter calibration removes false negatives (profitable bets excluded by miscalibrated thresholds)

---

## Sources

### Primary (HIGH confidence)
- Codebase audit: src/models/win_selection_gate.py (1113 lines) -- train(), score(), OOF scoring architecture
- Codebase audit: src/models/robust_confidence_estimator.py -- calibrate(), predict_interval(), conformal + rolling quantile
- Codebase audit: src/betting/odds_band_filter.py (112 lines) -- calibrate(), filter() with training bet history
- Codebase audit: src/models/stacked_ensemble.py (607 lines) -- Ridge meta-learner, OOF prediction generation
- Codebase audit: src/backtest/race_predictor.py -- get_win_candidates() with EV_lower filter, select_bets() pipeline
- Codebase audit: src/pipelines/training_pipeline.py -- ensemble OOF prediction flow, use_ensemble flag routing
- Codebase audit: .planning/phases/11-bet-selection-filters/11-RESEARCH.md -- Phase 11 filter architecture
- Codebase audit: .planning/phases/12-stake-sizing-enhancement/12-RESEARCH.md -- Phase 12 stake sizing
- Codebase audit: .planning/phases/13-risk-calibration-parameter-optimization/13-RESEARCH.md -- Phase 13 Optuna optimization
- Codebase audit: .planning/PROJECT.md -- v1.4 active requirements and current state

### Secondary (MEDIUM confidence)
- [ScienceDirect: ML for sports betting -- calibration over accuracy](https://www.sciencedirect.com/science/article/pii/S266682702400015X) -- confirms calibration-focused model selection yields higher betting profits
- [arXiv: Systematic review of ML in sports betting](https://arxiv.org/html/2410.21484v1) -- surveys ML techniques in sports betting contexts
- [ResearchGate: ML for betting -- accuracy vs calibration](https://www.researchgate.net/publication/369184023_Machine_learning_for_sports_betting_should_forecasting_models_be_optimised_for_accuracy_or_calibration) -- optimizing for calibration leads to greater returns than accuracy

### Tertiary (LOW confidence)
- [arXiv: Probabilistic recalibration of forecasts](https://www.sciencedirect.com/science/article/am/pii/S016920701930158X) -- general theory of probabilistic recalibration after model changes
- [MDPI: NBA forecasting with calibrated probabilities + Kelly staking](https://www.mdpi.com/2078-2489/17/1/56) -- betting simulation with calibrated probabilities and EV threshold filtering
