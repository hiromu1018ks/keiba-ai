# Feature Landscape: v2.1 MarketAware Calibration + Race-Level Ranker for ROI Recovery

**Domain:** Horse racing prediction / win bet calibration and ranking
**Researched:** 2026-05-27
**Scope:** 4 active features -- MarketAwareWinCalibrator, Segment Conditioning, Race-Level Ranker, Shadow Comparison Metrics
**Supersedes:** v2.0 FEATURES.md (that document covers OOF health + feature frame; this covers calibration + ranking for ROI recovery)

---

## Executive Summary

The v2.1 milestone targets ROI recovery from 87.8% (v2.0) to 100%+, leveraging the v2.0 infrastructure (OOF health, InvestmentFeatureFrame, 94-spec schema). The four active features form a causal chain: better probability calibration (MarketAwareWinCalibrator + segment conditioning) produces better race-level ordering (Race-Level Ranker), which is validated before deployment through shadow comparison metrics.

The critical insight from codebase analysis is that most building blocks already exist. `BenterCombination` already implements logit-blend with MLE fitting (alpha/beta/gamma). `WinBenterGate` already applies this to win probability with race normalization, isotonic calibration, and temperature scaling. `WinSegmentCalibrator` already segments by surface|odds_band|rank_band|ev_band with Bayesian shrinkage. The gap is that these components operate independently -- the Benter blend uses global alpha/beta without segment conditioning, and the ranker (`get_win_candidates`) uses a hand-tuned multi-factor scoring formula rather than a learned ranking model.

The design challenge is not "build from scratch" but "wire existing components together with segment-aware weights." Specifically: extend BenterCombination so alpha/beta vary by segment (popularity rank / odds band / probability rank), replace the `win_market_selection_score` hand-tuned formula with a learned ranker consuming InvestmentFeatureFrame outputs, and add shadow comparison infrastructure to validate before switching.

---

## Component 1: MarketAwareWinCalibrator

### What Already Exists

| Component | Location | Status |
|-----------|----------|--------|
| `BenterCombination` | `src/models/benter_combination.py` | Working. MLE fit (L-BFGS-B), alpha/beta/gamma logit blend, beta floor >= 0.20 |
| `WinBenterGate` | `src/models/win_benter_gate.py` | Working. Wraps BenterCombination + tanodds extraction + race normalization + optional isotonic/temp scaling |
| `generate_win_oof_predictions()` | `src/models/win_benter_gate.py` | Working. Walk-forward OOF for Benter fitting |
| `compare_calibrations()` | `src/models/win_benter_gate.py` | Working. Beta vs Isotonic comparison with Brier/ECE |
| `EVCorrectionModel` | `src/models/ev_correction_model.py` | Working. P-correction + E-correction + isotonic calibration + odds-band residual scaling |
| `TemperatureScaling` | `src/models/benter_combination.py` | Working. Post-hoc Brier+NLL minimization |

### Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Segment-conditioned alpha/beta | Global alpha/beta treats favorites and longshots identically; favorite-longshot bias literature proves they need different weights | High | Core new work: alpha_s, beta_s per segment, with regularization toward global values |
| OOF-based segment fitting | In-sample segment parameters overfit to noise; must use OOF predictions | Medium | `generate_win_oof_predictions()` already produces OOF arrays; add segment membership column |
| Surface-aware fitting | Turf and dirt have different market efficiency; v1.7 showed turf conservative regime is unprofitable | Low | Already proven in submodel split pattern; fit per-surface Benter parameters |
| Post-hoc isotonic or Beta calibration | Benter blend output may still be miscalibrated; calibration layer corrects residual bias | Low | `compare_calibrations()` already implements both; select winner by Brier score |
| Race normalization | Probabilities within a race must sum to 1.0; uncalibrated blend violates this | Low | `WinBenterGate.apply()` already does `p_combined / race_sum` |
| Probability clipping | Logit(0) and logit(1) are undefined; numerical safety | Low | Already in BenterCombination: clip to [1e-10, 1-1e-10] |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Bayesian shrinkage toward global alpha/beta | Prevents segment parameters from diverging when data is sparse; prior_strength controls regularization | Medium | `WinSegmentCalibrator` already uses this pattern with `prior_strength=500`; replicate for Benter parameters |
| Logit-gap interaction with segment | `if_logit_gap` from InvestmentFeatureFrame captures model-vs-market disagreement; conditioning on this improves calibration in disagreement regions | Medium | Derived feature already computed in `InvestmentFeatureFrameBuilder._compute_derived()` |
| Per-segment temperature scaling | Different segments may have different confidence levels; segment-specific T corrects this | Medium | One TemperatureScaling per segment; shrink toward global T with Bayesian prior |
| Calibration stability across OOF folds | If alpha_s swings wildly across folds, the segment conditioning is unstable | Medium | Compute CV of parameters across OOF folds; flag segments with CV > 0.3 |
| Pseudo-R2 (Benter) as quality metric | Benter paper shows `dR2 = R2_combined - R2_public` correlates with profitability; directly measures information added by fundamental model | Low | Compute per-segment; segments with negative dR2 should use higher market weight |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Neural network calibrator | 3-7 parameters per segment is tractable; NN adds opacity and overfitting risk | Parametric logit blend + temperature scaling |
| Dropping BenterCombination | Working MLE fitting with tested numerical stability; rewrite is waste | Extend with segment conditioning; do not replace |
| More than 3 parameters per segment | With ~20-50 segments, each extra parameter multiplies overfitting risk | alpha + beta + gamma per segment maximum; use shrinkage |
| Segment conditioning by regime | RegimeDetector uses market-level aggregate (fav_rate x overround); too coarse for horse-level calibration | Use horse-level segments (popularity rank, odds band) not race-level regimes |
| Independent fitting per segment without regularization | Sparse segments (e.g., 100+ odds band) will overfit to noise | Bayesian shrinkage toward global parameters; minimum sample size per segment |
| Using p_market_pred as input | MarketModel Rule 11: predicted market probability must not reach Stage2 | Calibrator uses tanodds-derived implied probability, not MarketModel's p_market_pred |

---

## Component 2: Popularity/Odds/Probability-Rank Segment Conditioning

### What Already Exists

| Component | Location | Status |
|-----------|----------|--------|
| `WinSegmentCalibrator` | `src/models/win_segment_calibrator.py` | Working. Segments by surface\|odds_band\|rank_band\|ev_band with Bayesian shrinkage. ODDS_BINS: (1,2,5,10,30,100,inf), RANK_BINS: (0,1,3,6,8,inf) |
| `OddsBandFilter` | `src/betting/odds_band_filter.py` | Working. BANDS and BAND_NAMES define odds groupings |
| `if_popularity_rank` | `src/investment/schema_registry.py` | In InvestmentFeatureFrame. Popularity rank from market |
| `if_odds_band_id` | `src/investment/schema_registry.py` | In InvestmentFeatureFrame. Odds band identifier |
| `if_p_win_race_rank` | `src/investment/schema_registry.py` | In InvestmentFeatureFrame. Percentile rank of p_win within race |
| `ensure_win_selection_columns()` | `src/models/win_selection_gate.py` | Working. Ensures win_selection_prob/ev/edge columns exist |

### Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Segment definition by popularity rank | Favorites (rank 1-3), mid-range (4-6), longshots (7+) have fundamentally different calibration needs | Low | WSC already defines RANK_BINS (0,1,3,6,8,inf) -- reuse exactly |
| Segment definition by odds band | Odds directly measure market confidence; different bands have different favorite-longshot bias magnitude | Low | WSC already defines ODDS_BINS (1,2,5,10,30,100,inf) -- reuse exactly |
| Segment definition by EV band | Horses with EV > 1.5 are where model disagrees with market most; calibration quality varies by EV level | Low | WSC already defines EV_BINS (-inf,0.8,1.0,1.2,1.5,2.0,inf) -- reuse exactly |
| Minimum sample size per segment | Sparse segments produce unreliable parameters; guardrail against noise | Low | WSC uses min_segment_rows=120, min_segment_wins=3 -- proven thresholds |
| Segment key format | Composite key for lookup; must be deterministic and serializable | Low | WSC uses `surface|odds_band|rank_band|ev_band` string format -- reuse |
| Default factor for missing segments | At inference time, horse may not match any trained segment | Low | Return 1.0 (no adjustment) for unmapped segments; WSC already does this |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Probability-rank percentile segment | `if_p_win_race_rank` from InvestmentFeatureFrame provides model-based ranking; captures cases where popularity rank disagrees with model | Low | Add as alternative to popularity_rank segmentation; compare calibration quality |
| Adaptive segment boundaries | Fixed bins may not capture the actual calibration boundary; quantile-based bins adapt to data distribution | Medium | `WinSelectionGateModel._quantile_edges()` already computes adaptive bins -- reuse pattern |
| Cross-segment smoothing | Adjacent segments (e.g., odds 1-2 and 2-5) should have correlated parameters; smoothing reduces variance | Medium | Kernel smoothing or weighted average of neighboring segment parameters |
| Segment diagnostics export | Per-segment Brier, hit rate, ROI, sample count; enables human review of calibration quality | Low | Similar to WSC training_summary; export as JSON manifest |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Fine-grained segments (< 50 samples each) | Statistical noise dominates; parameters are meaningless | Enforce minimum sample size; merge sparse segments into neighbors |
| More than 4 segment dimensions | 4 dimensions (surface, odds, rank, EV) already produce ~150+ segments; more causes combinatorial explosion | Stick with 4 dimensions; use WSC proven segment keys |
| Segment conditioning that reduces bet count | PROJECT.md explicitly requires "no bet count reduction" as deployment constraint | Calibration adjusts probabilities, not filtering; bet selection is separate |
| Dynamic segment boundaries at inference time | Boundaries must be frozen at training time; inference uses the same bins | Train boundaries on OOF data; store in manifest; apply identically at inference |
| One segment per individual odds value | Continuous segmentation defeats the purpose of grouping for statistical reliability | Use binned segments; continuous calibration is the Benter blend's job |

---

## Component 3: Race-Level Ranker

### What Already Exists

| Component | Location | Status |
|---------|------------|-------|
| `get_win_candidates()` | `src/backtest/race_predictor.py` | Working. 450-line method with hand-tuned multi-factor scoring: `win_market_selection_score = selection_score - late_odds_drop*weight - log_odds_penalty + prob_rank_bonus - ev_tail_pressure - market_risk_penalty` |
| `InvestmentFeatureFrameBuilder` | `src/investment/feature_frame.py` | Working. 94 specs / 9 categories. Includes `if_p_win_race_rank`, `if_ev_race_rank`, `if_edge_rank_in_race`, `if_edge_zscore_in_race`, `if_top3_gap`, `if_ev_top1_gap`, `if_ev_top3_indicator`, `if_field_ev_dispersion` |
| `WinSelectionGateModel` | `src/models/win_selection_gate.py` | Working. 1282-line learned gate with walk-forward OOF scoring, threshold grid search, runner-up candidate detection |
| `WinProfitSelector` | `src/models/win_profit_selector.py` | Working. Profit-based candidate selection with max_per_race |
| `win_market_selection_score` | `src/backtest/race_predictor.py` | Current ranking signal. Hand-tuned formula with surface-aware base score + 5 penalty/bonus terms |

### Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Learned ranking within race | Hand-tuned `win_market_selection_score` has 6 surface-specific parameters that Optuna must search; a learned ranker automates this | Medium | LightGBM `LGBMRanker` with `objective='lambdarank'`; race_id as group |
| InvestmentFeatureFrame as input | Curated 94-spec frame provides consistent feature set; avoids ad-hoc column selection | Low | `build_frame(df, mode="infer")` produces the ranker input |
| OOF-based ranker training | In-sample ranking metrics are meaningless; ranker must be trained on OOF predictions | Medium | Use OOF predictions from v2.0 OOFHealthValidator infrastructure |
| Group constraint (horses per race) | LambdaRank requires `group` parameter; data must be sorted by race_id | Medium | `df.groupby('race_id').size().values`; MUST sort by race_id first |
| NDCG as evaluation metric | Standard ranking quality metric; early stopping signal | Low | `eval_at=[1, 3, 5]` for top-k relevance |
| Position-graded relevance | Binary win/loss target wastes information; 1st >> 2nd >> 3rd for ranking | Medium | Relevance = `1/finish_position` or exponential decay; standard in learning-to-rank |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Calibrated probability as ranking feature | `p_win_final` from MarketAwareWinCalibrator provides superior probability signal for ranking | Low | Requires Component 1 completion; add to InvestmentFeatureFrame as `if_p_win_calibrated` |
| Per-surface ranker training | Turf and dirt have different ranking dynamics; turf conservative regime is unprofitable, suggesting ranker miscalibration on turf | Medium | Train separate rankers OR include surface interaction features |
| Ranker-derived confidence score | Ranker output margin between top-1 and top-2 within race indicates selection confidence | Low | Compute as `score_rank1 - score_rank2` per race; use as stake sizing signal |
| Conformal confidence as ranking feature | `conformal_confidence_score` from CQR model captures prediction interval width; tighter intervals = more reliable predictions | Low | Already computed in RacePredictor pipeline; add to InvestmentFeatureFrame |
| WinSelectionGate integration | Ranker replaces `win_market_selection_score` in sort order; gate policy (thresholds, filtering) remains | Medium | Ranker provides score column; WinSelectionGate applies learned thresholds on top |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Replacing entire WinSelectionGate | 1282 lines of evolved logic with runner-up detection, soft pass, market condition scoring; replacement is high risk | Ranker provides the ranking signal; gate policy still makes final selection |
| Neural ranking model | Training instability, hyperparameter sensitivity, no feature importance for diagnostics | LightGBM LambdaRank is mature, fast, interpretable |
| Pairwise ranking objective | O(n^2) per race with field sizes 8-18; computationally expensive | Pointwise LambdaRank is sufficient |
| Using raw odds as ranking target | Odds already embedded in market features; ranking by odds = replicating market | Rank by model+value signal; odds are input features, not targets |
| Multiple ranker models (win + value) | v2.1 scope is single ranker; adding value ranker doubles complexity | Start with single ranker targeting win probability; add value ranker post-MVP if needed |
| XGBoost/CatBoost ranker alternatives | One ranker implementation is sufficient; multiple creates maintenance burden | LightGBM LGBMRanker only |

---

## Component 4: Shadow Comparison Metrics

### What Already Exists

| Component | Location | Status |
|-----------|----------|--------|
| `compute_ece()` | `src/models/win_benter_gate.py` | Working. 10-bin Expected Calibration Error |
| `compare_calibrations()` | `src/models/win_benter_gate.py` | Working. Beta vs Isotonic with Brier + ECE |
| `generate_reliability_data()` | `src/models/win_benter_gate.py` | Working. Reliability diagram data for visualization |
| `BacktestEngine` | `src/backtest/engine.py` | Working. Full backtest with ROI, HR, DD metrics |
| `OOFHealthValidator` | `src/validation/oof_health_validator.py` | Working. OOF artifact validation with manifests |
| BT ROI tracking | `data/backtest/bt_*` | Historical ROI data from v1.5-v2.0 |

### Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Brier score (probability quality) | Standard calibration metric; missing means no calibration quality comparison | Low | `sklearn.metrics.brier_score_loss` on OOF predictions |
| Log-loss (probability quality) | Complementary to Brier; sensitive to confidence extremes | Low | `sklearn.metrics.log_loss` |
| ECE (probability quality) | Reveals WHERE mis-calibration occurs (which probability bins) | Low | `compute_ece()` already exists |
| Selection overlap rate | Measures how often new vs baseline ranker pick the same horse; >80% overlap = low disruption | Low | Compare `win_market_selection_score` top-1 per race between baseline and shadow |
| ROI comparison (betting performance) | Ultimate success metric; new ranker must not degrade ROI | Low | Both rankers run on same test period; compare realized ROI |
| Hit rate comparison | Complementary to ROI; less noisy at small sample sizes | Low | Compare win rate of selected horses |
| Max drawdown comparison | Risk metric; new ranker must not increase drawdown | Low | Compare cumulative drawdown profiles |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Closing Line Value (CLV) | Low-variance metric that measures prediction quality independent of outcome luck; model's odds vs closing odds | Medium | CLV = (model_odds - closing_odds) / closing_odds for selected horses; positive CLV indicates genuine edge |
| Pseudo-R2 (Benter) | `dR2 = R2_combined - R2_public` measures information added by fundamental model; Benter paper shows this correlates with profitability | Medium | Compute per-year; compare baseline vs shadow dR2 |
| Selection change analysis | When shadow picks a different horse, analyze WHY; which features changed the ranking | Medium | Log feature contributions for changed selections; compare ranker feature importance |
| Brier score decomposition | Decompose into reliability, resolution, uncertainty; identifies whether improvement comes from better calibration or better discrimination | Medium | Standard meteorological Brier decomposition |
| Regime-stratified metrics | Metrics broken down by RegimeDetector state; reveals if shadow helps specifically in conservative regime (known pain point) | Low | Group by regime; compute per-regime ROI/HR |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Deploying on ROI alone | ROI is noisy at sample sizes of 1000-3000 bets; a lucky streak should not trigger deployment | Use probability quality (Brier/ECE) as primary gate; ROI as secondary confirmation |
| A/B testing in production | Paper trading is the deployment model; live A/B is unnecessary complexity | Shadow comparison on historical data first; paper trading validation second |
| Metrics that require outcome data not in Parquet | All metrics must be computable from existing backtest data | Use only metrics derivable from race entries + odds + results |
| Comparing on different time periods | Fair comparison requires identical test periods | Run both baseline and shadow on identical 2024/2025 test data |
| Single aggregate metric | No single number captures calibration quality + selection quality + betting performance | Use 3-tier gate: probability quality AND selection overlap AND ROI |

---

## Feature Dependencies

```
Component 1: MarketAwareWinCalibrator
  |
  +--> Component 2: Segment Conditioning (provides segment keys for alpha_s/beta_s)
  |      |
  |      +--> Component 3: Race-Level Ranker (uses calibrated p_win_final as ranking feature)
  |             |
  |             +--> Component 4: Shadow Comparison (compares baseline vs new ranker)
  |
  +--> Component 4: Shadow Comparison (compares baseline vs new calibrator)

Existing infrastructure (from v2.0):
  OOFHealthValidator --> All components (OOF predictions for training)
  InvestmentFeatureFrame --> Components 1, 2, 3 (94-spec curated features)
  BenterCombination --> Component 1 (base logit blend to extend)
  WinSegmentCalibrator --> Component 2 (segment key pattern to reuse)
  WinSelectionGateModel --> Component 3 (gate policy to preserve)
```

Critical path: **Segment Conditioning definitions -> MarketAwareWinCalibrator -> Race-Level Ranker -> Shadow Comparison**

Components 1 and 2 are tightly coupled and should be implemented together. Component 3 depends on Component 1 producing `p_win_final` with segment-aware calibration. Component 4 depends on Component 3 producing ranking output.

### Within-Component Dependencies

```
MarketAwareWinCalibrator internal:
  BenterCombination.fit() (existing) --extend--> segment-conditioned fit
                                                  |
                                                  +--> Segment keys from WSC pattern
                                                  +--> Per-segment alpha/beta with Bayesian shrinkage
                                                  +--> Post-hoc isotonic or temp scaling (existing)
                                                  +--> Race normalization (existing in WinBenterGate)

Race-Level Ranker internal:
  InvestmentFeatureFrame.build_frame(mode="infer") --input--> LGBMRanker
                                                              |
                                                              +--> if_p_win_final (from calibrator)
                                                              +--> if_edge_rank_in_race (race-relative)
                                                              +--> if_ev_race_rank (race-relative)
                                                              +--> if_logit_gap (model-vs-market)
                                                              +--> if_field_ev_dispersion (race context)
                                                              +--> win_gate_score (from WinSelectionGate)
                                                              |
                                                              +--> Sort by race_id, compute group sizes
                                                              +--> Train with position-graded relevance
                                                              +--> Output: ranking score per horse

Shadow Comparison internal:
  Baseline pipeline --run--> metrics_baseline
  Shadow pipeline --run--> metrics_shadow
  Compare: Brier, ECE, CLV, ROI, HR, DD, selection overlap
  |
  +--> Deploy gate: probability quality PASS AND selection overlap >= 80% AND ROI not degraded
```

---

## MVP Recommendation

### Must-build first: Segment Conditioning + MarketAwareWinCalibrator

Prioritize:
1. Reuse WSC segment keys (surface|odds_band|rank_band|ev_band) -- Low
2. Extend BenterCombination with per-segment alpha/beta + Bayesian shrinkage -- High
3. OOF-based fitting via existing `generate_win_oof_predictions()` -- Medium
4. Post-hoc calibration (isotonic or Beta) via existing `compare_calibrations()` -- Low
5. Race normalization via existing `WinBenterGate.apply()` -- Low

Defer: Per-segment temperature scaling, calibration stability analysis, pseudo-R2 per segment

### Must-build second: Race-Level Ranker

Prioritize:
1. LGBMRanker with LambdaRank objective + group constraint -- Medium
2. InvestmentFeatureFrame as input (94-spec frame already built) -- Low
3. Position-graded relevance (1/finish_position) -- Medium
4. Integration with WinSelectionGate (ranker score replaces `win_market_selection_score`) -- Medium

Defer: Per-surface ranker, value ranker (2nd model), rank fusion, ranker-derived confidence score

### Must-build third: Shadow Comparison

Prioritize:
1. Brier + log-loss + ECE comparison on identical test period -- Low
2. Selection overlap rate (top-1 per race agreement) -- Low
3. ROI + HR + max DD comparison -- Low
4. Deploy gate logic (probability quality AND selection overlap AND ROI) -- Low

Defer: CLV computation, pseudo-R2 per segment, Brier decomposition, regime-stratified metrics, selection change analysis

### Post-MVP (future milestones)

- Per-surface ranker: Start with single ranker + surface features; split only if metrics demand it
- Value ranker (2nd model): Separate ranking by expected value; uncertain marginal gain
- Rank fusion: Requires both rankers; defer
- CLV computation: Requires closing odds vs model odds comparison infrastructure
- Per-segment temperature scaling: Adds complexity; only if global temp scaling proves insufficient
- Brier score decomposition: Nice-to-have diagnostic; not blocking
- Selection change analysis: Post-deployment diagnostic for understanding ranker behavior

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| MarketAwareWinCalibrator | HIGH | BenterCombination class already works; extending with segment conditioning is well-scoped; Benter (1994) provides theoretical foundation; WSC pattern proves segment conditioning works |
| Segment Conditioning | HIGH | WinSegmentCalibrator already implements identical segment keys with Bayesian shrinkage; just needs wiring to Benter parameters instead of prob_factor |
| Race-Level Ranker | MEDIUM | LambdaRank API verified via Context7 for LightGBM 4.6.0; race-specific tuning (relevance function, group sizes) needs empirical validation; integration with WinSelectionGate requires care |
| Shadow Comparison | HIGH | All required metrics (Brier, ECE, ROI, HR, DD) are standard and already partially implemented; comparison infrastructure is straightforward |
| ROI recovery to 100%+ | MEDIUM | v1.7 achieved 97.8% with different features; segment-aware calibration should recover this; but ROI depends on many factors beyond calibration quality |

## Sources

- Benter, A.W. (1994/2024 annotated). "Computer-Based Horse Race Handicapping and Wagering Systems." ActaMachina annotated edition. Tables 3-7 demonstrate logit-blend bias removal. Combined model removes bias completely. Pseudo-R2 comparison shows public-only R2 improved from 0.1325 (1986-1993) to 0.1863 (2016-2023). **HIGH confidence.**
- LightGBM LGBMRanker API. Context7 documentation for `lightgbm-org/lightgbm` v4.6.0. `objective='lambdarank'`, `group` parameter, `eval_at` for NDCG. **HIGH confidence.**
- Existing codebase: `src/models/benter_combination.py` (alpha/beta/gamma MLE with L-BFGS-B, beta floor 0.20). **HIGH confidence.**
- Existing codebase: `src/models/win_benter_gate.py` (WinBenterGate wrapping Benter + isotonic + temp + race normalization; `generate_win_oof_predictions` with walk-forward splits; `compare_calibrations` Beta vs Isotonic; `compute_ece` 10-bin). **HIGH confidence.**
- Existing codebase: `src/models/win_segment_calibrator.py` (WinSegmentCalibrator with ODDS_BINS/RANK_BINS/EV_BINS, Bayesian shrinkage prior_strength=500, min_segment_rows=120, max_deploy_factor=0.95). **HIGH confidence.**
- Existing codebase: `src/investment/schema_registry.py` (94 specs, 9 categories, InvestmentFeatureSpec frozen dataclass with dual-mode sources). **HIGH confidence.**
- Existing codebase: `src/investment/feature_frame.py` (InvestmentFeatureFrameBuilder with derived features: if_logit_gap, if_edge_rank_in_race, if_ev_race_rank, if_top3_gap, if_field_ev_dispersion). **HIGH confidence.**
- Existing codebase: `src/backtest/race_predictor.py` (get_win_candidates with win_market_selection_score formula, surface-aware base score, 6 penalty/bonus terms). **HIGH confidence.**
- Existing codebase: `src/models/win_selection_gate.py` (WinSelectionGateModel with walk-forward OOF scoring, threshold grid, runner-up detection). **HIGH confidence.**
- Snowberg & Wolfers (2010). "Explaining the Favorite-Longshot Bias." Evidence for probability misperception as primary cause. **MEDIUM confidence.**

---
*Feature research for: v2.1 MarketAware Calibration + Race-Level Ranker for ROI Recovery*
*Researched: 2026-05-27*
