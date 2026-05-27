# Feature Landscape: v2.0 Investment Pipeline Restructuring

**Domain:** Horse racing prediction / betting investment pipeline
**Researched:** 2026-05-27
**Scope:** 4 new components -- OOF Health, InvestmentFeatureFrame, MarketAwareWinCalibrator, Race-Level Ranker
**Supersedes:** v1.8 FEATURES.md (that document covers turf precision features; this covers investment pipeline restructuring)

---

## Executive Summary

The v2.0 milestone restructures the investment decision pipeline around four components. Unlike v1.8 (feature engineering for the turf model), v2.0 focuses on the post-model pipeline: how calibrated probabilities are combined with market information, how features are curated for investment decisions, and how horses are ranked within races for final bet selection.

The critical dependency chain is OOF Health (Phase 0) -> InvestmentFeatureFrame (Phase 1) -> MarketAwareWinCalibrator (Phase 2) -> Race-Level Ranker (Phase 4). Phase 3 (Segment Calibration) is absorbed into Phase 2 rather than standing alone. Every component after Phase 0 depends on reliable OOF predictions, making OOF health the single highest-priority deliverable.

---

## Component 1: OOF Health (Phase 0)

Infrastructure layer. All downstream components depend on reliable OOF predictions. This is not a user-facing feature but a pipeline integrity gate.

### Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| K-fold OOF prediction generation | Without OOF, every downstream component leaks training data into evaluation | Medium | Existing `predict_oof()` in MarketModel uses 5-fold KFold; replicate for win/place/wide hit models |
| OOF Brier score per fold | Standard calibration metric; missing means no calibration quality signal | Low | `sklearn.metrics.brier_score_loss` on OOF predictions vs actuals |
| OOF log-loss per fold | Complementary to Brier; sensitive to confidence extremes | Low | `sklearn.metrics.log_loss` |
| OOF ECE (Expected Calibration Error) | Brier/log-loss detect mis-calibration but not WHERE; ECE bins reveal it | Medium | 10-bin equal-mass ECE; no existing implementation in codebase |
| Fold-level consistency check | Wildly different per-fold scores indicate data leakage or distribution shift | Low | Std-dev of Brier across folds; flag if > threshold |
| Time-series OOF splits | Horse racing data has temporal dependencies; random KFold leaks future patterns into past | Medium | Use `TimeSeriesSplit` or year-based splits; existing MarketModel.predict_oof uses `shuffle=False` KFold which is acceptable but year-based is safer |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Per-surface OOF health | Turf vs dirt have different calibration profiles; aggregate masks problems | Low | Group by surface, compute metrics separately |
| Reliability diagram export | Visual diagnostic for calibration; invaluable for debugging | Medium | Matplotlib reliability diagram with perfect/calibrated/current curves |
| OOF drift detector | Year-over-year OOF quality degradation signals concept drift | Medium | Rolling window Brier trend; flag when slope exceeds threshold |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Retraining models to fix bad OOF | OOF health is diagnostic; retraining changes the model being evaluated | Log the issue, fix in next training cycle |
| OOF-based feature selection | Causes nested leakage without proper inner-loop CV | Use separate held-out set for feature decisions |
| Random shuffle KFold | Time-series horse racing data has temporal dependencies | Use TimeSeriesSplit or year-based splits |
| OOF health as deployment gate (block on threshold) | Too brittle; a single bad fold should not block entire pipeline | Report metrics; let human review decide |

---

## Component 2: InvestmentFeatureFrame (Phase 1)

Curated feature set (80-150 columns) that replaces ad-hoc column selection throughout the pipeline. This is a data organization layer, not a model.

### Table Stakes

| Feature Category | Column Count | Why Expected | Complexity | Notes |
|-----------------|-------------|--------------|------------|-------|
| Model probability columns | ~6 | p_win, p_place, p_hit, p_win_oof, p_place_oof are the primary model outputs | Low | Already exist; curate from existing columns |
| Market probability columns | ~3 | p_market_win_adj is the market view; essential for model-vs-market comparison | Low | Already exists from MarketModel |
| Model-vs-market gap features | ~4 | log_error, raw error, signed/abs versions; the core value signal | Low | Already exist (signed_log_error_win, abs_log_error_win) |
| Race-relative ranks | ~6 | rank of p_win within race; rank of EV within race; positional context | Low | Already computed in places; standardize |
| Odds band features | ~2 | Binned tanodds (1-2, 2-5, 5-10, 10-30, 30-100, 100+) | Low | Already in WinSegmentCalibrator ODDS_BINS |
| EV/edge columns | ~4 | win_selection_ev, win_selection_edge; the investment decision signal | Low | Already exist |
| Surface/condition indicators | ~3 | Surface, track_condition, distance_bin; fundamental context | Low | Already exist |
| Race-level features | ~6 | rl_log_odds_entropy, rl_odds_dispersion, etc.; race difficulty context | Low | Already exist in race_level_features.py |
| Market cross features | ~5 | Harville-based cross-checks; detect market mispricing | Low | Already exist in market_cross_features.py |
| OOF-derived columns | ~4 | p_win_final_oof, p_win_oof, win_selection_prob; the calibration backbone | Low | From Phase 0 OOF generation |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Late odds delta features | Late money movement is the strongest signal in horse racing; odds change from open to close captures smart money | Medium | Requires odds time-series data from `data/odds/time_series/`; column may need engineering |
| Harville place probability | Theoretical place probability from win probability; provides independent baseline for place betting | Medium | `p_place_harville = p_win + sum(p_2nd + p_3rd)` via Harville formula; known to be biased but useful as feature |
| Ability/form summary features | Standardized recent form composite; provides signal beyond raw model output | Medium | From horse_features.parquet; needs curation of which form columns to include |
| Segment calibration features | WinSegmentCalibrator factors (prob_factor, ev_factor, segment_key) as features for downstream models | Low | Already computed by WSC; add to frame |
| Feature provenance metadata | Track which model version produced each column; enables reproducibility and debugging | Low | Add version tag columns during feature frame assembly |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Raw p_market_pred in feature frame | MarketModel Rule 11: predicted market probability must not leak to Stage2 | Use only log_error derived features from MarketModel |
| One-hot encoding of categoricals | Bloats column count from 80-150 to 300+; tree models handle categoricals natively | Keep as category dtype; let LightGBM handle |
| Feature engineering inside frame assembly | Feature frame is a curation layer, not an engineering layer | Engineer features in feature modules; frame selects and organizes |
| More than 150 columns | Diminishing returns; increases overfitting risk and debugging complexity | Curate ruthlessly; every column must justify its inclusion |
| Including POST_RACE columns | Leakage risk even in curation; kakuteijyuni, time, harontime are outcomes | Strict pre-race-only curation policy |

---

## Component 3: MarketAwareWinCalibrator (Phase 2)

Benter-type logit blending of fundamental model probability with market probability, extended with segment calibration features. This is the core value proposition of v2.0.

### Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Logit-blend core (alpha, beta, gamma) | Benter (1994) foundation: `logit(p_combined) = alpha * logit(p_fund) + beta * logit(p_market) + gamma` | Medium | BenterCombination class already exists with MLE fitting via scipy.optimize.minimize (L-BFGS-B) |
| Beta floor (market weight >= 0.20) | Market is strongly informative; zero weight on market is never correct | Low | Already enforced in existing BenterCombination (beta lower bound = 0.20) |
| OOF-based parameter fitting | In-sample fitting leaks; parameters must be fit on held-out predictions | Medium | Use OOF predictions from Phase 0 as input to MLE |
| Probability clipping | Logit of 0 or 1 is undefined; numerical safety | Low | Clip to [0.01, 0.99] before logit transform |
| Post-hoc temperature scaling | Benter blend may be over/under-confident in aggregate; temperature scaling corrects this | Low | TemperatureScaling class already exists; reuse |
| Surface-aware fitting | Turf and dirt have different market efficiency profiles; separate parameters needed | Low | Fit separate alpha/beta/gamma per surface (same pattern as existing submodel split) |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Segment-conditioned calibration | Different odds bands and EV bands have different reliability; one global alpha/beta is suboptimal | High | Absorb WSC segment logic (surface\|odds_band\|rank_band\|ev_band) as conditioning features |
| Odds-band interaction terms | Favorite-longshot bias means low-odds and high-odds horses need different calibration | Medium | Add odds_band * logit(p_fund), odds_band * logit(p_market) interaction features to the blend |
| EV-band conditioning | High-EV horses are where mispricing concentrates; calibrator should be more aggressive there | Medium | Add EV band as conditioning feature for alpha/beta |
| Per-fold calibration stability | If alpha/beta swing wildly across OOF folds, the fit is unstable and likely overfits | Medium | Compute parameter variance across folds; flag if coefficient of variation > 0.3 |
| Benter residual features | `logit(p_actual) - logit(p_combined)` per segment; identifies WHERE the combined model still fails | Medium | Requires OOF actuals; segment-level analysis similar to existing WSC training |
| LogisticRegression as alternative | LogisticRegression directly models the Benter formula with regularization; may be more stable than MLE | Medium | `sklearn.linear_model.LogisticRegression` with `logit(p_fund)` and `logit(p_market)` as features |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Neural network calibrator | Overkill for 3-7 parameter model; harder to interpret and debug | Stick with parametric logit blend + temperature scaling |
| Dropping BenterCombination entirely | Existing class has working MLE fitting; rewrite wastes time and loses tested code | Extend the existing class with segment conditioning |
| Market probability as direct input to Stage2 | Market probability is already captured via log_error features; duplicate signal harms tree models | Calibrator outputs calibrated p_win; market signal stays in log_error features |
| Regime-conditional calibration | RegimeDetector is explicitly excluded from v2.0 scope per PROJECT.md | Use surface/odds_band/EV_band conditioning instead of regime |
| More than 2 blend parameters per segment | With ~50+ segments, each additional parameter exponentially increases overfitting risk | Alpha + beta + gamma per segment maximum; use Bayesian shrinkage toward global values |
| Beta calibration as primary model | BetaCalibration (betacal package) is a post-processing calibrator, not a full combination model | Use TemperatureScaling for post-hoc; Benter blend for combination |

---

## Component 4: Race-Level Ranker (Phase 4)

LightGBM LambdaRank model that ranks horses within a race for final bet selection. Replaces the current 41-column scoring in WinSelectionGate with a learned ranking model.

### Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| LambdaRank objective | Standard learning-to-rank for pointwise/group scoring; proven for race ranking | Medium | `lgb.LGBMRanker` with `objective='lambdarank'`; verified available in installed LightGBM 4.6.0 |
| Group parameter (horses per race) | LambdaRank requires group sizes; missing this causes silent incorrect training | Medium | `df.groupby('race_id').size().values` as group array; data MUST be sorted by race_id first |
| NDCG evaluation metric | Standard ranking metric; needed for early stopping and model comparison | Low | `eval_at=[1, 3, 5]` for top-k relevance |
| Win ranker (1st model) | Rank horses by probability of winning; primary ranking signal | Medium | Target = binary win indicator or graded relevance by finish position |
| InvestmentFeatureFrame as input | Curated feature set ensures consistent, complete features | Low | Direct consumer of Phase 1 output |
| OOF-based ranker evaluation | In-sample ranking metrics are meaningless; must evaluate on held-out folds | Medium | Use OOF predictions from Phase 0; compute per-fold NDCG |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Value ranker (2nd model) | Separate ranker for expected value; win probability and value are different objectives | High | Target = `EV = p_win * odds`; rank by investment attractiveness |
| Rank fusion (win + value) | Win ranker selects likely winners; value ranker selects profitable bets; fusion captures both | High | Weighted combination or learned fusion; requires validation |
| Position-graded relevance | 1st place more important than 2nd; relevance should be `1/finish_position` or exponential decay | Medium | Standard in learning-to-rank; better than binary win/loss |
| Calibrated probability as ranking feature | MarketAwareWinCalibrator's `p_win_market_aware` provides a superior probability signal for ranking | Low | Requires Phase 2 completion before ranker training |
| Per-surface ranker training | Turf and dirt races have different ranking dynamics | Medium | Train separate rankers OR include surface interaction terms |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Pairwise ranking objective | O(n^2) per race; computationally expensive and hard to debug | Use pointwise LambdaRank; sufficient for field sizes of 8-18 |
| Neural ranking model | Training instability, hyperparameter sensitivity, no interpretability | LightGBM LambdaRank is mature, fast, interpretable via feature importance |
| Replacing the entire WinSelectionGate | Gate has 1300+ lines of evolved logic; replacing all at once is high risk | Ranker provides ranking signal; gate policy still makes final selection |
| Ranking without group constraint | Each horse must be ranked within its own race; global ranking is meaningless | Always group by race_id; sort by race_id before training |
| Using raw odds as ranking target | Odds already embedded in market features; ranking by odds = replicating market | Rank by model+value signal; odds are input features, not targets |
| XGBoost/CatBoost ranker alternatives | One ranker implementation is enough; adding more creates maintenance burden without proven benefit | LightGBM LGBMRanker is the only one needed |

---

## Feature Dependencies

```
Phase 0: OOF Health
  |
  +--> Phase 1: InvestmentFeatureFrame (uses OOF-derived columns)
  |      |
  |      +--> Phase 2: MarketAwareWinCalibrator (uses Frame columns + OOF for fitting)
  |      |      |
  |      |      +--> Phase 4: Race-Level Ranker (uses Frame + calibrated probabilities)
  |      |
  |      +--> Phase 4: Race-Level Ranker (uses Frame as primary input)
  |
  +--> Phase 2: MarketAwareWinCalibrator (OOF predictions for parameter fitting)
```

Critical path: **OOF Health -> InvestmentFeatureFrame -> MarketAwareWinCalibrator -> Race-Level Ranker**

Phase 3 (Segment Calibration integration) is NOT a separate phase in this dependency graph -- it is absorbed into Phase 2 as features within MarketAwareWinCalibrator.

### Within-Component Dependencies

```
MarketAwareWinCalibrator internal:
  BenterCombination (existing) --extend--> segment conditioning
                                           |
                                           +--> WSC segment keys (surface|odds|rank|ev)
                                           +--> Per-segment alpha/beta/gamma
                                           +--> Temperature scaling post-hoc

InvestmentFeatureFrame internal:
  Existing feature columns --curate--> 80-150 column frame
                                        |
                                        +--> Model prob columns (existing)
                                        +--> Market prob columns (existing)
                                        +--> Model-vs-market features (existing)
                                        +--> Race-level features (existing, 6 cols)
                                        +--> Market cross features (existing, 5 cols)
                                        +--> Late odds features (may need engineering)
                                        +--> Harville place probability (new, medium)
                                        +--> Segment calibration features (from WSC)

Race-Level Ranker internal:
  InvestmentFeatureFrame --input--> LGBMRanker
                                     |
                                     +--> Win ranker (target: finish position)
                                     +--> Value ranker (target: EV)
                                     +--> Rank fusion (combine both)
```

---

## MVP Recommendation

### Phase 0 -- OOF Health (MUST be first, blocking)

Prioritize:
1. Time-series OOF generation for all hit models (win/place/wide) -- Medium
2. Per-fold Brier + log-loss reporting -- Low
3. 10-bin ECE implementation -- Medium
4. Year-stratified OOF (not random shuffle) -- Medium

Defer: Reliability diagrams, OOF drift detector (nice-to-have; requires historical baseline)

### Phase 1 -- InvestmentFeatureFrame (MUST be second, blocking)

Prioritize:
1. Curate existing columns into formal frame (model prob, market, gap, ranks, odds bands) -- Low
2. Add race-level features (6 cols) and market cross features (5 cols) -- Low
3. Add segment calibration features from WSC -- Low

Defer: Harville place probability (Medium; useful but not blocking), Late odds delta features (Medium; requires data engineering)

### Phase 2 -- MarketAwareWinCalibrator (depends on Phase 0 + 1)

Prioritize:
1. Extend BenterCombination with OOF-based fitting -- Medium
2. Surface-aware parameter fitting -- Low
3. Absorb WSC logic as calibrator features -- High
4. Post-hoc temperature scaling -- Low

Defer: Per-fold calibration stability analysis, Odds-band interaction terms (can add after core works)

### Phase 4 -- Race-Level Ranker (depends on Phase 1 + 2)

Prioritize:
1. Win ranker with LambdaRank + position-graded relevance -- Medium
2. Group constraint enforcement (sort by race_id) -- Medium
3. OOF-based NDCG evaluation -- Medium

Defer: Value ranker (2nd ranker model), Rank fusion, Per-surface ranker (start with single ranker + surface features)

### Post-MVP (future milestones)

- Value ranker (2nd ranker model): High complexity, uncertain marginal value over win ranker + EV threshold
- Rank fusion: Requires both rankers; defer
- Reliability diagrams: Nice-to-have visualization
- OOF drift detector: Requires multiple training cycles of history
- Per-surface ranker: Start with single ranker + surface features; split only if metrics demand it
- LogisticRegression alternative to MLE Benter: Worth evaluating post-MVP

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| OOF Health features | HIGH | Standard ML practice; well-documented metrics; existing codebase patterns |
| InvestmentFeatureFrame features | HIGH | Mostly curating existing columns; limited new engineering required |
| MarketAwareWinCalibrator features | HIGH | Benter (1994) paper provides theoretical foundation; existing BenterCombination class provides implementation base; Tables 3-7 demonstrate the approach works |
| Race-Level Ranker features | MEDIUM | LambdaRank API well-documented via Context7 but race-specific tuning patterns need empirical validation |
| Segment conditioning design | MEDIUM | WSC segment logic exists but absorbing it into calibrator is an architectural change with integration risk |
| Benter formula effectiveness | HIGH | Benter paper shows combined model removes fundamental model bias; pseudo-R2 analysis confirms delta-R2 is the key profitability metric |

## Sources

- Benter, A.W. (1994/2024 annotated). "Computer-Based Horse Race Handicapping and Wagering Systems." ActaMachina annotated edition. Tables 3-7 demonstrate logit-blend bias removal. Combined model removes bias completely. Pseudo-R2 comparison shows public-only R2 improved from 0.1325 (1986-1993) to 0.1863 (2016-2023) -- markets getting more efficient. Delta-R2 (combined - public) is the key profitability metric. **HIGH confidence.**
- LightGBM LGBMRanker API. Context7 documentation for `lightgbm-org/lightgbm` v4.6.0. `objective='lambdarank'`, `group` parameter, `eval_at` for NDCG positions, `eval_group` for validation groups. **HIGH confidence.**
- Existing codebase: `src/models/benter_combination.py` (BenterCombination class with alpha/beta/gamma MLE fitting, TemperatureScaling post-hoc). **HIGH confidence.**
- Existing codebase: `src/models/win_segment_calibrator.py` (WinSegmentCalibrator with segment keys, Bayesian shrinkage, OOF-based training). **HIGH confidence.**
- Existing codebase: `src/models/market_model.py` (MarketModel with log_error computation, Rule 11 for p_market_pred exclusion). **HIGH confidence.**
- Existing codebase: `src/models/win_selection_gate.py` (1302-line gate model, 41-column final selection). **HIGH confidence.**
- ROI_IMPROVEMENT_PLAN.md (project internal). Phase definitions, feature lists for all 4 components. **HIGH confidence.**
- PROJECT.md (project internal). v2.0 milestone scope, constraints (no bet count reduction, no regime dependency, deployment on probability quality not ROI). **HIGH confidence.**

---
*Feature research for: v2.0 Investment Pipeline Restructuring -- 4 components*
*Researched: 2026-05-27*
