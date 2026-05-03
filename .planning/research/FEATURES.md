# Feature Landscape: v1.1 ROI Advanced Model

**Domain:** Horse racing win prediction -- ensemble stacking, odds deviation EV, time-series features
**Researched:** 2026-05-03
**Confidence:** HIGH (existing codebase audit + academic literature + community practice)

## Context

This document covers ONLY new features for v1.1 milestone. The existing 14-module, 100+ column feature engine (documented in v1.0 FEATURES.md) is the foundation. Five new capability areas are researched:

1. 3-model stacking (LightGBM + XGBoost + CatBoost)
2. Odds deviation EV (model-vs-market probability gap exploitation)
3. Odds time-series features (change patterns as predictive signals)
4. Time-series features from past runs (temporal performance patterns)
5. Pace/position prediction (race dynamics modeling)

---

## Table Stakes

Features/capabilities that any competitive system at this level MUST have. Missing these means the model is fundamentally behind.

### A. Ensemble Stacking

| Feature | Why Expected | Complexity | Existing Status | Notes |
|---------|--------------|------------|-----------------|-------|
| **3-model base learners** (LightGBM + XGBoost + CatBoost) | Single GBM models have systematic biases. Different GBM implementations split differently, handle categoricals differently, and regularize differently. Combining them reduces model variance. Lessmann et al. (2020) showed stacking multiple GBMs consistently outperforms any single GBM in prediction tasks. | Medium | StackedEnsemble class EXISTS in `src/models/stacked_ensemble.py`. Already integrated into training pipeline via `use_ensemble` flag. Uses K-fold OOF + Ridge meta-learner. | Code is complete but uses fixed hyperparameters (lr=0.03, leaves=31, rounds=300). Needs hyperparameter diversity and tuning for win-specific optimization. |
| **Out-of-fold (OOF) meta-learner training** | Training the meta-learner on the same data as base models causes information leakage, making the stacking layer useless. OOF predictions ensure the meta-learner sees only held-out predictions. | Medium | DONE -- 3-fold expanding window OOF in StackedEnsemble.train() | Current expanding window (not random shuffle) correctly respects time-series ordering. |
| **Simple meta-learner** (Ridge/LogisticRegression) | Complex meta-learners (e.g., another GBM) overfit on 3 features (the base model predictions). Research consensus (Springer 2024, Kaggle competitions) shows Ridge/ElasticNet outperform complex meta-learners because the feature space is tiny. | Low | DONE -- Ridge(alpha=1.0) as meta-learner | Good choice. Could consider ElasticNet for automatic feature selection, but with only 3 features Ridge is optimal. |
| **Categorical encoding consistency** | XGBoost and CatBoost cannot handle pandas categorical columns directly. Must encode consistently between training and inference. | Low | DONE -- `_encode_cats()` converts categoricals to numeric codes | CatBoost has native categorical support but the current approach (pre-encoding) is simpler and correct. |

### B. Odds Deviation EV

| Feature | Why Expected | Complexity | Existing Status | Notes |
|---------|--------------|------------|-----------------|-------|
| **Model-vs-market probability gap** (p_model - p_market) | The core "edge" signal. Positive EV only exists where the model disagrees with the market. Benter (1994) and all subsequent racing ML papers use this as the primary bet selection criterion. | Low | PARTIALLY EXISTS -- MarketModel computes `signed_log_error_win` and `abs_log_error_win` (log-space residuals). WinBenterGate blends p_fundamental with p_market. However, a direct probability gap at the final prediction level is not computed. | The system already has the ingredients (p_ability_win, p_market_win_adj, p_win_final from BenterGate). What is missing is a clean `p_model_final - p_market_implied` residual at the point where betting decisions are made. |
| **EV = p_model * odds - 1** | The fundamental value betting formula. If EV > 0, the bet has positive expected value. This is not a prediction feature but a decision feature -- it determines whether to bet and how much. | Low | PARTIALLY EXISTS -- `ev_win` is computed in WinTwoStageModel. `win_selection_ev` is built in WinSelectionGate. | Already functional but may need refinement for the stacking model's output distribution. |

### C. Odds Time-Series

| Feature | Why Expected | Complexity | Existing Status | Notes |
|---------|--------------|------------|-----------------|-------|
| **Odds change rate** (early vs late snapshot) | Late odds movements reveal informed money flow. Academic research (Edith Cowan University study on Australian racing) shows late money moves have significant predictive value beyond closing odds. | Medium | DONE -- `odds_drop_rate_60_10`, `odds_drop_rate_30_10` in `odds_dynamics_features.py` | Uses t-60min and t-30min vs t-10min snapshots. Implementation is solid. |
| **Odds velocity** (linear regression slope) | Measures the trend direction and speed of odds movement. Steady decline = sustained smart money; sudden spike = injury/scratch rumor. | Medium | DONE -- `odds_velocity` computed via vectorized linear regression slope | Good implementation. Could add acceleration (2nd derivative) for richer signal. |
| **Odds volatility** (change magnitude) | High volatility = uncertain market = potential inefficiency. Low volatility = market consensus = harder to find edge. | Low | DONE -- `odds_volatility` as std of consecutive changes | Already used by RegimeDetector. |
| **Popularity rank change** | Shifts in predicted finishing order (based on odds) reveal which horses smart money is moving toward. | Low | DONE -- `popularity_change_30_10` | Simple t-30 to t-10 popularity shift. |

### D. Past-Run Time-Series

| Feature | Why Expected | Complexity | Existing Status | Notes |
|---------|--------------|------------|-----------------|-------|
| **Recent form trend** (linear slope of past finishes) | The most basic temporal pattern. Improving form = higher win probability. Declining form = lower. This is Benter's 3rd pillar. | Low | DONE -- `form_trend` in `form_cycle_features.py` (linear regression on normalized finishes) | Negative of the slope (positive = improving). Solid. |
| **Late vs early form comparison** (last 2 vs first 3) | Captures acceleration or deceleration in performance. The "closing kick" of a horse's form cycle. | Low | DONE -- `harontime_late_trend` (last 2 - first 3 avg; negative = improving) | Simple but effective. |
| **Corner position progression** (1C to 4C to finish) | Shows running style. Front-runners (low 1C) vs closers (high 1C, low finish). This determines pace aptitude. | Medium | DONE -- `jyuni1c_avg`, `jyuni4c_avg`, `closing_index_avg`, `pace_aptitude`, `front_pace_wr`, `closing_pace_wr` | Comprehensive implementation. `closing_index_avg` = (norm_4C - norm_finish) captures positional gain. |

---

## Differentiators

Features/capabilities that provide competitive edge beyond table stakes. These are where ROI improvement from 89% to 100%+ will come from.

### HIGH IMPACT -- Ensemble Enhancement

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Hyperparameter diversity across base models** | The current StackedEnsemble uses identical hyperparameters for all 3 models (lr=0.03, leaves/depth 6/31, 300 rounds). This defeats the purpose of stacking -- diversity is what makes stacking work. XGBoost should have different tree depth, CatBoost should leverage its ordered boosting, LightGBM should use leaf-wise growth. Research (Medium stacking guide, PMC study on GBM ensembles) shows diversity accounts for 60-80% of stacking improvement. | Low | StackedEnsemble code (already exists) | Simple code change: different params per model. LightGBM: leaf-wise, num_leaves=31. XGBoost: depth=4 (shallower), lr=0.05. CatBoost: depth=8 (deeper), lr=0.02, ordered boosting. |
| **Early stopping per base model** | Currently all models train for fixed 300 rounds. Different models converge at different speeds. Over-trained models memorize noise, under-trained models miss signal. Early stopping with a validation set ensures each model trains to its optimal point. | Low | StackedEnsemble code + validation split | Each base model should use the same validation fold and stop independently. |
| **Feature subsampling per base model** | Each base model should see a slightly different feature set. LightGBM uses `feature_fraction`, XGBoost has `colsample_bytree`, CatBoost has `rsm`. This forces models to learn different aspects of the data, improving ensemble diversity. | Low | StackedEnsemble code | Set feature_fraction=0.7 for LightGBM, colsample_bytree=0.8 for XGBoost, rsm=0.8 for CatBoost. |
| **Stacking at Stage1 (ability) level** | The current stacking replaces only the hit_model in WinTwoStageModel. But Stage1 (AbilityModel) is a Ranker, not a classifier. Stacking the Ranker (3 rankers -> meta ranker) would improve the fundamental ability estimate that flows into all downstream models. | High | AbilityModel training, requires Ranker-specific stacking (different from binary stacking) | Complex because Ranker output is a score, not a probability. Would need Ranker-specific OOF and a regression meta-learner. Defer to Phase 2 of v1.1. |

### HIGH IMPACT -- Odds Deviation Features

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Final EV residual** (p_win_final * tanodds - 1) | The definitive edge measure. After all model corrections (Benter blend, calibration, temperature scaling), how much does the model's final estimate disagree with the market? This is what should drive bet/not-bet decisions. Currently the system computes `ev_win` and `edge_win` but they may not incorporate the full stacking + Benter pipeline correctly. | Low | p_win_final (from BenterGate), tanodds | Validate that win_selection_ev correctly incorporates stacking output. If the stacking model produces different probability distributions than LightGBM alone, the downstream EV pipeline needs recalibration. |
| **Odds-to-ability ratio** (p_market / p_ability) | EXISTS from v1.0 as a feature idea. This ratio directly measures market inefficiency per horse. High ratio = market undervalues the horse (potential overlay). Low ratio = market overvalues (potential underlay). This is the single most important feature for ROI because it directly measures the betting edge. | Low | p_ability_win (Stage1), p_market_win_adj | Already conceptually designed. Needs implementation as a Stage2 feature column. Compute: `p_market_win_adj / p_ability_win.clip(0.01, 0.99)`. |
| **Edge confidence interval** | Rather than a point estimate of EV, compute a confidence interval using the conformal prediction infrastructure (already built in `robust_confidence_estimator.py`). A horse with EV=1.05 and narrow CI [1.02, 1.08] is a much better bet than EV=1.10 with CI [0.8, 1.4]. | Medium | RobustConfidenceEstimator (exists), p_win_final, tanodds | The conformal infrastructure already produces prediction intervals. What is missing is translating those into EV intervals and using them for bet selection. |
| **Kelly-optimal stake from stacked probability** | The Kelly criterion: f* = (b*p - q) / b, where b = (odds - 1), p = model probability, q = 1-p. Use the stacking ensemble's probability in Kelly formula to determine optimal bet size. This directly converts improved probability estimates into ROI improvement via better sizing. | Medium | p_win_final (from stacking), tanodds, bankroll state | The system has StakeCalculator but it may not use full Kelly from stacked probabilities. Verify and enhance. |

### HIGH IMPACT -- Odds Time-Series Enhancement

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Odds acceleration** (2nd derivative) | Currently only velocity (1st derivative) is computed. Acceleration captures whether odds movement is accelerating (strong steam) or decelerating (stabilizing). Strong steam moves near post time are the most predictive signals in pari-mutuel markets. | Low | odds_velocity (exists), raw odds time-series | Compute: diff(odds_velocity) or fit quadratic regression. Simple addition to existing `odds_dynamics_features.py`. |
| **Late money intensity** (t-5min vs t-10min movement) | The current system uses t-10min as the latest snapshot. But the LAST 5 minutes before post are where insider money hits the pool. A t-5min vs t-10min comparison captures "late money" specifically. | Medium | odds time-series with finer granularity, requires snapshots within 5 min of post | Depends on data availability. If `data/odds/time_series/` has sub-10-minute granularity, this is straightforward. If not, need to verify minimum snapshot interval. |
| **Odds movement consistency** (direction persistence) | A horse whose odds steadily decline across ALL time windows (60->30->10) is a stronger signal than one whose odds bounce around but end lower. Measure the fraction of consecutive windows where the direction is consistent. | Low | Existing odds time-series data | Compute: count(consistent_direction) / total_transitions. Binary feature: is the movement consistently in one direction? |
| **Volume-weighted odds movement** | Large odds movements on low volume (few bets) are noise. Small movements on high volume (many bets) are signal. If bet volume data is available, weight odds changes by volume. | Medium | Requires bet volume data per snapshot -- may not exist in EveryDB2 | NEEDS VERIFICATION: Check if `s_odds_tanpuku` time-series data includes volume (購入件数 or similar). If not available, this is not feasible. |
| **Race-level odds volatility trend** | Existing `compute_rolling_volatility()` computes race-level volatility rolling mean. Enhance with volatility trend (is the market becoming more or less uncertain about THIS race specifically?). | Low | compute_rolling_volatility (exists) | Simple: rolling_volatility(current_race) - rolling_volatility(rolling_mean). Positive = market getting more uncertain = more potential inefficiency. |

### HIGH IMPACT -- Past-Run Time-Series Enhancement

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Exponential decay weighting on past performances** | Currently the system uses simple averages (e.g., `harontimel5_avg` = mean of last 5). But recent races should matter more than old ones. Exponential decay weighting (e.g., weights = [0.4, 0.25, 0.15, 0.12, 0.08] for 5 past races) gives more emphasis to recent form. | Low | All existing history features that use averages | Drop-in enhancement: replace `mean()` with `np.average(weights=decay_weights)`. Affects: harontimel5_avg, timediff_avg, jyuni1c_avg, jyuni4c_avg, closing_index_avg. |
| **Performance vs similar-class opposition** | A horse's last-5 finishes are more meaningful when contextualized by class. A 3rd-place finish in a G1 is better than a 1st-place in a maiden claimer. Compute a class-adjusted form metric: form_trend * (class_level_avg_of_recent_races / current_class_level). | Medium | form_trend (exists), gradecd/jyokencd1 from history | Requires computing class_level for each past race (already in `horse_history_features.py` as `_class_level_from_values()`). |
| **Sectional time decomposition** (early pace vs closing speed) | Currently `harontimel3` (3-furlong closing time) is used as a single metric. Decompose into: early pace (first half of race) and closing speed (final 3f). Horses that close fast regardless of early pace are consistently undervalued. | Medium | jyuni1c, jyuni4c, harontimel3 from history | Compute: closing_speed = harontimel3_zscore; early_pace = (jyuni4c_avg - jyuni1c_avg) / distance. The ratio closing_speed/early_pace identifies true closers vs front-runners. |
| **Recency-weighted class move** | The existing `class_move` (current - previous) uses only the last start. But a horse that has been climbing classes over the last 3 starts (gradual improvement) is different from one that suddenly jumps. Compute: weighted_class_trend over last 3 starts. | Low | class_move logic (exists), history data | Simple: compute class_level for last 3 starts and fit a slope (similar to form_trend but for class). |
| **Improvement trajectory** (z-score trend) | The existing `harontime_late_trend` compares last-2 vs first-3 raw values. A better approach: compute z-scores (already done via expanding stats) and fit a linear trend to the z-scores. This normalizes for changing track conditions across races. | Medium | harontimel5_zscore (exists), expanding_stats infrastructure | Compute: linear regression slope of z-scores of last 5 races. Positive = improving in normalized terms. This is what form_trend should have been for timing data. |

### HIGH IMPACT -- Pace/Position Prediction

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Expected pace figure per horse** | Assign each horse a numerical pace figure from past sectional data (jyuni1c, jyuni4c, harontimel3). Currently pace_pressure uses only declared running style (kyakusitukubun_cd). Actual timing data provides a much richer signal. | High | jyuni1c, jyuni4c, harontimel3 from history | Requires computing a composite pace figure per horse: `pace_figure = weighted(jyuni1c_avg, jyuni4c_avg, closing_index_avg)`. Then project race pace from all entrants' pace figures. |
| **Projected position at each corner** | Based on the horse's past corner positions AND the race's pace scenario (who else is front-running), project where each horse will be at 1C, 4C. This is a mini-model within the feature pipeline. | High | jyuni1c/jyuni4c from history, field composition | Complex: requires modeling field-level interactions. Defer to Phase 2 of v1.1 if ROI target not met by simpler features. |
| **Pace scenario match score** | Given the projected race pace (from above) and the horse's running style, compute how well the horse's style fits the scenario. Front-runners in slow-paced races have high match scores. | Medium | Expected pace figure, kyakusitukubun_cd | Simpler version already exists as `pace_scenario_fit` in interaction_features.py. Enhance with actual timing data rather than just declared style. |

---

## Anti-Features

Features/capabilities to explicitly NOT build. These either introduce leakage, waste effort, or degrade performance.

### Ensemble Anti-Patterns

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Complex meta-learner (GBM/neural net)** | With only 3 features (base model predictions), a complex meta-learner will overfit. Research consistently shows Ridge/ElasticNet outperforms complex meta-learners for stacking with few features. | Keep Ridge(alpha=1.0). Consider ElasticNet only if feature count grows beyond 5. |
| **Random shuffle K-fold for OOF** | Horse racing data has temporal structure. Random shuffling causes lookahead bias -- training on future data to predict past races. The current expanding window approach is correct. | Keep the current expanding window OOF scheme. Never use random KFold. |
| **Stacking ALL model stages** | Stacking Stage1 (Ranker), Stage2 (hit binary), and Stage3 (EV correction) all separately would multiply training time by 9x (3 models x 3 stages). Focus stacking on the highest-leverage point: the hit model in Stage2. | Stack only the hit_model (binary) in WinTwoStageModel. Consider Stage1 stacking only if ROI target not met. |
| **Ensemble model as input to itself** | Using ensemble predictions as features for the same ensemble creates circular dependencies. | Each model stage uses the PREVIOUS stage's output, never its own. |
| **Equal weighting fallback** | Averaging the 3 base models without a meta-learner (simple average) throws away the information about which model is better for which predictions. | Always use the Ridge meta-learner, even if the coefficients end up close to uniform. |

### Odds Feature Anti-Patterns

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Confirmed/final odds as features** | kakuteitandds (final odds) are only available after the race. Using them is pure data leakage. The model would appear to work perfectly in backtest but fail in live betting. | Use ONLY pre-race odds snapshots (tanodds from t-10min or earlier). The system already enforces this. |
| **Raw EV as model input feature** | EV = p * odds - 1 includes the target variable (odds correlates with outcome). Feeding EV back into the model as a feature creates a feedback loop. | Use probability gap (p_model - p_market) as a feature, not EV. EV is for bet SELECTION, not prediction. |
| **Over-rounding the probability** | Clipping the probability gap to a narrow range (e.g., [-0.05, 0.05]) removes the signal. The model needs to see the full range of disagreement to learn the nonlinear relationship between edge and value. | Use log-space residuals (as in market_log_error_win) or full-range probability gap. Clip only for numerical stability (0.01 to 0.99). |
| **Odds movement from post-race** | Any odds snapshot taken after the race started is contaminated. | Use only snapshots with happyotime BEFORE hassotime (post time). The system already does this via _mins_before_anchor calculation. |

### Time-Series Anti-Patterns

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Future sectional times** | Using harontimel3 from the CURRENT race (the race being predicted) is leakage. The sectional times are only available after the race. | Use ONLY harontimel3 from PAST races. The system already enforces this via PIT (point-in-time) filtering with `race_date < target_date`. |
| **LSTM/transformer for temporal modeling** | With typically 5-15 past starts per horse, deep sequence models will overfit badly. The data is too sparse for recurrent architectures. | Use handcrafted temporal features (slopes, trends, decay-weighted averages) computed from the existing per-start data. |
| **All-time career averages** | A horse's career from 3 years ago is irrelevant to its current ability. Career averages dilute recent signal with stale data. | Use rolling windows (last 5 starts) with exponential decay weighting. The system already uses last-5 windows. |
| **Too-granular time-series features** | Computing features per individual past start (e.g., 5 separate columns for each of the last 5 starts) creates high-dimensional sparse features that GBMs handle poorly. | Use aggregated statistics (mean, trend, best, worst, std) over the window. The system already does this correctly. |
| **RNN-style positional prediction** | Building a separate RNN model to predict corner positions adds massive complexity for marginal gain. The feature engineering approach (past corner averages) captures most of the signal. | Use statistical pace features from history. Consider a separate pace model only if simpler features prove insufficient. |

### Pace Prediction Anti-Patterns

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Real-time pace model using same-day data** | Requires results from earlier races on the same card, which introduces dependency on race schedule and creates data availability issues for early races. | Use historical pace data only (past starts). The projected pace is based on the field's historical running styles. |
| **Over-engineered positional simulation** | Monte Carlo simulation of race positions at each furlong marker is computationally expensive and fragile. Small errors in pace estimation compound through the simulation. | Use simple projected pace figure per horse + field-level pace pressure (already exists). Reserve simulation for v1.2+ if needed. |
| **Weather-based pace adjustment** | Detailed weather features (wind speed, temperature) are not in EveryDB2 and would require external data. | Track condition code (baba_cd) already captures the ground condition effect on pace. |

---

## Feature Dependencies

```
NEW v1.1 Feature Dependencies:

Ensemble Stacking
  requires: StackedEnsemble class (DONE)
  requires: XGBoost, CatBoost in dependencies (DONE)
  requires: use_ensemble flag in training pipeline (DONE)
  enhancement: hyperparameter diversity (NEW, low effort)
  enhancement: early stopping per base model (NEW, low effort)
  enhancement: feature subsampling per base model (NEW, low effort)

Odds Deviation EV
  requires: p_win_final from BenterGate (DONE)
  requires: tanodds from odds snapshots (DONE)
  requires: ev_win computation in WinTwoStageModel (DONE)
  new: odds-to-ability ratio as Stage2 feature column
  new: edge confidence interval from conformal prediction
  new: Kelly stake sizing from stacked probability

Odds Time-Series Enhancement
  requires: data/odds/time_series/ parquet files (DONE, 2015-2025)
  requires: compute_odds_dynamics() in odds_dynamics_features.py (DONE)
  new: odds acceleration (2nd derivative)
  new: late money intensity (t-5 vs t-10, if data permits)
  new: direction consistency metric
  verify: snapshot granularity (is sub-10-min data available?)

Past-Run Time-Series Enhancement
  requires: horse_history_features.py infrastructure (DONE)
  requires: expanding_stats for z-scores (DONE)
  new: exponential decay weighting (drop-in replacement for means)
  new: class-adjusted form metric
  new: sectional time decomposition (early vs closing)
  new: improvement trajectory (z-score trend)

Pace/Position Prediction
  requires: pace_aptitude_features.py (DONE, basic version)
  requires: jyuni1c, jyuni4c, harontimel3 from history (DONE)
  new: composite pace figure per horse
  new: projected position at corners
  new: enhanced pace scenario match score
  depends-on: expected pace figure (must be built first)
```

## MVP Recommendation

### Phase 1: Quick Wins (Low complexity, HIGH expected ROI impact)

Build these first. They use existing data and simple computations.

1. **Hyperparameter diversity in StackedEnsemble** -- different lr/depth/rounds per base model. The single biggest improvement to the existing stacking implementation.
2. **Early stopping per base model** -- prevents overfitting in the ensemble.
3. **Feature subsampling per base model** -- forces model diversity.
4. **Odds-to-ability ratio** (p_market / p_ability) -- the single most important ROI signal as a Stage2 feature column.
5. **Exponential decay weighting** on past performance averages -- drop-in enhancement to all history averages.
6. **Odds acceleration** (2nd derivative of odds movement) -- simple addition to existing odds_dynamics_features.py.

Expected impact: These 6 changes should push ROI from 89% toward 95-98%. They require no new data pipelines, only code changes to existing modules.

### Phase 2: Medium Effort (Higher complexity, HIGH expected impact)

7. **Final EV residual validation** -- ensure the stacking output flows correctly through BenterGate to WinSelectionGate.
8. **Edge confidence interval** -- leverage existing conformal prediction for EV interval estimation.
9. **Kelly stake sizing from stacked probability** -- convert improved probabilities into better bet sizing.
10. **Class-adjusted form metric** -- contextualize form by class of opposition.
11. **Improvement trajectory** (z-score trend) -- more robust form trend using normalized data.
12. **Composite pace figure per horse** -- synthesize corner positions and closing times into a single pace metric.

Expected impact: These should push ROI from 95-98% to 100%+.

### Phase 3: High Effort (Defer if ROI target already met)

13. **Stacking at Stage1 (Ranker) level** -- complex but potentially highest-impact change.
14. **Projected position at each corner** -- requires field-level interaction modeling.
15. **Volume-weighted odds movement** -- only if volume data is available.
16. **Late money intensity (t-5 vs t-10)** -- only if snapshot granularity permits.

### Defer (v1.2 or later)

- Real-time pace simulation
- Weather-derived features
- Deep learning for temporal modeling
- Social media sentiment
- Biometric data features

---

## Existing Feature Effectiveness for v1.1 Context

These existing features become MORE important with ensemble stacking because the ensemble can capture interactions that a single model misses:

| Existing Feature | Why More Important with Stacking | Notes |
|-----------------|----------------------------------|-------|
| `odds_drop_rate_30_10` | CatBoost handles categorical interactions better; may discover that steam moves are more predictive on specific surface/distance combos | High priority to validate with stacking gain importance |
| `market_log_error_win` | The stacking ensemble's improved probability estimates will change the magnitude and distribution of this residual | Must recompute after stacking integration |
| `form_trend` | XGBoost's depth-based splitting may capture nonlinear form trend effects that LightGBM's leaf-wise growth misses | Key differentiator between base models |
| `pace_pressure` | Field-level pace dynamics are a higher-order interaction -- stacking should capture this better than a single model | Validate with SHAP interaction values post-stacking |
| `odds_to_ability_ratio` (once built) | This is the most direct edge signal. The ensemble should improve both sides of the ratio (better p_ability AND better p_market estimation) | Build and validate |

---

## Win-Specific vs Place Feature Differences (v1.1 Context)

Stacking benefits win prediction MORE than place because:

| Aspect | Win (Primary) | Place (Secondary) |
|--------|---------------|-------------------|
| **Base rate** | ~7% (harder) | ~21% (easier) |
| **Stacking benefit** | HIGH -- small probability improvements yield large ROI gains in rare-event prediction | MODERATE -- place pool has less inefficiency to exploit |
| **Odds deviation value** | VERY HIGH -- finding a 10-1 horse that should be 6-1 is massive ROI | MODERATE -- place payouts are more compressed |
| **Pace feature value** | HIGH -- pace determines WHO wins, not just who places | MODERATE -- closers place regardless of pace |
| **Form cycle value** | Need PEAK form (winning form) | Consistent form suffices |
| **Ensemble focus** | Prioritize stacking on hit_model for win | Can defer place stacking optimization |

---

## Sources

- Benter, W. (1994). "Computer Based Horse Race Handicapping and Wagering Systems" -- foundational model structure
- [Ensemble Stacking: XGBoost + LightGBM + CatBoost](https://medium.com/@stevechesa/stacking-ensembles-combining-xgboost-lightgbm-and-catboost-to-improve-model-performance-d4247d092c2e) -- stacking best practices
- [PMC: Ensemble ML of Gradient Boosting](https://pmc.ncbi.nlm.nih.gov/articles/PMC10611362/) -- peer-reviewed study on GBM stacking
- [Springer: View Selection in Multi-View Stacking](https://link.springer.com/article/10.1007/s11634-024-00587-5) -- Ridge/ElasticNet as optimal meta-learners
- [Optimizing Horse Racing Predictions through Ensemble Learning](https://www.researchgate.net/publication/385301910) -- ensemble methods + automated betting
- [Horse Race Predictions: XGBoost vs Betting Markets](https://www.kaggle.com/code/lukebyrne/horse-race-predictions-xgboost-vs-betting-markets) -- market comparison methodology
- [Beating the Odds: ML for Horse Racing](https://teddykoker.com/2019/12/beating-the-odds-machine-learning-for-horse-racing/) -- Benter-inspired approach
- [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion) -- optimal bet sizing theory
- [Late Money and Market Efficiency (ECU)](https://economics.ecu.edu/wp-content/pv-uploads/sites/165/2019/07/ECU1202.pdf) -- late money predictive value
- [Past Pace as Predictor of Future Performance](https://www.geegeez.co.uk/past-pace-as-a-predictor-of-future-performance-part-2/) -- pace persistence patterns
- [StackOverflow: Why stacking doesn't beat best base model](https://stats.stackexchange.com/questions/561584/why-is-my-stacking-meta-learning-not-outperforming-the-best-base-model) -- stacking failure modes
- Context7 LightGBM documentation -- binary classification, early stopping, categorical features
