# Domain Pitfalls: Horse Racing Win Prediction

**Domain:** Parimutuel horse racing win (tansho) prediction, JRA Japan
**Researched:** 2026-05-02
**Confidence:** HIGH (cross-validated with academic sources, Benter paper, and codebase analysis)

---

## Critical Pitfalls

Mistakes that cause systematic negative ROI or require major rewrites.

### Pitfall 1: Ignoring the JRA 25% Takeout in Edge Thresholds

**What goes wrong:** The model identifies "positive EV" bets where the raw edge (p * odds - 1) is only 5-10%. In JRA parimutuel betting, 25% is deducted from the pool before payouts. The effective break-even edge is far higher than most models assume.

**Why it happens:** Models compute EV using observed parimutuel odds (which already embed the takeout). The implied probability from odds = 1/odds already reflects the crowd's estimate PLUS the takeout. But if the model computes edge = p_model * odds - 1 and the model probability is only slightly better than the crowd's, the 25% house take still makes the bet unprofitable.

**Consequences:** The model bets on many "positive EV" horses that are actually negative EV after the takeout is accounted for. The system can have a theoretical edge but still lose money consistently -- exactly the 89% ROI observed.

**Current codebase status:** The edge thresholds in `RegimeDetector.get_strategy_params()` range from 0.04 (aggressive) to 0.08 (collapsed). For JRA's 25% takeout, a model needs p_model significantly above the crowd's implied probability just to break even. The 4-8% edge thresholds may be insufficient given that parimutuel odds are already "efficient" (large JRA pools).

**Prevention:**
- Calibrate edge thresholds against actual historical ROI. If edge_threshold=0.04 produces 89% ROI, the threshold needs to be higher.
- Consider computing edge relative to "fair odds" (odds without takeout) rather than raw parimutuel odds. Fair odds = odds * (1 / (1 - takeout_rate)).
- The minimum viable edge for positive ROI in JRA depends on calibration quality but is empirically around 25-30% above the crowd's estimate for the specific horse.

**Detection:** If backtest ROI is consistently below 100% despite many bets with "positive edge," the edge thresholds are miscalibrated relative to the takeout.

**Phase:** Must be addressed in the betting strategy phase, before any model tuning.

**Sources:**
- JRA official: 25% takeout, 75% return to bettors (https://japanracing.jp/en/jpn-racing/guide/pdf/horseracing_en_03.pdf)
- HIGH confidence: Official JRA documentation

---

### Pitfall 2: Calibration Mismatch Between P(hit) and E(odds|hit) in the 2-Stage Model

**What goes wrong:** The 2-stage decomposition EV = P(win) * E(odds|win) assumes independence between winning and the payout odds. In parimutuel systems, this assumption is violated: the odds of a horse are inversely related to its win probability. When P(win) is miscalibrated high, E(odds|win) is also miscalibrated (it was trained on actual winners whose odds were higher than the model predicts). The product P*E can systematically overestimate or underestimate EV.

**Why it happens:** The E(odds|win) model trains only on actual winners. If the P(win) model overestimates a horse's win probability, more "unlikely winners" get selected, and their actual odds are higher than the E model expects because the E model learned from genuinely strong winners. This creates a systematic EV bias.

**Current codebase status:** `EVCorrectionModel` in `src/models/ev_correction_model.py` attempts to fix this with P-correction and E-correction sub-models. The E-correction uses `weight=1/sqrt(p)` to upweight unlikely winners, which partially addresses this. However, the P-correction is trained on the same data with `init_score=logit(p_pred)`, meaning it can only nudge probabilities, not fix systematic miscalibration.

**Consequences:** The 2-stage model may show positive EV on horses where the model probability is very close to the crowd's, but systematically overestimate EV for longshots and underestimate for favorites. This directly feeds into the favorite-longshot bias issue (see Pitfall 3).

**Prevention:**
- Evaluate calibration separately for different odds ranges (favorites vs mid-range vs longshots).
- Apply the Benter combination (fundamental + market) BEFORE the 2-stage decomposition, not after. The current code applies Benter only for place predictions (`p_place_combined`), NOT for win predictions. Win EV uses raw `p_win_pred` from the 2-stage model without Benter blending.
- Consider a single-stage model for win prediction that directly estimates EV = p * odds, rather than decomposing.

**Detection:** Compute actual ROI by odds bucket (1-3, 3-5, 5-10, 10-20, 20+). If ROI is negative in specific buckets, the 2-stage independence assumption is breaking down there.

**Phase:** Model architecture phase (fundamental restructuring if needed).

**Sources:**
- Benter (1994) paper: warns about independence assumptions in multi-factor handicapping models
- One- and Two-Step Conditional Logit Models (Journal of Probability and Management): compares approaches
- HIGH confidence: Directly observed in codebase architecture

---

### Pitfall 3: Missing Benter Combination for Win Predictions

**What goes wrong:** The Benter combination (blending fundamental model probability with market-implied probability) is only applied to place predictions in the current codebase. Win predictions use the raw 2-stage model output without any market information blending. This means the win model completely ignores the market's efficiency signal.

**Why it happens:** The codebase was originally built for place betting and the Benter combination was added only for place. The `RacePredictor.predict()` method in `src/backtest/race_predictor.py` applies Benter combination only to `p_place_combined`. Win predictions go through `WinTwoStageModel.predict_ev()` then `EVCorrectionModel.correct_ev()` but never get the benefit of market probability blending.

**Current codebase status:** In `race_predictor.py:127-156`, the Benter combination and isotonic calibration are applied only for place. Win predictions use `p_win_pred` and `ev_win_corrected` without Benter. This is a critical gap for the win model pivot.

**Consequences:** The win model has no mechanism to "pull back" toward the market consensus. If the model thinks a 10/1 horse has a 15% win chance, there is no correction from the market saying "the crowd collectively disagrees -- they think it's 8%." Without this, the model makes more extreme bets than warranted.

**Prevention:**
- Implement Benter combination for win predictions: `logit(p_win_combined) = alpha * logit(p_win_pred) + beta * logit(1/tanodds) + gamma`.
- Fit alpha, beta, gamma on validation data using maximum likelihood.
- Apply temperature scaling after Benter combination.

**Detection:** Compare model calibration with and without Benter blending. If the model's raw probabilities are poorly calibrated (overconfident on longshots), Benter combination should significantly improve this.

**Phase:** First implementation phase for win model improvement. This is the single highest-impact change.

**Sources:**
- Benter (1994): the original paper explicitly uses market odds as a second information source
- Codebase analysis: `race_predictor.py` lines 127-156 show Benter only for place
- HIGH confidence: Directly observable in code

---

### Pitfall 4: Using Pre-Race Odds Snapshot Instead of Closing Odds in Backtest

**What goes wrong:** The backtest uses odds from a snapshot taken before the race (tanodds from odds snapshots), not the actual closing odds at race time. In parimutuel betting, odds move significantly in the final minutes as late money arrives. The model's EV calculation uses one set of odds, but the actual payout uses different odds.

**Why it happens:** The data pipeline captures odds snapshots at intervals, and the "best available" pre-race snapshot may not be the actual closing odds. The `confirmed_odds` (kakutei odds) are the actual payout odds but are post-race data that cannot be used for prediction. Using them for EV calculation in training would be look-ahead bias.

**Current codebase status:** The feature engine (`src/features/feature_engine.py:137-143`) uses `tanodds` from odds snapshots as the primary odds source, falling back to `confirmed_odds`. The `POST_RACE_COLS` list in `src/backtest/engine.py:35-49` strips post-race columns before prediction. However, the question is: what odds does the backtest use for payout calculation? If it uses `confirmed_odds` for settlement (which it should, since that's the actual payout), but the model makes decisions based on pre-race `tanodds`, there is a systematic discrepancy.

**Consequences:** The model identifies "value" based on pre-race odds, but by post time, heavy betting may have moved the odds enough to eliminate the edge. This is particularly severe for favorites where large late bets can compress odds significantly.

**Prevention:**
- Use the latest available pre-race odds snapshot (closest to post time) for both EV calculation and bet selection in backtest.
- Document which odds timestamp is used and how far from post time it is.
- Consider modeling odds drift: if a horse's odds typically drift from 5.0 to 4.2 in the last 10 minutes, the model's EV estimate at odds=5.0 is too optimistic.
- For live paper trading, the same odds timing issue exists -- ensure the system captures odds at a consistent point.

**Detection:** Compare the odds used for bet selection vs. the actual confirmed odds for each bet. If there is systematic drift (selected odds always higher than confirmed odds), the backtest is overestimating returns.

**Phase:** Data validation phase (before model training).

**Sources:**
- Quantitative Horse Racing with R (R-Bloggers, 2026): discusses odds timing in backtesting
- Medium (Michael Harris): look-ahead bias in backtests
- MEDIUM confidence: Odds timing effect is well-documented but specific JRA drift magnitude is unknown

---

### Pitfall 5: Probability Overconfidence and Kelly Criterion Ruin

**What goes wrong:** The Kelly criterion calculates optimal bet size based on the model's probability estimate. If the model is overconfident (probability too high), Kelly will overbet. Even small systematic overconfidence leads to bankroll depletion over many bets. This is the most dangerous interaction between model quality and betting strategy.

**Why it happens:** LightGBM's raw outputs are not well-calibrated probabilities. The codebase applies IsotonicRegression and TemperatureScaling for calibration, but these are fitted on validation data and may not generalize to test data. The comment in `race_predictor.py:144` explicitly notes that "Isotonic post-Benter is too aggressive (pushes mean 0.224 vs true ~0.375)" -- the calibration was so bad it had to be disabled.

**Current codebase status:** `StakeCalculator` uses half-Kelly (FRACTIONAL_KELLY=0.5) with a KELLY_FRACTION_CAP of 0.25 and race exposure cap of 2%. This provides some safety margin. However, the calibration for WIN predictions specifically (as opposed to place) has not been validated. The isotonic calibration is currently disabled for place and was never implemented for win.

**Consequences:** Research shows that calibration-optimized models yield +36.93% ROI while accuracy-optimized models yield -75.9% ROI when using Kelly staking (Walsh & Joshi, 2023). Even small probability errors are amplified by Kelly sizing.

**Prevention:**
- Implement and validate calibration specifically for win predictions using the reliability diagram approach.
- Use fractional Kelly (already doing half-Kelly, which is good).
- Add a calibration diagnostic: plot predicted probability vs. observed frequency by decile for win predictions specifically.
- Consider capping the maximum probability at a conservative value (e.g., 0.40 for a single horse in a 14-horse field).
- Never use raw LightGBM probabilities for Kelly sizing without calibration.

**Detection:** Compute a calibration curve for win predictions on out-of-sample data. If the curve deviates significantly from the diagonal, calibration is broken. If high-probability predictions (>0.20) have lower actual win rates, the model is overconfident.

**Phase:** Calibration phase (immediately after initial win model training).

**Sources:**
- Walsh & Joshi (2023), arXiv:2303.06021: calibration-optimized = +36.93% ROI, accuracy-optimized = -75.9%
- Kelly Betting with Uncertainty (arXiv:1701.02814): optimal sizing under probability uncertainty
- HIGH confidence: Peer-reviewed academic evidence

---

### Pitfall 6: Overfitting to 2024 Test Data by Tuning Hyperparameters

**What goes wrong:** When iterating on the win model, each change is evaluated against the 2024 backtest. After dozens of iterations, the model becomes implicitly overfitted to 2024 even without directly tuning hyperparameters for 2024. Features, thresholds, and model structures are all chosen because they "work" on 2024.

**Why it happens:** The pipeline uses a single train/test split (2020-2023 train, 2024 test). There is no holdout set beyond 2024. Every design decision is made with knowledge of how it performs on 2024 data. This is a form of "researcher degrees of freedom" overfitting.

**Current codebase status:** `run_backtest.py` supports multi-year backtests (`--years 2023 2024 2025`), which could provide more robust evaluation. However, the primary development loop has been single-year testing.

**Consequences:** The model may show 100%+ ROI on 2024 but fail on 2025 or future years. This is the most common failure mode in prediction model development and the hardest to detect before real-money deployment.

**Prevention:**
- Use walk-forward validation: train on 2020-2022, test on 2023; train on 2020-2023, test on 2024; train on 2021-2024, test on 2025. All must show positive ROI.
- Reserve 2025 as a final holdout. Do not look at 2025 results until the model is finalized.
- Limit the number of hyperparameter changes per iteration. Track how many "tries" have been made.
- Use the multi-year backtest mode as the primary evaluation metric, not single-year.

**Detection:** If ROI varies dramatically across years (e.g., +10% in 2023, -15% in 2024, +5% in 2025), the model is not robust. Consistent but modest positive ROI across all years is far more trustworthy than large positive ROI in one year.

**Phase:** Ongoing -- every model evaluation phase must use proper validation.

**Sources:**
- Cross-Validation vs Walk-Forward (GitConnected): walk-forward gives pessimistic but honest estimates
- ECB Working Paper: walk-forward validation for temporal data
- HIGH confidence: Standard ML best practice for time series

---

## Moderate Pitfalls

### Pitfall 7: Favorite-Longshot Bias Not Explicitly Modeled

**What goes wrong:** In parimutuel markets, favorites are underbet (better expected returns) and longshots are overbet (worse expected returns). If the model treats all odds ranges equally, it will bet on too many longshots where the market is actually more efficient.

**Current codebase status:** The `EVCorrectionModel` includes `place_bucket_multiplier` which applies decay factors for high-odds horses (odds >= 15: 0.95x, >= 22: 0.85x, >= 30: 0.7x). This is a crude correction. The `WinTwoStageModel.FEATURE_COLS` includes `odds_skewness` and `implied_prob_hhi` which capture some market structure, but there is no explicit FLB correction for win predictions.

**Prevention:**
- Add explicit FLB correction to win probability estimates: apply a discount factor to model probabilities for longshots and a premium for favorites.
- Evaluate calibration by odds bucket and add bucket-specific correction factors.
- The current `popularity_rank` feature helps but does not fully capture FLB.

**Phase:** Model calibration phase.

**Sources:**
- Favorite-Longshot Bias overview (ResearchGate): comprehensive explanation
- The Favorite-Longshot Midas (Wharton): empirical evidence
- HIGH confidence: Well-established academic finding

---

### Pitfall 8: Feature Importance Shift Between Place and Win

**What goes wrong:** Features that predict placing (top 3) may not predict winning (top 1). The model was developed and validated for place betting. Many features may have been selected because they help predict place outcomes, but have different (or opposite) importance for win prediction.

**Current codebase status:** The `WinTwoStageModel` uses a fixed feature set (`FEATURE_COLS`) that overlaps with place model features but is not independently validated for win prediction. The `AbilityModel` (stage 1) generates `p_ability_win` and `p_ability_place` from the same underlying model structure but with different targets.

**Prevention:**
- Run feature importance analysis (SHAP or gain-based) specifically for the win model, independent of place.
- Identify features that have opposite importance for win vs. place (e.g., consistency may help place but hurt win if it means the horse never wins).
- Consider win-specific features: "winning habit" (has this horse won before at this level?), "closing speed" (late-race acceleration that wins but may not place if the horse is inconsistent).

**Detection:** Compare feature importance rankings between win and place models. Large discrepancies indicate features that may be misleading for win prediction.

**Phase:** Feature analysis phase (first phase of win model work).

**Sources:**
- Codebase analysis: WinTwoStageModel.FEATURE_COLS vs PlaceTwoStageModel.FEATURE_COLS
- MEDIUM confidence: Inference from domain knowledge

---

### Pitfall 9: Inline Feature Pre-Computation Duplication Between Training and Backtest

**What goes wrong:** The architecture document already identifies this anti-pattern. `BacktestEngine.run()` contains 100+ lines of feature pre-computation that duplicates `TrainingPipelineV5._train_submodel()`. Any change to feature computation in training must be manually mirrored in backtest. If they diverge, the model is trained on one feature set but evaluated on a different one.

**Current codebase status:** Documented as an anti-pattern in `.planning/codebase/ARCHITECTURE.md`. The `RacePredictor` was partially extracted to share inference logic, but feature pre-computation remains duplicated.

**Prevention:**
- Extract shared feature pre-computation into a single function called by both `TrainingPipelineV5` and `BacktestEngine`.
- Add a feature parity test: verify that training features and backtest features produce identical values for the same input data.

**Detection:** Compare feature distributions between training and backtest data. If means/percentiles differ significantly for non-time-varying features, the computation paths have diverged.

**Phase:** Infrastructure phase (before or alongside feature analysis).

**Sources:**
- Architecture analysis: documented anti-pattern
- HIGH confidence: Directly observable in code

---

### Pitfall 10: Regime Detector Trained on Aggregate Metrics, Not Per-Horse Signals

**What goes wrong:** The `RegimeDetector` uses race-level aggregate metrics (overround, entropy, favorite implied probability) to decide whether to bet aggressively or conservatively. This determines the edge threshold and bet sizing. However, these aggregate metrics may not capture when the model has a genuine edge on a specific horse.

**Why it happens:** The regime detector was designed for place betting where the strategy is more conservative. For win betting, the edge may be concentrated in specific races (e.g., a strong favorite scratched, changing the race dynamics) rather than broad market conditions.

**Consequences:** In "collapsed" regime, the model sets edge_threshold=0.08 and nearly stops betting. But there may be genuine 10%+ edges in individual races during a "collapsed" market that are being missed.

**Prevention:**
- Evaluate whether the regime detector actually improves win prediction ROI, or whether a fixed edge threshold performs better.
- Consider horse-level confidence metrics (e.g., model agreement across folds) rather than race-level regime.
- Test regime-adaptive vs. fixed-threshold strategies on walk-forward validation.

**Detection:** Compare ROI with regime detector enabled vs. disabled. If regime-adaptive ROI is worse than fixed threshold, the regime detector is hurting.

**Phase:** Betting strategy evaluation phase.

**Sources:**
- Codebase analysis: `regime_detector.py`
- MEDIUM confidence: Architectural inference

---

### Pitfall 11: Insufficient Training Data for E(odds|win) Sub-Model

**What goes wrong:** The E(odds|win) model in the 2-stage decomposition trains only on actual winners. In a 14-horse field, only ~7% of horses win (1/14). With 4 years of training data (2020-2023), this may be only ~15,000-25,000 winners. Split into train/valid and by surface (turf/dirt), the effective training set for the E model could be as small as 5,000-8,000 samples per surface.

**Why it happens:** The 2-stage model by design trains the E sub-model only on winners. This is unavoidable but limits model complexity.

**Consequences:** The E model may not have enough data to learn the relationship between horse features and payout odds for winners. It may default to predicting the mean payout, which reduces the EV decomposition to effectively EV = P * constant, losing the value of the decomposition.

**Prevention:**
- Monitor the E model's out-of-sample R^2 or MAE. If it is near-zero predictive power, the 2-stage decomposition is not adding value.
- Consider simplifying: instead of predicting E(odds|win), use the pre-race odds directly as the payout estimate. This avoids the sparse-data problem entirely.
- Alternatively, use a larger training window for the E model (e.g., 2015-2023) since it only needs winners.

**Detection:** Check the E model's validation loss. If early stopping triggers very early (e.g., round 10-20) or the model has near-zero feature importance, it is underfitting due to data scarcity.

**Phase:** Model evaluation phase.

**Sources:**
- Codebase analysis: `two_stage_return_model.py` trains E model on winners only
- HIGH confidence: Mathematical fact about sample size

---

### Pitfall 12: Odds Dynamics Features May Contain Look-Ahead Bias

**What goes wrong:** The `WinTwoStageModel` uses odds dynamics features (`odds_drop_rate_60_10`, `odds_drop_rate_30_10`, `odds_velocity`, `odds_volatility`, `popularity_change_30_10`). These are computed from odds time series snapshots. If the time series includes snapshots too close to post time (or post-race), these features encode information not available at bet-placement time.

**Current codebase status:** `odds_dynamics_features.py` uses `post_datetime` as a reference point. The snapshot timing relative to post time needs careful validation. The `OddsExtractor` in `src/db/odds_extractor.py` may extract odds at various time points.

**Prevention:**
- Audit the timestamp of every odds snapshot used in feature computation.
- Ensure all odds dynamics features use only snapshots available before the bet-placement cutoff (e.g., 10 minutes before post time).
- Add a timestamp validation test: for each race, verify that the latest snapshot used in features is before post time.

**Detection:** Compute odds dynamics features using only pre-race snapshots and compare with features computed using all snapshots. If they differ significantly, there is look-ahead bias.

**Phase:** Data validation phase.

**Sources:**
- Look-Ahead Bias in Backtests (Medium/Michael Harris): general principle
- Codebase analysis: `odds_dynamics_features.py`
- MEDIUM confidence: Needs specific audit of snapshot timestamps

---

### Pitfall 13: Concept Drift in Horse Racing Features Over 2015-2025

**What goes wrong:** Horse racing characteristics change over time: jockey populations shift, training methods evolve, race scheduling changes, track maintenance practices change, and rule modifications occur. A model trained on 2020-2023 data may not generalize to 2025 if the underlying feature distributions have shifted.

**Why it happens:** Features like jockey win rates, trainer statistics, and bloodline features are computed from historical data. A jockey who was dominant in 2020 may have declined by 2025, but the expanding mean still gives them high ratings.

**Current codebase status:** The codebase uses `expanding().shift(1)` for cumulative statistics (preventing look-ahead bias), but expanding means are slow to adapt to recent changes. The `leakage_validators.py` module validates no future leakage but does not check for concept drift.

**Prevention:**
- Use exponentially weighted moving averages (EWMA) instead of simple expanding means for jockey/trainer stats. Give more weight to recent performance.
- Monitor feature distributions across years. If a feature's mean shifts significantly between 2020 and 2024, the model may be learning stale patterns.
- Implement periodic retraining (at minimum yearly, ideally quarterly).
- Consider adding a "recency" feature that captures recent form separately from career averages.

**Detection:** Compute feature distributions by year. Plot means and standard deviations over time. Significant trends indicate concept drift.

**Phase:** Feature engineering phase.

**Sources:**
- Reddit ML practitioner: "concept drift was a constant issue" requiring weekly retraining
- Unsupervised Concept Drift Detection (arXiv): general methodology
- MEDIUM confidence: General ML principle applied to racing domain

---

## Minor Pitfalls

### Pitfall 14: Race-Level Normalization May Wash Out Signal

**What goes wrong:** The `_normalize_probability_by_race` function normalizes probabilities so they sum to 1.0 within each race. If the model assigns genuine high probability to multiple horses in a competitive race, normalization forces them down, potentially eliminating valid "strong field" signals.

**Prevention:** Evaluate whether normalization helps or hurts. For betting, the absolute EV matters more than relative ranking. Consider skipping normalization and using raw probabilities for EV calculation.

**Phase:** Model evaluation phase.

---

### Pitfall 15: Flat Minimum Stake of 100 Yen Creates Discretization Noise

**What goes wrong:** The `StakeCalculator.MIN_STAKE = 100` means Kelly-optimal stakes are rounded to the nearest 100 yen. For small bankrolls or small Kelly fractions, this rounding can cause the actual bet to be 0 (eliminating valid bets) or significantly larger than optimal (doubling the intended stake for small fractions).

**Prevention:** Consider lowering the minimum stake or using a percentage-of-bankroll approach instead of absolute yen amounts.

**Phase:** Betting strategy phase.

---

### Pitfall 16: Single 80/20 Time-Series Split Instead of Walk-Forward

**What goes wrong:** All models use a single 80/20 temporal split for train/validation. This provides only one estimate of model performance. The validation set may contain unusual market conditions (e.g., COVID year, rule changes) that bias the evaluation.

**Current codebase status:** `_train_valid_split` in `two_stage_return_model.py` does a fixed 80/20 split. The multi-year backtest mode provides some additional validation but is not integrated into model training.

**Prevention:** Implement walk-forward cross-validation with expanding windows. Train on [2015-2020], test on 2021; train on [2015-2021], test on 2022; etc.

**Phase:** Model validation phase.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Feature analysis for win | Pitfall 8: Place features may not transfer to win | Run independent feature importance for win target |
| Win model training | Pitfall 11: Sparse E(odds\|win) training data | Monitor E model R^2; consider alternatives |
| Calibration | Pitfall 5: Overconfidence kills Kelly bettors | Implement calibration diagnostics; use fractional Kelly |
| Benter combination for win | Pitfall 3: Win model has no market signal | Implement win Benter as first priority |
| Betting strategy | Pitfall 1: Edge thresholds ignore 25% takeout | Calibrate thresholds against actual ROI |
| Backtest validation | Pitfall 6: Overfitting to single test year | Use walk-forward validation across multiple years |
| Odds data audit | Pitfall 4: Pre-race vs closing odds gap | Document and validate snapshot timing |
| Feature drift monitoring | Pitfall 13: Concept drift across years | Monitor feature distributions by year |
| Regime adaptation | Pitfall 10: Regime detector may hurt win ROI | A/B test regime vs. fixed threshold |

---

## Priority Action Items (Ordered by Impact)

1. **Implement Benter combination for win** (Pitfall 3) -- The win model currently ignores the market entirely. This is the single highest-leverage change.
2. **Recalibrate edge thresholds for JRA 25% takeout** (Pitfall 1) -- Current thresholds likely produce negative real ROI even with a good model.
3. **Add calibration diagnostics for win predictions** (Pitfall 5) -- Without calibration visibility, all downstream betting decisions are unreliable.
4. **Implement walk-forward validation** (Pitfall 6) -- Protect against overfitting before spending time on model iteration.
5. **Audit odds snapshot timing** (Pitfall 4) -- Ensure no look-ahead bias in odds dynamics features.
6. **Run win-specific feature importance** (Pitfall 8) -- Identify which existing features actually help win prediction.

---

## Sources

- [JRA Official Guide - Deduction Rate (25%)](https://japanracing.jp/en/jpn-racing/guide/pdf/horseracing_en_03.pdf) -- HIGH confidence
- [Walsh & Joshi (2023): ML for Sports Betting - Calibration vs Accuracy](https://arxiv.org/abs/2303.06021) -- HIGH confidence, peer-reviewed
- [Kelly Betting on Horse Races with Uncertainty in Probability Estimates](https://arxiv.org/pdf/1701.02814) -- HIGH confidence
- [Benter (1994): Computer Based Horse Race Handicapping and Wagering Systems](https://datagolf.com/static/blogs/benter_paper.pdf) -- HIGH confidence, foundational reference
- [Favorite-Longshot Bias: Overview of Main Explanations](https://www.researchgate.net/publication/228884358_The_Favorite-Longshot_Bias_An_Overview_of_the_Main_Explanations) -- HIGH confidence
- [Cross-Validation vs Walk-Forward: The Time Series Trap](https://levelup.gitconnected.com/cross-validation-vs-walk-forward-the-time-series-trap-that-cost-me-500k-a03c65c3d1f0) -- MEDIUM confidence
- [Look-Ahead Bias in Backtests](https://mikeharrisny.medium.com/look-ahead-bias-in-backtests-and-how-to-detect-it-ad5e42d97879) -- MEDIUM confidence
- [Systematic Review of ML in Sports Betting](https://arxiv.org/html/2410.21484v1) -- MEDIUM confidence
- [Quantitative Horse Racing with R: Calibration, Backtesting, and Deployment](https://www.r-bloggers.com/2026/02/quantitative-horse-racing-with-r-calibration-backtesting-and-deployment/) -- MEDIUM confidence
- [One- and Two-Step Conditional Logit Models](https://www.ubplj.org/index.php/jpm/article/download/419/450/1317) -- MEDIUM confidence
- [ML Sports Betting in Production: Concept Drift](https://www.reddit.com/r/learnmachinelearning/comments/1o5mcvy/ml_sports_betting_in_production_563_accuracy_real/) -- LOW confidence (anecdotal)
- Codebase analysis of `src/models/`, `src/betting/`, `src/backtest/` -- HIGH confidence (first-hand observation)
