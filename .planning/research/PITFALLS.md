# Domain Pitfalls: Adding Ensemble Stacking, Odds Features, and Time-Series Features

**Domain:** Parimutuel horse racing win (tansho) prediction -- JRA Japan
**Context:** Incremental additions to an existing working LightGBM pipeline (keiba-ai v5.5)
**Researched:** 2026-05-03
**Confidence:** HIGH (cross-validated with codebase analysis, academic sources, and domain literature)

---

## Critical Pitfalls

Mistakes that cause systematic negative ROI, silent model degradation, or require major rewrites when adding new features to the existing system.

### Pitfall 1: Stacking Meta-Learner Overfitting from Correlated Base Models

**What goes wrong:** The `StackedEnsemble` trains LightGBM, XGBoost, and CatBoost on the same features with the same objective. Since all three are gradient-boosted decision trees, their predictions are highly correlated (typically Pearson r > 0.90). The Ridge meta-learner then overfits to small differences between correlated predictions, learning noise rather than genuine complementary signal. The ensemble performs worse than the best single model out-of-sample.

**Why it happens:** All three GBDT variants learn similar tree structures from the same tabular features. LightGBM and XGBoost in particular, with the current hardcoded hyperparameters (lr=0.03, 300 rounds), produce nearly identical splits on this data. CatBoost adds ordered boosting but on structured racing data with few categorical interactions, its advantage is marginal. The Ridge meta-learner has only 3 input features (the 3 predictions) -- with high multicollinearity, the coefficients become unstable. On OOF data the fit looks good; on truly unseen data the coefficient assignments flip and performance degrades.

**Current codebase status:** `StackedEnsemble` in `src/models/stacked_ensemble.py` uses 3-fold expanding-window OOF (lines 59-70) and Ridge(alpha=1.0) meta-learner (line 75). The base models all use `num_boost_round=300`, `learning_rate=0.03`. No diversity enforcement exists -- no different feature subsets, no different objectives, no different tree structures across base models. Hyperparameters are hardcoded rather than tuned.

**Consequences:** The ensemble may show marginal improvement on the OOF validation used during training (because Ridge fits the noise in OOF differences), but the improvement does not generalize to backtest. Worse, the ensemble adds ~3x training time and inference latency for no gain, or even a loss.

**Prevention:**
- Force diversity across base models: use different `num_leaves` (15/31/63), different `feature_fraction` (0.5/0.7/0.9), and different learning rates. Give each model a different inductive bias rather than identical configurations.
- Add the original features as meta-learner inputs alongside the 3 base predictions. Currently only 3 columns feed the meta-learner. Adding 5-10 key features (e.g., `popularity_rank`, `overround`, `surface`) gives the meta-learner signal beyond correlated predictions.
- Use a higher Ridge alpha (e.g., 10.0) to aggressively regularize the meta-learner when base predictions are correlated. Lower alpha allows overfitting.
- Validate the ensemble against the single best base model (not against the average). If the ensemble does not beat the single best model by at least 1% AUC on a held-out year, it is not worth the complexity.

**Detection:**
- Compute pairwise correlation between the 3 base model predictions on validation data. If all r > 0.90, stacking will add minimal value regardless of meta-learner choice.
- Compare OOF meta-learner R^2 vs. out-of-sample (next year) R^2. A large drop (>50% relative) indicates meta-learner overfitting.
- Compare ensemble backtest ROI against single-model (LightGBM-only) backtest ROI. If ensemble is worse, the meta-learner is overfitting.

**Phase:** Must be addressed during the stacking implementation phase, before any production use.

**Sources:**
- [StackExchange: Stacking with Cross-Validation](https://stats.stackexchange.com/questions/239445/how-to-properly-do-stacking-meta-ensembling-with-cross-validation) -- meta-learner must train on separate data from base models
- [ResearchGate: Stacking with XGBoost, LightGBM, CatBoost](https://www.researchgate.net/publication/397047638_Stacking_Ensemble_Learning_Combining_XGBoost_LightGBM_CatBoost_and_AdaBoost_with_Random_Forest_Meta_Model) -- correlated base models reduce stacking benefit
- [ML4Devs: Ensemble Methods](https://www.ml4devs.com/what-is/ensemble-learning-bagging-boosting-stacking/) -- meta-learner overfitting risk with expressive models
- HIGH confidence: Directly observable in `stacked_ensemble.py` code + standard ensemble theory

---

### Pitfall 2: Meta-Learner Temporal Leakage in OOF Generation

**What goes wrong:** The `StackedEnsemble` generates OOF predictions using expanding-window folds (lines 62-70). However, the final base models are then retrained on ALL data including the validation portion (lines 79-84). When the meta-learner is used to predict on backtest data, the base models have already seen the patterns in the test period. This creates a subtle form of temporal leakage: the meta-learner learned on OOF predictions, but the production base models are stronger than the OOF base models because they trained on more data.

**Why it happens:** The standard stacking implementation retrains base models on full data after meta-learner fitting to maximize base model quality. This is correct for i.i.d. data but problematic for time-series data where the "future" folds contain information about market regime changes, jockey form changes, and other temporal patterns that should not leak into the base models.

**Current codebase status:** `StackedEnsemble.train()` lines 79-84:
```python
X_all = pd.concat([X_train, X_valid], ignore_index=True)
y_all = pd.concat([y_train, y_valid], ignore_index=True)
self.lgbm_model = self._train_lgbm_full(X_all, y_all, num_threads)
self.xgb_model = self._train_xgb_full(X_all, y_all, num_threads)
self.cat_model = self._train_cat_full(X_all, y_all, num_threads)
```
The OOF meta-learner was trained on predictions from models that only saw `X_train`, but the production models see `X_train + X_valid`. This means the production base models will make systematically different (better) predictions than the OOF base models. The meta-learner's learned coefficients are calibrated for weaker predictions but applied to stronger ones.

**Consequences:** The meta-learner systematically misweights the base model predictions in production/backtest. If the meta-learned that "LightGBM is more accurate than XGBoost" based on OOF, but the full-data LightGBM is now 10% better while the full-data XGBoost is only 2% better, the meta-learner's coefficient ratio is wrong. This can cause the ensemble to underperform a simple average.

**Prevention:**
- **Option A (safer):** Do NOT retrain base models on full data. Use the OOF-trained base models directly. The meta-learner coefficients will be correctly calibrated. The cost is slightly weaker base models (trained on ~80% instead of ~100% of data), but the meta-learner coefficients are honest.
- **Option B (current approach, needs validation):** Keep full-data retraining, but add a calibration step: compare meta-learner predictions on a small held-out set using OOF base models vs. full base models. If the predictions differ by more than a tolerance (e.g., 5% of predicted probability), the meta-learner needs recalibration.
- **Option C (hybrid):** Use the fold-1 base models (trained on least data) as the production models, and use folds 2-3 as validation for the meta-learner. This ensures the meta-learner sees predictions from models with similar data access to the production models.

**Detection:**
- Compare ensemble predictions on 2024 data using (a) OOF base models vs. (b) full-data base models. If the mean absolute difference in predictions exceeds 2%, the meta-learner is miscalibrated for the production models.
- Check if the meta-learner's Ridge coefficients are heavily weighted toward one model (e.g., [0.8, 0.1, 0.1]). This suggests one model dominates, and the meta-learner is fitting noise in the remaining weights.

**Phase:** Must be addressed during stacking implementation.

**Sources:**
- [Kaggle: Questions About Stacking](https://www.kaggle.com/general/24748) -- practical discussion of OOF vs full-data retraining
- [Cross Validated: Stacking with CV](https://stats.stackexchange.com/questions/239445/how-to-properly-do-stacking-meta-ensembling-with-cross-validation) -- golden rule of meta-learner separation
- HIGH confidence: Directly observable in code + standard stacking theory

---

### Pitfall 3: Odds Deviation Features Using Post-Time or Near-Post Odds (Look-Ahead Bias)

**What goes wrong:** The odds deviation feature (`odds_to_ability_ratio = p_market / p_ability`) and the odds dynamics features (`odds_drop_rate_*`, `odds_velocity`, `odds_volatility`) use market odds that may reflect information available only at or after post time. In backtesting, the model uses these "final" odds to make predictions, but in live operation, the odds are still moving. The model appears profitable in backtest but loses money in paper trading.

**Why it happens:** The JRA odds market is highly dynamic in the final 10 minutes before post time. Large bets from professional players ("smart money") arrive in the last 5 minutes, significantly moving odds. The `OddsExtractor` extracts pre-post odds at `minutes_before=5`, but:
1. The feature engine also uses `confirmed_odds` (kakutei odds) as a fallback when snapshot odds are missing (`feature_engine.py:137-143`).
2. The `odds_dynamics_features.py` uses `_mins_before_anchor` calculated relative to `post_datetime` (line 167-169). If `post_datetime` is estimated from `hassotime` and the snapshot timing is approximate, some "t-10min" snapshots may actually be closer to post time.
3. In backtest, `confirmed_odds` is used for settlement (correct) but if it leaks into features through any fallback path, the model has access to post-race information.

**Current codebase status:**
- `odds_extractor.py` extracts at `minutes_before=5` with `max_staleness_minutes=60`.
- `odds_dynamics_features.py` uses `_pick_target_snapshot()` with `target_minutes=10.0, tolerance_minutes=15.0` -- this means a snapshot at t-0 (post time) could match the "t-10" target if no closer snapshot exists, because the tolerance is wider than the gap.
- The `_build_post_time_map()` function falls back to `grouped["_ts_datetime"].transform("max") + pd.Timedelta(minutes=10)` when `hassotime` is missing (line 165-166). This fallback estimates post time as 10 minutes after the last snapshot, which is guesswork.
- The `odds_to_ability_ratio` depends on `p_market_win_adj` from `compute_market_bias()`, which uses `tanodds` from the snapshot. If the snapshot is too close to post time, this ratio contains information from the efficient closing market.

**Consequences:** The model learns that "odds that moved a lot in the last 10 minutes predict outcomes" -- but this information is not available when placing bets 10+ minutes before post. The backtest overstates ROI by incorporating late odds movements that contain genuine predictive signal (smart money).

**Prevention:**
- Use a stricter cutoff: only use snapshots from t-15min or earlier for feature computation in both training and backtest. The current `tolerance_minutes=15.0` for t-10 target is too loose.
- Never fall back to `confirmed_odds` for feature computation. If the pre-race snapshot is missing for a horse, fill the odds features with NaN rather than using post-race odds.
- Validate the timing: for each race in the training set, log the actual timestamp of the latest snapshot used in feature computation and verify it is at least 10 minutes before post time.
- In `odds_dynamics_features.py`, tighten `_pick_target_snapshot` tolerance to 5 minutes maximum, not 15.

**Detection:**
- Compare the model's predictive accuracy using (a) t-10min odds features vs. (b) t-30min odds features vs. (c) confirmed_odds features. If accuracy improves monotonically as you get closer to post time, the model is exploiting look-ahead information. A genuine model should show similar accuracy with t-30 and t-10 features.
- Run the backtest twice: once with the standard pipeline and once with all odds features set to NaN. If ROI drops significantly (e.g., from 100% to 80%), the model is heavily dependent on odds features, which increases look-ahead risk.

**Phase:** Must be addressed before trusting any odds-based feature results. This is a data-integrity issue.

**Sources:**
- [Look-Ahead Bias in Backtests (Michael Harris)](https://mikeharrisny.medium.com/look-ahead-bias-in-backtests-and-how-to-detect-it-ad5e42d97879) -- general principle of temporal information leakage
- [Horse Racing Prediction (CUHK)](https://www.cse.cuhk.edu.hk/lyu/_media/thesis/report-1805-1.pdf) -- using final odds creates optimistic bias
- Codebase analysis: `odds_dynamics_features.py:167-169`, `odds_extractor.py:15-22`
- HIGH confidence: Directly observable tolerance values in code

---

### Pitfall 4: Time-Series Horse Performance Features Use Future Race Data (Temporal Leakage)

**What goes wrong:** When computing time-series features from past races (e.g., `harontime_late_trend`, `timediff_avg`, `closing_index_avg`), the `searchsorted` cutoff must strictly exclude the current race. If the cutoff uses `side='right'` instead of `side='left'`, or if dates are compared with `<=` instead of `<`, the current race's results leak into the "past performance" features.

**Why it happens:** The feature computation sorts by `race_date` and uses `np.searchsorted` to find the cutoff point. In `pace_aptitude_features.py:218`, the cutoff correctly uses `side='left'` to exclude the current date. But in `horse_history_features.py`, the cutoff logic uses cumulative arrays that must be carefully aligned. A single off-by-one error (e.g., including the race at the boundary) leaks the current race's finishing position, time, and pace data into features that supposedly represent "past" performance.

**Current codebase status:** `pace_aptitude_features.py:218` uses `side='left'` which is correct. `horse_history_features.py` uses sorted arrays and cumulative sums -- the implementation needs a line-by-line audit to confirm the boundary is correct for every feature. The existing `leakage_validators.py` validates expanding-window features but does not cover horse-history or pace features.

**Consequences:** If the current race leaks into past-performance features, the model can infer "this horse finished 1st" from a feature that supposedly only contains past data. The model appears to predict well in training but cannot replicate this in production (because the current race result is not yet known). This is the most devastating form of leakage -- it can turn a losing model into an apparently profitable one.

**Prevention:**
- For every time-series feature computed from past races, add an explicit assertion or test: for each (horse, current_race_date), verify that no feature value was computed from data on or after `current_race_date`.
- Extend `leakage_validators.py` to cover horse history features (`norm_finish_logit_avg`, `harontimel5_avg`, `timediff_avg`, etc.) and pace features (`pace_aptitude`, `front_pace_wr`).
- Add a canary test: for a specific horse with known race dates, compute features at a specific date and verify that only prior race data is included.

**Detection:**
- Check if model performance drops dramatically when using strict `race_date < current_date` vs. `race_date <= current_date` for feature computation. If performance is identical, the boundary is correct. If `<=` is much better, there is leakage.
- Check feature importance: if the top-3 features are all time-series features from past races, and the model's AUC drops by >10% when these features are shuffled, the model may be relying on leaked information.

**Phase:** Data validation phase, before any new time-series features are trusted.

**Sources:**
- [ResearchGate: Ensemble Learning for Horse Racing with Temporal Integrity](https://www.researchgate.net/publication/385301910_Optimizing_Horse_Racing_Predictions_through_Ensemble_Learning_and_Automated_Betting_Systems) -- time-series CV prevents leakage
- Codebase analysis: `pace_aptitude_features.py:218`, `horse_history_features.py`
- HIGH confidence: Boundary condition verification from code

---

### Pitfall 5: Pace Prediction Features Require Knowing Race Outcome (Circular Reasoning)

**What goes wrong:** Pace prediction features (e.g., `pace_pressure`, `pace_scenario_fit`) attempt to predict the pace scenario of the upcoming race. However, computing these features requires knowing the running style and early speed of every horse in the race -- which itself requires knowing how each horse performed in previous races under different pace scenarios. If the pace feature uses the current race's actual pace (derived from actual sectional times or running positions), it is using the outcome to predict the outcome.

**Why it happens:** "Pace pressure" requires estimating how many horses will vie for the lead. The estimation uses each horse's past `jyuni1c` (1st corner position) as a proxy for their likely early position. But `jyuni1c` in the current race is post-race data. The model must use ONLY past `jyuni1c` values (from previous races) to estimate the pace scenario. If the pace computation inadvertently includes the current race's `jyuni1c`, the model "knows" how fast the pace actually was before predicting the winner.

**Current codebase status:** `pace_aptitude_features.py` uses `history[history["race_date"] < ts]` (line 36) to filter past races, which is correct. But `pace_pressure` and `pace_scenario_fit` in `interaction_features.py` compute pace estimates from the current race's entrant characteristics. If these computations use any column that is populated post-race (e.g., `kyakusitukubun_cd` which is the running style classification assigned after the race), the feature is contaminated.

**Consequences:** The model learns "horses that ran on the front in a slow-paced race win more" -- but the "slow-paced race" determination uses the actual race outcome. The model appears to predict pace scenarios accurately, but cannot do so before the race starts. Backtest ROI is inflated.

**Prevention:**
- Audit every pace-related feature to confirm it uses ONLY pre-race data. The source of `kyakusitukubun_cd` must be the horse's TYPICAL running style from past races, not the current race's assigned style.
- Compute `pace_pressure` from the PREDICTED running styles of entrants (based on their historical jyuni1c/jyuni4c averages), not from actual current-race positions.
- Add a feature provenance test: for each pace feature, trace back to the source columns and verify none are in `POST_RACE_COLS`.

**Detection:**
- Run the model with pace features set to NaN. If performance barely changes, pace features were not adding signal (or were adding leaked signal). If performance drops significantly, determine whether the drop comes from genuine pre-race pace prediction or from leaked post-race information.
- Compare `pace_pressure` values computed from pre-race estimates vs. from actual race results. If they correlate at r > 0.7, the pre-race estimates are close to reality (good). If they correlate at r < 0.3, the pre-race estimates are unreliable (the feature is noise unless it is leaking).

**Phase:** Feature engineering phase, specifically when adding new pace features.

**Sources:**
- [Geegeez: Pace Bias Analysis](https://www.geegeez.co.uk/running-well-against-a-pace-bias-part-1/) -- difficulty of pace assessment without outcome knowledge
- [Sage Journals: Hindsight Bias in Predictions](https://journals.sagepub.com/doi/10.1177/17456916231204579) -- retrospective judgments contaminated by outcome knowledge
- Codebase analysis: `interaction_features.py`, `pace_aptitude_features.py`
- MEDIUM confidence: Requires line-by-line audit of pace feature computation to confirm

---

### Pitfall 6: Adding Stacking to hit_model But Not return_model Breaks EV Decomposition

**What goes wrong:** The `StackedEnsemble` is only applied to `hit_model` (P(win) classification), not to `return_model` (E(odds|win) regression). The EV calculation is `p_win_corrected * e_return_win_corrected`. If P(win) becomes more accurate through stacking but E(odds|win) remains the same, the EV estimates become miscalibrated because the two components were jointly calibrated under the old P(win).

**Why it happens:** The current stacking implementation only wraps the binary classification model (`WinTwoStageModel.hit_model`). The return model (`WinTwoStageModel.return_model`) remains a single LightGBM regression model. The EV correction model (`EVCorrectionModel`) was trained on predictions from the non-stacked hit model. When the stacked hit model produces different probability estimates (e.g., systematically higher for some horses), the correction model's learned biases no longer apply correctly.

**Current codebase status:** In `training_pipeline.py:439-454`, stacking is conditionally applied only to `win_2s.hit_model`. The return model is always single LightGBM (line 458-459). The EV correction model trains on the combined PxE output (line 461) which will differ depending on whether stacking was used.

**Consequences:** The EV estimates from the stacked model are not directly comparable to the non-stacked model. If the stacking push P(win) up by 2% on average, the EV correction model may overcorrect or undercorrect because it was calibrated for different P(win) distributions. The net effect could be worse calibrated EV estimates despite better P(win) accuracy.

**Prevention:**
- Always retrain the full pipeline (hit model -> return model -> EV correction) when changing the hit model architecture. Never swap only the hit model and keep downstream models unchanged.
- The `EVCorrectionModel` must be retrained AFTER the new hit model produces predictions, because its P-correction uses `init_score = logit(p_pred)` from the hit model.
- Consider stacking the return model as well, using regression variants (XGBRegressor, CatBoostRegressor, LGBMRegressor). Even if the return model stacking does not improve, the consistency ensures the EV decomposition is calibrated.
- Add a calibration check after stacking: compare the reliability diagram of EV estimates before and after stacking. If the diagram gets worse despite better P(win) accuracy, the EV decomposition is misaligned.

**Detection:**
- Compute `ev_win` predictions with and without stacking on the same data. If the mean or median `ev_win` shifts by more than 5%, the downstream correction is miscalibrated.
- Check the EV correction model's P-correction `raw_margin` distribution. If the distribution changes significantly after stacking (different mean, wider spread), the correction model needs retraining.

**Phase:** Integration testing phase, after stacking is implemented and before any backtest results are trusted.

**Sources:**
- Codebase analysis: `training_pipeline.py:439-459`, `ev_correction_model.py:189-291`
- HIGH confidence: Architectural dependency chain directly observable in code

---

## Moderate Pitfalls

### Pitfall 7: Feature Dimension Explosion From Adding Time-Series and Odds Features

**What goes wrong:** Adding 10-20 new time-series features (harontime trends, sectional time deltas, pace features) plus 5-10 new odds features (deviation features, volatility features, rolling market statistics) to the existing 100+ feature set creates a high-dimensional feature space. With ~50,000 training samples (4 years of JRA data), the feature-to-sample ratio becomes unfavorable for tree-based models. The model overfits to noise in the new features, especially the ones with high missingness.

**Why it happens:** LightGBM's `feature_fraction=0.7` (used in `WinTwoStageModel`) means each tree sees 70% of features. With 120+ features, many trees will see several noisy new features. If the new features are correlated with existing features (e.g., `harontime_late_trend` correlates with `form_trend`), they add noise without adding signal. The model wastes capacity on redundant features.

**Current codebase status:** `WinTwoStageModel.FEATURE_COLS` has ~30 features. `AbilityModel.FEATURE_COLS` has ~35 features. Adding 15-20 more features would push the total to ~50 per model. This is not catastrophic but combined with the 300-500 training samples per feature ratio, it increases overfitting risk.

**Prevention:**
- Run feature importance analysis AFTER adding new features and BEFORE training the production model. Drop any new feature with importance < 1% of the top feature's importance.
- Use `feature_fraction=0.5` (more aggressive column sampling) when feature count exceeds 40 to increase tree diversity.
- Group new features by source (time-series vs. odds vs. pace) and test each group independently. Only add groups that improve validation AUC by at least 0.5%.
- Apply mutual information filtering before training: compute MI between each new feature and the target, drop features with MI below a threshold.

**Detection:**
- Track validation AUC as features are added one by one. If AUC plateaus or declines after the Nth feature, stop adding.
- Monitor the gap between training AUC and validation AUC. If the gap widens significantly after adding features, overfitting is occurring.

**Phase:** Feature engineering phase, specifically during the addition of new feature groups.

**Sources:**
- Codebase analysis: `WinTwoStageModel.FEATURE_COLS`, `AbilityModel.FEATURE_COLS`
- HIGH confidence: Standard ML practice for tabular data with limited samples

---

### Pitfall 8: Training Time Explosion With 3x Model Training Per Submodel

**What goes wrong:** The current training pipeline takes ~44 minutes for 4 years of data with single LightGBM models. Stacking adds 3x training for the hit model (3 base models x K folds for OOF + 3 base models for full training). For 3-fold OOF with 2 surfaces, the stacking adds approximately 18 additional model training runs (3 models x 3 folds x 2 surfaces). Total training time could increase to 2-3 hours.

**Why it happens:** The pipeline already runs turf and dirt submodels in parallel (`ThreadPoolExecutor`, `training_pipeline.py:208`). But within each submodel, the stacking adds sequential K-fold training. The OOF generation is inherently sequential (each fold depends on the previous fold's training).

**Current codebase status:** `StackedEnsemble.train()` runs the 3 base models sequentially within each fold (lines 68-70). This could be parallelized but is not currently.

**Consequences:** Slower iteration cycles make it harder to experiment with different feature sets and hyperparameters. Developers may be tempted to skip retraining the ensemble after feature changes, leading to stale models.

**Prevention:**
- Parallelize the 3 base models within each OOF fold (train LightGBM, XGBoost, CatBoost simultaneously).
- Use fewer OOF folds (2 instead of 3) if data size permits. With time-series data, 2 folds already capture temporal variation.
- Pre-train base models once and only retrain the meta-learner when features change (if base model features are unchanged). This is a partial optimization but reduces iteration time.
- Consider training the stacking only for the final production model, not during development iterations. Use single LightGBM for development and switch to stacking for the final model.

**Detection:**
- Track training time per phase. If the stacking phase exceeds 30 minutes per surface, optimization is needed.

**Phase:** Infrastructure phase, during stacking integration.

**Sources:**
- Codebase analysis: `stacked_ensemble.py:59-84`, `training_pipeline.py:208-226`
- HIGH confidence: Training time is directly observable

---

### Pitfall 9: Odds Time-Series Features Unavailable at Inference Time

**What goes wrong:** New odds time-series features (e.g., `odds_velocity`, `odds_volatility`) are computed from historical odds snapshots during training. At inference time (paper trading or live betting), the same time-series data may not be available in the same format or granularity. The features become NaN during inference, causing the model to default to a degraded prediction path.

**Why it happens:** Training uses bulk historical odds data loaded from Parquet files (`load_odds_time_series_range`). Paper trading uses `OddsCollector` which may capture odds at different intervals or with different timing. The `_pick_target_snapshot` function in `odds_dynamics_features.py` requires specific snapshots at t-10, t-30, t-60 minutes before post. If the paper trading system does not capture snapshots at these exact times, the features cannot be computed.

**Current codebase status:** `odds_dynamics_features.py` returns NaN for all features when `odds_ts` is None or empty (lines 137-139). The `FeatureEngine.build_features()` method (for inference) may not have access to the same odds time-series data as `build_all()` (for training).

**Consequences:** The model was trained expecting odds velocity and volatility features. During inference, these are NaN. LightGBM handles NaN by sending samples down the default branch, which was learned for training samples with NaN features (which are rare in training because historical data is complete). The inference path is effectively different from the training path, causing unpredictable performance degradation.

**Prevention:**
- Ensure the inference path (`build_features`) receives the same odds time-series data format as the training path (`build_all`). The `OddsCollector` must capture snapshots at intervals compatible with `_pick_target_snapshot`'s target times.
- Add a feature availability check: before inference, verify that all features used in training have non-NaN values. If critical features are NaN, log a warning and fall back to a simplified model that does not require those features.
- During training, intentionally mask some features as NaN in a portion of the training data to make the model robust to missing features at inference time.

**Detection:**
- Compare feature NaN rates between training data and inference data. If a feature has 0% NaN in training but 50% NaN in inference, the model will behave unpredictably.
- Run inference with and without odds time-series features. If predictions differ significantly, the model depends on features that may not be reliably available.

**Phase:** Integration testing phase, when connecting new features to the inference pipeline.

**Sources:**
- Codebase analysis: `odds_dynamics_features.py:137-139`, `feature_engine.py`
- HIGH confidence: Directly observable data availability gap

---

### Pitfall 10: Regime Detector Not Retrained After Model Architecture Changes

**What goes wrong:** The `RegimeDetector` classifies market conditions into 3 states (aggressive/conservative/collapsed) using features derived from model predictions (e.g., `rolling_roi`, `market_error`). When the model architecture changes (e.g., adding stacking changes the prediction distribution), the regime detector's learned boundaries become invalid. It may classify a "good" regime as "collapsed" because the model's error distribution shifted.

**Why it happens:** The regime detector is trained on market-level features computed from the OLD model's predictions. When the model changes, the prediction errors change, the rolling ROI changes, and the market efficiency metrics change. The regime boundaries (learned from the old model's prediction patterns) no longer apply.

**Current codebase status:** `RegimeDetector` in `training_pipeline.py:267-271` is trained after all submodels. Its features include `market_efficiency` computed from favorite win rate and overround, which are model-independent. However, the `rolling_roi` feature (if used) depends on the model's betting performance. If stacking changes ROI from 89% to 105%, the regime detector's ROI-based features shift.

**Consequences:** The regime detector may incorrectly suppress betting when the new model is actually performing well, or may allow aggressive betting when the new model is misfiring. The regime detector's strategy parameters (edge_threshold, stake_fraction) become misaligned with the model's actual edge.

**Prevention:**
- Always retrain the regime detector after any model architecture change. The training pipeline already does this (`training_pipeline.py:267-271`), but the regime detector's features must also be recomputed with the new model's predictions.
- Consider making the regime detector purely model-independent: use only market-level features (overround, entropy, favorite win rate) that do not depend on the model's predictions.
- Add a sanity check: after retraining, verify that the regime detector's classification distribution is not dramatically different from the previous version (e.g., if previously 40% aggressive, 40% conservative, 20% collapsed, and now 10% aggressive, 80% conservative, 10% collapsed, something changed that needs investigation).

**Detection:**
- Compare regime state distributions before and after model changes. Large shifts indicate the regime detector is responding to changed model behavior, not changed market conditions.
- Test backtest ROI with regime detector enabled vs. disabled using the new model. If enabling the regime detector reduces ROI, it is misaligned.

**Phase:** After any model architecture change, during integration testing.

**Sources:**
- Codebase analysis: `regime_detector.py`, `training_pipeline.py:267-271`
- HIGH confidence: Architectural dependency directly observable

---

## Minor Pitfalls

### Pitfall 11: Model Serialization Format Change Breaks Backward Compatibility

**What goes wrong:** The `StackedEnsemble` is serialized as joblib (`.joblib`) while single LightGBM models use native format (`.lgb`). The `ModelLoader._load_hit_model()` method handles both formats (lines 447-472), but the `meta.json` must be updated to record `use_ensemble=true/false`. If a developer trains with stacking enabled but forgets to update meta.json, or if the backtest uses a different meta.json than training, the model loading fails silently (loads single LightGBM instead of ensemble).

**Prevention:** Always update meta.json automatically during training. Never rely on manual flag setting. The `training_pipeline.py` should write `use_ensemble` to meta.json in the `_log_to_mlflow` or equivalent save step.

**Detection:** Compare model predictions loaded from `.joblib` vs. `.lgb`. If they differ for the same data, there is a loading mismatch.

**Phase:** Infrastructure phase, during model save/load integration.

---

### Pitfall 12: Memory Usage Spike During Stacking OOF Generation

**What goes wrong:** The stacking OOF generation creates 3 K-fold models for each base learner (9 total models in memory for 3-fold x 3 models). For 200K training rows with 50 features, each LightGBM/XGBoost/CatBoost model consumes 100-500MB of memory. During OOF generation, all intermediate models are in memory simultaneously, causing potential OOM on systems with less than 16GB RAM.

**Prevention:** Delete fold models immediately after generating OOF predictions. Only the final full-data models need to be retained. Add explicit `del` and `gc.collect()` calls in the fold loop.

**Detection:** Monitor peak memory usage during stacking training. If it exceeds 8GB, add model cleanup.

**Phase:** Infrastructure phase.

---

### Pitfall 13: Odds Deviation EV Double-Counting

**What goes wrong:** If odds deviation features (e.g., `odds_to_ability_ratio`) are added to the hit model features AND the model output is multiplied by odds in the EV calculation, the market information is effectively used twice. The hit model learns "when p_market >> p_ability, this horse is underbet by the market" and assigns higher P(win). Then EV = P(win) * odds captures the market inefficiency again through the odds term. This double-counts the edge.

**Why it happens:** The `odds_to_ability_ratio` is defined as `p_market / p_ability`. A value > 1.0 means the market rates the horse higher than the model. If the hit model uses this feature and learns to increase P(win) for high-ratio horses, the EV calculation `P * odds` will reflect both the increased P (from the feature) and the odds (which are the source of the feature). The edge is amplified beyond what is real.

**Prevention:** Decide on one path: either (a) use odds deviation features in the hit model and compute EV = P * odds, or (b) do not use odds deviation features in the hit model and instead use them in a Benter-style combination after prediction. Never do both.
- Path (a) is simpler but makes the model dependent on having pre-race odds at prediction time.
- Path (b) is cleaner conceptually because it separates the fundamental model (no odds) from the market combination step. This is the Benter approach already used for place predictions.

**Current codebase status:** `odds_to_ability_ratio` is already in `WinTwoStageModel.FEATURE_COLS` (line 88). The EV calculation uses `confirmed_odds` or pre-race odds. This means the feature is already being double-counted. The existing v1.0 implementation chose path (a).

**Detection:** Compare model predictions with and without `odds_to_ability_ratio`. If removing the feature causes P(win) to drop for horses where p_market > p_ability, the model is learning from the ratio. Then check if the EV estimates are inflated for these horses relative to actual returns.

**Phase:** Feature engineering phase, when evaluating odds deviation features.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Stacking implementation | Pitfall 1: Correlated base models | Force diversity via different hyperparameters per model |
| Stacking OOF generation | Pitfall 2: Temporal leakage from full-data retraining | Validate meta-learner calibration on held-out data |
| Odds features (any) | Pitfall 3: Look-ahead bias from near-post odds | Tighten snapshot tolerance to 5min max |
| Time-series horse features | Pitfall 4: Current race leaks into past features | Add boundary assertions for every searchsorted call |
| Pace prediction features | Pitfall 5: Circular reasoning with post-race data | Audit pace feature provenance against POST_RACE_COLS |
| Stacking integration | Pitfall 6: EV decomposition miscalibration | Retrain full pipeline including EV correction after stacking |
| Feature expansion | Pitfall 7: Dimension explosion with limited data | Feature importance filtering before production training |
| Training infrastructure | Pitfall 8: Training time 3x increase | Parallelize base models within OOF folds |
| Inference pipeline | Pitfall 9: Odds TS features unavailable at inference | Ensure inference data matches training data format |
| Model architecture change | Pitfall 10: Regime detector misalignment | Always retrain regime detector after model changes |
| Model save/load | Pitfall 11: Serialization format confusion | Auto-write meta.json during training |
| Stacking memory | Pitfall 12: OOM during OOF generation | Delete fold models after OOF prediction |
| Odds deviation features | Pitfall 13: EV double-counting | Choose one path: feature-in-model OR Benter-combination |

---

## Priority Action Items (Ordered by Risk Severity)

1. **Audit odds snapshot timing** (Pitfall 3) -- Before trusting any odds-based feature results, validate that `_pick_target_snapshot` tolerance is tightened and no fallback to confirmed_odds exists in feature paths. This is a data-integrity prerequisite.

2. **Validate stacking meta-learner on held-out data** (Pitfall 2) -- Before using the stacked model for any decisions, compare meta-learner predictions using OOF base models vs. full-data base models on the 2024 backtest year. If predictions differ by >2%, recalibrate or switch to Option A (no full-data retraining).

3. **Force base model diversity** (Pitfall 1) -- Before training the production stacked model, set different hyperparameters for each base model. At minimum: different `num_leaves` and `feature_fraction`.

4. **Add time-series boundary assertions** (Pitfall 4) -- Extend `leakage_validators.py` to cover horse history and pace features. Add `assert` statements in `horse_history_features.py` and `pace_aptitude_features.py`.

5. **Audit pace feature provenance** (Pitfall 5) -- Trace the source of `kyakusitukubun_cd` used in pace computation. Confirm it comes from past races only, not from the current race's post-race classification.

6. **Plan full pipeline retrain after stacking** (Pitfall 6) -- Ensure the training pipeline retrain order is: hit model (stacked) -> return model -> EV correction model. Never reuse old downstream models.

---

## Sources

### HIGH confidence (codebase analysis + established theory)
- [StackExchange: Stacking with Cross-Validation](https://stats.stackexchange.com/questions/239445/how-to-properly-do-stacking-meta-ensembling-with-cross-validation) -- meta-learner separation principle
- [ML4Devs: Ensemble Methods -- Overfitting Risk](https://www.ml4devs.com/what-is/ensemble-learning-bagging-boosting-stacking/) -- meta-learner overfitting with correlated base models
- [Look-Ahead Bias in Backtests (Michael Harris)](https://mikeharrisny.medium.com/look-ahead-bias-in-backtests-and-how-to-detect-it-ad5e42d97879) -- temporal information leakage
- Codebase analysis: `stacked_ensemble.py`, `training_pipeline.py`, `odds_dynamics_features.py`, `ev_correction_model.py`, `pace_aptitude_features.py`

### MEDIUM confidence (domain literature + inference)
- [ResearchGate: Optimizing Horse Racing Through Ensemble Learning](https://www.researchgate.net/publication/385301910_Optimizing_Horse_Racing_Predictions_through_Ensemble_Learning_and_Automated_Betting_Systems) -- time-series CV for racing
- [Geegeez: Pace Bias Without Outcome Knowledge](https://www.geegeez.co.uk/running-well-against-a-pace-bias-part-1/) -- difficulty of pace assessment
- [Sage Journals: Hindsight Bias in Predictions](https://journals.sagepub.com/doi/10.1177/17456916231204579) -- retrospective contamination
- [Kaggle: Practical Stacking Discussion](https://www.kaggle.com/general/24748) -- OOF vs full-data retraining tradeoffs

### LOW confidence (needs validation during implementation)
- Exact correlation between LightGBM/XGBoost/CatBoost predictions on this specific dataset (needs empirical measurement)
- Specific JRA odds drift magnitude in final 10 minutes (needs domain data analysis)
- Whether `kyakusitukubun_cd` in pace features comes from current race or past race (needs code tracing)

---
*Research completed: 2026-05-03*
*Ready for roadmap: yes*
