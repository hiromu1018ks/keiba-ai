# Domain Pitfalls: Ensemble Filter Recalibration (v1.4)

**Domain:** Recalibrating betting filters for ensemble model output distribution
**Context:** keiba-ai v1.4 milestone -- the 3-model stacked ensemble (LightGBM + XGBoost + CatBoost -> Ridge) produces a different probability distribution than the single LightGBM used to calibrate WinSelectionGate, EV_lower threshold, OddsBandFilter, and Optuna parameters. The symptom is 7 bets/year at ROI 0% (vs. target 100+ bets/year at ROI >100%). The ML model pipeline is frozen; only filter parameters and calibration change.
**Researched:** 2026-05-05
**Confidence:** HIGH (based on direct codebase analysis of all filter components, ensemble model, training pipeline, and strategy optimizer)

---

## Critical Pitfalls

Mistakes that cause complete filter rejection (0 bets), silent look-ahead bias, or fictitious ROI improvement.

### Pitfall 1: WinSelectionGate Quantile Bin Mismatch

**What goes wrong:**
`WinSelectionGateModel` stores `prob_edges`, `edge_edges`, and `odds_edges` computed from single-LightGBM OOF predictions (quantile-based binning via `_quantile_edges()` at win_selection_gate.py:57-71). When the ensemble produces a different `win_selection_prob` distribution, candidates land in wrong bins. The `combo_scores` and `pair_scores` lookup tables return stale bucket statistics, making `win_gate_score` invalid.

The ensemble's Ridge meta-learner produces probabilities that are (a) smoother (less extreme highs/lows), (b) calibrated differently (the Ridge alpha=1.0 regularization pulls toward the mean), and (c) have a different variance structure across the 3 base model predictions. A candidate with `win_selection_prob = 0.12` under the old edges might map to bin 3, but under ensemble-calibrated edges it would be bin 2. The `_score_row_from_tables()` lookup returns a score from a bucket trained on a different distribution.

**Why it happens:**
The quantile binning is data-dependent by design. `_quantile_edges()` computes `np.linspace(0, 1, n_bins+1)` quantiles from the training data. If the ensemble's probability distribution has a different shape (e.g., narrower interquartile range, different tail behavior), the bin boundaries shift. The `combo_scores` dict keys are `(prob_bin, edge_bin, odds_bin)` tuples -- any shift in any dimension makes the majority of lookup keys miss.

**Consequences:**
The `win_gate_score` for most candidates falls back to `global_score` (the overall mean, typically ~1.0). The `win_gate_pass` flag uses hard thresholds on `win_selection_prob`, `win_selection_edge`, and `tanoddslow` (lines 996-1001) which were also tuned on single-model distribution. Combined, this causes the gate to reject nearly all candidates.

**Prevention:**
1. Re-run `WinSelectionGateModel.train()` with ensemble OOF predictions. The training pipeline must be run with `use_ensemble=True`, and the resulting OOF `df` (which contains `p_win_corrected`, `ev_win_corrected` from the ensemble) must be passed to the gate's `train()` method.
2. Never load a single-model gate `.joblib` file for ensemble predictions. Verify by checking that the gate's `prob_edges` quantiles match the current model's output distribution.
3. After retraining, log the gate's new thresholds (`min_prob`, `min_edge`, `max_odds`) and compare with the old ones. If they change by more than 20%, the old gate was severely mismatched.

**Detection:**
- Compare `win_selection_prob` statistics (mean, std, p5/p95) between single and ensemble predictions on the same held-out data. If they differ by more than 10%, gate bins are stale.
- Log `self.prob_edges` after loading and compare against quantiles of current model output.
- If `n_ev_excluded` jumps dramatically (from ~500 to 3594 as documented in PROJECT.md), this confirms the mismatch.

**Phase to address:** First v1.4 task (WinSelectionGate retraining). Must complete before threshold tuning or Optuna.

---

### Pitfall 2: EV_lower Threshold Distribution Shift

**What goes wrong:**
`EV_lower_win_corrected >= 1.0` is a hard filter in `get_win_candidates()` (race_predictor.py:437-441). The `EV_lower` is computed by `RobustConfidenceEstimator.predict_interval()` which stores conformal residual quantiles (`_q_hat_*`) from the calibration set. When the ensemble produces different prediction errors, the conformal interval width is miscalibrated.

More critically, the ensemble's `ev_win_corrected` (P_corrected * E_corrected) has a different distribution because both P and E correction models were trained on single-model predictions. The EVCorrectionModel uses `init_score = logit(p_win_pred)` (ev_correction_model.py:208) -- if `p_win_pred` is systematically different from the ensemble's output, the correction margin is applied to the wrong baseline. This pushes `ev_win_corrected` lower, which pushes `EV_lower_win_corrected` further below 1.0.

**Why it happens:**
The chain is: ensemble hit model produces different `p_win_pred` -> `EVCorrectionModel.correct_ev()` applies a correction margin that was trained on single-model `p_win_pred` -> `p_win_corrected` is biased -> `ev_win_corrected = p_win_corrected * e_return_win_corrected` is biased -> `EV_lower_win_corrected` is biased -> the `>= 1.0` filter rejects most candidates.

The E-correction model compounds this: it uses `1/sqrt(p_win_pred)` as sample weights (ev_correction_model.py:254), so its training is also sensitive to the probability distribution.

**Consequences:**
The documented symptom of 3594 EV exclusions out of total candidates is consistent with the ensemble's corrected EV being systematically lower than the single model's. The `EV_lower_win_corrected >= 1.0` filter becomes the dominant exclusion mechanism.

**Prevention:**
1. The `EVCorrectionModel` and `RobustConfidenceEstimator` must be retrained alongside the ensemble. The training pipeline handles this automatically when `use_ensemble=True`.
2. Make the `EV_lower` threshold adaptive. Instead of fixed 1.0, use a percentile of the ensemble's OOF `EV_lower` distribution among winners. For example, set threshold = 60th percentile of `EV_lower_win_corrected` for horses with `kakuteijyuni == 1` in OOF data.
3. Log the distribution of `EV_lower_win_corrected` on a sample of ensemble predictions before setting any threshold.

**Detection:**
- Compute `EV_lower_win_corrected.describe()` on ensemble predictions. If median is below 0.90, the threshold 1.0 is too aggressive.
- Count exclusions: if `n_ev_excluded` exceeds 80% of total candidates, the threshold needs recalibration.
- Compare `ev_win_corrected` distribution between single-model and ensemble. If ensemble mean is 0.05+ lower, the correction models need retraining.

**Phase to address:** Second v1.4 task (EV_lower dynamic threshold). Depends on WinSelectionGate retraining completing first.

---

### Pitfall 3: OddsBandFilter Look-Ahead Bias via training_bet_history

**What goes wrong:**
The `StrategyOptimizer._run_single_backtest()` (strategy_optimizer.py:150-184) generates `training_bet_history` by running a backtest on the training period with the current trial's strategy parameters (including Optuna-optimized `roi_threshold`, `ev_threshold`, etc.). This training bet history is then passed to the test-period `engine.run()` for `OddsBandFilter.calibrate()`.

The look-ahead bias is subtle: the training bet history is generated using strategy parameters that Optuna is simultaneously optimizing to maximize test-period ROI. When Optuna discovers that a certain `roi_threshold` value works well on the test period, the training bet history is retroactively filtered through that threshold. The OddsBandFilter then "learns" band exclusions that are indirectly informed by test-period performance.

**Why it happens:**
In `_run_single_backtest()`, the same `strategy_config` dict (built from Optuna trial parameters) is used for both the training backtest (line 161) and the test backtest (line 181). The `regime_overrides` are injected into the RegimeDetector (line 148), affecting which races are skipped (COLLAPSED) in the training period based on test-optimized parameters. This means the training bet history's composition (which races were bet on, at what odds) depends on parameters chosen for test-period performance.

**Consequences:**
The OddsBandFilter will show inflated ROI during Optuna optimization because the band exclusions encode test-period information. In live trading or on truly unseen data, the filter will underperform. The Optuna "best" parameters may include an OddsBandFilter configuration that only works because the training bet history was generated with test-informed parameters.

**Prevention:**
1. Generate `training_bet_history` with default (non-optimized) strategy parameters. Use `strategy_params=None` for the training backtest, not the Optuna-tuned values.
2. Alternatively, use the previous walk-forward fold's finalized parameters for generating the current fold's training bet history.
3. Add a verification step: log the parameters used for training bet history generation and confirm they are NOT the Optuna-optimized values.

**Detection:**
- If Optuna's best `roi_threshold` consistently differs from default (1.0) by more than 0.1, look-ahead bias is likely present.
- Compare ROI on a held-out validation year vs. Optuna-reported ROI. If gap exceeds 5 percentage points, the filter is likely overfitted.
- Check if training bet history count varies significantly across Optuna trials. If it does, the strategy params are affecting the training data composition.

**Phase to address:** OddsBandFilter rebuild task. Must generate training bets with non-optimized parameters.

---

### Pitfall 4: Optuna Overfitting to Backtest Period

**What goes wrong:**
The `StrategyOptimizer` searches 14 dimensions (6 regime + 5 DD control + 2 EV scaling + 1 OddsBandFilter) to maximize walk-forward ROI. With only 2 folds (2024 and 2025 test), 100 trials, and a highly stochastic target (horse racing ROI has massive variance from a small number of high-payout winners), the optimizer finds parameters that fit the specific outcome sequence but fail to generalize.

Horse racing ROI is dominated by a few longshot winners. In a year with ~5000 JRA races, a single 50-1 winner passing through the filter can swing total ROI by 20+ percentage points. Optuna's TPE sampler will discover parameter combinations that happen to include those specific longshots (e.g., setting `ev_aggressive` just low enough, `fk_aggressive` just high enough) while excluding the losing bets. With 14 free parameters and only ~5000 data points per fold, the optimization has far more degrees of freedom than the data can constrain.

**Why it happens:**
The objective function `_objective()` at strategy_optimizer.py:194-224 computes `mean_roi` across folds with a minimum bet count constraint. The constraint (`min_bets_per_fold=1000`) helps but is insufficient for 14 dimensions. The search space includes highly correlated parameters (e.g., `fk_aggressive` and `ev_aggressive` both affect which bets pass and how much is staked on them), creating ridges in the optimization landscape where many parameter combinations give similar results on the specific test data.

**Consequences:**
The "optimal" parameters produce 100%+ ROI in 2024-2025 backtests but collapse to below 80% ROI on any other period. The system appears profitable during development but is actually overfitted to two specific years of race outcomes.

**Prevention:**
1. Increase fold count from 2 to at least 4 (2022, 2023, 2024, 2025) with expanding training windows.
2. Add parameter stability checks: if Optuna's top-5 trials have substantially different parameter values, the optimization surface is flat and the "best" is not reliable.
3. Apply parameter rounding: round the best parameters to the nearest grid point and verify the rounded values achieve >95% of the best ROI.
4. Run optimization with 5 different random seeds. If best parameters differ substantially across seeds, the optimization is unstable.
5. Consider reducing the search space. The most impactful parameters are `fk_aggressive`, `ev_aggressive`, `ev_conservative`, and `roi_threshold`. Freeze the less impactful DD parameters to reasonable defaults.

**Detection:**
- Compare per-fold ROI for the best trial. If one fold has 150% and the other has 50%, the parameters are fitting fold-specific noise.
- Run the best parameters on a year not included in the optimization (e.g., 2022). If ROI drops below 85%, the parameters are overfitted.
- Check parameter stability across top-10 trials. High variance = unstable optimization.

**Phase to address:** Final v1.4 task (Optuna optimization). Must complete after all other recalibrations.

---

## Moderate Pitfalls

### Pitfall 5: Model Output Distribution Mismatch Between OOF Training and Inference

**What goes wrong:**
The `StackedEnsemble` trains the Ridge meta-learner on K-fold OOF predictions from fold-specific base models (stacked_ensemble.py:69-98). At inference, the base models are retrained on ALL training data (lines 100-105), producing slightly different (typically more confident) predictions. The Ridge coefficients, optimized for the less-confident OOF predictions, now weight overly confident inputs.

This is the classic stacking distribution shift: the meta-learner learns `weights` such that `Ridge(OOF_preds) ~ y`, but `OOF_preds` come from fold models that saw 67-80% of the data, while inference preds come from models that saw 100% of the data. The systematic difference means the meta-learner's output is calibrated for OOF predictions but not for inference predictions.

**Why it happens:**
- Base models trained on more data tend to be more confident (lower log loss, sharper predictions).
- The Ridge alpha=1.0 regularization at line 97 provides some protection but does not address systematic shifts.
- The current K-fold uses expanding windows (lines 76-77: `val_start = int(n * (i+1) / (n_folds+1))`), which means early folds have less training data and later folds have more. The OOF predictions are heterogeneous in quality.

**Consequences:**
The ensemble's inference predictions are systematically biased compared to training predictions. This affects all downstream components: `ev_win_corrected`, `EV_lower_win_corrected`, `win_selection_prob`, and ultimately all filter decisions.

**Prevention:**
1. Monitor the ensemble's inference output distribution. Compare `p_win_pred` statistics from OOF predictions vs. inference predictions on the same data. If mean shifts by more than 0.02, consider adding calibration.
2. Add isotonic regression or temperature scaling on top of the Ridge meta-learner, trained on the OOF predictions as a held-out calibration set. The training pipeline already has infrastructure for this (`win_isotonic_calibrator`, `win_temperature_scaler` in SubmodelSet).
3. The `np.clip(output, 0, 1)` at line 127 prevents out-of-range predictions but does not address systematic bias.

**Detection:**
- Compare `p_win_pred.describe()` between OOF and inference on the same year's races. Mean shift > 0.02 is a warning sign.
- If ensemble's `win_selection_edge` distribution is systematically shifted left compared to single-model on the same data, this confirms distribution mismatch.

**Phase to address:** Check during WinSelectionGate retraining. If mismatch is large (>0.02 mean shift), add calibration step.

---

### Pitfall 6: SubmodelSet.use_ensemble Flag Inconsistency

**What goes wrong:**
`SubmodelSet.use_ensemble` (domain/models.py:245) controls serialization/deserialization paths. If the ensemble model is loaded from disk but this flag remains `False`, MLflow logging and model saving use the wrong code paths. While the `hit_model` IS a `StackedEnsemble` instance and will produce correct predictions, the loading logic in `StrategyOptimizer` (strategy_optimizer.py:137: `use_ensemble_override=True`) must correctly set this flag for all surfaces.

**Why it happens:**
The flag is set in the training pipeline when `use_ensemble=True` is passed to `run()`. But when models are loaded via `ModelLoader.load_from_dir()`, the flag must be reconstructed from the saved state. If the model directory contains a mix of ensemble and non-ensemble artifacts, or if the loader logic has a code path that misses the flag, it stays `False`.

**Consequences:**
Predictions are correct (the StackedEnsemble is called), but diagnostic logging, model saving, and gate model loading may behave differently. This can cause confusion during debugging: the model appears to be an ensemble but logs suggest otherwise.

**Prevention:**
1. After loading models, explicitly check `sub.use_ensemble` for each surface submodel. Log a warning if `False` but `isinstance(sub.win.hit_model, StackedEnsemble)`.
2. Add a type check assertion at the start of backtest: `assert all(isinstance(sub.win.hit_model, StackedEnsemble) == sub.use_ensemble for sub in models.submodels.values())`.

**Detection:**
- If backtest results with `--ensemble` are identical to non-ensemble results, the flag is not being propagated.

**Phase to address:** Verify at start of first v1.4 task. Quick sanity check.

---

### Pitfall 7: EVCorrectionModel Init_Score Dependency on Hit Model

**What goes wrong:**
`EVCorrectionModel.train()` uses `init_score = logit(p_win_pred)` as the starting point for the P-correction LightGBM model (ev_correction_model.py:208-209). This means the P-correction model learns an additive adjustment on top of the raw prediction. When the hit model changes from single LightGBM to StackedEnsemble, `p_win_pred` changes distribution, and the init_score shifts. The P-correction model's learned margin is calibrated for the single-model's logit space, not the ensemble's.

**Why it happens:**
The `init_score` mechanism in LightGBM allows incremental training -- the model only needs to learn the correction residual. But the correction residual's distribution depends on the base prediction. A margin of +0.1 in logit space means different probability adjustments depending on whether the base logit is -2.0 (low confidence) or +1.0 (high confidence).

**Consequences:**
`p_win_corrected` is systematically biased. If the ensemble's `p_win_pred` tends to be more moderate (pulled toward 0.5 by Ridge regularization), the init_score logit values are smaller in magnitude, and the P-correction model's learned margin overcorrects or undercorrects.

**Prevention:**
1. The training pipeline handles this automatically: when `use_ensemble=True`, the ensemble `p_win_pred` feeds into `EVCorrectionModel.train()`, producing a correctly calibrated P-correction model.
2. If loading pre-trained models via `ModelLoader`, verify the EVCorrectionModel was trained with the ensemble hit model, not the single model.

**Detection:**
- Compare `p_win_corrected` distribution between single-model and ensemble on the same data. Mean shift > 0.03 suggests the P-correction needs retraining.
- Compare `ev_win_corrected` distributions. If the ensemble's mean is more than 0.05 different from single-model's, the correction pipeline needs retraining.

**Phase to address:** Verify during WinSelectionGate retraining. Automatic with pipeline retraining using `use_ensemble=True`.

---

### Pitfall 8: RegimeDetector Sensitivity After Ensemble Switch

**What goes wrong:**
The `RegimeDetector` uses `market_error_std` and `market_error_mean` as features (regime_detector.py:49-62). These are computed from `signed_log_error_win = log(p_market) - log(p_pred)`. When the ensemble produces different `p_pred` values, the log errors change, shifting these features. The RegimeDetector's LightGBM model was trained on single-model errors and may misclassify regime states when given ensemble errors.

**Why it happens:**
The ensemble's predictions are typically more accurate (lower log error variance) but may be more biased in certain probability ranges. The RegimeDetector was trained to classify market states based on the single model's error patterns. If the ensemble's errors have a different mean/variance relationship, the regime boundaries shift.

**Consequences:**
More races classified as COLLAPSED (causing excessive skip via the `skip=True` parameter) or fewer (causing bets in poor conditions). The regime distribution change cascades through the entire filter chain.

**Prevention:**
1. The RegimeDetector is retrained automatically when the training pipeline runs with `use_ensemble=True`.
2. After retraining, compare regime distributions: count races in each state for a historical period and compare single vs. ensemble. If COLLAPSED rate changes by more than 5%, investigate.
3. The `regime_overrides` mechanism (injecting Optuna-optimized parameters into `_override_params`) should be re-evaluated for the ensemble regime distribution.

**Detection:**
- Log regime state distribution during backtest. If COLLAPSED rate jumps from ~10% to ~30%, the RegimeDetector needs retraining.
- Monitor `n_collapsed_skipped` in `BacktestResult`. Dramatic changes after ensemble switch indicate this issue.

**Phase to address:** Verify during WinSelectionGate retraining. Automatic with pipeline retraining.

---

## Minor Pitfalls

### Pitfall 9: WinSelectionGate Soft Pass Buffer Constants

**What goes wrong:**
`WinSelectionGateModel` has hardcoded soft pass margins: `SOFT_PROB_BUFFER = 0.01`, `SOFT_EDGE_BUFFER = 0.02`, `SOFT_ODDS_BUFFER = 1.0` (lines 112-114). These were reasonable for the single model's distribution range. With the ensemble's typically narrower probability range, these buffers may be proportionally too large, letting in too many near-miss candidates.

**Prevention:**
Retune these constants after ensemble OOF training. Consider making them proportional to the distribution's standard deviation (e.g., `SOFT_PROB_BUFFER = 0.1 * std(prob_distribution)`).

**Detection:**
If `soft_gate` selection reason dominates in diagnostic logs after ensemble recalibration, the soft buffers are too generous.

---

### Pitfall 10: OddsBandFilter Band Boundary Sensitivity

**What goes wrong:**
The four fixed bands (1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+) may split a profitable region after the ensemble shifts which odds ranges produce edge. A band that was unprofitable under the single model might be profitable under the ensemble, or vice versa.

**Prevention:**
Log exact ROI and sample count per band. If a band has fewer than 100 samples, merge it with an adjacent band before deciding exclusion. Consider adding a minimum sample threshold (e.g., 200 bets) to the `calibrate()` method before marking a band as excluded.

**Detection:**
If `band_counts` in OddsBandFilter calibration shows a band with fewer than 50 samples, the band statistics are unreliable and should not be used for exclusion decisions.

---

### Pitfall 11: Kelly Fraction Interaction with Ensemble Probability Scale

**What goes wrong:**
`StakeCalculator.calc_stake()` computes `kelly_fraction = edge / (odds - 1.0)` (stake_calculator.py:59). The `edge` comes from `win_selection_edge = ev - 1.0`. If the ensemble's EV estimates are systematically different from the single model's, the Kelly fractions change proportionally. This affects stake sizing even if the fractional Kelly parameter is correctly tuned.

**Prevention:**
The Optuna optimization handles this by tuning `fractional_kelly` for each regime. Ensure the optimization runs after ensemble recalibration so it accounts for the new edge distribution.

**Detection:**
If average stake size changes by more than 30% after ensemble recalibration (without Kelly parameter changes), the edge distribution has shifted significantly and the Kelly calculation is affected.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| WinSelectionGate retraining | Pitfall 1: Quantile bin mismatch | Retrain gate with ensemble OOF predictions; never load single-model gate |
| EV_lower dynamic threshold | Pitfall 2: Distribution shift + Pitfall 7: EVCorrectionModel dependency | Retrain confidence estimator with ensemble; compare EV_lower distributions |
| OddsBandFilter rebuild | Pitfall 3: Look-ahead bias in training_bet_history | Generate training bets with default (non-optimized) strategy params |
| Optuna 14-dim optimization | Pitfall 4: Overfitting to backtest period | Increase folds to 4+; add stability checks; round parameters |
| Regime re-evaluation | Pitfall 8: RegimeDetector threshold sensitivity | Retrain RegimeDetector with ensemble predictions; compare regime distributions |
| Pipeline integration | Pitfall 5: OOF/inference distribution mismatch | Compare OOF vs. inference prediction statistics |
| Model loading | Pitfall 6: use_ensemble flag inconsistency | Explicitly verify flag after loading models |

## Prevention Checklist

Before starting v1.4 tasks, verify:

- [ ] Training pipeline runs with `use_ensemble=True` and produces new models
- [ ] WinSelectionGate `.joblib` file is from the ensemble run (check timestamp, compare prob_edges)
- [ ] `RobustConfidenceEstimator` residuals are from ensemble predictions
- [ ] `RegimeDetector` was trained with ensemble `signed_log_error_win` values
- [ ] `EVCorrectionModel` was trained with ensemble `p_win_pred` init_scores
- [ ] `training_bet_history` for OddsBandFilter uses default params, not Optuna params
- [ ] Optuna optimization uses at least 4 folds with expanding windows
- [ ] Ensemble prediction distribution is logged and compared to single-model distribution
- [ ] `SubmodelSet.use_ensemble` flag is `True` after model loading

## Warning Signs During Implementation

| Symptom | Likely Root Cause | Immediate Check |
|---------|-------------------|-----------------|
| Ensemble backtest produces <20 bets/year | WinSelectionGate using single-model bins | Retrain gate with ensemble OOF data |
| EV_lower excludes >80% of candidates | Confidence estimator trained on single model | Retrain confidence estimator with ensemble |
| Optuna best ROI > 150% but validation ROI < 90% | Overfitting to backtest period | Increase folds, reduce search dimensions |
| OddsBandFilter excludes all bands | Training bet history too small or biased | Check training bet count and params used for generation |
| Regime stuck in COLLAPSED for >50% of races | RegimeDetector using single-model error stats | Retrain RegimeDetector with ensemble |
| Identical results with and without --ensemble | use_ensemble flag not propagated | Check SubmodelSet.use_ensemble for all surfaces |
| Win gate score == global_score for most candidates | Quantile bins not matching distribution | Verify prob_edges against current model output quantiles |

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Stale WinSelectionGate bins | LOW | Re-run training pipeline with use_ensemble=True; retrain gate from OOF output |
| EV_lower threshold too aggressive | LOW | Compute adaptive threshold from ensemble OOF winner distribution; retrain confidence estimator |
| OddsBandFilter look-ahead bias | MEDIUM | Regenerate training bet history with default params; re-run Optuna with corrected pipeline |
| Optuna overfitting | MEDIUM | Increase fold count; reduce search space; add stability checks across seeds |
| Distribution mismatch (OOF vs inference) | LOW | Add isotonic/temperature calibration on meta-learner output |
| Regime detector sensitivity | LOW | Pipeline retraining handles this automatically; verify regime distribution |

## Sources

### HIGH confidence (direct codebase analysis)
- `src/models/stacked_ensemble.py:1-607` -- ensemble training, OOF generation, Ridge meta-learner
- `src/models/win_selection_gate.py:1-1113` -- quantile binning, score tables, WF OOF training
- `src/betting/odds_band_filter.py:1-112` -- band calibration and filtering
- `src/tuning/strategy_optimizer.py:1-273` -- Optuna optimization, training bet history generation
- `src/backtest/engine.py:1-1207` -- filter chain execution, OddsBandFilter integration
- `src/backtest/race_predictor.py:1-925` -- EV filter, candidate selection, win bet generation
- `src/models/ev_correction_model.py:1-575` -- P/E correction with init_score, sample weights
- `src/models/regime_detector.py:1-264` -- regime classification features, strategy params
- `src/pipelines/training_pipeline.py:1-1300+` -- ensemble wiring, use_ensemble flag propagation
- `src/domain/models.py:1-283` -- SubmodelSet, TrainedModelsV5 dataclass definitions
- `src/betting/drawdown_controller.py:1-165` -- DD control, hysteresis, stake adjustment
- `src/backtest/parameter_freeze_protocol.py:1-187` -- parameter freezing, manifest verification

### MEDIUM confidence (domain knowledge, needs empirical validation)
- Ensemble stacking distribution shift: well-known in ML literature (Wolpert 1992, Ting & Witten 1999)
- Conformal prediction exchangeability assumption: the ensemble's error distribution is different but may still satisfy approximate exchangeability
- Optuna TPE overfitting risk with stochastic objectives: general Bayesian optimization theory

---
*Pitfalls research for: Ensemble Filter Recalibration (v1.4)*
*Researched: 2026-05-05*
*Ready for roadmap: yes*
