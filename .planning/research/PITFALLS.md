# Pitfalls Research: Betting Strategy Optimization (v1.3)

**Domain:** Adding Kelly criterion stake sizing, EV-proportional sizing, drawdown control, multi-criteria bet filtering, and backtest parameter optimization to an existing ML horse racing prediction system
**Context:** keiba-ai v1.3 milestone -- existing BacktestEngine + RacePredictor + RegimeDetector + StakeCalculator + DrawdownController are in place. Goal is ROI 91.6% -> 100%+ by optimizing bet selection and stake sizing. The ML model pipeline is frozen; only betting strategy parameters change.
**Researched:** 2026-05-04
**Confidence:** HIGH (cross-validated with full codebase analysis of engine.py, race_predictor.py, stake_calculator.py, drawdown_controller.py, regime_detector.py, win_selection_gate.py, robust_confidence_estimator.py)

---

## Critical Pitfalls

Mistakes that cause fictitious ROI improvement, silent overfitting, or systematic mis-sizing leading to bankroll ruin.

### Pitfall 1: Kelly Overbetting From Overconfident Edge Estimates

**What goes wrong:**
The Kelly formula `f* = edge / (odds - 1)` is only as good as the edge estimate fed into it. ML models systematically produce overconfident probability estimates, especially for win bets where the base rate is low (p_win ~0.05-0.15 for mid-range horses). When the model says p=0.12 but the true probability is p=0.08, Kelly computes f* based on the inflated edge, leading to systematic overbetting across all bets. With 9,074 bets per year, even a small systematic overestimate compounds into massive drawdowns.

In this codebase, `StakeCalculator.calc_stake()` at `stake_calculator.py:59` computes `kelly_fraction = edge / (odds - 1.0)`. The `edge` parameter comes from `bet.edge` which is `win_selection_edge` from `WinSelectionGateModel`. If the gate model's realized ROI training overfits to calibration data, the edge values embed that overfitting.

**Why it happens:**
- ML models (LightGBM/XGBoost/CatBoost) optimize log loss, not calibration. Log loss penalizes confident wrong predictions heavily, but models can still be systematically overconfident in certain probability ranges.
- The WinSelectionGateModel trains on realized ROI with prior-weight smoothing (`prior_weight=24`), but this smoothing may be insufficient for sparse bins (e.g., high-odds/low-prob combinations).
- The edge is computed from EV which is `p_model * odds - 1.0`. A 3% absolute error in p_model at odds 10.0 translates to a 30% error in edge (0.30 vs 0.00).

**How to avoid:**
1. **Use half-Kelly or quarter-Kelly consistently.** The codebase already uses `FRACTIONAL_KELLY = 0.5` (half-Kelly) at `stake_calculator.py:26`. This is correct. Never increase this to full Kelly. The `KELLY_FRACTION_CAP = 0.25` at line 27 provides an additional cap. Keep both.
2. **Apply Conformal confidence filtering BEFORE Kelly sizing.** Use `conformal_confidence_score` from `RobustConfidenceEstimator` to filter out bets where the confidence interval width exceeds a threshold. This removes bets with unreliable edge estimates before Kelly amplifies the sizing error.
3. **Add a minimum edge buffer.** The current `MIN_EDGE_THRESHOLD = 0.005` (0.5%) at `stake_calculator.py:25` is too low for win betting where JRA takeout is ~20-25%. The `GateKeeper.filter_bets()` at `gate_keeper.py:28` uses a 4% edge threshold, which is better. Ensure the Kelly edge input respects this gate.
4. **Verify calibration on OOF data.** Before adjusting Kelly parameters, plot predicted vs actual win rate in decile bins. If the model is overconfident in any bin, reduce Kelly fraction for that bin.

**Warning signs:**
- Backtest shows ROI improvement from Kelly sizing but the improvement comes from a few large bets (concentration risk). Check the distribution of bet sizes: if the top 10 bets by stake account for >30% of total return, Kelly is amplifying noise.
- Max drawdown increases when switching from flat to Kelly sizing. If max_drawdown goes from 15% to 40%, Kelly is overbetting.
- More than 5% of bets have stake = MAX_STAKE (10,000 yen cap at `stake_calculator.py:30`). This means Kelly wants to bet more but hits the cap, indicating systematically high edge estimates.

**Phase to address:**
Phase 1 (stake sizing implementation). Must be addressed before any backtest with Kelly sizing is trusted.

---

### Pitfall 2: Look-Ahead Bias in Backtest Parameter Optimization

**What goes wrong:**
When tuning Conformal alpha thresholds, odds band filters, regime edge thresholds, or Kelly parameters by running backtests and selecting the best-performing parameters, the optimization uses future information. You run backtest with alpha=0.05, alpha=0.10, alpha=0.15, pick the alpha that gives best ROI, and report that ROI. But that ROI embeds knowledge of which alpha would have worked best for the test period -- information you would not have had at the start of the test period.

In this codebase, the `WinSelectionGateModel.train()` at `win_selection_gate.py:804-878` already uses walk-forward OOF folds (`_build_walk_forward_folds`) with `_simulate_threshold_surface()` to find optimal thresholds. However, the regime detector's strategy parameters in `regime_detector.py:178-232` are hardcoded. If v1.3 optimizes these parameters by grid-searching backtest ROI on the test period, it introduces look-ahead bias.

The `BacktestEngine.run()` at `engine.py:364-1026` processes races sequentially and the regime detector updates per-race. But the regime detector was trained on ALL data before the backtest starts (the `train()` call happens in the training pipeline). If the training data includes 2020-2023 and the test is 2024, and regime thresholds are calibrated using 2024 backtest results, that is look-ahead bias.

**Why it happens:**
- Parameter optimization on backtest results is the most natural thing to do -- "which alpha gives the best ROI?" The temptation is strong because the infrastructure makes it easy to iterate.
- Walk-forward validation exists for the ML models but NOT for the betting strategy parameters. The strategy parameters (ev_threshold, edge_threshold, regime multiplier) are set once and applied to the entire test period.
- The existing `ParameterFreezeProtocol` at `parameter_freeze_protocol.py` freezes model parameters, not strategy parameters.

**How to avoid:**
1. **Nested walk-forward for strategy parameters.** Split the test period into two: use the first half for strategy parameter tuning, the second half for validation. Report only the second-half ROI.
2. **Use the existing WF framework.** `BacktestEngine` already supports `--years` with `--train-window` for multi-year backtests. Extend this to also hold out a portion of each test year for strategy parameter validation.
3. **Freeze strategy parameters before OOS evaluation.** Extend `ParameterFreezeProtocol` to also hash and freeze regime detector parameters, edge thresholds, and Kelly fraction settings.
4. **Limit the parameter search space.** If you must optimize, use at most 3-5 parameters with 3 values each (27-243 combinations). Large search spaces guarantee overfitting. The current regime detector has ~10 parameters per regime state -- do NOT optimize all of them.
5. **Apply the multiple testing penalty.** If you test N parameter combinations, the probability of at least one showing spurious ROI > 100% by chance is much higher than the nominal p-value. Use Bonferroni correction or the deflated Sharpe ratio approach.

**Warning signs:**
- Optimized parameters are "ugly" numbers (e.g., edge_threshold=0.047, alpha=0.13) rather than round numbers. This suggests curve-fitting to noise.
- The ROI improvement from optimization is similar in magnitude to the optimization search range. If you search 20 parameter combinations and the best ROI is 101% vs 91.6% baseline, the 9.4pt improvement is within the expected noise range for 20 trials.
- Parameters that work on 2024 test data fail on 2023 test data.

**Phase to address:**
Phase 1 (parameter selection). Must be addressed before any parameter tuning begins. If parameters are hand-picked from domain knowledge rather than optimized, this risk is minimal.

---

### Pitfall 3: Regime Detector Overfitting and State Oscillation

**What goes wrong:**
The `RegimeDetector` at `regime_detector.py:41-239` classifies market state into 3 regimes (AGGRESSIVE, CONSERVATIVE, COLLAPSED) and switches strategy parameters accordingly. Two failure modes:

**(A) Overfitting to regime labels.** The regime training at lines 75-131 uses a LightGBM multiclass model with only 8 features and `num_leaves=7`. The training labels are generated from market indicators: `market_condition_score = favorite_implied_prob * (1 - overround_adj)`. If this score's thresholds are calibrated on the full dataset including the test period, the regime model has seen future information.

**(B) State oscillation.** The hysteresis counter at `regime_detector.py:166` requires 5 consecutive races in the same alternative state before transitioning. In volatile periods, the regime can oscillate between states every 5-10 races. Each transition switches edge thresholds (0.05 vs 0.06 vs 0.09), which changes the bet set dramatically. This creates a hidden parameter that multiplies the effective parameter space by 3x.

In the BacktestEngine loop at `engine.py:682-688`, the regime is re-detected per race using `recent_stats_list[-200:]`. This means the regime can change mid-backtest based on the rolling window of recent races. If a cluster of bad races pushes the regime from CONSERVATIVE to COLLAPSED, bets are cut to near-zero for the next 5+ races, even if the underlying model edge is unchanged.

**Why it happens:**
- Regime detection is conceptually appealing but practically fragile. The 3-state model has low resolution (only 8 features, 7 leaves) and the thresholds are somewhat arbitrary.
- The hysteresis counter of 5 races is very low. In a year with ~5000 races, this allows ~1000 regime transitions per year, each potentially changing bet selection.
- The regime parameters in `get_strategy_params()` are hardcoded and have not been validated on OOS data for the WIN model (they were calibrated for PLACE).

**How to avoid:**
1. **Validate regime Win-specific parameters on OOS data.** The current AGGRESSIVE ev_threshold=1.10, CONSERVATIVE ev_threshold=1.30, COLLAPSED ev_threshold=1.50 were set for PLACE. Win EV distribution is different. Compute the Win EV distribution on OOF data and set thresholds based on percentiles.
2. **Increase hysteresis to 20-50 races.** The current `_transition_hysteresis = 5` at line 68 is too aggressive. Increasing to 20-50 races reduces oscillation and prevents knee-jerk regime switches.
3. **Consider disabling regime switching for v1.3 MVP.** If the regime detector is causing more harm than good, run the entire backtest in CONSERVATIVE mode and use fixed edge thresholds. This removes a source of overfitting and simplifies debugging.
4. **Log regime transitions per race.** The existing `DiagnosticLogger.log_race()` already records regime. After backtest, count transitions. If >50 transitions per year, the detector is oscillating.

**Warning signs:**
- Regime distribution is heavily skewed: >80% of races in one regime state. This means the other states are dead code and the thresholds need recalibration.
- ROI improvement comes primarily from COLLAPSED regime skipping races (reducing bets) rather than AGGRESSIVE regime adding value. This is just bet reduction, not genuine edge improvement.
- Different random seeds for the regime model produce different ROI outcomes. This indicates the regime model is unstable.

**Phase to address:**
Phase 1 (regime parameter calibration). Address before committing to regime-based strategy switching.

---

### Pitfall 4: Odds Band Survivorship Bias

**What goes wrong:**
Odds band analysis filters out bands where historical ROI is negative. For example, if horses with odds 8.0-12.0 have 85% ROI while horses with odds 1.0-3.0 have 105% ROI, you might exclude the 8.0-12.0 band. But this analysis suffers from survivorship bias:

1. **Small sample sizes in extreme bands.** High-odds bands (>30.0) have very few bets. A single big-win outlier can make the band look profitable or unprofitable. The 2024 test has 9,074 bets total; if odds >30.0 has 200 bets, the ROI estimate has a confidence interval of roughly +/-14 percentage points.
2. **Non-stationarity.** The odds band ROI in 2020-2023 may not apply to 2024. If the model improved, its edge may have shifted to different odds ranges.
3. **Selection interaction.** Filtering by odds band changes the bet population, which changes the edge distribution of remaining bets, which may invalidate the original ROI estimate.

In this codebase, the `WinSelectionGateModel` already bucketizes odds via `_quantile_edges()` and `_bucketize()` with `n_bins=6`. The score tables (`combo_scores`, `pair_scores`) use these buckets. But if the training data's ROI per bucket does not generalize to test data, the gate model is filtering on noise.

**Why it happens:**
- Odds band analysis is the most natural post-hoc filtering technique. "Just exclude the bands where we lose money" seems logical.
- The `WinSelectionGateModel` uses Bayesian smoothing (`prior_weight=24`) which helps with small samples, but 24 prior observations may be too few for high-odds buckets with very few training examples.
- The bucketization is quantile-based (equal-frequency bins), not equal-width. This means each bin has roughly the same number of observations, but the odds range varies. A bin might cover odds 1.0-1.5 (heavy favorite) while another covers 10.0-50.0 (longshots). The longshot bin has much higher variance.

**How to avoid:**
1. **Require minimum sample count per band.** Before reporting ROI for an odds band, require at least 200 bets in that band on OOS data. Bands with fewer bets should be marked "insufficient data" rather than "negative ROI."
2. **Use shrinkage toward global mean.** Instead of raw band ROI, compute `shrunken_roi = (band_roi * n_band + global_roi * prior_weight) / (n_band + prior_weight)`. This is exactly what `WinSelectionGateModel` does with `_smoothed_score()`. Ensure the prior weight is at least 50 for odds-band analysis.
3. **Validate on at least 2 separate OOS periods.** If an odds band is negative in 2023 AND 2024, it is more likely a genuine negative edge. If it is negative in 2024 but positive in 2023, it may be noise.
4. **Do not exclude bands; downweight instead.** Rather than binary exclude/include, use a continuous weight based on the band's estimated edge. This avoids the discontinuity at band boundaries.

**Warning signs:**
- Odds band ROI chart shows a sawtooth pattern (positive-negative-positive-negative across adjacent bands). This indicates noise, not signal.
- The excluded bands have <100 bets. The "improvement" from excluding them is within the confidence interval of the ROI estimate.
- ROI "improvement" from odds band filtering is >50% of the total ROI gap (8.4pt). If filtering alone gives you 5pt of the 8.4pt improvement, it is likely overfitting.

**Phase to address:**
Phase 1 (bet filtering). Address when implementing Conformal filter and odds band exclusion.

---

### Pitfall 5: Drawdown Controller Feedback Loop Destabilization

**What goes wrong:**
The `DrawdownController` at `drawdown_controller.py:15-167` adjusts bet size based on rolling ROI and drawdown. The multiplier decreases from 1.0 to as low as 0.05 when DD exceeds 25%. This creates a feedback loop:

1. Bad run -> DD increases -> multiplier decreases -> bet sizes shrink -> wins produce smaller absolute returns -> recovery takes longer -> DD persists -> multiplier stays low.
2. If the DD controller's rolling window (150 bets) is too short for win betting, it may react to normal variance as if it were a genuine edge loss.
3. The SMA + EWMA hybrid at `_calc_rolling_roi()` (lines 132-144) uses `ROLLING_WINDOW=150` and `EWMA_ALPHA=0.1`. For win bets with ~10% hit rate, 150 bets contains only ~15 wins. The ROI estimate is extremely noisy.

The `MAX_ADJUSTMENT_PER_N_BETS=20` and `MAX_ADJUSTMENT_AMOUNT=0.15` at lines 39-40 limit how fast the multiplier can change. This is good. But the underlying issue is that the DD controller was calibrated for PLACE betting (hit rate ~30%) where 150 bets gives ~45 wins and a more stable ROI estimate.

**Why it happens:**
- The DD controller was designed for PLACE betting with 3x higher hit rate. Win betting's lower hit rate means the rolling ROI estimate has 3x higher variance.
- The recovery logic (lines 86-111) transitions REDUCED -> RECOVERING when `roi >= 0.98` and `dd < 0.15`. For win betting, ROI oscillates wildly around 0.90, and the 0.98 threshold may never be reached, trapping the system in REDUCED state permanently.
- The `RECOVERY_INCREMENT = 0.05` per bet (line 33) means recovery from 0.30 multiplier to 1.00 takes 14 bets. With 15 wins in 150 bets, this is plausible for PLACE but marginal for WIN.

**How to avoid:**
1. **Increase rolling window for WIN mode.** Change `ROLLING_WINDOW` from 150 to 400-500 bets when `betting_target="win"`. This gives ~40-50 wins in the window for a more stable ROI estimate.
2. **Lower the recovery ROI threshold for WIN.** `RECOVERY_ROI_THRESHOLD = 0.98` is too strict for win betting where ROI is inherently noisier. Consider 0.92-0.95 for WIN mode.
3. **Add a maximum REDUCED duration.** If the system has been in REDUCED state for >500 bets without recovering, force a gradual recovery regardless of ROI. This prevents permanent low-stake trapping.
4. **Consider disabling DD control for Phase 1 MVP.** Run flat 100-yen bets first, verify the model edge is real, then add DD control. DD control can mask model problems by reducing bet sizes when the model is actually losing.

**Warning signs:**
- The DD controller spends >30% of races in REDUCED state. This means the model is losing more than expected or the thresholds are too aggressive.
- Average stake size is <80 yen (80% of base). The DD controller is frequently active.
- Recovery from a drawdown takes >200 bets. The controller is trapping the system.

**Phase to address:**
Phase 2 (drawdown control tuning). Should be calibrated after basic stake sizing works.

---

### Pitfall 6: Conformal Filter Threshold Creates a False Sense of Precision

**What goes wrong:**
The `RobustConfidenceEstimator` at `robust_confidence_estimator.py:14-252` produces `EV_lower_win_corrected` and `conformal_confidence_score`. The plan is to filter bets where `EV_lower_win_corrected < alpha_threshold`. But:

1. **Conformal prediction assumes exchangeability.** The calibration residuals must be exchangeable (i.i.d.-like). Horse racing residuals are heteroscedastic: high-odds horses have much larger prediction errors than low-odds horses. The race-condition-dependent quantile at lines 71-77 partially addresses this but only for `surface x distance_bin`, not for odds level.
2. **The alpha threshold is a tuning parameter.** Setting alpha=0.1 means "exclude bets where the 90% lower bound of EV is below threshold." If you optimize this threshold on backtest ROI, you are doing exactly the parameter optimization that Pitfall 2 warns about.
3. **The `conformal_confidence_score` at line 216 is a composite metric:** `ev_lower_secondary * (1 - normalized_width)`. This is reasonable but the normalization is within-race (`groupby race_id`). In a race with 3 strong candidates, all get high confidence scores even if they are mediocre in absolute terms.

**Why it happens:**
- Conformal prediction provides valid coverage guarantees under exchangeability, but "valid coverage" does not mean "useful for filtering." The 90% lower bound may be so conservative that it filters out all but the safest bets, or so permissive that it passes everything.
- The confidence score is relative within a race. A horse with confidence_score=0.5 in a weak race looks the same as a horse with confidence_score=0.5 in a strong race, but the absolute edge may differ by 2x.

**How to avoid:**
1. **Use conformal filter as a safety net, not a primary selector.** Keep the existing `WinSelectionGateModel` as the primary bet selector. Use conformal confidence only to exclude bets where the confidence interval is so wide that the edge estimate is unreliable (e.g., `EV_lower_win_corrected < 0.5`).
2. **Do not optimize the conformal alpha threshold.** Pick a conservative value (alpha=0.1, i.e., 90% confidence) and stick with it. Treat it as a fixed safety parameter, not a tuning knob.
3. **Add absolute confidence floor.** In addition to the within-race normalized confidence, check that the absolute `EV_lower_win_corrected >= 0.8` (or some minimum) before allowing a bet. This prevents passing bets in weak races where all horses have high relative scores.

**Warning signs:**
- Conformal filter rejects >50% of WinSelectionGateModel-approved bets. The filter is too aggressive and may be throwing away genuine edge.
- Conformal filter rejects <5% of bets. It is not doing anything useful and adds complexity without value.
- The ROI of conformal-filtered bets is only marginally better than unfiltered bets. The filter is not adding discriminative power.

**Phase to address:**
Phase 1 (confidence filter implementation).

---

### Pitfall 7: EV-Proportional Sizing Amplifies Tail Risk

**What goes wrong:**
EV-proportional sizing (stake proportional to EV) sounds logical: bet more when the edge is larger. But for win betting with high-odds horses, a small number of bets with very high EV dominate the bankroll exposure. For example:
- Horse A: EV=1.50, odds=5.0 -> 50% edge, Kelly fraction ~12.5%, stake = 12,500 yen (capped at 10,000)
- Horse B: EV=1.05, odds=2.0 -> 5% edge, Kelly fraction ~5%, stake = 5,000 yen

Horse A gets 2x the stake of Horse B. If Horse A loses (80% probability at odds 5.0), the bankroll drops 10%. If this happens repeatedly in a cluster, the DD controller kicks in and reduces ALL subsequent bets. The system becomes dominated by a few high-EV bets.

In this codebase, `StakeCalculator.calc_stake()` uses edge-based Kelly (`edge / (odds - 1)`), which already scales with EV. The `RACE_EXPOSURE_CAP = 0.02` (2% of bankroll) at `stake_calculator.py:28` and `check_race_exposure()` at lines 79-122 provide protection. But the cap is relative to bankroll, not to the EV distribution.

**Why it happens:**
- EV distribution for win bets is heavy-tailed. Most bets have EV 1.0-1.2, but a few have EV 1.5-3.0. EV-proportional sizing concentrates stake in the tail.
- The 10,000 yen MAX_STAKE cap at line 30 provides some protection, but at a 100,000 yen bankroll, 10,000 yen is already 10% exposure on a single bet -- far above the 2% race cap.

**How to avoid:**
1. **Cap EV-proportional scaling.** Instead of `stake = base * (ev / threshold)`, use `stake = base * min(ev / threshold, max_scale)` where `max_scale = 2.0`. This prevents extreme concentration.
2. **Use log-EV scaling instead of linear.** `stake = base * log(ev) / log(threshold)`. This compresses the tail and prevents a few high-EV bets from dominating.
3. **Enforce per-bet exposure cap independent of race cap.** Add `BET_EXPOSURE_CAP = 0.01` (1% of bankroll per bet) in addition to the existing 2% per-race cap.
4. **Monitor concentration.** After backtest, compute the Herfindahl index of bet stakes. If HHI > 0.01 (equivalent to 100 equal bets), the sizing is too concentrated.

**Warning signs:**
- Top 10 bets by stake account for >20% of total stake. Sizing is too concentrated.
- More than 3 bets hit the MAX_STAKE cap. The scaling function wants to bet more than the cap allows.
- Removing the top-10 bets by stake changes ROI by >5 percentage points. The system is dependent on a few large bets.

**Phase to address:**
Phase 1 (EV-proportional sizing implementation).

---

### Pitfall 8: Silent Interaction Between Multiple Filters

**What goes wrong:**
v1.3 plans to add multiple filters in sequence: Conformal confidence filter -> odds band filter -> regime edge filter -> DD-adjusted sizing. Each filter individually makes sense, but their interaction can create unexpected behavior:

1. **Cascading exclusion.** The Conformal filter removes 30% of bets. The odds band filter removes another 20% of remaining bets. The regime filter removes another 15%. Total removal: 30% + 14% + 10.5% = 54.5%. You end up with far fewer bets than any single filter would suggest, potentially below the minimum needed for statistical significance.
2. **Correlated filters.** If the Conformal confidence score is correlated with odds (it likely is -- high-odds horses have wider confidence intervals), the odds band filter and Conformal filter are removing overlapping bet populations. The "additional" filtering from adding the second filter is much less than expected.
3. **Order dependency.** Applying Conformal filter before regime filter gives different results than the reverse order, because regime parameters (edge_threshold) interact with the filtered population.

In the current codebase, `BacktestEngine.run()` at lines 681-798 applies filters in this order: regime detection -> candidate selection (WinSelectionGate) -> bet generation -> settlement. The new filters would be inserted at various points in this chain.

**Why it happens:**
- Each filter is designed and tested in isolation. "Conformal filter improves ROI by 2pt." "Odds band filter improves ROI by 1.5pt." The naive expectation is 3.5pt combined, but the actual improvement may be 2.5pt due to overlap.
- Filter order matters because each filter changes the population that subsequent filters see.

**How to avoid:**
1. **Test filters in combination, not just individually.** Run backtest with all filters enabled simultaneously and measure the combined effect. Do not assume effects are additive.
2. **Fix filter order and document it.** The recommended order: (1) Regime detection (sets parameters), (2) WinSelectionGate (primary selection), (3) Conformal confidence floor (safety net), (4) Odds band validation (optional post-filter), (5) Kelly sizing, (6) DD adjustment.
3. **Monitor filter cascade metrics.** For each filter, log: number of bets before, number after, and the overlap with the previous filter's exclusion set. This reveals correlation between filters.
4. **Set minimum bet count.** If the combined filters reduce bet count below 1,000 per year, the statistical power to detect genuine edge is very low. Add a guard that warns if bet count drops below this threshold.

**Warning signs:**
- Combined filter improvement < sum of individual improvements. Filters are correlated.
- Bet count drops below 2,000 per year. Too few bets for reliable ROI estimation.
- Adding a new filter changes ROI by <0.5pt. The filter is not adding value.

**Phase to address:**
Phase 1 (filter implementation and integration testing).

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcode regime thresholds | Quick implementation, no tuning infrastructure needed | Thresholds become stale as data distribution shifts; every threshold change requires manual backtest | MVP only. Must add config-driven thresholds before production. |
| Single-pass backtest for filter validation | Fast iteration | No OOS validation of filter parameters; all improvements may be overfit | Never for final reporting. Acceptable for exploratory analysis only. |
| Use PLACE-calibrated DD controller for WIN | No new code needed | DD reaction is too aggressive for WIN's lower hit rate; system may get trapped in REDUCED state | Only if WIN-specific calibration is planned for next phase |
| Optimize all parameters simultaneously | Potentially larger ROI improvement | Massive overfitting risk; no way to attribute improvement to individual parameters | Never. Optimize one parameter at a time, validate each on OOS. |
| Ignore filter interaction effects | Simpler implementation | Combined effect is unpredictable; may over- or under-filter | Acceptable in Phase 1 if combined test is done before Phase 2 |

---

## Integration Gotchas

Common mistakes when connecting new betting strategy components to existing infrastructure.

| Integration Point | Common Mistake | Correct Approach |
|-------------------|----------------|------------------|
| StakeCalculator + DDController | DD adjusts stake AFTER Kelly cap, making effective cap unpredictable | Apply DD multiplier before rounding, then re-apply 100-yen floor and MAX_STAKE cap |
| RegimeDetector + WinSelectionGate | Regime sets edge_threshold but gate model has its own min_edge; these may conflict | Regime should set high-level parameters (bet/not-bet), gate model should set fine-grained selection. Do not double-filter. |
| ConformalEstimator + BacktestEngine | Conformal is calibrated on OOF data from training pipeline, but backtest uses different feature computation path | Verify conformal calibration is loaded from the same model checkpoint used in training. The `predict_interval()` call in `race_predictor.py:150` must use the calibrated quantiles. |
| Odds band filter + WinSelectionGate | Gate model already encodes odds band information via `_odds_bin`. Adding a separate odds band filter double-counts odds information. | If WinSelectionGate is used, odds band filter should be redundant. Only add explicit odds band filter if gate model is disabled. |
| Kelly sizing + race exposure cap | `check_race_exposure()` scales down all bets proportionally when cap is exceeded, but does not re-check minimum stake. Bets with stake < 100 after scaling are silently dropped. | After `check_race_exposure()`, filter out bets with `stake < 100` and log the count of dropped bets. Current code at `engine.py:227` does this correctly. |
| DDController.update() + sequential bankroll | DD is updated per-bet, not per-race. A single race with 2 bets triggers 2 DD updates. | Consider updating DD once per race (aggregate result) rather than per-bet. This reduces DD noise from correlated within-race bets. |

---

## Performance Traps

Patterns that work at the current scale but would fail with different parameter choices.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-race regime detection in backtest loop | Backtest takes >2x longer when regime detection is enabled | Cache regime state and only re-detect every N races instead of every race | With N=5000 races and LightGBM inference per race, regime detection adds ~1-2 minutes. Acceptable. Breaks if model is heavier. |
| Grid search over strategy parameters | Parameter optimization takes hours for large grids | Limit grid to <=3 parameters with <=5 values each; use randomized search for larger spaces | With 5 params x 5 values = 3125 combinations x 57 min/backtest = 124 days. Never grid-search more than 3 params. |
| DDController SMA+EWMA hybrid recalculated per bet | Each `get_multiplier()` call recalculates SMA over 150-element list | Use deque with rolling sum instead of `np.mean(list)` per call | At 9074 bets, this is 9074 * 150 = 1.36M operations. Marginal but noticeable. Breaks at 100K+ bets. |
| DiagnosticLogger writing per-horse CSV | CSV grows linearly with bet count and feature count | Make diagnostics optional; only log for bet-relevant columns in production mode | At 9074 bets x 14 horses x 120 features = 15M cells. ~50MB CSV. Acceptable now, breaks at 50K+ bets. |

---

## "Looks Done But Isn't" Checklist

Things that appear complete when adding betting strategy optimization but are missing critical pieces.

- [ ] **Kelly criterion:** Often missing edge estimate validation -- verify that `edge` used in Kelly matches OOF-realized edge, not just in-sample edge. Check: compute realized edge per decile of predicted edge on OOF data.
- [ ] **EV-proportional sizing:** Often missing concentration risk check -- verify that no single bet accounts for >5% of total annual exposure. Check: compute top-10 bet exposure share after sizing.
- [ ] **Conformal filter:** Often missing calibration verification -- verify that `EV_lower_win_corrected` is actually a valid lower bound on OOS data (at least 90% of actual EVs should be above the lower bound). Check: compute coverage rate on OOS data.
- [ ] **Odds band exclusion:** Often missing minimum sample check -- verify excluded bands have >200 bets on OOS data. Check: log bet count per band after filtering.
- [ ] **Regime parameter update:** Often missing OOS validation -- verify regime parameters (ev_threshold, edge_threshold) are set on training data, not optimized on test data. Check: verify parameter_freeze_protocol covers regime parameters.
- [ ] **DD controller WIN calibration:** Often missing hit-rate adjustment -- verify ROLLING_WINDOW and recovery thresholds are calibrated for ~10% hit rate (WIN) not ~30% (PLACE). Check: run DD controller in isolation and verify multiplier distribution.
- [ ] **Filter cascade testing:** Often missing combined test -- verify all filters together on a held-out year that was NOT used for individual filter development. Check: reserve 2023 as "validation" year, develop filters on 2024, report 2023 results.

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Kelly overbetting (bankroll loss) | MEDIUM | (1) Revert to flat 100-yen bets, (2) Reduce FRACTIONAL_KELLY to 0.25 (quarter-Kelly), (3) Recalibrate edge estimates on OOF data |
| Look-ahead bias in parameters | LOW | (1) Identify which parameters were optimized on test data, (2) Revert to pre-optimization values or use separate validation period, (3) Re-run backtest with corrected parameters |
| Regime overfitting | LOW | (1) Disable regime switching (force CONSERVATIVE), (2) Re-run backtest, (3) If ROI improves, regime was harmful and can be removed |
| Odds band survivorship bias | LOW | (1) Remove odds band filter, (2) Re-run backtest, (3) If ROI drops significantly, the model edge genuinely varies by odds band -- but verify on 2+ OOS periods |
| DD controller trapping | LOW | (1) Increase ROLLING_WINDOW to 500, (2) Lower RECOVERY_ROI_THRESHOLD to 0.90, (3) If still trapped, disable DD control and use fixed 100-yen flat bets |
| Filter interaction surprise | LOW | (1) Remove the newest filter, (2) Re-run backtest to isolate the problematic interaction, (3) Fix filter order or remove redundant filter |
| Concentration from EV-proportional sizing | MEDIUM | (1) Cap per-bet exposure at 1% of bankroll, (2) Use log-EV scaling instead of linear, (3) Verify Herfindahl index of bet stakes |

---

## Pitfall-to-Phase Mapping

How v1.3 phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Kelly overbetting | Phase 1 (sizing) | Verify OOF realized edge vs predicted edge per decile. Confirm half-Kelly is used. |
| Look-ahead bias | Phase 1 (parameter selection) | Use held-out 2023 as validation. Freeze strategy parameters before 2024 test. |
| Regime overfitting | Phase 1 (regime calibration) | Count regime transitions on test period. Verify <50 per year. Test CONSERVATIVE-only as baseline. |
| Odds band survivorship | Phase 1 (filter implementation) | Require >200 bets per band on OOS. Use shrinkage toward global mean. |
| DD controller WIN mismatch | Phase 2 (DD tuning) | Verify ROLLING_WINDOW >= 400 for WIN. Check multiplier distribution: >70% of races at 1.0 is healthy. |
| Conformal filter precision | Phase 1 (confidence filter) | Verify 90% coverage rate on OOS data. Confirm filter passes >60% of WinSelectionGate bets. |
| EV-proportional tail risk | Phase 1 (sizing) | Compute top-10 exposure share. Verify <15% of total stake. |
| Filter interaction | Phase 2 (integration testing) | Run combined filter backtest on validation year. Compare combined improvement vs sum of individual improvements. |

---

## Sources

### HIGH confidence (codebase analysis + directly observable patterns + established domain knowledge)
- `src/betting/stake_calculator.py:1-123` -- Kelly sizing implementation with half-Kelly, cap, and race exposure
- `src/betting/drawdown_controller.py:1-181` -- DD controller with SMA+EWMA, hysteresis, per-N-bets rate limit
- `src/models/regime_detector.py:1-240` -- 3-state regime with hysteresis counter, hardcoded strategy parameters
- `src/backtest/engine.py:1-1175` -- backtest loop with per-race regime update, bet settlement, DD feedback
- `src/backtest/race_predictor.py:1-900` -- candidate selection, bet generation, regime parameter application
- `src/models/win_selection_gate.py:1-1113` -- WF OOF gate training with threshold surface, score tables
- `src/models/robust_confidence_estimator.py:1-253` -- conformal prediction with race-condition quantiles
- `src/betting/gate_keeper.py:1-42` -- edge-based final filter
- `src/backtest/parameter_freeze_protocol.py:1-101` -- model parameter freezing (does NOT cover strategy params)

### MEDIUM confidence (domain knowledge from web sources, needs empirical validation)
- Kelly overbetting from overconfident ML edge estimates: [Common Mistakes with Kelly](https://kellycriterion.co.uk/sport-betting-guides/common-mistakes-to-avoid-when-using-the-kelly-criterion-in-sports-betting/), [Analytics.Bet on Kelly](https://analytics.bet/articles/reasons-to-ignore-the-kelly-criterion/), [Kelly Criterion Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)
- Drawdown control feedback loops: [On Kelly Betting Limitations (arXiv)](https://arxiv.org/pdf/1710.01787), [Risk-Constrained Kelly (Stanford)](https://web.stanford.edu/~boyd/papers/pdf/kelly.pdf)
- Look-ahead bias in backtest optimization: [Backtesting Pitfalls (Quant Guild)](https://www.linkedin.com/posts/quant-guild_3-backtesting-pitfalls-that-ruin-your-trading-activity-7439369906346737664-qzwO), [Look-Ahead Bias Detection (Medium)](https://mikeharrisny.medium.com/look-ahead-bias-in-backtests-and-how-to-detect-it-ad5e42d97879)
- Fractional Kelly practical guidance: [Why Fractional Kelly (simulations)](https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html), [Good and Bad Properties of Kelly (Berkeley)](https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf)
- Regime detection overfitting: [Are most quant strategies overfit regime bets? (Reddit)](https://www.reddit.com/r/algotrading/comments/1r96n10/are_most_retail_quant_strategies_just_overfit/)

### LOW confidence (extrapolated from related domains, flag for validation)
- Odds band survivorship bias in horse racing: General statistical principle applied to this specific domain. The [Bookie Bashing odds band analysis](https://www.bookiebashing.net/2021/10/17/roi-of-bb-horse-racing-by-odds-bands-and-sp-2/) provides real-world data but the survivorship bias concern is inferred.
- DD controller WIN-specific calibration: The analysis of 150-bet window being too short for WIN is based on statistical reasoning (10% hit rate -> 15 wins in 150 bets -> high variance). Needs empirical validation with actual WIN backtest data.

---
*Pitfalls research for: Betting Strategy Optimization (v1.3)*
*Researched: 2026-05-04*
*Ready for roadmap: yes*
