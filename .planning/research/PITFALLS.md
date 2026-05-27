# Domain Pitfalls

**Domain:** Horse racing prediction -- MarketAware Calibration + Race-Level Ranker (v2.1)
**Researched:** 2026-05-27
**Context:** v2.1 milestone adding MarketAwareWinCalibrator, segment conditioning, Race-Level Ranker, and shadow comparison. Built on existing system at 87.8% BT ROI (v2.0), targeting 100%+. The prior v1.7 ROI of 97.8% degraded to 87.8% through v1.8 Phase 36 feature integration issues, which this milestone must reverse without repeating the same patterns.

## Critical Pitfalls

Mistakes that cause rewrites or major issues. Based on both external research and direct project history (v1.8 ROI collapse from 97.8% to 87.8%).

### Pitfall 1: Strong Feature Uniform Registration Collapses Specialized Models (PROVEN -- v1.8)

**What goes wrong:** Adding strong new features (Phase 36 closing_speed_ratio, form_trend, etc.) and registering them uniformly across ALL models (MarketModel, RaceQualityScreener, AbilityModel, WinTwoStage) causes specialized models to lose their function. The MarketModel becomes dominated by the new features and stops providing independent market-prediction signal. The RaceQualityScreener's quality detection becomes noisy. The whole pipeline degenerates.

**Why it happens:** When new features have high predictive power, they overwhelm the existing feature balance in models that were designed to work with a specific feature profile. The MarketModel's job is to predict market probability from fundamentals; adding strong race-level features makes it overfit to those features and underfit to the market-prediction task. This exact scenario happened in Phase 36 of v1.8.

**Consequences:** ROI drops 10+ percentage points (97.8% to 87.8% in v1.8). The effect is silent during development because individual model metrics may look fine -- the problem is systemic degradation across the pipeline. Debugging requires v[N-1] vs v[N] diff analysis scripts to isolate.

**Prevention:**
- For every new feature added to the MarketAwareWinCalibrator, audit which models it is registered in. The surgical routing approach (Phase 36.1.1) must be applied: exclude new features from MarketModel and RaceQualityScreener unless explicitly validated.
- Maintain a feature routing manifest that maps features to models. The existing 12-model SHA256 manifest tracks which features each model sees.
- Before BT validation, verify that MarketModel's feature importances have not shifted dramatically from the v1.7 baseline.

**Detection:** Compare GPD (Gain per Depth) diagnostics before and after feature changes. If MarketModel's top-3 gain features are all new features, the model has been captured. Run a quick correlation check: if new features have >0.3 correlation with MarketModel's output, exclude them from MarketModel.

---

### Pitfall 2: Double-Correction from Cascading Calibration Layers (PROVEN -- existing codebase risk)

**What goes wrong:** The existing pipeline already has multiple calibration points: EVCorrectionModel (P-correction + E-correction), BenterCombination (logit blend), WinBenterGate (calibration + temp scaling + race normalization), WinSegmentCalibrator (segment-based shrinkage), and WinSelectionPolicy (score-based selection). Adding MarketAwareWinCalibrator as yet another layer creates a chain where each layer assumes its input is the raw model output, but each input is already calibrated by the previous layer. The compound effect is over-shrunk probabilities that never trigger bets.

**Why it happens:** Each calibration layer is designed and tested in isolation. The BenterCombination blends model probability with market probability. The segment calibrator shrinks overconfident segments. The new MarketAwareWinCalibrator calibrates again. None communicate about what the previous layer already did. The existing codebase already shows this symptom: `BenterCombination.fit()` bounds beta at [0.20, 5.0] to prevent fundamental overconfidence, which is itself a correction on top of EVCorrection.

**Consequences:** Final probabilities systematically biased toward the prior (market). Edge estimates compressed toward zero. Bet count drops below viable levels. ROI may improve slightly (fewer bad bets) but total return drops. The system becomes inert. The v2.0 codebase already has `apply_ev_factor=False` by default in WinSegmentCalibrator to prevent this exact compounding.

**Prevention:**
- Define a single canonical calibration ordering. Recommended: raw model -> P/E correction (existing EVCorrectionModel) -> InvestmentFeatureFrame assembly -> MarketAwareWinCalibrator (new, replaces both WinBenterGate and WinSegmentCalibrator for win probability) -> race normalization -> edge calculation.
- The MarketAwareWinCalibrator MUST absorb the roles of both WinBenterGate and WinSegmentCalibrator. Do not run them in sequence with the new calibrator.
- Add a "calibration budget" test: after the full pipeline, check that the variance of `p_win_final` across horses in a race is not compressed below 1.5x the variance of raw `1/tanodds`. If variance is too low, the pipeline is over-correcting.
- When the new calibrator is active, the old WinBenterGate and WinSegmentCalibrator must be explicitly bypassed (not just given default passthrough parameters).

**Detection:** Compare variance of `p_win_final` vs `1/tanodds` across races. Track bet count per race -- if it drops below 0.5 bets/race from the current ~1-2, over-correction is the cause. Check the calibration budget metric (variance ratio) in CI.

---

### Pitfall 3: Race Normalization Destroys Calibrated Probabilities

**What goes wrong:** After Benter blending and calibration produce well-calibrated probabilities, race normalization (`p_win_final = p_combined / sum(race)`) recalibrates them relative to the race sum. If any single horse's probability is badly off (e.g., a scratched horse with stale odds still contributing 0.15), every other horse's probability shifts. The calibration is destroyed. The existing `WinBenterGate.apply()` normalizes after calibration (line 120), which is the vulnerable ordering.

**Why it happens:** The pipeline treats normalization as a harmless post-processing step. But normalization is a transformation that decouples output probabilities from the calibration mapping. Combined with a Benter blend where `beta` (market weight) is large, the market odds of one bad entry corrupt all others.

**Consequences:** Calibrated probabilities become uncalibrated. Brier score improvements from blending vanish after normalization. EV estimates built on `p_win_final` are systematically wrong. The team sees calibration metrics improve on intermediate columns but ROI does not track.

**Prevention:**
- For MarketAwareWinCalibrator, calibrate on race-normalized probabilities directly. The calibration mapping must account for the normalization step. This means the calibrator's training data should be normalized per-race before fitting.
- Alternatively, normalize before calibration, not after. Then the calibrator learns to correct already-normalized probabilities.
- Validate: compute ECE both before and after normalization. If ECE degrades by more than 10% after normalization, the ordering is wrong.
- The new calibrator must produce `p_win_final` that sums to 1.0 per race natively, without a separate normalization post-step.

**Detection:** ECE improves on `p_win_combined` but regresses on `p_win_final`. Calibration reliability diagram shows good calibration pre-normalization, poor calibration post-normalization.

---

### Pitfall 4: Segment Conditioning Overfits to Historical Odds Bands

**What goes wrong:** The existing WinSegmentCalibrator bins odds into bands like [1-2), [2-5), [5-10), [10-30), [30-100), [100+). A horse at odds 4.99 gets segment factor X, while a horse at 5.01 gets segment factor Y. If X and Y differ significantly (e.g., X=0.90, Y=0.95), a tiny odds movement causes a 5% probability shift. This creates discontinuous, unstable selection behavior. More fundamentally, segment corrections trained on 2020-2024 data may invert in 2025 -- a segment that was overconfident historically becomes underconfident and the "correction" makes things worse.

**Why it happens:** Hard bin boundaries create artificial discontinuities. Betting market efficiency changes year over year. A segment that was mispriced in 2022 may be correctly priced in 2025 as the market adapts. Segment corrections based on historical residuals do not generalize. The existing codebase uses `prior_strength=500` for Bayesian shrinkage, which mitigates but does not eliminate this risk.

**Consequences:** Marginal bets at segment boundaries are unstable -- they may or may not trigger depending on minute odds fluctuations. In backtest this manifests as high variance in bet selection across similar races. In production, the same race can produce different bets depending on when the odds snapshot is taken.

**Prevention:**
- Treat odds as a continuous feature in the MarketAwareWinCalibrator rather than binning it. The calibrator should learn smooth corrections, not step functions.
- Use the existing WinSegmentCalibrator's output as a feature (the p_factor and ev_factor) rather than as a standalone correction layer. This is the already-decided "Option B" integration approach.
- Validate segment corrections on a held-out year. If any segment's correction flips sign between the last two training years, increase the prior strength for that segment.
- If bins must be used, validate that adjacent segments have factors within 0.03 of each other.

**Detection:** In backtest, flag races where the selected horse changes when odds shift by less than 0.1. If more than 5% of races show this instability, the segment conditioning is too granular. Track year-over-year sign stability of segment corrections.

---

### Pitfall 5: Shadow Comparison Metric Mismatch Hides Degradation

**What goes wrong:** The shadow comparison framework runs the new calibrator alongside the baseline. If the comparison metrics focus on calibration quality (Brier, ECE) but the new calibrator changes WHICH horses get selected (different top-1 horse in 15-20% of races), the calibration metrics look fine while the ROI degrades. A calibrator that produces slightly better-calibrated probabilities but systematically misses the best value horses is worse than the baseline.

**Why it happens:** Calibration metrics measure probability accuracy, not selection quality. A perfectly calibrated model that always picks the second-best horse loses money. The deployment gate already specifies probability quality as the criterion (correct decision), but the shadow comparison must also track selection agreement and CLV (Closing Line Value). If selection agreement is below 85%, the new calibrator is making materially different bets even if its probabilities are better-calibrated.

**Consequences:** The new calibrator passes the deployment gate (better Brier/ECE) but ROI drops. The team cannot diagnose why because the metrics they checked all improved. This is the exact pattern that makes shadow testing unreliable when the proxy metric (calibration) does not fully capture the business metric (ROI).

**Prevention:**
- The shadow comparison MUST include these metrics beyond calibration: (1) top-1 selection agreement rate (same horse chosen), (2) CLV comparison (average closing-line value of selected horses), (3) ROI delta, (4) hit rate delta, (5) bet count ratio. All five must be reported.
- The deployment gate must require BOTH probability quality improvement AND selection agreement >= 85%. If the new calibrator changes the selected horse in >15% of races, it needs explicit human review before deployment.
- Run shadow comparison on BOTH 2024 and 2025 data. If the new calibrator helps 2024 but hurts 2025, it is overfitted to the training period.
- Track "regret" metrics: how often would the baseline have been correct where the shadow model was wrong, and vice versa.

**Detection:** Shadow shows Brier improvement but ROI regression. Selection agreement below 85%. CLV of shadow selections worse than baseline. These three together indicate metric mismatch.

---

### Pitfall 6: LightGBM Ranker Group Parameter Misalignment

**What goes wrong:** The `group` parameter in LightGBM's lambdarank objective must exactly match race boundaries. If the data is not sorted by `race_id` with contiguous groups, or if the group sizes do not match the actual number of horses per race, the ranker learns from wrong query boundaries. Horses from different races get compared against each other during lambda construction.

**Why it happens:** The training pipeline filters rows (removing horses with missing features, filtering by surface, excluding steeplechase entries) after computing group sizes. The group array no longer matches the filtered data. This is especially likely when InvestmentFeatureFrame produces NaN for some horses (first-time starters with no historical features) and those rows get dropped.

**Consequences:** The ranker produces garbage rankings. It may appear to work on average metrics but fails at the race level. ROI is unpredictable. Debugging is extremely difficult because LightGBM trains without error -- it does not validate group boundaries.

**Prevention:**
- Compute group sizes AFTER all filtering, immediately before passing to LightGBM.
- Add a validation assertion: `sum(groups) == len(X_train)`. This must be checked every training run.
- Sort by `race_id` before computing groups. Never assume data is already sorted.
- Log race_id boundaries and group sizes during training for post-hoc validation.
- When NaN-filled rows are kept instead of dropped, group computation is simpler and LightGBM handles NaN natively.

**Detection:** Random spot-check: pick 5 races, verify group boundaries align with race_id transitions. If average group size differs from expected field size (~12-18 horses), groups are wrong. Log per-race NDCG -- if many races show NDCG=1.0, the ranker is not discriminating.

---

### Pitfall 7: Post-Race Data Leakage Through Feature Pipeline

**What goes wrong:** InvestmentFeatureFrame includes features derived from data only available after the race. The most dangerous sources: `confirmed_odds` (final odds, post-race), `kakuteijyuni` (finishing position), `harontime` (sectional times), `time` (finish time). Even indirect leakage through features like "jockey win rate at this meeting" computed using today's results is fatal.

**Why it happens:** During feature engineering, it is natural to use all available columns. The existing codebase already has `confirmed_odds` vs `tanodds` confusion -- `WinSegmentCalibrator.train()` lines 153-158 explicitly falls back from `confirmed_odds` to `tanodds`. The 3-layer CI leak detection (v1.6) catches direct leakage but not indirect leakage through derived features.

**Consequences:** Backtest shows inflated ROI that collapses in production. This is the single most common cause of betting system failures that look good in testing but fail in deployment. The v1.6 POST_RACE whitelist prevents direct column usage but does not trace feature lineage.

**Prevention:**
- For every new feature added to MarketAwareWinCalibrator or Race-Level Ranker, trace data lineage. What column(s) is it derived from? When is that column populated?
- The existing leakage_validators.py framework must be extended with an explicit `PREDICTION_TIME_COLUMNS` allowlist.
- `tanodds` (morning/early odds) is acceptable; `confirmed_odds` is not. Document the assumed data availability timestamp for each feature.
- Late odds features (`tanoddslow`, `tanoddshigh`) are edge cases -- available during late betting but may not be available at prediction time. Document assumptions.

**Detection:** Train with and without candidate features. If removing a feature causes ROI to drop by more than 30%, it may be leaking. Feature importance >0.15 for odds-derived data is suspicious. Cross-validate with time-shifted features (shift odds by 1 race) -- if performance holds, the feature is genuine.

---

## Moderate Pitfalls

### Pitfall 8: Betting System Label Sparsity in Ranker Training

**What goes wrong:** Only one horse per race wins (label=1 for exactly one entry). In an 18-horse field, the positive class ratio is ~5.5%. This extreme sparsity makes lambdarank training unstable -- the gradient signal is dominated by negative pairs. The ranker learns to predict "nobody wins" and converges to trivial scores where all horses rank similarly.

**Why it happens:** Binary win/loss labels provide minimal gradient signal per race. Lambdarank constructs pairwise gradients, but with only 1 positive per ~15-18 horses, most pairs have identical labels and contribute zero gradient.

**Prevention:**
- Use graded relevance labels instead of binary win/loss. Map finishing position: 1st=5, 2nd=4, 3rd=3, 4th-5th=2, 6th-10th=1, rest=0.
- Set `label_gain` in LightGBM lambdarank to match the relevance mapping.
- Use `lambdarank_truncation_level=3` or `5` to focus learning on top positions.
- Ensure at least 10,000 races in training data.

### Pitfall 9: Feature Multicollinearity Between Model and Market Probabilities

**What goes wrong:** InvestmentFeatureFrame includes both model-derived probabilities (`p_win_corrected`, `p_win_combined`) and market-derived probabilities (`1/tanodds`, overround). These are highly correlated (typically rho > 0.8). When fed into the calibrator, it cannot distinguish model information from market information, and learned blend weights become unstable across training runs.

**Why it happens:** Both model and market probabilities estimate the same underlying quantity (true win probability). High correlation is expected but creates ill-conditioned feature matrices.

**Prevention:**
- Construct features that capture the DIFFERENCE: `logit(p_model) - logit(p_market)`, `p_model / p_market`, `rank(p_model) - rank(p_market)`.
- Use raw probabilities as inputs only, not as calibrator features.
- Check VIF (Variance Inflation Factor). Flag any feature with VIF > 10.

### Pitfall 10: Regime-Dependent Calibration Instability

**What goes wrong:** The calibrator is trained across all market regimes (aggressive/conservative/collapsed). In collapsed markets (low liquidity, high overround), the market is noisy and the model should be weighted more. In aggressive markets, the market is efficient and should get more weight. A single set of blend weights cannot handle both.

**Why it happens:** Each regime has different market efficiency. A calibrator trained on the full period learns average blend weights suboptimal for each regime. The existing RegimeDetector uses overround and favorite rate for classification.

**Prevention:**
- Include regime indicators as continuous features in the calibrator. Do not create separate regime-specific calibrators (per project decision: regime-independent structure).
- Validate calibration quality per regime. If ECE in collapsed regime is >2x ECE in aggressive regime, add regime features.

### Pitfall 11: OOF Health False Positives Blocking Valid Models

**What goes wrong:** OOF health checks include anomaly thresholds like "top1 hit rate >35% warning." In a small validation set or an unusual period (many favorites winning), a legitimate model triggers these thresholds. The pipeline refuses to deploy a valid model.

**Why it happens:** Fixed thresholds do not account for sample size or distribution variation. A 35% top-1 hit rate is suspicious for 1000 races but normal for 50 races.

**Prevention:**
- Set thresholds based on statistical significance, not fixed values. Use confidence intervals.
- Distinguish "stop" conditions (data corruption) from "warning" conditions (unusual but possible).
- Never block deployment on a warning; only block on a stop.

### Pitfall 12: Market Model Rule 11 Violation in Calibrator Features

**What goes wrong:** The existing `market_model.py` has a critical constraint (Rule 11): only `log_error` is passed to Stage2, never `p_market_pred` directly. If MarketAwareWinCalibrator accidentally uses `p_market_pred` or `p_market_win_adj` as a feature, it violates this isolation and creates feedback loop leakage.

**Why it happens:** InvestmentFeatureFrame assembles features from multiple sources. If market model outputs are included alongside raw market features, the calibrator may use both. The market model's prediction is already fitted to historical data and carries overfitting risk.

**Prevention:**
- InvestmentFeatureFrame must use raw market features (tanodds, implied probability, overround) but NOT market model outputs.
- Add CI check: verify no calibrator feature column starts with `p_market`.

### Pitfall 13: Shadow Comparison Insufficient Temporal Coverage

**What goes wrong:** Shadow comparison runs on 2024 only (or 2025 only). The new calibrator happens to work well on that year's market structure but fails on the other year. The deployment gate passes, but the calibrator is overfitted to the comparison period.

**Why it happens:** Market efficiency varies across years. JRA betting pools, favorite behavior, and field composition change. A single-year shadow is a single sample from the distribution of possible market conditions.

**Prevention:**
- Shadow comparison MUST run on both 2024 AND 2025 independently.
- If the new calibrator passes on 2024 but fails on 2025 (or vice versa), it requires investigation before deployment.
- Report per-year metrics separately, not as an aggregate.

---

## Minor Pitfalls

### Pitfall 14: LightGBM Ranker Position Bias

**What goes wrong:** In lambdarank, the model implicitly learns that position 1 in the input order is more important. If horses are sorted by `umaban` or starting gate position, the ranker learns gate-position bias rather than true ability.

**Prevention:** Randomize horse order within each race before training. Or enable `lambdarank_position_bias_regularization` (LightGBM 4.1.0+).

### Pitfall 15: Temperature Scaling Bounds Too Narrow

**What goes wrong:** The existing `TemperatureScaling` bounds temperature to [0.3, 3.0]. In unusual market conditions, the optimal temperature may be outside this range. The bounded optimization finds a suboptimal solution at the boundary.

**Prevention:** Monitor how often the optimizer hits the bounds. If >20% of runs hit bounds, widen them to [0.1, 5.0].

### Pitfall 16: Race-Level Aggregation Window Mismatch

**What goes wrong:** Race-level features (e.g., "average model probability of top 3 horses") require complete race data. If some horses are filtered out (missing features, surface filter), the aggregation is based on a partial field.

**Prevention:** Compute race-level aggregations before filtering. Use NaN-safe aggregation. Flag races where >30% of entries were excluded.

### Pitfall 17: EV Factor Compounding in Segment Calibration

**What goes wrong:** WinSegmentCalibrator can apply both `p_factor` and `ev_factor`. When both are active, effective edge is `p * odds * p_factor * ev_factor - 1`, compounding two shrinkage factors. Currently `apply_ev_factor=False` by default, but accidental activation is possible.

**Prevention:** The MarketAwareWinCalibrator should apply a single correction factor, not separate p and EV corrections. If EV correction is needed, apply it after the unified probability correction.

### Pitfall 18: Shadow-First Deployment Gate Too Strict

**What goes wrong:** The deployment gate requires probability quality + bet count maintenance + artifact reproducibility + diagnostics to ALL pass. If any single metric regresses by even a tiny amount (e.g., Brier worsens by 0.001), deployment is blocked even if ROI improves by 10pp. The system becomes impossible to improve because every change has tradeoffs.

**Prevention:**
- Define explicit tolerance ranges for each gate criterion. Brier worsening by <0.005 is acceptable if ROI improves by >5pp.
- The gate should allow "conditional deployment" where a regression in one metric is accepted when compensating improvements exist elsewhere.
- Human review should be the escape valve, not an automatic block.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| MarketAwareWinCalibrator design | Double-correction with existing WinBenterGate (Pitfall 2) | Replace, do not append. Explicit bypass of old pipeline. |
| MarketAwareWinCalibrator design | Race normalization destroying calibration (Pitfall 3) | Calibrate on normalized probabilities or normalize before calibration |
| MarketAwareWinCalibrator design | Feature leakage through p_market_pred (Pitfall 12) | Feature allowlist; CI check for p_market columns |
| MarketAwareWinCalibrator features | Multicollinearity (Pitfall 9) | Use difference features; check VIF |
| Segment conditioning integration | Year-dependent segment inversion (Pitfall 4) | Continuous features, strong priors, held-out year validation |
| Segment conditioning integration | Boundary discontinuity (Pitfall 4) | Continuous odds treatment; validate adjacent segment gap <0.03 |
| Segment conditioning integration | EV factor compounding (Pitfall 17) | Single correction factor; no parallel shrinkage |
| Race-Level Ranker training | Group parameter misalignment (Pitfall 6) | Compute groups after filtering; assert sum==len |
| Race-Level Ranker training | Label sparsity (Pitfall 8) | Graded relevance; lambdarank_truncation_level=3 |
| Race-Level Ranker training | Position bias (Pitfall 14) | Randomize horse order within races |
| Shadow comparison | Metric mismatch (Pitfall 5) | Track selection agreement, CLV, ROI -- not just calibration |
| Shadow comparison | Insufficient temporal coverage (Pitfall 13) | Both 2024 AND 2025 required |
| Shadow comparison | Deployment gate too strict (Pitfall 18) | Tolerance ranges; conditional deployment; human escape valve |
| Feature routing | Strong feature uniform registration (Pitfall 1) | Surgical routing; exclude from MarketModel/RaceQuality |

## Cross-Phase Integration Pitfalls

| Concern | Phases Involved | Risk | Mitigation |
|---------|----------------|------|------------|
| Cascading calibration over-correction | Calibrator + Segment | Critical | New calibrator absorbs old roles; old pipeline bypassed |
| Strong feature captures specialized models | Feature + All models | Critical (PROVEN v1.8) | Feature routing manifest; exclude from MarketModel/RaceQuality |
| Feature availability at prediction time | Features + Calibrator + Ranker | Critical | Feature allowlist with lineage tracing; CI validation |
| OOF quality affects all downstream calibration | OOF Health + Calibrator | Critical | OOF health must pass before calibrator training |
| Ranker training depends on calibrator output | Calibrator + Ranker | Moderate | Freeze calibrator before ranker training |
| Segment corrections conflict with ranker | Segment + Ranker | Moderate | Train ranker with segment features already applied |
| Shadow metrics miss selection changes | Shadow + Deployment | High | Track 5+ metrics including selection agreement and CLV |

## Project-Specific Historical Lessons

These pitfalls are drawn from the project's own history (v1.0-v2.0, 10 milestones, 38 phases):

1. **v1.8 Phase 36 (PROVEN):** Strong features (closing_speed_ratio, form_trend) registered in all models caused MarketModel/RaceQuality collapse. Required Phase 36.1.1 surgical routing to fix. ROI dropped from 97.8% to 87.8%. The new MarketAwareWinCalibrator features must NOT be registered in MarketModel or RaceQualityScreener.

2. **v1.6 (PROVEN):** 37 new features yielded only +1.3pp ROI improvement. Feature quantity does not equal quality. The MarketAwareWinCalibrator should use a small number of well-chosen features (logit difference, rank disagreement, regime indicator) rather than a large feature set.

3. **v1.6 (PROVEN):** Training/prediction path dual management caused 6 feature omissions in the inference path. The InvestmentFeatureFrame's dual-mode builder (train/infer same schema) was designed to prevent this. Any new calibrator must use the same dual-mode pattern.

4. **v1.5 (PROVEN):** CQR residual learning change caused overfitting that required a design revision. The new calibrator must be validated with OOF predictions, not in-sample metrics.

5. **v1.4 (PROVEN):** Filter thresholds must match model output distributions. The MarketAwareWinCalibrator will change the probability distribution; all downstream filters (EV_lower, OddsBand) must be recalibrated after the new calibrator is in place.

## Sources

- Codebase analysis: `src/models/win_benter_gate.py` (race normalization ordering, OOF walk-forward splits)
- Codebase analysis: `src/models/win_segment_calibrator.py` (segment calibration with Bayesian shrinkage, p_factor/ev_factor compounding)
- Codebase analysis: `src/models/benter_combination.py` (logit-space blending, beta bounds [0.20, 5.0])
- Codebase analysis: `src/backtest/race_predictor.py` (full inference chain with 14+ model stages)
- Codebase analysis: `src/models/ev_correction_model.py` (P/E decomposition, confirmed_odds usage)
- Codebase analysis: `.planning/RETROSPECTIVE.md` (v1.8 ROI collapse lesson, Phase 36.1.1 surgical routing)
- Codebase analysis: `.planning/codebase/CONCERNS.md` (tight coupling, NaN propagation, calibration gaps)
- LightGBM documentation: lambdarank parameters (group, label_gain, truncation_level, position_bias_regularization)
- Benter (1994): "Computer Based Horse Race Handicapping and Wagering Systems" -- logit-space blending methodology
- [Revisiting the Algorithm that Changed Horse Race Betting](http://actamachina.com/posts/annotated-benter-paper) -- calibration pitfalls in Benter blending
- [Wallaroo AI: A/B Testing and Shadow Deployments](https://wallaroo.ai/ai-production-experiments-the-art-of-a-b-testing-and-shadow-deployments/) -- shadow deployment pitfalls
- [MLOps Community: A/B Testing in ML](https://mlops.community/blog/the-what-why-and-how-of-a-b-testing-in-ml) -- proxy vs business metric misalignment
- [arXiv: Generative Approach to Multi-Competitor Races](https://arxiv.org/html/2310.01748v3) -- race-level ranking pitfalls
- Confidence: HIGH for codebase-derived pitfalls (directly observed in source code and project history), MEDIUM for LightGBM ranker pitfalls (documentation-verified but not yet tested in this specific codebase), MEDIUM for shadow comparison pitfalls (industry-standard patterns applied to this project's context)
