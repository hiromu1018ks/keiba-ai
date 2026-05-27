# Domain Pitfalls

**Domain:** Horse racing prediction -- Investment Pipeline Restructuring (v2.0)
**Researched:** 2026-05-27
**Context:** v2.0 milestone adding 5 components: OOF Health, InvestmentFeatureFrame, MarketAwareWinCalibrator, Segment Calibration integration, Race-Level Ranker. Built on existing system at 87.8% BT ROI (Phase #33), targeting 100%+.

## Critical Pitfalls

Mistakes that cause rewrites or major issues.

### Pitfall 1: Race Normalization Destroys Calibrated Probabilities

**What goes wrong:** After Benter blending and calibration produce well-calibrated probabilities, race normalization (`p_win_final = p_combined / sum(race)`) recalibrates them relative to the race sum. If any single horse's probability is badly off (e.g., a scratched horse with stale odds still contributing 0.15), every other horse's probability shifts. The calibration you carefully built is gone.

**Why it happens:** The pipeline treats normalization as a harmless post-processing step. But normalization is itself a transformation that decouples output probabilities from the calibration mapping. Combined with a Benter blend where `beta` (market weight) is large, the market odds of one bad entry corrupt all others. The existing `WinBenterGate.apply()` normalizes after calibration (line 120), which is the vulnerable ordering.

**Consequences:** Calibrated probabilities become uncalibrated. Brier score improvements from Benter blending vanish after normalization. EV estimates built on `p_win_final` are systematically wrong. The team sees calibration metrics improve but ROI does not, and cannot diagnose why.

**Prevention:**
- Normalize before calibration, not after. For the new MarketAwareWinCalibrator, reverse the order: normalize raw probabilities to race sum, then calibrate the normalized values.
- Alternatively, calibrate on race-normalized probabilities directly so the calibration mapping accounts for the normalization step.
- Validate: compute ECE both before and after normalization. If ECE degrades by more than 10% after normalization, the ordering is wrong.
- The new calibrator must produce `p_win_final` that sums to 1.0 per race. Build normalization into the calibrator itself rather than applying it as a separate step.

**Detection:** ECE improves on `p_win_combined` but regresses on `p_win_final`. Calibration reliability diagram shows good calibration pre-normalization, poor calibration post-normalization. ROI does not track with calibration metric improvements.

---

### Pitfall 2: Post-Race Data Leakage in Feature Pipeline

**What goes wrong:** InvestmentFeatureFrame includes features derived from data that is only available after the race finishes. The most dangerous sources: `confirmed_odds` (final odds, only known post-race), `kakuteijyuni` (finishing position), `harontime` (sectional times), `time` (finish time). Even indirect leakage through features like "jockey win rate at this meeting" computed using today's results is fatal.

**Why it happens:** During feature engineering, it is natural to use all available columns. The existing codebase already has `confirmed_odds` vs `tanodds` confusion -- `WinSegmentCalibrator.train()` line 153-158 explicitly falls back from `confirmed_odds` to `tanodds`, suggesting this has been a problem before. `EVCorrectionModel` uses `confirmed_odds` for training label construction, which is correct for training but becomes leakage if the same column is used as a feature.

**Consequences:** Backtest shows inflated ROI that collapses in production. Model appears to predict outcomes but is actually reading the answer from leaked features. This is the single most common cause of betting system failures that look good in testing but fail in deployment.

**Prevention:**
- Implement a strict feature allowlist. Only columns available at prediction time (before race start) may enter InvestmentFeatureFrame. `tanodds` (morning/early odds) is acceptable; `confirmed_odds` is not.
- Use the existing `leakage_validators.py` framework to validate every new feature. Extend it with an explicit `PREDICTION_TIME_COLUMNS` allowlist.
- For each feature, trace the data lineage: what column(s) is it derived from? When is that column populated? If any source column is populated after the race start bell, the feature is contaminated.
- Late odds features (`tanoddslow`, `tanoddshigh`) are edge cases -- they are available during late betting but may not be available at the moment of prediction in a real-time system. Document the assumed data availability timestamp for each feature.
- CI must fail if any feature column's lineage traces to post-race data.

**Detection:** Train with and without candidate features. If removing a feature causes ROI to drop by more than 30%, it may be leaking. Run feature importance -- any feature with importance >0.15 that derives from odds-related data is suspicious. Cross-validate with intentionally time-shifted features (shift odds by 1 race) -- if performance holds, the feature is likely genuine.

---

### Pitfall 3: LightGBM Ranker Group Parameter Misalignment

**What goes wrong:** The `group` parameter in LightGBM's lambdarank objective must exactly match race boundaries. If the data is not sorted by `race_id` with contiguous groups, or if the group sizes don't match the actual number of horses per race, the ranker learns from wrong query boundaries. Horses from different races get compared against each other during lambda construction.

**Why it happens:** The training pipeline filters rows (e.g., removing horses with missing features, filtering by surface, excluding `haronnashi` entries) after computing group sizes. The group array no longer matches the filtered data. This is especially likely when the InvestmentFeatureFrame produces NaN for some horses (e.g., first-time starters with no historical features), and those rows get dropped.

**Consequences:** The ranker produces garbage rankings. It may appear to work on average metrics but fails at the race level. ROI is unpredictable. Debugging is extremely difficult because the model trains without error -- LightGBM does not validate group boundaries against data content.

**Prevention:**
- Compute group sizes AFTER all filtering, immediately before passing to LightGBM.
- Add a validation assertion: `sum(groups) == len(X_train)`. This must be checked every training run.
- Sort by `race_id` before computing groups. Never assume data is already sorted.
- Log race_id boundaries and group sizes during training for post-hoc validation.
- When NaN-filled rows are kept (instead of dropped), the group computation is simpler but the model must handle NaN input -- LightGBM handles NaN natively, so this is the safer path.

**Detection:** Random spot-check: pick 5 races, verify the group boundaries align with race_id transitions. If average group size differs from expected field size (~12-18 horses), groups are wrong. Log per-race NDCG during validation -- if many races show NDCG=1.0, the ranker is not discriminating (possible sign of group misalignment or trivial labels).

---

### Pitfall 4: Double-Correction from Cascading Calibration Layers

**What goes wrong:** The existing pipeline already has multiple calibration points: `EVCorrectionModel` (P-correction + E-correction), `BenterCombination` (logit blend), `WinBenterGate` (calibration + temp scaling + normalization), `WinSegmentCalibrator` (segment-based shrinkage), and `WinSelectionPolicy` (score-based selection). Adding `MarketAwareWinCalibrator` as a new layer creates a chain where each layer assumes its input is the raw model output, but in reality each input is already calibrated by the previous layer. The compound effect is over-shrunk probabilities that never trigger bets.

**Why it happens:** Each calibration layer is designed and tested in isolation. The BenterCombination blends model probability with market probability. The segment calibrator shrinks overconfident segments. The new MarketAwareWinCalibrator calibrates again. None of these layers communicate about what the previous layer already did.

**Consequences:** Final probabilities are systematically biased toward the prior (market). Edge estimates (`p * odds - 1`) are compressed toward zero. Bet count drops below viable levels. ROI may improve slightly (fewer bad bets) but total return drops (too few bets). The system becomes effectively inert.

**Prevention:**
- Define a clear calibration pipeline with a single canonical ordering. Recommended order for v2.0: raw model -> P/E correction (existing) -> InvestmentFeatureFrame assembly -> MarketAwareWinCalibrator (new, replaces both BenterCombination and segment calibration for win probability) -> race normalization -> edge calculation.
- The MarketAwareWinCalibrator must absorb the roles of both `WinBenterGate` and `WinSegmentCalibrator` for win probability. Do not run them in sequence.
- Add a "calibration budget" test: after the full pipeline, check that the variance of `p_win_final` across all horses in a race is not compressed below 1.5x the variance of raw `1/tanodds`. If variance is too low, the pipeline is over-correcting.
- Explicitly disable or bypass the existing `WinBenterGate` and `WinSegmentCalibrator` when the new calibrator is active. The ROI_IMPROVEMENT_PLAN.md Phase 3 already recommends Option B (integrate WSC as features into MarketAwareWinCalibrator), which is correct.

**Detection:** Compare variance of `p_win_final` vs `1/tanodds` across races. If the ratio is consistently below 1.5, over-correction is happening. Track bet count per race in backtest -- if it drops below 0.5 bets/race (from current ~1-2), the system is too conservative.

---

### Pitfall 5: OOF Contamination Across Temporal Boundaries

**What goes wrong:** The OOF (out-of-fold) prediction generation uses expanding walk-forward splits, but if the data contains races from the same meeting in both train and validation folds, or if feature computation uses data from the validation period (e.g., rolling jockey stats computed across the fold boundary), the OOF predictions are contaminated. This makes all downstream calibration unreliable.

**Why it happens:** `_walk_forward_race_splits()` in `win_benter_gate.py` splits by race index position, not by date boundary. If races are sorted by `race_date` but not by time-of-day within a date, races from the same day can straddle the split point. Additionally, feature computation (e.g., `expanding().mean()` stats) may include data from the validation period if computed before the split. The `KFold(n_splits=5, shuffle=False)` in `market_model.py` preserves time order but does not guarantee date-level separation.

**Consequences:** OOF predictions appear better than they actually are. Benter blend weights are overfitted to contaminated data. Calibration metrics (ECE, Brier) are optimistic. When deployed, real out-of-sample performance is worse than expected.

**Prevention:**
- Split OOF by `race_date` with a 1-day gap between train and validation folds. Never allow same-date races in both folds.
- Compute all features BEFORE splitting, using only expanding windows that strictly exclude the current row's data. The existing `leakage_validators.py` already provides this check -- extend it to OOF generation.
- Add a validation check: for each fold, verify that `max(train_race_date) < min(val_race_date)`.
- Replace `KFold(n_splits=5, shuffle=False)` in market_model.py with explicit date-based splits.

**Detection:** Compare OOF-based metrics with truly held-out test metrics. If OOF Brier score is more than 10% better than test Brier score, contamination is likely. Check fold overlap: `set(train_race_dates) & set(val_race_dates)` must be empty.

---

## Moderate Pitfalls

### Pitfall 6: Betting System Label Sparsity in Ranker Training

**What goes wrong:** Only one horse per race wins (label=1 for exactly one entry). In an 18-horse field, the positive class ratio is ~5.5%. This extreme sparsity makes lambdarank training unstable -- the gradient signal is dominated by the vast majority of negative pairs. The ranker learns to predict "nobody wins" and converges to a trivial solution where all horses get similar scores.

**Why it happens:** Binary win/loss labels provide minimal gradient signal per race. Lambdarank constructs pairwise gradients, but with only 1 positive per ~15-18 horses, most pairs have identical labels and contribute zero gradient.

**Prevention:**
- Use graded relevance labels instead of binary win/loss. Map finishing position to relevance: 1st=5, 2nd=4, 3rd=3, 4th-5th=2, 6th-10th=1, rest=0. This provides much richer gradient signal.
- Set `label_gain` in LightGBM lambdarank to match the relevance mapping.
- Use `lambdarank_truncation_level=3` or `5` to focus learning on the top positions, which are most relevant for betting.
- Ensure at least 10,000 races in the training set to provide sufficient positive examples (~10,000 winning samples).

### Pitfall 7: Segment Boundary Edge Effects in Calibration

**What goes wrong:** The existing `WinSegmentCalibrator` bins odds into bands like [1-2), [2-5), [5-10), etc. A horse at odds 4.99 gets segment factor X, while a horse at odds 5.01 gets segment factor Y. If X and Y differ significantly (e.g., X=0.90, Y=0.95), a tiny odds movement causes a 5% probability shift. This creates discontinuous, unstable selection behavior.

**Why it happens:** Hard bin boundaries create artificial discontinuities. The Benter-style calibration is sensitive to input probability -- a 5% shift in probability at the segment boundary translates to a non-trivial change in edge calculation.

**Prevention:**
- Use overlapping bins with soft boundaries instead of hard cuts. Apply a weighted average of adjacent segment factors for horses near bin edges.
- The MarketAwareWinCalibrator should treat odds as a continuous feature rather than binning it. This is one reason Phase 3 recommends integrating WSC as features rather than using it standalone.
- If bins must be used, validate that adjacent segments have factors within 0.03 of each other. Flag and investigate any segment pair where the gap exceeds this threshold.

### Pitfall 8: Feature Multicollinearity Between Model and Market Probabilities

**What goes wrong:** InvestmentFeatureFrame includes both model-derived probabilities (`p_win_corrected`, `p_win_combined`) and market-derived probabilities (`1/tanodds`, overround). These are highly correlated (typically rho > 0.8). When fed into the MarketAwareWinCalibrator, the model cannot distinguish model information from market information, and the learned blend weights become unstable across training runs.

**Why it happens:** Model probability and market probability are both estimating the same underlying quantity (true win probability). High correlation is expected. But when both appear as raw features, the calibrator sees redundant information and its coefficients become ill-conditioned.

**Prevention:**
- Construct features that capture the DIFFERENCE between model and market, not both raw values. Key features: `logit(p_model) - logit(p_market)` (the "edge" in logit space), `p_model / p_market` (relative confidence), `rank(p_model) - rank(p_market)` (ranking disagreement).
- Use the raw model and market probabilities as inputs only, not as features for the calibrator. The calibrator's job is to produce a calibrated blend; the features should help it decide how to blend, not provide redundant copies of the inputs.
- Check VIF (Variance Inflation Factor) for the feature set. Flag any feature with VIF > 10.

### Pitfall 9: Regime-Dependent Calibration Instability

**What goes wrong:** The calibrator is trained across all market regimes (aggressive, conservative, collapsed per the existing RegimeDetector). In collapsed markets (low liquidity, high overround), the market is noisy and the Benter blend should weight the model more heavily. In aggressive markets, the market is efficient and should get more weight. A single set of blend weights cannot handle both.

**Why it happens:** The RegimeDetector uses market-level features (overround, favorite rate) to classify regimes. Each regime has different market efficiency characteristics. A calibrator trained on the full period learns average blend weights that are suboptimal for each individual regime.

**Prevention:**
- Include regime indicators as features in the MarketAwareWinCalibrator so it can adjust blend behavior by regime.
- The ROI_IMPROVEMENT_PLAN.md already specifies "regime-independent structure" as a key decision, meaning the calibrator should not have separate branches per regime. Instead, regime information should flow as a continuous feature that the calibrator uses internally.
- Validate calibration quality separately per regime. If ECE in collapsed regime is >2x ECE in aggressive regime, the calibrator is regime-blind and needs regime features.

### Pitfall 10: Year-Over-Year Segment Effect Inversion

**What goes wrong:** The existing `WinSegmentCalibrator` shows year-dependent effectiveness -- it helps 2025 turf, is neutral for 2024, and does not address 2025 dirt decline. If the new segment calibration (integrated into MarketAwareWinCalibrator) is trained on 2020-2024 data, the learned segment corrections may invert in 2025. A segment that was overconfident in the training period becomes underconfident in the test period, and the "correction" makes things worse.

**Why it happens:** Betting market efficiency changes year over year. A segment that was mispriced in 2022 may be correctly priced in 2025 as the market adapts. Segment corrections based on historical residuals do not generalize.

**Prevention:**
- Use Bayesian shrinkage with strong priors (the existing `prior_strength=500` is a reasonable starting point). Strong priors prevent segments from deviating too far from the global mean.
- Validate segment corrections on a held-out year. If any segment's correction flips sign between the last two training years, reduce that segment's prior or merge it with an adjacent segment.
- The project constraint "bet count reduction forbidden" provides a natural check: if segment calibration reduces bet count by >10%, the corrections are too aggressive.

### Pitfall 11: OOF Health Check False Positives Blocking Valid Models

**What goes wrong:** The OOF health checks (Phase 0) include anomaly thresholds like "top1 hit rate >35% warning" and "top1 ROI >200% stop." In a small validation set or an unusual period (e.g., many favorites winning), a legitimate model can trigger these thresholds. The pipeline refuses to deploy a model that is actually fine, and the team wastes time investigating false alarms.

**Why it happens:** Fixed thresholds do not account for sample size or distribution variation. A 35% top-1 hit rate is suspicious for 1000 races but normal for 50 races where a dominant favorite era occurred.

**Prevention:**
- Set thresholds based on statistical significance, not fixed values. Use confidence intervals: trigger a warning only if the metric is outside the 95th percentile of the expected distribution given the number of races.
- Distinguish between "stop" conditions (genuine data corruption, like empty OOF arrays) and "warning" conditions (unusual but possible performance). Never block deployment on a warning; only block on a stop.
- Log the exact threshold and the observed value when a check triggers, so the team can assess whether the trigger is reasonable.
- The "OOF rows <70% expected" threshold is good (catches fold generation failures). The "folds <3 stop" threshold is good (catches insufficient data). The ROI-based thresholds need more nuance.

### Pitfall 12: Market Model Rule 11 Violation in Calibrator Features

**What goes wrong:** The existing `market_model.py` has a critical constraint (Rule 11): only `log_error` (signed/absolute) and rank are passed to Stage2, never `p_market_pred` directly. If the new MarketAwareWinCalibrator accidentally uses `p_market_pred` or `p_market_win_adj` as a feature, it violates this isolation principle and creates a different kind of leakage -- the model's own market prediction feeds back into the calibration.

**Why it happens:** The InvestmentFeatureFrame assembles features from multiple sources. If market model outputs are included alongside raw market features (tanodds-derived), the calibrator may use both. The market model's prediction is already fitted to historical data and carries overfitting risk.

**Prevention:**
- InvestmentFeatureFrame must use raw market features (tanodds, implied probability, overround) but NOT market model outputs (`p_market_pred`, `p_market_win_adj`).
- Document which columns are "raw market" vs "market model output" in the feature catalog.
- Add a CI check: verify that no MarketAwareWinCalibrator feature column starts with `p_market`.

---

## Minor Pitfalls

### Pitfall 13: LightGBM Ranker Position Bias

**What goes wrong:** In lambdarank, the model implicitly learns that position 1 in the input order is more important. If horses are sorted by `umaban` (horse number) or starting gate position, the ranker learns gate-position bias rather than true ability.

**Why it happens:** LambdaRank constructs pairwise gradients between positions. Without position bias regularization, the model can learn spurious correlations between input order and relevance.

**Prevention:** Randomize horse order within each race before training. Or enable `lambdarank_position_bias_regularization` (LightGBM 4.1.0+) to explicitly model and reduce position bias.

### Pitfall 14: Temperature Scaling Bounds Constriction

**What goes wrong:** The existing `TemperatureScaling` bounds temperature to [0.3, 3.0]. In unusual market conditions (e.g., a strong model in an inefficient market), the optimal temperature might be outside this range. The bounded optimization finds a suboptimal solution at the boundary.

**Why it happens:** The bounds were set conservatively to prevent extreme temperature values that would collapse all probabilities to 0.5 or spread them to extremes. But the bounds may be too narrow for the new calibrator's input distribution.

**Prevention:** Monitor how often the optimizer hits the bounds. If it hits bounds in >20% of training runs, widen them. Consider [0.1, 5.0] for the new calibrator.

### Pitfall 15: Race-Level Aggregation Window Mismatch

**What goes wrong:** Features computed at the race level (e.g., "average model probability of top 3 horses in this race") require complete race data. If some horses are filtered out (e.g., missing features, surface filter removes wrong entries), the aggregation is based on a partial field and distorts the features for the remaining horses.

**Why it happens:** Data filtering happens at multiple stages. Race-level features computed before filtering include all horses; computed after filtering, they miss some. Neither is clearly correct.

**Prevention:** Compute race-level aggregations before filtering. Use NaN-safe aggregation (pandas `skipna=True`). Flag races where >30% of entries were excluded from aggregation.

### Pitfall 16: EV Factor Compounding in Segment Calibration

**What goes wrong:** The existing `WinSegmentCalibrator` can apply both `p_factor` (probability shrinkage) and `ev_factor` (EV shrinkage). When both are active, the effective edge is `p * odds * p_factor * ev_factor - 1`, which compounds two shrinkage factors. Even though `apply_ev_factor=False` by default, if it gets enabled, the double shrinkage can reduce edge below the betting threshold for marginal bets.

**Why it happens:** The two factors were designed to correct different things (probability calibration vs EV calibration) but they compound multiplicatively on the edge.

**Prevention:** The new MarketAwareWinCalibrator should apply a single correction factor, not separate p and EV corrections. If EV correction is needed, apply it as a post-processing step after the unified probability correction, not in parallel.

### Pitfall 17: Deployment Gate Optimizing for Wrong Metric

**What goes wrong:** The project mandates deployment decisions based on Brier/logloss/ECE, not ROI. But if the deployment gate implementation accidentally includes ROI as a criterion (or if the threshold tuning is done on ROI), the system will overfit to historical betting outcomes and deploy models that are lucky rather than calibrated.

**Why it happens:** ROI is the business metric everyone watches. There is strong temptation to use it as a gate. But ROI is noisy (depends on which bets were selected) while calibration metrics directly measure probability quality.

**Prevention:** Explicitly exclude ROI from the deployment decision function. The deployment gate should check: (1) Brier score improvement vs baseline, (2) logloss improvement vs baseline, (3) ECE improvement vs baseline, (4) no year-over-year regression in these metrics. ROI is reported but not a gate criterion.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Phase 0: OOF Health | Empty OOF overwriting valid artifacts (false negative -- corrupted OOF passes) | Write OOF to temporary file first; validate before atomic rename to final path |
| Phase 0: OOF Health | Same-date race leakage across folds | Split by race_date with 1-day gap; validate max(train_date) < min(val_date) |
| Phase 0: OOF Health | Overly strict thresholds blocking valid models | Use statistical significance tests instead of fixed thresholds for warning-level checks |
| Phase 1: InvestmentFeatureFrame | Post-race data in feature lineage | Feature allowlist + lineage tracing per feature + CI gate |
| Phase 1: InvestmentFeatureFrame | NaN cascades from missing historical data | Default to population-level imputation; log NaN rates per feature per race |
| Phase 1: InvestmentFeatureFrame | Late odds features unavailable at prediction time | Document data availability timestamp per feature; provide fallback to morning odds |
| Phase 1: InvestmentFeatureFrame | Market model Rule 11 violation (using p_market_pred) | Explicit allowlist: raw market features only, never model outputs |
| Phase 2: MarketAwareWinCalibrator | Double-correction with existing BenterCombination | Replace (not append to) WinBenterGate; disable old pipeline when new one is active |
| Phase 2: MarketAwareWinCalibrator | Race normalization destroying calibration | Normalize before or during calibration, never after; validate ECE pre/post normalization |
| Phase 2: MarketAwareWinCalibrator | Blend weight instability from multicollinear features | Use difference features (model-market) instead of raw duplicates; check VIF |
| Phase 3: Segment Calibration | Year-dependent segment effect inversion | Strong Bayesian priors; validate sign stability across training years |
| Phase 3: Segment Calibration | Segment boundary discontinuity | Use continuous features instead of hard bins; validate adjacent segment factor gap <0.03 |
| Phase 3: Segment Calibration | Compounding p_factor and ev_factor | Single unified correction factor; avoid parallel shrinkage paths |
| Phase 4: Race-Level Ranker | Group parameter misalignment with filtered data | Compute groups AFTER filtering; assert sum(groups)==len(X); sort by race_id |
| Phase 4: Race-Level Ranker | Label sparsity (1 winner per ~15 horses) | Graded relevance labels; lambdarank_truncation_level=3; minimum 10K training races |
| Phase 4: Race-Level Ranker | Position bias from input ordering | Randomize horse order within races or enable position bias regularization |
| Phase 4: Race-Level Ranker | Optimizing for ROI instead of probability quality | Deployment gate checks Brier/logloss/ECE only; ROI is monitored but not a gate |

## Cross-Phase Integration Pitfalls

| Concern | Phases Involved | Risk | Mitigation |
|---------|----------------|------|------------|
| Cascading calibration over-correction | 2 + 3 | Critical | MarketAwareWinCalibrator absorbs WSC; old pipeline bypassed |
| Feature availability at prediction time | 1 + 2 + 4 | Critical | Feature allowlist with data-availability timestamps; CI validation |
| OOF quality affects all downstream calibration | 0 + 2 + 3 | Critical | Phase 0 must complete and pass all checks before Phase 2 begins |
| Ranker training depends on calibrator output quality | 2 + 4 | Moderate | Freeze calibrator before training ranker; validate calibrator on ranker's training period |
| Segment corrections conflicting with ranker preferences | 3 + 4 | Moderate | Train ranker with segment features already applied; do not apply segment corrections post-rank |
| Market model Rule 11 isolation | 1 + 2 | Moderate | InvestmentFeatureFrame uses raw market features only; CI check for p_market_pred |

## Sources

- Codebase analysis: `src/models/win_benter_gate.py` (race normalization ordering, OOF walk-forward splits)
- Codebase analysis: `src/models/win_segment_calibrator.py` (segment calibration with Bayesian shrinkage, p_factor/ev_factor compounding)
- Codebase analysis: `src/models/benter_combination.py` (logit-space blending, beta bounds [0.20, 5.0], temperature scaling bounds [0.3, 3.0])
- Codebase analysis: `src/models/win_selection_policy.py` (surface-specific scoring, deployment conditions)
- Codebase analysis: `src/models/market_model.py` (Rule 11: only log_error to Stage2, KFold OOF without date-level separation)
- Codebase analysis: `src/models/ev_correction_model.py` (P/E decomposition, confirmed_odds usage for training labels)
- Codebase analysis: `src/features/leakage_validators.py` (expanding feature validation framework)
- LightGBM documentation: lambdarank parameters (group, label_gain, truncation_level, position_bias_regularization, min_data_per_group)
- ROI_IMPROVEMENT_PLAN.md: Phase specifications, deployment conditions, segment calibration options
- .planning/PROJECT.md: v2.0 milestone structure, key decisions (regime-independent, bet count reduction forbidden)
- Benter (1994): "Computer Based Horse Race Handicapping and Wagering Systems" -- logit-space blending methodology
- Confidence: HIGH for codebase-derived pitfalls (directly observed in source code), MEDIUM for LightGBM ranker pitfalls (documentation-verified but not yet tested in this specific codebase), MEDIUM for cross-phase integration pitfalls (predicted from code structure but not yet observed at runtime)
