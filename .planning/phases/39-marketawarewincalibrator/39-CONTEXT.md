# Phase 39: MarketAwareWinCalibrator - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace WinBenterGate + WinSegmentCalibrator with a single MarketAwareWinCalibrator that blends model and market logits via LogisticRegression + L2 regularization, incorporating segment conditioning features (popularity rank, odds band, probability rank) as regularized features/interactions — not per-segment coefficients — to produce calibrated win probabilities satisfying sum-to-1.0 per race.

**In scope:** CAL-01 through CAL-05 (Benter logit-blend, segment conditioning via InvestmentFeatureFrame, regularized global model, removal of dual gate+segment chain, probability quality + normalization).
**Out of scope:** Regime-dependent calibration, place/wide model changes, new data sources, Optuna optimization, deployment gate automation.

</domain>

<decisions>
## Implementation Decisions

### Calibrator Model Type

- **D-01:** LogisticRegression + L2 regularization (sklearn) as primary deployable model. Benter-style feature basis: logit(p_model_oof), logit(p_market_norm), segment features, regularized interactions.
- **D-02:** LightGBM may be trained only as a shadow benchmark — not the default deployable model.
- **D-03:** Beta_market effective contribution must retain a floor/guard equivalent to beta_market >= 0.20; deployment fails if violated. Enforce via coefficient inspection on the logit(p_market) column.
- **D-04:** C selection via deterministic WF grid search over [0.03, 0.1, 0.3, 1.0, 3.0]. Primary metric: logloss. Secondary: Brier and ECE. Tie-breaker: smaller C (stronger regularization). No Optuna.
- **D-05:** Race-level chronological folds for C selection. Require year-level and surface-level actual/predicted ratio not to worsen materially. If no C passes gates, do not deploy — run shadow only.
- **D-06:** Feature interactions: main effects + logit(p_model) × segment + logit(p_market) × segment only. No segment × segment interactions (recreates WSC overfit). Segment × segment may be logged as experimental shadow variant only.
- **D-07:** fit_intercept=True — the intercept is the Benter gamma equivalent for global calibration bias correction.

### Odds Band Definition

- **D-08:** Use both continuous log_odds = log1p(tanodds) as a main effect AND a coarse odds_band categorical feature for interactions.
- **D-09:** Fixed odds bands: [1.0, 2.0), [2.0, 3.0), [3.0, 5.0), [5.0, 10.0), [10.0, 30.0), [30.0, 100.0), [100.0+] — 7 bands.
- **D-10:** One-hot encoding for odds_band (not ordinal 0-6). Ordinal imposes linear spacing inappropriate for favorite-longshot bias.
- **D-11:** Fixed category order. All 7 expected columns present in output schema even if a band is absent in a fold. L2 controls added dimensions.

### Segment Feature Encoding

- **D-12:** popularity_rank_pct = popularity_rank / field_size, clipped to [0, 1] — continuous main effect.
- **D-13:** popularity_bucket one-hot with 5 fixed buckets: pop_1, pop_2_3, pop_4_6, pop_7_9, pop_10_plus — used for interactions with logit features.
- **D-14:** p_win_race_rank_pct — continuous [0, 1] percentile as main effect.
- **D-15:** p_rank bucket one-hot with 3 fixed buckets: top 25%, mid 25-75%, bottom 25% — used for interactions with logit features.
- **D-16:** All bucket boundaries are fixed, not data-fitted. Stable and domain-readable.
- **D-17:** Total feature dimensions: ~51 (6 main effects + 15 segment one-hot + 30 logit×segment interactions + intercept). Manageable with L2.

### OOF Training Data Generation

- **D-18:** Hybrid approach: OOF-dependent features recomputed from OOF predictions; market-only/race-static features joined from InvestmentFeatureFrame.
- **D-19:** OOF-dependent (MUST recompute): p_model_oof, p_win_race_rank_pct, p_rank_top/mid/bottom, any model-probability derived column.
- **D-20:** Market/static (reusable from IFF by race_id/umaban): p_market_norm, logit(p_market_norm), tanodds, log_odds, odds_band, popularity_rank_pct, popularity_bucket, field_size, surface, race_date, race_id, umaban.
- **D-21:** Extend existing generate_win_oof_predictions() — same fold definitions, emit additional columns. If existing loop cannot expose needed columns cleanly, refactor into reusable OOF generation helper first.
- **D-22:** Tests must verify: train-mode p_win_pred is REJECTED, p_win_oof is used for all probability-dependent segment features.
- **D-23:** Prefer simplest model that passes OOF/WF probability-quality gates. If no model passes, shadow-only deployment.

### Claude's Discretion

- Feature matrix standardization strategy for non-logit segment features (standardize if needed, but logit features are already on similar scale).
- Exact implementation of the checkpoint file and incremental save logic.
- Test structure and naming within existing conventions.
- Model serialization format (joblib consistent with existing patterns).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Components Being Replaced
- `src/models/win_benter_gate.py` — WinBenterGate class to be REMOVED (entire file). Current pipeline position at RacePredictor.predict() lines 282-290.
- `src/models/win_segment_calibrator.py` — WinSegmentCalibrator class to be REMOVED (entire file). Current application at RacePredictor.get_win_candidates() lines 657-679.
- `src/models/benter_combination.py` — BenterCombination class RETAINED for place prediction. win_benter instance replaced; benter_combo (place) unchanged.

### Pipeline Integration Points
- `src/backtest/race_predictor.py` — RacePredictor.predict() lines 282-290 (WinBenterGate slot → MarketAwareWinCalibrator slot). Also lines 153-164 (_get_win_segment_calibrator → remove) and lines 657-679 (segment calibrator application → remove).
- `src/pipelines/training_pipeline.py` — Lines 1329-1417 (win_benter training → replace with MarketAwareWinCalibrator training). Lines 1624-1634 (win_segment_calibrator training → remove). Lines 2318-2327, 2500-2502, 2521-2522 (MLflow/local save → update).
- `src/db/model_loader.py` — Lines 350-357, 425, 431, 746-750, 829-837, 906, 912 (model loading/saving → update for new calibrator).

### Feature Sources
- `src/investment/schema_registry.py` — Line 567 (if_odds_band_id — "バンド変換はPhase 39で実装"). Lines 603-627 (if_odds_band_median_ev, if_odds_band_count, if_odds_band_ev_rank — related odds band features).
- `src/investment/feature_frame.py` — InvestmentFeatureFrame builder providing segment keys.

### Domain Model
- `src/domain/models.py` — SubmodelSet lines 257-259 (win_benter, win_isotonic_calibrator, win_temperature_scaler → remove/replace). Line 264 (win_segment_calibrator → remove).

### Requirements
- `.planning/REQUIREMENTS.md` — CAL-01 through CAL-05 (Phase 39 requirements with acceptance criteria).
- `.planning/ROADMAP.md` — Phase 39 success criteria (4 items: Benter logit-blend, segment conditioning, probability quality, removal of dual chain).
- `.planning/PROJECT.md` — Key Decisions table (Benter型市場ブレンド, 配備条件=確率品質).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **BenterCombination** (`src/models/benter_combination.py`): The combine() method's logit-space math (alpha*logit(p_fund) + beta*logit(p_market) + gamma) is the conceptual foundation. The new calibrator generalizes this via LogisticRegression coefficients.
- **generate_win_oof_predictions()** (`src/pipelines/training_pipeline.py`): Existing WF OOF loop with chronological folds. Extended to emit additional columns for calibrator training.
- **InvestmentFeatureFrame** (`src/investment/feature_frame.py`): Already computes if_popularity_rank, if_p_win_race_rank. Schema registry at `src/investment/schema_registry.py` already earmarks if_odds_band_id for Phase 39 implementation.
- **IsotonicRegression** (sklearn): Used in WinBenterGate for optional post-blend calibration. MarketAwareWinCalibrator's LogisticRegression + L2 replaces this directly — no separate isotonic step needed.
- **OOFHealthValidator** (`src/validation/oof_health.py`): Validates OOF artifact integrity. Calibrator training must use OOF-validated predictions.

### Established Patterns
- **sklearn LogisticRegression with L2**: Used throughout the codebase (IsotonicRegression is sklearn, sklearn is a dependency). Consistent with "no new pip dependencies" constraint.
- **One-hot encoding with fixed schema**: RacePredictor and feature modules use fixed column sets. Calibrator must follow same pattern — all expected one-hot columns present even if absent in training fold.
- **Chronological WF folds**: Training pipeline uses race_date-sorted chronological folds for OOF. Calibrator C-selection uses same fold definitions.
- **sum-to-1.0 normalization**: WinBenterGate normalizes via `p_final = p_combined / p_combined.sum()` per race. MarketAwareWinCalibrator should use same approach.
- **SubmodelSet dataclass**: All per-surface models stored as fields on SubmodelSet. New calibrator follows same pattern.

### Integration Points
- **RacePredictor.predict()** lines 282-290: Replace WinBenterGate instantiation/apply with MarketAwareWinCalibrator instantiation/apply. Same pipeline position (after EV correction, before WinSelectionGate).
- **RacePredictor.get_win_candidates()** lines 657-679: Remove WinSegmentCalibrator application entirely.
- **SubmodelSet**: Remove win_benter, win_isotonic_calibrator, win_temperature_scaler, win_segment_calibrator fields. Add market_aware_win_calibrator field.
- **TrainingPipelineV5**: Replace win_benter training block (lines 1329-1417) with MarketAwareWinCalibrator training. Remove win_segment_calibrator training (lines 1624-1634). Update save/load.
- **ModelLoader**: Update local/MLflow load to handle new calibrator artifact instead of win_benter + win_segment_calibrator.

</code_context>

<specifics>
## Specific Ideas

- Feature matrix for LogisticRegression: [logit(p_model_oof), logit(p_market_norm), log_odds, popularity_rank_pct, p_win_race_rank_pct, odds_band_1hot(7), popularity_bucket_1hot(5), p_rank_bucket_1hot(3), interactions(30)] = ~51 dimensions.
- C grid [0.03, 0.1, 0.3, 1.0, 3.0] with logloss primary, smaller C tie-breaker.
- beta_market coefficient guard: check LogisticRegression.coef_ column corresponding to logit(p_market_norm) and ensure effective contribution >= 0.20 threshold.
- Train OOF frame built from win_selection_oof artifacts: recompute p_win_race_rank_pct from OOF predictions grouped by race_id, join static/market columns from InvestmentFeatureFrame by race_id/umaban.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.
</deferred>

---

*Phase: 39-MarketAwareWinCalibrator*
*Context gathered: 2026-05-27*
