# Phase 40: Race-Level Ranker - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

A learned ranker orders horses within each race by combining relevance (win/finishing-position graded relevance) and value/mispricing (EV residual + model-vs-market gap + uncertainty) signals into a single investment_score per horse, operating in shadow mode behind the existing baseline selectors.

**In scope:** RNK-01 through RNK-05 (learned Win relevance ranker, learned Value/mispricing ranker, investment_score combination, shadow mode deployment, one-bet-per-race baseline bet count preservation).
**Out of scope:** Place/wide ranker, ROI-label ranker (overfit risk), bet count reduction as ROI improvement, feature routing audit (Phase 42), shadow comparison framework (Phase 41), regime-dependent ranking.

</domain>

<decisions>
## Implementation Decisions

### Ranker Model Architecture

- **D-01:** Two separate Ridge/regularized linear models: `relevance_scorer` (learns win relevance) and `value_scorer` (learns value/mispricing). Each model is a per-surface independent scorer stored in SubmodelSet.
- **D-02:** LightGBM LambdaRank trained as shadow benchmark only — not the default deployable model. Do not deploy LightGBM in v2.1 unless it clearly improves OOF/WF metrics without worsening year/surface reliability, bet count, or drawdown.
- **D-03:** Combination formula: `investment_score = 0.35 * relevance_score_pct + 0.35 * value_score_pct + 0.20 * calibrated_log_ev_pct - 0.10 * uncertainty_penalty_pct`. All components are race-level robust percentile ranks before combination.
- **D-04:** Weights are pre-declared and NOT optimized on 2024/2025. OOF/WF diagnostics may report sensitivity for alternative weight sets, but deployment does not select weights by maximizing ROI on test periods.
- **D-05:** Report each component separately in shadow diagnostics so selection changes are explainable (relevance_pct, value_pct, log_ev_pct, uncertainty_pct, and final investment_score).
- **D-06:** Ridge alpha selection via deterministic grid: [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]. Primary metrics: relevance_scorer → NDCG@3 / top1 win relevance; value_scorer → rank correlation + top1/top3 value capture. Tie-breaker: larger alpha (stronger regularization). Do not use logloss for ranker selection.
- **D-07:** Validation uses chronological race-level WF folds (same definitions as Phase 39). Do not tune on 2024/2025 fixed test folds.

### Training Target / Label Design

- **D-08:** Relevance scorer target: graded relevance based on finishing position — 1.00 (1st), 0.55 (2nd), 0.30 (3rd), 0.10 (4th-5th), 0.00 (otherwise). kakuteijyuni used as label only, never as a feature.
- **D-09:** Value scorer target: composite OOF-safe `value_target = clipped_log_ev + mispricing_bonus - uncertainty_penalty`, where:
  - `clipped_log_ev = clip(log(calibrated_ev_oof), -1.0, 1.0)` using OOF calibrated probability and pre-race odds
  - `mispricing_bonus = clipped(logit(p_model_oof) - logit(p_market_norm))`, scaled/clip stabilized
  - `uncertainty_penalty` from Phase 38 uncertainty features
- **D-10:** Actual return/payout is diagnostic only — never a training target. Realized return is too sparse and turns the ranker into an ROI-label overfit machine.
- **D-11:** Binary is_win diagnostics reported separately: top1 win rate, NDCG@3 using relevance_target, rank of actual winner, top3 contains winner.

### OOF Training Data

- **D-12:** Extend Phase 39's `generate_win_oof_predictions()` to emit ranker-required columns. Same chronological fold definitions. No separate fold generation.
- **D-13:** Required columns: race_id, umaban, race_date, surface, fold_id, kakuteijyuni (label only), p_win_oof / p_win_market_aware_oof, p_market_norm, calibrated_ev_oof, model_market_gap features, uncertainty features, odds/return for diagnostics only.
- **D-14:** Build training data from: OOFHealthValidator-passed OOF artifacts + InvestmentFeatureFrame train-mode output + MarketAwareWinCalibrator OOF/shadow outputs.
- **D-15:** Probability-derived rank/bucket features (e.g., p_win_race_rank_pct) must be recomputed from OOF probabilities, not from train-mode predictions.

### Existing Selector Relationship

- **D-16:** Parallel shadow first. Ranker produces investment_score in shadow mode alongside existing selectors (WinSelectionGate, WinSelectionPolicy, WinProfitSelector). Does not replace baseline until validation gates pass.
- **D-17:** Existing selectors remain fully functional behind feature flags. No deletion of existing selectors in v2.1. Removal only considered in a later cleanup milestone after stable shadow/backtest evidence.
- **D-18:** Shadow diagnostics must compare: baseline selected horse vs ranker selected horse, score components breakdown, agreement rate.
- **D-19:** If gates pass: ranker may replace WinSelectionPolicy as race-internal ranking score. WinProfitSelector remains disabled/shadow-only unless revalidated with ranker score. WinSelectionGate remains as baseline/fallback.

### Integration Position

- **D-20:** Ranker scores computed after MarketAwareWinCalibrator and InvestmentFeatureFrame construction, before final candidate sorting. Score ALL race runners — do not restrict to WinSelectionGate-passed horses. This avoids inheriting gate selection bias.
- **D-21:** In shadow mode, compute investment_score for all runners and add columns to diagnostics. Baseline candidate selection remains unchanged.

### Feature Set

- **D-22:** Curated feature subsets (~12-16 features for relevance, ~14-18 features for value). Do not use full 94-feature InvestmentFeatureFrame for Ridge deployable model.
- **D-23:** Relevance scorer features (canonical IFF names, match Phase 38 schema):
  p_win_market_aware or p_win_final, p_win_race_rank_pct, if_p_ability_win, rel_p_ability_win_rank or rel_p_ability_win_zscore, if_norm_finish_avg, if_closing_index, if_weighted_recent_form_finish, if_weighted_recent_form_time, if_jockey_wr, if_trainer_wr (if available), if_blood_surface_wr, if_class_level, if_surface, if_distance_bin, if_grade_code, field_size
- **D-24:** Value scorer features (canonical IFF names, match Phase 38 schema):
  if_logit_gap, if_p_diff or if_market_residual, if_p_ratio / if_market_value_ratio, if_edge_win, if_ev_calibrated or if_calibrated_log_ev, if_odds_log, if_odds_band, if_odds_rank, if_model_prob_rank_pct / p_win_race_rank_pct, if_odds_drop_rate_60_10, if_odds_drop_rate_30_10, if_late_odds_drop_z, if_market_share_change (if available), if_overround_proxy, if_market_entropy, if_model_market_disagreement, if_conformal_width, if_ev_uncertainty_proxy
- **D-25:** Feature names must match actual Phase 38 schema. If a listed feature is unavailable, use registered missing/default behavior — no ad hoc dropping.
- **D-26:** No actual payout or realized ROI features as predictors. CLV may be diagnostic/auxiliary only if OOF-safe — not an inference-time feature unless known before bet placement.

### Standardization & Combination

- **D-27:** Race-level robust percentile ranks as primary normalization for all components. Use deterministic tie handling: rank(method="average" or "first" with stable sort by race_id/umaban).
- **D-28:** Do not use race-level z-score as primary — small field sizes and outliers destabilize the blend. Z-scores may be logged as diagnostics only.

### Claude's Discretion

- Exact feature matrix construction and missing-feature handling within IFF schema rules.
- LightGBM LambdaRank shadow training configuration (objective, hyperparameters).
- SubmodelSet field naming for ranker models (follow existing pattern).
- Test structure and naming within existing conventions.
- Model serialization format (joblib consistent with existing patterns).
- Exact integration code in RacePredictor.predict() and get_win_candidates().

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Components Being Shadowed (NOT deleted in v2.1)
- `src/models/win_selection_gate.py` — WinSelectionGateModel class. Bayesian bucket scorer. Preserved as baseline/fallback (RNK-04).
- `src/models/win_selection_policy.py` — WinSelectionPolicy class. Parameteric score formula. Potential future replacement candidate.
- `src/models/win_profit_selector.py` — WinProfitSelector class. 0-3 candidate grid search. Stays shadow-only if ranker deploys.

### Pipeline Integration Points
- `src/backtest/race_predictor.py` — RacePredictor.predict() lines 268-293 (after MAWC, before place inference). get_win_candidates() lines 573+ (final sorting). Ranker integrates between calibrator and final sort.
- `src/pipelines/training_pipeline.py` — TrainingPipelineV5. OOF generation (extend Phase 39), ranker training, MLflow/local save.
- `src/db/model_loader.py` — Model loading/saving. Add ranker artifacts following existing SubmodelSet pattern.

### Feature Sources
- `src/investment/feature_frame.py` — InvestmentFeatureFrameBuilder providing canonical if_* features.
- `src/investment/schema_registry.py` — 94 specs / 9 categories with dual-mode source resolution. Feature names MUST match this registry.
- `src/investment/leakage.py` — Leakage validators for train vs infer mode.

### Model Sources
- `src/models/market_aware_win_calibrator.py` — Phase 39 output providing calibrated p_win_final. Ranker integrates after this.
- `src/domain/models.py` — SubmodelSet dataclass. Add race_level_ranker fields following existing pattern (optional, None default).

### Validation
- `src/validation/oof_health.py` — OOFHealthValidator. Ranker training must use OOF-validated predictions.

### Requirements
- `.planning/REQUIREMENTS.md` — RNK-01 through RNK-05 (Phase 40 requirements).
- `.planning/ROADMAP.md` — Phase 40 success criteria (5 items).
- `.planning/PROJECT.md` — Key Decisions table (Race-Level Ranker is LEARNED, Shadow-first deployment).

### Prior Phase Context
- `.planning/phases/39-marketawarewincalibrator/39-CONTEXT.md` — Phase 39 decisions on calibrator architecture, OOF generation, shadow mode pattern.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **generate_win_oof_predictions()** (`src/pipelines/training_pipeline.py`): Phase 39 extended this to emit calibrator columns. Extend further for ranker labels and features. Same fold definitions.
- **InvestmentFeatureFrame** (`src/investment/feature_frame.py`): Dual-mode builder with leakage guard. Already computes if_logit_gap, if_edge_win, if_ev_calibrated, if_conformal_width, etc. Train-mode sources resolve to OOF-safe columns.
- **MarketAwareWinCalibrator** (`src/models/market_aware_win_calibrator.py`): Shadow mode pattern (is_trained + deployment_status). Ranker should follow same pattern.
- **OOFHealthValidator** (`src/validation/oof_health.py`): Validates OOF artifacts. Ranker training data must pass this validation.
- **sklearn Ridge**: Available via existing scikit-learn dependency. No new pip dependencies needed.

### Established Patterns
- **Shadow mode via is_trained property**: MarketAwareWinCalibrator sets _trained=False and deployment_status="shadow_only" when beta_market guard fails. RacePredictor checks `if model is not None and model.is_trained`. Ranker should follow this exact pattern.
- **Per-surface models in SubmodelSet**: Each surface (turf/dirt) has independent models stored as optional fields. Ranker follows same pattern: `win_relevance_scorer_turf`, `win_relevance_scorer_dirt`, etc.
- **Chronological WF folds**: Training pipeline uses race_date-sorted chronological folds for OOF. Ranker uses same fold definitions.
- **joblib serialization**: sklearn models stored as .joblib files. ModelLoader loads from `{name}_{surface}.joblib`.

### Integration Points
- **RacePredictor.predict()** after line 277 (MAWC): Add ranker scoring block. Compute investment_score for all runners.
- **RacePredictor.get_win_candidates()** lines 830-845: Current win_market_selection_score computation. In shadow mode, add ranker score as alternative diagnostic column.
- **SubmodelSet**: Add fields: `win_relevance_scorer: Ridge | None`, `win_value_scorer: Ridge | None`, optionally LightGBM shadow variants.
- **TrainingPipelineV5**: Add ranker training block after MarketAwareWinCalibrator training. Same OOF data, different target columns.
- **ModelLoader**: Add load/save for ranker .joblib artifacts per surface.

</code_context>

<specifics>
## Specific Ideas

- Graded relevance target: {1.00, 0.55, 0.30, 0.10, 0.00} — captures 2nd/3rd place information that binary is_win misses.
- Composite value target: clipped_log_ev + mispricing_bonus - uncertainty_penalty — avoids ROI-label overfitting while capturing expected mispricing.
- Fixed pre-declared weights: 0.35/0.35/0.20/0.10 — simple, stable, explainable. Not ROI-optimized.
- Robust percentile rank normalization — outlier-resistant, field-size-independent, deterministic tie handling.
- Shadow diagnostics compare baseline vs ranker selected horse per race with full score component breakdown.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.
</deferred>

---

*Phase: 40-Race-Level Ranker*
*Context gathered: 2026-05-28*
