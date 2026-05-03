# Project Research Summary

**Project:** keiba-ai v1.1 ROI Advanced Model
**Domain:** Parimutuel horse racing win prediction (JRA Japan) -- ensemble stacking, odds deviation EV, time-series features
**Researched:** 2026-05-03
**Confidence:** HIGH

## Executive Summary

This is a machine learning prediction system for Japanese horse racing (JRA) that currently achieves 89% ROI (a loss) in backtest after the v1.0 milestone. The v1.1 milestone aims to push ROI above 100% through three targeted improvements: (1) replacing the single LightGBM hit model with a 3-model stacking ensemble (LightGBM + XGBoost + CatBoost + Ridge meta-learner), (2) adding odds deviation features that quantify model-vs-market disagreement as a direct EV signal, and (3) enriching the time-series and pace feature pipeline to give the models better inputs. The existing codebase already has a `StackedEnsemble` class and all required dependencies (XGBoost 3.2.0, CatBoost 1.2.10) installed -- this is primarily an enhancement and optimization effort, not greenfield development.

The recommended approach follows a strict feature-first, model-second build order: add time-series and pace features first (they are the foundation), then add odds deviation features (which depend on improved ability estimates), and finally enhance the stacking ensemble (which amplifies everything). The existing `StackedEnsemble` class is the right foundation but is critically under-optimized -- all three base models use identical hyperparameters (lr=0.03, leaves/depth 31/6, 300 rounds), which defeats the purpose of stacking since the models produce nearly identical predictions. Forcing diversity through differentiated hyperparameters, feature subsampling, and early stopping per model is the single highest-leverage change.

The key risks are temporal data leakage (odds features using near-post-time snapshots, time-series features including current-race data) and meta-learner overfitting (Ridge fitting noise in correlated base model outputs). These are mitigated by tightening snapshot tolerance from 15 to 5 minutes, adding PIT boundary assertions to all `searchsorted` calls, and using higher Ridge regularization with diverse base model configurations. The EV decomposition chain (hit model -> return model -> EV correction) must always be retrained end-to-end after any model architecture change.

## Key Findings

### Recommended Stack

No new packages are needed. All required dependencies are installed at current stable versions. The stack recommendation is to keep the existing custom `StackedEnsemble` with manual OOF + Ridge meta-learner rather than switching to `sklearn.ensemble.StackingClassifier`, because the project uses native APIs (`lgb.train()`, `xgb.train()`, `CatBoostClassifier`) for low-level control over Dataset construction, group handling, and init_score injection. Version pin updates in `pyproject.toml` are recommended but not blocking.

**Core technologies:**
- LightGBM 4.6.0: Stage1 ranker + primary binary base model -- native categorical support, fastest training, leaf-wise growth
- XGBoost 3.2.0: Secondary base model in stack -- complementary depth-wise tree growth, different regularization
- CatBoost 1.2.10: Tertiary base model in stack -- symmetric trees create different decision boundaries, ordered boosting reduces overfitting
- scikit-learn Ridge 1.8.0: Meta-learner -- L2 regularization prevents overfitting on correlated base predictions
- pandas/numpy/scipy: Feature engineering -- rolling windows, expanding stats, vectorized computation; no specialized time-series library needed

### Expected Features

**Must have (table stakes):**
- 3-model stacking with hyperparameter diversity -- the existing identical-config stacking must be differentiated per model
- OOF expanding-window meta-learner training -- already correct, increase from 3 to 5 folds
- Model-vs-market probability gap (`odds_to_ability_ratio = p_market / p_ability`) as Stage2 feature -- the core edge signal
- Exponential decay weighting on past performance averages -- drop-in replacement for simple means in all history features
- Odds acceleration (2nd derivative) -- simple addition to existing `odds_dynamics_features.py`

**Should have (competitive):**
- Feature subsampling per base model (feature_fraction=0.7/0.8/0.8) -- forces diversity
- Early stopping per base model -- prevents overfitting
- Edge confidence interval from conformal prediction -- better bet selection
- Kelly stake sizing from stacked probability -- converts improved estimates into ROI
- Class-adjusted form metric -- contextualizes form by opposition quality
- Composite pace figure per horse -- synthesizes corner positions into pace metric

**Defer (v1.2+):**
- Stacking at Stage1 (Ranker) level -- complex, uncertain payoff
- Projected position at each corner -- requires field-level interaction modeling
- Volume-weighted odds movement -- depends on data availability (unverified)
- Late money intensity (t-5 vs t-10) -- depends on snapshot granularity
- LSTM/transformer temporal modeling -- data too sparse

### Architecture Approach

The v1.1 milestone extends the existing 2-stage decomposition pipeline (P(hit) x E(odds|hit)) at three insertion points. New feature modules (time_series_features, pace_prediction_features, odds_deviation_features) plug into `FeatureEngine.build_all()` and `_train_submodel()`. The StackedEnsemble replaces `lgb.Booster` via duck typing (`.predict()` + `.best_iteration`). The pipeline ordering is strict: features must exist before models that consume them, and single models must work before stacking layers on top.

**Major components:**
1. `src/features/time_series_features.py` (NEW) -- temporal trend features from past runs (time progression slope, closing speed trend, form volatility)
2. `src/features/pace_prediction_features.py` (NEW) -- race-level pace scenario prediction from field composition
3. `src/features/odds_deviation_features.py` (NEW) -- model-vs-market deviation metrics feeding into hit model
4. `src/models/stacked_ensemble.py` (MODIFY) -- enhance with diverse hyperparameters, 5-fold OOF, early stopping, MLflow logging

### Critical Pitfalls

1. **Meta-learner overfitting from correlated base models** -- Force diversity via differentiated hyperparameters (num_leaves 15/31/63, feature_fraction 0.5/0.7/0.9), add 5-10 key features alongside base predictions to meta-learner inputs, use Ridge alpha=10.0 or higher. Validate ensemble beats best single model by at least 1% AUC on held-out year.
2. **Temporal leakage from full-data retraining in stacking** -- After OOF meta-learner training, the code retrains base models on ALL data (train+valid), creating a mismatch between OOF-calibrated meta-learner and stronger production base models. Either skip full-data retraining (Option A, safer) or add a calibration step comparing OOF vs full-data predictions.
3. **Odds look-ahead bias from near-post-time snapshots** -- `_pick_target_snapshot` has `tolerance_minutes=15.0` which allows post-time snapshots to match "t-10" targets. Tighten to 5 minutes. Never fall back to `confirmed_odds` for features.
4. **Time-series feature PIT boundary violations** -- New time-series features using `searchsorted` must use strict `race_date < target_date` (not `<=`). Add explicit assertions. Extend `leakage_validators.py` to cover horse history and pace features.
5. **EV decomposition miscalibration after stacking** -- Stacking only the hit model while leaving return model and EV correction unchanged breaks the joint calibration. Always retrain the full chain: hit model -> return model -> EV correction after any model architecture change.

## Implications for Roadmap

### Phase 1: Time-Series Features
**Rationale:** Features are the foundation. Adding them first means all subsequent model improvements benefit from richer inputs. No risk of regression since LightGBM ignores unused features. Time-series features have no model dependencies -- they derive from raw data and past performance only.
**Delivers:** New feature module `time_series_features.py` with temporal trend features (time progression slope, closing speed trend, form volatility, peak form indicator). Exponential decay weighting on existing history averages.
**Addresses:** Past-Run Time-Series Enhancement features from FEATURES.md
**Avoids:** Pitfall 4 (temporal leakage) by implementing PIT-safe `searchsorted` patterns with explicit boundary assertions

### Phase 2: Pace Prediction Features
**Rationale:** Pace features are independent of time-series features but follow the same feature-first principle. They are computed at race level (looking at all entrants) rather than horse level, making them architecturally distinct and requiring separate integration.
**Delivers:** New feature module `pace_prediction_features.py` with predicted pace scenario, front runner count, pace pressure index, position fit score.
**Addresses:** Pace/Position Prediction features from FEATURES.md
**Avoids:** Pitfall 5 (circular reasoning) by auditing pace feature provenance to confirm only pre-race data is used

### Phase 3: Odds Deviation Features
**Rationale:** Deviation features depend on `p_ability_win` from AbilityModel (which is improved by Phase 1 and 2 features). Must come after feature phases. These features directly measure the betting edge and are the strongest ROI signal.
**Delivers:** New feature module `odds_deviation_features.py` with odds deviation signed/absolute/squared, deviation z-score, model confidence gap, deviation-adjusted EV.
**Addresses:** Odds Deviation EV features from FEATURES.md
**Avoids:** Pitfall 3 (look-ahead bias) by tightening snapshot tolerance, Pitfall 13 (EV double-counting) by choosing one path (feature-in-model, consistent with existing v1.0 approach)

### Phase 4: Stacking Ensemble Enhancement
**Rationale:** Stacking is the last layer because it amplifies feature quality. Better features first means the meta-learner has stronger signal to combine. This phase also includes the highest-risk pitfall (meta-learner overfitting), so having validated features first gives a stable baseline to compare against.
**Delivers:** Enhanced `StackedEnsemble` with differentiated hyperparameters per base model, 5-fold expanding window OOF, early stopping, feature subsampling, MLflow meta-learner logging.
**Addresses:** Ensemble Stacking Enhancement features from FEATURES.md (hyperparameter diversity, early stopping, feature subsampling)
**Avoids:** Pitfall 1 (correlated base models) via forced diversity, Pitfall 2 (temporal leakage) via validated OOF approach, Pitfall 6 (EV miscalibration) via full pipeline retrain

### Phase Ordering Rationale

- Features before models: Adding features is additive (LightGBM ignores unused columns), so there is no regression risk. Models before stacking: single-model performance must be validated before adding the complexity of stacking.
- Dependency chain respected: Time-series features (Phase 1) improve AbilityModel output, which feeds into odds deviation features (Phase 3), which improve hit model input, which is what stacking (Phase 4) amplifies.
- Risk escalation: Phases 1-3 are low-risk feature additions. Phase 4 is the highest-risk model architecture change. Isolating risk in the final phase means the feature improvements are safe even if stacking underperforms.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (Odds Deviation):** Needs verification of snapshot granularity in `data/odds/time_series/` -- whether sub-10-minute snapshots exist. Also needs decision on EV double-counting path (feature-in-model vs Benter-combination).
- **Phase 4 (Stacking):** Needs empirical measurement of base model prediction correlation on this specific dataset. If all pairwise correlations exceed 0.90, stacking adds minimal value regardless of meta-learner tuning.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Time-Series Features):** Well-documented pandas rolling/expanding patterns, established `HorseHistoryFeatures` pattern to follow
- **Phase 2 (Pace Features):** Follows existing `pace_aptitude_features.py` and `interaction_features.py` patterns

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All packages installed and verified. Existing StackedEnsemble class demonstrates integration pattern. XGBoost 3.x native API compatibility confirmed. |
| Features | HIGH | Most table-stakes features already exist. New features are well-understood pandas/numpy operations. Feature dependency chain is clear. |
| Architecture | HIGH | Existing pipeline structure is well-documented. Extension points are identified in code with specific line numbers. Build order follows strict data dependencies. |
| Pitfalls | HIGH | 6 critical pitfalls identified with specific code references and prevention strategies. All pitfalls traceable to observable code patterns. |

**Overall confidence:** HIGH

### Gaps to Address

- **Base model prediction correlation:** Unknown how correlated LightGBM, XGBoost, and CatBoost predictions are on this specific dataset. Must measure during Phase 4 before investing in stacking. If correlations are all >0.90, consider skipping stacking and focusing on feature engineering alone.
- **Odds snapshot granularity:** Unverified whether `data/odds/time_series/` contains sub-10-minute snapshots. Affects feasibility of late money intensity feature and snapshot tolerance tightening. Check during Phase 3 planning.
- **`kyakusitukubun_cd` provenance in pace features:** Whether the running style code used in pace computation comes from past races or current race assignment is unverified. Must trace during Phase 2 implementation.
- **Stacking ROI impact on this specific data:** Theoretical benefits of stacking are well-documented, but the actual ROI improvement on JRA data with ~50K training samples is uncertain. Phase 4 should include a baseline comparison (single LightGBM vs stacked) before committing to the ensemble in production.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/models/stacked_ensemble.py`, `src/pipelines/training_pipeline.py`, `src/features/*.py`, `src/models/two_stage_return_model.py`, `src/models/stage1_ability_model.py` -- all extension points and pitfalls verified against actual code
- XGBoost 3.2.0 documentation -- native API stability confirmed, learning-to-rank support
- CatBoost 1.2.10 documentation -- ranking objectives, native categorical handling
- LightGBM 4.6.0 documentation -- binary classification, early stopping, categorical features

### Secondary (MEDIUM confidence)
- Benter (1994) -- foundational model structure for horse racing prediction
- Lessmann et al. (2020) via PMC -- stacking multiple GBMs consistently outperforms single GBM
- Springer (2024) -- Ridge/ElasticNet as optimal meta-learners for stacking with few features
- ResearchGate: Ensemble Learning for Horse Racing -- time-series CV prevents leakage
- Kaggle community -- practical stacking discussion, OOF vs full-data retraining tradeoffs

### Tertiary (LOW confidence, needs validation)
- Exact LightGBM/XGBoost/CatBoost prediction correlation on this dataset -- needs empirical measurement
- Specific JRA odds drift magnitude in final 10 minutes -- needs domain data analysis
- `kyakusitukubun_cd` source (past vs current race) -- needs code tracing during implementation

---
*Research completed: 2026-05-03*
*Ready for roadmap: yes*
