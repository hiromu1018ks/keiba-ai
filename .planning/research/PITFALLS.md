# Domain Pitfalls: Feature Engineering Overhaul (v1.6)

**Domain:** Horse racing ML prediction system (LightGBM/XGBoost/CatBoost stacking)
**Context:** v1.6 milestone -- comprehensive feature engineering overhaul for existing v5.5 system with 100+ features across 20 modules
**Researched:** 2026-05-10
**Confidence:** HIGH (based on direct codebase analysis of all 22 feature modules, v1.5 CQR failure investigation, and domain research)

---

## Critical Pitfalls

Mistakes that cause rewrites or major issues. Ranked by severity based on v1.5 failure evidence and codebase analysis.

### Pitfall 1: Using Model Outputs as Features for Downstream Models (Cascade Overfitting)

**What goes wrong:** A downstream model (e.g., CQR, EV correction, second-stage model) receives the main model's predictions as input features. The downstream model overfits to the training data because the main model's predictions are already highly correlated with the target.

**Why it happens:** The "residual learning" pattern seems theoretically sound -- learn what the main model got wrong. But in practice, the downstream model sees the main model's output (trained on the same data) and essentially memorizes the residuals rather than learning generalizable patterns.

**Consequences:** In v1.5, CQR used 437 features including 44 model output columns (ev_win_calibrated, p_hit, e_return, etc.). Q_90 dropped to 0.0000 (infinitely tight conformal interval). Combined with cascade filters, this produced 272 bets/year with 99.6% win rate. After fix, ROI returned to 84.4% -- the entire v1.5 improvement was fake.

**Prevention:**
- Any new feature module MUST NOT consume model outputs (p_hit, ev_*, edge_*, selection_*, calibrated values) as inputs
- Maintain `_MODEL_OUTPUT_COLS` blacklist alongside `POST_RACE_COLS` in `domain/types.py` as hard boundaries
- If a "second stage" model is needed, it MUST use only raw features, never predictions from earlier stages
- Audit: grep for any import or reference to model output column names inside feature modules

**Detection:**
- Backtest bet count drops below 1000/year (was 272 in v1.5 failure)
- Calibration quantiles (Q_90, Q_80) are near 0.0000
- Win rate exceeds 90% in any segment (place base rate is 18-35%)
- ROI spikes above 200% in any single validation fold

**Phase to address:** First phase of feature audit -- safety gate before any new features are added.

---

### Pitfall 2: Post-Race Data Leakage Through Implicit Column Access

**What goes wrong:** `POST_RACE_COLS` (kakuteijyuni, confirmed_odds, ninki, time, timediff, harontimel3/4, jyuni1c-4c, honsyokin, chakusacd, dmjyuni, dmtime) are defined in `domain/types.py` but survive through multiple merge paths in the pipeline.

**Why it happens:** The architecture merges race_df, entry_df, odds_df, hist_df in multiple stages. Post-race columns leak through specific paths identified in the v1.5 spike investigation:

1. **`feature_engine.py:build_all()`** returns result_df that still contains post-race columns (spike M1 -- not yet fixed)
2. **`popularity_rank` fallback chain**: tanodds -> tanninki -> ninki, where ninki is confirmed popularity (spike M6). When tanodds is unavailable, ninki (post-race) silently becomes a feature
3. **JODDS DataKubun=3/4** (confirmed odds snapshots) are not explicitly filtered in `odds_dynamics_features.py` (spike M5)
4. **`build_features()` inference path** injects kakuteijyuni and ninki into DataFrame (spike L2) -- values should be 0 but have no protective assertion

**Consequences:** Any model that accidentally uses confirmed_odds or kakuteijyuni from the CURRENT race (not history) achieves near-perfect training accuracy but fails completely in production.

**Prevention:**
- At `build_all()` exit, explicitly drop all POST_RACE_COLS from returned DataFrame (spike M1 fix)
- Add CI test: after `build_all()`, assert no POST_RACE_COLS in output (except when explicitly needed for target labels)
- Extend `leakage_validators.py` to validate the final feature matrix, not just expanding() features
- For new features: any module reading from entry_df or race_df MUST document which columns it accesses and confirm they are pre-race

**Detection:**
- Feature importance shows kakuteijyuni, confirmed_odds, ninki, or time among top features
- Training AUC exceeds 0.99 (implausible for horse racing)
- Large gap between training and validation metrics

**Phase to address:** First phase, before feature audit begins. This is a prerequisite safety gate.

---

### Pitfall 3: Feature Audit Using In-Sample Importance (Selection Bias)

**What goes wrong:** When auditing existing features, using impurity-based importance (LightGBM's default `feature_importance()`) on the training set to decide which features to keep. This biases toward high-cardinality features and features that overfit the training data.

**Why it happens:** Tree-based impurity importance measures how much a feature reduces loss during training -- not how much it helps on unseen data. Features that are noisy but high-cardinality can score high. Features that are genuinely predictive but low-cardinality can score low.

**Consequences:**
- Removing genuinely useful low-cardinality features (e.g., categorical features like kyakusitukubun_cd)
- Keeping noisy high-cardinality features that overfit (e.g., engineered ratios that memorize training patterns)
- Net result: model performance degrades after "pruning"

**Prevention:**
- Use permutation importance on a held-out validation set (sklearn `permutation_importance`), not training-set impurity importance
- Use SHAP values for interpretability but NOT as the sole pruning criterion (they also reflect training set behavior)
- Perform feature ablation: remove one feature at a time, retrain, measure validation loss change
- Any feature with permutation importance near zero or negative on the validation set is a candidate for removal
- Validate pruning decisions with walk-forward validation (`run_wf_validation.py`)

**Detection:**
- Model improves on training set but degrades on validation set after feature changes
- Feature importance rankings differ significantly between training and validation sets
- Walk-forward validation shows high ROI gap (>30pp between folds)

**Phase to address:** Feature audit phase. Must establish evaluation methodology before touching features.

---

### Pitfall 4: Adding New Features Without Expanding Validation Rigor

**What goes wrong:** Each new feature increases model capacity. With 100+ features and ~50K training samples (2015-2025), the effective samples-per-feature ratio is already low. Adding 20+ more features without increasing validation rigor leads to overfitting.

**Why it happens:** GBM models handle hundreds of features well computationally, but each feature adds a dimension for memorizing training-specific patterns. The current 2-fold walk-forward validation is too coarse to detect this.

**Consequences:** New features appear to improve backtest ROI (same temporal period) but walk-forward validation reveals the improvement is not real.

**Prevention:**
- For each batch of new features, run walk-forward validation (not just single backtest)
- Use the existing 4-fold walk-forward in `run_strategy_optimization.py` as gold standard
- Require minimum 2pp ROI improvement on ALL folds (not average) before accepting new features
- Track feature count as a metric -- growth from 100 to 150 demands proportionally stricter validation
- Apply "one standard error rule": only accept feature additions if improvement exceeds variance across folds

**Detection:**
- Walk-forward ROI gap > 30pp between folds
- Backtest ROI improves but walk-forward ROI does not
- Feature importance of new features is inconsistent across folds

**Phase to address:** Every phase that adds features. Validation protocol must be established first.

---

### Pitfall 5: EveryDB2 Column Semantics Misunderstanding

**What goes wrong:** EveryDB2 (JRA-VAN DataLab) columns have specific semantics that are not obvious from column names. Some columns that appear pre-race are actually post-race. Some have different meanings depending on DataKubun values.

**Why it happens:** EveryDB2 is a Japanese horse racing database with opaque column names. Key pitfalls already identified:

| Column | Appears to be | Actually is | Status |
|--------|--------------|-------------|--------|
| `odds` (entries) | Pre-race odds | Confirmed odds (post-race) | Handled (tanodds preferred) |
| `ninki` | Pre-race popularity | Confirmed popularity (post-race) | Partially handled (fallback only) |
| `kyakusitukubun` | Running style code | Current race running style (post-race) | Correctly uses kyakusitukubun_cd |
| `kakuteijyuni` | Finishing position | Confirmed finishing position (post-race) | In POST_RACE_COLS |
| JODDS DataKubun 3/4 | Odds snapshot | Confirmed odds (post-race) | Not explicitly filtered |
| `hassotime` | Post time | Pre-race (race schedule) | Correctly used |

**Consequences:** A new feature built on what the developer thinks is pre-race data could silently contain post-race information.

**Prevention:**
- Before extracting any new column from EveryDB2, check `docs/everydb2/*.md` documentation
- Classify each new column as PRE_RACE or POST_RACE explicitly
- If a column's timing is uncertain, assume POST_RACE until proven otherwise
- Add new POST_RACE columns to `POST_RACE_COLS` in `domain/types.py` immediately
- Test: after adding new features, verify they produce NaN for future (unrun) races

**Detection:**
- New feature produces non-NaN values for horses that have not yet raced
- New feature correlates > 0.8 with any POST_RACE column
- Feature values change between "pre-race" and "post-race" snapshots of same race

**Phase to address:** Phase where EveryDB2 unused tables/columns are explored for new features.

---

## Moderate Pitfalls

### Pitfall 6: Feature Interaction Explosion

**What goes wrong:** Adding pairwise feature interactions (e.g., `feature_a * feature_b`) to a system with 100+ features creates thousands of new columns. Even selective interactions (only "meaningful" pairs) cause feature count to grow faster than the model's ability to distinguish signal from noise.

**Why it happens:** The existing `interaction_features.py` creates interactions like `kyakusitu_x_distance` (categorical product) and `weight_x_distance` (numeric product). Extending this pattern to more pairs creates an explosion. With N features, there are N*(N-1)/2 possible pairs.

**Prevention:**
- LightGBM, XGBoost, and CatBoost already capture feature interactions through tree structure. Do NOT create explicit interactions for pairs the model can learn naturally.
- Only create explicit interactions where the relationship is strongly non-linear AND domain knowledge confirms meaning (e.g., distance x surface is meaningful because turf/dirt have fundamentally different distance dynamics)
- Limit explicit interactions to at most 10-15 new features per phase
- Validate each interaction individually: add one, run ablation, confirm it helps

**Detection:**
- Feature count grows by more than 20% in a single phase
- New interaction features have near-zero permutation importance
- Model training time increases by more than 50% after adding interactions

### Pitfall 7: Feature Cache Invalidation After Schema Changes

**What goes wrong:** The feature engine uses content-hash-based cache (`compute_cache_key` in `feature_engine.py`). The cache key is computed from input file paths and date ranges, NOT from feature computation code. If feature logic changes without input changes, stale cached features are returned.

**Why it happens:** Modifying `horse_history_features.py` to compute a new feature does not change the input parquet file paths or date range, so `is_cache_valid()` returns True and old features (without the new one) are used.

**Prevention:**
- After ANY change to feature computation code, delete `data/features/cache/` before training
- Add the feature module source file hashes to cache key (engineering effort)
- Add a test: after adding a new feature, verify it appears in training DataFrame with non-NaN values

**Detection:**
- New feature column is all NaN after training
- Feature count in training data does not match expected count
- Backtest results are identical before and after feature changes

### Pitfall 8: Removing Features That Enable Downstream Interactions

**What goes wrong:** Feature audit identifies a feature with near-zero standalone importance and removes it. But the feature was an input to interaction features (e.g., kyakusitukubun_cd is used in kyakusitu_x_distance). Removing the base feature breaks the interaction.

**Why it happens:** Permutation importance measures standalone contribution, not contribution through interactions. A feature with zero direct importance may be critical when combined with another feature.

**Prevention:** Before removing any feature, check if it appears in `interaction_features.py`, `compute_intra_race_features()`, or is referenced by other computed features. Build a feature dependency graph. The `HorseHistoryFeatures.BASE_COLS` list partially documents this but is incomplete.

### Pitfall 9: Train/Test Feature Distribution Shift After Pruning

**What goes wrong:** Feature pruning changes model behavior on certain data subsets. Removing a feature primarily used for long-distance races may silently degrade long-distance predictions while improving overall metrics.

**Why it happens:** Aggregate metrics (overall ROI, overall AUC) hide segment-specific degradations. Horse racing has distinct segments: turf vs dirt, sprint vs long, high-grade vs low-grade.

**Prevention:** After feature pruning, evaluate performance by segment (surface, distance_bin, grade). Do not accept pruning if any major segment degrades by more than 5pp ROI.

### Pitfall 10: New Features With Sparse Coverage (High NaN Rate)

**What goes wrong:** New features from EveryDB2 unused tables may have high NaN rates (20-40%). Bloodline/pedigree features are missing for foreign-bred horses and NAR transfers. LightGBM may learn "has NaN" as a proxy for "low-quality horse" -- a real but non-generalizable signal.

**Prevention:** Compute NaN rate for any new feature before training. If NaN rate exceeds 30%, consider a binary "has_data" flag or hierarchical fallback (pattern already used for haron-time z-score in `horse_history_features.py`).

### Pitfall 11: Temporal Leakage in Expanding Window Features

**What goes wrong:** `info_asymmetry_features.py` correctly uses `expanding().mean().shift(1)`. New features computing historical statistics may forget the shift(1) or use wrong time boundaries.

**Why it happens:** The expanding window pattern requires careful handling. `expanding().mean()` at row i includes row i. To use only past data, shift(1) is required.

**Prevention:** Any new feature computing historical statistics MUST use `expanding().shift(1)` or `searchsorted(target_date, side="left")` (as in `horse_history_features.py`). Use `leakage_validators.py` to verify. Add test cases for every new historical feature.

### Pitfall 12: Grade/Class Encoding Duplication Across Modules

**What goes wrong:** `_GRADE_LEVEL_MAP` (A=8, B=7, C=6, D=5, E=4) is duplicated in three files: `feature_engine.py`, `horse_history_features.py`, and `high_odds_features.py`. Changes to one file without updating others create inconsistency.

**Prevention:** Centralize the mapping in `domain/types.py` alongside `POST_RACE_COLS`. Import from there in all feature modules.

---

## Minor Pitfalls

### Pitfall 13: Feature Naming Collisions

**What goes wrong:** A new feature module creates a column with the same name as an existing feature in a different module. The later module's `compute_*()` silently overwrites the earlier value.

**Prevention:** Maintain a central feature registry (dict of feature_name -> source_module). Check for collisions before adding new features.

### Pitfall 14: OOF Prediction Leakage Through Feature Cache

**What goes wrong:** The feature cache stores features computed on the full training set. During walk-forward validation, different folds use different train/test splits but may load the same cached features computed using future data.

**Prevention:** The cache key includes date range, partially handling this. Verify that fold-specific date ranges produce different cache keys during WF validation.

### Pitfall 15: Odds Snapshot Timing Assumptions

**What goes wrong:** `odds_dynamics_features.py` picks snapshots at t-10, t-30, t-60 minutes before post time with 15-20 minute tolerance. For races where final snapshot is at post time, the "t-10" snapshot may be confirmed odds.

**Prevention:** Verify JODDS DataKubun=3/4 (confirmed) snapshots have negative minutes_before_anchor and are not picked by `_pick_target_snapshot()`.

### Pitfall 16: Blacklist-Based Column Exclusion vs Whitelist

**What goes wrong:** The CQR model (spike M2) and training pipeline use blacklist-based feature selection: take all numeric columns except those in POST_RACE_COLS and _MODEL_OUTPUT_COLS. Any new column automatically becomes a feature, even if it should not be.

**Why it happens:** `_non_feature_cols` in training_pipeline.py (line 897-902) lists columns to exclude. Any column not listed becomes a feature by default.

**Prevention:** Consider moving toward explicit whitelist for at least the secondary models (CQR, EV correction). New columns added to the DataFrame should require explicit opt-in for model training, not automatic inclusion.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| POST_RACE safety gate | Pitfall 2: Implicit column leakage | Add explicit POST_RACE drop at build_all() exit; extend leakage_validators |
| Feature audit methodology | Pitfall 3: In-sample importance bias | Use permutation importance on held-out set; ablation testing |
| Feature pruning | Pitfall 8: Removing interaction enablers | Build feature dependency graph; check interaction_features.py |
| EveryDB2 new columns | Pitfall 5: Column semantics misunderstanding | Classify each column PRE/POST before use; check docs/everydb2/*.md |
| Feature interactions | Pitfall 6: Interaction explosion | Max 10-15 new interactions per phase; validate each individually |
| Historical statistics | Pitfall 11: Temporal leakage | Use expanding().shift(1) pattern; verify with leakage_validators |
| Bloodline/pedigree features | Pitfall 10: Sparse coverage | Check NaN rate; use hierarchical fallback for >30% NaN |
| Model retraining | Pitfall 7: Stale feature cache | Delete cache directory after any feature code change |
| Walk-forward validation | Pitfall 4: Insufficient validation rigor | Run 4-fold WF for every feature batch; require improvement on ALL folds |
| Segment evaluation | Pitfall 9: Distribution shift after pruning | Evaluate by surface, distance_bin, grade after each change |
| Cross-module consistency | Pitfall 12: Grade encoding duplication | Centralize constants in domain/types.py |
| Secondary model features | Pitfall 16: Blacklist vs whitelist | Move toward explicit feature whitelists for CQR/EV models |

---

## Warning Signs Checklist

After each feature engineering change, verify ALL of the following:

- [ ] Bet count >= 1000/year in backtest (if < 500, STOP and investigate overfitting)
- [ ] Win rate < 50% for win bets (if > 50%, likely leakage)
- [ ] Walk-forward ROI gap < 30pp between folds
- [ ] No POST_RACE_COLS in feature importance top 20
- [ ] New features have non-NaN values for at least 60% of samples
- [ ] Feature count changed as expected (check `len(feature_cols)`)
- [ ] Feature cache was cleared before training
- [ ] Permutation importance of new features > 0 on validation set
- [ ] No correlation > 0.8 between new features and existing features
- [ ] Performance by segment (turf/dirt, sprint/mile/long) did not degrade by > 5pp ROI
- [ ] No model output columns (ev_*, p_hit, edge_*) used as feature inputs
- [ ] `build_all()` output DataFrame does not contain POST_RACE_COLS (except target labels)

---

## Sources

### HIGH confidence (direct codebase analysis)
- `.planning/spikes/data-leak-phase-20-22.md` -- v1.5 data leak investigation (22-file audit, 3 parallel agents)
- CQR fix commit `f3a4c10` -- structural overfitting + selection bias root cause
- `src/domain/types.py` -- POST_RACE_COLS definition
- `src/features/feature_engine.py` -- main orchestrator, build_all(), cache mechanism
- `src/features/horse_history_features.py` -- 1324 lines, per-horse feature computation with PIT
- `src/features/leakage_validators.py` -- expanding window leak detection
- `src/features/interaction_features.py` -- explicit feature interactions
- `src/features/odds_dynamics_features.py` -- odds time series features
- `src/features/high_odds_features.py` -- 18 high-odds pattern features
- `src/features/info_asymmetry_features.py` -- expanding().shift(1) historical stats
- `src/features/market_bias_features.py` -- market distortion features
- `src/pipelines/training_pipeline.py` -- feature consumption, CQR feature selection (lines 897-907)
- `src/backtest/engine.py` -- POST_RACE drop before predict (lines 818-820)
- `src/models/conformal_ev_model.py` -- CQR blacklist approach (spike M2)

### MEDIUM confidence (general ML domain knowledge, verified with multiple sources)
- [Common Pitfalls in Feature Engineering -- Statsig](https://www.statsig.com/perspectives/feature-engineering-pitfalls)
- [Seven Common Causes of Data Leakage -- Towards Data Science](https://towardsdatascience.com/seven-common-causes-of-data-leakage-in-machine-learning-75f8a6243ea5/)
- [Data Leakage in Time-Dependent Feature Engineering -- BitPeak](https://bitpeak.com/data-leakage-in-time-dependent-feature-engineering/)
- [5 Critical Feature Engineering Mistakes -- KDnuggets](https://www.kdnuggets.com/5-critical-feature-engineering-mistakes-that-kill-machine-learning-projects)
- [Designing ML Systems -- Feature Engineering Summary](https://github.com/serodriguez68/designing-ml-systems-summary/blob/main/05-feature-engineering.md)

### HIGH confidence (official documentation, peer-reviewed)
- [Permutation Feature Importance -- scikit-learn](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Feature Importance in Gradient Boosting Trees -- PMC/NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC9140774/)

---

*Pitfalls research for: Feature Engineering Overhaul (v1.6)*
*Previous version: v1.4 Ensemble Filter Recalibration pitfalls (2026-05-05)*
*Ready for roadmap: yes*
