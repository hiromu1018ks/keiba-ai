# Technology Stack -- v1.1 ROI Advanced Model

**Project:** keiba-ai v1.1 milestone -- multi-model stacking, odds deviation EV, time-series features
**Researched:** 2026-05-03
**Scope:** Stack additions/changes for the 5 new features (stacking, odds deviation, odds time-series, past-run time-series, pace prediction)

## Current Installed Stack

| Package | Installed Version | pyproject.toml Minimum | Status |
|---------|-------------------|----------------------|--------|
| Python | 3.11 | >=3.11 | Pinned via mise |
| LightGBM | 4.6.0 | >=4.3 | Up to date |
| XGBoost | 3.2.0 | >=2.0 | Up to date |
| CatBoost | 1.2.10 | >=1.2 | Up to date |
| scikit-learn | 1.8.0 | >=1.4 | Up to date |
| scipy | 1.17.1 | (transitive) | Up to date |
| pandas | 2.3.3 | >=2.2 | Up to date |
| numpy | 2.4.3 | >=1.26 | Up to date |
| pyarrow | (installed) | >=14.0 | Up to date |
| mlflow | 3.10.1 | >=2.12 | Up to date |
| optuna | 4.8.0 | >=3.5 | Up to date |

**Key finding:** ALL required libraries are already installed at current stable versions. No new package installations are needed for this milestone.

## Recommended Stack (No Changes Needed)

### Core Gradient Boosting (Already Installed)

| Technology | Version | Purpose | Why This Choice |
|------------|---------|---------|-----------------|
| LightGBM | 4.6.0 | Stage1 ranker + primary binary model | Native categorical support, fastest training, leaf-wise growth. Already used as `lgb.Booster` via `lgb.train()` native API. LambdaRank objective for Stage1, binary objective for Stage2. |
| XGBoost | 3.2.0 | Secondary base model in stack | Complementary tree growth (depth-wise/level-wise vs LightGBM's leaf-wise). Different regularization defaults. Uses `xgb.train()` with `DMatrix` in existing code -- correct choice for low-level control. |
| CatBoost | 1.2.10 | Tertiary base model in stack | Symmetric tree growth creates fundamentally different decision boundaries. Best default performance with minimal tuning. Native categorical handling via `CatBoostClassifier`. Ordered boosting reduces overfitting on small datasets. |

**Confidence: HIGH** -- All three are installed, the existing `StackedEnsemble` class already demonstrates the integration pattern.

### Meta-Learner (Already Installed)

| Technology | Version | Purpose | Why This Choice |
|------------|---------|---------|-----------------|
| scikit-learn `Ridge` | 1.8.0 | Level-2 stacking meta-learner | L2 regularization prevents overfitting on correlated base model outputs. Alpha=1.0 default is appropriate. Lightweight and fast. Alternative: LogisticRegression with L2 penalty, but Ridge on raw probabilities is simpler and works equally well for stacking regression. |

**Confidence: HIGH** -- Already in use in `StackedEnsemble`. Do NOT replace with a more complex meta-learner (neural net, GBM meta) -- the base model predictions are highly correlated and a simple regularized linear model is theoretically optimal for this scenario.

### Feature Engineering (Already Installed)

| Technology | Version | Purpose | Why This Choice |
|------------|---------|---------|-----------------|
| pandas | 2.3.3 | Time-series feature computation | Rolling windows, expanding stats, groupby operations. All new odds/time-series features can be built with pandas alone. |
| numpy | 2.4.3 | Vectorized computation | `searchsorted` for PIT-safe expanding stats (already pattern in `HorseHistoryFeatures`). Linear algebra for velocity/regression features. |
| scipy | 1.17.1 | Statistical functions | `scipy.special.softmax` if needed for race-level normalization. `scipy.stats` for z-score and percentile calculations. |

**Confidence: HIGH** -- All new feature engineering is pandas/numpy operations. No specialized time-series library needed.

## What NOT to Add

| Rejected Library | Why Rejected | What to Use Instead |
|------------------|-------------|-------------------|
| `sktime` / `tsfresh` / `darts` | Overkill for the time-series patterns needed. Horse racing time-series are short (5-10 past runs per horse, ~60 odds snapshots per race). These libraries target longer sequences. Adding them introduces 50+ transitive dependencies. | Direct pandas rolling/expanding operations (already the project pattern in `HorseHistoryFeatures` and `PaceAptitudeFeatures`). |
| `sklearn.ensemble.StackingClassifier` | Requires sklearn-compatible estimators (`fit/predict/get_params`). The project uses native APIs: `lgb.Booster`, `xgb.Booster`, `CatBoostClassifier`. Converting to sklearn wrappers (`LGBMClassifier`, `XGBClassifier`) loses control over `Dataset`/`DMatrix` construction and group handling for ranking. The existing custom `StackedEnsemble` class is the right approach. | Existing `StackedEnsemble` class in `src/models/stacked_ensemble.py` with manual OOF + Ridge meta-learner. |
| `XGBRanker` / `CatBoostRanker` for Stage1 | The Stage1 ability model uses LightGBM LambdaRank (`lambdarank` objective) which is the industry standard for learning-to-rank. Adding XGBoost/CatBoost rankers at Stage1 would triple the Stage1 training time for marginal gain. The diversity benefit comes from using different boosting algorithms at Stage2 (binary classification), where the label space is simpler. | Keep LightGBM Ranker at Stage1. Use multi-model stacking only at Stage2 (binary hit prediction). |
| `optuna-integration` / `optuna-xgboost` | The existing `optuna` 4.8.0 has built-in XGBoost/CatBoost integration via `optuna.integration`. No extra packages needed. | `optuna.integration.XGBoostPruningCallback`, `optuna.integration.LightGBMPruningCallback`. |
| `polars` | Faster than pandas for some operations, but the entire codebase is pandas-based. Introducing polars would create dual-Df confusion, require conversion layers, and break the established pattern. | pandas (existing). Performance for the data sizes involved (~100K rows) is adequate. |
| `statsmodels` | Has SARIMAX and other time-series models, but the project does not need forecasting models. The time-series features are descriptive statistics (trends, velocities, volatility), not predictive time-series models. | numpy/pandas descriptive statistics. |

## Integration Points with Existing Pipeline

### Stacked Ensemble Integration

The existing `StackedEnsemble` class (`src/models/stacked_ensemble.py`) is the foundation. Key integration points:

**1. Where stacking plugs in:** The `StackedEnsemble` is designed as a drop-in replacement for `lgb.Booster` (it has `best_iteration=0` and `predict(X)` returning ndarray). It currently integrates into `WinTwoStageModel.hit_model`.

**2. What needs improvement in the existing code:**

| Current Code | Issue | Fix Needed |
|-------------|-------|------------|
| `_train_xgb_fold()` uses `xgb.train()` with `DMatrix` | Correct low-level API usage. No issue. | Consider adding `early_stopping_rounds` for fold models. |
| `_train_cat_fold()` uses `CatBoostClassifier` | Correct. But uses hardcoded `iterations=300`. | Make iterations/depth/learning_rate configurable. |
| `_encode_cats()` converts categoricals to integer codes | Works but loses CatBoost's native categorical advantage. | Consider passing `cat_features` parameter to CatBoost Pool for native handling. |
| OOF fold split uses expanding window | Correct for time-series. But `n_folds=3` is too few. | Increase to 5 folds for better meta-learner generalization. |
| No hyperparameter tuning | All three models use hardcoded params (lr=0.03, leaves/depth=31/6, rounds=300). | Add Optuna tuning per model. |
| Ridge `alpha=1.0` is hardcoded | May not be optimal. | Tune alpha with cross-validation on OOF predictions. |

**3. API compatibility concern with XGBoost 3.x:**

XGBoost 3.0 introduced breaking changes. The existing code uses:
```python
xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=300)
```
This native API is unchanged in 3.x and is the correct approach. The sklearn API (`XGBClassifier.fit()`) had the data parameter renamed from `data` to `X`, but the code does not use the sklearn API for XGBoost. **No migration needed.**

### Odds Time-Series Feature Integration

**Data source:** `data/odds/time_series/` (year/month partitioned Parquet) contains `jodds_tanpuku` with columns including `race_id`, `umaban`, `happyotime`, `tanodds`, `tanninki`.

**Existing feature module:** `src/features/odds_dynamics_features.py` already computes:
- `odds_drop_rate_60_10`, `odds_drop_rate_30_10`
- `odds_velocity` (linear regression slope)
- `odds_volatility` (std of consecutive diffs)
- `popularity_change_30_10`

**New features to add (same module or new module):**

| Feature | Computation | Required Stack |
|---------|-------------|----------------|
| Odds curvature (acceleration) | Second derivative of odds over time | numpy polynomial fit or manual finite differences |
| Late money indicator | Binary: odds drop > threshold in last 10 min | pandas comparison |
| Odds regime (3-class) | KMeans or rule-based on velocity+volatility | numpy or `sklearn.cluster.KMeans` (already available) |
| Deviation from closing odds | `model_prob * odds - 1` as EV proxy | numpy arithmetic |

All computable with existing pandas/numpy/scikit-learn. No new packages.

### Past-Run Time-Series Feature Integration

**Data source:** `data/raw/races.parquet` + `data/raw/entries.parquet` via `ParquetStore` -> `HorseHistoryFeatures`.

**Existing pattern:** `HorseHistoryFeatures` already processes past runs with PIT-safe `searchsorted` on `race_date`. The `expanding_stats` dictionary pattern (pre-computed cumulative mean/std per group) is the proven approach for leak-free z-scores.

**New features to add:**

| Feature | Computation | Required Stack |
|---------|-------------|----------------|
| Time progression trend | Linear regression of `harontimel3` over last 5 runs | numpy `polyfit` or manual slope |
| Late surge change | Delta of `closing_index` between recent 3 vs earlier 3 runs | numpy subtraction |
| Speed figure trend | `timediff` regression slope | numpy `polyfit` |
| Position pattern stability | Std of `jyuni1c` over last 5 runs | numpy `std` |
| Form cycle phase | Already exists as `form_trend`, `form_consistency`, `form_peak_flag` | existing |

All computable with existing numpy/pandas. The `searchsorted` + cumulative sum pattern from `PaceAptitudeFeatures.compute_batch()` is the reference implementation for vectorized computation.

### Pace Prediction Feature Integration

**Existing module:** `src/features/pace_aptitude_features.py` computes `pace_aptitude`, `front_pace_wr`, `closing_pace_wr`.

**New features to add:**

| Feature | Computation | Required Stack |
|---------|-------------|----------------|
| Predicted pace scenario | `sum(1/jyuni1c_avg)` across race entries as proxy for pace pressure | pandas groupby + numpy |
| Pace-pressure-fit | Interaction: horse's `closing_pace_wr` * predicted pace speed | numpy multiplication |
| Position prediction | Expected running position from `jyuni1c_avg` normalized by field | numpy arithmetic |
| Pace volatility | Std of position changes across past runs | numpy `std` |

All computable with existing stack. No new packages.

## Version Pinning Recommendations

The installed versions are all current and stable. Recommend updating `pyproject.toml` minimums:

```toml
# CURRENT pyproject.toml            # RECOMMENDED update
"lightgbm>=4.3",     -->    "lightgbm>=4.6",
"xgboost>=2.0",      -->    "xgboost>=3.0",
"catboost>=1.2",     -->    "catboost>=1.2.5",
"scikit-learn>=1.4",  -->    "scikit-learn>=1.6",
"optuna>=3.5",        -->    "optuna>=4.0",
```

**Rationale:**
- XGBoost >=3.0: The existing code uses native `xgb.train()` API which is stable across 3.x. Pinning to >=3.0 ensures access to XGBRanker improvements and `lambdarank_num_pair_per_sample` parameter.
- LightGBM >=4.6: CVE-2024-43598 (RCE vulnerability) fix. Improved categorical handling.
- CatBoost >=1.2.5: Bug fixes, stability improvements. 1.2.10 is current.
- scikit-learn >=1.6: `FrozenEstimator` for clean prefit model wrapping. `Array API` support.
- optuna >=4.0: API stability, improved samplers for hyperparameter tuning of ensemble.

**Do NOT bump Python.** Pin at 3.11 via `mise.toml`. All dependencies are compatible.

## Installation (Minimal)

```bash
# No new packages needed -- all dependencies already installed
# Only update pyproject.toml minimums if desired

# Optional: verify all packages at correct versions
pip install -e ".[dev]"
```

## Key Design Decision: Custom Stacking vs sklearn StackingClassifier

**Decision: Keep custom `StackedEnsemble` (manual OOF + Ridge). Do NOT switch to `sklearn.ensemble.StackingClassifier`.**

**Rationale:**

1. **Ranker incompatibility:** The project uses `lgb.Booster` (from `lgb.train()`, not `LGBMClassifier`). The native API gives control over `lgb.Dataset` construction, group specification for LambdaRank, and `init_score` for the binary P-correction model. Switching to sklearn wrappers (`LGBMClassifier`) would lose this control.

2. **Time-series OOF:** The existing code uses expanding-window folds (time-ordered, not random). `StackingClassifier` uses `KFold` or `StratifiedKFold` by default, which would violate the PIT (point-in-time) principle and introduce look-ahead bias.

3. **XGBoost native API:** The existing code uses `xgb.train()` with `DMatrix`, which is more memory-efficient and gives access to all parameters. The sklearn `XGBClassifier` wrapper would require converting the data format.

4. **CatBoost native categorical handling:** `CatBoostClassifier` already follows sklearn API, so it could technically work with `StackingClassifier`. But for consistency across the three models, keeping the custom stacking is simpler.

5. **Proven pattern:** The existing `StackedEnsemble` class is already tested and follows the Benter (1994) approach: OOF predictions as Level-2 features, Ridge regression as meta-learner. This is the standard approach in horse racing prediction literature.

## Stacking Architecture Detail

```
                    Feature Matrix X
                         |
            +------------+------------+
            |            |            |
       LightGBM      XGBoost     CatBoost
       (binary)      (binary)    (binary)
            |            |            |
       p_lgbm        p_xgb       p_cat
            |            |            |
            +------------+------------+
                         |
                    Ridge(alpha)
                         |
                   p_stacked
```

**Training flow (existing, with improvements noted):**

1. K-fold expanding window (change from 3 to 5 folds)
2. Each fold: train 3 models, predict on validation fold -> OOF predictions
3. Stack OOF predictions: `[p_lgbm, p_xgb, p_cat]` as 3-column matrix
4. Train Ridge on OOF predictions
5. Retrain all 3 base models on full data (train + valid combined)
6. At inference: 3 base models predict -> Ridge combines -> final probability

**Per-model feature handling:**

| Model | Categorical Handling | Training API | Key Params to Tune |
|-------|---------------------|-------------|-------------------|
| LightGBM | Native `category` dtype | `lgb.train()` with `lgb.Dataset` | `num_leaves`, `feature_fraction`, `min_data_in_leaf` |
| XGBoost | Manual integer encoding via `_encode_cats()` | `xgb.train()` with `xgb.DMatrix` | `max_depth`, `subsample`, `colsample_bytree` |
| CatBoost | `cat_features` param to Pool (currently NOT used -- improvement opportunity) | `CatBoostClassifier.fit()` | `depth`, `l2_leaf_reg`, `random_strength` |

## Odds Deviation EV Calculation

No new library needed. The calculation is:

```python
# Model probability vs market implied probability
p_model = p_stacked  # from ensemble
p_market = 1.0 / tanodds  # implied probability
ev_deviation = (p_model * tanodds - 1.0)  # expected value from deviation

# Or equivalently:
ev_deviation = p_model / p_market - 1.0  # edge ratio
```

This is pure numpy arithmetic. The `MarketModel` already computes `market_log_error_win` which captures a similar signal from a different angle. The odds deviation EV adds a direct EV measure rather than an error signal.

## Sources

- [XGBoost 3.2.0 Release Notes](https://github.com/dmlc/xgboost/releases) -- latest stable, API changes from 3.0
- [CatBoost 1.2.8 Release](https://github.com/catboost/catboost/releases) -- latest stable (1.2.10 on PyPI)
- [LightGBM 4.6.0](https://pypi.org/project/lightgbm/) -- latest stable, CVE fix
- [scikit-learn 1.8.0 StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) -- why NOT to use it for this project
- [XGBoost Learning to Rank](https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html) -- XGBRanker API with qid/group
- [CatBoost CatBoostRanker](https://catboost.ai/docs/en/concepts/python-reference_catboostranker) -- why NOT to use for Stage1
- [Stacking Ensembles: XGBoost, LightGBM, CatBoost (Medium)](https://medium.com/@stevechesa/stacking-ensembles-combining-xgboost-lightgbm-and-catboost-to-improve-model-performance-d4247d092c2e) -- stacking best practices
- [Ensemble caution: soft voting can degrade performance (PMC/NIH)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13075335/) -- validation that stacking needs careful tuning
- [Horse Racing Prediction with Ensemble ML (Medium)](https://medium.com/@cagdasgul/high-precision-prediction-of-horse-racing-durations-using-ensemble-machine-learning-models-a-d6af16a1ebf1) -- XGBoost+LightGBM best trade-off for horse racing
- [stackgbm: Model Stacking for Boosted Trees](https://nanx.me/stackgbm/articles/stackgbm.html) -- per-model categorical encoding best practices
