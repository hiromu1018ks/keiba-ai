# Technology Stack: Feature Engineering Overhaul (v1.6)

**Project:** keiba-ai v1.6 Feature Engineering Overhaul
**Researched:** 2026-05-10
**Scope:** Feature audit/pruning, new EveryDB2 feature extraction, feature interactions
**Supersedes:** v1.4 STACK.md (all prior dependencies remain valid)

## Verdict: Two Small Additions, Primarily In-House Code

The feature engineering overhaul needs exactly two lightweight additions: `scipy` (already installed as transitive dependency, just needs to be declared in pyproject.toml) for statistical tests during feature pruning, and no other new libraries. The project already has LightGBM's built-in TreeSHAP (`pred_contrib=True`), scikit-learn's `permutation_importance`, and all data access infrastructure. The work is 90% custom Python code against existing Parquet data.

**Key insight:** The EveryDB2 ETL already extracts 53+ tables to Parquet, but only ~10 are consumed by feature modules. The data is sitting on disk -- the engineering task is reading it and computing features, not installing new tools.

## Current Installed Stack

| Package | Installed Version | pyproject.toml Minimum | Status |
|---------|-------------------|----------------------|--------|
| Python | 3.11 | >=3.11 | Pinned via mise |
| LightGBM | 4.6.0 | >=4.3 | Up to date |
| XGBoost | 3.2.0 | >=2.0 | Up to date |
| CatBoost | 1.2.10 | >=1.2 | Up to date |
| scikit-learn | 1.8.0 | >=1.4 | Up to date |
| scipy | 1.17.1 | (transitive, not declared) | **Needs declaring** |
| pandas | 2.3.3 | >=2.2 | Up to date |
| numpy | 2.4.3 | >=1.26 | Up to date |
| pyarrow | 23.0.1 | >=14.0 | Up to date |
| mlflow | 3.10.1 | >=2.12 | Up to date |
| optuna | 4.8.0 | >=3.5 | Up to date |
| betacal | 1.0 | >=1.0 | Available |
| joblib | (installed, transitive) | (transitive) | Used for model persistence |

## Recommended Stack for v1.6

### Production Dependencies: One Declaration Change

| Technology | Version | Role in v1.6 | Why |
|------------|---------|-------------|-----|
| scipy | >=1.11 (installed: 1.17.1) | Feature importance significance testing, null importance distribution comparison | `scipy.stats.mannwhitneyu` for comparing actual vs null importance distributions. Already installed as sklearn transitive -- needs explicit declaration in pyproject.toml to prevent silent breakage if sklearn drops it. |
| scikit-learn `permutation_importance` | 1.8.0 (installed) | Model-agnostic feature importance on held-out data | Already available. Superior to built-in gain importance for feature auditing because it measures actual prediction impact, not just split frequency. Verified via Context7. |
| scikit-learn `mutual_info_classif` | 1.8.0 (installed) | Univariate feature relevance scoring before model training | Already available. Non-parametric measure of dependency between features and target. Catches non-linear relationships that correlation misses. |
| LightGBM `pred_contrib=True` | 4.6.0 (installed) | Native TreeSHAP values for feature importance analysis | Already used in `win_feature_analysis.py`. No external SHAP library needed -- LightGBM's built-in pred_contrib provides identical TreeSHAP values. |

### NOT Needed (Explicitly Rejected)

| Library | Purpose | Why Rejected |
|---------|---------|-------------|
| `shap` | SHAP value computation | LightGBM 4.6 provides native TreeSHAP via `pred_contrib=True`. The external `shap` library would add a heavy dependency (requires numpy, scipy, pandas, tqdm, packaging, scikit-learn, numba) for zero additional capability. The existing `win_feature_analysis.py` already uses this path successfully. |
| `boruta` / `boruta_py` | All-relevant feature selection | Boruta is designed for Random Forest estimators and requires wrapping sklearn-compatible estimators. Our 3-model stacking (LightGBM + XGBoost + CatBoost) with a Ridge meta-learner does not fit Boruta's sklearn API requirement. The null importance approach (target shuffling with our own models) is more appropriate and requires zero new dependencies. |
| `featuretools` | Automated feature engineering from relational data | Our data is already flat (single Parquet files), not truly relational. Featuretools' Deep Feature Synthesis (DFS) excels at multi-table joins, which we already handle via `ParquetStore.read()` + pandas `merge`. Adds 30+ transitive dependencies (including dask, distributed) for capability we already have. |
| `autofeat` | Automated feature generation and selection | Generates polynomial combinations of existing features. We need targeted domain-specific features from EveryDB2 tables, not brute-force polynomial expansion. Our 100+ features with ~500K rows would generate millions of polynomial combinations -- most meaningless in horse racing context. |
| `eli5` | Permutation importance visualization | `sklearn.inspection.permutation_importance` already provides the same computation. eli5 adds visualization for Jupyter, but this is a CLI backend system. |
| `statsmodels` | Statistical hypothesis testing | `scipy.stats` provides everything needed (Mann-Whitney U, Kolmogorov-Smirnov, rank correlation). statsmodels is 15MB+ for VIF, autocorrelation and other capabilities not relevant to feature selection. |
| `polars` | Faster DataFrame operations | All feature code is pandas-based with 500K-2M rows. This data size fits comfortably in memory with pandas. Rewriting 20+ feature modules for Polars is out of scope and unnecessary -- the bottleneck is model training (~17 min), not DataFrame operations. |
| `category_encoders` | Advanced categorical encoding | LightGBM/XGBoost/CatBoost handle categorical features natively. The existing code already uses `astype("category")` for categorical interactions. No additional encoding needed. |

## How Existing Tools Map to v1.6 Tasks

### Task 1: Feature Audit and Pruning (100+ features)

**Core approach:** Multi-method importance analysis with null importance testing.

**Tools used:** `lightgbm.Booster.predict(pred_contrib=True)`, `sklearn.inspection.permutation_importance`, `scipy.stats`, `numpy`, `pandas`

**Method pipeline:**
1. **Gain importance** (built-in, fast) -- `model.feature_importance(importance_type="gain")` -- already implemented in `win_feature_analysis.py`
2. **TreeSHAP** (built-in via pred_contrib) -- `model.predict(df, pred_contrib=True)` -- already implemented in `win_feature_analysis.py`
3. **Permutation importance** (sklearn) -- `permutation_importance(model, X_val, y_val, n_repeats=10)` -- measures actual prediction degradation when a feature is shuffled
4. **Null importance** (custom code) -- shuffle target, retrain, compute importance distributions, compare with actual using `scipy.stats.mannwhitneyu` -- identifies features that do not outperform random noise

**Null importance implementation (no new library needed):**
```python
import numpy as np
from scipy.stats import mannwhitneyu

def compute_null_importances(model, X, y, n_shuffles=50, importance_type="gain"):
    """Compare actual feature importance to null distribution."""
    actual_imp = model.feature_importance(importance_type=importance_type)
    null_imps = np.zeros((n_shuffles, len(actual_imp)))
    for i in range(n_shuffles):
        y_shuffled = np.random.permutation(y)
        # Re-fit or compute importance on shuffled target
        null_imps[i] = _compute_importance_with_target(model, X, y_shuffled)

    # Statistical test: does actual importance exceed null?
    scores = np.zeros(len(actual_imp))
    for j in range(len(actual_imp)):
        stat, pvalue = mannwhitneyu(
            [actual_imp[j]], null_imps[:, j], alternative="greater"
        )
        scores[j] = pvalue
    return scores  # Lower p-value = more confident feature is real
```

**Validation:** `win_feature_analysis.py` already has `validate_noise_removal()` that trains a new model on the pruned feature set and compares logloss/AUC. This is the guardrail against removing useful features.

### Task 2: New Features from Unused EveryDB2 Tables

**Data sources:** 40+ EveryDB2 tables already extracted to Parquet but never consumed by feature modules.

**Tools used:** `ParquetStore.read()`, `pandas.merge/groupby`, existing `readers.py` pattern, `numpy`

**Key unused tables and their feature potential:**

| EveryDB2 Table | Parquet Key | Potential Features | Category |
|----------------|-------------|-------------------|----------|
| `n_hansyoku` | `hansyoku` | Dam bloodline stats, siblings' performance | Pedigree |
| `n_bameiorigin` | `bameiorigin` | Detailed pedigree tree (up to 5 generations), inbreeding coefficients | Pedigree |
| `n_banusi` | `banusi` | Owner type (individual/corporate/racing club), owner win rate | Ownership |
| `n_seisan` | `seisan` | Breeder stats, farm-level performance metrics | Breeding |
| `n_mining` | `mining` | JRA official data mining per-horse stats (speed index, stamina index, etc.) | Performance |
| `n_taisengata_mining` | `taisengata_mining` | Pairwise comparison mining data (horse-vs-horse) | Comparison |
| `n_toku_race` / `n_toku` | `toku_race` / `toku` | Special race conditions, weight allowance details | Race Context |
| `n_jogaiba` | `jogaiba` | Rider change history, jockey switch indicators | Jockey Context |
| `n_hanro` | `hanro` | Course record times per venue/distance/surface | Course |
| `n_record` | `record` | Track records (best times per venue/distance/surface) | Course |
| `n_schedule` | `schedule` | Meeting schedule gaps, rest days between meetings | Schedule |
| `n_wood_chip` | `wood_chip` | Wood chip training data | Training |
| `n_sale` | `sale` | Auction/sale history, purchase price | Market |
| `n_hyosu*` (6 tables) | Various | Vote counts per bet type -- market conviction indicators | Market |
| `n_jyusyosiki*` | Various | Payout structure details | Payout |

**Implementation pattern:** Follow existing `BloodlineFeatures`, `JockeyContextFeatures`, `TrainerContextFeatures` pattern:
1. Create new reader in `readers.py` (e.g., `load_mining(store)`)
2. Create new feature module in `src/features/` (e.g., `mining_features.py`)
3. Compute features with point-in-time safety (use `race_date` filtering to prevent look-ahead)
4. Integrate into `FeatureEngine.build_all()` pipeline

**Point-in-time safety pattern (already established):**
```python
# PIT safety: only use data from before the current race
mask = stats_df["race_date"] < entry_df["race_date"]
past_stats = stats_df[mask]
```

### Task 3: Feature Interactions and Transformations

**Tools used:** `pandas` (vectorized operations), `numpy`, existing `interaction_features.py` pattern

**Types of interactions to add:**

| Interaction Type | Example | Implementation |
|------------------|---------|----------------|
| Horse-vs-horse comparison | `ev_ratio_vs_fav` (horse EV / favorite EV per race) | `groupby("race_id").transform()` with rank operations |
| Conditional interactions | `blood_wr_x_distance_bin` (bloodline WR * distance suitability) | String concatenation + `astype("category")`, same as existing `kyakusitu_x_distance` |
| Numeric cross-products | `jockey_wr_x_trainer_wr` (team quality composite) | Vectorized multiplication with NaN propagation |
| Ratio features | `prize_per_start` (cumulative prize / starts) | Safe division with `np.where(denom > 0, ...) |
| Ranking within group | `ev_rank_in_race` (relative EV positioning) | `groupby("race_id").rank()` |

**No new library needed.** All interactions are expressible with pandas vectorized operations. The existing `interaction_features.py` already demonstrates the pattern for categorical products and numeric cross-products.

**Horse-vs-horse comparison (new capability, pure pandas):**
```python
def compute_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Within-race relative positioning features."""
    df = df.copy()

    # EV ratio vs race favorite
    if "ev_win" in df.columns:
        fav_ev = df.groupby("race_id")["ev_win"].transform("max")
        df["ev_ratio_vs_fav"] = np.where(fav_ev > 0, df["ev_win"] / fav_ev, np.nan)

    # Jockey-trainer composite quality rank
    if "jockey_wr_overall" in df.columns and "trainer_wr_overall" in df.columns:
        df["team_quality"] = (
            df["jockey_wr_overall"].fillna(0) * 0.6
            + df["trainer_wr_overall"].fillna(0) * 0.4
        )
        df["team_quality_rank"] = df.groupby("race_id")["team_quality"].rank(
            ascending=False
        )

    return df
```

## Installation

```bash
# No new packages needed. Only declare scipy explicitly:
# In pyproject.toml, add to dependencies:
#   "scipy>=1.11"
# Then:
pip install -e ".[dev]"
```

## Integration Points Summary

| Integration Point | File | Change Type | New Dependency? |
|-------------------|------|-------------|-----------------|
| Feature importance audit | `src/features/win_feature_analysis.py` | Add permutation importance + null importance | No |
| New reader functions | `src/db/readers.py` | Add `load_mining()`, `load_hansyoku()`, etc. | No |
| New feature modules | `src/features/` (new files) | `mining_features.py`, `pedigree_features.py`, etc. | No |
| Feature engine integration | `src/features/feature_engine.py` | Call new feature modules in `build_all()` | No |
| Interaction features | `src/features/interaction_features.py` | Add horse-vs-horse comparisons | No |
| scipy declaration | `pyproject.toml` | Add `"scipy>=1.11"` to dependencies | Declaration only |

## Key Design Decision: Why Custom Code Over Libraries

The v1.6 milestone is fundamentally a **data mining and feature design** task. The three sub-tasks require:

1. **Feature audit:** Statistical comparison of importance distributions -- `scipy.stats` (already installed) provides the tests. No library automates the "should this feature be removed?" decision better than a human looking at gain + SHAP + permutation + null importance together.

2. **New features from EveryDB2:** The data is already in Parquet files on disk. The task is reading those files and computing domain-specific features (speed indices, pedigree coefficients, trainer specialization metrics). No library understands JRA horse racing semantics.

3. **Feature interactions:** All interactions are either categorical products (already implemented), numeric cross-products (already implemented), or within-group rankings (standard pandas groupby). Adding a library for `df["a"] * df["b"]` would be absurd.

## Feature Importance Methods: Capability Matrix

| Method | What It Measures | Already Available? | Speed | Use in v1.6 |
|--------|-----------------|-------------------|-------|-------------|
| Gain importance | Total information gain from splits | YES (`model.feature_importance("gain")`) | Fast | Quick filter |
| TreeSHAP | Marginal contribution per feature per sample | YES (`pred_contrib=True`) | Medium | Detailed analysis |
| Permutation importance | Prediction degradation when feature is shuffled | YES (`sklearn.inspection.permutation_importance`) | Slow | Validation |
| Null importance | Whether feature beats random noise | CUSTOM CODE (numpy + scipy.stats) | Very slow | Final arbiter |
| Mutual information | Non-parametric feature-target dependency | YES (`sklearn.feature_selection.mutual_info_classif`) | Medium | Pre-model screening |

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| No new libraries needed for feature audit | HIGH | `win_feature_analysis.py` already implements gain + TreeSHAP analysis. `sklearn.inspection.permutation_importance` confirmed available in installed sklearn 1.8.0 via Context7 docs. `scipy.stats.mannwhitneyu` confirmed available in installed scipy 1.17.1. Null importance is a standard pattern requiring only numpy + scipy. |
| EveryDB2 unused tables accessible | HIGH | ETL config (`config/etl_tables.yaml`) extracts 53 tables to Parquet. `ParquetStore.read("raw", "mining")` pattern works for any table. `readers.py` pattern is well-established. Data is on disk. |
| Feature interactions with pure pandas | HIGH | Existing `interaction_features.py` demonstrates the pattern. All proposed interactions are standard pandas operations (groupby, rank, multiply). No edge cases. |
| scipy should be declared in pyproject.toml | HIGH | scipy is currently an undeclared transitive dependency via sklearn. If sklearn ever drops or changes its scipy requirement, keiba-ai will break. Declaring it explicitly is standard Python packaging practice. |
| Null importance approach sound for feature pruning | MEDIUM | Standard Kaggle competition technique with proven track record. However, the 50-iteration shuffle + retrain cycle is computationally expensive (~50x model training time). May need to use gain-based null importance (no retrain) instead of full retrain. |
| Mining table data quality | LOW | `n_mining` and `n_taisengata_mining` are JRA-provided analytics tables. Column names and semantics are unknown without querying the actual EveryDB2 instance. The ETL extracts them, but feature design requires inspecting actual data. This is the biggest research gap. |

## Gaps Requiring Phase-Specific Research

| Gap | Risk | When to Address |
|-----|------|-----------------|
| EveryDB2 table column inspection | HIGH -- cannot design features without knowing column names/types | Phase 1 of v1.6: Query EveryDB2 to list columns for `n_mining`, `n_hansyoku`, `n_bameiorigin`, `n_banusi`, etc. |
| Null importance computational cost | MEDIUM -- 50 iterations of full model training may take 14+ hours | Phase 1 of v1.6: Benchmark single iteration, decide on gain-based vs retrain-based null importance |
| `n_taisengata_mining` structure | MEDIUM -- horse-vs-horse pairwise data structure unknown | Phase 1 of v1.6: Inspect Parquet file columns |
| Feature cache invalidation | LOW -- adding new features changes cache key computation | Phase implementing new features: Cache key already includes feature_type, but new modules need integration |
| Correlation analysis for redundant features | LOW -- scipy/numpy have all needed tools | Phase 2 of v1.6: After computing new features, run correlation matrix analysis |

## Sources

- LightGBM 4.6.0 installed, `pred_contrib=True` TreeSHAP verified in `src/features/win_feature_analysis.py` lines 46-48 (HIGH confidence)
- scikit-learn 1.8.0 installed, `permutation_importance` API verified via Context7 docs from `scikit-learn/scikit-learn` repository (HIGH confidence)
- scipy 1.17.1 installed, `mannwhitneyu` available as standard scipy.stats function (HIGH confidence)
- EveryDB2 ETL config `config/etl_tables.yaml` defines 53 tables with Parquet extraction paths (HIGH confidence)
- Feature module pattern established by 22 existing feature files in `src/features/` (HIGH confidence)
- Null importance technique: Olivier Grellier's Kaggle notebook "Feature Selection with Null Importances" -- standard competition technique (MEDIUM confidence for applicability to this dataset size)
- Boruta rejected: scikit-learn-contrib/boruta_py requires sklearn-compatible estimator, incompatible with 3-model stacking (HIGH confidence in rejection)
- SHAP library rejected: LightGBM native TreeSHAP provides identical capability at zero dependency cost (HIGH confidence in rejection)
