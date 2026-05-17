# Technology Stack: Market-Independent Edge Discovery (v1.7)

**Project:** keiba-ai v1.7 Market-Independent Edge Discovery
**Researched:** 2026-05-17
**Scope:** Race-level aggregation features, market cross-consistency features, Gain per Depth diagnostic, Residual IC evaluation
**Supersedes:** v1.6 STACK.md (all prior dependencies remain valid)

## Verdict: Zero New Dependencies -- All Tools Already Installed

Every capability required for v1.7 is already available in the installed stack. The four feature areas are:
1. **Race-level aggregation features** -- pure pandas `groupby("race_id")` with scipy.stats entropy utilities
2. **Market cross-consistency features** -- requires multi-bet-type odds data already in Parquet (needs ETL extraction for umaren/sanren tables)
3. **Gain per Depth** -- LightGBM 4.6.0 `trees_to_dataframe()` with all required columns (tree_index, node_depth, split_gain)
4. **Residual IC** -- scipy.stats.spearmanr + sklearn LinearRegression for orthogonalization, both installed

The only non-code action is running ETL for odds_umaren and odds_sanren Parquet files.

**Key insight:** The ETL config (`config/etl_tables.yaml`) already defines odds_umaren, odds_umatan, odds_sanren, odds_sanrentan, odds_waku tables with n_ + s_ source pairs. The schema is known. The data is in EveryDB2. It just needs a `run_etl.py --mode full` invocation with these tables specified.

## Current Installed Stack (Unchanged)

| Package | Installed Version | pyproject.toml Minimum | Status |
|---------|-------------------|----------------------|--------|
| Python | 3.11 | >=3.11 | Pinned via mise |
| LightGBM | 4.6.0 | >=4.3 | Up to date |
| XGBoost | 3.2.0 | >=2.0 | Up to date |
| CatBoost | 1.2.10 | >=1.2 | Up to date |
| scikit-learn | 1.8.0 | >=1.4 | Up to date |
| scipy | 1.17.1 | >=1.11 (declared in v1.6) | Up to date |
| pandas | 2.3.3 | >=2.2 | Up to date |
| numpy | 2.4.3 | >=1.26 | Up to date |
| pyarrow | 23.0.1 | >=14.0 | Up to date |
| mlflow | 3.10.1 | >=2.12 | Up to date |
| optuna | 4.8.0 | >=3.5 | Up to date |

## Recommended Stack for v1.7

### Production Dependencies: No New Packages

| Technology | Version | Role in v1.7 | Why |
|------------|---------|-------------|-----|
| pandas `groupby` + `agg` | 2.3.3 (installed) | Race-level aggregation features (entropy, dispersion, top-k gap) | Already the foundation for all existing feature modules. `groupby("race_id")` with `transform()` for broadcast or `agg()` for per-race features is the standard pattern in `market_bias_features.py`, `intra_race_features.py`, etc. |
| scipy.stats `entropy` | 1.17.1 (installed) | Shannon entropy computation for `rl_log_odds_entropy` | `scipy.stats.entropy(pk)` computes `-sum(pk * log(pk))` directly, more numerically stable than manual implementation. Current `market_bias_features.py` implements entropy manually -- new race-level module should use scipy for consistency. |
| scipy.stats `spearmanr` | 1.17.1 (installed) | Residual IC computation (B-diff, C-orthogonal, E-incremental) | Spearman rank correlation is the standard IC metric in quantitative finance. Already confirmed working. Returns (correlation, p-value) -- the correlation coefficient is the IC. |
| sklearn `LinearRegression` | 1.8.0 (installed) | Orthogonalization for C-orthogonal IC | `LinearRegression().fit(p_market, p_model)` then `residual = p_model - lr.predict(p_market)`. Standard partial correlation approach. More robust than `numpy.polyfit` for multi-dimensional orthogonalization (handles multiple market signals). |
| LightGBM `trees_to_dataframe()` | 4.6.0 (installed) | Gain per Depth diagnostic | Returns DataFrame with `tree_index`, `node_depth`, `split_gain` columns. Confirmed via official docs: 15 columns including all needed for depth-stratified gain analysis. |

### Data Requirements (ETL Action Required)

| Parquet File | ETL Source | Status | Columns (confirmed) | Needed For |
|-------------|-----------|--------|---------------------|------------|
| `data/odds/odds_umaren.parquet` | `n_odds_umaren` | **MISSING -- needs ETL** | year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, odds, ninki | Market cross-consistency (win x quinella) |
| `data/odds/odds_sanren.parquet` | `n_odds_sanren` | **MISSING -- needs ETL** | year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, umaban3, odds, ninki | Market cross-consistency (win x trio) |
| `data/odds/odds_umatan.parquet` | `n_odds_umatan` | **MISSING -- needs ETL** | year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, odds, ninki | Market cross-consistency (win x exacta) |
| `data/odds/odds_tanpuku.parquet` | `n_odds_tanpuku` | EXISTS (3.5 MB) | race_id, umaban, tanodds, fukuoddslow, fukuoddshigh, tanninki | Already consumed |
| `data/odds/odds_wide.parquet` | `n_odds_wide` | EXISTS (17.9 MB) | race_id, kumi, oddslow, oddshigh, ninki | Already consumed |
| `data/odds/odds_waku.parquet` | `n_odds_waku` | EXISTS (3.8 MB) | race_id, kumi, odds, ninki | Bracket quinella consistency |
| `data/raw/payouts.parquet` | `n_harai` | EXISTS (38,835 rows, 201 cols) | Contains payumarenkumi, paysanrenpukukumi, paysanrentankumi etc. | Cross-validation of odds consistency |

**ETL command to extract missing tables:**
```bash
python scripts/run_etl.py --mode full --start 20140101 --end 20251231 \
    --tables n_odds_umaren n_odds_umatan n_odds_sanren n_odds_sanrentan
```

**Note:** The payouts table already contains all bet-type payout data (201 columns including umaren, umatan, sanren, sanrentan, wide, waku pay columns). This is sufficient for post-race cross-consistency validation even before ETL of the odds tables.

### NOT Needed (Explicitly Rejected)

| Library | Purpose | Why Rejected |
|---------|---------|-------------|
| `pingouin` | Partial correlation computation | Provides `pg.partial_corr()` but adds 10+ dependencies (pandas-stubs, tabulate). `sklearn.LinearRegression` residual approach is mathematically identical and already installed. Partial correlation IC = Spearman(y, residual(p_model | p_market)). |
| `statsmodels` | OLS regression for orthogonalization | `statsmodels.api.OLS` provides richer regression diagnostics but is overkill here. We need the residual vector, not t-statistics or confidence intervals. `sklearn.LinearRegression` gives identical residuals at zero new dependency. |
| `shap` | Tree structure explainability for Gain per Depth | `trees_to_dataframe()` already exposes split_gain per node. SHAP values measure contribution per sample, not per tree depth. The Gain per Depth diagnostic needs node-level gain values, not sample-level SHAP contributions -- these are fundamentally different things. |
| `dtreeviz` | Decision tree visualization | Provides beautiful tree diagrams but is a visualization tool, not a diagnostic tool. Gain per Depth analysis is numerical (aggregate gain by depth level), not visual. The output is a table or chart of `depth -> cumulative_gain_fraction`, easily produced with pandas. |
| `graphviz` | Tree structure visualization | Same reasoning as dtreeviz. The diagnostic is quantitative, not visual. |
| `networkx` | Graph analysis of tree structure | LightGBM trees are not graphs requiring graph algorithms. `trees_to_dataframe()` provides the flat tabular view that is perfect for groupby-by-depth analysis. |

## How Existing Tools Map to v1.7 Tasks

### Task 1: Race-Level Aggregation Features

**Features:** `rl_odds_dispersion`, `rl_log_odds_entropy`, `rl_top3_odds_gap`, `rl_field_strength_hhi`, `rl_fav_dominance`, `rl_concentration_ratio`

**Tools:** pandas groupby/transform/agg, scipy.stats.entropy, numpy

**Implementation pattern:** Follow `market_bias_features.py` pattern -- all features are per-race aggregations broadcast to each horse via `groupby("race_id").transform()`.

```python
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

def compute_race_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Race-level market structure features.
    
    Args:
        df: race_id, tanodds, umaban columns required
        
    Returns:
        rl_odds_dispersion, rl_log_odds_entropy, rl_top3_odds_gap, etc.
    """
    df = df.copy()
    
    if "tanodds" not in df.columns:
        # All NaN fallback
        for col in ["rl_odds_dispersion", "rl_log_odds_entropy",
                     "rl_top3_odds_gap", "rl_fav_dominance"]:
            df[col] = np.nan
        return df
    
    odds = df["tanodds"].replace(0, np.nan)
    p_raw = 1.0 / odds
    p_sum = p_raw.groupby(df["race_id"], observed=True).transform("sum")
    p_norm = p_raw / p_sum.replace(0, np.nan)
    
    # rl_odds_dispersion: coefficient of variation of implied probabilities
    def _cv(group):
        vals = group.dropna().values
        if len(vals) < 2:
            return np.nan
        return float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else np.nan
    
    df["rl_odds_dispersion"] = p_norm.groupby(
        df["race_id"], observed=True
    ).transform(_cv)
    
    # rl_log_odds_entropy: scipy.stats.entropy on normalized implied probs
    def _entropy(group):
        vals = group.dropna().values
        if len(vals) < 2:
            return np.nan
        return float(scipy_entropy(vals))
    
    df["rl_log_odds_entropy"] = p_norm.groupby(
        df["race_id"], observed=True
    ).transform(_entropy)
    
    # rl_top3_odds_gap: (odds_3rd_fav - odds_1st_fav) / odds_1st_fav
    def _top3_gap(group):
        sorted_odds = group.dropna().sort_values().values
        if len(sorted_odds) < 3:
            return np.nan
        return float((sorted_odds[2] - sorted_odds[0]) / sorted_odds[0])
    
    df["rl_top3_odds_gap"] = odds.groupby(
        df["race_id"], observed=True
    ).transform(_top3_gap)
    
    return df
```

**Note:** The existing `market_entropy` in `market_bias_features.py` is already Shannon entropy on normalized probabilities. The new `rl_log_odds_entropy` is essentially the same concept but as a named race-level feature for the model. Consider whether to reuse `market_entropy` directly or create the new module. Recommendation: create a new `race_level_features.py` module that depends on `tanodds` and computes all race-level aggregations together.

### Task 2: Market Cross-Consistency Features

**Features:** `mc_win_wide_consistency`, `mc_win_umaren_consistency`, `mc_win_trio_consistency`, `mc_cross_entropy_divergence`

**Core idea:** Compare the implied probability from win odds with the implied probability from combination odds. In an efficient market, P(A wins and B wins) should approximate P(A wins) * P(B wins) after accounting for overround. Large divergences indicate either (a) the market has information not captured by win odds alone, or (b) the race is "unreadable" with inconsistent pricing.

**Data flow:**
1. `odds_tanpuku.parquet` -- tanodds per horse (already loaded)
2. `odds_wide.parquet` -- oddslow/oddshigh per pair (already loaded, kumi="0102" format)
3. `odds_umaren.parquet` -- odds per pair (needs ETL extraction)
4. `odds_sanren.parquet` -- odds per triple (needs ETL extraction, optional for v1.7)

**Tools:** pandas merge + vectorized operations, numpy

**Key computation (win x wide consistency):**
```python
def compute_market_cross_consistency(
    entries_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> pd.DataFrame:
    """Win x Wide odds cross-consistency per race.
    
    For each pair (A, B), compute:
        P_market(A and B place) from wide odds
        P_market(A wins) * P_market(B wins) from win odds (independence assumption)
    Consistency = ratio or log-ratio of these two.
    
    The per-horse feature is the max/mean divergence across all pairs 
    involving that horse.
    """
    # ... per-race cross-consistency computation
    # This is a race-level feature broadcast to each horse
```

**Implementation note:** The `odds_wide.parquet` already has `kumi` column in "0102" format (4-digit, zero-padded pair codes matching `umaban1` and `umaban2`). The `odds_umaren.parquet` will have the same format. No data format conversion needed.

**Simplification for v1.7 MVP:** Start with `odds_wide` only (already available, 17.9 MB). Compute win x wide consistency as the first cross-consistency feature. Add umaren/sanren later if wide consistency proves valuable.

### Task 3: Gain per Depth Diagnostic

**Purpose:** Analyze whether LightGBM implicitly learns a two-stage structure (Stage 1: ability ranking at shallow depths, Stage 2: EV refinement at deeper depths).

**API:** `lightgbm.Booster.trees_to_dataframe()`

**Confirmed output columns (verified with LightGBM 4.6.0 installed):**

| Column | Type | Description |
|--------|------|-------------|
| `tree_index` | int64 | Which tree the node belongs to (0-based) |
| `node_depth` | int64 | Distance from root (root=1, children=2, etc.) |
| `node_index` | str | Unique node identifier |
| `split_feature` | str or None | Feature name used for splitting (None for leaves) |
| `split_gain` | float64 or NaN | Information gain from this split |
| `threshold` | float64 or NaN | Split threshold value |
| `value` | float64 | Predicted value for leaf (scaled by learning rate) |
| `weight` | float64 or int64 | Sum of Hessian at this node |
| `count` | int64 | Number of training samples reaching this node |

**Implementation:**
```python
def compute_gain_per_depth(model: lgb.Booster) -> pd.DataFrame:
    """Analyze depth-stratified gain distribution.
    
    Returns DataFrame with:
        depth, total_gain, mean_gain, split_count, 
        cumulative_gain_fraction, top_feature_at_depth
    """
    tree_df = model.trees_to_dataframe()
    splits = tree_df[tree_df["split_feature"].notna()].copy()
    
    depth_stats = (splits.groupby("node_depth")
                   .agg(total_gain=("split_gain", "sum"),
                        mean_gain=("split_gain", "mean"),
                        split_count=("split_gain", "count"))
                   .reset_index())
    
    depth_stats["cumulative_gain_fraction"] = (
        depth_stats["total_gain"].cumsum() / depth_stats["total_gain"].sum()
    )
    
    # Top feature at each depth
    top_features = (splits.groupby(["node_depth", "split_feature"])
                    .size().reset_index(name="count"))
    top_feat = top_features.loc[
        top_features.groupby("node_depth")["count"].idxmax()
    ]
    depth_stats = depth_stats.merge(
        top_feat[["node_depth", "split_feature"]], on="node_depth"
    )
    
    return depth_stats
```

**Integration with StackedEnsemble:** The `StackedEnsemble` class stores `self.lgbm_model` as the LightGBM component (an `lgb.Booster`). For Gain per Depth analysis, access `ensemble.lgbm_model.trees_to_dataframe()`. The XGBoost and CatBoost models do not need equivalent analysis -- the diagnostic is LightGBM-specific to verify the two-stage hypothesis.

### Task 4: Residual IC Evaluation Metrics

**Metrics:**
- **B-diff IC:** `IC(y, p_model) - IC(y, p_market)` -- how much better is the model than the market?
- **C-orthogonal IC:** `IC(y, residual(p_model | p_market))` -- model's predictive power orthogonal to market
- **E-incremental IC:** `IC(y, p_model) - IC(y, p_market)` in bins -- where does the model add value?

**Tools:** `scipy.stats.spearmanr`, `sklearn.linear_model.LinearRegression`, numpy

**All three metrics use Spearman rank correlation as the IC measure.** This is the standard in quantitative finance because it is robust to outliers and non-linear relationships.

```python
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import numpy as np

def compute_residual_ic(
    y_true: np.ndarray,
    p_model: np.ndarray,
    p_market: np.ndarray,
) -> dict[str, float]:
    """Compute B-diff, C-orthogonal, and E-incremental IC metrics.
    
    Args:
        y_true: Actual outcomes (binary: 0/1 for win, or continuous for return)
        p_model: Model predicted probabilities
        p_market: Market implied probabilities (1/odds, normalized)
    
    Returns:
        Dictionary with ic_raw, ic_market, b_diff, c_orthogonal metrics
    """
    # Raw IC (model vs truth)
    ic_raw, _ = spearmanr(y_true, p_model)
    
    # Market IC (market vs truth)
    ic_market, _ = spearmanr(y_true, p_market)
    
    # B-diff: model's edge over market
    b_diff = ic_raw - ic_market
    
    # C-orthogonal: IC of model's residual after removing market component
    lr = LinearRegression()
    lr.fit(p_market.reshape(-1, 1), p_model)
    residual = p_model - lr.predict(p_market.reshape(-1, 1))
    c_orthogonal, _ = spearmanr(y_true, residual)
    
    return {
        "ic_raw": float(ic_raw),
        "ic_market": float(ic_market),
        "b_diff": float(b_diff),
        "c_orthogonal": float(c_orthogonal),
    }
```

**Extension for multi-dimensional orthogonalization:** If orthogonalizing against multiple market signals (win odds + wide odds + umaren odds), use `LinearRegression().fit(X_market_multi, p_model)` where `X_market_multi` has multiple columns. The residual captures model information orthogonal to all market signals simultaneously.

## Installation

```bash
# No new packages needed
# Only action: ETL extraction for missing odds tables
python scripts/run_etl.py --mode full --start 20140101 --end 20251231 \
    --tables n_odds_umaren n_odds_umatan n_odds_sanren n_odds_sanrentan

# Then verify:
pip install -e ".[dev]"
```

## Integration Points Summary

| Integration Point | File | Change Type | New Dependency? |
|-------------------|------|-------------|-----------------|
| Race-level features module | `src/features/race_level_features.py` (NEW) | New feature module | No |
| Market cross-consistency module | `src/features/market_cross_features.py` (NEW) | New feature module | No |
| Gain per Depth diagnostic | `src/features/gain_per_depth.py` (NEW) | Diagnostic tool | No |
| Residual IC evaluation | `src/models/residual_ic.py` (NEW) | Evaluation metric | No |
| Multi-bet odds reader | `src/db/readers.py` | Add `load_odds_umaren()` etc. | No |
| Feature engine integration | `src/features/feature_engine.py` | Call new modules in `build_all()` | No |
| FEATURE_COLS registration | `src/domain/types.py` | Add new feature column names | No |
| ETL extraction | `data/odds/odds_umaren.parquet` etc. | ETL run | No |
| Training pipeline | `src/pipelines/training_pipeline.py` | Load multi-bet odds data | No |

## Key Design Decision: Why These Tools Over Alternatives

**Race-level features with pandas + scipy:** Race-level aggregation is `groupby("race_id")` followed by standard statistical operations (CV, entropy, quantile gap). This is the exact same pattern as `compute_market_bias()` in `market_bias_features.py`. Using any other tool would mean rewriting 20+ feature modules for no benefit.

**Gain per Depth with `trees_to_dataframe()`:** This is the only built-in LightGBM API that provides per-node gain values with depth information. The alternative -- `dump_model()` returning a JSON tree structure -- requires recursive parsing to extract the same information. `trees_to_dataframe()` does the parsing for us and returns a flat pandas DataFrame, which is exactly what we need for groupby-by-depth analysis.

**Residual IC with spearmanr + LinearRegression:** The information coefficient in quantitative finance is universally computed as Spearman rank correlation. Orthogonalization via OLS residuals is the standard method for computing partial correlation. `scipy.stats.spearmanr` and `sklearn.linear_model.LinearRegression` together provide exactly this capability. No specialized financial library is needed.

**scipy.stats.entropy over manual implementation:** The existing `market_bias_features.py` manually computes Shannon entropy with `np.sum(p * np.log(p))`. The new race-level module should use `scipy.stats.entropy()` because (a) it handles edge cases (zero probabilities, normalization), (b) it is numerically stable, and (c) it is already installed.

## Data Availability Matrix

| Bet Type | Odds Source | Payout Source | Data on Disk? | ETL Needed? |
|----------|------------|---------------|---------------|-------------|
| Win (tanpuku) | `odds_tanpuku.parquet` | `payouts.paytansyo*` | YES | No |
| Place (fukushou) | `odds_tanpuku.parquet` (fukuodds*) | `payouts.payfukusyou*` | YES | No |
| Wide | `odds_wide.parquet` | `payouts.paywide*` | YES | No |
| Bracket quinella (wakuren) | `odds_waku.parquet` | `payouts.paywakuren*` | YES | No |
| Quinella (umaren) | `odds_umaren.parquet` | `payouts.payumaren*` | NO | **YES** |
| Exacta (umatan) | `odds_umatan.parquet` | `payouts.payumatan*` | NO | **YES** |
| Trio (sanrenpuku) | `odds_sanren.parquet` | `payouts.paysanrenpuku*` | NO | **YES** |
| Trifecta (sanrentan) | `odds_sanrentan.parquet` | `payouts.paysanrentan*` | NO | **YES** |

**Minimum viable data for v1.7:** Win + Wide odds are sufficient for the first market cross-consistency features. Umaren/sanren extraction can be deferred to a follow-up phase if the wide-based features show promise.

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| `trees_to_dataframe()` provides needed columns | HIGH | Verified with installed LightGBM 4.6.0. Official docs confirm tree_index, node_depth, split_gain, split_feature columns. Tested with live model. |
| Spearman IC + OLS residual for orthogonal IC | HIGH | Verified with scipy 1.17.1 and sklearn 1.8.0. Both `scipy.stats.spearmanr` and `sklearn.linear_model.LinearRegression` tested with synthetic data. Standard quantitative finance approach. |
| Race-level features with pandas groupby | HIGH | Exact same pattern as existing `market_bias_features.py`, `intra_race_features.py`. No novel patterns. |
| Wide odds usable for cross-consistency | HIGH | `odds_wide.parquet` exists with 3.68M rows. `kumi` column uses "0102" format matching umaban1+umaban2. Same format as `odds_waku.parquet` which is already consumed. |
| Umaren/sanren odds extractable via ETL | HIGH | ETL config defines all tables with correct PKs. EveryDB2 has the data (payout columns in `n_harai` confirm the tables exist). Standard `run_etl.py --tables` invocation. |
| Gain per Depth will reveal two-stage structure | MEDIUM | The hypothesis is that shallow depths capture ability ranking and deep depths capture EV refinement. This is plausible but untested. The diagnostic may show uniform gain distribution, which is also valuable information. |
| Market cross-consistency features will improve ROI | MEDIUM | Theoretically sound: cross-bet-type inconsistencies signal market inefficiency. However, the features' predictive power for win betting specifically is untested. The reference article claims +120% orthogonal IC improvement, but replication is needed. |
| Multi-dimensional orthogonal IC computation | LOW | Extending from single (win odds) to multi-signal (win+wide+umaren) orthogonalization increases dimensionality and may produce unstable residuals with correlated market signals. May need ridge regression instead of OLS for regularization. |

## Gaps Requiring Phase-Specific Research

| Gap | Risk | When to Address |
|-----|------|-----------------|
| Umaren/sanren odds column names and types | MEDIUM -- ETL config defines PKs but not odds column names | Phase 1: Run ETL, inspect parquet columns with `df.columns` |
| Market cross-consistency feature distribution | MEDIUM -- wide odds ratios may have extreme outliers | Phase 1: After feature computation, inspect distribution and clip/winsorize |
| Gain per Depth for StackedEnsemble vs raw Booster | LOW -- StackedEnsemble.lgbm_model is a raw Booster, works directly | Phase 2: Verify with actual trained model |
| Orthogonal IC with correlated market signals | LOW -- win/wide/umaren implied probs are correlated | Phase 2: If multi-signal IC is needed, test Ridge vs OLS for stability |

## Sources

- LightGBM 4.6.0 `trees_to_dataframe()` API verified via official ReadTheDocs (https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html) -- confirmed 15 columns including tree_index, node_depth, split_gain, split_feature, value, weight, count (HIGH confidence)
- LightGBM 4.6.0 live test: created model, called trees_to_dataframe(), confirmed column names and types (HIGH confidence)
- scipy 1.17.1: verified `spearmanr`, `entropy`, `orthogonal_procrustes` all importable and functional (HIGH confidence)
- scikit-learn 1.8.0: verified `LinearRegression` for orthogonalization works with synthetic data (HIGH confidence)
- EveryDB2 ETL config (`config/etl_tables.yaml`): defines odds_umaren, odds_umatan, odds_sanren, odds_sanrentan tables with n_ and s_ source pairs (HIGH confidence)
- `odds_wide.parquet` structure verified: 3,679,035 rows, kumi column in "0102" format, oddslow/oddshigh/ninki columns present (HIGH confidence)
- `payouts.parquet` structure verified: 38,835 rows, 201 columns including payumaren*, paysanren*, paywide*, paywaku* columns (HIGH confidence)
- Existing feature patterns: `market_bias_features.py`, `intra_race_features.py` provide proven templates for groupby-based race-level features (HIGH confidence)

---
*Stack research for: v1.7 Market-Independent Edge Discovery*
*Researched: 2026-05-17*
