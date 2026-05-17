# Architecture: Market-Independent Edge Discovery (v1.7)

**Project:** keiba-ai v1.7 -- Market-Independent Edge Discovery
**Researched:** 2026-05-17
**Scope:** Integration architecture for race-level aggregation features, market cross-consistency features, Gain per Depth diagnostics, and Residual IC evaluation within the existing pipeline.
**Confidence:** HIGH (verified against full source code of feature_engine.py, training_pipeline.py, backtest/engine.py, race_predictor.py, stacked_ensemble.py, all feature modules)

## Executive Summary

The v1.7 milestone introduces four capabilities that require different integration strategies. Race-level aggregation features and market cross-consistency features add new columns to the existing feature pipeline. Gain per Depth and Residual IC are diagnostic/evaluation capabilities that plug into the training and validation pipelines respectively.

The critical architectural insight is the race-level vs horse-level feature distinction. The existing pipeline is fundamentally per-horse: one DataFrame row = one horse. Race-level features (constant for all horses in a race) must be computed at race granularity then broadcast to every horse row via `groupby("race_id").transform("first")` or a merge on `race_id`. The existing codebase already handles this pattern in `compute_market_bias()` (overround, entropy are race-level) and `RecordFeatures.compute()` (returns unique-by-race_id DataFrame merged on race_id). The new race-level features follow this exact pattern.

The recommended build order follows the dependency chain: race-level features first (independent, feeds into market cross-consistency), then market cross-consistency (needs both race-level and multi-odds data), then Gain per Depth (diagnostic, needs trained model), then Residual IC (evaluation, needs model predictions and backtest output).

## Recommended Architecture

### Integration Overview

```
EXISTING PIPELINE (unchanged flow, new modules inserted):
================================================================

ParquetStore
  |
  +-> [data/raw/races.parquet]  +-> [data/raw/entries.parquet]
  +-> [data/odds/snapshots]     +-> [data/odds/time_series/]
  +-> [data/odds/odds_wide]     +-> [data/odds/wide.parquet]     <-- v1.7 uses this
  |
  v
FeatureEngine.build_all(race_df, entry_df, odds_df, odds_ts_df, store)
  |
  +-> ... existing modules ...
  +-> compute_market_bias(result_df)          # D: p_market, entropy, overround (RACE-LEVEL)
  +-> compute_flb_slope(result_df)            # D: skewness, HHI (RACE-LEVEL)
  |
  +-> [NEW] compute_race_level_aggregation(result_df)   # v1.7: entropy, dispersion, gap (RACE-LEVEL)
  +-> [NEW] compute_market_cross_consistency(result_df)  # v1.7: win-place-wide consistency (RACE-LEVEL)
  |
  +-> [CACHE WRITE] -> features/cache/feat_*.parquet
  |
  v
TrainingPipelineV5._train_submodel(df, surface)
  |
  +-> ... existing modules (horse_history, pace, sire, record, interaction, mining, relative) ...
  +-> MarketModel.train/predict_oof(df)
  +-> AbilityModel.train_oof(df)
  +-> ... rest of pipeline unchanged ...
  |
  v
[NEW DIAGNOSTIC HOOK - after model training]:
  +-> GainPerDepthAnalysis(model, df_oof)     # v1.7: depth-wise gain distribution
  |
  v
[NEW EVALUATION HOOK - after backtest]:
  +-> ResidualICEvaluator(df_oof, predictions) # v1.7: B/C/E decomposition
```

### Component Boundaries

| Component | Type | File | Responsibility | Communicates With |
|-----------|------|------|---------------|-------------------|
| `compute_race_level_aggregation` | NEW MODULE | `src/features/race_level_features.py` | Race-level odds entropy, dispersion, top-3 gap | FeatureEngine.build_all() |
| `compute_market_cross_consistency` | NEW MODULE | `src/features/market_cross_features.py` | Win-place-wide odds cross-consistency | FeatureEngine.build_all() |
| `GainPerDepthAnalysis` | NEW CLASS | `src/diagnostics/gain_per_depth.py` | LightGBM depth-wise gain extraction and analysis | TrainingPipeline._train_submodel() |
| `ResidualICEvaluator` | NEW CLASS | `src/diagnostics/residual_ic.py` | B/C/E IC decomposition on OOF predictions | TrainingPipeline._train_submodel() output |
| `FeatureEngine` | MODIFIED | `src/features/feature_engine.py` | Wire new modules into build_all() | New feature modules |
| `RacePredictor` | MODIFIED | `src/backtest/race_predictor.py` | Wire new modules into predict() | New feature modules |
| `BacktestEngine` | MODIFIED | `src/backtest/engine.py` | Wire new modules into run() pre-computation | New feature modules |
| `TrainingPipelineV5` | MODIFIED | `src/pipelines/training_pipeline.py` | Wire diagnostics hooks | New diagnostic classes |
| `FEATURE_COLS` | MODIFIED | Model files (two_stage_return_model.py, stage1_ability_model.py) | Add new feature column names to whitelist | Feature modules |

## Data Flow Changes

### CHANGE 1: Race-Level Aggregation Features

**What:** New features that capture the overall market structure of a race: odds dispersion, log-odds entropy, top-3 odds gap. These are RACE-LEVEL features -- same value for every horse in the race.

**Target features (from PROJECT.md):**
- `rl_odds_dispersion`: Std dev of implied probabilities within race (high = wide-open race)
- `rl_log_odds_entropy`: Shannon entropy of implied probabilities (information content)
- `rl_top3_odds_gap`: Gap between 3rd-favorite and 1st-favorite odds (competitive depth)
- `rl_favorite_dominance`: Ratio of 1st-favorite implied prob to sum (concentration)
- `rl_field_competitiveness`: 1 - HHI of implied probabilities (inverse concentration)
- `rl_longshot_ratio`: Max odds / median odds (longshot presence indicator)

**Data flow:**

```
FeatureEngine.build_all()
  |
  v  (after compute_market_bias, which creates p_market_win_adj)
compute_race_level_aggregation(result_df)
  Input columns: race_id, tanodds (from odds_df merge)
  Computation: groupby("race_id").agg(...)
  Output: 6 new columns broadcast to every row via transform("first")
  |
  v  (continues to existing cache write)
```

**Why this insertion point:** After `compute_market_bias()` because it needs `tanodds` which is already merged and cleaned. Before `BloodlineFeatures` because it is a fast vectorized computation. The features are available to both Stage1 (AbilityModel) and Stage2 (WinTwoStageModel) since they are computed in `build_all()`.

**Race-level broadcasting pattern (existing in market_bias_features.py):**
```python
# Example: overround is race-level, broadcast to all horses
overround = p_raw.groupby(df["race_id"], observed=True).transform("sum") - 1.0
df["overround"] = overround
```

New features use the same pattern:
```python
def compute_race_level_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tanodds" not in df.columns:
        # NaN fallback for all 6 features
        return df

    # Implied probabilities (per horse)
    p_raw = 1.0 / df["tanodds"].replace(0, np.nan)

    # --- Race-level features (constant within race, broadcast via transform) ---

    # Dispersion: std of implied probs within race
    df["rl_odds_dispersion"] = p_raw.groupby(
        df["race_id"], observed=True
    ).transform("std")

    # Log-odds entropy (already computed as market_entropy, but
    # this version uses raw implied probs without normalization for different signal)
    # Note: market_entropy already exists from compute_market_bias().
    # rl_log_odds_entropy uses a different normalization to capture dispersion vs entropy.
    # If identical, this column is redundant. Research should verify.
    p_sum = p_raw.groupby(df["race_id"], observed=True).transform("sum")
    p_norm = p_raw / p_sum.replace(0, np.nan)
    def _entropy(group):
        p = group.dropna().values
        p = p[p > 0]
        return float(-np.sum(p * np.log(p))) if len(p) > 0 else np.nan
    df["rl_log_odds_entropy"] = p_norm.groupby(
        df["race_id"], observed=True
    ).transform(_entropy)

    # Top-3 gap: ratio of 3rd-favorite to 1st-favorite odds
    odds_sorted = df.groupby("race_id", observed=True)["tanodds"].apply(
        lambda x: np.sort(x.dropna().values)
    )
    # ... broadcast back to df
    return df
```

**Inference path:** Same function called in `RacePredictor.predict()` (via `FeatureEngine.build_features()` single-race path) and `BacktestEngine.run()` (via `FeatureEngine.build_all()` batch path).

**PIT safety:** All features use only `tanodds` (pre-race snapshot). No post-race data. SAFE.

### CHANGE 2: Market Cross-Consistency Features

**What:** Features that capture how consistent the market is across different bet types (win, place, wide). A "readable race" has consistent odds across bet types; an "unreadable race" has divergent signals.

**Target features:**
- `mc_win_place_consistency`: Correlation between win implied prob and place implied prob across horses in race
- `mc_win_wide_divergence`: Spread between win favorite rank and wide favorite rank
- `mc_place_fav_gap`: Gap between 1st and 2nd in place odds (place market conviction)
- `mc_cross_entropy_ratio`: Win entropy / place entropy (market structure ratio)

**Data requirements:** This is the critical dependency. These features need odds from MULTIPLE bet types in the same feature computation call:
- **Win odds:** `tanodds` (already in feat_df from odds_df merge)
- **Place odds:** `fukuoddslow` (already in feat_df from odds_df merge)
- **Wide odds:** `wide_odds_{lo}_{hi}` columns (already merged in training_pipeline.py and backtest/engine.py)

**Data flow:**

```
TrainingPipeline._train_submodel()  /  BacktestEngine.run()
  |
  +-> feat_df = FeatureEngine.build_all(...)
  +-> wide_odds_df = load_wide_odds(store, start, end)
  +-> feat_df = feat_df.merge(wide_pivot, on="race_id", how="left")   # EXISTING
  |
  +-> [NEW] feat_df = compute_market_cross_consistency(feat_df)
  |     Input: race_id, tanodds, fukuoddslow, wide_odds_* columns
  |     Computation: groupby("race_id").agg(cross-type correlations)
  |     Output: 4 new race-level columns broadcast to every row
  |
  v
```

**CRITICAL INSERTION POINT:** `compute_market_cross_consistency()` MUST run AFTER the wide odds merge in the pipeline. Currently the wide odds merge happens:
- **Training:** `TrainingPipelineV5.run()` line ~191-201 (after `build_all()` and `add_distance_band_features()`)
- **Backtest:** `BacktestEngine.run()` line ~624-639 (after `build_all()` and `add_distance_band_features()`)

This means the new module cannot be called from inside `build_all()` -- it must be called separately in both `TrainingPipeline.run()` and `BacktestEngine.run()`, after the wide odds merge.

**Alternative approach (recommended):** Move the wide odds merge INTO `build_all()` so that all feature computation is centralized. Then `compute_market_cross_consistency()` can be called from inside `build_all()` along with all other feature modules. This would require passing `wide_odds_df` as an optional parameter to `build_all()`.

**Recommended parameter addition to FeatureEngine.build_all():**
```python
def build_all(
    self,
    race_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    odds_ts_df: pd.DataFrame | None = None,
    store: object | None = None,
    preserve_columns: list[str] | None = None,
    wide_odds_df: pd.DataFrame | None = None,  # NEW PARAMETER
) -> pd.DataFrame:
```

Then inside `build_all()`, after the existing odds merge:
```python
# Wide odds merge (consolidated into build_all)
if wide_odds_df is not None and not wide_odds_df.empty:
    # ... existing wide odds pivot logic from training_pipeline.py ...
    result_df = result_df.merge(wide_pivot, on="race_id", how="left")

# Market cross-consistency (needs tanodds + fukuoddslow + wide_odds)
from features.market_cross_features import compute_market_cross_consistency
result_df = compute_market_cross_consistency(result_df)
```

**This approach:** Eliminates code duplication between training and backtest pipelines. The wide odds merge logic is identical in both paths. Centralizing it in `build_all()` is cleaner.

**PIT safety:** All odds are pre-race snapshots. Wide odds are final pre-race odds from `odds_wide` Parquet. SAFE.

**Inference path:** For live inference (`build_features()` single-race method), wide odds may not be available for future races. The module must handle `wide_odds_df=None` gracefully by producing NaN features.

### CHANGE 3: Gain per Depth Diagnostic

**What:** Extracts LightGBM's internal tree structure via `trees_to_dataframe()` and aggregates gain contribution by tree depth. This reveals whether the model has an implicit two-stage structure (shallow splits = coarse groupings, deep splits = fine-grained).

**Data flow:**

```
TrainingPipeline._train_submodel()
  |
  +-> stage1 = AbilityModel()
  +-> df = stage1.train_oof(df, ...)
  +-> win_2s = WinTwoStageModel()
  +-> win_2s.train_hit_model(df_oof, ...)
  |
  +-> [NEW DIAGNOSTIC HOOK]:
  |     if use_ensemble:
  |         # StackedEnsemble wraps LightGBM + XGBoost + CatBoost
  |         # Gain per depth only works on LightGBM Booster directly
  |         gpd = GainPerDepthAnalysis()
  |         gpd_report = gpd.analyze(
  |             model=win_2s.hit_model.lgbm_model,  # LightGBM Booster
  |             feature_names=win_2s.hit_model.feature_name(),
  |             output_dir=Path("data/diagnostics"),
  |             surface=surface,
  |         )
  |     else:
  |         gpd = GainPerDepthAnalysis()
  |         gpd_report = gpd.analyze(
  |             model=win_2s.hit_model,  # LightGBM Booster directly
  |             feature_names=win_2s.hit_model.feature_name(),
  |             output_dir=Path("data/diagnostics"),
  |             surface=surface,
  |         )
  |
  v  (pipeline continues unchanged)
```

**LightGBM `trees_to_dataframe()` API (verified via LightGBM docs):**

Returns a pandas DataFrame with columns:
- `tree_index`: Tree number in the ensemble
- `node_depth`: Depth of the node (0 = root)
- `split_feature`: Feature name used for splitting
- `threshold`: Split threshold
- `gain`: Information gain from the split
- `left_child`, `right_child`: Child node indices
- `leaf_value`: Predicted value (leaf nodes only)

**Analysis output:**
```python
class GainPerDepthReport:
    depth_gain_distribution: dict[int, float]  # depth -> total gain fraction
    top_features_by_depth: dict[int, list[str]]  # depth -> top-5 features
    implicit_stage_boundary: int  # depth where gain accumulation plateaus
    shallow_vs_deep_gain_ratio: float  # depth<=3 gain / depth>3 gain
```

**Files to create:**
- `src/diagnostics/__init__.py`
- `src/diagnostics/gain_per_depth.py` (~120 lines)

**Files to modify:**
- `src/pipelines/training_pipeline.py`: Add diagnostic hook after model training (~15 lines)

**No effect on:** Feature computation, model inference, backtest results. Pure diagnostic.

### CHANGE 4: Residual IC Evaluation

**What:** Information Coefficient (IC) decomposition of model predictions into components that are independent of the market. This quantifies how much of the model's predictive power comes from the market (echo chamber) vs. from fundamental analysis (edge).

**Three IC metrics:**
- **B (Brute-force difference IC):** IC(model predictions) - IC(market implied probabilities). Measures how much IC the model adds over the market.
- **C (Conditional orthogonal IC):** IC of residuals after regressing model predictions on market probabilities. Measures the market-independent component.
- **E (Incremental IC):** IC of the change in predictions when adding the model to the market. Measures marginal information gain.

**Data flow:**

```
TrainingPipeline._train_submodel()
  |
  +-> ... (after all model training and OOF prediction) ...
  +-> df_oof contains:
  |     - p_ability_win (Stage1 OOF prediction)
  |     - ev_win / ev_win_corrected (Stage2 output)
  |     - p_market_win_adj (market probability from compute_market_bias)
  |     - kakuteijyuni (target variable)
  |
  +-> [NEW DIAGNOSTIC HOOK]:
  |     ic_eval = ResidualICEvaluator()
  |     ic_report = ic_eval.evaluate(
  |         y_true=(df_oof["kakuteijyuni"] == 1).astype(float),
  |         y_pred_model=df_oof["p_ability_win"],  # or ev_win_corrected
  |         y_pred_market=df_oof["p_market_win_adj"],
  |         groups=df_oof["race_id"],
  |     )
  |     ic_report.save(Path("data/diagnostics") / f"residual_ic_{surface}.json")
  |
  v  (pipeline continues unchanged)
```

**IC computation formulas:**

```python
class ResidualICEvaluator:
    def evaluate(self, y_true, y_pred_model, y_pred_market, groups) -> ICReport:
        # B: Brute-force difference
        ic_model = spearmanr(y_true, y_pred_model).statistic
        ic_market = spearmanr(y_true, y_pred_market).statistic
        ic_b = ic_model - ic_market

        # C: Conditional orthogonal IC
        # Regress model predictions on market predictions, take residuals
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(y_pred_market.values.reshape(-1, 1), y_pred_model.values)
        residuals = y_pred_model - lr.predict(y_pred_market.values.reshape(-1, 1))
        ic_c = spearmanr(y_true, residuals).statistic

        # E: Incremental IC
        # IC improvement when using both model + market vs market alone
        combined = y_pred_model + y_pred_market  # simple combination
        ic_combined = spearmanr(y_true, combined).statistic
        ic_e = ic_combined - ic_market

        return ICReport(ic_model=ic_model, ic_market=ic_market,
                       ic_b=ic_b, ic_c=ic_c, ic_e=ic_e)
```

**Per-race IC variant:** Compute IC within each race (rank correlation among horses in the same race), then average across races. This is more meaningful for horse racing than global IC because predictions are relative within races.

```python
# Per-race IC (Spearman rank correlation within each race)
def per_race_ic(y_true, y_pred, groups):
    race_ics = []
    for race_id in groups.unique():
        mask = groups == race_id
        if mask.sum() < 3:  # Need at least 3 horses for meaningful rank
            continue
        ic = spearmanr(y_true[mask], y_pred[mask]).statistic
        if not np.isnan(ic):
            race_ics.append(ic)
    return np.mean(race_ics) if race_ics else 0.0
```

**Files to create:**
- `src/diagnostics/residual_ic.py` (~150 lines)

**Files to modify:**
- `src/pipelines/training_pipeline.py`: Add IC evaluation hook after OOF prediction (~20 lines)
- `src/backtest/validation_report.py`: Optionally include IC metrics in validation reports (~10 lines)

**No effect on:** Feature computation, model training, backtest results. Pure evaluation.

## Patterns to Follow

### Pattern 1: Race-Level Feature Broadcasting

**What:** Compute a statistic per race, then broadcast (copy) it to every horse row in that race.

**When:** Any feature that is constant within a race (market structure, field composition, course record).

**Why:** The pipeline is fundamentally per-horse (one row = one horse). LightGBM needs every row to have the feature. Race-level features are the same value for all horses, which is fine -- the model learns that this value modulates the prediction context.

**Existing examples:** `overround`, `market_entropy`, `odds_skewness`, `implied_prob_hhi`, `course_record_time`, `difficulty_score`.

**Implementation:**
```python
# Approach 1: transform("first") -- compute per group, broadcast to all rows
race_entropy = df.groupby("race_id", observed=True)["p_market_win_adj"].transform(_entropy)
df["market_entropy"] = race_entropy

# Approach 2: compute unique-by-race DataFrame, then merge
result = race_df[["race_id"]].drop_duplicates()
result["new_feature"] = ...  # compute once per race
df = df.merge(result, on="race_id", how="left")
```

### Pattern 2: Feature Module Integration Points

**What:** Three distinct integration points exist for feature modules, depending on data dependencies.

| Point | Location | Available Data | When to Use |
|-------|----------|---------------|-------------|
| `build_all()` | `feature_engine.py` | race_df + entry_df + odds_df + odds_ts_df + store | Race-level features, basic horse features, odds features |
| `build_all()` (with wide_odds_df param) | `feature_engine.py` | above + wide odds pivot | Market cross-consistency features |
| `_train_submodel()` | `training_pipeline.py` | above + horse_history + pace + sire + course + interaction | Features requiring horse history or computed features |
| `RacePredictor.predict()` | `backtest/race_predictor.py` | same as _train_submodel for single race | Inference-time feature computation |

**Rule:** If a feature depends only on `tanodds`, `fukuoddslow`, and basic race/entry data, put it in `build_all()`. If it needs `wide_odds_*` columns, it needs the wide_odds_df parameter. If it needs horse history features (e.g., relative features), put it in `_train_submodel()`.

### Pattern 3: Diagnostic Hook Placement

**What:** Diagnostics are hooks that read model state and predictions but do not modify them.

**When:** Gain per Depth, Residual IC, drift diagnostics, EV diagnostics.

**Where:** After model training in `_train_submodel()`, gated by `use_ensemble` flag or a separate `--diagnostics` flag.

**Implementation:**
```python
# In _train_submodel(), after model training:
if use_ensemble:  # or: if self._run_diagnostics:
    from diagnostics.gain_per_depth import GainPerDepthAnalysis
    gpd = GainPerDepthAnalysis()
    gpd.analyze(win_2s.hit_model.lgbm_model, output_dir=..., surface=surface)

    from diagnostics.residual_ic import ResidualICEvaluator
    ic_eval = ResidualICEvaluator()
    ic_eval.evaluate(df_oof, output_dir=..., surface=surface)
```

### Pattern 4: Inference Path Mirroring

**What:** Every feature added to the training path must also be computed in the inference path.

**Training paths:**
1. `TrainingPipeline._train_submodel()` (full feature set for model training)
2. `BacktestEngine.run()` (full feature set for backtest simulation)

**Inference paths:**
3. `RacePredictor.predict()` (single-race inference for live betting)
4. `FeatureEngine.build_features()` (single-race feature generation for live betting)

**Rule:** If a feature is added to `build_all()` (paths 1 and 2), it is automatically available in both training and backtest. But it must also be added to `build_features()` (path 4) for live inference. The single-race path currently only calls `_map_basic_features()` and skips all sub-modules. Race-level features computed from `tanodds` should be computed in both paths.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Computing Race-Level Features Inside the Per-Horse Module Chain

**What:** Adding race-level feature computation inside `HorseHistoryFeatures.compute()` or other per-horse modules.

**Why bad:** Race-level features should be computed ONCE per race (via groupby), not N times per horse. Putting them in per-horse modules means they run for every horse unnecessarily or are computed per-horse when they should be per-race.

**Instead:** Compute race-level features in a dedicated module called from `build_all()` using vectorized groupby operations.

### Anti-Pattern 2: Wide Odds Dependency in build_all() Without Parameter

**What:** Calling `load_wide_odds()` from inside `build_all()` without receiving it as a parameter.

**Why bad:** `build_all()` is a pure feature computation function that takes DataFrames as input. Adding a side-effect data load inside it breaks the pattern and makes testing harder. It also means the wide odds load happens even when not needed (e.g., for Stage2 training).

**Instead:** Pass `wide_odds_df` as an optional parameter to `build_all()`. The caller (training pipeline or backtest engine) is responsible for loading it.

### Anti-Pattern 3: Diagnostics That Modify Model State

**What:** Diagnostic functions that modify the model's internal state (e.g., calling `save_model()` or changing model parameters).

**Why bad:** Diagnostics are read-only operations. Modifying model state during training can corrupt the model or introduce non-determinism.

**Instead:** Diagnostics should only read from `model.trees_to_dataframe()`, `model.feature_importance()`, `model.predict()`. Never write to the model.

### Anti-Pattern 4: Race-Level Features Without NaN Guard for Single-Horse Races

**What:** Computing groupby statistics without handling the case where a race has only 1 horse.

**Why bad:** `groupby("race_id").std()` returns NaN for single-element groups. `groupby("race_id").rank()` returns 1.0. These edge cases produce NaN features that may cause issues in downstream models if not handled.

**Instead:** Add `fillna(0)` or explicit NaN handling after groupby operations. The existing `compute_market_bias()` handles this with `_entropy()` returning 0.0 for empty groups.

### Anti-Pattern 5: Adding Features Without Updating FEATURE_COLS Manifest

**What:** Computing new features but forgetting to add them to the FEATURE_COLS lists in model files and the SHA256 manifest.

**Why bad:** The SHA256 feature manifest (`domain/types.py` + model FEATURE_COLS) is the PIT safety gate. New features that bypass the manifest are invisible to the leakage validator.

**Instead:** Add new feature column names to `AbilityModel.FEATURE_COLS` (Stage1) or `WinTwoStageModel.FEATURE_COLS` / `HIT_FEATURE_COLS` / `RETURN_FEATURE_COLS` (Stage2). Re-run the manifest generation after changes.

## Detailed Integration Points

### Integration Point 1: Race-Level Aggregation Features

**New file:** `src/features/race_level_features.py`

**Module signature:**
```python
FEATURE_COLS: list[str] = [
    "rl_odds_dispersion",
    "rl_log_odds_entropy",
    "rl_top3_odds_gap",
    "rl_favorite_dominance",
    "rl_field_competitiveness",
    "rl_longshot_ratio",
]

def compute_race_level_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """Race-level market structure features.

    All features are constant within a race (broadcast via groupby transform).
    Uses only tanodds (pre-race snapshot). PIT-safe.
    """
    ...
```

**Insertion in `FeatureEngine.build_all()`:**
After `compute_flb_slope()` (line ~338), before `compute_difficulty_score()` (line ~342):
```python
from features.race_level_features import compute_race_level_aggregation
with TimingContext("build_all/race_level_agg"):
    result_df = compute_race_level_aggregation(result_df)
```

**Insertion in `FeatureEngine.build_features()` (single-race inference):**
After `_map_basic_features()` and odds merge, add:
```python
from features.race_level_features import compute_race_level_aggregation
df = compute_race_level_aggregation(df)
```

**Training pipeline:** No changes needed (build_all handles it).

**Backtest engine:** No changes needed (build_all handles it).

**Model FEATURE_COLS updates:**
- `WinTwoStageModel.HIT_FEATURE_COLS`: Add all 6 `rl_*` features (market structure affects hit probability)
- `WinTwoStageModel.RETURN_FEATURE_COLS`: Add `rl_odds_dispersion`, `rl_top3_odds_gap` (affects return estimation)
- `AbilityModel.FEATURE_COLS`: Optionally add `rl_field_competitiveness`, `rl_favorite_dominance` (race context for ability estimation)

### Integration Point 2: Market Cross-Consistency Features

**New file:** `src/features/market_cross_features.py`

**Module signature:**
```python
FEATURE_COLS: list[str] = [
    "mc_win_place_consistency",
    "mc_win_wide_divergence",
    "mc_place_fav_gap",
    "mc_cross_entropy_ratio",
]

def compute_market_cross_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Market cross-bet-type consistency features.

    Requires: tanodds, fukuoddslow, and optionally wide_odds_* columns.
    All features are constant within a race (broadcast via groupby transform).
    Uses only pre-race odds snapshots. PIT-safe.
    """
    ...
```

**Option A: Consolidate wide odds into build_all() (recommended):**

Modify `FeatureEngine.build_all()` signature to accept `wide_odds_df`:
```python
def build_all(
    self,
    race_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    odds_ts_df: pd.DataFrame | None = None,
    store: object | None = None,
    preserve_columns: list[str] | None = None,
    wide_odds_df: pd.DataFrame | None = None,  # NEW
) -> pd.DataFrame:
```

Inside `build_all()`, after the existing odds merge (line ~300):
```python
# Wide odds merge (consolidated from training_pipeline and backtest_engine)
if wide_odds_df is not None and not wide_odds_df.empty:
    _wide = wide_odds_df[["race_id", "kumi", "oddslow"]].dropna(subset=["oddslow"])
    if not _wide.empty:
        wide_pivot = _wide.pivot_table(index="race_id", columns="kumi", values="oddslow")
        new_cols = [f"wide_odds_{int(c[:2])}_{int(c[2:])}" for c in wide_pivot.columns]
        wide_pivot.columns = new_cols
        wide_pivot = wide_pivot.reset_index()
        result_df = result_df.merge(wide_pivot, on="race_id", how="left")

# Market cross-consistency (needs tanodds + fukuoddslow + wide_odds)
from features.market_cross_features import compute_market_cross_consistency
with TimingContext("build_all/market_cross"):
    result_df = compute_market_cross_consistency(result_df)
```

**Then remove wide odds merge from:**
- `TrainingPipelineV5.run()` lines ~190-201
- `BacktestEngine.run()` lines ~624-639

**Option B: Call separately in pipeline (simpler but duplicates code):**

Add calls to `compute_market_cross_consistency()` after the wide odds merge in both `TrainingPipeline.run()` and `BacktestEngine.run()`. Less clean but avoids changing `build_all()` signature.

**Recommendation:** Option A. Consolidating wide odds into `build_all()` eliminates code duplication and makes the feature pipeline self-contained. The wide odds merge logic is identical in training and backtest paths.

**Model FEATURE_COLS updates:**
- `WinTwoStageModel.HIT_FEATURE_COLS`: Add all 4 `mc_*` features
- `AbilityModel.FEATURE_COLS`: NOT recommended -- cross-consistency is a market-level signal best used in Stage2

### Integration Point 3: Gain per Depth Diagnostic

**New file:** `src/diagnostics/gain_per_depth.py`

**Module signature:**
```python
@dataclass
class GainPerDepthReport:
    surface: str
    model_name: str  # "win_hit", "win_return", etc.
    depth_gain_distribution: dict[str, float]  # depth -> gain fraction
    top_features_by_depth: dict[str, list[str]]  # depth -> top-5 features
    total_trees: int
    total_gain: float
    shallow_gain_fraction: float  # depth 0-3 gain / total gain
    deep_gain_fraction: float  # depth 4+ gain / total gain
    implicit_stages: int  # number of gain plateaus (1 = no stages, 2 = implicit 2-stage)

class GainPerDepthAnalysis:
    def analyze(
        self,
        model: lgb.Booster,
        output_dir: Path | None = None,
        surface: str = "",
        model_name: str = "",
    ) -> GainPerDepthReport:
        """Extract depth-wise gain distribution from LightGBM model.

        Uses Booster.trees_to_dataframe() to get per-node gain and depth.
        Aggregates gain by depth level across all trees.
        """
        tree_df = model.trees_to_dataframe()
        # Filter to internal (split) nodes only (leaf_value is NaN)
        splits = tree_df[tree_df["leaf_value"].isna()].copy()
        # Aggregate gain by depth
        depth_gain = splits.groupby("node_depth")["gain"].sum()
        total_gain = depth_gain.sum()
        ...
```

**Insertion in `TrainingPipeline._train_submodel()`:**
After `win_2s.train_hit_model()` (or ensemble training), add:
```python
# Gain per Depth diagnostic (after win hit model training)
if use_ensemble:
    from diagnostics.gain_per_depth import GainPerDepthAnalysis
    _gpd = GainPerDepthAnalysis()
    # Access inner LightGBM model from StackedEnsemble
    _gpd.analyze(
        model=win_2s.hit_model.lgbm_model,
        output_dir=Path("data/diagnostics"),
        surface=surface,
        model_name="win_hit",
    )
else:
    from diagnostics.gain_per_depth import GainPerDepthAnalysis
    _gpd = GainPerDepthAnalysis()
    _gpd.analyze(
        model=win_2s.hit_model,
        output_dir=Path("data/diagnostics"),
        surface=surface,
        model_name="win_hit",
    )
```

**For ensemble mode:** `StackedEnsemble` wraps `lgb.Booster` as `self.lgbm_model`. The `trees_to_dataframe()` method is available on this attribute. XGBoost and CatBoost have different tree extraction APIs; the diagnostic focuses on LightGBM for consistency with the primary model.

### Integration Point 4: Residual IC Evaluation

**New file:** `src/diagnostics/residual_ic.py`

**Module signature:**
```python
@dataclass
class ICReport:
    surface: str
    ic_model: float           # Spearman IC of model predictions
    ic_market: float          # Spearman IC of market implied probabilities
    ic_b_difference: float    # B: IC_model - IC_market
    ic_c_orthogonal: float    # C: IC of model residuals orthogonal to market
    ic_e_incremental: float   # E: IC improvement of model+market over market alone
    per_race_ic_model: float  # Mean per-race Spearman IC of model
    per_race_ic_market: float # Mean per-race Spearman IC of market
    per_race_ic_c: float      # Mean per-race orthogonal IC
    n_races: int              # Number of races evaluated
    n_horses: int             # Total horses evaluated

class ResidualICEvaluator:
    def evaluate(
        self,
        y_true: pd.Series,
        y_pred_model: pd.Series,
        y_pred_market: pd.Series,
        groups: pd.Series,  # race_id for per-race IC
        output_dir: Path | None = None,
        surface: str = "",
    ) -> ICReport:
        """Compute B/C/E residual IC decomposition."""
        ...
```

**Insertion in `TrainingPipeline._train_submodel()`:**
After `AbilityModel.train_oof()` produces `p_ability_win`, add:
```python
# Residual IC evaluation (after Stage1 OOF predictions)
from diagnostics.residual_ic import ResidualICEvaluator
ic_eval = ResidualICEvaluator()
ic_report = ic_eval.evaluate(
    y_true=(df_oof["kakuteijyuni"] == 1).astype(float),
    y_pred_model=df_oof["p_ability_win"],
    y_pred_market=df_oof["p_market_win_adj"],
    groups=df_oof["race_id"],
    output_dir=Path("data/diagnostics"),
    surface=surface,
)
logger.info(
    "Residual IC for %s: B=%.4f, C=%.4f, E=%.4f (model=%.4f, market=%.4f)",
    surface, ic_report.ic_b_difference, ic_report.ic_c_orthogonal,
    ic_report.ic_e_incremental, ic_report.ic_model, ic_report.ic_market,
)
```

**Optional integration in `BacktestEngine`:**
Add IC metrics to the validation report generated at the end of backtest:
```python
# In _build_race_level_features or validation_report.py
# Compute IC on test-period predictions
```

## Build Order (Dependency-Driven)

### Phase 1: Race-Level Aggregation Features
**Dependencies:** None (uses only existing tanodds data)
**Changes:**
- New module: `src/features/race_level_features.py` (~80 lines)
- Modify: `src/features/feature_engine.py` (add module call in build_all, ~5 lines)
- Modify: `src/features/feature_engine.py` (add module call in build_features for inference, ~5 lines)
- Modify: model FEATURE_COLS lists (~10 lines)
**Verification:** Backtest with new features, verify ROI changes
**Estimated effort:** Small
**Risk:** LOW -- additive only, follows existing pattern exactly

### Phase 2: Market Cross-Consistency Features
**Dependencies:** Phase 1 (uses race-level patterns); wide odds data already ETL'd
**Changes:**
- New module: `src/features/market_cross_features.py` (~100 lines)
- Modify: `src/features/feature_engine.py` (add wide_odds_df parameter, add module call, ~20 lines)
- Modify: `src/pipelines/training_pipeline.py` (remove wide odds merge, pass to build_all, ~15 lines)
- Modify: `src/backtest/engine.py` (remove wide odds merge, pass to build_all, ~15 lines)
- Modify: model FEATURE_COLS lists (~5 lines)
**Verification:** Backtest with new features, verify ROI changes
**Estimated effort:** Medium (code consolidation required)
**Risk:** MEDIUM -- changing build_all signature affects two callers; wide odds may be sparse

### Phase 3: Gain per Depth Diagnostic
**Dependencies:** Trained LightGBM model (from existing pipeline)
**Changes:**
- New file: `src/diagnostics/__init__.py`
- New file: `src/diagnostics/gain_per_depth.py` (~120 lines)
- Modify: `src/pipelines/training_pipeline.py` (add diagnostic hook, ~15 lines)
**Verification:** Run training, verify diagnostic output JSON
**Estimated effort:** Small
**Risk:** LOW -- read-only diagnostic, no effect on model or predictions

### Phase 4: Residual IC Evaluation
**Dependencies:** OOF predictions from Stage1 (p_ability_win, p_market_win_adj)
**Changes:**
- New file: `src/diagnostics/residual_ic.py` (~150 lines)
- Modify: `src/pipelines/training_pipeline.py` (add evaluation hook, ~15 lines)
- Optionally modify: `src/backtest/validation_report.py` (include IC metrics, ~10 lines)
**Verification:** Run training, verify IC report output; verify C > 0 (market-independent edge exists)
**Estimated effort:** Small
**Risk:** LOW -- read-only evaluation, no effect on model or predictions

### Phase 5: Validation and Manifest Update
**Dependencies:** All phases complete
**Changes:**
- Update SHA256 feature manifest with new columns
- Run full backtest to validate ROI impact
- Run walk-forward validation to detect overfitting
**Verification:** WF validation passes, backtest ROI improvement
**Estimated effort:** Medium (mainly compute time)

## Modified vs New Components Summary

### New Components (4 files)

| File | Purpose | LOC Estimate |
|------|---------|-------------|
| `src/features/race_level_features.py` | Race-level odds aggregation features (6 features) | ~80 lines |
| `src/features/market_cross_features.py` | Market cross-consistency features (4 features) | ~100 lines |
| `src/diagnostics/__init__.py` | Package init | ~1 line |
| `src/diagnostics/gain_per_depth.py` | Gain per Depth analysis | ~120 lines |
| `src/diagnostics/residual_ic.py` | Residual IC (B/C/E) evaluation | ~150 lines |

### Modified Components (5 files)

| File | Change | LOC Impact |
|------|--------|-----------|
| `src/features/feature_engine.py` | Add race_level + market_cross module calls; add wide_odds_df param | ~30 lines |
| `src/pipelines/training_pipeline.py` | Remove wide odds merge (consolidate into build_all); add diagnostic hooks | ~30 lines net change |
| `src/backtest/engine.py` | Remove wide odds merge (consolidate into build_all); pass wide_odds_df | ~15 lines net change |
| `src/backtest/race_predictor.py` | No changes needed (race-level features from build_all) | 0 lines |
| Model FEATURE_COLS files | Add rl_* and mc_* column names | ~15 lines |

### Unchanged Components

| Component | Why Unchanged |
|-----------|--------------|
| `ParquetStore` | I/O layer, feature-agnostic |
| `DataRepository` / readers.py | Data access layer, feature-agnostic |
| `StackedEnsemble` | Model layer, consumes whatever features are provided |
| `EVCorrectionModel` | Model layer, features are independent |
| `RegimeDetector` | Uses race-level features but not the new ones |
| `RaceQualityScreener` | Uses race-level features but not the new ones |
| `DrawdownController` | Betting layer, feature-independent |
| `StakeCalculator` | Betting layer, feature-independent |
| `OddsBandFilter` | Betting layer, feature-independent |
| `ConformalEVModel` | Model layer, features are independent |
| `ETL pipeline` | Data source unchanged |
| `domain/types.py` | POST_RACE_COLS unchanged; new features are PRE-race |
| `leakage_validators.py` | Framework exists, new features pass by design |
| `all existing feature modules` | No changes to existing feature computation |

## Scalability Considerations

| Concern | Current | After v1.7 | Mitigation |
|---------|---------|-----------|------------|
| Feature count | ~120 | ~130 (+10) | LightGBM handles this easily |
| Feature computation time (build_all) | ~3 min | ~3.5 min (+10 race-level computations are cheap vectorized groupby) | Negligible |
| Memory per surface | ~4.5K races x 120 cols | ~4.5K races x 130 cols | +8% memory |
| Diagnostic output | ~100 KB | ~200 KB (2 new JSON reports) | Negligible |
| Backtest time per year | ~41 min | ~42 min (new features add ~1 min) | Acceptable |
| Training time | ~17 min | ~17.5 min (diagnostics add ~30s) | Acceptable |

## Sources

- Code analysis: `src/features/feature_engine.py` -- build_all() flow, cache mechanism, _map_basic_features()
- Code analysis: `src/features/market_bias_features.py` -- race-level feature pattern (overround, entropy)
- Code analysis: `src/features/intra_race_features.py` -- within-race relative features pattern
- Code analysis: `src/features/record_features.py` -- race-level feature merge pattern (unique-by-race_id)
- Code analysis: `src/features/relative_features.py` -- within-race z-score/rank pattern
- Code analysis: `src/pipelines/training_pipeline.py` -- _train_submodel() flow, wide odds merge, diagnostic hooks
- Code analysis: `src/backtest/engine.py` -- BacktestEngine.run() flow, wide odds merge, feature pre-computation
- Code analysis: `src/backtest/race_predictor.py` -- RacePredictor.predict() inference chain
- Code analysis: `src/models/stacked_ensemble.py` -- StackedEnsemble.lgbm_model access for diagnostics
- Code analysis: `src/db/readers.py` -- load_wide_odds(), load_odds_snapshots()
- Code analysis: `src/db/odds_extractor.py` -- odds column extraction (tanodds, fukuoddslow, tanninki)
- Code analysis: `src/domain/types.py` -- POST_RACE_COLS, Feature whitelist
- LightGBM docs: `Booster.trees_to_dataframe()` API -- returns DataFrame with tree_index, node_depth, split_feature, gain, threshold columns
- Project context: `.planning/PROJECT.md` -- v1.7 milestone scope and feature targets
