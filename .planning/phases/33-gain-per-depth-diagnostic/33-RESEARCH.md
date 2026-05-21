# Phase 33: Gain per Depth Diagnostic - Research

**Researched:** 2026-05-18
**Domain:** LightGBM tree structure analysis, feature importance by depth
**Confidence:** HIGH

## Summary

This phase creates a diagnostic tool that extracts LightGBM tree structures via `trees_to_dataframe()`, classifies each split feature into Market/Fundamental/Categorical categories, and aggregates gain contributions by tree depth. The tool validates whether Market features dominate at shallow depths and Fundamental features activate at deeper depths -- the implicit Two-Stage hypothesis.

The core API, `lgb.Booster.trees_to_dataframe()`, returns a DataFrame with columns including `tree_index`, `node_depth`, `split_feature`, and `split_gain`. Leaf nodes have `split_feature=None` and `split_gain=0`. Each model's Booster is accessed through well-defined attribute paths in `SubmodelSet`. The implementation follows the established diagnostic module pattern (function-based, JSON output, console_summary()).

**Primary recommendation:** Build a function-based `gpd_diagnostics.py` that iterates over all LightGBM Boosters in SubmodelSet, calls `trees_to_dataframe()` on each, maps `split_feature` to Market/Fundamental/Categorical via `FEATURE_CATEGORY_MAP`, and aggregates gain by depth.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** FEATURE_CATEGORY_MAP as explicit dict in gpd_diagnostics.py, key=feature name, value="market"|"fundamental"|"categorical"
- **D-02:** Test validates all models' FEATURE_COLS are registered in FEATURE_CATEGORY_MAP
- **D-03:** 3-class boundary: Market (odds/market structure/Harville ratio), Fundamental (past performance/bloodline/form), Categorical (jockey/trainer/sire/TE/race conditions)
- **D-04:** All SubmodelSet LightGBM Boosters analyzed, tiered output (primary + detailed)
- **D-05:** LightGBM primary; XGBoost/CatBoost documented for future extension only
- **D-06:** Code abstracted by model type for future XGBoost addition
- **D-07:** 3-layer output: JSON report + console_summary() + matplotlib PNG per model
- **D-08:** Matplotlib for graphs (existing dependency)
- **D-09:** Graph design at Claude's discretion. Recommended: depth-wise stacked bar + cumulative gain line
- **D-10:** Continuous depth analysis (no arbitrary binning or thresholds)
- **D-11:** Two quantitative metrics: Market Dominance Ratio + Fundamental Activation Depth
- **D-12:** Human judgment for hypothesis interpretation, no automatic PASS/FAIL
- **D-13:** Function-based module at `src/models/gpd_diagnostics.py` following ev_diagnostics.py pattern
- **D-14:** CLI script at `scripts/run_gpd.py`

### Claude's Discretion
- Graph design specifics (stacked bar layout, colors, subplot configuration)
- Internal function composition of gpd_diagnostics.py
- Complete contents of FEATURE_CATEGORY_MAP
- Model name -> Booster access path abstraction method
- Test case design specifics
- JSON output schema details
- PNG file naming convention per model

### Deferred Ideas (OUT OF SCOPE)
- XGBoost trees_to_dataframe() analysis (future phase)
- CatBoost tree structure analysis (API constraints, future consideration)
- GPD-05 multi-dimensional orthogonal IC (win+wide+umaren)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GPD-01 | LightGBM trees_to_dataframe() でdepth別gain寄与率を集計する機能 | trees_to_dataframe() API verified: returns DataFrame with tree_index, node_depth, split_feature, split_gain columns. Group by node_depth, sum split_gain by category. |
| GPD-02 | Market/Fundamental/Categorical 3分類でdepth別シェアを可視化する機能 | FEATURE_CATEGORY_MAP maps 179 unique features. Matplotlib stacked bar chart per depth level. |
| GPD-03 | StackedEnsemble内LightGBMモデルへのアクセスと分析機能 | StackedEnsemble.lgbm_model provides direct lgb.Booster access. feature_name() returns feature names. |
| GPD-04 | 暗黙的Two-Stage構造の検証 | Market Dominance Ratio (depth 1-3 vs 4+) and Fundamental Activation Depth (first depth where Fundamental > Market) metrics defined. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Tree structure extraction | API / Backend | -- | Reads Booster internals, no UI/DB involved |
| Feature classification | API / Backend | -- | Static dict mapping, pure computation |
| Gain aggregation by depth | API / Backend | -- | Pandas groupby/sum operations |
| JSON report output | CDN / Static | -- | File write to data/gpd/ |
| Console summary | API / Backend | -- | logging output |
| Matplotlib PNG generation | API / Backend | CDN / Static | Chart creation + file write |
| CLI script entry point | API / Backend | -- | argparse + ModelLoader orchestration |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lightgbm | 4.6.0 | trees_to_dataframe() API | Project's core ML framework, already installed [VERIFIED: pip show] |
| pandas | installed | DataFrame aggregation by depth | Project standard for all data manipulation |
| numpy | installed | Numerical operations | Standard dependency |
| matplotlib | installed | PNG chart generation | Already used in project for visualizations [VERIFIED: existing code] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | -- | JSON report serialization | All diagnostic output |
| logging (stdlib) | -- | console_summary() output | Following ev_diagnostics.py pattern |
| argparse (stdlib) | -- | CLI argument parsing | scripts/run_gpd.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib | plotly | plotly adds dependency; matplotlib already installed |
| manual feature classification | regex-based auto-classification | Auto-classification fragile; explicit dict is auditable (D-01) |

**Installation:**
No new packages required. All dependencies are already installed.

**Version verification:**
```
lightgbm==4.6.0 (verified via pip show)
matplotlib: installed (existing project dependency)
pandas: installed (existing project dependency)
numpy: installed (existing project dependency)
```

## Package Legitimacy Audit

> No new packages installed in this phase. All dependencies pre-exist.

| Package | Registry | Status |
|---------|----------|--------|
| lightgbm | PyPI | Pre-existing, VERIFIED |
| matplotlib | PyPI | Pre-existing, VERIFIED |
| pandas | PyPI | Pre-existing, VERIFIED |
| numpy | PyPI | Pre-existing, VERIFIED |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
CLI (run_gpd.py)
  |
  v
ModelLoader.load_from_dir() --> TrainedModelsV5
  |
  v
compute_gpd_diagnostics(TrainedModelsV5)
  |
  +-- extract_boosters(TrainedModelsV5) --> dict[str, lgb.Booster]
  |       |
  |       +-- Primary tier: stage1_{turf,dirt}, win_hit_{turf,dirt}, win_ret_{turf,dirt},
  |       |                market_{turf,dirt}, stacked_ensemble_lgbm (if use_ensemble)
  |       +-- Detailed tier: place_hit_{turf,dirt}, place_ret_{turf,dirt}, ev_corrector_p/e,
  |                         place_ev_corrector_p/e, wide_hit/ret, cqr_q_low/q_high
  |
  +-- For each Booster:
  |     booster.trees_to_dataframe()
  |       |
  |       v
  |     Filter: split_feature.notna()  (exclude leaf nodes)
  |       |
  |       v
  |     Map: split_feature --> FEATURE_CATEGORY_MAP[split_feature]
  |       |
  |       v
  |     Group by node_depth, category --> sum(split_gain)
  |       |
  |       v
  |     Compute: Market Dominance Ratio, Fundamental Activation Depth
  |
  +-- JSON report (data/gpd/gpd_report.json)
  +-- console_summary(result)
  +-- Matplotlib PNG per model (data/gpd/gpd_{model_name}.png)
```

### Recommended Project Structure
```
src/models/gpd_diagnostics.py    # New diagnostic module (function-based)
scripts/run_gpd.py               # New CLI entry point
data/gpd/                        # Output directory (created at runtime)
  gpd_report.json                # JSON diagnostic report
  gpd_{model_name}.png           # Per-model depth charts
```

### Pattern 1: Function-Based Diagnostic Module
**What:** Module-level constants + private compute functions + public orchestration function + JSON output + console_summary()
**When to use:** All diagnostic modules in this project (ev_diagnostics, drift_diagnostics)
**Example:**
```python
# Source: src/models/ev_diagnostics.py pattern
logger = logging.getLogger("models.gpd_diagnostics")

FEATURE_CATEGORY_MAP: dict[str, str] = { ... }  # module-level constant

def _compute_depth_gains(tree_df: pd.DataFrame) -> dict: ...
def _compute_market_dominance_ratio(depth_gains: dict) -> float: ...

def compute_gpd_diagnostics(
    models: TrainedModelsV5,
    output_dir: Path | None = None,
) -> dict: ...

def console_summary(result: dict) -> None: ...
```

### Pattern 2: Booster Extraction from TrainedModelsV5
**What:** Iterate over SubmodelSet fields to collect all lgb.Booster instances with descriptive names
**When to use:** Any analysis requiring per-model Booster access
**Example:**
```python
def _extract_boosters(models: TrainedModelsV5) -> dict[str, lgb.Booster]:
    boosters: dict[str, lgb.Booster] = {}
    for surface, sub in models.submodels.items():
        # Primary analysis targets
        for key, booster in sub.stage1.models.items():
            boosters[f"stage1_{key}"] = booster
        boosters[f"win_hit_{surface}"] = sub.win.hit_model
        boosters[f"win_ret_{surface}"] = sub.win.return_model
        boosters[f"market_{surface}"] = sub.market.model
        if sub.use_ensemble and hasattr(sub.win.hit_model, 'lgbm_model'):
            boosters[f"ensemble_lgbm_{surface}"] = sub.win.hit_model.lgbm_model
        # Detailed analysis targets
        if sub.ev_corrector.p_correction_model is not None:
            boosters[f"ev_corr_p_{surface}"] = sub.ev_corrector.p_correction_model
        if sub.ev_corrector.e_correction_model is not None:
            boosters[f"ev_corr_e_{surface}"] = sub.ev_corrector.e_correction_model
        # ... place, wide, cqr models ...
    return boosters
```

### Pattern 3: Model Loading from Local Directory
**What:** CLI script loads pre-trained models via ModelLoader.load_from_dir()
**When to use:** All CLI diagnostic scripts
**Example:**
```python
# Source: scripts/run_gpd.py pattern (from existing CLI scripts)
from db.model_loader import ModelLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--output-dir", default="data/gpd")
    parser.add_argument("--ensemble", action="store_true")
    args = parser.parse_args()

    loader = ModelLoader()
    trained_models, info = loader.load_from_dir(
        Path(args.models_dir), use_ensemble_override=args.ensemble
    )
    result = compute_gpd_diagnostics(trained_models, output_dir=Path(args.output_dir))
    console_summary(result)
```

### Anti-Patterns to Avoid
- **Class-based diagnostic module:** All existing diagnostics are function-based (ev_diagnostics, drift_diagnostics). Do not create a class.
- **Automatic PASS/FAIL judgment:** D-12 explicitly states human judgment only. Output metrics, do not classify.
- **Modifying FEATURE_COLS:** This phase is read-only analysis. Never modify model structures or feature lists.
- **Loading OOF data for GPD:** trees_to_dataframe() works on trained Boosters directly. No DataFrame input needed.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tree structure parsing | Custom tree traversal | booster.trees_to_dataframe() | LightGBM provides this natively with correct depth/gain info |
| Feature name mapping | String matching heuristics | FEATURE_CATEGORY_MAP explicit dict | D-01 requires explicit mapping, test validates completeness |
| JSON serialization of numpy types | Custom encoder | _json_default() pattern from drift_diagnostics.py | Handles np.integer, np.floating, np.ndarray, pd.Timestamp |
| Model loading | Manual file loading | ModelLoader.load_from_dir() | Handles ensemble/non-ensemble, all model types, backward compat |

**Key insight:** LightGBM's `trees_to_dataframe()` handles all tree structure complexity (leaf vs split nodes, feature indices to names, node indexing). No manual tree traversal needed.

## Common Pitfalls

### Pitfall 1: StackedEnsemble Booster Access
**What goes wrong:** StackedEnsemble's hit_model is not an lgb.Booster -- it's a StackedEnsemble object. Calling trees_to_dataframe() on it directly fails.
**Why it happens:** StackedEnsemble wraps lgb.Booster in self.lgbm_model, but the outer object mimics Booster's predict() interface.
**How to avoid:** Check isinstance(obj, lgb.Booster) before calling trees_to_dataframe(). For StackedEnsemble, access via `.lgbm_model` attribute.
**Warning signs:** AttributeError: 'StackedEnsemble' object has no attribute 'trees_to_dataframe'

### Pitfall 2: AbilityModel Per-Surface Boosters
**What goes wrong:** AbilityModel stores Boosters in `self.models: dict[str, lgb.Booster]` keyed by surface ("turf"/"dirt"), not as a single booster.
**Why it happens:** Stage1 is a lambdarank model trained per-surface, unlike Stage2 models which are single Boosters per surface.
**How to avoid:** Iterate `sub.stage1.models.items()` to get each surface's Booster. Each key gives a separate Booster with its own tree structure.
**Warning signs:** Trying to call trees_to_dataframe() on AbilityModel instance directly.

### Pitfall 3: Leaf Nodes Have No split_feature
**What goes wrong:** Leaf nodes in trees_to_dataframe() output have `split_feature=None` and `split_gain=0`. Including them in aggregation causes KeyError in FEATURE_CATEGORY_MAP.
**Why it happens:** Leaf nodes are terminal -- they have no split, only a value.
**How to avoid:** Filter `df[df["split_feature"].notna()]` before mapping to categories. Leaf nodes should be excluded from gain-by-depth analysis.
**Warning signs:** KeyError: None in FEATURE_CATEGORY_MAP lookup.

### Pitfall 4: PlaceTwoStageModel Has Separate HIT and RETURN Feature Sets
**What goes wrong:** PlaceTwoStageModel uses HIT_FEATURE_COLS (90 features) for hit_model and RETURN_FEATURE_COLS (92 features) for return_model. The FEATURE_COLS alias points to RETURN_FEATURE_COLS.
**Why it happens:** Hit and return sub-models have different feature sets in PlaceTwoStageModel.
**How to avoid:** For FEATURE_CATEGORY_MAP completeness test, use the union of HIT_FEATURE_COLS + RETURN_FEATURE_COLS for PlaceTwoStageModel. For trees_to_dataframe(), each Booster's feature_name() gives the actual features used.
**Warning signs:** FEATURE_CATEGORY_MAP test fails for features only in HIT_FEATURE_COLS but not in RETURN_FEATURE_COLS.

### Pitfall 5: CQR Model Feature Cols Differ from Feature Importance Cols
**What goes wrong:** ConformalEVModel trains q_low_model and q_high_model with feature_cols that may be a subset of FEATURE_COLS (filtered by available numeric columns at train time).
**Why it happens:** CQR model filters features dynamically: `self.feature_cols = [c for c in self.FEATURE_COLS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]`
**How to avoid:** Use `booster.feature_name()` (from the actual Booster) rather than `ConformalEVModel.FEATURE_COLS` when verifying feature coverage. FEATURE_CATEGORY_MAP should cover ConformalEVModel.FEATURE_COLS but some may not appear in actual tree splits.
**Warning signs:** Features in ConformalEVModel.FEATURE_COLS that are categorical (like "surface", "distance_bin") are excluded from actual training.

### Pitfall 6: StackedEnsemble Has No FEATURE_COLS Class Attribute
**What goes wrong:** StackedEnsemble does not define FEATURE_COLS. Feature names come from the training DataFrame passed to train().
**Why it happens:** StackedEnsemble is a meta-learner that uses the same feature space as WinTwoStageModel.hit_model.
**How to avoid:** For the completeness test, StackedEnsemble's features should be WinTwoStageModel.FEATURE_COLS (same feature space). For trees_to_dataframe(), use booster.feature_name().
**Warning signs:** AttributeError when trying to access StackedEnsemble.FEATURE_COLS.

### Pitfall 7: NaN split_gain Values
**What goes wrong:** Some split nodes may have split_gain=NaN or 0, especially in early stopping scenarios where the last trees have very small gains.
**Why it happens:** LightGBM may produce trees with zero or NaN gain when features provide no useful information.
**How to avoid:** Use `split_gain.fillna(0)` before aggregation. Zero-gain splits still contribute to depth structure analysis but should not affect Market Dominance Ratio.
**Warning signs:** Unexpected NaN values in aggregated depth-category gain table.

## Code Examples

### trees_to_dataframe() Schema (VERIFIED on lightgbm 4.6.0)
```python
# Source: Verified via live test on lightgbm 4.6.0
# Returns DataFrame with 15 columns:
# tree_index: int64       - Tree number (0-based)
# node_depth: int64       - Depth of node (1=root)
# node_index: object      - String node identifier
# left_child: object      - Left child node index (None for leaves)
# right_child: object     - Right child node index (None for leaves)
# parent_index: object    - Parent node index
# split_feature: object   - Feature name (None for leaf nodes)
# split_gain: float64     - Information gain at this split (0 for leaves)
# threshold: float64      - Split threshold value
# decision_type: object   - Split operator (<=, etc.)
# missing_direction: object - Which child NaN goes to
# missing_type: object    - NaN handling type
# value: float64          - Node prediction value
# weight: float64         - Node weight
# count: int64            - Number of samples at node

# Key usage pattern:
df = booster.trees_to_dataframe()
split_nodes = df[df["split_feature"].notna()]  # exclude leaves
depth_gains = split_nodes.groupby(["node_depth", "split_feature"])["split_gain"].sum()
```

### Extracting Boosters from TrainedModelsV5
```python
# Source: Derived from src/domain/models.py SubmodelSet structure (verified)
def _extract_boosters(models: TrainedModelsV5) -> dict[str, lgb.Booster]:
    """Extract all LightGBM Boosters from TrainedModelsV5 with descriptive names."""
    import lightgbm as lgb
    boosters: dict[str, lgb.Booster] = {}

    for surface, sub in models.submodels.items():
        # Stage1 AbilityModel: per-surface boosters
        for key, booster in sub.stage1.models.items():
            if isinstance(booster, lgb.Booster):
                boosters[f"stage1_{key}"] = booster

        # Win TwoStage: hit + return
        hit_model = sub.win.hit_model
        if isinstance(hit_model, lgb.Booster):
            boosters[f"win_hit_{surface}"] = hit_model
        elif hasattr(hit_model, 'lgbm_model') and hit_model.lgbm_model is not None:
            # StackedEnsemble: extract inner lgb.Booster
            boosters[f"ensemble_lgbm_{surface}"] = hit_model.lgbm_model

        if isinstance(sub.win.return_model, lgb.Booster):
            boosters[f"win_ret_{surface}"] = sub.win.return_model

        # Market Model
        if sub.market.model is not None:
            boosters[f"market_{surface}"] = sub.market.model

        # EV Correction: P + E models
        if sub.ev_corrector.p_correction_model is not None:
            boosters[f"ev_corr_p_{surface}"] = sub.ev_corrector.p_correction_model
        if sub.ev_corrector.e_correction_model is not None:
            boosters[f"ev_corr_e_{surface}"] = sub.ev_corrector.e_correction_model

        # Place (optional)
        if sub.place is not None:
            if isinstance(sub.place.hit_model, lgb.Booster):
                boosters[f"place_hit_{surface}"] = sub.place.hit_model
            if isinstance(sub.place.return_model, lgb.Booster):
                boosters[f"place_ret_{surface}"] = sub.place.return_model

        # Place EV Correction (optional)
        if sub.place_ev_corrector is not None:
            if sub.place_ev_corrector.p_correction_model is not None:
                boosters[f"place_ev_corr_p_{surface}"] = sub.place_ev_corrector.p_correction_model
            if sub.place_ev_corrector.e_correction_model is not None:
                boosters[f"place_ev_corr_e_{surface}"] = sub.place_ev_corrector.e_correction_model

        # Wide (optional)
        if sub.wide is not None:
            if sub.wide.hit_model is not None:
                boosters[f"wide_hit_{surface}"] = sub.wide.hit_model
            if sub.wide.return_model is not None:
                boosters[f"wide_ret_{surface}"] = sub.wide.return_model

        # ConformalEV / CQR (optional)
        if sub.conformal_ev_model is not None:
            if sub.conformal_ev_model.q_low_model is not None:
                boosters[f"cqr_q_low_{surface}"] = sub.conformal_ev_model.q_low_model
            if sub.conformal_ev_model.q_high_model is not None:
                boosters[f"cqr_q_high_{surface}"] = sub.conformal_ev_model.q_high_model

    return boosters
```

### JSON Output Pattern (from drift_diagnostics.py)
```python
# Source: src/models/drift_diagnostics.py lines 219-229
def _json_default(obj: object) -> object:
    """JSON non-serializable type fallback."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Usage in write:
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=_json_default)
```

### Console Summary Pattern (from ev_diagnostics.py)
```python
# Source: src/models/ev_diagnostics.py lines 374-449
def console_summary(result: dict) -> None:
    """Output formatted summary to log."""
    logger.info("=== GPD Diagnostics ===")
    for model_name, model_data in result.get("models", {}).items():
        tier = model_data.get("tier", "?")
        logger.info("  [%s] %s:", tier, model_name)
        mdr = model_data.get("market_dominance_ratio")
        fad = model_data.get("fundamental_activation_depth")
        logger.info("    Market Dominance Ratio: %.4f", mdr or float("nan"))
        logger.info("    Fundamental Activation Depth: %s", fad or "N/A")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Feature importance (global) | trees_to_dataframe() per-depth analysis | lightgbm 3.x+ | Can trace feature contributions at each tree depth level |
| SHAP (sample-level) | Gain per depth (tree-level) | -- | Complementary: SHAP explains individual predictions, GPD explains model structure |

**Deprecated/outdated:**
- `booster.dump_model()` text output: Use `trees_to_dataframe()` for structured data instead

## Booster Access Map

Complete map of every LightGBM Booster accessible from TrainedModelsV5, organized by analysis tier.

### Primary Analysis (Two-Stage hypothesis validation)

| Model Name | Access Path | Booster Attribute | Notes |
|------------|-------------|-------------------|-------|
| stage1_{surface} | submodels[surface].stage1.models[surface] | lgb.Booster directly | AbilityModel uses dict of per-surface boosters |
| win_hit_{surface} | submodels[surface].win.hit_model | lgb.Booster or StackedEnsemble | StackedEnsemble when use_ensemble=True |
| win_ret_{surface} | submodels[surface].win.return_model | lgb.Booster directly | Always plain Booster |
| market_{surface} | submodels[surface].market.model | lgb.Booster directly | Market prediction model |
| ensemble_lgbm_{surface} | submodels[surface].win.hit_model.lgbm_model | lgb.Booster | Only when use_ensemble=True, inner Booster from StackedEnsemble |

### Detailed Analysis (supplementary)

| Model Name | Access Path | Booster Attribute | Notes |
|------------|-------------|-------------------|-------|
| ev_corr_p_{surface} | submodels[surface].ev_corrector.p_correction_model | lgb.Booster | P-correction (binary) |
| ev_corr_e_{surface} | submodels[surface].ev_corrector.e_correction_model | lgb.Booster | E-correction (regression) |
| place_hit_{surface} | submodels[surface].place.hit_model | lgb.Booster or StackedEnsemble | Optional, may be None |
| place_ret_{surface} | submodels[surface].place.return_model | lgb.Booster | Optional |
| place_ev_corr_p_{surface} | submodels[surface].place_ev_corrector.p_correction_model | lgb.Booster | Optional |
| place_ev_corr_e_{surface} | submodels[surface].place_ev_corrector.e_correction_model | lgb.Booster | Optional |
| wide_hit_{surface} | submodels[surface].wide.hit_model | lgb.Booster | Optional |
| wide_ret_{surface} | submodels[surface].wide.return_model | lgb.Booster | Optional |
| cqr_q_low_{surface} | submodels[surface].conformal_ev_model.q_low_model | lgb.Booster | Optional |
| cqr_q_high_{surface} | submodels[surface].conformal_ev_model.q_high_model | lgb.Booster | Optional |

### Excluded (per D-04 decision)

| Model | Reason |
|-------|--------|
| RegimeDetector | Classification/routing model, not prediction-essential |
| RaceQualityScreener | Screening model, not prediction-essential |

## FEATURE_COLS Inventory

### Feature Counts by Model

| Model | FEATURE_COLS Count | Feature Set Name |
|-------|-------------------|------------------|
| AbilityModel | 102 | FEATURE_COLS |
| WinTwoStageModel | 87 | FEATURE_COLS |
| PlaceTwoStageModel (hit) | 90 | HIT_FEATURE_COLS |
| PlaceTwoStageModel (return) | 92 | RETURN_FEATURE_COLS |
| MarketModel | 14 | FEATURE_COLS |
| EVCorrectionModel | 30 | FEATURE_COLS |
| PlaceEVCorrectionModel | 30 | FEATURE_COLS |
| ConformalEVModel | 136 | FEATURE_COLS |
| WideTwoStageModel | 12 | SHARED_FEATURE_COLS |
| StackedEnsemble | N/A (uses WinTwoStageModel.FEATURE_COLS) | -- |

**Total unique features across all models: 179**

### 3-Class Feature Classification (FEATURE_CATEGORY_MAP)

Per D-03 boundary criteria, the 179 unique features are classified as follows. The planner should use this as the authoritative source for FEATURE_CATEGORY_MAP construction.

#### Market (41 features)
Odds-derived, market structure, market cross-consistency, FLB/overround:

```
abs_log_error_win, deviation_rank, deviation_zscore, dm_confidence_range,
dm_time_margin_to_fav, dm_time_rank, dm_time_zscore, e_return_place_pred,
e_return_win_pred, entropy_ema, fukuoddslow, implied_prob_hhi,
market_entropy, odds, odds_acceleration, odds_direction_consistency,
odds_drop_rate_30_10, odds_drop_rate_60_10, odds_gap_fav12,
odds_popularity_gap, odds_skewness, odds_to_ability_ratio, odds_velocity,
odds_volatility, overround, overround_ema, p_minus_e_gap, p_x_e_interaction,
popularity_change_30_10, popularity_rank, popularity_rank_fallback_used,
race_mean_fuku_odds, race_std_fuku_odds, rl_favorite_in_wide_top1,
rl_market_consistency, rl_trio_odds_ratio, rl_trio_overlap,
rl_wide_harville_ratio, signed_log_error_win, tanninki, tanodds
```

Note: Some features like `dm_time_*` and `deviation_*` straddle the Market/Fundamental boundary. They are classified as Market because they are derived from odds-based rankings and market signals.

#### Fundamental (125 features)
Past performance, bloodline, physical condition, form, pace, course aptitude, EMA:

```
actual_pace_fit, bataijyu, blinker_change, blood_condition_wr,
blood_distance_wr, blood_prize_log, blood_surface_wr,
blood_surface_wr_x_condition, blood_total_wr, bms_distance_wr,
bms_surface_wr, bms_wr, breeder_strength, class_adj_formetric,
class_demotions, class_drop_bounce, class_level_std, class_max_level,
class_move, class_net_change, class_promotions, closing_index_avg,
closing_index_avg_race_rank, closing_pace_wr, cond_change_avg_pos,
cond_change_exp_count, cond_change_win_rate, course_distance_wr,
course_record_time, course_wr, dam_prize_log, dam_surface_wr, dam_wr,
days_since_last_race, difficulty_score, dist_change_avg_pos,
dist_change_exp_count, dist_change_win_rate, distance_change, draw_ratio,
field_size, form_consistency, form_peak_flag, form_trend,
freshness_score, front_pace_wr, haron_x_distance, haron_zscore_trend,
harontime_late_trend, harontimel5_avg, harontimel5_avg_race_rank,
harontimel5_zscore, is_nar_transfer, jockey_prize_log,
jockey_wr_distance, jockey_wr_overall, jockey_wr_venue,
jt_combo_place_rate, jt_combo_prize_log, jt_combo_starts,
jt_combo_wr, jyuni1c_avg, jyuni1c_avg_race_rank, jyuni4c_avg,
nar_recent_ratio, norm_finish_logit_avg, norm_finish_logit_avg_race_rank,
pace_aptitude, pace_closing_power, pace_corner_stability,
pace_position_consistency, pace_pressure, pace_pressure_x_closing_index,
pace_scenario_fit, p_ability_place, p_ability_win,
position_improvement_rate, rel_blood_quality_rank, rel_closing_index_rank,
rel_fuku_odds_zscore, rel_haron_vs_mean, rel_norm_finish_zscore,
rel_odds_ability_deviation, rel_p_ability_win_rank,
rel_p_ability_win_zscore, rel_popularity_rank_zscore,
rel_sire_quality_rank, rel_timediff_rank, rel_weight_zscore,
rest_category, sire_distance_wr, sire_prize_avg, sire_surface_wr,
sire_wr, sire_wr_x_distance, surf_change_avg_pos,
surf_change_exp_count, surf_change_win_rate, surface_change,
time_improvement_rate, timediff_avg, timediff_avg_race_rank,
track_condition_delta, trainer_prize_log, trainer_wr_distance,
trainer_wr_overall, trainer_wr_venue, v_recovery_duration,
v_recovery_flag, weight_absolute, weight_change_ratio, weight_change_zone,
weight_diff_from_mean, weight_x_class, weight_x_distance, weight_zscore,
win_dominance, zogen_sa
```

Note: `jockey_*`, `trainer_*`, and `jt_combo_*` features are classified as Fundamental because they represent individual performance metrics rather than categorical identity.

#### Categorical (19 features)
Race conditions, categorical IDs, target encoding:

```
blood_keito_cd, blood_keito_x_surface, distance_bin, frame_number,
grade_code, grade_code_x_distance_bin, kyakusitu_x_distance,
kyakusitu_x_surface, kyakusitukubun_cd, kyori, surface,
surface_x_distance_bin, surface_x_past_perf, surface_track_interaction,
te_blood_keito_cd, te_chokyosicode, te_kisyucode, trackcd,
track_condition_code
```

Note: `surface`, `distance_bin`, `grade_code`, `track_condition_code` are race conditions classified as Categorical (they define the race context). `te_*` features are target-encoded categorical IDs. `*_x_*` interaction features involving race conditions are Categorical.

### Feature Category Boundary Clarification

The classification follows these rules:
- **Market**: Any feature derived from odds, implied probabilities, popularity rankings, or market structure indicators (including Harville ratios and cross-consistency metrics)
- **Fundamental**: Any feature measuring individual horse capability -- past performance, bloodline statistics, physical condition, form trends, pace aptitude, course records. Includes jockey/trainer performance statistics.
- **Categorical**: Any feature that is a categorical ID or race condition used for grouping -- surface type, distance band, grade code, bloodline code, target-encoded IDs, interaction terms between categorical features.

## Metrics Definitions

### Market Dominance Ratio (D-11)
```
MDR = (Market_gain_share at depth 1-3) - (Market_gain_share at depth 4+)
```
Where `Market_gain_share at depth D = Market_gain(D) / Total_gain(D)`.
- MDR > 0: Market features dominate at shallow depths (supports Two-Stage hypothesis)
- MDR < 0: Market features are more active at deeper depths (contradicts hypothesis)
- MDR = 0: No depth preference for Market features

### Fundamental Activation Depth (D-11)
```
FAD = min(depth D where Fundamental_gain_share(D) > Market_gain_share(D))
```
- Low FAD (e.g., 2-3): Fundamental features activate early, competing with Market at shallow depths
- High FAD (e.g., 5+): Fundamental features only become important at deeper depths (supports hypothesis)
- None: Market features dominate at all depths

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `dm_time_*` and `deviation_*` features are classified as Market | FEATURE_COLS Inventory | Minor: these could arguably be Fundamental; classification affects depth analysis results but not functionality |
| A2 | `jockey_*`, `trainer_*`, `jt_combo_*` are Fundamental, not Categorical | FEATURE_COLS Inventory | Minor: these are statistics derived from categorical IDs, so Fundamental is appropriate |
| A3 | StackedEnsemble.feature_name() returns feature names matching WinTwoStageModel.FEATURE_COLS | Booster Access Map | Medium: if feature names differ, FEATURE_CATEGORY_MAP coverage test may give false positives |
| A4 | trees_to_dataframe() returns feature names (not indices) in split_feature column | trees_to_dataframe() Schema | LOW: verified by live test on lightgbm 4.6.0 |
| A5 | `p_ability_win`, `p_ability_place` are Fundamental (model outputs) | FEATURE_COLS Inventory | Minor: these are Stage1 outputs feeding into Stage2, could arguably be a separate category |

**Claims needing user confirmation:** A1 (dm_time/deviation classification) and A2 (jockey/trainer classification) are judgment calls where reasonable people may disagree. The impact is low since the FEATURE_CATEGORY_MAP is explicitly defined and can be adjusted.

## Open Questions (RESOLVED)

1. **Feature classification edge cases** (RESOLVED: Classify `dm_time_*` as Market per D-03 ("market-derived"). The explicit FEATURE_CATEGORY_MAP is auditable and can be adjusted if analysis shows unexpected behavior.)
   - What we know: 179 features need classification. Most are unambiguous.
   - What was unclear: Whether `dm_time_*` features should be Market (odds-rank-derived) or Fundamental (data-mining predictions).
   - Resolution: Market per D-03 boundary criteria. Can adjust if analysis shows they behave differently.

2. **Model availability in test environment** (RESOLVED: Tests mock Booster objects via unittest.mock. CLI requires pre-trained models in `data/models/` — documented in `--help` text and README.)
   - What we know: CLI script needs trained models. Tests use mocks.
   - What was unclear: Whether `data/models/` has current models with Phase 31/32 features (rl_*, Harville ratios).
   - Resolution: Tests mock the Booster. CLI requires pre-trained models. Documented clearly in CLI help text.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | Runtime | -- | mise | -- |
| lightgbm | trees_to_dataframe() | -- | 4.6.0 | -- |
| matplotlib | PNG chart generation | -- | installed | -- |
| pandas | Data aggregation | -- | installed | -- |
| numpy | Numerical ops | -- | installed | -- |

**Missing dependencies with no fallback:** None

**Missing dependencies with fallback:** None

## Sources

### Primary (HIGH confidence)
- Live verification of lightgbm 4.6.0 `trees_to_dataframe()` API schema (run in this session)
- `src/models/ev_diagnostics.py` -- function-based diagnostic module pattern reference
- `src/models/drift_diagnostics.py` -- JSON output and console_summary() pattern reference
- `src/domain/models.py` -- SubmodelSet and TrainedModelsV5 structure (lines 230-279)
- `src/models/stacked_ensemble.py` -- StackedEnsemble.lgbm_model access pattern
- `src/models/stage1_ability_model.py` -- AbilityModel.models dict, FEATURE_COLS (102 features)
- `src/models/two_stage_return_model.py` -- WinTwoStageModel/PlaceTwoStageModel Booster and FEATURE_COLS
- `src/models/market_model.py` -- MarketModel.model access, FEATURE_COLS (14 features)
- `src/models/wide_two_stage_model.py` -- WideTwoStageModel SHARED_FEATURE_COLS (12 features)
- `src/models/ev_correction_model.py` -- EVCorrectionModel/PlaceEVCorrectionModel p/e_correction_model access
- `src/models/conformal_ev_model.py` -- ConformalEVModel q_low/q_high_model access
- `src/db/model_loader.py` -- ModelLoader.load_from_dir() pattern for CLI scripts
- `src/features/win_feature_analysis.py` -- dict[str, lgb.Booster] iteration pattern

### Secondary (MEDIUM confidence)
- Phase 31 CONTEXT.md -- rl_* feature definitions
- Phase 32 CONTEXT.md -- Harville ratio feature definitions (rl_wide_harville_ratio, rl_trio_odds_ratio, etc.)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All dependencies verified, API tested live
- Architecture: HIGH - All Booster access paths verified by reading source code
- Pitfalls: HIGH - Identified from actual code structure (StackedEnsemble wrapper, AbilityModel dict, leaf node filtering)
- Feature classification: MEDIUM - Boundary cases (dm_*, jockey_*) are judgment calls

**Research date:** 2026-05-18
**Valid until:** 2026-06-18 (stable: all APIs are mature LightGBM features)
