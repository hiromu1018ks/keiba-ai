# Phase 1: Feature Analysis & Enhancement - Research

**Researched:** 2026-05-02
**Domain:** LightGBM feature importance analysis, SHAP-based feature selection, horse racing feature engineering
**Confidence:** HIGH

## Summary

This phase adds a feature importance analysis layer and win-specific feature engineering to an existing mature LightGBM pipeline (v5.5). The system already has 100+ features across 14 modules, but they were designed for place prediction and have never been systematically evaluated for win (tansho) prediction. The WinTwoStageModel uses only 27 features (vs. 45+ for the PlaceTwoStageModel), and the gap is primarily in horse-level ability features that the place model has but the win model lacks.

**Critical discovery:** LightGBM 4.6.0 has built-in SHAP support via `model.predict(data, pred_contrib=True)` -- no external `shap` package required. This produces TreeSHAP values natively, with shape `[n_samples, n_features + 1]` where the last column is the expected value (base value). The external `shap` package is only needed for advanced visualizations (beeswarm, dependence plots). For the core requirement of feature importance ranking + noise identification, LightGBM native is sufficient and avoids a new dependency.

The core technical approach is straightforward: (1) use LightGBM `feature_importance('gain')` for quick scan, (2) use `predict(pred_contrib=True)` for SHAP-based ranking, (3) add new features to the shared `HorseHistoryFeatures.compute()` path, and (4) validate feature removal by retraining and comparing logloss/AUC. The analysis should focus on `WinTwoStageModel.hit_model` (P(win) binary classifier) which is the most critical model for determining which horses to bet on.

**Primary recommendation:** Use LightGBM native `pred_contrib=True` for SHAP analysis (no shap package needed for ranking), add new features to the shared `HorseHistoryFeatures` computation path, and validate feature removal by retraining with AUC comparison.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FEAT-01 | 既存特徴量のSHAP/gain重要度を分析し、単勝予測に寄与する特徴量とノイズ特徴量を特定する | LightGBM `feature_importance('gain')` + `predict(pred_contrib=True)` provides native SHAP values. Run on WinTwoStageModel.hit_model (binary P(win) classifier). No external shap package required. |
| FEAT-02 | 単勝特化の新特徴量を5つ以上追加する(odds-to-ability比、クラス落リバウンド、距離・芝ダート変更要検知、勝利dominance、フレッシュネス) | All 5+ features computable from existing Parquet data. Integration via HorseHistoryFeatures (history-based) and FeatureEngine/WinTwoStageModel (race-level). Raw data columns verified: `kakuteijyuni`, `gradecd`, `jyokencd1`, `distance_bin`, `surface`, `kyori`, `p_ability_win`, `p_market_win_adj`, `days_since_last_race`. |
| FEAT-03 | SHAP分析に基づき、単勝予測に寄与しないノイズ特徴量を特定し除外する | Zero/near-zero SHAP contribution features identified from FEAT-01. Removal validated by retraining + logloss/AUC comparison. Feature removal must be from `WinTwoStageModel.FEATURE_COLS` list only, not from shared computation. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SHAP/gain importance analysis | API / Backend | -- | Runs as a post-training analysis script on trained LightGBM models. Produces a report, not a runtime component. |
| New feature computation (history-based) | API / Backend | -- | HorseHistoryFeatures.compute() already iterates over past race data per horse. New features slot into this loop. |
| New feature computation (race-level) | API / Backend | -- | FeatureEngine._map_basic_features() handles race-level mappings. |
| Feature removal | API / Backend | -- | Remove from WinTwoStageModel.FEATURE_COLS only -- shared computation stays intact for place/wide models. |
| Validation (logloss/AUC) | API / Backend | -- | Retrain WinTwoStageModel with modified feature set, compare metrics against baseline. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lightgbm | 4.6.0 | SHAP via `pred_contrib=True`, gain importance | Already installed. Native SHAP support built into Booster.predict(). [VERIFIED: python -m pip list] |
| numpy | 2.4.3 | Array operations for feature computation | Already installed. [VERIFIED: python -m pip list] |
| pandas | 2.3.3 | DataFrame operations for feature engineering | Already installed. [VERIFIED: python -m pip list] |
| scikit-learn | 1.8.0 | AUC/logloss metrics for validation | Already installed. `sklearn.metrics.log_loss`, `sklearn.metrics.roc_auc_score`. [VERIFIED: python -m pip list] |
| matplotlib | 3.10.8 | Feature importance visualization | Already installed. [VERIFIED: python -m pip list] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| shap (optional) | (not installed) | Advanced SHAP visualizations (beeswarm, dependence plots) | Optional. Only if advanced SHAP plots beyond bar charts are needed. LightGBM native pred_contrib is sufficient for ranking + noise identification. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| LightGBM native `pred_contrib` | shap package (pip install shap) | shap package provides richer visualization (beeswarm, dependence plots) but adds a dependency. LightGBM native is sufficient for importance ranking + noise detection. Use shap package only if advanced plots are needed. [VERIFIED: LightGBM 4.6 docs via Context7] |
| LightGBM gain importance | SHAP importance | Gain is faster to compute but can be biased toward high-cardinality features. SHAP provides theoretically consistent attributions. Use both: gain for quick scan, SHAP for definitive ranking. |
| SHAP on AbilityModel (ranker) | LightGBM gain importance only | AbilityModel uses `lambdarank` objective. LightGBM native `pred_contrib` works for ranker models (returns per-group contributions), but interpretation is more complex. Use gain importance for AbilityModel as simpler and more interpretable. [ASSUMED] |

**Installation:**
```bash
# No new packages required for core functionality.
# Optional: pip install shap  (for advanced SHAP visualizations only)
```

## Architecture Patterns

### System Architecture Diagram

```
[Trained WinTwoStageModel.hit_model (lgb.Booster, binary)]
            |
            v
  +-------------------------+
  | FEAT-01: Analysis       |
  |  1. feature_importance   |
  |     ('gain')             |
  |  2. predict(data,        |
  |     pred_contrib=True)   |
  |     -> SHAP values       |
  |  3. |mean SHAP|/feature  |
  |     -> ranking           |
  +-------------------------+
            |
            v
  [Feature Importance Report]
     - gain ranking (split-based)
     - SHAP ranking (contribution-based)
     - noise candidates (near-zero contribution)
            |
            v
  +-----------------------------+
  | FEAT-02: New Features       |
  |  In HorseHistoryFeatures:   |
  |    - class_drop_bounce      |
  |    - win_dominance          |
  |    - freshness_score        |
  |    - distance_change        |
  |    - surface_change         |
  |  In WinTwoStageModel or     |
  |  _train_submodel():         |
  |    - odds_to_ability_ratio  |
  +-----------------------------+
            |
            v
  +-----------------------------+
  | FEAT-03: Noise Removal      |
  |  1. Identify zero-SHAP      |
  |  2. Remove from             |
  |     WinTwoStageModel.       |
  |     FEATURE_COLS only       |
  |  3. Retrain + compare       |
  |     logloss/AUC             |
  +-----------------------------+
            |
            v
  [Updated WinTwoStageModel with optimized feature set]
```

### Recommended Project Structure
```
src/
  features/
    feature_engine.py          # Add odds_to_ability_ratio call here (or defer to WinTwoStageModel)
    horse_history_features.py  # Add 5 new features to compute() loop + BASE_COLS
    win_feature_analysis.py    # NEW: SHAP/gain analysis functions (FEAT-01)
scripts/
  analyze_feature_importance.py # NEW: CLI entry point for running analysis
tests/
  test_win_feature_analysis.py  # NEW: tests for analysis functions
  test_horse_history_features.py # EXTEND: tests for new features (already exists)
```

### Pattern 1: SHAP Analysis via LightGBM Native Support
**What:** Use LightGBM's built-in `pred_contrib` to get SHAP values without external dependencies.
**When to use:** FEAT-01 -- computing feature importance for the win hit model.
**Example:**
```python
# Source: LightGBM 4.6 docs (lightgbm.readthedocs.io) - Context7 verified
# After training WinTwoStageModel.hit_model (lgb.Booster):

import numpy as np
import pandas as pd

def analyze_feature_importance(model, features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate both gain and SHAP-based feature importance rankings.

    Args:
        model: lgb.Booster (e.g., win_2s.hit_model)
        features_df: DataFrame of features used for prediction

    Returns:
        DataFrame with feature, gain, mean_abs_shap columns, sorted by SHAP.
    """
    feature_names = model.feature_name()

    # 1. Gain-based importance (fast, built-in)
    gain = model.feature_importance(importance_type='gain')
    gain_df = pd.DataFrame({'feature': feature_names, 'gain': gain})

    # 2. SHAP values via pred_contrib
    # IMPORTANT: returns matrix with EXTRA column (expected value) at the end
    shap_matrix = model.predict(features_df, pred_contrib=True)
    # shap_matrix shape: [n_samples, n_features + 1]
    # Last column is the expected value (base value), exclude it
    shap_values = shap_matrix[:, :-1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    result = gain_df.copy()
    result['mean_abs_shap'] = mean_abs_shap
    result = result.sort_values('mean_abs_shap', ascending=False)
    return result
```

### Pattern 2: Win-Specific Feature Computation (in HorseHistoryFeatures loop)
**What:** Add new features to the existing `HorseHistoryFeatures.compute()` iteration.
**When to use:** FEAT-02 -- features that need past race history.
**Example:**
```python
# Source: codebase pattern from src/features/horse_history_features.py
# Inside the compute() loop, following the existing pattern for class_move etc.

# class_drop_bounce: horse dropping class after poor results at higher class
# (Uses existing class_move computation + recent performance)
if hist_idx >= 2 and not np.isnan(class_move) and class_move < -0.5:
    last_2_kj = horse_arrs["kakuteijyuni"][valid_mask][start:idx][-2:].astype(float)
    last_2_ss = horse_arrs["syussotosu"][valid_mask][start:idx][-2:].astype(float)
    valid_recent = last_2_ss > 1
    if valid_recent.any():
        norm_recent = (last_2_kj[valid_recent] - 1) / (last_2_ss[valid_recent] - 1)
        avg_recent = float(np.nanmean(norm_recent))
        class_drop_bounce = abs(class_move) * avg_recent if avg_recent > 0.5 else 0.0
    else:
        class_drop_bounce = float("nan")
else:
    class_drop_bounce = 0.0 if not np.isnan(class_move) else float("nan")

# distance_change: current distance_bin != last race distance_bin
if hist_idx > 0 and "distance_bin" in horse_arrs:
    current_db = str(getattr(row, "distance_bin", "unknown"))
    last_db = str(horse_arrs["distance_bin"][history_mask][hist_start:hist_idx][-1])
    distance_change = 1.0 if current_db != last_db else 0.0
else:
    distance_change = float("nan")

# surface_change: current surface != last race surface
if hist_idx > 0 and "surface" in horse_arrs:
    current_surf = str(getattr(row, "surface", ""))
    last_surf = str(horse_arrs["surface"][history_mask][hist_start:hist_idx][-1])
    surface_change = 1.0 if current_surf != last_surf else 0.0
else:
    surface_change = float("nan")

# win_dominance: average field size of recent wins (proxy for win quality)
if n_past > 0:
    win_mask = hp_kakuteijyuni == 1
    if win_mask.any():
        win_sizes = hp_syussotosu[win_mask].astype(float)
        valid_sizes = win_sizes[~np.isnan(win_sizes) & (win_sizes > 0)]
        win_dominance = float(np.mean(valid_sizes)) if len(valid_sizes) > 0 else float("nan")
    else:
        win_dominance = 0.0  # no wins -- signal, not NaN
else:
    win_dominance = float("nan")

# freshness_score: rest quality * recent form quality
if not np.isnan(days_since) and n_past >= 3:
    # Rest score (optimal around 30-60 days)
    if days_since <= 7:
        rest_score = 0.3
    elif days_since <= 30:
        rest_score = 0.7
    elif days_since <= 60:
        rest_score = 1.0
    elif days_since <= 90:
        rest_score = 0.8
    else:
        rest_score = 0.4
    # Form score from recent 3 starts
    recent_kj = hp_kakuteijyuni[-3:].astype(float)
    recent_ss = hp_syussotosu[-3:].astype(float)
    valid_recent = recent_ss > 1
    if valid_recent.any():
        norm_pos = (recent_kj[valid_recent] - 1) / (recent_ss[valid_recent] - 1)
        form_score = 1.0 - float(np.nanmean(norm_pos))
        freshness_score = rest_score * max(form_score, 0.0)
    else:
        freshness_score = float("nan")
else:
    freshness_score = float("nan")
```

### Pattern 3: Odds-to-Ability Ratio (Late-Stage Feature)
**What:** Compute the ratio of market-implied probability to model-estimated ability.
**When to use:** FEAT-02 -- the single most important ROI signal for value betting.
**Example:**
```python
# This feature MUST be computed after p_ability_win is available (post-AbilityModel)
# Best location: inside _train_submodel() after AbilityModel.train_oof() runs
# OR as a derived feature in WinTwoStageModel._prepare_features()

# Compute:
if "p_market_win_adj" in df.columns and "p_ability_win" in df.columns:
    p_market = df["p_market_win_adj"].clip(lower=1e-6)
    p_ability = df["p_ability_win"].clip(lower=1e-6)
    df["odds_to_ability_ratio"] = (p_market / p_ability).clip(0.1, 10.0)

# Values > 1.0: market overvalues horse relative to ability (potential fade)
# Values < 1.0: market undervalues horse (potential value bet)
# Values ~ 1.0: market and model agree
```

### Anti-Patterns to Avoid
- **Removing features from shared computation:** Only remove from `WinTwoStageModel.FEATURE_COLS`. The feature computation modules serve place/wide models too. Removing computation would break other models.
- **SHAP analysis on the return model (E(odds|win)):** The return model trains only on winners (~7% of data), making SHAP unreliable. Focus FEAT-01 analysis on the hit model (P(win) binary classifier) which uses all samples.
- **Feature importance without retraining validation:** SHAP identifies which features the model uses, but removing low-importance features can still change model behavior. Always retrain and compare AUC/logloss after removal.
- **Computing odds_to_ability in FeatureEngine.build_all():** p_ability_win is not available at that point -- it is computed by AbilityModel later in the pipeline. This feature must be computed either (a) in `WinTwoStageModel._prepare_features()`, or (b) as a late-stage addition in `_train_submodel()` after AbilityModel produces p_ability_win.
- **Using confirmed_odds (post-race) in features:** The codebase handles this via `confirmed_odds` vs `tanodds` distinction. New features MUST use pre-race odds (`tanodds`) only. [VERIFIED: feature_engine.py lines 139-143]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SHAP value computation | Custom SHAP implementation | LightGBM `predict(pred_contrib=True)` | Native support in LightGBM 4.6. No external dependency needed. Handles edge cases, optimized for tree models. [VERIFIED: Context7 LightGBM docs] |
| Feature importance ranking | Custom permutation importance | LightGBM `feature_importance('gain')` + native SHAP | Built-in, well-tested, no additional dependencies. |
| AUC/logloss computation | Custom metric functions | sklearn.metrics.roc_auc_score, log_loss | Handles edge cases (tie-breaking, class imbalance). |
| Leakage detection for new features | Manual date checking | features.leakage_validators.validate_no_future_leakage | Already exists, enforces expanding().shift(1) semantics. |
| Class level computation | Custom grade/jyoken mapping | FeatureEngine._compute_class_level() + _class_level_from_values() | Already handles hierarchical grade/jyoken fallback. |

**Key insight:** The project already has a sophisticated feature engineering pipeline with 14 modules and a leakage validation framework. New features should slot into the existing `HorseHistoryFeatures.compute()` loop pattern, which already handles PIT (point-in-time) safety via `searchsorted` on `target_date`.

## Common Pitfalls

### Pitfall 1: SHAP pred_contrib Extra Column
**What goes wrong:** `model.predict(data, pred_contrib=True)` returns a matrix with shape `[n_samples, n_features + 1]`. The last column is the expected value (base value), not a feature contribution. Including it in the analysis corrupts the ranking.
**Why it happens:** LightGBM documentation states this clearly but the gotcha is easy to miss. [VERIFIED: Context7 LightGBM docs -- "we return a matrix with an extra column, where the last column is the expected value"]
**How to avoid:** Always slice `shap_matrix[:, :-1]` before computing feature importance. Verify with `assert shap_matrix.shape[1] == len(feature_names) + 1`.
**Warning signs:** SHAP ranking has one extra "feature" with very high contribution.

### Pitfall 2: odds_to_ability Timing in Pipeline
**What goes wrong:** Computing `odds_to_ability_ratio` in `FeatureEngine.build_all()` fails because `p_ability_win` does not exist yet. It is computed by `AbilityModel.add_ability_probs()` which runs later in `_train_submodel()`.
**Why it happens:** The feature pipeline has a strict ordering: FeatureEngine -> HorseHistoryFeatures -> InteractionFeatures -> MarketModel -> AbilityModel -> WinTwoStageModel. `p_ability_win` appears only after AbilityModel runs.
**How to avoid:** Compute `odds_to_ability_ratio` in `WinTwoStageModel._prepare_features()` or as a pre-step in `train_hit_model()`, where both `p_market_win_adj` and `p_ability_win` are available.
**Warning signs:** KeyError for 'p_ability_win' during feature computation.

### Pitfall 3: Feature Removal Breaking Place/Wide Models
**What goes wrong:** Removing a feature from the shared computation modules (e.g., `HorseHistoryFeatures.BASE_COLS`) because SHAP says it is noise for win prediction, but the feature is important for place/wide models.
**Why it happens:** The feature computation path is shared across all bet types. SHAP analysis is win-specific.
**How to avoid:** Only remove features from `WinTwoStageModel.FEATURE_COLS`. Never remove from `HorseHistoryFeatures.BASE_COLS` or shared computation modules. The shared computation can and should produce features that individual models choose not to use.
**Warning signs:** Place/wide model AUC drops after feature removal.

### Pitfall 4: Distance/Surface Change Needs distance_bin in History
**What goes wrong:** Computing `distance_change` requires `distance_bin` in the past_df used by HorseHistoryFeatures. The `distance_bin` column is computed in `_map_basic_features()` from `kyori` + `surface`, but this computation must also happen in `HorseHistoryFeatures.compute()` where past_df is built from raw entries+races.
**Why it happens:** The past_df in HorseHistoryFeatures.compute() is built independently from the main feature pipeline (lines 351-375). It already computes distance_bin from kyori + surface (line 362-376).
**How to avoid:** The codebase already computes distance_bin for past_df (verified at line 362-376). The new feature can safely reference it. But verify it exists before computing.
**Warning signs:** distance_change is always NaN because distance_bin is missing from past_df.

### Pitfall 5: Class Drop Bounce Requiring Multiple Past Races
**What goes wrong:** The class_drop_bounce feature needs at least 2 past races at a higher class level to determine if the horse is "dropping after poor results." With fewer than 2 past races, the feature should be NaN.
**Why it happens:** New horses or lightly raced horses have insufficient history.
**How to avoid:** Check `hist_idx >= 2` before computing. Return NaN for insufficient data, consistent with how `form_trend` handles the same case.
**Warning signs:** Class drop bounce is 0 for most horses (should be NaN for insufficient data).

### Pitfall 6: Win Dominance All-Zeros for Non-Winners
**What goes wrong:** Setting `win_dominance = 0.0` for horses with no recent wins creates a strong categorical split (0 vs NaN) rather than a continuous signal. This can confuse LightGBM into treating it as a binary feature.
**Why it happens:** 0.0 has meaning (no wins) while NaN means insufficient data. The distinction is intentional but creates a bimodal distribution.
**How to avoid:** Consider using NaN for non-winners too, or create a separate `has_recent_win` binary feature alongside a continuous `win_margin` feature. Let the planner decide.
**Warning signs:** win_dominance has 90%+ zeros, making it effectively binary.

## Code Examples

Verified patterns from official sources and codebase:

### LightGBM Feature Importance (Gain + SHAP)
```python
# Source: LightGBM 4.6 docs (lightgbm.readthedocs.io) via Context7
# Works on lgb.Booster objects (WinTwoStageModel.hit_model is lgb.Booster)

import numpy as np
import pandas as pd
import lightgbm as lgb

def analyze_feature_importance(model: lgb.Booster, features_df: pd.DataFrame) -> pd.DataFrame:
    """Generate both gain and SHAP-based feature importance rankings.

    Args:
        model: lgb.Booster (e.g., win_2s.hit_model)
        features_df: DataFrame of features used for prediction

    Returns:
        DataFrame with feature, gain, mean_abs_shap columns, sorted by SHAP.
    """
    feature_names = model.feature_name()

    # Gain-based importance
    gain = model.feature_importance(importance_type='gain')
    gain_df = pd.DataFrame({'feature': feature_names, 'gain': gain})

    # SHAP values via pred_contrib
    shap_matrix = model.predict(features_df, pred_contrib=True)
    assert shap_matrix.shape[1] == len(feature_names) + 1, "pred_contrib returns n_features+1 columns"
    shap_values = shap_matrix[:, :-1]  # drop expected value column
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    result = gain_df.copy()
    result['mean_abs_shap'] = mean_abs_shap
    result = result.sort_values('mean_abs_shap', ascending=False)
    return result
```

### SHAP Summary Bar Plot (matplotlib, no shap package)
```python
# Source: standard matplotlib pattern
import matplotlib.pyplot as plt

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20) -> None:
    """Plot SHAP-based feature importance as horizontal bar chart."""
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top['feature'][::-1], top['mean_abs_shap'][::-1])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Top {top_n} Features by SHAP Importance (Win Model)')
    plt.tight_layout()
    plt.savefig('shap_importance_win.png', dpi=150)
```

### Integration Point: Adding Features to HorseHistoryResults
```python
# Source: codebase pattern from horse_history_features.py lines 966-994
# New features are added to the results.append() dict at the end of the compute() loop

results.append(
    {
        "race_id": row.race_id,
        "umaban": row.umaban,
        # ... existing features ...
        # NEW win-specific features:
        "class_drop_bounce": class_drop_bounce,
        "win_dominance": win_dominance,
        "distance_change": distance_change,
        "surface_change": surface_change,
        "freshness_score": freshness_score,
    }
)

# Also add to BASE_COLS class attribute at line 240-266:
BASE_COLS: list[str] = [
    # ... existing cols ...
    "class_drop_bounce",
    "win_dominance",
    "distance_change",
    "surface_change",
    "freshness_score",
]
```

### Integration Point: Adding odds_to_ability_ratio
```python
# Source: codebase pattern from _train_submodel() in training_pipeline.py
# Best location: after AbilityModel.train_oof() produces p_ability_win (line 407)
# and before WinTwoStageModel.train_hit_model() uses it (line 443)

# In _train_submodel(), after line 409 (df_oof = df[oof_mask].copy()):
if "p_market_win_adj" in df_oof.columns and "p_ability_win" in df_oof.columns:
    p_market = df_oof["p_market_win_adj"].clip(lower=1e-6)
    p_ability = df_oof["p_ability_win"].clip(lower=1e-6)
    df_oof["odds_to_ability_ratio"] = (p_market / p_ability).clip(0.1, 10.0)

# Then add to WinTwoStageModel.FEATURE_COLS (line 44-77):
FEATURE_COLS: list[str] = [
    # ... existing cols ...
    "odds_to_ability_ratio",  # NEW
]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Permutation importance | SHAP values (TreeSHAP) | ~2018+ | Faster, consistent feature attributions for tree models. LightGBM has native support via pred_contrib. |
| Gain-based only | Gain + SHAP combined | ~2020+ | Gain is biased toward high-cardinality features; SHAP provides theoretically consistent rankings. Use both. |
| shap package required | LightGBM native pred_contrib | LightGBM 3.x+ | No external dependency needed for SHAP values. shap package only needed for advanced visualizations. |
| Remove features by importance threshold | Retrain and compare metrics | Standard ML practice | Importance alone does not predict impact of removal. Low-importance features may have synergistic value. |

**Deprecated/outdated:**
- Split importance (count-based): misleading for high-cardinality features. Always prefer `importance_type='gain'`.
- shap package as required dependency for basic SHAP: LightGBM native pred_contrib is sufficient for ranking.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `odds_to_ability_ratio` should be computed inside `_train_submodel()` after AbilityModel produces p_ability_win, not in FeatureEngine, because p_ability_win is not available at FeatureEngine time. | Architecture Patterns | If wrong, feature could be computed earlier, simplifying the pipeline. |
| A2 | LightGBM native `pred_contrib=True` produces valid SHAP values for binary classification models. [VERIFIED: Context7 LightGBM docs] | Standard Stack | Low risk -- documented and verified. |
| A3 | `p_market_win_adj` is available in `df_oof` by the time `_train_submodel()` reaches the odds_to_ability computation point, because `compute_market_bias()` runs in `FeatureEngine.build_all()`. [VERIFIED: traced through code flow] | Code Examples | If wrong, odds_to_ability_ratio needs different computation timing. |
| A4 | Removing noise features from `WinTwoStageModel.FEATURE_COLS` will not affect the EVCorrectionModel, which has its own separate `FEATURE_COLS` list (30 features, different set). [VERIFIED: ev_correction_model.py line 135-168] | Common Pitfalls | If wrong, EVCorrectionModel features may also need pruning. |
| A5 | The freshness_score feature can be derived from `days_since_last_race` and recent finish positions, both already computed in HorseHistoryFeatures. No additional data columns needed. [VERIFIED: horse_history_features.py has both available in compute() loop] | FEAT-02 | Low risk. |

## Open Questions (All Resolved)

1. **Odds-to-ability computation location**
   - What we know: p_ability_win is computed by AbilityModel in _train_submodel(), p_market_win_adj is computed by FeatureEngine.build_all() -> compute_market_bias().
   - What's unclear: Whether p_market_win_adj persists in df through the HorseHistoryFeatures merge and AbilityModel steps without being dropped.
   - Recommendation: Add odds_to_ability_ratio computation inside `_train_submodel()` after AbilityModel.train_oof() (line 407) and verify both columns exist with an assertion.
   - **RESOLVED**: Compute in _train_submodel() for training path (Plan 01-02 Task 2). For inference path, compute in WinTwoStageModel._prepare_features() when the column is missing but p_market_win_adj and p_ability_win are present (confirmed available in race_predictor.py lines 89-97). Dual-path approach adopted.

2. **SHAP analysis scope -- which models to analyze**
   - What we know: WinTwoStageModel has hit_model (P(win)) and return_model (E(odds|win)). EVCorrectionModel has p_correction_model and e_correction_model.
   - What's unclear: Whether FEAT-01 should analyze all win-related models or just the hit_model.
   - Recommendation: Focus on WinTwoStageModel.hit_model as primary. The return_model has too few samples (~7% winners) for reliable SHAP. EVCorrectionModel features are secondary corrections.
   - **RESOLVED**: FEAT-01 analyzes WinTwoStageModel.hit_model only. Return_model and EVCorrectionModel excluded from SHAP scope per anti-pattern guidance in research.

3. **Win dominance definition**
   - What we know: FEAT-02 mentions "win dominance" and the research summary says it should measure "how decisively the horse wins when it wins."
   - What's unclear: Whether win dominance should be based on (a) field size of winning races, (b) timediff (margin) in winning races, or (c) some other metric.
   - Recommendation: Use average field size of recent winning races as primary metric (proxy for win quality -- winning in larger fields is harder). Timediff is often NaN (from pitfall: harontimel3/harontime columns have high NaN rates). Document the definition choice for planner.
   - **RESOLVED**: Definition = average field size (syussotosu) of past winning races. Non-winners with history get 0.0 (meaningful signal). No history = NaN. Timediff avoided due to high NaN rate.

4. **Feature importance stability across surface submodels**
   - What we know: Turf and dirt models are trained independently; feature importance may differ.
   - What's unclear: Whether noise features are consistent across both surface models.
   - Recommendation: WinTwoStageModel is NOT split by surface (unlike AbilityModel). The hit_model uses all data. Analyze the single model. If planner wants surface-specific analysis, that would be a separate step.
   - **RESOLVED**: Single combined model analysis only. WinTwoStageModel.hit_model trains on all surfaces together. Surface-specific analysis not needed for this phase.

## Environment Availability

Step 2.6: SKIPPED (no external dependencies identified -- all required tools are Python packages already installed in the project environment)

## Key Codebase Integration Points

The following table documents exactly where new code must integrate with existing code:

| Integration Point | File | What to Modify |
|-------------------|------|----------------|
| New feature functions | `src/features/horse_history_features.py` `compute()` | Add 5 new features to the iteration loop + BASE_COLS |
| Late-stage feature | `src/pipelines/training_pipeline.py` `_train_submodel()` | Add odds_to_ability_ratio after AbilityModel.train_oof() (after line 409) |
| Win model feature list | `src/models/two_stage_return_model.py` `WinTwoStageModel.FEATURE_COLS` | Add new feature column names (line 44-77) |
| Feature importance analysis | `scripts/analyze_feature_importance.py` (NEW) | Create script to run SHAP + gain analysis |
| Tests | `tests/test_horse_history_features.py` (existing) | Extend with test cases for new features |

### Data Availability for New Features

| New Feature | Source Data | Available Columns | Computation Location |
|-------------|------------|-------------------|---------------------|
| odds_to_ability_ratio | Feature DataFrame after Stage1 | `p_market_win_adj`, `p_ability_win` | _train_submodel() after line 409 [VERIFIED] |
| class_drop_bounce | History entries + races | `gradecd`, `jyokencd1`, `kakuteijyuni`, `syussotosu` | HorseHistoryFeatures.compute() [VERIFIED: cols at line 408-427] |
| distance_change | History races | `distance_bin` (computed from kyori+surface at line 362-376) | HorseHistoryFeatures.compute() [VERIFIED] |
| surface_change | History races | `surface` (computed at line 156-158 in readers.py) | HorseHistoryFeatures.compute() [VERIFIED] |
| win_dominance | History entries | `kakuteijyuni`, `syussotosu` | HorseHistoryFeatures.compute() [VERIFIED] |
| freshness_score | History entries | `race_date`, `kakuteijyuni`, `syussotosu` + existing `days_since` | HorseHistoryFeatures.compute() [VERIFIED] |

## Sources

### Primary (HIGH confidence)
- LightGBM 4.6.0 documentation (lightgbm.readthedocs.io) via Context7 - `predict(pred_contrib=True)`, `feature_importance()`, confirmed extra column in pred_contrib output [VERIFIED]
- Codebase: `src/features/horse_history_features.py` - existing feature computation patterns (itertuples + searchsorted) [VERIFIED: file read]
- Codebase: `src/models/two_stage_return_model.py` - WinTwoStageModel.FEATURE_COLS (27 features), PlaceTwoStageModel.HIT_FEATURE_COLS (45 features) [VERIFIED: file read]
- Codebase: `src/models/stage1_ability_model.py` - AbilityModel.FEATURE_COLS (54 features) [VERIFIED: file read]
- Codebase: `src/pipelines/training_pipeline.py` - pipeline ordering, model training sequence, _train_submodel() [VERIFIED: file read]
- Codebase: `src/features/feature_engine.py` - build_all() ordering, _map_basic_features() [VERIFIED: file read]
- Codebase: `src/features/odds_dynamics_features.py` - compute_odds_dynamics() [VERIFIED: file read]
- Codebase: `src/features/market_bias_features.py` - compute_market_bias() produces p_market_win_adj [VERIFIED: file read]
- Codebase: `src/models/ev_correction_model.py` - EVCorrectionModel.FEATURE_COLS (30 features, separate list) [VERIFIED: file read]

### Secondary (MEDIUM confidence)
- LightGBM community discussion on SHAP vs gain importance for feature selection [CITED: reddit.com/r/datascience]
- SHAP GitHub repository documentation on TreeSHAP algorithm [CITED: github.com/shap/shap]
- Machine Learning Mastery tutorial on SHAP for tree-based models [CITED: machinelearningmastery.com]

### Tertiary (LOW confidence)
- None -- all key claims verified from codebase or official docs.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries installed and verified via pip list. No new dependencies needed.
- Architecture: HIGH - pipeline ordering confirmed by reading training_pipeline.py, data flow traced through 5 key files.
- Pitfalls: HIGH - all pitfalls derived from direct codebase analysis and LightGBM official documentation.
- New feature feasibility: HIGH - raw data columns verified present in entries/races Parquet files. Computation locations identified with line numbers.

**Research date:** 2026-05-02
**Valid until:** 2026-06-02 (30 days -- stable domain, no fast-moving dependencies)
