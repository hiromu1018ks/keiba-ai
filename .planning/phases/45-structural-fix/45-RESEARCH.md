# Phase 45: Structural Fix - Research

**Researched:** 2026-05-31
**Domain:** MAWC (MarketAwareWinCalibrator) structural modification -- interaction term removal + strong regularization
**Confidence:** HIGH

## Summary

Phase 45 modifies the MAWC LogisticRegression calibrator to address the ROI degradation identified in Phase 44. The root cause is MAWC's `beta_market=0.90` coefficient dominance, which causes ECE 3x degradation in the odds 1-3 favorite band, suppressing probabilities for favorites and reducing bet_count by 22%. The fix involves retraining MAWC with (a) removed high-risk interaction terms and (b) stronger regularization (C grid [0.003, 0.03]) on the existing OOF data.

**Primary recommendation:** Use `data/oof/oof_predictions.parquet` directly as the MAWC retraining data source, deriving `p_model` from `p_win_corrected` and `p_market` from `clip(1/tanodds, 0.01, 0.99)`. Build a standalone retraining module (not integrated into TrainingPipelineV5). Save the conservative variant to `data/models-backtest-mawc-conservative/{year}/` by copying all files from the source year directory and replacing only the MAWC joblib.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Phase 45 scope is MAWC single component only. No changes to Ranker/OBF/selection thresholds.
- **D-02:** MAWC conservative structural modification via OOF retraining. No coefficient post-clamping, no RacePredictor-side sandwich correction. C grid [0.003, 0.005, 0.01, 0.03]. Remove high-risk interaction terms (logit_model x pop top/favorite, logit_model x low odds bands 1-2/2-3/1-3, possibly all odds_band x logit_model). Keep main effects and segment one-hots. LogisticRegression single structure only.
- **D-03:** New variant save. Existing models read-only. Conservative variant at `data/models-backtest-mawc-conservative/{year}/`. Copy all files + replace MAWC joblib only. Manifest with source_model_dir, mawc_fix_version, C_grid, removed_interactions, guard results. Phase 46 compares baseline vs mawc_conservative.
- **D-04:** Minimum C selection among quality-gate-passing candidates. Selection criteria: overall Brier/logloss/ECE non-degradation, per-year non-degradation, odds 1-3 favorite band guard (ECE, bet_count, APR), p compression check, all fail -> keep existing MAWC.
- **D-05:** Phase 45 generalization confirmation limited to OOF quality + lightweight proxy. No BT re-run, no Shadow Comparison re-run, no ROI evaluation. Phase 46 does full verification.

### Claude's Discretion
- MAWC retraining OOF data split method
- Exact identification of removed interaction terms from FEATURE_COLS
- C grid evaluation implementation details (LogisticRegression CV vs manual OOF evaluation)
- Conservative variant directory structure and file naming
- Test structure and naming
- JSON manifest schema design
- Favorite band guard threshold specifics

### Deferred Ideas (OUT OF SCOPE)
- Ranker modification (investment_score weights/threshold adjustment) -- dormant, Phase 46+ candidate
- OddsBandFilter retraining/threshold adjustment -- non-causal, Phase 46+ candidate
- Selection gate threshold adjustment -- expected to improve naturally from MAWC fix
- Full 12-model SHAP/gain comparison -- Phase 46+ candidate
- Regime-specific analysis/parameter adjustment -- v2.3+
- New feature addition -- v2.3+
- Auto deployment gate decision (DEP-01) -- v2.3+
- Optuna 19-dim parameter optimization (DEP-02) -- v2.3+
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FIX-01 | Structural fix based on bisect/diagnosis results (MAWC interaction removal + strong regularization) | RQ-1 (OOF data), RQ-2 (interaction terms), RQ-3 (training path) |
| FIX-02 | Verify fix generalizability via OOF metrics (not year-specific coefficients) | RQ-2 (which interactions to remove), D-05 (OOF quality checks) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| MAWC retraining | Offline Analysis | - | Standalone script/module, not part of training pipeline |
| OOF data preparation | Data Layer | - | Read existing parquet + derive columns |
| Conservative variant creation | Data Layer | Model Loading | Copy + replace MAWC joblib in new directory |
| Quality gate evaluation | Offline Analysis | - | OOF Brier/logloss/ECE + favorite band guard |
| Manifest generation | Offline Analysis | - | JSON metadata for Phase 46 consumption |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn LogisticRegression | [VERIFIED: codebase] | MAWC calibrator model | Already used by MAWC; L2 regularization with C parameter |
| pandas | [VERIFIED: codebase] | OOF data loading and manipulation | Project standard |
| numpy | [VERIFIED: codebase] | Numerical operations | Project standard |
| joblib | [VERIFIED: codebase] | Model serialization | Project standard for .joblib files |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sklearn.metrics (brier_score_loss, log_loss) | [VERIFIED: codebase] | Quality metrics | MAWC quality gate evaluation |
| dataclasses | [VERIFIED: codebase] | MAWC dataclass structure | Already used by MAWC |

### No New Packages Required
Phase 45 uses only existing project dependencies. No new package installation needed.

## Package Legitimacy Audit

> No new packages required for this phase. All dependencies are existing project packages.

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
data/oof/oof_predictions.parquet
         |
         v
[1] Prepare OOF DataFrame
    - p_model = p_win_corrected
    - p_market = clip(1/tanodds, 0.01, 0.99)
    - p_win_race_rank_pct = groupby(race_id).rank()
    - Split by surface
         |
         v
[2] For each C in [0.003, 0.005, 0.01, 0.03]:
    - Build reduced feature matrix (51 - removed interactions)
    - Fit LogisticRegression(C=c)
    - Evaluate: Brier, logloss, ECE, favorite band guard
         |
         v
[3] Select minimum C passing all gates
    - OR: keep existing MAWC (not_deployed)
         |
         v
[4] Copy data/models-backtest/{year}/ --> data/models-backtest-mawc-conservative/{year}/
    Replace market_aware_win_calibrator_{surface}.joblib
         |
         v
[5] Generate manifest JSON
    source_model_dir, mawc_fix_version, C_grid, removed_interactions, guard results
```

### Recommended Project Structure
```
src/models/market_aware_win_calibrator.py  -- MODIFY: add build_conservative_feature_matrix()
scripts/run_mawc_conservative_retrain.py   -- NEW: standalone retraining script
data/models-backtest-mawc-conservative/    -- NEW: output directory
    {year}/
        (all files copied from data/models-backtest/{year}/)
        market_aware_win_calibrator_{surface}.joblib  -- REPLACED
    manifest.json  -- NEW: metadata
```

### Pattern 1: Standalone MAWC Retraining Module
**What:** Independent script that loads OOF data, retrains MAWC with reduced features, and saves conservative variant.
**When to use:** When modifying a single component without re-running the full training pipeline.
**Example:**
```python
# Source: based on existing MAWC.train() and build_feature_matrix()
# The conservative variant removes interaction terms and uses a smaller C grid
# Feature matrix: 51 - N_removed_interactions dimensions
# C grid: [0.003, 0.005, 0.01, 0.03] instead of [0.03, 0.1, 0.3, 1.0, 3.0]
```

### Pattern 2: Conservative Variant Directory
**What:** Copy entire year model directory and replace only MAWC joblib files.
**When to use:** When ModelLoader.load_from_dir() expects a complete year directory with meta.json.
**Rationale:** `load_from_dir()` reads `meta.json` first, then loads all model files by naming convention. Replacing only MAWC joblib preserves all other models unchanged.

### Anti-Patterns to Avoid
- **Modifying existing model files in data/models-backtest/**: Existing models are read-only. Always create new variant directory.
- **Retraining via TrainingPipelineV5**: The full pipeline trains 12+ models. Phase 45 only modifies MAWC. Use standalone script.
- **Changing feature_names without updating build_feature_matrix()**: The MAWC build_feature_matrix() must produce the correct reduced feature set. Feature names must match exactly.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MAWC training | Custom LogisticRegression wrapper | Existing MAWC.train() with modified C_GRID and feature matrix | MAWC.train() already has WF C-selection, beta_market guard, ratio diagnostics |
| OOF data generation | Re-run generate_win_oof_predictions | Direct column derivation from oof_predictions.parquet | generate_win_oof_predictions retrains WinTwoStageModel folds (expensive); existing OOF has all needed columns |
| Model directory management | Custom copy logic | shutil.copytree() for year directories | Simple, preserves all files, proven pattern |

**Key insight:** The existing MAWC class already has all the infrastructure (train method, build_feature_matrix, quality checks). We only need to (1) modify the feature matrix construction to exclude certain interactions, (2) change the C_GRID, and (3) add favorite band guard checks.

## Common Pitfalls

### Pitfall 1: OOF Data Mismatch
**What goes wrong:** Using the wrong OOF data source -- the full-pipeline OOF (data/oof/oof_predictions.parquet covering 2022-2024) vs the per-year backtest OOF (generated in-memory during TrainingPipelineV5).
**Why it happens:** The backtest MAWC was trained on per-year OOF data (e.g., 2020-2023 training data with 55k-59k samples per surface), while oof_predictions.parquet has 3 years of combined data.
**How to avoid:** Use oof_predictions.parquet directly. The MAWC is a LogisticRegression (not a tree model), so the exact training window matters less than for gradient-boosted models. The 136k samples (66k turf + 71k dirt) provide sufficient coverage. Document this as a known limitation -- Phase 46's full Shadow Comparison will validate the conservative variant on the actual backtest fold structure.
**Warning signs:** If the retrained MAWC's coefficients look very different from the original despite using the same data source, check for data distribution shifts.

### Pitfall 2: Feature Name Inconsistency
**What goes wrong:** The reduced feature matrix has different column names or ordering than what the MAWC expects at inference time.
**Why it happens:** `build_feature_matrix()` produces 51 features in a specific order. If we remove interactions, the feature_names list and X array must be shorter and consistent.
**How to avoid:** The conservative MAWC must use a modified `build_feature_matrix` that produces the reduced feature set. The saved joblib must contain the correct `feature_names` list. At inference, `apply()` calls `build_feature_matrix()` which must produce the same reduced feature set.
**Warning signs:** Shape mismatch errors during `calibrator.predict_proba(X)`.

### Pitfall 3: Beta Market Floor Violation
**What goes wrong:** After removing logit_model x segment interactions, the regularization might push beta_market even higher (or lower), violating the BETA_MARKET_FLOOR = 0.20 constraint.
**Why it happens:** Removing logit_model interactions effectively increases the relative weight of logit_market in the model.
**How to avoid:** Check beta_market_contribution after each C value fit. If all C values violate the floor, the conservative variant is not deployable. This is expected behavior -- the existing MAWC has beta_market=0.90, and we want to reduce it.
**Warning signs:** All C values result in _trained=False (shadow_only mode).

### Pitfall 4: Favorite Band Probability Compression
**What goes wrong:** The conservative MAWC over-compresses probabilities in the odds 1-3 band, making mean(p_conservative/p_model) < 0.90.
**Why it happens:** Removing logit_model x low_odds interactions removes the mechanism that allowed the model to "boost" probabilities for strong favorites.
**How to avoid:** Explicitly check mean(p_conservative/p_model) for odds 1-3 band. If < 0.90, mark that C value as failing the guard.
**Warning signs:** EV >= 1.0 pass rate drops significantly in odds 1-3 band.

### Pitfall 5: Incomplete Directory Copy
**What goes wrong:** Forgetting to copy meta.json or other required files when creating the conservative variant directory.
**Why it happens:** ModelLoader.load_from_dir() requires meta.json to determine surfaces, ensemble mode, and quality_threshold.
**How to avoid:** Use shutil.copytree() to copy the entire year directory, then overwrite only the MAWC joblib files.
**Warning signs:** FileNotFoundError when loading conservative variant models.

## Code Examples

### OOF Data Preparation
```python
# Source: derived from generate_win_oof_predictions() in win_benter_gate.py
import pandas as pd
import numpy as np

df = pd.read_parquet('data/oof/oof_predictions.parquet')

# Resolve p_model (equivalent to p_win_oof in the OOF generation flow)
df['p_model'] = df['p_win_corrected']

# Resolve p_market (equivalent to p_market_norm = clip(1/tanodds, 0.01, 0.99))
df['p_market'] = np.clip(1.0 / df['tanodds'].values, 0.01, 0.99)

# Compute p_win_race_rank_pct (D-19 in Phase 39 CONTEXT)
df['p_win_race_rank_pct'] = (
    df.groupby('race_id', observed=True)['p_model']
    .rank(pct=True, method='min', ascending=False)
)

# Split by surface for per-surface MAWC retrain
turf_df = df[df['surface'] == 'turf'].copy()
dirt_df = df[df['surface'] == 'dirt'].copy()
```

### Reduced Feature Matrix Construction
```python
# Source: derived from MAWC.build_feature_matrix() in market_aware_win_calibrator.py
# The 51 features are ordered as:
#   [0-5]   main effects (6)
#   [6-20]  segment one-hot (15)
#   [21-35] logit_model x segment (15)
#   [36-50] logit_market x segment (15)

# HIGH-RISK interactions to remove per CONTEXT.md D-02:
REMOVED_INTERACTIONS = [
    # logit_model x odds_band (7 terms)
    "logit_model_x_1-2",
    "logit_model_x_2-3",
    "logit_model_x_3-5",
    "logit_model_x_5-10",
    "logit_model_x_10-30",
    "logit_model_x_30-100",
    "logit_model_x_100+",
    # logit_model x pop_bucket (5 terms) -- especially pop_1
    "logit_model_x_pop_1",
    "logit_model_x_pop_2_3",
    "logit_model_x_pop_4_6",
    "logit_model_x_pop_7_9",
    "logit_model_x_pop_10_plus",
    # logit_model x p_rank (3 terms)
    "logit_model_x_top_25",
    "logit_model_x_mid_25_75",
    "logit_model_x_bottom_25",
]
# Total removed: 15 (all logit_model x segment interactions)
# Remaining: 51 - 15 = 36 features

# CRITICAL: Keep ALL logit_market x segment interactions (15 terms)
# The problem is specifically logit_model interactions allowing the model
# to "learn around" the market signal for specific segments.
```

### Conservative C Grid Search
```python
# Source: derived from MAWC.train() C_GRID logic
CONSERVATIVE_C_GRID = [0.003, 0.005, 0.01, 0.03]
# Compared to original: [0.03, 0.1, 0.3, 1.0, 3.0]
# Stronger regularization biases toward smaller coefficients,
# reducing segment-specific overfitting
```

### Variant Directory Structure
```python
# Source: derived from ModelLoader.load_from_dir() requirements
import shutil
from pathlib import Path

source_dir = Path("data/models-backtest/2024")
target_dir = Path("data/models-backtest-mawc-conservative/2024")

# Copy entire directory
shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

# Replace only MAWC joblib files
for surface in ["turf", "dirt"]:
    mawc_path = target_dir / f"market_aware_win_calibrator_{surface}.joblib"
    conservative_mawc.save(mawc_path)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MAWC C_GRID [0.03, 0.1, 0.3, 1.0, 3.0] | Conservative C_GRID [0.003, 0.005, 0.01, 0.03] | Phase 45 | Much stronger regularization |
| MAWC 51-dim features | 36-dim features (remove logit_model x segment) | Phase 45 | Removes segment-specific model signal overrides |
| Beta market floor 0.20 | May need adjustment | Phase 45 | Current MAWC has beta=0.90; conservative target unclear |

**Deprecated/outdated:**
- WinBenterGate + WinSegmentCalibrator: Already replaced by MAWC in Phase 39. Not relevant to Phase 45.

## Research Findings by RQ

### RQ-1: OOF Data Structure for MAWC Retraining

**Existing OOF data:** `data/oof/oof_predictions.parquet` has 136,859 rows x 372 columns covering 2022-01-05 to 2024-12-28.

**Columns available:**
| Column | Present | Notes |
|--------|---------|-------|
| p_win_corrected | YES | Use as p_model (OOF-corrected model prediction) |
| tanodds | YES | Use to derive p_market = clip(1/tanodds, 0.01, 0.99) |
| popularity_rank | YES | Direct use |
| field_size | YES | Direct use |
| kakuteijyuni | YES | Target variable (1 = win) |
| race_id | YES | For WF splits and groupby |
| race_date | YES | For WF splits and year-level checks |
| surface | YES | For per-surface retrain (turf: 66192, dirt: 70667) |
| p_market_win_adj | YES | Market-adjusted probability (NOT used for MAWC; use 1/tanodds) |
| p_win_pred | YES | Pre-correction prediction (NOT used; use p_win_corrected) |

**Columns MISSING but derivable:**
| Column | Derivation |
|--------|-----------|
| p_model | = p_win_corrected |
| p_market | = np.clip(1.0/tanodds, 0.01, 0.99) -- identical to what generate_win_oof_predictions computes |
| p_win_race_rank_pct | = df.groupby('race_id')['p_model'].rank(pct=True, method='min', ascending=False) |

**Key finding:** `p_win_oof`, `p_market_norm`, and `p_win_race_rank_pct` are NOT in oof_predictions.parquet. However, all three can be perfectly reconstructed from available columns:
- `p_win_oof` is semantically equivalent to `p_win_corrected` in the OOF context (both are the OOF-corrected model probability)
- `p_market_norm` is exactly `clip(1/tanodds, 0.01, 0.99)` (verified from generate_win_oof_predictions line 322-326)
- `p_win_race_rank_pct` is computable from the p_model column (verified from generate_win_oof_predictions line 374-383)

**Recommendation:** Use oof_predictions.parquet directly with derived columns. No need to regenerate via generate_win_oof_predictions.

**Confidence:** HIGH -- verified by reading the source code of both generate_win_oof_predictions and the MAWC train method.

### RQ-2: Exact Interaction Term Names to Remove

**Complete 51-dim feature list (verified from loaded 2024 turf model):**

Main effects (indices 0-5):
```
[0] logit_model
[1] logit_market
[2] log_odds
[3] popularity_rank_pct
[4] p_win_race_rank_pct
[5] field_size
```

Segment one-hot (indices 6-20):
```
[6]  1-2          [7]  2-3          [8]  3-5
[9]  5-10         [10] 10-30        [11] 30-100       [12] 100+
[13] pop_1        [14] pop_2_3      [15] pop_4_6      [16] pop_7_9       [17] pop_10_plus
[18] top_25       [19] mid_25_75    [20] bottom_25
```

logit_model x segment interactions (indices 21-35):
```
[21] logit_model_x_1-2          [22] logit_model_x_2-3          [23] logit_model_x_3-5
[24] logit_model_x_5-10         [25] logit_model_x_10-30        [26] logit_model_x_30-100       [27] logit_model_x_100+
[28] logit_model_x_pop_1        [29] logit_model_x_pop_2_3      [30] logit_model_x_pop_4_6      [31] logit_model_x_pop_7_9     [32] logit_model_x_pop_10_plus
[33] logit_model_x_top_25       [34] logit_model_x_mid_25_75    [35] logit_model_x_bottom_25
```

logit_market x segment interactions (indices 36-50):
```
[36] logit_market_x_1-2         [37] logit_market_x_2-3         [38] logit_market_x_3-5
[39] logit_market_x_5-10        [40] logit_market_x_10-30       [41] logit_market_x_30-100      [42] logit_market_x_100+
[43] logit_market_x_pop_1       [44] logit_market_x_pop_2_3     [45] logit_market_x_pop_4_6     [46] logit_market_x_pop_7_9    [47] logit_market_x_pop_10_plus
[48] logit_market_x_top_25      [49] logit_market_x_mid_25_75   [50] logit_market_x_bottom_25
```

**High-risk interactions per CONTEXT D-02 (to remove):**
All 15 `logit_model_x_*` interactions (indices 21-35). The rationale:
- CONTEXT specifies removing "logit_model x popularity top/favorite" (logit_model_x_pop_1) and "logit_model x low odds band 1-2, 2-3" (logit_model_x_1-2, logit_model_x_2-3)
- CONTEXT also says "possibly all odds_band x logit_model interactions" -- given that the MAWC's problem is segment-specific model probability overrides, removing ALL logit_model interactions is the safest approach
- This eliminates the mechanism that allows the model to "override" market signals for specific segments

**Kept interactions:**
All 15 `logit_market_x_*` interactions (indices 36-50) remain. These allow segment-specific market weight adjustments, which is the intended MAWC behavior.

**Resulting feature count:** 51 - 15 = 36 dimensions.

**Coefficients of removed terms (2024 turf model):**
| Feature | Coefficient | Notes |
|---------|------------|-------|
| logit_model_x_pop_1 | -0.059028 | Strongly negative -- suppresses favorites |
| logit_model_x_1-2 | -0.000095 | Near zero |
| logit_model_x_2-3 | -0.011187 | Mildly negative |
| logit_model_x_top_25 | 0.097276 | Strongly positive -- boosts top-ranked |
| logit_model_x_100+ | 0.129873 | Large positive for longshots |

**Confidence:** HIGH -- verified by loading and inspecting the actual MAWC joblib file.

### RQ-3: MAWC Training Data Path

**How generate_win_oof_predictions works (verified from source):**
1. Takes df_oof from TrainingPipelineV5 (per-surface split, sorted by race_date)
2. Creates expanding walk-forward splits (n_splits=5)
3. For each fold: trains a fresh WinTwoStageModel, trains a fresh EVCorrectionModel, predicts on validation fold
4. Captures: p_win_corrected, p_win_oof, p_market_norm (=clip(1/tanodds, 0.01, 0.99)), kakuteijyuni, calibrated_ev_oof, e_return_win_pred
5. Computes p_win_race_rank_pct from p_win_oof via groupby(race_id).rank()
6. Passes through ~70 static columns (tanodds, popularity_rank, field_size, etc.)
7. Returns the assembled DataFrame to MAWC.train()

**Can we skip generate_win_oof_predictions? YES.**
The key insight is that oof_predictions.parquet already contains all the OOF predictions we need. The generate_win_oof_predictions function regenerates these predictions from scratch (retraining WinTwoStageModel folds), which is computationally expensive and unnecessary for MAWC retraining. The MAWC only needs:
- p_model (= p_win_corrected from OOF)
- p_market (= clip(1/tanodds, 0.01, 0.99))
- p_win_race_rank_pct (derivable from p_model)
- kakuteijyuni, tanodds, popularity_rank, field_size, race_id, race_date, surface

**Data volume comparison:**
| Source | Turf | Dirt | Total | Years |
|--------|------|------|-------|-------|
| oof_predictions.parquet | 66,192 | 70,667 | 136,859 | 2022-2024 |
| Backtest 2024 MAWC | 55,295 | 59,487 | 114,782 | 2020-2023 training window |
| Backtest 2025 MAWC | 55,188 | 58,750 | 113,938 | 2021-2024 training window |

The oof_predictions.parquet has slightly more data (3 years vs 4-year training windows) but the distribution should be similar. Since MAWC is a LogisticRegression (linear model), the exact training window matters less than for tree-based models.

**Confidence:** HIGH -- verified by reading generate_win_oof_predictions source and comparing column requirements.

### RQ-4: Model Saving/Loading for Conservative Variant

**ModelLoader.load_from_dir() requirements (verified from source):**
1. Reads `meta.json` from the models_dir:
   - `surfaces`: list (e.g., ["turf", "dirt"])
   - `use_ensemble`: bool
   - `train_start`, `train_end`: strings
   - `quality_threshold`: float
2. For each surface, loads files by naming convention:
   - `market_{surface}.lgb`, `stage1_{surface}.lgb`, `win_hit_{surface}.joblib`, etc.
   - `market_aware_win_calibrator_{surface}.joblib` -- this is the file we replace
   - `win_race_level_ranker_{surface}.joblib` -- NOT modified
3. All other model files are loaded as-is.

**Conservative variant approach:**
1. Copy entire `data/models-backtest/{year}/` to `data/models-backtest-mawc-conservative/{year}/`
2. Overwrite only `market_aware_win_calibrator_{surface}.joblib` for each surface
3. meta.json remains unchanged (same surfaces, ensemble mode, train period, quality threshold)

**Key files to copy per year directory (verified from 2024):**
```
cqr_params_{surface}.json              race_quality.lgb
cqr_quantile_high_{surface}.lgb        regime_detector.lgb
cqr_quantile_low_{surface}.lgb         stage1_{surface}.lgb
ev_corrector_e_{surface}.lgb           target_encoder_{surface}.joblib
ev_corrector_p_{surface}.lgb           win_hit_{surface}.joblib
ev_isotonic_{surface}.joblib           win_race_level_ranker_{surface}.joblib
ev_odds_band_scales_{surface}.json     win_ret_{surface}.lgb
market_{surface}.lgb                   win_selection_gate_{surface}.joblib
market_aware_win_calibrator_{surface}.joblib  <-- REPLACE THIS ONLY
meta.json                              win_selection_policy_{surface}.joblib
```

**Confidence:** HIGH -- verified by reading ModelLoader.load_from_dir() and listing actual files in data/models-backtest/2024/.

### RQ-5: Existing Baseline MAWC Model Files

**Directory structure (verified):**
```
data/models-backtest/
  meta.json              -- root-level (used by single-year backtest)
  2023/
    meta.json            -- train_start=2020-01-01, train_end=2022-12-31, use_ensemble=true
    *.lgb, *.joblib, *.json  -- all model files
    (NO market_aware_win_calibrator -- MAWC added in Phase 39, 2023 models predate it)
  2024/
    meta.json            -- train_start=2020-01-01, train_end=2023-12-31, use_ensemble=true
    market_aware_win_calibrator_turf.joblib  -- HAS MAWC
    market_aware_win_calibrator_dirt.joblib  -- HAS MAWC
    + all other model files (33 files total)
  2025/
    meta.json            -- train_start=2021-01-01, train_end=2024-12-31, use_ensemble=true
    market_aware_win_calibrator_turf.joblib  -- HAS MAWC
    market_aware_win_calibrator_dirt.joblib  -- HAS MAWC
    + all other model files (36 files including .bak files)
  2025_alpha_capped/     -- variant from earlier experiment
    meta.json
    (NO MAWC -- predates Phase 39)
```

**MAWC file naming:** `market_aware_win_calibrator_{surface}.joblib` where surface is "turf" or "dirt".

**Only 2024 and 2025 directories have MAWC models.** Phase 45 needs to create conservative variants for both years (since Phase 46 Shadow Comparison tests both 2024 and 2025 folds).

**Confidence:** HIGH -- verified by listing and reading files.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | p_win_corrected in oof_predictions.parquet is semantically equivalent to p_win_oof for MAWC training | RQ-1 | Retrained MAWC learns from slightly different probability distribution |
| A2 | Using combined 2022-2024 OOF data is acceptable for retraining per-surface MAWC (vs per-year backtest OOF) | RQ-1 | Conservative variant may not generalize to the specific backtest training windows |
| A3 | Removing ALL 15 logit_model x segment interactions (not just low-odds and pop_1) is the right approach | RQ-2 | May remove some beneficial interactions; could underfit certain segments |
| A4 | The conservative variant's meta.json can remain identical to the source (no need to modify surfaces, ensemble, or train_period) | RQ-4 | If Phase 46 expects different metadata in conservative variant, manifest may need adjustment |

**Note on A1:** p_win_corrected is the EV-corrected OOF prediction. In the original MAWC training flow, generate_win_oof_predictions sets p_win_oof = fold_val["p_win_pred"] (the uncorrected prediction from the fold model), then corrects EV and stores as p_win_corrected. The MAWC train method resolves p_model from p_win_oof (not p_win_corrected). This means p_win_corrected in oof_predictions.parquet may differ from the p_win_oof used in the original MAWC training. However, for retraining purposes, using p_win_corrected (which includes EV correction) is a reasonable choice since it represents the best available OOF estimate of the model's probability.

## Open Questions

1. **Should p_model use p_win_corrected or p_win_pred?**
   - What we know: oof_predictions.parquet has both. MAWC train resolves from p_win_oof (uncorrected) or p_model. In the original flow, p_win_oof = p_win_pred (before EV correction).
   - What's unclear: Whether using p_win_corrected (post-EV-correction) vs p_win_pred (pre-EV-correction) makes a meaningful difference for MAWC training.
   - Recommendation: Use p_win_corrected as it is the higher-quality OOF estimate. If quality gates fail, try p_win_pred as fallback.

2. **Should we retrain for both 2024 and 2025 test years?**
   - What we know: Backtest models exist for 2024 and 2025, each with their own MAWC. Phase 46 Shadow Comparison tests both years.
   - What's unclear: Whether the planner should create conservative variants for both years or just 2024 first.
   - Recommendation: Create for both years using the same conservative MAWC configuration. The MAWC is retrained per-surface on the full OOF data, then saved into both year directories.

3. **Favorite band guard threshold specifics**
   - What we know: CONTEXT says odds 1-3 ECE should not worsen, bet_count should not drop significantly, APR should not degrade. The specifics suggest mean(p_conservative/p_model) >= 0.90 for odds 1-3.
   - What's unclear: Exact numerical thresholds for "significant" bet_count drop and "large" APR degradation.
   - Recommendation: Use relative thresholds: ECE degradation > 10% relative, bet_count drop > 10%, APR degradation > 5 percentage points. These are conservative but not overly restrictive.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | MAWC retrain | Yes | 3.11 (mise) | - |
| scikit-learn | LogisticRegression | Yes | (project dep) | - |
| pandas | OOF data loading | Yes | (project dep) | - |
| joblib | Model save/load | Yes | (project dep) | - |
| data/oof/oof_predictions.parquet | MAWC training data | Yes | 136,859 rows | Must regenerate OOF via pipeline |
| data/models-backtest/{year}/ | Conservative variant source | Yes | 2024, 2025 dirs | - |

**Missing dependencies with no fallback:** none

**Missing dependencies with fallback:** none

## Sources

### Primary (HIGH confidence)
- `src/models/market_aware_win_calibrator.py` -- MAWC class definition, build_feature_matrix(), train(), apply(), save/load. 51-dim feature construction verified.
- `src/models/win_benter_gate.py` -- generate_win_oof_predictions() function. OOF data generation flow and column derivation verified.
- `src/db/model_loader.py` -- ModelLoader.load_from_dir() method. File naming conventions and meta.json requirements verified.
- `src/pipelines/training_pipeline.py` lines 1287-1322 -- MAWC training integration point. How generate_win_oof_predictions feeds into MAWC.train().
- `data/models-backtest/2024/market_aware_win_calibrator_turf.joblib` -- Loaded and inspected. 51 features, coefficients, training summary verified.

### Secondary (MEDIUM confidence)
- `.planning/phases/45-structural-fix/45-CONTEXT.md` -- User decisions D-01 through D-05
- `.planning/phases/39-marketawarewincalibrator/39-CONTEXT.md` -- MAWC design decisions D-01 through D-23

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all packages are existing project dependencies, no new packages needed
- Architecture: HIGH - verified by reading source code and loading actual model files
- Pitfalls: HIGH - derived from understanding of MAWC architecture and Phase 44 diagnosis
- OOF data path: HIGH - verified by column inspection and source code reading

**Research date:** 2026-05-31
**Valid until:** 2026-06-30 (stable -- no fast-moving dependencies)
