# Phase 30: Residual IC Evaluation Framework - Research

**Researched:** 2026-05-18
**Domain:** Information Coefficient (IC) metrics for market-independent prediction evaluation
**Confidence:** HIGH

## Summary

Phase 30 builds an IC evaluation framework that measures how much independent predictive power the v1.6 horse racing model has beyond what the market already knows. This is a pure evaluation/diagnostic module -- no model changes, no new features, no ETL. The framework computes 4 complementary IC formulations (B-difference IC, C-orthogonal IC, E-incremental IC, Per-race IC) from OOF (out-of-fold) predictions, then records baselines for future comparison.

The technical core is straightforward statistical computation: Spearman rank correlations and OLS regression residuals, all using already-installed libraries (scipy 1.17.1, numpy 2.4.3, sklearn 1.8.0). The primary engineering work is (1) adding a Parquet save hook in TrainingPipeline to persist the OOF DataFrame, and (2) building the ic_evaluator module following the exact pattern of ev_diagnostics.py and drift_diagnostics.py.

**Primary recommendation:** Follow the ev_diagnostics.py pattern exactly -- module-level constants, private computation functions, a public `run_ic_evaluation()` orchestration function, JSON output, and `console_summary()`. Use `scipy.stats.spearmanr` with explicit NaN pre-filtering (not `nan_policy='omit'` which has edge cases with DataFrames), and `numpy.linalg.lstsq` for OLS residuals.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `src/models/ic_evaluator.py` -- single file, function-based, following ev_diagnostics.py and drift_diagnostics.py patterns
- **D-02:** No new directory (src/evaluation/). Consistency with existing diagnostic modules
- **D-03:** Add OOF prediction DataFrame save to TrainingPipeline. Save to `data/oof/oof_predictions.parquet`
- **D-04:** IC evaluation reads from saved OOF Parquet (offline evaluation). No retraining required
- **D-05:** Market probability (implied_prob) from OOF DataFrame columns. Fallback: compute from `1/tanodds`
- **D-06:** IC results in JSON + MLflow dual recording. JSON: `data/baseline/ic_baseline.json`, MLflow: metrics + tags
- **D-07:** JSON includes surface-specific (turf/dirt) + overall (3 patterns) IC values. Each with 4 metrics
- **D-08:** Direction consistency check (RIC-06): WARNING log + JSON `consistency_check` section + MLflow tag. No execution stop
- **D-09:** Surface-specific (turf/dirt) + overall (3 patterns). Surface-specific IC is most meaningful
- **D-10:** Module API + CLI script + TrainingPipeline integration -- 3 access paths
- **D-11:** CLI script accepts OOF Parquet path and runs IC evaluation. Pipeline integration is optional call after OOF generation

### Claude's Discretion
- ic_evaluator.py internal function structure (function signatures for each IC formulation)
- OOF Parquet file naming convention
- JSON output schema details
- Test case design

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RIC-01 | B-difference IC: Spearman(model - market, y) | Standard Spearman on prediction delta vs outcome. Uses `scipy.stats.spearmanr`. See B-difference IC pattern below. |
| RIC-02 | C-orthogonal IC: Spearman(orthog(model\|market), y) via OLS residual | Regress model_pred on market_prob, take residuals, correlate with y. Uses `numpy.linalg.lstsq`. See C-orthogonal IC pattern below. |
| RIC-03 | E-incremental IC: IC(model, y) - IC(market, y) | Difference of two Spearman correlations. Simple subtraction of independent IC measurements. |
| RIC-04 | Per-race IC: average of within-race Spearman correlations | Group by race_id, compute Spearman within each race, average across races. Needs min-horses-per-race filter. |
| RIC-05 | Baseline IC recording for v1.6 model | JSON + MLflow output with all 4 metrics x 3 surface patterns = 12 IC values. Timestamped. |
| RIC-06 | Direction consistency auto-verification | All 4 IC formulations should agree on sign (positive = model adds value). Cross-check and warn on contradiction. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| OOF prediction persistence | Pipeline (training_pipeline.py) | Storage (data/oof/) | Pipeline generates OOF predictions; storage layer receives the Parquet file |
| IC metric computation | Models (ic_evaluator.py) | -- | Pure statistical computation module, parallel to ev_diagnostics.py |
| CLI entry point | Scripts (run_ic_eval.py) | Models | Thin CLI wrapper calling ic_evaluator functions |
| Baseline recording | Storage (data/baseline/) + MLflow | Models | JSON for local persistence, MLflow for experiment tracking |
| Direction consistency check | Models (ic_evaluator.py) | Logging | Computed within the evaluator, outputs WARNING if contradictory |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy | 1.17.1 | `spearmanr` for Spearman rank correlation | Already installed, used in stacked_ensemble.py and walk_forward_cv.py [VERIFIED: pip show] |
| numpy | 2.4.3 | `linalg.lstsq` for OLS residual computation | Already installed, standard linear algebra [VERIFIED: pip show] |
| pandas | 2.3.3 | DataFrame operations, groupby for per-race IC | Already installed, core data structure [VERIFIED: pip show] |
| sklearn | 1.8.0 | `LinearRegression` as alternative OLS implementation | Already installed, fallback option [VERIFIED: pip show] |
| pyarrow | (installed) | Parquet read/write for OOF data | Already installed, used by ParquetStore [VERIFIED: existing codebase] |
| mlflow | (installed) | Experiment tracking for IC baseline | Already installed, used in training_pipeline.py [VERIFIED: existing codebase] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | 3.11 | JSON output for baseline | Always -- standard output format |
| logging (stdlib) | 3.11 | Module-level logging | Always -- follows diagnostic module pattern |
| argparse (stdstdlib) | 3.11 | CLI argument parsing | For run_ic_eval.py script |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `numpy.linalg.lstsq` | `sklearn.linear_model.LinearRegression` | lstsq is lighter weight, no fit/predict overhead. LinearRegression has coef_/intercept_ but unnecessary here. |
| `scipy.stats.spearmanr` | `pandas.DataFrame.corr(method='spearman')` | spearmanr returns both rho and p-value. DataFrame.corr only returns rho. p-value is useful for significance testing. |

**Installation:**
No new packages required. All dependencies are already installed in the project environment.

**Version verification:**
```
scipy 1.17.1 -- verified via python import
numpy 2.4.3 -- verified via python import
pandas 2.3.3 -- verified via python import
sklearn 1.8.0 -- verified via python import
```

## Package Legitimacy Audit

> No new packages are installed in this phase. All dependencies are pre-existing.

| Package | Registry | Status | Notes |
|---------|----------|--------|-------|
| scipy | PyPI | Pre-installed | v1.17.1, already used in codebase |
| numpy | PyPI | Pre-installed | v2.4.3, core dependency |
| pandas | PyPI | Pre-installed | v2.3.3, core dependency |
| scikit-learn | PyPI | Pre-installed | v1.8.0, already used in codebase |
| mlflow | PyPI | Pre-installed | Already used in training_pipeline.py |

**New packages installed: None**

## Architecture Patterns

### System Architecture Diagram

```
TrainingPipeline._train_submodel()
  |
  |--> [Existing] oof_dfs accumulation
  |         |
  |         v
  |    df_oof_for_save (line 958: df_oof.copy() before conformal_ev)
  |         |
  |         |  [NEW] Save hook (D-03)
  |         v
  |    data/oof/oof_predictions.parquet  <--- NEW FILE
  |
  = (pipeline continues unchanged)
  
Offline Evaluation Path:
  data/oof/oof_predictions.parquet
    |
    |--> ic_evaluator.run_ic_evaluation()
    |       |
    |       |--> _compute_b_difference_ic()  -- RIC-01
    |       |--> _compute_c_orthogonal_ic()  -- RIC-02
    |       |--> _compute_e_incremental_ic() -- RIC-03
    |       |--> _compute_per_race_ic()      -- RIC-04
    |       |--> _check_direction_consistency() -- RIC-06
    |       |
    |       v
    |     data/baseline/ic_baseline.json   <--- NEW FILE
    |     MLflow metrics + tags            <--- NEW TRACKING
    |
    |--> console_summary()
    |       |
    |       v
        Logger output (INFO/WARNING)

CLI Access Path:
  scripts/run_ic_eval.py --oof-path data/oof/oof_predictions.parquet
    |
    v
  ic_evaluator.run_ic_evaluation()
    (same flow as above)

Pipeline Integration Path:
  TrainingPipeline.run()
    |
    |--> _train_submodel() --> oof_dfs --> [NEW] save_oof_parquet()
    |
    |--> [NEW, optional] ic_evaluator.run_ic_evaluation(oof_path)
```

### Recommended Project Structure
```
src/
├── models/
│   ├── ic_evaluator.py          # NEW - IC evaluation module (D-01)
│   ├── ev_diagnostics.py        # EXISTING - pattern reference
│   └── drift_diagnostics.py     # EXISTING - pattern reference
├── pipelines/
│   └── training_pipeline.py     # MODIFY - add OOF Parquet save hook (D-03)
scripts/
└── run_ic_eval.py               # NEW - CLI entry point (D-11)
tests/
└── test_ic_evaluator.py         # NEW - tests for IC module
data/
├── oof/                          # NEW DIRECTORY
│   └── oof_predictions.parquet   # OOF predictions output (D-03)
└── baseline/                     # NEW DIRECTORY
    └── ic_baseline.json          # IC baseline output (D-06)
```

### Pattern 1: Diagnostic Module Pattern (from ev_diagnostics.py / drift_diagnostics.py)
**What:** Function-based diagnostic module with module-level constants, private computation functions, public orchestration function, JSON output, and console_summary.
**When to use:** All diagnostic/evaluation modules in this project.
**Example:**
```python
# Source: ev_diagnostics.py (canonical reference in codebase)
logger = logging.getLogger("models.ic_evaluator")

# Module-level constants
MIN_SAMPLE_SIZE = 30
IC_TARGET_COLUMN = "kakuteijyuni"  # binary: 1 = win

# Private computation functions
def _compute_b_difference_ic(model_pred, market_prob, y):
    ...

# Public orchestration function
def run_ic_evaluation(df_oof, output_path=None):
    ...

# Console summary
def console_summary(result):
    ...
```

### Pattern 2: OOF Parquet Save Hook
**What:** Save the OOF DataFrame to Parquet at the point where it is already being copied for feature audit (line 958 of training_pipeline.py).
**When to use:** Exactly once per training run, after all surface submodels are trained and oof_dfs are concatenated.
**Example:**
```python
# In TrainingPipeline.run(), after line 244 (pd.concat(oof_dfs)):
# EXISTING: save_features(self.store, full_features_df)
# NEW: Save OOF predictions to data/oof/
from pathlib import Path
oof_path = Path("data/oof/oof_predictions.parquet")
oof_path.parent.mkdir(parents=True, exist_ok=True)
full_features_df.to_parquet(oof_path, index=False)
logger.info("Saved OOF predictions: %d rows -> %s", len(full_features_df), oof_path)
```

### Pattern 3: JSON Output with NaN-safe serialization
**What:** Use a `_json_default` helper function (as in drift_diagnostics.py) to handle numpy/pandas types.
**When to use:** All JSON output in this project.
**Example:**
```python
# Source: drift_diagnostics.py lines 219-229
def _json_default(obj: object) -> object:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
```

### Anti-Patterns to Avoid
- **Don't use `nan_policy='omit'` in `spearmanr` with DataFrames**: Known edge case where it raises ValueError with sparse DataFrames. Pre-filter NaN values manually before passing to spearmanr. [CITED: github.com/scipy/scipy/issues/13900]
- **Don't compute IC on training data (in-sample)**: IC must use OOF predictions only. Training predictions give inflated IC. The OOF DataFrame must come from KFold cross-validation.
- **Don't average per-race IC without minimum horse count filter**: Races with fewer than ~5 horses produce unstable Spearman estimates. Filter to races with >= 5 entries.
- **Don't use Pearson correlation instead of Spearman**: IC in quantitative prediction evaluation uses rank correlation (Spearman) because it is robust to outliers and non-linear relationships. [ASSUMED]
- **Don't modify the training pipeline's model training logic**: The OOF save hook must be additive only. The pipeline's training flow stays unchanged.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rank correlation | Custom rank correlation | `scipy.stats.spearmanr` | Handles ties correctly, returns p-value, well-tested |
| OLS regression for residuals | Manual matrix inversion | `numpy.linalg.lstsq` | Numerically stable, handles rank-deficient cases |
| Parquet I/O | Custom binary format | `pandas.to_parquet()` / `pd.read_parquet()` | Compression, schema, compatibility |
| NaN filtering | Complex mask logic | `pd.to_numeric(..., errors='coerce').dropna()` | Follows existing pattern in ev_diagnostics.py |

**Key insight:** All the mathematical building blocks are single-function calls. The value is in the correct composition and interpretation, not in novel algorithms.

## Common Pitfalls

### Pitfall 1: OOF DataFrame missing required columns
**What goes wrong:** The saved OOF Parquet may not contain `p_win_corrected`, `tanodds`, or `implied_prob` if the save happens at the wrong pipeline stage.
**Why it happens:** The OOF DataFrame evolves through the pipeline -- columns are added (p_win_corrected by EVCorrectionModel), renamed, or dropped (confirmed_odds dropped at line 969).
**How to avoid:** Save OOF data at line 958 (`df_oof_for_save = df_oof.copy()`) which captures the state AFTER EV correction but BEFORE confirmed_odds is dropped. Verify that `p_win_corrected`, `tanodds`, `kakuteijyuni`, `surface`, `race_id`, and `race_date` are all present.
**Warning signs:** KeyError when ic_evaluator tries to access columns.

### Pitfall 2: Per-race IC unstable with small field sizes
**What goes wrong:** Per-race IC (RIC-04) produces wild values for races with few horses (e.g., 3-horse fields).
**Why it happens:** Spearman correlation with 3 data points has very high variance. The rho value swings between -1 and +1 with tiny data changes.
**How to avoid:** Set minimum horses-per-race threshold (e.g., `MIN_HORSES_PER_RACE = 5`). Exclude races below threshold from per-race IC computation. Log the number of excluded races.
**Warning signs:** Per-race IC near +-1.0 with high variance.

### Pitfall 3: C-orthogonal IC can be negative (valid but confusing)
**What goes wrong:** After regressing model predictions on market probability, the residual IC may be negative. This is mathematically valid but may confuse users.
**Why it happens:** OLS residual removes the market-correlated component. If the model's predictive power comes entirely from the market (no independent alpha), the orthogonal IC is zero or negative.
**How to avoid:** Document clearly that C-orthogonal IC is the strictest test. Negative values mean the model has no market-independent predictive power. This is valuable information, not an error.
**Warning signs:** C-orthogonal IC << B-difference IC.

### Pitfall 4: B-difference IC and E-incremental IC are NOT the same
**What goes wrong:** Assuming Spearman(model - market, y) == Spearman(model, y) - Spearman(market, y).
**Why it happens:** Spearman correlation is not linear. The difference of correlations is not the same as the correlation of differences.
**How to avoid:** Compute both independently. The direction consistency check (RIC-06) will catch if their signs diverge, but different magnitudes are expected.
**Warning signs:** Identical B-difference and E-incremental IC values.

### Pitfall 5: Missing implied_prob column
**What goes wrong:** The OOF DataFrame may not have an `implied_prob` column.
**Why it happens:** Market probability is stored as `tanodds` (raw odds), not pre-computed probability. The D-05 decision specifies fallback computation from `1/tanodds`.
**How to avoid:** Always check for `implied_prob` first, then fall back to `1/tanodds`. Use `np.clip(result, 0.01, 0.99)` to handle edge cases (same as win_benter_gate.py line 51-52).
**Warning signs:** ValueError or inf values in IC computation.

### Pitfall 6: Race-level grouping with empty groups
**What goes wrong:** `df.groupby("race_id")` may produce groups where all values are NaN after filtering, causing spearmanr to fail.
**Why it happens:** Some races may have all NaN in model predictions (e.g., horses with no feature data).
**How to avoid:** After groupby, check each group has enough valid (non-NaN) entries before computing Spearman. Skip groups that don't meet the minimum threshold.
**Warning signs:** "The input must have at least 3 entries" ValueError from spearmanr.

## Code Examples

### B-difference IC (RIC-01)
```python
# Spearman(model_pred - market_prob, y)
def _compute_b_difference_ic(
    model_pred: np.ndarray,
    market_prob: np.ndarray,
    y: np.ndarray,
) -> dict:
    """B-difference IC: Spearman(delta, y) where delta = model - market."""
    delta = model_pred - market_prob
    valid = np.isfinite(delta) & np.isfinite(y)
    if valid.sum() < MIN_SAMPLE_SIZE:
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(valid.sum())}
    rho, p_value = spearmanr(delta[valid], y[valid])
    return {"rho": float(rho), "p_value": float(p_value), "n": int(valid.sum())}
```

### C-orthogonal IC (RIC-02)
```python
# Spearman(residuals, y) where residuals = model - OLS(model|market)
def _compute_c_orthogonal_ic(
    model_pred: np.ndarray,
    market_prob: np.ndarray,
    y: np.ndarray,
) -> dict:
    """C-orthogonal IC: Spearman(resid, y) where resid = model - OLS(model|market)."""
    valid = np.isfinite(model_pred) & np.isfinite(market_prob) & np.isfinite(y)
    if valid.sum() < MIN_SAMPLE_SIZE:
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(valid.sum())}
    x = market_prob[valid].reshape(-1, 1)
    y_pred = model_pred[valid]
    # OLS: model_pred = beta * market_prob + intercept + residual
    x_with_intercept = np.column_stack([np.ones(len(x)), x])
    coeffs, _, _, _ = np.linalg.lstsq(x_with_intercept, y_pred, rcond=None)
    residuals = y_pred - x_with_intercept @ coeffs
    rho, p_value = spearmanr(residuals, y[valid])
    return {"rho": float(rho), "p_value": float(p_value), "n": int(valid.sum())}
```

### E-incremental IC (RIC-03)
```python
# IC(model, y) - IC(market, y)
def _compute_e_incremental_ic(
    model_pred: np.ndarray,
    market_prob: np.ndarray,
    y: np.ndarray,
) -> dict:
    """E-incremental IC: Spearman(model, y) - Spearman(market, y)."""
    valid = np.isfinite(model_pred) & np.isfinite(market_prob) & np.isfinite(y)
    if valid.sum() < MIN_SAMPLE_SIZE:
        return {
            "ic_model": float("nan"), "ic_market": float("nan"),
            "delta_ic": float("nan"), "n": int(valid.sum()),
        }
    ic_model, _ = spearmanr(model_pred[valid], y[valid])
    ic_market, _ = spearmanr(market_prob[valid], y[valid])
    return {
        "ic_model": float(ic_model),
        "ic_market": float(ic_market),
        "delta_ic": float(ic_model - ic_market),
        "n": int(valid.sum()),
    }
```

### Per-race IC (RIC-04)
```python
# Average Spearman within each race
def _compute_per_race_ic(
    df: pd.DataFrame,
    pred_col: str,
    y_col: str,
    group_col: str = "race_id",
    min_horses: int = 5,
) -> dict:
    """Per-race IC: average of within-race Spearman correlations."""
    results = []
    skipped = 0
    for _race_id, group in df.groupby(group_col, observed=True):
        pred = pd.to_numeric(group[pred_col], errors="coerce").dropna()
        actual = pd.to_numeric(group[y_col], errors="coerce").dropna()
        common = pred.index.intersection(actual.index)
        if len(common) < min_horses:
            skipped += 1
            continue
        rho, _ = spearmanr(pred.loc[common].values, actual.loc[common].values)
        if np.isfinite(rho):
            results.append(rho)
    if not results:
        return {"mean_rho": float("nan"), "n_races": 0, "skipped_races": skipped}
    return {
        "mean_rho": float(np.mean(results)),
        "std_rho": float(np.std(results)),
        "median_rho": float(np.median(results)),
        "n_races": len(results),
        "skipped_races": skipped,
    }
```

### Direction Consistency Check (RIC-06)
```python
def _check_direction_consistency(ic_results: dict) -> dict:
    """Verify all 4 IC formulations agree on direction."""
    ic_values = []
    for key in ["b_difference", "c_orthogonal", "e_incremental", "per_race"]:
        metric = ic_results.get(key, {})
        rho = metric.get("rho") or metric.get("delta_ic") or metric.get("mean_rho")
        if rho is not None and np.isfinite(rho):
            ic_values.append((key, rho))

    if len(ic_values) < 2:
        return {"consistent": True, "warning": "insufficient_data"}

    signs = [1 if v > 0 else -1 if v < 0 else 0 for _, v in ic_values]
    non_zero_signs = [s for s in signs if s != 0]

    consistent = len(set(non_zero_signs)) <= 1
    result = {
        "consistent": consistent,
        "n_metrics_checked": len(ic_values),
        "details": {k: v for k, v in ic_values},
    }
    if not consistent:
        result["warning"] = "IC direction inconsistency detected -- possible computation error"
    return result
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Pearson IC | Spearman rank IC | Standard in quant finance | More robust to outliers, standard for IC evaluation |
| Single IC metric | Multi-formulation IC battery | This phase | Cross-validation of model value across 4 independent definitions |
| In-sample IC | OOF (out-of-fold) IC | v1.0+ | Prevents overfitting bias in IC estimates |

**Deprecated/outdated:**
- Pearson IC for prediction evaluation: Still used in some contexts but Spearman is preferred for rank-based model evaluation [ASSUMED]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Spearman is preferred over Pearson for IC in prediction model evaluation | Architecture Patterns | Low -- both valid, Spearman is more conservative |
| A2 | OOF DataFrame at line 958 contains all needed columns (p_win_corrected, tanodds, surface, race_id) | Common Pitfalls | Medium -- if columns missing, need to trace pipeline to find alternative save point |
| A3 | min_horses_per_race = 5 is a reasonable threshold for per-race IC | Code Examples | Low -- threshold is configurable, 5 is conservative |
| A4 | B-difference IC, E-incremental IC, C-orthogonal IC, Per-race IC should all have the same sign | RIC-06 | Medium -- some divergence may be legitimate, warning level is appropriate |

**If this table is empty:** All claims in this research were verified or cited.

## Open Questions

1. **OOF save timing -- confirmed_odds availability**
   - What we know: Line 958 copies df_oof before confirmed_odds is dropped (line 969). The copy should have confirmed_odds.
   - What's unclear: Whether df_oof_for_save has been used to actually save features (line 244-248 in run() uses a different concat).
   - Recommendation: The save hook should go in run() after line 244 where full_features_df is already saved to features parquet. Alternatively, add a second save at line 958 in _train_submodel() for the richer per-surface OOF. The planner should verify which point has the most complete column set.

2. **MLflow run context for IC evaluation**
   - What we know: Training pipeline logs to MLflow in _log_to_mlflow(). IC evaluation runs after training.
   - What's unclear: Whether IC evaluation should log within the existing MLflow run or start a new one.
   - Recommendation: Start a new MLflow run for IC evaluation (independent experiment). This allows running IC evaluation standalone without retraining.

3. **Per-race IC: which prediction column to use for grouping**
   - What we know: p_win_corrected is the final model prediction. But there's also p_win_combined (after Benter) and p_win_final (after race normalization).
   - What's unclear: Which prediction column best represents the "model" for IC evaluation.
   - Recommendation: Use p_win_corrected as the primary model prediction (it is the model output before market blending). Also compute IC for p_win_combined to measure blended model performance. Default to p_win_corrected for consistency with RIC-01 through RIC-04 definitions which describe "model" predictions.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scipy | spearmanr | Yes | 1.17.1 | -- |
| numpy | lstsq | Yes | 2.4.3 | -- |
| pandas | DataFrame operations | Yes | 2.3.3 | -- |
| scikit-learn | LinearRegression (optional) | Yes | 1.8.0 | numpy.lstsq |
| pyarrow | Parquet I/O | Yes | installed | -- |
| mlflow | Experiment tracking | Yes | installed | -- |
| Python | Runtime | Yes | 3.11 (mise) | -- |

**Missing dependencies with no fallback:** None

**Missing dependencies with fallback:** None

## Sources

### Primary (HIGH confidence)
- Codebase: `src/models/ev_diagnostics.py` -- diagnostic module pattern reference (verified by reading)
- Codebase: `src/models/drift_diagnostics.py` -- diagnostic module pattern reference (verified by reading)
- Codebase: `src/pipelines/training_pipeline.py` -- OOF DataFrame generation and save point (verified by reading, lines 205-244, 958)
- Codebase: `src/models/win_benter_gate.py` -- OOF prediction generation pattern, tanodds/implied_prob extraction (verified by reading)
- [scipy.stats.spearmanr documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) -- nan_policy options and API

### Secondary (MEDIUM confidence)
- [SciPy GitHub Issue #13900](https://github.com/scipy/scipy/issues/13900) -- nan_policy='omit' DataFrame edge case
- [Information Coefficient as Performance Measure (arXiv)](https://arxiv.org/pdf/2010.08601) -- IC as correlation between predicted and realized returns

### Tertiary (LOW confidence)
- Spearman preferred over Pearson for IC evaluation in quantitative finance -- [ASSUMED], standard practice but not verified from official source in this session

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and verified
- Architecture: HIGH -- follows existing patterns exactly (ev_diagnostics.py, drift_diagnostics.py)
- Pitfalls: HIGH -- identified from codebase analysis and scipy known issues
- IC formulations: HIGH -- well-defined statistical methods, verified via scipy/numpy docs

**Research date:** 2026-05-18
**Valid until:** 2026-06-17 (30 days -- stable domain, no external dependencies)
