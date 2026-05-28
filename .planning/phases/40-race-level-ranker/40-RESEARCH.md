# Phase 40: Race-Level Ranker - Research

**Researched:** 2026-05-28
**Domain:** Learned ranking for horse race investment scoring (sklearn Ridge + optional LightGBM LambdaRank shadow)
**Confidence:** HIGH

## Summary

Phase 40 introduces a learned ranker that replaces the hand-tuned `win_market_selection_score` formula in `RacePredictor.get_win_candidates()`. The ranker combines two independent Ridge regression models -- a relevance scorer (learns finishing-position graded relevance) and a value scorer (learns composite mispricing signals) -- into a single `investment_score` via pre-declared fixed weights on race-level robust percentile ranks.

The implementation follows the exact shadow-mode pattern established by Phase 39's `MarketAwareWinCalibrator`: `_trained` boolean + `deployment_status` field + `is_trained` property. Integration occurs after MAWC in `RacePredictor.predict()` (line ~277) and before the final sort in `get_win_candidates()` (line ~838). Training data is built by extending the existing `generate_win_oof_predictions()` function in `src/models/win_benter_gate.py` to emit additional columns required for ranker targets and features.

No new pip dependencies are needed. sklearn 1.8.0 provides `Ridge` with the required alpha grid `[0.03, 0.1, 0.3, 1.0, 3.0, 10.0]`. LightGBM 4.6.0 supports `lambdarank` objective for the shadow benchmark.

**Primary recommendation:** Build two lightweight Ridge models per surface (4 total), extend OOF generation minimally, integrate as shadow-only column addition in RacePredictor, following MarketAwareWinCalibrator patterns exactly.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Two separate Ridge/regularized linear models: `relevance_scorer` + `value_scorer`. Each model is a per-surface independent scorer stored in SubmodelSet.
- **D-02:** LightGBM LambdaRank trained as shadow benchmark only -- not the default deployable model.
- **D-03:** Combination formula: `investment_score = 0.35 * relevance_score_pct + 0.35 * value_score_pct + 0.20 * calibrated_log_ev_pct - 0.10 * uncertainty_penalty_pct`. All components are race-level robust percentile ranks.
- **D-04:** Weights are pre-declared and NOT optimized on 2024/2025.
- **D-05:** Report each component separately in shadow diagnostics.
- **D-06:** Ridge alpha selection via deterministic grid: [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]. Primary metrics: relevance_scorer -> NDCG@3 / top1 win relevance; value_scorer -> rank correlation + top1/top3 value capture. Tie-breaker: larger alpha.
- **D-07:** Validation uses chronological race-level WF folds (same definitions as Phase 39).
- **D-08:** Relevance scorer target: graded relevance {1.00, 0.55, 0.30, 0.10, 0.00} based on finishing position.
- **D-09:** Value scorer target: composite `value_target = clipped_log_ev + mispricing_bonus - uncertainty_penalty` (OOF-safe).
- **D-10:** Actual return/payout is diagnostic only -- never a training target.
- **D-11:** Binary is_win diagnostics reported separately.
- **D-12:** Extend Phase 39's `generate_win_oof_predictions()` to emit ranker-required columns.
- **D-13:** Required columns: race_id, umaban, race_date, surface, fold_id, kakuteijyuni, p_win_oof / p_win_market_aware_oof, p_market_norm, calibrated_ev_oof, model_market_gap features, uncertainty features, odds/return for diagnostics.
- **D-14:** Build training data from: OOFHealthValidator-passed OOF artifacts + InvestmentFeatureFrame train-mode output + MarketAwareWinCalibrator OOF/shadow outputs.
- **D-15:** Probability-derived rank/bucket features must be recomputed from OOF probabilities.
- **D-16:** Parallel shadow first. Ranker produces investment_score in shadow mode alongside existing selectors.
- **D-17:** Existing selectors remain fully functional behind feature flags. No deletion.
- **D-18:** Shadow diagnostics must compare: baseline selected horse vs ranker selected horse, score components breakdown, agreement rate.
- **D-19:** If gates pass: ranker may replace WinSelectionPolicy as race-internal ranking score.
- **D-20:** Ranker scores computed after MAWC and IFF construction, before final candidate sorting. Score ALL runners.
- **D-21:** In shadow mode, compute investment_score for all runners and add columns to diagnostics.
- **D-22:** Curated feature subsets (~12-16 for relevance, ~14-18 for value), not full 94-feature IFF.
- **D-23:** Relevance scorer features (canonical IFF names).
- **D-24:** Value scorer features (canonical IFF names).
- **D-25:** Feature names must match actual Phase 38 schema. If unavailable, use registered missing/default.
- **D-26:** No actual payout or realized ROI features as predictors.
- **D-27:** Race-level robust percentile ranks as primary normalization. Deterministic tie handling.
- **D-28:** Do not use race-level z-score as primary.

### Claude's Discretion
- Exact feature matrix construction and missing-feature handling within IFF schema rules.
- LightGBM LambdaRank shadow training configuration.
- SubmodelSet field naming for ranker models.
- Test structure and naming within existing conventions.
- Model serialization format (joblib consistent with existing patterns).
- Exact integration code in RacePredictor.predict() and get_win_candidates().

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RNK-01 | Learned Win relevance ranker orders horses using is_win and finishing-position relevance | D-08 graded relevance target, Ridge model, curated ~12-16 features from IFF schema |
| RNK-02 | Learned Value/mispricing ranker detects mispriced horses using calibrated EV, model-vs-market gap, CLV diagnostics (OOF-safe) | D-09 composite value target, Ridge model, curated ~14-18 features from IFF schema |
| RNK-03 | Win ranker + Value ranker outputs combined into single investment_score per horse | D-03 fixed weight formula with race-level robust percentile rank normalization |
| RNK-04 | Ranker operates in shadow mode behind feature flag, baseline WinSelectionGate preserved | D-16 parallel shadow, D-17 no deletion, MAWC pattern replication |
| RNK-05 | One-bet-per-race baseline bet count maintained | D-20 score all runners, shadow diagnostic column addition only |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Relevance scoring (learned rank model) | API / Backend | -- | Ridge model trained offline in TrainingPipelineV5, applied at inference in RacePredictor |
| Value scoring (learned mispricing model) | API / Backend | -- | Same pattern as relevance -- Ridge trained offline, applied per-race |
| investment_score combination | API / Backend | -- | Deterministic formula applied after MAWC, before final sort |
| Shadow mode flag management | API / Backend | -- | Follows MAWC `_trained` + `deployment_status` pattern |
| Feature frame construction | API / Backend | -- | IFF builder resolves features from OOF (train) or production (infer) sources |
| OOF training data generation | API / Backend | -- | Extends `generate_win_oof_predictions()` in training pipeline |
| Model persistence (joblib) | API / Backend | -- | ModelLoader loads from `data/models/` or MLflow, same pattern as MAWC |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sklearn.linear_model.Ridge | 1.8.0 | Relevance + Value scorer models | Already in codebase, L2 regularization via alpha grid, consistent with MAWC's LogisticRegression |
| sklearn.metrics | 1.8.0 | NDCG, rank correlation metrics for alpha selection | Standard ML metrics, no new dependency |
| joblib | existing | Model serialization | Consistent with MAWC, WinSelectionPolicy, WinSelectionGate save/load patterns |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| LightGBM | 4.6.0 | LambdaRank shadow benchmark | D-02: shadow only, objective="lambdarank", not deployable |
| pandas | existing | DataFrame manipulation, groupby rank | All data pipeline operations |
| numpy | existing | Array operations, clipping | Numeric computations |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Ridge regression | XGBoost ranker | XGBoost adds complexity, D-01 explicitly chose Ridge for deployable model |
| Fixed weight combination | Learned weights | D-04 prohibits ROI-optimized weights; fixed weights prevent 2024/2025 overfitting |
| Robust percentile rank | Z-score normalization | D-28 rejects z-score as primary due to small-field instability |

**Installation:**
```bash
# No new packages needed -- all dependencies already in codebase
pip install -e ".[dev]"  # Existing install command
```

**Version verification:**
```
sklearn 1.8.0 (verified: python -c "import sklearn; print(sklearn.__version__)")
LightGBM 4.6.0 (verified: python -c "import lightgbm; print(lightgbm.__version__)")
```

## Package Legitimacy Audit

> No new packages installed in this phase. All dependencies are existing.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| scikit-learn | PyPI | 17+ yrs | 40M+/wk | github.com/scikit-learn/scikit-learn | -- | Existing -- no install needed |
| lightgbm | PyPI | 8+ yrs | 1M+/wk | github.com/microsoft/LightGBM | -- | Existing -- no install needed |
| joblib | PyPI | 14+ yrs | 50M+/wk | github.com/joblib/joblib | -- | Existing -- no install needed |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*No new packages installed -- all are existing codebase dependencies.*

## Architecture Patterns

### System Architecture Diagram

```
                    Training Time
                    =============
                    
df_oof (OOF DataFrame)
    |
    v
generate_win_oof_predictions()  [EXTENDED to emit ranker columns]
    |
    v
oof_cal_df (enriched OOF frame)
    |
    +---> IFF build_frame(mode="train")  ---->  Relevance Feature Matrix (~12-16 cols)
    |                                           Value Feature Matrix (~14-18 cols)
    |
    +---> Compute targets:
    |       relevance_target (graded by kakuteijyuni)
    |       value_target (clipped_log_ev + mispricing_bonus - uncertainty_penalty)
    |
    v
Ridge Alpha Selection (WF grid search)
    |
    +---> relevance_scorer_turf/dirt  (Ridge, best alpha by NDCG@3)
    +---> value_scorer_turf/dirt      (Ridge, best alpha by rank correlation)
    |
    v
SubmodelSet.win_race_level_ranker  (container for 4 Ridge models)
    |
    v
Save to data/models/ as .joblib + MLflow artifacts


                    Inference Time
                    ==============
                    
RacePredictor.predict(race_df)
    |
    v
[submodel.win.predict_ev]     [submodel.ev_corrector.correct_ev]
    |
    v
[submodel.market_aware_win_calibrator.apply]  <-- Phase 39
    |
    v
[IFF build_frame(mode="infer")]  <-- Build investment features
    |
    v
[RACE LEVEL RANKER BLOCK -- NEW]  <-- Line ~277 in predict()
    |   Score ALL runners:
    |   1. Extract relevance features -> Ridge.predict -> relevance_score
    |   2. Extract value features -> Ridge.predict -> value_score
    |   3. Compute calibrated_log_ev, uncertainty_penalty
    |   4. Robust percentile rank each component within race
    |   5. Combine: investment_score = 0.35*rel_pct + 0.35*val_pct + 0.20*log_ev_pct - 0.10*uncertainty_pct
    |   6. Add columns: relevance_score, value_score, investment_score + components
    |
    v
[WinSelectionGate]             <-- Preserved (RNK-04, D-16)
    |
    v
RacePredictor.get_win_candidates()
    |
    v
[win_market_selection_score]   <-- Baseline sorting (preserved)
[investment_score]             <-- Shadow diagnostic column (D-21)
    |
    v
Sort by baseline score, log shadow agreement
```

### Recommended Project Structure
```
src/
├── models/
│   └── race_level_ranker.py        # NEW: RaceLevelRanker class with Ridge scorers
├── pipelines/
│   └── training_pipeline.py        # MODIFY: Add ranker training after MAWC
├── backtest/
│   └── race_predictor.py           # MODIFY: Add ranker scoring block after MAWC
├── db/
│   └── model_loader.py             # MODIFY: Add ranker load/save
├── domain/
│   └── models.py                    # MODIFY: Add ranker fields to SubmodelSet
└── investment/
    ├── feature_frame.py             # USE: Feature extraction via build_frame
    └── schema_registry.py           # REFERENCE: Canonical feature names
```

### Pattern 1: Shadow Mode via is_trained + deployment_status
**What:** Model has `_trained: bool` property and `training_summary: dict` containing `deployment_status`. RacePredictor checks `if model is not None and model.is_trained` before applying.
**When to use:** All new models that should not immediately replace baseline.
**Example:**
```python
# Source: src/models/market_aware_win_calibrator.py (Phase 39 pattern)
@dataclass
class RaceLevelRanker:
    relevance_scorer_turf: Ridge | None = None
    relevance_scorer_dirt: Ridge | None = None
    value_scorer_turf: Ridge | None = None
    value_scorer_dirt: Ridge | None = None
    _trained: bool = False
    training_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def is_trained(self) -> bool:
        return self._trained and self.relevance_scorer_turf is not None
```

### Pattern 2: Per-Surface Independent Models in SubmodelSet
**What:** SubmodelSet stores per-surface models as optional fields. Naming: `{model_name}_{surface}` or as a single container object.
**When to use:** Any model that trains independently per surface.
**Example:**
```python
# Source: src/domain/models.py lines 240-269
@dataclass
class SubmodelSet:
    # ... existing fields ...
    market_aware_win_calibrator: MarketAwareWinCalibrator | None = None
    # Phase 40: Race-Level Ranker
    win_race_level_ranker: RaceLevelRanker | None = None
```

### Pattern 3: WF Alpha Grid Selection
**What:** Train Ridge models with multiple alpha values using walk-forward splits. Select best alpha by primary metric with tie-breaker.
**When to use:** Ridge alpha selection for ranker models.
**Example:**
```python
# Source: src/models/market_aware_win_calibrator.py lines 203-295
# MAWC uses LogisticRegression C-grid with WF splits.
# Ranker follows same pattern but with Ridge alpha-grid.
splits = _walk_forward_race_splits(df, n_splits=n_splits)
for alpha in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    fold_metrics = []
    for train_idx, val_idx in splits:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X[train_idx], y[train_idx])
        # Evaluate on val_idx
    # Select best alpha by primary metric, tie-breaker: larger alpha
```

### Pattern 4: Joblib Serialization for sklearn Models
**What:** sklearn models saved as .joblib files via `joblib.dump()`/`joblib.load()`. File naming: `{model_name}_{surface}.joblib`.
**When to use:** All sklearn model persistence.
**Example:**
```python
# Source: src/models/market_aware_win_calibrator.py lines 443-470
def save(self, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({...state dict...}, path)

@classmethod
def load(cls, path: Path) -> "RaceLevelRanker":
    state = joblib.load(path)
    return cls(...)
```

### Anti-Patterns to Avoid
- **Anti-pattern:** Computing p_win_race_rank_pct from train-mode predictions instead of OOF. **Why bad:** D-15 requires OOF-derived ranks. Train-mode ranks leak future information. **What to do:** Always use OOF probabilities for probability-derived features.
- **Anti-pattern:** Using ROI/payout as training target. **Why bad:** D-10 explicitly prohibits this. Realized returns are too sparse and cause overfitting. **What to do:** Use graded relevance target and composite value target.
- **Anti-pattern:** Full 94-feature IFF for Ridge models. **Why bad:** D-22 requires curated subsets. Full IFF causes Ridge coefficient instability and overfitting with limited training data. **What to do:** Use curated ~12-16 relevance features and ~14-18 value features.
- **Anti-pattern:** Using z-score normalization for combination. **Why bad:** D-28 prohibits this. Small field sizes (5-18 horses) make z-scores unstable. **What to do:** Use robust percentile ranks with deterministic tie handling.
- **Anti-pattern:** Restricting ranker scoring to WinSelectionGate-passed horses. **Why bad:** D-20 says score ALL runners. Gate-passed subset inherits selection bias. **What to do:** Score all runners after MAWC, before final sort.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| NDCG computation | Custom NDCG loop | sklearn.metrics.ndcg_score | Standard implementation, handles ties correctly |
| Rank correlation | Manual Spearman | scipy.stats.spearmanr or pandas.Series.corr(method='spearman') | Well-tested, handles edge cases |
| Walk-forward splits | Custom time split | utils.wf_splits.walk_forward_race_splits | Already used by MAWC and OOF generation |
| Robust percentile rank | Custom percentile function | pandas.DataFrame.groupby().rank(pct=True, method='average') | Deterministic, handles ties correctly |
| Feature schema resolution | Manual column mapping | InvestmentFeatureFrameBuilder.build_frame() | Dual-mode resolution with leakage guard |
| Model persistence | Custom pickle format | joblib.dump/load (following MAWC pattern) | Consistent with codebase, handles sklearn models |

**Key insight:** The entire infrastructure exists. Phase 40 is a composition task -- assembling existing patterns (shadow mode, WF alpha selection, joblib persistence, IFF features) into a new component. No novel infrastructure needed.

## Runtime State Inventory

> Not a rename/refactor phase -- skip.

## Common Pitfalls

### Pitfall 1: Feature Name Mismatch Between D-23/D-24 and Actual Schema
**What goes wrong:** CONTEXT.md D-23/D-24 lists feature names that don't exactly match the IFF schema registry. Several listed features have no corresponding `if_*` spec.
**Why it happens:** CONTEXT.md was written during discussion phase, not verified against actual schema.
**How to avoid:** Map each D-23/D-24 feature name to the actual schema registry entry. Use `_resolve_source()` fallback for missing features.
**Warning signs:** KeyError or ValueError at feature extraction time.
**Actual gaps found (see Feature Availability Audit below):**
- `if_p_diff`, `if_market_residual` -- NOT in schema. `if_logit_gap` serves equivalent purpose.
- `if_p_ratio` / `if_market_value_ratio` -- NOT in schema. `win_market_value_ratio` exists in win_selection_gate but not as `if_*` feature.
- `if_odds_rank` -- NOT in schema. Must derive from `if_odds` + groupby rank.
- `if_odds_drop_rate_60_10` / `if_odds_drop_rate_30_10` -- Schema has `if_odds_drop_60_10` / `if_odds_drop_30_10` (different naming).
- `if_late_odds_drop_z` -- NOT in schema. Must compute as race-level z-score of `if_odds_drop_30_10`.
- `if_market_share_change` -- NOT in schema. No source available.
- `if_overround_proxy` -- NOT in schema. `if_overround` exists.
- `if_model_market_disagreement` -- NOT in schema. No source available.
- `if_ev_uncertainty_proxy` -- NOT in schema. `if_ev_uncertainty_ratio` serves this purpose.
- `if_calibrated_log_ev` -- NOT in schema. Must derive from `if_ev_calibrated` + `if_odds_log`.
- `rel_p_ability_win_rank` / `rel_p_ability_win_zscore` -- Exist in `relative_features.py` but NOT as `if_*` features. Must add to IFF schema or compute at ranker level.
- `field_size` -- Available as `if_n_horses` in schema.

### Pitfall 2: OOF Data Columns Not Available at Training Time
**What goes wrong:** Ranker needs columns (calibrated_ev_oof, p_win_market_aware_oof, uncertainty features) that `generate_win_oof_predictions()` doesn't emit.
**Why it happens:** D-12 says extend the function, but the OOF loop in `win_benter_gate.py` only trains WinTwoStageModel + EVCorrection per fold. It doesn't run the full pipeline including MAWC or IFF.
**How to avoid:** The extension strategy must handle this carefully. Two options: (a) run MAWC per fold inside OOF generation (expensive, adds ~10 min), or (b) use columns available from the existing OOF loop and compute value target from simpler proxies. CONTEXT.md D-14 says use OOFHealthValidator-passed OOF artifacts + IFF train-mode output + MAWC OOF/shadow outputs -- this implies joining multiple sources, not running everything inside one function.
**Warning signs:** Missing column errors during ranker training data construction.

### Pitfall 3: Small Field Edge Cases
**What goes wrong:** Fields with fewer than 5 horses produce degenerate percentile ranks (all 0.0 or 1.0).
**Why it happens:** Robust percentile rank with method="average" on 3-4 horses gives only 2-3 distinct values.
**How to avoid:** Minimum field size guard. If field_size < 3, skip ranker scoring or fall back to deterministic ordering. Use `method="first"` with stable sort by umaban for deterministic tie-breaking (D-27).
**Warning signs:** All runners get the same investment_score in small fields.

### Pitfall 4: Surface Submodel Missing at Inference
**What goes wrong:** RacePredictor tries to score with a ranker that only has turf models when processing a dirt race.
**Why it happens:** Per-surface models may fail to train independently if data is insufficient.
**How to avoid:** Check both surface-specific models exist before scoring. Fall back to MAWC-only pipeline if ranker not available for the surface (consistent with MAWC pattern).
**Warning signs:** AttributeError when accessing `None` model.

### Pitfall 5: Investment Score Scale Mismatch with Baseline
**What goes wrong:** investment_score is on a different scale than win_market_selection_score, causing confusion in diagnostics.
**Why it happens:** Percentile ranks are [0,1] while selection_score can be any range.
**How to avoid:** D-05 requires reporting all components separately. investment_score should NOT replace win_market_selection_score in shadow mode -- it adds new diagnostic columns alongside the existing score.
**Warning signs:** Unexpected candidate selections in shadow comparison.

## Code Examples

### Relevance Target Construction (D-08)
```python
# Graded relevance target: {1.00, 0.55, 0.30, 0.10, 0.00}
def _compute_relevance_target(kakuteijyuni: pd.Series) -> pd.Series:
    """D-08: Graded relevance by finishing position."""
    pos = pd.to_numeric(kakuteijyuni, errors="coerce")
    return np.select(
        [pos == 1, pos == 2, pos == 3, pos.isin([4, 5])],
        [1.00, 0.55, 0.30, 0.10],
        default=0.00,
    ).astype(float)
```

### Value Target Construction (D-09)
```python
def _compute_value_target(df: pd.DataFrame) -> pd.Series:
    """D-09: Composite value target (OOF-safe)."""
    # clipped_log_ev = clip(log(calibrated_ev_oof), -1.0, 1.0)
    calibrated_ev = df.get("calibrated_ev_oof", df.get("ev_win_corrected", pd.Series(np.nan, index=df.index)))
    clipped_log_ev = np.log(pd.to_numeric(calibrated_ev, errors="coerce").clip(lower=1e-6)).clip(-1.0, 1.0)

    # mispricing_bonus = clipped(logit(p_model_oof) - logit(p_market_norm))
    p_model = df.get("p_win_oof", pd.Series(np.nan, index=df.index)).clip(1e-10, 1 - 1e-10)
    p_market = df.get("p_market_norm", pd.Series(np.nan, index=df.index)).clip(1e-10, 1 - 1e-10)
    logit_gap = np.log(p_model / (1 - p_model)) - np.log(p_market / (1 - p_market))
    mispricing_bonus = logit_gap.clip(-1.0, 1.0)

    # uncertainty_penalty from conformal width
    uncertainty = df.get("if_conformal_width", df.get("conformal_width", pd.Series(0.0, index=df.index)))
    uncertainty_penalty = pd.to_numeric(uncertainty, errors="coerce").fillna(0.0) * 0.1

    return clipped_log_ev + mispricing_bonus - uncertainty_penalty
```

### Robust Percentile Rank (D-27)
```python
def _race_pct_rank(
    values: pd.Series, race_id: pd.Series
) -> pd.Series:
    """D-27: Race-level robust percentile rank with deterministic tie handling."""
    return values.groupby(race_id, observed=True).rank(
        pct=True, method="average", ascending=True
    )
```

### Investment Score Combination (D-03)
```python
def compute_investment_score(
    df: pd.DataFrame, race_id: pd.Series
) -> pd.DataFrame:
    """D-03: Combine components into investment_score."""
    df["relevance_score_pct"] = _race_pct_rank(df["relevance_score"], race_id)
    df["value_score_pct"] = _race_pct_rank(df["value_score"], race_id)

    log_ev = np.log(df.get("ev_win_calibrated", df.get("ev_win_corrected", pd.Series(1.0, index=df.index))).clip(lower=1e-6))
    df["calibrated_log_ev_pct"] = _race_pct_rank(log_ev, race_id)

    uncertainty = df.get("if_conformal_width", pd.Series(0.0, index=df.index))
    df["uncertainty_penalty_pct"] = _race_pct_rank(uncertainty.fillna(0.0), race_id)

    df["investment_score"] = (
        0.35 * df["relevance_score_pct"]
        + 0.35 * df["value_score_pct"]
        + 0.20 * df["calibrated_log_ev_pct"]
        - 0.10 * df["uncertainty_penalty_pct"]
    )
    return df
```

### RacePredictor Integration Point (after MAWC, line ~277)
```python
# Source: src/backtest/race_predictor.py lines 268-277
# --- MarketAwareWinCalibrator (CAL-04) ---
mawc = getattr(submodel, "market_aware_win_calibrator", None)
if mawc is not None and mawc.is_trained:
    df = mawc.apply(df)
else:
    # Fallback normalization
    ...

# --- Race-Level Ranker (RNK-03, shadow mode) ---
ranker = getattr(submodel, "win_race_level_ranker", None)
if ranker is not None and ranker.is_trained:
    df = ranker.score(df)  # Adds investment_score columns, shadow only
```

### SubmodelSet Field Addition
```python
# Source: src/domain/models.py
@dataclass
class SubmodelSet:
    # ... existing fields ...
    market_aware_win_calibrator: MarketAwareWinCalibrator | None = None
    # Phase 40: Race-Level Ranker (RNK-01/02/03)
    win_race_level_ranker: RaceLevelRanker | None = None  # Container for 4 Ridge models
```

### ModelLoader Local Save Pattern
```python
# Source: src/pipelines/training_pipeline.py lines 2325-2331 (MAWC pattern)
if sub.win_race_level_ranker is not None and sub.win_race_level_ranker.is_trained:
    sub.win_race_level_ranker.save(
        models_dir / f"win_race_level_ranker_{surface}.joblib"
    )
```

### ModelLoader Local Load Pattern
```python
# Source: src/db/model_loader.py lines 723-732 (MAWC pattern)
win_race_level_ranker = None
rlr_file = models_dir / f"win_race_level_ranker_{surface}.joblib"
if rlr_file.is_file():
    try:
        from models.race_level_ranker import RaceLevelRanker
        win_race_level_ranker = RaceLevelRanker.load(rlr_file)
    except Exception:
        logger.warning("Failed to load %s, skipping", rlr_file)
```

## Feature Availability Audit

### D-23 Relevance Scorer Features vs Schema Registry

| D-23 Feature Name | IFF Schema Name | Status | Notes |
|-------------------|----------------|--------|-------|
| p_win_market_aware or p_win_final | `if_p_win_final` | AVAILABLE | Optional, default=NaN |
| p_win_race_rank_pct | `if_p_win_race_rank` | AVAILABLE | Derived in IFF |
| if_p_ability_win | `if_p_ability_win` | AVAILABLE | Required |
| rel_p_ability_win_rank or rel_p_ability_win_zscore | -- | NOT IN IFF | Exists in `relative_features.py` as non-if_ columns. Must add to schema or compute separately |
| if_norm_finish_avg | `if_norm_finish_avg` | AVAILABLE | Optional |
| if_closing_index | `if_closing_index` | AVAILABLE | Optional |
| if_weighted_recent_form_finish | `if_weighted_recent_form` | AVAILABLE | Optional |
| if_jockey_wr | `if_jockey_wr` | AVAILABLE | Optional |
| if_trainer_wr | `if_trainer_wr` | AVAILABLE | Optional |
| if_blood_surface_wr | `if_blood_surface_wr` | AVAILABLE | Optional |
| if_class_level | `if_class_level` | AVAILABLE | Optional |
| if_surface | `if_surface` | AVAILABLE | Required |
| if_distance_bin | `if_distance_bin` | AVAILABLE | Required |
| if_grade_code | `if_grade_code` | AVAILABLE | Optional |
| field_size | `if_n_horses` | AVAILABLE | Required |

**Relevance summary:** ~14 of 15 features available in IFF. `rel_p_ability_win_rank` / `rel_p_ability_win_zscore` need schema addition or separate computation.

### D-24 Value Scorer Features vs Schema Registry

| D-24 Feature Name | IFF Schema Name | Status | Notes |
|-------------------|----------------|--------|-------|
| if_logit_gap | `if_logit_gap` | AVAILABLE | Derived in IFF |
| if_p_diff or if_market_residual | -- | NOT IN IFF | Use `if_logit_gap` instead (equivalent) |
| if_p_ratio / if_market_value_ratio | -- | NOT IN IFF | `win_market_value_ratio` exists in win_selection_gate but not as if_* feature |
| if_edge_win | `if_edge_win` | AVAILABLE | Optional, derived in IFF |
| if_ev_calibrated or if_calibrated_log_ev | `if_ev_calibrated` | AVAILABLE | Optional. `if_calibrated_log_ev` does not exist -- must derive |
| if_odds_log | `if_odds_log` | AVAILABLE | Optional, derived in IFF |
| if_odds_band | `if_odds_band_id` | AVAILABLE | Required |
| if_odds_rank | -- | NOT IN IFF | Derive from `if_odds` + groupby rank |
| if_model_prob_rank_pct / p_win_race_rank_pct | `if_p_win_race_rank` | AVAILABLE | Derived in IFF |
| if_odds_drop_rate_60_10 | `if_odds_drop_60_10` | AVAILABLE | Optional |
| if_odds_drop_rate_30_10 | `if_odds_drop_30_10` | AVAILABLE | Optional |
| if_late_odds_drop_z | -- | NOT IN IFF | Compute from `if_odds_drop_30_10` via race-level z-score |
| if_market_share_change | -- | NOT IN IFF | No source available -- skip or use proxy |
| if_overround_proxy | `if_overround` | AVAILABLE | Use if_overround directly |
| if_market_entropy | `if_market_entropy` | AVAILABLE | Required |
| if_model_market_disagreement | -- | NOT IN IFF | No direct source -- approximate via if_abs_logit_gap |
| if_conformal_width | `if_conformal_width` | AVAILABLE | Optional, derived in IFF |
| if_ev_uncertainty_proxy | `if_ev_uncertainty_ratio` | AVAILABLE | Use if_ev_uncertainty_ratio |

**Value summary:** ~14 of 18 features available. 4 features need substitution or derivation from existing IFF features. None are blocking -- all have viable substitutes within the existing schema.

### Resolved Substitutions for Missing Features

| Missing Feature | Substitute | Rationale |
|----------------|-----------|-----------|
| if_p_diff / if_market_residual | `if_logit_gap` | Same signal -- model vs market logit difference |
| if_p_ratio / if_market_value_ratio | Add to IFF or use `if_logit_gap` + `if_edge_rank_in_race` | Could compute from if_p_win / if_implied_prob |
| if_odds_rank | Derive: `df.groupby("race_id")["if_odds"].rank(pct=True, ascending=True)` | Simple groupby rank |
| if_late_odds_drop_z | Derive: race-level zscore of `if_odds_drop_30_10` | Already computed in get_win_candidates as win_late_odds_drop_z |
| if_market_share_change | Skip or use `if_popularity_change` | D-25 says use registered missing/default for unavailable features |
| if_model_market_disagreement | `if_abs_logit_gap` | Absolute logit gap captures disagreement magnitude |
| if_ev_uncertainty_proxy | `if_ev_uncertainty_ratio` | Width/EV ratio serves same purpose |
| rel_p_ability_win_rank | Add to IFF schema or compute from `if_p_ability_win` groupby rank | Source exists in relative_features.py |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hand-tuned win_market_selection_score formula | Learned Ridge ranker with investment_score | Phase 40 | Replaces parametric scoring with data-driven ranking |
| WinSelectionPolicy surface-aware weights | Fixed combination weights (0.35/0.35/0.20/0.10) | D-03/D-04 | Simpler, more stable, not ROI-optimized |
| Binary is_win target | Graded relevance {1.00/0.55/0.30/0.10/0.00} | D-08 | Captures 2nd/3rd place information |
| Z-score race normalization | Robust percentile rank | D-27/D-28 | Stable across field sizes (5-18 horses) |

**Deprecated/outdated:**
- `win_market_selection_score` formula: Will be shadowed by investment_score. Not deleted (D-17).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `generate_win_oof_predictions()` can be extended to emit ranker columns without structural changes | OOF Training Data | May need significant refactoring if the OOF loop cannot access required columns |
| A2 | IFF build_frame(mode="train") produces all required features when called on OOF data joined with win_benter_gate output | Feature Matrix | If train-mode sources don't resolve for OOF data, features will be NaN |
| A3 | Ridge with ~30 features on ~50K training rows is sufficient for meaningful ranking | Model Choice | May need more features or different model if signal is too weak |
| A4 | LightGBM lambdarank objective is available as `objective="lambdarank"` in version 4.6.0 | Shadow Model | API may differ; needs verification |
| A5 | rel_p_ability_win_rank/zscore can be added to IFF schema or computed at ranker level | Features | If neither is possible, relevance scorer loses one useful feature |

## Open Questions

1. **OOF Data Construction Strategy**
   - What we know: D-12 says extend `generate_win_oof_predictions()`, D-14 says build from multiple sources (OOF artifacts + IFF train-mode + MAWC outputs).
   - What's unclear: Whether the OOF loop should run MAWC per fold (expensive) or whether MAWC OOF outputs can be joined post-hoc by race_id/umaban.
   - Recommendation: Join MAWC OOF outputs post-hoc. The MAWC is already trained on the same data -- its OOF outputs from Phase 39's training can be reused. This avoids the expensive per-fold MAWC retraining.

2. **Feature Schema Extensions**
   - What we know: Several D-23/D-24 features don't exist as `if_*` features. Some exist as non-IF columns (rel_p_ability_win_rank).
   - What's unclear: Whether to add missing features to IFF schema_registry or compute them inside the ranker class.
   - Recommendation: Claude's discretion area. Adding to IFF schema is cleaner but increases Phase scope. Computing inside ranker keeps changes localized. Recommend computing inside ranker for features that are simple derivations (odds_rank, late_odds_drop_z) and using existing IFF features for everything else.

3. **RacePredictor Integration Timing**
   - What we know: D-20 says after MAWC, before final sort. MAWC runs at line ~271 in predict(). Final sort is in get_win_candidates() at line ~967.
   - What's unclear: Whether ranker scoring should happen inside predict() (where df has all model outputs but not IFF features) or at the beginning of get_win_candidates() (where IFF features are available).
   - Recommendation: The ranker needs IFF features, which are built in BacktestEngine before calling RacePredictor. The most natural integration point is at the beginning of get_win_candidates() where the DataFrame already has all model outputs + IFF features. Alternatively, add a separate method `ranker.score(df)` that can be called from either predict() or get_win_candidates().

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scikit-learn | Ridge models, NDCG metrics | Yes | 1.8.0 | -- |
| LightGBM | LambdaRank shadow | Yes | 4.6.0 | -- |
| joblib | Model persistence | Yes | existing | -- |
| pandas | DataFrame operations | Yes | existing | -- |
| numpy | Array operations | Yes | existing | -- |
| scipy | Spearman correlation | Yes | existing | -- |

**Missing dependencies with no fallback:** None

**Missing dependencies with fallback:** None

## Validation Architecture

> workflow.nyquist_validation is explicitly false in .planning/config.json. Skip this section.

## Security Domain

> No external-facing endpoints, no authentication, no user input validation, no cryptography. This phase adds internal ML models only.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | Yes | IFF schema_registry validates feature names; schema enforcement via InvestmentFeatureSpec |
| V6 Cryptography | No | N/A |

### Known Threat Patterns for Learned Ranker

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Training data leakage (OOF contamination) | Tampering | OOFHealthValidator + D-15 OOF-only probability ranks |
| Feature routing pollution (ranker features in MarketModel) | Tampering | SAF-01 audit in Phase 42 |
| Shadow mode bypass (premature deployment) | Elevation | `_trained` + `deployment_status` guard, same as MAWC |
| Overfitting to small dataset | Tampering | L2 regularization via Ridge alpha grid, WF fold validation |

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/models/market_aware_win_calibrator.py` -- shadow mode pattern, joblib save/load, alpha grid selection
- Codebase analysis: `src/pipelines/training_pipeline.py` -- training flow, OOF generation, model save/load patterns
- Codebase analysis: `src/backtest/race_predictor.py` -- integration points, current selection logic
- Codebase analysis: `src/investment/schema_registry.py` -- 94 features across 9 categories, verified feature availability
- Codebase analysis: `src/domain/models.py` -- SubmodelSet field naming pattern
- Codebase analysis: `src/db/model_loader.py` -- local + MLflow load/save patterns
- Codebase analysis: `src/models/win_benter_gate.py` -- `generate_win_oof_predictions()` function structure
- Codebase analysis: `src/features/relative_features.py` -- rel_p_ability_win_rank/zscore availability

### Secondary (MEDIUM confidence)
- CONTEXT.md D-01 through D-28 -- verified against codebase patterns
- Phase 39 CONTEXT.md -- verified shadow mode and MAWC integration patterns

### Tertiary (LOW confidence)
- LightGBM LambdaRank API (`objective="lambdarank"`) -- [ASSUMED] based on LightGBM documentation and version 4.6.0

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all dependencies verified in codebase, no new packages needed
- Architecture: HIGH -- follows exact MAWC pattern, integration points confirmed via code analysis
- Pitfalls: HIGH -- feature availability audit complete, gaps identified with substitutes
- Feature availability: HIGH -- schema registry fully analyzed, all D-23/D-24 features mapped

**Research date:** 2026-05-28
**Valid until:** 2026-06-28 (stable -- no fast-moving dependencies)
