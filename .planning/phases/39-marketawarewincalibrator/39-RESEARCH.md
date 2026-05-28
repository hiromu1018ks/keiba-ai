# Phase 39: MarketAwareWinCalibrator - Research

**Researched:** 2026-05-27
**Domain:** sklearn LogisticRegression + L2 logit-blend calibration for win probability
**Confidence:** HIGH

## Summary

Phase 39 replaces the two-component WinBenterGate + WinSegmentCalibrator chain with a single MarketAwareWinCalibrator that uses sklearn LogisticRegression with L2 regularization to blend model and market logits. The calibrator incorporates segment conditioning features (popularity rank, odds band, probability rank) as regularized features and interactions rather than per-segment coefficients, preventing the sparse segment overfitting that plagued WinSegmentCalibrator.

The existing BenterCombination class provides the conceptual foundation -- its `alpha*logit(p_fund) + beta*logit(p_market) + gamma` formula is generalized by LogisticRegression's learned coefficients. The key architectural change is treating segment conditioning as regularized interactions within a global model rather than separate per-segment shrinkage factors. The `generate_win_oof_predictions()` function already generates OOF predictions for the current WinBenterGate fitting; it must be extended to emit additional columns (popularity_rank, tanodds, race_id, umaban, field_size, surface, race_date) needed for calibrator feature construction.

**Primary recommendation:** Build MarketAwareWinCalibrator as a standalone class in `src/models/market_aware_win_calibrator.py` using sklearn LogisticRegression (sklearn 1.8 new API without `penalty` param), train it from extended OOF data with C-selection via WF grid search, and integrate it at the same pipeline position as the current WinBenterGate (RacePredictor.predict() lines 282-290).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** LogisticRegression + L2 regularization (sklearn) as primary deployable model
- **D-02:** LightGBM may be trained only as a shadow benchmark, not default deployable
- **D-03:** beta_market effective contribution must retain floor/guard equivalent to beta_market >= 0.20; enforce via coefficient inspection
- **D-04:** C selection via deterministic WF grid search over [0.03, 0.1, 0.3, 1.0, 3.0]; primary metric: logloss; tie-breaker: smaller C
- **D-05:** Race-level chronological folds for C selection; require year-level and surface-level actual/predicted ratio not to worsen materially; if no C passes gates, do not deploy
- **D-06:** Feature interactions: main effects + logit(p_model) x segment + logit(p_market) x segment only; no segment x segment
- **D-07:** fit_intercept=True (Benter gamma equivalent for global calibration bias correction)
- **D-08:** Both continuous log_odds = log1p(tanodds) AND coarse odds_band categorical for interactions
- **D-09:** Fixed odds bands: [1.0,2.0), [2.0,3.0), [3.0,5.0), [5.0,10.0), [10.0,30.0), [30.0,100.0), [100.0+)
- **D-10:** One-hot encoding for odds_band (not ordinal)
- **D-11:** Fixed category order; all 7 expected columns present even if band absent in fold
- **D-12:** popularity_rank_pct = popularity_rank / field_size clipped [0,1]
- **D-13:** popularity_bucket one-hot: pop_1, pop_2_3, pop_4_6, pop_7_9, pop_10_plus
- **D-14:** p_win_race_rank_pct continuous [0,1] percentile
- **D-15:** p_rank bucket one-hot: top 25%, mid 25-75%, bottom 25%
- **D-16:** All bucket boundaries are fixed, not data-fitted
- **D-17:** Total feature dimensions: ~51
- **D-18:** Hybrid OOF approach: OOF-dependent features recomputed; market/static joined from InvestmentFeatureFrame
- **D-19:** OOF-dependent (MUST recompute): p_model_oof, p_win_race_rank_pct, p_rank buckets
- **D-20:** Market/static (reusable from IFF): p_market_norm, logit(p_market_norm), tanodds, log_odds, odds_band, popularity_rank_pct, popularity_bucket, field_size, surface, race_date, race_id, umaban
- **D-21:** Extend existing generate_win_oof_predictions() to emit additional columns
- **D-22:** Tests must verify train-mode p_win_pred is REJECTED, p_win_oof is used
- **D-23:** Prefer simplest model that passes gates; if none pass, shadow-only

### Claude's Discretion
- Feature matrix standardization strategy for non-logit segment features
- Exact implementation of checkpoint file and incremental save logic
- Test structure and naming within existing conventions
- Model serialization format (joblib consistent with existing patterns)

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CAL-01 | MarketAwareWinCalibrator generates Benter-type logit(p_model) + logit(p_market) blended probabilities with segment conditioning | LogisticRegression with L2 on feature matrix [logit(p_model_oof), logit(p_market_norm), segment features, interactions]; D-01, D-06 |
| CAL-02 | Segment conditioning uses popularity rank, odds band, probability rank from InvestmentFeatureFrame output | IFF already provides if_popularity_rank, if_p_win_race_rank, if_odds; D-12 through D-16 define encoding; D-18/D-19/D-20 define hybrid OOF approach |
| CAL-03 | Segment effects are regularized features/interactions in global calibrator, not per-segment coefficients | LogisticRegression L2 regularization on ~51-dim feature matrix; D-06 restricts interactions to logit x segment only; no segment x segment |
| CAL-04 | MarketAwareWinCalibrator replaces WinBenterGate + WinSegmentCalibrator, preventing double correction | Remove win_benter_gate.py and win_segment_calibrator.py references from RacePredictor, TrainingPipeline, ModelLoader, SubmodelSet |
| CAL-05 | Calibrator output maintains probability quality (Brier/logloss/ECE) and sum-to-1.0 constraint | C-selection WF grid search (D-04/D-05); race-level normalization p_final = p_raw / p_raw.sum() per race; D-03 beta_market guard |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| OOF feature matrix construction | Training Pipeline | -- | OOF predictions generated during training only; feature matrix built from OOF results |
| Calibrator model training (LogisticRegression fit) | Training Pipeline | -- | Model parameters learned during training via C-selection WF grid search |
| C-selection grid search | Training Pipeline | -- | Deterministic hyperparameter selection on WF folds |
| Race-level probability calibration (inference) | RacePredictor | -- | Apply trained calibrator at inference time (same position as current WinBenterGate) |
| Race normalization (sum-to-1.0) | RacePredictor | -- | Post-calibration normalization per race_id groupby |
| Feature encoding (one-hot, percentile) | MarketAwareWinCalibrator | -- | Encoding logic lives within the calibrator class for consistency |
| Model persistence (save/load) | ModelLoader | -- | joblib serialization consistent with existing patterns |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.8.0 | LogisticRegression with L2 for calibrator | Already installed; used throughout codebase for IsotonicRegression, calibration metrics, TimeSeriesSplit |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 2.4.3 | Logit computation, array operations | Feature matrix construction |
| pandas | 2.3.3 | DataFrame operations, groupby rank | OOF data manipulation |
| joblib | (bundled) | Model serialization | Save/load calibrator (consistent with existing WinSegmentCalibrator, WinSelectionGate patterns) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sklearn LogisticRegression | scipy.optimize.minimize (current BenterCombination) | LogisticRegression provides L2 regularization, predict_proba, C grid search natively; scipy requires manual implementation of regularization and probability output |
| joblib serialization | JSON serialization (current BenterCombination) | joblib handles sklearn objects natively; JSON would require custom serialization of coef_, intercept_, feature_names_ |

**Installation:**
```bash
# No new dependencies required -- all libraries already installed
pip install -e ".[dev]"  # existing installation covers everything
```

**Version verification:**
```
scikit-learn==1.8.0 [VERIFIED: runtime check]
numpy==2.4.3 [VERIFIED: runtime check]
pandas==2.3.3 [VERIFIED: runtime check]
```

## Package Legitimacy Audit

> No new packages installed in this phase. All dependencies (scikit-learn, numpy, pandas, joblib) are existing project dependencies.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| scikit-learn | PyPI | 14+ yrs | 50M+/mo | github.com/scikit-learn/scikit-learn | N/A (existing) | Approved |
| numpy | PyPI | 14+ yrs | 100M+/mo | github.com/numpy/numpy | N/A (existing) | Approved |
| pandas | PyPI | 14+ yrs | 60M+/mo | github.com/pandas-dev/pandas | N/A (existing) | Approved |
| joblib | PyPI | 14+ yrs | 40M+/mo | github.com/joblib/joblib | N/A (existing) | Approved |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
[Training Pipeline]
       |
       v
[generate_win_oof_predictions()] -- extended to emit additional columns
       |
       v
[OOF DataFrame] -- p_model_oof, tanodds, popularity_rank, field_size, surface, race_date, race_id, umaban
       |
       v
[MarketAwareWinCalibrator.build_feature_matrix()]
       |  Input: OOF df + IFF-joined market/static columns
       |  Output: X (N x ~51), y (N,)
       |  Steps:
       |    1. Compute logit(p_model_oof), logit(p_market_norm)
       |    2. Compute continuous: log_odds, popularity_rank_pct, p_win_race_rank_pct
       |    3. One-hot encode: odds_band(7), popularity_bucket(5), p_rank_bucket(3)
       |    4. Build interactions: logit_model x segment(15) + logit_market x segment(15)
       v
[C-Selection WF Grid Search] -- [0.03, 0.1, 0.3, 1.0, 3.0]
       |  Metric: logloss primary, Brier secondary, ECE tertiary
       |  Tie-break: smaller C (stronger regularization)
       |  Guard: beta_market >= 0.20 equivalent
       |  Guard: year/surface actual/predicted ratio not worsen
       v
[LogisticRegression.fit(X, y)] with best C
       |
       v
[joblib save] --> market_aware_win_calibrator_{surface}.joblib


[Inference: RacePredictor.predict()]
       |
       v
[EV Correction] -- p_win_corrected produced
       |
       v
[MarketAwareWinCalibrator.apply(df)]  <-- REPLACES WinBenterGate.apply()
       |  Input: df with p_win_corrected, tanodds, popularity_rank, field_size
       |  Steps:
       |    1. Extract p_model = p_win_corrected
       |    2. Compute p_market = clip(1/tanodds, 0.01, 0.99)
       |    3. Build feature matrix (same encoder as training)
       |    4. p_raw = LogisticRegression.predict_proba(X)[:, 1]
       |    5. p_win_final = p_raw / p_raw.groupby(race_id).sum()
       |    6. edge_win = p_win_final * tanodds - 1.0
       v
[WinSelectionGate] -- unchanged downstream
```

### Recommended Project Structure
```
src/models/
  market_aware_win_calibrator.py    # NEW: MarketAwareWinCalibrator class
  benter_combination.py             # RETAINED: still used by place prediction
  win_benter_gate.py                # REMOVED: replaced by market_aware_win_calibrator
  win_segment_calibrator.py         # REMOVED: replaced by market_aware_win_calibrator
```

### Pattern 1: LogisticRegression Calibrator with Fixed One-Hot Schema
**What:** LogisticRegression trained on a fixed-dimension feature matrix with one-hot encoded categoricals, where all expected columns are present even if a category is absent in a fold.
**When to use:** Calibrator training and inference -- ensures consistent feature dimensionality across folds and deployment.
**Example:**
```python
# Source: sklearn 1.8 API [VERIFIED: runtime check]
from sklearn.linear_model import LogisticRegression

# sklearn 1.8: penalty param is DEPRECATED, l1_ratio=0.0 (default) = L2
calibrator = LogisticRegression(
    C=best_c,           # Inverse regularization strength
    max_iter=1000,
    fit_intercept=True,  # D-07: intercept = Benter gamma equivalent
)
calibrator.fit(X_train, y_train)

# Predict calibrated probabilities
p_raw = calibrator.predict_proba(X)[:, 1]

# Race-level normalization (sum-to-1.0)
p_final = p_raw / df.groupby("race_id")["p_raw"].transform("sum")
```

### Pattern 2: Fixed One-Hot Encoding with Guaranteed Schema
**What:** One-hot encode categoricals with fixed bins, ensuring all expected columns exist in output.
**When to use:** Feature matrix construction for calibrator training and inference.
**Example:**
```python
# D-09: Fixed odds bands
ODDS_BAND_EDGES = [1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, float("inf")]
ODDS_BAND_NAMES = ["1-2", "2-3", "3-5", "5-10", "10-30", "30-100", "100+"]

def encode_odds_band_onehot(tanodds: pd.Series) -> pd.DataFrame:
    """One-hot encode with ALL 7 bands present (D-11)."""
    band = pd.cut(tanodds, bins=ODDS_BAND_EDGES, labels=ODDS_BAND_NAMES, right=False)
    dummies = pd.get_dummies(band, dtype=float)
    # Ensure all 7 expected columns exist
    for name in ODDS_BAND_NAMES:
        if name not in dummies.columns:
            dummies[name] = 0.0
    return dummies[ODDS_BAND_NAMES]  # Fixed order
```

### Anti-Patterns to Avoid
- **Per-segment coefficients (WinSegmentCalibrator approach):** Creates sparse segment tables with few samples per segment, leading to unstable estimates. Use regularized interactions instead.
- **Segment x segment interactions:** Recreates the overfitting problem of WinSegmentCalibrator by creating high-dimensional sparse interaction space. D-06 explicitly forbids this.
- **Data-fitted bucket boundaries:** Would create fold-dependent schemas and prevent consistent feature dimensions. D-16 mandates fixed boundaries.
- **sklearn `penalty='l2'` parameter:** Deprecated in sklearn 1.8, will be removed in 1.10. Use default (no penalty param, l1_ratio=0.0 default = L2).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Logit-space blending with regularization | Manual scipy.optimize like BenterCombination | sklearn LogisticRegression | L2 regularization built-in, predict_proba for probabilities, C grid search standardized |
| Feature one-hot encoding | Manual mapping dict | pd.get_dummies + fixed schema alignment | pd.get_dummies handles edge cases, fixed schema ensures deployment consistency |
| C grid search | Custom WF loop with manual metric tracking | sklearn-style grid with manual WF splits | WF splits already exist in codebase (_walk_forward_race_splits) |
| Calibration metrics | Custom ECE/Brier | sklearn.metrics.brier_score_loss + compute_ece (existing) | compute_ece already exists in win_benter_gate.py, brier_score_loss from sklearn |

**Key insight:** The existing BenterCombination uses scipy.optimize.minimize for 3-parameter fitting (alpha, beta, gamma). LogisticRegression generalizes this to N-parameter fitting with built-in L2 regularization -- the intercept maps to gamma, and the coefficients map to alpha/beta plus segment interactions. No need to reimplement optimization.

## Common Pitfalls

### Pitfall 1: sklearn 1.8 `penalty` Parameter Deprecation
**What goes wrong:** Using `penalty='l2'` triggers FutureWarning and will break in sklearn 1.10.
**Why it happens:** sklearn 1.8 deprecated the `penalty` parameter in favor of `l1_ratio` control.
**How to avoid:** Do NOT pass `penalty` parameter at all. Default `l1_ratio=0.0` gives L2 regularization.
**Warning signs:** FutureWarning in test output, CI failures when sklearn upgrades.
**Confidence:** HIGH -- [VERIFIED: runtime test]

### Pitfall 2: Missing One-Hot Columns in Sparse Folds
**What goes wrong:** A WF fold may not contain horses in odds_band [100.0+), causing the one-hot column to be absent, creating dimension mismatch between training and inference.
**Why it happens:** Rare odds bands may not appear in every fold of training data.
**How to avoid:** Always ensure all 7 odds_band columns, 5 popularity_bucket columns, and 3 p_rank_bucket columns exist in the feature matrix, filling absent categories with 0.0 (D-11).
**Warning signs:** LogisticRegression n_features_in_ differs between folds or between train/inference.
**Confidence:** HIGH -- pattern already used in existing WinSegmentCalibrator and schema_registry.

### Pitfall 3: Normalization Breaking Probability Quality
**What goes wrong:** Dividing by race sum can distort the calibrated probabilities if the calibrator already produces well-calibrated outputs.
**Why it happens:** LogisticRegression produces independent binary probabilities, not multinomial ones. Race normalization is needed for sum-to-1.0 but may hurt calibration.
**How to avoid:** Verify Brier/logloss/ECE AFTER normalization (D-04, D-05). STATE.md explicitly flags this as a known concern. If quality degrades, the C-selection grid search must account for post-normalization metrics.
**Warning signs:** Post-normalization ECE significantly worse than pre-normalization ECE.
**Confidence:** HIGH -- STATE.md identifies this as a known blocker.

### Pitfall 4: Train-Mode Leakage via p_win_pred
**What goes wrong:** Using p_win_pred (train-mode predictions) instead of p_win_oof for segment features causes data leakage.
**Why it happens:** p_win_pred contains in-sample predictions that are systematically overconfident.
**How to avoid:** D-22 mandates tests that REJECT p_win_pred and REQUIRE p_win_oof for all probability-dependent features.
**Warning signs:** Unusually high training-set calibration metrics that don't generalize.
**Confidence:** HIGH -- existing generate_win_oof_predictions already handles this.

### Pitfall 5: Beta_Market Guard Violation
**What goes wrong:** LogisticRegression L2 may shrink the logit(p_market) coefficient below effective 0.20 contribution threshold.
**Why it happens:** Strong regularization (small C) can suppress individual coefficients.
**How to avoid:** D-03 requires coefficient inspection on the logit(p_market_norm) column. If effective contribution < 0.20, deployment fails.
**Warning signs:** coef_[logit_market_idx] * mean(|logit_market|) < 0.20 * mean(|logit_fund|) equivalent.
**Confidence:** MEDIUM -- exact interpretation of "effective contribution >= 0.20" needs clarification during planning.

## Code Examples

### MarketAwareWinCalibrator Core Structure
```python
# Source: sklearn 1.8 API [VERIFIED: runtime check]
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

@dataclass
class MarketAwareWinCalibrator:
    """Benter-type logit-blend calibrator with segment conditioning.

    Replaces WinBenterGate + WinSegmentCalibrator with a single
    LogisticRegression + L2 model that blends model and market logits
    with regularized segment features and interactions.
    """

    calibrator: LogisticRegression | None = None
    feature_names: list[str] = field(default_factory=list)
    best_c: float | None = None
    c_selection_results: dict[str, Any] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)
    _trained: bool = False

    # D-09: Fixed odds bands
    ODDS_BAND_EDGES: ClassVar[list[float]] = [1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, float("inf")]
    ODDS_BAND_NAMES: ClassVar[list[str]] = ["1-2", "2-3", "3-5", "5-10", "10-30", "30-100", "100+"]

    # D-13: Fixed popularity buckets
    POP_BUCKET_EDGES: ClassVar[list[float]] = [0, 1.5, 3.5, 6.5, 9.5, float("inf")]
    POP_BUCKET_NAMES: ClassVar[list[str]] = ["pop_1", "pop_2_3", "pop_4_6", "pop_7_9", "pop_10_plus"]

    # D-15: Fixed p_rank buckets
    P_RANK_NAMES: ClassVar[list[str]] = ["top_25", "mid_25_75", "bottom_25"]

    C_GRID: ClassVar[list[float]] = [0.03, 0.1, 0.3, 1.0, 3.0]

    @property
    def is_trained(self) -> bool:
        return self._trained and self.calibrator is not None
```

### Feature Matrix Construction
```python
# Source: pattern from existing InvestmentFeatureFrame + CONTEXT.md D-17
def build_feature_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Build ~51-dim feature matrix for LogisticRegression."""
    eps = 1e-10

    # Main effects (6 continuous)
    logit_model = np.log(np.clip(df["p_model"].values, eps, 1 - eps) /
                         np.clip(1 - df["p_model"].values, eps, 1 - eps))
    logit_market = np.log(np.clip(df["p_market"].values, eps, 1 - eps) /
                          np.clip(1 - df["p_market"].values, eps, 1 - eps))
    log_odds = np.log1p(df["tanodds"].values)
    popularity_rank_pct = (df["popularity_rank"] / df["field_size"]).clip(0, 1).values
    p_win_race_rank_pct = df["p_win_race_rank_pct"].values  # computed from OOF

    # One-hot encodings (7 + 5 + 3 = 15 segment features)
    odds_band_oh = self._encode_odds_band(df["tanodds"])      # (N, 7)
    pop_bucket_oh = self._encode_pop_bucket(df["popularity_rank"], df["field_size"])  # (N, 5)
    p_rank_oh = self._encode_p_rank(df["p_win_race_rank_pct"])  # (N, 3)

    # Interactions: logit x segment (D-06) -- 30 features
    segment_features = np.hstack([odds_band_oh, pop_bucket_oh, p_rank_oh])  # (N, 15)
    interactions = np.hstack([
        logit_model[:, None] * segment_features,   # (N, 15)
        logit_market[:, None] * segment_features,   # (N, 15)
    ])

    # Assemble: 6 main + 15 one-hot + 30 interactions = 51
    X = np.hstack([
        np.column_stack([logit_model, logit_market, log_odds,
                         popularity_rank_pct, p_win_race_rank_pct,
                         df["field_size"].values]),
        segment_features,
        interactions,
    ])
    return X
```

### C-Selection with WF Grid Search
```python
# Source: CONTEXT.md D-04, D-05
def select_c(self, X: np.ndarray, y: np.ndarray,
             race_ids: np.ndarray) -> float:
    """Deterministic WF grid search over C values (D-04)."""
    best_c = None
    best_logloss = float("inf")

    for c in self.C_GRID:
        fold_losses = []
        for train_idx, val_idx in wf_splits:
            lr = LogisticRegression(C=c, max_iter=1000, fit_intercept=True)
            lr.fit(X[train_idx], y[train_idx])
            p_val = lr.predict_proba(X[val_idx])[:, 1]
            fold_losses.append(log_loss(y[val_idx], p_val))

        mean_loss = np.mean(fold_losses)
        if (mean_loss < best_logloss or
            (mean_loss == best_logloss and c < best_c)):  # D-04 tie-breaker
            best_logloss = mean_loss
            best_c = c

    return best_c
```

### Inference Apply Pattern
```python
# Source: mirrors WinBenterGate.apply() pattern
def apply(self, df: pd.DataFrame) -> pd.DataFrame:
    """Apply calibrator to inference DataFrame (replaces WinBenterGate)."""
    df = df.copy()

    # Build feature matrix for inference
    df["p_model"] = df["p_win_corrected"]  # After EV correction
    df["p_market"] = np.clip(1.0 / df["tanodds"].values, 0.01, 0.99)

    # Compute segment features at inference time
    df["popularity_rank_pct"] = (df["popularity_rank"] / df["field_size"]).clip(0, 1)
    df["p_win_race_rank_pct"] = df.groupby("race_id")["p_model"].rank(
        pct=True, method="min", ascending=False
    )

    X = self.build_feature_matrix(df)
    p_raw = self.calibrator.predict_proba(X)[:, 1]

    df["p_win_combined"] = p_raw
    # D-09/D-10: Race normalization (sum-to-1.0)
    race_sums = df.groupby("race_id", observed=True)["p_win_combined"].transform("sum")
    df["p_win_final"] = df["p_win_combined"] / race_sums
    df["edge_win"] = df["p_win_final"] * df["tanodds"] - 1.0

    return df
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| scipy.optimize NLL minimization (BenterCombination) | sklearn LogisticRegression + L2 | Phase 39 | Built-in regularization, standardized API, C grid search |
| Per-segment shrinkage factors (WinSegmentCalibrator) | Regularized feature interactions in global model | Phase 39 | Prevents sparse overfitting, unified model |
| Dual gate+segment chain | Single calibrator | Phase 39 | Eliminates double correction, simpler pipeline |
| sklearn `penalty='l2'` | Default (no penalty param, l1_ratio=0.0) | sklearn 1.8 | Avoid deprecation warning |

**Deprecated/outdated:**
- `penalty='l2'` in sklearn 1.8: deprecated, will be removed in 1.10. Use default behavior instead.
- WinBenterGate: entire file removed, replaced by MarketAwareWinCalibrator
- WinSegmentCalibrator: entire file removed, replaced by MarketAwareWinCalibrator

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `popularity_rank` column is available in the OOF training data | Feature Matrix | Need to verify column exists in OOF artifact or add to generate_win_oof_predictions() |
| A2 | `field_size` column is available or computable from OOF data | Feature Matrix | Need to verify or compute from groupby count |
| A3 | beta_market >= 0.20 guard means: the coefficient on logit(p_market) column should have effective contribution >= 0.20 relative to logit(p_model) | D-03 Guard | Exact interpretation needs planner/human confirmation |
| A4 | sklearn 1.8 default LogisticRegression (no penalty param, l1_ratio=0.0) produces identical L2 regularization behavior | Standard Stack | Verified via runtime test -- default produces non-zero coefficients scaled by C |
| A5 | No new pip dependencies needed -- sklearn 1.8.0 already installed | Standard Stack | If sklearn missing, `pip install -e ".[dev]"` covers it |

## Open Questions (RESOLVED)

1. **Beta_market guard interpretation (D-03)** — RESOLVED
   - Decision: Implement as relative coefficient check: `abs(coef_market) / (abs(coef_model) + abs(coef_market)) >= 0.20`, matching BenterCombination's alpha/beta semantics.
   - Rationale: This matches the Benter alpha/beta ratio interpretation and is invariant to coefficient scale.

2. **p_win_race_rank_pct computation at inference time** — RESOLVED
   - Decision: Compute from `p_win_corrected` at inference time (same source used for `logit_model`), since calibrator hasn't been applied yet.
   - Rationale: p_win_corrected is the best available probability estimate before calibration; using raw p_win_pred would be inconsistent with the logit_model input.

3. **Standardization of non-logit features** — RESOLVED
   - Decision: Skip standardization. L2 regularization handles the small fixed feature set with known ranges. Logit features ~[-5,5], log_odds ~[0,5], pct features ~[0,1], field_size ~[8,18].
   - Rationale: Adding StandardScaler would introduce an extra serialization step and fold-dependent scaling, complicating save/load for negligible benefit with ~51 features.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | All | Yes (mise) | 3.11.15 | -- |
| scikit-learn | LogisticRegression calibrator | Yes | 1.8.0 | -- |
| numpy | Array operations | Yes | 2.4.3 | -- |
| pandas | DataFrame manipulation | Yes | 2.3.3 | -- |
| joblib | Model serialization | Yes | bundled | -- |
| LightGBM | Shadow benchmark (D-02) | Yes | 4.6.0 | Shadow-only is optional |
| pytest | Tests | Yes | -- | -- |

**Missing dependencies with no fallback:** None
**Missing dependencies with fallback:** None

## Sources

### Primary (HIGH confidence)
- [sklearn 1.8 LogisticRegression API] -- runtime verified: penalty deprecated, l1_ratio=0.0 default = L2, predict_proba[:, 1] for binary probability
- [CONTEXT.md D-01 through D-23] -- locked decisions from discuss-phase
- [Codebase: src/models/win_benter_gate.py] -- current BenterCombination integration, generate_win_oof_predictions(), compute_ece(), compare_calibrations()
- [Codebase: src/models/win_segment_calibrator.py] -- current segment calibrator to be replaced
- [Codebase: src/models/benter_combination.py] -- BenterCombination class retained for place prediction
- [Codebase: src/investment/schema_registry.py] -- IFF feature specs including if_popularity_rank, if_p_win_race_rank, if_odds
- [Codebase: src/investment/feature_frame.py] -- IFF builder with _compute_derived for if_p_win_race_rank_pct
- [Codebase: src/domain/models.py] -- SubmodelSet dataclass with fields to remove/add
- [Codebase: src/backtest/race_predictor.py] -- RacePredictor integration points (lines 153-164, 282-290, 657-679)
- [Codebase: src/pipelines/training_pipeline.py] -- Win Benter training block (lines 1329-1417), segment calibrator training (lines 1624-1634), OOF artifact preparation
- [Codebase: src/db/model_loader.py] -- Model load/save patterns for MLflow and local

### Secondary (MEDIUM confidence)
- [CLAUDE.md] -- project conventions (ruff, mypy, pytest patterns, joblib serialization)

### Tertiary (LOW confidence)
- None -- all findings verified from codebase or runtime checks

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- sklearn 1.8.0 verified at runtime, no new dependencies
- Architecture: HIGH -- detailed canonical refs from CONTEXT.md, code fully read
- Pitfalls: HIGH -- sklearn 1.8 deprecation verified, normalization concern from STATE.md, leakage pattern from existing code
- Feature dimensions: MEDIUM -- ~51 dims per D-17 is a design decision; actual count depends on interaction construction detail

**Research date:** 2026-05-27
**Valid until:** 2026-06-27 (stable -- no fast-moving dependencies)
