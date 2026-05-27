# Phase 39: MarketAwareWinCalibrator - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 39-MarketAwareWinCalibrator
**Areas discussed:** Calibrator Model Type, Odds Band Definition, Segment Feature Encoding, OOF Training Data Generation

---

## Calibrator Model Type

| Option | Description | Selected |
|--------|-------------|----------|
| Extended Benter parametric | Add segment interaction terms to existing alpha/beta/gamma MLE. 10-15 params. No new deps. | |
| LogisticRegression + L2 | Feature vector [logit(p_model), logit(p_market), segments, interactions] with sklearn LogisticRegression. CV for C. | ✓ |
| LightGBM (small) | Shallow trees (depth=3-4) for non-linear interactions. Higher complexity. | |

**User's choice:** LogisticRegression + L2 as primary deployable model, with Benter-style feature basis. LightGBM as shadow benchmark only.
**Notes:**
- beta_market effective contribution must retain floor >= 0.20 (deployment fails if violated)
- No independent per-segment MLE models; segment effects are features/interactions in one global regularized model
- Prefer simplest model that passes OOF/WF probability-quality gates

### C Selection Method

| Option | Description | Selected |
|--------|-------------|----------|
| CV-based C selection (WF grid) | Deterministic grid [0.03, 0.1, 0.3, 1.0, 3.0], primary logloss, no Optuna | ✓ |
| Fixed C=1.0 | Default sklearn C, adjust only if gates fail | |
| Fixed C=1.0 (no CV) | No cross-validation at all | |

**User's choice:** Deterministic WF grid [0.03, 0.1, 0.3, 1.0, 3.0]. Primary logloss. Tie-breaker smaller C. Optuna not needed for tiny search space.

### Interaction Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Logit × segment only | logit(p_model)×segment and logit(p_market)×segment only. Min params. | ✓ |
| Segment × segment also | Include e.g. popularity×odds_band. Higher dims, overfit risk. | |
| No interactions (main effects only) | Simplest. Limited segment conditioning effect. | |

**User's choice:** Main effects + logit × segment interactions only. Segment × segment excluded to prevent recreating WSC overfit problem.

### Intercept

| Option | Description | Selected |
|--------|-------------|----------|
| fit_intercept=True | Benter gamma equivalent, global calibration bias correction | ✓ |
| fit_intercept=False | No bias term | |

**User's choice:** fit_intercept=True. Intercept is Benter gamma equivalent for global bias correction.

---

## Odds Band Definition

### Odds Representation

| Option | Description | Selected |
|--------|-------------|----------|
| Continuous log-odds only | log1p(tanodds) as single feature. No banding. | |
| Existing bands only | WSC bands (1-2,2-5,5-10,10-30,30-100,100+) | |
| Data-driven bands | Quantile-based. Unstable boundaries. | |
| Both continuous + coarse bands | log1p(tanodds) + 7 coarse bands [1-2,2-3,3-5,5-10,10-30,30-100,100+] | ✓ |

**User's choice:** Use both continuous log_odds and coarse odds_band. Continuous captures smooth favorite-longshot effect; bands allow stable segment-specific calibration shifts. Boundaries: [1.0,2.0), [2.0,3.0), [3.0,5.0), [5.0,10.0), [10.0,30.0), [30.0,100.0), [100.0+].

### Odds Band Encoding

| Option | Description | Selected |
|--------|-------------|----------|
| Ordinal (0-6) | 1 dim, linear spacing assumption | |
| One-hot (7 dims) | Independent coefficients per band, L2 controls | ✓ |

**User's choice:** One-hot encoding. Ordinal imposes linear spacing inappropriate for favorite-longshot bias. Fixed category order, all columns present even if absent in fold.

---

## Segment Feature Encoding

### Popularity Rank

| Option | Description | Selected |
|--------|-------------|----------|
| 3-bucket (1-3/4-6/7+) | Simple but too coarse (merges 7th with 15th) | |
| Continuous (1-18) | Simple, linear assumption | |
| 4-bucket (1/2-3/4-6/7+) | Finer, JRA bet-type boundaries | |
| Continuous + 5-bucket one-hot | popularity_rank_pct + pop_1/2-3/4-6/7-9/10+ | ✓ |

**User's choice:** Both continuous popularity_rank_pct (rank/field_size) and 5-bucket one-hot. 3-bucket too coarse (merges 7th with 15th). 5-bucket captures favorite/mid/longshot structure cleanly.

### Probability Rank

| Option | Description | Selected |
|--------|-------------|----------|
| Continuous [0,1] only | Already percentile, simplest | |
| 3-bucket (top/mid/bottom) | Non-linearity via buckets | |
| Continuous + 3-bucket | Both main effect and interaction buckets | ✓ |

**User's choice:** Continuous p_win_race_rank_pct as main effect + 3-bucket one-hot (top25%/mid25-75%/bottom25%) for interactions. Fixed boundaries, not data-fitted.

---

## OOF Training Data Generation

### Data Source Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Full OOF recomputation | All features recomputed in OOF loop. Leak-free but complex. | |
| Reuse IFF values | Use existing InvestmentFeatureFrame. Simple but leak risk. | |
| Hybrid | OOF-dependent recompute, static/market reuse from IFF | ✓ |

**User's choice:** Hybrid. OOF-dependent (p_model_oof, p_win_race_rank_pct, p_rank buckets) recomputed. Market/static (p_market_norm, tanodds, odds_band, popularity_rank_pct, field_size, surface, race_date) joined from IFF by race_id/umaban.

### OOF Loop Implementation

| Option | Description | Selected |
|--------|-------------|----------|
| Extend existing generate_win_oof_predictions() | Same folds, emit additional columns. Consistent. | ✓ |
| New dedicated OOF loop | Independent but duplicate code risk. | |

**User's choice:** Extend existing loop. If it cannot expose needed columns cleanly, refactor into reusable helper first. Same fold definitions. Record fold metadata at generation time.

---

## Claude's Discretion

- Feature matrix standardization for non-logit segment features
- Checkpoint file implementation details
- Test structure and naming
- Model serialization format (joblib consistent with existing patterns)

## Deferred Ideas

None — discussion stayed within phase scope.
