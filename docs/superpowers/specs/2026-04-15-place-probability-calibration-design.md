# Place Probability Calibration Design

**Date:** 2026-04-15
**Status:** Draft
**Target:** ROI 65.3% → 100%+ (profitability recovery)

## Problem Statement

Backtest results (2025 test, Benter alpha=0.4) show catastrophic losses:

| Metric | Value |
|---|---|
| ROI | 65.3% (losing 34.7% per bet) |
| Bets | 2,772 |
| Avg p_place_pred | 0.5118 |
| Actual hit rate | 0.215 |
| **Overestimation factor** | **2.38x** |

Root cause analysis identifies three critical issues:

1. **No calibration on p_place_pred**: PlaceTwoStageModel hit model outputs raw LightGBM probabilities without isotonic calibration, temperature scaling, or race-sum normalization.
2. **Selection bias amplification**: Edge-based betting selects horses where p_place_pred >> p_market, which are precisely the most overestimated horses (gap +0.136 to +0.441).
3. **Fixed Benter alpha=0.4**: Instead of MLE-estimated parameters as described in Benter (1994), a fixed weight is used, ignoring relative model vs market quality.

### Calibration Evidence

Full-horse calibration table (46,499 horses):

| p_place_pred bin | n | p_pred | actual | gap |
|---|---|---|---|---|
| [0.0-0.1) | 21,419 | 0.037 | 0.091 | -0.054 (underestimate) |
| [0.3-0.4) | 3,602 | 0.347 | 0.338 | +0.009 (roughly calibrated) |
| [0.5-0.6) | 2,142 | 0.549 | 0.456 | +0.094 (overestimate) |
| [0.7-0.8) | 1,143 | 0.746 | 0.597 | +0.149 (overestimate) |
| [0.9-1.0) | 164 | 0.924 | 0.750 | +0.174 (overestimate) |

Pattern: well-calibrated around p=0.3-0.4, systematically overestimates above that, underestimates below. Classic overconfidence pattern.

### Research Findings

- **Ali (1998)**: ALL ranking models overestimate place probability for high-probability horses across 15,000+ races. This is a structural bias, not just a calibration artifact.
- **Walsh & Joshi (2024, University of Bath)**: Calibration-optimized models produced +34.69% ROI vs -35.17% for accuracy-optimized models in sports betting. Calibration > accuracy.
- **Benter (1994)**: Alpha and beta should be estimated via MLE on out-of-sample data. The combined model uses `log(probability)`, not `logit(probability)`, with race-level softmax normalization.
- **Benter on Harville**: "This formula is significantly biased, and should not be used for betting purposes." Proposes corrected conditional probability model (gamma=0.81, delta=0.65).

## Design

### Section 1: Isotonic Calibration Layer for PlaceTwoStageModel

**File:** `src/models/two_stage_return_model.py` — `PlaceTwoStageModel`

Add isotonic calibration to the hit model output using OOF (out-of-fold) predictions.

**Training:**

1. After `PlaceTwoStageModel.train()` completes the 80/20 time-based split training, collect validation set (20%) predictions from the hit model.
2. Fit `IsotonicRegression(out_of_bounds='clip')` on validation predictions vs actual outcomes (`kakuteijyuni <= 3`).
3. Save as `self._place_calibrator`.
4. Minimum sample guard: skip calibration if validation set < 1,000 samples (overfit risk per sklearn recommendation).

```python
from sklearn.isotonic import IsotonicRegression

# In train(), after hit model validation predictions:
if len(val_predictions) >= 1000:
    self._place_calibrator = IsotonicRegression(out_of_bounds='clip')
    self._place_calibrator.fit(val_predictions, val_labels)
else:
    self._place_calibrator = None  # fallback to raw output
```

**Inference:**

```python
# In predict_ev():
raw_p = self.hit_model.predict(hit_features, ...)
if self._place_calibrator is not None:
    p_calibrated = self._place_calibrator.transform(raw_p)
else:
    p_calibrated = raw_p
df["p_place_pred"] = p_calibrated
```

**Why isotonic regression over Platt scaling:**
- Dataset size (~37K validation samples) is well above the 5,000+ recommendation for isotonic.
- The calibration table shows a non-sigmoid miscalibration pattern (underestimate at low p, overestimate at high p). Isotonic regression handles arbitrary monotonic distortions; Platt scaling assumes sigmoid shape.

### Section 2: Race-Sum Normalization

**File:** `src/models/two_stage_return_model.py` — `PlaceTwoStageModel.predict_ev()`

After isotonic calibration, normalize probabilities so `sum(p_place) ~ 3.0` per race (JRA pays top 3 places).

```python
# After isotonic calibration:
df["p_place_pred"] = self._place_calibrator.transform(raw_p)

# Race-sum normalization
race_sum = df.groupby("race_id")["p_place_pred"].transform("sum")
df["p_place_pred"] = df["p_place_pred"] * (3.0 / race_sum)

# Consistency constraint: p_place >= p_ability_win
mask = df["p_place_pred"] < df["p_ability_win"]
df.loc[mask, "p_place_pred"] = df.loc[mask, "p_ability_win"]
race_sum = df.groupby("race_id")["p_place_pred"].transform("sum")
df["p_place_pred"] = df["p_place_pred"] * (3.0 / race_sum)

# Final clip
df["p_place_pred"] = df["p_place_pred"].clip(0.01, 0.99)
```

This follows the same pattern already used in `PlaceAbilityModel` (lines 190-200).

**Ali (1998) structural bias note:** The remaining overestimation for high-probability horses (p > 0.5) will be corrected by the logistic regression in Section 3, whose intercept term absorbs systematic bias.

### Section 3: Benter Combination via Binary Logistic Regression (MLE)

**Files:** `src/backtest/race_predictor.py`, `src/pipelines/training_pipeline.py`

Replace the fixed `alpha=0.4` Benter combination with a data-driven logistic regression.

**Why binary logistic regression instead of Benter's multinomial logit:**
- Benter's formula `c_i = exp(alpha*log(f_i) + beta*log(pi_i)) / sum(...)` uses race-level softmax normalization. This is correct for WIN probabilities (one winner per race, sum = 1).
- PLACE is NOT a mutually exclusive outcome: multiple horses can place simultaneously (top 3). Sum of place probabilities ~ 3.0, not 1.0.
- Therefore, binary logistic regression per horse is the correct adaptation of Benter's MLE approach for place betting.

**Training (in `training_pipeline.py`):**

After PlaceTwoStageModel and EV correction training, fit a logistic regression on OOF data:

```python
from sklearn.linear_model import LogisticRegression

# Collect OOF data (validation set predictions)
p_model = p_place_norm.clip(1e-6, 1 - 1e-6)  # calibrated + race-normalized
p_market = (1.0 / df["fukuoddslow"]).clip(1e-6, 1 - 1e-6)
y = (df["kakuteijyuni"] <= 3).astype(int)

X = np.column_stack([
    np.log(p_model / (1 - p_model)),   # logit(p_model)
    np.log(p_market / (1 - p_market))  # logit(p_market)
])
benter_lr = LogisticRegression(fit_intercept=True, penalty=None)
benter_lr.fit(X, y)

# Extract parameters
alpha = benter_lr.coef_[0][0]   # model weight
beta = benter_lr.coef_[0][1]    # market weight
gamma = benter_lr.intercept_[0] # bias correction
```

**Inference (in `race_predictor.py`):**

```python
# Current (REMOVE):
# logit_combined = 0.4 * logit(p_place_pred) + 0.6 * logit(p_market)
# p_combined = sigmoid(logit_combined)

# New:
logit_m = np.log(p_place_norm.clip(1e-6, 1-1e-6) / (1 - p_place_norm.clip(1e-6, 1-1e-6)))
logit_mk = np.log(p_market.clip(1e-6, 1-1e-6) / (1 - p_market.clip(1e-6, 1-1e-6)))
logit_combined = alpha * logit_m + beta * logit_mk + gamma
p_combined = 1.0 / (1.0 + np.exp(-logit_combined))
edge = p_combined - p_market
```

**Parameter interpretation:**
- `alpha >> beta`: Model is more informative than market (strong model).
- `beta >> alpha`: Market is more informative (weak model, rely on market).
- `gamma < 0`: Systematic overestimation correction (expected for place models per Ali 1998).
- `gamma > 0`: Systematic underestimation correction.

## Implementation Scope

### Files Modified

| File | Change |
|---|---|
| `src/models/two_stage_return_model.py` | Add `_place_calibrator` (IsotonicRegression), race-sum normalization to `PlaceTwoStageModel` |
| `src/backtest/race_predictor.py` | Replace fixed alpha with logistic regression parameters |
| `src/pipelines/training_pipeline.py` | Add Benter logistic regression training step |
| `src/models/submodel_manager.py` | Add benter_lr storage in submodel |

### Files NOT Modified

- `PlaceAbilityModel` — already has isotonic + temperature + race-norm
- `EVCorrectionModel` — operates on p_place_pred post-calibration (benefits from better input)
- `StakeCalculator` — unchanged (Kelly formula is correct)
- `RegimeDetector` — unchanged

### Training Pipeline Order (Updated)

```
Existing steps 1-12 unchanged...
13. Place Two-Stage Model (with isotonic calibration + race-norm) ← MODIFIED
14. Place EV Correction
15. Benter Logistic Regression (new step) ← NEW
16. RobustConfidenceEstimator
```

### Serialization

The following must be saved/loaded with the submodel:
- `self._place_calibrator` (IsotonicRegression) — pickled with PlaceTwoStageModel
- `benter_lr` (LogisticRegression) — stored in SubModelManager, pickled separately

## Expected Impact

### Before (current)

| Metric | Value |
|---|---|
| ROI | 65.3% |
| Avg p_place_pred | 0.5118 |
| Actual hit rate | 0.215 |
| Overestimation | 2.38x |

### After (expected)

- Isotonic calibration should bring avg p_place_pred close to actual hit rate (~0.22 for full population)
- Race-sum normalization ensures within-race consistency
- Logistic regression naturally down-weights model when it's wrong (alpha < 0.4 expected)
- Intercept corrects structural bias (gamma < 0 expected per Ali 1998)
- Edge distribution will shift: fewer false-positive edges → fewer but more profitable bets

### Risk Assessment

| Risk | Mitigation |
|---|---|
| Isotonic overfitting on validation set | 37K+ samples well above 1K minimum; `out_of_bounds='clip'` guards edge cases |
| Logistic regression overfitting | `penalty=None` (2 features, no overfitting risk); OOF data |
| Calibration degrades on new data | Re-train with each backtest (already the design: run_backtest.py retrains every time) |
| Benter LR produces unexpected parameters | Log parameters; if alpha < 0 or beta < 0, warn and fall back to market-only |

## References

- Benter, W. (1994). "Computer Based Horse Race Handicapping and Wagering Systems: A Report"
- Ali, M.M. (1998). "Probability models on horse-race outcomes" — structural overestimation of place probability
- Walsh, C. & Joshi, A. (2024). Machine Learning with Applications — calibration > accuracy for sports betting ROI
- Zadrozny, B. & Elkan, C. (2002). "Transforming Classifier Scores into Accurate Multiclass Probability Estimates"
- sklearn IsotonicRegression documentation
