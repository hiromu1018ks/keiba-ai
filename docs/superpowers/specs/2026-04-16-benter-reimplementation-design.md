# Benter Two-Stage Place Prediction + Settlement Fix

Date: 2026-04-16
Status: Draft

## Background

### Current Problems

Three root causes prevent profitability:

1. **Settlement uses pre-race odds** (`fukuoddslow`) instead of actual payouts (`payfukusyopay`).
   Actual payouts average 17.8% higher than pre-race odds, making reported ROI ~18% lower than reality.
   Current ROI 63.6% is actually ~75%.

2. **Double-counting of market odds.** The hit model uses `fukuoddslow` and `tanodds` as input features
   (commit 9a03a1d), AND the edge formula multiplies `p_place_pred * fukuoddslow`. When the model
   learns to approximate `1/fukuoddslow` (market probability), `p * odds -> 1.0` for all horses,
   eliminating edge differentiation.

3. **p_place_pred overestimation.** Model predicts 0.512 avg for selected bet horses; actual hit rate
   is 0.215 (2.38x overestimation). Ali (1998) showed this is a structural problem for ALL ranking
   models in place betting — they systematically overestimate high-probability horses' place chances.

### Why the Previous Benter Implementation Failed

Commit 9a03a1d removed Benter LR because of "place probability overestimation." The root cause was
not the Benter combination itself but double-counting: the underlying model already had `fukuoddslow`
as a feature, so the Benter layer was combining a market-informed prediction with the market again.

### Academic Foundation

Benter (1994) demonstrated that combining a fundamental handicapping model with the public's implied
probability via a second-stage logit model produces R²=0.1327, exceeding both the fundamental model
(R²=0.1016) and the public estimate alone (R²=0.1237). The combination formula uses log(probability)
as independent variables:

```
c_i = exp(alpha * f_i + beta * x_i) / sum_j(exp(alpha * f_j + beta * x_j))
```

where `f_i = log(p_fundamental_i)` and `x_i = log(p_market_i)`.

For place betting (binary per horse, not multinomial), we adapt with logit-space blending.
A bias term is included because the multinomial version implicitly has one via its normalization
constant. Without it, alpha+beta != 1 causes systematic under/over-prediction:
```
logit(p_combined) = alpha * logit(p_fundamental) + beta * logit(p_market) + gamma
```

Machine Learning for Sports Betting (2024) showed calibration-based model selection yields
ROI +34.69% vs accuracy-based -35.17%. Kelly betting only works with well-calibrated models.

## Design

### Overview

```
[Current]                                [New]
odds features -> LightGBM -> p_pred      fundamental features -> LightGBM -> p_fund
      |                                       |
      v                                       v
p_pred * odds - 1 = edge                 Benter: logit(p_c) = a*logit(p_fund) + b*logit(p_market) + c
      |                                       |
      v                                       v
settle with fukuoddslow (18% low)        p_combined * odds - 1 = edge
                                            |
                                            v
                                      settle with payfukusyopay (actual payout)
```

### Change 1: Settlement Fix

**File:** `src/backtest/engine.py`

Replace `fukuoddslow`-based settlement with actual payouts from `data/raw/payouts.parquet`.

Current flow (lines 189-196):
```python
final_odds_map: dict[tuple[str, int], float] = {}
if not final_odds_df.empty:
    for _, r in final_odds_df.iterrows():
        key = (str(r["race_id"]), int(r["umaban"]))
        if pd.notna(r.get("fukuoddslow")):
            final_odds_map[key] = float(r["fukuoddslow"])
```

New flow:
1. Load `payouts.parquet` via `load_payouts(store, start, end)` from `src/db/readers.py`
2. Build `payout_map: dict[tuple[str, int], float]` from `payfukusyopay1-5` / `payfukusyoumaban1-5`
3. Convert payout to odds multiplier: `pay_value / 100.0`
4. Use `payout_map` in `_settle_bet()` instead of `final_odds_map`
5. Keep `final_odds_map` (from odds snapshots) as fallback for races without payout data

**Implementation details for `_settle_bet()`:**
- Pass `payout_map` as an instance variable on BacktestEngine (populated during `run()`)
- `_settle_bet()` looks up `(race_id, umaban)` in payout_map
- If found (horse placed), return `stake * actual_payout_odds`
- If not found (horse didn't place), return 0
- Fallback: if payout_map has no entry for the race at all (data gap), use `final_odds_map`

**Edge cases:**
- **Field size < 8:** JRA pays only 2 place positions (not 3). `payfukusyopay3` is NaN for these races.
  The payout_map handles this correctly — only placed horses have entries.
- **`syussotosu` type:** The payouts parquet stores this as string; compare with `field_size` column
  from entries/races data (already numeric) rather than from payouts.

**Bet decision odds** (pre-race) remain unchanged — we still use `fukuoddslow` from 5-min-before
snapshot for bet selection. Only settlement changes.

**Note on odds vs payout distinction:** `fukuoddslow` from `odds_tanpuku` is JRA's confirmed place
odds (includes overround of ~20-25%), while `payfukusyopay` is the actual amount paid. The ~18%
difference is the overround embedded in the odds snapshot that doesn't apply to actual payouts.

### Change 2: Fundamental Model (No Market Odds)

**File:** `src/models/two_stage_return_model.py`

Remove from `HIT_FEATURE_COLS` (lines 184-212):
- `fukuoddslow` — direct place odds
- `tanodds` — direct win odds

Keep (19 features remain):
- `p_ability_win`, `p_ability_place` — ability model outputs
- `signed_log_error_win`, `abs_log_error_win` — market model residuals (indirect market info)
- `odds_drop_rate_60_10`, `odds_drop_rate_30_10` — odds dynamics (indirect market info)
- `odds_velocity`, `odds_volatility` — odds dynamics
- `popularity_change_30_10` — popularity trend
- `market_entropy`, `popularity_rank`, `overround` — market structure
- `surface`, `distance_bin`, `track_condition_code`, `grade_code` — race conditions
- `field_size`, `odds_skewness` — race conditions

**Rationale:** Direct odds values (`fukuoddslow`, `tanodds`) tell the model "what the market thinks."
The Benter combination will handle this information in a principled way. Keeping indirect market
signals (dynamics, residuals, structure) provides the model with market context without the raw
probability estimate that causes double-counting.

### Change 3: Benter Combination Layer

**New class:** `BenterCombination` in `src/models/benter_combination.py`

```python
class BenterCombination:
    """Second-stage logit combination of fundamental model + market probability.

    Benter (1994) adaptation: logit(p_c) = a*logit(p_fund) + b*logit(p_market) + c
    Includes bias term c (the multinomial version has one implicitly via normalization).
    alpha, beta, gamma estimated via maximum likelihood on out-of-sample validation data.
    """

    def __init__(self, alpha: float, beta: float, gamma: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # bias/intercept term

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def combine(self, p_fund: np.ndarray, p_market: np.ndarray) -> np.ndarray:
        """Combine fundamental and market probabilities in logit space."""
        logit_combined = (self.alpha * self._logit(p_fund)
                          + self.beta * self._logit(p_market)
                          + self.gamma)
        return self._sigmoid(logit_combined)

    @classmethod
    def fit(cls, p_fund: np.ndarray, p_market: np.ndarray,
            y: np.ndarray) -> "BenterCombination":
        """Estimate alpha, beta, gamma via maximum likelihood with constraints."""
        from scipy.optimize import minimize

        logit_f = cls._logit(p_fund)
        logit_m = cls._logit(p_market)

        def neg_log_likelihood(params):
            alpha, beta, gamma = params
            logit_c = alpha * logit_f + beta * logit_m + gamma
            p_c = cls._sigmoid(logit_c)
            return -np.sum(y * np.log(p_c + 1e-10) + (1 - y) * np.log(1 - p_c + 1e-10))

        # L-BFGS-B with bounds: alpha, beta >= 0 (both sources should contribute positively)
        result = minimize(neg_log_likelihood, x0=[0.5, 0.5, 0.0],
                         method='L-BFGS-B',
                         bounds=[(0.01, 5.0), (0.01, 5.0), (-5.0, 5.0)])
        return cls(alpha=result.x[0], beta=result.x[1], gamma=result.x[2])
```

**Integration in RacePredictor.predict():**

```python
# After getting p_place_pred from fundamental model
p_market = 1.0 / df["fukuoddslow"]
p_market = p_market.clip(0.01, 0.99)

# Benter combination
df["p_place_combined"] = benter.combine(df["p_place_pred"], p_market)

# Edge calculation
df["edge_place"] = df["p_place_combined"] * df["fukuoddslow"] - 1.0
```

**Training flow:**
1. Train fundamental model (no odds features) on training data
2. Generate out-of-sample predictions on validation data
3. Fit BenterCombination via MLE on validation data
4. Save alpha, beta alongside model artifacts

**Serialization:**
- Save as JSON: `{"alpha": 0.35, "beta": 0.65, "gamma": -0.05}` in model directory
- Replace existing `benter_lr_{surface}.joblib` with the new JSON format
- Update `SubmodelSet.benter_lr` field to `benter_combo: BenterCombination | None`
- Update `model_loader.py` (lines 472-479) to load from JSON instead of joblib

### Change 4: Edge Calculation Update

**File:** `src/backtest/race_predictor.py`

Current (lines 120-129):
```python
df["ev_place_direct"] = df["p_place_pred"] * df["fukuoddslow"]
df["p_place_combined"] = df["p_place_pred"]
df["edge_place"] = df["ev_place_direct"] - 1.0
```

New:
```python
# p_place_pred is now from fundamental model (no odds features)
p_market = 1.0 / df["fukuoddslow"].clip(lower=0.1)
df["p_market"] = p_market
df["p_place_combined"] = benter.combine(df["p_place_pred"], p_market)
df["edge_place"] = df["p_place_combined"] * df["fukuoddslow"] - 1.0
```

### Change 5: Isotonic Calibration (Optional Post-Processing)

Given Ali (1998) finding that ALL models overestimate place probability for high-probability horses,
apply isotonic regression calibration after Benter combination.

**Why NOT race-sum normalization:** Place probabilities are NOT mutually exclusive — multiple horses
can place simultaneously. Market-implied place probabilities (`1/fukuoddslow`) sum to ~4.5 on average
(not 3.0), due to overround. Forcing sum=3.0 would systematically underpredict.

Instead, use isotonic regression (non-parametric monotonic mapping) fitted on validation data:

```python
from sklearn.isotonic import IsotonicRegression

# Fit on validation data
iso_reg = IsotonicRegression(out_of_bounds="clip")
iso_reg.fit(p_combined_validation, actual_place_validation)

# Apply at inference
df["p_place_calibrated"] = iso_reg.transform(df["p_place_combined"])
```

This can be toggled via config to compare with/without.

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/models/two_stage_return_model.py` | Remove `fukuoddslow`, `tanodds` from HIT_FEATURE_COLS | ~2 |
| `src/models/benter_combination.py` | New file: BenterCombination class | ~80 |
| `src/backtest/race_predictor.py` | Use Benter combination for edge | ~15 |
| `src/backtest/engine.py` | Settlement via payfukusyopay | ~40 |
| `src/db/readers.py` | Add `load_payouts_for_settlement()` helper | ~20 |
| `src/db/model_loader.py` | Load BenterCombination from model dir | ~15 |
| `src/domain/models.py` | Add benter_alpha/beta to SubmodelSet | ~5 |

## Expected Impact

**True baseline** (after settlement fix): ~75% ROI (this is a measurement correction, not an improvement)

| Change | Expected Effect | Confidence |
|--------|----------------|------------|
| Settlement fix (payfukusyopay) | Reveals true ROI ~75% | High — based on data |
| Remove double-counting | +5-15% ROI on top of true baseline | Medium — plausible but not yet empirically proven |
| Benter combination | +5-10% (optimal model-market blend) | Medium — academically validated framework |
| Isotonic calibration | +0-5% (fixes remaining overestimation) | Low — depends on calibration gap |
| **Total estimated** | **~85-105% ROI** | **Medium** |

Note: 100% ROI = break-even. The settlement fix is a measurement correction. All other changes
are genuine improvements to the prediction and betting system.

## Risks

1. **Fundamental model accuracy drops** without odds features. This is expected and intentional —
   the Benter layer compensates. But if p_fundamental is too poor, even optimal combination won't help.

2. **Overfitting alpha/beta.** Mitigate by estimating on held-out validation data and checking
   stability across years.

3. **Isotonic calibration may overfit** on small validation sets. Mitigate by using large validation
   window (full year) and monitoring calibration stability.

4. **Data availability.** `payfukusyopay` may have missing values for some races (cancelled races,
   walkovers). Need fallback to `fukuoddslow` for these cases.

## Validation

1. Run multi-year backtest (2023-2025, train-window=4) with settlement fix only — establish true baseline
2. Run same backtest with all changes — measure improvement
3. **Calibration protocol:** Bin `p_place_combined` into deciles, plot mean predicted vs mean actual
   hit rate (reliability diagram). Compute Expected Calibration Error (ECE). Include 95% CI.
4. Report alpha, beta, gamma values and their stability across validation folds
5. Compare AUC of: fundamental model, market, Benter combination
6. Verify that fitted alpha + beta > 0 (both sources contribute) and gamma is small (< |0.5|)
