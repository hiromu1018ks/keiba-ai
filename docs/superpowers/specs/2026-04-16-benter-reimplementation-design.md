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

For place betting (binary per horse, not multinomial), we use logit-space blending:
```
logit(p_combined) = alpha * logit(p_fundamental) + beta * logit(p_market)
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
p_pred * odds - 1 = edge                 Benter: logit(p_c) = a*logit(p_fund) + b*logit(p_market)
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

Also need to update the payout lookup in `_settle_bet()` (lines 675-698):
- Currently checks `1 <= finish_pos <= 3` and returns `stake * settle_odds`
- New: look up actual payout by `(race_id, umaban)` in payout_map
- If payout exists for this horse (it placed), return `stake * actual_payout_odds`
- If no payout (horse didn't place), return 0

**Bet decision odds** (pre-race) remain unchanged — we still use `fukuoddslow` from 5-min-before
snapshot for bet selection. Only settlement changes.

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

    Benter (1994): logit(p_c) = alpha * logit(p_fund) + beta * logit(p_market)
    alpha and beta estimated via maximum likelihood on out-of-sample validation data.
    """

    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def combine(self, p_fund: pd.Series, p_market: pd.Series) -> pd.Series:
        """Combine fundamental and market probabilities in logit space."""
        logit_fund = np.log(p_fund / (1 - p_fund))
        logit_market = np.log(p_market / (1 - p_market))
        logit_combined = self.alpha * logit_fund + self.beta * logit_market
        return 1 / (1 + np.exp(-logit_combined))

    @classmethod
    def fit(cls, p_fund: pd.Series, p_market: pd.Series,
            y: pd.Series) -> "BenterCombination":
        """Estimate alpha, beta via maximum likelihood."""
        from scipy.optimize import minimize

        def neg_log_likelihood(params):
            alpha, beta = params
            logit_f = np.log(p_fund / (1 - p_fund))
            logit_m = np.log(p_market / (1 - p_market))
            logit_c = alpha * logit_f + beta * logit_m
            p_c = 1 / (1 + np.exp(-logit_c))
            p_c = np.clip(p_c, 1e-10, 1 - 1e-10)
            return -np.sum(y * np.log(p_c) + (1 - y) * np.log(1 - p_c))

        result = minimize(neg_log_likelihood, x0=[0.5, 0.5],
                         method='Nelder-Mead')
        return cls(alpha=result.x[0], beta=result.x[1])
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
- Save as JSON: `{"alpha": 0.35, "beta": 0.65}` in model directory
- Load in model_loader.py alongside other model artifacts

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

### Change 5: Calibration (Optional Post-Processing)

Given Ali (1998) finding that ALL models overestimate place probability for high-probability horses,
add optional race-sum normalization after Benter combination:

```python
# Normalize within race so sum of place probs ≈ min(3, field_size)
race_sum = df.groupby("race_id")["p_place_combined"].transform("sum")
target_sum = df.groupby("race_id")["field_size"].transform(
    lambda x: x.clip(upper=3)
)
df["p_place_calibrated"] = df["p_place_combined"] * (target_sum / race_sum)
```

This is applied AFTER Benter combination, as a final calibration step. It can be toggled via
config to compare with/without.

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

| Change | Expected ROI Improvement | Confidence |
|--------|-------------------------|------------|
| Settlement fix (payfukusyopay) | +18% (63.6% -> ~75%) | High — based on data |
| Remove double-counting | +5-15% (edge differentiation restored) | Medium |
| Benter combination | +5-10% (optimal model-market blend) | Medium |
| Race-sum normalization | +0-5% (calibration correction) | Low |
| **Total estimated** | **~85-105% ROI** | **Medium** |

Note: 100% ROI = break-even. A successful implementation should at minimum achieve ~85% (significant
improvement from current ~75% after settlement fix alone), with potential for profitability (>100%)
if the Benter combination captures genuine model edge over the market.

## Risks

1. **Fundamental model accuracy drops** without odds features. This is expected and intentional —
   the Benter layer compensates. But if p_fundamental is too poor, even optimal combination won't help.

2. **Overfitting alpha/beta.** Mitigate by estimating on held-out validation data and checking
   stability across years.

3. **Place probability normalization may distort.** The target sum of 3.0 is approximate (in reality,
   exactly 3 horses place, but their probabilities are correlated). Monitor calibration before/after.

4. **Data availability.** `payfukusyopay` may have missing values for some races (cancelled races,
   walkovers). Need fallback to `fukuoddslow` for these cases.

## Validation

1. Run multi-year backtest (2023-2025, train-window=4) with settlement fix only — establish true baseline
2. Run same backtest with all changes — measure improvement
3. Compare calibration tables (predicted vs actual hit rate by bin) before/after
4. Report alpha/beta values and their stability across validation folds
5. Compare AUC of: fundamental model, market, Benter combination
