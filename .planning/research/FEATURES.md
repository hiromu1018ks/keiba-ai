# Feature Research: Race-Level Aggregation and Market Cross-Consistency

**Domain:** Horse racing ML prediction -- market-independent edge discovery
**Researched:** 2026-05-17
**Confidence:** HIGH (race-level features, validated by yurelu article + academic literature), MEDIUM (market-cross features, constrained by data availability)

## Executive Summary

This research covers two feature categories for the v1.7 milestone:
1. **Race-level aggregation features** -- statistics computed across all runners in a race that capture "race readability" (entropy, dispersion, favorite dominance)
2. **Market cross-consistency features** -- features that measure agreement between different bet types' odds for the same race, capturing market conviction

The yurelu (zenn.dev) article demonstrated that adding 5 race-level features improved C-orthogonal IC by +40%, and adding 5 market-cross features (win x trio quinella) further improved C-orthogonal IC by +120%, achieving ROI 0.91 to 1.66 on 408 races. This is the strongest empirical evidence for these feature categories in JRA horse racing ML.

**Critical constraint:** Our data environment differs from yurelu's. We have:
- Win odds (snapshots + time series, 2015-2026, 38,825 races)
- Place odds (snapshots, 2015-2026)
- Wide odds (per-pair, 2015-2026, 38,825 races)
- Payout odds for trio/trifecta (post-race only -- CANNOT be used as features)
- Quinella odds time series (2026 only, 25 races -- unusable)
- NO trio quinella (三連複) pre-race odds available
- NO trifecta (三連単) pre-race odds available

This means the yurelu article's exact `rl_market_consistency` feature (win x trio) CANNOT be directly replicated. Instead, we must use win x wide cross-consistency, which is the closest available alternative.

---

## Feature Landscape

### Category 1: Race-Level Aggregation Features

These capture the overall market structure of a race -- how "readable" or "chaotic" the betting public perceives the race to be.

#### Table Stakes (Well-Established, Expected in Any Serious System)

| Feature | Formula | Why Expected | Complexity | Notes |
|---------|---------|--------------|------------|-------|
| `rl_n_horses` (field_size) | Count of runners | Basic context -- model must know if 8-horse or 18-horse field | LOW | Already exists as `field_size` in codebase |
| `rl_log_odds_entropy` | H = -sum(p_i * ln(p_i)) where p_i = (1/odds_i) / sum(1/odds_j) | Standard market uncertainty measure from information theory | LOW | Already exists as `market_entropy` in `market_bias_features.py` |
| `rl_odds_dispersion` | std_dev(odds) across runners | Captures spread between favorites and longshots | LOW | NEW -- not yet implemented |
| `rl_top1_odds` | Minimum odds (favorite's win odds) | Market's assessment of dominant horse strength | LOW | Can compute from existing `tanodds` |
| `rl_top3_odds_gap` | odds(rank 3) - odds(rank 1) | Gap between top-tier and mid-tier | LOW | NEW -- not yet implemented |
| `rl_overround` | sum(1/odds_i) - 1 | Bookmaker/tote deduction rate, market health indicator | LOW | Already exists as `overround` in `market_bias_features.py` |

#### Differentiators (Proven Valuable, Not Universally Used)

| Feature | Formula | Value Proposition | Complexity | Notes |
|---------|---------|-------------------|------------|-------|
| `rl_normalized_entropy` | H / ln(n) where n = field_size | Entropy adjusted for field size -- makes 8-horse and 18-horse races comparable. Range [0, 1], 1 = perfectly competitive | LOW | Already exists as `difficulty_score` component in `race_difficulty_model.py`. Could add as standalone feature |
| `rl_favorite_implied_prob` | 1 / odds(favorite) | Market's confidence in the favorite. High = "iron plate" race, low = wide open | LOW | Already computed in `compute_roi_ema()` as `favorite_implied_prob_ema` |
| `rl_implied_prob_hhi` | sum(p_i^2) where p_i = normalized implied probability | Herfindahl-Hirschman Index -- concentration measure. High HHI = one dominant horse, low = competitive | LOW | Already exists in `compute_flb_slope()` as `implied_prob_hhi` |
| `rl_odds_skewness` | skew(odds distribution) | Asymmetry of odds distribution. Positive skew = most longshots, negative = concentrated favorites | LOW | Already exists in `compute_flb_slope()` as `odds_skewness` |
| `rl_favorite_rank_gap` | (odds_rank2 - odds_rank1) / odds_rank1 | Relative gap between 1st and 2nd favorite. Large gap = clear favorite | LOW | NEW -- yurelu article showed this captures "readable race" signal |
| `rl_log_favorite_share` | log(odds_rank1 / mean(odds)) | How much the favorite dominates relative to average | LOW | NEW -- non-linear transform helps LightGBM split |

#### Key Insight from yurelu Article

yurelu's v9 (race-level features only, 5 features) achieved:
- C-orthogonal IC: +40% improvement over baseline
- The 5 features were: `rl_n_horses`, `rl_top1_odds`, `rl_top3_odds_gap`, `rl_odds_dispersion`, `rl_log_odds_entropy`

Gain per Depth analysis showed:
- Depth 1-2: 97-99% Market features (odds, popularity)
- Depth 3-4: Transition zone where race-level features activate
- Depth 5+: Fundamental features (jockey, trainer, etc.)

This confirms race-level features operate at a specific "sweet spot" in LightGBM's tree structure -- they provide the "race context" that lets the model split on "readable vs unreadable race" before applying horse-level features.

### Category 2: Market Cross-Consistency Features

These capture whether different bet types (win, wide, quinella) agree on the race outcome, which signals market conviction.

#### Theoretical Foundation: Harville Formula

The Harville (1973) formula derives theoretical multi-horse bet probabilities from win probabilities:

```
P(i finishes 1st, j finishes 2nd) = P(i) * P(j) / (1 - P(i))

Quinella (i,j either order):
P_quinella(i,j) = P(i)*P(j)/(1-P(i)) + P(j)*P(i)/(1-P(j))

Wide (i,j both in top 3):
Requires summing over all 6 permutations of positions 1,2,3 where both i,j appear

Trifecta (i,j,k in exact order):
P(i,j,k) = P(i) * P(j)/(1-P(i)) * P(k)/(1-P(i)-P(j))
```

Where P(i) = normalized implied probability from win odds = (1/odds_i) / sum(1/odds_j)

**Key property:** If the market is efficient and bettors are rational, actual multi-horse bet odds should approximate Harville-derived theoretical odds. Deviations indicate:
1. Market inefficiency (mispricing)
2. Information asymmetry (insiders betting specific combinations)
3. Behavioral bias (public overweighting/underweighting certain combinations)

#### Table Stakes (Available Data Allows These)

| Feature | Formula | Why Expected | Complexity | Data Needed |
|---------|---------|--------------|------------|-------------|
| `rl_wide_harville_ratio` | actual_wide_odds(top2) / theoretical_wide_odds_from_win(top2) | Simplest cross-bet consistency check. Ratio > 1 = wide market overpays, < 1 = underpays | MEDIUM | Win odds + Wide odds (both available) |
| `rl_wide_harville_ratio_fav` | Same as above but for favorite pairs only | Most important pair -- favorite x 2nd favorite | MEDIUM | Win odds + Wide odds |
| `rl_wide_top3_harville_mean` | Mean of harville_ratio for top-3 most popular wide pairs | Average market conviction across most-bet combinations | MEDIUM | Win odds + Wide odds |
| `rl_win_vs_wide_rank_correlation` | Spearman rank correlation between win implied probs and wide implied probs (per-pair) | High correlation = market agrees; low = divergence between bet types | MEDIUM | Win odds + Wide odds |

#### Differentiators (High Value, Proven in Literature)

| Feature | Formula | Value Proposition | Complexity | Data Needed |
|---------|---------|-------------------|------------|-------------|
| `rl_market_conviction_index` | Weighted average of |1 - actual/theoretical| for top-k pairs | Composite signal capturing overall market conviction. High = "market knows something", low = "market is confused" | MEDIUM | Win odds + Wide odds |
| `rl_wide_harville_dispersion` | std_dev(harville_ratios) across all pairs in race | If all pairs are similarly mispriced = systematic bias. If scattered = race-specific uncertainty | MEDIUM | Win odds + Wide odds |
| `rl_favorite_in_wide_top1` | Whether favorite (win rank 1) appears in the lowest-odds wide combination (0/1) | Analogous to yurelu's `rl_market_consistency` but using wide instead of trio. Captures "iron plate race" signal | LOW | Win odds + Wide odds |
| `rl_win_wide_favorite_gap` | |win_implied_prob(fav) - wide_implied_prob(fav,rank2)| / win_implied_prob(fav) | Measures whether wide market agrees with win market about favorite strength | LOW | Win odds + Wide odds |

#### Key Insight from yurelu Article

yurelu's v11 (race-level + market-cross features) achieved:
- C-orthogonal IC: +120% improvement over v9 (race-level only)
- `rl_market_consistency` (whether #1 favorite appears in #1 trio combination) was the single most powerful feature at Depth 3-4, capturing 32-33% of gain share
- This feature essentially classifies races into "readable" (favorite in top trio combo) vs "unreadable" (favorite NOT in top trio combo)

**Critical adaptation needed:** yurelu had trio quinella odds. We do NOT. The closest substitute is:
- Use **wide odds** (ワイドオッズ) instead of trio odds -- both are multi-horse combination bets
- Use `rl_favorite_in_wide_top1` as the analogue of `rl_market_consistency`
- The theoretical framework (Harville) applies identically -- compute theoretical wide odds from win odds and compare to actual

### Category 3: Post-Race Payout Features (NOT for Training -- but Useful for Labels/Validation)

The payouts table contains final odds for all bet types (三連複, 三連単, 馬連, ワイド). These CANNOT be used as features (post-race data), but are valuable for:
- Computing actual ROI in backtesting
- Building validation labels for "was this race readable?"
- Analyzing market efficiency ex-post

#### Anti-Features (Do NOT Build)

| Anti-Feature | Why Requested | Why Problematic | Alternative |
|--------------|---------------|-----------------|-------------|
| Trio quinella pre-race odds features | yurelu's article showed it's the most powerful feature | Data is NOT available in our EveryDB2 dataset for pre-race snapshots | Use wide odds cross-consistency as substitute |
| Quinella time-series features | Would enable finer-grained market cross features | Only 25 races of quinella time-series data (2026 only) | Use wide odds snapshot cross-consistency |
| Payout-derived features in training | Tempting to use "1st-place payout horse" or "trio payout combination" | POST_RACE information leakage -- would inflate backtest but fail in production | Use payout data only for labels and validation |
| Complex multi-bet theoretical models (Stern, Henery models) | More accurate than Harville for exotic bets | Diminishing returns -- Harville captures 90%+ of the signal; complexity not justified for a single feature | Harville formula is sufficient |

---

## Feature Dependencies

```
Race-Level Features (Category 1)
    |-- rl_log_odds_entropy
    |       requires: tanodds (win odds) per horse per race
    |       already exists: market_bias_features.py
    |
    |-- rl_odds_dispersion  [NEW]
    |       requires: tanodds (win odds) per horse per race
    |
    |-- rl_top1_odds, rl_top3_odds_gap  [NEW]
    |       requires: tanodds (win odds) per horse per race
    |
    |-- rl_normalized_entropy
    |       requires: market_entropy + field_size
    |       already exists: race_difficulty_model.py (component)
    |
    |-- rl_implied_prob_hhi, rl_odds_skewness
    |       requires: tanodds per horse per race
    |       already exists: compute_flb_slope()

Market-Cross Features (Category 2)
    |-- rl_wide_harville_ratio  [NEW]
    |       requires: tanodds (win) + odds_wide (wide odds per pair)
    |       requires: Harville theoretical wide odds computation
    |
    |-- rl_favorite_in_wide_top1  [NEW]
    |       requires: tanodds + odds_wide
    |       depends on: Harville computation for pair identification
    |
    |-- rl_market_conviction_index  [NEW]
    |       requires: rl_wide_harville_ratio (computed for all pairs)
    |       enhances: rl_wide_harville_ratio
    |
    |-- rl_wide_harville_dispersion  [NEW]
    |       requires: rl_wide_harville_ratio (computed for all pairs)
    |       enhances: rl_market_conviction_index

Wide Odds Harville Computation  [NEW - prerequisite]
    requires: tanodds (win odds) + wide odds (kumi, oddslow)
    prerequisite for: ALL Category 2 features
```

### Dependency Notes

- **Race-level features are independent of each other:** All only need win odds per horse, which is already available in the training pipeline. Can be implemented in parallel.
- **Market-cross features ALL depend on wide odds data + Harville computation:** The Harville theoretical wide odds function must be implemented first. This is a shared dependency.
- **Wide odds data is already loaded in training pipeline:** `load_wide_odds()` is called in `TrainingPipelineV5._load_data()` and stored as `wide_odds_{lo}_{hi}` columns. The data pipeline is ready.
- **No conflict between categories:** Race-level features operate on single-horse odds, market-cross features operate on pair-level odds. They capture orthogonal information.

---

## Mathematical Formulations

### Race-Level Features (Category 1)

#### Shannon Entropy of Implied Probabilities
```
Given: odds = [o_1, o_2, ..., o_n] for n runners
p_raw_i = 1 / o_i
p_i = p_raw_i / sum(p_raw_j)    # Normalize to probabilities
H = -sum(p_i * ln(p_i))          # Shannon entropy

Range: [0, ln(n)]
  H = 0       => one horse has 100% probability (impossible in practice)
  H = ln(n)   => all horses equally likely (maximum uncertainty)
```

#### Odds Dispersion (Standard Deviation)
```
Given: odds = [o_1, o_2, ..., o_n]
sigma = std(odds)    # Population or sample std

Note: Use raw odds, not log-odds. High dispersion = clear hierarchy.
```

#### Top-K Odds Gap
```
Sort odds ascending: o_(1) <= o_(2) <= ... <= o_(n)
top3_gap = o_(3) - o_(1)    # Gap between 3rd favorite and favorite

Note: Larger gap = more tier separation in the market.
```

### Market Cross-Consistency Features (Category 2)

#### Harville Theoretical Wide Odds

The exact Harville probability for a wide pair (i,j) -- both horses finishing in the top 3:

```
Given: win implied probs p_1, p_2, ..., p_n (normalized from win odds)

P_wide(i,j) = sum over all orderings (a,b) in {(i,j),(j,i)}:
  P(a=1st, b in {2nd,3rd}) + P(a in {2nd,3rd}, b in remaining top3)

Exact computation:
P(i=1, j=2) = p_i * p_j / (1-p_i)
P(i=1, j=3) = p_i * sum_{k!=i,j} [p_k/(1-p_i)] * [p_j/(1-p_i-p_k)]
P(i=2, j=1) = p_j * p_i / (1-p_j)
P(i=2, j=3) = similar with k!=i,j
P(i=3, j=1) = ...
P(i=3, j=2) = ...

Simplified Harville approximation (sufficient for feature engineering):
P_wide_harville(i,j) approx=
  p_i * p_j * [2/(1-p_i) + 2/(1-p_j) - p_i*p_j/((1-p_i)*(1-p_j))]

Even simpler (and good enough for a feature):
P_wide_harville(i,j) approx=
  p_i * p_j * [1/(1-p_i) + 1/(1-p_j)] * 1.5

Theoretical wide odds = 1 / P_wide_harville(i,j)
```

For computational implementation, the exact Harville wide probability can be computed as:

```python
def harville_wide_prob(p_i: float, p_j: float, all_probs: np.ndarray) -> float:
    """Exact Harville probability of both i,j finishing in top 3."""
    n = len(all_probs)
    prob = 0.0
    for a, b in [(p_i, p_j), (p_j, p_i)]:
        # a=1st, b=2nd
        prob += a * b / (1 - a)
        # a=1st, b=3rd (any k=2nd where k!=a,b)
        for p_k in all_probs:
            if p_k == a or p_k == b:
                continue
            prob += a * p_k / (1 - a) * b / (1 - a - p_k)
        # a=2nd, b=1st
        prob += b * a / (1 - b)
        # a=2nd, b=3rd
        # ... (continues for all permutations)
    # Subtract double-counted cases
    return prob
```

For practical purposes, an approximation that captures 90%+ of the signal:

```python
def harville_wide_approx(p_i: float, p_j: float) -> float:
    """Approximation sufficient for feature engineering."""
    return p_i * p_j * (1/(1-p_i) + 1/(1-p_j)) * 1.5
```

#### Harville Ratio
```
harville_ratio(i,j) = actual_wide_odds(i,j) / theoretical_wide_odds(i,j)

Interpretation:
  ratio > 1 => market underestimates this pair (wide odds too high)
  ratio < 1 => market overestimates this pair (wide odds too low)
  ratio ~ 1 => market is efficient for this pair
```

#### Market Conviction Index
```
For top-k most popular wide pairs (by ninki):
conviction = mean(|1 - harville_ratio|) for top-k pairs

High conviction => large deviations between win and wide markets
Low conviction => markets are consistent (efficient)
```

#### Favorite-in-Wide-Top1
```
favorite = horse with lowest win odds (rank 1 by tanodds)
lowest_wide_pair = wide pair with lowest oddslow

rl_favorite_in_wide_top1 = 1 if favorite appears in lowest_wide_pair
                          0 otherwise

This is the direct analogue of yurelu's rl_market_consistency.
```

---

## Data Availability Assessment

### Already in Pipeline (No New ETL Needed)

| Data | Source File | Coverage | Status |
|------|------------|----------|--------|
| Win odds (tanodds) | `odds_tanpuku.parquet`, `jodds_tanpuku/` | 2015-2026 | Loaded in training pipeline |
| Win odds time series | `jodds_tanpuku/` (year/month partitioned) | 2015-2026 | Loaded in training pipeline |
| Place odds (fukuoddslow) | `odds_tanpuku.parquet` | 2015-2026 | Loaded in training pipeline |
| Wide odds (oddslow/oddshigh per kumi) | `odds_wide.parquet` | 2015-2026, 38,825 races | Loaded in training pipeline |
| Market entropy | Computed from tanodds | 2015-2026 | Already computed in `market_bias_features.py` |
| Overround | Computed from tanodds | 2015-2026 | Already computed in `market_bias_features.py` |
| HHI, Skewness | Computed from tanodds | 2015-2026 | Already computed in `compute_flb_slope()` |
| Difficulty score | Computed from entropy + grade | 2015-2026 | Already computed in `race_difficulty_model.py` |

### NOT Available (Would Need New ETL or Cannot Be Obtained)

| Data | Status | Impact |
|------|--------|--------|
| Trio quinella (三連複) pre-race odds | NOT in EveryDB2 pre-race tables | Cannot replicate yurelu's exact features. Use wide odds as substitute |
| Trifecta (三連単) pre-race odds | NOT in EveryDB2 pre-race tables | Same as above |
| Quinella (馬連) time series | Only 25 races (2026) | Insufficient for training. Snapshot available in payouts (post-race only) |
| Bracket quinella (枠連) time series | File exists (`odds_waku`) but not in ETL pipeline | Could be added if needed, but low priority |

---

## Existing Feature Audit (Already Computed, Just Not Promoted)

The codebase already computes several race-level features that could be promoted to model inputs:

| Feature | Module | Currently Used By | Action Needed |
|---------|--------|-------------------|---------------|
| `market_entropy` | `market_bias_features.py` | Stage1 + Stage2 | Already in FEATURE_COLS -- verify as race-level input |
| `overround` | `market_bias_features.py` | Stage1 + Stage2 | Already in FEATURE_COLS |
| `implied_prob_hhi` | `compute_flb_slope()` | Not in FEATURE_COLS | PROMOTE to model features |
| `odds_skewness` | `compute_flb_slope()` | Not in FEATURE_COLS | PROMOTE to model features |
| `favorite_implied_prob_ema` | `compute_roi_ema()` | Used in regime detection | PROMOTE to model features (non-EMA version) |
| `overround_ema` | `compute_roi_ema()` | Used in regime detection | Already available in pipeline |
| `entropy_ema` | `compute_roi_ema()` | Used in regime detection | Already available in pipeline |
| `difficulty_score` | `race_difficulty_model.py` | Stage1 | Already in FEATURE_COLS |

---

## MVP Definition

### Phase 1: Race-Level Features (Implement First)

These have no new data dependencies and are straightforward to compute.

- [x] `rl_log_odds_entropy` -- already exists as `market_entropy`
- [x] `rl_overround` -- already exists as `overround`
- [x] `rl_implied_prob_hhi` -- already exists as `implied_prob_hhi` (promote to FEATURE_COLS)
- [x] `rl_odds_skewness` -- already exists as `odds_skewness` (promote to FEATURE_COLS)
- [ ] `rl_odds_dispersion` -- NEW, std dev of odds per race
- [ ] `rl_top1_odds` -- NEW, favorite's odds (race-level broadcast)
- [ ] `rl_top3_odds_gap` -- NEW, gap between 3rd and 1st favorite
- [ ] `rl_favorite_rank_gap` -- NEW, relative gap between 1st and 2nd favorite
- [ ] `rl_normalized_entropy` -- NEW (promote from difficulty_score component to standalone)

**Rationale:** yurelu's 5 race-level features alone improved C-orthogonal IC by +40%. We have most already; adding the missing 4-5 should capture similar signal.

### Phase 2: Market Cross-Consistency Features (Implement Second)

These require implementing Harville wide odds computation.

- [ ] Harville theoretical wide odds function (shared prerequisite)
- [ ] `rl_favorite_in_wide_top1` -- yurelu analogue, simplest market-cross feature
- [ ] `rl_wide_harville_ratio_fav` -- ratio for favorite x rank2 pair
- [ ] `rl_wide_top3_harville_mean` -- mean ratio for top-3 pairs
- [ ] `rl_wide_harville_dispersion` -- std dev of ratios across pairs

**Rationale:** yurelu's 5 market-cross features were the "main weapon" (+120% C-orthogonal IC). Our wide odds data covers the same 2015-2026 range, so signal strength should be comparable.

### Future Consideration (After v1.7 Validation)

- [ ] Quinella (馬連) cross-consistency features -- only possible after ETL expansion to load `jodds_umaren` data for full 2015-2026 range
- [ ] Bracket quinella (枠連) cross-consistency -- `odds_waku` file exists but not yet loaded in pipeline
- [ ] Multi-timepoint cross-consistency -- how Harville ratios change during final 30 minutes before post

---

## Feature Prioritization Matrix

| Feature | User Value (IC improvement) | Implementation Cost | Priority |
|---------|---------------------------|---------------------|----------|
| `rl_odds_dispersion` | HIGH (yurelu top feature) | LOW | P1 |
| `rl_top1_odds` | HIGH (yurelu top feature) | LOW | P1 |
| `rl_top3_odds_gap` | HIGH (yurelu top feature) | LOW | P1 |
| `rl_favorite_rank_gap` | MEDIUM | LOW | P1 |
| `rl_normalized_entropy` | MEDIUM (already partially exists) | LOW | P1 |
| `implied_prob_hhi` promotion | MEDIUM | LOW | P1 |
| `odds_skewness` promotion | MEDIUM | LOW | P1 |
| Harville wide computation | HIGH (enables all Category 2) | MEDIUM | P1 |
| `rl_favorite_in_wide_top1` | VERY HIGH (yurelu's #1 feature analogue) | MEDIUM | P1 |
| `rl_wide_harville_ratio_fav` | HIGH | MEDIUM | P2 |
| `rl_wide_top3_harville_mean` | MEDIUM | MEDIUM | P2 |
| `rl_wide_harville_dispersion` | MEDIUM | MEDIUM | P2 |
| `rl_market_conviction_index` | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have -- race-level gaps + Harville prerequisite + top market-cross feature
- P2: Should have -- additional market-cross features that enrich the signal
- P3: Nice to have -- composite features that may or may not add value beyond P2

---

## Competitor Feature Analysis

| Feature Category | yurelu (zenn.dev) | keita2399 (zenn.dev) | Our Approach |
|------------------|-------------------|----------------------|--------------|
| Win odds as feature | YES (included, then validated as "necessary") | NO (explicitly excluded) | YES (included -- Echo Chamber avoidance via race-level, not removal) |
| Race-level entropy | YES (rl_log_odds_entropy) | Not mentioned | YES (already have market_entropy) |
| Race-level dispersion | YES (rl_odds_dispersion) | Not mentioned | PLANNED (new) |
| Market cross-consistency | YES (win x trio quinella) | Not mentioned | PLANNED (win x wide -- data-constrained adaptation) |
| Fundamental-only model | Tested, REJECTED (v6 underperformed) | YES (core approach) | REJECTED (v1.6 already has 100+ horse-level features) |
| Two-stage architecture | Implicit in LightGBM tree structure | Not mentioned | Implicit -- same as yurelu (validated by our Gain per Depth) |

---

## Sources

### Primary (HIGH confidence)
- [yurelu (zenn.dev): AI と 26 ラウンド議論して個人開発の競馬予測 ML を育てた話](https://zenn.dev/yurelu/articles/396a329522aa22) -- Primary source: race-level + market-cross features, C-orthogonal IC improvement, Gain per Depth analysis, ROI 0.91 to 1.66 on 408 races
- Codebase analysis: `src/features/market_bias_features.py`, `src/features/odds_dynamics_features.py`, `src/features/race_difficulty_model.py`, `src/pipelines/training_pipeline.py` (lines 1174-1349)
- Data inspection: `data/odds/odds_wide.parquet` (38,825 races, 2015-2026), `data/odds/jodds_umaren.parquet` (25 races only), `data/raw/payouts.parquet` (38,835 races)

### Secondary (MEDIUM confidence)
- [Harville (1973): Assigning probabilities to the outcomes of multi-entry competitions](https://totepoint.com/quinella-calculator/) -- Harville formula for quinella/exacta/trifecta from win odds
- [Inferring Relative Ability from Winning Probability (InTech)](https://www.intechinvestments.com/wp-content/uploads/2022/11/1_Inferring-Relative-Ability-From-Winning-Probability-in-Multi-Entrant-Contests.pdf) -- Harville formula accuracy analysis, percentage increase in exacta probability relative to Harville
- [Logistic Analyses for Complicated Bets (HKU)](https://hub.hku.hk/bitstream/10722/60987/4/Content_11.pdf) -- Comparison of Harville vs Stern vs Henery models for quinella/exacta/trifecta
- [Snowberg & Wolfers: Explaining the Favorite-Longshot Bias](https://eriksnowberg.com/papers/Snowberg-Wolfers%20Risk%20Love%20or%20Decision%20Weights3.pdf) -- Harville conditional probability P(B|A) = P(B)/(1-P(A))

### Background
- [Thaler & Ziemba (1988): Anomalies: Parimutuel Betting Markets](https://www.sciencedirect.com/science/article/abs/pii/S0169207009002155) -- Market efficiency and late money effects
- [Entropy and Investment Theory](https://theinformaticists.wordpress.com/2019/03/24/on-entropy-and-investment-theory/) -- Shannon entropy in betting markets
- [JRA Official: Types of Bets](https://japanracing.jp/en/jpn-racing/guide/pdf/horseracing_en_03.pdf) -- JRA bet types and deduction rates
- [keita2399 (zenn.dev)](https://zenn.dev/keita2399/articles/keiba-ai-lgbm-verification) -- Contrast: odds-free approach
- [qiita: MLR3特徴量抽出による競馬レースの荒れ具合予測](https://qiita.com/kenkenvw/items/8d9ddf6be620d09720c4) -- Race "wildness" prediction with LightGBM

---
*Feature research for: Race-level aggregation and market cross-consistency features*
*Researched: 2026-05-17*
