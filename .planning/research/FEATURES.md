# Feature Research: v1.8 Turf Precision Calibration (5 Feature Groups)

**Domain:** Horse racing ML prediction -- turf model improvement, ROI 97.8% to 100%+
**Researched:** 2026-05-19
**Confidence:** HIGH (A: haron time -- existing pattern, well-understood), HIGH (C: race_rank -- existing pattern), HIGH (D: interactions -- existing pattern), MEDIUM (B: popularity band calibration -- novel, needs validation), HIGH (E: EV correction features -- additive to existing model)

## Executive Summary

This research covers 5 feature groups for the v1.8 milestone, all targeting the turf model's negative b_difference (-0.004). The goal is to flip turf from market-losing to market-beating, pushing overall ROI from 97.8% past 100%.

The 5 groups fall into two tiers by implementation risk:

**Tier 1 (Safe, proven patterns):**
- A: Haron time features -- extends existing `_compute_haron_stats` with HaronTimeL4 support and per-horse closing speed rank. Pure history aggregation of already-ETL'd data.
- C: Turf relative features -- adds race_rank variants for existing features (form_trend, blood_total_wr, etc.) using the established `add_race_transforms` pattern.
- D: Conditional interaction features -- adds 3 new domain-knowledge interaction terms to the existing `interaction_features.py` module.
- E: Regime x surface EV correction -- adds 3 columns to `EVCorrectionModel.FEATURE_COLS`.

**Tier 2 (Novel, needs validation):**
- B: Popularity band calibration -- OOF residual ratio by popularity band, producing per-band scaling factors. This is a new calibration layer that sits between Isotonic calibration and OddsBandFilter.

---

## Feature Group A: Haron Time Features (Per-Horse Closing Speed)

### Data Source

EveryDB2 SE (UMA_RACE) fields already ETL'd into `entries.parquet`:
- `HaronTimeL3` (varchar 3, 99.9 sec) -- last 3 furlongs time, per horse, per race. Already in POST_RACE_COLS, already used by `_compute_haron_stats`.
- `HaronTimeL4` (varchar 3, 99.9 sec) -- last 4 furlongs time, per horse, per race. Already in POST_RACE_COLS, NOT yet used in feature computation.

EveryDB2 RA (RACE) fields also available:
- `HaronTimeS3` (varchar 3) -- first 3 furlongs, race-level (leader only)
- `HaronTimeS4` (varchar 3) -- first 4 furlongs, race-level (leader only)
- `HaronTimeL3` (varchar 3) -- last 3 furlongs, race-level (leader only)
- `HaronTimeL4` (varchar 3) -- last 4 furlongs, race-level (leader only)

### PIT Safety Analysis

**CRITICAL:** All HaronTime fields are POST_RACE. The existing pattern in `horse_history_features.py` is PIT-safe:
1. Line 227: `past = history[history["race_date"] < target_date]` -- only uses past races
2. The `_compute_haron_stats` function aggregates across past races only
3. The `POST_RACE_COLS` whitelist in `domain/types.py` already includes `harontimel3` and `harontimel4`
4. The `SAFE-01` gate in `feature_engine.py` (line 391-400) drops all POST_RACE columns from output

**New features must follow the same pattern:** aggregate HaronTimeL3/L4 from PAST races only, never from the current race.

### Table Stakes

| Feature | Why Expected | Complexity | Dependency | Notes |
|---------|--------------|------------|------------|-------|
| `harontimel5_avg` | Already exists. EMA-weighted closing speed average | LOW (done) | harontimel3 from past races | Already implemented |
| `harontimel5_zscore` | Already exists. Z-score vs distance_bin distribution | LOW (done) | expanding_stats lookup | Already implemented |
| `harontime_late_trend` | Already exists. Last 2 vs first 3 trend | LOW (done) | harontimel3 from past races | Already implemented |

### Differentiators (New for v1.8)

| Feature | Value Proposition | Complexity | Dependency | Notes |
|---------|-------------------|------------|------------|-------|
| `harontimel4_avg` | 4-furlong closing speed (longer distance indicator, especially for 2000m+) | LOW | harontimel4 from past races | HaronTimeL4 field note: "basically only L3 is set, but older data has L4 set (with L3 at default)". Must handle mutual exclusivity |
| `harontimel4_zscore` | Z-scored version of L4 average | MEDIUM | expanding_stats for L4 | Need separate expanding stats computation |
| `haron_l3l4_ratio` | Ratio of L3/L4 average -- captures sustained vs burst closing ability | LOW | harontimel3 + harontimel4 | When L4 unavailable, use NaN |
| `harontimel3_race_rank` | Per-horse closing speed rank within race (from past) | LOW | harontimel3 from past races | Uses existing `add_race_transforms` pattern |

### Data Quality Concern: HaronTimeL3 vs L4 Mutual Exclusivity

From the EveryDB2 spec (04-UMA_RACE field 58):
> "basically only L3 is set (L4 is default). However, older data has L4 set (in that case L3 is default)"

This means for most races we get L3, for some older races we get L4, rarely both. The implementation must:
1. Coalesce: use L3 if available, else L4 (converting scale appropriately)
2. Or: maintain separate L3/L4 feature streams with NaN for missing
3. A combined "best available closing speed" feature is the safest approach

### Recommended Implementation

Extend `_compute_haron_stats` to also process `harontimel4`:
- Add `harontimel4` to `cols_horse` in `compute()` (line 503-522)
- Compute `harontimel4_avg` (EMA-weighted, same as L3)
- Compute `haron_l3l4_ratio` when both are available
- Add `harontimel3_avg` (not EMA, just raw average for rank computation) to `add_race_transforms` list

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Current race HaronTime as feature | POST_RACE leakage -- would inflate backtest but fail in production | Aggregate from past races only |
| LapTime-based per-horse pace features | LapTime1~25 is race-level (leader only), not per-horse | Use for race-level pace profile (Feature Group A extension) |
| Per-furlong position from Jyuni1c~4c | Already aggregated as jyuni1c_avg, jyuni4c_avg, closing_index_avg | Existing features sufficient |

---

## Feature Group A Extension: Lap Pace Features (Race-Level)

### Data Source

EveryDB2 RA (RACE) fields -- LapTime1 through LapTime25 (varchar 3, 99.9 sec each).
These are LEADER's lap times at each 200m point. NOT per-horse.

### PIT Safety Analysis

LapTime1~25 are POST_RACE (only available after the race). However, these are RACE-level fields, not horse-level. The safe approach is:
1. Aggregate LapTime from PAST races (same venue/distance) to build pace profiles
2. Use the pace profile as context for the current race
3. NEVER use current race's LapTime

### Differentiators

| Feature | Value Proposition | Complexity | Dependency | Notes |
|---------|-------------------|------------|------------|-------|
| `lap_early_ratio` | Ratio of first 3 furlongs to total time -- "early pace pressure" indicator | MEDIUM | LapTime from past races at same venue/distance | Must handle variable lap counts by distance |
| `lap_closing_ratio` | Ratio of last 3 furlongs to total time -- "closing bias" indicator | MEDIUM | LapTime from past races | Captures track-specific pace patterns |
| `pace_profile_class` | Classification: "fast-early", "slow-early", "even" based on pace shape | MEDIUM | LapTime from past races | Categorical feature for LightGBM |

### ETL Requirements

LapTime1~25 are in the RA (RACE) table, which IS already ETL'd into `races.parquet`. However, the current ETL only maps limited int columns (`trackcd`, `kyori`, `tenkocd`, `syussotosu`, `honsyokin`). LapTime columns need to be added to the ETL float conversion rules.

**Required ETL change:** Add to `_TABLE_TYPE_RULES["races"]["float"]`:
```python
"float": ["trackcd", "kyori", "tenkocd", "syussotosu", "honsyokin",
          "laptime1", "laptime2", ..., "laptime25",
          "harontimes3", "harontimes4", "harontimel3", "harontimel4"],
```

**Complexity:** MEDIUM -- requires ETL modification + full re-extract for historical data to populate these fields. This is the primary risk for this feature group.

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Current race LapTime features | POST_RACE leakage | Aggregate from past races at same venue/distance |
| Per-horse lap decomposition | LapTime is leader-only, not per-horse | Use race-level aggregation |

---

## Feature Group B: Popularity Band Calibration

### Concept

The idea is to add a calibration layer that corrects systematic EV biases by popularity band (odds range). Currently:
- Isotonic calibration operates on the full EV distribution
- OddsBandFilter operates on the betting decision layer (skip bands with ROI < 100%)

The new layer would:
1. Compute OOF residual ratio by popularity band during training
2. Apply per-band scaling factors to EV predictions

### Expected Behavior

For each popularity band (e.g., 1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+):
1. Collect OOF predictions and actual results
2. Compute `actual_return / predicted_EV` for each band
3. Derive a scaling factor: `scale = mean(actual_EV / predicted_EV)` per band
4. Apply: `ev_calibrated *= scale` for horses in that band

### Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| OOF-based calibration | Standard practice in competitive ML | MEDIUM | Must use OOF (not in-fold) to avoid leakage |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `pop_band_scale_1_3` | Scaling factor for odds 1.0-3.0 band | MEDIUM | Most populated band, should be closest to 1.0 |
| `pop_band_scale_3_10` | Scaling factor for odds 3.0-10.0 | MEDIUM | Where most value bets live |
| `pop_band_scale_10_30` | Scaling factor for odds 10.0-30.0 | MEDIUM | Longshot bias correction territory |
| `pop_band_scale_30_plus` | Scaling factor for odds 30.0+ | MEDIUM | Sparse data, needs regularization |

### PIT Safety Analysis

**CRITICAL:** The calibration factors must be computed from OOF predictions only, NOT from in-fold predictions. The existing v1.4 already handles this correctly:
- `OddsBandFilter.calibrate()` uses `training_bet_history` which is OOF-generated
- Isotonic calibration uses OOF predictions

The popularity band calibration should follow the same OOF-only pattern. Implementation approaches:

1. **Post-hoc layer (safer):** Compute after Isotonic calibration, applying residual correction by band. This is additive and can be validated independently.
2. **Integrated layer:** Modify EVCorrectionModel to include band interaction terms.

**Recommendation:** Start with the post-hoc layer approach. It is safer, easier to validate, and can be toggled on/off for A/B testing.

### Integration with Existing Layers

Current EV pipeline:
```
P_pred x E_pred -> EVCorrectionModel -> Isotonic -> OddsBand scaling -> Final EV
```

With popularity band calibration:
```
P_pred x E_pred -> EVCorrectionModel -> Isotonic -> Pop-band calibration -> OddsBand scaling -> Final EV
```

The pop-band calibration sits between Isotonic and OddsBand. It corrects systematic EV miscalibration within each band, while OddsBand makes the binary skip/no-skip decision.

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Overfitting to training period's band ratios | HIGH | Use expanding window (not full-history) for band ratios. Regularize toward 1.0 with Bayesian shrinkage |
| Sparse bands (30.0+) having unreliable ratios | MEDIUM | Apply hierarchical shrinkage: sparse bands borrow from parent band's ratio |
| Interaction with existing OddsBand scaling | MEDIUM | Must ensure the two layers don't double-correct. Pop-band calibration should correct EV values; OddsBand should only make skip decisions |

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Per-horse popularity band features in Stage1 | Stage1 predicts P(hit) -- popularity is a market feature, adding it creates circularity | Apply calibration at EV level only (after Stage1+Stage2) |
| Fine-grained bands (< 5 bands) | Overfitting risk, unstable ratios | Use 4-5 bands (matching OddsBandFilter boundaries) |
| Band-specific model retraining | Exponential complexity increase | Post-hoc scaling layer only |

---

## Feature Group C: Turf Relative Features (Race-Rank Variants)

### Concept

Convert absolute feature values into within-race relative rankings. A horse with `form_trend = 0.3` means nothing in isolation, but if it ranks 2nd out of 16 horses in that race, it is highly significant.

### Existing Pattern

The codebase already has `add_race_transforms` in `horse_history_features.py` (lines 1325-1351). It computes percentile ranks for:
- `norm_finish_logit_avg` -> `norm_finish_logit_avg_race_rank`
- `harontimel5_avg` -> `harontimel5_avg_race_rank`
- `harontimel5_zscore` -> `harontimel5_zscore_race_rank`
- `timediff_avg` -> `timediff_avg_race_rank`
- `jyuni1c_avg` -> `jyuni1c_avg_race_rank`
- `jyuni4c_avg` -> `jyuni4c_avg_race_rank`
- `closing_index_avg` -> `closing_index_avg_race_rank`

These race_rank features are already in the model (confirmed by `gpd_diagnostics.py` line 91: `harontimel5_avg_race_rank` exists).

### New Race-Rank Features for v1.8

| Feature | Source Feature | Why Needed | Complexity | Notes |
|---------|---------------|------------|------------|-------|
| `form_trend_race_rank` | form_trend | Relative form trajectory -- a horse improving in a field of improving horses is different from improving in a field of declining horses | LOW | Simple `groupby("race_id").rank(pct=True)` |
| `blood_total_wr_race_rank` | blood_total_wr | Bloodline win rate relative to competition | LOW | Already exists as blood feature |
| `haron_zscore_trend_race_rank` | haron_zscore_trend | Relative closing speed improvement trajectory | LOW | |
| `class_adj_formetric_race_rank` | class_adj_formetric | Class-adjusted form relative to competition | LOW | |
| `time_improvement_rate_race_rank` | time_improvement_rate | Relative speed improvement rate | LOW | |
| `position_improvement_rate_race_rank` | position_improvement_rate | Relative position improvement rate | LOW | |
| `freshness_score_race_rank` | freshness_score | Relative freshness (rest quality x recent form) | LOW | |

### PIT Safety

These race_rank features are computed from OTHER features that are already PIT-safe (all are from past-race aggregations). The race-rank computation itself uses `groupby("race_id").rank(pct=True)` which operates on already-computed per-horse features. No new data access required.

**However:** The race-rank requires knowing ALL horses' feature values in the same race. This is inherently available at both training time (full data) and inference time (all entries known). No PIT concern.

### Implementation

Extend the `add_race_transforms` method to include the new columns in `race_rank_cols`.

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Race-rank for jockey features | jockey features already in Stage2 (EV correction), not Stage1 | Keep race_rank for fundamental features only |
| Race-rank for categorical features | Meaningless -- cannot rank categories | Only apply to numeric features |
| Race-rank for market features | Market features (odds, popularity) already provide relative positioning via implied probabilities | Redundant information |

---

## Feature Group D: Conditional Interaction Features

### Concept

Add domain-knowledge interaction terms that capture non-linear relationships between race conditions and horse capabilities. The existing `interaction_features.py` already has 12 interaction features (3 original + 9 from v1.6).

### New Interactions for v1.8

| Feature | Components | Domain Rationale | Complexity | Type |
|---------|-----------|-----------------|------------|------|
| `grade_x_form_trend` | grade_code x form_trend | Form improvement matters more in higher-grade races (G1/G2). A horse trending upward in a Grade 1 has a different signal than in a maiden claimer | LOW | Categorical x Numeric -> Numeric |
| `distance_x_closing_index` | distance_bin x closing_index_avg | Closing ability (come-from-behind) is more valuable at longer distances. At sprint distances, early speed matters more | LOW | Categorical x Numeric -> Numeric |
| `grade_x_blood_prize_log` | grade_code x blood_prize_log | Bloodline prize money is a stronger signal in higher-class races where competition is tighter | LOW | Categorical x Numeric -> Numeric |

### Implementation Pattern

Following the existing `compute_interaction_features` pattern:

```python
# Categorical x Numeric: encode grade as numeric, then multiply
_GRADE_NUM = {"G1": 5, "G2": 4, "G3": 3, "OP": 2, ...}

if "grade_code" in df.columns and "form_trend" in df.columns:
    grade_num = df["grade_code"].map(_GRADE_NUM).fillna(1.0)
    df["grade_x_form_trend"] = (grade_num * df["form_trend"]).where(
        df["form_trend"].notna() & df["grade_code"].notna(),
        other=float("nan"),
    )

# Categorical x Numeric: encode distance_bin as numeric
if "distance_bin" in df.columns and "closing_index_avg" in df.columns:
    dist_num = df["distance_bin"].map({"sprint": 1, "mile": 2, "intermediate": 3, "long": 4}).fillna(0)
    df["distance_x_closing_index"] = (dist_num * df["closing_index_avg"]).where(
        df["closing_index_avg"].notna(),
        other=float("nan"),
    )

if "grade_code" in df.columns and "blood_prize_log" in df.columns:
    grade_num = df["grade_code"].map(_GRADE_NUM).fillna(1.0)
    df["grade_x_blood_prize_log"] = (grade_num * df["blood_prize_log"]).where(
        df["blood_prize_log"].notna() & df["grade_code"].notna(),
        other=float("nan"),
    )
```

### PIT Safety

All component features are already PIT-safe:
- `grade_code` -- pre-race condition
- `form_trend` -- computed from past races
- `distance_bin` -- pre-race condition
- `closing_index_avg` -- computed from past races
- `blood_prize_log` -- bloodline feature (lifetime, pre-race)

Interaction of PIT-safe features is PIT-safe.

### Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| More than 3-5 new interactions | v1.6 already added 12. Diminishing returns + overfitting risk. LightGBM learns interactions automatically | Limit to 3 high-conviction interactions |
| Interaction with market features | Market features (odds, popularity) are in a different IC tier. Mixing fundamental x market interactions dilutes the signal | Keep interactions within fundamental x condition space |
| Polynomial interactions (x^2, x^3) | LightGBM handles non-linearity natively | Only multiplicative interactions between different domains |

---

## Feature Group E: Regime x Surface EV Correction

### Concept

Add regime detection and surface interaction features to the EVCorrectionModel's FEATURE_COLS, allowing the P-correction and E-correction models to learn surface-specific and regime-specific correction patterns.

### Current State of EVCorrectionModel.FEATURE_COLS

The model already includes:
- `surface` (categorical)
- `distance_bin` (categorical)
- `track_condition_code` (numeric)
- `field_size` (numeric)
- Market features: `signed_log_error_win`, `abs_log_error_win`, `market_entropy`, `popularity_rank`
- Race-level: `rl_log_odds_entropy`, `rl_odds_dispersion`, etc.
- Market-cross: `rl_favorite_in_wide_top1`, `rl_trio_overlap`, etc.
- Jockey/trainer context features

### New Features to Add to FEATURE_COLS

| Feature | Source | Why Needed | Complexity | Notes |
|---------|--------|------------|------------|-------|
| `regime_state` | RegimeDetector output (aggressive/conservative/collapsed) | EV correction should differ by regime -- aggressive regime may need less correction, conservative needs more | LOW | Add as categorical feature. RegimeDetector already outputs this |
| `surface_x_popularity` | surface x popularity_rank interaction | The favorite-longshot bias differs between turf and dirt. Turf has more efficient markets, dirt has more noise | LOW | Encode surface as numeric (1=turf, 2=dirt) and multiply by popularity_rank |
| `market_entropy_x_surface` | market_entropy x surface interaction | Market entropy means different things on turf vs dirt. High entropy turf = genuinely competitive; high entropy dirt = potentially random | LOW | Encode surface as numeric and multiply by market_entropy |

### Implementation

1. Add 3 columns to `EVCorrectionModel.FEATURE_COLS`
2. Ensure `regime_state` is available in the DataFrame passed to `correct_ev()`
3. Compute `surface_x_popularity` and `market_entropy_x_surface` in `_add_interaction_features()`
4. Add `regime_state` to the categorical feature list in `_prepare_features()`

### PIT Safety

- `regime_state` -- computed from pre-race market indicators (fav_rate x overround). Already PIT-safe.
- `surface` -- pre-race condition
- `popularity_rank` -- pre-race (from tanodds snapshot)
- `market_entropy` -- pre-race (from tanodds snapshot)

All components are PIT-safe, so their interactions are also PIT-safe.

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Regime state may be unstable across training/inference | LOW | RegimeDetector is deterministic given the same input features. Already used in betting pipeline. |
| Surface interaction may not improve E-correction | LOW | E-correction only uses winners (kakuteijyuni==1), which is a small sample. The LightGBM will naturally ignore if not useful. |

---

## Feature Dependencies

```
Feature Group A (Haron Time)
    |-- harontimel4_avg
    |       requires: harontimel4 in cols_horse (already in POST_RACE_COLS)
    |       requires: ETL float conversion for harontimel4 (already done -- etl.py line 97)
    |       depends on: _compute_haron_stats extension
    |
    |-- harontimel4_zscore
    |       requires: expanding_stats for L4 (new computation)
    |       depends on: harontimel4_avg
    |
    |-- haron_l3l4_ratio
    |       requires: both harontimel3 and harontimel4 from past
    |       depends on: harontimel4_avg
    |
    |-- harontimel3_race_rank
            requires: add_race_transforms extension
            depends on: existing harontimel3 data

Feature Group A Extension (Lap Pace)
    |-- lap_early_ratio, lap_closing_ratio, pace_profile_class
    |       requires: LapTime1~25 in races.parquet
    |       requires: ETL modification to extract LapTime columns
    |       BLOCKER: ETL re-extraction needed (full run, ~10 min)

Feature Group B (Popularity Band Calibration)
    |-- pop_band_scale_* (4 features)
    |       requires: OOF predictions (already generated in training)
    |       requires: Isotonic calibration (already implemented)
    |       depends on: Post-isotonic calibration layer (new)
    |       NO ETL changes needed

Feature Group C (Turf Relative Features)
    |-- *_race_rank (7 features)
    |       requires: add_race_transforms extension
    |       requires: source features already computed (all exist)
    |       NO ETL changes needed

Feature Group D (Conditional Interactions)
    |-- grade_x_form_trend, distance_x_closing_index, grade_x_blood_prize_log
    |       requires: interaction_features.py extension
    |       requires: source features available (all exist)
    |       NO ETL changes needed

Feature Group E (Regime x Surface EV Correction)
    |-- regime_state, surface_x_popularity, market_entropy_x_surface
    |       requires: EVCorrectionModel.FEATURE_COLS update
    |       requires: regime_state available in EV pipeline (already computed)
    |       NO ETL changes needed
```

### Implementation Order (Dependency-Driven)

1. **C (race_rank)** -- Zero new data, extends existing pattern, lowest risk
2. **D (interactions)** -- Zero new data, extends existing pattern
3. **A (haron time, excluding Lap)** -- Uses existing ETL data, extends existing pattern
4. **E (EV correction features)** -- Extends existing FEATURE_COLS
5. **B (popularity band calibration)** -- New calibration layer, needs careful validation
6. **A Extension (Lap pace)** -- Requires ETL change, highest implementation risk

---

## Anti-Features Summary

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Current race POST_RACE as features | Leakage -- backtest inflated, production fails | Aggregate from past races only (existing PIT pattern) |
| LSTM/Transformer for time-series closing speed | 5-15 past races is too few for sequence models | Statistical aggregation (mean, z-score, trend) |
| More than 7 new race_rank features | Diminishing returns, adds noise | Focus on features with domain justification |
| Per-band model retraining for calibration | Exponential complexity | Post-hoc scaling layer only |
| Removing odds features | Proven to hurt IC (v1.6 experiment) | Add features, don't remove |
| LapTime per-horse decomposition | LapTime is leader-only | Use race-level pace profile |
| Complex multi-order interactions | LightGBM learns these automatically | Limit to 2-way interactions with domain justification |

---

## Complexity Summary

| Feature Group | New Features | ETL Change | Code Changes | PIT Risk | Validation Need |
|---------------|-------------|------------|--------------|----------|-----------------|
| A: Haron Time (core) | 3-4 | None | horse_history_features.py, domain/types.py | LOW (existing pattern) | IC evaluation |
| A: Lap Pace (extension) | 3 | REQUIRED (ETL yaml + re-extract) | New module + etl.py | MEDIUM (new ETL path) | IC evaluation + ETL verification |
| B: Pop Band Calibration | 4 scales | None | New calibration layer | MEDIUM (OOF-only validation) | OOF residual analysis |
| C: Turf Relative | 7 race_rank | None | horse_history_features.py | LOW (existing pattern) | IC evaluation |
| D: Conditional Interactions | 3 | None | interaction_features.py | LOW (existing pattern) | IC evaluation |
| E: EV Correction | 3 | None | ev_correction_model.py | LOW (additive) | IC evaluation |

---

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/features/horse_history_features.py` (existing haron stats pattern, lines 210-262)
- Codebase analysis: `src/features/interaction_features.py` (existing interaction pattern)
- Codebase analysis: `src/models/ev_correction_model.py` (FEATURE_COLS, EV correction architecture)
- Codebase analysis: `src/domain/types.py` (POST_RACE_COLS including harontimel3/l4)
- Codebase analysis: `src/db/etl.py` (ETL float conversion for entries, line 97)
- EveryDB2 spec: `docs/everydb2/03-RACE.md` (LapTime1~25, HaronTimeS3/S4/L3/L4)
- EveryDB2 spec: `docs/everydb2/04-UMA_RACE.md` (HaronTimeL3/L4, Jyuni1c~4c)

### Secondary (MEDIUM confidence)
- PROJECT.md v1.8 milestone requirements (5 active feature groups)
- Existing GPD diagnostics: `src/models/gpd_diagnostics.py` (FEATURE_CATEGORY_MAP with 179 features)

---
*Feature research for: v1.8 Turf Precision Calibration -- 5 feature groups*
*Researched: 2026-05-19*
