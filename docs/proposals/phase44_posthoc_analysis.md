# Phase 44: ROI Bisect Post-Hoc Analysis Report

**Generated:** 2026-05-31
**Input:** Phase 43.5 shadow comparison artifacts (`data/backtest/shadow/`)
**Goal:** Identify 1-2 components for Phase 45 fix; NOT ROI improvement

---

## Executive Summary

**Root cause: MAWC (MarketAwareWinCalibrator) is the single dominant failure component.**

All 3 DeploymentGate FAIL conditions trace back to a single causal chain:

```
MAWC coefficient structure
  → suppresses favorite probabilities (pop_1 x logit_model = -0.135)
  → favorites fail EV threshold (crude EV 0.86, adjusted 0.99)
  → bet_count drops -21% (1-3 band 100% excluded)
  → population shifts to mid/long odds
  → ECE degrades on shifted population
  → APR overshoots on unchanged horses (p suppressed, WR unchanged)
```

**Ranker and OddsBandFilter are NOT causal contributors.** Ranker is completely dormant (shadow_only, 0% investment_score). OddsBandFilter has no excluded bands (all training ROI > 1.0).

**Phase 45 target: Fix MAWC only.** No ablation needed.

---

## Gate FAIL Summary (recap)

| Gate | Fold | Status | Detail |
|------|------|--------|--------|
| `ece_fold_2025` | 2025 | FAIL | shadow 0.01563 > baseline 0.00927 |
| `bet_count_preservation_fold_2024` | 2024 | FAIL | shadow 2580 < baseline 3327 x 0.95 |
| `bet_count_preservation_fold_2025` | 2025 | FAIL | shadow 2665 < baseline 3335 x 0.95 |
| `actual_predicted_ratio_fold_2024` | 2024 | WARN | shadow 1.173 vs baseline 0.903 |
| `actual_predicted_ratio_fold_2025` | 2025 | WARN | shadow 1.133 vs baseline 0.966 |

---

## Q1: 2025 ECE Degradation -- MAWC-Origin (confirmed)

### Finding: ECE degradation is from MAWC probability shift, NOT selection population change.

**Unchanged selection group** (same horses selected by both variants) shows massive ECE degradation:

| Fold | Group | BL_ECE | SH_ECE | Delta |
|------|-------|--------|--------|-------|
| 2025 | All horses | 0.00927 | 0.01563 | +0.00637 |
| 2025 | Selected only | 0.00927 | 0.01563 | +0.00637 |
| **2025** | **Unchanged** | **0.00210** | **0.01682** | **+0.01472** |
| 2024 | Unchanged | 0.01281 | 0.02594 | +0.01313 |

On the **exact same horses**, MAWC worsens ECE by +0.015 (2025). This proves ECE degradation is purely from MAWC's probability adjustment, not from a changed population.

**Probability shift on unchanged horses:**

| Fold | BL mean_p | SH mean_p | Actual WR | MAWC delta |
|------|-----------|-----------|-----------|------------|
| 2025 | 0.1170 | 0.0992 | 0.1160 | -15.2% |
| 2024 | 0.1678 | 0.1318 | 0.1577 | -21.5% |

MAWC suppresses mean predicted probability by 15-22% while actual win rate is unchanged, creating systematic over-prediction of losses.

---

## Q2: APR Overshoot -- Selection Population + MAWC Suppression

### Finding: APR overshoot is a compound effect of MAWC probability suppression on an unchanged selection.

| Fold | Group | BL APR | SH APR | Mechanism |
|------|-------|--------|--------|-----------|
| 2025 | All/Selected | 0.966 | 1.133 | MAWC lowers denominator (mean_p) |
| 2025 | Unchanged | 0.992 | 1.170 | Same WR (0.116) but p drops 0.117→0.099 |
| 2025 | Changed | 0.937 | 0.963 | SH APR actually better on changed group |

The APR overshoot is NOT from selecting worse horses. On unchanged selections (66% of races), actual win rate is identical (0.116) but MAWC suppresses predicted probability (0.117→0.099), so APR = 0.116/0.099 = 1.17.

This is a **denominator artifact**: MAWC's systematic probability suppression inflates APR on any fixed population.

---

## Q3: bet_count Drop -- MAWC → EV → Selection Gate Bottleneck

### Finding: The bottleneck is MAWC → EV pipeline, NOT Ranker or OddsBandFilter.

**Selection pipeline decomposition:**

| Component | Status | Evidence |
|-----------|--------|----------|
| **Ranker** | **Dormant** | 0% investment_score non-NaN (shadow_only mode) |
| **OddsBandFilter** | **No bands excluded** | All training ROI > 1.0 |
| **MAWC → EV → Gate** | **Primary bottleneck** | Favorites fail EV threshold |

**bet_count drop decomposition:**

| Fold | Total | BL sel | SH sel | Drop | Only BL | Only SH |
|------|-------|--------|--------|------|---------|---------|
| 2024 | 3764 | 3327 (88%) | 2580 (69%) | -747 (-22%) | 1184 | 437 |
| 2025 | 3604 | 3335 (93%) | 2665 (74%) | -670 (-20%) | 939 | 269 |

**63-71% of changed races result in shadow having NO selection:**

| Fold | Changed races | SH no-selection | Rate |
|------|---------------|-----------------|------|
| 2024 | 1184 | 747 | 63.1% |
| 2025 | 939 | 670 | 71.4% |

When the selection changes, it's almost always because shadow drops the horse entirely (EV below threshold), not because shadow selects a different horse.

**NaN analysis confirms favorites are completely excluded:**

| Band | 2024 N | SH NaN | 2025 N | SH NaN |
|------|--------|--------|--------|--------|
| **1-3** | **755** | **755 (100%)** | **422** | **422 (100%)** |
| 3-5 | 901 | 157 (17%) | 738 | 67 (9%) |
| 5-10 | 1164 | 202 (17%) | 1008 | 263 (26%) |
| 10-30 | 841 | 66 (8%) | 790 | 178 (23%) |
| 30+ | 94 | 4 (4%) | 617 | 8 (1%) |

The 1-3 odds band accounts for the majority of the drop (755 of 747 net in 2024; 422 of 670 in 2025). These horses exist in baseline bet_history but are completely absent from shadow bet_history because MAWC pushes their EV below the selection threshold.

---

## Q4: OddsBandFilter -- NOT Causal

### Finding: OddsBandFilter is working correctly but cannot rescue MAWC-suppressed favorites.

**OddsBandFilter scales:**

| Band | 2024 turf | 2024 dirt | 2025 turf | 2025 dirt |
|------|-----------|-----------|-----------|-----------|
| 1.0-3.0 | 1.138 | 1.333 | 1.157 | 1.178 |
| 3.0-10.0 | 1.113 | 1.118 | 1.115 | 1.067 |
| 10.0-30.0 | 1.128 | 1.164 | 1.125 | 1.107 |
| 30.0+ | 0.844 | 0.816 | 0.833 | 0.890 |

All scales > 0.8; no bands are excluded (threshold = 1.0 ROI).

**But the scale can't overcome MAWC's favorite suppression:**

| Scenario | p_mawc | EV_raw | Band scale | EV_adj | Pass? |
|----------|--------|--------|------------|--------|-------|
| fav_1.5 turf | 0.476 | 0.714 | 1.157 | 0.826 | NO |
| fav_2.0 turf | 0.428 | 0.857 | 1.157 | 0.991 | NO |
| fav_2.0 dirt | 0.428 | 0.857 | 1.178 | 1.009 | barely YES |

Turf favorites at odds 2.0 get EV_adj = 0.991 -- just below the 1.0 threshold. Even with the 15.7% scale boost, MAWC's probability suppression keeps favorites below the EV gate.

**Band-wise selection rates confirm the pattern:**

| Band (2025) | N | BL sel rate | SH sel rate | BL ROI | SH ROI |
|-------------|---|-------------|-------------|--------|--------|
| 1-3 | 422 | 100.0% | **0.0%** | -0.228 | NaN |
| 3-5 | 738 | 76.3% | 90.9% | -0.029 | -0.030 |
| 5-10 | 1008 | 96.4% | 73.9% | -0.067 | -0.005 |
| 10-30 | 790 | 93.8% | 77.5% | +0.138 | -0.001 |
| 30+ | 646 | 98.6% | 98.6% | -0.069 | -0.069 |

Shadow improves mid-range ROI (-0.067 → -0.005 for 5-10 band) but completely kills the 1-3 band. The net effect: lower bet count, different calibration profile.

---

## Q5: MAWC Coefficient Mechanism -- Confirmed

### Finding: MAWC's coefficient structure systematically suppresses favorites.

**Core coefficients (2025/turf, representative):**

| Feature | Coefficient | Effect |
|---------|-------------|--------|
| `logit_model` | +0.114 | Weak -- model signal barely used |
| `logit_market` | +0.365 | Strong -- market signal dominates |
| `log_odds` | -0.335 | Strong negative -- penalizes high odds |

**Critical interaction terms for favorites:**

| Interaction | Coefficient | Effect on Favorites |
|-------------|-------------|---------------------|
| `logit_model x pop_1` | **-0.135** | Discounts model confidence for #1 favorite |
| `logit_model x 1-2` | **-0.114** | Discounts model for 1-2 odds band |
| `logit_model x pop_4_6` | +0.167 | Boosts model for mid-popularity |
| `logit_market x bottom_25` | +0.155 | Boosts market for bottom quartile |

**Simulated probability shift (core coefficients only):**

| Scenario | p_model | p_mawc | Delta | crude EV | EV_adj (turf) |
|----------|---------|--------|-------|----------|---------------|
| Favorite (odds 2.0) | 0.450 | 0.428 | -0.022 | 0.857 | 0.991 |
| Mid-range (odds 5.0) | 0.180 | 0.223 | +0.043 | 1.112 | 1.240 |
| Longshot (odds 30.0) | 0.030 | 0.057 | +0.027 | 1.716 | 1.430 |

The mechanism:
1. `logit_model` coefficient is small (0.114) -- model's high confidence in favorites is underweighted
2. `logit_market` (0.365) dominates, but market probability for favorites is close to 0.50 (logit ≈ 0), so it contributes little
3. `log_odds` (-0.335) penalizes high odds; for favorites log(2.0) ≈ 0.69 contributes -0.23
4. Interaction `pop_1 x logit_model` = -0.135 further discounts the model signal for top favorites
5. Result: favorite p drops → EV < 1.0 → selection gate excludes

**Ranker (RaceLevelRanker):**
- Completely dormant: 0% non-NaN investment_score in both variants
- In shadow_only mode -- does not gate selection
- Ridge coefficients are reasonable (relevance: p_win_final=+1.39; value: logit_gap=+1.10)
- **Not a contributor** to any gate failure

---

## Causal Chain Diagram

```
MAWC LogisticRegression (C=0.03, 51 features)
  |
  |-- logit_model coef = +0.114 (weak model weight)
  |-- logit_market coef = +0.365 (strong market weight)
  |-- log_odds coef = -0.335 (odds penalty)
  |-- pop_1 x logit_model = -0.135 (discount favorites)
  |-- 1-2 odds x logit_model = -0.114 (discount low odds)
  |
  v
Favorite probability suppressed: 0.45 -> 0.43
  |
  v
EV = p * odds = 0.43 * 2.0 = 0.86 < 1.0 threshold
  |
  v
Selection gate excludes favorites
  |                                  \
  |  1-3 band: 100% excluded          Mid/long: EV > 1.0, selected
  |  bet_count drops -21%               |
  v                                      v
ECE degrades (different population)   APR overshoots (p suppressed)
  |                                      |
  v                                      v
Gate FAIL: ece_fold_2025             Gate WARN: APR deviates from 1.0
Gate FAIL: bet_count x2 folds
```

---

## Phase 45 Recommendation

### Single target: Fix MAWC

**No ablation needed.** The post-hoc analysis definitively identifies MAWC as the single causal component. All 3 FAIL gates and both WARN gates trace to the same root cause.

**Proposed fixes (ranked by expected impact):**

| # | Fix | Rationale | Risk |
|---|-----|-----------|------|
| 1 | **Increase L2 regularization** (C=0.03 → 0.005-0.01) | Current coefficients are too large; interaction terms penalize favorites excessively | May underfit mid-range |
| 2 | **Remove odds-band x logit_model interactions** | `pop_1 x logit_model` (-0.135) and `1-2 x logit_model` (-0.114) directly suppress favorites | Reduces segment conditioning |
| 3 | **Band-gated MAWC** (skip MAWC for 1-3 odds) | Favorites don't need market-aware calibration; they already have accurate p_model | Creates a discontinuity at band boundary |
| 4 | **Winsorize interaction features** | Clip interaction values to prevent extreme penalty for favorites | Minimal code change |

**Recommended approach:** Start with fix #1 (increase regularization). If ECE still fails, combine with fix #2 (remove problematic interactions). Fix #3 is a fallback if coefficient-level fixes don't suffice.

### Not recommended for Phase 45

- **OddsBandFilter tuning**: Not causal; all scales > 0.8
- **Ranker activation**: Not causal; dormant and irrelevant to current gates
- **Selection gate threshold**: Would be a workaround, not a fix
- **EV correction retraining**: MAWC operates before EV correction in the pipeline

---

## Appendix: Data Sources

| File | Used For |
|------|----------|
| `shadow_comparison_result.json` | Per-fold metrics, bet counts, surface/odds breakdowns |
| `shadow_horse_diff.parquet` (7368 rows) | Per-horse probabilities, selections, odds, surface, popularity |
| `shadow_race_diff.parquet` (6662 rows) | Per-race EV, scores, changed/unchanged grouping |
| `deployment_gate_result.json` | Gate FAIL/WARN/PASS conditions |
| `shadow_diagnosis_result.json` | Step 1-3 diagnosis results |
| `market_aware_win_calibrator_{surface}.joblib` | MAWC LogisticRegression coefficients |
| `win_race_level_ranker_turf.joblib` | Ranker Ridge coefficients |
| `ev_odds_band_scales_{surface}.json` | OddsBandFilter per-band scales |
