# Feature Validity Report: BT(2025) vs PT(2026/4)

> Date: 2026-04-14
> Purpose: Explain 62pt ROI gap (BT 136.6% vs PT 74.4%)

## Executive Summary

**Primary Root Cause: track_condition_code=0 data missing for 4/11 and 4/12**

The model was trained exclusively on TCC values 1-4 (Good to Heavy), but PT races on 4/11 and 4/12 have TCC=0 (data missing). This causes:
- `is_good_track = (TCC <= 1) = True` — model treats unknown/bad tracks as "good"
- Biased predictions on 70 races (50% of PT sample)
- These results should be discarded

## Detailed Findings

### 1. track_condition_code Distribution (CRITICAL)

| Source | Mean | Median | Range | Notes |
|--------|------|--------|-------|-------|
| BT 2025 | 1.39 | 1.0 | [1, 4] | 72.8% Good, 17.9% Good-Soft |
| PT 4/4 | 1.7 | — | [1, 3] | OK — normal spring conditions |
| PT 4/5 | 3.0 | — | [2, 4] | WARN — genuinely soft/heavy |
| PT 4/11 | **0.0** | **0** | [0] | **FAIL — all data missing** |
| PT 4/12 | **0.0** | **0** | [0] | **FAIL — all data missing** |

**Z-score: +3.157** (highly significant shift)

### 2. New Features (pace/course) — ALL STABLE

| Feature | BT Mean | PT Mean | Z-score | NaN(BT) | NaN(PT) | Status |
|---------|---------|---------|---------|---------|---------|--------|
| pace_aptitude | 0.066 | 0.351 | +0.998 | 65.3% | 62.9% | OK (near-threshold) |
| front_pace_wr | 0.103 | 0.081 | -0.544 | 11.1% | 0.6% | OK |
| closing_pace_wr | 0.089 | 0.089 | +0.004 | 11.1% | 0.6% | OK |
| course_wr | 0.093 | 0.091 | -0.072 | 0.3% | 0.0% | OK |
| course_distance_wr | 0.093 | 0.091 | -0.075 | 0.3% | 0.0% | OK |

**Conclusion**: Phase 4 feature implementation is correct. No distribution drift.

### 3. Other Notable Shifts (|Z| > 1.0)

| Feature | Z-score | Explanation |
|---------|---------|-------------|
| dmkubun | -7.45 | Different data source encoding (expected) |
| track_condition_code | +3.16 | **See #1 above** |
| kettonum1 | -2.77 | ID column (not predictive) |
| dmgosap | -1.57 | Prize money diff (expected variance) |
| kakuteijyuni | -1.39 | Bet selection bias (PT picks better finishers) |
| trackcd | +1.30 | PT more dirt-heavy |
| tozaicd | -1.07 | PT more central-track races |

### 4. Sample Size Concern

| Metric | BT 2025 | PT 2026/4 |
|--------|---------|-----------|
| Total rows | 46,499 | ~219 |
| Valid bets | 5,859 | 95 |
| Time span | 365 days | 4 days |
| Statistical power | High | Very Low (noise +/-30%) |

## Recommended Actions

### Immediate (P0)
1. **Exclude 4/11 and 4/12 from PT evaluation** — TCC=0 makes results unreliable
2. **Re-evaluate PT with only 4/4 and 4/5** — these have valid track condition data
3. **Fix ETL/data pipeline** — investigate why TCC=0 for recent dates

### Short-term (P1)
4. **Extend PT to 90+ days** — need statistical significance for ROI estimation
5. **Add TCC=0 guard in FeatureEngine** — default to TCC=2 (Good-Soft) or skip race
6. **Consider track-condition sub-models** — separate models for Good vs Soft/Heavy tracks

### Long-term (P2)
7. **Monitor pace_aptitude distribution** — Z=0.998 near significance threshold
8. **Add data quality checks to paper trading** — warn/skip when critical features are missing

## Files Generated

- `data/backtest/bt_pt_feature_comparison_v2.csv` — Full comparison data
- `scripts/compare_bt_pt_features.py` — Comparison script (fixed merge collision)
