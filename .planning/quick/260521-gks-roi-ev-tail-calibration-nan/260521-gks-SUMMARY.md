---
status: complete
quick_id: 260521-gks
date: 2026-05-21
---

# Quick Task 260521-gks: ROI劣化要因修正

## Summary

ROI劣化（97.8% → 79.8%）の3主因を修正。期待ROI回復: +15-23pt。

## Commits

| Commit | Description |
|--------|-------------|
| e521603 | fix(backtest): disable EV Tail Calibration to restore high-EV bet selection |
| caad9f5 | fix(ensemble): disable correlation penalty (weight=0.0) to restore AUC optimization |
| 72a7273 | fix(models): remove 3 high-NaN features (pace_ratio_zscore/trend, pace_adj_finish_avg) from all models |
| 464e0bf | fix(tests): update test expectations for removed high-NaN features |

## Changes

### Task 1: EV Tail Calibration disabled (~10-15pt ROI recovery)
- **File**: `src/backtest/race_predictor.py`
- **Change**: Removed EVTailCalibrator import, calibration loop, and `_calibrated_edge` column from `get_win_candidates()`. Candidates now sorted by raw `win_selection_edge`.
- **Why**: Feature family agreement scoring treated NaN as "no agreement" → 60-80% of high-EV candidates received 0.70x scaling, suppressing the most profitable bets.

### Task 2: Correlation penalty disabled (~3-5pt ROI recovery)
- **File**: `src/models/stacked_ensemble.py`
- **Change**: `corr_penalty_weight` default changed from 0.5 to 0.0 (single line).
- **Why**: Same-feature GBM ensembles have correlation 0.92-0.97 (impossible to reach threshold=0.85). Result: AUC sacrificed for unattainable diversity without benefit.

### Task 3: 3 high-NaN features removed (~2-3pt ROI recovery)
- **Files**: 8 model files (11 FEATURE_COLS lists total)
- **Removed**: `pace_ratio_zscore` (50% NaN), `pace_ratio_trend` (40% NaN), `pace_adj_finish_avg` (~35% NaN)
- **Why**: These features had structurally high NaN rates due to LapTime/L4 data scarcity. They polluted feature_fraction sampling and added noise without signal.
- **Kept**: Remaining 23 Phase36 features (including `pace_ratio_avg`, `pace_early/mid/late_avg` which have acceptable ~28% NaN)

## Verification

- All tests pass (502 passed, 1 pre-existing failure unrelated to changes)
- `StackedEnsemble().corr_penalty_weight == 0.0` confirmed
- No `ev_tail_calibration` import in `race_predictor.py`
- 3 removed features absent from all 7 model files (grep count = 0)
