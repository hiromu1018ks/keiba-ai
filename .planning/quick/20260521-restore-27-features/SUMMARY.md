---
status: complete
slug: restore-27-features
date: 2026-05-21
commit: 5123774
---

# Restore 26 Phase36 Features to MarketModel and RegimeDetector

## What
Restored 26 Phase36 features (form/blood rank, interaction, closing speed, harontime, haron gap, pace) to `MarketModel.FEATURE_COLS` and `RegimeDetector.FEATURE_COLS` that were removed in commit 672e283.

## Changes
- `src/models/market_model.py` — FEATURE_COLS: 20 → 46
- `src/models/regime_detector.py` — FEATURE_COLS: 21 → 47
- `src/models/gpd_diagnostics.py` — Added 4 missing features to FEATURE_CATEGORY_MAP (haron_race_gap_*, pace_adj_finish_avg)
- 7 test files updated to reflect restored features and fix pre-existing test gaps
