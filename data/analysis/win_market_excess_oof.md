# Market Excess OOF Lightweight Experiment

Generated: 2026-06-02T20:23:11.416730

## 1. Conclusion

**Verdict: C: dangerous (few high-odds hits)**

> **Note:** This is a lightweight out-of-sample verification (2024→2025),
> not a strict production OOF. Features include existing pipeline outputs
> (p_win_final, edge_ratio, etc.) which already contain market-diff/MAWC/correction.
> Good results do not guarantee independent new learning.

## 2. Target Variable

`market_excess_diff = is_win - p_market_win_norm`

| Component | Description |
|-----------|-------------|
| is_win | `(kakuteijyuni == 1)` — actual win (1) or loss (0) |
| p_market_win_norm | Market-implied probability, normalized within race |
| market_excess_diff | Positive = beat market, Negative = fell short |

## 3. Data

- Train: 2024 (45,437 rows)
- Test: 2025 (46,160 rows)
- Features: 23 dimensions
- Model: lightgbm (MAE objective)
- Train MAE: 0.0725
- Train correlation: 0.0036

## 4. Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | p_market_win_norm | 1722.0 |
| 2 | tanodds | 1315.0 |
| 3 | p_win_final | 627.0 |
| 4 | overround | 577.0 |
| 5 | p_win_corrected | 419.0 |
| 6 | edge_diff_pred | 419.0 |
| 7 | edge_diff_final | 273.0 |
| 8 | p_win_pred | 209.0 |
| 9 | win_market_value_ratio | 146.0 |
| 10 | edge_ratio_final | 141.0 |

## 5. market_excess_pred Quantile ROI (Test: 2025)

| Quantile | N | Hit% | ROI(payout) | ROI(tanodds) | Avg CO | Avg Excess |
|----------|---|------|-------------|--------------|--------|------------|
| 0-20% | 9232 | 21.6% | 78.3% | 83.0% | 4.3 | +0.0059 |
| 20-40% | 9232 | 7.9% | 75.6% | 78.7% | 11.4 | -0.0014 |
| 40-60% | 9232 | 3.7% | 85.8% | 84.3% | 27.9 | +0.0019 |
| 60-80% | 9232 | 1.3% | 69.4% | 64.6% | 72.7 | -0.0032 |
| 80-95% | 6924 | 0.5% | 56.5% | 50.9% | 190.6 | -0.0021 |
| 95-99% | 1846 | 0.1% | 34.0% | 22.4% | 375.3 | -0.0023 |
| 99-100% | 462 | 26.2% | 35.6% | 34.2% | 285.2 | -0.0162 |

> ROI(payout) = confirmed_odds-based (primary). ROI(tanodds) = reference.

## 6. Baseline Comparison (Test: 2025, Top 5%)

| Metric | N | ROI(payout) | ROI(tanodds) | Hit% | Avg CO |
|--------|---|-------------|--------------|------|--------|
| market_excess_pred | 2308 | 34.3% | 24.8% | 5.3% | 357.2 |
| edge_ratio_final | 2308 | 66.9% | 67.1% | 7.3% | 19.9 |
| pred_ev_final | 2308 | 66.4% | 67.6% | 8.1% | 19.2 |
| ev_win | 2308 | 100.2% | 103.4% | 6.6% | 57.4 |
| win_market_value_ratio | 2308 | 66.4% | 67.6% | 8.1% | 19.2 |
| win_market_logit_edge | 2308 | 64.1% | 65.3% | 14.5% | 15.3 |

## 7. Correlation Analysis (Test: 2025)

| Pair | Correlation |
|------|------------|
| pred_vs_actual_excess | -0.0094 |
| pred_vs_is_win | -0.2909 |
| pred_vs_p_market | -0.7629 |
| pred_vs_tanodds | 0.4862 ⚠️ |
| pred_vs_ev_win | -0.2283 |
| pred_vs_edge_ratio_final | -0.4934 |

## 8. Year × Surface Breakdown (Top 10%)

| Group | N_total | N_top10 | ROI(payout) | ROI(tanodds) | Hit% | Avg CO |
|-------|---------|---------|-------------|--------------|------|--------|
| 2025_dirt | 23491 | 2350 | 32.4% | 25.3% | 2.5% | 312.9 |
| 2025_turf | 22669 | 2267 | 68.3% | 48.8% | 3.1% | 294.3 |

## 9. Sanity Check

- **top_improves**: ❌ FAIL — top=0.356 vs mid=0.694
- **not_few_high_odds**: ❌ FAIL — count=462, avg_odds=285.2
- **not_copying_market**: ✅ PASS — corr_p_market=0.763
- **not_copying_odds**: ✅ PASS — corr_tanodds=0.486
- **beats_baselines**: ❌ FAIL — excess_top5=0.343 vs ev_win=1.002

**Overall: SOME FAILURES ❌**

## 10. Next Steps

- Results depend on few high-odds hits — statistically unreliable
- Need more data or different approach

## 11. Execution Command

```bash
python scripts/analyze_win_market_excess_oof.py
```

## 12. Generated Files

- `data/analysis/win_market_excess_oof.json`
- `data/analysis/win_market_excess_oof.md`
