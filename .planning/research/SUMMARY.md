# Research Summary: keiba-ai v2.1 MarketAware Calibration + Race-Level Ranker

**Project:** keiba-ai v2.1 MarketAware Calibration + Race-Level Ranker for ROI Recovery
**Domain:** Horse racing ML prediction system — calibration and ranking for ROI recovery
**Researched:** 2026-05-27
**Overall confidence:** HIGH

## Executive Summary

v2.1 aims to recover BT ROI from 87.8% (v2.0) to 100%+, using the v2.0 foundation (OOFHealthValidator + InvestmentFeatureFrame). The recovery centers on MarketAwareWinCalibrator, which replaces (not augments) WinBenterGate + WinSegmentCalibrator with Benter (1994) logit-blend methodology, producing segment-conditioned probabilities with per-segment alpha/beta/gamma fitted via MLE. A deterministic Race-Level Ranker (Optuna-tuned weighted sum) replaces the hand-tuned win_market_selection_score formula. A shadow comparison framework validates the new pipeline against baseline before deployment.

The critical architecture decision: MarketAwareWinCalibrator must **absorb** both WinBenterGate and WinSegmentCalibrator roles to prevent double-correction — the #1 pitfall identified across all 4 research dimensions. Appending a new calibrator on top of the existing 5-point calibration chain would compress probabilities to zero marginal information.

No new pip dependencies required. All 4 features implementable with existing scipy/sklearn/pandas/numpy.

## Key Findings

**Stack:** Zero new dependencies. MarketAwareWinCalibrator extends existing BenterCombination (scipy L-BFGS-B). Race-Level Ranker uses pandas groupby.rank() + numpy. Shadow comparison uses scipy.stats (ks_2samp, mannwhitneyu) + sklearn.metrics.

**Architecture:** Replace-not-append pattern. MarketAwareWinCalibrator replaces WinBenterGate + WinSegmentCalibrator at the same pipeline position in RacePredictor.predict(). Race-Level Ranker provides deterministic composite score to WinSelectionGate. Shadow comparison wraps BacktestEngine without modifying it.

**Critical pitfall #1 — Double correction:** Pipeline already has 5 calibration points. Adding another layer compresses probabilities. New calibrator must absorb existing roles.

**Critical pitfall #2 — Feature routing:** v1.8 ROI collapse (97.8%→87.8%) caused by uniformly registering strong features across all models, degrading specialized models (MarketModel, RaceQualityScreener). Surgical routing audit required.

**Critical pitfall #3 — Normalization breaks calibration:** WinBenterGate.apply() normalizes AFTER calibration, decoupling output probabilities from calibration mapping. New calibrator must train on normalized probabilities or natively produce sum-to-1.0.

**Benter paper insight:** Combined model delta-R2 (combined minus public-only) determines profitability, not absolute R2. Markets are getting more efficient, so the fundamental model must capture information NOT already in market prices.

## Implications for Roadmap

Suggested phase structure:

1. **Segment Conditioning + MarketAwareWinCalibrator** — Core new model, replaces WinBenterGate + WinSegmentCalibrator. Per-segment alpha/beta/gamma via MLE. Segment keys from InvestmentFeatureFrame (if_popularity_rank, if_odds_band_id, if_p_win_race_rank). Highest-risk phase.

2. **Race-Level Ranker** — Deterministic weighted sum: `score = w_p*p_win + w_edge*edge + w_ev*ev_corrected` with Optuna-tuned weights (extending 16-dim to 19-dim parameter space). NOT learned LGBMRanker (5-10 features with ~15 items per group would overfit).

3. **Shadow Comparison + Deployment Gate** — Runs BacktestEngine twice (baseline vs shadow) on 2024+2025. Deployment gate requires ALL FIVE conditions on BOTH years: Brier improvement, ECE improvement, selection agreement >= 85%, ROI stability, bet-count preservation.

4. **Feature Routing Audit + Integration Testing** — Safety net against v1.8 collapse. Verify calibrator features are NOT registered in MarketModel or RaceQualityScreener.

**Research flags for phases:**
- Phase 1: Benter segment conditioning is novel; segment boundary optimization needs empirical validation
- Phase 2: Optuna weight initialization for deterministic ranker; integration boundary with WinSelectionGate
- Phase 3: Standard comparison framework; compute cost doubles training time (~34min to ~60min)
- Phase 4: Standard audit; requires careful review of all model FEATURE_COLS

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new dependencies; all APIs verified |
| Features | HIGH | Benter (1994) theoretical foundation; InvestmentFeatureFrame has 94 specs ready |
| Architecture | HIGH | Replace-not-append pattern; all integration points verified against source |
| Pitfalls | HIGH | v1.8 ROI collapse provides project-specific evidence; 5 critical pitfalls identified |

## Gaps to Address

- **Segment boundary optimization:** Current WSC boundaries (1-2, 2-5, 5-10, 10-30, 30-100, 100+) may not be optimal for Benter blend. Continuous odds treatment needs empirical comparison.
- **Ranker-gate integration boundary:** How ranker output feeds into WinSelectionGate (1282 lines of evolved logic) without breaking runner-up detection, soft-pass, and market condition scoring.
- **Calibration ordering:** Win odds normalization should occur BEFORE or AFTER calibration? Current pipeline normalizes after, which breaks calibration. New calibrator must address this.
- **OOF health after calibrator:** New calibrator changes probability distributions — EV_lower and OddsBandFilter may need recalibration (v1.4 pattern).

---
*Research completed: 2026-05-27*
*Ready for roadmap: yes*
