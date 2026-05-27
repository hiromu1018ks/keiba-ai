# Research Summary: keiba-ai v2.0 Investment Pipeline Restructuring

**Project:** keiba-ai v2.0 Investment Pipeline Restructuring
**Domain:** Horse racing ML prediction system -- investment pipeline restructuring
**Researched:** 2026-05-27
**Overall confidence:** HIGH

## Executive Summary

keiba-ai v2.0 restructures the post-model investment decision pipeline with four new components: OOF Health validation, InvestmentFeatureFrame, MarketAwareWinCalibrator (Benter-type logit blending), and Race-Level Ranker (LightGBM LambdaRank). The current BT ROI is 87.8% (Phase #33); the target is 100%+. The restructuring focuses on how calibrated probabilities are combined with market information, how features are curated for investment decisions, and how horses are ranked within races.

The critical insight from Benter (1994) is that fundamental model probabilities are systematically biased relative to market prices -- when the market estimates higher probability than the model, actual win frequencies are much higher than the model predicts (and vice versa). The Benter logit-blend formula removes this bias completely. The existing BenterCombination class already implements this with MLE fitting, but needs extension for segment conditioning and OOF-based validation.

No new external dependencies are required. All components use the existing installed stack (LightGBM 4.6.0 with LGBMRanker, scikit-learn 1.8.0, scipy 1.17.1, betacal 1.1.0). The segment calibration integration (Phase 3 in the original plan) is absorbed into MarketAwareWinCalibrator as conditioning features rather than remaining a standalone model.

## Key Findings

**Stack:** Zero new dependencies. All 4 components implementable with currently installed packages.

**Architecture:** Extend-not-replace pattern. MarketAwareWinCalibrator extends existing BenterCombination. Race-Level Ranker provides ranking signal to existing WinSelectionGate (does not replace it). InvestmentFeatureFrame curates existing features (does not re-engineer them).

**Critical pitfall:** Calibrator trained on in-sample predictions (not OOF). If MarketAwareWinCalibrator parameters are fit on in-sample predictions, the calibrator overfits and backtest ROI is inflated while forward testing fails.

**Benter paper insight:** Combined model pseudo-R2 is not the key metric -- delta-R2 (combined minus public-only) determines profitability. Markets are getting more efficient (public-only R2 rose from 0.1325 to 0.1863 over 30 years), so the fundamental model must capture information NOT already in market prices.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Phase 0: OOF Health Infrastructure** - Foundation for all downstream components
   - Addresses: Reliable held-out predictions for calibrator fitting and ranker training
   - Avoids: OOF leakage that would invalidate all downstream components
   - Effort: Medium

2. **Phase 1: InvestmentFeatureFrame** - Curate existing features into formal frame
   - Addresses: Consistent feature access for calibrator and ranker
   - Avoids: Ad-hoc column selection causing missing-feature bugs
   - Effort: Low-Medium

3. **Phase 2: MarketAwareWinCalibrator** - Benter blend + segment conditioning
   - Addresses: Systematic model-vs-market bias removal
   - Avoids: Double-correction by absorbing WSC (not appending)
   - Depends on: Phase 0 (OOF for fitting), Phase 1 (feature frame)
   - Effort: High

4. **Phase 3: Race-Level Ranker** - LightGBM LambdaRank for within-race ranking
   - Addresses: Learned ranking replacing heuristic gate scoring
   - Avoids: Replacing entire WinSelectionGate (ranker provides signal, gate makes final selection)
   - Depends on: Phase 1 (feature frame), Phase 2 (calibrated probabilities)
   - Effort: Medium-High

5. **Phase 4: Integrated Validation** - Full pipeline validation with walk-forward testing
   - Addresses: End-to-end ROI measurement and overfitting detection
   - Avoids: Component-level testing only (integration bugs hide at boundaries)
   - Depends on: All phases complete
   - Effort: Medium (mainly compute time)

**Phase ordering rationale:**
- OOF Health is the critical path bottleneck -- calibrator fitting and ranker training both require reliable OOF predictions
- InvestmentFeatureFrame must precede both calibrator and ranker because they consume its output
- MarketAwareWinCalibrator precedes Ranker because the ranker benefits from calibrated probabilities as input features
- Segment calibration (original Phase 3) is absorbed into Phase 2 rather than standing alone

**Research flags for phases:**
- Phase 0: Standard OOF generation -- no deep research needed
- Phase 1: Feature curation -- standard data engineering, no research needed
- Phase 2: Benter segment conditioning is novel; may need empirical validation of segment boundaries and shrinkage strength
- Phase 3: LambdaRank group handling requires careful data preparation; well-documented but easy to get wrong
- Phase 4: Walk-forward validation design -- standard but computationally expensive

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new dependencies; all APIs verified via Context7 and pip show |
| Features | HIGH | Benter paper provides theoretical foundation; existing code provides implementation base |
| Architecture | HIGH | Extend-not-replace pattern; all integration points verified against source code |
| Pitfalls | HIGH | OOF leakage is the critical risk; well-understood mitigation (use OOF for all fitting) |

## Gaps to Address

- **Benter segment boundary optimization:** The optimal segment boundaries for odds-band and EV-band conditioning are unknown. The current WSC boundaries (1-2, 2-5, 5-10, 10-30, 30-100, 100+) may not be optimal for the Benter blend.
- **LambdaRank relevance grading:** Binary (win/loss) vs position-graded (1/finish_position) relevance labels need empirical comparison for 8-18 horse fields.
- **Ranker-gate integration boundary:** How exactly the ranker output feeds into WinSelectionGate final selection needs design work.

---
*Research completed: 2026-05-27*
*Ready for roadmap: yes*
