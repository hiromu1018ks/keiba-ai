---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: MarketAware Calibration + Race-Level Ranker for ROI Recovery
status: planning
last_updated: "2026-05-28T12:00:00.000Z"
---

# Roadmap: keiba-ai Win Model Improvement

## Milestones

- **v1.0 Win Model** -- Phases 1-4 (shipped 2026-05-03)
- **v1.1 ROI Advanced Model** -- Phases 5-7 (shipped 2026-05-03)
- **v1.2 Win Backtest Validation** -- Phases 8-10 (shipped 2026-05-04)
- **v1.3 Betting Strategy Optimization** -- Phases 11-13 (shipped 2026-05-05)
- **v1.4 Ensemble Filter Recalibration** -- Phases 14-18 (shipped 2026-05-07)
- **v1.5 Model Accuracy Improvement** -- Phases 19-22 (shipped 2026-05-10)
- **v1.6 Feature Engineering Overhaul** -- Phases 23-28 (shipped 2026-05-17)
- **v1.7 Market-Independent Edge Discovery** -- Phases 29-34 (shipped 2026-05-19)
- **v1.8 Turf Precision Calibration** -- Phases 35-36.1.1 (shipped 2026-05-20)
- **v2.0 Investment Pipeline Restructuring** -- Phases 37-38 (shipped 2026-05-27)
- **v2.1 MarketAware Calibration + Race-Level Ranker** -- Phases 39-42 (in progress)

## Phases

**Phase Numbering:**
- Integer phases: Planned milestone work
- Decimal phases: Urgent insertions (marked with INSERTED)

<details>
<summary>v1.0-v2.0 (Phases 1-38) — All Shipped</summary>

Phases 1-38 complete across milestones v1.0 through v2.0.
See `.planning/milestones/` for archived roadmaps.

</details>

### v2.1 MarketAware Calibration + Race-Level Ranker (In Progress)

**Milestone Goal:** Recover BT ROI from 87.8% to 100%+ via MarketAwareWinCalibrator (replacing WinBenterGate + WinSegmentCalibrator) and learned Race-Level Ranker, validated by shadow comparison against baseline.

- [x] **Phase 39: MarketAwareWinCalibrator** - Replace WinBenterGate + WinSegmentCalibrator with Benter logit-blend calibrator producing segment-conditioned probabilities (completed 2026-05-27)
- [x] **Phase 40: Race-Level Ranker** - Learned ranker combining relevance (is_win/finishing-position) and value/mispricing signals into investment_score (completed 2026-05-28)
- [x] **Phase 41: Shadow Comparison Framework** - Fixed-fold 2024/2025 baseline vs shadow comparison tracking probability quality, selection agreement, CLV, ROI (completed 2026-05-28)
- [ ] **Phase 42: Feature Routing Audit & Safety Gates** - Verify calibrator features do not pollute MarketModel/RaceQualityScreener, OOF health passes, deployment gate conditions met

## Phase Details

### Phase 39: MarketAwareWinCalibrator
**Goal**: Win probabilities are produced by a single MarketAwareWinCalibrator that blends model and market logits with segment-conditioned regularization, replacing the previous dual WinBenterGate + WinSegmentCalibrator chain and preventing double-correction
**Depends on**: Phase 38 (InvestmentFeatureFrame provides segment keys)
**Requirements**: CAL-01, CAL-02, CAL-03, CAL-04, CAL-05
**Success Criteria** (what must be TRUE):
  1. MarketAwareWinCalibrator produces calibrated win probabilities via Benter logit(p_model) + logit(p_market) blend, absorbing both WinBenterGate and WinSegmentCalibrator roles at the same pipeline position in RacePredictor.predict()
  2. Segment conditioning uses popularity rank, odds band, and probability rank from InvestmentFeatureFrame output as regularized features/interactions in a global calibrator (not per-segment coefficients), preventing sparse segment overfitting
  3. Calibrator output maintains probability quality (Brier, logloss, ECE) after normalization and satisfies sum-to-1.0 constraint per race
  4. WinBenterGate and WinSegmentCalibrator are removed from the pipeline with no remaining call sites
**Plans**: 3 plans

Plans:
- [x] 39-01-PLAN.md -- MarketAwareWinCalibrator class with feature encoding, training, inference, guards
- [x] 39-02-PLAN.md -- TrainingPipeline integration + SubmodelSet field update
- [x] 39-03-PLAN.md -- RacePredictor + ModelLoader integration (remove old components)

### Phase 40: Race-Level Ranker
**Goal**: A learned ranker orders horses within each race by combining relevance (win/finishing-position) and value/mispricing signals, producing an investment_score that replaces hand-tuned formulas
**Depends on**: Phase 39 (calibrated probabilities feed the ranker)
**Requirements**: RNK-01, RNK-02, RNK-03, RNK-04, RNK-05
**Success Criteria** (what must be TRUE):
  1. A learned Win relevance ranker orders horses within each race using is_win and finishing-position relevance signals
  2. A learned value/mispricing ranker detects mispriced horses using calibrated EV, model-vs-market gap, and CLV diagnostics (OOF-safe)
  3. Win ranker and Value ranker outputs are combined into a single investment_score per horse
  4. The ranker operates in shadow mode behind a feature flag, with baseline WinSelectionGate preserved and functional
  5. One-bet-per-race baseline bet count is maintained without explicit approval to reduce it
**Plans**: 3 plans

Plans:
- [x] 40-01-PLAN.md -- RaceLevelRanker class with Ridge training, scoring, shadow mode, persistence (Wave 1)
- [x] 40-02-PLAN.md -- OOF extension + TrainingPipeline ranker integration (Wave 2, depends on 40-01)
- [x] 40-03-PLAN.md -- RacePredictor + ModelLoader integration (Wave 3, depends on 40-02)

### Phase 41: Shadow Comparison Framework
**Goal**: The shadow pipeline (new calibrator + ranker) can be compared against baseline on 2024/2025 test periods with comprehensive metrics, enabling data-driven deployment decisions
**Depends on**: Phase 39, Phase 40
**Requirements**: SHD-01, SHD-02, SHD-03
**Success Criteria** (what must be TRUE):
  1. Shadow comparison runs BacktestEngine twice (baseline TrainedModelsV5 vs shadow TrainedModelsV5) on both 2024 and 2025 test periods with fixed folds
  2. Comparison tracks Brier, logloss, ECE, selection agreement, CLV, ROI, HR, DD, and bet count for both baseline and shadow
  3. Selection horse differences between baseline and shadow (selection agreement) are measured and explainable per-race
**Plans**: 2 plans

Plans:
- [x] 41-01-PLAN.md -- Core framework, dataclasses, feature flag injection, alignment, metrics engine (Wave 1)
- [x] 41-02-PLAN.md -- Output artifacts, HTML report, CLI script (Wave 2, depends on 41-01)

### Phase 42: Feature Routing Audit & Safety Gates
**Goal**: All safety checks pass -- calibrator features do not leak into MarketModel/RaceQualityScreener, OOF health is clean, and the new pipeline only replaces baseline after meeting all quality gates
**Depends on**: Phase 39, Phase 40, Phase 41
**Requirements**: SAF-01, SAF-02, SAF-03
**Success Criteria** (what must be TRUE):
  1. Feature routing audit confirms calibrator features (segment conditioning inputs) are NOT registered in MarketModel or RaceQualityScreener FEATURE_COLS
  2. OOFHealthValidator reports no anomalies when the full pipeline (new calibrator + ranker) runs end-to-end
  3. The new calibrator/ranker does NOT replace baseline until probability quality gates + bet-count preservation + artifact reproducibility + all diagnostics pass
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 39 → 40 → 41 → 42

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-38 | v1.0-v2.0 | 84/84 | Complete | 2026-05-27 |
| 39. MarketAwareWinCalibrator | v2.1 | 3/3 | Complete    | 2026-05-27 |
| 40. Race-Level Ranker | v2.1 | 3/3 | Complete | 2026-05-28 |
| 41. Shadow Comparison | v2.1 | 2/2 | Complete    | 2026-05-28 |
| 42. Feature Routing Audit & Safety | v2.1 | 0/? | Not started | - |
