---
gsd_state_version: 1.0
milestone: v1.8
milestone_name: Turf Precision Calibration
status: planning
last_updated: "2026-05-19T15:00:00Z"
progress:
  total_phases: 38
  completed_phases: 34
  total_plans: 80
  completed_plans: 80
  percent: 89
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
- **v1.8 Turf Precision Calibration** -- Phases 35-38 (in progress)

## Phases

<details>
<summary>v1.0 Win Model (Phases 1-4) -- SHIPPED 2026-05-03</summary>

- [x] Phase 1: Feature Analysis & Enhancement (2/2 plans) -- completed 2026-05-02
- [x] Phase 2: Win Benter Combination & Calibration (2/2 plans) -- completed 2026-05-02
- [x] Phase 3: Selection Gate, Confidence & Betting (2/2 plans) -- completed 2026-05-02
- [x] Phase 4: Walk-Forward Validation (1/1 plan) -- completed 2026-05-03

</details>

<details>
<summary>v1.1 ROI Advanced Model (Phases 5-7) -- SHIPPED 2026-05-03</summary>

- [x] Phase 5: Foundation Features (2/2 plans) -- completed 2026-05-03
- [x] Phase 6: Odds Deviation EV (1/1 plan) -- completed 2026-05-03
- [x] Phase 7: Ensemble Enhancement (1/1 plan) -- completed 2026-05-03

</details>

<details>
<summary>v1.2 Win Backtest Validation (Phases 8-10) -- SHIPPED 2026-05-04</summary>

- [x] Phase 8: Win Backtest Core (2/2 plans) -- completed 2026-05-04
- [x] Phase 9: Win Reporting (1/1 plan) -- completed 2026-05-04
- [x] Phase 10: Pipeline Performance (2/2 plans) -- completed 2026-05-04

</details>

<details>
<summary>v1.3 Betting Strategy Optimization (Phases 11-13) -- SHIPPED 2026-05-05</summary>

- [x] Phase 11: Bet Selection Filters (2/2 plans) -- completed 2026-05-04
- [x] Phase 12: Stake Sizing Enhancement (2/2 plans) -- completed 2026-05-05
- [x] Phase 13: Risk Calibration & Parameter Optimization (3/3 plans) -- completed 2026-05-05

</details>

<details>
<summary>v1.4 Ensemble Filter Recalibration (Phases 14-18) -- SHIPPED 2026-05-07</summary>

- [x] Phase 14: Gate Recalibration (2/2 plans) -- completed 2026-05-06
- [x] Phase 15: EV Filter Enhancement (2/2 plans) -- completed 2026-05-06
- [x] Phase 16: Odds Band Rebuild (2/2 plans) -- completed 2026-05-06
- [x] Phase 17: Optuna Optimization (2/2 plans) -- completed 2026-05-06
- [x] Phase 18: Validation & Freeze (2/2 plans) -- completed 2026-05-07

</details>

<details>
<summary>v1.5 Model Accuracy Improvement (Phases 19-22) -- SHIPPED 2026-05-10</summary>

- [x] Phase 19: EV推定キャリブレーション (2/2 plans) -- completed 2026-05-07
- [x] Phase 19.1: バックテスト高速化 (5/5 plans) -- completed 2026-05-08
- [x] Phase 20: 高オッズ的中パターン特徴量 (3/3 plans) -- completed 2026-05-09
- [x] Phase 21: Conformal EV予測区間 (2/2 plans) -- completed 2026-05-09
- [x] Phase 22: 統合検証とバックテスト (1/1 plan) -- completed 2026-05-10

</details>

<details>
<summary>v1.6 Feature Engineering Overhaul (Phases 23-28) -- SHIPPED 2026-05-17</summary>

- [x] Phase 23: Safety Gate (2/2 plans) -- completed 2026-05-11
- [x] Phase 24: Feature Audit & Pruning (2/2 plans) -- completed 2026-05-12
- [x] Phase 25: Quick Win Wire Existing (2/2 plans) -- completed 2026-05-12
- [x] Phase 26: EveryDB2 New Features (3/3 plans) -- completed 2026-05-14
- [x] Phase 27: Feature Interactions (3/3 plans) -- completed 2026-05-15
- [x] Phase 28: Validation & Freeze (2/2 plans) -- completed 2026-05-17

</details>

<details>
<summary>v1.7 Market-Independent Edge Discovery (Phases 29-34) -- SHIPPED 2026-05-19</summary>

- [x] Phase 29: ETL Expansion (3/3 plans) -- completed 2026-05-17
- [x] Phase 30: Residual IC Evaluation Framework (2/2 plans) -- completed 2026-05-18
- [x] Phase 31: Race-Level Aggregation Features (2/2 plans) -- completed 2026-05-18
- [x] Phase 32: Market Cross-Consistency Features (2/2 plans) -- completed 2026-05-18
- [x] Phase 33: Gain per Depth Diagnostic (2/2 plans) -- completed 2026-05-18
- [x] Phase 34: Validation and Manifest Update (4/4 plans) -- completed 2026-05-19

</details>

### v1.8 Turf Precision Calibration (In Progress)

**Milestone Goal:** IC b_difference --> positive, ROI 97.8%-->100%+

- [x] **Phase 35: ETL Data Foundation** - HaronTime/LapTime/Jyuni float64 + POST_RACE_COLS (completed 2026-05-19)
- [ ] **Phase 36: Feature Computation** - Turf relative features + conditional interactions + Haron/Lap PIT-safe
- [ ] **Phase 37: EV Calibration Layers** - Pop band calibration + regime x surface EV correction
- [ ] **Phase 38: Integrated Validation** - CI leakage tests + IC confirmation + BT ROI 100% + Manifest freeze

## Phase Details

### Phase 35: ETL Data Foundation
**Goal**: HaronTimeL3/L4, LapTime1~25, Jyuni1c~4c are available as float64 in Parquet with sentinel values handled and POST_RACE safety enforced
**Depends on**: Nothing (first phase of v1.8)
**Requirements**: ETL-01, ETL-02, ETL-03, ETL-04, ETL-05
**Success Criteria** (what must be TRUE):
  1. entries.parquet contains HaronTimeL3/L4 columns as float64 with 000/999 sentinel values replaced by NaN
  2. races.parquet contains LapTime1~25 columns as float64 with 000 sentinel values replaced by NaN
  3. entries.parquet contains Jyuni1c~4c corner position columns as numeric values
  4. All new POST_RACE columns appear in domain/types.py POST_RACE_COLS and 3-layer CI leakage tests detect any misuse
  5. HaronTimeL3/L4 mutual exclusivity is validated and coalescing logic (harontime_last3f) is documented
**Plans**: 2 plans
Plans:
- [x] 35-01-PLAN.md -- sentinel_float/sentinel_int rules + _TABLE_TYPE_RULES update + readers.py
- [x] 35-02-PLAN.md -- POST_RACE_COLS consolidation + HaronTime mutual exclusivity analysis

### Phase 36: Feature Computation
**Goal**: All new turf-focused features (relative ranks, conditional interactions, haron/lap history) are computed PIT-safe and registered across all 12 models
**Depends on**: Phase 35
**Requirements**: TRF-01, TRF-02, TRF-03, INT-01, INT-02, INT-03, INT-04, HLF-01, HLF-02, HLF-03, HLF-04, HLF-05
**Success Criteria** (what must be TRUE):
  1. form_trend_race_rank, blood_total_wr_race_rank, blood_surface_wr_race_rank, and weighted_recent_form appear in training data via add_race_transforms()
  2. grade_x_form_trend, distance_x_closing_index, grade_x_blood_prize_log interaction features appear in interaction_features.py output
  3. Haron/Lap history features (harontimel4_avg, harontimel4_zscore, haron_l3l4_ratio, lap pace ratios) are computed using expanding_stats + searchsorted (PIT-safe, no current-race data)
  4. All HLF/TRF/INT features appear in all 12 models' FEATURE_COLS lists
  5. Both training pipeline (_train_submodel) and inference path (BettingOrchestrator.build_features) produce identical feature sets
**Plans**: TBD

### Phase 37: EV Calibration Layers
**Goal**: Popularity band calibration and regime-surface EV correction improve EV accuracy for turf middle-popularity horses
**Depends on**: Phase 36
**Requirements**: CAL-01, CAL-02, CAL-03, CAL-04, CAL-05
**Success Criteria** (what must be TRUE):
  1. Popularity band calibration (5 bands: 1-3, 4-6, 7-9, 10-12, 13+) produces per-band EV scaling factors computed from OOF residuals with extended-window OOF to prevent look-ahead bias
  2. EVCorrectionModel.FEATURE_COLS includes regime_state, surface_x_popularity, and market_entropy_x_surface columns
  3. regime_state propagates from RacePredictor through to EVCorrectionModel without manual intervention during backtest
  4. Regime-EV feedback loop forced-transition test passes (regime state cannot be gamed by EV outputs)
**Plans**: TBD

### Phase 38: Integrated Validation
**Goal**: All v1.8 features and calibration layers are validated safe, turf IC is positive, and BT ROI exceeds 100%
**Depends on**: Phase 37
**Requirements**: VAL-01, VAL-02, VAL-03, VAL-04, VAL-05, VAL-06
**Success Criteria** (what must be TRUE):
  1. 3-layer CI leakage tests cover all new HLF/TRF/INT feature columns and all pass
  2. Turf Stage1 IC b_difference is positive (was -0.004 in v1.7)
  3. Turf pop 4-12 calibration ratio improves from 0.527
  4. BT 2024 ROI exceeds 100% (was 97.8% in v1.7)
  5. Turf conservative regime ROI improves (currently unprofitable)
  6. Manifest v1.8 is frozen with SHA256 feature hashes for all 12 models
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 35 -> 36 -> 37 -> 38

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Feature Analysis & Enhancement | v1.0 | 2/2 | Complete | 2026-05-02 |
| 2. Win Benter Combination & Calibration | v1.0 | 2/2 | Complete | 2026-05-02 |
| 3. Selection Gate, Confidence & Betting | v1.0 | 2/2 | Complete | 2026-05-02 |
| 4. Walk-Forward Validation | v1.0 | 1/1 | Complete | 2026-05-03 |
| 5. Foundation Features | v1.1 | 2/2 | Complete | 2026-05-03 |
| 6. Odds Deviation EV | v1.1 | 1/1 | Complete | 2026-05-03 |
| 7. Ensemble Enhancement | v1.1 | 1/1 | Complete | 2026-05-03 |
| 8. Win Backtest Core | v1.2 | 2/2 | Complete | 2026-05-04 |
| 9. Win Reporting | v1.2 | 1/1 | Complete | 2026-05-04 |
| 10. Pipeline Performance | v1.2 | 2/2 | Complete | 2026-05-04 |
| 11. Bet Selection Filters | v1.3 | 2/2 | Complete | 2026-05-04 |
| 12. Stake Sizing Enhancement | v1.3 | 2/2 | Complete | 2026-05-05 |
| 13. Risk Calibration & Parameter Optimization | v1.3 | 3/3 | Complete | 2026-05-05 |
| 14. Gate Recalibration | v1.4 | 2/2 | Complete | 2026-05-06 |
| 15. EV Filter Enhancement | v1.4 | 2/2 | Complete | 2026-05-06 |
| 16. Odds Band Rebuild | v1.4 | 2/2 | Complete | 2026-05-06 |
| 17. Optuna Optimization | v1.4 | 2/2 | Complete | 2026-05-06 |
| 18. Validation & Freeze | v1.4 | 2/2 | Complete | 2026-05-07 |
| 19. EV推定キャリブレーション | v1.5 | 2/2 | Complete | 2026-05-07 |
| 19.1. バックテスト高速化 | v1.5 | 5/5 | Complete | 2026-05-08 |
| 20. 高オッズ的中パターン特徴量 | v1.5 | 3/3 | Complete | 2026-05-09 |
| 21. Conformal EV予測区間 | v1.5 | 2/2 | Complete | 2026-05-09 |
| 22. 統合検証とバックテスト | v1.5 | 1/1 | Complete | 2026-05-10 |
| 23. Safety Gate | v1.6 | 2/2 | Complete | 2026-05-11 |
| 24. Feature Audit & Pruning | v1.6 | 2/2 | Complete | 2026-05-12 |
| 25. Quick Win Wire Existing | v1.6 | 2/2 | Complete | 2026-05-12 |
| 26. EveryDB2 New Features | v1.6 | 3/3 | Complete | 2026-05-14 |
| 27. Feature Interactions | v1.6 | 3/3 | Complete | 2026-05-15 |
| 28. Validation & Freeze | v1.6 | 2/2 | Complete | 2026-05-17 |
| 29. ETL Expansion | v1.7 | 3/3 | Complete | 2026-05-17 |
| 30. Residual IC Evaluation Framework | v1.7 | 2/2 | Complete | 2026-05-18 |
| 31. Race-Level Aggregation Features | v1.7 | 2/2 | Complete | 2026-05-18 |
| 32. Market Cross-Consistency Features | v1.7 | 2/2 | Complete | 2026-05-18 |
| 33. Gain per Depth Diagnostic | v1.7 | 2/2 | Complete | 2026-05-18 |
| 34. Validation and Manifest Update | v1.7 | 4/4 | Complete | 2026-05-19 |
| 35. ETL Data Foundation | v1.8 | 2/2 | Complete   | 2026-05-19 |
| 36. Feature Computation | v1.8 | 0/? | Not started | - |
| 37. EV Calibration Layers | v1.8 | 0/? | Not started | - |
| 38. Integrated Validation | v1.8 | 0/? | Not started | - |
