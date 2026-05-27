---
gsd_state_version: 1.0
milestone: v1.8
milestone_name: Turf Precision Calibration
status: planning
last_updated: "2026-05-20T17:30:00Z"
progress:
  total_phases: 40
  completed_phases: 34
  total_plans: 84
  completed_plans: 80
  percent: 85
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
- [x] **Phase 36: Feature Computation** - Turf relative features + conditional interactions + Haron/Lap PIT-safe (completed 2026-05-20)
- [x] **Phase 36.1: HaronTime L4/LapTime Redesign** - クロスレベル派生特徴量 + BT hist_features修正 (completed 2026-05-20)
- [x] **Phase 36.1.1: MarketModel & RaceQuality配線修正** - Phase36特徴量ルーティング修正 + EV Tail Calibration (INSERTED) (completed 2026-05-20)
- [x] **Phase 37: OOF Health Infrastructure** - OOF成果物の健全性検査基盤 (fail-fast validation) (completed 2026-05-27)
- [ ] **Phase 38: InvestmentFeatureFrame** - 投資判断用統合特徴量フレーム (90-130列)

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
**Plans**: 2 plans
Plans:
- [x] 36-01-PLAN.md -- TRF + INT features: race-rank, weighted_recent_form, interactions, model registration
- [x] 36-02-PLAN.md -- HLF features: HaronTime L4/unified history, LapTime pace, model registration, dual-path

### Phase 36.1: HaronTime L4/LapTime Feature Redesign - クロスレベル派生特徴量への再設計 + backtest engine hist_features欠落修正 (INSERTED)

**Goal:** HaronTimeL4 データソースを races レベルに修正し、L3/L4 クロスレベル派生特徴量（closing_speed_ratio, haron_race_gap, pace_adj_finish）を PIT-safe に実装し、harontime_last3f を L3 ベースに統一し、backtest engine の hist_features 欠落バグを修正する
**Requirements**: RED-01, RED-02, RED-03, RED-04, RED-05, RED-06
**Depends on:** Phase 36
**Plans:** 2 plans
Plans:
- [x] 36.1-01-PLAN.md -- D-09 データソース修正 + D-01/02/03 新規派生特徴量 + D-07 last3f L3統一 + D-08 harontimel4 置換
- [x] 36.1-02-PLAN.md -- D-11 backtest engine hist_features マージ修正 + 全10モデル FEATURE_COLS 更新

### Phase 36.1.1: MarketModel & RaceQuality配線修正 — Phase36特徴量ルーティング修正 + EV Tail Calibration (INSERTED)

**Goal:** Phase36/36.1の強いfundamental特徴量が全モデルに一律登録されたことで、MarketModelの市場歪み検出役割とRaceQualityScreenerのレース品質判定が崩壊した問題を修復する。Phase36特徴量は残すが、モデルごとの役割に応じたルーティングに変更し、高EV長穴のtail calibrationを実装して、BT 2024 ROIをv1.7水準(97.8%)以上に回復させる

**主因診断:**
1. RaceQualityScreenerが利益源レースを大量脱落（旧ROI 103.4%の1831件を除外）
2. 高EV長穴の較正崩壊（EV>=1.5: 160件0勝、人気7+ ROI 58.3%）
3. 共通レースの馬選択悪化（別馬394件でROI 59.8%）
4. Phase36特徴量がMarketModelを支配（gain share 80%+、市場モデル契約が崩壊）

**Requirements**: RTG-01, RTG-02, RTG-03, RTG-04, RTG-05
**Depends on:** Phase 36.1
**Success Criteria** (what must be TRUE):
  1. MarketModel.FEATURE_COLS からPhase36 fundamental特徴量を除外し、market-only特徴量(odds distribution, overround, market_entropy, popularity_rank, late_money)に戻す
  2. RaceQualityScreenerが quality_score=0 固定問題を解消し、実効的なscore/thresholdを出力する
  3. RaceQualityScreenerが馬単位Phase36列ではなくrace aggregate特徴量(phase36_top1_strength, phase36_top1_top2_gap, form_signal_dispersion等)を使用する
  4. EV>=1.5高EV長穴をfeature family合意度で分類し、Phase36単独跳ねは縮小、複数合意はaggressiveに扱う
  5. v1.7 vs 現行 共通632レースの同馬/別馬差分レポートでPhase36寄与分解が可能
  6. BT 2024再学習で ROI >= 97.8% (v1.7水準) に回復、ベット数 >= 1500 **[Phase 38で検証]** -- Phase 36.1.1はコード修正を完了し、Phase 38 (Integrated Validation) でBT再実行によるROI検証を実施
**Plans:** 4/4 plans complete

Plans:
- [x] 36.1.1-01-PLAN.md -- Phase36特徴量の3モデルFEATURE_COLS除外 (RTG-01)
- [x] 36.1.1-02-PLAN.md -- Race-level aggregate追加 + quality_score修正 (RTG-02, RTG-03)
- [x] 36.1.1-03-PLAN.md -- EV Tail Calibration feature family合意度 (RTG-04)
- [x] 36.1.1-04-PLAN.md -- v1.7 vs 現行 差分診断スクリプト (RTG-05)

### Phase 37: OOF Health Infrastructure
**Goal**: 全OOF成果物が健全性検査を通過し、下流コンポーネント(キャリブレータ・ランカー)が信頼できるOOF予測を利用できる状態になる
**Depends on**: Phase 36.1.1
**Requirements**: OOF-01, OOF-02, OOF-03, OOF-04, OOF-05, OOF-06, OOF-07, OOF-08, XCT-05, XCT-08
**Success Criteria** (what must be TRUE):
  1. 空OOF保存がfail-fastで異常終了し、空ファイルが下流パイプラインに流入しない
  2. race_id単位でtrain/valid重複・同一race_idの複数fold混入が検出され、混入時は学習が停止する
  3. OOF top1 hit rate > 35% または top1 ROI > 200% の異常値が検出され停止する
  4. health manifestに行数、レース数、fold数、fold別race_id一意性、top1 hit rate/ROI、日付範囲、source model hashが記録され、同一入力から決定的な出力が生成される(XCT-05)
  5. 全health manifest artifactにversion、schema hash、source OOF manifest path、train日付範囲が含まれる(XCT-08)
**Plans:** 2 plans complete

Plans:
- [x] 37-01-PLAN.md -- OOFHealthValidator + artifact profiles + AbilityModel fold column
- [x] 37-02-PLAN.md -- Pipeline integration + fold wiring + test updates

### Phase 38: InvestmentFeatureFrame
**Goal**: 投資判断用統合特徴量フレーム (90-130列) を構築。モデル出力・市場データ・OOF予測を統合し、投資判断に特化した構造化特徴量を生成する
**Depends on**: Phase 37
**Requirements**: IFF-01, IFF-02, IFF-03, IFF-04, IFF-05, IFF-06, IFF-07, VAL-01
**Success Criteria** (what must be TRUE):
  1. InvestmentFeatureFrameBuilder.build_frame(df, mode="train"|"infer") が9カテゴリ90-130列の特徴量を生成 (IFF-01)
  2. train modeはOOF-safe列のみ使用、p_win_pred等in-sample列を拒否。infer modeは本番列を使用 (IFF-02)
  3. train/infer出力スキーマ同一(列名・列順・dtype同一)。テストが同一性をアサート (IFF-03)
  4. InvestmentFeatureSpec frozen dataclassによるスキーマレジストリ。全特徴量にメタデータあり (IFF-04)
  5. POST_RACE列非混入。漏洩テスト(VAL-01 scoped to frame)通過 (IFF-05)
  6. Parquetキャッシュ + sidecar manifest。決定性出力(同一入力→同一出力) (IFF-06)
  7. artifact manifest: feature_version, schema_hash, source_artifact_hash, OOF health manifest path, builder_version, mode, generated_at (IFF-07)
**Plans**: 3 plans
Plans:
- [ ] 38-01-PLAN.md -- Schema registry + leakage detection foundation (IFF-04, IFF-05)
- [ ] 38-02-PLAN.md -- Feature frame builder + manifest + cache (IFF-01, IFF-02, IFF-03, IFF-06, IFF-07)
- [ ] 38-03-PLAN.md -- Test suite + integration validation (VAL-01, full coverage)

## Progress

**Execution Order:**
Phases execute in numeric order: 35 -> 36 -> 36.1 -> 36.1.1 -> 37 -> 38

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
| 35. ETL Data Foundation | v1.8 | 2/2 | Complete    | 2026-05-19 |
| 36. Feature Computation | v1.8 | 2/2 | Complete | 2026-05-20 |
| 36.1. HaronTime L4/LapTime Redesign | v1.8 | 2/2 | Complete | 2026-05-20 |
| 36.1.1. MarketModel & RaceQuality配線修正 | v1.8 | 4/4 | Complete    | 2026-05-20 |
| 37. OOF Health Infrastructure | v2.0 | 2/2 | Complete | 2026-05-27 |
| 38. InvestmentFeatureFrame | v2.0 | 0/3 | Planned | - |
