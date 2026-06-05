---
gsd_state_version: 1.0
milestone: v2.3
milestone_name: Track Condition Feature Integration
status: in_progress
last_updated: "2026-06-05T00:00:00.000Z"
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
- **v2.1 MarketAware Calibration + Race-Level Ranker** -- Phases 39-42 (shipped 2026-05-28)
- **v2.2 ROI Recovery Analysis** -- Phases 43-46 (closed 2026-06-02, not_deployable)
- **v2.3 Track Condition Feature Integration** -- Phases 47-50 (in progress)

## Phases

<details>
<summary>v1.0-v2.2 (Phases 1-46) — All Closed</summary>

Phases 1-42 shipped across milestones v1.0 through v2.1.
Phases 43-46 closed as not_deployable (v2.2 ROI Recovery Analysis).
See `.planning/milestones/` for archived roadmaps.

</details>

### v2.3 Track Condition Feature Integration (In Progress)

**Milestone Goal:** 含水率・クッション値の連続値データを特徴量として統合し、BT ROI 97%+(v1.7レベル)を回復する

- [x] **Phase 47: ETL Data Pipeline** - 外部CSV(含水率/クッション値)をParquetに変換しDataRepositoryに統合 ✅ 2026-06-04
- [x] **Phase 48: Core Edge Features** - Tier 1+2の交互作用特徴量(含水率x脚質/枠位置、クッションx脚質/種牡馬)を実装・登録 ✅ 2026-06-05
- [x] **Phase 49: Derived & Higher-Order Features** - Tier 3+4の馬個体適性・ペース予測・異常値検出・既存インタラクション拡張を実装 ✅ 2026-06-05
- [x] **Phase 50: Safety & Validation** - Feature Routing Audit/POST_RACE CI検証 + BT ROI 97%+ + IC評価 + WF可用性確認 (completed 2026-06-05)

## Phase Details

### Phase 47: ETL Data Pipeline

**Goal**: 外部CSVデータ(含水率・クッション値)がParquetとしてDataRepository経由で利用可能になる
**Depends on**: Nothing (first phase of v2.3)
**Requirements**: ETL-01, ETL-02, ETL-03, ETL-04
**Status**: ✅ COMPLETED (2026-06-04)
**Success Criteria** (what must be TRUE):

  1. ✅ ダート含水率CSV(189K行)がParquetに変換され、エントリ単位ID→race_id集約でrace-levelデータとして保存される
  2. ✅ 芝クッション値CSV(133K行)がParquetに変換され、同様にrace-level集約される
  3. ✅ DataRepositoryから含水率・クッション値Parquetをロードでき、FeatureEngineにマージ可能なDataFrameが返る
  4. ✅ 含水率/クッション値がPOST_RACE_COLSに含まれていないことがCIテストで確認される

**Plans**: 2 plans (both completed)

Plans:

- [x] 47-01-PLAN.md — CSV→Parquet変換モジュール + precomputeスクリプト (ETL-01, ETL-02) ✅
- [x] 47-02-PLAN.md — DataRepository.load_track_conditions() + POST_RACE CI検証 (ETL-03, ETL-04) ✅

### Phase 48: Core Edge Features

**Goal**: 含水率・クッション値のTier 1+2交互作用特徴量がFeatureEngineに登録され、単独BTでROI寄与が観測できる
**Depends on**: Phase 47
**Requirements**: T1-01, T1-02, T2-01, T2-02, T2-03, REG-01
**Status**: ✅ COMPLETED (2026-06-05)
**Success Criteria** (what must be TRUE):

  1. ✅ dirt_moisture_x_kyakusitu特徴量がダートレースで計算され、含水率上昇時の逃げ馬有利バイアスを捉える
  2. ✅ turf_cushion_track_relative / turf_cushion_track_zscore特徴量が芝レースで計算され、コース間差が正規化される
  3. ✅ 含水率x枠位置交互作用 + 高含水/低含水フラグ、クッションx脚質交互作用、種牡馬xクッションビン交互作用が全て計算される
  4. ✅ 新特徴量がFEATURE_COLSの対象モデル(6モデル11リスト)に登録される（run_train.py実行はPhase 50で検証）

**Plans**: 1 plan (completed)

Plans:

- [x] 48-01-PLAN.md — track_condition_features module + pipeline integration + surgical routing (T1-01, T1-02, T2-01, T2-02, T2-03, REG-01) ✅

### Phase 49: Derived & Higher-Order Features

**Goal**: 馬個体の馬場状態適性・ペース予測・異常値検出・既存特徴量インタラクションが実装され、全特徴量層が揃う
**Depends on**: Phase 48
**Requirements**: T3-01, T3-02, T3-03, T3-04, T4-01, T4-02, T4-03, T4-04
**Success Criteria** (what must be TRUE):

  1. 馬個体の含水率/クッション値適性(horse_dirt_wet_hit_rate等)がPIT-safeに計算され、過走履歴から適性カテゴリ(湿得意/乾得意/万能)が分類される
  2. クッション値/含水率のコース別月別偏差(season_deviation)が計算される
  3. 含水率/クッション値から先行バイアススコア・蹴り返りリスク・ペース予測が算出され、レースフィールド条件マッチスコアが計算される
  4. クッション/含水率異常値検出(2σ逸脱) + 既存特徴量とのインタラクション(距離/馬齢/脚質等)が全て計算される

**Status**: ✅ COMPLETED (2026-06-05)
**Plans**: 2 plans (both completed)

Plans:
**Wave 1**

- [x] 49-01-PLAN.md — T3 precompute parquet (horse_track_aptitude) + repository + FeatureEngine merge (T3-01, T3-02, T3-03) ✅

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 49-02-PLAN.md — T3-04/T4-01~04 feature computation + pipeline integration + surgical routing (T3-04, T4-01, T4-02, T4-03, T4-04) ✅

### Phase 50: Safety & Validation

**Goal**: 全新特徴量の安全性がCI検証され、BT ROI 97%+が確認されてデプロイ可能になる
**Depends on**: Phase 49
**Requirements**: REG-02, REG-03, VLD-01, VLD-02, VLD-03
**Success Criteria** (what must be TRUE):

  1. Feature Routing Auditが新特徴量の外科的ルーティング(MarketModel/RaceQualityScreener除外等)を検証し、PASSする
  2. 新特徴量のPOST_RACE分類が3層CI検証(whitelist/forbidden/manual)で正しく確認される
  3. マルチ年度BT(2024/2025)でBT ROI 97%+が達成される(v1.7レベル回復)
  4. 新特徴量のIC評価(C直交IC)が実行され、既存特徴量と独立したシグナルであることが確認される
  5. クッション値データのWF Fold0(2020-2023学習)でのNaN率が許容範囲内であることが確認される

**Plans**: 2 plans

Plans:
**Wave 1**

- [x] 50-01-PLAN.md — CI検証: Feature Routing Audit拡張 + Surface-aware NaN CI + POST_RACE 3層CI (REG-02, REG-03, VLD-03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 50-02-PLAN.md — Training再実行(特徴量キャッシュ/OOF再生成) + WF Fold0 NaN診断 + 段階BT ROI検証 + IC評価 (VLD-01, VLD-02, VLD-03)

## Progress

**Execution Order:**
Phases execute in numeric order: 47 → 48 → 49 → 50

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-42 | v1.0-v2.1 | 85/85 | Complete | 2026-05-28 |
| 43-46 | v2.2 | 8/8 | Complete (not_deployable) | 2026-06-02 |
| 47. ETL Data Pipeline | v2.3 | 2/2 | Complete | 2026-06-04 |
| 48. Core Edge Features | v2.3 | 1/1 | Complete | 2026-05-05 |
| 49. Derived & Higher-Order Features | v2.3 | 2/2 | Complete | 2026-06-05 |
| 50. Safety & Validation | v2.3 | 2/2 | Complete   | 2026-06-05 |
