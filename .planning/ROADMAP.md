---
gsd_state_version: 1.0
milestone: v2.4
milestone_name: Paper Trading Pipeline Integration
status: in_progress
last_updated: "2026-06-06T00:00:00.000Z"
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
- **v2.3 Track Condition Feature Integration** -- Phases 47-50 (shipped 2026-06-05)
- **v2.4 Paper Trading Pipeline Integration** -- Phases 51-54 (in progress)

## Phases

<details>
<summary>v1.0-v2.2 (Phases 1-46) — All Closed</summary>

Phases 1-42 shipped across milestones v1.0 through v2.1.
Phases 43-46 closed as not_deployable (v2.2 ROI Recovery Analysis).
See `.planning/milestones/` for archived roadmaps.

</details>

<details>
<summary>v2.3 Track Condition Feature Integration (Phases 47-50) — SHIPPED 2026-06-05</summary>

- [x] Phase 47: ETL Data Pipeline (2/2 plans) — completed 2026-06-04
- [x] Phase 48: Core Edge Features (1/1 plan) — completed 2026-06-05
- [x] Phase 49: Derived & Higher-Order Features (2/2 plans) — completed 2026-06-05
- [x] Phase 50: Safety & Validation (2/2 plans) — completed 2026-06-05

23 track condition features (T1/T2: 8, T3: 4, T4: 11) with surgical routing, CI safety validation, NaN diagnostics, and IC evaluation framework. BT ROI 87.3% (raw), post-hoc --min-win-ev 1.40 → 124.4%.

</details>

### v2.4 Paper Trading Pipeline Integration (In Progress)

**Milestone Goal:** BT で検証済みの学習・推論パイプラインを用いて PT を実行し、推論から精算まで1コマンドで完遂する。精算整合性と同一実装・同一設定契約を確保し、ROI を正確に測定する。

- [ ] **Phase 51: Settlement Integrity & Training Pipeline** - 精算整合性(ROI過大評価修正)と学習パイプライン修正(PT用モデル生成)
- [ ] **Phase 52: Shared Feature Builder & Consistency** - BT/PT共通特徴量構築関数の抽出とパイプライン一貫性検証基盤
- [ ] **Phase 53: Strategy Alignment & Live Data** - 戦略完全整合と当日データ取得でBT検証済みパイプラインをPTで再現
- [ ] **Phase 54: Automation & Reporting** - 1コマンドrun modeと評価レポート拡張で運用完全自動化

## Phase Details

### Phase 51: Settlement Integrity & Training Pipeline

**Goal**: PT の ROI 測定が信頼できること。全ベット(的中・不的中)が正しく精算され、学習パイプラインが PT 用モデルを生成できる
**Depends on**: Nothing (first phase of v2.4)
**Requirements**: STL-01, STL-02, STL-03, STL-04, STL-05, TRN-01, TRN-02, TRN-03, TRN-04, TRN-05
**Success Criteria** (what must be TRUE):

  1. PT の bets.parquet に settlement_status 列(pending/settled)と outcome 列(won/lost)が分離して含まれ、全ベットが settlement_status=settled になる
  2. Win ベットが build_win_payout_map() で精算され、Place ベットも負けを含めて正確に精算される
  3. ROI 計算が的中のみではなく負け含む全ベットで算出され、従来の過大評価が修正される
  4. DB 遅延時に払戻データをリトライ取得し、最終レース後に一括リトライが実行される
  5. run_train.py --betting-target win で単勝 PT 用モデルを学習でき、学習前に必須 Parquet(track_conditions/horse_track_aptitude含む)の日付範囲・NaN率・更新日時検証が走る
  6. 特徴量キャッシュの依存元に track_conditions.parquet/horse_track_aptitude.parquet が追加され、更新後の古いキャッシュ使用が防止される
  7. track_stats/track_month_stats がモデル成果物に保存・復元され、PTで季節偏差等が NaN にならない

**Plans**: 3 plans
Plans:
**Wave 1**

- [ ] 51-01-PLAN.md — Extract payout_maps.py pure functions from BacktestEngine
- [ ] 51-03-PLAN.md — Training pipeline fixes (--betting-target, track_stats persistence, ModelLoader priority)

**Wave 2** *(blocked on Wave 1 completion)*

- [ ] 51-02-PLAN.md — Overhaul PaperReconciler with 3-column state model and thin _run_reconcile

### Phase 52: Shared Feature Builder & Consistency

**Goal**: BT と PT と TrainingPipeline が同一の特徴量生成関数を呼び出し、パイプラインの同一実装・同一設定契約が検証可能であること
**Depends on**: Phase 51
**Requirements**: PLN-01, PLN-02, PLN-03, PLN-04
**Success Criteria** (what must be TRUE):

  1. BacktestEngine.prepare_data() から build_inference_features() が抽出され、BT/PT/TrainingPipeline が同じ関数を呼び出す。7つのギャップ(DamPedigree/Record/Mining/PaceAptitude 3列/Sire/Course)が一括解消される
  2. PT 実行記録に MLflow run ID・学習期間・コードハッシュ・feature manifest hash が保存される
  3. 2026年 PT で予測日以降のデータ(特徴量統計・OddsBandFilter 校正・HP・strategy manifest)が使用されていないことを検証する仕組みが動作する
  4. PT 実行中のパラメータ不変性を ParameterFreezeProtocol で検証する仕組みが動作する

**Plans**: TBD

### Phase 53: Strategy Alignment & Live Data

**Goal**: PT が BT で検証済みの戦略パラメータを適用して推論を実行し、当日のトラック条件データを取得して特徴量に反映できること
**Depends on**: Phase 52
**Requirements**: STR-01, STR-02, STR-03, STR-04, STR-05, STR-06, LIV-01, LIV-02, LIV-03 (9)
**Success Criteria** (what must be TRUE):

  1. PT で strategy_manifest を読み込み manifest/PFP を適用し、--betting-target(win|place) と --betting-mode を指定できる。Wide は v2.4 対象外
  2. DrawdownController・OddsBandFilter・RaceQualityScreener が PT パイプラインで BT と同一に動作する
  3. BT/PT の regime 検出が統一(AGGRESSIVE固定 vs 動体の決定を含む)されている
  4. JRA 公式サイトから開催場ごとの芝クッション値・ダート含水率を取得し、ゴール前・4コーナー含水率を既存 dirt_moisture への集約規則で race_id へ展開できる
  5. 取得値・測定時刻・取得時刻・取得元が保存され、取得失敗・値が古い・HTML構造変更検知時に予測を停止し非ゼロ終了する

**Plans**: TBD

### Phase 54: Automation & Reporting

**Goal**: モデル検証から精算・集計まで1コマンドで完遂し、週次集計・累積履歴・target別集計で PT の結果を正確に評価できること
**Depends on**: Phase 53
**Requirements**: AUT-01, AUT-02, AUT-03, RPT-01, RPT-02, RPT-03, RPT-04
**Success Criteria** (what must be TRUE):

  1. --mode run で事前学習済みモデルの検証から開始し、予測→監視→精算→集計の全工程が1コマンドで実行される(学習は含めない)
  2. 処理済みレースの再実行がスキップされ、クラッシュ後の再起動で未処理レースのみ再開する
  3. DB接続障害・データ欠損・モデル不整合時に非ゼロ終了コードを返す
  4. 週次 ROI・的中率・ベット数の JSON 集計と pending/settled/won/lost を含む累積ベット履歴が出力される
  5. Win/Place 別 ROI・的中率集計に MLflow run ID・学習期間・manifest hash が含まれる

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 51 → 52 → 53 → 54

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-42 | v1.0-v2.1 | 85/85 | Complete | 2026-05-28 |
| 43-46 | v2.2 | 8/8 | Complete (not_deployable) | 2026-06-02 |
| 47-50 | v2.3 | 7/7 | Complete | 2026-06-05 |
| 51. Settlement Integrity & Training | v2.4 | 2/3 | In progress | - |
| 52. Shared Feature Builder & Consistency | v2.4 | 0/? | Not started | - |
| 53. Strategy Alignment & Live Data | v2.4 | 0/? | Not started | - |
| 54. Automation & Reporting | v2.4 | 0/? | Not started | - |
