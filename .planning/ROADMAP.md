# Roadmap: keiba-ai Win Model Improvement

## Milestones

- **v1.0 Win Model** - Phases 1-4 (shipped 2026-05-03)
- **v1.1 ROI Advanced Model** - Phases 5-7 (shipped 2026-05-03)
- **v1.2 Win Backtest Validation** - Phases 8-10 (shipped 2026-05-04)
- **v1.3 Betting Strategy Optimization** - Phases 11-13 (shipped 2026-05-05)
- **v1.4 Ensemble Filter Recalibration** - Phases 14-18 (shipped 2026-05-07)
- **v1.5 Model Accuracy Improvement** - Phases 19-22 (shipped 2026-05-10)
- **v1.6 Feature Engineering Overhaul** - Phases 23-28 (in progress)

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

**Result:** ROI v1.4:83.1% -> v1.5:84.4% (+1.3pp improvement). 95% target not reached.

</details>

### v1.6 Feature Engineering Overhaul (In Progress)

**Milestone Goal:** CQRに頼らず、特徴量の質と量を根本から見直し、バックテストROI 100%超えを目指す

- [x] **Phase 23: Safety Gate** - POST_RACE漏洩修正 + feature importance監査スクリプト構築 (completed 2026-05-11)
- [x] **Phase 24: Feature Audit & Pruning** - 100+特徴量の有効性監査 + ノイズ除去 + キャッシュ無効化 (2/2 plans, completed 2026-05-12)
- [x] **Phase 25: Quick Win Wire Existing** - 既存実装済み12特徴量(Jockey/Trainer/Combo)の配線 (2/2 plans, completed 2026-05-13)
- [ ] **Phase 26: EveryDB2 New Features** - 未活用テーブルからの血統/タイム/相対比較/mining特徴量抽出
- [ ] **Phase 27: Feature Interactions** - 条件付き交互作用項 + ターゲットエンコーディング
- [ ] **Phase 28: Validation & Freeze** - 統合バックテスト + ROI 100%検証 + 特徴量凍結

## Phase Details

### Phase 23: Safety Gate
**Goal**: 全特徴量パイプラインからレース後情報漏洩を排除し、特徴量品質監査の基盤を構築する
**Depends on**: Phase 22 (v1.5 complete)
**Requirements**: SAFE-01, SAFE-02
**Success Criteria** (what must be TRUE):
  1. build_all()の出力DataFrameにPOST_RACE_COLSに含まれる列が一つも含まれない
  2. POST_RACE漏洩を検出するCIテストが追加され、パスする
  3. permutation重要度 + gain重要度を計算する監査スクリプトが実行可能で、OOFデータの結果をCSV/JSONで出力する
**Plans**: 2 plans

Plans:
- [x] 23-01-PLAN.md -- M1-M6漏洩修正 + 3層CIテスト (SAFE-01)
- [x] 23-02-PLAN.md -- permutation+gain重要度監査スクリプト拡張 (SAFE-02)

### Phase 24: Feature Audit & Pruning
**Goal**: 100+特徴量の有効性を定量化してノイズ特徴量を除外し、クリーンな特徴量ベースラインを確立する
**Depends on**: Phase 23
**Requirements**: AUDIT-01, AUDIT-02, AUDIT-03
**Success Criteria** (what must be TRUE):
  1. 全特徴量のpermutation重要度がOOFデータで計算され、各特徴量のスコアが確認できる
  2. 重要度ゼロ/負のノイズ特徴量がFEATURE_COLSから除外され、除外前後のROIが比較できる
  3. 特徴量モジュール変更時にhorse_features.parquetキャッシュが自動クリアされる
**Plans**: 2 plans

Plans:
- [x] 24-01-PLAN.md -- Tier分類ロジック + コードハッシュキャッシュ無効化 (AUDIT-01, AUDIT-03)
- [x] 24-02-PLAN.md -- Tier 1プルーニング適用 + OOF安全性確認 + ロールバック (AUDIT-02)

### Phase 25: Quick Win Wire Existing
**Goal**: 既に実装・テスト済みのJockey/Trainer/Combo合計12特徴量をパイプラインに接続し、ROI改善を確認する
**Depends on**: Phase 24
**Requirements**: WIRE-01, WIRE-02, WIRE-03
**Success Criteria** (what must be TRUE):
  1. JockeyContextFeatures(4特徴量: jockey_wr_overall等)がtraining_pipelineで生成され、FEATURE_COLSに含まれる
  2. TrainerContextFeatures(4特徴量: trainer_wr_overall等)がtraining_pipelineで生成され、FEATURE_COLSに含まれる
  3. JockeyTrainerComboFeatures(4特徴量: jt_combo_place_rate等)がtraining_pipelineで生成され、FEATURE_COLSに含まれる
**Plans**: 2 plans

Plans:
- [x] 25-01-PLAN.md -- FEATURE_COLS配線 + paper_trading predictor更新 + テスト更新 (WIRE-01, WIRE-02, WIRE-03)
- [x] 25-02-PLAN.md -- フルバックテスト実行 + ROI検証 (WIRE-01, WIRE-02, WIRE-03)

### Phase 26: EveryDB2 New Features
**Goal**: EveryDB2の未活用テーブル(n_hansyoku, n_record, n_mining)から高価値特徴量を抽出し、モデル入力を拡張する
**Depends on**: Phase 25
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. n_hansyokuテーブルから血統特徴量(種牡馬系統、母系BMS等)が生成され、PRE/POST分類が文書化されている
  2. n_recordテーブルからコース別タイム指数等の特徴量が生成される
  3. レース内全馬の相対比較特徴量(相対ランク、偏差値等)が生成される
  4. n_miningテーブル(82列)のPRE/POST分類が完了し、PRE列から特徴量が抽出される
**Plans**: 3 plans

Plans:
- [x] 26-01-PLAN.md -- n_mining PIT監査 + mining_features.py + FEATURE_COLS更新 (DATA-04)
- [x] 26-02-PLAN.md -- dam_pedigree_features.py + sire BMS拡張 + record_features.py + FEATURE_COLS更新 (DATA-01, DATA-02)
- [ ] 26-03-PLAN.md -- relative_features.py + FEATURE_COLS更新 + 全テスト通過確認 (DATA-03)

### Phase 27: Feature Interactions
**Goal**: 最終ベース特徴量セット上にドメイン知識に基づく交互作用項を生成し、高カーディナリティカテゴリ変数をターゲットエンコーディングで処理する
**Depends on**: Phase 26
**Requirements**: INTER-01, INTER-02, INTER-03
**Success Criteria** (what must be TRUE):
  1. オッズ・能力値等のレース内相対ランク特徴量が生成され、FEATURE_COLSに含まれる
  2. 10-15個のドメイン知識に基づく条件付き交互作用項が生成される(例: 芝x距離、クラスxオッズ帯)
  3. 高カーディナリティカテゴリ変数(血統コード、騎手コード等)がターゲットエンコーディングされ、OOFリークなしでFEATURE_COLSに含まれる
**Plans**: TBD

### Phase 28: Validation & Freeze
**Goal**: 全特徴量変更を統合したバックテストでROI 100%超えを確認し、特徴量セットを凍結する
**Depends on**: Phase 27
**Requirements**: (validation of all prior phase outputs)
**Success Criteria** (what must be TRUE):
  1. 統合バックテストROIが100%以上である
  2. 既存テスト全通過(回帰なし)
  3. FEATURE_COLSが凍結され、ハッシュが記録されている
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 23 -> 24 -> 25 -> 26 -> 27 -> 28

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
| 23. Safety Gate | v1.6 | 2/2 | Complete    | 2026-05-11 |
| 24. Feature Audit & Pruning | v1.6 | 2/2 | Complete | 2026-05-12 |
| 25. Quick Win Wire Existing | v1.6 | 2/2 | Complete | 2026-05-13 |
| 26. EveryDB2 New Features | v1.6 | 2/3 | In Progress|  |
| 27. Feature Interactions | v1.6 | 0/? | Not started | - |
| 28. Validation & Freeze | v1.6 | 0/? | Not started | - |
