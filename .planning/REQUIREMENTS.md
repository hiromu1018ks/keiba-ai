# Requirements: keiba-ai v1.6 Feature Engineering Overhaul

## Milestone Requirements

### Category 1: Safety & Quality Gate

- [x] **SAFE-01**: build_all()出口でPOST_RACE_COLSを確実にドロップするリーク修正を適用できる
- [x] **SAFE-02**: permutation重要度 + gain重要度を計算するfeature importance監査スクリプトを使用できる

### Category 2: Feature Audit & Pruning

- [ ] **AUDIT-01**: 100+特徴量のpermutation重要度をOOFデータで計算し、各特徴量の有効性を定量化できる
- [ ] **AUDIT-02**: 重要度ゼロ/負のノイズ特徴量をFEATURE_COLSから除外し、ROI改善を検証できる
- [ ] **AUDIT-03**: 特徴量モジュール変更時にキャッシュを自動クリアする仕組みを導入できる

### Category 3: Quick Win — Wire Existing Modules

- [ ] **WIRE-01**: JockeyContextFeatures(4特徴量: jockey_wr_overall等)をtraining_pipelineに配線できる
- [ ] **WIRE-02**: TrainerContextFeatures(4特徴量: trainer_wr_overall等)をtraining_pipelineに配線できる
- [ ] **WIRE-03**: JockeyTrainerComboFeatures(4特徴量: jt_combo_place_rate等)をtraining_pipelineに配線できる

### Category 4: EveryDB2 New Features

- [x] **DATA-01**: n_hansyokuテーブルから血統特徴量（種牡馬系統、母系BMS等）を抽出・生成できる
- [x] **DATA-02**: n_recordテーブルからコース別タイム指数等の特徴量を生成できる
- [x] **DATA-03**: レース内の全馬に対する相対比較特徴量（相対ランク、偏差値等）を生成できる
- [x] **DATA-04**: n_miningテーブル(82列)を分析し、PRE/POST分類後に特徴量として抽出できる

### Category 5: Feature Interactions & Transformations

- [x] **INTER-01**: レース内相対ランク特徴量（オッズ、能力値等の相対位置）を生成できる
- [x] **INTER-02**: ドメイン知識に基づく10-15個の条件付き交互作用項を生成できる
- [x] **INTER-03**: 高カーディナリティカテゴリ変数のターゲットエンコーディング（血統コード、騎手コード等）を実装できる

## Future Requirements (Deferred)

- n_taisyogata_miningペアワイズ比較特徴量（テーブル構造未検証）
- n_saleオークション価格特徴量
- n_banusi馬主統計特徴量
- WF検証スクリプトの実際の実行（~4時間、PostgreSQL環境必要）

## Out of Scope

| Feature | Reason |
|---------|--------|
| LSTM/Transformer時系列特徴量 | 過去5-15走では過学習リスク高（PROJECT.md既存決定） |
| 外部データソースの導入 | EveryDB2データで十分（PROJECT.md既存決定） |
| CQR設計見直し | v1.5で過学習の原因と判明、別アプローチを採用 |
| 複勝/ワイドモデルの変更 | 単勝に集中（PROJECT.md既存決定） |
| モデル構造の抜本変更 | 特徴量品質の改善を先に行い、構造変更は最終手段 |

## Traceability

| REQ-ID | Phase | Plan | Status |
|--------|-------|------|--------|
| SAFE-01 | Phase 23 | TBD | Pending |
| SAFE-02 | Phase 23 | TBD | Pending |
| AUDIT-01 | Phase 24 | TBD | Pending |
| AUDIT-02 | Phase 24 | TBD | Pending |
| AUDIT-03 | Phase 24 | TBD | Pending |
| WIRE-01 | Phase 25 | TBD | Pending |
| WIRE-02 | Phase 25 | TBD | Pending |
| WIRE-03 | Phase 25 | TBD | Pending |
| DATA-01 | Phase 26 | TBD | Pending |
| DATA-02 | Phase 26 | TBD | Pending |
| DATA-03 | Phase 26 | TBD | Pending |
| DATA-04 | Phase 26 | TBD | Pending |
| INTER-01 | Phase 27 | TBD | Pending |
| INTER-02 | Phase 27 | TBD | Pending |
| INTER-03 | Phase 27 | TBD | Pending |

---
*Last updated: 2026-05-11 — v1.6 roadmap created, traceability updated*
