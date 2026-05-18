---
gsd_state_version: 1.0
milestone: v1.7
milestone_name: Market-Independent Edge Discovery
status: executing
last_updated: "2026-05-17T13:11:04Z"
progress:
  total_phases: 34
  completed_phases: 30
  total_plans: 65
  completed_plans: 65
  v17_plans: 5
  v17_plans_complete: 5
  percent: 88
---

# Roadmap: keiba-ai Win Model Improvement

## Milestones

- **v1.0 Win Model** - Phases 1-4 (shipped 2026-05-03)
- **v1.1 ROI Advanced Model** - Phases 5-7 (shipped 2026-05-03)
- **v1.2 Win Backtest Validation** - Phases 8-10 (shipped 2026-05-04)
- **v1.3 Betting Strategy Optimization** - Phases 11-13 (shipped 2026-05-05)
- **v1.4 Ensemble Filter Recalibration** - Phases 14-18 (shipped 2026-05-07)
- **v1.5 Model Accuracy Improvement** - Phases 19-22 (shipped 2026-05-10)
- **v1.6 Feature Engineering Overhaul** - Phases 23-28 (shipped 2026-05-17)
- **v1.7 Market-Independent Edge Discovery** - Phases 29-34 (in progress)

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

### v1.7 Market-Independent Edge Discovery (In Progress)

**Milestone Goal:** Echo Chamber脱却 -- 市場と独立な予測成分を獲得し、ROI 100%超えを実現する

- [x] **Phase 29: ETL Expansion** - 三連複/馬連/三連単オッズのParquet抽出 -- completed 2026-05-17
- [ ] **Phase 30: Residual IC Evaluation Framework** - 市場独立性の測定器を構築
- [x] **Phase 31: Race-Level Aggregation Features** - レース全体の市場構造を捉える6特徴量 + 既存特徴量昇格 -- completed 2026-05-18
- [ ] **Phase 32: Market Cross-Consistency Features** - Harville理論オッズによる馬券種クロス整合性
- [ ] **Phase 33: Gain per Depth Diagnostic** - LightGBM木構造のdepth別gain分析
- [ ] **Phase 34: Validation and Manifest Update** - 全特徴量統合検証 + manifest凍結

## Phase Details

### Phase 29: ETL Expansion
**Goal**: 三連複/馬連/三連単オッズがParquetファイルとして利用可能になり、将来の市場クロス整合性特徴量のデータ基盤が整う
**Depends on**: Nothing (no dependency on other v1.7 phases; can run in parallel with Phase 30)
**Requirements**: ETL-01, ETL-02, ETL-03, ETL-04
**Success Criteria** (what must be TRUE):
  1. run_etl.py --mode full で三連複(n_odds_sanren)、馬連(n_odds_umaren)、三連単(n_odds_sanrentan)のParquetファイルが生成される
  2. 抽出データが2015-2025年の期間をカバーし、欠損率が許容範囲(30%以下)であることをカバレッジレポートで確認できる
  3. DataRepository経由で新しいParquetファイルからDataFrameとしてデータを読み込める
**Plans**: 3 plans
Plans:
- [x] 29-01-PLAN.md -- Fix PK definitions in etl_tables.yaml + add type rules in etl.py (complete)
- [x] 29-02-PLAN.md — Create DataRepository class with 3 load methods + tests (complete)
- [x] 29-03-PLAN.md -- Add coverage verification to run_etl.py + tests (complete)

### Phase 30: Residual IC Evaluation Framework
**Goal**: v1.6モデルの市場独立予測力を4定式化バッテリーで定量的に測定でき、新特徴量の効果を客観的に評価できる状態になる
**Depends on**: Nothing (no dependency on ETL; uses existing model predictions)
**Requirements**: RIC-01, RIC-02, RIC-03, RIC-04, RIC-05, RIC-06
**Success Criteria** (what must be TRUE):
  1. B差分IC / C直交IC / E Incremental IC / Per-race IC の4指標がOOF予測に対して計算され、数値レポートとして出力される
  2. 4定式化バッテリーの方向一致性が自動検証され、矛盾がある場合に警告が出力される
  3. v1.6モデルのベースラインIC値が記録され、将来の改善測定の基準として利用可能である
**Plans**: 2 plans
Plans:
- [ ] 30-01-PLAN.md -- IC evaluation module with 4 formulations + tests (RIC-01~06)
- [ ] 30-02-PLAN.md -- OOF save hook + CLI script + pipeline integration (RIC-05)

### Phase 31: Race-Level Aggregation Features
**Goal**: レース全体の市場構造を表す6特徴量が追加され、既存の未登録特徴量2つがFEATURE_COLSに昇格し、train/inference両パスで同じ特徴量が計算される
**Depends on**: Phase 29 (for trio odds data; race-level features can use existing tanodds immediately)
**Requirements**: RLF-01, RLF-02, RLF-03, RLF-04, RLF-05, RLF-06, RLF-07, EFP-01, EFP-02, EFP-03
**Success Criteria** (what must be TRUE):
  1. 6つのrace-level特徴量(rl_log_odds_entropy, rl_odds_dispersion, rl_top3_odds_gap, rl_top1_odds, rl_favorite_rank_gap, rl_n_horses)がbuild_all()とbuild_features()の両方で計算される
  2. implied_prob_hhi と odds_skewness がFEATURE_COLSに昇格し、SHA256 manifestが更新される
  3. 全テストが通過し、新特徴量がPOST_RACE情報漏洩テストで安全であることが確認される
**Plans**: 2 plans
Plans:
- [x] 31-01-PLAN.md -- race_level_features.py新規作成 + 全12モデルFEATURE_COLS昇格 (RLF-01~06, EFP-01~02) (complete)
- [x] 31-02-PLAN.md — feature_engine統合 + POST_RACEテスト拡張 + manifest更新 (RLF-07, EFP-03) (complete)

### Phase 32: Market Cross-Consistency Features
**Goal**: Harville理論オッズによる馬券種クロス整合性特徴量が追加され、単勝×ワイド×三連複の市場構造矛盾を捉えられるようになる
**Depends on**: Phase 29 (needs trio/wide odds data), Phase 31 (race-level patterns established)
**Requirements**: MCF-01, MCF-02, MCF-03, MCF-04, MCF-05, MCF-06, MCF-07
**Success Criteria** (what must be TRUE):
  1. Harville公式による理論ワイドオッズが計算され、実オッズとの比率特徴量が生成される
  2. 5つの市場クロス整合性特徴量(rl_favorite_in_wide_top1, rl_trio_overlap, rl_market_consistency, rl_trio_odds_ratio, rl_wide_harville_ratio)がFEATURE_COLSに追加される
  3. ワイドオッズmergeをbuild_all()に統合され、training/backtestでの重複コードが排除される
  4. 全特徴量がpre-race snapshot oddsのみを使用し、post-race payout oddsの情報漏洩がないことがテストで検証される
**Plans**: TBD

### Phase 33: Gain per Depth Diagnostic
**Goal**: LightGBMの木構造をdepth別に分析し、race-level/market-cross特徴量が適切なshallow depthで機能していることを検証できる
**Depends on**: Phase 31, Phase 32 (needs trained models with new features)
**Requirements**: GPD-01, GPD-02, GPD-03, GPD-04
**Success Criteria** (what must be TRUE):
  1. LightGBM trees_to_dataframe()でdepth別gain寄与率が集計され、Market/Fundamental/Categoricalの3分類でレポートが出力される
  2. StackedEnsemble内のLightGBMモデルにアクセスし、depth別分析が実行できる
  3. 暗黙的Two-Stage構造(上位depth=Market, 下位depth=Fundamental)の仮説がデータで検証される
**Plans**: TBD

### Phase 34: Validation and Manifest Update
**Goal**: 新特徴量追加後のフルバックテストでROI改善を確認し、IC改善をベースラインと比較し、FEATURE_COLS manifestを凍結する
**Depends on**: Phase 30, Phase 31, Phase 32, Phase 33
**Requirements**: VAL-01, VAL-02, VAL-03, VAL-04, VAL-05
**Success Criteria** (what must be TRUE):
  1. マルチ年度バックテストが完了し、ROIがベースライン(85.7%)と比較できる結果が出力される
  2. C直交ICがv1.6ベースライン比で改善されていることが確認される
  3. Gain per Depthレポートで新特徴量がdepth 3-5で機能していることが確認される
  4. FEATURE_COLS manifestがSHA256ハッシュで凍結され、全12モデルのmanifestが更新される
  5. 新特徴量に対するPOST_RACE情報漏洩テストが全て通過する
**Plans**: TBD

## Progress

**Execution Order:**
Phases 29 and 30 can run in parallel (no dependency between them).
Then: 31 -> 32 -> 33 -> 34 (sequential).

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
| 30. Residual IC Evaluation Framework | v1.7 | 0/2 | Planned | - |
| 31. Race-Level Aggregation Features | v1.7 | 0/2 | Planned | - |
| 32. Market Cross-Consistency Features | v1.7 | 0/? | Not started | - |
| 33. Gain per Depth Diagnostic | v1.7 | 0/? | Not started | - |
| 34. Validation and Manifest Update | v1.7 | 0/? | Not started | - |
