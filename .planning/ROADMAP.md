# Roadmap: keiba-ai Win Model Improvement

## Milestones

- ✅ **v1.0 Win Model** - Phases 1-4 (shipped 2026-05-03)
- ✅ **v1.1 ROI Advanced Model** - Phases 5-7 (shipped 2026-05-03)
- ✅ **v1.2 Win Backtest Validation** - Phases 8-10 (shipped 2026-05-04)
- ✅ **v1.3 Betting Strategy Optimization** - Phases 11-13 (shipped 2026-05-05)
- ✅ **v1.4 Ensemble Filter Recalibration** - Phases 14-18 (shipped 2026-05-07)
- 🔄 **v1.5 Model Accuracy Improvement** - Phases 19-22 (active)

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, ...): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

<details>
<summary>✅ v1.0 Win Model (Phases 1-4) — SHIPPED 2026-05-03</summary>

### Phase 1: Feature Analysis & Enhancement
**Goal**: 単勝予測に寄与する特徴量を特定し、ノイズを排除し、単勝特化の新特徴量を追加して、モデル入力の質を最大化する
**Depends on**: Nothing (first phase)
**Plans**: 2 plans

Plans:
- [x] 01-01: SHAP/gain特徴量重要度分析モジュールとノイズ特定・除外
- [x] 01-02: 単勝特化新特徴量6つの実装・統合

### Phase 2: Win Benter Combination & Calibration
**Goal**: 単勝予測に市場効率信号を組み込み、確率を正規化・キャリブレーションすることでEV推定精度を飛躍的に向上させる
**Depends on**: Phase 1
**Plans**: 2 plans

Plans:
- [x] 02-01: WinBenterGate実装・OOF予測生成・レース正規化・パイプライン統合
- [x] 02-02: Beta/Isotonicキャリブレーション比較・ECE評価・信頼性ダイアグラム

### Phase 3: Selection Gate, Confidence & Betting
**Goal**: 学習済みゲートで低信頼レースを除外し、JRA控除率25%を考慮した最適ベッティング戦略を統合する
**Depends on**: Phase 2
**Plans**: 2 plans

Plans:
- [x] 03-01: WinSelectionGate実装・Conformal信頼性拡張・パイプライン統合
- [x] 03-02: JRA控除率考慮edge_threshold更新・ベッティング戦略調整

### Phase 4: Walk-Forward Validation
**Goal**: 複数年度のウォークフォワード検証で過学習を検出し、ROI>100%が単年度の偶然でないことを証明する
**Depends on**: Phase 3
**Plans**: 1 plan

Plans:
- [x] 04-01: WFValidationResultデータクラス + 過学習検出ユーティリティ + run_wf_validation.py CLIスクリプト

</details>

<details>
<summary>✅ v1.1 ROI Advanced Model (Phases 5-7) — SHIPPED 2026-05-03</summary>

### Phase 5: Foundation Features
**Goal**: 過去走の時系列特徴量・展開予測特徴量・オッズ変動特徴量を追加し、後続のモデル改善がより豊かな入力から恩恵を受けられるようにする
**Depends on**: Phase 4 (v1.0 complete)
**Requirements**: TSER-01, TSER-02, TSER-03, PACE-01, PACE-02, ODTS-01, ODTS-02
**Plans**: 2 plans

Plans:
- [x] 05-01: Time-series and pace features (TSER-01, TSER-02, TSER-03, PACE-01, PACE-02)
- [x] 05-02: Odds time-series features (ODTS-01, ODTS-02)

### Phase 6: Odds Deviation EV
**Goal**: モデル予測確率と市場オッズの乖離をEV信号としてモデルに直接組み込み、Conformal予測区間でベット選択の信頼性を最適化する
**Depends on**: Phase 5
**Requirements**: ODDS-01, ODDS-02, ODDS-03
**Plans**: 1 plan

Plans:
- [x] 06-01: Odds deviation EV features and pipeline integration (ODDS-01, ODDS-02, ODDS-03)

### Phase 7: Ensemble Enhancement
**Goal**: 3モデルスタッキング(LightGBM+XGBoost+CatBoost)の多様性を強制するハイパーパラメータ最適化・early stopping・特徴量サブセット分割を実装し、予測精度を最大化する
**Depends on**: Phase 6
**Requirements**: ENS-01, ENS-02, ENS-03
**Plans**: 1 plan

Plans:
- [x] 07-01: Ensemble stacking enhancement with forced diversity (ENS-01, ENS-02, ENS-03)

</details>

<details>
<summary>✅ v1.2 Win Backtest Validation (Phases 8-10) — SHIPPED 2026-05-04</summary>

### Phase 8: Win Backtest Core
**Goal**: ユーザーが単勝モードのバックテストを実行し、正しい単勝ROI・的中率・バンクロール推移を得られる
**Depends on**: Phase 7 (v1.1 complete)
**Requirements**: WIN-01, WIN-02, WIN-03, WIN-04, WIN-05
**Plans**: 2 plans

Plans:
- [x] 08-01: Win payout map + final odds map + betting_target dispatch (WIN-01, WIN-02, WIN-04)
- [x] 08-02: Win candidate selection + Conformal confidence integration + WF validation (WIN-03, WIN-05)

### Phase 9: Win Reporting
**Goal**: ユーザーが単勝バックテスト結果のベット履歴・ROI診断・オッズバンド別内訳を確認できる
**Depends on**: Phase 8
**Requirements**: RPT-01, RPT-02, RPT-03
**Plans**: 1 plan

Plans:
- [x] 09-01: Win bet history fields + regime/odds band analysis + AI diagnostics + HTML/CLI extension (RPT-01, RPT-02, RPT-03)

### Phase 10: Pipeline Performance
**Goal**: バックテスト・学習パイプラインの実行時間が短縮され、ボトルネックが定量測定可能になる
**Depends on**: Phase 8
**Requirements**: PERF-01, PERF-02, PERF-03, PERF-04
**Plans**: 2 plans

Plans:
- [x] 10-01: Vectorize payout maps + groupby dict lookups (PERF-01, PERF-02)
- [x] 10-02: Feature cache + pyinstrument profiling (PERF-03, PERF-04)

</details>

<details>
<summary>✅ v1.3 Betting Strategy Optimization (Phases 11-13) — SHIPPED 2026-05-05</summary>

### Phase 11: Bet Selection Filters
**Goal**: 低信頼ベット・不安定レジーム・赤字オッズバンドを自動除外し、バックテストのベット品質が向上する
**Depends on**: Phase 10 (v1.2 complete)
**Requirements**: BSEL-01, BSEL-02, BSEL-03
**Plans**: 2 plans

Plans:
- [x] 11-01: OddsBandFilter クラス + EV_lower フィルター + RegimeDetector skip=True (BSEL-01, BSEL-03)
- [x] 11-02: Engine統合 (COLLAPSED skip + OddsBandFilter + counters + guard) + レポート除外統計 (BSEL-01, BSEL-02, BSEL-03)

### Phase 12: Stake Sizing Enhancement
**Goal**: レジーム状態に応じたKelly分数とEV比例乗算器により、高確信ベットに重点配分された賭け金が算出される
**Depends on**: Phase 11
**Requirements**: SIZE-01, SIZE-02
**Plans**: 2 plans

Plans:
- [x] 12-01: StakeCalculator コンストラクタ注入リファクタリング + apply_ev_scaling() 実装 (SIZE-01, SIZE-02)
- [x] 12-02: RegimeDetector/MetaSwitcher パラメータ追加 + Kelly→EV乗算→DD パイプライン統合 (SIZE-01, SIZE-02)

### Phase 13: Risk Calibration & Parameter Optimization
**Goal**: WIN向中率10%に最適化されたDD制御が動作し、ルックアヘッドバイアスを防いだ上で全戦略パラメータが最適化される
**Depends on**: Phase 12
**Requirements**: RISK-01, VAL-01, VAL-02
**Plans**: 3 plans

Plans:
- [x] 13-01: DrawdownController再設計 (ROI除去・3段階制御・コンストラクタ注入・ヒステリシス) (RISK-01)
- [x] 13-02: RegimeDetector外部化 + ParameterFreezeProtocol JSON manifest (VAL-01)
- [x] 13-03: StrategyOptimizer (Optuna TPE最適化・軽量WFループ・CLI) (VAL-02)

</details>

<details>
<summary>✅ v1.4 Ensemble Filter Recalibration (Phases 14-18) — SHIPPED 2026-05-07</summary>

- [x] Phase 14: Gate Recalibration (2/2 plans) — completed 2026-05-06
- [x] Phase 15: EV Filter Enhancement (2/2 plans) — completed 2026-05-06
- [x] Phase 16: Odds Band Rebuild (2/2 plans) — completed 2026-05-06
- [x] Phase 17: Optuna Optimization (2/2 plans) — completed 2026-05-06
- [x] Phase 18: Validation & Freeze (2/2 plans) — completed 2026-05-07

</details>

## Phase Details

_(v1.0-v1.3 phase details archived in respective milestone archives)_

_(v1.4 phase details archived in .planning/milestones/v1.4-ROADMAP.md)_

<details open>
<summary>🔄 v1.5 Model Accuracy Improvement (Phases 19-22) — ACTIVE</summary>

### Phase 19: EV推定キャリブレーション

**Goal**: P×E分解の独立性仮定に依存せず、OOF予測ベースでEVを直接キャリブレーションし、全セグメントのEV過大評価倍率を1.0±0.2に収束させる
**Depends on**: Phase 18 (v1.4 complete)
**Requirements**: EVC-01, EVC-02, EVC-03
**Plans**: 2 plans

Plans:
- [x] 19-01: OOF予測ベースのIsotonic EVキャリブレーション + オッズバンド別補正層 (EVC-01, EVC-02)
- [x] 19-02: EVCorrectionModel統合 + パイプライン適用 + テスト (EVC-03)

**Success Criteria:**
  1. IsotonicRegression で OOF ev_win を actual_return にキャリブレーションし、ECEが改善
  2. 高オッズ帯(20+)のEV過大評価倍率が2.08から1.2以下に改善
  3. 全セグメントのEV過大評価倍率が1.0±0.2に収束

### Phase 19.1: バックテスト高速化（Spike 002ボトルネック改善） (INSERTED)

**Goal**: Spike 002で特定したバックテスト実行時間ボトルネック（特徴量計算・学習・推論）を改善し、run_backtest.pyの実行時間を半減させる
**Depends on**: Phase 19 (v1.5 EV calibration complete)
**Plans**: 5 plans

Plans:
- [ ] 19.1-01: P0 キャリブレーションBT条件付きスキップ + P1 MLflow pip高速化 + P1 _coerce_types早期return
- [ ] 19.1-02: P1 oddsデータ受け渡し最適化 (preloaded_odds_ts)
- [ ] 19.1-03: P2 Categorical包括的適用 (データ層+特徴量ファイル) + odds接続
- [ ] 19.1-04: P2 observed=True追加 (モデル+バックテストファイル)
- [ ] 19.1-05: P3 HorseHistoryFeaturesキャッシュ強化 + per-race重複排除 + 段階的検証テスト

### Phase 20: 高オッズ的中パターン特徴量

**Goal**: 高オッズ帯(20+)の的中率を2.1%から3%+に引き上げる新特徴量を追加し、AbilityModelとWinTwoStageModelに統合する
**Depends on**: Phase 18 (v1.4 complete) — Phase 19と並行可能
**Requirements**: HODDS-01, HODDS-02, HODDS-03, HODDS-04, HODDS-05
**Plans**: 3 plans

Plans:
- [ ] 20-01: 高オッズ的中パターン分析 + クラストラジェクトリ/フォーム改善率特徴量 (HODDS-01, HODDS-02, HODDS-03)
- [ ] 20-02: 環境変化適性特徴量 + FeatureEngine統合 (HODDS-04, HODDS-05)
- [ ] 20-03: Feature importance分析 + モデルFEATURE_COLS更新 (HODDS-05)

**Success Criteria:**
  1. 10+新特徴量が生成され、欠損率10%以下
  2. Feature importance上位50%に新特徴量が含まれる
  3. 高オッズ帯(20+)のOOF予測AUCが改善 (ベースライン比較)

### Phase 21: Conformal EV予測区間

**Goal**: EV推定の不確実性をConformal Predictionで定量化し、信頼区間下界に基づくベット選択により、EV_excluded=0の問題を解消する
**Depends on**: Phase 19 (EV Calibration complete)
**Requirements**: CONF-01, CONF-02, CONF-03
**Plans**: 2 plans

Plans:
- [ ] 21-01: Conformal Prediction EV区間実装 + 動的フィルタリング (CONF-01, CONF-02)
- [ ] 21-02: パイプライン統合 + 診断レポート更新 (CONF-03)

**Success Criteria:**
  1. 90%信頼区間のカバレッジ率が90%以上
  2. EV_lower_bound < 1.0 のベットが除外され、EV_excluded > 0
  3. 従来フィルターとの互換性が維持

### Phase 22: 統合検証とバックテスト

**Goal**: 全改善を適用したバックテストでROI改善を確認する
**Depends on**: Phase 19, Phase 20, Phase 21
**Requirements**: VAL-01, VAL-02
**Plans**: 1 plan

Plans:
- [ ] 22-01: 統合バックテスト + EV診断 + セグメント別ROI検証 (VAL-01, VAL-02)

**Success Criteria:**
  1. バックテストROIが95%以上に改善
  2. 高オッズ帯(20+)のROIが50%以上に改善
  3. 既存テスト全通過 (回帰なし)

</details>

## Progress

**Execution Order:**
Phases execute in numeric order: 14 → 15 → 16 → 17 → 18

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
| 16. Odds Band Rebuild | v1.4 | 2/2 | Complete    | 2026-05-06 |
| 17. Optuna Optimization | v1.4 | 2/2 | Complete    | 2026-05-06 |
| 18. Validation & Freeze | v1.4 | 2/2 | Complete | 2026-05-07 |
| 19. EV推定キャリブレーション | v1.5 | 2/2 | Complete    | 2026-05-07 |
| 19.1. バックテスト高速化 | v1.5 | 0/5 | Planned | — |
| 20. 高オッズ的中パターン特徴量 | v1.5 | 0/3 | Active | — |
| 21. Conformal EV予測区間 | v1.5 | 0/2 | Active | — |
| 22. 統合検証とバックテスト | v1.5 | 0/1 | Active | — |
