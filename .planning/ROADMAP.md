# Roadmap: keiba-ai Win Model Improvement

## Milestones

- ✅ **v1.0 Win Model** - Phases 1-4 (shipped 2026-05-03)
- ✅ **v1.1 ROI Advanced Model** - Phases 5-7 (shipped 2026-05-03)
- ✅ **v1.2 Win Backtest Validation** - Phases 8-10 (shipped 2026-05-04)
- ✅ **v1.3 Betting Strategy Optimization** - Phases 11-13 (shipped 2026-05-05)
- 🚧 **v1.4 Ensemble Filter Recalibration** - Phases 14-18 (in progress)

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

### 🚧 v1.4 Ensemble Filter Recalibration (In Progress)

**Milestone Goal:** アンサンブルモデルの出力分布にフィルター群を適合させ、年間100+ベット・ROI>100%を達成する

- [ ] **Phase 14: Gate Recalibration** - WinSelectionGateをアンサンブルOOF予測で再学習し、分布ドリフト診断とフラグ伝播検証を完了する
- [ ] **Phase 15: EV Filter Enhancement** - EV_lower閾値をアンサンブルOOF分布に動的適合させ、EV推定精度を診断する
- [ ] **Phase 16: Odds Band Rebuild** - ルックアヘッドバイアス修正後、アンサンブルベースtraining_bet_historyでOddsBandFilterを再キャリブレーションする
- [ ] **Phase 17: Optuna Optimization** - アンサンブルモデルで14次元Optuna最適化を実行し、fold増強とパラメータ安定性検証を完了する
- [ ] **Phase 18: Validation & Freeze** - アンサンブルバックテストでROI>100%を確認し、パラメータを固定・改ざん検知を適用する

## Phase Details

### Phase 14: Gate Recalibration
**Goal**: WinSelectionGateがアンサンブルOOF予測で再学習され、quantile binとscore tableが新しい分布に適合し、use_ensembleフラグがパイプライン全体で正しく伝播されている状態になる
**Depends on**: Phase 13 (v1.3 complete)
**Requirements**: GATE-01, GATE-02, GATE-03
**Success Criteria** (what must be TRUE):
  1. run_backtest.py --ensemble 実行時にWinSelectionGateがアンサンブルOOF予測で再学習され、再計算されたprob_edges/edge_edges/odds_edgesが単一モデルと異なる値になる
  2. ks_2samp/wasserstein_distanceで単一モデルとアンサンブルのOOF確率分布を比較した診断レポートが出力され、ドリフト量が定量化されている
  3. use_ensemble=TrueがModelLoader→RacePredictor→BacktestEngine→WinSelectionGate全経路で一貫して伝播され、アンサンブルモデル推論結果が各コンポーネントに到達していることをテストで確認できる
**Plans**: 2 plans

Plans:
- [ ] 14-01: ドリフト診断モジュール作成 + パイプライン統合 + ゲート再学習検証テスト (GATE-01, GATE-02)
- [ ] 14-02: use_ensembleフラグ伝播統合テスト (GATE-03)

### Phase 15: EV Filter Enhancement
**Goal**: EV_lower閾値がアンサンブルOOF分布に基づく動的閾値に置き換わり、過剰除外が解消されるとともにEV推定精度が可視化されている状態になる
**Depends on**: Phase 14
**Requirements**: EVF-01, EVF-02
**Success Criteria** (what must be TRUE):
  1. バックテスト実行時にEV_lower閾値が固定1.0ではなくアンサンブルOOF分布の分位点から計算された動的値を使用し、除外件数が3,594件から大幅に減少する
  2. OOF EV推定値と実際の払戻額を比較した診断レポートが出力され、EV推定の過大/過小評価が定量化されている
**Plans**: TBD

### Phase 16: Odds Band Rebuild
**Goal**: strategy_optimizer.pyのルックアヘッドバイアスが修正され、アンサンブルモデルで生成されたtraining_bet_historyに基づいてOddsBandFilterが正しく再キャリブレーションされている状態になる
**Depends on**: Phase 15
**Requirements**: ODDS-01, ODDS-02
**Success Criteria** (what must be TRUE):
  1. strategy_optimizer.pyがtraining_bet_history生成にデフォルトパラメータを使用し、Optuna最適化済みパラメータが学習データに漏洩していないことをテストで確認できる
  2. OddsBandFilter.calibrate()がアンサンブルモデル由来のtraining_bet_historyで実行され、各オッズバンドのROIがアンサンブルの実際の精度を反映している
**Plans**: TBD

### Phase 17: Optuna Optimization
**Goal**: アンサンブルモデルで再キャリブレーション済みのフィルター群に対してOptuna 14次元最適化が実行され、fold増強とmulti-seed安定性検証を経て過学習耐性のある最適パラメータが導出されている状態になる
**Depends on**: Phase 16
**Requirements**: OPT-01, OPT-02, OPT-03
**Success Criteria** (what must be TRUE):
  1. Optuna TPE最適化がアンサンブルモデル + 再キャリブレーション済みフィルターで14次元探索を完了し、デフォルト値を上回るROIを達成する
  2. walk-forward fold数が2から4以上に増加し、過学習リスクが軽減されている（fold間ROI分散が閾値以下）
  3. 複数seedで最適化を実行した結果、上位パラメータの安定性が検証され、不安定な次元が特定・報告されている
**Plans**: TBD

### Phase 18: Validation & Freeze
**Goal**: アンサンブルバックテストで年間100+ベットかつROI>100%が確認され、最適化済みパラメータが改ざん検知付きで固定されている状態になる
**Depends on**: Phase 17
**Requirements**: VAL-01, VAL-02
**Success Criteria** (what must be TRUE):
  1. アンサンブルバックテスト(--ensemble)の結果が年間100+ベットを生成し、ROIが100%を超えている
  2. ParameterFreezeProtocolが最適化済みパラメータをJSON manifestに固定し、SHA256ハッシュで改ざん検知が有効になっている
**Plans**: TBD

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
| 14. Gate Recalibration | v1.4 | 0/2 | Planned | - |
| 15. EV Filter Enhancement | v1.4 | 0/? | Not started | - |
| 16. Odds Band Rebuild | v1.4 | 0/? | Not started | - |
| 17. Optuna Optimization | v1.4 | 0/? | Not started | - |
| 18. Validation & Freeze | v1.4 | 0/? | Not started | - |
