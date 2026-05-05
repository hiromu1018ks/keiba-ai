# Roadmap: keiba-ai Win Model Improvement

## Milestones

- ✅ **v1.0 Win Model** - Phases 1-4 (shipped 2026-05-03)
- ✅ **v1.1 ROI Advanced Model** - Phases 5-7 (shipped 2026-05-03)
- ✅ **v1.2 Win Backtest Validation** - Phases 8-10 (shipped 2026-05-04)
- 🚧 **v1.3 Betting Strategy Optimization** - Phases 11-13 (in progress)

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

### 🚧 v1.3 Betting Strategy Optimization (In Progress)

**Milestone Goal:** バックテストROI 91.6% → 100%超えを達成するため、ベット選択の厳格化とステークサイジング最適化を実装する

- [x] **Phase 11: Bet Selection Filters** - Conformal信頼区間・レジームスキップ・オッズバンド除外でベット対象を厳格化
- [x] **Phase 12: Stake Sizing Enhancement** - レジーム別Kelly分数とEV比例乗算器で賭け金を最適化 (2026-05-05)
- [ ] **Phase 13: Risk Calibration & Parameter Optimization** - WIN向けDD調整 + パラメータ凍結 + Optuna最適化

## Phase Details

### Phase 11: Bet Selection Filters
**Goal**: 低信頼ベット・不安定レジーム・赤字オッズバンドを自動除外し、バックテストのベット品質が向上する
**Depends on**: Phase 10 (v1.2 complete)
**Requirements**: BSEL-01, BSEL-02, BSEL-03
**Success Criteria** (what must be TRUE):
  1. バックテスト実行時、EV_lower_win_corrected < 1.0 のベットが自動除外され、除外件数がログ/レポートに出力される (BSEL-01)
  2. RegimeDetectorがCOLLAPSEDと判定したレースでベット数が0になり、スキップレース数がレポートに記録される (BSEL-02)
  3. 過去バックテストROI分析で赤字のオッズバンドに該当するベットがOddsBandFilterで除外され、除外バンド・件数がレポートに出力される (BSEL-03)
  4. 全フィルター適用後の残存ベット数が年間1,000件以上を維持する（フィルター過剰除外のガード）
**Plans**: 2 plans

Plans:
- [x] 11-01: OddsBandFilter クラス + EV_lower フィルター + RegimeDetector skip=True (BSEL-01, BSEL-03)
- [x] 11-02: Engine統合 (COLLAPSED skip + OddsBandFilter + counters + guard) + レポート除外統計 (BSEL-01, BSEL-02, BSEL-03)

### Phase 12: Stake Sizing Enhancement
**Goal**: レジーム状態に応じたKelly分数とEV比例乗算器により、高確信ベットに重点配分された賭け金が算出される
**Depends on**: Phase 11
**Requirements**: SIZE-01, SIZE-02
**Success Criteria** (what must be TRUE):
  1. レジーム状態別にKelly分数が異なり、AGGRESSIVE > CONSERVATIVE > COLLAPSED(=0)の順で賭け金が計算される (SIZE-01)
  2. 高EVベットの賭け金にEV比例乗算器(min(ev/target_ev, max_scale))が適用され、同一レジーム内でEVが高いほど賭け金が大きくなる (SIZE-02)
  3. フィルター+サイジング変更後のバックテストROIがベースライン(89.0%)を上回る
**Plans**: 2 plans

Plans:
- [x] 12-01: StakeCalculator コンストラクタ注入リファクタリング + apply_ev_scaling() 実装 (SIZE-01, SIZE-02)
- [x] 12-02: RegimeDetector/MetaSwitcher パラメータ追加 + Kelly→EV乗算→DD パイプライン統合 (SIZE-01, SIZE-02)

### Phase 13: Risk Calibration & Parameter Optimization
**Goal**: WIN向中率10%に最適化されたDD制御が動作し、ルックアヘッドバイアスを防いだ上で全戦略パラメータが最適化される
**Depends on**: Phase 12
**Requirements**: RISK-01, VAL-01, VAL-02
**Success Criteria** (what must be TRUE):
  1. DrawdownControllerのローリングウィンドウが400+に拡張され、WIN的中率10%環境でDD乗数がNORMAL/REDUCED/STOPを適切に遷移する (RISK-01)
  2. ParameterFreezeProtocolが戦略パラメータ（Kelly分数・EV閾値・DD閾値・オッズバンド）を記録・固定し、最適化後のテスト期間でパラメータ変更を検出・警告する (VAL-01)
  3. Optuna TPEで全戦略パラメータの同時最適化が実行され、最適設定のバックテストROIがベースライン(89.0%)を上回る (VAL-02)
**Plans**: 3 plans

Plans:
- [ ] 13-01: DrawdownController再設計 (ROI除去・3段階制御・コンストラクタ注入・ヒステリシス) (RISK-01)
- [ ] 13-02: RegimeDetector外部化 + ParameterFreezeProtocol JSON manifest (VAL-01)
- [ ] 13-03: StrategyOptimizer (Optuna TPE最適化・軽量WFループ・CLI) (VAL-02)

## Progress

**Execution Order:**
Phases execute in numeric order: 11 → 12 → 13

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
| 13. Risk Calibration & Parameter Optimization | v1.3 | 0/3 | Not started | - |
