# Roadmap: keiba-ai Win Model Improvement

## Milestones

- ✅ **v1.0 Win Model** - Phases 1-4 (shipped 2026-05-03)
- ✅ **v1.1 ROI Advanced Model** - Phases 5-7 (shipped 2026-05-03)
- 🚧 **v1.2 Win Backtest Validation** - Phases 8-10 (in progress)

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

### 🚧 v1.2 Win Backtest Validation (In Progress)

**Milestone Goal:** バックテスト・WF検証を単勝ベースに修正し、実際に実行してROI>100%を確認する

- [x] **Phase 8: Win Backtest Core** - 単勝決済・候補選択・ベット生成の修正
- [x] **Phase 9: Win Reporting** - 単勝ベット履歴・ROI診断・オッズバンド分析
- [ ] **Phase 10: Pipeline Performance** - ベクトル化・groupby辞書・特徴量キャッシュ・プロファイリング

## Phase Details

### Phase 8: Win Backtest Core
**Goal**: ユーザーが単勝モードのバックテストを実行し、正しい単勝ROI・的中率・バンクロール推移を得られる
**Depends on**: Phase 7 (v1.1 complete)
**Requirements**: WIN-01, WIN-02, WIN-03, WIN-04, WIN-05
**Success Criteria** (what must be TRUE):
  1. バックテスト実行時、単勝払戻しデータ(paytansyoumaban1/paytansyopay1)から正確なpayout_mapが構築され、単勝ベットが正しい払戻金額で決済される
  2. バックテスト実行時、tanodds(単勝オッズ)ベースのfinal_win_odds_mapで単勝ベットのオッズ参照が行われる
  3. `--betting-target win` フラグでBacktestEngineが単勝/複勝モードを切り替えられる(デフォルト=win)
  4. WinSelectionGateのwin_selection_ev/edge/prob列に基づき、Conformal信頼性スコア付きの単勝ベット候補が生成される
  5. WF検証スクリプト(run_wf_validation.py)が単勝ROIで過学習検出を行う
**Plans**: 2 plans

Plans:
- [x] 08-01: Win payout map + final odds map + betting_target dispatch (WIN-01, WIN-02, WIN-04)
- [x] 08-02: Win candidate selection + Conformal confidence integration + WF validation (WIN-03, WIN-05)

### Phase 9: Win Reporting
**Goal**: ユーザーが単勝バックテスト結果のベット履歴・ROI診断・オッズバンド別内訳を確認できる
**Depends on**: Phase 8
**Requirements**: RPT-01, RPT-02, RPT-03
**Success Criteria** (what must be TRUE):
  1. バックテスト結果のJSON/レポートに各単勝ベットの馬番・オッズ・EV・的中結果が記録されている
  2. バックテスト終了時に単勝ROI・回収率・的中率・ベット数の集計診断が標準出力される
  3. レポートにオッズバンド別(人気1-3番人気・中穴4-6番人気・大穴7番人気以降)のROI内訳が表示される
**Plans**: 1 plan

Plans:
- [x] 09-01: Win bet history fields + regime/odds band analysis + AI diagnostics + HTML/CLI extension (RPT-01, RPT-02, RPT-03)

### Phase 10: Pipeline Performance
**Goal**: バックテスト・学習パイプラインの実行時間が短縮され、ボトルネックが定量測定可能になる
**Depends on**: Phase 8
**Requirements**: PERF-01, PERF-02, PERF-03, PERF-04
**Success Criteria** (what must be TRUE):
  1. build_payout_map()/build_wide_payout_map()のiterrows()がベクトル化pandas操作に置き換わり、マップ構築が高速化される
  2. レースごとのDataFrameフィルタリングがgroupby辞書の前処理に置き換わり、O(1)ルックアップでレースデータを取得できる
  3. HorseHistoryFeatures等の履歴特徴量がParquetキャッシュされ、バックテスト再実行時にキャッシュヒットすれば再計算をスキップできる
  4. pyinstrumentプロファイリングを統合し、バックテスト実行時のボトルネック関数と所要時間を定量測定できる
**Plans**: 2 plans

Plans:
- [x] 10-01: Vectorize payout maps + groupby dict lookups (PERF-01, PERF-02)
- [x] 10-02: Feature cache + pyinstrument profiling (PERF-03, PERF-04)

## Progress

**Execution Order:**
Phases execute in numeric order: 8 → 9 → 10

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
