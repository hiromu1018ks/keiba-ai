# Roadmap: keiba-ai Win Model Improvement

## Milestones

- ✅ **v1.0 Win Model** - Phases 1-4 (shipped 2026-05-03)
- 🚧 **v1.1 ROI Advanced Model** - Phases 5-7 (in progress)

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, ...): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

<details>
<summary>✅ v1.0 Win Model (Phases 1-4) - SHIPPED 2026-05-03</summary>

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

### 🚧 v1.1 ROI Advanced Model (In Progress)

**Milestone Goal:** 単勝バックテストROI 100%超えに向けて、アンサンブル・オッズ活用・特徴量改良の3本柱でモデル精度を大幅向上させる

- [x] **Phase 5: Foundation Features** - 時系列・ペース・オッズ変動の3系統の新特徴量を追加し、モデル入力の情報量を最大化する
- [ ] **Phase 6: Odds Deviation EV** - 市場オッズとモデル予測の乖離をEV計算に直接活用し、ベッティングエッジを定量化する
- [ ] **Phase 7: Ensemble Enhancement** - 3モデルスタッキングの多様性を強制し、予測精度の向上と過学習防止を両立する

## Phase Details

### Phase 5: Foundation Features
**Goal**: 過去走の時系列特徴量・展開予測特徴量・オッズ変動特徴量を追加し、後続のモデル改善がより豊かな入力から恩恵を受けられるようにする
**Depends on**: Phase 4 (v1.0 complete)
**Requirements**: TSER-01, TSER-02, TSER-03, PACE-01, PACE-02, ODTS-01, ODTS-02
**Success Criteria** (what must be TRUE):
  1. 過去走のタイム特徴量が指数減衰重み付けで計算され、直近の成績に高い重みが付与されていることをbacktest feature importanceで確認できる
  2. クラス調整済みフォーメトリックとz-score改善トラジェクトリが新特徴量としてfeature engineに組み込まれ、NaN率50%未満で生成される
  3. ペースフィグアが各馬のペース能力を数値化し、実績ベースのペース適性が既存のpace_scenario_fitを強化していることを確認できる
  4. オッズ変動の2次微分(加速度)と方向一貫性特徴量がodds_dynamics_featuresに追加され、steam moveの強さを検出できる
**Plans**: 2 plans

Plans:
- [x] 05-01: Time-series and pace features (TSER-01, TSER-02, TSER-03, PACE-01, PACE-02)
- [x] 05-02: Odds time-series features (ODTS-01, ODTS-02)

### Phase 6: Odds Deviation EV
**Goal**: モデル予測確率と市場オッズの乖離をEV信号としてモデルに直接組み込み、Conformal予測区間でベット選択の信頼性を最適化する
**Depends on**: Phase 5
**Requirements**: ODDS-01, ODDS-02, ODDS-03
**Success Criteria** (what must be TRUE):
  1. p_market/p_ability比率がStage2特徴量カラムとして追加され、バックテストfeature importance上位に位置していることを確認できる
  2. スタッキング出力がBenterGate→WinSelectionGateに正しく流れることをend-to-endテストで検証し、EV計算パイプラインの整合性が確保されている
  3. Conformal予測区間をEV区間に変換し、エッジ信頼性に基づくベット選択がConformal信頼性スコア付きで動作する
**Plans**: 1 plan

Plans:
- [x] 06-01: Odds deviation EV features and pipeline integration (ODDS-01, ODDS-02, ODDS-03)

### Phase 7: Ensemble Enhancement
**Goal**: 3モデルスタッキング(LightGBM+XGBoost+CatBoost)の多様性を強制するハイパーパラメータ最適化・early stopping・特徴量サブセット分割を実装し、予測精度を最大化する
**Depends on**: Phase 6
**Requirements**: ENS-01, ENS-02, ENS-03
**Success Criteria** (what must be TRUE):
  1. 各ベースモデル(LightGBM/XGBoost/CatBoost)に異なるハイパーパラメータ(lr, depth, rounds)が設定され、3モデル間の予測相関が0.95未満になっていることを検証できる
  2. 各ベースモデルにバリデーションベースのearly stoppingが追加され、単一モデルに対する過学習が防止されていることをOOF AUC推移で確認できる
  3. feature_fraction/colsample_bytree/rsmで各モデルに異なる特徴量サブセットが与えられ、アンサンブル多様性が向上していることをfeature importance分散で確認できる
**Plans**: 1 plan

Plans:
- [ ] 07-01: Ensemble stacking enhancement with forced diversity (ENS-01, ENS-02, ENS-03)

## Progress

**Execution Order:**
Phases execute in numeric order: 5 → 6 → 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Feature Analysis & Enhancement | v1.0 | 2/2 | Complete | 2026-05-02 |
| 2. Win Benter Combination & Calibration | v1.0 | 2/2 | Complete | 2026-05-02 |
| 3. Selection Gate, Confidence & Betting | v1.0 | 2/2 | Complete | 2026-05-02 |
| 4. Walk-Forward Validation | v1.0 | 1/1 | Complete | 2026-05-03 |
| 5. Foundation Features | v1.1 | 2/2 | Complete | 2026-05-03 |
| 6. Odds Deviation EV | v1.1 | 1/1 | Complete | 2026-05-03 |
| 7. Ensemble Enhancement | v1.1 | 0/1 | Not started | - |
