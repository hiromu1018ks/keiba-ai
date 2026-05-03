# Roadmap: keiba-ai Win Model Improvement

## Milestones

- ✅ **v1.0 Win Model** - Phases 1-4 (shipped 2026-05-03)
- ✅ **v1.1 ROI Advanced Model** - Phases 5-7 (shipped 2026-05-03)
- 📋 **v1.2** - TBD (next milestone)

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

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Feature Analysis & Enhancement | v1.0 | 2/2 | Complete | 2026-05-02 |
| 2. Win Benter Combination & Calibration | v1.0 | 2/2 | Complete | 2026-05-02 |
| 3. Selection Gate, Confidence & Betting | v1.0 | 2/2 | Complete | 2026-05-02 |
| 4. Walk-Forward Validation | v1.0 | 1/1 | Complete | 2026-05-03 |
| 5. Foundation Features | v1.1 | 2/2 | Complete | 2026-05-03 |
| 6. Odds Deviation EV | v1.1 | 1/1 | Complete | 2026-05-03 |
| 7. Ensemble Enhancement | v1.1 | 1/1 | Complete | 2026-05-03 |
