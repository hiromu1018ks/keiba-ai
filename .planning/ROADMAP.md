# Roadmap: keiba-ai Win Model Improvement

## Overview

単勝モデルのバックテストROIを89%から100%超えに引き上げる。現在の単勝パイプラインは複勝パイプラインに比べてアーキテクチャ的に不完全（Benter組み合わせ、キャリブレーション、選択ゲートが未実装）。4フェーズでこの差を埋め、ROI陽性を達成して検証する。

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Feature Analysis & Enhancement** - 既存特徴量の単勝寄与分析と新特徴量追加 ✓ 2026-05-02
- [x] **Phase 2: Win Benter Combination & Calibration** - 市場確率ブレンドとキャリブレーション実装 ✓ 2026-05-02
- [x] **Phase 3: Selection Gate, Confidence & Betting** - 関連性推定・ベッティング戦略統合 ✓ 2026-05-02
- [ ] **Phase 4: Walk-Forward Validation** - 多年度時系列検証でROI>100%を確認

## Phase Details

### Phase 1: Feature Analysis & Enhancement
**Goal**: 単勝予測に寄与する特徴量を特定し、ノイズを排除し、単勝特化の新特徴量を追加して、モデル入力の質を最大化する
**Depends on**: Nothing (first phase)
**Requirements**: FEAT-01, FEAT-02, FEAT-03
**Success Criteria** (what must be TRUE):
  1. SHAP/gain重要度ランキングが生成され、各特徴量の単勝予測への寄与度が定量的に把握できる
  2. odds-to-ability比、クラス落リバウンド、距離変更要検知、芝ダート変更要検知、勝利dominance等の新特徴量が5つ以上追加され、特徴量エンジンに統合されている
  3. SHAP分析に基づき、単勝予測に寄与しないノイズ特徴量が特定・除外され、特徴量数が最適化されている
  4. 新特徴量追加後のバックテストで、既存モデルと同等以上のlogloss/AUCを維持している
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — SHAP/gain特徴量重要度分析モジュールとノイズ特定・除外(FEAT-01, FEAT-03) ✓ 2026-05-02
- [x] 01-02-PLAN.md — 単勝特化新特徴量6つの実装・統合(FEAT-02) ✓ 2026-05-02

### Phase 2: Win Benter Combination & Calibration
**Goal**: 単勝予測に市場効率信号を組み込み、確率を正規化・キャリブレーションすることでEV推定精度を飛躍的に向上させる
**Depends on**: Phase 1
**Requirements**: BENT-01, BENT-02, BENT-03
**Success Criteria** (what must be TRUE):
  1. WinBenterGateが実装され、基本確率と市場確率のブレンド済み単勝確率が出力される
  2. Beta calibrationとIsotonic calibrationが比較評価され、単勝に最適な手法が採用されている
  3. Benter組み合わせ後の確率がレース単位で正規化され、各レースのP合計が1.0になる
  4. 信頼性ダイアグラム（reliability diagram）により、キャリブレーション品質がオッズバケット毎に視覚的に確認できる
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md — WinBenterGate実装・OOF予測生成・レース正規化・パイプライン統合(BENT-01, BENT-03) ✓ 2026-05-02
- [x] 02-02-PLAN.md — Beta/Isotonicキャリブレーション比較・ECE評価・信頼性ダイアグラム(BENT-02) ✓ 2026-05-02

### Phase 3: Selection Gate, Confidence & Betting
**Goal**: 学習済みゲートで低信頼レースを除外し、JRA控除率25%を考慮した最適ベッティング戦略を統合する
**Depends on**: Phase 2
**Requirements**: SELC-01, SELC-02, BETT-01
**Success Criteria** (what must be TRUE):
  1. PlaceSelectionGateパターンを踏襲したWinSelectionGateが実装され、単勝ベットのパス/リジェクト判定が学習済みモデルで行われる
  2. Conformal predictionに基づく信頼性推定が実装され、低信頼度レースがベット対象から除外される
  3. JRA控除率25%を考慮したエッジ閾値が設定され、fair oddsに対する真のエッジが計算される
  4. 統合後のバックテストで、ベット数が適切にフィルタリングされROIの改善が確認される
**Plans**: 2 plans

Plans:
- [x] 03-01-PLAN.md — WinSelectionGate実装・Conformal信頼性拡張・パイプライン統合 (SELC-01, SELC-02) (Wave 1) ✓ 2026-05-02
- [x] 03-02-PLAN.md — JRA控除率考慮edge_threshold更新・ベッティング戦略調整 (BETT-01) (Wave 2) ✓ 2026-05-02

### Phase 4: Walk-Forward Validation
**Goal**: 複数年度のウォークフォワード検証で過学習を検出し、ROI>100%が単年度の偶然でないことを証明する
**Depends on**: Phase 3
**Requirements**: VALI-01, VALI-02
**Success Criteria** (what must be TRUE):
  1. 2024-2025のウォークフォワード交差検証が実行され、各テスト年度のROIが個別に確認できる
  2. 訓練期間とテスト期間のROIギャップが分析され、過学習の兆候が評価されている
  3. 複数年度の加重平均ROIが100%を超えている
**Plans**: 1 plan

Plans:
- [ ] 04-01-PLAN.md — WFValidationResultデータクラス + 過学習検出ユーティリティ + run_wf_validation.py CLIスクリプト (VALI-01, VALI-02)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Feature Analysis & Enhancement | 2/2 | Complete | 2026-05-02 |
| 2. Win Benter Combination & Calibration | 2/2 | Complete | 2026-05-02 |
| 3. Selection Gate, Confidence & Betting | 2/2 | Complete | 2026-05-02 |
| 4. Walk-Forward Validation | 0/1 | Not started | - |
