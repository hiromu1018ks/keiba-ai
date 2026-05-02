# Requirements: keiba-ai Win Model Improvement

**Defined:** 2026-05-02
**Core Value:** 単勝モデルのバックテストROIを100%超えにすること

## v1 Requirements

### Feature Analysis

- [ ] **FEAT-01**: 既存特徴量のSHAP/gain重要度を分析し、単勝予測に寄与する特徴量とノイズ特徴量を特定する
- [ ] **FEAT-02**: 単勝特化の新特徴量を5つ以上追加する(odds-to-ability比、クラス落リバウンド、距離・芝ダート変更要検知、勝利dominance、フレッシュネス)
- [ ] **FEAT-03**: SHAP分析に基づき、単勝予測に寄与しないノイズ特徴量を特定し除外する

### Benter Combination & Calibration

- [ ] **BENT-01**: 単勝予測にBenter組み合わせ(基本確率×市場確率のブレンド)を実装する
- [ ] **BENT-02**: Beta calibrationとIsotonic calibrationを比較し、単勝に最適なキャリブレーション手法を採用する
- [ ] **BENT-03**: Benter組み合わせ後の確率をレース単位で正規化する(P合計=1.0)

### Selection & Confidence

- [ ] **SELC-01**: PlaceSelectionGateパターンを踏襲した単勝選択ゲート(学習済みバイナリフィルター)を実装する
- [ ] **SELC-02**: Conformal predictionに基づく信頼性推定を実装し、低信頼度レースを除外する

### Betting Strategy

- [ ] **BETT-01**: JRA控除率25%を考慮したエッジ閾値を設定・調整する

### Validation

- [ ] **VALI-01**: Walk-forward交差検証で過学習を検出・防止する
- [ ] **VALI-02**: 複数年度(2024-2025)のバックテストでROI > 100%を確認する

## v2 Requirements

### Betting Strategy (deferred)

- **BETT-02**: Kelly基準による最適賭け金計算(プールサイズ考慮、fractional Kelly)
- **BETT-03**: レジーム適応型ベッティング(単勝用レジームパラメータ)

### Ensemble (deferred)

- **ENSB-01**: StackedEnsembleのハイパーパラメータチューニング(Optuna)
- **ENSB-02**: WinTwoStageModel特徴量拡張(27→45+に拡大、place水準へ)

## Out of Scope

| Feature | Reason |
|---------|--------|
| 複勝/ワイドモデルの変更 | 単勝に集中するため |
| 新データ源の導入 | 既存EveryDB2データで十分 |
| 実馬券購入機能 | ペーパートレードまで |
| Web UI | CLIベースで十分 |
| リアルタイムオッズ収集の改善 | 既存機能をそのまま使用 |
| MAPIEライブラリ導入 | Conformal predictionは手動実装で対応可能 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FEAT-01 | Phase 1 | Pending |
| FEAT-02 | Phase 1 | Pending |
| FEAT-03 | Phase 1 | Pending |
| BENT-01 | Phase 2 | Pending |
| BENT-02 | Phase 2 | Pending |
| BENT-03 | Phase 2 | Pending |
| SELC-01 | Phase 3 | Pending |
| SELC-02 | Phase 3 | Pending |
| BETT-01 | Phase 3 | Pending |
| VALI-01 | Phase 4 | Pending |
| VALI-02 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0

---
*Requirements defined: 2026-05-02*
*Last updated: 2026-05-02 after roadmap creation*
