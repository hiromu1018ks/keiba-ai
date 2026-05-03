# Requirements: keiba-ai v1.1 ROI Advanced Model

**Defined:** 2026-05-03
**Core Value:** 単勝モデルのバックテストROIを100%超えにすること

## v1.1 Requirements

### Ensemble Stacking (ENS)

- [ ] **ENS-01**: 各ベースモデル(LightGBM/XGBoost/CatBoost)に異なるハイパーパラメータ(lr, depth, rounds)を設定し、モデル間の多様性を確保できる
- [ ] **ENS-02**: 各ベースモデルにバリデーションベースのearly stoppingを追加し、過学習を防止できる
- [ ] **ENS-03**: feature_fraction/colsample_bytree/rsm で各モデルに異なる特徴量サブセットを与え、アンサンブル多様性を向上できる

### Odds Deviation EV (ODDS)

- [ ] **ODDS-01**: p_market/p_ability の比率をStage2特徴量カラムとして追加し、ROIに直結するエッジ信号をモデルに学習させることができる
- [ ] **ODDS-02**: スタッキング出力がBenterGate→WinSelectionGateに正しく流れることを検証し、EV計算パイプラインの整合性を確保できる
- [ ] **ODDS-03**: Conformal予測区間をEV区間に変換し、エッジの信頼性に基づいてベット選択を最適化できる

### Odds Time-Series (ODTS)

- [x] **ODTS-01**: オッズ変動の2次微分(加速度)を計算し、steam moveの強さを特徴量として追加できる
- [x] **ODTS-02**: オッズ変動方向の一貫性を測定する特徴量を追加し、持続的スマートマネーの流入を検出できる

### Time-Series Features (TSER)

- [x] **TSER-01**: 過去走の全平均値特徴量を指数減衰重み付けに置き換え、直近の成績により高い重みを付与できる
- [x] **TSER-02**: form_trendを対戦相手のクラスレベルでコンテキスト化し、クラス調整済みフォーメトリックを算出できる
- [x] **TSER-03**: z-scoreの線形トレンドを計算し、トラック条件を正規化した改善トラジェクトリ特徴量を追加できる

### Pace Prediction (PACE)

- [x] **PACE-01**: コーナー位置と上がりタイムから総合ペースフィグアを算出し、各馬のペース能力を数値化できる
- [x] **PACE-02**: 実際のタイミングデータを用いて既存のpace_scenario_fitを強化し、宣言脚質だけでなく実績ベースのペース適性を評価できる

## v2 Requirements (Deferred)

### Ensemble

- **ENS-04**: Stage1 (AbilityModel Ranker) のスタッキング — 3つのRanker→メタRanker構成。複雑度高、ROI目標未達の場合に検討

### Odds

- **ODDS-04**: Kelly最適賭け金 — スタッキング確率を用いたKelly基準による賭け金最適化。ベッティング戦略はv1.2以降
- **ODTS-03**: Late money intensity (t-5 vs t-10) — オッズスナップショット粒度の検証が必要
- **ODTS-04**: Volume-weighted odds movement — 購入件数データの可用性未検証

### Pace

- **PACE-03**: 投影コーナーポジション — フィールドレベル相互作用モデリングが必要
- **PACE-04**: リアルタイムペースシミュレーション — v1.2以降

## Out of Scope

| Feature | Reason |
|---------|--------|
| 複勝/ワイドモデルの変更 | 単勝に集中するため |
| 新データ源の導入 | 既存EveryDB2データで十分 |
| ベッティング戦略高度化(Kelly/RegimeDetector) | v1.2以降で検討 |
| LSTM/Transformer時系列モデリング | 過去5-15走では深層学習は過学習リスク高 |
| 複雑メタラーナー(GBM/NN) | 特徴量3個ではRidgeが最適 |
| sklearn StackingClassifier | ネイティブブースティングAPIとPIT安全フォールドに非対応 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENS-01 | Phase 7 | Pending |
| ENS-02 | Phase 7 | Pending |
| ENS-03 | Phase 7 | Pending |
| ODDS-01 | Phase 6 | Pending |
| ODDS-02 | Phase 6 | Pending |
| ODDS-03 | Phase 6 | Pending |
| ODTS-01 | Phase 5 | Complete |
| ODTS-02 | Phase 5 | Complete |
| TSER-01 | Phase 5 | Pending |
| TSER-02 | Phase 5 | Pending |
| TSER-03 | Phase 5 | Pending |
| PACE-01 | Phase 5 | Pending |
| PACE-02 | Phase 5 | Pending |

**Coverage:**
- v1.1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0

---
*Requirements defined: 2026-05-03*
*Last updated: 2026-05-03 after v1.1 roadmap creation*
