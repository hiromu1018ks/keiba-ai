# Phase 37: EV Calibration Layers - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-20
**Phase:** 37-EV Calibration Layers
**Areas discussed:** Pop band calibration ordering, Extended-window OOF strategy, Regime propagation architecture, Feedback loop test design

---

## Pop band calibration ordering

### Q1: キャリブレーション層の挿入位置

| Option | Description | Selected |
|--------|-------------|----------|
| Isotonic後 + Odds-band後 | P×E → Isotonic → Odds-band → Pop-band。最も外側で残差補正。 | ✓ |
| Isotonic後 + Odds-band前 | P×E → Isotonic → Pop-band → Odds-band。2つの残差補正が直列で相互作用。 | |
| Isotonic前 (生EVに対して) | P×E → Pop-band → Isotonic → Odds-band。Isotonic再学習が必要。 | |

**User's choice:** ベストな方法を選定してくれ（Claude discretion → Isotonic後 + Odds-band後）
**Notes:** Isotonic前だとOOF計算が複雑化。Odds-bandとPop-bandは独立した補正として最外層に配置。

### Q2: スケーリング係数の計算方式

| Option | Description | Selected |
|--------|-------------|----------|
| Median residual ratio | actual/calibratedの中央値。既存Odds-bandパターンと一致。外れ値に頑健。 | ✓ |
| Mean residual ratio | actual/calibratedの平均値。分布の非対称性を反映。外れ値に弱い。 | |
| Per-band LightGBM model | バンド別にLightGBM学習。データ不足リスク。 | |

**User's choice:** ベストプラクティスを追求（Claude discretion → Median residual ratio）

### Q3: 適用範囲

| Option | Description | Selected |
|--------|-------------|----------|
| Win + Place 両方 | Placeのハードコードpenaltyを残差ベースに統合可能。 | ✓ |
| Winのみ | 最小スコープ。Placeはスキップ。 | |
| 全ベットタイプ | Win/Place/Wide。Wideはスコープ外。 | |

**User's choice:** Win + Place 両方

### Q4: バンド境界の決定方法

| Option | Description | Selected |
|--------|-------------|----------|
| CAL-01の5バンド固定 | 1-3, 4-6, 7-9, 10-12, 13+。REQUIREMENTS指定。 | ✓ |
| Surface別動的バンド | 芝/ダート別に最適境界を自動決定。 | |
| データ駆動自動バンド | 分位点ベース等。解釈しにくい。 | |

**User's choice:** 実装難易度問わずベストプラクティス追求（Claude discretion → 5バンド固定境界 + surface別スケール係数）

---

## Extended-window OOF strategy

### Q5: OOF計算アプローチ

| Option | Description | Selected |
|--------|-------------|----------|
| Expanding window LOOC | 時系列expand。正確だが計算コスト高。 | ✓ |
| 既存5-fold OOF流用 | シンプルだがfold境界でわずかなルックアヘッド。 | |
| Rolling window OOF | 季節性対応。ウィンドウサイズチューニング必要。 | |

**User's choice:** 実装難易度問わずベストプラクティス追求（Claude discretion → Expanding window 5-fold OOF）

### Q6: Fold数

| Option | Description | Selected |
|--------|-------------|----------|
| 5 folds | 既存パターンと一貫。十分なサンプル数。 | ✓ |
| 8 folds | より正確だがfoldごとサンプル減。 | |
| 3 folds | foldごとサンプル多いが精度低下。 | |

**User's choice:** 5 folds

### Q7: fit_ev_calibration()との統合

| Option | Description | Selected |
|--------|-------------|----------|
| 既存サイクル拡張 | 同一OOF内でOdds-band + Pop-band同時計算。コスト追加なし。 | ✓ |
| 独立したOOFサイクル | 完全独立予測。学習時間倍増。 | |

**User's choice:** 既存サイクル拡張

---

## Regime propagation architecture

### Q8: regime_stateの伝播方法

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-compute regime → df column | BacktestEngineで事前計算、DataFrame列として渡す。最小変更。 | ✓ |
| predict()内でregime検出 | predict()にRegimeDetector注入。カプセル化。 | |
| EV補正内でregime検出 | 密結合。単一責任原則違反。 | |

**User's choice:** ベストプラクティスを追求（Claude discretion → Pre-compute regime → df column）

### Q9: regime_stateのエンコーディング

| Option | Description | Selected |
|--------|-------------|----------|
| Ordinal encoding (0/1/2) | 自然な順序。RegimeState.valueが既にint。 | ✓ |
| One-hot encoding | 3バイナリ列。列数増加。 | |
| Categorical type | LightGBM native categorical。順序仮定なし。 | |

**User's choice:** ベストな方法を選定（Claude discretion → Ordinal encoding）

### Q10: interaction featuresの計算方法

| Option | Description | Selected |
|--------|-------------|----------|
| 乗算相互作用 | surface*popularity_rank, market_entropy*surface。既存INTパターン。 | ✓ |
| カテゴリカル組み合わせ | surface×popularity_band(10パターン)。複雑。 | |

**User's choice:** 実装難易度問わずベストプラクティス追求（Claude discretion → 乗算相互作用。EVCorrectionModel.FEATURE_COLSのみに追加）

---

## Feedback loop test design

### Q11: テスト範囲

| Option | Description | Selected |
|--------|-------------|----------|
| Regime independence test | regime_stateを0/1/2に変更してEV出力比較。 | ✓ |
| EV non-influence test | EV出力操作でregime検出が変化しないかテスト。 | |
| 双方向テスト両方 | 完全な双方向検証。 | |

**User's choice:** Regime independence test

### Q12: 検証内容

| Option | Description | Selected |
|--------|-------------|----------|
| 相対変動率テスト | max_diff/median_ev < 5%。過度依存を検証。 | ✓ |
| 順位安定性テスト | Spearman rho > 0.99。順位が変わらないことを確認。 | |
| 両方 | より厳密。 | |

**User's choice:** 相対変動率テスト

---

## Claude's Discretion

- Q1: キャリブレーション層の挿入位置 → Isotonic後 + Odds-band後（最外層）
- Q2: スケーリング係数 → Median residual ratio
- Q4: バンド境界 → 5バンド固定 + surface別係数
- Q5: OOF計算 → Expanding window 5-fold
- Q8: regime伝播 → Pre-compute regime → DataFrame列
- Q9: エンコーディング → Ordinal encoding (0/1/2)
- Q10: interaction → 乗算相互作用、EVCorrectionModel.FEATURE_COLSのみ

## Deferred Ideas

None — all discussed items were within Phase 37 scope
