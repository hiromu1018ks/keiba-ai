# Phase 2: Win Benter Combination & Calibration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-02
**Phase:** 02-Win Benter Combination & Calibration
**Areas discussed:** Benter入力の設計, キャリブレーション手法選択, レース正規化の設計, Placeパターンからの逸脱

---

## Benter入力の設計

### fundamental確率の段階

| Option | Description | Selected |
|--------|-------------|----------|
| 2Stage EV補正後 | WinTwoStageModel.predict_ev() → EVCorrection後の確率 | ✓ |
| Stage1生の確率 | AbilityModelのOOF予測そのまま | |
| WinHit生の確率 | Stage2通過前のP(hit)推定 | |

**User's choice:** 2Stage EV補正後 (Recommended)
**Notes:** Placeと同じ方式。Benterの役割が「市場信号の追加」に明確化される

### 市場確率ソース

| Option | Description | Selected |
|--------|-------------|----------|
| 最終オッズ (tanoddslow) | Placeのfukuoddslowと同じ方式 | ✓ |
| 中間オッズ | 複数スナップショット | |
| ハイブリッド | 学習=最終、推論=切替可 | |

**User's choice:** 最終オッズ (tanoddslow)
**Notes:** ユーザーは「最終オッズがリークにならないならいい」と確認。リークではない（レース前オッズ）ことを説明して合意

### 市場確率の前処理

| Option | Description | Selected |
|--------|-------------|----------|
| 1/oddsそのまま | βパラメータが控除率を吸収 | ✓ |
| 控除率補正 (0.75/odds) | 理論的には正しい | |

**User's choice:** 1/oddsそのまま (Recommended)

### Benter学習データ生成方法

| Option | Description | Selected |
|--------|-------------|----------|
| 既存ensembleパスに依存 | use_ensemble=True必須 | |
| Benter専用OOF予測生成 | use_ensembleに依存しない | ✓ |

**User's choice:** Benter専用OOF予測生成
**Notes:** ユーザー「実装難易度は問わない、ベストプラクティスを追求」

---

## キャリブレーション手法選択

### Beta vs Isotonicの方針

| Option | Description | Selected |
|--------|-------------|----------|
| 両方実装＋比較 | BENT-02要件を直接満たす | ✓ |
| Betaのみ | IsotonicはPlaceで失敗済み | |
| TemperatureScalingのみ | 最小実装 | |

**User's choice:** 両方実装＋比較 (Recommended)

### キャリブレーションパイプライン構成

| Option | Description | Selected |
|--------|-------------|----------|
| Benter → Calib → TempScale | 3段階パイプライン | ✓ |
| Benterのみ（Calib省略可） | バックテスト結果で判断 | |
| Benter → Calib のみ | TempScale省略 | |

**User's choice:** ベストプラクティスを追求
**Notes:** 推奨パイプライン raw_p → Benter → {Beta|Isotonic} → TempScale(オプション) で決定。TempScaleは追加改善がある場合のみ

### 評価指標

| Option | Description | Selected |
|--------|-------------|----------|
| Brier Score + ECE | 定量的・再現可能 | ✓ |
| バックテストROI | 実用的だが他要因混在 | |
| 全指標総合評価 | Brier+ECE+ROI | |

**User's choice:** Brier Score + ECE (Recommended)

---

## レース正規化の設計

### 正規化方式

| Option | Description | Selected |
|--------|-------------|----------|
| 単純正規化 (P/ΣP) | シンプル・堅牢 | ✓ |
| Benter最適化内に制約 | 理論的だが複雑 | |
| Softmax再スケール | 追加パラメータ必要 | |

**User's choice:** ベストプラクティスを追求
**Notes:** 単純正規化(P/ΣP)で決定。Benter学習は馬単位、正規化は後処理として独立適用

---

## Placeパターンからの逸脱

### アーキテクチャ

| Option | Description | Selected |
|--------|-------------|----------|
| 新クラス WinBenterGate | Placeコードに影響なし | ✓ |
| 既存BenterCombinationの拡張 | bet_typeパラメータで切替 | |
| 既存Benterをそのまま流用 | 入力だけ差し替え | |

**User's choice:** 新クラス WinBenterGate (Recommended)

### SubmodelSet統合方法

| Option | Description | Selected |
|--------|-------------|----------|
| win_* フィールド追加 | Placeと並列構造 | ✓ |
| 汎用辞書 bet_type→model | Placeリファクタリング必要 | |
| 既存フィールド共用 | Win/Place独立学習不可 | |

**User's choice:** win_* フィールド追加 (Recommended)

### 最適化パラメータ

| Option | Description | Selected |
|--------|-------------|----------|
| Win固有パラメータ調整 | 初期値をWin最適化 | |
| Placeと同じパラメータ範囲 | 最適化が解決 | |
| グリッドサーチで初期値探索 | 最も確実・実行時間増 | ✓ |

**User's choice:** グリッドサーチで初期値探索

---

## Claude's Discretion

- キャリブレーションパイプラインの詳細実装（各ステップ有効/無効判定）
- グリッドサーチの範囲・粒度設定
- TempScale適用閾値
- 信頼性ダイアグラムの出力形式

## Deferred Ideas

None — discussion stayed within phase scope
