# Phase 21: Conformal EV予測区間 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-09
**Phase:** 21-Conformal EV予測区間
**Areas discussed:** CP手法の設計, 動的フィルタリング閾値, キャリブレーションデータ源, パイプライン統合設計

---

## CP手法の設計

### Q1: 既存RobustConfidenceEstimatorをどう改善するか

| Option | Description | Selected |
|--------|-------------|----------|
| 既存拡張 | 絶対値残差ベースを維持、正規化残差等を追加。変更少 | |
| CQRへの置き換え | CQRで分散正規化。理論的に最も厳密 | |
| 新クラス追加（2層） | 新ConformalEVModelを追加。既存に影響なし | |

**User's choice:** ベストプラクティスを追求（実装難易度問わず）→ CQR採用
**Notes:** ユーザーは一貫してベストプラクティス追求方針

### Q2: CQRの分位点モデルの実装方法

| Option | Description | Selected |
|--------|-------------|----------|
| QRベースCQR（推奨） | LightGBM quantile regressionで2分位点モデル学習 | ✓ |
| 残差/EV正規化 | QRモデル不要だが異分散性捕捉精度が劣る | |
| オッズバンド別独立CP | バンド別CP。サンプル不足リスク | |

**User's choice:** QRベースCQR（推奨）
**Notes:** α=0.1の場合、quantile=0.05と0.95の2モデル×2サーフェス=4モデル

### Q3: CQRのターゲット変数

| Option | Description | Selected |
|--------|-------------|----------|
| EV直接ターゲット | ev_win_calibratedを直接ターゲット。最も直接的 | |
| 残差ターゲット | 残差分布をモデリング。actual_returnはバックテスト時のみ | |
| P×E分離CQR | PとE別々にCQR。理論的だが複雑 | |

**User's choice:** ベストプラクティス追求 → EV直接ターゲット
**Notes:** CQR理論（Romano et al., 2019）でも予測対象そのものの区間推定が標準

### Q4: 信頼区間のalpha構成

| Option | Description | Selected |
|--------|-------------|----------|
| 単一alpha構成 | 90%のみ。シンプル | |
| 2-alpha構成（推奨） | 80%+90%。confidence_scoreとフィルタリング分離 | ✓ |

**User's choice:** ベストプラクティス追求 → 2-alpha構成（80%+90%）
**Notes:** 既存conformal_confidence_scoreとの互換性維持

---

## 動的フィルタリング閾値

### Q5: CQR区間下界に基づくベット除外の閾値戦略

| Option | Description | Selected |
|--------|-------------|----------|
| EV_lower < 1.0固定閾値（推奨） | JRA控除率後の損益分岐点。解釈性高い | |
| OOF分布ベース動的閾値 | Phase 15パターン。適応的だが解釈困難 | |
| ハイブリッド（固定+動的） | 固定基本+動的調整。複雑 | |

**User's choice:** ベストプラクティス追求 → EV_lower_90 < 1.0固定閾値 + OOF分布カバレッジ検証
**Notes:** 経済的解釈（EV<1.0=期待損失）が明確。CQRが既に異分散性を扱うため動的閾値不要

### Q6: CQR下界フィルターのパイプライン統合方法

| Option | Description | Selected |
|--------|-------------|----------|
| 既存フィルター統合（推奨） | EV_lower_win_corrected列を上書き。変更最小 | |
| 新規フィルター追加 | CQR専用フィルター。2段階構成 | |
| 完全置き換え | RobustConfidenceEstimator完全置き換え | |

**User's choice:** ベストプラクティス追求 → 完全置き換え
**Notes:** 新ConformalEVModelが同じ出力列を生成。race_predictor.pyは変更不要。デッドコード排除

---

## キャリブレーションデータ源

### Q7: CQRのキャリブレーションデータをどう生成するか

| Option | Description | Selected |
|--------|-------------|----------|
| OOF EV予測から生成（推奨） | Phase 19パターン。データリーク防止保証 | |
| バックテスト残差から生成 | 実際の推論結果ベース。バックテスト前は不可 | |
| OOF+バックテスト併用 | 両方の利点。実装複雑 | |

**User's choice:** ベストプラクティス追求。但しPhase 19のK-fold OOFはrun_backtest.pyを3時間+にしたため効率的な方法を指定
**Notes:** K-fold OOFの実行時間問題が重要な制約として明示された

### Q8: K-fold OOFを使わずにCQRを効率的に学習する方法

| Option | Description | Selected |
|--------|-------------|----------|
| バリデーション分割活用（推奨） | 既存後方20%分割。K-fold不要。追加推論1回 | ✓ |
| Phase 19 OOF再利用 | 追加計算なし。Phase 19保存が必要 | |
| キャリブレーションなし | QRのみ。理論的保証弱い | |

**User's choice:** バリデーション分割活用（推奨）
**Notes:** CP理論の標準アプローチ。実行時間への影響は最小（数秒〜数十秒）

---

## パイプライン統合設計

### Q9: CQR学習のTrainingPipelineV5統合

| Option | Description | Selected |
|--------|-------------|----------|
| 学習チェーン末尾に追加（推奨） | Isotonic → CQR。既存フローに統合 | |
| 独立メソッド化 | 別メソッド(_train_conformal_ev)。呼び出し追加必要 | |
| 別スクリプト化 | パイプライン外。別途実行 | |

**User's choice:** ベストプラクティス追求 → 学習チェーン末尾に追加
**Notes:** 依存関係が自然な順序で処理

### Q10: CQRモデルの保存先と保存形式

| Option | Description | Selected |
|--------|-------------|----------|
| SubmodelSet統合（推奨） | Phase 19パターン。PFP対象。自動読み込み | |
| グローバルモデル化 | RegimeDetectorと同位置。全サーフェス共通 | |
| 独立保存 | 別ディレクトリ。PFP対象外 | |

**User's choice:** ベストプラクティス追求 → SubmodelSet統合
**Notes:** .lgb形式、PFP改ざん検知対象、ModelLoader自動読み込み

### Q11: CQRの品質診断実装

| Option | Description | Selected |
|--------|-------------|----------|
| 既存EV診断拡張（推奨） | ev_diagnostics.pyにCQR指標追加。一元管理 | ✓ |
| 新規診断モジュール | CP専用診断。特化 | |
| レポート統合のみ | 最小実装 | |

**User's choice:** 既存EV診断拡張（推奨）
**Notes:** Success Criteria（90%カバレッジ率）の自動検証

---

## Claude's Discretion

- ConformalEVModelクラスの具体的なAPI設計
- LightGBM quantile regressionのハイパーパラメータ
- CQR非適合スコアの計算詳細
- サーフェス別CQRモデルのSubmodelSetフィールド命名規則
- バリデーション分割でのactual_return計算方法
- テストのfixtureデータとモック構成
- RobustConfidenceEstimatorの削除に伴う既存テストの移行

## Deferred Ideas

None — discussion stayed within phase scope
