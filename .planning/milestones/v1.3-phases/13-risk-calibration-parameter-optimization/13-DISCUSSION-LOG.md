# Phase 13: Risk Calibration & Parameter Optimization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-05
**Phase:** 13-Risk Calibration & Parameter Optimization
**Areas discussed:** DD制御WIN特化, Optuna最適化設計, パラメータ凍結・注入

---

## DD制御WIN特化

### ROLLING_WINDOW拡張方法

| Option | Description | Selected |
|--------|-------------|----------|
| コンストラクタ注入 (推奨) | DrawdownControllerのコンストラクタにrolling_windowを引数追加。Phase 12パターンと統一 | ✓ |
| 設定ファイル駆動 | settings.yamlにdd_controller sectionを追加 | |
| 固定値400にハードコード | 固定値変更、後でOptunaで探索 | |

**User's choice:** コンストラクタ注入 (推奨)
**Notes:** Phase 12のStakeCalculatorパターンと統一。Optuna最適化でも探索可能に

### 乗数テーブル再設計方針

| Option | Description | Selected |
|--------|-------------|----------|
| 単純化3段階 (推奨) | DD%のみの3段階(NORMAL/REDUCED/STOP)に整理。Optuna探索空間も小さく | |
| 現行構造維持・閾値調整 | 8行テーブル構造を維持して閾値を再調整 | |
| Optunaに全乘数を探索 | 各DDバンドの乘数を連続変数としてOptunaが探索 | |

**User's choice:** ベストプラクティスを追求（実装難易度問わず）
**Notes:** 結論: DD%のみ3段階 + ヒステリシス + 段階的リカバリ。ROI依存を完全に除去

### WIN/PLACE DD制御分離

| Option | Description | Selected |
|--------|-------------|----------|
| ベットタイプ別独立 (推奨) | WIN用とPLACE用で別々のDDControllerインスタンス | |
| 共通（1インスタンス） | 1つのDDControllerをWIN/PLACE共通で使用 | |

**User's choice:** ベストプラクティスを追求
**Notes:** WIN的中率10%とPLACE的中率30-40%で最適パラメータが全く異なるため、独立インスタンスが最適

### DDシグナル設計

| Option | Description | Selected |
|--------|-------------|----------|
| DD%のみ3段階 (推奨) | ROI依存を完全に除去し、DD%のみを主信号に | ✓ |
| DD%主+ROI補助 | DD%を主信号としつつROIも補助信号として保持 | |

**User's choice:** DD%のみ3段階 (推奨)
**Notes:** WIN 的率10%ではROIがノイジーすぎる。DD%は銀行ロール健全性の直接的指標

### DDリカバリパス

| Option | Description | Selected |
|--------|-------------|----------|
| 段階的リカバリ (推奨) | STOP→REDUCED→NORMALの段階的リカバリ + 最低滞在レース数 | ✓ |
| 即時復帰 | DD%閾値を下回ったら即座にNORMALに復帰 | |

**User's choice:** 段階的リカバリ (推奨)
**Notes:** 低的中率環境での発振防止

---

## Optuna最適化設計

### 探索空間の定義

| Option | Description | Selected |
|--------|-------------|----------|
| 全パラメータ一括 (推奨) | ~20次元を一括でTPE探索。クロスカテゴリ効果を捕捉 | |
| 段階別最適化 | フィルター→サイジング→DDの段階的最適化 | |
| 重要度フィルタリング | 予備解析で上位パラメータのみに絞る | |

**User's choice:** ベストプラクティスを追求
**Notes:** 結論: 全パラメータ一括。~16次元（COLLAPSED固定を除く）をTPEで100トライアル探索

### 目的関数

| Option | Description | Selected |
|--------|-------------|----------|
| ROI単一 (推奨) | バックテストROIのみを最大化 | |
| Sharpe比/複合スコア | ROI + バンクロール標準偏差の複合 | |
| ROI主 + ベット数制約 | ROI主目的 + 年間1000件以上のベット数制約 | |

**User's choice:** ベストプラクティスを追求
**Notes:** 結論: ROI主 + ベット数制約。ROI単一は過度なフィルタリングの危険あり

### WF評価方法

| Option | Description | Selected |
|--------|-------------|----------|
| Walk-forward枠組み (推奨) | 既存WalkForwardCVを拡張して各foldでOptuna評価 | |
| 単一train/test分割 | 80/20の単一分割 | |
| 時系列CV (5-fold) | 5-fold expanding window | |

**User's choice:** ベストプラクティスを追求
**Notes:** 結論: Walk-forward枠組み。ルックアヘッドバイアスを構造的に防止

### 試行数

| Option | Description | Selected |
|--------|-------------|----------|
| 100トライアル (推奨) | TPEで16次元なら100回で十分。MedianPruner付き | ✓ |
| 200-300トライアル | より徹底的だが計算時間倍増 | |
| Claudeの裁量 | データサイズ・次元数を考慮して自動設定 | |

**User's choice:** 100トライアル (推奨)
**Notes:** 16次元 × 100トライアル = TPEで十分な収束

---

## パラメータ凍結・注入

### 凍結形式

| Option | Description | Selected |
|--------|-------------|----------|
| JSON manifest (推奨) | 戦略パラメータをJSON + SHA256ハッシュ。人間可読・diff容易 | |
| 既存pickle拡張 | 既存のmodel pickle凍結を拡張 | |
| Optuna study保存 | OptunaのRDB/SQLite storageに履歴保存 | |

**User's choice:** ベストプラクティスを追求
**Notes:** 結論: JSON manifest + SHA256。人間可読・diff容易・改ざん検知

### RegimeDetectorパラメータ外部化範囲

| Option | Description | Selected |
|--------|-------------|----------|
| 主要パラメータのみ注入 (推奨) | fractional_kelly, ev_threshold, edge_threshold（各×3 = 9個）のみ | |
| 全パラメータ外部化 | 全25+パラメータを外部化 | |

**User's choice:** ベストプラクティスを追求
**Notes:** 結論: 主要パラメータのみ。ドメイン駆動パラメータ（runner-up rescue等）は固定値維持

### MetaSwitcher/RegimeDetector乖離解消

| Option | Description | Selected |
|--------|-------------|----------|
| RegimeDetectorに統合 (推奨) | MetaSwitcherの値をRegimeDetectorに揃える | |
| 独立維持 | それぞれ独立したまま | |
| Claudeの裁量 | コードベース状況を見て判断 | |

**User's choice:** ベストプラクティスを追求
**Notes:** 結論: MetaSwitcherのdefault_params値をRegimeDetectorに揃える。MetaSwitcher自体はリファクタリングしない（ライブパス変更回避）

### 設定注入パス

| Option | Description | Selected |
|--------|-------------|----------|
| コンストラクタ注入 (推奨) | Optuna最適化ではコンストラクタ引数で直接注入。settings.yamlはデフォルト値のみ | ✓ |
| 設定ファイル駆動 | settings.yaml → BacktestEngine → 各コンポーネント | |

**User's choice:** コンストラクタ注入 (推奨)
**Notes:** Phase 12 D-10/D-12と同じパターン。Optunaフィット性が高い

---

## Claude's Discretion

- DrawdownControllerのコンストラクタ引数の具体的なシグネチャ設計
- 3段階乗数テーブルの具体的なデータ構造
- ヒステリシスバンドの実装方法（状態マシン vs バンド幅パラメータ）
- WalkForwardCVへの戦略パラメータ注入インターフェース
- strategy_optimizer.py のクラス設計
- JSON manifestのスキーマ設計
- RegimeDetectorの主要パラメータ外部化のリファクタリング
- テスト戦略

## Deferred Ideas

None — discussion stayed within phase scope
