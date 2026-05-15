# Phase 28: Validation & Freeze - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-15
**Phase:** 28-validation-freeze
**Areas discussed:** バックテスト構成, ROI未達時の対応, 特徴量凍結方法, 検証範囲

---

## バックテスト構成

### テスト年度

| Option | Description | Selected |
|--------|-------------|----------|
| マルチ年度(3年) | 2023/2024/2025。--train-window 4。~3時間。最も信頼性が高い | ✓ |
| 単一年度(2024) | 2024年のみ。~57分。1年のみでROI安定性不明 | |
| 2年度(2024-2025) | バランス型。~2時間 | |

**User's choice:** マルチ年度(3年)
**Notes:** Phase 22/v1.4の実績ある構成。信頼性を優先。

### strategy_manifest使用

| Option | Description | Selected |
|--------|-------------|----------|
| manifestあり | Optuna最適化済み16次元パラメータ使用。ROI最大化に寄与。~57分/年 | ✓ |
| manifestなし | デフォルトパラメータ。~41分/年 | |
| 両方比較 | 差を確認。~2x時間 | |

**User's choice:** manifestあり
**Notes:** Phase 17で最適化済みmanifestを信頼。

### バックテストフラグ構成

| Option | Description | Selected |
|--------|-------------|----------|
| --ensemble --calibration-bt --report | Phase 25 D-04と同じ。キャリブレーションBT + HTMLレポート含む | ✓ |
| --ensembleのみ | 最小構成。~41分/年。詳細分析なし | |
| Claude判断 | 最適な構成をClaudeに判断させる | |

**User's choice:** --ensemble --calibration-bt --report

### Optuna再最適化

| Option | Description | Selected |
|--------|-------------|----------|
| 再最適化する | 新特徴量(~50個)追加後に再最適化。~2.5h/trial追加 | |
| 既存manifestのまま | 既存manifestを使用。速い | ✓ |
| Claude判断 | 判断を委ねる | |

**User's choice:** 既存manifestのまま
**Notes:** 新特徴量追加後の再最適化は時間コストが高い。既存manifestを信頼する判断。

---

## ROI未達時の対応

### 対応方針

| Option | Description | Selected |
|--------|-------------|----------|
| 結果を記録して完了 | ROI結果を記録しPhase 28完了。次マイルストーンで改善 | ✓ |
| 再最適化を試す | Optuna再最適化を試す。~2.5h追加 | |
| 追加Phaseで対応 | 追加Phaseで更なる改善 | |

**User's choice:** 結果を記録して完了

### 評価基準

| Option | Description | Selected |
|--------|-------------|----------|
| 改善幅ベースで記録 | ROI絶対値より改善幅を重視。「v1.5: 84.4% → v1.6: XX% (+Y.Ypp)」 | ✓ |
| 100%到達のみ成功 | 厳密な成功/失敗判定 | |
| Claude判断 | 判断を委ねる | |

**User's choice:** 改善幅ベースで記録
**Notes:** 100%到達は目標だが、改善があれば有効性を確認できたとみなす。

---

## 特徴量凍結方法

### 記録方法

| Option | Description | Selected |
|--------|-------------|----------|
| PFPパターン踏襲 | Phase 13のJSON + SHA256パターン。sort_keys=True + indent=2。実績あり | ✓ |
| シンプルJSONダンプ | FEATURE_COLSをJSONにダンプ。改ざん検知なし | |
| Claude判断 | 判断を委ねる | |

**User's choice:** PFPパターン踏襲

### hash記録粒度

| Option | Description | Selected |
|--------|-------------|----------|
| モデル毎にhash | 各モデルのFEATURE_COLS毎にSHA256記録。7+モデル | ✓ |
| 統合hash 1つ | 全モデル統合の単一hash。シンプル | |

**User's choice:** モデル毎にhash
**Notes:** モデル間でFEATURE_COLSが異なるため、個別管理が適切。

---

## 検証範囲

### 検証スコープ

| Option | Description | Selected |
|--------|-------------|----------|
| 最小スコープ | pytest + バックテストのみ | |
| Feature importance追加 | pytest + BT + feature importance再計算 | ✓ |
| 全検証含む | pytest + BT + importance + WF検証(~4時間) | |

**User's choice:** Feature importance追加

### Feature importance実行方法

| Option | Description | Selected |
|--------|-------------|----------|
| 既存スクリプト使用 | analyze_feature_importance.py --all-models。Phase 23実績あり | ✓ |
| Claude判断 | 判断を委ねる | |

**User's choice:** 既存スクリプト使用

---

## Claude's Discretion

- バックテスト結果の具体的な分析・解釈
- Feature importanceの結果に基づく推奨事項の記述
- 凍結manifestファイルの出力パス
- テスト結果レポートのフォーマット
- ROADMAP.md/PROJECT.mdの更新内容

## Deferred Ideas

None — discussion stayed within phase scope
