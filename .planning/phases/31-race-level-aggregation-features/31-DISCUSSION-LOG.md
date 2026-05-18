# Phase 31: Race-Level Aggregation Features - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 31-Race-Level Aggregation Features
**Areas discussed:** build_features() パリティ, 特徴量昇格の対象モデル, rl_favorite_rank_gap 定義

---

## build_features() パリティ

| Option | Description | Selected |
|--------|-------------|----------|
| 共通関数抽出 | 新しい race_level_features.py に実装し両方から呼び出し | |
| インライン計算 | build_features() 内に直接記述 | |
| Claude に任せる | Claudeの判断で最適な方法を選択 | ✓ |

**User's choice:** Claude に任せる
**Notes:** ユーザーは実装詳細をClaudeに委ねた。推奨は共通関数抽出。

---

## 特徴量昇格の対象モデル

| Option | Description | Selected |
|--------|-------------|----------|
| 全モデルに追加 | 未登録の全12モデルのFEATURE_COLSに implied_prob_hhi / odds_skewness を追加 | ✓ |
| 主要予測モデルのみ | Stage1 + Win/Place 2Stage + EVCorrection に限定 | |
| Claude に任せる | Claudeの判断で最適な範囲を選択 | |

**User's choice:** 全モデルに追加
**Notes:** 一貫性を優先。全モデルで再学習が必要になることを了解済み。

---

## rl_favorite_rank_gap 定義

| Option | Description | Selected |
|--------|-------------|----------|
| インプライド確率差 | p_fav1 - p_fav2。直感的で解釈しやすい | |
| オッズ比 | odds_fav1 / odds_fav2。非対称スケール | |
| 対数オッズ差 | log(odds_fav2/odds_fav1)。対称的でLightGBMと相性良い | |

**User's choice:** 実装難易度は問わないのでベストプラクティスを追求
**Notes:** ベストプラクティスとして対数オッズ差を採用。Benter手法や金融工学の標準。対称性、モデル空間との整合、Kelly基準対応の利点。

---

## Claude's Discretion

- build_features() へのパリティ統合の具体的実装方法
- race_level_features.py の内部関数構成
- エッジケース処理 (少頭数レース、オッズ欠損等)
- テストケースの具体的な設計

## Deferred Ideas

None — discussion stayed within phase scope
