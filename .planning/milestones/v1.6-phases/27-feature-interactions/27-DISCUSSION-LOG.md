# Phase 27: Feature Interactions - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-15
**Phase:** 27-Feature Interactions
**Areas discussed:** INTER-01 (相対特徴量の拡張), INTER-02 (交互作用項の設計), INTER-03 (ターゲットエンコーディング)

---

## INTER-01: 相対特徴量の拡張

### Q1: 残り4個 + オッズ相対の扱い

| Option | Description | Selected |
|--------|-------------|----------|
| 残り4個追加 + オッズ相対を新規生成 | 残り4個をStage1+Stage2に追加 + オッズ相対特徴量を新規生成。INTER-01の「オッズ・能力値等」要件を完全に満たす | ✓ |
| 残り4個のみ追加 | 残り4個をStage1+Stage2に追加するが新しい相対特徴量は追加しない | |
| Claude's discretion | Claudeの判断に委ねる | |

**User's choice:** 残り4個追加 + オッズ相対を新規生成

### Q2: 既存race_rankとの関係

| Option | Description | Selected |
|--------|-------------|----------|
| 両方維持 | Stage1の既存race_rank 5個と新規相対特徴量は別物として両方維持。LightGBMに選ばせる | ✓ |
| 統合して片方を削除 | Stage1のrace_rankを新規relative_features.pyの実装に統一 | |
| Claude's discretion | Claudeの判断に委ねる | |

**User's choice:** 両方維持

### Q3: 新しいオッズ相対特徴量の内容

| Option | Description | Selected |
|--------|-------------|----------|
| オッズ + 能力値の相対 | オッズ相対 + 能力値相対の両方をStage2に追加 | ✓ (Recommended) |
| オッズ相対のみ | オッズ相対のみ追加 | |
| Claude's discretion | Claudeの判断に委ねる | |

**User's choice:** 実装難易度は問わないのでベストプラクティスを追求（オッズ + 能力値の相対を選択）

### Q4: 実装場所

| Option | Description | Selected |
|--------|-------------|----------|
| relative_features.py拡張 | _BASE_FEATURESに追加して既存パターンを踏襲 | ✓ |
| 新規モジュールを作成 | オッズ相対専用の新規モジュール | |
| Claude's discretion | Claudeの判断に委ねる | |

**User's choice:** relative_features.py拡張

---

## INTER-02: 交互作用項の設計

### Q1: 既存3個のカウント扱い

| Option | Description | Selected |
|--------|-------------|----------|
| 既存3個をカウント + 新規7-12個 | 合計10-15個にする | |
| 既存を含まず新規10-15個 | 合計13-18個になる | |
| Claude's discretion | Claudeの判断に委ねる | ✓ |

**User's choice:** Claude's discretion

### Q2: 表現方法

| Option | Description | Selected |
|--------|-------------|----------|
| カテゴリ積 + 数値積の混合 | 特徴量の性質に応じて適切な表現を選択 | ✓ (Recommended) |
| 数値積のみ | LightGBMが非線形組み合わせを自動学習 | |
| カテゴリ積のみ | 高次元化するがLightGBMのカテゴリ分割が効果的 | |
| Claude's discretion | Claudeの判断に委ねる | |

**User's choice:** 最新のデファクトスタンダード、ベストプラクティスを追求（カテゴリ積 + 数値積の混合を選択）

### Q3: 実装場所

| Option | Description | Selected |
|--------|-------------|----------|
| interaction_features.py拡張 | 既存モジュールに追加 | |
| 新規モジュールを作成 | 既存との分離で安全に追加 | |
| Claude's discretion | Claudeの判断に委ねる | ✓ |

**User's choice:** Claude's discretion

---

## INTER-03: ターゲットエンコーディング

### Q1: 対象変数

| Option | Description | Selected |
|--------|-------------|----------|
| 血統 + 騎手 + 調教師 | blood_keito_cd + 騎手コード + 調教師コード | ✓ |
| 血統のみ | 騎手・調教師は既存コンテキスト特徴量で表現済み | |
| Claude's discretion | Claudeの判断に委ねる | |

**User's choice:** 血統 + 騎手 + 調教師

### Q2: OOFリーク防止手法

| Option | Description | Selected |
|--------|-------------|----------|
| K-Fold TE (5-fold) | OOF予測の平均を取る。リーク防止が確実 | ✓ (Recommended) |
| Leave-One-Out TE | 1サンプルずつ除外して計算。最もリークが少ないが計算コストが高い | |
| Expanding window TE | PIT-safeに計算。時系列の前のデータのみ使用 | |
| Claude's discretion | Claudeの判断に委ねる | |

**User's choice:** 一番最適な方法を選定してくれ。リークは絶対しないように。（リーク防止最優先で最適手法を選択）

### Q3: 追加先モデル

| Option | Description | Selected |
|--------|-------------|----------|
| 全モデルに追加 | Stage1 + Stage2 + Place全モデルにTE列を追加 | |
| Stage2のみに追加 | Phase 25 D-02決定と同じ方針 | |
| Claude's discretion | Claudeの判断に委ねる | ✓ |

**User's choice:** 最適な方法を選択してくれ

### Q4: 実装場所

| Option | Description | Selected |
|--------|-------------|----------|
| 新規モジュール target_encoding.py | TEは独立した前処理ステップ | ✓ (Recommended) |
| 既存モジュールに組み込み | 血統TEはbloodline_features.pyに等 | |
| Claude's discretion | Claudeの判断に委ねる | ✓ |

**User's choice:** Claude's discretion

---

## Claude's Discretion

- INTER-02: 既存3個のカウント扱い + 新規追加数の決定
- INTER-02: 実装場所（interaction_features.py拡張 vs 新規モジュール）
- INTER-03: TE追加先モデル（Stage1 + Stage2 + Place）
- INTER-03: TE実装場所（新規target_encoding.py vs 既存モジュール組み込み）
- INTER-03: 平滑化パラメータ、最小サンプル数閾値の設定
- 各特徴量のFEATURE_COLSへの具体的な挿入位置

## Deferred Ideas

None — discussion stayed within phase scope
