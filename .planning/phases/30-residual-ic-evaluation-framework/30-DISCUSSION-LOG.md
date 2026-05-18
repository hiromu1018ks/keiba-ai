# Phase 30: Residual IC Evaluation Framework - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 30-Residual IC Evaluation Framework
**Areas discussed:** コード配置, 入力データフロー, 出力とベースライン管理, 計算粒度とエントリポイント

---

## コード配置

| Option | Description | Selected |
|--------|-------------|----------|
| src/models/ に追加 | 既存診断モジュール(ev_diagnostics, drift_diagnostics)と同じ場所。コードベース一貫性を優先 | ✓ |
| 新規 src/evaluation/ ディレクトリ | 評価レイヤーをモデル層から分離。将来の評価機能追加も考慮 | |
| src/pipelines/ に統合 | OOF予測生成はtraining pipeline内で行われるためパイプラインの一部として配置 | |

**User's choice:** src/models/ に追加 (Recommended)
**Notes:** 既存パターンとの一貫性を重視

### モジュール構造

| Option | Description | Selected |
|--------|-------------|----------|
| 単一ファイル | ic_evaluator.py 1ファイルに全4定式化 + 方向一致チェック + ベースライン記録を含む。ev_diagnostics.pyパターン踏襲 | ✓ |
| 複数ファイル分割 | ic_formulas.py + ic_evaluator.py + ic_baseline.py に分割。Phase 30の規模に対しては過分解 | |

**User's choice:** 単一ファイル (Recommended)
**Notes:** RIC-01〜06の6要件に対して単一ファイルで十分な規模

---

## 入力データフロー

| Option | Description | Selected |
|--------|-------------|----------|
| OOF Parquet保存 + オフライン評価 | TrainingPipelineにOOF予測全列のParquet保存機能を追加。IC評価は保存済みファイルから読み込み | ✓ |
| パイプライン統合 | TrainingPipeline内でOOF予測が生成される際にIC評価を統合 | |
| バックテスト結果 | バックテスト結果からICを計算。OOFではないため前提と合わない | |

**User's choice:** ベストプラクティスを追求 → OOF Parquet保存 + オフライン評価
**Notes:** 再トレーニングなしで再分析可能、バージョン間比較も容易

### OOF保存列

| Option | Description | Selected |
|--------|-------------|----------|
| IC必要列のみ | race_id, race_date, umaban, surface, kakuteijyuni, p_win_pred, implied_probのみ | |
| OOF DataFrame全列 | 全列保存。将来の追加分析にも対応 | ✓ |

**User's choice:** OOF DataFrame全列
**Notes:** IC評価以外の分析にも再利用可能

### 市場確率取得

| Option | Description | Selected |
|--------|-------------|----------|
| OOF内既存列から取得 | implied_probがOOF DataFrameに含まれていれば直接利用 | |
| オッズデータから再計算 | IC評価時にオッズデータ(Parquet)から再計算 | |

**User's choice:** ベストプラクティスを追求 → OOF内既存列優先(1/tanoddsフォールバック付き)

---

## 出力とベースライン管理

| Option | Description | Selected |
|--------|-------------|----------|
| JSON固定ファイル | data/baseline/ic_baseline.jsonにIC値を保存。既存パターンと同じ | |
| MLflow metrics | MLflowにmetricsとして記録。実験管理との統合 | |
| 両方 (JSON + MLflow) | JSON即時保存 + MLflowオプション記録 | ✓ |

**User's choice:** ベストプラクティスを追求 → JSON + MLflow二重記録
**Notes:** MLflowが利用可能なら記録、JSONは常に出力

### 方向一致性チェック

| Option | Description | Selected |
|--------|-------------|----------|
| WARNING log + JSON警告フィールド | 4指標の符号一致を検証し矛盾があればWARNING。実行停止なし | |
| エラー停止 | 矛盾時は処理停止。初期実装では厳しすぎる | |
| コンソールのみ | コンソール出力のみ | |

**User's choice:** ベストプラクティスを追求 → WARNING log + JSON consistency_checkセクション + MLflowタグ

---

## 計算粒度とエントリポイント

### 計算粒度

| Option | Description | Selected |
|--------|-------------|----------|
| Surface別 + 全体 | turf/dirt別 + 全体の3パターン。モデルがsurface別学習のため有意 | ✓ |
| 全体のみ | surface区別なし。シンプルだが重要情報を見落とす | |

**User's choice:** Surface別 + 全体 (Recommended)

### エントリポイント

| Option | Description | Selected |
|--------|-------------|----------|
| 新規スクリプト | scripts/run_ic_eval.py。OOF Parquetを読み込んでIC評価を実行 | |
| run_train.py拡張 | --ic-evalフラグで学習後にIC評価。トレーニングが毎回必要 | |
| 両方 | モジュールAPI + スクリプト + パイプライン統合 | ✓ |

**User's choice:** ベストプラクティスを追求 → モジュールAPI + CLIスクリプト + TrainingPipeline統合
**Notes:** ロジックはモジュール、実行はスクリプトとパイプラインの両方から

---

## Claude's Discretion

- ic_evaluator.pyの内部関数構成 (各IC定式化の関数シグネチャ)
- OOF Parquetのファイル名規則
- JSON出力のスキーマ詳細
- テストケースの具体的な設計

## Deferred Ideas

None — discussion stayed within phase scope
