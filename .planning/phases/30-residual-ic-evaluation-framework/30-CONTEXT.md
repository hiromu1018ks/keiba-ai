# Phase 30: Residual IC Evaluation Framework - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning

<domain>
## Phase Boundary

v1.6モデルの市場独立予測力を4定式化バッテリー(B差分IC/C直交IC/E Incremental IC/Per-race IC)で定量的に測定する評価フレームワークの構築。新特徴量の効果を客観的に評価できる状態にする。

**In scope:**
- RIC-01〜06 の実装 (4種IC指標計算 + 方向一致性検証 + ベースライン記録)
- TrainingPipelineへのOOF予測Parquet保存機能の追加
- IC評価モジュール (src/models/ic_evaluator.py)
- CLIスクリプト (scripts/run_ic_eval.py)
- テスト作成 (ic_evaluator + OOF保存)

**Out of scope:**
- 新特徴量の設計・実装 (Phase 31/32)
- バックテスト実行 (Phase 34)
- モデル構造の変更
- ETL拡張 (Phase 29)
- Gain per Depth診断 (Phase 33)

</domain>

<decisions>
## Implementation Decisions

### コード配置
- **D-01:** `src/models/ic_evaluator.py` — 単一ファイル、関数ベース。ev_diagnostics.py, drift_diagnostics.py のパターンを踏襲
- **D-02:** 新規ディレクトリ(src/evaluation/)は作成しない。既存診断モジュールとの一貫性を優先

### 入力データフロー
- **D-03:** TrainingPipelineにOOF予測DataFrame(全列)をParquetに保存する機能を追加。保存先: `data/oof/oof_predictions.parquet`
- **D-04:** IC評価は保存済みOOF Parquetから読み込んで実行(オフライン評価)。再トレーニングなしで再分析可能
- **D-05:** 市場確率(implied_prob)はOOF DataFrame内の既存列から取得。含まれない場合は`1/tanodds`からフォールバック計算

### 出力とベースライン管理
- **D-06:** IC評価結果をJSON + MLflow二重記録。JSON: `data/baseline/ic_baseline.json`、MLflow: metrics + tags
- **D-07:** JSONにはsurface別(turf/dirt) + 全体の3パターンのIC値を含む。各パターンに4指標(B差分/C直交/E Incremental/Per-race)
- **D-08:** 方向一致性チェック(RIC-06): WARNING log + JSON `consistency_check`セクション + MLflowタグ。実行停止なし、呼び出し元が重大度判断

### 計算粒度とエントリポイント
- **D-09:** Surface別(turf/dirt) + 全体の3パターンでIC計算。モデルがsurface別学習のため、surface別ICが最も有意
- **D-10:** モジュールAPI(`ic_evaluator.py`内の関数) + CLIスクリプト(`scripts/run_ic_eval.py`) + TrainingPipeline統合の3アクセスパス
- **D-11:** CLIスクリプトはOOF Parquetパスを受け取りIC評価を実行。パイプライン統合はOOF生成後のオプション呼び出し

### Claude's Discretion
- ic_evaluator.pyの内部関数構成 (各IC定式化の関数シグネチャ)
- OOF Parquetのファイル名規則
- JSON出力のスキーマ詳細
- テストケースの具体的な設計

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 既存診断モジュール (パターン参照)
- `src/models/ev_diagnostics.py` — EV推定精度診断。関数ベース、JSON出力、logging.getLoggerパターンの参照実装
- `src/models/drift_diagnostics.py` — ドリフト診断。ks_2samp/wasserstein_distance、JSON出力パターン

### OOF予測生成 (入力データ)
- `src/pipelines/training_pipeline.py` — OOF DataFrame生成箇所(oof_dfs, generate_ev_oof_predictions, generate_win_oof_predictions)。OOF保存フックの追加先
- `src/models/win_benter_gate.py::generate_win_oof_predictions()` — OOF予測生成の参照。p_win_pred/implied_probの列名確認
- `src/models/stacked_ensemble.py` — OOF予測生成パターンの参照(KFold、oof_pred)

### 要件定義
- `.planning/REQUIREMENTS.md` §Residual IC Evaluation — RIC-01〜06

### テストパターン
- `tests/test_ev_diagnostics.py` — 診断モジュールのテストパターン参照

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/models/ev_diagnostics.py`: 関数ベース診断モジュールのテンプレート。`_compute_ece()`, `_brier_decomposition()` 等の関数構成、JSON出力パターン
- `src/models/drift_diagnostics.py`: ドリフト診断の参照。`_compute_column_stats()`, `_compare_columns()` 等のパターン
- `src/pipelines/training_pipeline.py` lines 205-244: `oof_dfs` の蓄積と結合ロジック。OOF保存フックの追加箇所
- `scipy.stats.spearmanr`: Spearman順位相関の計算(Phase 30で使用)
- `sklearn.linear_model.LinearRegression` or `numpy.linalg.lstsq`: OLS残差計算(C直交IC用)

### Established Patterns
- 診断モジュールパターン: モジュールレベル定数 → 計算関数 → run_* オーケストレーション関数 → JSON出力
- OOF予測パターン: KFold分割 → モデル学習 → 予測 → oof_dfs蓄積 → full_features_df結合
- JSON出力パターン: `json.dump(result, f, indent=2, ensure_ascii=False)`
- loggingパターン: `logging.getLogger(__name__)` でモジュールロガー取得

### Integration Points
- `src/pipelines/training_pipeline.py::_train_submodel()`: OOF DataFrameのParquet保存フック追加先 (lines 205-244)
- `src/models/ic_evaluator.py` → `scripts/run_ic_eval.py`: CLIラッパーからの関数呼び出し
- `src/models/ic_evaluator.py` → `src/pipelines/training_pipeline.py`: パイプラインからのオプション呼び出し
- `data/oof/oof_predictions.parquet`: OOF予測の保存先(新規)
- `data/baseline/ic_baseline.json`: ICベースラインの保存先(新規)

</code_context>

<specifics>
## Specific Ideas

- OOF全列保存により、IC評価以外の分析(特徴量重要度との相関等)にも同じデータを再利用可能
- C直交IC(OLS残差)は最も厳しい評価基準。モデルが「市場と独立なエッジ」を持つことを証明する指標
- Per-race ICはレース内Spearmanの平均。サンプルサイズの小さいレース(少頭数)の取り扱いに注意
- E Incremental IC = IC(model, y) - IC(market, y) は直感的だが、B差分ICとは微妙に異なる性質を持つ
- 4定式化の方向一致性は実装バグ検出にも有用。矛盾があれば計算ミスの可能性が高い

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 30-Residual IC Evaluation Framework*
*Context gathered: 2026-05-18*
