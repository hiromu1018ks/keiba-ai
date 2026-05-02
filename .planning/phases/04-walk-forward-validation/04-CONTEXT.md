# Phase 4: Walk-Forward Validation - Context

**Gathered:** 2026-05-03
**Status:** Ready for planning

<domain>
## Phase Boundary

複数年度のウォークフォワード検証で過学習を検出し、ROI>100%が単年度の偶然でないことを証明する。Phase 1-3で実装した特徴量分析・Benterキャリブレーション・選択ゲートの妥当性を、未知データで検証する。

**In scope (from ROADMAP.md):**
- VALI-01: Walk-forward交差検証で過学習を検出・防止する
- VALI-02: 複数年度(2024-2025)のバックテストでROI > 100%を確認する

**Out of scope:**
- 新モデル・新特徴量の追加
- ベッティング戦略の変更
- 複勝/ワイドモデルの検証
- パイプラインの修正・最適化（過学習が検出された場合の対応は次マイルストーン）

</domain>

<decisions>
## Implementation Decisions

### Walk-forwardウィンドウ設計
- **D-01:** Expanding window方式を採用。学習データを最大化し、データ量が少ないMLでは標準的。既存WalkForwardCVのデフォルト動作と一致
- **D-02:** train_years=4, test_years=1を維持。Phase 1-3の検証設定と同一条件。Fold構成: 2020-2023学習→2024テスト、2021-2024学習→2025テスト
- **D-03:** 2フォールド(2024, 2025テスト)で実行。Success Criteriaの「2024-2025」に対応。実行時間~2時間
- **D-04:** WalkForwardCVを拡張して過学習検出と加重平均ROI計算を統合。CVResultに新フィールド(train_roi, test_roi, gap等)を追加する専用データクラスWFValidationResultを作成
- **D-05:** train期間もバックテストを実行しtrain ROIを取得。過学習検出にtrain-test ROI gapが不可欠。実行時間は倍になるがベストプラクティス
- **D-06:** 新規スクリプト `scripts/run_wf_validation.py` を作成。run_backtest.pyとは独立したエントリポイントで単一責任を担保

### 過学習検出基準
- **D-07:** 複合判定アプローチ: (1) ROI gap閾値 (2) 両年度ROI一貫性 (3) feature importance安定性 の3観点で総合評価
- **D-08:** ROI gap閾値は初回20%ポイントで実行し、結果を見て調整。train ROI - test ROI > 20%でWARNING、> 30%でFAIL
- **D-09:** Feature importanceの年度間比較を実装。トップ10特徴量の順位相関(Spearman)を計算し、ρ < 0.5で「特徴量依存の不安定性」をWARNING

### 加重平均ROI計算方法
- **D-10:** プールROI（総払戻額/総投資額）を主要指標とする。金融バックテストで最も誠実な指標
- **D-11:** ベット数加重ROIを参考指標として併記。年度間のベット数差が大きい場合の補完

### 結果レポート・PASS/FAIL判定
- **D-12:** JSON形式で検証結果を出力 + MLflow記録。年度別ROI、プールROI、過学習スコア、feature importance比較を含む
- **D-13:** 3基準の自動PASS/FAIL判定を実装: (1) 各年度ROI確認可能 (2) 過学習兆候評価 (3) 加重平均ROI>100%。全PASSで全体PASS
- **D-14:** 結果保存先: `data/backtest/wf_validation_result.json`

### Claude's Discretion
- ROI gap閾値の初期値と調整ロジックの詳細
- WFValidationResultデータクラスのフィールド設計
- Feature importance安定性評価の具体的な計算方法
- MLflow記録のメトリクス名とパラメータ
- PASS/FAIL判定のロジック詳細（WARNING/FAILの区分）
- JSONレポートのスキーマ設計

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### WalkForwardCV関連コード
- `src/models/walk_forward_cv.py` — 既存WalkForwardCVクラス。Fold/CVResultデータクラス、generate_folds()/run()メソッド。拡張対象
- `src/backtest/engine.py` — BacktestEngine。run()がBacktestResultを返す。train期間バックテストにも再利用
- `src/backtest/validation_suite.py` — BacktestValidationSuite。run_walk_forward_cv()が既存。参考実装

### バックテスト既存インフラ
- `scripts/run_backtest.py` — _run_multi_year()関数。マルチ年度バックテストの実装パターン。lines 366-535
- `src/backtest/race_predictor.py` — RacePredictor。予測パイプライン。各フォールドで使用
- `src/pipelines/training_pipeline.py` — TrainingPipelineV5。各フォールドの学習で使用

### 結果保存・MLflow
- `data/backtest/` — バックテスト結果保存先。wf_validation_result.jsonの配置先
- MLflow experiment tracking — 既存のmlruns/にWF検証結果を記録

### ドメインモデル
- `src/domain/models.py` — SubmodelSet, TrainedModelsV5。WFValidationResultの追加対象

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **WalkForwardCV** (`src/models/walk_forward_cv.py`): generate_folds()とrun()はそのまま再利用可能。CVResultを拡張してWFValidationResultを作成
- **BacktestEngine** (`src/backtest/engine.py`): run()でBacktestResult取得。train期間バックテストにも同じエンジンを使用
- **_run_multi_year()** (`scripts/run_backtest.py`): 年度別学習+バックテスト+JSON保存の実装パターンを流用
- **BacktestValidationSuite** (`src/backtest/validation_suite.py`): run_walk_forward_cv()の3-window WF CVパターンを参考

### Established Patterns
- **スクリプトエントリポイント**: `sys.path.insert(0, ROOT)` + `sys.path.insert(0, os.path.join(ROOT, "src"))` パターン
- **JSON結果出力**: `json.dumps(data, indent=2, ensure_ascii=False)` + Path.write_text パターン
- **MLflow記録**: mlflow.log_metrics/log_params パターン（training_pipeline.pyで既に使用）
- **データクラス拡張**: @dataclassで新フィールド追加（Phase 2, 3のSubmodelSet拡張パターン）

### Integration Points
- **WalkForwardCV.run()**: 各フォールドでpipeline.run() → engine.run() を実行。train期間のバックテストを追加
- **BacktestEngine**: 既存のまま使用。train期間とtest期間の両方でrun()を呼び出し
- **TrainingPipelineV5**: 既存のまま使用。各フォールドの学習
- **結果出力**: `data/backtest/wf_validation_result.json` に保存

</code_context>

<specifics>
## Specific Ideas

- ユーザーは「ベストプラクティスを追求」「学習時間や実装難易度は問わない」方針。品質優先で実装する
- 過学習検出はROI gap + feature importance安定性の複合評価
- プールROI（総払戻/総投資）を加重平均ROIの主要指標とする

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 4-Walk-Forward Validation*
*Context gathered: 2026-05-03*
