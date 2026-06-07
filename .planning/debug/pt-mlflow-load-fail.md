---
status: resolved
trigger: "PT setup が失敗 — MLflow からモデルをロードできない。run_paper_trading.py が MLflow run_id のみサポートし、data/models/ からの直接ロード経路がない"
created: 2026-06-07
updated: 2026-06-07
---

# Debug Session: pt-mlflow-load-fail

## Symptoms

- **Expected behavior:** `run_paper_trading.py --mode setup --date YYYY-MM-DD` が data/models/ からモデルを正常ロードして PT セッションを初期化する
- **Actual behavior:** MLflow file store (file:///mlruns) が deprecated で search_runs() がランを返さず、--run-id 未指定だと mlflow_run_id=None → エラー
- **Error messages:** MLflow からランを検索できない、ModelLoader.load(run_id=None) が失敗
- **Timeline:** MLflow file store 非対応以降発生。モデル自体は data/models/ に正常保存されている
- **Reproduction:** `python scripts/run_paper_trading.py --mode setup --date 2026-06-07` (MLflow サーバー未起動 or file store にランなし)

## Root Cause Analysis (user-provided)

1. `run_train.py` は `mlflow.start_run()` + `_save_models_local()` → data/models/ にモデル保存済み
2. MLflow file store は deprecated、`search_runs()` がランを返さない
3. `run_paper_trading.py` の `_load_models()` は `ModelLoader.load(run_id=config.mlflow_run_id)` のみサポート
4. `--run-id` 未指定 → `mlflow_run_id=None` → エラー

## Proposed Fix

- `run_paper_trading.py` に `--models-dir` CLI 引数を追加
- `_load_models()` で `ModelLoader.load_from_dir()` 経由で `models_dir` をサポート
- モデル自体は data/models/ に存在するのでロード経路の追加のみで解決

## Current Focus

- **hypothesis:** _load_models() が MLflow run_id のみサポートしており、ローカルディレクトリからのロード経路が欠落している
- **test:** --models-dir data/models を渡して PT setup が正常動作するか確認
- **expecting:** MLflow に依存せず data/models/ から TrainedModelsV5 をロードできる
- **next_action:** run_paper_trading.py と ModelLoader のコードを読んで修正箇所を特定し、修正を適用する
- **reasoning_checkpoint:** ユーザーが原因チェーンを特定済み。ModelLoader.load_from_dir() は既に存在するため、_load_models() に経路を追加するだけで解決見込み

## Evidence

- (to be collected by session manager)

## Eliminated

- (none yet)

## Resolution

- **root_cause:** _load_models() in run_paper_trading.py only supported MLflow run_id loading via ModelLoader.load(run_id=...), with no CLI argument or code path to pass models_dir to ModelLoader.load(models_dir=...) despite load_from_dir() already existing
- **fix:** Added --models-dir CLI arg to run_paper_trading.py, models_dir field to PaperTradingConfig, and conditional models_dir/run_id routing in _load_models(). --run-id and --models-dir are mutually exclusive at the parser level.
- **verification:** 72 related tests pass (model_loader, paper_trading, config). 2630 total tests pass (7 pre-existing failures unrelated to change).
- **files_changed:**
  - scripts/run_paper_trading.py (added --models-dir arg, updated load_config and _load_models)
  - src/paper_trading/config.py (added models_dir field)
