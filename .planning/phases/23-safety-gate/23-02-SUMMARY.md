---
phase: 23-safety-gate
plan: 02
subsystem: feature-analysis
tags: [permutation-importance, feature-audit, all-models, sklearn]
dependency_graph:
  requires: [win_feature_analysis.py, LightGBM, sklearn]
  provides: [compute_permutation_importance, compute_all_model_importance, all-model CLI]
  affects: [scripts/analyze_feature_importance.py]
tech_stack:
  added: [sklearn.inspection.permutation_importance]
  patterns: [pivot importance table, auto scoring detection, model name to file prefix mapping]
key_files:
  created: []
  modified:
    - src/features/win_feature_analysis.py
    - scripts/analyze_feature_importance.py
decisions:
  - scoring="auto"でbinary/regression自動判定 (yのunique値で判定)
  - ターゲット未取得時はpermutation計算スキップしgainのみ出力
  - モデルファイル名プレフィックス→表示名の明示的マッピング (win_ret→win_return等)
metrics:
  duration: 1103s
  completed: "2026-05-12"
  tasks_completed: 2
  tasks_total: 2
  tests_passed: 1396
  tests_failed: 0
---

# Phase 23 Plan 02: Feature Importance Audit Infrastructure Summary

permutation importance + gain importanceを全モデルに対して計算する監査基盤を構築。sklearn permutation_importanceを追加し、CSV/JSONの両形式で包括的な特徴量評価を出力可能にした。

## Changes

### Task 1: win_feature_analysis.pyにpermutation importance追加 (66a1fb9)

- `compute_permutation_importance()`: sklearn permutation_importanceベースの計算関数
  - scoring="auto": yが{0,1}のみならneg_log_loss、それ以外はneg_mean_absolute_errorを自動選択
  - max_samples (default 5000) でのサブサンプリング対応
  - 戻り値: DataFrame with columns ["feature", "perm_importance_mean", "perm_importance_std"]
- `compute_all_model_importance()`: 全モデルのgain+permutation重要度一括計算
  - pivot_df: CSV出力用 (feature x model_gain/model_perm)
  - metadata_dict: JSON出力用 (モデル別gain/perm_mean/perm_std + メタデータ)
- 既存 `analyze_feature_importance` / `identify_noise_features` / `validate_noise_removal` は無変更 (後方互換)

### Task 2: analyze_feature_importance.pyを全モデル対応に拡張 (612838c)

- 新規CLI引数: `--all-models`, `--model` (7 choices), `--format` (csv/json/both), `--n-repeats`, `--output-json`
- `_find_model_file()`: model_name引数を受け取り、モデル名→ファイルプレフィックス自動マッピング
  - stage1→stage1, win_return→win_ret, ev_correction→ev_corrector_p 等
- `_load_features_for_analysis()`: 戻り値を (features_df, target_series) タプルに拡張
- `--all-models` モード: model_dir内の*.lgbファイルをglob検索、ファイル名からモデル名/surface推定
  - compute_all_model_importance呼び出しでピボットCSV + メタデータJSON出力
  - numpy型のPython型変換付きJSON保存 (_save_json)
- 既存 `--model-dir` 単体指定時の動作は完全に維持 (後方互換)

## Verification Results

- pytest: 1396 passed, 1 skipped, 0 failed
- ruff: src/features/win_feature_analysis.py and scripts/analyze_feature_importance.py pass
- --help: --all-models, --model, --format, --output-json, --n-repeats confirmed present
- Function signatures verified: compute_permutation_importance, compute_all_model_importance

## Deviations from Plan

None - plan executed exactly as written.

## Commits

| Commit | Message |
|--------|---------|
| 66a1fb9 | feat(23-02): permutation importance + all-model audit functions追加 |
| 612838c | feat(23-02): 全モデル対応のfeature importance監査CLI拡張 |
