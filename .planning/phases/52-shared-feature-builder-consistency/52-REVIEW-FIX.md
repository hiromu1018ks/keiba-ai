---
phase: 52-shared-feature-builder-consistency
fixed_at: 2026-06-06T06:41:50Z
review_path: .planning/phases/52-shared-feature-builder-consistency/52-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 3
skipped: 4
status: partial
---

# Phase 52: Code Review Fix Report

**Fixed at:** 2026-06-06T06:41:50Z
**Source review:** .planning/phases/52-shared-feature-builder-consistency/52-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (2 Critical, 5 Warning)
- Fixed: 3
- Skipped: 4

## Fixed Issues

### WR-02: PaperPredictor.setup() で surface ごとに race_df をフィルタリング

**Files modified:** `src/paper_trading/predictor.py`
**Commit:** 6d8c32b
**Applied fix:** `setup()` の surface ループ内で、各 `_surface_key` に該当するレースのみを `race_df`/`entry_df`/`odds_df`/`odds_ts_df` からフィルタリングしてから `build_for_inference()` に渡すよう変更。これにより turf/dirt 双方の submodel で同じレースが重複して特徴量生成される問題を解消。`surface` 列がない場合のフォールバックパスも維持。

### WR-03: session_manifest.py get_code_version() の PATH 依存を解消

**Files modified:** `src/features/session_manifest.py`
**Commit:** 5369c62
**Applied fix:** `shutil.which("git")` で git コマンドの可用性を事前チェックし、見つからない場合は即座に `RuntimeError` を送出。また、3箇所の `subprocess.run(["git", ...])` を `subprocess.run([git_path, ...])` に変更し、PATH 解決済みのパスを使用するよう修正。これにより Git for Windows がインストールされていない環境での `FileNotFoundError` がより明確なエラーメッセージで報告される。

### WR-04: DataCutoffManifest.verify() の日付比較を datetime.date に変更

**Files modified:** `src/features/data_cutoff_manifest.py`
**Commit:** 35e5fde
**Applied fix:** `verify()` メソッド内の日付比較を文字列比較から `datetime.date.fromisoformat()` によるパース比較に変更。不正フォーマット (`2024-1-5` vs `2024-01-05` 等) の場合でも正確な比較が可能。`prediction_date` または `actual` が不正フォーマットの場合は、警告ログを出力した上で文字列比較にフォールバックする安全な設計。

## Skipped Issues

### CR-01: FeatureBuilder に BloodlineFeatures が欠落

**File:** `src/features/feature_builder.py:204-420`
**Reason:** already_fixed (by design). BloodlineFeatures は FeatureEngine.build_all() Group B で暗黙的に実行される。_enrich_features() の docstring (L217-220) に明記済み。feat_df には build_all() 時点で既に blood_* カラムが含まれているため、FeatureBuilder での明示的なステップは不要。
**Original issue:** _enrich_features() に BloodlineFeatures が含まれていない

### CR-02: run_paper_trading.py の predict/diagnose/dry-run モードが FeatureBuilder を使用していない

**File:** `scripts/run_paper_trading.py:463-532,898-954,1211-1277`
**Reason:** already_fixed (Plan 04 GAP-1 closure). 全3モード (predict L498, diagnose L864, dry-run L1120) が `_build_features_fb()` ヘルパー経由で `FeatureBuilder.build_for_training()` を使用するよう修正済み。
**Original issue:** 3モードが FeatureEngine.build_all() + 手動エンリッチメントを使用している

### WR-01: FeatureBuilder に DamPedigreeFeatures が欠落している可能性

**File:** `src/features/feature_builder.py:302-315`
**Reason:** not_applicable. DamPedigreeFeatures は feature_builder.py L311-321 で (f) ステップとして正しく実装されている。merge キーは `[race_id, umaban]` で、DamPedigreeFeatures.compute() の出力も同じキーを使用。実際の問題は存在しない。
**Original issue:** DamPedigreeFeatures の merge キー不一致の可能性

### WR-05: training_pipeline.py _train_submodel の track_stats 計算の冗長性

**File:** `src/pipelines/training_pipeline.py:799-806`
**Reason:** skipped: intentional design, not a bug. _train_submodel 内の track_stats/track_month_stats (L826-833) は SubmodelSet コンストラクタ (L1378-1379) で永続化するために計算している。FeatureBuilder._enrich_features() でも内部的に計算されるが、その値は FeatureBuilder のスコープ内で消費され、外部には返されない。したがって _train_submodel での再計算は SubmodelSet のメタデータ永続化に必須。冗長ではあるが、機能的に正しく、修正によるリスク (SubmodelSet への伝播ミス) が恩恵を上回る。
**Original issue:** FeatureBuilder と _train_submodel で同じ track_stats が2回計算される

---

_Fixed: 2026-06-06T06:41:50Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
