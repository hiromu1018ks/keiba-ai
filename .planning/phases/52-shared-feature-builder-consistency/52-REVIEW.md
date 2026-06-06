---
phase: 52-shared-feature-builder-consistency
reviewed: 2026-06-06T12:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - src/features/feature_manifest.py
  - src/features/feature_builder.py
  - src/features/pit_registry.py
  - src/features/data_cutoff_manifest.py
  - src/features/pipeline_consistency.py
  - src/features/session_manifest.py
  - src/backtest/engine.py
  - src/backtest/race_predictor.py
  - src/paper_trading/predictor.py
  - src/pipelines/training_pipeline.py
  - scripts/run_paper_trading.py
findings:
  critical: 2
  warning: 5
  info: 2
  total: 9
status: issues_found
---

# Phase 52: Code Review Report

**Reviewed:** 2026-06-06T12:00:00Z
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Phase 52 は FeatureBuilder による BT/PT/TrainingPipeline 間の特徴量生成統一を目的としている。新規モジュール (feature_manifest, feature_builder, pit_registry, data_cutoff_manifest, pipeline_consistency, session_manifest) は概ね堅牢に設計されているが、FeatureBuilder に欠落している特徴量モジュール (BloodlineFeatures, odds_deviation_features) があり、BT と PT/diagnose/dry-run パス間で特徴量不一致が発生する。また、run_paper_trading.py の predict/diagnose/dry-run モードは未だ FeatureBuilder を使用しておらず、Phase 52 の統一目標が完全には達成されていない。

## Critical Issues

### CR-01: FeatureBuilder に BloodlineFeatures が欠落 -- BT/Training と PT の特徴量不一致

**File:** `src/features/feature_builder.py:204-420`
**Issue:** `_enrich_features()` は13モジュール (a)-(n) を実行するが、`BloodlineFeatures` (血統特徴量、`blood_*` カラム) が含まれていない。`run_paper_trading.py` の `_run_predict` (L477) では `BloodlineFeatures` を個別に `feat_df.merge()` している。BacktestEngine の `prepare_data` / `run` では `FeatureBuilder.build_for_training()` を使用するため、`blood_*` カラムが `feat_df` に含まれなくなる。これにより BT と PT で特徴量が不一致となり、モデルの推論結果が変わる可能性がある。

**Fix:**
```python
# _enrich_features() に (o) BloodlineFeatures を追加
# (o) BloodlineFeatures
from features.bloodline_features import BloodlineFeatures

blood_feat = BloodlineFeatures(store=self.store)
blood_df = blood_feat.compute(entry_df)
_blood_merge_cols = [
    c for c in blood_df.columns
    if c not in df.columns or c in {"race_id", "umaban"}
]
if _blood_merge_cols:
    df = df.merge(blood_df[_blood_merge_cols], on=["race_id", "umaban"], how="left")
```

### CR-02: run_paper_trading.py の predict/diagnose/dry-run モードが FeatureBuilder を使用していない

**File:** `scripts/run_paper_trading.py:463-532,898-954,1211-1277`
**Issue:** `_run_predict` (L463), `_run_diagnose` (L898), `_run_dry_run` (L1211) はいずれも `FeatureEngine.build_all()` + 手動エンリッチメント (Sire/Pace/Course/Bloodline) を使用している。Phase 52 の設計意図は `FeatureBuilder` でこれらを統一することだが、このスクリプトは旧パスのまま。`paper_trading/predictor.py` の `setup()` は FeatureBuilder を使用するよう更新されているが、`_run_predict` は別コードパス (EveryDB2 から直接ロード) を実行するため、統一されていない。結果として、PaperPredictor.setup() 経由と _run_predict 経由で異なる特徴量が生成される。

**Fix:** `_run_predict`, `_run_diagnose`, `_run_dry_run` の特徴量生成を `FeatureBuilder.build_for_training()` に置き換える。

## Warnings

### WR-01: FeatureBuilder に DamPedigreeFeatures が欠落している可能性

**File:** `src/features/feature_builder.py:302-315`
**Issue:** `_enrich_features()` の (f) で `DamPedigreeFeatures` を実行しているが、merge キーが `[race_id, umaban]` であるのに対し、`run_paper_trading.py` の他のモジュールでは `kettonum` ベースの merge もある。DamPedigreeFeatures の出力に `kettonum` 列が含まれる場合、merge キーの不一致で結合漏れが発生する可能性がある。

**Fix:** DamPedigreeFeatures.compute() の出力スキーマを確認し、merge キーが一致していることを検証するテストを追加。

### WR-02: PaperPredictor.setup() で複数 surface の推論結果を pd.concat する際の重複

**File:** `src/paper_trading/predictor.py:92-110`
**Issue:** `setup()` は各 surface_key に対して `build_for_inference()` を呼び出し、結果を `pd.concat` している。しかし、`build_for_inference()` は `race_df` 全体 (全サーフェス) を入力として受け取るため、同じレースが turf/dirt 双方の submodel で重複して特徴量生成される可能性がある。surface ごとに該当するレースのみをフィルタリングせずにビルドすると、不要な特徴量計算と潜在的な重複行が発生する。

**Fix:** surface_key に基づいて `race_df` をフィルタリングしてから `build_for_inference()` に渡す。

### WR-03: session_manifest.py の get_code_version() で git コマンドが shell=False だが PATH 依存

**File:** `src/features/session_manifest.py:41-79`
**Issue:** `subprocess.run(["git", ...])` は `shell=True` ではないが、Windows 環境では `git.exe` の PATH 解決に依存する。Git for Windows がインストールされていない環境では `FileNotFoundError` が送出される。例外はキャッチされているが、エラーメッセージが `RuntimeError` で隠蔽される。

**Fix:** `git` コマンドのパスを設定可能にするか、`shutil.which("git")` で事前チェックする。

### WR-04: DataCutoffManifest.verify() が文字列比較で日付を検証している

**File:** `src/features/data_cutoff_manifest.py:61`
**Issue:** `if actual > self.prediction_date:` は文字列の辞書式比較 (YYYY-MM-DD 形式) に依存している。この形式では正しく動作するが、予期せぬフォーマット (e.g., `2024-1-5` vs `2024-01-05`) が入力された場合、比較結果が不正確になる。

**Fix:** `datetime.date` にパースしてから比較する。
```python
from datetime import date as date_type
actual_dt = date_type.fromisoformat(actual)
pred_dt = date_type.fromisoformat(self.prediction_date)
if actual_dt > pred_dt:
```

### WR-05: training_pipeline.py の _train_submodel が FeatureBuilder を使用していない

**File:** `src/pipelines/training_pipeline.py:799-806`
**Issue:** `TrainingPipeline.run()` は L376-390 で `FeatureBuilder.build_for_training()` を使用して特徴量を生成するよう更新されているが、`_train_submodel()` 内 (L819-833) では `track_stats` / `track_month_stats` を個別に `_compute_track_stats()` / `_compute_track_month_stats()` で計算している。FeatureBuilder の `build_for_training()` でも track_stats が学習時データから計算される (L347-353) ため、同じ統計が2回計算される。不整合はないが冗長。

**Fix:** `_train_submodel()` 内の track_stats 計算を、FeatureBuilder で計算済みの値を df から抽出する形に変更する。

## Info

### IN-01: _enrich_features() 内のインラインインポート

**File:** `src/features/feature_builder.py:216-418`
**Issue:** 全13モジュールが関数内で `from features.xxx import Yyy` としてインポートされている。これは循環インポート回避の一般的なパターンだが、初回呼び出し時のレイテンシとモジュール依存関係の可視性低下を招く。

**Fix:** 遅延インポートの意図が明示的になるようコメントを追加する程度で十分。

### IN-02: FeatureBuildResult が frozen=True だが frame 属性が mutable

**File:** `src/features/feature_manifest.py:129-145`
**Issue:** `FeatureBuildResult` は `frozen=True` だが、`frame: pd.DataFrame` は mutable なので、`result.frame.drop(..., inplace=True)` が可能。`build_for_inference()` L127 で実際にこの操作を行っている。dataclass の不変性はフィールドの再代入のみを防止し、mutable 属性の内容変更は防止しない。

**Fix:** 設計上の意図なので現状維持で問題ないが、docstring に "frame は内容が変更される可能性がある" 旨を追記するとよい。

---

_Reviewed: 2026-06-06T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
