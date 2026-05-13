---
phase: 25-quick-win-wire-existing
reviewed: 2026-05-13T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - src/models/two_stage_return_model.py
  - src/paper_trading/predictor.py
  - tests/test_two_stage_return_model.py
  - tests/test_win_feature_analysis.py
findings:
  critical: 1
  warning: 3
  info: 3
  total: 7
fixed:
  critical: 1
  warning: 3
  total_fixed: 4
status: fixed
---

# Phase 25: Code Review Report

**Reviewed:** 2026-05-13T00:00:00Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** fixed (4/4 Critical+Warning findings auto-fixed)

## Summary

4ファイルをstandard深度でレビューした。2段階モデル（Win/Place）とPaper Trading予測器、および対応するテストコードが対象。

最も重要な発見は、`PlaceTwoStageModel._prepare_features()` にODDS-01特徴量（`deviation_rank`, `deviation_zscore`）の遅延計算が欠落していること。`RETURN_FEATURE_COLS` にこれらの列が定義されているのに、推論時にDataFrameに欠けていた場合のフォールバック計算が実装されていない。`WinTwoStageModel._prepare_features()` には同等の処理が存在するため、実装漏れと判断される。

**全 Critical + Warning (4件) を自動修正し、57テスト全て通過を確認済み。**

## Critical Issues

### CR-01: PlaceTwoStageModel に ODDS-01 deviation 特徴量の遅延計算が欠落 [FIXED]

**File:** `src/models/two_stage_return_model.py:448-473`
**Issue:** `PlaceTwoStageModel._prepare_features()` には、`deviation_rank`/`deviation_zscore` がDataFrameに存在しない場合の遅延計算（`compute_odds_deviation_features()` の呼び出し）が実装されていない。一方、`WinTwoStageModel._prepare_features()` (行186-191) には同等の処理が存在する。

`PlaceTwoStageModel.RETURN_FEATURE_COLS` の行422-423に `deviation_rank` と `deviation_zscore` が定義されているため、推論パスでこれらの列が入力DataFrameに含まれない場合、`available_cols` フィルタ (行464) で単に除外される。結果として、Return model の推論時にこれらの特徴量が黙って欠落し、LightGBM は当該特徴量をNaNとして扱い、予測精度が静かに劣化する。エラーや警告は一切出力されない。

**Fix applied:** `odds_to_ability_ratio` 計算ブロックの直後に `compute_odds_deviation_features()` の遅延呼び出しを追加（WinTwoStageModel と同じパターン）。

## Warnings

### WR-01: PlaceTwoStageModel.train_hit_model が fukuoddslow 列を暗黙的に要求 [FIXED]

**File:** `src/models/two_stage_return_model.py:508`
**Issue:** `self._val_fukuoddslow = df["fukuoddslow"].iloc[split:].values` が `fukuoddslow` 列に無条件でアクセスしている。もし渡されたDataFrameに `fukuoddslow` 列が含まれない場合、`KeyError` が発生する。docstring にこの前提条件が明記されていない。

同様に、`WinTwoStageModel.train_return_model` (行244) も `hit_df["confirmed_odds"]` に無条件アクセスしている。

**Fix applied:** `PlaceTwoStageModel.train_hit_model` と `WinTwoStageModel.train_return_model` の docstring に前提条件 (`fukuoddslow` / `confirmed_odds` 列が必須) を追記。

### WR-02: predictor.predict_race の race_id パースに長さ検証なし [FIXED]

**File:** `src/paper_trading/predictor.py:177`
**Issue:** `pd.Timestamp(f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}")` が `race_id` が8文字以上であることを前提としている。8文字未満の `race_id` が渡された場合、`IndexError` または不正なTimestampが生成される。防御的チェックが存在しない。

**Fix applied:** `len(race_id) < 8` の場合に警告ログを出力して空リストを返すガードを追加。

### WR-03: テストがクラス変数を変更し失敗時に復元されない [FIXED]

**File:** `tests/test_win_feature_analysis.py:170-178`
**Issue:** `TestRemoveNoiseFeatures.test_removes_specified_features` が `WinTwoStageModel.FEATURE_COLS` を変更した後、手動で復元している。もし `remove_noise_features()` 呼び出し (行174) または assert (行175-176) が失敗した場合、行178の復元は実行されず、後続テストにクラス変数の汚染が波及する。`pytest` がテストメソッド単位で実行順序を保証しないため、他のテストクラスにも影響する可能性がある。

**Fix applied:** `try/finally` パターンに変更し、テスト失敗時も確実に復元されるようにした。

## Info

### IN-01: predictor.predict_race の surface アクセスが無条件

**File:** `src/paper_trading/predictor.py:176`
**Issue:** `surface = result_df["surface"].iloc[0]` が無条件アクセスだが、`kyori` (行188) は条件付きアクセス（`if "kyori" in result_df.columns`）。一貫性の観点から、`surface` も条件付きアクセスにするか、両方とも必須であることを明示するとよい。

### IN-02: WinTwoStageModel._prepare_features の遅延インポートが関数内で毎回実行される

**File:** `src/models/two_stage_return_model.py:190-191`
**Issue:** `from features.odds_deviation_features import compute_odds_deviation_features` がメソッド呼び出しのたびに条件付きで実行される。推論時は毎回このパスが通る可能性がある。Python のモジュールキャッシュがあるため実用上の問題は小さいが、モジュールレベルの定数参照に比べてオーバーヘッドがある。

### IN-03: PlaceTwoStageModel.FEATURE_COLS が RETURN_FEATURE_COLS のコピーとして定義

**File:** `src/models/two_stage_return_model.py:443`
**Issue:** `FEATURE_COLS: list[str] = list(RETURN_FEATURE_COLS)` は後方互換のため Return model の列定義をコピーしている。新規コードでは `HIT_FEATURE_COLS` と `RETURN_FEATURE_COLS` のどちらを使うべきか判断が難しくなる可能性がある。docstringでの説明はあるが、`FEATURE_COLS` を直接参照する外部コードの挙動に注意が必要。

---
_Reviewed: 2026-05-13T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Fixed by: gsd-code-fixer_
_Depth: standard_
