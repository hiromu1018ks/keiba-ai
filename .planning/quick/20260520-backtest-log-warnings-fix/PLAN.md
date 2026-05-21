---
name: backtest-log-warnings-fix
created: 2026-05-20
---

# Backtest Log Warnings Fix

## Summary
バックテスト実行時に出力される5つの警告ログのうち、即修正可能な3項目を修正する。

## Tasks

### Task 1: betacal typo 修正
- **File:** `src/models/win_benter_gate.py:278`
- **Fix:** `BetaCalibration(parameters="abc")` → `BetaCalibration(parameters="abm")`
- **Reason:** `"abc"` is invalid. Only `"abm"`, `"am"`, `"ab"`, `"a"` are valid. The 3-param version is `"abm"` (a, b, midpoint).
- **Impact:** betacalライブラリが正常動作し、scipy fallbackが不要になる。ログ警告が消える。
- **Test:** `python -m pytest tests/ -v -k "benter or calibration"`

### Task 2: kakuteijyuni=0 の取消馬を特徴量計算前に除外
- **File:** `src/features/feature_engine.py`
- **Fix:** `build_all()` メソッドの障害除外の直後（`_exclude_steeple` 処理の後）、`kakuteijyuni==0` の行を除外する。
- **Location:** 現在 `result_df = result_df[result_df["trackcd"] < 51]` の直後（~line 313）
- **Add:** `result_df = result_df[result_df["kakuteijyuni"] > 0]`
- **Reason:** 出走取消・競走除外馬はオッズが設定されず、popularity_rank が NaN になる。取消馬に特徴量を計算する意味がない。
- **Note:** `kakuteijyuni` 列は `preserve_columns` で渡されるため、この時点で存在する。ただし `build_all` の `preserve_columns=["kakuteijyuni", "confirmed_odds"]` で渡されるので、列が存在することを確認してからフィルタする。
- **Test:** `python -m pytest tests/ -v -k "feature_engine or popularity"`

### Task 3: サーフェス別学習時に定数特徴量（surface, surface_x_*）を除外
- **File:** `src/pipelines/training_pipeline.py`
- **Fix:** `_train_submodel()` 内で surface split 後、`surface` 列と `surface_x_*` で始まる列（すべて同じsurface値で定数）を特徴量から除外する。
- **Reason:** per-surface学習時、`surface`列は定数（"turf" or "dirt"）。すべての`surface_x_*`相互作用特徴量も定数。これらは情報量ゼロでモデル容量を無駄に消費し、3モデルが同じ不要特徴量を学習して多様性を損なう一因。
- **Implementation:** `two_stage_return_model._prepare_features()` が返す特徴量のうち、surface定数列を除外。または `training_pipeline.py` の `_train_submodel` 内で除外。
- **Test:** `python -m pytest tests/ -v -k "ensemble or training or submodel"`

## Scope
- Task 1-3 のみ。Optuna相関ペナルティ（④）は別phaseで対応。
- WinSelectionGate（⑤）は上流改善後に再評価。

## Verification
- `python -m pytest tests/ -v` が全パス
- `ruff check src/` がクリーン
