---
status: resolved
trigger: "BacktestEngine.run() で3つのランタイムバグと1つの設計問題が実行時に発見された"
created: 2026-05-04
updated: 2026-05-04
---

# Debug Session: backtest-engine-multi-bugs

## Symptoms

**Expected behavior:**
- BacktestEngine.run() は betting_target (win/place/wide) に応じた払戻マップのみを構築する
- ワイド馬番は文字列型で一貫して扱われる
- LightGBM の特徴量数は学習時と予測時で一致する
- win モードで place_selection_reason 列にアクセスしてもエラーにならない

**Actual behavior:**
1. `build_wide_payout_map()` で ValueError -- ワイド馬番が float 型
2. `add_ability_probs()` で LightGBMError (feature count 62!=59) -- PaceAptitude 3列のマージ漏れ
3. `place_selection_reason` で KeyError -- win モード時の列不在
4. 設計問題: betting_target に関わらず複勝・単勝・ワイド全払戻マップを一括構築

**Error messages:**
- ValueError in build_wide_payout_map()
- LightGBMError: feature size mismatch (62 vs 59)
- KeyError: place_selection_reason

**Timeline:** 実行時に発生（run_backtest.py 実行で確認）

**Reproduction:** backtest 実行で発生

## Current Focus

hypothesis: "BacktestEngine.run() の初期化が無条件に全払戻マップを構築し、型変換・マージ・列参照の各所で betting_target を考慮していない"
next_action: "resolved"

## Evidence

- 2026-05-04: src/backtest/engine.py L183-196 -- paywidekumi columns may be float from Parquet; astype(str) produces "513.0" (length 4 with decimal), breaking length-based kumi parsing
- 2026-05-04: src/backtest/engine.py L540-543 -- _pace_cols only includes 3 of 6 pace features; PACE-01 cols (pace_corner_stability, pace_closing_power, pace_position_consistency) are excluded, causing feature count mismatch vs training pipeline (src/pipelines/training_pipeline.py L308-333 merges all 6)
- 2026-05-04: src/backtest/engine.py L669-686 -- when betting_target=="win", get_win_candidates() is called (L671-672) which does NOT produce place_selection_reason; then L684-686 unconditionally accesses place_selection_reason from candidate_df, causing KeyError
- 2026-05-04: src/backtest/engine.py L439-447 -- build_payout_map(), build_win_payout_map(), build_wide_payout_map() are all called unconditionally regardless of betting_target value

## Eliminated

## Resolution

root_cause: |
  Bug 1 (float kumi): Parquet stores paywidekumi as float64. When melted, astype(str) produces "513.0"
  instead of "513", breaking the length-based kumi parser.

  Bug 2 (feature mismatch 62 vs 59): BacktestEngine.run() only merges 3 pace_aptitude columns but
  omits 3 PACE-01 columns (pace_corner_stability, pace_closing_power, pace_position_consistency).
  Training pipeline merges all 6.

  Bug 3 (KeyError place_selection_reason): When betting_target=="win", get_win_candidates() returns
  a DataFrame without place_selection_reason. Engine then tries to extract this column unconditionally.

  Bug 4 (design): All payout maps built unconditionally regardless of betting_target.

fix: |
  Applied to src/backtest/engine.py:
  1. build_wide_payout_map() L198: use str.replace(r"\.0$", "") to strip ".0" from float-as-string
     while preserving zero-padded strings like "0102"
  2. L557-563: expanded _pace_cols from 3 to 6 columns (added pace_corner_stability,
     pace_closing_power, pace_position_consistency) matching training pipeline
  3. L704-708: guard place_selection_reason column extraction with column-presence check
     ("place_selection_reason" in candidate_df.columns)
  4. L439-463: conditionally build payout maps based on betting_target (needs_place/needs_win/needs_wide)

  All 1162 tests pass (77 backtest engine tests + full suite).
