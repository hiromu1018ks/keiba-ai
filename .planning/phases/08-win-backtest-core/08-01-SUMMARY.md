---
phase: 08-win-backtest-core
plan: 01
subsystem: backtest
tags: [win-payout, settle-bet, betting-target, cli, etl-sql]
dependency_graph:
  requires: []
  provides: [build_win_payout_map, final_win_odds_map, _settle_bet-win-branch, betting_target-param, --betting-target-cli, get_payouts-win-columns]
  affects: [engine.py, run_backtest.py, everydb2_queries.py]
tech_stack:
  added: []
  patterns: [payout-map-construction, betting-target-dispatch, tdd-red-green]
key_files:
  created: []
  modified:
    - src/backtest/engine.py
    - scripts/run_backtest.py
    - src/db/everydb2_queries.py
    - tests/test_backtest_engine.py
    - tests/test_run_backtest_args.py
    - tests/test_everydb2_queries.py
decisions:
  - tanodds列名を使用(tanoddslowではなく)。Parquet/feat_dfの実際の列名に合わせる
  - get_win_candidates()未実装時はget_place_candidates()にフォールバック(Plan 08-02で実装)
  - WIN betのfinal_odds代入にfinal_win_odds_mapを使用(PLACEは従来通りfinal_odds_map)
metrics:
  duration: 833s
  completed: 2026-05-04
  tasks_completed: 2
  tests_added: 11
  files_modified: 6
---

# Phase 8 Plan 01: Win Settlement Infrastructure Summary

build_win_payout_map() + final_win_odds_map + _settle_bet() WIN bug fix + betting_target dispatch + --betting-target CLI + ETL SQL update

## Changes Made

### Task 1: engine.py -- win settlement core (TDD)

**build_win_payout_map()** -- paytansyoumaban1/paytansyopay1から(race_id, umaban)->multiplierマップを構築。paytansyopay1/100で倍率変換。1着のみなのでループ不要。

**_settle_bet() WIN fix** -- 既存のバグ(WIN betがplace payout_mapで誤決済)を修正。WIN branchを追加し、win_payout_mapを参照。payout_map汚染を排除。フォールバック時はWARNINGログ + finish_pos==1チェック。

**BacktestEngine betting_target** -- __init__()にbetting_target: str = "win"を追加。"win"|"place"|"wide"をバリデーション。

**final_win_odds_map** -- feat_dfのtanodds列から(race_id, umaban)->oddsマップを構築。WIN betのfinal_odds代入に使用。

**candidate_df dispatch** -- betting_target=="win"時にget_win_candidates()を呼び出し(未実装時はget_place_candidates()にフォールバック)。

**6 TDD tests**: build_win_payout_map(3), betting_target(3), win_settle_bet(3のうち1つは既存place確認)

### Task 2: CLI + ETL SQL (TDD)

**--betting-target** -- run_backtest.pyにchoices=["win","place","wide"], default="win"を追加。単一年度・マルチ年度双方のBacktestEngine生成に渡す。

**get_payouts() SQL** -- paytansyoumaban1, paytansyopay1をSELECTに追加。ETL type rulesは既に対応済み。

**5 TDD tests**: CLI defaults/choices(4), SQL column verification(1)

## Test Results

- Total: 1123 passed, 2 failed (pre-existing DB connection tests), 2 skipped
- New tests: 11 (6 engine + 5 CLI/SQL)
- All plan-specific tests pass

## Deviations from Plan

None -- plan executed exactly as written.

## TDD Gate Compliance

- Task 1: RED commit (tests written first, all failed) -> GREEN commit (implementation, all passed)
- Task 2: RED commit (tests written first, 4/5 failed) -> GREEN commit (implementation, all passed)
- Both tasks followed strict TDD RED/GREEN cycle

## Commits

| Commit | Message |
|--------|---------|
| 37f58cc | feat(08-01): add win payout map, fix _settle_bet WIN bug, add betting_target dispatch |
| f37d665 | feat(08-01): add --betting-target CLI flag and update get_payouts() SQL |

## Self-Check

- [x] build_win_payout_map() exists in src/backtest/engine.py
- [x] win_payout_map constructed in BacktestEngine.run()
- [x] final_win_odds_map constructed from tanodds column
- [x] _settle_bet() has WIN branch before PLACE branch
- [x] betting_target param in BacktestEngine.__init__() with validation
- [x] --betting-target in run_backtest.py with choices/default
- [x] paytansyoumaban1 in everydb2_queries.py get_payouts() SQL
- [x] Both BacktestEngine construction sites pass betting_target
- [x] All 1123+ tests pass
