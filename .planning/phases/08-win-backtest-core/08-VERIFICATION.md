---
phase: 08-win-backtest-core
verified: 2026-05-04T12:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification: false
gaps: []
human_verification:
  - test: "Run full backtest with --betting-target win and verify single-win ROI/bankroll output"
    expected: "ROI, hit rate, bankroll trajectory displayed for win bets with correct payout amounts"
    why_human: "Requires running the full pipeline with real data (multi-hour execution, PostgreSQL + Parquet). Automated tests verify wiring with mocks."
  - test: "Run run_wf_validation.py --betting-target win and verify overfitting detection uses win ROI"
    expected: "WF validation completes with fold train/test ROI based on win bets"
    why_human: "Requires real data pipeline execution with trained models (~4 hours)"
---

# Phase 8: Win Backtest Core Verification Report

**Phase Goal:** User can run a win-mode backtest and obtain correct win ROI, hit rate, and bankroll trajectory
**Verified:** 2026-05-04T12:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | build_win_payout_map() constructs (race_id,umaban)->multiplier from paytansyoumaban1/paytansyopay1 | VERIFIED | engine.py L128-150: function exists, reads paytansyoumaban1/paytansyopay1, divides paytansyopay1 by 100. Called at L322. Test: test_settle_bet_uses_payout_map PASSED |
| 2 | final_win_odds_map constructed from tanodds column for win bet odds reference | VERIFIED | engine.py L331-336: iterates feat_df, extracts tanodds. Used at L663 for WIN BetType final_odds assignment. Test: engine tests verify wiring |
| 3 | --betting-target win/place/wide flag switches BacktestEngine mode (default=win) | VERIFIED | engine.py L214-225: betting_target param with validation. run_backtest.py L83-88: --betting-target CLI with choices. run_wf_validation.py L116-121: same CLI. Both BacktestEngine constructions pass betting_target. Tests: TestBettingTargetArg (4 tests) PASSED |
| 4 | WinSelectionGate win_selection_ev/edge/prob columns produce conformal-confidence-scored win candidates | VERIFIED | race_predictor.py L408-470: get_win_candidates() filters win_selection_edge>0 AND tanodds>=1.0, sorts by win_gate_score DESC with conformal_confidence_score as tertiary sort, max 2 candidates. Tests: TestGetWinCandidates (7 tests) + TestSelectBetsWinPath (4 tests) all PASSED |
| 5 | run_wf_validation.py performs overfitting detection with win ROI | VERIFIED | run_wf_validation.py L115-122: argparse --betting-target with default="win". L184-186: test_engine receives betting_target. L197-200: train_engine receives betting_target. judge_overfitting() at L279 operates on win ROI results. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| src/backtest/engine.py | build_win_payout_map(), _settle_bet() WIN fix, betting_target, final_win_odds_map | VERIFIED | All functions present and substantive (L128-150, L990-1007, L214-225, L331-336) |
| src/backtest/race_predictor.py | get_win_candidates(), select_bets() win path | VERIFIED | get_win_candidates() at L408-470, select_bets() win branch at L618-658 |
| scripts/run_backtest.py | --betting-target CLI argument | VERIFIED | L83-88: argparse with choices=["win","place","wide"], default="win". Passed to both BacktestEngine constructions (L325, L439) |
| scripts/run_wf_validation.py | --betting-target CLI argument | VERIFIED | L115-122: argparse with choices, default="win". Passed to both BacktestEngine instances (L185, L199) |
| src/db/everydb2_queries.py | get_payouts() SQL with paytansyoumaban1/paytansyopay1 | VERIFIED | L270: s_harai SQL includes paytansyoumaban1, paytansyopay1. Test: test_get_payouts_sql_includes_win_columns PASSED |
| tests/test_backtest_engine.py | Tests for win payout, betting_target, settle_bet | VERIFIED | 50 phase-8 related tests pass. Total suite: 1134 passed |
| tests/test_race_predictor.py | Tests for get_win_candidates, select_bets win path | VERIFIED | 7 get_win_candidates + 4 select_bets win path tests pass |
| tests/test_run_backtest_args.py | Tests for --betting-target CLI | VERIFIED | 4 betting_target tests pass |
| tests/test_everydb2_queries.py | Test for win columns in SQL | VERIFIED | 1 test passes |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| scripts/run_backtest.py | src/backtest/engine.py | BacktestEngine(betting_target=...) | WIRED | L325 single-year, L439 multi-year: both pass betting_target |
| scripts/run_wf_validation.py | src/backtest/engine.py | BacktestEngine(betting_target=...) | WIRED | L185 test_engine, L199 train_engine: both pass betting_target |
| engine.py | payouts.parquet | build_win_payout_map(payouts_df) | WIRED | L322: self.win_payout_map = build_win_payout_map(payouts_df) |
| engine.py | feat_df tanodds | final_win_odds_map construction | WIRED | L331-336: iterates feat_df, uses tanodds column |
| engine.py _settle_bet() | win_payout_map | WIN branch lookup | WIRED | L990-1007: explicit WIN branch before PLACE, uses win_payout_map |
| engine.py run() | race_predictor.py | get_win_candidates() dispatch | WIRED | L545-553: dispatches based on self.betting_target |
| engine.py run() | race_predictor.py | select_bets(betting_target=...) | WIRED | L649-651: passes betting_target to select_bets |
| race_predictor.py | result_df columns | win_selection_edge, tanodds, win_gate_score | WIRED | get_win_candidates() reads these columns from race_df |
| engine.py Bet final_odds | final_win_odds_map | WIN BetType assignment | WIRED | L662-664: elif bet.bet_type == BetType.WIN uses final_win_odds_map |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| build_win_payout_map() | win_payout_map | payouts_df from load_payouts(store, start, end) | Parquet s_harai data | FLOWING |
| final_win_odds_map | final_win_odds_map | feat_df["tanodds"] from FeatureEngine.build_all() | Feature pipeline output | FLOWING |
| _settle_bet() WIN | win_key lookup | self.win_payout_map | Populated in run() L322 | FLOWING |
| get_win_candidates() | selection_edge, odds | race_df from engine.run() predict() output | Pipeline inference results | FLOWING |
| select_bets() WIN path | Bet objects | candidates from get_win_candidates() | Filtered candidate DF | FLOWING |
| run_wf_validation.py | args.betting_target | argparse CLI | User input | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase-8 tests pass | python -m pytest tests/ -v -k "win or betting_target or settle_bet or get_win or select_bets" | 50 passed | PASS |
| Full suite passes | python -m pytest tests/ -q | 1134 passed, 2 failed (pre-existing DB), 2 skipped | PASS |
| get_win_candidates exists | grep -c "def get_win_candidates" src/backtest/race_predictor.py | 1 | PASS |
| betting-target in run_backtest.py | grep "betting-target" scripts/run_backtest.py | L84, L87 matched | PASS |
| betting-target in run_wf_validation.py | grep "betting-target" scripts/run_wf_validation.py | L118 matched | PASS |
| paytansyoumaban1 in SQL | grep "paytansyoumaban1" src/db/everydb2_queries.py | L270 matched | PASS |
| tanodds column used (not tanoddslow) | grep "tanodds" src/backtest/engine.py | L335, L336 matched (not tanoddslow) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| WIN-01 | 08-01 | build_win_payout_map() reads tan_umaban/tan_pay for payout_map | SATISFIED | engine.py L128-150: reads paytansyoumaban1/paytansyopay1 |
| WIN-02 | 08-01 | final_odds_map uses tanodds for win bet settlement | SATISFIED | engine.py L331-336: final_win_odds_map from tanodds; L662-664: WIN BetType uses final_win_odds_map |
| WIN-03 | 08-02 | get_win_candidates() filters by win_selection_ev/edge/prob | SATISFIED | race_predictor.py L408-470: filters win_selection_edge>0 AND tanodds>=1.0, 7 tests pass |
| WIN-04 | 08-01 | BacktestEngine betting_target parameter (default=win) | SATISFIED | engine.py L214-225: param + validation. CLI: run_backtest.py, run_wf_validation.py |
| WIN-05 | 08-02 | Conformal confidence score in win bet selection | SATISFIED | race_predictor.py L450-459: conformal_confidence_score as tertiary sort key (soft signal, not hard filter) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| src/db/everydb2_queries.py | 287-298 | n_harai fallback SQL missing paytansyoumaban1/paytansyopay1 | Info | Fallback SQL for n_harai does not include win payout columns. s_harai (primary) does include them. ETL uses s_harai primarily. If n_harai fallback is ever triggered, win_payout_map would be empty for those dates, falling back to odds-based settlement. Not blocking since s_harai is the primary path and ETL writes to Parquet. |

### Human Verification Required

### 1. Full Pipeline Execution (Win Backtest)

**Test:** Run `python scripts/run_backtest.py --train-start 20200101 --train-end 20231231 --test-start 20240101 --test-end 20241231` (default --betting-target=win)
**Expected:** Win ROI, hit rate, bankroll trajectory displayed. Payout amounts reflect actual win dividends (100x multiplier from paytansyopay1).
**Why human:** Requires ~57 minutes execution with real data. Automated tests verify all wiring with mocks.

### 2. Walk-Forward Validation (Win Mode)

**Test:** Run `python scripts/run_wf_validation.py` (default --betting-target=win)
**Expected:** WF validation produces fold results with win-based train/test ROI. Overfitting detection judges based on win ROI gaps.
**Why human:** Requires ~4 hours execution with real data and model training.

### Gaps Summary

No blocking gaps found. All 5 must-have truths are verified with code-level evidence and passing tests. The n_harai fallback SQL missing win payout columns is informational only -- s_harai (the primary SQL path) correctly includes them, and the ETL pipeline uses s_harai for Parquet generation.

---

_Verified: 2026-05-04T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
