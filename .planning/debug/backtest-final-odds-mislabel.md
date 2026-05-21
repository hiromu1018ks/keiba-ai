---
status: resolved
---

# Debug: backtest-final-odds-mislabel

## Status: RESOLVED

## Root Cause

`Bet.final_odds` stores `fukuoddslow` (複勝オッズ下限) but was labeled as 精算用オッズ（確定オッズ）.
The bet_history output key `"final_odds"` suggested 確定単勝オッズ, while the actual value is place odds lower bound.
Additionally, the WIN settlement fallback at engine.py:1380 used `bet.final_odds` (fukuoddslow) for
tanodds-based settlement, which is semantically incorrect.

## Fix

1. **src/domain/models.py:162** -- Updated `final_odds` comment from `精算用オッズ（確定オッズ）` to
   `複勝オッズ下限（fukuoddslow）。place/wide 精算のフォールバック用`

2. **src/backtest/engine.py:1162** -- Renamed bet_history output key from `"final_odds"` to `"fuku_odds_low"`

3. **src/backtest/engine.py:1376-1381** -- WIN settlement fallback now uses `bet.odds` (tanodds) directly
   instead of `bet.final_odds` (fukuoddslow). Updated warning message to include odds value.

4. **src/backtest/validation_report.py:143** -- Updated `b.get("final_odds", ...)` to `b.get("fuku_odds_low", ...)`
   to match new bet_history key name.

5. **tests/test_backtest_validation.py** -- Updated all fixture dicts to use `"fuku_odds_low"` instead of
   `"final_odds"` (12 occurrences across sample_bet_history and fail_result fixtures).

## Scope

- Bet dataclass field name `final_odds` intentionally NOT renamed (too many references)
- WIN primary settlement via win_payout_map (engine.py:1374-1375) left unchanged (correct)
- WIDE/PLACE fallback paths (engine.py:1361, 1393) left unchanged (use fukuoddslow correctly)

## Verification

- Full test suite: 1626 passed, 1 skipped, 0 failures
- Commit: `aeb1505 fix(backtest): rename misleading final_odds to fuku_odds_low in bet_history`
