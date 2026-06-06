---
status: testing
phase: 54-automation-reporting
source: [54-VERIFICATION.md]
started: 2026-06-06T22:00:00Z
updated: 2026-06-06T22:00:00Z
---

## Current Test

number: 1
name: Live Race End-to-End with --mode run --betting-target win
expected: |
  Orchestrator generates win-type bets (not place bets). RacePredictor.select_bets receives betting_target='win'.
awaiting: user response

## Tests

### 1. Live Race End-to-End

expected: Run --mode run --betting-target win on a live race day. Orchestrator generates win-type bets (not place bets). RacePredictor.select_bets receives betting_target='win'.
result: [pending]

### 2. Crash Resume via Ctrl+C

expected: Start --mode run, Ctrl+C mid-prediction, re-run same command. Re-run skips already-predicted races, continues from interrupted race. Exit code 130 on first run, 0 on completed re-run.
result: [pending]

### 3. HTML Report Visual

expected: Open generated report.html in browser. KPI cards display correct ROI/bankroll, target breakdown shows win/place separately, model identity footer visible, bet history table shows settlement_status and outcome badges.
result: [pending]

### 4. Max Drawdown Accuracy (WR-01)

expected: Review _compute_max_dd behavior when all bets lose. Should show 100% or near-100% drawdown, not 0.0%.
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps
