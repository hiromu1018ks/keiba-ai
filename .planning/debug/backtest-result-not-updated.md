---
slug: backtest-result-not-updated
status: resolved
trigger: run_backtest.py 実行時に backtest_result.json が更新されない
goal: find_and_fix
tdd_mode: false
created: 2026-05-08
---

# Debug: backtest-result-not-updated

## Symptoms
- `run_backtest.py` を実行しても `backtest_result.json` が更新されない
- スクリプト自体は正常終了する可能性があるが、結果ファイルが古いまま

## Investigation

### Files checked
- `scripts/run_backtest.py` — メインスクリプト (root cause)

### Evidence

- 2026-05-08: `backtest_result.json` (root) has same 2024 test data, `data/backtest/backtest_result.json` also has same data
- 2026-05-08: Root cause found: two bugs in `scripts/run_backtest.py`

### Root Cause

**Bug 1 (multi-year mode):** In `_run_multi_year()`, the `multi_year_result.json` write was gated
behind `if args.report:` (line 677). Without `--report`, no JSON result file was written at all --
only console output. This contradicts the single-year behavior which always writes a JSON file.

**Bug 2 (single-year mode):** In `_run_single_year()`, without `--report` the result was written
to `ROOT/backtest_result.json` (project root), while with `--report` it went to
`data/backtest/backtest_result.json`. A user checking `data/backtest/` would see a stale file.

## Fix Applied

1. **Single-year mode:** Moved `backtest_result.json` write out of the `if args.report:` block.
   Now always writes to `data/backtest/backtest_result.json` regardless of `--report`.
   The `--report` flag now only controls additional outputs (HTML report, bet_history, parquet).

2. **Multi-year mode:** Moved `multi_year_result.json` write out of the `if args.report:` block.
   Now always writes to `data/backtest/multi_year_result.json` regardless of `--report`.
   The `--report` flag now only controls additional outputs (HTML report, bet_history JSON).

### Verification
- All 1349 tests pass (0 failures, 1 skipped)
- `test_run_backtest_args.py` -- 15 tests pass

## Resolution

- **Root cause:** JSON result output gated behind `--report` flag in multi-year mode;
  inconsistent output path in single-year mode
- **Fix:** Always write JSON results to `data/backtest/` regardless of `--report` flag.
  `--report` now only controls supplementary outputs (HTML, bet_history, parquet).
