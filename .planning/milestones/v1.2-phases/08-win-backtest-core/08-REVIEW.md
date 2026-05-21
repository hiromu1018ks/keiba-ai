---
phase: 08-win-backtest-core
reviewed: 2026-05-04T12:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - scripts/run_backtest.py
  - scripts/run_wf_validation.py
  - src/backtest/engine.py
  - src/backtest/race_predictor.py
  - src/db/everydb2_queries.py
  - tests/test_backtest_engine.py
  - tests/test_everydb2_queries.py
  - tests/test_race_predictor.py
  - tests/test_run_backtest_args.py
findings:
  critical: 1
  warning: 5
  info: 4
  total: 10
status: issues_found
---

# Phase 8: Code Review Report

**Reviewed:** 2026-05-04T12:00:00Z
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Reviewed 9 files implementing Phase 8 (win betting mode for backtest). The core logic for `build_win_payout_map`, `get_win_candidates`, `_settle_bet` WIN branch, `--betting-target` CLI dispatch, and `conformal_confidence_score` ranking is structurally sound. Tests provide good coverage of the new code paths. However, one data-integrity bug in `_parse_kumi` will produce incorrect wide payout lookups for certain horse number combinations, and several quality issues merit attention.

## Critical Issues

### CR-01: `_parse_kumi` mishandles 3-digit kumi when first horse has 2 digits

**File:** `src/backtest/engine.py:167-173`
**Issue:** The `_parse_kumi` function inside `build_wide_payout_map` assumes a fixed split point for 3-character kumi strings: always `[:1]` + `[1:]` (first 1 character, then 2 characters). This produces wrong results when the first horse number is 2 digits and the second is 1 digit.

For example, kumi `"129"` (horse 12 + horse 9) would parse as `(1, 29)` instead of `(9, 12)` after sorting. Since horse numbers range from 1-18, a second component of `29` is impossible and the key will never match the wide odds lookup. Additionally, the function does not enforce `lo <= hi`, so the key `(race_id, umaban_lo, umaban_hi)` is inconsistent with how wide_odds columns are constructed (line 350: `f"wide_odds_{lo}_{hi}"` always has lo < hi).

Concrete failure case: kumi `"159"` (horse 15 + horse 9) parses as `(1, 59)` -- completely wrong.

**Fix:**
```python
def _parse_kumi(kumi_str: str) -> tuple[int, int] | None:
    """Parse kumi with ambiguous 3-char split by trying both splits."""
    n = len(kumi_str)
    if n == 4:
        lo, hi = int(kumi_str[:2]), int(kumi_str[2:])
    elif n == 3:
        # Two possible splits: X|YZ or XY|Z
        split_a = (int(kumi_str[:1]), int(kumi_str[1:]))
        split_b = (int(kumi_str[:2]), int(kumi_str[2:]))
        # Valid horse numbers are 1-18; pick the valid split
        valid_a = all(1 <= v <= 18 for v in split_a)
        valid_b = all(1 <= v <= 18 for v in split_b)
        if valid_a and not valid_b:
            lo, hi = split_a
        elif valid_b and not valid_a:
            lo, hi = split_b
        elif valid_a:
            # Both valid -- use lexicographic convention
            lo, hi = split_a
        else:
            return None
    elif n == 2:
        lo, hi = int(kumi_str[:1]), int(kumi_str[1:])
    else:
        return None
    return (min(lo, hi), max(lo, hi))
```

## Warnings

### WR-01: `get_payouts()` n_harai fallback SQL missing win payout columns

**File:** `src/db/everydb2_queries.py:287-298`
**Issue:** The `s_harai` query (lines 267-279) selects `paytansyoumaban1, paytansyopay1` (win payout columns). However, the `n_harai` fallback query (lines 287-298) only selects the `payfukusyou*` (place payout) columns and omits `paytansyoumaban1, paytansyopay1`. When the system falls back to `n_harai`, `build_win_payout_map()` will find no win payout data and return an empty map, causing all win bets to use the odds-based fallback in `_settle_bet()`.

**Fix:** Add `paytansyoumaban1, paytansyopay1` to the `n_harai` SELECT:
```python
sql = f"""
    SELECT
        CAST(year || monthday || jyocd || kaiji || nichiji || racenum AS varchar) AS race_id,
        paytansyoumaban1, paytansyopay1,
        payfukusyoumaban1, payfukusyopay1,
        payfukusyoumaban2, payfukusyopay2,
        payfukusyoumaban3, payfukusyopay3,
        payfukusyoumaban4, payfukusyopay4,
        payfukusyoumaban5, payfukusyopay5
    FROM n_harai
    WHERE year || monthday = %s
      AND datakubun IN ('1', '2')
"""
```

### WR-02: `_load_cached_models` accesses private method `_load_from_local`

**File:** `scripts/run_backtest.py:124`
**Issue:** The function calls `loader._load_from_local(model_dir, ...)`, which is a private method (prefixed with `_`). This breaks encapsulation and will silently break if the internal API changes. The public method `ModelLoader.load()` should be used instead if it supports the same use case, or `_load_from_local` should be made public.

**Fix:** Use the public API:
```python
loader = ModelLoader()
models, info = loader.load(model_dir, use_ensemble_override=True)
```
Or if this is intentionally using an internal API, add a comment explaining why and add a `# noqa` annotation.

### WR-03: `display_single_year_result` uses wrong field for average win odds

**File:** `scripts/run_backtest.py:248-253`
**Issue:** The average odds calculation uses `b.get("tanoddslow", 0)` to compute displayed average odds for win mode. The field `tanoddslow` is the pre-race win odds, not the final settlement odds (`final_odds`) or the model-input odds (`odds`). This may mislead users about actual payout odds. Additionally, if `tanoddslow` is not present in the bet_history dict for some reason, it silently falls back to 0, skewing the average downward without any warning.

**Fix:** Use `final_odds` (the actual settlement odds) for the average display, or at minimum use `odds`:
```python
avg_odds = (
    sum(b.get("final_odds", b.get("odds", 0)) for b in result.bet_history)
    / len(result.bet_history)
    if result.bet_history
    else 0.0
)
```

### WR-04: Hardcoded `before_roi = 0.638` benchmark value

**File:** `scripts/run_backtest.py:258`
**Issue:** The comparison baseline ROI is hardcoded as `0.638`. This value is not configurable, not documented, and has no reference to where it comes from (which experiment, which time period). As the model improves over iterations, this static benchmark becomes meaningless and may mislead users about improvement status.

**Fix:** Extract to a constant with a docstring explaining the reference, or make it configurable via CLI argument:
```python
# Baseline ROI from Phase 7 backtest (2024 test, place mode, flat betting)
BASELINE_ROI = 0.638
```

### WR-05: `wide_payout_map` key ordering does not guarantee `lo < hi`

**File:** `src/backtest/engine.py:186-188`
**Issue:** In `build_wide_payout_map`, the key is constructed as `(race_id, umaban_lo, umaban_hi)` where `umaban_lo, umaban_hi = parsed`. The `_parse_kumi` function does not sort the pair, so `umaban_lo` may be greater than `umaban_hi`. Meanwhile, the wide odds lookup in `_settle_bet` (line 1006) sorts the pair: `lo, hi = min(bet.umaban, pair_b), max(bet.umaban, pair_b)`. This mismatch means some wide payout lookups will silently fail (return 0 payout).

**Fix:** Sort the parsed pair before constructing the key:
```python
parsed = _parse_kumi(str(kumi).strip())
if parsed is None:
    continue
a, b = parsed
umaban_lo, umaban_hi = min(a, b), max(a, b)
```

## Info

### IN-01: `_generate_bets` and `_build_race_features` are dead code

**File:** `src/backtest/engine.py:913-998`
**Issue:** Both `_generate_bets` and `_build_race_features` have comments stating they are kept "for backwards compatibility" and that the logic has been delegated to `RacePredictor`. No code in the reviewed files calls these methods. If they are truly unused, they add maintenance burden and confusion. The test suite references `build_race_features` on `RacePredictor` directly (not the one on `BacktestEngine`).

**Fix:** Remove or mark with `@deprecated` and schedule for deletion in a future cleanup.

### IN-02: `warnings.filterwarnings("ignore")` suppresses all warnings globally

**File:** `scripts/run_backtest.py:37` and `scripts/run_wf_validation.py:28`
**Issue:** Both scripts suppress all warnings globally with `warnings.filterwarnings("ignore")`. This hides deprecation warnings, future warnings, and legitimate runtime warnings that could indicate bugs (e.g., pandas SettingWithCopyWarning). This is a common pattern in ML scripts but can mask real issues.

**Fix:** Consider filtering only specific warning categories:
```python
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
```

### IN-03: `_query` method in `EveryDB2Queries` opens new connection per call

**File:** `src/db/everydb2_queries.py:35-39`
**Issue:** Each `_query` call opens a new psycopg2 connection, executes one query, and closes it. For methods that call `_query` multiple times (e.g., `get_races` which tries `n_race` then `s_race`), this creates unnecessary connection overhead. This is a design choice rather than a bug, but worth noting for future optimization.

**Fix:** Consider connection pooling or reusing a connection within a single public method call.

### IN-04: `test_build_parser` is imported via `sys.path` manipulation in tests

**File:** `tests/test_run_backtest_args.py:12-18`
**Issue:** The test helper `_import_build_parser` modifies `sys.path` at import time and imports from `scripts.run_backtest`, which is not a proper Python package. This is fragile and depends on the working directory. The project's `pyproject.toml` likely does not include `scripts/` as a package.

**Fix:** This is a pre-existing pattern. For future improvement, consider making scripts importable via a package structure or using `importlib` with explicit path resolution.

---

_Reviewed: 2026-05-04T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
