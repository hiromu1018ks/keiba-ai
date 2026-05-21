---
phase: 10-pipeline-performance
reviewed: 2026-05-04T12:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - scripts/run_backtest.py
  - scripts/run_wf_validation.py
  - src/backtest/engine.py
  - src/features/feature_engine.py
  - src/utils/profiling.py
  - tests/test_backtest_engine.py
  - tests/test_feature_engine.py
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-05-04T12:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Reviewed 7 files from Phase 10 (pipeline performance). The phase added vectorized pandas operations replacing iterrows(), Parquet feature caching with SHA-256 keys, and pyinstrument profiling integration.

One critical bug found in `run_wf_validation.py` where `--betting-target` is unprocessable at runtime due to duplicate argparse. Several warnings found including a misleading docstring in the cache validation function, a silent cache read failure that could return stale results, a variable shadowing issue, and a redundant code smell in `_run_single_year`.

## Critical Issues

### CR-01: Duplicate argparse in run_wf_validation.py makes --betting-target unusable

**File:** `scripts/run_wf_validation.py:108-137`
**Issue:** `main()` (line 108-113) creates an `ArgumentParser` that only knows about `--profile` and calls `parse_args()`, consuming all of `sys.argv`. Then `_run_validation()` (line 130-137) creates a second `ArgumentParser` that only knows about `--betting-target` and also calls `parse_args()`. When a user runs `python scripts/run_wf_validation.py --betting-target place`, the first parser in `main()` encounters an unrecognized argument `--betting-target` and raises `SystemExit(2)`. Conversely, if called with `--profile --betting-target place`, the first parser fails on `--betting-target`. The two parsers are mutually incompatible -- only one set of flags can work at a time.

Furthermore, when called as `python scripts/run_wf_validation.py` with no arguments, `main()` parses `--profile=False`, then `_run_validation()` re-parses `sys.argv` (still empty after the first parse consumed nothing), getting `--betting-target=win`. This happens to work by accident, but the two separate `parse_args()` calls on the same `sys.argv` is fragile and incorrect design.

**Fix:**
```python
def main() -> None:
    """WF validation main"""
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Enable pyinstrument profiling")
    parser.add_argument(
        "--betting-target",
        choices=["win", "place", "wide"],
        default="win",
        help="betting target (default: win)",
    )
    args = parser.parse_args()

    from utils.profiling import ProfileContext

    with ProfileContext(enabled=args.profile, label="wf_validation"):
        _run_validation(args.betting_target)


def _run_validation(betting_target: str = "win") -> None:
    # ... remove the second argparse block ...
    # Use betting_target parameter instead of args.betting_target
```

## Warnings

### WR-01: is_cache_valid docstring claims hybrid timestamp+content-hash but only does timestamp check

**File:** `src/features/feature_engine.py:54-70`
**Issue:** The `is_cache_valid()` docstring (line 58) states: "Hybrid invalidation: timestamp comparison (fast) -> content hash (strict)". However, the function body only performs mtime comparison. There is no content hash fallback at all. If a source file is modified and then restored to its original content within the same second (or without changing mtime), the cache would be incorrectly invalidated. More importantly, the documented "strict" validation path simply does not exist, which misleads callers into believing stronger guarantees than what is provided. This is not merely a documentation issue -- callers relying on the documented hybrid behavior may make incorrect assumptions about cache correctness.

**Fix:** Either implement the content-hash comparison as documented, or update the docstring to accurately reflect that only timestamp comparison is performed:
```python
def is_cache_valid(
    cache_path: Path,
    source_paths: list[Path],
) -> bool:
    """Timestamp-based cache invalidation.

    Returns True if cache_path exists and is newer than all source_paths.
    """
```

### WR-02: Feature cache read failure is silently swallowed, returning potentially empty DataFrame

**File:** `src/features/feature_engine.py:190-195`
**Issue:** When `is_cache_valid()` returns `True` (cache appears fresh) but `store.read()` fails (corrupt file, permission error, format mismatch), the code catches the exception with `logger.warning("Feature cache read failed, recomputing")` but then falls through to line 196 (`else:` branch), which logs "Feature cache MISS". The `else` only executes when `is_cache_valid` is False, so after the try/except on line 194, execution falls through to the normal computation code at line 200+. This works correctly by accident -- the exception handler does not `return`, so computation proceeds. However, the control flow is misleading: the `else` clause on line 196 belongs to the `if is_cache_valid(...)` on line 188, not to the try/except. A reader might assume the `else` handles the cache-miss case, but the except-block also flows into the computation path. This non-obvious control flow should be made explicit.

**Fix:** Restructure the cache logic to be explicit about both code paths:
```python
cache_hit = False
if self._use_cache and _cache_name is not None:
    if is_cache_valid(cache_path, source_paths):
        logger.info("Feature cache HIT: %s", _cache_name)
        try:
            cached_df = store.read(self._cache_dir, _cache_name)
            if not cached_df.empty:
                return cached_df
        except Exception:
            logger.warning("Feature cache read failed, recomputing")
        # If we get here, cache was invalid or read failed
        cache_hit = True  # was a hit but failed to read

if not cache_hit or _cache_name is None:
    logger.info("Feature cache MISS: %s (computing...)", _cache_name)
```

### WR-03: Variable name `race_df` shadows the outer-scope variable in BacktestEngine.run()

**File:** `src/backtest/engine.py:586,589`
**Issue:** In the `run()` method, the outer scope has a variable `race_df` (line 379) holding all races. Inside the per-race loop (line 584), `race_id` is used to look up a single race's data from `feat_groups.get(race_id)` into `race_df_single` (line 586-589). This is correct. However, the comment at line 565 says "Groupby dict preprocessing" and the variable `race_df` from line 379 is used later at line 495 (`hist_df_all = hist_all.compute(race_df, entry_df, race_ids)`). The outer `race_df` is the full DataFrame, while `race_df_single` is the per-race slice. This naming convention is acceptable but the outer variable is also filtered (line 390: `race_df = race_df[race_df["race_id"].isin(jra_race_ids)]`) which mutates it. If any future developer accidentally uses `race_df` instead of `race_df_single` inside the loop, they would get the full dataset instead of a single race. This is a code quality / maintainability warning.

**Fix:** Consider renaming the outer-scope `race_df` to `race_df_all` or `all_races_df` to make the distinction more obvious and prevent accidental misuse.

### WR-04: Redundant JRA filter applied twice in BacktestEngine.run()

**File:** `src/backtest/engine.py:387-392,473-483`
**Issue:** The JRA filter (excluding races where jyocd is not between 1-10) is applied twice. First at lines 387-392 on the raw `race_df`/`entry_df`/`final_odds_df`, and then again at lines 473-483 on `feat_df`. While double-filtering does not produce incorrect results, it indicates redundant work. The second filter (lines 473-483) was likely added as a safety net, but since `feat_df` is derived from the already-filtered `race_df` and `entry_df`, the second filter should never exclude anything. If it does, it means the feature generation pipeline is injecting non-JRA data, which would be a separate bug worth catching explicitly.

**Fix:** Either remove the second filter and add a comment explaining the first filter is sufficient, or convert the second filter to an assertion that validates no NAR entries leaked through:
```python
# Safety check: verify feature generation did not introduce NAR entries
if "jyocd" in feat_df.columns:
    jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
    nar_count = (~jyocd_int.between(1, 10)).sum()
    if nar_count > 0:
        logger.warning("NAR entries leaked into feat_df: %d", nar_count)
        feat_df = feat_df[jyocd_int.between(1, 10)]
```

## Info

### IN-01: Unused variable kumi5_len in build_wide_payout_map

**File:** `src/backtest/engine.py:249`
**Issue:** The variable `kumi5_len` is assigned on line 249 (`kumi5_len = lengths[mask5]`) but never used in the subsequent code. This is dead code.

**Fix:** Remove the unused assignment:
```python
# Remove line 249: kumi5_len = lengths[mask5]
```

### IN-02: Unused variable kumi5 in build_wide_payout_map

**File:** `src/backtest/engine.py:248`
**Issue:** The variable `kumi5` is assigned on line 248 (`kumi5 = combined.loc[mask5, "kumi"]`) but never used in the subsequent code. This is dead code.

**Fix:** Remove the unused assignment:
```python
# Remove line 248: kumi5 = combined.loc[mask5, "kumi"]
```

### IN-03: sys.path manipulation at module level in both scripts

**File:** `scripts/run_backtest.py:44-46` and `scripts/run_wf_validation.py:33-35`
**Issue:** Both scripts insert `ROOT` and `ROOT/src` into `sys.path` at module level. This is a long-standing pattern in this codebase (not introduced in Phase 10), so it is flagged as info only. The path manipulation can cause import shadowing if a package with the same name exists in both directories.

**Fix:** No action needed for this phase. Consider consolidating path setup into a single helper if addressing in the future.

---

_Reviewed: 2026-05-04T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
