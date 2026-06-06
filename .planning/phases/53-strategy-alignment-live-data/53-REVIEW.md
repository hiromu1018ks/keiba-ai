---
phase: 53-strategy-alignment-live-data
reviewed: 2026-06-06T12:00:00Z
depth: deep
files_reviewed: 9
files_reviewed_list:
  - scripts/run_paper_trading.py
  - src/backtest/race_predictor.py
  - src/features/feature_builder.py
  - src/features/session_manifest.py
  - src/ingestion/track_condition_fetcher.py
  - src/paper_trading/predictor.py
  - tests/test_strategy_alignment.py
  - tests/test_track_condition_fetcher.py
  - tests/test_live_data_integration.py
findings:
  critical: 3
  warning: 5
  info: 3
  total: 11
status: issues_found
---

# Phase 53: Code Review Report

**Reviewed:** 2026-06-06
**Depth:** deep
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Reviewed all 9 Phase 53 files (6 source, 3 test). Found 3 critical bugs, 5 warnings, and 3 info-level items. The 55 tests all pass but leave critical code paths untested. The three plans (53-01 strategy alignment, 53-02 track condition fetcher, 53-03 live data integration) have significant integration gaps: the live data path from JRA to FeatureBuilder is never connected in the primary predict mode, and the `select_bets` call always defaults to `"place"` regardless of CLI `--betting-target`.

## Critical Issues

### CR-01: `select_bets()` always uses `betting_target="place"` regardless of CLI args

**File:** `scripts/run_paper_trading.py:725`
**Issue:** `race_predictor.select_bets(result_df, bankroll)` is called without passing `betting_target`. The `select_bets()` method signature has `betting_target: str = "place"` as default (race_predictor.py:1261). When `--betting-target win` is specified, the code constructs `RacePredictor` with `betting_target="win"` but then calls `select_bets()` without forwarding the target. This means PT predict mode **always generates place bets**, making the entire win-target path dead code in production.

The same bug exists in `_run_diagnose` (line 1064) and `_run_dry_run` (line 1295).

Note: `RacePredictor.select_bets()` accepts `betting_target` as a method parameter (not reading from `self.betting_target`). This is a design inconsistency -- the constructor's `betting_target` is used in `predict()` but not as the default for `select_bets()`.

**Fix:**
```python
# Line 725 in _run_predict:
bets = race_predictor.select_bets(result_df, bankroll, betting_target=betting_target)

# Line 1064 in _run_diagnose:
bets = race_predictor.select_bets(result_df, bankroll=0, betting_target=args.betting_target)

# Line 1295 in _run_dry_run:
bets = race_predictor.select_bets(result_df, bankroll, betting_target=args.betting_target)
```

### CR-02: `build_for_inference()` accepts `live_track_conditions` but never passes it through

**File:** `src/features/feature_builder.py:189,215`
**Issue:** `build_for_inference()` accepts `live_track_conditions: pd.DataFrame | None = None` as a parameter (line 189), but at line 215 it calls `self._build()` without forwarding it. The `_build()` method has no `live_track_conditions` parameter, and `_enrich_features()` has no live merge step. The `_merge_live_track_conditions()` method exists (line 528) but is never called from any code path.

This means the entire live data integration from Plan 53-03 is non-functional. The `PaperPredictor.setup()` passes `live_track_conditions` to `builder.build_for_inference()` (predictor.py:136), but the value is silently discarded.

**Fix:**
```python
# feature_builder.py: build_for_inference() must pass live_track_conditions to _build
result = self._build(
    race_df,
    entry_df,
    odds_df,
    odds_ts_df=odds_ts_df,
    preserve_columns=None,
    feature_state=feature_state,
    feature_version=feature_version,
    live_track_conditions=live_track_conditions,  # ADD THIS
)

# _build() signature must accept and forward it:
def _build(
    self,
    race_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    *,
    odds_ts_df: pd.DataFrame | None = None,
    preserve_columns: list[str] | None = None,
    feature_state: FeatureState | None = None,
    feature_version: str = "1.0",
    live_track_conditions: pd.DataFrame | None = None,  # ADD THIS
) -> FeatureBuildResult:
    # ... after _enrich_features:
    if live_track_conditions is not None and not live_track_conditions.empty:
        feat_df = self._merge_live_track_conditions(feat_df, live_track_conditions)
```

### CR-03: `_run_predict` uses `build_for_training()` instead of `build_for_inference()` -- live data path unreachable

**File:** `scripts/run_paper_trading.py:148-163,583-585`
**Issue:** The `_build_features_fb()` helper (line 148) always calls `FeatureBuilder.build_for_training()`, which does not accept `live_track_conditions`. Even if CR-02 were fixed, the PT predict mode would never use the live data path because it calls `build_for_training()` instead of `build_for_inference()`.

Additionally, Plan 53-03 specified that `run_paper_trading.py _run_predict()` should call `JRATrackConditionFetcher` to fetch live data before prediction, but no such code exists. The `JRATrackConditionFetcher` import and call is entirely absent from `run_paper_trading.py`. This means the entire LIV-03 integration is unimplemented in the primary predict flow.

**Fix:** `_run_predict` should:
1. Call `JRATrackConditionFetcher().fetch_all_venues(ymd)` to get live data
2. Call `build_for_inference()` (not `build_for_training()`) with `live_track_conditions`
3. Handle the live data fetch failure with `sys.exit(1)` per the plan

## Warnings

### WR-01: `_run_diagnose` and `_run_dry_run` do not use `dd_shadow_only=True` or `OddsBandFilter`

**File:** `scripts/run_paper_trading.py:1003,1214`
**Issue:** Plan 53-01 Task 1 item 8 states: "_run_diagnose() と _run_dry_run() にも --betting-target/--betting-mode を反映する。RacePredictor の構築を _run_predict() と同じパターンにする (manifest なし版)". However:
- `_run_diagnose` (line 1003): `RacePredictor(models, betting_target=args.betting_target)` -- no `dd_shadow_only=True`
- `_run_dry_run` (line 1214): `RacePredictor(models, betting_target=args.betting_target)` -- no `dd_shadow_only=True`, no OddsBandFilter

This is inconsistent with `_run_predict` which sets `dd_shadow_only=True` and injects OddsBandFilter for win-target.

**Fix:** Construct `RacePredictor` in diagnose/dry-run with the same kwargs as predict mode.

### WR-02: `_run_dry_run` date parsing crash -- `date.fromisoformat` on `YYYYMMDD` format

**File:** `scripts/run_paper_trading.py:1203-1204`
**Issue:** `args.start` is in `"YYYY-MM-DD"` format. The code does `args.start.replace("-", "")` producing `"YYYYMMDD"`, then passes it to `date.fromisoformat()`. But `date.fromisoformat()` expects `"YYYY-MM-DD"` format with dashes. This will raise `ValueError` at runtime whenever `--start` is used in dry-run mode.

Compare with `_run_diagnose` (line 969) which correctly uses `args.start.replace("-", "")` only for the Parquet reader ymd format, not for date parsing.

**Fix:**
```python
# Lines 1203-1204: remove the .replace() calls
start = date.fromisoformat(args.start)
end = date.fromisoformat(args.end)
```

### WR-03: `fetch_all_venues` silently swallows real parse errors

**File:** `src/ingestion/track_condition_fetcher.py:305-308`
**Issue:** In `fetch_all_venues()`, `TrackConditionParseError` is caught and silently continued with a debug log, under the assumption that non-racing venues will lack required DOM elements. However, this also swallows genuine parse errors for active racing venues -- for example, if the JRA site returns a valid page for venue "05" (Tokyo) but the HTML structure has changed and both `#turf_line` and `#dirt_line` are missing, the error is silently dropped and the venue is skipped. The caller has no way to distinguish "non-racing venue" from "parsing failure on active venue".

This partially contradicts the plan requirement (T-53-05): "1場でも失敗した場合は例外を送出 (予測停止)".

**Fix:** At minimum, log at WARNING level instead of DEBUG for `TrackConditionParseError`. Better: track whether any venue returned data, and if all 10 venues fail to parse, raise an error.

### WR-04: `aggregate_dirt_moisture` has unbounded recursion for unknown rules

**File:** `src/features/feature_builder.py:121-122`
**Issue:** When `rule` is not `"goal"`, `"4c"`, or `"mean"`, the function logs a warning and recursively calls itself with `rule="mean"`. While this only recurses once (since `"mean"` is handled), the pattern is fragile. If `"mean"` handling were ever refactored to also call recursively, it would create infinite recursion.

**Fix:** Replace the recursive call with direct inline logic:
```python
else:
    logger.warning("Unknown moisture rule '%s', falling back to mean", rule)
    if goal is not None and four_c is not None:
        return (goal + four_c) / 2.0
    elif goal is not None:
        return goal
    elif four_c is not None:
        return four_c
    return None
```

### WR-05: `_run_predict` bankroll tracking incorrect when `dd_shadow_only=True`

**File:** `scripts/run_paper_trading.py:782`
**Issue:** In the PT predict loop, `bankroll -= bet.stake` is executed for each bet (line 782). When `dd_shadow_only=True`, the stake in the `Bet` object is set to `100.0` (flat), which is correct for the recorded bet. However, the bankroll decrement uses the recorded stake, not the DD-adjusted stake. This means the bankroll tracking in PT mode is always flat (100 yen per bet), which may not match the intended simulation. This is arguably correct for shadow mode (the recorded stake IS 100), but the discrepancy between DD state and actual stake could cause confusion in bankroll-dependent downstream logic.

**Fix:** Add a comment clarifying this is intentional for shadow mode, or record both the shadow stake and the DD-adjusted stake.

## Info

### IN-01: Plan 53-03 live data fetching is unimplemented in `run_paper_trading.py`

**File:** `scripts/run_paper_trading.py`
**Issue:** Plan 53-03 specifies that `_run_predict()` should call `JRATrackConditionFetcher().fetch_all_venues()` to fetch live track conditions before feature generation, convert results to a DataFrame, save raw HTML to session directory, and call `session_manifest.set_live_data()`. None of this code exists in `run_paper_trading.py`. The `live_data` field in `SessionManifest` is never populated. The `PaperPredictor` class has the `live_track_conditions` parameter but is not called from `_run_predict()`.

### IN-02: `_detect_html_structure_change` is dead code

**File:** `src/ingestion/track_condition_fetcher.py:323-336`
**Issue:** `_detect_html_structure_change()` is defined but never called from any code path. It is exported in the module but has no callers. The function is trivially `return reference_hash != current_hash` and provides no value beyond inline comparison.

### IN-03: Test coverage gap -- no integration test for `_run_predict` win-target end-to-end

**File:** `tests/test_strategy_alignment.py`
**Issue:** The test `TestRacePredictorOddsBandFilterWin` mocks `get_win_candidates` and calls `select_bets` directly, bypassing the actual `RacePredictor.predict()` flow. There is no test that exercises the full `_run_predict` code path with `betting_target="win"` to verify that OddsBandFilter is applied in context. The CR-01 bug (`select_bets` always defaults to "place") would not be caught by any existing test because `select_bets` is called in tests with explicit `betting_target`.

---

_Reviewed: 2026-06-06_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
