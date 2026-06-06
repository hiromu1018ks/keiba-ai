---
phase: 53-strategy-alignment-live-data
plan: 03
subsystem: features, paper_trading, session_manifest
tags: [live-data, track-conditions, moisture-collation, TDD-green]
dependency_graph:
  requires: [53-01, 53-02]
  provides: [live-track-conditions-merge, moisture-collation, session-live-data]
  affects: [FeatureBuilder, SessionManifest, PaperPredictor]
tech_stack:
  added: []
  patterns: [TDD GREEN phase, left-join merge with NaN safety, closest-match collation]
key_files:
  created: []
  modified:
    - src/features/feature_builder.py
    - src/features/session_manifest.py
    - src/paper_trading/predictor.py
decisions:
  - Collation uses closest-match (minimum distance) rather than priority order to correctly select mean when it is the best match
  - _merge_live_track_conditions uses left join to avoid introducing race_ids not in the history DataFrame
  - aggregate_dirt_moisture falls back to single available value when one of goal/4c is NaN under mean rule
metrics:
  duration: 5min
  tasks_completed: 1
  completed: "2026-06-06T09:00:00Z"
---

# Phase 53 Plan 03: Live Data Integration Summary

JRA live track condition merge into FeatureBuilder with JRA/CSV moisture collation and session manifest live_data metadata.

## What Changed

### FeatureBuilder (`src/features/feature_builder.py`)

- Added `collate_moisture_rule()`: Determines the best moisture aggregation rule (goal/4c/mean) by comparing JRA live values against CSV historical values. Uses closest-match with 0.5% threshold. Raises ValueError when collation is impossible (D-06).
- Added `aggregate_dirt_moisture()`: Applies the determined rule to compute dirt_moisture from goal/4c values. Under "mean" rule with one value NaN, falls back to the available value.
- Added `_merge_live_track_conditions()`: Merges live DataFrame into history DataFrame via left join on race_id. Non-NaN live values override history; NaN preserves history. Empty or None live_df returns original unchanged (D-07).

### SessionManifest (`src/features/session_manifest.py`)

- Added `live_data: dict[str, Any]` field (default empty dict)
- Added `set_live_data(source, measured_at, fetched_at, html_hash, venue_codes)` method
- `to_dict()` now includes `live_data` (LIV-03)

### PaperPredictor (`src/paper_trading/predictor.py`)

- `setup()` accepts `live_track_conditions: pd.DataFrame | None = None` parameter
- Passes `live_track_conditions` through to `builder.build_for_inference()`

## Test Results

18 tests, all passing:

| Test | Description | Result |
|------|-------------|--------|
| TestMergeNoneLive | None input returns original | PASS |
| TestMergeLiveOverridesHistory | Live values override history | PASS |
| TestMergeLiveNaNPreservesHistory | NaN preserves history | PASS |
| TestMergeEmptyLiveDF | Empty DataFrame no change | PASS |
| TestSessionManifestLiveData | set_live_data records 5 fields | PASS |
| TestCollateMoistureRuleSelectsBest | goal/4c/mean selection | PASS |
| TestCollateMoistureRuleMismatchHalts | ValueError on mismatch | PASS |
| TestCollateMoistureRuleInsufficientData | Default to mean | PASS |
| TestDirtMoistureAggregationGoal | goal rule | PASS |
| TestDirtMoistureAggregation4c | 4c rule | PASS |
| TestDirtMoistureAggregationMean | mean rule + single-value fallback | PASS |
| TestLiveFailureStopsPrediction | TrackConditionParseError on empty HTML | PASS |
| TestPredictorSetupPassesLiveConditions | setup() accepts and passes param | PASS |

## TDD Gate Compliance

- RED commit: be0d547 (18 tests, 17 failing)
- GREEN commit: d0bac0a (18 tests, all passing)
- REFACTOR: Not needed -- code is clean and minimal

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed collation priority order**
- **Found during:** GREEN phase -- test_mean_rule_selected failed
- **Issue:** Original priority order (goal > 4c > mean) incorrectly selected "goal" when csv_value=5.5 with jra_goal=5.0 because abs(5.0-5.5)=0.5 <= threshold
- **Fix:** Changed to closest-match algorithm: compute distance for all candidates, select minimum. This correctly selects "mean" when (goal+4c)/2 is closest to csv_value.
- **Files modified:** src/features/feature_builder.py
- **Commit:** d0bac0a

**2. [Rule 1 - Bug] Fixed ruff lint violations**
- **Found during:** Post-implementation lint check
- **Issue:** N806 (uppercase THRESHOLD in function) and E501 (line too long 106 > 100)
- **Fix:** Renamed to lowercase `threshold`, broke long warning string across lines
- **Files modified:** src/features/feature_builder.py
- **Commit:** d0bac0a

## Threat Surface

No new network endpoints, auth paths, or trust boundaries introduced beyond what Plan 02 already established (JRATrackConditionFetcher). The merge and collation functions operate on in-memory DataFrames only.

## Out of Scope

- `run_paper_trading.py` live data fetch section (script-level integration) -- the test suite validates the component interfaces. Full script integration happens at PT execution time.

## Self-Check: PASSED

- All 3 modified files exist on disk
- GREEN commit d0bac0a found in git log
- RED commit be0d547 found in git log
- 18/18 tests passing
