---
phase: 53-strategy-alignment-live-data
plan: 01
subsystem: strategy-alignment
tags: [strategy-manifest, odds-band-filter, dd-shadow-only, pfp, session-manifest]
dependency_graph:
  requires: [Phase 52]
  provides: [STR-01, STR-02, STR-03, STR-04, STR-05, STR-06, D-01, D-02, D-03, D-08]
  affects: [scripts/run_paper_trading.py, src/backtest/race_predictor.py, src/features/session_manifest.py]
tech_stack:
  added: [compute_obf_config_hash helper, _validate_betting_target_alignment function]
  patterns: [TDD RED/GREEN, BT/PT identical RacePredictor construction]
key_files:
  created:
    - tests/test_strategy_alignment.py
  modified:
    - scripts/run_paper_trading.py
    - src/backtest/race_predictor.py
    - src/features/session_manifest.py
decisions:
  - "PT uses dd_shadow_only=True so DDController records state but always uses 100 flat stake"
  - "OddsBandFilter is injected for win-target only, place-target leaves it None"
  - "PT does not recalibrate OddsBandFilter (BT calibrated state reused via manifest)"
  - "3-way target alignment checks model/manifest/CLI with fail-fast on mismatch"
metrics:
  duration: 471s
  completed: 2026-06-06
  tasks: 2
  files: 4
  tests: 12
---

# Phase 53 Plan 01: Strategy Alignment and OddsBandFilter Summary

PT pipeline strategy parameter injection with BT-verified manifest, OddsBandFilter for win-target, and DD shadow-only mode.

## Commits

| Hash | Message |
|------|---------|
| 16903e0 | test(53-01): add failing tests for strategy alignment |
| 297a433 | feat(53-01): strategy manifest integration, OddsBandFilter injection, dd_shadow_only |

## Changes Made

### Task 1: Strategy manifest integration and CLI required args

- **parse_args()**: Added `--betting-target` (required, choices=["win","place"]), `--betting-mode` (required, choices=["flat","kelly"]), `--strategy-manifest` (optional path). Wide is not in choices so argparse rejects it automatically.
- **_validate_betting_target_alignment()**: New function performing 3-way fail-fast validation (model meta / manifest _betting_target / CLI args). Mismatch triggers `sys.exit(1)`.
- **_run_predict()**: Strategy manifest is loaded with SHA256 verification, converted to strategy_config via `build_strategy_config_from_params()`. RacePredictor construction follows BT engine.py pattern with Kelly/DD injection when `betting_mode=="kelly"`.
- **PFPVerifier**: Uses `args.betting_target` and `args.betting_mode` instead of hardcoded `"place"/"flat"`.
- **SessionManifest**: Added `betting_target`, `betting_mode`, `strategy_manifest_path`, `strategy_manifest_sha256` fields plus `set_strategy_params()` method.
- **_run_diagnose() / _run_dry_run()**: Pass `args.betting_target` to RacePredictor construction.
- **STR-05 verification**: Test confirms RacePredictor.should_bet() calls RaceQualityScreener.should_bet() internally.

### Task 2: OddsBandFilter injection for PT win-target

- **RacePredictor**: Added `odds_band_filter: OddsBandFilter | None = None` and `dd_shadow_only: bool = False` constructor parameters.
- **select_bets()**: Win-target branch applies `odds_band_filter.filter(candidates)` before `head(max_bets)`, identical to engine.py L1186-1188 pattern. Place-target does NOT call filter.
- **dd_shadow_only**: When True, DDController adjust_stake results are overridden to 100 flat. DD state is still computed and can be logged for diagnostics.
- **SessionManifest**: Added `odds_band_filter_metadata` dict field and `set_obf_metadata()` method recording 4 metadata fields: calibration_data_end_date, roi_threshold, excluded_bands, config_hash.
- **compute_obf_config_hash()**: New helper computing SHA256 of OddsBandFilter.BANDS + roi_threshold string.
- **run_paper_trading.py**: Win-target constructs OddsBandFilter(roi_threshold=...) and records metadata to session_manifest. Place-target skips OBF injection.

## Tests

12 tests in `tests/test_strategy_alignment.py`:

| Test | Description |
|------|-------------|
| TestParseArgsRejectsWide (3) | Wide rejected, win/place accepted |
| TestStrategyConfigFromManifest (1) | Manifest JSON loaded and config built |
| TestThreeWayTargetMismatch (3) | Model/CLI, all match, manifest/CLI mismatches |
| TestSessionManifestStrategyFields (1) | set_strategy_params records 4 fields |
| TestRaceQualityScreenerIntegrated (1) | should_bet() calls screener (STR-05) |
| TestRacePredictorOddsBandFilterWin (1) | Filter called for win-target |
| TestRacePredictorNoOddsBandFilterPlace (1) | Filter NOT called for place-target |
| TestOBFMetadataFourFields (1) | 4 OBF metadata fields recorded |

All 37 tests passing (12 new + 25 existing pipeline_consistency).

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

No new security-relevant surface beyond plan threat model.

## Self-Check

## Self-Check: PASSED

All 5 key files exist on disk:
- tests/test_strategy_alignment.py
- scripts/run_paper_trading.py
- src/backtest/race_predictor.py
- src/features/session_manifest.py
- .planning/phases/53-strategy-alignment-live-data/53-01-SUMMARY.md

Both commits exist in git log:
- 16903e0: test(53-01): add failing tests for strategy alignment
- 297a433: feat(53-01): strategy manifest integration, OddsBandFilter injection, dd_shadow_only
