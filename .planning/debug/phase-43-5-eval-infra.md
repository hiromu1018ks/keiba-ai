---
slug: phase-43-5-eval-infra
status: resolved
trigger: Phase 44 pre-requisite: 13 bugs/design issues identified in evaluation infrastructure
created: 2026-05-30
resolved: 2026-05-30
goal: find_and_fix
tdd_mode: false
specialist_dispatch_enabled: false
---

# Debug Session: Phase 43.5 Evaluation Infrastructure Fixes

## Trigger

Phase 44 requires trustworthy evaluation infrastructure. 13 issues identified across OOFHealthValidator, BacktestEngine, ShadowDiagnosis, and ShadowComparison. Must fix P0 items (1-5) before re-running backtests.

## Resolution

### P0-1: OOFHealthValidator -- NOT A BUG
- Profile already correct: score_col="p_ability_win", ROI counts odds only for winners
- No code change needed

### P0-2: training_bet_history 0 bets -- FIXED
- **Root cause:** Inner engine used `betting_target="place"` to avoid OddsBandFilter recursion, but win-only models have no place models -> 0 bets
- **Fix:** Inner engine now uses same `betting_target` as parent + `_skip_odds_band_calibration=True` flag to prevent recursion
- **Files:** `src/backtest/engine.py`, `tests/test_backtest_engine_autocalibrate.py`

### P0-3: ShadowDiagnosis Step2 baseline-only metrics -- FIXED (rev2)
- **Root cause:** `_compute_group_metrics()` hardcoded `baseline_stake`/`baseline_result` column names; JSON included shadow metrics but Markdown/HTML/CLI displayed baseline only
- **Fix:** Added `variant_prefix` parameter; `_step2_selection_pattern()` now computes both baseline and shadow metrics; added `changed_shadow`/`unchanged_shadow` fields to `SelectionPatternResult`; JSON output includes both; Markdown/HTML/CLI all show baseline vs shadow comparison tables
- **Files:** `src/backtest/shadow_diagnosis.py`, `src/backtest/templates/shadow_diagnosis_report.html`, `scripts/run_shadow_diagnosis.py`

### P0-4: horse_diff kakuteijyuni NaN for shadow-only horses -- FIXED
- **Root cause:** `_align_horse_level()` merged kakuteijyuni from baseline_df only; outer join left shadow-only horses with NaN
- **Fix:** Concatenate kakuteijyuni from all variant DataFrames, dropna+dedup before merging
- **Files:** `src/backtest/shadow_comparison.py`

### P0-5: Missing columns in horse_diff -- FIXED (rev2)
- **Root cause:** `_align_horse_level()` propagated `surface`, `tanodds`, `closing_win_odds`, `popularity` from baseline_df only -- shadow-only horses (outer join) had NaN
- **Fix (rev2):** Applied same concat-from-all-variants pattern as kakuteijy: concatenate from all variant DataFrames, dropna, dedup by first non-null per (race_id, umaban)
- **Files:** `src/backtest/shadow_comparison.py`

### Test Results (rev2)
- 64 affected-module tests pass (shadow_comparison: 52, shadow_diagnosis: 12)
- Ruff: all checks passed

## Remaining Work (P1/P2 -- NOT in this session)
- P1: probability_rank_band redefinition, DeploymentGateEvaluator re-run, Win payout audit, PlaceAbilityModel warnings
- P2: 4-line ablation, 2025 changed race low-odds, MAWC year stability, odds band re-verification
- Re-run: backtest 2024/2025, shadow_comparison, shadow_diagnosis, deployment_gates
