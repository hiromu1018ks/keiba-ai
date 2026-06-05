---
phase: 50-safety-validation
verified: 2026-06-05T23:30:00Z
status: gaps_found
score: 2/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "BT ROI 97%+ achieved (SC-3 / VLD-01)"
    status: failed
    reason: "2025 single-year BT ROI = 87.3% (3,335 bets, profit -42,420). Primary gate (>=97%) failed. Secondary gate not evaluated. NOT_DEPLOYABLE verdict."
    artifacts:
      - path: "data/backtest/multi_year_result.json"
        issue: "ROI 87.3%, below 97% threshold"
    missing:
      - "BT ROI >= 97% — feature set does not achieve recovery target"
  - truth: "IC evaluation report generated for all 23 features (SC-4 / VLD-02)"
    status: failed
    reason: "data/audit/track_condition_ic_report.json does not exist. Script exists and tests pass (16/16), but OOF predictions were not regenerated separately. IC report was not produced against live data."
    artifacts:
      - path: "data/audit/track_condition_ic_report.json"
        issue: "File does not exist"
    missing:
      - "Execute run_track_condition_ic_eval.py against OOF data to produce IC report"
  - truth: "WF Fold0 NaN rate in acceptable range (SC-5 / VLD-03)"
    status: failed
    reason: "NaN report verdict = FAIL. 5/23 features above threshold: sire_x_cushion_band (51.63%), surface_condition_transition/race_condition_match_score/max/ratio (100% each). All 5 are derived_cause, not raw data issues."
    artifacts:
      - path: "data/audit/track_condition_nan_report.json"
        issue: "overall_verdict = FAIL, 5 features above NaN threshold"
    missing:
      - "Fix derived NaN causes for 5 features, or accept as known limitation"
---

# Phase 50: Safety & Validation Verification Report

**Phase Goal:** Validate CI safety and BT ROI for all Phase 48/49 track condition features, with deployment verdict.
**Verified:** 2026-06-05
**Status:** gaps_found
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria + PLAN must_haves)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Feature Routing Audit passes: excluded 4 models have 0 TC features, included 7 have all 23 (SC-1 / REG-02) | VERIFIED | 58/58 CI tests pass. test_feature_routing_audit.py::TestTrackConditionRouting verifies exclusion and inclusion. run_feature_audit() returns PASS. |
| 2 | POST_RACE 3-layer CI confirms all 23 TC features NOT in POST_RACE_COLS (SC-2 / REG-03) | VERIFIED | test_post_race_leakage.py::TestTrackConditionPostRace passes: 23 features not in POST_RACE_COLS, all registered in models, raw values excluded. |
| 3 | BT ROI 97%+ achieved on multi-year BT (SC-3 / VLD-01) | FAILED | 2025 ROI = 87.3% (3,335 bets, -42,420 profit). Primary gate failed. Deployment verdict: NOT_DEPLOYABLE. |
| 4 | IC evaluation report generated with C-orthogonal IC per feature (SC-4 / VLD-02) | FAILED | Script (513 lines) + tests (16/16 pass) exist. But data/audit/track_condition_ic_report.json not generated -- OOF data not available for IC evaluation run. |
| 5 | WF Fold0 NaN rate in acceptable range (SC-5 / VLD-03) | FAILED | NaN report exists. Verdict: FAIL. 5 features above threshold (sire_x_cushion_band 51.6%, 4 race_condition features 100%). All derived_cause. |

**Score:** 2/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_feature_routing_audit.py` | Extended with TC routing verification | VERIFIED | TestTrackConditionRouting class with 4 tests, all passing |
| `tests/test_track_condition_routing.py` | Surface-aware NaN CI test | VERIFIED | 125 lines, TestSurfaceAwareNaN with 5 tests, all passing |
| `tests/test_post_race_leakage.py` | Extended with TC POST_RACE verification | VERIFIED | TestTrackConditionPostRace class with 3 tests, all passing |
| `tests/test_track_condition_nan.py` | WF Fold0 NaN rate tests | VERIFIED | 179 lines, TestWFold0NaNRate with 5 tests, all passing |
| `scripts/validate_track_condition_nan.py` | NaN diagnostic script | VERIFIED | 298 lines, CLI with --features-path/--start/--end/--output |
| `scripts/run_track_condition_ic_eval.py` | IC evaluation script | VERIFIED | 513 lines, full CLI implementation. Tests pass but report not generated. |
| `tests/test_track_condition_ic.py` | IC evaluation tests | VERIFIED | 293 lines, 16 tests, all passing |
| `data/audit/track_condition_nan_report.json` | WF Fold0 NaN report | VERIFIED (exists) | Generated. Verdict = FAIL (5 features above threshold) |
| `data/audit/track_condition_ic_report.json` | Per-feature IC report | MISSING | Not generated. Requires OOF data run. |
| `data/backtest/bt_2025_backtest.csv` | 2025 BT results | MISSING | bt_2025 CSV not found. multi_year_result.json confirms 87.3% ROI. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| test_feature_routing_audit.py | src/audit/feature_routing_registry.py | run_feature_audit + model FEATURE_COLS | WIRED | Import confirmed, audit runs and passes |
| test_track_condition_routing.py | src/features/track_condition_features.py | TRACK_CONDITION_COLS imports | WIRED | Imports verified in passing tests |
| scripts/validate_track_condition_nan.py | data/features/horse_features.parquet | read_parquet + NaN computation | WIRED | Script ran and produced NaN report |
| scripts/run_track_condition_ic_eval.py | data/oof/oof_predictions.parquet | read_parquet + Spearman IC | NOT_WIRED | Script exists but never executed against live OOF data |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| validate_track_condition_nan.py | NaN report JSON | horse_features.parquet | Yes -- 23 features measured, 5 FAIL | FLOWING |
| run_track_condition_ic_eval.py | IC report JSON | oof_predictions.parquet | No -- script never executed | DISCONNECTED |
| run_backtest.py | ROI result | PostgreSQL + models | Yes -- 87.3% ROI confirmed | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All Phase 50 CI tests pass | `python -m pytest tests/test_feature_routing_audit.py tests/test_post_race_leakage.py tests/test_track_condition_routing.py tests/test_track_condition_nan.py tests/test_track_condition_ic.py -v` | 58/58 passed | PASS |
| NaN diagnostic script runs | Script produced track_condition_nan_report.json | Verdict: FAIL, 5 features above threshold | PASS (runs correctly) |
| BT ROI gate | 2025 BT ROI >= 97% | 87.3% | FAIL |

### Probe Execution

No probes declared for this phase. Skipped.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| REG-02 | 50-01 | Feature Routing Audit -- surgical routing verification | SATISFIED | 4 excluded + 7 included models verified by CI tests |
| REG-03 | 50-01 | POST_RACE 3-layer CI for track condition features | SATISFIED | 23 TC features not in POST_RACE_COLS, all registered |
| VLD-01 | 50-02 | Multi-year BT ROI >= 97% | BLOCKED | 2025 ROI = 87.3%, below 97% threshold |
| VLD-02 | 50-02 | IC evaluation for new features | BLOCKED | Script + tests complete. IC report not generated (no OOF run). |
| VLD-03 | 50-01, 50-02 | WF Fold0 NaN rate acceptable | BLOCKED | NaN report = FAIL. 5/23 features above threshold (all derived_cause). |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No TBD/FIXME/XXX markers found in any Phase 50 file |

### Human Verification Required

None required -- all truths are programmatically verifiable.

### Gaps Summary

**3 of 5 roadmap success criteria FAILED:**

1. **BT ROI 87.3% < 97% threshold (VLD-01):** The primary deployment gate failed. The 23 track condition features did not recover BT ROI to v1.7 level. Per D-02, no diagnostic retry was warranted (no structural anomaly -- routing correct, NaN anomaly is inert features). The phase correctly recorded NOT_DEPLOYABLE verdict.

2. **IC report not generated (VLD-02):** The IC evaluation script is complete (513 lines, 16 tests passing) but was never executed against live OOF data. The SUMMARY notes "IC report generation requires OOF predictions from run_train.py which was not executed separately." This is a gap -- the script exists but the data artifact does not.

3. **NaN report verdict FAIL (VLD-03):** 5 of 23 features have NaN rates above 50%: sire_x_cushion_band (51.6%) and 4 race_condition features (100% each). All are derived_cause (track_month_stats unavailable during build_all, insufficient cross-data). The diagnostic script correctly identified these, but the features are effectively inert in the model.

**Deployment verdict: NOT_DEPLOYABLE** -- correctly recorded by the phase executor based on BT ROI gate failure.

---

_Verified: 2026-06-05T23:30:00Z_
_Verifier: Claude (gsd-verifier)_
