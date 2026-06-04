---
phase: 42-feature-routing-audit-safety-gates
plan: 01
subsystem: testing
tags: [audit, feature-routing, safety-gate, saf-01]

# Dependency graph
requires:
  - phase: 39-marketawarewincalibrator
    provides: MarketAwareWinCalibrator.build_feature_matrix() producing 51 features
  - phase: 40-race-level-ranker
    provides: RaceLevelRanker RELEVANCE/VALUE/DERIVED_VALUE feature lists
  - phase: 41-shadow-comparison-framework
    provides: Shadow comparison pattern for audit script structure
provides:
  - Feature routing audit registry (FORBIDDEN_CALIBRATOR_FEATURES, FORBIDDEN_RANKER_FEATURES)
  - Fail-fast unit tests detecting forbidden feature leakage in CI
  - Diff tests catching stale registry entries
  - CLI audit script producing JSON + Markdown reports
  - Advisory model warnings (non-fail-fast)
affects: [42-02, 42-03, deployment-gates, CI]

# Tech tracking
tech-stack:
  added: []
  patterns: [audit-registry-single-source-of-truth, forbidden-feature-frozenset, dynamic-model-import-for-audit]

key-files:
  created:
    - src/audit/__init__.py
    - src/audit/feature_routing_registry.py
    - scripts/run_feature_routing_audit.py
    - tests/test_feature_routing_audit.py

key-decisions:
  - "field_size excluded from FORBIDDEN_CALIBRATOR_FEATURES as raw input (Pitfall 3)"
  - "Advisory model class names corrected to match actual codebase (EVCorrectionModel, PlaceEVCorrectionModel, AbilityModel, WinTwoStageModel)"
  - "WinTwoStageModel WARN for rel_p_ability_win_rank intersection (advisory, not fail-fast)"

patterns-established:
  - "Audit registry pattern: frozen dataclass AuditTarget + frozenset forbidden features + run_feature_audit() function"
  - "CALIBRATOR_EXCLUDED_RAW_INPUTS for documenting raw inputs that pass through build_feature_matrix()"

requirements-completed: [SAF-01]

# Metrics
duration: 7min
completed: 2026-05-28
---

# Phase 42 Plan 01: Feature Routing Audit Infrastructure Summary

**Audit registry with 50 calibrator + 28 ranker forbidden features, fail-fast CI tests, diff tests, and JSON/Markdown audit CLI script**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-28T12:27:24Z
- **Completed:** 2026-05-28T12:34:18Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Single source of truth audit registry at `src/audit/feature_routing_registry.py` with FORBIDDEN_CALIBRATOR_FEATURES (50) and FORBIDDEN_RANKER_FEATURES (28)
- 12 unit tests: count checks, fail-fast intersection tests (MarketModel + RaceQualityScreener), diff tests (stale registry detection), advisory targets, run_feature_audit integration
- CLI audit script producing JSON + Markdown reports with per-model status, intersections, feature counts
- All critical targets PASS (MarketModel, RaceQualityScreener have zero forbidden intersections)
- WinTwoStageModel advisory WARN for rel_p_ability_win_rank (expected, not fail-fast)

## Task Commits

Each task was committed atomically:

1. **Task 1 (TDD RED): Add failing tests for audit registry** - `bb0431e` (test)
2. **Task 1 (TDD GREEN): Implement feature routing audit registry** - `d5df663` (feat)
3. **Task 2: Add feature routing audit CLI script** - `c9e8b3e` (feat)

## Files Created/Modified

- `src/audit/__init__.py` - Package init for audit infrastructure
- `src/audit/feature_routing_registry.py` - Single source of truth: FORBIDDEN_CALIBRATOR_FEATURES (50), FORBIDDEN_RANKER_FEATURES (28), CRITICAL_TARGET_MODELS, ADVISORY_TARGET_MODELS, run_feature_audit()
- `tests/test_feature_routing_audit.py` - 12 fail-fast + diff tests for SAF-01
- `scripts/run_feature_routing_audit.py` - CLI audit script producing JSON + Markdown reports

## Decisions Made

- **field_size excluded from forbidden set (Pitfall 3):** `field_size` is a raw input passed through `build_feature_matrix()` with the same column name, legitimately used by MarketModel and RaceQualityScreener. Excluding it reduces the count from 51 to 50.
- **Advisory model class names corrected:** Actual codebase uses `EVCorrectionModel`, `PlaceEVCorrectionModel`, `AbilityModel`, `WinTwoStageModel` (not the names in the plan).
- **WinTwoStageModel WARN for rel_p_ability_win_rank:** The ranker feature `rel_p_ability_win_rank` appears in WinTwoStageModel's FEATURE_COLS. This is advisory (warning), not fail-fast.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Correctness] Excluded field_size from FORBIDDEN_CALIBRATOR_FEATURES**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Plan listed 51 forbidden features including `field_size`, but `field_size` is a raw input (not derived) used by both MarketModel and RaceQualityScreener. Per RESEARCH Pitfall 3, raw inputs should not be forbidden.
- **Fix:** Removed `field_size` from `_CALIBRATOR_MAIN`, added `CALIBRATOR_EXCLUDED_RAW_INPUTS` documentation constant, updated count to 50, updated diff test to subtract excluded raw inputs.
- **Files modified:** `src/audit/feature_routing_registry.py`, `tests/test_feature_routing_audit.py`
- **Verification:** All 12 tests pass, ruff clean, audit script exits 0 with PASS

**2. [Rule 3 - Blocking] Fixed advisory model class names**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Plan used incorrect class names (EvCorrectionModel, EPCorrectionModel, Stage1AbilityModel, WinModel) that don't exist in the codebase. Dynamic imports failed with warnings.
- **Fix:** Corrected to actual class names: EVCorrectionModel, PlaceEVCorrectionModel, AbilityModel, WinTwoStageModel by inspecting source files.
- **Files modified:** `src/audit/feature_routing_registry.py`
- **Verification:** All advisory models import successfully, run_feature_audit() returns no ERROR statuses

---

**Total deviations:** 2 auto-fixed (1 correctness, 1 blocking)
**Impact on plan:** Both auto-fixes essential for correctness. field_size exclusion follows RESEARCH Pitfall 3 guidance. Class name corrections are trivial lookup fixes.

## Issues Encountered

None beyond the deviations documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SAF-01 audit infrastructure complete and verified
- Registry serves as foundation for SAF-02 (OOF artifact profiles) and SAF-03 (deployment gates)
- `run_feature_audit()` function available for import by future gate evaluator
- Advisory WARN for WinTwoStageModel (rel_p_ability_win_rank) documented for awareness

---
*Phase: 42-feature-routing-audit-safety-gates*
*Completed: 2026-05-28*

## Self-Check: PASSED

All files verified:
- FOUND: src/audit/__init__.py
- FOUND: src/audit/feature_routing_registry.py
- FOUND: scripts/run_feature_routing_audit.py
- FOUND: tests/test_feature_routing_audit.py
- FOUND: .planning/phases/42-feature-routing-audit-safety-gates/42-01-SUMMARY.md

All commits verified:
- FOUND: bb0431e (test)
- FOUND: d5df663 (feat)
- FOUND: c9e8b3e (feat)
