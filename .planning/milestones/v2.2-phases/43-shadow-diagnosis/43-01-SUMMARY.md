---
phase: 43-shadow-diagnosis
plan: 01
subsystem: backtest
tags: [shadow-diagnosis, calibration, brier, ece, logloss, segmentation]

# Dependency graph
requires:
  - phase: 41-shadow-comparison-framework
    provides: ShadowComparisonFramework, ComparisonMetrics, _compute_ece, horse_diff/race_diff Parquet artifacts
provides:
  - ShadowDiagnosis class with 3-step progressive exclusion diagnostic
  - ShadowDiagnosisResult with ProbabilityQualityResult, SelectionPatternResult, CalibrationResult
  - Segment calibration by popularity_band/probability_rank_band/odds_band/surface/selected_changed
  - missing_inputs detection for absent columns in Phase 41 artifacts
affects: [44-roi-bisect, 45-structural-fix]

# Tech tracking
tech-stack:
  added: []
  patterns: [progressive-exclusion-diagnostic, segment-calibration, dynamic-variant-resolution]

key-files:
  created:
    - src/backtest/shadow_diagnosis.py
    - tests/test_shadow_diagnosis.py
  modified: []

key-decisions:
  - "Reused ShadowComparisonFramework._compute_ece() for ECE calculation instead of custom implementation"
  - "probability_rank_band computed from p_win_final rank within race (always available)"
  - "Empty segments skipped rather than falling back to 'unknown' label"

patterns-established:
  - "Post-hoc diagnostic pattern: read Phase 41 artifacts, compute metrics without re-running BacktestEngine"
  - "Dynamic variant resolution: manifest variants[] -> column prefix mapping"

requirements-completed: [DIAG-01, DIAG-02, DIAG-03]

# Metrics
duration: 8min
completed: 2026-05-28
---

# Phase 43 Plan 01: Shadow Diagnosis Summary

**3-step progressive exclusion diagnostic engine for baseline vs shadow probability quality, selection patterns, and calibration drift across 5 segment dimensions**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-28T22:09:12Z
- **Completed:** 2026-05-28T22:17:00Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- ShadowDiagnosis class reads Phase 41 artifacts and runs 3-step diagnostic analysis
- Step 1 computes Brier/logloss/ECE/actual_predicted_ratio for baseline vs shadow across all horses
- Step 2 splits races by selected_changed flag and computes ROI/hit_rate/avg_odds per group
- Step 3 computes calibration drift across 5 segment dimensions (popularity_band, probability_rank_band, odds_band, surface, selected_changed)
- Dynamic variant name resolution from manifest (Pitfall 5 mitigation)
- missing_inputs detection for absent columns (popularity, surface, tanodds, closing_win_odds)

## Task Commits

Each task was committed atomically (TDD):

1. **Task 1 RED: ShadowDiagnosis test suite** - `94dd69d` (test)
2. **Task 1 GREEN: ShadowDiagnosis implementation** - `0f96e1f` (feat)

_Note: No REFACTOR commit needed -- implementation is clean after initial pass._

## Files Created/Modified
- `src/backtest/shadow_diagnosis.py` - ShadowDiagnosis class with 6 dataclasses, segment constants, and 3-step diagnostic logic
- `tests/test_shadow_diagnosis.py` - 7 unit tests covering all diagnostic steps, missing_inputs, empty segment handling, and variant resolution

## Decisions Made
- Reused `ShadowComparisonFramework._compute_ece()` static method for ECE calculation (Phase 41 validated logic)
- `probability_rank_band` always computable from p_win_final rank within race -- no dependency on external columns
- Empty segments (e.g., "7+" in small fields) are skipped entirely rather than labeled "unknown"
- Baseline variant resolved as `variant_names[0]`, shadow as `variant_names[1]` from manifest

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ShadowDiagnosis class ready for CLI wrapper (Plan 02: `scripts/run_shadow_diagnosis.py`)
- `ShadowDiagnosisResult` output structure ready for Phase 44 (ROI Bisect) consumption
- missing_inputs tracking enables Phase 41 artifact extension decisions

## Self-Check: PASSED

- FOUND: src/backtest/shadow_diagnosis.py
- FOUND: tests/test_shadow_diagnosis.py
- FOUND: .planning/phases/43-shadow-diagnosis/43-01-SUMMARY.md
- FOUND: 94dd69d (test commit)
- FOUND: 0f96e1f (feat commit)

---
*Phase: 43-shadow-diagnosis*
*Completed: 2026-05-28*
