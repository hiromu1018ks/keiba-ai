---
phase: 44-roi-bisect
plan: 02
subsystem: backtest
tags: [attribution, cli, html-report, bisect, jinja2, json, markdown]

# Dependency graph
requires:
  - phase: 44-roi-bisect/plan-01
    provides: "ComponentAttribution + HistoricalBisect analysis engines"
  - phase: 43-shadow-diagnosis
    provides: "shadow_diagnosis artifacts + output pattern (save_diagnosis_results)"
  - phase: 41-shadow-comparison-framework
    provides: "shadow_horse_diff, shadow_race_diff"
provides:
  - "scripts/run_component_attribution.py CLI entry point"
  - "save_attribution_results() producing bisect_result.json + coefficient_analysis.json + bisect_summary.md"
  - "ComponentAttributionReportGenerator in separate module (component_attribution_report.py)"
  - "component_attribution_report.html Jinja2 template with 4 sections + recommendations"
affects: [45-structural-fix]

# Tech tracking
tech-stack:
  added: []
  patterns: ["CLI + JSON + MD + HTML multi-output pattern (Phase 41/43 convention)", "Report generator extracted to separate module for coupling reduction"]

key-files:
  created:
    - scripts/run_component_attribution.py
    - src/backtest/component_attribution_report.py
    - src/backtest/templates/component_attribution_report.html
  modified:
    - src/backtest/component_attribution.py
    - tests/test_component_attribution.py

key-decisions:
  - "Report generator extracted to component_attribution_report.py to decouple analysis from presentation"
  - "bisect_result.json uses condensed coefficient summary (top-5) while coefficient_analysis.json has full details"
  - "CLI imports HistoricalBisect optionally with try/except for robustness"
  - "HTML template uses same CSS variables and .negative pattern as Phase 43 shadow_diagnosis_report.html"

patterns-established:
  - "Multi-format output: save_*_results() in engine module produces JSON + MD, report generator in separate module produces HTML"
  - "CLI entry point pattern: build_parser() + main(args) + Windows cp932 encoding fix + optional --report flag"

requirements-completed: [BISECT-01, BISECT-02]

# Metrics
duration: 5min
completed: 2026-05-30
---

# Phase 44 Plan 02: Component Attribution CLI + Report Summary

**CLI script with JSON/Markdown/HTML multi-output layer for ComponentAttribution, producing Phase 45-consumable attribution artifacts via extracted report generator**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-30T13:14:23Z
- **Completed:** 2026-05-30T13:19:42Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- save_attribution_results() produces 3 machine/human-readable output files (bisect_result.json, coefficient_analysis.json, bisect_summary.md)
- CLI script (scripts/run_component_attribution.py) with --input-dir, --output-dir, --model-dir, --report flags
- ComponentAttributionReportGenerator extracted to separate module with Jinja2 HTML template (4 attribution sections + recommendations)
- 30 tests total pass (7 new output tests + 4 new HTML tests added)

## Task Commits

Each task was committed atomically:

1. **Task 1: CLI script + save_attribution_results + output tests** - `1dbb4a6` (feat)
2. **Task 2: HTML report generator in separate module + template** - `a006d33` (feat)

## Files Created/Modified
- `scripts/run_component_attribution.py` - CLI entry point for component attribution with --report flag
- `src/backtest/component_attribution.py` - Added save_attribution_results() + helper functions for JSON/MD output
- `src/backtest/component_attribution_report.py` - ComponentAttributionReportGenerator (extracted from component_attribution.py)
- `src/backtest/templates/component_attribution_report.html` - Jinja2 HTML template with 4 sections + recommendations
- `tests/test_component_attribution.py` - 30 tests total (11 new: JSON structure, MD sections, CLI dry-run, HTML generation)

## Decisions Made
- Report generator extracted to component_attribution_report.py to decouple analysis engine from presentation layer
- bisect_result.json contains condensed coefficient summary (top-5 per component) while coefficient_analysis.json has full 51-dim details
- CLI imports HistoricalBisect optionally with try/except -- graceful degradation if artifacts missing
- HTML template reuses same CSS variable system and .negative class as Phase 43 shadow_diagnosis_report.html

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CLI and output layer ready for Phase 45 to consume bisect_result.json
- HTML report provides human-readable visualization of all attribution results
- bisect_summary.md provides 6-section human-readable summary with Phase 45 recommendations
- All 37 tests pass (30 attribution + 7 historical bisect)

## Self-Check: PASSED

- scripts/run_component_attribution.py: FOUND
- src/backtest/component_attribution.py: FOUND
- src/backtest/component_attribution_report.py: FOUND
- src/backtest/templates/component_attribution_report.html: FOUND
- tests/test_component_attribution.py: FOUND
- Commit 1dbb4a6: FOUND
- Commit a006d33: FOUND

---
*Phase: 44-roi-bisect*
*Completed: 2026-05-30*
