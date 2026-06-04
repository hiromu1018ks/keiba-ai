---
phase: 43-shadow-diagnosis
plan: 02
subsystem: backtest
tags: [shadow-diagnosis, cli, jinja2, html-report, json-output, markdown-summary]

# Dependency graph
requires:
  - phase: 43-shadow-diagnosis
    plan: 01
    provides: ShadowDiagnosis class, ShadowDiagnosisResult dataclass hierarchy
provides:
  - save_diagnosis_results() for JSON + Markdown output (D-04)
  - ShadowDiagnosisReportGenerator for HTML report generation
  - scripts/run_shadow_diagnosis.py CLI entry point
  - shadow_diagnosis_report.html Jinja2 template with 3-step diagnostic display
affects: [44-roi-bisect, 45-structural-fix]

# Tech tracking
tech-stack:
  added: []
  patterns: [cli-output-layer, jinja2-diagnosis-report, json-diagnosis-schema]

key-files:
  created:
    - scripts/run_shadow_diagnosis.py
    - src/backtest/templates/shadow_diagnosis_report.html
  modified:
    - src/backtest/shadow_diagnosis.py
    - tests/test_shadow_diagnosis.py

key-decisions:
  - "save_diagnosis_results outputs JSON (machine-readable) + MD (human review) as separate files"
  - "HTML template groups calibration segments by segment_name with per-group subtables"
  - "Delta degradation shown in red via CSS .negative class for Brier/logloss/ECE increase and APR decrease"
  - "CLI follows Phase 41 pattern: build_parser + main with --input-dir/--output-dir/--report flags"

patterns-established:
  - "Diagnosis output pattern: ShadowDiagnosisResult -> save_diagnosis_results (JSON+MD) + optional HTML report"
  - "HTML segment grouping: Jinja2 loop.previtem for per-segment_name subtables"

requirements-completed: [DIAG-01, DIAG-02, DIAG-03]

# Metrics
duration: 5min
completed: 2026-05-28
---

# Phase 43 Plan 02: Shadow Diagnosis Output Layer Summary

**CLI script + JSON/HTML/Markdown output layer for ShadowDiagnosis, producing Phase 44/45-consumable diagnosis artifacts**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-28T22:15:24Z
- **Completed:** 2026-05-28T22:20:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- save_diagnosis_results() outputs shadow_diagnosis_result.json (step1/step2/step3/missing_inputs/recommendations) and shadow_diagnosis_summary.md (5-section Markdown)
- scripts/run_shadow_diagnosis.py CLI with --input-dir, --output-dir, --report flags following Phase 41 pattern
- ShadowDiagnosisReportGenerator generates self-contained HTML report with 3-step diagnostic sections
- Jinja2 HTML template with delta degradation highlighting (red for Brier/logloss/ECE increase, APR decrease)
- 5 new tests added (12 total, all pass): JSON output, MD summary, CLI dry run, HTML report, missing inputs

## Task Commits

Each task was committed atomically:

1. **Task 1: CLI script + save_diagnosis_results + output tests** - `f211fe3` (feat)
2. **Task 2: Full Jinja2 HTML report template** - `e4ba24d` (feat)

## Files Created/Modified
- `scripts/run_shadow_diagnosis.py` - CLI entry point with --input-dir/--output-dir/--report flags
- `src/backtest/shadow_diagnosis.py` - Added save_diagnosis_results(), ShadowDiagnosisReportGenerator, _result_to_dict(), _build_summary_md()
- `src/backtest/templates/shadow_diagnosis_report.html` - Jinja2 HTML template with 3-step diagnostic sections
- `tests/test_shadow_diagnosis.py` - Added TestSaveDiagnosisResults (2 tests), TestCLIDryRun (1 test), TestReportGenerator (2 tests)

## Decisions Made
- JSON schema uses nested step1_probability_quality/step2_selection_pattern/step3_calibration structure for Phase 44/45 consumption
- Markdown summary includes top 5 calibration gaps sorted by |delta_apr|+|delta_ece| descending
- HTML template groups Step 3 calibration segments by segment_name with subtables, following Phase 41 pattern
- CLI stdout summary prints Step 1/2/3 key metrics in compact format, matching Phase 41 output style
- Windows cp932 encoding handled in CLI via sys.stdout.reconfigure(encoding="utf-8", errors="replace")

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- subprocess.run with text=True on Windows uses cp932 encoding, causing UnicodeDecodeError when reading CLI --help output containing Japanese characters. Fixed by adding encoding="utf-8" and errors="replace" to subprocess.run parameters.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- shadow_diagnosis_result.json schema ready for Phase 44 (ROI Bisect) consumption
- CLI fully functional for Phase 41 artifact analysis
- HTML report provides human-readable diagnostic output for review
- All 3 output files (JSON/HTML/MD) include missing_inputs section for Phase 41 extension decisions

## Self-Check: PASSED

- FOUND: scripts/run_shadow_diagnosis.py
- FOUND: src/backtest/templates/shadow_diagnosis_report.html
- FOUND: src/backtest/shadow_diagnosis.py
- FOUND: tests/test_shadow_diagnosis.py
- FOUND: f211fe3 (Task 1 commit)
- FOUND: e4ba24d (Task 2 commit)

---
*Phase: 43-shadow-diagnosis*
*Completed: 2026-05-28*
