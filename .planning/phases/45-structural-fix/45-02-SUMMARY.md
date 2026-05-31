---
phase: 45-structural-fix
plan: 02
subsystem: ml-models
tags: [mawc, cli, html-report, jinja2, manifest, markdown, retrain]

# Dependency graph
requires:
  - phase: 45-structural-fix
    provides: MawcConservativeRetrainer engine with retrain + quality gates + variant creation
provides:
  - CLI entry point (run_mawc_conservative_retrain.py) with --oof-path/--source-model-dir/--target-root/--years/--report
  - save_retrain_results() producing manifest.json + retrain_summary.md
  - MawcConservativeReportGenerator (separate module) with Jinja2 HTML template
  - HTML report with 5 sections: Configuration, Per-Surface Results, Quality Gate Comparison, Favorite Band Guard, C Grid Candidates
affects: [46-quality-gate-verification]

# Tech tracking
tech-stack:
  added: []
  patterns: [cli-output-triple, report-generator-separation, manifest-json-schema]

key-files:
  created:
    - scripts/run_mawc_conservative_retrain.py
    - src/models/mawc_conservative_report.py
    - src/models/templates/mawc_conservative_report.html
  modified:
    - src/models/mawc_conservative_retrainer.py
    - tests/test_mawc_conservative_retrainer.py

key-decisions:
  - "Report generator extracted to separate module (mawc_conservative_report.py) to decouple presentation from retraining engine"
  - "CLI re-runs retrain per surface to collect ConservativeRetrainResult objects for summary/report generation"
  - "retrain_summary.md has 6 sections matching plan spec: Configuration, Per-Surface Results, Quality Gate Details, Favorite Band Guard, C Grid Candidates, Phase 46 Next Steps"

patterns-established:
  - "Output triple pattern: manifest.json (machine) + retrain_summary.md (human) + HTML report (visual)"
  - "Report generator in separate module from engine (following Phase 44 ComponentAttribution pattern)"
  - "Jinja2 FileSystemLoader pointing to src/models/templates/ for report templates"

requirements-completed: [FIX-01, FIX-02]

# Metrics
duration: 8min
completed: 2026-05-31
---

# Phase 45 Plan 02: CLI + Output Layer Summary

**CLI script + JSON/Markdown/HTML output layer wrapping MawcConservativeRetrainer, producing Phase 46-consumable conservative variant artifacts**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-31T13:07:23Z
- **Completed:** 2026-05-31T13:15:18Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- CLI entry point (run_mawc_conservative_retrain.py) with full argument parsing following established Phase 41/43/44 CLI pattern
- save_retrain_results() function producing manifest.json and retrain_summary.md with 6 required sections
- MawcConservativeReportGenerator in separate module with Jinja2 HTML template containing 5 data sections + Phase 46 footer
- 6 new tests added (manifest JSON, summary MD, CLI help x2, report HTML, CSS classes) -- 33 total tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1: CLI script + manifest/summary output + output tests** - `f49e552` (feat)
2. **Task 2: HTML report generator in separate module + template** - `1fbae0f` (feat)

## Files Created/Modified
- `scripts/run_mawc_conservative_retrain.py` - CLI entry point with --oof-path, --source-model-dir, --target-root, --years, --report flags
- `src/models/mawc_conservative_retrainer.py` - Added save_retrain_results() + _write_retrain_summary() for manifest.json + retrain_summary.md output
- `src/models/mawc_conservative_report.py` - MawcConservativeReportGenerator class using Jinja2 FileSystemLoader
- `src/models/templates/mawc_conservative_report.html` - Self-contained HTML template with 5 sections + Phase 46 footer
- `tests/test_mawc_conservative_retrainer.py` - Added 6 new tests for output functions, CLI help, and report generation

## Decisions Made
- Report generator extracted to separate module (mawc_conservative_report.py) following Phase 44 ComponentAttribution pattern, keeping retraining engine and presentation layer decoupled
- CLI re-runs prepare_oof_data + retrain per surface to collect ConservativeRetrainResult objects for summary/report generation, since run_full_pipeline() only returns the manifest dict
- retrain_summary.md uses 6 sections as specified in the plan: Configuration, Per-Surface Results, Quality Gate Details, Favorite Band Guard (Odds 1-3), C Grid Candidates, Phase 46 Next Steps

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Test helper _setup_full_pipeline initially passed raw combined DataFrame to run_retrain() instead of prepared surface DataFrames (missing p_model column). Fixed by using prepare_oof_data() to get derived DataFrames before calling run_retrain().
- Summary MD test assertion checked for hardcoded path "data/models-backtest-mawc-conservative" but tmp_path was used. Fixed by checking for "mawc-conservative" substring instead.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- CLI script is fully functional and ready for end-to-end execution with real data
- Phase 46 can consume manifest.json from data/models-backtest-mawc-conservative/
- Phase 46 Shadow Comparison command is documented in both retrain_summary.md and HTML report footer:
  `python scripts/run_shadow_comparison.py --baseline-root data/models-backtest --shadow-root data/models-backtest-mawc-conservative --folds 2024 2025 --report`

---
*Phase: 45-structural-fix*
*Completed: 2026-05-31*

## Self-Check: PASSED

- FOUND: scripts/run_mawc_conservative_retrain.py
- FOUND: src/models/mawc_conservative_report.py
- FOUND: src/models/templates/mawc_conservative_report.html
- FOUND: src/models/mawc_conservative_retrainer.py
- FOUND: tests/test_mawc_conservative_retrainer.py
- FOUND: .planning/phases/45-structural-fix/45-02-SUMMARY.md
- FOUND: f49e552 (Task 1 commit)
- FOUND: 1fbae0f (Task 2 commit)
