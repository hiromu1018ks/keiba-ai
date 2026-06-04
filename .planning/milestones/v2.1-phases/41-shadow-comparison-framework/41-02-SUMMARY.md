---
phase: 41-shadow-comparison-framework
plan: 02
subsystem: backtest
tags: [shadow-comparison, output-artifacts, html-report, cli, manifest, sha256, jinja2]
dependency_graph:
  requires: [ShadowComparisonFramework, FoldDefinition, VariantConfig, ComparisonMetrics, ShadowComparisonResult]
  provides: [save_results, save_manifest, ShadowComparisonReportGenerator, shadow_comparison_report.html, run_shadow_comparison.py]
  affects: [src/backtest/shadow_comparison.py, src/backtest/shadow_report.py, src/backtest/templates/shadow_comparison_report.html, scripts/run_shadow_comparison.py]
tech_stack:
  added: []
  patterns: [JSON manifest with SHA256 hashes, Jinja2 HTML report, Parquet source-of-truth, CLI argparse entry point]
key_files:
  created:
    - src/backtest/shadow_report.py
    - src/backtest/templates/shadow_comparison_report.html
    - scripts/run_shadow_comparison.py
    - tests/test_shadow_report.py
  modified:
    - src/backtest/shadow_comparison.py
    - tests/test_shadow_comparison.py
decisions:
  - save_results and save_manifest as module-level functions (not class methods) for clean import
  - ShadowComparisonReportGenerator follows BacktestReportGenerator Jinja2 pattern
  - HTML is self-contained with inline CSS, no external dependencies
  - JSON/Parquet are source of truth; HTML is human review only (D-17)
  - Baseline variant definition recorded in manifest per D-22
  - Grouped metrics include surface, odds_band, value_score_band, selected_changed per D-13
metrics:
  duration: 14m
  completed: "2026-05-28T10:12:00Z"
  tasks: 3
  tests: 61
  files: 6
  loc_added: 3488
---

# Phase 41 Plan 02: Shadow Comparison Output Artifacts & CLI Summary

Output artifact pipeline (JSON metrics, Parquet race/horse diff, CSV, HTML report, SHA256 manifest) and CLI entry point for end-to-end shadow comparison execution.

## Completed Tasks

| Task | Name | Status |
|------|------|--------|
| 1 | Add output artifact methods to ShadowComparisonFramework | Done |
| 2 | Create ShadowComparisonReportGenerator and Jinja2 HTML template | Done |
| 3 | Create CLI script and integration test | Done |

## Key Artifacts

### src/backtest/shadow_comparison.py (modified, +288 lines)
- **save_results()**: Module-level function writing 4 output artifacts:
  - `shadow_comparison_result.json` with overall + grouped metrics (surface, odds_band, prob_rank_band, value_score_band, selected_changed) per D-13
  - `shadow_race_diff.parquet` + `shadow_race_diff.csv` with fold_year column per D-11
  - `shadow_horse_diff.parquet` with horse-level alignment data per D-11
- **save_manifest()**: Module-level function writing `shadow_manifest.json` per D-20, D-22:
  - Variants with model dirs, flag states, baseline_definition (D-22)
  - Fold definitions from comparison results
  - Artifacts with SHA256 hashes computed from actual file contents
  - Metric definitions (Brier, logloss, ECE, selection agreement, CLV)
- **_compute_sha256()**: Binary-mode SHA256 hash helper
- **_metrics_to_dict()**: ComparisonMetrics to JSON-compatible dict converter

### src/backtest/shadow_report.py (new, 139 lines)
- **ShadowComparisonReportGenerator**: Follows BacktestReportGenerator pattern
  - Constructor creates output_dir, stores template_dir
  - `generate()` builds context dict and renders Jinja2 template
  - Sections: variants info, overall summary, fold breakdown, surface/odds band, selection agreement with examples, calibration metrics
  - Footer notes JSON/Parquet are source of truth per D-17

### src/backtest/templates/shadow_comparison_report.html (new, 332 lines)
- Self-contained HTML with inline CSS (dark header, alternating rows, responsive)
- Sections: Variants table, Overall Summary, Fold Breakdown, Surface Breakdown, Odds Band Breakdown, Selection Agreement with examples, Calibration Metrics, Footer
- No external CDN links (no Chart.js, no DataTables)

### scripts/run_shadow_comparison.py (new, 234 lines)
- CLI entry point per D-07 with argparse
- Arguments: --baseline-root, --shadow-root, --folds, --train-window, --betting-target, --output-dir, --report, --baseline-name, --shadow-name, --betting-mode
- Baseline variant: MAWC=False, ranker=False per D-18
- Shadow variant: MAWC=True, ranker=True
- Default folds: [2024, 2025] with 4-year train window per D-05
- --report flag triggers ShadowComparisonReportGenerator
- Human-readable summary printed to stdout

### tests/test_shadow_comparison.py (modified, +544 lines)
- TestSaveResults: 7 tests for JSON, Parquet, CSV, multi-fold, grouped metrics
- TestSaveManifest: 6 tests for variants, SHA256 hashes, fold definitions, baseline definition, metric definitions
- TestCLIScript: 6 tests for baseline/shadow flags, fold definitions, full flow, report flag, help output

### tests/test_shadow_report.py (new, 410 lines)
- 9 tests covering: output dir creation, HTML file generation, overall summary, selection agreement, calibration, variant names, self-contained check, fold breakdown, source-of-truth footer

## Decisions Made

1. **save_results/save_manifest as module-level functions**: Cleaner import path than static methods on ShadowComparisonFramework, avoids constructor dependency.
2. **Grouped metrics computed at save time**: Surface, odds_band, value_score_band, selected_changed groupings computed from horse_diff/race_diff during save_results() call.
3. **Self-contained HTML**: No Chart.js, no DataTables — simpler template without external dependencies, consistent with D-16 "human review only" purpose.
4. **CLI imports inside main()**: Deferred imports for framework classes in run_shadow_comparison.py to match existing script patterns.

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

- `python -m pytest tests/test_shadow_comparison.py tests/test_shadow_report.py -v`: 61/61 passed
- `python scripts/run_shadow_comparison.py --help`: CLI help prints correctly
- `python -m ruff check src/backtest/shadow_comparison.py src/backtest/shadow_report.py scripts/run_shadow_comparison.py`: All checks passed
- `python -m mypy src/backtest/shadow_report.py`: Pre-existing import resolution errors only (no new issues)

## Self-Check: PASSED

- FOUND: src/backtest/shadow_comparison.py
- FOUND: src/backtest/shadow_report.py
- FOUND: src/backtest/templates/shadow_comparison_report.html
- FOUND: scripts/run_shadow_comparison.py
- FOUND: tests/test_shadow_comparison.py
- FOUND: tests/test_shadow_report.py
- FOUND: commit d6b3805 (Task 1)
- FOUND: commit 5a0ca82 (Task 2)
- FOUND: commit 067345e (Task 3)
- FOUND: commit e1fe129 (Summary)
