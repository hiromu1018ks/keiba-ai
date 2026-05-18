---
phase: 33-gain-per-depth-diagnostic
plan: 02
subsystem: models
tags: [gpd, cli, visualization, matplotlib, diagnostics]
dependency_graph:
  requires: [33-01]
  provides: [run_gpd_cli, gpd_png_charts]
  affects: [scripts/run_gpd.py, tests/test_run_gpd.py]
tech_stack:
  added: [matplotlib (stacked bar + cumulative line), argparse CLI]
  patterns: [TDD RED/GREEN, Agg backend headless rendering]
key_files:
  created:
    - scripts/run_gpd.py
    - tests/test_run_gpd.py
  modified: []
decisions:
  - D-09: Category color scheme Market=#2196F3, Fundamental=#4CAF50, Categorical=#FF9800
  - Module-level imports for ModelLoader/gpd_diagnostics to enable mock patching
  - 2-subplot layout (stacked bar top, cumulative gain line bottom)
metrics:
  duration_min: 8
  completed: "2026-05-18"
  tasks_total: 1
  tasks_completed: 1
  tests_added: 14
  files_created: 2
  files_modified: 0
---

# Phase 33 Plan 02: GPD CLI and Visualization Summary

CLI script `run_gpd.py` loads trained models, runs GPD diagnostics, and generates per-model depth-by-category PNG charts with stacked bars (Market/Fundamental/Categorical) and cumulative gain lines. MDR and FAD metrics are annotated on each chart.

## What Was Built

### scripts/run_gpd.py
- **build_parser()**: argparse with `--models-dir` (default: data/models), `--output-dir` (default: data/gpd), `--ensemble` flag
- **plot_gpd_charts(result, output_dir)**: Generates one PNG per model with:
  - Top subplot: stacked bar chart showing Market/Fundamental/Categorical gain by tree depth
  - Bottom subplot: cumulative gain percentage lines per category (dashed, thicker)
  - MDR (Market Dominance Ratio) and FAD (Fundamental Activation Depth) annotated in yellow box
  - Saved as `gpd_{model_name}.png` at 150 DPI
  - Agg backend for headless operation, figures closed after saving
- **main()**: Loads models via ModelLoader, runs compute_gpd_diagnostics, calls console_summary, generates charts

### tests/test_run_gpd.py (14 tests)
- Test 1-3: CLI argument parsing (defaults, custom, Path type)
- Test 4-5: PNG generation (created, valid PNG header)
- Test 6-7: Per-model PNG output (count, naming convention)
- Test 8-11: Stacked bar edge cases (single depth, single category, zero gains, >20 depths)
- Test 12: Output directory auto-creation
- Test 13-14: Integration with compute_gpd_diagnostics (ensemble flag, default args)

## TDD Gate Compliance

| Gate | Commit | Hash |
|------|--------|------|
| RED | test(33-02): add failing tests | 51a141d |
| GREEN | feat(33-02): implement CLI + visualization | f61a204 |
| REFACTOR | N/A -- code is clean, no redundancy | -- |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

- `python -m pytest tests/test_run_gpd.py -v` -- 14/14 passed
- `python -m pytest tests/test_gpd_diagnostics.py tests/test_run_gpd.py -v` -- 33/33 passed
- `python -m ruff check scripts/run_gpd.py tests/test_run_gpd.py` -- All checks passed
- `python -m ruff format --check scripts/run_gpd.py tests/test_run_gpd.py` -- Passed
- `python scripts/run_gpd.py --help` -- Prints usage without error

## Self-Check: PASSED

- scripts/run_gpd.py: FOUND
- tests/test_run_gpd.py: FOUND
- 33-02-SUMMARY.md: FOUND
- Commit 51a141d (RED): FOUND
- Commit f61a204 (GREEN): FOUND
