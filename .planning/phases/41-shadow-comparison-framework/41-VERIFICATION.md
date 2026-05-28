---
phase: 41-shadow-comparison-framework
verified: 2026-05-28T13:00:00Z
status: passed
score: 11/11 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 41: Shadow Comparison Framework Verification Report

**Phase Goal:** Fixed-fold 2024/2025 baseline vs shadow comparison tracking probability quality, selection agreement, CLV, ROI
**Verified:** 2026-05-28T13:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### ROADMAP Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Shadow comparison runs BacktestEngine twice (baseline vs shadow) on both 2024 and 2025 with fixed folds | VERIFIED | `ShadowComparisonFramework.run_fold()` at line 489 iterates `self.variants`, loads TrainedModelsV5 per variant via `ModelLoader.load_from_dir()`, injects `_shadow_flags`, constructs `BacktestEngine` per variant. `run()` defaults to `FoldDefinition.create_folds([2024, 2025])` (line 569). |
| 2 | Comparison tracks Brier, logloss, ECE, selection agreement, CLV, ROI, HR, DD, bet count | VERIFIED | `ComparisonMetrics` dataclass (line 377) has all 12 fields. `compute_metrics()` (line 727) computes Brier (line 781), logloss (line 785), ECE (line 791), ROI (line 747), HR (line 748), bet_count (line 746), max_drawdown (line 754), CLV (line 757), selection_agreement (line 809), avg_investment_score (line 803), actual_predicted_ratio (line 796). |
| 3 | Selection horse differences measured and explainable per-race | VERIFIED | `_align_race_level()` (line 576) merges by race_id, produces `selected_changed` column (line 638), `baseline_selected_umaban`, `shadow_selected_umaban`. `_align_horse_level()` (line 656) produces per-horse probability/score columns. `save_results()` computes `metrics_by_selected_changed` by filtering bet_history per group (line 142-162). |

### Observable Truths (Plan 01)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | ShadowComparisonFramework runs BacktestEngine twice with different TrainedModelsV5 and fixed 2024/2025 folds | VERIFIED | `run_fold()` (line 489) loops over variants, loads different models from `variant_cfg.model_dir / str(fold.year)`, constructs separate BacktestEngine per variant. `FoldDefinition.create_folds([2024, 2025])` (line 343) produces correct fold definitions. |
| 2 | Feature flags in RacePredictor disable MAWC and ranker independently for baseline runs | VERIFIED | RacePredictor constructor (line 100) accepts `enable_market_aware_calibrator: bool = True` and `enable_race_level_ranker: bool = True`. `_shadow_flags` propagation (line 133-140). MAWC guard (line 288) checks `self.enable_market_aware_calibrator`. Ranker guard (line 309) checks `self.enable_race_level_ranker`. |
| 3 | Post-hoc alignment merges baseline and shadow bet_history at race-level and horse-level | VERIFIED | `_align_race_level()` (line 576) merges by race_id with suffixes. `_align_horse_level()` (line 656) merges by race_id + umaban, using variant-specific column prefixes `{vname}_col` to prevent N-way collision. |
| 4 | Metrics computation produces Brier, logloss, ECE, ROI, HR, DD, bet count, CLV, selection agreement | VERIFIED | All 12 metrics computed in `compute_metrics()` (line 727-813). `_compute_ece()` (line 889) uses 10-bin equal-width with `<=` on last bin (WR-01 from review was addressed). |
| 5 | Selection agreement measures per-race baseline vs shadow horse selection differences | VERIFIED | `_compute_selection_agreement()` (line 882) computes `1.0 - race_diff["selected_changed"].mean()`. Per-race `selected_changed` column produced in `_align_race_level()`. |

### Observable Truths (Plan 02)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 6 | JSON metrics file contains aggregate and grouped metrics for both baseline and shadow | VERIFIED | `save_results()` (line 54) writes `shadow_comparison_result.json` with per-fold metrics, `metrics_by_surface`, `metrics_by_odds_band`, `metrics_by_selected_changed`, and overall aggregated metrics (line 214-221). |
| 7 | Parquet race diff and horse diff files contain aligned comparison data | VERIFIED | `save_results()` writes `shadow_race_diff.parquet` (line 230) and `shadow_horse_diff.parquet` (line 242) with `fold_year` column. |
| 8 | CSV race diff available for quick human inspection | VERIFIED | `save_results()` writes `shadow_race_diff.csv` (line 236) with `utf-8-sig` encoding. |
| 9 | HTML report shows side-by-side baseline vs shadow summary with fold breakdown | VERIFIED | `ShadowComparisonReportGenerator` (shadow_report.py) renders `shadow_comparison_report.html` template with Overall Summary, Fold Breakdown, Surface Breakdown, Odds Band, Selection Agreement, Calibration sections. Template at templates/shadow_comparison_report.html (332 lines). |
| 10 | CLI runs end-to-end: load models, run folds, produce all output artifacts | VERIFIED | `scripts/run_shadow_comparison.py` (234 lines) with argparse CLI. Baseline flags=False (line 128-130), shadow flags=True (line 133-136). Calls `framework.run()`, `save_results()`, `save_manifest()`, optionally `ShadowComparisonReportGenerator`. `--help` prints correctly. |
| 11 | shadow_manifest.json records model dirs, flag states, artifact hashes, fold definitions | VERIFIED | `save_manifest()` (line 257) writes variants with model_dir/flag_states, fold definitions from results, artifacts with SHA256 hashes via `_compute_sha256()`, baseline_definition per D-22 (line 289-292). |

**Score:** 11/11 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `src/backtest/shadow_comparison.py` | Framework, dataclasses, alignment, metrics | VERIFIED (913 lines) | All 5 dataclasses, ShadowComparisonFramework class with run_fold/run/_align_race_level/_align_horse_level/compute_metrics/compute_metrics_by_group/save_results/save_manifest |
| `src/backtest/race_predictor.py` | Feature flag injection | VERIFIED | `enable_market_aware_calibrator`/`enable_race_level_ranker` constructor args, `_shadow_flags` propagation, MAWC/ranker guards |
| `src/backtest/shadow_report.py` | HTML report generator | VERIFIED (139 lines) | ShadowComparisonReportGenerator with Jinja2 FileSystemLoader |
| `src/backtest/templates/shadow_comparison_report.html` | Jinja2 template | VERIFIED (332 lines) | Variants, Overall, Fold, Surface, Odds Band, Selection Agreement, Calibration sections |
| `scripts/run_shadow_comparison.py` | CLI entry point | VERIFIED (234 lines) | All D-07 arguments, baseline/shadow variant configs, save_results/save_manifest/report |
| `tests/test_shadow_comparison.py` | Unit + integration tests | VERIFIED (1465 lines) | 52 tests covering all components |
| `tests/test_shadow_report.py` | Report tests | VERIFIED (410 lines) | 9 tests covering HTML generation |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `shadow_comparison.py` | `engine.py` | BacktestEngine constructor + run() | WIRED | Line 491: `from backtest.engine import BacktestEngine`, line 515: `engine = BacktestEngine(...)`, line 523: `engine.run(...)` |
| `shadow_comparison.py` | `model_loader.py` | ModelLoader.load_from_dir() | WIRED | Line 501: `loader.load_from_dir(model_dir)` |
| `race_predictor.py` | MAWC | Feature flag guard | WIRED | Line 288: `if self.enable_market_aware_calibrator and mawc is not None and mawc.is_trained` |
| `race_predictor.py` | ranker | Feature flag guard | WIRED | Line 309: `if self.enable_race_level_ranker and ranker is not None and ranker.is_trained` |
| `run_shadow_comparison.py` | `shadow_comparison.py` | import | WIRED | Line 115-121: imports ShadowComparisonFramework, FoldDefinition, VariantConfig, save_manifest, save_results |
| `shadow_report.py` | template | Jinja2 FileSystemLoader | WIRED | Line 44: `FileSystemLoader(str(self.template_dir))`, line 47: `env.get_template("shadow_comparison_report.html")` |
| `shadow_comparison.py` | TrainedModelsV5 | _shadow_flags field | WIRED | Line 510: `loaded_models._shadow_flags = {...}`, field defined in models.py line 287 |
| `race_predictor.py` | TrainedModelsV5 | _shadow_flags read | WIRED | Line 133: `getattr(models, "_shadow_flags", None)`, line 135-139: reads flag values |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| `shadow_comparison.py:run_fold` | results dict | BacktestEngine.run() per variant | Yes -- real BacktestResult with bet_history | FLOWING |
| `shadow_comparison.py:_align_race_level` | race_diff DataFrame | bet_history -> DataFrame -> merge by race_id | Yes -- produces selected_changed, baseline/shadow columns | FLOWING |
| `shadow_comparison.py:_align_horse_level` | horse_diff DataFrame | bet_history -> DataFrame -> merge by race_id+umaban | Yes -- produces variant-prefixed probability/score columns | FLOWING |
| `shadow_comparison.py:compute_metrics` | ComparisonMetrics | race_diff + horse_diff + bet_history | Yes -- computes all 12 metrics from real data | FLOWING |
| `race_predictor.py` | p_win_final | p_win_corrected with race normalization (baseline path) | Yes -- groupby race_id normalization | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| All tests pass | `python -m pytest tests/test_shadow_comparison.py tests/test_shadow_report.py -v` | 61/61 passed in 3.15s | PASS |
| Backward compatibility | `python -m pytest tests/test_backtest_engine.py tests/test_backtest_engine_autocalibrate.py -q` | 76/77 passed (1 pre-existing failure) | PASS |
| CLI help | `python scripts/run_shadow_comparison.py --help` | Prints correct usage with all D-07 arguments | PASS |
| Ruff lint | `ruff check src/backtest/shadow_comparison.py src/backtest/shadow_report.py scripts/run_shadow_comparison.py` | All checks passed | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| SHD-01 | 41-01, 41-02 | Shadow comparison runs baseline vs shadow TrainedModelsV5 on 2024/2025 with fixed folds | SATISFIED | ShadowComparisonFramework.run_fold() loads models per variant, runs BacktestEngine twice. FoldDefinition.create_folds([2024, 2025]). CLI --folds default [2024, 2025]. |
| SHD-02 | 41-01, 41-02 | Comparison tracks Brier, logloss, ECE, selection agreement, CLV, ROI, HR, DD, bet count | SATISFIED | ComparisonMetrics has all required fields. compute_metrics() computes all of them. save_results() outputs them in JSON. |
| SHD-03 | 41-01, 41-02 | Selection horse differences measured and explainable per-race | SATISFIED | _align_race_level() produces selected_changed per race. metrics_by_selected_changed computed in save_results(). HTML report shows selection-change examples. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `shadow_comparison.py` | 838 | `# Would need prob rank data - skip if not available` -- prob_rank_band grouping returns empty | Warning | `compute_metrics_by_group("prob_rank_band")` always returns {}. Not a success criterion (D-13 optional dimension). |
| `race_predictor.py` | 1051, 1056, 1192 | `# TODO: Regime` comments | Info | Pre-existing from earlier phases. Not introduced by Phase 41. |

**Code Review Status:** The REVIEW.md found 2 critical issues (CR-01: N-way column collision, CR-02: metrics_by_selected_changed wrong metrics). Both were fixed in the implementation. The traceback import (WR-02) and dynamic attribute (WR-03) were also addressed. WR-01 (ECE last bin boundary) was fixed with the `<=` condition on the last bin.

### Human Verification Required

No human verification items identified. All truths are verifiable programmatically through code inspection and test execution.

### Gaps Summary

No blocking gaps found. All 11 must-have truths are verified with code evidence. All 3 ROADMAP success criteria are met. All 3 requirements (SHD-01, SHD-02, SHD-03) are satisfied.

Minor note: `prob_rank_band` grouping is a stub (returns empty dict) but is not a success criterion or must-have truth. It's an optional D-13 aggregation dimension that could be implemented in a future enhancement if needed.

---

_Verified: 2026-05-28T13:00:00Z_
_Verifier: Claude (gsd-verifier)_
