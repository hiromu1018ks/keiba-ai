---
phase: 33-gain-per-depth-diagnostic
verified: 2026-05-18T12:00:00Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
---

# Phase 33: Gain per Depth Diagnostic Verification Report

**Phase Goal:** Build Gain per Depth (GPD) diagnostic tool that analyzes LightGBM tree structures to validate the Two-Stage hypothesis -- that Market features dominate shallow tree depths while Fundamental features activate at deeper levels.
**Verified:** 2026-05-18T12:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1: LightGBM trees_to_dataframe()でdepth別gain寄与率が集計され、Market/Fundamental/Categoricalの3分類でレポートが出力される | VERIFIED | `_compute_depth_gains()` calls `booster.trees_to_dataframe()` (line 343), filters leaves, maps features via FEATURE_CATEGORY_MAP, groups by (depth, category). JSON output via `compute_gpd_diagnostics()`. 179 features classified: 41 market + 119 fundamental + 19 categorical. Full coverage verified: 0 unregistered, 0 extra features. |
| 2 | SC2: StackedEnsemble内のLightGBMモデルにアクセスし、depth別分析が実行できる | VERIFIED | `_extract_boosters()` unwraps StackedEnsemble via `.lgbm_model` attribute (lines 270-271). Test `test_stacked_ensemble_unwrapped` verifies `ensemble_lgbm_turf` and `ensemble_lgbm_dirt` keys in output. |
| 3 | SC3: 暗黙的Two-Stage構造(上位depth=Market, 下位depth=Fundamental)の仮説がデータで検証される | VERIFIED | `_compute_market_dominance_ratio()` computes MDR = Market_share(depth 1-3) - Market_share(depth 4+). `_compute_fundamental_activation_depth()` computes FAD = min depth where Fundamental > Market. Both metrics are output per model in JSON and console_summary. console_summary produces no PASS/FAIL judgment per D-12 -- outputs metrics for human interpretation. |
| 4 | FEATURE_CATEGORY_MAPに全モデルFEATURE_COLSの全特徴量が登録され、未登録特徴量0でテストが通る | VERIFIED | Runtime verification: 179 total features across all 9 model classes, 179 in map, 0 unregistered, 0 extra. Test `test_all_features_registered` passes. |
| 5 | MDR/FAD数値として出力される | VERIFIED | `compute_gpd_diagnostics()` outputs both per model (lines 512-519). Tests verify positive/negative/None MDR and correct FAD detection. |
| 6 | CLIスクリプト run_gpd.py が学習済みモデルを読み込みGPD診断を実行する | VERIFIED | `run_gpd.py` imports ModelLoader, calls `loader.load_from_dir()` then `compute_gpd_diagnostics()`. `--help` works. Tests verify main() calls all expected functions with correct args. |
| 7 | matplotlib stacked bar + cumulative gain line PNGがモデル毎に生成される | VERIFIED | `plot_gpd_charts()` generates one PNG per model with 2 subplots (stacked bar top, cumulative line bottom), 3-color scheme, MDR/FAD annotation. Tests verify PNG creation, valid PNG header, per-model naming, edge cases. |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/models/gpd_diagnostics.py` | GPD diagnostic module | VERIFIED | 589 lines. Contains FEATURE_CATEGORY_MAP (179 entries), _extract_boosters(), _compute_depth_gains(), MDR/FAD computation, compute_gpd_diagnostics(), console_summary(). All functions are substantive implementations, not stubs. |
| `tests/test_gpd_diagnostics.py` | Test coverage (min 200 lines) | VERIFIED | 582 lines. 19 tests across 6 test classes covering feature map completeness, booster extraction, depth-gain computation, MDR, FAD, full pipeline. |
| `scripts/run_gpd.py` | CLI entry point | VERIFIED | 259 lines. argparse with --models-dir/--output-dir/--ensemble, plot_gpd_charts() with stacked bar + cumulative gain, main() orchestration. |
| `tests/test_run_gpd.py` | CLI + visualization tests (min 100 lines) | VERIFIED | 412 lines. 14 tests across 6 test classes covering CLI parsing, PNG generation, per-model output, edge cases, directory creation, integration. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/models/gpd_diagnostics.py` | `src/domain/models.py::TrainedModelsV5` | `_extract_boosters()` iteration over submodels | WIRED | Line 23: `from domain.models import TrainedModelsV5`. Line 260: `models.submodels.items()`. |
| `src/models/gpd_diagnostics.py` | `lightgbm.Booster.trees_to_dataframe()` | Direct call in `_compute_depth_gains()` | WIRED | Line 343: `booster.trees_to_dataframe()`. |
| `scripts/run_gpd.py` | `src/db/model_loader.py::ModelLoader` | `load_from_dir()` for model loading | WIRED | Line 42: `from db.model_loader import ModelLoader`. Line 226-229: `loader.load_from_dir()`. |
| `scripts/run_gpd.py` | `src/models/gpd_diagnostics.py` | `compute_gpd_diagnostics()` + `console_summary()` | WIRED | Lines 43-45: imports. Lines 233-235: calls. |
| `scripts/run_gpd.py` | `data/gpd/` | PNG + JSON output files | WIRED | Line 211: `fig.savefig()`. Line 526-528: `json.dump()`. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `gpd_diagnostics.py` | `FEATURE_CATEGORY_MAP` | Module-level constant (179 entries) | Yes -- hardcoded classification verified against all model FEATURE_COLS | FLOWING |
| `_compute_depth_gains()` | `tree_df` | `booster.trees_to_dataframe()` | Yes -- direct LightGBM API call producing split node data | FLOWING |
| `plot_gpd_charts()` | `result["models"]` | `compute_gpd_diagnostics()` output | Yes -- per-model depth/category/gain aggregated data | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All GPD tests pass | `python -m pytest tests/test_gpd_diagnostics.py tests/test_run_gpd.py -v` | 33 passed, 0 failed | PASS |
| FEATURE_CATEGORY_MAP completeness | `python -c "... verify coverage ..."` | 179 total, 0 unregistered, 0 extra | PASS |
| CLI --help works | `python scripts/run_gpd.py --help` | Prints usage with --models-dir, --output-dir, --ensemble | PASS |
| Module exports callable | `python -c "from models.gpd_diagnostics import compute_gpd_diagnostics, console_summary"` | Both are functions | PASS |
| Ruff lint clean | `ruff check src/models/gpd_diagnostics.py scripts/run_gpd.py tests/` | All checks passed | PASS |

### Probe Execution

Step 7c: SKIPPED (no probe scripts defined for this phase -- diagnostic tooling phase, not migration)

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GPD-01 | 33-01 | LightGBM trees_to_dataframe() でdepth別gain寄与率を集計する機能 | SATISFIED | `_compute_depth_gains()` calls `trees_to_dataframe()`, groups by (depth, category). |
| GPD-02 | 33-02 | Market/Fundamental/Categorical 3分類でdepth別シェアを可視化する機能 | SATISFIED | `plot_gpd_charts()` generates stacked bar + cumulative line PNGs per model. |
| GPD-03 | 33-01 | StackedEnsemble内LightGBMモデルへのアクセスと分析機能 | SATISFIED | `_extract_boosters()` unwraps StackedEnsemble via `.lgbm_model`. |
| GPD-04 | 33-01 | 暗黙的Two-Stage構造の検証 | SATISFIED | MDR and FAD metrics computed per model. console_summary outputs metrics for human interpretation. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | -- | -- | -- | -- |

No TBD/FIXME/XXX, no TODO/HACK/PLACEHOLDER, no empty implementations, no hardcoded empty data, no console.log-only handlers found in any phase file.

### Human Verification Required

None required. All truths verified programmatically. The MDR/FAD metrics and PNG visualizations are diagnostic outputs for human interpretation, but their generation and correctness is fully verified by automated tests.

---

_Verified: 2026-05-18T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
