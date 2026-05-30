---
phase: 44-roi-bisect
verified: 2026-05-30T17:30:00Z
status: passed
score: 11/11 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 44: ROI Bisect Verification Report

**Phase Goal:** DeploymentGate FAIL no direct cause wo component unit (MAWC/Ranker/OBF/Selection) de attribute shi, MAWC/Ranker coefficient analysis de worsening contribution features wo specific suru
**Verified:** 2026-05-30T17:30:00Z
**Status:** PASSED
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | ECE degradation segmented by odds_band/popularity_band/probability_rank_band with MAWC direct vs selection population attribution | VERIFIED | `attribute_ece_degradation()` iterates 3 segment types, computes per-segment ECE delta + p_win shift, classifies as MAWC direct or selection population (lines 135-206) |
| 2 | APR deviation separated into all-horse vs selected-horse, attributing MAWC probability level vs Ranker selection bias | VERIFIED | `attribute_apr_deviation()` computes both APRs, compares delta magnitudes for attribution (lines 212-271) |
| 3 | Bet count loss decomposed into Ranker exclusion count + OBF pass-through rate changes | VERIFIED | `attribute_bet_count_loss()` counts selection_changed races, excluded_by_ranker, and `_analyze_obf_impact()` (lines 277-401) |
| 4 | MAWC 51-dim coef_ analyzed for logit_market/logit_model/interaction segment contributions | VERIFIED | `analyze_mawc_coefficients()` extracts 51 features, `_compute_segment_contributions()` computes effective_market_contribution per segment (lines 407-537) |
| 5 | Ranker Ridge coefficients analyzed for relevance (15-dim) + value (15-dim) feature weights | VERIFIED | `analyze_ranker_coefficients()` extracts both scorer coef_ arrays, identifies top features (lines 539-617) |
| 6 | changed/dropped/retained race groups compared for MAWC and Ranker coefficient contribution | VERIFIED | `analyze_segment_coefficient_contribution()` splits by selected_changed, computes per-group p_win delta stats (lines 623-694) |
| 7 | v1.7(Phase 34)->v2.0(Phase 38) artifact-level comparison with degradation phase estimation | VERIFIED | `HistoricalBisect.run_historical_comparison()` uses multi_year_result.json + git log + OOF metrics, estimates Phase 35-36 as source (lines 286-395) |
| 8 | CLI generates bisect_result.json | VERIFIED | `save_attribution_results()` creates JSON with generated_at, ece/apr/bet_count attribution, coefficient summary, upstream check, recommendations. CLI `main()` calls it (lines 964-1223) |
| 9 | coefficient_analysis.json has structured MAWC/Ranker output | VERIFIED | `_build_coefficient_analysis_dict()` outputs mawc.per_feature (51 items), mawc.per_segment, ranker.relevance, ranker.value, segment_contribution (lines 999-1025) |
| 10 | bisect_summary.md has 6 sections including Historical Context and Phase 45 recommendations | VERIFIED | `_build_bisect_summary_md()` produces 6 sections: ECE Attribution, APR Attribution, Bet Count Attribution, Coefficient Analysis, Historical Context, Recommendations for Phase 45 (lines 1028-1177) |
| 11 | HTML report has 4 attribution sections + recommendations | VERIFIED | `component_attribution_report.html` Jinja2 template has sections: ECE Attribution, APR Attribution, Bet Count Attribution, Coefficient Analysis, Recommendations (407 lines). `ComponentAttributionReportGenerator.generate()` renders it |

**Score:** 11/11 truths verified

### ROADMAP Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| SC1 | Phase 34-38 artifact-level bisect identifies ROI degradation phase(s) | VERIFIED | `HistoricalBisect` loads multi_year_result.json, runs git log v1.7..v2.0, categorizes commits by phase, estimates Phase 35-36 as degradation source with confidence level |
| SC2 | Non-reproducible phases documented via git diff/OOF/BT logs/existing artifacts | VERIFIED | `_estimate_degradation_phase()` produces documented estimate with confidence (LOW/MEDIUM). `auxiliary_findings` records ROI delta, commit counts, OOF metrics. Falls back to "known timeline only" when git tags unavailable |
| SC3 | Degradation phase OOF feature contribution (SHAP/gain) compared, worsening features/parameters identified | VERIFIED (coefficient-level) | MAWC 51-dim coefficient analysis identifies logit_market (coef=0.39, beta_market=0.90) as dominant. Ranker analysis identifies if_p_win_final (0.80) and if_ev_calibrated (0.83) as dominant. Conditional upstream SHAP/gain check implemented as anomaly gate (_check_upstream_anomaly). Note: actual SHAP/gain computation is conditional-only, not unconditional -- coefficient analysis provides the primary feature attribution |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/backtest/component_attribution.py` | ComponentAttribution + output functions | VERIFIED | 1222 lines. Exports: ComponentAttribution, ComponentAttributionResult, CoefficientAnalysisResult, save_attribution_results |
| `src/backtest/historical_bisect.py` | HistoricalBisect auxiliary comparison | VERIFIED | 395 lines. Exports: HistoricalBisect, HistoricalBisectResult |
| `src/backtest/component_attribution_report.py` | Report generator (separate module) | VERIFIED | 75 lines. Exports: ComponentAttributionReportGenerator |
| `scripts/run_component_attribution.py` | CLI entry point | VERIFIED | 178 lines. Exports: build_parser, main. --help exits 0 |
| `src/backtest/templates/component_attribution_report.html` | Jinja2 HTML template | VERIFIED | 407 lines. 4 sections + recommendations + historical context |
| `tests/test_component_attribution.py` | Attribution unit tests | VERIFIED | 894 lines (min 200 required). 30 test methods across 13 test classes |
| `tests/test_historical_bisect.py` | Historical bisect tests | VERIFIED | 241 lines. 7 test methods across 4 test classes |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `component_attribution.py` | `shadow_horse_diff.parquet` | `pd.read_parquet(input_dir / "shadow_horse_diff.parquet")` | WIRED | Line 93 |
| `component_attribution.py` | `shadow_diagnosis_result.json` | `json.loads(diag_path.read_text(...))` | WIRED | Line 101-104 |
| `component_attribution.py` | `market_aware_win_calibrator_*.joblib` | `joblib.load(model_path)` + `coef_` extraction | WIRED | Lines 429-441, 454 |
| `component_attribution.py` | `shadow_diagnosis.py` segment constants | `from backtest.shadow_diagnosis import ODDS_BAND_EDGES, ...` | WIRED | Line 27-34 |
| `historical_bisect.py` | `multi_year_result.json` | `json.loads(myr_path.read_text(...))` | WIRED | Lines 74-77 |
| `run_component_attribution.py` | `component_attribution.py` | `from backtest.component_attribution import ComponentAttribution, save_attribution_results` | WIRED | Line 78-81 |
| `component_attribution_report.py` | `component_attribution_report.html` | `FileSystemLoader(template_dir)` + Jinja2 render | WIRED | Lines 17, 55 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `ComponentAttribution` | `self.horse_diff` | `pd.read_parquet(shadow_horse_diff.parquet)` | Real shadow comparison horse-level diffs | FLOWING |
| `ComponentAttribution` | `self.race_diff` | `pd.read_parquet(shadow_race_diff.parquet)` | Real shadow comparison race-level diffs | FLOWING |
| `ComponentAttribution` | MAWC coef (51-dim) | `joblib.load(mawc joblib)` -> `state["calibrator"].coef_[0]` | Real trained model coefficients | FLOWING |
| `ComponentAttribution` | Ranker coef (15+15-dim) | `joblib.load(ranker joblib)` -> `scorer.coef_` | Real trained Ridge coefficients | FLOWING |
| `HistoricalBisect` | `multi_year_result` | `json.loads(multi_year_result.json)` | Real backtest results | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All tests pass | `python -m pytest tests/test_component_attribution.py tests/test_historical_bisect.py -v` | 37 passed in 2.49s | PASS |
| Lint clean | `python -m ruff check` (4 files) | All checks passed! | PASS |
| CLI --help | `python scripts/run_component_attribution.py --help` | Exit code 0, shows all 4 args | PASS |
| Import check (ComponentAttribution) | `python -c "from backtest.component_attribution import ..."` | OK, all 3 exports accessible | PASS |
| Import check (HistoricalBisect) | `python -c "from backtest.historical_bisect import ..."` | OK, both exports accessible | PASS |
| Import check (ReportGenerator) | `python -c "from backtest.component_attribution_report import ..."` | OK | PASS |

### Probe Execution

No probes defined for this phase. Phase 44 is an analysis/reporting phase (no migration, no CLI tooling that requires probe validation).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| BISECT-01 | 44-01 | v1.7->v2.0 artifact-level bisect identifying ROI degradation phase | SATISFIED | `HistoricalBisect` class with `run_historical_comparison()` producing `HistoricalBisectResult` with `estimated_degradation_phase` and `confidence` |
| BISECT-02 | 44-01 | Degradation phase OOF feature contribution (SHAP/gain) identifying worsening features/parameters | SATISFIED | MAWC 51-dim coefficient analysis + Ranker Ridge coefficient analysis + conditional upstream anomaly check. Feature-level attribution identifies logit_market dominance (beta=0.90), if_p_win_final (rel=0.80), if_ev_calibrated (val=0.83) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers found in any phase 44 file |

### Notable Observations

1. **Uncommitted change in shadow_diagnosis.py** (46 insertions, 8 deletions): A `_resolve_variant_col()` static method was added to `ShadowDiagnosis` to handle cross-variant column name resolution in race_diff vs horse_diff. This appears to be a Phase 43.5 compatibility fix (P0-6) that was committed as `0cf81d8` but is showing as modified in working tree. Not a phase 44 artifact but worth noting.

2. **Conditional SHAP/gain**: The upstream anomaly check uses coefficient magnitude thresholds (>1.5 for MAWC, >1.0 for Ranker relevance, >0.95 for beta_market) rather than performing actual SHAP/gain computation. This is the designed behavior per D-03 clause 4 ("conditional only"). The coefficient analysis itself provides the feature-level attribution that BISECT-02 requires.

3. **Output artifacts not pre-generated**: The bisect_result.json, coefficient_analysis.json, bisect_summary.md, and HTML report are generated at CLI runtime. The CLI + output layer is fully tested with synthetic fixtures (30 tests verify JSON structure, MD sections, HTML content).

### Human Verification Required

None. All truths are programmatically verified. The phase produces analysis code + tests + CLI -- no visual or runtime behavior that requires human judgment.

### Gaps Summary

No gaps found. All 11 must-haves verified across both plans. All 3 ROADMAP success criteria satisfied. 37 tests pass, lint clean, CLI functional.

---
_Verified: 2026-05-30T17:30:00Z_
_Verifier: Claude (gsd-verifier)_
