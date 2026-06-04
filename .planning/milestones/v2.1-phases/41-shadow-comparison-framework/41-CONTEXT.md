# Phase 41: Shadow Comparison Framework - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a comparison infrastructure that runs BacktestEngine twice (baseline vs shadow) on fixed 2024/2025 test folds with comprehensive metrics tracking, enabling data-driven deployment decisions for the MarketAwareWinCalibrator and RaceLevelRanker introduced in Phases 39-40.

**In scope:** SHD-01 through SHD-03 (fixed-fold baseline vs shadow comparison on 2024/2025, comprehensive metrics tracking, selection agreement measurement with per-race explainability).
**Out of scope:** LightGBM LambdaRank ranker (deferred to v2.2), training inside comparison runner, deployment gate automation (DEP-01), place/wide model changes, regime-dependent comparison.

</domain>

<decisions>
## Implementation Decisions

### Comparison Runner Architecture

- **D-01:** ShadowComparisonFramework class in `src/backtest/shadow_comparison.py` with ShadowComparisonResult and ShadowRunConfig dataclasses. Thin CLI wrapper in `scripts/run_shadow_comparison.py`. Do NOT extend run_backtest.py — keep it focused on single-pipeline backtests.
- **D-02:** Pre-trained model directories only — baseline_dir and shadow_dir must already exist. No training inside the comparison framework. ModelLoader.load_from_dir() for both model sets. Training inside comparison runner makes failures harder to debug and doubles runtime.
- **D-03:** Post-hoc alignment at two levels: (1) race-level by race_id — compare baseline vs shadow selected umaban, selected_changed flag, odds/p/EV/score/result/return/stake; (2) horse-level by race_id + umaban — merge diagnostic rows for p_win, p_win_market_aware, investment_score, rank, selected flag. Run BacktestEngine twice with identical inputs, then merge artifacts.
- **D-04:** If bet_history lacks any required key/column for alignment, Phase 41 should add that diagnostic output explicitly rather than relying only on aggregate BacktestResult.
- **D-05:** Fixed 2-fold definitions matching established WF validation: Fold 2024 (train 2020-01-01 to 2023-12-31, test 2024-01-01 to 2024-12-31), Fold 2025 (train 2021-01-01 to 2024-12-31, test 2025-01-01 to 2025-12-31). CLI accepts --folds 2024 2025 with --baseline-root and --shadow-root resolving year-specific model subdirectories.
- **D-06:** Framework accepts variant_name / model_dir pairs so future shadow variants (e.g., LightGBM LambdaRank) can be added without redesign. Phase 41 only requires baseline vs Ridge ranker shadow. Reports are N-way capable in structure.

### CLI Design

- **D-07:** CLI arguments: --baseline-root, --shadow-root, --folds (year list), --train-window (default 4), --betting-target (default win), --output-dir (default data/backtest/shadow), --report (flag for HTML generation).
- **D-08:** Output directory structure: data/backtest/shadow/{fold_year}/ for per-fold artifacts.

### LightGBM LambdaRank Shadow

- **D-09:** Deferred to v2.2+. Phase 41 compares baseline vs Ridge ranker (Phase 40) only. Do not implement LightGBM LambdaRank training in Phase 41. Do not add skeleton code that cannot be trained or validated. Framework's N-way design (D-06) accommodates future addition.

### Metrics Output Format

- **D-10:** Five output artifacts:
  - `shadow_comparison_result.json` — aggregate metrics by fold/year, surface, odds_band, prob_rank_band, value_score_band, selected_changed vs unchanged, and overall
  - `shadow_race_diff.parquet` + `shadow_race_diff.csv` — one row per race with baseline/shadow selected horse, selected_changed, score components, odds, result, return, stake, CLV
  - `shadow_horse_diff.parquet` — one row per race_id/umaban with baseline/shadow probabilities, ranks, investment_score, selected flags, diagnostic components
  - `shadow_comparison_report.html` — side-by-side summary for human review via Jinja2 template
  - `shadow_manifest.json` — input model dirs, artifact hashes, code version, test date ranges, metric definitions, generated_at, flag states
- **D-11:** CSV for quick inspection, Parquet as source-of-truth for large diff tables, JSON for automation, HTML for human review.

### Metrics Collection

- **D-12:** Metrics tracked: ROI, hit rate, bet count, average odds, max drawdown, Brier, logloss, ECE, actual/predicted ratio, CLV (if available), selection agreement, average investment_score components.
- **D-13:** Aggregation dimensions: overall, fold/year, surface, odds_band, prob_rank_band, value_score_band (if ranker output exists), selected_changed vs unchanged.
- **D-14:** CLV computation: only when both betting_line_odds and closing_odds are present with valid pre/post timing. Formula: clv = closing_odds / betting_odds - 1 (decimal odds). Report as diagnostic only, not deployment gate. Output null with clv_available=false if inputs missing.
- **D-15:** Selection agreement = fraction of races where baseline and shadow select the same horse. This is a diagnostic metric, NOT a deployment gate (per PROJECT.md).

### HTML Report

- **D-16:** Dedicated shadow_comparison_report.html Jinja2 template following existing BacktestReportGenerator pattern. Contains: side-by-side baseline vs shadow summary, fold/year breakdown, surface/odds/prob-rank/value-band tables, selected_changed vs unchanged table, top selection-change examples with score component decomposition, calibration metrics section.
- **D-17:** JSON/Parquet remain source of truth. HTML is for human review only.

### Baseline Model Strategy

- **D-18:** Baseline = current pipeline with MarketAwareWinCalibrator and RaceLevelRanker explicitly disabled via feature flags. Not pre-Phase39 deleted code artifacts. The comparison framework supports reproducible baseline generation from current code via flags.
- **D-19:** Explicit runtime flags: enable_market_aware_calibrator (bool), enable_race_level_ranker (bool). If enable_market_aware_calibrator=false, RacePredictor uses legacy probability path even if MAWC artifact exists. If enable_race_level_ranker=false, RacePredictor uses existing selector stack even if ranker artifact exists.
- **D-20:** Flag states recorded in shadow_manifest.json. None fallback (is_trained / deployment_status) is acceptable for backward compatibility but baseline-vs-shadow comparison controlled by explicit flags.
- **D-21:** If flag=true but artifact is missing or deployment_status is not deployable, fail or fall back according to explicit strict/shadow mode config — not silently.
- **D-22:** Baseline definition recorded in shadow_manifest.json: "MAWC/ranker disabled, existing p_win_final + existing selector stack".

### Claude's Discretion

- Exact implementation of ShadowComparisonFramework internal methods and data flow.
- ShadowComparisonResult / ShadowRunConfig dataclass field design.
- Jinja2 template layout and styling (follow existing report patterns).
- Test structure and naming within existing conventions.
- Exact feature flag injection mechanism in RacePredictor (constructor arg vs config dict).
- Per-race diff table column selection and ordering.
- Statistical comparison utilities (paired metrics) if needed.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Components Being Compared
- `src/models/market_aware_win_calibrator.py` — Phase 39 calibrator. Shadow mode pattern (is_trained + deployment_status).
- `src/models/race_level_ranker.py` — Phase 40 ranker. Shadow mode pattern. Produces investment_score.
- `src/models/win_selection_gate.py` — WinSelectionGateModel. Baseline selector (preserved).
- `src/models/win_selection_policy.py` — WinSelectionPolicy. Baseline selector (preserved).
- `src/models/win_profit_selector.py` — WinProfitSelector. Baseline selector (preserved).

### Pipeline Integration Points
- `src/backtest/engine.py` — BacktestEngine. Constructor takes TrainedModelsV5. run() returns BacktestResult. Race loop at line 420+.
- `src/backtest/race_predictor.py` — RacePredictor. Shadow diagnostics block at lines 860-884 (baseline_selected_umaban vs ranker_selected_umaban, baseline_ranker_agreement). MAWC application at lines 269-277. Ranker scoring at lines 279-285.
- `src/db/model_loader.py` — ModelLoader.load_from_dir() at line 594. Loads from arbitrary directories. Handles per-surface optional models.
- `src/domain/models.py` — SubmodelSet (lines 234-273), TrainedModelsV5. Contains market_aware_win_calibrator and win_race_level_ranker fields.

### Report Infrastructure
- `src/backtest/report.py` — BacktestReportGenerator (Jinja2 pattern). MultiYearReportGenerator for aggregated reports.
- `src/backtest/templates/report.html` — Existing HTML report template. Pattern to follow.

### Feature Sources
- `src/investment/feature_frame.py` — InvestmentFeatureFrameBuilder providing canonical if_* features.
- `src/investment/schema_registry.py` — 94 specs / 9 categories.

### Validation
- `src/validation/oof_health.py` — OOFHealthValidator.

### Prior Phase Context
- `.planning/phases/39-marketawarewincalibrator/39-CONTEXT.md` — Phase 39 decisions on calibrator architecture, shadow mode pattern.
- `.planning/phases/40-race-level-ranker/40-CONTEXT.md` — Phase 40 decisions on ranker architecture, shadow diagnostics, D-02 LambdaRank deferral.

### Requirements
- `.planning/REQUIREMENTS.md` — SHD-01 through SHD-03 (Phase 41 requirements).
- `.planning/ROADMAP.md` — Phase 41 success criteria (3 items: fixed-fold comparison, comprehensive metrics, selection agreement).
- `.planning/PROJECT.md` — Key Decisions table (selection agreement = diagnostic not gate, deployment condition = probability quality).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **BacktestEngine** (`src/backtest/engine.py`): Constructor accepts TrainedModelsV5 injection. Two instances with different model sets can run the same test period independently. Returns BacktestResult with bet_history containing race_id, umaban, odds, result columns.
- **ModelLoader.load_from_dir()** (`src/db/model_loader.py` line 594): Loads TrainedModelsV5 from arbitrary directories. Handles per-surface optional models including market_aware_win_calibrator and win_race_level_ranker.
- **BacktestReportGenerator** (`src/backtest/report.py`): Jinja2 + HTML template pattern. ShadowComparisonReportGenerator should follow same pattern with dedicated template.
- **RacePredictor shadow diagnostics** (lines 860-884): Already computes baseline_selected_umaban, ranker_selected_umaban, baseline_ranker_agreement per race. Phase 41 extends this to cross-model-set comparison.
- **drift_diagnostics.py** (`src/models/drift_diagnostics.py`): KS-test / Wasserstein for probability distribution comparison. Reusable for baseline vs shadow probability quality metrics.

### Established Patterns
- **Shadow mode via is_tried / deployment_status**: MAWC and ranker set deployment_status = "shadow_only" when not promoted. RacePredictor checks before using.
- **Pre-trained model loading from directories**: run_backtest.py --skip-train loads from data/models-backtest/{year}/. Same pattern for shadow comparison.
- **Jinja2 HTML reports**: BacktestReportGenerator generates self-contained HTML. MultiYearReportGenerator aggregates. Follow for shadow comparison report.
- **Parquet as source-of-truth**: All diagnostic data stored as Parquet. CSV copies for quick inspection. JSON for aggregate metrics.
- **Manifest pattern**: ParameterFreezeProtocol uses JSON manifest with SHA256. shadow_manifest.json follows same reproducibility pattern.
- **SubmodelSet optional fields**: New models added as Optional[Type] = None. Backward compatible. None = not trained / not loaded.

### Integration Points
- **RacePredictor.predict()**: Add enable_market_aware_calibrator / enable_race_level_ranker flags. When false, skip MAWC/ranker even if loaded.
- **RacePredictor.get_win_candidates()**: Existing shadow diagnostics block records agreement. Extend for cross-model-set comparison.
- **BacktestEngine.run()**: No changes needed. Comparison framework runs it twice with different TrainedModelsV5.
- **run_backtest.py**: No changes. Shadow comparison is a separate script.

</code_context>

<specifics>
## Specific Ideas

- Fixed fold definitions: Fold 2024 (train 2020-2023, test 2024), Fold 2025 (train 2021-2024, test 2025). Explicit in CLI, matching WF validation.
- Baseline = current pipeline with enable_market_aware_calibrator=false + enable_race_level_ranker=false. No deleted code dependency.
- N-way framework design: ShadowComparisonFramework accepts variant_name/model_dir pairs. Phase 41 uses ["baseline", "ridge_shadow"]. Future: ["baseline", "ridge_shadow", "lambdarank_shadow"].
- Five output artifacts: JSON metrics, Parquet race diff, CSV race diff, Parquet horse diff, HTML report + manifest.
- CLV formula: closing_odds / betting_odds - 1 (decimal odds). Diagnostic only, null if inputs missing.
- Per-race diff table columns: race_id, baseline_selected_umaban, shadow_selected_umaban, selected_changed, baseline/shadow odds/p/EV/score/result/return/stake, score_component_deltas.

</specifics>

<deferred>
## Deferred Ideas

- **LightGBM LambdaRank shadow variant:** Deferred to v2.2+. Phase 41 framework supports N-way comparison for future addition. No skeleton code in Phase 41.
- **Training orchestration inside comparison framework:** Baseline/shadow training before comparison is out of scope. Separate orchestration script may handle this in future milestones.
- **Deployment gate automation (DEP-01):** Already deferred in REQUIREMENTS.md to v2.2+.

</deferred>

---

*Phase: 41-Shadow Comparison Framework*
*Context gathered: 2026-05-28
