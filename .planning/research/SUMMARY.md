# Project Research Summary

**Project:** keiba-ai v1.2 -- Win Backtest Validation & Pipeline Optimization
**Domain:** Horse racing ML prediction system -- backtest engine conversion from place-mode to win-mode + training/backtest pipeline performance optimization
**Researched:** 2026-05-04
**Confidence:** HIGH

## Executive Summary

The keiba-ai v1.2 milestone is a targeted conversion of an existing backtest pipeline from place (fukushou) betting validation to win (tanshou) betting validation. The system already has fully trained win models -- WinTwoStageModel, WinSelectionGateModel, WinBenterGate -- that produce correct win EV columns during inference. The gap is entirely in the backtest orchestration layer: the engine hardcodes place-specific payout maps, place-only candidate selection, place odds for settlement, and place-oriented diagnostics. Converting to win-mode requires surgical modifications across four layers (ETL, payout settlement, candidate selection, reporting) but does NOT require any changes to the ML models themselves or the training pipeline.

The recommended approach is a three-phase build: (1) Win-mode core -- fix the payout data pipeline, add win candidate selection and win bet generation, wire them into BacktestEngine with a `betting_target` dispatch parameter; (2) Win reporting -- update diagnostics, bet history fields, and report sections for win-specific metrics; (3) Pipeline optimization -- vectorize iterrows() hotspots, add feature caching, and batch operations to cut the ~57-minute backtest runtime. Phase 1 is the critical path; phases 2 and 3 can proceed independently once phase 1 produces correct win ROI numbers.

The key risk is settlement correctness. The current payout map and final odds map both use place-specific columns (`payfukusyo*`, `fukuoddslow`). If these are not replaced with win equivalents (`paytansyou*`, `tanodds`/`tanoddslow`), win bets will settle at place odds, producing wildly incorrect ROI. Mitigation is straightforward: build parallel win-specific maps and dispatch by `bet.bet_type` at settlement time. A secondary risk is edge threshold miscalibration: place-calibrated regime parameters will generate too many or too few win bets. This should be addressed empirically after the first correct win backtest run.

## Key Findings

### Recommended Stack

No new production dependencies are needed. The entire win-mode conversion and pipeline optimization uses the existing stack (pandas, numpy, LightGBM, pyarrow). One dev-only addition is recommended: **pyinstrument** (>=4.6) as a statistical profiler for the ~57-minute backtest runs, replacing heavier tools like cProfile. The optimization techniques (vectorized payout maps, groupby dict lookups, batch diagnostic logging) are all native pandas/numpy operations. Polars, Cython, numba, ray, and dask were all evaluated and rejected: the gains come from eliminating iterrows() and batching DataFrame operations, not from switching data frameworks.

**Core technologies:**
- **pandas >=2.2 / numpy >=1.26:** Vectorized payout maps, groupby dict lookups, batch operations -- replaces 18+ iterrows() call sites
- **LightGBM >=4.3:** Batch inference via `booster.predict(df)` -- already supports batch, calling code just needs restructuring
- **pyarrow >=14.0:** Parquet-backed feature caching for expensive pre-computation (HorseHistory, Jockey, Trainer features)
- **pyinstrument >=4.6 (dev-only):** Statistical profiler at ~1-5% overhead for identifying bottlenecks in the 57-min backtest run

### Expected Features

**Must have (table stakes):**
- **Win payout settlement** (`build_win_payout_map()` reading `paytansyoumaban1`/`paytansyopay1`) -- without this, ROI is wrong
- **Win final odds map** (using `tanodds`/`tanoddslow` instead of `fukuoddslow`) -- settlement fallback must use win odds
- **Win candidate selection** (`get_win_candidates()` using `win_selection_ev`, `win_selection_edge`, `win_gate_pass`) -- currently only place candidates exist
- **Win bet generation** (`BetType.WIN` bets with `tanoddslow` odds) -- currently only `BetType.PLACE` is generated
- **Betting target dispatch** (`betting_target` parameter on BacktestEngine and RacePredictor) -- single flag controls all behavior

**Should have (competitive):**
- **Feature cache for backtest inference** -- save pre-computed feat_df as Parquet, cut backtest time by ~60%
- **Vectorized payout map construction** -- replace iterrows() with melt/zip, 10-50x speedup on map building
- **Parallel feature pre-computation** -- ThreadPoolExecutor for independent feature modules
- **Win ROI by odds band analysis** -- identify which odds ranges have actual edge
- **Win-specific diagnostics and reporting** -- ev_win, win_selection_ev, win_gate_score in logs and reports

**Defer (future milestones):**
- Batch race prediction (requires major RacePredictor refactor, ~600 lines of gate logic)
- Win vs place comparison mode (needs both paths working first)
- Confidence interval coverage validation for win EV intervals
- Streaming backtest results (current memory usage is manageable)

### Architecture Approach

The conversion follows a parallel-method pattern: each place-specific operation gets a win-specific counterpart, and a `betting_target` dispatch parameter switches between them. This avoids duplicating logic -- `get_win_candidates()` mirrors `get_place_candidates()` using a column-name mapping (`place_selection_ev` -> `win_selection_ev`, `fukuoddslow` -> `tanoddslow`). The shared gating, pruning, and runner-up logic from WinSelectionGateModel is already implemented and only needs to be called from the backtest path. For optimization, the key architectural change is replacing per-race DataFrame filters with pre-built groupby dictionaries, converting O(n_races * n_rows) filtering to O(n_rows) groupby + O(1) dict lookup.

**Major components:**
1. **BacktestEngine** (`src/backtest/engine.py`) -- Add `betting_target` param, `build_win_payout_map()`, win final odds map, win-mode dispatch in run() loop (~80 lines modified)
2. **RacePredictor** (`src/backtest/race_predictor.py`) -- Add `get_win_candidates()`, win-mode `select_bets()` path, thin `get_candidates()` dispatcher (~120 lines new)
3. **EveryDB2Queries** (`src/db/everydb2_queries.py`) -- Add `paytansyoumaban1`/`paytansyopay1` to `get_payouts()` SQL SELECT
4. **DiagnosticLogger** (`src/backtest/diagnostic_logger.py`) -- Add win-specific diagnostic fields (p_win_pred, ev_win, win_gate_score)
5. **Scripts** (`run_backtest.py`, `run_wf_validation.py`) -- Add `--betting-target` CLI flag, default to win mode

### Critical Pitfalls

1. **Payout map uses place payouts for win settlement** (Pitfall 1) -- `build_payout_map()` reads `payfukusyo*` columns; win bets would settle against place odds. Create `build_win_payout_map()` reading `paytansyoumaban1`/`paytansyopay1`.

2. **Final odds map uses place odds for win** (Pitfall 2) -- `final_odds_map` built from `fukuoddslow`; win settlement fallback uses place odds. Build separate map from `tanodds` and dispatch by bet type.

3. **Bet generation hardcodes BetType.PLACE** (Pitfall 3) -- `select_bets()` only generates place bets. Create `get_win_candidates()` and win bet generation path using `WinSelectionGateModel` outputs.

4. **WF validation tests wrong model** (Pitfall 5) -- Walk-forward overfitting detection compares place ROI, not win ROI. Propagate `betting_target` to WF validation script.

5. **Diagnostics log place-only columns** (Pitfall 4) -- Post-backtest analysis shows place metrics when win metrics are needed. Add parallel win diagnostic logging path.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Win-Mode Backtest Core
**Rationale:** The settlement and bet generation layer must be fixed before any win validation results are meaningful. This is the critical path -- everything else depends on correct win ROI numbers. The four critical pitfalls (1-3, 5) all converge here.
**Delivers:** Correct win-mode backtest producing reliable win ROI, hit rate, and bankroll curves.
**Addresses:** Table-stakes features A (win settlement), B (win bet generation), and WF validation mode switching.
**Avoids:** Pitfalls 1, 2, 3, 5, 6 -- settlement correctness, odds correctness, bet type correctness, WF mode, fallback path.
**Estimated scope:** ~200 lines new/modified across engine.py, race_predictor.py, everydb2_queries.py, run_backtest.py, run_wf_validation.py.

### Phase 2: Win Reporting and Analysis
**Rationale:** Once win ROI is correct, the reporting layer needs to show win-specific metrics for analysis and threshold calibration. This is additive work that does not change backtest outcomes.
**Delivers:** Win-specific diagnostics, bet history fields, report sections, odds-band ROI breakdown.
**Addresses:** Table-stakes features C (win reporting) + differentiators (win ROI by odds band).
**Avoids:** Pitfalls 4 (diagnostics), 7 (baseline), 8 (thresholds), 14 (baseline display), 15 (bet history), 16 (JSON mode tag).
**Uses:** Existing BacktestReportGenerator with win-mode conditional sections.

### Phase 3: Pipeline Performance Optimization
**Rationale:** Optimization must come after validation correctness is confirmed. Mixing optimization with the win-mode conversion risks conflating two types of changes. Each optimization is independent and can be applied incrementally.
**Delivers:** Reduced backtest time from ~57 min/year to ~35-40 min/year. Reduced training iteration time with feature caching.
**Addresses:** Differentiators -- vectorized payout maps, feature caching, parallel pre-computation, optional diagnostics.
**Avoids:** Pitfalls 9 (per-race loop), 10 (iterrows payout), 11 (no feature cache), 12 (Optuna overhead), 13 (diagnostic overhead).
**Uses:** pandas vectorization, Parquet-backed caching, ThreadPoolExecutor, pyinstrument profiling.

### Phase Ordering Rationale

- **Phase 1 before Phase 2:** Cannot analyze win metrics until win settlement is correct. The dependency is strict.
- **Phase 2 before Phase 3:** Optimization changes must not be mixed with correctness changes. If backtest results change after optimization, you need a correct baseline to compare against.
- **Phase 3 is internally order-independent:** Each optimization (vectorize payout maps, groupby dicts, feature caching, optional diagnostics) is standalone and can be done in any order.
- **Win reporting (Phase 2) can overlap with Phase 3 start:** Once Phase 1 produces correct ROI, reporting and optimization can proceed in parallel.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (win threshold calibration):** The win edge distribution is unknown. The first correct win backtest run will reveal whether place-calibrated edge thresholds are reasonable. If win hit rate is very different from expected, threshold tuning needs empirical analysis.
- **Phase 3 (feature caching invalidation):** The caching strategy needs a concrete invalidation mechanism. Research identified "manual deletion" as the simplest approach, but the exact cache key design (date range hash vs. feature version) needs validation during implementation.

Phases with standard patterns (skip research-phase):
- **Phase 1:** All changes are directly observable in the codebase. The column mappings, payout data availability, and WinSelectionGateModel API are fully documented. No external research needed.
- **Phase 3 (vectorization):** Standard pandas optimization patterns well-documented across all research sources.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new production deps. All optimization uses existing pandas/numpy/LightGBM. Only dev-only pyinstrument added. |
| Features | HIGH | Full codebase audit of engine.py (950 LOC), race_predictor.py (788 LOC), training_pipeline.py (1290 LOC). All place-specific hardcodings directly observable. Win model infrastructure verified as complete. |
| Architecture | HIGH | Parallel-method pattern is straightforward. Column mapping between place and win paths is documented. WinSelectionGateModel API confirmed with score(), annotate_race_context(), soft_pass_mask(). |
| Pitfalls | HIGH | 18 pitfalls identified and cross-validated against source code. Critical pitfalls (1-6) are directly observable mismatches. Moderate pitfalls (7-13) need empirical validation post-implementation. |

**Overall confidence:** HIGH

### Gaps to Address

- **Win edge distribution unknown:** Place edge thresholds are calibrated for place betting (hit rate ~21%). Win hit rate (~7%) and edge distribution will be different. First win backtest run should include edge distribution analysis to inform threshold tuning. Address during Phase 2 by plotting `win_selection_edge` histogram.
- **Pipeline timing breakdown estimated, not measured:** The ~57 min/year timing is from CLAUDE.md, but the breakdown into feature computation (26%), pre-computation (14%), and race loop (60%) is estimated from code structure, not profiling. Address at start of Phase 3 by running pyinstrument on a single-year backtest.
- **Win payout data completeness unknown:** The ETL type rules include `paytansyoumaban1`/`paytansyopay1`, but the SQL query does not currently SELECT them. Re-running ETL is needed, and there may be gaps in historical win payout data for older years. Address at start of Phase 1 by checking payouts Parquet schema.

## Sources

### Primary (HIGH confidence)
- Full codebase audit: `src/backtest/engine.py` (950 LOC), `src/backtest/race_predictor.py` (788 LOC), `src/pipelines/training_pipeline.py` (1290 LOC), `scripts/run_backtest.py` (549 LOC), `scripts/run_wf_validation.py` (325 LOC)
- Win model implementation: `src/models/win_selection_gate.py` (1113 LOC), `src/models/win_benter_gate.py`
- ETL schema: `src/db/etl.py` (type rules include win payout columns), `src/db/everydb2_queries.py` (SQL missing win columns)
- Domain types: `src/domain/models.py` (SubmodelSet, Bet, BetType enum with WIN)

### Secondary (MEDIUM confidence)
- pyinstrument GitHub -- statistical profiler for Python, ~1-5% overhead
- Pandas performance hierarchy -- vectorization > itertuples > apply > iterrows (well-documented)
- LightGBM batch prediction documentation -- booster.predict() supports batch inference
- Pipeline timing estimates from CLAUDE.md -- ETL ~10min, training ~44min, backtest ~57min/year

### Tertiary (LOW confidence)
- Speedup estimates for vectorization (10-50x claimed for iterrows replacement) -- needs validation with actual data volumes
- Optuna overhead fraction (2-3x training time from PROJECT.md) -- needs timing data from v1.1 runs
- Win payout data completeness for historical years -- needs ETL verification

---
*Research completed: 2026-05-04*
*Ready for roadmap: yes*
