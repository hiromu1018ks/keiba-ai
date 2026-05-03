# Feature Landscape: v1.2 Win Backtest Validation + Pipeline Optimization

**Domain:** Horse racing prediction -- fixing backtest from place-mode to win-mode + training/backtest pipeline performance optimization
**Researched:** 2026-05-04
**Confidence:** HIGH (full codebase audit of engine.py, race_predictor.py, training_pipeline.py, run_backtest.py, run_wf_validation.py)

## Context

This document covers ONLY features for the v1.2 milestone. The system already has working win models (WinTwoStageModel, WinSelectionGate, WinBenterGate) that train correctly. The problem is that the **backtest and WF validation scripts still operate in place-mode** -- they settle bets using place payout logic, generate place bets via `get_place_candidates()`, and report place-oriented diagnostics. The goal is to switch these to win-mode while also speeding up the training + backtest pipeline.

Three capability areas:

1. Win-mode backtest (settlement, bet generation, reporting)
2. Pipeline performance optimization (caching, vectorization, parallelism)
3. Win-mode result analysis and reporting

---

## Table Stakes

Features/capabilities that any win-focused backtest MUST have. Missing these means the backtest results are wrong or meaningless for win betting.

### A. Win-Mode Settlement

| Feature | Why Expected | Complexity | Current Status | Change Required |
|---------|--------------|------------|----------------|-----------------|
| **Win payout settlement** (`kakuteijyuni == 1` for payoff) | Win bets pay ONLY when the horse finishes 1st. The current `_settle_bet()` already handles `BetType.WIN` with `finish_pos == 1`, but the backtest engine never GENERATES win bets. The settlement logic is correct but unused. | Low | `_settle_bet()` in engine.py line 946-948 correctly checks `finish_pos == 1` for WIN bets | No change needed to settlement itself. The issue is upstream (bet generation). |
| **Win odds for settlement** (tanoddslow, not fukuoddslow) | Win bets settle at win odds (tanoddslow), not place odds (fukuoddslow). The current `final_odds_map` uses `fukuoddslow` (engine.py line 280-281). For win-mode backtest, must build a tanoddslow-based map. | Low | `final_odds_map` built from `fukuoddslow` column only | Change `final_odds_map` to use `tanoddslow` when in win mode. Alternatively, build a separate `win_odds_map` alongside the existing place map. |
| **Win payout from payouts table** | The payouts table (`paytansyopay`) contains actual win dividends (100-yen units). Similar to `build_payout_map()` for place, need `build_win_payout_map()` that reads `paytansyoumaban` and `paytansyopay`. | Low | `build_payout_map()` reads `payfukusyoumaban/payfukusyopay` (place). No win equivalent exists. | Add `build_win_payout_map()` reading `paytansyoumaban1` and `paytansyopay1` columns from payouts DataFrame. |

### B. Win-Mode Bet Generation

| Feature | Why Expected | Complexity | Current Status | Change Required |
|---------|--------------|------------|----------------|-----------------|
| **Win candidate selection** via `get_win_candidates()` | The backtest engine calls `get_place_candidates()` (race_predictor.py line 504) and never considers win bets. For win-mode backtest, must call win-specific candidate selection using `win_selection_ev`, `win_selection_edge`, `win_selection_prob`, and `win_gate_pass`. | Medium | `RacePredictor.get_place_candidates()` uses place-specific columns (`place_selection_ev`, `place_selection_edge`, `fukuoddslow`, `place_gate_pass`). Win analogues exist in `WinSelectionGateModel` but no `get_win_candidates()` method. | Add `get_win_candidates()` to RacePredictor that mirrors `get_place_candidates()` but uses win columns: `win_selection_ev`, `win_selection_edge`, `win_selection_prob`, `tanoddslow`, `win_gate_pass`. |
| **Win bet creation** (BetType.WIN with tanoddslow) | Currently `select_bets()` creates only `BetType.PLACE` bets (race_predictor.py line 629). For win-mode, must create `BetType.WIN` bets using tanoddslow. | Low | `select_bets()` hardcodes `BetType.PLACE` and `fukuoddslow` | Add win-mode path that creates `Bet(bet_type=BetType.WIN, odds=tanoddslow, ...)`. The existing `WinStrategy` class is unused by the backtest. |
| **Win regime threshold adaptation** | RegimeDetector thresholds (`edge_threshold`, `ev_threshold`, `max_bets_per_race`) are tuned for place betting. Win betting has lower hit rates (~7% vs ~21%) but higher payouts. Edge thresholds may need to be tighter for win. | Low | `RegimeDetector.get_strategy_params()` returns same params for all bet types | Add win-specific regime parameter overrides or a separate config section. The simplest approach: when in win-mode, use higher `ev_threshold` (e.g., 1.15 instead of 1.10) and `max_bets_per_race=1`. |

### C. Win-Mode Reporting

| Feature | Why Expected | Complexity | Current Status | Change Required |
|---------|--------------|------------|----------------|-----------------|
| **Win-specific bet_history fields** | Current `bet_history` entries use `p_place_pred`, `e_return_place_pred`, `ev_place` etc. Win-mode needs `p_win_pred`, `e_return_win_pred`, `ev_win`, `win_selection_ev`. | Low | bet_history dict keys in engine.py lines 700-752 are place-oriented | Add win-specific keys to bet_history. Keep place keys for backward compatibility or replace entirely. |
| **Win-specific diagnostics** | `DiagnosticLogger.log_horse()` logs `p_place_pred`, `ev_place`, `place_gate_score`. Win-mode needs `p_win_pred`, `ev_win`, `win_gate_score`. | Low | DiagnosticLogger has place-specific fields | Add win-mode diagnostic fields or make diagnostic logging mode-aware. |
| **Win-specific report sections** | HTML reports show "win_rate" (actually place-win-rate), popularity bands, EV bands. Win-mode reports should show win-specific metrics: win-hit-rate, average win odds, ROI by odds band. | Medium | `BacktestReportGenerator` is bet-type agnostic (works on generic bet_history) | Enhance report to show win-specific sections when bet_type=WIN. Add win-hit-rate, average winning odds, ROI by odds range. |

---

## Differentiators

Features that go beyond minimum win-mode backtest. These improve the quality of validation and the speed of iteration.

### HIGH IMPACT -- Pipeline Performance

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Feature cache for backtest inference** | The backtest engine recomputes ALL features (HorseHistoryFeatures, JockeyContextFeatures, TrainerContextFeatures, SireFeatures, PaceAptitudeFeatures, CourseFeatures) every run (~5-10 min per year). These features depend only on historical data that doesn't change between runs. Caching them would reduce backtest time from ~57 min to ~15 min per year. | Medium | ParquetStore, FeatureEngine | Cache `feat_df` after `build_all()` + all pre-computed features as a Parquet file keyed by test period. On subsequent runs, load from cache if the feature cache exists and is newer than the model. |
| **Vectorize payout map construction** | `build_payout_map()` iterates rows with a Python for loop over payouts DataFrame (engine.py lines 112-125). For 5 years of data, this processes ~50,000 rows. Vectorizing with pandas operations would be 10-50x faster. | Low | None | Replace iterrows() with vectorized melt + division. Same for `build_wide_payout_map()`. |
| **Batch race prediction** | The backtest engine processes races one-by-one in a for loop (engine.py line 420-786). Each iteration creates a DataFrame slice, merges features, and calls predict(). Batch prediction would process all races at once using vectorized operations, reducing Python loop overhead. | High | RacePredictor, SubmodelSet | The current per-race loop is needed because `predict()` has race-level operations (surface selection, feature merge). Batch prediction would require restructuring predict() to handle multi-race DataFrames. This is high complexity but high reward. |
| **Parallel feature pre-computation** | HorseHistoryFeatures, JockeyContextFeatures, TrainerContextFeatures, and JockeyTrainerComboFeatures are computed sequentially (engine.py lines 332-351). These are independent and could run in parallel using ThreadPoolExecutor. | Low | ThreadPoolExecutor (already used in training pipeline) | The training pipeline already uses ThreadPoolExecutor for surface parallelism. Apply the same pattern to feature pre-computation in backtest engine. |
| **Optuna warm-start / reduced search space** | Optuna HP optimization (if `--ensemble` is used) explores a large search space. Warm-starting from previous best parameters or reducing the search space for backtest (not final training) would cut tuning time. | Medium | OptunaTuner | Add `--fast-tune` flag that reduces n_trials from 100 to 20 and narrows search bounds around previous best. |
| **Pre-compute market model predictions** | MarketModel predictions are computed twice: once during training (with OOF) and once during backtest (full predict). If the backtest uses the same training period, the training predictions can be cached and reused. | Low | MarketModel, TrainingPipelineV5 | Save market model OOF predictions alongside the model. In backtest, load and use them instead of recomputing. |

### MEDIUM IMPACT -- Win-Mode Analysis

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Win ROI by odds band analysis** | Win betting ROI is heavily dependent on the odds range. A strategy that is profitable at 3-6x odds may be unprofitable at 10-20x odds. Breaking down ROI by odds band identifies where the edge actually exists. | Low | bet_history with tanoddslow | Add odds-band breakdown to report: [<3, 3-6, 6-10, 10-15, 15-20, 20+]. |
| **Win calibration plot in WF validation** | The WF validation script logs feature importance stability (Spearman rho) but not calibration. Adding a reliability diagram for win probabilities (predicted vs actual win rate by decile) detects over/under-confidence. | Medium | matplotlib or text-based output | The training pipeline already computes reliability data (`generate_reliability_data()`). Expose this in WF validation output. |
| **Win vs place comparison mode** | Running both win and place backtests side-by-side for the same period shows which bet type has better ROI. This helps decide whether to focus resources on win or place optimization. | Low | Both win and place backtest paths working | Add `--compare-mode` flag that runs both and outputs comparison table. |
| **Confidence interval coverage validation** | The conformal EV intervals (80%/90%) should be validated: do 80% of actual outcomes fall within the 80% interval? If coverage is too low, intervals are overconfident. If too high, underconfident. | Medium | RobustConfidenceEstimator | Compute coverage rate for win EV intervals in WF validation. Target: 80% coverage at alpha=0.2. |

### LOW IMPACT -- Nice to Have

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Win-specific before/after comparison** | The single-year display has a hardcoded `before_roi = 0.638` (run_backtest.py line 233). For win-mode, this should be the pre-v1.2 win ROI baseline. | Low | None | Make before_roi configurable or compute from a reference run. |
| **Streaming backtest results** | Instead of accumulating all bet_history in memory, stream results to JSON/Parquet incrementally. Reduces memory for multi-year runs. | Medium | File I/O | Use append-mode JSONL or incremental Parquet writes. Low priority since current memory usage is manageable. |
| **Multi-year WF validation** | Current WF validation has 2 fixed folds (2024, 2025 test). Making fold definitions configurable allows testing more windows. | Low | FOLDS constant in run_wf_validation.py | Make FOLDS a CLI argument with default to current 2-fold definition. |

---

## Anti-Features

Features to explicitly NOT build for this milestone.

### Not In Scope

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Changes to the win model itself** (WinTwoStageModel, WinSelectionGate training) | The models train correctly already. The issue is the backtest/validation pipeline not using them for settlement. Changing model architecture risks introducing bugs without addressing the actual problem. | Keep all model training code unchanged. Only change the backtest/validation scripts to use win-mode paths. |
| **Real-time odds integration for backtest** | The backtest uses pre-race odds from time-series data. Adding real-time odds streaming is a production feature, not a backtest feature. | Continue using `extract_pre_post_odds()` with 5-min-before snapshots. |
| **GPU acceleration** | LightGBM/XGBoost/CatBoost already use multi-threaded CPU training. GPU support requires different builds (lightgbm-gpu, xgboost-gpu) and may not be faster for datasets under 1M rows. | Keep CPU-based training. Optimize data pipeline (I/O and feature engineering) instead, which is where most time is spent. |
| **Rewriting the entire backtest engine** | The engine is complex (950 lines) but well-structured. Rewriting it risks introducing subtle bugs in settlement logic, DD tracking, and regime detection. | Make targeted modifications: add win-mode paths alongside existing place paths. Use a mode flag (`betting_target="win"|"place"`) to switch behavior. |
| **Database schema changes** | The Parquet schema is stable and used by ETL, training, and backtest. Changing it would require re-running ETL. | Read existing Parquet files as-is. Add win-specific columns to bet_history output only. |
| **New ML models for win prediction** | The system already has 3-model stacking, Benter combination, calibration, and conformal intervals. Adding more models increases training time without addressing the backtest validation gap. | Focus on validating the existing win model through correct backtesting. Model improvements are for later milestones. |
| **Complex stake sizing optimization** | Kelly criterion and drawdown control are already implemented. Optimizing further before validating basic win ROI is premature. | Use flat 100-yen bets for initial win backtest validation. Switch to Kelly only after confirming positive flat-bet ROI. |
| **Wide bet mode in backtest** | Wide (quinella place) is explicitly out of scope per PROJECT.md. | Skip wide settlement in win-mode backtest. The existing wide code can remain dormant. |

---

## Feature Dependencies

```
Win-Mode Backtest
  requires: WinSelectionGateModel (DONE -- trained in pipeline)
  requires: build_win_selection_ev() (DONE -- in win_selection_gate.py)
  requires: ensure_win_selection_columns() (DONE -- in win_selection_gate.py)
  requires: WinBenterGate (DONE -- trained in pipeline)
  requires: win_selection_ev, win_selection_edge, win_selection_prob columns (DONE -- produced by predict())
  requires: tanoddslow column in feat_df (DONE -- from odds snapshots)
  NEW: get_win_candidates() in RacePredictor
  NEW: win bet generation in select_bets() or new select_win_bets()
  NEW: build_win_payout_map() for actual dividend settlement
  NEW: win final_odds_map using tanoddslow
  NEW: win-mode regime params (higher ev_threshold, max_bets=1)
  depends-on: All of above must work together for correct win backtest

Pipeline Performance
  independent: Each optimization is independent and can be done in any order
  feature_cache: depends on stable feat_df schema (currently stable)
  vectorize_payout: no dependencies, pure refactoring
  parallel_features: depends on thread-safety of feature modules (verify needed)
  batch_prediction: depends on RacePredictor refactor (HIGH complexity, defer)

Win Reporting
  depends-on: Win-mode backtest working (above)
  win_roi_by_odds_band: depends on bet_history with tanoddslow
  win_calibration_plot: depends on WF validation producing win probabilities
  compare_mode: depends on both win and place paths working
```

## MVP Recommendation

### Phase 1: Win-Mode Backtest Core (Must Do First)

These are the minimum changes to get a correct win-mode backtest running.

1. **Add `get_win_candidates()` to RacePredictor** -- mirrors `get_place_candidates()` using win columns. Uses `win_selection_ev`, `win_selection_edge`, `win_selection_prob`, `tanoddslow`, and `win_gate_pass`/`win_gate_score` from `WinSelectionGateModel`.

2. **Add `select_win_bets()` to RacePredictor** -- creates `BetType.WIN` bets using tanoddslow. Uses `get_win_candidates()` for selection, applies max_bets_per_race=1, flat 100-yen stake.

3. **Build `win_final_odds_map` and `win_payout_map`** -- change `final_odds_map` construction in engine.py to use tanoddslow. Add `build_win_payout_map()` reading paytansyoumaban/paytansyopay from payouts.

4. **Add `betting_target` parameter to BacktestEngine** -- when `betting_target="win"`, call `get_win_candidates()` and `select_win_bets()` instead of place equivalents. Update diagnostic logging to use win columns.

5. **Update `run_backtest.py` to pass `--betting-target win`** -- add CLI flag, default to "place" for backward compatibility.

6. **Update `run_wf_validation.py` similarly** -- add `betting_target="win"` to BacktestEngine instantiation.

Complexity: Medium. Changes are localized to engine.py and race_predictor.py. Estimated effort: 2-3 sessions.

### Phase 2: Win Reporting and Analysis (After Phase 1)

7. **Update bet_history keys for win mode** -- add `p_win_pred`, `ev_win`, `win_selection_ev`, `win_selection_edge`, `tanoddslow` to bet_history entries when in win mode.

8. **Update DiagnosticLogger for win mode** -- add win-specific diagnostic fields.

9. **Add win ROI by odds band to report** -- enhance `BacktestReportGenerator` to show ROI breakdown by win odds range.

10. **Update before/after comparison baseline** -- change `before_roi` to configurable or compute from reference.

Complexity: Low. All reporting changes are additive. Estimated effort: 1 session.

### Phase 3: Performance Optimization (After Validation Works)

11. **Vectorize payout map construction** -- replace iterrows() with pandas melt/merge. Quick win, ~10-50x speedup on map construction.

12. **Feature cache for backtest inference** -- save pre-computed feat_df as Parquet after first backtest run. Load from cache on subsequent runs. Cuts backtest time by ~60%.

13. **Parallel feature pre-computation** -- run HorseHistoryFeatures, JockeyContextFeatures, TrainerContextFeatures in parallel using ThreadPoolExecutor.

14. **Reduced Optuna search for fast iteration** -- add `--fast-tune` flag with fewer trials.

Complexity: Low-Medium. Each optimization is independent. Estimated effort: 2 sessions.

### Defer (Future Milestones)

- Batch race prediction (requires major RacePredictor refactor)
- Win calibration plot in WF validation (nice but not blocking)
- Win vs place comparison mode
- Confidence interval coverage validation
- Streaming backtest results
- Configurable WF fold definitions

---

## Key Design Decisions for This Milestone

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| How to switch between win and place mode? | `betting_target` parameter on BacktestEngine and RacePredictor | Single flag controls all behavior. Default "place" preserves backward compatibility. |
| Should win and place run simultaneously? | No -- separate runs | Win and place have different edge distributions, different optimal thresholds. Running separately gives cleaner results. |
| Should the win backtest use WinStrategy class? | No -- integrate into RacePredictor | WinStrategy is a standalone class not connected to the backtest pipeline. Better to add win paths to the existing RacePredictor flow that already handles regime detection, quality screening, and diagnostic logging. |
| Payout settlement: actual dividends vs odds-based? | Use actual dividends (paytansyopay) when available, fall back to tanoddslow | Actual dividends are the true payout. tanoddslow may not exactly match the final dividend in some edge cases. |
| Feature cache invalidation? | Manual -- delete cache file to force recomputation | The feature cache is a development optimization, not a production feature. Complex cache invalidation logic is over-engineering. |

---

## Code Change Map

| File | Changes Required | LOC Affected (est.) |
|------|-----------------|---------------------|
| `src/backtest/engine.py` | Add `betting_target` param, build win payout map, win final odds map, win-mode diagnostic logging, win bet_history keys | ~80 lines |
| `src/backtest/race_predictor.py` | Add `get_win_candidates()`, `select_win_bets()`, win-mode `should_bet()` adaptation | ~120 lines |
| `scripts/run_backtest.py` | Add `--betting-target` CLI flag, pass to BacktestEngine | ~15 lines |
| `scripts/run_wf_validation.py` | Pass `betting_target="win"` to BacktestEngine | ~5 lines |
| `src/backtest/report.py` | Add win-specific report sections | ~40 lines |
| (new) feature caching module | Optional: cache feat_df as Parquet | ~30 lines |

Total estimated new/modified code: ~290 lines.

---

## Sources

- Full codebase audit: engine.py (950 LOC), race_predictor.py (788 LOC), training_pipeline.py (1290 LOC), run_backtest.py (549 LOC), run_wf_validation.py (325 LOC)
- LightGBM Parameters documentation (Context7): num_threads, feature_fraction, bin_construct_sample_cnt
- Existing WinSelectionGateModel (1113 LOC) -- already implements all win selection logic
- Existing WinStrategy (77 LOC) -- standalone class, not integrated into backtest
- PROJECT.md v1.2 milestone definition
