# Project Research Summary

**Project:** keiba-ai v1.4 -- Ensemble Filter Recalibration
**Domain:** Betting filter recalibration for 3-model GBM stacking ensemble (horse racing ML prediction)
**Researched:** 2026-05-05
**Confidence:** HIGH

## Executive Summary

keiba-ai v1.4 is a targeted rewiring effort on a mature ML betting system. The 3-model stacked ensemble (LightGBM + XGBoost + CatBoost, Ridge meta-learner) was integrated in v1.1, but the downstream betting filters (WinSelectionGate, EV_lower threshold, OddsBandFilter) were calibrated against single-LightGBM OOF predictions during Phase 11-12. The resulting probability distribution mismatch causes the EV_lower filter to exclude 3,594 candidates, producing only 7 bets/year at 0% ROI instead of the target 100+ bets/year at >100% ROI. The root cause is not missing algorithms or components -- it is stale calibration data flowing into existing filter training methods.

The recommended approach is a strict dependency-ordered recalibration: (1) retrain WinSelectionGate on ensemble OOF predictions so its quantile bins match the new distribution, (2) make the EV_lower threshold dynamic rather than hardcoded at 1.0, (3) rebuild OddsBandFilter calibration with ensemble-derived training bet history, and (4) execute the already-built Optuna 14-dim parameter search against the recalibrated pipeline. All four research files converge on the same conclusion: zero new dependencies or components are needed. The work is rewiring data flow, not adding capability.

The primary risk is Optuna overfitting to a 2-fold walk-forward with 14 free parameters on a target dominated by a few longshot outcomes. Increasing fold count to 4+ and adding parameter stability checks across random seeds is essential. A secondary risk is look-ahead bias in the OddsBandFilter training pipeline, where the same Optuna-tuned strategy parameters are used to generate training bet history. The mitigations are well-understood and documented below.

## Key Findings

### Recommended Stack

Zero new production dependencies. Every v1.4 task maps to already-installed packages. The stack is scipy (ks_2samp, wasserstein_distance for distribution shift diagnostics), scikit-learn (calibration_curve, IsotonicRegression for probability validation), optuna 4.8 (TPESampler + MedianPruner for 14-dim search), betacal 1.0 (Beta calibration for ensemble probabilities), and numpy (quantile computation for dynamic thresholds). All verified installed and compatible.

**Core technologies:**
- **scipy.stats (1.17.1):** ks_2samp + wasserstein_distance for quantifying distribution shift between single-model and ensemble OOF predictions -- already available as sklearn transitive dependency
- **optuna (4.8.0):** TPE sampler for 14-dim parameter search -- already installed, StrategyOptimizer already defines full search space
- **scikit-learn (1.8.0):** calibration_curve, brier_score_loss for ensemble probability validation -- already integrated in win_benter_gate.py
- **betacal (1.0):** Beta calibration (3-param) for ensemble probability outputs -- already integrated, works identically for ensemble inputs
- **numpy (2.4.3):** np.quantile for computing distribution-adaptive EV_lower thresholds from ensemble OOF -- standard operation

### Expected Features

**Must have (table stakes):**
- **WinSelectionGate retrained on ensemble OOF** (TS-01) -- Gate score tables (combo_scores, pair_scores) use quantile edges computed from training distribution. Ensemble shifts these edges. Retraining is feeding the existing train() method with ensemble-derived DataFrame.
- **EV_lower threshold recalibrated to ensemble distribution** (TS-02) -- RobustConfidenceEstimator conformal intervals are calibrated on single-model residuals. Retraining with ensemble residuals + making the threshold dynamic (not hardcoded 1.0) fixes the 3,594-exclusion problem.
- **OddsBandFilter recalibrated with ensemble training_bet_history** (TS-03) -- Band ROI depends on model accuracy per band. Ensemble accuracy profile differs. Filter already has calibrate() that accepts arbitrary bet history -- just needs ensemble-generated history.
- **Optuna 14-dim parameter optimization executed** (TS-04) -- All parameters are at hardcoded defaults, never tuned. StrategyOptimizer exists but has never been run. Must execute after TS-01/02/03 complete.

**Should have (competitive):**
- **Dynamic EV_lower threshold by regime** (D-01) -- Vary threshold by market regime: lower in AGGRESSIVE, higher in CONSERVATIVE. Natural extension of existing regime architecture.
- **Ensemble-aware conformal confidence scoring** (D-02) -- Weight confidence intervals by base model agreement. Already have pairwise OOF correlations from _check_diversity().

**Defer (post-v1.4):**
- Walk-forward filter recalibration within backtest (D-04) -- too architecturally complex for this milestone.
- Quantile-adaptive EV threshold per probability bin (D-03) -- would require gate architecture changes.

### Architecture Approach

The architecture change is purely data-flow rewiring within the existing backtest pipeline. No new components needed. The three filters (WinSelectionGate, EV_lower, OddsBandFilter) are all model-agnostic: they consume DataFrame columns (probabilities, edges, odds) without knowing which model produced them. The change is ensuring the DataFrames passed to their training/calibration methods contain ensemble-derived values. The Optuna parameter search wires everything together through the existing StrategyOptimizer -> BacktestEngine -> RacePredictor pipeline, which already loads ensemble models via use_ensemble_override=True.

**Major components:**
1. **WinSelectionGateModel** (models/win_selection_gate.py) -- Retrain with ensemble OOF predictions; quantile bin edges and score tables adapt to new distribution automatically.
2. **RacePredictor.get_win_candidates()** (backtest/race_predictor.py) -- Replace hardcoded EV_lower_win_corrected >= 1.0 with dynamic threshold from gate model or Optuna parameter.
3. **OddsBandFilter** (betting/odds_band_filter.py) -- No code change; automatically recalibrated when StrategyOptimizer runs with ensemble models.
4. **StrategyOptimizer** (tuning/strategy_optimizer.py) -- Execute existing optimization; optionally add ev_lower_threshold as 15th dimension.

### Critical Pitfalls

1. **WinSelectionGate quantile bin mismatch** -- The gate stores prob_edges/edge_edges/odds_edges from single-model OOF distribution. Ensemble probabilities have different quantiles, so candidates land in wrong bins and get garbage scores. Prevention: always retrain gate when switching models; never load single-model gate for ensemble inference.

2. **EV_lower threshold distribution shift** -- Conformal prediction bands computed from single-model residuals are miscalibrated for ensemble. Combined with hardcoded >= 1.0 threshold, this excludes 3,594 candidates. Prevention: retrain RobustConfidenceEstimator with ensemble residuals; make threshold dynamic from ensemble OOF winner distribution.

3. **OddsBandFilter look-ahead bias** -- StrategyOptimizer generates training_bet_history using the same Optuna-tuned parameters being optimized for test performance, leaking test information into filter calibration. Prevention: generate training bets with default (non-optimized) strategy parameters.

4. **Optuna overfitting to 2-fold backtest** -- 14 dimensions with only 2 folds on a stochastic target (few longshot winners dominate ROI) allows fitting noise. Prevention: increase to 4+ folds with expanding windows; add parameter stability checks across top trials and random seeds.

5. **OOF/inference distribution mismatch** -- Stacked ensemble trains Ridge on fold-model OOF predictions (67-80% data) but inference uses full-data models (100%), producing systematically different probabilities. Prevention: compare OOF vs inference prediction statistics; add isotonic calibration if mean shift exceeds 0.02.

## Implications for Roadmap

Based on combined research, the recommended phase structure follows the strict dependency chain identified across all four research files: gate retraining must precede threshold changes, which must precede OddsBandFilter rebuild, which must precede Optuna execution.

### Phase 1: WinSelectionGate Ensemble Retraining
**Rationale:** The gate is the first filter in the candidate selection chain. Its quantile bins and score tables are distribution-specific. Everything downstream depends on the gate producing valid scores for ensemble predictions. This is the root cause of the 7-bets/year problem.
**Delivers:** Gate model with ensemble-adapted thresholds, valid scores for ensemble candidates, restored candidate throughput.
**Addresses:** TS-01 from FEATURES.md. Fixes Pitfall 1 (quantile bin mismatch).
**Avoids:** Loading stale single-model gate for ensemble inference (anti-pattern 1 from ARCHITECTURE.md).
**Estimated scope:** ~20 lines pipeline data routing. No new code in gate model itself.

### Phase 2: Dynamic EV_lower Threshold
**Rationale:** After the gate is retrained, the EV_lower filter is the second exclusion mechanism. The hardcoded >= 1.0 threshold is calibrated for single-model EV_lower distribution and over-excludes ensemble candidates. Must be dynamic.
**Delivers:** Adaptive EV_lower threshold computed from ensemble OOF winner distribution. Expected to restore 100+ bets/year from 7.
**Addresses:** TS-02 from FEATURES.md. Fixes Pitfall 2 (distribution shift) and Pitfall 7 (EVCorrectionModel init_score dependency).
**Uses:** numpy quantile computation, WinSelectionGateModel for threshold storage (Option A from ARCHITECTURE.md) or Optuna search dimension (Option B).
**Implements:** Dynamic threshold in get_win_candidates(), replacing hardcoded >= 1.0.
**Estimated scope:** ~30 lines calibration data routing + ~10 lines filter mask modification.

### Phase 3: OddsBandFilter Ensemble Recalibration
**Rationale:** Band ROI statistics depend on model accuracy in each odds range. Ensemble accuracy profile differs from single model. The filter must be calibrated with ensemble-era bet history. The StrategyOptimizer already generates ensemble-derived training_bet_history, but it must use default params (not Optuna-tuned) to avoid look-ahead bias.
**Delivers:** OddsBandFilter calibrated on ensemble performance per band. Band exclusions reflect actual ensemble ROI, not single-model ROI.
**Addresses:** TS-03 from FEATURES.md. Fixes Pitfall 3 (look-ahead bias in training_bet_history).
**Uses:** Existing OddsBandFilter.calibrate() method, BacktestEngine training-phase backtest.
**Estimated scope:** ~10 lines to ensure default params for training bet generation. No changes to OddsBandFilter itself.

### Phase 4: Optuna 14-dim Parameter Optimization
**Rationale:** All filters must be ensemble-calibrated before parameter tuning begins. Tuning on miscalibrated filters embeds structural deficiencies into optimal parameters. The StrategyOptimizer is fully implemented but has never been executed.
**Delivers:** Optimal strategy parameters for production use. Quantified ROI improvement vs defaults.
**Addresses:** TS-04 from FEATURES.md. Must manage Pitfall 4 (overfitting) with 4+ folds and stability checks.
**Uses:** optuna TPE sampler, existing StrategyOptimizer, run_strategy_optimization.py.
**Estimated scope:** ~0 code changes (execution only). Optionally ~5 lines to add ev_lower_threshold as 15th dimension.

### Phase 5: Manifest Freeze and OOS Validation
**Rationale:** After Optuna finds parameters, freeze them to prevent drift during out-of-sample evaluation. The ParameterFreezeProtocol already exists. OOS validation confirms generalization.
**Delivers:** Frozen parameter manifest (JSON + SHA256). OOS ROI measurement on truly unseen data.
**Uses:** Existing save_strategy_manifest() and verify_strategy_manifest().
**Estimated scope:** ~0 code changes. Script execution and result analysis.

### Phase Ordering Rationale

- **Gate before EV_lower:** The gate candidate ranking affects which candidates reach the EV_lower filter. If gate scores are invalid (stale bins), EV_lower receives garbage input. Architecture analysis identifies this as anti-pattern 1.
- **EV_lower before OddsBandFilter:** EV_lower is the dominant exclusion mechanism (3,594 exclusions). Fixing it first restores candidate volume, giving OddsBandFilter meaningful data to calibrate on.
- **All filters before Optuna:** Tuning parameters on miscalibrated filters gives false optima. The pitfalls research emphasizes this as the highest-risk sequencing error.
- **Optuna before manifest freeze:** Parameters must be optimized before they are frozen. The freeze-then-validate pattern from Phase 13 prevents post-hoc parameter tampering.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (EV_lower dynamic threshold):** The exact percentile/range for the dynamic threshold is data-dependent. Research recommends 25th percentile of positive-edge ensemble OOF winners as a starting point, but optimal value requires empirical testing on actual ensemble predictions.
- **Phase 4 (Optuna optimization):** The search space boundaries, fold count, trial count, and stability check thresholds need careful design. The current 2-fold setup is insufficient. Research recommends 4+ folds but the exact training window expansion strategy needs specification during planning.
- **Phase 5 (OOS validation):** The choice of held-out validation year(s) and the acceptable ROI degradation threshold need definition. If 2022 is held out, the training data shrinks and may affect model quality.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Gate retraining):** The gate train() method is already model-agnostic. Feeding it ensemble OOF data is a data routing change with well-documented column requirements.
- **Phase 3 (OddsBandFilter rebuild):** The filter calibrate() already accepts arbitrary bet history. The change is ensuring default strategy params for training bet generation.
- **Phase 5 (Manifest freeze):** The ParameterFreezeProtocol is fully implemented and tested from Phase 13.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new dependencies. All required packages verified installed at compatible versions. Every v1.4 task mapped to existing tool. |
| Features | HIGH | Full codebase audit of 8+ key files (3000+ LOC total). Every feature mapped to specific component with identified gap. Feature dependencies well-understood. Anti-features explicitly scoped out. |
| Architecture | HIGH | Data flow traced through actual backtest path with line references. Component boundaries verified. Three anti-patterns identified with specific code locations. Build order derived from dependency analysis. |
| Pitfalls | HIGH | 11 pitfalls identified from direct code analysis with line numbers. Each has prevention strategy, detection method, and recovery steps. Phase-specific warnings provided. |

**Overall confidence:** HIGH

All research was grounded in direct codebase analysis with line-level references across all four domains. No findings depend on assumptions. The main uncertainty is empirical: the actual ROI impact of each recalibration step requires running the pipeline, which is the purpose of the implementation phases.

### Gaps to Address

- **Dynamic EV_lower threshold value:** Research recommends distribution-adaptive threshold (e.g., 25th percentile of ensemble OOF winner EV_lower), but the exact percentile needs empirical tuning. The Optuna search (Phase 4) can optimize this, but Phase 2 needs a reasonable starting value.
- **Optuna fold count and training window:** Research recommends 4+ folds over the current 2, but expanding training windows for earlier folds (2020-2021, 2020-2022) may produce models with insufficient data. The exact fold structure needs specification during Phase 4 planning.
- **OOF vs inference distribution shift magnitude:** The stacking distribution shift (fold-model vs full-data model predictions) is theoretically expected but its magnitude in this specific ensemble is unknown. Phase 1 should measure this before deciding whether additional calibration is needed.
- **RegimeDetector behavior under ensemble:** The regime classifier was trained on single-model error patterns. Ensemble errors may shift regime distributions, affecting COLLAPSED skip rates. Verify after pipeline retraining.

## Sources

### Primary (HIGH confidence)
- Full codebase audit: src/models/win_selection_gate.py (1113 lines), src/models/stacked_ensemble.py (607 lines), src/betting/odds_band_filter.py (112 lines), src/tuning/strategy_optimizer.py (273 lines), src/backtest/race_predictor.py (~925 lines), src/backtest/engine.py (~1207 lines), src/models/ev_correction_model.py (~575 lines), src/models/regime_detector.py (~264 lines)
- Package verification: scipy 1.17.1, scikit-learn 1.8.0, optuna 4.8.0, betacal 1.0, numpy 2.4.3 -- all verified installed via python -c execution
- Pipeline verification: use_ensemble_override=True in ModelLoader, ensemble OOF generation in training_pipeline.py, StrategyOptimizer parameter search space

### Secondary (MEDIUM confidence)
- ScienceDirect: ML for sports betting -- calibration over accuracy -- confirms calibration-focused approach yields higher betting profits
- arXiv: Systematic review of ML in sports betting -- surveys ML techniques in sports betting contexts
- ResearchGate: ML for betting -- accuracy vs calibration -- optimizing for calibration leads to greater returns
- Wolpert (1992), Ting & Witten (1999) -- stacking distribution shift theory
- Benter (1994), Thorp (2006) -- foundational Kelly criterion and betting strategy literature

### Tertiary (LOW confidence)
- Exact ROI impact of each recalibration step on JRA data -- requires running backtests
- Optimal Optuna trial count for 14-dim search on horse racing ROI -- theoretical guidance only
- Ensemble agreement signal effectiveness for confidence scoring -- untested in this codebase

---
*Research completed: 2026-05-05*
*Ready for roadmap: yes*
