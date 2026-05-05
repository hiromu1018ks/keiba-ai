# Technology Stack: Ensemble Filter Recalibration (v1.4)

**Project:** keiba-ai v1.4 Ensemble Filter Recalibration
**Researched:** 2026-05-05
**Scope:** Recalibrating WinSelectionGate for ensemble OOF predictions, dynamic EV_lower threshold, OddsBandFilter with ensemble bet_history, Optuna 14-dim parameter optimization
**Supersedes:** v1.3 STACK.md (betting strategy optimization -- all still current, zero new dependencies needed for v1.4)

## Verdict: No New Dependencies Required

The existing stack already contains every capability needed for ensemble-aware filter recalibration. The work is entirely about **rewiring existing code** -- feeding ensemble OOF predictions into the filter training path, making thresholds derive from distribution statistics rather than hardcoded values, and running the already-built Optuna 14-dim optimizer against ensemble outputs.

## Current Installed Stack

| Package | Installed Version | pyproject.toml Minimum | Status |
|---------|-------------------|----------------------|--------|
| Python | 3.11 | >=3.11 | Pinned via mise |
| LightGBM | 4.6.0 | >=4.3 | Up to date |
| XGBoost | 3.2.0 | >=2.0 | Up to date |
| CatBoost | 1.2.10 | >=1.2 | Up to date |
| scikit-learn | 1.8.0 | >=1.4 | Up to date |
| scipy | 1.17.1 | (transitive via sklearn) | Available, not in pyproject.toml |
| pandas | 2.3.3 | >=2.2 | Up to date |
| numpy | 2.4.3 | >=1.26 | Up to date |
| pyarrow | (installed) | >=14.0 | Up to date |
| mlflow | 3.10.1 | >=2.12 | Up to date |
| optuna | 4.8.0 | >=3.5 | Up to date |
| betacal | 1.0 | >=1.0 | Available |
| joblib | (installed, transitive) | (transitive via sklearn) | Used for model persistence |

## Recommended Stack for v1.4

### Production Dependencies: Zero New Additions

| Technology | Version | Role in v1.4 | Why Sufficient |
|------------|---------|-------------|----------------|
| scipy.stats | 1.17.1 | `ks_2samp`, `wasserstein_distance` for distribution drift detection between single-model and ensemble OOF | Kolmogorov-Smirnov test quantifies whether ensemble OOF distribution differs from single-model OOF distribution. Wasserstein distance measures magnitude of shift. Both already available as transitive dependency of scikit-learn. |
| scikit-learn | 1.8.0 | `calibration_curve`, `brier_score_loss`, `IsotonicRegression` for ensemble probability calibration validation | Already used in `win_benter_gate.py` for Beta vs Isotonic calibration comparison. Same code path works identically for ensemble probabilities. |
| optuna | 4.8.0 | `TPESampler` + `MedianPruner` for 14-dim parameter search | `StrategyOptimizer` already defines the full 14-dim search space (6 regime + 5 DD control + 2 EV scaling + 1 OddsBandFilter). The optimizer loads models from `--models-dir` -- pointing it at ensemble models is a config change, not a code change. |
| betacal | 1.0 | Beta calibration (3-param) for ensemble probability outputs | Already integrated in `compare_calibrations()` in `win_benter_gate.py`. Works identically whether fed single-model or ensemble probabilities. Ensemble `predict()` clips to [0,1], so input constraints are met. |
| numpy | 2.4.3 | `np.percentile`, `np.quantile` for computing distribution-adaptive thresholds from ensemble OOF predictions | The core operation for dynamic thresholds. Already available. |
| pandas | 2.3.3 | DataFrame manipulation in all filter/recalibration code | All existing filter code (`WinSelectionGateModel`, `OddsBandFilter`, `get_win_candidates`) operates on DataFrames. No change needed. |
| joblib | (bundled) | Model serialization for `WinSelectionGateModel.save/load` | Already used. No change needed. |

### NOT Needed (Explicitly Rejected)

| Library | Purpose | Why Rejected |
|---------|---------|-------------|
| `torch` / `tensorflow` | Learned adaptive thresholds | Massive dependency for a problem that pandas quantile binning solves better. `WinSelectionGateModel` already uses quantile-based binning with smoothed scoring -- this is the right approach for ~50K rows of training data. |
| `river` / `creme` | Online/streaming adaptive thresholds | The system is batch-mode (walk-forward validation on historical data). No streaming inference needed for recalibration. |
| `bayesian-optimization` | Alternative optimizer to Optuna | Optuna is already installed and `StrategyOptimizer` is already built with the exact 14-dim search space. Replacing it adds risk with zero benefit. |
| `polars` | Faster DataFrame operations | All filter code is pandas-based. Rewriting for Polars is out of scope and unnecessary -- the bottleneck is model inference, not DataFrame manipulation. |
| `matplotlib` / `plotly` | Calibration visualization | Not needed for the recalibration logic itself -- this is a backend system. Implicitly available through sklearn/betacal if diagnostic plots are needed. |
| `scipy.stats.entropy` (KL divergence) | Distribution shift measurement | `ks_2samp` is superior for continuous probability distributions. KL divergence requires density estimation (kernel density or histogram binning) which adds noise for small sample sizes. KS test is nonparametric, robust, and already available. |
| `scipy.optimize` | Constrained threshold optimization | Optuna TPE already handles the 14-dim parameter space. Adding a second optimizer creates confusion. |
| `emcee` / `dynesty` | Bayesian posterior sampling for threshold uncertainty | Overkill. The walk-forward validation in `WinSelectionGateModel` already provides robust threshold estimates. |
| `statsmodels` | Statistical testing infrastructure | `scipy.stats` provides everything needed (KS test, Wasserstein distance). statsmodels adds no value for this use case. |

## How Existing Tools Map to v1.4 Tasks

### Task 1: WinSelectionGate Recalibration with Ensemble OOF

**Tools used:** `numpy`, `pandas`, `scipy.stats.ks_2samp`, `scipy.stats.wasserstein_distance`, existing `WinSelectionGateModel`

The `WinSelectionGateModel.train()` method (1113 lines in `win_selection_gate.py`) already implements the full pipeline:
- Walk-forward fold generation (`_build_walk_forward_folds`)
- Quantile-based binning of prob/edge/odds (`_quantile_edges`, `_bucketize`)
- Smoothed ROI scoring with Bayesian prior (`_smoothed_score`)
- Grid search for optimal threshold combination (`_build_threshold_grid`)
- OOF score computation across folds (`_build_oof_scores`)
- Second-horse reranker (`_fit_add_second_reranker`)

**What changes:** The training DataFrame passed to `WinSelectionGateModel.train()` must contain ensemble OOF predictions (`p_win_final` from `StackedEnsemble.predict()`) instead of single LightGBM predictions. The distribution comparison (`ks_2samp`) is used to quantify the shift and verify the new quantile edges are appropriate for the ensemble distribution.

**Distribution comparison approach:**
```python
from scipy.stats import ks_2samp, wasserstein_distance

# Compare single-model vs ensemble OOF probability distributions
stat, pvalue = ks_2samp(single_model_oof_probs, ensemble_oof_probs)
shift_magnitude = wasserstein_distance(single_model_oof_probs, ensemble_oof_probs)
```

This tells us whether the ensemble distribution is materially different. If the KS test rejects (p < 0.05), the quantile edges computed from the ensemble distribution will differ from the single-model edges, and the gate needs full retraining. If not, only threshold tuning is needed.

### Task 2: Dynamic EV_lower Threshold

**Tools used:** `numpy` (quantile computation), `pandas` (rolling/groupby), existing `get_win_candidates()` in `race_predictor.py`

The current `EV_lower_win_corrected >= 1.0` filter in `race_predictor.py` (line 440) is hardcoded. Making it dynamic means computing the threshold from the ensemble OOF distribution characteristics.

**Approach:** Use `np.quantile` to compute an ensemble-aware threshold that targets a specific selection rate (e.g., top 30% of candidates pass). The threshold is computed during `WinSelectionGateModel.train()` and stored alongside the existing threshold parameters.

```python
# Compute dynamic EV_lower threshold from ensemble OOF distribution
# Target: keep enough candidates for 100+ bets/year
ev_lower_values = oof_df["EV_lower_win_corrected"].dropna()
target_percentile = 0.30  # Keep top 30% by EV_lower
dynamic_ev_threshold = float(np.quantile(ev_lower_values, target_percentile))
```

This is pure numpy/pandas -- no new libraries.

### Task 3: OddsBandFilter Rebuild with Ensemble training_bet_history

**Tools used:** Existing `OddsBandFilter.calibrate()`, `BacktestEngine.run()`

The `OddsBandFilter.calibrate()` method (line 38 of `odds_band_filter.py`) already accepts a `bet_history: list[dict[str, Any]]`. The `StrategyOptimizer._run_single_backtest()` already generates `training_bet_history` from a training-phase backtest run (lines 151-169 of `strategy_optimizer.py`).

**What changes:** The training-phase backtest must be run with ensemble models loaded. This is controlled by `use_ensemble_override=True` in `ModelLoader.load_from_dir()` (already present on line 137 of `strategy_optimizer.py`). The `roi_threshold` parameter is already in the Optuna search space (line 79).

**No new tools needed.** The pipeline is:
1. Load ensemble models (already supported via `use_ensemble_override=True`)
2. Run training-phase backtest (already implemented in `StrategyOptimizer`)
3. Feed `training_bet_history` to `OddsBandFilter.calibrate()` (already wired in `BacktestEngine`)
4. Run test-phase backtest with calibrated filter (already wired)

### Task 4: Optuna 14-dim Parameter Optimization

**Tools used:** `optuna` 4.8.0 (already installed), existing `StrategyOptimizer`

The 14-dimensional search space is already fully defined in `StrategyOptimizer._suggest_params()`:
- 6 regime parameters: `fk_aggressive`, `ev_aggressive`, `edge_aggressive`, `fk_conservative`, `ev_conservative`, `edge_conservative`
- 5 DD control parameters: `dd_threshold_1`, `dd_threshold_2`, `multiplier_reduced`, `rolling_window`, `min_stay_races`
- 2 EV scaling parameters: `target_ev`, `max_scale`
- 1 OddsBandFilter parameter: `roi_threshold`

**What changes:** Run the optimizer against ensemble models by pointing `--models-dir` at the ensemble model directory. The `MedianPruner` and `TPESampler` are already configured. The walk-forward 2-fold evaluation (2024 test, 2025 test) is already set up.

## Integration Points Summary

| Integration Point | File | Change Type | New Dependency? |
|-------------------|------|-------------|-----------------|
| WinSelectionGate retraining | `models/win_selection_gate.py` | Feed ensemble OOF df to `.train()` | No |
| EV_lower dynamic threshold | `backtest/race_predictor.py` (line 440) | Read dynamic threshold from gate model instead of hardcoded `1.0` | No |
| Distribution comparison | New utility or inline in pipeline | `scipy.stats.ks_2samp` + `wasserstein_distance` | No (transitive dep) |
| OddsBandFilter calibration | `betting/odds_band_filter.py` | No code change -- already accepts `bet_history` | No |
| Training-phase backtest | `tuning/strategy_optimizer.py` | Already loads ensemble models via `use_ensemble_override=True` | No |
| Optuna optimization | `tuning/strategy_optimizer.py` | Already supports 14-dim search with TPE + MedianPruner | No |
| Model loading | `db/model_loader.py` | `use_ensemble_override=True` already parameterized | No |

## Key Design Decision: Why Not Add Libraries

The v1.4 milestone is fundamentally a **calibration and rewiring** task. The filters exist, the optimizer exists, the ensemble model exists. The gap is that:
1. The gate was trained on single-model OOF predictions, not ensemble OOF
2. The EV_lower threshold is hardcoded at 1.0, not distribution-aware
3. The OddsBandFilter has never been calibrated against ensemble bet history
4. The Optuna optimizer has never been run (default parameters only)

All four gaps are addressed by **changing data flow** (what gets fed into existing code), not by adding new tools. Adding dependencies would increase surface area without addressing the actual problem.

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| scipy distribution tools sufficient | HIGH | Verified installed (1.17.1), `ks_2samp` and `wasserstein_distance` confirmed available via `python -c` test. Standard nonparametric tests, well-documented, stable API. |
| sklearn calibration tools sufficient | HIGH | Already integrated in `win_benter_gate.py`. Same code path works for ensemble outputs. `calibration_curve` and `brier_score_loss` are stable sklearn APIs. |
| Optuna 4.8 handles 14-dim search | HIGH | `StrategyOptimizer` already defines the full search space. TPESampler handles 14 dimensions routinely -- the Optuna docs show it used with 20+ dimensions. MedianPruner already wired. Verified via Context7 docs. |
| No new dependencies needed | HIGH | Every v1.4 task maps to existing installed tools. The work is rewiring, not adding. Verified by reading 8 source files totaling ~3000 lines. |
| betacal works with ensemble probs | MEDIUM | Already used for single-model calibration. Ensemble probabilities should be in [0,1] range (clipped by `StackedEnsemble.predict()` line 127). Should work, but ensemble meta-learner (Ridge) could produce more extreme probabilities that need additional clipping. Low risk. |
| Dynamic threshold via numpy quantile | HIGH | `np.quantile` is a basic numpy function. The approach (compute percentile of OOF distribution) is standard practice. |

## Sources

- scipy 1.17.1 installed, `ks_2samp` and `wasserstein_distance` verified via `python -c` execution (HIGH confidence)
- scikit-learn 1.8.0 installed, `calibration_curve`, `brier_score_loss`, `IsotonicRegression` verified (HIGH confidence)
- optuna 4.8.0 installed, `TPESampler`, `MedianPruner` verified via Context7 docs (HIGH confidence)
- betacal 1.0 installed and integrated in `src/models/win_benter_gate.py` lines 276-283 (HIGH confidence)
- Context7 docs: Optuna TPE + MedianPruner examples from `optuna.readthedocs.io`
- Context7 docs: scipy `ks_2samp` documentation from `scipy/scipy` repository
- Context7 docs: sklearn `CalibratedClassifierCV`, `calibration_curve`, `brier_score_loss` from `scikit-learn.org`
- Existing code analysis: `WinSelectionGateModel` (1113 lines), `OddsBandFilter` (112 lines), `StrategyOptimizer` (273 lines), `StackedEnsemble` (607 lines), `BacktestEngine` (relevant sections ~200 lines), `RacePredictor.get_win_candidates()` (~80 lines)
