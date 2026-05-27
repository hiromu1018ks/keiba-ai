# Technology Stack

**Project:** keiba-ai v2.1 MarketAware Calibration + Race-Level Ranker
**Researched:** 2026-05-27
**Supersedes:** v2.0 STACK.md

## Verdict: Zero New External Dependencies

All 4 v2.1 features (MarketAwareWinCalibrator, segment conditioning, Race-Level Ranker, shadow comparison) are implementable entirely with the currently installed stack. No new pip packages required.

**Rationale:** The project already has every building block:
- Benter-style logit blending via `BenterCombination` (MLE via `scipy.optimize.minimize`)
- Per-segment calibration via `WinSegmentCalibrator` (OOF-based Bayesian shrinkage)
- Race-level ranking via pandas `groupby.rank()` (deterministic, no ML ranker needed)
- Shadow comparison via `scipy.stats` (KS-test, Mann-Whitney) + `sklearn.metrics` (Brier, log-loss, ECE)
- Isotonic regression via `sklearn.isotonic.IsotonicRegression`
- Feature engineering via `InvestmentFeatureFrame` (94 specs / 9 categories, dual-mode builder)

## Current Installed Stack

| Package | Installed Version | Role in v2.1 |
|---------|-------------------|-------------|
| Python | 3.11 | Pinned via mise |
| LightGBM | 4.6.0 | Existing 3-model stacking (unchanged); no LGBMRanker needed for v2.1 |
| XGBoost | 3.2.0 | Existing stacking component (unchanged) |
| CatBoost | 1.2.10 | Existing stacking component (unchanged) |
| scikit-learn | 1.8.0 | IsotonicRegression, LogisticRegression, brier_score_loss, log_loss, calibration_curve, TimeSeriesSplit |
| scipy | 1.17.1 | scipy.optimize.minimize (Benter MLE), scipy.stats (ks_2samp, mannwhitneyu, spearmanr), scipy.special (logit, expit) |
| pandas | 2.3.3 | InvestmentFeatureFrame, groupby.rank(), pd.cut for segments |
| numpy | 2.4.3 | logit/expit, array operations for calibration |
| betacal | 1.1.0 | Beta calibration as alternative post-processing (already used in WinBenterGate) |
| joblib | (bundled) | Model serialization for calibrator artifacts |
| optuna | 4.8.0 | Existing 16-dim strategy optimization (unchanged) |
| mlflow | 3.10.1 | Log calibrator metrics as MLflow artifacts |

## Recommended Stack for v2.1

### MarketAwareWinCalibrator

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `scipy.optimize.minimize` | 1.17.1 | MLE fitting of alpha/beta/gamma Benter blend weights | Already used in `BenterCombination.fit()`. L-BFGS-B with bounds is proven stable for this use case. No reason to change. |
| `scipy.special.logit`, `expit` | 1.17.1 | Numerically stable logit/sigmoid transforms | Used in `benter_combination.py`. Avoid manual log(p/(1-p)) which needs explicit clipping. |
| `sklearn.isotonic.IsotonicRegression` | 1.8.0 | Post-blend monotonic calibration | Already used in `EVCorrectionModel`. Ensures calibrated output is monotonic in input, critical for probability ordering. |
| Existing `BenterCombination` | (codebase) | Base class to extend or wrap | Proven MLE fitting with bounded optimization. Extension adds segment-conditioned weights rather than global alpha/beta. |
| Existing `WinBenterGate` | (codebase) | Pipeline wrapper for market prob extraction + race normalization | Already handles tanodds->implied_prob, race-level sum normalization, and edge calculation. New calibrator slots into this pipeline. |

**Approach:** Extend `BenterCombination` to accept segment-conditioned alpha/beta rather than global constants. The segment keys come from `popularity_rank`, `if_odds_band_id`, and `if_p_win_race_rank` (already in `InvestmentFeatureFrame`). Each segment gets its own (alpha, beta, gamma) triplet fitted via MLE on OOF data. This is the Benter (1994) approach generalized to condition on market structure.

### Segment Conditioning as Calibrator Features

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `pandas.cut` / `pd.qcut` | 2.3.3 | Bin popularity_rank, odds, prob_rank into discrete segments | Already used in `WinSegmentCalibrator.ODDS_BINS` / `RANK_BINS`. Proven pattern. |
| `InvestmentFeatureFrame` | (codebase) | Source of `if_popularity_rank`, `if_odds`, `if_p_win_race_rank`, `if_edge_rank_in_race` | 94 specs / 9 categories already built. No new feature computation needed. |
| `numpy.ndarray` groupby operations | 2.4.3 | Compute per-segment Benter weights from OOF residuals | Vectorized operations on segment-grouped data. No new library needed. |

**Approach:** Segments are defined by crossing `(popularity_rank_band, odds_band, prob_rank_band)` -- 3-4 bands per dimension yields 27-64 segments, each with enough OOF samples (min 120 rows, matching `WinSegmentCalibrator.min_segment_rows`). The segment key is computed at calibration time from InvestmentFeatureFrame outputs, not stored as a new feature column. This avoids adding features to the ML model itself -- the calibrator is a post-hoc adjustment layer.

### Race-Level Ranker

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `pandas.DataFrame.groupby.rank()` | 2.3.3 | Deterministic within-race ranking by composite score | Already used in `InvestmentFeatureFrame._compute_derived()` for `if_p_win_race_rank`, `if_ev_race_rank`, `if_edge_rank_in_race`. No ML ranker needed. |
| `numpy` weighted sum | 2.4.3 | Composite score = w1*p_win + w2*edge + w3*ev_corrected | Deterministic weighting avoids the complexity of learning-to-rank. With 3-5 inputs and ~15 horses per race, a simple weighted score is sufficient. |
| `scipy.stats.spearmanr` | 1.17.1 | Rank correlation for ranker evaluation (IC metric) | Already used in `ic_evaluator.py` and `stacked_ensemble.py`. Measures rank stability of the ranker across segments. |

**Approach:** The Race-Level Ranker is NOT a learned model. It is a deterministic scoring function that combines InvestmentFeatureFrame outputs into a single race-internal rank. The composite score formula is:

```
score = w_p * if_p_win_final + w_edge * if_edge_win + w_ev * if_ev_corrected
```

Weights `w_p`, `w_edge`, `w_ev` are tuned by Optuna (extending the existing 16-dim parameter space to 19-dim). This avoids training an LGBMRanker, which would require a separate model lifecycle, OOF generation, and health checks for marginal gain.

**Why NOT LGBMRanker:** The previous STACK.md recommended `lightgbm.LGBMRanker` with `objective='lambdarank'`. After deeper analysis of the codebase, this is over-engineered. The ranker has at most 5-10 input features from InvestmentFeatureFrame. A GBM ranker on 5 features with ~15 items per group would overfit. A deterministic weighted sum with Optuna-tuned weights achieves the same goal with zero new model complexity.

### Baseline vs Shadow Comparison Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `scipy.stats.ks_2samp` | 1.17.1 | Distribution comparison: baseline vs shadow probabilities | Kolmogorov-Smirnov test for detecting probability distribution shifts. Standard non-parametric test. |
| `scipy.stats.mannwhitneyu` | 1.17.1 | Rank comparison: baseline vs shadow ROI/HR | Non-parametric test for comparing central tendencies when distributions may not be normal. |
| `sklearn.metrics.brier_score_loss` | 1.8.0 | Probability quality: Brier Score comparison | Standard proper scoring rule. Already used in `win_benter_gate.py`. |
| `sklearn.metrics.log_loss` | 1.8.0 | Probability quality: Log Loss comparison | Strictly proper scoring rule, more sensitive to extreme predictions. |
| `compute_ece` (codebase) | (codebase) | Expected Calibration Error comparison | Already implemented in `win_benter_gate.py`. Per-bin accuracy vs confidence. |
| `spearmanr` (codebase) | 1.17.1 | Rank correlation: horse selection stability | Already used in IC evaluation. Measures whether shadow and baseline select the same horses. |
| `pandas.DataFrame` | 2.3.3 | Alignment, filtering, aggregation of baseline/shadow results | Standard operations. No new patterns needed. |

**Approach:** The comparison framework runs the backtest pipeline twice (baseline config vs shadow config with new calibrator/ranker) on the same 2024/2025 test period. Comparison metrics are computed by joining results on `(race_id, umaban)` and computing paired differences. The deployment gate checks: Brier improvement, log-loss improvement, selection overlap >= 85%, ROI not degraded > 5%, bet count within 90-110% of baseline.

## What NOT to Add

| Library/Approach | Why Rejected | Use Instead |
|-----------------|-------------|-------------|
| `lightgbm.LGBMRanker` | Over-engineered for 5-10 input features. GBM ranker on ~15 items per group overfits. Adds model lifecycle complexity (OOF, health checks, serialization) for marginal gain. | Deterministic weighted sum with Optuna-tuned weights |
| `torch` / `tensorflow` | No deep learning in v2.1. Tabular data with ~100K rows, ~100 features. LightGBM/XGB/CatBoost stack is already proven. | Existing 3-model stacking (unchanged) |
| `polars` | Pandas is the project standard. 83K LOC uses pandas throughout. Dual-dataframe ecosystem adds import confusion. | pandas (existing) |
| `statsmodels` | Regression diagnostics not needed. scipy.stats has all needed hypothesis tests. sklearn has calibration metrics. | scipy.stats + sklearn.metrics (existing) |
| `cvxpy` for portfolio optimization | Overkill for max-3-horse allocation within a race. Kelly criterion already implemented. | Existing StakeCalculator + Kelly criterion |
| `evidently` / `nannyml` for drift detection | Custom drift diagnostics already implemented in `drift_diagnostics.py`. Adding a SaaS-oriented monitoring tool violates "no external services" constraint. | Existing `drift_diagnostics.py` + custom shadow comparison |
| `wandb` | MLflow already integrated for experiment tracking. Dual experiment trackers add confusion. | MLflow (existing) |
| New calibration libraries (e.g., `netcal`) | scikit-learn IsotonicRegression + betacal + custom Benter MLE already covers all calibration approaches. | IsotonicRegression + betacal + BenterCombination (existing) |
| Learned meta-learner for ranker | PROJECT.md explicitly scopes out: "sklearn StackingClassifier -- native boosting API and PIT-safe fold incompatible" and "complex meta-learner (GBM/NN) -- 3 features, Ridge is optimal." Same reasoning applies to ranker. | Deterministic weighted sum |

## Integration Points with Existing Stack

### 1. MarketAwareWinCalibrator <-> BenterCombination

The new calibrator wraps `BenterCombination` rather than replacing it. Integration path:
- `MarketAwareWinCalibrator` holds a dict of `BenterCombination` instances keyed by segment
- `train()` fits one BenterCombination per segment on OOF data
- `apply()` looks up the segment from InvestmentFeatureFrame columns, applies the segment-specific blend
- Serialized via `joblib` (same pattern as `WinSegmentCalibrator.save()`)

### 2. Segment Conditioning <-> WinSegmentCalibrator

The new segment conditioning and the existing `WinSegmentCalibrator` serve different purposes:
- `WinSegmentCalibrator`: Post-hoc probability shrinkage for over-confident turf segments
- `MarketAwareWinCalibrator`: Pre-selection logit blending conditioned on market structure
- Both can coexist. The Benter blend happens first (probability generation), segment calibration happens second (probability adjustment)

### 3. Race-Level Ranker <-> WinSelectionPolicy

The ranker feeds into the existing selection pipeline:
- `WinRaceLevelRanker.score()` produces a deterministic composite score
- `WinSelectionPolicy` consumes the score for race-internal horse ranking
- The ranker replaces the ad-hoc `edge_win` ranking currently used, but the downstream filter chain (COLLAPSED skip -> dynamic EV_lower -> OddsBandFilter) is unchanged

### 4. Shadow Comparison <-> BacktestEngine

The comparison framework wraps the existing backtest:
- Run `BacktestEngine.run()` with baseline config -> `baseline_results.parquet`
- Run `BacktestEngine.run()` with shadow config -> `shadow_results.parquet`
- `ShadowComparator.compare()` joins on `(race_id, umaban)`, computes paired metrics
- Deployment gate is a pure function of the comparison results

## New Files to Create

| File | Purpose | Est. LOC |
|------|---------|----------|
| `src/models/market_aware_win_calibrator.py` | `MarketAwareWinCalibrator` class (segment-conditioned Benter blend) | ~250 |
| `src/models/win_race_level_ranker.py` | `WinRaceLevelRanker` class (deterministic composite scoring) | ~120 |
| `src/validation/shadow_comparator.py` | `ShadowComparator` class (baseline vs shadow metrics) | ~200 |
| `tests/test_market_aware_win_calibrator.py` | Unit tests | ~300 |
| `tests/test_win_race_level_ranker.py` | Unit tests | ~150 |
| `tests/test_shadow_comparator.py` | Unit tests | ~250 |

## Stack Decision Rationale Summary

| Decision | Rationale |
|----------|-----------|
| Extend BenterCombination, not replace | Global Benter blend is proven (in production since v1.1). Segment conditioning is an additive improvement, not a fundamental change. Wrapping is safer than rewriting. |
| Deterministic ranker, not LGBMRanker | 5-10 features, ~15 items per group = guaranteed overfitting with any learned ranker. Optuna-tuned weights give the same benefit with zero model complexity. |
| Segment keys from InvestmentFeatureFrame | The 94-spec frame already computes `if_popularity_rank`, `if_odds_band_id`, `if_p_win_race_rank`. No new feature computation. The calibrator consumes existing outputs. |
| scipy.stats for shadow comparison | KS-test and Mann-Whitney are standard non-parametric tests for distribution comparison. No domain-specific library needed. |
| No new pip dependencies | Every v2.1 feature is achievable with scipy + sklearn + pandas + numpy. Adding dependencies would require justification that does not exist. |

## Sources

- Codebase analysis: `src/models/benter_combination.py`, `src/models/win_benter_gate.py`, `src/models/win_segment_calibrator.py`, `src/investment/feature_frame.py`, `src/investment/schema_registry.py`
- Installed versions verified: `pip show` for lightgbm 4.6.0, scikit-learn 1.8.0, scipy 1.17.1, numpy 2.4.3, pandas 2.3.3
- scipy API: `scipy.optimize.minimize`, `scipy.stats.ks_2samp`, `scipy.stats.mannwhitneyu` confirmed available
- sklearn API: `IsotonicRegression`, `brier_score_loss`, `log_loss`, `calibration_curve` confirmed available
- PROJECT.md Key Decisions: "sklearn StackingClassifier rejected", "complex meta-learner (GBM/NN) rejected", "Benter-type market blend validated"

---
*Stack research for: v2.1 MarketAware Calibration + Race-Level Ranker*
*Researched: 2026-05-27*
