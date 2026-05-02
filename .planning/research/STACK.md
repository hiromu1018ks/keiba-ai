# Technology Stack -- Win Model Improvement

**Project:** keiba-ai win (tansho) model improvement
**Researched:** 2026-05-02
**Scope:** Incremental stack additions needed to push win betting ROI above 100%

## Context

The project already has a mature ML stack: Python 3.11, LightGBM >=4.3, XGBoost >=2.0, CatBoost >=1.2, scikit-learn >=1.4, pandas, pyarrow, MLflow. The 2-stage model (P(win) x E(odds|win)) is built. The Benter logit combination, RegimeDetector, StackedEnsemble (LGB+XGB+CAT -> Ridge), EV correction with init_score, and conformal confidence intervals are all in place.

**Problem:** Backtest ROI is 89% (loss). The stack gap is not in model architectures but in calibration precision, EV confidence estimation, and betting edge exploitation.

## Recommended Additions

### 1. Probability Calibration -- Upgrade from Isotonic to Ensemble Calibration

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `betacal` | >=1.1.0 | Beta calibration (3-parameter) | Smoother calibration curve than isotonic; data-efficient with 3 params; avoids step-function artifacts that distort EV estimates at boundary probability ranges | HIGH |
| scikit-learn `CalibratedClassifierCV` | >=1.6 (already have >=1.4) | Isotonic calibration (keep as-is) | Existing IsotonicRegression works well for high-sample regimes (>1000 calibration samples). Keep as the primary calibration path | HIGH |
| `TemperatureScaling` (custom, already built) | existing | Post-hoc temperature scaling | Already implemented in `benter_combination.py`. Sigmoid-based; good for correcting global over/underconfidence | HIGH |

**Recommendation:** Add `betacal` as a secondary calibration method. Run a calibration comparison (Isotonic vs Beta vs Temperature) on OOF validation data and select the best per-regime. The project already noted that isotonic can cause over-correction (CLAUDE.md references "Isotonic calibration skip -- overcorrection prevention"). Beta calibration's smooth parametric form is less prone to this.

**What NOT to use:**
- Platt scaling (logistic calibration) -- strictly inferior to beta calibration per Kull et al. (2017, ECML-PKDD). Beta calibration is a strict generalization (3 params vs 2).
- No deep learning calibration (e.g., mixup, label smoothing) -- not applicable to tree-based models.

### 2. Conformal Prediction -- Upgrade to MAPIE

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `mapie` | >=0.8.0 | Model-agnostic conformal prediction intervals for EV confidence bounds | Drop-in sklearn-compatible wrapper. Provides CQR (Conformalized Quantile Regression), split conformal, and jackknife+ methods. Replaces the hand-rolled conformal logic in `RobustConfidenceEstimator` with a well-tested library | MEDIUM |

**Rationale:** The existing `RobustConfidenceEstimator` (in `src/models/robust_confidence_estimator.py`) implements conformal prediction manually. MAPIE provides:
- CQR for adaptive-width prediction intervals (wider for longshots, narrower for favorites) -- critical for EV lower bound estimation
- Proper exchangeability guarantees
- sklearn-compatible API (fit/predict pattern)

However, confidence is MEDIUM because integrating MAPIE requires adapting the existing pipeline to work with MAPIE's API, and the current manual implementation may be sufficient if fixed.

**Alternative (lower risk):** Keep the manual conformal implementation but add Conformalized Quantile Regression logic (CQR) directly. The algorithm is simple enough to implement without MAPIE:
1. Train quantile regression models at alpha/2 and 1-alpha/2 quantiles
2. Compute nonconformity scores as max(q_low - y, y - q_high) on calibration set
3. Adjust quantile predictions by the conformity score quantile

**What NOT to use:**
- Full Bayesian approaches (e.g., PyMC, Stan) -- too heavy for the inference speed requirements; tree-based models don't have natural Bayesian posteriors.
- Dropout-based uncertainty (MC Dropout) -- only applies to neural networks.

### 3. Ensemble Strategy -- Fix the Stacked Ensemble

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| scikit-learn `Ridge` | existing | Meta-learner (keep) | Simple, regularized, prevents overfitting with correlated base model outputs | HIGH |
| LightGBM, XGBoost, CatBoost | existing | Level-1 base models (keep) | Already in `StackedEnsemble`. The issue is not the models but the hyperparameters | HIGH |
| `optuna` | >=3.5 (existing) | Hyperparameter tuning for ensemble base models | The current StackedEnsemble uses hardcoded params (lr=0.03, leaves=31, rounds=300). These need tuning per bet type | HIGH |

**Recommendation:** Do NOT add new model types (neural networks, etc.). The current 3-model stack is correct. The gap is:
1. **Tune each base model's hyperparameters with Optuna** -- currently hardcoded to generic values
2. **Add win-specific feature subsets** -- the ensemble currently shares the same features across all base models. LightGBM can handle categoricals natively; XGBoost needs encoding; CatBoost handles categoricals best. Exploit these differences.
3. **Use separate OOF generation** -- the current expanding-window fold split in `StackedEnsemble.train()` is correct for time-series, but `n_folds=3` may be too few. Increase to 5 folds for better meta-learner training.

**What NOT to use:**
- Neural network base models (PyTorch, TensorFlow) -- the data size (~9K bets/year) and feature types (tabular, categorical) favor GBDT. NNs are unlikely to add signal and introduce training complexity.
- Blending (hold-out meta-learner) -- the current stacking approach with OOF is correct and avoids information leakage.

### 4. EV Optimization -- Kelly Criterion Enhancement

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| `scipy.optimize` | existing (transitive) | Kelly optimization with constraints | Already used in `benter_combination.py`. Extend for constrained Kelly stake sizing | HIGH |
| Custom fractional Kelly with regime adaptation | N/A (code) | Dynamic Kelly fraction based on regime and EV confidence | The current `StakeCalculator` uses hardcoded half-Kelly (0.5) and a flat cap. Racing literature and practice show that Kelly fraction should vary with confidence in the edge estimate | HIGH |

**Recommendation:** The math for Kelly is already correct in `StakeCalculator.calc_stake()`. The improvements are all in the parameters:
1. **Regime-adaptive Kelly fraction:** Aggressive regime -> 0.5 Kelly, Conservative -> 0.3 Kelly, Collapsed -> 0.1 Kelly. Currently fixed at 0.5.
2. **EV-confidence-weighted Kelly:** Scale the Kelly fraction by the confidence in the EV estimate (from conformal prediction width). Narrow confidence interval -> full Kelly fraction; wide interval -> reduced fraction.
3. **Edge-dependent sizing:** Already partially implemented via `MIN_EDGE_THRESHOLD=0.005`. Consider a smooth ramp rather than a hard cutoff.

**What NOT to use:**
- Full Kelly -- too volatile for real-money betting. Fractional Kelly (0.25-0.5) is standard in professional sports betting.
- Optimal-f (Ralph Vince) -- maximizes geometric growth but requires exact distribution knowledge; too aggressive for model-based probability estimates.
- Fixed-percentage staking -- ignores edge magnitude and wastes EV.

### 5. Race-Level Probability Normalization

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Custom softmax over race entries | N/A (code) | Normalize per-horse probabilities to sum to 1.0 within each race | The current `_normalize_probability_by_race()` in `ev_correction_model.py` uses a capped iterative proportional fitting approach. Consider adding a proper softmax alternative | HIGH |
| `scipy.special.softmax` | existing (transitive) | Numerically stable softmax | Already available via scipy. Use for race-level normalization after calibration | HIGH |

**Recommendation:** The existing normalization is acceptable but could be simplified. Two options:
1. **Softmax:** Apply `logit(p) -> softmax over race entries -> normalized p`. Standard approach from the Benter (1994) multinomial logit formulation.
2. **Keep current iterative normalization:** It handles edge cases (capping at 1.0) better than naive softmax. The current approach is actually more robust.

Verdict: Keep the current normalization. It already handles caps correctly. The softmax approach can produce very small probabilities for longshots that harm EV estimation.

## Summary: What to Install

```bash
# NEW dependency -- Beta calibration
pip install betacal>=1.1.0

# OPTIONAL -- MAPIE for conformal prediction (or implement CQR manually)
pip install mapie>=0.8.0

# Everything else is already in pyproject.toml
# No new ML frameworks needed
```

## Version Bumps Worth Considering

| Package | Current | Target | Reason | Priority |
|---------|---------|--------|--------|----------|
| scikit-learn | >=1.4 | >=1.6 | FrozenEstimator for clean calibration workflow; Array API support | Low (1.4 works fine) |
| LightGBM | >=4.3 | >=4.6 | Linear tree boosting; improved categorical handling; bug fixes | Medium |
| Optuna | >=3.5 | >=4.0 | API stability; improved samplers | Low (3.5 works fine) |

**Do NOT bump Python beyond 3.11.** The project uses `mise.toml` to pin 3.11, and all dependencies are compatible. Python 3.12+ is unnecessary for this codebase.

## Alternatives Considered and Rejected

| Category | Recommended | Rejected | Why |
|----------|-------------|----------|-----|
| Calibration | Isotonic + BetaCal | Platt scaling | Beta is strictly superior to Platt (Kull 2017); isotonic is already in place |
| Calibration | Isotonic + BetaCal | Dirichlet calibration | Only relevant for multiclass; win prediction is binary (win/not-win) |
| Ensemble | GBDT stack (LGB+XGB+CAT) | Neural network (PyTorch/TF) | Tabular data, small sample size; GBDT dominates |
| Ensemble | GBDT stack | Random Forest base | RF adds diversity but less signal than another GBDT variant; not worth the complexity |
| Conformal | MAPIE or manual CQR | Full Bayesian (PyMC) | Overkill for tree models; speed penalty |
| Staking | Fractional Kelly | Full Kelly | Too volatile; professional betting universally uses fractional |
| Staking | Fractional Kelly | Fixed stake | Wastes edge information |
| Data | Existing EveryDB2 | New data sources (weather, sectional times) | Out of scope per PROJECT.md; focus on model improvement first |

## Key Technical Insights

### Why ROI < 100%: Root Cause Analysis from Stack Perspective

1. **Calibration drift:** The isotonic regression was noted to cause over-correction. When calibration distorts probabilities at the tails (high-odds horses), the EV estimates become unreliable, leading to poor bet selection.

2. **Ensemble underfitting:** The `StackedEnsemble` uses hardcoded hyperparameters (lr=0.03, leaves=31, rounds=300). These are conservative defaults that may not capture the win-specific signal. Win prediction has a ~7% base rate (1 winner per ~14 horses), making it a harder classification problem than place (~21% base rate).

3. **EV lower bound too loose:** The conformal prediction intervals may be too wide, causing the `ev_lower_win_corrected` threshold to reject too many true-positive bets. Or too narrow, causing acceptance of negative-EV bets. The manual conformal implementation needs validation against a library like MAPIE.

4. **Static Kelly fraction:** Using 0.5 regardless of confidence or regime means over-betting in uncertain conditions and under-betting in clear-edge situations.

5. **Benter combination weights:** The `beta` lower bound of 0.20 forces a minimum 20% weight on market probabilities. This may be too high for races where the model has a strong fundamental edge, or too low for races where the market is highly efficient.

## Sources

- [betacal on PyPI](https://pypi.org/project/betacal/) -- version 1.1.0, official package
- [betacal on GitHub](https://github.com/betacal/python) -- source code and paper links
- [Kull et al. (2017) -- Beta Calibration](https://betacal.github.io/) -- original paper proving beta calibration superiority
- [scikit-learn calibration docs](https://scikit-learn.org/stable/modules/calibration.html) -- IsotonicRegression and CalibratedClassifierCV
- [LightGBM releases](https://github.com/lightgbm-org/LightGBM/releases) -- version 4.6.0 changelog
- [MAPIE documentation](https://mapie.readthedocs.io/) -- conformal prediction library
- [Benter (1994) paper](https://gwern.net/doc/statistics/decision/1994-benter.pdf) -- foundational horse racing prediction model
- [Kelly Criterion -- Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion) -- mathematical foundation
- [Optimizing Horse Racing Predictions through Ensemble Learning (ResearchGate)](https://www.researchgate.net/publication/385301910) -- stacking ensemble validation
- [scikit-learn FrozenEstimator (v1.6)](https://scikit-learn.org/stable/whats_new/v1.6.html) -- clean prefit calibration workflow
