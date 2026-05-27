# Technology Stack

**Project:** keiba-ai v2.0 Investment Pipeline Restructuring
**Researched:** 2026-05-27
**Supersedes:** v1.8 STACK.md

## Verdict: Zero New External Dependencies

All 4 v2.0 components are implementable with the currently installed stack.

## Current Installed Stack

| Package | Installed Version | Role in v2.0 |
|---------|-------------------|-------------|
| Python | 3.11 | Pinned via mise |
| LightGBM | 4.6.0 | LGBMRanker for race-level ranking; LGBMClassifier as alternative calibrator |
| scikit-learn | 1.8.0 | LogisticRegression, CalibratedClassifierCV, brier_score_loss, log_loss, IsotonicRegression |
| scipy | 1.17.1 | MLE optimization for Benter blend (scipy.optimize.minimize) |
| pandas | 2.3.3 | InvestmentFeatureFrame construction; groupby for race-level operations |
| numpy | 2.4.3 | logit/expit for Benter blend; array operations |
| betacal | 1.1.0 | Beta calibration as alternative post-processing |
| joblib | (bundled) | Model serialization for calibrator and ranker |
| optuna | 4.8.0 | HP tuning for LGBMRanker and MarketAwareWinCalibrator |
| mlflow | 3.10.1 | Log new calibrator/ranker as MLflow artifacts |

## Recommended Stack for v2.0

| Component | Implementation | Why This Choice |
|-----------|---------------|-----------------|
| **MarketAwareWinCalibrator** | Extend existing `BenterCombination` class; LogisticRegression as alternative | BenterCombination already has working MLE fitting with alpha/beta/gamma. Extension adds segment conditioning. LogisticRegression provides regularized alternative if MLE is unstable. |
| **Race-Level Ranker** | `lightgbm.LGBMRanker` with `objective='lambdarank'` | Already in installed LightGBM. LambdaRank handles within-race ranking directly. The `group` parameter maps to race_id group sizes. |
| **InvestmentFeatureFrame** | Pure pandas/numpy | Column curation from existing feature modules. Uses groupby, rank, cut -- all existing patterns. |
| **OOF Health** | Pure Python + sklearn metrics | brier_score_loss, log_loss, custom ECE. No new libs needed. |

## What NOT to Add

| Library/Approach | Why Rejected |
|-----------------|-------------|
| torch / tensorflow | No deep learning in v2.0. Tabular data with ~100K rows is LightGBM's domain. |
| CatBoostRanker / XGBRanker | One ranker library is enough. LightGBM LGBMRanker is sufficient. |
| cvxpy for portfolio optimization | Deferred to future milestone. Overkill for max-3-horse allocation. |
| polars | Pandas is the project standard. Dual-dataframe ecosystem is not justified. |

## New Files to Create

| File | Purpose |
|------|---------|
| `src/features/investment_features.py` | `build_win_investment_features()`, `validate_investment_features()` |
| `src/models/market_aware_win_calibrator.py` | `MarketAwareWinCalibrator` class (train/apply/save/load) |
| `src/models/win_race_level_ranker.py` | `WinRaceLevelRanker` class (train/predict/save/load) |
| `src/validation/oof_health.py` | OOF health check functions |
| `tests/test_market_aware_win_calibrator.py` | Unit tests |
| `tests/test_win_race_level_ranker.py` | Unit tests |
| `tests/test_oof_health.py` | Unit tests |

## Sources

- LightGBM LGBMRanker API: Context7 `/lightgbm-org/lightgbm`, `objective='lambdarank'`, `group` parameter, `eval_at`
- sklearn calibration metrics: Context7 `/websites/scikit-learn_stable`
- pip show verification: LightGBM 4.6.0, scikit-learn 1.8.0, scipy 1.17.1
- Codebase analysis: `src/models/benter_combination.py`, `src/models/win_segment_calibrator.py`

---
*Stack research for: v2.0 Investment Pipeline Restructuring*
*Researched: 2026-05-27*
