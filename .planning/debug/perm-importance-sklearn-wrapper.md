---
status: awaiting_human_verify
trigger: "compute_permutation_importance passes predict_fn to sklearn permutation_importance instead of estimator object with fit method"
created: 2026-05-12T00:00:00Z
updated: 2026-05-12T00:01:00Z
---

## Current Focus
hypothesis: CONFIRMED - compute_permutation_importance() passed a raw callable to sklearn's permutation_importance(), which requires an estimator object.
test: Implemented _LGBMClassifierWrapper and _LGBMRegressorWrapper (inheriting ClassifierMixin/BaseEstimator and RegressorMixin/BaseEstimator respectively), replaced predict_fn usage, ran manual verification and existing test suite.
expecting: All tests pass and permutation_importance works for both binary and regression models.
next_action: Await human verification of the fix in real workflow.

## Symptoms
expected: analyze_feature_importance.py --tier-report runs successfully, computing permutation importance for all models.
actual: Crashes with InvalidParameterError: The 'estimator' parameter of permutation_importance must be an object implementing 'fit'.
errors: sklearn.utils._param_validation.InvalidParameterError
reproduction: python scripts/analyze_feature_importance.py --tier-report --model-dir data/models
started: Always (the code was never correct for this path)

## Eliminated

- hypothesis: Plain function wrapper (predict_fn) would work as sklearn estimator
  evidence: sklearn 1.8 requires an object with fit() method, not a callable
  timestamp: 2026-05-12T00:00:00Z

- hypothesis: Single _LGBMWrapper class with is_binary flag would work
  evidence: sklearn 1.8 requires proper MRO ordering (Mixin before BaseEstimator) and separate ClassifierMixin/RegressorMixin for tag introspection
  timestamp: 2026-05-12T00:00:30Z

## Evidence
- timestamp: 2026-05-12T00:00:00Z
  checked: src/features/win_feature_analysis.py lines 103-168
  found: Line 141-143 creates predict_fn (a plain function) and passes it as the first arg to permutation_importance() at line 155. sklearn requires an estimator with a fit() method.
  implication: Root cause confirmed. Need to wrap lgb.Booster in an sklearn-compatible estimator class.

- timestamp: 2026-05-12T00:00:15Z
  checked: _LGBMWrapper(BaseEstimator) without Mixin classes
  found: sklearn 1.8 calls __sklearn_tags__() which is provided by BaseEstimator, but estimator_type was None without ClassifierMixin/RegressorMixin
  implication: Need separate wrapper classes with proper Mixin inheritance

- timestamp: 2026-05-12T00:00:30Z
  checked: MRO ordering BaseEstimator before Mixin
  found: sklearn 1.8 requires Mixin BEFORE BaseEstimator in MRO for estimator_type tag to be set correctly
  implication: Changed to ClassifierMixin, BaseEstimator and RegressorMixin, BaseEstimator ordering

- timestamp: 2026-05-12T00:01:00Z
  checked: compute_permutation_importance with mock Booster
  found: Both binary (neg_log_loss) and regression (neg_mean_absolute_error) paths work correctly. All 23 existing tests pass.
  implication: Fix is verified in isolation and no regressions introduced.

## Resolution

root_cause: compute_permutation_importance() passes a raw predict_fn callable to sklearn's permutation_importance(), which requires an estimator object implementing the fit() method.
fix: Replaced predict_fn with two sklearn-compatible wrapper classes: _LGBMClassifierWrapper (ClassifierMixin + BaseEstimator) for binary models and _LGBMRegressorWrapper (RegressorMixin + BaseEstimator) for regression models. Both implement fit() (no-op) and predict()/predict_proba() delegating to the underlying lgb.Booster.
verification: Manual test with mock Booster for both binary and regression modes - both work. 23 existing tests pass with zero regressions.
files_changed: [src/features/win_feature_analysis.py]
