---
status: awaiting_human_verify
trigger: "validate_noise_removal() fails for stage1 models with y_prob values outside [0,1]"
created: 2026-05-12T00:00:00Z
updated: 2026-05-12T00:00:01Z
---

## Current Focus

hypothesis: CONFIRMED - booster.predict() returns raw logits (outside [0,1]) for stage1 binary models, passed directly to sklearn log_loss/roc_auc_score
test: sigmoid guard applied at both predict sites, all 23 tests pass, end-to-end script completes without ValueError
expecting: no regression in existing tests, prune_noise_features.py no longer crashes on stage1 models
next_action: await human verification

## Symptoms

expected: validate_noise_removal() computes logloss/AUC for stage1 models without error
actual: ValueError: y_prob contains values greater than 1: 1.18 / values lower than 0: -1.72
errors: "y_prob contains values greater than 1: 1.18", "y_prob contains values lower than 0: -1.72"
reproduction: call validate_noise_removal() with a stage1 model whose booster.predict() returns raw logits
started: always been the case for models that output raw logits

## Eliminated

(none needed - root cause was confirmed in bug report)

## Evidence

- timestamp: 2026-05-12T00:00:00Z
  checked: _LGBMClassifierWrapper.predict_proba() at line 79
  found: Already has sigmoid guard: `if raw.min() < 0.0 or raw.max() > 1.0: raw = 1.0 / (1.0 + np.exp(-raw))`
  implication: The pattern is established in the codebase, just missing from validate_noise_removal()

- timestamp: 2026-05-12T00:00:00Z
  checked: validate_noise_removal() lines 571, 620
  found: Both calls to model.predict() pass raw output directly to log_loss/roc_auc_score without sigmoid conversion
  implication: Root cause confirmed - raw logits from booster.predict() reach sklearn metrics

- timestamp: 2026-05-12T00:00:01Z
  checked: pytest tests/test_win_feature_analysis.py
  found: All 23 tests pass (7.25s)
  implication: Fix does not break any existing behavior

- timestamp: 2026-05-12T00:00:01Z
  checked: python scripts/prune_noise_features.py --model-dir data/models --features-path data/features/horse_features.parquet
  found: Script completes successfully, no ValueError on stage1 models. stage1 models still have safety_passed=false but due to a DIFFERENT issue (dtype mismatch for distance_bin, grade_code columns) - not the raw logits bug.
  implication: The raw logits bug is fixed; a separate dtype issue remains for OOF safety validation

## Resolution

root_cause: validate_noise_removal() calls booster.predict() directly and passes raw logits to sklearn log_loss/roc_auc_score, which require probabilities in [0,1]. The _LGBMClassifierWrapper.predict_proba() already had a sigmoid guard for this, but validate_noise_removal() did not.
fix: Added sigmoid conversion (1.0 / (1.0 + np.exp(-pred))) after both predict calls in validate_noise_removal() when values fall outside [0,1]. Applied at lines 572-576 (original model) and 626-630 (new model), matching the existing pattern in _LGBMClassifierWrapper.
verification: 23/23 tests pass. End-to-end prune_noise_features.py completes without ValueError on stage1 models.
files_changed: [src/features/win_feature_analysis.py]
