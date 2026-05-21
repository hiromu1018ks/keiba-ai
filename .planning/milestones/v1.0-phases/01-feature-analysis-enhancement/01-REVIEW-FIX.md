---
phase: 01-feature-analysis-enhancement
fixed_at: 2026-05-02T00:00:00Z
review_path: .planning/phases/01-feature-analysis-enhancement/01-REVIEW.md
iteration: 1
findings_in_scope: 10
fixed: 10
skipped: 0
status: all_fixed
---

# Phase 1: Code Review Fix Report

**Fixed at:** 2026-05-02
**Source review:** .planning/phases/01-feature-analysis-enhancement/01-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 10
- Fixed: 10
- Skipped: 0

## Fixed Issues

### CR-01: validate_noise_removal trains and evaluates on the same data

**Files modified:** `src/features/win_feature_analysis.py`
**Commit:** `20862d8`
**Applied fix:** Replaced full-data training+evaluation with chronological 80/20 train/valid split. Both original and new model metrics are now computed on the validation split only, eliminating in-sample bias.

### CR-02: PlaceTwoStageModel silently drops odds_to_ability_ratio at inference time

**Files modified:** `src/models/two_stage_return_model.py`
**Commit:** `19c89c9`
**Applied fix:** Added the same `odds_to_ability_ratio` fallback computation to `PlaceTwoStageModel._prepare_features()` that `WinTwoStageModel._prepare_features()` already has. The fallback computes the ratio from `p_market_win_adj` and `p_ability_win` when the column is missing.

### CR-03: remove_noise_features mutates class-level mutable list -- unsafe for concurrent use

**Files modified:** `src/models/two_stage_return_model.py`, `tests/test_two_stage_return_model.py`
**Commit:** `883cddc`
**Applied fix:** Added `get_filtered_feature_cols()` classmethod that returns a new list without mutating the class variable. Added thread-safety warning in `remove_noise_features()` docstring. Added 2 new tests to verify the new method works correctly and does not mutate class state.

### WR-01: odds_to_ability_ratio computation placement in training pipeline is fragile

**Files modified:** `src/pipelines/training_pipeline.py`
**Commit:** `d3a30f9`
**Applied fix:** Added clarifying comment explaining the intentional dependency ordering: the computation uses `p_ability_win` only (not place), must run before `PlaceAbilityModel.train()`, and should be moved if future changes require place probabilities.

### WR-02: history_mask uses only kakuteijyuni > 0 filter but valid_mask also requires syussotosu >= 8

**Files modified:** `src/features/horse_history_features.py`
**Commit:** `1142e35`
**Applied fix:** Added `valid_field == 1` (syussotosu >= 8) condition to `history_mask`, consistent with `valid_mask` criteria used for other features. Uses `np.ones()` as default fallback when `valid_field` is not available in the array dict.

### WR-03: class_drop_bounce logic is counterintuitive without comment

**Files modified:** `src/features/horse_history_features.py`
**Commit:** `99221fd`
**Applied fix:** Added clarifying comment explaining that `norm_recent_b` maps 0=first place, 1=last place; `avg_recent_b > 0.5` means back-half finish (poor form); and higher average means worse form = stronger bounce signal.

### WR-04: win_dominance returns 0.0 for no-wins -- semantically ambiguous

**Files modified:** `src/features/horse_history_features.py`
**Commit:** `0a492b`
**Applied fix:** Changed return value for horses with history but no wins from `0.0` to `float("nan")`, consistent with the no-history case. This removes semantic ambiguity between "never won" and a legitimate dominance score.

### WR-05: analyze_feature_importance assert provides no useful error message

**Files modified:** `src/features/win_feature_analysis.py`
**Commit:** `40b9c06`
**Applied fix:** Replaced bare `assert` with a descriptive `ValueError` including shape information: actual columns, expected columns (n_features + 1), and model feature count.

### WR-06: _load_features_for_analysis fallback creates meaningless SHAP values

**Files modified:** `scripts/analyze_feature_importance.py`
**Commit:** `f0243f8`
**Applied fix:** Replaced zero-filled DataFrame fallback with `sys.exit(1)` and descriptive error log message. Zero-filled SHAP values would produce misleading analysis results.

### WR-07: _find_model_file picks arbitrary surface model without user control

**Files modified:** `scripts/analyze_feature_importance.py`
**Commit:** `1cc0264`
**Applied fix:** Added `--surface` argument to argparse with choices=["turf", "dirt"], default="turf". Updated `_find_model_file` to accept `preferred_surface` parameter and prioritize the user-selected surface while deduplicating the search order.

---

_Fixed: 2026-05-02_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
