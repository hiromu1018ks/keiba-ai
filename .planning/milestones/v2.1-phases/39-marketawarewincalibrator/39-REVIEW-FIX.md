---
phase: 39-marketawarewincalibrator
fixed_at: 2026-05-28T12:30:00Z
review_path: .planning/phases/39-marketawarewincalibrator/39-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 39: Code Review Fix Report

**Fixed at:** 2026-05-28T12:30:00Z
**Source review:** .planning/phases/39-marketawarewincalibrator/39-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: ZeroDivisionError in apply() when race probabilities sum to zero

**Files modified:** `src/models/market_aware_win_calibrator.py`
**Commit:** d4c81dc
**Applied fix:** Added `.clip(lower=1e-10)` to `race_sums` before division in `apply()` method to prevent `inf`/`nan` in `p_win_final` when all calibrated probabilities collapse to zero.

### CR-02: train() mutates caller's DataFrame via conditional df["p_model"] assignment

**Files modified:** `src/models/market_aware_win_calibrator.py`
**Commit:** 23deb55
**Applied fix:** Moved `df = df.copy()` to be unconditional and placed before the conditional `p_model` column assignment, ensuring the caller's DataFrame is never mutated regardless of input columns.

### WR-01: Broad exception catch silently swallows OOF fold training failures

**Files modified:** `src/models/win_benter_gate.py`
**Commit:** 5d6249d
**Applied fix:** Narrowed exception type from `Exception` to `(ValueError, RuntimeError)` and added a failure counter that raises `RuntimeError` when all folds fail, preventing silent empty-result returns.

### WR-02: Duplicated _walk_forward_race_splits across three modules

**Files modified:** `src/utils/wf_splits.py` (new), `src/models/market_aware_win_calibrator.py`, `src/models/win_benter_gate.py`, `src/pipelines/training_pipeline.py`
**Commit:** 819da04
**Applied fix:** Extracted the function to a new shared module `src/utils/wf_splits.py` with the most robust variant (includes `df.empty` check and default `n_splits=5`). All three call sites now import from the shared module via `from utils.wf_splits import walk_forward_race_splits as _walk_forward_race_splits`. Removed inline function definitions and unused `TimeSeriesSplit` imports from all three modules.

### WR-03: load_from_dir target_encoder missing in MLflow path

**Files modified:** `src/db/model_loader.py`
**Commit:** 749d7ad
**Applied fix:** Added `target_encoder` loading in the MLflow `load()` method using the same pattern as `load_from_dir()`: downloads the artifact directory for `target_encoder_{surface}`, finds `.joblib` files, and loads via `joblib.load`. Added `target_encoder=target_encoder` to the `SubmodelSet` constructor in the MLflow path.

### WR-04: _check_ratio_gates only logs, never acts on detected imbalances

**Files modified:** `src/models/market_aware_win_calibrator.py`
**Commit:** 969ebd0
**Applied fix:** Renamed `_check_ratio_gates` to `_check_ratio_diagnostics` (both the method definition and the call site) to accurately reflect that the method is purely diagnostic and does not gate deployment.

### WR-05: compare_calibrations betacal dead code

**Files modified:** `src/models/win_benter_gate.py`
**Commit:** f5d5f57
**Applied fix:** Removed the `betacal` import attempt and `try/except` block (19 lines of dead code). The manual `BetaCalibrationManual` was always used regardless of whether `betacal` was available. Replaced with a simple 4-line manual-only path.

---

_Fixed: 2026-05-28T12:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
