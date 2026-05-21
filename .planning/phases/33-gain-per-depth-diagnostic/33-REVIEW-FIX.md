---
phase: 33-gain-per-depth-diagnostic
fixed_at: 2026-05-18T13:30:00Z
review_path: .planning/phases/33-gain-per-depth-diagnostic/33-REVIEW.md
iteration: 1
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 33: Code Review Fix Report

**Fixed at:** 2026-05-18T13:30:00Z
**Source review:** .planning/phases/33-gain-per-depth-diagnostic/33-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 3
- Fixed: 3
- Skipped: 0

## Fixed Issues

### WR-01: Non-Booster models silently dropped without `_is_booster` guard

**Files modified:** `src/models/gpd_diagnostics.py`
**Commit:** c733b33
**Applied fix:** Added `_is_booster()` validation with `logger.warning()` fallback to all model extractions that previously used bare `is not None` checks: Market Model, EV Correction (P/E), Place EV Correction (P/E), Wide (hit/return), and ConformalEV/CQR (q_low/q_high). Non-Booster objects now log a warning with the model name and type instead of being silently included.

### WR-02: Inconsistent StackedEnsemble unwrapping -- `place` path does not unwrap

**Files modified:** `src/models/gpd_diagnostics.py`
**Commit:** 7e2ec3f
**Applied fix:** Added StackedEnsemble unwrapping logic (`elif hasattr(place_hit, "lgbm_model")`) to the place extraction path, mirroring the existing win path. If `place.hit_model` is wrapped in a StackedEnsemble, it now extracts `.lgbm_model` as `place_ensemble_lgbm_{surface}`.

### WR-03: Broad exception catch silently skips boosters

**Files modified:** `src/models/gpd_diagnostics.py`
**Commit:** 5df4f3c
**Applied fix:** Added `failed_names: list[str]` tracker in `compute_gpd_diagnostics`. Failed booster names are collected during the exception handler and written to `result["metadata"]["failed_boosters"]` and `result["metadata"]["num_failed"]` when any failures occur.

---

_Fixed: 2026-05-18T13:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
