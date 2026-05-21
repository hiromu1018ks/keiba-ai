---
phase: 30-residual-ic-evaluation-framework
plan: 01
status: complete
---

# Plan 30-01 Summary: IC Evaluation Module

## What was done

Created `src/models/ic_evaluator.py` — IC evaluation module implementing 4 IC formulations (B-difference, C-orthogonal, E-incremental, Per-race) with surface-specific computation (turf/dirt/all), direction consistency check, JSON output, and MLflow integration.

Created `tests/test_ic_evaluator.py` — 17 unit tests covering all computation functions, edge cases, and integration.

## Files created/modified

- `src/models/ic_evaluator.py` (new) — 345 lines
- `tests/test_ic_evaluator.py` (new) — 170 lines

## Verification

- 17/17 tests pass
- ruff check clean
- No new external dependencies

## Key decisions

- Followed `ev_diagnostics.py` pattern exactly (module-level constants, private functions, public orchestration, JSON output)
- Used `numpy.linalg.lstsq` for OLS residuals (lighter than sklearn LinearRegression)
- Pre-filter NaN with `np.isfinite` before `spearmanr` (avoids scipy edge cases)
- `model_prob_filter` helper for surface-specific array slicing
