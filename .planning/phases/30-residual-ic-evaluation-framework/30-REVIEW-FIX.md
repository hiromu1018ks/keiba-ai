---
phase: 30-residual-ic-evaluation-framework
fixed_at: 2026-05-18T14:16:00+09:00
review_path: .planning/phases/30-residual-ic-evaluation-framework/30-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 30: Code Review Fix Report

**Fixed at:** 2026-05-18T14:16:00+09:00
**Source review:** .planning/phases/30-residual-ic-evaluation-framework/30-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5
- Fixed: 5
- Skipped: 0

## Fixed Issues

### CR-01: `_check_direction_consistency` crashes when Spearman rho is exactly 0.0

**Files modified:** `src/models/ic_evaluator.py`
**Commit:** 582d77b
**Applied fix:** Replaced `or`-chain (`metric.get("rho") or metric.get("delta_ic") or metric.get("mean_rho")`) with explicit `if rho is None` checks. This prevents falsy short-circuit on `rho = 0.0`, avoiding `TypeError` from `np.isfinite(None)`.

### CR-02: Per-race IC has inverted sign convention vs B/C/E formulations

**Files modified:** `src/models/ic_evaluator.py`
**Commit:** 3896c6e
**Applied fix:** Negated the Spearman rho in `_compute_per_race_ic` so that positive rho always means "model adds predictive value", consistent with B/C/E formulations which use binary y. Added explanatory comments. **Requires human verification** of sign convention correctness with production data.

### WR-01: `_get_market_probability` returns unvalidated implied_prob values

**Files modified:** `src/models/ic_evaluator.py`
**Commit:** 5fae3ce
**Applied fix:** Added `np.clip(np.where(np.isfinite(raw), raw, np.nan), 0.01, 0.99)` to the `implied_prob` path, matching the validation already applied to the `tanodds` fallback path.

### WR-02: `model_prob_filter` should be `_model_prob_filter`

**Files modified:** `src/models/ic_evaluator.py`
**Commit:** e6a6042
**Applied fix:** Renamed `model_prob_filter` to `_model_prob_filter` and updated the internal call site in `run_ic_evaluation`.

### WR-03: Per-race IC operates on different data subset than B/C/E

**Files modified:** `src/models/ic_evaluator.py`
**Commit:** a8de87b
**Applied fix:** Constructed `sub_df_valid` by combining `surface_filter` with a `valid_mask` that checks `np.isfinite` on all three arrays (`model_pred`, `market_prob`, `y`). Passed `sub_df_valid` to `_compute_per_race_ic` instead of the raw `sub_df`, ensuring all four IC formulations operate on the same row subset.

## Skipped Issues

None -- all 5 findings were successfully fixed.

---

_Fixed: 2026-05-18T14:16:00+09:00_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
