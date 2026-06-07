---
status: resolved
trigger: phase46-runtime-verification
tdd_mode: false
goal: find_and_fix
created: 2026-06-01
---

# Debug Session: Phase 46 Runtime Verification

## Current Focus

**Hypothesis:** CONFIRMED. Stage 1 fails because `is_shadow_candidate` property in `mawc_conservative_retrainer.py:111-125` has a logic bug: the comment says "ECE intentionally excluded" but `favorite_band_guard.overall_passed` includes the favorite-band ECE gate. Since the baseline MAWC was trained on the same OOF data and achieves ECE ~0.003 (overall) and ~0.008 (favorite band), the conservative variant's ECE of 0.014-0.023 (overall) and 0.063-0.109 (favorite band) structurally cannot pass. The `is_shadow_candidate` path was designed to bypass this issue but fails because it includes `favorite_band_guard.overall_passed` which wraps the ECE check.

**Next Action:** Fix `is_shadow_candidate` to exclude favorite-band ECE, using only p_compression and ev_pass_rate as the sanity guards (matching the documented intent at lines 116-119).

## Evidence

- 2026-06-01T07:20:Z: Stage 1 re-run with `--force` produces fresh manifest (CR-01 fixed, per_year_surface now properly keyed by year). All surfaces still `deployment_status: rejected`, `shadow_candidate_saved: false`.
- 2026-06-01T07:25:Z: Diagnostic probe on turf surface shows:
  - Baseline: Brier=0.059684, Logloss=0.209939, ECE=0.003139
  - C=0.03 (best): Brier=0.038280 (PASS), Logloss=0.137135 (PASS), ECE=0.014449 (FAIL -- 4.6x baseline)
  - Favorite band: ECE_base=0.008074, ECE_cons=0.062673 (FAIL -- 7.8x baseline), p_comp=1.0086 (PASS), EV=0.3203 vs 0.0000 (PASS)
  - ALL C values: Brier PASS, Logloss PASS, overall ECE FAIL, fav ECE FAIL
- 2026-06-01T07:25:Z: Root cause identified: `is_shadow_candidate` (line 124) checks `self.favorite_band_guard.overall_passed` which includes `ece_passed`. The comment at lines 116-119 explicitly states "ECE is intentionally excluded" but the implementation includes it.
- 2026-06-01T07:25:Z: All other sanity guards pass: p_compression >= 0.90 (1.02-1.03), ev_pass_rate_passes. Only fav ECE blocks shadow candidacy.

## Investigation Log

### Cycle 1: Initial State Assessment

- Examined existing artifacts: manifest.json, retrain_summary.md
- All surfaces show n_passing=0, deployed=false
- Manifest format predates `deployment_status`/`shadow_candidate_saved` fields
- CR-01 pattern confirmed: 2024 and 2025 rows are identical in retrain_summary
- Code now has per_year_surface keyed by year, fixing CR-01

### Cycle 2: Stage 1 Execution

- Ran `python scripts/run_phase46_quality_gates.py --stage 1 --force --report`
- Result: EXIT CODE 1 -- "No shadow candidates saved in conservative variant. All surfaces rejected"
- New manifest generated: deployment_status=rejected for all surfaces, shadow_candidate_saved=false
- CR-01 fixed: 2024 and 2025 now have separate entries in per_year_surface

### Cycle 3: Diagnostic Probe

- Ran inline Python to compare baseline vs conservative metrics per C value
- Baseline ECE is ~0.003 (nearly zero -- trained on same data)
- Conservative ECE ranges 0.014-0.023 (7x baseline) -- structurally cannot pass 10% tolerance
- Favorite band ECE: baseline 0.008, conservative 0.063-0.109 -- 8-14x baseline
- Brier and Logloss consistently pass (conservative is actually BETTER)
- p_compression passes (>0.90), EV pass rate passes
- Only ECE gates fail

### Cycle 4: Code Trace

- `select_best_for_shadow()` (line 575-600): 3-level selection
  1. Try `all_gates_passed` -- fails (ECE)
  2. Try `is_shadow_candidate` -- ALSO fails because `is_shadow_candidate` includes `favorite_band_guard.overall_passed` which wraps `ece_passed`
  3. All rejected
- `is_shadow_candidate` property (lines 111-125): checks `brier_non_degraded AND logloss_non_degraded AND favorite_band_guard.overall_passed`
- `favorite_band_guard.overall_passed` = `ece_passed AND p_compression_passed AND ev_pass_rate_passed`
- Comment says "ECE is intentionally excluded" but implementation includes it via `overall_passed`
- BUG: should check `p_compression_passed AND ev_pass_rate_passed` only (excluding `ece_passed`)

## Resolution

### root_cause

`is_shadow_candidate` property at `src/models/mawc_conservative_retrainer.py:124` uses `self.favorite_band_guard.overall_passed` which includes the favorite-band ECE gate. The documented intent (lines 116-119) is to exclude ECE from the shadow candidate check, but the implementation includes it via `overall_passed`. Since baseline ECE is ~0.003 (trained on same data), the conservative variant's ECE of 0.014-0.023 structurally cannot pass, making `shadow_only` status unreachable.

### fix

Change `is_shadow_candidate` property to exclude favorite-band ECE from the sanity guard:
```python
@property
def is_shadow_candidate(self) -> bool:
    return (
        self.brier_non_degraded
        and self.logloss_non_degraded
        and self.favorite_band_guard.p_compression_passed
        and self.favorite_band_guard.ev_pass_rate_passed
    )
```
This matches the documented intent at lines 116-119: "ECE is intentionally excluded because the baseline ECE (~0.003) with a 10% relative tolerance makes the gate structurally impossible for conservative candidates."
