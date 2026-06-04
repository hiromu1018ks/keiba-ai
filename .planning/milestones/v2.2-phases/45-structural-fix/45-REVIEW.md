---
phase: 45-structural-fix
reviewed: 2026-05-31T14:30:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - src/models/mawc_conservative_retrainer.py
  - src/models/mawc_conservative_report.py
  - scripts/run_mawc_conservative_retrain.py
  - src/models/templates/mawc_conservative_report.html
  - tests/test_mawc_conservative_retrainer.py
findings:
  critical: 2
  warning: 4
  info: 2
  total: 8
status: issues_found
---

# Phase 45: Code Review Report

**Reviewed:** 2026-05-31T14:30:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Reviewed the MAWC Conservative Retrainer implementation (5 source files). The core ML logic (36-dim feature matrix, C grid search, quality gates) is sound and well-structured. However, two critical bugs were found in the multi-year pipeline: (1) manifest `per_surface` silently overwrites year-specific results, losing data for all but the last year, and (2) the CLI script re-runs the entire retraining pipeline a second time, wasting computation and producing results that may differ from the manifest if data is non-deterministic. Additional warnings include a class variable shared mutable state risk and missing error handling.

## Critical Issues

### CR-01: `generate_manifest` silently overwrites per-surface results in multi-year runs

**File:** `src/models/mawc_conservative_retrainer.py:701-704`
**Issue:** `generate_manifest()` builds `per_surface` as `dict[str, dict]` keyed by surface name. When `all_results` contains entries for the same surface across multiple years (e.g., turf-2024 and turf-2025), the dict assignment `per_surface[result.surface] = {...}` silently overwrites earlier years with the last year's data. This means the manifest always reflects only the last year's metrics, losing quality gate and deployment data from all prior years.

The root cause is in `run_full_pipeline()` (lines 775-812) which accumulates `all_results` from every `(year, surface)` pair, then passes them all to `generate_manifest()`. For a 2-year run with 2 surfaces, `all_results` has 4 entries, but the manifest only keeps 2 (the last year's).

The downstream Phase 46 consumer reads this manifest to decide whether to use the conservative variant. If turf-2024 passed gates but turf-2025 did not, the manifest would incorrectly show turf-2025's failed metrics as the sole turf entry.

**Fix:** Restructure `per_surface` to be keyed by year or aggregate across years:

```python
# Option A: per-year-per-surface
per_year_surface: dict[str, dict[str, dict]] = {}
for result in retrain_results:
    # Include year in key (requires adding year to ConservativeRetrainResult)
    ...

# Option B: aggregate across years (current intent seems to be this)
# Deduplicate: keep only one result per surface (e.g., last or best)
seen_surfaces: set[str] = set()
for result in reversed(retrain_results):
    if result.surface not in seen_surfaces:
        seen_surfaces.add(result.surface)
        per_surface[result.surface] = { ... }
```

Alternatively, restructure `run_full_pipeline` so it only calls `generate_manifest` with one result per surface (the one intended for the manifest).

### CR-02: CLI script re-runs the entire retraining pipeline a second time

**File:** `scripts/run_mawc_conservative_retrain.py:128-149`
**Issue:** After `run_full_pipeline()` completes (which already trains all models, creates variant directories, and generates the manifest), the CLI script re-invokes `prepare_oof_data()` + `run_retrain()` for every `(year, surface)` pair. This doubles the computation time and, more critically, can produce **different** `ConservativeRetrainResult` objects than what was used during the first run. LogisticRegression fitting is deterministic given the same data, but if any prior step introduces non-determinism (e.g., pandas groupby ordering for edge cases), the quality gate results could differ from what's recorded in the manifest.

The comments at lines 120-126 acknowledge this is a workaround. The root cause is that `run_full_pipeline()` returns only the manifest dict, discarding the `ConservativeRetrainResult` objects that are needed for `_write_retrain_summary()` and the HTML report.

Additionally, for the multi-year case, the CLI only re-runs the last year's surfaces (the `year` loop at line 136 iterates all years, but the results are accumulated into `all_results` which will again have the same duplication issue as CR-01 when passed to `save_retrain_results`).

**Fix:** Modify `run_full_pipeline()` to return both the manifest and the `all_results` list:

```python
def run_full_pipeline(
    self, oof_path: Path, source_model_dir: Path,
    target_root: Path, years: list[int],
) -> tuple[dict, list[ConservativeRetrainResult]]:
    # ... existing logic ...
    all_results: list[ConservativeRetrainResult] = []
    # ... accumulate results ...
    manifest = self.generate_manifest(all_results, source_model_dir, target_root, years)
    return manifest, all_results
```

Then in `main()`:
```python
manifest, all_results = trainer.run_full_pipeline(...)
manifest_path, summary_path = save_retrain_results(manifest, all_results, args.target_root)
```

## Warnings

### WR-01: Class-level `_mawc_helper` is shared mutable state across all instances

**File:** `src/models/mawc_conservative_retrainer.py:184`
**Issue:** `_mawc_helper: ClassVar[MarketAwareWinCalibrator] = MarketAwareWinCalibrator()` creates a single `MarketAwareWinCalibrator` instance shared across all `MawcConservativeRetrainer` instances. While it is only used for its encoding helpers (`_encode_odds_band`, `_encode_pop_bucket`, `_encode_p_rank`), if any code path mutates its `training_summary` or `calibrator` state, it would affect all users. Currently safe because only the encoding methods are called, but this is fragile.

**Fix:** Either document the invariant that `_mawc_helper` must not be mutated, or instantiate it per-call:

```python
def build_conservative_feature_matrix(self, df):
    helper = MarketAwareWinCalibrator()
    # use helper._encode_odds_band etc.
```

### WR-02: `predict_proba` may return single-column array for degenerate training data

**File:** `src/models/mawc_conservative_retrainer.py:512`
**Issue:** `lr.predict_proba(X)[:, 1]` assumes the LogisticRegression has two classes. If the training data has only one class (all winners or all non-winners), `predict_proba` returns a single-column array, and `[:, 1]` will raise an `IndexError`. While this is unlikely with 200+ samples, there is no guard.

**Fix:** Add a check after fitting:

```python
lr.fit(X, y)
if lr.classes_.shape[0] < 2:
    logger.warning("Only one class present for C=%.4f, skipping", c)
    continue
p_conservative = lr.predict_proba(X)[:, 1]
```

### WR-03: `create_conservative_variant` uses `assert` for runtime validation

**File:** `src/models/mawc_conservative_retrainer.py:673-675`
**Issue:** `assert (target_dir / "meta.json").is_file()` is used to verify file existence after copytree. Assertions can be disabled with `python -O`, meaning this safety check silently disappears in optimized mode. This is runtime validation, not an invariant.

**Fix:** Replace with a proper check:

```python
if not (target_dir / "meta.json").is_file():
    raise FileNotFoundError(f"meta.json missing in {target_dir} after copy")
```

### WR-04: `_write_retrain_summary` generates duplicate sections for same surface across years

**File:** `src/models/mawc_conservative_retrainer.py:905-939`
**Issue:** When `retrain_results` contains multiple entries for the same surface (e.g., turf-2024 and turf-2025), the Markdown summary generates duplicate "### turf" subsections under "Quality Gate Details" and "Favorite Band Guard". This produces confusing output with repeated headings. The same issue affects the HTML template at lines 163-244 and 251-310 of `mawc_conservative_report.html`.

**Fix:** Either deduplicate results before writing (keeping one per surface), or include year in the section heading:

```python
lines.append(f"### {result.surface} ({result.manifest_metadata.get('year', 'unknown')})")
```

## Info

### IN-01: Unused `json` import in test file

**File:** `tests/test_mawc_conservative_retrainer.py:10`
**Issue:** `import json` is used in some test methods but is listed at module level even in test classes that don't use it directly. Minor -- not a bug.

**Fix:** No action needed; ruff would flag if truly unused.

### IN-02: HTML template uses inline `style` attribute

**File:** `src/models/templates/mawc_conservative_report.html:255,307`
**Issue:** Lines 255 and 307 use inline `style="color: var(--text-muted); font-style: italic;"` instead of CSS classes. Minor inconsistency with the rest of the template which uses CSS classes.

**Fix:** Add a `.muted-italic` CSS class and apply it consistently:

```css
.muted-italic { color: var(--text-muted); font-style: italic; }
```

---

_Reviewed: 2026-05-31T14:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
