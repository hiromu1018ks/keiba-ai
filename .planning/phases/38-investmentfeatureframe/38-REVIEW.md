---
phase: 38-investmentfeatureframe
reviewed: 2026-05-27T12:00:00Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - src/investment/__init__.py
  - src/investment/cache.py
  - src/investment/feature_frame.py
  - src/investment/leakage.py
  - src/investment/manifest.py
  - src/investment/schema_registry.py
  - tests/test_investment_cache.py
  - tests/test_investment_feature_frame.py
  - tests/test_investment_integration.py
  - tests/test_investment_leakage.py
  - tests/test_investment_manifest.py
  - tests/test_investment_schema.py
findings:
  critical: 2
  warning: 4
  info: 2
  total: 8
status: issues_found
---

# Phase 38: Code Review Report

**Reviewed:** 2026-05-27T12:00:00Z
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Reviewed the new `src/investment/` package (6 source files, 6 test files) implementing the InvestmentFeatureFrame pipeline. The codebase is well-structured with clean separation of concerns: schema registry, feature frame builder, leakage detection, manifest generation, and caching. Tests are thorough and cover the spec requirements.

Two critical bugs were found where feature specs have misleading names that promise transformations (race ranking, within-race standard deviation) but the builder passes raw source values without any computation. Four warnings cover a naming/description mismatch, a fragile test dependency, an unused function parameter, and a test that silently passes without verifying its intended scenario.

## Critical Issues

### CR-01: `if_ability_race_rank` outputs raw probability, not a race rank

**File:** `src/investment/schema_registry.py:443-452`
**Issue:** The spec `if_ability_race_rank` has `description="レース内能力順位 (pct)"` and a name that clearly promises a rank computation within each race. However, `train_sources=("p_ability_win",)` is a raw probability column, and there is no corresponding derived computation in `_compute_derived()` in `feature_frame.py`. The builder's `_resolve_source()` simply copies the raw `p_ability_win` value as `if_ability_race_rank` without any `groupby("race_id").rank()` transformation.

This means downstream consumers expecting a rank (0-1 percentile) receive a raw probability value instead. If any betting strategy uses this column for relative ordering logic (e.g., "rank <= 0.3" thresholds), it will get incorrect results.

The same source column `p_ability_win` is used directly by `if_p_ability` (model_prob category), so the two columns will be identical despite having different semantic names.

**Fix:**
Either (a) change the spec to be derived with empty sources and add a `_compute_derived` handler:
```python
# In schema_registry.py, change to:
InvestmentFeatureSpec(
    name="if_ability_race_rank",
    category="race_relative",
    dtype="float64",
    train_sources=(),  # derived: rank(p_ability_win) in race
    infer_sources=(),
    ...
)

# In feature_frame.py _compute_derived(), add:
if name == "if_ability_race_rank":
    return result.groupby("race_id")["if_p_ability"].rank(
        pct=True, method="min", ascending=False
    )
```
Or (b) change the spec name and description to accurately reflect that it carries a raw probability value, not a rank.

### CR-02: `if_odds_to_ability_dispersion` outputs raw ratio, not race-level dispersion

**File:** `src/investment/schema_registry.py:1218-1229`
**Issue:** The spec `if_odds_to_ability_dispersion` has `description="オッズ/能力比分散 (race内std)"`, promising a within-race standard deviation of the odds-to-ability ratio. However, it uses the same source column as `if_odds_ability_ratio` (model_market_gap category): `train_sources=("odds_to_ability_ratio",)`. There is no derived computation for this spec. The builder passes the raw per-horse ratio value through without any `groupby("race_id").transform("std")`.

If any model or strategy consumes this column expecting a race-level dispersion metric (how spread out the ratio is across the field), it will receive a per-horse ratio instead. This is a silent data corruption issue -- the column name and description say "dispersion" but the data is a point estimate.

**Fix:**
Either change to a derived feature:
```python
# In schema_registry.py:
InvestmentFeatureSpec(
    name="if_odds_to_ability_dispersion",
    category="uncertainty",
    dtype="float64",
    train_sources=(),  # derived: std(odds_to_ability_ratio) in race
    infer_sources=(),
    ...
)

# In feature_frame.py _compute_derived(), add:
if name == "if_odds_to_ability_dispersion":
    return result.groupby("race_id")["if_odds_ability_ratio"].transform("std")
```
Or rename the spec to `if_odds_to_ability_ratio_duplicate` and fix the description to match the actual output.

## Warnings

### WR-01: `if_odds_band_id` passes raw odds value as band ID

**File:** `src/investment/schema_registry.py:567-577`
**Issue:** The spec `if_odds_band_id` has `description="オッズ帯ID"` and sources `("tanodds",)`. The builder passes the raw odds value directly as `if_odds_band_id` without any banding transformation (e.g., binning odds into discrete buckets like 1.0-1.5, 1.5-2.0, etc.). If the downstream system expects discrete band identifiers, it will receive continuous odds values instead.

This may be intentional if the ML model can handle continuous values as a band proxy, but the name and description are misleading. At minimum, the description should clarify whether this is binned or raw.

**Fix:** Either add a banding computation in `_compute_derived` or update the description to `"単勝オッズ (バンドID未変換)"`.

### WR-02: `load_or_compute` caches on input schema hash, not output schema hash

**File:** `src/investment/cache.py:150-167`
**Issue:** `load_or_compute` computes `schema_hash` from the *input* DataFrame's columns via `compute_investment_schema_hash(df)`. This means two different DataFrames with the same columns but different data values will share the same cache key. This is a semantic choice that may cause stale cache hits when the source data changes but the column structure stays the same.

The `source_artifact_hash` parameter is intended to guard against this, but the caller must ensure it changes when source data changes. If a caller forgets to update `source_artifact_hash` after data updates, stale cached results will be returned. This is a correctness risk that should be documented more explicitly.

**Fix:** Add a docstring note warning that `source_artifact_hash` must change whenever source data content changes, or alternatively hash the input DataFrame's content (not just its schema).

### WR-03: `builder_version` parameter accepted but unused

**File:** `src/investment/feature_frame.py:226-227`
**Issue:** The `build_frame` method accepts `builder_version: str = BUILDERS_VERSION` as a parameter but never uses it inside the method body. The parameter exists in the signature and is passed through from `build_train_frame`/`build_inference_frame`, suggesting it was intended for cache keying or version validation but was never wired up.

This is dead code in the API surface that could confuse callers.

**Fix:** Either use the parameter (e.g., validate it matches `BUILDERS_VERSION` or pass it to a cache layer) or remove it from the method signature.

### WR-04: Integration test `test_optional_missing_produces_nan_with_indicator` may silently pass without verification

**File:** `tests/test_investment_integration.py:222-258`
**Issue:** The test iterates over `FEATURE_SPECS` looking for an optional spec whose sources are not in the required set. If no such spec exists (e.g., all optional specs happen to have sources that overlap with required sources), the test will execute zero assertions and pass silently without verifying anything. The `return` at line 258 exits after the first match, but there is no assertion that at least one match was found.

**Fix:** Add a guard assertion at the end of the test:
```python
# After the for loop, add:
pytest.fail("No optional spec with non-required sources found to verify")
```

## Info

### IN-01: `_compute_schema_dtype_hash` is private but could be useful for validation

**File:** `src/investment/manifest.py:39-49`
**Issue:** `_compute_schema_dtype_hash` is a private function that computes a hash over column-dtype pairs. It is called by `generate_investment_manifest` but not exposed in `__init__.py`. If downstream validation needs to verify dtype consistency (e.g., in `validate_schema_identity`), this function would need to be duplicated or made public.

**Fix:** Consider exporting it in `__init__.py` if dtype hashing is needed for validation elsewhere.

### IN-02: `FEATURE_SPECS` iteration order depends on insertion order

**File:** `src/investment/schema_registry.py:1236-1249`
**Issue:** `FEATURE_SPECS` is a dict comprehension from a tuple of specs spread across 9 category tuples. The iteration order is insertion order, which happens to be category order because the tuples are concatenated in `CATEGORY_ORDER` sequence. This works correctly in Python 3.7+ but the implicit dependency on insertion order could break if someone reorders the tuple concatenation.

**Fix:** This is fine for Python 3.11 (the project's target). No action needed unless backward compatibility with older Python is required.

---

_Reviewed: 2026-05-27T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
