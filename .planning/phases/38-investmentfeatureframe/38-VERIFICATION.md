---
phase: 38-investmentfeatureframe
verified: 2026-05-27T12:00:00Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
---

# Phase 38: InvestmentFeatureFrame Verification Report

**Phase Goal:** Invest judgment integrated feature frame (90-130 columns). Integrate model output, market data, and OOF predictions to generate structured features specialized for investment judgment.
**Verified:** 2026-05-27T12:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | InvestmentFeatureFrameBuilder.build_frame(df, mode="train" or "infer") generates features in 9 categories and 90-130 columns (IFF-01) | VERIFIED | `feature_frame.py` lines 221-336: `build_frame()` resolves all 94 specs from FEATURE_SPECS. Test `test_investment_integration.py::TestEndToEnd::test_column_count_in_range` confirms 90-130 range. Test `test_investment_integration.py::TestDeterminismAndSchema::test_all_9_categories_in_output` confirms all 9 categories present. |
| 2 | Train mode uses only OOF-safe columns, rejects in-sample columns (p_win_pred etc.). Infer mode uses production columns (IFF-02) | VERIFIED | `schema_registry.py`: all specs use `train_sources=("p_win_oof",...)` not `("p_win_pred",...)`. `leakage.py` line 17-23: `IN_SAMPLE_ONLY_COLS` frozenset. Test `test_investment_integration.py::TestEndToEnd::test_train_mode_zero_in_sample_only_columns` confirms zero in-sample columns in output. Test `test_investment_integration.py::TestLeakageAudit::test_all_specs_oof_safe` confirms zero violations. |
| 3 | Train/infer output schema identical (same column names, order, dtypes). Test asserts identity (IFF-03) | VERIFIED | `test_investment_feature_frame.py::TestBuildFrameInferMode::test_produces_identical_schema_as_train` and `test_identical_dtypes_as_train` both pass. `test_investment_integration.py::TestDeterminismAndSchema::test_train_infer_produce_same_schema_hash` passes. `validate_schema_identity()` in `leakage.py` lines 68-99 asserts column list AND per-column dtype equality. |
| 4 | InvestmentFeatureSpec frozen dataclass schema registry with metadata for all features (IFF-04) | VERIFIED | `schema_registry.py` lines 15-47: 10-field frozen dataclass with `__post_init__` validation. 94 specs in FEATURE_SPECS dict. Test `test_investment_schema.py::TestSpecMetadata::test_all_specs_have_10_fields` confirms field set. Test `TestFeatureSpecsCategories` confirms 9 categories with D-05 range counts. |
| 5 | No POST_RACE column in InvestmentFeatureFrame output. Leakage test passes (IFF-05, VAL-01) | VERIFIED | `feature_frame.py` line 311: `validate_no_post_race_leakage(result.columns.tolist())` called during build. `leakage.py` lines 26-42: validates no overlap with POST_RACE_COLS. Test `test_investment_integration.py::TestLeakageAudit` runs 5 separate audits: OOF-safe sources, spec names vs POST_RACE, train_sources vs POST_RACE, infer_sources vs POST_RACE, leakage_class validation -- all pass. |
| 6 | Parquet cache + sidecar manifest. Deterministic output: same input + same builder_version = same output (IFF-06) | VERIFIED | `cache.py`: `InvestmentFrameCache` with `save()` creating parquet + sidecar JSON (lines 97-122), `load_cached()` with schema_hash verification (lines 67-95), `load_or_compute()` (lines 124-177). Test `test_investment_cache.py`: all 11 tests pass including cache hit/miss/determinism/corrupted manifest. Test `test_investment_feature_frame.py::TestDeterminism::test_same_input_produces_identical_output` confirms byte-identical output via `pd.testing.assert_frame_equal`. |
| 7 | Manifest contains all D-30 required fields (IFF-07) | VERIFIED | `manifest.py` lines 52-91: `generate_investment_manifest()` returns dict with all 11 D-30 keys: artifact_name, builder_version, feature_version, generated_at, mode, row_count, schema_hash, schema_dtype_hash, source_artifact_hash, source_oof_manifest_path, column_count. Test `test_investment_manifest.py::TestGenerateInvestmentManifest` has 10 tests covering all fields. |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/investment/__init__.py` | Package init with public API exports | VERIFIED | Exports all 12 public symbols: InvestmentFeatureSpec, FEATURE_SPECS, CATEGORY_ORDER, ALL_IF_COLUMNS, validate_no_post_race_leakage, validate_oof_safe_sources, validate_schema_identity, IN_SAMPLE_ONLY_COLS, InvestmentFeatureFrameBuilder, build_frame, compute_investment_schema_hash, generate_investment_manifest, InvestmentFrameCache |
| `src/investment/schema_registry.py` | InvestmentFeatureSpec + FEATURE_SPECS + CATEGORY_ORDER + ALL_IF_COLUMNS | VERIFIED | 1263 lines. 94 specs across 9 categories. Frozen dataclass with 10 fields. `__post_init__` validates "if_" prefix. CATEGORY_ORDER tuple with 9 categories. ALL_IF_COLUMNS built from specs. |
| `src/investment/leakage.py` | 3 validation functions + IN_SAMPLE_ONLY_COLS | VERIFIED | 100 lines. `validate_no_post_race_leakage()`, `validate_oof_safe_sources()`, `validate_schema_identity()`, `IN_SAMPLE_ONLY_COLS` frozenset. All with type annotations. Imports POST_RACE_COLS from domain.types. |
| `src/investment/manifest.py` | Manifest generation + schema hash computation | VERIFIED | 92 lines. `compute_investment_schema_hash()` with SHA256 deterministic hashing. `generate_investment_manifest()` with all D-30 fields. ISO 8601 generated_at. |
| `src/investment/cache.py` | InvestmentFrameCache with Parquet + sidecar JSON | VERIFIED | 178 lines. Deterministic cache key via SHA256. `load_cached()` with schema_hash verification. `save()` creates parquet + JSON. `load_or_compute()` convenience method. |
| `src/investment/feature_frame.py` | InvestmentFeatureFrameBuilder + build_frame | VERIFIED | 370 lines. `BUILDERS_VERSION = "1.0.0"`. Two-pass builder (source resolution then derived features). 20 derived feature computations. Convenience wrappers `build_train_frame()`, `build_inference_frame()`. Module-level `build_frame()`. No FeatureEngine import (D-35). |
| `tests/test_investment_schema.py` | Schema registry tests | VERIFIED | 208 lines, 15 tests. Covers frozen enforcement, categories, unique names, column counts, metadata, init exports. |
| `tests/test_investment_leakage.py` | Leakage detection tests | VERIFIED | 124 lines, 9 tests. Covers POST_RACE detection, OOF-safe validation, schema identity, IN_SAMPLE_ONLY_COLS. |
| `tests/test_investment_feature_frame.py` | Builder integration tests | VERIFIED | 335 lines, 17 tests. Covers dual mode, schema identity, validation, optional/required, derived features, column order, wrappers, leakage, determinism. |
| `tests/test_investment_manifest.py` | Manifest generation tests | VERIFIED | 250 lines, 15 tests. Covers deterministic hash, D-30 fields, ISO 8601, schema hash consistency. |
| `tests/test_investment_cache.py` | Cache tests | VERIFIED | 292 lines, 11 tests. Covers cache key determinism, hit/miss, save format, corrupted manifest, load_or_compute. |
| `tests/test_investment_integration.py` | End-to-end integration tests | VERIFIED | 382 lines, 18 tests. Covers full pipeline, 5-dimension leakage audit, required/optional behavior, column counts, determinism. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `leakage.py` | `domain/types.py` | `import POST_RACE_COLS` | WIRED | Line 13: `from domain.types import POST_RACE_COLS` |
| `feature_frame.py` | `schema_registry.py` | `import FEATURE_SPECS, CATEGORY_ORDER` | WIRED | Lines 16-20: `from investment.schema_registry import CATEGORY_ORDER, FEATURE_SPECS, InvestmentFeatureSpec` |
| `feature_frame.py` | `leakage.py` | `import validate_no_post_race_leakage` | WIRED | Line 15: `from investment.leakage import validate_no_post_race_leakage` |
| `cache.py` | `manifest.py` | `import compute_investment_schema_hash` | WIRED | Line 154: `from investment.manifest import compute_investment_schema_hash` |
| `__init__.py` | All submodules | `import` statements | WIRED | Lines 7-27: imports from cache, feature_frame, leakage, manifest, schema_registry |
| `test_investment_feature_frame.py` | `feature_frame.py` | `import InvestmentFeatureFrameBuilder` | WIRED | Line 13: `from investment.feature_frame import InvestmentFeatureFrameBuilder, build_frame` |
| `test_investment_leakage.py` | `domain.types` | `import POST_RACE_COLS` | WIRED | Implicit via `investment.schema_registry` which imports `InvestmentFeatureSpec` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `feature_frame.py` | `result` DataFrame | FEATURE_SPECS -> source resolution + derived computation | Yes -- real data via dual-mode source columns | FLOWING |
| `manifest.py` | manifest dict | DataFrame columns/dtypes -> SHA256 hash | Yes -- deterministic from real column data | FLOWING |
| `cache.py` | cached parquet + JSON | Input df schema_hash -> cache key -> parquet I/O | Yes -- real parquet write with schema verification | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 85 investment tests pass | `python -m pytest tests/test_investment_*.py -v` | 85 passed in 2.02s | PASS |
| Spec count in 90-130 range | pytest `test_feature_specs_count_in_range` | PASSED (94 specs) | PASS |
| Train/infer schema identity | pytest `test_produces_identical_schema_as_train` + `test_train_infer_produce_same_schema_hash` | PASSED | PASS |
| No POST_RACE leakage | pytest `test_all_specs_oof_safe` + `test_no_post_race_columns_in_output` | PASSED | PASS |

### Probe Execution

Step 7c: SKIPPED (no probe scripts defined for this phase)

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| IFF-01 | Plan 02, 03 | InvestmentFeatureFrameBuilder.build_frame generates 9-category 90-130 columns | SATISFIED | 94 specs, 9 categories, build_frame() implemented, tests pass |
| IFF-02 | Plan 02, 03 | Train mode OOF-safe only, infer mode production sources | SATISFIED | train_sources use OOF columns, IN_SAMPLE_ONLY_COLS defined, zero violations in audit |
| IFF-03 | Plan 02, 03 | Train/infer identical schema (names, order, dtypes) | SATISFIED | validate_schema_identity() asserts column list + dtype equality, tests confirm |
| IFF-04 | Plan 01, 03 | InvestmentFeatureSpec frozen dataclass schema registry | SATISFIED | 10-field frozen dataclass, 94 specs, CATEGORY_ORDER, ALL_IF_COLUMNS |
| IFF-05 | Plan 01, 03 | POST_RACE exclusion, leakage test passes | SATISFIED | 5-dimension leakage audit passes, validate_no_post_race_leakage() called during build |
| IFF-06 | Plan 02, 03 | Parquet cache + sidecar manifest, deterministic output | SATISFIED | InvestmentFrameCache implemented, load_or_compute works, determinism test passes |
| IFF-07 | Plan 02, 03 | Manifest with D-30 fields | SATISFIED | generate_investment_manifest() with 11 required fields, all tests pass |
| VAL-01 | Plan 03 | 3-layer CI leakage test applied to InvestmentFeatureFrame | SATISFIED | 5-dimension leakage audit covers POST_RACE, OOF-safe, schema identity |

Orphaned requirements: None. All 8 requirement IDs (IFF-01 through IFF-07 plus VAL-01) from ROADMAP.md are claimed by plans and verified.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No anti-patterns detected |

No TBD, FIXME, XXX, TODO, HACK, or PLACEHOLDER markers found in any `src/investment/` file.
No empty implementations (`return None`, `return {}`, `return []`) found.
No FeatureEngine import (D-35 compliant).

### Human Verification Required

None. All verification items are covered by automated tests:
- Schema identity: `validate_schema_identity()` + test assertions
- Leakage: 5-dimension automated audit across all 94 specs
- Column counts: range-checked by tests
- Determinism: `pd.testing.assert_frame_equal` comparison
- Cache integrity: parquet + sidecar JSON round-trip tested

---

_Verified: 2026-05-27T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
