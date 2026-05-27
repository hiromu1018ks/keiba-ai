---
phase: 37-ev-calibration-layers
verified: 2026-05-27T16:45:00Z
status: passed
score: 11/11 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 37: OOF Health Infrastructure Verification Report

**Phase Goal:** 全OOF成果物が健全性検査を通過し、下流コンポーネント(キャリブレータ・ランカー)が信頼できるOOF予測を利用できる状態になる
**Verified:** 2026-05-27T16:45:00Z
**Status:** PASSED
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | OOFHealthValidator.validate() raises ValueError on empty DataFrame (OOF-01) | VERIFIED | `oof_health_validator.py:97-98`: `if df.empty: raise ValueError("OOF artifact is empty (OOF-01)")`. Test `TestOOF01Empty::test_empty_df_raises_value_error` passes. |
| 2 | OOFHealthValidator.validate() fails when fold_count < 3 (OOF-05) | VERIFIED | `oof_health_validator.py:109-114`: fold_count < min_fold_count appends failure. Test `TestOOF05MinFoldCount::test_fold_count_below_minimum` passes. |
| 3 | OOFHealthValidator.validate() fails when same race_id appears in multiple folds (OOF-06) | VERIFIED | `oof_health_validator.py:117-123`: groupby race_id fold_col nunique check. Test `TestOOF06SameRaceMultipleFold::test_race_in_multiple_folds` passes. |
| 4 | OOFHealthValidator.validate() fails when required columns or fold_col are missing (OOF-07) | VERIFIED | `oof_health_validator.py:101-106`: checks required_columns + fold_col presence. Tests `TestOOF07RequiredColumns` (2 tests) pass. |
| 5 | OOFHealthValidator.validate() fails when row_coverage_ratio < 0.70 threshold (OOF-04) | VERIFIED | `oof_health_validator.py:126-133`: coverage check with threshold. Test `TestOOF04RowCoverage::test_low_coverage_returns_fail` passes. |
| 6 | OOFHealthValidator.validate() detects top1 hit rate/ROI anomalies (OOF-03) | VERIFIED | `oof_health_validator.py:136-218`: `_check_top1_anomaly` with profile-dependent activation. Test `TestOOF03Top1Anomaly::test_high_hit_rate_fails` passes. |
| 7 | OOFHealthValidator OOF-02 infrastructure exists: fail-fast when enable_train_valid_overlap=True and split_metadata=None (D-04) | VERIFIED | `oof_health_validator.py:152-160`: raises ValueError with "D-04 fail-fast". Test `TestOOF02TrainValidOverlap::test_enabled_without_metadata_raises` passes. |
| 8 | generate_manifest() produces deterministic JSON with D-10 fields plus XCT-08 fields (OOF-08, XCT-08) | VERIFIED | `oof_health_validator.py:237-320`: manifest generation with all fields. Tests `TestXCT05DeterministicManifest`, `TestXCT08ManifestFields`, `TestManifestContent` all pass. |
| 9 | schema_hash and schema_dtype_hash are SHA256 of sorted column names / column:dtype pairs (XCT-05) | VERIFIED | `oof_health_validator.py:322-335`: `_compute_schema_hashes` using hashlib.sha256. Test `test_schema_hash_order_independent` passes. |
| 10 | AbilityModel.train_oof() records ability_oof_fold column on output DataFrame (OOF-07, D-05) | VERIFIED | `stage1_ability_model.py:363`: `oof_folds = pd.Series(pd.NA, ...)` init, line 396: `oof_folds.loc[test_mask] = i`, line 403: `df["ability_oof_fold"] = oof_folds`. Fallback path line 372: `df["ability_oof_fold"] = pd.NA` (CR-02 fix applied). |
| 11 | ability_oof_fold column verified by automated tests: fold assignment correctness, NA for non-validation rows | VERIFIED | `test_oof_leakage.py:TestAbilityOofFold` (3 tests): column existence, assignment correctness with prediction alignment, nullable integer dtype. All pass. |

**Score:** 11/11 truths verified

### ROADMAP Success Criteria Gap Analysis

ROADMAP Phase 37 defines CAL-01~05 (Pop band calibration, regime_state propagation). The actual Phase 37 was executed as **OOF Health Infrastructure** with requirements OOF-01~08, XCT-05, XCT-08. This is an explicit scope change documented in:
- RESEARCH.md (created 2026-05-27, focused on OOF validation)
- PLAN frontmatter (requirements: OOF-01~08, XCT-05, XCT-08)

The ROADMAP CAL-01~05 requirements remain in REQUIREMENTS.md as `Pending` and are NOT addressed by this phase. They should be addressed in a future phase or the ROADMAP should be updated to reflect the scope change.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/validation/__init__.py` | Package init | VERIFIED | 2 lines, proper docstring |
| `src/validation/oof_health_validator.py` | OOFHealthValidator, OOFHealthProfile, ValidationResult, load_validated_oof, _update_index | VERIFIED | 410 lines. All 6 exports present: OOFHealthValidator, OOFHealthProfile, ValidationResult, OOF_PREDICTIONS_PROFILE, WIN_SELECTION_OOF_PROFILE, load_validated_oof |
| `src/models/stage1_ability_model.py` | Modified train_oof() with ability_oof_fold | VERIFIED | Lines 363, 396, 403 in train_oof(). Fallback path line 372 also sets fold column (CR-02 fix) |
| `tests/test_oof_health_validator.py` | Comprehensive unit tests | VERIFIED | 29 tests covering all checks, determinism, manifest fields, consumer-side, concrete profiles |
| `tests/test_oof_leakage.py` | Extended fold column tests | VERIFIED | 3 new tests in TestAbilityOofFold class (8 total) |
| `src/pipelines/training_pipeline.py` | OOFHealthValidator wiring, ev_oof_fold | VERIFIED | Import at line 54-58. Validation at save point lines 287-316. ev_oof_fold at line 912. generate_ev_oof_predictions 4-tuple at line 906 |
| `tests/test_training_pipeline.py` | Updated mocks, D-13 tests | VERIFIED | 24/24 tests pass. TestOOFHealthValidatorIntegration with 2 new D-13 fail-fast tests |
| `tests/test_ev_isotonic.py` | 4-tuple return update | VERIFIED | 18/18 tests pass. test_generate_ev_oof_returns_four_arrays renamed from three |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/pipelines/training_pipeline.py` | `src/validation/oof_health_validator.py` | `from validation.oof_health_validator import OOFHealthValidator, OOF_PREDICTIONS_PROFILE, _update_index` | WIRED | Import at lines 54-58. Usage at lines 287, 304, 316 |
| `src/models/stage1_ability_model.py` | `src/validation/oof_health_validator.py` | `ability_oof_fold` column consumed by OOFHealthProfile.fold_col check | WIRED | OOF_PREDICTIONS_PROFILE.fold_col="ability_oof_fold" matches stage1_ability_model.py line 403 column name |
| `src/pipelines/training_pipeline.py` | `src/models/stage1_ability_model.py` | `train_oof()` returns ability_oof_fold, consumed by pipeline | WIRED | Pipeline calls train_oof() which produces ability_oof_fold. Validation at line 290 uses OOF_PREDICTIONS_PROFILE |
| `src/pipelines/training_pipeline.py` | generate_ev_oof_predictions() | 4-tuple return with ev_fold_full | WIRED | Line 906: destructured to 4 values. Line 912: `df_oof["ev_oof_fold"] = pd.array(ev_fold_full, dtype=pd.Int64Dtype())` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `oof_health_validator.py:validate()` | failures list | DataFrame health checks | Real computed metrics (fold_count, coverage, hit_rate) | FLOWING |
| `oof_health_validator.py:generate_manifest()` | manifest dict | validate() result + DataFrame | Real schema hashes, fold counts, status | FLOWING |
| `stage1_ability_model.py:train_oof()` | ability_oof_fold | Fold loop index `i` | Real fold indices (0-based) for validation rows | FLOWING |
| `training_pipeline.py:906-912` | ev_oof_fold | generate_ev_oof_predictions() 4th return | Real fold assignments from KFold | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| OOF Health Validator tests | `python -m pytest tests/test_oof_health_validator.py -v` | 29/29 passed | PASS |
| OOF Leakage tests | `python -m pytest tests/test_oof_leakage.py -v` | 8/8 passed | PASS |
| Training Pipeline tests | `python -m pytest tests/test_training_pipeline.py -v` | 24/24 passed | PASS |
| EV Isotonic tests | `python -m pytest tests/test_ev_isotonic.py -v` | 18/18 passed | PASS |
| Full test suite | `python -m pytest tests/ --tb=short` | 1928 passed, 1 skipped | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| N/A | N/A | No probes defined for this phase | SKIP |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| OOF-01 | 37-01 | Empty DataFrame detection | SATISFIED | ValueError raised on df.empty (line 97-98) |
| OOF-02 | 37-01 | Train/valid overlap check | SATISFIED | Profile-dependent with D-04 fail-fast (lines 152-160) |
| OOF-03 | 37-01 | Top1 hit rate/ROI anomaly | SATISFIED | _check_top1_anomaly with min_guard_races guard (lines 170-218) |
| OOF-04 | 37-01 | Row coverage threshold | SATISFIED | Coverage check with 0.70 default (lines 126-133) |
| OOF-05 | 37-01 | Minimum fold count | SATISFIED | fold_count < min_fold_count check (lines 109-114) |
| OOF-06 | 37-01 | Same race in multiple folds | SATISFIED | groupby race_id fold_col nunique (lines 117-123) |
| OOF-07 | 37-01 | Required columns + fold_col | SATISFIED | Missing column detection (lines 101-106) |
| OOF-08 | 37-01 | Manifest generation | SATISFIED | generate_manifest() with all D-10 fields (lines 237-320) |
| XCT-05 | 37-01 | Deterministic JSON output | SATISFIED | SHA256 schema hash, sort_keys=True (lines 322-335) |
| XCT-08 | 37-01 | Manifest field requirements | SATISFIED | artifact_version, schema_hash, source_oof_manifest_path, train_date_range present |

**Orphaned ROADMAP requirements:** CAL-01, CAL-02, CAL-03, CAL-04, CAL-05 are mapped to Phase 37 in REQUIREMENTS.md but are NOT addressed by any plan. Phase 37 scope was changed to OOF Health Infrastructure. These requirements remain `Pending` and should be re-planned.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | - | - | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER found in any modified file |

### Code Review Status

REVIEW.md identified 8 findings (2 critical, 4 warning, 2 info). Commit `cb081ea` addressed:
- **CR-01**: Pipeline validation crash on missing columns -- FIXED: `is_oof` and `oof_artifact_version` columns added before validation (lines 279-280)
- **CR-02**: AbilityModel.train_oof() fallback missing fold column -- FIXED: `df["ability_oof_fold"] = pd.NA` added at line 372
- **WR-02**: Zero-division risk in _check_top1_anomaly -- FIXED: `top1_idx.dropna()` guard at line 181

Remaining REVIEW findings (warnings/info) do not block goal achievement:
- WR-01 (path traversal): Internal tool, low risk
- WR-03 (vacuous coverage): known limitation, tracked
- WR-04 (redundant validation): acceptable overhead
- IN-01 (NA fold key in manifest): cosmetic
- IN-02 (_update_index naming): cosmetic

### Human Verification Required

No items requiring human verification. All must-haves are verified programmatically through automated tests.

### Gaps Summary

No gaps blocking the Phase 37 OOF Health Infrastructure goal. All 11 must-haves are verified with passing tests and substantive implementations.

**ROADMAP scope note:** The ROADMAP's original Phase 37 goal (CAL-01~05: Pop band calibration + regime_state propagation) was NOT implemented. Phase 37 was re-scoped to OOF Health Infrastructure during execution. CAL-01~05 requirements remain pending and should be addressed in a future phase.

---

_Verified: 2026-05-27T16:45:00Z_
_Verifier: Claude (gsd-verifier)_
