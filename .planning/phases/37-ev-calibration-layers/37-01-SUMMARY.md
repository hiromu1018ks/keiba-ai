---
phase: 37-ev-calibration-layers
plan: 01
subsystem: validation
tags: [oof, health-check, manifest, artifact-validation, fold-column]
dependency_graph:
  requires: []
  provides: [OOFHealthValidator, OOFHealthProfile, ValidationResult, load_validated_oof, ability_oof_fold]
  affects: [src/validation/, src/models/stage1_ability_model.py, tests/]
tech_stack:
  added: []
  patterns: [frozen-dataclass-profile, two-stage-validation, sha256-schema-hash, manifest-first-consumer]
key_files:
  created:
    - src/validation/__init__.py
    - src/validation/oof_health_validator.py
    - tests/test_oof_health_validator.py
  modified:
    - src/models/stage1_ability_model.py
    - tests/test_oof_leakage.py
decisions:
  - OOFHealthValidator as frozen-dataclass-based profile-driven validation class
  - Median-based top1 hit rate/ROI anomaly detection for OOF-03
  - D-04 fail-fast pattern for profile-dependent checks missing metadata
  - SHA256 schema hash from sorted column names (matching freeze_feature_manifest.py pattern)
  - Nullable Int64Dtype for ability_oof_fold column
metrics:
  duration: 414s
  completed: 2026-05-27T06:30:15Z
  tasks_total: 2
  tasks_completed: 2
  tests_added: 32
  tests_passed: 37
  files_created: 3
  files_modified: 2
  loc_added: ~500
---

# Phase 37 Plan 01: OOF Health Validator Summary

OOFHealthValidator基盤 + AbilityModel fold列記録。8つのOOF検査(OOF-01~08) + XCT-05/XCT-08を標準ライブラリのみで実装。37テスト全通過。

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | OOFHealthValidator基盤 + 29テスト | c8a5c80 | src/validation/oof_health_validator.py, tests/test_oof_health_validator.py |
| 2 | AbilityModel.train_oof() fold列追加 | 027cbac | src/models/stage1_ability_model.py, tests/test_oof_leakage.py |

## What Was Built

### Task 1: OOFHealthValidator with artifact profiles

**OOFHealthValidator** class with:
- `validate()` method: 8 health checks (OOF-01~08)
  - OOF-01: Empty DataFrame detection (ValueError)
  - OOF-02: Train/valid overlap check (profile-dependent, D-04 fail-fast)
  - OOF-03: Top1 hit rate/ROI anomaly (profile-dependent)
  - OOF-04: Row coverage threshold
  - OOF-05: Minimum fold count
  - OOF-06: Same race in multiple folds
  - OOF-07: Required columns + fold_col presence (ValueError)
  - OOF-08: Manifest generation
- `generate_manifest()` method: D-10 compliant manifest with XCT-08 fields
- `_compute_schema_hashes()` static method: SHA256 of sorted columns and column:dtype pairs

**OOFHealthProfile** frozen dataclass with per-artifact thresholds and 2 concrete instances:
- `OOF_PREDICTIONS_PROFILE` (ability_oof_fold, p_win_oof score)
- `WIN_SELECTION_OOF_PROFILE` (win_selection_oof_fold, win_market_selection_score)

**ValidationResult** frozen dataclass for immutable results.

**load_validated_oof()** consumer-side function with manifest status check and artifact_hash verification.

**_update_index()** helper for manifest index management.

29 tests covering all checks, determinism (XCT-05), manifest fields (XCT-08), consumer-side verification, and concrete profiles.

### Task 2: AbilityModel.train_oof() fold column

Added `ability_oof_fold` column (pd.Int64Dtype) to train_oof() output:
- Rows with OOF predictions get their 0-based fold index
- Rows without predictions (first fold training period) get pd.NA
- No fold inference/guessing logic (D-06)
- 3 tests verifying column existence, assignment correctness, and nullable integer dtype

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Plan referenced non-existent code patterns**
- **Found during:** Task 1 read_first
- **Issue:** Plan referenced `_validate_win_selection_oof_health()`, `_walk_forward_race_splits()`, `_win_selection_oof_return_unit()`, and module-level constants (lines 62-64) in training_pipeline.py. None of these exist in the current codebase.
- **Fix:** Implemented OOFHealthValidator from scratch using the plan's behavioral specification and the freeze_feature_manifest.py SHA256 pattern as the sole reference. The validate() and generate_manifest() implementations follow D-01 through D-15 decisions.
- **Files modified:** src/validation/oof_health_validator.py
- **Commit:** c8a5c80

**2. [Rule 1 - Bug] Consumer-side test mocking pattern**
- **Found during:** Task 1 test execution
- **Issue:** Initial test mocking for load_validated_oof() used Path mocking that didn't intercept the `open()` call properly. `json.load` returned index data for both calls (index and manifest).
- **Fix:** Restructured tests to mock `builtins.open` and use `side_effect` for `json.load` to return different data for each call.
- **Files modified:** tests/test_oof_health_validator.py
- **Commit:** c8a5c80

**3. [Rule 1 - Bug] Deterministic manifest test failure**
- **Found during:** Task 1 test execution
- **Issue:** `generated_at` timestamp used `datetime.now(timezone.utc)` which changed between calls, making the manifest non-deterministic.
- **Fix:** Patched `datetime` in the test to return a fixed timestamp.
- **Files modified:** tests/test_oof_health_validator.py
- **Commit:** c8a5c80

## Verification Results

- OOFHealthValidator tests: 29/29 passed
- AbilityModel fold tests: 3/3 passed (8/8 total including existing)
- Combined: 37/37 passed
- No new external dependencies introduced
- OOFHealthProfile instances defined for both artifact types
- OOF-02 infrastructure confirms fail-fast when enabled without split_metadata (D-04)

## TDD Gate Compliance

- Task 1: RED commit (tests written first, module did not exist) -> GREEN commit (c8a5c80)
- Task 2: RED verified (3 tests failed with KeyError for ability_oof_fold) -> GREEN commit (027cbac)
- Both tasks followed TDD RED/GREEN cycle as specified by `tdd="true"` attribute

## Self-Check: PASSED

| File | Status |
|------|--------|
| src/validation/__init__.py | FOUND |
| src/validation/oof_health_validator.py | FOUND |
| tests/test_oof_health_validator.py | FOUND |
| src/models/stage1_ability_model.py | FOUND (modified) |
| tests/test_oof_leakage.py | FOUND (modified) |
| Commit c8a5c80 | FOUND |
| Commit 027cbac | FOUND |
