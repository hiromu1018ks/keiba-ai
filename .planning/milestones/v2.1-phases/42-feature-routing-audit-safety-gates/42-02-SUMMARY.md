---
phase: 42-feature-routing-audit-safety-gates
plan: 02
subsystem: validation
tags: [saf-02, oof-health, artifact-profiles, tdd]
dependency_graph:
  requires: [oof_health_validator]
  provides: [calibrator_artifact_profile, ranker_artifact_profile, profiles_registry]
  affects: []
tech_stack:
  added: []
  patterns: [plugin-profile, validate-method-returns-failures]
key_files:
  created:
    - src/validation/artifact_profiles.py
    - tests/test_artifact_profiles.py
  modified: []
decisions:
  - Regular class (not frozen dataclass) for profiles since they have validate() behavior methods
  - PROFILES dict as module-level registry for OOFHealthValidator plugin discovery (D-06)
  - Rank determinism check uses duplicated-score detection per race as WARNING (not failure)
metrics:
  duration: 2m
  completed: 2026-05-28
  tasks: 1
  files: 2
---

# Phase 42 Plan 02: OOF Artifact Health Profiles Summary

CalibratorArtifactProfile and RankerArtifactProfile for MAWC/Ranker OOF artifact validation with PROFILES plugin registry (SAF-02).

## What Was Done

Created two artifact profile classes that extend the OOFHealthValidator infrastructure for Phase 39 (MarketAwareWinCalibrator) and Phase 40 (RaceLevelRanker) without modifying the validator core (D-06 plugin pattern).

### CalibratorArtifactProfile (D-07)
- NaN/inf detection in probability columns (p_win_combined, p_win_final)
- [0, 1] range validation for probability columns
- Sum-to-1.0 per race_id check with configurable tolerance (default 1e-6)
- Forbidden column detection (p_win_pred -- train-mode prediction guard)
- Required column enforcement (race_id, p_win_combined, p_win_final, fold)

### RankerArtifactProfile (D-08)
- NaN/inf detection in score columns (investment_score, relevance_score, value_score)
- Race-level rank determinism warning for duplicated investment_scores within a race
- Required column enforcement (race_id, investment_score, fold)

### PROFILES Registry (D-06)
- Module-level `PROFILES: dict[str, type]` mapping profile names to classes
- OOFHealthValidator can import PROFILES for plugin discovery without core modification
- Default convenience instances: `DEFAULT_CALIBRATOR_PROFILE`, `DEFAULT_RANKER_PROFILE`

## TDD Gate Compliance

| Gate | Commit | Hash | Status |
|------|--------|------|--------|
| RED | test(42-02): add failing tests | 69e0389 | Pass |
| GREEN | feat(42-02): implement profiles | 3f43c9e | Pass |

All 19 tests pass. OOFHealthValidator core unchanged (verified by `git diff --exit-code`).

## Commits

| Hash | Message |
|------|---------|
| 69e0389 | test(42-02): add failing tests for OOF artifact profiles (SAF-02) |
| 3f43c9e | feat(42-02): implement CalibratorArtifactProfile and RankerArtifactProfile (SAF-02) |

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None.

## Self-Check

PASSED -- all files and commits verified.
