---
phase: "34"
plan: "04"
subsystem: validation
tags: [manifest, freeze, v1.7, rl-features, MCF]
dependency_graph:
  requires: [34-01, 34-02, 34-03]
  provides: [VAL-04, feature-freeze-v1.7]
  affects: [freeze_feature_manifest.py, feature_freeze_manifest.json]
tech-stack:
  added: []
  patterns: [deterministic-json-serialization, sha256-feature-hashing]
key-files:
  created: []
  modified:
    - scripts/freeze_feature_manifest.py
    - data/feature_freeze_manifest.json
decisions:
  - D-10: Manifest freeze proceeds regardless of validation results (BT/GPD failed, IC eval passed)
metrics:
  duration: ~2min
  completed: "2026-05-19"
---

# Phase 34 Plan 04: Manifest Freeze v1.7 Summary

Manifest frozen at v1.7 with rl_* race-level features and MCF features registered across all 12 models.

## What Was Done

Updated the feature freeze manifest version from v1.6 to v1.7 and executed the freeze script. All 12 models now include the 6 rl_* features (rl_log_odds_entropy, rl_odds_dispersion, rl_top3_odds_gap, rl_top1_odds, rl_favorite_rank_gap, rl_n_horses) and 5 MCF features (rl_favorite_in_wide_top1, rl_trio_overlap, rl_market_consistency, rl_trio_odds_ratio, rl_wide_harville_ratio).

## Manifest Summary

- **Version:** v1.7
- **Models:** 12
- **Overall SHA256:** e08c25509769fc3237dacb00c4d60329...

### Model Feature Counts

| Model | Features |
|-------|----------|
| AbilityModel | 108 |
| WinTwoStageModel | 93 |
| PlaceTwoStageModel.HIT | 96 |
| PlaceTwoStageModel.RETURN | 98 |
| EVCorrectionModel | 36 |
| PlaceEVCorrectionModel | 36 |
| ConformalEVModel | 142 |
| RegimeDetector | 21 |
| MarketModel | 20 |
| PlaceAbilityModel | 74 |
| RaceQualityScreener | 35 |
| WideTwoStageModel.SHARED | 18 |

### New Features Verified (11 total, present in all 12 models)

- rl_log_odds_entropy: 12/12
- rl_odds_dispersion: 12/12
- rl_top3_odds_gap: 12/12
- rl_top1_odds: 12/12
- rl_favorite_rank_gap: 12/12
- rl_n_horses: 12/12
- rl_favorite_in_wide_top1: 12/12
- rl_trio_overlap: 12/12
- rl_market_consistency: 12/12
- rl_trio_odds_ratio: 12/12
- rl_wide_harville_ratio: 12/12

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

- **D-10:** Per plan instructions, manifest freeze proceeds regardless of validation results from Plans 34-01/34-02/34-03 (BT and GPD diagnostics failed due to fixture mismatches; IC evaluation passed successfully).

## Task Completion

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Update manifest version to v1.7 and freeze | 0ade846 | scripts/freeze_feature_manifest.py, data/feature_freeze_manifest.json |

## Self-Check

- PASSED
  - scripts/freeze_feature_manifest.py: FOUND (version string confirmed "v1.7")
  - data/feature_freeze_manifest.json: FOUND (version "v1.7", 12 models, overall_sha256 populated)
  - Commit 0ade846: FOUND
