---
phase: 24-feature-audit-pruning
plan: 01
subsystem: features
tags: [tier-classification, cache-invalidation, cli]
dependency_graph:
  requires: [23-02 (audit infrastructure)]
  provides: [classify_feature_tiers, compute_code_hash, --tier-report CLI]
  affects: [win_feature_analysis, feature_engine, analyze_feature_importance]
tech_stack:
  added: []
  patterns: [code-hash cache invalidation, tier classification, percentile-based thresholding]
key_files:
  created:
    - tests/test_tier_report_cli.py
  modified:
    - src/features/win_feature_analysis.py
    - src/features/feature_engine.py
    - scripts/analyze_feature_importance.py
    - tests/test_win_feature_analysis.py
    - tests/test_feature_engine.py
decisions:
  - Tier 2 threshold uses percentile-based bottom 10% gain (adaptive across models)
  - Tier 1 for NaN-perm models (return/E-correction) uses gain=0 only
  - Stale cache deletion removes all feat_*.parquet files except current key
  - _run_tier_report() called within _run_all_models() flow for code reuse
metrics:
  duration_minutes: 19
  completed: "2026-05-12"
  tasks_completed: 3
  tests_added: 15
  files_modified: 5
  files_created: 1
  total_tests_passing: 1411
---

# Phase 24 Plan 01: Tier Classification + Cache Invalidation Summary

Tier 1/2 feature classification logic with code-hash cache invalidation and stale cache auto-deletion. CLI `--tier-report` flag generates structured JSON audit reports.

## Changes Made

### Task 1: classify_feature_tiers() (commit 58ca95a)
- Added `classify_feature_tiers()` to `src/features/win_feature_analysis.py`
- Tier 1: gain=0 AND (perm<=0 OR perm is NaN) -- definite noise
- Tier 2: gain>0 AND gain <= np.percentile(nonzero_gains, 10) -- low importance flag
- Per-model classification with no Tier 1/2 overlap
- 5 tests in TestClassifyFeatureTiers

### Task 2: Code hash cache invalidation + auto-deletion (commit c01af0b)
- Added `compute_code_hash()` to `src/features/feature_engine.py` -- hashes all .py files in src/features/
- Extended `compute_cache_key()` with `code_hash` parameter (backward compatible, None = empty string)
- Added `_cleanup_stale_cache()` method to FeatureEngine -- deletes old feat_*.parquet files before new cache write
- `build_all()` now includes code hash in cache key computation
- 6 tests in TestCodeHash + TestCleanupStaleCache

### Task 3: --tier-report CLI flag (commit e507d5d)
- Added `--tier-report` flag to `scripts/analyze_feature_importance.py`
- `--tier-report` auto-enables `--all-models` mode
- Added `--tier-output` for custom output path (default: data/audit/tier_report.json)
- Output JSON includes timestamp, per-model tier1/tier2 lists, counts, total features, tier definitions
- `_run_tier_report()` function generates report from classify_feature_tiers() output
- 4 tests in TestTierReportCLI

## Test Results

- 1411 passed, 1 skipped, 0 failures
- 15 new tests added (5 + 6 + 4)
- ruff check: all modified files pass

## Key Decisions

1. **Tier 2 percentile-based**: Bottom 10% of non-zero gains, adaptive to each model's feature count
2. **NaN perm handling**: Return/E-correction models lack permutation targets; Tier 1 uses gain=0 only
3. **Stale cache strategy**: Delete-all-except-current before writing new cache; prevents disk accumulation
4. **CLI integration**: --tier-report embedded in _run_all_models() flow for code reuse

## Self-Check: PASSED

All 7 files verified present. All 3 commits verified in git log. 1411 tests passing.

## Deviations from Plan

None - plan executed exactly as written.

## Commits

| Commit | Description |
|--------|-------------|
| 58ca95a | feat(24-01): Tier分類ロジック classify_feature_tiers() を実装 |
| c01af0b | feat(24-01): コードハッシュキャッシュ無効化 + 自動削除を実装 |
| e507d5d | feat(24-01): --tier-report CLI flag でTier 1/2分類レポートをJSON出力 |
