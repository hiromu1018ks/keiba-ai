---
phase: 26-everydb2-new-features
reviewed: 2026-05-15T00:00:00Z
depth: quick
files_reviewed: 13
files_reviewed_list:
  - src/features/mining_features.py
  - src/features/dam_pedigree_features.py
  - src/features/record_features.py
  - src/features/relative_features.py
  - src/features/sire_features.py
  - src/pipelines/training_pipeline.py
  - src/models/stage1_ability_model.py
  - src/models/two_stage_return_model.py
  - tests/test_mining_features.py
  - tests/test_dam_pedigree_features.py
  - tests/test_record_features.py
  - tests/test_relative_features.py
  - tests/test_post_race_leakage.py
findings:
  critical: 2
  warning: 2
  info: 0
  total: 4
  fixed: 4
status: all_fixed
---

# Phase 26: Code Review Report

**Reviewed:** 2026-05-15
**Depth:** quick
**Files Reviewed:** 13
**Status:** all_fixed (4/4 findings resolved)

## Summary

Reviewed 13 files for Phase 26 (EveryDB2 New Features): 5 new feature modules, 2 model files with updated FEATURE_COLS, the training pipeline integration, and 5 test files. Found 2 critical bugs and 2 warnings.

The most severe finding is a copy-paste error in `sire_features.py` line 158 where `"sire_long_wins"` is duplicated instead of `"sire_long_starts"`, causing `sire_distance_wr` for long-distance races (kyori > 1600) to always use NaN for starts (producing incorrect Beta-smoothed values). The second critical finding is a PIT safety violation in `dam_pedigree_features.py` where `groupby("kettonum").last()` picks up the most recent career stats row regardless of the target race date, allowing post-race cumulative data to leak into pre-race feature computation.

## Critical Issues

### CR-01: Copy-paste bug -- sire_long_starts column never populated in compute_batch() [FIXED: a12e57b]

**File:** `src/features/sire_features.py:158`
**Issue:** In the `compute_batch()` method, the column list on line 158 contains `"sire_long_wins"` twice (lines 157-158), and `"sire_long_starts"` is missing entirely. This means `result["sire_long_starts"]` remains NaN for all rows. When computing `sire_distance_wr` for long-distance races (kyori > 1600), the `_beta_smooth_vec` call on line 180 divides by `sire_long_starts` which is NaN-filled, producing incorrect NaN output instead of valid Beta-smoothed win rates. This affects the `sire_distance_wr` feature for every horse in long-distance (steep+) races.

**Fix:**
```python
# Line 158: Change the second "sire_long_wins" to "sire_long_starts"
for col in ["sire_wins", "sire_starts", "sire_places",
            "sire_turf_wins", "sire_turf_starts",
            "sire_dirt_wins", "sire_dirt_starts",
            "sire_short_wins", "sire_short_starts",
            "sire_long_wins", "sire_long_starts",  # <-- fix here
            "sire_prize_total"]:
```

### CR-02: PIT safety violation -- DamPedigreeFeatures uses career stats without race-date filtering [FIXED: 4e68372]

**File:** `src/features/dam_pedigree_features.py:129-132`
**Issue:** The `compute()` method retrieves career stats via `career.sort_values("race_date").groupby("kettonum", observed=True).last()`, which always picks the chronologically latest row for each horse across all of history. When computing features for a past race, this uses cumulative statistics from races that occurred *after* the target race, constituting a look-ahead bias (post-race data leakage). The correct approach is to filter career stats to only include rows where `race_date < target_race_date`, or use the pre-computed PIT `horse_career_stats` rows that correspond to the specific `(kettonum, race_id)` being predicted.

**Fix:**
```python
# Instead of taking .last() across all dates, join on (kettonum, race_id)
# to get the PIT-correct cumulative stats for each specific race entry:
if "race_id" in career.columns:
    career_latest = career.groupby(["kettonum", "race_id"], observed=True).last()
    # Then merge on both kettonum and race_id instead of just kettonum
else:
    career_latest = career.groupby("kettonum", observed=True).last()
```
Note: This requires restructuring the lookup logic to join career stats per `(kettonum, race_id)` pair rather than per `kettonum` alone. The `horse_career_stats` parquet already contains PIT-safe per-row cumulative values keyed by `(race_id, kettonum)`.

## Warnings

### WR-01: race-level merge on ["race_id"] may explode if record_df has duplicate race_ids [FIXED: caa979c]

**File:** `src/pipelines/training_pipeline.py:485`
**Issue:** `df.merge(record_df, on=["race_id"], how="left")` merges `record_df` (which has one row per race_id from `RecordFeatures.compute()`) onto `df` (which has one row per horse). While `RecordFeatures.compute()` does apply `drop_duplicates(subset=keys)`, if the source data has any inconsistency where multiple `course_record_time` values exist for the same `(jyocd, trackcd, kyori)` after the dedup step, this merge would produce a Cartesian product. The risk is mitigated by the dedup in `RecordFeatures`, but the merge key should include a uniqueness assertion or at minimum a comment documenting the assumption.

**Fix:** Add a guard after `record_df` is computed:
```python
record_df = record_feat.compute(df)
# Guard: record_df must have exactly 1 row per race_id
assert record_df["race_id"].is_unique, (
    f"record_df has duplicate race_ids: {record_df['race_id'].duplicated().sum()}"
)
```

### WR-02: DamPedigreeFeatures.compute() silently drops rows when entry_df has duplicate (race_id, umaban) [FIXED: caa979c]

**File:** `src/features/dam_pedigree_features.py:85`
**Issue:** Line 85 does `entry_df.drop_duplicates(subset=["race_id", "umaban"], keep="first")` which modifies the row count of the input DataFrame. The returned DataFrame then has fewer rows than the caller expects. While the docstring does mention "Returns race_id, umaban + FEATURE_COLS", callers (like `training_pipeline.py:469`) merge on `["race_id", "umaban"]` and would silently lose rows. The mining_features module has the same pattern at `compute()` line 143 where it copies `entry_df[["race_id", "umaban"]]` without deduplication, making the two modules inconsistent in how they handle duplicate keys.

**Fix:** Either remove the `drop_duplicates` call (let callers handle dedup) or document the contract clearly:
```python
# Document that callers must ensure (race_id, umaban) uniqueness
# or accept that only the first occurrence is retained.
```

---

_Reviewed: 2026-05-15_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: quick_
