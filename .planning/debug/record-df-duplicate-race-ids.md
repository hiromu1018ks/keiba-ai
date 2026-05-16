---
status: resolved
trigger: "Phase 28 multi-year backtest: record_df has duplicate race_ids: 82882 despite drop_duplicates fix"
created: "2026-05-16T09:30:00+09:00"
updated: "2026-05-16T10:15:00+09:00"
---

# Debug: record-df-duplicate-race-ids

## Symptoms

- **Expected:** RecordFeatures.compute() returns DataFrame with unique race_ids after drop_duplicates(subset=["race_id"])
- **Actual:** training_pipeline.py:482 assertion fails: `record_df has duplicate race_ids: 82882`
- **Error:** `2023年 学習失敗: record_df has duplicate race_ids: 82882 — スキップ`
- **Timeline:** Occurs during multi-year backtest with --years 2023 2024 2025. Fix commit 795619c added drop_duplicates but issue persists.
- **Reproduction:** `python scripts/run_backtest.py --years 2023 --train-window 4 --ensemble`

## Current Focus

hypothesis: "null"
next_action: "fixed"

## Evidence

- timestamp: 2026-05-16T09:45
  type: code_analysis
  detail: "Commit 795619c added drop_duplicates(subset=['race_id']) at line 164 of record_features.py, but ONLY on the happy-path return. Three early-return paths (lines 98, 111, 118) that fire when record.parquet is missing or empty still return horse-level DataFrame without deduplication."
- timestamp: 2026-05-16T09:50
  type: data_check
  detail: "data/raw/record.parquet does NOT exist on disk. Therefore _load_records() returns empty DataFrame, triggering the early return at line 98. This is the path that produces duplicates."
- timestamp: 2026-05-16T10:00
  type: reproduction
  detail: "Confirmed bug: passing horse-level DataFrame (6 rows, 2 unique race_ids) to compute() with empty store returns 6 rows with 4 duplicate race_ids. Post-fix: returns 2 rows with unique race_ids."
- timestamp: 2026-05-16T10:05
  type: test
  detail: "All 10 existing tests pass. ruff check passes. The fix adds a _nan_result() helper that applies drop_duplicates(subset=['race_id']) to all early-return paths."

## Eliminated

- Data type mismatch in race_id column (all race_ids are consistently computed 16-digit strings)
- Lookup merge producing extra rows (tested with large synthetic data, merge is correct)
- groupby() issues (observed=True already set)

## Resolution

root_cause: "RecordFeatures.compute() has 3 early-return paths (lines 98, 111, 118) that return race_df[['race_id']] with NaN features when record.parquet is missing or empty. Since the caller (_train_submodel) passes a horse-level DataFrame where each row is (race_id, umaban), these early returns produce duplicate race_ids. The fix in commit 795619c only addressed the happy-path return at line 164."
fix: "Extracted a _nan_result() static method that applies drop_duplicates(subset=['race_id']) to the early-return DataFrame. All 3 early-return paths now use this helper. The data/raw/record.parquet file does not exist, so the early return at line 98 is always hit in practice."
files:
  - src/features/record_features.py
