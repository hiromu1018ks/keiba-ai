---
phase: 35-etl-data-foundation
reviewed: 2026-05-19T12:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - scripts/run_paper_trading.py
  - src/db/etl.py
  - src/db/readers.py
  - src/domain/types.py
  - tests/test_etl_type_conversion.py
  - tests/test_paper_trading_guards.py
findings:
  critical: 2
  warning: 4
  info: 2
  total: 8
status: issues_found
---

# Phase 35: Code Review Report

**Reviewed:** 2026-05-19T12:00:00Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

6 files reviewed across the ETL data foundation and paper trading modules. Two critical bugs found: a KeyError crash in dry-run mode where `kakuteijyuni` is accessed after being dropped, and unbounded data duplication in reconcile mode where the entire predictions DataFrame is appended to `bets.parquet` on every run. Several warnings include a data-leakage-prone result sentinel value, nullable Int64 type incompatibility in `_compute_surface`, an ambiguous date argument format in dry-run mode, and `result == 0.0` being unable to distinguish lost bets from unsettled ones.

## Critical Issues

### CR-01: KeyError on `kakuteijyuni` in dry-run mode after POST_RACE drop

**File:** `scripts/run_paper_trading.py:1255-1270`
**Issue:** In `_run_dry_run`, `_drop_post_race_cols()` is called at line 1255, which removes `kakuteijyuni` (it is listed in `POST_RACE_COLS` in `src/domain/types.py:39`). Then at line 1270, `int(horse.iloc[0]["kakuteijyuni"])` is accessed on `result_df` (which is the output of `race_predictor.predict()` using the dropped `single_race`). This will raise a `KeyError` at runtime for every race in dry-run mode.

The predict and diagnose modes correctly avoid this because they do not try to read `kakuteijyuni` from the prediction output. Only the dry-run mode attempts to use the finish position from the input features.

**Fix:**
```python
# Option A: Save kakuteijyuni before dropping, then access from original feat_df
# Before line 1255:
race_df_full = feat_df[feat_df["race_id"] == race_id].copy()
race_df_single = _drop_post_race_cols(race_df_full.copy())

# Then at line 1270, use race_df_full instead:
if not horse.empty:
    horse_full = race_df_full[race_df_full["umaban"] == bet.umaban]
    if not horse_full.empty and "kakuteijyuni" in horse_full.columns:
        finish_pos = int(horse_full.iloc[0]["kakuteijyuni"])
    else:
        continue
```

### CR-02: Unbounded data duplication in reconcile -- entire pred_df appended to bets.parquet

**File:** `scripts/run_paper_trading.py:962-969`
**Issue:** Every time `_run_reconcile` runs, the **entire** `pred_df` (all predictions for the day, including previously reconciled ones) is appended to `bets.parquet`. On the second reconcile run for the same day, all predictions from the first run are duplicated. On the third run, they are duplicated again. This grows without bound and corrupts cumulative ROI statistics (lines 972-974) computed from `combined`.

**Fix:**
```python
# Only append newly settled rows to bets.parquet
settled_rows = pred_df[pred_df["result"] != 0.0]
if not settled_rows.empty:
    bets_path = config.paper_trading_dir / "bets.parquet"
    if bets_path.exists():
        existing = pd.read_parquet(bets_path)
        combined = pd.concat([existing, settled_rows], ignore_index=True)
    else:
        combined = settled_rows
    combined.to_parquet(bets_path, index=False)
```

## Warnings

### WR-01: `result == 0.0` cannot distinguish lost bets from unsettled bets

**File:** `scripts/run_paper_trading.py:544,900,953`
**Issue:** Predictions are created with `"result": 0.0` (line 544) meaning "unsettled". When a bet loses, it is also set to `result = 0.0` (line 953). This means on a subsequent reconcile run, lost bets from the first run will be re-processed as "unsettled" (line 900: `pred_df[pred_df["result"] == 0.0]`). While the re-processing would produce the same `0.0` result, it is fragile and semantically incorrect. A more robust approach would use a sentinel value (e.g., `-1.0` or `NaN`) for "unsettled" and reserve `0.0` for "settled as loss".

**Fix:** Use a distinct sentinel for unsettled state:
```python
# At line 544:
"result": float("nan"),  # 未確定

# At line 900:
unsettle = pred_df[pred_df["result"].isna()]

# At line 953:
pred_df.at[idx, "result"] = 0.0  # LOSE
```

### WR-02: `_compute_surface` lambda crashes on nullable Int64 with pd.NA values

**File:** `src/db/etl.py:232-234`
**Issue:** `_apply_type_conversions` converts `trackcd` to nullable `Int64` dtype (line 176: `.astype("Int64")`). Then `_compute_surface` at line 232 uses `df["trackcd"].apply(lambda x: "turf" if 10 <= x <= 22 ...)`. When `x` is `pd.NA`, the comparison `10 <= pd.NA` raises `TypeError: boolean value of NA is ambiguous`. The same issue exists in `src/db/readers.py:176-177`.

This is data-dependent: if every row has a valid `trackcd`, the bug is not triggered. But if any row has a missing `trackcd`, the entire ETL batch fails.

**Fix:**
```python
def _compute_surface(df: pd.DataFrame) -> pd.DataFrame:
    if "trackcd" in df.columns:
        import numpy as np
        trackcd = df["trackcd"]
        df["surface"] = np.where(
            trackcd.isna(), "other",
            np.where(
                trackcd.between(10, 22), "turf",
                np.where(trackcd.between(23, 29), "dirt", "other")
            )
        )
    return df
```

### WR-03: Dry-run date argument parsing inconsistency with other modes

**File:** `scripts/run_paper_trading.py:1123-1125`
**Issue:** In `_run_dry_run`, when `--start` and `--end` are used, the code at line 1124-1125 parses them as `YYYYMMDD` format (`args.start[:4]`, `args.start[4:6]`, `args.start[6:8]`). But the argparse help text at line 132 says `--start` expects `YYYY-MM-DD` format, and the actual usage shown at line 18 uses `--start 2024-07-01`. When the user passes `YYYY-MM-DD` as documented, `int(args.start[:4])` would fail on the hyphens. This contradicts the diagnose mode which correctly uses `args.start.replace("-", "")` at line 705-706.

**Fix:**
```python
# Replace lines 1124-1125 with:
start = date.fromisoformat(args.start.replace("-", ""))
end = date.fromisoformat(args.end.replace("-", ""))
```

### WR-04: Reconcile bets.parquet append does not deduplicate against existing data

**File:** `scripts/run_paper_trading.py:964-969`
**Issue:** Even with CR-02 fixed to only append settled rows, there is no deduplication logic. If reconcile is interrupted after writing `pred_df.to_parquet` (line 960) but before `bets.parquet` write completes, a re-run would re-append the same settled rows. There should be a PK-based deduplication (e.g., on `race_id + umaban + race_date`) when writing to `bets.parquet`.

**Fix:** Add deduplication after concat:
```python
combined = combined.drop_duplicates(subset=["race_id", "umaban", "race_date", "bet_type"], keep="last")
```

## Info

### IN-01: `_TABLE_TYPE_RULES` type hint is inaccurate for `sentinel_float` entries

**File:** `src/db/etl.py:83`
**Issue:** The type annotation `dict[str, list[str] | dict | list[dict]]` does not accurately describe the actual structure. The "entries" table has `sentinel_float` as a bare `dict` (not wrapped in a list), while "races" has it as `list[dict]`. The runtime code handles both forms, but the type annotation `dict` is too loose and would not help static analysis catch future mistakes.

**Fix:** Consider using a TypedDict or a stricter Union type:
```python
from typing import Union
SentinelRule = dict[str, list[str] | int]  # columns, sentinels, divisor
_TABLE_TYPE_RULES: dict[str, dict[str, list[str] | list[SentinelRule]]] = { ... }
```
Then normalize entries to also use `list[dict]`:
```python
"sentinel_float": [
    {"columns": ["harontimel3", "harontimel4", "jyuni2c", "jyuni3c"], "sentinels": ["000", "999", "00"]},
],
```

### IN-02: Massive code duplication between predict/diagnose/dry-run modes

**File:** `scripts/run_paper_trading.py:294-871, 682-871, 1095-1308`
**Issue:** The feature engineering pipeline (FeatureEngine, SubModelManager, bloodline, sire, pace, course features) is duplicated nearly identically across `_run_predict` (lines 367-426), `_run_diagnose` (lines 732-778), and `_run_dry_run` (lines 1172-1228). Any change to feature engineering must be replicated in three places. This is a maintainability risk, not a correctness bug.

**Fix:** Extract a shared `_build_features(feat_df, store, race_df, entry_df, race_ids)` function that all three modes call.

---

_Reviewed: 2026-05-19T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
