# Phase 47 Research: ETL Data Pipeline for Track Condition Data

## 1. Source Data Analysis

### Dirt Moisture CSV (`data/20180728~20260531ダート含水率.csv`)
- **Rows:** 189,334 (matches ROADMAP estimate)
- **Format:** No header, 2 columns: `{18-digit-id},{float}`
- **Date range:** 2018-07-28 ~ 2026-05-31
- **Example:** `201807280101010201,2.8`

### Turf Cushion CSV (`data/20200912~20260531クッション値.csv`)
- **Rows:** 133,672 (matches ROADMAP estimate)
- **Format:** No header, 2 columns: `{18-digit-id},{float}`
- **Date range:** 2020-09-12 ~ 2026-05-31
- **Example:** `202009120604010301,11.2`

### ID Format
- 18-digit ID = 16-digit `race_id` + 2-digit `umaban` (entry number)
- `race_id` = `YYYYMMDD` (8) + `jyocd` (2) + `kaiji` (2) + `nichiji` (2) + `racenum` (2)
- Values are **identical for all entries** within a race (race-level property)
- Roadmap requires aggregation to race-level (deduplicate by race_id)

### Key Observation: Turf Cushion Starts in 2020
- Dirt moisture data starts 2018-07-28
- Turf cushion data starts 2020-09-12
- This means **WF Fold0 (train 2020-2023) will have partial NaN for cushion data in 2020**
- Phase 50 success criterion #5 explicitly validates this NaN rate

## 2. Existing ETL Infrastructure

### `src/db/etl.py` (PostgreSQL → Parquet)
- `_compute_race_id(df)`: year + monthday + jyocd + kaiji + nichiji + racenum → 16-digit race_id
- `_compute_race_date(df)`: year + monthday → race_date (datetime64)
- `_compute_surface(df)`: trackcd → "turf"/"dirt"/"other"
- `_compute_track_condition_code(df)`: sibababacd/dirtbabacd → track_condition_code
- The new CSV data is NOT from PostgreSQL — it's external CSV → requires a **separate conversion script**, not modification of the existing ETL

### `src/db/parquet_store.py` (Parquet I/O)
- `ParquetStore.write(category, name, df, partition_cols=None)`: atomic write to `data/{category}/{name}.parquet`
- `ParquetStore.read(category, name, filters=None)`: read with pyarrow push-down filters
- `ParquetStore.exists(category, name)`: existence check
- Target location: `data/raw/dirt_moisture.parquet` and `data/raw/turf_cushion.parquet`

### `src/db/readers.py` (Load helpers)
- `load_races(store, start, end)`, `load_entries(store, start, end)`, etc.
- `date_filters(start, end)`: generates `[("race_date", ">=", dt), ...]`
- `coerce_types(df)`: type conversions for numeric columns
- Pattern: new `load_dirt_moisture(store, start, end)` and `load_turf_cushion(store, start, end)` functions

### `src/db/repository.py` (DataRepository)
- Facade over ParquetStore with domain-specific load methods
- `load_wide_odds()`, `load_trio_odds()`, etc.
- Pattern: add `load_dirt_moisture()` and `load_turf_cushion()` methods

## 3. FeatureEngine Integration Points

### `src/features/feature_engine.py`
- `build_all()`: merges race_df + entry_df + odds_df by race_id → result_df
- Step 1: `pd.merge(race_dedup, entry_subset, on="race_id", how="inner")`
- Step 2: `pd.merge(result_df, odds_df[...], on=["race_id", "umaban"], how="left")`
- External data pattern (bloodline features): `store.read("raw", "horses")`
- Track condition data should merge as **race-level left join on race_id** after step 1

### `src/features/bloodline_features.py`
- Example of store-based external data loading
- Uses `store.read("raw", "horses")`, `store.read("raw", "keito")`
- Lazy-loaded and cached via `_keito_cache`

## 4. POST_RACE_COLS Verification

### Current `POST_RACE_COLS` (domain/types.py)
- 41 columns: kakuteijyuni, confirmed_odds, ninki, time, laptime*, etc.
- **None of the new columns (dirt_moisture, turf_cushion) are currently listed**
- These values are published by JRA before race start → **NOT post-race**
- Phase 47 SC #4 requires CI test to confirm they are NOT in POST_RACE_COLS

## 5. CSV → Parquet Conversion Design

### Parsing Strategy
```python
# Read CSV (no header)
df = pd.read_csv(path, header=None, names=["id", "value"])

# Extract race_id (first 16 chars) and umaban (last 2 chars)
df["race_id"] = df["id"].str[:16]
df["umaban"] = df["id"].str[16:18]

# Deduplicate to race-level (values are identical per race)
race_level = df.drop_duplicates(subset=["race_id"])[["race_id", "value"]]

# Compute race_date from race_id
race_level["race_date"] = pd.to_datetime(race_level["race_id"].str[:8], format="%Y%m%d")

# Rename value column
race_level = race_level.rename(columns={"value": "dirt_moisture"})  # or "turf_cushion"
```

### Output Schema
```
dirt_moisture.parquet:
  - race_id: str (16-digit, merge key)
  - race_date: datetime64 (for filtering)
  - dirt_moisture: float64

turf_cushion.parquet:
  - race_id: str (16-digit, merge key)
  - race_date: datetime64 (for filtering)
  - turf_cushion: float64
```

### ~~Location: `data/raw/dirt_moisture.parquet`, `data/raw/turf_cushion.parquet`~~

> **[PLAN D-03 OVERRIDE]** 個別Parquetではなく、単一 `data/raw/track_conditions.parquet` に統合する決定。
> 理由: 同一race_idキーのrace-levelデータであり、結合が不要になる。PLAN.mdのD-03を参照。

## 6. Implementation Scope (What Phase 47 Does NOT Do)

Phase 47 is **data plumbing only** — no feature computation:
- ❌ No new feature columns in FeatureEngine
- ❌ No FEATURE_COLS registration
- ❌ No model retraining
- ✅ CSV → Parquet conversion script
- ✅ DataRepository integration (load methods)
- ✅ readers.py helper functions
- ✅ CI test: not in POST_RACE_COLS
- ✅ FeatureEngine.build_all() race-level merge (prepare for Phase 48)

## 7. Risks & Considerations

1. **ID format validation**: The 18-digit ID format assumption needs verification with more samples across different courses/dates
2. **NaN coverage**: Turf cushion starts 2020-09 → ~9 months of 2020 has no data. Needs NaN rate analysis.
3. **Race matching**: Some race_ids in CSV may not exist in races.parquet (data gaps). Left join from races side ensures no data loss.
4. **Value consistency**: Should verify that values within a race are truly identical (sampling check)
5. **Script naming**: Follow existing convention: `scripts/convert_track_condition.py` or similar

## 8. Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `scripts/convert_track_condition.py` | **CREATE** | CSV → Parquet conversion script |
| `src/db/readers.py` | **MODIFY** | Add `load_dirt_moisture()`, `load_turf_cushion()` |
| `src/db/repository.py` | **MODIFY** | Add `load_dirt_moisture()`, `load_turf_cushion()` methods |
| `src/features/feature_engine.py` | **MODIFY** | Add race-level merge for track condition data |
| `src/domain/types.py` | **VERIFY** | Confirm dirt_moisture/turf_cushion NOT in POST_RACE_COLS |
| `tests/test_etl_type_conversion.py` | **MODIFY** | Add CI test for POST_RACE_COLS verification |
| `data/raw/dirt_moisture.parquet` | **CREATE** | Converted dirt moisture data |
| `data/raw/turf_cushion.parquet` | **CREATE** | Converted turf cushion data |
