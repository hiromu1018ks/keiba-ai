# Technology Stack

**Project:** keiba-ai v1.8 Turf Precision Calibration
**Researched:** 2026-05-19
**Scope:** LapTime/HaronTime ETL, popularity band calibration, turf intra-race relative features, conditional interaction features, regime x surface EV correction
**Supersedes:** v1.7 STACK.md (all prior dependencies remain valid)

## Verdict: Zero New Dependencies -- All Tools Already Installed

All 5 new features (ETL extension, popularity band calibration, turf relative features, conditional interaction features, regime x surface EV correction) are implementable entirely within the existing technology stack. No new library dependencies are required. The changes are confined to:

1. **ETL configuration** -- adding float type rules for columns already extracted via `SELECT *`
2. **Feature computation** -- pure pandas/numpy operations following existing module patterns
3. **Column list extensions** -- adding entries to `FEATURE_COLS`, `POST_RACE_COLS`, `_TABLE_TYPE_RULES`

The only infrastructure action is an ETL re-extraction so that LapTime1~25 (RA table) and HaronTimeL4 (SE table) columns are stored as floats in Parquet instead of being silently passed through as strings.

## Current Installed Stack (Unchanged from v1.7)

| Package | Installed Version | pyproject.toml Minimum | Status |
|---------|-------------------|----------------------|--------|
| Python | 3.11 | >=3.11 | Pinned via mise |
| LightGBM | 4.6.0 | >=4.3 | Up to date |
| XGBoost | 3.2.0 | >=2.0 | Up to date |
| CatBoost | 1.2.10 | >=1.2 | Up to date |
| scikit-learn | 1.8.0 | >=1.4 | Up to date |
| scipy | 1.17.1 | >=1.11 | Up to date |
| pandas | 2.3.3 | >=2.2 | Up to date |
| numpy | 2.4.3 | >=1.26 | Up to date |
| pyarrow | 23.0.1 | >=14.0 | Up to date |
| mlflow | 3.10.1 | >=2.12 | Up to date |
| optuna | 4.8.0 | >=3.5 | Up to date |

## Recommended Stack for v1.8

### Production Dependencies: No New Packages

| Technology | Version | Role in v1.8 | Why |
|------------|---------|-------------|-----|
| pandas `groupby` + `transform` | 2.3.3 | Popularity band calibration, turf relative features, LapTime aggregation | Already the foundation for all feature modules. `groupby("race_id").transform()` for broadcast, `groupby().agg()` for per-race features. |
| pandas `cut` | 2.3.3 | Popularity band binning (1-3, 4-6, 7-9, 10-12, 13+) | Fixed domain-knowledge bins, not learned. `pd.cut()` with explicit bin edges is more transparent than `KBinsDiscretizer`. |
| numpy `polyfit` | 2.4.3 | Lap time trend features (linear regression on per-furlong times) | Already used for `haron_zscore_trend`. Same pattern for LapTime trajectory. |
| scikit-learn `IsotonicRegression` | 1.8.0 | Existing EV calibration (no change) | Already in stack. Popularity band scaling operates at a different layer (multiplicative correction on top of isotonic). |
| SQLAlchemy `text()` | >=2.0 | ETL queries (no change) | `SELECT * FROM n_race` already captures LapTime columns -- only type conversion rules need updating. |

### Data Requirements: ETL Re-extraction Required

| Parquet File | New Columns | Source Table | ETL Action |
|-------------|-------------|--------------|------------|
| `data/raw/races.parquet` | `laptime1`..`laptime25`, `harontimes3`, `harontimes4`, `harontimel3` (RA-level), `harontimel4` (RA-level) | `n_race` (RA table) | Add float type rules in `_TABLE_TYPE_RULES["races"]`; re-extract |
| `data/raw/entries.parquet` | `harontimel4` (SE-level, per-horse) | `n_uma_race` (SE table) | Add float type rule for `harontimel4` in `_TABLE_TYPE_RULES["entries"]`; re-extract |

**Key insight:** The ETL already does `SELECT * FROM n_race` and `SELECT * FROM n_uma_race`. LapTime1~25 and HaronTimeL4 exist in EveryDB2 but are currently passed through as strings because they are not listed in `_TABLE_TYPE_RULES`. Adding float conversion rules is the only code change. The data is already being read -- it just needs type coercion.

**RA table (RACE) POST_RACE columns to extract:**
- `LapTime1`..`LapTime25` -- varchar(3), 99.9秒 units, per-furlong leader times (fields 68-92 in 03-RACE.md)
- `HaronTimeS3`, `HaronTimeS4` -- varchar(3), first 3/4 furlong totals (fields 94-95)
- `HaronTimeL3`, `HaronTimeL4` -- varchar(3), last 3/4 furlong totals (fields 96-97)

**SE table (UMA_RACE) POST_RACE columns to extract:**
- `HaronTimeL4` -- varchar(3), per-horse last-4-furlong time (field 58 in 04-UMA_RACE.md)
- `HaronTimeL3` -- already extracted and typed as float

**ETL re-extraction command:**
```bash
python scripts/run_etl.py --mode full --start 20140101 --end 20251231 --tables races entries
```

## What NOT to Add

| Library/Approach | Why Rejected |
|-----------------|-------------|
| featuretools | Overkill for 4-6 domain-knowledge interaction pairs. Adds heavy dependency. Manual string concat + numeric product is more auditable for PIT safety. |
| sklearn.preprocessing.KBinsDiscretizer | Popularity bands are fixed domain boundaries (1-3, 4-6, 7-9, 10-12, 13+), not data-driven bins. `pd.cut()` with explicit edges is more transparent. |
| Separate lap_times Parquet file | LapTime1~25 are per-race columns in RA table. Adding 25 columns to `races.parquet` is simpler than a separate file + join. Flat columns match EveryDB2 schema directly. |
| Separate popularity_band_model.py | The calibration is a simple per-band mean residual ratio applied as a multiplicative scalar. A single method in `ev_correction_model.py` is sufficient. |
| Separate turf_feature_engine.py | Turf relative features use the same `groupby("race_id").rank()` pattern as existing `add_race_transforms()`. Just add columns to the existing `race_rank_cols` list. |
| statsmodels | No regression needed beyond LightGBM's native handling of categorical interactions. |
| betacal / Venn-ABERS for popularity bands | The popularity band correction is a ratio scaler (mean(actual/predicted) per band), not a probability calibration problem. |
| Separate regime-surface EV model | LightGBM handles categorical interactions natively when both `surface` and `regime_state` are in FEATURE_COLS as categorical. No separate model needed. |

## Integration Points

### 1. ETL Extension (LapTime + HaronTimeL4)

**Files to modify:**

| File | Change | Complexity |
|------|--------|------------|
| `src/db/etl.py` `_TABLE_TYPE_RULES` | Add LapTime1~25 float rules for "races" key; add `harontimel4` float rule for "entries" key | Low (3 lines) |
| `src/domain/types.py` `POST_RACE_COLS` | Add `laptime1`..`laptime25`, `harontimes3`, `harontimes4`, `harontimel4` (RA-level), `harontimes3` (RA-level) | Low (list append) |
| `src/db/readers.py` `_FLOAT_COLS` | Add `harontimel4` to entries float coercion set | Low (1 line) |
| `src/features/feature_engine.py` `_race_entry_shared` | Add `harontimes3`, `harontimes4`, `laptime*` if RA-level HaronTimes are shared with entries | Low (list append) |

**PIT safety:** LapTime and HaronTime are POST_RACE data. They must be added to `POST_RACE_COLS` immediately. Feature computation must only use past-race aggregations (HorseHistoryFeatures pattern: `history[history["race_date"] < target_date]`), never the current race's values.

**Data flow for LapTime features:**
```
RA table LapTime1~25 (per-race, leader times)
  -> ETL -> races.parquet (laptime1..laptime25 float columns)
  -> HorseHistoryFeatures: compute lap_time_pace_profile from past races
     (e.g., last 3 races' lap patterns, front-loaded vs closing pace)
  -> Aggregated features only (e.g., avg_lap_front_ratio, lap_closing_ratio)
  -> Current race's LapTime is in POST_RACE_COLS, dropped by SAFE-01
```

**Data flow for HaronTimeL4 (per-horse, SE table):**
```
SE table HaronTimeL4 (per-horse, last 4 furlongs)
  -> ETL -> entries.parquet (harontimel4 float column)
  -> HorseHistoryFeatures: compute avg_haron_l4 from past races
  -> Already handled: harontimel3 is already in BASE_COLS and history computation
  -> Add harontimel4_avg alongside harontimel5_avg
```

### 2. Popularity Band Calibration

**Files to modify:**

| File | Change | Complexity |
|------|--------|------------|
| `src/models/ev_correction_model.py` | Add `_compute_popularity_band_scales()` method; apply in `correct_ev()` after Isotonic/OddsBand scaling | Medium |
| `src/models/ev_correction_model.py` | Store `self._pop_band_scales: dict[str, float]` | Low |

**Implementation pattern:**
```python
POP_BANDS = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 99)]
POP_BAND_NAMES = ["1-3", "4-6", "7-9", "10-12", "13+"]

def _compute_popularity_band_scales(self, df: pd.DataFrame) -> dict[str, float]:
    """OOF residual ratio per popularity band."""
    pop = df["popularity_rank"].astype(float)
    actual = df["confirmed_odds"].astype(float)  # actual return for winners
    predicted = df["ev_win_calibrated"].astype(float)

    scales = {}
    for (lo, hi), name in zip(POP_BANDS, POP_BAND_NAMES):
        mask = (pop >= lo) & (pop <= hi)
        winners_mask = mask & (df["kakuteijyuni"] == 1)
        if winners_mask.sum() > 10:
            ratio = actual[winners_mask].mean() / predicted[winners_mask].clip(lower=0.01).mean()
            scales[name] = float(np.clip(ratio, 0.5, 2.0))  # safety clip
        else:
            scales[name] = 1.0
    return scales
```

**No new dependencies.** Uses `pd.cut()` conceptual pattern (manual masks for fixed bins) + `groupby().mean()` + `numpy.clip()`.

**Ordering in pipeline:** Popularity band scaling applies AFTER Isotonic calibration and OddsBand scaling:
```
ev_win_corrected -> Isotonic -> ev_win_calibrated -> OddsBand scaling -> popularity band scaling -> final
```

### 3. Turf Intra-Race Relative Features

**Files to modify:**

| File | Change | Complexity |
|------|--------|------------|
| `src/features/horse_history_features.py` `add_race_transforms()` | Add columns to `race_rank_cols` list | Low (2-4 lines) |
| `src/features/relative_features.py` `_BASE_FEATURES` | Add entries for `form_trend`, `blood_total_wr` zscore/rank variants | Low (2-4 dict entries) |

**Existing columns to promote to race_rank:**

| Column | Already Computed In | race_rank Transform |
|--------|-------------------|-------------------|
| `form_trend` (or `haron_zscore_trend`) | `horse_history_features.py` TSER-03 | Percentile rank within race |
| `blood_total_wr` | `bloodline_features.py` | Percentile rank (descending) within race |
| `sire_wr` | `bloodline_features.py` | Already in `race_rank_cols` via `rel_sire_quality_rank` |
| `time_improvement_rate` | `horse_history_features.py` HODDS-03 | Percentile rank within race |

**Key insight:** The `add_race_transforms()` method in `horse_history_features.py` already generates `{col}_race_rank` for any column in `race_rank_cols` via `groupby("race_id").rank(pct=True)`. Adding new columns to this list is a one-line change per feature. The existing `compute_relative_features()` function already handles z-score/rank transforms generically through `_BASE_FEATURES`.

**Turf-only scope:** The v1.8 milestone targets turf model improvement. These race_rank features should be computed for all surfaces (LightGBM handles surface interaction), but evaluation focus is on turf IC.

### 4. Conditional Interaction Features

**Files to modify:**

| File | Change | Complexity |
|------|--------|------------|
| `src/features/interaction_features.py` | Add 4-6 new features to `INTERACTION_COLS` and `compute_interaction_features()` | Medium |

**Proposed interactions (all using existing columns):**

| Interaction | Type | Source Columns | Rationale |
|-------------|------|---------------|-----------|
| `grade_x_form_trend` | Numeric product | `class_level_current * haron_zscore_trend` | Higher-class races + improving form = stronger signal |
| `distance_x_closing` | Numeric product | `kyori * closing_index_avg` | Closing ability matters more at longer distances |
| `surface_x_condition_grade` | Categorical concat | `surface + "_" + track_condition_code + "_" + grade_code` | Surface/track/grade triple captures racing context |
| `field_size_x_closing` | Numeric product | `field_size * closing_index_avg` | Closing harder in larger fields |
| `class_x_sire_quality` | Numeric product | `class_level_current * sire_wr` | Sire quality matters more at higher classes |

**All source columns already exist in the DataFrame** at the point where `compute_interaction_features()` is called. String concatenation uses `.astype("category")` following existing pattern. Numeric products use `.where()` for NaN safety.

**PIT safety:** All source columns are pre-race (distance, surface, class, sire_wr are known before race; closing_index_avg and form_trend are computed from past races only). No POST_RACE data used.

### 5. Regime x Surface EV Correction

**Files to modify:**

| File | Change | Complexity |
|------|--------|------------|
| `src/models/ev_correction_model.py` `FEATURE_COLS` | Add `"regime_state"` | Low (1 line) |
| `src/models/ev_correction_model.py` `_prepare_features()` | Add `"regime_state"` to categorical conversion list | Low (1 line) |

**Implementation:** Add `regime_state` to `FEATURE_COLS`. LightGBM handles categorical features natively -- when both `surface` and `regime_state` are categorical columns, LightGBM automatically learns their interaction through tree splits.

The `regime_state` value is available at prediction time from `RegimeDetector`, which runs before `EVCorrectionModel` in the pipeline (confirmed in `race_predictor.py` flow). The `_prepare_features()` method already converts `surface` and `distance_bin` to categorical -- add `regime_state` to the same list.

**Optional explicit interaction:** If desired, add `"regime_x_surface"` as a pre-computed string column:
```python
df["regime_x_surface"] = (
    df["regime_state"].astype(str) + "_" + df["surface"].astype(str)
).astype("category")
```
This is optional because LightGBM learns categorical interactions natively, but explicit encoding can speed convergence for rare combinations.

**Regime availability at EV correction time:**
- Training: `regime_state` is computed from rolling market statistics (features available before race start)
- Inference: `RegimeDetector.predict()` produces `regime_state` before `EVCorrectionModel.correct_ev()` is called
- Both paths: `regime_state` is in the DataFrame before EV correction runs

## Installation

```bash
# No new packages needed. Verify existing installation:
pip install -e ".[dev]"

# ETL re-extraction required for LapTime/HaronTimeL4:
python scripts/run_etl.py --mode full --start 20140101 --end 20251231 --tables races entries
```

## Sources

### Codebase Analysis (HIGH confidence)
- `src/db/etl.py`: `_TABLE_TYPE_RULES` structure, `run_full_load()` SELECT * pattern
- `src/db/parquet_store.py`: ParquetStore read/write API
- `src/db/repository.py`: DataRepository DI pattern
- `src/features/feature_engine.py`: `build_all()` pipeline, `_map_basic_features()`, `_race_entry_shared` column list
- `src/features/horse_history_features.py`: `add_race_transforms()`, `race_rank_cols`, HaronTimeL3 usage pattern
- `src/features/interaction_features.py`: `INTERACTION_COLS`, `compute_interaction_features()` pattern
- `src/features/relative_features.py`: `_BASE_FEATURES` structure, `compute_relative_features()`
- `src/features/pace_aptitude_features.py`: HaronTimeL3 usage for pace features
- `src/models/ev_correction_model.py`: `FEATURE_COLS`, `_prepare_features()`, `correct_ev()` flow
- `src/models/regime_detector.py`: `RegimeState` enum, feature columns
- `src/betting/odds_band_filter.py`: `BANDS`, `BAND_NAMES` structure
- `src/domain/types.py`: `POST_RACE_COLS` definition
- `config/etl_tables.yaml`: n_race/n_uma_race configuration (already extracts all columns via SELECT *)

### EveryDB2 Schema Documentation (HIGH confidence)
- `docs/everydb2/03-RACE.md`: RA table fields 68-97 (LapTime1~25, HaronTimeS3/S4, HaronTimeL3/L4)
- `docs/everydb2/04-UMA_RACE.md`: SE table fields 58-59 (HaronTimeL4, HaronTimeL3 per-horse)

---
*Stack research for: v1.8 Turf Precision Calibration*
*Researched: 2026-05-19*
