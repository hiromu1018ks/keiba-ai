# Phase 26: EveryDB2 New Features - Research

**Researched:** 2026-05-14
**Domain:** Feature engineering from EveryDB2 unused tables (n_hansyoku, n_record, n_mining) + intra-race relative features
**Confidence:** HIGH (full table doc audit + codebase integration pattern analysis)

## Summary

This phase extracts high-value features from three EveryDB2 tables (n_hansyoku, n_record, n_mining) that are already defined in `config/etl_tables.yaml` but not yet extracted to Parquet or used in ML pipelines. Additionally, a new `relative_features.py` module generates 5-10 intra-race comparison features using existing horse-level metrics via `groupby("race_id")` transforms.

**Critical PIT audit finding:** n_mining (82 columns) is entirely PRE-race data. The `DataKubun` field classifies each record as one of three pre-race prediction tiers (1=day-before/entry-announcement, 2=same-day/weather-track-announcement, 3=final/weight-announcement). All 82 columns are structured as per-horse predicted finish times with confidence intervals for up to 18 horses. There are NO post-race columns in this table. The highest-value DataKubun=3 (final prediction) is available after weight announcement but before race start, making it PIT-safe for all training and inference scenarios.

**Primary recommendation:** Use DataKubun=3 (final predictions) as the primary n_mining source. Compute features: `dm_time_rank` (predicted time rank within race), `dm_time_vs_mean` (z-score of predicted time), `dm_confidence_range` (GosaP+GosaM as spread), and `dm_time_margin_to_fav` (gap between horse's predicted time and race-favorite's predicted time).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** n_hansyoku, n_record, n_mining -- 3 tables extracted individually via `run_etl.py --tables`
- **D-02:** ETL execution is in Phase 26 scope (user runs locally, PostgreSQL-dependent)
- **D-03:** n_mining 82-column PRE/POST classification uses `docs/everyDB2/44-MINING.md` column descriptions as primary source
- **D-04:** POST columns excluded; only PRE columns used
- **D-05:** PRE/POST classification results documented as output
- **D-06:** n_hansyoku + n_sanku combined for comprehensive bloodline features
- **D-07:** New module (e.g., `features/dam_pedigree_features.py`) created
- **D-08:** BMS extension implementation location is Claude discretion
- **D-09:** FEATURES.md TS-05 recommended 5-10 features in new `relative_features.py`
- **D-10:** Relative feature selection is Claude discretion
- **D-11:** Computation method is Claude discretion

### Claude's Discretion
- Specific feature content and target model placement (Stage1 / Stage2 / both)
- BMS extension location (sire_features.py extension vs new module)
- Relative comparison feature names and computation methods
- n_record feature design (course time index calculation method)
- n_mining PRE column feature selection
- FEATURE_COLS insertion positions
- Test addition/update content
- POST_RACE leakage test pass verification method
- Parquet schema verification after ETL

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | n_hansyoku + n_sanku comprehensive bloodline features | n_hansyoku (19 cols) + n_sanku (26 cols) schema analysis below; proposed features: dam offspring wr, BMS extended stats, breeding farm indicator |
| DATA-02 | n_record course time index features | n_record (48 cols) schema analysis below; proposed features: course record ratio, track record delta |
| DATA-03 | Intra-race relative comparison features | 7 proposed features using groupby("race_id") transforms on existing HorseHistoryFeatures outputs |
| DATA-04 | n_mining PRE/POST classification + feature extraction | Full PIT audit: all 82 columns are PRE (DataKubun=1/2/3 pre-race predictions); proposed features from DataKubun=3 records |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| ETL (EveryDB2 to Parquet) | API / Backend | -- | PostgreSQL connection required, offline batch |
| Bloodline feature computation | API / Backend | -- | Static master data, precomputed at training time |
| n_mining feature extraction | API / Backend | -- | Race-level prediction data, merged at training |
| n_record feature computation | API / Backend | -- | Static course records, joined at training |
| Relative feature computation | API / Backend | -- | groupby("race_id") transforms, computed after all per-horse features |
| FEATURE_COLS definition | API / Backend | -- | Python model class lists |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | 2.x | DataFrame operations, groupby transforms | Already core dependency |
| numpy | 1.26+ | Numerical operations | Already core dependency |
| pyarrow | 14+ | Parquet I/O | Already core dependency via ParquetStore |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| LightGBM | 4.x | Model training with new FEATURE_COLS | After features are added to FEATURE_COLS |

### Installation
No new packages needed. All features use existing dependencies.

## Architecture Patterns

### System Architecture Diagram

```
                  EveryDB2 (PostgreSQL)
                  ┌──────────────────────────┐
                  │ n_hansyoku (19 cols)      │
                  │ n_sanku (26 cols)         │──── ETL (run_etl.py --tables) ────┐
                  │ n_record (48 cols)        │                                    │
                  │ n_mining (82 cols)        │────────────────────────────────────┘
                  └──────────────────────────┘                                              │
                                                                                              v
                  ┌───────────────────────────────────────────────────────────────────────────┐
                  │                      Parquet Files (data/raw/)                           │
                  │  hansyoku.parquet, sanku.parquet, record.parquet, mining.parquet          │
                  └──────────┬────────────┬─────────────────┬────────────────┬────────────────┘
                             │            │                 │                │
                             v            v                 v                v
                  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
                  │ DamPedigree  │ │ (n_record    │ │ Mining       │ │ Relative     │
                  │ Features     │ │  features)   │ │ Features     │ │ Features     │
                  │ (new module) │ │ (new module) │ │ (new module) │ │ (new module) │
                  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                         │                │                │                │
                         v                v                v                v
                  ┌─────────────────────────────────────────────────────────────────┐
                  │              feature_engine.py build_all()                       │
                  │   ┌─────────────────────────────────────────────────────────┐   │
                  │   │  compute_dam_pedigree_features()  ← after bloodline    │   │
                  │   │  compute_record_features()        ← new insertion      │   │
                  │   │  compute_mining_features()         ← new insertion      │   │
                  │   │  compute_relative_features()       ← after intra_race   │   │
                  │   └─────────────────────────────────────────────────────────┘   │
                  └───────────────────────────┬─────────────────────────────────────┘
                                              │
                                              v
                  ┌─────────────────────────────────────────────────────────────────┐
                  │  FEATURE_COLS (Stage1, Stage2, Place)                           │
                  │  + dam_wr, dam_surface_wr, bms_distance_wr, ...                │
                  │  + record_time_ratio, record_track_delta, ...                  │
                  │  + dm_time_rank, dm_confidence_range, ...                      │
                  │  + rel_norm_finish_zscore, rel_haron_vs_mean, ...              │
                  └─────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure
```
src/features/
├── dam_pedigree_features.py   # NEW: n_hansyoku + n_sanku bloodline features
├── record_features.py         # NEW: n_record course time index features
├── mining_features.py         # NEW: n_mining pre-race prediction features
├── relative_features.py       # NEW: intra-race relative comparison features
├── bloodline_features.py      # EXISTING: unchanged
├── sire_features.py           # EXISTING: extend with bms_distance_wr, bms_surface_wr
├── intra_race_features.py     # EXISTING: unchanged (2 features)
├── feature_engine.py          # EXISTING: add 4 new module calls to build_all()
└── ...
```

### Pattern 1: ParquetStore-based Feature Module (PIT-safe static master data)
**What:** New modules read from ParquetStore, compute features, return DataFrame with (race_id, umaban) keys
**When to use:** For all new feature modules that use static master data
**Example:**
```python
# Pattern from bloodline_features.py (verified in codebase)
class DamPedigreeFeatures:
    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._hansyoku_cache: pd.DataFrame | None = None

    def _load_hansyoku(self) -> pd.DataFrame:
        if self._hansyoku_cache is None:
            self._hansyoku_cache = self.store.read("raw", "hansyoku")
        return self._hansyoku_cache

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        # Static data -- no PIT filtering needed
        # Merge on kettonum, compute features
        ...
        return result[["race_id", "umaban"] + FEATURE_COLS]
```

### Pattern 2: groupby("race_id") Relative Features
**What:** Transform per-horse absolute features into within-race relative features
**When to use:** For features that capture competitive positioning within the race field
**Example:**
```python
# Pattern from intra_race_features.py (verified in codebase)
def compute_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # z-score normalization within each race
    df["rel_norm_finish_zscore"] = df.groupby("race_id", observed=True)[
        "norm_finish_logit_avg"
    ].transform(lambda x: (x - x.mean()) / x.std().replace(0, 1))
    # rank within race (ascending = better)
    df["rel_haron_rank"] = df.groupby("race_id", observed=True)[
        "harontimel5_avg"
    ].rank(method="min", ascending=True)
    return df
```

### Anti-Patterns to Avoid
- **Per-horse loop for mining data:** n_mining is structured as (race_id, umaban1..18) wide format. Pivot/melt to long format, then merge. Do NOT iterate per horse.
- **Joining n_record without filtering:** n_record has RecInfoKubun=1 (course record) and =2 (GI record). Always filter to RecInfoKubun=1 for general features, or handle separately.
- **Using DataKubun=1 (day-before) as primary source:** DataKubun=3 (final prediction) includes all available information (weight, track condition). Always prefer the highest available DataKubun for maximum information.
- **Duplicate BMS features:** sire_features.py already computes `bms_wr`. Extend sire_features.py for BMS distance/surface variants rather than duplicating BMS resolution logic.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bloodline code resolution | Custom ketto3info -> parent mapping | horses.parquet ketto3infohansyokunum columns | Already resolved in sire_features.py pattern |
| Beta smoothing for win rates | Custom Bayesian prior | Beta(1,10) pattern from bloodline_features.py | Consistent prior across all bloodline features |
| PIT-safe career stats | Custom cumulative calculation | horse_career_stats.parquet (shift(1)+cumsum) | Already precomputed, verified PIT-safe |
| Parquet I/O | Custom file reading | ParquetStore.read(category, key) | Handles partitioning, predicate pushdown |
| Cache invalidation | Custom timestamp checks | compute_code_hash() from feature_engine.py | Already handles code-change detection |

**Key insight:** All four new modules follow the exact same pattern as existing feature modules. The only novel aspect is the n_mining wide-to-long pivot.

## Common Pitfalls

### Pitfall 1: n_mining Wide Format Handling
**What goes wrong:** n_mining stores 18 horses as Umaban1..18, DMTime1..18, DMGosaP1..18, DMGosaM1..18. Direct merge on umaban fails because the data is in wide format.
**Why it happens:** Most EveryDB2 tables use long format (one row per horse). n_mining uses wide format (one row per race with 18 horse slots).
**How to avoid:** Melt/pivot to long format first: extract (UmabanN, DMTimeN, DMGosaPN, DMGosaMN) for N=1..18 into separate rows keyed by (race_id, UmabanN).
**Warning signs:** Feature merge produces all-NaN columns; row count mismatch after merge.

### Pitfall 2: n_record Key Mismatch
**What goes wrong:** n_record PK is (Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum, SyubetuCD, Kyori, TrackCD). Joining on (jyocd, trackcd, kyori) alone may produce multiple matches (different years/records for same track/distance).
**Why it happens:** Course records get updated when new records are set. Multiple years of records exist.
**How to avoid:** Filter to the most recent record per (jyocd, trackcd, kyori) combination. Use `sort_values('makedate').groupby([keys]).last()`.
**Warning signs:** Duplicate rows after merge; row count explosion.

### Pitfall 3: n_mining DataKubun Selection
**What goes wrong:** Using DataKubun=1 (day-before predictions) misses weight information. Using all DataKubun values creates duplicates per (race_id, umaban).
**Why it happens:** Each race has up to 3 mining records (one per DataKubun tier). Not filtering creates 3x duplication.
**How to avoid:** Filter to DataKubun=3 (final predictions, after weight announcement). This is the most information-rich and PIT-safe tier. Fall back to DataKubun=2 if 3 is unavailable (e.g., early data years).
**Warning signs:** 3x row count after merge; inconsistent DMTime values for same horse.

### Pitfall 4: BMS Extension Scope Creep
**What goes wrong:** Adding too many BMS features dilutes signal. BMS stats have smaller sample sizes than sire stats because BMS (broodmare sire) has fewer offspring tracked.
**Why it happens:** The same SireFeatures lookup works for BMS, tempting addition of many variants.
**How to avoid:** Add only bms_distance_wr and bms_surface_wr (2 features). These are the highest-value BMS extensions per FEATURES.md D-02 analysis.
**Warning signs:** BMS features show very high NaN rates (>50%) in training data.

### Pitfall 5: Relative Feature NaN Propagation
**What goes wrong:** Relative features computed from features that are NaN for many horses (e.g., harontimel5_avg for debut horses) produce NaN relative features, reducing effective training data.
**Why it happens:** z-score and rank transforms propagate NaN; a single NaN in a race can affect the mean/std calculation.
**How to avoid:** Use `skipna=True` in groupby transforms. For features with high NaN rates, use fallback (e.g., 0 for missing). Document NaN rates per feature.
**Warning signs:** >20% NaN rate in new relative feature columns.

## Code Examples

### n_mining Wide-to-Long Pivot
```python
# Source: docs/everydb2/44-MINING.md analysis
def _pivot_mining_to_long(mining_df: pd.DataFrame) -> pd.DataFrame:
    """Convert n_mining from wide format (18 horses per row) to long format."""
    rows = []
    for i in range(1, 19):
        cols = {
            f"Umaban{i}": "umaban",
            f"DMTime{i}": "dm_time",
            f"DMGosaP{i}": "dm_gosa_plus",
            f"DMGosaM{i}": "dm_gosa_minus",
        }
        subset = mining_df.rename(columns=cols)[
            ["race_id", "umaban", "dm_time", "dm_gosa_plus", "dm_gosa_minus"]
        ].copy()
        # Filter out empty slots (sp = initial value)
        subset = subset[subset["umaban"].notna() & (subset["umaban"] != "sp")]
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)
```

### Dam Pedigree Feature Computation
```python
# Source: bloodline_features.py pattern + n_hansyoku/n_sanku schema
class DamPedigreeFeatures:
    FEATURE_COLS = [
        "dam_wr",           # Dam's offspring overall win rate (Beta smoothed)
        "dam_surface_wr",   # Dam's offspring surface-specific win rate
        "bms_distance_wr",  # Broodmare Sire distance-specific win rate
        "bms_surface_wr",   # Broodmare Sire surface-specific win rate
    ]

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        # 1. From horses.parquet, get ketto3infohansyokunum3 (BMS = dam's sire)
        # 2. From horses.parquet, get MNum (dam kettonum) via n_sanku
        # 3. Use horse_career_stats.parquet to compute dam offspring stats
        # 4. Use sire_career_stats.parquet for BMS extended features
        ...
```

### Course Record Feature
```python
# Source: n_record schema analysis (48 cols)
# RecInfoKubun=1: course records, keyed by (jyocd, trackcd, kyori)
# RecTime: record time in m:ss.s format
def compute_record_time_ratio(
    entry_df: pd.DataFrame, record_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute horse's recent time vs course record ratio."""
    # Filter to course records only
    records = record_df[record_df["recinfokubun"] == "1"].copy()
    # Get most recent record per (jyocd, trackcd, kyori)
    records = records.sort_values("makedate").groupby(
        ["jyocd", "trackcd", "kyori"]
    ).last().reset_index()
    # Compute: ratio = horse_avg_time / record_time
    # Higher ratio = slower (worse) relative to record
    ...
```

### Relative Feature (z-score)
```python
# Source: intra_race_features.py existing pattern
def compute_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Intra-race relative comparison features (TS-05)."""
    df = df.copy()

    # Base features to relativize
    _BASE_FEATURES = [
        "norm_finish_logit_avg",  # Past performance strength
        "harontimel5_avg",        # Late speed
        "timediff_avg",           # Average time differential
        "blood_total_wr",         # Bloodline quality
        "sire_wr",                # Sire quality
    ]

    for feat in _BASE_FEATURES:
        if feat not in df.columns:
            continue
        # z-score: how many std above/below race mean
        grp = df.groupby("race_id", observed=True)[feat]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, 1)
        df[f"rel_{feat}_zscore"] = (df[feat] - mean) / std

    return df
```

## Detailed PIT Audit: n_mining (82 Columns)

### DataKubun Classification

The `DataKubun` field (column 2) is the critical PIT discriminator:
- **DataKubun=1**: "前日予想(出馬発表後)" -- Day-before prediction (after entry announcement). Available the evening before race day.
- **DataKubun=2**: "当日予想(天候馬場発表後)" -- Same-day prediction (after weather/track condition announcement). Available morning of race day.
- **DataKubun=3**: "直前予想(馬体重発表後)" -- Final prediction (after horse weight announcement). Available ~30 min before race start.

**All three tiers are PRE-race.** The distinction is the information cutoff:
- DataKubun=1: No weather/track/weight info included in prediction
- DataKubun=2: Weather and track condition included
- DataKubun=3: Weight info also included (most information-rich)

### Column-by-Column Classification

| Columns | Count | Classification | Rationale |
|---------|-------|---------------|-----------|
| RecordSpec (1) | 1 | **PRE** | Record format identifier, always "DM" |
| DataKubun (2) | 1 | **PRE** | Prediction tier indicator (1/2/3), not a result |
| MakeDate (3) | 1 | **PRE** | Data creation date, metadata |
| Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum (4-9) | 6 | **PRE** | Race identification PKs |
| MakeHM (10) | 1 | **PRE** | Creation time, metadata |
| Umaban1-18 (11,15,19,...,79) | 18 | **PRE** | Horse number slots, not results |
| DMTime1-18 (12,16,20,...,80) | 18 | **PRE** | JRA-VAN **predicted** finish time ("予想走破タイム"). NOT actual finish time. |
| DMGosaP1-18 (13,17,21,...,81) | 18 | **PRE** | Predicted +error (confidence upper bound). Pre-race estimate. |
| DMGosaM1-18 (14,18,22,...,82) | 18 | **PRE** | Predicted -error (confidence lower bound). Pre-race estimate. |

**Total PRE: 82/82 columns (100%)**
**Total POST: 0/82 columns (0%)**

### Key Observation

The n_mining table is NOT what its name might suggest ("mining" sounds like post-race analytics). It is entirely JRA-VAN's proprietary pre-race prediction engine output. The "DM" (Data Mining) refers to JRA's internal model that generates predicted finish times with confidence intervals. This is extremely valuable because:
1. It provides a strong prior from a competing model
2. The confidence intervals (DMGosaP/DMGosaM) encode uncertainty
3. DataKubun=3 incorporates weight and track condition information

### Recommended Feature Extraction Strategy

1. **Filter to DataKubun=3** (final predictions) for maximum information
2. **Pivot from wide to long format**: 18 horse columns -> 18 rows per race
3. **Compute derived features:**
   - `dm_time_rank`: Rank of predicted time within race (1=fastest predicted)
   - `dm_time_zscore`: z-score of predicted time within race
   - `dm_confidence_range`: DMGosaP + DMGosaM (total uncertainty spread)
   - `dm_time_margin_to_fav`: Gap between horse's predicted time and favorite's (rank 1) predicted time

## n_hansyoku + n_sanku Schema Analysis

### n_hansyoku (19 columns) -- Breeding Horse Master

| Column | Field Name | PIT | Usefulness |
|--------|-----------|-----|-----------|
| 1 | RecordSpec | PRE | Metadata |
| 2 | DataKubun | PRE | 1=new, 2=update, 0=delete |
| 3 | MakeDate | PRE | Metadata |
| 4 | **HansyokuNum** (PK) | PRE | Breeding registration number. Links to n_sanku.KettoNum |
| 5 | reserved | PRE | Unused |
| 6 | **KettoNum** | PRE | Bloodline registration number. Links to horses.kettonum |
| 7 | DelKubun | PRE | Deletion flag |
| 8 | Bamei | PRE | Horse name (text) |
| 9 | BameiKana | PRE | Horse name kana |
| 10 | BameiEng | PRE | Horse name English |
| 11 | **BirthYear** | PRE | Birth year (useful for age computation) |
| 12 | **SexCD** | PRE | Sex code (useful for filtering mares) |
| 13 | HinsyuCD | PRE | Breed code |
| 14 | KeiroCD | PRE | Coat color |
| 15 | HansyokuMochiKubun | PRE | Domestic/import classification |
| 16 | ImportYear | PRE | Import year |
| 17 | **SanchiName** | PRE | Production area name |
| 18 | **HansyokuFNum** | PRE | Sire (father) breeding number |
| 19 | **HansyokuMNum** | PRE | Dam (mother) breeding number |

**All 19 columns are PRE (static master data).**

### n_sanku (26 columns) -- Offspring Master

| Column | Field Name | PIT | Usefulness |
|--------|-----------|-----|-----------|
| 1-3 | RecordSpec, DataKubun, MakeDate | PRE | Metadata |
| 4 | **KettoNum** (PK) | PRE | Bloodline registration number (same horse as n_hansyoku.KettoNum) |
| 5 | **BirthDate** | PRE | Birth date |
| 6 | **SexCD** | PRE | Sex code |
| 7 | HinsyuCD | PRE | Breed code |
| 8 | KeiroCD | PRE | Coat color |
| 9 | SankuMochiKubun | PRE | Domestic/import |
| 10 | ImportYear | PRE | Import year |
| 11 | **BreederCode** | PRE | Breeder code (links to n_seisan) |
| 12 | **SanchiName** | PRE | Production area |
| 13 | **FNum** | PRE | Father breeding number (= sire) |
| 14 | **MNum** | PRE | Mother breeding number (= dam) |
| 15 | FFNum | PRE | Sire's sire |
| 16 | FMNum | PRE | Sire's dam |
| 17 | **MFNum** | PRE | Dam's sire (= BMS!) |
| 18 | MMNum | PRE | Dam's dam |
| 19-26 | FFFNum..MMMNum | PRE | 3rd generation ancestors |

**All 26 columns are PRE (static master data).**

### Cross-Reference Chain for Dam Features

```
racing horse
  -> kettonum (from entries)
  -> horses.ketto3infohansyokunum3 (BMS = dam's sire)
  -> horses.ketto3infohansyokunum2 (dam kettonum, indirectly via n_sanku.MNum)

For dam offspring stats:
  racing horse kettonum
  -> horses.kettonum -> n_sanku.KettoNum -> n_sanku.MNum (dam's breeding number)
  -> Find all n_sanku entries where MNum matches this dam
  -> Those entries are the dam's offspring
  -> Aggregate their career stats from horse_career_stats.parquet

For BMS extended features:
  racing horse kettonum
  -> horses.ketto3infohansyokunum3 (BMS kettonum)
  -> sire_career_stats.parquet (already loaded by SireFeatures)
  -> Compute bms_distance_wr, bms_surface_wr using same Beta smoothing
```

### Proposed Bloodline Features (from n_hansyoku + n_sanku)

| Feature Name | Source | Calculation | Target Model |
|-------------|--------|-------------|-------------|
| `dam_wr` | horse_career_stats via dam offspring | Beta(1,10) smoothed win rate of dam's offspring | Stage1 |
| `dam_surface_wr` | horse_career_stats via dam offspring | Surface-specific win rate of dam's offspring | Stage1 |
| `bms_distance_wr` | sire_career_stats (BMS lookup) | Distance-specific win rate for BMS | Stage1 |
| `bms_surface_wr` | sire_career_stats (BMS lookup) | Surface-specific win rate for BMS | Stage1 |
| `dam_prize_log` | horse_career_stats via dam offspring | log(1 + avg offspring prize) | Stage1 |

**Implementation note:** `bms_distance_wr` and `bms_surface_wr` should be added to `sire_features.py` since the BMS lookup logic is already there. The dam offspring features should go in the new `dam_pedigree_features.py` module.

## n_record Schema Analysis

### n_record (48 columns) -- Record Master

| Column | Field Name | PIT | Usefulness |
|--------|-----------|-----|-----------|
| 1-3 | RecordSpec, DataKubun, MakeDate | PRE | Metadata |
| 4 | **RecInfoKubun** (PK) | PRE | **1=course record, 2=GI record** (critical filter) |
| 5-10 | Year, MonthDay, JyoCD, Kaiji, Nichiji, RaceNum (PK) | PRE | Race identification |
| 11 | TokuNum (PK) | PRE | Special race number (GI record key only) |
| 12 | Hondai | PRE | Race name |
| 13 | GradeCD | PRE | Grade code |
| 14 | **SyubetuCD** (PK) | PRE | Race type code (flat/jump) |
| 15 | **Kyori** (PK) | PRE | Distance in meters |
| 16 | **TrackCD** (PK) | PRE | Track code (turf/dirt direction) |
| 17 | RecKubun | PRE | 1=reference, 2=record, 3=note, 4=remark |
| 18 | **RecTime** | PRE | **Record time (m:ss.s format)** |
| 19-21 | TenkoCD, SibaBabaCD, DirtBabaCD | PRE | Weather/track condition at record time |
| 22-39 | RecUmaKettoNum1-3, Bamei1-3, etc. | PRE | Record holder horse info (up to 3 holders) |
| 40-48 | RecUmaKettoNum3, etc. | PRE | 3rd record holder info |

**All 48 columns are PRE (historical static data).**

### Proposed Record Features

| Feature Name | Source | Calculation | Target Model |
|-------------|--------|-------------|-------------|
| `record_time_ratio` | n_record.RecTime + race kyori | horse_avg_haron_time / course_record_time for same (jyocd, trackcd, kyori). Higher = slower. | Stage1 |
| `record_track_delta` | n_record.RecTime | Difference between track record and distance-record. Captures track-specific speed bias. | Stage1 |

**Important:** The record features require the horse's actual past times to compare against records. Since past times come from HorseHistoryFeatures (harontimel5_avg), these features must be computed after HorseHistoryFeatures has run -- i.e., in `_train_submodel()`, not in `build_all()`.

**Alternative simpler approach:** Instead of comparing horse time to record, compute a track-difficulty index from the record data alone: the ratio of course record to distance-record across tracks. This is a track-level feature that can be computed in `build_all()` without depending on horse history.

**Recommendation:** Implement both:
1. `course_record_time` as a race-level feature in `build_all()` (the absolute record time for the race's track/distance/surface)
2. `harontimel5_vs_record` as a horse-level feature in `_train_submodel()` (ratio of horse's avg late time to the record)

## Relative Features Design (TS-05)

### Proposed 7 Relative Features

| # | Feature Name | Base Feature | Transform | Rationale |
|---|-------------|-------------|-----------|-----------|
| 1 | `rel_norm_finish_zscore` | norm_finish_logit_avg | z-score within race | Captures relative past performance strength |
| 2 | `rel_haron_vs_mean` | harontimel5_avg | (value - race_mean) / race_std | Captures relative late speed |
| 3 | `rel_timediff_rank` | timediff_avg | rank within race (ascending) | Captifies relative time differential |
| 4 | `rel_blood_quality_rank` | blood_total_wr | rank within race (descending) | Captures relative bloodline quality |
| 5 | `rel_sire_quality_rank` | sire_wr | rank within race (descending) | Captures relative sire quality |
| 6 | `rel_weight_zscore` | weight_zscore | already z-scored, re-compute vs race mean | Re-contextualize weight within race field |
| 7 | `rel_closing_index_rank` | closing_index_avg | rank within race (ascending) | Captures relative closing ability |

### Computation Method

Use `groupby("race_id").transform()` for vectorized computation:
- **z-score features:** `(x - group_mean) / group_std` -- captures how many standard deviations above/below race mean
- **rank features:** `.rank(method="min", ascending=...)` -- captures ordinal position within race
- **vs_mean features:** `x - group_mean` -- captures absolute difference from race average

### NaN Handling

- For z-score: if `group_std == 0` (all same value in race), output 0.0
- For rank: NaN values get NaN rank (consistent with LightGBM NaN handling)
- Skip computation if base feature column is missing from DataFrame

### Integration Point

Add `compute_relative_features()` call in `_train_submodel()` after all per-horse features are computed (after HorseHistoryFeatures, PaceAptitudeFeatures, CourseFeatures, SireFeatures, and InteractionFeatures), but before the Market Model. This ensures all base features are available.

The same function should also be callable from `build_all()` for features that don't depend on HorseHistoryFeatures outputs (e.g., `rel_blood_quality_rank`, `rel_sire_quality_rank`).

**Recommendation:** Add to `build_all()` after bloodline and sire features, but the primary call remains in `_train_submodel()` after hist features are added.

## Integration Plan

### Step 1: ETL Execution
```bash
python scripts/run_etl.py --mode full --tables n_hansyoku n_record n_mining --start 20140101 --end 20251231
```
Also need n_sanku (already defined in etl_tables.yaml):
```bash
python scripts/run_etl.py --mode full --tables n_sanku --start 20140101 --end 20251231
```

### Step 2: New Module Files
1. `src/features/dam_pedigree_features.py` -- Dam pedigree + BMS extended features
2. `src/features/record_features.py` -- Course record ratio features
3. `src/features/mining_features.py` -- JRA-VAN prediction features
4. `src/features/relative_features.py` -- Intra-race relative comparison features

### Step 3: Integration Points

| Module | Called From | When | Dependencies |
|--------|-----------|------|-------------|
| DamPedigreeFeatures | build_all() | After BloodlineFeatures | hansyoku.parquet, sanku.parquet, horse_career_stats.parquet, sire_career_stats.parquet |
| RecordFeatures | _train_submodel() | After CourseFeatures | record.parquet, harontimel5_avg (from HorseHistoryFeatures) |
| MiningFeatures | _train_submodel() | After all per-horse features | mining.parquet |
| compute_relative_features | _train_submodel() | After all per-horse features | Existing feature columns |
| BMS extension (sire_features.py) | _train_submodel() | Within existing SireFeatures block | sire_career_stats.parquet |

### Step 4: FEATURE_COLS Updates

**Stage1AbilityModel.FEATURE_COLS additions** (~12 new features):
```python
# Bloodline extension (from dam_pedigree_features.py)
"dam_wr",
"dam_surface_wr",
# BMS extension (from sire_features.py extension)
"bms_distance_wr",
"bms_surface_wr",
# Record features (from record_features.py)
"course_record_time",          # Absolute record time for track/distance
"harontimel5_vs_record",       # Horse's late time / record time ratio
# Mining features (from mining_features.py)
"dm_time_rank",
"dm_time_zscore",
"dm_confidence_range",
# Relative features (from relative_features.py)
"rel_norm_finish_zscore",
"rel_haron_vs_mean",
"rel_sire_quality_rank",
```

**WinTwoStageModel.RETURN_FEATURE_COLS additions** (~3-4 features):
```python
"dm_time_rank",          # Strong prior for market comparison
"dm_confidence_range",   # Prediction uncertainty
"rel_norm_finish_zscore", # Relative strength within race
```

**PlaceTwoStageModel.HIT_FEATURE_COLS additions** (~4-5 features):
```python
"dam_wr",
"bms_distance_wr",
"dm_time_rank",
"rel_norm_finish_zscore",
"rel_haron_vs_mean",
```

### Step 5: POST_RACE Leakage Verification

New features do NOT touch any of the 16 POST_RACE_COLS:
- `kakuteijyuni` -- not used
- `confirmed_odds` -- not used (dm_time is pre-race prediction)
- `ninki` -- not used
- `kyakusitukubun` -- not used
- `time` -- not used
- `timediff` -- not used (our `timediff_avg` comes from HorseHistoryFeatures which is PIT-safe)
- `harontimel3/4` -- not used
- `jyuni1c/2c/3c/4c` -- not used (jyuni1c_avg etc. come from HorseHistoryFeatures)
- `honsyokin` -- not used
- `chakusacd` -- not used
- `dmjyuni` -- not used (DataMining finish position prediction -- but this is NOT in n_mining table, it's in a different context)
- `dmtime` -- not used (DataMining time -- similarly separate from n_mining.DMTime which is a prediction)

The Phase 23 CI test (`tests/test_post_race_leakage.py`) will automatically verify that new FEATURE_COLS have no overlap with POST_RACE_COLS.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | n_mining DataKubun=3 records exist for all races in training period | n_mining PIT audit | Some races may only have DataKubun=1 or 2; fallback logic needed |
| A2 | n_record has entries for all (jyocd, trackcd, kyori) combinations in training data | n_record features | Some rare track/distance combos may lack records; NaN handling needed |
| A3 | n_hansyoku.KettoNum can be linked to racing horses via horses.kettonum | dam pedigree features | Some foreign-bred horses may not have entries in n_hansyoku |
| A4 | n_sanku.MNum provides the dam's KettoNum that links to horse_career_stats | dam pedigree features | Need to verify this chain works end-to-end with actual data |

**All assumptions are LOW risk** -- they affect feature coverage (NaN rates) but not correctness. LightGBM handles NaN natively.

## Open Questions

1. **n_mining historical coverage:** What percentage of races in the training period (2014-2025) have DataKubun=3 records? If coverage is low for early years, the feature may be NaN-heavy.
   - What we know: JRA-VAN provides mining data as part of their standard DataLab offering
   - What's unclear: Whether DataKubun=3 was available from the start or added later
   - Recommendation: After ETL, run a coverage check: `mining_df.groupby('DataKubun').size()`

2. **n_record update frequency:** How often are course records updated? Do we need year-specific records or can we use the all-time record?
   - What we know: Records have a MakeDate field indicating when they were set
   - What's unclear: Whether using the all-time record (including future records at training time) creates lookahead bias
   - Recommendation: Use PIT-safe approach -- filter to records with MakeDate < race_date

3. **Dam offspring aggregation performance:** Aggregating dam offspring stats requires finding all offspring for each dam in horse_career_stats. With ~100K horses this could be slow.
   - What we know: horse_career_stats has ~500K rows; n_sanku has ~100K rows
   - What's unclear: How many offspring per dam on average
   - Recommendation: Precompute dam offspring stats as a separate Parquet file (similar to sire_career_stats pattern)

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PostgreSQL (EveryDB2) | ETL | LOCAL ONLY | 15.x | -- |
| Python 3.11 | All | Yes | 3.11 | -- |
| pandas | Feature computation | Yes | 2.x | -- |
| pyarrow | Parquet I/O | Yes | 14+ | -- |

**Missing dependencies with no fallback:**
- PostgreSQL with EveryDB2 data is required for ETL step. This must be run by user locally.

**Missing dependencies with fallback:**
- None (all computation dependencies are available)

## Sources

### Primary (HIGH confidence)
- `docs/everydb2/44-MINING.md` -- n_mining 82-column schema, DataKubun definition [VERIFIED: codebase read]
- `docs/everydb2/34-HANSYOKU.md` -- n_hansyoku 19-column schema [VERIFIED: codebase read]
- `docs/everydb2/35-SANKU.md` -- n_sanku 26-column schema [VERIFIED: codebase read]
- `docs/everydb2/36-RECORD.md` -- n_record 48-column schema [VERIFIED: codebase read]
- `docs/everydb2/53-KEITO.md` -- n_keito bloodline system codes [VERIFIED: codebase read]
- `src/features/bloodline_features.py` -- BloodlineFeatures pattern [VERIFIED: codebase read]
- `src/features/sire_features.py` -- SireFeatures pattern + BMS lookup [VERIFIED: codebase read]
- `src/features/intra_race_features.py` -- Intra-race feature pattern [VERIFIED: codebase read]
- `src/features/feature_engine.py` -- build_all() integration pattern [VERIFIED: codebase read]
- `src/models/stage1_ability_model.py` -- FEATURE_COLS 89 features [VERIFIED: codebase read]
- `src/models/two_stage_return_model.py` -- WinTwoStage/PlaceTwoStage FEATURE_COLS [VERIFIED: codebase read]
- `config/etl_tables.yaml` -- ETL table configuration [VERIFIED: codebase read]
- `src/pipelines/training_pipeline.py:346-500` -- _train_submodel() integration pattern [VERIFIED: codebase read]

### Secondary (MEDIUM confidence)
- `.planning/research/FEATURES.md` -- Feature landscape analysis, TS-05/D-01/D-02/D-08 recommendations [CITED: existing research]
- `.planning/phases/26-everydb2-new-features/26-CONTEXT.md` -- Phase decisions [CITED: phase context]

### Tertiary (LOW confidence)
- None -- all claims verified against codebase and documentation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new packages needed, all verified in codebase
- Architecture: HIGH -- integration pattern follows existing _train_submodel() flow exactly
- Pitfalls: HIGH -- identified from schema analysis and existing codebase patterns
- n_mining PIT audit: HIGH -- all 82 columns verified as PRE via docs/everyDB2/44-MINING.md
- n_hansyoku/n_sanku analysis: HIGH -- all columns verified as static master data
- n_record analysis: HIGH -- all columns verified as historical static data

**Research date:** 2026-05-14
**Valid until:** 2026-06-14 (stable -- table schemas don't change)
