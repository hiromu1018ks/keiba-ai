# Architecture: Turf Precision Calibration (v1.8)

**Project:** keiba-ai v1.8 -- Turf Precision Calibration
**Researched:** 2026-05-19
**Scope:** Integration architecture for 5 features: ETL extraction (HaronTimeL4 + LapTime), popularity band EV calibration, haron/lap feature computation, new interaction features, and regime-surface EV propagation.
**Confidence:** HIGH (verified against full source code of etl.py, feature_engine.py, horse_history_features.py, interaction_features.py, ev_correction_model.py, race_predictor.py, backtest/engine.py, regime_detector.py, domain/models.py, domain/types.py)

## Executive Summary

The v1.8 milestone introduces five features that tighten turf-specific predictions. Four features are additive extensions to existing pipeline stages; one (regime propagation) bridges two previously disconnected components. The critical architectural invariant is PIT (point-in-time) safety: harontimel4 and LapTime1~25 are POST_RACE information, meaning they are only known after a race finishes. Feature computation must use past-race data exclusively, following the expanding_stats + searchsorted pattern already established in horse_history_features.py.

The recommended build order is ETL first (data availability), then feature computation (C + D in parallel), then EV calibration layers (B + E), then validation. This follows the natural dependency chain: model training cannot use features until the data exists and is computed correctly.

## Recommended Architecture

### Integration Overview

```
EXISTING PIPELINE (v1.8 additions marked with [+]):

ETL Layer (src/db/etl.py)
  |
  +-> N_UMA_RACE -> entries.parquet
  |     harontimel3: already ETL'd as float
  |     harontimel4: [+] add to "float" type rules
  |
  +-> N_RACE -> races.parquet
  |     LapTime1~25: [+] add to "float" type rules
  |
  v
Feature Layer (src/features/)
  |
  +-> feature_engine.py::build_all()
  |     |
  |     +-> ... existing modules ...
  |     +-> horse_history_features.py (existing harontimel3 features)
  |     +-> [+] lap_features.py::compute_lap_features()  (Feature C)
  |     +-> interaction_features.py::compute_interaction_features()
  |     |     +-> [+] grade_x_form_score (Feature D)
  |     |     +-> [+] distance_x_closing_index (Feature D)
  |     +-> ... existing modules continue ...
  |
  v
Model Layer (src/models/)
  |
  +-> ev_correction_model.py::correct_ev()
  |     |-- existing: Isotonic calibration + OddsBand scaling
  |     |-- [+] Popularity band scaling (Feature B)
  |     |-- [+] regime_state feature (Feature E)
  |
  v
Betting Layer (src/backtest/)
  |
  +-> race_predictor.py::predict()
  |     |-- existing: calls correct_ev() without regime context
  |     |-- [+] inject regime_state into DataFrame before correct_ev() (Feature E)
  |
  +-> engine.py::run() per-race loop
        |-- existing: detects regime, uses for stake/skip only
        |-- [+] regime passed to predictor for EV correction (Feature E)
```

### Component Boundaries

| Component | Type | File | Responsibility | Communicates With |
|-----------|------|------|---------------|-------------------|
| ETL type rules | MODIFIED | `src/db/etl.py` | Type conversion for new columns | Parquet files |
| DB reader columns | MODIFIED | `src/db/readers.py` | Float column whitelist for history load | DataRepository |
| Lap feature module | NEW | `src/features/lap_features.py` | PIT-safe harontimel4 + LapTime features | FeatureEngine, RacePredictor |
| History features | MODIFIED | `src/features/horse_history_features.py` | Extend with harontimel4 stats | lap_features.py (shared pattern) |
| Interaction features | MODIFIED | `src/features/interaction_features.py` | Add grade_x_form, distance_x_closing | FeatureEngine, RacePredictor |
| Feature engine | MODIFIED | `src/features/feature_engine.py` | Wire lap_features into build sequence | lap_features.py |
| EV correction model | MODIFIED | `src/models/ev_correction_model.py` | Add popularity band + regime features | SubmodelSet |
| SubmodelSet | MODIFIED | `src/domain/models.py` | Add ev_popularity_band_scales field | TrainingPipeline |
| Training pipeline | MODIFIED | `src/pipelines/training_pipeline.py` | Wire popularity band computation + regime features | ev_correction_model.py |
| Race predictor | MODIFIED | `src/backtest/race_predictor.py` | Inject regime into DataFrame | RegimeDetector output |
| Popularity band filter | NEW/EXTENDED | `src/betting/odds_band_filter.py` | Band boundary definitions | ev_correction_model.py |

## Data Flow

### Flow A: ETL Data Extraction

```
EveryDB2 PostgreSQL
  |
  +-> N_UMA_RACE table (SE recordspec)
  |     columns: harontimel3, harontimel4, ...
  |     harontimel3: already in _TABLE_TYPE_RULES["entries"]["float"]
  |     harontimel4: currently in POST_RACE_COLS but NOT in type rules
  |                   [+] add to _TABLE_TYPE_RULES["entries"]["float"]
  |
  +-> N_RACE table (RA recordspec)
        columns: LapTime1 ~ LapTime25, ...
        LapTime columns: NOT in any current type rules
                         [+] add "laptime1"..."laptime25" to
                             _TABLE_TYPE_RULES["races"]["float"]
        |
        v
    data/raw/entries.parquet  (harontimel4 now float64)
    data/raw/races.parquet    (laptime1~25 now float64)
        |
        v
    DataRepository.load_history_entries()
        returns DataFrame with harontimel4 in past-race rows
        (also need [+] harontimel4 in _FLOAT_COLS in readers.py)
```

**Key detail:** harontimel4 already appears in `domain/types.py` POST_RACE_COLS (line-level confirmation), meaning the system already recognizes it as a post-race column. It is also in `_race_entry_shared` in feature_engine.py (line 278), so it gets dropped from the current-race merge. The only gap is the ETL type rule that converts it from raw string to float64 during Parquet export.

**LapTime table source:** LapTime columns belong to the RA (N_RACE) table, not SE (N_UMA_RACE). The `config/etl_tables.yaml` maps N_RACE to "races". The column names follow the pattern `laptime1` through `laptime25` (lowercase, as per EveryDB2 naming convention). If the actual column names differ, the ETL type rules must match exactly. This should be verified against the actual database schema before implementation.

### Flow B: Popularity Band EV Calibration

```
Training time (in TrainingPipelineV5._train_submodel):
    |
    +-> After EV correction model training
    +-> After backtest bet_history generation
    |
    +-> [NEW] Compute ROI per popularity_rank band:
    |     band_names = ["1-3", "4-6", "7-12", "13+"]
    |     band_boundaries = [(1, 4), (4, 7), (7, 13), (13, 19)]
    |     for each band:
    |         mask = (bet_history["popularity_rank"] >= lo) & (rank < hi)
    |         roi = sum(result) / sum(stake) for band
    |         scale = clip(roi, 0.8, 1.2)  # bounded scaling
    |     ev_popularity_band_scales = {"1-3": 0.95, "4-6": 1.02, ...}
    |
    +-> Store in SubmodelSet.ev_popularity_band_scales
    |
    v
Inference time (in EVCorrectionModel.correct_ev):
    |
    +-> After existing Isotonic calibration
    +-> After existing OddsBand scaling (lines 396-409)
    |
    +-> [NEW] Popularity band scaling (parallel layer):
    |     if self.ev_popularity_band_scales is not None:
    |         ranks = df["popularity_rank"].values
    |         for (lo, hi), band_name in zip(POP_BANDS, POP_BAND_NAMES):
    |             scale = self.ev_popularity_band_scales.get(band_name, 1.0)
    |             mask = (ranks >= lo) & (ranks < hi) & np.isfinite(ranks)
    |             calibrated[mask] *= scale
    |         df["ev_win_calibrated"] = calibrated
    |
    v
    Output: ev_win_calibrated with both odds-band and popularity-band adjustments
```

**Why parallel to OddsBand, not replacing it:** Odds band captures market-implied probability accuracy (how well the model calibrates at different odds levels). Popularity band captures a different signal -- how well the model handles horses at different public attention levels. A popular favorite and a 50:1 longshot at the same odds have different crowd behavior. Both corrections are multiplicative and independent.

### Flow C: Haron/Lap Feature Computation

```
Past-race history (from DataRepository.load_history_entries)
  |
  +-> entries: race_id, umaban, race_date, harontimel3, harontimel4, ...
  |     joined with races: laptime1~25 (via race_id)
  |
  v
lap_features.py::compute_lap_features(df, target_date, expanding_stats)
  |
  +-> PIT filter: past = history[history["race_date"] < target_date]
  |     (strict less-than, same as horse_history_features.py line 227)
  |
  +-> HaronTimeL4 features:
  |     - harontimel4_l5_avg: EMA-weighted last-5 harontimel4 (analogous to harontimel5_avg)
  |     - harontimel4_l5_zscore: expanding hierarchical z-score (analogous to harontimel5_zscore)
  |     - harontime_l3_l4_gap: avg(harontimel3) - avg(harontimel4) per horse (finish vs 3rd-last)
  |
  +-> Lap-derived features:
  |     - lap_closing_speed_avg: average of last-3f lap times / race average (past races)
  |     - lap_pace_differential: first-half pace vs second-half pace (past races)
  |     - lap_final_furlong_rank: how fast the horse's last furlong is relative to field
  |     All computed via expanding_stats + searchsorted (PIT-safe)
  |
  v
Output: ~6 new feature columns per horse
  merged on (race_id, umaban) into feature DataFrame
```

**Expanding stats pattern (from horse_history_features.py):**
```python
# Pre-compute expanding mean/std per (distance_bin, surface, baba_cd) group
for group_key, group_df in history.groupby(group_cols):
    sorted_df = group_df.sort_values("race_date")
    expanding = sorted_df["harontimel4"].astype(float).expanding()
    stats = np.column_stack([
        sorted_df["race_date"].values.astype("datetime64[ns]"),
        expanding.mean().values,
        expanding.std().values,
    ])
    expanding_stats[group_key] = stats

# Lookup at inference time (PIT-safe)
def _lookup_expanding_stats(target_date, db_val, surf_val, baba_val, expanding_stats):
    for cols, min_n in FALLBACK_LEVELS:
        key = build_key(cols, db_val, surf_val, baba_val)
        arr = expanding_stats.get(key)
        if arr is None:
            continue
        dates = arr[:, 0].astype("datetime64[ns]")
        idx = dates.searchsorted(target_date, side="left")  # strict <
        if idx > 0:
            return float(arr[idx - 1, 1]), float(arr[idx - 1, 2])
    return nan, nan
```

**LapTime availability note:** LapTime columns are race-level (one set per race, not per horse). When computing per-horse lap features, join the horse's past race entries with their corresponding race rows to get the lap profile for each past race. A horse's "closing speed" is derived from the lap profile of the races it ran in the past.

### Flow D: New Interaction Features

```
In compute_interaction_features(df):
    |
    +-> Existing: kyakusitu_x_distance, kyakusitu_x_surface, weight_x_distance, etc.
    |
    +-> [NEW] grade_x_form_score:
    |     if "form_improvement_rate" in df.columns and "grade_code" in df.columns:
    |         grade_weight = df["grade_code"].map({"A": 1.5, "B": 1.3, "C": 1.1, "D": 1.0, "E": 0.9})
    |         df["grade_x_form_score"] = (grade_weight * df["form_improvement_rate"]).where(
    |             df["form_improvement_rate"].notna() & grade_weight.notna(),
    |             other=float("nan"),
    |         )
    |
    +-> [NEW] distance_x_closing_index:
    |     if "kyori" in df.columns and "closing_index_avg" in df.columns:
    |         df["distance_x_closing_index"] = (df["kyori"] * df["closing_index_avg"]).where(
    |             df["kyori"].notna() & df["closing_index_avg"].notna(),
    |             other=float("nan"),
    |         )
    |
    v
    INTERACTION_COLS += ["grade_x_form_score", "distance_x_closing_index"]
```

**Design rationale:** `grade_x_form_score` captures whether a horse improving its form can carry that improvement into graded stakes (where competition is stronger). `distance_x_closing_index` captures whether closing ability is more valuable at longer distances (where there is more race to close into). Both are continuous numeric interactions, following the existing `weight_x_distance` pattern.

### Flow E: Regime x Surface EV Propagation

```
BacktestEngine.run() per-race loop:
    |
    +-> recent_stats_df = pd.DataFrame(recent_stats_list[-200:])
    +-> regime = models.regime_detector.detect(recent_stats_df)
    +-> regime_params = models.regime_detector.get_strategy_params(regime)
    |
    +-> CURRENT: regime used only for:
    |     - fractional_kelly (stake sizing)
    |     - COLLAPSED skip (skip betting entirely)
    |     - ev_threshold / edge_threshold (bet filtering)
    |
    +-> [NEW] Inject regime into prediction DataFrame:
    |     df["regime_state"] = regime.value  # "aggressive"/"conservative"/"collapsed"
    |
    v
RacePredictor.predict():
    |
    +-> ... existing feature computation ...
    +-> [NEW] df["regime_state"] already injected by caller
    +-> submodel.ev_corrector.correct_ev(df)
    |     FEATURE_COLS now includes "regime_state" (as category)
    |     Model learns: conservative regime -> less aggressive EV correction
    |                    aggressive regime -> more aggressive EV correction
    |                    collapsed regime -> heavy discount on EV
    |
    v
    EV correction model learns regime-dependent P/E adjustments
```

**Why add regime_state as a feature, not a parameter:** The EV correction model is trained during `TrainingPipelineV5` where regime labels can be computed for each training row (via `_build_regime_stats` in the pipeline). The model learns the relationship between regime and optimal correction. At inference time, the same feature must be present in the DataFrame. This keeps the train/inference paths consistent without changing method signatures.

**Regime feature computation during training:**
```python
# In TrainingPipeline._train_submodel():
# _build_regime_stats already constructs regime features from race-level aggregates
# Need to also assign regime_state label per row:
regime_labels = regime_detector.detect_batch(race_level_stats)  # or per-race
df_train["regime_state"] = regime_labels  # "aggressive"/"conservative"/"collapsed"
```

**Surface interaction:** The regime correction may have different effects on turf vs dirt. Since models are already split by surface (turf/dirt in `submodels: dict[str, SubmodelSet]`), the regime-surface interaction is captured implicitly -- the turf EV correction model learns turf-specific regime adjustments, and the dirt model learns dirt-specific ones. No explicit surface interaction feature is needed.

## Patterns to Follow

### Pattern 1: PIT-Safe Expanding Stats (MUST follow for Features A, C)

**What:** When computing features from POST_RACE columns (harontimel3, harontimel4, laptime, jyuni1c, jyuni4c, kakuteijyuni), use the expanding_stats + searchsorted pattern to guarantee only past data is used.

**When:** ANY feature derived from columns in `POST_RACE_COLS`.

**Existing implementation (horse_history_features.py):**
```python
# PIT filter: strict < (not <=)
past = history[history["race_date"] < target_date]

# Expanding stats: cumulative mean/std up to each point in time
for group_key, group_df in history.groupby(group_cols):
    expanding = group_df.sort_values("race_date")["harontimel3"].expanding()
    stats[group_key] = np.column_stack([dates, expanding.mean(), expanding.std()])

# PIT-safe lookup: searchsorted finds last index BEFORE target_date
idx = dates.searchsorted(target_date, side="left")
if idx > 0:
    return arr[idx - 1, 1], arr[idx - 1, 2]
```

**Apply to new features:** harontimel4 features use the same expanding_stats structure. Lap features use expanding stats on past-race lap data, grouped by (distance_bin, surface).

### Pattern 2: NaN-Safe Feature Computation (MUST follow for Features C, D)

**What:** All new features handle missing columns gracefully with guard checks and produce NaN (not 0) for missing values.

**When:** All feature computation functions.

**Example (from interaction_features.py lines 56-60):**
```python
if weight_col in df.columns and "kyori" in df.columns:
    df["weight_x_distance"] = (df[weight_col] * df["kyori"]).where(
        df[weight_col].notna() & df["kyori"].notna(),
        other=float("nan"),
    )
```

**Why NaN not 0:** LightGBM handles NaN natively (learns optimal split direction for missing values). Zero would create false signal. `fillna(0)` is prohibited for feature computation.

### Pattern 3: Parallel Calibration Layer (MUST follow for Feature B)

**What:** Add new EV correction layers as parallel multiplicative adjustments alongside existing OddsBand scaling, not as replacements.

**When:** Popularity band calibration.

**Example (extending correct_ev):**
```python
# Existing OddsBand scaling (lines 396-409):
if self.ev_odds_band_scales is not None:
    # ... apply per-odds-band scale factors ...

# NEW: Popularity band scaling (after OddsBand, parallel layer):
if self.ev_popularity_band_scales is not None:
    ranks = df["popularity_rank"].values.astype(float)
    calibrated = df["ev_win_calibrated"].values.astype(float)
    for (lo, hi), band_name in zip(POP_BANDS, POP_BAND_NAMES):
        scale = self.ev_popularity_band_scales.get(band_name, 1.0)
        if abs(scale - 1.0) < 1e-9:
            continue
        mask = (ranks >= lo) & (ranks < hi) & np.isfinite(ranks)
        calibrated[mask] *= scale
    df["ev_win_calibrated"] = calibrated
```

### Pattern 4: Extend-Not-Replace (MUST follow for Features B, C, D, E)

**What:** New features extend existing lists (INTERACTION_COLS, FEATURE_COLS, BASE_COLS) and existing methods (correct_ev) rather than creating parallel structures.

**When:** All v1.8 features.

**Examples:**
- Feature B: Add `ev_popularity_band_scales` field to existing `SubmodelSet` dataclass (not a new container)
- Feature C: Add new columns to BASE_COLS or create LAP_COLS list (not a new model class)
- Feature D: Append to INTERACTION_COLS (not a separate feature module)
- Feature E: Add "regime_state" to EVCorrectionModel.FEATURE_COLS (not a separate regime-corrector class)

### Pattern 5: Inference Path Mirroring

**What:** Every feature computed in the training path must also be computed in the inference path.

**Training paths:**
1. `TrainingPipeline._train_submodel()` -> `FeatureEngine.build_all()` + sub-modules
2. `BacktestEngine.run()` -> `FeatureEngine.build_all()` + sub-modules

**Inference path:**
3. `RacePredictor.predict()` -> direct sub-module calls

**For v1.8:** New lap features must be called in both `FeatureEngine.build_all()` and `RacePredictor.predict()` (after history merge). New interaction features are already handled because `compute_interaction_features()` is called in both paths. Regime injection must happen in both backtest engine (per-race loop) and any future live prediction path.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Using Current-Race POST_RACE Columns in Features

**What:** Computing features from the target race's own harontimel4, LapTime, or other POST_RACE columns.
**Why bad:** Target leakage. The model sees information only available AFTER the race finishes, making backtest results meaningless.
**Instead:** Only use past-race data: `past = history[history["race_date"] < target_date]`. The existing `_race_entry_shared` drop in `feature_engine.py` removes current-race harontimel3/l4. New features must only read these columns from the history (past-race) DataFrame.
**Detection:** Extend the v1.6 CI leakage detection tests to cover all new feature column names. Verify no new feature column is computable from current-race POST_RACE data.

### Anti-Pattern 2: Creating Separate Turf Model Containers

**What:** Creating a `TurfSubmodelSet` or `TurfEVCorrectionModel` class for turf-specific calibration.
**Why bad:** The existing surface-split architecture (`submodels: dict[str, SubmodelSet]` with keys "turf" and "dirt") already handles per-surface specialization. Parallel containers create confusion about which is authoritative.
**Instead:** Use the existing `SubmodelSet`. Turf-specific calibration is achieved through per-surface model training (each surface trains its own EV correction model with its own feature distributions). The `ev_popularity_band_scales` dict is trained per-surface naturally.

### Anti-Pattern 3: Modifying correct_ev() Signature for Regime

**What:** Adding `regime: RegimeState` as a method parameter to `correct_ev()`.
**Why bad:** Train/inference path inconsistency. During training, regime labels are row-level features in the DataFrame. At inference, passing regime as a separate argument creates a different code path.
**Instead:** Inject regime info into the DataFrame before calling `correct_ev()`. Add `regime_state` to `FEATURE_COLS`. The model learns the regime-adjusted correction from the feature value.

### Anti-Pattern 4: Computing Lap Features from Current-Race Laps

**What:** Reading LapTime1~25 from the current race row to compute "pace" or "closing" features.
**Why bad:** Lap times are POST_RACE. Using them for the current race is direct leakage.
**Instead:** Lap features are computed from PAST races only. Use expanding_stats over past-race lap profiles. "This horse's typical closing speed based on past races" is PIT-safe. "This horse's closing speed in today's race" is leakage.

### Anti-Pattern 5: Duplicating Band Boundary Definitions

**What:** Defining popularity band boundaries in multiple files (training_pipeline.py, ev_correction_model.py, odds_band_filter.py).
**Why bad:** Boundary mismatches between training and inference cause silent calibration errors.
**Instead:** Define boundaries once, following the `OddsBandFilter.BANDS` / `OddsBandFilter.BAND_NAMES` pattern. Add `POP_BANDS` / `POP_BAND_NAMES` as module-level constants in `odds_band_filter.py` (or a new shared location).

### Anti-Pattern 6: Modifying POST_RACE_COLS Without Updating Feature Drop

**What:** Adding new POST_RACE columns to the system without adding them to `_race_entry_shared` in feature_engine.py.
**Why bad:** Current-race POST_RACE data leaks into the feature DataFrame.
**Instead:** LapTime columns (from races table, not entries) are not in `_race_entry_shared`. They must be explicitly excluded from current-race feature computation. Add LapTime columns to the shared-column drop list or guard against them in the feature computation module.

## Detailed Integration Points

### Integration Point A: ETL Type Rules

**File:** `src/db/etl.py`

**Current state (entries type rules, lines 83-101):**
```python
"entries": {
    "int": ["umaban", "kakuteijyuni", "ninki", "kyakusitukubun", "jyuni1c", "jyuni4c", "zogenfugo"],
    "float": ["time", "bataijyu", "zogensa", "harontimel3", "timediff"],
    "odds10": ["odds"],
},
```

**Required change:**
```python
"entries": {
    "float": [...existing..., "harontimel4"],  # ADD
},
"races": {
    "float": [...existing..., "laptime1", "laptime2", ..., "laptime25"],  # ADD
},
```

**File:** `src/db/readers.py`

**Current state:** `_FLOAT_COLS` includes "harontimel3" but not "harontimel4".

**Required change:** Add "harontimel4" to `_FLOAT_COLS`.

**Verification:** After ETL, check `data/raw/entries.parquet` has harontimel4 as float64, and `data/raw/races.parquet` has laptime1~25 as float64.

### Integration Point B: Popularity Band Calibration

**File:** `src/models/ev_correction_model.py`

**Required changes:**
1. Add popularity band scaling code in `correct_ev()` method (after line 409, after OddsBand scaling)
2. Accept `ev_popularity_band_scales` in constructor (parallel to `ev_odds_band_scales`)

**File:** `src/domain/models.py`

**Required change:** Add to SubmodelSet:
```python
ev_popularity_band_scales: dict[str, float] | None = None
```

**File:** `src/pipelines/training_pipeline.py`

**Required change:** After EV correction training, compute popularity band scales from bet_history:
```python
# Compute ROI per popularity rank band
pop_scales = compute_popularity_band_scales(bet_history, pop_bands, pop_names)
submodels[surface].ev_popularity_band_scales = pop_scales
```

**File:** `src/betting/odds_band_filter.py`

**Required change:** Add band boundary constants:
```python
POP_BANDS: list[tuple[int, int]] = [(1, 4), (4, 7), (7, 13), (13, 19)]
POP_BAND_NAMES: list[str] = ["1-3", "4-6", "7-12", "13+"]
```

### Integration Point C: Lap Feature Computation

**New file:** `src/features/lap_features.py`

**Module signature:**
```python
LAP_FEATURE_COLS: list[str] = [
    "harontimel4_l5_avg",
    "harontimel4_l5_zscore",
    "harontime_l3_l4_gap",
    "lap_closing_speed_avg",
    "lap_pace_differential",
    "lap_final_furlong_rank",
]

def compute_lap_features(
    df: pd.DataFrame,
    history: pd.DataFrame,
    races_df: pd.DataFrame,  # for LapTime columns
    target_date: pd.Timestamp,
    expanding_stats: dict | None = None,
) -> pd.DataFrame:
    """PIT-safe harontimel4 and LapTime-derived features.

    All features use ONLY past-race data (race_date < target_date).
    Uses expanding_stats + searchsorted for PIT safety.
    """
```

**Integration in FeatureEngine.build_all():**
After horse_history_features merge, before interaction_features:
```python
from features.lap_features import compute_lap_features
# Note: needs history and races DataFrames passed through
# May require extending the build_all() method or computing during
# the horse_history_features.compute() call
```

**Integration in RacePredictor.predict():**
After history merge (line 81), before interaction_features (line 99):
```python
from features.lap_features import compute_lap_features
lap_feats = compute_lap_features(df, hist_df, races_df, target_date, expanding_stats)
if lap_feats is not None:
    df = df.merge(lap_feats, on=["race_id", "umaban"], how="left")
```

### Integration Point D: Interaction Features

**File:** `src/features/interaction_features.py`

**Required changes:**
1. Add to INTERACTION_COLS list:
```python
INTERACTION_COLS: list[str] = [
    ...existing 12...,
    # v1.8: turf precision interactions
    "grade_x_form_score",
    "distance_x_closing_index",
]
```

2. Add computation logic in `compute_interaction_features()`:
```python
# grade_x_form_score (turf-specific form improvement in graded stakes)
if "form_improvement_rate" in df.columns and "grade_code" in df.columns:
    _grade_weight = {"A": 1.5, "B": 1.3, "C": 1.1, "D": 1.0, "E": 0.9, "000": 0.8}
    gw = df["grade_code"].map(_grade_weight)
    df["grade_x_form_score"] = (gw * df["form_improvement_rate"]).where(
        df["form_improvement_rate"].notna() & gw.notna(),
        other=float("nan"),
    )

# distance_x_closing_index (closing ability at distance)
if "kyori" in df.columns and "closing_index_avg" in df.columns:
    df["distance_x_closing_index"] = (
        df["kyori"].astype(float) * df["closing_index_avg"]
    ).where(
        df["kyori"].notna() & df["closing_index_avg"].notna(),
        other=float("nan"),
    )
```

**No wiring changes needed** -- `compute_interaction_features()` is already called from both `feature_engine.py` and `race_predictor.py`.

### Integration Point E: Regime Propagation to EV Correction

**File:** `src/models/ev_correction_model.py`

**Required changes:**
1. Add "regime_state" to FEATURE_COLS:
```python
FEATURE_COLS: list[str] = [
    ...existing 39...,
    # v1.8: regime context
    "regime_state",
]
```

2. In `_prepare_features()`, add regime_state to categorical handling:
```python
for col in ["surface", "distance_bin", "regime_state"]:  # ADD regime_state
    if col in features.columns:
        features[col] = features[col].astype("category")
```

**File:** `src/backtest/race_predictor.py`

**Required change:** Before calling `correct_ev()` (line 154), inject regime_state:
```python
# Inject regime state for EV correction (Feature E)
if "regime_state" not in df.columns and hasattr(self, '_regime_state'):
    df["regime_state"] = self._regime_state
```

The RacePredictor constructor or predict method must receive the regime state from the caller.

**File:** `src/backtest/engine.py`

**Required change:** In the per-race loop (after regime detection, around line 898), pass regime to predictor:
```python
regime = self.models.regime_detector.detect(recent_stats_df)
# ... existing code ...
predictor._regime_state = regime.value  # Inject for EV correction
```

**File:** `src/pipelines/training_pipeline.py`

**Required change:** During EV correction model training, add regime_state labels to training DataFrame:
```python
# In _train_submodel, before ev_corrector.train():
# Compute regime labels for training data
df_train["regime_state"] = _assign_regime_labels(df_train, regime_detector)
```

## Build Order (Dependency-Driven)

### Phase 1: ETL Foundation (Feature A)

**Dependencies:** None
**Rationale:** All downstream features depend on data being available in Parquet files. This is the lowest-risk change.

**Changes:**
- `src/db/etl.py`: Add "harontimel4" to entries "float"; add "laptime1"..."laptime25" to races "float"
- `src/db/readers.py`: Add "harontimel4" to `_FLOAT_COLS`
- Run ETL to regenerate Parquet files

**Verification:** Column dtype check in Parquet files
**Effort:** Small
**Risk:** LOW -- declarative type rules, no logic change

### Phase 2: Feature Computation (Features C + D)

**Dependencies:** Phase 1 (data availability)
**Rationale:** Features C and D are independent and can be built in parallel. Both must be complete before model training can use them.

**Changes (Feature C):**
- New: `src/features/lap_features.py`
- Modified: `src/features/feature_engine.py` (wire module)
- Modified: `src/backtest/race_predictor.py` (wire module)
- Modified: `src/features/horse_history_features.py` (extend BASE_COLS)

**Changes (Feature D):**
- Modified: `src/features/interaction_features.py` (add 2 features to INTERACTION_COLS)

**Verification:** PIT leakage tests for new feature columns; feature value distributions
**Effort:** Medium
**Risk:** MEDIUM -- Feature C requires careful PIT validation

### Phase 3: EV Calibration Layers (Features B + E)

**Dependencies:** Phase 2 (new features available for model training)
**Rationale:** These modify model training and affect all betting decisions. Must be validated with A/B backtest comparison.

**Changes (Feature B):**
- Modified: `src/betting/odds_band_filter.py` (add POP_BANDS constants)
- Modified: `src/models/ev_correction_model.py` (add popularity band scaling)
- Modified: `src/domain/models.py` (add ev_popularity_band_scales field)
- Modified: `src/pipelines/training_pipeline.py` (wire scale computation)

**Changes (Feature E):**
- Modified: `src/models/ev_correction_model.py` (add regime_state to FEATURE_COLS)
- Modified: `src/backtest/race_predictor.py` (inject regime into DataFrame)
- Modified: `src/backtest/engine.py` (pass regime to predictor)
- Modified: `src/pipelines/training_pipeline.py` (add regime labels to training data)

**Verification:** Full backtest with ROI comparison against baseline; walk-forward validation
**Effort:** Medium
**Risk:** MEDIUM-HIGH -- EV correction affects all betting decisions

### Phase 4: Validation

**Dependencies:** All phases complete
**Rationale:** Final validation that all features work together without regression.

**Verification:**
- Extend v1.6 POST_RACE leakage tests for all new feature columns
- Full backtest: target ROI 97.8% -> 100%+
- Walk-forward validation for overfitting detection
- Turf-specific b_difference target: -0.004 -> positive

**Effort:** Medium (mainly compute time)
**Risk:** LOW -- validation only, no code changes

## PIT Safety Architecture

### 3-Layer Defense for POST_RACE Features

**Layer 1: Feature Drop (feature_engine.py)**
- `_race_entry_shared` drops harontimel3, harontimel4 from current-race merge
- [ADD] LapTime columns must also be excluded from current-race feature computation
- LapTime is in races table, not entries, so it is not in `_race_entry_shared`
- Guard: lap_features.py must only read from history (past-race) DataFrames

**Layer 2: Feature Computation Guard (horse_history_features.py, lap_features.py)**
- `past = history[history["race_date"] < target_date]` -- strict less-than
- expanding_stats with searchsorted guarantees no future data
- New features must follow this exact pattern

**Layer 3: CI Leakage Detection Tests**
- Existing tests verify no feature column correlates with POST_RACE information
- Must extend to cover:
  - All new lap feature column names (harontimel4_l5_avg, lap_closing_speed_avg, etc.)
  - New interaction feature columns (grade_x_form_score, distance_x_closing_index)
  - Popularity band and regime features (not POST_RACE, but verify for safety)

### LapTime-Specific PIT Guard

LapTime columns are race-level (from N_RACE table). When computing lap-derived features for a horse's past races:

1. Join past race entries with their corresponding race rows (via race_id)
2. Extract LapTime columns from the PAST race row only
3. Compute features from these past-race laps
4. NEVER read LapTime from the current (target) race row

The feature computation module must contain an explicit guard:
```python
# In lap_features.py:
def compute_lap_features(df, history, races_df, target_date, ...):
    # PIT: Only use past races
    past_races = races_df[races_df["race_date"] < target_date]
    past_entries = history[history["race_date"] < target_date]
    # Join past_entries with past_races (NOT with current race)
    ...
```

## Scalability Considerations

| Concern | Current (4yr/15K races) | At 10yr/40K races | Production |
|---------|------------------------|--------------------|------------|
| Lap feature computation | Negligible (<1min) | ~3min (linear scaling) | Pre-compute feature cache |
| Popularity band calibration | O(bet_history) ~10K rows | ~30K rows | Constant-time lookup per horse |
| Regime feature injection | Constant per race | Same | Same |
| ETL column addition | 25 new float cols in races | Same | Columnar storage handles sparsity |
| Memory for lap features | ~6 extra float cols in history | ~6 cols * 40K * 18 horses = ~4MB | Negligible |
| New feature count | ~120 -> ~130 | ~130 | LightGBM handles easily |

All features scale linearly. No architectural bottleneck introduced.

## Sources

- `src/db/etl.py` lines 83-101: `_TABLE_TYPE_RULES` for entries and races
- `src/db/readers.py`: `_FLOAT_COLS` whitelist
- `src/config/etl_tables.yaml`: N_RACE -> races, N_UMA_RACE -> entries mappings
- `src/features/feature_engine.py` lines 275-289: `_race_entry_shared` drop list
- `src/features/horse_history_features.py` lines 166-259: expanding_stats pattern and PIT guard
- `src/features/interaction_features.py` lines 1-60: INTERACTION_COLS and NaN-safe computation
- `src/models/ev_correction_model.py` lines 151-199, 336-412: FEATURE_COLS and correct_ev() flow
- `src/models/regime_detector.py` lines 52-290: regime detection and strategy params
- `src/backtest/race_predictor.py` lines 80-154: inference pipeline sequence
- `src/backtest/engine.py` lines 892-898: per-race regime detection loop
- `src/domain/models.py` lines 230-266: SubmodelSet dataclass fields
- `src/domain/types.py`: POST_RACE_COLS list (16 items including harontimel3, harontimel4)
- `src/betting/odds_band_filter.py`: OddsBandFilter.BANDS/BAND_NAMES pattern
- `src/pipelines/training_pipeline.py`: EV correction training, regime stats construction
