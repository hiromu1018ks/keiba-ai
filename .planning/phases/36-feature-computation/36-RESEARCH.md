# Phase 36: Feature Computation - Research

**Researched:** 2026-05-19
**Domain:** Feature engineering (HaronTime/LapTime history, relative ranks, interaction features, model registration)
**Confidence:** HIGH

## Summary

Phase 36 computes 12 new HaronTime features (avg/zscore/trend/race_rank across L3/L4/unified), LapTime pace features, 2 weighted_recent_form features, 3 new race_rank columns, and 3 new interaction products. All must be PIT-safe, registered in 12 model FEATURE_COLS, and computed on both training and inference paths.

The codebase has well-established patterns for every one of these operations. `HorseHistoryFeatures.compute()` uses `searchsorted(target_date, side="left")` on sorted `race_date` arrays for PIT safety, `expanding_stats` with hierarchical fallback for z-scores, and `add_race_transforms()` for race-rank computation. The interaction_features.py module uses `.where(notna)` NaN-safe numeric products. The 12-model FEATURE_COLS registration follows a mechanical pattern: add column names to each model's class-level list.

The primary complexity is in the LapTime lookup: LapTime1~25 live in `races.parquet` (RA table), not `entries.parquet` (SE table), so past-race LapTime data requires joining past entries with races data -- both already loaded in `compute()` via `_get_history()`.

**Primary recommendation:** Extend existing patterns mechanically. Each feature category maps to a clear insertion point in existing code. No new architectural components needed.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: harontime_last3f unified column with distance-based auto-selection (default threshold 2000m)
- D-02: Independent history stats (avg/zscore/trend) for L3, L4, and unified columns
- D-03: 12 HaronTime features = 4 stats x 3 columns (L3/L4/unified)
  - avg: EMA weighted (halflife=3) -- same as existing harontimel5_avg
  - zscore: expanding_stats hierarchical fallback (FALLBACK_LEVELS)
  - trend: linear regression slope of last 3 runs
  - race_rank: groupby("race_id").rank(pct=True)
- D-04: LapTime 25 columns split into 3 equal segments. n_laps = kyori/200
- D-05: pace_ratio = late segment mean pace / early segment mean pace
- D-06: LapTime history: avg/zscore/trend of pace_ratio + segment averages. PIT-safe
- D-07: weighted_recent_form uses EMA (halflife=3)
- D-08: 2 indicators: weighted_recent_form_finish (norm_finish_logit), weighted_recent_form_time (timediff)
- D-09: norm_finish_logit is field-size normalized (logit), timediff is winner time gap (seconds)
- D-10: INT-01: grade_code x form_trend (numeric product)
- D-11: INT-02: kyori x closing_index_avg (numeric product)
- D-12: INT-03: grade_code x blood_prize_log (numeric product, NaN-safe)
- D-13: 3 new race_rank cols in add_race_transforms()
- D-14: Training path = groupby("race_id").rank(), backtest path = direct .rank()
- D-15: All new features registered in 12 model FEATURE_COLS
- D-16: Both training and inference paths compute all features

### Claude's Discretion
- Distance threshold default value (before Phase 35 quality check)
- LapTime expanding_stats implementation details
- NaN handling for zero-past-race cases
- Test case design (PIT safety, dual-path parity, FEATURE_COLS completeness)
- harontime_last3f coalesce logic
- LapTime column name normalization

### Deferred Ideas (OUT OF SCOPE)
- Corner position deployment features (HLF-06)
- Pace profile classification (HLF-07)
- Closing kick index (HLF-08)
- LapTime mid-segment individual expanding_stats
- harontime_last3f distance threshold final decision (depends on Phase 35 quality check)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRF-01 | Add form_trend_race_rank, blood_total_wr_race_rank, blood_surface_wr_race_rank to add_race_transforms() | add_race_transforms() pattern verified (lines 1325-1351 of horse_history_features.py); 3 new cols appended to race_rank_cols list |
| TRF-02 | Add weighted_recent_form (last-3 EMA weighted performance) to horse_history_features.py | EMA(halflife=3) pattern exists at lines 736-753; norm_finish_logit at lines 726-733; timediff at lines 858-866 |
| TRF-03 | Register all TRF features in 12 model FEATURE_COLS | 12 model classes identified with FEATURE_COLS locations verified |
| INT-01 | Add grade_x_form_trend interaction to interaction_features.py | Numeric product pattern with .where(notna) at lines 86-135 |
| INT-02 | Add distance_x_closing_index interaction | Same pattern; kyori and closing_index_avg already in df |
| INT-03 | Add grade_x_blood_prize_log interaction | Same pattern; grade_code numeric mapping exists (_GRADE_MAP) |
| INT-04 | Register all INT features in 12 model FEATURE_COLS | Same as TRF-03 |
| HLF-01 | Compute HaronTime L3/L4/unified avg/zscore/trend PIT-safe | expanding_stats pattern at lines 584-645; searchsorted at line 669; harontimel4 column confirmed in entries parquet |
| HLF-02 | Compute HaronTime race-rank (harontime_l3/l4/unified_race_rank) | add_race_transforms() pattern; extend race_rank_cols |
| HLF-03 | Compute LapTime pace features from LapTime1~25 | LapTime1~25 in races.parquet; load_history_races() loads races with laptime columns; pace_ratio formula from D-05 |
| HLF-04 | Register all HLF features in 12 model FEATURE_COLS | Same registration pattern |
| HLF-05 | Verify both training and inference paths compute HLF features | Training: _train_submodel() line 374-377; Inference: RacePredictor.predict() line 79-96; PaperPredictor.setup() line 96-99 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| HaronTime history stats | Backend (feature computation) | -- | Historical lookup + expanding_stats in Python/numpy |
| LapTime pace features | Backend (feature computation) | -- | Cross-table join (entries + races) + statistical aggregation |
| Race-rank transforms | Backend (feature computation) | -- | groupby("race_id").rank() within compute pipeline |
| Interaction products | Backend (feature computation) | -- | Numeric/category products between existing columns |
| FEATURE_COLS registration | Model layer | -- | Class-level list constants in 12 model files |
| Training path feature compute | Pipeline layer | -- | _train_submodel() orchestrates HorseHistoryFeatures + interactions |
| Inference path feature compute | Inference layer | -- | RacePredictor.predict() + PaperPredictor.setup() |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | already installed | Vectorized stats (EMA, expanding, searchsorted) | Core pattern in horse_history_features.py |
| pandas | already installed | DataFrame manipulation, groupby, rank | Used throughout all feature modules |
| LightGBM | already installed | Model training/inference with FEATURE_COLS | 12 models use lgb.Booster |

### No New Packages Needed
This phase adds features purely through extending existing Python modules. No external package installation required.

## Package Legitimacy Audit

No new packages installed in this phase. All changes are code-level extensions to existing modules.

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
entries.parquet (HaronTimeL3/L4)     races.parquet (LapTime1~25)
         |                                      |
         v                                      v
  load_history_entries()              load_history_races()
         |                                      |
         +----> past_df merge <------------------+
                    |
                    v
         HorseHistoryFeatures.compute()
         /        |          |          \
   HaronTime   LapTime    weighted_   norm_finish
   avg/zscore/ pace_ratio  recent_    logit/timediff
   trend/L4    features    form       (existing)
        |           |          |
        v           v          v
   BASE_COLS output (dict per horse)
                    |
                    v
         add_race_transforms() --> race_rank cols
                    |
                    v
         compute_interaction_features() --> 3 new products
                    |
                    v
         12 Model FEATURE_COLS registration
```

### Recommended Project Structure
```
src/features/
  horse_history_features.py  # HLF-01/02, TRF-02 additions
  interaction_features.py    # INT-01/02/03 additions
src/models/
  stage1_ability_model.py       # FEATURE_COLS += TRF/INT/HLF
  two_stage_return_model.py     # FEATURE_COLS += TRF/INT/HLF
  ev_correction_model.py        # FEATURE_COLS += TRF/INT/HLF
  conformal_ev_model.py         # FEATURE_COLS += TRF/INT/HLF
  market_model.py               # FEATURE_COLS += TRF/INT/HLF
  place_ability_model.py        # FEATURE_COLS += TRF/INT/HLF
  race_quality_screener.py      # FEATURE_COLS += TRF/INT/HLF
  regime_detector.py            # FEATURE_COLS += TRF/INT/HLF
  stacked_ensemble.py           # No FEATURE_COLS (uses base model cols)
  wide_two_stage_model.py       # SHARED_FEATURE_COLS += TRF/INT/HLF
  ev_correction_model.py (PlaceEVCorrectionModel)  # FEATURE_COLS += TRF/INT/HLF
tests/
  test_hlf_features.py          # HLF PIT-safety + dual-path tests (NEW)
  test_trf_features.py          # TRF race_rank + weighted_form tests (NEW)
  test_int_features.py          # INT interaction product tests (NEW)
  test_post_race_leakage.py     # Existing: extend Layer 2 for new features
```

### Pattern 1: PIT-safe History Lookup (searchsorted)
**What:** Get strictly past data for a horse at a given target_date
**When to use:** Every per-horse historical feature computation
**Example:**
```python
# Source: src/features/horse_history_features.py lines 663-673
target_date_np = np.datetime64(race_date, "ns")
dates_all = horse_arrs["race_date"].astype("datetime64[ns]")
valid_mask = horse_arrs["_valid_mask"]
valid_dates = dates_all[valid_mask]
idx = valid_dates.searchsorted(target_date_np, side="left")
# idx is the cutoff: data[:idx] is strictly before target_date
start = max(0, idx - self._n_past)
hp_kakuteijyuni = horse_arrs["kakuteijyuni"][valid_mask][start:idx]
```

### Pattern 2: Expanding Stats Z-score (Hierarchical Fallback)
**What:** Compute z-scores with fallback when subgroup sample size is too small
**When to use:** HaronTime zscore for L3, L4, and unified columns
**Example:**
```python
# Source: src/features/horse_history_features.py lines 166-202
FALLBACK_LEVELS: list[tuple[list[str], int]] = [
    (["distance_bin", "surface", "baba_cd"], 50),  # L1
    (["distance_bin", "surface"], 30),              # L2
    (["distance_bin"], 20),                         # L3
    ([], 0),                                        # L4: global
]

def _lookup_expanding_stats(target_date, db_val, surf_val, baba_val, expanding_stats):
    for cols, _min_n in FALLBACK_LEVELS:
        key = tuple(col_map[c] for c in cols) if cols else ("all",)
        arr = expanding_stats.get(key)
        if arr is None or len(arr) == 0:
            continue
        dates = arr[:, 0].astype("datetime64[ns]")
        idx = dates.searchsorted(target_date, side="left")
        if idx > 0:
            return float(arr[idx - 1, 1]), float(arr[idx - 1, 2])  # mean, std
    return float("nan"), float("nan")
```

### Pattern 3: EMA Weighted Average (halflife=3)
**What:** Exponentially weighted average with halflife=3, newer data gets higher weight
**When to use:** harontimel5_avg, weighted_recent_form_finish/time
**Example:**
```python
# Source: src/features/horse_history_features.py lines 740-749
halflife = 3
decay = np.log(2) / halflife  # ~ 0.231
n_ht = len(ht_valid)
weights = (1 - decay) ** np.arange(n_ht)  # i=0 is oldest
weights = weights[::-1]  # index 0 = newest (highest weight)
weights = weights / weights.sum()
harontimel5_avg = float(np.sum(ht_valid * weights))
```

### Pattern 4: add_race_transforms (race-rank)
**What:** Compute within-race percentile ranks for numeric BASE_COLS
**When to use:** TRF-01 (add 3 new race_rank cols)
**Example:**
```python
# Source: src/features/horse_history_features.py lines 1325-1351
race_rank_cols = [
    "norm_finish_logit_avg",
    "harontimel5_avg",
    # ... existing 7 cols ...
    # NEW: form_trend, blood_total_wr, blood_surface_wr (TRF-01)
]
for col in race_rank_cols:
    if col not in df.columns:
        continue
    df[f"{col}_race_rank"] = (
        df.groupby("race_id", observed=True)[col]
        .rank(pct=True, method="average")
    )
```

### Pattern 5: Interaction Feature (NaN-safe numeric product)
**What:** Multiply two columns with NaN propagation via .where(notna)
**When to use:** INT-01, INT-02, INT-03
**Example:**
```python
# Source: src/features/interaction_features.py lines 86-135
if "pace_pressure" in df.columns and "closing_index_avg" in df.columns:
    df["pace_pressure_x_closing_index"] = (
        df["pace_pressure"] * df["closing_index_avg"]
    ).where(
        df["pace_pressure"].notna() & df["closing_index_avg"].notna(),
        other=float("nan"),
    )
```

### Anti-Patterns to Avoid
- **NaN fillna(0) in interaction products:** Must use .where(notna) to preserve NaN semantics. fillna(0) creates false signals. [VERIFIED: interaction_features.py pattern]
- **Accessing current-race LapTime for pace features:** LapTime1~25 are POST_RACE_COLS. Only past-race LapTime is PIT-safe. [VERIFIED: domain/types.py line 56]
- **Modifying FEATURE_COLS at instance level:** Must modify class-level list (e.g., AbilityModel.FEATURE_COLS), not instance. [VERIFIED: all models use class-level list]
- **Using race_date <= target_date instead of < target_date:** searchsorted side="left" gives strict < comparison. Using <= creates same-day look-ahead. [VERIFIED: horse_history_features.py line 669]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Z-score with small samples | Simple z-score (mean/std) without fallback | FALLBACK_LEVELS hierarchical expanding_stats | Small subgroups give unstable z-scores; 4-level fallback handles edge cases |
| EMA weights | Manual for-loop accumulation | numpy vectorized (1-decay)^n pattern | Same pattern as existing harontimel5_avg, proven correct |
| PIT-safe date filtering | df[df["race_date"] < target] per iteration | Pre-sorted numpy arrays + searchsorted | O(log m) vs O(m) per horse; critical for 5000+ horses |
| NaN handling in products | df[col1].fillna(0) * df[col2].fillna(0) | df[col1] * df[col2]).where(notna) | fillna(0) creates false zero-interaction signals |

**Key insight:** Every feature type needed has a proven pattern in the existing codebase. The task is purely mechanical extension.

## Common Pitfalls

### Pitfall 1: HaronTimeL4 Column Availability
**What goes wrong:** HaronTimeL4 may not be present in older parquet data (pre-Phase 35 ETL)
**Why it happens:** Phase 35 ETL added L4 float64 conversion. If ETL hasn't been re-run, column may be missing.
**How to avoid:** Guard with `_has_harontimel4 = "harontimel4" in _sample_arrs` pattern (same as existing `_has_harontimel3` check at line 570).
**Warning signs:** KeyError on "harontimel4" access in dict-of-numpy arrays.

### Pitfall 2: LapTime Column Name Casing
**What goes wrong:** LapTime columns in races.parquet may be "laptime1" (lowercase) vs "LapTime1" (mixed case)
**Why it happens:** ETL normalizes to lowercase (verified: domain/types.py line 56 uses `f"laptime{i}"`). But historical data or references may use mixed case.
**How to avoid:** Use `f"laptime{i}"` (lowercase) consistently. The _FLOAT_COLS in readers.py line 50 also uses lowercase.
**Warning signs:** KeyError when accessing laptime columns from races_hist DataFrame.

### Pitfall 3: LapTime in Past Races Only
**What goes wrong:** Using current race's LapTime for pace_ratio computation (POST_RACE leak)
**Why it happens:** LapTime is in POST_RACE_COLS. The compute() method already filters by race_date < target_date, but if LapTime columns are accessed outside this filter, leakage occurs.
**How to avoid:** Only access LapTime from past_df (already filtered by PIT). Never access current race's LapTime columns.
**Warning signs:** LapTime values appearing in features for races that haven't run yet.

### Pitfall 4: HaronTime Unified Column Coalesce Order
**What goes wrong:** If L3 and L4 have NaN in different rows, wrong fallback logic
**Why it happens:** L3 and L4 have mutual exclusivity per Phase 35 (ETL-05). But both can be NaN if the race had no haron time.
**How to avoid:** coalesce(harontimel4, harontimel3) for distance >= threshold, coalesce(harontimel3, harontimel4) for distance < threshold. Use np.where or pd.Series.combine_first.
**Warning signs:** All NaN values in harontime_last3f when one of L3/L4 should have data.

### Pitfall 5: Missing Feature in Inference Path
**What goes wrong:** New features computed in _train_submodel() but not in RacePredictor.predict()
**Why it happens:** RacePredictor has its own race_rank computation (lines 84-96) and interaction_features call (line 99). New race_rank cols and interactions must be added there too.
**How to avoid:** Mirror all feature additions in both _train_submodel() and RacePredictor.predict().
**Warning signs:** Feature column missing at inference time, causing NaN-only predictions.

### Pitfall 6: StackedEnsemble FEATURE_COLS
**What goes wrong:** StackedEnsemble doesn't have its own FEATURE_COLS -- it acts as a drop-in replacement for lgb.Booster and uses the parent model's feature columns.
**Why it happens:** StackedEnsemble inherits feature handling from the model it replaces (hit_model or return_model of WinTwoStageModel).
**How to avoid:** Do NOT add features to StackedEnsemble. Add to WinTwoStageModel.FEATURE_COLS instead.
**Warning signs:** AttributeError when trying to access StackedEnsemble.FEATURE_COLS.

### Pitfall 7: LapTime Segment Division with Non-Integer Laps
**What goes wrong:** kyori/200 may not be an integer (e.g., 1600m = 8 laps, but 1800m = 9 laps; 3-segment division of 9 = 3/3/3 is fine, but 8 = 2/3/3 needs floor division)
**Why it happens:** Not all distances are multiples of 600 (3 segments x 200m).
**How to avoid:** Use `n_laps = int(kyori / 200)`, then `seg_size = n_laps // 3`, with remainder distributed to later segments. Or use np.array_split for equal division.
**Warning signs:** IndexError when slicing LapTime array for segments.

## Code Examples

### HaronTime L4 History Stats (HLF-01)
```python
# Extends horse_history_features.py compute() per-horse loop
# Pattern mirrors existing harontimel5_avg (lines 736-753) and harontimel5_zscore (lines 788-841)

# Step 1: Add harontimel4 to cols_horse list (line 503-522)
cols_horse = [
    "race_date", "valid_field", "kakuteijyuni", "syussotosu",
    "harontimel3",
    "harontimel4",        # NEW: ETL-01 column
    "distance_bin", "surface", "baba_cd",
    "timediff", "jyuni1c", "jyuni4c", ...
]

# Step 2: Pre-compute expanding_stats for L4 (parallel to L3 at lines 584-645)
# L4 uses same FALLBACK_LEVELS as L3
_has_harontimel4 = "harontimel4" in _sample_arrs

# Step 3: In per-horse loop, compute L4 stats
if _has_harontimel4 and n_past > 0:
    ht4_raw = horse_arrs["harontimel4"][valid_mask][start:idx].astype(float)
    ht4_valid = ht4_raw[~np.isnan(ht4_raw)]
    if len(ht4_valid) > 0:
        # EMA avg (halflife=3)
        halflife = 3
        decay = np.log(2) / halflife
        weights = (1 - decay) ** np.arange(len(ht4_valid))
        weights = weights[::-1] / weights.sum()
        harontimel4_avg = float(np.sum(ht4_valid * weights))
    else:
        harontimel4_avg = float("nan")
else:
    harontimel4_avg = float("nan")
```

### HaronTime Unified Column (harontime_last3f)
```python
# In per-horse loop, after L3 and L4 stats are computed
# D-01: distance-based selection (default threshold 2000m)
current_kyori = float(getattr(row, 'kyori', 0))
DISTANCE_THRESHOLD = 2000  # default, to be confirmed after Phase 35

if current_kyori >= DISTANCE_THRESHOLD:
    # Long distance: prefer L4 (4F haron)
    harontime_last3f_raw = horse_arrs["harontimel4"][valid_mask][start:idx] \
        if _has_harontimel4 and n_past > 0 else np.array([])
    # Fallback to L3 if L4 is NaN
    if len(harontime_last3f_raw) == 0 or np.all(np.isnan(harontime_last3f_raw.astype(float))):
        ht_raw_l3 = horse_arrs["harontimel3"][valid_mask][start:idx].astype(float) \
            if _has_harontimel3 and n_past > 0 else np.array([])
        harontime_last3f_raw = ht_raw_l3
else:
    # Short/middle distance: prefer L3 (3F haron)
    ht_raw_l3 = horse_arrs["harontimel3"][valid_mask][start:idx].astype(float) \
        if _has_harontimel3 and n_past > 0 else np.array([])
    harontime_last3f_raw = ht_raw_l3
    # Fallback to L4 if L3 is NaN
    if len(harontime_last3f_raw) == 0 or np.all(np.isnan(harontime_last3f_raw)):
        ht_raw_l4 = horse_arrs["harontimel4"][valid_mask][start:idx].astype(float) \
            if _has_harontimel4 and n_past > 0 else np.array([])
        harontime_last3f_raw = ht_raw_l4
```

### LapTime Pace Features (HLF-03)
```python
# In compute(), after past_df is built (line ~470)
# past_df already has race_id from entries merged with races

# Step 1: Add laptime columns to past_df via merge with races_hist
lap_cols = [f"laptime{i}" for i in range(1, 26)]
# races_hist already loaded via _get_history() -- includes laptime1~25

# Step 2: For each horse's past races, compute pace_ratio
def _compute_pace_ratio(laptimes: np.ndarray, n_laps: int) -> float:
    """Compute pace_ratio from lap times. Returns NaN if insufficient data."""
    if n_laps < 3:
        return float("nan")
    seg_size = n_laps // 3
    remainder = n_laps % 3
    # Distribute remainder to later segments
    seg1_end = seg_size
    seg2_end = seg1_end + seg_size + (1 if remainder >= 2 else 0)
    # Actually, use np.array_split for simplicity
    segments = np.array_split(laptimes[:n_laps], 3)
    early_avg = np.nanmean(segments[0]) if len(segments[0]) > 0 else float("nan")
    late_avg = np.nanmean(segments[2]) if len(segments[2]) > 0 else float("nan")
    if early_avg > 0 and not np.isnan(early_avg) and not np.isnan(late_avg):
        return late_avg / early_avg  # < 1.0 = closing fast, > 1.0 = even pace
    return float("nan")

# Step 3: Pre-compute per-horse pace_ratio history arrays
# Add to cols_horse or compute inline from races_hist merge
```

### Interaction Features (INT-01/02/03)
```python
# In compute_interaction_features(), after existing weight_x_class block (line ~135)

# INT-01: grade_x_form_trend (grade_code x form_trend numeric product)
# grade_code needs numeric mapping; form_trend is already float
if "grade_code" in df.columns and "form_trend" in df.columns:
    _GRADE_NUMERIC = {"G1": 5, "G2": 4, "G3": 3, "OP": 2, "J.G1": 5, "J.G2": 4, "J.G3": 3}
    grade_num = df["grade_code"].map(_GRADE_NUMERIC).fillna(1.0)
    df["grade_x_form_trend"] = (grade_num * df["form_trend"]).where(
        grade_num.notna() & df["form_trend"].notna(),
        other=float("nan"),
    )

# INT-02: distance_x_closing_index (kyori x closing_index_avg)
if "kyori" in df.columns and "closing_index_avg" in df.columns:
    df["distance_x_closing_index"] = (df["kyori"] * df["closing_index_avg"]).where(
        df["kyori"].notna() & df["closing_index_avg"].notna(),
        other=float("nan"),
    )

# INT-03: grade_x_blood_prize_log (grade_code x blood_prize_log)
if "grade_code" in df.columns and "blood_prize_log" in df.columns:
    df["grade_x_blood_prize_log"] = (grade_num * df["blood_prize_log"]).where(
        grade_num.notna() & df["blood_prize_log"].notna(),
        other=float("nan"),
    )
```

### Race-Rank Extension (TRF-01)
```python
# In add_race_transforms(), extend race_rank_cols list
race_rank_cols = [
    "norm_finish_logit_avg",
    "harontimel5_avg",
    "harontimel5_zscore",
    "timediff_avg",
    "jyuni1c_avg",
    "jyuni4c_avg",
    "closing_index_avg",
    # TRF-01: New race-rank columns
    "form_trend",
    "blood_total_wr",
    "blood_surface_wr",
]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Simple mean of last-N haron times | EMA weighted avg (halflife=3) | v1.6 TSER-01 | More responsive to recent form |
| Global z-score for haron time | Hierarchical expanding_stats (4 levels) | v1.6 TSER-01 | Stable z-scores even with small subgroups |
| L3 only haron time stats | L3 + L4 + unified (distance-based) | Phase 36 (this) | Covers both 3F and 4F haron time data |
| No LapTime features | LapTime pace_ratio history | Phase 36 (this) | Captures race pace pattern from past runs |

**Deprecated/outdated:**
- `harontime_late_trend` (last-2 vs first-3): Still valid, but D-03 adds linear regression trend as complement

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | harontimel4 column exists in entries.parquet after Phase 35 ETL | HLF-01 | If ETL not re-run, column absent; guard with _has_harontimel4 check |
| A2 | LapTime1~25 columns exist in races.parquet as lowercase "laptimeN" | HLF-03 | Verified in domain/types.py line 56 and readers.py line 50; LOW risk |
| A3 | blood_total_wr and blood_surface_wr columns are available when add_race_transforms() is called | TRF-01 | Bloodline features computed in build_all() before hist_df merge; HIGH confidence |
| A4 | StackedEnsemble does not need separate FEATURE_COLS registration | INT-04 | Verified: StackedEnsemble uses parent model's features; no class-level FEATURE_COLS |
| A5 | HaronTime L3/L4 mutual exclusivity holds (from Phase 35 ETL-05) | HLF-01 | If both present in same race, unified column uses distance-based preference |

## Open Questions

1. **Distance threshold for harontime_last3f**
   - What we know: Default 2000m per D-01. Phase 35 quality check will verify L3/L4 distribution.
   - What's unclear: Exact threshold may change after Phase 35 analysis.
   - Recommendation: Implement with configurable threshold constant. Default 2000m.

2. **LapTime availability in historical data**
   - What we know: Phase 35 ETL adds LapTime1~25 to races.parquet. load_history_races() reads all columns.
   - What's unclear: How many past races actually have valid LapTime data (sentinel-000 NaN rate).
   - Recommendation: Add NaN rate logging in compute(). Expect high NaN rate for older races.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | All | Yes (mise) | 3.11.x | -- |
| numpy | Feature computation | Yes | installed | -- |
| pandas | Feature computation | Yes | installed | -- |
| LightGBM | Model FEATURE_COLS | Yes | installed | -- |
| pytest | Testing | Yes | installed | -- |

**Missing dependencies with no fallback:** none
**Missing dependencies with fallback:** none

## Sources

### Primary (HIGH confidence)
- `src/features/horse_history_features.py` -- Full source read. All patterns (searchsorted, expanding_stats, EMA, add_race_transforms) verified at line level.
- `src/features/interaction_features.py` -- Full source read. NaN-safe product pattern verified.
- `src/features/feature_engine.py` -- Full source read. Orchestration flow verified.
- `src/domain/types.py` -- POST_RACE_COLS verified (line 38-57). LapTime column naming confirmed.
- `src/models/stage1_ability_model.py` -- FEATURE_COLS verified (28-164).
- `src/models/two_stage_return_model.py` -- FEATURE_COLS verified (48-171).
- `src/models/ev_correction_model.py` -- FEATURE_COLS verified for both Win and Place models (151-199, 430-477).
- `src/models/conformal_ev_model.py` -- FEATURE_COLS verified (81-162). Most comprehensive list.
- `src/models/market_model.py` -- FEATURE_COLS verified (39-60).
- `src/models/place_ability_model.py` -- FEATURE_COLS verified (26-80).
- `src/models/race_quality_screener.py` -- FEATURE_COLS verified (37-80).
- `src/models/regime_detector.py` -- FEATURE_COLS verified (60-89).
- `src/models/wide_two_stage_model.py` -- SHARED_FEATURE_COLS verified (58-80).
- `src/models/stacked_ensemble.py` -- No FEATURE_COLS (uses parent model features). Verified.
- `src/backtest/race_predictor.py` -- Inference path verified (lines 51-100). Race-rank and interaction computation confirmed.
- `src/pipelines/training_pipeline.py` -- Training path verified (_train_submodel lines 354-505).
- `src/paper_trading/predictor.py` -- PaperPredictor path verified (setup method lines 42-120).
- `src/db/readers.py` -- load_history_entries/races verified. _FLOAT_COLS includes harontimel4 and laptime1~25.

### Secondary (MEDIUM confidence)
- `.planning/phases/35-etl-data-foundation/35-CONTEXT.md` -- Phase 35 decisions about ETL output format (read from CONTEXT.md canonical_refs).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- No new packages. All patterns verified in existing codebase.
- Architecture: HIGH -- All insertion points identified at line level. Both training and inference paths verified.
- Pitfalls: HIGH -- Known issues identified from codebase patterns and POST_RACE constraints.

**Research date:** 2026-05-19
**Valid until:** 2026-06-19 (stable codebase, no external dependencies)
