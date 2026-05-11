# Architecture: Feature Engineering Overhaul (v1.6)

**Project:** keiba-ai v1.6 -- Feature Engineering Overhaul
**Researched:** 2026-05-10
**Scope:** How feature audit, new feature addition, and interaction engineering integrate with the existing ML pipeline without disrupting the data flow or model structure.
**Confidence:** HIGH (verified against full source code)

## Executive Summary

The v1.6 milestone overhauls feature engineering within the existing pipeline architecture. The system already has a well-structured, modular feature pipeline: `FeatureEngine.build_all()` orchestrates 14 feature modules that generate 100+ columns, consumed downstream by `AbilityModel` (Stage1, ~50 features) and `WinTwoStageModel` (Stage2, ~37 features). The overhaul has three workstreams: (1) audit and prune noisy features from the existing 100+ set, (2) extract new features from unused EveryDB2 tables, and (3) engineer feature interactions and transformations.

The architecture requires no structural changes. Feature modules are pure functions (input DataFrame -> output DataFrame with new columns) that chain through `build_all()` and `_train_submodel()`. Adding or removing features means changing module logic and updating `FEATURE_COLS` lists in model files. The critical constraint is the Point-in-Time (PIT) safety protocol: every new feature must use only data available before race start. The existing `leakage_validators.py` framework enforces this.

The recommended build order follows the dependency chain: audit first (to establish a clean baseline), then new features from unused data (to expand signal), then interactions (which depend on clean base features). Each phase validates against backtest ROI to ensure no regression.

## Recommended Architecture

### Current Feature Pipeline Data Flow

```
TrainingPipelineV5.run()
  |
  +-> FeatureEngine.build_all(race_df, entry_df, odds_df, odds_ts_df, store)
  |     |
  |     +-> [MERGE] race_df + entry_df + odds_df -> result_df
  |     +-> [EXCLUDE] steeple races (trackcd >= 51)
  |     +-> _map_basic_features(result_df)           # 12 basic feature mappings
  |     |
  |     +-> compute_intra_race_features(result_df)    # B: weight_diff, odds_rank
  |     +-> compute_odds_dynamics(result_df, ts_df)   # C: 7 odds dynamics features
  |     +-> compute_market_bias(result_df)             # D: p_market, entropy, overround
  |     +-> compute_flb_slope(result_df)               # D: skewness, HHI
  |     +-> compute_difficulty_score(result_df)        # E: difficulty_score
  |     +-> BloodlineFeatures(store).compute(result_df) # B: 6 bloodline features
  |     |
  |     +-> [CACHE WRITE] -> features/cache/feat_*.parquet
  |
  +-> SubModelManager.add_distance_band_features(feat_df)  # F: surface/distance one-hot
  |
  v
TrainingPipelineV5._train_submodel(df, surface)
  |
  +-> HorseHistoryFeatures.compute(race_df, entry_df, target_race_ids)  # ~45 horse features
  +-> HorseHistoryFeatures.add_race_transforms(df)                       # 5 race_rank features
  +-> PaceAptitudeFeatures(store).compute_batch(df)                      # 6 pace features
  +-> CourseFeatures(store).compute_batch(df)                            # 2 course features
  +-> SireFeatures(sire_stats).compute_batch(df)                         # 5 sire features
  +-> compute_interaction_features(df)                                    # 9 interaction features
  |
  +-> MarketModel.train/predict_oof(df)          # market signed/abs log_error
  +-> AbilityModel.train_oof(df)                 # Stage1: p_ability_win
  +-> odds_to_ability_ratio computation          # FEAT-02 derived feature
  +-> compute_odds_deviation_features(df)        # ODDS-01: deviation_rank, deviation_zscore
  +-> PlaceAbilityModel.train/predict(df)
  +-> WinTwoStageModel: hit + return stages      # Stage2: p_win_pred, ev_win
  +-> EVCorrectionModel + ConformalEVModel
  +-> WinSelectionGateModel / PlaceSelectionGateModel
  |
  v
BacktestEngine / RacePredictor (inference path mirrors training path)
```

### Feature Module Inventory

| Module | File | Category | Features Generated | Used By |
|--------|------|----------|-------------------|---------|
| `_map_basic_features` | `feature_engine.py` | A: Basic | distance_bin, grade_code, class_level, field_size, popularity_rank, draw_ratio, frame_number, blinker_on, weight_change_zone, weight_change_ratio | AbilityModel, WinTwoStageModel |
| `compute_intra_race_features` | `intra_race_features.py` | B: Intra-race | weight_diff_from_mean, odds_rank | AbilityModel |
| `compute_odds_dynamics` | `odds_dynamics_features.py` | C: Odds dynamics | odds_drop_rate_60_10, odds_drop_rate_30_10, odds_velocity, odds_volatility, odds_acceleration, odds_direction_consistency, popularity_change_30_10 | WinTwoStageModel |
| `compute_market_bias` | `market_bias_features.py` | D: Market bias | p_market_win_adj, market_entropy, overround | AbilityModel, WinTwoStageModel, RegimeDetector |
| `compute_flb_slope` | `market_bias_features.py` | D: Market shape | odds_skewness, implied_prob_hhi | WinTwoStageModel |
| `compute_difficulty_score` | `race_difficulty_model.py` | E: Difficulty | difficulty_score | AbilityModel |
| `BloodlineFeatures` | `bloodline_features.py` | B: Bloodline | blood_surface_wr, blood_distance_wr, blood_condition_wr, blood_total_wr, blood_prize_log, blood_keito_cd | AbilityModel |
| `HorseHistoryFeatures` | `horse_history_features.py` | A: Horse history | ~45 features (norm_finish_logit_avg, harontimel5_*, form_*, class_*, jockey_*, weight_*, etc.) | AbilityModel |
| `PaceAptitudeFeatures` | `pace_aptitude_features.py` | C: Pace | pace_aptitude, front_pace_wr, closing_pace_wr, pace_corner_stability, pace_closing_power, pace_position_consistency | AbilityModel, interaction_features |
| `CourseFeatures` | `course_features.py` | D: Course | course_wr, course_distance_wr | AbilityModel |
| `SireFeatures` | `sire_features.py` | B: Sire | sire_wr, sire_surface_wr, sire_distance_wr, sire_prize_avg, bms_wr | AbilityModel |
| `compute_interaction_features` | `interaction_features.py` | E: Interaction | kyakusitu_x_distance, kyakusitu_x_surface, weight_x_distance, race_mean_fuku_odds, race_std_fuku_odds, odds_gap_fav12, odds_popularity_gap, surface_track_interaction, pace_pressure, closer_share, pace_scenario_fit, actual_pace_fit | AbilityModel, WinTwoStageModel |
| `compute_odds_deviation` | `odds_deviation_features.py` | F: Deviation | deviation_rank, deviation_zscore | WinTwoStageModel |
| `compute_form_features` | `form_cycle_features.py` | B: Form cycle | form_trend, form_consistency, form_peak_flag | (via HorseHistoryFeatures) |
| `compute_class_trajectory` etc. | `high_odds_features.py` | A: High-odds | class_promotions/demotions, v_recovery, time/position_improvement_rate, env_adaptability (9) | (via HorseHistoryFeatures) |

### Target Architecture for v1.6

```
Phase 1: FEATURE AUDIT (no architecture changes)
  |
  +-> FeatureAuditTool (NEW SCRIPT, not a module)
  |     - Extract feature importance from trained models (gain, split, SHAP)
  |     - Compute permutation importance on OOF predictions
  |     - Identify zero-importance and noise features
  |     - Generate prune_candidates.json
  |
  +-> Prune features from FEATURE_COLS in model files
        - AbilityModel.FEATURE_COLS (Stage1, ~50 features)
        - WinTwoStageModel.FEATURE_COLS (Stage2, ~37 features)
        - WinTwoStageModel.HIT_FEATURE_COLS (Stage2 hit sub-model)
        - WinTwoStageModel.RETURN_FEATURE_COLS (Stage2 return sub-model)
  |
  v
Phase 2: NEW FEATURES from unused EveryDB2 data
  |
  +-> JockeyTrainerContextFeatures (EXISTING, partially used)
  |     Currently available but NOT wired into _train_submodel()
  |     Wire: jockey_context_features.py + trainer_context_features.py
  |
  +-> RaceContextFeatures (NEW MODULE)
  |     Source: n_toku_race, n_schedule, n_hyosu (vote counts)
  |     Features: tokubetsu_race_flag, vote_total, vote_concentration
  |
  +-> HorsePhysicalFeatures (NEW MODULE, expand existing)
  |     Source: n_uma (horses master), n_uma_race existing columns
  |     Features: age_at_race, sex_encoding, distant_past_form_features
  |
  +-> JockeyTrainerComboFeatures (EXISTING, unused)
  |     Currently available but NOT wired into _train_submodel()
  |     Wire: jockey_trainer_combo.py
  |
  v
Phase 3: INTERACTION & TRANSFORMATION FEATURES
  |
  +-> expand compute_interaction_features()
  |     - Horse-vs-horse relative features (normalized gap within race)
  |     - Conditional interactions (surface x form, class x distance)
  |     - Polynomial features for key continuous variables
  |
  +-> Target encoding for high-cardinality categoricals
        - blood_keito_cd, kisyucode, chokyosicode
        - PIT-safe: expanding mean with shift(1)
```

## Component Boundaries

### Components to MODIFY

| Component | File | Change | Scope | Risk |
|-----------|------|--------|-------|------|
| `AbilityModel` | `models/stage1_ability_model.py` | Update `FEATURE_COLS` list (prune/add features) | List edit only | LOW -- LightGBM handles missing columns gracefully |
| `WinTwoStageModel` | `models/two_stage_return_model.py` | Update `FEATURE_COLS`, `HIT_FEATURE_COLS`, `RETURN_FEATURE_COLS` | List edit only | LOW -- same |
| `HorseHistoryFeatures` | `features/horse_history_features.py` | Add/remove features in `BASE_COLS` and `compute()` loop | Medium -- 1300-line file, per-horse loop | MEDIUM -- changes affect all downstream models |
| `compute_interaction_features` | `features/interaction_features.py` | Expand with new interaction features | Medium -- new functions | LOW -- additive only |
| `_train_submodel` | `pipelines/training_pipeline.py` | Wire new feature modules into the pipeline | ~20 lines per module | MEDIUM -- insertion point order matters |
| `build_all` | `features/feature_engine.py` | Potentially add new module calls for batch-level features | ~10 lines per module | LOW -- additive |

### Components to CREATE

| Component | File | Purpose | Dependencies |
|-----------|------|---------|-------------|
| `FeatureAuditScript` | `scripts/run_feature_audit.py` | Extract importance, identify noise features | Trained models, OOF predictions |
| `RaceContextFeatures` | `features/race_context_features.py` | Extract from n_toku_race, n_schedule, n_hyosu | ParquetStore, existing readers |
| `HorsePhysicalFeatures` | `features/horse_physical_features.py` | Age, sex, extended career stats | n_uma master table |
| `RelativeFeatureBuilder` | `features/relative_features.py` | Horse-vs-horse comparison features within race | Existing features from other modules |

### Components UNCHANGED

| Component | Why Unchanged |
|-----------|--------------|
| `ParquetStore` | I/O layer, feature-agnostic |
| `DataRepository` | Data access layer, feature-agnostic |
| `StackedEnsemble` | Model layer, consumes whatever features are provided |
| `EVCorrectionModel` | Model layer, features are independent |
| `RegimeDetector` | Uses race-level features, not horse-level |
| `RaceQualityScreener` | Uses race-level features |
| `DrawdownController` | Betting layer, feature-independent |
| `StrategyOptimizer` | Betting strategy, feature-independent |
| `BacktestEngine` | Orchestrates pipeline, feature-agnostic |
| `ConformalEVModel` | Model layer, features are independent |
| `ETL pipeline` | Data source unchanged |
| `all odds modules` | No changes to odds processing |
| `leakage_validators.py` | Framework exists, new features must pass it |

## Data Flow Changes

### CHANGE 1: Feature Audit and Pruning

No data flow changes. Pruning removes columns from model `FEATURE_COLS` lists. The feature modules still compute all features, but models simply ignore the pruned columns. This is safe because LightGBM silently ignores missing columns.

**Why not remove from modules too:** Keeping feature computation intact preserves the option to re-add features later. The cost of computing unused features is minimal compared to the risk of breaking the pipeline.

### CHANGE 2: New Features from Unused EveryDB2 Data

New feature modules follow the existing pattern: pure functions or classes with `compute()` methods that take a DataFrame and return a DataFrame with new columns.

**Insertion points in the pipeline:**

```
build_all() insertion (batch-level, before surface split):
  - RaceContextFeatures: race-level features available to all surfaces
  - HorsePhysicalFeatures: horse-level features from master tables

_train_submodel() insertion (after HorseHistoryFeatures):
  - JockeyContextFeatures: jockey annual stats (Stage2 feature)
  - TrainerContextFeatures: trainer annual stats (Stage2 feature)
  - JockeyTrainerComboFeatures: combo statistics (Stage2 feature)
```

**EveryDB2 data sources not yet used for features:**

| Table | Parquet Key | Potential Features | PIT Safety |
|-------|-------------|-------------------|------------|
| `n_toku_race` | `toku_race` | Special race metadata (prize, conditions) | SAFE -- pre-race |
| `n_toku` | `toku` | Special race details per entry | SAFE -- pre-race |
| `n_hyosu` | `hyosu` | Total vote count per race | SAFE -- pre-race snapshot |
| `n_hyosu_tanpuku` | `hyosu_tanpuku` | Vote count per horse (popularity proxy) | SAFE -- pre-race snapshot |
| `n_kisyu_seiseki` | `kisyu_seiseki` | Jockey annual stats (already has module, unused) | SAFE -- SetYear < race_year |
| `n_chokyo_seiseki` | `chokyo_seiseki` | Trainer annual stats (already has module, unused) | SAFE -- SetYear < race_year |
| `n_jogaiba` | `jogaiba` | Late scratch/changes info | PARTIAL -- depends on timing |
| `n_mining` | `mining` | Pre-computed analytics (index values) | NEEDS AUDIT -- verify pre-race |
| `n_uma` | `horses` | Horse master (sex, birth year for age) | SAFE -- static |
| `n_hansyoku` | `hansyoku` | Breeding details | SAFE -- static |
| `n_bameiorigin` | `bameiorigin` | Extended pedigree | SAFE -- static |
| `n_record` | `record` | Course records | SAFE -- historical |
| `n_schedule` | `schedule` | Race schedule metadata | SAFE -- pre-race |

### CHANGE 3: Interaction and Transformation Features

Interactions are added in `compute_interaction_features()`, which already runs after all base features are computed. This is the correct insertion point because it has access to all feature columns.

**New interaction categories:**

1. **Relative features (horse-vs-horse within race):**
   - Gap between horse's norm_finish_logit and race average
   - Gap between horse's harontimel5_avg and race best
   - Gap between horse's p_ability_win and race max (requires Stage1 output)

2. **Conditional interactions:**
   - surface x form_trend (form on specific surface)
   - class_level x distance_change (class change at specific distance)
   - weight_change_zone x rest_category (weight pattern after rest)

3. **Target encoding (PIT-safe):**
   - blood_keito_cd target-encoded win rate (expanding mean with shift(1))
   - kisyucode target-encoded win rate (expanding, not used in Stage1 to avoid leak)

## Patterns to Follow

### Pattern 1: Pure Function Feature Modules

**What:** Feature modules are pure functions or classes with `compute()` methods that take a DataFrame and return a DataFrame with new columns. No side effects on the input.

**When:** All feature modules follow this pattern.

**Why:** Makes modules composable, testable, and reorderable. The pipeline chains them without coupling.

**Example (existing pattern):**
```python
def compute_intra_race_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Never mutate input
    df["weight_diff_from_mean"] = ...
    return df
```

### Pattern 2: PIT-Safe Time-Series Features

**What:** Any feature using historical data must use `expanding().shift(1)` or `searchsorted(side="left")` to exclude the current race.

**When:** All features computed from past race results (horse history, jockey stats, sire stats, etc.).

**Why:** Including current-race results in features is look-ahead bias that inflates backtest ROI but fails in live betting.

**Example (existing pattern in HorseHistoryFeatures):**
```python
# searchsorted with side="left" excludes current-race date
target_date_np = np.datetime64(race_date, "ns")
idx = valid_dates.searchsorted(target_date_np, side="left")
start = max(0, idx - self._n_past)
# Only use data BEFORE idx (never includes current race)
```

### Pattern 3: Feature Module Caching

**What:** `build_all()` uses SHA-256 cache key based on input file paths and date range. If inputs unchanged, cache hit returns pre-computed features.

**When:** Batch feature generation in `build_all()`.

**Implication for new features:** Adding new features to `build_all()` changes the output columns, automatically invalidating the cache (the cache key does not include feature columns, so manual cache invalidation may be needed if only adding columns).

**Caution:** The cache key is based on input file paths and date range, NOT on the feature computation code. Adding new feature modules requires manually clearing the feature cache (`data/features/cache/`).

### Pattern 4: Two-Stage Feature Consumption

**What:** Features flow through two model stages with different feature subsets.

**When:** All features flow through Stage1 (AbilityModel, ~50 features) then Stage2 (WinTwoStageModel, ~37 features).

**Implication:** New features must be added to the correct `FEATURE_COLS` list:
- Horse-level features (history, bloodline, physical) -> `AbilityModel.FEATURE_COLS` (Stage1)
- Race-level/market features (odds dynamics, market bias, interactions) -> `WinTwoStageModel.FEATURE_COLS` (Stage2)
- Some features appear in both stages (e.g., `distance_bin`, `surface`)

### Pattern 5: Beta-Smoothed Win Rates

**What:** Win rates computed from small samples use Beta prior smoothing: `(alpha + wins) / (alpha + beta + starts)`.

**When:** All win-rate features (bloodline, sire, pace, course, jockey, trainer).

**Why:** Raw win rates from 3-5 starts are unreliable. Beta smoothing shrinks toward the prior mean.

**Constants:** alpha=1, beta=10 (used consistently across all modules).

## Anti-Patterns to Avoid

### Anti-Pattern 1: Using Post-Race Data in Features

**What:** Including `kakuteijyuni`, `odds` (confirmed odds), `time`, `honsyokin` in features for the current race.

**Why bad:** These values are only available after the race. Using them creates look-ahead bias that inflates backtest ROI but fails in production.

**Instead:** The codebase already uses `confirmed_odds` vs `tanodds` separation. New features must follow the same discipline. The `POST_RACE_COLS` constant in `domain/types.py` lists all post-race columns that must never appear in features.

**Detection:** Run `leakage_validators.validate_no_future_leakage()` after adding any new expanding-window feature.

### Anti-Pattern 2: Adding Features Without Checking Model Capacity

**What:** Adding 50+ new features without pruning existing noise features.

**Why bad:** LightGBM handles high dimensionality well, but adding noise features (zero importance) can dilute signal in small-sample splits. With ~9K training races and surface submodels, each surface gets ~4.5K races. Adding many noisy features increases overfitting risk in tree splits.

**Instead:** Audit first, prune noise, then add new features incrementally with backtest validation.

### Anti-Pattern 3: Computing Expensive Features in the Per-Horse Loop

**What:** Adding database queries or complex computations inside `HorseHistoryFeatures.compute()` per-horse loop.

**Why bad:** The per-horse loop iterates ~500-2000 times per surface. Each microsecond of overhead per horse adds seconds to total pipeline time.

**Instead:** Pre-load all data before the loop (as done with `past_by_ketto_arr`, `expanding_stats`). Use vectorized numpy operations within the loop.

### Anti-Pattern 4: Target Encoding Without PIT Safety

**What:** Computing target-encoded features using the full training data (including the current row).

**Why bad:** Target encoding with current-row inclusion is equivalent to look-ahead bias. The model sees information about the outcome in the feature.

**Instead:** Use `expanding().shift(1)` or leave-one-out encoding that excludes the current row. The `info_asymmetry_features.py` module demonstrates the correct pattern.

### Anti-Pattern 5: Modifying Feature Computation Without Cache Invalidation

**What:** Adding new features to a module, then running training with feature cache enabled.

**Why bad:** The cache key is based on input files, not on computation code. If the cache is stale, training uses old features without the new additions. This silently produces incorrect results.

**Instead:** Delete `data/features/cache/` when adding or modifying feature modules. Or set `use_cache=False` in `FeatureEngine.__init__()` during development.

## Detailed Integration Points

### Integration Point 1: Feature Audit Script

**New file:** `scripts/run_feature_audit.py`

The audit script loads trained models and their OOF predictions, then:
1. Extracts `feature_importance("gain")` from each LightGBM model
2. Computes permutation importance on OOF predictions (Stage1 and Stage2)
3. Optionally computes SHAP values for top features
4. Identifies features with zero importance across all models
5. Identifies features with negative permutation importance (noise)
6. Outputs `data/feature_audit/prune_candidates.json`

**Data requirements:**
- Trained models from `data/models/` or `data/models-backtest/`
- OOF predictions (must re-run training pipeline to generate)
- Feature DataFrame (from `FeatureEngine.build_all()`)

**Does NOT require:** Changes to any existing modules. Script-only.

### Integration Point 2: Jockey/Trainer Context Features (Wiring Existing Modules)

**Files to modify:** `src/pipelines/training_pipeline.py`

The `JockeyContextFeatures` and `TrainerContextFeatures` modules already exist in `src/features/` but are NOT wired into the training pipeline. They use `kisyu_seiseki` and `chokyo_seiseki` tables (already ETL'd to Parquet).

**Insertion point:** `_train_submodel()`, after `SireFeatures` and before `compute_interaction_features`:

```python
# Group D: Jockey context features (Stage2 only -- uses annual jockey stats)
from features.jockey_context_features import JockeyContextFeatures
with TimingContext(f"{surface}/jockey_context"):
    jockey_feat = JockeyContextFeatures(store=self.store)
    jockey_df = jockey_feat.compute_batch(df)
    df = df.merge(jockey_df, on=["race_id", "umaban"], how="left")

# Group D: Trainer context features (Stage2 only -- uses annual trainer stats)
from features.trainer_context_features import TrainerContextFeatures
with TimingContext(f"{surface}/trainer_context"):
    trainer_feat = TrainerContextFeatures(store=self.store)
    trainer_df = trainer_feat.compute_batch(df)
    df = df.merge(trainer_df, on=["race_id", "umaban"], how="left")
```

**PIT safety:** Both modules already enforce `SetYear < race_year` (year-before comparison).

### Integration Point 3: Horse-vs-Horse Relative Features

**New file:** `src/features/relative_features.py`

Relative features compute within-race comparisons. These run AFTER all per-horse features are computed.

```python
def compute_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Horse-vs-horse relative features within each race."""
    df = df.copy()

    # Gap between horse's ability and race mean
    for col in ["norm_finish_logit_avg", "harontimel5_avg", "harontimel5_zscore"]:
        if col in df.columns:
            race_mean = df.groupby("race_id", observed=True)[col].transform("mean")
            df[f"{col}_vs_mean"] = df[col] - race_mean

            race_max = df.groupby("race_id", observed=True)[col].transform("max")
            df[f"{col}_vs_max"] = df[col] - race_max

    return df
```

**Insertion point:** `_train_submodel()`, after `compute_interaction_features` (needs all base features).

**Model update:** Add to `AbilityModel.FEATURE_COLS` (Stage1 features that capture relative ability).

### Integration Point 4: Race Context Features from Unused Tables

**New file:** `src/features/race_context_features.py`

```python
class RaceContextFeatures:
    """Features from n_hyosu (vote totals), n_toku_race (special race metadata)."""

    def __init__(self, store: ParquetStore) -> None:
        self.store = store

    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        # vote concentration from hyosu_tanpuku
        # special race flags from toku_race
        ...
```

**Data sources:** `hyosu_tanpuku` (per-horse vote count), `toku_race` (special race metadata), `schedule` (race schedule data).

**PIT safety:** Vote counts (`n_hyosu_tanpuku`) are pre-race snapshots. `toku_race` is pre-race metadata. Both are safe.

**Insertion point:** `FeatureEngine.build_all()` -- these are race-level features available before surface split.

## Build Order (Dependency-Driven)

### Phase 1: Feature Audit and Pruning
**Dependencies:** Existing trained models
**Changes:**
- New script: `scripts/run_feature_audit.py`
- Modify `FEATURE_COLS` in `AbilityModel`, `WinTwoStageModel` (and sub-lists)
**Verification:** Backtest ROI does not decrease after pruning (noise removal should improve or maintain ROI)
**Estimated effort:** Small

### Phase 2A: Wire Existing Unused Feature Modules
**Dependencies:** Phase 1 (clean baseline)
**Changes:**
- Modify `_train_submodel()` to call `JockeyContextFeatures.compute_batch()` and `TrainerContextFeatures.compute_batch()`
- Add feature columns to appropriate `FEATURE_COLS` lists
**Verification:** Backtest ROI improves
**Estimated effort:** Small (modules exist, just need wiring)

### Phase 2B: New Features from Unused EveryDB2 Tables
**Dependencies:** Phase 1
**Changes:**
- New module: `features/race_context_features.py` (vote concentration, special race metadata)
- New module: `features/horse_physical_features.py` (age, sex, extended career)
- Wire into `build_all()` and `_train_submodel()` as appropriate
- Add to `FEATURE_COLS` lists
**Verification:** Incremental backtest ROI improvement per feature group
**Estimated effort:** Medium

### Phase 3: Interaction and Transformation Features
**Dependencies:** Phase 2 (needs complete base feature set)
**Changes:**
- Expand `compute_interaction_features()` with relative features, conditional interactions
- Optionally add target encoding for high-cardinality categoricals
- Add to `FEATURE_COLS` lists
**Verification:** Backtest ROI improvement, interaction features appear in importance ranking
**Estimated effort:** Medium

### Phase 4: Validation and Cleanup
**Dependencies:** Phases 1-3
**Changes:**
- Run walk-forward validation to detect overfitting from new features
- Run Optuna strategy re-optimization with new feature set
- Final backtest validation targeting ROI > 100%
**Verification:** WF validation passes, final backtest ROI > 100%
**Estimated effort:** Medium (mainly compute time)

## Modified vs New Components Summary

### Modified Components (7 files)

| File | Change | LOC Impact |
|------|--------|-----------|
| `models/stage1_ability_model.py` | Update `FEATURE_COLS` | ~5 lines |
| `models/two_stage_return_model.py` | Update `FEATURE_COLS`, `HIT_FEATURE_COLS`, `RETURN_FEATURE_COLS` | ~15 lines |
| `features/horse_history_features.py` | Possibly add/remove features from `BASE_COLS` and `compute()` | ~20-50 lines |
| `features/interaction_features.py` | Expand with relative/conditional interactions | ~50-80 lines |
| `pipelines/training_pipeline.py` | Wire new modules into `_train_submodel()` | ~30 lines |
| `features/feature_engine.py` | Optionally add new batch-level module calls in `build_all()` | ~10 lines |
| `features/horse_history_features.py` | Update `BASE_COLS` list | ~5 lines |

### New Components (3-4 files)

| File | Purpose | LOC Estimate |
|------|---------|-------------|
| `scripts/run_feature_audit.py` | Feature importance analysis and noise identification | ~150 lines |
| `features/race_context_features.py` | Vote concentration, special race flags | ~100 lines |
| `features/horse_physical_features.py` | Age, sex, extended career features | ~120 lines |
| `features/relative_features.py` | Horse-vs-horse within-race comparisons | ~80 lines |

## Scalability Considerations

| Concern | Current (100+ features) | After v1.6 (potentially 150+ features) | Mitigation |
|---------|------------------------|---------------------------------------|------------|
| Feature computation time | ~17 min training | ~18-20 min (new features are cheap) | Feature cache handles batch features; per-horse loop is the bottleneck |
| Memory per surface | ~4.5K races x 100 cols | ~4.5K races x 150 cols | +50% memory but still fits in RAM easily |
| LightGBM training time | ~5 min per surface | ~6-7 min per surface | LightGBM handles 150 features efficiently; early stopping prevents overfitting |
| Per-horse loop (HorseHistoryFeatures) | ~100s for ~2000 horses | ~120s if adding 5 features to loop | New features should be vectorized where possible; avoid DB queries in loop |
| Feature cache size | ~200 MB | ~300 MB | Manageable; cache invalidation is manual |
| Backtest time per year | ~41 min | ~45 min | Acceptable increase |

## Sources

- Code analysis: `src/features/feature_engine.py` -- orchestrator, build_all() flow, cache mechanism
- Code analysis: `src/features/horse_history_features.py` -- per-horse loop, PIT safety via searchsorted
- Code analysis: `src/features/intra_race_features.py` -- pure function pattern
- Code analysis: `src/features/odds_dynamics_features.py` -- vectorized computation pattern
- Code analysis: `src/features/market_bias_features.py` -- market feature computation
- Code analysis: `src/features/bloodline_features.py` -- PIT-safe bloodline features from career stats
- Code analysis: `src/features/pace_aptitude_features.py` -- vectorized batch computation with cumulative sums
- Code analysis: `src/features/course_features.py` -- course-specific features with searchsorted
- Code analysis: `src/features/sire_features.py` -- sire features with Beta smoothing
- Code analysis: `src/features/interaction_features.py` -- interaction/pace features
- Code analysis: `src/features/high_odds_features.py` -- class trajectory, form improvement, env adaptability
- Code analysis: `src/features/leakage_validators.py` -- PIT safety verification framework
- Code analysis: `src/features/jockey_context_features.py` -- existing but unwired jockey features
- Code analysis: `src/features/trainer_context_features.py` -- existing but unwired trainer features
- Code analysis: `src/models/stage1_ability_model.py` -- Stage1 FEATURE_COLS (50 features)
- Code analysis: `src/models/two_stage_return_model.py` -- Stage2 FEATURE_COLS (37 features), hit/return split
- Code analysis: `src/pipelines/training_pipeline.py` -- pipeline orchestration, feature module wiring
- Code analysis: `config/etl_tables.yaml` -- 103 EveryDB2 tables available for feature extraction
- Code analysis: `.planning/PROJECT.md` -- v1.6 milestone context
