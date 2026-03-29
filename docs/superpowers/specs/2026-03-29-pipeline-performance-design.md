# Pipeline Performance Optimization Design

**Date:** 2026-03-29
**Scope:** Training pipeline + Backtest pipeline
**Constraint:** Model prediction results must remain identical (pure speed optimization)
**Approach:** Code-level optimization only (no external frameworks)

## Problem Statement

Current pipeline execution times:
- `run_train.py`: ~68 min (22 LightGBM models, sequential)
- `run_backtest.py`: ~80 min (includes re-training + per-race simulation)

Development cycle is slow due to:
1. Python for-loops in feature computation (HorseHistoryFeatures, BloodlineFeatures, etc.)
2. Per-race feature recomputation in backtest
3. Sequential model training (turf and dirt are independent but run serially)
4. Repeated data loading across instances

## Identified Bottlenecks

| Priority | Bottleneck | Impact | Type |
|----------|-----------|--------|------|
| P0 | HorseHistoryFeatures — iterrows() per horse | Training x2 + Backtest x all races | Compute |
| P1 | Backtest: HorseHistoryFeatures re-instantiated per race | Backtest overall | Memory/IO |
| P2 | 22 models trained sequentially (turf/dirt independent) | Training overall | Parallelism |
| P3 | BloodlineFeatures — iterrows() | Training | Compute |
| P4 | JockeyContext/TrainerContext — iterrows() | Training | Compute |
| P5 | WidePairBuilder — O(n^2) nested Python loops | Training | Compute |

## Design

### S1: HorseHistoryFeatures Vectorization

**File:** `src/features/horse_history_features.py`

Replace `iterrows()` loop (lines 229-391) with vectorized pandas operations:

**Horse features** (norm_finish_logit_avg, haron_time_l3_avg, etc.):
1. Sort `past_df` by `(ketto_num, race_date)`
2. Use `groupby("ketto_num")` with rolling/shift for last-3-race stats
3. Merge with target horses using `pd.merge_asof()` to enforce `race_date < target_date`

**Jockey features** (jockey_surprise, jockey_cond_wr):
1. `groupby("kisyu_code")` with rolling window for jockey history stats
2. Vectorized Beta smoothing instead of per-row function calls

**`_norm_finish_logit`**: Convert from `math.log` to `np.log` for vectorized operation.

**Result identity:** Same numerical computation, vectorized. Float64 precision difference is negligible (< 1e-15).

### S2: Backtest Single-Pass Feature Computation

**File:** `src/backtest/engine.py`

Replace per-race instantiation (lines 139-144):

```python
# Before (per race):
hist = HorseHistoryFeatures(repo=self.repo)
hist_df = hist.compute(self._race_df, self._entry_df, [race_id])

# After (once before loop):
hist = HorseHistoryFeatures(repo=self.repo)
hist_all = hist.compute(self._race_df, self._entry_df)
# Loop: merge from pre-computed hist_all
```

Similarly pre-compute JockeyContextFeatures and TrainerContextFeatures before the race loop.

Pre-index `hist_all` by `race_id` for O(1) lookup during loop iterations.

**Result identity:** `target_race_ids` filter only affects which horses are selected, not how features are computed. Removing the filter produces identical features.

### S3: Parallel Model Training

**File:** `src/pipelines/training_pipeline.py`

Replace sequential surface loop (lines 102-111) with `concurrent.futures.ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {}
    for surface in ["turf", "dirt"]:
        subset_df = feat_df[feat_df["surface"] == surface].copy()
        if len(subset_df) >= 1000:
            futures[executor.submit(self._train_submodel, subset_df)] = surface
    for future in as_completed(futures):
        models[futures[future]] = future.result()
```

**Why threads, not processes:**
- LightGBM is C++ and releases GIL during training
- No DataFrame serialization cost (shared memory)
- Instance caches in `self.repo` remain valid

**Thread safety considerations:**
- Each `_train_submodel` operates on its own `subset_df` copy (already `.copy()`)
- LightGBM models are created fresh within each call
- `self.repo` caches are read-only after initial load (safe for concurrent reads)

**LightGBM `num_threads` adjustment:**
- If CPU has N cores, set `num_threads = max(1, N // 2)` per surface
- Prevents oversubscription from ThreadPoolExecutor + OpenMP

**Result identity:** LightGBM training is deterministic given fixed `random_state`. Thread scheduling does not affect model parameters.

### S4a: BloodlineFeatures Vectorization

**File:** `src/features/bloodline_features.py`

Replace `iterrows()` loop (lines 91-138) with vectorized column operations:

```python
# Convert all ba1chakukaisu columns at once
ba_cols = [f"ba1chakukaisu{i}" for i in range(1, 7)]
merged[ba_cols] = merged[ba_cols].fillna(0).astype(int)
ba1_wins = merged["ba1chakukaisu1"]
ba1_total = merged[ba_cols].sum(axis=1)
result["blood_surface_wr"] = (ba1_wins + 1) / (ba1_total + 11)
```

Same pattern for kyori1 (distance), chuo (total), and ruikeihonsyoheiti (prize).

### S4b: Jockey/Trainer Context Vectorization

**Files:** `src/features/jockey_context_features.py`, `src/features/trainer_context_features.py`

Replace `iterrows()` with `groupby().rolling()` pattern (same approach as S1).

### S4c: WidePairBuilder Optimization

**File:** `src/models/wide_pair_builder.py`

Replace inner double loop with `itertools.combinations`:
- Use pre-extracted numpy arrays for `umaban`, `finish_pos`, `popularity_rank`, `running_style`
- Build pair dicts via list comprehension instead of per-pair method calls

## Expected Impact

| Metric | Before | After (estimated) | Improvement |
|--------|--------|-------------------|-------------|
| Training time | ~68 min | ~20-30 min | 60-70% |
| Backtest time | ~80 min | ~15-25 min | 70-80% |
| Memory peak | Unchanged | Unchanged | — |
| Prediction results | — | Identical | — |

## Files Changed

1. `src/features/horse_history_features.py` — S1: Vectorize compute()
2. `src/backtest/engine.py` — S2: Single-pass feature computation
3. `src/pipelines/training_pipeline.py` — S3: Parallel surface training
4. `src/features/bloodline_features.py` — S4a: Vectorize compute()
5. `src/features/jockey_context_features.py` — S4b: Vectorize compute()
6. `src/features/trainer_context_features.py` — S4b: Vectorize compute()
7. `src/models/wide_pair_builder.py` — S4c: Optimize pair generation

## Out of Scope

- External frameworks (Ray, Dask, Polars)
- Model hyperparameter tuning
- Feature caching to Parquet (DataRepository.load_features/save_features)
- ETL pipeline optimization (~5 min, already fast)
- Incremental/differential training
