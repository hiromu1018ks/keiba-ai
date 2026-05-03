# Technology Stack: Win Backtest Fix & Pipeline Optimization

**Project:** keiba-ai v1.2 Win Backtest Validation
**Researched:** 2026-05-04
**Scope:** Backtest mode switching (place to win) + pipeline performance optimization
**Supersedes:** v1.1 STACK.md (stacking/odds deviation stack -- all still current)

## Current Installed Stack

| Package | Installed Version | pyproject.toml Minimum | Status |
|---------|-------------------|----------------------|--------|
| Python | 3.11 | >=3.11 | Pinned via mise |
| LightGBM | 4.6.0 | >=4.3 | Up to date |
| XGBoost | 3.2.0 | >=2.0 | Up to date |
| CatBoost | 1.2.10 | >=1.2 | Up to date |
| scikit-learn | 1.8.0 | >=1.4 | Up to date |
| scipy | 1.17.1 | (transitive) | Up to date |
| pandas | 2.3.3 | >=2.2 | Up to date |
| numpy | 2.4.3 | >=1.26 | Up to date |
| pyarrow | (installed) | >=14.0 | Up to date |
| mlflow | 3.10.1 | >=2.12 | Up to date |
| optuna | 4.8.0 | >=3.5 | Up to date |

## Recommended Stack

### Production Dependencies: Zero New Additions

| Technology | Version | Purpose | Why No Change |
|------------|---------|---------|----------------|
| pandas | >=2.2 | Vectorization | Replace `iterrows()` with `.where()`, `.groupby().transform()`, numpy broadcasting. No new library needed. |
| numpy | >=1.26 | Vectorized ops | `np.where()`, dict comprehension from `zip()` for payout map construction. |
| LightGBM | >=4.3 | Batch inference | `booster.predict(df)` already supports batch. Restructure calling code to accumulate predictions. |
| pyarrow | >=14.0 | Parquet I/O | Already used; pyarrow compute kernels available for additional speed if needed. |
| cProfile | stdlib | Initial profiling | Built-in, zero-install. Use for quick bottleneck identification before pyinstrument. |

### Dev-Only Addition

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pyinstrument | >=4.6 | Statistical profiler | ~1-5% overhead vs cProfile's 20-50%. Produces readable call-tree. Ideal for profiling the ~57-minute backtest run without distorting timings. Dev dependency only. |

### NOT Recommended

| Technology | Why Not |
|------------|---------|
| scalene | Adds GPU profiling overhead irrelevant to this CPU-bound pandas/LightGBM workload. pyinstrument is lighter and sufficient. |
| line_profiler | Requires `@profile` decorators -- too intrusive for codebase with strict mypy and 1113 tests. |
| memory_profiler | Memory is not the bottleneck; Parquet pipeline is already memory-efficient. |
| polars | Would require rewriting entire feature engine (14 modules, 100+ features). Negative ROI for an optimization milestone. |
| Cython / numba | Only 2-3 hot functions would benefit; main gains come from eliminating iterrows() and batch-ifying LightGBM predict. |
| ray / dask | Distributed computing is overkill for single-machine backtests on ~5000 races. |
| py-spy | May require elevated permissions; pyinstrument is pip-install-clean and equally capable for offline profiling. |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Profiler | pyinstrument | scalene | Overkill for CPU-only workload; pyinstrument is lighter |
| Profiler | pyinstrument | cProfile | Higher overhead distorts timing; less readable output |
| Profiler | pyinstrument | py-spy | May require elevated permissions on Windows |
| Vectorization | native pandas/numpy | polars | Would require rewriting 14 feature modules |
| Parallelism | ThreadPoolExecutor (existing) | multiprocessing | LightGBM uses OpenMP threads; GIL limits Python parallelism |
| Build optimization | N/A | Cython/Numba | Only 2-3 functions benefit; not worth build complexity |

## Backtest Mode Switch: No New Stack Required

The switch from place (fukushou) to win (tanshou) verification is a **code refactoring task** requiring no new dependencies. Analysis of the codebase confirms:

**Win model infrastructure already exists:**
- `WinTwoStageModel` trains and predicts `p_win`, `ev_win`, `ev_win_corrected`
- `WinSelectionGateModel` scores and gates win candidates with `win_gate_score`, `win_gate_pass`
- `WinBenterGate` applies Benter combination for win probability calibration
- `ensure_win_selection_columns()` computes `win_selection_ev`, `win_selection_edge`, `win_selection_prob`
- `_settle_bet()` already handles `BetType.WIN` with `finish_pos == 1` check

**The gap is orchestration only:** BacktestEngine and RacePredictor hardcode place-oriented logic in candidate selection and payout settlement. The `BetType.WIN` enum is defined but never used in the backtest path.

**Payout data source:** The payouts DataFrame (from `load_payouts()`) contains win payout columns `paytansyoushowumaban{1}`, `paytansyoushowpay{1}` alongside the place columns `payfukusyoumaban{1-5}`, `payfukusyopay{1-5}` already in use. A vectorized `build_win_payout_map()` reads these columns.

## Pipeline Optimization: Bottleneck Analysis

### Current Timing (from CLAUDE.md)

| Pipeline | Duration | In Scope? |
|----------|----------|-----------|
| ETL (run_etl.py) | ~10 min | No |
| Training (run_train.py) | ~44 min | Partially (feature computation) |
| Backtest per year | ~57 min | Yes -- primary target |

### Identified Bottleneck Patterns

**Bottleneck 1: 18+ iterrows() call sites in backtest path**

| Location | What It Does | Vectorizable? |
|----------|-------------|---------------|
| `engine.py:112` | Build payout map | YES -- df.melt() + dict comprehension |
| `engine.py:151` | Build wide payout map | YES -- vectorize with numpy |
| `engine.py:278` | Build final_odds_map | YES -- dict(zip()) |
| `engine.py:451` | Extract top3 finishers | YES -- .nsmallest() + to_dict |
| `engine.py:892` | Bet generation | YES -- vectorize stake calc |
| `race_predictor.py:607` | Create Bet objects | Partial -- Kelly logic per-row |
| `race_predictor.py:681` | Wide pair selection | YES -- numpy cross-product |

**Bottleneck 2: Per-race sequential loop (engine.py:420)**

```python
for race_id in race_ids:  # ~5000 races, each ~14 horses
```

Each iteration: DataFrame filtering, RacePredictor.predict() (7 model inference calls), candidate selection, bet settlement. Bankroll dependency prevents trivial parallelism, but per-iteration cost can be reduced.

**Bottleneck 3: Redundant DataFrame copies**

`RacePredictor.predict()` calls `df = race_df.copy()` then multiple `.merge()` operations, creating 5-6 intermediate DataFrames per race (~14 rows x ~100 cols each). Over 5000 races this is significant Python object allocation overhead.

### Optimization Techniques (All Native pandas/numpy)

**Technique 1: Vectorized payout map (replace iterrows at engine.py:112)**

Replace the row-by-row loop with DataFrame melt + groupby:

```python
def build_payout_map_vectorized(payouts_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    frames = []
    for i in range(1, 6):
        col_umaban = f"payfukusyoumaban{i}"
        col_pay = f"payfukusyopay{i}"
        if col_umaban in payouts_df.columns and col_pay in payouts_df.columns:
            sub = payouts_df[["race_id", col_umaban, col_pay]].dropna()
            sub = sub.rename(columns={col_umaban: "umaban", col_pay: "pay"})
            frames.append(sub)
    if not frames:
        return {}
    melted = pd.concat(frames, ignore_index=True)
    melted["umaban"] = pd.to_numeric(melted["umaban"], errors="coerce")
    melted["pay"] = pd.to_numeric(melted["pay"], errors="coerce")
    melted = melted.dropna()
    return dict(zip(
        zip(melted["race_id"].astype(str), melted["umaban"].astype(int)),
        melted["pay"] / 100.0
    ))
```

**Technique 2: Dict comprehension for final_odds_map (replace iterrows at engine.py:278)**

```python
valid = final_odds_df[final_odds_df["fukuoddslow"].notna()]
final_odds_map = dict(zip(
    zip(valid["race_id"].astype(str), valid["umaban"].astype(int)),
    valid["fukuoddslow"].astype(float)
))
```

**Technique 3: Batch predict across races where feasible**

The current per-race predict chain calls 7 models sequentially for ~14 horses. If the feature preprocessing (hist, jockey, trainer features) is batched (it already is in BacktestEngine lines 329-405), the model prediction could also be batched by accumulating all race DataFrames and running predict once.

**Technique 4: Gate diagnostic logging behind flag**

`diag_logger.log_horse()` calls `hr.to_dict()` for every horse in every race (lines 591, 678). This dict construction is expensive. Gate behind `--report` flag or a `--diag` flag.

### Profiling Strategy

**Step 1: Instrument with pyinstrument**

```bash
pip install pyinstrument
pyinstrument scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231 \
  --skip-train --ensemble
```

This produces a call-tree showing exactly where the ~57 minutes are spent.

**Step 2: Validate with existing TimingContext**

The `utils/timing.py` TimingContext already measures feature computation steps. Extend it to measure the backtest loop body.

**Step 3: Expected top consumers (based on code analysis)**

| Component | Estimated % | Optimization |
|-----------|-------------|-------------|
| Feature pre-computation (HorseHistory, Jockey, Trainer) | ~40% | Already batched; minor gains |
| Per-race predict chain (7 model calls x 5000 races) | ~35% | Batch predict where possible |
| Bet selection + settlement (get_place_candidates, _settle_bet) | ~15% | Vectorize payout lookup |
| Diagnostic logging + DataFrame copies | ~10% | Gate behind flag, reduce copies |

## Installation

```bash
# Dev-only profiling tool
pip install pyinstrument

# Or add to pyproject.toml [project.optional-dependencies] dev:
# "pyinstrument>=4.6"

# No production dependencies needed
```

## Sources

- [pyinstrument GitHub](https://github.com/joerick/pyinstrument) -- statistical profiler for Python
- [Pandas performance hierarchy](https://python.plainenglish.io/optimization-of-pandas-performance-on-large-data-c4cbe6b1b064) -- vectorization > itertuples > apply > iterrows
- [LightGBM batch prediction](https://letsdatascience.com/blog/lightgbm-the-definitive-guide-to-speed-and-efficiency) -- vectorized batch predict vs sequential
- [Beyond cProfile](https://pythonspeed.com/articles/beyond-cprofile/) -- profiling tool comparison
- [700x Speedup with vectorization](https://www.linkedin.com/pulse/tutorial-basic-vectorization-pandas-iterrows-apply-duc-lai-trung-minh-75d4c) -- iterrows vs vectorized benchmarks
- Code analysis: `src/backtest/engine.py`, `src/backtest/race_predictor.py`, `src/domain/models.py`, `src/pipelines/training_pipeline.py`, `scripts/run_backtest.py`, `scripts/run_wf_validation.py`, `src/models/win_selection_gate.py`
