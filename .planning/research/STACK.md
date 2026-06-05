# Technology Stack

**Project:** keiba-ai v2.4 -- Paper Trading Pipeline Integration
**Researched:** 2026-06-06
**Supersedes:** v2.1/v2.3 STACK.md

## Verdict: Zero New External Dependencies

All v2.4 features (settlement integrity, shared feature builder, manifest alignment, one-command run, reporting expansion) are implementable with the existing installed stack. No new pip packages required.

**Rationale:** v2.4 is integration work -- connecting existing components that already work independently. The shared feature builder extracts code that already exists in `BacktestEngine.prepare_data()` and in the duplicated inline feature construction in `scripts/run_paper_trading.py` (predict/diagnose/dry-run modes each have ~150 lines of identical feature-building code). Status lifecycle adds a string column. Manifest integration reuses existing `ParameterFreezeProtocol`. One-command run chains existing CLI modes.

## Current Installed Stack

| Package | Version | Role in v2.4 |
|---------|---------|-------------|
| Python | 3.11 | Pinned via mise |
| pandas | 2.3.3 | Bet record management, Parquet I/O, aggregation |
| pyarrow | 23.0.1 | Parquet read/write for predictions/bets; schema evolution handled natively |
| numpy | 2.4.3 | Numerical operations in feature construction |
| scikit-learn | 1.8.0 | Calibrators, Ridge ranker (inference only) |
| LightGBM | 4.6.0 | Primary model (inference only in PT) |
| XGBoost | 3.2.0 | Stacking model (inference only in PT) |
| CatBoost | 1.2.10 | Stacking model (inference only in PT) |
| joblib | 1.5.3 | Model serialization (unchanged) |
| mlflow | 3.10.1 | Run ID tracking + artifact loading |
| Jinja2 | 3.1.6 | HTML report generation |
| PyYAML | 6.0.3 | Config loading (`config/settings.yaml`) |
| python-dotenv | 1.0 | Environment variables (`.env`) |
| tqdm | 4.67.3 | Progress bars (optional) |
| Optuna | 4.8.0 | Strategy optimization (not used at PT runtime) |
| psycopg2-binary | 2.9+ | PostgreSQL EveryDB2 queries for reconciliation |
| SQLAlchemy | 2.0+ | DB connection for ETL layer |
| pathlib | (stdlib) | Path management for PT output directories |
| typing | (stdlib) | Type annotations for shared builder interface |
| json | (stdlib) | Manifest I/O, daily summary, audit log |
| logging | (stdlib) | Pipeline execution logging |
| hashlib | (stdlib) | SHA256 integrity verification |
| subprocess | (stdlib) | git rev-parse for commit hash in reports |

## Recommended Stack for v2.4

### 1. Shared Feature Builder

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `pandas.DataFrame` | 2.3.3 | Feature construction output format | All existing feature modules return DataFrames. No reason to change. |
| `ParquetStore` | (codebase) | Data source for feature computation | Already injected into BacktestEngine and PT. Shared builder receives it. |
| `FeatureEngine` | (codebase) | Base feature orchestration | `build_all()` already works for both BT and PT paths. |
| 15 feature modules | (codebase) | Individual feature computation | `HorseHistoryFeatures`, `JockeyContextFeatures`, `TrainerContextFeatures`, `JockeyTrainerComboFeatures`, `SireFeatures`, `PaceAptitudeFeatures`, `CourseFeatures`, `BloodlineFeatures`, plus interaction/relative/odds-deviation/track-condition modules |

**Approach:** Extract the feature construction section from `scripts/run_paper_trading.py::_run_predict()` (lines 368-436) and the equivalent in `_run_diagnose()` (lines 743-798) into a standalone function `build_inference_features(store, race_df, entry_df, odds_df, odds_ts_df, *, betting_target="win") -> pd.DataFrame`. This function encapsulates all feature module calls. Both BT and PT call it with the same arguments.

**Key insight from codebase analysis:** The `run_paper_trading.py` script has THREE nearly-identical copies of the feature construction pipeline:
1. `_run_predict()` lines 368-436 (~70 lines)
2. `_run_diagnose()` lines 743-798 (~55 lines)
3. `_run_dry_run()` lines 1200-1266 (~65 lines)

All three call the same sequence: `FeatureEngine.build_all()` -> `SubModelManager.add_distance_band_features()` -> `HorseHistoryFeatures.compute()` -> `JockeyContextFeatures.compute()` -> `TrainerContextFeatures.compute()` -> `JockeyTrainerComboFeatures.compute()` -> `BloodlineFeatures.compute()` -> `SireFeatures.compute_batch()` -> `PaceAptitudeFeatures.compute_batch()` -> `CourseFeatures.compute_batch()`. Extracting this into a shared function eliminates ~190 lines of duplication and guarantees BT/PT feature parity.

### 2. Bet Status Lifecycle

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `pandas.DataFrame` column | 2.3.3 | `status` column ("pending" / "settled") | Simple string column. No new library needed. |
| `pandas.read_parquet` / `.to_parquet` | 2.3.3 | Persistence of bet records | Existing pattern. Schema evolution is native: new columns default to NaN. |
| `build_win_payout_map()` | (codebase) | Win payout lookup for settlement | Already exists in `backtest.engine`. Reuse directly. |
| `build_payout_map()` | (codebase) | Place payout lookup for settlement | Already exists in `backtest.engine`. Reuse directly. |
| `build_wide_payout_map()` | (codebase) | Wide payout lookup for settlement | Already exists in `backtest.engine`. Reuse directly. |

**Approach:** Add `status: str` column to prediction/bet records. Default `"pending"`. Reconcile sets to `"settled"` for all bets where payout data is available, regardless of win/loss. Import and reuse the three `build_*_payout_map()` functions from `backtest.engine`.

**Parquet schema evolution pattern** (no migration library needed):
```python
def ensure_bets_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing bet status columns with defaults."""
    for col, default in {
        "status": "pending",
        "settled_at": pd.NaT,
        "actual_payout": 0.0,
        "actual_odds": 0.0,
        "settlement_source": "",
    }.items():
        if col not in df.columns:
            df[col] = default
    return df
```

This works because: (a) bets Parquet is written daily (one file per day), (b) schema changes are additive only, (c) `pandas.read_parquet()` reads whatever columns exist, and (d) `to_parquet()` writes all current columns.

### 3. Strategy Manifest / PFP Integration

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `ParameterFreezeProtocol` | (codebase) | Immutability verification during PT execution | Already exists in `backtest/parameter_freeze_protocol.py`. Already used by BT. |
| `verify_strategy_manifest()` | (codebase) | SHA256 manifest verification | Already exists. Already used by BT. |
| `freeze_feature_manifest.py` | (scripts/) | Feature column SHA256 manifest | Already exists. Reuse pattern for PT feature schema verification. |
| `hashlib.sha256` | (stdlib) | Content hashing | Existing pattern in PFP. No need for xxHash or other checksum libraries. |
| `json.load/dump` | (stdlib) | Manifest I/O | Existing pattern. |

**Approach:** Load manifest at predict start. Freeze params via PFP. Verify at reconcile. Same pattern as BT `engine.py` lines 1119-1127. The `ModelLoader` already records `mlflow_run_id`, `train_start`, `train_end` in `model_info.json` -- extend this to include `code_hash`, `feature_manifest_sha256`, and `strategy_manifest_sha256`.

**Data cutoff validation** is a simple date comparison:
```python
train_end = date.fromisoformat(model_info.train_end)
if train_end >= predict_date:
    raise DataCutoffViolationError(
        f"Training data ends {train_end} >= prediction date {predict_date}. "
        f"Future information leak risk."
    )
```

### 4. One-Command Run Mode

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `argparse` | (stdlib) | Add `--mode run` to existing CLI | Extend existing `parse_args()`. No need for `click`/`typer`. |
| Sequential function calls | (stdlib) | Chain existing modes | Call existing `_run_*` functions in sequence. No need for Prefect/Airflow. |
| `Path.exists()` | (stdlib) | Restart resumption detection | Check for partial outputs to determine where to resume. |
| `sys.exit(code)` | (stdlib) | Exit codes (0=success, 1=partial, 2=fatal) | Standard process exit pattern. |
| `RaceWatcher` | (codebase) | Per-race timing during run mode | Already handles `time.sleep()` + retry + idempotent skip. No need for `schedule` library. |
| `SafetyGuard` | (codebase) | Pre-bet safety check | Already implements bankroll/daily-loss/consecutive-loss checks. |
| `RaceScheduler` | (codebase) | Race-day task sequencing | Protocol-based DI with `OrchestratorProtocol`, `OddsCollectorProtocol`, etc. |

**Approach:** Add `--mode run` to CLI that chains: model verify -> predict -> wait-for-last-race -> reconcile -> report. Uses processed-race tracking for restart resumption. The `RaceWatcher.watch()` already provides per-race timing with `wait_until()` and processed-race detection via Parquet file existence.

**Why not `schedule` library:** Race times change daily (dynamic schedule), the PT pipeline is a single session (not a daemon), `RaceWatcher` already handles per-race timing, and `schedule` would introduce an event-loop pattern inconsistent with the synchronous design.

**Why not `tenacity` for retry:** The existing manual retry pattern in `RaceWatcher` (3 attempts with configurable interval) is sufficient. Adding Tenacity would introduce a decorator-based abstraction for just 3 retry points, inconsistent with the codebase style.

### 5. Reporting Expansion

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `pandas.DataFrame.groupby` | 2.3.3 | Weekly aggregation, per-target breakdown | Existing pattern in `report.py` `_compute_monthly_stats`. |
| `jinja2` | 3.1.6 | HTML report rendering | Already used in `PaperTradingReport`. Extend template. |
| `pandas.Grouper(freq='W')` | 2.3.3 | Weekly time-based grouping | Built into pandas. No new library needed. |

**Approach:** Extend `_compute_monthly_stats` to also compute weekly stats. Add per-bet_type grouping. Add model identity section to HTML template (MLflow run ID, train period, code hash, manifest hashes).

## What NOT to Add

| Library/Approach | Why Rejected | Use Instead |
|-----------------|-------------|-------------|
| `celery` / `dramatiq` / `rq` | No distributed task queue needed. Single-machine, single-operator. | Sequential function calls in one-command run mode. |
| `sqlite3` for bet storage | Parquet files work fine for current scale (< 10K bets). Adding a DB layer is over-engineering. | Parquet files with append + dedup pattern. |
| `click` / `typer` for CLI | `argparse` already used. Adding a CLI framework for one new mode is unnecessary complexity. | Extend existing `argparse` setup. |
| `pydantic` for bet schema | Bet records are dicts stored in DataFrames. Adding schema validation is over-engineering for PT scale. Pydantic 2.12.5 is already installed (transitive dep) but not needed here. | DataFrame column checks with `ensure_bets_schema()`. |
| `schedule` / `croniter` / `APScheduler` | Race-day scheduling uses `RaceWatcher` which waits by time. No need for cron-like scheduling. Single-session pipeline, not a daemon. | Existing `time.sleep()` pattern in `RaceWatcher.watch()`. |
| `tenacity` for retry | PT already has retry logic in `RaceWatcher` (retry_count, retry_interval_seconds). 3 retry points total. | Existing manual retry pattern. |
| `fastapi` / `flask` | No web interface needed. CLI + HTML report is sufficient. Out of scope per PROJECT.md. | Existing CLI + Jinja2 HTML report. |
| `transitions` / `python-statemachine` | Bet status has only 2 states (pending/settled). Pipeline phases are linear. A library is overkill. | Simple string column with if/elif checks. |
| `xxhash` / `crc32c` | `hashlib.sha256` already used consistently across codebase. Performance irrelevant for metadata hashing. | `hashlib.sha256` (existing pattern). |
| Parquet page checksums | Optional and off by default in Parquet spec. PyArrow API for page-level access is unstable. | File-level SHA256 via hashlib. |
| Delta Lake / Lance | New dependency; single-process writes; no concurrent access; over-engineering for daily files. | Pandas `to_parquet()` with column addition pattern. |
| Prefect / Airflow / Dagster | Single-machine, single-session pipeline. No DAG complexity. No distributed scheduling. | Sequential function calls. |

## New Files to Create

| File | Purpose | Est. LOC |
|------|---------|----------|
| `src/paper_trading/feature_builder.py` | Shared `build_inference_features()` extracted from BT engine and PT script | ~200 |
| `src/paper_trading/settlement.py` | Unified settlement with win/place/wide support + status lifecycle | ~150 |
| `src/paper_trading/consistency.py` | Data cutoff validation, MLflow identity tracking, manifest verification for PT | ~100 |
| `src/paper_trading/orchestrator.py` | One-command run mode orchestrator | ~200 |
| `tests/test_pt_feature_builder.py` | Unit tests for shared builder | ~200 |
| `tests/test_pt_settlement.py` | Unit tests for settlement | ~200 |
| `tests/test_pt_consistency.py` | Unit tests for consistency checks | ~150 |
| `tests/test_pt_orchestrator.py` | Unit tests for orchestrator | ~150 |

## Modified Files

| File | Change |
|------|--------|
| `src/backtest/engine.py` | Replace inline feature construction with call to shared `build_inference_features()` |
| `scripts/run_paper_trading.py` | Replace 3x duplicated feature construction (~190 lines) with shared builder call. Add `--mode run`, `--strategy-manifest`, `--betting-target`, `--betting-mode`, `--regime` flags |
| `src/paper_trading/predictor.py` | Refactor to use shared feature builder |
| `src/paper_trading/reconciler.py` | Add win/wide settlement, status lifecycle, loss recording |
| `src/paper_trading/report.py` | Add weekly aggregation, per-target breakdown, model identity section |
| `src/paper_trading/config.py` | Add manifest_path, betting_target, betting_mode fields |

## Existing Code to Reuse As-Is

| Component | File | Feature |
|-----------|------|---------|
| `RaceWatcher` | `src/paper_trading/watcher.py` | One-command run (watch/monitor phase) |
| `SafetyGuard` | `src/automation/safety_guard.py` | Strategy alignment (bet blocking) |
| `RaceScheduler` | `src/automation/scheduler.py` | One-command run (per-race timing) |
| `PatVoter` | `src/automation/pat_voter.py` | Not used in PT (paper trading, no IPAT) |
| `ParameterFreezeProtocol` | `src/backtest/parameter_freeze_protocol.py` | Consistency verification (model immutability) |
| `verify_strategy_manifest` | `src/backtest/parameter_freeze_protocol.py` | Strategy alignment (manifest integrity) |
| `save_strategy_manifest` | `src/backtest/parameter_freeze_protocol.py` | Manifest creation |
| `ModelLoader` | `src/db/model_loader.py` | MLflow run ID tracking |
| `ModelInfo` | `src/db/model_loader.py` | Model metadata (run_id, train_start, train_end) |
| `RacePredictor` | `src/backtest/race_predictor.py` | Shared inference pipeline (already used by both BT and PT) |
| `SlackNotifier` | `src/monitoring/notifier.py` | Notification |
| `LoggingNotifier` | `src/monitoring/notifier.py` | Fallback notification |
| `PaperTradingReport` | `src/paper_trading/report.py` | Report generation (extend for weekly) |
| `ModelMonitor` | `src/monitoring/model_monitor.py` | Drift detection (optional for PT) |
| `ParquetStore` | `src/db/parquet_store.py` | Parquet I/O |
| `freeze_feature_manifest.py` | `scripts/freeze_feature_manifest.py` | Feature column SHA256 pattern |
| `_drop_post_race_cols()` | `scripts/run_paper_trading.py` | POST_RACE column removal (already shared concept, needs formal extraction) |
| `_apply_jra_filter()` | `scripts/run_paper_trading.py` | NAR race exclusion (already shared concept, needs formal extraction) |

## Sources

- Codebase analysis: `scripts/run_paper_trading.py` (1384 lines), `src/paper_trading/` (6 files), `src/automation/` (3 files), `src/monitoring/` (3 files), `src/backtest/race_predictor.py` (1645 lines), `src/backtest/parameter_freeze_protocol.py` (187 lines), `src/db/model_loader.py` (931 lines)
- Installed versions verified via `pip list` on 2026-06-06: pandas 2.3.3, pyarrow 23.0.1, mlflow 3.10.1, jinja2 3.1.6, pydantic 2.12.5, lightgbm 4.6.0, xgboost 3.2.0, catboost 1.2.10
- Parquet schema evolution: https://dev.to/alexmercedcoder/all-about-parquet-part-04-schema-evolution-in-parquet-57l3
- Parquet checksums (optional, off by default): https://stackoverflow.com/questions/79398345/how-to-determine-if-checksums-are-present-in-parquet-file
- xxHash for fast data integrity (considered and rejected): https://jolynch.github.io/posts/use_fast_data_algorithms/
- Tenacity retry library (considered and rejected): https://tenacity.readthedocs.io/
- Idempotent pipeline patterns: https://www.prefect.io/blog/the-importance-of-idempotent-data-pipelines-for-resilience
- `schedule` library (considered and rejected): https://dev.to/whoakarsh/automate-scheduled-jobs-in-python-using-the-schedule-library-a-cron-alternative-811

---
*Stack research for: v2.4 Paper Trading Pipeline Integration*
*Researched: 2026-06-06*
