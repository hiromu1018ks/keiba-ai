# Architecture Patterns

**Domain:** Paper Trading Pipeline Integration (v2.4)
**Researched:** 2026-06-06
**Overall confidence:** HIGH (based on direct codebase analysis of all integration points)

## Executive Summary

v2.4 aligns the paper trading pipeline with the backtest pipeline so both produce comparable results. The architecture follows one core pattern: extract shared code into a single function called by both paths. The shared feature builder (`build_inference_features()`) is extracted from `BacktestEngine.prepare_data()` (lines 792-1099) and called by both BT and PT. Bet settlement uses the existing `build_*_payout_map()` functions already proven in BT. Strategy alignment reuses `ParameterFreezeProtocol` already proven in BT. The one-command run mode chains existing functions rather than introducing new orchestration infrastructure.

## Recommended Architecture

### Before v2.4 (current state)

```
BacktestEngine.prepare_data()          run_paper_trading._run_predict()
  |                                        |
  +-- FeatureEngine.build_all()           +-- FeatureEngine.build_all()
  +-- SubModelManager                     +-- SubModelManager
  +-- HorseHistoryFeatures                +-- HorseHistoryFeatures
  +-- JockeyContextFeatures               +-- JockeyContextFeatures
  +-- TrainerContextFeatures              +-- TrainerContextFeatures
  +-- JockeyTrainerComboFeatures          +-- JockeyTrainerComboFeatures
  +-- SireFeatures                        +-- SireFeatures
  +-- PaceAptitudeFeatures                +-- PaceAptitudeFeatures
  +-- CourseFeatures                      +-- CourseFeatures
  +-- DamPedigreeFeatures  <-- MISSING in PT
  +-- RecordFeatures      <-- MISSING in PT
  +-- MiningFeatures      <-- MISSING in PT
  +-- BloodlineFeatures   <-- MISSING in BT (present in PT only)
  |                                        |
  v                                        v
  RacePredictor.predict()                RacePredictor.predict()
  +-- OddsBandFilter       <-- MISSING in PT
  +-- Strategy manifest    <-- MISSING in PT
  +-- Hardcoded AGGRESSIVE              +-- Dynamic regime detection
  +-- StakeCalculator/DD    <-- MISSING in PT (always flat 100)
```

### After v2.4 (proposed)

```
build_inference_features(store, race_df, entry_df, odds_df, odds_ts_df, *)
  |
  +-- FeatureEngine.build_all()
  +-- SubModelManager.add_distance_band_features()
  +-- HorseHistoryFeatures.compute()
  +-- JockeyContextFeatures.compute()
  +-- TrainerContextFeatures.compute()
  +-- JockeyTrainerComboFeatures.compute()
  +-- SireFeatures.compute_batch()
  +-- PaceAptitudeFeatures.compute_batch()
  +-- CourseFeatures.compute_batch()
  +-- DamPedigreeFeatures.compute()
  +-- RecordFeatures.compute()
  +-- MiningFeatures.compute()
  +-- BloodlineFeatures.compute()
  |
  v  shared feature DataFrame

BacktestEngine.prepare_data()          run_paper_trading._run_predict()
  |                                        |
  calls build_inference_features()       calls build_inference_features()
  |                                        |
  v                                        v
  RacePredictor.predict()                RacePredictor.predict()
  +-- OddsBandFilter                    +-- OddsBandFilter (same calibration)
  +-- Strategy manifest                 +-- Strategy manifest (same params)
  +-- Hardcoded AGGRESSIVE              +-- Hardcoded AGGRESSIVE (same)
  +-- StakeCalculator/DD (if kelly)     +-- StakeCalculator/DD (if kelly)

settle_bets(predictions, payout_data, betting_target)  <-- NEW shared function
  |
  +-- build_win_payout_map()   (reuse from engine.py)
  +-- build_payout_map()       (reuse from engine.py)
  +-- build_wide_payout_map()  (reuse from engine.py)
  |
  +-- Set status="settled" for all resolved bets (wins AND losses)
  +-- Set result=payout for wins, result=0.0 for losses
```

## Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `build_inference_features()` | Feature construction for both BT and PT | ParquetStore (data), FeatureEngine (base), 12 feature modules |
| `settle_bets()` | Win/place/wide payout calculation + status update | Payouts DataFrame, predictions DataFrame |
| `PaperTradingConsistency` | Data cutoff validation, identity tracking | Model info (train period), ParquetStore (data dates) |
| `PaperTradingOrchestrator` | One-command run mode lifecycle | PaperPredictor, RaceWatcher, PaperReconciler, PaperTradingReport |
| `ParameterFreezeProtocol` | (existing) Strategy parameter immutability | TrainedModelsV5, manifest JSON |
| `RacePredictor` | (existing, unchanged) Per-race inference | TrainedModelsV5, feature DataFrames |

## Data Flow

### PT Predict Flow (after v2.4)

```
run_paper_trading.py --mode predict --date 2026-06-06 --strategy-manifest data/strategy_manifest.json
  |
  +-- Load models from MLflow (existing)
  +-- Load strategy manifest + PFP freeze (NEW)
  +-- Load data from Parquet (existing)
  +-- build_inference_features(store, race_df, entry_df, odds_df, odds_ts_df) (NEW shared call)
  +-- Create RacePredictor with strategy params (NEW: StakeCalculator, DDController if kelly)
  +-- For each race:
  |     +-- RacePredictor.predict() (existing, unchanged)
  |     +-- Apply OddsBandFilter (NEW)
  |     +-- Select candidates (existing)
  |     +-- Record bet with status="pending", mlflow_run_id, train_period, code_hash (NEW columns)
  +-- Save predictions parquet (existing)
```

### PT Reconcile Flow (after v2.4)

```
run_paper_trading.py --mode reconcile --date 2026-06-06
  |
  +-- Load predictions parquet (existing)
  +-- Filter: status == "pending" (NEW: was result == 0.0)
  +-- Load payouts from EveryDB2/Parquet (existing)
  +-- build_win_payout_map(payouts_df) (NEW: reuse from engine.py)
  +-- build_payout_map(payouts_df) (existing)
  +-- build_wide_payout_map(payouts_df) (NEW: for wide bets)
  +-- For each pending bet:
  |     +-- Lookup payout by (race_id, umaban, bet_type)
  |     +-- If payout found: set result=payout, status="settled" (NEW: explicit loss)
  |     +-- If not found but race complete: set result=0.0, status="settled" (NEW: loss)
  |     +-- If race not complete: leave status="pending"
  +-- Save updated predictions (existing)
  +-- Update bets.parquet (existing)
  +-- Compute summary with weekly + per-target aggregation (NEW)
  +-- Generate HTML report (existing, enhanced)
```

### One-Command Run Flow (after v2.4)

```
run_paper_trading.py --mode run --date 2026-06-06 --strategy-manifest data/strategy_manifest.json
  |
  +-- [Verify] Load models + manifest + PFP freeze
  +-- [Verify] Data cutoff validation
  +-- [Predict] build_inference_features() + predict all races
  +-- [Predict] Save predictions with status="pending"
  +-- [Wait] Wait until last race post time + 30 min
  +-- [Reconcile] settle_bets() for all completed races
  +-- [Report] Generate daily summary + HTML report
  +-- [Exit] exit code 0 if all races settled, 1 if partial, 2 if fatal
```

## Patterns to Follow

### Pattern 1: Shared Feature Builder Extraction

**What:** Extract feature construction into a standalone function called by both BT and PT.
**When:** Feature construction that must be identical between BT and PT.
**Example:**

```python
# src/paper_trading/feature_builder.py
def build_inference_features(
    store: ParquetStore,
    race_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    odds_ts_df: pd.DataFrame,
    *,
    betting_target: str = "win",
) -> pd.DataFrame:
    """Build complete feature set for inference.

    Extracted from BacktestEngine.prepare_data() to ensure
    BT and PT use identical feature construction.
    """
    from features.feature_engine import FeatureEngine
    from features.horse_history_features import HorseHistoryFeatures
    from features.jockey_context_features import JockeyContextFeatures
    # ... all 12+ feature module imports ...

    feat_engine = FeatureEngine()
    submodel_mgr = SubModelManager()
    feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df, store=store)
    feat_df = submodel_mgr.add_distance_band_features(feat_df)

    # All additional features
    hist_df = HorseHistoryFeatures(store=store).compute(race_df, entry_df, feat_df["race_id"].unique())
    # ... merge all feature modules ...

    return feat_df
```

### Pattern 2: Status Lifecycle for Bet Records

**What:** Add explicit `status` column to track bet resolution state.
**When:** Any system that records bets for later reconciliation.
**Example:**

```python
# At predict time:
bet_record = {
    "race_id": race_id,
    "umaban": umaban,
    "bet_type": "win",
    "stake": 100.0,
    "odds": 5.2,
    "result": 0.0,      # placeholder
    "status": "pending",  # NEW
    "mlflow_run_id": model_info.mlflow_run_id,  # NEW
    "train_start": model_info.train_start,      # NEW
    "train_end": model_info.train_end,          # NEW
}

# At reconcile time:
if race_id in payout_map:
    if umaban in payout_map[race_id]:
        record["result"] = stake * payout_map[race_id][umaban]
    else:
        record["result"] = 0.0  # explicit loss
    record["status"] = "settled"  # both wins AND losses
```

### Pattern 3: Reuse Existing Settlement Functions

**What:** Import and call `build_*_payout_map()` from `backtest.engine` rather than reimplementing.
**When:** Any settlement logic that must match BT exactly.
**Example:**

```python
# src/paper_trading/settlement.py
from backtest.engine import build_win_payout_map, build_payout_map, build_wide_payout_map

def settle_bets(
    predictions_df: pd.DataFrame,
    payouts_df: pd.DataFrame,
    betting_target: str = "win",
) -> pd.DataFrame:
    """Settle all pending bets with actual payouts."""
    pending = predictions_df[predictions_df["status"] == "pending"]

    win_payouts = build_win_payout_map(payouts_df) if betting_target in ("win", "wide") else {}
    place_payouts = build_payout_map(payouts_df) if betting_target in ("place", "wide") else {}
    wide_payouts = build_wide_payout_map(payouts_df) if betting_target == "wide" else {}

    for idx, row in pending.iterrows():
        # ... settlement logic using same maps as BT ...
```

### Pattern 4: Strategy Manifest Passthrough

**What:** Load and verify manifest at PT start, just like BT.
**When:** PT must use same strategy parameters as BT.
**Example:**

```python
# In run_paper_trading.py:
if args.strategy_manifest:
    from backtest.parameter_freeze_protocol import verify_strategy_manifest, ParameterFreezeProtocol
    verify_strategy_manifest(args.strategy_manifest)
    pfp = ParameterFreezeProtocol(models)
    pfp.freeze()
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Two Parallel Feature Construction Paths

**What:** Maintaining separate feature construction code in BT and PT.
**Why bad:** They will drift apart over time (already have: Gap 1 in FEATURES.md). Missing features in one path produce different predictions.
**Instead:** Single shared function `build_inference_features()` called by both.

### Anti-Pattern 2: Using result=0.0 as "Pending" Status

**What:** Checking `result == 0.0` to determine if a bet is unsettled.
**Why bad:** A lost bet also has `result=0.0`. Cannot distinguish "not yet reconciled" from "lost". Leads to ROI overstatement (losses excluded from denominator).
**Instead:** Explicit `status` column: `"pending"` / `"settled"`.

### Anti-Pattern 3: Only Recording Wins

**What:** Reconcile loop that only updates `result` for winning bets.
**Why bad:** Losers stay at `result=0.0` with no way to know if they were processed. Cumulative stats only count winning returns, inflating apparent ROI.
**Instead:** Mark ALL bets as settled (both wins and losses) once results are available.

### Anti-Pattern 4: Recreating Settlement Logic in PT

**What:** Writing new settlement code for PT instead of reusing BT's `build_*_payout_map()` functions.
**Why bad:** Different implementations will diverge. Edge cases (kumi parsing for wide, pay_100 normalization) are handled differently.
**Instead:** Import and call existing functions from `backtest.engine`.

### Anti-Pattern 5: Dynamic Regime in PT While BT Uses Hardcoded

**What:** PT uses `regime_detector.detect()` while BT hardcodes `AGGRESSIVE`.
**Why bad:** Different regime = different bet selection. PT and BT select different horses for the same race. ROI comparison is meaningless.
**Instead:** Both use hardcoded `AGGRESSIVE` (or both use dynamic -- but they must match).

### Anti-Pattern 6: One-Command Run That Hides Failures

**What:** Run mode that catches all exceptions and continues, reporting success even when partial failures occur.
**Why bad:** Operator trusts the "success" output but some races were not processed. Bets may be missing from the record.
**Instead:** Explicit exit codes: 0=all settled, 1=partial (some races not settled), 2=fatal error. Log every failure.

## Scalability Considerations

| Concern | At 1 day (12 races) | At 1 month (300 races) | At 1 year (3600 races) |
|---------|---------------------|------------------------|------------------------|
| Predictions parquet size | ~1 MB (< 50 bets) | ~30 MB (~1500 bets) | ~360 MB (~18K bets) |
| Settlement lookup time | O(1) dict lookup | O(1) dict lookup | O(1) dict lookup |
| Cumulative stats computation | Trivial | Trivial | Trivial (pandas groupby) |
| HTML report rendering | ~100ms | ~500ms | ~2s (Jinja2 template) |
| Feature construction time | ~10s (single day) | ~30s (batch month) | N/A (daily execution) |

All operations are bounded by single-machine, single-process execution. No scalability concerns at projected volumes.

## Sources

- Direct codebase analysis: `src/backtest/engine.py` (2392 lines, prepare_data + settlement)
- Direct codebase analysis: `src/paper_trading/reconciler.py` (153 lines, current settlement)
- Direct codebase analysis: `scripts/run_paper_trading.py` (1384 lines, PT CLI)
- Direct codebase analysis: `src/backtest/parameter_freeze_protocol.py` (manifest/PFP)
- Direct codebase analysis: `src/paper_trading/report.py` (157 lines, reporting)

---
*Architecture research for: v2.4 Paper Trading Pipeline Integration*
*Researched: 2026-06-06*
