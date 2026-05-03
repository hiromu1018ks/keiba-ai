# Architecture Patterns: Win Backtest Fix and Pipeline Optimization

**Domain:** Backtest engine conversion from place-mode to win-mode + pipeline performance optimization
**Researched:** 2026-05-04
**Parent system:** keiba-ai v5.5 (LightGBM + 2-stage decomposition P(hit) x E(odds|hit))

## Executive Summary

The backtest system is architecturally hardwired for place (fukusho) betting across three layers: (1) the payout data pipeline only extracts fukusho payouts from EveryDB2, missing win (tansho) payouts entirely; (2) the BacktestEngine and RacePredictor orchestrate candidate selection exclusively through `get_place_candidates()` and `place_selection_ev`/`place_selection_edge` columns; (3) diagnostic logging records only place-specific EV metrics. Converting to win-mode requires changes at all three layers, but the changes are surgical -- the inference chain already computes win EV columns (`ev_win`, `ev_win_corrected`, `win_selection_ev`, `win_selection_edge`, `win_selection_prob`), and the WinSelectionGateModel is already trained and stored in SubmodelSet. The missing pieces are: win payout ETL, a `get_win_candidates()` method on RacePredictor parallel to the existing `get_place_candidates()`, and win-specific settlement logic.

For pipeline optimization, the primary bottleneck is redundant feature computation. HorseHistoryFeatures, JockeyContextFeatures, TrainerContextFeatures, JockeyTrainerComboFeatures, PaceAptitudeFeatures, CourseFeatures, and SireFeatures are each instantiated and computed twice -- once during TrainingPipelineV5._train_submodel() and again inside BacktestEngine.run(). The backtest also recomputes features race-by-race in the main loop, when it already pre-computes them in bulk above the loop. The optimization strategy is: (1) extract the pre-computation block into a reusable FeatureComputer class, (2) add Parquet-based caching for expensive feature modules, (3) vectorize the per-race merge operations.

## Recommended Architecture

### Current State: Place-Mode Hardcoding Map

```
PLACE-MODE HARDCODING (what needs to change)
=============================================

Layer 1: ETL / Payout Data
  everydb2_queries.get_payouts()
    SQL: SELECT payfukusyoumaban1..5, payfukusyopay1..5
    MISSING: paytansyoumaban1, paytansyopay1  (win payout columns exist in EveryDB2 but not extracted)

  etl._TABLE_TYPE_RULES["payouts"]["int"]
    Has: paytansyoumaban1 + payfukusyoumaban1..5
    Has: paytansyopay1 + payfukusyopay1..5
    BUT: get_payouts() SQL does not SELECT paytanshou columns

  engine.build_payout_map()
    Reads payfukusyoumaban/payfukusyopay only
    Returns dict[(race_id, umaban) -> float]  (place payout multiplier)

Layer 2: BacktestEngine.run() -- Candidate Selection
  Line 504: candidate_df = self._race_predictor.get_place_candidates(...)
  Line 886: ev_col = "ev_place_corrected" / "ev_place"
  Line 544: diagnostic logs ev_place, p_place_pred, e_return_place_pred, etc.
  Line 609: odds_val = float(row["fukuoddslow"])
  Line 633: bet_type=BetType.PLACE
  Line 281: final_odds_map uses fukuoddslow

Layer 3: RacePredictor.select_bets()
  Line 552: candidates = self.get_place_candidates(...)
  Line 609: odds_val = float(row["fukuoddslow"])
  Line 633: Bet(bet_type=BetType.PLACE, ...)
  No get_win_candidates() method exists

Layer 4: Settlement
  engine._settle_bet() line 943: BetType.PLACE checks finish_pos 1-3
  engine._settle_bet() line 946: BetType.WIN checks finish_pos == 1
  BUT: payout_map only contains place payouts (Layer 1 gap)
  AND: final_odds_map uses fukuoddslow, not tanoddslow
```

### Target Architecture: Dual-Mode Backtest

```
PROPOSED ARCHITECTURE
=====================

BacktestEngine
  +-- betting_target: BetType = BetType.WIN  (constructor param, default=WIN)
  |
  +-- run(test_start, test_end)
  |     1. Load data (unchanged)
  |     2. Build features (unchanged)
  |     3. Pre-compute context features (unchanged)
  |     4. For each race:
  |         a. predict() -> result_df (unchanged -- already computes win + place EV)
  |         b. get_candidates() -> delegates to get_win_candidates() or get_place_candidates()
  |         c. select_bets() -> uses tanoddslow (win) or fukuoddslow (place)
  |         d. settle_bet() -> uses win_payout_map or place_payout_map
  |         e. log diagnostics -> win-specific or place-specific columns
  |
  +-- build_win_payout_map(payouts_df)  (NEW)
  +-- build_place_payout_map(payouts_df)  (renamed from build_payout_map)

RacePredictor
  +-- get_win_candidates(race_df, regime_params)  (NEW -- mirrors get_place_candidates)
  +-- select_bets()  (MODIFIED -- dispatches based on self.betting_target)
  +-- get_candidates()  (NEW -- thin dispatcher)
```

## Component Boundaries

### Components to MODIFY (existing)

| Component | File | What Changes | Why |
|-----------|------|-------------|-----|
| BacktestEngine | `src/backtest/engine.py` | Add `betting_target` param, add `build_win_payout_map()`, modify `run()` loop to dispatch on target, modify `final_odds_map` to use `tanoddslow` for win mode | Core change: engine must support win settlement and win candidate selection |
| RacePredictor | `src/backtest/race_predictor.py` | Add `get_win_candidates()` method, modify `select_bets()` to support win mode | The inference chain already computes win EV columns; only the candidate selection + bet generation layer is missing |
| DiagnosticLogger | `src/backtest/diagnostic_logger.py` | Add win-specific HorseDiagnostic fields (p_win_pred, ev_win, win_selection_ev, etc.) | Win-mode diagnostics must log win model outputs, not place outputs |
| EveryDB2Queries | `src/db/everydb2_queries.py` | Add `paytansyoumaban1, paytansyopay1` to `get_payouts()` SQL SELECT | Win payout data exists in EveryDB2 but is not extracted |
| ParquetStore ETL | `src/db/etl.py` | Verify `paytansyoumaban1`/`paytansyopay1` are in type rules (already there), ensure schema migration | Data pipeline must persist win payouts to Parquet |
| Backtest Schema | `src/db/schema.py` | Schema already has `tan_umaban` + `tan_pay` in payouts table; no schema change needed | Already accounted for |
| run_backtest.py | `scripts/run_backtest.py` | Add `--betting-target win\|place` CLI argument | User-facing entry point must expose target selection |
| run_wf_validation.py | `scripts/run_wf_validation.py` | Default to win-mode target | WF validation currently runs place-mode implicitly |

### Components to CREATE (new)

| Component | File | Responsibility | Inserted At |
|-----------|------|---------------|-------------|
| `build_win_payout_map()` | `src/backtest/engine.py` (add to module) | Parse paytansyoumaban1/paytansyopay1 into (race_id, umaban) -> odds_multiplier dict | Called in `BacktestEngine.run()` after loading payouts |
| `get_win_candidates()` | `src/backtest/race_predictor.py` (add method) | Select win-bet candidates using win_selection_ev, win_selection_edge, win_selection_prob, WinSelectionGateModel | Called from `BacktestEngine.run()` race loop |
| FeatureComputer (optional) | `src/features/feature_computer.py` | Extract shared feature pre-computation logic from BacktestEngine + TrainingPipelineV5 | Used by both callers to avoid duplication |

### Components UNCHANGED

| Component | Why Unchanged |
|-----------|--------------|
| FeatureEngine | `build_all()` already generates all features for both win and place models |
| TrainingPipelineV5 | Already trains WinTwoStageModel, WinSelectionGateModel, WinBenterGate |
| WinTwoStageModel | Already produces `ev_win`, `ev_win_corrected` columns during predict |
| WinSelectionGateModel | Already trained, stored, and invoked in predict chain |
| EVCorrectionModel | Win EV correction already works; `ev_corrector.correct_ev(df)` at line 117 of race_predictor.py operates on win columns |
| WinBenterGate | Already applied during predict() at line 120-128 of race_predictor.py |
| SubmodelSet | Already holds `win_selection_gate`, `win_benter`, `win_isotonic_calibrator`, `win_temperature_scaler` |
| BetType enum | Already has `WIN = "win"` value |
| Domain types | Bet dataclass already supports `BetType.WIN`; `_settle_bet()` already handles `finish_pos == 1` check for win |
| RegimeDetector | Regime detection uses pre-race features only; no bet-type dependency |
| RaceQualityScreener | Quality screening uses market-level features; no bet-type dependency |
| ParquetStore | Data storage layer is format-agnostic |
| Parquet ETL type rules | Already include `paytansyoumaban1` (int) and `paytansyopay1` (float) in `_TABLE_TYPE_RULES["payouts"]` |

## Data Flow Changes

### Current Flow (Place-mode only)

```
EveryDB2 (s_harai/n_harai)
  -> get_payouts() SQL: payfukusyoumaban1-5, payfukusyopay1-5  [MISSING win payout]
  -> ETL to Parquet: data/raw/payouts.parquet
  -> BacktestEngine.run():
       build_payout_map() -> place payout dict
       final_odds_map: (race_id, umaban) -> fukuoddslow
       RacePredictor.get_place_candidates() -> place edge/ev threshold filtering
       RacePredictor.select_bets() -> BetType.PLACE with fukuoddslow odds
       _settle_bet() -> check place payout map, fallback to finish_pos <= 3
```

### Target Flow (Dual-mode)

```
EveryDB2 (s_harai/n_harai)
  -> get_payouts() SQL: payfukusyoumaban1-5, payfukusyopay1-5,
                        paytansyoumaban1, paytansyopay1           [ADDED win payout]
  -> ETL to Parquet: data/raw/payouts.parquet (now includes tan columns)
  -> BacktestEngine.run(betting_target=WIN):
       build_win_payout_map()  -> win payout dict (NEW)
       build_place_payout_map() -> place payout dict (renamed)
       final_odds_map: (race_id, umaban) -> tanoddslow            [CHANGED for win]
       RacePredictor.get_win_candidates() -> win edge/ev threshold filtering (NEW)
       RacePredictor.select_bets() -> BetType.WIN with tanoddslow odds (MODIFIED)
       _settle_bet() -> check win payout map, fallback to finish_pos == 1
```

### Win Candidate Selection Flow (NEW)

```
result_df (from RacePredictor.predict())
  already contains: win_selection_ev, win_selection_edge, win_selection_prob,
                    win_gate_score, win_gate_pass, tanoddslow, ev_win, ev_win_corrected,
                    EV_lower_win_corrected, conformal_confidence_score

get_win_candidates():
  1. Ensure win_selection_columns via ensure_win_selection_columns()
  2. Get regime_params (edge_threshold, min_win_prob, max_win_odds)
  3. If WinSelectionGateModel.is_trained:
       a. score(df) -> win_gate_score, win_gate_pass
       b. annotate_race_context(df) -> aggressive_strength, aggressive_tier
       c. Hard gate: win_gate_pass AND edge >= 0 AND prob >= min AND odds <= max
       d. Soft gate: near-threshold candidates (buffer zones)
       e. Runner-up: add_second/rescue candidates from aggressive regime
  4. Else (no gate model):
       Simple threshold: win_selection_edge >= edge_threshold AND prob >= min AND odds <= max
  5. Sort by win_gate_score (or edge) descending
  6. Return candidates DataFrame

select_bets() [win path]:
  1. Get candidates from get_win_candidates()
  2. For each candidate:
       odds_val = tanoddslow (NOT fukuoddslow)
       edge_val = win_selection_edge
       ev_val = win_selection_ev
       Bet(bet_type=BetType.WIN, odds=tanoddslow, ...)
  3. Return bet list
```

## Patterns to Follow

### Pattern 1: Parallel Method Pattern (get_win_candidates mirrors get_place_candidates)

**What:** The `get_win_candidates()` method should be structurally identical to `get_place_candidates()`, operating on win-specific columns instead of place-specific columns.
**When:** Implementing win candidate selection.

The WinSelectionGateModel already has `score()`, `annotate_race_context()`, `soft_pass_mask()`, and `runner_up_candidate_reason()` methods that are the exact analogues of PlaceSelectionGateModel's methods. The mapping is:

```
Place column              -> Win column
place_selection_ev        -> win_selection_ev
place_selection_edge      -> win_selection_edge
place_selection_prob      -> win_selection_prob
place_gate_score          -> win_gate_score
place_gate_pass           -> win_gate_pass
fukuoddslow               -> tanoddslow
edge_place                -> edge_win
```

### Pattern 2: Betting Target Dispatch Pattern

**What:** BacktestEngine and RacePredictor accept a `betting_target` parameter and dispatch to target-specific methods.
**When:** Any code path that differs between win and place.

```python
class BacktestEngine:
    def __init__(self, models, betting_target: BetType = BetType.WIN, ...):
        self.betting_target = betting_target

    def run(self, test_start, test_end):
        # ...
        if self.betting_target == BetType.WIN:
            self.payout_map = self.build_win_payout_map(payouts_df)
            odds_col = "tanoddslow"
        else:
            self.payout_map = self.build_place_payout_map(payouts_df)
            odds_col = "fukuoddslow"
        # ...

class RacePredictor:
    def get_candidates(self, race_df, *, regime_params=None, target=BetType.WIN):
        if target == BetType.WIN:
            return self.get_win_candidates(race_df, regime_params=regime_params)
        else:
            return self.get_place_candidates(race_df, regime_params=regime_params)

    def select_bets(self, race_df, bankroll, *, candidates=None, target=BetType.WIN):
        if candidates is None:
            candidates = self.get_candidates(race_df, target=target)
        # ... use target-appropriate odds column
```

### Pattern 3: Payout Map Builder Pattern

**What:** Separate payout map builders for each bet type, following the existing `build_payout_map()` pattern.
**When:** Building settlement data from payouts Parquet.

```python
def build_win_payout_map(payouts_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Build (race_id, umaban) -> win odds multiplier from payouts DataFrame.

    paytansyopay is 'yen per 100 yen bet', so divide by 100 for multiplier.
    """
    payout_map: dict[tuple[str, int], float] = {}
    if payouts_df.empty:
        return payout_map
    for _, row in payouts_df.iterrows():
        race_id = str(row.get("race_id", ""))
        umaban = row.get("paytansyoumaban1") or row.get("tan_umaban")
        pay = row.get("paytansyopay1") or row.get("tan_pay")
        if pd.notna(umaban) and pd.notna(pay):
            try:
                payout_map[(race_id, int(umaban))] = float(pay) / 100.0
            except (ValueError, TypeError):
                continue
    return payout_map
```

Note: The column names may differ between EveryDB2 raw (`paytansyoumaban1`, `paytansyopay1`) and the Parquet schema (`tan_umaban`, `tan_pay`). The ETL type rules map these during extraction. The payout map builder must handle both naming conventions.

### Pattern 4: Feature Pre-Computation Extraction

**What:** Extract the repeated feature pre-computation block into a shared class.
**When:** Both BacktestEngine and TrainingPipeline compute the same features.

```python
class FeatureComputer:
    """Pre-computes expensive features once, shared across pipeline stages."""

    def __init__(self, store: ParquetStore):
        self.store = store
        self._cache: dict[str, pd.DataFrame] = {}

    def compute_context_features(
        self, race_df: pd.DataFrame, entry_df: pd.DataFrame, race_ids: np.ndarray
    ) -> dict[str, pd.DataFrame]:
        """Compute horse_history, jockey, trainer, jt_combo features for all races.

        Returns dict keyed by feature name for per-race lookup.
        """
        # HorseHistoryFeatures
        hist = HorseHistoryFeatures(store=self.store)
        hist_df = hist.compute(race_df, entry_df, race_ids)

        # JockeyContextFeatures
        jockey_ctx = JockeyContextFeatures(self.store)
        jockey_df = jockey_ctx.compute(entry_df)

        # TrainerContextFeatures
        trainer_ctx = TrainerContextFeatures(self.store)
        trainer_df = trainer_ctx.compute(entry_df)

        # JockeyTrainerComboFeatures
        jt_combo = JockeyTrainerComboFeatures(self.store)
        jt_df = jt_combo.compute(entry_df)

        return {
            "hist": hist_df,
            "jockey": jockey_df,
            "trainer": trainer_df,
            "jt_combo": jt_df,
        }

    def compute_derived_features(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Compute pace_aptitude, course, and sire features."""
        # PaceAptitudeFeatures
        pace_feat = PaceAptitudeFeatures(store=self.store)
        pace_df = pace_feat.compute_batch(feat_df)
        # ... merge logic ...

        # CourseFeatures
        course_feat = CourseFeatures(store=self.store)
        course_df = course_feat.compute_batch(feat_df)
        # ... merge logic ...

        # SireFeatures
        # ... same as current BacktestEngine.run() lines 356-376 ...

        return feat_df
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Duplicating RacePredictor Logic

**What:** Copy-pasting the entire get_place_candidates() method and changing column names.
**Why bad:** 600+ lines of logic (gate scoring, pruning, aggressive runner-up, soft pass) would need to be maintained in two places. Any fix or improvement to one would need manual synchronization to the other.
**Instead:** Create `get_win_candidates()` that reuses the same structural pattern but with a column-name mapping. Better yet, refactor the shared logic into private helper methods parameterized by column names:

```python
def _get_candidates_impl(self, race_df, *, regime_params, target):
    col_map = {
        "selection_ev": "win_selection_ev" if target == WIN else "place_selection_ev",
        "selection_edge": "win_selection_edge" if target == WIN else "place_selection_edge",
        "selection_prob": "win_selection_prob" if target == WIN else "place_selection_prob",
        "gate_score": "win_gate_score" if target == WIN else "place_gate_score",
        "gate_pass": "win_gate_pass" if target == WIN else "place_gate_pass",
        "odds_col": "tanoddslow" if target == WIN else "fukuoddslow",
    }
    gate_model = (submodel.win_selection_gate if target == WIN
                  else submodel.place_selection_gate)
    # ... shared logic using col_map ...
```

### Anti-Pattern 2: Re-Running ETL for Win Payouts Separately

**What:** Creating a separate ETL step or Parquet file for win payouts.
**Why bad:** The payouts Parquet already exists; win payout data comes from the same EveryDB2 table (s_harai / n_harai). Adding a separate file doubles I/O and introduces sync issues.
**Instead:** Modify the existing `get_payouts()` SQL query to include win payout columns. The ETL type rules already list `paytansyoumaban1` and `paytansyopay1`. The schema already has `tan_umaban` and `tan_pay` columns. Just add the columns to the SQL SELECT and the payout map builder.

### Anti-Pattern 3: Breaking Place-Mode Backtest

**What:** Changing BacktestEngine in a way that breaks existing place-mode testing.
**Why bad:** The project may need to compare win vs place ROI side-by-side. Place-mode backtest must continue to work.
**Instead:** Default to `betting_target=BetType.WIN` (as the milestone requires) but keep the `BetType.PLACE` path fully functional. All existing tests mock the BacktestEngine; ensure new tests cover both targets.

### Anti-Pattern 4: Vectorizing Inside the Race Loop

**What:** Trying to vectorize operations that are inherently per-race (bankroll tracking, DD controller updates).
**Why bad:** Bankroll state is sequential -- each bet depends on the previous bet's outcome. This cannot be parallelized.
**Instead:** Vectorize the feature computation (before the loop) and the settlement aggregation (after the loop), but keep the loop for stateful operations. The pre-computation block (lines 329-405 of engine.py) is the main vectorization target.

## Pipeline Optimization Architecture

### Current Bottleneck Analysis

The backtest pipeline has three distinct phases with the following time profile (based on ~57 min/year from CLAUDE.md):

```
Phase 1: Data Load + Feature Build (~15 min, 26%)
  - load_races, load_entries, load_odds_snapshots, load_odds_time_series_range
  - extract_pre_post_odds (year-by-year)
  - FeatureEngine.build_all() -- 14 sub-modules
  - SubModelManager.add_distance_band_features()

Phase 2: Pre-Computation (~8 min, 14%)
  - HorseHistoryFeatures.compute() -- per-race historical queries
  - JockeyContextFeatures.compute() -- jockey stats
  - TrainerContextFeatures.compute() -- trainer stats
  - JockeyTrainerComboFeatures.compute() -- combo stats
  - PaceAptitudeFeatures.compute_batch()
  - CourseFeatures.compute_batch()
  - SireFeatures.compute_batch()

Phase 3: Race-by-Race Loop (~34 min, 60%)
  - Per-race: predict() -> candidate selection -> bet generation -> settlement
  - Per-race: filter pre-computed DataFrames by race_id (4 DataFrame filters)
  - Per-race: merge result_df with race_df_single columns
  - Per-race: DiagnosticLogger.log_horse() for each horse (iterrows)
```

### Optimization Strategy

#### Optimization 1: Parquet-Backed Feature Cache

**Target:** Phase 2 pre-computation (8 min)
**Approach:** Cache expensive feature computation results to Parquet files keyed by (feature_name, date_range). On subsequent runs with overlapping date ranges, load from cache instead of recomputing.

```python
class CachedFeatureComputer:
    """Feature computation with Parquet-backed caching."""

    def __init__(self, store: ParquetStore, cache_dir: Path = Path("data/features")):
        self.store = store
        self.cache_dir = cache_dir

    def compute_horse_history(self, race_df, entry_df, race_ids):
        cache_key = f"horse_history_{race_ids.min()}_{race_ids.max()}"
        cache_path = self.cache_dir / f"{cache_key}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        hist = HorseHistoryFeatures(store=self.store)
        result = hist.compute(race_df, entry_df, race_ids)
        result.to_parquet(cache_path, index=False)
        return result
```

**Risk:** Stale cache if underlying data changes. Mitigation: Use ETL timestamp or race_id range as cache invalidation key. The existing `data/features/horse_features.parquet` path in CLAUDE.md suggests this pattern was partially considered.

#### Optimization 2: Pre-Race DataFrame GroupBy Instead of Per-Race Filter

**Target:** Phase 3 per-race DataFrame filtering (~5-10 min of the 34 min loop)
**Approach:** Replace the 4 per-race DataFrame filters (hist_df, jockey_df, trainer_df, jt_df each filtered by race_id) with a pre-built groupby dictionary.

```python
# Current (O(n_races * 4) filter operations):
for race_id in race_ids:
    hist_df_race = hist_df_all[hist_df_all["race_id"] == race_id]       # filter
    jockey_df_race = jockey_df_all[jockey_df_all["race_id"] == race_id] # filter
    trainer_df_race = trainer_df_all[trainer_df_all["race_id"] == race_id]
    jt_df_race = jt_df_all[jt_df_all["race_id"] == race_id]

# Optimized (O(n_total) groupby + O(1) dict lookup):
hist_groups = dict(iter(hist_df_all.groupby("race_id")))
jockey_groups = dict(iter(jockey_df_all.groupby("race_id")))
trainer_groups = dict(iter(trainer_df_all.groupby("race_id")))
jt_groups = dict(iter(jt_df_all.groupby("race_id")))

for race_id in race_ids:
    hist_df_race = hist_groups.get(race_id, pd.DataFrame())
    jockey_df_race = jockey_groups.get(race_id, pd.DataFrame())
    trainer_df_race = trainer_groups.get(race_id, pd.DataFrame())
    jt_df_race = jt_groups.get(race_id, pd.DataFrame())
```

This converts O(n_races * n_rows) filtering to O(n_rows) groupby + O(n_races) dict lookup. For ~5000 races with ~80000 entries, this is a significant speedup.

#### Optimization 3: Batch Diagnostic Logging

**Target:** Phase 3 per-horse iterrows logging (~5-8 min of the 34 min loop)
**Approach:** Replace per-horse `iterrows()` + `log_horse()` with batch DataFrame collection and bulk write at the end.

```python
# Current: ~14 log_horse calls per race * ~5000 races = ~70000 individual log_horse calls
# Each call creates a HorseDiagnostic dataclass and appends to a list

# Optimized: Collect horse diagnostic rows into a DataFrame during the loop,
# write once at the end via DiagnosticLogger
```

#### Optimization 4: Skip Feature Pre-Computation When Using Cached Models

**Target:** Phase 2 (8 min) when `--skip-train` is used
**Approach:** When `--skip-train` loads cached models, the feature pre-computation for the test period can be parallelized since there is no training dependency.

**Note:** The existing `--skip-train` flag already skips training but still recomputes all features. This is by design (reproducibility), but for rapid iteration, an additional `--skip-features` flag that loads cached feature Parquets would reduce backtest-only runs from ~57 min to ~34 min.

### Optimization Priority

| Optimization | Time Saved | Complexity | Risk | Priority |
|-------------|-----------|------------|------|----------|
| GroupBy dict for per-race filtering | 5-10 min | Low | None | HIGH |
| Batch diagnostic logging | 5-8 min | Low | None | HIGH |
| Feature Parquet cache | 5-8 min | Medium | Stale cache | MEDIUM |
| Skip-features flag | 8 min (conditional) | Low | None | LOW |

## Scalability Considerations

| Concern | Current (1 year) | With Win Mode | With Optimization | Multi-Year (5 years) |
|---------|-----------------|---------------|-------------------|---------------------|
| Backtest time | ~57 min | ~57 min (same data volume) | ~35-40 min | ~175-285 min (linear) |
| Memory (backtest) | ~3 GB | ~3 GB (no new data) | ~2.5 GB (groupby is more efficient) | ~6-8 GB |
| Payout map size | ~15000 entries (place) | ~5000 entries (win, fewer winners) | Same | ~75000 entries |
| Parquet size (payouts) | ~5 MB | ~6 MB (+ 1 win column) | Same | ~30 MB |
| Diagnostic output | ~70 MB CSV/year | ~70 MB (same rows, different cols) | ~50 MB (batch write) | ~350 MB |

## Build Order Recommendation

The build order prioritizes the win-mode conversion first (it is the milestone goal), then optimization as a separate pass.

```
Phase 1: Win Payout Data Pipeline (ETL layer)
  1a. Modify get_payouts() SQL to include paytanshoumaban1, paytansyopay1
  1b. Re-run ETL for affected date range to update payouts.parquet
  1c. Add build_win_payout_map() to engine.py
  1d. Verify win payout data loads correctly (unit test with mock Parquet)

  Rationale: Data must flow before anything else can work. This is the
  foundation for all win-mode settlement.

Phase 2: Win Candidate Selection (RacePredictor layer)
  2a. Add get_win_candidates() to RacePredictor
      - Mirror get_place_candidates() structure
      - Use WinSelectionGateModel.score()/annotate_race_context()
      - Use win_selection_ev, win_selection_edge, win_selection_prob
      - Use tanoddslow for odds filtering
  2b. Add betting_target dispatch to select_bets()
  2c. Add BetType.WIN path in select_bets() using tanoddslow
  2d. Unit tests with mock result DataFrames

  Rationale: The inference chain already produces all win EV columns.
  Only the candidate selection + bet generation layer needs to be
  activated for win mode.

Phase 3: BacktestEngine Integration
  3a. Add betting_target parameter to BacktestEngine.__init__()
  3b. Modify run() to use tanoddslow for final_odds_map in win mode
  3c. Modify run() to call get_win_candidates() in win mode
  3d. Add win-specific diagnostic logging fields
  3e. Modify _settle_bet() to use win_payout_map for WIN bets
  3f. Integration test: run single-year backtest in win mode

  Rationale: This wires the new data + candidate selection into the
  simulation loop. Each sub-change is small and testable.

Phase 4: Script and WF Validation Updates
  4a. Add --betting-target to run_backtest.py CLI
  4b. Update run_wf_validation.py to use win mode by default
  4c. Update BacktestResult.summary() to show win-specific metrics
  4d. Run full WF validation in win mode

  Rationale: Scripts are the user-facing layer. Must be updated after
  the engine changes are validated.

Phase 5: Pipeline Optimization (separate pass)
  5a. Replace per-race DataFrame filters with groupby dict
  5b. Batch diagnostic logging
  5c. Add FeatureComputer class (optional, for cache abstraction)
  5d. Add --skip-features flag for rapid iteration

  Rationale: Optimization does not change outcomes, only speed. It
  should come after the win-mode conversion is validated to avoid
  conflating two types of changes.
```

## Integration Verification Checklist

After each phase, verify:

- [ ] Phase 1: `load_payouts()` returns DataFrame with tan_umaban/tan_pay columns (or raw EveryDB2 names)
- [ ] Phase 1: `build_win_payout_map()` produces non-empty dict for known win races
- [ ] Phase 2: `get_win_candidates()` returns non-empty DataFrame for races with strong win EV
- [ ] Phase 2: Candidates have win_selection_ev > 1.0 and tanoddslow > 0
- [ ] Phase 3: `_settle_bet()` correctly settles WIN bets (finish_pos == 1 pays, else 0)
- [ ] Phase 3: Bankroll tracking works correctly for WIN bets
- [ ] Phase 3: Diagnostic logs contain win-specific columns (ev_win, win_selection_ev, etc.)
- [ ] Phase 4: `run_backtest.py --betting-target win` completes without errors
- [ ] Phase 4: WF validation produces win-mode ROI results
- [ ] Phase 5: GroupBy optimization produces identical backtest results
- [ ] Phase 5: Timing comparison shows speedup without result changes
- [ ] All phases: Existing tests pass (`python -m pytest tests/ -v`)
- [ ] All phases: mypy type checking passes (`mypy src/`)

## Confidence Assessment

| Area | Confidence | Notes |
|------|-----------|-------|
| Win payout data availability | HIGH | EveryDB2 type rules already list paytansyoumaban1/paytansyopay1; schema has tan_umaban/tan_pay |
| WinSelectionGateModel completeness | HIGH | Already trained, stored, loaded, and invoked; has score(), annotate_race_context(), soft_pass_mask(), runner_up_candidate_reason() |
| Win EV column availability | HIGH | race_predictor.py predict() already computes ev_win, ev_win_corrected, win_selection_ev/edge/prob, win_gate_score/pass |
| GroupBy optimization correctness | HIGH | Pure data structure change; groupby preserves all rows |
| Backtest result equivalence | HIGH | Settlement logic already handles BetType.WIN in _settle_bet() |
| Pipeline speedup estimates | MEDIUM | Based on code structure analysis, not profiling data |

## Sources

- `src/backtest/engine.py` -- BacktestEngine.run(), build_payout_map(), _settle_bet()
- `src/backtest/race_predictor.py` -- RacePredictor.get_place_candidates(), select_bets(), predict()
- `src/backtest/diagnostic_logger.py` -- HorseDiagnostic dataclass (place-only fields)
- `src/pipelines/training_pipeline.py` -- TrainingPipelineV5._train_submodel(), feature computation
- `src/db/everydb2_queries.py` -- get_payouts() SQL (missing tanshou columns)
- `src/db/etl.py` -- _TABLE_TYPE_RULES (includes paytansyoumaban1, paytansyopay1)
- `src/db/schema.py` -- payouts table schema (has tan_umaban, tan_pay)
- `src/models/win_selection_gate.py` -- WinSelectionGateModel (fully implemented)
- `src/domain/models.py` -- SubmodelSet, Bet, BetType
- `scripts/run_backtest.py` -- CLI entry point
- `scripts/run_wf_validation.py` -- WF validation entry point
