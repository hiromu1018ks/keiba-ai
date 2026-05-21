# Phase 11: Bet Selection Filters - Research

**Researched:** 2026-05-04
**Domain:** Backtest bet quality filtering (EV lower bound, regime skip, odds band exclusion)
**Confidence:** HIGH

## Summary

Phase 11 adds three sequential filters to the backtest pipeline to improve bet quality: (1) a hard EV lower bound filter (`EV_lower_win_corrected >= 1.0`) in `get_win_candidates()`, (2) a COLLAPSED regime race-level skip in `BacktestEngine.run()`, and (3) a dynamic `OddsBandFilter` that excludes unprofitable odds bands based on training-period ROI. All three filters are already architecturally mapped with clear insertion points in the existing code. The data flow is well-understood: `EV_lower_win_corrected` is computed by `RobustConfidenceEstimator.predict_interval()` during `RacePredictor.predict()` and is available in `result_df` before candidate selection. The `RegimeDetector` already classifies COLLAPSED states; the task is simply to act on that classification earlier in the race loop. The `OddsBandFilter` is the only new component and follows the same `pd.cut()` + `groupby` + ROI calculation pattern already established in `BacktestReportGenerator._compute_condition_stats()`.

**Primary recommendation:** Implement the three filters in the specified order (COLLAPSED skip -> EV filter -> OddsBandFilter), using the established patterns for DataFrame filtering and statistics logging. The OddsBandFilter should be calibrated from accumulated training-period bet data during the race loop itself, not from a separate data load, to keep the architecture simple and avoid look-ahead bias.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 既存 `win_selection_edge > 0` に加えて `EV_lower_win_corrected >= 1.0` のハードフィルターを追加。二重フィルター構成
- **D-02:** フィルターは `get_win_candidates()` 内で早期適用
- **D-03:** `EV_lower_win_corrected` が NaN の場合、既存 `win_selection_edge > 0` のみで判定
- **D-04:** 除外統計ログ出力: 除外件数・EV_lower < 1.0 の割合をINFO レベルでログ出力
- **D-05:** 動的解析アプローチ。トレーニング期間のベットデータから各オッズバンドのROIを自動計算。ルックアヘッドバイアス防止
- **D-06:** 新規クラス `OddsBandFilter` を `src/betting/odds_band_filter.py` に作成
- **D-07:** 赤字判定条件: トレーニング期間ROI < 100% のバンドを除外
- **D-08:** 除外統計ログ出力: 除外バンド名・件数・各バンドのROIをINFO レベルでログ出力
- **D-09:** フィルター適用順序: COLLAPSED skip (race-level) -> EV filter (candidate-level) -> OddsBandFilter (candidate-level)
- **D-10:** ベット数ガード: 残存ベット数が1,000件/年未満なら WARNING。自動緩和は行わない
- **D-11:** COLLAPSEDスキップ: BacktestEngine.run() のレースループ内で `regime == RegimeState.COLLAPSED` なら `continue`。スキップ件数をカウントしてログ出力

### Claude's Discretion
- EV_lowerフィルターの具体的なpandasフィルター条件の実装
- OddsBandFilterのインターフェース設計（calibrate() + filter() メソッド等）
- バンド境界定義（Phase 9レポートと同じ 1.0-3.0/3.0-10.0/10.0-30.0/30.0+）
- 除外統計ログのフォーマット（INFO レベル、構造化ログ）
- WARNING の出力条件とフォーマット
- レポート拡張の具体的なコード変更（report.py への除外済みバンド表示追加）
- テスト戦略（フィルターごとの単体テスト + 統合テスト）

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BSEL-01 | EV_lower_win_corrected >= 1.0 を満たさないベットが自動除外される | `EV_lower_win_corrected` column is computed in `predict()` via `predict_interval()` (race_predictor.py:149-151). Filter goes in `get_win_candidates()` (race_predictor.py:408-470) alongside existing `win_selection_edge > 0` mask. |
| BSEL-02 | COLLAPSEDレースでベットが完全スキップされる | `RegimeDetector.detect()` returns `RegimeState.COLLAPSED` (regime_detector.py:133-176). Insert `continue` at engine.py:687 after regime detection. `get_strategy_params()` already has COLLAPSED branch (regime_detector.py:221-232). |
| BSEL-03 | 赤字バンドのベットがOddsBandFilterで除外される | Follow `_compute_condition_stats()` odds band pattern (report.py:398-406): `pd.cut` with bands 1.0-3.0/3.0-10.0/10.0-30.0/30.0+. New class in `src/betting/odds_band_filter.py`. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| EV lower bound filtering | API / Backend (ML pipeline) | -- | Filter applied in RacePredictor.get_win_candidates() during candidate selection, purely data transformation |
| COLLAPSED race skip | API / Backend (backtest engine) | -- | Skip logic lives in BacktestEngine.run() race loop, before any candidate processing |
| Odds band dynamic filtering | API / Backend (backtest engine) | -- | OddsBandFilter calibrated and called during BacktestEngine.run(), operates on candidate DataFrames |
| Exclusion statistics logging | API / Backend (backtest engine) | -- | Counters accumulated in engine race loop, logged at INFO level |
| Report exclusion display | API / Backend (report generation) | -- | BacktestReportGenerator consumes filter statistics |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | already installed | DataFrame filtering, groupby, pd.cut for banding | Existing codebase standard for all data operations |
| numpy | already installed | Numeric operations, NaN handling | Required by pandas, used throughout |
| pytest | already installed | Unit/integration testing with mocks | Project test standard (all tests use mocks, no DB) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock | stdlib | Mocking for tests | All existing tests use MagicMock/patch patterns |
| logging | stdlib | INFO/WARNING log output | All components use Python logging |
| dataclasses | stdlib | Data structures (BacktestResult, Bet) | Existing pattern for result containers |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual pd.cut banding | sklearn KBinsDiscretizer | pd.cut is simpler and matches existing report.py pattern exactly. KBinsDiscretizer adds unnecessary dependency. |
| Custom filter class method | Standalone function | Class with calibrate()+filter() follows established pattern (RobustConfidenceEstimator, RegimeDetector) and provides clean state encapsulation |

**No new dependencies needed for this phase.** [VERIFIED: codebase analysis -- all imports are from existing stdlib/pandas/numpy]

## Architecture Patterns

### System Architecture Diagram

```
BacktestEngine.run()
  |
  v
[Race Loop Start] --> Load race_df --> predict() --> result_df with EV_lower_win_corrected
  |
  v
[Regime Detection] --> detect() --> regime state
  |
  v
[COLLAPSED? -- D-11] --> YES --> skip counter++ --> continue (skip race)
  |
  v (NO)
[get_win_candidates() -- D-01/D-02]
  |   Filter: win_selection_edge > 0
  |   Filter: EV_lower_win_corrected >= 1.0 (NaN -> fallback to edge-only)  <-- NEW
  |   Sort by win_gate_score DESC -> max 2 candidates
  v
[OddsBandFilter.filter() -- D-06]
  |   Band: pd.cut(tanodds, bands 1.0-3.0/3.0-10.0/10.0-30.0/30.0+)
  |   Exclude candidates in bands with training ROI < 100%
  v
[select_bets()] --> Bet objects --> settlement
  |
  v
[Race Loop End] --> Log exclusion stats (D-04, D-08, D-10)
  |
  v
[Report Generation] --> BacktestReportGenerator + AI diagnostics with exclusion info
```

### Recommended Project Structure
```
src/
  betting/
    odds_band_filter.py    # NEW: OddsBandFilter class
  backtest/
    race_predictor.py       # MODIFY: get_win_candidates() EV filter
    engine.py               # MODIFY: COLLAPSED skip + OddsBandFilter integration
    report.py               # MODIFY: exclusion stats display
  models/
    regime_detector.py      # MODIFY: get_strategy_params() COLLAPSED skip=True
```

### Pattern 1: DataFrame Candidate Filtering
**What:** Filter candidates in `get_win_candidates()` using boolean mask on DataFrame columns.
**When to use:** EV lower bound filter follows the exact same pattern as existing `win_selection_edge > 0` filter.
**Example:**
```python
# Source: race_predictor.py:426-431 (existing pattern)
selection_edge = pd.to_numeric(race_df[edge_col], errors="coerce")
odds = pd.to_numeric(race_df[odds_col], errors="coerce")
mask = selection_edge.fillna(0.0) > 0.0
mask &= odds.fillna(0.0) >= 1.0

# NEW: EV lower bound filter (D-01, D-02, D-03)
ev_lower_col = "EV_lower_win_corrected"
if ev_lower_col in race_df.columns:
    ev_lower = pd.to_numeric(race_df[ev_lower_col], errors="coerce")
    # D-03: NaN -> True (fallback to edge-only)
    ev_mask = ev_lower.fillna(1.0) >= 1.0
    mask &= ev_mask
```

### Pattern 2: Odds Band Analysis (pd.cut + groupby)
**What:** Bin continuous odds into bands using lambda or pd.cut, then compute ROI per band.
**When to use:** OddsBandFilter calibration and filtering use the same banding scheme as report.py.
**Example:**
```python
# Source: report.py:398-406 (existing pattern)
odds_multiplier_bands = _band_stats(
    bets,
    lambda b: (
        "1.0-3.0" if b.get("tanoddslow", 0) < 3.0
        else "3.0-10.0" if b.get("tanoddslow", 0) < 10.0
        else "10.0-30.0" if b.get("tanoddslow", 0) < 30.0
        else "30.0+"
    ),
    ["1.0-3.0", "3.0-10.0", "10.0-30.0", "30.0+"],
)
```

### Pattern 3: Race Loop Counter + End-of-Loop Logging
**What:** Increment counters during race loop, log summary at INFO level after loop completes.
**When to use:** All exclusion statistics (EV excluded, COLLAPSED skipped, band excluded).
**Example:**
```python
# Source: engine.py:597-598, 997-1000 (existing pattern)
n_pre_post_odds_bets = 0  # counter
# ... during loop:
n_pre_post_odds_bets += len(bets)
# ... after loop:
logger.info("Total bets: %d", len(bet_history))
```

### Anti-Patterns to Avoid
- **Look-ahead bias in OddsBandFilter calibration:** NEVER use test-period data to determine which bands to exclude. Only accumulate training-period bets. The engine.run() method receives only test dates -- calibration data must come from elsewhere or from the accumulated first portion of the test data after sufficient accumulation. [CITED: CONTEXT.md D-05]
- **Filtering after select_bets():** Filters must happen BEFORE bet generation to avoid wasted computation. The filter chain (COLLAPSED -> EV -> OddsBand) all happen before `select_bets()` is called. [CITED: CONTEXT.md D-09]
- **Auto-relaxation of filters:** If bet count drops below 1000/year, only log a WARNING -- do not automatically weaken filters. Parameter tuning belongs in Phase 13. [CITED: CONTEXT.md D-10]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Odds band binning | Custom if-else banding function | Same lambda pattern as report.py:398-406 | Consistency with existing banding scheme, same band boundaries |
| NaN-safe numeric filtering | Custom NaN checking per row | `pd.to_numeric(col, errors="coerce").fillna(fallback)` | pandas handles NaN/inf edge cases, existing pattern in get_win_candidates() |
| Exclusion counter tracking | Custom metrics class | Simple integer counters in engine.run() race loop | Existing pattern (n_pre_post_odds_bets), no need for abstraction |
| ROI calculation per band | Custom aggregation | pandas groupby -> sum(result)/sum(stake) | Same pattern as _band_stats() in report.py:339-373 |

**Key insight:** Every component in this phase has a direct precedent in the existing codebase. No novel algorithms needed.

## Common Pitfalls

### Pitfall 1: OddsBandFilter Calibration Data Availability
**What goes wrong:** `BacktestEngine.run()` only receives test_start/test_end -- it has no access to training-period bet data. If OddsBandFilter requires training-period ROI by band, the data source must be explicitly provided.
**Why it happens:** The engine.run() signature is `run(self, test_start, test_end)` with no training parameters. The OddsBandFilter.calibrate() call needs historical bet results with odds and outcomes.
**How to avoid:** Two viable approaches:
  - **Option A (Recommended):** Calibrate OddsBandFilter from accumulated early test-period data with a warm-up buffer (e.g., first 500 races). This avoids needing to change the engine.run() signature and naturally prevents look-ahead since only past data within the test period is used.
  - **Option B:** Add training bet history as a parameter to BacktestEngine.__init__() or engine.run(), calibrated externally in run_backtest.py. This is cleaner conceptually but requires API changes.
  Given D-05 says "training period data", Option B aligns better with the decision text. The run_backtest.py script has access to both train and test dates and can pre-calibrate the filter.
**Warning signs:** If the filter is calibrated with test-period outcomes, look-ahead bias exists.

### Pitfall 2: EV_lower Column Name Mismatch
**What goes wrong:** The column is named `EV_lower_win_corrected` (uppercase) in the DataFrame, but code might reference `ev_lower_win_corrected` (lowercase).
**Why it happens:** The codebase uses mixed casing -- `ev_win_corrected` (lowercase) vs `EV_lower_win_corrected` (uppercase). The uppercase version is set in `robust_confidence_estimator.py:179`.
**How to avoid:** Always use the exact column name `EV_lower_win_corrected` (uppercase). Verify with `if "EV_lower_win_corrected" in race_df.columns` before accessing. [VERIFIED: robust_confidence_estimator.py:179]
**Warning signs:** KeyError or NaN-only filter when column name is wrong.

### Pitfall 3: COLLAPSED Skip Prevents Regime Transition Data
**What goes wrong:** If all COLLAPSED races are skipped before accumulating `recent_stats_list`, the regime detector may not transition out of COLLAPSED since it never sees recovery data.
**Why it happens:** `recent_stats_list` is appended at engine.py:967-991 AFTER the bet loop. If we `continue` before this append, no stats are collected for COLLAPSED races.
**How to avoid:** Accumulate `recent_stats_list` BEFORE the COLLAPSED skip check. The regime detection needs market statistics from ALL races, including COLLAPSED ones, to properly detect transitions. Move the COLLAPSED skip to AFTER the stats accumulation. [VERIFIED: engine.py:966-991 -- stats are accumulated after bet settlement but before the loop continues]
**Warning signs:** Regime stuck in COLLAPSED permanently after first transition.

### Pitfall 4: Empty Candidates After Combined Filters
**What goes wrong:** After applying both edge > 0 AND EV_lower >= 1.0, too many candidates are excluded, leaving 0 candidates for some races.
**Why it happens:** In the current backtest (89% ROI), many bets have EV < 1.0. The dual filter may exclude a large fraction.
**How to avoid:** This is expected behavior (D-01 specifically uses dual filter for safety). The bet count guard (D-10) monitors this: if total bets drop below 1000/year, WARNING is logged. No auto-relaxation. [CITED: CONTEXT.md D-10]
**Warning signs:** Total bet count drops significantly -- but this is the INTENDED outcome of the quality filters.

### Pitfall 5: OddsBandFilter Using Wrong Odds Column
**What goes wrong:** The filter uses `odds` from bet history (which is `tanodds` at bet-decision time) vs `tanoddslow` (which the report uses for banding). These may differ.
**Why it happens:** Bet history stores `bet.odds` which comes from `row.get("tanodds", 0)` (race_predictor.py:629). The report bands on `tanoddslow` (report.py:401). `tanodds` is the predicted odds; `tanoddslow` is the final pre-race odds. They can differ slightly.
**How to avoid:** For OddsBandFilter, use `tanodds` (the odds available at decision time) for banding since this is what the candidate DataFrame has. The report's `tanoddslow` is only available post-race. For consistency, document that the filter uses decision-time odds.
**Warning signs:** Band boundaries differ between filter and report display.

## Code Examples

### EV Lower Bound Filter in get_win_candidates()
```python
# Source: race_predictor.py:408-470 (modification point)
# After line 431 (existing mask), add:

# D-01: EV lower bound filter (dual filter with edge > 0)
ev_lower_col = "EV_lower_win_corrected"
if ev_lower_col in race_df.columns:
    ev_lower = pd.to_numeric(race_df[ev_lower_col], errors="coerce")
    # D-03: NaN -> fallback to edge-only (fillna(1.0) passes the >= 1.0 check)
    ev_mask = ev_lower.fillna(1.0) >= 1.0
    mask &= ev_mask
```

### COLLAPSED Skip in BacktestEngine.run()
```python
# Source: engine.py:681-687 (insertion point)
# After regime_params = ... line, add:

# D-11: COLLAPSED regime skip (race-level)
if regime == RegimeState.COLLAPSED:
    n_collapsed_skipped += 1
    # NOTE: recent_stats_list must be appended BEFORE this check
    # to allow regime transitions
    continue
```

### RegimeDetector.get_strategy_params() Extension
```python
# Source: regime_detector.py:221-232 (modification point)
# In the COLLAPSED branch, add skip=True:

else:  # COLLAPSED
    return {
        "ev_threshold": 1.50,
        "edge_threshold": 0.09,
        "min_place_prob": 0.10,
        "max_place_odds": 16.0,
        "wide_enabled": False,
        "score_threshold": 0.050,
        "max_bets_per_race": 1,
        "weak_prob_prune_threshold": 0.35,
        "skip": True,  # NEW: D-11
        "description": "崩壊 -> ほぼ停止",
    }
```

### OddsBandFilter Class Interface
```python
# New file: src/betting/odds_band_filter.py

class OddsBandFilter:
    """D-06: 動的オッズバンドフィルター"""

    BANDS = [(1.0, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, float("inf"))]
    BAND_NAMES = ["1.0-3.0", "3.0-10.0", "10.0-30.0", "30.0+"]

    def __init__(self) -> None:
        self._excluded_bands: set[str] = set()
        self._band_roi: dict[str, float] = {}
        self._band_counts: dict[str, int] = {}

    def calibrate(self, bet_history: list[dict[str, Any]]) -> None:
        """D-05: Calculate ROI per band from training period bets. D-07: exclude ROI < 100%."""

    def filter(self, candidate_df: pd.DataFrame, odds_col: str = "tanodds") -> pd.DataFrame:
        """Exclude candidates in unprofitable odds bands."""

    @property
    def excluded_bands(self) -> dict[str, dict[str, Any]]:
        """D-08: Return excluded band names with their ROI and counts."""
```

### Exclusion Statistics Logging
```python
# After the race loop in engine.run(), before BacktestResult construction:
logger.info(
    "Bet Selection Filters: EV_excluded=%d (%.1f%%), COLLAPSED_skipped=%d, "
    "OddsBand_excluded=%d (bands: %s)",
    n_ev_excluded,
    100.0 * n_ev_excluded / max(n_total_candidates, 1),
    n_collapsed_skipped,
    n_band_excluded,
    odds_band_filter.excluded_bands if odds_band_filter else {},
)

# D-10: Bet count guard
bets_per_year = total_bets / n_test_years  # approximate
if bets_per_year < 1000:
    logger.warning(
        "Bet count guard: %.0f bets/year (below 1000 threshold). "
        "Consider parameter adjustment in Phase 13.",
        bets_per_year,
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single edge > 0 filter | Dual filter (edge > 0 + EV_lower >= 1.0) | This phase | Reduces low-confidence bets significantly |
| No regime-based skip | COLLAPSED race-level skip | This phase | Eliminates bets in unstable market conditions |
| No odds band filtering | Dynamic OddsBandFilter | This phase | Removes systematically unprofitable odds ranges |

**Not deprecated -- this phase is additive.**

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `EV_lower_win_corrected` column is always present in `result_df` when `get_win_candidates()` is called (set by `predict()` line 149-151) | Architecture Patterns | Filter silently falls back to edge-only (D-03 NaN fallback), so low risk |
| A2 | `recent_stats_list` accumulation (engine.py:966-991) happens AFTER the COLLAPSED skip would fire, requiring stats to be accumulated BEFORE the skip check | Pitfalls | Regime could get stuck in COLLAPSED -- medium risk, must verify insertion point |
| A3 | OddsBandFilter calibration should use training-period data passed from `run_backtest.py` rather than accumulated test data (aligns with D-05 "training period data") | Architecture Patterns | If wrong, could use warm-up approach instead -- both viable |
| A4 | The `tanodds` column (not `tanoddslow`) is the correct odds to use for OddsBandFilter banding since it's the decision-time odds | Pitfalls | Minor inconsistency with report.py which uses `tanoddslow` -- acceptable since filter operates pre-race |

**If this table is empty:** All claims in this research were verified or cited.

## Open Questions (RESOLVED)

1. **OddsBandFilter Calibration Data Source** — RESOLVED: `run()` に `training_bet_history` オプションパラメータを追加。呼び出し元 (`run_backtest.py`) がトレーニング期間のベットデータを供給。未提供の場合は OddsBandFilter は非アクティブ (除外バンドなし)。
   - What we know: D-05 says "training period bet data". `BacktestEngine.run()` only receives test dates.
   - Resolution: Plan 02 Task 2 adds `training_bet_history: list[dict[str, Any]] | None = None` parameter to `run()`. Caller provides training data; if absent, filter stays inactive.

2. **Statistics Accumulation vs COLLAPSED Skip Order** — RESOLVED: `recent_stats_list.append()` を regime_params 取得直後（COLLAPSED skip の前）に移動。これによりレジーム遷移に必要な全レースの統計が蓄積される。
   - What we know: `recent_stats_list` is appended at engine.py:966-991, after bet settlement. COLLAPSED skip at ~688 would skip stats.
   - Resolution: Plan 02 Task 1 moves `recent_stats_list.append()` to right after `regime_params = ...` (line 687), before the COLLAPSED `continue`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | All code | Yes (mise) | 3.11.x | -- |
| pandas | DataFrame operations | Yes | installed | -- |
| numpy | Numeric operations | Yes | installed | -- |
| pytest | Testing | Yes | installed | -- |
| PostgreSQL | Backtest execution verification | Not checked | -- | Tests use mocks only |

**Missing dependencies with no fallback:** None -- all changes are code-only using existing dependencies.

**Missing dependencies with fallback:** N/A

## Sources

### Primary (HIGH confidence)
- Code analysis: `src/backtest/race_predictor.py` lines 408-470 -- get_win_candidates() structure, filter pattern
- Code analysis: `src/backtest/engine.py` lines 603-1026 -- run() race loop, insertion points for COLLAPSED skip and OddsBandFilter
- Code analysis: `src/models/regime_detector.py` lines 133-232 -- detect(), get_strategy_params(), COLLAPSED handling
- Code analysis: `src/models/robust_confidence_estimator.py` lines 96-234 -- EV_lower_win_corrected computation
- Code analysis: `src/backtest/report.py` lines 155-416 -- odds band analysis pattern
- Code analysis: `src/domain/types.py` lines 29-34 -- RegimeState enum

### Secondary (MEDIUM confidence)
- Code analysis: `src/backtest/engine.py` lines 332-360 -- BacktestEngine.__init__(), betting_target setup
- Code analysis: `scripts/run_backtest.py` -- backtest orchestration, no training bet data currently passed to engine

### Tertiary (LOW confidence)
- None -- all findings verified against source code

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all existing codebase patterns
- Architecture: HIGH -- all insertion points and data flow verified against source code
- Pitfalls: HIGH -- all pitfalls identified from direct code analysis

**Research date:** 2026-05-04
**Valid until:** 2026-06-04 (stable -- no fast-moving dependencies)
