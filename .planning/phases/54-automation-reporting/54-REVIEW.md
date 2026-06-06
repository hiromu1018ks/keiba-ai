---
phase: 54-automation-reporting
reviewed: 2026-06-06T00:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - src/paper_trading/exit_codes.py
  - src/paper_trading/race_progress.py
  - src/paper_trading/report_aggregator.py
  - src/paper_trading/run_orchestrator.py
  - src/paper_trading/report.py
  - scripts/run_paper_trading.py
  - tests/test_race_progress.py
  - tests/test_report_aggregator.py
  - tests/test_run_orchestrator.py
  - tests/test_cli_run_mode.py
  - tests/test_paper_trading_report.py
findings:
  critical: 2
  warning: 4
  info: 2
  total: 8
fixed:
  - CR-01 (commit 503e83b)
  - CR-02 (commit 503e83b)
  - WR-04 (commit 503e83b)
  - WR-01 (commit TBD -- _compute_max_dd initial_bankroll fix)
  - WR-02 (commit TBD -- bankroll instance cache)
  - WR-03 (commit TBD -- shared _validate_bet_schema_basic)
remaining:
  - IN-01 (info, acceptable fallback)
  - IN-02 (info, minor UX)
status: fixes_applied
---

# Phase 54: Code Review Report

**Reviewed:** 2026-06-06
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Reviewed 11 files (5 source modules, 1 CLI script, 5 test files) comprising the automation and reporting layer for paper trading. Found 2 critical bugs in `run_orchestrator.py` where the orchestrator's `--mode run` path ignores the user-specified `--betting-target` flag, always defaulting to `"place"`. Also found 4 warnings including a max-drawdown calculation that silently returns 0.0 for all-loss scenarios, a race condition in bankroll tracking, an unused `_build_race_predictor` helper method, and a schema validation discrepancy between the aggregator and reconciler.

## Critical Issues

### CR-01: `RunModeOrchestrator._predict_single_race` ignores `--betting-target`, always defaults to "place"

**File:** `src/paper_trading/run_orchestrator.py:314`
**Issue:** `_predict_single_race` creates `RacePredictor(self.models)` with no arguments, and calls `select_bets(result_df, bankroll)` at line 342 without passing `betting_target`. The `RacePredictor.select_bets` signature is `def select_bets(self, race_df, bankroll, *, candidates=None, betting_target="place")`, so it always defaults to `"place"`. If the user passes `--betting-target win`, the orchestrator still generates place bets. This is a functional correctness bug that silently produces the wrong bet type.

**Fix:**
```python
# Line 314: pass betting_target when constructing RacePredictor
race_predictor = RacePredictor(self.models)

# Line 342: pass betting_target from args
betting_target = getattr(self.args, "betting_target", "place")
bets = race_predictor.select_bets(result_df, bankroll, betting_target=betting_target)
```

### CR-02: `RunModeOrchestrator._build_race_predictor` ignores strategy config and betting_target

**File:** `src/paper_trading/run_orchestrator.py:659-662`
**Issue:** The `_build_race_predictor` helper constructs `RacePredictor(self.models)` without passing any kwargs -- no `betting_target`, no `dd_shadow_only`, no `stake_calculator`, no `odds_band_filter`. This means the orchestrator's run mode diverges from the `_run_predict` CLI path in `scripts/run_paper_trading.py` which carefully configures OddsBandFilter, DrawdownController, and other parameters. While the method is currently unused in the prediction path (line 314 inlines its own `RacePredictor` construction), it is called nowhere, suggesting it was intended for use but the inline construction at line 314 was a mistake. If anyone later switches to calling `_build_race_predictor`, the same CR-01 bug would manifest. The method should either be deleted or properly implemented.

**Fix:**
```python
def _build_race_predictor(self) -> Any:
    """Build RacePredictor with full config parity to scripts/run_paper_trading.py."""
    from backtest.race_predictor import RacePredictor

    kwargs: dict[str, Any] = {
        "betting_target": getattr(self.args, "betting_target", "place"),
        "dd_shadow_only": True,
    }
    # Add OddsBandFilter, StakeCalculator, etc. from strategy_config if available
    return RacePredictor(self.models, **kwargs)
```

## Warnings

### WR-01: `_compute_max_dd` returns 0.0 when all bets lose

**File:** `src/paper_trading/report.py:85-93`
**Issue:** The max drawdown computation starts with `peak = 0.0` and `cumulative = 0.0`. If every bet loses (pnl < 0), cumulative goes negative but peak stays at 0.0. Since `peak > 0` is False, `dd` is always 0.0. The method reports 0.0% drawdown even when 100% of capital is lost. This gives users a false sense of safety.

**Fix:** Initialize peak from the initial bankroll or track absolute cumulative:
```python
@staticmethod
def _compute_max_dd(bets: list[dict[str, Any]], initial_bankroll: float = 100000.0) -> float:
    if not bets:
        return 0.0
    cumulative = initial_bankroll
    peak = initial_bankroll
    max_dd = 0.0
    for b in bets:
        pnl = float(b.get("payout", 0) or 0) - float(b.get("stake", 0) or 0)
        cumulative += pnl
        peak = max(peak, cumulative)
        dd = (peak - cumulative) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd
```

### WR-02: Bankroll race condition in `_predict_single_race` -- bankroll read is stale

**File:** `src/paper_trading/run_orchestrator.py:341`
**Issue:** `_compute_current_bankroll()` at line 341 reads `bets.parquet` to compute the current bankroll, but the bankroll is also decremented in-memory at line 369 for each bet. If `_predict_single_race` is called sequentially for multiple races, the bankroll read from `bets.parquet` on disk may not reflect the in-memory bankroll decrements from previous bets in the same session. This can lead to over-betting beyond the actual bankroll.

**Fix:** Track bankroll as instance state on the orchestrator, updated after each bet, rather than re-reading from parquet each time:
```python
# In __init__:
self._current_bankroll: float = 0.0  # lazily initialized

# In _predict_single_race:
if self._current_bankroll == 0.0:
    self._current_bankroll = self._compute_current_bankroll()
bankroll = self._current_bankroll
# ... after placing bets:
self._current_bankroll = bankroll
```

### WR-03: Schema validation discrepancy between `report_aggregator._validate_bet_schema` and `reconciler._validate_bet_schema`

**File:** `src/paper_trading/report_aggregator.py:76-86`
**Issue:** The aggregator defines its own `_validate_bet_schema` function (line 76) with a different required-columns list than the `PaperReconciler._validate_bet_schema` method. The aggregator checks `("schema_version", "settlement_status", "outcome", "payout", "bet_id", "stake")` while the reconciler may check a different set. Having two independent schema validators risks accepting data in one place that the other rejects, or vice versa.

**Fix:** Consolidate into a single shared validator, e.g., import and reuse `PaperReconciler._validate_bet_schema` in the aggregator.

### WR-04: `_cross_validate_race` has dead-code branch

**File:** `src/paper_trading/run_orchestrator.py:570-575`
**Issue:** Both branches of the if/else in `_cross_validate_race` return the same expression: `progress.verify_bet_ids_present(race_id, bets_df)`. The `if bets_df.empty` branch at lines 570-573 is effectively dead code -- it does the same thing as the `else` at line 575. This suggests either an incomplete implementation (the empty case should behave differently) or unnecessary branching.

**Fix:** Simplify to a single return, or differentiate the empty case if a different behavior was intended:
```python
def _cross_validate_race(self, race_id: str, bets_df: pd.DataFrame, progress: RaceProgress) -> bool:
    return progress.verify_bet_ids_present(race_id, bets_df)
```

## Info

### IN-01: `report.py` uses `subprocess` to get git hash -- may fail in non-git environments

**File:** `src/paper_trading/report.py:54-63`
**Issue:** `generate()` calls `git rev-parse --short HEAD` via subprocess. This will produce a warning and "unknown" commit hash when run outside a git repository (e.g., in a Docker container or packaged deployment). The error is already handled (lines 62-63), but the warning may confuse operators.

**Fix:** No action required -- the fallback to "unknown" is acceptable. Consider adding a debug-level log when this occurs.

### IN-02: `parse_args()` in `run_paper_trading.py` requires `--betting-target` and `--betting-mode` for all modes including `setup` and `reconcile`

**File:** `scripts/run_paper_trading.py:268-275`
**Issue:** `--betting-target` and `--betting-mode` are marked `required=True`, so `setup` and `reconcile` modes that do not need these parameters must still provide them. This is a minor UX issue but not a bug.

**Fix:** Consider making these conditional on mode, or removing `required=True` and validating in the mode handlers.

---

## Fixes Applied (--fix)

### WR-01 Fix: `_compute_max_dd` — initial_bankroll パラメータ追加

**File:** `src/paper_trading/report.py:77-97`

`cumulative` と `peak` を `initial_bankroll` (default: 100,000) から開始するよう変更。全損シナリオでも正しいドローダウン率を返す。

### WR-02 Fix: Bankroll インスタンスキャッシュ

**File:** `src/paper_trading/run_orchestrator.py:71,432-449,398-399`

- `__init__` に `_current_bankroll: float | None` を追加（遅延初期化）
- `_compute_current_bankroll()` をキャッシュ対応に変更（初回のみ disk 読み込み）
- `_predict_single_race` のベット保存後に `self._current_bankroll = bankroll` で更新

### WR-03 Fix: スキーマバリデーション統一（DRY）

**Files:** `src/paper_trading/reconciler.py:94-115`, `src/paper_trading/report_aggregator.py`

- `PaperReconciler._validate_bet_schema_basic()` (基本チェック: 旧スキーマ拒否 + 必須列) を新設
- `PaperReconciler._validate_bet_schema()` は basic を呼び出し後に書き込み時の厳密検証を追加
- `report_aggregator` の重複定義を削除し `PaperReconciler._validate_bet_schema_basic()` に委譲

### Verification

```
47/47 tests passed (report + aggregator + orchestrator + CLI + race_progress)
ruff check: no new errors (3 pre-existing line-length in reconciler.py)
```

### Remaining (Info, no fix required)

- **IN-01:** `subprocess` git hash — fallback "unknown" is acceptable
- **IN-02:** `--betting-target` required for all modes — minor UX, deferred

---

_Reviewed: 2026-06-06_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
_Fixes applied: 2026-06-06_
