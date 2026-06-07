---
phase: 51-settlement-integrity-training-pipeline
reviewed: 2026-06-06T12:00:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - src/betting/payout_maps.py
  - src/backtest/engine.py
  - src/paper_trading/reconciler.py
  - scripts/run_paper_trading.py
  - scripts/run_train.py
  - src/pipelines/training_pipeline.py
  - src/features/feature_engine.py
  - src/db/model_loader.py
  - tests/test_payout_maps.py
  - tests/test_paper_reconciler.py
  - tests/test_model_loader.py
findings:
  critical: 2
  warning: 6
  info: 3
  total: 11
status: issues_found
---

# Phase 51: Code Review Report

**Reviewed:** 2026-06-06
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Reviewed 11 files across the settlement integrity and training pipeline overhaul. The payout_maps extraction is clean and well-tested. The PaperReconciler overhaul introduces a sound 3-column state model with proper schema validation and atomic writes. The ModelLoader explicit source selection (D-16) is correctly implemented with mutual exclusivity checks.

Two critical issues were found: (1) the reconciler silently mishandles wide bets by treating them as place bets and losing the umaban_b partner, and (2) a potential crash on Windows due to `Path.replace()` when the target file is held open by another process. Six warnings cover data integrity risks and logic gaps.

## Critical Issues

### CR-01: PaperReconciler silently mishandles wide bets -- loses umaban_b partner

**File:** `src/paper_trading/reconciler.py:198-201`
**Issue:** When `bet_type` is `"wide"`, the reconciler falls into the `else` branch and uses `place_map` (build_payout_map) for lookup. This is wrong for two reasons:
1. Wide payout maps use `(race_id, umaban_lo, umaban_hi)` keys, not `(race_id, umaban)`. The lookup `(race_id, umaban)` will never match a wide map entry.
2. Wide bets have a partner horse (`umaban_b`) which is ignored -- the bet is settled as lost regardless of actual result.

The function `build_wide_payout_map` is imported in `engine.py` but never imported or used in `reconciler.py`. While paper trading currently generates only place bets (RacePredictor with default betting_target), the code explicitly sets `bet_type` from `bet.bet_type.value` in `run_paper_trading.py:573`, meaning any future wide bet would be silently mishandled rather than raising an error.

**Fix:**
```python
# In reconciler.py, add import at top:
from betting.payout_maps import build_payout_map, build_win_payout_map, build_wide_payout_map

# In reconcile(), add wide payout map construction:
wide_map = build_wide_payout_map(payouts_df)

# In the settlement loop, handle wide bets:
if bet_type == "win":
    pmap = win_map
elif bet_type == "wide":
    # Wide settlement requires partner umaban_b
    umaban_b = int(row.get("umaban_b", 0))
    if umaban_b == 0:
        logger.warning("Wide bet missing umaban_b for %s umaban=%d, keeping pending", race_id, umaban)
        continue
    lo, hi = min(umaban, umaban_b), max(umaban, umaban_b)
    wide_key = (race_id, lo, hi)
    if wide_key in wide_map:
        multiplier = wide_map[wide_key]
        if multiplier <= 0:
            continue
        payout = stake * multiplier
        bets_df.at[idx, "outcome"] = "won"
        bets_df.at[idx, "payout"] = payout
        bets_df.at[idx, "settlement_status"] = "settled"
        n_wins += 1
    else:
        bets_df.at[idx, "outcome"] = "lost"
        bets_df.at[idx, "payout"] = 0.0
        bets_df.at[idx, "settlement_status"] = "settled"
    n_settled += 1
    continue
else:
    pmap = place_map
```

### CR-02: Atomic write on Windows may crash when target file is open

**File:** `src/paper_trading/reconciler.py:70-80`
**Issue:** The `_atomic_write_parquet` method uses `tmp_path.replace(target)` which on Windows raises `PermissionError` if the target file is open (e.g., by a concurrent `pd.read_parquet` call, an antivirus scanner, or a file explorer). Since the project runs on Windows (the development environment is Windows 11), this is a real risk in production.

The `reconcile()` method reads `bets.parquet` at the top (line 149), then writes it back at line 239. If `retry_pending()` calls `reconcile()` in a loop and another process reads the file between write iterations, the replace can fail.

**Fix:**
```python
import os

@staticmethod
def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    """Atomic Parquet write via NamedTemporaryFile + replace (D-07)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        dir=str(target.parent),
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path, index=False)
    # Windows: os.replace is more robust than Path.replace for open targets
    max_retries = 3
    for attempt in range(max_retries):
        try:
            os.replace(str(tmp_path), str(target))
            return
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            import time
            time.sleep(0.1)
```

## Warnings

### WR-01: PaperReconciler n_refunded and n_voided counters are never incremented

**File:** `src/paper_trading/reconciler.py:187-188`
**Issue:** The local variables `n_refunded` and `n_voided` are initialized to 0 but never modified. They are passed to `_compute_roi` which returns them as-is. The `_compute_roi` method does compute totals from the DataFrame directly, so the return values are correct -- but the function signature and the local counters are misleading dead code.

**Fix:** Remove `n_refunded` and `n_voided` local variables and the parameters from `_compute_roi`, or add actual refunded/voided settlement logic.

### WR-02: Reconciler bets_df iteration uses .at[] which is slow for row-by-row mutation

**File:** `src/paper_trading/reconciler.py:190-235`
**Issue:** The reconciliation loop iterates over `pending.index` and sets values one at a time using `bets_df.at[idx, col] = value`. While functionally correct, this is O(n) pandas operations and could be vectorized. For the typical paper trading scale (< 50 bets/day), this is acceptable but worth noting.

**Fix:** Low priority -- the current approach is correct and readable for the expected data volumes.

### WR-03: build_wide_payout_map length-3 heuristic may misparse edge-case kumi values

**File:** `src/betting/payout_maps.py:141-159`
**Issue:** The heuristic for 3-character kumi strings uses `first_two <= 18` to decide whether to split as (XX, Y) or (X, YY). This is reasonable for JRA where horse numbers range 1-18, but for 3-character kumi like "218" (first_two=21 > 18), it splits as (2, 18) which is correct. However, "019" (first_two=01, treated as int 1 <= 18) would split as (1, 9) instead of potentially (0, 19). While horse number 0 does not exist, the Parquet float conversion could produce edge cases.

The existing test coverage is good for common cases (513 -> 5,13; 111 -> 11,1) but does not test boundary values like "118" (first_two=11, split as 11,8) or "181" (first_two=18, split as 18,1).

**Fix:** Add tests for boundary kumi values: "118", "181", "918" (9,18), "109" (10,9).

### WR-04: PaperReconciler compute_bet_id cannot produce unique IDs for wide bets with umaban_b

**File:** `src/paper_trading/reconciler.py:64-67`
**Issue:** The D-02 spec defines `bet_id = SHA256(session_id|race_id|bet_type|umaban)[:32]`. Wide bets involve two horses (umaban + umaban_b), but the hash input only includes one umaban. Two wide bets on different pairs in the same race/session would produce the same bet_id. Currently paper trading only generates place bets, but this is a latent design flaw.

**Fix:** Include `umaban_b` in the hash when bet_type is "wide":
```python
@staticmethod
def compute_bet_id(session_id: str, race_id: str, bet_type: str, umaban: int, umaban_b: int | None = None) -> str:
    raw = f"{session_id}|{race_id}|{bet_type}|{umaban}"
    if umaban_b is not None:
        raw += f"|{umaban_b}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
```

### WR-05: run_paper_trading.py does not pass `store` to PaperReconciler, using MagicMock instead

**File:** `scripts/run_paper_trading.py:972-975`
**Issue:** In `_run_reconcile`, the `PaperReconciler` is constructed with `store=store if store is not None else MagicMock()`. The `store` parameter is available from the caller (`store=store` on line 1285), but the reconciler itself never uses `store` -- it reads `bets_path` directly via `pd.read_parquet` and fetches payouts from `everydb2`. The MagicMock fallback is harmless but misleading; the `store` parameter in `PaperReconciler.__init__` appears to be unused vestigial code.

**Fix:** Remove the `store` parameter from `PaperReconciler.__init__` since it is never used internally, or document why it is retained.

### WR-06: race_date column stored as string "YYYYMMDD" in predict bets but as pd.Timestamp in reconcile filter

**File:** `scripts/run_paper_trading.py:582` and `src/paper_trading/reconciler.py:163-164`
**Issue:** In `_run_predict`, bets are created with `"race_date": ymd` where `ymd = target_date.strftime("%Y%m%d")` -- this is a string like "20260405". But in `reconciler.py:163-164`, the filter compares `bets_df["race_date"] == target_ts` where `target_ts = pd.Timestamp(target_date)`. When Parquet stores the string "20260405" and it is read back, the comparison `string == Timestamp` may fail depending on pandas type inference. The tests use `pd.Timestamp(race_date)` in `_make_bet_row` which avoids this, but the production code path stores a plain string.

**Fix:** In `run_paper_trading.py`, change line 582 to use a consistent type:
```python
"race_date": pd.Timestamp(ymd),
```

## Info

### IN-01: payout_maps.py build_wide_payout_map has unused lo/hi swap for length-5 kumi

**File:** `src/betting/payout_maps.py:168-183`
**Issue:** The length-5 path (lines 168-183) uses a split_at_2 / split_at_3 heuristic similar to length-3, but `split_at_3` is never populated -- only `split_at_2` and `split_at_3` are defined, and `idx5[~use_first_two]` is assigned to `split_at_3`. The logic is correct but the variable name `split_at_3` is slightly misleading since it refers to the complement of the first-two-digits check, not a literal split at position 3.

**Fix:** Consider renaming `split_at_3` to `split_at_first_two_complement` for clarity, or adding a comment.

### IN-02: Feature cache includes track_conditions and horse_track_aptitude as dependencies

**File:** `src/features/feature_engine.py:232-233`
**Issue:** The feature cache key computation now includes `("raw", "track_conditions")` and `("raw", "horse_track_aptitude")` as source paths. This is correct and ensures cache invalidation when these files change. However, if these files do not exist, they are silently skipped (no path added to `source_paths`). This means the cache key could be identical with or without these files, potentially serving stale cache if the files are created after initial caching.

**Fix:** Low priority -- the cache is keyed on source content, and the features computed from these files would simply be NaN if the files don't exist.

### IN-03: ModelLoader.load() does not populate ModelInfo.betting_target from MLflow run

**File:** `src/db/model_loader.py:491-496`
**Issue:** When loading from MLflow `run_id`, the `ModelInfo` is created without setting `betting_target`, so it defaults to `"place"`. Only `load_from_dir` reads `betting_target` from `meta.json` (line 963). This means paper trading and other MLflow-based load paths will always report `betting_target="place"` even if the model was trained with `--betting-target win`.

**Fix:** Add `betting_target` to the MLflow params during training (`_log_to_mlflow`) and read it back in `load()`:
```python
# In _log_to_mlflow:
mlflow.log_param("betting_target", self._betting_target)

# In load():
betting_target = params.get("betting_target", "place")
info = ModelInfo(..., betting_target=betting_target)
```

---

_Reviewed: 2026-06-06_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
