# Phase 51: Settlement Integrity & Training Pipeline - Pattern Map

**Mapped:** 2026-06-06
**Files analyzed:** 14 (6 new, 8 modified)
**Analogs found:** 14 / 14

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/betting/payout_maps.py` | utility | transform | `src/backtest/engine.py:163-229` | exact (extraction) |
| `src/paper_trading/reconciler.py` | service | CRUD | `src/paper_trading/reconciler.py` (existing) | exact (modification) |
| `src/backtest/engine.py` | service | CRUD | `src/backtest/engine.py` (existing) | exact (modification: import change) |
| `src/db/model_loader.py` | service | CRUD | `src/db/model_loader.py` (existing) | exact (modification) |
| `src/pipelines/training_pipeline.py` | service | CRUD | `src/pipelines/training_pipeline.py` (existing) | exact (modification) |
| `src/features/feature_engine.py` | service | transform | `src/features/feature_engine.py` (existing) | exact (modification) |
| `scripts/run_train.py` | config | request-response | `scripts/run_train.py` (existing) | exact (modification) |
| `scripts/run_paper_trading.py` | config | request-response | `scripts/run_paper_trading.py` (existing) | exact (modification) |
| `tests/test_payout_maps.py` | test | transform | `tests/test_paper_reconciler.py` | role-match |
| `tests/test_paper_reconciler.py` | test | CRUD | `tests/test_paper_reconciler.py` (existing) | exact (modification) |
| `tests/test_model_loader.py` | test | CRUD | `tests/test_model_loader.py` (existing) | exact (modification) |
| `tests/test_training_pipeline.py` | test | CRUD | `tests/test_training_pipeline.py` (existing) | exact (modification) |
| `src/domain/types.py` | model | N/A | `src/domain/types.py` (existing) | exact (minor addition) |
| `src/domain/models.py` | model | N/A | `src/domain/models.py` (existing) | no change needed |

## Pattern Assignments

### `src/betting/payout_maps.py` (utility, transform) -- NEW

**Analog:** `src/backtest/engine.py` lines 163-229 (functions being extracted)

**Imports pattern** (new file, follow project convention):
```python
"""Payout map construction -- pure functions, no I/O.

Extracted from backtest/engine.py for shared use by BT and PT.
"""

import pandas as pd
```

**Core pattern -- `build_payout_map`** (extract from engine.py:163-208):
```python
def build_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) → odds_multiplier のマップを構築。

    payfukusyopay は「100円あたりの円」なので、100で割って倍率に変換する。
    ベクトル化: melt + groupby で一括処理。同一 (race_id, umaban) の最大値を保持。
    """
    if payouts_df.empty:
        return {}
    id_vars = ["race_id"]
    maban_cols = [f"payfukusyoumaban{i}" for i in range(1, 6)]
    pay_cols = [f"payfukusyopay{i}" for i in range(1, 6)]

    maban_melted = payouts_df[id_vars + maban_cols].melt(
        id_vars=id_vars,
        value_vars=maban_cols,
        value_name="umaban",
    )
    pay_melted = payouts_df[id_vars + pay_cols].melt(
        id_vars=id_vars,
        value_vars=pay_cols,
        value_name="pay",
    )

    combined = pd.DataFrame(
        {
            "race_id": maban_melted["race_id"].values,
            "umaban": maban_melted["umaban"].values,
            "pay": pay_melted["pay"].values,
        }
    )
    combined = combined.dropna(subset=["umaban", "pay"])
    combined["umaban"] = combined["umaban"].astype(int)
    combined["pay_100"] = combined["pay"] / 100.0

    # 同一 (race_id, umaban) の最大値を保持
    idx = combined.groupby(["race_id", "umaban"], observed=True)["pay_100"].idxmax()
    deduped = combined.loc[idx]

    payout_map: dict[tuple[str, int], float] = {}
    for race_id, umaban, pay_100 in zip(
        deduped["race_id"].values, deduped["umaban"].values, deduped["pay_100"].values
    ):
        payout_map[(str(race_id), int(umaban))] = float(pay_100)
    return payout_map
```

**Core pattern -- `build_win_payout_map`** (extract from engine.py:211-229):
```python
def build_win_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) → odds_multiplier のマップを構築 (単勝用)。

    paytansyopay1 は「100円あたりの円」なので、100で割って倍率に変換する。
    ベクトル化: dropna → astype → dict comprehension。
    """
    if payouts_df.empty:
        return {}
    df = payouts_df.dropna(subset=["paytansyoumaban1", "paytansyopay1"]).copy()
    if df.empty:
        return {}
    df["umaban"] = df["paytansyoumaban1"].astype(int)
    df["pay_100"] = df["paytansyopay1"] / 100.0
    return {
        (str(race_id), int(umaban)): float(pay_100)
        for (race_id, umaban), pay_100 in df.set_index(["race_id", "umaban"])["pay_100"].items()
    }
```

**Important naming note:** CONTEXT.md refers to `build_place_payout_map` but the actual function in `engine.py:163` is `build_payout_map`. Extract using the real name. Optionally add `build_place_payout_map = build_payout_map` alias for clarity.

**Constraint (D-12):** No EveryDB2 access or file I/O in this module. Pure functions only.

---

### `src/paper_trading/reconciler.py` (service, CRUD) -- MODIFIED

**Analog:** `src/paper_trading/reconciler.py` (existing, lines 1-153)

**Imports pattern** (existing + additions):
```python
"""Paper Trading 結果照合・ROI計算"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from betting.payout_maps import build_payout_map, build_win_payout_map

if TYPE_CHECKING:
    from db.everydb2_queries import EveryDB2Queries
    from db.parquet_store import ParquetStore

logger = logging.getLogger(__name__)
```

**Class constructor pattern** (existing lines 19-35, extend with retry config):
```python
class PaperReconciler:
    """reconcile フェーズ: 予測と結果を照合して ROI を追跡。

    冪等性: 同一 race_id + umaban のレコードが既に存在する場合はスキップ。
    """

    def __init__(
        self,
        store: ParquetStore,
        bets_path: Path,
        everydb2: EveryDB2Queries,
        monitor: Any | None = None,
        retry_interval: int = 60,
        retry_timeout: int = 600,
    ) -> None:
        self.store = store
        self.bets_path = bets_path
        self.everydb2 = everydb2
        self.monitor = monitor
        self.retry_interval = retry_interval
        self.retry_timeout = retry_timeout
```

**bet_id generation pattern** (D-02):
```python
@staticmethod
def compute_bet_id(session_id: str, race_id: str, bet_type: str, umaban: int) -> str:
    """bet_id = SHA256(session_id | race_id | bet_type | canonical_selection)[:32]"""
    payload = f"{session_id}|{race_id}|{bet_type}|{umaban}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]
```

**Atomic write pattern** (D-07, follows `src/db/etl.py:358-366`):
```python
@staticmethod
def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    """Atomic replace via temp file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        dir=target.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(target)  # Atomic on same filesystem
```

**Schema validation pattern** (D-18, D-19, D-20):
```python
@staticmethod
def _validate_bet_schema(df: pd.DataFrame) -> list[str]:
    """D-20: Write-time consistency validation."""
    errors: list[str] = []
    # Reject old schema (D-18)
    if "result" in df.columns and "payout" not in df.columns:
        errors.append("Old schema detected (result column without payout). Rejecting.")
        return errors
    # schema_version must be 2
    if "schema_version" in df.columns:
        if not (df["schema_version"] == 2).all():
            errors.append("schema_version must be 2 for all rows")
    else:
        errors.append("Missing schema_version column")
    # pending: outcome=NULL, payout=NULL
    pending = df[df["settlement_status"] == "pending"]
    if not pending.empty:
        if pending["outcome"].notna().any():
            errors.append("Pending bets must have outcome=NULL")
        if pending["payout"].notna().any():
            errors.append("Pending bets must have payout=NULL")
    # settled: outcome!=NULL, payout>=0
    settled = df[df["settlement_status"] == "settled"]
    if not settled.empty:
        if settled["outcome"].isna().any():
            errors.append("Settled bets must have outcome!=NULL")
        if (settled["payout"] < 0).any():
            errors.append("Settled bets must have payout>=0")
    # bet_id: non-null and unique
    if df["bet_id"].isna().any():
        errors.append("bet_id must be non-NULL")
    if df["bet_id"].duplicated().any():
        errors.append("bet_id must be unique")
    # stake > 0
    if (df["stake"] <= 0).any():
        errors.append("stake must be > 0")
    return errors
```

**ROI calculation pattern** (D-05):
```python
def _compute_roi(self, bets_df: pd.DataFrame) -> dict[str, float]:
    """D-05: ROI with effective_stake excluding refunded/voided."""
    settled = bets_df[bets_df["settlement_status"] == "settled"]
    effective = settled[settled["outcome"].isin(["won", "lost"])]
    effective_stake = effective["stake"].sum()
    total_return = effective["payout"].sum()
    roi = total_return / effective_stake if effective_stake > 0 else 0.0
    net_profit = total_return - effective_stake
    return {
        "effective_stake": float(effective_stake),
        "total_return": float(total_return),
        "roi": float(roi),
        "net_profit": float(net_profit),
    }
```

---

### `src/backtest/engine.py` (service, CRUD) -- MODIFIED

**Analog:** `src/backtest/engine.py` (existing)

**Change:** Replace inline `build_payout_map` and `build_win_payout_map` with imports from `payout_maps.py`.

**Import change** (add to existing imports):
```python
from betting.payout_maps import build_payout_map, build_win_payout_map
```

**Delete:** Remove `build_payout_map` (lines 163-208), `build_win_payout_map` (lines 211-229) from engine.py. The functions are now imported from `payout_maps.py`.

---

### `src/db/model_loader.py` (service, CRUD) -- MODIFIED

**Analog:** `src/db/model_loader.py` (existing, lines 39-931)

**Priority fix pattern** (D-16, modify `load()` method at lines 45-59):
```python
def load(
    self,
    run_id: str | None = None,
    *,
    models_dir: Path | None = None,
    use_ensemble: bool | None = None,
) -> tuple[TrainedModelsV5, ModelInfo]:
    """学習済みモデルを読み込み、TrainedModelsV5 を再構築。

    D-16: Explicit source selection.
    - run_id only → MLflow only (no local fallback)
    - models_dir only → local only (no MLflow)
    - neither → ERROR
    - both → ERROR (mutually exclusive)
    """
    if run_id is not None and models_dir is not None:
        raise ValueError(
            "Cannot specify both run_id and models_dir (mutually exclusive per D-16)"
        )
    if run_id is None and models_dir is None:
        raise ValueError(
            "Must specify either run_id or models_dir (no implicit selection per D-16)"
        )

    if models_dir is not None:
        return self.load_from_dir(models_dir, use_ensemble_override=use_ensemble)

    # run_id is not None → MLflow only
    # ... rest of existing MLflow loading logic ...
```

**track_stats restore pattern** (add to both `load()` and `load_from_dir()`, after SubmodelSet construction):
```python
# Restore track_stats from JSON artifacts
track_stats = None
try:
    ts_path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/track_stats_{surface}.json"
    )
    with open(ts_path) as f:
        track_stats = json.load(f)
except Exception:
    logger.debug("track_stats not found for %s, setting to None", surface)

track_month_stats = None
try:
    tms_path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/track_month_stats_{surface}.json"
    )
    with open(tms_path) as f:
        track_month_stats = json.load(f)
except Exception:
    logger.debug("track_month_stats not found for %s, setting to None", surface)

# Then include in SubmodelSet construction:
# track_stats=track_stats,
# track_month_stats=track_month_stats,
```

**betting_target validation pattern** (D-14):
```python
# After loading meta.json, verify betting_target match
meta_betting_target = meta.get("betting_target")
# Caller must pass expected betting_target; compare against meta
```

---

### `src/pipelines/training_pipeline.py` (service, CRUD) -- MODIFIED

**Analog:** `src/pipelines/training_pipeline.py` (existing)

**track_stats save pattern** (add after SubmodelSet construction at ~line 1570, inside `_train_submodel` and both `_save_models_local` and MLflow logging):

```python
# Save track_stats as JSON artifacts (D-15)
if sub.track_stats is not None:
    ts_path = models_dir / f"track_stats_{surface}.json"
    with open(ts_path, "w", encoding="utf-8") as f:
        json.dump(sub.track_stats, f, indent=2, ensure_ascii=False)
    # SHA256 for manifest (follow parameter_freeze_protocol.py pattern)
    ts_sha256 = hashlib.sha256(ts_path.read_bytes()).hexdigest()
    logger.info("track_stats SHA256 for %s: %s", surface, ts_sha256[:16])

if sub.track_month_stats is not None:
    tms_path = models_dir / f"track_month_stats_{surface}.json"
    with open(tms_path, "w", encoding="utf-8") as f:
        json.dump(sub.track_month_stats, f, indent=2, ensure_ascii=False)
    tms_sha256 = hashlib.sha256(tms_path.read_bytes()).hexdigest()
    logger.info("track_month_stats SHA256 for %s: %s", surface, tms_sha256[:16])
```

**MLflow artifact logging pattern** (follow existing pattern at lines 2097-2103):
```python
# Inside the MLflow logging block, after existing artifact logs:
if sub.track_stats is not None:
    _ts_tmp: str | None = None
    try:
        ts_path = models_dir / f"track_stats_{surface}.json"
        if ts_path.is_file():
            mlflow.log_artifact(str(ts_path), artifact_path=None)
    except Exception as e:
        logger.warning("Failed to log track_stats artifact for %s: %s", surface, e)
```

**meta.json betting_target field** (D-14, modify at line 2502-2511):
```python
meta = {
    "train_start": train_start,
    "train_end": train_end,
    "surfaces": list(models.keys()),
    "quality_threshold": quality_screen.threshold,
    "saved_at": pd.Timestamp.now().isoformat(),
    "use_ensemble": all(sub.use_ensemble for sub in models.values()),
    "betting_target": self._betting_target,  # D-14
}
```

---

### `src/features/feature_engine.py` (service, transform) -- MODIFIED

**Analog:** `src/features/feature_engine.py` (existing, lines 226-237)

**Cache dependency addition** (TRN-03, modify source_paths at lines 228-237):
```python
for cat, name in [
    ("raw", "races"),
    ("raw", "entries"),
    ("odds", "snapshots"),
    ("raw", "track_conditions"),       # TRN-03: track condition features
    ("raw", "horse_track_aptitude"),   # TRN-03: track aptitude features
]:
    p = data_dir / cat / name
    if p.with_suffix(".parquet").exists():
        source_paths.append(p.with_suffix(".parquet"))
    elif p.is_dir():
        source_paths.append(p)
```

Note: `horse_track_aptitude.parquet` may not exist yet. The existing pattern already handles this gracefully -- if the file doesn't exist, the path is simply not added to `source_paths`.

---

### `scripts/run_train.py` (config, request-response) -- MODIFIED

**Analog:** `scripts/run_train.py` (existing, lines 1-83)

**Argument addition pattern** (follow existing argparse style at lines 33-41):
```python
parser.add_argument(
    "--betting-target",
    choices=["win", "place", "wide"],
    default="place",
    help="Betting target scope: win=common+win models, place=common+win+place models (default: place)",
)
```

**Parquet validation pattern** (TRN-02, extend after line 50):
```python
# Pre-training Parquet validation (TRN-02)
store = ParquetStore()
if not store.exists("raw", "races"):
    logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
    sys.exit(1)

# Additional validation: date range, NaN rates
from datetime import datetime
import pandas as pd

races_df = store.read("raw", "races")
# Date range check
if "race_date" in races_df.columns:
    min_date = pd.to_datetime(races_df["race_date"]).min()
    max_date = pd.to_datetime(races_df["race_date"]).max()
    logger.info("Data range: %s ~ %s", min_date.date(), max_date.date())

# track_conditions check (TRN-02)
if not store.exists("raw", "track_conditions"):
    logger.warning("track_conditions.parquet not found. Track features will use defaults.")
```

**Wide rejection pattern** (D-13):
```python
if args.betting_target == "wide":
    logger.error("v2.4 does not support --betting-target wide. Use win or place.")
    sys.exit(1)
```

**Passing betting_target to pipeline** (modify line 68):
```python
models = pipeline.run(train_start, train_end, use_ensemble=args.ensemble, betting_target=args.betting_target)
```

---

### `scripts/run_paper_trading.py` (config, request-response) -- MODIFIED

**Analog:** `scripts/run_paper_trading.py` (existing, lines 899-1115)

**`_run_reconcile` thinning pattern** (D-01): Replace the 217-line inline reconciliation (lines 899-1115) with a thin wrapper:
```python
def _run_reconcile(
    args: argparse.Namespace,
    config: "PaperTradingConfig",
    models: "TrainedModelsV5 | None" = None,
    store: "ParquetStore | None" = None,
) -> None:
    """Thin CLI wrapper: build args, call PaperReconciler, display results, control exit code."""
    from paper_trading.reconciler import PaperReconciler

    target_date = date.fromisoformat(args.date)
    # ... instantiate reconciler ...
    reconciler = PaperReconciler(
        store=store,
        bets_path=config.bets_path,
        everydb2=everydb2,
        ...
    )
    result = reconciler.reconcile(target_date)
    # Display results + exit code control
    # ... formatting code for console output ...
```

The key change is: DELETE all inline settlement logic (lines 920-998: payout map construction, iterrows loop, result writing). All settlement logic lives in `PaperReconciler.reconcile()`.

---

### `tests/test_payout_maps.py` (test, transform) -- NEW

**Analog:** `tests/test_paper_reconciler.py` (existing, lines 1-156)

**Test file pattern** (follow project convention):
```python
"""payout_maps.py のテスト"""

import pandas as pd
import pytest


class TestBuildWinPayoutMap:
    def test_empty_dataframe_returns_empty_dict(self) -> None:
        from betting.payout_maps import build_win_payout_map

        result = build_win_payout_map(pd.DataFrame())
        assert result == {}

    def test_single_payout(self) -> None:
        from betting.payout_maps import build_win_payout_map

        df = pd.DataFrame({
            "race_id": ["2026040510010101"],
            "paytansyoumaban1": [3],
            "paytansyopay1": [250],  # 100円あたり250円 = 2.5倍率
        })
        result = build_win_payout_map(df)
        assert result[("2026040510010101", 3)] == pytest.approx(2.5)

    def test_nan_values_skipped(self) -> None:
        from betting.payout_maps import build_win_payout_map

        df = pd.DataFrame({
            "race_id": ["2026040510010101"],
            "paytansyoumaban1": [pd.NA],
            "paytansyopay1": [pd.NA],
        })
        result = build_win_payout_map(df)
        assert result == {}


class TestBuildPayoutMap:
    def test_empty_dataframe_returns_empty_dict(self) -> None:
        from betting.payout_maps import build_payout_map

        result = build_payout_map(pd.DataFrame())
        assert result == {}

    def test_place_payout_with_multiple_positions(self) -> None:
        from betting.payout_maps import build_payout_map

        df = pd.DataFrame({
            "race_id": ["2026040510010101"],
            "payfukusyoumaban1": [3],
            "payfukusyopay1": [150],
            "payfukusyoumaban2": [5],
            "payfukusyopay2": [120],
            "payfukusyoumaban3": [pd.NA],
            "payfukusyopay3": [pd.NA],
            "payfukusyoumaban4": [pd.NA],
            "payfukusyopay4": [pd.NA],
            "payfukusyoumaban5": [pd.NA],
            "payfukusyopay5": [pd.NA],
        })
        result = build_payout_map(df)
        assert result[("2026040510010101", 3)] == pytest.approx(1.5)
        assert result[("2026040510010101", 5)] == pytest.approx(1.2)
```

---

### `tests/test_paper_reconciler.py` (test, CRUD) -- MODIFIED

**Analog:** `tests/test_paper_reconciler.py` (existing, lines 1-156)

**Key changes needed:**
1. Replace `result` column references with `payout`/`outcome`/`settlement_status` (D-18)
2. Add `schema_version=2` column to test DataFrames (D-19)
3. Add `bet_id` and `session_id` columns to test DataFrames (D-02)
4. Add tests for: schema rejection (old schema), ROI calculation (D-05), retry logic, atomic write

---

### `tests/test_model_loader.py` (test, CRUD) -- MODIFIED

**Analog:** `tests/test_model_loader.py` (existing, lines 1-201)

**Key changes needed:**
1. Add test for D-16: `run_id` + `models_dir` together raises `ValueError`
2. Add test for D-16: neither `run_id` nor `models_dir` raises `ValueError`
3. Add test for D-16: `run_id` only goes to MLflow, never checks local
4. Update existing `test_load_uses_latest_run_when_no_run_id` (this behavior is now REMOVED per D-16)
5. Add test for track_stats restore from JSON artifacts

---

### `tests/test_training_pipeline.py` (test, CRUD) -- MODIFIED

**Analog:** Existing `tests/test_training_pipeline.py`

**Key changes needed:**
1. Add test for `--betting-target win` skips place/wide model training
2. Add test for `--betting-target place` includes win + place models
3. Add test for `--betting-target wide` raises error (D-13)
4. Add test for track_stats JSON persistence after training
5. Add test for meta.json includes `betting_target` field (D-14)

---

## Shared Patterns

### SHA256 Hashing
**Source:** `src/backtest/parameter_freeze_protocol.py` lines 107-128
**Apply to:** bet_id generation (D-02), track_stats checksums (D-15), session_id integrity
```python
import hashlib

def compute_checksum(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()
```

### Atomic Parquet Write
**Source:** `src/db/etl.py` lines 358-366, `src/pipelines/training_pipeline.py` lines 2097-2103
**Apply to:** All bets.parquet writes in PaperReconciler (D-07)
```python
import tempfile
from pathlib import Path

def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        dir=target.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(target)
```

### Fail-Fast on Missing Artifacts
**Source:** `src/validation/oof_health_validator.py` lines 1-50
**Apply to:** ModelLoader (D-17), run_train.py Parquet validation (TRN-02), schema rejection (D-18)
```python
# Pattern: validate upfront, fail with clear message
if not store.exists("raw", "races"):
    logger.error("Parquetデータが見つかりません。先に run_etl.py を実行してください。")
    sys.exit(1)
```

### Parquet as Source of Truth
**Source:** `src/db/parquet_store.py` (project-wide pattern)
**Apply to:** `bets.parquet` as settlement source of truth (D-08), predictions as audit record
```python
# Read from source of truth
if self.bets_path.exists():
    bets_df = pd.read_parquet(self.bets_path)
# Write back atomically
self._atomic_write_parquet(bets_df, self.bets_path)
```

### JSON Manifest with Checksums
**Source:** `src/backtest/parameter_freeze_protocol.py` lines 107-128 (`save_strategy_manifest`)
**Apply to:** meta.json (D-14), track_stats JSON files (D-15)
```python
data = json.dumps(params, sort_keys=True, indent=2)
sha256 = hashlib.sha256(data.encode()).hexdigest()
manifest = {"params": params, "sha256": sha256}
path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

### Test Pattern: Mock-based DB-free Testing
**Source:** `tests/test_paper_reconciler.py` lines 1-156
**Apply to:** All new test files (tests use `unittest.mock`, no DB required)
```python
from unittest.mock import MagicMock

mock_everydb2 = MagicMock()
mock_everydb2.get_race_results.return_value = pd.DataFrame({...})
mock_everydb2.get_payouts.return_value = pd.DataFrame({...})
```

## No Analog Found

All files have close existing analogs in the codebase. No files require patterns from external sources.

| File | Role | Data Flow | Note |
|------|------|-----------|------|
| (none) | -- | -- | All 14 files have existing codebase analogs |

## Metadata

**Analog search scope:** `src/`, `scripts/`, `tests/` (16 packages per CLAUDE.md)
**Files scanned:** 18
**Pattern extraction date:** 2026-06-06
