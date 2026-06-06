# Phase 54: Automation & Reporting - Pattern Map

**Mapped:** 2026-06-06
**Files analyzed:** 9 (3 new, 6 modified)
**Analogs found:** 9 / 9

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `scripts/run_paper_trading.py` | controller | request-response | `scripts/run_paper_trading.py` (self) | exact |
| `src/paper_trading/run_mode_orchestrator.py` | service | event-driven | `src/automation/scheduler.py` | role-match |
| `src/paper_trading/report_aggregator.py` | service | CRUD | `src/paper_trading/reconciler.py` `_compute_roi()` | role-match |
| `src/paper_trading/race_progress.py` | model | file-I/O | `src/features/session_manifest.py` | role-match |
| `src/paper_trading/exit_codes.py` | utility | n/a | `src/domain/types.py` (Enum patterns) | partial |
| `src/paper_trading/report.py` | component | transform | `src/paper_trading/report.py` (self) | exact |
| `src/paper_trading/reconciler.py` | service | CRUD | `src/paper_trading/reconciler.py` (self) | exact |
| `src/paper_trading/watcher.py` | service | event-driven | `src/paper_trading/watcher.py` (self) | exact |
| `src/automation/scheduler.py` | service | event-driven | `src/automation/scheduler.py` (self, read-only ref) | exact |

## Pattern Assignments

### `scripts/run_paper_trading.py` (controller, request-response)

**Analog:** Self (existing file, adding `--mode run` branch)

**Mode branch pattern** (lines 1416-1445):
```python
def main() -> None:
    args = parse_args()
    config = load_config(args)

    # --- モデルロード ---
    models, model_info = _load_models(config, use_ensemble=args.ensemble)

    # --- ParquetStore ---
    from db.parquet_store import ParquetStore
    store = ParquetStore()

    if args.mode == "setup":
        _run_setup(args, config, models, store)
    elif args.mode == "predict":
        _run_predict(args, config, models, store)
    elif args.mode == "reconcile":
        _run_reconcile(args, config, store=store)
    elif args.mode == "dry-run":
        _run_dry_run(args, config, models, store)
    elif args.mode == "diagnose":
        _run_diagnose(args, config, models, store)
    # ADD: elif args.mode == "run": _run_run_mode(args, config, models, store)
```

**parse_args modification** (lines 236-282):
```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Trading")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["setup", "predict", "reconcile", "dry-run", "diagnose"],
        # ADD "run" to choices
        help="実行モード",
    )
```

**Signal handler pattern** (new, no existing analog -- RESEARCH.md Pattern 2):
```python
import signal

def _handle_sigint(signum, frame):
    """Set cancellation flag (D-17: exit code 130)."""
    if hasattr(main, '_orchestrator') and main._orchestrator:
        main._orchestrator._cancelled = True
    else:
        sys.exit(130)
```

**Key integration points:**
- `_build_race_predictor()` (lines 166-233): Shared across all modes. Run mode reuses directly.
- `_load_models()` (lines 334-365): Shared model loading. Run mode reuses directly.
- `load_config()` (lines 315-331): Shared config loading. Run mode reuses directly.
- `_send_slack()` (lines 368-377): Shared notification. Run mode reuses directly.

**Error handling pattern** (lines 285-312, `_validate_betting_target_alignment`):
```python
def _validate_betting_target_alignment(
    models: "TrainedModelsV5",
    manifest_target: str | None,
    cli_target: str,
) -> None:
    # fail-fast on mismatch: logger.error + sys.exit(1)
```

---

### `src/paper_trading/run_mode_orchestrator.py` (service, event-driven) -- NEW

**Analog:** `src/automation/scheduler.py`

**Imports pattern** (scheduler.py lines 1-16):
```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from domain.models import Bet, Race, SafetyCheckResult

if TYPE_CHECKING:
    from betting.orchestrator import DrawdownControllerProtocol

logger = logging.getLogger(__name__)
```

**Protocol-based DI pattern** (scheduler.py lines 23-71):
```python
@runtime_checkable
class OrchestratorProtocol(Protocol):
    def process_race(
        self,
        race: Race,
        feats: dict,
        bankroll: float,
        dd_ctrl: object,
    ) -> list[Bet]: ...
    def finalize_bets(
        self,
        race: Race,
        pending_bets: list[Bet],
        odds_t3_snapshot: dict[int, float],
        odds_t10_snapshot: dict[int, float],
    ) -> list[Bet]: ...

@runtime_checkable
class SafetyGuardProtocol(Protocol):
    def check(self, bankroll: float) -> SafetyCheckResult: ...
```

**Constructor pattern** (scheduler.py lines 84-99):
```python
class RaceScheduler:
    def __init__(
        self,
        orchestrator: OrchestratorProtocol,
        odds_collector: OddsCollectorProtocol,
        pat_voter: PatVoterProtocol,
        safety_guard: SafetyGuardProtocol,
        late_money_filter: LateMoneyFilterProtocol,
        fetcher: FetcherProtocol,
    ) -> None:
        self._orchestrator = orchestrator
        self._odds_collector = odds_collector
        # ... etc
```

**Apply to RunModeOrchestrator:**
- Constructor takes config, models, store, session_manifest, reconciler, race_progress as dependencies
- `execute()` method returns ExitCode
- Internal methods: `_ensure_schedule()`, `_fetch_track_conditions()`, `_predict_races()`, `_reconcile()`, `_aggregate_and_report()`
- Cancellation flag `_cancelled: bool` for SIGINT graceful shutdown

**Sequential per-race loop pattern** (run_paper_trading.py lines 722-853):
```python
for race_id in race_ids:
    if race_id in skipped_race_ids:
        continue
    if race_id in existing_race_ids:
        continue

    single_race = feat_df[feat_df["race_id"] == race_id].copy()
    single_race = _drop_post_race_cols(single_race)

    result_df = race_predictor.predict(single_race, ...)
    if result_df.empty:
        continue

    # ... regime, quality check, bet selection ...

    for bet in bets:
        bet_id = PaperReconciler.compute_bet_id(session_id, race_id, ...)
        all_bets.append({...})
```

**Wait-until pattern** (watcher.py lines 20-27, 91-92):
```python
def wait_until(target_time: datetime) -> None:
    """target_time まで待機 (テスト用にモック可能)"""
    now = datetime.now()
    if target_time > now:
        import time
        time.sleep((target_time - now).total_seconds())
```

---

### `src/paper_trading/report_aggregator.py` (service, CRUD) -- NEW

**Analog:** `src/paper_trading/reconciler.py` `_compute_roi()` (lines 345-398)

**ROI calculation pattern** (reconciler.py lines 345-398):
```python
def _compute_roi(
    self,
    bets_df: pd.DataFrame,
    target_date: date,
    n_settled: int = 0,
    n_wins: int = 0,
    n_refunded: int = 0,
    n_voided: int = 0,
) -> dict[str, Any]:
    """累積統計を計算 (D-05: effective_stake = won + lost only)."""
    if bets_df.empty:
        return self._empty_result(target_date)

    total_bets = len(bets_df)

    # D-05: effective_stake excludes refunded/voided
    decidable = bets_df[bets_df["outcome"].isin(["won", "lost"])]
    effective_stake = float(decidable["stake"].sum()) if not decidable.empty else 0.0
    total_return = float(decidable["payout"].sum()) if not decidable.empty else 0.0
    cumulative_roi = total_return / effective_stake if effective_stake > 0 else 0.0
    net_profit = total_return - effective_stake

    total_wins = int((bets_df["outcome"] == "won").sum())
    n_pending = int((bets_df["settlement_status"] == "pending").sum())
    # ...
```

**Bets.parquet reading pattern** (reconciler.py lines 157-168):
```python
# 1. Load bets.parquet (source of truth, D-08)
if not self.bets_path.exists():
    logger.info("No bets file for %s", target_date)
    return self._empty_result(target_date)

bets_df = pd.read_parquet(self.bets_path)

# Old schema rejection (D-18)
if "result" in bets_df.columns and "payout" not in bets_df.columns:
    raise ValueError("Old schema detected in bets.parquet...")
```

**Date filtering pattern** (reconciler.py lines 175-177):
```python
target_ts = pd.Timestamp(target_date)
date_mask = bets_df["race_date"] == target_ts
pending_mask = date_mask & (bets_df["settlement_status"] == "pending")
```

**ISO week calculation** (new, RESEARCH.md):
```python
from datetime import date, timedelta

def _iso_week_range(iso_year: int, iso_week: int) -> tuple[date, date]:
    jan4 = date(iso_year, 1, 4)
    start_of_week1 = jan4 - timedelta(days=jan4.weekday())
    week_start = start_of_week1 + timedelta(weeks=iso_week - 1)
    week_end = week_start + timedelta(days=6)
    return week_start, week_end
```

---

### `src/paper_trading/race_progress.py` (model, file-I/O) -- NEW

**Analog:** `src/features/session_manifest.py`

**Dataclass pattern** (session_manifest.py lines 113-139):
```python
@dataclass
class SessionManifest:
    """PT 実行記録 (D-09)."""
    session_id: str
    prediction_date: str
    code_version: dict[str, Any] = field(default_factory=dict)
    model_run_id: str = ""
    training_start: str = ""
    training_end: str = ""
    manifest_hash: str = ""
    pfp_result: dict[str, Any] = field(default_factory=dict)
    status: str = "started"
    exit_code: int = 0
    # ... more fields
```

**Atomic JSON write pattern** (session_manifest.py lines 239-268):
```python
def write_session_manifest(manifest: SessionManifest, path: Path) -> None:
    """SessionManifest を JSON ファイルにアトミック書き込み (D-09)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(manifest.to_dict(), indent=2, default=str, ensure_ascii=False)

    fd, tmp_path = tempfile.mkstemp(
        suffix=".json",
        prefix=".session_manifest_",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

**Atomic Parquet write with Windows retry** (reconciler.py lines 73-92):
```python
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
    max_retries = 3
    for attempt in range(max_retries):
        try:
            os.replace(str(tmp_path), str(target))
            return
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1)
```

**Apply to RaceProgress:**
- Use session_manifest.py atomic JSON write for `race_progress.json`
- States as StrEnum: `PENDING`, `PROCESSING`, `PREDICTED`, `NO_BET`, `FAILED`
- `load()` / `mark()` / `pending_or_failed_race_ids()` methods
- Store as `dict[str, dict]` mapping race_id to {state, timestamp, metadata}

---

### `src/paper_trading/exit_codes.py` (utility, n/a) -- NEW

**Analog:** `src/domain/types.py` (Enum patterns)

**Enum pattern** (domain/types.py lines 6-34):
```python
from enum import Enum

class Surface(str, Enum):
    """芝/ダートのサーフェス"""
    TURF = "turf"
    DIRT = "dirt"

class BetType(str, Enum):
    """投票タイプ"""
    WIN = "win"
    PLACE = "place"
    WIDE = "wide"
```

**Apply to ExitCode (use IntEnum for integer exit codes):**
```python
from enum import IntEnum

class ExitCode(IntEnum):
    """Run mode exit codes (D-17)."""
    SUCCESS = 0
    GENERAL_ERROR = 1
    PENDING_REMAIN = 2
    DB_FETCH_ERROR = 3
    DATA_INTEGRITY_ERROR = 4
    MODEL_VALIDATION_ERROR = 5
    REPORT_ERROR = 6
    SIGINT = 130
```

Note: No existing IntEnum in codebase. Project uses `str, Enum` pattern. IntEnum is appropriate here since exit codes are integers used with `sys.exit()`.

---

### `src/paper_trading/report.py` (component, transform) -- MODIFIED

**Analog:** Self (shrinking to pure HTML renderer)

**Current generate() signature** (report.py lines 20-45):
```python
def generate(
    self,
    bets: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    """HTML レポートを生成"""
    enriched = self._derive_fields(bets)        # REMOVES: old `result` column access
    monthly = self._compute_monthly_stats(enriched)  # REMOVES: old aggregation
    bankroll_series = self._compute_bankroll_series(enriched)

    # ... Jinja2 HTML rendering ...
```

**Current `_derive_fields` (references old `result` column)** (report.py lines 47-58):
```python
@staticmethod
def _derive_fields(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not bets:
        return []
    return [
        {
            **b,
            "profit": b["result"] - b["stake"],  # OLD: uses v1 `result` column
            "is_win": b["result"] > 0,
        }
        for b in bets
    ]
```

**Modification plan (D-12):**
- `generate()` accepts Aggregator results instead of raw bets
- Remove `_derive_fields()`, `_compute_monthly_stats()` (Aggregator owns these)
- Keep `_render_html()` but update template to use Aggregator-provided fields
- Replace `b["result"]` references with Aggregator's `outcome` + `payout` fields

**Jinja2 template pattern** (report.py lines 108-156):
```python
env = Environment(loader=BaseLoader(), autoescape=True)
env.filters["pct"] = lambda x: f"{x:.1%}"
env.filters["yen"] = lambda x: f"¥{x:,.0f}"
# ... inline template string ...
template = env.from_string(template_str)
return template.render(bets=bets, monthly=monthly, summary=summary, ...)
```

---

### `src/paper_trading/reconciler.py` (service, CRUD) -- MINOR MODIFICATION

**Analog:** Self (integration point)

**Integration pattern for run mode:**
- `reconcile()` (lines 150-292): Called by RunModeOrchestrator after last race
- `retry_pending()` (lines 294-339): Called by RunModeOrchestrator with last_race_time
- `compute_bet_id()` (lines 61-70): Static helper, already shared
- `_atomic_write_parquet()` (lines 73-92): Static helper, already shared
- `_compute_roi()` (lines 345-398): Reused by Aggregator or Aggregator replaces it

**No structural changes needed** -- RunModeOrchestrator calls existing methods.

---

### `src/paper_trading/watcher.py` (service, event-driven) -- REFERENCE ONLY

**Analog:** Self (patterns reused by RunModeOrchestrator)

**Reusable patterns:**
- `_parse_post_time()` (lines 137-141): Parse HH:MM to datetime
- `wait_until()` (lines 20-27): Sleep until target time
- `_already_processed()` (lines 127-135): Skip already processed races

**Run mode adaptation:**
- RunModeOrchestrator copies/adapts `_parse_post_time()` rather than importing RaceWatcher (which has tight coupling to PaperPredictor)
- Add cancellation check to wait loop (RESEARCH.md Pattern 2)

---

### `src/automation/scheduler.py` (service, event-driven) -- REFERENCE ONLY

**Analog:** Protocol-based DI pattern for RunModeOrchestrator

**Key patterns (read-only reference):**
- Protocol definitions (lines 23-71): `@runtime_checkable` Protocols for DI
- Constructor DI (lines 84-99): Dependencies injected via constructor
- Delegation pattern (lines 114-140): Method delegates to injected Protocol

## Shared Patterns

### Atomic Write (JSON)
**Source:** `src/features/session_manifest.py` lines 239-267
**Apply to:** `race_progress.py`, `report_aggregator.py` (weekly/target JSON output)
```python
def _atomic_write_json(data: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".json", prefix=".tmp_", dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, str(target))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

### Atomic Write (Parquet)
**Source:** `src/paper_trading/reconciler.py` lines 73-92
**Apply to:** `run_mode_orchestrator.py` (input snapshot saves)
```python
@staticmethod
def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", dir=str(target.parent), delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path, index=False)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            os.replace(str(tmp_path), str(target))
            return
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1)
```

### Fail-Fast Validation
**Source:** `src/paper_trading/reconciler.py` `_validate_bet_schema()` lines 95-144
**Apply to:** RunModeOrchestrator startup validation, Aggregator input validation
```python
@staticmethod
def _validate_bet_schema(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if "result" in df.columns and "payout" not in df.columns:
        errors.append("Old schema detected: 'result' column present without 'payout'")
        return errors
    for col in ("schema_version", "settlement_status", "outcome", "payout", "bet_id", "stake"):
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    # ... more validation
    return errors
```

### Bet ID Generation
**Source:** `src/paper_trading/reconciler.py` lines 61-70
**Apply to:** RunModeOrchestrator bet creation
```python
@staticmethod
def compute_bet_id(
    session_id: str, race_id: str, bet_type: str, umaban: int, umaban_b: int | None = None,
) -> str:
    raw = f"{session_id}|{race_id}|{bet_type}|{umaban}"
    if umaban_b is not None:
        raw += f"|{umaban_b}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
```

### Session Manifest Integration
**Source:** `src/features/session_manifest.py` lines 113-231
**Apply to:** RunModeOrchestrator session lifecycle management
```python
@dataclass
class SessionManifest:
    session_id: str
    prediction_date: str
    # ... set_code_version(), set_model_identity(), set_status(), to_dict()

def write_session_manifest(manifest: SessionManifest, path: Path) -> None:
    # atomic write pattern
```

**Extension for run mode (D-18):** Add `errors: list[dict]` field to SessionManifest for multi-error tracking. Manifest `to_dict()` already serializes all fields.

### Composition Root Pattern
**Source:** `scripts/run_paper_trading.py` main() lines 1416-1445
**Apply to:** `_run_run_mode()` function (new branch in main)
```python
def main() -> None:
    args = parse_args()
    config = load_config(args)
    models, model_info = _load_models(config, use_ensemble=args.ensemble)
    store = ParquetStore()

    # Each mode gets its own branch
    if args.mode == "run":
        _run_run_mode(args, config, models, store)
```

## No Analog Found

Files with no close match in the codebase (planner should use RESEARCH.md patterns instead):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | -- | -- | All files have sufficient analogs in existing codebase |

## Metadata

**Analog search scope:** `scripts/`, `src/paper_trading/`, `src/automation/`, `src/features/session_manifest.py`, `src/domain/types.py`, `src/betting/payout_maps.py`, `tests/`
**Files scanned:** 15
**Pattern extraction date:** 2026-06-06
