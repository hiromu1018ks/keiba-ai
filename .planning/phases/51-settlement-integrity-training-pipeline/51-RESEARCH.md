# Phase 51: Settlement Integrity & Training Pipeline - Research

**Researched:** 2026-06-06
**Domain:** Paper Trading Settlement + Training Pipeline Modifications
**Confidence:** HIGH

## Summary

Phase 51 addresses two critical gaps in the keiba-ai paper trading system: (1) settlement integrity -- ensuring all bets (wins AND losses) are correctly settled with a proper state model, and (2) training pipeline fixes -- enabling `--betting-target` scoped training with Parquet validation, track_stats persistence, and ModelLoader priority correction.

The current `PaperReconciler` only tracks `result` (float, 0.0=unsettled, >0=payout) and never records losses. The `_run_reconcile` function in `run_paper_trading.py` duplicates reconciliation logic inline rather than delegating to `PaperReconciler`. The current reconciliation treats all zero-result bets as "unsettled" with no way to distinguish a loss from a pending bet. Additionally, `track_stats` and `track_month_stats` are computed during training and attached to `SubmodelSet` objects in memory, but **never persisted to disk or MLflow**, meaning at inference time (PT) they are always `None`, causing NaN values in season deviation and zscore features. The `ModelLoader.load()` method silently falls back to `data/models/` local directory even when a `run_id` is specified, which is dangerous for PT reproducibility.

**Primary recommendation:** Implement the 3-column state model (settlement_status/outcome/payout) in PaperReconciler, extract payout map functions to a new `payout_maps.py` module, add track_stats JSON persistence to the training pipeline, and fix ModelLoader to enforce MLflow-only loading when `--run-id` is specified.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** PaperReconciler クラスを精算の唯一実装とする。Phase 51で二重実装を解消し、`_run_reconcile` インラインは薄いCLIラッパー(引数構築・結果表示・終了コード制御)に縮小する。Phase 52に先送りしない。
- **D-02:** `bet_id = SHA256(session_id | race_id | bet_type | canonical_selection)[:32]`。session_id は当日run開始時に生成・永続化し、クラッシュ復旧時も再利用する。canonical_selection は馬番。時刻・stakeは含めない。
- **D-03:** 状態モデルは3列で管理: `settlement_status` (pending/settled), `outcome` (NULL/won/lost/refunded/voided), `payout` (NULL/float)
- **D-04:** `voided` = レース不成立, `refunded` = 出走取消/競走除外. 両方とも payout=stake, effective_stake/ROI分母から除外. 同着は `won` として実払戻額を記録.
- **D-05:** ROI集計公式: effective_stake = sum(stake WHERE outcome IN (won, lost)), return = sum(payout WHERE outcome IN (won, lost)), ROI = return / effective_stake, net_profit = return - effective_stake
- **D-06:** リトライ戦略: per-race 1回取得(未到着ならpending維持) + 最終レース後60s間隔で最大10分間(絶対期限)全pending一括再取得。DB接続エラーと「接続成功・払戻未掲載」を区別して記録。期限後もpendingが残れば保存して終了コード2。再実行はpendingのみ処理。
- **D-07:** 部分精算の保存は一時ファイル経由のatomic replace。
- **D-08:** 累積 `bets.parquet` を精算状態の正本(source of truth)とする。`predictions/` は予測時点の監査記録とする。
- **D-09:** `src/betting/payout_maps.py` に `build_win_payout_map` / `build_place_payout_map` を純粋関数として抽出。BT engine と PaperReconciler の両方が同一関数を使用。
- **D-10:** 払戻マップの出力は「100円あたりの円」ではなく倍率に統一。入力の文字列・数値・欠損表現を正規化。同着による複数単勝払戻に対応。
- **D-11:** 精算判定順序: (1) 返還/不成立 → refunded/voided, (2) 払戻データにレース存在 → 精算可能, (3) 対象馬が払戻マップに存在 → won, (4) 存在しない → lost, (5) 同一race_id払戻行複数 → 安全統合, (6) 不正な払戻値 → pending維持
- **D-12:** ヘルパーにEveryDB2アクセスやファイルI/Oを含めない。
- **D-13:** `--betting-target` 別の学習スコープ: win=共通+Win固有, place=共通+Win基盤+Place固有, wide=v2.4では拒否(エラー終了)
- **D-14:** 学習targetをMLflow + `meta.json` に保存。PT起動時にモデルtargetと`--betting-target`の一致を必須検証。
- **D-15:** track_stats / track_month_stats はローカル + MLflow artifacts の両方に保存。`track_stats_{surface}.json` / `track_month_stats_{surface}.json` をモデル成果物の必須ファイルとする。SHA256を meta.json + MLflow params/tags に記録。
- **D-16:** ModelLoader優先度: PTでは `--run-id` 必須。MLflowからのみロード、暗黙ローカルフォールバックなし。ローカル利用は `--models-dir` 明示指定時のみ。`--run-id` と `--models-dir` の同時指定は禁止。「最新run自動選択」も禁止。ロード元を実行記録・レポートに保存。
- **D-17:** 必須成果物欠落時は fail-fast (非ゼロ終了)。
- **D-18:** `result` 列を廃止 → `payout` に完全置換。旧スキーマ(`result`列のみ)の成果物は自動変換せず明示的に拒否。
- **D-19:** `schema_version=2` 列を追加。
- **D-20:** 書き込み前整合性検証: pending: outcome=NULL, payout=NULL; settled: outcome!=NULL, payout>=0; lost: payout=0; won: payout>0; refunded/voided: payout=stake; bet_id: 非NULLかつ一意; stake>0; schema_version=2固定

### Claude's Discretion
- payout_maps.py の内部実装詳細(正規化ロジック、統合方法)
- PaperReconciler の内部リトライループ実装
- Pre-training Parquet検証の具体的なチェック内容(NaN率閾値等)
- Feature cache dependency tracking のキャッシュキー計算方式
- atomic replace の一時ファイル命名規則

### Deferred Ideas (OUT OF SCOPE)
- Wide bet settlement — v2.5+ (WID-01, WID-02)
- SafetyGuard integration — v2.5+ (SAF-01, SAF-02)
- Shared feature builder extraction — Phase 52 (PLN-01)
- Pipeline identity recording — Phase 52 (PLN-02, PLN-03, PLN-04)
- Strategy manifest integration — Phase 53 (STR-01~06)
- Live data fetcher — Phase 53 (LIV-01~03)
- One-command run mode — Phase 54 (AUT-01~03)
- Weekly/cumulative reporting — Phase 54 (RPT-01~04)
- Conservative MAWC redesign — v2.5+
- WinSegmentCalibrator dead code removal — v2.5+ (WRN-01)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STL-01 | Win/Place bet status tracking -- `bets.parquet` に `settlement_status` 列 (pending/settled) と `outcome` 列 (won/lost) を分離して追加 | D-03 3列状態モデル。現行 `result` 列は float 1列で負け/未確定を区別不可。schema_version=2 で新スキーマ識別 |
| STL-02 | Win actual payout settlement -- `build_win_payout_map()` パターンを再利用 | `engine.py:211-229` に `build_win_payout_map()` 既存。純粋関数として payout_maps.py に抽出 |
| STL-03 | Place actual payout settlement -- 複勝払戻を精算し負けも記録 | `engine.py:163-208` に `build_payout_map()` 既存 (複勝用)。重要: CONTEXT.md の "build_place_payout_map" は実際には `build_payout_map` という名前 |
| STL-04 | ROI calculation fix -- 的中のみでなく負け含む全ベットで正確なROI | D-05 ROI公式。現行 `_compute_summary` は `total_return = bets_df["result"].sum()` で負けを0と扱い、`total_stake` も全bet分を分母に使わず |
| STL-05 | Payout retry fetch -- DB遅延時に払戻データをリトライ取得 | D-06 リトライ戦略。現行は `get_payouts()` 1回限りで空ならreturn |
| TRN-01 | run_train.py `--betting-target` support | 現行 run_train.py には `--betting-target` 引数なし。TrainingPipelineV5.run() も全モデル常に学習。D-13 でスコープ定義済み |
| TRN-02 | Pre-training Parquet validation | 現行 run_train.py は `store.exists("raw", "races")` のみチェック。日付範囲・NaN率・更新日時の検証なし |
| TRN-03 | Feature cache dependency tracking | 現行 `compute_cache_key()` は `source_paths` に races/entries/snapshots のみ。track_conditions/horse_track_aptitude が未含 | 
| TRN-04 | track_stats persistence | **重要バグ**: `_compute_track_stats()`/`_compute_track_month_stats()` は学習時に計算して `SubmodelSet` に設定するが、`_save_to_local_dir()` も MLflow log_artifact でも保存していない。ModelLoader でも復元しない |
| TRN-05 | ModelLoader priority fix | 現行 `ModelLoader.load()` は L54 で常に `data/models/` を先にチェック。run_id 指定時もローカルが優先される (D-16 違反) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Bet settlement state management | Backend (PaperReconciler) | Storage (Parquet) | Settlement logic is pure business logic; Parquet is persistence |
| Payout map construction | Backend (payout_maps.py) | -- | Pure function, no I/O, shared by BT engine and PT reconciler |
| Win/Place payout settlement | Backend (PaperReconciler) | Database (EveryDB2) | Reconciler drives settlement; EveryDB2 provides raw payout data |
| ROI calculation | Backend (PaperReconciler) | -- | Aggregation from settled bet records |
| Retry orchestration | Backend (PaperReconciler) | Database (EveryDB2) | Reconciler manages retry loop; EveryDB2 is data source |
| Training scope control | CLI (run_train.py) | Pipeline (TrainingPipelineV5) | CLI parses args; pipeline executes scoped training |
| Parquet pre-validation | Pipeline (TrainingPipelineV5) | Storage (ParquetStore) | Pipeline orchestrates checks; ParquetStore provides data access |
| Feature cache dependencies | Feature Engine | Storage (Parquet) | FeatureEngine computes cache keys; Parquet files are dependency sources |
| track_stats persistence | Pipeline (TrainingPipelineV5) | MLflow + Local FS | Pipeline saves; both MLflow artifacts and local JSON files |
| ModelLoader priority | ModelLoader | MLflow | ModelLoader enforces loading policy; MLflow is primary source |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | 2.3.3 | DataFrame operations, Parquet I/O | Existing project standard |
| pyarrow | 23.0.1 | Parquet read/write, dataset API | Existing project standard |
| pytest | 9.0.2 | Testing framework (2503 tests) | Existing project standard |
| mlflow | 3.10.1 | Experiment tracking, artifact storage | Existing project standard |
| lightgbm | 4.6.0 | ML model format | Existing project standard |
| joblib | installed | Model serialization (.joblib) | Existing project standard |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib (stdlib) | 3.11 | SHA256 for bet_id, cache keys, manifest checksums | bet_id generation, track_stats SHA256 |
| json (stdlib) | 3.11 | track_stats JSON serialization, meta.json | track_stats persistence |
| tempfile (stdlib) | 3.11 | Atomic replace via temp file | Partial settlement saves (D-07) |
| hashlib (stdlib) | 3.11 | SHA256 hashing | bet_id, cache keys, artifact checksums |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual state transitions | State machine library | Overkill for 2 states (pending/settled) + 4 outcomes |
| Custom retry logic | tenacity library | tenacity adds dependency for simple 60s polling; stdlib time.sleep sufficient |
| Parquet schema migration | Automatic converter | D-18 explicitly rejects auto-conversion; explicit reject is safer |

**Installation:**
No new packages needed for this phase. All dependencies are already installed.

```bash
# Verify existing packages (already installed)
python -c "import pandas; print(pandas.__version__)"   # 2.3.3
python -c "import mlflow; print(mlflow.__version__)"    # 3.10.1
python -c "import pytest; print(pytest.__version__)"    # 9.0.2
python -c "import pyarrow; print(pyarrow.__version__)"  # 23.0.1
```

## Package Legitimacy Audit

No new packages installed in this phase. All work uses existing project dependencies.

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────────────────┐
                    │                 run_paper_trading.py                  │
                    │  (CLI layer -- thin wrappers only per D-01)          │
                    └───────┬──────────────────────────┬──────────────────┘
                            │ predict mode             │ reconcile mode
                            ▼                          ▼
                  ┌─────────────────┐      ┌────────────────────────────┐
                  │  _run_predict() │      │  _run_reconcile() wrapper  │
                  │  - Generate bets│      │  - Build args              │
                  │  - Write pred/  │      │  - Call PaperReconciler    │
                  │  - Append bets/ │      │  - Display results         │
                  └────────┬────────┘      │  - Exit code control       │
                           │               └───────────┬────────────────┘
                           ▼                           │
                  ┌──────────────────┐                 ▼
                  │ predictions/{date│    ┌───────────────────────────────┐
                  │   }.parquet      │    │     PaperReconciler           │
                  │  (audit record)  │    │  ┌─────────────────────────┐  │
                  └──────────────────┘    │  │ 1. Load pending bets    │  │
                                          │  │ 2. Fetch payouts (DB)   │  │
                  ┌──────────────────┐    │  │ 3. build_win_payout_map │  │
                  │  bets.parquet    │◄───│  │ 4. build_payout_map     │  │
                  │  (source of      │    │  │ 5. Settlement logic     │  │
                  │   truth, v2)     │    │  │ 6. Atomic write (temp)  │  │
                  │                  │    │  │ 7. Retry pending         │  │
                  │  Columns:        │    │  └─────────────────────────┘  │
                  │  - bet_id        │    └───────────────────────────────┘
                  │  - session_id    │                    │
                  │  - settlement_   │                    │ payout_maps.py
                  │    status        │                    │ (pure functions)
                  │  - outcome       │                    ▼
                  │  - payout        │          ┌───────────────────┐
                  │  - schema_ver=2  │          │ build_win_payout_ │
                  └──────────────────┘          │ map()             │
                                                │ build_payout_map()│
                                                └───────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │                     Training Pipeline                                │
    │                                                                      │
    │  run_train.py --betting-target win|place                             │
    │       │                                                              │
    │       ▼                                                              │
    │  ┌─────────────────────────────────────┐                             │
    │  │ 1. Parquet Validation               │                             │
    │  │    - Date range check               │                             │
    │  │    - NaN rate check                 │                             │
    │  │    - Freshness check                │                             │
    │  │    - track_conditions.parquet       │                             │
    │  │    - horse_track_aptitude.parquet   │                             │
    │  └─────────────────┬───────────────────┘                             │
    │                    ▼                                                 │
    │  ┌─────────────────────────────────────┐                             │
    │  │ 2. Feature Cache Check              │                             │
    │  │    - compute_cache_key() enhanced   │                             │
    │  │    - track_conditions in deps       │                             │
    │  │    - horse_track_aptitude in deps   │                             │
    │  └─────────────────┬───────────────────┘                             │
    │                    ▼                                                 │
    │  ┌─────────────────────────────────────┐                             │
    │  │ 3. TrainingPipelineV5.run()         │                             │
    │  │    - Scoped by --betting-target     │                             │
    │  │    - win: common + win models       │                             │
    │  │    - place: common + win + place    │                             │
    │  │    - wide: REJECTED (exit 1)        │                             │
    │  └─────────────────┬───────────────────┘                             │
    │                    ▼                                                 │
    │  ┌─────────────────────────────────────┐                             │
    │  │ 4. Save Artifacts                   │                             │
    │  │    - track_stats_{surface}.json     │──► data/models/ + MLflow    │
    │  │    - track_month_stats_{surface}.json│                             │
    │  │    - meta.json with betting_target  │                             │
    │  │    - SHA256 checksums               │                             │
    │  └─────────────────────────────────────┘                             │
    └──────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │                        ModelLoader                                   │
    │                                                                      │
    │  OLD (broken):   data/models/ always first → MLflow fallback        │
    │  NEW (D-16):     --run-id → MLflow ONLY (no local fallback)         │
    │                  --models-dir → local ONLY (no MLflow)               │
    │                  neither → ERROR (no implicit selection)             │
    │                  both → ERROR (mutually exclusive)                   │
    └──────────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure
```
src/
├── betting/
│   └── payout_maps.py          # NEW: Pure functions extracted from engine.py
├── paper_trading/
│   └── reconciler.py           # MODIFIED: 3-column state model, retry, Win/Place
├── backtest/
│   └── engine.py               # MODIFIED: Import from payout_maps.py instead of local
├── pipelines/
│   └── training_pipeline.py    # MODIFIED: track_stats save, betting-target scope
├── db/
│   └── model_loader.py         # MODIFIED: Priority fix (D-16), track_stats restore
├── domain/
│   └── models.py               # UNCHANGED: SubmodelSet already has track_stats fields
└── features/
    └── feature_engine.py       # MODIFIED: Cache key includes track_conditions deps

scripts/
├── run_train.py                # MODIFIED: --betting-target, Parquet validation
└── run_paper_trading.py        # MODIFIED: _run_reconcile thinned to wrapper

tests/
├── test_payout_maps.py         # NEW: Pure function tests for payout maps
├── test_paper_reconciler.py    # MODIFIED: Update for new schema, state model
├── test_model_loader.py        # MODIFIED: Priority fix tests
└── test_training_pipeline.py   # MODIFIED: betting-target scope tests
```

### Pattern 1: Pure Function Extraction (payout_maps.py)
**What:** Extract `build_win_payout_map()` and `build_payout_map()` from `engine.py` into `src/betting/payout_maps.py` as pure functions with no I/O dependencies.
**When to use:** Both BacktestEngine and PaperReconciler need identical payout map construction.
**Important naming note:** CONTEXT.md refers to `build_place_payout_map` but the actual function name in `engine.py:163` is `build_payout_map` (it builds the place/fuku payout map). The extracted module should preserve the existing function names to minimize downstream changes.
**Example:**
```python
# src/betting/payout_maps.py
"""Payout map construction -- pure functions, no I/O.

Extracted from backtest/engine.py for shared use by BT and PT.
"""

import pandas as pd


def build_win_payout_map(payouts_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    """(race_id, umaban) -> odds_multiplier for single-win payouts."""
    # ... existing implementation from engine.py:211-229 ...


def build_payout_map(payouts_df: pd.DataFrame) -> dict[tuple[str, int], float]:
    """(race_id, umaban) -> odds_multiplier for place (fuku) payouts."""
    # ... existing implementation from engine.py:163-208 ...


def build_wide_payout_map(payouts_df: pd.DataFrame) -> dict[tuple[str, int, int], float]:
    """(race_id, umaban_lo, umaban_hi) -> odds_multiplier for wide payouts."""
    # ... existing implementation from engine.py:232+ ...
```

### Pattern 2: 3-Column State Model
**What:** Replace the single `result` float column with `settlement_status`, `outcome`, `payout` columns.
**When to use:** All bet records in `bets.parquet`.
**Example:**
```python
# Bet record schema v2
{
    "bet_id": "a1b2c3d4...",       # SHA256(session_id | race_id | bet_type | umaban)[:32]
    "session_id": "...",            # Persistent session identifier
    "schema_version": 2,            # D-19
    "race_id": "2026040510010101",
    "umaban": 3,
    "bet_type": "win",
    "stake": 100.0,
    "settlement_status": "pending", # "pending" | "settled"
    "outcome": None,                # NULL | "won" | "lost" | "refunded" | "voided"
    "payout": None,                 # NULL | float (0.0=loss, >0=win, =stake=refunded/voided)
    # ... other columns ...
}

# Validation (D-20):
# pending: outcome=NULL, payout=NULL
# settled: outcome in {won, lost, refunded, voided}, payout>=0
# won: payout>0
# lost: payout==0.0
# refunded/voided: payout==stake
```

### Pattern 3: Atomic Parquet Replace
**What:** Write partial settlement results via temp file, then atomic rename.
**When to use:** Every time PaperReconciler saves bets.parquet with partial updates.
**Example:**
```python
import tempfile
from pathlib import Path

def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    """Atomic replace via temp file (D-07)."""
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

### Pattern 4: Training Scope Control (--betting-target)
**What:** `--betting-target` controls which models are trained and saved.
**When to use:** run_train.py invocation.
**Example:**
```python
# run_train.py argument
parser.add_argument(
    "--betting-target",
    choices=["win", "place", "wide"],
    help="Betting target scope for training",
)

# In TrainingPipelineV5.run():
if betting_target == "wide":
    parser.error("v2.4 does not support --betting-target wide")

# win: stage1, market, regime, quality, win models
# place: all win models + place models (win is prerequisite)
```

### Anti-Patterns to Avoid
- **Duplicate reconciliation logic:** `_run_reconcile` must NOT contain settlement logic. It must only build args, call `PaperReconciler`, display results, and control exit codes. This is the current bug (D-01).
- **Using `result` column after Phase 51:** All new code uses `payout`/`outcome`/`settlement_status`. Old `result` column is dead. Writing `result` in new code is a bug.
- **Silent local fallback in ModelLoader:** When `--run-id` is specified, falling back to `data/models/` silently loads stale/wrong models. This must raise an error.
- **Treating all zero-payout bets as losses:** Some payout values may be invalid (NULL, negative, zero in raw data). Per D-11 item 6, invalid payout values should keep the bet as `pending`, not mark it as `lost`.
- **Auto-migrating old schema:** D-18 explicitly rejects auto-conversion of `result`-only parquets. The code must detect and reject them with a clear error message.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Payout map construction | Duplicate in PaperReconciler | Extracted functions from `engine.py` -> `payout_maps.py` | D-09 requires shared functions |
| Atomic file replace | Custom locking mechanism | `tempfile.NamedTemporaryFile` + `Path.replace()` | OS-level atomic rename, same pattern used in PFP |
| SHA256 hashing | Custom hash implementation | `hashlib.sha256` (stdlib) | Already used throughout project for manifests, bet_id, cache keys |
| Session persistence | Complex database | Simple JSON file | session_id + metadata in a single file, loaded on startup |
| Feature cache invalidation | Manual timestamp checks everywhere | Enhanced `compute_cache_key()` with additional source paths | Already has proper SHA256-based key computation |

**Key insight:** The codebase already has all the building blocks (SHA256 hashing in PFP, atomic write pattern, cache key computation, pure function pattern in engine.py). Phase 51 is primarily about reorganizing existing patterns, not creating new ones.

## Common Pitfalls

### Pitfall 1: CONTEXT.md names `build_place_payout_map` but actual function is `build_payout_map`
**What goes wrong:** Implementer creates a new function `build_place_payout_map` in `payout_maps.py` instead of extracting the existing `build_payout_map` from `engine.py:163`.
**Why it happens:** CONTEXT.md canonical_refs (line 103) says `build_place_payout_map` but this function does not exist. The place/fuku payout builder is called `build_payout_map` in engine.py.
**How to avoid:** Extract the function as `build_payout_map` (its real name) and optionally add an alias `build_place_payout_map` for clarity. Update engine.py imports to use the new module.
**Warning signs:** If `payout_maps.py` has a function `build_place_payout_map` but `engine.py` still has a local `build_payout_map`, the migration was incomplete.

### Pitfall 2: track_stats never persisted -- models work in training but NaN at inference
**What goes wrong:** `SubmodelSet` has `track_stats` and `track_month_stats` fields that get populated during training, but `_save_to_local_dir()` and MLflow artifact logging never save these as JSON files. At inference time (PT), `ModelLoader` creates `SubmodelSet` with `track_stats=None`, causing `compute_track_condition_features()` to compute NaN for `turf_cushion_track_relative`, `turf_cushion_track_zscore`, `cushion_season_deviation`, `moisture_season_deviation`.
**Why it happens:** The training pipeline was extended (Phase 48/49) to compute track stats and wire them into SubmodelSet, but the save/load code was never updated.
**How to avoid:** After `_train_submodel()` completes, save `track_stats_{surface}.json` and `track_month_stats_{surface}.json` to both `data/models/` and MLflow artifacts. In `ModelLoader.load()` and `load_from_dir()`, restore these into `SubmodelSet`.
**Warning signs:** `turf_cushion_track_relative` is NaN in PT predictions; `track_stats_{surface}.json` files don't exist in `data/models/`.

### Pitfall 3: ModelLoader silently loads stale local models in PT
**What goes wrong:** `ModelLoader.load()` line 54 checks `data/models/meta.json` first and returns immediately if found, ignoring the requested `run_id`. This means PT with `--run-id <specific>` may use a completely different model set.
**Why it happens:** The load() method was designed for backtest convenience (local-first), not for production PT where exact run_id reproducibility matters.
**How to avoid:** Implement D-16 priority: `--run-id` means MLflow-only, `--models-dir` means local-only, neither means error, both means error. Remove `_find_latest_run()` auto-selection.
**Warning signs:** PT reports don't show the expected MLflow run_id; models produce different predictions than expected.

### Pitfall 4: _run_reconcile and PaperReconciler duplicate logic
**What goes wrong:** After Phase 51, `_run_reconcile` still contains inline settlement logic (payout_map construction, iterrows loop, result writing) instead of delegating to PaperReconciler.
**Why it happens:** D-01 says to thin `_run_reconcile` to a wrapper, but implementer may "keep the old code working" and only add the new path, leaving two code paths.
**How to avoid:** Delete ALL settlement logic from `_run_reconcile`. It should only: (1) parse args, (2) instantiate PaperReconciler, (3) call `reconcile()`, (4) format and display results, (5) handle exit codes.
**Warning signs:** `_run_reconcile` is longer than ~50 lines; it contains `for _, row in unsettled.iterrows()`; it writes to parquet directly.

### Pitfall 5: ROI calculation with old schema mixed with new schema
**What goes wrong:** After introducing schema_version=2, old bets.parquet with `result` column still exists. Loading and combining old+new schema data causes KeyError or incorrect aggregation.
**Why it happens:** D-18 says to reject old schema, but if the rejection is a warning instead of an error, mixed data can persist.
**How to avoid:** On load, check for `schema_version` column. If missing (old schema), raise an error with a clear message. Do NOT auto-convert.
**Warning signs:** bets.parquet has both `result` and `payout` columns; schema_version column is missing on some rows.

### Pitfall 6: Feature cache key doesn't include track_conditions.parquet
**What goes wrong:** `compute_cache_key()` in `feature_engine.py:54-82` uses `source_paths` that only includes races, entries, snapshots. After track_conditions.parquet is updated (e.g., new ETL run), the feature cache is still considered valid because its mtime wasn't checked.
**Why it happens:** The cache key was designed before track_conditions was added as a merge source (Phase 48).
**How to avoid:** Add `("raw", "track_conditions")` and `("raw", "horse_track_aptitude")` to the `source_paths` list in `build_all()`.
**Warning signs:** After ETL update, `turf_cushion` values in features don't change even though the raw parquet was updated.

## Code Examples

### build_win_payout_map extraction (from engine.py:211-229)
```python
# Source: engine.py:211-229 (verified by codebase read)
def build_win_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) -> odds_multiplier のマップを構築 (単勝用)。

    paytansyopay1 は「100円あたりの円」なので、100で割って倍率に変換する。
    ベクトル化: dropna -> astype -> dict comprehension。
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

### build_payout_map extraction (from engine.py:163-208) -- the "place" payout builder
```python
# Source: engine.py:163-208 (verified by codebase read)
# NOTE: This is the function CONTEXT.md calls "build_place_payout_map"
# but its actual name is build_payout_map
def build_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) -> odds_multiplier のマップを構築。

    payfukusyopay は「100円あたりの円」なので、100で割って倍率に変換する。
    ベクトル化: melt + groupby で一括処理。同一 (race_id, umaban) の最大値を保持。
    """
    if payouts_df.empty:
        return {}
    # ... melt + groupby implementation ...
```

### track_stats computation (from training_pipeline.py:970-990)
```python
# Source: training_pipeline.py:970-990 (verified by codebase read)
from features.track_condition_features import (
    _compute_track_month_stats,
    _compute_track_stats,
)

_track_stats: dict | None = None
_track_month_stats: dict | None = None
with TimingContext(f"{surface}/track_condition"):
    if "turf_cushion" in df.columns and "trackcd" in df.columns:
        _track_stats = _compute_track_stats(df)
    if "trackcd" in df.columns and (
        "turf_cushion" in df.columns or "dirt_moisture" in df.columns
    ):
        _track_month_stats = _compute_track_month_stats(df)

# Then wired into SubmodelSet at line 1568-1569:
# track_stats=_track_stats,
# track_month_stats=_track_month_stats,
```

### ModelLoader current priority bug (from model_loader.py:46-55)
```python
# Source: model_loader.py:46-55 (verified by codebase read)
def load(
    self, run_id: str | None = None, *, use_ensemble: bool | None = None
) -> tuple[TrainedModelsV5, ModelInfo]:
    # 1. ローカルディレクトリから読み込み -- BUG: Always checks local first!
    models_dir = Path("data/models")
    if models_dir.is_dir() and (models_dir / "meta.json").is_file():
        return self.load_from_dir(models_dir, use_ensemble_override=use_ensemble)
    # 2. MLflow 経由 (フォールバック)
    if run_id is None:
        run_id = self._find_latest_run()  # BUG: Auto-selects latest run
```

### Feature cache key computation (from feature_engine.py:54-82)
```python
# Source: feature_engine.py:54-82 (verified by codebase read)
def compute_cache_key(
    input_paths: list[Path],
    date_range: tuple[str, str] | None,
    feature_type: str,
    *,
    code_hash: str | None = None,
) -> str:
    payload = json.dumps(
        {
            "paths": [str(p) for p in sorted(input_paths)],
            "start": date_range[0] if date_range else "",
            "end": date_range[1] if date_range else "",
            "type": feature_type,
            "code_hash": code_hash or "",
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

### Feature cache source_paths construction (from feature_engine.py:226-238)
```python
# Source: feature_engine.py:226-238 (verified by codebase read)
# CURRENT: Only includes races, entries, snapshots
for cat, name in [
    ("raw", "races"),
    ("raw", "entries"),
    ("odds", "snapshots"),
]:
    p = data_dir / cat / name
    if p.with_suffix(".parquet").exists():
        source_paths.append(p.with_suffix(".parquet"))
    elif p.is_dir():
        source_paths.append(p)

# TRN-03 FIX: Also include track_conditions and horse_track_aptitude
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single `result` float column | 3-column state model (settlement_status/outcome/payout) | Phase 51 (this phase) | Fixes ROI overestimation, enables loss tracking |
| Inline `_run_reconcile` logic | PaperReconciler as single source of truth | Phase 51 (this phase) | Eliminates code duplication, consistent settlement |
| Local-first ModelLoader | Explicit source selection (--run-id or --models-dir) | Phase 51 (this phase) | Prevents stale model loading in PT |
| track_stats in-memory only | JSON persistence to disk + MLflow | Phase 51 (this phase) | Fixes NaN in PT season deviation features |
| Full model training always | --betting-target scoped training | Phase 51 (this phase) | Faster training, correct model packaging |

**Deprecated/outdated:**
- `result` column: Replaced by `payout` + `outcome` + `settlement_status` (D-18)
- `_find_latest_run()` auto-selection: Prohibited by D-16
- `data/models/` implicit fallback: Prohibited by D-16

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `horse_track_aptitude.parquet` does not exist in `data/raw/` (only `track_conditions.parquet` found) | TRN-02/TRN-03 | If it exists under different path, cache deps need different config |
| A2 | `_compute_track_stats()` and `_compute_track_month_stats()` return JSON-serializable dicts | TRN-04 | If they return complex objects, JSON save will fail |
| A3 | `build_wide_payout_map` should also be extracted to payout_maps.py for completeness, but wide settlement is deferred | Architecture | Low risk; wide is out of scope per D-13 |
| A4 | Existing `test_paper_reconciler.py` tests use `result` column and will need updating to new schema | Testing | Tests will break and need migration |
| A5 | `_run_predict` in run_paper_trading.py will need modification to add `bet_id`, `session_id`, and new columns to bet records | STL-01 | Scope may be larger than expected |

## Open Questions

1. **`horse_track_aptitude.parquet` file location**
   - What we know: `track_conditions.parquet` exists at `data/raw/track_conditions.parquet` (193KB, updated 2026-06-04). `DataRepository.load_horse_track_aptitude()` checks for `data/raw/horse_track_aptitude`.
   - What's unclear: Whether `horse_track_aptitude.parquet` has been generated yet by ETL or if it's expected to be created in a future phase.
   - Recommendation: The TRN-02/TRN-03 changes should handle the case where this file doesn't exist gracefully (skip check, log warning).

2. **Place training dependency on Win models**
   - What we know: D-13 says place training includes "Win基盤モデル" (Win foundation models). The current training pipeline trains all models sequentially in `_train_submodel()`.
   - What's unclear: Whether place-only training should re-train Win models from scratch or load pre-trained Win models.
   - Recommendation: Re-train Win models as part of place training (same pipeline run). This is simpler and ensures consistency.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | All | Yes | 3.11.15 | -- |
| pandas | Data processing | Yes | 2.3.3 | -- |
| pyarrow | Parquet I/O | Yes | 23.0.1 | -- |
| pytest | Testing | Yes | 9.0.2 | -- |
| lightgbm | Model loading | Yes | 4.6.0 | -- |
| mlflow | Model tracking | Yes | 3.10.1 | -- |
| PostgreSQL (EveryDB2) | Reconcile (get_payouts) | Not checked | -- | Tests use mocks; no DB needed for tests |
| data/raw/track_conditions.parquet | TRN-02/TRN-03 | Yes | 193KB, 2026-06-04 | -- |
| data/raw/horse_track_aptitude.parquet | TRN-02/TRN-03 | No | -- | Skip validation, log warning |

**Missing dependencies with no fallback:**
- None (all test dependencies available; DB-dependent features use mocks)

**Missing dependencies with fallback:**
- `horse_track_aptitude.parquet`: Not present; validation should skip gracefully and log warning

## Validation Architecture

> **Note:** `workflow.nyquist_validation` is explicitly set to `false` in `.planning/config.json`. This section is skipped per instructions.

## Security Domain

> Security enforcement follows project defaults. This phase handles financial settlement data integrity but does not introduce new external attack surfaces.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Phase does not add auth |
| V3 Session Management | No | session_id is for crash recovery, not user auth |
| V4 Access Control | No | No new access patterns |
| V5 Input Validation | Yes | D-20 schema validation on bet records |
| V6 Cryptography | Yes | SHA256 for bet_id and track_stats checksums (stdlib hashlib) |

### Known Threat Patterns for Settlement/Pipeline Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Data tampering in bets.parquet | Tampering | Atomic replace (D-07), SHA256 bet_id integrity |
| Stale model loading | Elevation of Privilege | D-16 explicit source selection, no implicit fallback |
| Schema confusion (v1 vs v2) | Tampering | D-18/19 schema_version check, explicit rejection of v1 |
| Missing track_stats causing NaN predictions | Denial of Service | D-15 mandatory artifact check, D-17 fail-fast |

## Sources

### Primary (HIGH confidence)
- Codebase read: `src/backtest/engine.py:163-229` -- payout map function implementations
- Codebase read: `src/paper_trading/reconciler.py` -- current PaperReconciler implementation
- Codebase read: `src/db/model_loader.py:39-931` -- ModelLoader load/load_from_dir methods
- Codebase read: `src/pipelines/training_pipeline.py:970-990,1540-1574,2342-2514` -- track_stats computation and model saving
- Codebase read: `src/features/feature_engine.py:35-82,215-270,390-435` -- cache key computation and source paths
- Codebase read: `scripts/run_paper_trading.py:290-684,895-1115` -- predict and reconcile functions
- Codebase read: `scripts/run_train.py` -- training script (no --betting-target currently)
- Codebase read: `src/domain/models.py:233-277` -- SubmodelSet with track_stats fields
- Codebase read: `src/features/track_condition_features.py:1-60,66-93,96+` -- track stats computation functions
- Codebase read: `src/db/repository.py:77-109` -- load_track_conditions, load_horse_track_aptitude

### Secondary (MEDIUM confidence)
- CONTEXT.md canonical references -- verified against actual code where function names differ (build_payout_map vs build_place_payout_map)

### Tertiary (LOW confidence)
- None -- all findings verified by direct codebase inspection

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all packages verified via `python -c` checks
- Architecture: HIGH - all referenced files read and verified
- Pitfalls: HIGH - discovered by direct codebase inspection (naming discrepancy, track_stats gap, ModelLoader bug)
- Naming discrepancy: HIGH - CONTEXT.md says `build_place_payout_map`, actual code has `build_payout_map`

**Research date:** 2026-06-06
**Valid until:** 2026-07-06 (stable codebase, no fast-moving dependencies)
