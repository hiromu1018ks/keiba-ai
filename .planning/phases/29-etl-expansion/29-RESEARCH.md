# Phase 29: ETL Expansion - Research

**Researched:** 2026-05-17
**Domain:** ETL pipeline / Parquet data layer
**Confidence:** HIGH

## Summary

Phase 29 extends the existing YAML-driven ETL pipeline to extract 6 new odds tables (trio/quinella/trifecta + their _head counterparts) from EveryDB2 into Parquet files. The ETL engine (`src/db/etl.py`) already handles these tables -- the YAML config entries exist in `config/etl_tables.yaml` with `type: raced`. The primary work is (1) fixing PK definitions from `umaban1/umaban2/umaban3` to `kumi` in the YAML, (2) adding `_TABLE_TYPE_RULES` entries for the 3 main odds tables so odds/ninki columns get proper type conversion, (3) creating a new `DataRepository` class with 3 load methods, and (4) adding coverage verification to `run_etl.py`.

The wide odds (`n_odds_wide`) is the closest analog. Its pattern is: YAML config entry -> ETL extracts with `type: raced` -> `_apply_type_conversions` with `odds100` rule -> ParquetStore writes to `data/odds/odds_wide.parquet` -> `readers.py::load_wide_odds()` reads with date filters. The new tables follow this exact pattern but use a single `odds` column with `odds10` divisor instead of `oddslow/oddshigh` with `odds100`.

**Primary recommendation:** Fix PK definitions in YAML, add 3 entries to `_TABLE_TYPE_RULES`, create `src/db/repository.py` as a clean class-based wrapper following the `load_wide_odds` pattern, and add a lightweight coverage report function to `run_etl.py`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 新規 `src/db/repository.py` に `DataRepository` クラスを作成。3メソッドのみ実装: `load_trio_odds()`, `load_exacta_odds()`, `load_trifecta_odds()`
- **D-02:** 既存 `readers.py` は変更しない
- **D-03:** DataRepository は ParquetStore を内部で使用し、`_date_filters` + `_coerce_types` パターンを踏襲
- **D-04:** カバレッジ検証を `run_etl.py` に組み込む。ETL実行後、抽出テーブルの行数・年度カバレッジ・欠損率を自動出力
- **D-05:** 検証基準: 2015-2025カバー、欠損率30%以下
- **D-06:** 本体 + head 両方抽出
- **D-07:** headテーブルは datakubun (5=確定/9=最終) と sanrenflag/sanrentanflag/umarenflag を含む
- **D-08:** 特定テーブルのみ抽出: `--tables odds_sanren odds_sanren_head odds_umaren odds_umaren_head odds_sanrentan odds_sanrentan_head`
- **D-09:** PK定義修正: n_odds_sanren/sanrentan/umaren + s_系 → pk に kumi を使用
- **D-10:** 既存 n_odds_wide のPK定義も確認 (umaban1/umaban2 → kumi)
- **D-11:** 3テーブル共通カラム: makedate, year, monthday, jyocd, kaiji, nichiji, racenum, kumi, odds, ninki
- **D-12:** kumiフォーマット: 三連複="010203"(6桁), 馬連="0102"(4桁), 三連単="010203"(6桁)
- **D-13:** n_odds_wide は oddslow/oddshigh の2列だが、sanren/umaren/sanrentanは単一 odds 列
- **D-14:** データは継続蓄積中 (sanren=454K, sanrentan=2.6M, umaren=108K 行)

### Claude's Discretion
- DataRepository の内部実装詳細 (型アノテーション、キャッシュ等)
- カバレッジレポートの出力フォーマット
- テストケースの具体的な設計

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ETL-01 | EveryDB2から三連複オッズ (n_odds_sanren) をParquetに抽出 | YAML config fix (D-09), _TABLE_TYPE_RULES entry, run_etl.py --tables |
| ETL-02 | EveryDB2から馬連オッズ (n_odds_umaren) をParquetに抽出 | Same pattern as ETL-01 |
| ETL-03 | EveryDB2から三連単オッズ (n_odds_sanrentan) をParquetに抽出 | Same pattern as ETL-01, largest table (2.6M rows) |
| ETL-04 | ETL抽出データのカバレッジ検証 (2015-2025、欠損率確認) | Coverage verification function in run_etl.py |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| ETL config (YAML) | Data / Storage | -- | Static configuration defines PK and table mapping |
| ETL engine (read DB + write Parquet) | Data / Storage | -- | `src/db/etl.py` orchestrates read-transform-write |
| Type conversion rules | Data / Storage | -- | `_TABLE_TYPE_RULES` in `etl.py` handles column casting |
| DataRepository (read Parquet) | Data / Storage | API / Backend | New class reads Parquet for downstream ML pipelines |
| Coverage verification | Script / CLI | -- | `run_etl.py` post-ETL validation |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | (existing) | DataFrame I/O and transformation | Project standard for tabular data |
| pyarrow | (existing) | Parquet read/write with predicate pushdown | ParquetStore backend |
| SQLAlchemy | (existing) | PostgreSQL query execution via `text()` | ETL engine uses Core SQL |
| PyYAML | (existing) | etl_tables.yaml loading | Config-driven ETL |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | (existing) | ETL progress bar | Already used in `run_full_load` |

No new dependencies needed. All required libraries are already installed.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────┐
                    │  config/            │
                    │  etl_tables.yaml    │
                    │  (6 table entries)  │
                    └────────┬────────────┘
                             │ load_table_config()
                             ▼
┌──────────────┐    ┌─────────────────────┐    ┌──────────────────┐
│ PostgreSQL   │───>│  src/db/etl.py      │───>│ ParquetStore     │
│ EveryDB2     │    │  run_full_load()    │    │ .write()         │
│ n_odds_*     │    │  _read_db_table()   │    │                  │
└──────────────┘    │  _apply_type_conv() │    │ data/odds/       │
                    │  _compute_race_*()  │    │  odds_sanren.parquet
                    └────────┬────────────┘    │  odds_umaren.parquet
                             │                 │  odds_sanrentan.parquet
                             │                 │  odds_*_head.parquet
                             ▼                 └────────┬─────────┘
                    ┌─────────────────────┐             │
                    │ scripts/run_etl.py   │             │ store.read()
                    │ --mode full          │             ▼
                    │ --tables odds_sanren │    ┌─────────────────────┐
                    │ ...                  │    │ DataRepository      │
                    │ + coverage verify    │    │ (src/db/repository) │
                    └─────────────────────┘    │ .load_trio_odds()   │
                                               │ .load_exacta_odds() │
                                               │ .load_trifecta_odds()│
                                               └────────┬────────────┘
                                                        │
                                                        ▼
                                               Phase 32 (downstream)
```

### Recommended Project Structure
```
src/db/
├── etl.py              # MODIFY: add _TABLE_TYPE_RULES entries for 3 odds tables
├── repository.py       # CREATE: DataRepository class (3 methods)
├── parquet_store.py    # NO CHANGE
├── readers.py          # NO CHANGE (D-02)
├── connection.py       # NO CHANGE
├── schema.py           # NO CHANGE
config/
├── etl_tables.yaml     # MODIFY: fix PK definitions for 6 tables (umaban1/2/3 -> kumi)
scripts/
├── run_etl.py          # MODIFY: add coverage verification post-ETL
tests/
├── test_repository.py  # CREATE: DataRepository tests
├── test_etl_coverage.py # CREATE: coverage verification tests
data/odds/              # OUTPUT: new Parquet files created by ETL
├── odds_sanren.parquet
├── odds_umaren.parquet
├── odds_sanrentan.parquet
├── odds_sanren_head.parquet
├── odds_umaren_head.parquet
├── odds_sanrentan_head.parquet
```

### Pattern 1: ETL Config-Driven Table Extraction
**What:** Each table in `etl_tables.yaml` defines db_table, parquet_key, category, type, and pk. The ETL engine iterates over configs and processes each table generically.
**When to use:** For every new table extraction. Just add a YAML entry and the engine handles it.
**Example:**
```yaml
# config/etl_tables.yaml - CURRENT (incorrect PK)
- db_table: n_odds_sanren
  parquet_key: odds_sanren
  category: odds
  type: raced
  pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, umaban3]

# config/etl_tables.yaml - FIXED (D-09)
- db_table: n_odds_sanren
  parquet_key: odds_sanren
  category: odds
  type: raced
  pk: [year, monthday, jyocd, kaiji, nichiji, racenum, kumi]
```

### Pattern 2: Type Conversion Rules
**What:** `_TABLE_TYPE_RULES` in `etl.py` maps parquet_key to column conversion rules (int, float, odds10, odds100). When no rule exists, `_apply_type_conversions` returns the DataFrame unchanged.
**When to use:** For every odds table that has varchar columns needing numeric conversion.
**Example:**
```python
# Source: src/db/etl.py lines 83-115
_TABLE_TYPE_RULES: dict[str, dict[str, list[str]]] = {
    # ... existing entries ...
    "odds_wide": {
        "odds100": ["oddslow", "oddshigh"],
    },
    # NEW entries needed:
    "odds_sanren": {
        "int": ["ninki"],
        "odds10": ["odds"],      # JRA-VAN stores as "150" meaning 15.0
    },
    "odds_umaren": {
        "int": ["ninki"],
        "odds10": ["odds"],
    },
    "odds_sanrentan": {
        "int": ["ninki"],
        "odds10": ["odds"],
    },
}
```

**Key detail on odds divisor:** The `odds10` rule divides by 10 (e.g., "150" becomes 15.0). The `odds100` rule divides by 100. The new tables use a single `odds` column that follows the `odds10` pattern (same as entries.odds). The `_head` tables have no odds column -- they contain metadata flags (datakubun, sanrenflag, etc.) and do NOT need type conversion entries since they have no numeric odds columns. [VERIFIED: codebase analysis of etl.py and CONTEXT.md D-11/D-13]

### Pattern 3: Reader Function Pattern (load_wide_odds reference)
**What:** Reader functions follow: `store.read(category, name, filters=date_filters)` -> `_coerce_types(df)` -> return.
**When to use:** For every new Parquet reader.
**Example:**
```python
# Source: src/db/readers.py lines 251-253
def load_wide_odds(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("odds", "odds_wide", filters=_date_filters(start, end))
    return _coerce_types(df)
```

### Pattern 4: DataRepository Class (to be created)
**What:** Class-based data access wrapping ParquetStore. Constructor takes a ParquetStore instance. Methods follow the reader pattern but are instance methods.
**When to use:** Per D-01, this is the new pattern for Phase 29 and beyond.
**Example:**
```python
# Planned: src/db/repository.py
from __future__ import annotations

import pandas as pd
from db.parquet_store import ParquetStore
from db.readers import _coerce_types, _date_filters


class DataRepository:
    """Data access layer for ML pipeline. Wraps ParquetStore with domain logic."""

    def __init__(self, store: ParquetStore | None = None) -> None:
        self._store = store or ParquetStore()

    def load_trio_odds(self, start: str, end: str) -> pd.DataFrame:
        """Load trio (sanren) odds from Parquet."""
        df = self._store.read("odds", "odds_sanren", filters=_date_filters(start, end))
        return _coerce_types(df)

    def load_exacta_odds(self, start: str, end: str) -> pd.DataFrame:
        """Load exacta (umaren) odds from Parquet."""
        df = self._store.read("odds", "odds_umaren", filters=_date_filters(start, end))
        return _coerce_types(df)

    def load_trifecta_odds(self, start: str, end: str) -> pd.DataFrame:
        """Load trifecta (sanrentan) odds from Parquet."""
        df = self._store.read("odds", "odds_sanrentan", filters=_date_filters(start, end))
        return _coerce_types(df)
```

**Note on `_coerce_types` and `_date_filters`:** These are module-level functions in `readers.py`. D-03 says to follow the pattern. The DataRepository can import them directly (`from db.readers import _coerce_types, _date_filters`). They are already tested in `tests/test_readers.py`. [VERIFIED: readers.py exports these as module-level functions]

### Pattern 5: Test Pattern (mock-based, no DB)
**What:** All tests use `unittest.mock.MagicMock` to mock ParquetStore. No actual DB connection or file I/O.
**When to use:** For all new test files.
**Example:**
```python
# Source: tests/test_readers.py lines 24-28, 56-61
@pytest.fixture
def mock_store():
    store = MagicMock()
    store.read.return_value = MagicMock()
    return store


class TestLoadWideOdds:
    def test_calls_store_with_correct_args(self, mock_store):
        load_wide_odds(mock_store, "20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("odds", "odds_wide")
```

### Anti-Patterns to Avoid
- **Adding reader functions to readers.py:** D-02 explicitly says no changes to readers.py. All new data access goes through the new DataRepository class.
- **Adding type conversion in DataRepository:** Type conversion happens at ETL time via `_TABLE_TYPE_RULES`. The `_coerce_types` call in the reader is a fallback for legacy Parquet files. Do not duplicate type conversion logic.
- **Forgetting _head tables in _TABLE_TYPE_RULES:** The _head tables (odds_sanren_head, etc.) have no odds column. They should NOT have entries in `_TABLE_TYPE_RULES`. The ETL engine handles them fine without type rules (returns df unchanged).
- **Using partition_cols for new tables:** The new odds tables are not large enough to need year/month partitioning. sanrentan at 2.6M rows is the largest but still manageable as a single Parquet file. Only `jodds_tanpuku` uses `partition_cols`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Type conversion for odds columns | Custom parsing logic | `_TABLE_TYPE_RULES` + `_apply_type_conversions` | Handles "150" -> 15.0 (odds10), null strings, edge cases |
| Date filtering | Custom WHERE clauses | `_date_filters(start, end)` | Consistent datetime filter format across all readers |
| Parquet I/O | Custom file operations | `ParquetStore.read/write` | Atomic writes, pyarrow pushdown, partition support |
| Delta merge | Custom upsert logic | `etl.py::_merge_delta` | PK-based upsert/delete with datakubun handling |

## Common Pitfalls

### Pitfall 1: Wrong PK Definition Breaks Delta Merge
**What goes wrong:** If PK columns don't match actual DB schema, delta ETL (`_merge_delta`) will fail because it validates PK existence in both existing and delta DataFrames.
**Why it happens:** The YAML currently has `umaban1, umaban2, umaban3` but the actual DB uses a single `kumi` column.
**How to avoid:** Fix ALL 12 entries (6 n_ tables + 6 s_ tables) to use `kumi` instead of `umaban1/umaban2/umaban3`.
**Warning signs:** Delta ETL throws `ValueError: PK columns mismatch`.

### Pitfall 2: Missing _TABLE_TYPE_RULES Leaves Columns as Strings
**What goes wrong:** Without type rules, `odds` and `ninki` columns remain as varchar strings in Parquet. Downstream code expecting float/int will break.
**Why it happens:** `_apply_type_conversions` returns df unchanged when no rule exists for the table key.
**How to avoid:** Add entries for `odds_sanren`, `odds_umaren`, `odds_sanrentan` with `"int": ["ninki"]` and `"odds10": ["odds"]`.
**Warning signs:** `df["odds"].dtype == object` instead of `float64`.

### Pitfall 3: n_odds_wide PK Also Wrong
**What goes wrong:** n_odds_wide PK is `[year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2]` but D-10 says the actual column is `kumi`. This affects both n_ and s_ entries for odds_wide.
**Why it happens:** Historical oversight from initial YAML creation.
**How to avoid:** Fix n_odds_wide and s_odds_wide PK to use `kumi` as well. Note: odds_wide also has `oddslow`/`oddshigh` (not single `odds`) so its type rules are different.
**Warning signs:** Delta merge for odds_wide fails.

### Pitfall 4: Coverage Verification Fails for Empty Tables
**What goes wrong:** Coverage check reads Parquet files that don't exist yet (first-time extraction).
**Why it happens:** `run_etl.py` runs coverage immediately after ETL, but if ETL failed for a table, the file won't exist.
**How to avoid:** Guard coverage verification with `store.exists()` checks. Log warnings for missing files instead of crashing.
**Warning signs:** FileNotFoundError during coverage step.

### Pitfall 5: Large trifecta Table (2.6M rows) ETL Time
**What goes wrong:** n_odds_sanrentan has 2.6M rows. Single SQL SELECT without partitioning may be slow.
**Why it happens:** The ETL engine reads all rows in one query for non-partitioned tables.
**How to avoid:** This is acceptable for full load mode. The table is `type: raced` so it gets date-filtered via `year || monthday BETWEEN :start AND :end`. For 2015-2025 that's ~10 years of data. Monitor execution time.
**Warning signs:** ETL timeout or excessive memory usage.

## Code Examples

### ETL PK Fix (etl_tables.yaml)
```yaml
# BEFORE (lines 95-111) - incorrect PK with umaban1/2/3
- db_table: n_odds_sanren
  parquet_key: odds_sanren
  category: odds
  type: raced
  pk: [year, monthday, jyocd, kaiji, nichiji, racenum, umaban1, umaban2, umaban3]

# AFTER - fixed with kumi
- db_table: n_odds_sanren
  parquet_key: odds_sanren
  category: odds
  type: raced
  pk: [year, monthday, jyocd, kaiji, nichiji, racenum, kumi]
```
The same fix applies to all 12 entries: n_odds_sanren, s_odds_sanren, n_odds_sanrentan, s_odds_sanrentan, n_odds_umaren, s_odds_umaren, n_odds_wide, s_odds_wide.

### Type Rules Addition (src/db/etl.py)
```python
# Add after "odds_wide" entry (line 106) in _TABLE_TYPE_RULES:
    "odds_sanren": {
        "int": ["ninki"],
        "odds10": ["odds"],
    },
    "odds_umaren": {
        "int": ["ninki"],
        "odds10": ["odds"],
    },
    "odds_sanrentan": {
        "int": ["ninki"],
        "odds10": ["odds"],
    },
```
[VERIFIED: codebase analysis confirms the existing pattern at etl.py:83-115]

### Coverage Verification Function (scripts/run_etl.py)
```python
# Add after ETL execution (after the for loop at line 72)
def _verify_coverage(store: ParquetStore, tables: list[str], start_year: int, end_year: int) -> None:
    """Post-ETL coverage verification for extracted tables."""
    for key in tables:
        if not store.exists("odds", key):
            logger.warning("  Coverage SKIP: %s (file not found)", key)
            continue
        df = store.read("odds", key)
        n_rows = len(df)
        years_covered = sorted(df["race_date"].dt.year.unique()) if "race_date" in df.columns else []
        missing_pct = (df.isnull().mean().max() * 100) if not df.empty else 0

        logger.info("  Coverage %s: %d rows, years=%s, max_missing=%.1f%%",
                     key, n_rows, years_covered, missing_pct)

        # Validate (D-05)
        expected_years = set(range(start_year, end_year + 1))
        actual_years = set(years_covered)
        missing_years = expected_years - actual_years
        if missing_years:
            logger.warning("  Coverage WARN: %s missing years %s", key, sorted(missing_years))
        if missing_pct > 30:
            logger.warning("  Coverage WARN: %s missing rate %.1f%% exceeds 30%%", key, missing_pct)
```
[ASSUMED: coverage verification pattern -- needs confirmation on desired output format]

### DataRepository Test Pattern (tests/test_repository.py)
```python
"""DataRepository のテスト（モック使用・DB接続不要）"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from db.repository import DataRepository


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.read.return_value = pd.DataFrame({
        "race_date": pd.to_datetime(["2024-01-01"]),
        "kumi": ["0102"],
        "odds": [15.0],
        "ninki": [1],
    })
    return store


class TestLoadTrioOdds:
    def test_calls_store_with_correct_args(self, mock_store):
        repo = DataRepository(store=mock_store)
        repo.load_trio_odds("20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("odds", "odds_sanren")
        assert kwargs["filters"] is not None

    def test_returns_coerced_dataframe(self, mock_store):
        repo = DataRepository(store=mock_store)
        result = repo.load_trio_odds("20240101", "20241231")
        assert isinstance(result, pd.DataFrame)
```
[VERIFIED: test pattern matches tests/test_readers.py]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| readers.py module functions | DataRepository class (Phase 29) | Now | Class-based pattern for new data access |

**Deprecated/outdated:**
- None in this phase scope. The existing readers.py functions remain valid; DataRepository is additive.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | New odds tables use `odds10` divisor (divide by 10), not `odds100` | Pattern 2 | Odds values would be 10x too small or too large |
| A2 | _head tables need NO `_TABLE_TYPE_RULES` entries | Pattern 2 | _head tables would have unconverted columns (minor since no odds column) |
| A3 | Single Parquet file (no partition_cols) is acceptable for sanrentan (2.6M rows) | Anti-Patterns | Slow reads if file grows beyond ~5M rows |
| A4 | `kumi` is already a string column in `_STRING_COLUMNS` in readers.py (line 74) | Pattern 4 | If not in _STRING_COLUMNS, `_coerce_types` would try to convert it to numeric |

**A1 verification needed:** The CONTEXT.md D-13 says "sanren/umaren/sanrentanは単一 odds 列" and D-11 confirms all columns are varchar. The existing `entries` table uses `"odds10": ["odds"]` for the same JRA-VAN odds encoding, which is strong evidence. However, the exact divisor should be confirmed by checking one actual data value against a known odds.

**A4 confirmed:** `kumi` is already in `_STRING_COLUMNS` at readers.py line 74. No risk. [VERIFIED: readers.py]

## Open Questions

1. **Odds divisor verification**
   - What we know: Entries table uses odds10 (divide by 10) for its `odds` column. Wide uses odds100 (divide by 100) for oddslow/oddshigh.
   - What's unclear: Whether the new odds tables follow odds10 or odds100 encoding.
   - Recommendation: Use `odds10` based on the entries table precedent (same `odds` column name, same JRA-VAN source). Planners should flag this for runtime verification after ETL -- check that extracted odds values are in reasonable ranges (e.g., trio odds 5.0 - 500.0).

2. **Wide PK fix scope**
   - What we know: D-10 says to check n_odds_wide PK. Current PK uses `umaban1, umaban2`.
   - What's unclear: Whether to fix this now or defer.
   - Recommendation: Fix it now since it's a trivial YAML edit and prevents future delta merge failures.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PostgreSQL (EveryDB2) | ETL extraction | -- | -- | Cannot proceed without DB |
| Python 3.11 (mise) | All code | -- | -- | -- |
| pip packages | All code | -- | -- | -- |

**Note:** ETL execution requires a running PostgreSQL instance with EveryDB2 data. This is a runtime dependency for the actual data extraction step, not for code development or testing (all tests use mocks).

## Sources

### Primary (HIGH confidence)
- `config/etl_tables.yaml` - Full 103-table ETL configuration with PK definitions for all target tables
- `src/db/etl.py` - ETL engine: `_TABLE_TYPE_RULES`, `_apply_type_conversions`, `run_full_load`, `_merge_delta`
- `src/db/parquet_store.py` - ParquetStore read/write API
- `src/db/readers.py` - Reader function patterns, `_coerce_types`, `_date_filters`, `_STRING_COLUMNS`
- `scripts/run_etl.py` - CLI entry point with `--tables` filtering
- `.planning/phases/29-etl-expansion/29-CONTEXT.md` - D-01 through D-14 decisions
- `.planning/REQUIREMENTS.md` - ETL-01 through ETL-04 requirements

### Secondary (MEDIUM confidence)
- `tests/test_readers.py` - Test pattern for reader functions with mock ParquetStore
- `tests/test_db.py` - Test pattern for DB module tests
- `src/db/schema.py` - Schema definitions (no changes needed)

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in use, no new dependencies
- Architecture: HIGH - exact code paths traced through existing codebase
- Pitfalls: HIGH - PK issue confirmed by reading YAML, type rules gap confirmed by reading etl.py
- Odds divisor: MEDIUM - assumed odds10 based on entries table precedent, not verified against actual data

**Research date:** 2026-05-17
**Valid until:** 2026-06-17 (stable codebase, no fast-moving dependencies)
