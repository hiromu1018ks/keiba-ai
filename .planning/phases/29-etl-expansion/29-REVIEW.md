---
phase: 29-etl-expansion
reviewed: 2026-05-17T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - config/etl_tables.yaml
  - src/db/etl.py
  - src/db/repository.py
  - tests/test_repository.py
  - scripts/run_etl.py
  - tests/test_etl_coverage.py
findings:
  critical: 2
  warning: 4
  info: 1
  total: 7
status: issues_found
---

# Phase 29: Code Review Report

**Reviewed:** 2026-05-17
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Reviewed 6 files from the ETL Expansion phase (Plans 29-01/02/03): YAML table config, ETL engine, DataRepository, test suites, and the run_etl script. Found 2 critical issues and 4 warnings.

The most impactful bug is in `_verify_coverage` which hardcodes category `"odds"`, causing it to silently skip all `raw`-category tables (races, entries, payouts, etc.). The second critical issue is a missing type conversion rule for `odds_umatan` whose `odds` and `ninki` columns will remain as raw strings in Parquet.

## Critical Issues

### CR-01: _verify_coverage hardcodes category "odds" -- raw-category tables silently skipped

**File:** `scripts/run_etl.py:39`
**Issue:** `_verify_coverage` calls `store.exists("odds", table)` and `store.read("odds", table)` with a hardcoded `"odds"` category. However, it is invoked at line 128 with `list(counts.keys())` which includes all tables from the ETL run -- including `raw`-category tables like `races`, `entries`, `payouts`, `horses`, etc. For every `raw`-category table, `store.exists("odds", table_name)` returns `False` (since the data lives at `data/raw/table_name`, not `data/odds/table_name`), and the table is silently skipped with a "file not found" warning. The post-ETL coverage check provides zero value for the majority of tables.

**Fix:** Pass the category information alongside the table name. Either accept a `config` parameter and look up each table's category, or accept a `dict[str, str]` mapping table keys to their categories:

```python
def _verify_coverage(
    store: "ParquetStore",
    tables: list[dict],  # each with 'parquet_key' and 'category'
    start_year: int,
    end_year: int,
) -> None:
    for cfg in tables:
        table = cfg["parquet_key"]
        category = cfg["category"]
        if not store.exists(category, table):
            logger.warning("Coverage SKIP: %s (file not found)", table)
            continue
        df = store.read(category, table)
        ...
```

And at the call site (line 128), pass the config entries instead of just keys.

### CR-02: Missing type conversion rule for odds_umatan -- odds/ninki columns left as strings

**File:** `src/db/etl.py:83-127`
**Issue:** `_TABLE_TYPE_RULES` defines conversion rules for `odds_sanren`, `odds_umaren`, `odds_sanrentan`, and others, but `odds_umatan` is absent. The `n_odds_umatan` table (parquet_key `odds_umatan`) contains `odds` and `ninki` columns that are character varying in EveryDB2 and need `odds10` and `int` conversion respectively -- the same pattern used by `odds_umaren` and `odds_sanren`. Without this rule, the `odds` column will remain as a raw string in the Parquet output, breaking any downstream consumer that expects a numeric odds value.

Similarly, `odds_waku` likely has odds columns that need conversion but has no entry either.

**Fix:** Add type conversion entries for `odds_umatan` (and `odds_waku` if applicable):

```python
_TABLE_TYPE_RULES: dict[str, dict[str, list[str]]] = {
    # ... existing entries ...
    "odds_umatan": {
        "int": ["ninki"],
        "odds10": ["odds"],
    },
    # ... if odds_waku has odds columns ...
    "odds_waku": {
        "odds10": ["odds"],  # or whatever the odds column is named
    },
}
```

## Warnings

### WR-01: DataRepository imports private functions from readers module

**File:** `src/db/repository.py:12`
**Issue:** `from db.readers import _coerce_types, _date_filters` imports two private (underscore-prefixed) functions from another module. Private functions are not part of the public API contract and may change without notice, breaking DataRepository silently. The docstring even mentions "plan to integrate readers.py functions here (D-01)" suggesting these should eventually be moved.

**Fix:** Either (a) promote `_date_filters` and `_coerce_types` to public functions in `readers.py` by removing the underscore prefix, or (b) duplicate the small `_date_filters` logic directly in `repository.py` and move `_coerce_types` there as part of the planned D-01 integration.

### WR-02: _merge_delta uses fragile temporary column names that collide with real data

**File:** `src/db/etl.py:350,356`
**Issue:** The merge logic uses `_delete` and `_upsert` as temporary column names in `DataFrame.assign()`. If the existing DataFrame already contains a column named `_delete` or `_upsert`, the subsequent `merge()` call will produce `_delete_x` / `_delete_y` suffixed columns, and the lookup `merge["_delete"]` at line 351 will raise a `KeyError`. While unlikely in current data, using double-underscore or UUID-suffixed names would be more robust.

**Fix:** Use less collision-prone temporary column names:

```python
marker = "__merge_delete__"
merge = result.merge(delete_keys.assign(**{marker: True}), on=pk, how="left", indicator=False)
result = result[merge[marker] != True].copy()
```

### WR-03: _verify_coverage reads entire partitioned tables without filters

**File:** `scripts/run_etl.py:43`
**Issue:** `store.read("odds", table)` loads the entire Parquet dataset into memory with no predicate pushdown. For partitioned tables like `jodds_tanpuku` (partitioned by year/month), this can be very large. Since the coverage check already knows `start_year` and `end_year`, it could pass date filters to limit the read.

**Fix:** Pass date filters to `store.read()`:

```python
from db.readers import _date_filters
start_str = f"{start_year}0101"
end_str = f"{end_year}1231"
df = store.read(category, table, filters=_date_filters(start_str, end_str))
```

### WR-04: _compute_race_date mutates DataFrame in-place but callers inconsistently capture return

**File:** `src/db/etl.py:256-257`
**Issue:** In the partitioned ETL path (line 256-257), `_compute_race_date(df)` and `_compute_race_id(df)` are called without capturing their return values. These functions happen to mutate `df` in-place and return the same object, so this works today. But in the non-partitioned path (lines 283-285) and delta path (lines 416-417), the same pattern is used. The function signatures suggest they return a DataFrame, which could mislead future maintainers into thinking they return a new object. If anyone changes these functions to return a copy, all three call sites will silently discard the result.

**Fix:** Either consistently capture the return value (`df = _compute_race_date(df)`) or document that these are in-place mutations and change the return type to `None`.

## Info

### IN-01: Test mock DataFrame uses generic schema not matching actual table structures

**File:** `tests/test_repository.py:15-23`
**Issue:** The `mock_store` fixture returns a DataFrame with columns `["race_date", "kumi", "odds", "ninki"]` that is shared across all three load method tests (trio, exacta, trifecta). The actual tables have different schemas: `odds_sanren` and `odds_sanrentan` have `kumi` but `odds_umaren` also has `kumi`. The mock works for testing the method dispatch logic but does not validate that column names or types match actual data. This is acceptable for mock-based tests but worth noting.

**Fix:** Consider using table-specific mock DataFrames if column-level validation is desired in the future.

---

_Reviewed: 2026-05-17_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
