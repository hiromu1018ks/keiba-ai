---
phase: 29-etl-expansion
verified: 2026-05-17T14:00:00Z
status: human_needed
score: 5/5 must-haves verified
overrides_applied: 0
human_verification:
  - test: "run_etl.py --mode full --start 20150101 --end 20251231 を実行し、odds_sanren/odds_umaren/odds_sanrentanのParquetファイルが生成されることを確認"
    expected: "data/odds/odds_sanren.parquet, data/odds/odds_umaren.parquet, data/odds/odds_sanrentan.parquet が生成され、カバレッジログに row count, years=2015-2025, max_missing <= 30% が表示される"
    why_human: "PostgreSQL (EveryDB2) への接続が必要なため、自動検証では実データ抽出をテストできない。インフラの完全性はコード検証で確認済みだが、実際の抽出にはDB接続が必要"
  - test: "カバレッジレポートで2015-2025年の全データが抽出され、欠損率30%以下であることを目視確認"
    expected: "_verify_coverage が WARNING を出力しないこと (missing years なし、missing rate <= 30%)"
    why_human: "実際のDB内容に依存するため、データ品質の最終確認には実際のETL実行が必要"
---

# Phase 29: ETL Expansion Verification Report

**Phase Goal:** 三連複/馬連/三連単オッズがParquetファイルとして利用可能になり、将来の市場クロス整合性特徴量のデータ基盤が整う
**Verified:** 2026-05-17T14:00:00Z
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | PK definitions in etl_tables.yaml use kumi (not umaban1/2/3) for odds_sanren, odds_sanrentan, odds_umaren, odds_wide, odds_umatan (both n_ and s_ variants) | VERIFIED | Python script verified all 10 entries use `kumi` in pk. Targeted odds tables have zero umaban references in pk. |
| 2 | _TABLE_TYPE_RULES has entries for odds_sanren, odds_umaren, odds_sanrentan with odds10 and int conversions | VERIFIED | `odds_sanren: {'int': ['ninki'], 'odds10': ['odds']}`, same for odds_umaren and odds_sanrentan. All 3 verified via import check. |
| 3 | run_etl.py --mode full can extract odds_sanren, odds_umaren, odds_sanrentan tables to Parquet (infrastructure complete) | VERIFIED | etl_tables.yaml has n_odds_sanren/sanrentan/umaren entries with correct category=odds, type=raced. run_full_load reads config, applies _apply_type_conversions with the type rules. _verify_coverage called after extraction in full mode. |
| 4 | Coverage report logs row count, year coverage, max missing rate; warns on missing years and >30% missing rate; gracefully skips nonexistent files | VERIFIED | `_verify_coverage()` in run_etl.py lines 31-77 implements all behaviors. 5/5 tests pass: full coverage, missing years warning, high missing rate warning, nonexistent skip, empty DataFrame handling. |
| 5 | DataRepository.load_trio_odds/load_exacta_odds/load_trifecta_odds return DataFrames via ParquetStore with date filters and type coercion | VERIFIED | DataRepository class (63 lines) has 3 methods delegating to `self._store.read("odds", correct_key, filters=_date_filters(start, end))` then `_coerce_types(df)`. 11/11 tests pass covering init, correct delegation, date filters, return types. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `config/etl_tables.yaml` | Correct PK definitions for 10 odds table entries | VERIFIED | 10 entries (5 odds types x 2 n_/s_ variants) use kumi. No umaban in pk for target tables. |
| `src/db/etl.py` | Type conversion rules for 3 new odds tables | VERIFIED | `_TABLE_TYPE_RULES` has odds_sanren, odds_umaren, odds_sanrentan entries with int/odds10 rules. |
| `src/db/repository.py` | DataRepository class with 3 load methods | VERIFIED | 63 lines, full type annotations, 3 methods delegating to ParquetStore + _coerce_types. |
| `tests/test_repository.py` | Mock-based tests for DataRepository | VERIFIED | 11 tests across TestInit, TestLoadTrioOdds, TestLoadExactaOdds, TestLoadTrifectaOdds. All pass. |
| `scripts/run_etl.py` | Coverage verification post-ETL | VERIFIED | `_verify_coverage()` function (lines 31-77) integrated in main() for full mode (line 128). |
| `tests/test_etl_coverage.py` | Mock-based tests for coverage verification | VERIFIED | 5 tests covering all behavioral cases. All pass. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `config/etl_tables.yaml` | `src/db/etl.py` | load_table_config() loads YAML -> run_full_load uses pk + _TABLE_TYPE_RULES | WIRED | `load_table_config()` reads YAML, `run_full_load()` iterates configs, calls `_apply_type_conversions(df, key)` |
| `src/db/repository.py` | `src/db/parquet_store.py` | ParquetStore injection + read() calls | WIRED | `from db.parquet_store import ParquetStore` (line 11), used in `__init__` and all 3 load methods via `self._store.read()` |
| `src/db/repository.py` | `src/db/readers.py` | imports _coerce_types, _date_filters | WIRED | `from db.readers import _coerce_types, _date_filters` (line 12), used in all 3 load methods |
| `scripts/run_etl.py` | `src/db/parquet_store.py` | ParquetStore.exists() and .read() for coverage | WIRED | `from db.parquet_store import ParquetStore` (line 94), `_verify_coverage()` calls `store.exists()` and `store.read()` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `src/db/repository.py` load_trio_odds | `df` (return value) | `self._store.read("odds", "odds_sanren", filters=...)` | N/A (delegates to ParquetStore; real data requires actual Parquet files) | FLOWING (wired correctly) |
| `src/db/repository.py` load_exacta_odds | `df` (return value) | `self._store.read("odds", "odds_umaren", filters=...)` | N/A | FLOWING (wired correctly) |
| `src/db/repository.py` load_trifecta_odds | `df` (return value) | `self._store.read("odds", "odds_sanrentan", filters=...)` | N/A | FLOWING (wired correctly) |
| `scripts/run_etl.py` _verify_coverage | `df` (from store.read) | `store.read("odds", table)` after `store.exists()` check | N/A (reads from Parquet files produced by ETL) | FLOWING (wired correctly) |

Note: DataRepository and _verify_coverage are correctly wired to ParquetStore, which reads from Parquet files. Actual data flow depends on running ETL against PostgreSQL, which requires human verification.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| PK definitions correct | `python -c "import yaml; c=yaml.safe_load(open('config/etl_tables.yaml')); targets=['odds_sanren','odds_sanrentan','odds_umaren','odds_wide','odds_umatan']; entries=[t for t in c['tables'] if any(t['parquet_key']==x for x in targets)]; bad=[t for t in entries if any('umaban' in str(p) for p in t['pk'])]; assert len(bad)==0; print(f'All {len(entries)} entries verified')"` | "All 10 entries verified: pk uses kumi" | PASS |
| Type rules exist | `python -c "import sys; sys.path.insert(0,'src'); from db.etl import _TABLE_TYPE_RULES; keys=['odds_sanren','odds_umaren','odds_sanrentan']; ..."` | All 3 entries verified with correct int/odds10 rules | PASS |
| Repository tests pass | `python -m pytest tests/test_repository.py -v` | 11 passed | PASS |
| Coverage tests pass | `python -m pytest tests/test_etl_coverage.py -v` | 5 passed | PASS |
| Full test suite no regressions | `python -m pytest tests/ -q` | 1542 passed, 1 skipped | PASS |

### Probe Execution

Step 7c: SKIPPED -- no probe scripts declared in PLAN files and no conventional probe scripts found.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ETL-01 | 29-01, 29-02 | EveryDB2から三連複オッズ (n_odds_sanren) をParquetに抽出するETL拡張 | SATISFIED | PK fixed to kumi, type rules added, DataRepository.load_trio_odds reads "odds_sanren" |
| ETL-02 | 29-01, 29-02 | EveryDB2から馬連オッズ (n_odds_umaren) をParquetに抽出するETL拡張 | SATISFIED | PK fixed to kumi, type rules added, DataRepository.load_exacta_odds reads "odds_umaren" |
| ETL-03 | 29-01, 29-02 | EveryDB2から三連単オッズ (n_odds_sanrentan) をParquetに抽出するETL拡張 | SATISFIED | PK fixed to kumi, type rules added, DataRepository.load_trifecta_odds reads "odds_sanrentan" |
| ETL-04 | 29-03 | ETL抽出データのカバレッジ検証 (2015-2025、欠損率確認) | SATISFIED | _verify_coverage() implemented with year range check and 30% missing rate threshold. 5 tests pass. |

No orphaned requirements: REQUIREMENTS.md maps ETL-01 through ETL-04 to Phase 29, matching all PLAN frontmatter declarations.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No TBD/FIXME/XXX found. No TODO/HACK/PLACEHOLDER found. No empty implementations. |

No anti-patterns detected in any phase 29 files.

### Human Verification Required

### 1. ETL Full Extraction Test

**Test:** `python scripts/run_etl.py --mode full --start 20150101 --end 20251231`
**Expected:** Parquet files `data/odds/odds_sanren.parquet`, `data/odds/odds_umaren.parquet`, `data/odds/odds_sanrentan.parquet` are generated. Coverage log shows row counts, years=[2015..2025], and max_missing <= 30% with no WARNING messages.
**Why human:** Requires running PostgreSQL with EveryDB2 data. Cannot be tested without a live database connection.

### 2. Data Quality Visual Confirmation

**Test:** After ETL, review coverage log output for all new odds tables.
**Expected:** No "missing years" warnings, no "missing rate exceeds 30%" warnings.
**Why human:** Data quality depends on actual DB contents. The coverage verification infrastructure is verified (code + tests), but the actual data content requires real DB access.

### Gaps Summary

No gaps found in the code. All infrastructure is correctly wired:

- PK definitions corrected for 10 entries (5 odds types x n_/s_ variants)
- Type conversion rules added for 3 new odds tables
- DataRepository provides 3 clean load methods following the established load_wide_odds pattern
- Post-ETL coverage verification is implemented and tested
- Full test suite passes (1542 tests, 0 failures)

The only remaining verification is running the actual ETL against PostgreSQL to confirm data extraction works end-to-end with real data.

---

_Verified: 2026-05-17T14:00:00Z_
_Verifier: Claude (gsd-verifier)_
