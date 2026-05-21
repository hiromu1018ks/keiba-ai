# Phase 35: ETL Data Foundation - Research

**Researched:** 2026-05-19
**Domain:** ETL type conversion, sentinel value handling, POST_RACE safety
**Confidence:** HIGH

## Summary

Phase 35 adds sentinel-aware type conversion rules to the ETL pipeline for HaronTimeL3/L4, LapTime1~25, and Jyuni2c/3c columns. The current ETL engine (`src/db/etl.py`) uses a declarative `_TABLE_TYPE_RULES` dictionary with rule types `int`, `float`, `odds10`, and `odds100`. This phase adds two new rule types: `sentinel_float` and `sentinel_int`, which replace designated sentinel string values with NaN before converting to float64.

The HaronTimeL3 column is already registered under `entries.float` rules, meaning it gets converted to float64 but **sentinel values "000" and "999" pass through as 0.0 and 999.0** -- they are NOT converted to NaN by the current `_to_float` logic. This is the primary data quality issue. HaronTimeL4, Jyuni2c/3c, and all LapTime1~25 are completely unconverted (remain as varchar strings in Parquet).

POST_RACE_COLS currently has 16 entries and needs 25 more (LapTime1~25), expanding to 41 entries. Three files define POST_RACE_COLS independently; two need to become imports from `domain/types.py`.

**Primary recommendation:** Extend `_TABLE_TYPE_RULES` with `sentinel_float`/`sentinel_int` entries, migrate HaronTimeL3 from `float` to `sentinel_float`, and add the new columns. Then consolidate POST_RACE_COLS and verify the 3-layer leakage tests still pass.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** LapTime1~25 (all 25 columns) added to POST_RACE_COLS. 16 -> 41 columns.
- **D-02:** Existing 3-layer CI leakage tests need no modification. LapTime is race-level so build_all() output naturally excludes it.
- **D-03:** Claude manual quality verification after ETL execution (float64 type, NaN conversion).
- **D-04:** No automated tests for Parquet quality verification (all tests use mocks, no DB).
- **D-05:** HaronTimeL3/L4 separately float64 (no coalesce). Integration logic deferred to Phase 36.
- **D-06:** Mutual exclusivity verified by Claude after ETL (L3-only/L4-only/both/neither distribution).
- **D-07:** Add declarative sentinel rule types (`sentinel_float` / `sentinel_int`) to `_TABLE_TYPE_RULES`. Structure: `{"sentinel_float": {"columns": [...], "sentinels": ("000", "999")}}`. Two-phase processing in `_apply_type_conversions`: sentinel replacement, then type conversion.
- **D-08:** Sentinel values: HaronTimeL3/L4 = 000+999, LapTime1~25 = 000, Jyuni2c/3c = 000. All processed as sentinel_float (Jyuni also float->Int64 conversion).
- **D-09:** readers.py _INT_COLS/_FLOAT_COLS updated for backward compatibility. Add: _FLOAT_COLS += harontimel4 + laptime1~25, _INT_COLS += jyuni2c/jyuni3c.
- **D-10:** POST_RACE_COLS consolidated to types.py. test_paper_trading_guards.py and run_paper_trading.py change duplicate definitions to imports.

### Claude's Discretion
- Specific implementation of sentinel_float/sentinel_int rules in _TABLE_TYPE_RULES
- Processing order within _apply_type_conversions (sentinel replacement -> type conversion pipeline)
- Specific column lists for readers.py updates
- ETL quality verification procedures
- Test case design (sentinel rule validity, type conversion, POST_RACE_COLS updates)

### Deferred Ideas (OUT OF SCOPE)
- harontime_last3f integration logic (coalesce/distance-based selection) -- Phase 36
- LapTime feature engineering (front/middle/back pace ratios etc.) -- Phase 36 HLF-03
- Jyuni corner position -> running style features -- Phase v2 HLF-06
- Automated ETL quality tests -- PostgreSQL environment dependency
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ETL-01 | HaronTimeL3/L4 (SE table) float64 conversion with sentinel (000/999) -> NaN | HaronTimeL3 currently in entries.float (no sentinel handling); HaronTimeL4 not registered. Migrate both to new sentinel_float rule. |
| ETL-02 | LapTime1~25 (RA table) float64 conversion with sentinel (000) -> NaN | 25 columns all varchar(3) in RA table. Need sentinel_float rule for races table. Format: "345" = 34.5 sec (divide by 10). |
| ETL-03 | Jyuni1c~4c corner position numeric conversion | jyuni1c/4c already in entries.int (converted). jyuni2c/3c not registered. Need sentinel_float for jyuni2c/3c with 000 sentinel. |
| ETL-04 | All new POST_RACE columns registered in types.py, 3-layer CI leakage tests work | Add laptime1~25 to POST_RACE_COLS. Leakage tests reference POST_RACE_COLS from types.py -- new columns auto-protected. |
| ETL-05 | HaronTimeL3/L4 mutual exclusivity validation and documentation | Post-ETL Claude verification. RA table also has HaronTimeL3/L4 at race level (fields 96-97); SE table has per-horse values (fields 58-59). |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Sentinel-aware type conversion | ETL (etl.py) | -- | _TABLE_TYPE_RULES + _apply_type_conversions own all type rules |
| Backward-compatible type coercion | Data access (readers.py) | -- | _INT_COLS/_FLOAT_COLS for old Parquet files |
| POST_RACE column registry | Domain (types.py) | -- | Single source of truth for leakage detection |
| POST_RACE leakage detection | Test layer (tests/) | -- | 3-layer CI tests validate no leakage |
| POST_RACE column dedup | Scripts/tests | -- | run_paper_trading.py, test_paper_trading_guards import from types.py |

## Current State Analysis

### _TABLE_TYPE_RULES structure (src/db/etl.py lines 83-131)

```python
_TABLE_TYPE_RULES: dict[str, dict[str, list[str]]] = {
    "races": {
        "int": ["trackcd", "kyori", "tenkocd", "syussotosu", "honsyokin"],
    },
    "entries": {
        "int": ["umaban", "kakuteijyuni", "ninki", "kyakusitukubun",
                "jyuni1c", "jyuni4c", "zogenfugo"],
        "float": ["time", "bataijyu", "zogensa", "harontimel3", "timediff"],
        "odds10": ["odds"],
    },
    # ... odds tables ...
    "payouts": {
        "int": ["paytansyoumaban1"] + [f"payfukusyoumaban{i}" for i in range(1, 6)],
        "float": ["paytansyopay1"] + [f"payfukusyopay{i}" for i in range(1, 6)],
    },
}
```

Rule types iterate in `_apply_type_conversions()` (lines 134-176):
1. `int` -> `_to_int()` -> `Int64` nullable integer
2. `float` -> `_to_float()` -> float64 (NaN for empty/unparseable)
3. `odds10` -> `_to_odds(v, 10)` -> float64 / 10
4. `odds100` -> `_to_odds(v, 100)` -> float64 / 100

### Critical gap: HaronTimeL3 sentinel handling

`_to_float()` converts `"000"` to `0.0` and `"999"` to `999.0` -- these are NOT sentinel-corrected. Current entries.float rule for harontimel3 produces **corrupted data** where:
- Sentinels "000" (initial/no-data) appear as 0.0 seconds (impossible for 3-furlong time)
- Sentinels "999" (DNF/scratched) appear as 999.0 seconds

**CONFIDENCE: HIGH** -- verified by reading `_to_float` at etl.py:148-154 which uses bare `float(val)` with no sentinel check.

### What exists vs. what's needed

| Column | Table | Current Rule | Needs |
|--------|-------|-------------|-------|
| harontimel3 | entries | `float` (no sentinel) | Migrate to `sentinel_float` with sentinels ["000", "999"] |
| harontimel4 | entries | None (varchar) | Add `sentinel_float` with sentinels ["000", "999"] |
| jyuni2c | entries | None (varchar) | Add `sentinel_float` with sentinels ["000"] |
| jyuni3c | entries | None (varchar) | Add `sentinel_float` with sentinels ["000"] |
| laptime1~25 | races | None (varchar) | Add `sentinel_float` with sentinels ["000"], divide by 10 |
| harontimel3 | races (RA) | None (varchar) | Add `sentinel_float` with sentinels ["000", "999"] |
| harontimel4 | races (RA) | None (varchar) | Add `sentinel_float` with sentinels ["000", "999"] |

### POST_RACE_COLS triplication (3 locations)

1. **`src/domain/types.py` lines 38-55** -- canonical list (16 entries). This is the authoritative source.
2. **`tests/test_paper_trading_guards.py` lines 5-22** -- duplicate tuple (16 entries, identical content).
3. **`scripts/run_paper_trading.py` lines 57-74** -- duplicate tuple (16 entries, identical content).

All three are currently identical. D-10 requires consolidating to types.py import.

### readers.py backward-compat sets

```python
_INT_COLS: set[str] = {
    "trackcd", "kyori", "tenkocd", "syussotosu", "honsyokin",
    "umaban", "kakuteijyuni", "ninki", "kyakusitukubun",
    "jyuni1c", "jyuni4c", "zogenfugo", "tanninki",
}
_FLOAT_COLS: set[str] = {
    "time", "bataijyu", "zogensa", "harontimel3", "timediff",
}
```

Need to add: `harontimel4`, `laptime1`~`laptime25` to _FLOAT_COLS, and `jyuni2c`, `jyuni3c` to _INT_COLS.

## Technical Approach

### 1. sentinel_float / sentinel_int rule structure

Per D-07, the structure should be a dict with `columns` and `sentinels`:

```python
"sentinel_float": {
    "columns": ["harontimel3", "harontimel4"],
    "sentinels": ["000", "999"],
}
```

However, `_TABLE_TYPE_RULES` currently has a flat `dict[str, list[str]]` structure where each rule type maps to a simple list of column names. The sentinel rules need richer structure. Two design options:

**Option A (Recommended): Nested dict for sentinel rules**
```python
_TABLE_TYPE_RULES: dict[str, dict[str, list[str] | dict]] = {
    "entries": {
        "int": ["umaban", ...],
        "float": ["time", ...],
        "sentinel_float": {
            "columns": ["harontimel3", "harontimel4"],
            "sentinels": ["000", "999"],
        },
    },
}
```

This changes the type signature from `dict[str, dict[str, list[str]]]` to `dict[str, dict[str, list[str] | dict]]`. The `_apply_type_conversions` function needs to handle the new dict-typed rules.

**Option B: Separate top-level sentinel dict**
Keep `_TABLE_TYPE_RULES` unchanged, add `_TABLE_SENTINEL_RULES` as a new dict.

Option A is recommended because it keeps all rules for a table in one place and the CONTEXT.md decision D-07 specifies adding to _TABLE_TYPE_RULES.

### 2. _apply_type_conversions extension

Add a new processing block after existing rules:

```python
# Sentinel float: replace sentinel strings with NaN, then convert to float64
sentinel_float_rule = rules.get("sentinel_float")
if isinstance(sentinel_float_rule, dict):
    cols = sentinel_float_rule.get("columns", [])
    sentinels = sentinel_float_rule.get("sentinels", [])
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(sentinels, float("nan"))
            df[col] = pd.to_numeric(df[col], errors="coerce")
```

The processing order matters: sentinel_float rules must run BEFORE any regular float rules for the same column. However, per the design, HaronTimeL3 should be REMOVED from the `float` list and moved entirely to `sentinel_float`. This avoids double-processing.

For the races table, multiple sentinel_float rules are needed (HaronTime + LapTime with different sentinels/divisors). The implementation uses a list-of-dicts structure for `sentinel_float` in races:

```python
"races": {
    "sentinel_float": [
        {"columns": ["harontimel3", "harontimel4"], "sentinels": ["000", "999"]},
        {"columns": [f"laptime{i}" for i in range(1, 26)], "sentinels": ["000"], "divisor": 10},
    ],
},
```

The `_apply_type_conversions` function handles both single-dict and list-of-dicts forms.

### 3. HaronTimeL3 migration from float to sentinel_float

Remove `"harontimel3"` from `entries.float` list. Add it to `entries.sentinel_float.columns`. This is a clean migration -- the sentinel_float rule produces float64 output identical to the old float rule for non-sentinel values, but correctly handles sentinels.

### 4. LapTime1~25 format handling

EveryDB2 schema (03-RACE.md fields 68-92): LapTime1~25 are varchar(3), stored as "345" meaning 34.5 seconds. The value needs division by 10 (similar to odds10 pattern). Sentinel "000" means no data.

Two sub-options:
- **sentinel_float then divide**: After sentinel replacement, divide by 10
- **Custom sentinel_float10 rule**: Like odds10, divide by 10 after sentinel replacement

The simplest approach: use sentinel_float for NaN replacement, then add a post-processing step for the /10 division. OR create a new rule type `sentinel_odds10` that combines sentinel replacement + /10 division.

**Recommendation:** Extend sentinel_float with an optional `divisor` key:
```python
"sentinel_float": {
    "columns": [f"laptime{i}" for i in range(1, 26)],
    "sentinels": ["000"],
    "divisor": 10,
}
```

This keeps one rule type and handles both HaronTime (no divisor) and LapTime (divisor=10).

### 5. Jyuni2c/3c handling

EveryDB2 schema (04-UMA_RACE.md fields 49-50): Jyuni2c/3c are varchar(2), representing corner position. Initial value "00" means no data. Per D-08, sentinel is "000" -- but since these are varchar(2), the sentinel is actually "00". This needs verification.

Per D-08, use sentinel_float with sentinels=["000"]. If the actual sentinel in 2-char fields is "00", adjust accordingly. The implementation should use `pd.to_numeric(errors="coerce")` which naturally handles both "00" and "000" by converting them to 0.0 (before sentinel replacement).

Wait -- if the sentinel is "00", it would be replaced with NaN, and remaining numeric values like "05" (position 5) would convert correctly. But "00" is 2 characters while the decision says "000" for 3-character sentinels. The planner must verify the actual sentinel pattern from the data.

**CONFIDENCE: MEDIUM** -- the EveryDB2 schema says initial value is "0" for 2-character fields. The sentinel could be "00" or just the default "0" padded. The sentinel_float rule should handle both.

### 6. POST_RACE_COLS consolidation

Current state: All 3 locations have identical 16-column lists.

Changes:
1. `src/domain/types.py`: Add `laptime1` through `laptime25` (25 columns). Total: 41 entries.
2. `tests/test_paper_trading_guards.py`: Replace inline `POST_RACE_COLS` tuple with `from domain.types import POST_RACE_COLS`.
3. `scripts/run_paper_trading.py`: Replace inline `POST_RACE_COLS` tuple with `from domain.types import POST_RACE_COLS`.

The backtest engine (`src/backtest/engine.py`) already imports from types.py -- no change needed.

### 7. 3-layer leakage test impact

Per D-02, existing tests pass without modification because:
- **Layer 1** (build_all output): LapTime columns are race-level, not added by FeatureEngine.build_all()
- **Layer 2** (FEATURE_COLS): No model currently references laptime* columns
- **Layer 3** (EV odds): Unrelated

The test_post_race_leakage.py uses `POST_RACE_COLS` from `domain.types` -- after adding laptime1~25, these columns are automatically covered by the whitelist check.

## Common Pitfalls

### Pitfall 1: HaronTimeL3 sentinel data leakage
**What goes wrong:** If HaronTimeL3 stays in `float` rules without sentinel handling, "000" becomes 0.0 and "999" becomes 999.0 in training data. The ML model learns from these corrupted values.
**Why it happens:** The current `_to_float` function has no sentinel awareness -- it converts any parseable string to float.
**How to avoid:** Migrate harontimel3 from `float` to `sentinel_float` and verify sentinel NaN conversion in tests.
**Warning signs:** Feature distributions showing spikes at 0.0 or 999.0 for harontimel3.

### Pitfall 2: LapTime divisor omission
**What goes wrong:** LapTime values stored as "345" (34.5 sec) are converted to 345.0 instead of 34.5.
**Why it happens:** Missing the /10 division step, similar to how odds10 requires /10.
**How to avoid:** Include `divisor: 10` in the sentinel_float rule for laptime columns.
**Warning signs:** LapTime values in range 300-999 instead of 30-99.

### Pitfall 3: HaronTimeL3 double-conversion
**What goes wrong:** If harontimel3 appears in both `float` and `sentinel_float` rules, it gets processed twice.
**Why it happens:** Forgetting to remove from the `float` list when adding to `sentinel_float`.
**How to avoid:** Remove from `float`, add to `sentinel_float` only. Test verifies no double-processing.

### Pitfall 4: readers.py out of sync with etl.py
**What goes wrong:** Old Parquet files read through readers.py don't get type coercion for new columns.
**Why it happens:** Adding columns to _TABLE_TYPE_RULES but not to readers.py _INT_COLS/_FLOAT_COLS.
**How to avoid:** D-09 explicitly requires readers.py updates in the same task.

### Pitfall 5: POST_RACE_COLS import breaking test_paper_trading_guards.py
**What goes wrong:** Import from domain.types fails because the test's sys.path doesn't include `src/`.
**Why it happens:** test_paper_trading_guards.py currently defines POST_RACE_COLS locally. Changing to import requires the same path setup as test_post_race_leakage.py (which already imports from domain.types).
**How to avoid:** Verify test_post_race_leakage.py imports work (they do -- it uses `from domain.types import POST_RACE_COLS` at line 16). The same pattern applies.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sentinel value replacement | Custom if/else per column | Declarative sentinel_float rule in _TABLE_TYPE_RULES | Scalable, testable, DRY |
| Float conversion with NaN | Per-column replace+astype | pd.to_numeric(errors="coerce") after sentinel replacement | Handles edge cases (empty string, whitespace) |
| POST_RACE column registry | Scattered lists in 3 files | domain.types.POST_RACE_COLS single source | D-10 decision, prevents future drift |
| LapTime /10 division | Post-ETL calculation | divisor key in sentinel_float rule | Keeps all ETL logic in one place |

## Code Examples

### Current _to_float (no sentinel handling)

```python
# Source: src/db/etl.py lines 148-154
def _to_float(val: object) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
```

This converts "000" to 0.0 and "999" to 999.0 -- no sentinel awareness.

### Proposed sentinel_float processing block

```python
# Source: pattern derived from D-07 decision + existing _apply_type_conversions structure
sentinel_float_rule = rules.get("sentinel_float")
if isinstance(sentinel_float_rule, dict):
    cols = sentinel_float_rule.get("columns", [])
    sentinels = sentinel_float_rule.get("sentinels", [])
    divisor = sentinel_float_rule.get("divisor", 1)
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(sentinels, float("nan"))
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if divisor != 1:
                df[col] = df[col] / divisor
```

### Updated _TABLE_TYPE_RULES for entries

```python
"entries": {
    "int": [
        "umaban", "kakuteijyuni", "ninki", "kyakusitukubun",
        "jyuni1c", "jyuni4c", "zogenfugo",
    ],
    "float": ["time", "bataijyu", "zogensa", "timediff"],  # harontimel3 REMOVED
    "odds10": ["odds"],
    "sentinel_float": {
        "columns": ["harontimel3", "harontimel4", "jyuni2c", "jyuni3c"],
        "sentinels": ["000", "999"],
    },
},
```

### Updated _TABLE_TYPE_RULES for races (with RA HaronTime)

```python
"races": {
    "int": ["trackcd", "kyori", "tenkocd", "syussotosu", "honsyokin"],
    "sentinel_float": [
        # RA table HaronTimeL3/L4: race-level, sentinels 000/999, no divisor
        {"columns": ["harontimel3", "harontimel4"], "sentinels": ["000", "999"]},
        # RA table LapTime1~25: varchar(3), sentinels 000, divisor=10 (e.g., "345" = 34.5 sec)
        {"columns": [f"laptime{i}" for i in range(1, 26)], "sentinels": ["000"], "divisor": 10},
    ],
},
```

### POST_RACE_COLS update (types.py)

```python
POST_RACE_COLS: list[str] = [
    "kakuteijyuni",
    "confirmed_odds",
    "ninki",
    "kyakusitukubun",
    "time",
    "timediff",
    "harontimel3",
    "harontimel4",
    "jyuni1c",
    "jyuni2c",
    "jyuni3c",
    "jyuni4c",
    "honsyokin",
    "chakusacd",
    "dmjyuni",
    "dmtime",
    # ETL-02: LapTime1~25 (RA table, race-level POST_RACE)
    *[f"laptime{i}" for i in range(1, 26)],
]
```

### Import consolidation (test_paper_trading_guards.py)

```python
# Before (lines 5-22):
POST_RACE_COLS = (
    "kakuteijyuni",
    ...
)

# After:
from domain.types import POST_RACE_COLS
```

### readers.py updates

```python
_INT_COLS: set[str] = {
    # ... existing ...
    "jyuni2c",   # NEW: ETL-03
    "jyuni3c",   # NEW: ETL-03
}

_FLOAT_COLS: set[str] = {
    # ... existing ...
    "harontimel4",  # NEW: ETL-01
    *[f"laptime{i}" for i in range(1, 26)],  # NEW: ETL-02
}
```

## Test Strategy

### Existing test coverage

- `tests/test_etl_type_conversion.py` -- Tests _apply_type_conversions for int, float, odds10, odds100. **Need to add sentinel_float tests.**
- `tests/test_etl.py` -- Tests ETL pipeline (full load, delta merge). No sentinel coverage.
- `tests/test_post_race_leakage.py` -- 3-layer leakage detection. Uses POST_RACE_COLS from types.py.
- `tests/test_paper_trading_guards.py` -- POST_RACE drop verification. Uses local POST_RACE_COLS.

### New tests needed

1. **test_etl_type_conversion.py extensions:**
   - `test_sentinel_float_replaces_sentinels`: Input "000", "999" -> NaN, valid values preserved
   - `test_sentinel_float_with_divisor`: LapTime "345" -> 34.5, "000" -> NaN
   - `test_sentinel_float_missing_columns`: Column not in DataFrame -> no error
   - `test_sentinel_float_jyuni`: "00" -> NaN (if "00" is sentinel)

2. **test_etl_type_conversion.py migration test:**
   - `test_haron_timel3_migrated_from_float`: Verify harontimel3 NOT in regular float rules (it's in sentinel_float)

3. **test_etl_type_conversion.py RA HaronTime test:**
   - `test_races_harontime_sentinel`: RA table HaronTimeL3/L4 "000"/"999" -> NaN, valid values preserved

4. **test_paper_trading_guards.py (after import consolidation):**
   - Existing tests should pass unchanged (they use POST_RACE_COLS for drop verification)

5. **test_post_race_leakage.py:**
   - Should pass unchanged (laptime* not in build_all output, not in any FEATURE_COLS)

### Test execution commands

```bash
python -m pytest tests/test_etl_type_conversion.py -v
python -m pytest tests/test_post_race_leakage.py -v
python -m pytest tests/test_paper_trading_guards.py -v
python -m pytest tests/test_etl.py -v
```

## Implementation Order

Recommended task sequence for the planner:

1. **Task 1: Add sentinel_float/sentinel_int rule handling to _apply_type_conversions** (src/db/etl.py)
   - Extend function to process sentinel_float dict rules (single dict and list-of-dicts)
   - Add optional divisor support
   - Write unit tests for the new rule type

2. **Task 2: Update _TABLE_TYPE_RULES for entries table** (src/db/etl.py)
   - Remove harontimel3 from float list
   - Add sentinel_float rule for harontimel3, harontimel4, jyuni2c, jyuni3c
   - Write tests verifying sentinel NaN conversion for each column

3. **Task 3: Update _TABLE_TYPE_RULES for races table** (src/db/etl.py)
   - Add sentinel_float rules (list-of-dicts) for RA HaronTimeL3/L4 and LapTime1~25 with divisor=10
   - Write tests verifying both HaronTime and LapTime conversion

4. **Task 4: Update readers.py _INT_COLS/_FLOAT_COLS** (src/db/readers.py)
   - Add jyuni2c, jyuni3c to _INT_COLS
   - Add harontimel4, laptime1~25 to _FLOAT_COLS
   - Write test for coerce_types with new columns

5. **Task 5: Consolidate POST_RACE_COLS** (types.py + consumers)
   - Add laptime1~25 to domain/types.py POST_RACE_COLS
   - Replace inline definitions in test_paper_trading_guards.py with import
   - Replace inline definition in scripts/run_paper_trading.py with import
   - Run all leakage tests to verify

6. **Task 6: Run full test suite and verify**
   - `python -m pytest tests/ -v`
   - Verify no regression in existing tests

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| HaronTimeL3 sentinel data already in Parquet | HIGH (existing data corrupted) | MEDIUM (training on bad data) | Full ETL re-run required after code change |
| LapTime "000" not the only sentinel | LOW (schema docs say 0=initial) | LOW (extra sentinels remain as 0.0) | ETL quality verification catches this |
| Jyuni2c/3c sentinel value is "00" not "000" | MEDIUM (2-char vs 3-char field) | LOW (both handled by string match) | Use set of sentinels: ["000", "00"] |
| HaronTimeL4 and HaronTimeL3 both valid for same horse | MEDIUM (documented in schema) | INFO (expected per D-05, Phase 36 handles) | Document distribution in ETL-05 |
| POST_RACE_COLS expansion breaks test_paper_trading_guards | LOW (import consolidation is clean) | LOW (test fix) | Verify import works before removing inline def |
| RA HaronTimeL3/L4 remain unhandled in races | MEDIUM (SELECT * includes them) | MEDIUM (varchar in Parquet) | Add to races sentinel_float rules (Plan 35-01 Task 1) |

## Open Questions (RESOLVED)

1. **Jyuni2c/3c actual sentinel values** -- RESOLVED: Use sentinels=["000", "999", "00"] in the entries sentinel_float rule to cover all padding variants (Plan 35-01 Task 1). The ETL quality check (D-03) will verify actual sentinel patterns in the data.

2. **HaronTimeL3/L4 in RA table vs SE table** -- RESOLVED: RA table HaronTimeL3/L4 ARE included in races Parquet (SELECT * extracts all columns). Plan 35-01 Task 1 adds them to the races sentinel_float rules alongside LapTime with sentinels=["000", "999"] and no divisor (race-level values are already in final units).

3. **Existing Parquet data quality** -- RESOLVED: Full ETL re-run (`--mode full`) is required after code changes to fix historical data. Plan 35-02 Task 2 documents the ETL quality verification procedure (D-03) including specific Parquet inspection commands and expected dtypes/ranges.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Inline float rules only | sentinel_float declarative rules | This phase | Sentinel values properly NaN-handled |
| POST_RACE_COLS in 3 files | Single source in types.py | This phase | DRY, prevents future drift |
| No LapTime in Parquet | LapTime1~25 as float64 | This phase | Phase 36 can compute pace features |
| RA HaronTime as varchar | RA HaronTimeL3/L4 as float64 | This phase | Race-level harontime available for features |

**Deprecated/outdated:**
- HaronTimeL3 in `entries.float` rule: Will be replaced by `entries.sentinel_float` rule.

## Sources

### Primary (HIGH confidence)
- `src/db/etl.py` -- Full file read: _TABLE_TYPE_RULES, _apply_type_conversions, all ETL functions
- `src/db/readers.py` -- Full file read: _INT_COLS, _FLOAT_COLS, coerce_types
- `src/domain/types.py` -- Full file read: POST_RACE_COLS definition
- `docs/everydb2/04-UMA_RACE.md` -- SE table schema: HaronTimeL3/L4 sentinels documented
- `docs/everydb2/03-RACE.md` -- RA table schema: LapTime1~25 format (varchar(3), "345"=34.5sec), HaronTimeL3/L4 (fields 96-97)
- `tests/test_etl_type_conversion.py` -- Existing test patterns
- `tests/test_post_race_leakage.py` -- 3-layer leakage test structure
- `tests/test_paper_trading_guards.py` -- POST_RACE_COLS duplicate definition

### Secondary (MEDIUM confidence)
- `tests/test_etl.py` -- ETL pipeline test patterns
- `scripts/run_paper_trading.py` -- POST_RACE_COLS duplicate definition
- `config/etl_tables.yaml` -- Table definitions (no changes needed)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all code read directly, schema docs verified
- Architecture: HIGH -- existing patterns are clear, extension points well-defined
- Pitfalls: HIGH -- sentinel handling gap verified by reading _to_float implementation

**Research date:** 2026-05-19
**Valid until:** 30 days (stable codebase, no external dependencies)
