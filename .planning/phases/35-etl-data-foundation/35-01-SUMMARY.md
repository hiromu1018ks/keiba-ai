---
phase: 35-etl-data-foundation
plan: 01
subsystem: etl
tags: [sentinel, type-conversion, etl, harontime, laptime, jyuni]
dependency_graph:
  requires: []
  provides: [sentinel_float-rule, sentinel_int-rule, entries-sentinel-conversion, races-sentinel-conversion, readers-backward-compat]
  affects: [src/db/etl.py, src/db/readers.py]
tech_stack:
  added: []
  patterns: [declarative-sentinel-rules, sentinel_float-list-of-dicts, divisor-key]
key_files:
  created: []
  modified:
    - src/db/etl.py
    - src/db/readers.py
    - tests/test_etl_type_conversion.py
decisions:
  - sentinel_float rules use dict structure with columns/sentinels/divisor keys
  - races sentinel_float uses list-of-dicts to support multiple rule sets (HaronTime + LapTime)
  - HaronTimeL3 migrated from entries.float to entries.sentinel_float to prevent double-processing
  - Jyuni2c/3c use sentinels ["000", "999", "00"] to cover both 2-char and 3-char padding variants
  - RA table HaronTimeL3/L4 added to races sentinel_float with no divisor (race-level values already in final units)
  - LapTime1~25 added to races sentinel_float with divisor=10 (varchar(3) "345" = 34.5 sec)
metrics:
  duration: 6min
  completed: 2026-05-19
  tasks_completed: 2
  tests_added: 8
  files_modified: 3
---

# Phase 35 Plan 01: Sentinel Float Rules Summary

ETLエンジンにsentinel_float/sentinel_intルールタイプを追加し、HaronTimeL3/L4、LapTime1~25、Jyuni2c/3cのセンチネル値(000/999/00)をNaN化してfloat64変換する仕組みを実装。既存のHaronTimeL3がfloatルールからsentinel_floatへ移行され、000/999が0.0/999.0として不正確に格納される問題を修正。

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | sentinel_float/sentinel_intルール追加と_TABLE_TYPE_RULES更新 | 066e55e + cc37656 | src/db/etl.py, tests/test_etl_type_conversion.py |
| 2 | readers.py _INT_COLS/_FLOAT_COLS更新 | cc37656 | src/db/readers.py |

## Key Changes

### src/db/etl.py
- `_TABLE_TYPE_RULES` 型アノテーション変更: `dict[str, dict[str, list[str]]]` -> `dict[str, dict[str, list[str] | dict | list[dict]]]`
- entries.sentinel_float 追加: harontimel3, harontimel4, jyuni2c, jyuni3c (sentinels: 000, 999, 00)
- races.sentinel_float 追加 (list-of-dicts):
  - HaronTimeL3/L4: sentinels [000, 999], no divisor
  - LapTime1~25: sentinels [000], divisor=10
- `_apply_type_conversions`: sentinel_float/sentinel_int 処理ブロック追加 (replace -> to_numeric -> optional divisor)
- entries.float から harontimel3 を削除 (sentinel_float に移行)

### src/db/readers.py
- _FLOAT_COLS: harontimel4 + laptime1~25 を追加
- _INT_COLS: jyuni2c, jyuni3c を追加

### tests/test_etl_type_conversion.py
- TestSentinelRules: 6テスト (sentinel置換, divisor, 欠損列, float移行確認, 二重処理防止, RA HaronTime)
- TestReadersCompat: 2テスト (_FLOAT_COLS/_INT_COLS 新列確認)

## TDD Gate Compliance

- RED commit: 066e55e `test(35-01): add failing sentinel rule tests` (6 failed, 12 passed)
- GREEN commit: cc37656 `feat(35-01): implement sentinel_float rules + readers.py updates` (18 passed, 0 failed)
- REFACTOR: Not needed (clean implementation)

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None. All new surface is internal ETL type conversion with no network endpoints or auth paths.
