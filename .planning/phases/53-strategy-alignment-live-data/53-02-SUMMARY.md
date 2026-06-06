---
phase: 53-strategy-alignment-live-data
plan: 02
subsystem: ingestion
tags: [tdd, protocol-based-di, playwright, html-parser, track-condition]
dependency_graph:
  requires: [beautifulsoup4, playwright]
  provides: [TrackConditionFetcherProtocol, JRATrackConditionFetcher, parse_track_condition_html]
  affects: [src/ingestion/track_condition_fetcher.py, tests/test_track_condition_fetcher.py]
tech_stack:
  added: [beautifulsoup4 (pre-existing)]
  patterns: [Protocol-based DI, pure function parser, hash-based structure change detection]
key_files:
  created:
    - src/ingestion/track_condition_fetcher.py
    - tests/test_track_condition_fetcher.py
  modified: []
decisions:
  - D-05 に基づき Playwright 取得と BeautifulSoup 解析を完全分離
  - 必須 DOM 要素 (#turf_line, #dirt_line) 欠落時は即座に TrackConditionParseError を送出
  - 取得失敗時はフォールバックせず例外を送出 → 予測停止
  - html_hash (SHA256) で HTML 構造変更を検知可能
metrics:
  duration: 5m 14s
  completed: 2026-06-06T07:55:08Z
  tasks: 1
  files: 2
  tests: 25
---

# Phase 53 Plan 02: TrackConditionFetcher + HTML Parser Summary

JRA 馬場情報 HTML 取得 (Playwright) と純粋関数パーサー (BeautifulSoup) を完全分離し、Protocol-based DI でテスト可能な TrackConditionFetcher を実装。

## One-liner

TrackConditionFetcherProtocol + JRATrackConditionFetcher (Playwright) + parse_track_condition_html() pure function with SHA256 structure change detection

## Tasks Completed

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | TrackConditionFetcherProtocol + JRATrackConditionFetcher + pure function parser | a82b3cc, 4ae1b02, 182f48c | DONE (TDD) |

## TDD Gate Compliance

| Gate | Commit | Description |
|------|--------|-------------|
| RED | a82b3cc | 25 failing tests (module not yet created) |
| GREEN | 4ae1b02 | Implementation: all 25 tests passing |
| REFACTOR | 182f48c | ruff format whitespace cleanup |

All three TDD gates present and in correct order.

## What Was Built

### src/ingestion/track_condition_fetcher.py (NEW)

- **TrackConditionParseError** -- Exception for missing DOM elements, empty HTML, parse failures
- **_parse_percent(text)** -- `"16.2%"` to `16.2` helper; returns `None` on failure
- **parse_track_condition_html(html)** -- Pure function (no Playwright dependency):
  - Extracts `turf_cushion` from `#cushion_num strong` (optional -- `None` if absent)
  - Extracts `dirt_moisture_goal`, `dirt_moisture_4c` from `#dirt_line .gm` / `.c4`
  - Extracts `turf_moisture_goal`, `turf_moisture_4c` from `#turf_line .gm` / `.c4`
  - Extracts `measured_at_moist` from `#moist_list option[selected]`
  - Extracts `measured_at_cushion` from `#cushion_list option[selected]`
  - Computes `html_hash` (SHA256) for structure change detection
  - Raises `TrackConditionParseError` if `#turf_line` or `#dirt_line` missing
- **TrackConditionFetcherProtocol** -- `runtime_checkable` Protocol with `fetch_track_conditions_html(venue_code: str) -> str`
- **JRATrackConditionFetcher** -- Playwright `sync_api` implementation:
  - Fetches HTML from `https://www.jra.go.jp/keiba/baba/`
  - Venue tab click for same-page DOM switching
  - `fetch_all_venues(track_date)` for batch retrieval
  - No fallback on failure (raises exception -> prediction halt, D-05)
- **_detect_html_structure_change(reference_hash, current_hash)** -- Hash comparison for structure change detection

### tests/test_track_condition_fetcher.py (NEW)

- 25 tests across 11 test classes:
  - TestParseValidHtml (5 tests): all values from full HTML fixture
  - TestParseCushionMissing (2 tests): `turf_cushion=None` graceful handling
  - TestParseTurfLineMissing (1 test): TrackConditionParseError
  - TestParseDirtLineMissing (1 test): TrackConditionParseError
  - TestParseBothLinesMissing (1 test): TrackConditionParseError
  - TestParsePercent (4 tests): normal, empty, no-%, invalid
  - TestHtmlHash (3 tests): presence, SHA256 correctness, format
  - TestMeasuredAtExtraction (3 tests): moist, cushion, missing
  - TestProtocolConformance (2 tests): isinstance checks
  - TestParseEmptyHtml (1 test): empty string raises error
  - TestDetectHtmlStructureChange (2 tests): same/different hash

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

```
python -m pytest tests/test_track_condition_fetcher.py -v: 25 passed
ruff check src/ingestion/track_condition_fetcher.py: All checks passed
ruff check tests/test_track_condition_fetcher.py: All checks passed
PYTHONPATH=src python -c "from ingestion.track_condition_fetcher import ...": Import OK
```

## Known Stubs

None -- all functionality is fully implemented and tested.

## Threat Flags

None -- no new security surface beyond what is documented in the plan's threat_model (T-53-04, T-53-05, T-53-06, T-53-SC).
