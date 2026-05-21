---
phase: 11-bet-selection-filters
verified: 2026-05-04T15:25:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
gaps: []
    missing:
      - "get_win_candidates() から EV 除外件数を返す仕組み (戻り値タプル化、またはcandidate_dfに除外数メタデータ付与)"
      - "engine.py のレースループ内で n_ev_excluded をインクリメント"
---

# Phase 11: Bet Selection Filters Verification Report

**Phase Goal:** 低信頼ベット・不安定レジーム・赤字オッズバンドを自動除外し、バックテストのベット品質が向上する
**Verified:** 2026-05-04T15:25:00Z
**Status:** passed
**Re-verification:** Yes (gap fixed: n_ev_excluded counter propagation)

## Goal Achievement

### Observable Truths

Derived from ROADMAP Success Criteria + PLAN frontmatter must_haves (merged, deduplicated):

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | get_win_candidates() は EV_lower_win_corrected < 1.0 の候補を除外する | VERIFIED | `src/backtest/race_predictor.py` lines 433-450: `ev_mask = ev_lower.fillna(1.0) >= 1.0; mask &= ev_mask`. 4 tests pass (excludes below, NaN fallback, column missing, above kept). |
| 2 | EV_lower_win_corrected が NaN の場合、既存 edge>0 のみで判定 (フォールバック) | VERIFIED | `race_predictor.py` line 439: `fillna(1.0) >= 1.0` passes NaN through. Test 9 confirms both NaN candidates pass. |
| 3 | OddsBandFilter はトレーニング期間ROI < 100% のバンドを除外する | VERIFIED | `src/betting/odds_band_filter.py` lines 37-75: `calibrate()` computes ROI per band, excludes if `roi < 1.0`. 6 tests pass. |
| 4 | RegimeDetector.get_strategy_params() が COLLAPSED 時に skip=True を返す | VERIFIED | `src/models/regime_detector.py` line 231: `"skip": True` in COLLAPSED branch. Test confirms skip=True for COLLAPSED and skip not True for AGGRESSIVE. |
| 5 | 除外件数が INFO レベルでログ出力される | VERIFIED | EV filter logging: `race_predictor.py` lines 442-450 logs exclusion count. EV exclusion count propagated via `DataFrame.attrs["n_ev_excluded"]` to engine.py (line 749). OddsBandFilter: `odds_band_filter.py` lines 70-75 (calibration) and 93-98 (filter). Engine: `engine.py` logs filter summary. All three filter types have correct count propagation. |
| 6 | COLLAPSED レジームのレースはベットが完全スキップされ、スキップ件数がカウントされる | VERIFIED | `engine.py` lines 741-744: `if regime_params.get("skip", False): n_collapsed_skipped += 1; continue`. Lines 713-739: recent_stats_list.append() happens BEFORE skip (Pitfall 3). Test confirms n_collapsed_skipped >= 1 and total_bets == 0. |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `src/betting/odds_band_filter.py` | OddsBandFilter class (calibrate + filter + excluded_bands) | VERIFIED | 109 lines. Class with BANDS, BAND_NAMES, calibrate(), filter(), excluded_bands property. All 6 tests pass. |
| `src/backtest/race_predictor.py` | get_win_candidates() に EV_lower >= 1.0 filter added | VERIFIED | Lines 433-450: EV_lower_win_corrected filter with fillna(1.0) fallback. 4 tests pass. |
| `src/models/regime_detector.py` | COLLAPSED branch with skip=True | VERIFIED | Line 231: `"skip": True` in COLLAPSED return dict. 2 tests pass. |
| `src/backtest/engine.py` | COLLAPSED skip + OddsBandFilter integration + counters + bet count guard | VERIFIED | Lines 358-361: OddsBandFilter init for win. Lines 379: training_bet_history param. Lines 617-620: counters. Lines 623-624: calibrate. Lines 741-744: COLLAPSED skip. Line 749: n_ev_excluded from attrs. Lines 762-768: OddsBandFilter.filter(). Lines 1064-1079: bet count guard. Lines 1081-1110: BacktestResult with exclusion fields. |
| `src/backtest/report.py` | exclusion stats in generate() + save_ai_diagnostics() | VERIFIED | Lines 50-57: exclusion_stats in generate(). Lines 197-207: "exclusion" section in save_ai_diagnostics(). 3 tests pass. |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| engine.py | odds_band_filter.py | `OddsBandFilter` import + calibrate() + filter() | WIRED | Line 18: import. Line 361: init. Line 624: calibrate(). Line 767: filter(). |
| engine.py | regime_detector.py | `regime_params.get('skip', False)` check | WIRED | Line 742: `if regime_params.get("skip", False)`. regime_detector.get_strategy_params() returns skip=True for COLLAPSED. |
| report.py | engine.py | BacktestResult exclusion fields | WIRED | report.py reads result.n_collapsed_skipped, n_ev_excluded, n_odds_band_excluded, exclusion_stats. |
| race_predictor.py | EV_lower_win_corrected column | fillna(1.0) mask in get_win_candidates() | WIRED | Lines 433-450: Column existence check + fillna(1.0) >= 1.0 mask. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| engine.py run() | n_collapsed_skipped | regime_params skip check in race loop | Yes - incremented on COLLAPSED skip | FLOWING |
| engine.py run() | n_odds_band_excluded | len diff before/after OddsBandFilter.filter() | Yes - computed from actual filter | FLOWING |
| engine.py run() | n_ev_excluded | DataFrame.attrs from get_win_candidates() | Yes - read from attrs per race | FLOWING |
| odds_band_filter.py | excluded_bands | calibrate() from training_bet_history | Yes - ROI computed from real bet data | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| OddsBandFilter tests | `python -m pytest tests/test_odds_band_filter.py -v` | 6/6 passed | PASS |
| EV filter tests | `python -m pytest tests/test_race_predictor.py -v -k "ev_lower"` | 4/4 passed | PASS |
| RegimeDetector skip tests | `python -m pytest tests/test_regime_detector.py -v -k "skip"` | 2/2 passed | PASS |
| Engine filter tests | `python -m pytest tests/test_backtest_engine.py::TestBetSelectionFilters -v` | 6/6 passed | PASS |
| Report exclusion tests | `python -m pytest tests/test_backtest_report.py::TestExclusionStatsReporting -v` | 3/3 passed | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| BSEL-01 | 11-01, 11-02 | EV_lower_win_corrected >= 1.0 でベット自動除外、除外件数ログ/レポート出力 | SATISFIED | race_predictor.py EV filter works + logs. EV exclusion count propagated via DataFrame.attrs to engine.py n_ev_excluded counter. Report output correct. |
| BSEL-02 | 11-02 | COLLAPSED レースでベット完全スキップ、スキップ件数レポート記録 | SATISFIED | engine.py COLLAPSED skip + n_collapsed_skipped counter + report output. Test confirms. |
| BSEL-03 | 11-01, 11-02 | 赤字オッズバンド除外 via OddsBandFilter、除外バンド/件数レポート出力 | SATISFIED | OddsBandFilter calibrate+filter in engine. Report shows excluded_bands. |

Note: REQUIREMENTS.md shows BSEL-02 as "Pending" -- this is stale and should be updated to "Complete".

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| src/backtest/engine.py | 256-257 | Unused variables kumi5, kumi5_len | Info | Pre-existing (not Phase 11 code) |
| src/backtest/engine.py | 350, 822, 824 | E501 line too long | Info | Pre-existing (not Phase 11 code) |
| src/backtest/race_predictor.py | 18 | F401 unused import build_win_selection_ev | Info | Pre-existing (noted in 11-01-SUMMARY) |

No anti-patterns introduced in Phase 11 code. All ruff errors are pre-existing.

### Human Verification Required

### 1. EV exclusion count reporting accuracy

**Test:** Run a backtest with known EV_lower values and verify the report shows accurate exclusion counts (not 0).
**Expected:** `BacktestResult.n_ev_excluded` should be > 0 when EV filter excludes candidates.
**Why human:** Requires running actual backtest with real data; automated tests mock the EV filter at unit level but do not verify end-to-end count propagation.

### Gaps Summary

**Gap 1: n_ev_excluded counter disconnected (partial truth #5)**

The EV lower bound filter in `get_win_candidates()` correctly excludes candidates and logs the count at INFO level within `race_predictor.py`. However, this exclusion count is not propagated back to `engine.py`. The `n_ev_excluded` counter in `engine.py` (line 618) is initialized to 0 and never incremented. Consequently:

- `BacktestResult.n_ev_excluded` is always 0
- `exclusion_stats["ev_excluded"]` in the report is always 0
- `save_ai_diagnostics()` shows `"ev_excluded": 0`

The logging within `race_predictor.py` does work (EV filter exclusions are visible in logs), but the structured reporting is inaccurate.

**Root cause:** `get_win_candidates()` returns only a filtered `pd.DataFrame` without metadata about how many candidates were excluded by the EV filter. Engine has no way to know the EV exclusion count.

**Fix needed:** Either return exclusion count from `get_win_candidates()` (e.g., as a tuple or via metadata), or move the EV filtering to `engine.py` where it can be counted.

---

_Verified: 2026-05-04T15:10:00Z_
_Verifier: Claude (gsd-verifier)_
