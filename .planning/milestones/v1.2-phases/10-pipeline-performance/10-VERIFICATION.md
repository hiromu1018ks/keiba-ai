---
phase: 10-pipeline-performance
verified: 2026-05-04T12:00:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 10: Pipeline Performance Verification Report

**Phase Goal:** バックテスト・学習パイプラインの実行時間が短縮され、ボトルネックが定量測定可能になる
**Verified:** 2026-05-04T12:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | build_payout_map()/build_wide_payout_map()のiterrows()がベクトル化pandas操作に置き換わり、マップ構築が高速化される | VERIFIED | engine.py: iterrows() count = 0. build_payout_map uses melt (L116-121) + groupby (L133). build_wide_payout_map uses str.len() (L203) + str.slice() (14 calls). build_win_payout_map uses set_index (L161-163). final_odds_map uses set_index+items (L433). |
| 2 | レースごとのDataFrameフィルタリングがgroupby辞書の前処理に置き換わり、O(1)ルックアップでレースデータを取得できる | VERIFIED | engine.py: build_race_groups() defined at L296. Called 5 times at L566-570. Race loop uses feat_groups.get(race_id) at L586, hist_groups.get(race_id) at L632, jockey_groups/trainer_groups/jt_groups.get at L633-635. str(race_id) cast at L585 ensures type consistency with str-keyed dicts. |
| 3 | HorseHistoryFeatures等の履歴特徴量がParquetキャッシュされ、バックテスト再実行時にキャッシュヒットすれば再計算をスキップできる | VERIFIED | feature_engine.py: compute_cache_key() at L36 using SHA-256 (16 hex chars). is_cache_valid() at L54 using hybrid timestamp check. Cache HIT path at L189 reads from store.read(). Cache MISS path at L197 proceeds to computation. Cache WRITE at L294-298 via single-return-point pattern. isinstance(store, ParquetStore) guard at L160. TestFeatureCache: 10 test methods all passing. |
| 4 | pyinstrumentプロファイリングを統合し、バックテスト実行時のボトルネック関数と所要時間を定量測定できる | VERIFIED | profiling.py: ProfileContext class at L12 with lazy pyinstrument import in __enter__ (L31). Graceful ImportError degradation (L35-40). --profile flag on run_backtest.py (L98-102) wired to ProfileContext at L590-592. --profile flag on run_wf_validation.py (L111-117). Both --help outputs confirmed. ProfileContext import verified. No overhead when enabled=False (_profiler stays None). |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/backtest/engine.py` | Vectorized payout maps + groupby dict lookups | VERIFIED | 0 iterrows(), melt at L116/119/183/186, str.len/slice at 14 locations, build_race_groups at L296, 5 groupby dicts at L566-570, dict.get() at L586/632-635, itertuples at L617/721/813/1075, nsmallest preserved at L615, set_index at L161/433 |
| `tests/test_backtest_engine.py` | Vectorization regression tests | VERIFIED | TestVectorizedPayoutMaps class with 4 test methods (L1360-1470). All 45 tests pass. |
| `src/utils/profiling.py` | ProfileContext context manager | VERIFIED | 55 lines. class ProfileContext with __enter__/__exit__, lazy pyinstrument import, HTML+text output to data/profiles/ |
| `src/features/feature_engine.py` | Feature cache with hybrid invalidation | VERIFIED | compute_cache_key (L36), is_cache_valid (L54), cache HIT/MISS/SAVED log messages, single-return-point pattern, store.read/write integration |
| `scripts/run_backtest.py` | --profile CLI flag | VERIFIED | --profile at L98-102, ProfileContext import at L590, wrapping at L592 |
| `scripts/run_wf_validation.py` | --profile CLI flag | VERIFIED | --profile at L111, ProfileContext import at L115, wrapping at L117 |
| `tests/test_feature_engine.py` | Cache tests | VERIFIED | TestFeatureCache class with 10 test methods (L613-767). All 40 tests pass. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| engine.py::build_payout_map | pandas melt + groupby | melt(id_vars, value_vars) -> dropna -> groupby('race_id') | WIRED | 2 melts at L116/119, groupby+idxmax at L133 |
| engine.py::build_wide_payout_map | pandas str vectorized ops | str.len() + str.slice() for kumi parsing | WIRED | str.len() at L203, str.slice() at L212/219/226/232/238/253/256/263/265 |
| engine.py::run | build_race_groups() | groupby dict lookup replacing per-race filtering | WIRED | 5 calls at L566-570, dict.get() at L586/632-635 |
| feature_engine.py::build_all | data/features/cache/ | compute_cache_key -> is_cache_valid -> store.read/write | WIRED | Cache check at L157-197, cache write at L294-298 |
| run_backtest.py::main | profiling.py::ProfileContext | with ProfileContext(enabled=args.profile) | WIRED | Import at L590, context at L592 |
| run_wf_validation.py::main | profiling.py::ProfileContext | with ProfileContext(enabled=args.profile) | WIRED | Import at L115, context at L117 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| engine.py::build_payout_map | payout_map | melt+groupby on payouts_df | Yes -- processes real pay columns | FLOWING |
| engine.py::build_wide_payout_map | wide_payout_map | melt+str ops on payouts_df | Yes -- processes real kumi/pay columns | FLOWING |
| engine.py::run (groupby dicts) | feat_groups, hist_groups, etc. | build_race_groups(feat_df) | Yes -- groups real feature DataFrames | FLOWING |
| feature_engine.py::build_all (cache) | cached_df / result_df | ParquetStore.read/write | Yes -- real Parquet I/O with SHA-256 key | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| iterrows() count = 0 | grep -c "iterrows" src/backtest/engine.py | 0 | PASS |
| ProfileContext importable | python -c "from utils.profiling import ProfileContext" | OK (no error) | PASS |
| --profile flag in run_backtest.py | python scripts/run_backtest.py --help | Shows --profile | PASS |
| --profile flag in run_wf_validation.py | python scripts/run_wf_validation.py --help | Shows --profile | PASS |
| Backtest engine tests pass | python -m pytest tests/test_backtest_engine.py -v | 45 passed | PASS |
| Feature engine tests pass | python -m pytest tests/test_feature_engine.py -v | 40 passed | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PERF-01 | 10-01 | build_payout_map()/build_wide_payout_map()のiterrows()をベクトル化pandas操作に置き換えられる | SATISFIED | 0 iterrows() in engine.py. melt+groupby for fukushou, str.len/slice for wide kumi, set_index+items for final_odds_map. TestVectorizedPayoutMaps: 4 tests pass. |
| PERF-02 | 10-01 | レースごとのDataFrameフィルタリングをgroupby辞書の前処理に置き換え、O(n_races * n_rows) -> O(1)ルックアップにできる | SATISFIED | build_race_groups() helper at L296, 5 groupby dicts at L566-570, dict.get() lookups at L586/632-635. |
| PERF-03 | 10-02 | HorseHistoryFeatures等の履歴特徴量をParquetキャッシュし、バックテスト再実行時に再計算をスキップできる | SATISFIED | compute_cache_key (SHA-256), is_cache_valid (timestamp), HIT/MISS/SAVED paths in build_all(). TestFeatureCache: 10 tests pass. |
| PERF-04 | 10-02 | pyinstrumentによるプロファイリングを統合し、ボトルネックの定量測定ができる | SATISFIED | ProfileContext in profiling.py, --profile on both CLI scripts, lazy import + graceful degradation. |

### Anti-Patterns Found

No anti-patterns detected. No TODO/FIXME/placeholder/hack comments in modified files. No stub implementations or empty returns in critical paths.

### Human Verification Required

1. **--profile actual execution**
   **Test:** Run `python scripts/run_backtest.py --train-start 20200101 --train-end 20231231 --test-start 20240101 --test-end 20241231 --profile` with pyinstrument installed
   **Expected:** Text profile output to stdout, HTML profile written to `data/profiles/backtest.html`
   **Why human:** Requires full pipeline execution (~57 min), pyinstall install, and Parquet data availability

2. **Cache HIT verification with real data**
   **Test:** Run backtest twice with same parameters, observe "Feature cache HIT" log on second run
   **Expected:** Second run should log "Feature cache HIT" and skip feature computation
   **Why human:** Requires full pipeline execution and log observation across two runs

### Gaps Summary

No gaps found. All 4 success criteria from ROADMAP.md are met with codebase evidence:

1. **PERF-01 (vectorized payout maps):** All 7 iterrows() eliminated. melt+groupby for fukushou, str vectorized ops for wide kumi parsing, set_index+items for odds maps, itertuples for top3/diag/bets.
2. **PERF-02 (groupby dict lookups):** build_race_groups() helper with str-key conversion, logging, and memory tracking. All 5 per-race DataFrame filterings replaced with O(1) dict.get().
3. **PERF-03 (Parquet feature cache):** SHA-256 cache key (16 hex chars), hybrid timestamp invalidation, single-return-point pattern for guaranteed cache write, 10 cache tests passing.
4. **PERF-04 (pyinstrument profiling):** ProfileContext with lazy import, graceful ImportError handling, --profile flag on both CLI scripts, HTML+text output to data/profiles/.

---

_Verified: 2026-05-04T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
