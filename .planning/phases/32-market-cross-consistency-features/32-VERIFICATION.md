---
phase: 32-market-cross-consistency-features
verified: 2026-05-18T11:00:00Z
status: passed
score: 11/11 must-haves verified
overrides_applied: 0
---

# Phase 32: Market Cross-Consistency Features Verification Report

**Phase Goal:** Harville理論オッズによる馬券種クロス整合性特徴量5つを追加し、全12モデルのFEATURE_COLSに統合する
**Verified:** 2026-05-18T11:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Harville公式でワイド理論確率が計算され、実オッズ/理論オッズ比率が生成される | VERIFIED | `_harville_wide_prob()` implements both-order sum with epsilon=1e-10; `_compute_wide_harville_ratio()` computes actual_mid/theo_odds. Test `test_wide_harville_ratio` PASS. |
| 2 | Harville公式で三連複理論確率が計算され、実オッズ/理論オッズ比率が生成される | VERIFIED | `_harville_trio_prob()` implements 6-permutation sum; `_compute_trio_features()` computes actual/theo odds ratio. Test `test_trio_odds_ratio` PASS. |
| 3 | 1番人気がワイドninki=1組合せに含まれるかの0/1フラグが生成される | VERIFIED | `_check_favorite_in_wide()` returns 1.0/0.0. Tests `test_favorite_in_wide_top1` and `test_favorite_not_in_wide_top1` both PASS. |
| 4 | 三連複ninki=1組合せと単勝上位3頭のオーバーラップ数(0-3)が生成される | VERIFIED | `_compute_trio_features()` computes `len(trio_horses & top3_set)`. Tests `test_trio_overlap` (overlap=3) and `test_trio_overlap_partial` (overlap=2) PASS. |
| 5 | 1番人気が三連複ninki=1組合せに含まれるかの0/1フラグが生成される | VERIFIED | `_compute_trio_features()` returns consistency=1.0 if fav in trio_horses. Tests `test_market_consistency` and `test_market_consistency_not_included` PASS. |
| 6 | 全特徴量はpre-race tanoddsのみを使用しPOST_RACE列を一切参照しない | VERIFIED | AST scan test `test_market_cross_features_no_post_race_input` PASS -- no POST_RACE_COLS string literals in source. `test_mcf_cols_not_in_post_race` PASS -- no overlap. Source code uses only `tanodds`, `umaban`, `kumi`, `oddslow`, `oddshigh`, `ninki`, `odds`. |
| 7 | wide_df/trio_dfがNoneの場合は全MCF列がNaNとなる | VERIFIED | Lines 476-479: early return with NaN. Tests `test_none_wide_trio_single_race` and `test_none_wide_trio_multi_race` PASS. |
| 8 | build_all()がwide/trioオッズをDataRepositoryからロードし、compute_market_cross_features()に渡す | VERIFIED | `feature_engine.py` lines 350-372: imports `compute_market_cross_features`, creates `DataRepository(store)`, calls `repo.load_wide_odds()` and `repo.load_trio_odds()`, passes to `compute_market_cross_features()`. NaN fallback on exception. |
| 9 | build_features()でもmarket_cross_featuresが計算される (wide_df/trio_dfなしでNaNフォールバック) | VERIFIED | `feature_engine.py` lines 488-491: `compute_market_cross_features(df)` called without wide/trio (NaN fallback). |
| 10 | 全12モデルのFEATURE_COLSに5つのMCF特徴量が追加される | VERIFIED | All 12 models verified programmatically: AbilityModel(102), MarketModel(14), RegimeDetector(15), PlaceAbilityModel(68), RaceQualityScreener(29), WideTwoStageModel(12), WinTwoStageModel(87), PlaceTwoStageModel.HIT(90), PlaceTwoStageModel.RETURN(92), EVCorrectionModel(30), PlaceEVCorrectionModel(30), ConformalEVModel(136). All `has_all_MCF=True`. |
| 11 | POST_RACE情報漏洩テストのAST scanがcompute_market_cross_features()でPASSする | VERIFIED | `TestMarketCrossFeatures` 4 tests all PASS: AST scan, col overlap, build_all output, all-models-have-mcf. |

**Score:** 11/11 truths verified

### ROADMAP Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Harville公式による理論ワイドオッズが計算され、実オッズとの比率特徴量が生成される | VERIFIED | `_harville_wide_prob()` + `rl_wide_harville_ratio` in output |
| 2 | 5つの市場クロス整合性特徴量がFEATURE_COLSに追加される | VERIFIED | All 12 models have all 5 MCF features |
| 3 | ワイドオッズmergeをbuild_all()に統合され、training/backtestでの重複コードが排除される | VERIFIED | `build_all()` calls `repo.load_wide_odds()` + `repo.load_trio_odds()` with TimingContext |
| 4 | 全特徴量がpre-race snapshot oddsのみを使用し、post-race payout oddsの情報漏洩がないことがテストで検証される | VERIFIED | 4 POST_RACE tests PASS (AST scan, col overlap, build_all output, model FEATURE_COLS) |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/features/market_cross_features.py` | compute_market_cross_features() + MCF_COLS | VERIFIED | ~495 lines, 5 features, Harville engine, single/multi-race branching |
| `src/db/repository.py` | DataRepository.load_wide_odds() | VERIFIED | Method at line 64, reads "odds_wide" via ParquetStore |
| `tests/test_market_cross_features.py` | 16 unit tests | VERIFIED | 16/16 PASS in 2.51s |
| `src/features/feature_engine.py` | build_all() + build_features() MCF integration | VERIFIED | build_all: lines 350-372, build_features: lines 488-491 |
| `src/models/*.py` (10 files) | 12 FEATURE_COLS updated with 5 MCF | VERIFIED | All 12 model lists contain all 5 MCF features |
| `tests/test_post_race_leakage.py` | TestMarketCrossFeatures class | VERIFIED | 4/4 tests PASS |
| `data/feature_freeze_manifest.json` | Updated SHA256 for all 12 models | VERIFIED | AbilityModel=102, ConformalEVModel=136 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| feature_engine.py | market_cross_features.py | `from features.market_cross_features import compute_market_cross_features` | WIRED | Import + call in both build_all() and build_features() |
| feature_engine.py | repository.py | `repo.load_wide_odds(start_str, end_str)` | WIRED | DataRepository(store) created, load_wide_odds + load_trio_odds called |
| feature_engine.py | repository.py | `repo.load_trio_odds(start_str, end_str)` | WIRED | Same repo instance, load_trio_odds called |
| market_cross_features.py | repository.py | wide_df/trio_df parameter passing | WIRED | DataRepository provides DataFrames passed as function arguments |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| MCF unit tests (16) | `python -m pytest tests/test_market_cross_features.py -v` | 16 passed in 2.51s | PASS |
| POST_RACE leakage tests (4) | `python -m pytest tests/test_post_race_leakage.py -k "MarketCross" -v` | 4 passed, 7 deselected in 3.17s | PASS |
| MCF_COLS count | `python -c "from features.market_cross_features import MCF_COLS; print(len(MCF_COLS))"` | 5 | PASS |
| load_wide_odds exists | `python -c "from db.repository import DataRepository; print(hasattr(DataRepository, 'load_wide_odds'))"` | True | PASS |
| Manifest feature count | `python -c "import json; m=json.load(open('data/feature_freeze_manifest.json')); print(m['models']['AbilityModel']['feature_count'])"` | 102 | PASS |

### Probe Execution

Step 7c: SKIPPED (no probe scripts declared or conventionally expected for this phase type)

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| MCF-01 | 32-01 | Harville公式による理論ワイドオッズ計算機能 | SATISFIED | `_harville_wide_prob()` + `rl_wide_harville_ratio` |
| MCF-02 | 32-01 | rl_favorite_in_wide_top1 (0/1) | SATISFIED | `_check_favorite_in_wide()` |
| MCF-03 | 32-01 | rl_trio_overlap (0-3) | SATISFIED | `_compute_trio_features()` overlap calculation |
| MCF-04 | 32-01 | rl_market_consistency (0/1) | SATISFIED | `_compute_trio_features()` consistency check |
| MCF-05 | 32-01 | rl_trio_odds_ratio | SATISFIED | `_compute_trio_features()` Harville trio ratio |
| MCF-06 | 32-01 | rl_wide_harville_ratio | SATISFIED | `_compute_wide_harville_ratio()` |
| MCF-07 | 32-02 | build_all()へのワイドオッズmerge統合 | SATISFIED | feature_engine.py build_all() + build_features() integration |

No orphaned requirements found. All 7 MCF requirements mapped to Phase 32 are covered.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER found in modified files |

### Human Verification Required

No human verification items identified. All truths are programmatically verifiable:
- Feature computations validated by unit tests with numeric assertions
- POST_RACE safety validated by AST scan tests
- FEATURE_COLS integration validated by programmatic model inspection

---

_Verified: 2026-05-18T11:00:00Z_
_Verifier: Claude (gsd-verifier)_
