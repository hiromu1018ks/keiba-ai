---
phase: 31-race-level-aggregation-features
verified: 2026-05-18T08:30:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 31: Race-Level Aggregation Features Verification Report

**Phase Goal:** レース全体の市場構造を表す6特徴量が追加され、既存の未登録特徴量2つがFEATURE_COLSに昇格し、train/inference両パスで同じ特徴量が計算される
**Verified:** 2026-05-18
**Status:** PASSED
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | compute_race_level_features() が 6つの rl_* 特徴量を計算し DataFrame に追加して返す | VERIFIED | `src/features/race_level_features.py` L274-306: compute_race_level_features() returns df with 6 RL_COLS. Tests: 8/8 passed including value assertions for entropy, std, top3_gap, top1_odds, rank_gap, n_horses |
| 2 | tanodds が欠損の場合でも NaN フォールバックでエラーにならない | VERIFIED | `race_level_features.py` L294-297: "tanodds" not in df.columns -> NaN init. L300: `pd.to_numeric(..., errors="coerce").replace(0, np.nan)`. Test 2 (test_no_tanodds_returns_nan) + Test 3 (test_tanodds_with_zero_and_nan) both pass |
| 3 | POST_RACE_COLS に含まれる列を一切使用しない | VERIFIED | AST-based test (test_race_level_features_no_post_race_input) parses source and verifies no POST_RACE string literals. Also test_no_post_race_columns_used: with/without POST_RACE cols produce identical results. test_rl_feature_cols_not_in_post_race: RL_COLS intersection with POST_RACE_COLS is empty |
| 4 | implied_prob_hhi が全12モデルの FEATURE_COLS に含まれる | VERIFIED | Runtime check: all 12 models have `implied_prob_hhi` in FEATURE_COLS. grep confirms presence in all model files |
| 5 | odds_skewness が全12モデルの FEATURE_COLS に含まれる | VERIFIED | Runtime check: all 12 models have `odds_skewness` in FEATURE_COLS. grep confirms presence in all model files |
| 6 | FEATURE_COLS 内に重複する列名が存在しない | VERIFIED | Runtime check: `len(cols) == len(set(cols))` passes for all 12 models |

**Score:** 6/6 truths verified

### ROADMAP Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | 6つのrace-level特徴量がbuild_all()とbuild_features()の両方で計算される | VERIFIED | build_all: L345-348 (TimingContext). build_features: L460-462 (RLF-07 parity comment). Both call compute_race_level_features(). Test test_build_all_produces_rl_features confirms 6 rl_* columns in output |
| 2 | implied_prob_hhi と odds_skewness がFEATURE_COLSに昇格し、SHA256 manifestが更新される | VERIFIED | All 12 models have both features. Manifest: data/feature_freeze_manifest.json updated 2026-05-18T07:30:41, 12 models with correct feature counts (e.g. AbilityModel: 97=95+2) |
| 3 | 全テストが通過し、新特徴量がPOST_RACE情報漏洩テストで安全であることが確認される | VERIFIED | 15/15 tests passed (8 race_level + 7 post_race_leakage). Includes Layer 1 (build_all output), Layer 2 (FEATURE_COLS whitelist), AST source analysis, and rl_* presence check |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/features/race_level_features.py` | compute_race_level_features() with 6 rl_* features | VERIFIED | 307 lines, substantive implementation with _compute_for_single_race/multi_race, RL_COLS constant exported |
| `tests/test_race_level_features.py` | 8 unit tests | VERIFIED | 8 tests, all pass. Covers 3-horse, no-tanodds, zero/NaN, 2-horse, no-race-id, POST_RACE safety, multi-race, immutability |
| `src/features/feature_engine.py` | build_all + build_features integration | VERIFIED | L345-348 (build_all with TimingContext), L460-462 (build_features). Inserted after difficulty_score / _map_basic_features respectively |
| `tests/test_post_race_leakage.py` | Extended with TestRaceLevelFeatures class | VERIFIED | 3 new tests added (AST analysis, RL_COLS vs POST_RACE, build_all output check). Total 7 tests pass |
| `data/feature_freeze_manifest.json` | Updated SHA256 for 12 models | VERIFIED | 2026-05-18 timestamp, 12 models, feature counts updated (e.g. AbilityModel: 97, MarketModel: 9, RegimeDetector: 10) |
| `src/models/stage1_ability_model.py` | implied_prob_hhi + odds_skewness added | VERIFIED | L149-150. Docstring updated: Rule 1 exception for market structure indicators (D-06) |
| `src/models/market_model.py` | implied_prob_hhi + odds_skewness added | VERIFIED | L33-34 |
| `src/models/regime_detector.py` | implied_prob_hhi + odds_skewness added | VERIFIED | L63-64 |
| `src/models/place_ability_model.py` | implied_prob_hhi + odds_skewness added | VERIFIED | L103-104 |
| `src/models/race_quality_screener.py` | implied_prob_hhi + odds_skewness added | VERIFIED | L55-56 |
| `src/models/wide_two_stage_model.py` | implied_prob_hhi + odds_skewness added | VERIFIED | L51-52 |
| `src/models/two_stage_return_model.py` | implied_prob_hhi added to Win/Place models | VERIFIED | L157 (Win), L451 (Place.HIT), L568 (Place.RETURN). odds_skewness already present |
| `src/models/ev_correction_model.py` | odds_skewness added to EVC/PlaceEVC | VERIFIED | L165 (EVC), L420 (PlaceEVC). implied_prob_hhi already present |
| `src/models/conformal_ev_model.py` | No changes needed | VERIFIED | L123: both already present. Confirmed unchanged |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `feature_engine.py::build_all()` | `race_level_features.py::compute_race_level_features()` | import + TimingContext call | WIRED | L345-348: import + `with TimingContext("build_all/race_level")` |
| `feature_engine.py::build_features()` | `race_level_features.py::compute_race_level_features()` | import + direct call | WIRED | L460-462: import + `df = compute_race_level_features(df)` |
| `race_level_features.py` | `market_bias_features.py` pattern | Shannon entropy pattern reuse | WIRED | `_calc_log_odds_entropy()` follows `_calc_entropy()` pattern: `p = p[p > 0]; -np.sum(p * np.log(p))` |
| `build_all()` output | SAFE-01 POST_RACE stripping | rl_* cols generated before SAFE-01 | WIRED | L348 (race_level call) before L367 (SAFE-01). rl_* cols subject to leakage check |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `race_level_features.py` | `p_norm` (implied probability) | `tanodds` via `1.0 / tanodds_valid` + groupby normalization | Yes | FLOWING |
| `race_level_features.py` | `entropy_values` | `p_norm` via `_calc_log_odds_entropy()` | Yes | FLOWING |
| `race_level_features.py` | `rank_results` | `tanodds_valid` via `_rank_features()` groupby apply | Yes | FLOWING |
| `feature_engine.py::build_all()` | `result_df` with rl_* columns | `compute_race_level_features(result_df)` at L348 | Yes | FLOWING |
| `feature_engine.py::build_features()` | `df` with rl_* columns | `compute_race_level_features(df)` at L462 | Yes | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 12 models have implied_prob_hhi + odds_skewness, no duplicates | Python runtime check | "ALL 12 MODELS: implied_prob_hhi + odds_skewness present, no duplicates" | PASS |
| Race-level features tests pass | `python -m pytest tests/test_race_level_features.py -v` | 8 passed | PASS |
| POST_RACE leakage tests pass | `python -m pytest tests/test_post_race_leakage.py -v` | 7 passed | PASS |
| Domain tests still pass | `python -m pytest tests/test_domain.py -v` | 22 passed | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| RLF-01 | 31-01 | rl_log_odds_entropy -- Shannon entropy | SATISFIED | `race_level_features.py` L96-100, L162-169 |
| RLF-02 | 31-01 | rl_odds_dispersion -- odds std | SATISFIED | `race_level_features.py` L103-106, L172-174 |
| RLF-03 | 31-01 | rl_top3_odds_gap -- 1st/3rd odds gap | SATISFIED | `race_level_features.py` L109-114, L190 |
| RLF-04 | 31-01 | rl_top1_odds -- 1st favorite broadcast | SATISFIED | `race_level_features.py` L117, L203-205 |
| RLF-05 | 31-01 | rl_favorite_rank_gap -- log odds gap | SATISFIED | `race_level_features.py` L120-124, L195 |
| RLF-06 | 31-01 | rl_n_horses -- field size | SATISFIED | `race_level_features.py` L127, L259-270 |
| RLF-07 | 31-02 | build_all/build_features parity | SATISFIED | `feature_engine.py` L345-348 + L460-462 |
| EFP-01 | 31-01 | implied_prob_hhi FEATURE_COLS promotion | SATISFIED | All 12 models verified with runtime check |
| EFP-02 | 31-01 | odds_skewness FEATURE_COLS promotion | SATISFIED | All 12 models verified with runtime check |
| EFP-03 | 31-02 | Feature manifest SHA256 update | SATISFIED | `data/feature_freeze_manifest.json` regenerated 2026-05-18 |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No debt markers or stubs found |

### Human Verification Required

None -- all truths are programmatically verifiable and all tests pass.

---

_Verified: 2026-05-18T08:30:00Z_
_Verifier: Claude (gsd-verifier)_
