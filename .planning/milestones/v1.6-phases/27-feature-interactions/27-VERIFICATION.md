---
phase: 27-feature-interactions
verified: 2026-05-15T14:20:00Z
status: passed
score: 10/10 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 27: Feature Interactions Verification Report

**Phase Goal:** 最終ベース特徴量セット上にドメイン知識に基づく交互作用項を生成し、高カーディナリティカテゴリ変数をターゲットエンコーディングで処理する
**Verified:** 2026-05-15T14:20:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | オッズ相対特徴量 (rel_popularity_rank_zscore, rel_fuku_odds_zscore) がfukuoddslow/popularity_rankのDataFrame内相対zscoreとして計算される | VERIFIED | `_BASE_FEATURES` has 9 entries (7+2 odds). `RELATIVE_FEATURE_COLS` = 9. Computation uses `groupby("race_id").transform("mean"/"std")` zscore. |
| 2 | 能力値相対特徴量 (rel_p_ability_win_zscore, rel_p_ability_win_rank, rel_odds_ability_deviation) がStage1 OOF後のdf上で計算される | VERIFIED | `compute_stage2_relative_features()` defined with p_ability_win zscore+rank, odds_to_ability_ratio zscore. `STAGE2_RELATIVE_FEATURE_COLS` = 3. |
| 3 | Phase 26残り4特徴量がStage1 FEATURE_COLSに含まれる | VERIFIED | `rel_haron_vs_mean`, `rel_blood_quality_rank`, `rel_sire_quality_rank`, `rel_weight_zscore` all present at indices 91-94. Stage1 FEATURE_COLS = 95. |
| 4 | 新規オッズ相対・能力値相対特徴量がWin/Place FEATURE_COLSに含まれる | VERIFIED | 5 relative features in Win (81), Place HIT (84), Place RETURN (86). |
| 5 | 12個の交互作用項 (既存3+新規9) が生成される | VERIFIED | `INTERACTION_COLS` = 12 (3 category product + 6 numeric product + 3 existing). All 12 in Win/Place FEATURE_COLS. |
| 6 | カテゴリ積はastype(str)+'_'+astype(str) -> category型で生成される | VERIFIED | Code uses `df[col].astype(str) + "_" + df[col2].astype(str)).astype("category")` pattern for `surface_x_distance_bin`, `blood_keito_x_surface`, `grade_code_x_distance_bin`. |
| 7 | 数値積は.where(notna()) NaN安全パターンで生成される | VERIFIED | All 6 numeric products use `.where(df[col].notna() & df[col2].notna(), other=float("nan"))` pattern. |
| 8 | blood_keito_cd, kisyucode, chokyosicodeがターゲットエンコーディングされる | VERIFIED | `TargetEncoder` class with `fit_transform_oof()` processing all 3 columns. `TE_STAGE2_FEATURE_COLS` = 3. All 3 in Win/Place FEATURE_COLS. |
| 9 | TEはOOFリークなしで計算される (expanding window + fold分割) | VERIFIED | 3-fold expanding window by race_date, same boundaries as AbilityModel.train_oof(). Test `test_no_future_information_leakage` passes. |
| 10 | 全テスト通過、POST_RACE漏洩テスト通過 | VERIFIED | 120 phase-27 tests pass. `test_post_race_leakage.py` 4/4 pass. Full suite: 1523 passed, 3 pre-existing pipeline failures unrelated to Phase 27. |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/features/relative_features.py` | _BASE_FEATURES 9 + compute_stage2_relative_features() | VERIFIED | 9 specs, STAGE2_RELATIVE_FEATURE_COLS=3, full computation with guard clauses |
| `src/features/interaction_features.py` | 12 interaction features + INTERACTION_COLS | VERIFIED | 12 INTERACTION_COLS exported, 3 cat+6 numeric new, NaN-safe |
| `src/features/target_encoding.py` | TargetEncoder class | VERIFIED | fit_transform_oof + transform methods, smoothing, cold-start fill |
| `src/models/stage1_ability_model.py` | FEATURE_COLS with Phase 26 remainder | VERIFIED | 95 features, rel_weight_zscore at index 94 |
| `src/models/two_stage_return_model.py` | Win/Place FEATURE_COLS updated | VERIFIED | Win=81, Place HIT=84, Place RETURN=86 |
| `src/pipelines/training_pipeline.py` | Pipeline integration | VERIFIED | TargetEncoder at line 558, compute_stage2_relative_features at line 585, interaction at line 494 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| relative_features.py | training_pipeline.py | _train_submodel() | WIRED | `compute_stage2_relative_features` imported and called at line 585-586 |
| interaction_features.py | training_pipeline.py | _train_submodel() Group E | WIRED | `compute_interaction_features` imported and called at line 494-497 |
| target_encoding.py | training_pipeline.py | _train_submodel() Stage1 OOF後 | WIRED | `TargetEncoder` imported at line 558, instantiated at line 565, `fit_transform_oof` at line 569 |
| two_stage_return_model.py | relative_features.py | FEATURE_COLS参照 | WIRED | All 5 relative features in Win/Place FEATURE_COLS |
| two_stage_return_model.py | interaction_features.py | FEATURE_COLS参照 | WIRED | All 12 INTERACTION_COLS in Win/Place FEATURE_COLS |
| two_stage_return_model.py | target_encoding.py | FEATURE_COLS参照 | WIRED | All 3 TE features in Win/Place FEATURE_COLS |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| compute_relative_features | fukuoddslow, popularity_rank | DataFrame groupby("race_id") zscore | Real z-score computation | FLOWING |
| compute_stage2_relative_features | p_ability_win, odds_to_ability_ratio | Stage1 OOF output + market probability | Real z-score/rank from model output | FLOWING |
| compute_interaction_features | surface, distance_bin, sire_wr, etc. | DataFrame base columns | Real category/numeric products | FLOWING |
| TargetEncoder.fit_transform_oof | blood_keito_cd, kisyucode, chokyosicode | Stage1 OOF df + kakuteijyuni target | Real smoothed TE values | FLOWING |

### Probe Execution

Not applicable -- this phase produces feature computation modules, not scripts with probes.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All Phase 27 tests pass | `python -m pytest tests/test_relative_features.py tests/test_interaction_features.py tests/test_target_encoding.py tests/test_two_stage_return_model.py tests/test_post_race_leakage.py -v` | 120 passed in 3.51s | PASS |
| INTERACTION_COLS count in 10-15 range | Python assertion | 12 (within range) | PASS |
| TE OOF safety test passes | `pytest tests/test_target_encoding.py::TestFitTransformOof::test_no_future_information_leakage` | PASSED | PASS |
| Stage1 has no TE features (safety) | Python assertion | 0 TE features in Stage1 | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| INTER-01 | 27-01 | レース内相対ランク特徴量(オッズ、能力値等の相対位置)を生成できる | SATISFIED | RELATIVE_FEATURE_COLS=9, STAGE2_RELATIVE_FEATURE_COLS=3, pipeline wired |
| INTER-02 | 27-02 | ドメイン知識に基づく10-15個の条件付き交互作用項を生成できる | SATISFIED | INTERACTION_COLS=12 (10-15 range), category+numeric products |
| INTER-03 | 27-03 | 高カーディナリティカテゴリ変数のターゲットエンコーディングを実装できる | SATISFIED | TargetEncoder class, 3 TE features, OOF-safe expanding window |

No orphaned requirements found. All Phase 27 requirements (INTER-01, INTER-02, INTER-03) are covered by plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No TBD/FIXME/XXX/TODO/HACK markers found in any Phase 27 files |

### Human Verification Required

None -- all truths are programmatically verifiable through code inspection and test execution.

### Gaps Summary

No gaps found. All 3 ROADMAP success criteria are satisfied:

1. **SC1 -- Relative rank features**: 5 relative features (2 odds + 3 ability) generated and included in Win/Place FEATURE_COLS. Pipeline wired with compute_relative_features() and compute_stage2_relative_features().
2. **SC2 -- 10-15 interaction features**: 12 domain-knowledge interaction features generated (3 category product + 6 numeric product + 3 existing). All NaN-safe with proper guard clauses.
3. **SC3 -- Target encoding OOF-safe**: TargetEncoder with 3-fold expanding window, Beta(1,10) smoothing, cold-start global mean fallback. 3 TE features in Win/Place FEATURE_COLS, Stage1 excluded for safety.

Pre-existing issue (out of scope): 3 training_pipeline tests fail with "record_df has duplicate race_ids: 3600" -- documented in deferred-items.md, unrelated to Phase 27 changes.

---

_Verified: 2026-05-15T14:20:00Z_
_Verifier: Claude (gsd-verifier)_
