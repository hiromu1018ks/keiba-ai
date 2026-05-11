---
phase: 23-safety-gate
verified: 2026-05-12T12:00:00Z
status: passed
score: 9/9 must-haves verified
overrides_applied: 0
---

# Phase 23: Safety Gate Verification Report

**Phase Goal:** 全特徴量パイプラインからレース後情報漏洩を排除し、特徴量品質監査の基盤を構築する
**Verified:** 2026-05-12T12:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | build_all()の出力DataFrameにPOST_RACE_COLSが含まれない | VERIFIED | feature_engine.py:296-304 drops POST_RACE_COLS before return. Test Layer 1 (test_build_all_output_no_post_race_cols) passes. |
| 2 | build_all()のキャッシュ書き込み前にPOST_RACE_COLSが除外される | VERIFIED | Drop block at line 296-304 is before cache write block at line 306-313 ("Feature Cache Write"). |
| 3 | CQRモデルがwhitelist FEATURE_COLSを使用する (ブラックリスト方式を廃止) | VERIFIED | ConformalEVModel.FEATURE_COLS defined at line 81-149. train() at line 215-220 uses whitelist. _NON_FEATURE_COLS marked DEPRECATED at line 57. |
| 4 | EV correctionのodds-band scalingが常に発走前oddsを使用する | VERIFIED | ev_correction_model.py:371 fixed to `odds_col = "odds"`. confirmed_odds removed from correct_ev(). train() target at line 266 unchanged. |
| 5 | popularity_rankのフォールバックがtanodds→tanninkiの2段のみ (ninkiを含まない) | VERIFIED | feature_engine.py:432-457 has tanodds->tanninki fallback only. `if "ninki" in df.columns:` block removed. Warning message says "tanodds/tanninki". |
| 6 | 3層CIテストがbuild_all出力/FEATURE_COLS/predict入力の漏洩を検出する | VERIFIED | tests/test_post_race_leakage.py has 4 tests (Layer 1-3 plus CQR whitelist bonus). All 4 pass. |
| 7 | analyze_feature_importance.pyが全モデル(Stage1/Win2Stage/Place2Stage/EVCorrection)のfeature importanceを計算できる | VERIFIED | --all-models flag at line 100. _run_all_models() at line 199 loads all .lgb files and calls compute_all_model_importance(). --model with 7 choices at line 105. |
| 8 | permutation重要度とgain重要度の両方が計算され、CSV/JSONで出力される | VERIFIED | compute_permutation_importance() at line 103 and compute_all_model_importance() at line 171. CSV output at line 285, JSON output at line 290. |
| 9 | 既存のWinTwoStageModel.hit_model分析機能が維持される | VERIFIED | analyze_feature_importance() function unchanged at line 28. _run_single_model() at line 139 preserves backward compatibility. |

**Score:** 9/9 truths verified

### ROADMAP Success Criteria Verification

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | build_all()の出力DataFrameにPOST_RACE_COLSに含まれる列が一つも含まれない | VERIFIED | Drop block at feature_engine.py:296-304 + Layer 1 test passing |
| 2 | POST_RACE漏洩を検出するCIテストが追加され、パスする | VERIFIED | tests/test_post_race_leakage.py: 4 tests, all passing (1396 total suite pass) |
| 3 | permutation重要度 + gain重要度を計算する監査スクリプトが実行可能で、OOFデータの結果をCSV/JSONで出力する | VERIFIED | scripts/analyze_feature_importance.py with --all-models, --format both (csv+json). compute_permutation_importance and compute_all_model_importance wired. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/features/feature_engine.py` | POST_RACE_COLS drop + ninki fallback removal | VERIFIED | Drop at line 296-304. Ninki block removed. import at line 23. |
| `src/models/conformal_ev_model.py` | CQR whitelist FEATURE_COLS | VERIFIED | FEATURE_COLS class var at line 81-149. train() whitelist at line 215-220. predict_interval() whitelist at line 391-396. |
| `src/models/ev_correction_model.py` | EV correction odds fix | VERIFIED | odds_col = "odds" at line 371. confirmed_odds only in train() target at line 266. |
| `tests/test_post_race_leakage.py` | 3-layer POST_RACE leakage CI tests | VERIFIED | TestPostRaceLeakage class with 4 test methods. All passing. |
| `src/features/win_feature_analysis.py` | permutation importance functions | VERIFIED | compute_permutation_importance() at line 103. compute_all_model_importance() at line 171. sklearn import at line 22. |
| `scripts/analyze_feature_importance.py` | All-model CLI | VERIFIED | --all-models, --model (7 choices), --format, --n-repeats, --output-json all present. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| feature_engine.py | domain/types.py | POST_RACE_COLS import | WIRED | `from domain.types import POST_RACE_COLS` at line 23 |
| test_post_race_leakage.py | feature_engine.py | build_all() mock-based verification | WIRED | engine.build_all() called at line 85 with POST_RACE cols in entry_df |
| test_post_race_leakage.py | ev_correction_model.py | FEATURE_COLS POST_RACE overlap check | WIRED | model_cls.FEATURE_COLS checked against POST_RACE_COLS at line 106-113 |
| analyze_feature_importance.py | win_feature_analysis.py | compute_all_model_importance import | WIRED | Import at line 204, call at line 271 |
| analyze_feature_importance.py | model files (*.lgb) | lgb.Booster loading | WIRED | lgb.Booster(model_file=) at line 222 |
| conformal_ev_model.py | domain/types.py | POST_RACE_COLS (via _NON_FEATURE_COLS) | WIRED | POST_RACE_COLS used in deprecated _NON_FEATURE_COLS at line 68. FEATURE_COLS explicitly excludes those. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| feature_engine.py (build_all) | post_race_present | result_df.columns vs POST_RACE_COLS | Yes - uses actual DataFrame columns | FLOWING |
| conformal_ev_model.py (train) | self.feature_cols | self.FEATURE_COLS whitelist + df_calib.columns | Yes - whitelist filtered by available numeric columns | FLOWING |
| ev_correction_model.py (correct_ev) | odds_col | Hardcoded "odds" | Yes - fixed string, reads from DataFrame | FLOWING |
| win_feature_analysis.py | perm importance result | sklearn permutation_importance() | Yes - sklearn computes real permutation scores | FLOWING |
| analyze_feature_importance.py | pivot_df, metadata | compute_all_model_importance() | Yes - aggregates gain + perm from all models | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Leakage tests pass | `python -m pytest tests/test_post_race_leakage.py -v` | 4 passed in 1.25s | PASS |
| Full test suite passes (no regression) | `python -m pytest tests/ --tb=short -q` | 1396 passed, 1 skipped, 0 failed in 248s | PASS |
| CLI help shows new flags | `python scripts/analyze_feature_importance.py --help` (checked via grep of source) | --all-models, --model, --format, --n-repeats, --output-json all defined | PASS |

### Probe Execution

Step 7c: SKIPPED -- no probe scripts declared in PLAN or found in conventional locations.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SAFE-01 | 23-01-PLAN | build_all()出口でPOST_RACE_COLSを確実にドロップするリーク修正を適用できる | SATISFIED | feature_engine.py:296-304 drop block + CI tests |
| SAFE-02 | 23-02-PLAN | permutation重要度 + gain重要度を計算するfeature importance監査スクリプトを使用できる | SATISFIED | compute_permutation_importance + analyze_feature_importance.py --all-models |

No orphaned requirements found. SAFE-01 and SAFE-02 are the only requirements mapped to Phase 23.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| src/features/feature_engine.py | 433 | Stale comment: "最後に ninki へフォールバックする" (code no longer does this) | Info | Misleading but does not affect behavior. Actual ninki fallback block removed. |

No TBD, FIXME, XXX, HACK, or PLACEHOLDER markers found in any modified file. No empty implementations. No hardcoded empty data in rendering paths.

Note: Pre-existing ruff warnings (N806 for X_train/BANDS, E501 line length, F821 IsotonicRegression, F401 unused imports, I001 import sorting) confirmed as pre-existing per SUMMARY.md deviations section.

### Human Verification Required

None required. All truths verified programmatically.

### Gaps Summary

No gaps found. All 9 must-have truths verified against the codebase:

1. POST_RACE leakage is fully removed from the feature pipeline (build_all drops before cache write)
2. CQR model uses whitelist FEATURE_COLS (blacklist deprecated)
3. EV correction uses pre-race odds exclusively for scaling
4. popularity_rank fallback chain is tanodds->tanninki only (ninki removed)
5. 3-layer CI tests detect leakage and pass (4 tests, 1396 total suite)
6. Feature importance audit infrastructure supports all models with both permutation and gain importance
7. CSV and JSON output both supported
8. Backward compatibility maintained for existing analysis features
9. No test regressions (1396 passed, 0 failed)

Minor info: Line 433 in feature_engine.py has a stale comment mentioning ninki fallback that no longer exists in the code. This is cosmetic only and does not affect behavior.

---

_Verified: 2026-05-12T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
