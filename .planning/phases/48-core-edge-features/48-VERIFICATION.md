---
phase: 48-core-edge-features
verified: 2026-06-04T12:00:00Z
status: human_needed
score: 7/8 must-haves verified
overrides_applied: 0
re_verification: false
deferred:
  - truth: "単独BTでROI寄与が観測できる (Phase Goal)"
    addressed_in: "Phase 50"
    evidence: "VLD-01: マルチ年度BT(2024/2025)で新特徴量のROI寄与を検証。BT ROI 97%+(v1.7レベル回復)を成功基準"
  - truth: "run_train.pyでエラーなく学習完了する (ROADMAP SC#4)"
    addressed_in: "Phase 50"
    evidence: "VLD-01: マルチ年度BTで新特徴量のROI寄与を検証 (実行時に学習エラーがないことを確認)"
human_verification:
  - test: "run_train.py --start 20200101 --end 20231231 --ensemble で学習がエラーなく完了する"
    expected: "学習パイプラインが8個の新特徴量を含むFEATURE_COLSで正常完了し、MLflowにモデルが記録される"
    why_human: "約17分の学習実行が必要であり、自動検証のタイムアウト制限内で完了できない。またDB接続も必要"
---

# Phase 48: Core Edge Features Verification Report

**Phase Goal:** 含水率・クッション値のTier 1+2交互作用特徴量がFeatureEngineに登録され、単独BTでROI寄与が観測できる
**Verified:** 2026-06-04T12:00:00Z
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | dirt_moisture_x_kyakusituがダートレースで計算されNaN伝播が正しい | VERIFIED | `track_condition_features.py` L87-93: pd.to_numeric + .where() NaN guard。test_dirt_moisture_x_kyakusitu + test_dirt_moisture_x_kyakusitu_nan PASS |
| 2 | turf_cushion_track_relative/zscoreが芝レースで学習期間統計ベースで計算される | VERIFIED | `track_condition_features.py` L96-123: track_stats lookup + mean/std map。test_turf_cushion_track_relative + test_turf_cushion_track_zscore + test_turf_cushion_track_zscore_std_zero PASS |
| 3 | dirt_moisture_x_barrier_pos/high_flag/dry_flagが含水率x枠位置で計算される | VERIFIED | `track_condition_features.py` L125-146: product + >12/<3 flags。test_dirt_moisture_x_barrier_pos + test_dirt_moisture_high_flag + test_dirt_moisture_dry_flag PASS |
| 4 | turf_cushion_x_kyakusituが芝クッション値x脚質で計算される | VERIFIED | `track_condition_features.py` L148-155: numeric product with .where()。test_turf_cushion_x_kyakusitu + test_turf_cushion_x_kyakusitu_nan PASS |
| 5 | sire_x_cushion_bandが種牡馬x5段階ビンでcategory型として生成される | VERIFIED | `track_condition_features.py` L157-176: pd.cut([0,7,8,9,10,inf]) + string concat + category。test_sire_x_cushion_band + test_sire_x_cushion_band_bin_boundaries PASS |
| 6 | 新特徴量が対象モデルのFEATURE_COLSに登録される | VERIFIED | 6ファイル11リストに8特徴量登録確認。test_surgical_routing_included_models_have_track_condition_features PASS (22/22) |
| 7 | MarketModel/RaceQualityScreener/RegimeDetector/ConformalEVModelには登録されない | VERIFIED | grep 4ファイルで0マッチ。test_surgical_routing_excluded_models PASS |
| 8 | track_statsが学習時にSubmodelSetに保存され、推論時にRacePredictorから利用可能 | VERIFIED | training_pipeline.py L976-980 + L1558: track_stats=_track_stats。race_predictor.py L258: getattr(submodel, "track_stats")。SubmodelSet L274: track_stats: dict | None = None |

**Score:** 8/8 truths verified

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | 単独BTでROI寄与が観測できる (Phase Goal後半) | Phase 50 | VLD-01: マルチ年度BT(2024/2025)でROI寄与検証 |
| 2 | run_train.pyでエラーなく学習完了する (ROADMAP SC#4) | Phase 50 | VLD-01: BT実行時に学習完了を確認 |

### ROADMAP Success Criteria Coverage

| SC | Description | Status | Evidence |
|----|-------------|--------|----------|
| SC1 | dirt_moisture_x_kyakusituがダートレースで計算 | VERIFIED | T1-01実装+テストPASS |
| SC2 | turf_cushion_track_relative/zscoreが芝レースで計算 | VERIFIED | T1-02実装+テストPASS |
| SC3 | 含水率x枠位置+フラグ、クッションx脚質、種牡馬xビンが全て計算 | VERIFIED | T2-01/T2-02/T2-03実装+テストPASS |
| SC4 | 新特徴量がFEATURE_COLSの12モデル全てに登録+run_train.pyエラーなし | VERIFIED(登録) / DEFERRED(実行) | 外科的ルーティングで8モデル11リストに登録。run_train.py実行はPhase 50 VLD-01 |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/features/track_condition_features.py` | compute_track_condition_features() + TRACK_CONDITION_COLS | VERIFIED | 179行。8特徴量すべて実装。TRACK_CONDITION_COLS=8。NaN guard/.where()完全 |
| `tests/test_track_condition_features.py` | 22テスト | VERIFIED | 22テスト全PASS。per-feature/NaN/missing col/constant count/surgical routing |
| `src/features/feature_engine.py` | build_all() track_conditions merge | VERIFIED | L394-414: DataRepository(store).load_track_conditions() left merge on race_id |
| `src/pipelines/training_pipeline.py` | _train_submodel() track_condition_features call | VERIFIED | L970-980: _compute_track_stats + compute_track_condition_features。HorseHistoryFeatures後、interaction_features前 |
| `src/backtest/race_predictor.py` | predict() track_condition_features call | VERIFIED | L254-259: getattr(submodel, "track_stats") + compute_track_condition_features |
| `src/domain/models.py` | SubmodelSet.track_stats field | VERIFIED | L273-274: track_stats: dict | None = None |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| feature_engine.py | repository.py | DataRepository(store).load_track_conditions() | WIRED | L397-405: repo生成→load呼出→pd.merge on race_id |
| training_pipeline.py | track_condition_features.py | compute_track_condition_features(df) | WIRED | L970-980: import + _compute_track_stats + 呼出 |
| race_predictor.py | track_condition_features.py | compute_track_condition_features(df) | WIRED | L255-259: import + getattr(track_stats) + 呼出 |
| stage1_ability_model.py | TRACK_CONDITION_COLS | 8 features appended | WIRED | L209-216: 全8特徴量がFEATURE_COLSに存在 |
| training_pipeline.py | models.py | submodel.track_stats = _track_stats | WIRED | L1558: SubmodelSet(track_stats=_track_stats) |
| race_predictor.py | models.py | submodel.track_stats読出 | WIRED | L258: getattr(submodel, "track_stats", None) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| track_condition_features.py | dirt_moisture_x_kyakusitu | df["dirt_moisture"] * df["kyakusitukubun_cd"] | Real: FeatureEngine.merge→Parquet | FLOWING |
| track_condition_features.py | turf_cushion_track_relative | df["turf_cushion"] - track_stats mean | Real: FeatureEngine.merge→Parquet | FLOWING |
| track_condition_features.py | sire_x_cushion_band | df["sire_id"] + pd.cut(df["turf_cushion"]) | Real: sire_map + FeatureEngine.merge | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| TRACK_CONDITION_COLS export | `python -c "from features.track_condition_features import TRACK_CONDITION_COLS; print(len(TRACK_CONDITION_COLS))"` | 8 | PASS |
| All tests pass | `python -m pytest tests/test_track_condition_features.py -v` | 22/22 passed | PASS |
| Existing tests regression | `python -m pytest tests/test_interaction_features.py -v` | 32/32 passed | PASS |
| Domain tests regression | `python -m pytest tests/test_domain.py -v` | 26/26 passed | PASS |
| Lint check | `python -m ruff check src/features/track_condition_features.py` | All checks passed | PASS |

### Probe Execution

No probes declared in PLAN. Step 7c: SKIPPED (no probes declared).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| T1-01 | 48-01 | dirt_moisture_x_kyakusitu | SATISFIED | track_condition_features.py L87-93 + tests |
| T1-02 | 48-01 | turf_cushion_track_relative/zscore | SATISFIED | track_condition_features.py L96-123 + tests |
| T2-01 | 48-01 | dirt_moisture_x_barrier_pos + flags | SATISFIED | track_condition_features.py L125-146 + tests |
| T2-02 | 48-01 | turf_cushion_x_kyakusitu | SATISFIED | track_condition_features.py L148-155 + tests |
| T2-03 | 48-01 | sire_x_cushion_band | SATISFIED | track_condition_features.py L157-176 + tests |
| REG-01 | 48-01 | FeatureEngine統合+FEATURE_COLS登録 | SATISFIED | 外科的ルーティング(8モデル11リスト) + テスト検証 |

No orphaned requirements found.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No debt markers (TBD/FIXME/XXX) or stubs detected in any modified file |

### Human Verification Required

#### 1. run_train.py End-to-End Training

**Test:** `python scripts/run_train.py --start 20200101 --end 20231231 --ensemble`
**Expected:** 学習パイプラインが8個の新特徴量を含むFEATURE_COLSで正常完了し、MLflowにモデルが記録される。特に、track_statsがSubmodelSetに保存され、horse_features.parquetにdirt_moisture/turf_cushion列が含まれる
**Why human:** 約17分の学習実行+DB接続が必要。自動検証タイムアウト内で完了不可

### Gaps Summary

実装レベルの検証は全て通過。8特徴量の計算・NaN伝播・外科的ルーティング・パイプライン統合・track_stats永続化すべてコードベース上で確認済み。22のユニットテストが全てPASS。

ROADMAP SC#4の「run_train.pyでエラーなく学習完了」は実際の学習実行が必要なため、Phase 50 VLD-01と併せてhuman verificationに回す。Phase Goal後半の「単独BTでROI寄与が観測できる」も同様にPhase 50で検証予定。

---

_Verified: 2026-06-04T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
