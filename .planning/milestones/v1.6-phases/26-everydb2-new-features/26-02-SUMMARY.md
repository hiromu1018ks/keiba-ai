---
phase: 26-everydb2-new-features
plan: 02
subsystem: features
tags: [n_sanku, n_record, dam_pedigree, BMS, breeder_strength, course_record_time, LightGBM]

# Dependency graph
requires:
  - phase: 25-quick-win-wire-existing
    provides: _train_submodel()統合パターン
  - phase: 26-01
    provides: MiningFeatures, _train_submodel() Group F block
provides:
  - DamPedigreeFeatures: n_sankuから繁殖牝馬産駒成績4特徴量を生成するモジュール
  - RecordFeatures: n_recordからコースレコード1特徴量を生成するモジュール
  - SireFeatures BMS拡張: bms_distance_wr, bms_surface_wrの2特徴量
  - FEATURE_COLS更新: Stage1(90), Win(61), Place HIT(61), Place RETURN(63)
  - PIT監査ドキュメント: n_hansyoku 19列+n_sanku 26列全PRE分類
affects: [26-03, feature-engineering]

# Tech tracking
tech-stack:
  added: []
patterns:
  - "dam cross-reference: entry_df.kettonum -> sanku.mnum (dam lookup) -> sanku again (offspring lookup)"
  - "course_record merge: (jyocd, trackcd, kyori) lookup → race-level feature, merge on race_id"
  - "BMS extension pattern: same sire_career_stats row lookup for bms surface/distance stats"

key-files:
  created:
    - src/features/dam_pedigree_features.py
    - src/features/record_features.py
    - tests/test_dam_pedigree_features.py
    - tests/test_record_features.py
  modified:
    - src/features/sire_features.py
    - src/pipelines/training_pipeline.py
    - src/models/stage1_ability_model.py
    - src/models/two_stage_return_model.py
    - tests/test_two_stage_return_model.py
    - tests/test_win_feature_analysis.py

key-decisions:
  - "DamPedigreeFeatures: sankuでcross-reference chain (kettonum→MNum→offspring)で産駎成績を集計"
  - "breeder_strength: log(1 + unique breeder count) で繁殖牝馬の生産者多様性を指標化"
  - "RecordFeatures: RecInfoKubun=1フィルタ+最新レコード選択+varchar→秒変換"
  - "Place RETURN_FEATURE_COLSにWin全dam/bms/record featuresを追加 (既存テスト制約)"
  - "BMS拡張: compute_batch()の既存bms_id lookupループ内で8列追加取得 (追加ループ不要)"

patterns-established:
  - "dam cross-reference chain: sanku 2段lookup (dam MNum → offspring list → career aggregation)"
  - "RecTime varchar→秒変換: char[0]=分, chars[1:4]=ss.s (/10.0), NaN for invalid"

requirements-completed: [DATA-01, DATA-02]

# Metrics
duration: 36min
completed: 2026-05-14
---

# Phase 26 Plan 02: 血統特徴量 + BMS拡張 + コースレコード Summary

**n_sanku繁殖牝馬産駒4特徴量 + n_recordコースレコード1特徴量 + BMS拡張2特徴量を_train_submodel()に統合 (計7新特徴量)**

## Performance

- **Duration:** 36 min
- **Started:** 2026-05-14T13:40:28Z
- **Completed:** 2026-05-14T14:16:11Z
- **Tasks:** 3
- **Files modified:** 10

## Accomplishments
- DamPedigreeFeatures実装: sanku cross-reference chainでdam_wr/dam_surface_wr/dam_prize_log/breeder_strength
- RecordFeatures実装: n_recordからRecInfoKubun=1フィルタ+最新レコード選択+varchar秒変換
- SireFeatures BMS拡張: compute_batch()にbms_distance_wr/bms_surface_wr追加 (既存ループ内で8列追加取得)
- _train_submodel() Group B-2 (dam_pedigree) + B-3 (record) block追加
- Stage1 83→90, Win 57→61, Place HIT 57→61, Place RETURN 59→63
- n_hansyoku 19列+n_sanku 26列 全PRE分類文書化
- 17新テスト追加 (7 DamPedigree + 10 RecordFeatures/BMS), 1448テスト全通過

## Task Commits

Each task was committed atomically:

1. **Task 1: DamPedigreeFeatures + テスト (TDD RED->GREEN)** - `9147e72` (feat)
2. **Task 2: RecordFeatures + sire_features BMS拡張 + テスト (TDD RED->GREEN)** - `a2e9258` (feat)
3. **Task 3: _train_submodel()統合 + FEATURE_COLS更新** - `5ffdcc6` (feat)

## Files Created/Modified
- `src/features/dam_pedigree_features.py` - 繁殖牝馬産駒成績4特徴量 (dam_wr, dam_surface_wr, dam_prize_log, breeder_strength)
- `src/features/record_features.py` - コースレコード1特徴量 (course_record_time)
- `src/features/sire_features.py` - BMS拡張2特徴量 (bms_distance_wr, bms_surface_wr)
- `tests/test_dam_pedigree_features.py` - 7テスト (mock-based, DB不要)
- `tests/test_record_features.py` - 10テスト (7 RecordFeatures + 3 BMS拡張)
- `src/pipelines/training_pipeline.py` - Group B-2/B-3 block追加、_sire_cols_needed 7列化
- `src/models/stage1_ability_model.py` - 7特徴量追加 (83→90)
- `src/models/two_stage_return_model.py` - Win +4, Place HIT +4, Place RETURN +4
- `tests/test_two_stage_return_model.py` - feature_df fixtureに7列追加
- `tests/test_win_feature_analysis.py` - original_allリストに4特徴量追加

## Decisions Made
- DamPedigreeFeatures: sankuのMNum列で2段cross-reference (馬→母→産駎一覧)を実行し、各産駎のcareer statsを集計
- breeder_strength: 繁殖牝馬の産駎を生産したユニーク生産者数をlog変換 (多様な生産者が関与する繁殖牝馬ほど高い評価)
- RecordFeatures: RecTime 4文字varchar (msss形式) を分*60+ss.s/10 で秒数に変換
- Place RETURN_FEATURE_COLSにWin全dam/bms/record featuresを含めた (既存テストが「Win全列がPlace RETURNに含まれる」ことを検証)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Place RETURN_FEATURE_COLS不足でテスト失敗**
- **Found during:** Task 3 (テスト実行時)
- **Issue:** Winにbreeder_strength, bms_distance_wr, course_record_timeを追加したがPlace RETURNに未追加。test_place_return_feature_cols_include_place_specificが失敗
- **Fix:** Place RETURN_FEATURE_COLSにもbreeder_strength, bms_distance_wr, course_record_timeを追加
- **Files modified:** src/models/two_stage_return_model.py
- **Verification:** test_two_stage_return_model.py全テスト通過
- **Committed in:** 5ffdcc6 (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** 既存テスト制約 (Win全列がPlace RETURNに含まれる) への対応。スコープクリープなし。

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- DamPedigreeFeatures, RecordFeaturesモジュール完成、_train_submodel()統合済み
- Plan 03 (interaction features) も同じ_train_submodel()統合パターンを使用可能
- POST_RACE漏洩テスト自動検証済み
- ETL実行 (`run_etl.py --tables n_sanku n_record`) はユーザーがローカルで実行必要

---
*Phase: 26-everydb2-new-features*
*Completed: 2026-05-14*

## Self-Check: PASSED

All 11 files verified. All 3 commits (9147e72, a2e9258, 5ffdcc6) verified in git log.
