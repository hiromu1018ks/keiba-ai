---
phase: 52-shared-feature-builder-consistency
plan: 04
subsystem: features
tags: [feature-builder, paper-trading, consistency, gap-closure]
dependency_graph:
  requires: [52-01, 52-02, 52-03]
  provides: [_build_features_fb helper, GAP-1 closure, GAP-2 closure]
  affects: [scripts/run_paper_trading.py, src/features/feature_builder.py]
tech_stack:
  added: []
  patterns: [shared-feature-helper, FeatureBuilder-delegation]
key_files:
  created: []
  modified:
    - scripts/run_paper_trading.py
    - src/features/feature_builder.py
decisions:
  - "_build_features_fb() は build_for_training() を使用し、JRAフィルタも適用済みの DataFrame を返す"
  - "RacePredictor.predict() のシグネチャ上 hist/jockey/trainer/jt の個別計算は維持"
  - "BloodlineFeatures 暗黙実行を docstring に明記 (実装変更なし)"
metrics:
  duration: 386s
  completed: "2026-06-06T06:06:44Z"
  tasks: 6
  files: 2
  tests: 46
---

# Phase 52 Plan 04: GAP クロージャ (PT FeatureBuilder 統一 + docstring 整備) Summary

run_paper_trading.py の predict/diagnose/dry-run 3モードの特徴量生成を FeatureBuilder.build_for_training() に統一し、BT/PT/Train パイプラインの完全同一化を達成 (GAP-1 解消)。また _enrich_features() docstring に BloodlineFeatures 暗黙実行を明記 (GAP-2 解消)。

## Performance

- **Duration:** 6 min
- **Started:** 2026-06-06T06:00:18Z
- **Completed:** 2026-06-06T06:06:44Z
- **Tasks:** 6
- **Files modified:** 2

## Accomplishments

- _run_predict() / _run_diagnose() / _run_dry_run() の3モード全てが FeatureBuilder 経由の特徴量生成に統一
- 重複していた手動 BloodlineFeatures/SireFeatures/PaceAptitudeFeatures/CourseFeatures エンリッチメントコードを除去 (約200行削減)
- _enrich_features() docstring に BloodlineFeatures (blood_* カラム) が FeatureEngine.build_all() Group B で暗黙実行される旨を明記
- 46件の Phase 52 ユニットテストが全て通過、import/lint チェック通過

## Task Commits

Each task was committed atomically:

1. **Task 1: _build_features_fb() ヘルパー抽出** - `8dc60ff` (feat)
2. **Task 2: _run_predict() FeatureBuilder 置換** - `1ad6978` (feat)
3. **Task 3: _run_diagnose() FeatureBuilder 置換** - `fd6cb86` (feat)
4. **Task 4: _run_dry_run() FeatureBuilder 置換** - `ef96ef0` (feat)
5. **Task 5: _enrich_features() docstring 更新** - `96465cd` (docs)
6. **Task 6: Verification (lint + import + tests)** - no code changes (verification only)

## Files Created/Modified

- `scripts/run_paper_trading.py` - _build_features_fb() ヘルパー追加、_run_predict/diagnose/dry_run の特徴量生成を FeatureBuilder に置換
- `src/features/feature_builder.py` - モジュール docstring + _enrich_features() docstring に BloodlineFeatures 暗黙実行を明記

## Decisions Made

- **_build_features_fb() の設計:** FeatureBuilder.build_for_training() をラップし、JRAフィルタを適用済みの DataFrame を返す。ParquetStore を引数に取るため、AnyDB2/Parquet 両モードで同じ関数を使用可能
- **hist/jockey/trainer/jt の個別計算維持:** RacePredictor.predict() のシグネチャが個別 DataFrame を要求するため、FeatureBuilder に統合済みの特徴量とは別に計算を維持。将来的な predict() シグネチャ変更で除去可能
- **BloodlineFeatures は docstring のみ更新:** 実装変更なし。FeatureEngine.build_all() Group B で既に実行されており、blood_* カラムが正しく生成されているため

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## Next Phase Readiness

- GAP-1 (BLOCKER) と GAP-2 (WARNING) が両方解消。VERIFICATION.md の 9 truths が全て VERIFIED/PARTIAL → VERIFIED になる見込み
- Phase 52 の全 PLAN (01-04) が完了。再検証で SC1 完全達成を確認可能

---
*Phase: 52-shared-feature-builder-consistency*
*Completed: 2026-06-06*
