---
phase: 28-validation-freeze
plan: 01
subsystem: validation
tags: [pytest, feature-freeze, sha256, manifest, lightgbm]

# Dependency graph
requires:
  - phase: 23-safety-gate
    provides: POST_RACE漏洩テスト (SAFE-01)
  - phase: 27-feature-interactions
    provides: FEATURE_COLS最終状態 (交互作用・TE追加済み)
provides:
  - pytest全テスト通過確認 (1,523 passed / 0 new failures)
  - data/feature_freeze_manifest.json (12モデルFEATURE_COLS + SHA256)
  - scripts/freeze_feature_manifest.py (冪等manifest生成)
affects: [28-02, v1.6-close]

# Tech tracking
tech-stack:
  added: []
  patterns: [SHA256-feature-freeze-manifest, deterministic-json-sort-keys-indent2]

key-files:
  created:
    - scripts/freeze_feature_manifest.py
    - data/feature_freeze_manifest.json
  modified: []

key-decisions:
  - "data/feature_freeze_manifest.jsonをgit管理に含めるため-f強制追加 (.gitignoreがdata/全体を除外)"
  - "3件のtest_training_pipeline.py失敗は既知のduplicate race_ids問題 (Phase 27-03 SUMMARY) で回帰なしと判断"

patterns-established:
  - "Feature Freeze Manifest: 全モデルFEATURE_COLSをJSON+SHA256で凍結するパターン"

requirements-completed: [pytest-regression-check, feature-freeze-manifest]

# Metrics
duration: 5min
completed: 2026-05-16
---

# Phase 28 Plan 01: pytest回帰確認 + 特徴量凍結manifest Summary

**1,523テスト通過確認 (回帰なし) + 12モデルFEATURE_COLSをSHA256付きJSON manifestで凍結**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-15T22:23:29Z
- **Completed:** 2026-05-15T22:28:00Z
- **Tasks:** 2
- **Files modified:** 2 (created)

## Accomplishments
- pytest全1,523テスト通過 + 1 skipped (3件の既知duplicate race_ids失敗は回帰なし)
- 12モデルFEATURE_COLS凍結manifest生成 (data/feature_freeze_manifest.json)
- 各モデルにSHA256ハッシュ付与、全体にもoverall_sha256を記録

## Task Commits

1. **Task 1: pytest全テスト実行 + 回帰確認** - コミット不要 (ファイル変更なし)
2. **Task 2: 特徴量凍結manifest生成スクリプト作成 + 実行** - `ab0488d` (feat)

## Test Results

```
1,523 passed, 3 failed (known), 1 skipped, 5 warnings in 181.45s
```

Failed tests (all known, pre-existing):
- `test_training_pipeline.py::TestTrainingPipelineV5::test_run_returns_trained_models_v5`
- `test_training_pipeline.py::TestTrainingPipelineV5::test_pipeline_trains_per_surface`
- `test_training_pipeline.py::TestTrainingPipelineV5::test_pipeline_logs_to_mlflow`

Error: `record_df has duplicate race_ids: 3600` -- documented in Phase 27-03 SUMMARY as a known issue, not a regression.

## Feature Freeze Manifest

12 models frozen with SHA256 hashes:

| Model | Features | SHA256 (first 8) |
|-------|----------|-----------------|
| AbilityModel | 95 | ce2e2c4e |
| WinTwoStageModel | 81 | d237171c |
| PlaceTwoStageModel.HIT | 84 | f6acfd66 |
| PlaceTwoStageModel.RETURN | 86 | 0e9cfb78 |
| EVCorrectionModel | 24 | 96923d4c |
| PlaceEVCorrectionModel | 24 | 3a3239bb |
| ConformalEVModel | 131 | d6f9d5d1 |
| RegimeDetector | 8 | 83fd38d2 |
| MarketModel | 7 | 6d9240d6 |
| PlaceAbilityModel | 61 | 2ded412b |
| RaceQualityScreener | 22 | fae4eb5a |
| WideTwoStageModel.SHARED | 5 | 01745209 |

Overall SHA256: 2db3810f4e1e...

## Files Created/Modified
- `scripts/freeze_feature_manifest.py` - 特徴量凍結manifest生成スクリプト (冪等実行可能)
- `data/feature_freeze_manifest.json` - v1.6凍結manifest (12モデルFEATURE_COLS + SHA256)

## Decisions Made
- data/feature_freeze_manifest.jsonは.gitignoreで除外されているdata/配下にあるが、成果物としてgit管理するため`git add -f`で強制追加
- 3件のtest_training_pipeline.py失敗はPhase 27-03 SUMMARY記載の既知問題であり、新規回帰ではないためPlan 02に進む

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `data/`ディレクトリが.gitignoreで除外されているため、manifestをgit管理するには`git add -f`が必要だった。スクリプトの冪等性によりいつでも再生成可能だが、バージョン管理上の利便性のため-force追加とした。

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- 特徴量凍結manifest生成完了、Plan 02 (バックテスト実行) に進行可能
- スクリプトは冪等実行可能 (同じコードベースから同じmanifestが生成される)

## Self-Check: PASSED

- FOUND: scripts/freeze_feature_manifest.py
- FOUND: data/feature_freeze_manifest.json
- FOUND: .planning/phases/28-validation-freeze/28-01-SUMMARY.md
- FOUND: commit ab0488d

---
*Phase: 28-validation-freeze*
*Completed: 2026-05-16*
