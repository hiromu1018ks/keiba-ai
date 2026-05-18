---
phase: 31-race-level-aggregation-features
plan: 02
subsystem: features/engine
tags: [feature-engineering, integration, post-race-safety, manifest]
dependency_graph:
  requires: [31-01]
  provides: [build_all-race-level-integration, build_features-race-level-parity, rl-leakage-tests]
  affects: [feature_engine.py, test_post_race_leakage.py]
tech_stack:
  added: []
  patterns: [submodule-integration, ast-based-source-analysis]
key_files:
  created: []
  modified:
    - src/features/feature_engine.py
    - tests/test_post_race_leakage.py
    - data/feature_freeze_manifest.json
decisions:
  - build_features()にはTimingContextなし (単レース推論でオーバーヘッド不要)
  - AST解析でPOST_RACE列の非使用を検証するテストパターンを採用
metrics:
  duration: 180s
  completed: "2026-05-18"
  tasks: 2
  files: 3
---

# Phase 31 Plan 02: Feature Engine Integration & Test Verification Summary

feature_engineの両パス(build_all/build_features)にcompute_race_level_features()を統合し、POST_RACE安全性を3テストで検証、manifestを再生成

## Completed Tasks

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | feature_engine統合 | `690c5c6` | src/features/feature_engine.py |
| 2 | POST_RACEテスト拡張 + manifest再生成 | `3b9d170` | tests/test_post_race_leakage.py, data/feature_freeze_manifest.json |

## Implementation Details

### Task 1: feature_engine統合

**build_all()** (line 344-347): `compute_difficulty_score()`の直後に`TimingContext("build_all/race_level")`付きで呼び出しを追加。SAFE-01 POST_RACE strippingの前に実行されるため、rl_*列も漏洩チェックの対象となる。

**build_features()** (line 456-459): `_map_basic_features()`の直後に`compute_race_level_features()`を呼び出し。TimingContextなし (単レース推論でオーバーヘッド不要)。コメント `# 6b. レース構造特徴量 (RLF-07 parity)` を付与。

両パスで同じ`compute_race_level_features()`関数を呼び出す (per D-04)。関数はrace_id有無で自動的に単レース/マルチレース処理を分岐する設計。

### Task 2: POST_RACEテスト拡張

新規テストクラス `TestRaceLevelFeatures` を3テストで追加:

1. **test_race_level_features_no_post_race_input**: `ast.parse()`で`compute_race_level_features()`のソースコード内の全文字列リテラルを抽出し、POST_RACE_COLSに含まれる列名が一つも参照されていないことを確認
2. **test_rl_feature_cols_not_in_post_race**: 6つのrl_*列名がPOST_RACE_COLSと重複しないことを検証 (自明だが二重チェック)
3. **test_build_all_produces_rl_features**: mockデータで`build_all()`を実行し、6つのrl_*列が全て出力に含まれ、値がNaNでないことを確認

**manifest再生成**: `freeze_feature_manifest.py`を実行し、data/feature_freeze_manifest.jsonを再生成。全12モデルのfeature_countとSHA256が更新:
- AbilityModel: 97 features (95+2)
- MarketModel: 9 features (7+2)
- RegimeDetector: 10 features (8+2)
- 他9モデルも全て更新

## Verification Results

```
tests/test_race_level_features.py — 8 passed
tests/test_post_race_leakage.py — 7 passed (4 existing + 3 new)
Total: 15 passed, 0 failed
```

```
freeze_feature_manifest.py — 12 models updated, overall SHA256: 003189225d7b...
```

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- src/features/feature_engine.py: FOUND (compute_race_level_features in build_all and build_features)
- tests/test_post_race_leakage.py: FOUND (TestRaceLevelFeatures class added)
- data/feature_freeze_manifest.json: FOUND (regenerated with 12 models)
- Commit 690c5c6: FOUND
- Commit 3b9d170: FOUND
