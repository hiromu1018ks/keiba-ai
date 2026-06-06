---
phase: 52-shared-feature-builder-consistency
plan: 01
subsystem: features
tags: [feature-builder, manifest, pit-registry, consistency]
dependency_graph:
  requires: [PLN-01]
  provides: [FeatureBuilder, FeatureManifest, FeatureState, FeatureBuildResult, PITModuleRegistry]
  affects: [src/features/]
tech_stack:
  added: []
  patterns: [frozen-dataclass, SHA256-hash-manifest, PIT-contract-registry]
key_files:
  created:
    - src/features/feature_manifest.py
    - src/features/feature_builder.py
    - src/features/pit_registry.py
    - tests/test_feature_manifest.py
    - tests/test_feature_builder.py
  modified: []
decisions:
  - FeatureBuildResult.frame の __eq__/__hash__ をカスタム実装 (manifest ベース) にし DataFrame 比較を回避
  - PITModuleRegistry は SireFeatures/RecordFeatures を max_date_column=None で登録 (設計上 PIT 安全)
metrics:
  duration: 848s
  completed: 2026-06-06T04:14:17Z
  tasks: 2
  files: 5
  tests: 21
---

# Phase 52 Plan 01: FeatureBuilder コアクラス新設 Summary

FeatureManifest/FeatureState/FeatureBuildResult dataclass 群と PITModuleRegistry を新設し、
13エンリッチメントモジュールを _train_submodel と同一順序で実行する FeatureBuilder クラスを構築。
BT/PT/Train の3コピー特徴量構築分岐を統一する基盤を提供。

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | FeatureManifest/FeatureState/FeatureBuildResult + PITModuleRegistry 新設 | `88a4ab7` | `src/features/feature_manifest.py`, `src/features/pit_registry.py`, `tests/test_feature_manifest.py` |
| 2 | FeatureBuilder クラス新設 (build_for_training/build_for_inference) | `c38c014` | `src/features/feature_builder.py`, `tests/test_feature_builder.py` |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FeatureBuildResult の field(eq=False) 互換性**
- **Found during:** Task 1
- **Issue:** Python 3.11 の `dataclasses.field()` は `eq` パラメータをサポートしない
- **Fix:** `__eq__` と `__hash__` をカスタム実装し、manifest のみで等価性判定
- **Files modified:** `src/features/feature_manifest.py`

None -- 計画通りに実行完了。

## Verification Results

```
21 tests passed (15 test_feature_manifest + 6 test_feature_builder)
ruff check: All checks passed
Import check: All imports OK
```

## Key Decisions

1. **FeatureBuildResult の等価性**: `frame` は DataFrame の等価性比較が重いため、`__eq__` では `manifest` のみを比較。`__hash__` も `manifest` ベース。これにより同じ特徴量スキーマなら等価と判定される。
2. **PIT 契約の None 扱い**: SireFeatures と RecordFeatures は事前計算済みキャリア統計/静的データを使用するため `max_date_column=None` とし、PIT 検証をスキップ。
3. **FeatureState.from_submodel_set() の fail-fast**: `track_stats` が `None` の場合に `ValueError` を送出。メッセージに「TRN-04」を含め、Phase 51 の要件を明示。

## Threat Flags

特になし。計画の `<threat_model>` 通りの対応を実装済み:
- T-52-01: SHA256 ハッシュが column_names/dtypes/version をカバー
- T-52-02: build_for_inference が POST_RACE 列を除去
- T-52-03: PITModuleRegistry が max_date < prediction_date を検証

## Known Stubs

なし。全ての公開 API はテスト済み。
