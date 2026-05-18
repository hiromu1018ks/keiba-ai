---
phase: 31-race-level-aggregation-features
plan: 01
subsystem: features/models
tags: [feature-engineering, market-structure, feature-promotion]
dependency_graph:
  requires: []
  provides: [compute_race_level_features, FEATURE_COLS-hhi-skewness]
  affects: [feature_engine.py, all-models]
tech_stack:
  added: []
  patterns: [groupby-race-level-aggregation, tuple-unpacking-groupby-apply]
key_files:
  created:
    - src/features/race_level_features.py
    - tests/test_race_level_features.py
  modified:
    - src/models/stage1_ability_model.py
    - src/models/market_model.py
    - src/models/regime_detector.py
    - src/models/place_ability_model.py
    - src/models/race_quality_screener.py
    - src/models/wide_two_stage_model.py
    - src/models/two_stage_return_model.py
    - src/models/ev_correction_model.py
    - tests/test_post_race_leakage.py
decisions:
  - tuple返却パターンでgroupby.apply結果をマップ (dictパターンはpandas単一グループ時に失敗)
  - AbilityModel docstringを更新: Rule 1の例外として市場構造指標を明記 (D-06)
metrics:
  duration: 759s
  completed: "2026-05-18"
  tasks: 2
  files: 10
---

# Phase 31 Plan 01: Race-Level Aggregation Features Module Summary

シャノンエントロピー/オッズ散らばり/人気間ギャップ等の6特徴量モジュール新規作成 + implied_prob_hhi/odds_skewnessの全12モデルFEATURE_COLS昇格

## Completed Tasks

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | race_level_features.py + tests (TDD) | `2538f3a` (RED), `463026d` (GREEN) | src/features/race_level_features.py, tests/test_race_level_features.py |
| 2 | FEATURE_COLS promotion (12 models) | `877414e` | 8 model files + test_post_race_leakage.py |

## Implementation Details

### Task 1: race_level_features.py

`compute_race_level_features(df)` を実装。6つのrl_*特徴量を計算:

- **rl_log_odds_entropy** (RLF-01): インプライド確率のシャノンエントロピー `-sum(p * log(p))`。market_bias_features.pyの`_calc_entropy`パターンを踏襲
- **rl_odds_dispersion** (RLF-02): tanodds.groupby("race_id").transform("std")
- **rl_top3_odds_gap** (RLF-03): 3番人気 - 1番人気のオッズ差。2頭以下はNaN
- **rl_top1_odds** (RLF-04): 1番人気のtanodds値を全馬にブロードキャスト
- **rl_favorite_rank_gap** (RLF-05): log(odds_fav2 / odds_fav1)。D-08対数オッズ差
- **rl_n_horses** (RLF-06): field_sizeを優先、0/NaN時は有効オッズ数で補完

設計上の特徴:
- `_compute_for_single_race()`: race_idなしの単一レース用 (build_features()パリティ)
- `_compute_for_multi_race()`: groupby("race_id")版 (build_all()パス)
- groupby.applyにはtuple返却パターンを採用 (dictは単一グループ時にスカラーに unpackされて失敗)
- 入力DataFrameは一切変更しない (df.copy()で保護)

### Task 2: FEATURE_COLS Promotion

| Model | implied_prob_hhi | odds_skewness |
|-------|-----------------|---------------|
| AbilityModel | added | added |
| MarketModel | added | added |
| RegimeDetector | added | added |
| PlaceAbilityModel | added | added |
| RaceQualityScreener | added | added |
| WideTwoStageModel.SHARED | added | added |
| WinTwoStageModel | added | already present |
| PlaceTwoStageModel.HIT | added | already present |
| PlaceTwoStageModel.RETURN | added | already present |
| EVCorrectionModel | already present | added |
| PlaceEVCorrectionModel | already present | added |
| ConformalEVModel | already present | already present |

全12モデルに重複なしで両特徴量が含まれることを確認。

## Verification Results

```
tests/test_race_level_features.py — 8 passed
tests/test_post_race_leakage.py — 4 passed
Total: 12 passed, 0 failed
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] groupby.apply の dict返却がpandas単一グループ時に失敗**
- **Found during:** Task 1 GREEN phase
- **Issue:** `_rank_features` が dict を返すと、groupby.apply が単一グループの Series を返した際に `.map(lambda x: x["fav1"])` が TypeError になる (float は subscriptable ではない)
- **Fix:** dict の代わりに tuple を返すように変更し、`x[0]`, `x[1]`, `x[2]` でアクセス。compute_flb_slope() の `_race_shape` パターンを踏襲
- **Files modified:** src/features/race_level_features.py
- **Commit:** `463026d`

**2. [Rule 3 - Blocking] test_post_race_leakage のテストDataFrameに odds_skewness 列が不足**
- **Found during:** Task 2 verification
- **Issue:** EVCorrectionModel.FEATURE_COLS に odds_skewness を追加したため、テスト内の DataFrame に同列が存在せず KeyError 発生
- **Fix:** テスト DataFrame に `"odds_skewness": [0.5] * 3` を追加
- **Files modified:** tests/test_post_race_leakage.py
- **Commit:** `877414e`

## TDD Gate Compliance

- RED gate: `2538f3a` — 全8テスト失敗 (ModuleNotFoundError) を確認
- GREEN gate: `463026d` — 全8テスト通過を確認
- No REFACTOR gate needed (implementation is clean)

## Self-Check: PASSED

- src/features/race_level_features.py: FOUND
- tests/test_race_level_features.py: FOUND
- src/models/stage1_ability_model.py: FOUND (modified)
- src/models/market_model.py: FOUND (modified)
- src/models/regime_detector.py: FOUND (modified)
- Commit 2538f3a: FOUND
- Commit 463026d: FOUND
- Commit 877414e: FOUND
