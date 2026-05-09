---
phase: 20-high-odds-pattern-features
plan: 01
subsystem: features
tags: [high-odds, analysis, class-trajectory, form-improvement, tdd]
dependency_graph:
  requires: []
  provides: [analyze_high_odds.py, high_odds_features.py]
  affects: []
tech_stack:
  added: []
  patterns: [Cohen's d, LightGBM TreeSHAP, EMA halflife=3]
key_files:
  created:
    - scripts/analyze_high_odds.py
    - src/features/high_odds_features.py
    - tests/test_high_odds_features.py
  modified: []
decisions:
  - _CLASS_LEVEL_MAPをインライン化 (循環参照回避)
  - _class_level_from_valuesのbug fix (三項演算子の変数参照順序)
metrics:
  duration: 257s
  completed: "2026-05-09"
  tasks: 2
  files: 3
---

# Phase 20 Plan 01: 高オッズ的中パターン分析 + 特徴量実装 Summary

Cohen's d統計プロファイリングとLightGBM TreeSHAPを組み合わせたハイブリッド分析スクリプト(HODDS-01)と、クラストラジェクトリ7特徴量(HODDS-02) + フォーム改善率2特徴量(HODDS-03)の純粋関数群をTDDで実装。

## What Was Done

### Task 1: 高オッズ的中パターン分析スクリプト (HODDS-01)

`scripts/analyze_high_odds.py` を新規作成。高オッズ帯(20倍+)の的中馬と非的中馬を比較するオフライン分析スクリプト。

- **Cohen's d 統計プロファイリング**: 的中群 vs 非的中群の全数値特徴量についてCohen's dを計算。pooled standard deviationを使用
- **LightGBM TreeSHAP**: `pred_contrib=True` で高オッズ馬限定のSHAP値を計算
- **CLI**: `--odds-threshold` (default=20.0), `--start`, `--end`, `--model-dir`, `--output`, `--surface`
- **サンプル不足警告**: 高オッズ的中 < 50件の場合に`logger.warning`出力
- **結果出力**: JSON + Markdownテーブル (上位20特徴量)

### Task 2: クラストラジェクトリ + フォーム改善率特徴量 (HODDS-02, HODDS-03) [TDD]

TDD (RED/GREEN) で実装。17テスト全通過。

**compute_class_trajectory (7特徴量):**
- `class_promotions`: 昇級回数 (diff > 0)
- `class_demotions`: 降級回数 (diff < 0)
- `class_net_change`: ネット変化 (最終 - 最初)
- `class_max_level`: 最高クラス到達レベル
- `class_level_std`: クラス分散 (std)
- `v_recovery_flag`: V字回復パターンフラグ (降級→再昇級)
- `v_recovery_duration`: 降級から再昇級までの走数

**compute_form_improvement_rate (2特徴量):**
- `time_improvement_rate`: EMA重み付けタイム(z-score)改善率
- `position_improvement_rate`: EMA重み付け着順改善率
- halflife=3 (horse_history_features.py と同一の decay = ln(2)/halflife パターン)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] _class_level_from_values 三項演算子の変数参照順序**
- **Found during:** Task 2 GREEN phase
- **Issue:** `grade = str(grade_code).strip() if not _is_nan(grade) else ""` で `grade` を定義する前に参照していた (UnboundLocalError)
- **Fix:** `_is_nan(grade)` を `_is_nan(grade_code)` に修正
- **Files modified:** src/features/high_odds_features.py
- **Commit:** 9f9f14b

## TDD Gate Compliance

- RED commit: 3a54972 (test file only, ModuleNotFoundError confirmed)
- GREEN commit: 9f9f14b (implementation, 17/17 tests passed)
- No REFACTOR needed (clean implementation)

## Commits

| Commit | Message | Files |
|--------|---------|-------|
| f75990e | feat(20-01): 高オッズ的中パターン分析スクリプト (HODDS-01) | scripts/analyze_high_odds.py |
| 3a54972 | test(20-01): add failing tests (RED) | tests/test_high_odds_features.py |
| 9f9f14b | feat(20-01): クラストラジェクトリ + フォーム改善率特徴量 (GREEN) | src/features/high_odds_features.py |

## Self-Check: PASSED

- scripts/analyze_high_odds.py: FOUND
- src/features/high_odds_features.py: FOUND
- tests/test_high_odds_features.py: FOUND
- 20-01-SUMMARY.md: FOUND
- Commit f75990e: FOUND
- Commit 3a54972: FOUND
- Commit 9f9f14b: FOUND
