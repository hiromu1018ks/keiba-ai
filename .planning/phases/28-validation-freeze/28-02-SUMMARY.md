---
phase: 28-validation-freeze
plan: 02
subsystem: validation
tags: [backtest, multi-year, feature-importance, roadmap, roi]

# Dependency graph
requires:
  - phase: 28-01
    provides: pytest回帰確認 + 特徴量凍結manifest
  - phase: 27-feature-interactions
    provides: FEATURE_COLS最終状態
provides:
  - マルチ年度BT結果 (2023/2024/2025)
  - feature_importance_report.json (14モデルgain重要度)
  - ROADMAP.md更新 (v1.6完了記録)
affects: [v1.6-close]

# Tech tracking
tech-stack:
  added: []
  patterns: [multi-year-backtest, gain-importance-analysis]

key-files:
  created:
    - data/backtest/multi_year_result.json
    - data/backtest/multi_year_report.html
    - data/backtest/multi_year_bet_history.json
    - data/validation/multi_year_validation_report.json
    - feature_importance_report.json
    - feature_importance_report.csv
  modified:
    - .planning/ROADMAP.md
    - .planning/STATE.md

key-decisions:
  - "strategy_manifest.json不存在のためデフォルト戦略パラメータでBT実行 (blockerではない)"
  - "ROI 100%超え未達だが+1.3pp改善を記録"

requirements-completed: [integrated-backtest-roi, feature-importance-recalc, roadmap-documentation]

# Metrics
duration: 5h
completed: 2026-05-17
---

# Phase 28 Plan 02: マルチ年度BT + Feature Importance + ROADMAP更新 Summary

**3年マルチ年度BT (ROI 85.7%) + Feature importance再計算 + v1.6完了ドキュメント更新**

## Performance

- **Duration:** 5 hours
- **Started:** 2026-05-17T07:48:00Z
- **Completed:** 2026-05-17T12:22:00Z
- **Tasks:** 4 (auto x2, checkpoint x1, auto x1)

## Accomplishments
- マルチ年度BT (2023/2024/2025) 実行完了、全体ROI 85.7%
- Feature importance 14モデル再計算 (gain重要度)
- Phase 26-27新特徴量がimportance reportに含まれることを確認
- ROADMAP.md + STATE.md更新、v1.6マイルストーン完了記録

## Backtest Results

| Year | ROI | Bets | Profit |
|------|-----|------|--------|
| 2023 | 87.6% | 2,256 | -28,010 |
| 2024 | 76.1% | 2,389 | -57,180 |
| 2025 | 93.6% | 2,403 | -15,340 |
| **Overall** | **85.7%** | **7,048** | **-100,530** |

**v1.5: 84.4% -> v1.6: 85.7% (+1.3pp improvement). 100% target not reached.**

## Feature Importance Highlights

- 14モデルのgain重要度を計算
- stage1_dirt top 3: blood_prize_log, blood_total_wr, norm_finish_logit_avg_race_rank
- Phase 26-27新特徴量: sire_surface_wr (高重要度), sire_wr, sire_distance_wr, sire_prize_avg
- permutation重要度は多くのモデルでターゲット列メタデータ欠損のためスキップ

## Decisions Made
- strategy_manifest.json不存在のためデフォルトパラメータで実行 (D-02)
- ROI 100%超え未達だが+1.3pp改善をD-05フォーマットで記録

## Issues Encountered

None — BT正常完了、エラーなし。

## User Setup Required

None.

## Next Phase Readiness
- v1.6マイルストーン完結
- 次マイルストーン (v1.7) の検討が必要

## Self-Check: PASSED

- FOUND: data/backtest/multi_year_result.json
- FOUND: feature_importance_report.json
- FOUND: .planning/ROADMAP.md (updated)
- FOUND: .planning/STATE.md (updated)

---
*Phase: 28-validation-freeze*
*Completed: 2026-05-17*
