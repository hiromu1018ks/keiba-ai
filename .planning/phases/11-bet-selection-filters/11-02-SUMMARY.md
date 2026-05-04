---
phase: 11-bet-selection-filters
plan: 02
subsystem: backtest
tags: [odds-band-filter, regime-skip, exclusion-stats, bet-count-guard, ev-filter]

# Dependency graph
requires:
  - phase: 11-01
    provides: "OddsBandFilter, EV_lower filter, RegimeDetector skip=True"
provides:
  - "BacktestEngine COLLAPSED skip (race-level) + OddsBandFilter 統合 (candidate-level)"
  - "BacktestResult 除外フィールド (n_collapsed_skipped, n_ev_excluded, n_odds_band_excluded, exclusion_stats)"
  - "Bet count guard WARNING (bets/year < 1000)"
  - "レポート/AI診断除外統計セクション"
affects: [backtest-engine, backtest-report, run-backtest-script]

# Tech tracking
tech-stack:
  added: []
patterns: [filter-chain-pipeline, exclusion-counter, bet-count-guard]

key-files:
  created: []
  modified:
    - src/backtest/engine.py
    - src/backtest/report.py
    - tests/test_backtest_engine.py
    - tests/test_backtest_report.py

key-decisions:
  - "recent_stats_list 蓄積を COLLAPSED skip の continue より前に移動 (Pitfall 3: レジーム遷移に統計が必要)"
  - "OddsBandFilter は betting_target='win' の場合のみ初期化 (place/wide は対象外)"
  - "bet count guard は WARNING のみ、自動緩和なし (D-10)"

patterns-established:
  - "フィルター適用順序: COLLAPSED skip (race-level) -> EV filter (candidate-level) -> OddsBandFilter (candidate-level)"
  - "除外カウンター + exclusion_stats dict パターン"

requirements-completed: [BSEL-01, BSEL-02, BSEL-03]

# Metrics
duration: 9min
completed: 2026-05-04
---

# Phase 11 Plan 02: Engine Filter Integration Summary

**COLLAPSED regime skip + OddsBandFilter + 除外カウンター + bet count guard を BacktestEngine に統合し、レポートに除外統計表示を追加**

## Performance

- **Duration:** 9 min
- **Started:** 2026-05-04T14:20:43Z
- **Completed:** 2026-05-04T14:30:40Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- BacktestResult dataclass に除外フィールド追加 (n_collapsed_skipped, n_ev_excluded, n_odds_band_excluded, exclusion_stats)
- COLLAPSED regime skip を recent_stats_list 蓄積後に実行 (Pitfall 3 対策)
- OddsBandFilter を engine に統合 (calibrate + filter)、候補選択後に除外適用
- training_bet_history パラメータで OddsBandFilter キャリブレーション
- Bet count guard WARNING (< 1000 bets/year)
- レポート generate() + save_ai_diagnostics() に除外統計セクション追加
- 9新規テスト追加 (engine: 6, report: 3)

## Task Commits

Each task was committed atomically:

1. **Task 1: BacktestResult拡張 + COLLAPSED skip + 除外カウンター** - `7eaed4c` (feat)
2. **Task 2: OddsBandFilter統合 + training_bet_history + bet count guard** - `ad28104` (feat)
3. **Task 3: レポート除外統計表示 + AI診断拡張** - `6bb0b1f` (feat)

## Files Created/Modified
- `src/backtest/engine.py` - BacktestResult dataclass 拡張、COLLAPSED skip、OddsBandFilter 統合、bet count guard
- `src/backtest/report.py` - generate() exclusion_stats、save_ai_diagnostics() exclusion セクション
- `tests/test_backtest_engine.py` - TestBetSelectionFilters 6テスト
- `tests/test_backtest_report.py` - TestExclusionStatsReporting 3テスト

## Decisions Made
- recent_stats_list 蓄積を COLLAPSED skip の continue より前に移動: レジーム遷移のためにCOLLAPSEDレースでも統計蓄積が必要 (Pitfall 3 対策)
- OddsBandFilter は betting_target='win' の場合のみ初期化: place/wide モードではオッズバンドフィルタリングが適用外
- bet count guard は WARNING のみ: 自動緩和なし、Phase 13 でのパラメータ調整を推奨 (D-10)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] テスト3のモック不足修正**
- **Found during:** Task 1 (test_collapsed_skip_increments_counter)
- **Issue:** extract_pre_post_odds モックなし → レースループに入らず n_collapsed_skipped=0
- **Fix:** extract_pre_post_odds + FeatureEngine + SubModelManager のフルモックパイプラインを追加
- **Files modified:** tests/test_backtest_engine.py
- **Verification:** テスト通過確認
- **Committed in:** 7eaed4c (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** テスト修正のみ。実装は計画通り。

## Issues Encountered
- 既存の ruff エラー (F841 kumi5 変数、E501 行長超過) は今回の変更ではないためスコープ外として扱った

## Next Phase Readiness
- BacktestEngine に全フィルターが統合済み、Phase 12 (Stake Sizing) で賭け金最適化が可能
- run_backtest.py の training_bet_history 受け渡しは Phase 13 (Tuning) で対応
- 1184テスト収集エラーなし、86テスト (engine+report) 全通過

---
*Phase: 11-bet-selection-filters*
*Completed: 2026-05-04*

## Self-Check: PASSED

- All 4 created/modified files verified present
- All 3 commits verified in git log (7eaed4c, ad28104, 6bb0b1f)
- 1184 tests collected with no collection errors
- 86 plan-specific tests pass (51 engine + 35 report)
