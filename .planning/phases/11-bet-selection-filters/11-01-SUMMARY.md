---
phase: 11-bet-selection-filters
plan: 01
subsystem: betting
tags: [odds-band-filter, ev-lower-filter, regime-detector, lightgbm, conformal]

# Dependency graph
requires:
  - phase: existing
    provides: "race_predictor.py get_win_candidates(), regime_detector.py get_strategy_params()"
provides:
  - "OddsBandFilter クラス (calibrate/filter/excluded_bands)"
  - "EV_lower_win_corrected >= 1.0 フィルター in get_win_candidates()"
  - "RegimeDetector COLLAPSED skip=True フラグ"
affects: [11-02, backtest-engine]

# Tech tracking
tech-stack:
  added: []
  patterns: [odds-band-filtering, ev-lower-bound-filter, regime-skip-flag]

key-files:
  created:
    - src/betting/odds_band_filter.py
    - tests/test_odds_band_filter.py
  modified:
    - src/backtest/race_predictor.py
    - src/models/regime_detector.py
    - tests/test_race_predictor.py
    - tests/test_regime_detector.py

key-decisions:
  - "EV_lower NaN フォールバック: fillna(1.0) で >= 1.0 を通す (edge>0 のみで判定)"
  - "OddsBandFilter BANDS 境界は report.py _band_stats と同一"

patterns-established:
  - "OddsBandFilter: calibrate(トレーニングデータ) → filter(推論候補) の2段階パターン"
  - "EV_lower フィルター: 列存在チェック → fillna(1.0) → mask &= のパターン"

requirements-completed: [BSEL-01, BSEL-03]

# Metrics
duration: 4min
completed: 2026-05-04
---

# Phase 11 Plan 01: Bet Selection Filters Summary

**OddsBandFilter クラス (ROI<100%バンド除外), EV_lower>=1.0 フィルター, RegimeDetector COLLAPSED skip=True の3コンポーネントを実装**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-04T14:12:43Z
- **Completed:** 2026-05-04T14:16:51Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- OddsBandFilter クラス: calibrate() でトレーニング期間ROI計算、filter() で除外バンド候補を除去
- get_win_candidates() に EV_lower_win_corrected >= 1.0 フィルターを追加 (NaN はフォールバック)
- RegimeDetector COLLAPSED 分岐に skip=True を追加
- 全12新規テスト通過 (OddsBandFilter: 6, EV_lower: 4, RegimeDetector: 2)

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: OddsBandFilter テスト作成** - `40e8c56` (test)
2. **Task 1 GREEN: OddsBandFilter クラス実装** - `4cefe4b` (feat)
3. **Task 2 RED: EV_lower + RegimeDetector テスト作成** - `ad52c1f` (test)
4. **Task 2 GREEN: EV_lower フィルター + skip=True 実装** - `4185db6` (feat)

_Note: TDD tasks have multiple commits (test -> feat)_

## Files Created/Modified
- `src/betting/odds_band_filter.py` - OddsBandFilter クラス (calibrate/filter/excluded_bands)
- `tests/test_odds_band_filter.py` - 6テスト (空リスト/ROI除外/ROI保持/filter/未calibrate/excluded_bands)
- `src/backtest/race_predictor.py` - get_win_candidates() に EV_lower >= 1.0 フィルター追加
- `src/models/regime_detector.py` - COLLAPSED 分岐に "skip": True 追加
- `tests/test_race_predictor.py` - 4テスト追加 (EV_lower 除外/NaN/列なし/保持)
- `tests/test_regime_detector.py` - 2テスト追加 (skip=True/skipなし)

## Decisions Made
- EV_lower NaN フォールバック: fillna(1.0) で >= 1.0 チェックを通す。これにより EV_lower が NaN の場合、既存の edge>0 のみで判定される
- OddsBandFilter のバンド境界 [1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+] は report.py の _band_stats と完全に同一

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- ruff F401 (未使用インポート `build_win_selection_ev`) が race_predictor.py に既存していたが、今回の変更ではないためスコープ外として扱った

## Next Phase Readiness
- Plan 02 はこれら3コンポーネントの engine.py への統合が可能
- OddsBandFilter は calibrate() 呼び出しが必要 (トレーニング期間データの受け渡し)
- RegimeDetector skip=True は engine.py でチェックして使用する

---
*Phase: 11-bet-selection-filters*
*Completed: 2026-05-04*

## Self-Check: PASSED

- All 6 created/modified files verified present
- All 4 commits verified in git log (40e8c56, 4cefe4b, ad52c1f, 4185db6)
- 1175 tests collected with no collection errors
- All 32 plan-specific tests pass
