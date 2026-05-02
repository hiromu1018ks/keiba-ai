---
phase: 03-selection-gate-confidence-betting
plan: 02
subsystem: betting
tags: [edge-threshold, jra-takeout, regime-detector, gate-keeper, meta-switcher]

# Dependency graph
requires:
  - phase: 03-selection-gate-confidence-betting
    provides: "WinSelectionGate + RegimeDetector/MetaSwitcher/GateKeeper既存実装"
provides:
  - "JRA控除率25%考慮のedge_threshold更新 (RegimeDetector/MetaSwitcher/GateKeeper)"
  - "レジーム別閾値対応表: AGGRESSIVE=0.05, CONSERVATIVE=0.06~0.07, COLLAPSED=0.09~0.10"
  - "GateKeeperデフォルト閾値0.04"
affects: [phase-4-walk-forward-validation, backtest]

# Tech tracking
tech-stack:
  added: []
  patterns: ["JRA控除率25%を上回るedgeのみベット対象とする安全マージン設計"]

key-files:
  created: []
  modified:
    - src/models/regime_detector.py
    - src/betting/meta_switcher.py
    - src/betting/gate_keeper.py
    - tests/test_meta_switcher.py
    - tests/test_gate_keeper.py

key-decisions:
  - "RegimeDetector/MetaSwitcher/GateKeeper全て+0.01引き上げで統一 (微小変更でベット数激減リスク回避)"
  - "MetaSwitcherのCONSERVATIVE/COLLAPSEDがRegimeDetectorより+0.01高い関係を維持"
  - "WinStrategyのKelly計算は変更不要 (賭け金計算のみでベット可否はGateが担当)"

patterns-established:
  - "Phase Nコメント付き閾値更新: コメントにPhase番号と理由を記載し追跡性を確保"

requirements-completed: [BETT-01]

# Metrics
duration: 5min
completed: 2026-05-02
---

# Phase 3 Plan 02: JRA控除率考慮edge_threshold更新 Summary

**JRA控除率25%を考慮し、RegimeDetector/MetaSwitcher/GateKeeperのedge_thresholdを全て+0.01引き上げてベット安全性マージンを確保**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-02T15:12:40Z
- **Completed:** 2026-05-02T15:17:40Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- RegimeDetectorの3レジームedge_thresholdを+0.01引き上げ (AGGRESSIVE=0.05, CONSERVATIVE=0.06, COLLAPSED=0.09)
- MetaSwitcherの3レジームedge_thresholdを+0.01引き上げ (AGGRESSIVE=0.05, CONSERVATIVE=0.07, COLLAPSED=0.10)
- GateKeeperのデフォルト閾値を0.03から0.04に更新
- WinStrategyのKelly計算(edge/(odds-1), cap=25%)が変更不要であることを確認
- 全1042テスト通過 (閾値変更に伴うテスト期待値の更新含む)

## Task Commits

Each task was committed atomically:

1. **Task 1: RegimeDetector + MetaSwitcher edge_threshold更新** - `0067b8c` (feat)
2. **Task 2: GateKeeperデフォルト閾値更新 + WinStrategy確認** - `ca49df4` (feat)

## Files Created/Modified

- `src/models/regime_detector.py` - 3レジームのedge_thresholdを+0.01引き上げ (0.04->0.05, 0.05->0.06, 0.08->0.09)
- `src/betting/meta_switcher.py` - 3レジームのedge_thresholdを+0.01引き上げ (0.04->0.05, 0.06->0.07, 0.09->0.10)
- `src/betting/gate_keeper.py` - デフォルト閾値0.03->0.04 (should_bet + filter_bets)
- `tests/test_meta_switcher.py` - AGGRESSIVE閾値期待値を0.04->0.05に更新
- `tests/test_gate_keeper.py` - 境界値テストのedgeを0.03->0.04に更新

## Decisions Made

- **+0.01の微小引き上げ**: バックテストでベット数>0を確認する必要があるが、大きな変更はベット数激減リスクがあるため最小限の+0.01とした
- **MetaSwitcherの閾値差維持**: AGGRESSIVEは同値、CONSERVATIVEは+0.01、COLLAPSEDは+0.01の関係を維持し、MetaSwitcherがRegimeDetectorより一段厳しくフィルタリングする設計を保持
- **Kelly計算は不変更**: WinStrategy._calc_stake()は賭け金計算のみ担当し、ベット可否はWinSelectionGateが判定するため閾値変更の影響を受けない

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] テスト期待値の更新**
- **Found during:** Task 1 (MetaSwitcher閾値更新後のテスト実行)
- **Issue:** テストが旧値0.04をハードコードしており、閾値変更でテストが失敗
- **Fix:** test_meta_switcher.pyの期待値を0.05に更新、test_gate_keeper.pyの境界値テストedgeを0.04に更新
- **Files modified:** tests/test_meta_switcher.py, tests/test_gate_keeper.py
- **Verification:** 全1042テスト通過
- **Committed in:** 0067b8c (Task 1), ca49df4 (Task 2)

---

**Total deviations:** 2 auto-fixed (2 missing critical: テスト期待値の旧値参照)
**Impact on plan:** 閾値変更に伴うテストメンテナンスは必須作業。スコープクリープなし。

## Issues Encountered

None - 全ての変更は具体的な値の置換で、機械的に実行可能だった。

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 3の全2プラン完了 (WinSelectionGate + edge_threshold更新)
- Phase 4 (Walk-Forward Validation) で更新後の閾値によるバックテストROIを検証可能
- 更新された閾値でベット数>0となることをPhase 4で確認する必要あり

## Self-Check: PASSED

- [x] src/models/regime_detector.py exists
- [x] src/betting/meta_switcher.py exists
- [x] src/betting/gate_keeper.py exists
- [x] .planning/phases/03-selection-gate-confidence-betting/03-02-SUMMARY.md exists
- [x] Commit 0067b8c (Task 1) found in git log
- [x] Commit ca49df4 (Task 2) found in git log

---
*Phase: 03-selection-gate-confidence-betting*
*Completed: 2026-05-02*
