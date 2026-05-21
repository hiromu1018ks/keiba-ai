---
phase: 18-validation-freeze
plan: 02
subsystem: backtest
tags: [validation-report, roi-judgment, cause-analysis, d-06, d-07, d-08, d-11]

# Dependency graph
requires:
  - phase: 18-01
    provides: [pfp-dual-verification-in-engine, manifest-path-wiring]
provides:
  - validation-report-json-generation (generate_validation_report)
  - roi-judgment-logic (evaluate_validation: PASS/FAIL by D-06)
  - cause-analysis-report (generate_cause_analysis: 5-item breakdown for ROI<100%)
  - engine-run-validation-output (data/validation/validation_report.json)
  - multi-year-validation-output (data/validation/multi_year_validation_report.json)
affects: [src/backtest/engine.py, scripts/run_backtest.py]

# Tech tracking
tech-stack:
  added: []
  patterns: [validation-report-in-engine-run, cause-analysis-5-item-breakdown, yearly-breakdown-from-bet-history]

key-files:
  created:
    - src/backtest/validation_report.py
    - tests/test_backtest_validation.py
  modified:
    - src/backtest/engine.py
    - scripts/run_backtest.py

key-decisions:
  - "PFP verify resultをpfp_result変数に保持し、generate_validation_report()に渡す"
  - "検証レポート出力はtry/exceptでラップしバックテスト自体を壊さない"

patterns-established:
  - "validation_report pattern: engine.run()末尾でBacktestResultを変数に格納→report生成→JSON出力→return"
  - "cause_analysis pattern: .get()アクセスで全フィールド欠損を防御(Pitfall 4)"

requirements-completed: [VAL-01, VAL-02]

# Metrics
duration: 4min
completed: "2026-05-06T23:01:35Z"
---

# Phase 18 Plan 02: 検証結果JSON生成 + 原因分析レポート Summary

検証結果JSON生成モジュール(validation_report.py)を作成し、BacktestEngine.run()完了時にdata/validation/に検証レポートを出力する統合を完了。ROI<100%時の5項目原因分析(オッズバンド別/レジーム別/EV診断/ベット数/芝ダート別)を自動生成。

## Performance

- **Duration:** 4min
- **Started:** 2026-05-06T22:57:25Z
- **Completed:** 2026-05-06T23:01:35Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- validation_report.py新規作成: 3公開関数(evaluate_validation/generate_validation_report/generate_cause_analysis)
- D-06基準(ROI>1.0 and bets>=100)でPASS/FAIL判定が正しく動作
- engine.py run()末尾に検証レポートJSON出力(data/validation/validation_report.json)を統合
- run_backtest.py マルチ年度モードに全体検証レポート出力(data/validation/multi_year_validation_report.json)を統合
- ROI<=100%時にcause_analysis(5項目)が自動生成される(D-11)
- 8テスト全てPASS + 既存59テスト回帰なし

## Task Commits

Each task was committed atomically:

1. **Task 1: 検証結果JSON生成モジュール + 原因分析レポート生成 (TDD)** - `1fdb475` (test)
2. **Task 2: engine.py + run_backtest.pyに検証レポート出力を統合** - `7a156ae` (feat)

## Files Created/Modified

- `src/backtest/validation_report.py` - 検証結果JSON生成 + 原因分析レポート生成モジュール(3公開関数)
- `tests/test_backtest_validation.py` - VAL-01/VAL-02検証テスト(8テスト、全mockベース)
- `src/backtest/engine.py` - run()末尾に検証レポート出力統合(BacktestResult変数格納→report生成→JSON出力)
- `scripts/run_backtest.py` - マルチ年度全体検証レポート出力追加

## Decisions Made

- PFP verifyの結果をpfp_result変数に保持し、generate_validation_report()の引数として渡すことで、検証レポートにPFP情報を含める
- 検証レポート出力はtry/exceptでラップし、レポート生成失敗がバックテスト自体の実行を止めない設計
- engine.pyのrun()でBacktestResultを直接returnしていた箇所を変数格納にリファクタリングし、report生成とreturnで共有

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Self-Check: PASSED

All files verified present. All commits verified in git log.

## Next Phase Readiness

- Phase 18 Plan 01+02両方完了。PFP二重検証 + 検証レポート生成が統合済み
- Human UATで`run_backtest.py --ensemble --strategy-manifest PATH`を実行し、data/validation/に検証レポートが出力されることを確認する必要がある
- PostgreSQL環境が必要(Human UAT)

---
*Phase: 18-validation-freeze*
*Completed: 2026-05-06*
