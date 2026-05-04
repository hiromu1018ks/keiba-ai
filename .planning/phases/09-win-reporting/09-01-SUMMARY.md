---
phase: 09-win-reporting
plan: 01
subsystem: backtest-reporting
tags: [win-betting, diagnostics, odds-bands, regime-analysis, html-report, tdd]
dependency_graph:
  requires: []
  provides: [win-bet-history-fields, regime-stats, odds-multiplier-bands, ai-diagnostics-json, win-cli-output, win-html-sections]
  affects: [src/backtest/engine.py, src/backtest/report.py, src/backtest/templates/report.html, scripts/run_backtest.py, tests/test_backtest_report.py]
tech_stack:
  added: []
  patterns: [_band_stats helper, defaultdict aggregation, Jinja2 conditional sections, TDD RED/GREEN cycle]
key_files:
  created: []
  modified:
    - src/backtest/engine.py
    - src/backtest/report.py
    - src/backtest/templates/report.html
    - scripts/run_backtest.py
    - tests/test_backtest_report.py
decisions:
  - オッズ倍率帯を4区分 (1.0-3.0/3.0-10.0/10.0-30.0/30.0+) に設定。JRA控除率25%と実用性を考慮
  - regime フィールドは全betting_targetモードでbet_historyに保存 (RESEARCH推奨に従い)
  - save_ai_diagnostics() は5ベット未満のバンドを統計的有意性判定で除外
  - _compute_condition_stats の戻り値に常におdds_multiplier_bands/regime_bandsキーを含め、place時は空リスト
metrics:
  duration_minutes: 8
  completed_date: 2026-05-04
  tasks_completed: 2
  files_modified: 5
  tests_added: 5
  tests_passing: 25
---

# Phase 09 Plan 01: Win Bet History + ROI Diagnostics + Odds Band Analysis Summary

Win bet history の包括的フィールド拡張 + Regime別/オッズ倍率帯別ROI診断 + AI分析用JSON + HTML/CLI表示拡張をTDDで実装。place/wideモードの後方互換を維持。

## Changes Made

### Task 1: bet_history win-specific フィールド追加 (TDD RED + GREEN)

**engine.py** の bet_history dict 構築部に7つのwin-specificフィールドを追加:
- `win_selection_ev` / `win_selection_edge` / `win_selection_prob`: WinSelectionGateの評価値
- `win_gate_score`: ゲートスコア(ランキング指標)
- `conformal_confidence_score`: Conformal信頼性スコア
- `tanoddslow`: 確定単勝オッズ (bet.oddsとは別フィールド)
- `regime`: 市場状態 (aggressive/conservative/collapsed)

テスト追加: `TestComputeRegimeStats` (2テスト), `TestComputeConditionStatsWin` (3テスト)

### Task 2: BacktestReportGenerator 拡張 + HTML + CLI

**report.py**:
- `_compute_condition_stats()` に `betting_target` 引数追加。win時にodds_multiplier_bands + regime_bandsを計算
- `_compute_regime_stats()` メソッド追加: regime別ROI/的中率/ベット数集計
- `save_ai_diagnostics()` メソッド追加: AI分析用JSON (highlights, best/worst band, monthly trend, over/underperforming conditions)
- `generate()` と `MultiYearReportGenerator.generate()` に betting_target 伝播

**report.html**:
- Regime別テーブル + オッズ倍率帯テーブル追加 (条件分析セクション内)
- Win モード用 KPI cards (ベット数/投資額/払戻額) 追加
- 全セクション `{% if betting_target == "win" %}` でガード

**run_backtest.py**:
- `display_single_year_result()` に `betting_target` 引数追加、win時に的中率/平均オッズ/Edge統計をCLI表示
- `gen.generate()` と `gen.save_ai_diagnostics()` に betting_target 伝播

## TDD Gate Compliance

- RED commit: `b1a276d` - test(09-01): add failing tests for regime stats and win condition stats
- GREEN commit (engine.py): `6adfd45` - feat(09-01): add win-specific fields to bet_history in engine.py
- GREEN commit (report+html+cli+tests): `f613158` - feat(09-01): extend BacktestReportGenerator with win diagnostics + HTML + CLI

全ゲート通過済み。

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] テストの place モード判定を "not in" から "empty list" に修正**
- **Found during:** Task 2 GREEN phase
- **Issue:** `_compute_condition_stats` の戻り値が常に `odds_multiplier_bands` キーを含むため、`"odds_multiplier_bands" not in result` が常に False になった
- **Fix:** テストを `assert result["odds_multiplier_bands"] == []` に変更。placeモードでは空リストが返るのが正しい動作
- **Files modified:** tests/test_backtest_report.py
- **Commit:** f613158

### Planned Deviations

None - plan executed exactly as written.

## Verification Results

```
25 passed, 1 skipped in 1.65s
grep "win_selection_ev" src/backtest/engine.py: 2
grep "_compute_regime_stats" src/backtest/report.py: 3
grep "odds_multiplier_bands" src/backtest/report.py: 6
grep "save_ai_diagnostics" src/backtest/report.py: 1 (+ 1 in run_backtest.py)
grep "betting_target" src/backtest/report.py: 11
grep "regime" src/backtest/templates/report.html: 3
grep "odds_multiplier" src/backtest/templates/report.html: 2
grep "betting_target" src/backtest/templates/report.html: 2
ruff check: passed (all Task 2 files)
```

## Commits

| Commit | Message |
|--------|---------|
| b1a276d | test(09-01): add failing tests for regime stats and win condition stats |
| 6adfd45 | feat(09-01): add win-specific fields to bet_history in engine.py |
| f613158 | feat(09-01): extend BacktestReportGenerator with win diagnostics + HTML + CLI |

## Self-Check: PASSED

- All 6 files verified as existing on disk
- All 3 commits (b1a276d, 6adfd45, f613158) verified in git log
- 25 tests passed, 0 failed
