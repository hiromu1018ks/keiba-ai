---
phase: 09-win-reporting
verified: 2026-05-04T12:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 0
---

# Phase 9: Win Reporting Verification Report

**Phase Goal:** ユーザーが単勝バックテスト結果のベット履歴・ROI診断・オッズバンド別内訳を確認できる
**Verified:** 2026-05-04
**Status:** passed
**Re-verification:** No (initial verification)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | バックテスト結果のJSON/bet_historyに各単勝ベットのumaban/odds/EV/edge/gate_score/confidence/結果が記録されている | VERIFIED | engine.py lines 758-843: win_selection_ev (812), win_selection_edge (817), win_selection_prob (822), win_gate_score (827), conformal_confidence_score (832), tanoddslow (837), regime (842) |
| 2 | バックテスト終了時に単勝ROI/回収率/的中率/ベット数の集計がCLI標準出力される | VERIFIED | run_backtest.py lines 226-238: ROI/stake/return/bets always printed; lines 240-256: win-specific win-rate/avg-odds/edge when betting_target=="win" |
| 3 | レポートにオッズバンド別(人気帯1-3/4-6/7+ と オッズ倍率帯)のROI内訳が表示される | VERIFIED | report.py lines 371-375: popularity_bands (1-3,4-6,7+); lines 394-403: odds_multiplier_bands (1.0-3.0,3.0-10.0,10.0-30.0,30.0+). HTML lines 196-279 renders both tables |
| 4 | HTMLレポートにregime別/月別/表面x距離別/EVバンド別の診断セクションが表示される | VERIFIED | report.html: regime table (lines 240-258), monthly (140-170), surface-distance (176-194), EV bands (218-236) |
| 5 | AI分析用JSON(改善点自動特定付き)が保存される | VERIFIED | report.py lines 90-193: save_ai_diagnostics() with highlights (best/worst band, monthly_trend, over/underperforming). run_backtest.py lines 373-380: called when betting_target=="win" |
| 6 | place/wideモードでのバックテストが既存の動作を維持する | VERIFIED | report.py line 294: default="place". _compute_condition_stats only adds win bands when betting_target=="win". Test test_odds_multiplier_bands_absent_in_place_mode verifies empty lists. All 1141 tests pass, 0 failures |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/backtest/engine.py` | bet_historyにwin-specific 7フィールド追加 | VERIFIED | win_selection_ev/edge/prob, win_gate_score, conformal_confidence_score, tanoddslow, regime at lines 811-843 |
| `src/backtest/report.py` | BacktestReportGenerator拡張 | VERIFIED | _compute_regime_stats (244-271), save_ai_diagnostics (90-193), betting_target param on generate (30), _compute_condition_stats (291-412) |
| `scripts/run_backtest.py` | display_single_year_result win出力 | VERIFIED | betting_target param (223), win section (240-256), gen.generate with betting_target (391), save_ai_diagnostics (373-380) |
| `src/backtest/templates/report.html` | regime/odds_multiplier_bandsセクション | VERIFIED | Regime table (240-258), odds multiplier bands (259-278), guarded by {% if betting_target == "win" %} |
| `tests/test_backtest_report.py` | 新規メソッドテスト | VERIFIED | TestComputeRegimeStats (539-568), TestComputeConditionStatsWin (571-634) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| engine.py | bet_history dict | win_selection_* fields extraction from horse_rows | WIRED | Lines 812-843 extract from horse_rows.iloc[0] |
| report.py | report.html | Jinja2 template.render() with condition_stats + betting_target | WIRED | report.py line 70-78: template.render(betting_target=betting_target, condition_stats=conditions) |
| run_backtest.py | report.py | gen.generate() + save_ai_diagnostics() | WIRED | run_backtest.py lines 373-380 (diagnostics), 386-391 (generate), both pass betting_target |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| save_ai_diagnostics | diagnostic dict | _compute_regime_stats + _compute_condition_stats + _compute_monthly_stats on actual bet data | Yes | FLOWING |
| report.html regime/odds tables | condition_stats | _compute_condition_stats with real band_stats aggregation | Yes | FLOWING |
| bet_history JSON | bet_history list | engine.py race loop populating each bet from horse_rows DataFrame | Yes | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Tests pass | python -m pytest tests/test_backtest_report.py -v | 25 passed, 1 skipped | PASS |
| No regressions | python -m pytest tests/ -v | 1141 passed, 2 skipped | PASS |
| Method existence | python -c "from backtest.report import BacktestReportGenerator; ..." | hasattr returns True for _compute_regime_stats and save_ai_diagnostics | PASS |
| Pattern counts | grep patterns in engine.py/report.py/html | win_selection_ev=2, _compute_regime_stats=3, odds_multiplier_bands=6, save_ai_diagnostics=1, betting_target in html=1, regime in html=3 | PASS |

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| RPT-01 | バックテスト結果のベット履歴に単勝ベットの馬番・オッズ・EV・結果を記録 | SATISFIED | engine.py lines 811-843: umaban, odds, win_selection_ev/edge/prob, tanoddslow, result fields in bet_history |
| RPT-02 | 単勝ROI・回収率・的中率・ベット数の集計診断を出力 | SATISFIED | run_backtest.py lines 226-256 CLI output; report.py save_ai_diagnostics() JSON output |
| RPT-03 | オッズバンド別のROI内訳を分析・表示 | SATISFIED | report.py odds_multiplier_bands (1.0-3.0,3.0-10.0,10.0-30.0,30.0+); popularity_bands (1-3,4-6,7+); HTML tables in report.html |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| scripts/run_backtest.py | 537 | gen.generate(all_results, all_metadata) missing betting_target arg | Warning | Multi-year report defaults to "place", win-specific sections not shown in multi-year mode. Single-year path unaffected. |

### Notes

**Multi-year betting_target gap (Warning, not blocking):** Line 537 in `run_backtest.py` calls `gen.generate(all_results, all_metadata)` without `betting_target=args.betting_target`. The `MultiYearReportGenerator.generate()` accepts `betting_target` (default="place"), so the win-specific regime/odds band sections will not appear in multi-year reports. This does not affect the single-year path (lines 386-391 correctly pass `betting_target`). The phase goal specifically targets single-year backtest results, so this is a non-blocking warning for future improvement.

### Human Verification Required

1. **Win-mode HTML report visual check**
   **Test:** Run a backtest with `--betting-target win --report` and open the generated HTML report
   **Expected:** Regime table, odds multiplier band table, and win-specific KPI values are visible and correctly formatted
   **Why human:** Visual layout and rendering cannot be verified programmatically

2. **AI diagnostics JSON content**
   **Test:** Open `data/backtest/ai_diagnostics.json` after a win backtest
   **Expected:** JSON contains highlights.best_band, highlights.worst_band, monthly_trend, overperforming/underperforming conditions
   **Why human:** Requires running actual backtest with real data

---

_Verified: 2026-05-04_
_Verifier: Claude (gsd-verifier)_
