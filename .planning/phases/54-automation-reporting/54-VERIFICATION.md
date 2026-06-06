---
phase: 54-automation-reporting
verified: 2026-06-06T22:00:00Z
status: human_needed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "CR-01: RunModeOrchestrator._predict_single_race now passes betting_target from args to select_bets() (commit 503e83b)"
    - "WR-04: _cross_validate_race dead-code if/else branch removed (commit 503e83b)"
    - "CR-02: _build_race_predictor annotated with CR-02 risk note (commit 503e83b)"
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Run --mode run --betting-target win on a live race day and verify win bets are generated"
    expected: "Orchestrator generates win-type bets (not place bets). RacePredictor.select_bets receives betting_target='win'."
    why_human: "Requires active DB connection with live race data; cannot simulate full DB interactions programmatically"
  - test: "Start --mode run, Ctrl+C mid-prediction, re-run same command"
    expected: "Re-run skips already-predicted races, continues from interrupted race. Exit code 130 on first run, 0 on completed re-run."
    why_human: "Requires real-time race timing and manual interruption timing"
  - test: "Open generated report.html in browser after a completed run"
    expected: "KPI cards display correct ROI/bankroll, target breakdown shows win/place separately, model identity footer visible, bet history table shows settlement_status and outcome badges"
    why_human: "Visual rendering quality, layout correctness, CSS styling"
  - test: "Review _compute_max_dd behavior when all bets lose (WR-01)"
    expected: "Should show 100% or near-100% drawdown, not 0.0%"
    why_human: "Need to evaluate whether the current behavior (0.0% DD for all-loss) is acceptable or requires fix"
---

# Phase 54: Automation & Reporting Verification Report

**Phase Goal:** モデル検証から精算・集計まで1コマンドで完遂し、週次集計・累積履歴・target別集計で PT の結果を正確に評価できること
**Verified:** 2026-06-06T22:00:00Z
**Status:** human_needed
**Re-verification:** Yes -- after CR-01 gap closure (commit 503e83b)

## Goal Achievement

### ROADMAP Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | --mode run で事前学習済みモデルの検証から開始し、予測→監視→精算→集計の全工程が1コマンドで実行される | VERIFIED | RunModeOrchestrator.execute() calls _ensure_schedule -> _fetch_track_conditions -> _predict_races -> _reconcile -> _aggregate_and_report. CR-01 FIXED: line 342-344 now passes betting_target from args to select_bets(). 47 tests pass. |
| 2 | 処理済みレースの再実行がスキップされ、クラッシュ後の再起動で未処理レースのみ再開する | VERIFIED | RaceProgress 5-state machine + D-08 cross-validation. Lines 256-274 skip PREDICTED/NO_BET, reprocess FAILED/PROCESSING/PENDING. |
| 3 | DB接続障害・データ欠損・モデル不整合時に非ゼロ終了コードを返す | VERIFIED | ExitCode IntEnum (8 codes, D-17), EXIT_SEVERITY ordering (D-18), determine_final_exit_code(). |
| 4 | 週次 ROI・的中率・ベット数の JSON 集計と pending/settled/won/lost を含む累積ベット履歴が出力される | VERIFIED | PaperTradingReportAggregator: aggregate_daily/weekly/by_target with _base_stats computing n_won/n_lost/n_pending. save_outputs creates daily_summary/, weekly_summary/, target_summary/ JSON. |
| 5 | Win/Place 別 ROI・的中率集計に MLflow run ID・学習期間・manifest hash が含まれる | VERIFIED | aggregate_by_target groups by bet_type with per-target _base_stats. _model_identity() extracts model_run_id/training_start/training_end/manifest_hash. HTML footer D-19. |

**Score:** 5/5 ROADMAP success criteria verified

### Re-verification: CR-01 Fix Analysis

**Commit 503e83b** modified `src/paper_trading/run_orchestrator.py` (12 insertions, 7 deletions):

1. **CR-01 FIXED:** Line 342 now extracts `betting_target = getattr(self.args, "betting_target", "place")` and passes it to `race_predictor.select_bets(result_df, bankroll, betting_target=betting_target)`. This correctly propagates the user's `--betting-target` choice instead of always defaulting to "place".

2. **WR-04 FIXED:** `_cross_validate_race` (lines 563-575) simplified to a single return statement, removing the dead-code if/else branch that had identical behavior in both paths.

3. **CR-02 ADDRESSED:** `_build_race_predictor` (lines 659-667) retained with NOTE comment documenting CR-02 risk if re-enabled, ensuring future developers are aware of the betting_target/strategy_config propagation requirement.

### PLAN 01 Must-Haves (Foundation Classes)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | ExitCode IntEnum defines 8 exit codes matching D-17 specification | VERIFIED | 8 members: SUCCESS=0, GENERAL_ERROR=1, PENDING_REMAIN=2, DB_FETCH_ERROR=3, DATA_INTEGRITY_ERROR=4, MODEL_VALIDATION_ERROR=5, REPORT_ERROR=6, SIGINT=130. Verified programmatically. |
| 2 | RaceProgress tracks per-race state with atomic JSON writes and resume support | VERIFIED | RaceState StrEnum (5 states), RaceProgress.load/mark/pending_or_failed_race_ids. Atomic writes via tempfile.mkstemp + os.replace with Windows retry. |
| 3 | ReportAggregator produces daily, weekly, and per-target statistics from bets.parquet | VERIFIED | aggregate_daily/aggregate_weekly/aggregate_by_target/aggregate_all/save_outputs. All methods read from bets.parquet. |
| 4 | All aggregation uses settled-only ROI with pending count and unsettled stake | VERIFIED | _base_stats: effective_stake = won+lost only (D-05). aggregate_daily adds pending_count, unsettled_stake, data_completeness. |
| 5 | Model identity from session_manifest appears in all aggregator output | VERIFIED | _model_identity() extracts model_run_id/training_start/training_end/manifest_hash. Included in daily, weekly, target outputs. |
| 6 | ReportAggregator reads only bets.parquet as cumulative history source | VERIFIED | _load_bets() reads self._bets_path (bets.parquet) exclusively. No JSON/CSV row-level reads. |

### PLAN 02 Must-Haves (Run Mode Orchestrator)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | --mode run executes setup -> TC fetch -> predict -> reconcile -> aggregate -> report | VERIFIED | execute() calls 5 phases in order. CR-01 FIXED: betting_target now propagated from args. |
| 2 | Crash resume: skips predicted/no_bet races, reprocesses pending/failed/processing | VERIFIED | Lines 256-274: PREDICTED/NO_BET skipped with cross-validation. FAILED/PROCESSING/PENDING reprocessed. |
| 3 | DB failure, data missing, model validation errors produce non-zero exit codes | VERIFIED | ExitCode.DB_FETCH_ERROR on DB error, ExitCode.MODEL_VALIDATION_ERROR on model error. |
| 4 | Ctrl+C sets cancellation flag, exits with code 130 | VERIFIED | _handle_sigint sets _cancelled=True. _determine_exit_code returns ExitCode.SIGINT=130 when _cancelled. |
| 5 | race_progress.json cross-validates against bets.parquet on resume (D-08) | VERIFIED | _cross_validate_race calls progress.verify_bet_ids_present. WR-04 dead code removed. |
| 6 | Pending bets at end of run produce exit code 2 (D-02) | VERIFIED | _reconcile appends ExitCode.PENDING_REMAIN when pending > 0. |
| 7 | Existing reconcile mode calls PaperTradingReportAggregator after reconciliation (D-15) | VERIFIED | scripts/run_paper_trading.py lines 1241-1274: aggregator constructed, save_outputs() called. D-16 exception handling. |
| 8 | Input snapshots include hash, parent_session_id, and source metadata fields (D-09) | VERIFIED | _save_input_snapshot adds _snapshot_hash (SHA256), _parent_session_id, _source_info columns. |

### PLAN 03 Must-Haves (Report Renderer)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | PaperTradingReport.generate() accepts Aggregator results instead of raw bets (D-12) | VERIFIED | `report.py` line 24: `generate(self, aggregate_results, bets=None)`. |
| 2 | HTML report displays ROI, hit rate, bet counts from Aggregator with new schema fields | VERIFIED | Template renders cumulative_roi, n_bets, max_dd, bankroll, hit_rate, total_return, effective_stake KPI cards. |
| 3 | HTML report shows settlement_status (pending/settled) and outcome (won/lost) per bet | VERIFIED | Template: badge-settled/badge-pending for settlement_status, badge-won/badge-lost for outcome. |
| 4 | Model identity (MLflow run ID, training period, manifest hash) shown in HTML footer | VERIFIED | Footer section with model_identity fields (D-19). |
| 5 | Old _derive_fields() and _compute_monthly_stats() methods are removed | VERIFIED | grep confirms only comment reference at line 16. Methods do not exist. |
| 6 | HTML report reads bet data from bets.parquet only (D-10) | VERIFIED | generate() accepts optional bets list; caller provides from bets.parquet. No JSON/CSV row-level duplication. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/paper_trading/exit_codes.py` | ExitCode IntEnum with EXIT_SEVERITY | VERIFIED | 39 lines, 8 ExitCode members, EXIT_SEVERITY dict, determine_final_exit_code() |
| `src/paper_trading/race_progress.py` | RaceProgress state machine with atomic writes | VERIFIED | 142 lines, RaceState StrEnum (5 states), RaceProgress class with load/mark/verify_bet_ids_present |
| `src/paper_trading/report_aggregator.py` | PaperTradingReportAggregator | VERIFIED | 309 lines, aggregate_daily/weekly/by_target/all, save_outputs with D-13 directory structure |
| `src/paper_trading/run_orchestrator.py` | RunModeOrchestrator class | VERIFIED | 662 lines, 5-phase lifecycle, crash resume, cross-validation. CR-01 FIXED: betting_target propagated. |
| `src/paper_trading/report.py` | PaperTradingReport as pure HTML renderer | VERIFIED | 235 lines, generate(aggregate_results, bets), old methods removed, D-19 footer |
| `scripts/run_paper_trading.py` | --mode run CLI, D-15 reconcile Aggregator | VERIFIED | parse_args includes "run", _handle_sigint, _run_run_mode, _run_reconcile D-15 Aggregator call |
| `tests/test_race_progress.py` | Tests for ExitCode and RaceProgress | VERIFIED | 11 tests, all passing |
| `tests/test_report_aggregator.py` | Tests for aggregator | VERIFIED | 11 tests, all passing |
| `tests/test_run_orchestrator.py` | Tests for orchestrator lifecycle and resume | VERIFIED | 13 tests, all passing |
| `tests/test_cli_run_mode.py` | CLI structure tests | VERIFIED | 4 tests, all passing |
| `tests/test_paper_trading_report.py` | Tests for new renderer | VERIFIED | 8 tests, all passing |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| run_orchestrator.py | reconciler.py | PaperReconciler.reconcile() and retry_pending() | WIRED | Lines 492-506: PaperReconciler constructed, reconcile() and retry_pending() called |
| run_orchestrator.py | race_progress.py | RaceProgress state tracking | WIRED | Line 21: import, line 240: load(), line 292: mark() |
| run_orchestrator.py | report_aggregator.py | PaperTradingReportAggregator.save_outputs() | WIRED | Lines 527-534: aggregator constructed, save_outputs() called |
| scripts/run_paper_trading.py | run_orchestrator.py | _run_run_mode() construction | WIRED | Line 1470: import, line 1513: orchestrator constructed, line 1527: execute() |
| scripts/run_paper_trading.py | report_aggregator.py | _run_reconcile() calls Aggregator.save_outputs() (D-15) | WIRED | Lines 1241-1274: import, construct, save_outputs() |
| report.py | report_aggregator.py | generate() accepts aggregator results | WIRED | generate() takes aggregate_results dict, extracts daily/target data |
| report_aggregator.py | features/session_manifest.py | model identity extraction (D-19) | WIRED | _model_identity() uses getattr on session_manifest for model_run_id, training_start, etc. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| report_aggregator.py | bets_df | bets.parquet via _load_bets() | Yes -- pd.read_parquet with schema validation | FLOWING |
| run_orchestrator.py | bets | RacePredictor.select_bets() with betting_target | Yes -- betting_target propagated from args | FLOWING |
| report.py | aggregate_results | PaperTradingReportAggregator.aggregate_all() | Yes -- dict with daily/weekly/target stats | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All Phase 54 tests pass | python -m pytest tests/test_race_progress.py tests/test_report_aggregator.py tests/test_run_orchestrator.py tests/test_cli_run_mode.py tests/test_paper_trading_report.py -v | 47 passed in 5.14s | PASS |
| ExitCode IntEnum has 8 members | python -c "from paper_trading.exit_codes import ExitCode; print(len(ExitCode))" | 8 | PASS |
| select_bets signature accepts betting_target | grep in race_predictor.py | def select_bets(self, race_df, bankroll, *, candidates=None, betting_target="place") | PASS |

### Probe Execution

Step 7c: SKIPPED -- no probe scripts defined for this phase.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| AUT-01 | 54-02 | One-command run mode (--mode run) | SATISFIED | RunModeOrchestrator 5-phase lifecycle. CR-01 FIXED: betting_target propagated. |
| AUT-02 | 54-01, 54-02 | Restart resumption | SATISFIED | RaceProgress 5-state machine, PREDICTED/NO_BET skipped, D-08 cross-validation |
| AUT-03 | 54-01, 54-02 | DB failure exit codes | SATISFIED | ExitCode IntEnum (8 codes), severity ordering, tests confirm |
| RPT-01 | 54-01, 54-03 | Weekly aggregation JSON | SATISFIED | aggregate_weekly() with ISO week range, save_outputs creates weekly_summary/ |
| RPT-02 | 54-01, 54-03 | Cumulative history with losses | SATISFIED | _base_stats tracks n_won/n_lost/n_pending, HTML shows settlement_status/outcome badges |
| RPT-03 | 54-01, 54-03 | Per-target aggregation | SATISFIED | aggregate_by_target groups by bet_type, HTML shows target breakdown |
| RPT-04 | 54-01, 54-03 | Model identity in reports | SATISFIED | _model_identity() from session_manifest, HTML footer, JSON common_fields |

All 7 requirement IDs (AUT-01, AUT-02, AUT-03, RPT-01, RPT-02, RPT-03, RPT-04) accounted for. No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| report.py | 85-93 | _compute_max_dd returns 0.0 for all-loss scenarios (WR-01 from review) | Warning | Misleading DD metric when 100% of capital lost |
| run_orchestrator.py | 341 | _compute_current_bankroll reads stale parquet vs in-memory (WR-02 from review) | Warning | Potential over-betting in sequential race processing |
| run_orchestrator.py | 659-667 | _build_race_predictor unused helper (documented with CR-02 note) | Info | Retained with risk annotation; not called in production path |

No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers found in any Phase 54 source files.

### Human Verification Required

### 1. Live Race End-to-End Test

**Test:** Run `python scripts/run_paper_trading.py --mode run --date YYYY-MM-DD --betting-target win --betting-mode flat --ensemble` on a live race day
**Expected:** Orchestrator executes full lifecycle (schedule fetch, TC fetch, predict, reconcile, aggregate) and generates correct win bets (not place bets)
**Why human:** Requires active DB connection with live race data; cannot simulate full DB interactions programmatically

### 2. Crash Resume Verification

**Test:** Start --mode run, manually interrupt (Ctrl+C) mid-prediction, then re-run same command
**Expected:** Re-run skips already-predicted races, continues from interrupted race. Exit code 130 on first run, 0 on completed re-run
**Why human:** Requires real-time race timing and manual interruption timing

### 3. HTML Report Visual Verification

**Test:** Open generated report.html in browser after a completed run
**Expected:** KPI cards display correct ROI/bankroll, target breakdown shows win/place separately, model identity footer visible, bet history table shows settlement_status and outcome badges
**Why human:** Visual rendering quality, layout correctness, CSS styling

### 4. Max Drawdown Accuracy

**Test:** Review _compute_max_dd behavior when all bets lose (WR-01)
**Expected:** Should show 100% or near-100% drawdown, not 0.0%
**Why human:** Need to evaluate whether the current behavior (0.0% DD for all-loss) is acceptable or requires fix

---

_Verified: 2026-06-06T22:00:00Z_
_Verifier: Claude (gsd-verifier)_
