# Feature Landscape

**Domain:** Paper Trading Pipeline Integration (within existing horse racing prediction system)
**Researched:** 2026-06-06
**Scope:** v2.4 milestone -- PT settlement integrity, pipeline consistency, shared features, strategy alignment, one-command run, evaluation expansion

---

## Executive Summary

v2.4 integrates the paper trading pipeline with the backtest pipeline so that PT produces trustworthy ROI measurements. The current PT system has seven identified code gaps that make its ROI unreliable: (1) win payout settlement is missing, (2) bet status conflates pending with lost, (3) feature construction diverges from BT (missing 3 feature modules), (4) no strategy manifest/PFP enforcement, (5) no OddsBandFilter, (6) dynamic regime in PT vs hardcoded in BT, (7) PaperPredictor class vs script inline code have drifted apart. Each gap independently makes PT ROI incomparable to BT ROI.

The feature landscape splits into three tiers. Tier 1 (settlement integrity + shared features) is the critical path -- without it, PT ROI is meaningless. Tier 2 (consistency verification + strategy alignment) ensures BT-validated strategies are faithfully executed in PT. Tier 3 (automation + reporting) improves operator experience but does not affect measurement validity.

---

## Table Stakes

Features the operator expects. Missing = PT pipeline produces unreliable or misleading ROI.

| # | Feature | Why Expected | Complexity | Dependencies | Notes |
|---|---------|--------------|------------|--------------|-------|
| 1 | Bet status lifecycle (pending/settled) | `result=0.0` is ambiguous: could mean "not yet reconciled" or "lost bet, payout 0". Current PT reconcile uses `result == 0.0` for both. ROI is overstated because losses are never explicitly recorded. | Low | None | Add `status` column (`"pending"` / `"settled"`). Replace `result == 0.0` checks with `status == "pending"`. Backward compatible: old parquets default status to `"pending"` if column missing. |
| 2 | Win payout settlement | BT uses `build_win_payout_map()` which reads `paytansyoumaban1`/`paytansyopay1` from payouts. PT reconcile only reads `payfukusyoumaban`/`payfukusyopay` (place). Win bets remain `result=0.0` forever, appearing as permanent losses. | Low | Feature 1 (status) | Reuse `build_win_payout_map()` from engine.py. Add win settlement branch to reconcile. Use `paytansyoumaban1`/`paytansyopay1` lookup. |
| 3 | Explicit loss recording | Current reconcile only updates `result` for winners (`payout > 0`). Losers stay `result=0.0` which conflates with "unreconciled". | Low | Feature 1 (status) | When `race_id in payout_map` and `umaban not in winners`, set `result=0.0, status="settled"`. This is the key fix: both winners and losers get `status="settled"`, but only winners have `result > 0`. |
| 4 | Shared feature builder function | BT `engine.py` and PT `run_paper_trading.py` both construct features independently. PT is missing `DamPedigreeFeatures`, `RecordFeatures`, `MiningFeatures`. BT includes all three. This means PT predictions use different features than BT predictions. | Medium | None | Extract `build_inference_features(store, race_df, entry_df, odds_df, odds_ts_df)` from `BacktestEngine.prepare_data()` lines 792-1099. Both BT and PT call it. This is the single most impactful refactor: it eliminates 7 code gaps simultaneously. |
| 5 | MLflow run ID + train period in records | PT saves `model_info.json` but predictions parquet does not include `mlflow_run_id`, `train_start`, `train_end`, `code_hash`. Cannot trace which model version produced a bet. | Low | None | Add these columns to prediction/bet records at predict time. Use `model_info` already loaded by `_load_models()`. |
| 6 | Strategy manifest/PFP for PT | BT supports `--strategy-manifest` + `ParameterFreezeProtocol`. PT does not use manifest at all. Strategy params (fractional_kelly, edge_threshold, regime fractions) are hardcoded defaults. | Medium | Feature 4 (shared builder) | Add `--strategy-manifest` to PT. Freeze params at predict start. Verify immutability at reconcile. Reuse existing `ParameterFreezeProtocol` class. |
| 7 | betting_target passthrough | BT supports `--betting-target win/place/wide`. PT hardcodes place-only logic in reconcile (`bet.bet_type.value == "place"`). Win settlement not implemented. | Low | Feature 2 (win payout) | Pass `--betting-target` through PT pipeline. Route to appropriate settlement logic in reconcile. |
| 8 | Flat/Kelly mode passthrough | BT supports `--betting-mode flat/kelly`. PT creates `RacePredictor(models)` without `StakeCalculator` or `DrawdownController`, always flat 100 yen. | Low | Feature 6 (manifest) | Pass betting_mode + strategy_params to PT `RacePredictor` constructor, matching BT. Add `--betting-mode` flag. |
| 9 | Regime state alignment | BT hardcodes `RegimeState.AGGRESSIVE`. PT uses dynamic `regime_detector.detect()` based on 200-race rolling window. Different regime = different bet selection for same race. | Low | Feature 6 (manifest) | Use same hardcoded AGGRESSIVE in PT until dynamic regime is re-enabled in BT. Add `--regime` CLI flag. |
| 10 | OddsBandFilter in PT | BT applies `OddsBandFilter` after candidate selection (engine.py line 1694). PT does not create or apply an OddsBandFilter. Some candidates that BT excludes appear in PT results. | Medium | Feature 6 (manifest) | Create OddsBandFilter in PT with same calibration source. Requires training period bet history (can reuse BT's `_generate_training_bet_history()` pattern or load from manifest). |
| 11 | Idempotent reconciliation | Current reconcile deduplicates by `(race_id, umaban)`. If a race has multiple bet types (place + wide), the key is insufficient. Re-running reconcile re-processes settled bets. | Low | Feature 1 (status) | Dedup key: `(race_id, umaban, bet_type)`. Skip rows with `status == "settled"`. |
| 12 | Processed-race idempotency | PT predict skips `existing_race_ids` but if process crashes mid-save, partial parquet may be written. | Low | None | Write to temp file, atomic rename. Or append-only with dedup on read. |

## Differentiators

Features that would set the PT pipeline apart. Not strictly expected, but valuable.

| # | Feature | Value Proposition | Complexity | Dependencies | Notes |
|---|---------|-------------------|------------|--------------|-------|
| 13 | One-command run mode (`--mode run`) | Eliminates operator error from running 4 separate commands in wrong order. Single command does: model verify, predict, wait-for-last-race, reconcile, report. | High | Features 1-12 | Requires orchestrator that chains existing modes. Restart resumption via processed-race tracking. DB failure exit codes (0=success, 1=partial, 2=fatal). |
| 14 | Data cutoff validation | Automated check that all feature source data predates prediction date. If Parquet includes rows after `train_end`, feature stats may leak future info. | Medium | Feature 5 (train period) | Verify `max(race_date) <= train_end` for each Parquet source. Log source file, row count, max_date. |
| 15 | Weekly aggregation reports | Current reports are daily JSON + monthly HTML. Weekly aggregation catches degradation earlier. | Low | Feature 3 (loss recording) | Add `_compute_weekly_stats()` to report.py. 7-day rolling ROI, hit rate, avg edge, bet count. |
| 16 | Per-target aggregation | ROI breakdown by bet_type (win/place/wide). Currently all types mixed in cumulative stats. | Low | Feature 3 (loss recording) | Group by `bet_type` in summary. Separate bankroll tracking per target. |
| 17 | Model/manifest identity in reports | HTML report shows `commit_hash` but not model version, train period, manifest SHA256. | Low | Feature 5 (identity columns) | Add model identity section to report header. |
| 18 | DD controller state persistence | BT creates `DrawdownController` fresh each run. For PT spanning multiple days, DD state should persist between sessions. | Medium | Feature 8 (Kelly mode) | Serialize DD state to JSON after each reconcile. Load at predict start. |
| 19 | Pipeline consistency contract verification | Automated check that PT produces same predictions as BT on a calibration race. Not just "same code path" but "same output for same input." | High | Feature 4 (shared builder) | Run both BT and PT on a held-out race. Compare predictions element-wise. Log divergence as WARNING. |
| 20 | Data cutoff audit log | Record exact timestamps and row counts of all data sources used for each prediction day. Enables post-hoc verification. | Medium | Feature 14 (cutoff validation) | Log `source_file, row_count, max_date, loaded_at` for each Parquet read during predict. |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Real-time odds streaming | System architecture is batch-oriented (Parquet files, periodic DB reads). Real-time would require fundamental redesign for marginal value in paper trading. | Keep 5-minute pre-post odds snapshot approach. |
| Automatic model retraining | PT should use a fixed model version. Retraining during PT invalidates ROI measurement across days. | `run_train.py` is a separate, operator-controlled step. PT loads pre-trained model from MLflow. |
| Web dashboard | CLI + HTML report is sufficient for a single-operator system. | Enhance existing HTML report with missing data. |
| Multi-user access control | Single operator system. No auth, no concurrent access control needed. | Single-process, file-based persistence. |
| Live bet execution (PatVoter integration) | Paper trading by definition does not place real bets. PatVoter is for future use. | Keep `is_paper=True` flag. PatVoter integration is out of scope for v2.4. |
| Dynamic regime detection in PT | BT hardcodes `RegimeState.AGGRESSIVE` with TODO comments. PT must match BT behavior exactly. | Use same hardcoded AGGRESSIVE until dynamic regime is re-enabled in both BT and PT simultaneously. |
| PT-specific feature modules | PT must use identical features as BT. Adding PT-specific features defeats consistency verification. | Shared feature builder ensures 1:1 alignment. |
| Backfilling historical PT records | Old predictions were generated with different (incomplete) features. Retroactively "fixing" them would be dishonest. | Start fresh measurement from v2.4 deployment date. Mark pre-v2.4 records as legacy. |

## Feature Dependencies

```
Feature 1: Bet status lifecycle
  +--> Feature 2: Win payout settlement (needs status to distinguish loss from pending)
  +--> Feature 3: Loss recording (needs status field)
  +--> Feature 11: Idempotent reconciliation (depends on status field)
  
Feature 4: Shared feature builder
  +--> Feature 6: Strategy manifest/PFP (depends on same inference path)
  +--> Feature 10: OddsBandFilter (same calibration source)
  +--> Feature 19: Pipeline consistency verification (depends on same features)

Feature 5: MLflow run ID tracking
  +--> Feature 14: Data cutoff validation (uses train_end from model info)
  +--> Feature 17: Model identity in reports (depends on stored identity)

Feature 6: Strategy manifest/PFP
  +--> Feature 8: Flat/Kelly mode (params from manifest)
  +--> Feature 9: Regime alignment (regime from manifest or hardcoded)
  +--> Feature 10: OddsBandFilter (calibration params from manifest)
  +--> Feature 18: DD controller state persistence (DD config from manifest)

Feature 13: One-command run mode
  +--> All Tier 1 features (1-4)
  +--> All Tier 2 features (5-10)
  +--> Feature 18: DD state persistence
```

## MVP Recommendation

### Phase 1: Settlement Integrity (Critical Path)

Build first because without it, all downstream ROI numbers are unreliable.

1. Bet status lifecycle: add `status` column, pending/settled states
2. Win payout settlement: reuse `build_win_payout_map()` pattern
3. Explicit loss recording: set `status="settled"` for both wins and losses
4. Idempotent reconciliation: dedup key `(race_id, umaban, bet_type)`

### Phase 2: Shared Feature Builder

Build second because it eliminates the feature divergence and enables all consistency work.

1. Extract `build_inference_features()` from `BacktestEngine.prepare_data()`
2. Refactor BT to call shared builder
3. Refactor PT to call shared builder
4. Verify feature column parity between BT and PT paths

### Phase 3: Pipeline Consistency + Strategy Alignment

Build third because it ensures the shared builder is used with identical configuration.

1. MLflow run ID + train period + code_hash in prediction records
2. Strategy manifest/PFP integration with PT
3. betting_target + betting_mode + regime alignment
4. OddsBandFilter calibration for PT
5. Data cutoff validation

### Phase 4: Automation + Reporting

Build last because it chains the now-reliable pipeline components.

1. One-command run mode with restart resumption
2. Weekly aggregation + per-target breakdown
3. Model identity in reports
4. DD controller state persistence (if Kelly mode is used)

**Defer:**
- Pipeline consistency contract verification (complex, can be manual initially)
- Data cutoff audit log (medium complexity, low immediate value)

## Existing Code Gaps (from direct code analysis)

### Gap 1: PT predict duplicates BT feature construction, with omissions

`_run_predict` in `run_paper_trading.py` (lines 367-437) and `BacktestEngine.prepare_data()` (lines 728-1099) both build features independently. PT predict is missing: `DamPedigreeFeatures`, `RecordFeatures`, `MiningFeatures`. BT includes all three. This means PT predictions are computed with different features than BT, making ROI comparison invalid.

**Impact:** HIGH -- different features = different predictions = incomparable ROI.

### Gap 2: PT reconcile only handles place bets

`_run_reconcile` (lines 899-1115) only looks up `payfukusyoumaban`/`payfukusyopay`. It never checks `paytansyoumaban1`/`paytansyopay1` for win bets. Win bets remain `result=0.0` forever, appearing as permanent losses.

**Impact:** HIGH -- win bet ROI is always 0% regardless of actual performance.

### Gap 3: PT reconcile treats result=0.0 as "pending"

Line 920: `unsettled = pred_df[pred_df["result"] == 0.0]`. A horse that finished 4th (place loss) has `result=0.0` which is the same as "not yet reconciled". The reconcile loop only processes bets with `umaban in winners`, leaving losers at `result=0.0` perpetually "unsettled".

**Impact:** HIGH -- cumulative ROI is overstated because losses are excluded from the denominator.

### Gap 4: No DDController or StakeCalculator in PT

PT creates `RacePredictor(models)` without `stake_calculator` or `dd_controller`, so `self._betting_mode = "flat"` and stake is always 100 yen. BT with `--betting-mode kelly` uses fractional Kelly sizing and DD control. PT cannot replicate BT's bet sizing.

**Impact:** MEDIUM -- flat vs Kelly affects ROI when edge varies.

### Gap 5: PT uses dynamic regime, BT uses hardcoded AGGRESSIVE

PT `_run_predict` line 478: `regime = models.regime_detector.detect(recent_stats_df)`. BT line 1619: `regime = RegimeState.AGGRESSIVE` (with TODO comment). Different regime = different `regime_params` = different bet selection.

**Impact:** MEDIUM -- different regime can select different bets.

### Gap 6: No OddsBandFilter in PT

BT applies `OddsBandFilter` after candidate selection (engine.py line 1694). PT does not create or apply an OddsBandFilter. Some candidates that BT excludes will appear in PT results.

**Impact:** MEDIUM -- PT may include bets that BT would filter out.

### Gap 7: PaperPredictor class vs script inline code divergence

`PaperPredictor.setup()` and `PaperPredictor.predict_race()` in `src/paper_trading/predictor.py` duplicate logic from `_run_predict` in `run_paper_trading.py` but with differences (e.g., PaperPredictor uses `everydb2.get_race_schedule()` while script uses `load_races_from_db()`). These two code paths have drifted apart.

**Impact:** LOW -- the class-based path (`PaperPredictor`) is not currently used by the CLI. But it represents a maintenance burden and potential confusion.

## Sources

- `src/backtest/engine.py` -- BacktestEngine with full settlement, feature construction, strategy integration (2392 lines)
- `src/backtest/race_predictor.py` -- RacePredictor shared inference component (1645 lines)
- `src/paper_trading/predictor.py` -- PaperPredictor class (202 lines, partially used)
- `src/paper_trading/reconciler.py` -- PaperReconciler class (153 lines, place-only settlement)
- `src/paper_trading/watcher.py` -- RaceWatcher class (142 lines, schedule-based prediction)
- `src/paper_trading/report.py` -- PaperTradingReport (157 lines, daily JSON + monthly HTML)
- `src/paper_trading/config.py` -- PaperTradingConfig dataclass (38 lines)
- `src/automation/scheduler.py` -- RaceScheduler class (228 lines, race-day orchestration)
- `scripts/run_paper_trading.py` -- CLI script with 5 modes (1384 lines)
- `src/domain/models.py` -- Bet dataclass (26 lines)
- `.planning/PROJECT.md` -- v2.4 milestone requirements (308 lines)
- `CLAUDE.md` -- Architecture documentation (full system overview)

---
*Feature research for: v2.4 Paper Trading Pipeline Integration*
*Researched: 2026-06-06*
