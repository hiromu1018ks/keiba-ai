# Domain Pitfalls

**Domain:** Paper Trading Pipeline Integration (v2.4)
**Researched:** 2026-06-06
**Context:** v2.4 milestone integrating PT pipeline with BT pipeline for trustworthy ROI measurement. The existing PT system has 7 identified code gaps that make its ROI unreliable. Each gap independently makes PT-BT ROI comparison invalid.

## Critical Pitfalls

Mistakes that cause misleading ROI or require rewrites.

### Pitfall 1: Feature Divergence Between BT and PT (PROVEN -- current codebase)

**What goes wrong:** PT `_run_predict()` is missing `DamPedigreeFeatures`, `RecordFeatures`, `MiningFeatures` that BT `BacktestEngine.prepare_data()` includes. PT produces predictions with fewer features, leading to different model outputs for the same race. ROI comparison is meaningless.

**Why it happens:** Feature construction code is duplicated in BT (`engine.py` lines 792-1099) and PT (`run_paper_trading.py` lines 368-436, 743-798, 1200-1266 -- three copies). When new features are added to BT, PT copies are not always updated. Three separate copies guarantee future divergence.

**Consequences:** PT ROI and BT ROI measure different things. A 95% PT ROI does not mean BT will also be 95%. Operator makes deployment decisions on invalid data.

**Prevention:** Extract shared `build_inference_features()` function. Both BT and PT call it. Single source of truth. When a feature module is added, it automatically appears in both paths.

**Detection:** Add feature column count assertion in PT: `assert len(feat_df.columns) >= N` where N is the expected column count from BT. Log column count at predict time. If it differs from BT by more than 5 columns, raise WARNING.

---

### Pitfall 2: result=0.0 Ambiguity Masks Losses (PROVEN -- current codebase)

**What goes wrong:** `result=0.0` means both "not yet reconciled" and "lost bet, payout 0". The reconcile loop at `run_paper_trading.py` line 958 only processes bets where `umaban in winners`, leaving losers at `result=0.0` perpetually "unsettled". Cumulative ROI only counts winning returns in the numerator while excluding losing stakes from the denominator.

**Why it happens:** The reconcile design assumes bets are either "won" (result > 0) or "pending" (result == 0). There is no "settled loss" state. The BT engine does not have this problem because it settles every bet immediately using `_settle_bet()` which returns 0.0 for losses.

**Consequences:** Cumulative ROI is overstated. For example, with 10 bets at 100 yen each, 2 wins returning 300 yen each: actual ROI = 600/1000 = 60%. But if 8 losses stay "pending" and only 2 wins are recorded: reported return = 600, reported stake = 200 (only winning bets counted), reported ROI = 300%. This is a 5x overstatement.

**Prevention:** Add `status` column (`"pending"` / `"settled"`). Reconcile sets `status="settled"` for ALL bets where race results are available -- both wins AND losses. ROI calculation uses `status == "settled"` filter.

**Detection:** Check if any bets have `status == "pending"` for dates more than 1 day in the past. If so, reconcile is not processing losses. Also check: `total_stake` should equal `sum(stake for status=="settled")`, not `sum(stake for result > 0)`.

---

### Pitfall 3: Win Bet Settlement Missing in PT (PROVEN -- current codebase)

**What goes wrong:** PT reconcile (`_run_reconcile` lines 899-1115) only looks up `payfukusyoumaban`/`payfukusyopay` (place payouts). It never checks `paytansyoumaban1`/`paytansyopay1` for win payouts. All win bets appear as permanent losses regardless of actual outcome.

**Why it happens:** The reconcile was written when PT only supported place betting. Win betting support was added to predict mode but reconcile was not updated. The BT engine handles both via `_settle_bet()` which dispatches by `bet_type`.

**Consequences:** If PT generates win bets (which it does when `betting_target="win"` or the model selects win candidates), ALL of them show `result=0.0` even for winning horses. Win ROI appears to be 0%.

**Prevention:** Import `build_win_payout_map()` from `backtest.engine`. Add win settlement branch: lookup `(race_id, umaban)` in win payout map. If found and horse finished 1st, `result = stake * payout_multiplier`.

**Detection:** After reconcile, check for any `bet_type == "win"` bets still at `status == "pending"`. If the race has completed (payout data available), win settlement is broken.

---

### Pitfall 4: Regime Mismatch Between BT and PT

**What goes wrong:** BT hardcodes `regime = RegimeState.AGGRESSIVE` at `engine.py` line 1619. PT uses dynamic `regime = models.regime_detector.detect(recent_stats_df)` at `run_paper_trading.py` line 478. Different regime = different `regime_params` = different edge_threshold, fractional_kelly, max_bets_per_race. PT selects different bets than BT for the same race.

**Why it happens:** BT switched to hardcoded AGGRESSIVE (with TODO to re-enable dynamic later) but PT was not updated to match. The regime detector uses a 200-race rolling window which may produce conservative/collapsed states during losing streaks, while BT always uses aggressive parameters.

**Consequences:** BT may skip a race (if dynamic regime was conservative) while PT bets on it (dynamic regime is aggressive), or vice versa. The set of bets differs, making ROI comparison invalid.

**Prevention:** Both BT and PT must use the same regime determination logic. Since BT hardcodes AGGRESSIVE, PT should too. Add `--regime` CLI flag defaulting to `"aggressive"`. When BT re-enables dynamic regime, PT should match.

**Detection:** Log regime state for each race in both BT and PT. Compare regime column in bet history. If they differ for any race, the pipelines are misaligned.

---

### Pitfall 5: Shared Builder Extraction Breaks BT

**What goes wrong:** Extracting `build_inference_features()` from `BacktestEngine.prepare_data()` introduces a regression in BT. The extraction might miss a subtle dependency (e.g., `preserve_columns=["kakuteijyuni", "confirmed_odds"]` parameter passed to `FeatureEngine.build_all()`, or `jyocd` NAR filtering timing).

**Why it happens:** `prepare_data()` is 300+ lines with interwoven data loading, filtering, feature construction, and map building. The feature construction section is not cleanly separable from the data loading section. There are dependencies on variables computed during data loading (e.g., `race_ids` from `feat_df["race_id"].unique()`, which is used by `HorseHistoryFeatures.compute()`).

**Consequences:** BT ROI changes after refactoring. All historical BT benchmarks become invalid. This is a silent regression that may not be caught until the next full BT run (~40 minutes).

**Prevention:** Run full BT before and after extraction. Compare `bet_history` element-wise. Assert `len(bet_history)` is identical, `sum(b["stake"] for b in bet_history)` matches within 1 yen, and `sum(b["result"] for b in bet_history)` matches within 1 yen. Add CI test that runs a small BT and checks result hash.

**Detection:** Run `python scripts/run_backtest.py --train-start 20230101 --train-end 20231231 --test-start 20240101 --test-end 20240630 --ensemble` before and after extraction. Compare ROI, bet count, and total stake.

---

### Pitfall 6: OddsBandFilter Calibration Without Training Bet History

**What goes wrong:** BT calibrates OddsBandFilter by running an inner BacktestEngine on the training period (`_generate_training_bet_history()`). PT does not have training period models available and cannot run this calibration. If OddsBandFilter is applied with default parameters, it may filter out different bets than BT's calibrated version.

**Why it happens:** The calibration requires running the full inference pipeline on training data to generate bet history. PT operates in inference-only mode -- it loads pre-trained models and does not have access to the training pipeline.

**Consequences:** PT applies different OddsBandFilter bands than BT. Some bets that BT includes are excluded in PT, and vice versa.

**Prevention:** Two options: (a) Save OddsBandFilter calibration data as a model artifact during `run_train.py`. PT loads the calibration from the artifact. (b) Pre-compute calibration during strategy optimization and include it in the strategy manifest. Option (a) is simpler and more reliable.

**Detection:** Compare OddsBandFilter excluded bands between BT and PT. If they differ, calibration source mismatch.

---

## Moderate Pitfalls

### Pitfall 7: Parquet Write During Crash Loses Data

**What goes wrong:** PT predict writes predictions to parquet incrementally. If the process crashes mid-write, the parquet file may be corrupted or partial. On restart, the corrupted file causes read errors or missing data.

**Prevention:** Write to a temporary file first, then atomically rename. Or use append-only writes with explicit flush.

### Pitfall 8: PaperPredictor Class vs Script Inline Code Confusion

**What goes wrong:** `PaperPredictor` class in `src/paper_trading/predictor.py` is not used by the CLI script. `RaceWatcher` uses it, but the CLI has its own inline prediction code. Two code paths exist for the same task.

**Prevention:** Decide on one canonical path. Either (a) refactor CLI to use `PaperPredictor` class, or (b) remove `PaperPredictor` class and keep CLI inline. Option (b) is safer because CLI inline is what's currently working and tested.

### Pitfall 9: Pre-v2.4 Records Contaminate Cumulative Stats

**What goes wrong:** Existing PT records (generated with incomplete features and no status tracking) are mixed with new v2.4 records in `bets.parquet`. Cumulative ROI includes invalid historical data.

**Prevention:** Mark pre-v2.4 records with a `schema_version` column. New records get `schema_version="v2.4"`. Cumulative stats can optionally filter by schema version for clean measurement.

### Pitfall 10: Kelly Sizing in PT Without DD State Persistence

**What goes wrong:** PT creates `DrawdownController` fresh each run. If using Kelly mode across multiple days, DD state resets daily. A drawdown that should trigger STOP mode on day 2 is ignored because DD controller starts fresh.

**Prevention:** Serialize DD state to JSON after each reconcile. Load at predict start. Include `dd_state.json` in PT output directory.

### Pitfall 11: Wide Bet Settlement kumi Parsing

**What goes wrong:** Wide payout lookup uses `paywidekumi1-7` format which requires parsing "513" as (horse 5, horse 13) or (horse 51, horse 3). BT has a complex 80-line `build_wide_payout_map()` with heuristics for this. PT would need to replicate this exactly.

**Prevention:** Import `build_wide_payout_map()` directly from `backtest.engine`. Do not reimplement.

### Pitfall 12: PFP Verification Failure During PT Run

**What goes wrong:** PFP `verify()` checks model parameter hashes. If any model parameter has changed between predict and reconcile (e.g., due to memory mutation, lazy loading, or model reload), PFP raises `RuntimeError` and PT aborts.

**Prevention:** PFP freeze should happen once at predict start. Reconcile should verify the same frozen state. If reconcile loads models again, it must use the same MLflow run ID. Store frozen state hash in predictions parquet.

---

## Minor Pitfalls

### Pitfall 13: Missing BloodlineFeatures in BT

**What goes wrong:** PT `_run_predict` includes `BloodlineFeatures` but `BacktestEngine.prepare_data()` does not. When extracting shared builder, if BloodlineFeatures is included, BT results change. If excluded, PT results change.

**Prevention:** Check whether `blood_*` features are in any model's `FEATURE_COLS`. If not, they are unused and can be safely included or excluded. If yes, add to BT's `prepare_data()` as well.

### Pitfall 14: Daily Summary JSON Missing Losses

**What goes wrong:** `PaperReconciler._compute_summary()` computes `total_return = bets_df[bets_df["result"] > 0]["result"].sum()`. This only sums positive returns. Losses are implicit (missing from sum). The summary does not distinguish "total return from wins" from "total stake lost".

**Prevention:** Add explicit loss tracking: `total_losses = bets_df[(bets_df["status"] == "settled") & (bets_df["result"] == 0)]["stake"].sum()`.

### Pitfall 15: HTML Report Assumes Place-Only Bets

**What goes wrong:** Report template shows "fuku" (place odds) for all bets. Win bets would show place odds, not win odds. Wide bets have no odds display at all.

**Prevention:** Add bet_type-aware display in report template. Show `tanodds` for win, `fukuoddslow` for place, pair notation for wide.

### Pitfall 16: Time Zone Issues in Race Completion Detection

**What goes wrong:** PT runs on JST but system clock may be UTC. "Last race post time + 30 min" calculation uses local time. If system is UTC, the wait may be 9 hours off.

**Prevention:** Use explicit timezone-aware datetime. `post_time = JST.localize(datetime.combine(target_date, time(h, m)))`.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Settlement integrity | result=0.0 ambiguity (Pitfall 2) | Add status column before any settlement changes |
| Settlement integrity | Win payout missing (Pitfall 3) | Reuse `build_win_payout_map()` from engine.py |
| Shared feature builder | Breaking BT regression (Pitfall 5) | Full BT before/after comparison test |
| Shared feature builder | BloodlineFeatures asymmetry (Pitfall 13) | Check feature manifest for blood_* usage |
| Strategy alignment | OddsBandFilter calibration (Pitfall 6) | Save calibration as model artifact |
| Strategy alignment | Regime mismatch (Pitfall 4) | Hardcode AGGRESSIVE in PT to match BT |
| One-command run | Crash data loss (Pitfall 7) | Atomic write with temp file |
| One-command run | PFP verify failure (Pitfall 12) | Freeze once, store hash in predictions |
| One-command run | DD state reset (Pitfall 10) | Serialize DD state to JSON |
| Reporting | Loss tracking missing (Pitfall 14) | Add explicit loss columns to summary |

## Project-Specific Historical Lessons

1. **v1.6 (PROVEN):** Feature code duplication between training and inference caused 6 feature omissions. The same pattern is now happening between BT and PT. Shared builder prevents this class of bug entirely.

2. **v1.8 (PROVEN):** Feature additions in one path without updating another caused ROI degradation. PT's three inline copies of feature construction guarantee this will happen again without shared builder.

3. **v2.1 (PROVEN):** Shadow comparison framework validates changes before deployment. The same principle should apply to PT: any change to shared builder must pass BT regression test before PT uses it.

## Sources

- Direct codebase analysis: `scripts/run_paper_trading.py` (1384 lines, three feature construction copies)
- Direct codebase analysis: `src/backtest/engine.py` (2392 lines, canonical feature construction)
- Direct codebase analysis: `src/paper_trading/reconciler.py` (153 lines, settlement gaps)
- Direct codebase analysis: `src/backtest/parameter_freeze_protocol.py` (PFP patterns)
- Project history: `.planning/PROJECT.md` (v2.4 milestone, gaps identified)
- Confidence: HIGH for all pitfalls -- every one identified by direct source code comparison

---
*Pitfall research for: v2.4 Paper Trading Pipeline Integration*
*Researched: 2026-06-06*
