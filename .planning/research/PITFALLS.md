# Domain Pitfalls: Win Backtest Validation & Pipeline Optimization

**Domain:** Switching backtest/WF validation from place (fukusho) to win (tansho) + ML pipeline performance optimization
**Context:** keiba-ai v1.2 milestone -- existing v1.1 pipeline produces place bets, needs win bet generation, win payout settlement, and faster training/backtest cycles
**Researched:** 2026-05-04
**Confidence:** HIGH (cross-validated with full codebase analysis of backtest engine, race predictor, training pipeline, and payout settlement logic)

---

## Critical Pitfalls

Mistakes that cause systematically wrong ROI numbers, silent mis-settlement, or require major rewrites.

### Pitfall 1: Payout Map Uses Place Payouts (`payfukusyo*`) For Win Settlement

**What goes wrong:** `build_payout_map()` in `engine.py:102-125` reads `payfukusyoumaban1-5` and `payfukusyopay1-5` (place payout columns). When switching to win bets, the settlement still uses place payouts. This produces systematically wrong ROI numbers: place payouts pay for top-3 finishers, but win bets should only pay for 1st place. The ROI will be inflated because more bets "win" against place payout odds than actually win at win odds.

**Why it happens:** The function was written for the place-only era. It builds `payout_map: dict[(race_id, umaban), odds_multiplier]` from place payout columns. The `_settle_bet()` method at line 931-934 uses this map for both place AND win bets:
```python
# line 931-934: applies to BOTH win and place
if hasattr(self, "payout_map") and payout_key in self.payout_map:
    return float(bet.stake * self.payout_map[payout_key])
```
Since the map contains place multipliers (which include top-3 finishers), a win bet that finished 2nd would incorrectly receive a payout.

**Consequences:** Backtest ROI is completely unreliable. You cannot validate the win model because the settlement logic is wrong. This is the single highest-priority fix.

**Prevention:**
- Create a new `build_win_payout_map()` that reads `paytansyoumaban1` and `paytansyopay1` (win payout columns that exist in the ETL schema at `etl.py:112-113`).
- The win payout map should map `(race_id, umaban) -> win_odds_multiplier` using ONLY the single win payout entry (1st place only).
- In `_settle_bet()`, dispatch to the correct payout map based on `bet.bet_type`: win bets use `win_payout_map`, place bets use `payout_map`.
- If `win_payout_map` lookup fails, fall back to `confirmed_odds` (line 941-948 already handles this for WIN via `finish_pos == 1`).

**Detection:**
- After the switch, verify that no win bet with `kakuteijyuni != 1` receives a non-zero payout. Count should be zero.
- Cross-check total payout sum against manually computed `sum(stake * confirmed_odds)` for 1st-place finishers only.

**Phase:** Must be fixed FIRST, before any other win validation work. Phase 1.

**Sources:**
- Codebase analysis: `engine.py:102-125` (build_payout_map), `engine.py:931-948` (_settle_bet), `etl.py:112-113` (payout columns in schema)
- HIGH confidence: Directly observable mismatch between payout columns and bet type

---

### Pitfall 2: `final_odds_map` Uses `fukuoddslow` (Place Odds) For Win Bet Settlement

**What goes wrong:** At `engine.py:276-281`, the final odds map is built from `fukuoddslow` (place odds):
```python
final_odds_map: dict[tuple[str, int], float] = {}
if not final_odds_df.empty:
    for _, r in final_odds_df.iterrows():
        key = (str(r["race_id"]), int(r["umaban"]))
        if pd.notna(r.get("fukuoddslow")):
            final_odds_map[key] = float(r["fukuoddslow"])
```
When win bets are generated, `final_odds_map` provides place odds for the `bet.final_odds` field, which is used for settlement when payout map lookup fails. Win odds are much higher than place odds, so using place odds understates settlement and deflates ROI.

**Why it happens:** The map was built exclusively for the place betting system. The `load_odds_snapshots()` reader returns both `tanodds` and `fukuoddslow`, but only `fukuoddslow` is used for the final odds map.

**Consequences:** Win bets settled via `final_odds_map` fallback receive place odds, dramatically understating ROI. This makes it impossible to assess the true win model performance.

**Prevention:**
- Build separate `final_win_odds_map` using `tanodds` column (confirmed win odds) instead of `fukuoddslow`.
- When updating bet final odds at line 607, dispatch based on `bet.bet_type`: use `final_win_odds_map` for WIN, `final_odds_map` for PLACE.
- Alternatively, build a unified map keyed by `(race_id, umaban, bet_type)` to avoid parallel map confusion.

**Detection:**
- Log the final_odds value for each settled bet. If win bets show odds < 5.0 for mid-range popularity horses, the map is using place odds.
- Compare `final_odds` in bet_history against `tanodds` column for a sample of races.

**Phase:** Must be fixed alongside Pitfall 1, before any backtest execution.

**Sources:**
- Codebase analysis: `engine.py:276-281`, `engine.py:601-608`
- HIGH confidence: Column name directly observable

---

### Pitfall 3: Bet Generation Hardcodes `BetType.PLACE` And Place Odds Columns

**What goes wrong:** `RacePredictor.select_bets()` at `race_predictor.py:532-642` always generates `BetType.PLACE` bets using `fukuoddslow` as the odds source. The candidate selection calls `get_place_candidates()` which uses place-specific columns (`place_selection_ev`, `place_selection_edge`, `place_selection_prob`, `fukuoddslow`). When switching to win validation, there is no corresponding `get_win_candidates()` or win-based bet generation path.

**Why it happens:** The entire `select_bets()` method was designed for place betting. The method:
1. Gets candidates from `get_place_candidates()` (line 552)
2. Uses `place_selection_ev` and `place_selection_edge` columns (line 608-609)
3. Uses `fukuoddslow` for odds (line 609)
4. Creates `BetType.PLACE` (line 631)

The win model DOES compute `win_selection_ev`, `ev_win_corrected`, `EV_lower_win_corrected` (via `WinSelectionGate` at `race_predictor.py:131-143`), but no code path uses these for bet generation.

**Consequences:** Without implementing win bet generation, the backtest will continue to generate place bets even if the rest of the pipeline is "switched" to win mode. The validation will test the wrong thing.

**Prevention:**
- Create `get_win_candidates()` analogous to `get_place_candidates()`, using `win_selection_ev`, `win_selection_edge`, and `tanodds` instead of place equivalents.
- Create `select_win_bets()` (or add a mode parameter to `select_bets()`) that generates `BetType.WIN` bets using `tanodds` and win EV columns.
- The `WinSelectionGate` already scores candidates with `win_gate_score`, `win_gate_pass`, etc. Use these for candidate filtering.
- Add a `betting_target` parameter to `BacktestEngine.__init__()` to control whether to generate win or place bets.

**Detection:**
- After implementation, verify that generated bets have `bet_type == "win"` (not `"place"`).
- Check that bet odds come from `tanodds` (typically 1.1-100+) not `fukuoddslow` (typically 1.0-10.0).

**Phase:** Core implementation work, Phase 1 alongside payout fixes.

**Sources:**
- Codebase analysis: `race_predictor.py:408-642`, `win_selection_gate.py`, `engine.py:504-596`
- HIGH confidence: All place-specific column references directly observable

---

### Pitfall 4: Diagnostics Log Place-Specific Columns, Breaking Win Analysis

**What goes wrong:** The `DiagnosticLogger.log_horse()` calls in `engine.py:545-591` and `engine.py:632-678` log place-specific diagnostic fields: `p_place_pred`, `e_return_place_pred`, `ev_place`, `p_place_corrected`, `e_return_place_corrected`, `ev_place_corrected`, `EV_lower_place`, `place_selection_ev`, `place_selection_edge`, `place_selection_prob`, `place_bucket_multiplier`, `place_gate_score`, `place_gate_pass`, `place_gate_rank`, `place_gate_score_gap`. For win validation, these are the wrong metrics. The diagnostic data will show place model performance when you need win model performance.

**Why it happens:** The `if "ev_place" in result_df.columns:` guard (line 544, 630) means diagnostics are only logged when place EV columns exist. For win-focused validation, `ev_place` may still exist (both models run), but the diagnostic focus should shift to `ev_win`, `p_win_pred`, `ev_win_corrected`, `EV_lower_win_corrected`, `win_selection_ev`, etc.

**Consequences:** Post-backtest analysis of horse diagnostics CSV will show place metrics, making it impossible to diagnose why specific win bets were or were not placed.

**Prevention:**
- Add parallel diagnostic logging for win-specific columns when they exist.
- At minimum, log `p_win_pred`, `ev_win`, `ev_win_corrected`, `EV_lower_win_corrected`, `win_selection_ev`, `win_selection_edge`, `win_gate_score`, `win_gate_pass`, `tanodds`.
- Consider making the diagnostic logger mode-aware (place vs. win) to control which columns are logged in detail.

**Detection:**
- After backtest, check `bt_*_horse_diagnostics.csv` columns. If they only contain place metrics, the diagnostics are incomplete for win analysis.

**Phase:** Should be fixed alongside bet generation. Phase 1-2.

**Sources:**
- Codebase analysis: `engine.py:544-591, 630-678`
- HIGH confidence: Column names directly observable

---

### Pitfall 5: WF Validation Uses Place Backtest For Overfitting Detection

**What goes wrong:** `run_wf_validation.py` at lines 171-195 creates `BacktestEngine` instances and runs backtests for both test and train periods. The engine uses the default behavior, which generates place bets. The WF validation then compares `train_roi` vs `test_roi` to detect overfitting (via `judge_overfitting()` at `walk_forward_cv.py:300-351`). But the ROI being compared is PLACE ROI, not WIN ROI. The overfitting detection is testing the wrong model's performance.

**Why it happens:** `BacktestEngine` has no parameter to switch between place and win mode. `run_wf_validation.py` has no mode flag. The WF validation script was designed before win model became the priority.

**Consequences:** WF validation may report "PASS" for overfitting on place metrics while the win model is heavily overfit. Or it may report "FAIL" on place metrics when the win model is fine. Either way, the overfitting verdict is unreliable for the win model.

**Prevention:**
- Add a `betting_target` parameter (or similar) to `BacktestEngine.__init__()` and propagate it through `RacePredictor`.
- Add a `--betting-target win|place` flag to `run_wf_validation.py` (or change the default to `win`).
- The WF validation feature ranking extraction (`_extract_all_feature_rankings`) already correctly pulls from `sub.win.hit_model` (line 92-98), so feature stability analysis is already win-focused. Only the ROI comparison needs fixing.

**Detection:**
- If WF validation report shows `total_bets` count similar to the place-only era (~9000 bets/year), it is still generating place bets. Win bets should be fewer (~1000-3000 bets/year depending on edge thresholds).

**Phase:** Phase 1, alongside engine changes.

**Sources:**
- Codebase analysis: `run_wf_validation.py:171-195`, `walk_forward_cv.py:300-351`
- HIGH confidence: BacktestEngine constructor and WF script directly observable

---

### Pitfall 6: `_settle_bet()` Fallback Uses `finish_pos == 1` For Win But Settlement Odds Are Wrong

**What goes wrong:** The `_settle_bet()` fallback path at `engine.py:940-949` correctly checks `finish_pos == 1` for win bets. However, `settle_odds` is set to `bet.final_odds` which, as identified in Pitfall 2, currently contains place odds. The settlement will pay `stake * fukuoddslow` for a win, which understates the actual return.

**Why it happens:** The fallback is architecturally correct (use `final_odds` for settlement when payout map lookup fails) but the data flowing into `final_odds` is wrong.

**Consequences:** Even after fixing the payout map (Pitfall 1), if any race's win payout data is missing from the payout map, the fallback will underpay. This creates inconsistency: most bets settle correctly but some settle incorrectly, making the aggregate ROI unreliable.

**Prevention:**
- Fix Pitfall 2 (final_odds_map uses tanodds) first. Then the fallback path will work correctly.
- Add logging when fallback settlement is used: `logger.warning("Win payout map miss for %s/%d, using final_odds=%.1f", race_id, umaban, settle_odds)`.
- Track `n_fallback_settlements` in `BacktestResult` and alert if > 5% of bets use fallback.

**Detection:**
- After backtest, check fallback settlement count. If high, the win payout data has gaps.

**Phase:** Phase 1, alongside Pitfalls 1 and 2.

**Sources:**
- Codebase analysis: `engine.py:940-949`
- HIGH confidence: Fallback logic directly observable

---

## Moderate Pitfalls

### Pitfall 7: `run_backtest.py` `before_roi` Reference Is Place-Specific

**What goes wrong:** `display_single_year_result()` at `run_backtest.py:233` compares against a hardcoded `before_roi = 0.638` (63.8% place ROI). When switching to win mode, this baseline is meaningless because win ROI has a different distribution (fewer winning bets, higher payouts per win). The comparison will be misleading.

**Prevention:**
- Replace `before_roi` with the win-model baseline from the v1.1 backtest (or remove the comparison entirely until a win baseline is established).
- Add a comment explaining what the baseline represents.

**Phase:** Phase 2 (analysis).

---

### Pitfall 8: Edge Thresholds Calibrated For Place, Not Win

**What goes wrong:** The `RegimeDetector` strategy parameters (`edge_threshold`, `ev_threshold`, `min_place_prob`, `max_place_odds`) in `config/settings.yaml` or hardcoded defaults are calibrated for place betting. Place edge distribution is different from win edge distribution: place edges are smaller and more frequent (p_place ~0.3 for average horse, odds ~1.5-3.0) while win edges are larger but rarer (p_win ~0.08 for average horse, odds ~5.0-50.0).

**Why it happens:** The regime detector was trained on place ROI and place edge statistics. Win edge distribution has heavier tails and higher variance.

**Consequences:** Using place-calibrated edge thresholds for win bet selection may either (a) generate too many bets (threshold too low for win edge scale) causing overbetting, or (b) generate too few bets (threshold too high) causing missed opportunities.

**Prevention:**
- Add win-specific strategy parameters to the regime detector config: `win_edge_threshold`, `win_ev_threshold`, `min_win_prob`, `max_win_odds`.
- Calibrate these from the win model's edge distribution on validation data.
- In `get_win_candidates()`, use the win-specific parameters instead of place parameters.

**Detection:**
- Plot the distribution of `win_selection_edge` values. If the 95th percentile is much higher than `edge_threshold=0.03`, the threshold is too low. If it is lower, the threshold is too high.

**Phase:** Phase 2 (after win bet generation works).

**Sources:**
- Codebase analysis: `regime_detector.py`, `race_predictor.py:427-429`
- MEDIUM confidence: Threshold values are configurable, exact calibration needs empirical data

---

### Pitfall 9: Race-Level Per-Race Loop Is The Dominant Performance Bottleneck

**What goes wrong:** The backtest engine iterates over individual races in a Python for-loop (`engine.py:420: for race_id in race_ids:`). For each race, it:
1. Filters the full feature DataFrame for that race (line 421)
2. Merges pre-computed features (hist_df, jockey_df, trainer_df, jt_df) for that race (lines 466-469)
3. Drops POST_RACE columns (line 472-475)
4. Calls `RacePredictor.predict()` which itself does multiple DataFrame operations per race
5. Iterates over `result_df` rows to log diagnostics (lines 545-591, 632-678) using `.iterrows()`

For a year of ~5000 races, this is 5000 iterations of DataFrame filtering, merging, and row-level iteration. The per-race overhead compounds.

**Why it happens:** The original design processes races one-by-one because bet settlement needs sequential bankroll tracking. But the feature engineering and model inference could be batched.

**Prevention:**
- Batch model inference: instead of calling `predict()` per race, call it once on the full `feat_df` with a `race_id` groupby. LightGBM/XGBoost inference is much faster in batch mode.
- Pre-compute and cache `predict()` results for all races before entering the settlement loop.
- The settlement loop (bankroll tracking) must remain sequential but should be lightweight (just arithmetic, no DataFrame operations).
- Replace `.iterrows()` in diagnostic logging with vectorized column extraction.
- Replace the `for _, row in payouts_df.iterrows()` in `build_payout_map()` with vectorized `zip()` and list comprehension.

**Detection:**
- Profile `engine.py` run time. If per-race processing exceeds 0.1 seconds (500+ seconds total for 5000 races), the loop overhead is significant.
- Check if `predict()` call dominates the per-race time (it likely does due to DataFrame copies and merges).

**Phase:** Phase 3 (pipeline optimization).

**Sources:**
- Codebase analysis: `engine.py:420-785`, `race_predictor.py:51-222`
- HIGH confidence: Per-race loop pattern is directly observable; ~5000 iterations with DataFrame operations per iteration

---

### Pitfall 10: `build_payout_map()` Uses `iterrows()` Over Full Payouts DataFrame

**What goes wrong:** `build_payout_map()` at `engine.py:112` iterates row-by-row with `for _, row in payouts_df.iterrows()`. For a year of data with ~5000 races and 5 place payout entries each, this processes ~25,000 rows via slow Python iteration. The same pattern exists in `build_wide_payout_map()` with up to 35,000 iterations (7 wide entries x 5000 races).

**Why it happens:** The function was written for correctness, not performance. `iterrows()` is the slowest pandas iteration method (creates a Series per row).

**Prevention:**
- Replace with vectorized operations:
```python
# Win payout map (vectorized)
tansyo_cols = ["race_id", "paytansyoumaban1", "paytansyopay1"]
t = payouts_df[tansyo_cols].dropna()
t["key"] = t["race_id"].astype(str) + "_" + t["paytansyoumaban1"].astype(int).astype(str)
t["val"] = t["paytansyopay1"].astype(float) / 100.0
win_payout_map = dict(zip(t["key"], t["val"]))
```
- Use `itertuples()` as a middle ground if full vectorization is complex.
- For the wide payout map, parse kumi strings with `.str` operations.

**Detection:**
- Time `build_payout_map()` and `build_wide_payout_map()`. If they take > 5 seconds combined, optimization is needed.

**Phase:** Phase 3 (pipeline optimization), quick win.

**Sources:**
- Codebase analysis: `engine.py:102-168`
- HIGH confidence: `iterrows()` performance impact is well-documented

---

### Pitfall 11: Training Pipeline Recomputes Features Every Run (No Feature Caching)

**What goes wrong:** `TrainingPipelineV5.run()` always recomputes ALL features from raw data (load races -> load entries -> load odds -> build_all -> add submodel features). For a 4-year training window (~200K entries), this takes significant time. When running WF validation with 2 folds, the same feature computation runs twice for overlapping periods.

**Why it happens:** No feature caching mechanism exists. The pipeline was designed for simplicity, not iteration speed. The existing `data/features/horse_features.parquet` cache is referenced in `readers.py:297-300` but is never written to by the training pipeline.

**Consequences:** Each training run takes ~44 minutes (per CLAUDE.md). WF validation with 2 folds takes ~4 hours. Development iteration is slow.

**Prevention:**
- Cache `feat_df` (before submodel-specific features) to `data/features/horse_features.parquet` after the first computation. Subsequent runs load from cache if the input data range matches.
- Add a `--cache-features` flag to `run_train.py` and `run_wf_validation.py`.
- Hash the input parameters (date range, feature version) into a cache key. If the key matches, skip feature computation.
- NOTE: Submodel-specific features (HorseHistoryFeatures, PaceAptitudeFeatures, etc.) depend on the training data and cannot be simply cached. Only cache the base features from `FeatureEngine.build_all()`.

**Detection:**
- Add timing instrumentation. If `build_all()` takes > 10 minutes, feature caching is worth implementing.

**Phase:** Phase 3 (pipeline optimization).

**Sources:**
- Codebase analysis: `training_pipeline.py:84-188`, `readers.py:297-300`
- HIGH confidence: No caching mechanism observable in pipeline code

---

### Pitfall 12: Optuna Hyperparameter Search In Ensemble Adds 2-3x Training Time

**What goes wrong:** The v1.1 ensemble uses Optuna for per-model hyperparameter optimization (per PROJECT.md). This adds significant training time on top of the already expensive 3-model stacking. Combined with the lack of feature caching (Pitfall 11), a single WF fold could take 2+ hours.

**Why it happens:** Optuna runs multiple trials per model, each requiring a full model training. With 3 base models x N trials x K OOF folds, the trial count multiplies.

**Prevention:**
- For development iterations, disable Optuna and use fixed reasonable hyperparameters. Only run Optuna for the final production model.
- Add a `--quick-train` flag that skips Optuna, uses fewer OOF folds (2 instead of 5), and reduces boost rounds (200 instead of 300).
- Cache Optuna results: if hyperparameters have been tuned for a given data range, reuse them instead of re-optimizing.

**Detection:**
- If a single training run exceeds 90 minutes, Optuna is likely the bottleneck.

**Phase:** Phase 3 (pipeline optimization).

**Sources:**
- PROJECT.md context: "学習時間がOptunaチューニングにより推定2-3倍に増加"
- HIGH confidence: Optuna overhead is documented in project context

---

### Pitfall 13: Diagnostic CSV `iterrows()` Calls In Per-Race Loop Are Expensive

**What goes wrong:** Within the per-race loop, the engine calls `diag_logger.log_horse()` for every horse in every race (lines 545-591 for quality-failed races, lines 632-678 for quality-passed races). Each call uses `hr.to_dict()` on a row from `result_df.iterrows()`. For a race with 14 horses, that is 14 dict constructions per race, 70,000 total for 5000 races. The dict construction is slow because it includes all feature columns (~120+).

**Why it happens:** Diagnostics were designed for completeness, not performance. Logging every horse's every feature is overkill for routine backtests.

**Prevention:**
- Make diagnostic logging optional. Add a `--diagnostics` flag to `run_backtest.py`. Only enable for debugging.
- When diagnostics are disabled, skip the entire `log_horse` / `log_horse_features` calls.
- When diagnostics are enabled, only log the bet-relevant columns (not all ~120 feature columns).
- The `log_race()` call is lightweight (1 per race) and can always run.

**Detection:**
- Profile `diag_logger.log_horse()` total time. If it exceeds 10% of total backtest time, it is worth optimizing.

**Phase:** Phase 3 (pipeline optimization).

**Sources:**
- Codebase analysis: `engine.py:545-591, 632-678`
- HIGH confidence: `iterrows()` + `to_dict()` overhead well-documented

---

## Minor Pitfalls

### Pitfall 14: `run_backtest.py` Report Compares Against Place Baseline ROI

**What goes wrong:** The `before_roi = 0.638` hardcoded at line 233 and the display logic at lines 233-243 compare backtest ROI against a place-model baseline. This is misleading when testing win bets.

**Prevention:** Replace with a win-model baseline or make it configurable via command-line argument.

**Phase:** Phase 2 (analysis).

---

### Pitfall 15: Bet History Records Place-Specific Fields For Win Bets

**What goes wrong:** `bet_history` entries at `engine.py:699-752` record `p_place_pred`, `e_return_place_pred` in the bet history. For win validation, these should be `p_win_pred`, `e_return_win_pred`, `ev_win`, `ev_win_corrected`, `EV_lower_win_corrected`. The multi-year JSON report and parquet output will contain the wrong prediction values.

**Prevention:** Add win-specific fields to bet_history entries. Record both win and place metrics for completeness, or switch based on the bet type.

**Phase:** Phase 2 (analysis/reporting).

---

### Pitfall 16: `backtest_result.json` Does Not Distinguish Win vs. Place Mode

**What goes wrong:** The JSON output from `run_backtest.py` does not record which betting mode was used. If someone runs a place backtest and later a win backtest, the JSON files look identical in structure. Comparing them requires external knowledge.

**Prevention:** Add `betting_target: "win"` or `"place"` field to the JSON output.

**Phase:** Phase 2 (reporting).

---

### Pitfall 17: `_generate_bets()` Legacy Method Still Uses Place-Only Logic

**What goes wrong:** `_generate_bets()` at `engine.py:870-907` is a legacy method that creates only `BetType.PLACE` bets using `fukuoddslow`. It is marked as "kept for compatibility" (tests may reference it). If any code path falls back to this method, it will generate place bets regardless of the intended mode.

**Prevention:** Ensure all code paths use `RacePredictor.select_bets()` and never fall back to `_generate_bets()`. Add a deprecation warning.

**Phase:** Phase 1 (ensure no regression).

---

### Pitfall 18: `compute_roi_ema()` And Market Bias Features May Use Wrong Odds Column

**What goes wrong:** Various feature computations use `tanodds` (win odds) for some calculations and `fukuoddslow` (place odds) for others. When the betting target is win, some feature computations that use `fukuoddslow` in the EV pipeline (e.g., `compute_market_bias`, Benter combination at `race_predictor.py:158-193`) need to use `tanodds` instead. If this is not consistent, the EV estimates will mix win and place odds.

**Why it happens:** The Benter combination at `race_predictor.py:158-193` computes `p_market = 1.0 / fukuoddslow` which is the market-implied PLACE probability. This feeds into `edge_place = p_combined * fukuoddslow - 1.0`. This is correct for place but wrong for win. A parallel path using `1.0 / tanodds` and `ev_win` must be used for win bets.

**Prevention:** The Win Benter Combination (`WinBenterGate.apply()`) at `race_predictor.py:119-128` already handles this correctly using tanodds. Ensure `get_win_candidates()` uses the Win Benter outputs rather than the place Benter outputs.

**Phase:** Phase 1 (verify win Benter is used in win path).

**Sources:**
- Codebase analysis: `race_predictor.py:119-128, 158-193`, `win_benter_gate.py`
- HIGH confidence: Separate win and place Benter paths observable

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Win payout map creation | Pitfall 1: Using place payouts for win settlement | Create `build_win_payout_map()` reading `paytansyoumaban1/paytansyopay1` |
| Final odds map for win | Pitfall 2: Using `fukuoddslow` for win odds | Build separate map from `tanodds` or `confirmed_odds` |
| Win bet generation | Pitfall 3: No `get_win_candidates()` or win bet path | Create win candidate selection using `WinSelectionGate` outputs |
| WF validation mode | Pitfall 5: Engine defaults to place bets | Add betting target parameter to engine and WF script |
| Diagnostic logging | Pitfall 4: Logging place metrics only | Add win metric logging path |
| Edge threshold calibration | Pitfall 8: Place thresholds applied to win edges | Add win-specific regime parameters |
| Per-race loop performance | Pitfall 9: 5000 DataFrame operations in loop | Batch inference, cache predict results |
| Payout map performance | Pitfall 10: `iterrows()` on payouts | Vectorize payout map construction |
| Feature caching | Pitfall 11: Recomputing features every run | Cache `feat_df` between training runs |
| Optuna overhead | Pitfall 12: 2-3x training time from HP search | Add `--quick-train` flag for development |
| Diagnostic overhead | Pitfall 13: Per-horse `to_dict()` in loop | Make diagnostics optional |
| Bet history analysis | Pitfall 15: Place-specific fields in history | Add win fields to bet_history entries |
| Report baseline | Pitfall 14: Hardcoded place baseline ROI | Make baseline configurable or remove |

---

## Priority Action Items (Ordered by Impact)

1. **Build `build_win_payout_map()`** (Pitfall 1) -- Without this, ALL win backtest ROI numbers are wrong. Must read `paytansyoumaban1` and `paytansyopay1`.

2. **Fix `final_odds_map` to use `tanodds` for win** (Pitfall 2) -- Settlement fallback uses place odds. Must dispatch by bet type.

3. **Create `get_win_candidates()` and win bet generation** (Pitfall 3) -- Without this, no win bets are generated. Use `WinSelectionGate` outputs.

4. **Add betting target parameter to `BacktestEngine`** (Pitfall 5) -- Engine must know whether to generate win or place bets. Propagate to `RacePredictor`.

5. **Add win diagnostic logging** (Pitfall 4) -- After backtest runs, diagnostics must show win metrics for analysis.

6. **Batch model inference in backtest loop** (Pitfall 9) -- The biggest performance win for the optimization phase. Call `predict()` once on full DataFrame instead of per-race.

7. **Vectorize payout map construction** (Pitfall 10) -- Quick performance win, easy to implement.

8. **Add feature caching to training pipeline** (Pitfall 11) -- Largest time savings for iteration speed. Cache base features between runs.

9. **Add `--quick-train` mode** (Pitfall 12) -- Disable Optuna for development iterations. Saves 2-3x per training run.

10. **Make diagnostics optional** (Pitfall 13) -- Disable by default for routine backtests. Saves ~10% of loop time.

---

## Dependency Graph

```
Pitfall 1 (payout map) ──┐
Pitfall 2 (final odds) ──┤── Must be done FIRST (Phase 1)
Pitfall 3 (bet gen)    ──┤
Pitfall 5 (WF mode)    ──┘
         │
         ▼
Pitfall 4 (diagnostics) ── Phase 1-2 (alongside first backtest run)
Pitfall 6 (fallback)    ── Phase 1 (resolved by Pitfall 2 fix)
Pitfall 17 (legacy)     ── Phase 1 (verify no regression)
Pitfall 18 (Benter)     ── Phase 1 (verify win path)
         │
         ▼
Pitfall 7 (baseline)    ── Phase 2 (analysis)
Pitfall 8 (thresholds)  ── Phase 2 (calibration)
Pitfall 14 (baseline)   ── Phase 2
Pitfall 15 (history)    ── Phase 2
Pitfall 16 (JSON)       ── Phase 2
         │
         ▼
Pitfall 9  (loop)       ── Phase 3 (optimization)
Pitfall 10 (payout)     ── Phase 3
Pitfall 11 (cache)      ── Phase 3
Pitfall 12 (Optuna)     ── Phase 3
Pitfall 13 (diags)      ── Phase 3
```

---

## Sources

### HIGH confidence (codebase analysis + directly observable patterns)
- `src/backtest/engine.py:102-168` -- payout map construction using place-only columns
- `src/backtest/engine.py:276-281` -- final odds map using `fukuoddslow`
- `src/backtest/engine.py:420-785` -- per-race loop with DataFrame operations
- `src/backtest/engine.py:909-950` -- settlement logic dispatching
- `src/backtest/race_predictor.py:408-642` -- place-only candidate selection and bet generation
- `src/backtest/race_predictor.py:119-143` -- WinSelectionGate application (correct win path exists)
- `src/db/etl.py:112-113` -- win payout columns in schema (`paytansyoumaban1`, `paytansyopay1`)
- `src/domain/types.py:13-18` -- BetType enum includes WIN
- `scripts/run_wf_validation.py:171-195` -- WF backtest engine instantiation
- `scripts/run_backtest.py:233` -- hardcoded place baseline ROI

### MEDIUM confidence (needs empirical validation)
- Pitfall 8: Win edge threshold calibration -- needs win model edge distribution data
- Pitfall 11: Feature caching time savings -- needs profiling to confirm `build_all()` is the bottleneck
- Pitfall 12: Optuna overhead fraction -- needs timing data from v1.1 training runs

---
*Research completed: 2026-05-04*
*Ready for roadmap: yes*
