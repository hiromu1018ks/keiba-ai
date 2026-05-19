# Domain Pitfalls

**Domain:** Horse racing prediction system (keiba-ai v5.5) -- adding 5 feature enhancements to an existing system that has already survived 8 milestones and 34 phases
**Researched:** 2026-05-19
**Context:** v1.8 Turf Precision Calibration -- haron/lap ETL, popularity band calibration, turf relative features, condition interactions, regime x surface EV correction

## Critical Pitfalls

Mistakes that cause model invalidation or silent data leakage.

### Pitfall 1: POST_RACE Leakage in Haron/Lap Time Features

**What goes wrong:** Using the *current race's* haron time (harontimel3/harontimel4) or lap times (LapTime1~25) as features for predicting the current race. These values are only known after the race finishes -- they are the outcome, not the input.

**Why it happens:** The haron/lap columns already exist in the entries Parquet alongside pre-race columns. When building history-based features (e.g., "average closing speed"), a careless implementation will merge the current race's data into the horse's history *before* computing aggregates, thereby including the outcome in the feature.

The existing PaceAptitudeFeatures.compute_batch() already uses `np.searchsorted(h_dates, target_dates, side='left')` for PIT safety (pace_aptitude_features.py line 232), but any new haron/lap feature module must independently replicate this pattern. There is no shared utility for PIT-safe history aggregation -- each feature module implements it independently.

**Consequences:** Model shows artificially high backtest ROI (often 120%+), then collapses in live paper trading. The v1.6 milestone explicitly built a 3-layer CI detection system (test_post_race_leakage.py) to prevent this -- any new feature that bypasses it invalidates the entire validation framework.

**Prevention:**
1. Add `harontimel4` and all `LapTime*` columns to `POST_RACE_COLS` in `domain/types.py`. Currently only `harontimel3` is listed (line 45). The `harontimel4` column is already present in the ETL float rules (etl.py `_TABLE_TYPE_RULES`) but is NOT in POST_RACE_COLS -- a gap.
2. When computing haron-based history features, use the same `searchsorted(h_dates, target_dates, side='left')` pattern as PaceAptitudeFeatures. The `side='left'` ensures the current race date is excluded.
3. The ETL already converts harontimel3 to float in `_TABLE_TYPE_RULES` (etl.py line 97), but LapTime columns are NOT in any type conversion rule -- they must be added to the `"entries"` rules.
4. Add new haron-derived feature column names to the 3-layer CI test (`test_post_race_leakage.py`) Layer 2 check, verifying they are not in any model's FEATURE_COLS.

**Detection:**
- Run the existing 3-layer POST_RACE test suite after implementation.
- Compare backtest ROI with and without haron features. If haron features boost ROI by >10pp in backtest but paper trading ROI stays flat, suspect leakage.
- Check that new feature values for the last race in a horse's history are NaN (not the race outcome).

### Pitfall 2: Look-Ahead Bias in Popularity Band Calibration

**What goes wrong:** Computing popularity band calibration ratios using OOF residuals that include future data. If the calibration is computed globally (e.g., "horses ranked 1-3 popularity have average residual X") using all OOF data at once rather than in a time-ordered expanding window, it leaks information.

**Why it happens:** The existing EV calibration pipeline already uses K-fold OOF in `generate_ev_oof_predictions()` (training_pipeline.py line 1102), which is correct for the isotonic calibrator. But popularity band calibration is a *new* layer that segments by popularity rank and applies residual scaling. If this segmentation uses the full OOF residual distribution (including fold 4 residuals when calibrating fold 1), it leaks information.

The v1.4 milestone already fixed this exact pattern in OddsBandFilter: the `calibrate()` method was changed to use `training_bet_history` generated with default parameters (v1.4 Key Decision). Popularity band calibration must follow the same discipline.

**Consequences:** Overconfident EV estimates in backtest that fail to reproduce in forward testing. The calibration appears to work because it "knows" which popularity bands were profitable in the test period.

**Prevention:**
1. Compute popularity band calibration ratios using ONLY expanding-window or rolling-window past data. The training_pipeline already sorts by race_date for PIT safety (line 240-241).
2. If using OOF residuals for calibration, ensure the residual computation for each fold uses only data from *earlier* folds. This means popularity_band_calibration must be computed inside the OOF loop, not after.
3. Alternative safe approach: compute band ratios on the training set only (before OOF), then apply to OOF predictions. This is less precise but guaranteed leak-free.
4. The OddsBandFilter.calibrate() method already uses `bet_history` which is generated with default parameters to avoid look-ahead. Popularity band calibration must follow the same pattern.

**Detection:**
- Compare calibration ratios computed with expanding window vs. global computation. Large discrepancies indicate leakage.
- Run the existing `test_ev_correction_odds_col_uses_pre_race_odds` test pattern to verify no post-race odds are used.
- If popularity band calibration alone boosts ROI by >5pp, suspect leakage.

### Pitfall 3: Regime-Surface Circular Dependency

**What goes wrong:** Adding `regime` state as a feature to the EV correction model creates a potential circular dependency: the EV model's output determines betting decisions, which affect recent race outcomes, which feed into RegimeDetector, which classifies the regime, which feeds back into the EV model.

**Why it happens:** The RegimeDetector.detect() method (regime_detector.py line 167) uses `recent_stats` -- aggregated statistics from recent races. During training, this is not a problem because regime is computed from market data (pre-race odds features only), not from model predictions. But during live inference, regime depends on recent results which may be influenced by the model's own betting behavior in paper trading.

The current RegimeDetector.FEATURE_COLS (regime_detector.py lines 60-89) uses ONLY market-level pre-race features (market_error_std, overround_rolling, odds_skewness, rl_* race-level features, etc.). It does NOT use model outputs. This isolation is correct and must be maintained.

**Consequences:** In live paper trading, the model may exhibit oscillating regime-dependent corrections. If regime shifts to AGGRESSIVE, EV corrections amplify, bets increase, losses accumulate, regime shifts to CONSERVATIVE, corrections shrink, bets decrease, losses recover, regime shifts back -- creating a feedback loop.

**Prevention:**
1. When adding regime-derived features to EVCorrectionModel.FEATURE_COLS, ensure the regime value is the *market-derived* regime (computed from pre-race odds), NOT a feedback loop from betting results.
2. The regime is already race-level (same for all horses in a race) -- verify that adding it to EV correction does not create per-horse regime interactions that could amplify feedback.
3. Add regime to EVCorrectionModel.FEATURE_COLS only after the RegimeDetector is trained and its predictions are stable (i.e., regime is computed from market features that are independent of the EV model's output).
4. Test: force regime transitions in backtest and check for ROI oscillation. If ROI swings >5pp per forced transition, the feedback is too strong.

**Detection:**
- Simulate regime transitions in backtest by forcing regime state changes. If ROI oscillates wildly with forced regime shifts, the feedback loop is too strong.
- Check that regime detector's training features (line 60-89) have zero overlap with EV model outputs.

### Pitfall 4: HaronTimeL3/L4 Data Quality -- Sentinel Values (000/999)

**What goes wrong:** EveryDB2 stores measurement failures and special cases as 000 or 999 values in harontimel3/harontimel4 columns. Treating these as legitimate times (e.g., 0.0 seconds or 99.9 seconds) introduces extreme outliers that corrupt averages, z-scores, and any model that sees them.

**Why it happens:** The ETL pipeline currently converts harontimel3 to float via `_to_float()` (etl.py line 97) without any sentinel value handling. A value of "000" becomes 0.0, and "999" becomes 999.0. Both are physically impossible (normal haron times are 32-42 seconds for 3 furlongs). The existing PaceAptitudeFeatures (pace_aptitude_features.py line 270-273) uses `ht_valid_pace = ht_past[~np.isnan(ht_past)]` which catches NaN but NOT 0.0 or 999.0.

**Consequences:** "Average haron time" features get wildly wrong values. A single 0.0 value can pull the average below 30s, making a horse look like it has superhuman closing speed. A 999.0 value pushes averages above 100s. Both corrupt the model's learned relationship between haron time and winning probability.

**Prevention:**
1. In the ETL pipeline, add sentinel value handling: replace harontimel3/harontimel4 values of 0.0 and >= 99.0 with NaN during `_apply_type_conversions()`.
2. In feature computation, add an explicit validity check: `valid_mask = (haron > 30.0) & (haron < 50.0)` for harontimel3 (3 furlongs) and `valid_mask = (haron > 40.0) & (haron < 70.0)` for harontimel4 (4 furlongs).
3. For LapTime features, JRA uses similar sentinel patterns. Each LapTime should be validated against a reasonable range (e.g., 10-20 seconds per furlong).
4. Document the expected range for each column in the feature module's docstring.

**Detection:**
- After ETL, check `data/raw/entries.parquet` for harontimel3 values outside [30, 50] range. Count should be 0 after sentinel handling.
- In feature computation, log the percentage of valid haron values per horse. Horses with <50% valid history should get NaN features rather than unreliable averages.
- Add a data quality assertion in tests: after ETL, assert harontimel3 is either NaN or in [30, 50].

## Moderate Pitfalls

### Pitfall 5: Overfitting with Interaction Features

**What goes wrong:** Adding too many interaction features (grade x form, distance x closing, etc.) for the dataset size. The system already has 12 interaction features (interaction_features.py) plus ~179 total features. Adding more interactions without pruning can exceed the model's effective dimensionality.

**Why it happens:** Interaction features multiply the feature space combinatorially. `grade_code` has ~8 unique values, form features are continuous -- their product creates ~8 new feature distributions. With a dataset of ~50K races (entries.parquet 2015-2025), adding 10+ interaction features with rare categories can leave some combinations with <100 samples.

**Prevention:**
1. Use the existing IC evaluation framework (B-difference / C-orthogonal / E-incremental) to validate each new interaction independently. Only keep interactions with E-incremental IC > 0.
2. Limit interaction features to combinations with strong domain justification: grade x form (graded race form is more predictive); distance x closing (closing speed matters more in longer races).
3. Apply the existing GPD diagnostics (FEATURE_CATEGORY_MAP) to track new interactions' contribution. If MDR/FAD metrics show no contribution, prune immediately.
4. The existing system already has `feature_fraction=0.7` in LightGBM configs, which provides implicit feature selection. But this is insufficient if 90% of new features are noise.

**Detection:**
- Run IC evaluation before and after adding interactions. If E-incremental IC is negative, the interaction is hurting.
- Check LightGBM feature importance: if new interactions consistently rank in the bottom 20%, they are noise.

### Pitfall 6: Turf-Specific Features That Do Not Generalize to Dirt

**What goes wrong:** Building features that work well for turf races (the majority in JRA) but actively harm dirt race predictions. The v1.8 milestone explicitly targets turf model improvement (turf b_difference is -0.004, target positive), but features must not harm dirt model performance (currently profitable at 107.4% ROI).

**Why it happens:** Turf races in JRA have distinct characteristics: more distance variation, more pronounced track condition effects, different running styles. Features derived from these patterns (e.g., turf-specific closing speed rankings) may encode turf-specific biases that LightGBM cannot fully separate via the surface submodel.

**Prevention:**
1. Train and evaluate turf and dirt submodels separately (already done via SubModelManager). Ensure new features are evaluated on BOTH surfaces.
2. For turf-specific features, add surface conditioning: `feature_value = np.where(surface == 'turf', computed_value, np.nan)`. This prevents dirt races from getting noisy turf-derived values.
3. The existing `haron_x_distance` interaction (interaction_features.py line 114) is already surface-agnostic. New turf-specific interactions should explicitly include surface in the interaction.
4. Monitor dirt ROI after each new feature addition. If dirt ROI drops below 100%, the feature is hurting and must be modified or gated.

**Detection:**
- Compare b_difference for turf and dirt separately after each feature addition.
- Run the IC evaluation framework with surface stratification.

### Pitfall 7: Feature Cache Invalidation After ETL Schema Changes

**What goes wrong:** The FeatureEngine uses a code-hash-based cache (feature_engine.py line 37-58). When new haron/lap ETL columns are added to the entries Parquet, the cache key includes input_paths timestamps and code_hash. If the ETL atomically replaces the parquet file (which it does -- etl.py line 122-124), the timestamp changes and cache invalidation works correctly. But if the entries.parquet gains new columns (harontimel4, LapTime) while the code has not been updated to reference them, the cache may serve features that include the new columns as unused NaN, masking the fact that the data is available.

**Why it happens:** `compute_cache_key()` uses input_paths, date_range, feature_type, and code_hash. It does NOT include the schema (column list) of the source parquet files. If the entries.parquet gains new columns but the feature code has not changed, the code_hash stays the same and the cache key is identical -- but the data is different.

**Prevention:**
1. After ETL changes that add columns, explicitly delete the feature cache directory (`data/features/cache/`) before re-training.
2. Add the entries.parquet schema (column list hash) to the cache key computation in `compute_cache_key()`.
3. Add a cache validation check that verifies the cached DataFrame has all expected columns.

**Detection:**
- After ETL + training, check the feature cache hit/miss log. A cache HIT immediately after ETL schema changes is suspicious.
- Compare the number of columns in cached features vs. freshly computed features.

### Pitfall 8: Inference Path Missing New Feature Computation

**What goes wrong:** `FeatureEngine.build_features()` (the single-race inference method, feature_engine.py line 414) is a SEPARATE code path from `build_all()` (the batch training method). The inference path currently computes only basic features via `_map_basic_features()`, flb_slope, race_level_features, and market_cross_features. If new features (haron history, interactions, regime x surface) are added only to `build_all()` or the training pipeline's submodel path, they will be missing at inference time.

**Why it happens:** The current design intentionally separates batch and inference paths. Haron/lap history features and interaction features are computed in the TrainingPipeline's `_train_submodel()` method (not in FeatureEngine at all). This means they are also missing from the BacktestEngine's inference path -- which is fine because backtest re-trains each year. But live paper trading via BettingOrchestrator calls `build_features()`, which would be missing new features.

**Prevention:**
1. Any new feature computed in the training pipeline must also be computable in the inference path. For haron/lap history features, the inference path needs access to ParquetStore to load historical entries.
2. Add an explicit test that verifies feature parity between training and inference paths for new columns.
3. Consider refactoring: move feature computation from TrainingPipeline into FeatureEngine so both paths share the same code.

**Detection:**
- Live predictions producing different rankings than backtest predictions.
- NaN values for new features in the inference path but not in training.

## Minor Pitfalls

### Pitfall 9: LapTime ETL Schema Discovery

**What goes wrong:** The `n_jyusyosiki` table in EveryDB2 contains lap time data (LapTime1~25), but the column names and count are not yet verified. If the table has fewer than 25 LapTime columns (shorter races have fewer laps), the ETL and feature code must handle variable column counts.

**Prevention:**
1. Before implementing LapTime ETL, query EveryDB2 to discover the actual schema: `SELECT column_name FROM information_schema.columns WHERE table_name = 'n_jyusyosiki'`.
2. Design the feature computation to handle variable LapTime counts per race.
3. The `n_jyusyosiki` ETL config is already defined (etl_tables.yaml line 224-225) with `jyuni` as a positional column -- verify this matches the actual data structure.

### Pitfall 10: Popularity Band Boundary Effects

**What goes wrong:** The popularity band calibration uses fixed band boundaries (e.g., popularity rank 1-3, 4-6, 7-9, 10-12, 13+). Horses near the boundary get unstable calibration factors -- a horse ranked 3rd in popularity gets a very different correction than one ranked 4th, even though their actual winning probability differs only slightly.

**Prevention:**
1. Use overlapping bands or smooth boundary transitions (e.g., triangular window centered on the boundary).
2. Alternative: use continuous calibration based on popularity rank rather than discrete bands.

### Pitfall 11: Interaction Feature NaN Propagation

**What goes wrong:** New interaction features multiply two base features. If either base feature is NaN (which is common -- many features have 10-30% missing rates), the interaction becomes NaN. This can dramatically increase the NaN rate of the feature matrix, reducing effective training data.

**Prevention:**
1. The existing interaction_features.py already uses `.where()` with NaN checks (lines 57-59, 89-91, etc.). New interactions must follow the same pattern.
2. For categorical interactions (string concatenation), handle NaN by producing "unknown_X" or "X_unknown" rather than NaN.

### Pitfall 12: POST_RACE_COLS Whitelist Completeness

**What goes wrong:** The POST_RACE_COLS list in domain/types.py (lines 38-55) is a whitelist that must be complete. If LapTime columns are added to the ETL but not to POST_RACE_COLS, the SAFE-01 guard in FeatureEngine.build_all() (line 393) will NOT drop them, and they could leak into features. Currently missing from POST_RACE_COLS: `harontimel4`, `dmjyuni`, `dmtime` (partially present), and all future LapTime columns.

**Prevention:**
1. When adding any new POST_RACE column to the ETL, immediately add it to POST_RACE_COLS.
2. Add a CI test that verifies all columns in entries.parquet that are NOT in any model's FEATURE_COLS are either in POST_RACE_COLS or are explicitly documented as pre-race.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation | Priority |
|-------------|---------------|------------|----------|
| A: HaronTime ETL | Pitfall 4 (sentinel values), Pitfall 12 (whitelist gap) | Add sentinel handling in ETL; add harontimel4 + LapTime to POST_RACE_COLS | CRITICAL |
| A: LapTime ETL | Pitfall 9 (schema discovery), Pitfall 1 (leakage) | Query actual EveryDB2 schema first; add LapTime to POST_RACE_COLS | CRITICAL |
| A: Haron/Lap feature computation | Pitfall 1 (leakage), Pitfall 4 (data quality) | Replicate PaceAptitudeFeatures PIT pattern; add range validation | CRITICAL |
| B: Popularity band calibration | Pitfall 2 (look-ahead bias), Pitfall 10 (boundary effects) | Use expanding-window OOF calibration; consider continuous calibration | HIGH |
| C: Turf relative features | Pitfall 6 (turf overfitting) | Evaluate on both surfaces; add surface conditioning | MEDIUM |
| D: Condition interaction features | Pitfall 5 (overfitting), Pitfall 11 (NaN propagation) | Validate with E-incremental IC; use .where() for NaN safety | MEDIUM |
| E: Regime x surface EV correction | Pitfall 3 (circular dependency) | Verify regime features are market-derived only; test forced regime transitions | HIGH |
| All phases: Cache | Pitfall 7 (cache invalidation) | Delete feature cache after ETL schema changes; add schema hash to cache key | LOW |
| All phases: Inference path | Pitfall 8 (missing inference features) | Verify feature parity between training and inference | HIGH |
| All phases: CI tests | Pitfall 12 (test coverage gaps) | Add new feature column names to 3-layer POST_RACE test | CRITICAL |

## Critical Integration Checklist

Before merging any v1.8 feature, these checks MUST pass:

1. **POST_RACE_COLS update:** Verify that harontimel4 and all LapTime columns are added to `domain/types.py` POST_RACE_COLS list.

2. **3-layer CI test update:** Add new feature column names to `test_post_race_leakage.py` Layer 2 model coverage checks.

3. **ETL type rules update:** Add LapTime columns to `_TABLE_TYPE_RULES["entries"]["float"]` in etl.py.

4. **Sentinel value handling:** Verify that harontimel3/harontimel4 values of 0.0 and >=99.0 are replaced with NaN in ETL.

5. **PIT pattern replication:** Any feature computed from horse history must use `searchsorted(dates, target_dates, side='left')` or equivalent to exclude the current race.

6. **Surface submodel evaluation:** After each feature addition, evaluate both turf and dirt models separately. Dirt ROI must not drop below 100%.

7. **Feature cache bust:** Delete `data/features/cache/` after any ETL schema change.

8. **Inference path parity:** Verify `build_features()` and `build_all()` produce the same new columns for a single race.

## Sources

- Code analysis: `src/domain/types.py` (POST_RACE_COLS definition -- harontimel3 present, harontimel4 missing), `src/features/pace_aptitude_features.py` (PIT-safe history pattern via searchsorted side='left'), `src/models/regime_detector.py` (regime feature isolation from model outputs), `src/models/ev_correction_model.py` (FEATURE_COLS, isotonic calibration, odds band scaling), `src/features/interaction_features.py` (existing 12 interaction patterns with NaN-safe .where()), `src/db/etl.py` (type conversion rules -- harontimel3 float, no LapTime rules), `config/etl_tables.yaml` (n_jyusyosiki table definition), `tests/test_post_race_leakage.py` (3-layer CI test framework), `src/pipelines/training_pipeline.py` (OOF calibration, time-series split), `src/betting/odds_band_filter.py` (v1.4 look-ahead bias fix pattern)
- Milestone history: v1.6 POST_RACE leakage prevention (3-layer CI), v1.4 look-ahead bias fix in OddsBandFilter, v1.7 race-level and market-cross feature patterns
- Confidence: HIGH (all findings verified against source code)
