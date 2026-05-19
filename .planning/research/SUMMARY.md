# Project Research Summary

**Project:** keiba-ai v1.8 Turf Precision Calibration
**Domain:** Horse racing ML prediction system -- turf model improvement for ROI 97.8% to 100%+
**Researched:** 2026-05-19
**Confidence:** HIGH

## Executive Summary

keiba-ai v1.8 targets the turf model's negative IC b_difference (-0.004), the single largest bottleneck preventing overall ROI from crossing the 100% profitability threshold. The system is a mature LightGBM-based prediction pipeline (8 milestones, 34 phases, ~24K LOC) with a well-established PIT (point-in-time) safety architecture. The v1.8 milestone adds 5 feature groups -- all implementable within the existing technology stack with zero new dependencies.

The recommended approach is a dependency-ordered build: ETL data extraction first (making HaronTimeL4 and LapTime columns available as floats in Parquet), then feature computation (turf relative features + conditional interactions in parallel), then EV calibration layers (popularity band calibration + regime-surface correction), then integrated validation. Four of the five features are low-risk extensions of existing patterns (add_race_transforms, INTERACTION_COLS, FEATURE_COLS). The fifth -- popularity band calibration -- is a novel calibration layer that requires careful OOF-only validation to avoid look-ahead bias.

The dominant risk is POST_RACE data leakage. HaronTimeL4, LapTime1~25, and related columns are outcome data known only after a race finishes. The existing 3-layer PIT defense (feature drop, computation guard, CI tests) must be extended to cover every new feature. A secondary risk is the HaronTimeL3/L4 sentinel value problem (000/999 stored as 0.0/999.0 in Parquet), which must be handled during ETL to prevent corrupted feature averages.
## Key Findings

### Recommended Stack

No new dependencies required. All 5 feature groups use existing libraries (pandas 2.3.3, numpy 2.4.3, LightGBM 4.6.0, scikit-learn 1.8.0). The only infrastructure action is an ETL re-extraction to convert HaronTimeL4 and LapTime1~25 from strings to floats in Parquet files.

**Core technologies:**
- pandas `groupby` + `transform`: Race-rank computation, popularity band aggregation, lap time aggregation -- already the foundation for all feature modules
- pandas `cut` (conceptual): Fixed popularity band binning with domain-knowledge boundaries (1-3, 4-6, 7-9, 10-12, 13+)
- numpy `polyfit`: Lap time trend features using existing pattern from haron_zscore_trend
- LightGBM native categorical handling: Regime-surface interaction learned automatically when both columns are categorical
- SQLAlchemy `text()`: ETL queries already use `SELECT *` -- only type conversion rules need updating

### Expected Features

**Must have (table stakes for v1.8 goal):**
- HaronTimeL4-based closing speed features (harontimel4_avg, harontimel4_zscore, haron_l3l4_ratio) -- extends existing `_compute_haron_stats` pattern
- Turf race-rank features (7 new _race_rank columns for form_trend, blood_total_wr, etc.) -- extends existing `add_race_transforms`
- Conditional interaction features (grade_x_form_trend, distance_x_closing_index) -- extends existing `interaction_features.py`
- Regime x surface EV correction -- adds regime_state to EVCorrectionModel.FEATURE_COLS

**Should have (high-impact differentiators):**
- Popularity band calibration (per-band EV scaling from OOF residuals) -- new calibration layer between Isotonic and OddsBand
- Lap pace features (lap_early_ratio, lap_closing_ratio from LapTime1~25) -- requires ETL schema addition

**Defer (v2+):**
- Lap pace features if ETL re-extraction proves problematic -- haron features alone provide closing speed signal
- Fine-grained calibration beyond 5 popularity bands
- Per-horse lap decomposition (LapTime is leader-only, not per-horse)

### Architecture Approach

The v1.8 changes are all additive extensions to existing pipeline stages. No new components are created except a small `lap_features.py` module. The architecture follows an "extend-not-replace" principle: new features are appended to existing column lists (INTERACTION_COLS, FEATURE_COLS, BASE_COLS), new calibration is added as a parallel multiplicative layer alongside existing OddsBand scaling, and regime propagation injects a DataFrame column rather than changing method signatures.

**Major components (modified):**
1. **ETL type rules** (`src/db/etl.py`) -- Add float conversion for harontimel4 and laptime1~25
2. **Lap feature module** (`src/features/lap_features.py` NEW) -- PIT-safe harontimel4 and LapTime feature computation using expanding_stats + searchsorted pattern
3. **Interaction features** (`src/features/interaction_features.py`) -- Add 2-3 domain-knowledge interaction terms
4. **EV correction model** (`src/models/ev_correction_model.py`) -- Add popularity band scaling layer + regime_state feature
5. **Training pipeline** (`src/pipelines/training_pipeline.py`) -- Wire popularity band computation and regime label assignment
6. **Backtest engine** (`src/backtest/engine.py` + `race_predictor.py`) -- Inject regime_state into prediction DataFrame
### Critical Pitfalls

1. **POST_RACE leakage in haron/lap features** -- Using current race's HaronTimeL4 or LapTime as features inflates backtest ROI but fails in production. Prevent by adding all new POST_RACE columns to POST_RACE_COLS, using expanding_stats + searchsorted (side='left') for all history aggregation, and extending the 3-layer CI leakage tests.

2. **Look-ahead bias in popularity band calibration** -- Computing band ratios from full OOF data (including future folds) creates overconfident EV estimates. Prevent by computing ratios inside the OOF loop or using training-set-only data, following the v1.4 OddsBandFilter discipline.

3. **HaronTime sentinel values (000/999)** -- EveryDB2 stores measurement failures as 0.0 or 999.0 after float conversion, corrupting feature averages. Prevent by replacing values outside physiological range (haron < 30 or > 50) with NaN during ETL.

4. **Feature cache invalidation after ETL schema changes** -- The code-hash-based cache does not detect Parquet schema changes (new columns). Prevent by explicitly deleting `data/features/cache/` after ETL re-extraction.

5. **Inference path missing new features** -- `FeatureEngine.build_features()` (single-race inference) is a separate code path from `build_all()` (batch training). New lap features must be wired into both paths to prevent training-inference feature mismatch.

## Implications for Roadmap

Based on research, suggested phase structure follows the natural dependency chain:

### Phase 1: ETL Data Foundation
**Rationale:** All downstream features depend on HaronTimeL4 and LapTime data being available as float64 in Parquet files. This is the lowest-risk change -- declarative type rule additions, no logic change.
**Delivers:** harontimel4 in entries.parquet, laptime1~25 in races.parquet, sentinel value handling
**Addresses:** Feature Groups A (haron time), A-extension (lap pace)
**Avoids:** Pitfall 3 (sentinel values), Pitfall 12 (POST_RACE_COLS whitelist gap)
**Effort:** Small

### Phase 2: Turf Relative Features + Conditional Interactions
**Rationale:** These are the safest features -- zero new data required, extending existing well-tested patterns. They can be built in parallel since they touch different modules. Feature C adds 7 race_rank columns via groupby.rank(pct=True). Feature D adds 2-3 interaction terms via multiplication with NaN guards.
**Delivers:** form_trend_race_rank, blood_total_wr_race_rank, grade_x_form_score, distance_x_closing_index, and 5 more race_rank features
**Addresses:** Feature Group C (turf relative), Feature Group D (conditional interactions)
**Uses:** pandas groupby + rank (existing pattern), interaction_features.py (existing module)
**Avoids:** Pitfall 5 (overfitting -- limited to high-conviction interactions), Pitfall 11 (NaN propagation -- using existing .where() pattern)
**Effort:** Medium
### Phase 3: Haron/Lap Feature Computation
**Rationale:** Depends on Phase 1 ETL data being available. Extends existing `_compute_haron_stats` with HaronTimeL4 support and creates new lap feature module. Requires careful PIT validation since these features derive from POST_RACE columns.
**Delivers:** harontimel4_avg, harontimel4_zscore, haron_l3l4_ratio, lap_closing_speed_avg, lap_pace_differential (~6-9 new features)
**Addresses:** Feature Group A (haron time), Feature Group A-extension (lap pace)
**Implements:** New `src/features/lap_features.py` module, wired into both training and inference paths
**Avoids:** Pitfall 1 (POST_RACE leakage -- expanding_stats + searchsorted pattern), Pitfall 4 (data quality -- sentinel handling from Phase 1)
**Effort:** Medium

### Phase 4: EV Calibration Layers
**Rationale:** Depends on all new features being available for model training. Modifies the EV correction model and training pipeline -- affects all betting decisions so requires A/B backtest comparison. Popularity band calibration is the riskiest change (novel, needs OOF validation). Regime-surface propagation is lower risk (additive feature column).
**Delivers:** Popularity band EV scaling factors, regime_state in EV correction model
**Addresses:** Feature Group B (popularity band calibration), Feature Group E (regime x surface EV correction)
**Uses:** LightGBM native categorical handling, parallel multiplicative calibration layer pattern
**Avoids:** Pitfall 2 (look-ahead bias -- OOF-only computation), Pitfall 3 (circular dependency -- regime is market-derived only)
**Effort:** Medium

### Phase 5: Integrated Validation
**Rationale:** Final validation that all features work together. Extends the existing 3-layer PIT test framework with all new feature column names. Full backtest with walk-forward validation to confirm ROI improvement without overfitting.
**Delivers:** Extended CI tests, full backtest results, walk-forward validation, turf b_difference measurement
**Avoids:** Pitfall 1 (leakage -- comprehensive CI test extension), Pitfall 7 (cache invalidation -- explicit cache bust), Pitfall 8 (inference parity -- feature parity verification)
**Effort:** Medium (mainly compute time)

### Phase Ordering Rationale

- **Data before computation:** ETL (Phase 1) must complete before any feature can use the new columns. This is a hard dependency.
- **Safe features before risky features:** Phase 2 (race_rank + interactions) uses only existing data and patterns, providing early validation that the pipeline integration works before tackling the PIT-sensitive haron/lap features.
- **Features before models:** Phases 2-3 (feature computation) must complete before Phase 4 (EV calibration layers) because model training needs all features available.
- **Validation last:** Phase 5 runs the full test suite against the integrated system.
- **Popularity band calibration is last among model changes:** It is the riskiest feature (novel, OOF-sensitive) and should be validated in isolation against a baseline with features C+D+A+E already working.
### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1 (ETL):** LapTime column names in EveryDB2 are not yet verified against the actual database schema. Need to query `information_schema.columns` to confirm exact names and count (Pitfall 9).
- **Phase 4 (EV Calibration):** Popularity band calibration is a novel approach. The optimal band boundaries, regularization strength, and interaction with existing OddsBand scaling need empirical validation. Consider `--research-phase 4` for OOF residual analysis.

Phases with standard patterns (skip research-phase):
- **Phase 2 (Features C+D):** Well-documented patterns (add_race_transforms, interaction_features). Existing code provides clear templates.
- **Phase 3 (Haron/Lap):** Follows existing `_compute_haron_stats` and PaceAptitudeFeatures patterns exactly. PIT-safe by design if searchsorted pattern is replicated.
- **Phase 5 (Validation):** Extends existing 3-layer CI test framework. Standard validation approach.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new dependencies. All features verified implementable with existing libraries. Codebase analysis confirms every API surface exists. |
| Features | HIGH | 4 of 5 feature groups extend existing proven patterns. Only popularity band calibration (Group B) is novel and rated MEDIUM confidence. |
| Architecture | HIGH | All integration points verified against source code. File names, line numbers, and method signatures confirmed. 6 detailed data flow diagrams. |
| Pitfalls | HIGH | All pitfalls verified against source code. POST_RACE leakage prevention has 3-layer defense already in place. Sentinel value handling is a known pattern from harontimel3. |

**Overall confidence:** HIGH

### Gaps to Address

- **LapTime schema discovery:** The exact column names and count for LapTime in EveryDB2 are inferred from docs (`docs/everydb2/03-RACE.md` fields 68-92) but not verified against the live database. Phase 1 must start with a schema query before writing ETL type rules.
- **Popularity band optimal boundaries:** The research proposes 5 bands (1-3, 4-6, 7-9, 10-12, 13+) but the optimal boundaries depend on the actual OOF residual distribution. Phase 4 should include exploratory analysis of residual patterns by popularity rank before finalizing boundaries.
- **HaronTimeL3/L4 mutual exclusivity handling:** EveryDB2 docs indicate L3 and L4 are rarely both available for the same race. The coalescing strategy (use L3 first, fall back to L4) needs validation against actual data distribution after Phase 1 ETL.
- **Inference path for lap features:** `RacePredictor.predict()` currently does not have access to `races_df` for LapTime columns. The integration may require extending the predictor's data access or pre-computing race-level lap features.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/db/etl.py` (ETL type rules, SELECT * pattern)
- Codebase analysis: `src/features/horse_history_features.py` (expanding_stats + searchsorted PIT pattern)
- Codebase analysis: `src/features/interaction_features.py` (NaN-safe interaction computation)
- Codebase analysis: `src/models/ev_correction_model.py` (FEATURE_COLS, correct_ev flow)
- Codebase analysis: `src/backtest/race_predictor.py` (inference pipeline sequence)
- Codebase analysis: `src/backtest/engine.py` (per-race regime detection loop)
- Codebase analysis: `src/domain/types.py` (POST_RACE_COLS definition)
- EveryDB2 schema docs: `docs/everydb2/03-RACE.md` (LapTime1~25, HaronTimeS3/S4/L3/L4)
- EveryDB2 schema docs: `docs/everydb2/04-UMA_RACE.md` (HaronTimeL4, HaronTimeL3)

### Secondary (MEDIUM confidence)
- PROJECT.md v1.8 milestone requirements (5 feature group definitions)
- Existing GPD diagnostics: `src/models/gpd_diagnostics.py` (179 features tracked)
- Milestone history: v1.6 POST_RACE leakage prevention, v1.4 OddsBandFilter look-ahead fix

### Tertiary (LOW confidence)
- LapTime column names in EveryDB2 -- inferred from docs, not verified against live database schema
- Popularity band calibration effectiveness -- theoretical analysis, no empirical validation yet

---
*Research completed: 2026-05-19*
*Ready for roadmap: yes*
