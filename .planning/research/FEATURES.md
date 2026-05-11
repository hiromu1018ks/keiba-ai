# Feature Landscape: v1.6 Feature Engineering Overhaul

**Domain:** Horse racing ML feature engineering -- audit, new feature extraction from unused data, and interaction/transform engineering
**Researched:** 2026-05-10
**Confidence:** HIGH (full codebase audit of 22 feature modules, 9 model files, 103 ETL tables, and 2 model FEATURE_COLS lists totaling 65+37 features)

## Context

This document covers features for the v1.6 milestone ONLY. The system currently generates 100+ features across 22 feature modules, consumed by two model stages:
- **AbilityModel** (Stage1): 65 features -- horse history, bloodline, sire, pace, course, form, class trajectory, environment adaptability
- **WinTwoStageModel** (Stage2): 37 features (RETURN_FEATURE_COLS) + 40 HIT_FEATURE_COLS -- market dynamics, odds, race context, plus select horse features for the hit model

Current ROI: 84.4%. Target: 100%+. The v1.6 approach: CQR/calibration tuning has reached diminishing returns; the next lever is feature quality.

**Three workstreams:**
1. Audit existing 100+ features and prune noise
2. Extract new features from unused EveryDB2 tables
3. Engineer interactions and transformations (relative comparisons, conditional interactions, target encoding)

---

## Table Stakes

Features/actions that any serious horse racing ML system needs. Missing these means the model is operating below potential.

| # | Feature | Why Expected | Complexity | Data Source | Notes |
|---|---------|--------------|------------|-------------|-------|
| TS-01 | Feature audit: gain/split/permutation importance + noise identification | Without knowing which features are noise, adding more features risks diluting signal. 100+ features with no pruning means LightGBM wastes splits on noise columns, especially with ~4.5K training races per surface. | Medium | Trained models (gain importance), OOF predictions (permutation importance) | Script-only task. No module changes. Output: prune_candidates.json with features to remove from FEATURE_COLS. |
| TS-02 | Jockey context features (wire existing module) | `jockey_context_features.py` already computes 4 features (jockey_wr_overall, jockey_wr_distance, jockey_wr_venue, jockey_prize_log) from kisyu_seiseki.parquet. HIT_FEATURE_COLS already references `jockey_wr_overall`. The module exists but is NOT wired into `_train_submodel()`. | Low | kisyu_seiseki.parquet (SetYear < race_year PIT-safe) | Just wiring: ~15 lines in training_pipeline.py. Module is complete and tested. |
| TS-03 | Trainer context features (wire existing module) | `trainer_context_features.py` already computes 4 features (trainer_wr_overall, trainer_wr_distance, trainer_wr_venue, trainer_prize_log). HIT_FEATURE_COLS already references `trainer_wr_overall`. Module exists but NOT wired. | Low | chokyo_seiseki.parquet (SetYear < race_year PIT-safe) | Same pattern as TS-02. ~15 lines wiring. |
| TS-04 | Jockey-trainer combo features (wire existing module) | `jockey_trainer_combo.py` computes 4 features (jt_combo_wr, jt_combo_place_rate, jt_combo_starts, jt_combo_prize_log) from history_entries. HIT_FEATURE_COLS already references `jt_combo_place_rate`. Module exists but NOT wired. | Low | history_entries via ParquetStore, PIT-safe via searchsorted | ~15 lines wiring. The combo captures synergy effects that individual jockey/trainer stats miss. |
| TS-05 | Horse-vs-horse relative features (within-race comparison) | Current features are absolute per-horse values. A horse with norm_finish_logit_avg=0.5 looks good, but if the race average is 0.7, that horse is actually below-average. Relative features capture competitive positioning within the field. This is standard in Benter-style models. | Medium | Existing features from HorseHistoryFeatures, computed via groupby("race_id") transforms | New module: `features/relative_features.py`. ~80 lines. 5-10 new features: norm_finish_logit_vs_mean, harontimel5_vs_mean, harontimel5_vs_max, etc. |
| TS-06 | Prune noise features from FEATURE_COLS | After TS-01 identifies zero/negative importance features, remove them from AbilityModel.FEATURE_COLS (65 features) and WinTwoStageModel FEATURE_COLS/HIT/RETURN sublists. | Low | prune_candidates.json from TS-01 | Pure list editing. LightGBM silently ignores missing columns, so feature modules can keep computing them for backward compatibility. |

### Feature Dependencies

```
TS-01 (audit) → TS-06 (prune) [TS-06 needs audit results]
TS-02 (jockey wiring) — independent, can run anytime
TS-03 (trainer wiring) — independent
TS-04 (combo wiring) — independent
TS-05 (relative features) — independent but benefits from pruned feature set

Recommended order: TS-01 → TS-06 → TS-02/03/04 (parallel) → TS-05
```

---

## Differentiators

Features that would set this system apart from typical horse racing ML approaches. Not expected by default, but provide measurable edge.

| # | Feature | Value Proposition | Complexity | Data Source | Notes |
|---|---------|-------------------|------------|-------------|-------|
| D-01 | Dam/broodmare pedigree features | Current bloodline features (blood_surface_wr etc.) use career_stats from horse_career_stats.parquet, which aggregates offspring performance. But dam-side pedigree (n_hansyoku has 19 columns including dam info, n_sanku has 26 columns for offspring) is completely unused. Dam influence on distance aptitude and surface preference is a known edge in Japanese racing. | Medium | n_hansyoku (breeding master), n_sanku (offspring master), n_bameiorigin (extended pedigree) | New module needed. Static data (PIT-safe). Must cross-reference kettonum to find dam, then aggregate dam's offspring performance for same surface/distance. |
| D-02 | Extended sire line features (BMS = Broodmare Sire) | `sire_features.py` already computes bms_wr, but only win rate. The dam's sire (BMS) has strong influence on stamina and distance aptitude. Adding BMS distance_wr, BMS surface_wr would capture this. | Low | sire_career_stats.parquet (already loaded by SireFeatures) | Extend existing sire_features.py with 2-3 more columns using the BMS ketto number already resolved. |
| D-03 | Age-at-race and sex features from horse master | n_uma (horses.parquet, 227 columns) has birth year (seinen) and sex (sexcd) that are currently not used as features. Age is critical: 3-year-olds have different profiles than 5+ year-olds. Sex captures gelding vs mare vs colt differences. Age progression (improvement/decline curve) is a known predictive signal. | Low | n_uma (horses.parquet), static data | Compute age = race_year - seinen. Encode sex as numeric. Add to AbilityModel.FEATURE_COLS. ~30 lines in a new function or extension of _map_basic_features. |
| D-04 | Vote concentration features from n_hyosu_tanpuku | The hyosu (vote count) data captures market attention distribution. n_hyosu_tanpuku has per-horse vote counts, allowing computation of: vote_share, vote_concentration (HHI of vote shares), vote_gap_fav12 (vote share gap between 1st and 2nd favorite). This is independent information from odds -- vote counts reflect bettor behavior at a finer granularity. | Low | n_hyosu_tanpuku (per-horse vote count), n_hyosu (race total) | New module: `features/vote_features.py`. Pre-race data, PIT-safe. ~80 lines. |
| D-05 | Conditional interaction features | Current interactions (kyakusitu_x_distance, kyakusitu_x_surface) are simple cross-products. More powerful: surface x form_trend (is the horse improving on this specific surface?), class_level x distance_change (is the horse changing class at a distance where class matters?), weight_change_zone x rest_category (weight pattern after rest). LightGBM can learn these, but explicit features give it a head start. | Medium | Existing features from multiple modules | Extend interaction_features.py. ~50 lines. Add 4-6 conditional interactions. |
| D-06 | Target encoding for high-cardinality categoricals | blood_keito_cd is already a feature but as raw category. Target-encoded blood_keito_cd (expanding mean win rate per keito system) would capture the actual performance of each bloodline. Similarly, jockey code and trainer code target-encoded would be powerful. Must use expanding().shift(1) for PIT safety, following info_asymmetry_features.py pattern. | Medium | Existing entries + race results | Extend interaction_features.py or new module. PIT safety is critical: expanding().shift(1) with leave-one-out. ~100 lines. |
| D-07 | Career trajectory features from horse_career_stats | horse_career_stats.py already precomputes PIT-safe cumulative stats (cum_starts, cum_wins, cum_prize, surface-specific cumulative stats). But these are currently only consumed by bloodline_features.py for blood_* features. Direct career features (career_win_rate_trend, career_starts_at_distance, career_surface_transition) from these pre-computed stats are unused. | Low | horse_career_stats.parquet (already pre-computed, PIT-safe) | Extend horse_history_features.py to consume career_stats columns directly. Add 3-5 features. ~40 lines. |
| D-08 | Mining index features from n_mining | EveryDB2's n_mining table has 82 columns of pre-computed analytics per horse per race (JRA's own ML-derived indices). These include composite ability scores, form indices, and class assessments computed by JRA-VAN. If available pre-race (needs verification), they provide a strong prior. | High | n_mining (82 cols), n_taisengata_mining (46 cols, pairwise) | REQUIRES PIT AUDIT: must verify n_mining data is available pre-race, not post-race. If pre-race: extremely valuable. If post-race: must exclude entirely. New reader + module. ~200 lines. |

### Differentiator Dependencies

```
D-01 (dam pedigree) ← n_hansyoku, n_sanku readers (new readers in readers.py)
D-02 (BMS extension) ← sire_features.py extension
D-03 (age/sex) ← n_uma reader (already exists: load_horses)
D-04 (vote concentration) ← n_hyosu_tanpuku reader (may need new reader)
D-05 (conditional interactions) ← existing features (form_trend, class_level, etc.)
D-06 (target encoding) ← race results + existing categoricals
D-07 (career trajectory) ← horse_career_stats.parquet (already pre-computed)
D-08 (mining indices) ← n_mining PIT audit (blocking dependency)
```

---

## Anti-Features

Features to explicitly NOT build. These are tempting but counterproductive for this specific system.

| # | Anti-Feature | Why Avoid | What to Do Instead |
|---|-------------|-----------|-------------------|
| AF-01 | LSTM/Transformer time-series models for horse history | The horse has typically 5-15 past races. Deep sequence models overfit severely on this sample size. Additionally, PROJECT.md explicitly scopes this out. | Use EMA-weighted features (halflife=3) and expanding-window stats, which are the correct approach for short sequences with limited data. |
| AF-02 | Post-race data as features (kakuteijyuni, confirmed odds, time, honsyokin) | The ultimate look-ahead bias. Using results to predict results produces 90%+ backtest ROI that collapses to zero in production. The codebase has POST_RACE_COLS in domain/types.py to guard against this. | Always use pre-race features only. The leakage_validators.py framework must be run after every new feature addition. |
| AF-03 | External data sources (weather APIs, social media sentiment, international race results) | PROJECT.md explicitly scopes out new data sources. The EveryDB2 data (2015-2025) is comprehensive and sufficient. External sources add latency, maintenance burden, and reproducibility problems. | Focus on extracting maximum value from existing EveryDB2 tables that are already loaded to Parquet but unused for features. |
| AF-04 | Adding 50+ features without pruning existing noise | More features is not better when training data is limited (~4.5K races per surface). Each noisy feature competes for tree splits, diluting the signal from genuinely predictive features. This is the primary risk of feature engineering without audit. | Audit first (TS-01), prune noise (TS-06), then add new features incrementally with backtest validation at each step. |
| AF-05 | Complex feature interactions beyond 2nd order | Polynomial features of degree 3+ and interaction terms involving 4+ base features are almost always noise in horse racing ML with limited training data. The combinatorial explosion makes interpretation impossible and overfitting certain. | Use 2nd-order interactions only (surface x form, class x distance). Let LightGBM discover higher-order patterns through tree structure. |
| AF-06 | Horse-v-horse pairwise features for all combinations | Computing features for every (horse_i, horse_j) pair in a race produces O(n^2) features per race. With 18-horse fields, this is 153 pairs. Memory and computation explode. The model already implicitly captures relative positioning through race-level features. | Use within-race aggregate comparisons (horse vs mean, horse vs max) as in D-05. These are O(n) and capture the same competitive positioning information. |
| AF-07 | Re-architecting the two-stage model structure | The P(hit) x E(odds|hit) decomposition is theoretically sound. Changing the model structure is out of scope and risks breaking the entire downstream pipeline (EV correction, betting filters, CQR). | Focus on feature quality, not model architecture. The existing 3-model stacking with Ridge meta-learner is sufficient. |

---

## Feature Inventory: Current State

### By Module (22 feature modules)

| Module | Category | Features | Used By | Status |
|--------|----------|----------|---------|--------|
| `_map_basic_features` (feature_engine.py) | A: Basic | 12 (distance_bin, grade_code, class_level, field_size, popularity_rank, draw_ratio, frame_number, blinker_on, weight_change_zone, weight_change_ratio, etc.) | Stage1, Stage2 | Active |
| `compute_intra_race_features` | B: Intra-race | 2 (weight_diff_from_mean, odds_rank) | Stage1 | Active |
| `compute_odds_dynamics` | C: Odds | 7 (odds_drop_rate_60/30_10, velocity, volatility, acceleration, direction_consistency, popularity_change) | Stage2 | Active |
| `compute_market_bias` + `compute_flb_slope` | D: Market | 5 (p_market_win_adj, market_entropy, overround, odds_skewness, implied_prob_hhi) | Stage1, Stage2 | Active |
| `compute_difficulty_score` | E: Difficulty | 1 (difficulty_score) | Stage1 | Active |
| `BloodlineFeatures` | B: Bloodline | 6 (blood_surface/distance/condition/total_wr, blood_prize_log, blood_keito_cd) | Stage1 | Active |
| `HorseHistoryFeatures` | A: History | ~45 (norm_finish_logit_avg, harontimel5_*, form_*, class_*, jockey_*, weight_*, etc.) | Stage1 | Active, largest module (~1350 lines) |
| `PaceAptitudeFeatures` | C: Pace | 6 (pace_aptitude, front/closing_pace_wr, pace_corner_stability, pace_closing_power, pace_position_consistency) | Stage1, interaction_features | Active |
| `CourseFeatures` | D: Course | 2 (course_wr, course_distance_wr) | Stage1 | Active |
| `SireFeatures` | B: Sire | 5 (sire_wr, sire_surface_wr, sire_distance_wr, sire_prize_avg, bms_wr) | Stage1 | Active |
| `compute_interaction_features` | E: Interaction | 12 (kyakusitu_x_*, weight_x_distance, race_mean/std_fuku_odds, odds_gap_fav12, odds_popularity_gap, surface_track_interaction, pace_pressure/closer_share/scenario_fit, actual_pace_fit) | Stage1, Stage2 | Active |
| `compute_odds_deviation` | F: Deviation | 2 (deviation_rank, deviation_zscore) | Stage2 | Active |
| `compute_form_features` (form_cycle) | B: Form | 3 (form_trend, form_consistency, form_peak_flag) | Stage1 (via HorseHistoryFeatures) | Active |
| `compute_class_trajectory` + form improvement + env adaptability (high_odds_features) | A: High-odds | 18 (class_promotions/demotions/net_change/max_level/std, v_recovery_flag/duration, time/position_improvement_rate, 9 env adaptability) | Stage1 | Active |
| `JockeyContextFeatures` | D: Jockey | 4 (jockey_wr_overall/distance/venue, jockey_prize_log) | Referenced in HIT_FEATURE_COLS | EXISTS but NOT WIRED |
| `TrainerContextFeatures` | D: Trainer | 4 (trainer_wr_overall/distance/venue, trainer_prize_log) | Referenced in HIT_FEATURE_COLS | EXISTS but NOT WIRED |
| `JockeyTrainerComboFeatures` | D: Combo | 4 (jt_combo_wr/place_rate/starts/prize_log) | Referenced in HIT_FEATURE_COLS | EXISTS but NOT WIRED |
| `compute_hist_features` (info_asymmetry) | E: Info asym | 5 (hist_hit_rate_topk, hist_roi_topk, hist_positive_return_ratio, hist_win_rate_same_condition, hist_market_entropy_avg) | Race-level features | Active |
| `compute_roi_ema` (odds_dynamics) | C: Market EMA | 3 (favorite_implied_prob_ema, overround_ema, entropy_ema) | Used internally | Active |

### By Model Stage

**AbilityModel (Stage1): 65 features**
- Race conditions (7): surface, distance_bin, track_condition_code, grade_code, field_size, weight_diff_from_mean, difficulty_score
- Past performance (8): norm_finish_logit_avg, harontimel5_avg/zscore, harontime_late_trend, timediff_avg, jyuni1c/4c_avg, closing_index_avg
- Categorical (1): kyakusitukubun_cd
- Bloodline (6): blood_surface/distance/condition/total_wr, blood_prize_log, blood_keito_cd
- Interactions (3): kyakusitu_x_distance, kyakusitu_x_surface, weight_x_distance
- Race-relative ranks (5): norm_finish/harontimel5/timediff/jyuni1c/closing_index_race_rank
- Body (3): weight_absolute, weight_zscore, weight_change_zone
- Rest (2): days_since_last_race, rest_category
- Form cycle (3): form_trend, form_consistency, form_peak_flag
- Sire (5): sire_wr, sire_surface_wr, sire_distance_wr, sire_prize_avg, bms_wr
- Pace (3): pace_aptitude, front/closing_pace_wr
- Course (2): course_wr, course_distance_wr
- Additional (5): draw_ratio, class_move, blinker_change, pace_pressure, pace_scenario_fit
- Time-series (2): class_adj_formetric, haron_zscore_trend
- Pace sub (4): pace_corner_stability, pace_closing_power, pace_position_consistency, actual_pace_fit
- High-odds (18): class trajectory (7), form improvement (2), env adaptability (9)

**WinTwoStageModel (Stage2):**
- RETURN_FEATURE_COLS: 37 features (market/odds/race context + select horse features)
- HIT_FEATURE_COLS: 40 features (includes horse-level features like norm_finish_logit_avg, harontimel5_zscore + jockey/trainer/combo features)

---

## Unused EveryDB2 Data Sources

Tables already loaded to Parquet but NOT used for feature generation:

| Table | Parquet Key | Columns | Potential Features | PIT Safety |
|-------|-------------|---------|-------------------|------------|
| `n_mining` | mining | 82 | JRA pre-computed ability indices, form scores | NEEDS AUDIT -- verify pre-race availability |
| `n_taisengata_mining` | taisengata_mining | 46 | Pairwise horse comparison indices | NEEDS AUDIT |
| `n_hansyoku` | hansyoku | 19 | Dam info, breeding farm, birth year | SAFE -- static |
| `n_sanku` | sanku | 26 | Offspring performance per dam | SAFE -- static |
| `n_uma` (additional columns) | horses | 227 total | sex (sexcd), birth year (seinen), 14 ketto3info columns (5-14 for dam-side pedigree) | SAFE -- static |
| `n_bameiorigin` | bameiorigin | ~30 | Extended 5-generation pedigree | SAFE -- static |
| `n_hyosu` / `n_hyosu_tanpuku` | hyosu / hyosu_tanpuku | ~10 / ~8 | Vote counts per horse, vote concentration (HHI) | SAFE -- pre-race snapshot |
| `n_record` | record | 48 | Course records, track-specific best times | SAFE -- historical |
| `n_course` | course | 8 | Course characteristics (circumference, straight length) | SAFE -- static |
| `n_wood_chip` | wood_chip | 29 | Training data (wood chip track performance) | SAFE -- static (but may be post-update) |
| `n_hanro` | hanro | 14 | Slope training data | SAFE -- static |
| `n_banusi` | banusi | 27 | Owner data (owner win rate, owner horse count) | SAFE -- static |
| `n_sale` | sale | 14 | Horse auction prices (market value proxy) | SAFE -- static |
| `n_toku_race` / `n_toku` | toku_race / toku | ~10 / ~10 | Special race metadata, entry conditions | SAFE -- pre-race |
| `n_schedule` | schedule | ~8 | Race schedule, weather info | SAFE -- pre-race |

---

## Feature Dependencies

```
# Phase 1: Audit (no new features)
TS-01 (audit script) → TS-06 (prune FEATURE_COLS)

# Phase 2: Wire existing modules (no new development)
TS-02 (jockey) — standalone
TS-03 (trainer) — standalone
TS-04 (jockey-trainer combo) — standalone
# All three can be wired in parallel

# Phase 3: New features from unused data
D-03 (age/sex) ← load_horses (existing reader)
D-04 (vote concentration) ← hyosu_tanpuku reader
D-07 (career trajectory) ← horse_career_stats.parquet (existing)
D-02 (BMS extension) ← sire_features.py extension

# Phase 4: Pedigree and advanced features
D-01 (dam pedigree) ← n_hansyoku, n_sanku readers (new)
D-08 (mining indices) ← n_mining PIT audit (blocking)

# Phase 5: Interactions and transformations
D-05 (conditional interactions) ← all base features complete
D-06 (target encoding) ← existing categoricals
D-05/TS-05 (relative features) ← base features + groupby transforms
```

---

## MVP Recommendation

**Priority 1 (Quick wins -- wire existing code):**
1. **TS-01 + TS-06**: Feature audit and pruning -- establishes a clean baseline. Without this, adding features is guesswork.
2. **TS-02 + TS-03 + TS-04**: Wire jockey, trainer, and combo features. 12 new features from existing modules with ~45 lines of pipeline code. These are already referenced in HIT_FEATURE_COLS.

**Priority 2 (Low-hanging fruit -- simple new features):**
3. **D-03**: Age-at-race and sex encoding. Static data, trivial computation, strong predictive signal.
4. **TS-05**: Horse-vs-horse relative features. Standard Benter approach, captures competitive positioning.
5. **D-02**: BMS distance/surface features. Extend existing sire_features.py with 2-3 more columns.

**Priority 3 (Medium effort -- new data sources):**
6. **D-04**: Vote concentration from hyosu_tanpuku. Independent information from odds.
7. **D-07**: Career trajectory from horse_career_stats. Data already pre-computed.
8. **D-05**: Conditional interactions (surface x form, class x distance).

**Priority 4 (Higher effort -- new modules):**
9. **D-01**: Dam/broodmare pedigree features. Requires new readers and cross-referencing.
10. **D-06**: Target encoding for high-cardinality categoricals. Requires careful PIT implementation.

**Defer (needs PIT audit or high complexity):**
- **D-08**: Mining indices -- requires PIT audit to verify pre-race availability. If confirmed pre-race, elevate to Priority 2.
- **n_wood_chip / n_hanro**: Training data features -- potentially valuable but complex to integrate and may not have sufficient coverage.
- **n_banusi / n_sale**: Owner and auction price features -- niche signal, low priority.

---

## Expected Impact Assessment

| Feature Group | Expected Features Added | Signal Strength | Overfitting Risk | Priority |
|---------------|------------------------|-----------------|------------------|----------|
| Feature audit/pruning | -10 to -20 (removed) | Removes noise | Reduces overfitting | TS-01/06 |
| Jockey/trainer/combo wiring | +12 | HIGH (human factors) | LOW (annual stats) | TS-02/03/04 |
| Age/sex | +2-3 | HIGH (fundamental) | VERY LOW | D-03 |
| Relative features | +5-10 | HIGH (competitive context) | LOW (race-level transforms) | TS-05 |
| BMS extension | +2-3 | MEDIUM (dam sire influence) | LOW | D-02 |
| Vote concentration | +3-4 | MEDIUM (market behavior) | LOW (pre-race data) | D-04 |
| Career trajectory | +3-5 | MEDIUM (career progression) | LOW (PIT-safe cumulative) | D-07 |
| Conditional interactions | +4-6 | MEDIUM (nonlinear combos) | MEDIUM (interaction explosion) | D-05 |
| Target encoding | +2-3 | MEDIUM (categorical signal) | MEDIUM (needs regularization) | D-06 |
| Dam pedigree | +5-8 | MEDIUM (breeding edge) | LOW (static data) | D-01 |
| Mining indices | +10-20 (TBD) | HIGH if pre-race | LOW if pre-race | D-08 |

**ROI improvement estimate:**
- Phase 1 (audit/prune): +0 to +3pp (noise removal)
- Phase 2 (jockey/trainer/combo + age/sex + relative): +3 to +8pp (strong new signal)
- Phase 3 (vote, career, interactions): +2 to +5pp (incremental)
- Phase 4 (dam, mining if pre-race): +2 to +5pp (breeding edge)

**Total estimated improvement: 7 to 21pp, from 84.4% to 91-105% ROI.**

The 100% target is achievable if: (a) feature pruning removes significant noise, (b) jockey/trainer/combo wiring captures human-factor signal, and (c) at least one of the high-signal features (relative, mining, or dam pedigree) delivers expected impact.

---

## Sources

### Primary (HIGH confidence -- direct codebase analysis)
- `src/models/stage1_ability_model.py` -- 65-feature FEATURE_COLS list (lines 28-128)
- `src/models/two_stage_return_model.py` -- 37-feature RETURN_FEATURE_COLS + 40-feature HIT_FEATURE_COLS (lines 48-404)
- `src/features/horse_history_features.py` -- ~1350 lines, ~45 features, per-horse loop with PIT safety via searchsorted
- `src/features/bloodline_features.py` -- 6 bloodline features from career_stats
- `src/features/sire_features.py` -- 5 sire features including bms_wr
- `src/features/jockey_context_features.py` -- 4 jockey features, EXISTS but NOT WIRED
- `src/features/trainer_context_features.py` -- 4 trainer features, EXISTS but NOT WIRED
- `src/features/jockey_trainer_combo.py` -- 4 combo features, EXISTS but NOT WIRED
- `src/features/high_odds_features.py` -- 18 features (class trajectory, form improvement, env adaptability)
- `src/features/interaction_features.py` -- 12 interaction features
- `src/features/horse_career_stats.py` -- PIT-safe pre-computed cumulative career stats
- `src/features/info_asymmetry_features.py` -- expanding().shift(1) PIT-safe pattern
- `src/features/pace_aptitude_features.py` -- 6 pace features, vectorized
- `src/features/course_features.py` -- 2 course features with Beta smoothing
- `src/features/odds_dynamics_features.py` -- 7 odds dynamics features + 3 market EMA features
- `config/etl_tables.yaml` -- 103 EveryDB2 tables with parquet keys
- `.planning/PROJECT.md` -- v1.6 milestone context, constraints, out-of-scope

### Secondary (MEDIUM confidence -- domain knowledge)
- Benter (1994) -- Hong Kong horse racing model architecture: relative features within race are fundamental
- Lessmann et al. (2020) -- Feature engineering for prediction markets: human-factor features (jockey, trainer) consistently rank high
- EveryDB2/JRA-VAN DataLab documentation -- table structure and data availability
