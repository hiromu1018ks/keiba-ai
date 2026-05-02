# Feature Landscape: Win (単勝) Prediction Model

**Domain:** Horse racing win prediction (pari-mutuel betting)
**Researched:** 2026-05-02
**Confidence:** HIGH (code review + academic literature + community practice)

## Current Feature Inventory

The system already has 14 feature modules generating 100+ features across these categories:

| Module | File | Features Generated |
|--------|------|--------------------|
| Basic features | `feature_engine.py` | surface, distance_bin, track_condition, grade, field_size, draw_ratio, weight_change_zone/ratio, class_level, popularity_rank, blinker_on |
| Intra-race relative | `intra_race_features.py` | weight_diff_from_mean, odds_rank |
| Horse history | `horse_history_features.py` | norm_finish_logit_avg, harontimel5_avg/zscore, harontime_late_trend, timediff_avg, jyuni1c_avg, jyuni4c_avg, closing_index_avg, kyakusitukubun_cd, jockey_surprise, jockey_cond_wr, weight_absolute/zscore, days_since_last_race, rest_category, form_trend/consistency/peak_flag, class_move, blinker_change, is_nar_transfer, nar_recent_ratio, track_condition_delta |
| Bloodline | `bloodline_features.py` | blood_surface_wr, blood_distance_wr, blood_condition_wr, blood_total_wr, blood_prize_log, blood_keito_cd |
| Sire/BMS | `sire_features.py` | sire_wr, sire_surface_wr, sire_distance_wr, sire_prize_avg, bms_wr |
| Jockey context | `jockey_context_features.py` | jockey_wr_overall, jockey_wr_distance, jockey_wr_venue, jockey_prize_log |
| Trainer context | `trainer_context_features.py` | trainer_wr_overall, trainer_wr_distance, trainer_wr_venue, trainer_prize_log |
| JT combo | `jockey_trainer_combo.py` | jt_combo_wr, jt_combo_place_rate, jt_combo_starts, jt_combo_prize_log |
| Pace aptitude | `pace_aptitude_features.py` | pace_aptitude, front_pace_wr, closing_pace_wr |
| Course features | `course_features.py` | course_wr, course_distance_wr |
| Form cycle | `form_cycle_features.py` | form_trend, form_consistency, form_peak_flag |
| Interaction | `interaction_features.py` | kyakusitu_x_distance, kyakusitu_x_surface, weight_x_distance, race_mean_fuku_odds, race_std_fuku_odds, odds_gap_fav12, odds_popularity_gap, surface_track_interaction, pace_pressure, closer_share, pace_scenario_fit |
| Market bias | `market_bias_features.py` | p_market_win_adj, market_entropy, overround, odds_skewness, implied_prob_hhi |
| Odds dynamics | `odds_dynamics_features.py` | odds_drop_rate_60_10, odds_drop_rate_30_10, odds_velocity, odds_volatility, popularity_change_30_10 |
| Info asymmetry | `info_asymmetry_features.py` | hist_hit_rate_topk, hist_roi_topk, hist_positive_return_ratio, hist_win_rate_same_condition, hist_market_entropy_avg |
| Race difficulty | `race_difficulty_model.py` | difficulty_score |
| Career stats | `horse_career_stats.py` | Pre-computes PIT cumulative stats (not direct features) |

## Model Feature Usage Matrix

### Stage1: AbilityModel (Ranker, no odds)
Uses 55 features from: race conditions (7), past performance (9), bloodline (6), interaction (3), race-rank normalized (5), body condition (3), rest period (2), form cycle (3), sire/BMS (5), pace aptitude (3), course aptitude (2), additional (8).

### Stage2: WinTwoStageModel (Hit + Return)
Uses 25 features: p_ability_win (Stage1 output), market log errors (2), odds dynamics (5), market bias (3), race conditions (5), FLB slope (1), additional (7).

### Stage3: WinEVCorrectionModel
Uses 25 features: e_return prediction, p-e interaction, market features, jockey/trainer context (8), jt combo (4).

---

## Table Stakes

Features that any competitive win prediction model MUST have. Missing these means the model is fundamentally incomplete.

| Feature | Current Status | Why Expected | Complexity | Notes |
|---------|---------------|--------------|------------|-------|
| **Speed/ability rating** | EXISTS (norm_finish_logit_avg + harontimel5_zscore) | Core handicapping factor since Benter (1994). Horses with faster relative times win more. | Low | Already well-implemented with expanding z-score and hierarchical fallback |
| **Class level** | EXISTS (class_level_current, class_move) | Class is the 2nd pillar of Benter's model. Horses moving up/down in class show strong win-rate changes. | Low | class_move (current - previous) already captures transitions |
| **Recent form** | EXISTS (form_trend, form_consistency, form_peak_flag) | Form cycle is the 3rd Benter pillar. Horses in improving form outperform declining form. | Low | Well-implemented with linear regression slope on normalized finishes |
| **Jockey quality** | EXISTS (jockey_wr_overall/distance/venue, jockey_surprise, jockey_cond_wr) | Top jockeys consistently win 2-3x more than average. Beta-smoothed surface-specific win rates are standard. | Low | Very thorough: overall + distance + venue + surprise (vs market expectation) |
| **Odds/implied probability** | EXISTS (p_market_win_adj, popularity_rank) | Market odds encode collective intelligence. Any model ignoring odds is leaving signal on the table. | Low | Correctly uses pre-race odds, not confirmed odds |
| **Surface/distance suitability** | EXISTS (blood_surface_wr, blood_distance_wr, pace_aptitude, course_wr) | Fundamental domain knowledge: turf/dirt specialists, sprint/stay distance aptitude. | Low | Multi-layered: blood + sire + pace + course |
| **Track condition** | EXISTS (track_condition_code, track_condition_delta) | Track condition (good/yielding/soft) drastically changes outcomes. Mud specialists win on heavy tracks. | Low | track_condition_delta captures change from previous outing |
| **Weight carried** | EXISTS (weight_absolute, weight_zscore, weight_diff_from_mean, weight_change_zone) | Weight is the primary handicapping tool. Weight changes signal form/fitness. | Low | Both absolute and relative (vs mean, vs own history) covered |
| **Field size** | EXISTS (field_size) | Win probability mathematically scales with field size. Larger fields = lower win rates for all. | Low | Used as direct feature in all models |
| **Market efficiency measures** | EXISTS (market_entropy, overround, odds_skewness) | Overround (house take), entropy (race competitiveness), and skewness measure market structure. | Low | Critical for identifying mispriced horses |

## Differentiators

Features that provide competitive edge. These are where ROI improvement comes from.

### Already Implemented (validate effectiveness)

| Feature | Current Status | Value Proposition | Complexity | Notes |
|---------|---------------|-------------------|------------|-------|
| **Odds dynamics (late money)** | EXISTS (odds_drop_rate, odds_velocity, odds_volatility) | Late odds movements reveal insider information. Sharp drops = smart money. HIGHEST ROI lever per academic literature. | Medium | Already computed from time-series data; needs win-specific tuning |
| **Jockey surprise** | EXISTS | Measures jockey over/under-performance vs market expectation. Captures jockey value beyond raw win rate. | Medium | Beta-smoothed, statistically sound |
| **Pace scenario fit** | EXISTS (pace_pressure, pace_scenario_fit, closer_share) | Projects race pace from declared running styles. Front-runners win more in slow-paced races; closers in fast-paced. | Medium | Solid implementation using kyakusitukubun_cd |
| **Closing index** | EXISTS (closing_index_avg) | (4C position - finish position) normalized. Horses that gain ground late show sustained ability. | Low | Direct measure of late-race speed |
| **Bloodline (sire/BMS)** | EXISTS (sire_wr, bms_wr, blood_*) | Sire lines encode genetic predisposition to surface/distance. BMS (broodmare sire) captures dam-side influence. | Medium | PIT-safe career stats, Beta-smoothed |
| **JT combo** | EXISTS (jt_combo_wr, jt_combo_place_rate) | Specific jockey-trainer partnerships outperform individual stats. Captures training/riding synergy. | Medium | Small sample sizes smoothed with Beta(1,10) |
| **Race difficulty** | EXISTS (difficulty_score) | High-difficulty races have more unpredictable outcomes. Model should bet more selectively in tough races. | Low | Grade x field_size x entropy composite |
| **Draw position** | EXISTS (draw_ratio) | Starting position bias at certain courses/distances. Inner draw advantage in sprints at tight courses. | Low | Normalized by field_size |

### New Features to Build (HIGH IMPACT)

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Win-specific odds residual** | Model's p_ability vs market probability gap, computed at win-model level. This is the core "edge" signal: how much does the model disagree with the market? Currently only signed_log_error_win exists from MarketModel, but a direct p_ability_win - p_market_win residual would be more informative for the hit model. | Low | p_ability_win (Stage1), p_market_win_adj | Currently the system uses log_error from MarketModel. A direct probability gap at WinTwoStage level may carry more signal. |
| **Expected pace figure per horse** | Assign each horse a numerical pace figure from past sectional times (jyuni1c, jyuni4c, harontimel3). Then project the race's expected pace and each horse's positional advantage. This goes beyond the current pace_pressure (which uses only declared running style) by incorporating actual timing data. | High | jyuni1c, jyuni4c, harontimel3 from history | The raw data exists (corner positions, furlong times). What's missing is the synthesis into a per-horse pace figure that feeds into a race-level pace projection. |
| **Trainer recent form (30/60/90 day)** | Trainer win rate in the last 30/60/90 days (expanding window with decay). Trainers have hot and cold streaks that annual stats mask. Currently only annual stats from x_CHOKYO_SEISEKI are used. | Medium | entries_hist data (already loaded) | Requires expanding window computation per trainer, similar to jockey_surprise pattern |
| **Horse freshness composite** | Combine days_since_last_race + rest_category + number_of_starts_this_year + starts_last_90_days into a single "freshness" score. Horses with optimal freshness (not too fresh, not too raced) win more. | Low | days_since_last_race (exists), start counts from history | Simple composite from existing data |
| **Distance change delta** | Difference between today's distance and average distance of last 3 starts. Horses switching distance categories (sprint to mile, mile to long) show predictable performance changes. | Low | kyori (current), history distances | Just need to compute avg(past_3_distances) and subtract |
| **Surface change flag** | Binary: is horse switching from turf to dirt or vice versa? Surface switches are strong negative signals for win probability. | Low | surface (current), history surfaces | Simple: current_surface != last_start_surface |
| **Weight x class interaction** | Higher-class races carry more weight. The interaction of weight carried and class level captures the handicap dynamic. In high-class races, lightly-weighted horses have disproportionate advantage. | Low | weight_absolute, class_level_current | Simple multiplication or ratio feature |
| **Trainer course specialty** | Trainer win rate at this specific venue (jyocd). Some trainers are course specialists (e.g., strong at Nakayama, weak at Tokyo). Currently trainer_wr_venue exists but may not use jyocd correctly. | Medium | trainer stats, jyocd | Verify existing trainer_wr_venue uses correct venue code |
| **Recent jockey-horse pairing** | Has this jockey ridden this horse before? If yes, what's the win rate? Familiarity between jockey and horse improves coordination. | Medium | history entries (kisyucode + kettonum pairs) | New lookup: (kisyucode, kettonum) -> past performance |
| **Odds-to-ability ratio** | p_market / p_ability. High ratio = market undervalues horse (potential value bet). Low ratio = market overvalues. This is the single most important feature for ROI because it directly measures the betting edge. | Low | p_ability_win, p_market_win_adj | Simple division; critical for value detection |

### New Features to Build (MEDIUM IMPACT)

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| **Horse age (exact)** | 3-year-olds and 4-year-olds have different developmental curves. Include exact age as numeric feature. Currently class_level captures some of this but age is a direct signal. | Low | kettonum -> horses.parquet birthdate, race_date | Simple computation from existing data |
| **Grade race debut flag** | Is this horse racing in a graded stakes for the first time? First-time graded runners often underperform. | Low | gradecd (current), past grade history | Binary flag from history data |
| **Seasonal pattern** | Month or season feature. Some horses perform better in spring (turf firm) vs winter (dirt heavy). | Low | race_date -> month | Already partially captured by track_condition but seasonal pattern is independent |
| **Layoff return performance** | After 90+ day layoffs, what is this horse's historical win rate? Some horses return well, others don't. | Medium | days_since_last_race (exists), past layoff-return results from history | Requires identifying past layoff periods and their outcomes |
| **Class drop bounce** | Horses dropping in class after 2+ losses at higher class. "Class drop bounce" is a well-known pattern where dropped horses win at high rates. | Low | class_move (exists), recent_finishes | class_move < 0 AND recent finishes > 3rd |
| **Win-streak / losing-streak** | Number of consecutive wins or losses. Hot horses tend to keep winning; long losing streaks predict continued poor performance. | Low | history finishes | Simple counter from recent starts |
| **Margin of victory/defeat** | Average margin (timediff) in recent wins/losses. Dominant winners by large margins are stronger candidates. | Low | timediff (exists), kakuteijyuni | Filter timediff by win/loss and average separately |

## Anti-Features

Features to explicitly NOT build. These either introduce leakage, add noise, or waste development effort.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Confirmed/final odds as features** | Data leakage: final odds are only available after race concludes. Model would learn from future information. | Use pre-race odds snapshot (tanodds) as currently implemented. The system already guards this via confirmed_odds separation. |
| **Post-race sectional times** | Only available after race. Using 3f/4f sectional times from the current race is pure leakage. | Use ONLY historical sectional times (harontimel3 from past starts). The system already does this correctly. |
| **Current race finishing position** | kakuteijyuni is the TARGET, not a feature. Including it would make the model trivially perfect on training data. | The system already has POST_RACE_COLS exclusion in backtest engine. Maintain this. |
| **Future horse career stats** | Using career stats computed at ETL time (horses.parquet cumulative) includes future results. | Use horse_career_stats.parquet (PIT-safe) as currently implemented. Do NOT revert to horses.parquet cumulative values. |
| **Publicly available "expert picks"** | Not in EveryDB2 data source. Would require new data pipeline and introduces non-reproducible, subjective data. | Focus on quantitative features from race/entry/odds data. |
| **Weather-derived features beyond baba_cd** | Detailed weather features (temperature, humidity, wind speed) are not in EveryDB2 and would require external data source. | track_condition_code captures the ground condition effect. Surface + baba_cd is sufficient without adding external dependencies. |
| **Horse heart rate / biometric data** | Not available in JRA-VAN DataLab. Would require entirely new data source. | Stick with performance-based features that encode fitness indirectly (weight, form, recent results). |
| **Social media sentiment** | Noisy, hard to collect, and not in data source. The signal-to-noise ratio is poor for horse racing. | Odds dynamics already capture informed money flow, which subsumes most publicly available sentiment. |
| **Track bias per meeting** | Computing real-time track bias (inner vs outer advantage) per race meeting requires same-day results from earlier races. Not feasible in pre-race prediction. | Use course_wr and historical draw_bias features which capture long-term track configuration bias. |
| **Trainer/jockey intention features** | Cannot be measured from data. "The trainer is trying hard today" is not quantifiable. | Use objective stats: trainer win rate, recent form, equipment changes (blinker_change). These are proxies for intent. |

## Feature Dependencies

```
Stage1 (AbilityModel) -- no odds, pure handicapping
  |
  +-- Horse History Features (norm_finish_logit_avg, harontimel5_*, timediff_avg, ...)
  |     requires: entries_hist + races_hist (past performance data)
  |
  +-- Bloodline Features (blood_*)
  |     requires: horse_career_stats.parquet (PIT cumulative)
  |
  +-- Sire/BMS Features (sire_wr, bms_wr, ...)
  |     requires: sire_stats_df (pre-computed sire cumulative stats)
  |
  +-- Pace Aptitude (pace_aptitude, front_pace_wr, closing_pace_wr)
  |     requires: entries_hist with jyuni1c/jyuni4c
  |
  +-- Course Features (course_wr, course_distance_wr)
  |     requires: entries_hist with jyocd + distance_bin
  |
  +-- Form Cycle (form_trend, form_consistency, form_peak_flag)
  |     requires: kakuteijyuni + syussotosu from history (computed inside HorseHistoryFeatures)
  |
  +-- Interaction Features (kyakusitu_x_*, weight_x_distance, pace_pressure, ...)
        requires: kyakusitukubun_cd (from HorseHistoryFeatures), surface, distance_bin

Stage2 (WinTwoStageModel) -- uses Stage1 output + odds
  |
  +-- p_ability_win (from Stage1)
  |
  +-- Market log errors (from MarketModel)
  |     requires: p_market_win_adj (MarketModel output), p_ability_win
  |
  +-- Odds dynamics (odds_drop_rate, odds_velocity, odds_volatility)
  |     requires: odds time-series data (s_odds_tanpuku)
  |
  +-- Market bias (market_entropy, overround, odds_skewness)
        requires: tanodds snapshot

Stage3 (WinEVCorrectionModel) -- refines EV estimate
  |
  +-- Stage2 output (e_return_win_pred, p_win_pred)
  |
  +-- Jockey context (jockey_wr_*)
  |     requires: x_KISYU_SEISEKI (annual jockey stats)
  |
  +-- Trainer context (trainer_wr_*)
  |     requires: x_CHOKYO_SEISEKI (annual trainer stats)
  |
  +-- JT combo (jt_combo_wr, ...)
        requires: entries_hist (jockey+trainer pair history)
```

## Proposed New Feature Dependencies

```
Win-specific odds residual
  requires: p_ability_win (Stage1 output) + p_market_win_adj (already computed)

Odds-to-ability ratio
  requires: p_ability_win + p_market_win_adj

Expected pace figure per horse
  requires: jyuni1c, jyuni4c, harontimel3 from history (all exist)
  blocks: nothing, but enhances pace_scenario_fit

Trainer recent form (30/60/90 day)
  requires: entries_hist with chokyosicode + kakuteijyuni (already loaded)
  independent: yes

Horse freshness composite
  requires: days_since_last_race (exists) + starts_this_year (from history)
  independent: yes

Distance change delta
  requires: kyori (current) + history distances (already loaded)
  independent: yes

Surface change flag
  requires: surface (current) + history surfaces (already loaded)
  independent: yes

Class drop bounce
  requires: class_move (exists) + recent finishes (from history)
  independent: yes

Win/losing streak
  requires: history finishes (already loaded)
  independent: yes

Jockey-horse pairing
  requires: history entries filtered by (kisyucode, kettonum) pairs
  independent: yes
```

## MVP Recommendation for Win Model Improvement

### Priority 1: Quick wins (Low complexity, HIGH expected impact)
These use existing data and simple computations.

1. **Odds-to-ability ratio** (p_market / p_ability) -- the single most important ROI signal
2. **Distance change delta** -- horses switching distance categories
3. **Surface change flag** -- turf/dirt switch detection
4. **Class drop bounce** -- class_move + recent poor finishes
5. **Win/losing streak** -- consecutive result counter

### Priority 2: Medium effort, HIGH expected impact
These require new computation but use existing data.

6. **Trainer recent form (30/60/90 day)** -- captures trainer hot/cold streaks
7. **Jockey-horse pairing history** -- familiarity effect
8. **Horse freshness composite** -- optimal racing frequency
9. **Weight x class interaction** -- handicap dynamic

### Priority 3: Higher effort, HIGH expected impact
Requires significant new feature engineering.

10. **Expected pace figure per horse** -- synthetic pace projection from actual sectional data
11. **Trainer course specialty** -- venue-specific trainer performance

### Defer
- **Layoff return performance**: Medium complexity, lower signal for win specifically (more useful for place)
- **Grade race debut flag**: Low signal, niche scenario
- **Seasonal pattern**: Mostly captured by track_condition_code already

## Win vs Place Feature Differences

Win prediction differs fundamentally from place/wide prediction:

| Aspect | Win Prediction | Place/Wide Prediction |
|--------|---------------|----------------------|
| **Target rarity** | ~7% base rate (1/14) | ~21% base rate (3/14) |
| **Odds skew** | High variance payouts; long shots pay 50-300x | Lower variance; favorites place often |
| **Key signal** | Dominance (winning margins, clear superiority) | Consistency (reliable top-3 finishes) |
| **Class sensitivity** | HIGH -- class drops that win are gold | MEDIUM -- class drops that place are common |
| **Pace importance** | HIGH -- pace setup determines who wins | MEDIUM -- pace affects placement but less deterministically |
| **Odds dynamics importance** | VERY HIGH -- sharp money targets wins | MEDIUM -- place pool has less sharp money |
| **Form cycle** | Need PEAK form (winning form) | Consistent form is sufficient |
| **Value detection** | Critical -- must find mispriced WINNERS | Easier -- favorites are often correctly priced for place |

This means features that capture **dominance signals** (large winning margins, fast sectional times, strong closing ability) are more valuable for win prediction than for place. The system's `closing_index_avg`, `timediff_avg`, and `harontimel5_zscore` are well-suited for this, but could be enhanced with:

- **Best-ever performance metrics** (peak harontime z-score, best timediff) rather than just averages
- **Win-specific form indicators** (wins in last 5 vs top-3s in last 5)
- **Dominance margin** (average winning margin when the horse won)

## Feature Effectiveness Assessment

Based on academic literature (Benter 1994, various ML racing papers) and community practice:

### HIGH importance for win prediction (must validate with feature importance analysis)
1. Market probability / odds-implied probability
2. Speed/ability rating (harontimel5_zscore, norm_finish_logit_avg)
3. Jockey quality (jockey_wr_overall, jockey_surprise)
4. Class level and class transitions
5. Odds dynamics (late money movement)
6. Form trend (improving vs declining)
7. Trainer quality

### MEDIUM importance (useful but secondary)
8. Bloodline/sire influence
9. Pace scenario fit
10. Course specialty
11. Weight factors
12. Rest period / freshness
13. Draw position (course-specific)

### LOW importance (limited signal for win specifically)
14. Race difficulty/entropy (more useful for bet sizing than prediction)
15. Overround (market structure, not horse-specific)
16. Field size (already accounted for in normalization)

## Recommended Validation Approach

Before building new features, run feature importance analysis on the EXISTING model:

```python
# After training WinTwoStageModel:
import lightgbm as lgb
import matplotlib.pyplot as plt

lgb.plot_importance(model.hit_model, importance_type='gain', max_num_features=30)
lgb.plot_importance(model.return_model, importance_type='gain', max_num_features=30)

# Also run SHAP for interaction effects:
import shap
explainer = shap.TreeExplainer(model.hit_model)
shap_values = explainer.shap_values(features)
shap.summary_plot(shap_values, features)
```

This will reveal which existing features actually contribute to win prediction vs noise, and guide where new feature investment yields the highest return.

## Sources

- Benter, W. (1994). "Computer Based Horse Race Handicapping and Wagering Systems" -- foundational paper on fundamental handicapping factors (speed, class, form)
- [Predicting Horse Racing Results Using LightGBM](https://ayatoashihara.github.io/myblog_multi/en/post/post16/) -- SHAP feature importance analysis
- [Horse Racing Prediction: ML Approach Part 2](https://medium.com/codeworksparis/horse-racing-prediction-a-machine-learning-approach-part-2-e9f5eb9a92e9) -- feature engineering for Hong Kong racing
- [Optimizing Horse Racing Predictions through Ensemble Learning](https://www.researchgate.net/publication/385301910) -- ensemble methods + betting optimization
- [Winning the Race: Profitability in Pari-Mutuel Horse Betting](https://repository.upenn.edu/bitstreams/9b580267-c2ab-45f5-a3e0-cace231c4fa4/download) -- UPenn paper on ROI-focused modeling with market features
- [Pace and Draw Analysis](https://lightspeedstats.com/horse-racing/pace-and-draw-analysis/) -- positional bias and pace analysis
- [JRA-VAN DataLab feature importance discussion](https://pc-keiba.com/wp/binary/) -- LightGBM binary classification with feature importance visualization
- [Feature engineering patterns for keiba AI](https://note.com/dijzpeb/n/n12e6f02db76a) -- 4 patterns for feature addition in horse racing ML
