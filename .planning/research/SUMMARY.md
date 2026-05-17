# Project Research Summary

**Project:** keiba-ai v1.7 Market-Independent Edge Discovery
**Domain:** Horse racing ML prediction -- race-level aggregation, market cross-consistency, diagnostic evaluation
**Researched:** 2026-05-17
**Confidence:** HIGH

## Executive Summary

The v1.7 milestone adds four capabilities to the keiba-ai horse racing ML system: race-level aggregation features (entropy, dispersion, odds gaps), market cross-consistency features (win-wide Harville ratios), Gain per Depth diagnostics (LightGBM tree structure analysis), and Residual IC evaluation (B/C/E information coefficient decomposition). These are inspired by the yurelu (zenn.dev) article, which demonstrated +40% C-orthogonal IC from race-level features and +120% from market-cross features, taking ROI from 0.91 to 1.66 across 408 races.

The recommended approach is additive: keep all existing 100+ features (never remove odds features -- that destroys the implicit two-stage tree structure), add 6 race-level features computed from existing win odds via standard groupby patterns, then add 4+ market cross-consistency features using existing win and wide odds data with Harville theoretical odds computation. Zero new library installations are required -- every capability is covered by the installed stack (pandas, scipy, sklearn, LightGBM 4.6.0). The only external action is an ETL run to extract odds_umaren/sanren Parquet files for future expansion.

The critical risks are: (1) binary outcome Spearman breakdown in Residual IC computation -- using a naive formulation produces mechanically negative IC values, requiring the 4-formulation battery as a cross-check; (2) look-ahead bias from using post-race payout odds instead of pre-race snapshot odds in cross-consistency features; (3) train-inference mismatch if race-level features are added to build_all() but not to the build_features() single-race inference path. Each of these has a clear prevention strategy documented in the pitfalls research.

## Key Findings

### Recommended Stack

Zero new dependencies. Every required tool is already installed at current versions. The v1.7 milestone is purely additive feature modules and diagnostic tools built with pandas groupby, scipy.stats, sklearn LinearRegression, and LightGBM trees_to_dataframe().

**Core technologies:**
- pandas groupby().transform(): Race-level feature broadcasting (constant within race, copied to every horse row) -- identical pattern to existing market_bias_features.py
- scipy.stats entropy + spearmanr: Shannon entropy for race readability, Spearman IC for model evaluation -- more numerically stable than manual implementations
- sklearn LinearRegression: OLS residualization for C-orthogonal IC (model predictions regressed on market probabilities, residual IC measured) -- mathematically identical to partial correlation
- LightGBM 4.6.0 trees_to_dataframe(): Per-node gain and depth extraction for Gain per Depth analysis -- confirmed 15-column output including tree_index, node_depth, split_gain
- Harville (1973) formula: Theoretical multi-horse odds from win implied probabilities -- enables win x wide cross-consistency without trio quinella data
### Expected Features

**Must have (table stakes):**
- Race-level aggregation features (rl_odds_dispersion, rl_top3_odds_gap, rl_favorite_rank_gap, rl_log_odds_entropy) -- yurelu 5 race-level features improved C-orthogonal IC by +40%
- rl_favorite_in_wide_top1 -- analogue of yurelu single most powerful feature (market consistency), uses wide odds instead of trio quinella
- Residual IC evaluation with 4-formulation battery (B-diff, C-orthogonal, E-incremental, per-race) -- without this instrument, there is no way to measure market-independent edge
- Promotion of existing computed-but-unregistered features (implied_prob_hhi, odds_skewness) to FEATURE_COLS

**Should have (competitive):**
- Harville wide odds ratio features (rl_wide_harville_ratio_fav, rl_wide_top3_harville_mean, rl_wide_harville_dispersion) -- deeper market cross-consistency signal
- Gain per Depth diagnostic -- validates that race-level features appear at shallow tree depths (implicit two-stage structure)
- Market conviction index (composite cross-consistency signal) -- aggregates multiple Harville ratios

**Defer (v2+):**
- Quinella (umaren) cross-consistency features -- requires ETL expansion for jodds_umaren time-series data beyond 2026
- Multi-dimensional orthogonal IC (win+wide+umaren simultaneously) -- correlated market signals may produce unstable residuals
- Stern/Henery models for more accurate theoretical odds -- Harville captures 90%+ of signal, diminishing returns

### Architecture Approach

Four new modules integrate into the existing pure-function feature pipeline (DataFrame in, DataFrame out). Race-level and market-cross features are new src/features/ modules wired into FeatureEngine.build_all(). Gain per Depth and Residual IC are new src/diagnostics/ classes hooked into TrainingPipeline._train_submodel(). The critical architectural decision is consolidating the wide odds merge into build_all() (via new wide_odds_df parameter) to eliminate code duplication between training and backtest pipelines.

**Major components:**
1. race_level_features.py (NEW): 6 race-level aggregation features from tanodds groupby -- entropy, dispersion, top-3 gap, favorite dominance, field competitiveness, longshot ratio
2. market_cross_features.py (NEW): 4+ market cross-consistency features using Harville theoretical wide odds -- win-wide divergence, favorite-in-wide-top1, Harville ratios
3. gain_per_depth.py (NEW): Diagnostic extracting LightGBM tree structure via trees_to_dataframe(), aggregating gain by depth level
4. residual_ic.py (NEW): B/C/E IC decomposition on OOF predictions, measuring market-independent predictive power
### Critical Pitfalls

1. **Binary outcome Spearman breakdown** -- Naive Residual IC (Spearman on binary residuals) produces mechanically negative IC (~-0.68). Use the 4-formulation battery (B-diff, C-orthogonal, E-incremental, per-race) and require consistent direction across all metrics.
2. **Look-ahead bias in multi-bet-type odds** -- Post-race payout odds from payouts.parquet must never be used as features. Only pre-race snapshot odds (tanodds, fukuoddslow, wide oddslow) are PIT-safe. Feature design must document the odds source for every cross-consistency feature.
3. **Train-inference feature parity** -- Race-level features added to build_all() must also be added to build_features() (the single-race inference path). The inference path currently only calls _map_basic_features() and skips all sub-modules. Both paths must produce identical new columns.
4. **Race-level feature redundancy** -- These features are constant within a race and provide no within-race discriminative power. They are infrastructure features that improve C-orthogonal IC, not Simple IC. yurelu 5 race-level features contributed +0.03 to ROI; the 5 market-cross features contributed +0.37. Set expectations accordingly.
5. **Never remove odds features** -- The Echo Chamber instinct is to remove odds (tanodds, popularity_rank) to force fundamental-only prediction. yurelu proved this wrong: removing odds dropped C-orthogonal IC from +0.0856 to -0.0261. The correct strategy is additive: keep existing features + add new ones.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Residual IC Evaluation Framework
**Rationale:** Must build the measurement instrument before adding features. Without Residual IC, there is no way to determine whether new features add market-independent information or just echo the market. The 4-formulation battery prevents the binary Spearman breakdown pitfall.
**Delivers:** src/diagnostics/residual_ic.py, IC reports on current model (baseline), 4-formulation cross-check system
**Addresses:** Pitfall #1 (binary Spearman breakdown)
**Avoids:** Building features without a way to measure their value

### Phase 2: Race-Level Aggregation Features
**Rationale:** No new data dependencies -- uses only existing tanodds already in the pipeline. Follows the exact same groupby-transform pattern as market_bias_features.py. These are the foundation features that enable market-cross features to work better (yurelu showed they activate at tree depth 3-4, providing race context).
**Delivers:** src/features/race_level_features.py, 6 new race-level features, promotion of implied_prob_hhi and odds_skewness to FEATURE_COLS, both build_all() and build_features() paths updated
**Uses:** pandas groupby/transform, scipy.stats.entropy
**Avoids:** Pitfall #3 (inference path missing features), Pitfall #7 (feature cache invalidation)

### Phase 3: Market Cross-Consistency Features
**Rationale:** The main weapon -- yurelu market-cross features delivered +120% C-orthogonal IC improvement. Uses existing wide odds data (17.9 MB, 38,825 races). Requires Harville theoretical odds computation as a shared prerequisite. Consolidates wide odds merge into build_all() to eliminate training/backtest duplication.
**Delivers:** src/features/market_cross_features.py, Harville wide odds computation, 4+ cross-consistency features, wide odds merge consolidation into build_all()
**Uses:** pandas merge + vectorized operations, Harville formula
**Avoids:** Pitfall #2 (look-ahead bias -- uses only pre-race wide odds snapshots), Pitfall #4 (redundancy -- cross-consistency is orthogonal to race-level)

### Phase 4: Gain per Depth Diagnostic
**Rationale:** Read-only diagnostic that validates race-level features are functioning correctly (appearing at shallow tree depths). Pure analysis tool with no effect on model predictions. Runs after model training as a verification step.
**Delivers:** src/diagnostics/gain_per_depth.py, depth-stratified gain distribution reports, implicit two-stage structure verification
**Uses:** LightGBM trees_to_dataframe(), pandas groupby
**Avoids:** Pitfall #5 (over-interpretation -- used as diagnostic only, not optimization target)

### Phase 5: Validation and Manifest Update
**Rationale:** All features and diagnostics are in place. Now run full validation to confirm ROI improvement, update feature manifests, and verify no leakage.
**Delivers:** Updated FEATURE_COLS manifest, SHA256 feature hash, full backtest with new features, walk-forward validation
**Avoids:** Pitfall #8 (stale cache after feature changes), all leakage risks

### Phase Ordering Rationale

- Residual IC first because it is the measurement instrument -- you cannot evaluate features without it
- Race-level features second because they have zero new data dependencies and follow proven patterns
- Market cross-consistency third because it depends on race-level patterns being established and requires the Harville computation prerequisite
- Gain per Depth fourth because it needs a trained model with the new features to analyze
- Validation last because it needs the frozen feature set and trained models

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Harville wide odds computation -- exact numerical implementation and edge case handling (small fields, zero odds, wide odds sparsity in early years)
- **Phase 3:** Wide odds pivot memory -- pivoting wide odds for all races at once may consume excessive memory; may need per-race or sparse approach

Phases with standard patterns (skip research-phase):
- **Phase 1:** Standard Spearman IC + OLS residualization, well-documented in quantitative finance
- **Phase 2:** Identical groupby-transform pattern to existing market_bias_features.py
- **Phase 4:** LightGBM API well-documented, code example verified in STACK.md

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All tools verified installed at required versions. LightGBM API tested live. scipy/sklearn imports verified. Zero new dependencies needed. |
| Features | HIGH | yurelu article provides strong empirical evidence (+40% IC race-level, +120% IC market-cross). Data availability verified (38,825 races with wide odds). Existing codebase audit confirms most race-level features already computed but not registered. |
| Architecture | HIGH | Full source code analysis of feature_engine.py, training_pipeline.py, backtest/engine.py, race_predictor.py. Integration points and insertion locations identified at line-level granularity. |
| Pitfalls | HIGH | 8 pitfalls identified with specific prevention strategies, warning signs, and recovery costs. Code-level analysis of leakage paths, cache mechanism, and dual-path (train/inference) feature computation. |

**Overall confidence:** HIGH

### Gaps to Address

- **Trio quinella (sanrenpuku) pre-race odds unavailable:** yurelu strongest feature used trio odds, which we do not have. The wide odds substitute (rl_favorite_in_wide_top1) is the closest analogue but signal strength is untested. Must validate during Phase 3 implementation.
- **Harville approximation accuracy for wide bets:** The exact Harville wide probability requires iterating over all permutations of positions 1-3. The proposed approximation is untested. Need numerical comparison during Phase 3.
- **Multi-seed Gain per Depth stability:** The two-stage structure hypothesis (shallow=market, deep=fundamental) is plausible but untested on our models. If depth patterns differ across seeds, the diagnostic has less interpretive value.
- **Wide odds sparsity in early years (2015-2017):** Market cross-consistency features may have many NaN values for early data. Coverage percentage needs checking before committing to features.

## Sources

### Primary (HIGH confidence)
- yurelu (zenn.dev): AI to 26 round giron shite kojin kaihatsu no keiba yosoku ML wo sodateta hanashi -- Race-level features, market-cross features, C-orthogonal IC improvements, Gain per Depth analysis, 4-formulation Residual IC battery, ROI 0.91 to 1.66 on 408 races
- Codebase analysis: feature_engine.py (build_all/build_features dual paths), market_bias_features.py (race-level pattern), training_pipeline.py (wide odds merge, OOF prediction), backtest/engine.py (wide odds merge), race_predictor.py (inference chain), stacked_ensemble.py (lgbm_model access)
- LightGBM 4.6.0: trees_to_dataframe() API verified via official docs and live test
- Data verification: odds_wide.parquet (3.68M rows, 38,825 races), payouts.parquet (38,835 races, 201 cols)

### Secondary (MEDIUM confidence)
- Harville (1973): Assigning probabilities to the outcomes of multi-entry competitions -- Harville formula for quinella/exacta/trifecta from win odds
- scipy 1.17.1: Verified spearmanr, entropy importable and functional
- sklearn 1.8.0: Verified LinearRegression for orthogonalization with synthetic data
- Snowberg and Wolfers: Explaining the Favorite-Longshot Bias -- Harville conditional probability derivation

### Tertiary (LOW confidence)
- Gain per Depth two-stage structure hypothesis -- plausible but untested on keiba-ai models
- Market cross-consistency ROI improvement replication -- yurelu results may not transfer due to different data environment (no trio odds)
- Harville approximation accuracy for wide bets -- numerical comparison with exact computation not yet performed

---
*Research completed: 2026-05-17*
*Ready for roadmap: yes*