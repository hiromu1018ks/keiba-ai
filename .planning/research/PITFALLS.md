# Pitfalls Research

**Domain:** Horse racing ML -- adding race-level aggregation features, market cross-consistency features, Residual IC evaluation, and Gain per Depth analysis to an existing 100+ feature LightGBM pipeline
**Researched:** 2026-05-17
**Confidence:** HIGH (reference article zenn.dev/yurelu + direct codebase analysis + domain knowledge)

## Critical Pitfalls

### Pitfall 1: Binary Outcome Spearman Breakdown in Residual IC

**What goes wrong:**
Using Spearman(model_score, y - market_probs) where y is binary {0,1} produces mechanically negative IC values that make the model look worse than random. The residual = y - market_probs is bimodal: mostly large-negative (y=0, market_probs moderate-to-high) with a thin spike at large-positive (y=1, market_probs moderate). The reference article (zenn.dev/yurelu) demonstrated this exactly: the naive formulation produced IC = -0.68 and led to false "Echo Chamber" conclusions.

**Why it happens:**
The residual y - market_probs for binary outcomes is dominated by the y=0 class (12-17 out of 18 horses per race). When model_score correlates with market_probs (which it always does for any competent model), the majority of data points cluster at (high model_score, large-negative residual), producing a mechanically negative Spearman. This is a structural artifact of applying a financial metric (designed for continuous returns) to a binary outcome.

**How to avoid:**
Use the 4-formulation battery from the reference article as a cross-check system:
- **A (naive):** Spearman(model, y - market) -- KNOWN BROKEN for binary. Include only as a negative control to demonstrate the failure.
- **B (difference):** Spearman(model - market, y) -- safe but sensitive to model-market correlation direction.
- **C (orthogonal):** Spearman(orthog(model | market), y) -- the primary metric for market-independent predictive power. Use GAM (SplineTransformer + LogisticRegression) to compute market_probs from p_implied, then residualize.
- **D (per-race):** Mean of per-race Spearman(model, y) -- auxiliary metric for ranking quality.
- **E (incremental):** IC(model, y) - IC(market, y) -- most interpretable, simplest to explain.

All 4 should move in consistent directions. If C is positive but B is negative (or vice versa), investigate before drawing conclusions.

**Warning signs:**
- Residual IC values below -0.3 (mechanically negative, not real signal).
- A and C giving opposite-signed results.
- Any single IC metric being used as the sole decision criterion for feature evaluation.

**Phase to address:**
Residual IC implementation phase. Must be built BEFORE race-level features are evaluated -- otherwise there is no instrument to measure whether new features add market-independent information.

---

### Pitfall 2: Look-Ahead Bias in Multi-Bet-Type Odds Features

**What goes wrong:**
Market cross-consistency features (e.g., "Does 1st-favorite appear in trifecta-box top combination?") require multi-bet-type odds data. Using POST-RACE odds (final payout odds) instead of pre-race snapshot odds introduces look-ahead bias. The reference article used "odds at voting deadline" for all bet types, but keiba-ai currently only has:
- `tanodds` / `fukuoddslow` (win/place snapshot from `jodds_tanpuku`)
- `odds_wide` (wide odds with `oddslow` / `oddshigh`)
- Time series odds (`jodds_tanpuku` time-series)
- **NO trio (sanrenpuku) or trifecta (sanrentan) odds data in the current pipeline**

EveryDB2 queries only fetch `s_jodds_tanpuku` / `n_jodds_tanpuku`. Trio/trifecta odds would require new EveryDB2 tables (e.g., `s_jodds_sanrenpuku`) that are not currently extracted. The reference article's author had the same constraint: their dataset had only single, trio, and trifecta but NOT place odds, forcing a design pivot.

**Why it happens:**
- The reference article had trio odds in their dataset but keiba-ai does not extract them from EveryDB2.
- If you construct "theoretical trio odds from win odds" (e.g., p1 * p2 * p3 from normalized win implied probabilities), you create a feature that is a deterministic function of existing features (tanodds). LightGBM discovers this relationship on its own, making the feature redundant.
- Payout data (`raw.payouts`) exists for post-race verification but contains POST-RACE information and must never be used as a feature.

**How to avoid:**
1. Audit EveryDB2 for available multi-bet-type odds tables BEFORE designing cross-consistency features. Specifically check if `s_jodds_sanrenpuku`, `s_jodds_sanrentan`, or `s_jodds_umaren` external tables exist in EveryDB2.
2. If trio/trifecta odds ARE available, add them to the ETL pipeline (`run_etl.py`) BEFORE building features.
3. If they are NOT available, redesign cross-consistency features to use only available data:
   - **Win-place consistency:** Compare tanodds implied probability with fukuoddslow implied probability per horse. A horse with high win implied prob but low place implied prob may indicate market disagreement.
   - **Wide odds structure:** The system already has `odds_wide` data -- use wide-odds dispersion across pairs as a race-level signal.
   - **Win-wide cross-consistency:** Does the top-favorite horse appear in the lowest wide-odds pair? Similar to the reference article's `rl_market_consistency`.
4. NEVER use payout data (`paytansyopay1`, `payfukusyopay*`) as feature input. These are POST_RACE by definition.

**Warning signs:**
- Feature that perfectly separates winners from losers (suspicious IC > 0.5).
- Cross-consistency feature that uses data only available after race completion.
- Features computed from theoretical odds products (p1*p2*p3) that add zero information beyond individual p_i values.

**Phase to address:**
Market cross-consistency feature implementation phase. ETL extension (if needed) must come first. Feature design must document which odds snapshot (pre-race vs post-race) each feature uses.

---

### Pitfall 3: Race-Level Feature Leakage via Self-Inclusion in Aggregation

**What goes wrong:**
When computing race-level aggregation features (e.g., `rl_odds_dispersion` = std of odds across all horses in a race), you include ALL horses in the groupby, including the horse being predicted. For odds-based aggregation, this is fine -- at prediction time, you have all horses' odds. But for performance-based aggregation (e.g., "average past win rate of all runners in this race"), including the current horse's own past performance creates a subtle inconsistency: the feature has different information content depending on whether the horse is the first or last in the prediction order.

More critically: if race-level features are computed from fields that include post-race information for OTHER horses in the same race (e.g., `kakuteijyuni` of other runners), this is direct leakage.

**Why it happens:**
The existing `POST_RACE_COLS` guard in `feature_engine.py` (line 362-370) drops post-race columns from the final output. But it does NOT prevent a race-level aggregation computed BEFORE the guard from having used those columns. For example:
```python
# WRONG: compute aggregate BEFORE dropping post-race cols
df["avg_finish_pos_in_race"] = df.groupby("race_id")["kakuteijyuni"].transform("mean")
# NOW drop post-race cols -- but the aggregate already leaked!
```

The current system's `market_bias_features.py` and `intra_race_features.py` use only `tanodds`, `bataijyu`, and `umaban` for aggregation, which are pre-race features. This is safe. New race-level features must be held to the same standard.

**How to avoid:**
1. Only aggregate over columns available at prediction time (before race start).
2. Safe aggregation sources: `tanodds`, `fukuoddslow`, `umaban`, `bataijyu`, `popularity_rank`, `field_size`, `grade_code`, bloodline features, jockey/trainer stats.
3. Never aggregate over: `kakuteijyuni`, `confirmed_odds`, `ninki` (confirmed popularity), `time`, `timediff`, `harontimel3/4`, corner positions, `honsyokin`.
4. Add a CI test similar to the existing POST_RACE_COLS whitelist check that verifies race-level features only use pre-race columns.
5. Compute race-level features AFTER the POST_RACE_COLS drop, or explicitly document that the aggregation source columns are pre-race.

**Warning signs:**
- Race-level feature with per-race IC > 0.4 (suspiciously high for a single aggregate).
- Feature that is constant within a race but varies across races -- verify it does not encode "who won."
- Backtest ROI jumping >10pp from a single race-level feature addition.

**Phase to address:**
Race-level feature implementation phase. Must include leakage test as part of the implementation phase, not as a separate verification phase later.

---

### Pitfall 4: Race-Level Features Become Redundant Constants Within a Race

**What goes wrong:**
Race-level aggregation features (entropy, dispersion, top-odds gap) are constant for all horses within the same race. With 18 horses per race, a single race-level feature adds 18 identical values per race. LightGBM handles this correctly (it can split on constant-within-group features), but these features provide NO discriminative power between horses within the same race for the binary win/lose target. They only help distinguish between "hard races" and "easy races" at the race level.

**Why it happens:**
The reference article demonstrates that race-level features work because they improve the IMPLICIT two-stage structure of LightGBM (upper nodes = race context, lower nodes = individual horse). This only works when strong individual-horse features exist in the lower nodes. In a system with 100+ features, the upper nodes may already be saturated with individual-horse features that act as proxies for race-level information (e.g., `popularity_rank` already encodes relative standing).

**How to avoid:**
1. Expect race-level features to improve C-orthogonal IC and B-difference IC, NOT Simple IC. The reference article showed v9 (race-level) improved C-orthogonal by +40% but Simple IC by only +0.001.
2. Evaluate race-level features using Gain per Depth analysis. If they appear at Depth 1-3, they are functioning as race-context features. If they appear at Depth 5+, they are not being used correctly.
3. Do NOT expect race-level features alone to improve ROI substantially. They are infrastructure features that enable other features (especially market-cross features) to work better.
4. The reference article's race-level features (5 features) contributed +0.03 to ROI, while market-cross features (5 features) contributed +0.37. Race-level features are a foundation, not the payoff.

**Warning signs:**
- Race-level features appearing only at very low depths (< 2) or very high depths (> 8) in Gain per Depth.
- Zero or negative SHAP values for race-level features.
- C-orthogonal IC not improving after adding race-level features.

**Phase to address:**
Race-level feature implementation phase. Set expectations correctly: these are "infrastructure" features, not "payoff" features.

---

### Pitfall 5: Gain per Depth Over-Interpretation and Over-Optimization

**What goes wrong:**
Gain per Depth analysis (using `trees_to_dataframe()` from LightGBM) reveals which features dominate at which tree depth. The reference article shows a clean pattern: Depth 1-2 = Market (99%), Depth 3-4 = Market-cross (70%), Depth 5+ = Categorical/Fundamental. This is seductively interpretable, but over-interpreting this structure and trying to "optimize" it (e.g., forcing certain features to specific depths via feature engineering) leads to overfitting to a particular model checkpoint.

**Why it happens:**
- Gain per Depth is a POST-HOC analysis of a specific trained model. If the model is retrained with different seeds, data, or hyperparameters, the depth allocation shifts.
- LightGBM uses leaf-wise growth (not level-wise like XGBoost). The "depth" in `trees_to_dataframe()` does NOT correspond to uniform tree levels. A depth-3 node in one tree may correspond to a completely different decision boundary than a depth-3 node in another tree.
- Feature correlation distorts gain allocation: if two features are highly correlated, LightGBM picks one arbitrarily for the split, making the other appear "unused" at that depth even though it carries equivalent information.

**How to avoid:**
1. Use Gain per Depth as a DIAGNOSTIC tool, not an optimization target. Verify that race-level features appear at upper depths and individual features at lower depths. Do NOT try to force this pattern.
2. Aggregate across all trees (the reference article's table shows aggregate percentages per depth, which is the correct approach).
3. Run Gain per Depth analysis on at least 2-3 models (different seeds) to verify the pattern is stable, not an artifact of one training run.
4. Do NOT make feature engineering decisions based solely on Gain per Depth. Always cross-validate with Residual IC metrics.
5. The keiba-ai system does NOT currently have `trees_to_dataframe()` usage. Only LightGBM supports this natively. XGBoost requires `trees_to_dataframe()` from `xgboost.plotting` and CatBoost has `calc_feature_statistics()`. The multi-model stacking means analysis must cover all 3 base models, not just LightGBM.

**Warning signs:**
- Gain per Depth analysis showing dramatically different patterns across training seeds.
- Making feature engineering decisions to "push" features to specific depths.
- Interpreting depth-1 splits as "the model's most important decision" without considering that leaf-wise growth means depth-1 may apply to only a subset of data.

**Phase to address:**
Gain per Depth diagnostic phase. Use it as validation that race-level features are working as intended, not as a design tool.

---

### Pitfall 6: "Removing Odds Features Makes Things Better" Fallacy

**What goes wrong:**
After discovering Echo Chamber (model is a market clone), the instinct is to remove odds features (tanodds, popularity_rank) from the model. The reference article PROVES this is wrong: v3_fundamental (odds removed) had C-orthogonal IC = -0.0261 while v3 (odds kept) had C-orthogonal IC = +0.0856. Removing odds made things WORSE, not better. The correct approach is to ADD new information (race-level, market-cross) while KEEPING odds features.

**Why it happens:**
LightGBM's tree structure implicitly creates a two-stage architecture: upper nodes capture market information (via tanodds/popularity), lower nodes capture fundamental information (via bloodline/past performance/jockey). Removing odds features destroys the upper-node market context, making it harder for the lower nodes to refine predictions. The model becomes weaker overall because it loses the "easy wins" from market information.

**How to avoid:**
1. NEVER remove odds features from the model. The keiba-ai system's 100+ features already include tanodds, fukuoddslow, and popularity-related features. Keep all of them.
2. The correct strategy is ADDITIVE: keep existing features + add race-level + add market-cross.
3. Verify this approach by comparing two models: (a) current features + new features vs (b) current features with odds removed + new features. If (a) wins, the principle holds.
4. The keiba-ai system already has `market_bias_features.py` computing `p_market_win_adj`, `market_entropy`, `overround` -- these are "odds" features that must be kept. The new race-level features are a DIFFERENT type of odds usage (aggregation vs individual).

**Warning signs:**
- A/B test showing that removing odds features improves Simple IC but worsens C-orthogonal IC.
- Backtest ROI improving from feature removal (likely overfitting to a specific test period).
- Any proposal to "simplify" the model by removing correlated features.

**Phase to address:**
All phases. This is a meta-principle that must be maintained throughout the entire v1.7 milestone.

---

### Pitfall 7: Feature Cache Invalidation After Adding New Feature Modules

**What goes wrong:**
The `FeatureEngine.build_all()` method uses a caching mechanism (`compute_cache_key()`) that includes a code hash from `src/features/*.py` via `compute_code_hash()`. This was added in v1.6 and should correctly invalidate when new `.py` files are added to `src/features/`. However, if new features are computed in a module outside `src/features/` or added as inline code in `feature_engine.py` without creating a new file, the hash may not change and stale cached features will be served without new columns.

**Why it happens:**
The `compute_code_hash()` function scans `src/features/*.py` files. If new features are:
- Located outside `src/features/` (e.g., in a notebook or script)
- Added as inline code in `feature_engine.py` (which IS scanned, but the hash change may be subtle)
- Named with a different extension (`.pyx`, `.pyi`)

...then the hash may not change or the cache may not invalidate properly.

**How to avoid:**
1. All new feature computation must live in `src/features/*.py` modules.
2. After adding new feature modules, verify cache invalidation by checking that the `feat_*.parquet` filename changes.
3. The `build_all()` method calls feature modules in a specific order. New race-level features should be computed AFTER existing `market_bias_features` but BEFORE the POST_RACE_COLS drop.
4. During development, delete `data/features/cache/` to force recomputation.

**Warning signs:**
- New features showing zero variance in training data.
- Feature importance of new features being exactly zero across all models.
- Cache HIT log message when you expected a MISS after adding new code.

**Phase to address:**
First training run after adding any new feature module.

---

### Pitfall 8: Inference Path Missing Race-Level Feature Computation

**What goes wrong:**
`FeatureEngine.build_features()` (the single-race inference method) is a SEPARATE code path from `build_all()` (the batch training method). The inference path (lines 385-455) currently only computes basic features via `_map_basic_features()` and does NOT call the sub-modules (intra_race, odds_dynamics, market_bias, etc.). If race-level features are added only to `build_all()`, they will be missing at inference time, causing a train-inference mismatch.

**Why it happens:**
The current design has an explicit comment at line 453: "6. sub-module feature computation (inference -- hist features excluded)" with an empty implementation. The single-race path returns after `_map_basic_features()`. This was acceptable because the inference pipeline (`BettingOrchestrator`) handles some features differently, but race-level features MUST be computed at inference time for live prediction.

**How to avoid:**
1. Race-level features must be computed in BOTH `build_all()` and `build_features()`.
2. For `build_features()`, the race-level aggregation is simpler because all horses for one race are already in the DataFrame. No groupby across races is needed.
3. The inference path receives `odds_snapshot` -- verify this contains `tanodds` for all horses in the race (needed for entropy/dispersion computation).
4. Add an explicit test that verifies feature parity between `build_all()` and `build_features()` for new race-level columns.

**Warning signs:**
- Live predictions producing different rankings than backtest predictions.
- NaN values for race-level features in the inference path but not in training.
- `FeatureEngine.build_features()` returning fewer columns than `build_all()` for the same race.

**Phase to address:**
Race-level feature implementation phase. Must update BOTH code paths simultaneously.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Compute race-level features inline in `build_all()` | Faster to implement | Duplicated logic in `build_features()`, easy to drift | Never -- must be in a separate module |
| Use theoretical trio odds (p1*p2*p3) instead of real trio odds | No new ETL needed | Feature is deterministic function of tanodds, adds zero information | Only as a baseline comparison, never as a shipped feature |
| Skip `build_features()` update for race-level features | Ship faster | Train-inference mismatch, broken live predictions | Never |
| Single IC formulation instead of 4-formulation battery | Simpler evaluation | Misleading conclusions from binary Spearman breakdown | Never -- all 4 formulations are cheap to compute |
| Gain per Depth on single model seed | Faster analysis | Unstable conclusions, overfitting to one training run | Acceptable for initial exploration only; must verify with multi-seed |
| Use payout odds as "approximation" for pre-race odds | Easier data access | Direct look-ahead bias, inflated metrics | Never |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| EveryDB2 trio/trifecta odds | Assuming EveryDB2 has trio odds tables without checking | Query EveryDB2 schema first; if unavailable, design features around win+place+wide odds only |
| Parquet cache invalidation | Adding features without verifying cache key changes | Run with explicit cache check after any `src/features/` change |
| `build_features()` vs `build_all()` parity | Only adding features to `build_all()` | Must update both paths; add feature-parity test |
| POST_RACE_COLS whitelist | Computing race-level aggregates BEFORE the POST_RACE drop guard | Either compute after the drop, or explicitly verify source columns are pre-race |
| Residual IC with existing OOF predictions | Using training predictions instead of OOF predictions for IC calculation | Must use OOF (out-of-fold) predictions to avoid in-sample bias in IC |
| Stacking meta-learner and IC | Computing Residual IC on meta-learner output vs base model output | Compute IC at the level where features are being evaluated (base model output for feature evaluation) |
| LightGBM `trees_to_dataframe()` | Assuming all 3 stacking models support the same API | Only LightGBM supports this natively; XGBoost needs `xgboost.plotting`; CatBoost has different API |
| Multi-bet odds join keys | Joining wide odds by `race_id` only (missing `kumi` column) | Wide odds use (race_id, kumi) composite key; pivot required for per-horse features |
| FEATURE_COLS manifest | Forgetting to update SHA256 manifest after adding new features | Must regenerate manifest after any feature column change |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Race-level features on small fields | High variance entropy/dispersion for 5-6 horse fields | Add `min_field_size` guard; return NaN for fields < 5 | Races with < 8 runners (common at small venues) |
| Multi-bet-type odds sparse coverage | Many NaN values in cross-consistency features | Check coverage percentage before committing to feature; set NaN fallback | Early years (2015-2017) when data collection was incomplete |
| Gain per Depth on shallow trees | Few depth levels to analyze, inconclusive patterns | Ensure `max_depth` is sufficient (existing system uses default 31 for LightGBM) | Models with aggressive early stopping |
| Residual IC with small holdout | Wildly fluctuating IC values across different holdout periods | Use holdout of at least 300+ races; report IC with confidence intervals | Any evaluation with < 200 races (reference article uses 336 + 72 OOH) |
| Wide odds pivot memory | Pivoting wide odds for all races at once consumes excessive memory | Pivot per race or use sparse representation | Full-dataset training (50K+ races) |

## "Looks Done But Isn't" Checklist

- [ ] **Race-level features:** Often missing the `build_features()` inference path update -- verify both `build_all()` and `build_features()` produce the same columns for a single race
- [ ] **Residual IC:** Often implemented with only formulation A (naive Spearman) -- verify all 4 formulations (B/C/D/E) are implemented and cross-checked
- [ ] **Market cross-consistency:** Often uses post-race payout odds instead of pre-race snapshot odds -- verify every odds source used is a pre-race snapshot
- [ ] **Gain per Depth:** Often analyzed on a single model/seed -- verify analysis across at least 3 seeds to confirm pattern stability
- [ ] **Feature cache:** Often stale after adding new features -- verify `feat_*.parquet` filename changes after code modification
- [ ] **Leakage test for race-level:** Often forgotten because "it is just aggregation" -- verify CI test covers race-level feature source columns
- [ ] **Multi-bet-type odds availability:** Often assumed to exist -- verify EveryDB2 has the required tables before designing features around them
- [ ] **FEATURE_COLS manifest:** Often not updated -- verify SHA256 manifest is regenerated after feature column changes
- [ ] **OOF predictions for IC:** Often uses in-sample predictions -- verify Residual IC uses OOF predictions only

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Binary Spearman breakdown | LOW | Switch to B/C/D/E formulations; no retraining needed |
| Look-ahead bias in multi-bet features | HIGH | Remove affected features, retrain model, re-run backtest (~1 hour) |
| Race-level self-inclusion leakage | MEDIUM | Re-compute features excluding target horse, retrain (~30 min) |
| Feature cache serving stale data | LOW | Delete `data/features/cache/`, re-run training |
| Missing inference path features | MEDIUM | Add to `build_features()`, re-deploy inference pipeline |
| Over-optimized Gain per Depth | LOW | Revert to original features, use Gain per Depth only as diagnostic |
| Removed odds features (fallacy) | MEDIUM | Restore odds features, retrain, verify C-orthogonal IC improvement |
| Multi-bet-type odds not available | HIGH | Redesign features to use available data (win+place+wide only), or extend ETL to extract new odds tables |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Binary Spearman breakdown | Residual IC implementation phase | All 4 formulations produce consistent direction |
| Look-ahead bias in multi-bet features | Market cross-consistency design phase | Feature audit: all odds sources are pre-race snapshots |
| Race-level self-inclusion leakage | Race-level feature implementation phase | CI test: race-level features use only pre-race columns |
| Race-level features become redundant constants | Race-level feature evaluation phase | Gain per Depth shows features at Depth 1-3; C-orthogonal IC improves |
| Gain per Depth over-interpretation | Gain per Depth analysis phase | Multi-seed stability check; depth pattern consistent across seeds |
| Removing odds features fallacy | All phases (meta-principle) | A/B test: current+new vs current-without-odds+new |
| Feature cache invalidation | First training run with new features | Verify cache filename changes; verify new columns in output |
| Inference path missing features | Race-level feature implementation phase | Feature parity test between `build_all()` and `build_features()` |

## Sources

- zenn.dev/yurelu "AI to 26 round giron shite kojin kaihatsu no keiba yosoku ML wo sodateta hanashi" -- primary reference for Echo Chamber, Residual IC 4-formulation battery, Gain per Depth analysis, race-level features (v9), and market cross-consistency features (v11). Includes quantitative results from 408-race holdout evaluation.
- keiba-ai codebase analysis: `src/features/feature_engine.py` (build_all/build_features dual paths, cache mechanism), `src/features/market_bias_features.py` (existing entropy/overround computation), `src/features/intra_race_features.py` (existing per-race aggregation), `src/features/odds_dynamics_features.py` (odds time series), `src/db/etl.py` (table type rules, odds conversion), `src/db/readers.py` (odds loading, wide odds), `src/db/schema.py` (wide_odds table definition), `src/db/everydb2_queries.py` (available EveryDB2 tables), `src/domain/types.py` (POST_RACE_COLS definition)
- ML Digest "The Illustrated LightGBM" -- leaf-wise vs level-wise growth strategy and implications for depth interpretation
- Bill Benter (1994) "Computer Based Horse Race Handicapping and Wagering Systems" -- original Two-Stage architecture context

---
*Pitfalls research for: keiba-ai v1.7 Market-Independent Edge Discovery milestone*
*Previous version: v1.6 Feature Engineering Overhaul (2026-05-10)*
*Researched: 2026-05-17*
