# Project Research Summary

**Project:** keiba-ai v1.6 Feature Engineering Overhaul
**Domain:** Horse racing ML prediction — feature engineering
**Researched:** 2026-05-10
**Confidence:** HIGH

## Executive Summary

This research covers a comprehensive feature engineering overhaul for the keiba-ai horse racing prediction system. The system currently has 100+ features across 14 modules, with a backtest ROI of 84.4% (target: 100%+). The v1.5 CQR failure demonstrated that adding complex post-processing layers doesn't improve prediction — the quality of model inputs (features) is the primary lever.

The recommended approach is a three-pronged strategy: (1) audit and prune existing noisy features, (2) activate 12 already-implemented but unwired features (jockey/trainer/combo stats), and (3) extract new features from 40+ unused EveryDB2 tables. No new library installations are needed. Estimated ROI improvement: +7~21pp (91~105%).

The critical risk is data leakage from post-race columns. EveryDB2 contains both pre-race and post-race data, and one known leakage path (`build_all()` exit drop) is unfixed. A safety gate phase must come first, and all new features must be classified PRE/POST before use.

## Key Findings

### Recommended Stack

Zero new library installations needed. The existing stack (LightGBM, XGBoost, CatBoost, pandas, numpy, scikit-learn, scipy) covers all feature engineering tasks. The only change is declaring `scipy>=1.11` explicitly in pyproject.toml (currently an undeclared transitive dependency).

**Core technologies:**
- LightGBM `feature_importance()` (gain/split): Feature audit baseline
- sklearn `permutation_importance`: Gold-standard feature importance on held-out data
- pandas vectorized operations: Feature interactions and transformations (already demonstrated in `interaction_features.py`)

### Expected Features

**Must have (table stakes):**
- Feature importance audit (permutation + gain): Identify noisy features diluting signal
- POST_RACE leakage fix: Drop post-race columns at `build_all()` exit
- JockeyContext/TrainerContext/Combo wiring: 12 already-implemented features, ~45 lines to connect
- Feature cache invalidation protocol: Clear cache when feature modules change

**Should have (competitive):**
- EveryDB2 unused table features: n_mining (82 cols), n_taisyogata_mining (pairwise), n_hansyoku (pedigree), n_record (course records)
- Relative comparison features: Horse-vs-horse within-race rankings
- Target encoding for high-cardinality categoricals: blood_keito_cd, kisyucode
- Feature interaction engineering: 10-15 domain-motivated interaction terms

**Defer (v2+):**
- n_mining deep analytics: Column semantics unknown, needs DB inspection first
- LSTM/Transformer features: Overkill for 5-15 past runs
- External data sources: Out of scope per PROJECT.md

### Architecture Approach

The existing feature pipeline follows a pure-function pattern (DataFrame in, DataFrame out) with 22 feature modules chained through `FeatureEngine.build_all()`. Adding new features means writing a new module (~100-200 lines) and wiring it into the pipeline (~20 lines). No structural changes needed. Two modules (JockeyContextFeatures, TrainerContextFeatures) are already implemented and tested but not wired.

**Major components:**
1. Feature audit script (new): Extracts feature importance from trained models, produces prune list
2. New feature modules (additive): Follow existing 22-module pattern, pure functions
3. Feature interaction module (additive): Vectorized pandas operations on existing features
4. POST_RACE leakage validator (new): CI test ensuring no post-race columns in feature output

### Critical Pitfalls

1. **Post-race data leakage** — `build_all()` exit doesn't drop POST_RACE_COLS; one path unfixed from v1.5. Fix: Add drop at `build_all()` return.
2. **Impurity importance bias** — LightGBM default gain importance favors high-cardinality features. Fix: Use permutation importance on held-out data for audit.
3. **Feature cache invalidation** — Cache key uses input file paths, not computation code. Adding features silently serves stale cache. Fix: Manual cache clearing protocol.
4. **EveryDB2 column semantics** — Many columns are post-race (odds, ninki, kyakusitukubun). Fix: PRE/POST classification document before any new feature extraction.
5. **Feature explosion** — Adding interactions without limits can overfit. Fix: Cap at 10-15 domain-motivated interactions.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Safety Gate — Leakage Fix and Audit Framework
**Rationale:** Safety first. The unfixed POST_RACE drop at `build_all()` exit is a known leak. Must fix before any feature work. Also build the audit tooling that subsequent phases depend on.
**Delivers:** POST_RACE leak fix, feature importance audit script, CI validation test
**Addresses:** Pitfall #1 (post-race leakage), Pitfall #2 (impurity importance bias)
**Avoids:** Building features on top of a leaky pipeline

### Phase 2: Feature Audit and Pruning
**Rationale:** Establish a clean baseline ROI before adding features. Pruning noisy features can itself improve ROI (+0~3pp).
**Delivers:** Pruned feature set, baseline ROI measurement with audit script
**Uses:** LightGBM feature_importance, sklearn permutation_importance
**Implements:** Audit script component from Phase 1

### Phase 3: Quick Win — Wire Existing Modules
**Rationale:** JockeyContextFeatures (4), TrainerContextFeatures (4), JockeyTrainerComboFeatures (4) are implemented, tested, and PIT-safe. ~45 lines to connect. Estimated ROI: +3~8pp.
**Delivers:** 12 new active features, ROI measurement
**Implements:** Wiring in training_pipeline.py

### Phase 4: New Features from EveryDB2
**Rationale:** With clean baseline and quick wins in place, now add high-value features from unused EveryDB2 tables.
**Delivers:** New feature modules from n_mining, n_hansyoku, n_record, relative comparison features
**Research flag:** n_mining column semantics unknown — needs DB inspection during planning

### Phase 5: Feature Interactions
**Rationale:** Interactions depend on final base feature set (from Phases 2-4). Limited to 10-15 domain-motivated interactions.
**Delivers:** Interaction features, final ROI measurement
**Avoids:** Pitfall #5 (feature explosion)

### Phase 6: Validation and Freeze
**Rationale:** Final walk-forward validation of complete feature set. ROI 100%+ target verification.
**Delivers:** WF validation results, frozen feature set, performance report

### Phase Ordering Rationale

- Safety gate first because building on a leaky pipeline wastes all downstream work
- Audit before adding because noisy features dilute the signal of new ones
- Quick wins before complex features because 12 free features at ~45 lines > any new module
- Interactions last because they depend on the final base feature set
- Validation last because it needs the frozen feature set

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4:** n_mining table (82 cols, unknown semantics, PRE/POST classification needed)
- **Phase 4:** n_taisyogata_mining (pairwise comparison data, structure unknown)

Phases with standard patterns (skip research-phase):
- **Phase 1:** Well-documented leakage fix pattern
- **Phase 2:** Standard permutation importance methodology
- **Phase 3:** Direct wiring of existing modules
- **Phase 5:** Standard pandas interaction patterns

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All tools verified installed, no new dependencies needed |
| Features | HIGH | 22 feature modules audited, 103 ETL tables cross-referenced, 12 unwired features identified |
| Architecture | HIGH | Pure-function pattern well-established across 22 modules, no structural changes needed |
| Pitfalls | HIGH | v1.5 CQR failure provides concrete evidence, code-level analysis of leakage paths |

**Overall confidence:** HIGH

### Gaps to Address

- **n_mining column semantics:** 82-column table with unknown column names/meanings. Need DB inspection during Phase 4 planning.
- **n_taisyogata_mining structure:** Pairwise comparison data potentially valuable but structure unknown. DB inspection needed.
- **POST_RACE completeness:** Current `POST_RACE_COLS` list may not cover all post-race columns in new tables. Need per-table PRE/POST classification.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: 22 feature modules in src/features/, training_pipeline.py, feature_engine.py
- ETL configuration: 103 tables extracted, 40+ unused
- v1.5 failure analysis: CQR overfitting root cause and fix (commit f3a4c10)

### Secondary (MEDIUM confidence)
- LightGBM documentation: feature importance methods, categorical handling
- sklearn documentation: permutation_importance API
- Horse racing ML literature: feature engineering patterns

### Tertiary (LOW confidence)
- EveryDB2 table semantics: Inferred from table/column names, needs DB verification
- ROI improvement estimates: Based on similar feature engineering improvements in literature

---
*Research completed: 2026-05-10*
*Ready for roadmap: yes*
