# Project Research Summary

**Project:** keiba-ai v2.4 -- Paper Trading Pipeline Integration
**Domain:** ML prediction pipeline integration (BT-to-PT alignment)
**Researched:** 2026-06-06
**Confidence:** HIGH

## Executive Summary

v2.4 is an integration milestone, not a feature milestone. Its purpose is to make the paper trading pipeline produce trustworthy ROI measurements by aligning it with the backtest pipeline that has already been validated. Direct codebase analysis reveals seven independent code gaps that each make PT ROI incomparable to BT ROI: three missing feature modules in PT, no win/wide payout settlement, result=0.0 conflating pending with lost, no strategy manifest enforcement, no OddsBandFilter, and a regime mismatch (dynamic in PT vs hardcoded AGGRESSIVE in BT). The current PT system effectively measures a different pipeline than BT.

The recommended approach is a four-phase extraction-and-alignment strategy. Phase 1 fixes settlement integrity so ROI numbers become meaningful at all. Phase 2 extracts a shared build_inference_features() function from the 300+ lines of feature construction duplicated across four locations (BT engine plus three copies in the PT script), guaranteeing feature parity by construction. Phase 3 aligns strategy parameters (manifest, regime, OddsBandFilter, betting mode) so PT executes the same strategy BT validated. Phase 4 adds the one-command run mode and reporting expansion on top of the now-reliable pipeline.

The key risk is Phase 2: extracting the shared feature builder from BacktestEngine.prepare_data() without breaking BT. A full BT regression test before and after extraction is mandatory. The second risk is OddsBandFilter calibration, which currently requires training-period bet history that PT does not have; the solution is to save calibration data as a model artifact during run_train.py.

## Key Findings

### Recommended Stack

Zero new external dependencies. Every v2.4 feature is implementable with the existing installed stack (Python 3.11, pandas 2.3.3, pyarrow 23.0.1, LightGBM 4.6.0, mlflow 3.10.1, Jinja2 3.1.6). This is an integration milestone reusing proven components: build_*_payout_map() from backtest.engine, ParameterFreezeProtocol from backtest.parameter_freeze_protocol, and RaceWatcher/SafetyGuard/RaceScheduler from automation/.

**Core technologies:**
- **Shared feature builder function** (src/paper_trading/feature_builder.py): extracted from BT engine, called by both BT and PT -- single source of truth for feature construction
- **Bet status lifecycle** (pandas string column): pending / settled replaces ambiguous result=0.0 check
- **Existing settlement functions** (build_win_payout_map, build_payout_map, build_wide_payout_map): imported directly from backtest.engine, no reimplementation
- **ParameterFreezeProtocol**: reused for PT manifest verification, same pattern as BT
- **argparse + sequential function calls**: one-command run mode chains existing modes, no orchestration framework needed

### Expected Features

**Must have (table stakes):**
- Bet status lifecycle (pending/settled) -- eliminates result=0.0 ambiguity
- Win payout settlement -- reuse build_win_payout_map() from engine.py
- Explicit loss recording -- both wins and losses get status settled
- Shared feature builder -- eliminates 3 missing feature modules in PT (~190 lines of duplicated code)
- MLflow run ID + train period in prediction records -- traceability
- Strategy manifest/PFP for PT -- parameter immutability matching BT
- betting_target + betting_mode + regime passthrough -- alignment with BT CLI flags
- OddsBandFilter in PT -- same candidate filtering as BT
- Idempotent reconciliation -- dedup key (race_id, umaban, bet_type), skip settled rows

**Should have (competitive):**
- One-command run mode (--mode run) -- eliminates operator error from 4-command sequence
- Weekly aggregation reports -- catch degradation earlier than monthly
- Per-target ROI breakdown -- separate win/place/wide tracking
- Data cutoff validation -- prevent future information leak
- Model/manifest identity in reports -- full provenance in HTML output

**Defer (post-v2.4):**
- Pipeline consistency contract verification (run both BT and PT on held-out race, compare element-wise)
- Data cutoff audit log (timestamp/row-count logging per Parquet source)
- DD controller state persistence (needed only when Kelly mode spans multiple days)

### Architecture Approach

The architecture follows one core principle: extract shared code into functions called by both paths, rather than maintaining parallel implementations. The shared build_inference_features() is the keystone -- it replaces 4 independent copies of feature construction (BT engine + 3 locations in PT script). Settlement reuses existing build_*_payout_map() functions via import rather than reimplementation. The one-command run mode chains existing _run_* functions sequentially rather than introducing workflow orchestration.

**Major components:**
1. **build_inference_features()** (src/paper_trading/feature_builder.py) -- feature construction for both BT and PT, calls 12+ feature modules
2. **settle_bets()** (src/paper_trading/settlement.py) -- unified win/place/wide payout lookup + status lifecycle, imports existing payout map builders
3. **PaperTradingConsistency** (src/paper_trading/consistency.py) -- data cutoff validation, MLflow identity tracking, manifest verification
4. **PaperTradingOrchestrator** (src/paper_trading/orchestrator.py) -- one-command run mode: verify -> predict -> wait -> reconcile -> report

### Critical Pitfalls

1. **Feature divergence (PROVEN)** -- PT missing DamPedigreeFeatures, RecordFeatures, MiningFeatures that BT includes. Prevention: shared build_inference_features() with column count assertion.
2. **result=0.0 ambiguity (PROVEN)** -- pending and lost are indistinguishable, overstating ROI by up to 5x. Prevention: explicit status column, settle ALL bets (wins and losses).
3. **Win settlement missing in PT (PROVEN)** -- reconcile only looks up place payouts, never win payouts. Prevention: import build_win_payout_map() from backtest.engine.
4. **Regime mismatch (PROVEN)** -- BT hardcodes AGGRESSIVE, PT uses dynamic detection. Prevention: hardcode AGGRESSIVE in PT to match BT.
5. **Shared builder extraction breaks BT** -- extraction from prepare_data() may introduce silent regression. Prevention: full BT before/after comparison with element-wise bet_history equality check.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Settlement Integrity
**Rationale:** Without correct settlement, all ROI numbers are meaningless. This is the critical path -- every downstream feature depends on correct bet resolution.
**Delivers:** Trustworthy ROI measurement for all bet types (win/place/wide)
**Addresses:** Features 1 (status lifecycle), 2 (win payout), 3 (loss recording), 11 (idempotent reconciliation)
**Avoids:** Pitfalls 2 (result=0.0 ambiguity), 3 (win settlement missing), 11 (wide kumi parsing)

### Phase 2: Shared Feature Builder
**Rationale:** Feature parity between BT and PT is the second pillar of trustworthy ROI. The shared builder eliminates 7 code gaps simultaneously and prevents future drift.
**Delivers:** Identical feature construction for both BT and PT paths
**Addresses:** Feature 4 (shared builder)
**Avoids:** Pitfalls 1 (feature divergence), 5 (extraction breaks BT), 13 (BloodlineFeatures asymmetry)
**Risk:** HIGH -- requires BT regression test before and after extraction

### Phase 3: Pipeline Consistency + Strategy Alignment
**Rationale:** With correct settlement and identical features, the remaining gaps are parameter alignment. PT must execute the same strategy BT validated.
**Delivers:** End-to-end BT-PT alignment (manifest, regime, OddsBandFilter, betting mode)
**Addresses:** Features 5 (MLflow identity), 6 (manifest/PFP), 7 (betting_target), 8 (betting_mode), 9 (regime), 10 (OddsBandFilter)
**Avoids:** Pitfalls 4 (regime mismatch), 6 (OddsBandFilter calibration), 12 (PFP verify failure)
**Uses:** ParameterFreezeProtocol, strategy manifest JSON, ModelLoader

### Phase 4: Automation + Reporting
**Rationale:** With a reliable, aligned pipeline, automate the operator workflow and expand reporting.
**Delivers:** One-command race-day execution, weekly stats, per-target breakdown
**Addresses:** Features 13 (one-command run), 15 (weekly reports), 16 (per-target aggregation), 17 (model identity in reports)
**Avoids:** Pitfalls 7 (crash data loss via atomic write), 14 (loss tracking in summary), 15 (bet_type-aware display)
**Uses:** RaceWatcher, SafetyGuard, RaceScheduler (existing, no changes)

### Phase Ordering Rationale

- Phase 1 first because it makes ROI measurement valid. Without it, Phases 2-4 produce numbers that cannot be trusted.
- Phase 2 second because feature parity is the largest single source of BT-PT divergence (3 missing modules). It also eliminates the maintenance burden of 4 independent feature construction copies.
- Phase 3 depends on Phase 2 (shared builder enables manifest verification in PT) and Phase 1 (status column used by settle_bets).
- Phase 4 depends on all prior phases. The one-command run mode chains components that must already be correct.
- The dependency graph is strictly linear: each phase builds on the previous one. No parallel phase execution is safe.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Complex extraction from 300+ line prepare_data() method. Needs careful analysis of interwoven data loading vs feature construction sections. The BT regression test strategy needs definition.
- **Phase 3 (OddsBandFilter calibration):** Current calibration requires training-period bet history. Saving calibration as a model artifact requires changes to run_train.py. The integration point between training output and PT input needs specification.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Settlement logic is a straightforward import of existing functions. Status lifecycle is a simple string column. Well-understood patterns.
- **Phase 4:** One-command run mode chains existing functions. Weekly aggregation uses standard pandas groupby. Standard patterns throughout.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new dependencies. All components verified in installed environment. Direct codebase analysis of every integration point. |
| Features | HIGH | All 7 gaps identified by direct source code comparison. Feature dependency graph verified against codebase. |
| Architecture | HIGH | Component boundaries derived from existing code structure. All reuse targets verified to exist and work. |
| Pitfalls | HIGH | 6 critical pitfalls all proven by direct codebase analysis. Historical project lessons (v1.6, v1.8) confirm recurrence pattern. |

**Overall confidence:** HIGH

### Gaps to Address

- **OddsBandFilter calibration artifact format:** Need to define the serialization format for calibration data produced by run_train.py and consumed by PT. Deciding during Phase 3 planning.
- **BloodlineFeatures presence in BT:** Need to verify whether blood_* features appear in any model FEATURE_COLS before deciding whether to include them in the shared builder. Checking during Phase 2.
- **Pre-v2.4 record migration strategy:** Existing PT records were generated with incomplete features. Need to decide whether to mark them with schema_version or start fresh. Deciding during Phase 1.

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis: scripts/run_paper_trading.py (1384 lines) -- three duplicated feature construction copies, reconcile logic
- Direct codebase analysis: src/backtest/engine.py (2392 lines) -- canonical feature construction, settlement functions
- Direct codebase analysis: src/paper_trading/reconciler.py (153 lines) -- place-only settlement gaps
- Direct codebase analysis: src/backtest/parameter_freeze_protocol.py (187 lines) -- PFP patterns
- Direct codebase analysis: src/backtest/race_predictor.py (1645 lines) -- shared inference pipeline
- Direct codebase analysis: src/automation/ (3 files) -- scheduler, watcher, safety guard
- Installed versions verified via pip list on 2026-06-06

### Secondary (MEDIUM confidence)
- Parquet schema evolution documentation -- confirmed additive-only pattern works for bet records
- Idempotent pipeline patterns (Prefect blog) -- confirmed atomic write + dedup pattern

### Tertiary (LOW confidence)
- Rejected library evaluations (tenacity, schedule, Delta Lake) -- assessed as unnecessary, no validation needed

---
*Research completed: 2026-06-06*
*Ready for roadmap: yes*
