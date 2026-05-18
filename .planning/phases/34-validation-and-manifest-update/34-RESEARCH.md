# Phase 34: Validation and Manifest Update - Research

**Researched:** 2026-05-18
**Domain:** ML validation pipeline (backtest, IC evaluation, GPD diagnostics, manifest freezing)
**Confidence:** HIGH

## Summary

Phase 34 is the final validation step for milestone v1.7 (Market-Independent Edge Discovery). It validates the new features added in Phases 31-32 (6 race-level + 5 market-cross features) through a prescribed sequence: POST_RACE leakage test, single-year backtest, IC evaluation, GPD diagnostics, and manifest freeze. All required scripts and modules exist and are production-ready.

The key finding is that rl_* features (RLF-01~06) are computed by FeatureEngine but are NOT registered in any model's FEATURE_COLS -- only MCF features (rl_favorite_in_wide_top1, etc.) are registered in all 12 models. This means rl_* features exist in the training data but are not used by LightGBM models. The manifest freeze will capture the current state accurately.

The existing backtest models (data/models-backtest/2024/) were trained on 2026-05-17 BEFORE Phase 31/32 completed, so a full retrain is required. The OOF predictions file (data/oof/oof_predictions.parquet) is currently empty (0 rows), confirming no v1.7 training has occurred yet.

**Primary recommendation:** Execute validation steps in the exact order prescribed by D-12 (leakage -> BT -> IC -> GPD -> manifest), using the existing scripts with no code changes needed for the first 4 steps. Only manifest freeze requires a version string update from "v1.6" to "v1.7".

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: Target year 2024 only (single-year BT)
- D-02: No strategy manifest (default parameters)
- D-03: Betting mode flat (100 yen fixed)
- D-04: --calibration-bt skipped
- D-05: v1.6 baseline IC comparison SKIPPED (OOF predictions not saved at v1.6 time)
- D-06: v1.7 IC values recorded as NEW baseline for future comparison
- D-07: IC evaluation executed AFTER BT (uses OOF predictions saved during BT training)
- D-08: GPD report executed, Claude judges MDR > 0 and FAD <= 5 as success
- D-09: New features (race-level 6 + market-cross 5) should function as Market category at shallow depth
- D-10: If validation fails goals, record results as-is and freeze manifest anyway
- D-11: Improvements deferred to v1.8
- D-12: Execution order: leakage test -> BT 2024 -> IC eval -> GPD -> manifest freeze

### Claude's Discretion
- BT result ROI interpretation vs v1.6 (85.7%)
- IC value "good/bad" judgment (no baseline, Claude decides)
- GPD result PASS/WARN judgment based on MDR/FAD values
- Error handling between validation steps (e.g., skip IC if BT fails)
- Validation result report format (JSON, console summary, etc.)

### Deferred Ideas (OUT OF SCOPE)
- Optuna strategy parameter optimization (future phase)
- v1.6 baseline IC value acquisition (impossible -- OOF predictions not saved)
- Additional BT for 2023/2025 years
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VAL-01 | Multi-year backtest after new feature addition (ROI measurement) | run_backtest.py --years 2024 --train-window 4 --ensemble --betting-mode flat --betting-target win. Expected duration: ~41 min. Outputs: data/backtest/multi_year_result.json, data/validation/multi_year_validation_report.json |
| VAL-02 | Residual IC improvement confirmation (C-orthogonal IC vs baseline) | ic_evaluator.py computes 4 formulations (B-diff/C-orth/E-incr/Per-race) x 3 surfaces (turf/dirt/all). Input: data/oof/oof_predictions.parquet (saved by BT training). Output: data/baseline/ic_baseline.json |
| VAL-03 | Gain per Depth confirmation of new features at depth 3-5 | gpd_diagnostics.py + run_gpd.py --models-dir data/models-backtest/2024 --ensemble. Outputs JSON + PNG charts. Success criteria: MDR > 0 and FAD <= 5 |
| VAL-04 | FEATURE_COLS manifest freeze + SHA256 hash update | freeze_feature_manifest.py reads 12 model FEATURE_COLS, generates SHA256 hashes. REQUIRES version string update from "v1.6" to "v1.7" in script |
| VAL-05 | POST_RACE information leakage test re-run for new features | pytest tests/test_post_race_leakage.py -v (11 test cases including TestRaceLevelFeatures and TestMarketCrossFeatures) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| POST_RACE leakage test | Test suite | -- | Pure pytest execution, validates feature safety |
| Backtest execution | Pipeline + ML | Data layer | TrainingPipeline trains models, BacktestEngine runs simulation |
| IC evaluation | ML diagnostics | -- | ic_evaluator.py processes OOF predictions |
| GPD diagnostics | ML diagnostics | Visualization | gpd_diagnostics.py analyzes tree structures |
| Manifest freeze | Configuration | -- | freeze_feature_manifest.py reads model metadata |

## Standard Stack

### Core (all pre-existing, no installation needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lightgbm | 4.6.0 | ML model ( Booster ) | Project core ML framework [VERIFIED: runtime check] |
| scikit-learn | 1.8.0 | CalibratedClassifierCV, KFold, IsotonicRegression | Project standard [VERIFIED: runtime check] |
| scipy | 1.17.1 | spearmanr (IC evaluation) | Statistical computation [VERIFIED: runtime check] |
| pandas | (project) | DataFrame operations | Data manipulation standard |
| numpy | (project) | Numerical computation | Array operations standard |
| pytest | 9.0.2 | Test framework | Project test standard [VERIFIED: runtime check] |
| mlflow | 3.10.1 | Experiment tracking | Optional logging for IC eval [VERIFIED: runtime check] |
| matplotlib | 3.10.8 | GPD chart generation | Visualization [VERIFIED: runtime check] |

### No new packages needed

This phase uses exclusively existing scripts and dependencies. Zero package installation required.

**Installation:**
```bash
# No installation needed -- all dependencies already present
```

## Package Legitimacy Audit

> No new packages installed in this phase. All dependencies pre-existing from prior phases.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| (none) | -- | -- | -- | -- | -- | N/A -- phase uses existing packages only |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
POST_RACE Leakage Test (VAL-05)
    |
    v
[pytest tests/test_post_race_leakage.py]  -- 11 test cases
    |                                         Layer 1: build_all() output
    |                                         Layer 2: FEATURE_COLS whitelist
    |                                         Layer 3: predict() propagation
    v
PASS? --> Backtest (VAL-01)
            |
            +--> [TrainingPipeline.run()]  -- trains all models with new features
            |        |
            |        +--> saves OOF predictions to data/oof/oof_predictions.parquet
            |        +--> saves models to data/models-backtest/2024/
            |        +--> outputs BT result JSON
            |
            v
        IC Evaluation (VAL-02)
            |
            +--> [ic_evaluator.py] reads OOF predictions
            |        |
            |        +--> computes 4 IC formulations x 3 surfaces
            |        +--> outputs data/baseline/ic_baseline.json
            |
            v
        GPD Diagnostic (VAL-03)
            |
            +--> [gpd_diagnostics.py] reads trained models
            |        |
            |        +--> analyzes depth-by-category gain
            |        +--> outputs data/gpd/gpd_report.json + PNG charts
            |
            v
        Manifest Freeze (VAL-04)
            |
            +--> [freeze_feature_manifest.py] reads 12 model FEATURE_COLS
                     |
                     +--> generates SHA256 hashes
                     +--> outputs data/feature_freeze_manifest.json (v1.7)
```

### Recommended Project Structure
```
scripts/
  run_backtest.py          -- VAL-01: backtest execution
  run_ic_eval.py           -- VAL-02: IC evaluation
  run_gpd.py               -- VAL-03: GPD diagnostics
  freeze_feature_manifest.py -- VAL-04: manifest freeze

src/models/
  ic_evaluator.py          -- IC evaluation module
  gpd_diagnostics.py       -- GPD diagnostic module

tests/
  test_post_race_leakage.py -- VAL-05: leakage tests (11 cases)

data/
  oof/oof_predictions.parquet      -- saved by BT training
  baseline/ic_baseline.json        -- IC eval output
  gpd/gpd_report.json              -- GPD diagnostic output
  feature_freeze_manifest.json     -- manifest output
  models-backtest/2024/            -- trained models
  backtest/multi_year_result.json  -- BT results
```

### Pattern 1: Validation Script Execution
**What:** Run existing CLI scripts in sequence, capture output
**When to use:** Each validation step (VAL-01 through VAL-05)
**Example:**
```bash
# Step 1: Leakage tests (fastest, early detection)
python -m pytest tests/test_post_race_leakage.py -v

# Step 2: Backtest (~41 min)
python scripts/run_backtest.py --years 2024 --train-window 4 --ensemble --betting-mode flat --betting-target win

# Step 3: IC evaluation (from BT-saved OOF)
python scripts/run_ic_eval.py data/oof/oof_predictions.parquet --output data/baseline/ic_baseline.json

# Step 4: GPD diagnostics (from trained models)
python scripts/run_gpd.py --models-dir data/models-backtest/2024 --ensemble --output-dir data/gpd

# Step 5: Manifest freeze
python scripts/freeze_feature_manifest.py
```

### Pattern 2: OOF Data Flow
**What:** TrainingPipeline saves full feature DataFrame to Parquet during training
**When to use:** IC evaluation needs OOF predictions
**Key code path (training_pipeline.py lines 251-257):**
```python
# 3c. OOF predictions Parquet save (IC evaluation, Phase 30)
oof_path = Path("data/oof/oof_predictions.parquet")
oof_path.parent.mkdir(parents=True, exist_ok=True)
full_features_df.to_parquet(oof_path, index=False)
```
The OOF file is overwritten during each BT training run. The ic_evaluator reads columns: `p_win_corrected` (or `p_win_pred`), `tanodds`/`implied_prob`, `kakuteijyuni`, `surface`, `race_id`.

### Anti-Patterns to Avoid
- **Running BT without --ensemble:** CONTEXT D-02 specifies no manifest, but --ensemble is required for StackedEnsemble (project convention)
- **Running IC eval before BT completes:** OOF file is currently empty (0 rows). Must wait for BT training to populate it
- **Running GPD on stale models:** data/models-backtest/2024/ was trained 2026-05-17 before Phase 31/32. Must use models from the new BT run
- **Forgetting to update manifest version:** freeze_feature_manifest.py hardcodes "v1.6" -- must update to "v1.7" before running

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ROI measurement | Custom backtest loop | run_backtest.py | Full pipeline integration, OOF save, validation report |
| IC computation | Manual Spearman correlation | ic_evaluator.py | 4 formulations + 3 surfaces + direction consistency |
| Depth analysis | Custom tree parsing | gpd_diagnostics.py | trees_to_dataframe() + MDR/FAD metrics |
| Manifest generation | Manual JSON assembly | freeze_feature_manifest.py | Deterministic SHA256, 12 model coverage |
| Leakage detection | Manual column audit | test_post_race_leakage.py | 3-layer verification (build_all, FEATURE_COLS, predict) |

**Key insight:** All 5 validation requirements have turnkey implementations. No new code to write except the version string update in freeze_feature_manifest.py.

## Runtime State Inventory

> Not applicable -- this is not a rename/refactor/migration phase.

## Common Pitfalls

### Pitfall 1: Stale Models in GPD Analysis
**What goes wrong:** Running GPD diagnostics on pre-Phase-31/32 models (data/models-backtest/2024/ from 2026-05-17)
**Why it happens:** GPD reads from data/models-backtest/ -- if BT is not run first, old models are analyzed
**How to avoid:** Always run BT (VAL-01) before GPD (VAL-03). D-12 execution order enforces this
**Warning signs:** GPD report shows no new features (rl_*, mcf_*) in the analysis

### Pitfall 2: Empty OOF Predictions
**What goes wrong:** IC evaluation fails because data/oof/oof_predictions.parquet has 0 rows
**Why it happens:** OOF file is currently empty. It gets populated during TrainingPipeline.run() -> BT training
**How to avoid:** Run BT first (VAL-01), then IC eval (VAL-02). D-12 enforces this
**Warning signs:** IC eval raises "insufficient_samples" warning or ValueError on missing columns

### Pitfall 3: Manifest Version Not Updated
**What goes wrong:** freeze_feature_manifest.py outputs version "v1.6" instead of "v1.7"
**Why it happens:** Version string is hardcoded at line 77 of freeze_feature_manifest.py
**How to avoid:** Update version string from "v1.6" to "v1.7" before running manifest freeze
**Warning signs:** Manifest JSON shows "version": "v1.6" instead of "v1.7"

### Pitfall 4: rl_* Features Not in FEATURE_COLS
**What goes wrong:** Expecting 6 rl_* features to appear in model FEATURE_COLS and GPD analysis
**Why it happens:** rl_* features (RLF-01~06) are computed by FeatureEngine.build_all() but were never added to any model's FEATURE_COLS during Phase 31. Only MCF features (rl_favorite_in_wide_top1, etc.) were added to all 12 models
**How to avoid:** Document this as expected behavior. rl_* features exist in training data but are NOT selected by models. They may be considered for FEATURE_COLS promotion in a future phase
**Warning signs:** GPD analysis shows 5 new MCF features but 0 new rl_* features. Manifest shows same feature counts as v1.6 for rl_* features

### Pitfall 5: v1.6 vs v1.7 ROI Comparison Mismatch
**What goes wrong:** Directly comparing 2024-only ROI (v1.7) against 3-year average ROI (v1.6: 85.7%)
**Why it happens:** v1.6 ROI 85.7% is the average of 2023/2024/2025. 2024 alone may have different ROI
**How to avoid:** CONTEXT D-10 says record results as-is. Comparison is informational only, not a gate
**Warning signs:** ROI appears much higher or lower than 85.7% -- may just be 2024-specific performance

### Pitfall 6: BT Duration Underestimation
**What goes wrong:** Planning assumes ~41 min but actual runtime is longer
**Why it happens:** First-time training with new features (Phase 31/32) may trigger additional computation paths (wide odds merge, Harville calculation)
**How to avoid:** Budget 60 min for BT instead of 41 min
**Warning signs:** BT takes more than 45 min

## Code Examples

### VAL-01: Backtest Command
```bash
# Source: CONTEXT D-01/D-02/D-03/D-04 + CLAUDE.md run_backtest.py docs
python scripts/run_backtest.py \
  --years 2024 --train-window 4 \
  --ensemble --betting-mode flat --betting-target win
```
- Train period: 2020-01-01 ~ 2023-12-31
- Test period: 2024-01-01 ~ 2024-12-31
- Outputs: data/backtest/multi_year_result.json, data/validation/multi_year_validation_report.json, data/oof/oof_predictions.parquet, data/models-backtest/2024/

### VAL-02: IC Evaluation Command
```bash
# Source: CONTEXT canonical_refs + scripts/run_ic_eval.py
python scripts/run_ic_eval.py data/oof/oof_predictions.parquet \
  --output data/baseline/ic_baseline.json
```
- Requires: data/oof/oof_predictions.parquet (populated by VAL-01)
- Columns needed: p_win_corrected (or p_win_pred), tanodds/implied_prob, kakuteijyuni, surface, race_id
- Output: 4 IC formulations x 3 surfaces + direction consistency check

### VAL-03: GPD Diagnostic Command
```bash
# Source: scripts/run_gpd.py
python scripts/run_gpd.py \
  --models-dir data/models-backtest/2024 \
  --ensemble \
  --output-dir data/gpd
```
- Requires: trained models in data/models-backtest/2024/ (from VAL-01)
- Output: data/gpd/gpd_report.json + per-model PNG charts
- Success criteria (D-08): MDR > 0 and FAD <= 5

### VAL-04: Manifest Freeze (requires version update)
```python
# Source: scripts/freeze_feature_manifest.py line 77
# BEFORE running, update version string:
"version": "v1.7",  # was "v1.6"
```
```bash
python scripts/freeze_feature_manifest.py
```
- Output: data/feature_freeze_manifest.json
- Contains: 12 models x (feature_count, features list, sha256) + overall_sha256

### VAL-05: Leakage Test Command
```bash
python -m pytest tests/test_post_race_leakage.py -v
```
- 11 test cases across 3 test classes
- TestPostRaceLeakage (4 tests): build_all output, model FEATURE_COLS, EV correction odds, conformal whitelist
- TestRaceLevelFeatures (3 tests): rl_* source code audit, RL_COLS vs POST_RACE, build_all produces rl_*
- TestMarketCrossFeatures (4 tests): MCF source code audit, MCF_COLS vs POST_RACE, build_all produces MCF, all 12 models have MCF

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No IC baseline recording | OOF-based IC evaluation with JSON persistence | Phase 30 (v1.7) | IC values now reproducible and comparable |
| No depth-based feature analysis | GPD diagnostics with MDR/FAD metrics | Phase 33 (v1.7) | Can validate Two-Stage hypothesis quantitatively |
| Feature manifest v1.6 (without MCF features) | Manifest v1.7 (with 5 MCF features in all 12 models) | Phase 34 | SHA256 changes for all 12 models |

**Deprecated/outdated:**
- data/oof/oof_predictions.parquet (currently empty): Will be overwritten by BT training
- data/models-backtest/2024/ (pre-Phase 31/32): Will be overwritten by new BT training
- data/feature_freeze_manifest.json (version "v1.6"): Will be updated to "v1.7"

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | rl_* features (RLF-01~06) intentionally not in model FEATURE_COLS -- only computed in build_all()/build_features() but not used by models | Architecture Patterns / Pitfall 4 | If wrong, rl_* should be added to FEATURE_COLS before BT, otherwise models cannot use them |
| A2 | OOF predictions file will contain required columns (p_win_corrected, tanodds, kakuteijyuni, surface, race_id) after BT training | VAL-02 | IC evaluation will fail with ValueError |
| A3 | v1.7 manifest version should be "v1.7" (matching milestone name) | VAL-04 | Incorrect version tracking |
| A4 | BT execution time ~41-60 min (single year, manifestなし) | VAL-01 | Phase planning underestimated if longer |
| A5 | PostgreSQL is accessible at localhost:5432/everydb2 for Parquet data loading | Environment | BT will fail with FileNotFoundError on Parquet store |

## Open Questions

1. **rl_* features in FEATURE_COLS**
   - What we know: 6 rl_* features (RLF-01~06) are computed by build_all()/build_features() but are NOT in any model's FEATURE_COLS. 5 MCF features are in all 12 models' FEATURE_COLS.
   - What's unclear: Was this intentional (Phase 31 only implemented computation + leakage tests, with FEATURE_COLS registration deferred) or an oversight?
   - Recommendation: Run BT as-is. If rl_* features should be in models, that requires a code change (adding to FEATURE_COLS) and re-running BT. Per D-10, record results as-is. This can be addressed in v1.8 if needed.

2. **IC baseline comparison semantics**
   - What we know: D-05 says v1.6 baseline IC comparison is skipped. D-06 says v1.7 IC values become the new baseline.
   - What's unclear: What JSON format should the new baseline use? The existing data/baseline/ic_baseline.json may have stale content.
   - Recommendation: ic_evaluator.py overwrites the file. The new content will serve as baseline automatically.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | All scripts | Y | 3.11.15 | -- |
| PostgreSQL (via Parquet) | BT training | Y (via Parquet files) | -- | -- |
| lightgbm | BT + GPD | Y | 4.6.0 | -- |
| scipy | IC evaluation | Y | 1.17.1 | -- |
| scikit-learn | BT training | Y | 1.8.0 | -- |
| mlflow | IC eval (optional) | Y | 3.10.1 | --mlflow flag not required |
| matplotlib | GPD charts | Y | 3.10.8 | -- |
| pytest | Leakage tests | Y | 9.0.2 | -- |

**Missing dependencies with no fallback:**
- None identified -- all dependencies available

**Missing dependencies with fallback:**
- None needed

## Validation Architecture

> nyquist_validation is explicitly false in .planning/config.json. This section is SKIPPED per instructions.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes (minimal) | Parquet schema validation in ParquetStore |
| V8 Data Protection | yes | POST_RACE leakage test (VAL-05) ensures no post-race data in features |

### Known Threat Patterns for ML Validation

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Data leakage (POST_RACE info) | Information Disclosure | 3-layer leakage test (VAL-05) |
| Feature count mismatch | Tampering | SHA256 manifest freeze (VAL-04) |
| Stale model analysis | Repudiation | BT timestamp verification in meta.json |

## Sources

### Primary (HIGH confidence)
- Source code inspection of all 5 canonical scripts (run_backtest.py, run_ic_eval.py, run_gpd.py, freeze_feature_manifest.py, test_post_race_leakage.py)
- Source code inspection of ic_evaluator.py, gpd_diagnostics.py, training_pipeline.py
- CONTEXT.md decisions verified against actual code behavior
- Runtime verification: Python 3.11.15, lightgbm 4.6.0, scipy 1.17.1, pytest 9.0.2

### Secondary (MEDIUM confidence)
- Feature manifest comparison (v1.6 vs current FEATURE_COLS)
- OOF file state verification (empty, 0 rows)
- Model directory timestamps (data/models-backtest/2024/ trained 2026-05-17)

### Tertiary (LOW confidence)
- rl_* feature exclusion from FEATURE_COLS being intentional (A1) -- inferred from Phase 31/32 CONTEXT and test coverage

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all dependencies verified at runtime, no new packages needed
- Architecture: HIGH -- all scripts exist, tested code paths confirmed by reading source
- Pitfalls: HIGH -- identified 6 concrete pitfalls with verification from code inspection

**Research date:** 2026-05-18
**Valid until:** 30 days (stable project, no external API dependencies)
