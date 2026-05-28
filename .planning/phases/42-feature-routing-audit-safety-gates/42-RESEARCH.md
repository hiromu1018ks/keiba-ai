# Phase 42: Feature Routing Audit & Safety Gates - Research

**Researched:** 2026-05-28
**Domain:** Feature leak detection, OOF artifact validation, deployment gate evaluation
**Confidence:** HIGH

## Summary

This phase creates three independent safety mechanisms for the v2.1 milestone. First, a feature routing audit (SAF-01) verifies that MarketAwareWinCalibrator's ~51 features and RaceLevelRanker's ~29 features never leak into MarketModel or RaceQualityScreener. Second, OOF health validation (SAF-02) extends the existing OOFHealthValidator with Phase 39/40-specific artifact profiles. Third, a DeploymentGateEvaluator (SAF-03) reads shadow_comparison_result.json and shadow_manifest.json to produce a PASS/FAIL/WARN report based on probability quality, bet count preservation, artifact reproducibility, and diagnostic checks.

The core technical risk is the v1.8 ROI collapse (87.8% down from 97.8%) caused by feature routing contamination. The audit registry pattern (single source of truth, diff-tested against actual model FEATURE_COLS) prevents recurrence. All three components are new files with no modifications to existing model classes.

**Primary recommendation:** Build the audit registry first (SAF-01), then OOF profiles (SAF-02), then the gate evaluator (SAF-03). Each is independently testable with mock data. No external dependencies required.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- SAF-01: Both unit tests (fail-fast) + audit script (JSON/Markdown report). Safety guarantee via tests, audit script for review.
- SAF-02: Two-layer: CI mock-based validation + manual/nightly audit command for full E2E.
- SAF-03: Independent DeploymentGateEvaluator that outputs report only, does not auto-change deployment status.
- Audit registry in `src/audit/feature_routing_registry.py` as single source of truth.
- Required audit targets (fail-fast) = MarketModel + RaceQualityScreener.
- Advisory audit targets (warning) = other models with FEATURE_COLS.
- GatePolicy frozen dataclass with specific conditions (Brier/logloss/ECE <= baseline, bet count >= 95%, etc.).
- OOF artifact profiles as plugin-like objects registered with OOFHealthValidator.
- MAWC profile checks: NaN/inf, [0,1] range, sum-to-1.0 per race_id, p_win_pred forbidden, fold column required.
- Ranker profile checks: NaN/inf in investment_score/component scores, race-level rank determinism, fold column required.

### Claude's Discretion
- Exact feature list content in audit registry (extracted from Phase 39/40 model definitions).
- OOF artifact profile implementation method (how to integrate with OOFHealthValidator).
- DeploymentGateEvaluator internal methods and data flow design.
- Test structure/naming (follow existing conventions).
- GatePolicy dataclass field design.

### Deferred Ideas (OUT OF SCOPE)
- Auto-deploy decision (DEP-01, v2.2+).
- Auto shadow_only on FAIL (v2.2+).
- MarketAwareWinCalibrator/RaceLevelRanker implementation changes.
- New models/features.
- ROI optimization.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SAF-01 | Feature routing audit confirms calibrator features do not pollute MarketModel/RaceQualityScreener | Audit registry with FORBIDDEN_CALIBRATOR_FEATURES (51 features from MAWC.build_feature_matrix) and FORBIDDEN_RANKER_FEATURES (29 features from RLR.RELEVANCE_FEATURES + VALUE_FEATURES + DERIVED_VALUE_FEATURES). Diff tests verify registry matches actual model definitions. |
| SAF-02 | OOFHealthValidator anomaly-free for all components | OOFHealthValidator already supports OOF-01~08 checks via OOFHealthProfile. New CalibratorArtifactProfile and RankerArtifactProfile extend with MAWC/ranker-specific checks (NaN/inf, range, sum-to-1, p_win_pred guard, fold required). |
| SAF-03 | New calibrator/ranker do not replace baseline until probability quality gate + bet count preservation + artifact reproducibility + diagnostics all pass | DeploymentGateEvaluator reads shadow_comparison_result.json (per-fold metrics, overall metrics, manifest SHA256) and applies GatePolicy conditions. Reports PASS/FAIL/WARN per condition with overall verdict. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Feature routing audit (SAF-01) | Python code (src/audit/) | CI (pytest) | Pure code introspection -- reads FEATURE_COLS class attributes, compares sets. No DB, no network. |
| OOF artifact validation (SAF-02) | Python code (src/validation/) | CI (pytest) + manual CLI | OOFHealthValidator already exists. New profiles are plugin-like additions. |
| Deployment gate evaluation (SAF-03) | Python code (src/backtest/) | CLI script | Reads JSON artifacts from Phase 41 shadow comparison. Pure computation, no external dependencies. |
| Audit report generation | CLI script (scripts/) | -- | Standalone script reads registry, introspects models, outputs JSON + Markdown. |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | [installed] | Test framework | Project standard per CLAUDE.md |
| pandas | [installed] | DataFrame manipulation | Project standard |
| numpy | [installed] | Numerical operations | Project standard |
| dataclasses (stdlib) | Python 3.11 | Frozen dataclasses for registry/gate policy | Project convention (Phase 38 InvestmentFeatureSpec) |
| json (stdlib) | Python 3.11 | JSON read/write for reports | Existing pattern in shadow_comparison.py |
| hashlib (stdlib) | Python 3.11 | SHA256 for artifact verification | Existing pattern in oof_health_validator.py |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| joblib | [installed] | Model loading for FEATURE_COLS introspection | Audit script loading models |
| unittest.mock | [installed] | Test mocking | All tests per CLAUDE.md convention |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Frozen dataclass for GatePolicy | Pydantic BaseModel | dataclasses is stdlib, project convention. Pydantic not in dependencies. |
| Separate CalibratorArtifactProfile class | Extend OOFHealthValidator directly | Plugin pattern keeps OOFHealthValidator generic (D-06 decision). |

**Installation:**
```bash
# No new dependencies required -- all stdlib or already installed
pip install -e ".[dev]"  # project standard
```

**Version verification:** All dependencies already installed in project environment.

## Package Legitimacy Audit

> This phase installs zero new external packages. All code uses stdlib + existing project dependencies.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| (none) | — | — | — | — | — | N/A |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*No new packages installed in this phase.*

## Architecture Patterns

### System Architecture Diagram

```
[MarketAwareWinCalibrator.FEATURE_COLS (51)]
[MarketLevelRanker.RELEVANCE_FEATURES (15)]
[MarketLevelRanker.VALUE_FEATURES (13)]
[MarketLevelRanker.DERIVED_VALUE_FEATURES (2)]
         |
         v
+------------------------------------------+
|    src/audit/feature_routing_registry.py  |  <-- Single source of truth
|  FORBIDDEN_CALIBRATOR_FEATURES (51)       |
|  FORBIDDEN_RANKER_FEATURES (29)           |
|  CRITICAL_TARGET_MODELS (2 models)        |
|  ADVISORY_TARGET_MODELS (5+ models)       |
+------------------------------------------+
         |                        |
         v                        v
+------------------+    +-----------------------+
| tests/test_      |    | scripts/run_feature_  |
| feature_routing_ |    | routing_audit.py      |
| audit.py         |    | (JSON + Markdown)     |
| (fail-fast)      |    +-----------------------+
+------------------+

[shadow_comparison_result.json]
[shadow_manifest.json]
         |
         v
+------------------------------------------+
| src/backtest/deployment_gates.py           |
| DeploymentGateEvaluator                   |
|   + GatePolicy (frozen dataclass)         |
|   + GateEvaluationResult                  |
+------------------------------------------+
         |
         v
   [gate_evaluation.json]
   [gate_evaluation.md]

[OOF DataFrame (MAWC/ranker)]
         |
         v
+------------------------------------------+
| src/validation/oof_health_validator.py     |
|   + CalibratorArtifactProfile (new)        |
|   + RankerArtifactProfile (new)            |
+------------------------------------------+
```

### Recommended Project Structure

```
src/
  audit/
    __init__.py                        # NEW
    feature_routing_registry.py        # NEW: audit registry
  validation/
    oof_health_validator.py            # EXTEND: add MAWC/ranker profiles
    artifact_profiles.py               # NEW: CalibratorArtifactProfile, RankerArtifactProfile
  backtest/
    deployment_gates.py                # NEW: DeploymentGateEvaluator + GatePolicy
scripts/
  run_feature_routing_audit.py         # NEW: CLI audit script
tests/
  test_feature_routing_audit.py        # NEW: SAF-01 tests
  test_artifact_profiles.py            # NEW: SAF-02 profile tests
  test_deployment_gates.py             # NEW: SAF-03 tests
```

### Pattern 1: Audit Registry as Single Source of Truth

**What:** Define forbidden feature sets and target models in one file. Both unit tests and audit script import from this file.

**When to use:** Any feature leak detection across model boundaries.

**Example:**
```python
# src/audit/feature_routing_registry.py
from dataclasses import dataclass

@dataclass(frozen=True)
class AuditTarget:
    model_class_name: str
    model_module: str
    feature_cols_attr: str  # "FEATURE_COLS", "RELEVANCE_FEATURES", etc.

FORBIDDEN_CALIBRATOR_FEATURES: frozenset[str] = frozenset({
    # 6 main effects
    "logit_model", "logit_market", "log_odds",
    "popularity_rank_pct", "p_win_race_rank_pct", "field_size",
    # 7 odds band one-hot
    "1-2", "2-3", "3-5", "5-10", "10-30", "30-100", "100+",
    # 5 pop bucket one-hot
    "pop_1", "pop_2_3", "pop_4_6", "pop_7_9", "pop_10_plus",
    # 3 p_rank one-hot
    "top_25", "mid_25_75", "bottom_25",
    # 15 logit_model x segment interactions
    "logit_model_x_1-2", "logit_model_x_2-3", ...,
    # 15 logit_market x segment interactions
    "logit_market_x_1-2", "logit_market_x_2-3", ...,
})

CRITICAL_TARGET_MODELS: tuple[AuditTarget, ...] = (
    AuditTarget("MarketModel", "models.market_model", "FEATURE_COLS"),
    AuditTarget("RaceQualityScreener", "models.race_quality_screener", "FEATURE_COLS"),
)
```

### Pattern 2: OOF Artifact Profile Plugin

**What:** Separate CalibratorArtifactProfile and RankerArtifactProfile from OOFHealthValidator core. Each profile defines artifact-specific checks while the validator handles common checks (fold count, race_id uniqueness, etc.).

**When to use:** Adding new OOF artifact types to existing validation infrastructure.

**Example:**
```python
# src/validation/artifact_profiles.py
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class CalibratorArtifactProfile:
    """Phase 39 MAWC artifact validation rules."""
    required_columns: tuple[str, ...] = ("race_id", "p_win_combined", "p_win_final", "fold")
    forbidden_columns: tuple[str, ...] = ("p_win_pred",)
    probability_columns: tuple[str, ...] = ("p_win_combined", "p_win_final")
    fold_col: str = "fold"

    def validate(self, df: pd.DataFrame) -> list[str]:
        failures = []
        # Check NaN/inf in probability columns
        # Check [0,1] range
        # Check sum-to-1.0 per race_id
        # Check p_win_pred not present
        return failures
```

### Pattern 3: GatePolicy Frozen Dataclass

**What:** Immutable policy defining gate thresholds. Read by DeploymentGateEvaluator, never modified at runtime.

**When to use:** Any gate/threshold evaluation that needs to be explicit and auditable.

**Example:**
```python
# src/backtest/deployment_gates.py
from dataclasses import dataclass

@dataclass(frozen=True)
class GatePolicy:
    brier_tolerance: float = 1e-6
    logloss_tolerance: float = 1e-6
    ece_tolerance: float = 1e-6
    bet_count_ratio_threshold: float = 0.95
    require_oof_pass: bool = True
    require_audit_pass: bool = True
    require_manifest_complete: bool = True

DEFAULT_GATE_POLICY = GatePolicy()
```

### Anti-Patterns to Avoid

- **Hardcoding feature lists in tests:** Tests must import from the registry, not duplicate lists. Diff tests catch stale registries.
- **Modifying OOFHealthValidator core logic:** Phase 39/40 checks go into separate profile classes (D-06). The validator remains generic.
- **Putting gate logic in RacePredictor:** Gate evaluation is independent (D-12). RacePredictor only reads deployment_status.
- **Using mutable default arguments in dataclasses:** Use `field(default_factory=...)` for list/dict defaults.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OOF schema hashing | Custom hash function | OOFHealthValidator._compute_schema_hashes() | Already exists, handles dtype-level hashing |
| Shadow comparison JSON parsing | Custom parser | json.load() + known structure from Phase 41 | Structure is documented in shadow_comparison.py |
| Probability quality metrics (Brier/logloss/ECE) | Custom implementations | numpy vectorized ops following ShadowComparisonFramework._compute_ece() pattern | ECE binning is tricky to get right |
| Feature set comparison | Manual list comparison | Python set operations (intersection, issubset) | Correct by construction, handles duplicates |

**Key insight:** All infrastructure needed (OOFHealthValidator, ShadowComparisonFramework, FEATURE_COLS pattern) already exists. This phase is primarily about connecting them with audit-specific logic, not building new foundations.

## Common Pitfalls

### Pitfall 1: MAWC features are dynamically constructed, not a class-level FEATURE_COLS

**What goes wrong:** MarketAwareWinCalibrator does NOT define `FEATURE_COLS` as a class attribute. Features are built dynamically in `build_feature_matrix()` via one-hot encoding and interaction term generation. The audit registry must enumerate the expected 51 feature names that the method produces.

**Why it happens:** MAWC uses sklearn LogisticRegression which doesn't require feature names to be declared upfront -- they're constructed from the data at build time.

**How to avoid:** Extract the exact 51 feature names from `build_feature_matrix()` code (6 main + 15 one-hot + 30 interactions). Verify via the assert at line 194: `assert len(feature_names) == 51`. The diff test should call `build_feature_matrix()` on a minimal DataFrame and compare returned names with the registry.

**Warning signs:** If registry lists differ from what `build_feature_matrix()` produces, the diff test catches it immediately.

### Pitfall 2: RaceLevelRanker has three separate feature lists, not one FEATURE_COLS

**What goes wrong:** RaceLevelRanker defines `RELEVANCE_FEATURES` (15), `VALUE_FEATURES` (13), and `DERIVED_VALUE_FEATURES` (2) as separate class attributes. The forbidden set must include all 30 unique features.

**Why it happens:** The ranker uses separate Ridge models for relevance and value scoring, each with its own feature set.

**How to avoid:** FORBIDDEN_RANKER_FEATURES = union of RELEVANCE_FEATURES + VALUE_FEATURES + DERIVED_VALUE_FEATURES. The diff test should verify this union matches what `_build_relevance_features()` and `_build_value_features()` actually produce.

**Warning signs:** If a new feature is added to RELEVANCE_FEATURES but not the registry, the diff test fails.

### Pitfall 3: Overlap between MAWC features and model FEATURE_COLS that is NOT a leak

**What goes wrong:** Some features like "field_size", "surface", "popularity_rank" appear in both the MAWC construction inputs AND model FEATURE_COLS. These are NOT forbidden features -- the forbidden set is the 51 features that MAWC's `build_feature_matrix()` outputs (logit_model, logit_market_x_top_25, etc.), not the raw inputs (p_model, p_market, tanodds, etc.).

**Why it happens:** MAWC transforms raw inputs into derived features (logits, one-hot encodings, interactions). The derived features are what's forbidden in downstream models, not the raw inputs.

**How to avoid:** Clearly document in the registry that FORBIDDEN_CALIBRATOR_FEATURES contains the 51 output features from `build_feature_matrix()`, not the 6 raw input columns. The raw inputs (p_model, p_market, tanodds, popularity_rank, field_size, p_win_race_rank_pct) are legitimate features that other models may use.

**Warning signs:** If the audit flags "field_size" as forbidden because it appears in MAWC inputs, the registry definition is wrong.

### Pitfall 4: shadow_comparison_result.json structure varies by variant count

**What goes wrong:** The JSON structure uses variant names (e.g., "baseline", "shadow") as dictionary keys. DeploymentGateEvaluator must handle any variant naming, not hardcode "baseline"/"shadow".

**Why it happens:** ShadowComparisonFramework supports N-way comparison, not just 2-way.

**How to avoid:** Gate evaluator should identify baseline/shadow by `flag_states` in the manifest (enable_market_aware_calibrator=False = baseline), not by variant name strings.

**Warning signs:** If evaluator crashes with KeyError on variant names when run with non-standard names.

### Pitfall 5: Tolerance comparison for floating-point gate checks

**What goes wrong:** Direct equality comparison on Brier/logloss/ECE between shadow and baseline fails due to floating-point precision.

**Why it happens:** Even identical computations can produce slightly different results on different hardware.

**How to avoid:** Use `shadow <= baseline + tolerance` with tolerance ~1e-6 per D-11. This allows shadow to be marginally better or equal.

**Warning signs:** Gate FAIL on "Brier shadow 0.123456 > baseline 0.123456" when values are effectively equal.

## Code Examples

### Forbidden Feature Set Construction (MAWC)

```python
# Source: src/models/market_aware_win_calibrator.py lines 115-197
# The 51 features are constructed in build_feature_matrix():
# 6 main effects: logit_model, logit_market, log_odds, popularity_rank_pct, p_win_race_rank_pct, field_size
# 7 odds band one-hot: 1-2, 2-3, 3-5, 5-10, 10-30, 30-100, 100+
# 5 pop bucket one-hot: pop_1, pop_2_3, pop_4_6, pop_7_9, pop_10_plus
# 3 p_rank one-hot: top_25, mid_25_75, bottom_25
# 15 logit_model x segment interactions: logit_model_x_{segment}
# 15 logit_market x segment interactions: logit_market_x_{segment}
# Total: 6 + 7 + 5 + 3 + 15 + 15 = 51

# IMPORTANT: These are OUTPUT features of build_feature_matrix(), not raw inputs.
# The raw inputs (p_model, p_market, tanodds, popularity_rank, field_size, p_win_race_rank_pct)
# are NOT in the forbidden set -- they're legitimate features for other models.
```

### Forbidden Feature Set Construction (Ranker)

```python
# Source: src/models/race_level_ranker.py lines 62-101
# RELEVANCE_FEATURES (15): if_p_win_final, if_p_win_race_rank, if_p_ability_win,
#   rel_p_ability_win_rank, if_norm_finish_avg, if_closing_index,
#   if_weighted_recent_form, if_jockey_wr, if_trainer_wr, if_blood_surface_wr,
#   if_class_level, if_surface, if_distance_bin, if_grade_code, if_n_horses
#
# VALUE_FEATURES (13): if_logit_gap, if_edge_win, if_ev_calibrated, if_odds_log,
#   if_odds_band_id, if_odds_drop_60_10, if_odds_drop_30_10, if_overround,
#   if_market_entropy, if_conformal_width, if_ev_uncertainty_ratio,
#   if_p_win_race_rank, if_n_horses
#
# DERIVED_VALUE_FEATURES (2): if_odds_rank, if_abs_logit_gap
# Total: 15 + 13 + 2 = 30 unique (if_p_win_race_rank and if_n_horses appear in both)
# Unique count: 28 unique features
```

### Shadow Comparison Result JSON Structure

```json
// Source: src/backtest/shadow_comparison.py save_results() lines 215-221
{
  "generated_at": "2026-05-28T12:00:00+00:00",
  "folds": {
    "2024": {
      "metrics": {
        "baseline": {"brier": 0.12, "logloss": 0.34, "ece": 0.05, "roi": 0.87, "bet_count": 1500, ...},
        "shadow": {"brier": 0.12, "logloss": 0.33, "ece": 0.04, "roi": 1.02, "bet_count": 1480, ...}
      },
      "metrics_by_surface": {...},
      "metrics_by_odds_band": {...},
      "selection_agreement": 0.82,
      "bet_counts": {"baseline": 1500, "shadow": 1480}
    },
    "2025": {...}
  },
  "overall": {
    "metrics": {
      "baseline": {"brier": 0.12, "logloss": 0.34, ...},
      "shadow": {"brier": 0.12, "logloss": 0.33, ...}
    }
  }
}
```

### Shadow Manifest JSON Structure

```json
// Source: src/backtest/shadow_comparison.py save_manifest() lines 257-325
{
  "generated_at": "...",
  "framework_version": "1.0",
  "variants": [
    {
      "variant_name": "baseline",
      "model_dir": "...",
      "flag_states": {"enable_market_aware_calibrator": false, "enable_race_level_ranker": false},
      "baseline_definition": "MAWC/ranker disabled, ..."
    },
    {
      "variant_name": "shadow",
      "model_dir": "...",
      "flag_states": {"enable_market_aware_calibrator": true, "enable_race_level_ranker": true}
    }
  ],
  "folds": [{"year": 2024, "train_start": "...", ...}],
  "artifacts": {
    "metrics_json": {"path": "shadow_comparison_result.json", "sha256": "..."},
    "race_diff_parquet": {"path": "shadow_race_diff.parquet", "sha256": "..."}
  }
}
```

### OOFHealthValidator Profile Pattern

```python
# Source: src/validation/oof_health_validator.py lines 19-40
# Existing profile pattern:
OOF_PREDICTIONS_PROFILE = OOFHealthProfile(
    artifact_name="oof_predictions",
    required_columns=("race_id", "race_date", "is_oof", "oof_artifact_version"),
    fold_col="ability_oof_fold",
    score_col="p_win_oof",
    return_cols=("confirmed_odds", "tanodds"),
    manifest_path="data/oof/manifests/oof_predictions.health.json",
)
# New MAWC/Ranker profiles follow this pattern but with additional
# validation logic in separate profile classes (D-06).
```

### RacePredictor Feature Flag Pattern (no changes needed)

```python
# Source: src/backtest/race_predictor.py lines 108-128, 285-310
# MAWC control (lines 285-301):
mawc = getattr(submodel, "market_aware_win_calibrator", None)
if self.enable_market_aware_calibrator and mawc is not None and mawc.is_trained:
    df = mawc.apply(df)

# Ranker control (lines 308-310):
ranker = getattr(submodel, "win_race_level_ranker", None)
if self.enable_race_level_ranker and ranker is not None and ranker.is_trained:
    df = ranker.score(df)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Feature leak detection by manual review | Automated audit registry + diff tests | Phase 42 | Catches leaks at CI time, not post-deploy |
| No gate evaluation | DeploymentGateEvaluator with GatePolicy | Phase 42 | Explicit PASS/FAIL/WARN with thresholds |
| Hardcoded OOF checks | Profile-based plugin architecture | Phase 42 | New artifact types added without modifying validator core |

**Deprecated/outdated:**
- WinBenterGate + WinSegmentCalibrator: Replaced by MarketAwareWinCalibrator (Phase 39, CAL-04)
- Direct feature list in model FEATURE_COLS for MAWC: MAWC constructs features dynamically in build_feature_matrix()

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | MAWC produces exactly 51 features (6 main + 15 one-hot + 30 interactions) | Standard Stack / Code Examples | LOW -- verified at line 194 of market_aware_win_calibrator.py with assert |
| A2 | Ranker has 28 unique features across three lists (15 + 13 + 2 with 2 overlapping) | Code Examples | LOW -- verified by reading RELEVANCE_FEATURES, VALUE_FEATURES, DERIVED_VALUE_FEATURES |
| A3 | No existing src/audit/ directory exists | Architecture Patterns | LOW -- verified by Glob returning no results |
| A4 | OOFHealthValidator profiles can be extended without modifying the core validate() method | Pattern 2 | LOW -- D-06 decision explicitly states this |
| A5 | shadow_comparison_result.json uses variant names as dict keys under "metrics" per fold | Code Examples | LOW -- verified by reading save_results() in shadow_comparison.py |

## Open Questions (RESOLVED)

1. **MAWC OOF fold column name** -- RESOLVED: Configurable fold_col with default `"ability_oof_fold"` per `generate_win_oof_predictions()` output. Plan 42-02 CalibratorArtifactProfile implements exactly this.

2. **Ranker OOF fold column name** -- RESOLVED: Same approach -- configurable fold_col with default `"ability_oof_fold"`. Ranker OOF shares the same pipeline fold column. Plan 42-02 RankerArtifactProfile implements this.

## Environment Availability

> Step 2.6: SKIPPED (no external dependencies identified). This phase uses only Python stdlib and existing project dependencies (pandas, numpy, pytest). No new packages to install.

## Validation Architecture

> nyquist_validation is explicitly set to `false` in `.planning/config.json`. Skipping this section.

## Security Domain

> This phase creates audit/validation/gate evaluation infrastructure with no user-facing endpoints, no authentication, and no cryptographic key generation. Applicable security considerations are limited to:

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | Gate input JSON validated via json.load() + schema checks |
| V6 Cryptography | partial | SHA256 verification of artifacts (existing pattern from Phase 37/41) |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Artifact tampering (modified shadow_comparison_result.json) | Tampering | SHA256 verification against manifest (D-22) |
| Feature leak via stale registry | Elevation of Privilege | Diff test catches registry drift vs actual FEATURE_COLS |

## Sources

### Primary (HIGH confidence)
- `src/models/market_aware_win_calibrator.py` -- Full file read. MAWC architecture, build_feature_matrix() producing 51 features, train(), apply() methods.
- `src/models/race_level_ranker.py` -- Full file read. Ranker architecture, RELEVANCE_FEATURES (15), VALUE_FEATURES (13), DERIVED_VALUE_FEATURES (2).
- `src/models/market_model.py` -- Full file read. MarketModel.FEATURE_COLS (27 features).
- `src/models/race_quality_screener.py` -- Full file read. RaceQualityScreener.FEATURE_COLS (41 features).
- `src/validation/oof_health_validator.py` -- Full file read. OOFHealthProfile, validate(), generate_manifest(), existing profiles.
- `src/backtest/shadow_comparison.py` -- Full file read. ShadowComparisonFramework, ComparisonMetrics, save_results(), save_manifest(), shadow_comparison_result.json structure.
- `src/backtest/race_predictor.py` -- Full file read. Feature flag pattern for MAWC (lines 285-301) and ranker (lines 308-310).
- `src/pipelines/training_pipeline.py` -- Relevant sections read. MAWC OOF generation (lines 1287-1322), Ranker training (lines 1324-1355).
- `src/domain/models.py` -- SubmodelSet fields (lines 234-273), TrainedModelsV5.

### Secondary (MEDIUM confidence)
- `.planning/phases/42-CONTEXT.md` -- User decisions constraining implementation.
- `.planning/REQUIREMENTS.md` -- SAF-01/02/03 requirement definitions.
- `.planning/STATE.md` -- Project state, v1.8 ROI collapse context.

### Tertiary (LOW confidence)
- (none -- all findings verified by direct code inspection)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all existing code inspected
- Architecture: HIGH -- patterns established by Phase 37-41 codebase
- Pitfalls: HIGH -- identified by direct code inspection of feature construction patterns

**Research date:** 2026-05-28
**Valid until:** 2026-06-28 (stable codebase, no fast-moving dependencies)
