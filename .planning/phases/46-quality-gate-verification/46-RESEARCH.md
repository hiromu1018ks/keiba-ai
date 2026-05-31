# Phase 46: Quality Gate Verification - Research

**Researched:** 2026-05-31
**Domain:** Quality gate orchestration for MAWC conservative variant verification
**Confidence:** HIGH

## Summary

Phase 46 is a pure orchestration and verification phase that invokes 5 existing quality infrastructure components in sequence, collects their PASS/FAIL results, and produces a milestone completion certificate. All components already exist and have CLI wrappers and/or function APIs. The planner's primary task is designing `scripts/run_phase46_quality_gates.py` that calls them in order with correct parameters, handles skip/resume logic, and aggregates results into `phase46_quality_gate_result.json`.

The two-stage flow (D-01) is: Stage 1 runs the MAWC conservative retrain CLI from Phase 45; Stage 2 runs 5 quality checks in order. Every component returns a JSON-serializable status. No new ML models or quality gates are created.

**Primary recommendation:** Build the orchestration CLI as a thin wrapper around existing CLIs/function APIs. Each step checks for existing output artifacts before running. Use `subprocess.run()` for long-running steps (Shadow Comparison) and direct function calls for fast steps (OOFHealthValidator, FeatureRoutingAudit).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 2-stage execution flow. Stage 1: `run_mawc_conservative_retrain.py`. Stage 2: FeatureRoutingAudit -> OOFHealthValidator -> Shadow Comparison -> Shadow Diagnosis -> DeploymentGateEvaluator -> Final summary.
- **D-02:** Orchestration CLI + runbook both created. CLI wraps existing components. No new judgment logic.
- **D-03:** 3-label separation framework: Quality Gate (PASS/FAIL), ROI Trend (recovered/weak_recovery/not_recovered), Deployment (deployable/not_deployable/manual_review).
- **D-04:** No retry/re-exploration within Phase 46. Only execution bugs may be fixed.
- **D-05:** Specific artifact paths defined. HTML not mandatory for Phase 46 new artifacts.

### Claude's Discretion
- FeatureRoutingAudit/OOFHealthValidator call method (function vs CLI)
- Skip/resume detection logic
- Stage 2 inter-step artifact path passing
- phase46_quality_gate_result.json schema
- v2.2-MILESTONE-SUMMARY.md structure
- Test structure/naming

### Deferred Ideas (OUT OF SCOPE)
- Phase 45b (MAWC re-adjustment)
- Ranker modification
- OddsBandFilter retraining
- DEP-01 auto-deploy
- DEP-02 Optuna 19-dim optimization
- Regime analysis/regime parameter tuning
- New features
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| QUAL-01 | OOFHealthValidator PASS (post-fix OOF health verification) | `OOFHealthValidator.validate()` with `OOF_PREDICTIONS_PROFILE` |
| QUAL-02 | FeatureRoutingAudit PASS (50+28 forbidden feature CI safety audit) | `run_feature_audit()` or `scripts/run_feature_routing_audit.py` |
| QUAL-03 | DeploymentGateEvaluator PASS (probability quality, bet count, reproducibility, diagnostics 4-gate) | `run_deployment_gates()` with Shadow Comparison artifacts |
| QUAL-04 | ROI recovery trend confirmation. Required: Brier/logloss/ECE non-degradation, actual/predicted non-degradation, bet_count maintained | Shadow Comparison + Shadow Diagnosis + DeploymentGateEvaluator aggregate |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Stage 1 retrain | CLI script | Python function | Long-running (~minutes), subprocess invocation |
| OOF validation | Python function | - | Fast (<1s), in-process call |
| Feature routing audit | Python function | - | Fast (<1s), in-process call |
| Shadow comparison | CLI script | - | Long-running (~41min/year), subprocess invocation |
| Shadow diagnosis | Python function | CLI script | Moderate (~seconds), function call preferred |
| Deployment gate evaluation | Python function | - | Fast (<1s), reads existing JSON artifacts |
| Final result aggregation | Python function | - | Reads all JSON outputs, produces summary |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| OOFHealthValidator | in-tree | OOF artifact health validation | Existing Phase 37 infrastructure [VERIFIED: codebase] |
| FeatureRoutingAuditRegistry | in-tree | 50+28 forbidden feature audit | Existing Phase 42 infrastructure [VERIFIED: codebase] |
| DeploymentGateEvaluator | in-tree | 4-gate deployment evaluation | Existing Phase 41 infrastructure [VERIFIED: codebase] |
| ShadowComparisonFramework | in-tree | Baseline vs shadow backtest comparison | Existing Phase 41 infrastructure [VERIFIED: codebase] |
| ShadowDiagnosis | in-tree | 3-step progressive exclusion diagnosis | Existing Phase 43 infrastructure [VERIFIED: codebase] |
| MawcConservativeRetrainer | in-tree | Conservative MAWC retraining | Existing Phase 45 infrastructure [VERIFIED: codebase] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| subprocess (stdlib) | - | Long-running CLI invocation | Shadow Comparison, MAWC retrain |
| json (stdlib) | - | Artifact reading/writing | All steps |
| jinja2 | installed | HTML report generation | Optional report for existing components |

**Installation:** No new external packages needed. Phase 46 is pure orchestration of existing code.

## Architecture Patterns

### Recommended Project Structure
```
scripts/
  run_phase46_quality_gates.py    # Orchestration CLI (NEW)
.planning/phases/46-quality-gate-verification/
  46-RUNBOOK.md                   # Manual reproduction commands (NEW)
  46-VERIFICATION.md              # Phase verification (NEW)
  46-RESEARCH.md                  # This file
data/backtest/phase46_quality_gates/
  phase46_quality_gate_result.json
  phase46_quality_gate_summary.md
data/backtest/shadow_mawc_conservative/
  shadow_comparison_result.json   # Shadow Comparison output
  shadow_manifest.json            # Shadow Comparison manifest
  shadow_race_diff.parquet
  shadow_horse_diff.parquet
  shadow_race_diff.csv
  diagnosis/
    shadow_diagnosis_result.json
    shadow_diagnosis_summary.md
  gates/
    deployment_gate_result.json
    deployment_gate_report.md
data/models-backtest-mawc-conservative/
  manifest.json                   # Stage 1 output
  retrain_summary.md              # Stage 1 output
  {2024,2025}/                    # Conservative variant model dirs
.planning/v2.2-MILESTONE-SUMMARY.md  # Milestone completion certificate
```

### Pattern 1: Two-Stage Orchestration
**What:** Stage 1 runs retrain; Stage 2 runs quality checks. Stage 2 only if Stage 1 succeeds.
**When to use:** Phase 46 execution flow.
**Example:**
```python
# Stage 1: subprocess for long-running retrain
result = subprocess.run(
    [sys.executable, "scripts/run_mawc_conservative_retrain.py",
     "--oof-path", str(oof_path),
     "--source-model-dir", str(source_dir),
     "--target-root", str(target_root),
     "--years", "2024,2025"],
    capture_output=True, text=True, cwd=ROOT,
)
if result.returncode != 0:
    logger.error("Stage 1 FAILED: %s", result.stderr)
    record_result("stage1", "FAIL", ...)
    sys.exit(1)

# Verify manifest exists
manifest_path = target_root / "manifest.json"
if not manifest_path.exists():
    logger.error("Stage 1 manifest not found")
    sys.exit(1)

# Stage 2: quality gates
...
```

### Pattern 2: Skip/Resume Detection
**What:** Check if output artifact exists before running each step.
**When to use:** Every Stage 2 step.
**Example:**
```python
def _should_run(output_path: Path, force: bool = False) -> bool:
    if force:
        return True
    if output_path.exists():
        logger.info("SKIP: %s already exists", output_path)
        return False
    return True
```

### Pattern 3: Fast Step via Function Call
**What:** Import and call existing functions directly for fast operations.
**When to use:** OOFHealthValidator, FeatureRoutingAudit, DeploymentGateEvaluator.
**Example:**
```python
from audit.feature_routing_registry import run_feature_audit
results = run_feature_audit()
if results["overall_status"] != "PASS":
    # Record FAIL and stop
```

### Pattern 4: Long Step via Subprocess
**What:** Use `subprocess.run()` for long-running operations.
**When to use:** Shadow Comparison (~82 min for 2 years).
**Example:**
```python
result = subprocess.run(
    [sys.executable, "scripts/run_shadow_comparison.py",
     "--baseline-root", "data/models-backtest",
     "--shadow-root", "data/models-backtest-mawc-conservative",
     "--folds", "2024", "2025",
     "--output-dir", "data/backtest/shadow_mawc_conservative",
     "--shadow-name", "mawc_conservative",
     "--report"],
    cwd=ROOT,
)
```

### Anti-Patterns to Avoid
- **Adding new quality gates:** Phase 46 only orchestrates existing gates. D-02 explicitly forbids new judgment logic.
- **Calling ShadowComparisonFramework directly:** It requires full BacktestEngine execution with model loading. Use the CLI wrapper which handles all setup.
- **Modifying models in Phase 46:** D-04 forbids all model changes. Only execution bugs may be fixed.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OOF validation | Custom OOF checks | `OOFHealthValidator.validate()` | Handles OOF-01 through OOF-08 edge cases |
| Feature leak detection | Manual feature list comparison | `run_feature_audit()` | Dynamic import of model FEATURE_COLS ensures always-current |
| Deployment gate evaluation | Manual metric comparison | `run_deployment_gates()` | Handles SHA256 verification, per-fold evaluation, WARN/FAIL logic |
| Shadow model comparison | Direct BacktestEngine calls | `scripts/run_shadow_comparison.py` CLI | Handles model loading, fold creation, N-way variant setup |
| Diagnosis | Manual metric comparison | `ShadowDiagnosis(input_dir).run()` | Handles variant name resolution, segment computation |

**Key insight:** Phase 46 is orchestration-only. Every quality gate already exists. The CLI just calls them in sequence and records results.

## Common Pitfalls

### Pitfall 1: Shadow Comparison variant naming mismatch
**What goes wrong:** Shadow Comparison CLI defaults to `--shadow-name ridge_shadow`. Phase 45 conservative variant should use `mawc_conservative` or similar.
**Why it happens:** The CLI default doesn't match the Phase 45 output context.
**How to avoid:** Explicitly set `--shadow-name mawc_conservative` in the Shadow Comparison invocation.
**Warning signs:** ShadowDiagnosis and DeploymentGateEvaluator report `variant_names=[]` or fall back to "baseline"/"shadow".

### Pitfall 2: Stage 1 output directory does not exist
**What goes wrong:** `data/models-backtest-mawc-conservative/` is empty or missing if Stage 1 was never run.
**Why it happens:** Phase 45 created the CLI but may not have run it on production data yet.
**How to avoid:** Stage 1 always runs first. If the target directory is empty, run retrain. If manifest.json exists and is valid, skip Stage 1.
**Warning signs:** `manifest.json` not found at `data/models-backtest-mawc-conservative/manifest.json`.

### Pitfall 3: MAWC enable flag mismatch
**What goes wrong:** Shadow Comparison's baseline variant has `enable_market_aware_calibrator=False` by design. But the conservative variant should have `enable_market_aware_calibrator=True` (it uses a MAWC, just a conservative one).
**Why it happens:** Confusion about what "conservative" means -- it's still a MAWC, just with reduced features.
**How to avoid:** For Shadow Comparison, baseline: MAWC=False, Ranker=False. Shadow (mawc_conservative): MAWC=True, Ranker=True (or False if ranker not in conservative variant). Check manifest.json for actual deployment status.
**Warning signs:** DeploymentGateEvaluator `_identify_variants()` returns wrong variant names.

### Pitfall 4: DeploymentGateEvaluator diagnostic gates are SKIP
**What goes wrong:** DeploymentGateEvaluator always emits `diagnostic_oof_health: SKIP` and `diagnostic_feature_routing_audit: SKIP` because these are placeholder gates (D-05).
**Why it happens:** By design, these are CI-independent gates that must be run separately.
**How to avoid:** Phase 46 runs OOFHealthValidator and FeatureRoutingAudit *before* DeploymentGateEvaluator. The orchestration CLI should record these separate results and treat them as PASS/FAIL independently from the DeploymentGateEvaluator result.
**Warning signs:** DeploymentGateEvaluator result shows SKIP for OOF and audit gates but overall PASS -- this doesn't mean OOF/audit passed.

### Pitfall 5: OOFHealthValidator has no CLI wrapper
**What goes wrong:** Searching for `scripts/run_oof_health_validator.py` -- it doesn't exist.
**Why it happens:** OOFHealthValidator was designed as a function API (Phase 37), not a CLI tool.
**How to avoid:** Call `OOFHealthValidator().validate()` directly from the orchestration CLI. Read `data/oof/oof_predictions.parquet`, validate with `OOF_PREDICTIONS_PROFILE`, record status.
**Warning signs:** Trying to run a non-existent CLI script.

### Pitfall 6: manifest.json per-year surface key collision
**What goes wrong:** Phase 45 CR-01 fixed a bug where multi-year manifests silently overwrote earlier year data.
**Why it happens:** Original manifest used flat `per_surface` key instead of `per_year_surface`.
**How to avoid:** When reading manifest.json, use `per_year_surface` (year-keyed) not the deprecated `per_surface` key.
**Warning signs:** Only last year's data visible in manifest.

## Code Examples

### OOFHealthValidator invocation
```python
# Source: src/validation/oof_health_validator.py (VERIFIED: codebase)
import pandas as pd
from validation.oof_health_validator import OOFHealthValidator, OOF_PREDICTIONS_PROFILE

df = pd.read_parquet("data/oof/oof_predictions.parquet")
validator = OOFHealthValidator()
result = validator.validate(df, OOF_PREDICTIONS_PROFILE)
# result = {"status": "PASS"|"FAIL", "failures": [...], "warnings": [...], ...}
status = result["status"]  # "PASS" or "FAIL"
```

### FeatureRoutingAudit invocation
```python
# Source: src/audit/feature_routing_registry.py (VERIFIED: codebase)
from audit.feature_routing_registry import run_feature_audit

results = run_feature_audit()
# results = {
#   "registry_version": "1.0",
#   "critical_models": [...],
#   "advisory_models": [...],
#   "overall_status": "PASS"|"FAIL"
# }
# For CLI: python scripts/run_feature_routing_audit.py --output-dir data/audit
```

### FeatureRoutingAudit CLI invocation
```bash
# Source: scripts/run_feature_routing_audit.py (VERIFIED: codebase)
python scripts/run_feature_routing_audit.py --output-dir data/audit
# Exits 0 on PASS, 1 on FAIL
# Writes: data/audit/feature_routing_audit.json + .md
```

### DeploymentGateEvaluator invocation
```python
# Source: src/backtest/deployment_gates.py (VERIFIED: codebase)
from backtest.deployment_gates import run_deployment_gates

result = run_deployment_gates(
    result_path="data/backtest/shadow_mawc_conservative/shadow_comparison_result.json",
    manifest_path="data/backtest/shadow_mawc_conservative/shadow_manifest.json",
    output_dir="data/backtest/shadow_mawc_conservative/gates",
)
# result.overall_status: "PASS"|"FAIL"|"WARN"
# Writes: deployment_gate_result.json + deployment_gate_report.md
# NOTE: import is `from backtest.deployment_gates import run_deployment_gates`
# (NOT `evaluate_deployment_gates` as CLAUDE.md incorrectly states)
```

### ShadowComparison CLI invocation
```bash
# Source: scripts/run_shadow_comparison.py (VERIFIED: codebase)
python scripts/run_shadow_comparison.py \
  --baseline-root data/models-backtest \
  --shadow-root data/models-backtest-mawc-conservative \
  --folds 2024 2025 \
  --output-dir data/backtest/shadow_mawc_conservative \
  --baseline-name baseline \
  --shadow-name mawc_conservative \
  --report
# Runtime: ~82 min (2 years)
# Outputs: shadow_comparison_result.json, shadow_manifest.json,
#          shadow_race_diff.parquet/.csv, shadow_horse_diff.parquet
#          + optional HTML report
```

### ShadowDiagnosis invocation
```python
# Source: src/backtest/shadow_diagnosis.py (VERIFIED: codebase)
from backtest.shadow_diagnosis import ShadowDiagnosis, save_diagnosis_results
from pathlib import Path

sd = ShadowDiagnosis(Path("data/backtest/shadow_mawc_conservative"))
result = sd.run()
# result: ShadowDiagnosisResult with step1/step2/step3
save_diagnosis_results(result, Path("data/backtest/shadow_mawc_conservative/diagnosis"))
# Writes: shadow_diagnosis_result.json + shadow_diagnosis_summary.md
```

### ShadowDiagnosis CLI invocation
```bash
# Source: scripts/run_shadow_diagnosis.py (VERIFIED: codebase)
python scripts/run_shadow_diagnosis.py \
  --input-dir data/backtest/shadow_mawc_conservative \
  --output-dir data/backtest/shadow_mawc_conservative/diagnosis \
  --report
# Outputs: shadow_diagnosis_result.json, shadow_diagnosis_summary.md,
#          optional HTML report
```

### MAWC Conservative Retrain CLI invocation
```bash
# Source: scripts/run_mawc_conservative_retrain.py (VERIFIED: codebase)
python scripts/run_mawc_conservative_retrain.py \
  --oof-path data/oof/oof_predictions.parquet \
  --source-model-dir data/models-backtest \
  --target-root data/models-backtest-mawc-conservative \
  --years 2024,2025
# Runtime: depends on OOF data size and model loading
# Outputs: manifest.json, retrain_summary.md in target-root
#          + optional HTML report with --report
```

### MAWC manifest.json structure
```json
{
  "mawc_fix_version": "45-conservative",
  "source_model_dir": "data/models-backtest",
  "target_variant_dir": "data/models-backtest-mawc-conservative",
  "C_grid": [0.003, 0.005, 0.01, 0.03],
  "removed_interactions": ["logit_model_x_1-2", ...],
  "feature_dim": 36,
  "original_feature_dim": 51,
  "years": ["2024", "2025"],
  "per_year_surface": {
    "2024": {
      "turf": {
        "best_c": 0.01,
        "deployed": true,
        "beta_market_contribution": 0.35,
        "quality_gate_summary": { ... }
      },
      "dirt": { ... }
    },
    "2025": { ... }
  },
  "generated_at": "..."
}
```

### Stage 1 fail check (manifest inspection)
```python
import json
manifest = json.loads(Path("data/models-backtest-mawc-conservative/manifest.json").read_text())

# Check if any surface was deployed
any_deployed = False
for year_data in manifest.get("per_year_surface", {}).values():
    for surface_data in year_data.values():
        if surface_data.get("deployed", False):
            any_deployed = True

if not any_deployed:
    logger.error("No surfaces deployed in conservative variant")
    # Record BLOCKED/FAIL and exit Stage 2
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `evaluate_deployment_gates` import | `run_deployment_gates` function | Phase 41 | CLAUDE.md still references old name; use `run_deployment_gates` |
| Flat per_surface manifest | per_year_surface (CR-01) | Phase 45 CR-01 | Prevents multi-year manifest collision |

**Deprecated/outdated:**
- `from backtest.deployment_gates import evaluate_deployment_gates` -- does not exist. Use `run_deployment_gates` [VERIFIED: codebase grep shows no `evaluate_deployment_gates` function exists]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `data/models-backtest-mawc-conservative/` does not exist yet (not run on production data) | Phase 45 output | Stage 1 will create it |
| A2 | Shadow Comparison with `--shadow-name mawc_conservative` is the correct variant name | Architecture | If wrong, Diagnosis/GateEvaluator may misidentify variants |
| A3 | Shadow variant should have `enable_market_aware_calibrator=True` | Shadow Comparison | If wrong, comparison won't use conservative MAWC |
| A4 | Shadow variant should have `enable_race_level_ranker=True` | Shadow Comparison | CONTEXT.md D-01 implies full pipeline for shadow |

**Note:** A2-A4 should be verified against the manifest.json produced by Stage 1 before Stage 2 runs.

## Open Questions

1. **Shadow variant MAWC/Ranker flags for Shadow Comparison**
   - What we know: Baseline is MAWC=False, Ranker=False. Conservative variant uses a MAWC (conservative retrain).
   - What's unclear: Should `enable_race_level_ranker` be True for the shadow variant?
   - Recommendation: Check manifest.json after Stage 1. If conservative variant has ranker artifacts, enable it. If not, only enable MAWC=True.

2. **ROI baseline value for QUAL-04**
   - What we know: v2.0 close ROI was 87.8%. v1.7 achieved 97.8%.
   - What's unclear: Exact baseline ROI from the most recent backtest run.
   - Recommendation: Extract from Shadow Comparison baseline metrics (`overall.metrics.baseline.roi`).

3. **Stage 1 execution time**
   - What we know: MawcConservativeRetrainer uses OOF data (no backtest engine).
   - What's unclear: Actual runtime for production OOF data.
   - Recommendation: Estimate ~5-10 minutes based on LogisticRegression fitting on OOF data.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 | All components | Yes (mise) | 3.11 | - |
| data/oof/oof_predictions.parquet | OOF validation + MAWC retrain | Yes | - | BLOCKING if missing |
| data/models-backtest/ | Shadow Comparison baseline | Yes | - | BLOCKING if missing |
| data/features/horse_features.parquet | Feature engine cache | Yes | - | - |
| jinja2 | HTML report generation | Yes | installed | Skip HTML |

**Missing dependencies with no fallback:**
- None identified -- all dependencies are existing data files and in-tree code.

**Missing dependencies with fallback:**
- HTML report generation: optional, can skip with no impact on quality gate results.

## Component API Reference

### 1. OOFHealthValidator (QUAL-01)

**File:** `src/validation/oof_health_validator.py`
**CLI:** None. Function API only.
**Import:** `from validation.oof_health_validator import OOFHealthValidator, OOF_PREDICTIONS_PROFILE`

```python
validator = OOFHealthValidator()
df = pd.read_parquet("data/oof/oof_predictions.parquet")
result: dict = validator.validate(df, OOF_PREDICTIONS_PROFILE)
# Returns: {"status": "PASS"|"FAIL", "failures": list[str], "warnings": list[str], ...}
```

**Key parameters:**
- `df`: DataFrame of OOF predictions
- `profile`: `OOF_PREDICTIONS_PROFILE` or `WIN_SELECTION_OOF_PROFILE`
- Optional: `train_date_range`, `expected_row_count`, `split_metadata`

**Checks performed:** OOF-01 (empty), OOF-02 (train/valid overlap), OOF-03 (top1 anomaly), OOF-04 (row coverage), OOF-05 (fold count), OOF-06 (multi-fold races), OOF-07 (required columns)

### 2. FeatureRoutingAuditRegistry (QUAL-02)

**File:** `src/audit/feature_routing_registry.py`
**CLI:** `scripts/run_feature_routing_audit.py --output-dir data/audit`
**Import:** `from audit.feature_routing_registry import run_feature_audit`

```python
# Function API:
results: dict = run_feature_audit()
# Returns: {"registry_version": "1.0", "overall_status": "PASS"|"FAIL",
#           "critical_models": [...], "advisory_models": [...]}

# CLI API:
# python scripts/run_feature_routing_audit.py --output-dir data/audit
# Exits: 0=PASS, 1=FAIL
# Writes: data/audit/feature_routing_audit.json + .md
```

**Recommendation:** Use function API for orchestration (faster, direct result access).

### 3. DeploymentGateEvaluator (QUAL-03)

**File:** `src/backtest/deployment_gates.py`
**CLI:** None standalone. `run_deployment_gates()` is the module-level convenience function.
**Import:** `from backtest.deployment_gates import run_deployment_gates`

```python
result = run_deployment_gates(
    result_path="path/to/shadow_comparison_result.json",
    manifest_path="path/to/shadow_manifest.json",  # Optional
    output_dir="path/to/output/dir",  # Optional, writes JSON+MD reports
)
# result.overall_status: "PASS"|"FAIL"|"WARN"
# result.conditions: list[GateConditionResult]
# result.report_metrics: dict (ROI, selection_agreement, etc.)
```

**Gate conditions:**
1. `brier_fold_{year}` / `brier_overall` -- shadow <= baseline + tolerance (1e-6)
2. `logloss_fold_{year}` / `logloss_overall` -- shadow <= baseline + tolerance
3. `ece_fold_{year}` / `ece_overall` -- shadow <= baseline + tolerance
4. `bet_count_preservation_fold_{year}` -- shadow >= baseline * 0.95
5. `actual_predicted_ratio_fold_{year}` -- WARN only (not FAIL)
6. `artifact_reproducibility_sha256` -- manifest SHA256 verification
7. `artifact_reproducibility_completeness` -- all artifacts have path + sha256
8. `diagnostic_oof_health` -- SKIP (D-05)
9. `diagnostic_feature_routing_audit` -- SKIP (D-05)

**IMPORTANT:** The SKIP gates for OOF and audit mean DeploymentGateEvaluator alone does NOT verify QUAL-01 and QUAL-02. The orchestration CLI must run these separately.

### 4. ShadowComparisonFramework (Stage 2)

**File:** `src/backtest/shadow_comparison.py`
**CLI:** `scripts/run_shadow_comparison.py`
**Import:** Multiple imports needed for direct use; CLI recommended.

**CLI for Phase 46:**
```bash
python scripts/run_shadow_comparison.py \
  --baseline-root data/models-backtest \
  --shadow-root data/models-backtest-mawc-conservative \
  --folds 2024 2025 \
  --output-dir data/backtest/shadow_mawc_conservative \
  --baseline-name baseline \
  --shadow-name mawc_conservative \
  --report
```

**Runtime:** ~41 min/year, so ~82 min for 2 years.
**Outputs:** `shadow_comparison_result.json`, `shadow_manifest.json`, `shadow_race_diff.parquet`, `shadow_horse_diff.parquet`, `shadow_race_diff.csv`, optional HTML report.

**Variant configuration built by CLI:**
- baseline: `enable_market_aware_calibrator=False`, `enable_race_level_ranker=False`
- shadow (mawc_conservative): `enable_market_aware_calibrator=True`, `enable_race_level_ranker=True`

### 5. ShadowDiagnosis (Stage 2)

**File:** `src/backtest/shadow_diagnosis.py`
**CLI:** `scripts/run_shadow_diagnosis.py`
**Import:** `from backtest.shadow_diagnosis import ShadowDiagnosis, save_diagnosis_results`

```python
# Function API:
sd = ShadowDiagnosis(Path("data/backtest/shadow_mawc_conservative"))
result = sd.run()  # ShadowDiagnosisResult
save_diagnosis_results(result, Path("data/backtest/shadow_mawc_conservative/diagnosis"))

# CLI API:
# python scripts/run_shadow_diagnosis.py \
#   --input-dir data/backtest/shadow_mawc_conservative \
#   --output-dir data/backtest/shadow_mawc_conservative/diagnosis \
#   --report
```

**Required input files:**
- `shadow_comparison_result.json`
- `shadow_race_diff.parquet`
- `shadow_horse_diff.parquet`
- `shadow_manifest.json`

**3-step diagnosis:**
- Step 1 (DIAG-01): Probability quality -- Brier/logloss/ECE/APR for baseline vs shadow
- Step 2 (DIAG-02): Selection pattern -- changed/unchanged race ROI/HR for both variants
- Step 3 (DIAG-03): Calibration by segment -- per-segment APR/ECE comparison

### 6. MawcConservativeRetrainer (Stage 1)

**File:** `scripts/run_mawc_conservative_retrain.py`
**CLI only** -- invoked via subprocess.

```bash
python scripts/run_mawc_conservative_retrain.py \
  --oof-path data/oof/oof_predictions.parquet \
  --source-model-dir data/models-backtest \
  --target-root data/models-backtest-mawc-conservative \
  --years 2024,2025
```

**Stage 1 success criteria (for Stage 2 gate):**
1. `manifest.json` exists in target-root
2. At least one surface in at least one year has `deployed: true`
3. Favorite band guard passed for deployed surfaces
4. Year directories `{2024,2025}` exist under target-root with model artifacts

### 7. Existing Test Patterns

**Test files relevant to Phase 46:**
- `tests/test_oof_health_validator.py` -- Tests OOFHealthValidator with synthetic DataFrames
- `tests/test_feature_routing_audit.py` -- Tests `run_feature_audit()` function
- `tests/test_deployment_gates.py` -- Tests DeploymentGateEvaluator with mock shadow_comparison_result.json
- `tests/test_shadow_comparison.py` -- Tests ShadowComparisonFramework with mock BacktestEngine
- `tests/test_shadow_diagnosis.py` -- Tests ShadowDiagnosis with synthetic horse/race diff DataFrames
- `tests/test_mawc_conservative_retrainer.py` -- Tests MawcConservativeRetrainer

**Test pattern:** All tests use `unittest.mock` for BacktestEngine, model loading, and file I/O. No database required. Synthetic DataFrames with known shapes and values.

## Sources

### Primary (HIGH confidence)
- `src/validation/oof_health_validator.py` -- Full source read, API signatures verified
- `src/audit/feature_routing_registry.py` -- Full source read, API signatures verified
- `src/backtest/deployment_gates.py` -- Full source read, API signatures verified
- `src/backtest/shadow_comparison.py` -- Full source read, API signatures verified
- `src/backtest/shadow_diagnosis.py` -- Full source read, API signatures verified
- `src/models/mawc_conservative_retrainer.py` -- Full source read, API signatures verified
- `scripts/run_shadow_comparison.py` -- Full source read, CLI args verified
- `scripts/run_shadow_diagnosis.py` -- Full source read, CLI args verified
- `scripts/run_feature_routing_audit.py` -- Full source read, CLI args verified
- `scripts/run_mawc_conservative_retrain.py` -- Full source read, CLI args verified

### Secondary (MEDIUM confidence)
- `.planning/phases/46-quality-gate-verification/46-CONTEXT.md` -- Decisions D-01 through D-05

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all components are in-tree, source code fully read
- Architecture: HIGH -- 2-stage orchestration pattern well-defined in CONTEXT.md
- Pitfalls: HIGH -- identified from source code analysis and previous phase context
- API signatures: HIGH -- verified by reading actual source files

**Research date:** 2026-05-31
**Valid until:** 2026-06-30 (stable -- all components are committed code)
