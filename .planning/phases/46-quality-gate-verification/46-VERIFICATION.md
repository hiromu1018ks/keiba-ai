---
phase: 46-quality-gate-verification
verified: 2026-06-01
status: failed
score: 0/4 QUAL runtime gates
---

# Phase 46: Quality Gate Verification -- Runtime Results

**Phase Goal:** 全修正が安全ゲートを通過し、ROI回復傾向と品質指標非悪化が確認されている
**Verified:** 2026-06-01 (runtime execution)
**Status:** FAILED -- Stage 1 FAIL (no surfaces deployed)
**Executor:** `python scripts/run_phase46_quality_gates.py --years 2024,2025 --report`

## Runtime Execution Result

### Stage 1: MAWC Conservative Retrain -- FAIL

**Result:** All 4 surface/year combinations failed quality gates. No surfaces deployed.

| Year | Surface | Candidates | Passing | Best C | Deployed |
|------|---------|-----------|---------|--------|----------|
| 2024 | turf | 4 | 0 | None | No |
| 2024 | dirt | 4 | 0 | None | No |
| 2025 | turf | 4 | 0 | None | No |
| 2025 | dirt | 4 | 0 | None | No |

### Root Cause Analysis

**Symptom:** All C grid candidates (0.003, 0.005, 0.01, 0.03) fail ECE gate.

**Baseline metrics (turf, all OOF years):**
- Brier=0.059684, Logloss=0.209939, **ECE=0.003139**
- 10% tolerance threshold: ECE <= 0.003453

**Conservative candidates (turf):**

| C | Brier | Logloss | ECE | Brier Gate | Logloss Gate | ECE Gate |
|---|-------|---------|-----|-----------|-------------|----------|
| 0.003 | 0.046383 | 0.160639 | 0.022820 | PASS | PASS | **FAIL** |
| 0.005 | 0.043748 | 0.152885 | 0.022435 | PASS | PASS | **FAIL** |
| 0.010 | 0.040869 | 0.144509 | 0.020012 | PASS | PASS | **FAIL** |
| 0.030 | 0.038280 | 0.137135 | 0.014449 | PASS | PASS | **FAIL** |

**Root cause:** Baseline MAWC (51-dim) achieves ECE=0.003 on OOF data because it was **trained on this same data** (in-sample evaluation). The conservative variant (36-dim, strong regularization) is a legitimate out-of-distribution model with ECE=0.014~0.023 -- still excellent calibration by general standards (ECE < 0.025), but 4.5-7.3x higher than the baseline's near-zero in-sample ECE.

**Structural issue:** The quality gate compares conservative OOF predictions against baseline OOF predictions on the **same data the baseline was trained on**. This creates a structurally impossible bar -- the baseline has an unfair advantage due to memorization.

### Stage 2: Quality Gate Steps -- SKIPPED

Stage 2 was not executed because Stage 1 failed. Per D-04 (no retry policy), the orchestration CLI records FAIL and exits.

## QUAL-01~04 Runtime Status

| Requirement | Description | Runtime Status | Evidence |
|-------------|-------------|---------------|----------|
| QUAL-01 | OOFHealthValidator PASS | NOT EXECUTED | Stage 2 skipped after Stage 1 FAIL |
| QUAL-02 | FeatureRoutingAudit PASS | NOT EXECUTED | Stage 2 skipped after Stage 1 FAIL |
| QUAL-03 | DeploymentGateEvaluator PASS | NOT EXECUTED | Stage 2 skipped after Stage 1 FAIL |
| QUAL-04 | ROI recovery trend confirmed | NOT EXECUTED | No shadow comparison produced |

## 3-Label Verdict

| Label | Value | Rationale |
|-------|-------|-----------|
| Quality Gate | **FAIL** | Stage 1 FAIL: no surfaces deployed. All C candidates fail ECE gate. |
| ROI Trend | **unknown** | Shadow comparison not produced. Cannot compute ROI trend. |
| Deployment | **not_deployable** | Quality gate FAIL per 3-label framework (D-03) |

## Artifacts Produced

| Artifact | Path | Status |
|----------|------|--------|
| Manifest | data/models-backtest-mawc-conservative/manifest.json | EXISTS (all deployed=false) |
| Retrain summary | data/models-backtest-mawc-conservative/retrain_summary.md | EXISTS |
| HTML report | data/models-backtest-mawc-conservative/mawc_conservative_report.html | EXISTS |
| Quality gate result | data/backtest/phase46_quality_gates/phase46_quality_gate_result.json | NOT PRODUCED (Stage 1 FAIL exit) |

## Next Steps

Per D-04 (no retry policy), Phase 46 does not retry or adjust models. Options:

1. **Fix the quality gate baseline comparison** -- Use a fair baseline (e.g., OOF cross-validated predictions from an independent model, or apply same regularization to baseline before comparison)
2. **Loosen ECE tolerance** -- Current 10% relative tolerance is insufficient for the in-sample vs out-of-sample gap. Consider absolute ECE threshold (e.g., ECE < 0.03 = PASS)
3. **Phase 45b** -- Return to Phase 45 to redesign the quality gate with an apples-to-apples comparison methodology
4. **Skip Stage 1 gate** -- Accept the conservative variant without gate and proceed directly to Stage 2 shadow comparison for the real ROI assessment

## Implementation Verification (Pre-Runtime)

| # | Must-Have | Status |
|---|-----------|--------|
| 1 | QualityGateOrchestrator 2-stage flow | VERIFIED (30/30 tests pass) |
| 2 | Skip/resume via artifact detection | VERIFIED |
| 3 | 3-label framework (quality_gate, roi_trend, deployment) | VERIFIED |
| 4 | JSON + Markdown result output | VERIFIED |
| 5 | CLI with 9 arguments + --stage/--force/--report | VERIFIED |
| 6 | RUNBOOK with manual reproduction commands | VERIFIED |
| 7 | VERIFICATION checklist with QUAL-01~04 | VERIFIED |
| 8 | v2.2-MILESTONE-SUMMARY with 11 requirements | VERIFIED |
| 9 | All key links wired (6 imports verified) | VERIFIED |

---
_Verified: 2026-06-01_
_Runtime execution: Stage 1 FAIL, Stage 2 NOT EXECUTED_
