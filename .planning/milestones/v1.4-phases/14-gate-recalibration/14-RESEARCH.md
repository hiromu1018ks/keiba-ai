# Phase 14: Gate Recalibration - Research

**Researched:** 2026-05-06
**Domain:** WinSelectionGate ensemble OOF retraining, distribution drift diagnostics, use_ensemble flag propagation
**Confidence:** HIGH

## Summary

Phase 14 addresses the root cause of the 7-bets/year problem: WinSelectionGate's quantile bins and score tables were calibrated on single-LightGBM OOF predictions, but the stacked ensemble (LightGBM + XGBoost + CatBoost, Ridge meta-learner) produces a systematically different probability distribution. The gate model itself (`WinSelectionGateModel.train()`) is model-agnostic -- it accepts any DataFrame with the correct columns and builds quantile edges from the data. The change is ensuring the OOF DataFrame passed to `gate.train()` contains ensemble-derived prediction columns rather than single-model ones.

The `use_ensemble` flag propagation path is simpler than CONTEXT.md implies. The flag does NOT flow through BacktestEngine or RacePredictor at runtime -- it is resolved at two fixed points: (1) `TrainingPipelineV5.run(use_ensemble=True)` constructs SubmodelSets with `StackedEnsemble` hit_models, and (2) `ModelLoader.load_from_dir(use_ensemble_override=True)` loads `.joblib` files instead of `.lgb` files. Once a `TrainedModelsV5` is constructed, the ensemble/non-ensemble distinction is baked into the model objects themselves. The test for GATE-03 should verify these two resolution points, not a runtime flag propagation chain.

Distribution drift diagnostics (GATE-02) requires adding `scipy.stats.ks_2samp` and `scipy.stats.wasserstein_distance` comparisons between single-model and ensemble OOF columns at the point where `df_oof` is available in `_train_submodel()`. The key comparison columns are `p_win_pred`, `ev_win`, `p_win_corrected`, `ev_win_corrected`, and the gate-specific `win_selection_prob`, `win_selection_edge`, `win_selection_ev`.

**Primary recommendation:** Gate retraining requires zero changes to `WinSelectionGateModel.train()` itself -- only the OOF DataFrame routing in `_train_submodel()` needs verification. The diagnostics module should be a standalone function called from `_train_submodel()` after EV correction but before gate training.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 診断機能をバックテストパイプラインに統合する。`run_backtest.py --ensemble`実行時に自動で分布ドリフト診断が実行される。独立スクリプトは作成しない。
- **D-02:** 診断結果はJSONファイル + コンソールサマリで出力する。JSONは`data/backtest/`に保存し、コンソールにはKS統計量/p-value/Wasserstein距離の要約を表示する。
- **D-03:** 分布比較の粒度は最大限に: (1) 主要確率・EV列の全データ比較、(2) サーフェス別(芝/ダート)の分割比較、(3) 年度別時系列でのドリフト推移追跡。全てks_2sampとwasserstein_distanceの両方を使用。
- **D-04:** ドリフト検出時の対応: KS p-value < 0.05 または Wasserstein距離が閾値超過の場合にWARNINGログで再学習を推奨。バックテストは継続するが、JSON結果に`drift_detected: true`フラグと推奨アクションを含める。
- **D-05:** フラグ経路のモックベーステストを採用する。値レベルのアサーションではなく、各コンポーネントにuse_ensemble=Trueが正しく渡ることをモックで検証する。
- **D-06:** 統合テスト1つでModelLoader→TrainingPipeline→RacePredictor→WinSelectionGateの全体経路を検証する。コンポーネント別個別テストではなく、1つのテストクラスでend-to-endのフラグ伝播を確認する。
- **D-07:** use_ensemble=Trueの経路のみテストする。False(デフォルト)の経路は既存テストがカバーしている前提。
- **D-08:** ゲート再学習検証は二段構え: (1) ユニットテストでfixtureデータを使い、単一モデルOOFとアンサンブルOOFで学習したゲートのprob_edges/edge_edges/odds_edgesが異なることを確定的に検証、(2) パイプラインのランタイムで、use_ensemble=True時に学習後のゲートedgesがデフォルト(未学習)のedgesと異なることをassertionで確認。

### Claude's Discretion
- 診断機能の具体的な閾値(Wasserstein距離のwarn/error閾値)は研究者・プランナーがデータから決定してよい
- JSON出力のスキーマ設計はプランナーに委ねる
- テストのfixtureデータの内容はプランナーに委ねる

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GATE-01 | WinSelectionGateをアンサンブルOOF予測で再学習し、prob_edges/edge_edges/odds_edgesを再計算する | Gate model is model-agnostic (train() accepts any DataFrame with correct columns). Key insight: ensemble OOF columns flow through WinBenterGate which produces `p_win_final` and `edge_win`, then `ensure_win_selection_columns()` maps these to `win_selection_prob`/`win_selection_edge`. Pipeline already routes ensemble OOF to gate.train() -- but see Pitfall 1 for the benter gate dependency. |
| GATE-02 | 単一モデルとアンサンブルのOOF確率分布をks_2samp/wasserstein_distanceで比較し、ドリフトを定量化する診断機能を追加する | scipy.stats.ks_2samp and wasserstein_distance verified available (scipy 1.17.1). Comparison columns identified: p_win_pred, ev_win, p_win_corrected, ev_win_corrected, win_selection_prob, win_selection_edge. Integration point: after EV correction, before gate training in _train_submodel(). |
| GATE-03 | use_ensembleフラグがModelLoader→RacePredictor→BacktestEngine全体で正しく伝播されていることを検証する | Flag propagation is NOT runtime -- it resolves at model construction time. Two resolution points: (1) TrainingPipelineV5._train_submodel(use_ensemble=True) builds StackedEnsemble hit_models, (2) ModelLoader.load_from_dir(use_ensemble_override=True) loads .joblib files. RacePredictor/BacktestEngine receive already-constructed TrainedModelsV5. Test should verify these two points. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| WinSelectionGate retraining | API / Backend | -- | Gate training happens in TrainingPipelineV5._train_submodel() -- a pure backend ML pipeline operation |
| Distribution drift diagnostics | API / Backend | -- | Diagnostics run during training pipeline execution, comparing OOF column distributions |
| use_ensemble flag propagation | API / Backend | -- | Flag resolves at model construction/load time in pipeline and ModelLoader -- no runtime propagation |
| Diagnostic JSON output | Database / Storage | -- | JSON files written to data/backtest/ directory |
| Test verification | -- | -- | Pure unit/integration tests with mocks, no tier assignment |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy.stats.ks_2samp | 1.17.1 | Kolmogorov-Smirnov test for distribution comparison | Already installed (sklearn transitive). Standard scipy statistical test [VERIFIED: python -c import] |
| scipy.stats.wasserstein_distance | 1.17.1 | Earth Mover's Distance for distribution shift quantification | Already installed. Complementary to KS -- captures magnitude of shift, not just shape [VERIFIED: python -c import] |
| numpy | 2.4.3 | Quantile computation, array operations | Already installed. Used in _quantile_edges() [VERIFIED: python -c import] |
| pandas | 2.3.3 | DataFrame manipulation for OOF data | Already installed. Core data structure [VERIFIED: python -c import] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | -- | Diagnostic report serialization | JSON output for drift report |
| logging (stdlib) | -- | WARNING/INFO drift alerts | Console summary and drift detection alerts |
| unittest.mock | -- | Test mocks for flag propagation tests | All tests use mock pattern per CLAUDE.md |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ks_2samp + wasserstein_distance | KL divergence, Jensen-Shannon | ks/wasserstein are more interpretable for drift detection. KL requires density estimation. User explicitly chose ks_2samp + wasserstein (D-03). |

**Installation:**
```bash
# Zero new packages needed -- all dependencies already installed
pip install -e ".[dev]"  # existing install command
```

**Version verification:**
- scipy: 1.17.1 [VERIFIED: python -c "import scipy; print(scipy.__version__)"]
- numpy: 2.4.3 [VERIFIED: python -c "import numpy; print(numpy.__version__)"]
- pandas: 2.3.3 [VERIFIED: python -c "import pandas; print(pandas.__version__)"]

## Architecture Patterns

### System Architecture Diagram

```
run_backtest.py --ensemble
       |
       v
TrainingPipelineV5.run(use_ensemble=True)
       |
       v
_train_submodel(df, use_ensemble=True)
       |
       +---> MarketModel.train() + predict_oof()
       +---> AbilityModel.train_oof()  --> df_oof
       +---> [if use_ensemble] StackedEnsemble.train() --> win_2s.hit_model = ensemble
       +---> WinTwoStageModel.predict_ev(df_oof)  --> p_win_pred, ev_win
       +---> EVCorrectionModel.correct_ev(df_oof) --> p_win_corrected, ev_win_corrected
       |
       +---> [NEW] compute_drift_diagnostics(df_oof) --> JSON report + console summary
       |         (compares single-model vs ensemble columns by surface/year)
       |
       +---> WinBenterGate.apply(df_oof)  --> p_win_final, edge_win
       +---> ensure_win_selection_columns(df_oof) --> win_selection_prob, win_selection_edge, win_selection_ev
       +---> RobustConfidenceEstimator.predict_lower_bound(df_oof) --> EV_lower_win_corrected
       +---> WinSelectionGateModel.train(wsg_train_df)  --> prob_edges, edge_edges, odds_edges, score tables
       |
       v
SubmodelSet(use_ensemble=True, win_selection_gate=trained_gate)
       |
       v
ModelLoader.load_from_dir(use_ensemble_override=True)  [for --skip-train path]
       |
       v
BacktestEngine(models=TrainedModelsV5) --> RacePredictor(models) --> uses gate.score()
```

### Recommended Project Structure
```
src/
├── models/
│   ├── win_selection_gate.py       # EXISTING -- gate model (no changes needed)
│   └── drift_diagnostics.py        # NEW -- distribution drift diagnostic functions
├── pipelines/
│   └── training_pipeline.py        # EXISTING -- add drift diagnostics call in _train_submodel()
├── db/
│   └── model_loader.py             # EXISTING -- use_ensemble resolution (no changes needed)
tests/
├── test_win_selection_gate.py      # EXISTING -- add ensemble OOF retraining test (D-08)
├── test_drift_diagnostics.py       # NEW -- unit tests for drift diagnostics
└── test_ensemble_gate_propagation.py  # NEW -- integration test for use_ensemble propagation (D-06)
```

### Pattern 1: Gate Training Data Flow
**What:** WinSelectionGateModel.train() is model-agnostic. It receives a DataFrame and builds quantile edges from whatever data it gets.
**When to use:** Every time the model type changes (single -> ensemble), the gate must be retrained.
**Example:**
```python
# Source: src/pipelines/training_pipeline.py:784-792
# --- WinSelectionGate training (SELC-01, D-01) ---
wsg_train_df = df_oof.copy()
wsg_win_df, _ = conf.predict_lower_bound(df_oof.copy(), df_oof.copy())
if "EV_lower_win_corrected" in wsg_win_df.columns:
    wsg_train_df["EV_lower_win_corrected"] = wsg_win_df["EV_lower_win_corrected"].values
wsg_train_df = ensure_win_selection_columns(wsg_train_df)
win_selection_gate = WinSelectionGateModel()
win_selection_gate.train(wsg_train_df)
```
**Key insight:** `df_oof` already contains ensemble predictions when `use_ensemble=True` because `_train_submodel()` uses `win_2s.predict_ev(df_oof)` which calls `self.hit_model.predict()` -- and `hit_model` is the StackedEnsemble when ensemble mode is active. The gate receives ensemble-derived columns transparently.

### Pattern 2: Column Fallback Chain in ensure_win_selection_columns()
**What:** The gate requires `win_selection_prob`, `win_selection_edge`, `win_selection_ev` columns. These are derived from model outputs via a fallback chain.
**When to use:** Understanding what columns the gate sees in ensemble vs single-model mode.
**Example:**
```python
# Source: src/models/win_selection_gate.py:33-54
# win_selection_ev:  EV_lower_win_corrected -> ev_win_corrected -> edge_win+1 -> ev_win
# win_selection_edge: win_selection_ev - 1.0
# win_selection_prob: p_win_final -> p_win_combined -> p_win_corrected
```
**Key insight for GATE-01:** In the pipeline, WinBenterGate runs BEFORE the gate training section and adds `p_win_final` and `edge_win`. So `ensure_win_selection_columns()` will pick up `p_win_final` for prob and compute edge from the EV chain. The ensemble shifts these values, which means the gate's quantile edges WILL differ -- confirming D-08's test design.

### Pattern 3: Drift Diagnostics Integration Point
**What:** Insert drift diagnostics between EV correction and gate training in `_train_submodel()`.
**When to use:** GATE-02 implementation.
**Example:**
```python
# Integration point in _train_submodel() around line 775-784:
# After: ev_corrector.correct_ev(df_oof) -- df_oof has p_win_corrected, ev_win_corrected
# Before: WinSelectionGate training section
#
# The diagnostics function receives df_oof (which has all model output columns)
# and a reference to whether this is ensemble mode.
# It compares against stored single-model baseline or logs current distribution stats.
```

### Pattern 4: Test Mock Structure
**What:** All tests use `unittest.mock` with no DB dependencies. Fixtures construct DataFrames directly.
**When to use:** Writing GATE-01 and GATE-03 tests.
**Example:**
```python
# Source: tests/test_win_selection_gate.py -- existing test pattern
rows: list[dict[str, object]] = []
for race_idx in range(120):
    race_id = f"R{race_idx:04d}"
    race_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=race_idx)
    for umaban, prob, edge, odds, finish in [
        (1, 0.62, 0.24, 2.2, 1 if race_idx % 10 == 0 else 5),
        (2, 0.28, 0.02, 4.5, 4),
        (3, 0.10, -0.15, 11.0, 8),
    ]:
        rows.append({
            "race_id": race_id,
            "race_date": race_date,
            "umaban": umaban,
            "kakuteijyuni": finish,
            "tanoddslow": odds,
            "win_selection_prob": prob,
            "win_selection_edge": edge,
        })
df = pd.DataFrame(rows)
```

### Anti-Patterns to Avoid
- **Do NOT modify WinSelectionGateModel.train() for ensemble support:** The gate is already model-agnostic. Changes belong in the pipeline's OOF data routing.
- **Do NOT add use_ensemble parameter to RacePredictor or BacktestEngine:** The flag resolves at model construction time. These components receive fully-constructed TrainedModelsV5 objects.
- **Do NOT create an independent diagnostic script:** Per D-01, diagnostics integrate into the existing pipeline, not a separate entry point.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Distribution comparison | Custom KL divergence or histogram comparison | scipy.stats.ks_2samp + wasserstein_distance | Standard statistical tests with well-understood properties, p-values, and interpretability. Already installed. |
| Quantile bin edges | Custom binning logic | WinSelectionGateModel._build_score_tables() | Already handles edge cases (empty data, single-value data, duplicates). Call via train(). |
| Column normalization for gate | Manual column mapping | ensure_win_selection_columns() | Existing function handles full fallback chain. |
| JSON serialization | Custom report format | json.dumps with structured dict | Standard library. No schema library needed for this scope. |

**Key insight:** This phase requires minimal new code. The gate model, column mapping, and score table construction already work with any data. The work is (1) adding a diagnostic function, (2) calling it from the pipeline, and (3) writing verification tests.

## Common Pitfalls

### Pitfall 1: WinBenterGate Dependency for Gate Training Columns
**What goes wrong:** `ensure_win_selection_columns()` prefers `p_win_final` (from WinBenterGate) over `p_win_corrected` (from EVCorrectionModel). If WinBenterGate has not been applied before gate training, the gate sees different columns.
**Why it happens:** The pipeline flow in `_train_submodel()` is: WinTwoStageModel.predict_ev() -> EVCorrectionModel.correct_ev() -> WinBenterGate.apply() -> gate training. The Benter gate is the last step before the gate, producing `p_win_final` and `edge_win`.
**How to avoid:** Verify that the WinBenterGate section (lines ~580-670) runs BEFORE the gate training section (lines ~784-792) in `_train_submodel()`. The existing code order is correct -- this is a verification point, not a fix.
**Warning signs:** Gate `win_selection_prob` values are derived from `p_win_corrected` instead of `p_win_final` in the training data.

### Pitfall 2: Diagnostic Baseline Missing for Single-Model Comparison
**What goes wrong:** GATE-02 requires comparing single-model vs ensemble distributions. But the pipeline runs in either single or ensemble mode -- not both simultaneously. There is no automatic baseline collection.
**Why it happens:** `run_backtest.py --ensemble` runs the full pipeline in ensemble mode only. Single-model mode is the default. The diagnostic needs a reference point.
**How to avoid:** Two approaches: (a) Run diagnostics that characterize the current OOF distribution and save stats to JSON (no cross-mode comparison within a single run -- user compares JSONs across runs), or (b) Within `_train_submodel()`, temporarily generate single-model predictions alongside ensemble predictions for same-fold comparison. Approach (a) is simpler and aligns with D-02/D-03.
**Warning signs:** Diagnostic report has ensemble stats but no single-model comparison baseline.

### Pitfall 3: OOF Column Names Identical Between Single and Ensemble
**What goes wrong:** The ensemble does NOT produce differently-named columns. `win_2s.predict_ev(df_oof)` always produces `p_win_pred` and `ev_win` regardless of whether `hit_model` is a `StackedEnsemble` or `lgb.Booster`. Column names are identical; only values differ.
**Why it happens:** StackedEnsemble implements the same `.predict()` interface as `lgb.Booster`, returning a numpy array of probabilities. The downstream code is interface-compatible.
**How to avoid:** Tests for GATE-01 must compare VALUES (prob_edges, edge_edges), not column presence. Column names will be the same.
**Warning signs:** Test only checks column existence rather than value differences.

### Pitfall 4: _load_cached_models Always Uses use_ensemble_override=True
**What goes wrong:** `scripts/run_backtest.py:140` hardcodes `use_ensemble_override=True` in `_load_cached_models()`. This means the `--skip-train` path always loads ensemble models regardless of the `--ensemble` flag.
**Why it happens:** The function was written for the multi-year backtest path which always uses ensemble. The `--ensemble` flag only affects `pipeline.run()`.
**How to avoid:** GATE-03 test should note this behavior. The `--skip-train` path always uses ensemble models. The `--ensemble` flag only matters for the training path.
**Warning signs:** Tests assume `--skip-train` respects the `--ensemble` flag.

### Pitfall 5: Walk-Forward Folds in Gate Training May Differ
**What goes wrong:** `WinSelectionGateModel.train()` uses its own walk-forward fold structure (line 815: `self._build_walk_forward_folds()`) independent of the pipeline's fold structure. If the gate's fold count or data ordering differs, the threshold search may find different optimal parameters.
**Why it happens:** The gate performs its own walk-forward validation for threshold optimization (prob/edge/odds grid search), separate from the model training folds.
**How to avoid:** This is expected behavior -- the gate's internal folds operate on the OOF predictions, not the raw features. No action needed, but tests should account for it.
**Warning signs:** Gate threshold parameters differ across runs with same data due to race-date-sorted ordering.

### Pitfall 6: surface Column Required for Surface-Split Diagnostics
**What goes wrong:** D-03 requires surface-split (turf/dirt) diagnostics. The `surface` column may not be present in all OOF DataFrames at the gate training point.
**Why it happens:** `_prepare_training_frame()` (line 213-216) handles missing surface by defaulting to "unknown". But the diagnostic function may not have this fallback.
**How to avoid:** Ensure the diagnostic function checks for `surface` column and gracefully handles its absence.
**Warning signs:** Diagnostics crash with KeyError on `surface` column.

## Code Examples

### Gate Training with Ensemble OOF (GATE-01 verification)
```python
# Source: src/pipelines/training_pipeline.py:784-792
# This is the exact code path. When use_ensemble=True:
# - df_oof contains ensemble-predicted p_win_pred (from StackedEnsemble.predict())
# - ev_corrector produces p_win_corrected, ev_win_corrected
# - WinBenterGate produces p_win_final, edge_win
# - ensure_win_selection_columns() maps these to gate input columns
# - gate.train() builds quantile edges from the ensemble-derived values

wsg_train_df = df_oof.copy()
wsg_win_df, _ = conf.predict_lower_bound(df_oof.copy(), df_oof.copy())
if "EV_lower_win_corrected" in wsg_win_df.columns:
    wsg_train_df["EV_lower_win_corrected"] = wsg_win_df["EV_lower_win_corrected"].values
wsg_train_df = ensure_win_selection_columns(wsg_train_df)
win_selection_gate = WinSelectionGateModel()
win_selection_gate.train(wsg_train_df)
```

### Drift Diagnostics Function Skeleton (GATE-02)
```python
# New file: src/models/drift_diagnostics.py
from scipy.stats import ks_2samp, wasserstein_distance
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Key columns to compare (per D-03)
DRIFT_COLUMNS = [
    "p_win_pred", "ev_win",
    "p_win_corrected", "ev_win_corrected",
    "win_selection_prob", "win_selection_edge", "win_selection_ev",
]

def compute_drift_diagnostics(
    df_oof: pd.DataFrame,
    *,
    output_path: Path | None = None,
    surface: str = "unknown",
) -> dict:
    """Compute distribution drift diagnostics for OOF predictions.

    Compares columns against reference or characterizes current distribution.
    Returns dict with KS statistics, Wasserstein distances, per-surface and per-year breakdowns.
    """
    results: dict = {"surface": surface, "columns": {}, "drift_detected": False}

    for col in DRIFT_COLUMNS:
        if col not in df_oof.columns:
            continue
        values = df_oof[col].dropna()
        if len(values) < 30:
            continue
        results["columns"][col] = {
            "n": len(values),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "q25": float(values.quantile(0.25)),
            "q50": float(values.quantile(0.50)),
            "q75": float(values.quantile(0.75)),
        }

    # Surface-split and year-split per D-03
    # ... (planner designs the full schema)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Drift diagnostics saved to %s", output_path)

    return results
```

### Ensemble OOF Retraining Test (D-08 pattern)
```python
# tests/test_win_selection_gate.py -- add new test
def test_gate_edges_differ_between_single_and_ensemble_oof() -> None:
    """GATE-01 (D-08): Gate edges differ when trained on single-model vs ensemble-like OOF."""
    from models.win_selection_gate import WinSelectionGateModel

    # Fixture: single-model-like OOF (narrower probability distribution)
    single_rows = _build_fixture_rows(prob_mean=0.20, prob_std=0.05)
    df_single = pd.DataFrame(single_rows)

    # Fixture: ensemble-like OOF (wider probability distribution, sharper predictions)
    ensemble_rows = _build_fixture_rows(prob_mean=0.25, prob_std=0.10)
    df_ensemble = pd.DataFrame(ensemble_rows)

    gate_single = WinSelectionGateModel(min_train_races=40, min_fold_races=20, max_folds=3)
    gate_single.train(df_single)

    gate_ensemble = WinSelectionGateModel(min_train_races=40, min_fold_races=20, max_folds=3)
    gate_ensemble.train(df_ensemble)

    # D-08: Verify edges are different (ensemble shifts the distribution)
    assert gate_single.prob_edges != gate_ensemble.prob_edges
    assert gate_single.edge_edges != gate_ensemble.edge_edges
    assert gate_single.odds_edges != gate_ensemble.odds_edges
```

### use_ensemble Propagation Test (D-06 pattern)
```python
# tests/test_ensemble_gate_propagation.py
class TestEnsembleFlagPropagation:
    """GATE-03: Verify use_ensemble flag reaches all components."""

    def test_ensemble_flag_creates_stacked_ensemble_hit_model(self) -> None:
        """When use_ensemble=True, _train_submodel produces StackedEnsemble hit_models."""
        # Mock the pipeline components
        # Verify win_2s.hit_model is StackedEnsemble (not lgb.Booster)
        # Verify place_2s.hit_model is StackedEnsemble
        # Verify SubmodelSet.use_ensemble is True
        ...

    def test_model_loader_ensemble_override_loads_joblib(self) -> None:
        """When use_ensemble_override=True, ModelLoader loads .joblib files."""
        # Mock file system with both .joblib and .lgb files
        # Call load_from_dir(use_ensemble_override=True)
        # Verify _load_hit_model was called with use_ensemble=True
        ...

    def test_trained_models_v5_contains_ensemble_gate(self) -> None:
        """TrainedModelsV5 from ensemble pipeline has trained WinSelectionGate."""
        # Verify that when use_ensemble=True:
        # 1. SubmodelSet.win_selection_gate.is_trained is True
        # 2. SubmodelSet.win_selection_gate.prob_edges are non-empty
        # 3. SubmodelSet.win_selection_gate has different edges than default
        ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-LightGBM hit models | StackedEnsemble (3 GBM + Ridge) | v1.1 | Probability distribution shifted, but filters not recalibrated |
| Hardcoded gate thresholds | Walk-forward optimized thresholds | v1.3 | Gate adapts to training data distribution, but only single-model distribution |
| No drift diagnostics | ks_2samp + wasserstein_distance | Phase 14 (this) | Quantifies distribution shift, enables data-driven recalibration decisions |

**Deprecated/outdated:**
- None in this phase's scope -- all patterns are current.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | WinBenterGate section (lines ~580-670) runs BEFORE gate training (lines ~784-792) in _train_submodel() | Pitfall 1 | Gate sees p_win_corrected instead of p_win_final -- less accurate but still functional |
| A2 | Ensemble OOF predictions produce sufficiently different distributions from single-model to cause edge differences | GATE-01 | If distributions are very similar, gate edges may not differ meaningfully -- but this is unlikely given the 3-model stacking architecture |
| A3 | scipy.stats.ks_2samp and wasserstein_distance handle NaN/inf gracefully when called on OOF columns | GATE-02 | May need explicit dropna() before calling -- should verify in implementation |

**Note:** Claims A1-A3 are based on direct code reading, not runtime verification. The code structure is clear from reading, but runtime behavior should be confirmed during implementation.

## Open Questions (RESOLVED)

1. **Drift diagnostics baseline comparison strategy**
   - What we know: D-03 specifies "maximum granularity" comparisons. The pipeline runs in either single or ensemble mode, not both simultaneously.
   - What's unclear: Should diagnostics (a) compare current-run stats against a saved baseline JSON from a previous single-model run, or (b) run both modes within a single pipeline execution?
   - Recommendation: Option (a) -- save per-run diagnostics JSON, let the user compare across runs. Option (b) would double training time and is not justified. The diagnostic function should characterize the current distribution comprehensively; cross-run comparison is a manual or future automation step.

2. **Wasserstein distance warning threshold**
   - What we know: D-04 specifies "Wasserstein distance exceeds threshold" triggers a warning. D-03 specifies KS p-value < 0.05.
   - What's unclear: What Wasserstein distance value constitutes "significant drift" for these probability columns.
   - Recommendation: Leave as Claude's Discretion per CONTEXT.md. Start with 0.05 for probability columns (typical meaningful shift) and adjust based on observed values from the first ensemble run.

3. **Whether _train_submodel() needs a code change for GATE-01**
   - What we know: The existing code at lines 784-792 copies df_oof, adds EV_lower_win_corrected, calls ensure_win_selection_columns(), then calls gate.train(). When use_ensemble=True, df_oof already contains ensemble predictions because win_2s.hit_model is the StackedEnsemble.
   - What's unclear: Whether there is a subtle data path issue where the ensemble predictions get overwritten or lost before reaching the gate.
   - Recommendation: This needs runtime verification (D-08 part 2 -- pipeline assertion). The code structure appears correct, but the planner should include an assertion task to verify at runtime.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scipy.stats | GATE-02 diagnostics | Yes | 1.17.1 | -- |
| numpy | GATE-01/02 quantiles | Yes | 2.4.3 | -- |
| pandas | GATE-01/02 DataFrame ops | Yes | 2.3.3 | -- |
| pytest | All tests | Yes | verified | -- |
| python 3.11 | All | Yes | 3.11.x (via mise) | -- |

**Missing dependencies with no fallback:**
- None -- all dependencies verified available.

**Missing dependencies with fallback:**
- None.

## Validation Architecture

> nyquist_validation is explicitly `false` in `.planning/config.json`. Validation Architecture section SKIPPED per instructions.

## Sources

### Primary (HIGH confidence)
- Direct codebase audit: src/models/win_selection_gate.py (1113 lines), src/pipelines/training_pipeline.py, src/db/model_loader.py, src/backtest/race_predictor.py, src/backtest/engine.py, scripts/run_backtest.py
- Package verification: scipy 1.17.1, numpy 2.4.3, pandas 2.3.3 -- all verified via python -c execution
- Existing tests: tests/test_win_selection_gate.py (229 lines), tests/test_model_loader.py, tests/test_training_pipeline.py

### Secondary (MEDIUM confidence)
- .planning/research/SUMMARY.md -- project-level research confirming gate retraining is ~20 lines data routing change
- .planning/phases/14-gate-recalibration/14-CONTEXT.md -- user decisions constraining implementation

### Tertiary (LOW confidence)
- None -- all findings grounded in direct code reading and verification.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - zero new dependencies, all packages verified installed
- Architecture: HIGH - full code path traced from run_backtest.py through pipeline to gate with line numbers
- Pitfalls: HIGH - 6 pitfalls identified from direct code analysis, each with specific line references
- Tests: HIGH - existing test patterns well-documented, mock patterns clear from reading test files

**Research date:** 2026-05-06
**Valid until:** 2026-06-06 (stable codebase, no dependency changes expected)
