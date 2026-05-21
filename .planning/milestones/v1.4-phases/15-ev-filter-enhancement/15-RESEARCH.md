# Phase 15: EV Filter Enhancement - Research

**Researched:** 2026-05-06
**Domain:** Dynamic EV threshold calibration + EV estimation accuracy diagnostics
**Confidence:** HIGH

## Summary

Phase 15 replaces the hardcoded EV_lower >= 1.0 filter in `get_win_candidates()` with a dynamic threshold computed from the ensemble OOF winner distribution. The root cause of 3,594 exclusions is twofold: (1) `RobustConfidenceEstimator.calibrate()` uses single-LightGBM residuals, producing over-wide conformal prediction intervals that push EV_lower artificially low; (2) the fixed 1.0 threshold is blind to the actual distribution of profitable bets. The fix involves rerouting calibration data from ensemble OOF residuals (approximately 20-line change in pipeline data routing) and computing a percentile-based threshold from positive-edge OOF winners per surface.

The second requirement (EVF-02) adds a depth diagnostic module (`ev_diagnostics.py`) that evaluates EV prediction quality through ECE, Brier score decomposition, reliability diagrams, and temporal drift tracking. This follows the exact pattern established by Phase 14's `drift_diagnostics.py` -- standalone module with `compute_*()` function + `console_summary()`, integrated into `_train_submodel()` under `use_ensemble=True` guard with its own `TimingContext`.

**Primary recommendation:** Two-wave plan -- Wave 1: Recalibrate `RobustConfidenceEstimator` with ensemble residuals + replace fixed threshold with dynamic percentile threshold. Wave 2: Create `ev_diagnostics.py` module and integrate into pipeline.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: 複合方式を採用 -- Percentile方式(25th percentile of positive-edge OOF winners)を初期値とし、Phase 17 Optunaの14次元探索に閾値を15次元目として追加して最適化する
- D-02: 閾値はSurface別(芝/ダート)で各OOF winners分布から独立に計算する。Optunaの探索次元は16次元(14 + サーフェス別2閾値)
- D-03: EV_lowerがNaNの場合、サーフェス別のデフォルト閾値にフォールバックする
- D-04: 深度診断(学術的)を実装: EV予測vs実際払戻の相関/RMSE + Reliability diagram(ECE) + Brier score分解 + 時系列ドリフト追跡
- D-05: パイプライン統合 -- run_backtest.py --ensemble実行時に自動でEV診断が実行される。独立スクリプトは作成しない
- D-06: RobustConfidenceEstimatorをアンサンブルOOF残差で再calibrateする(~20行変更)

### Claude's Discretion
- EV診断モジュールの具体的なJSONスキーマ設計
- Percentile計算の実装詳細(どのOOFサブセットを正のエッジ勝利馬とするか)
- Brier score分解の実装方法(reliability/uncertainty/resolution)
- Reliability diagramのビン数と表示形式
- 時系列ドリフト追跡の粒度(年度別/四半期別)
- サーフェス別フォールバック閾値の具体的な計算方法
- テスト戦略(モックベース、既存パターン踏襲)

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVF-01 | EV_lower閾値を固定1.0からアンサンブルOOF分布の分位点に基づく動的閾値に変更する | RobustConfidenceEstimator再キャリブレーション(D-06) + percentile計算(D-01) + race_predictor.py行434-451のフィルター変更 |
| EVF-02 | OOF EV推定値と実際の払戻額を比較し、EV推定精度を評価する診断機能を追加する | ev_diagnostics.py新規作成(D-04) + パイプライン統合(D-05) |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Conformal再キャリブレーション | ML Pipeline (training_pipeline.py) | -- | キャリブレーションは学習時のデータルーティング問題。pipelineがOOF DataFrameを構築しRobustConfidenceEstimatorに渡す |
| 動的閾値計算 | ML Pipeline (training_pipeline.py) | -- | OOF分布からのパーセンタイル計算は学習時実行。結果はSubmodelSetに格納 |
| EV_lowerフィルター適用 | Backtest Engine (race_predictor.py) | -- | 推論時フィルタリング。動的閾値をSubmodelSetから取得して適用 |
| EV診断モジュール | ML Models (models/ev_diagnostics.py) | ML Pipeline (統合ポイント) | モジュール自体はmodels/層。パイプライン統合はtraining_pipeline.py |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.4.3 | np.quantile, binning operations for percentile threshold and ECE | Already installed, core dependency |
| pandas | (transitive) | DataFrame operations for OOF filtering | Already installed |
| scipy.stats | 1.17.1 | stats.pearsonr for EV correlation diagnostic | Already installed, used in drift_diagnostics.py |
| sklearn.metrics | 1.8.0 | brier_score_loss, calibration_curve | Already used in win_benter_gate.py, training_pipeline.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | -- | JSON diagnostic output | Module output to data/backtest/ |
| logging (stdlib) | -- | Console diagnostic summary | console_summary() pattern from drift_diagnostics.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual ECE implementation | MAPIE.expected_calibration_error | MAPIE not installed; manual ~10 lines is trivial. [VERIFIED: pip show mapie returns not found] |
| Manual Brier decomposition | torchmetrics.BrierScore | PyTorch dependency inappropriate for this project. numpy implementation is ~20 lines. [ASSUMED] |
| Reliability diagram from scratch | sklearn.calibration_curve | Already used in win_benter_gate.py:342. Same pattern. [VERIFIED: code inspection] |

**Installation:**
```bash
# No new packages required -- all dependencies already installed
pip install -e ".[dev]"  # existing install command
```

**Version verification:** All packages verified installed:
- numpy 2.4.3, scipy 1.17.1, sklearn 1.8.0 [VERIFIED: python -c execution]

## Architecture Patterns

### System Architecture Diagram

```
[Training Pipeline: _train_submodel()]
         |
         v
[Ensemble OOF DataFrame (df_oof)]
         |
         |--- win_calib_df = {ev_win_corrected, actual_ev_win}
         |         |
         |         v
         |    [RobustConfidenceEstimator.calibrate()]  <--- D-06: ensemble residuals (not single-LightGBM)
         |         |
         |         v
         |    conf.predict_lower_bound(df_oof) --> EV_lower_win_corrected
         |         |
         |         v
         |    [Dynamic Threshold Computation]  <--- D-01: 25th pct of positive-edge OOF winners
         |         |
         |         |--- Surface: turf --> ev_threshold_turf
         |         |--- Surface: dirt --> ev_threshold_dirt
         |         v
         |    [SubmodelSet stores threshold] ---> SubmodelSet.ev_lower_threshold (new field)
         |
         |--- [EV Diagnostics Module]  <--- D-04/D-05: compute_ev_diagnostics()
         |         |
         |         |--- ECE (Expected Calibration Error)
         |         |--- Brier Score Decomposition (reliability/uncertainty/resolution)
         |         |--- Reliability Diagram data
         |         |--- Temporal Drift (yearly)
         |         v
         |    data/backtest/ev_diagnostics_{surface}.json
         |    console_summary()
         |
         v
[Backtest Engine: RacePredictor.get_win_candidates()]
         |
         v
    ev_lower = EV_lower_win_corrected
    threshold = submodel.ev_lower_threshold  <--- D-03: NaN fallback to surface default
    mask = ev_lower >= threshold
         |
         v
    [Filtered candidates --> ranking --> top 2]
```

### Recommended Project Structure
```
src/
├── models/
│   ├── ev_diagnostics.py          # NEW: EV estimation accuracy diagnostic module
│   ├── drift_diagnostics.py       # Phase 14 pattern reference
│   └── robust_confidence_estimator.py  # MODIFIED: no code change (data routing change in pipeline)
├── backtest/
│   └── race_predictor.py          # MODIFIED: dynamic threshold in get_win_candidates()
├── pipelines/
│   └── training_pipeline.py       # MODIFIED: ensemble residual routing + threshold computation + EV diagnostics integration
├── domain/
│   └── models.py                  # MODIFIED: SubmodelSet.ev_lower_threshold field
tests/
├── test_ev_diagnostics.py         # NEW: EV diagnostics tests (mock-based, ~8 tests)
├── test_drift_diagnostics.py      # Phase 14 pattern reference
└── test_race_predictor.py         # MODIFIED: dynamic threshold tests
```

### Pattern 1: Pipeline-Integrated Diagnostics (from Phase 14)
**What:** Standalone diagnostic module with `compute_*()` function + `console_summary()`, called from `_train_submodel()` under `use_ensemble=True` guard with own `TimingContext`.
**When to use:** All new diagnostic modules that run during ensemble training.
**Example:**
```python
# Source: src/models/drift_diagnostics.py + src/pipelines/training_pipeline.py:792-803
# Pattern: module function + pipeline integration
if use_ensemble:
    with TimingContext(f"{surface}/ev_diagnostics"):
        from models.ev_diagnostics import compute_ev_diagnostics, console_summary

        ev_output_path = Path("data/backtest") / f"ev_diagnostics_{surface}.json"
        ev_result = compute_ev_diagnostics(
            df_oof,
            output_path=ev_output_path,
            surface=surface,
        )
        console_summary(ev_result)
```

### Pattern 2: Surface-Specific Analysis (from Phase 14 D-03)
**What:** Split DataFrame by surface column, compute metrics independently for each.
**When to use:** Any analysis that should account for turf/dirt differences.
**Example:**
```python
# Source: Phase 14 pattern, used in drift_diagnostics.py:172-188
for surf in ["turf", "dirt"]:
    surf_df = df_oof[df_oof["surface"] == surf]
    # ... compute metric for surf_df independently ...
```

### Pattern 3: Dynamic Threshold Computation from OOF Winners
**What:** Identify positive-edge winners in OOF data, compute percentile of their EV_lower values as threshold.
**When to use:** Setting the EV_lower threshold.
**Example:**
```python
# D-01: 25th percentile of positive-edge OOF winners, per surface
def _compute_ev_threshold(df_oof: pd.DataFrame, surface: str) -> float:
    """Compute dynamic EV_lower threshold from OOF winner distribution."""
    surf_df = df_oof[df_oof["surface"] == surface]
    # "positive-edge winners": horses that won AND had win_selection_edge > 0
    winners = surf_df[
        (surf_df["kakuteijyuni"] == 1) &
        (surf_df["win_selection_edge"] > 0)
    ]
    if len(winners) < 30:
        # D-03: fallback to surface default
        return 0.8 if surface == "turf" else 0.7  # defaults TBD at Claude's discretion
    return float(winners["EV_lower_win_corrected"].quantile(0.25))
```

### Anti-Patterns to Avoid
- **Single-model residuals for ensemble inference:** RobustConfidenceEstimator.calibrate() must receive ensemble OOF residuals, not single-LightGBM residuals. Using single-model residuals produces over-wide intervals that are the root cause of 3,594 exclusions.
- **Hardcoded threshold in filter:** The >= 1.0 threshold in race_predictor.py:440 must not remain hardcoded. It must read from a configurable source (SubmodelSet field).
- **Skipping NaN handling:** EV_lower can be NaN for races where Conformal Prediction has no calibration data. D-03 requires surface-specific fallback, not fillna(1.0).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Brier score computation | Custom MSE formula | sklearn.metrics.brier_score_loss | Already used in codebase (win_benter_gate.py:246, training_pipeline.py:707) |
| Reliability diagram data | Custom binning | sklearn.calibration.calibration_curve | Already used in win_benter_gate.py:340. Returns fraction_of_positives + mean_predicted_value |
| Distribution comparison | Custom statistics | scipy.stats.ks_2samp, wasserstein_distance | Already used in drift_diagnostics.py. Well-tested |
| Correlation coefficient | Custom formula | scipy.stats.pearsonr | Returns both r and p-value. Handles edge cases |

**Key insight:** The project already has calibration_curve and brier_score_loss in use. ECE and Brier decomposition are straightforward numpy operations on top of these building blocks.

## Common Pitfalls

### Pitfall 1: Recalibration Uses Wrong Residuals
**What goes wrong:** RobustConfidenceEstimator.calibrate() is called with single-LightGBM OOF residuals instead of ensemble residuals. The resulting CP quantile is too large, EV_lower values are too low, and the dynamic threshold is fighting against over-conservative intervals.
**Why it happens:** The current pipeline code (training_pipeline.py:762-773) constructs win_calib_df from df_oof, which after ensemble mode has ensemble predictions. But the residuals come from `ev_win_corrected` vs `actual_ev_win`, where `ev_win_corrected` may still reflect single-model calibration if the EV correction model was trained on single-model data.
**How to avoid:** Verify that when `use_ensemble=True`, the entire OOF chain (hit_model -> predict_ev -> correct_ev -> calibrate) uses ensemble models. The pipeline already replaces hit_model with StackedEnsemble (line 468), so df_oof should have ensemble-derived values by the time calibrate() is called.
**Warning signs:** If post-recalibration EV_lower distribution hasn't shifted significantly from pre-recalibration, the residual source may be wrong.

### Pitfall 2: Threshold Too Aggressive or Too Conservative
**What goes wrong:** 25th percentile of positive-edge winners produces a threshold that still excludes too many candidates (too high) or includes too many unprofitable ones (too low).
**Why it happens:** The OOF winner sample may be small or biased (e.g., mostly favorites with high EV_lower, or mostly longshots with low EV_lower).
**How to avoid:** Log the distribution statistics (n, mean, q25, q50, q75) of positive-edge winners' EV_lower for each surface. Include these in the diagnostic JSON. The Phase 17 Optuna optimization will refine the threshold further.
**Warning signs:** Threshold below 0.5 (too permissive) or above 1.5 (still too restrictive).

### Pitfall 3: NaN Handling Creates Silent Pass-Through
**What goes wrong:** Replacing fillna(1.0) with a surface-specific fallback that is too low (e.g., 0.0) allows all NaN-EV_lower candidates through, defeating the filter's purpose.
**Why it happens:** D-03 says "surface-specific default threshold" but the specific value needs careful selection.
**How to avoid:** Use a conservative fallback (e.g., 0.8 for turf, 0.7 for dirt -- below 1.0 but not dangerously low). Log when fallback is used so it's visible in diagnostics.
**Warning signs:** Sudden spike in bet count without corresponding quality improvement.

### Pitfall 4: ECE Bin Strategy Choice Affects Results
**What goes wrong:** Using "uniform" binning (equal width) produces empty bins for skewed EV distributions. Using "quantile" binning may hide miscalibration in the tails.
**Why it happens:** EV predictions are not uniformly distributed -- they cluster around 1.0 with a long right tail.
**How to avoid:** Use "quantile" strategy (equal-frequency bins) for ECE to ensure each bin has sufficient samples. Use 10 bins as standard. This matches the approach recommended in Guo et al. 2017. [CITED: towardsdatascience.com/ece-visual-explanation]
**Warning signs:** Empty bins in reliability diagram; ECE dominated by a single bin.

### Pitfall 5: Temporal Drift Granularity Too Fine
**What goes wrong:** Quarterly drift tracking produces too few samples per quarter for meaningful statistics, especially for winners-only subsets.
**Why it happens:** JRA runs ~3000 races/year. Winners subset is much smaller (1 per race).
**How to avoid:** Use yearly granularity for temporal drift tracking. This matches the Phase 14 drift_diagnostics.py pattern (year-level breakdown).
**Warning signs:** Per-year sample count below 30 (MIN_SAMPLE_SIZE from drift_diagnostics.py).

## Code Examples

### ECE (Expected Calibration Error) Implementation
```python
# Source: numpy implementation based on Guo et al. 2017
# [CITED: towardsdatascience.com/ece-step-by-step-visual-explanation]
import numpy as np

def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error via equal-frequency (quantile) binning."""
    bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bin_boundaries[0] = -np.inf  # include minimum
    bin_boundaries[-1] = np.inf  # include maximum

    ece = 0.0
    n_total = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)
    return float(ece)
```

### Brier Score Decomposition (Murphy 1973)
```python
# Source: Murphy (1973) decomposition via binned expectations
# [CITED: wikipedia.org/wiki/Brier_score, stats.stackexchange.com/brier-decomposition]
import numpy as np

def _brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Decompose Brier score into reliability, resolution, uncertainty."""
    brier = float(np.mean((y_prob - y_true) ** 2))

    bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bin_boundaries[0] = -np.inf
    bin_boundaries[-1] = np.inf

    n = len(y_true)
    o_bar = y_true.mean()  # base rate

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        n_k = mask.sum()
        if n_k == 0:
            continue
        o_k = y_true[mask].mean()  # observed frequency in bin
        f_k = y_prob[mask].mean()  # predicted probability in bin
        reliability += (n_k / n) * (o_k - f_k) ** 2
        resolution += (n_k / n) * (o_k - o_bar) ** 2

    uncertainty = o_bar * (1 - o_bar)  # base rate variance

    return {
        "brier_score": brier,
        "reliability": float(reliability),  # lower is better (calibration)
        "resolution": float(resolution),    # higher is better (discrimination)
        "uncertainty": float(uncertainty),  # data-dependent (not model-dependent)
    }
```

### Reliability Diagram Data (existing pattern from win_benter_gate.py)
```python
# Source: src/models/win_benter_gate.py:335-350 [VERIFIED: code inspection]
from sklearn.calibration import calibration_curve

def _reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Generate reliability diagram data."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
        "bin_edges": np.linspace(0.0, 1.0, n_bins + 1).tolist(),
    }
```

### Dynamic Threshold Storage in SubmodelSet
```python
# New field in SubmodelSet (src/domain/models.py)
@dataclass
class SubmodelSet:
    # ... existing fields ...
    ev_lower_threshold_turf: float = 1.0   # D-01/D-02: dynamic per-surface threshold
    ev_lower_threshold_dirt: float = 1.0   # D-01/D-02: dynamic per-surface threshold
```

### Dynamic Threshold in get_win_candidates()
```python
# Modified section of src/backtest/race_predictor.py:434-451
# Current: ev_mask = ev_lower.fillna(1.0) >= 1.0
# New:
submodel = self.models.submodels.get(surface_key)
if submodel is not None:
    surface = str(surface_key)
    if surface == "turf":
        threshold = submodel.ev_lower_threshold_turf
    elif surface == "dirt":
        threshold = submodel.ev_lower_threshold_dirt
    else:
        threshold = 1.0  # D-03: unknown surface fallback
else:
    threshold = 1.0
ev_mask = ev_lower.fillna(threshold) >= threshold  # D-03: NaN uses surface default
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded EV_lower >= 1.0 | Percentile-based dynamic threshold from OOF winners | Phase 15 | Reduces exclusions from 3,594 to data-driven level |
| Single-model conformal residuals | Ensemble OOF residuals for calibration | Phase 15 | Narrows confidence intervals to match actual ensemble error |
| No EV estimation quality metrics | ECE + Brier decomposition + reliability diagram + temporal drift | Phase 15 | Enables data-driven EV accuracy assessment |

**Deprecated/outdated:**
- `fillna(1.0)` in race_predictor.py:440 -- replaced by surface-specific fallback threshold (D-03)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Ensemble OOF residuals are already in df_oof when calibrate() is called at line 762-773 | Conformal再キャリブレーション | If residual source is still single-model, the recalibration has no effect |
| A2 | 25th percentile of positive-edge OOF winners produces a reasonable starting threshold | 動的閾値計算 | Threshold may be too high or too low for actual data distribution |
| A3 | Brier decomposition implementation via binned expectations is correct for binary outcomes | EV診断モジュール | Wrong decomposition would give misleading diagnostic values |
| A4 | Surface key in RacePredictor is "turf" or "dirt" (matching domain types) | フィルター適用 | Wrong surface key would use wrong threshold |

**Validation needed:** A1 should be verified by tracing the data flow in `_train_submodel()` -- when `use_ensemble=True`, the ensemble replaces hit_model at line 468, so by line 762 the df_oof should contain ensemble-predicted values. A4 is verified by checking Phase 14's drift_diagnostics.py which uses "turf"/"dirt" strings.

## Open Questions (RESOLVED)

1. **Positive-edge winner definition for threshold computation (Claude's Discretion)**
   - What we know: D-01 says "25th percentile of positive-edge OOF winners"
   - What's unclear: "positive-edge" could mean (a) win_selection_edge > 0, (b) ev_win_corrected > 1.0, or (c) both
   - Recommendation: Use `win_selection_edge > 0` as the edge criterion, since it's the primary filter in get_win_candidates(). Also require `kakuteijyuni == 1` for "winners"

2. **Surface-specific fallback threshold values (Claude's Discretion)**
   - What we know: D-03 requires surface-specific fallback when threshold computation fails (too few winners)
   - What's unclear: The exact fallback values
   - Recommendation: Use 0.8 for turf, 0.7 for dirt (below 1.0 but conservative). These can be refined by Phase 17 Optuna

3. **ECE binning strategy**
   - What we know: "uniform" bins are standard but "quantile" bins are better for skewed distributions
   - What's unclear: Which is more appropriate for EV predictions (which are right-skewed)
   - Recommendation: Use "quantile" (equal-frequency) bins with n_bins=10

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| numpy | All computation | Yes | 2.4.3 | -- |
| scipy | pearsonr, stats | Yes | 1.17.1 | -- |
| scikit-learn | brier_score_loss, calibration_curve | Yes | 1.8.0 | -- |
| Python 3.11 | Runtime | Yes | 3.11 (mise) | -- |
| pytest | Testing | Yes | (dev install) | -- |

**Missing dependencies with no fallback:**
- None -- all required tools are available

**Missing dependencies with fallback:**
- None

## Sources

### Primary (HIGH confidence)
- Full codebase inspection: src/models/robust_confidence_estimator.py (253 lines), src/backtest/race_predictor.py (lines 410-490), src/pipelines/training_pipeline.py (lines 450-830), src/models/drift_diagnostics.py (283 lines), src/backtest/engine.py (lines 590-830, 1080-1140), src/backtest/report.py (568 lines), src/domain/models.py (lines 229-255)
- Package verification: numpy 2.4.3, scipy 1.17.1, sklearn 1.8.0 -- all verified via python -c execution
- Phase 14 summaries: 14-01-SUMMARY.md, 14-02-SUMMARY.md -- verified pattern references

### Secondary (MEDIUM confidence)
- [Towards Data Science: ECE Visual Explanation](https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d/) -- ECE implementation pattern
- [scikit-learn calibration docs](https://scikit-learn.org/stable/modules/calibration.html) -- calibration_curve, brier_score_loss usage
- [Wikipedia: Brier Score](https://en.wikipedia.org/wiki/Brier_score) -- decomposition formulas

### Tertiary (LOW confidence)
- Optimal ECE bin count for EV prediction calibration -- 10 bins is standard but empirical validation needed
- Exact impact of ensemble residual recalibration on EV_lower distribution -- requires pipeline execution

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - zero new dependencies, all verified installed
- Architecture: HIGH - exact code locations identified for all changes, Phase 14 pattern well-established
- Pitfalls: HIGH - based on direct code analysis with line references
- Code examples: HIGH - ECE/Brier patterns verified against academic sources, reliability diagram pattern verified in existing codebase

**Research date:** 2026-05-06
**Valid until:** 2026-06-06 (stable codebase, no fast-moving dependencies)
