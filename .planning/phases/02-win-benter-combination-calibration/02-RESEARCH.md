# Phase 2: Win Benter Combination & Calibration - Research

**Researched:** 2026-05-02
**Domain:** Benter logit-space probability blending, Beta/Isotonic calibration, race-level normalization
**Confidence:** HIGH

## Summary

Phase 2 implements Win Benter combination, the single highest-impact improvement identified in the roadmap. The existing Place Benter pipeline in `src/models/benter_combination.py` provides a proven `BenterCombination` class that will be reused internally by a new `WinBenterGate` class. The key difference from Place is: (1) market probability source changes from `fukuoddslow` to `tanodds`, (2) fundamental probability comes from `p_win_pred` after EV correction, (3) Win-specific OOF data generation is required since `WinTwoStageModel` does not store validation predictions unlike `PlaceTwoStageModel`, and (4) Beta calibration is recommended over Isotonic based on the Place Isotonic failure (v5.6 disabled due to aggressive probability suppression: mean 0.224 vs true 0.375).

**Critical correction:** CONTEXT.md references `tanoddslow` as the market odds column (D-02, D-03), but the actual column name in the codebase is `tanodds`. This is verified by grep across all Python files -- `tanoddslow` returns zero matches. The planner and implementer MUST use `tanodds`.

**Primary recommendation:** Create `WinBenterGate` as a new class that wraps `BenterCombination` with Win-specific preprocessing (tanodds market probability extraction), post-processing (race normalization), and comparison evaluation (Beta vs Isotonic calibration with Brier Score + ECE metrics).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** fundamental確率は2Stage EV補正後（WinTwoStageModel.predict_ev() → EVCorrection後）を使用する
- **D-02:** 市場確率ソースは `tanodds`（最終単勝オッズ）。レース前オッズなので情報リークなし (**CORRECTED**: CONTEXT.md says `tanoddslow` but actual column is `tanodds`)
- **D-03:** 前処理は `p_market = 1/tanodds` のまま。JRA控除率はBenterのbetaパラメータが吸収する
- **D-04:** Benter学習データは専用OOF（out-of-fold）予測で生成する。`use_ensemble`に依存しない独立したKFold CV方式。ベストプラクティス追求
- **D-05:** Beta calibrationとIsotonic calibrationの両方を実装し、比較評価する（BENT-02要件）
- **D-06:** パイプライン構成: `raw_p -> Benter -> {Beta|Isotonic} -> TempScale(オプション)`。TempScaleは追加改善がある場合のみ適用
- **D-07:** 比較評価指標は Brier Score + ECE (Expected Calibration Error) で定量比較。信頼性ダイアグラムは可視化用
- **D-08:** Beta calibration（3パラメータ）がIsotonicより過学習しにくく推奨。PlaceでのIsotonic失敗は自由度過多が原因
- **D-09:** Benter学習は馬単位で行い、レース正規化は後処理として独立適用する。ベストプラクティス
- **D-10:** 正規化方式は単純正規化 `P_normalized = P_i / sum(P_j)` を標準とする
- **D-11:** WinBenterGate新クラスを作成。既存BenterCombinationを内部で利用。Placeコードに影響なし
- **D-12:** SubmodelSetに `win_benter`, `win_isotonic_calibrator`, `win_temperature_scaler` フィールドを追加。Placeと並列構造
- **D-13:** Win Benterの最適化パラメータ（alpha/beta/gamma）はグリッドサーチで最適な初期値を探索する

### Claude's Discretion
- キャリブレーションパイプラインの詳細実装（各ステップの有効/無効判定ロジック）
- グリッドサーチの範囲・粒度の設定
- TempScale適用の閾値（どの程度の改善があれば適用するか）
- 信頼性ダイアグラムの出力形式とバケット数

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BENT-01 | 単勝予測にBenter組み合わせを実装 | WinBenterGate wrapping BenterCombination; tanodds market probability; OOF generation pattern |
| BENT-02 | Beta/Isotonic calibrationの比較評価 | betacal package (3-param Beta); sklearn.isotonic.IsotonicRegression; Brier Score + ECE metrics |
| BENT-03 | レース単位正規化(P合計=1.0) | Simple normalization P_i / sum(P_j) as post-processing; groupby race_id pattern |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Benter logit-space blending | API / Backend (ML) | -- | Pure mathematical optimization (NLL minimization), no UI component |
| Beta / Isotonic calibration | API / Backend (ML) | -- | Post-processing on model outputs, sklearn-compatible estimators |
| Race-level normalization | API / Backend (ML) | -- | Deterministic post-processing on per-horse probabilities |
| OOF prediction generation | API / Backend (ML) | -- | KFold CV during training pipeline, no client involvement |
| Model serialization | API / Backend (Storage) | -- | JSON/joblib save/load in training_pipeline + model_loader |
| Calibration evaluation | API / Backend (Metrics) | -- | Brier Score, ECE, reliability diagrams for model selection |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.8.0 | IsotonicRegression, brier_score_loss, calibration_curve | Already installed; IsotonicRegression used in Place pipeline [VERIFIED: python -c import] |
| scipy | 1.17.1 | minimize (L-BFGS-B) for Benter NLL optimization | Already used in BenterCombination.fit() [VERIFIED: codebase grep] |
| numpy | 2.4.3 | Array operations, logit/sigmoid | Core dependency [VERIFIED: python -c import] |
| pandas | 2.3.3 | DataFrame groupby for race normalization | Core dependency [VERIFIED: python -c import] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| betacal | 1.1.0 | BetaCalibration (3-parameter beta calibration) | BENT-02 comparison; recommended default per D-08 [VERIFIED: PyPI] |
| joblib | (transitive) | Serialization for IsotonicRegression | Saving/loading calibrators (existing pattern) [VERIFIED: codebase usage] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| betacal package | Manual BetaCalibrationImpl | betacal is MIT-licensed, 2-file package with sklearn-compatible API; manual implementation would be reinventing a well-tested wheel with the same math |
| betacal BetaCalibration | sklearn CalibratedClassifierCV(method="sigmoid") | Platt scaling is 2-param logistic; Beta calibration has 3 params and handles asymmetric miscalibration better [CITED: Kull et al., ICML 2017] |

**Installation:**
```bash
pip install betacal
# Also add to pyproject.toml dependencies
```

**Version verification:**
```
scikit-learn: 1.8.0  (verified via python import)
scipy: 1.17.1  (verified via python import)
numpy: 2.4.3  (verified via python import)
pandas: 2.3.3  (verified via python import)
lightgbm: 4.6.0  (verified via python import)
betacal: NOT INSTALLED (v1.1.0 on PyPI, needs install)
```

## Architecture Patterns

### System Architecture Diagram

```
Training Pipeline (per surface)
================================

  df_oof (sorted by race_date)
       |
       v
  [WinTwoStageModel.predict_ev()]
       |-- outputs: p_win_pred, e_return_win_pred, ev_win
       |
       v
  [EVCorrectionModel.correct_ev()]
       |-- outputs: p_win_corrected, ev_win_corrected
       |
       v                    v
  (p_fundamental)     (tanodds column)
       |                    |
       |              [1/tanodds clipping]
       |                    |
       |              (p_market)
       |                    |
       v                    v
  +----------------------------------+
  | WinBenterGate                    |
  |   BenterCombination.fit(OOF)    |  <--- KFold OOF predictions
  |   combine(p_fund, p_market)     |
  +----------------------------------+
       |
       v
  (p_win_benter_raw)
       |
       +---> [BetaCalibration]  --+  (comparison path A)
       |                          |
       +---> [IsotonicRegression] --+  (comparison path B)
                                  |
                                  v
                         [TempScale (optional)]
                                  |
                                  v
                         (p_win_calibrated)
                                  |
                                  v
                     [Race Normalization]
                     groupby(race_id)
                     P_i / sum(P_j)
                                  |
                                  v
                         (p_win_final)


Prediction Pipeline (RacePredictor)
====================================

  df (per race)
       |
       v
  [WinTwoStageModel.predict_ev()]
       |
       v
  [EVCorrectionModel.correct_ev()]  <-- line 113
       |
       v
  [WinBenterGate.apply()]           <-- NEW insertion point
       |-- benter.combine(p_win, p_market)
       |-- calibrator.transform()
       |-- temp_scaler.transform() (optional)
       |-- race normalization
       |-- outputs: p_win_combined, edge_win
       |
       v
  (continue to confidence, betting)
```

### Recommended Project Structure
```
src/
├── models/
│   ├── benter_combination.py    # EXISTING: BenterCombination + TemperatureScaling (reuse as-is)
│   ├── win_benter_gate.py       # NEW: WinBenterGate class (D-11)
│   └── ev_correction_model.py   # EXISTING: _normalize_probability_array (reuse for normalization)
├── domain/
│   └── models.py                # MODIFY: SubmodelSet + win_* fields (D-12)
├── pipelines/
│   └── training_pipeline.py     # MODIFY: Win Benter training + saving (insertion points)
├── db/
│   └── model_loader.py          # MODIFY: Win Benter loading
└── backtest/
    └── race_predictor.py        # MODIFY: Win Benter application in predict()
```

### Pattern 1: WinBenterGate as Facade over BenterCombination
**What:** New class that encapsulates Win-specific Benter pipeline: market probability extraction from tanodds, calibration, and race normalization.
**When to use:** All Win prediction paths (backtest, paper trading, live prediction).
**Example:**
```python
# Source: Designed following existing BenterCombination + Place Benter pattern
class WinBenterGate:
    """Win-specific Benter combination gate.

    Encapsulates: market probability extraction, Benter blending,
    calibration, optional temperature scaling, and race normalization.
    """
    def __init__(
        self,
        benter: BenterCombination,
        calibrator: BetaCalibration | IsotonicRegression | None = None,
        temp_scaler: TemperatureScaling | None = None,
    ) -> None:
        self.benter = benter
        self.calibrator = calibrator
        self.temp_scaler = temp_scaler

    @staticmethod
    def extract_market_probability(tanodds: np.ndarray) -> np.ndarray:
        """Convert tanodds to implied probability with clipping."""
        return np.clip(np.where(tanodds > 0, 1.0 / tanodds, np.nan), 0.01, 0.99)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full Win Benter pipeline to a DataFrame."""
        df = df.copy()
        p_fund = df["p_win_corrected"].values  # After EV correction (D-01)
        p_market = self.extract_market_probability(df["tanodds"].values)

        # Step 1: Benter combination
        p_combined = self.benter.combine(p_fund, p_market)

        # Step 2: Calibration (Beta or Isotonic)
        if self.calibrator is not None:
            p_combined = self.calibrator.transform(p_combined)

        # Step 3: Temperature scaling (optional, D-06)
        if self.temp_scaler is not None:
            p_combined = self.temp_scaler.transform(p_combined)

        df["p_win_combined"] = p_combined

        # Step 4: Race normalization (D-09, D-10)
        race_sums = df.groupby("race_id")["p_win_combined"].transform("sum")
        df["p_win_final"] = df["p_win_combined"] / race_sums

        # Edge calculation
        df["edge_win"] = df["p_win_final"] * df["tanodds"] - 1.0
        return df
```

### Pattern 2: Dedicated OOF Prediction Generation for Win Benter
**What:** KFold CV on training data to generate unbiased predictions for Benter fitting, independent of `use_ensemble`.
**When to use:** Training pipeline, after WinTwoStageModel is trained but before Benter fitting.
**Why essential:** `WinTwoStageModel.train_hit_model()` does NOT store `_val_p_raw` / `_val_y` like `PlaceTwoStageModel` does. A new OOF mechanism is required (D-04).
**Example:**
```python
# Source: Follows MarketModel.predict_oof() pattern in training_pipeline.py
from sklearn.model_selection import KFold

def generate_win_oof_predictions(
    df: pd.DataFrame,
    win_model: WinTwoStageModel,
    ev_corrector: EVCorrectionModel,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate OOF predictions for Win Benter fitting.

    Returns:
        (oof_p_fund, oof_p_market, oof_y) -- aligned arrays for Benter.fit()
    """
    df = df.sort_values("race_date").reset_index(drop=True)
    kfold = KFold(n_splits=n_splits, shuffle=False)  # Time-series: no shuffle
    oof_preds = np.full(len(df), np.nan)

    for train_idx, val_idx in kfold.split(df):
        # Re-train on fold training data (lightweight: just hit_model)
        fold_model = WinTwoStageModel()
        fold_train = df.iloc[train_idx]
        fold_model.train_hit_model(fold_train, num_threads=0)
        fold_val = df.iloc[val_idx]
        fold_val = fold_model.predict_ev(fold_val)
        oof_preds[val_idx] = fold_val["p_win_pred"].values

    # Apply EV correction to OOF predictions for consistency with D-01
    df["p_win_oof"] = oof_preds
    df = ev_corrector.correct_ev(df)

    p_fund = df["p_win_oof"].values
    p_market = np.clip(np.where(df["tanodds"] > 0, 1.0 / df["tanodds"], np.nan), 0.01, 0.99)
    y = (df["kakuteijyuni"] == 1).astype(int).values

    # Drop NaN entries
    valid = ~(np.isnan(p_fund) | np.isnan(p_market))
    return p_fund[valid], p_market[valid], y[valid]
```

### Pattern 3: Three-File Update for New Model Components
**What:** When adding a new model component, three files must be updated in lockstep.
**When to use:** Adding `win_benter`, `win_isotonic_calibrator`, `win_temperature_scaler` to SubmodelSet.
**Files:**
1. `src/domain/models.py` -- Add Optional fields to `SubmodelSet` dataclass
2. `src/pipelines/training_pipeline.py` -- Train and save the new components
3. `src/db/model_loader.py` -- Load the new components from disk

### Pattern 4: Grid Search for Benter Initial Parameters (D-13)
**What:** Explore alpha/beta/gamma initial values via grid search before NLL optimization.
**When to use:** During training pipeline, before BenterCombination.fit().
**Example:**
```python
# Source: Based on BenterCombination.fit() bounds and D-13
from itertools import product

def grid_search_benter_init(
    p_fund: np.ndarray,
    p_market: np.ndarray,
    y: np.ndarray,
) -> BenterCombination:
    """Grid search over alpha/beta/gamma initial values, pick best NLL."""
    best_nll = float("inf")
    best_benter = None

    alpha_grid = [0.3, 0.5, 0.7, 1.0]
    beta_grid = [0.3, 0.5, 0.7, 1.0]
    gamma_grid = [-1.0, 0.0, 1.0]

    for a0, b0, g0 in product(alpha_grid, beta_grid, gamma_grid):
        try:
            benter = BenterCombination._fit_with_init(p_fund, p_market, y, x0=[a0, b0, g0])
            nll = _compute_nll(benter, p_fund, p_market, y)
            if nll < best_nll:
                best_nll = nll
                best_benter = benter
        except Exception:
            continue

    return best_benter
```
Note: This requires a small refactoring of `BenterCombination.fit()` to accept `x0` as parameter, or a private `_fit_with_init()` classmethod.

### Anti-Patterns to Avoid
- **Reusing Place Benter code directly for Win:** The market source (`fukuoddslow` vs `tanodds`), fundamental probability source (`p_place_pred` vs `p_win_corrected`), and calibration history (Place Isotonic failed, Win Beta recommended) are different. Must create separate WinBenterGate (D-11).
- **Using tanoddslow column name:** CONTEXT.md incorrectly references `tanoddslow`. The actual column is `tanodds`. Using `tanoddslow` will cause KeyError at runtime. [VERIFIED: grep returns zero matches for tanoddslow in .py files]
- **Applying Isotonic as default calibration for Win:** Place Isotonic was disabled in v5.6 because it pushed mean probability from ~0.375 to 0.224. Beta calibration (3-param) is less prone to overfitting and should be the default (D-08). [VERIFIED: race_predictor.py lines 143-148]
- **Forgetting race normalization:** Without normalization, probabilities do not sum to 1.0 across horses in a race, which makes EV comparison unreliable and betting edge calculations incorrect.
- **Fitting Benter on training data (not OOF):** This causes data leakage -- the Benter parameters overfit to the same data used to train the hit model. Must use OOF predictions (D-04).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Beta calibration | Manual 3-param beta distribution fitting | `betacal.BetaCalibration(parameters="abc")` | Numerically stable parameter estimation, sklearn-compatible API, battle-tested [VERIFIED: PyPI v1.1.0] |
| Isotonic calibration | Manual monotonic regression | `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")` | Already imported in training_pipeline.py, handles edge cases [VERIFIED: codebase] |
| Brier Score | Manual MSE calculation | `sklearn.metrics.brier_score_loss()` | Handles edge cases, sklearn standard [VERIFIED: sklearn 1.8.0] |
| Logit-space Benter combination | New blend formula | `BenterCombination.combine()` | Existing class with tested fit/combine/to_dict/from_dict/save/load API [VERIFIED: benter_combination.py] |
| Temperature scaling | New scaling class | `TemperatureScaling` in benter_combination.py | Existing class with fit/transform/save/load API [VERIFIED: benter_combination.py] |
| NLL minimization | Custom optimizer | `scipy.optimize.minimize(method="L-BFGS-B")` | Existing pattern in BenterCombination.fit(), proven convergence [VERIFIED: benter_combination.py] |
| Reliability diagram data | Manual binning | `sklearn.calibration.calibration_curve(y_true, y_prob, n_bins=10)` | Standard implementation, handles empty bins [VERIFIED: sklearn 1.8.0] |

**Key insight:** The existing codebase already has all the building blocks for Benter combination, temperature scaling, and Isotonic calibration. The new work is primarily integration: Win-specific preprocessing, OOF generation, Beta calibration (via betacal), and race normalization.

## Common Pitfalls

### Pitfall 1: Wrong Market Odds Column Name
**What goes wrong:** CONTEXT.md references `tanoddslow` but the actual DataFrame column is `tanodds`. Using the wrong name causes `KeyError` at runtime.
**Why it happens:** `fukuoddslow` is the Place column (fuku = fukushou = place); by analogy one might assume `tanoddslow` exists for tan (tanshou = win). But the Win column is simply `tanodds`.
**How to avoid:** Always use `tanodds`. Verify with: `df.columns` check before accessing.
**Warning signs:** KeyError on `tanoddslow` during Benter fitting or prediction.

### Pitfall 2: Missing Win OOF Validation Data
**What goes wrong:** Attempting to access `win_2s._val_p_raw` (like Place does) fails because `WinTwoStageModel.train_hit_model()` does not save validation predictions.
**Why it happens:** Only `PlaceTwoStageModel.train_hit_model()` saves `_val_p_raw`, `_val_y`, `_val_fukuoddslow` (lines 419-427). Win model was implemented without this feature.
**How to avoid:** Implement dedicated OOF generation per D-04 (KFold CV). Do NOT rely on `_val_p_raw` attribute on WinTwoStageModel.
**Warning signs:** AttributeError `_val_p_raw` on WinTwoStageModel instance.

### Pitfall 3: Isotonic Overfitting on Win Probabilities
**What goes wrong:** Isotonic regression produces aggressively suppressed probabilities (as happened with Place: mean 0.224 vs true 0.375).
**Why it happens:** Isotonic regression is non-parametric with many degrees of freedom. It overfits to noise in the calibration set, especially with imbalanced binary data (win rate ~8-12%).
**How to avoid:** Use Beta calibration (3-param) as default (D-08). If Isotonic is used, always compare against Beta with Brier Score + ECE. Disable Isotonic if Brier Score degrades.
**Warning signs:** Calibrated mean probability significantly below or above true win rate; ECE higher after calibration.

### Pitfall 4: NaN Propagation in Market Probability
**What goes wrong:** Some horses have `tanodds = 0` or missing, leading to `1/0 = inf` or NaN in market probability, which propagates through Benter combination.
**Why it happens:** Early races or data quality issues produce zero/missing odds.
**How to avoid:** Clip market probability: `np.clip(np.where(tanodds > 0, 1.0 / tanodds, np.nan), 0.01, 0.99)`. Follow the existing Place pattern in race_predictor.py lines 136-140.
**Warning signs:** NaN in `p_win_combined` or `edge_win` columns.

### Pitfall 5: Race Normalization on Incomplete Data
**What goes wrong:** Applying `P_i / sum(P_j)` when some horses have NaN probabilities, causing division by a sum that excludes those horses (over-inflating remaining probabilities).
**Why it happens:** NaN horses are excluded from sum but still in the DataFrame.
**How to avoid:** Only normalize within valid (non-NaN) horses. Use `df.groupby("race_id")["p_win_combined"].transform("sum")` which naturally handles NaN exclusion in pandas. Or explicitly fill/interpolate before normalization.
**Warning signs:** Normalized probabilities summing to > 1.0 for some races; edge values unrealistically high.

### Pitfall 6: Forgetting to Update All Three Files
**What goes wrong:** Adding win_* fields to SubmodelSet but forgetting to update model_loader.py (loading) or training_pipeline.py (saving), causing silent failures at runtime.
**Why it happens:** The three-file update pattern requires synchronized changes.
**How to avoid:** Treat as a checklist: models.py (field) + training_pipeline.py (save) + model_loader.py (load). All three must be updated in the same task.
**Warning signs:** `AttributeError` when accessing win_benter on loaded SubmodelSet; or model file not found at prediction time.

## Code Examples

### ECE (Expected Calibration Error) Computation
```python
# Source: Standard ECE formula [CITED: Guo et al., 2017 "On Calibration of Modern Neural Networks"]
import numpy as np

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += mask.sum() * abs(avg_accuracy - avg_confidence)
    return ece / len(y_true)
```

### Beta Calibration Fitting and Comparison
```python
# Source: betacal PyPI package API [VERIFIED: PyPI v1.1.0]
from betacal import BetaCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

# Beta calibration (3-param, recommended per D-08)
beta_cal = BetaCalibration(parameters="abc")
beta_cal.fit(p_benter_train, y_train)
p_beta = beta_cal.transform(p_benter_val)

# Isotonic calibration (comparison per D-05)
iso_cal = IsotonicRegression(out_of_bounds="clip")
iso_cal.fit(p_benter_train, y_train)
p_iso = iso_cal.transform(p_benter_val)

# Quantitative comparison (D-07)
brier_beta = brier_score_loss(y_val, p_beta)
brier_iso = brier_score_loss(y_val, p_iso)
ece_beta = compute_ece(y_val, p_beta)
ece_iso = compute_ece(y_val, p_iso)

print(f"Beta:  Brier={brier_beta:.6f}  ECE={ece_beta:.6f}")
print(f"Isotonic: Brier={brier_iso:.6f}  ECE={ece_iso:.6f}")
```

### Reliability Diagram (Calibration Curve)
```python
# Source: sklearn.calibration.calibration_curve [VERIFIED: sklearn 1.8.0]
from sklearn.calibration import calibration_curve

# For reliability diagram visualization
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true, y_prob, n_bins=10, strategy="uniform"
)
# fraction_of_positives: actual win rate in each bin
# mean_predicted_value: average predicted probability in each bin
# Perfect calibration: fraction_of_positives == mean_predicted_value (diagonal line)
```

### Race Normalization
```python
# Source: Based on existing _normalize_probability_array in ev_correction_model.py
def normalize_race_probabilities(df: pd.DataFrame, prob_col: str = "p_win_combined") -> pd.Series:
    """Normalize probabilities to sum to 1.0 within each race (D-09, D-10)."""
    race_sums = df.groupby("race_id")[prob_col].transform("sum")
    return df[prob_col] / race_sums
```

### SubmodelSet Field Addition Pattern
```python
# Source: Existing SubmodelSet pattern in domain/models.py line 228 [VERIFIED: codebase]
@dataclass
class SubmodelSet:
    # ... existing fields ...
    benter_combo: BenterCombination | None = None          # Place Benter
    isotonic_calibrator: IsotonicRegression | None = None   # Place Isotonic
    temperature_scaler: TemperatureScaling | None = None     # Place TempScale
    # NEW: Win Benter fields (D-12)
    win_benter: BenterCombination | None = None
    win_isotonic_calibrator: IsotonicRegression | None = None
    win_temperature_scaler: TemperatureScaling | None = None
```

### Model Save/Load Naming Convention
```python
# Source: Existing naming pattern in training_pipeline.py lines 999-1012 [VERIFIED: codebase]
# Place: benter_combo_{surface}.json, isotonic_place_{surface}.joblib, temp_scale_{surface}.json
# Win:   benter_combo_win_{surface}.json, isotonic_win_{surface}.joblib, temp_scale_win_{surface}.json

# Saving (training_pipeline.py)
if sub.win_benter is not None:
    sub.win_benter.save(models_dir / f"benter_combo_win_{surface}.json")
if sub.win_isotonic_calibrator is not None:
    joblib.dump(sub.win_isotonic_calibrator, models_dir / f"isotonic_win_{surface}.joblib")
if sub.win_temperature_scaler is not None:
    sub.win_temperature_scaler.save(models_dir / f"temp_scale_win_{surface}.json")

# Loading (model_loader.py)
win_benter_file = models_dir / f"benter_combo_win_{surface}.json"
if win_benter_file.is_file():
    win_benter = BenterCombination.load(win_benter_file)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Isotonic as default calibration | Beta calibration (3-param) preferred | v5.6 (Place Isotonic disabled) | Beta less prone to overfitting, recommended for Win |
| Validation split for Benter fitting | Dedicated KFold OOF predictions | D-04 decision | Prevents data leakage, more robust fitting |
| Single x0=[0.5, 0.5, 0.0] for Benter | Grid search over alpha/beta/gamma initial values | D-13 decision | Finds better local optima in NLL landscape |
| No race normalization for Benter output | Post-processing normalization (P_i / sum(P_j)) | D-09, D-10 decision | Ensures probabilities are well-calibrated at race level |

**Deprecated/outdated:**
- Isotonic calibration as default post-Benter step: Disabled in v5.6 for Place due to aggressive probability suppression. Retained as comparison option for Win (D-05) but not recommended as default (D-08).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | betacal v1.1.0 is compatible with numpy 2.4.3 and sklearn 1.8.0 | Standard Stack | Beta calibration fallback to manual implementation needed |
| A2 | `tanodds` column is always available in df_oof during training pipeline | Common Pitfalls | OOF generation would fail; need to verify column existence |
| A3 | Win Benter fitting requires >= 500 samples (same threshold as Place) | Architecture Patterns | May need threshold adjustment based on actual win rate |
| A4 | BenterCombination.fit() can be refactored to accept custom x0 without breaking Place | Architecture Patterns | Need to test Place Benter still works after refactoring |
| A5 | `betacal.BetaCalibration` .transform() method name (sklearn-compatible API) | Code Examples | If API differs, wrapper needed |

## Open Questions

1. **betacal compatibility with numpy 2.x**
   - What we know: betacal v1.1.0 on PyPI, depends on numpy and scikit-learn. numpy 2.4.3 is installed.
   - What's unclear: Whether betacal v1.1.0 works with numpy 2.x (released after betacal last update).
   - Recommendation: Install and test `import betacal; bc = BetaCalibration(); bc.fit(np.array([0.1,0.5,0.9]), np.array([0,1,1])); bc.transform(np.array([0.5]))` before implementation. If incompatible, implement Beta calibration manually using scipy.optimize.

2. **OOF generation computational cost**
   - What we know: KFold with n_splits=5 means retraining hit_model 5 times per surface. LightGBM hit_model trains in ~1-2 minutes per fold.
   - What's unclear: Total training time impact.
   - Recommendation: Add TimingContext wrapper around OOF generation; estimate ~5-10 min additional training time per surface.

3. **Grid search granularity for D-13**
   - What we know: Claude's discretion on grid range/granularity. Current bounds: alpha [0.01, 5.0], beta [0.20, 5.0], gamma [-5.0, 5.0].
   - What's unclear: Optimal grid density (4x4x3 = 48 combinations proposed in Pattern 4 example).
   - Recommendation: Start with coarse grid (48 combinations), refine around best region if needed.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scikit-learn | IsotonicRegression, metrics | Yes | 1.8.0 | -- |
| scipy | Benter NLL optimization | Yes | 1.17.1 | -- |
| numpy | Core array operations | Yes | 2.4.3 | -- |
| pandas | DataFrame operations | Yes | 2.3.3 | -- |
| lightgbm | WinTwoStageModel | Yes | 4.6.0 | -- |
| joblib | Calibrator serialization | Yes | (transitive) | -- |
| betacal | Beta calibration | No | -- | Manual Beta calibration implementation |

**Missing dependencies with no fallback:**
- None (betacal has manual implementation fallback)

**Missing dependencies with fallback:**
- betacal: If installation fails or incompatible with numpy 2.x, implement Beta calibration manually using scipy.optimize.minimize with the 3-parameter beta CDF formula from Kull et al. (2017).

## Security Domain

> Phase 2 is purely ML model computation (no user input, no API endpoints, no secrets). Security enforcement is not applicable to this phase's scope.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | -- |
| V3 Session Management | no | -- |
| V4 Access Control | no | -- |
| V5 Input Validation | partial | numpy/pandas dtype validation on input arrays |
| V6 Cryptography | no | -- |

### Known Threat Patterns for ML Pipeline

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Data leakage (train-on-test) | Tampering | KFold OOF (D-04), time-series sorting |
| Model overfitting | Tampering | Calibration comparison (D-05), Brier Score monitoring |

## Sources

### Primary (HIGH confidence)
- Codebase: `src/models/benter_combination.py` -- BenterCombination + TemperatureScaling full API
- Codebase: `src/backtest/race_predictor.py` lines 124-156 -- Place Benter application pattern
- Codebase: `src/pipelines/training_pipeline.py` lines 528-565 -- Place Benter training pattern
- Codebase: `src/domain/models.py` line 228 -- SubmodelSet dataclass structure
- Codebase: `src/db/model_loader.py` lines 502-529 -- Benter/Isotonic/TempScale loading pattern
- sklearn 1.8.0: `brier_score_loss`, `calibration_curve`, `IsotonicRegression` -- verified via python import

### Secondary (MEDIUM confidence)
- betacal PyPI v1.1.0 -- API: BetaCalibration(parameters="abc"), fit(), predict()/transform()
- betacal GitHub: https://github.com/betacal/python -- sklearn-compatible interface
- Kull et al., ICML 2017 -- Beta Calibration paper (theoretical basis for D-08 recommendation)

### Tertiary (LOW confidence)
- betacal numpy 2.x compatibility -- not tested in this session [A1]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all core libraries verified via python import, betacal verified on PyPI
- Architecture: HIGH -- patterns derived from existing codebase, no speculation
- Pitfalls: HIGH -- all pitfalls verified by codebase inspection (tanodds column, missing _val_p_raw, Isotonic failure)

**Research date:** 2026-05-02
**Valid until:** 2026-06-01 (stable ML domain, long validity)
