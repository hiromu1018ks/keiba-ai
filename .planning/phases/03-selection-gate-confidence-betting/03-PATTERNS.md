# Phase 3: Selection Gate, Confidence & Betting - Pattern Map

**Mapped:** 2026-05-02
**Files analyzed:** 10 (1 new, 9 modified)
**Analogs found:** 10 / 10

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/models/win_selection_gate.py` | model | CRUD (train/score) | `src/models/place_selection_gate.py` | exact |
| `src/models/robust_confidence_estimator.py` | model | transform | `src/models/robust_confidence_estimator.py` (self) | self-mod |
| `src/models/regime_detector.py` | model | config | `src/models/regime_detector.py` (self) | self-mod |
| `src/betting/meta_switcher.py` | service | config | `src/betting/meta_switcher.py` (self) | self-mod |
| `src/betting/gate_keeper.py` | middleware | filter | `src/betting/gate_keeper.py` (self) | self-mod |
| `src/betting/win_strategy.py` | service | request-response | `src/betting/win_strategy.py` (self) | self-mod |
| `src/betting/orchestrator.py` | controller | orchestration | `src/betting/orchestrator.py` (self) | self-mod |
| `src/backtest/race_predictor.py` | controller | pipeline | `src/backtest/race_predictor.py` (self) | self-mod |
| `src/pipelines/training_pipeline.py` | controller | pipeline | `src/pipelines/training_pipeline.py` (self) | self-mod |
| `src/db/model_loader.py` | service | file-I/O | `src/db/model_loader.py` (self) | self-mod |
| `src/domain/models.py` | model | config | `src/domain/models.py` (self) | self-mod |
| `tests/test_win_selection_gate.py` | test | test | `tests/test_place_selection_gate.py` | exact |

## Pattern Assignments

### `src/models/win_selection_gate.py` (model, train/score) -- NEW FILE

**Analog:** `src/models/place_selection_gate.py` (1044 lines)

This is a mechanical copy of PlaceSelectionGateModel with the following substitutions:
- Class name: `PlaceSelectionGateModel` -> `WinSelectionGateModel`
- Column names: see mapping table below
- Hit condition: `kakuteijyuni <= 3` -> `kakuteijyuni == 1`
- Odds source: `fukuoddslow` -> `tanoddslow`

**Column name mapping:**

| Place column | Win column |
|-------------|-----------|
| `fukuoddslow` | `tanoddslow` |
| `place_selection_prob` | `win_selection_prob` |
| `place_selection_edge` | `win_selection_edge` |
| `place_selection_ev` | `win_selection_ev` |
| `place_gate_score` | `win_gate_score` |
| `place_gate_pass` | `win_gate_pass` |
| `place_gate_rank` | `win_gate_rank` |
| `place_gate_score_gap` | `win_gate_score_gap` |
| `log_place_odds` | `log_win_odds` |
| `realized_place_roi` | `realized_win_roi` |
| `EV_lower_place` | `EV_lower_win_corrected` |
| `ev_place_corrected` | `ev_win_corrected` |
| `ev_place_direct` | `ev_win` |
| `p_place_corrected` | `p_win_final` |
| `p_place_combined` | `p_win_combined` |
| `p_place_pred` | `p_win_corrected` |
| `edge_place` | `edge_win` |
| `runner_up_place_selection_prob` | `runner_up_win_selection_prob` |
| `runner_up_place_selection_edge` | `runner_up_win_selection_edge` |
| `runner_up_fukuoddslow` | `runner_up_tanoddslow` |

**Class constants pattern** (analog lines 100-114):
```python
class WinSelectionGateModel:
    """OOF-learned gate and reranker for final win bet selection."""

    SCORE_COL = "win_gate_score"
    PASS_COL = "win_gate_pass"
    RANK_COL = "win_gate_rank"
    GAP_COL = "win_gate_score_gap"
    MARKET_CONDITION_COL = "market_condition_score"   # shared with place
    AGGRESSIVE_STRENGTH_COL = "aggressive_strength"    # shared with place
    AGGRESSIVE_TIER_COL = "aggressive_tier"            # shared with place
    RUNNER_UP_SCORE_COL = "runner_up_gate_score"       # shared with place
    RUNNER_UP_GAP_COL = "runner_up_gate_score_gap"     # shared with place
    RUNNER_UP_PROB_COL = "runner_up_win_selection_prob"
    RUNNER_UP_EDGE_COL = "runner_up_win_selection_edge"
    RUNNER_UP_ODDS_COL = "runner_up_tanoddslow"
```

**Helper functions pattern** (analog lines 13-54, with Win column names):
```python
def build_win_selection_ev(df: pd.DataFrame) -> pd.Series:
    lower_ev = _numeric_or_nan(df, "EV_lower_win_corrected")
    corrected_ev = _numeric_or_nan(df, "ev_win_corrected")
    direct_ev = _numeric_or_nan(df, "ev_win")

    if corrected_ev.notna().any():
        selection_ev = lower_ev.where(lower_ev.notna(), corrected_ev)
        safety_floor = corrected_ev * 0.85
        return pd.concat([selection_ev, safety_floor], axis=1).max(axis=1).astype(float)
    if lower_ev.notna().any():
        return lower_ev.astype(float)
    return direct_ev.astype(float)


def ensure_win_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "win_selection_ev" not in prepared.columns:
        if "EV_lower_win_corrected" in prepared.columns or "ev_win_corrected" in prepared.columns:
            prepared["win_selection_ev"] = build_win_selection_ev(prepared)
        elif "edge_win" in prepared.columns:
            prepared["win_selection_ev"] = _numeric_or_nan(prepared, "edge_win") + 1.0
        else:
            prepared["win_selection_ev"] = _numeric_or_nan(prepared, "ev_win")
    if "win_selection_edge" not in prepared.columns:
        prepared["win_selection_edge"] = _numeric_or_nan(prepared, "win_selection_ev") - 1.0
    if "win_selection_prob" not in prepared.columns:
        if "p_win_final" in prepared.columns:
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_final")
        elif "p_win_combined" in prepared.columns:
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_combined")
        else:
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_corrected")
    return prepared
```

**Critical difference in `_prepare_training_frame`** (analog lines 229-237):
```python
# Place version (analog):
prepared = prepared[prepared["fukuoddslow"] > 0].copy()
prepared["log_place_odds"] = np.log1p(prepared["fukuoddslow"])
prepared["realized_place_roi"] = np.where(
    prepared["kakuteijyuni"] <= 3,
    prepared["fukuoddslow"],
    0.0,
)

# Win version (NEW):
prepared = prepared[prepared["tanoddslow"] > 0].copy()
prepared["log_win_odds"] = np.log1p(prepared["tanoddslow"])
prepared["realized_win_roi"] = np.where(
    prepared["kakuteijyuni"] == 1,      # Win: only 1st place
    prepared["tanoddslow"],
    0.0,
)
```

**`_build_threshold_grid` odds column change** (analog lines 434-481):
```python
# Place version: fukuoddslow quantiles
# Win version: tanoddslow quantiles

odds_values = sorted(
    {
        4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0,
        *(float(df["tanoddslow"].quantile(q)) for q in [0.50, 0.60, 0.70, 0.80, 0.90]),
    }
)
```

**`score()` method** (analog lines 915-934, with Win columns):
```python
def score(self, df: pd.DataFrame) -> pd.DataFrame:
    prepared = ensure_win_selection_columns(df)
    if not self._trained:
        prepared[self.SCORE_COL] = np.nan
        prepared[self.PASS_COL] = False
        return prepared
    # ... (same structure as Place but using win_selection_*/tanoddslow columns)
    odds = _numeric_or_nan(prepared, "tanoddslow")
    prepared[self.SCORE_COL] = scores
    prepared[self.PASS_COL] = (
        (pd.to_numeric(prepared["win_selection_prob"], errors="coerce") >= self.min_prob)
        & (pd.to_numeric(prepared["win_selection_edge"], errors="coerce") >= self.min_edge)
        & (odds > 0)
        & (odds <= self.max_odds)
    )
    return prepared
```

**save/load pattern** (analog lines 979-1044):
```python
def save(self, path: Path) -> None:
    state = {
        "n_bins": self.n_bins,
        # ... same keys as Place ...
        "_trained": self._trained,
    }
    joblib.dump(state, path)

@classmethod
def load(cls, path: Path) -> WinSelectionGateModel:
    state = joblib.load(path)
    model = cls(
        n_bins=int(state["n_bins"]),
        prior_weight=float(state["prior_weight"]),
        min_train_races=int(state["min_train_races"]),
        min_fold_races=int(state["min_fold_races"]),
        max_folds=int(state["max_folds"]),
    )
    # ... same attribute assignment pattern ...
    model._trained = bool(state["_trained"])
    return model
```

---

### `src/domain/models.py` (model, config) -- MODIFY

**Analog:** `src/domain/models.py` line 228-253 (SubmodelSet dataclass)

**Current SubmodelSet** (lines 228-253):
```python
@dataclass
class SubmodelSet:
    """サブモデル（芝/ダート）のセット"""

    market: MarketModel
    stage1: AbilityModel
    place_ability: PlaceAbilityModel
    win: WinTwoStageModel
    ev_corrector: EVCorrectionModel
    place: PlaceTwoStageModel
    place_ev_corrector: PlaceEVCorrectionModel
    wide: WideTwoStageModel
    confidence: RobustConfidenceEstimator
    place_selection_gate: PlaceSelectionGateModel | None = None
    use_ensemble: bool = False
    benter_combo: BenterCombination | None = None
    isotonic_calibrator: IsotonicRegression | None = None
    temperature_scaler: TemperatureScaling | None = None
    win_benter: BenterCombination | None = None
    win_isotonic_calibrator: IsotonicRegression | None = None
    win_temperature_scaler: TemperatureScaling | None = None
```

**Addition pattern** -- add after `win_temperature_scaler`:
```python
    # Win Selection Gate (Phase 3, SELC-01)
    win_selection_gate: WinSelectionGateModel | None = None
```

**TYPE_CHECKING import addition** (line 12-23):
```python
if TYPE_CHECKING:
    # ... existing imports ...
    from models.win_selection_gate import WinSelectionGateModel   # ADD THIS
```

---

### `src/pipelines/training_pipeline.py` (controller, pipeline) -- MODIFY

**Analog:** existing PlaceSelectionGate training block (lines 760-767) and SubmodelSet construction (lines 769-787)

**Import addition** (line 40):
```python
from models.place_selection_gate import PlaceSelectionGateModel, ensure_place_selection_columns
# ADD:
from models.win_selection_gate import WinSelectionGateModel, ensure_win_selection_columns
```

**Training block pattern** (insert after line 767):
```python
# --- PlaceSelectionGate training (existing, lines 760-767) ---
with TimingContext(f"{surface}/place_selection_gate"):
    gate_train_df = df_oof.copy()
    _, gate_place_df = conf.predict_lower_bound(df_oof.copy(), df_oof.copy())
    if "EV_lower_place" in gate_place_df.columns:
        gate_train_df["EV_lower_place"] = gate_place_df["EV_lower_place"].values
    gate_train_df = ensure_place_selection_columns(gate_train_df)
    place_selection_gate = PlaceSelectionGateModel()
    place_selection_gate.train(gate_train_df)

# --- [NEW] WinSelectionGate training ---
with TimingContext(f"{surface}/win_selection_gate"):
    wsg_train_df = df_oof.copy()
    wsg_win_df, _ = conf.predict_lower_bound(df_oof.copy(), df_oof.copy())
    if "EV_lower_win_corrected" in wsg_win_df.columns:
        wsg_train_df["EV_lower_win_corrected"] = wsg_win_df["EV_lower_win_corrected"].values
    wsg_train_df = ensure_win_selection_columns(wsg_train_df)
    win_selection_gate = WinSelectionGateModel()
    win_selection_gate.train(wsg_train_df)
```

**SubmodelSet construction** (line 769-787, add field):
```python
return SubmodelSet(
    ...,
    place_selection_gate=place_selection_gate,   # existing
    win_selection_gate=win_selection_gate,        # ADD
    ...
)
```

**MLflow logging pattern** (insert after line 1050, following place_selection_gate MLflow block):
```python
if (
    sub.win_selection_gate is not None
    and sub.win_selection_gate.is_trained
):
    wsg_tmp: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as wsg_file:
            wsg_tmp = wsg_file.name
        sub.win_selection_gate.save(Path(wsg_tmp))
        mlflow.log_artifact(wsg_tmp, f"win_selection_gate_{surface}")
    finally:
        if wsg_tmp and os.path.exists(wsg_tmp):
            os.unlink(wsg_tmp)
```

**Local save pattern** (insert after line 1163):
```python
if sub.win_selection_gate is not None and sub.win_selection_gate.is_trained:
    sub.win_selection_gate.save(
        models_dir / f"win_selection_gate_{surface}.joblib"
    )
```

---

### `src/db/model_loader.py` (service, file-I/O) -- MODIFY

**Analog:** PlaceSelectionGate loading in both MLflow and local paths

**Import addition** (line 79):
```python
from models.place_selection_gate import PlaceSelectionGateModel
# ADD:
from models.win_selection_gate import WinSelectionGateModel
```

**MLflow loading pattern** (insert after place_selection_gate loading block, around line 142):
```python
# --- [NEW] WinSelectionGate (MLflow) ---
win_selection_gate = None
try:
    wsg_dir = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/win_selection_gate_{surface}"
    )
except Exception:
    try:
        wsg_dir = self._find_artifact_dir(run_id, f"win_selection_gate_{surface}")
    except Exception:
        wsg_dir = None
if wsg_dir is not None:
    wsg_files = list(Path(wsg_dir).glob("*.joblib"))
    if wsg_files:
        win_selection_gate = WinSelectionGateModel.load(wsg_files[0])
```

**MLflow SubmodelSet construction** (line 268-285, add field):
```python
submodels[surface] = SubmodelSet(
    ...,
    place_selection_gate=place_selection_gate,
    win_selection_gate=win_selection_gate,     # ADD
    ...
)
```

**Local loading pattern** (insert after line 558):
```python
# --- [NEW] WinSelectionGate (local) ---
win_selection_gate = None
wsg_file = models_dir / f"win_selection_gate_{surface}.joblib"
if wsg_file.is_file():
    try:
        win_selection_gate = WinSelectionGateModel.load(wsg_file)
    except Exception:
        logger.warning("Failed to load %s, skipping", wsg_file)
```

**Local SubmodelSet construction** (around line 646-660, add field):
```python
submodels[surface] = SubmodelSet(
    ...,
    place_selection_gate=place_selection_gate,
    win_selection_gate=win_selection_gate,     # ADD
    ...
)
```

---

### `src/backtest/race_predictor.py` (controller, pipeline) -- MODIFY

**Analog:** existing PlaceSelectionGate integration in predict() (lines 192-202)

**Import addition** (line 17):
```python
from models.place_selection_gate import build_place_selection_ev, ensure_place_selection_columns
# ADD:
from models.win_selection_gate import build_win_selection_ev, ensure_win_selection_columns
```

**Insertion point** (after line 124, after Win Benter apply, before `df = submodel.place.predict_ev(df)`):
```python
        # --- Win Benter Combination (existing, lines 116-124) ---
        if getattr(submodel, "win_benter", None) is not None:
            from models.win_benter_gate import WinBenterGate
            win_gate = WinBenterGate(...)
            df = win_gate.apply(df)

        # --- [NEW] WinSelectionGate (D-14: after Benter, before Place) ---
        df_winsel = ensure_win_selection_columns(df)
        if "win_selection_ev" not in df.columns:
            df = df_winsel
        win_gate_model = getattr(submodel, "win_selection_gate", None)
        win_gate_enabled = bool(
            win_gate_model is not None and getattr(win_gate_model, "is_trained", False) is True
        )
        if win_gate_enabled:
            assert win_gate_model is not None
            df = win_gate_model.score(df)
            win_annotate = getattr(win_gate_model, "annotate_race_context", None)
            if callable(win_annotate):
                df = win_annotate(df)

        # --- Place inference (existing, line 126-) ---
        df = submodel.place.predict_ev(df)
```

---

### `src/models/robust_confidence_estimator.py` (model, transform) -- MODIFY

**Analog:** self (existing code)

**Current calibrate() Win CP pattern** (lines 53-66):
```python
win_pred = pd.to_numeric(win_df["ev_win_corrected"], errors="coerce")
win_actual = pd.to_numeric(win_df["actual_ev_win"], errors="coerce")
win_mask = win_pred.notna() & win_actual.notna()
win_residuals = (win_actual[win_mask] - win_pred[win_mask]).abs()
if win_residuals.empty:
    self._win_cp_quantile = 0.0
else:
    self._win_cp_quantile = float(np.quantile(win_residuals.values, 1 - self.alpha))
```

**Extension: race-condition-dependent calibration** (add to calibrate method, after global quantile):
```python
# GroupBy (surface, distance_bin) for conditional CP quantile
self._win_cp_quantile_by_condition: dict[str, float] = {}
if "surface" in win_df.columns and "distance_bin" in win_df.columns:
    for (surf, dist), group in win_df[win_mask].groupby(["surface", "distance_bin"]):
        if len(group) >= 30:  # min sample threshold
            group_residuals = (win_actual.loc[group.index] - win_pred.loc[group.index]).abs()
            self._win_cp_quantile_by_condition[f"{surf}_{dist}"] = float(
                np.quantile(group_residuals.values, 1 - self.alpha)
            )
```

**predict_lower_bound() extension** (add conditional quantile lookup after line 109):
```python
# Race-condition-dependent CP quantile (fallback to global)
cp_quantile = self._win_cp_quantile
if hasattr(self, "_win_cp_quantile_by_condition"):
    cond_key = f"{win_df.get('surface', pd.Series('unknown')).iloc[0]}_{win_df.get('distance_bin', pd.Series('unknown')).iloc[0]}"
    cp_quantile = self._win_cp_quantile_by_condition.get(cond_key, self._win_cp_quantile)
cp_lower_win = win_ev - cp_quantile
```

---

### `src/models/regime_detector.py` (model, config) -- MODIFY

**Analog:** self (existing code, lines 178-232)

**Current edge_threshold values** (lines 183, 211, 224):
```python
# AGGRESSIVE:
"edge_threshold": 0.04,
# CONSERVATIVE:
"edge_threshold": 0.05,
# COLLAPSED:
"edge_threshold": 0.08,
```

**Updated values (BETT-01, JRA takeout +0.01):**
```python
# AGGRESSIVE:
"edge_threshold": 0.05,
# CONSERVATIVE:
"edge_threshold": 0.06,
# COLLAPSED:
"edge_threshold": 0.09,
```

---

### `src/betting/meta_switcher.py` (service, config) -- MODIFY

**Analog:** self (existing code, lines 42-67)

**Current values** (lines 47, 55, 63):
```python
# AGGRESSIVE: "edge_threshold": 0.04,
# CONSERVATIVE: "edge_threshold": 0.06,
# COLLAPSED: "edge_threshold": 0.09,
```

**Updated values (BETT-01, +0.01):**
```python
# AGGRESSIVE: "edge_threshold": 0.05,
# CONSERVATIVE: "edge_threshold": 0.07,
# COLLAPSED: "edge_threshold": 0.10,
```

---

### `src/betting/gate_keeper.py` (middleware, filter) -- MODIFY

**Analog:** self (existing code)

**Current default threshold** (line 28):
```python
return bet.edge >= 0.03
```

**Updated default** (line 30):
```python
def filter_bets(self, bets: list[Bet], edge_threshold: float = 0.04) -> list[Bet]:
```

No structural change. Only the default parameter value changes from 0.03 to 0.04.

---

### `src/betting/win_strategy.py` (service, request-response) -- MODIFY

**Analog:** self (existing code)

This file likely needs only minor adjustments if any. The existing Kelly calculation pattern (lines 70-77) is already correct per D-11:
```python
def _calc_stake(self, ev_lower: float, odds: float, bankroll: float) -> float:
    if bankroll <= 0 or odds <= 1.0:
        return 0.0
    edge = ev_lower - 1.0
    kelly = min(edge / (odds - 1.0), self.KELLY_FRACTION_CAP)
    raw = bankroll * kelly
    return float(max(0, int(math.floor(raw / self.MIN_STAKE)) * self.MIN_STAKE))
```

---

### `src/betting/orchestrator.py` (controller, orchestration) -- MODIFY

**Analog:** self (existing code)

No structural change expected. The Orchestrator already reads `edge_threshold` from MetaSwitcher (line 155) and passes it to GateKeeper (line 202). The threshold value propagation is automatic.

---

### `tests/test_win_selection_gate.py` (test, test) -- NEW FILE

**Analog:** `tests/test_place_selection_gate.py` (92+ lines)

**Test structure pattern** (analog lines 1-54):
```python
"""WinSelectionGateModel tests."""

from __future__ import annotations

import pandas as pd


def test_win_selection_gate_trains_and_scores() -> None:
    from models.win_selection_gate import WinSelectionGateModel

    rows: list[dict[str, object]] = []
    for race_idx in range(120):
        race_id = f"R{race_idx:04d}"
        race_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=race_idx)
        for umaban, prob, edge, odds, finish in [
            (1, 0.62, 0.24, 2.2, 1 if race_idx % 10 == 0 else 5),  # Win: only 1st
            (2, 0.28, 0.02, 4.5, 4),
            (3, 0.10, -0.15, 11.0, 8),
        ]:
            rows.append({
                "race_id": race_id,
                "race_date": race_date,
                "umaban": umaban,
                "kakuteijyuni": finish,
                "tanoddslow": odds,               # Win: tanoddslow
                "win_selection_prob": prob,        # Win: win_selection_*
                "win_selection_edge": edge,        # Win: win_selection_*
            })

    df = pd.DataFrame(rows)
    model = WinSelectionGateModel(min_train_races=40, min_fold_races=20, max_folds=3)
    model.train(df)

    assert model.is_trained is True

    scored = model.score(
        pd.DataFrame({
            "race_id": ["T1", "T1"],
            "race_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
            "umaban": [1, 2],
            "tanoddslow": [2.2, 11.0],
            "win_selection_prob": [0.60, 0.09],
            "win_selection_edge": [0.22, -0.18],
        })
    )

    assert "win_gate_score" in scored.columns       # Win column names
    assert "win_gate_pass" in scored.columns
    assert scored.loc[0, "win_gate_score"] > scored.loc[1, "win_gate_score"]
```

---

## Shared Patterns

### SubmodelSet Extension (3-point update: domain + pipeline + loader)

**Source:** `src/domain/models.py` + `src/pipelines/training_pipeline.py` + `src/db/model_loader.py`

**Apply to:** All new Optional model fields in SubmodelSet

The pattern for adding a new model component to SubmodelSet requires updates in exactly 3 files:
1. `domain/models.py`: Add `TYPE_CHECKING` import + Optional field to dataclass
2. `training_pipeline.py`: Add import + training block + SubmodelSet field + MLflow save + local save
3. `db/model_loader.py`: Add import + MLflow load + local load + SubmodelSet field

The order is: `place_selection_gate` was added this way, then `win_benter`/`win_isotonic_calibrator`/`win_temperature_scaler` were added the same way. Follow the same pattern for `win_selection_gate`.

### Win vs Place Column Mapping Convention

**Source:** `src/models/place_selection_gate.py`

**Apply to:** `src/models/win_selection_gate.py` and `src/backtest/race_predictor.py`

Consistent naming convention:
- Odds source: `fukuoddslow` -> `tanoddslow`
- Selection prefix: `place_selection_*` -> `win_selection_*`
- Gate prefix: `place_gate_*` -> `win_gate_*`
- EV source priority: `EV_lower_place`/`ev_place_corrected`/`ev_place_direct` -> `EV_lower_win_corrected`/`ev_win_corrected`/`ev_win`
- Prob source priority: `p_place_corrected` -> `p_win_final` -> `p_win_combined` -> `p_win_corrected`
- Hit condition: `kakuteijyuni <= 3` -> `kakuteijyuni == 1`

### Edge Threshold Update (RegimeDetector + MetaSwitcher sync)

**Source:** `src/models/regime_detector.py` + `src/betting/meta_switcher.py`

**Apply to:** All edge_threshold values

Both files must be updated in lockstep. RegimeDetector is used by BacktestEngine/RacePredictor, MetaSwitcher is used by BettingOrchestrator. Current delta between them (+0.01 in MetaSwitcher) should be preserved:
- RegimeDetector: 0.05 / 0.06 / 0.09
- MetaSwitcher: 0.05 / 0.07 / 0.10

### GateKeeper Default Threshold

**Source:** `src/betting/gate_keeper.py`

**Apply to:** Default edge_threshold parameter

Update from 0.03 to 0.04 to reflect JRA takeout baseline. The actual runtime value comes from MetaSwitcher/RegimeDetector, so this default is only a safety net.

### Model Serialization (joblib)

**Source:** `src/models/place_selection_gate.py` lines 979-1044

**Apply to:** `src/models/win_selection_gate.py`

Pattern:
1. `save(path)`: Collect all attributes into a dict, `joblib.dump(state, path)`
2. `load(path)`: `joblib.load(path)`, create new instance with constructor params, assign remaining attributes with `.get(key, default)` for backward compatibility

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | -- | -- | All files have either exact analogs (new file from Place) or self-modification patterns |

## Metadata

**Analog search scope:** `src/models/`, `src/betting/`, `src/backtest/`, `src/pipelines/`, `src/db/`, `src/domain/`, `tests/`
**Files scanned:** 12 source files + 2 test files
**Pattern extraction date:** 2026-05-02
