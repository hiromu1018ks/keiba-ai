# Place Probability Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 2.38x place probability overestimation by adding isotonic calibration, race-sum normalization, and learned Benter combination to the PlaceTwoStageModel pipeline.

**Architecture:** Three-layer calibration stack: (1) Isotonic regression fitted on validation predictions within `train_hit_model()`, (2) race-sum normalization to ensure `sum(p_place) ~ 3.0` per race, (3) logistic regression to learn optimal model/market weight combination replacing fixed `alpha=0.4`. Each layer is independent and testable in isolation.

**Tech Stack:** scikit-learn `IsotonicRegression`, `LogisticRegression`; LightGBM; numpy/pandas; pytest with unittest.mock

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/models/two_stage_return_model.py` | Modify | Isotonic calibration + race-sum normalization in `PlaceTwoStageModel` |
| `src/domain/models.py` | Modify | Add `benter_lr` field to `SubmodelSet` dataclass |
| `src/pipelines/training_pipeline.py` | Modify | Calibrator fitting, Benter LR training, serialization |
| `src/backtest/race_predictor.py` | Modify | Use `benter_lr` instead of fixed alpha |
| `src/db/model_loader.py` | Modify | Load `benter_lr` and `_place_calibrator` from disk |
| `tests/test_two_stage_return_model.py` | Modify | Tests for calibration + normalization |
| `tests/test_domain.py` | Modify | Tests for SubmodelSet field |
| `tests/test_race_predictor.py` | Modify/Create | Tests for benter_lr inference |

---

## Task 1: Isotonic Calibration Layer for PlaceTwoStageModel

**Files:**
- Modify: `src/models/two_stage_return_model.py:262-285` (`train_hit_model`)
- Modify: `src/models/two_stage_return_model.py:324-337` (`predict_ev`)
- Test: `tests/test_two_stage_return_model.py`

### Step 1.1: Write failing test — train_hit_model stores validation predictions

Add to `TestPlaceTwoStageModel` in `tests/test_two_stage_return_model.py` (after line 271):

```python
@patch("models.two_stage_return_model.lgb")
def test_train_hit_model_stores_val_predictions(
    self, mock_lgb: MagicMock, feature_df: pd.DataFrame
) -> None:
    """train_hit_model が _val_predictions と _val_labels を保存すること"""
    mock_booster = MagicMock()
    mock_booster.best_iteration = 50
    mock_booster.predict.return_value = np.array([0.3, 0.7, 0.4, 0.6])
    mock_lgb.train.return_value = mock_booster
    mock_lgb.Dataset.return_value = MagicMock()
    mock_lgb.early_stopping.return_value = lambda: None

    df = feature_df.copy()
    df["kakuteijyuni"] = [1, 2, 3, 4, 5, 6, 7, 8]

    model = PlaceTwoStageModel()
    model.train_hit_model(df)

    assert hasattr(model, "_val_predictions")
    assert hasattr(model, "_val_labels")
    # 8 rows * 20% = 1.6 → int(8*0.8)=6, so 2 validation samples
    assert len(model._val_predictions) == 2
    assert len(model._val_labels) == 2
    # Val labels: kakuteijyuni <= 3 → [False, False] for rows 7,8
    np.testing.assert_array_equal(model._val_labels, [0, 0])
```

### Step 1.2: Run test to verify it fails

Run: `python -m pytest tests/test_two_stage_return_model.py::TestPlaceTwoStageModel::test_train_hit_model_stores_val_predictions -v`
Expected: FAIL — `_val_predictions` attribute does not exist yet.

### Step 1.3: Implement — initialize attributes in `__init__` and inline split in `train_hit_model`

First, update `PlaceTwoStageModel.__init__` in `src/models/two_stage_return_model.py` (line 245-246) to initialize calibration attributes:

```python
def __init__(self, cfg: TwoStageConfig | None = None) -> None:
    self.cfg = cfg or TwoStageConfig()
    self._place_calibrator: IsotonicRegression | None = None
    self._val_predictions: np.ndarray | None = None
    self._val_labels: np.ndarray | None = None
```

Add the required import at the top of the file (after line 7):

```python
import numpy as np
from sklearn.isotonic import IsotonicRegression
```

**Note:** Use `TYPE_CHECKING` guard if sklearn should not be imported at module level:

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklearn.isotonic import IsotonicRegression
```

Then replace `PlaceTwoStageModel.train_hit_model` (lines 262-285) with:

```python
def train_hit_model(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
    """P(place) の学習 (3着以内=1 / それ以外=0)

    80/20 時系列分割で学習し、バリデーション予測を保存して
    後続の Isotonic 校正に使用する。
    """
    if num_threads <= 0:
        num_threads = max(1, (os.cpu_count() or 4) // 2)
    features = self._prepare_features(df, use_cols=self.HIT_FEATURE_COLS)
    y = (df["kakuteijyuni"] <= 3).astype(int)

    # Inline split to capture raw validation data for calibration
    n = len(features)
    split = int(n * 0.8)
    train_data = lgb.Dataset(features.iloc[:split], label=y.iloc[:split])
    valid_data = lgb.Dataset(
        features.iloc[split:], label=y.iloc[split:], reference=train_data
    )

    self.hit_model = lgb.train(
        {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": self.cfg.hit_lr,
            "num_leaves": self.cfg.hit_leaves,
            "is_unbalance": True,
            "feature_fraction": 0.7,
            "num_threads": num_threads,
            "verbose": -1,
        },
        train_data,
        num_boost_round=self.cfg.hit_rounds,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    # Store validation predictions for isotonic calibration
    self._val_predictions = self.hit_model.predict(features.iloc[split:])
    self._val_labels = y.iloc[split:].values
```

### Step 1.4: Run test to verify it passes

Run: `python -m pytest tests/test_two_stage_return_model.py::TestPlaceTwoStageModel::test_train_hit_model_stores_val_predictions -v`
Expected: PASS

### Step 1.5: Write failing test — fit_calibrator creates IsotonicRegression

Add to `TestPlaceTwoStageModel`:

```python
def test_fit_calibrator_creates_isotonic(self) -> None:
    """fit_calibrator が _place_calibrator を作成すること"""
    model = PlaceTwoStageModel()
    model._val_predictions = np.random.rand(1500)
    model._val_labels = (model._val_predictions > 0.5).astype(int)

    model.fit_calibrator()

    assert model._place_calibrator is not None
    from sklearn.isotonic import IsotonicRegression
    assert isinstance(model._place_calibrator, IsotonicRegression)
```

### Step 1.6: Run test to verify it fails

Run: `python -m pytest tests/test_two_stage_return_model.py::TestPlaceTwoStageModel::test_fit_calibrator_creates_isotonic -v`
Expected: FAIL — `AttributeError: 'PlaceTwoStageModel' object has no attribute 'fit_calibrator'`

### Step 1.7: Implement — add `fit_calibrator` method

Add to `PlaceTwoStageModel` in `src/models/two_stage_return_model.py` (after `train_hit_model`, before `train_return_model`):

```python
def fit_calibrator(self) -> None:
    """バリデーション予測に Isotonic 校正を適合させる。

    train_hit_model() またはアンサンブル学習後に呼び出すこと。
    サンプル数 < 1000 の場合は校正をスキップ (過学習リスク)。
    """
    from sklearn.isotonic import IsotonicRegression

    if self._val_predictions is None or len(self._val_predictions) < 1000:
        self._place_calibrator = None
        return

    self._place_calibrator = IsotonicRegression(out_of_bounds="clip")
    self._place_calibrator.fit(self._val_predictions, self._val_labels)
```

### Step 1.8: Run test to verify it passes

Run: `python -m pytest tests/test_two_stage_return_model.py::TestPlaceTwoStageModel::test_fit_calibrator_creates_isotonic -v`
Expected: PASS

### Step 1.9: Write failing test — predict_ev applies calibration

Add to `TestPlaceTwoStageModel`:

```python
def test_predict_ev_applies_isotonic_calibration(
    self, feature_df: pd.DataFrame
) -> None:
    """predict_ev が _place_calibrator を適用して p_place_pred を補正すること"""
    model = PlaceTwoStageModel()
    model.hit_model = _make_mock_booster([0.40, 0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01])
    model.return_model = _make_mock_booster([1.4, 1.7, 2.2, 3.8, 5.5, 9.0, 16.0, 32.0])

    # Fit a fake calibrator that maps p → p * 0.5
    from sklearn.isotonic import IsotonicRegression
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(np.array([0.01, 0.5, 0.99]), np.array([0.005, 0.25, 0.495]))
    model._place_calibrator = cal

    result = model.predict_ev(feature_df)

    # Raw predictions from mock: [0.40, 0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01]
    # After calibration (roughly p * 0.5): values should be approximately halved
    raw_preds = np.array([0.40, 0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01])
    expected = cal.transform(raw_preds)
    np.testing.assert_allclose(result["p_place_pred"].values, expected, rtol=1e-6)
```

### Step 1.10: Run test to verify it fails

Run: `python -m pytest tests/test_two_stage_return_model.py::TestPlaceTwoStageModel::test_predict_ev_applies_isotonic_calibration -v`
Expected: FAIL — predict_ev returns raw predictions, not calibrated ones.

### Step 1.11: Implement — apply calibration in `predict_ev`

In `src/models/two_stage_return_model.py`, replace `PlaceTwoStageModel.predict_ev` (lines 324-337) with:

```python
def predict_ev(self, df: pd.DataFrame) -> pd.DataFrame:
    """EV_place = P(place) × E(place_odds | place)

    Isotonic 校正 + レース内正規化を適用後、EV を計算する。
    """
    df = df.copy()
    hit_features = self._prepare_features(df, use_cols=self.HIT_FEATURE_COLS)
    ret_features = self._prepare_features(df, use_cols=self.RETURN_FEATURE_COLS)

    hit_iter = self.hit_model.best_iteration if self.hit_model.best_iteration > 0 else None
    ret_iter = (
        self.return_model.best_iteration if self.return_model.best_iteration > 0 else None
    )

    # --- Isotonic calibration ---
    raw_p = self.hit_model.predict(hit_features, num_iteration=hit_iter)
    if self._place_calibrator is not None:
        p_calibrated = self._place_calibrator.transform(raw_p)
    else:
        p_calibrated = raw_p
    df["p_place_pred"] = p_calibrated

    # --- Return model ---
    df["e_return_place_pred"] = self.return_model.predict(ret_features, num_iteration=ret_iter)
    df["ev_place"] = df["p_place_pred"] * df["e_return_place_pred"]
    return df
```

### Step 1.12: Run test to verify it passes

Run: `python -m pytest tests/test_two_stage_return_model.py::TestPlaceTwoStageModel::test_predict_ev_applies_isotonic_calibration -v`
Expected: PASS

### Step 1.13: Handle ensemble mode in training pipeline

In `src/pipelines/training_pipeline.py`, after the ensemble path assigns `place_2s.hit_model = ensemble_place` (line 470), add validation prediction storage:

```python
        if use_ensemble:
            from models.stacked_ensemble import StackedEnsemble

            with TimingContext(f"{surface}/place_hit_ensemble"):
                features = place_2s._prepare_features(df_oof, use_cols=place_2s.HIT_FEATURE_COLS)
                y = (df_oof["kakuteijyuni"] <= 3).astype(int)
                split = int(len(features) * 0.8)
                ensemble_place = StackedEnsemble(
                    cat_cols=["surface", "distance_bin", "grade_code"]
                )
                ensemble_place.train(
                    features.iloc[:split], y.iloc[:split],
                    features.iloc[split:], y.iloc[split:],
                    num_threads=num_threads,
                )
                place_2s.hit_model = ensemble_place
                # Store validation predictions for isotonic calibration
                place_2s._val_predictions = ensemble_place.predict(features.iloc[split:])
                place_2s._val_labels = y.iloc[split:].values
        else:
            with TimingContext(f"{surface}/place_hit"):
                place_2s.train_hit_model(df_oof, num_threads=num_threads)

        # Fit isotonic calibrator (both paths)
        place_2s.fit_calibrator()

        with TimingContext(f"{surface}/place_return"):
            place_2s.train_return_model(df_oof, num_threads=num_threads)
```

### Step 1.14: Write test — fit_calibrator skips when samples < 1000

Add to `TestPlaceTwoStageModel`:

```python
def test_fit_calibrator_skips_below_min_samples(self) -> None:
    """サンプル数 < 1000 の場合、校正をスキップすること"""
    model = PlaceTwoStageModel()
    model._val_predictions = np.random.rand(500)
    model._val_labels = (model._val_predictions > 0.5).astype(int)

    model.fit_calibrator()

    assert model._place_calibrator is None
```

### Step 1.15: Run all tests

Run: `python -m pytest tests/test_two_stage_return_model.py -v`
Expected: ALL PASS (including existing tests)

### Step 1.16: Commit

```bash
git add src/models/two_stage_return_model.py tests/test_two_stage_return_model.py src/pipelines/training_pipeline.py
git commit -m "feat: add isotonic calibration layer to PlaceTwoStageModel

- train_hit_model stores validation predictions for calibration
- fit_calibrator() fits IsotonicRegression on val predictions (min 1000 samples)
- predict_ev() applies calibration before computing EV
- Ensemble mode also stores val predictions for calibration"
```

---

## Task 2: Race-Sum Normalization

**Files:**
- Modify: `src/models/two_stage_return_model.py:324-337` (`predict_ev`, already modified in Task 1)
- Test: `tests/test_two_stage_return_model.py`

### Step 2.1: Write failing test — predict_ev normalizes to sum ~3 per race

Add to `TestPlaceTwoStageModel`:

```python
def test_predict_ev_race_sum_normalization(self) -> None:
    """predict_ev がレース内で sum(p_place_pred) ≈ 3.0 に正規化すること"""
    model = PlaceTwoStageModel()
    # Raw probabilities sum > 3.0 (typical overestimation pattern)
    model.hit_model = _make_mock_booster([0.70, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30])
    model.return_model = _make_mock_booster([1.4, 1.7, 2.2, 3.8, 5.5, 9.0, 16.0, 32.0])
    model._place_calibrator = None  # Skip calibration for this test

    result = model.predict_ev(feature_df)

    # sum(p_place_pred) should be ~ 3.0 per race
    race_sum = result.groupby("race_id")["p_place_pred"].sum()
    np.testing.assert_allclose(race_sum.values, 3.0, rtol=1e-6)
```

### Step 2.2: Run test to verify it fails

Run: `python -m pytest tests/test_two_stage_return_model.py::TestPlaceTwoStageModel::test_predict_ev_race_sum_normalization -v`
Expected: FAIL — raw probabilities sum to ~3.85, not 3.0.

### Step 2.3: Implement — add race-sum normalization in `predict_ev`

Update `predict_ev` in `src/models/two_stage_return_model.py` (the version from Task 1, Step 1.11). After the isotonic calibration block and before the return model block, add normalization:

```python
def predict_ev(self, df: pd.DataFrame) -> pd.DataFrame:
    """EV_place = P(place) × E(place_odds | place)

    Isotonic 校正 → レース内正規化 → EV 計算のパイプライン。
    """
    df = df.copy()
    hit_features = self._prepare_features(df, use_cols=self.HIT_FEATURE_COLS)
    ret_features = self._prepare_features(df, use_cols=self.RETURN_FEATURE_COLS)

    hit_iter = self.hit_model.best_iteration if self.hit_model.best_iteration > 0 else None
    ret_iter = (
        self.return_model.best_iteration if self.return_model.best_iteration > 0 else None
    )

    # --- Isotonic calibration ---
    raw_p = self.hit_model.predict(hit_features, num_iteration=hit_iter)
    if self._place_calibrator is not None:
        p_calibrated = self._place_calibrator.transform(raw_p)
    else:
        p_calibrated = raw_p
    df["p_place_pred"] = p_calibrated

    # --- Race-sum normalization: sum(p_place) ~ 3.0 per race ---
    race_sum = df.groupby("race_id")["p_place_pred"].transform("sum")
    df["p_place_pred"] = df["p_place_pred"] * (3.0 / race_sum)

    # --- Consistency constraint: p_place >= p_ability_win ---
    if "p_ability_win" in df.columns:
        mask = df["p_place_pred"] < df["p_ability_win"]
        df.loc[mask, "p_place_pred"] = df.loc[mask, "p_ability_win"]
        race_sum = df.groupby("race_id")["p_place_pred"].transform("sum")
        df["p_place_pred"] = df["p_place_pred"] * (3.0 / race_sum)

    # --- Final clip ---
    df["p_place_pred"] = df["p_place_pred"].clip(0.01, 0.99)

    # --- Return model ---
    df["e_return_place_pred"] = self.return_model.predict(ret_features, num_iteration=ret_iter)
    df["ev_place"] = df["p_place_pred"] * df["e_return_place_pred"]
    return df
```

### Step 2.4: Run test to verify it passes

Run: `python -m pytest tests/test_two_stage_return_model.py::TestPlaceTwoStageModel::test_predict_ev_race_sum_normalization -v`
Expected: PASS

### Step 2.5: Write test — consistency constraint p_place >= p_ability_win

Add to `TestPlaceTwoStageModel`:

```python
def test_predict_ev_consistency_constraint(self) -> None:
    """p_place_pred >= p_ability_win の整合性制約が機能すること"""
    model = PlaceTwoStageModel()
    model.hit_model = _make_mock_booster([0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01])
    model.return_model = _make_mock_booster([1.4, 1.7, 2.2, 3.8, 5.5, 9.0, 16.0, 32.0])
    model._place_calibrator = None

    df = feature_df.copy()
    # Set p_ability_win high for horse 0 — should enforce floor
    df["p_ability_win"] = [0.50, 0.25, 0.20, 0.10, 0.08, 0.04, 0.02, 0.01]

    result = model.predict_ev(df)

    # After normalization, p_place_pred should be >= p_ability_win for all horses
    assert (result["p_place_pred"] >= result["p_ability_win"] - 1e-10).all()
    # Race sum should still be ~ 3.0
    race_sum = result.groupby("race_id")["p_place_pred"].sum()
    np.testing.assert_allclose(race_sum.values, 3.0, rtol=1e-6)
```

### Step 2.6: Run test to verify it passes

Run: `python -m pytest tests/test_two_stage_return_model.py::TestPlaceTwoStageModel::test_predict_ev_consistency_constraint -v`
Expected: PASS (the code from Step 2.3 already handles this)

### Step 2.7: Run all tests

Run: `python -m pytest tests/test_two_stage_return_model.py -v`
Expected: ALL PASS

### Step 2.8: Commit

```bash
git add src/models/two_stage_return_model.py tests/test_two_stage_return_model.py
git commit -m "feat: add race-sum normalization to PlaceTwoStageModel.predict_ev

- Normalize p_place_pred so sum ~ 3.0 per race (JRA top-3 place)
- Enforce consistency: p_place >= p_ability_win
- Re-normalize after consistency constraint
- Clip final probabilities to [0.01, 0.99]"
```

---

## Task 3: Benter LR Infrastructure (SubmodelSet + Serialization)

**Files:**
- Modify: `src/domain/models.py:220-237` (`SubmodelSet`)
- Modify: `src/pipelines/training_pipeline.py:803-895` (`_save_models_local`)
- Modify: `src/db/model_loader.py:429-474` (`_load_from_local`)
- Test: `tests/test_domain.py`

### Step 3.1: Write failing test — SubmodelSet accepts benter_lr

Add to the relevant test class in `tests/test_domain.py` (find existing SubmodelSet tests):

```python
def test_submodel_set_accepts_benter_lr(self) -> None:
    """SubmodelSet が benter_lr フィールドを受け入れること"""
    from unittest.mock import MagicMock
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(fit_intercept=True, penalty=None)
    # Quick fit on dummy data
    X = np.array([[0.5, -0.5], [1.0, 0.5]])
    y = np.array([0, 1])
    lr.fit(X, y)

    sub = SubmodelSet(
        market=MagicMock(),
        stage1=MagicMock(),
        place_ability=MagicMock(),
        win=MagicMock(),
        ev_corrector=MagicMock(),
        place=MagicMock(),
        place_ev_corrector=MagicMock(),
        wide=MagicMock(),
        confidence=MagicMock(),
        use_ensemble=False,
        benter_lr=lr,
    )
    assert sub.benter_lr is not None
    assert sub.benter_lr is lr
```

### Step 3.2: Run test to verify it fails

Run: `python -m pytest tests/test_domain.py -v -k "benter_lr"`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'benter_lr'`

### Step 3.3: Implement — add benter_lr field to SubmodelSet

In `src/domain/models.py`, add `benter_lr` field to `SubmodelSet` (line 237, before `use_ensemble`):

```python
@dataclass
class SubmodelSet:
    """サブモデル（芝/ダート）のセット

    TrainingPipelineV5 が各 surface ごとに生成する。
    """

    market: MarketModel
    stage1: AbilityModel
    place_ability: PlaceAbilityModel
    win: WinTwoStageModel
    ev_corrector: EVCorrectionModel
    place: PlaceTwoStageModel
    place_ev_corrector: PlaceEVCorrectionModel
    wide: WideTwoStageModel
    confidence: RobustConfidenceEstimator
    use_ensemble: bool = False
    benter_lr: LogisticRegression | None = None
```

Add the import at the top of the file (if `LogisticRegression` is not already imported):

```python
from sklearn.linear_model import LogisticRegression
```

**Note:** If the file uses `TYPE_CHECKING` guard for imports to avoid sklearn at import time, use:

```python
if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression
```

And change the field type annotation to a string:

```python
    benter_lr: "LogisticRegression | None" = None
```

### Step 3.4: Run test to verify it passes

Run: `python -m pytest tests/test_domain.py -v -k "benter_lr"`
Expected: PASS

### Step 3.5: Implement — save benter_lr and place_calibrator in `_save_models_local`

In `src/pipelines/training_pipeline.py`, inside the `for surface, sub in models.items():` loop in `_save_models_local()` (after line 849), add:

```python
            # Place calibrator (IsotonicRegression)
            if hasattr(sub.place, "_place_calibrator") and sub.place._place_calibrator is not None:
                joblib.dump(
                    sub.place._place_calibrator,
                    models_dir / f"place_calibrator_{surface}.joblib",
                )

            # Benter logistic regression
            if sub.benter_lr is not None:
                joblib.dump(
                    sub.benter_lr,
                    models_dir / f"benter_lr_{surface}.joblib",
                )
```

### Step 3.6: Implement — load benter_lr and place_calibrator in model loader

In `src/db/model_loader.py`, after the PlaceTwoStageModel loading block (after line 434), add:

```python
            # Place calibrator (IsotonicRegression)
            calibrator_file = models_dir / f"place_calibrator_{surface}.joblib"
            if calibrator_file.is_file():
                try:
                    place._place_calibrator = joblib.load(calibrator_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", calibrator_file)
```

After the SubmodelSet construction (after line 474, before the closing of the surface loop), add `benter_lr` loading. The `SubmodelSet` constructor needs to receive it:

```python
            # Benter logistic regression
            benter_lr = None
            benter_file = models_dir / f"benter_lr_{surface}.joblib"
            if benter_file.is_file():
                try:
                    benter_lr = joblib.load(benter_file)
                except Exception:
                    logger.warning("Failed to load %s, skipping", benter_file)

            submodels[surface] = SubmodelSet(
                market=market,
                stage1=ability,
                place_ability=pa,
                win=win,
                ev_corrector=ev_corr,
                place=place,
                place_ev_corrector=place_ev_corr,
                wide=wide,
                confidence=confidence,
                benter_lr=benter_lr,
            )
```

### Step 3.7: Implement — save benter_lr and place_calibrator in `_log_to_mlflow`

In `src/pipelines/training_pipeline.py`, inside the `_log_to_mlflow()` method, add serialization for the new objects following the existing pattern (which uses `tempfile` + `joblib.dump` + `mlflow.log_artifact`):

```python
            # Place calibrator (IsotonicRegression)
            if hasattr(sub.place, "_place_calibrator") and sub.place._place_calibrator is not None:
                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                    joblib.dump(sub.place._place_calibrator, f.name)
                    mlflow.log_artifact(f.name, artifact_path=f"model/place_calibrator_{surface}")

            # Benter logistic regression
            if sub.benter_lr is not None:
                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                    joblib.dump(sub.benter_lr, f.name)
                    mlflow.log_artifact(f.name, artifact_path=f"model/benter_lr_{surface}")
```

Similarly, update the MLflow `load()` path in `src/db/model_loader.py` (the `load()` method around lines 37-204) to download and load these artifacts when loading from MLflow.

### Step 3.8: Run all tests

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

### Step 3.9: Commit

```bash
git add src/domain/models.py src/pipelines/training_pipeline.py src/db/model_loader.py tests/test_domain.py
git commit -m "feat: add benter_lr to SubmodelSet with save/load serialization

- Add benter_lr: LogisticRegression | None field to SubmodelSet
- Save benter_lr and place_calibrator as joblib in _save_models_local
- Save both in _log_to_mlflow (tempfile + joblib + mlflow.log_artifact)
- Load both in model_loader._load_from_local (and MLflow load path)"
```

---

## Task 4: Benter LR Training + RacePredictor Update

**Files:**
- Modify: `src/pipelines/training_pipeline.py:477-520` (after Place TwoStage, before SubmodelSet construction)
- Modify: `src/backtest/race_predictor.py:119-141` (Benter combination logic)
- Test: `tests/test_race_predictor.py`

### Step 4.1: Write failing test — Benter LR is trained in pipeline

Check if `tests/test_race_predictor.py` exists. If not, create it.

Add test:

```python
"""src/backtest/race_predictor.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


class TestRacePredictorBenterLR:
    def test_predict_uses_benter_lr_when_available(self) -> None:
        """benter_lr が利用可能な場合、固定 alpha ではなく LR パラメータを使うこと"""
        from backtest.race_predictor import RacePredictor
        from domain.models import SubmodelSet, TrainedModelsV5

        # Create a fitted logistic regression
        lr = LogisticRegression(fit_intercept=True, penalty=None)
        X = np.array([[0.5, -0.5], [1.0, 0.5], [-0.5, 1.0], [0.3, 0.2]])
        y = np.array([1, 0, 0, 1])
        lr.fit(X, y)

        # Build mock models
        mock_sub = MagicMock(spec=SubmodelSet)
        mock_sub.benter_lr = lr
        mock_sub.use_ensemble = False

        # Mock the inference chain
        mock_sub.market.predict_and_calc_error.side_effect = lambda df: df
        mock_sub.stage1.add_ability_probs.side_effect = lambda df: df
        mock_sub.place_ability.predict.side_effect = lambda df: df
        mock_sub.win.predict_ev.side_effect = lambda df: df
        mock_sub.ev_corrector.correct_ev.side_effect = lambda df: df
        mock_sub.place.predict_ev.side_effect = lambda df: df
        mock_sub.place_ev_corrector.correct_ev.side_effect = lambda df: df

        mock_conf = MagicMock()
        mock_conf.predict_lower_bound.return_value = (df := pd.DataFrame(), pd.DataFrame())
        mock_sub.confidence = mock_conf

        mock_models = MagicMock(spec=TrainedModelsV5)
        mock_models.submodels = {"turf": mock_sub}
        mock_models.quality_screener = MagicMock()

        predictor = RacePredictor(mock_models, alpha=0.4)

        # Create race data
        race_df = pd.DataFrame({
            "race_id": ["R1"] * 4,
            "umaban": [1, 2, 3, 4],
            "surface": ["turf"] * 4,
            "kakuteijyuni": [1, 2, 3, 4],
            "fukuoddslow": [1.5, 2.0, 3.0, 5.0],
            "p_place_pred": [0.6, 0.5, 0.35, 0.2],
            "p_ability_win": [0.15, 0.12, 0.10, 0.08],
        })

        with patch("features.horse_history_features.HorseHistoryFeatures.add_race_transforms", side_effect=lambda df: df), \
             patch("features.interaction_features.compute_interaction_features", side_effect=lambda df: df):
            result = predictor.predict(race_df)

        # Should have p_place_combined and edge_place columns
        assert "p_place_combined" in result.columns
        assert "edge_place" in result.columns
        # edge should be computed from LR, not fixed alpha
        # The specific values depend on LR parameters, just verify they're computed
        assert not result["p_place_combined"].isna().all()

    def test_predict_falls_back_to_fixed_alpha_without_benter_lr(self) -> None:
        """benter_lr が None の場合、固定 alpha にフォールバックすること"""
        from backtest.race_predictor import RacePredictor
        from domain.models import SubmodelSet, TrainedModelsV5

        mock_sub = MagicMock(spec=SubmodelSet)
        mock_sub.benter_lr = None
        mock_sub.use_ensemble = False

        mock_sub.market.predict_and_calc_error.side_effect = lambda df: df
        mock_sub.stage1.add_ability_probs.side_effect = lambda df: df
        mock_sub.place_ability.predict.side_effect = lambda df: df
        mock_sub.win.predict_ev.side_effect = lambda df: df
        mock_sub.ev_corrector.correct_ev.side_effect = lambda df: df
        mock_sub.place.predict_ev.side_effect = lambda df: df
        mock_sub.place_ev_corrector.correct_ev.side_effect = lambda df: df

        mock_conf = MagicMock()
        mock_conf.predict_lower_bound.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_sub.confidence = mock_conf

        mock_models = MagicMock(spec=TrainedModelsV5)
        mock_models.submodels = {"turf": mock_sub}
        mock_models.quality_screener = MagicMock()

        predictor = RacePredictor(mock_models, alpha=0.4)

        race_df = pd.DataFrame({
            "race_id": ["R1"] * 4,
            "umaban": [1, 2, 3, 4],
            "surface": ["turf"] * 4,
            "kakuteijyuni": [1, 2, 3, 4],
            "fukuoddslow": [1.5, 2.0, 3.0, 5.0],
            "p_place_pred": [0.6, 0.5, 0.35, 0.2],
            "p_ability_win": [0.15, 0.12, 0.10, 0.08],
        })

        with patch("features.horse_history_features.HorseHistoryFeatures.add_race_transforms", side_effect=lambda df: df), \
             patch("features.interaction_features.compute_interaction_features", side_effect=lambda df: df):
            result = predictor.predict(race_df)

        assert "p_place_combined" in result.columns
        assert "edge_place" in result.columns

        # With alpha=0.4, manually compute expected:
        # p_market = [1/1.5, 1/2.0, 1/3.0, 1/5.0] = [0.667, 0.5, 0.333, 0.2]
        # logit_combined = 0.4 * logit(p_model) + 0.6 * logit(p_market)
        p_model = np.clip(np.array([0.6, 0.5, 0.35, 0.2]), 1e-6, 1 - 1e-6)
        p_mkt = np.clip(np.array([1/1.5, 1/2.0, 1/3.0, 1/5.0]), 1e-6, 1 - 1e-6)
        logit_m = np.log(p_model / (1 - p_model))
        logit_mk = np.log(p_mkt / (1 - p_mkt))
        logit_combined = 0.4 * logit_m + 0.6 * logit_mk
        expected = 1.0 / (1.0 + np.exp(-logit_combined))
        np.testing.assert_allclose(result["p_place_combined"].values, expected, rtol=1e-6)
```

### Step 4.2: Run tests to verify they fail

Run: `python -m pytest tests/test_race_predictor.py -v`
Expected: FAIL — `predict()` currently uses fixed alpha regardless of `benter_lr`.

### Step 4.3: Implement — update RacePredictor.predict() to use benter_lr

In `src/backtest/race_predictor.py`, replace the Benter combination block (lines 119-141) with:

```python
        # --- Value Betting with Benter combined probability ---
        # logit(p_combined) = alpha * logit(p_model) + (1-alpha) * logit(p_market)
        # OR: learned logistic regression coefficients if benter_lr available
        p_market = np.where(
            df["fukuoddslow"] > 0,
            1.0 / df["fukuoddslow"],
            np.nan,
        )

        # Clip to avoid logit(0) or logit(1) = ±inf
        p_model = np.clip(df["p_place_pred"], 1e-6, 1 - 1e-6)
        p_mkt = np.clip(p_market, 1e-6, 1 - 1e-6)

        logit_model = np.log(p_model / (1 - p_model))
        logit_market = np.log(p_mkt / (1 - p_mkt))

        if submodel.benter_lr is not None:
            # Learned Benter combination via logistic regression
            logit_combined = (
                submodel.benter_lr.coef_[0][0] * logit_model
                + submodel.benter_lr.coef_[0][1] * logit_market
                + submodel.benter_lr.intercept_[0]
            )
        else:
            # Fallback: fixed alpha
            logit_combined = self.alpha * logit_model + (1 - self.alpha) * logit_market

        p_combined = 1.0 / (1.0 + np.exp(-logit_combined))

        df["p_place_combined"] = p_combined
        df["edge_place"] = p_combined - p_mkt

        return df
```

### Step 4.4: Run tests to verify they pass

Run: `python -m pytest tests/test_race_predictor.py -v`
Expected: PASS

### Step 4.5: Write failing test — Benter LR training in pipeline

Add a test that validates the Benter LR is trained with correct features. This should be a unit test for the training step logic, using mock data.

Add to `tests/test_training_pipeline.py` (or create a new test file if needed):

```python
def test_benter_lr_training_produces_valid_coefficients(self) -> None:
    """Benter LR 学習が logit(p_model), logit(p_market) 特徴量で行われること"""
    from sklearn.linear_model import LogisticRegression

    # Simulate training data
    n = 2000
    rng = np.random.default_rng(42)
    p_model = rng.uniform(0.05, 0.95, n)
    p_market = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < (0.5 * p_model + 0.5 * p_market)).astype(int)

    p_m = np.clip(p_model, 1e-6, 1 - 1e-6)
    p_mk = np.clip(p_market, 1e-6, 1 - 1e-6)

    X = np.column_stack([
        np.log(p_m / (1 - p_m)),
        np.log(p_mk / (1 - p_mk)),
    ])

    lr = LogisticRegression(fit_intercept=True, penalty=None)
    lr.fit(X, y)

    # Both coefficients should be positive (more prob → more likely to place)
    assert lr.coef_[0][0] > 0, f"Model coef should be positive, got {lr.coef_[0][0]}"
    assert lr.coef_[0][1] > 0, f"Market coef should be positive, got {lr.coef_[0][1]}"
```

### Step 4.6: Run test to verify it passes (this is a pure sklearn test)

Run: `python -m pytest tests/test_training_pipeline.py -v -k "benter_lr"`
Expected: PASS (this validates the ML approach, not the pipeline integration)

### Step 4.7: Implement — add Benter LR training step in `_train_submodel`

In `src/pipelines/training_pipeline.py`, after Place EV correction (line 483) and before the Wide TwoStage Model (line 485), add:

```python
        # 5c. Benter logistic regression (data-driven model/market combination)
        benter_lr = None
        with TimingContext(f"{surface}/benter_lr"):
            from sklearn.linear_model import LogisticRegression

            # Guard: require valid fukuoddslow (non-NaN, positive) for Benter LR
            valid_mask = df_oof["fukuoddslow"].notna() & (df_oof["fukuoddslow"] > 0) & df_oof["p_place_pred"].notna()
            df_benter = df_oof[valid_mask].copy()

            if len(df_benter) >= 1000:
                p_m = df_benter["p_place_pred"].clip(1e-6, 1 - 1e-6)
                p_mk = (1.0 / df_benter["fukuoddslow"]).clip(1e-6, 1 - 1e-6)
                y_place = (df_benter["kakuteijyuni"] <= 3).astype(int)

                X_benter = np.column_stack([
                    np.log(p_m / (1 - p_m)),
                    np.log(p_mk / (1 - p_mk)),
                ])

                lr = LogisticRegression(fit_intercept=True, penalty=None)
                lr.fit(X_benter, y_place)

                alpha_lr = lr.coef_[0][0]
                beta_lr = lr.coef_[0][1]

                # Parameter validation: model weight should be positive
                if alpha_lr < 0 or beta_lr < 0:
                    logger.warning(
                        "Benter LR has unexpected coefficients: alpha=%.4f, beta=%.4f. "
                        "Falling back to fixed alpha.",
                        alpha_lr, beta_lr,
                    )
                else:
                    benter_lr = lr
                    logger.info(
                        "Benter LR: alpha=%.4f, beta=%.4f, gamma=%.4f",
                        alpha_lr, beta_lr, lr.intercept_[0],
                    )
            else:
                logger.warning(
                    "Insufficient valid samples for Benter LR: %d (need 1000)", len(df_benter)
                )
```

Then update the `SubmodelSet` construction (line 509) to include `benter_lr`:

```python
        return SubmodelSet(
            market=market,
            stage1=stage1,
            place_ability=place_ability,
            win=win_2s,
            ev_corrector=ev_corrector,
            place=place_2s,
            place_ev_corrector=place_ev_corrector,
            wide=wide_2s,
            confidence=conf,
            use_ensemble=use_ensemble,
            benter_lr=benter_lr,
        )
```

### Step 4.8: Run all tests

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

### Step 4.9: Commit

```bash
git add src/backtest/race_predictor.py src/pipelines/training_pipeline.py tests/test_race_predictor.py tests/test_training_pipeline.py
git commit -m "feat: add Benter logistic regression training and inference

- Train LogisticRegression on [logit(p_model), logit(p_market)] → place outcome
- Replace fixed alpha=0.4 with learned alpha/beta/gamma parameters
- Fallback to fixed alpha when benter_lr is None (backward compat)
- Parameter validation: warn and skip if coefficients are negative"
```

---

## Final Verification

### Run full test suite

```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### Run linting

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

### Run type check

```bash
mypy src/
```

### Summary of Changes

| Component | Before | After |
|-----------|--------|-------|
| `p_place_pred` | Raw LightGBM output | Isotonic calibrated + race-sum normalized |
| Race consistency | No constraint | `sum(p_place) ~ 3.0`, `p_place >= p_ability_win` |
| Benter combination | Fixed `alpha=0.4` | Learned `LogisticRegression` on `[logit(p_model), logit(p_market)]` |
| Bias correction | None | LR intercept absorbs systematic overestimation |
| Fallback | N/A | Fixed alpha when LR not available |
