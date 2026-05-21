---
phase: 19-ev-calibration
reviewed: 2026-05-07T12:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - src/db/model_loader.py
  - src/domain/models.py
  - src/models/ev_correction_model.py
  - src/models/ev_diagnostics.py
  - src/pipelines/training_pipeline.py
  - tests/test_ev_correction.py
  - tests/test_ev_isotonic.py
findings:
  critical: 2
  warning: 2
  info: 0
  total: 4
status: issues_found
---

# Phase 19: Code Review Report

**Reviewed:** 2026-05-07T12:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Reviewed the Phase 19 EV Isotonic calibration + odds band scaling implementation across 7 files. Two BLOCKER issues found, both in `model_loader.py`, which together render the entire Phase 19 calibration ineffective at runtime. The training path correctly fits the Isotonic calibrator and band scales, and the `EVCorrectionModel` correctly applies them when configured -- but the model loader never wires these components into the `EVCorrectionModel` instance, so all downstream consumers (race_predictor, backtest, etc.) get uncalibrated EV predictions.

Additionally, a redundant MLflow call in the fallback path can crash the loader when MLflow is unavailable.

## Critical Issues

### CR-01: Model loader does not inject EV Isotonic calibrator or odds band scales into EVCorrectionModel instance

**File:** `src/db/model_loader.py:104-106` (MLflow path) and `src/db/model_loader.py:566-572` (local dir path)
**Issue:** In both `load()` and `load_from_dir()`, the `EVCorrectionModel` is instantiated without arguments (`EVCorrectionModel()`), and only the P/E correction LightGBM models are set on it. The `ev_isotonic_calibrator` and `ev_odds_band_scales` are loaded separately and stored in the `SubmodelSet` dataclass fields -- but they are **never set on the `EVCorrectionModel` instance itself**.

When `correct_ev()` runs (called by `race_predictor.py:117`, `win_benter_gate.py:131`, etc.), it checks `self.ev_isotonic_calibrator` and `self.ev_odds_band_scales` -- both remain `None`. The Isotonic calibration and odds band scaling are silently skipped, producing uncalibrated `ev_win_calibrated = ev_win_corrected` (the fallback path at line 352 of ev_correction_model.py).

This means the entire Phase 19 calibration has no effect at prediction time. The training pipeline correctly generates the calibrator and scales, they get saved to disk/MLflow, they get loaded back into `SubmodelSet` fields -- but the `EVCorrectionModel` that actually applies them never sees them.

**Fix:**
```python
# In load() -- after line 106:
ev_corr = EVCorrectionModel()
ev_corr.p_correction_model = self._load_lgbm(f"{artifact_uri}/ev_corrector_p_{surface}")
ev_corr.e_correction_model = self._load_lgbm(f"{artifact_uri}/ev_corrector_e_{surface}")
# DO NOT set ev_isotonic_calibrator/ev_odds_band_scales here yet --
# they are loaded later. Instead, after loading them (lines 291-309), set them:
ev_corr.ev_isotonic_calibrator = ev_isotonic_calibrator
ev_corr.ev_odds_band_scales = ev_odds_band_scales

# Same pattern for load_from_dir() after lines 566-572 and 707-724.
```

Alternatively, restructure to construct `EVCorrectionModel` last, passing both the calibrator and band scales into the constructor:
```python
ev_corr = EVCorrectionModel(
    ev_isotonic_calibrator=ev_isotonic_calibrator,
    ev_odds_band_scales=ev_odds_band_scales,
)
ev_corr.p_correction_model = self._load_lgbm(...)
ev_corr.e_correction_model = self._load_lgbm(...)
```

### CR-02: Redundant mlflow.get_artifact_uri() call crashes loader when MLflow is unavailable

**File:** `src/db/model_loader.py:73`
**Issue:** Line 73 unconditionally calls `artifact_uri = mlflow.get_artifact_uri(run_id)`, overwriting the `artifact_uri` value. This line is reached regardless of whether the try-block at lines 59-65 succeeded or the fallback at lines 67-70 was used. If MLflow is unavailable (the exact scenario that triggers the fallback), this second call will also throw an exception, and since it is outside any try/except block, the entire `load()` method crashes. This defeats the purpose of the filesystem fallback mechanism.

Even when MLflow is available, this is a redundant duplicate of line 65.

**Fix:**
Remove line 73 entirely. The `artifact_uri` is already correctly set either by line 65 (MLflow path) or by line 68 (filesystem fallback path):
```python
        try:
            run = mlflow.get_run(run_id)
            params = run.data.params
            train_end = params.get("train_end", "unknown")
            train_start = params.get("train_start", "2020-01-01")
            quality_threshold = float(params.get("quality_threshold", "0.0"))
            artifact_uri = mlflow.get_artifact_uri(run_id)
        except Exception:
            artifact_uri, train_start, train_end, quality_threshold = self._resolve_run_from_fs(
                run_id
            )

        surfaces = ["turf", "dirt"]
        # REMOVED: artifact_uri = mlflow.get_artifact_uri(run_id)
```

## Warnings

### WR-01: EVCorrectionModel lacks _trained guard -- no graceful fallback if correct_ev() called before train()

**File:** `src/models/ev_correction_model.py:306`
**Issue:** `EVCorrectionModel.correct_ev()` accesses `self.p_correction_model` and `self.e_correction_model` without any guard. If `correct_ev()` is called before `train()`, these attributes are undefined, causing an unhandled `AttributeError`. By contrast, `PlaceEVCorrectionModel` (same file, line 567) has a `_trained` flag that provides a graceful passthrough fallback. This inconsistency is a reliability risk.

**Fix:**
Add a `_trained` attribute and a fallback path mirroring `PlaceEVCorrectionModel`:
```python
def __init__(self, ...) -> None:
    self.ev_isotonic_calibrator = ev_isotonic_calibrator
    self.ev_odds_band_scales = ev_odds_band_scales
    self._trained: bool = False
    ...

def correct_ev(self, df: pd.DataFrame) -> pd.DataFrame:
    if not self._trained:
        df = df.copy()
        df["p_win_corrected"] = _normalize_probability_by_race(
            df, "p_win_pred", target_sum=1.0,
        )
        df["e_return_win_corrected"] = df["e_return_win_pred"].copy()
        df["ev_win_corrected"] = df["p_win_corrected"] * df["e_return_win_corrected"]
        df["ev_win_calibrated"] = df["ev_win_corrected"].copy()
        return df
    ...
```

### WR-02: OOF sort-by-race_date test only verifies mock was called, not that data was sorted

**File:** `tests/test_ev_isotonic.py:450`
**Issue:** `test_generate_ev_oof_sorts_by_race_date` is intended to verify that `generate_ev_oof_predictions` sorts the input DataFrame by `race_date` before splitting. However, the only assertion is `mock_win.train_hit_model.assert_called()`, which merely checks that the mock was invoked. It does not verify that the data passed to the mock was sorted. A regression that removes the sort would still pass this test.

**Fix:**
Use a side-effect callback to capture the DataFrame passed to `train_hit_model`, then assert that its `race_date` column is monotonically increasing:
```python
captured_dfs = []
mock_win.train_hit_model.side_effect = lambda d: captured_dfs.append(d)
...
TrainingPipelineV5.generate_ev_oof_predictions(df, n_splits=3, num_threads=1)
for train_df in captured_dfs:
    assert train_df["race_date"].is_monotonic_increasing, (
        "Training data must be sorted by race_date"
    )
```

---

_Reviewed: 2026-05-07T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
