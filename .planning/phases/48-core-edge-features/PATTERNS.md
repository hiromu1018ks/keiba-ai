# Phase 48: Core Edge Features - Pattern Map

**Mapped:** 2026-06-04
**Files analyzed:** 10 (new + modified)
**Analogs found:** 10 / 10

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/features/track_condition_features.py` | utility | transform | `src/features/interaction_features.py` | exact |
| `tests/test_track_condition_features.py` | test | transform | `tests/test_interaction_features.py` | exact |
| `src/features/feature_engine.py` (modified) | service | request-response | N/A (surgical insertion) | existing |
| `src/pipelines/training_pipeline.py` (modified) | service | batch | N/A (surgical insertion) | existing |
| `src/backtest/race_predictor.py` (modified) | service | request-response | N/A (surgical insertion) | existing |
| `src/models/stage1_ability_model.py` (modified) | model | batch | N/A (FEATURE_COLS append) | existing |
| `src/models/two_stage_return_model.py` (modified) | model | batch | N/A (FEATURE_COLS append) | existing |
| `src/models/ev_correction_model.py` (modified) | model | batch | N/A (FEATURE_COLS append) | existing |
| `src/models/place_ability_model.py` (modified) | model | batch | N/A (FEATURE_COLS append) | existing |
| `src/models/wide_two_stage_model.py` (modified) | model | batch | N/A (FEATURE_COLS append) | existing |

## Pattern Assignments

### `src/features/track_condition_features.py` (utility, transform)

**Analog:** `src/features/interaction_features.py`

**Imports pattern** (interaction_features.py lines 1-6):
```python
"""Group E: 交互作用特徴量 + v5 レースコンテキスト特徴量"""

from __future__ import annotations

import numpy as np
import pandas as pd
```

**Column constant pattern** (interaction_features.py lines 9-29):
```python
INTERACTION_COLS: list[str] = [
    # 既存 (3)
    "kyakusitu_x_distance",
    "kyakusitu_x_surface",
    ...
]
```
-- New module should define `TRACK_CONDITION_COLS: list[str]` with all T1/T2 feature names.

**Column existence guard + NaN-safe computation** (interaction_features.py lines 44-65):
```python
# LEAK防止: kyakusitukubun_cd (過去) のみ使用。kyakusitukubun (現在=ポスト) は不可。
if "kyakusitukubun_cd" in df.columns and "distance_bin" in df.columns:
    df["kyakusitu_x_distance"] = (
        df["kyakusitukubun_cd"].astype(str) + "_" + df["distance_bin"].astype(str)
    ).astype("category")

# NaNポリシー: いずれかがNaNなら結果もNaN (fillna(0)は使わない)
if weight_col in df.columns and "kyori" in df.columns:
    df["weight_x_distance"] = (df[weight_col] * df["kyori"]).where(
        df[weight_col].notna() & df["kyori"].notna(),
        other=float("nan"),
    )
```
-- Every feature computation must guard on column existence and propagate NaN via `.where()`.

**Surface mapping pattern** (interaction_features.py lines 125-130):
```python
if "norm_finish_logit_avg" in df.columns and "surface" in df.columns:
    surface_code = df["surface"].map({"turf": 1, "dirt": 2}).fillna(0)
    df["surface_x_past_perf"] = (df["norm_finish_logit_avg"] * surface_code).where(
        df["norm_finish_logit_avg"].notna(),
        other=float("nan"),
    )
```
-- Surface-aware computation: dirt features only apply to dirt races (NaN for turf), turf features only apply to turf races.

**Category type for interaction features** (interaction_features.py lines 70-73):
```python
if "surface" in df.columns and "distance_bin" in df.columns:
    df["surface_x_distance_bin"] = (
        df["surface"].astype(str) + "_" + df["distance_bin"].astype(str)
    ).astype("category")
```
-- `sire_x_cushion_band` should follow this pattern: string concatenation + `.astype("category")`.

**Core function signature** (interaction_features.py line 32):
```python
def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
```
-- New module should expose `compute_track_condition_features(df: pd.DataFrame) -> pd.DataFrame` returning `df.copy()` with new columns appended.

---

### `tests/test_track_condition_features.py` (test, transform)

**Analog:** `tests/test_interaction_features.py`

**Test file structure** (test_interaction_features.py lines 1-6):
```python
# tests/test_interaction_features.py
import pandas as pd
import pytest

from features.interaction_features import compute_interaction_features
```

**Per-feature test pattern** (test_interaction_features.py lines 8-18):
```python
def test_kyakusitu_x_distance():
    """脚質x距離bin の文字列結合"""
    df = pd.DataFrame(
        {
            "kyakusitukubun_cd": [1.0, 2.0, 3.0],
            "distance_bin": ["sprint", "mile", "intermediate"],
        }
    )
    result = compute_interaction_features(df)
    assert "kyakusitu_x_distance" in result.columns
    assert result["kyakusitu_x_distance"].tolist() == ["1.0_sprint", "2.0_mile", "3.0_intermediate"]
```

**NaN propagation test pattern** (test_interaction_features.py lines 47-58):
```python
def test_weight_x_distance_nan():
    """NaN伝播: いずれかがNaNなら結果もNaN"""
    df = pd.DataFrame(
        {
            "weight_absolute": [450.0, float("nan"), 500.0],
            "kyori": [1200, 2400, float("nan")],
        }
    )
    result = compute_interaction_features(df)
    assert result["weight_x_distance"].iloc[0] == 450.0 * 1200
    assert pd.isna(result["weight_x_distance"].iloc[1])
    assert pd.isna(result["weight_x_distance"].iloc[2])
```

**Missing column skip test** (test_interaction_features.py lines 89-94):
```python
def test_missing_columns():
    """必要列がない場合は追加しない"""
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    result = compute_interaction_features(df)
    assert "kyakusitu_x_distance" not in result.columns
    assert "weight_x_distance" not in result.columns
```

**Constant count test** (test_interaction_features.py lines 341-349):
```python
def test_interaction_cols_constant():
    """INTERACTION_COLS定数が15個の交互作用名を含む"""
    from features.interaction_features import INTERACTION_COLS
    assert len(INTERACTION_COLS) == 15
    assert "surface_x_distance_bin" in INTERACTION_COLS
```

**FEATURE_COLS registration test** (test_interaction_features.py lines 444-487):
```python
def test_all_models_have_new_features():
    """全12モデル+WideTwoStageのFEATURE_COLSに新規特徴量が含まれる"""
    from models.stage1_ability_model import AbilityModel
    from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
    ...
    model_lists = {
        "AbilityModel": AbilityModel.FEATURE_COLS,
        "WinTwoStageModel": WinTwoStageModel.FEATURE_COLS,
        ...
    }
    for model_name, feature_cols in model_lists.items():
        for feat in all_new:
            assert feat in feature_cols, f"{model_name} missing feature: {feat}"
```

---

### `src/features/feature_engine.py` (modified - build_all insertion)

**Analog:** BloodlineFeatures integration in build_all()

**Track condition merge pattern** (feature_engine.py lines 385-392):
```python
# Group B: 血統特徴量
if store is not None:
    with TimingContext("build_all/bloodline"):
        from features.bloodline_features import BloodlineFeatures

        bloodline = BloodlineFeatures(store)
        bloodline_df = bloodline.compute(result_df)
        result_df = pd.merge(result_df, bloodline_df, on=["race_id", "umaban"], how="left")
```
-- New insertion should follow the same pattern: `if store is not None`, `TimingContext`, lazy import, DataRepository(store).load_track_conditions(), merge on `race_id` with `how="left"`.

**Date range derivation pattern** (feature_engine.py lines 370-374):
```python
rd_valid = rd.dropna()
if len(rd_valid) > 0:
    start_str = rd_valid.min().strftime("%Y%m%d")
    end_str = rd_valid.max().strftime("%Y%m%d")
```
-- Use this pattern to derive start/end from `result_df["race_date"]` for `load_track_conditions(start, end)`.

---

### `src/pipelines/training_pipeline.py` (modified - _train_submodel insertion)

**Analog:** interaction_features integration in _train_submodel()

**Insertion point** (training_pipeline.py lines 967-971):
```python
# Group E: 交互作用特徴量 (HorseHistoryFeatures 後に実行 — kyakusitu_cd が必要)
from features.interaction_features import compute_interaction_features

with TimingContext(f"{surface}/interaction"):
    df = compute_interaction_features(df)
```
-- New `compute_track_condition_features(df)` should be called AFTER HorseHistoryFeatures (line 816-823) but BEFORE interaction_features (line 967-971), per CONTEXT.md D-02 and D-09.

**Sire_id mapping context** (training_pipeline.py lines 903-904):
```python
sire_map = horses_df.set_index("kettonum")["ketto3infohansyokunum1"]
df["sire_id"] = df["kettonum"].map(sire_map)
```
-- sire_id is already available in df by this point, so `sire_x_cushion_band` can use it.

---

### `src/backtest/race_predictor.py` (modified - predict insertion)

**Analog:** interaction_features integration in predict()

**Insertion point** (race_predictor.py lines 254-255):
```python
# 3. interaction_features (kyakusitu_cd が必要なため HorseHistoryFeatures 後)
df = compute_interaction_features(df)
```
-- New `compute_track_condition_features(df)` should be called BEFORE this line, after HorseHistoryFeatures merge (line 229-230).

---

### Model FEATURE_COLS modifications (surgical routing)

**Analog:** Phase 36 FEATURE_COLS registration pattern

**AbilityModel registration target** (stage1_ability_model.py lines 29-93):
-- Append new track condition feature names to `AbilityModel.FEATURE_COLS`.
-- Category-type columns must be cast in `_prepare_features()` (line 217-231).

**WinTwoStageModel registration target** (two_stage_return_model.py):
-- Three separate lists: `HIT_FEATURE_COLS` (line 426), `RETURN_FEATURE_COLS` (line 583), `FEATURE_COLS` (line 747 = `list(RETURN_FEATURE_COLS)`).
-- Append to all three per CONTEXT.md D-04.

**EVCorrectionModel registration target** (ev_correction_model.py line 154):
-- Append to `EVCorrectionModel.FEATURE_COLS` and `PlaceEVCorrectionModel.FEATURE_COLS` (line 485).
-- Missing column NaN fill pattern already exists (line 248-258):
```python
missing = [c for c in self.FEATURE_COLS if c not in df.columns]
if missing:
    for c in missing:
        df[c] = float("nan")
```

**Models to EXCLUDE per D-05/06:** MarketModel, RaceQualityScreener, RegimeDetector.

---

## Shared Patterns

### Column Existence Guard
**Source:** `src/features/interaction_features.py` (lines 44, 50, 56, 70, 76, 85, 93, 100, 109, 117, 125, 134, 164, 171, 180)
**Apply to:** All feature computations in `track_condition_features.py`
```python
if "column_a" in df.columns and "column_b" in df.columns:
    # compute feature
```

### NaN-Safe Multiplication
**Source:** `src/features/interaction_features.py` (lines 60-65, 94-97, 100-106, etc.)
**Apply to:** All numeric interaction features (dirt_moisture_x_kyakusitu, etc.)
```python
df["new_col"] = (df["a"] * df["b"]).where(
    df["a"].notna() & df["b"].notna(),
    other=float("nan"),
)
```

### Category Type for String Concatenation
**Source:** `src/features/interaction_features.py` (lines 45-47, 70-73, 76-82, 85-88)
**Apply to:** `sire_x_cushion_band` and any other categorical interactions
```python
df["new_cat"] = (
    df["col_a"].astype(str) + "_" + df["col_b"].astype(str)
).astype("category")
```

### Merge on race_id (Left Join)
**Source:** `src/features/feature_engine.py` (line 392)
**Apply to:** build_all() track_conditions merge
```python
result_df = pd.merge(result_df, tc_df, on=["race_id"], how="left")
```

### Lazy Import + TimingContext
**Source:** `src/features/feature_engine.py` (lines 387-388), `src/pipelines/training_pipeline.py` (lines 968-970)
**Apply to:** All integration points
```python
with TimingContext("build_all/track_conditions"):
    from features.track_condition_features import compute_track_condition_features
    ...
```

### DataRepository.load_track_conditions() Pattern
**Source:** `src/db/repository.py` (lines 77-91)
**Apply to:** build_all() data loading
```python
from db.repository import DataRepository

repo = DataRepository(store)
start_str = rd_valid.min().strftime("%Y%m%d")
end_str = rd_valid.max().strftime("%Y%m%d")
tc_df = repo.load_track_conditions(start_str, end_str)
```
Returns columns: `[race_id, race_date, dirt_moisture, turf_cushion]`. Empty DataFrame if parquet does not exist.

## No Analog Found

All files have close analogs. No novel patterns required.

## Metadata

**Analog search scope:** `src/features/`, `src/pipelines/`, `src/backtest/`, `src/models/`, `tests/`
**Files scanned:** 10+
**Pattern extraction date:** 2026-06-04
