# EveryDB2生カラム名一貫化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DataRepositoryを削除し、ETLで型変換を一度だけ実行、全パイプラインでEveryDB2生カラム名を使用する。

**Architecture:** ETLが型変換済みParquetを書き出し、readers.pyが薄いI/Oヘルパーを提供。FeatureEngineは派生列（distance_bin等）とMLモデル用別名（field_size, popularity_rank等）のみを計算。DataRepositoryは削除。

**重要な制約**: 全変更は原子性を持って実行する必要がある。ETL型変換後のParquetを旧Repositoryが読むとオッズ二重除算等のデータ破壊が発生するため、Repository削除とパイプライン移行は同時に行う。

**Tech Stack:** Python 3.11, pandas, pyarrow, LightGBM, pytest

**Spec:** `docs/superpowers/specs/2026-03-31-raw-column-names-design.md`

---

## Task 1: ETL型変換レイヤー追加

**Files:**
- Modify: `src/db/etl.py`

- [ ] **Step 1: `_TABLE_TYPE_RULES` 定数と `_apply_type_conversions()` を追加**

`etl.py` の `_compute_race_id()` の後に追加:

```python
_TABLE_TYPE_RULES: dict[str, dict[str, list[str]]] = {
    "races": {
        "int": ["trackcd", "kyori", "tenkocd", "syussotosu", "honsyokin"],
    },
    "entries": {
        "int": ["umaban", "kakuteijyuni", "ninki", "kyakusitukubun",
                "jyuni1c", "jyuni4c", "zogenfugo"],
        "float": ["time", "bataijyu", "zogensa", "harontimel3", "timediff"],
        "odds10": ["odds"],
    },
    "odds_tanpuku": {
        "int": ["umaban"],
        "odds10": ["tanodds", "fukuoddslow"],
    },
    "odds_wide": {
        "odds100": ["oddslow", "oddshigh"],
    },
    "jodds_tanpuku": {
        "int": ["umaban", "tanninki"],
        "odds10": ["tanodds", "fukuoddslow"],
    },
    "payouts": {
        "int": ["paytansyoumaban1"] + [f"payfukusyoumaban{i}" for i in range(1, 6)],
        "float": ["paytansyopay1"] + [f"payfukusyopay{i}" for i in range(1, 6)],
    },
}


def _apply_type_conversions(df: pd.DataFrame, table_key: str) -> pd.DataFrame:
    """Apply type conversions based on table key rules."""
    rules = _TABLE_TYPE_RULES.get(table_key)
    if rules is None:
        return df

    def _to_int(val: object) -> int | None:
        if val is None or val == "":
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _to_float(val: object) -> float | None:
        if val is None or val == "":
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    for col in rules.get("int", []):
        if col in df.columns:
            df[col] = df[col].apply(_to_int)

    for col in rules.get("float", []):
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    for col in rules.get("odds10", []):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _to_float(v) / 10 if v is not None and v != "" else None)

    for col in rules.get("odds100", []):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _to_float(v) / 100 if v is not None and v != "" else None)

    return df
```

- [ ] **Step 2: `_compute_surface()` と `_compute_track_condition_code()` を追加**

```python
def _compute_surface(df: pd.DataFrame) -> pd.DataFrame:
    """trackcd → surface (turf/dirt/other)."""
    if "trackcd" in df.columns:
        df["surface"] = df["trackcd"].apply(
            lambda x: "turf" if 10 <= x <= 22 else "dirt" if 23 <= x <= 29 else "other"
        )
    return df


def _compute_track_condition_code(df: pd.DataFrame) -> pd.DataFrame:
    """sibababacd/dirtbabacd + trackcd → track_condition_code.

    芝(trackcd 10-29)はsibababacd、ダート(23-29)はdirtbabacdを使用。
    """
    if "sibababacd" in df.columns and "dirtbabacd" in df.columns and "trackcd" in df.columns:
        import numpy as np
        is_turf = df["trackcd"].between(10, 29)
        df["track_condition_code"] = np.where(is_turf, df["sibababacd"], df["dirtbabacd"])
    return df
```

- [ ] **Step 3: `run_full_load()` に型変換を組み込み**

非パーティションテーブルの `store.write(category, key, df)` の前に:
```python
df = _apply_type_conversions(df, key)
if table_type == "raced":
    df = _compute_surface(df)
    df = _compute_track_condition_code(df)
```

パーティションテーブル（jodds_tanpuku等）の `pq.write_to_dataset()` の前に:
```python
df = _apply_type_conversions(df, key)
```

- [ ] **Step 4: `_merge_delta()` の書き込み前に型変換を追加**

`store.write(category, key, merged)` の前に:
```python
merged = _apply_type_conversions(merged, key)
if is_raced:
    merged = _compute_surface(merged)
```

- [ ] **Step 5: テストを作成して実行**

Create `tests/test_etl_type_conversion.py`:
```python
"""ETL型変換のテスト"""
import pandas as pd
import pytest
from db.etl import _apply_type_conversions, _compute_surface


class TestApplyTypeConversions:
    def test_entries_int_conversion(self):
        df = pd.DataFrame({"umaban": ["1", "2", ""], "kakuteijyuni": ["1", "0", ""]})
        result = _apply_type_conversions(df, "entries")
        assert result["umaban"].tolist() == [1, 2, None]
        assert result["kakuteijyuni"].tolist() == [1, 0, None]

    def test_entries_float_conversion(self):
        df = pd.DataFrame({"time": ["65.3", "", "N/A"], "bataijyu": ["500", "", ""]})
        result = _apply_type_conversions(df, "entries")
        assert result["time"].tolist() == [65.3, None, None]
        assert result["bataijyu"].tolist() == [500.0, None, None]

    def test_entries_odds10_conversion(self):
        df = pd.DataFrame({"odds": ["0054", "0100", ""]})
        result = _apply_type_conversions(df, "entries")
        assert result["odds"].tolist() == [5.4, 10.0, None]

    def test_odds_wide_odds100_conversion(self):
        df = pd.DataFrame({"oddslow": ["00150", "00200"], "oddshigh": ["00500", ""]})
        result = _apply_type_conversions(df, "odds_wide")
        assert result["oddslow"].tolist() == [1.50, 2.00]
        assert result["oddshigh"].tolist() == [5.00, None]

    def test_unknown_table_key(self):
        df = pd.DataFrame({"col": ["1"]})
        result = _apply_type_conversions(df, "nonexistent")
        assert result["col"].tolist() == ["1"]  # unchanged

    def test_missing_columns(self):
        df = pd.DataFrame({"other": ["1"]})
        result = _apply_type_conversions(df, "entries")
        assert "umaban" not in result.columns


class TestComputeSurface:
    def test_turf(self):
        df = pd.DataFrame({"trackcd": [10, 22, 23, 29, 51]})
        result = _compute_surface(df)
        assert result["surface"].tolist() == ["turf", "turf", "dirt", "dirt", "other"]

    def test_no_trackcd(self):
        df = pd.DataFrame({"col": [1]})
        result = _compute_surface(df)
        assert "surface" not in result.columns


class TestComputeTrackConditionCode:
    def test_turf_uses_sibababacd(self):
        df = pd.DataFrame({
            "trackcd": [10, 22, 23, 29],
            "sibababacd": ["2", "3", "1", "2"],
            "dirtbabacd": ["1", "2", "3", "4"],
        })
        result = _compute_track_condition_code(df)
        assert result["track_condition_code"].tolist() == [2, 3, 3, 4]

    def test_missing_columns(self):
        df = pd.DataFrame({"trackcd": [10]})
        result = _compute_track_condition_code(df)
        assert "track_condition_code" not in result.columns
```

Run: `python -m pytest tests/test_etl_type_conversion.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/db/etl.py tests/test_etl_type_conversion.py
git commit -m "feat: ETLに型変換レイヤー追加 (int/float/odds)"
```

---

## Task 2: readers.py 作成

**Files:**
- Create: `src/db/readers.py`

- [ ] **Step 1: readers.py を作成**

```python
"""Parquet読み取りヘルパー。型変換・リネームは一切しない。

前提: racedテーブルのParquetには race_id, race_date, surface が
ETLで事前計算されて含まれていること。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from db.parquet_store import ParquetStore


def _to_dt(yyyymmdd: str) -> datetime:
    return datetime.strptime(yyyymmdd, "%Y%m%d")


def _date_filters(start: str, end: str) -> list[tuple]:
    s, e = _to_dt(start), _to_dt(end)
    return [("race_date", ">=", s), ("race_date", "<=", e)]


def _exclude_steeple(df: pd.DataFrame) -> pd.DataFrame:
    """障害レース除外（trackcd 51-59）。trackcd列がなければそのまま返す。"""
    if "trackcd" not in df.columns:
        return df
    return df[~df["trackcd"].between(51, 59)].copy()


def load_races(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("raw", "races", filters=_date_filters(start, end))
    return _exclude_steeple(df)


def load_entries(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("raw", "entries", filters=_date_filters(start, end))
    return _exclude_steeple(df)


def load_odds_snapshots(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    return store.read("odds", "odds_tanpuku", filters=_date_filters(start, end))


def load_odds_time_series_range(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    s, e = _to_dt(start), _to_dt(end)
    filters = [
        ("year", ">=", s.year), ("year", "<=", e.year),
        ("race_date", ">=", s), ("race_date", "<=", e),
    ]
    return store.read("odds", "jodds_tanpuku", filters=filters)


def load_odds_time_series(store: ParquetStore, race_id: str) -> pd.DataFrame:
    return store.read("odds", "jodds_tanpuku", filters=[("race_id", "==", race_id)])


def load_wide_odds(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    return store.read("odds", "odds_wide", filters=_date_filters(start, end))


def load_payouts(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    return store.read("raw", "payouts", filters=_date_filters(start, end))


def load_history_entries(store: ParquetStore, lookback_years: int = 5) -> pd.DataFrame:
    cutoff = datetime.now() - timedelta(days=lookback_years * 365)
    return store.read("raw", "entries", filters=[("race_date", ">=", cutoff)])


def load_history_races(store: ParquetStore, lookback_years: int = 5) -> pd.DataFrame:
    cutoff = datetime.now() - timedelta(days=lookback_years * 365)
    return store.read("raw", "races", filters=[("race_date", ">=", cutoff)])


def load_horses(store: ParquetStore) -> pd.DataFrame:
    return store.read("raw", "horses")


def load_jockey_stats(store: ParquetStore) -> pd.DataFrame:
    return store.read("raw", "kisyu_seiseki")


def load_trainer_stats(store: ParquetStore) -> pd.DataFrame:
    return store.read("raw", "chokyo_seiseki")


def load_features(store: ParquetStore, start: str, end: str) -> pd.DataFrame | None:
    if not store.exists("features", "horse_features"):
        return None
    return store.read("features", "horse_features", filters=_date_filters(start, end))


def save_features(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("features", "horse_features", df)


def save_predictions(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("predictions", "predictions", df)


def save_bets(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("bets", "bets", df)
```

- [ ] **Step 2: Commit**

```bash
git add src/db/readers.py
git commit -m "feat: Parquet読み取りヘルパー readers.py を作成"
```

---

## Task 3: DataRepository削除 + db/__init__.py更新

**Files:**
- Delete: `src/db/repository.py`
- Delete: `src/db/schema.py`
- Modify: `src/db/__init__.py`

- [ ] **Step 1: repository.py と schema.py を削除**

```bash
git rm src/db/repository.py src/db/schema.py
```

- [ ] **Step 2: db/__init__.py の DataRepository参照を削除**

DataRepositoryのimportと__all__エクスポートを削除。

- [ ] **Step 3: Commit**

```bash
git add src/db/__init__.py
git commit -m "refactor: DataRepositoryとschema.pyを削除"
```

---

## Task 4: FeatureEngine簡素化

**Files:**
- Modify: `src/features/feature_engine.py`

- [ ] **Step 1: `_map_basic_features()` を書き換え**

以下の処理のみにする:
1. `distance_bin`: `kyori` + `surface` から計算
2. `track_condition_code`: ETLで既にParquetに保存済み。FeatureEngineでは `if "track_condition_code" not in df.columns` ガードのみ残す（推論パス用フォールバック）
3. `grade_code`: `gradecd` → `grade_code` にコピー
4. `field_size`: `syussotosu` → `field_size` にコピー
5. `popularity_rank`: `ninki` → `popularity_rank` にコピー
6. `running_style`: `kyakusitukubun` → `running_style` にコピー (int変換)

削除する処理:
- `distance_band` → `distance_bin` リネーム
- `baba_cd` → `track_condition_code` リネーム
- `grade_cd` → `grade_code` リネーム
- `surface` 計算（ETLで済）
- `track_condition_code` 計算（ETLで済 — `_compute_track_condition_code()` でParquetに保存）
- `surface_key` コピー
- `win_odds` → `win_odds_actual` コピー
- `fuku_odds` → `place_odds_actual` コピー

- [ ] **Step 2: `build_all()` を更新**

- 引数 `repo: object | None = None` → `store: ParquetStore | None = None`
- `df["track_cd"] < 51` → `df["trackcd"] < 51`
- `BloodlineFeatures(repo)` → `BloodlineFeatures(store)` (import変更)

- [ ] **Step 3: `build_features()` を更新**

race_data/entry_data dictを生カラム名に変更:
```python
race_data = {
    "race_id": race.race_id, "trackcd": race.track_cd, "kyori": race.distance,
    "gradecd": race.grade_cd, "syussotosu": race.field_size, "tenkocd": race.tenko_cd,
    "syubetucd": race.syubetu_cd, "jyokencd1": race.jyoken_cd,
    "track_condition_code": race.baba_cd,
}
entry_data = {
    "race_id": race.race_id, "umaban": e.umaban, "kettonum": e.ketto_num,
    "kakuteijyuni": e.finish_pos, "odds": e.win_odds_actual, "ninki": e.popularity_rank,
    "bataijyu": e.ba_taijyu, "kisyucode": e.kisyu_code, "chokyosicode": e.chokyosi_code,
}
```

- [ ] **Step 4: Commit**

```bash
git add src/features/feature_engine.py
git commit -m "refactor: FeatureEngineを生カラム名対応に簡素化"
```

---

## Task 5: サブ特徴量モジュール更新

**Files:**
- Modify: `src/features/horse_history_features.py`
- Modify: `src/features/bloodline_features.py`
- Modify: `src/features/jockey_context_features.py`
- Modify: `src/features/trainer_context_features.py`
- Modify: `src/features/intra_race_features.py`
- Modify: `src/features/odds_dynamics_features.py`
- Modify: `src/features/market_bias_features.py`
- Modify: `src/features/interaction_features.py`
- Modify: `src/features/race_difficulty_model.py`

各ファイルで以下を実行:

- [ ] **Step 1: 共通変更 — DataRepository import → ParquetStore**

```python
# 変更前 (TYPE_CHECKING)
from db.repository import DataRepository
# 変更後
from db.parquet_store import ParquetStore
```

`__init__(self, repo: DataRepository)` → `__init__(self, store: ParquetStore)`
`self.repo` → `self.store`

- [ ] **Step 2: HorseHistoryFeatures**

全カラム参照を生名に変更:
- `ketto_num` → `kettonum`
- `kisyu_code` → `kisyucode`
- `finish_pos` → `kakuteijyuni`
- `win_odds` → `odds`
- `track_cd` → `trackcd`
- `distance` → `kyori`
- `haron_time_l3` → `harontimel3`
- `time_diff` → `timediff`
- `corner_1c` → `jyuni1c`
- `corner_4c` → `jyuni4c`
- `kyakusitu` → `kyakusitukubun`
- `ba_taijyu` → `bataijyu`
- `field_size` は変更なし（FeatureEngine別名）

distance_bin計算ブロック（lines 217-227）を削除。代わりに `past_df` に `FeatureEngine._map_basic_features(past_df)` を適用。

`self.repo.load_history_entries()` → `from db.readers import load_history_entries; load_history_entries(self.store)`
`self.repo.load_history_races()` → 同様

- [ ] **Step 3: BloodlineFeatures**

merge を `on="kettonum"` に統一:
```python
merged = entry_df[["race_id", "umaban", "kettonum"]].merge(
    horses_df, on="kettonum", how="left"
)
```

- [ ] **Step 4: JockeyContextFeatures**

merge を `on="kisyucode"` に統一。groupby も `kisyucode` に変更。

- [ ] **Step 5: TrainerContextFeatures**

merge を `on="chokyosicode"` に統一。groupby も `chokyosicode` に変更。

- [ ] **Step 6: intra_race_features.py**

`win_odds` → `odds`。`ba_taijyu` は既に生名なので変更なし。

- [ ] **Step 7: odds_dynamics_features.py**

`happyo_time` → `happyotime`, `tan_odds` → `tanodds`, `fuku_odds` → `fukuoddslow`, `finish_pos` → `kakuteijyuni`。

- [ ] **Step 8: market_bias_features.py**

`tan_odds` → `tanodds`, `finish_pos` → `kakuteijyuni`。

- [ ] **Step 9: interaction_features.py**

`distance` → `kyori`, `kyakusitu` → `kyakusitukubun`。`ba_taijyu` は既に生名。

- [ ] **Step 10: race_difficulty_model.py**

`grade_cd` fallback → `gradecd`:
```python
grade_col = "gradecd" if "gradecd" in df.columns else "grade_code"
```

- [ ] **Step 11: Commit**

```bash
git add src/features/
git commit -m "refactor: 全サブ特徴量モジュールを生カラム名に対応"
```

---

## Task 6: MLモデルlabel列更新

**Files:**
- Modify: `src/models/stage1_ability_model.py`
- Modify: `src/models/place_ability_model.py`
- Modify: `src/models/two_stage_return_model.py`
- Modify: `src/models/ev_correction_model.py`
- Modify: `src/models/wide_pair_builder.py`
- Modify: `src/models/submodel_manager.py`

- [ ] **Step 1: label列の変更**

全モデルで `finish_pos` → `kakuteijyuni`:
- `stage1_ability_model.py`: `.train()` / `train_oof()` 内
- `place_ability_model.py`: `.train()` 内
- `two_stage_return_model.py`: `.train_hit_model()`, `.train_return_model()`, `.predict_ev()` 内
- `ev_correction_model.py`: `.train()`, `.correct_ev()` 内
- `wide_pair_builder.py`: `.build()` 内

- [ ] **Step 2: キャリブレーション列の変更**

- `two_stage_return_model.py`: `win_odds_actual` → `odds`
- `two_stage_return_model.py` (PlaceTwoStageModel): `place_odds_actual` → `fukuoddslow`
- `ev_correction_model.py`: `win_odds_actual` → `odds`

- [ ] **Step 3: SubModelManager更新**

`add_distance_band_features()`: `df["distance"]` → `df["kyori"]` (7箇所)

- [ ] **Step 4: Commit**

```bash
git add src/models/
git commit -m "refactor: MLモデルのlabel列を生カラム名に変更"
```

---

## Task 7: TrainingPipeline更新

**Files:**
- Modify: `src/pipelines/training_pipeline.py`

- [ ] **Step 1: import とコンストラクタ変更**

```python
# 変更前
from db.repository import DataRepository
# 変更後
from db.readers import (
    load_races, load_entries, load_odds_snapshots,
    load_wide_odds, load_history_entries, load_history_races,
    load_horses, load_jockey_stats, load_trainer_stats,
    save_predictions, save_bets,
)
```

`repo: DataRepository | None = None` → `store: ParquetStore | None = None`
`self.repo = repo or DataRepository(ParquetStore())` → `self.store = store or ParquetStore()`

- [ ] **Step 2: データロード変更**

```python
race_df = load_races(self.store, start, end)
entry_df = load_entries(self.store, start, end)
odds_df = load_odds_snapshots(self.store, start, end)
wide_odds_df = load_wide_odds(self.store, start, end)
```

- [ ] **Step 3: wide_odds pivot変更**

`values="odds_low"` → `values="oddslow"`

- [ ] **Step 4: キャリブレーション変更**

`win_odds_actual` → `odds`, `place_odds_actual` → `fukuoddslow`, `finish_pos` → `kakuteijyuni`

- [ ] **Step 5: Sub-model引数変更**

`HorseHistoryFeatures(repo=self.repo)` → `HorseHistoryFeatures(store=self.store)`
`JockeyContextFeatures(self.repo)` → `JockeyContextFeatures(self.store)`
`TrainerContextFeatures(self.repo)` → `TrainerContextFeatures(self.store)`
`BloodlineFeatures(repo)` → `BloodlineFeatures(store)` (build_all内)

- [ ] **Step 6: Commit**

```bash
git add src/pipelines/training_pipeline.py
git commit -m "refactor: TrainingPipelineをreaders.py + store に移行"
```

---

## Task 8: BacktestEngine + RacePredictor更新

**Files:**
- Modify: `src/backtest/engine.py`
- Modify: `src/backtest/race_predictor.py`
- Modify: `src/backtest/validation_suite.py`
- Modify: `src/backtest/report.py`

- [ ] **Step 1: engine.py — import とコンストラクタ変更**

`DataRepository` → `ParquetStore` + readers imports
`self.repo` → `self.store`, 全 `self.repo.load_xxx()` → `load_xxx(self.store, ...)`

- [ ] **Step 2: engine.py — カラム名変更**

- `surface_key` → `surface`
- `distance` → `kyori`
- `place_odds_actual` → `fukuoddslow`
- `finish_pos` → `kakuteijyuni`
- bet_history dict key `"distance"` → `"kyori"`

- [ ] **Step 3: race_predictor.py — カラム名変更**

- `surface_key` → `surface`
- `place_odds_actual` → `fukuoddslow`

- [ ] **Step 4: validation_suite.py — repo→store + カラム名変更**

- [ ] **Step 5: report.py — bet_history dict key変更**

`b["distance"]` → `b["kyori"]` (bet_history dict)

- [ ] **Step 6: Commit**

```bash
git add src/backtest/
git commit -m "refactor: BacktestEngineを生カラム名に対応"
```

---

## Task 9: Ingestion + Paper Trading更新

**Files:**
- Modify: `src/ingestion/jvlink_fetcher.py`
- Modify: `src/ingestion/odds_collector.py`
- Modify: `src/paper_trading/predictor.py`
- Modify: `src/paper_trading/reconciler.py`

- [ ] **Step 1: JVLinkFetcher — コンストラクタ変更**

`repo: DataRepository` → `store: ParquetStore`
全 `self.repo.load_xxx()` → `load_xxx(self.store, ...)`

- [ ] **Step 2: JVLinkFetcher — `_row_to_race()` カラム名変更**

Spec Section 7マッピングに従い変更。ただし:
- `baba_cd` → ETLで `track_condition_code` として保存済み → `row["track_condition_code"]` を読む
- `field_size` → `row["syussotosu"]` を読み、ドメインモデルには `field_size` として渡す

- [ ] **Step 3: JVLinkFetcher — `_row_to_entry()` カラム名変更**

**重要**: Spec Section 7のマッピングと以下の点で異なる:
- `popularity_rank` はFeatureEngine別名。生Parquetには `ninki` → `row["ninki"]` を読む
- `running_style` はFeatureEngine別名。生Parquetには `kyakusitukubun` → `row["kyakusitukubun"]` を読む
- `baba_cd` / `track_condition_code` はETLで `track_condition_code` としてParquetに保存済み → `row["track_condition_code"]` を読む

変更:
- `ketto_num` → `kettonum`
- `finish_pos` → `kakuteijyuni`
- `win_odds_actual` → `odds`
- `ba_taijyu` → `bataijyu`
- `kisyu_code` → `kisyucode`
- `chokyosi_code` → `chokyosicode`
- `popularity_rank` → `ninki` (生Parquetのカラム名)
- `running_style` → `kyakusitukubun` (生Parquetのカラム名)

- [ ] **Step 4: JVLinkFetcher — オッズ時系列参照変更**

`load_odds_time_series(race_id)` は現在DataRepositoryでもtransformを適用せず生カラム名を返す。
よって `happyo_time` → `happyotime`, `tan_odds` → `tanodds`, `fuku_odds` → `fukuoddslow` の変更は
既存の動作に合わせるもの。念のためコード内の参照を生名に統一。

- [ ] **Step 5: OddsCollector — repo→store**

`repo.save_predictions(df)` → `save_predictions(self.store, df)`

- [ ] **Step 6: PaperTrading predictor.py — repo→store + カラム名変更**

- `surface_key` → `surface`
- `distance` → `kyori`
- `place_odds_actual` → `fukuoddslow`
- `fuku_odds` → `fukuoddslow`
- `tan_odds` → `tanodds`
- `win_odds` → `odds`
- `ba_taijyu` → `bataijyu`

- [ ] **Step 7: PaperTrading reconciler.py — repo→store**

- [ ] **Step 8: Commit**

```bash
git add src/ingestion/ src/paper_trading/
git commit -m "refactor: Ingestion + PaperTradingを生カラム名に対応"
```

---

## Task 10: テスト更新 (1) — Feature/Model系

**Files:**
- Delete: `tests/test_repository.py`
- Create: `tests/test_readers.py`
- Modify: `tests/test_feature_engine.py`
- Modify: `tests/test_intra_race_features.py`
- Modify: `tests/test_horse_history_features.py`
- Modify: `tests/test_bloodline_features.py`
- Modify: `tests/test_jockey_context_features.py`
- Modify: `tests/test_trainer_context_features.py`
- Modify: `tests/test_interaction_features.py`
- Modify: `tests/test_market_bias_features.py`
- Modify: `tests/test_odds_dynamics_features.py`
- Modify: `tests/test_ev_correction.py`
- Modify: `tests/test_wide_pair_builder.py`

- [ ] **Step 1: test_repository.py を削除、test_readers.py を作成**

```bash
git rm tests/test_repository.py
```

`test_readers.py`: mock ParquetStoreで各load関数が正しいcategory/name/filtersを渡すことを確認。

- [ ] **Step 2: test_feature_engine.py 更新**

全モックデータのカラム名を生名に変更。`_map_basic_features()` テスト:
- 入力に `kyori`, `surface`, `syussotosu`, `ninki`, `gradecd`, `sibababacd`, `dirtbabacd`, `trackcd`, `kyakusitukubun` が含まれることを確認
- 出力に `distance_bin`, `track_condition_code`, `grade_code`, `field_size`, `popularity_rank`, `running_style` が含まれることを確認

- [ ] **Step 3: 各テストファイルのカラム名を生名に変更**

各ファイルでgrepして旧カラム名を一括置換:
- `ketto_num` → `kettonum`
- `kisyu_code` → `kisyucode`
- `chokyosi_code` → `chokyosicode`
- `finish_pos` → `kakuteijyuni`
- `win_odds` → `odds` (注意: `win_odds_actual` は `odds` に)
- `track_cd` → `trackcd`
- `distance` → `kyori`
- `haron_time_l3` → `harontimel3`
- `time_diff` → `timediff`
- `corner_1c` → `jyuni1c`
- `corner_4c` → `jyuni4c`
- `kyakusitu` → `kyakusitukubun`
- `tan_odds` → `tanodds`
- `fuku_odds` → `fukuoddslow`
- `happyo_time` → `happyotime`
- `surface_key` → `surface`
- `popularity_rank` はテストモックデータでは `ninki` として入力、FeatureEngineが `popularity_rank` にコピーする
- `running_style` はテストモックデータでは `kyakusitukubun` として入力

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: Feature/Model系テストを生カラム名に更新"
```

---

## Task 11: テスト更新 (2) — Pipeline/Backtest/Ingestion系

**Files:**
- Modify: `tests/test_training_pipeline.py`
- Modify: `tests/test_backtest_engine.py`
- Modify: `tests/test_validation_suite.py`
- Modify: `tests/test_race_predictor.py`
- Modify: `tests/test_jvlink_fetcher.py`
- Modify: `tests/test_odds_collector.py`
- Modify: `tests/test_history_features_v2.py`
- Modify: `tests/test_backtest_report.py`
- Modify: `tests/test_paper_predictor.py`
- Modify: `tests/test_paper_reconciler.py`
- Modify: `tests/test_dry_run.py`
- Modify: `tests/test_db.py`

- [ ] **Step 1: 全テストで旧カラム名をgrepして置換**

各ファイルでTask 10と同様の置換を実施。

`test_training_pipeline.py`:
- `mock_repo` → `mock_store`
- `@patch("db.repository.DataRepository")` → 削除
- 全カラム名を生名に変更

`test_backtest_engine.py`:
- 同上

`test_jvlink_fetcher.py`:
- JVLinkFetcher spec Section 7マッピングに従い変更
- `MagicMock(spec=DataRepository)` → `MagicMock()`

- [ ] **Step 2: test_db.py の schema.py 関連テストを削除/更新**

schema.pyが削除されたため、schema検証テストを削除。

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test: Pipeline/Backtest/Ingestion系テストを生カラム名に更新"
```

---

## Task 12: 最終検証

- [ ] **Step 1: ruff check**

```bash
ruff check src/ tests/
```

- [ ] **Step 2: ruff format**

```bash
ruff format --check src/ tests/
```

- [ ] **Step 3: mypy**

```bash
mypy src/
```

- [ ] **Step 4: 全テスト実行**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: ALL PASS

- [ ] **Step 5: 旧カラム名の残存チェック**

```bash
grep -r "finish_pos\|win_odds_actual\|place_odds_actual\|surface_key\|ba_taijyu\|\"distance\"\|ketto_num\|kisyu_code\|chokyosi_code\|track_cd\|corner_1c\|corner_4c\|time_diff\|haron_time_l3\|tan_odds\|fuku_odds\|happyo_time\|odds_low\|odds_high\|month_day\|jyo_cd\|race_num\|tenko_cd\|syubetu_cd\|jyoken_cd[^1]\|grade_cd[^_]\|zogen_fugo\|zogen_sa\|DataRepository" src/ tests/ --include="*.py" | grep -v "__pycache__" | grep -v "\.pyc" | grep -v "domain/models.py" | grep -v "# " | grep -v "TODO" | grep -v "旧名\|変更前\|was "
```

Expected: No hits in src/ (except domain/models.py Python attributes)

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: リント・フォーマット修正"
```
