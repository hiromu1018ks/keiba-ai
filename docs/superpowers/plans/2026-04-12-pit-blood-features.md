# Point-in-Time 血統特徴量: ルックアヘッドバイアス修正

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 血統特徴量 (`blood_total_wr`, `blood_surface_wr`, `blood_distance_wr`, `blood_prize_log`) を ETL 時点の累積値から、レース開催時点での累積値に変更し、BT のルックアヘッドバイアスを排除する。

**Architecture:** `entries.parquet` (出走履歴) + `races.parquet` (レース条件) から各 (kettonum, race_id) ごとの事前累積成績を事前計算し、Parquet ファイルとして保存。`BloodlineFeatures` は horses.parquet の代わりにこの事前計算ファイルを読み込む。

**Tech Stack:** pandas, numpy, pyarrow

---

## 背景

### ルート原因

BT テスト (2025年) で `horses.parquet` (2026-04-10 ETLスナップショット) を使用すると、
2025年のレースを予測する際に 2025年〜2026年のレース結果が累積成績に含まれる。

**定量的証拠:**

| 指標 | 値 |
|------|-----|
| bwr_pit (正当) AUC | 0.555 |
| bwr_etl (ルックアヘッド) AUC | 0.684 |
| ルックアヘッド利得 | **+0.129 AUC** |
| bwr_gap (純粋な未来情報) の place 相関 | r=0.380 |
| デビュー馬での bwr_etl と place 相関 | r=0.478 |
| BT テスト馬の 86% が 20戦未満 | 設計書想定 (100戦) の不成立 |

### 修正対象の特徴量

| 特徴量 | 現在のデータソース | 修正後 |
|--------|-------------------|--------|
| `blood_total_wr` | horses.parquet `chuochakukaisu*` | entries.parquet 累積着回数 |
| `blood_prize_log` | horses.parquet `ruikeihonsyoheiti` | entries.parquet 累積賞金 |
| `blood_surface_wr` | horses.parquet `ba1chakukaisu*` (芝直線) | entries+races 芝全般 (近似) |
| `blood_distance_wr` | horses.parquet `kyori1chakukaisu*` (芝1600以下) | entries+races 芝1600以下 (近似) |

**注意:** `ba1` は「芝直線コース」限定だが、races.parquet にコース形状列がないため、
「芝全般」で近似する。学習時も同じ近似値を使うため、BT/PT 間の整合性は保たれる。

---

## File Structure

| 操作 | ファイル | 説明 |
|------|---------|------|
| Create | `src/features/horse_career_stats.py` | 事前計算モジュール |
| Create | `scripts/precompute_career_stats.py` | ETL 後の事前計算スクリプト |
| Create | `tests/test_horse_career_stats.py` | 事前計算のユニットテスト |
| Create | `tests/test_bloodline_features_pit.py` | PIT 版 BloodlineFeatures のテスト |
| Modify | `src/features/bloodline_features.py` | horses.parquet → career_stats 利用 |
| Modify | `src/db/readers.py` | `load_career_stats()` 追加 |
| Generate | `data/raw/horse_career_stats.parquet` | 事前計算済みキャリア統計 |

---

### Task 1: キャリア統事事前計算モジュール

**Files:**
- Create: `src/features/horse_career_stats.py`
- Test: `tests/test_horse_career_stats.py`

- [ ] **Step 1: テストを書く**

```python
# tests/test_horse_career_stats.py
"""horse_career_stats: point-in-time 累積成績のテスト"""
import pandas as pd
import numpy as np
import pytest

from features.horse_career_stats import precompute_career_stats


@pytest.fixture
def sample_data():
    """3頭の馬 × 数レースのテストデータ"""
    entries = pd.DataFrame({
        "race_id": ["20250101A01", "20250115A01", "20250201A01", "20250101A02", "20250101A03"],
        "kettonum": ["H001", "H001", "H001", "H002", "H003"],
        "kakuteijyuni": pd.array([1, 3, 2, 5, pd.NA], dtype="Int64"),
        "honsyokin": pd.array([50000, 10000, 20000, 0, pd.NA], dtype="Int64"),
        "race_date": pd.to_datetime(["2025-01-01", "2025-01-15", "2025-02-01",
                                      "2025-01-01", "2025-01-01"]),
        "jyocd": pd.array([5, 5, 5, 5, 5], dtype="Int64"),
    })
    races = pd.DataFrame({
        "race_id": ["20250101A01", "20250115A01", "20250201A01", "20250101A02", "20250101A03"],
        "trackcd": pd.array([17, 17, 24, 17, 24], dtype="Int64"),
        "kyori": pd.array([1600, 1200, 1800, 1400, 1200], dtype="Int64"),
    })
    return entries, races


def test_total_stats_cumulative(sample_data):
    """累積勝利数・出走数が正しいことを確認"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    h001 = result[result["kettonum"] == "H001"].sort_values("race_date")

    # 1レース目: デビュー → cum_starts=0, cum_wins=0
    first = h001.iloc[0]
    assert first["cum_starts"] == 0
    assert first["cum_wins"] == 0

    # 2レース目: 1戦1勝 → cum_starts=1, cum_wins=1
    second = h001.iloc[1]
    assert second["cum_starts"] == 1
    assert second["cum_wins"] == 1

    # 3レース目: 2戦1勝(1着1回,3着1回) → cum_starts=2, cum_wins=1
    third = h001.iloc[2]
    assert third["cum_starts"] == 2
    assert third["cum_wins"] == 1


def test_debut_horse_zero_starts(sample_data):
    """デビュー馬は cum_starts=0 であること"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    h003 = result[result["kettonum"] == "H003"]
    assert len(h003) == 1
    assert h003.iloc[0]["cum_starts"] == 0
    assert h003.iloc[0]["cum_wins"] == 0


def test_prize_cumulative(sample_data):
    """累積賞金が正しいことを確認"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    h001 = result[result["kettonum"] == "H001"].sort_values("race_date")

    # 1レース目: デビュー → cum_prize=0
    assert h001.iloc[0]["cum_prize"] == 0

    # 2レース目: 前走で50000円獲得 → cum_prize=50000
    assert h001.iloc[1]["cum_prize"] == 50000

    # 3レース目: 前走までに60000円獲得 → cum_prize=60000
    assert h001.iloc[2]["cum_prize"] == 60000


def test_surface_specific_stats(sample_data):
    """芝/ダート別の累積成績が正しいこと"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    h001 = result[result["kettonum"] == "H001"].sort_values("race_date")

    # 1レース目: 芝(trackcd=17) → cum_turf_starts=0, cum_dirt_starts=0
    first = h001.iloc[0]
    assert first["cum_turf_starts"] == 0

    # 3レース目: ダート(trackcd=24), 前に芝2戦 → cum_turf_starts=2, cum_dirt_starts=0
    third = h001.iloc[2]
    assert third["cum_turf_starts"] == 2
    assert third["cum_dirt_starts"] == 0


def test_distance_specific_stats(sample_data):
    """芝1600以下の累積成績が正しいこと (kyori1 近似)"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    h001 = result[result["kettonum"] == "H001"].sort_values("race_date")

    # 1レース目: 芝1600m → 条件該当, でもデビューなので cum_short_starts=0
    first = h001.iloc[0]
    assert first["cum_short_starts"] == 0

    # 2レース目: 芝1200m, 前走は芝1600m(該当) → cum_short_starts=1, cum_short_wins=1
    second = h001.iloc[1]
    assert second["cum_short_starts"] == 1
    assert second["cum_short_wins"] == 1

    # 3レース目: ダート1800m, 前走は芝1200m(該当) → cum_short_starts=2, cum_short_wins=1
    third = h001.iloc[2]
    assert third["cum_short_starts"] == 2
    assert third["cum_short_wins"] == 1


def test_output_columns(sample_data):
    """出力に必要なカラムが全て含まれること"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    expected_cols = [
        "race_id", "kettonum", "race_date",
        "cum_starts", "cum_wins", "cum_prize",
        "cum_turf_starts", "cum_turf_wins",
        "cum_dirt_starts", "cum_dirt_wins",
        "cum_short_starts", "cum_short_wins",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_horse_career_stats.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'features.horse_career_stats'`)

- [ ] **Step 3: 実装を書く**

```python
# src/features/horse_career_stats.py
"""horse_career_stats.py — Point-in-Time 累積キャリア統計の事前計算

entries.parquet (出走履歴) + races.parquet (レース条件) から、
各 (kettonum, race_id) ごとの「レース開催前時点」での累積成績を計算する。

これにより、horses.parquet (x_UMA) の ETL 時点累積値に含まれる
ルックアヘッドバイアス (未来のレース結果が特徴量に混入) を排除する。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# JRA 競走条件
_TURF_TRACKCD_RANGE = (10, 22)  # 芝 (trackcd 10-22)
_DIRT_TRACKCD_RANGE = (23, 29)  # ダート (trackcd 23-29)
_SHORT_DISTANCE_MAX = 1600      # 芝1600M以下 (x_UMA kyori1 の定義)


def _classify_surface(trackcd: pd.Series) -> pd.Series:
    """trackcd から surface を分類。"""
    trackcd_num = pd.to_numeric(trackcd, errors="coerce")
    return np.where(
        trackcd_num.between(*_TURF_TRACKCD_RANGE), "turf",
        np.where(trackcd_num.between(*_DIRT_TRACKCD_RANGE), "dirt", "other"),
    )


def _compute_cumulative_before(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> pd.Series:
    """グループごとに「現在行より前」の累積和を計算。

    shift(1) で現在行を除外してから cumsum を取る。
    """
    return df.groupby(group_col)[value_col].transform(
        lambda x: x.shift(1).fillna(0).cumsum()
    )


def precompute_career_stats(
    entries_df: pd.DataFrame,
    races_df: pd.DataFrame,
) -> pd.DataFrame:
    """Point-in-Time 累積キャリア統計を計算。

    Args:
        entries_df: 出走履歴 (race_id, kettonum, kakuteijyuni, honsyokin, race_date, jyocd)
        races_df: レース情報 (race_id, trackcd, kyori)

    Returns:
        DataFrame with columns:
            race_id, kettonum, race_date,
            cum_starts, cum_wins, cum_prize,
            cum_turf_starts, cum_turf_wins,
            cum_dirt_starts, cum_dirt_wins,
            cum_short_starts, cum_short_wins
    """
    # JRA レースのみ (jyocd 1-10)
    jyocd_num = pd.to_numeric(entries_df["jyocd"], errors="coerce")
    jra_mask = jyocd_num.between(1, 10)
    ent = entries_df[jra_mask].copy()

    if ent.empty:
        return pd.DataFrame()

    # レース条件をマージ
    race_info = races_df[["race_id", "trackcd", "kyori"]].copy()
    ent = ent.merge(race_info, on="race_id", how="left")

    # surface / short distance 判定
    ent["surface"] = _classify_surface(ent["trackcd"])
    ent["is_turf"] = (ent["surface"] == "turf").astype(int)
    ent["is_dirt"] = (ent["surface"] == "dirt").astype(int)
    ent["is_short"] = (
        (ent["surface"] == "turf")
        & (pd.to_numeric(ent["kyori"], errors="coerce") <= _SHORT_DISTANCE_MAX)
    ).astype(int)

    # 着順・賞金の数値化
    ent["kakuteijyuni_int"] = pd.to_numeric(ent["kakuteijyuni"], errors="coerce")
    ent["is_win"] = (ent["kakuteijyuni_int"] == 1).astype(int)
    ent["honsyokin_num"] = pd.to_numeric(ent["honsyokin"], errors="coerce").fillna(0)
    ent["is_turf_win"] = (ent["is_turf"] & ent["is_win"]).astype(int)
    ent["is_dirt_win"] = (ent["is_dirt"] & ent["is_win"]).astype(int)
    ent["is_short_win"] = (ent["is_short"] & ent["is_win"]).astype(int)

    # 馬ごとに日付順でソート
    ent = ent.sort_values(["kettonum", "race_date", "race_id"]).reset_index(drop=True)

    # 累積和 (現在行を除外 = shift(1) → cumsum)
    ent["one"] = 1
    ent["cum_starts"] = _compute_cumulative_before(ent, "kettonum", "one")
    ent["cum_wins"] = _compute_cumulative_before(ent, "kettonum", "is_win")
    ent["cum_prize"] = _compute_cumulative_before(ent, "kettonum", "honsyokin_num")
    ent["cum_turf_starts"] = _compute_cumulative_before(ent, "kettonum", "is_turf")
    ent["cum_turf_wins"] = _compute_cumulative_before(ent, "kettonum", "is_turf_win")
    ent["cum_dirt_starts"] = _compute_cumulative_before(ent, "kettonum", "is_dirt")
    ent["cum_dirt_wins"] = _compute_cumulative_before(ent, "kettonum", "is_dirt_win")
    ent["cum_short_starts"] = _compute_cumulative_before(ent, "kettonum", "is_short")
    ent["cum_short_wins"] = _compute_cumulative_before(ent, "kettonum", "is_short_win")

    result = ent[[
        "race_id", "kettonum", "race_date",
        "cum_starts", "cum_wins", "cum_prize",
        "cum_turf_starts", "cum_turf_wins",
        "cum_dirt_starts", "cum_dirt_wins",
        "cum_short_starts", "cum_short_wins",
    ]].copy()

    logger.info(
        "Career stats: %d entries, %d horses",
        len(result),
        result["kettonum"].nunique(),
    )
    return result
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_horse_career_stats.py -v`
Expected: all PASS

- [ ] **Step 5: コミット**

```bash
git add src/features/horse_career_stats.py tests/test_horse_career_stats.py
git commit -m "feat: add point-in-time career stats precomputation module"
```

---

### Task 2: キャリア統事事前計算スクリプト

**Files:**
- Create: `scripts/precompute_career_stats.py`
- Modify: `src/db/readers.py` — `load_career_stats()` 追加

- [ ] **Step 1: テストを書く (`load_career_stats`)**

`tests/test_readers.py` に追加:

```python
def test_load_career_stats_returns_dataframe(tmp_path, monkeypatch):
    """load_career_stats が DataFrame を返すこと"""
    from db.readers import load_career_stats
    from db.parquet_store import ParquetStore

    # career_stats.parquet が存在しない場合は空 DataFrame
    store = ParquetStore(str(tmp_path))
    result = load_career_stats(store)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_readers.py::test_load_career_stats_returns_dataframe -v`
Expected: FAIL (`ImportError: cannot import name 'load_career_stats'`)

- [ ] **Step 3: `load_career_stats` を readers.py に追加**

`src/db/readers.py` の末尾に追加:

```python
def load_career_stats(store: ParquetStore) -> pd.DataFrame:
    """Point-in-time キャリア統計を読み込む。"""
    if not store.exists("raw", "horse_career_stats"):
        return pd.DataFrame()
    df = store.read("raw", "horse_career_stats")
    return _coerce_types(df)
```

- [ ] **Step 4: 事前計算スクリプトを作成**

```python
# scripts/precompute_career_stats.py
"""Point-in-Time キャリア統計の事前計算

ETL 実行後に実行する:
  python scripts/run_etl.py --mode full --start 20150101 --end 20260412
  python scripts/precompute_career_stats.py
"""
from __future__ import annotations

import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def main() -> None:
    import pandas as pd
    from db.parquet_store import ParquetStore
    from features.horse_career_stats import precompute_career_stats

    store = ParquetStore()

    logger.info("Loading entries.parquet...")
    t0 = time.time()
    entries_df = store.read("raw", "entries")
    logger.info("  %d rows (%.1fs)", len(entries_df), time.time() - t0)

    logger.info("Loading races.parquet...")
    races_df = store.read("raw", "races")

    logger.info("Computing career stats...")
    t0 = time.time()
    stats = precompute_career_stats(entries_df, races_df)
    logger.info("  %d rows (%.1fs)", len(stats), time.time() - t0)

    logger.info("Saving to data/raw/horse_career_stats.parquet...")
    store.write("raw", "horse_career_stats", stats)

    # 検証
    debut_rate = (stats["cum_starts"] == 0).mean()
    logger.info("Debut rate: %.1f%% (%d / %d)", debut_rate * 100,
                (stats["cum_starts"] == 0).sum(), len(stats))
    logger.info("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: テストが通ることを確認**

Run: `python -m pytest tests/test_readers.py::test_load_career_stats_returns_dataframe -v`
Expected: PASS

- [ ] **Step 6: 事前計算を実行**

Run: `python scripts/precompute_career_stats.py`
Expected: `data/raw/horse_career_stats.parquet` が生成される

- [ ] **Step 7: コミット**

```bash
git add scripts/precompute_career_stats.py src/db/readers.py tests/test_readers.py data/raw/horse_career_stats.parquet
git commit -m "feat: add career stats precomputation script and load_career_stats reader"
```

---

### Task 3: BloodlineFeatures を PIT 版に変更

**Files:**
- Modify: `src/features/bloodline_features.py`
- Create: `tests/test_bloodline_features_pit.py`

- [ ] **Step 1: テストを書く**

```python
# tests/test_bloodline_features_pit.py
"""BloodlineFeatures PIT (point-in-time) 版のテスト"""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

from features.bloodline_features import BloodlineFeatures, ALPHA_PRIOR, TOTAL_OFFSET


def _make_store_mock(career_stats_df: pd.DataFrame | None = None) -> MagicMock:
    """ParquetStore モックを作成。"""
    store = MagicMock()

    def read_side_effect(group, name, **kwargs):
        if name == "horse_career_stats" and career_stats_df is not None:
            return career_stats_df
        return pd.DataFrame()

    store.read = MagicMock(side_effect=read_side_effect)
    store.exists = MagicMock(return_value=False)
    return store


def test_pit_debut_horse_gets_nan_blood_total_wr():
    """デビュー馬は blood_total_wr = NaN"""
    career = pd.DataFrame({
        "race_id": ["20250101A01"],
        "kettonum": ["H001"],
        "race_date": pd.to_datetime(["2025-01-01"]),
        "cum_starts": [0],
        "cum_wins": [0],
        "cum_prize": [0.0],
        "cum_turf_starts": [0], "cum_turf_wins": [0],
        "cum_dirt_starts": [0], "cum_dirt_wins": [0],
        "cum_short_starts": [0], "cum_short_wins": [0],
    })
    entry_df = pd.DataFrame({
        "race_id": ["20250101A01"],
        "umaban": [1],
        "kettonum": ["H001"],
    })

    store = _make_store_mock(career)
    bf = BloodlineFeatures(store)
    result = bf.compute(entry_df)

    assert result.iloc[0]["blood_total_wr"] is np.nan or pd.isna(result.iloc[0]["blood_total_wr"])


def test_pit_experienced_horse_gets_correct_wr():
    """既出走馬は正しい point-in-time 勝率を得る"""
    career = pd.DataFrame({
        "race_id": ["20250115A01"],
        "kettonum": ["H001"],
        "race_date": pd.to_datetime(["2025-01-15"]),
        "cum_starts": [5],
        "cum_wins": [2],
        "cum_prize": [100000.0],
        "cum_turf_starts": [3], "cum_turf_wins": [1],
        "cum_dirt_starts": [2], "cum_dirt_wins": [1],
        "cum_short_starts": [2], "cum_short_wins": [1],
    })
    entry_df = pd.DataFrame({
        "race_id": ["20250115A01"],
        "umaban": [1],
        "kettonum": ["H001"],
    })

    store = _make_store_mock(career)
    bf = BloodlineFeatures(store)
    result = bf.compute(entry_df)

    # blood_total_wr = (2 + 1) / (5 + 11) = 3/16 = 0.1875
    assert abs(result.iloc[0]["blood_total_wr"] - (2 + ALPHA_PRIOR) / (5 + TOTAL_OFFSET)) < 1e-6


def test_pit_prize_log():
    """累積賞金の log 変換が正しいこと"""
    career = pd.DataFrame({
        "race_id": ["20250115A01"],
        "kettonum": ["H001"],
        "race_date": pd.to_datetime(["2025-01-15"]),
        "cum_starts": [3],
        "cum_wins": [1],
        "cum_prize": [50000.0],
        "cum_turf_starts": [3], "cum_turf_wins": [1],
        "cum_dirt_starts": [0], "cum_dirt_wins": [0],
        "cum_short_starts": [0], "cum_short_wins": [0],
    })
    entry_df = pd.DataFrame({
        "race_id": ["20250115A01"],
        "umaban": [1],
        "kettonum": ["H001"],
    })

    store = _make_store_mock(career)
    bf = BloodlineFeatures(store)
    result = bf.compute(entry_df)

    # blood_prize_log = log(1 + 50000) = log(50001)
    assert abs(result.iloc[0]["blood_prize_log"] - np.log1p(50000)) < 1e-6


def test_pit_surface_wr():
    """芝別勝率が正しいこと"""
    career = pd.DataFrame({
        "race_id": ["20250115A01"],
        "kettonum": ["H001"],
        "race_date": pd.to_datetime(["2025-01-15"]),
        "cum_starts": [5],
        "cum_wins": [2],
        "cum_prize": [0.0],
        "cum_turf_starts": [3], "cum_turf_wins": [1],
        "cum_dirt_starts": [2], "cum_dirt_wins": [1],
        "cum_short_starts": [0], "cum_short_wins": [0],
    })
    entry_df = pd.DataFrame({
        "race_id": ["20250115A01"],
        "umaban": [1],
        "kettonum": ["H001"],
    })

    store = _make_store_mock(career)
    bf = BloodlineFeatures(store)
    result = bf.compute(entry_df)

    # blood_surface_wr = (1 + 1) / (3 + 11) = 2/14 ≈ 0.1429
    expected = (1 + ALPHA_PRIOR) / (3 + TOTAL_OFFSET)
    assert abs(result.iloc[0]["blood_surface_wr"] - expected) < 1e-6


def test_pit_surface_wr_zero_turf_starts_is_nan():
    """芝出走が0の場合は blood_surface_wr = NaN"""
    career = pd.DataFrame({
        "race_id": ["20250115A01"],
        "kettonum": ["H001"],
        "race_date": pd.to_datetime(["2025-01-15"]),
        "cum_starts": [3],
        "cum_wins": [1],
        "cum_prize": [0.0],
        "cum_turf_starts": [0], "cum_turf_wins": [0],
        "cum_dirt_starts": [3], "cum_dirt_wins": [1],
        "cum_short_starts": [0], "cum_short_wins": [0],
    })
    entry_df = pd.DataFrame({
        "race_id": ["20250115A01"],
        "umaban": [1],
        "kettonum": ["H001"],
    })

    store = _make_store_mock(career)
    bf = BloodlineFeatures(store)
    result = bf.compute(entry_df)

    assert pd.isna(result.iloc[0]["blood_surface_wr"])
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python -m pytest tests/test_bloodline_features_pit.py -v`
Expected: FAIL (現在の実装は horses.parquet を使うため、PIT 値を返さない)

- [ ] **Step 3: `bloodline_features.py` を PIT 版に書き換え**

```python
# src/features/bloodline_features.py — PIT 版 (完全書き換え)
"""bloodline_features.py — Group B: 血統・産駒成績特徴量 (Point-in-Time)

主な特徴量:
  - blood_surface_wr:  芝別勝率 Beta平滑化 (entries+races から再構成)
  - blood_distance_wr: 芝1600M以下勝率 Beta平滑化 (entries+races から再構成)
  - blood_condition_wr: 馬場状態別勝率 (Phase 2, 現在NaN)
  - blood_total_wr:    総合成績勝率 Beta平滑化 (entries から再構成)
  - blood_prize_log:   log(1 + 累計賞金)
  - blood_keito_cd:    系統コード (Phase 2, 現在NaN)

ルックアヘッドバイアス修正:
  従来は horses.parquet (x_UMA ETL時点の累積値) を使用しており、
  BT で未来のレース結果が特徴量に混入していた。
  修正後は horse_career_stats.parquet (各レース時点での事前累積値) を使用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Beta prior parameters for win-rate smoothing
ALPHA_PRIOR: int = 1
BETA_PRIOR: int = 10
TOTAL_OFFSET: int = ALPHA_PRIOR + BETA_PRIOR  # = 11

FEATURE_COLS: list[str] = [
    "blood_surface_wr",
    "blood_distance_wr",
    "blood_condition_wr",
    "blood_total_wr",
    "blood_prize_log",
    "blood_keito_cd",
]


class BloodlineFeatures:
    """Point-in-Time 血統特徴量を生成。

    horse_career_stats.parquet から各レース時点での累積成績を読み込み、
    Beta 平滑化勝率を計算する。
    """

    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._career_cache: pd.DataFrame | None = None

    def _load_career_stats(self) -> pd.DataFrame:
        if self._career_cache is None:
            from db.readers import load_career_stats

            self._career_cache = load_career_stats(self.store)
        return self._career_cache

    @staticmethod
    def _smoothed_wr(wins: float, total: float) -> float:
        """Beta(alpha, beta) 平滑化勝率: (wins+1)/(total+11)。

        total=0 の場合は NaN を返す (未出走カテゴリ)。
        """
        if total == 0:
            return float("nan")
        return (wins + ALPHA_PRIOR) / (total + TOTAL_OFFSET)

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """entry_df (race_id, umaban, kettonum) -> 血統特徴量 DataFrame。

        horse_career_stats.parquet から point-in-time 累積成績を取得し、
        Beta 平滑化勝率を計算する。
        """
        career = self._load_career_stats()

        if "kettonum" not in entry_df.columns or career.empty:
            return entry_df[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        # entry_df と career_stats を (race_id, kettonum) で結合
        merge_keys = ["race_id", "kettonum"]
        merged = entry_df[["race_id", "umaban", "kettonum"]].merge(
            career, on=merge_keys, how="left"
        )

        result = merged[["race_id", "umaban"]].copy()

        # --- 総合成績勝率 ---
        result["blood_total_wr"] = np.where(
            merged["cum_starts"].fillna(0) == 0,
            np.nan,
            (merged["cum_wins"].fillna(0) + ALPHA_PRIOR)
            / (merged["cum_starts"].fillna(0) + TOTAL_OFFSET),
        )

        # --- 累計賞金 (log変換) ---
        prize = merged["cum_prize"].fillna(0)
        result["blood_prize_log"] = np.where(
            prize > 0, np.log1p(prize), np.nan
        )

        # --- 芝別勝率 (全芝 = ba1chakukaisu の近似) ---
        result["blood_surface_wr"] = np.where(
            merged["cum_turf_starts"].fillna(0) == 0,
            np.nan,
            (merged["cum_turf_wins"].fillna(0) + ALPHA_PRIOR)
            / (merged["cum_turf_starts"].fillna(0) + TOTAL_OFFSET),
        )

        # --- 芝1600以下勝率 (kyori1chakukaisu の近似) ---
        result["blood_distance_wr"] = np.where(
            merged["cum_short_starts"].fillna(0) == 0,
            np.nan,
            (merged["cum_short_wins"].fillna(0) + ALPHA_PRIOR)
            / (merged["cum_short_starts"].fillna(0) + TOTAL_OFFSET),
        )

        # --- 馬場状態別勝率 — Phase 2 ---
        result["blood_condition_wr"] = np.nan

        # --- 系統コード — Phase 2 ---
        result["blood_keito_cd"] = np.nan

        return result[["race_id", "umaban"] + FEATURE_COLS]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python -m pytest tests/test_bloodline_features_pit.py -v`
Expected: all PASS

- [ ] **Step 5: 既存テスト `tests/test_bloodline_features.py` を更新**

PIT 版では `load_career_stats(store)` を呼ぶため、既存のモック (`store.read.return_value = horses_df`)
では career_stats が返らない。テストヘルパーを PIT 版に適合させる。

`_make_store` を差し替え、`_make_horses_row` の代わりに career stats 行を返すモックに変更:

```python
# tests/test_bloodline_features.py — PIT 版に更新
"""test_bloodline_features.py — BloodlineFeatures (PIT 版) の単体テスト"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from features.bloodline_features import ALPHA_PRIOR, TOTAL_OFFSET, BloodlineFeatures


def _make_store(career_df: pd.DataFrame) -> MagicMock:
    """PIT 版モック: load_career_stats 経由で career_df を返す。"""
    store = MagicMock()
    store.exists.return_value = not career_df.empty
    store.read.return_value = career_df
    return store


def _make_entry(n: int = 1, ketto_nums: list[str] | None = None) -> pd.DataFrame:
    if ketto_nums is None:
        ketto_nums = ["K001"] * n
    return pd.DataFrame({
        "race_id": ["r1"] * n,
        "umaban": list(range(1, n + 1)),
        "kettonum": ketto_nums,
    })


def _make_career_row(
    kettonum: str = "K001",
    cum_starts: int = 80,
    cum_wins: int = 10,
    cum_prize: float = 50000.0,
    cum_turf_starts: int = 50,
    cum_turf_wins: int = 5,
    cum_dirt_starts: int = 30,
    cum_dirt_wins: int = 5,
    cum_short_starts: int = 30,
    cum_short_wins: int = 3,
) -> dict:
    """Build one row of career stats with sensible defaults."""
    return {
        "race_id": "r1",
        "kettonum": kettonum,
        "race_date": pd.Timestamp("2025-01-01"),
        "cum_starts": cum_starts,
        "cum_wins": cum_wins,
        "cum_prize": cum_prize,
        "cum_turf_starts": cum_turf_starts,
        "cum_turf_wins": cum_turf_wins,
        "cum_dirt_starts": cum_dirt_starts,
        "cum_dirt_wins": cum_dirt_wins,
        "cum_short_starts": cum_short_starts,
        "cum_short_wins": cum_short_wins,
    }


# === Tests ===


class TestBloodTotalWr:
    def test_blood_total_wr(self):
        """cum_wins=10, cum_starts=80 -> (10+1)/(80+11)"""
        career = pd.DataFrame([_make_career_row(cum_wins=10, cum_starts=80)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        expected = (10 + ALPHA_PRIOR) / (80 + TOTAL_OFFSET)
        assert abs(result["blood_total_wr"].iloc[0] - expected) < 1e-10

    def test_debut_horse_nan(self):
        """cum_starts=0 -> NaN"""
        career = pd.DataFrame([_make_career_row(cum_starts=0, cum_wins=0)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert np.isnan(result["blood_total_wr"].iloc[0])


class TestBloodSurfaceWr:
    def test_blood_surface_wr(self):
        """cum_turf_wins=5, cum_turf_starts=50 -> (5+1)/(50+11)"""
        career = pd.DataFrame([_make_career_row(cum_turf_wins=5, cum_turf_starts=50)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        expected = (5 + ALPHA_PRIOR) / (50 + TOTAL_OFFSET)
        assert abs(result["blood_surface_wr"].iloc[0] - expected) < 1e-10

    def test_no_turf_starts_is_nan(self):
        """cum_turf_starts=0 -> NaN"""
        career = pd.DataFrame([_make_career_row(cum_turf_starts=0, cum_turf_wins=0)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert np.isnan(result["blood_surface_wr"].iloc[0])


class TestBloodDistanceWr:
    def test_blood_distance_wr(self):
        """cum_short_wins=3, cum_short_starts=30 -> (3+1)/(30+11)"""
        career = pd.DataFrame([_make_career_row(cum_short_wins=3, cum_short_starts=30)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        expected = (3 + ALPHA_PRIOR) / (30 + TOTAL_OFFSET)
        assert abs(result["blood_distance_wr"].iloc[0] - expected) < 1e-10

    def test_no_short_starts_is_nan(self):
        """cum_short_starts=0 -> NaN"""
        career = pd.DataFrame([_make_career_row(cum_short_starts=0, cum_short_wins=0)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert np.isnan(result["blood_distance_wr"].iloc[0])


class TestBloodPrizeLog:
    def test_blood_prize_log(self):
        career = pd.DataFrame([_make_career_row(cum_prize=50000.0)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert abs(result["blood_prize_log"].iloc[0] - np.log1p(50000)) < 1e-6

    def test_blood_prize_log_zero(self):
        career = pd.DataFrame([_make_career_row(cum_prize=0.0)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert np.isnan(result["blood_prize_log"].iloc[0])


class TestEdgeCases:
    def test_missing_horse(self):
        """kettonum not in career -> all NaN"""
        career = pd.DataFrame([_make_career_row(kettonum="K999")])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry(ketto_nums=["K001"]))
        from features.bloodline_features import FEATURE_COLS
        for col in FEATURE_COLS:
            if col in ("blood_condition_wr", "blood_keito_cd"):
                continue
            assert np.isnan(result[col].iloc[0]), f"Expected NaN for {col}"

    def test_empty_entry(self):
        career = pd.DataFrame([_make_career_row()])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(
            pd.DataFrame(columns=["race_id", "umaban", "kettonum"])
        )
        assert len(result) == 0

    def test_empty_career(self):
        """career が空 -> all NaN"""
        store = _make_store(pd.DataFrame())
        result = BloodlineFeatures(store).compute(_make_entry())
        from features.bloodline_features import FEATURE_COLS
        for col in FEATURE_COLS:
            assert np.isnan(result[col].iloc[0]), f"Expected NaN for {col}"

    def test_phase2_columns_are_nan(self):
        career = pd.DataFrame([_make_career_row()])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert np.isnan(result["blood_condition_wr"].iloc[0])
        assert np.isnan(result["blood_keito_cd"].iloc[0])

    def test_multiple_horses(self):
        career = pd.DataFrame([
            _make_career_row(kettonum="K001", cum_turf_wins=5, cum_turf_starts=50),
            _make_career_row(kettonum="K002", cum_turf_wins=10, cum_turf_starts=100),
        ])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(
            _make_entry(n=2, ketto_nums=["K001", "K002"])
        )
        assert len(result) == 2
        assert abs(result["blood_surface_wr"].iloc[0] - (5+1)/(50+11)) < 1e-10
        assert abs(result["blood_surface_wr"].iloc[1] - (10+1)/(100+11)) < 1e-10
```

- [ ] **Step 6: 全テストの回帰確認**

Run: `python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 7: コミット**

```bash
git add src/features/bloodline_features.py tests/test_bloodline_features.py tests/test_bloodline_features_pit.py
git commit -m "feat: switch BloodlineFeatures to point-in-time career stats"
```

**注意:** `tests/test_bloodline_features.py` は完全に書き換え (旧: x_UMA データモック → 新: career stats モック)。
`tests/test_bloodline_features_pit.py` は追加の PIT 固有テストとして残す。
`_smoothed_wr` のテストは `test_bloodline_features.py` に残さない (PIT 版では vectorized `np.where` を使用しメソッドが dead code になるため)。

---

### Task 4: 設計書の更新

**Files:**
- Modify: `docs/superpowers/specs/2026-03-29-feature-engineering-design.md`

- [ ] **Step 1: 設計書の時点制約セクションを更新**

「血統 (Group B) | 時点制約なし」の記述を以下に変更:

```markdown
| 血統 (Group B) | **race_id 時点制約あり** (horse_career_stats.parquet) | x_UMAはETL時点累積値でルックアヘッドが発生。entries.parquetから各レース時点の累積値を事前計算して使用。 |
```

「血統特徴量の時点制約なしについての設計根拠」セクションを以下に置き換え:

```markdown
**血統特徴量の point-in-time 制約についての設計根拠:**
- x_UMA は ETL 時点の累積値。BT で最大15ヶ月分の未来情報が混入する (ルックアヘッド)
- 影響: BT 2025テストで AUC +0.129, デビュー馬の r=0.478
- 修正: entries.parquet から (kettonum, race_id) ごとの事前累積値を計算
- 精度: ba1(芝直線)→芝全般, kyori1(芝1600以下)→同一定義 で近似
- **結論: point-in-time でルックアヘッドを排除。近似値でも学習・推論整合性が保たれる**
```

- [ ] **Step 2: コミット**

```bash
git add docs/superpowers/specs/2026-03-29-feature-engineering-design.md
git commit -m "docs: update feature engineering spec for PIT blood features"
```

---

### Task 5: 事前計算の実行と検証

**Files:**
- Generate: `data/raw/horse_career_stats.parquet`

- [ ] **Step 1: 事前計算を実行**

Run: `python scripts/precompute_career_stats.py`

Expected output:
```
Loading entries.parquet...
  852216 rows
Loading races.parquet...
Computing career stats...
  ~546000 rows
Saving to data/raw/horse_career_stats.parquet...
Debut rate: ~XX%
Done.
```

- [ ] **Step 2: デビュー馬の NaN 率を確認**

```python
import pandas as pd
stats = pd.read_parquet("data/raw/horse_career_stats.parquet")
debut = (stats["cum_starts"] == 0).sum()
print(f"Debut entries: {debut} / {len(stats)} ({debut/len(stats)*100:.1f}%)")
```

BT 2025テストのデビュー馬 (4,989頭) が全て cum_starts=0 であることを確認。

- [ ] **Step 3: コミット**

```bash
git add data/raw/horse_career_stats.parquet
git commit -m "data: add precomputed point-in-time career stats"
```

---

### Task 6: バックテストでルックアヘッド修正を検証

- [ ] **Step 1: BT を実行 (2025年テスト)**

Run:
```bash
python scripts/run_backtest.py \
  --train-start 20210101 --train-end 20241231 \
  --test-start 20250101 --test-end 20251231 \
  --ensemble
```

- [ ] **Step 2: 結果を確認**

修正前 (ルックアヘッドあり): ROI 216.6%, 的中率 48.6%
修正後の期待: ROI 低下は正常 (未来情報の排除)。的中率も低下するはず。

**重要:** 修正後の ROI が PT (119.4%) に近い値になれば、ルックアヘッドが排除された証拠。
BT ROI < PT ROI の場合は過剰修正の可能性あり (feature definition の差異を要調査)。

- [ ] **Step 3: PT でも実行して BT/PT の整合性を確認**

```bash
python scripts/run_paper_trading.py --mode diagnose --start 2025-04-01 --end 2025-04-12
```

BT 2025年4月の ROI と PT 4月の ROI を比較し、乖離が縮小したことを確認。

---

## リスクと注意点

1. **feature definition の変更:** ba1(芝直線)→芝全般 の近似により、モデルの再学習が必要。
   修正後の BT は学習時から PIT 値を使うため、テスト結果は「新しい feature definition での性能」となる。

2. **entries.parquet のカバレッジ:** 2015年以前のデータがないため、2015年より前にデビューした馬の初期成績が欠損。
   影響は限定的 (2021年以降の学習データでは該当馬はベテランで、初期成績の欠損影響は小)。

3. **PT 新規エントリ:** EveryDB2 から取得した当日エントリが career_stats.parquet に存在しない場合、NaN になる。
   これは正しい挙動 (デビュー馬 = NaN)。再エントリ (既存馬) は career_stats にヒットする。
