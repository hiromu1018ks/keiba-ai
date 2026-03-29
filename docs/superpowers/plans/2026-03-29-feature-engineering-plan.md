# Feature Engineering 実装計画 — ROI >100%

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 特徴量エンジニアリング設計書 (spec) に従い、Stage1を30特徴量に拡張しStage2に騎手/調教師コンテキストを移動することでROI >100%を目指す。

**Architecture:** 既存のLightGBMパイプラインに新規特徴量グループを追加。ETL層で新規Parquetファイルとカラムを追加し、特徴量層でGroup A〜Fを実装し、モデル層のFEATURE_COLSを更新する。Stage1から騎手特徴量を削除しStage2に移動。

**Tech Stack:** Python 3.11, LightGBM (lambdarank/binary), pandas, pyarrow, pytest

**Spec:** `docs/superpowers/specs/2026-03-29-feature-engineering-design.md`

---

## File Structure

### Create (new)
| File | Responsibility |
|------|---------------|
| `src/features/bloodline_features.py` | Group B: 血統・産駒成績 (x_UMA) |
| `src/features/jockey_context_features.py` | Group C: 騎手コンテキスト (Stage2) |
| `src/features/trainer_context_features.py` | Group D: 調教師コンテキスト (Stage2) |
| `src/features/interaction_features.py` | Group E: 交互作用特徴量 |
| `tests/test_bloodline_features.py` | Group B test |
| `tests/test_jockey_context_features.py` | Group C test |
| `tests/test_trainer_context_features.py` | Group D test |
| `tests/test_interaction_features.py` | Group E test |
| `tests/test_history_features_v2.py` | Group A 拡張 test |

### Modify (existing)
| File | Lines to change | What |
|------|----------------|------|
| `src/db/etl.py` | ~674-726 | entries SELECTにTimeDIFN/Jyuni1c/Jyuni4c追加, +新規3テーブルETL |
| `src/db/schema.py` | ~64-81 | raw.entriesに3カラム追加 |
| `src/db/repository.py` | ~33-98 | load_horses/load_jockey_stats/load_trainer_stats 追加 |
| `src/features/horse_history_features.py` | ~125-309 | BASE_COLS拡張, compute()に7新規特徴量 |
| `src/features/feature_engine.py` | ~19-211 | 新規import + build_all()にGroup B/E追加 |
| `src/features/intra_race_features.py` | ~11-31 | race_z→race_rank統一 |
| `src/models/stage1_ability_model.py` | ~27-47 | FEATURE_COLS 30列に更新 |
| `src/models/place_ability_model.py` | ~25-48,59 | FEATURE_COLS更新 + dropna()修正 |
| `src/models/ev_correction_model.py` | ~25-41 | 騎手/調教師FEATURE_COLS追加 |
| `src/pipelines/training_pipeline.py` | ~404 | Group B/C/Dのcompute統合 |

---

## Task 1: ETL — entries追加カラム (TimeDIFN, Jyuni1c, Jyuni4c)

**Files:**
- Modify: `src/db/etl.py` (~line 674 entries SQL, ~707 entries_out columns)
- Modify: `src/db/schema.py` (~line 64 raw.entries DDL)
- Test: `tests/test_etl.py` (新規 or 既存)

- [ ] **Step 1: schema.py に3カラム追加**

```python
# src/db/schema.py — raw.entries テーブル定義 (line 64-81)
# 追加する3カラム (haron_time_l3 の後):
    time_diff       FLOAT,      -- 勝馬差タイム (TimeDIFN)
    corner_1c       INTEGER,    -- 1コーナー通過順位 (Jyuni1c)
    corner_4c       INTEGER,    -- 4コーナー通過順位 (Jyuni4c)
```

- [ ] **Step 2: etl.py Parquet path の SELECT に3カラム追加**

```python
# src/db/etl.py — entries_sql (line 674-682)
# 追加: timedifn, jyuni1c, jyuni4c
entries_sql = f"""
    SELECT
        {race_id_expr} AS race_id,
        {race_date_expr} AS race_date,
        s.umaban,
        s.kettonum,
        s.kakuteijyuni,
        s.time,
        s.timedifn,
        s.odds,
        s.ninki,
        s.bataijyu,
        s.zogenfugo,
        s.zogensa,
        s.kisyucode,
        s.chokyosicode,
        s.harontimel3,
        s.jyuni1c,
        s.jyuni4c,
        s.honsyokin,
        s.kyakusitukubun
    FROM n_uma_race s
    WHERE ...
"""
```

- [ ] **Step 3: etl.py 変換ブロックに3カラム追加**

```python
# src/db/etl.py — type conversion block (after line 703)
entries_df["time_diff"] = entries_df["timedifn"].apply(_to_float)
entries_df["corner_1c"] = entries_df["jyuni1c"].apply(_to_int)
entries_df["corner_4c"] = entries_df["jyuni4c"].apply(_to_int)
```

- [ ] **Step 4: etl.py entries_out 列リストに3カラム追加**

```python
# src/db/etl.py — entries_out column list (after line 714)
# haron_time_l3 の次に追加:
    "time_diff",
    "corner_1c",
    "corner_4c",
```

- [ ] **Step 5: テスト確認**

```bash
python -m pytest tests/ -v -k "etl"
```

- [ ] **Step 6: コミット**

```bash
git add src/db/etl.py src/db/schema.py
git commit -m "feat: ETL entriesにtime_diff/corner_1c/corner_4cを追加"
```

---

## Task 2: ETL — 新規Parquetファイル (horses, jockey_stats, trainer_stats)

**Files:**
- Modify: `src/db/etl.py` (新規関数3つ + `run_full_etl_to_parquet` に追加)
- Test: `tests/test_etl_new_tables.py` (新規)

- [ ] **Step 1: `_etl_horses_to_parquet()` 関数を追加**

```python
# src/db/etl.py — 新規関数
def _etl_horses_to_parquet(store: ParquetStore) -> None:
    """x_UMA → data/raw/horses.parquet"""
    sql = """
        SELECT
            kettonum,
            ketto3infohansyokunum1, ketto3infohansyokunum2, ketto3infohansyokunum3, ketto3infohansyokunum4,
            ketto3infohansyokunum5, ketto3infohansyokunum6, ketto3infohansyokunum7, ketto3infohansyokunum8,
            ketto3infohansyokunum9, ketto3infohansyokunum10, ketto3infohansyokunum11, ketto3infohansyokunum12,
            ketto3infohansyokunum13, ketto3infohansyokunum14,
            ba1chakukaisu1, ba1chakukaisu2, ba1chakukaisu3, ba1chakukaisu4, ba1chakukaisu5, ba1chakukaisu6,
            ba2chakukaisu1, ba2chakukaisu2, ba2chakukaisu3, ba2chakukaisu4, ba2chakukaisu5, ba2chakukaisu6,
            ba3chakukaisu1, ba3chakukaisu3, ba3chakukaisu4, ba3chakukaisu5, ba3chakukaisu6,
            ba4chakukaisu1, ba4chakukaisu2, ba4chakukaisu3, ba4chakukaisu4, ba4chakukaisu5, ba4chakukaisu6,
            ba5chakukaisu1, ba5chakukaisu2, ba5chakukaisu3, ba5chakukaisu4, ba5chakukaisu5, ba5chakukaisu6,
            ba6chakukaisu1, ba6chakukaisu2, ba6chakukaisu3, ba6chakukaisu4, ba6chakukaisu5, ba6chakukaisu6,
            kyori1chakukaisu1, kyori1chakukaisu2, kyori1chakukaisu3, kyori1chakukaisu4, kyori1chakukaisu5, kyori1chakukaisu6,
            kyori2chakukaisu1, kyori2chakukaisu2, kyori2chakukaisu3, kyori2chakukaisu4, kyori2chakukaisu5, kyori2chakukaisu6,
            kyori3chakukaisu1, kyori3chakukaisu2, kyori3chakukaisu3, kyori3chakukaisu4, kyori3chakukaisu5, kyori3chakukaisu6,
            kyori4chakukaisu1, kyori4chakukaisu2, kyori4chakukaisu3, kyori4chakukaisu4, kyori4chakukaisu5, kyori4chakukaisu6,
            kyori5chakukaisu1, kyori5chakukaisu2, kyori5chakukaisu3, kyori5chakukaisu4, kyori5chakukaisu5, kyori5chakukaisu6,
            kyori6chakukaisu1, kyori6chakukaisu2, kyori6chakukaisu3, kyori6chakukaisu4, kyori6chakukaisu5, kyori6chakukaisu6,
            chuochakukaisu1, chuochakukaisu2, chuochakukaisu3, chuochakaisu4, chuochakukaisu5, chuochakukaisu6,
            ruikeihonsyoheichi,
            kyakusitu1, kyakusitu2, kyakusitu3, kyakusitu4
        FROM x_uma
        WHERE kettonum IS NOT NULL
    """
    # ... execute + ParquetStore.write("raw", "horses", df)
```

- [ ] **Step 2: `_etl_jockey_stats_to_parquet()` 関数を追加**

```python
def _etl_jockey_stats_to_parquet(store: ParquetStore) -> None:
    """x_KISYU_SEISEKI → data/raw/jockey_stats.parquet"""
    # SELECT: setyear, kisyucode, heichichakukaisu1-6, jyo1-20chakukaisu1-6,
    #        kyori1-6chakukaisu1-6, honsyokinheichi
    # WHERE setyear IS NOT NULL
```

- [ ] **Step 3: `_etl_trainer_stats_to_parquet()` 関数を追加**

```python
def _etl_trainer_stats_to_parquet(store: ParquetStore) -> None:
    """x_CHOKYO_SEISEKI → data/raw/trainer_stats.parquet"""
    # SELECT: setyear, chokyosicode, heichichakukaisu1-6, jyo1-20chakukaisu1-6,
    #        kyori1-6chakukaisu1-6, honsyokinheichi
    # WHERE setyear IS NOT NULL
```

- [ ] **Step 4: `run_full_etl_to_parquet()` に3関数呼び出しを追加**

```python
# src/db/etl.py — run_full_etl_to_parquet() の末尾に追加
_etl_horses_to_parquet(store)
_etl_jockey_stats_to_parquet(store)
_etl_trainer_stats_to_parquet(store)
```

- [ ] **Step 5: テスト確認**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 6: コミット**

```bash
git add src/db/etl.py
git commit -m "feat: ETLにhorses/jockey_stats/trainer_stats Parquet出力を追加"
```

---

## Task 3: DataRepository拡張

**Files:**
- Modify: `src/db/repository.py` (~line 33-98)
- Test: `tests/test_repository.py` (新規 or 既存)

- [ ] **Step 1: 3つのload メソッドを追加**

```python
# src/db/repository.py — DataRepository に追加
def load_horses(self) -> pd.DataFrame:
    """x_UMA 馬マスターデータ (血統・産駒成績)"""
    return self._store.read("raw", "horses")

def load_jockey_stats(self) -> pd.DataFrame:
    """x_KISYU_SEISEKI 騎手年度別成績"""
    return self._store.read("raw", "jockey_stats")

def load_trainer_stats(self) -> pd.DataFrame:
    """x_CHOKYO_SEISEKI 調教師年度別成績"""
    return self._store.read("raw", "trainer_stats")
```

- [ ] **Step 2: テスト確認**

```bash
python -m pytest tests/test_repository.py -v
```

- [ ] **Step 3: コミット**

```bash
git add src/db/repository.py
git commit -m "feat: DataRepositoryにhorses/jockey_stats/trainer_statsローダーを追加"
```

---

## Task 4: Group A — HorseHistoryFeatures拡張 (7新規特徴量)

**Files:**
- Modify: `src/features/horse_history_features.py` (~line 125-309)
- Test: `tests/test_history_features_v2.py` (新規)

- [ ] **Step 1: テストファイル作成**

```python
# tests/test_history_features_v2.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from features.horse_history_features import HorseHistoryFeatures

def test_haron_time_l3_avg():
    """haron_time_l3_avg が過去3走の平均を返す"""
    # setup mock repo with history data containing haron_time_l3
    # verify result matches expected average

def test_time_diff_avg():
    """time_diff_avg が勝馬差タイムの3走平均を返す"""

def test_corner_1c_avg():
    """corner_1c_avg が1コーナー通過順位の3走平均を返す"""

def test_corner_4c_avg():
    """corner_4c_avg が4コーナー通過順位の3走平均を返す"""

def test_closing_index_avg():
    """closing_index_avg = (4C rank - finish rank) の3走平均を返す"""

def test_kyakusitu_cd():
    """kyakusitu_cd が最新走の公式脚質コードを返す"""

def test_leak_prevention():
    """target_date以降のレースが含まれないことを確認"""

def test_new_horse_nan():
    """過去成績がない馬はNaNを返す"""
```

- [ ] **Step 2: テスト実行 (FAIL確認)**

```bash
python -m pytest tests/test_history_features_v2.py -v
```

- [ ] **Step 3: BASE_COLS拡張**

```python
# src/features/horse_history_features.py — BASE_COLS (line 128)
BASE_COLS: list[str] = [
    "norm_finish_logit_avg",
    "haron_time_l3_avg",      # haron_time_zscore_avg から置換
    "haron_time_l3_zscore",   # 新規
    "time_diff_avg",           # 新規
    "corner_1c_avg",           # 新規
    "corner_4c_avg",           # 新規
    "closing_index_avg",       # 新規
    "kyakusitu_cd",            # 新規 (非数値・カテゴリ)
    "jockey_surprise",         # 既存 (Phase 1では残す → Task 8でStage2に移動)
    "jockey_cond_wr",          # 同上
]
```

- [ ] **Step 4: compute() 内 per-horse loop に新規特徴量追加**

```python
# src/features/horse_history_features.py — compute() 内の per-horse loop
# horse_past に以下を追加 (haron_time_l3, time_diff, corner_1c, corner_4c 列を利用)

# haron_time_l3_avg: 上り3Fタイムの3走平均
if "haron_time_l3" in horse_past.columns:
    ht_vals = horse_past["haron_time_l3"].dropna()
    haron_time_l3_avg = float(ht_vals.tail(3).mean()) if len(ht_vals) > 0 else float("nan")
else:
    haron_time_l3_avg = float("nan")

# haron_time_l3_zscore: 距離bin別z-scoreの3走平均
# (距離bin別平均/stdを過去データから expanding で計算)

# time_diff_avg: 勝馬差タイム3走平均
if "time_diff" in horse_past.columns:
    td_vals = horse_past["time_diff"].dropna()
    time_diff_avg = float(td_vals.tail(3).mean()) if len(td_vals) > 0 else float("nan")
else:
    time_diff_avg = float("nan")

# corner_1c_avg, corner_4c_avg
# closing_index_avg = normalized(4c) - normalized(finish) の3走平均
# kyakusitu_cd = 最新走の KyakusituKubun
```

- [ ] **Step 5: テスト実行 (PASS確認)**

```bash
python -m pytest tests/test_history_features_v2.py -v
```

- [ ] **Step 6: コミット**

```bash
git add src/features/horse_history_features.py tests/test_history_features_v2.py
git commit -m "feat: HorseHistoryFeaturesに7新規特徴量を追加 (haron_time/time_diff/corner/closing)"
```

---

## Task 5: Group B — BloodlineFeatures (新規ファイル)

**Files:**
- Create: `src/features/bloodline_features.py`
- Create: `tests/test_bloodline_features.py`

- [ ] **Step 1: テストファイル作成**

```python
# tests/test_bloodline_features.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from features.bloodline_features import BloodlineFeatures

def test_blood_surface_wr_smoothing():
    """Beta(1,10)平滑化: (wins+1)/(total+11)"""

def test_blood_surface_wr_zero_total():
    """total=0 → NaN"""

def test_blood_prize_log():
    """log(1 + prize) 変換"""

def test_blood_distance_wr():
    """距離別勝率 (距離bin分類)"""
```

- [ ] **Step 2: テスト実行 (FAIL確認)**

```bash
python -m pytest tests/test_bloodline_features.py -v
```

- [ ] **Step 3: BloodlineFeatures 実装**

```python
# src/features/bloodline_features.py
"""Group B: 血統・産駒成績特徴量 — x_UMA (静的)"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.repository import DataRepository

# Beta事前分布パラメータ
ALPHA_PRIOR = 1
BETA_PRIOR = 10
TOTAL_OFFSET = ALPHA_PRIOR + BETA_PRIOR  # = 11

FEATURE_COLS = [
    "blood_surface_wr",
    "blood_distance_wr",
    "blood_condition_wr",
    "blood_total_wr",
    "blood_prize_log",
    "blood_keito_cd",
]

class BloodlineFeatures:
    """x_UMA の産駒成績から血統特徴量を生成。静的 (馬ごとに1回計算)。"""

    def __init__(self, repo: DataRepository) -> None:
        self.repo = repo
        self._horses_cache: pd.DataFrame | None = None

    def _load_horses(self) -> pd.DataFrame:
        if self._horses_cache is None:
            self._horses_cache = self.repo.load_horses()
        return self._horses_cache

    @staticmethod
    def _smoothed_wr(wins: int, total: int) -> float:
        """Beta(α,β)平滑化勝率: (wins+1)/(total+11)。total=0→NaN"""
        if total == 0:
            return float("nan")
        return (wins + ALPHA_PRIOR) / (total + TOTAL_OFFSET)

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """
        entry_df (race_id, umaban, ketto_num) → 血統特徴量DataFrame
        """
        horses_df = self._load_horses()
        # entry_df.ketto_num で join
        # 各馬の x_UMA 成績列から smooth_wr を計算
        # surface: ba1-6 の着数1/(着数1+...+着数6)
        # distance: kyori1-6 の着数1/(着数1+...+着数6)
        # condition: jyotai は entries にはない → Phase 2
        # total: chuo 着数1/(着数1+...+着数6)
        # prize: log(1 + ruikeihonsyoheichi)
        # keito_cd: 保留 (x_KEITO join が必要) → Phase 2
        ...
```

- [ ] **Step 4: テスト実行 (PASS確認)**

```bash
python -m pytest tests/test_bloodline_features.py -v
```

- [ ] **Step 5: コミット**

```bash
git add src/features/bloodline_features.py tests/test_bloodline_features.py
git commit -m "feat: BloodlineFeatures (Group B) 血統・産駒成績を追加"
```

---

## Task 6: Group F — レース内正規化 rank統一

**Files:**
- Modify: `src/features/intra_race_features.py` (~line 11-31)
- Modify: `src/features/horse_history_features.py` — `add_race_transforms()` (line ~296)
- Test: `tests/test_intra_race_features.py` (新規 or 既存)

- [ ] **Step 1: `add_race_transforms()` を race_rank に統一**

```python
# src/features/horse_history_features.py — add_race_transforms()
# 変更前: _race_z と _race_pct の両方を生成
# 変更後: _race_rank (percentile rank) のみ生成

@staticmethod
def add_race_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """BASE_COLS の各列についてレース内 percentile rank を追加。"""
    df = df.copy()
    for col in HorseHistoryFeatures.BASE_COLS:
        if col not in df.columns:
            continue
        # percentile rank only (z-scoreは削除)
        df[f"{col}_race_rank"] = df.groupby("race_id")[col].rank(
            pct=True, method="average"
        )
    return df
```

- [ ] **Step 2: テスト確認**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 3: コミット**

```bash
git add src/features/horse_history_features.py
git commit -m "refactor: race_z/race_pct → race_rank に統一 (Group F)"
```

---

## Task 7: Group E — InteractionFeatures (新規ファイル)

**Files:**
- Create: `src/features/interaction_features.py`
- Create: `tests/test_interaction_features.py`

- [ ] **Step 1: テストファイル作成**

```python
# tests/test_interaction_features.py
import pandas as pd
from features.interaction_features import compute_interaction_features

def test_kyakusitu_x_distance():
    """脚質×距離bin の文字列結合"""
    df = pd.DataFrame({"kyakusitu_cd": [1, 2, 3], "distance_bin": ["sprint", "mile", "intermediate"]})
    result = compute_interaction_features(df)
    assert result["kyakusitu_x_distance"].tolist() == ["1_sprint", "2_mile", "3_intermediate"]

def test_weight_x_distance():
    """馬体重×距離の数値積"""
    df = pd.DataFrame({"weight_absolute": [450.0, 500.0], "distance": [1200, 2400]})
    result = compute_interaction_features(df)
    assert result["weight_x_distance"].tolist() == [450.0 * 1200, 500.0 * 2400]
```

- [ ] **Step 2: テスト実行 (FAIL確認)**

```bash
python -m pytest tests/test_interaction_features.py -v
```

- [ ] **Step 3: InteractionFeatures 実装**

```python
# src/features/interaction_features.py
"""Group E: 交互作用特徴量"""

from __future__ import annotations
import pandas as pd

def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    脚質×距離/馬場 + 体重×距離 の交互作用特徴量を追加。
    LightGBMカテゴリとして扱うため、文字列結合 → astype("category")。
    """
    df = df.copy()

    # 脚質×距離bin (カテゴリ積)
    if "kyakusitu_cd" in df.columns and "distance_bin" in df.columns:
        df["kyakusitu_x_distance"] = (
            df["kyakusitu_cd"].astype(str) + "_" + df["distance_bin"].astype(str)
        ).astype("category")

    # 脚質×馬場 (カテゴリ積)
    if "kyakusitu_cd" in df.columns and "surface" in df.columns:
        df["kyakusitu_x_surface"] = (
            df["kyakusitu_cd"].astype(str) + "_" + df["surface"].astype(str)
        ).astype("category")

    # 馬体重×距離 (数値積)
    if "weight_absolute" in df.columns and "distance" in df.columns:
        df["weight_x_distance"] = (
            df["weight_absolute"].fillna(0) * df["distance"].fillna(0)
        ).astype(float)

    return df
```

- [ ] **Step 4: テスト実行 (PASS確認)**

```bash
python -m pytest tests/test_interaction_features.py -v
```

- [ ] **Step 5: コミット**

```bash
git add src/features/interaction_features.py tests/test_interaction_features.py
git commit -m "feat: InteractionFeatures (Group E) 脚質×距離/馬場/体重を追加"
```

---

## Task 8: FeatureEngine統合 + Stage1 FEATURE_COLS更新

**Files:**
- Modify: `src/features/feature_engine.py` (~line 19-211)
- Modify: `src/models/stage1_ability_model.py` (~line 27-47)
- Modify: `src/models/place_ability_model.py` (~line 25-48, line 59)
- Modify: `src/pipelines/training_pipeline.py` (~line 404)

- [ ] **Step 1: FeatureEngine.build_all() にGroup B/Eを追加**

```python
# src/features/feature_engine.py — build_all() 内
# import 追加:
from features.bloodline_features import BloodlineFeatures
from features.interaction_features import compute_interaction_features

# build_all() のステップ追加 (既存Step 5の後):
# Step 6: Group B — 血統特徴量
bloodline = BloodlineFeatures(repo)
bloodline_df = bloodline.compute(merged_df)
merged_df = merged_df.merge(bloodline_df, on=["race_id", "umaban"], how="left")

# Step 7: Group E — 交互作用特徴量
merged_df = compute_interaction_features(merged_df)
```

- [ ] **Step 2: AbilityModel.FEATURE_COLS を30列に更新**

```python
# src/models/stage1_ability_model.py — FEATURE_COLS (line 27-47)
FEATURE_COLS: list[str] = [
    # レース条件 (7) — 変更なし
    "surface", "distance_bin", "track_condition_code",
    "grade_code", "field_size",
    "weight_diff_from_mean", "difficulty_score",
    # 過走成績 (8)
    "norm_finish_logit_avg",
    "haron_time_l3_avg",        # 新規
    "haron_time_l3_zscore",     # 新規
    "time_diff_avg",             # 新規
    "corner_1c_avg",             # 新規
    "corner_4c_avg",             # 新規
    "closing_index_avg",         # 新規
    "kyakusitu_cd",              # 新規 (cat)
    # 血統 (6) — 新規
    "blood_surface_wr",
    "blood_distance_wr",
    "blood_condition_wr",
    "blood_total_wr",
    "blood_prize_log",
    "blood_keito_cd",            # (cat, Phase 2)
    # 交互作用 (3) — 新規
    "kyakusitu_x_distance",     # (cat)
    "kyakusitu_x_surface",      # (cat)
    "weight_x_distance",
    # レース内正規化 (5) — rank_pctに統一
    "norm_finish_logit_avg_race_rank",
    "haron_time_l3_avg_race_rank",
    "time_diff_avg_race_rank",
    "corner_1c_avg_race_rank",
    "closing_index_avg_race_rank",
    # 馬体 (1)
    "weight_absolute",
]
# 合計: 30特徴量

# categorical columns (train内のastype):
# "surface", "distance_bin", "grade_code", "kyakusitu_cd",
# "blood_keito_cd", "kyakusitu_x_distance", "kyakusitu_x_surface"
```

- [ ] **Step 3: PlaceAbilityModel.FEATURE_COLS を更新 + dropna()修正**

```python
# src/models/place_ability_model.py

# FEATURE_COLS: AbilityModelと同じ30列 + "p_ability_win" = 31列
FEATURE_COLS: list[str] = [
    # AbilityModelと同じ30列 (コピー)
    ...,
    "p_ability_win",  # Stage1出力
]

# train() の dropna() 修正 (line 59):
# 変更前: df = df.dropna(subset=self.FEATURE_COLS).copy()
# 変更後: LightGBMにNaN処理を任せる (dropnaを削除)
df = df.copy()  # NaNはLightGBMが処理
```

- [ ] **Step 4: training_pipeline.py で Group B の repo 渡しを確認**

FeatureEngine.build_all() に `repo` を渡す必要がある。現在のシグネチャを確認し、必要なら`repo`を引数に追加。

- [ ] **Step 5: 全テスト実行**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 6: コミット**

```bash
git add src/features/feature_engine.py src/models/stage1_ability_model.py src/models/place_ability_model.py src/pipelines/training_pipeline.py
git commit -m "feat: Stage1 FEATURE_COLS 30列化 + FeatureEngine統合 + PlaceModel dropna修正"
```

---

## Task 9: Group C/D — Jockey/Trainer Context Features (Stage2)

**Files:**
- Create: `src/features/jockey_context_features.py`
- Create: `src/features/trainer_context_features.py`
- Create: `tests/test_jockey_context_features.py`
- Create: `tests/test_trainer_context_features.py`
- Modify: `src/models/ev_correction_model.py` (~line 25-41)

- [ ] **Step 1: JockeyContextFeatures テスト作成**

```python
# tests/test_jockey_context_features.py
def test_jockey_wr_overall():
    """SetYear < race_year の最新年を使用"""

def test_jockey_prize_log():
    """log(1 + prize) 変換"""

def test_year_boundary():
    """2024年1月レースでは2023年の成績のみ使用"""
```

- [ ] **Step 2: JockeyContextFeatures 実装**

```python
# src/features/jockey_context_features.py
"""Group C: 騎手コンテキスト特徴量 (Stage2のみ)"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.repository import DataRepository

FEATURE_COLS = [
    "jockey_surprise",       # 既存 (HorseHistoryFeaturesから取得)
    "jockey_cond_wr",        # 既存
    "jockey_wr_overall",     # 新規
    "jockey_wr_distance",    # 新規
    "jockey_wr_venue",       # 新規
    "jockey_prize_log",      # 新規
]

class JockeyContextFeatures:
    """x_KISYU_SEISEKI から騎手年度別特徴量を生成。SetYear < race_year。"""

    def __init__(self, repo: DataRepository) -> None:
        self.repo = repo
        self._stats_cache: pd.DataFrame | None = None

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """
        entry_df (race_id, umaban, kisyu_code, race_date) → 騎手特徴量DataFrame
        SetYear < race_year の最新年を使用。
        """
        ...
```

- [ ] **Step 3: TrainerContextFeatures 実装 (同パターン)**

```python
# src/features/trainer_context_features.py
"""Group D: 調教師コンテキスト特徴量 (Stage2のみ)"""
# JockeyContextFeatures と同パターン。x_CHOKYO_SEISEKI を使用。
```

- [ ] **Step 4: EVCorrectionModel.FEATURE_COLS に騎手/調教師を追加**

```python
# src/models/ev_correction_model.py — FEATURE_COLS (line 25-41)
# 既存11 + 騎手6 + 調教師4 = 21特徴量
FEATURE_COLS: list[str] = [
    # 既存 (11)
    "e_return_win_pred", "p_x_e_interaction", "p_minus_e_gap",
    "signed_log_error_win", "abs_log_error_win", "market_entropy",
    "popularity_rank", "surface", "distance_bin",
    "track_condition_code", "field_size",
    # 騎手コンテキスト (6)
    "jockey_surprise", "jockey_cond_wr",
    "jockey_wr_overall", "jockey_wr_distance",
    "jockey_wr_venue", "jockey_prize_log",
    # 調教師コンテキスト (4)
    "trainer_wr_overall", "trainer_wr_distance",
    "trainer_wr_venue", "trainer_prize_log",
]
```

- [ ] **Step 5: training_pipeline.py に Group C/D のcompute統合**

`_train_submodel()` 内でEVCorrectionModel.train()の前に JockeyContextFeatures と TrainerContextFeatures のcomputeを追加。

- [ ] **Step 6: 全テスト実行**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 7: コミット**

```bash
git add src/features/jockey_context_features.py src/features/trainer_context_features.py src/models/ev_correction_model.py src/pipelines/training_pipeline.py tests/test_jockey_context_features.py tests/test_trainer_context_features.py
git commit -m "feat: Jockey/Trainer Context (Group C/D) Stage2に追加 + EVCorrectionModel 21列化"
```

---

## Task 10: バックテスト検証

**Files:**
- Run: `scripts/run_backtest.py`
- Output: `backtest_result.json`

- [ ] **Step 1: ETL再実行 (新規データ取得)**

```bash
export PGPASSWORD=<password>
python scripts/run_etl.py --start 20140101 --end 20241231
```

Expected: 新規 `data/raw/horses.parquet`, `data/raw/jockey_stats.parquet`, `data/raw/trainer_stats.parquet` が生成。`entries.parquet` に `time_diff`, `corner_1c`, `corner_4c` 列が追加される。

- [ ] **Step 2: バックテスト実行**

```bash
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

Expected: ROI 66.3% (既存) からの改善を確認。

- [ ] **Step 3: 結果確認**

```bash
cat backtest_result.json | python -m json.tool
```

- [ ] **Step 4: コミット (結果ファイル)**

```bash
git add backtest_result.json
git commit -m "feat: 新特徴量30列でのバックテスト結果"
```

---

## 注意事項

### 既存テストへの影響
- `HorseHistoryFeatures` の BASE_COLS 変更により、既存の mock-based テストが壊れる可能性あり
- 各Task完了後に `python -m pytest tests/ -v` で全体テストを実行し、壊れたテストを修正

### PlaceAbilityModel の dropna() 問題
- 現在 `df.dropna(subset=self.FEATURE_COLS)` している (line 59)
- 新規特徴量 (blood_*, corner_*) がNaNの場合、行が削除される
- 修正: dropnaを削除し LightGBM に NaN処理を任せる (Task 8 Step 3)

### ETL実行前提
- Task 10 のバックテストには ETL再実行が必須
- ETLはPostgreSQLアクセスが必要 (localhost:5432/everydb2)
- ETLが実行できない環境では、既存Parquet + 手動列追加で代替可能
