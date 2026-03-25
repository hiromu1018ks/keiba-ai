# Phase B: 特徴量エンジン 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 競馬AI予測システムの特徴量エンジンを構築し、生データ(race/entry/odds)からMLモデル入力用特徴量DataFrameを生成するパイプラインを実装する。

**Architecture:** FeatureEngine クラスがオーケストレータとして機能し、4つのサブモジュール(レース内相対値/オッズ変化率/市場歪み/情報非対称性) + 難易度モデルに処理を委譲。バッチ学習(`build_all`)と単レース推論(`build_features`)の2つのインタフェースを提供。TDDで実装し、全テストはDB不要(mock使用)。

**Tech Stack:** pandas, numpy, pytest, unittest.mock, Python 3.11

**設計書参照:** `docs/design.md` §10(特徴量エンジン v5.3), §14.5(B-1〜B-6)

---

## ファイル構造

```
src/features/
├── __init__.py                    # パブリックAPIエクスポート
├── feature_engine.py              # メインオーケストレータ (B-1)
├── intra_race_features.py         # カテゴリB: レース内相対特徴量 (B-2)
├── odds_dynamics_features.py      # カテゴリC: オッズ変化率特徴量 (B-3)
├── market_bias_features.py        # カテゴリD: 市場歪み特徴量 (B-4)
├── info_asymmetry_features.py     # カテゴリE: 履歴特徴量 (B-5)
├── race_difficulty_model.py       # カテゴリE: レース難易度スコア (B-5)
└── leakage_validators.py          # 未来情報リーク検証 (B-6)

tests/
├── test_feature_engine.py
├── test_intra_race_features.py
├── test_odds_dynamics_features.py
├── test_market_bias_features.py
├── test_info_asymmetry_features.py
├── test_race_difficulty.py
└── test_leakage.py
```

## 特徴量カテゴリ対応表

| カテゴリ | モジュール | 主な出力列 | 消費モデル |
|----------|-----------|-----------|-----------|
| B: レース内相対値 | `intra_race_features.py` | `popularity_rank`, `weight_diff_from_mean`, `odds_rank` | WinTwoStageModel |
| C: オッズ変化率 | `odds_dynamics_features.py` | `odds_drop_rate_60_10`, `odds_drop_rate_30_10`, `odds_velocity`, `odds_volatility` | WinTwoStageModel |
| D: 市場歪み | `market_bias_features.py` | `p_market_win_adj`, `market_entropy`, `overround` | MarketModel, WinTwoStageModel, RaceQualityScreener |
| E: 情報非対称性 | `info_asymmetry_features.py` | `hist_hit_rate_topk`, `hist_roi_topk`, `hist_positive_return_ratio`, `hist_win_rate_same_condition`, `hist_market_entropy_avg` | RaceQualityScreener |
| E: 難易度 | `race_difficulty_model.py` | `difficulty_score` | RaceQualityScreener |

## 重要な制約 (設計書 §1.2)

- **Rule 1:** Stage1にオッズを入れない（特徴量エンジンが出力するオッズ由来特徴量はStage2以降でのみ使用）
- **Rule 11:** Market Modelの出力は差分(log_error)のみStage2に入力（p_market_predは不使用）
- **Rule 18:** hist系特徴量は `expanding().shift(1)` で未来情報リークを完全遮断

---

## Task B-1: Feature Engine (メインオーケストレータ)

**Files:**
- Create: `src/features/__init__.py`
- Create: `src/features/feature_engine.py`
- Create: `tests/test_feature_engine.py`

**Dependencies:** A-2 (domain types), A-3 (DB schema)

`build_all()` は `TrainingPipelineV5` から呼ばれ、3つのDataFrame(race_df, entry_df, odds_df)をマージして全特徴量を計算する。
`build_features()` は `BettingOrchestrator` から呼ばれ、単一レースの `Race` + `list[Entry]` から推論用特徴量を計算する。

### Step 1: 失敗するテストを書く — build_all() データマージ + 基本特徴量

```python
# tests/test_feature_engine.py
"""src/features/feature_engine.py のテスト"""

import pandas as pd
import pytest

from features.feature_engine import FeatureEngine


@pytest.fixture
def sample_race_df() -> pd.DataFrame:
    """1レース分の race データ（18頭立て）"""
    return pd.DataFrame({
        "year": [2024] * 18,
        "month_day": ["0324"] * 18,
        "jyo_cd": ["05"] * 18,
        "kaiji": ["03"] * 18,
        "nichiji": ["02"] * 18,
        "race_num": ["08"] * 18,
        "track_cd": [11] * 18,
        "distance": [1600] * 18,
        "tenko_cd": [1] * 18,
        "baba_cd": [1] * 18,
        "syubetu_cd": ["13"] * 18,
        "jyoken_cd": ["999"] * 18,
        "grade_cd": ["_"] * 18,
        "field_size": [18] * 18,
        "race_id": ["2024032405030208"] * 18,
        "surface": ["turf"] * 18,
        "distance_band": ["mile"] * 18,
    })


@pytest.fixture
def sample_entry_df() -> pd.DataFrame:
    """18頭の出走馬データ"""
    umaban = list(range(1, 19))
    win_odds = [1.5, 2.3, 3.1, 5.0, 8.2, 12.5, 18.0, 25.0, 35.0, 45.0,
                55.0, 68.0, 80.0, 95.0, 110.0, 130.0, 150.0, 200.0]
    return pd.DataFrame({
        "race_id": ["2024032405030208"] * 18,
        "umaban": umaban,
        "ketto_num": [f"000{i:07d}" for i in range(1, 19)],
        "finish_pos": [1, 2, 3, 4, 5, 0, 7, 8, 0, 10, 11, 12, 13, 14, 15, 16, 0, 18],
        "win_odds": win_odds,
        "ninki": list(range(1, 19)),
        "ba_taijyu": [480, 472, 488, 464, 496, 458, 500, 484, 468, 492,
                      476, 504, 460, 482, 498, 470, 486, 454],
        "zogen_fugo": [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3],
        "zogen_sa": [-4, 2, 0, -6, 4, 0, -2, 6, 0, -8, 2, 0, -4, 8, 0, -2, 4, 0],
        "kisyu_code": [f"010{i:02d}" for i in range(1, 19)],
        "chokyosi_code": [f"010{i:02d}" for i in range(1, 19)],
    })


@pytest.fixture
def sample_odds_df() -> pd.DataFrame:
    """18頭のオッズスナップショット"""
    umaban = list(range(1, 19))
    tan_odds = [1.5, 2.3, 3.1, 5.0, 8.2, 12.5, 18.0, 25.0, 35.0, 45.0,
                55.0, 68.0, 80.0, 95.0, 110.0, 130.0, 150.0, 200.0]
    return pd.DataFrame({
        "race_id": ["2024032405030208"] * 18,
        "umaban": umaban,
        "tan_odds": tan_odds,
        "fuku_odds": [1.1, 1.2, 1.3, 1.5, 1.8, 2.1, 2.5, 2.9, 3.3, 3.7,
                      4.1, 4.5, 4.9, 5.3, 5.7, 6.1, 6.5, 7.0],
    })


class TestFeatureEngineBuildAll:
    def test_merge_produces_correct_shape(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ):
        """race_df + entry_df + odds_df をマージして18行のDataFrameを返す"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert result.shape[0] == 18
        assert "race_id" in result.columns
        assert "umaban" in result.columns

    def test_output_has_basic_features(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ):
        """基本特徴量列が存在する"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        expected_cols = [
            "surface", "distance_bin", "track_condition_code",
            "grade_code", "field_size", "popularity_rank",
        ]
        for col in expected_cols:
            assert col in result.columns, f"列 '{col}' が不足"

    def test_distance_band_renamed_to_distance_bin(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ):
        """DBの distance_band を FEATURE_COLS 名の distance_bin にリネーム"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert "distance_bin" in result.columns
        # 全行が "mile" であることを確認
        assert (result["distance_bin"] == "mile").all()

    def test_track_condition_code_renamed_from_baba_cd(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ):
        """baba_cd を track_condition_code にリネーム"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert "track_condition_code" in result.columns
        assert (result["track_condition_code"] == 1).all()

    def test_surface_key_column_exists(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ):
        """downstream SubModelManager フィルタ用の surface_key 列が存在"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert "surface_key" in result.columns
        assert (result["surface_key"] == "turf").all()

    def test_exclude_steeple(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ):
        """障害レース(track_cd >= 51)を除外"""
        steeple_race = sample_race_df.copy()
        steeple_race["track_cd"] = 51
        steeple_race["surface"] = "exclude"
        steeple_entry = sample_entry_df.copy()
        steeple_entry["race_id"] = "1999010101010101"
        steeple_odds = sample_odds_df.copy()
        steeple_odds["race_id"] = "1999010101010101"

        combined_races = pd.concat([sample_race_df, steeple_race], ignore_index=True)
        combined_entries = pd.concat([sample_entry_df, steeple_entry], ignore_index=True)
        combined_odds = pd.concat([sample_odds, steeple_odds], ignore_index=True)

        engine = FeatureEngine(exclude_steeple=True)
        result = engine.build_all(combined_races, combined_entries, combined_odds)
        assert result.shape[0] == 18  # 障害レースの18頭は除外

    def test_no_exclude_steeple(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ):
        """exclude_steeple=False では障害レースも含む"""
        steeple_race = sample_race_df.copy()
        steeple_race["track_cd"] = 51
        steeple_entry = sample_entry_df.copy()
        steeple_entry["race_id"] = "1999010101010101"
        steeple_odds = sample_odds_df.copy()
        steeple_odds["race_id"] = "1999010101010101"

        combined_races = pd.concat([sample_race_df, steeple_race], ignore_index=True)
        combined_entries = pd.concat([sample_entry_df, steeple_entry], ignore_index=True)
        combined_odds = pd.concat([sample_odds, steeple_odds], ignore_index=True)

        engine = FeatureEngine(exclude_steeple=False)
        result = engine.build_all(combined_races, combined_entries, combined_odds)
        assert result.shape[0] == 36
```

### Step 2: テストを実行して失敗を確認

Run: `python -m pytest tests/test_feature_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features'`

### Step 3: FeatureEngine の最小実装

```python
# src/features/__init__.py
from features.feature_engine import FeatureEngine

__all__ = ["FeatureEngine"]
```

```python
# src/features/feature_engine.py
"""特徴量エンジン v5.3 — メインオーケストレータ

カテゴリ:
  A: 馬の能力 (Stage1出力、本モジュールでは計算しない)
  B: レース内相対値 (intra_race_features.py)
  C: オッズ変化率 (odds_dynamics_features.py)
  D: 市場歪み (market_bias_features.py)
  E: 情報非対称性 (info_asymmetry_features.py, race_difficulty_model.py)
  F: 距離帯・馬場 one-hot (SubModelManager が担当)
"""

from __future__ import annotations

import pandas as pd

from domain.models import Entry, Race


class FeatureEngine:
    """特徴量エンジンのメインオーケストレータ

    build_all(): バッチ学習用 — 3つのDataFrameをマージして全特徴量を計算
    build_features(): 推論用 — Race + list[Entry] から単レース特徴量を計算
    """

    def __init__(self, exclude_steeple: bool = True) -> None:
        self._exclude_steeple = exclude_steeple

    def build_all(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        odds_ts_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """バッチ特徴量生成（TrainingPipelineV5 から呼ばれる）

        Args:
            race_df: レースメタデータ (load_races() の出力)
            entry_df: 出走馬データ (load_entries_with_results() の出力)
            odds_df: オッズスナップショット (load_odds_snapshots() の出力)
            odds_ts_df: オッズ時系列データ (省略可)

        Returns:
            全馬の特徴量を含むDataFrame (1行 = 1馬)
        """
        # 1. race + entry を race_id で結合
        df = pd.merge(race_df, entry_df, on="race_id", how="inner")

        # 2. odds を (race_id, umaban) で結合
        df = pd.merge(df, odds_df, on=["race_id", "umaban"], how="left")

        # 3. 障害レース除外
        if self._exclude_steeple:
            df = df[df["track_cd"] < 51]

        # 4. 基本特徴量のマッピング
        df = self._map_basic_features(df)

        # 5. サブモジュールの特徴量計算（B-2〜B-5 で追加）
        # from features.intra_race_features import compute_intra_race_features
        # from features.market_bias_features import compute_market_bias
        # etc.

        return df

    def build_features(
        self,
        race: Race,
        entries: list[Entry],
        odds_snapshot: pd.DataFrame | None = None,
        odds_ts: pd.DataFrame | None = None,
        snap_minutes: int | None = None,
    ) -> pd.DataFrame:
        """単レース推論用特徴量生成（BettingOrchestrator から呼ばれる）

        設計書 §12 呼び出し: self.feat_engine.build_features(race, entries, snap_minutes=10)

        Args:
            race: レース情報ドメインモデル
            entries: 出走馬ドメインモデルのリスト
            odds_snapshot: 現在のオッズスナップショット
            odds_ts: オッズ時系列データ (省略可)
            snap_minutes: オッズスナップショットの取得分前 (例: 10 = t-10min)

        Returns:
            全馬の特徴量を含むDataFrame (1行 = 1馬)
        """
        # 1. Race → DataFrame
        race_data = {
            "race_id": race.race_id,
            "surface": race.surface.value,
            "distance_band": race.distance_band,
            "track_cd": race.track_cd,
            "distance": race.distance,
            "baba_cd": race.baba_cd,
            "grade_cd": race.grade_cd,
            "field_size": race.field_size,
            "tenko_cd": race.tenko_cd,
            "syubetu_cd": race.syubetu_cd,
            "jyoken_cd": race.jyoken_cd,
        }
        race_row = pd.DataFrame([race_data])

        # 2. list[Entry] → DataFrame
        entry_rows = []
        for e in entries:
            entry_rows.append({
                "race_id": race.race_id,
                "umaban": e.umaban,
                "ketto_num": e.ketto_num,
                "finish_pos": e.finish_pos,
                "win_odds": e.win_odds_actual,
                "ninki": e.popularity_rank,
                "ba_taijyu": e.ba_taijyu,
                "zogen_fugo": e.zogen_fugo,
                "zogen_sa": e.zogen_sa,
                "kisyu_code": e.kisyu_code,
                "chokyosi_code": e.chokyosi_code,
            })
        entry_df = pd.DataFrame(entry_rows)

        # 3. 結合
        df = pd.merge(race_row, entry_df, on="race_id", how="inner")

        # 4. オッズ結合
        if odds_snapshot is not None:
            df = pd.merge(df, odds_snapshot, on=["race_id", "umaban"], how="left")

        # 5. 基本特徴量マッピング
        df = self._map_basic_features(df)

        # 6. サブモジュールの特徴量計算（推論用 — hist特徴量は除く）

        return df

    def _map_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """DB列名 → FEATURE_COLS 名へのマッピング（旧列は削除して重複を回避）"""
        rename_map: dict[str, str] = {}
        if "distance_band" in df.columns:
            rename_map["distance_band"] = "distance_bin"
        if "baba_cd" in df.columns:
            rename_map["baba_cd"] = "track_condition_code"

        df = df.rename(columns=rename_map)

        # ninki → popularity_rank (ninki は別用途でも使うためコピー)
        if "ninki" in df.columns:
            df["popularity_rank"] = df["ninki"]

        # surface_key (downstream SubModelManager フィルタ用)
        if "surface" in df.columns:
            df["surface_key"] = df["surface"]

        return df
```

### Step 4: テストを実行して成功を確認

Run: `python -m pytest tests/test_feature_engine.py -v`
Expected: ALL PASS (7 tests)

### Step 5: build_features() 推論パスのテストを追加

```python
# tests/test_feature_engine.py に追加
from domain.models import Entry, Race


@pytest.fixture
def sample_race() -> Race:
    return Race(
        year=2024, month_day="0324", jyo_cd="05", kaiji="03", nichiji="02",
        race_num="08", track_cd=11, distance=1600, tenko_cd=1, baba_cd=1,
        syubetu_cd="13", jyoken_cd="999", grade_cd="_", field_size=18,
    )


@pytest.fixture
def sample_entries() -> list[Entry]:
    entries = []
    for i in range(1, 4):
        entries.append(Entry(
            race_id="2024032405030208",
            umaban=i,
            ketto_num=f"0000000{i}",
            finish_pos=i,
            win_odds_actual=float(i + 1),
            popularity_rank=i,
            running_style=i,
            ba_taijyu=480.0,
            zogen_fugo=2,
            zogen_sa=-2.0,
            kisyu_code="01001",
            chokyosi_code="01001",
        ))
    return entries


class TestFeatureEngineBuildFeatures:
    def test_build_features_returns_dataframe(self, sample_race, sample_entries):
        """Race + list[Entry] からDataFrameを生成"""
        engine = FeatureEngine()
        result = engine.build_features(sample_race, sample_entries)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3

    def test_build_features_has_basic_columns(self, sample_race, sample_entries):
        """推論結果に基本特徴量列が含まれる"""
        engine = FeatureEngine()
        result = engine.build_features(sample_race, sample_entries)
        assert "surface" in result.columns
        assert "distance_bin" in result.columns
        assert "track_condition_code" in result.columns
        assert "surface_key" in result.columns

    def test_build_features_with_odds_snapshot(self, sample_race, sample_entries):
        """オッズスナップショットを結合できる"""
        odds_df = pd.DataFrame({
            "race_id": ["2024032405030208"] * 3,
            "umaban": [1, 2, 3],
            "tan_odds": [2.0, 3.0, 4.0],
            "fuku_odds": [1.1, 1.3, 1.5],
        })
        engine = FeatureEngine()
        result = engine.build_features(sample_race, sample_entries, odds_snapshot=odds_df)
        assert "tan_odds" in result.columns
        assert result["tan_odds"].tolist() == [2.0, 3.0, 4.0]
```

### Step 6: テストを実行して成功を確認

Run: `python -m pytest tests/test_feature_engine.py -v`
Expected: ALL PASS (10 tests)

### Step 7: 全テストが通ることを確認

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

### Step 8: Commit

```bash
git add src/features/__init__.py src/features/feature_engine.py tests/test_feature_engine.py
git commit -m "feat: FeatureEngine メインオーケストレータ (B-1)

build_all() でバッチ学習用特徴量生成、build_features() で推論用特徴量生成。
race_df/entry_df/odds_df のマージ、基本列名マッピング、障害レース除外を実装。"
```

---

## Task B-2: レース内相対特徴量 (intra_race_features.py)

**Files:**
- Create: `src/features/intra_race_features.py`
- Create: `tests/test_intra_race_features.py`

**Dependencies:** B-1 (feature_engine.py)

レース内の各馬の相対的な位置づけを表す特徴量を計算する。
- `popularity_rank`: 人気順位 (ninki の別名、feature_engineでマッピング済み)
- `weight_diff_from_mean`: 馬体重とレース平均との差（※将来のモデル拡張用。現行 FEATURE_COLS には含まれない）
- `odds_rank`: 単勝オッズのレース内順位

### Step 1: 失敗するテストを書く

```python
# tests/test_intra_race_features.py
"""src/features/intra_race_features.py のテスト"""

import pandas as pd
import pytest

from features.intra_race_features import compute_intra_race_features


@pytest.fixture
def merged_df() -> pd.DataFrame:
    """feature_engine.build_all() 出力を模擬したマージ済みDataFrame"""
    return pd.DataFrame({
        "race_id": ["2024032405030208"] * 5,
        "umaban": [1, 2, 3, 4, 5],
        "win_odds": [2.0, 5.0, 3.0, 10.0, 8.0],
        "ninki": [1, 3, 2, 5, 4],
        "ba_taijyu": [480.0, 470.0, 490.0, 460.0, 500.0],
        "popularity_rank": [1, 3, 2, 5, 4],
    })


@pytest.fixture
def multi_race_df() -> pd.DataFrame:
    """複数レースを含むDataFrame（グループ処理の確認用）"""
    return pd.DataFrame({
        "race_id": ["R1"] * 3 + ["R2"] * 4,
        "umaban": [1, 2, 3, 1, 2, 3, 4],
        "win_odds": [2.0, 5.0, 8.0, 3.0, 4.0, 10.0, 15.0],
        "ninki": [1, 2, 3, 1, 2, 3, 4],
        "ba_taijyu": [480.0, 470.0, 490.0, 485.0, 475.0, 465.0, 495.0],
        "popularity_rank": [1, 2, 3, 1, 2, 3, 4],
    })


class TestIntraRaceFeatures:
    def test_weight_diff_from_mean(self, merged_df: pd.DataFrame):
        """馬体重とレース平均との差を計算"""
        result = compute_intra_race_features(merged_df)
        mean_weight = (480 + 470 + 490 + 460 + 500) / 5  # 480.0
        expected = [0.0, -10.0, 10.0, -20.0, 20.0]
        for i, exp in enumerate(expected):
            assert abs(result.iloc[i]["weight_diff_from_mean"] - exp) < 1e-10

    def test_weight_diff_from_mean_multi_race(self, multi_race_df: pd.DataFrame):
        """複数レースでそれぞれ独立に平均を計算"""
        result = compute_intra_race_features(multi_race_df)
        r1_rows = result[result["race_id"] == "R1"]
        r2_rows = result[result["race_id"] == "R2"]
        # R1: mean = (480+470+490)/3 = 480.0
        assert abs(r1_rows.iloc[0]["weight_diff_from_mean"] - 0.0) < 1e-10
        assert abs(r1_rows.iloc[1]["weight_diff_from_mean"] - (-10.0)) < 1e-10
        # R2: mean = (485+475+465+495)/4 = 480.0
        assert abs(r2_rows.iloc[0]["weight_diff_from_mean"] - 5.0) < 1e-10

    def test_odds_rank(self, merged_df: pd.DataFrame):
        """単勝オッズのレース内順位（低い順=1位）"""
        result = compute_intra_race_features(merged_df)
        # win_odds: 2.0(rank1), 5.0(rank3), 3.0(rank2), 10.0(rank5), 8.0(rank4)
        odds_rank_map = {1: 1, 2: 3, 3: 2, 4: 5, 5: 4}
        for _, row in result.iterrows():
            umaban = int(row["umaban"])
            assert row["odds_rank"] == odds_rank_map[umaban]

    def test_odds_rank_multi_race(self, multi_race_df: pd.DataFrame):
        """複数レースでそれぞれ独立に順位を計算"""
        result = compute_intra_race_features(multi_race_df)
        r1_rows = result[result["race_id"] == "R1"]
        # R1 odds: 2.0(1), 5.0(2), 8.0(3)
        assert r1_rows.iloc[0]["odds_rank"] == 1
        assert r1_rows.iloc[1]["odds_rank"] == 2
        assert r1_rows.iloc[2]["odds_rank"] == 3

    def test_preserves_existing_columns(self, merged_df: pd.DataFrame):
        """既存列を保持する"""
        result = compute_intra_race_features(merged_df)
        assert "race_id" in result.columns
        assert "umaban" in result.columns
        assert "win_odds" in result.columns

    def test_returns_new_dataframe(self, merged_df: pd.DataFrame):
        """入力DataFrameを変更しない"""
        original_cols = set(merged_df.columns)
        _ = compute_intra_race_features(merged_df)
        assert set(merged_df.columns) == original_cols
```

### Step 2: テストを実行して失敗を確認

Run: `python -m pytest tests/test_intra_race_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.intra_race_features'`

### Step 3: 実装

```python
# src/features/intra_race_features.py
"""カテゴリB: レース内相対特徴量

各馬のレース内での相対的な位置づけを表す特徴量を計算する。
"""

from __future__ import annotations

import pandas as pd


def compute_intra_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内相対特徴量を計算

    Args:
        df: race_id, umaban, win_odds, ba_taijyu を含むDataFrame

    Returns:
        weight_diff_from_mean, odds_rank 列が追加されたDataFrame
    """
    df = df.copy()

    if "ba_taijyu" in df.columns:
        weight_mean = df.groupby("race_id")["ba_taijyu"].transform("mean")
        df["weight_diff_from_mean"] = df["ba_taijyu"] - weight_mean

    if "win_odds" in df.columns:
        df["odds_rank"] = df.groupby("race_id")["win_odds"].rank(
            method="min", ascending=True
        ).astype(int)

    return df
```

### Step 4: テストを実行して成功を確認

Run: `python -m pytest tests/test_intra_race_features.py -v`
Expected: ALL PASS (6 tests)

### Step 5: feature_engine.py に統合

```python
# src/features/feature_engine.py の build_all() 内、
# "# 5. サブモジュールの特徴量計算" のコメントを以下に置換:

        # 5. サブモジュールの特徴量計算
        from features.intra_race_features import compute_intra_race_features

        df = compute_intra_race_features(df)
```

### Step 6: 全テストを実行

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

### Step 7: Commit

```bash
git add src/features/intra_race_features.py tests/test_intra_race_features.py src/features/feature_engine.py
git commit -m "feat: レース内相対特徴量モジュール (B-2)

weight_diff_from_mean (馬体重と平均の差), odds_rank (オッズ順位) を計算。
feature_engine.build_all() に統合済み。"
```

---

## Task B-3: オッズ変化率特徴量 (odds_dynamics_features.py)

**Files:**
- Create: `src/features/odds_dynamics_features.py`
- Create: `tests/test_odds_dynamics_features.py`

**Dependencies:** B-1 (feature_engine.py)

オッズ時系列データから変化率・速度・ボラティリティ・人気変化を計算する。
時系列データが無い場合は NaN で埋める。

**設計書 FEATURE_COLS 対応 (§2.3 WinTwoStageModel):**
- `odds_drop_rate_60_10`: t-60min → t-10min のオッズ変化率
- `odds_drop_rate_30_10`: t-30min → t-10min のオッズ変化率
- `odds_velocity`: 単位時間あたりのオッズ変化量
- `odds_volatility`: 連続するオッズ変化量の標準偏差
- `popularity_change_30_10`: t-30min → t-10min の人気順位変化

### Step 1: 失敗するテストを書く

```python
# tests/test_odds_dynamics_features.py
"""src/features/odds_dynamics_features.py のテスト"""

import numpy as np
import pandas as pd
import pytest

from features.odds_dynamics_features import compute_odds_dynamics


@pytest.fixture
def odds_ts_df() -> pd.DataFrame:
    """3頭のオッズ時系列データ（6時点: t-60, t-50, t-40, t-30, t-20, t-10）

    umaban=1: オッズ上昇（人気が落ちる） 3.0 → 5.5
    umaban=2: オッズ下降（人気が上がる） 10.0 → 5.0
    umaban=3: 安定 5.0 → 5.0
    """
    times = ["03241000", "03241010", "03241020", "03241030", "03241040", "03241050"]
    # index 0=t-60min相当, index 3=t-30min相当, index 5=t-10min相当
    data = []
    for t_idx, time in enumerate(times):
        data.append({"race_id": "R1", "happyo_time": time, "umaban": 1,
                      "tan_odds": 3.0 + t_idx * 0.5, "fuku_odds": 1.5,
                      "ninki": 1 + t_idx})  # 1→6 (人気が落ちる)
        data.append({"race_id": "R1", "happyo_time": time, "umaban": 2,
                      "tan_odds": 10.0 - t_idx * 1.0, "fuku_odds": 3.0,
                      "ninki": 6 - t_idx})  # 6→1 (人気が上がる)
        data.append({"race_id": "R1", "happyo_time": time, "umaban": 3,
                      "tan_odds": 5.0, "fuku_odds": 2.0,
                      "ninki": 3})  # 安定
    return pd.DataFrame(data)


@pytest.fixture
def base_df() -> pd.DataFrame:
    """feature_engine出力を模擬したベースDataFrame"""
    return pd.DataFrame({
        "race_id": ["R1"] * 3,
        "umaban": [1, 2, 3],
    })


class TestOddsDynamicsFeatures:
    def test_odds_drop_rate_60_10(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """t-60→t-10 のオッズ変化率を正しく計算"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        # umaban=1: idx0=3.0, idx5=5.5 → (3.0-5.5)/3.0 = -0.833
        assert abs(result.iloc[0]["odds_drop_rate_60_10"] - (-2.5 / 3.0)) < 1e-10
        # umaban=2: idx0=10.0, idx5=5.0 → (10.0-5.0)/10.0 = 0.5
        assert abs(result.iloc[1]["odds_drop_rate_60_10"] - 0.5) < 1e-10
        # umaban=3: 変化なし → 0.0
        assert abs(result.iloc[2]["odds_drop_rate_60_10"] - 0.0) < 1e-10

    def test_odds_drop_rate_30_10(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """t-30→t-10 のオッズ変化率を正しく計算"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        # umaban=1: idx3=4.5, idx5=5.5 → (4.5-5.5)/4.5 = -0.222
        assert abs(result.iloc[0]["odds_drop_rate_30_10"] - (-1.0 / 4.5)) < 1e-10
        # umaban=2: idx3=7.0, idx5=5.0 → (7.0-5.0)/7.0 = 0.286
        assert abs(result.iloc[1]["odds_drop_rate_30_10"] - (2.0 / 7.0)) < 1e-10

    def test_odds_velocity(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """オッズ変化速度（線形回帰の傾き）を計算"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        # umaban=1: 傾き=0.5 (等間隔で+0.5ずつ)
        assert abs(result.iloc[0]["odds_velocity"] - 0.5) < 1e-10
        # umaban=2: 傾き=-1.0
        assert abs(result.iloc[1]["odds_velocity"] - (-1.0)) < 1e-10
        # umaban=3: 傾き=0.0
        assert abs(result.iloc[2]["odds_velocity"] - 0.0) < 1e-10

    def test_odds_volatility(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """オッズボラティリティ（変化量の標準偏差）を計算"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        # umaban=1: changes=[0.5, 0.5, 0.5, 0.5, 0.5] → std=0.0
        assert abs(result.iloc[0]["odds_volatility"]) < 1e-10
        # umaban=2: changes=[-1.0, -1.0, -1.0, -1.0, -1.0] → std=0.0
        assert abs(result.iloc[1]["odds_volatility"]) < 1e-10

    def test_popularity_change_30_10(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """t-30→t-10 の人気順位変化を計算"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        # umaban=1: ninki at idx3=4, idx5=6 → change = 4-6 = -2 (人気が落ちた)
        assert result.iloc[0]["popularity_change_30_10"] == -2
        # umaban=2: ninki at idx3=3, idx5=1 → change = 3-1 = 2 (人気が上がった)
        assert result.iloc[1]["popularity_change_30_10"] == 2
        # umaban=3: ninki 安定 → change = 0
        assert result.iloc[2]["popularity_change_30_10"] == 0

    def test_no_time_series_returns_nan(self, base_df: pd.DataFrame):
        """時系列データがない場合は NaN を返す"""
        result = compute_odds_dynamics(base_df, pd.DataFrame())
        for col in ["odds_drop_rate_60_10", "odds_drop_rate_30_10",
                     "odds_velocity", "odds_volatility", "popularity_change_30_10"]:
            assert result[col].isna().all(), f"{col} should be NaN"

    def test_none_time_series_returns_nan(self, base_df: pd.DataFrame):
        """時系列データがNoneの場合は NaN を返す"""
        result = compute_odds_dynamics(base_df, None)
        assert result["odds_drop_rate_60_10"].isna().all()

    def test_missing_horse_in_ts(self, base_df: pd.DataFrame):
        """時系列データに存在しない馬は NaN"""
        odds_ts = pd.DataFrame({
            "race_id": ["R1", "R1", "R1", "R1", "R1", "R1"],
            "happyo_time": ["03241000", "03241010", "03241020",
                            "03241030", "03241040", "03241050"],
            "umaban": [1, 1, 1, 1, 1, 1],
            "tan_odds": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            "fuku_odds": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            "ninki": [1, 2, 3, 4, 5, 6],
        })
        result = compute_odds_dynamics(base_df, odds_ts)
        assert not np.isnan(result.iloc[0]["odds_drop_rate_60_10"])
        assert np.isnan(result.iloc[1]["odds_drop_rate_60_10"])
        assert np.isnan(result.iloc[2]["odds_drop_rate_60_10"])

    def test_preserves_existing_columns(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """既存列を保持する"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        assert "race_id" in result.columns
        assert "umaban" in result.columns
```

### Step 2: テストを実行して失敗を確認

Run: `python -m pytest tests/test_odds_dynamics_features.py -v`
Expected: FAIL — `ModuleNotFoundError`

### Step 3: 実装

```python
# src/features/odds_dynamics_features.py
"""カテゴリC: オッズ変化率特徴量

オッズ時系列データから以下を計算 (設計書 §2.3 WinTwoStageModel.FEATURE_COLS):
- odds_drop_rate_60_10: t-60min → t-10min のオッズ変化率
- odds_drop_rate_30_10: t-30min → t-10min のオッズ変化率
- odds_velocity: 単位時間あたりのオッズ変化量（線形回帰の傾き）
- odds_volatility: 連続するオッズ変化量の標準偏差
- popularity_change_30_10: t-30min → t-10min の人気順位変化

計算方式:
- 時系列DataFrameを happyo_time でソート後、先頭を t-60min、末尾を t-10min とみなす
- 中間地点を t-30min として 30-10 変化率を計算
- popularity_change は時系列の ninki 列が必要
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_odds_dynamics(
    df: pd.DataFrame,
    odds_ts: pd.DataFrame | None,
) -> pd.DataFrame:
    """オッズ変化率特徴量を計算

    Args:
        df: race_id, umaban を含むベースDataFrame
        odds_ts: race_id, happyo_time, umaban, tan_odds を含む時系列DataFrame
                 ninki 列がある場合は popularity_change も計算

    Returns:
        odds_drop_rate_60_10, odds_drop_rate_30_10, odds_velocity,
        odds_volatility, popularity_change_30_10 列が追加されたDataFrame
    """
    df = df.copy()

    nan_cols = [
        "odds_drop_rate_60_10", "odds_drop_rate_30_10",
        "odds_velocity", "odds_volatility", "popularity_change_30_10",
    ]

    if odds_ts is None or odds_ts.empty:
        for col in nan_cols:
            df[col] = np.nan
        return df

    ts = odds_ts.sort_values(["race_id", "umaban", "happyo_time"]).copy()

    grouped = ts.groupby(["race_id", "umaban"])

    # --- 変化率: (early_odds - late_odds) / early_odds ---
    first_odds = grouped["tan_odds"].first()
    last_odds = grouped["tan_odds"].last()

    # 60→10: 先頭(=t-60) → 末尾(=t-10)
    df["odds_drop_rate_60_10"] = (
        (first_odds - last_odds) / first_odds.replace(0, np.nan)
    )

    # 30→10: 中間(=t-30) → 末尾(=t-10)
    def _get_mid_odds(group: pd.DataFrame) -> float:
        n = len(group)
        if n < 3:
            return np.nan
        mid_idx = n // 2
        return float(group["tan_odds"].iloc[mid_idx])

    mid_odds = grouped.apply(_get_mid_odds)
    df["odds_drop_rate_30_10"] = (
        (mid_odds - last_odds) / mid_odds.replace(0, np.nan)
    )

    # --- 速度: 線形回帰の傾き ---
    def _calc_velocity(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return np.nan
        x = np.arange(len(group), dtype=float)
        y = group["tan_odds"].values.astype(float)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    df["odds_velocity"] = grouped.apply(_calc_velocity)

    # --- ボラティリティ: 連続変化量の標準偏差 ---
    def _calc_volatility(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return np.nan
        changes = group["tan_odds"].diff().dropna()
        return float(changes.std()) if len(changes) > 0 else np.nan

    df["odds_volatility"] = grouped.apply(_calc_volatility)

    # --- 人気変化: t-30 → t-10 ---
    if "ninki" in ts.columns:
        first_ninki = grouped["ninki"].first()
        last_ninki = grouped["ninki"].last()

        def _get_mid_ninki(group: pd.DataFrame) -> float:
            n = len(group)
            if n < 3:
                return np.nan
            mid_idx = n // 2
            return float(group["ninki"].iloc[mid_idx])

        mid_ninki = grouped.apply(_get_mid_ninki)
        # 正値 = 人気が上がった（順位が小さくなった）
        df["popularity_change_30_10"] = mid_ninki - last_ninki
    else:
        df["popularity_change_30_10"] = np.nan

    # groupby の結果を df にマージ (grouped の index = (race_id, umaban))
    # 上記で既に df に列を追加済みなので、groupby 結果を index でマージ
    for col in nan_cols:
        if col in df.columns:
            # df に既に groupby apply の結果 (Series with MultiIndex) が
            # 直接代入されている場合、race_id/umaban で merge して上書き
            series = df[col]
            if hasattr(series, "index") and isinstance(series.index, pd.MultiIndex):
                df[col] = series.reset_index()  # fallback safety

    return df
```

### Step 4: テストを実行して成功を確認

Run: `python -m pytest tests/test_odds_dynamics_features.py -v`
Expected: ALL PASS (9 tests)

### Step 5: feature_engine.py に統合

```python
# src/features/feature_engine.py の build_all() 内に追加:

        from features.odds_dynamics_features import compute_odds_dynamics

        df = compute_odds_dynamics(df, odds_ts_df)
```

### Step 6: 全テストを実行

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

### Step 7: Commit

```bash
git add src/features/odds_dynamics_features.py tests/test_odds_dynamics_features.py src/features/feature_engine.py
git commit -m "feat: オッズ変化率特徴量モジュール (B-3)

odds_drop_rate_60_10, odds_drop_rate_30_10 (時間窓別変化率),
odds_velocity (速度), odds_volatility (ボラティリティ),
popularity_change_30_10 (人気変化) を計算。
設計書 §2.3 WinTwoStageModel.FEATURE_COLS に完全対応。"
```

---

## Task B-4: 市場歪み特徴量 (market_bias_features.py)

**Files:**
- Create: `src/features/market_bias_features.py`
- Create: `tests/test_market_bias_features.py`

**Dependencies:** B-1 (feature_engine.py)

市場の歪み度合いを表す特徴量を計算する。
- `p_market_win_adj`: 正規化された市場確率 (sum=1)
- `market_entropy`: シャノンエントロピー（拮抗度）
- `overround`: 胴元控除率 (sum(p_raw) - 1)

### Step 1: 失敗するテストを書く

```python
# tests/test_market_bias_features.py
"""src/features/market_bias_features.py のテスト"""

import math

import numpy as np
import pandas as pd
import pytest

from features.market_bias_features import compute_market_bias


@pytest.fixture
def simple_odds_df() -> pd.DataFrame:
    """単純なオッズDataFrame（全馬均等オッズ）"""
    return pd.DataFrame({
        "race_id": ["R1"] * 4,
        "umaban": [1, 2, 3, 4],
        "tan_odds": [4.0, 4.0, 4.0, 4.0],
    })


@pytest.fixture
def skewed_odds_df() -> pd.DataFrame:
    """歪んだオッズDataFrame（人気馬+穴馬）"""
    return pd.DataFrame({
        "race_id": ["R1"] * 3,
        "umaban": [1, 2, 3],
        "tan_odds": [2.0, 5.0, 10.0],
    })


@pytest.fixture
def multi_race_df() -> pd.DataFrame:
    """複数レース"""
    return pd.DataFrame({
        "race_id": ["R1", "R1", "R2", "R2"],
        "umaban": [1, 2, 1, 2],
        "tan_odds": [2.0, 2.0, 3.0, 3.0],
    })


class TestMarketBiasFeatures:
    def test_p_market_win_adj_sums_to_one(self, simple_odds_df: pd.DataFrame):
        """正規化確率の合計が1になる"""
        result = compute_market_bias(simple_odds_df)
        p_sum = result.groupby("race_id")["p_market_win_adj"].sum()
        assert abs(p_sum.iloc[0] - 1.0) < 1e-10

    def test_p_market_win_adj_equal_odds(self, simple_odds_df: pd.DataFrame):
        """均等オッズ(4頭@4.0)の場合、各馬の確率は0.25"""
        result = compute_market_bias(simple_odds_df)
        for _, row in result.iterrows():
            assert abs(row["p_market_win_adj"] - 0.25) < 1e-10

    def test_overround(self, simple_odds_df: pd.DataFrame):
        """均等オッズの overround = 0 (sum(1/odds)=1.0)"""
        result = compute_market_bias(simple_odds_df)
        # 4頭@4.0 → sum(1/4.0) = 1.0 → overround = 0.0
        assert abs(result.iloc[0]["overround"]) < 1e-10

    def test_overround_skewed(self, skewed_odds_df: pd.DataFrame):
        """歪んだオッズの overround > 0"""
        result = compute_market_bias(skewed_odds_df)
        # sum(1/2 + 1/5 + 1/10) = 0.5 + 0.2 + 0.1 = 0.8 → overround = -0.2
        # (このオッズ構成では overround が負 = 馬券売上が控除を下回る非現実的なケース)
        # 実データでは通常 overround > 0
        expected_overround = 0.5 + 0.2 + 0.1 - 1.0
        assert abs(result.iloc[0]["overround"] - expected_overround) < 1e-10

    def test_market_entropy_equal(self, simple_odds_df: pd.DataFrame):
        """均等確率のエントロピーは最大 (= ln(n))"""
        result = compute_market_bias(simple_odds_df)
        max_entropy = math.log(4)
        assert abs(result.iloc[0]["market_entropy"] - max_entropy) < 1e-10

    def test_market_entropy_skewed(self, skewed_odds_df: pd.DataFrame):
        """歪んだ確率のエントロピーは最大より小さい"""
        result = compute_market_bias(skewed_odds_df)
        max_entropy = math.log(3)
        assert result.iloc[0]["market_entropy"] < max_entropy

    def test_multi_race_independent(self, multi_race_df: pd.DataFrame):
        """複数レースで独立に計算される"""
        result = compute_market_bias(multi_race_df)
        for rid in ["R1", "R2"]:
            p_sum = result[result["race_id"] == rid]["p_market_win_adj"].sum()
            assert abs(p_sum - 1.0) < 1e-10

    def test_market_entropy_formula(self, skewed_odds_df: pd.DataFrame):
        """エントロピー公式: H = -sum(p_i * ln(p_i))"""
        result = compute_market_bias(skewed_odds_df)
        p_values = result["p_market_win_adj"].values
        expected = -sum(p * math.log(p) for p in p_values)
        assert abs(result.iloc[0]["market_entropy"] - expected) < 1e-10

    def test_preserves_existing_columns(self, skewed_odds_df: pd.DataFrame):
        """既存列を保持する"""
        result = compute_market_bias(skewed_odds_df)
        assert "race_id" in result.columns
        assert "umaban" in result.columns
        assert "tan_odds" in result.columns
```

### Step 2: テストを実行して失敗を確認

Run: `python -m pytest tests/test_market_bias_features.py -v`
Expected: FAIL — `ModuleNotFoundError`

### Step 3: 実装

```python
# src/features/market_bias_features.py
"""カテゴリD: 市場歪み特徴量

市場の効率性・歪み度合いを表す特徴量を計算:
- p_market_win_adj: 正規化市場確率 (Σ=1)
- market_entropy: シャノンエントロピー (拮抗度の指標)
- overround: 胴元控除率 (Σ(p_raw) - 1)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_market_bias(df: pd.DataFrame) -> pd.DataFrame:
    """市場歪み特徴量を計算

    Args:
        df: race_id, umaban, tan_odds を含むDataFrame

    Returns:
        p_market_win_adj, market_entropy, overround 列が追加されたDataFrame
    """
    df = df.copy()

    if "tan_odds" not in df.columns:
        df["p_market_win_adj"] = np.nan
        df["market_entropy"] = np.nan
        df["overround"] = np.nan
        return df

    # 生の含み確率
    p_raw = 1.0 / df["tan_odds"].replace(0, np.nan)

    # Overround: 胴元控除率 (正=控除あり, 負=非現実的)
    overround = p_raw.groupby(df["race_id"]).transform("sum") - 1.0
    df["overround"] = overround

    # 正規化確率 (Σ=1)
    p_sum = p_raw.groupby(df["race_id"]).transform("sum")
    df["p_market_win_adj"] = p_raw / p_sum.replace(0, np.nan)

    # シャノンエントロピー: H = -Σ(p_i * ln(p_i))
    def _calc_entropy(group: pd.Series) -> float:
        p = group.values.astype(float)
        p = p[p > 0]  # log(0) を回避
        if len(p) == 0:
            return 0.0
        return float(-np.sum(p * np.log(p)))

    entropy = df.groupby("race_id")["p_market_win_adj"].transform(_calc_entropy)
    df["market_entropy"] = entropy

    return df
```

### Step 4: テストを実行して成功を確認

Run: `python -m pytest tests/test_market_bias_features.py -v`
Expected: ALL PASS (9 tests)

### Step 5: feature_engine.py に統合

```python
# src/features/feature_engine.py の build_all() 内に追加:

        from features.market_bias_features import compute_market_bias

        df = compute_market_bias(df)
```

### Step 6: 全テストを実行

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

### Step 7: Commit

```bash
git add src/features/market_bias_features.py tests/test_market_bias_features.py src/features/feature_engine.py
git commit -m "feat: 市場歪み特徴量モジュール (B-4)

p_market_win_adj (正規化確率), market_entropy (シャノンエントロピー),
overround (胴元控除率) を計算。"
```

---

## Task B-5: 情報非対称性特徴量 + レース難易度モデル

**Files:**
- Create: `src/features/race_difficulty_model.py`
- Create: `tests/test_race_difficulty.py`
- Create: `src/features/info_asymmetry_features.py`
- Create: `tests/test_info_asymmetry_features.py`

**Dependencies:** B-1 (feature_engine.py), B-4 (market_bias — entropy使用)

**重要: `info_asymmetry_features.py` はレースレベルユーティリティ**

`compute_hist_features()` は **レースレベルのDataFrame** (1行=1レース) で動作する。
`FeatureEngine.build_all()` は **馬レベルのDataFrame** (1行=1馬) を返すため、
`build_all()` 内では `compute_hist_features()` を呼ばない。

呼び出し元は `TrainingPipelineV5._build_race_level_features()` で、
per-horse DataFrame → per-race 集約 → `compute_hist_features()` の順に処理する (Phase E)。

### 5a: race_difficulty_model.py

レース難易度スコアを計算。頭数・グレード・拮抗度から総合的に評価。
`build_all()` では per-horse に計算されるが、レース内では同値（レース属性のみに依存）。

#### Step 1: 失敗するテストを書く

```python
# tests/test_race_difficulty.py
"""src/features/race_difficulty_model.py のテスト"""

import pandas as pd
import pytest

from features.race_difficulty_model import compute_difficulty_score


@pytest.fixture
def race_df() -> pd.DataFrame:
    """様々な条件のレースデータ"""
    return pd.DataFrame({
        "race_id": ["G1", "G3", "GENERAL", "BIG_FIELD"],
        "field_size": [18, 16, 14, 18],
        "grade_cd": ["A", "C", "_", "_"],
        "market_entropy": [2.8, 2.5, 1.5, 2.89],  # ln(18)≈2.89 が最大
    })


class TestDifficultyScore:
    def test_g1_harder_than_general(self, race_df: pd.DataFrame):
        """G1レースの難易度が一般レースより高い"""
        result = compute_difficulty_score(race_df)
        g1_score = result[result["race_id"] == "G1"]["difficulty_score"].iloc[0]
        gen_score = result[result["race_id"] == "GENERAL"]["difficulty_score"].iloc[0]
        assert g1_score > gen_score

    def test_big_field_harder(self, race_df: pd.DataFrame):
        """大頭数レースの方が難易度が高い（同グレード・同entropyの場合）"""
        result = compute_difficulty_score(race_df)
        big_score = result[result["race_id"] == "BIG_FIELD"]["difficulty_score"].iloc[0]
        gen_score = result[result["race_id"] == "GENERAL"]["difficulty_score"].iloc[0]
        assert big_score > gen_score

    def test_high_entropy_harder(self, race_df: pd.DataFrame):
        """高エントロピ（拮抗）レースの方が難易度が高い"""
        result = compute_difficulty_score(race_df)
        big_score = result[result["race_id"] == "BIG_FIELD"]["difficulty_score"].iloc[0]
        g3_score = result[result["race_id"] == "G3"]["difficulty_score"].iloc[0]
        assert big_score > g3_score

    def test_score_range(self, race_df: pd.DataFrame):
        """スコアが 0.0〜1.0 の範囲に収まる"""
        result = compute_difficulty_score(race_df)
        scores = result["difficulty_score"]
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_preserves_columns(self, race_df: pd.DataFrame):
        """既存列を保持する"""
        result = compute_difficulty_score(race_df)
        assert "race_id" in result.columns
        assert "field_size" in result.columns
```

#### Step 2: テストを実行して失敗を確認

Run: `python -m pytest tests/test_race_difficulty.py -v`
Expected: FAIL

#### Step 3: 実装

```python
# src/features/race_difficulty_model.py
"""カテゴリE: レース難易度スコア

difficulty_score = grade_weight × field_factor × (1 - entropy_normalized)

- grade_weight: G1=1.0, G2=0.8, G3=0.6, 重賞(D)=0.4, 特別(E)=0.2, 一般(_)=0.1
- field_factor: field_size / 18 (最大18頭で正規化)
- entropy_normalized: market_entropy / ln(field_size) (0〜1、高いほど拮抗)
"""

from __future__ import annotations

import math

import pandas as pd

_GRADE_WEIGHTS: dict[str, float] = {
    "A": 1.0,   # G1
    "B": 0.8,   # G2
    "C": 0.6,   # G3
    "D": 0.4,   # 重賞
    "E": 0.2,   # 特別
    "_": 0.1,   # 一般
}

_MAX_FIELD_SIZE = 18


def compute_difficulty_score(df: pd.DataFrame) -> pd.DataFrame:
    """レース難易度スコアを計算

    Args:
        df: race_id, field_size, grade_cd, market_entropy を含むDataFrame

    Returns:
        difficulty_score 列が追加されたDataFrame (0.0〜1.0)
    """
    df = df.copy()

    # グレード重み
    df["_grade_weight"] = df["grade_cd"].map(_GRADE_WEIGHTS).fillna(0.1)

    # 頭数係数 (正規化)
    df["_field_factor"] = (df["field_size"] / _MAX_FIELD_SIZE).clip(upper=1.0)

    # エントロピ正規化 (0〜1、高いほど拮抗)
    max_entropy = df["field_size"].apply(lambda n: math.log(n) if n > 1 else 1.0)
    df["_entropy_norm"] = (df["market_entropy"] / max_entropy.replace(0, 1.0)).clip(0, 1)

    # 難易度スコア: 高グレード × 大頭数 × 高拮抗 = 高難易度
    df["difficulty_score"] = (
        df["_grade_weight"] * df["_field_factor"] * df["_entropy_norm"]
    ).clip(0, 1)

    # 作業列を削除
    df = df.drop(columns=["_grade_weight", "_field_factor", "_entropy_norm"])

    return df
```

#### Step 4: テストを実行して成功を確認

Run: `python -m pytest tests/test_race_difficulty.py -v`
Expected: ALL PASS (5 tests)

#### Step 5: Commit (race_difficulty_model)

```bash
git add src/features/race_difficulty_model.py tests/test_race_difficulty.py
git commit -m "feat: レース難易度スコアモデル (B-5a)

grade_weight × field_factor × entropy_normalized で 0.0〜1.0 のスコアを計算。"
```

### 5b: info_asymmetry_features.py

履歴特徴量を `expanding().shift(1)` でリークフリーに計算 (Rule 18)。

#### Step 6: 失敗するテストを書く

```python
# tests/test_info_asymmetry_features.py
"""src/features/info_asymmetry_features.py のテスト"""

import pandas as pd
import pytest

from features.info_asymmetry_features import compute_hist_features


@pytest.fixture
def historical_df() -> pd.DataFrame:
    """時系列ソート済みの履歴DataFrame（5レース分）

    race_date でソート済み。expanding().shift(1) により
    各行は自分より前のデータのみから計算される。
    """
    return pd.DataFrame({
        "race_id": ["R1", "R2", "R3", "R4", "R5"],
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03",
                                     "2024-01-04", "2024-01-05"]),
        "surface": ["turf", "turf", "dirt", "turf", "turf"],
        "distance_band": ["mile", "mile", "sprint", "mile", "mile"],
        "market_entropy": [2.5, 2.7, 2.0, 2.6, 2.8],
        "topk_hit": [1, 0, 1, 1, 0],  # 上位K頭が的中したか
        "topk_roi": [1.5, -0.5, 2.0, 1.2, -0.3],  # 上位K頭のROI
        "positive_return": [True, False, True, True, False],  # 正のリターン
        "is_winner": [1, 0, 0, 1, 0],  # 1着フラグ
    })


class TestHistFeatures:
    def test_first_row_is_nan(self, historical_df: pd.DataFrame):
        """最初の行は履歴データがないため NaN"""
        result = compute_hist_features(historical_df)
        assert pd.isna(result.iloc[0]["hist_hit_rate_topk"])
        assert pd.isna(result.iloc[0]["hist_roi_topk"])
        assert pd.isna(result.iloc[0]["hist_positive_return_ratio"])
        assert pd.isna(result.iloc[0]["hist_win_rate_same_condition"])
        assert pd.isna(result.iloc[0]["hist_market_entropy_avg"])

    def test_second_row_uses_first_only(self, historical_df: pd.DataFrame):
        """2行目は1行目のデータのみから計算（未来情報なし）"""
        result = compute_hist_features(historical_df)
        # hist_hit_rate_topk: R1 の topk_hit=1 のみ → mean=1.0
        assert abs(result.iloc[1]["hist_hit_rate_topk"] - 1.0) < 1e-10
        # hist_roi_topk: R1 の topk_roi=1.5 のみ → mean=1.5
        assert abs(result.iloc[1]["hist_roi_topk"] - 1.5) < 1e-10
        # hist_positive_return_ratio: R1 の positive_return=True → 1.0
        assert abs(result.iloc[1]["hist_positive_return_ratio"] - 1.0) < 1e-10

    def test_third_row_excludes_future(self, historical_df: pd.DataFrame):
        """3行目は1-2行目のデータのみから計算"""
        result = compute_hist_features(historical_df)
        # hist_hit_rate_topk: (1+0)/2 = 0.5
        assert abs(result.iloc[2]["hist_hit_rate_topk"] - 0.5) < 1e-10
        # hist_roi_topk: (1.5 + (-0.5))/2 = 0.5
        assert abs(result.iloc[2]["hist_roi_topk"] - 0.5) < 1e-10

    def test_no_future_leakage(self, historical_df: pd.DataFrame):
        """expanding().shift(1) により未来情報が含まれないことを検証"""
        result = compute_hist_features(historical_df)
        for i in range(len(result)):
            # 手動で i 行目より前のデータから計算した値と一致するか
            past = historical_df.iloc[:i]
            if len(past) == 0:
                assert pd.isna(result.iloc[i]["hist_hit_rate_topk"])
            else:
                expected_hit = past["topk_hit"].mean()
                actual_hit = result.iloc[i]["hist_hit_rate_topk"]
                assert abs(actual_hit - expected_hit) < 1e-10, (
                    f"行{i}: hist_hit_rate_topk に未来情報リークの疑い"
                )

    def test_same_condition_filtering(self, historical_df: pd.DataFrame):
        """同条件（surface + distance_band）で絞り込んで計算"""
        result = compute_hist_features(historical_df)
        # R4 (turf/mile): 同条件は R1, R2 (turf/mile)
        # hist_win_rate_same_condition: R1=1, R2=0 → mean=0.5
        assert abs(result.iloc[3]["hist_win_rate_same_condition"] - 0.5) < 1e-10
        # hist_market_entropy_avg: R1=2.5, R2=2.7 → mean=2.6
        assert abs(result.iloc[3]["hist_market_entropy_avg"] - 2.6) < 1e-10

    def test_different_condition_excluded(self, historical_df: pd.DataFrame):
        """異なる条件のレースは同条件計算に含まれない"""
        result = compute_hist_features(historical_df)
        # R3 (dirt/sprint): 同条件の過去レースなし → NaN
        assert pd.isna(result.iloc[2]["hist_win_rate_same_condition"])
        assert pd.isna(result.iloc[2]["hist_market_entropy_avg"])

    def test_preserves_columns(self, historical_df: pd.DataFrame):
        """既存列を保持する"""
        result = compute_hist_features(historical_df)
        assert "race_id" in result.columns
        assert "race_date" in result.columns
        assert "surface" in result.columns
```

#### Step 7: テストを実行して失敗を確認

Run: `python -m pytest tests/test_info_asymmetry_features.py -v`
Expected: FAIL

#### Step 8: 実装

```python
# src/features/info_asymmetry_features.py
"""カテゴリE: 情報非対称性特徴量（履歴ベース）

expanding().shift(1) で未来情報リークを完全遮断 (Rule 18)。
各行は自分より前のデータのみから履歴統計を計算する。

特徴量:
- hist_hit_rate_topk: 同条件で上位K頭の過去的中率
- hist_roi_topk: 同条件で上位K頭の過去ROI
- hist_positive_return_ratio: 正のリターンだったレースの割合
- hist_win_rate_same_condition: 同条件の過去的中率
- hist_market_entropy_avg: 同条件の過去平均エントロピー
"""

from __future__ import annotations

import pandas as pd


def compute_hist_features(df: pd.DataFrame) -> pd.DataFrame:
    """履歴特徴量を expanding().shift(1) でリークフリーに計算

    **重要: レースレベルDataFrameで使用すること (1行=1レース)。**
    馬レベルDataFrameではレース単位の expanding window が正しく動作しない。
    呼び出し元は TrainingPipelineV5._build_race_level_features() (Phase E)。

    Args:
        df: race_date, surface, distance_band, market_entropy,
            topk_hit, topk_roi, positive_return, is_winner を含むDataFrame
            race_date でソート済みであること (1行=1レース)

    Returns:
        hist_hit_rate_topk, hist_roi_topk, hist_positive_return_ratio,
        hist_win_rate_same_condition, hist_market_entropy_avg 列が追加されたDataFrame
    """
    df = df.copy()

    if "race_date" not in df.columns:
        df["hist_hit_rate_topk"] = float("nan")
        df["hist_roi_topk"] = float("nan")
        df["hist_positive_return_ratio"] = float("nan")
        df["hist_win_rate_same_condition"] = float("nan")
        df["hist_market_entropy_avg"] = float("nan")
        return df

    # 全体の expanding 統計 (shift(1) で未来情報を遮断)
    df["hist_hit_rate_topk"] = (
        df["topk_hit"].expanding().mean().shift(1)
    )
    df["hist_roi_topk"] = (
        df["topk_roi"].expanding().mean().shift(1)
    )
    df["hist_positive_return_ratio"] = (
        df["positive_return"].astype(float)
        .expanding().mean()
        .shift(1)
    )

    # 同条件 (surface + distance_band) の expanding 統計
    df["_condition"] = df["surface"] + "_" + df["distance_band"]
    grouped = df.groupby("_condition")

    df["hist_win_rate_same_condition"] = (
        grouped["is_winner"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    df["hist_market_entropy_avg"] = (
        grouped["market_entropy"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # 作業列を削除
    df = df.drop(columns=["_condition"])

    return df
```

#### Step 9: テストを実行して成功を確認

Run: `python -m pytest tests/test_info_asymmetry_features.py -v`
Expected: ALL PASS (7 tests)

#### Step 10: feature_engine.py に difficulty_score のみ統合

```python
# src/features/feature_engine.py の build_all() 内に追加:

        from features.race_difficulty_model import compute_difficulty_score

        df = compute_difficulty_score(df)
```

**注意: `compute_hist_features()` は `build_all()` からは呼ばない。**
理由: hist特徴量はレースレベル (1行=1レース) で計算されるが、`build_all()` は
馬レベル (1行=1馬) のDataFrameを返す。hist特徴量は
`TrainingPipelineV5._build_race_level_features()` で
per-horse → per-race 集約後に呼ばれる (Phase E)。

#### Step 11: 全テストを実行

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

#### Step 12: Commit

```bash
git add src/features/info_asymmetry_features.py tests/test_info_asymmetry_features.py src/features/race_difficulty_model.py tests/test_race_difficulty.py src/features/feature_engine.py
git commit -m "feat: 情報非対称性特徴量 + レース難易度モデル (B-5)

- race_difficulty_model: grade×field×entropy で難易度スコア計算
- info_asymmetry_features: expanding().shift(1) でリークフリーな履歴統計
  hist_hit_rate_topk, hist_roi_topk, hist_positive_return_ratio,
  hist_win_rate_same_condition, hist_market_entropy_avg"
```

---

## Task B-6: 未来情報リーク検証 (leakage_validators.py)

**Files:**
- Create: `src/features/leakage_validators.py`
- Create: `tests/test_leakage.py`

**Dependencies:** B-1〜B-5 (全モジュール)

expanding 系特徴量に未来情報が含まれていないことを検証するヘルパー。
設計書 §13 のテスト要件に対応。

### Step 1: 失敗するテストを書く

```python
# tests/test_leakage.py
"""src/features/leakage_validators.py のテスト"""

import pandas as pd
import pytest

from features.leakage_validators import validate_no_future_leakage


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """リークなしの正しいDataFrame"""
    return pd.DataFrame({
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "hist_value": [float("nan"), 10.0, 15.0],  # expanding().shift(1) で計算
    })


@pytest.fixture
def leaky_df() -> pd.DataFrame:
    """リークありのDataFrame（3行目に未来の値が混入）"""
    return pd.DataFrame({
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "hist_value": [float("nan"), 10.0, 25.0],  # 25.0 は未来データを含む
    })


@pytest.fixture
def source_df() -> pd.DataFrame:
    """hist_value の計算元データ（正しい値は expanding mean of source_col）"""
    return pd.DataFrame({
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "source_col": [10.0, 20.0, 15.0],
    })


class TestLeakageValidators:
    def test_clean_data_passes(self, clean_df: pd.DataFrame, source_df: pd.DataFrame):
        """リークなしのデータはバリデーションをパス"""
        issues = validate_no_future_leakage(
            clean_df, source_df, hist_cols=["hist_value"],
            source_cols=["source_col"],
        )
        assert issues == []

    def test_leaky_data_detected(self, leaky_df: pd.DataFrame, source_df: pd.DataFrame):
        """リークありのデータは検出される"""
        issues = validate_no_future_leakage(
            leaky_df, source_df, hist_cols=["hist_value"],
            source_cols=["source_col"],
        )
        assert len(issues) > 0
        assert any("hist_value" in issue for issue in issues)

    def test_nan_first_row_is_ok(self, clean_df: pd.DataFrame, source_df: pd.DataFrame):
        """最初の行が NaN でもエラーにならない"""
        issues = validate_no_future_leakage(
            clean_df, source_df, hist_cols=["hist_value"],
            source_cols=["source_col"],
        )
        assert issues == []

    def test_all_nan_column_passes(self):
        """全NaNの列はバリデーションをパス（計算不能 = リークなし）"""
        df = pd.DataFrame({
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "hist_value": [float("nan"), float("nan")],
        })
        src = pd.DataFrame({
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "source_col": [10.0, 20.0],
        })
        issues = validate_no_future_leakage(
            df, src, hist_cols=["hist_value"], source_cols=["source_col"],
        )
        assert issues == []

    def test_multiple_columns(self):
        """複数列を同時にバリデーション"""
        df = pd.DataFrame({
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "hist_a": [float("nan"), 10.0, 999.0],  # リークあり
            "hist_b": [float("nan"), 10.0, 15.0],   # OK
        })
        src = pd.DataFrame({
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "source_a": [10.0, 20.0, 15.0],
            "source_b": [10.0, 20.0, 15.0],
        })
        issues = validate_no_future_leakage(
            df, src,
            hist_cols=["hist_a", "hist_b"],
            source_cols=["source_a", "source_b"],
        )
        assert any("hist_a" in issue for issue in issues)
        assert not any("hist_b" in issue for issue in issues)
```

### Step 2: テストを実行して失敗を確認

Run: `python -m pytest tests/test_leakage.py -v`
Expected: FAIL

### Step 3: 実装

```python
# src/features/leakage_validators.py
"""未来情報リーク検証モジュール

expanding 系特徴量 (hist_*) に未来情報が含まれていないことを検証する。
設計書 Rule 18: hist系特徴量は expanding().shift(1) で未来情報リークを完全遮断。

使い方:
    issues = validate_no_future_leakage(
        df=feat_df,          # 検証対象のDataFrame
        source_df=source_df, # 計算元データのDataFrame
        hist_cols=["hist_hit_rate_topk", ...],
        source_cols=["topk_hit", ...],
    )
    assert issues == [], f"リーク検出: {issues}"
"""

from __future__ import annotations

import pandas as pd


def validate_no_future_leakage(
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    hist_cols: list[str],
    source_cols: list[str],
    date_col: str = "race_date",
    tolerance: float = 1e-10,
) -> list[str]:
    """expanding 系特徴量の未来情報リークを検証

    各行の hist 値が、その行より前の source データのみから計算されているか確認。
    expanding().shift(1) のセマンティクスに準拠。

    Args:
        df: 検証対象DataFrame (race_date + hist_cols を含む)
        source_df: 計算元DataFrame (race_date + source_cols を含む)
        hist_cols: 検証する履歴特徴量列名のリスト
        source_cols: hist_cols に対応する計算元列名のリスト (同じ順序)
        date_col: 日付列名
        tolerance: 浮動小数点誤差の許容範囲

    Returns:
        リークが検出された列のエラーメッセージリスト (空=問題なし)
    """
    issues: list[str] = []

    if len(hist_cols) != len(source_cols):
        issues.append(
            f"hist_cols ({len(hist_cols)}) と source_cols ({len(source_cols)}) の数が不一致"
        )
        return issues

    for hist_col, source_col in zip(hist_cols, source_cols):
        if hist_col not in df.columns or source_col not in source_df.columns:
            continue

        merged = pd.merge(
            df[[date_col, hist_col]],
            source_df[[date_col, source_col]],
            on=date_col,
            how="inner",
        )
        merged = merged.sort_values(date_col).reset_index(drop=True)

        # 全行NaNの場合はスキップ（計算不能 = リークなし）
        if merged[hist_col].isna().all():
            continue

        for i in range(len(merged)):
            actual = merged.iloc[i][hist_col]

            # NaN はスキップ（最初の行など）
            if pd.isna(actual):
                continue

            # i行目より前のデータのみで expanding mean を計算
            past_values = merged.iloc[:i][source_col].dropna()
            if len(past_values) == 0:
                # 過去データがないのに値がある = リーク
                issues.append(
                    f"{hist_col}: 行{i} に値 {actual} があるが過去データが不存在"
                )
                continue

            expected = past_values.mean()

            if abs(actual - expected) > tolerance:
                issues.append(
                    f"{hist_col}: 行{i} に未来情報リークの疑い "
                    f"(actual={actual:.10f}, expected={expected:.10f})"
                )

    return issues
```

### Step 4: テストを実行して成功を確認

Run: `python -m pytest tests/test_leakage.py -v`
Expected: ALL PASS (5 tests)

### Step 5: __init__.py を更新して全モジュールをエクスポート

```python
# src/features/__init__.py
from features.feature_engine import FeatureEngine
from features.intra_race_features import compute_intra_race_features
from features.info_asymmetry_features import compute_hist_features
from features.leakage_validators import validate_no_future_leakage
from features.market_bias_features import compute_market_bias
from features.odds_dynamics_features import compute_odds_dynamics
from features.race_difficulty_model import compute_difficulty_score

__all__ = [
    "FeatureEngine",
    "compute_intra_race_features",
    "compute_hist_features",
    "validate_no_future_leakage",
    "compute_market_bias",
    "compute_odds_dynamics",
    "compute_difficulty_score",
]
```

### Step 6: 全テストを実行

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

### Step 7: リント・型チェック

Run: `ruff check src/features/ tests/test_*.py`
Run: `mypy src/features/`

### Step 8: Commit

```bash
git add src/features/leakage_validators.py tests/test_leakage.py src/features/__init__.py
git commit -m "feat: 未来情報リーク検証モジュール (B-6)

validate_no_future_leakage() で expanding().shift(1) のセマンティクスを検証。
features/__init__.py に全モジュールをエクスポート。"
```

---

## 統合確認

全タスク完了後、以下を実行して全体の動作確認を行う。

### Step 1: 全テスト実行

Run: `python -m pytest tests/ -v --cov=src/features --cov-report=term-missing`
Expected: ALL PASS, カバレッジ 90%+

### Step 2: リント・型チェック

Run: `ruff check src/ tests/ && ruff format --check src/ tests/`
Run: `mypy src/`
Expected: No errors

### Step 3: feature_engine.py の最終版確認

`build_all()` が全サブモジュールを正しく呼び出していることを確認:
1. データマージ (race + entry + odds)
2. 障害レース除外
3. 基本列マッピング (distance_bin, track_condition_code, popularity_rank, surface_key)
4. `compute_intra_race_features()` → weight_diff_from_mean, odds_rank
5. `compute_market_bias()` → p_market_win_adj, market_entropy, overround
6. `compute_odds_dynamics()` → odds_drop_rate_60_10, odds_drop_rate_30_10, odds_velocity, odds_volatility, popularity_change_30_10
7. `compute_difficulty_score()` → difficulty_score

**`compute_hist_features()` は `build_all()` から呼ばない** (Phase E で TrainingPipelineV5 が呼ぶ)

---

## Phase C で計算される特徴量（本Phaseでは対象外）

以下の特徴量は Phase B では計算せず、Phase C 以降で対応する:

### MarketModel 由来 (Phase C-2)
- `signed_log_error_win`: MarketModel の `predict_and_calc_error()` で計算
- `abs_log_error_win`: 同上
- `market_error_rank_in_race`: 同上

### RaceQualityScreener 向けレースレベル集約 (Phase E)
- `market_log_error_max_abs`: レース内の max |log_error| (MarketModel 出力の集約)
- `market_log_error_std`: レース内の log_error 標準偏差
- `market_log_error_top_q75`: 75パーセンタイル
- `n_positive_errors`: log_error > 0 の馬の数
- `top_k_error_sum`: 上位K頭の log_error 合計 (K=3)
- `positive_error_ratio`: 過小評価馬の割合
- `overround_deviation`: 過去平均 overround からの乖離

### RegimeDetector 向け集約 (Phase C-8)
- `market_error_std/mean`: 直近200レースの log_error 統計
- `market_entropy_mean`: 直近200レースの平均エントロピー
- `overround_mean`: 直近200レースの平均胴元控除率
- `favorite_win_rate`: 1番人気の勝率
- `flb_slope`: favorite-longshot bias の傾き
- `odds_volatility_mean`: オッズ変動量の平均
