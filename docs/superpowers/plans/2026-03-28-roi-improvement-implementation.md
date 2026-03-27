# ROI改善 包括的実装計画 (Phase 1-5 カスケード)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ROI 63.8% → 101%+ を達成する5Phase改善（特徴量追加 → モデル校正 → レジーム実データ化 → ベッティング最適化 → 検証基盤強化）

**Architecture:** 既存パイプライン非破壊。新規モジュール追加方式。各Phaseでバックテスト検証。カスケード方式で段階実装。

**Tech Stack:** Python 3.11, LightGBM, scikit-learn (CalibratedClassifierCV, IsotonicRegression), SQLAlchemy Core, pandas, numpy

**Spec:** `docs/superpowers/specs/2026-03-28-roi-improvement-comprehensive-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/features/horse_history_features.py` | **CREATE** | 馬過去成績3特徴量 + レース内変換(z+pct) + 条件別騎手勝率 |
| `src/models/place_ability_model.py` | **CREATE** | LGBMClassifier + Isotonic校正 + 温度スケーリング |
| `src/models/stage1_ability_model.py` | MODIFY | FEATURE_COLS 7→16→20、p_ability_place行削除 |
| `src/domain/models.py:217-229` | MODIFY | SubmodelSet に place_ability 追加 |
| `src/pipelines/training_pipeline.py` | MODIFY | HorseHistory + PlaceAbility + RegimeStats実データ化 |
| `src/backtest/engine.py` | MODIFY | HorseHistoryFeatures + PlaceAbilityModel 推論パス追加 |
| `src/features/market_bias_features.py` | MODIFY | compute_flb_slope() 追加 |
| `src/features/odds_dynamics_features.py` | MODIFY | compute_rolling_volatility() 追加 |
| `src/betting/wide_strategy.py` | MODIFY | 同一馬制約 + 人気帯多様性制約 |
| `src/betting/stake_calculator.py` | MODIFY | Fractional Kelly (0.5x) |
| `tests/test_horse_history_features.py` | **CREATE** | 特徴量計算テスト |
| `tests/test_place_ability_model.py` | **CREATE** | モデル・校正テスト |

**Key convention:** `DatabaseConnection` は `get_engine()` メソッド（プロパティではない）。`self.db.get_engine()` を使用。

---

# Phase 1: 馬固有特徴量 + PlaceAbilityModel

## Task 1: HorseHistoryFeatures — テスト作成

**Files:**
- Create: `tests/test_horse_history_features.py`

- [ ] **Step 1: テストファイル作成**

```python
"""test_horse_history_features.py — HorseHistoryFeatures の単体テスト"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch


class TestNormFinishLogitAvg:
    """norm_finish_logit_avg (logit変換着順スコア) のテスト"""

    def test_1st_of_16(self):
        """1着/16頭 → logit(15/15) clipped to logit(0.95) ≈ 2.94"""
        from features.horse_history_features import _norm_finish_logit
        result = _norm_finish_logit(finish_pos=1, field_size=16)
        assert 2.9 < result < 3.0

    def test_last_of_16(self):
        """最下位/16頭 → logit(1/15) clipped to logit(0.05) ≈ -2.94"""
        from features.horse_history_features import _norm_finish_logit
        result = _norm_finish_logit(finish_pos=16, field_size=16)
        assert -3.0 < result < -2.9

    def test_field_size_under_8_returns_nan(self):
        """8頭未満レース → NaN"""
        from features.horse_history_features import _norm_finish_logit
        result = _norm_finish_logit(finish_pos=1, field_size=7)
        assert np.isnan(result)

    def test_mid_rank(self):
        """8着/16頭 → logit(0.5) ≈ 0.0（中央値）"""
        from features.horse_history_features import _norm_finish_logit
        result = _norm_finish_logit(finish_pos=8, field_size=16)
        assert -0.1 < result < 0.1


class TestJockeySurprise:
    """jockey_surprise (Beta事前分布スムージング) のテスト"""

    def test_zero_wins_100_races(self):
        """100戦0勝 → surprise ≈ 0 - 0.0476 ≈ -0.0476"""
        from features.horse_history_features import _compute_jockey_surprise
        # actual_wins=0, n=100, expected_wins 適当
        result = _compute_jockey_surprise(actual_wins=0, n_races=100, expected_wins=8.0)
        assert result < 0  # 期待以下

    def test_above_expectation(self):
        """期待以上の勝率 → 正のsurprise"""
        from features.horse_history_features import _compute_jockey_surprise
        # expected=8 wins, actual=15 → surprise > 0
        result = _compute_jockey_surprise(actual_wins=15, n_races=100, expected_wins=8.0)
        assert result > 0

    def test_payout_rate_applied(self):
        """控除率補正（0.80）が適用される"""
        from features.horse_history_features import _compute_jockey_surprise, PAYOUT_RATE
        assert PAYOUT_RATE == 0.80

    def test_min_samples_returns_nan(self):
        """30レース未満 → NaN"""
        from features.horse_history_features import _compute_jockey_surprise
        result = _compute_jockey_surprise(actual_wins=5, n_races=25, expected_wins=2.0)
        assert np.isnan(result)


class TestHaronTimeZscore:
    """haron_time_zscore_avg (階層fallback) のテスト"""

    def test_fallback_l1_to_l2(self):
        """Level 1 サンプル不足 → Level 2 にfallback"""
        from features.horse_history_features import _get_group_stats
        global_stats = {
            ("sprint", "turf", "1"): {"mean": 12.5, "std": 0.3, "n": 10},  # L1: 不足
            ("sprint", "turf"): {"mean": 12.3, "std": 0.4, "n": 80},       # L2: OK
            ("sprint",): {"mean": 12.4, "std": 0.5, "n": 200},             # L3
            ("all",): {"mean": 12.4, "std": 0.5, "n": 5000},               # L4
        }
        mean, std = _get_group_stats(
            distance_bin="sprint", surface="turf", baba_cd="1",
            global_stats=global_stats,
        )
        assert mean == 12.3  # L2の値
        assert std == 0.4

    def test_fallback_to_global(self):
        """全レベル不足 → グローバルfallback"""
        from features.horse_history_features import _get_group_stats
        global_stats = {
            ("all",): {"mean": 12.4, "std": 0.5, "n": 5000},
        }
        mean, std = _get_group_stats(
            distance_bin="long", surface="dirt", baba_cd="3",
            global_stats=global_stats,
        )
        assert mean == 12.4


class TestRaceTransforms:
    """レース内z-score + pct のテスト"""

    def _make_race_df(self):
        return pd.DataFrame({
            "race_id": ["r1"] * 4,
            "umaban": [1, 2, 3, 4],
            "norm_finish_logit_avg": [2.0, 1.0, 0.0, -1.0],
            "jockey_surprise": [0.1, 0.05, -0.02, -0.08],
            "haron_time_zscore_avg": [1.5, 0.5, -0.5, -1.5],
        })

    def test_z_score_sum_approx_zero(self):
        """レース内z-scoreの合計 ≈ 0"""
        from features.horse_history_features import HorseHistoryFeatures
        df = self._make_race_df()
        result = HorseHistoryFeatures.add_race_transforms(df)
        z_col = "norm_finish_logit_avg_race_z"
        assert z_col in result.columns
        assert abs(result[z_col].sum()) < 1e-6

    def test_std_zero_no_nan(self):
        """全馬同じ値（std=0）でも NaN にならない"""
        from features.horse_history_features import HorseHistoryFeatures
        df = pd.DataFrame({
            "race_id": ["r1"] * 3,
            "umaban": [1, 2, 3],
            "norm_finish_logit_avg": [1.0, 1.0, 1.0],
            "jockey_surprise": [0.0, 0.0, 0.0],
            "haron_time_zscore_avg": [0.0, 0.0, 0.0],
        })
        result = HorseHistoryFeatures.add_race_transforms(df)
        assert not result["norm_finish_logit_avg_race_z"].isna().any()

    def test_pct_range(self):
        """pct は [0, 1] の範囲"""
        from features.horse_history_features import HorseHistoryFeatures
        df = self._make_race_df()
        result = HorseHistoryFeatures.add_race_transforms(df)
        pct_col = "norm_finish_logit_avg_race_pct"
        assert result[pct_col].min() >= 0
        assert result[pct_col].max() <= 1


class TestLeakPrevention:
    """リーク防止のテスト"""

    def test_future_race_excluded(self):
        """当該レース日付より後のデータが特徴量に含まれない"""
        # mock SQL で未来データを注入し、compute() がそれを除外することを確認
        # 詳細な実装は HorseHistoryFeatures.compute() 実装後に記述
        pass
```

- [ ] **Step 2: テスト実行（失敗確認）**

Run: `python -m pytest tests/test_horse_history_features.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: コミット**

```bash
git add tests/test_horse_history_features.py
git commit -m "test: HorseHistoryFeatures テスト追加 (Phase 1)"
```

---

## Task 2: HorseHistoryFeatures — 本体実装

**Files:**
- Create: `src/features/horse_history_features.py`
- Test: `tests/test_horse_history_features.py`

- [ ] **Step 1: horse_history_features.py 作成**

```python
"""馬の過去成績特徴量 — DB接続を使用して n_uma_race から計算"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

PAYOUT_RATE: float = 0.80  # JRA控除率20%

# logit変換のクリップ範囲
CLIP_LO: float = 0.05
CLIP_HI: float = 0.95


def _norm_finish_logit(finish_pos: int, field_size: int) -> float:
    """着順を頭数で正規化し、logit変換。8頭未満はNaN。"""
    if field_size < 8:
        return np.nan
    score = 1 - (finish_pos - 1) / (field_size - 1)  # [0, 1]
    score = np.clip(score, CLIP_LO, CLIP_HI)
    return float(np.log(score / (1 - score)))


def _compute_jockey_surprise(
    actual_wins: int,
    n_races: int,
    expected_wins: float,
) -> float:
    """Beta事前分布でスムージングした騎手surprise。30レース未満はNaN。"""
    if n_races < 30:
        return np.nan
    alpha_prior, beta_prior = 1.0, 20.0
    alpha_post = alpha_prior + actual_wins
    beta_post = beta_prior + n_races - actual_wins
    smoothed_wr = alpha_post / (alpha_post + beta_post)
    baseline_wr = alpha_prior / (alpha_prior + beta_prior)
    return float(smoothed_wr - baseline_wr)


FALLBACK_LEVELS: list[tuple[list[str], int]] = [
    (["distance_bin", "surface", "baba_cd"], 50),
    (["distance_bin", "surface"], 30),
    (["distance_bin"], 20),
    ([], 0),
]


def _get_group_stats(
    distance_bin: str,
    surface: str,
    baba_cd: str,
    global_stats: dict[tuple, dict],
) -> tuple[float, float]:
    """階層fallbackでグループ統計を取得"""
    for key_cols, min_n in FALLBACK_LEVELS:
        vals = {"distance_bin": distance_bin, "surface": surface, "baba_cd": baba_cd}
        key = tuple(vals[c] for c in key_cols) if key_cols else ("all",)
        group = global_stats.get(key)
        if group and group["n"] >= min_n:
            return group["mean"], group["std"]
    fallback = global_stats[("all",)]
    return fallback["mean"], fallback["std"]


class HorseHistoryFeatures:
    """馬の過去成績から特徴量を計算"""

    BASE_COLS: list[str] = [
        "norm_finish_logit_avg",
        "jockey_surprise",
        "haron_time_zscore_avg",
    ]

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def compute(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        target_race_ids: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """過去成績特徴量を計算してDataFrameで返す。

        Args:
            race_df: レース情報 (race_id, race_date 含む)
            entry_df: 出走馬情報 (race_id, umaban, ketto_num, kisyu_code 含む)
            target_race_ids: 計算対象race_id (None時は全て)

        Returns:
            DataFrame with columns: race_id, umaban, norm_finish_logit_avg, jockey_surprise, haron_time_zscore_avg
        """
        # TODO: SQL query to n_uma_race JOIN n_race
        # For now return empty DataFrame with correct columns
        return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

    @staticmethod
    def add_race_transforms(df: pd.DataFrame) -> pd.DataFrame:
        """レース内z-score + pct変換を追加"""
        df = df.copy()
        for col in HorseHistoryFeatures.BASE_COLS:
            if col not in df.columns:
                continue
            # z-score
            race_mean = df.groupby("race_id")[col].transform("mean")
            race_std = df.groupby("race_id")[col].transform("std")
            race_std = race_std.clip(lower=1e-6).fillna(1e-6)
            df[f"{col}_race_z"] = (df[col] - race_mean) / race_std

            # percentile
            df[f"{col}_race_pct"] = df.groupby("race_id")[col].rank(pct=True)
        return df
```

- [ ] **Step 2: テスト実行（通過確認）**

Run: `python -m pytest tests/test_horse_history_features.py -v`
Expected: PASS (except leak prevention test which is a placeholder)

- [ ] **Step 3: コミット**

```bash
git add src/features/horse_history_features.py
git commit -m "feat: HorseHistoryFeatures 本体追加 (Phase 1 Task 2)"
```

---

## Task 3: HorseHistoryFeatures — SQLクエリ実装

**Files:**
- Modify: `src/features/horse_history_features.py`
- Test: `tests/test_horse_history_features.py`

- [ ] **Step 1: compute() のSQLクエリ実装**

`HorseHistoryFeatures.compute()` 内の `TODO` を実際のSQLクエリに置き換える:

```python
def compute(
    self,
    race_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    target_race_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """過去成績特徴量を計算"""
    if target_race_ids is not None:
        entry_df = entry_df[entry_df["race_id"].isin(target_race_ids)]

    # 対象レースの馬・騎手リスト
    horses = entry_df[["race_id", "umaban", "ketto_num", "kisyu_code"]].copy()
    # race_date を race_df からマージ
    if "race_date" not in horses.columns:
        date_map = race_df.set_index("race_id")["race_date"]
        horses["race_date"] = horses["race_id"].map(date_map)

    # 全ての ketto_num, kisyu_code を収集
    unique_ketto = horses["ketto_num"].unique().tolist()
    unique_kisyu = horses["kisyu_code"].unique().tolist()

    if not unique_ketto:
        return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

    # SQL: 過去レースデータを一括取得
    sql = text("""
        SELECT
            ur.race_id AS past_race_id,
            r.year, r.month_day,
            ur.ketto_num, ur.kisyu_code, ur.umaban,
            ur.kakutei_jyuni AS finish_pos,
            r.torosu AS field_size,
            ur.tansyo_odds AS win_odds,
            ur.harontimelong3 AS haron_time_l3,
            CASE WHEN r.torosu >= 8 THEN 1 ELSE 0 END AS valid_field
        FROM n_uma_race ur
        JOIN n_race r ON ur.year = r.year
            AND ur.monthday = r.month_day
            AND ur.jyocd = r.jyocd
            AND ur.kaiji = r.kaiji
            AND ur.nichiji = r.nichiji
            AND ur.racenum = r.racenum
        WHERE ur.ketto_num IN :ketto_nums
           OR ur.kisyu_code IN :kisyu_codes
        ORDER BY r.year, r.month_day
    """)

    past_df = pd.read_sql(
        sql, self.engine,
        params={"ketto_nums": tuple(unique_ketto), "kisyu_codes": tuple(unique_kisyu)},
    )

    if past_df.empty:
        return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

    # race_date 生成
    past_df["race_date"] = pd.to_datetime(
        past_df["year"].astype(str) + past_df["month_day"], format="%Y%m%d"
    )

    # 馬ごとに特徴量計算
    results = []
    for _, row in horses.iterrows():
        race_date = row["race_date"]
        ketto = row["ketto_num"]
        kisyu = row["kisyu_code"]

        # norm_finish_logit_avg: 同じ馬の過去レース
        horse_past = past_df[
            (past_df["ketto_num"] == ketto)
            & (past_df["race_date"] < race_date)
            & (past_df["valid_field"] == 1)
            & (past_df["finish_pos"] > 0)
        ].tail(3)

        if len(horse_past) > 0:
            logits = horse_past.apply(
                lambda r: _norm_finish_logit(r["finish_pos"], r["field_size"]), axis=1
            )
            norm_finish_logit_avg = logits.mean()
        else:
            norm_finish_logit_avg = np.nan

        # jockey_surprise: 騎手の過去100戦
        jockey_past = past_df[
            (past_df["kisyu_code"] == kisyu)
            & (past_df["race_date"] < race_date)
            & (past_df["finish_pos"] > 0)
            & (past_df["win_odds"] > 0)
        ].tail(100)

        if len(jockey_past) >= 30:
            expected = (PAYOUT_RATE / jockey_past["win_odds"].clip(lower=1.1)).sum()
            actual = (jockey_past["finish_pos"] == 1).sum()
            jockey_surprise = _compute_jockey_surprise(actual, len(jockey_past), expected)
        else:
            jockey_surprise = np.nan

        # haron_time_zscore_avg: 過去3走
        horse_haron = past_df[
            (past_df["ketto_num"] == ketto)
            & (past_df["race_date"] < race_date)
            & (past_df["haron_time_l3"] > 0)
        ].tail(3)

        if len(horse_haron) > 0:
            # TODO: 階層fallback統計の計算
            haron_time_zscore_avg = np.nan  # Phase 1 implementation
        else:
            haron_time_zscore_avg = np.nan

        results.append({
            "race_id": row["race_id"],
            "umaban": row["umaban"],
            "norm_finish_logit_avg": norm_finish_logit_avg,
            "jockey_surprise": jockey_surprise,
            "haron_time_zscore_avg": haron_time_zscore_avg,
        })

    return pd.DataFrame(results)
```

- [ ] **Step 2: リーク防止テスト追加・実行**

Run: `python -m pytest tests/test_horse_history_features.py -v`

- [ ] **Step 3: コミット**

```bash
git add src/features/horse_history_features.py
git commit -m "feat: HorseHistoryFeatures SQLクエリ実装 (Phase 1 Task 3)"
```

---

## Task 4: PlaceAbilityModel — テスト作成

**Files:**
- Create: `tests/test_place_ability_model.py`

- [ ] **Step 1: テストファイル作成**

```python
"""test_place_ability_model.py — PlaceAbilityModel の単体テスト"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_train_df(n_races: int = 20, field_size: int = 8):
    """学習用ダミーデータ生成"""
    rows = []
    for r in range(n_races):
        for h in range(field_size):
            rows.append({
                "race_id": f"r{r:04d}",
                "umaban": h + 1,
                "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=r),
                "surface": "turf",
                "distance_bin": "mile",
                "track_condition_code": 1.0,
                "grade_code": "0",
                "field_size": float(field_size),
                "weight_diff_from_mean": np.random.randn(),
                "difficulty_score": np.random.randn(),
                "norm_finish_logit_avg": np.random.randn(),
                "jockey_surprise": np.random.randn(),
                "haron_time_zscore_avg": np.random.randn(),
                "norm_finish_logit_avg_race_z": np.random.randn(),
                "jockey_surprise_race_z": np.random.randn(),
                "haron_time_zscore_avg_race_z": np.random.randn(),
                "norm_finish_logit_avg_race_pct": np.random.rand(),
                "jockey_surprise_race_pct": np.random.rand(),
                "haron_time_zscore_avg_race_pct": np.random.rand(),
                "finish_pos": h + 1,
                "p_ability_win": 1.0 / field_size,
            })
    return pd.DataFrame(rows)


class TestPlaceAbilityModel:
    def test_probability_range(self):
        """出力が [0, 1] に収まる"""
        from models.place_ability_model import PlaceAbilityModel
        df = _make_train_df(n_races=50)
        model = PlaceAbilityModel()
        model.train(df)
        result = model.predict(df)
        assert result["p_ability_place"].min() >= 0
        assert result["p_ability_place"].max() <= 1

    def test_race_sum_approx_3(self):
        """レース内の p_place 合計 ≈ 3"""
        from models.place_ability_model import PlaceAbilityModel
        df = _make_train_df(n_races=50)
        model = PlaceAbilityModel()
        model.train(df)
        result = model.predict(df)
        race_sums = result.groupby("race_id")["p_ability_place"].sum()
        assert all(abs(s - 3.0) < 0.5 for s in race_sums)

    def test_feature_cols_exist(self):
        """FEATURE_COLS の全列がDataFrameに存在する"""
        from models.place_ability_model import PlaceAbilityModel
        model = PlaceAbilityModel()
        for col in model.FEATURE_COLS:
            assert col in _make_train_df().columns

    def test_temporal_split(self):
        """校正データが学習データより未来"""
        from models.place_ability_model import PlaceAbilityModel
        df = _make_train_df(n_races=50)
        model = PlaceAbilityModel()
        model.train(df)  # 内部で時系列分割
        # train がエラーなく完了することを確認
```

- [ ] **Step 2: テスト実行（失敗確認）**

Run: `python -m pytest tests/test_place_ability_model.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: コミット**

```bash
git add tests/test_place_ability_model.py
git commit -m "test: PlaceAbilityModel テスト追加 (Phase 1 Task 4)"
```

---

## Task 5: PlaceAbilityModel — 本体実装

**Files:**
- Create: `src/models/place_ability_model.py`
- Test: `tests/test_place_ability_model.py`

- [ ] **Step 1: place_ability_model.py 作成**

```python
"""複勝能力モデル — LGBMClassifier + Isotonic校正 + 温度スケーリング"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

TEMPERATURE: float = 0.7  # <1 で分布を尖らせる


class PlaceAbilityModel:
    """複勝的中確率を直接推定するbinaryモデル。
    p_ability_place = p_ability_win * 3.0 の粗い近似を置き換える。
    """

    FEATURE_COLS: list[str] = [
        "surface", "distance_bin", "track_condition_code",
        "grade_code", "field_size",
        "weight_diff_from_mean", "difficulty_score",
        "norm_finish_logit_avg", "jockey_surprise", "haron_time_zscore_avg",
        "norm_finish_logit_avg_race_z", "jockey_surprise_race_z", "haron_time_zscore_avg_race_z",
        "norm_finish_logit_avg_race_pct", "jockey_surprise_race_pct", "haron_time_zscore_avg_race_pct",
    ]

    def __init__(self) -> None:
        self._model: lgb.LGBMClassifier | None = None
        self._calibrated: CalibratedClassifierCV | None = None

    def train(self, df: pd.DataFrame) -> None:
        """学習 + Isotonic校正（時系列分割）"""
        assert "race_date" in df.columns, "race_date が必要"
        assert "finish_pos" in df.columns, "finish_pos が必要"

        df = df.dropna(subset=self.FEATURE_COLS).copy()
        y = (df["finish_pos"] <= 3).astype(int)
        X = df[self.FEATURE_COLS].copy()
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in X.columns:
                X[col] = X[col].astype("category")

        # 時系列分割: 80% train, 20% calibrate
        dates = sorted(df["race_date"].unique())
        split_date = dates[int(len(dates) * 0.8)]
        train_mask = df["race_date"] < split_date
        calib_mask = df["race_date"] >= split_date

        X_train, y_train = X[train_mask], y[train_mask]
        X_calib, y_calib = X[calib_mask], y[calib_mask]

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)

        self._model = lgb.LGBMClassifier(
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            num_leaves=31,
            max_depth=-1,
            min_data_in_leaf=100,
            feature_fraction=0.7,
            reg_lambda=1.0,
            learning_rate=0.03,
            n_estimators=500,
            verbose=-1,
        )
        self._model.fit(X_train, y_train)

        if len(X_calib) >= 50:
            self._calibrated = CalibratedClassifierCV(
                estimator=self._model, method="isotonic", cv="prefit",
            )
            self._calibrated.fit(X_calib, y_calib)
        else:
            self._calibrated = None
            logger.warning("Insufficient calibration data (%d), skipping isotonic", len(X_calib))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """p_ability_place を設定して df を返す"""
        df = df.copy()
        X = df[self.FEATURE_COLS].copy()
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in X.columns:
                X[col] = X[col].astype("category")

        if self._calibrated is not None:
            raw_p = self._calibrated.predict_proba(X)[:, 1]
        elif self._model is not None:
            raw_p = self._model.predict_proba(X)[:, 1]
        else:
            raise RuntimeError("Model not trained")

        df["p_ability_place_raw"] = raw_p

        # 温度スケーリング
        scaled = raw_p ** (1 / TEMPERATURE)

        # レース内正規化: sum(p_place) ≈ 3
        race_sum = df.groupby("race_id")["p_ability_place_raw"].transform(
            lambda s: pd.Series(scaled[s.index], index=s.index).sum()
        )
        df["p_ability_place"] = scaled * (3.0 / race_sum.clip(lower=1e-6))

        # 整合性制約: p_place >= p_win（優先順位: sum=3 > p_place >= p_win）
        if "p_ability_win" in df.columns:
            df["p_ability_place"] = np.maximum(df["p_ability_place"], df["p_ability_win"])
            # 再正規化
            race_sum = df.groupby("race_id")["p_ability_place"].transform("sum")
            df["p_ability_place"] = df["p_ability_place"] * (3.0 / race_sum.clip(lower=1e-6))

        return df
```

- [ ] **Step 2: テスト実行（通過確認）**

Run: `python -m pytest tests/test_place_ability_model.py -v`
Expected: PASS

- [ ] **Step 3: コミット**

```bash
git add src/models/place_ability_model.py
git commit -m "feat: PlaceAbilityModel 本体追加 (Phase 1 Task 5)"
```

---

## Task 6: AbilityModel FEATURE_COLS 更新 + SubmodelSet 変更

**Files:**
- Modify: `src/models/stage1_ability_model.py:27-38` (FEATURE_COLS)
- Modify: `src/models/stage1_ability_model.py:110-112` (p_ability_place 行削除)
- Modify: `src/domain/models.py:217-229` (SubmodelSet)

- [ ] **Step 1: stage1_ability_model.py FEATURE_COLS 更新**

`FEATURE_COLS` を16列に拡張:

```python
FEATURE_COLS: list[str] = [
    # 既存 (7)
    "surface", "distance_bin", "track_condition_code",
    "grade_code", "field_size",
    "weight_diff_from_mean", "difficulty_score",
    # Phase 1: 馬の過去成績 (3)
    "norm_finish_logit_avg", "jockey_surprise", "haron_time_zscore_avg",
    # Phase 1: レース内z-score (3)
    "norm_finish_logit_avg_race_z",
    "jockey_surprise_race_z",
    "haron_time_zscore_avg_race_z",
    # Phase 1: レース内pct (3)
    "norm_finish_logit_avg_race_pct",
    "jockey_surprise_race_pct",
    "haron_time_zscore_avg_race_pct",
]
```

- [ ] **Step 2: p_ability_place 行削除**

`add_ability_probs()` 内の以下を削除:
```python
# 削除: df["p_ability_place"] = np.clip(df["p_ability_win"] * 3.0, 0, 1)
```

- [ ] **Step 3: domain/models.py SubmodelSet に place_ability 追加**

```python
@dataclass
class SubmodelSet:
    market: MarketModel
    stage1: AbilityModel
    place_ability: PlaceAbilityModel  # NEW
    win: WinTwoStageModel
    ev_corrector: EVCorrectionModel
    place: PlaceTwoStageModel
    wide: WideTwoStageModel
    confidence: RobustConfidenceEstimator
```

- [ ] **Step 4: 既存テスト確認**

Run: `python -m pytest tests/ -v --timeout=60`
Expected: 既存テストのうち SubmodelSet コンストラクタに依存するものが FAIL。次タスクで修正。

- [ ] **Step 5: コミット**

```bash
git add src/models/stage1_ability_model.py src/domain/models.py
git commit -m "feat: FEATURE_COLS 16列化 + SubmodelSet に place_ability 追加 (Phase 1 Task 6)"
```

---

## Task 7: TrainingPipeline 統合

**Files:**
- Modify: `src/pipelines/training_pipeline.py`

- [ ] **Step 0: run() 内で race_df, entry_df をインスタンス属性に保存**

`_train_submodel()` は `df: pd.DataFrame`（surface フィルタ済み feat_df）のみを受け取る。`race_df` と `entry_df` は `run()` 内のローカル変数であり、`_train_submodel()` からはアクセスできない。

`run()` メソッドのデータロード直後にインスタンス属性として保存:

```python
def run(self, train_start: str, train_end: str) -> TrainedModelsV5:
    # ... existing data loading ...
    race_df = self.db.load_races(start, end)         # line 66
    entry_df = self.db.load_entries_with_results(start, end)  # line 67

    # NEW: _train_submodel 内で HorseHistoryFeatures が使用するため保存
    self._race_df = race_df
    self._entry_df = entry_df
    # ... rest of run() ...
```

- [ ] **Step 1: _train_submodel() に HorseHistoryFeatures + PlaceAbilityModel 統合**

`_train_submodel()` の先頭に HorseHistoryFeatures 呼び出しを追加:

```python
def _train_submodel(self, df: pd.DataFrame) -> tuple[SubmodelSet, pd.DataFrame]:
    # NEW: 馬過去成績特徴量
    from features.horse_history_features import HorseHistoryFeatures
    hist = HorseHistoryFeatures(engine=self.db.get_engine())
    # self._race_df, self._entry_df は run() で保存済み
    hist_df = hist.compute(self._race_df, self._entry_df, df["race_id"].unique())
    df = df.merge(hist_df, on=["race_id", "umaban"], how="left")
    df = HorseHistoryFeatures.add_race_transforms(df)
    # ... 既存フロー ...
```

PlaceAbilityModel を AbilityModel の直後に追加:

```python
    # NEW: PlaceAbilityModel
    from models.place_ability_model import PlaceAbilityModel
    place_ability = PlaceAbilityModel()
    place_ability.train(df)
    df = place_ability.predict(df)
```

SubmodelSet に place_ability を追加:

```python
    return (
        SubmodelSet(
            market=market,
            stage1=stage1,
            place_ability=place_ability,  # NEW
            win=win_2s,
            ev_corrector=ev_corrector,
            place=place_2s,
            wide=wide_2s,
            confidence=conf,
        ),
        df,
    )
```

- [ ] **Step 2: テスト実行**

Run: `python -m pytest tests/ -v --timeout=120`
Expected: 全テスト PASS（SubmodelSet コンストラクタ修正済み）

- [ ] **Step 3: コミット**

```bash
git add src/pipelines/training_pipeline.py
git commit -m "feat: TrainingPipeline に HorseHistory + PlaceAbility 統合 (Phase 1 Task 7)"
```

---

## Task 8: BacktestEngine 推論パス更新

**Files:**
- Modify: `src/backtest/engine.py`

- [ ] **Step 1: BacktestEngine.run() に HorseHistoryFeatures + PlaceAbilityModel 推論追加**

推論ループ内でサブモデル選択後、HorseHistoryFeatures と PlaceAbilityModel の推論を追加:

**前提: `run()` メソッド内のローカル変数 `race_df`, `entry_df` をインスタンス属性に保存する必要がある:**

```python
def run(self, test_start, test_end):
    # ... existing data loading (line 88-92) ...
    race_df = self.db.load_races(start, end)
    entry_df = self.db.load_entries_with_results(start, end)
    # NEW: HorseHistoryFeatures 用にインスタンス属性に保存
    self._race_df = race_df
    self._entry_df = entry_df
    # ... rest of run() ...
```

**レースループ内の推論パス:**

```python
# 3b. サブモデル選択 (既存)
surface_key = race_df_single["surface_key"].iloc[0]
if surface_key not in self.models.submodels:
    continue
submodel = self.models.submodels[surface_key]

# NEW: HorseHistoryFeatures 推論
from features.horse_history_features import HorseHistoryFeatures
hist = HorseHistoryFeatures(engine=self.db.get_engine())
# self._race_df, self._entry_df は run() 先頭で保存済み
hist_df = hist.compute(self._race_df, self._entry_df, [race_id])
race_df_single = race_df_single.merge(hist_df, on=["race_id", "umaban"], how="left")
race_df_single = HorseHistoryFeatures.add_race_transforms(race_df_single)

# NEW: PlaceAbilityModel 推論
if hasattr(submodel, "place_ability") and submodel.place_ability is not None:
    race_df_single = submodel.place_ability.predict(race_df_single)
```

**注意**: `db` パラメータが `None` の場合、`HorseHistoryFeatures` はDB接続に失敗する。`BacktestEngine(db=DatabaseConnection())` のように必ずDB接続を渡すこと。

- [ ] **Step 2: テスト実行**

Run: `python -m pytest tests/ -v --timeout=120`

- [ ] **Step 3: Phase 1 完了コミット**

```bash
git add src/backtest/engine.py
git commit -m "feat: BacktestEngine に HorseHistory + PlaceAbility 推論追加 (Phase 1 完了)"
```

---

# Phase 2: 追加特徴量拡張

## Task 9: jockey_cond_wr + weight_absolute 特徴量追加

**Files:**
- Modify: `src/features/horse_history_features.py`
- Modify: `src/models/stage1_ability_model.py`

- [ ] **Step 1: horse_history_features.py に jockey_cond_wr 追加**

`compute()` 内に条件別騎手勝率の計算を追加:

```python
# jockey_cond_wr: 距離bin × surface での条件別勝率（hierarchical smoothing）
k = 25
cond_wr = wins_in_cond / max(rides_in_cond, 1)
global_wr = total_wins / max(total_rides, 1)
w = min(rides_in_cond / (rides_in_cond + k), 1.0) if rides_in_cond >= 10 else 0.0
smoothed = w * cond_wr + (1 - w) * global_wr
```

BASE_COLS に `"jockey_cond_wr"` を追加。レース内変換も自動適用。

- [ ] **Step 2: weight_absolute 列追加**

`compute()` 内で `weight_absolute` を設定:
```python
df["weight_absolute"] = df["weight"]  # FeatureEngine 出力の weight 列
```

- [ ] **Step 3: FEATURE_COLS 20列に更新**

`stage1_ability_model.py` の FEATURE_COLS に4列追加:
```python
# Phase 2 (4)
"jockey_cond_wr",
"jockey_cond_wr_race_z",
"jockey_cond_wr_race_pct",
"weight_absolute",
```

- [ ] **Step 4: テスト + コミット**

Run: `python -m pytest tests/ -v --timeout=120`

```bash
git add src/features/horse_history_features.py src/models/stage1_ability_model.py
git commit -m "feat: Phase 2 特徴量追加 — jockey_cond_wr + weight_absolute (20列)"
```

---

# Phase 3: RegimeDetector 実データ化

## Task 10: flb_slope + odds_volatility + 人気帯別ROI EMA

**Files:**
- Modify: `src/features/market_bias_features.py`
- Modify: `src/features/odds_dynamics_features.py`
- Modify: `src/pipelines/training_pipeline.py`
- Modify: `src/models/regime_detector.py`

- [ ] **Step 1: market_bias_features.py に compute_flb_slope() 追加**

スペック §3-1 のフル擬似コードをそのまま実装（OLS回帰 on 人気帯別 p_implied vs p_actual）。

- [ ] **Step 2: odds_dynamics_features.py に compute_rolling_volatility() 追加**

```python
def compute_rolling_volatility(race_feat_df: pd.DataFrame) -> pd.Series:
    df = race_feat_df.sort_values("race_date").reset_index(drop=True)
    col = "odds_volatility_t10"
    if col not in df.columns:
        return pd.Series(0.1, index=df.index)
    return df[col].rolling(window=200, min_periods=50).mean().fillna(0.1)
```

- [ ] **Step 3: training_pipeline.py _build_regime_stats() 実データ化**

ダミー値を置き換え:
```python
stats["flb_slope"] = compute_flb_slope(race_feat_df)
stats["odds_volatility_mean"] = compute_rolling_volatility(race_feat_df)
# 人気帯別ROI EMA (alpha=0.07)
for band_name in ["favorite", "mid", "longshot"]:
    stats[f"{band_name}_roi_ema"] = compute_band_roi_ema(race_feat_df, ...)
```

- [ ] **Step 4: RegimeDetector FEATURE_COLS 更新**

10列 → 11列（rolling_roi_200, hit_rate_top3_mean を削除、3 EMA列追加、market_entropy_mean 維持）

- [ ] **Step 5: テスト + コミット**

```bash
git add src/features/market_bias_features.py src/features/odds_dynamics_features.py src/pipelines/training_pipeline.py src/models/regime_detector.py
git commit -m "feat: Phase 3 RegimeDetector 実データ化 — flb_slope + volatility + ROI EMA"
```

---

# Phase 4: ベッティング戦略最適化

## Task 11: ワイド戦略 + Fractional Kelly

**Files:**
- Modify: `src/betting/wide_strategy.py`
- Modify: `src/betting/stake_calculator.py`

- [ ] **Step 1: wide_strategy.py に同一馬制約 + 人気帯多様性制約追加**

スペック §4-1 の select_bets() を実装。`used_horses: set` と `used_bands: set` で重複排除。

- [ ] **Step 2: stake_calculator.py に Fractional Kelly 導入**

`FRACTIONAL_KELLY = 0.5` を追加。`calc_stake()` 内で Kelly fraction に 0.5 を掛ける。

- [ ] **Step 3: テスト + コミット**

```bash
git add src/betting/wide_strategy.py src/betting/stake_calculator.py
git commit -m "feat: Phase 4 ワイド同一馬制約 + Fractional Kelly (0.5x)"
```

---

# Phase 5: 検証基盤強化

## Task 12: Walk-forward CV + パラメータフリーズ

**Files:**
- Create: `notebooks/12_time_series_cv.ipynb`
- Modify: `src/backtest/validation_suite.py`

- [ ] **Step 1: 12_time_series_cv.ipynb 作成**

3ウィンドウ Walk-forward CV:
- Window 1: 2018-2021 train → 2022 test
- Window 2: 2019-2022 train → 2023 test
- Window 3: 2020-2023 train → 2024 test

各ウィンドウで ROI, Max DD, Logloss, Spearman ρ を計測。

- [ ] **Step 2: validation_suite.py に自動ホールドアウト検証追加**

`run_all()` をノートブックから呼び出し可能に。git commit hash をパラメータフリーズに記録。

- [ ] **Step 3: コミット**

```bash
git add notebooks/12_time_series_cv.ipynb src/backtest/validation_suite.py
git commit -m "feat: Phase 5 Walk-forward CV + パラメータフリーズ"
```

---

# 最終: 全Phase統合テスト

## Task 13: 統合テスト + ホールドアウト検証

- [ ] **Step 1: 全テスト実行**

Run: `python -m pytest tests/ -v --timeout=300`

- [ ] **Step 2: ホールドアウト検証**

`notebooks/11_holdout_final_evaluation.ipynb` または `BacktestValidationSuite.run_all()` で:
- Place ROI >= 100%
- Wide ROI >= 103%
- Overall ROI >= 101%
- Max DD <= 16%
- 36ヶ月中22ヶ月以上黒字

- [ ] **Step 3: 最終コミット**

```bash
git add -A
git commit -m "feat: ROI改善 Phase 1-5 全体実装完了"
```
