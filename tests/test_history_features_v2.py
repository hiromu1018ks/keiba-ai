"""test_history_features_v2.py — HorseHistoryFeatures 新規特徴量 (8個) のテスト

Task 4: Group A — HorseHistoryFeatures拡張
  1. harontimel5_avg — 直近5走のハロンタイム平均
  2. harontimel5_zscore — 距離ビンz-score平均
  3. harontime_late_trend — 最後2走 vs 最初3走 (負=改善)
  4. timediff_avg — 直近5走のタイム差平均
  5. jyuni1c_avg — 直近5走の1コーナー位置平均
  6. jyuni4c_avg — 直近5走の4コーナー位置平均
  7. closing_index_avg — (4C正規化 - 着順正規化) の直近5走平均
  8. kyakusitukubun_cd — 直近走の脚質コード
+ リーク防止テスト (target_date以降のレース除外)
+ 新馬テスト (過去成績なし → NaN)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from db.parquet_store import ParquetStore


def _make_history(
    entries_hist: pd.DataFrame,
    races_hist: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """テスト用の履歴データを作成するヘルパー。"""
    return entries_hist, races_hist


def _make_target(
    race_id: str = "r_target",
    race_date: str = "2024-06-01",
    umaban: int = 1,
    ketto_num: str = "H001",
    kisyu_code: str = "J001",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """テスト対象レース・出走データを作成する。"""
    race_df = pd.DataFrame({"race_id": [race_id], "race_date": [pd.Timestamp(race_date)]})
    entry_df = pd.DataFrame(
        {
            "race_id": [race_id],
            "umaban": [umaban],
            "kettonum": [ketto_num],
            "kisyucode": [kisyu_code],
            "weight": [480.0],
        }
    )
    return race_df, entry_df


def _build_entries_hist(rows: list[dict]) -> pd.DataFrame:
    """entries_hist DataFrameを構築する。必須カラムを補完。"""
    defaults = {
        "race_id": "p1",
        "umaban": 1,
        "kettonum": "H001",
        "kisyucode": "J001",
        "kakuteijyuni": 1,
        "odds": 5.0,
        "harontimel3": float("nan"),
        "timediff": float("nan"),
        "jyuni1c": float("nan"),
        "jyuni4c": float("nan"),
        "kyakusitukubun": float("nan"),
    }
    for row in rows:
        for k, v in defaults.items():
            row.setdefault(k, v)
    return pd.DataFrame(rows)


def _build_races_hist(rows: list[dict]) -> pd.DataFrame:
    """races_hist DataFrameを構築する。必須カラムを補完。"""
    defaults = {
        "race_id": "p1",
        "syussotosu": 16,
        "race_date": pd.Timestamp("2024-05-01"),
        "trackcd": 11,
        "kyori": 1600,
        "surface": "turf",
    }
    for row in rows:
        for k, v in defaults.items():
            row.setdefault(k, v)
    return pd.DataFrame(rows)


def _compute_features(
    entries_hist: pd.DataFrame,
    races_hist: pd.DataFrame,
    race_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    target_race_ids: list[str] | None = None,
) -> pd.DataFrame:
    """HorseHistoryFeatures.compute()をモックstoreで実行する。"""
    from features.horse_history_features import HorseHistoryFeatures

    mock_store = MagicMock(spec=ParquetStore)
    with patch("db.readers.load_history_entries", return_value=entries_hist):
        with patch("db.readers.load_history_races", return_value=races_hist):
            hhf = HorseHistoryFeatures(store=mock_store)
            ids = np.array(target_race_ids) if target_race_ids else None
            return hhf.compute(race_df, entry_df, ids)


# ============================================================
# Test 1: harontimel5_avg
# ============================================================


class TestHaronTimeL5Avg:
    """harontimel5_avg — 直近5走のハロンタイム平均"""

    def test_last3_average(self) -> None:
        """直近5走のハロンタイム平均を計算"""
        entries_hist = _build_entries_hist(
            [
                {"race_id": "p1", "harontimel3": 34.0, "kakuteijyuni": 3, "odds": 5.0},
                {"race_id": "p2", "harontimel3": 35.0, "kakuteijyuni": 2, "odds": 4.0},
                {"race_id": "p3", "harontimel3": 36.0, "kakuteijyuni": 1, "odds": 3.0},
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-03-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p2",
                    "race_date": pd.Timestamp("2024-04-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p3",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
            ]
        )
        race_df, entry_df = _make_target()
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["harontimel5_avg"].iloc[0]
        assert not np.isnan(val)
        assert abs(val - 35.0) < 1e-6  # (34+35+36)/3 = 35.0

    def test_nan_values_excluded(self) -> None:
        """NaN値は平均から除外される"""
        entries_hist = _build_entries_hist(
            [
                {"race_id": "p1", "harontimel3": 34.0, "kakuteijyuni": 3, "odds": 5.0},
                {"race_id": "p2", "harontimel3": float("nan"), "kakuteijyuni": 2, "odds": 4.0},
                {"race_id": "p3", "harontimel3": 36.0, "kakuteijyuni": 1, "odds": 3.0},
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-03-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p2",
                    "race_date": pd.Timestamp("2024-04-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p3",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
            ]
        )
        race_df, entry_df = _make_target()
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["harontimel5_avg"].iloc[0]
        assert not np.isnan(val)
        assert abs(val - 35.0) < 1e-6  # (34+36)/2 = 35.0


# ============================================================
# Test 2: timediff_avg
# ============================================================


class TestTimeDiffAvg:
    """timediff_avg — 直近3走のタイム差平均"""

    def test_last3_average(self) -> None:
        entries_hist = _build_entries_hist(
            [
                {"race_id": "p1", "timediff": 0.5, "kakuteijyuni": 3, "odds": 5.0},
                {"race_id": "p2", "timediff": 0.3, "kakuteijyuni": 2, "odds": 4.0},
                {"race_id": "p3", "timediff": 0.1, "kakuteijyuni": 1, "odds": 3.0},
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-03-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p2",
                    "race_date": pd.Timestamp("2024-04-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p3",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
            ]
        )
        race_df, entry_df = _make_target()
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["timediff_avg"].iloc[0]
        assert not np.isnan(val)
        assert abs(val - 0.3) < 1e-6  # (0.5+0.3+0.1)/3


# ============================================================
# Test 3: jyuni1c_avg
# ============================================================


class TestCorner1cAvg:
    """jyuni1c_avg — 直近3走の1コーナー通過位置平均"""

    def test_last3_average(self) -> None:
        entries_hist = _build_entries_hist(
            [
                {"race_id": "p1", "jyuni1c": 5, "kakuteijyuni": 3, "odds": 5.0},
                {"race_id": "p2", "jyuni1c": 3, "kakuteijyuni": 2, "odds": 4.0},
                {"race_id": "p3", "jyuni1c": 1, "kakuteijyuni": 1, "odds": 3.0},
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-03-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p2",
                    "race_date": pd.Timestamp("2024-04-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p3",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
            ]
        )
        race_df, entry_df = _make_target()
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["jyuni1c_avg"].iloc[0]
        assert not np.isnan(val)
        assert abs(val - 3.0) < 1e-6  # (5+3+1)/3


# ============================================================
# Test 4: jyuni4c_avg
# ============================================================


class TestCorner4cAvg:
    """jyuni4c_avg — 直近3走の4コーナー通過位置平均"""

    def test_last3_average(self) -> None:
        entries_hist = _build_entries_hist(
            [
                {"race_id": "p1", "jyuni4c": 8, "kakuteijyuni": 3, "odds": 5.0},
                {"race_id": "p2", "jyuni4c": 4, "kakuteijyuni": 2, "odds": 4.0},
                {"race_id": "p3", "jyuni4c": 2, "kakuteijyuni": 1, "odds": 3.0},
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-03-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p2",
                    "race_date": pd.Timestamp("2024-04-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p3",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
            ]
        )
        race_df, entry_df = _make_target()
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["jyuni4c_avg"].iloc[0]
        assert not np.isnan(val)
        assert abs(val - 4.666667) < 1e-3  # (8+4+2)/3


# ============================================================
# Test 5: closing_index_avg
# ============================================================


class TestClosingIndexAvg:
    """closing_index_avg — (4C正規化 - 着順正規化) の直近3走平均

    追い込み馬は closing_index > 0 (4Cで後方→ゴールで上位)
    逃げ馬は closing_index < 0 (4Cで前方→ゴールで下位)
    """

    def test_closing_from_behind(self) -> None:
        """後方から追い込み (4C=10→着順=2, 16頭) → 正の値"""
        entries_hist = _build_entries_hist(
            [
                {
                    "race_id": "p1",
                    "jyuni4c": 10,
                    "kakuteijyuni": 2,
                    "odds": 5.0,
                    "kyakusitukubun": 4,
                },
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
            ]
        )
        race_df, entry_df = _make_target()
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["closing_index_avg"].iloc[0]
        assert not np.isnan(val)
        # norm_4c = (10-1)/(16-1) = 9/15 = 0.6
        # norm_finish = (2-1)/(16-1) = 1/15 ≈ 0.0667
        # closing = 0.6 - 0.0667 ≈ 0.5333
        assert val > 0.5

    def test_early_speed_fading(self) -> None:
        """逃げて失速 (4C=1→着順=10, 16頭) → 負の値"""
        entries_hist = _build_entries_hist(
            [
                {
                    "race_id": "p1",
                    "jyuni4c": 1,
                    "kakuteijyuni": 10,
                    "odds": 5.0,
                    "kyakusitukubun": 1,
                },
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
            ]
        )
        race_df, entry_df = _make_target()
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["closing_index_avg"].iloc[0]
        assert not np.isnan(val)
        # norm_4c = (1-1)/(16-1) = 0
        # norm_finish = (10-1)/(16-1) = 9/15 = 0.6
        # closing = 0 - 0.6 = -0.6
        assert val < -0.5


# ============================================================
# Test 6: kyakusitukubun_cd
# ============================================================


class TestKyakusituCd:
    """kyakusitukubun_cd — 直近走の脚質コード"""

    def test_latest_value(self) -> None:
        """直近走のkyakusitukubun値を取得"""
        entries_hist = _build_entries_hist(
            [
                {"race_id": "p1", "kyakusitukubun": 1, "kakuteijyuni": 3, "odds": 5.0},
                {"race_id": "p2", "kyakusitukubun": 3, "kakuteijyuni": 2, "odds": 4.0},
                {"race_id": "p3", "kyakusitukubun": 4, "kakuteijyuni": 1, "odds": 3.0},
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-03-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p2",
                    "race_date": pd.Timestamp("2024-04-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p3",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
            ]
        )
        race_df, entry_df = _make_target()
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["kyakusitukubun_cd"].iloc[0]
        assert not np.isnan(val)
        assert val == 4  # 直近(p3)のkyakusitukubun値


# ============================================================
# Test 7: Leak prevention — target_date以降のレース除外
# ============================================================


class TestLeakPrevention:
    """target_date以降のレースが特徴量に含まれないことを確認"""

    def test_future_haron_time_excluded(self) -> None:
        """target_dateより後のハロンタイムが平均に含まれない"""
        entries_hist = _build_entries_hist(
            [
                {"race_id": "p1", "harontimel3": 34.0, "kakuteijyuni": 3, "odds": 5.0},
                {"race_id": "p2", "harontimel3": 99.0, "kakuteijyuni": 2, "odds": 4.0},  # 未来
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p2",
                    "race_date": pd.Timestamp("2024-07-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },  # 未来
            ]
        )
        race_df, entry_df = _make_target()  # target_date = 2024-06-01
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["harontimel5_avg"].iloc[0]
        assert not np.isnan(val)
        assert abs(val - 34.0) < 1e-6  # p2(7月)は除外 → p1のみ

    def test_same_day_excluded(self) -> None:
        """target_dateと同日のレースも除外される"""
        entries_hist = _build_entries_hist(
            [
                {"race_id": "p1", "harontimel3": 34.0, "kakuteijyuni": 3, "odds": 5.0},
                {"race_id": "p2", "harontimel3": 99.0, "kakuteijyuni": 2, "odds": 4.0},  # 同日
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p2",
                    "race_date": pd.Timestamp("2024-06-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },  # 同日
            ]
        )
        race_df, entry_df = _make_target()  # target_date = 2024-06-01
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["harontimel5_avg"].iloc[0]
        assert not np.isnan(val)
        assert abs(val - 34.0) < 1e-6  # p2(同日)は除外


# ============================================================
# Test 8: 新馬 (過去成績なし → 全NaN)
# ============================================================


class TestNewHorse:
    """過去成績のない新馬は全ての新規特徴量がNaN"""

    def test_no_history_returns_nan(self) -> None:
        """histに該当馬のデータがない → 新規特徴量は全てNaN

        Note: The horse must have SOME history entry (possibly from jockey matches)
        for the compute() loop to process it. We include the jockey's past data
        but ensure the horse itself has no races with valid_field/finish_pos > 0.
        """
        # The horse has no direct history, but the jockey has other entries
        # This ensures the horse is in the loop (jockey matches)
        entries_hist = _build_entries_hist(
            [
                {
                    "race_id": "p1",
                    "kettonum": "OTHER_HORSE",
                    "kisyucode": "NEW_JOCKEY",
                    "kakuteijyuni": 1,
                    "odds": 5.0,
                    "harontimel3": 34.0,
                },
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
            ]
        )
        race_df, entry_df = _make_target(ketto_num="NEW_HORSE", kisyu_code="NEW_JOCKEY")
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        new_cols = [
            "harontimel5_avg",
            "harontimel5_zscore",
            "harontime_late_trend",
            "timediff_avg",
            "jyuni1c_avg",
            "jyuni4c_avg",
            "closing_index_avg",
            "kyakusitukubun_cd",
        ]
        for col in new_cols:
            assert col in result.columns, f"{col} not in result columns"
            assert np.isnan(result[col].iloc[0]), f"{col} should be NaN for new horse"


# ============================================================
# Test: harontimel5_zscore
# ============================================================


class TestHaronTimeL5Zscore:
    """harontimel5_zscore — 距離ビンz-score平均"""

    def test_zscore_calculation(self) -> None:
        """距離ビン内でz-scoreを計算"""
        entries_hist = _build_entries_hist(
            [
                {"race_id": "p1", "harontimel3": 34.0, "kakuteijyuni": 3, "odds": 5.0},
                {"race_id": "p2", "harontimel3": 36.0, "kakuteijyuni": 2, "odds": 4.0},
                {"race_id": "p3", "harontimel3": 35.0, "kakuteijyuni": 1, "odds": 3.0},
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-03-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p2",
                    "race_date": pd.Timestamp("2024-04-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
                {
                    "race_id": "p3",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
            ]
        )
        race_df, entry_df = _make_target()
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["harontimel5_zscore"].iloc[0]
        assert not np.isnan(val)
        # 3走全て同じ距離ビン → mean=35.0, std=1.0
        # z-scores: (34-35)/1=-1, (36-35)/1=1, (35-35)/1=0
        # avg zscore = 0
        assert abs(val) < 1e-6

    def test_single_run_zscore_is_zero(self) -> None:
        """1走のみの場合、std=NaN → zscoreもNaN"""
        entries_hist = _build_entries_hist(
            [
                {"race_id": "p1", "harontimel3": 35.0, "kakuteijyuni": 3, "odds": 5.0},
            ]
        )
        races_hist = _build_races_hist(
            [
                {
                    "race_id": "p1",
                    "race_date": pd.Timestamp("2024-05-01"),
                    "syussotosu": 16,
                    "trackcd": 11,
                    "kyori": 1600,
                },
            ]
        )
        race_df, entry_df = _make_target()
        result = _compute_features(entries_hist, races_hist, race_df, entry_df, ["r_target"])

        val = result["harontimel5_zscore"].iloc[0]
        # 1走のみのビンでは std=0 or NaN → zscoreはNaN
        assert np.isnan(val)


# ============================================================
# Test: BASE_COLS updated
# ============================================================


class TestBaseCols:
    """BASE_COLS が新規特徴量を含むことを確認"""

    def test_base_cols_contains_new_features(self) -> None:
        from features.horse_history_features import HorseHistoryFeatures

        expected = [
            "harontimel5_avg",
            "harontimel5_zscore",
            "harontime_late_trend",
            "timediff_avg",
            "jyuni1c_avg",
            "jyuni4c_avg",
            "closing_index_avg",
            "kyakusitukubun_cd",
        ]
        for col in expected:
            assert col in HorseHistoryFeatures.BASE_COLS, f"{col} missing from BASE_COLS"

    def test_old_placeholder_removed(self) -> None:
        """haron_time_zscore_avg (旧プレースホルダー) がBASE_COLSから削除されている"""
        from features.horse_history_features import HorseHistoryFeatures

        assert "haron_time_zscore_avg" not in HorseHistoryFeatures.BASE_COLS
