"""payout_maps.py 純粋関数のテスト (D-09, D-12)

DB不要。全テスト mock / インメモリ DataFrame のみ使用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from betting.payout_maps import (
    build_payout_map,
    build_place_payout_map,
    build_wide_payout_map,
    build_win_payout_map,
)


def _make_place_df(rows: list[dict]) -> pd.DataFrame:
    """複勝払戻テスト用の完全な列を持つ DataFrame を構築。

    テストで部分的な列を指定した場合、残りは NaN で埋める。
    """
    all_cols = ["race_id"] + [
        c for i in range(1, 6) for c in (f"payfukusyoumaban{i}", f"payfukusyopay{i}")
    ]
    df = pd.DataFrame(rows)
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[all_cols]


def _make_wide_df(rows: list[dict]) -> pd.DataFrame:
    """ワイド払戻テスト用の完全な列を持つ DataFrame を構築。"""
    all_cols = ["race_id"] + [
        c for i in range(1, 8) for c in (f"paywidekumi{i}", f"paywidepay{i}")
    ]
    df = pd.DataFrame(rows)
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[all_cols]


# ---------------------------------------------------------------------------
# TestBuildWinPayoutMap
# ---------------------------------------------------------------------------


class TestBuildWinPayoutMap:
    """build_win_payout_map のテスト。"""

    def test_empty_dataframe_returns_empty_dict(self) -> None:
        result = build_win_payout_map(pd.DataFrame())
        assert result == {}

    def test_single_payout_row(self) -> None:
        df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "paytansyoumaban1": [3],
                "paytansyopay1": [250],
            }
        )
        result = build_win_payout_map(df)
        assert result == {("20240101010101", 3): 2.5}

    def test_nan_values_skipped(self) -> None:
        df = pd.DataFrame(
            {
                "race_id": ["20240101010101", "20240101010102"],
                "paytansyoumaban1": [3, np.nan],
                "paytansyopay1": [250, np.nan],
            }
        )
        result = build_win_payout_map(df)
        assert result == {("20240101010101", 3): 2.5}

    def test_multiple_races(self) -> None:
        df = pd.DataFrame(
            {
                "race_id": ["20240101010101", "20240101010102"],
                "paytansyoumaban1": [1, 5],
                "paytansyopay1": [150, 400],
            }
        )
        result = build_win_payout_map(df)
        assert len(result) == 2
        assert result[("20240101010101", 1)] == pytest.approx(1.5)
        assert result[("20240101010102", 5)] == pytest.approx(4.0)

    def test_all_nan_returns_empty_dict(self) -> None:
        df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "paytansyoumaban1": [np.nan],
                "paytansyopay1": [np.nan],
            }
        )
        result = build_win_payout_map(df)
        assert result == {}


# ---------------------------------------------------------------------------
# TestBuildPayoutMap (place/fuku)
# ---------------------------------------------------------------------------


class TestBuildPayoutMap:
    """build_payout_map のテスト。"""

    def test_empty_dataframe_returns_empty_dict(self) -> None:
        result = build_payout_map(pd.DataFrame())
        assert result == {}

    def test_multiple_place_positions(self) -> None:
        """複数の複勝着順 (1-3位) が正しくマップされる。"""
        df = _make_place_df(
            [
                {
                    "race_id": "20240101010101",
                    "payfukusyoumaban1": 1,
                    "payfukusyopay1": 130,
                    "payfukusyoumaban2": 3,
                    "payfukusyopay2": 250,
                    "payfukusyoumaban3": 5,
                    "payfukusyopay3": 180,
                }
            ]
        )
        result = build_payout_map(df)
        assert len(result) == 3
        assert result[("20240101010101", 1)] == pytest.approx(1.3)
        assert result[("20240101010101", 3)] == pytest.approx(2.5)
        assert result[("20240101010101", 5)] == pytest.approx(1.8)

    def test_nan_columns_skipped(self) -> None:
        """NaN の着順・払戻列はスキップされる。"""
        df = _make_place_df(
            [
                {
                    "race_id": "20240101010101",
                    "payfukusyoumaban1": 1,
                    "payfukusyopay1": 130,
                    # payfukusyoumaban2/payfukusyopay2 は NaN
                    "payfukusyoumaban3": 5,
                    "payfukusyopay3": 180,
                }
            ]
        )
        result = build_payout_map(df)
        assert len(result) == 2
        assert ("20240101010101", 1) in result
        assert ("20240101010101", 5) in result

    def test_same_race_umaban_keeps_max(self) -> None:
        """同一 (race_id, umaban) の複数着順がある場合、最大値を保持する。"""
        df = _make_place_df(
            [
                {
                    "race_id": "20240101010101",
                    "payfukusyoumaban1": 3,
                    "payfukusyopay1": 200,
                    "payfukusyoumaban2": 3,
                    "payfukusyopay2": 350,
                }
            ]
        )
        result = build_payout_map(df)
        assert result[("20240101010101", 3)] == pytest.approx(3.5)

    def test_multiple_races_place(self) -> None:
        df = _make_place_df(
            [
                {
                    "race_id": "20240101010101",
                    "payfukusyoumaban1": 1,
                    "payfukusyopay1": 130,
                    "payfukusyoumaban2": 3,
                    "payfukusyopay2": 250,
                },
                {
                    "race_id": "20240101010102",
                    "payfukusyoumaban1": 2,
                    "payfukusyopay1": 200,
                    "payfukusyoumaban2": 4,
                    "payfukusyopay2": 150,
                },
            ]
        )
        result = build_payout_map(df)
        assert len(result) == 4
        assert result[("20240101010101", 1)] == pytest.approx(1.3)
        assert result[("20240101010102", 2)] == pytest.approx(2.0)

    def test_build_place_payout_map_alias(self) -> None:
        """build_place_payout_map は build_payout_map のエイリアス。"""
        assert build_place_payout_map is build_payout_map


# ---------------------------------------------------------------------------
# TestBuildWidePayoutMap
# ---------------------------------------------------------------------------


class TestBuildWidePayoutMap:
    """build_wide_payout_map のテスト。"""

    def test_empty_dataframe_returns_empty_dict(self) -> None:
        result = build_wide_payout_map(pd.DataFrame())
        assert result == {}

    def test_valid_wide_pairs(self) -> None:
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": "13",
                    "paywidepay1": 570,
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert result == {("20240101010101", 1, 3): pytest.approx(5.7)}

    def test_length3_kumi_split(self) -> None:
        """3文字 kumi "513" は (5, 13) に分割される (first_two=51 > 18 なので split at 1)。"""
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": "513",
                    "paywidepay1": 800,
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert result == {("20240101010101", 5, 13): pytest.approx(8.0)}

    def test_length3_kumi_split_first_two_valid(self) -> None:
        """3文字 kumi "111" の場合 first_two=11 <= 18 なので (11, 1) に分割。"""
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": "111",
                    "paywidepay1": 600,
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert result == {("20240101010101", 1, 11): pytest.approx(6.0)}

    def test_length4_kumi(self) -> None:
        """4文字 kumi "1113" は (11, 13) に分割。"""
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": "1113",
                    "paywidepay1": 1200,
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert result == {("20240101010101", 11, 13): pytest.approx(12.0)}

    def test_float_kumi_converted(self) -> None:
        """Parquet が kumi を float64 で保存していても正しく処理される。"""
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": 13.0,
                    "paywidepay1": 570,
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert result == {("20240101010101", 1, 3): pytest.approx(5.7)}

    def test_multiple_wide_pairs(self) -> None:
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": "13",
                    "paywidepay1": 570,
                    "paywidekumi2": "25",
                    "paywidepay2": 400,
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert len(result) == 2
        assert result[("20240101010101", 1, 3)] == pytest.approx(5.7)
        assert result[("20240101010101", 2, 5)] == pytest.approx(4.0)

    def test_nan_kumi_skipped(self) -> None:
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": "13",
                    "paywidepay1": 570,
                    # paywidekumi2/paywidepay2 は NaN
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert len(result) == 1
        assert result[("20240101010101", 1, 3)] == pytest.approx(5.7)

    def test_lo_hi_ordering_ensured(self) -> None:
        """lo <= hi の順序が保証される (例: "31" -> lo=1, hi=3)。"""
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": "31",
                    "paywidepay1": 570,
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert result == {("20240101010101", 1, 3): pytest.approx(5.7)}

    def test_boundary_kumi_118(self) -> None:
        """3文字 kumi "118" -> first_two=11 <= 18 -> (11, 8)。"""
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": "118",
                    "paywidepay1": 700,
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert result == {("20240101010101", 8, 11): pytest.approx(7.0)}

    def test_boundary_kumi_181(self) -> None:
        """3文字 kumi "181" -> first_two=18 <= 18 -> (18, 1)。"""
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": "181",
                    "paywidepay1": 900,
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert result == {("20240101010101", 1, 18): pytest.approx(9.0)}

    def test_boundary_kumi_918(self) -> None:
        """3文字 kumi "918" -> first_two=91 > 18 -> (9, 18)。"""
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": "918",
                    "paywidepay1": 650,
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert result == {("20240101010101", 9, 18): pytest.approx(6.5)}

    def test_boundary_kumi_109(self) -> None:
        """3文字 kumi "109" -> first_two=10 <= 18 -> (10, 9), reordered to (9, 10)。"""
        df = _make_wide_df(
            [
                {
                    "race_id": "20240101010101",
                    "paywidekumi1": "109",
                    "paywidepay1": 480,
                }
            ]
        )
        result = build_wide_payout_map(df)
        assert result == {("20240101010101", 9, 10): pytest.approx(4.8)}
