"""BacktestReportGenerator のテスト"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


class TestDeriveFields:
    """_derive_fields のテスト"""

    def test_adds_race_date(self) -> None:
        """race_id の先頭8文字から race_date (YYYY-MM-DD) を抽出"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_id": "20240105101011", "stake": 100.0, "result": 240.0},
            {"race_id": "20241225123456", "stake": 100.0, "result": 0.0},
        ]
        result = gen._derive_fields(bets)
        assert result[0]["race_date"] == "2024-01-05"
        assert result[1]["race_date"] == "2024-12-25"

    def test_computes_profit(self) -> None:
        """profit = result - stake"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_id": "20240101010101", "stake": 100.0, "result": 240.0},
            {"race_id": "20240102010101", "stake": 100.0, "result": 0.0},
        ]
        result = gen._derive_fields(bets)
        assert result[0]["profit"] == 140.0
        assert result[1]["profit"] == -100.0

    def test_computes_is_win(self) -> None:
        """is_win = result > 0"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_id": "20240101010101", "stake": 100.0, "result": 240.0},
            {"race_id": "20240102010101", "stake": 100.0, "result": 0.0},
        ]
        result = gen._derive_fields(bets)
        assert result[0]["is_win"] is True
        assert result[1]["is_win"] is False

    def test_preserves_original_fields(self) -> None:
        """元のフィールドが保持される"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_id": "20240101010101", "stake": 100.0, "result": 240.0, "surface": "turf"},
        ]
        result = gen._derive_fields(bets)
        assert result[0]["surface"] == "turf"
        assert result[0]["race_id"] == "20240101010101"

    def test_empty_input(self) -> None:
        """空リストは空リストを返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        assert gen._derive_fields([]) == []


class TestComputeMonthlyStats:
    """_compute_monthly_stats のテスト"""

    def _make_bets(self) -> list[dict[str, Any]]:
        return [
            {"race_date": "2024-01-05", "stake": 100.0, "result": 240.0, "is_win": True},
            {"race_date": "2024-01-15", "stake": 100.0, "result": 0.0, "is_win": False},
            {"race_date": "2024-01-20", "stake": 100.0, "result": 180.0, "is_win": True},
            {"race_date": "2024-02-10", "stake": 100.0, "result": 0.0, "is_win": False},
            {"race_date": "2024-02-20", "stake": 100.0, "result": 0.0, "is_win": False},
        ]

    def test_monthly_aggregation(self) -> None:
        """月次集計が正しい ROI, 的中率, ベット数を返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_monthly_stats(self._make_bets())

        assert len(result) == 2  # 2 months

        jan = [m for m in result if m["month"] == "2024-01"][0]
        assert jan["bets"] == 3
        assert jan["wins"] == 2
        assert jan["win_rate"] == pytest.approx(2 / 3)
        assert jan["stake"] == 300.0
        assert jan["total_return"] == 420.0
        assert jan["roi"] == pytest.approx(420.0 / 300.0)

        feb = [m for m in result if m["month"] == "2024-02"][0]
        assert feb["bets"] == 2
        assert feb["wins"] == 0
        assert feb["roi"] == pytest.approx(0.0)

    def test_empty_input(self) -> None:
        """空リストは空リストを返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        assert gen._compute_monthly_stats([]) == []

    def test_all_losses(self) -> None:
        """全額ロスの月の ROI が 0.0"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_date": "2024-03-01", "stake": 100.0, "result": 0.0, "is_win": False},
            {"race_date": "2024-03-15", "stake": 100.0, "result": 0.0, "is_win": False},
        ]
        result = gen._compute_monthly_stats(bets)
        assert result[0]["roi"] == 0.0
        assert result[0]["win_rate"] == 0.0


class TestComputeConditionStats:
    """_compute_condition_stats のテスト"""

    def _make_bets(self) -> list[dict[str, Any]]:
        """多様な条件を持つテストデータ"""
        return [
            # turf sprint, popular, high EV, win
            {
                "surface": "turf",
                "distance": 1200,
                "popularity": 1,
                "ev": 1.8,
                "stake": 100.0,
                "result": 250.0,
                "is_win": True,
            },
            # turf sprint, popular, high EV, lose
            {
                "surface": "turf",
                "distance": 1200,
                "popularity": 2,
                "ev": 1.6,
                "stake": 100.0,
                "result": 0.0,
                "is_win": False,
            },
            # turf mile, mid-pop, mid EV, win
            {
                "surface": "turf",
                "distance": 1600,
                "popularity": 5,
                "ev": 1.3,
                "stake": 100.0,
                "result": 300.0,
                "is_win": True,
            },
            # dirt sprint, low-pop, low EV, lose
            {
                "surface": "dirt",
                "distance": 1200,
                "popularity": 8,
                "ev": 0.9,
                "stake": 100.0,
                "result": 0.0,
                "is_win": False,
            },
            # dirt mile, low-pop, high EV, win
            {
                "surface": "dirt",
                "distance": 1600,
                "popularity": 7,
                "ev": 1.5,
                "stake": 100.0,
                "result": 400.0,
                "is_win": True,
            },
        ]

    def test_surface_distance_analysis(self) -> None:
        """路面×距離帯の集計が正しい"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_condition_stats(self._make_bets())
        sd = result["surface_distance"]

        # turf/sprint: 2 bets, 1 win
        turf_sprint = [r for r in sd if r["surface"] == "turf" and r["distance_band"] == "sprint"][
            0
        ]
        assert turf_sprint["bets"] == 2
        assert turf_sprint["wins"] == 1
        assert turf_sprint["win_rate"] == pytest.approx(0.5)

        # dirt/sprint: 1 bet, 0 wins
        dirt_sprint = [r for r in sd if r["surface"] == "dirt" and r["distance_band"] == "sprint"][
            0
        ]
        assert dirt_sprint["bets"] == 1
        assert dirt_sprint["wins"] == 0

    def test_popularity_bands(self) -> None:
        """人気帯 (1-3, 4-6, 7+) の集計"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_condition_stats(self._make_bets())
        bands = result["popularity_bands"]

        band_1_3 = [b for b in bands if b["band"] == "1-3"][0]
        assert band_1_3["bets"] == 2  # popularity 1, 2

        band_4_6 = [b for b in bands if b["band"] == "4-6"][0]
        assert band_4_6["bets"] == 1  # popularity 5

        band_7p = [b for b in bands if b["band"] == "7+"][0]
        assert band_7p["bets"] == 2  # popularity 7, 8

    def test_ev_bands(self) -> None:
        """EV帯 (<1.0, 1.0-1.2, 1.2-1.5, 1.5+) の集計"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_condition_stats(self._make_bets())
        bands = result["ev_bands"]

        band_low = [b for b in bands if b["band"] == "<1.0"][0]
        assert band_low["bets"] == 1  # ev 0.9

        band_high = [b for b in bands if b["band"] == "1.5+"][0]
        assert band_high["bets"] == 3  # ev 1.8, 1.6, 1.5

    def test_empty_input(self) -> None:
        """空リストは空の統計を返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_condition_stats([])
        assert result["surface_distance"] == []
        assert result["popularity_bands"] == []
        assert result["ev_bands"] == []


class TestComputeBankrollSeries:
    """_compute_bankroll_series のテスト"""

    def test_bankroll_trajectory(self) -> None:
        """資金推移とドローダウンが正しく計算される"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [
            {"race_date": "2024-01-05", "bankroll_after": 100500.0},
            {"race_date": "2024-01-10", "bankroll_after": 100200.0},
            {"race_date": "2024-01-15", "bankroll_after": 100800.0},
            {"race_date": "2024-02-01", "bankroll_after": 99500.0},
        ]
        result = gen._compute_bankroll_series(bets)

        assert len(result) == 4
        assert result[0]["date"] == "2024-01-05"
        assert result[0]["bankroll"] == 100500.0
        assert result[0]["drawdown"] == 0.0  # peak = 100500, no DD

        # At 2024-01-10: bankroll=100200, peak=100500 → DD = (100500-100200)/100500
        assert result[1]["drawdown"] == pytest.approx(300.0 / 100500.0)

        # At 2024-01-15: bankroll=100800 > peak=100500 → new peak, DD=0
        assert result[2]["drawdown"] == 0.0

        # At 2024-02-01: bankroll=99500, peak=100800 → DD = (100800-99500)/100800
        assert result[3]["drawdown"] == pytest.approx(1300.0 / 100800.0)

    def test_single_bet(self) -> None:
        """ベット1件の場合 DD=0"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        bets = [{"race_date": "2024-01-01", "bankroll_after": 100000.0}]
        result = gen._compute_bankroll_series(bets)
        assert len(result) == 1
        assert result[0]["drawdown"] == 0.0

    def test_empty_input(self) -> None:
        """空リストは空リストを返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        assert gen._compute_bankroll_series([]) == []
