"""BacktestReportGenerator のテスト"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
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
                "kyori": 1200,
                "popularity": 1,
                "ev": 1.8,
                "stake": 100.0,
                "result": 250.0,
                "is_win": True,
            },
            # turf sprint, popular, high EV, lose
            {
                "surface": "turf",
                "kyori": 1200,
                "popularity": 2,
                "ev": 1.6,
                "stake": 100.0,
                "result": 0.0,
                "is_win": False,
            },
            # turf mile, mid-pop, mid EV, win
            {
                "surface": "turf",
                "kyori": 1600,
                "popularity": 5,
                "ev": 1.3,
                "stake": 100.0,
                "result": 300.0,
                "is_win": True,
            },
            # dirt sprint, low-pop, low EV, lose
            {
                "surface": "dirt",
                "kyori": 1200,
                "popularity": 8,
                "ev": 0.9,
                "stake": 100.0,
                "result": 0.0,
                "is_win": False,
            },
            # dirt mile, low-pop, high EV, win
            {
                "surface": "dirt",
                "kyori": 1600,
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
        assert result["odds_multiplier_bands"] == []
        assert result["regime_bands"] == []


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


class TestHtmlGeneration:
    """HTMLレポート生成のテスト"""

    def _make_result_and_history(self) -> tuple:
        from backtest.engine import BacktestResult

        result = BacktestResult(
            total_bets=3,
            total_stake=300.0,
            total_return=420.0,
            winning_bets=1,
            total_roi=1.4,
            max_drawdown=0.05,
            final_bankroll=100200.0,
        )
        bet_history = [
            {
                "race_id": "20240105010101",
                "bet_type": "place",
                "umaban": 3,
                "stake": 100.0,
                "odds": 2.4,
                "result": 240.0,
                "surface": "turf",
                "kyori": 1200,
                "ev": 1.5,
                "popularity": 3,
                "bankroll_after": 100200.0,
            },
            {
                "race_id": "20240110010101",
                "bet_type": "place",
                "umaban": 5,
                "stake": 100.0,
                "odds": 3.0,
                "result": 0.0,
                "surface": "dirt",
                "kyori": 1600,
                "ev": 1.3,
                "popularity": 6,
                "bankroll_after": 100100.0,
            },
            {
                "race_id": "20240115010101",
                "bet_type": "place",
                "umaban": 1,
                "stake": 100.0,
                "odds": 1.8,
                "result": 180.0,
                "surface": "turf",
                "kyori": 1800,
                "ev": 1.6,
                "popularity": 2,
                "bankroll_after": 100280.0,
            },
        ]
        return result, bet_history

    def test_html_contains_sections(self, tmp_path: Path) -> None:
        """HTMLに全セクションが含まれる"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        result, bet_history = self._make_result_and_history()
        path = gen.generate(result, bet_history, train_period="2020-2023", test_period="2024")

        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "サマリー" in html
        assert "資金推移" in html
        assert "月次ダッシュボード" in html
        assert "条件分析" in html
        assert "ベット明細" in html
        assert "140.0%" in html  # ROI

    def test_html_with_empty_history(self, tmp_path: Path) -> None:
        """空の bet_history でもHTMLが生成される"""
        from backtest.engine import BacktestResult
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        result = BacktestResult()
        path = gen.generate(result, [])

        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "サマリー" in html
        assert "データなし" in html

    def test_output_path(self, tmp_path: Path) -> None:
        """出力パスが backtest_report.html"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        result, bet_history = self._make_result_and_history()
        path = gen.generate(result, bet_history)

        assert path.name == "backtest_report.html"
        assert path.parent == tmp_path


class TestBetHistorySerialization:
    """bet_history JSON保存/読み込みのテスト"""

    def test_json_round_trip(self, tmp_path: Path) -> None:
        """JSON保存→読み込みでデータが保持される"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        original = [
            {
                "race_id": "20240101010101",
                "stake": 100.0,
                "result": 240.0,
                "surface": "turf",
                "kyori": 1200,
                "ev": 1.5,
                "popularity": 3,
                "bankroll_after": 100200.0,
            },
        ]
        json_path = gen.save_bet_history(original)
        loaded = gen.load_bet_history(json_path)

        assert len(loaded) == 1
        assert loaded[0]["race_id"] == "20240101010101"
        assert loaded[0]["ev"] == 1.5
        assert loaded[0]["bankroll_after"] == 100200.0

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """save_bet_history がJSONファイルを作成する"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        path = gen.save_bet_history([{"race_id": "20240101010101", "stake": 100.0, "result": 0.0}])
        assert path.exists()
        assert path.suffix == ".json"


class TestCliReportFlag:
    """--report フラグのテスト"""

    @patch("backtest.report.BacktestReportGenerator")
    @patch("pipelines.training_pipeline.TrainingPipelineV5")
    @patch("backtest.engine.BacktestEngine")
    @patch("db.parquet_store.ParquetStore")
    def test_report_flag_triggers_generation(
        self,
        mock_store_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_pipeline_cls: MagicMock,
        mock_report_gen_cls: MagicMock,
    ) -> None:
        """--report フラグでレポート生成が呼ばれる"""
        # Setup mocks
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        mock_store.exists.return_value = True

        mock_models = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline
        mock_pipeline.run.return_value = mock_models

        from backtest.engine import BacktestResult

        mock_result = BacktestResult(
            total_bets=10,
            total_stake=1000.0,
            total_return=1500.0,
            winning_bets=3,
            total_roi=1.5,
            max_drawdown=0.05,
            final_bankroll=101500.0,
            bet_history=[
                {
                    "race_id": "20240101010101",
                    "bet_type": "place",
                    "umaban": 1,
                    "stake": 100.0,
                    "odds": 2.4,
                    "result": 240.0,
                    "surface": "turf",
                    "kyori": 1200,
                    "ev": 1.5,
                    "popularity": 3,
                    "bankroll_after": 100200.0,
                },
            ],
        )
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.run.return_value = mock_result

        mock_gen = MagicMock()
        mock_report_gen_cls.return_value = mock_gen
        mock_gen.generate.return_value = MagicMock()
        mock_gen.save_bet_history.return_value = MagicMock()

        # sys.argv を直接操作 (main() は argparse で読む)
        with patch(
            "sys.argv",
            [
                "run_backtest.py",
                "--train-start",
                "20200101",
                "--train-end",
                "20231231",
                "--test-start",
                "20240101",
                "--test-end",
                "20241231",
                "--report",
            ],
        ):
            from scripts.run_backtest import main

            main()

        # Verify report generator was called
        mock_report_gen_cls.assert_called_once()
        mock_gen.generate.assert_called_once()
        mock_gen.save_bet_history.assert_called_once()
        call_args = mock_gen.generate.call_args
        assert call_args[0][0].total_roi == 1.5  # BacktestResult passed


class TestComputeRegimeStats:
    """_compute_regime_stats のテスト"""

    def _make_bets(self) -> list[dict[str, Any]]:
        return [
            {"regime": "aggressive", "stake": 100.0, "result": 250.0},
            {"regime": "aggressive", "stake": 100.0, "result": 0.0},
            {"regime": "conservative", "stake": 100.0, "result": 180.0},
            {"regime": "conservative", "stake": 100.0, "result": 0.0},
            {"regime": "collapsed", "stake": 100.0, "result": 0.0},
        ]

    def test_regime_aggregation(self) -> None:
        """regime別に bets/wins/roi を正しく集計する"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_regime_stats(self._make_bets())
        assert len(result) == 3
        agg = [r for r in result if r["regime"] == "aggressive"][0]
        assert agg["bets"] == 2
        assert agg["wins"] == 1
        assert agg["roi"] == pytest.approx(250.0 / 200.0)

    def test_empty_input(self) -> None:
        """空リストは空リストを返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        assert gen._compute_regime_stats([]) == []


class TestComputeConditionStatsWin:
    """_compute_condition_stats の win 拡張テスト"""

    def _make_win_bets(self) -> list[dict[str, Any]]:
        return [
            {
                "surface": "turf", "kyori": 1200, "popularity": 1,
                "ev": 1.8, "stake": 100.0, "result": 250.0, "is_win": True,
                "tanoddslow": 2.5, "regime": "aggressive",
            },
            {
                "surface": "turf", "kyori": 1600, "popularity": 5,
                "ev": 1.3, "stake": 100.0, "result": 300.0, "is_win": True,
                "tanoddslow": 5.0, "regime": "conservative",
            },
            {
                "surface": "dirt", "kyori": 1200, "popularity": 8,
                "ev": 0.9, "stake": 100.0, "result": 0.0, "is_win": False,
                "tanoddslow": 15.0, "regime": "aggressive",
            },
            {
                "surface": "dirt", "kyori": 1800, "popularity": 3,
                "ev": 1.5, "stake": 100.0, "result": 400.0, "is_win": True,
                "tanoddslow": 8.0, "regime": "collapsed",
            },
        ]

    def test_odds_multiplier_bands_present_in_win_mode(self) -> None:
        """win モード時に odds_multiplier_bands と regime_bands が含まれる"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_condition_stats(
            self._make_win_bets(), betting_target="win"
        )
        assert "odds_multiplier_bands" in result
        assert "regime_bands" in result

    def test_odds_multiplier_bands_absent_in_place_mode(self) -> None:
        """place モード時に odds_multiplier_bands と regime_bands が空リスト"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_condition_stats(
            self._make_win_bets(), betting_target="place"
        )
        assert result["odds_multiplier_bands"] == []
        assert result["regime_bands"] == []

    def test_odds_multiplier_bands_values(self) -> None:
        """オッズ倍率帯の値が正しい"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=Path("/tmp"))
        result = gen._compute_condition_stats(
            self._make_win_bets(), betting_target="win"
        )
        bands = result["odds_multiplier_bands"]
        # tanoddslow: 2.5, 5.0, 15.0, 8.0
        # bands: 1.0-3.0 (1), 3.0-10.0 (2), 10.0-30.0 (1)
        band_names = [b["band"] for b in bands]
        assert "1.0-3.0" in band_names
        assert "3.0-10.0" in band_names
        assert "10.0-30.0" in band_names


class TestSaveAiDiagnostics:
    """save_ai_diagnostics のテスト"""

    def _make_win_result(self, n_bets: int = 10) -> tuple[Any, list[dict[str, Any]]]:
        """win モード用の BacktestResult + bet_history を生成"""
        from backtest.engine import BacktestResult

        bets = []
        for i in range(n_bets):
            month = 1 + (i % 6)  # 1-6 月に分散
            bets.append({
                "race_id": f"2024{month:02d}{10 + i:02d}010101",
                "stake": 100.0,
                "result": 240.0 if i % 2 == 0 else 0.0,
                "surface": "turf" if i % 3 != 0 else "dirt",
                "kyori": 1200 + (i % 4) * 400,
                "popularity": 1 + (i % 8),
                "ev": 1.0 + (i % 5) * 0.2,
                "tanoddslow": 2.0 + i * 1.5,
                "regime": ["aggressive", "conservative", "collapsed"][i % 3],
                "race_date": f"2024-{month:02d}-{10 + i:02d}",
                "is_win": i % 2 == 0,
                "bankroll_after": 100000.0 + i * 100,
            })

        n_wins = sum(1 for b in bets if b["result"] > 0)
        total_return = sum(b["result"] for b in bets)
        result = BacktestResult(
            total_bets=len(bets),
            total_stake=sum(b["stake"] for b in bets),
            total_return=total_return,
            winning_bets=n_wins,
            total_roi=total_return / (len(bets) * 100.0),
            max_drawdown=0.05,
            final_bankroll=100000.0 + len(bets) * 100,
        )
        return result, bets

    def test_win_mode_produces_valid_json(self, tmp_path: Path) -> None:
        """win モードで有効な JSON が生成され、highlights が含まれる"""
        import json

        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        result, bets = self._make_win_result(n_bets=10)

        path = gen.save_ai_diagnostics(bets, result, betting_target="win")
        assert path is not None
        assert path.exists()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["meta"]["betting_target"] == "win"
        assert "summary" in data
        assert data["summary"]["total_bets"] == 10
        assert "highlights" in data
        assert "monthly_trend" in data["highlights"]

    def test_empty_bets_returns_none(self, tmp_path: Path) -> None:
        """空の bet リストでは None を返す"""
        from backtest.engine import BacktestResult
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        result = BacktestResult()

        path = gen.save_ai_diagnostics([], result, betting_target="win")
        assert path is None

    def test_place_mode_returns_none(self, tmp_path: Path) -> None:
        """place モードでは None を返す"""
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)
        result, bets = self._make_win_result(n_bets=5)

        path = gen.save_ai_diagnostics(bets, result, betting_target="place")
        assert path is None

    def test_trend_improving(self, tmp_path: Path) -> None:
        """後半ROIが前半の1.1倍以上なら improving"""
        import json

        from backtest.engine import BacktestResult
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)

        # 6ヶ月分のデータ: 前半ROI低、後半ROI高
        bets = []
        for i in range(30):
            month = 1 + i // 5  # 月1-6 (各5件)
            is_second_half = month >= 4
            bets.append({
                "race_id": f"2024{month:02d}{10 + i:02d}010101",
                "stake": 100.0,
                "result": 150.0 if is_second_half else 50.0,  # 後半が高い
                "surface": "turf",
                "kyori": 1200,
                "popularity": 3,
                "ev": 1.2,
                "tanoddslow": 3.0,
                "regime": "aggressive",
                "race_date": f"2024-{month:02d}-{10 + i:02d}",
                "is_win": is_second_half,
                "bankroll_after": 100000.0 + i * 100,
            })

        total_return = sum(b["result"] for b in bets)
        result = BacktestResult(
            total_bets=len(bets),
            total_stake=len(bets) * 100.0,
            total_return=total_return,
            winning_bets=sum(1 for b in bets if b["result"] > 0),
            total_roi=total_return / (len(bets) * 100.0),
            max_drawdown=0.05,
            final_bankroll=103000.0,
        )

        path = gen.save_ai_diagnostics(bets, result, betting_target="win")
        assert path is not None
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["highlights"]["monthly_trend"] == "improving"

    def test_trend_declining(self, tmp_path: Path) -> None:
        """後半ROIが前半の0.9倍以下なら declining"""
        import json

        from backtest.engine import BacktestResult
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)

        # 6ヶ月分: 前半ROI高、後半ROI低
        bets = []
        for i in range(30):
            month = 1 + i // 5
            is_first_half = month < 4
            bets.append({
                "race_id": f"2024{month:02d}{10 + i:02d}010101",
                "stake": 100.0,
                "result": 150.0 if is_first_half else 50.0,  # 前半が高い
                "surface": "turf",
                "kyori": 1200,
                "popularity": 3,
                "ev": 1.2,
                "tanoddslow": 3.0,
                "regime": "aggressive",
                "race_date": f"2024-{month:02d}-{10 + i:02d}",
                "is_win": is_first_half,
                "bankroll_after": 100000.0 + i * 100,
            })

        total_return = sum(b["result"] for b in bets)
        result = BacktestResult(
            total_bets=len(bets),
            total_stake=len(bets) * 100.0,
            total_return=total_return,
            winning_bets=sum(1 for b in bets if b["result"] > 0),
            total_roi=total_return / (len(bets) * 100.0),
            max_drawdown=0.05,
            final_bankroll=103000.0,
        )

        path = gen.save_ai_diagnostics(bets, result, betting_target="win")
        assert path is not None
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["highlights"]["monthly_trend"] == "declining"

    def test_trend_stable(self, tmp_path: Path) -> None:
        """前後半のROI差が小さいなら stable"""
        import json

        from backtest.engine import BacktestResult
        from backtest.report import BacktestReportGenerator

        gen = BacktestReportGenerator(output_dir=tmp_path)

        # 6ヶ月分: 前後半とも同程度のROI
        bets = []
        for i in range(30):
            month = 1 + i // 5
            bets.append({
                "race_id": f"2024{month:02d}{10 + i:02d}010101",
                "stake": 100.0,
                "result": 100.0,  # 全ベット同額返し → ROI=1.0
                "surface": "turf",
                "kyori": 1200,
                "popularity": 3,
                "ev": 1.2,
                "tanoddslow": 3.0,
                "regime": "aggressive",
                "race_date": f"2024-{month:02d}-{10 + i:02d}",
                "is_win": True,
                "bankroll_after": 100000.0 + i * 100,
            })

        total_return = sum(b["result"] for b in bets)
        result = BacktestResult(
            total_bets=len(bets),
            total_stake=len(bets) * 100.0,
            total_return=total_return,
            winning_bets=len(bets),
            total_roi=total_return / (len(bets) * 100.0),
            max_drawdown=0.0,
            final_bankroll=103000.0,
        )

        path = gen.save_ai_diagnostics(bets, result, betting_target="win")
        assert path is not None
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["highlights"]["monthly_trend"] == "stable"
