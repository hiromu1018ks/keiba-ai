"""MultiYearReportGenerator のテスト"""

from __future__ import annotations

from pathlib import Path

from backtest.engine import BacktestResult


def _make_result(
    bets: int = 3,
    stake: float = 300.0,
    ret: float = 420.0,
) -> BacktestResult:
    return BacktestResult(
        total_bets=bets,
        total_stake=stake,
        total_return=ret,
        winning_bets=1,
        total_roi=ret / stake if stake > 0 else 0.0,
        max_drawdown=0.05,
        final_bankroll=100000 + ret - stake,
        bet_history=[
            {
                "race_id": "20240106061101",
                "bet_type": "place",
                "umaban": 5,
                "stake": 100.0,
                "odds": 2.3,
                "result": 230.0,
                "surface": "turf",
                "kyori": 1600,
                "ev": 1.25,
                "popularity": 3,
                "bankroll_after": 100130.0,
                "race_date": "2024-01-06",
                "jyocd": "06",
                "racenum": 11,
                "grade_code": "C",
                "race_name": "ポプリS",
                "bamei": "テスト馬",
                "kisyu": "テスト騎手",
                "kakuteijyuni": 2,
                "track_condition_code": 1,
                "p_place_pred": 0.65,
                "e_return_place_pred": 1.80,
                "top3_finishers": [
                    {"umaban": 8, "bamei": "1着馬", "kisyuryakusyo": "川田", "kakuteijyuni": 1},
                    {
                        "umaban": 5,
                        "bamei": "テスト馬",
                        "kisyuryakusyo": "テスト騎手",
                        "kakuteijyuni": 2,
                    },
                ],
            },
        ],
    )


class TestMultiYearHtmlGeneration:
    """マルチ年度HTML生成のテスト"""

    def test_html_contains_all_year_tabs(self, tmp_path: Path) -> None:
        """全年度タブが含まれる"""
        from backtest.report import MultiYearReportGenerator

        gen = MultiYearReportGenerator(output_dir=tmp_path)
        results = {2023: _make_result(), 2024: _make_result()}
        metadata = {
            2023: {"train_start": "2020-01-01", "train_end": "2022-12-31",
                   "test_start": "2023-01-01", "test_end": "2023-12-31"},
            2024: {"train_start": "2021-01-01", "train_end": "2023-12-31",
                   "test_start": "2024-01-01", "test_end": "2024-12-31"},
        }
        path = gen.generate(results, metadata)
        html = path.read_text(encoding="utf-8")

        assert "2023" in html
        assert "2024" in html
        assert "全体サマリー" in html

    def test_html_contains_bet_detail_tab(self, tmp_path: Path) -> None:
        """ベット明細タブが含まれる"""
        from backtest.report import MultiYearReportGenerator

        gen = MultiYearReportGenerator(output_dir=tmp_path)
        results = {2024: _make_result()}
        metadata = {2024: {"train_start": "2021-01-01", "train_end": "2023-12-31",
                           "test_start": "2024-01-01", "test_end": "2024-12-31"}}
        path = gen.generate(results, metadata)
        html = path.read_text(encoding="utf-8")

        assert "ベット明細" in html
        assert "テスト馬" in html
        assert "テスト騎手" in html

    def test_output_path(self, tmp_path: Path) -> None:
        """出力パスが multi_year_report.html"""
        from backtest.report import MultiYearReportGenerator

        gen = MultiYearReportGenerator(output_dir=tmp_path)
        results = {2024: _make_result()}
        metadata = {2024: {"train_start": "2021-01-01", "train_end": "2023-12-31",
                           "test_start": "2024-01-01", "test_end": "2024-12-31"}}
        path = gen.generate(results, metadata)

        assert path.name == "multi_year_report.html"
        assert path.parent == tmp_path

    def test_empty_results(self, tmp_path: Path) -> None:
        """空結果でもHTMLが生成される"""
        from backtest.report import MultiYearReportGenerator

        gen = MultiYearReportGenerator(output_dir=tmp_path)
        path = gen.generate({}, {})
        html = path.read_text(encoding="utf-8")

        assert path.exists()
        assert "データなし" in html
