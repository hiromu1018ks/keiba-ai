"""PaperTradingReport のテスト"""

from pathlib import Path


class TestPaperTradingReport:
    def test_generate_creates_html(self, tmp_path: Path) -> None:
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(output_dir=tmp_path)
        bets = [
            {
                "race_id": "2026040510010101",
                "bet_type": "place",
                "umaban": 3,
                "stake": 100.0,
                "odds": 2.4,
                "result": 240.0,
                "surface": "turf",
                "distance": 1200,
                "ev": 1.5,
                "popularity": 3,
                "bankroll_after": 100140.0,
                "race_date": "2026-04-05",
                "horse_name": "テスト馬",
                "is_paper": True,
            },
        ]
        summary = {
            "n_bets": 1,
            "n_wins": 1,
            "cumulative_roi": 1.40,
            "max_dd": 0.0,
            "bankroll": 100140.0,
        }

        report_path = report.generate(bets, summary)
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Paper Trading" in content
        assert "テスト馬" in content

    def test_generate_with_empty_bets(self, tmp_path: Path) -> None:
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(output_dir=tmp_path)
        summary = {
            "n_bets": 0,
            "n_wins": 0,
            "cumulative_roi": 0.0,
            "max_dd": 0.0,
            "bankroll": 100000.0,
        }

        report_path = report.generate([], summary)
        assert report_path.exists()
