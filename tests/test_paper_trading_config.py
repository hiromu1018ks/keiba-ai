"""PaperTradingConfig のテスト"""

from pathlib import Path


class TestPaperTradingConfig:
    def test_default_values(self) -> None:
        from paper_trading.config import PaperTradingConfig

        cfg = PaperTradingConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            everydb2_connection_string="postgresql://localhost/everydb2",
        )
        assert cfg.ev_threshold == 1.0
        assert cfg.initial_bankroll == 100000.0
        assert cfg.stake == 100.0
        assert cfg.watch_lead_minutes == 5
        assert cfg.retry_count == 3
        assert cfg.mlflow_run_id is None

    def test_paper_trading_dir_default(self) -> None:
        from paper_trading.config import PaperTradingConfig

        cfg = PaperTradingConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            everydb2_connection_string="postgresql://localhost/everydb2",
        )
        assert cfg.paper_trading_dir == Path("data/paper_trading")

    def test_custom_values(self) -> None:
        from paper_trading.config import PaperTradingConfig

        cfg = PaperTradingConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            everydb2_connection_string="postgresql://localhost/everydb2",
            ev_threshold=1.3,
            initial_bankroll=200000.0,
            stake=200.0,
            mlflow_run_id="abc123",
        )
        assert cfg.ev_threshold == 1.3
        assert cfg.initial_bankroll == 200000.0
        assert cfg.stake == 200.0
        assert cfg.mlflow_run_id == "abc123"

    def test_data_dir_structure(self, tmp_path: Path) -> None:
        from paper_trading.config import PaperTradingConfig

        cfg = PaperTradingConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            everydb2_connection_string="postgresql://localhost/everydb2",
            paper_trading_dir=tmp_path / "pt",
        )
        dirs = cfg.ensure_dirs()
        assert (dirs["predictions"]).exists()
        assert (dirs["bets"]).parent == cfg.paper_trading_dir
        assert (dirs["daily_summary"]).exists()
        assert (dirs["dry_run"]).exists()
        assert (dirs["model"]).exists()
