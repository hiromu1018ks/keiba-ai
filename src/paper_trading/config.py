"""Paper Trading 設定"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PaperTradingConfig:
    """Paper Trading の全設定"""

    # Slack
    slack_webhook_url: str

    # EveryDB2
    everydb2_connection_string: str

    # モデル
    mlflow_run_id: str | None = None
    mlflow_tracking_uri: str = "file:///mlruns"

    # ベット
    ev_threshold: float = 1.0
    initial_bankroll: float = 100000.0
    stake: float = 100.0

    # タイミング
    watch_lead_minutes: int = 5
    retry_count: int = 3
    retry_interval_seconds: int = 60

    # EveryDB2 クエリ
    query_timeout_seconds: int = 30

    # パス
    paper_trading_dir: Path = Path("data/paper_trading")

    def ensure_dirs(self) -> dict[str, Path]:
        """必要なディレクトリを作成してパスを返す"""
        dirs = {
            "predictions": self.paper_trading_dir / "predictions",
            "daily_summary": self.paper_trading_dir / "daily_summary",
            "dry_run": self.paper_trading_dir / "dry_run",
            "model": self.paper_trading_dir / "model",
            "bets": self.paper_trading_dir / "bets",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs
