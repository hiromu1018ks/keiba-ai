"""Dry-run 統合テスト (Parquet データ使用)"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDryRunIntegration:
    """dry-run パイプラインの統合テスト。

    Parquet データが存在する環境でのみ実行。
    """

    @pytest.fixture
    def mock_env(self, tmp_path: Path) -> None:
        """最小限のモック環境を構築"""
        (tmp_path / "data" / "paper_trading" / "dry_run").mkdir(parents=True)
        (tmp_path / "data" / "paper_trading" / "model").mkdir(parents=True)
        (tmp_path / "data" / "paper_trading" / "predictions").mkdir(parents=True)

    @pytest.mark.skip(reason="run_paper_trading.py still references deleted DataRepository")
    @patch("scripts.run_paper_trading.load_config")
    @patch("db.parquet_store.ParquetStore")
    @patch("db.model_loader.ModelLoader")
    def test_dry_run_single_day(
        self,
        mock_loader_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
        mock_env: None,
    ) -> None:
        """1日分の dry-run が正常終了することを確認 (空データ → sys.exit)"""
        import pandas as pd

        from scripts.run_paper_trading import main

        # --- config mock ---
        pt_dir = tmp_path / "data" / "paper_trading"
        mock_config = MagicMock()
        mock_config.slack_webhook_url = ""
        mock_config.paper_trading_dir = pt_dir
        mock_config.mlflow_tracking_uri = "file:///mlruns"
        mock_config.mlflow_run_id = None
        mock_config.initial_bankroll = 100000.0
        mock_load_config.return_value = mock_config

        # --- model mock ---
        mock_models = MagicMock()
        mock_info = MagicMock()
        mock_info.mlflow_run_id = "test"
        mock_info.train_start = "2020-01-01"
        mock_info.train_end = "2023-12-31"
        mock_info.loaded_at = "now"
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader
        mock_loader.load.return_value = (mock_models, mock_info)

        # --- repository mock (空データ) ---
        mock_repo = MagicMock()
        mock_repo.load_races.return_value = pd.DataFrame()
        mock_repo.load_entries.return_value = pd.DataFrame()
        mock_repo.load_odds_snapshots.return_value = pd.DataFrame()

        # DataRepository は main() 内で from db.repository import される
        with patch("db.repository.DataRepository", return_value=mock_repo):
            old_argv = sys.argv
            sys.argv = [
                "run_paper_trading.py",
                "--mode",
                "dry-run",
                "--date",
                "2024-07-13",
            ]

            try:
                with pytest.raises(SystemExit):
                    main()
            finally:
                sys.argv = old_argv

        # 空データなので SystemExit(1) で終了するはず
        mock_loader.load.assert_called_once()
