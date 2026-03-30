"""run_paper_trading.py CLI のテスト"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRunPaperTradingCLI:
    def test_parse_args_setup_mode(self) -> None:
        sys.argv = ["run_paper_trading.py", "--mode", "setup", "--date", "2026-04-05"]
        from scripts.run_paper_trading import parse_args

        args = parse_args()
        assert args.mode == "setup"
        assert args.date == "2026-04-05"

    def test_parse_args_watch_mode(self) -> None:
        sys.argv = ["run_paper_trading.py", "--mode", "watch", "--date", "2026-04-05"]
        from scripts.run_paper_trading import parse_args

        args = parse_args()
        assert args.mode == "watch"

    def test_parse_args_reconcile_mode(self) -> None:
        sys.argv = ["run_paper_trading.py", "--mode", "reconcile", "--date", "2026-04-05"]
        from scripts.run_paper_trading import parse_args

        args = parse_args()
        assert args.mode == "reconcile"

    def test_parse_args_dry_run_mode(self) -> None:
        sys.argv = ["run_paper_trading.py", "--mode", "dry-run", "--date", "2024-07-13"]
        from scripts.run_paper_trading import parse_args

        args = parse_args()
        assert args.mode == "dry-run"
        assert args.date == "2024-07-13"

    def test_parse_args_dry_run_range(self) -> None:
        sys.argv = [
            "run_paper_trading.py",
            "--mode",
            "dry-run",
            "--start",
            "2024-07-01",
            "--end",
            "2024-07-31",
        ]
        from scripts.run_paper_trading import parse_args

        args = parse_args()
        assert args.mode == "dry-run"
        assert args.start == "2024-07-01"
        assert args.end == "2024-07-31"

    @patch("scripts.run_paper_trading.load_config")
    @patch("db.parquet_store.ParquetStore")
    @patch("db.model_loader.ModelLoader")
    def test_main_setup_mode(
        self,
        mock_loader_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """setup モードが正しいコンポーネントを呼び出すことを確認"""
        from scripts.run_paper_trading import main

        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_models = MagicMock()
        mock_info = MagicMock()
        mock_info.mlflow_run_id = "test-run-123"
        mock_info.train_start = "2020-01-01"
        mock_info.train_end = "2023-12-31"
        mock_info.loaded_at = "2026-04-05T10:00:00"
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader
        mock_loader.load.return_value = (mock_models, mock_info)

        with (
            patch("backtest.race_predictor.RacePredictor"),
            patch("paper_trading.predictor.PaperPredictor") as mock_pred_cls,
            patch("db.everydb2_queries.EveryDB2Queries"),
        ):
            mock_pred = MagicMock()
            mock_pred_cls.return_value = mock_pred
            mock_pred.setup.return_value = []

            sys.argv = [
                "run_paper_trading.py",
                "--mode",
                "setup",
                "--date",
                "2026-04-05",
            ]
            main()

            mock_pred.setup.assert_called_once()

    @patch("scripts.run_paper_trading.load_config")
    @patch("db.parquet_store.ParquetStore")
    @patch("db.model_loader.ModelLoader")
    def test_main_watch_mode(
        self,
        mock_loader_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        """watch モードがスケジュールなしでエラー終了することを確認"""
        from scripts.run_paper_trading import main

        pt_dir = tmp_path / "paper_trading"
        pt_dir.mkdir()
        (pt_dir / "model").mkdir()

        mock_config = MagicMock()
        mock_config.slack_webhook_url = ""
        mock_config.paper_trading_dir = pt_dir
        mock_config.mlflow_tracking_uri = "file:///mlruns"
        mock_load_config.return_value = mock_config

        mock_models = MagicMock()
        mock_info = MagicMock()
        mock_info.mlflow_run_id = "test-run-123"
        mock_info.train_start = "2020-01-01"
        mock_info.train_end = "2023-12-31"
        mock_info.loaded_at = "2026-04-05T10:00:00"
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader
        mock_loader.load.return_value = (mock_models, mock_info)

        with (
            patch("backtest.race_predictor.RacePredictor"),
            patch("paper_trading.predictor.PaperPredictor"),
            patch("db.everydb2_queries.EveryDB2Queries"),
        ):
            # schedule.json が存在しないので sys.exit(1)
            with pytest.raises(SystemExit):
                sys.argv = [
                    "run_paper_trading.py",
                    "--mode",
                    "watch",
                    "--date",
                    "2026-04-05",
                ]
                main()
