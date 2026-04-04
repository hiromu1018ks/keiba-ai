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

    def test_parse_args_predict_mode(self) -> None:
        sys.argv = ["run_paper_trading.py", "--mode", "predict", "--date", "2026-04-05"]
        from scripts.run_paper_trading import parse_args

        args = parse_args()
        assert args.mode == "predict"

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

    @patch("scripts.run_paper_trading._load_models")
    @patch("scripts.run_paper_trading.load_config")
    @patch("db.parquet_store.ParquetStore")
    def test_main_setup_mode(
        self,
        mock_store_cls: MagicMock,
        mock_load_config: MagicMock,
        mock_load_models: MagicMock,
    ) -> None:
        """setup モードが正しいコンポーネントを呼び出すことを確認"""
        from scripts.run_paper_trading import main

        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_models = MagicMock()
        mock_info = MagicMock()
        mock_load_models.return_value = (mock_models, mock_info)

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        with (
            patch("db.readers.load_races_from_db") as mock_load_races,
            patch("db.readers.load_entries_from_db") as mock_load_entries,
            patch("db.everydb2_queries.EveryDB2Queries"),
        ):
            import pandas as pd

            mock_load_races.return_value = pd.DataFrame()
            mock_load_entries.return_value = pd.DataFrame()

            sys.argv = [
                "run_paper_trading.py",
                "--mode",
                "setup",
                "--date",
                "2026-04-05",
            ]
            main()

            mock_load_models.assert_called_once()
            mock_load_races.assert_called_once()
