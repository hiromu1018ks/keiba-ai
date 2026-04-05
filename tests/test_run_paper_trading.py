"""run_paper_trading.py CLI のテスト"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


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

    def test_parse_args_diagnose_mode(self) -> None:
        sys.argv = [
            "run_paper_trading.py",
            "--mode",
            "diagnose",
            "--start",
            "2024-07-01",
            "--end",
            "2024-07-31",
        ]
        from scripts.run_paper_trading import parse_args

        args = parse_args()
        assert args.mode == "diagnose"
        assert args.start == "2024-07-01"
        assert args.end == "2024-07-31"

    def test_diagnose_mode_is_in_choices(self) -> None:
        """--mode diagnose が argparse の選択肢に含まれていることを確認"""
        import subprocess

        result = subprocess.run(
            ["python", "scripts/run_paper_trading.py", "--mode", "diagnose", "--help"],
            capture_output=True,
            text=True,
        )
        assert "invalid choice" not in result.stderr.lower() or "diagnose" in result.stderr

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

    @patch("scripts.run_paper_trading._load_models")
    @patch("scripts.run_paper_trading.load_config")
    @patch("db.parquet_store.ParquetStore")
    def test_main_diagnose_mode(
        self,
        mock_store_cls: MagicMock,
        mock_load_config: MagicMock,
        mock_load_models: MagicMock,
    ) -> None:
        """diagnose モードが Parquet リーダー (EveryDB2 なし) を呼び出すことを確認"""
        from scripts.run_paper_trading import main

        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_models = MagicMock()
        mock_info = MagicMock()
        mock_load_models.return_value = (mock_models, mock_info)

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        with (
            patch("db.readers.load_races") as mock_load_races,
            patch("db.readers.load_entries") as mock_load_entries,
            patch("db.readers.load_odds_snapshots") as mock_load_odds,
        ):
            import pandas as pd

            mock_load_races.return_value = pd.DataFrame()

            sys.argv = [
                "run_paper_trading.py",
                "--mode",
                "diagnose",
                "--start",
                "2024-07-01",
                "--end",
                "2024-07-31",
            ]
            main()

            mock_load_races.assert_called_once()
            mock_load_entries.assert_called_once()  # called before empty check
            mock_load_odds.assert_called_once()


def test_diagnose_mode_no_everydb2_import():
    """_run_diagnose 関数が EveryDB2 リーダーをインポートしないことを確認"""
    # scripts ディレクトリをモジュールとしてロード
    import importlib.util
    import inspect

    spec = importlib.util.spec_from_file_location(
        "run_paper_trading",
        Path(__file__).resolve().parent.parent / "scripts" / "run_paper_trading.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # 実行時インポートをスキップ (DB接続不要)
    with patch.dict(
        "sys.modules",
        {
            "db.parquet_store": MagicMock(),
            "db.model_loader": MagicMock(),
            "paper_trading.config": MagicMock(),
            "backtest.race_predictor": MagicMock(),
            "features.feature_engine": MagicMock(),
            "models.submodel_manager": MagicMock(),
        },
    ):
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    source = inspect.getsource(mod._run_diagnose)
    assert "EveryDB2Queries" not in source, "_run_diagnose should not use EveryDB2!"
    assert "load_races_from_db" not in source, "_run_diagnose should not use EveryDB2 readers!"
    assert "load_entries_from_db" not in source, "_run_diagnose should not use EveryDB2 readers!"
    assert "load_races" in source  # Parquet版の load_races
    assert "load_entries" in source  # Parquet版の load_entries
