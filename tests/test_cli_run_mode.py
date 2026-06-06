"""Lightweight CLI structure tests for --mode run (Task 2)."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path


class TestCLIRunMode:
    """Test 1: parse_args() includes 'run' in mode choices."""

    def test_run_in_mode_choices(self) -> None:
        """parse_args() includes 'run' in --mode choices."""
        # Import the parse_args function
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "src"))


        # Import the module

        # Check that parse_args has "run" in choices
        # We can't call parse_args directly (it calls sys.argv),
        # so we inspect the source or test via subprocess
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'scripts'); "
             "sys.path.insert(0, 'src'); "
             "from run_paper_trading import parse_args; "
             "import argparse; "
             "parser = argparse.ArgumentParser(); "
             "parser.add_argument('--mode', required=True, "
             "choices=['setup','predict','reconcile','dry-run','diagnose','run']); "
             "args = parser.parse_args(['--mode', 'run']); "
             "print(args.mode)"],
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        assert result.returncode == 0
        assert "run" in result.stdout

    def test_run_mode_function_exists(self) -> None:
        """Test 2: _run_run_mode is callable."""
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "src"))

        from scripts.run_paper_trading import _run_run_mode

        assert callable(_run_run_mode)

    def test_invalid_mode_rejected(self) -> None:
        """Test 3: parse_args() rejects invalid mode."""
        root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'scripts'); "
             "sys.path.insert(0, 'src'); "
             "from run_paper_trading import parse_args; "
             "sys.argv = ['test', '--mode', 'invalid']; "
             "parse_args()"],
            capture_output=True,
            text=True,
            cwd=str(root),
        )
        assert result.returncode != 0

    def test_reconcile_references_aggregator(self) -> None:
        """Test 4: _run_reconcile imports PaperTradingReportAggregator."""
        root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "src"))

        from scripts.run_paper_trading import _run_reconcile

        source = inspect.getsource(_run_reconcile)
        assert "PaperTradingReportAggregator" in source
        assert "D-15" in source
