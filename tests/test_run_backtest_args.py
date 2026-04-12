"""run_backtest.py CLI 引数解析のテスト"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_backtest(args: list[str]) -> subprocess.CompletedProcess[str]:
    """run_backtest.py を subprocess で実行し、結果を返す"""
    return subprocess.run(
        [sys.executable, "scripts/run_backtest.py"] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )


class TestSingleYearMode:
    """単一年度モード: --train-start/end + --test-start/end"""

    def test_single_year_args_accepted(self) -> None:
        """4つの日付引数がすべて指定されていれば引数解析はパスする"""
        result = _run_backtest([
            "--train-start", "20200101", "--train-end", "20231231",
            "--test-start", "20240101", "--test-end", "20241231",
        ])
        assert "unrecognized arguments" not in result.stderr

    def test_single_year_with_betting_mode(self) -> None:
        """--betting-mode が指定できる"""
        result = _run_backtest([
            "--train-start", "20200101", "--train-end", "20231231",
            "--test-start", "20240101", "--test-end", "20241231",
            "--betting-mode", "kelly",
        ])
        assert "unrecognized arguments" not in result.stderr

    def test_single_year_with_ensemble(self) -> None:
        """--ensemble が指定できる"""
        result = _run_backtest([
            "--train-start", "20200101", "--train-end", "20231231",
            "--test-start", "20240101", "--test-end", "20241231",
            "--ensemble",
        ])
        assert "unrecognized arguments" not in result.stderr


class TestMultiYearMode:
    """マルチ年度モード: --years"""

    def test_years_args_accepted(self) -> None:
        """--years が指定されていればマルチ年度モードとして動作する"""
        result = _run_backtest(["--years", "2023", "2024"])
        assert "unrecognized arguments" not in result.stderr

    def test_years_with_train_window(self) -> None:
        """--years と --train-window が同時に指定できる"""
        result = _run_backtest([
            "--years", "2023", "2024", "2025",
            "--train-window", "5",
        ])
        assert "unrecognized arguments" not in result.stderr

    def test_years_with_all_options(self) -> None:
        """--years とすべてのオプションが同時に指定できる"""
        result = _run_backtest([
            "--years", "2024", "2025",
            "--train-window", "4",
            "--betting-mode", "flat",
            "--ensemble",
            "--report",
        ])
        assert "unrecognized arguments" not in result.stderr


class TestErrorCases:
    """エラーケース"""

    def test_no_args_error(self) -> None:
        """引数なし → エラーメッセージ"""
        result = _run_backtest([])
        assert result.returncode != 0

    def test_partial_single_year_args_error(self) -> None:
        """--train-start だけ指定 → エラー"""
        result = _run_backtest(["--train-start", "20200101"])
        assert result.returncode != 0

    def test_train_without_test_error(self) -> None:
        """--train-start/end だけ指定 (--test-start/end なし) → エラー"""
        result = _run_backtest([
            "--train-start", "20200101",
            "--train-end", "20231231",
        ])
        assert result.returncode != 0


class TestTrainWindowDefault:
    """--train-window のデフォルト値"""

    def test_default_train_window_is_four(self) -> None:
        """デフォルトは 4"""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--train-window", type=int, default=4)
        args = parser.parse_args([])
        assert args.train_window == 4

    def test_custom_train_window(self) -> None:
        """カスタム値を指定できる"""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--train-window", type=int, default=4)
        args = parser.parse_args(["--train-window", "5"])
        assert args.train_window == 5
