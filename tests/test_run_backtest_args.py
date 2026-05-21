"""run_backtest.py CLI 引数解析のテスト"""

from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd


def _import_build_parser():
    """build_parser をインポートするヘルパー"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "src"))

    from scripts.run_backtest import build_parser

    return build_parser


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
        build_parser = _import_build_parser()
        args = build_parser().parse_args(
            [
                "--train-start",
                "20200101",
                "--train-end",
                "20231231",
                "--test-start",
                "20240101",
                "--test-end",
                "20241231",
            ]
        )
        assert args.train_start == "20200101"
        assert args.test_end == "20241231"

    def test_single_year_with_betting_mode(self) -> None:
        """--betting-mode が指定できる"""
        build_parser = _import_build_parser()
        args = build_parser().parse_args(
            [
                "--train-start",
                "20200101",
                "--train-end",
                "20231231",
                "--test-start",
                "20240101",
                "--test-end",
                "20241231",
                "--betting-mode",
                "kelly",
            ]
        )
        assert args.betting_mode == "kelly"

    def test_single_year_with_ensemble(self) -> None:
        """--ensemble が指定できる"""
        build_parser = _import_build_parser()
        args = build_parser().parse_args(
            [
                "--train-start",
                "20200101",
                "--train-end",
                "20231231",
                "--test-start",
                "20240101",
                "--test-end",
                "20241231",
                "--ensemble",
            ]
        )
        assert args.ensemble is True


class TestMultiYearMode:
    """マルチ年度モード: --years"""

    def test_years_args_accepted(self) -> None:
        """--years が指定されていればマルチ年度モードとして動作する"""
        build_parser = _import_build_parser()
        args = build_parser().parse_args(["--years", "2023", "2024"])
        assert args.years == [2023, 2024]

    def test_years_with_train_window(self) -> None:
        """--years と --train-window が同時に指定できる"""
        build_parser = _import_build_parser()
        args = build_parser().parse_args(
            ["--years", "2023", "2024", "2025", "--train-window", "5"]
        )
        assert args.years == [2023, 2024, 2025]
        assert args.train_window == 5

    def test_years_with_all_options(self) -> None:
        """--years とすべてのオプションが同時に指定できる"""
        build_parser = _import_build_parser()
        args = build_parser().parse_args(
            [
                "--years",
                "2024",
                "2025",
                "--train-window",
                "4",
                "--betting-mode",
                "flat",
                "--ensemble",
                "--report",
            ]
        )
        assert args.years == [2024, 2025]
        assert args.ensemble is True
        assert args.report is True


class TestErrorCases:
    """エラーケース (subprocess で実際のスクリプトを実行)"""

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
        result = _run_backtest(
            [
                "--train-start",
                "20200101",
                "--train-end",
                "20231231",
            ]
        )
        assert result.returncode != 0


class TestTrainWindowDefault:
    """--train-window のデフォルト値"""

    def test_default_train_window_is_four(self) -> None:
        """デフォルトは 4"""
        build_parser = _import_build_parser()
        args = build_parser().parse_args([])
        assert args.train_window == 4

    def test_custom_train_window(self) -> None:
        """カスタム値を指定できる"""
        build_parser = _import_build_parser()
        args = build_parser().parse_args(["--years", "2024", "--train-window", "5"])
        assert args.train_window == 5


class TestBettingTargetArg:
    """--betting-target CLI 引数のテスト"""

    def test_default_betting_target_is_win(self) -> None:
        """デフォルトの --betting-target は 'win'"""
        build_parser = _import_build_parser()
        args = build_parser().parse_args(
            [
                "--train-start", "20200101",
                "--train-end", "20231231",
                "--test-start", "20240101",
                "--test-end", "20241231",
            ]
        )
        assert args.betting_target == "win"

    def test_betting_target_place(self) -> None:
        """--betting-target place を指定できる"""
        build_parser = _import_build_parser()
        args = build_parser().parse_args(
            [
                "--train-start", "20200101",
                "--train-end", "20231231",
                "--test-start", "20240101",
                "--test-end", "20241231",
                "--betting-target", "place",
            ]
        )
        assert args.betting_target == "place"

    def test_betting_target_wide(self) -> None:
        """--betting-target wide を指定できる"""
        build_parser = _import_build_parser()
        args = build_parser().parse_args(
            [
                "--train-start", "20200101",
                "--train-end", "20231231",
                "--test-start", "20240101",
                "--test-end", "20241231",
                "--betting-target", "wide",
            ]
        )
        assert args.betting_target == "wide"

    def test_betting_target_invalid_rejected(self) -> None:
        """--betting-target invalid は引数エラー"""
        build_parser = _import_build_parser()
        import pytest

        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "--train-start", "20200101",
                    "--train-end", "20231231",
                    "--test-start", "20240101",
                    "--test-end", "20241231",
                    "--betting-target", "invalid",
                ]
            )


class TestYearParquetOutput:
    """年度別 predictions parquet 出力のテスト"""

    def test_save_year_parquet_marks_actual_bets_from_stake(self, tmp_path, monkeypatch) -> None:
        """is_bet 互換列は is_actual_bet == stake.notna() と一致する"""
        import scripts.run_backtest as run_backtest
        from backtest.engine import BacktestResult

        monkeypatch.setattr(run_backtest, "ROOT", str(tmp_path))
        diag_dir = tmp_path / "data" / "backtest"
        diag_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 2],
                "is_bet": [True, True],
            }
        ).to_csv(diag_dir / "bt_2024_horse_diagnostics.csv", index=False)

        result = BacktestResult(
            bet_history=[
                {
                    "race_id": "R1",
                    "umaban": 2,
                    "stake": 100.0,
                    "result": 0.0,
                    "odds": 5.0,
                    "final_odds": 5.0,
                }
            ]
        )

        run_backtest.save_year_parquet(2024, result)

        df = pd.read_parquet(diag_dir / "predictions" / "2024.parquet")
        actual = df.sort_values("umaban")["is_actual_bet"].tolist()
        compat = df.sort_values("umaban")["is_bet"].tolist()
        assert actual == [False, True]
        assert compat == actual
