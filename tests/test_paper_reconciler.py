"""PaperReconciler のテスト"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd


class TestPaperReconciler:
    def test_reconcile_settles_winning_bets(self, tmp_path: Path) -> None:
        from paper_trading.reconciler import PaperReconciler

        mock_repo = MagicMock()
        mock_everydb2 = MagicMock()
        mock_everydb2.get_race_results.return_value = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "umaban": [3],
                "kakuteijyuni": [2],
                "place_pay": [240.0],
                "place_odds": [2.4],
                "horse_name": ["テスト馬"],
            }
        )

        reconciler = PaperReconciler(
            store=mock_repo,
            bets_path=tmp_path / "bets.parquet",
            everydb2=mock_everydb2,
        )

        # 既存のベット履歴 (未確定)
        existing_bets = pd.DataFrame(
            [
                {
                    "race_id": "2026040510010101",
                    "bet_type": "place",
                    "umaban": 3,
                    "stake": 100.0,
                    "odds": 2.4,
                    "result": 0.0,
                    "surface": "turf",
                    "distance": 1200,
                    "ev": 1.5,
                    "popularity": 3,
                    "bankroll_after": 99900.0,
                    "race_date": pd.Timestamp("2026-04-05"),
                    "horse_name": "テスト馬",
                    "is_paper": True,
                }
            ]
        )
        existing_bets.to_parquet(tmp_path / "bets.parquet", index=False)

        result = reconciler.reconcile(date(2026, 4, 5))

        assert result["n_settled"] == 1
        assert result["n_wins"] == 1

    def test_reconcile_idempotent(self, tmp_path: Path) -> None:
        """重複実行時は既存レコードをスキップ (race_id + umaban で判定)"""
        from paper_trading.reconciler import PaperReconciler

        mock_everydb2 = MagicMock()
        mock_everydb2.get_race_results.return_value = pd.DataFrame()

        reconciler = PaperReconciler(
            store=MagicMock(),
            bets_path=tmp_path / "bets.parquet",
            everydb2=mock_everydb2,
        )

        # 既に確定済みのベット (result > 0 の勝ちケース)
        existing_bets = pd.DataFrame(
            [
                {
                    "race_id": "2026040510010101",
                    "bet_type": "place",
                    "umaban": 3,
                    "stake": 100.0,
                    "odds": 2.4,
                    "result": 240.0,  # 既に確定
                    "surface": "turf",
                    "distance": 1200,
                    "ev": 1.5,
                    "popularity": 3,
                    "bankroll_after": 100140.0,
                    "race_date": pd.Timestamp("2026-04-05"),
                    "horse_name": "テスト馬",
                    "is_paper": True,
                }
            ]
        )
        existing_bets.to_parquet(tmp_path / "bets.parquet", index=False)

        result = reconciler.reconcile(date(2026, 4, 5))
        assert result["n_settled"] == 0  # スキップされる

    def test_reconcile_uses_actual_payout_when_available(self, tmp_path: Path) -> None:
        """payfukusyopay がある場合は実際の配当を使用する (100円あたりの円 / 100)"""
        from paper_trading.reconciler import PaperReconciler

        mock_repo = MagicMock()
        mock_everydb2 = MagicMock()
        # payfukusyopay=110 → 100円betで110円払戻 → multiplier=1.1
        mock_everydb2.get_race_results.return_value = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "umaban": [3],
                "kakuteijyuni": [2],
                "place_pay": [240.0],
                "place_odds": [2.4],
                "horse_name": ["テスト馬"],
                "payfukusyopay": [110],  # 実際の配当: 100円→110円
            }
        )

        reconciler = PaperReconciler(
            store=mock_repo,
            bets_path=tmp_path / "bets.parquet",
            everydb2=mock_everydb2,
        )

        # odds=2.4 だが payfukusyopay=110 が優先されるべき
        existing_bets = pd.DataFrame(
            [
                {
                    "race_id": "2026040510010101",
                    "bet_type": "place",
                    "umaban": 3,
                    "stake": 100.0,
                    "odds": 2.4,  # 発走前オッズ (使用されないはず)
                    "result": 0.0,
                    "surface": "turf",
                    "distance": 1200,
                    "ev": 1.5,
                    "popularity": 3,
                    "bankroll_after": 99900.0,
                    "race_date": pd.Timestamp("2026-04-05"),
                    "horse_name": "テスト馬",
                    "is_paper": True,
                }
            ]
        )
        existing_bets.to_parquet(tmp_path / "bets.parquet", index=False)

        result = reconciler.reconcile(date(2026, 4, 5))

        assert result["n_settled"] == 1
        assert result["n_wins"] == 1

        # 払戻を確認: stake(100) * payfukusyopay(110)/100 = 110円
        bets_df = pd.read_parquet(tmp_path / "bets.parquet")
        assert bets_df.iloc[0]["result"] == 110.0
