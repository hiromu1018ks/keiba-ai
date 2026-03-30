"""RaceWatcher のテスト"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class TestRaceWatcher:
    def test_watch_processes_scheduled_races(self, tmp_path: Path) -> None:
        from paper_trading.watcher import RaceWatcher

        mock_predictor = MagicMock()
        mock_everydb2 = MagicMock()
        mock_notifier = MagicMock()

        schedule = [
            {
                "race_id": "2026040510010101",
                "venue": "中山",
                "race_num": 1,
                "post_time": "10:05",
                "surface": "turf",
                "distance": 1200,
                "horses": ["馬1", "馬2"],
            },
        ]

        mock_predictor.predict_race.return_value = [
            {
                "race_id": "2026040510010101",
                "umaban": 1,
                "stake": 100.0,
                "odds": 2.4,
                "ev": 1.5,
                "bankroll_after": 99900.0,
            },
        ]
        mock_everydb2.get_horse_weights.return_value = MagicMock()
        mock_everydb2.get_latest_odds.return_value = MagicMock()

        # Create pre-computed features file
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"race_id": ["2026040510010101"]}).to_parquet(
            pred_dir / "20260405_pre.parquet", index=False
        )

        watcher = RaceWatcher(
            predictor=mock_predictor,
            everydb2=mock_everydb2,
            notifier=mock_notifier,
            predictions_dir=pred_dir,
        )

        # wait_until をモック (待機しない)
        with patch("paper_trading.watcher.wait_until"):
            watcher.watch(date(2026, 4, 5), schedule, bankroll=100000.0)

        mock_predictor.predict_race.assert_called_once()
        mock_notifier.send_prediction.assert_called_once()

    def test_watch_skips_processed_races(self, tmp_path: Path) -> None:
        """既に処理済みのレースはスキップ (冪等性)"""
        from paper_trading.watcher import RaceWatcher

        mock_predictor = MagicMock()
        mock_everydb2 = MagicMock()
        mock_notifier = MagicMock()

        schedule = [
            {
                "race_id": "2026040510010101",
                "venue": "中山",
                "race_num": 1,
                "post_time": "10:05",
                "surface": "turf",
                "distance": 1200,
                "horses": ["馬1"],
            },
        ]

        watcher = RaceWatcher(
            predictor=mock_predictor,
            everydb2=mock_everydb2,
            notifier=mock_notifier,
            predictions_dir=tmp_path / "predictions",
        )

        # 既に最終予測ファイルが存在
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"race_id": ["2026040510010101"]}).to_parquet(
            pred_dir / "20260405.parquet", index=False
        )

        with patch("paper_trading.watcher.wait_until"):
            watcher.watch(date(2026, 4, 5), schedule, bankroll=100000.0)

        mock_predictor.predict_race.assert_not_called()
