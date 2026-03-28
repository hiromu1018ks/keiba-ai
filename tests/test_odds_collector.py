"""src/ingestion/odds_collector.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

from db.repository import DataRepository
from domain.models import Race


def _make_race() -> Race:
    return Race(
        year=2024,
        month_day="0325",
        jyo_cd="01",
        kaiji="01",
        nichiji="01",
        race_num="01",
        track_cd=10,
        distance=1600,
        tenko_cd=1,
        baba_cd=1,
        syubetu_cd="0",
        jyoken_cd="0",
        grade_cd="0",
        field_size=8,
    )


class TestOddsCollector:
    def test_collect_t3_snapshot_returns_odds_dict(self) -> None:
        """t-3min スナップショットを取得して horse_no → odds を返す"""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_odds_snapshot.return_value = {1: 3.5, 3: 8.2, 5: 12.0}

        from ingestion.odds_collector import OddsCollector

        collector = OddsCollector(fetcher=mock_fetcher)
        snapshot = collector.collect_t3_snapshot("2024032501010101")

        assert snapshot == {1: 3.5, 3: 8.2, 5: 12.0}
        mock_fetcher.fetch_odds_snapshot.assert_called_once_with("2024032501010101")

    def test_collect_t2_snapshot_returns_odds_dict(self) -> None:
        """t-2min スナップショットを取得（ログ用途）"""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_odds_snapshot.return_value = {1: 3.2, 3: 7.8}

        from ingestion.odds_collector import OddsCollector

        collector = OddsCollector(fetcher=mock_fetcher)
        snapshot = collector.collect_t2_snapshot("2024032501010101")

        assert snapshot == {1: 3.2, 3: 7.8}

    def test_store_snapshot_saves_to_db(self) -> None:
        """スナップショットを DB に保存"""
        mock_fetcher = MagicMock()
        mock_repo = MagicMock(spec=DataRepository)
        mock_fetcher.fetch_odds_snapshot.return_value = {1: 3.5}

        from ingestion.odds_collector import OddsCollector

        collector = OddsCollector(fetcher=mock_fetcher, repo=mock_repo)
        collector.store_snapshot("2024032501010101", "t3", {1: 3.5})

        mock_repo.save_predictions.assert_called_once()

    def test_get_odds_change_computes_rate(self) -> None:
        """t-10min → t-3min の変化率を計算"""
        from ingestion.odds_collector import OddsCollector

        collector = OddsCollector(fetcher=MagicMock())

        rate = collector.get_odds_change_rate(5.0, 3.5)
        assert abs(rate - 0.30) < 1e-6  # (5.0 - 3.5) / 5.0 = 0.30

    def test_get_odds_change_rate_zero_odds(self) -> None:
        """オッズが0の場合は None を返す"""
        from ingestion.odds_collector import OddsCollector

        collector = OddsCollector(fetcher=MagicMock())

        assert collector.get_odds_change_rate(0.0, 3.5) is None
        assert collector.get_odds_change_rate(5.0, 0.0) is None

    def test_store_snapshot_without_db_does_nothing(self) -> None:
        """repo なしでは store_snapshot は何もしない"""
        mock_fetcher = MagicMock()

        from ingestion.odds_collector import OddsCollector

        collector = OddsCollector(fetcher=mock_fetcher, repo=None)
        # エラーなく実行できることを確認
        collector.store_snapshot("2024032501010101", "t3", {1: 3.5})
