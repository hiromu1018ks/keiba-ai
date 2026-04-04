"""src/ingestion/jvlink_fetcher.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from domain.models import Entry, Race


class TestJVLinkFetcher:
    def test_fetch_race_cards_returns_races(self) -> None:
        """指定日のレースカードを取得して Race リストを返す"""
        mock_store = MagicMock()

        from ingestion.jvlink_fetcher import JVLinkFetcher

        with patch(
            "ingestion.jvlink_fetcher.load_races",
            return_value=pd.DataFrame(
                {
                    "year": [2024],
                    "monthday": ["0325"],
                    "jyocd": ["01"],
                    "kaiji": ["01"],
                    "nichiji": ["01"],
                    "racenum": ["01"],
                    "trackcd": [10],
                    "kyori": [1600],
                    "tenkocd": [1],
                    "babacd": [1],
                    "syubetucd": ["0"],
                    "jyokencd": ["0"],
                    "gradecd": ["0"],
                    "syussotosu": [8],
                }
            ),
        ):
            fetcher = JVLinkFetcher(store=mock_store)
            races = fetcher.fetch_race_cards("2024-03-25")

        assert isinstance(races, list)
        assert len(races) == 1
        assert isinstance(races[0], Race)

    def test_fetch_results_returns_entries(self) -> None:
        """指定日の出走馬結果を取得して Entry リストを返す"""
        mock_store = MagicMock()

        from ingestion.jvlink_fetcher import JVLinkFetcher

        with patch(
            "ingestion.jvlink_fetcher.load_entries",
            return_value=pd.DataFrame(
                {
                    "race_id": ["2024032501010101"],
                    "umaban": [1],
                    "kettonum": ["1234"],
                    "kakuteijyuni": [1],
                    "odds": [3.5],
                    "ninki": [2],
                    "kyakusitukubun": [2],
                    "bataijyu": [480.0],
                    "zogenfugo": [1],
                    "zogensa": [4.0],
                    "kisyucode": ["00123"],
                    "chokyosicode": ["00456"],
                }
            ),
        ):
            fetcher = JVLinkFetcher(store=mock_store)
            entries = fetcher.fetch_results("2024-03-25")

        assert isinstance(entries, list)
        assert len(entries) == 1
        assert isinstance(entries[0], Entry)
        assert entries[0].is_winner

    def test_fetch_odds_snapshot_returns_dict(self) -> None:
        """特定レースの最新オッズを horse_no → odds の dict で返す"""
        mock_store = MagicMock()

        from ingestion.jvlink_fetcher import JVLinkFetcher

        with patch(
            "ingestion.jvlink_fetcher.load_odds_time_series",
            return_value=pd.DataFrame(
                {
                    "race_id": ["2024032501010101", "2024032501010101"],
                    "happyotime": ["03251500", "03251500"],
                    "umaban": [1, 3],
                    "tanodds": [3.5, 8.2],
                    "fukuoddslow": [1.6, 3.1],
                }
            ),
        ):
            fetcher = JVLinkFetcher(store=mock_store)
            snapshot = fetcher.fetch_odds_snapshot("2024032501010101")

        assert isinstance(snapshot, dict)
        assert snapshot[1] == 3.5
        assert snapshot[3] == 8.2

    def test_fetch_race_cards_empty_date(self) -> None:
        """レースがない日は空リストを返す"""
        mock_store = MagicMock()

        from ingestion.jvlink_fetcher import JVLinkFetcher

        with patch("ingestion.jvlink_fetcher.load_races", return_value=pd.DataFrame()):
            fetcher = JVLinkFetcher(store=mock_store)
            races = fetcher.fetch_race_cards("2024-01-01")

        assert races == []

    def test_fetch_results_empty_dataframe(self) -> None:
        """結果がない日は空リストを返す"""
        mock_store = MagicMock()

        from ingestion.jvlink_fetcher import JVLinkFetcher

        with patch("ingestion.jvlink_fetcher.load_entries", return_value=pd.DataFrame()):
            fetcher = JVLinkFetcher(store=mock_store)
            entries = fetcher.fetch_results("2024-01-01")

        assert entries == []

    def test_fetch_odds_snapshot_empty_dataframe(self) -> None:
        """オッズがないレースは空 dict を返す"""
        mock_store = MagicMock()

        from ingestion.jvlink_fetcher import JVLinkFetcher

        with patch("ingestion.jvlink_fetcher.load_odds_time_series", return_value=pd.DataFrame()):
            fetcher = JVLinkFetcher(store=mock_store)
            snapshot = fetcher.fetch_odds_snapshot("2024010101010101")

        assert snapshot == {}
