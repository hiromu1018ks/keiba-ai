"""src/ingestion/jvlink_fetcher.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from domain.models import Entry, Race


class TestJVLinkFetcher:
    def test_fetch_race_cards_returns_races(self) -> None:
        """指定日のレースカードを取得して Race リストを返す"""
        mock_db = MagicMock()
        mock_db.load_races.return_value = pd.DataFrame(
            {
                "year": [2024], "month_day": ["0325"], "jyo_cd": ["01"],
                "kaiji": ["01"], "nichiji": ["01"], "race_num": ["01"],
                "track_cd": [10], "distance": [1600], "tenko_cd": [1],
                "baba_cd": [1], "syubetu_cd": ["0"], "jyoken_cd": ["0"],
                "grade_cd": ["0"], "field_size": [8],
            }
        )

        from ingestion.jvlink_fetcher import JVLinkFetcher
        fetcher = JVLinkFetcher(db=mock_db)
        races = fetcher.fetch_race_cards("2024-03-25")

        assert isinstance(races, list)
        assert len(races) == 1
        assert isinstance(races[0], Race)

    def test_fetch_results_returns_entries(self) -> None:
        """指定日の出走馬結果を取得して Entry リストを返す"""
        mock_db = MagicMock()
        mock_db.load_entries_with_results.return_value = pd.DataFrame(
            {
                "race_id": ["2024032501010101"],
                "umaban": [1], "ketto_num": ["1234"], "finish_pos": [1],
                "win_odds_actual": [3.5], "popularity_rank": [2],
                "running_style": [2], "ba_taijyu": [480.0],
                "zogen_fugo": [1], "zogen_sa": [4.0],
                "kisyu_code": ["00123"], "chokyosi_code": ["00456"],
            }
        )

        from ingestion.jvlink_fetcher import JVLinkFetcher
        fetcher = JVLinkFetcher(db=mock_db)
        entries = fetcher.fetch_results("2024-03-25")

        assert isinstance(entries, list)
        assert len(entries) == 1
        assert isinstance(entries[0], Entry)
        assert entries[0].is_winner

    def test_fetch_odds_snapshot_returns_dict(self) -> None:
        """特定レースの最新オッズを horse_no → odds の dict で返す"""
        mock_db = MagicMock()
        mock_db.load_odds_time_series.return_value = pd.DataFrame(
            {
                "race_id": ["2024032501010101", "2024032501010101"],
                "happyo_time": ["03251500", "03251500"],
                "umaban": [1, 3],
                "tan_odds": [3.5, 8.2],
                "fuku_odds": [1.6, 3.1],
            }
        )

        from ingestion.jvlink_fetcher import JVLinkFetcher
        fetcher = JVLinkFetcher(db=mock_db)
        snapshot = fetcher.fetch_odds_snapshot("2024032501010101")

        assert isinstance(snapshot, dict)
        assert snapshot[1] == 3.5
        assert snapshot[3] == 8.2

    def test_fetch_race_cards_empty_date(self) -> None:
        """レースがない日は空リストを返す"""
        mock_db = MagicMock()
        mock_db.load_races.return_value = pd.DataFrame()

        from ingestion.jvlink_fetcher import JVLinkFetcher
        fetcher = JVLinkFetcher(db=mock_db)
        races = fetcher.fetch_race_cards("2024-01-01")

        assert races == []

    def test_fetch_results_empty_dataframe(self) -> None:
        """結果がない日は空リストを返す"""
        mock_db = MagicMock()
        mock_db.load_entries_with_results.return_value = pd.DataFrame()

        from ingestion.jvlink_fetcher import JVLinkFetcher
        fetcher = JVLinkFetcher(db=mock_db)
        entries = fetcher.fetch_results("2024-01-01")

        assert entries == []

    def test_fetch_odds_snapshot_empty_dataframe(self) -> None:
        """オッズがないレースは空 dict を返す"""
        mock_db = MagicMock()
        mock_db.load_odds_time_series.return_value = pd.DataFrame()

        from ingestion.jvlink_fetcher import JVLinkFetcher
        fetcher = JVLinkFetcher(db=mock_db)
        snapshot = fetcher.fetch_odds_snapshot("2024010101010101")

        assert snapshot == {}
