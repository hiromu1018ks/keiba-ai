"""odds_extractor のテスト"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


class TestExtractPrePostOdds:
    """extract_pre_post_odds のテスト"""

    def test_basic_extraction(self) -> None:
        """発走5分前のオッズを正しく抽出する"""
        from db.odds_extractor import extract_pre_post_odds

        race_df = pd.DataFrame(
            {
                "race_id": ["20250401110101"],
                "hassotime": [930],  # 09:30 発走
            }
        )
        odds_ts_df = pd.DataFrame(
            {
                "race_id": ["20250401110101"] * 3,
                "umaban": [1, 1, 1],
                "year": [2025, 2025, 2025],
                "happyotime": ["04010920", "04010925", "04010930"],
                "tanodds": [5.0, 4.8, 4.5],
                "fukuoddslow": [1.3, 1.3, 1.2],
                "tanninki": [3, 3, 3],
            }
        )
        # cutoff = 09:30 - 5min = 09:25
        # valid entries: 09:20 (min_cutoff=08:25以内) and 09:25
        # latest = 09:25
        now = datetime(2025, 4, 1, 12, 0)
        result = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5, _now=now)

        assert len(result) == 1
        assert result.iloc[0]["fukuoddslow"] == 1.3  # 09:25時点の値
        assert result.iloc[0]["tanodds"] == 4.8

    def test_empty_odds_returns_empty(self) -> None:
        """空の時系列データは空DataFrameを返す"""
        from db.odds_extractor import extract_pre_post_odds

        result = extract_pre_post_odds(
            pd.DataFrame(),
            pd.DataFrame({"race_id": ["20250401110101"], "hassotime": [930]}),
        )
        assert result.empty
        assert "fukuoddslow" in result.columns

    def test_no_valid_entries_returns_empty(self) -> None:
        """有効なエントリがない場合は空DataFrameを返す"""
        from db.odds_extractor import extract_pre_post_odds

        race_df = pd.DataFrame({"race_id": ["20250401110101"], "hassotime": [930]})
        # cutoff = 09:25, このデータは09:25より後なので除外
        odds_ts_df = pd.DataFrame(
            {
                "race_id": ["20250401110101"],
                "umaban": [1],
                "year": [2025],
                "happyotime": ["04010930"],  # 09:30 は cutoff 09:25 より後
                "tanodds": [4.5],
                "fukuoddslow": [1.2],
                "tanninki": [3],
            }
        )
        now = datetime(2025, 4, 1, 12, 0)
        result = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5, _now=now)
        assert result.empty

    def test_multiple_horses_per_race(self) -> None:
        """1レース複数馬の最新エントリを取得"""
        from db.odds_extractor import extract_pre_post_odds

        race_df = pd.DataFrame({"race_id": ["20250401110101"], "hassotime": [930]})
        odds_ts_df = pd.DataFrame(
            {
                "race_id": ["20250401110101"] * 4,
                "umaban": [1, 1, 2, 2],
                "year": [2025, 2025, 2025, 2025],
                "happyotime": ["04010920", "04010925", "04010920", "04010925"],
                "tanodds": [5.0, 4.8, 10.0, 9.5],
                "fukuoddslow": [1.3, 1.3, 2.5, 2.4],
                "tanninki": [3, 3, 7, 7],
            }
        )
        now = datetime(2025, 4, 1, 12, 0)
        result = extract_pre_post_odds(odds_ts_df, race_df, minutes_before=5, _now=now)

        assert len(result) == 2
        horse1 = result[result["umaban"] == 1].iloc[0]
        horse2 = result[result["umaban"] == 2].iloc[0]
        assert horse1["fukuoddslow"] == 1.3  # 09:25時点
        assert horse2["fukuoddslow"] == 2.4  # 09:25時点
