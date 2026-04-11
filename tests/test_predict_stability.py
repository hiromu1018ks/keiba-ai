"""tests/test_predict_stability.py -- ペーパートレード予測安定化のテスト"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# _extract_pre_post_odds() のテスト
# ---------------------------------------------------------------------------


def _make_odds_ts_df(entries: list[dict]) -> pd.DataFrame:
    """テスト用の odds_ts_df を構築するヘルパー。"""
    return pd.DataFrame(entries)


def _make_race_df(entries: list[dict]) -> pd.DataFrame:
    """テスト用の race_df を構築するヘルパー。"""
    return pd.DataFrame(entries)


class TestExtractPrePostOdds:
    """_extract_pre_post_odds() のテスト群。"""

    @pytest.fixture(autouse=True)
    def _setup_path(self) -> None:
        """スクリプトのパスを通す。"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, root)
        sys.path.insert(0, os.path.join(root, "src"))

    def _import_target(self):
        from scripts.run_paper_trading import _extract_pre_post_odds

        return _extract_pre_post_odds

    def test_basic_extraction(self) -> None:
        """発走5分前のスナップショットが正しく抽出される。"""
        extract = self._import_target()
        race_df = _make_race_df(
            [{"race_id": "20260411060101", "hassotime": 1005}]
        )
        ts_df = _make_odds_ts_df(
            [
                {
                    "race_id": "20260411060101",
                    "umaban": 1,
                    "year": 2026,
                    "happyotime": "04110955",
                    "tanodds": 100.0,
                    "fukuoddslow": 30.0,
                    "tanninki": 5,
                },
                {
                    "race_id": "20260411060101",
                    "umaban": 1,
                    "year": 2026,
                    "happyotime": "04111000",
                    "tanodds": 120.0,
                    "fukuoddslow": 35.0,
                    "tanninki": 5,
                },
                {
                    "race_id": "20260411060101",
                    "umaban": 1,
                    "year": 2026,
                    "happyotime": "04111003",
                    "tanodds": 150.0,
                    "fukuoddslow": 40.0,
                    "tanninki": 5,
                },
            ]
        )
        result = extract(ts_df, race_df, minutes_before=5)
        assert len(result) == 1
        assert result.iloc[0]["fukuoddslow"] == 35.0  # 10:00 の値

    def test_no_snapshot_before_cutoff_skips_race(self) -> None:
        """cutoff 以前のエントリがないレースは結果に含まれない。"""
        extract = self._import_target()
        race_df = _make_race_df(
            [{"race_id": "20260411060101", "hassotime": 1005}]
        )
        ts_df = _make_odds_ts_df(
            [
                {
                    "race_id": "20260411060101",
                    "umaban": 1,
                    "year": 2026,
                    "happyotime": "04111003",
                    "tanodds": 100.0,
                    "fukuoddslow": 30.0,
                    "tanninki": 5,
                },
            ]
        )
        result = extract(ts_df, race_df, minutes_before=5)
        assert len(result) == 0

    def test_stale_snapshot_excluded(self) -> None:
        """cutoff の60分以上前のスナップショットは除外される。"""
        extract = self._import_target()
        race_df = _make_race_df(
            [{"race_id": "20260411060101", "hassotime": 1005}]
        )
        ts_df = _make_odds_ts_df(
            [
                {
                    "race_id": "20260411060101",
                    "umaban": 1,
                    "year": 2026,
                    "happyotime": "04110830",
                    "tanodds": 100.0,
                    "fukuoddslow": 30.0,
                    "tanninki": 5,
                },
            ]
        )
        result = extract(ts_df, race_df, minutes_before=5, max_staleness_minutes=60)
        assert len(result) == 0

    def test_output_schema_compatible_with_build_all(self) -> None:
        """出力 DataFrame が必須5列を含む。"""
        extract = self._import_target()
        race_df = _make_race_df(
            [{"race_id": "20260411060101", "hassotime": 1005}]
        )
        ts_df = _make_odds_ts_df(
            [
                {
                    "race_id": "20260411060101",
                    "umaban": 1,
                    "year": 2026,
                    "happyotime": "04111000",
                    "tanodds": 100.0,
                    "fukuoddslow": 30.0,
                    "tanninki": 5,
                },
            ]
        )
        result = extract(ts_df, race_df, minutes_before=5)
        required = {"race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"}
        assert required.issubset(set(result.columns))

    def test_empty_inputs(self) -> None:
        """空の DataFrame が入力された場合、空の結果を返す。"""
        extract = self._import_target()
        result = extract(pd.DataFrame(), pd.DataFrame())
        assert len(result) == 0
        assert "race_id" in result.columns

    def test_boundary_inclusive_at_cutoff(self) -> None:
        """cutoff ちょうどのエントリは含まれる。"""
        extract = self._import_target()
        race_df = _make_race_df(
            [{"race_id": "20260411060101", "hassotime": 1005}]
        )
        ts_df = _make_odds_ts_df(
            [
                {
                    "race_id": "20260411060101",
                    "umaban": 1,
                    "year": 2026,
                    "happyotime": "04111000",
                    "tanodds": 100.0,
                    "fukuoddslow": 30.0,
                    "tanninki": 5,
                },
            ]
        )
        result = extract(ts_df, race_df, minutes_before=5)
        assert len(result) == 1  # 境界値は包含

    def test_happyotime_short_padding(self) -> None:
        """happyotime が7桁の場合でも正しくパースされる (zfill対応)。"""
        extract = self._import_target()
        race_df = _make_race_df(
            [{"race_id": "20260411060101", "hassotime": 1005}]
        )
        # "4110930" -> zfill(8) -> "04110930" -- represents 04/11 09:30
        # cutoff for 10:05 race is 10:00. min_cutoff = 10:00 - 60min = 09:00.
        # 09:30 is within [09:00, 10:00].
        ts_df = _make_odds_ts_df(
            [
                {
                    "race_id": "20260411060101",
                    "umaban": 1,
                    "year": 2026,
                    "happyotime": "4110930",
                    "tanodds": 100.0,
                    "fukuoddslow": 30.0,
                    "tanninki": 5,
                },
            ]
        )
        result = extract(ts_df, race_df, minutes_before=5)
        assert len(result) == 1
