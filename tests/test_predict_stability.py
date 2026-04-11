"""tests/test_predict_stability.py -- ペーパートレード予測安定化のテスト"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# extract_pre_post_odds() のテスト
# ---------------------------------------------------------------------------


def _make_odds_ts_df(entries: list[dict]) -> pd.DataFrame:
    """テスト用の odds_ts_df を構築するヘルパー。"""
    return pd.DataFrame(entries)


def _make_race_df(entries: list[dict]) -> pd.DataFrame:
    """テスト用の race_df を構築するヘルパー。"""
    return pd.DataFrame(entries)


# テスト内で一律使う「現在時刻」: レース日(2026-04-11)の12:00。
# これにより、hassotime=1005 (10:05発走 → cutoff=10:00) は常に過去。
_FIXED_NOW = datetime(2026, 4, 11, 12, 0)


class TestExtractPrePostOdds:
    """extract_pre_post_odds() のテスト群。"""

    @pytest.fixture(autouse=True)
    def _setup_path(self) -> None:
        """スクリプトのパスを通す。"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, root)
        sys.path.insert(0, os.path.join(root, "src"))

    def _import_target(self):
        from db.odds_extractor import extract_pre_post_odds

        return extract_pre_post_odds

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
        result = extract(ts_df, race_df, minutes_before=5, _now=_FIXED_NOW)
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
        result = extract(ts_df, race_df, minutes_before=5, _now=_FIXED_NOW)
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
        result = extract(ts_df, race_df, minutes_before=5, max_staleness_minutes=60, _now=_FIXED_NOW)
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
        result = extract(ts_df, race_df, minutes_before=5, _now=_FIXED_NOW)
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
        result = extract(ts_df, race_df, minutes_before=5, _now=_FIXED_NOW)
        assert len(result) == 1  # 境界値は包含

    def test_happyotime_short_padding(self) -> None:
        """happyotime が7桁の場合でも正しくパースされる (zfill対応)。"""
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
                    "happyotime": "4110930",
                    "tanodds": 100.0,
                    "fukuoddslow": 30.0,
                    "tanninki": 5,
                },
            ]
        )
        result = extract(ts_df, race_df, minutes_before=5, _now=_FIXED_NOW)
        assert len(result) == 1

    def test_future_cutoff_excludes_race(self) -> None:
        """cutoff時刻がまだ来ていないレースは除外される。"""
        extract = self._import_target()
        # レース: 14:00発走 → cutoff = 13:55
        race_df = _make_race_df(
            [{"race_id": "20260411060101", "hassotime": 1400}]
        )
        # 13:30時点で実行 → cutoff(13:55)はまだ未来
        now_1330 = datetime(2026, 4, 11, 13, 30)
        ts_df = _make_odds_ts_df(
            [
                {
                    "race_id": "20260411060101",
                    "umaban": 1,
                    "year": 2026,
                    "happyotime": "04111330",
                    "tanodds": 100.0,
                    "fukuoddslow": 30.0,
                    "tanninki": 5,
                },
            ]
        )
        result = extract(ts_df, race_df, minutes_before=5, _now=now_1330)
        assert len(result) == 0  # cutoffが未来なので除外

    def test_cutoff_just_reached_includes_race(self) -> None:
        """cutoff時刻ちょうどになればレースが含まれる。"""
        extract = self._import_target()
        # レース: 14:00発走 → cutoff = 13:55
        race_df = _make_race_df(
            [{"race_id": "20260411060101", "hassotime": 1400}]
        )
        # 13:55時点で実行 → cutoff(13:55)にちょうど到達
        now_1355 = datetime(2026, 4, 11, 13, 55)
        ts_df = _make_odds_ts_df(
            [
                {
                    "race_id": "20260411060101",
                    "umaban": 1,
                    "year": 2026,
                    "happyotime": "04111355",
                    "tanodds": 100.0,
                    "fukuoddslow": 30.0,
                    "tanninki": 5,
                },
            ]
        )
        result = extract(ts_df, race_df, minutes_before=5, _now=now_1355)
        assert len(result) == 1  # cutoff時刻に到達 → 含まれる
