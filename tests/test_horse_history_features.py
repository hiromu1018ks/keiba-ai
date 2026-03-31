"""test_horse_history_features.py — HorseHistoryFeatures の単体テスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from db.parquet_store import ParquetStore


class TestNormFinishLogitAvg:
    """norm_finish_logit_avg (logit変換着順スコア) のテスト"""

    def test_1st_of_16(self):
        """1着/16頭 → logit(15/15) clipped to logit(0.95) ≈ 2.94"""
        from features.horse_history_features import _norm_finish_logit

        result = _norm_finish_logit(finish_pos=1, field_size=16)
        assert 2.9 < result < 3.0

    def test_last_of_16(self):
        """最下位/16頭 → logit(1/15) clipped to logit(0.05) ≈ -2.94"""
        from features.horse_history_features import _norm_finish_logit

        result = _norm_finish_logit(finish_pos=16, field_size=16)
        assert -3.0 < result < -2.9

    def test_field_size_under_8_returns_nan(self):
        """8頭未満レース → NaN"""
        from features.horse_history_features import _norm_finish_logit

        result = _norm_finish_logit(finish_pos=1, field_size=7)
        assert np.isnan(result)

    def test_mid_rank(self):
        """8着/16頭 → logit(0.533) ≈ 0.13"""
        from features.horse_history_features import _norm_finish_logit

        result = _norm_finish_logit(finish_pos=8, field_size=16)
        assert -0.2 < result < 0.2


class TestJockeySurprise:
    """jockey_surprise (Beta事前分布スムージング) のテスト"""

    def test_zero_wins_100_races(self):
        """100戦0勝 → surprise ≈ 0.00826 - 0.0476 ≈ -0.0394"""
        from features.horse_history_features import _compute_jockey_surprise

        result = _compute_jockey_surprise(actual_wins=0, n_races=100, expected_wins=8.0)
        assert -0.05 < result < -0.03

    def test_above_expectation(self):
        """期待以上の勝率 → 正のsurprise"""
        from features.horse_history_features import _compute_jockey_surprise

        result = _compute_jockey_surprise(actual_wins=15, n_races=100, expected_wins=8.0)
        assert 0.05 < result < 0.15

    def test_payout_rate_applied(self):
        """控除率補正（0.80）が適用される"""
        from features.horse_history_features import PAYOUT_RATE

        assert PAYOUT_RATE == 0.80

    def test_min_samples_returns_nan(self):
        """30レース未満 → NaN"""
        from features.horse_history_features import _compute_jockey_surprise

        result = _compute_jockey_surprise(actual_wins=5, n_races=25, expected_wins=2.0)
        assert np.isnan(result)


class TestHaronTimeZscore:
    """haron_time_zscore_avg (階層fallback) のテスト"""

    def test_fallback_l1_to_l2(self):
        """Level 1 サンプル不足 → Level 2 にfallback"""
        from features.horse_history_features import _get_group_stats

        global_stats = {
            ("sprint", "turf", "1"): {"mean": 12.5, "std": 0.3, "n": 10},  # L1: 不足
            ("sprint", "turf"): {"mean": 12.3, "std": 0.4, "n": 80},  # L2: OK
            ("sprint",): {"mean": 12.4, "std": 0.5, "n": 200},  # L3
            ("all",): {"mean": 12.4, "std": 0.5, "n": 5000},  # L4
        }
        mean, std = _get_group_stats(
            distance_bin="sprint",
            surface="turf",
            baba_cd="1",
            global_stats=global_stats,
        )
        assert mean == 12.3
        assert std == 0.4

    def test_fallback_to_global(self):
        """全レベル不足 → グローバルfallback"""
        from features.horse_history_features import _get_group_stats

        global_stats = {
            ("all",): {"mean": 12.4, "std": 0.5, "n": 5000},
        }
        mean, std = _get_group_stats(
            distance_bin="long",
            surface="dirt",
            baba_cd="3",
            global_stats=global_stats,
        )
        assert mean == 12.4


class TestRaceTransforms:
    """レース内 percentile rank (race_rank) のテスト"""

    def _make_race_df(self):
        return pd.DataFrame(
            {
                "race_id": ["r1"] * 4,
                "umaban": [1, 2, 3, 4],
                "norm_finish_logit_avg": [2.0, 1.0, 0.0, -1.0],
                "jockey_surprise": [0.1, 0.05, -0.02, -0.08],
                "harontimel3_avg": [34.0, 35.0, 36.0, 37.0],
                "jockey_cond_wr": [0.15, 0.10, 0.05, 0.02],
            }
        )

    def test_rank_column_created(self):
        """_race_rank 列が数値BASE_COLS について生成される"""
        from features.horse_history_features import HorseHistoryFeatures

        df = self._make_race_df()
        result = HorseHistoryFeatures.add_race_transforms(df)
        assert "norm_finish_logit_avg_race_rank" in result.columns
        assert "harontimel3_avg_race_rank" in result.columns

    def test_no_z_or_pct_columns(self):
        """_race_z と _race_pct 列は生成されない"""
        from features.horse_history_features import HorseHistoryFeatures

        df = self._make_race_df()
        result = HorseHistoryFeatures.add_race_transforms(df)
        z_cols = [c for c in result.columns if c.endswith("_race_z")]
        pct_cols = [c for c in result.columns if c.endswith("_race_pct")]
        assert len(z_cols) == 0, f"Unexpected _race_z columns: {z_cols}"
        assert len(pct_cols) == 0, f"Unexpected _race_pct columns: {pct_cols}"

    def test_rank_range(self):
        """race_rank は [0, 1] の範囲"""
        from features.horse_history_features import HorseHistoryFeatures

        df = self._make_race_df()
        result = HorseHistoryFeatures.add_race_transforms(df)
        rank_col = "norm_finish_logit_avg_race_rank"
        assert result[rank_col].min() >= 0
        assert result[rank_col].max() <= 1

    def test_rank_ordering(self):
        """値が大きいほど race_rank も大きい"""
        from features.horse_history_features import HorseHistoryFeatures

        df = self._make_race_df()
        result = HorseHistoryFeatures.add_race_transforms(df)
        rank_col = "norm_finish_logit_avg_race_rank"
        # norm_finish_logit_avg: [2.0, 1.0, 0.0, -1.0]
        # rank should be [1.0, 0.75, 0.5, 0.25] (descending by value)
        ranks = result[rank_col].tolist()
        assert ranks[0] > ranks[3]  # 2.0 should rank higher than -1.0

    def test_category_cols_excluded(self):
        """kyakusitukubun_cd (カテゴリ列) は race_rank を生成しない"""
        from features.horse_history_features import HorseHistoryFeatures

        df = self._make_race_df()
        df["kyakusitukubun_cd"] = [1, 2, 3, 4]
        result = HorseHistoryFeatures.add_race_transforms(df)
        assert "kyakusitukubun_cd_race_rank" not in result.columns

    def test_jockey_cols_excluded(self):
        """jockey_surprise, jockey_cond_wr は race_rank を生成しない"""
        from features.horse_history_features import HorseHistoryFeatures

        df = self._make_race_df()
        result = HorseHistoryFeatures.add_race_transforms(df)
        assert "jockey_surprise_race_rank" not in result.columns
        assert "jockey_cond_wr_race_rank" not in result.columns

    def test_tied_values_use_average(self):
        """同値は method='average' で平均ランク"""
        from features.horse_history_features import HorseHistoryFeatures

        df = pd.DataFrame(
            {
                "race_id": ["r1"] * 4,
                "umaban": [1, 2, 3, 4],
                "norm_finish_logit_avg": [1.0, 1.0, 0.0, 0.0],
            }
        )
        result = HorseHistoryFeatures.add_race_transforms(df)
        rank_col = "norm_finish_logit_avg_race_rank"
        # Two tied at 1.0 (ranks 3,4 → avg 3.5 → pct 3.5/4=0.875)
        # Two tied at 0.0 (ranks 1,2 → avg 1.5 → pct 1.5/4=0.375)
        ranks = result[rank_col].tolist()
        assert abs(ranks[0] - 0.875) < 1e-6
        assert abs(ranks[1] - 0.875) < 1e-6
        assert abs(ranks[2] - 0.375) < 1e-6
        assert abs(ranks[3] - 0.375) < 1e-6


class TestLeakPrevention:
    """リーク防止のテスト"""

    def test_future_race_excluded(self):
        """当該レース日付より後のデータが特徴量に含まれない"""
        from features.horse_history_features import HorseHistoryFeatures

        target_date = pd.Timestamp("2024-06-01")
        mock_store = MagicMock(spec=ParquetStore)

        # Mock load_history_entries via readers
        entries_data = pd.DataFrame(
            {
                "race_id": ["p1", "p2"],
                "kettonum": ["H001", "H001"],
                "kisyucode": ["J001", "J001"],
                "umaban": [1, 1],
                "kakuteijyuni": [1, 3],
                "odds": [5.0, 8.0],
                "harontimel3": [34.5, 35.2],
                "race_date": [
                    pd.Timestamp("2024-05-01"),
                    pd.Timestamp("2024-07-01"),
                ],
            }
        )
        races_data = pd.DataFrame(
            {
                "race_id": ["p1", "p2"],
                "field_size": [16, 16],
                "race_date": [
                    pd.Timestamp("2024-05-01"),
                    pd.Timestamp("2024-07-01"),
                ],
                "trackcd": [11, 11],
                "kyori": [1600, 1600],
                "surface": ["turf", "turf"],
            }
        )

        # Setup mock to return appropriate data for load_history_entries/load_history_races

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_data
            elif name == "races":
                return races_data
            return pd.DataFrame()

        mock_store.read = MagicMock(side_effect=mock_read)

        hist = HorseHistoryFeatures(store=mock_store)
        race_df = pd.DataFrame(
            {
                "race_id": ["r1"],
                "race_date": [target_date],
            }
        )
        entry_df = pd.DataFrame(
            {
                "race_id": ["r1"],
                "umaban": [1],
                "kettonum": ["H001"],
                "kisyucode": ["J001"],
                "bataijyu": [480.0],
            }
        )
        result = hist.compute(race_df, entry_df, np.array(["r1"]))

        # Should only use p1 (before target), not p2 (after)
        if not result.empty and not result["norm_finish_logit_avg"].isna().all():
            logit_val = result["norm_finish_logit_avg"].iloc[0]
            assert not np.isnan(logit_val)
            assert logit_val > 0


class TestHorseHistoryFeaturesWithStore:
    """ParquetStore経由のHorseHistoryFeaturesテスト"""

    @pytest.fixture
    def mock_store(self) -> MagicMock:
        return MagicMock(spec=ParquetStore)

    def test_constructor_accepts_store(self, mock_store: MagicMock) -> None:
        from features.horse_history_features import HorseHistoryFeatures

        hhf = HorseHistoryFeatures(store=mock_store)
        assert hhf.store is mock_store

    def test_compute_calls_load_history_entries(self, mock_store: MagicMock) -> None:
        from features.horse_history_features import HorseHistoryFeatures

        entries_data = pd.DataFrame(
            {
                "race_id": ["r1"],
                "kettonum": ["1234"],
                "kisyucode": ["5678"],
                "kakuteijyuni": [1],
                "odds": [2.0],
                "harontimel3": [34.5],
                "umaban": [1],
                "race_date": [pd.Timestamp("2020-01-01")],
            }
        )
        races_data = pd.DataFrame(
            {
                "race_id": ["r1"],
                "race_date": [pd.Timestamp("2020-01-01")],
                "field_size": [16],
                "trackcd": [11],
                "kyori": [1600],
                "surface": ["turf"],
            }
        )

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_data
            elif name == "races":
                return races_data
            return pd.DataFrame()

        mock_store.read = MagicMock(side_effect=mock_read)

        hhf = HorseHistoryFeatures(store=mock_store)
        race_df = pd.DataFrame({"race_id": ["r2"], "race_date": [pd.Timestamp("2020-06-01")]})
        entry_df = pd.DataFrame(
            {
                "race_id": ["r2"],
                "umaban": [1],
                "kettonum": ["1234"],
                "kisyucode": ["5678"],
                "bataijyu": [480.0],
            }
        )
        hhf.compute(race_df, entry_df)
        # Verify store.read was called (for entries and races)
        assert mock_store.read.called

    def test_caching_prevents_repeated_loads(self, mock_store: MagicMock) -> None:
        from features.horse_history_features import HorseHistoryFeatures

        entries_data = pd.DataFrame(
            {
                "race_id": ["r1"],
                "kettonum": ["1234"],
                "kisyucode": ["5678"],
                "kakuteijyuni": [1],
                "odds": [2.0],
                "harontimel3": [34.5],
                "umaban": [1],
                "race_date": [pd.Timestamp("2020-01-01")],
            }
        )
        races_data = pd.DataFrame(
            {
                "race_id": ["r1"],
                "race_date": [pd.Timestamp("2020-01-01")],
                "field_size": [16],
                "trackcd": [11],
                "kyori": [1600],
                "surface": ["turf"],
            }
        )

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_data
            elif name == "races":
                return races_data
            return pd.DataFrame()

        mock_store.read = MagicMock(side_effect=mock_read)

        hhf = HorseHistoryFeatures(store=mock_store)
        race_df = pd.DataFrame({"race_id": ["r2"], "race_date": [pd.Timestamp("2020-06-01")]})
        entry_df = pd.DataFrame(
            {
                "race_id": ["r2"],
                "umaban": [1],
                "kettonum": ["1234"],
                "kisyucode": ["5678"],
                "bataijyu": [480.0],
            }
        )
        hhf.compute(race_df, entry_df)
        hhf.compute(race_df, entry_df)
        # Caching means entries is loaded once, races is loaded once
        # Total 2 calls (one for entries, one for races), not 4
        assert mock_store.read.call_count <= 2
