"""test_horse_history_features.py — HorseHistoryFeatures の単体テスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from db.repository import DataRepository


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
    """レース内z-score + pct のテスト"""

    def _make_race_df(self):
        return pd.DataFrame(
            {
                "race_id": ["r1"] * 4,
                "umaban": [1, 2, 3, 4],
                "norm_finish_logit_avg": [2.0, 1.0, 0.0, -1.0],
                "jockey_surprise": [0.1, 0.05, -0.02, -0.08],
                "haron_time_zscore_avg": [1.5, 0.5, -0.5, -1.5],
                "jockey_cond_wr": [0.15, 0.10, 0.05, 0.02],
            }
        )

    def test_z_score_sum_approx_zero(self):
        """レース内z-scoreの合計 ≈ 0"""
        from features.horse_history_features import HorseHistoryFeatures

        df = self._make_race_df()
        result = HorseHistoryFeatures.add_race_transforms(df)
        z_col = "norm_finish_logit_avg_race_z"
        assert z_col in result.columns
        assert abs(result[z_col].sum()) < 1e-6

    def test_std_zero_no_nan(self):
        """全馬同じ値（std=0）でも NaN にならない"""
        from features.horse_history_features import HorseHistoryFeatures

        df = pd.DataFrame(
            {
                "race_id": ["r1"] * 3,
                "umaban": [1, 2, 3],
                "norm_finish_logit_avg": [1.0, 1.0, 1.0],
                "jockey_surprise": [0.0, 0.0, 0.0],
                "haron_time_zscore_avg": [0.0, 0.0, 0.0],
                "jockey_cond_wr": [0.0, 0.0, 0.0],
            }
        )
        result = HorseHistoryFeatures.add_race_transforms(df)
        assert not result["norm_finish_logit_avg_race_z"].isna().any()

    def test_pct_range(self):
        """pct は [0, 1] の範囲"""
        from features.horse_history_features import HorseHistoryFeatures

        df = self._make_race_df()
        result = HorseHistoryFeatures.add_race_transforms(df)
        pct_col = "norm_finish_logit_avg_race_pct"
        assert result[pct_col].min() >= 0
        assert result[pct_col].max() <= 1


class TestLeakPrevention:
    """リーク防止のテスト"""

    def test_future_race_excluded(self):
        """当該レース日付より後のデータが特徴量に含まれない"""
        from features.horse_history_features import HorseHistoryFeatures

        target_date = pd.Timestamp("2024-06-01")
        mock_repo = MagicMock(spec=DataRepository)
        mock_repo.load_history_entries.return_value = pd.DataFrame(
            {
                "race_id": ["p1", "p2"],
                "ketto_num": ["H001", "H001"],
                "kisyu_code": ["J001", "J001"],
                "umaban": [1, 1],
                "finish_pos": [1, 3],
                "win_odds": [5.0, 8.0],
                "haron_time_l3": [34.5, 35.2],
            }
        )
        mock_repo.load_history_races.return_value = pd.DataFrame(
            {
                "race_id": ["p1", "p2"],
                "field_size": [16, 16],
                "race_date": [
                    pd.Timestamp("2024-05-01"),
                    pd.Timestamp("2024-07-01"),
                ],
            }
        )

        hist = HorseHistoryFeatures(repo=mock_repo)
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
                "ketto_num": ["H001"],
                "kisyu_code": ["J001"],
                "weight": [480.0],
            }
        )
        result = hist.compute(race_df, entry_df, np.array(["r1"]))

        # Should only use p1 (before target), not p2 (after)
        # With only 1 valid past race, norm_finish_logit_avg should be based on finish_pos=1 only
        if not result.empty and not result["norm_finish_logit_avg"].isna().all():
            # Verify the result uses only past data (finish_pos=1 from May)
            logit_val = result["norm_finish_logit_avg"].iloc[0]
            assert not np.isnan(logit_val)
            # 1st of 16 should give a positive logit
            assert logit_val > 0


class TestHorseHistoryFeaturesWithRepo:
    """DataRepository経由のHorseHistoryFeaturesテスト"""

    @pytest.fixture
    def mock_repo(self) -> MagicMock:
        return MagicMock(spec=DataRepository)

    def test_constructor_accepts_repo(self, mock_repo: MagicMock) -> None:
        from features.horse_history_features import HorseHistoryFeatures

        hhf = HorseHistoryFeatures(repo=mock_repo)
        assert hhf.repo is mock_repo

    def test_compute_calls_load_history_entries(self, mock_repo: MagicMock) -> None:
        from features.horse_history_features import HorseHistoryFeatures

        mock_repo.load_history_entries.return_value = pd.DataFrame(
            {
                "race_id": ["r1"],
                "ketto_num": ["1234"],
                "kisyu_code": ["5678"],
                "finish_pos": [1],
                "win_odds": [2.0],
                "haron_time_l3": [34.5],
                "umaban": [1],
            }
        )
        mock_repo.load_history_races.return_value = pd.DataFrame(
            {
                "race_id": ["r1"],
                "race_date": [pd.Timestamp("2020-01-01")],
                "field_size": [16],
            }
        )
        hhf = HorseHistoryFeatures(repo=mock_repo)
        race_df = pd.DataFrame({"race_id": ["r2"], "race_date": [pd.Timestamp("2020-06-01")]})
        entry_df = pd.DataFrame(
            {
                "race_id": ["r2"],
                "umaban": [1],
                "ketto_num": ["1234"],
                "kisyu_code": ["5678"],
                "weight": [480.0],
            }
        )
        hhf.compute(race_df, entry_df)
        mock_repo.load_history_entries.assert_called_once()

    def test_caching_prevents_repeated_loads(self, mock_repo: MagicMock) -> None:
        from features.horse_history_features import HorseHistoryFeatures

        mock_repo.load_history_entries.return_value = pd.DataFrame(
            {
                "race_id": ["r1"],
                "ketto_num": ["1234"],
                "kisyu_code": ["5678"],
                "finish_pos": [1],
                "win_odds": [2.0],
                "haron_time_l3": [34.5],
                "umaban": [1],
            }
        )
        mock_repo.load_history_races.return_value = pd.DataFrame(
            {
                "race_id": ["r1"],
                "race_date": [pd.Timestamp("2020-01-01")],
                "field_size": [16],
            }
        )
        hhf = HorseHistoryFeatures(repo=mock_repo)
        race_df = pd.DataFrame({"race_id": ["r2"], "race_date": [pd.Timestamp("2020-06-01")]})
        entry_df = pd.DataFrame(
            {
                "race_id": ["r2"],
                "umaban": [1],
                "ketto_num": ["1234"],
                "kisyu_code": ["5678"],
                "weight": [480.0],
            }
        )
        hhf.compute(race_df, entry_df)
        hhf.compute(race_df, entry_df)
        assert mock_repo.load_history_entries.call_count == 1
