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
    """expanding hierarchical fallback z-score のテスト"""

    def test_lookup_returns_stats_before_target_date(self):
        """target_date 以前の最新 expanding stats を返す"""
        from features.horse_history_features import _lookup_expanding_stats

        # Build expanding_stats: key=("sprint","turf","1") -> array of (date_float, mean, std)
        # Dates: 2024-01-01 and 2024-03-01
        d1 = np.datetime64("2024-01-01", "ns").astype(float)
        d2 = np.datetime64("2024-03-01", "ns").astype(float)
        expanding_stats = {
            ("sprint", "turf", "1"): np.array([[d1, 12.0, 0.2], [d2, 12.3, 0.4]]),
        }
        # Query date after d2 → should get d2's stats
        target = np.datetime64("2024-06-01", "ns")
        mean, std = _lookup_expanding_stats(target, "sprint", "turf", "1", expanding_stats)
        assert mean == 12.3
        assert std == 0.4

    def test_lookup_fallback_l1_to_l2(self):
        """Level 1 key に target_date 以前のデータが無い → Level 2 に fallback"""
        from features.horse_history_features import _lookup_expanding_stats

        d1 = np.datetime64("2024-03-01", "ns").astype(float)
        d2 = np.datetime64("2024-01-01", "ns").astype(float)
        expanding_stats = {
            # L1: data only after target_date, so idx=0 → skip
            ("sprint", "turf", "1"): np.array([[d1, 12.5, 0.3]]),
            # L2: has data before target
            ("sprint", "turf"): np.array([[d2, 12.3, 0.4]]),
        }
        target = np.datetime64("2024-02-01", "ns")
        mean, std = _lookup_expanding_stats(target, "sprint", "turf", "1", expanding_stats)
        assert mean == 12.3
        assert std == 0.4

    def test_lookup_fallback_to_global(self):
        """全レベル該当なし → グローバル fallback (all,)"""
        from features.horse_history_features import _lookup_expanding_stats

        d1 = np.datetime64("2024-01-01", "ns").astype(float)
        expanding_stats = {
            ("all",): np.array([[d1, 12.4, 0.5]]),
        }
        target = np.datetime64("2024-06-01", "ns")
        mean, std = _lookup_expanding_stats(target, "long", "dirt", "3", expanding_stats)
        assert mean == 12.4

    def test_lookup_returns_nan_if_no_data(self):
        """該当データが全く無い場合 → (nan, nan)"""
        from features.horse_history_features import _lookup_expanding_stats

        expanding_stats: dict[tuple, np.ndarray] = {}
        target = np.datetime64("2024-06-01", "ns")
        mean, std = _lookup_expanding_stats(target, "long", "dirt", "3", expanding_stats)
        assert np.isnan(mean)
        assert np.isnan(std)

    def test_expanding_stats_not_leaky(self):
        """compute() が expanding_stats を使っていることをソースコードで確認"""
        import inspect

        from features.horse_history_features import HorseHistoryFeatures

        source = inspect.getsource(HorseHistoryFeatures.compute)
        assert "expanding_stats" in source, "Should use expanding_stats"
        assert "_lookup_expanding_stats" in source, "Should use _lookup_expanding_stats"
        assert "global_stats: dict" not in source, "Should not have old global_stats pattern"


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


class TestNPastParameter:
    """n_past パラメータ化のテスト"""

    def test_n_past_parameter(self):
        """n_past=5 の場合、過去5走分のデータが使用される"""
        from features.horse_history_features import HorseHistoryFeatures

        mock_store = MagicMock(spec=ParquetStore)
        hist = HorseHistoryFeatures(store=mock_store, n_past=5)
        assert hist._n_past == 5
        # デフォルト値は5 (B3仕様)
        hist_default = HorseHistoryFeatures(store=mock_store)
        assert hist_default._n_past == 5


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
                "syussotosu": [16, 16],
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
                "syussotosu": [16],
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


class TestWeightZscore:
    """A2: weight_zscore が results DataFrame に含まれることを確認"""

    def _make_mock_store_with_weights(self) -> MagicMock:
        """過去出走データに bataijyu 列を含むモックストア"""
        store = MagicMock(spec=ParquetStore)
        entries_hist = pd.DataFrame(
            {
                "race_id": ["p1", "p2", "p3"],
                "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                "kettonum": ["K1", "K1", "K1"],
                "kisyucode": ["J1", "J1", "J1"],
                "umaban": [1, 1, 1],
                "kakuteijyuni": [3, 5, 2],
                "odds": [5.0, 8.0, 3.0],
                "harontimel3": [35.0, 36.0, 34.5],
                "distance_bin": ["mile", "mile", "sprint"],
                "timediff": [0.3, -0.2, 0.5],
                "jyuni1c": [5, 8, 3],
                "jyuni4c": [4, 6, 2],
                "kyakusitukubun": [2, 2, 1],
                "bataijyu": [480.0, 482.0, 484.0],
            }
        )
        races_hist = pd.DataFrame(
            {
                "race_id": ["p1", "p2", "p3"],
                "syussotosu": [10, 12, 8],
                "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                "trackcd": [11, 11, 11],
                "kyori": [1600, 1600, 1200],
                "surface": ["turf", "turf", "turf"],
                "track_condition_code": [1, 2, 1],
            }
        )

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            elif name == "races":
                return races_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)
        return store

    def test_weight_zscore_in_output_columns(self) -> None:
        """compute() の出力に weight_zscore 列が含まれる"""
        from features.horse_history_features import HorseHistoryFeatures

        store = self._make_mock_store_with_weights()
        hhf = HorseHistoryFeatures(store=store)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-04-01"]),
                "surface": ["turf"],
                "kyori": [1600],
            }
        )
        entry_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "kettonum": ["K1"],
                "kisyucode": ["J1"],
                "bataijyu": [486.0],
                "kakuteijyuni": [1],
                "syussotosu": [10],
            }
        )
        result = hhf.compute(race_df, entry_df)
        assert "weight_zscore" in result.columns

    def test_weight_zscore_computed_correctly(self) -> None:
        """weight_zscore が正しく計算される (past weights: 480, 482, 484 → mean=482, std≈2)"""
        from features.horse_history_features import HorseHistoryFeatures

        store = self._make_mock_store_with_weights()
        hhf = HorseHistoryFeatures(store=store)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-04-01"]),
                "surface": ["turf"],
                "kyori": [1600],
            }
        )
        entry_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "kettonum": ["K1"],
                "kisyucode": ["J1"],
                "bataijyu": [486.0],
                "kakuteijyuni": [1],
                "syussotosu": [10],
            }
        )
        result = hhf.compute(race_df, entry_df)
        # Past weights: 480, 482, 484 → mean=482, std≈2.0 (population std of [480,482,484])
        # Current weight: 486 → zscore = (486 - 482) / 2.0 = 2.0
        assert not result["weight_zscore"].isna().all()
        zscore = result["weight_zscore"].iloc[0]
        assert abs(zscore - 2.0) < 0.5, f"Expected zscore ≈ 2.0, got {zscore}"

    def test_weight_zscore_nan_when_no_past(self) -> None:
        """過去出走がない場合 weight_zscore は NaN"""
        from features.horse_history_features import HorseHistoryFeatures

        store = MagicMock(spec=ParquetStore)
        entries_hist = pd.DataFrame(
            columns=[
                "race_id",
                "race_date",
                "kettonum",
                "kisyucode",
                "kakuteijyuni",
                "syussotosu",
                "odds",
            ]
        )
        races_hist = pd.DataFrame(columns=["race_id", "syussotosu", "race_date"])

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            elif name == "races":
                return races_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)

        hhf = HorseHistoryFeatures(store=store)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-04-01"]),
            }
        )
        entry_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "kettonum": ["NEW_HORSE"],
                "kisyucode": ["J1"],
                "bataijyu": [480.0],
            }
        )
        result = hhf.compute(race_df, entry_df)
        assert result["weight_zscore"].isna().all()


class TestRestPeriodFeatures:
    """A3: days_since_last_race, rest_category のテスト"""

    def _make_mock_store_with_dates(self) -> MagicMock:
        """過去出走データに race_date を含むモックストア"""
        store = MagicMock(spec=ParquetStore)
        entries_hist = pd.DataFrame(
            {
                "race_id": ["p1", "p2", "p3"],
                "race_date": pd.to_datetime(["2024-01-15", "2024-03-01", "2024-05-10"]),
                "kettonum": ["K1", "K1", "K1"],
                "kisyucode": ["J1", "J1", "J1"],
                "umaban": [1, 1, 1],
                "kakuteijyuni": [3, 5, 2],
                "odds": [5.0, 8.0, 3.0],
                "harontimel3": [35.0, 36.0, 34.5],
                "distance_bin": ["mile", "mile", "sprint"],
                "surface": ["turf", "turf", "turf"],
                "track_condition_code": [1, 2, 1],
                "timediff": [0.3, -0.2, 0.5],
                "jyuni1c": [5, 8, 3],
                "jyuni4c": [4, 6, 2],
                "kyakusitukubun": [2, 2, 1],
                "bataijyu": [480.0, 482.0, 484.0],
            }
        )
        races_hist = pd.DataFrame(
            {
                "race_id": ["p1", "p2", "p3"],
                "syussotosu": [10, 12, 8],
                "race_date": pd.to_datetime(["2024-01-15", "2024-03-01", "2024-05-10"]),
                "trackcd": [11, 11, 11],
                "kyori": [1600, 1600, 1200],
                "surface": ["turf", "turf", "turf"],
                "track_condition_code": [1, 2, 1],
            }
        )

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            elif name == "races":
                return races_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)
        return store

    def test_days_since_last_race_in_output(self) -> None:
        """compute() の出力に days_since_last_race 列が含まれる"""
        from features.horse_history_features import HorseHistoryFeatures

        store = self._make_mock_store_with_dates()
        hhf = HorseHistoryFeatures(store=store)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-07-01"]),
                "surface": ["turf"],
                "kyori": [1600],
            }
        )
        entry_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "kettonum": ["K1"],
                "kisyucode": ["J1"],
                "bataijyu": [486.0],
                "kakuteijyuni": [1],
                "syussotosu": [10],
            }
        )
        result = hhf.compute(race_df, entry_df)
        assert "days_since_last_race" in result.columns
        assert "rest_category" in result.columns
        # 2024-05-10 → 2024-07-01 = 52日 → rest_category = 3 (medium: 31-90日)
        assert result["rest_category"].iloc[0] == 3.0
        assert result["days_since_last_race"].iloc[0] == 52.0

    def test_rest_category_consecutive(self) -> None:
        """consecutive (≤7日) → rest_category = 1"""
        from features.horse_history_features import HorseHistoryFeatures

        store = MagicMock(spec=ParquetStore)
        entries_hist = pd.DataFrame(
            {
                "race_id": ["p1"],
                "race_date": pd.to_datetime(["2024-06-28"]),
                "kettonum": ["K1"],
                "kisyucode": ["J1"],
                "umaban": [1],
                "kakuteijyuni": [3],
                "odds": [5.0],
                "harontimel3": [35.0],
                "distance_bin": ["mile"],
                "surface": ["turf"],
                "track_condition_code": [1],
                "timediff": [0.3],
                "jyuni1c": [5],
                "jyuni4c": [4],
                "kyakusitukubun": [2],
                "bataijyu": [480.0],
            }
        )
        races_hist = pd.DataFrame(
            {
                "race_id": ["p1"],
                "syussotosu": [10],
                "race_date": pd.to_datetime(["2024-06-28"]),
                "trackcd": [11],
                "kyori": [1600],
                "surface": ["turf"],
                "track_condition_code": [1],
            }
        )

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            elif name == "races":
                return races_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)
        hhf = HorseHistoryFeatures(store=store)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-07-01"]),
                "surface": ["turf"],
                "kyori": [1600],
            }
        )
        entry_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "kettonum": ["K1"],
                "kisyucode": ["J1"],
                "bataijyu": [486.0],
                "kakuteijyuni": [1],
                "syussotosu": [10],
            }
        )
        result = hhf.compute(race_df, entry_df)
        assert result["rest_category"].iloc[0] == 1.0
        assert result["days_since_last_race"].iloc[0] == 3.0

    def test_rest_category_short(self) -> None:
        """short (8-30日) → rest_category = 2"""
        from features.horse_history_features import HorseHistoryFeatures

        store = MagicMock(spec=ParquetStore)
        entries_hist = pd.DataFrame(
            {
                "race_id": ["p1"],
                "race_date": pd.to_datetime(["2024-06-15"]),
                "kettonum": ["K1"],
                "kisyucode": ["J1"],
                "umaban": [1],
                "kakuteijyuni": [3],
                "odds": [5.0],
                "harontimel3": [35.0],
                "distance_bin": ["mile"],
                "surface": ["turf"],
                "track_condition_code": [1],
                "timediff": [0.3],
                "jyuni1c": [5],
                "jyuni4c": [4],
                "kyakusitukubun": [2],
                "bataijyu": [480.0],
            }
        )
        races_hist = pd.DataFrame(
            {
                "race_id": ["p1"],
                "syussotosu": [10],
                "race_date": pd.to_datetime(["2024-06-15"]),
                "trackcd": [11],
                "kyori": [1600],
                "surface": ["turf"],
                "track_condition_code": [1],
            }
        )

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            elif name == "races":
                return races_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)
        hhf = HorseHistoryFeatures(store=store)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-07-01"]),
                "surface": ["turf"],
                "kyori": [1600],
            }
        )
        entry_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "kettonum": ["K1"],
                "kisyucode": ["J1"],
                "bataijyu": [486.0],
                "kakuteijyuni": [1],
                "syussotosu": [10],
            }
        )
        result = hhf.compute(race_df, entry_df)
        assert result["rest_category"].iloc[0] == 2.0  # 16日 = short (8-30)
        assert result["days_since_last_race"].iloc[0] == 16.0

    def test_rest_category_long(self) -> None:
        """long (91-180日) → rest_category = 4"""
        from features.horse_history_features import HorseHistoryFeatures

        store = MagicMock(spec=ParquetStore)
        entries_hist = pd.DataFrame(
            {
                "race_id": ["p1"],
                "race_date": pd.to_datetime(["2024-01-15"]),
                "kettonum": ["K1"],
                "kisyucode": ["J1"],
                "umaban": [1],
                "kakuteijyuni": [3],
                "odds": [5.0],
                "harontimel3": [35.0],
                "distance_bin": ["mile"],
                "surface": ["turf"],
                "track_condition_code": [1],
                "timediff": [0.3],
                "jyuni1c": [5],
                "jyuni4c": [4],
                "kyakusitukubun": [2],
                "bataijyu": [480.0],
            }
        )
        races_hist = pd.DataFrame(
            {
                "race_id": ["p1"],
                "syussotosu": [10],
                "race_date": pd.to_datetime(["2024-01-15"]),
                "trackcd": [11],
                "kyori": [1600],
                "surface": ["turf"],
                "track_condition_code": [1],
            }
        )

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            elif name == "races":
                return races_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)
        hhf = HorseHistoryFeatures(store=store)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-07-01"]),
                "surface": ["turf"],
                "kyori": [1600],
            }
        )
        entry_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "kettonum": ["K1"],
                "kisyucode": ["J1"],
                "bataijyu": [486.0],
                "kakuteijyuni": [1],
                "syussotosu": [10],
            }
        )
        result = hhf.compute(race_df, entry_df)
        assert result["rest_category"].iloc[0] == 4.0  # 168日 = long (91-180)
        assert result["days_since_last_race"].iloc[0] == 168.0

    def test_rest_category_return(self) -> None:
        """return (>180日) → rest_category = 5"""
        from features.horse_history_features import HorseHistoryFeatures

        store = MagicMock(spec=ParquetStore)
        entries_hist = pd.DataFrame(
            {
                "race_id": ["p1"],
                "race_date": pd.to_datetime(["2023-06-01"]),
                "kettonum": ["K1"],
                "kisyucode": ["J1"],
                "umaban": [1],
                "kakuteijyuni": [3],
                "odds": [5.0],
                "harontimel3": [35.0],
                "distance_bin": ["mile"],
                "surface": ["turf"],
                "track_condition_code": [1],
                "timediff": [0.3],
                "jyuni1c": [5],
                "jyuni4c": [4],
                "kyakusitukubun": [2],
                "bataijyu": [480.0],
            }
        )
        races_hist = pd.DataFrame(
            {
                "race_id": ["p1"],
                "syussotosu": [10],
                "race_date": pd.to_datetime(["2023-06-01"]),
                "trackcd": [11],
                "kyori": [1600],
                "surface": ["turf"],
                "track_condition_code": [1],
            }
        )

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            elif name == "races":
                return races_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)
        hhf = HorseHistoryFeatures(store=store)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-07-01"]),
                "surface": ["turf"],
                "kyori": [1600],
            }
        )
        entry_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "kettonum": ["K1"],
                "kisyucode": ["J1"],
                "bataijyu": [486.0],
                "kakuteijyuni": [1],
                "syussotosu": [10],
            }
        )
        result = hhf.compute(race_df, entry_df)
        assert result["rest_category"].iloc[0] == 5.0  # 396日 = return (>180)
        assert result["days_since_last_race"].iloc[0] == 396.0

    def test_rest_category_nan_for_no_history(self) -> None:
        """過去データなしの場合はNaN"""
        from features.horse_history_features import HorseHistoryFeatures

        store = MagicMock(spec=ParquetStore)
        # 馬がいるが有効な過去レースがない (= field_size < 8 なので valid_field==0)
        entries_hist = pd.DataFrame(
            {
                "race_id": ["p1"],
                "race_date": pd.to_datetime(["2024-06-01"]),
                "kettonum": ["K99"],
                "kisyucode": ["J1"],
                "umaban": [1],
                "kakuteijyuni": [1],
                "odds": [5.0],
                "harontimel3": [35.0],
                "distance_bin": ["mile"],
                "surface": ["turf"],
                "track_condition_code": [1],
                "timediff": [0.3],
                "jyuni1c": [5],
                "jyuni4c": [4],
                "kyakusitukubun": [2],
                "bataijyu": [480.0],
            }
        )
        races_hist = pd.DataFrame(
            {
                "race_id": ["p1"],
                "syussotosu": [6],  # 8未満 → valid_field == 0
                "race_date": pd.to_datetime(["2024-06-01"]),
                "trackcd": [11],
                "kyori": [1600],
                "surface": ["turf"],
                "track_condition_code": [1],
            }
        )

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            elif name == "races":
                return races_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)
        hhf = HorseHistoryFeatures(store=store)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-07-01"]),
                "surface": ["turf"],
                "kyori": [1600],
            }
        )
        entry_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "kettonum": ["K99"],  # 履歴なし
                "kisyucode": ["J1"],
                "bataijyu": [486.0],
                "kakuteijyuni": [1],
                "syussotosu": [10],
            }
        )
        result = hhf.compute(race_df, entry_df)
        assert not result.empty
        assert np.isnan(result["days_since_last_race"].iloc[0])
        assert np.isnan(result["rest_category"].iloc[0])


class TestHorseHistoryFeaturesWithStore2:
    """ParquetStore経由のHorseHistoryFeaturesテスト (caching)"""

    @pytest.fixture
    def mock_store(self) -> MagicMock:
        return MagicMock(spec=ParquetStore)

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
                "syussotosu": [16],
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
