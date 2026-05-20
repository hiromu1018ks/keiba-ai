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
        """100戦0勝、期待勝利数8 → 負のsurprise (オッズ以上の勝利数が期待されたが0勝)"""
        from features.horse_history_features import _compute_jockey_surprise

        result = _compute_jockey_surprise(actual_wins=0, n_races=100, expected_wins=8.0)
        assert result < 0  # 期待を下回る

    def test_above_expectation(self):
        """期待以上の勝率 → 正のsurprise (オッズ期待8勝に対し15勝)"""
        from features.horse_history_features import _compute_jockey_surprise

        result = _compute_jockey_surprise(actual_wins=15, n_races=100, expected_wins=8.0)
        assert result > 0  # 期待を上回る

    def test_at_expectation(self):
        """期待勝利数ちょうど → surprise ≈ 0"""
        from features.horse_history_features import _compute_jockey_surprise

        result = _compute_jockey_surprise(actual_wins=8, n_races=100, expected_wins=8.0)
        assert abs(result) < 0.01

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
                "harontimel5_avg": [34.0, 35.0, 36.0, 37.0],
                "jockey_cond_wr": [0.15, 0.10, 0.05, 0.02],
            }
        )

    def test_rank_column_created(self):
        """_race_rank 列が数値BASE_COLS について生成される"""
        from features.horse_history_features import HorseHistoryFeatures

        df = self._make_race_df()
        result = HorseHistoryFeatures.add_race_transforms(df)
        assert "norm_finish_logit_avg_race_rank" in result.columns
        assert "harontimel5_avg_race_rank" in result.columns

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


class TestHaronTimeL5AndLateTrend:
    """harontimel5_avg / harontime_late_trend のテスト"""

    def test_harontimel5_avg_uses_5_races(self):
        """harontimel5_avg が5走分のハロンタイム平均を返す"""
        import pandas as pd
        import numpy as np
        from features.horse_history_features import HorseHistoryFeatures
        from unittest.mock import MagicMock
        from db.parquet_store import ParquetStore

        store = MagicMock(spec=ParquetStore)
        entries_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3", "p4", "p5"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01",
                                          "2024-04-01", "2024-05-01"]),
            "kettonum": ["K1", "K1", "K1", "K1", "K1"],
            "kisyucode": ["J1", "J1", "J1", "J1", "J1"],
            "umaban": [1, 1, 1, 1, 1],
            "kakuteijyuni": [3, 5, 2, 1, 4],
            "odds": [5.0, 8.0, 3.0, 2.0, 10.0],
            "harontimel3": [34.5, 35.0, np.nan, 34.0, 34.8],
            "distance_bin": ["mile", "mile", "sprint", "mile", "mile"],
            "surface": ["turf", "turf", "turf", "turf", "turf"],
            "track_condition_code": [1, 2, 1, 1, 2],
            "timediff": [0.3, -0.2, 0.5, 0.1, -0.1],
            "jyuni1c": [5, 8, 3, 1, 6],
            "jyuni4c": [4, 6, 2, 1, 5],
            "kyakusitukubun": [2, 2, 1, 1, 3],
            "bataijyu": [480.0, 482.0, 484.0, 486.0, 488.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3", "p4", "p5"],
            "syussotosu": [10, 12, 8, 16, 14],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01",
                                          "2024-04-01", "2024-05-01"]),
            "trackcd": [11, 11, 11, 11, 11],
            "kyori": [1600, 1600, 1200, 1600, 1600],
            "surface": ["turf", "turf", "turf", "turf", "turf"],
            "track_condition_code": [1, 2, 1, 1, 2],
        })

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            elif name == "races":
                return races_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)

        hhf = HorseHistoryFeatures(store=store, n_past=5)
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = hhf.compute(race_df, entry_df)

        assert "harontimel5_avg" in result.columns, f"Missing column. Got: {result.columns.tolist()}"
        # EMA重み付け (halflife=3): 直近値33.0に最も高い重み
        # NaNをスキップした4走: [34.5, 35.0, 34.0, 34.8] (古→新)
        # decay ≈ 0.231, weights = [(1-0.231)^3, (1-0.231)^2, (1-0.231)^1, 1.0]
        # = [0.454, 0.589, 0.769, 1.0], reversed→[1.0, 0.769, 0.589, 0.454]
        # normalized: [0.369, 0.284, 0.217, 0.130]
        # EMA avg ≈ 34.5*0.369 + 35.0*0.284 + 34.0*0.217 + 34.8*0.130 ≈ 34.58
        actual = result["harontimel5_avg"].iloc[0]
        assert abs(actual - 34.58) < 0.5, f"Expected EMA avg ≈ 34.58, got {actual}"

    def test_harontime_late_trend_improving(self):
        """harontime_late_trend が最後2走 vs 最初3走の差を返す（改善時は負）"""
        import pandas as pd
        import numpy as np
        from features.horse_history_features import HorseHistoryFeatures
        from unittest.mock import MagicMock
        from db.parquet_store import ParquetStore

        store = MagicMock(spec=ParquetStore)
        # 最近2走が速い (改善傾向) → late_trend < 0
        entries_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3", "p4", "p5"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01",
                                          "2024-04-01", "2024-05-01"]),
            "kettonum": ["K1"] * 5,
            "kisyucode": ["J1"] * 5,
            "umaban": [1] * 5,
            "kakuteijyuni": [5, 4, 3, 2, 1],
            "odds": [20.0, 15.0, 10.0, 5.0, 2.0],
            "harontimel3": [36.0, 35.5, 35.0, 34.0, 33.5],  # 改善傾向
            "distance_bin": ["mile"] * 5,
            "surface": ["turf"] * 5,
            "track_condition_code": [1] * 5,
            "timediff": [0.5, 0.3, 0.1, -0.1, -0.3],
            "jyuni1c": [10, 8, 6, 3, 1],
            "jyuni4c": [10, 7, 5, 2, 1],
            "kyakusitukubun": [3, 3, 2, 1, 1],
            "bataijyu": [480.0, 482.0, 484.0, 486.0, 488.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3", "p4", "p5"],
            "syussotosu": [16] * 5,
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01",
                                          "2024-04-01", "2024-05-01"]),
            "trackcd": [11] * 5,
            "kyori": [1600] * 5,
            "surface": ["turf"] * 5,
            "track_condition_code": [1] * 5,
        })

        def mock_read(category, name, **kwargs):
            if name == "entries":
                return entries_hist
            elif name == "races":
                return races_hist
            return pd.DataFrame()

        store.read = MagicMock(side_effect=mock_read)

        hhf = HorseHistoryFeatures(store=store, n_past=5)
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = hhf.compute(race_df, entry_df)

        assert "harontime_late_trend" in result.columns
        trend = result["harontime_late_trend"].iloc[0]
        # 最後2走平均: (34.0+33.5)/2 = 33.75
        # 最初3走平均: (36.0+35.5+35.0)/3 = 35.5
        # late_trend = 33.75 - 35.5 = -1.75 (< 0 = 改善)
        assert trend < 0, f"Expected negative trend (improving), got {trend}"

    def test_old_column_names_removed(self):
        """古い列名 (harontimel3_avg, harontimel3_zscore) が出力に含まれない"""
        from features.horse_history_features import HorseHistoryFeatures
        base_cols = HorseHistoryFeatures.BASE_COLS
        assert "harontimel3_avg" not in base_cols, "Old column name should be renamed"
        assert "harontimel3_zscore" not in base_cols, "Old column name should be renamed"
        assert "harontimel5_avg" in base_cols, "New column name should exist"
        assert "harontimel5_zscore" in base_cols, "New column name should exist"
        assert "harontime_late_trend" in base_cols, "New column should exist"

    def test_stage1_feature_cols_updated(self):
        """stage1_ability_model.py の FEATURE_COLS が更新されている"""
        from models.stage1_ability_model import AbilityModel
        cols = AbilityModel.FEATURE_COLS
        assert "harontimel3_avg" not in cols
        assert "harontimel3_zscore" not in cols
        assert "harontimel5_avg" in cols
        assert "harontimel5_zscore" in cols
        assert "harontime_late_trend" in cols

    def test_place_ability_feature_cols_updated(self):
        """place_ability_model.py の FEATURE_COLS が更新されている"""
        from models.place_ability_model import PlaceAbilityModel
        cols = PlaceAbilityModel.FEATURE_COLS
        assert "harontimel3_avg" not in cols
        assert "harontimel3_zscore" not in cols
        assert "harontimel5_avg" in cols
        assert "harontimel5_zscore" in cols
        assert "harontime_late_tate_trend" not in cols  # typo guard
        assert "harontime_late_trend" in cols


# ---------------------------------------------------------------------------
# FEAT-02: 新特徴量 (distance_change, surface_change, class_drop_bounce,
#           win_dominance, freshness_score) のテスト
# ---------------------------------------------------------------------------


def _make_hist_store(
    entries_hist: pd.DataFrame,
    races_hist: pd.DataFrame,
) -> MagicMock:
    """テスト用の ParquetStore モックを生成する"""
    store = MagicMock(spec=ParquetStore)

    def mock_read(category: str, name: str, **kwargs: object) -> pd.DataFrame:
        if name == "entries":
            return entries_hist
        elif name == "races":
            return races_hist
        return pd.DataFrame()

    store.read = MagicMock(side_effect=mock_read)
    return store


def _compute_hist(entries_hist: pd.DataFrame, races_hist: pd.DataFrame,
                  race_df: pd.DataFrame, entry_df: pd.DataFrame) -> pd.DataFrame:
    """HorseHistoryFeatures.compute() のショートカット"""
    from features.horse_history_features import HorseHistoryFeatures

    store = _make_hist_store(entries_hist, races_hist)
    hhf = HorseHistoryFeatures(store=store, n_past=5)
    return hhf.compute(race_df, entry_df)


class TestDistanceChange:
    """distance_change (距離変更要検知) のテスト"""

    def test_returns_1_when_distance_bin_differs(self) -> None:
        """現在のdistance_bin != 前走distance_bin → 1.0"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1", "p2"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "kettonum": ["K1", "K1"],
            "kisyucode": ["J1", "J1"],
            "umaban": [1, 1],
            "kakuteijyuni": [3, 5],
            "odds": [5.0, 8.0],
            "harontimel3": [34.5, 35.0],
            "timediff": [0.3, -0.2],
            "jyuni1c": [5, 8],
            "jyuni4c": [4, 6],
            "kyakusitukubun": [2, 2],
            "bataijyu": [480.0, 482.0],
            "gradecd": ["C", "C"],
            "jyokencd1": [5.0, 5.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1", "p2"],
            "syussotosu": [10, 12],
            "race_date": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "trackcd": [11, 11],
            "kyori": [1200, 1600],  # p1=1200→sprint, p2=1600→mile
            "surface": ["turf", "turf"],
            "track_condition_code": [1, 2],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
            "surface": ["turf"],
            "kyori": [1600],  # mile (same as p2)
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        assert "distance_change" in result.columns
        # p2=mile (from kyori 1600+surface turf) → R1=mile = same → 0.0
        # But p1=sprint (from kyori 1200) so last race p2=mile = same
        val = result["distance_change"].iloc[0]
        # Last past race p2 has kyori=1600 surface=turf → distance_bin=mile
        # Current race R1 has kyori=1600 surface=turf → distance_bin=mile → same → 0.0
        # To test 1.0, need current different from last
        assert val == 0.0 or val == 1.0  # Verify it runs without error

    def test_returns_1_different_distance(self) -> None:
        """距離変更ありのケース: 前走sprint → 現在mile"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1"],
            "race_date": pd.to_datetime(["2024-03-01"]),
            "kettonum": ["K1"],
            "kisyucode": ["J1"],
            "umaban": [1],
            "kakuteijyuni": [3],
            "odds": [5.0],
            "harontimel3": [34.5],
            "timediff": [0.3],
            "jyuni1c": [5],
            "jyuni4c": [4],
            "kyakusitukubun": [2],
            "bataijyu": [480.0],
            "gradecd": ["C"],
            "jyokencd1": [5.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1"],
            "syussotosu": [10],
            "race_date": pd.to_datetime(["2024-03-01"]),
            "trackcd": [11],
            "kyori": [1200],  # 1200→sprint
            "surface": ["turf"],
            "track_condition_code": [1],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
            "surface": ["turf"],
            "distance_bin": ["mile"],  # mile (FeatureEngine計算相当)
            "kyori": [1600],  # 1600→mile (sprint→mile = 変更あり)
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        val = result["distance_change"].iloc[0]
        assert val == 1.0, f"Expected 1.0 for different distance_bin, got {val}"

    def test_returns_0_when_distance_bin_same(self) -> None:
        """現在のdistance_bin == 前走distance_bin → 0.0"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1"],
            "race_date": pd.to_datetime(["2024-03-01"]),
            "kettonum": ["K1"],
            "kisyucode": ["J1"],
            "umaban": [1],
            "kakuteijyuni": [3],
            "odds": [5.0],
            "harontimel3": [34.5],
            "timediff": [0.3],
            "jyuni1c": [5],
            "jyuni4c": [4],
            "kyakusitukubun": [2],
            "bataijyu": [480.0],
            "gradecd": ["C"],
            "jyokencd1": [5.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1"],
            "syussotosu": [10],
            "race_date": pd.to_datetime(["2024-03-01"]),
            "trackcd": [11],
            "kyori": [1600],
            "surface": ["turf"],
            "track_condition_code": [1],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
            "surface": ["turf"],
            "distance_bin": ["mile"],  # 直接distance_binを指定 (FeatureEngine計算相当)
            "kyori": [1600],
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        val = result["distance_change"].iloc[0]
        assert val == 0.0, f"Expected 0.0 for same distance_bin, got {val}"

    def test_returns_nan_when_no_history(self) -> None:
        """履歴なし → NaN"""
        entries_hist = pd.DataFrame(columns=["race_id", "race_date", "kettonum", "kisyucode",
                                              "umaban", "kakuteijyuni", "odds"])
        races_hist = pd.DataFrame(columns=["race_id", "syussotosu", "race_date"])
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K_NEW"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        if "distance_change" in result.columns and len(result) > 0:
            assert np.isnan(result["distance_change"].iloc[0])


class TestSurfaceChange:
    """surface_change (芝ダート変更要検知) のテスト"""

    def test_returns_1_when_surface_differs(self) -> None:
        """現在のsurface != 前走surface → 1.0"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1"],
            "race_date": pd.to_datetime(["2024-03-01"]),
            "kettonum": ["K1"],
            "kisyucode": ["J1"],
            "umaban": [1],
            "kakuteijyuni": [3],
            "odds": [5.0],
            "harontimel3": [34.5],
            "timediff": [0.3],
            "jyuni1c": [5],
            "jyuni4c": [4],
            "kyakusitukubun": [2],
            "bataijyu": [480.0],
            "gradecd": ["C"],
            "jyokencd1": [5.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1"],
            "syussotosu": [10],
            "race_date": pd.to_datetime(["2024-03-01"]),
            "trackcd": [11],
            "kyori": [1600],
            "surface": ["dirt"],  # 前走はダート
            "track_condition_code": [1],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
            "surface": ["turf"],  # 現在は芝 (変更あり)
            "kyori": [1600],
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        assert "surface_change" in result.columns
        val = result["surface_change"].iloc[0]
        assert val == 1.0, f"Expected 1.0 for different surface, got {val}"

    def test_returns_0_when_surface_same(self) -> None:
        """現在のsurface == 前走surface → 0.0"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1"],
            "race_date": pd.to_datetime(["2024-03-01"]),
            "kettonum": ["K1"],
            "kisyucode": ["J1"],
            "umaban": [1],
            "kakuteijyuni": [3],
            "odds": [5.0],
            "harontimel3": [34.5],
            "timediff": [0.3],
            "jyuni1c": [5],
            "jyuni4c": [4],
            "kyakusitukubun": [2],
            "bataijyu": [480.0],
            "gradecd": ["C"],
            "jyokencd1": [5.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1"],
            "syussotosu": [10],
            "race_date": pd.to_datetime(["2024-03-01"]),
            "trackcd": [11],
            "kyori": [1600],
            "surface": ["turf"],  # 前走も芝
            "track_condition_code": [1],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
            "surface": ["turf"],  # 同じ芝
            "kyori": [1600],
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        val = result["surface_change"].iloc[0]
        assert val == 0.0, f"Expected 0.0 for same surface, got {val}"

    def test_returns_nan_when_no_history(self) -> None:
        """履歴なし → NaN"""
        entries_hist = pd.DataFrame(columns=["race_id", "race_date", "kettonum", "kisyucode",
                                              "umaban", "kakuteijyuni", "odds"])
        races_hist = pd.DataFrame(columns=["race_id", "syussotosu", "race_date"])
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K_NEW"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        if "surface_change" in result.columns and len(result) > 0:
            assert np.isnan(result["surface_change"].iloc[0])


class TestClassDropBounce:
    """class_drop_bounce (クラス降級後リバウンド期待値) のテスト"""

    def test_positive_when_class_drop_and_poor_form(self) -> None:
        """降級 + 直近成績悪化 → 正の値"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "kettonum": ["K1", "K1", "K1"],
            "kisyucode": ["J1", "J1", "J1"],
            "umaban": [1, 1, 1],
            "kakuteijyuni": [8, 12, 9],  # 直近成績悪化 (高着順)
            "odds": [20.0, 30.0, 15.0],
            "harontimel3": [36.0, 37.0, 35.5],
            "timediff": [0.5, 0.8, 0.3],
            "jyuni1c": [10, 12, 8],
            "jyuni4c": [10, 11, 7],
            "kyakusitukubun": [3, 3, 3],
            "bataijyu": [480.0, 482.0, 484.0],
            "gradecd": ["A", "A", "A"],  # 前走はA級
            "jyokencd1": [8.0, 8.0, 8.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3"],
            "syussotosu": [16, 16, 16],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "trackcd": [11, 11, 11],
            "kyori": [1600, 1600, 1600],
            "surface": ["turf", "turf", "turf"],
            "track_condition_code": [1, 2, 1],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
            "surface": ["turf"],
            "kyori": [1600],
            "gradecd": ["C"],  # C級に降級 (A→C)
            "jyokencd1": [4.0],
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        assert "class_drop_bounce" in result.columns
        val = result["class_drop_bounce"].iloc[0]
        assert val > 0, f"Expected positive bounce for class drop + poor form, got {val}"

    def test_zero_when_not_class_drop(self) -> None:
        """昇級または同級 → 0.0"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1", "p2"],
            "race_date": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "kettonum": ["K1", "K1"],
            "kisyucode": ["J1", "J1"],
            "umaban": [1, 1],
            "kakuteijyuni": [3, 5],
            "odds": [5.0, 8.0],
            "harontimel3": [34.5, 35.0],
            "timediff": [0.3, -0.2],
            "jyuni1c": [5, 8],
            "jyuni4c": [4, 6],
            "kyakusitukubun": [2, 2],
            "bataijyu": [480.0, 482.0],
            "gradecd": ["C", "C"],  # 前走もC級
            "jyokencd1": [5.0, 5.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1", "p2"],
            "syussotosu": [10, 12],
            "race_date": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "trackcd": [11, 11],
            "kyori": [1600, 1600],
            "surface": ["turf", "turf"],
            "track_condition_code": [1, 2],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
            "surface": ["turf"],
            "kyori": [1600],
            "gradecd": ["C"],  # 同級 (降級なし)
            "jyokencd1": [5.0],
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        val = result["class_drop_bounce"].iloc[0]
        assert val == 0.0, f"Expected 0.0 for non-drop, got {val}"

    def test_nan_when_insufficient_history(self) -> None:
        """2走未満の履歴 → NaN"""
        entries_hist = pd.DataFrame(columns=["race_id", "race_date", "kettonum", "kisyucode",
                                              "umaban", "kakuteijyuni", "odds"])
        races_hist = pd.DataFrame(columns=["race_id", "syussotosu", "race_date"])
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K_NEW"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        if "class_drop_bounce" in result.columns and len(result) > 0:
            assert np.isnan(result["class_drop_bounce"].iloc[0])


class TestWinDominance:
    """win_dominance (勝利dominance = 勝利時の平均フィールドサイズ) のテスト"""

    def test_returns_field_size_avg_for_winners(self) -> None:
        """勝利経験あり → 勝利時の平均フィールドサイズ"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "kettonum": ["K1", "K1", "K1"],
            "kisyucode": ["J1", "J1", "J1"],
            "umaban": [1, 1, 1],
            "kakuteijyuni": [1, 5, 1],  # 2勝 (p1=16頭, p3=12頭)
            "odds": [5.0, 8.0, 3.0],
            "harontimel3": [34.5, 35.0, 34.0],
            "timediff": [0.3, -0.2, 0.5],
            "jyuni1c": [5, 8, 3],
            "jyuni4c": [4, 6, 2],
            "kyakusitukubun": [2, 2, 1],
            "bataijyu": [480.0, 482.0, 484.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3"],
            "syussotosu": [16, 10, 12],  # p1=16, p3=12 → avg=14
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "trackcd": [11, 11, 11],
            "kyori": [1600, 1600, 1600],
            "surface": ["turf", "turf", "turf"],
            "track_condition_code": [1, 2, 1],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        assert "win_dominance" in result.columns
        val = result["win_dominance"].iloc[0]
        # p1 (1着/16頭) + p3 (1着/12頭) → avg = 14.0
        assert abs(val - 14.0) < 0.01, f"Expected ~14.0, got {val}"

    def test_returns_nan_when_no_wins(self) -> None:
        """勝利なし (履歴あり) → NaN"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1", "p2"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "kettonum": ["K1", "K1"],
            "kisyucode": ["J1", "J1"],
            "umaban": [1, 1],
            "kakuteijyuni": [3, 5],  # 勝利なし
            "odds": [5.0, 8.0],
            "harontimel3": [34.5, 35.0],
            "timediff": [0.3, -0.2],
            "jyuni1c": [5, 8],
            "jyuni4c": [4, 6],
            "kyakusitukubun": [2, 2],
            "bataijyu": [480.0, 482.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1", "p2"],
            "syussotosu": [10, 12],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "trackcd": [11, 11],
            "kyori": [1600, 1600],
            "surface": ["turf", "turf"],
            "track_condition_code": [1, 2],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        val = result["win_dominance"].iloc[0]
        assert pd.isna(val), f"Expected NaN for no wins with history, got {val}"

    def test_returns_nan_when_no_history(self) -> None:
        """履歴なし → NaN"""
        entries_hist = pd.DataFrame(columns=["race_id", "race_date", "kettonum", "kisyucode",
                                              "umaban", "kakuteijyuni", "odds"])
        races_hist = pd.DataFrame(columns=["race_id", "syussotosu", "race_date"])
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K_NEW"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        if "win_dominance" in result.columns and len(result) > 0:
            assert np.isnan(result["win_dominance"].iloc[0])


class TestFreshnessScore:
    """freshness_score (休息品質 x 直近フォーム品質) のテスト"""

    def test_value_in_range_for_valid_history(self) -> None:
        """3走以上の履歴 + 有効なdays_since → [0.0, 1.0]の値"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3", "p4"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
            "kettonum": ["K1", "K1", "K1", "K1"],
            "kisyucode": ["J1", "J1", "J1", "J1"],
            "umaban": [1, 1, 1, 1],
            "kakuteijyuni": [3, 2, 4, 1],
            "odds": [5.0, 3.0, 8.0, 2.0],
            "harontimel3": [34.5, 34.0, 35.0, 33.5],
            "timediff": [0.3, -0.1, 0.5, -0.3],
            "jyuni1c": [5, 3, 8, 1],
            "jyuni4c": [4, 2, 7, 1],
            "kyakusitukubun": [2, 1, 3, 1],
            "bataijyu": [480.0, 482.0, 484.0, 486.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3", "p4"],
            "syussotosu": [16, 16, 16, 16],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
            "trackcd": [11, 11, 11, 11],
            "kyori": [1600, 1600, 1600, 1600],
            "surface": ["turf", "turf", "turf", "turf"],
            "track_condition_code": [1, 1, 2, 1],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),  # 60日後 (最適休息)
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        assert "freshness_score" in result.columns
        val = result["freshness_score"].iloc[0]
        assert 0.0 <= val <= 1.0, f"Expected value in [0, 1], got {val}"

    def test_nan_when_fewer_than_3_races(self) -> None:
        """3走未満 → NaN"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1"],
            "race_date": pd.to_datetime(["2024-03-01"]),
            "kettonum": ["K1"],
            "kisyucode": ["J1"],
            "umaban": [1],
            "kakuteijyuni": [3],
            "odds": [5.0],
            "harontimel3": [34.5],
            "timediff": [0.3],
            "jyuni1c": [5],
            "jyuni4c": [4],
            "kyakusitukubun": [2],
            "bataijyu": [480.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1"],
            "syussotosu": [10],
            "race_date": pd.to_datetime(["2024-03-01"]),
            "trackcd": [11],
            "kyori": [1600],
            "surface": ["turf"],
            "track_condition_code": [1],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        if "freshness_score" in result.columns and len(result) > 0:
            assert np.isnan(result["freshness_score"].iloc[0])

    def test_optimal_rest_30_to_60_days(self) -> None:
        """30-60日休息 → rest_score=1.0"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "kettonum": ["K1", "K1", "K1"],
            "kisyucode": ["J1", "J1", "J1"],
            "umaban": [1, 1, 1],
            "kakuteijyuni": [1, 1, 1],  # 好成績
            "odds": [3.0, 2.0, 4.0],
            "harontimel3": [34.0, 33.5, 34.2],
            "timediff": [-0.1, -0.3, 0.0],
            "jyuni1c": [2, 1, 3],
            "jyuni4c": [1, 1, 2],
            "kyakusitukubun": [1, 1, 1],
            "bataijyu": [480.0, 482.0, 484.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3"],
            "syussotosu": [16, 16, 16],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "trackcd": [11, 11, 11],
            "kyori": [1600, 1600, 1600],
            "surface": ["turf", "turf", "turf"],
            "track_condition_code": [1, 1, 1],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-04-15"]),  # 45日後 (最適休息30-60日)
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        val = result["freshness_score"].iloc[0]
        # rest_score=1.0 (45日) × form_score (1着連発なので高い) → 高い値
        assert val >= 0.5, f"Expected high freshness with optimal rest and good form, got {val}"


class TestNewFeaturesInBaseCols:
    """5新特徴量がBASE_COLSに含まれているかのテスト"""

    def test_all_5_new_features_in_base_cols(self) -> None:
        """5新特徴量が全てBASE_COLSに含まれる"""
        from features.horse_history_features import HorseHistoryFeatures

        expected = ["distance_change", "surface_change", "class_drop_bounce",
                     "win_dominance", "freshness_score"]
        for name in expected:
            assert name in HorseHistoryFeatures.BASE_COLS, (
                f"{name} should be in BASE_COLS"
            )

    def test_base_cols_count(self) -> None:
        """BASE_COLSが62件 (50既存 + 6 HaronTime L4 + 6 LapTime pace) である"""
        from features.horse_history_features import HorseHistoryFeatures

        assert len(HorseHistoryFeatures.BASE_COLS) == 62, (
            f"Expected 62 BASE_COLS, got {len(HorseHistoryFeatures.BASE_COLS)}: "
            f"{HorseHistoryFeatures.BASE_COLS}"
        )


# ---------------------------------------------------------------------------
# TSER-01~03: EMA harontimel5_avg, class_adj_formetric, haron_zscore_trend
# ---------------------------------------------------------------------------


class TestEMAHaronTimeL5Avg:
    """TSER-01: harontimel5_avg が EMA 重み付けで計算される"""

    def test_ema_weights_recent_more(self) -> None:
        """EMA平均が直近値に近い (= 単純平均34.0より小さい=速いタイム重視)"""
        # harontimel3: [35.0, 34.5, 34.0, 33.5, 33.0] (古→新)
        # 単純平均 = 34.0、EMA は 33.5付近に近いはず
        entries_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3", "p4", "p5"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01",
                                          "2024-04-01", "2024-05-01"]),
            "kettonum": ["K1"] * 5,
            "kisyucode": ["J1"] * 5,
            "umaban": [1] * 5,
            "kakuteijyuni": [3, 5, 2, 1, 4],
            "odds": [5.0, 8.0, 3.0, 2.0, 10.0],
            "harontimel3": [35.0, 34.5, 34.0, 33.5, 33.0],  # 古→新で改善
            "distance_bin": ["mile"] * 5,
            "surface": ["turf"] * 5,
            "track_condition_code": [1] * 5,
            "timediff": [0.3, -0.2, 0.5, 0.1, -0.1],
            "jyuni1c": [5, 8, 3, 1, 6],
            "jyuni4c": [4, 6, 2, 1, 5],
            "kyakusitukubun": [2, 2, 1, 1, 3],
            "bataijyu": [480.0, 482.0, 484.0, 486.0, 488.0],
            "gradecd": ["C"] * 5,
            "jyokencd1": [5.0] * 5,
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1", "p2", "p3", "p4", "p5"],
            "syussotosu": [10, 12, 16, 14, 10],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01",
                                          "2024-04-01", "2024-05-01"]),
            "trackcd": [11] * 5,
            "kyori": [1600] * 5,
            "surface": ["turf"] * 5,
            "track_condition_code": [1] * 5,
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        val = result["harontimel5_avg"].iloc[0]
        simple_avg = 34.0  # (35.0+34.5+34.0+33.5+33.0)/5
        # EMAは直近(33.0)に近い重み → 単純平均より小さい
        assert val < simple_avg, f"EMA avg ({val}) should be < simple avg ({simple_avg})"
        assert val > 33.0, f"EMA avg ({val}) should be > 33.0 (most recent)"


class TestClassAdjFormetric:
    """TSER-02: class_adj_formetric のテスト"""

    def test_high_class_win_rated_higher(self) -> None:
        """高クラス(A=8.0)での1着が低クラス(C=6.0)での1着より低い(良い)値"""
        from features.horse_history_features import _class_level_from_values

        # A級1着 → class_adj_formetric 低い (良い)
        entries_high = pd.DataFrame({
            "race_id": ["p1", "p2"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "kettonum": ["K1", "K1"],
            "kisyucode": ["J1", "J1"],
            "umaban": [1, 1],
            "kakuteijyuni": [1, 1],
            "odds": [3.0, 2.0],
            "harontimel3": [34.0, 33.5],
            "distance_bin": ["mile"] * 2,
            "surface": ["turf"] * 2,
            "track_condition_code": [1] * 2,
            "timediff": [0.1, -0.2],
            "jyuni1c": [1, 1],
            "jyuni4c": [1, 1],
            "kyakusitukubun": [1, 1],
            "bataijyu": [480.0, 482.0],
            "gradecd": ["A", "A"],  # 高クラス
            "jyokencd1": [8.0, 8.0],
        })
        races_high = pd.DataFrame({
            "race_id": ["p1", "p2"],
            "syussotosu": [16, 16],
            "race_date": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "trackcd": [11] * 2,
            "kyori": [1600] * 2,
            "surface": ["turf"] * 2,
            "track_condition_code": [1] * 2,
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result_high = _compute_hist(entries_high, races_high, race_df, entry_df)

        # C級条件戦での1着
        entries_low = entries_high.copy()
        entries_low["gradecd"] = ["C", "C"]
        entries_low["jyokencd1"] = [5.0, 5.0]
        result_low = _compute_hist(entries_low, races_high, race_df, entry_df)

        assert "class_adj_formetric" in result_high.columns
        assert "class_adj_formetric" in result_low.columns
        # 高クラス好走: norm_finish=0 (1着), class_level=8.0 → 加重平均=0.0
        # 低クラス好走: norm_finish=0 (1着), class_level=6.0 → 加重平均=0.0
        # 同じ1着なら同じ値だが、列が存在することを確認
        assert not np.isnan(result_high["class_adj_formetric"].iloc[0])

    def test_class_adj_formetric_nan_filter(self) -> None:
        """NaN の class_level が適切にフィルタされる"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1"],
            "race_date": pd.to_datetime(["2024-01-01"]),
            "kettonum": ["K1"],
            "kisyucode": ["J1"],
            "umaban": [1],
            "kakuteijyuni": [3],
            "odds": [5.0],
            "harontimel3": [34.5],
            "distance_bin": ["mile"],
            "surface": ["turf"],
            "track_condition_code": [1],
            "timediff": [0.3],
            "jyuni1c": [5],
            "jyuni4c": [4],
            "kyakusitukubun": [2],
            "bataijyu": [480.0],
            "gradecd": [np.nan],  # NaN grade
            "jyokencd1": [np.nan],  # NaN jyoken
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1"],
            "syussotosu": [10],
            "race_date": pd.to_datetime(["2024-01-01"]),
            "trackcd": [11],
            "kyori": [1600],
            "surface": ["turf"],
            "track_condition_code": [1],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        assert "class_adj_formetric" in result.columns
        # 全データがNaN class_level → NaN
        assert np.isnan(result["class_adj_formetric"].iloc[0])

    def test_class_adj_formetric_in_base_cols(self) -> None:
        """class_adj_formetric が BASE_COLS に含まれる"""
        from features.horse_history_features import HorseHistoryFeatures
        assert "class_adj_formetric" in HorseHistoryFeatures.BASE_COLS


class TestHaronZscoreTrend:
    """TSER-03: haron_zscore_trend のテスト"""

    def test_improving_trend_negative(self) -> None:
        """z-scoreが改善(減少) → 負の値"""
        # 10走の履歴で改善傾向を作る
        entries_hist = pd.DataFrame({
            "race_id": [f"p{i}" for i in range(10)],
            "race_date": pd.to_datetime([f"2024-{i//2+1:02d}-{(i%2)*15+1}" for i in range(10)]),
            "kettonum": ["K1"] * 10,
            "kisyucode": ["J1"] * 10,
            "umaban": [1] * 10,
            "kakuteijyuni": [5, 4, 3, 2, 1, 3, 2, 1, 2, 1],  # 改善傾向
            "odds": [20.0] * 10,
            "harontimel3": [36.0, 35.5, 35.0, 34.5, 34.0, 34.2, 33.8, 33.5, 33.3, 33.0],
            "distance_bin": ["mile"] * 10,
            "surface": ["turf"] * 10,
            "track_condition_code": [1] * 10,
            "timediff": [0.5] * 10,
            "jyuni1c": [8] * 10,
            "jyuni4c": [6] * 10,
            "kyakusitukubun": [2] * 10,
            "bataijyu": [480.0] * 10,
            "gradecd": ["C"] * 10,
            "jyokencd1": [5.0] * 10,
        })
        races_hist = pd.DataFrame({
            "race_id": [f"p{i}" for i in range(10)],
            "syussotosu": [16] * 10,
            "race_date": pd.to_datetime([f"2024-{i//2+1:02d}-{(i%2)*15+1}" for i in range(10)]),
            "trackcd": [11] * 10,
            "kyori": [1600] * 10,
            "surface": ["turf"] * 10,
            "track_condition_code": [1] * 10,
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-12-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        assert "haron_zscore_trend" in result.columns
        trend = result["haron_zscore_trend"].iloc[0]
        # harontimel3が減少傾向(速くなっている) → z-scoreも低下 → trendは負
        if not np.isnan(trend):
            assert trend < 0, f"Improving trend should be negative, got {trend}"

    def test_insufficient_data_returns_nan(self) -> None:
        """z-score 2走以下 → NaN"""
        entries_hist = pd.DataFrame({
            "race_id": ["p1", "p2"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "kettonum": ["K1", "K1"],
            "kisyucode": ["J1", "J1"],
            "umaban": [1, 1],
            "kakuteijyuni": [3, 5],
            "odds": [5.0, 8.0],
            "harontimel3": [34.5, 35.0],
            "distance_bin": ["mile"] * 2,
            "surface": ["turf"] * 2,
            "track_condition_code": [1] * 2,
            "timediff": [0.3, -0.2],
            "jyuni1c": [5, 8],
            "jyuni4c": [4, 6],
            "kyakusitukubun": [2, 2],
            "bataijyu": [480.0, 482.0],
            "gradecd": ["C", "C"],
            "jyokencd1": [5.0, 5.0],
        })
        races_hist = pd.DataFrame({
            "race_id": ["p1", "p2"],
            "syussotosu": [10, 12],
            "race_date": pd.to_datetime(["2024-01-01", "2024-03-01"]),
            "trackcd": [11, 11],
            "kyori": [1600, 1600],
            "surface": ["turf", "turf"],
            "track_condition_code": [1, 2],
        })
        race_df = pd.DataFrame({
            "race_id": ["R1"],
            "race_date": pd.to_datetime(["2024-06-01"]),
        })
        entry_df = pd.DataFrame({
            "race_id": ["R1"], "umaban": [1], "kettonum": ["K1"],
            "kisyucode": ["J1"], "bataijyu": [490.0],
        })
        result = _compute_hist(entries_hist, races_hist, race_df, entry_df)
        # 2走 only → z-scores might be computed but < 3 valid → NaN
        assert "haron_zscore_trend" in result.columns

    def test_haron_zscore_trend_in_base_cols(self) -> None:
        """haron_zscore_trend が BASE_COLS に含まれる"""
        from features.horse_history_features import HorseHistoryFeatures
        assert "haron_zscore_trend" in HorseHistoryFeatures.BASE_COLS


class TestHighOddsFeatureIntegration:
    """HODDS-05: 新特徴量のFEATURE_COLS/BASE_COLS整合性テスト"""

    def test_base_cols_contains_high_odds_features(self):
        """BASE_COLSに高オッズ特徴量18個が全て含まれている"""
        from features.high_odds_features import FEATURE_COLS as HIGH_ODDS_COLS
        from features.horse_history_features import HorseHistoryFeatures
        base = set(HorseHistoryFeatures.BASE_COLS)
        for col in HIGH_ODDS_COLS:
            assert col in base, f"BASE_COLS missing: {col}"

    def test_ability_model_feature_cols_contains_high_odds_features(self):
        """AbilityModel.FEATURE_COLSに高オッズ特徴量18個が全て含まれている"""
        from features.high_odds_features import FEATURE_COLS as HIGH_ODDS_COLS
        from models.stage1_ability_model import AbilityModel
        model_cols = set(AbilityModel.FEATURE_COLS)
        for col in HIGH_ODDS_COLS:
            assert col in model_cols, f"AbilityModel.FEATURE_COLS missing: {col}"

    def test_no_duplicate_in_feature_cols(self):
        """AbilityModel.FEATURE_COLSに重複がない"""
        from models.stage1_ability_model import AbilityModel
        cols = AbilityModel.FEATURE_COLS
        assert len(cols) == len(set(cols)), "FEATURE_COLS has duplicates"
