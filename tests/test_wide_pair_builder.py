"""WideJointPairBuilder のテスト"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.wide_pair_builder import WideJointPairBuilder


class TestWideJointPairBuilder:
    """WideJointPairBuilder のテスト"""

    @pytest.fixture
    def sample_entries(self) -> pd.DataFrame:
        """3頭のレースデータ — 生カラム名"""
        return pd.DataFrame(
            [
                {
                    "race_id": "20240101010101",
                    "umaban": 1,
                    "surface": "turf",
                    "distance_bin": "mile",
                    "track_condition_code": 2,
                    "grade_code": "C",
                    "field_size": 3,
                    "kakuteijyuni": 1,
                    "popularity_rank": 1,
                    "running_style": 1,
                    "odds": 3.0,
                    "wide_odds_1_2": 5.0,
                    "wide_odds_1_3": 8.0,
                    "wide_odds_2_3": 12.0,
                },
                {
                    "race_id": "20240101010101",
                    "umaban": 2,
                    "surface": "turf",
                    "distance_bin": "mile",
                    "track_condition_code": 2,
                    "grade_code": "C",
                    "field_size": 3,
                    "kakuteijyuni": 2,
                    "popularity_rank": 2,
                    "running_style": 2,
                    "odds": 5.0,
                    "wide_odds_1_2": 5.0,
                    "wide_odds_1_3": 8.0,
                    "wide_odds_2_3": 12.0,
                },
                {
                    "race_id": "20240101010101",
                    "umaban": 3,
                    "surface": "turf",
                    "distance_bin": "mile",
                    "track_condition_code": 2,
                    "grade_code": "C",
                    "field_size": 3,
                    "kakuteijyuni": 3,
                    "popularity_rank": 3,
                    "running_style": 3,
                    "odds": 10.0,
                    "wide_odds_1_2": 5.0,
                    "wide_odds_1_3": 8.0,
                    "wide_odds_2_3": 12.0,
                },
            ]
        )

    def test_build_creates_all_pairs(self, sample_entries: pd.DataFrame) -> None:
        """3頭から C(3,2)=3 ペアが生成される"""
        builder = WideJointPairBuilder()
        pairs = builder.build(sample_entries)
        assert len(pairs) == 3

    def test_build_has_required_columns(self, sample_entries: pd.DataFrame) -> None:
        """ペアDFに必要な列が含まれる"""
        builder = WideJointPairBuilder()
        pairs = builder.build(sample_entries)
        required = [
            "race_id",
            "umaban_a",
            "umaban_b",
            "surface",
            "distance_bin",
            "track_condition_code",
            "grade_code",
            "field_size",
            "joint_hit",
            "wide_odds",
            "popularity_sum",
            "running_style_combo",
        ]
        for col in required:
            assert col in pairs.columns, f"Missing column: {col}"

    def test_joint_hit_label(self, sample_entries: pd.DataFrame) -> None:
        """joint_hit: 両馬が3着以内なら1"""
        builder = WideJointPairBuilder()
        pairs = builder.build(sample_entries)
        # 全馬が3着以内なので全ペアが hit
        assert (pairs["joint_hit"] == 1).all()

    def test_joint_hit_with_outside_finish(self) -> None:
        """4着の馬を含むペアは joint_hit=0"""
        df = pd.DataFrame(
            [
                {
                    "race_id": "r1",
                    "umaban": 1,
                    "surface": "turf",
                    "distance_bin": "mile",
                    "track_condition_code": 2,
                    "grade_code": "C",
                    "field_size": 4,
                    "kakuteijyuni": 1,
                    "popularity_rank": 1,
                    "running_style": 1,
                    "odds": 3.0,
                    "wide_odds_1_2": 5.0,
                    "wide_odds_1_3": 8.0,
                    "wide_odds_1_4": 15.0,
                    "wide_odds_2_3": 12.0,
                    "wide_odds_2_4": 20.0,
                    "wide_odds_3_4": 25.0,
                },
                {
                    "race_id": "r1",
                    "umaban": 2,
                    "surface": "turf",
                    "distance_bin": "mile",
                    "track_condition_code": 2,
                    "grade_code": "C",
                    "field_size": 4,
                    "kakuteijyuni": 2,
                    "popularity_rank": 2,
                    "running_style": 2,
                    "odds": 5.0,
                    "wide_odds_1_2": 5.0,
                    "wide_odds_1_3": 8.0,
                    "wide_odds_1_4": 15.0,
                    "wide_odds_2_3": 12.0,
                    "wide_odds_2_4": 20.0,
                    "wide_odds_3_4": 25.0,
                },
                {
                    "race_id": "r1",
                    "umaban": 3,
                    "surface": "turf",
                    "distance_bin": "mile",
                    "track_condition_code": 2,
                    "grade_code": "C",
                    "field_size": 4,
                    "kakuteijyuni": 3,
                    "popularity_rank": 3,
                    "running_style": 3,
                    "odds": 10.0,
                    "wide_odds_1_2": 5.0,
                    "wide_odds_1_3": 8.0,
                    "wide_odds_1_4": 15.0,
                    "wide_odds_2_3": 12.0,
                    "wide_odds_2_4": 20.0,
                    "wide_odds_3_4": 25.0,
                },
                {
                    "race_id": "r1",
                    "umaban": 4,
                    "surface": "turf",
                    "distance_bin": "mile",
                    "track_condition_code": 2,
                    "grade_code": "C",
                    "field_size": 4,
                    "kakuteijyuni": 4,
                    "popularity_rank": 4,
                    "running_style": 4,
                    "odds": 20.0,
                    "wide_odds_1_2": 5.0,
                    "wide_odds_1_3": 8.0,
                    "wide_odds_1_4": 15.0,
                    "wide_odds_2_3": 12.0,
                    "wide_odds_2_4": 20.0,
                    "wide_odds_3_4": 25.0,
                },
            ]
        )
        builder = WideJointPairBuilder()
        pairs = builder.build(df)
        # ペア (3,4) は 3着+4着 → 4着が3着外なので joint_hit=0
        pair_3_4 = pairs[(pairs["umaban_a"] == 3) & (pairs["umaban_b"] == 4)]
        if len(pair_3_4) == 0:
            pair_3_4 = pairs[(pairs["umaban_a"] == 4) & (pairs["umaban_b"] == 3)]
        assert pair_3_4["joint_hit"].values[0] == 0

    def test_popularity_sum(self, sample_entries: pd.DataFrame) -> None:
        """popularity_sum = 人気順位の合計"""
        builder = WideJointPairBuilder()
        pairs = builder.build(sample_entries)
        for _, row in pairs.iterrows():
            a = row["umaban_a"]
            b = row["umaban_b"]
            entry_a = sample_entries[sample_entries["umaban"] == a].iloc[0]
            entry_b = sample_entries[sample_entries["umaban"] == b].iloc[0]
            expected = entry_a["popularity_rank"] + entry_b["popularity_rank"]
            assert row["popularity_sum"] == expected

    def test_wide_odds_lookup(self, sample_entries: pd.DataFrame) -> None:
        """wide_odds が正しく検索される"""
        builder = WideJointPairBuilder()
        pairs = builder.build(sample_entries)
        # ペア (1,2) → wide_odds_1_2 = 5.0
        pair_1_2 = pairs[
            ((pairs["umaban_a"] == 1) & (pairs["umaban_b"] == 2))
            | ((pairs["umaban_a"] == 2) & (pairs["umaban_b"] == 1))
        ]
        assert pair_1_2["wide_odds"].values[0] == 5.0

    def test_multiple_races(self) -> None:
        """複数レースが正しく処理される"""
        rows = []
        for race_id in ["r1", "r2"]:
            for i, pos in enumerate([1, 2, 3], 1):
                rows.append(
                    {
                        "race_id": race_id,
                        "umaban": i,
                        "surface": "turf",
                        "distance_bin": "mile",
                        "track_condition_code": 2,
                        "grade_code": "C",
                        "field_size": 3,
                        "kakuteijyuni": pos,
                        "popularity_rank": i,
                        "running_style": i,
                        "odds": float(i * 3),
                        "wide_odds_1_2": 5.0,
                        "wide_odds_1_3": 8.0,
                        "wide_odds_2_3": 12.0,
                    }
                )
        df = pd.DataFrame(rows)
        builder = WideJointPairBuilder()
        pairs = builder.build(df)
        assert len(pairs) == 6  # 3 pairs per race × 2 races


class TestWideTwoStageModelTraining:
    """WideTwoStageModel の学習メソッドのテスト"""

    @pytest.fixture
    def sample_pair_df(self) -> pd.DataFrame:
        """学習用ペアDataFrame (min_hit_samples=200 を満たすよう十分な的中ペアを含む)"""
        np.random.seed(42)
        n = 1000
        hits = np.random.binomial(1, 0.5, n)
        # 的中ペアが min_hit_samples (200) 以上になることを保証
        if hits.sum() < 200:
            hits[:200] = 1
        return pd.DataFrame(
            {
                "race_id": [f"r{i // 10}" for i in range(n)],
                "umaban_a": np.random.randint(1, 10, n),
                "umaban_b": np.random.randint(1, 10, n),
                "surface": np.random.choice(["turf", "dirt"], n),
                "distance_bin": np.random.choice(["sprint", "mile", "intermediate", "long"], n),
                "track_condition_code": np.random.randint(1, 4, n),
                "grade_code": np.random.choice(["A", "B", "C", "D", "E"], n),
                "field_size": np.random.randint(8, 18, n),
                "joint_hit": hits,
                "wide_odds": np.random.uniform(3.0, 50.0, n),
                "popularity_sum": np.random.randint(2, 20, n),
                "running_style_combo": np.random.randint(2, 8, n),
            }
        )

    def test_train_hit_model_sets_model(self, sample_pair_df: pd.DataFrame) -> None:
        """train_hit_model が hit_model を設定する"""
        from models.wide_two_stage_model import WideTwoStageModel

        model = WideTwoStageModel()
        model.train_hit_model(sample_pair_df)
        assert model.hit_model is not None

    def test_train_return_model_sets_model(self, sample_pair_df: pd.DataFrame) -> None:
        """train_return_model が return_model を設定する"""
        from models.wide_two_stage_model import WideTwoStageModel

        model = WideTwoStageModel()
        model.train_return_model(sample_pair_df)
        assert model.return_model is not None

    def test_predict_score_after_training(self, sample_pair_df: pd.DataFrame) -> None:
        """学習後に predict_score が動作する"""
        from models.wide_two_stage_model import WideTwoStageModel

        model = WideTwoStageModel()
        model.train_hit_model(sample_pair_df)
        model.train_return_model(sample_pair_df)
        result = model.predict_score(sample_pair_df.head(10))
        assert "p_hit" in result.columns
        assert "e_return_given_hit" in result.columns
        assert "ev_wide" in result.columns
        assert "wide_score_adj" in result.columns
        assert (result["p_hit"] >= 0).all()
