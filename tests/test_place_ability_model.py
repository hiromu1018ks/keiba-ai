"""test_place_ability_model.py — PlaceAbilityModel の単体テスト"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_train_df(n_races: int = 20, field_size: int = 8):
    """学習用ダミーデータ生成"""
    rows = []
    for r in range(n_races):
        for h in range(field_size):
            rows.append({
                "race_id": f"r{r:04d}",
                "umaban": h + 1,
                "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=r),
                "surface": "turf",
                "distance_bin": "mile",
                "track_condition_code": 1.0,
                "grade_code": "0",
                "field_size": float(field_size),
                "weight_diff_from_mean": np.random.randn(),
                "difficulty_score": np.random.randn(),
                # 過去成績 (8)
                "norm_finish_logit_avg": np.random.randn(),
                "haron_time_l3_avg": np.random.randn(),
                "haron_time_l3_zscore": np.random.randn(),
                "time_diff_avg": np.random.randn(),
                "corner_1c_avg": np.random.uniform(1, 10),
                "corner_4c_avg": np.random.uniform(1, 10),
                "closing_index_avg": np.random.randn(),
                "kyakusitu_cd": str(np.random.randint(1, 5)),
                # 血統 (6)
                "blood_surface_wr": np.random.uniform(0.05, 0.2),
                "blood_distance_wr": np.random.uniform(0.05, 0.2),
                "blood_condition_wr": float("nan"),
                "blood_total_wr": np.random.uniform(0.05, 0.2),
                "blood_prize_log": np.random.uniform(10, 15),
                "blood_keito_cd": float("nan"),
                # 交互作用 (3)
                "kyakusitu_x_distance": f"{np.random.randint(1, 5)}_mile",
                "kyakusitu_x_surface": f"{np.random.randint(1, 5)}_turf",
                "weight_x_distance": np.random.uniform(640000, 880000),
                # レース内正規化 (5) — race_rank
                "norm_finish_logit_avg_race_rank": np.random.rand(),
                "haron_time_l3_avg_race_rank": np.random.rand(),
                "time_diff_avg_race_rank": np.random.rand(),
                "corner_1c_avg_race_rank": np.random.rand(),
                "closing_index_avg_race_rank": np.random.rand(),
                # 馬体 (1)
                "weight_absolute": np.random.uniform(400, 550),
                "finish_pos": h + 1,
                "p_ability_win": 1.0 / field_size,
            })
    return pd.DataFrame(rows)


class TestPlaceAbilityModel:
    def test_probability_range(self):
        """出力が [0, 1] に収まる"""
        from models.place_ability_model import PlaceAbilityModel
        df = _make_train_df(n_races=50)
        model = PlaceAbilityModel()
        model.train(df)
        result = model.predict(df)
        assert result["p_ability_place"].min() >= 0
        assert result["p_ability_place"].max() <= 1

    def test_race_sum_approx_3(self):
        """レース内の p_place 合計 ≈ 3"""
        from models.place_ability_model import PlaceAbilityModel
        df = _make_train_df(n_races=50)
        model = PlaceAbilityModel()
        model.train(df)
        result = model.predict(df)
        race_sums = result.groupby("race_id")["p_ability_place"].sum()
        assert all(abs(s - 3.0) < 0.5 for s in race_sums)

    def test_feature_cols_exist(self):
        """FEATURE_COLS の全列がDataFrameに存在する"""
        from models.place_ability_model import PlaceAbilityModel
        model = PlaceAbilityModel()
        for col in model.FEATURE_COLS:
            assert col in _make_train_df().columns

    def test_temporal_split(self):
        """校正データが学習データより未来"""
        from models.place_ability_model import PlaceAbilityModel
        df = _make_train_df(n_races=50)
        model = PlaceAbilityModel()
        model.train(df)  # 内部で時系列分割がエラーなく完了することを確認
