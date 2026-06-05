"""test_place_ability_model.py — PlaceAbilityModel の単体テスト"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


def _make_train_df(n_races: int = 20, field_size: int = 8):
    """学習用ダミーデータ生成"""
    rows = []
    for r in range(n_races):
        for h in range(field_size):
            rows.append(
                {
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
                    "harontimel5_avg": np.random.randn(),
                    "harontimel5_zscore": np.random.randn(),
                    "harontime_late_trend": np.random.randn(),
                    "timediff_avg": np.random.randn(),
                    "jyuni1c_avg": np.random.uniform(1, 10),
                    "jyuni4c_avg": np.random.uniform(1, 10),
                    "closing_index_avg": np.random.randn(),
                    "kyakusitukubun_cd": str(np.random.randint(1, 5)),
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
                    "harontimel5_avg_race_rank": np.random.rand(),
                    "timediff_avg_race_rank": np.random.rand(),
                    "jyuni1c_avg_race_rank": np.random.rand(),
                    "closing_index_avg_race_rank": np.random.rand(),
                    # 馬体 (3)
                    "weight_absolute": np.random.uniform(400, 550),
                    "weight_zscore": np.random.uniform(-2, 2),
                    "weight_change_zone": float(np.random.choice([-1, 0, 1, 2])),
                    "weight_change_known": 1.0,
                    "weight_change_abs_capped": np.random.uniform(0, 20),
                    "weight_change_missing_flag": 0.0,
                    # 休養期間 (2)
                    "days_since_last_race": np.random.uniform(1, 200),
                    "rest_category": float(np.random.choice([1, 2, 3, 4, 5])),
                    "is_debut": 0.0,
                    # フォームサイクル (3)
                    "form_trend": np.random.uniform(-1, 1),
                    "form_consistency": np.random.uniform(0, 1),
                    "form_peak_flag": float(np.random.choice([0, 1])),
                    # 種牡馬産駎 (5)
                    "sire_wr": np.random.uniform(0.05, 0.2),
                    "sire_surface_wr": np.random.uniform(0.03, 0.15),
                    "sire_distance_wr": np.random.uniform(0.03, 0.15),
                    "sire_prize_avg": np.random.uniform(10, 15),
                    "bms_wr": np.random.uniform(0.02, 0.10),
                    "bms_distance_wr": np.random.uniform(0.02, 0.10),
                    "bms_surface_wr": np.random.uniform(0.02, 0.10),
                    "bms_has_history": 1.0,
                    "bms_starts_log": np.random.uniform(0.0, 5.0),
                    "bms_surface_starts_log": np.random.uniform(0.0, 5.0),
                    "bms_distance_starts_log": np.random.uniform(0.0, 5.0),
                    # ペース適性 (3)
                    "pace_aptitude": np.random.uniform(-0.5, 0.5),
                    "front_pace_wr": np.random.uniform(0.05, 0.3),
                    "closing_pace_wr": np.random.uniform(0.05, 0.3),
                    # コース適性 (2)
                    "course_wr": np.random.uniform(0.05, 0.3),
                    "course_distance_wr": np.random.uniform(0.05, 0.3),
                    # 追加改善特徴量
                    "draw_ratio": np.random.uniform(0.0, 1.0),
                    "class_level_current": np.random.uniform(1.0, 5.0),
                    "class_level_source_flag": 1.0,
                    "class_regime_after_202406": 0.0,
                    "class_move": np.random.uniform(-5.0, 5.0),
                    "blinker_change": float(np.random.choice([-1, 0, 1])),
                    "is_nar_transfer": float(np.random.choice([0, 1])),
                    "nar_recent_ratio": np.random.uniform(0.0, 1.0),
                    "track_condition_delta": np.random.uniform(-3.0, 3.0),
                    "pace_pressure": np.random.uniform(0.0, 1.0),
                    "pace_scenario_fit": np.random.uniform(-1.0, 1.0),
                    "kakuteijyuni": h + 1,
                    "p_ability_win": 1.0 / field_size,
                }
            )
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
        """FEATURE_COLS の主要列がDataFrameに存在する (v5コンテキスト特徴量は実行時のみ)"""
        from models.place_ability_model import PlaceAbilityModel

        model = PlaceAbilityModel()
        test_df = _make_train_df()
        # v5: レースコンテキスト特徴量はバックテスト/本番でのみ存在
        v5_context_cols = {"race_mean_fuku_odds", "race_std_fuku_odds",
                           "odds_popularity_gap", "surface_track_interaction",
                           # rl_* features added by Plan 34-01 (pipeline-only, not in test fixtures)
                           "rl_log_odds_entropy", "rl_odds_dispersion", "rl_top3_odds_gap",
                           "rl_top1_odds", "rl_favorite_rank_gap", "rl_n_horses",
                           "rl_favorite_in_wide_top1", "rl_trio_overlap", "rl_market_consistency",
                           "rl_trio_odds_ratio", "rl_wide_harville_ratio",
                           # FLB slope features (pipeline-only)
                           "implied_prob_hhi", "odds_skewness",
                           # Phase 36 TRF/HLF/interaction features (pipeline-only, not in test fixtures)
                           "form_trend_race_rank", "blood_total_wr_race_rank",
                           "blood_surface_wr_race_rank",
                           "grade_x_form_trend", "grade_x_blood_prize_log",
                           "distance_x_closing_index",
                           "closing_speed_ratio_avg", "closing_speed_ratio_zscore",
                           "closing_speed_ratio_trend", "closing_speed_ratio_avg_race_rank",
                           "harontime_last3f_avg", "harontime_last3f_zscore",
                           "harontime_last3f_trend", "harontime_last3f_avg_race_rank",
                           "pace_ratio_avg",
                           "pace_early_avg", "pace_mid_avg", "pace_late_avg",
                           "weighted_recent_form_finish", "weighted_recent_form_time",
                           # Phase 36.1 horse-level features
                           "haron_race_gap_avg", "haron_race_gap_zscore",
                           "haron_race_gap_trend",
                           # Phase 48: track condition features (pipeline-only)
                           "dirt_moisture_x_kyakusitu", "turf_cushion_track_relative",
                           "turf_cushion_track_zscore", "dirt_moisture_x_barrier_pos",
                           "dirt_moisture_high_flag", "dirt_moisture_dry_flag",
                           "turf_cushion_x_kyakusitu", "sire_x_cushion_band"}
        for col in model.FEATURE_COLS:
            if col in v5_context_cols:
                continue
            assert col in test_df.columns, f"{col} not in test data"

    def test_temporal_split(self):
        """校正データが学習データより未来"""
        from models.place_ability_model import PlaceAbilityModel

        df = _make_train_df(n_races=50)
        model = PlaceAbilityModel()
        model.train(df)  # 内部で時系列分割がエラーなく完了することを確認

    def test_untrained_predict_warns_once(self, caplog):
        """未学習predictを複数回呼んでもwarningが1回のみ (log-once)"""
        from models.place_ability_model import PlaceAbilityModel

        model = PlaceAbilityModel()
        df = pd.DataFrame({"race_id": ["r1"], "umaban": [1]})

        with caplog.at_level(logging.WARNING, logger="models.place_ability_model"):
            result1 = model.predict(df)
            result2 = model.predict(df)
            result3 = model.predict(df)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f"Expected 1 warning, got {len(warnings)}"
        assert "PlaceAbilityModel not trained" in warnings[0].message
        # NaN フォールバックが動作することも確認
        assert result1["p_ability_place"].isna().all()
        assert result2["p_ability_place"].isna().all()
