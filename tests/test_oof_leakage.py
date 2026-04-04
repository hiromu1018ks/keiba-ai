"""AbilityModel.train_oof() のテスト -- K-fold expanding window OOF でリークージを防止"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.stage1_ability_model import AbilityModel


def _make_oof_df(n_races: int = 40, horses_per_race: int = 8) -> pd.DataFrame:
    """train_oof() テスト用の DataFrame を生成する。

    AbilityModel.FEATURE_COLS 全30列 + race_id, race_date, umaban, finish_pos を含む。
    """
    rng = np.random.default_rng(123)
    n_rows = n_races * horses_per_race
    dates = pd.date_range("2020-01-01", periods=n_races, freq="20D")

    race_ids = [f"R{i:04d}" for i in range(n_races) for _ in range(horses_per_race)]
    race_dates = [d for d in dates for _ in range(horses_per_race)]
    umabans = [j + 1 for _ in range(n_races) for j in range(horses_per_race)]
    finish_positions = rng.integers(1, horses_per_race + 1, size=n_rows)

    df = pd.DataFrame(
        {
            "race_id": race_ids,
            "race_date": race_dates,
            "umaban": umabans,
            "finish_pos": finish_positions,
            # レース条件 (7)
            "surface": ["turf"] * n_rows,
            "distance_bin": ["mile"] * n_rows,
            "track_condition_code": np.ones(n_rows, dtype=int),
            "grade_code": ["_"] * n_rows,
            "field_size": np.full(n_rows, horses_per_race, dtype=int),
            "weight_diff_from_mean": rng.uniform(-3, 3, n_rows),
            "difficulty_score": rng.uniform(0, 1, n_rows),
            # 過去成績 (8)
            "norm_finish_logit_avg": rng.uniform(-2, 2, n_rows),
            "haron_time_l3_avg": rng.uniform(-3, 3, n_rows),
            "haron_time_l3_zscore": rng.uniform(-2, 2, n_rows),
            "time_diff_avg": rng.uniform(-1, 1, n_rows),
            "corner_1c_avg": rng.uniform(1, 10, n_rows),
            "corner_4c_avg": rng.uniform(1, 10, n_rows),
            "closing_index_avg": rng.uniform(-0.5, 0.5, n_rows),
            "kyakusitu_cd": rng.integers(1, 5, n_rows),
            # 血統 (6)
            "blood_surface_wr": rng.uniform(0.05, 0.2, n_rows),
            "blood_distance_wr": rng.uniform(0.05, 0.2, n_rows),
            "blood_condition_wr": np.full(n_rows, np.nan),  # Phase 2 placeholder
            "blood_total_wr": rng.uniform(0.05, 0.2, n_rows),
            "blood_prize_log": rng.uniform(10, 15, n_rows),
            "blood_keito_cd": np.full(n_rows, np.nan),  # Phase 2 placeholder
            # 交互作用 (3)
            "kyakusitu_x_distance": [f"{k}_mile" for k in rng.integers(1, 5, n_rows)],
            "kyakusitu_x_surface": [f"{k}_turf" for k in rng.integers(1, 5, n_rows)],
            "weight_x_distance": rng.uniform(640000, 880000, n_rows),
            # レース内正規化 (5)
            "norm_finish_logit_avg_race_rank": rng.uniform(0, 1, n_rows),
            "haron_time_l3_avg_race_rank": rng.uniform(0, 1, n_rows),
            "time_diff_avg_race_rank": rng.uniform(0, 1, n_rows),
            "corner_1c_avg_race_rank": rng.uniform(0, 1, n_rows),
            "closing_index_avg_race_rank": rng.uniform(0, 1, n_rows),
            # 馬体 (1)
            "weight_absolute": rng.uniform(400, 550, n_rows),
        }
    )
    return df


class TestTrainOof:
    """AbilityModel.train_oof() のテスト群"""

    def test_train_oof_returns_df_with_oof_predictions(self) -> None:
        """train_oof() は p_ability_win 列を含む DataFrame を返す"""
        df = _make_oof_df(n_races=40, horses_per_race=8)

        with (
            patch.object(AbilityModel, "train"),
            patch.object(
                AbilityModel,
                "add_ability_probs",
                side_effect=lambda d: d.assign(p_ability_win=np.full(len(d), 0.125)),
            ),
        ):
            model = AbilityModel()
            result = model.train_oof(df)

        assert "p_ability_win" in result.columns
        # 最初のfoldの予測はNaN、残りは非NaN
        non_nan = result["p_ability_win"].dropna()
        assert len(non_nan) > 0

    def test_train_oof_expanding_window_no_date_overlap(self) -> None:
        """各foldで train と predict の日付範囲が重ならない"""
        df = _make_oof_df(n_races=40, horses_per_race=8)
        train_calls: list[pd.DataFrame] = []
        predict_calls: list[pd.DataFrame] = []

        def mock_train(self: AbilityModel, train_df: pd.DataFrame, **kwargs: object) -> None:
            train_calls.append(train_df)

        def mock_predict(self: AbilityModel, test_df: pd.DataFrame) -> pd.DataFrame:
            predict_calls.append(test_df)
            return test_df.assign(p_ability_win=np.full(len(test_df), 0.1))

        with (
            patch.object(AbilityModel, "train", mock_train),
            patch.object(AbilityModel, "add_ability_probs", mock_predict),
        ):
            model = AbilityModel()
            model.train_oof(df, n_folds=3)

        # 最終モデルのtrain呼び出しを除外 (全データ)
        final_call = train_calls[-1]
        fold_train_calls = train_calls[:-1]

        for i, (tr, te) in enumerate(zip(fold_train_calls, predict_calls)):
            train_max_date = tr["race_date"].max()
            test_min_date = te["race_date"].min()
            assert train_max_date < test_min_date, (
                f"Fold {i}: train max date {train_max_date} >= predict min date {test_min_date}"
            )

    def test_train_oof_trains_final_model_on_all_data(self) -> None:
        """最後の train() 呼び出しは全データを受け取る (推論用モデル)"""
        df = _make_oof_df(n_races=40, horses_per_race=8)
        train_calls: list[pd.DataFrame] = []

        def mock_train(self: AbilityModel, train_df: pd.DataFrame, **kwargs: object) -> None:
            train_calls.append(train_df)

        with (
            patch.object(AbilityModel, "train", mock_train),
            patch.object(
                AbilityModel,
                "add_ability_probs",
                side_effect=lambda d: d.assign(p_ability_win=np.full(len(d), 0.1)),
            ),
        ):
            model = AbilityModel()
            model.train_oof(df, n_folds=3)

        # 最後の呼び出しが全データ
        assert len(train_calls) > 0
        final_df = train_calls[-1]
        assert len(final_df) == len(df), (
            f"Final model should get all {len(df)} rows, got {len(final_df)}"
        )

    def test_train_oof_first_fold_has_nan_predictions(self) -> None:
        """最初のfoldのテスト期間データは OOF 予測を持たない (NaN)"""
        df = _make_oof_df(n_races=40, horses_per_race=8)

        with (
            patch.object(AbilityModel, "train"),
            patch.object(
                AbilityModel,
                "add_ability_probs",
                side_effect=lambda d: d.assign(p_ability_win=np.full(len(d), 0.125)),
            ),
        ):
            model = AbilityModel()
            result = model.train_oof(df, n_folds=3)

        # 最初の ~25% のrace_dateは最初のfoldのトレーニング期間に相当
        dates = sorted(df["race_date"].unique())
        first_fold_end_idx = len(dates) // (3 + 1)  # n_folds=3 → 最初の境界
        first_fold_dates = set(dates[:first_fold_end_idx])
        first_fold_mask = df["race_date"].isin(first_fold_dates)

        # 最初のfoldのトレーニング期間データは OOF 予測が NaN
        first_fold_preds = result.loc[first_fold_mask, "p_ability_win"]
        assert first_fold_preds.isna().all(), (
            "First fold training period should have NaN predictions"
        )

    def test_train_oof_fallback_when_insufficient_data(self) -> None:
        """データ不足時は通常の train + add_ability_probs にフォールバック"""
        df = _make_oof_df(n_races=1, horses_per_race=8)
        train_calls: list[pd.DataFrame] = []

        def mock_train(self: AbilityModel, train_df: pd.DataFrame, **kwargs: object) -> None:
            train_calls.append(train_df)

        with (
            patch.object(AbilityModel, "train", mock_train),
            patch.object(
                AbilityModel,
                "add_ability_probs",
                side_effect=lambda d: d.assign(p_ability_win=np.full(len(d), 0.125)),
            ),
        ):
            model = AbilityModel()
            result = model.train_oof(df)

        # フォールバック: train 1回 + add_ability_probs 1回
        assert len(train_calls) == 1
        assert "p_ability_win" in result.columns
        assert result["p_ability_win"].notna().all()
