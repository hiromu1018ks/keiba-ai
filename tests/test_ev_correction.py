"""src/models/ev_correction_model.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.ev_correction_model import EVCorrectionModel


@pytest.fixture
def pre_ev_df() -> pd.DataFrame:
    """WinTwoStageModel.predict_ev() 出力後のテストデータ — 生カラム名"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 8,
            "kakuteijyuni": [1, 2, 3, 4, 5, 6, 7, 8],
            "odds": [4.0, 6.0, 9.0, 16.0, 28.0, 45.0, 90.0, 160.0],
            "p_win_pred": [0.28, 0.22, 0.18, 0.12, 0.08, 0.06, 0.04, 0.02],
            "e_return_win_pred": [4.0, 6.0, 9.0, 16.0, 28.0, 45.0, 90.0, 160.0],
            "ev_win": [1.12, 1.32, 1.62, 1.92, 2.24, 2.70, 3.60, 3.20],
            "signed_log_error_win": [0.1, -0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.3],
            "abs_log_error_win": [0.1, 0.1, 0.2, 0.3, 0.0, 0.5, 0.2, 0.3],
            "market_entropy": [2.5] * 8,
            "popularity_rank": [1, 2, 3, 4, 5, 6, 7, 8],
            # FLB slope (市場集中度)
            "implied_prob_hhi": [0.08] * 8,
            "surface": ["turf"] * 8,
            "distance_bin": ["mile"] * 8,
            "track_condition_code": [1] * 8,
            "field_size": [8] * 8,
            # 騎手コンテキスト (Group C)
            "jockey_wr_overall": [0.12] * 8,
            "jockey_wr_distance": [0.10] * 8,
            "jockey_wr_venue": [0.11] * 8,
            "jockey_prize_log": [11.0] * 8,
            # 調教師コンテキスト (Group D)
            "trainer_wr_overall": [0.14] * 8,
            "trainer_wr_distance": [0.12] * 8,
            "trainer_wr_venue": [0.13] * 8,
            "trainer_prize_log": [10.5] * 8,
            # 騎手-調教師コンビ (B4)
            "jt_combo_wr": [0.15] * 8,
            "jt_combo_place_rate": [0.25] * 8,
            "jt_combo_starts": [5.0] * 8,
            "jt_combo_prize_log": [4.0] * 8,
        }
    )


@pytest.fixture
def trained_ev_model(pre_ev_df: pd.DataFrame) -> EVCorrectionModel:
    """学習済みEVCorrectionModel (mock)"""
    model = EVCorrectionModel()
    # P補正: 小さい補正値 (correction_logit)
    mock_p = MagicMock()
    mock_p.best_iteration = 100  # 早期停止後の best_iteration
    mock_p.predict.return_value = np.array(
        [
            0.01,
            -0.01,
            0.02,
            -0.02,
            0.0,
            0.03,
            -0.03,
            0.01,
        ]
    )
    # E補正: log residual
    mock_e = MagicMock()
    mock_e.best_iteration = 80  # 早期停止後の best_iteration
    mock_e.predict.return_value = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    model.p_correction_model = mock_p
    model.e_correction_model = mock_e
    return model


class TestEVCorrectionModel:
    def test_p_corrected_in_0_1(
        self,
        trained_ev_model: EVCorrectionModel,
        pre_ev_df: pd.DataFrame,
    ) -> None:
        """P_corrected が [0, 1] に制約される"""
        result = trained_ev_model.correct_ev(pre_ev_df)
        assert (result["p_win_corrected"] >= 0).all()
        assert (result["p_win_corrected"] <= 1).all()

    def test_ev_corrected_equals_p_times_e(
        self,
        trained_ev_model: EVCorrectionModel,
        pre_ev_df: pd.DataFrame,
    ) -> None:
        """EV_corrected = P_corrected × E_corrected"""
        result = trained_ev_model.correct_ev(pre_ev_df)
        expected = result["p_win_corrected"] * result["e_return_win_corrected"]
        assert np.allclose(result["ev_win_corrected"].values, expected.values, atol=1e-10)

    def test_e_corrected_positive(
        self,
        trained_ev_model: EVCorrectionModel,
        pre_ev_df: pd.DataFrame,
    ) -> None:
        """E_corrected は正値 (オッズは1.0以上)"""
        result = trained_ev_model.correct_ev(pre_ev_df)
        assert (result["e_return_win_corrected"] > 0).all()

    def test_feature_cols_no_p_win_pred(self) -> None:
        """p_win_pred は特徴量から除外される (init_scoreで代替)"""
        assert "p_win_pred" not in EVCorrectionModel.FEATURE_COLS

    def test_has_interaction_features(self) -> None:
        """交互作用特徴量が含まれる"""
        assert "p_x_e_interaction" in EVCorrectionModel.FEATURE_COLS
        assert "p_minus_e_gap" in EVCorrectionModel.FEATURE_COLS

    def test_train_asserts_ev_win(self, pre_ev_df: pd.DataFrame) -> None:
        """train() は ev_win 列を要求する"""
        model = EVCorrectionModel()
        bad_df = pre_ev_df.drop(columns=["ev_win"])
        with pytest.raises(AssertionError, match="ev_win"):
            model.train(bad_df)

    def test_e_clip_floor(self) -> None:
        assert EVCorrectionModel.E_CLIP_FLOOR == 1.0

    def test_ev_correction_no_random_split(self) -> None:
        """EVCorrectionModel.train() がランダム分割を使わないことを確認"""
        import inspect

        from models.ev_correction_model import EVCorrectionModel

        source = inspect.getsource(EVCorrectionModel.train)
        assert "permutation" not in source, "Still using random permutation in train()!"
        assert "RandomState" not in source, "Still using RandomState in train()!"

    def test_correct_ev_uses_best_iteration(
        self,
        trained_ev_model: EVCorrectionModel,
        pre_ev_df: pd.DataFrame,
    ) -> None:
        """correct_ev が best_iteration を使って predict を呼び出す"""
        trained_ev_model.correct_ev(pre_ev_df)
        # P補正モデルの predict が num_iteration で呼ばれることを確認
        trained_ev_model.p_correction_model.predict.assert_called_once()
        call_kwargs_p = trained_ev_model.p_correction_model.predict.call_args
        assert call_kwargs_p.kwargs.get("num_iteration") == 100
        # E補正モデルの predict も同様
        trained_ev_model.e_correction_model.predict.assert_called_once()
        call_kwargs_e = trained_ev_model.e_correction_model.predict.call_args
        assert call_kwargs_e.kwargs.get("num_iteration") == 80

    def test_correct_ev_best_iteration_zero_uses_none(self, pre_ev_df: pd.DataFrame) -> None:
        """best_iteration が 0 の場合は num_iteration=None になる"""
        model = EVCorrectionModel()
        mock_p = MagicMock()
        mock_p.best_iteration = 0
        mock_p.predict.return_value = np.zeros(len(pre_ev_df))
        mock_e = MagicMock()
        mock_e.best_iteration = 0
        mock_e.predict.return_value = np.zeros(len(pre_ev_df))
        model.p_correction_model = mock_p
        model.e_correction_model = mock_e

        model.correct_ev(pre_ev_df)
        assert mock_p.predict.call_args.kwargs.get("num_iteration") is None
        assert mock_e.predict.call_args.kwargs.get("num_iteration") is None


@pytest.fixture
def large_ev_df() -> pd.DataFrame:
    """EV補正の統計テスト用 大規模データ (200行, winner 60+頭, 中穴ゾーン 120+行) — 生カラム名"""
    np.random.seed(123)
    n_races = 200
    rows: list[dict] = []
    for i in range(n_races):
        n_horses = np.random.randint(8, 16)
        p_preds = np.sort(np.random.dirichlet(np.ones(n_horses) * 0.5))[::-1]
        for j in range(n_horses):
            finish = j + 1
            odds = max(1.1, np.random.lognormal(2.0, 0.7))
            rows.append(
                {
                    "race_id": f"LR{i:04d}",
                    "kakuteijyuni": finish,
                    "odds": odds,
                    "p_win_pred": float(p_preds[j]),
                    "e_return_win_pred": odds,
                    "ev_win": float(p_preds[j]) * odds,
                    "signed_log_error_win": np.random.normal(0, 0.3),
                    "abs_log_error_win": abs(np.random.normal(0, 0.3)),
                    "market_entropy": float(np.random.uniform(2.0, 3.5)),
                    "popularity_rank": j + 1,
                    # FLB slope (市場集中度)
                    "implied_prob_hhi": float(np.random.uniform(0.05, 0.15)),
                    "surface": np.random.choice(["turf", "dirt"]),
                    "distance_bin": np.random.choice(["sprint", "mile", "intermediate", "long"]),
                    "track_condition_code": 1,
                    "field_size": n_horses,
                    # 騎手コンテキスト (Group C)
                    "jockey_wr_overall": float(np.random.uniform(0.05, 0.20)),
                    "jockey_wr_distance": float(np.random.uniform(0.03, 0.18)),
                    "jockey_wr_venue": float(np.random.uniform(0.04, 0.19)),
                    "jockey_prize_log": float(np.random.uniform(8.0, 12.0)),
                    # 調教師コンテキスト (Group D)
                    "trainer_wr_overall": float(np.random.uniform(0.05, 0.20)),
                    "trainer_wr_distance": float(np.random.uniform(0.03, 0.18)),
                    "trainer_wr_venue": float(np.random.uniform(0.04, 0.19)),
                    "trainer_prize_log": float(np.random.uniform(7.0, 11.5)),
                    # 騎手-調教師コンビ (B4)
                    "jt_combo_wr": float(np.random.uniform(0.05, 0.20)),
                    "jt_combo_place_rate": float(np.random.uniform(0.10, 0.35)),
                    "jt_combo_starts": float(np.random.randint(1, 30)),
                    "jt_combo_prize_log": float(np.random.uniform(2.0, 6.0)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_ev_model_large(large_ev_df: pd.DataFrame) -> EVCorrectionModel:
    """大規模データ用 EVCorrectionModel (mock: winner の P を引き上げる補正)"""
    np.random.seed(456)
    model = EVCorrectionModel()
    n = len(large_ev_df)
    is_winner = large_ev_df["kakuteijyuni"].values == 1
    is_mid_range = large_ev_df["p_win_pred"].between(0.05, 0.15).values

    # P補正: winner を引き上げ、mid_range の非winner を押し下げ
    p_corrections = np.where(
        is_winner,
        np.random.uniform(0.3, 0.8, n),  # winner: 大幅にPを引き上げ
        np.where(
            is_mid_range,
            np.random.uniform(-0.6, -0.2, n),  # mid_range非winner: Pを押し下げ
            np.random.uniform(-0.3, 0.0, n),  # その他非winner: やや押し下げ
        ),
    )
    mock_p = MagicMock()
    mock_p.best_iteration = 120
    mock_p.predict.return_value = p_corrections
    mock_e = MagicMock()
    mock_e.best_iteration = 90
    mock_e.predict.return_value = np.random.normal(0, 0.02, n)
    model.p_correction_model = mock_p
    model.e_correction_model = mock_e
    return model


class TestEVCorrectionLargeData:
    """EV補正モデル 大規模データ統計テスト (§13.1)"""

    def test_ev_correction_reduces_error(
        self,
        trained_ev_model_large: EVCorrectionModel,
        large_ev_df: pd.DataFrame,
    ) -> None:
        """EV補正モデルがEVのMAEを改善することを確認 (§13.1)"""
        result = trained_ev_model_large.correct_ev(large_ev_df)
        actual_ev = result["odds"] * (result["kakuteijyuni"] == 1).astype(int)
        mae_raw = float(np.mean(np.abs(result["ev_win"] - actual_ev)))
        mae_corrected = float(np.mean(np.abs(result["ev_win_corrected"] - actual_ev)))
        assert mae_corrected < mae_raw, (
            f"EV補正後のMAE({mae_corrected:.4f})が補正前({mae_raw:.4f})より大きい"
        )

    def test_ev_correction_mid_range_improvement(
        self,
        trained_ev_model_large: EVCorrectionModel,
        large_ev_df: pd.DataFrame,
    ) -> None:
        """中穴ゾーン(P=0.05-0.15)で補正改善>10% (§13.1)"""
        result = trained_ev_model_large.correct_ev(large_ev_df)
        mid_range = result[result["p_win_pred"].between(0.05, 0.15)].copy()
        if len(mid_range) < 100:
            pytest.skip("中穴ゾーンのサンプル不足")
        actual_ev = mid_range["odds"] * (mid_range["kakuteijyuni"] == 1).astype(int)
        mae_raw = float(np.mean(np.abs(mid_range["ev_win"] - actual_ev)))
        mae_corrected = float(np.mean(np.abs(mid_range["ev_win_corrected"] - actual_ev)))
        if mae_raw == 0:
            pytest.skip("MAE_raw がゼロのため改善率を計算できない")
        improvement = (mae_raw - mae_corrected) / mae_raw
        assert improvement > 0.10, f"中穴ゾーンの補正改善率が低い: {improvement:.1%}"

    def test_ev_correction_winner_weight(
        self,
        trained_ev_model_large: EVCorrectionModel,
        large_ev_df: pd.DataFrame,
    ) -> None:
        """1着馬のP_corrected中央値>=P_pred中央値 (§13.1)"""
        result = trained_ev_model_large.correct_ev(large_ev_df)
        winners = result[result["kakuteijyuni"] == 1]
        if len(winners) < 50:
            pytest.skip("1着馬サンプル不足")
        assert winners["p_win_corrected"].median() >= winners["p_win_pred"].median(), (
            "P補正が1着馬の確率を適切に引き上げていません"
        )


# --- PlaceEVCorrectionModel tests ---


def _make_mock_booster(predictions: np.ndarray) -> MagicMock:
    """Mock LightGBM booster with given predictions and best_iteration=100."""
    mock = MagicMock()
    mock.best_iteration = 100
    mock.predict.return_value = predictions
    return mock


@pytest.fixture
def pre_place_ev_df():
    """PlaceEVCorrectionModel.correct_ev() の入力 DataFrame (8行)"""
    n = 8
    return pd.DataFrame({
        "race_id": ["R001"] * n,
        "umaban": list(range(1, n + 1)),
        "kakuteijyuni": [1, 2, 3, 4, 5, 6, 7, 8],
        "p_place_pred": [0.65, 0.55, 0.50, 0.40, 0.30, 0.25, 0.20, 0.10],
        "e_return_place_pred": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0],
        "fukuoddslow": [1.3, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 5.0],
        "p_ability_place": [0.60, 0.52, 0.45, 0.38, 0.30, 0.22, 0.18, 0.10],
        "signed_log_error_win": [0.1, -0.05, 0.02, -0.1, 0.15, -0.08, 0.03, -0.12],
        "abs_log_error_win": [0.1, 0.05, 0.02, 0.1, 0.15, 0.08, 0.03, 0.12],
        "market_entropy": [2.5] * n,
        "popularity_rank": list(range(1, n + 1)),
        "surface": ["turf"] * n,
        "distance_bin": [2] * n,
        "track_condition_code": [1] * n,
        "field_size": [n] * n,
        "jockey_wr_overall": [0.3] * n,
        "jockey_wr_distance": [0.25] * n,
        "jockey_wr_venue": [0.28] * n,
        "jockey_prize_log": [13.0] * n,
        "trainer_wr_overall": [0.2] * n,
        "trainer_wr_distance": [0.18] * n,
        "trainer_wr_venue": [0.22] * n,
        "trainer_prize_log": [12.0] * n,
        "jt_combo_wr": [0.15] * n,
        "jt_combo_place_rate": [0.4] * n,
        "jt_combo_starts": [50] * n,
        "jt_combo_prize_log": [12.0] * n,
        "implied_prob_hhi": [0.15] * n,
    })


class TestPlaceEVCorrectionModel:
    def test_correct_ev_outputs_place_columns(self, pre_place_ev_df):
        """correct_ev should output p_place_corrected, e_return_place_corrected, ev_place_corrected"""
        from models.ev_correction_model import PlaceEVCorrectionModel

        model = PlaceEVCorrectionModel()
        model.p_correction_model = _make_mock_booster(np.array([0.1, -0.05, 0.02, -0.03, 0.08, -0.01, 0.04, -0.06]))
        model.e_correction_model = _make_mock_booster(np.array([0.05, -0.02, 0.01, -0.04, 0.03, -0.01, 0.02, -0.03]))
        model._trained = True

        result = model.correct_ev(pre_place_ev_df)

        assert "p_place_corrected" in result.columns
        assert "e_return_place_corrected" in result.columns
        assert "ev_place_corrected" in result.columns

    def test_place_p_corrected_bounds(self, pre_place_ev_df):
        """P(place) corrected should be in [0, 1]"""
        from models.ev_correction_model import PlaceEVCorrectionModel

        model = PlaceEVCorrectionModel()
        model.p_correction_model = _make_mock_booster(np.array([2.0, -3.0, 1.0, -1.0, 1.5, -2.0, 0.5, -1.5]))
        model.e_correction_model = _make_mock_booster(np.zeros(8))
        model._trained = True

        result = model.correct_ev(pre_place_ev_df)

        assert (result["p_place_corrected"] >= 0).all()
        assert (result["p_place_corrected"] <= 1).all()

    def test_place_e_corrected_positive(self, pre_place_ev_df):
        """E(return|place) corrected should always be positive"""
        from models.ev_correction_model import PlaceEVCorrectionModel

        model = PlaceEVCorrectionModel()
        model.p_correction_model = _make_mock_booster(np.zeros(8))
        model.e_correction_model = _make_mock_booster(np.array([-0.1, 0.1, -0.05, 0.05, -0.08, 0.08, -0.03, 0.03]))
        model._trained = True

        result = model.correct_ev(pre_place_ev_df)

        assert (result["e_return_place_corrected"] > 0).all()

    def test_place_ev_decomposition(self, pre_place_ev_df):
        """ev_place_corrected = p_place_corrected * e_return_place_corrected"""
        from models.ev_correction_model import PlaceEVCorrectionModel

        model = PlaceEVCorrectionModel()
        model.p_correction_model = _make_mock_booster(np.array([0.1, -0.05, 0.02, -0.03, 0.08, -0.01, 0.04, -0.06]))
        model.e_correction_model = _make_mock_booster(np.array([0.05, -0.02, 0.01, -0.04, 0.03, -0.01, 0.02, -0.03]))
        model._trained = True

        result = model.correct_ev(pre_place_ev_df)

        expected = result["p_place_corrected"] * result["e_return_place_corrected"]
        assert np.allclose(result["ev_place_corrected"], expected, atol=1e-10)

    def test_untrained_fallback_passes_through(self, pre_place_ev_df):
        """Untrained model should pass through ev_place as ev_place_corrected"""
        from models.ev_correction_model import PlaceEVCorrectionModel

        model = PlaceEVCorrectionModel()  # _trained = False

        # ev_place 列を事前に設定
        pre_place_ev_df["ev_place"] = pre_place_ev_df["p_place_pred"] * pre_place_ev_df["e_return_place_pred"]
        result = model.correct_ev(pre_place_ev_df)

        # フォールバック: ev_place_corrected == ev_place
        assert "ev_place_corrected" in result.columns
        assert np.allclose(result["ev_place_corrected"], result["ev_place"])


class TestEVCorrectionTemporalSplit:
    """EV補正モデルの時系列分割テスト (PITリーク防止)"""

    def test_train_sorts_by_race_date_before_split(self):
        """train() は race_date でソートしてから train/valid 分割すること"""
        import lightgbm as lgb
        from models.ev_correction_model import EVCorrectionModel

        np.random.seed(42)
        n = 100
        # 時系列順でないデータ（race_date がランダム）
        dates = pd.date_range("2020-01-01", periods=n).to_numpy()
        np.random.shuffle(dates)  # 意図的にシャッフル

        df = pd.DataFrame({
            "race_id": [f"R{i:04d}" for i in range(n)],
            "race_date": dates,
            "kakuteijyuni": np.random.randint(1, 9, n),
            "odds": np.random.uniform(1.1, 100, n),
            "confirmed_odds": np.random.uniform(1.1, 100, n),
            "p_win_pred": np.random.uniform(0.01, 0.5, n),
            "e_return_win_pred": np.random.uniform(1.1, 100, n),
            "ev_win": np.random.uniform(0.5, 10, n),
            "signed_log_error_win": np.random.normal(0, 0.3, n),
            "abs_log_error_win": np.abs(np.random.normal(0, 0.3, n)),
            "market_entropy": np.random.uniform(2.0, 3.5, n),
            "popularity_rank": np.random.randint(1, 10, n),
            "implied_prob_hhi": np.random.uniform(0.05, 0.15, n),
            "surface": np.random.choice(["turf", "dirt"], n),
            "distance_bin": np.random.choice(["sprint", "mile", "long"], n),
            "track_condition_code": np.random.randint(1, 4, n),
            "field_size": np.random.randint(8, 16, n),
            "jockey_wr_overall": np.random.uniform(0.05, 0.20, n),
            "jockey_wr_distance": np.random.uniform(0.03, 0.18, n),
            "jockey_wr_venue": np.random.uniform(0.04, 0.19, n),
            "jockey_prize_log": np.random.uniform(8.0, 12.0, n),
            "trainer_wr_overall": np.random.uniform(0.05, 0.20, n),
            "trainer_wr_distance": np.random.uniform(0.03, 0.18, n),
            "trainer_wr_venue": np.random.uniform(0.04, 0.19, n),
            "trainer_prize_log": np.random.uniform(7.0, 11.5, n),
            "jt_combo_wr": np.random.uniform(0.05, 0.20, n),
            "jt_combo_place_rate": np.random.uniform(0.10, 0.35, n),
            "jt_combo_starts": np.random.uniform(1, 30, n),
            "jt_combo_prize_log": np.random.uniform(2.0, 6.0, n),
        })

        model = EVCorrectionModel()
        model.train(df, num_threads=1)

        # モデルが学習されていることを確認
        assert hasattr(model, "p_correction_model")
        assert hasattr(model, "e_correction_model")
        assert model.p_correction_model is not None
        assert model.e_correction_model is not None

    def test_place_ev_train_sorts_by_race_date_before_split(self):
        """PlaceEVCorrectionModel.train() も race_date でソートしてから分割すること"""
        from models.ev_correction_model import PlaceEVCorrectionModel

        np.random.seed(42)
        n = 100
        # 時系列順でないデータ（race_date がランダム）
        dates = pd.date_range("2020-01-01", periods=n).to_numpy()
        np.random.shuffle(dates)

        df = pd.DataFrame({
            "race_id": [f"R{i:04d}" for i in range(n)],
            "umaban": np.random.randint(1, 10, n),
            "race_date": dates,
            "kakuteijyuni": np.random.randint(1, 9, n),
            "p_place_pred": np.random.uniform(0.1, 0.7, n),
            "e_return_place_pred": np.random.uniform(1.1, 10, n),
            "p_ability_place": np.random.uniform(0.1, 0.7, n),
            "fukuoddslow": np.random.uniform(1.1, 50, n),
            "ev_place": np.random.uniform(0.5, 10, n),
            "signed_log_error_win": np.random.normal(0, 0.3, n),
            "abs_log_error_win": np.abs(np.random.normal(0, 0.3, n)),
            "market_entropy": np.random.uniform(2.0, 3.5, n),
            "popularity_rank": np.random.randint(1, 10, n),
            "surface": np.random.choice(["turf", "dirt"], n),
            "distance_bin": np.random.choice([1, 2, 3, 4], n),
            "track_condition_code": np.random.randint(1, 4, n),
            "field_size": np.random.randint(8, 16, n),
            "jockey_wr_overall": np.random.uniform(0.05, 0.20, n),
            "jockey_wr_distance": np.random.uniform(0.03, 0.18, n),
            "jockey_wr_venue": np.random.uniform(0.04, 0.19, n),
            "jockey_prize_log": np.random.uniform(8.0, 12.0, n),
            "trainer_wr_overall": np.random.uniform(0.05, 0.20, n),
            "trainer_wr_distance": np.random.uniform(0.03, 0.18, n),
            "trainer_wr_venue": np.random.uniform(0.04, 0.19, n),
            "trainer_prize_log": np.random.uniform(7.0, 11.5, n),
            "jt_combo_wr": np.random.uniform(0.05, 0.20, n),
            "jt_combo_place_rate": np.random.uniform(0.10, 0.35, n),
            "jt_combo_starts": np.random.uniform(1, 30, n),
            "jt_combo_prize_log": np.random.uniform(2.0, 6.0, n),
            "implied_prob_hhi": np.random.uniform(0.05, 0.15, n),
        })

        model = PlaceEVCorrectionModel()
        model.train(df, num_threads=1)

        # モデルが学習されていることを確認
        assert hasattr(model, "p_correction_model")
        assert hasattr(model, "e_correction_model")
        assert model.p_correction_model is not None
        assert model.e_correction_model is not None
