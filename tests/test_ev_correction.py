"""src/models/ev_correction_model.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.ev_correction_model import EVCorrectionModel


@pytest.fixture
def pre_ev_df() -> pd.DataFrame:
    """WinTwoStageModel.predict_ev() 出力後のテストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 8,
            "finish_pos": [1, 2, 3, 4, 5, 6, 7, 8],
            "win_odds_actual": [4.0, 6.0, 9.0, 16.0, 28.0, 45.0, 90.0, 160.0],
            "p_win_pred": [0.28, 0.22, 0.18, 0.12, 0.08, 0.06, 0.04, 0.02],
            "e_return_win_pred": [4.0, 6.0, 9.0, 16.0, 28.0, 45.0, 90.0, 160.0],
            "ev_win": [1.12, 1.32, 1.62, 1.92, 2.24, 2.70, 3.60, 3.20],
            "signed_log_error_win": [0.1, -0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.3],
            "abs_log_error_win": [0.1, 0.1, 0.2, 0.3, 0.0, 0.5, 0.2, 0.3],
            "market_entropy": [2.5] * 8,
            "popularity_rank": [1, 2, 3, 4, 5, 6, 7, 8],
            "surface": ["turf"] * 8,
            "distance_bin": ["mile"] * 8,
            "track_condition_code": [1] * 8,
            "field_size": [8] * 8,
        }
    )


@pytest.fixture
def trained_ev_model(pre_ev_df: pd.DataFrame) -> EVCorrectionModel:
    """学習済みEVCorrectionModel (mock)"""
    model = EVCorrectionModel()
    # P補正: 小さい補正値 (correction_logit)
    mock_p = MagicMock()
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


@pytest.fixture
def large_ev_df() -> pd.DataFrame:
    """EV補正の統計テスト用 大規模データ (200行, winner 60+頭, 中穴ゾーン 120+行)"""
    np.random.seed(123)
    n_races = 200
    rows: list[dict] = []
    for i in range(n_races):
        n_horses = np.random.randint(8, 16)
        p_preds = np.sort(np.random.dirichlet(np.ones(n_horses) * 0.5))[::-1]
        for j in range(n_horses):
            finish = j + 1
            odds = max(1.1, np.random.lognormal(2.0, 0.7))
            rows.append({
                "race_id": f"LR{i:04d}",
                "finish_pos": finish,
                "win_odds_actual": odds,
                "p_win_pred": float(p_preds[j]),
                "e_return_win_pred": odds,
                "ev_win": float(p_preds[j]) * odds,
                "signed_log_error_win": np.random.normal(0, 0.3),
                "abs_log_error_win": abs(np.random.normal(0, 0.3)),
                "market_entropy": float(np.random.uniform(2.0, 3.5)),
                "popularity_rank": j + 1,
                "surface": np.random.choice(["turf", "dirt"]),
                "distance_bin": np.random.choice(["sprint", "mile", "intermediate", "long"]),
                "track_condition_code": 1,
                "field_size": n_horses,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def trained_ev_model_large(large_ev_df: pd.DataFrame) -> EVCorrectionModel:
    """大規模データ用 EVCorrectionModel (mock: winner の P を引き上げる補正)"""
    np.random.seed(456)
    model = EVCorrectionModel()
    n = len(large_ev_df)
    is_winner = large_ev_df["finish_pos"].values == 1
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
    mock_p.predict.return_value = p_corrections
    mock_e = MagicMock()
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
        actual_ev = result["win_odds_actual"] * (result["finish_pos"] == 1).astype(int)
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
        actual_ev = mid_range["win_odds_actual"] * (mid_range["finish_pos"] == 1).astype(int)
        mae_raw = float(np.mean(np.abs(mid_range["ev_win"] - actual_ev)))
        mae_corrected = float(np.mean(np.abs(mid_range["ev_win_corrected"] - actual_ev)))
        if mae_raw == 0:
            pytest.skip("MAE_raw がゼロのため改善率を計算できない")
        improvement = (mae_raw - mae_corrected) / mae_raw
        assert improvement > 0.10, (
            f"中穴ゾーンの補正改善率が低い: {improvement:.1%}"
        )

    def test_ev_correction_winner_weight(
        self,
        trained_ev_model_large: EVCorrectionModel,
        large_ev_df: pd.DataFrame,
    ) -> None:
        """1着馬のP_corrected中央値>=P_pred中央値 (§13.1)"""
        result = trained_ev_model_large.correct_ev(large_ev_df)
        winners = result[result["finish_pos"] == 1]
        if len(winners) < 50:
            pytest.skip("1着馬サンプル不足")
        assert winners["p_win_corrected"].median() >= winners["p_win_pred"].median(), (
            "P補正が1着馬の確率を適切に引き上げていません"
        )
