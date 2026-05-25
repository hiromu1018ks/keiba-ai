"""Isotonic EV Calibration + Odds Band Scaling テストスイート (EVC-03)

EVCorrectionModel.correct_ev() の Isotonic キャリブレーション、
オッズバンド別スケーリング、OOF EV生成、パイプライン統合の品質を検証する。
全テスト DB不要 (mock ベース)。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from models.ev_correction_model import EVCorrectionModel

# ── 共通 fixture ──────────────────────────────────────────────


@pytest.fixture
def ev_corrected_df() -> pd.DataFrame:
    """correct_ev()のPxE補正後の出力をシミュレートするDataFrame。
    ev_win_correctedが含まれている状態。"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 8,
            "kakuteijyuni": [1, 2, 3, 4, 5, 6, 7, 8],
            "confirmed_odds": [4.0, 6.0, 9.0, 16.0, 28.0, 45.0, 90.0, 160.0],
            "odds": [4.0, 6.0, 9.0, 16.0, 28.0, 45.0, 90.0, 160.0],
            "p_win_pred": [0.28, 0.22, 0.18, 0.12, 0.08, 0.06, 0.04, 0.02],
            "e_return_win_pred": [4.0, 6.0, 9.0, 16.0, 28.0, 45.0, 90.0, 160.0],
            "ev_win": [1.12, 1.32, 1.62, 1.92, 2.24, 2.70, 3.60, 3.20],
            "ev_win_corrected": [1.05, 1.20, 1.40, 1.60, 1.80, 2.20, 2.80, 2.50],
            "p_win_corrected": [0.26, 0.20, 0.16, 0.10, 0.07, 0.05, 0.03, 0.02],
            "e_return_win_corrected": [4.04, 6.0, 8.75, 16.0, 25.7, 44.0, 93.3, 125.0],
            "signed_log_error_win": [0.1, -0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.3],
            "abs_log_error_win": [0.1, 0.1, 0.2, 0.3, 0.0, 0.5, 0.2, 0.3],
            "market_entropy": [2.5] * 8,
            "popularity_rank": [1, 2, 3, 4, 5, 6, 7, 8],
            "implied_prob_hhi": [0.08] * 8,
            "surface": ["turf"] * 8,
            "distance_bin": ["mile"] * 8,
            "track_condition_code": [1] * 8,
            "field_size": [8] * 8,
            "jockey_wr_overall": [0.12] * 8,
            "jockey_wr_distance": [0.10] * 8,
            "jockey_wr_venue": [0.11] * 8,
            "jockey_prize_log": [11.0] * 8,
            "trainer_wr_overall": [0.14] * 8,
            "trainer_wr_distance": [0.12] * 8,
            "trainer_wr_venue": [0.13] * 8,
            "trainer_prize_log": [10.5] * 8,
            "jt_combo_wr": [0.15] * 8,
            "jt_combo_place_rate": [0.25] * 8,
            "jt_combo_starts": [5.0] * 8,
            "jt_combo_prize_log": [4.0] * 8,
        }
    )


@pytest.fixture
def mock_isotonic() -> IsotonicRegression:
    """IsotonicRegression mock: 高EVを押し下げるキャリブレーションをシミュレート"""
    iso = IsotonicRegression(y_min=0, out_of_bounds="clip")
    # 学習: EV過大評価パターン (高EVほどactualが低い)
    ev_train = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0])
    actual_train = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15])
    iso.fit(ev_train, actual_train)
    return iso


@pytest.fixture
def band_scales() -> dict[str, float]:
    """オッズバンド別スケーリング係数 (高オッズ帯を押し下げるパターン)"""
    return {
        "1.0-3.0": 1.0,
        "3.0-10.0": 0.95,
        "10.0-30.0": 0.85,
        "30.0+": 0.70,
    }


def _setup_mock_boosters(model: EVCorrectionModel, n: int) -> None:
    """テスト用にmock boosterを設定して_trainedフラグを立てる"""
    model.p_correction_model = MagicMock()
    model.p_correction_model.best_iteration = 100
    model.p_correction_model.predict.return_value = np.zeros(n)
    model.e_correction_model = MagicMock()
    model.e_correction_model.best_iteration = 80
    model.e_correction_model.predict.return_value = np.zeros(n)
    model._trained = True


# ── TestEVIsotonicCalibration ─────────────────────────────────


class TestEVIsotonicCalibration:
    """Isotonic適用のテスト"""

    def test_correct_ev_produces_calibrated_column(
        self,
        mock_isotonic: IsotonicRegression,
        ev_corrected_df: pd.DataFrame,
    ) -> None:
        """Isotonic設定時、ev_win_calibrated列が生成される"""
        model = EVCorrectionModel(ev_isotonic_calibrator=mock_isotonic)
        _setup_mock_boosters(model, len(ev_corrected_df))

        result = model.correct_ev(ev_corrected_df)
        assert "ev_win_calibrated" in result.columns

    def test_calibrated_reduces_overestimation(
        self,
        mock_isotonic: IsotonicRegression,
        ev_corrected_df: pd.DataFrame,
    ) -> None:
        """Isotonic適用後の高EV値が補正前より低いことを確認"""
        model = EVCorrectionModel(ev_isotonic_calibrator=mock_isotonic)
        _setup_mock_boosters(model, len(ev_corrected_df))

        result = model.correct_ev(ev_corrected_df)
        # 高EV値 (ev_win_corrected >= 1.5) では Isotonic が押し下げる
        high_ev = result[result["ev_win_corrected"] >= 1.5]
        if len(high_ev) > 0:
            assert (high_ev["ev_win_calibrated"] <= high_ev["ev_win_corrected"] + 1e-9).all(), (
                "Isotonic should reduce high EV values"
            )

    def test_calibrated_non_negative(
        self,
        mock_isotonic: IsotonicRegression,
        ev_corrected_df: pd.DataFrame,
    ) -> None:
        """ev_win_calibrated >= 0 (y_min=0制約)"""
        model = EVCorrectionModel(ev_isotonic_calibrator=mock_isotonic)
        _setup_mock_boosters(model, len(ev_corrected_df))

        result = model.correct_ev(ev_corrected_df)
        assert (result["ev_win_calibrated"] >= 0).all()

    def test_calibrated_preserves_order(
        self,
        mock_isotonic: IsotonicRegression,
        ev_corrected_df: pd.DataFrame,
    ) -> None:
        """ev_win_correctedの順序がev_win_calibratedでも維持される (単調増加)"""
        model = EVCorrectionModel(ev_isotonic_calibrator=mock_isotonic)
        _setup_mock_boosters(model, len(ev_corrected_df))

        result = model.correct_ev(ev_corrected_df)
        # IsotonicRegression は単調変換
        corrected = result["ev_win_corrected"].values
        calibrated = result["ev_win_calibrated"].values
        # corrected をソートした順序と calibrated をソートした順序が一致する
        order_corrected = np.argsort(corrected)
        order_calibrated = np.argsort(calibrated)
        assert np.array_equal(order_corrected, order_calibrated), (
            "Isotonic should preserve order (monotonic)"
        )

    def test_correct_ev_no_isotonic_fallback(
        self,
        ev_corrected_df: pd.DataFrame,
    ) -> None:
        """Isotonic未設定時、ev_win_calibrated == ev_win_corrected"""
        model = EVCorrectionModel()  # no isotonic
        _setup_mock_boosters(model, len(ev_corrected_df))

        result = model.correct_ev(ev_corrected_df)
        assert "ev_win_calibrated" in result.columns
        assert np.allclose(
            result["ev_win_calibrated"].values,
            result["ev_win_corrected"].values,
            atol=1e-10,
        )

    def test_correct_ev_can_use_oof_probability_column(self) -> None:
        """OOF補正時はp_win_predではなく指定されたp_win_oofを基準にする"""
        model = EVCorrectionModel()
        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "p_win_pred": [0.9, 0.1],
                "p_win_oof": [0.2, 0.8],
                "e_return_win_pred": [4.0, 4.0],
            }
        )

        result = model.correct_ev(df, probability_col="p_win_oof")

        assert result["p_win_corrected"].tolist() == pytest.approx([0.2, 0.8])
        assert result["ev_win_corrected"].tolist() == pytest.approx([0.8, 3.2])


# ── TestOddsBandScaling ───────────────────────────────────────


class TestOddsBandScaling:
    """オッズバンド別スケーリングのテスト"""

    def test_band_scaling_applied_to_calibrated(
        self,
        mock_isotonic: IsotonicRegression,
        band_scales: dict[str, float],
        ev_corrected_df: pd.DataFrame,
    ) -> None:
        """band_scales設定時、高オッズ帯のev_win_calibratedが縮小される"""
        model = EVCorrectionModel(
            ev_isotonic_calibrator=mock_isotonic,
            ev_odds_band_scales=band_scales,
        )
        _setup_mock_boosters(model, len(ev_corrected_df))

        # Isotonicのみのモデル (band_scalesなし) との比較
        model_iso_only = EVCorrectionModel(ev_isotonic_calibrator=mock_isotonic)
        model_iso_only.p_correction_model = model.p_correction_model
        model_iso_only.e_correction_model = model.e_correction_model
        model_iso_only._trained = True

        result = model.correct_ev(ev_corrected_df)
        result_iso_only = model_iso_only.correct_ev(ev_corrected_df.copy())

        # 高オッズ帯 (30.0+) のスケール=0.70 → 縮小されているはず
        high_odds_mask = result["confirmed_odds"] >= 30.0
        if high_odds_mask.any():
            assert (
                result.loc[high_odds_mask, "ev_win_calibrated"]
                <= result_iso_only.loc[high_odds_mask, "ev_win_calibrated"] + 1e-9
            ).all(), "Band scaling should shrink high-odds EV"

    def test_band_scaling_no_scales(
        self,
        mock_isotonic: IsotonicRegression,
        ev_corrected_df: pd.DataFrame,
    ) -> None:
        """band_scales未設定時、ev_win_calibratedがIsotonic結果と同じ"""
        model = EVCorrectionModel(ev_isotonic_calibrator=mock_isotonic)
        _setup_mock_boosters(model, len(ev_corrected_df))

        result = model.correct_ev(ev_corrected_df)
        # band_scales が None → Isotonic 結果がそのまま ev_win_calibrated に入る
        iso_only = mock_isotonic.transform(result["ev_win_corrected"].values.astype(float))
        assert np.allclose(result["ev_win_calibrated"].values, iso_only, atol=1e-9)

    def test_band_scaling_per_band(
        self,
        mock_isotonic: IsotonicRegression,
        ev_corrected_df: pd.DataFrame,
    ) -> None:
        """各オッズバンドのスケーリングが正しく適用される"""
        band_scales = {
            "1.0-3.0": 1.0,
            "3.0-10.0": 0.90,
            "10.0-30.0": 0.80,
            "30.0+": 0.60,
        }
        model = EVCorrectionModel(
            ev_isotonic_calibrator=mock_isotonic,
            ev_odds_band_scales=band_scales,
        )
        _setup_mock_boosters(model, len(ev_corrected_df))

        result = model.correct_ev(ev_corrected_df)

        # 各バンドのスケーリングを検証
        iso_values = mock_isotonic.transform(result["ev_win_corrected"].values.astype(float))
        odds = result["confirmed_odds"].values
        expected = iso_values.copy()

        # OddsBandFilter.BANDS に基づくスケーリング
        from betting.odds_band_filter import OddsBandFilter

        for (lo, hi), band_name in zip(OddsBandFilter.BANDS, OddsBandFilter.BAND_NAMES):
            scale = band_scales.get(band_name, 1.0)
            mask = (odds >= lo) & (odds < hi)
            expected[mask] *= scale

        assert np.allclose(result["ev_win_calibrated"].values, expected, atol=1e-9)

    def test_band_scaling_missing_odds_column(
        self,
        mock_isotonic: IsotonicRegression,
        band_scales: dict[str, float],
        ev_corrected_df: pd.DataFrame,
    ) -> None:
        """odds列がない場合、スケーリングがスキップされる"""
        # confirmed_odds も odds も削除
        df_no_odds = ev_corrected_df.drop(columns=["confirmed_odds", "odds"], errors="ignore")

        model = EVCorrectionModel(
            ev_isotonic_calibrator=mock_isotonic,
            ev_odds_band_scales=band_scales,
        )
        _setup_mock_boosters(model, len(df_no_odds))

        result = model.correct_ev(df_no_odds)
        # odds列なし → Isotonic結果がそのまま入る (band scaling スキップ)
        iso_values = mock_isotonic.transform(result["ev_win_corrected"].values.astype(float))
        assert np.allclose(result["ev_win_calibrated"].values, iso_values, atol=1e-9)


# ── TestOOFEVGeneration ───────────────────────────────────────


class TestOOFEVGeneration:
    """OOF EV生成のテスト — mockベース"""

    def _make_oof_df(self, n: int = 100) -> pd.DataFrame:
        """OOF生成テスト用DataFrame"""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "race_id": [f"R{i // 10:04d}" for i in range(n)],
                "race_date": pd.date_range("2020-01-01", periods=n, freq="D"),
                "kakuteijyuni": np.random.randint(1, 16, n),
                "odds": np.random.uniform(1.1, 100, n),
                "confirmed_odds": np.random.uniform(1.1, 100, n),
                "p_win_pred": np.random.uniform(0.01, 0.5, n),
                "e_return_win_pred": np.random.uniform(1.1, 100, n),
                "ev_win": np.random.uniform(0.5, 10, n),
                "signed_log_error_win": np.random.normal(0, 0.3, n),
                "abs_log_error_win": np.abs(np.random.normal(0, 0.3, n)),
                "market_entropy": np.random.uniform(2.0, 3.5, n),
                "popularity_rank": np.random.randint(1, 16, n),
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
            }
        )

    def test_generate_ev_oof_returns_three_arrays(self) -> None:
        """generate_ev_oof_predictions()が3つのndarrayを返す"""
        from pipelines.training_pipeline import TrainingPipelineV5

        df = self._make_oof_df(50)
        with (
            patch("pipelines.training_pipeline.WinTwoStageModel") as mock_win_cls,
            patch("pipelines.training_pipeline.EVCorrectionModel") as mock_ev_cls,
        ):
            # WinTwoStageModel mock
            mock_win = MagicMock()
            mock_win.predict_ev.side_effect = lambda d: d.assign(
                ev_win_corrected=np.random.uniform(0.5, 3.0, len(d)),
            )
            mock_win_cls.return_value = mock_win

            # EVCorrectionModel mock
            mock_ev = MagicMock()
            mock_ev.correct_ev.side_effect = lambda d: d.assign(
                ev_win_corrected=np.random.uniform(0.5, 3.0, len(d)),
            )
            mock_ev_cls.return_value = mock_ev

            result = TrainingPipelineV5.generate_ev_oof_predictions(
                df,
                n_splits=3,
                num_threads=1,
            )
            assert len(result) == 3
            oof_ev, oof_actual, oof_odds = result
            assert isinstance(oof_ev, np.ndarray)
            assert isinstance(oof_actual, np.ndarray)
            assert isinstance(oof_odds, np.ndarray)

    def test_generate_ev_oof_uses_walk_forward_split(self) -> None:
        """TimeSeriesSplitで未来データをtrainに混ぜないこと"""
        from pipelines.training_pipeline import TrainingPipelineV5

        df = self._make_oof_df(50)
        with (
            patch("pipelines.training_pipeline.WinTwoStageModel") as mock_win_cls,
            patch("pipelines.training_pipeline.EVCorrectionModel") as mock_ev_cls,
        ):
            mock_win = MagicMock()
            mock_win.predict_ev.side_effect = lambda d: d.assign(
                ev_win_corrected=np.random.uniform(0.5, 3.0, len(d)),
            )
            mock_win_cls.return_value = mock_win
            mock_ev = MagicMock()
            mock_ev.correct_ev.side_effect = lambda d: d.assign(
                ev_win_corrected=np.random.uniform(0.5, 3.0, len(d)),
            )
            mock_ev_cls.return_value = mock_ev

            from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit

            with patch(
                "pipelines.training_pipeline.TimeSeriesSplit",
                wraps=SklearnTimeSeriesSplit,
            ) as mock_split_cls:
                TrainingPipelineV5.generate_ev_oof_predictions(
                    df,
                    n_splits=3,
                    num_threads=1,
                )
                mock_split_cls.assert_called_once()

    def test_generate_ev_oof_sorts_by_race_date(self) -> None:
        """入力dfがrace_dateでソートされること"""
        from pipelines.training_pipeline import TrainingPipelineV5

        # ランダム順のrace_date
        df = self._make_oof_df(50)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        assert not df["race_date"].is_monotonic_increasing, "Precondition: data is unsorted"

        with (
            patch("pipelines.training_pipeline.WinTwoStageModel") as mock_win_cls,
            patch("pipelines.training_pipeline.EVCorrectionModel") as mock_ev_cls,
        ):
            mock_win = MagicMock()
            mock_win.predict_ev.side_effect = lambda d: d.assign(
                ev_win_corrected=np.random.uniform(0.5, 3.0, len(d)),
            )
            mock_win_cls.return_value = mock_win
            mock_ev = MagicMock()
            mock_ev.correct_ev.side_effect = lambda d: d.assign(
                ev_win_corrected=np.random.uniform(0.5, 3.0, len(d)),
            )
            mock_ev_cls.return_value = mock_ev

            # train_hit_modelに渡されたDataFrameをキャプチャ
            captured_dfs: list[pd.DataFrame] = []
            mock_win.train_hit_model.side_effect = lambda d, **_: captured_dfs.append(d)

            TrainingPipelineV5.generate_ev_oof_predictions(
                df,
                n_splits=3,
                num_threads=1,
            )
            # KFold.split に渡された DataFrame が race_date ソート済みであることを確認
            assert len(captured_dfs) > 0, "train_hit_model should have been called"
            for train_df in captured_dfs:
                assert train_df["race_date"].is_monotonic_increasing, (
                    "Training data must be sorted by race_date"
                )

    def test_generate_ev_oof_returns_only_valid_walk_forward_rows(self) -> None:
        """初期train期間を除き、検証foldの有効OOFだけを返す"""
        from pipelines.training_pipeline import TrainingPipelineV5

        n = 50
        df = self._make_oof_df(n)
        with (
            patch("pipelines.training_pipeline.WinTwoStageModel") as mock_win_cls,
            patch("pipelines.training_pipeline.EVCorrectionModel") as mock_ev_cls,
        ):
            mock_win = MagicMock()
            mock_win.predict_ev.side_effect = lambda d: d.assign(
                ev_win_corrected=np.random.uniform(0.5, 3.0, len(d)),
            )
            mock_win_cls.return_value = mock_win
            mock_ev = MagicMock()
            mock_ev.correct_ev.side_effect = lambda d: d.assign(
                ev_win_corrected=np.random.uniform(0.5, 3.0, len(d)),
            )
            mock_ev_cls.return_value = mock_ev

            oof_ev, oof_actual, oof_odds = TrainingPipelineV5.generate_ev_oof_predictions(
                df,
                n_splits=5,
                num_threads=1,
            )
            assert np.isfinite(oof_ev).all(), "All OOF EV values must be finite"
            assert np.isfinite(oof_actual).all(), "All OOF actual values must be finite"
            assert np.isfinite(oof_odds).all(), "All OOF odds values must be finite"
            assert 0 < len(oof_ev) < n


# ── TestEVCorrectionIntegration ───────────────────────────────


class TestEVCorrectionIntegration:
    """統合テスト"""

    def test_full_pipeline_isotonic_and_band_scaling(
        self,
        mock_isotonic: IsotonicRegression,
        band_scales: dict[str, float],
        ev_corrected_df: pd.DataFrame,
    ) -> None:
        """Isotonic + band scaling のフルパイプラインが正しく動作"""
        model = EVCorrectionModel(
            ev_isotonic_calibrator=mock_isotonic,
            ev_odds_band_scales=band_scales,
        )
        _setup_mock_boosters(model, len(ev_corrected_df))

        result = model.correct_ev(ev_corrected_df)

        assert "ev_win_calibrated" in result.columns
        assert "ev_win_corrected" in result.columns
        assert (result["ev_win_calibrated"] >= 0).all()

        # 高オッズ帯 (30.0+) は band_scale=0.70 で縮小
        high_odds = result[result["confirmed_odds"] >= 30.0]
        if len(high_odds) > 0:
            # Isotonic後の値より小さいことを確認 (0.70スケール)
            iso_only = mock_isotonic.transform(high_odds["ev_win_corrected"].values)
            assert (high_odds["ev_win_calibrated"].values <= iso_only + 1e-9).all()

    def test_ev_corrected_column_unchanged(
        self,
        mock_isotonic: IsotonicRegression,
        band_scales: dict[str, float],
        ev_corrected_df: pd.DataFrame,
    ) -> None:
        """ev_win_corrected列はIsotonic/band scalingの有無に関わらず同じ値"""
        # モック P/E booster (同じ補正値を返す)
        p_return = np.array([0.01, -0.01, 0.02, -0.02, 0.0, 0.03, -0.03, 0.01])
        e_return = np.zeros(8)

        # Isotonic + band scaling あり
        model_full = EVCorrectionModel(
            ev_isotonic_calibrator=mock_isotonic,
            ev_odds_band_scales=band_scales,
        )
        model_full.p_correction_model = MagicMock()
        model_full.p_correction_model.best_iteration = 100
        model_full.p_correction_model.predict.return_value = p_return
        model_full.e_correction_model = MagicMock()
        model_full.e_correction_model.best_iteration = 80
        model_full.e_correction_model.predict.return_value = e_return
        model_full._trained = True

        # Isotonic + band scaling なし
        model_plain = EVCorrectionModel()
        model_plain.p_correction_model = MagicMock()
        model_plain.p_correction_model.best_iteration = 100
        model_plain.p_correction_model.predict.return_value = p_return
        model_plain.e_correction_model = MagicMock()
        model_plain.e_correction_model.best_iteration = 80
        model_plain.e_correction_model.predict.return_value = e_return
        model_plain._trained = True

        result_full = model_full.correct_ev(ev_corrected_df.copy())
        result_plain = model_plain.correct_ev(ev_corrected_df.copy())

        # ev_win_corrected は両モデルで同じ値 (Isotonic/band scaling に依存しない)
        assert np.allclose(
            result_full["ev_win_corrected"].values,
            result_plain["ev_win_corrected"].values,
            atol=1e-10,
        )

    def test_model_init_accepts_isotonic(
        self,
        mock_isotonic: IsotonicRegression,
    ) -> None:
        """EVCorrectionModel(ev_isotonic_calibrator=iso)が正常に初期化"""
        model = EVCorrectionModel(ev_isotonic_calibrator=mock_isotonic)
        assert model.ev_isotonic_calibrator is mock_isotonic
        assert model.ev_odds_band_scales is None

    def test_model_init_accepts_band_scales(
        self,
        band_scales: dict[str, float],
    ) -> None:
        """EVCorrectionModel(ev_odds_band_scales=scales)が正常に初期化"""
        model = EVCorrectionModel(ev_odds_band_scales=band_scales)
        assert model.ev_isotonic_calibrator is None
        assert model.ev_odds_band_scales is band_scales

    def test_submodelset_new_fields_default_none(self) -> None:
        """SubmodelSetの新フィールドのデフォルト値がNone"""
        # dataclass field の default を確認
        import dataclasses

        from domain.models import SubmodelSet

        fields = {f.name: f for f in dataclasses.fields(SubmodelSet)}
        assert "ev_isotonic_calibrator" in fields
        assert fields["ev_isotonic_calibrator"].default is None
        assert "ev_odds_band_scales" in fields
        assert fields["ev_odds_band_scales"].default is None
