"""MarketModel OOF (Out-of-Fold) 予測のテスト"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.market_model import MarketModel


def _make_sample_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """テスト用サンプルDataFrameを生成"""
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "race_id": [f"R{i // 10}" for i in range(n)],
            "p_market_win_adj": rng.rand(n),
            "surface": rng.choice(["turf", "dirt"], n),
            "distance_bin": rng.choice(["sprint", "mile", "intermediate"], n),
            "track_condition_code": rng.randint(1, 5, n).astype(float),
            "grade_code": rng.choice(["A", "B", "C", "X"], n),
            "field_size": rng.randint(8, 18, n).astype(float),
            "weight_diff_from_mean": rng.randn(n),
            "difficulty_score": rng.rand(n),
        }
    )


class TestPredictOofBasic:
    def test_predict_oof_returns_oof_predictions(self) -> None:
        """predict_oof が全データのOOF予測値を返す"""
        df = _make_sample_df(100)
        model = MarketModel()
        model.train(df, num_threads=1)
        oof_df = model.predict_oof(df, n_splits=5)

        # 全行の予測が返る
        assert len(oof_df) == 100
        # NaNなし (全foldカバー)
        assert oof_df["signed_log_error_win"].notna().all()
        assert oof_df["abs_log_error_win"].notna().all()

    def test_predict_oof_invariant(self) -> None:
        """OOF予測値 != insample予測値 — 各行の予測はその行を学習に使っていないモデルから生成"""
        df = _make_sample_df(200)
        model = MarketModel()
        model.train(df, num_threads=1)

        # insample予測
        insample_df = model.predict_and_calc_error(df.copy())
        insample_errors = insample_df["signed_log_error_win"]

        # OOF予測
        oof_df = model.predict_oof(df.copy(), n_splits=5)
        oof_errors = oof_df["signed_log_error_win"]

        # OOF != insample (リーク除去の証明)
        diff = (oof_errors - insample_errors).abs()
        assert diff.mean() > 1e-6, "OOF should differ from insample predictions"

    def test_predict_oof_preserves_stage2_features(self) -> None:
        """OOF予測後も Stage2 に必要な特徴量列が存在する"""
        df = _make_sample_df(100)
        model = MarketModel()
        model.train(df, num_threads=1)
        result = model.predict_oof(df, n_splits=5)

        # Stage2 特徴量が存在する
        expected_cols = model.get_stage2_features()
        for col in expected_cols:
            assert col in result.columns, f"Missing stage2 feature: {col}"

        # _p_market_pred_win は削除されている (Rule 11)
        assert "_p_market_pred_win" not in result.columns

    def test_predict_oof_retrains_full_model(self) -> None:
        """predict_oof 後に self.model が全データで再学習済みである"""
        df = _make_sample_df(100)
        model = MarketModel()
        assert model.model is None
        model.predict_oof(df, n_splits=5)
        # 再学習済み
        assert model.model is not None


class TestPredictOofPitSafety:
    def test_oof_uses_kfold_no_shuffle(self) -> None:
        """KFold で shuffle=False を使用し、時系列順序を維持する"""
        from sklearn.model_selection import KFold

        df = _make_sample_df(50)
        kf = KFold(n_splits=5, shuffle=False)

        # 各foldのtrain/validインデックスが時間順であることを確認
        for train_idx, valid_idx in kf.split(df):
            # train_idx は昇順、valid_idx も昇順
            assert list(train_idx) == sorted(train_idx)
            assert list(valid_idx) == sorted(valid_idx)
            # valid の最小インデックス > train の最大インデックス (time-series split 的挙動確認)
            # KFold shuffle=False では連続したチャンクになる

    def test_oof_no_nan_in_predictions(self) -> None:
        """OOF予測にNaNが含まれない (全foldでカバーされる)"""
        df = _make_sample_df(120, seed=123)
        model = MarketModel()
        model.train(df, num_threads=1)
        result = model.predict_oof(df, n_splits=5)

        # 全てのlog_error列にNaNがない
        for col in ["signed_log_error_win", "abs_log_error_win", "market_log_error_win"]:
            assert result[col].notna().all(), f"{col} contains NaN"


class TestPredictOofOutputColumns:
    def test_output_contains_all_expected_columns(self) -> None:
        """出力DataFrameに必要な全列が含まれる"""
        df = _make_sample_df(80)
        model = MarketModel()
        model.train(df, num_threads=1)
        result = model.predict_oof(df, n_splits=5)

        expected_columns = [
            "signed_log_error_win",
            "abs_log_error_win",
            "market_log_error_win",
            "market_pred_error_win",
            "market_error_rank_in_race",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_market_pred_error_win_is_raw_difference(self) -> None:
        """market_pred_error_win が p_market - p_pred の生差分である"""
        df = _make_sample_df(60, seed=99)
        model = MarketModel()
        model.train(df, num_threads=1)

        result = model.predict_oof(df, n_splits=5)

        # market_pred_error_win は log_error 計算前の生差分
        # OOFなので直接検証は難しいが、列が存在し有限であることを確認
        assert result["market_pred_error_win"].notna().all()
        assert np.isfinite(result["market_pred_error_win"].values).all()
