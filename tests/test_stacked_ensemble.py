import numpy as np
import pandas as pd
import pytest
from models.stacked_ensemble import StackedEnsemble


def _make_binary_data(n: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "f3": rng.rand(n),
    })
    y = pd.Series((rng.rand(n) > 0.8).astype(int))
    return X, y


class TestStackedEnsemble:
    def test_train_and_predict(self):
        """学習→予測で [0,1] の確率が返る"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[])
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        preds = ensemble.predict(X.iloc[split:])
        assert len(preds) == len(X) - split
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_base_models_trained(self):
        """3つのベースモデルが学習されている"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[])
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        assert ensemble.lgbm_model is not None
        assert ensemble.xgb_model is not None
        assert ensemble.cat_model is not None
        assert ensemble.meta_model is not None

    def test_different_from_single_lgbm(self):
        """アンサンブル予測が単一LGBMと異なる"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[])
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        preds_ens = ensemble.predict(X.iloc[split:])
        preds_lgbm = ensemble.lgbm_model.predict(X.iloc[split:])
        # アンサンブルとLGBM単体で予測が異なることを確認
        assert not np.allclose(preds_ens, preds_lgbm, atol=1e-6)

    def test_best_iteration_compatible(self):
        """lgb.Booster 互換の best_iteration 属性がある"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[])
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        assert hasattr(ensemble, "best_iteration")
        assert ensemble.best_iteration == 0  # アンサンブルでは使用しない
