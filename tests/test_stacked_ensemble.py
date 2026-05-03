import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, call
import lightgbm as lgb
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


class TestOptunaTuning:
    """Task 1: Optuna HP最適化 + Early Stopping + 特徴量サブセット"""

    def test_optuna_tuning_produces_different_params(self):
        """StackedEnsemble.train()後、3モデルのlearning_rateがそれぞれ異なる"""
        X, y = _make_binary_data(n=300, seed=10)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3)
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        lgb_lr = ensemble.best_params["lgbm"]["lgb_lr"]
        xgb_lr = ensemble.best_params["xgb"]["xgb_lr"]
        cat_lr = ensemble.best_params["cat"]["cat_lr"]
        assert lgb_lr != xgb_lr or lgb_lr != cat_lr or xgb_lr != cat_lr

    @patch("models.stacked_ensemble.lgb.train")
    def test_early_stopping_in_fold(self, mock_lgb_train):
        """_train_lgbm_foldがlgb.early_stopping(stopping_rounds=100)を使用"""
        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.5, 0.5])
        mock_lgb_train.return_value = mock_booster

        ensemble = StackedEnsemble(cat_cols=[])
        X_tr = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8]})
        y_tr = pd.Series([0, 1, 0, 1])
        X_va = pd.DataFrame({"f1": [9, 10], "f2": [11, 12]})

        params = {"lgb_num_leaves": 31, "lgb_lr": 0.03, "lgb_feat_frac": 0.7}
        ensemble._train_lgbm_fold(X_tr, y_tr, X_va, 1, params)

        # lgb.trainが呼ばれた際、callbacksにearly_stoppingが含まれることを確認
        call_kwargs = mock_lgb_train.call_args
        callbacks = call_kwargs[1].get("callbacks") if call_kwargs[1] else call_kwargs.kwargs.get("callbacks")
        assert callbacks is not None
        assert any("early_stopping" in str(type(cb).__name__).lower() for cb in callbacks)

    @patch("models.stacked_ensemble.lgb.train")
    def test_early_stopping_in_full(self, mock_lgb_train):
        """_train_lgbm_fullがlgb.early_stopping(stopping_rounds=100)を使用"""
        mock_booster = MagicMock()
        mock_lgb_train.return_value = mock_booster

        ensemble = StackedEnsemble(cat_cols=[])
        X = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8]})
        y = pd.Series([0, 1, 0, 1])

        params = {"lgb_num_leaves": 31, "lgb_lr": 0.03, "lgb_feat_frac": 0.7}
        ensemble._train_lgbm_full(X, y, 1, params)

        call_kwargs = mock_lgb_train.call_args
        callbacks = call_kwargs.kwargs.get("callbacks")
        assert callbacks is not None
        assert any("early_stopping" in str(type(cb).__name__).lower() for cb in callbacks)

    def test_80_20_split_in_fold(self):
        """_train_lgbm_foldがX_trを80/20に分割してvalidation確保"""
        ensemble = StackedEnsemble(cat_cols=[])
        # params=Noneで後方互換 — テストデータが小さくても動作確認
        X_tr = pd.DataFrame({"f1": range(20), "f2": range(20, 40)})
        y_tr = pd.Series([0] * 10 + [1] * 10)
        X_va = pd.DataFrame({"f1": [1, 2], "f2": [21, 22]})

        # params=Noneで呼び出し(後方互換) — 80/20分割が内部で行われる
        result = ensemble._train_lgbm_fold(X_tr, y_tr, X_va, 1, params=None)
        # 正常にpredictionが返ることを確認(80/20分割でデータが減っても動作)
        assert result is not None
        assert len(result) == len(X_va)

    def test_feature_fraction_in_lgbm_params(self):
        """best_params["lgbm"]にfeature_fractionキーが含まれ、0.3-0.9の範囲"""
        X, y = _make_binary_data(n=300, seed=11)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3)
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        feat_frac = ensemble.best_params["lgbm"]["lgb_feat_frac"]
        assert 0.3 <= feat_frac <= 0.9

    def test_feature_fraction_in_xgb_params(self):
        """best_params["xgb"]にcolsample_bytreeキーが含まれ、0.3-0.9の範囲"""
        X, y = _make_binary_data(n=300, seed=12)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3)
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        col_sample = ensemble.best_params["xgb"]["xgb_col_sample"]
        assert 0.3 <= col_sample <= 0.9

    def test_feature_fraction_in_cat_params(self):
        """best_params["cat"]にrsmキーが含まれ、0.3-0.9の範囲"""
        X, y = _make_binary_data(n=300, seed=13)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3)
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        rsm = ensemble.best_params["cat"]["cat_rsm"]
        assert 0.3 <= rsm <= 0.9

    def test_exploration_space_separation(self):
        """各モデルが異なる木複雑度空間を探索"""
        ensemble = StackedEnsemble(cat_cols=[])
        import optuna
        # LightGBM: num_leaves max=63
        study_lgb = optuna.create_study(direction="maximize")
        study_lgb.optimize(lambda t: 0.0, n_trials=1)
        lgb_params = ensemble._suggest_lgbm_params(study_lgb.best_trial)
        assert lgb_params["lgb_num_leaves"] <= 63

        # XGBoost: max_depth max=8 → leaf換算 2^8=256
        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(lambda t: 0.0, n_trials=1)
        xgb_params = ensemble._suggest_xgb_params(study_xgb.best_trial)
        assert xgb_params["xgb_max_depth"] <= 8

        # CatBoost: depth max=10 → leaf換算 2^10=1024
        study_cat = optuna.create_study(direction="maximize")
        study_cat.optimize(lambda t: 0.0, n_trials=1)
        cat_params = ensemble._suggest_cat_params(study_cat.best_trial)
        assert cat_params["cat_depth"] <= 10

        # 複雑度順序: LGB leaves(63) < XGB leaf換算(256) < CAT leaf換算(1024)
        lgb_leaves = lgb_params["lgb_num_leaves"]
        xgb_leaves_equiv = 2 ** xgb_params["xgb_max_depth"]
        cat_leaves_equiv = 2 ** cat_params["cat_depth"]
        assert lgb_leaves < xgb_leaves_equiv
        assert xgb_leaves_equiv <= cat_leaves_equiv

    @patch("models.stacked_ensemble.xgb")
    def test_xgb_early_stopping_in_fold(self, mock_xgb):
        """_train_xgb_foldがearly_stopping_rounds=100をxgb.trainに渡す"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.5])
        mock_xgb.train.return_value = mock_model
        mock_xgb.DMatrix.return_value = MagicMock()

        ensemble = StackedEnsemble(cat_cols=[])
        X_tr = pd.DataFrame({"f1": range(20), "f2": range(20, 40)})
        y_tr = pd.Series([0] * 10 + [1] * 10)
        X_va = pd.DataFrame({"f1": [1, 2], "f2": [21, 22]})

        params = {"xgb_max_depth": 6, "xgb_lr": 0.03, "xgb_col_sample": 0.7}
        ensemble._train_xgb_fold(X_tr, y_tr, X_va, 1, params)

        call_kwargs = mock_xgb.train.call_args.kwargs
        assert call_kwargs.get("early_stopping_rounds") == 100

    @patch("models.stacked_ensemble.CatBoostClassifier")
    def test_cat_early_stopping_in_fold(self, mock_cat_cls):
        """_train_cat_foldがearly_stopping_rounds=100をCatBoostClassifierに渡す"""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.4, 0.6]])
        mock_cat_cls.return_value = mock_model

        ensemble = StackedEnsemble(cat_cols=[])
        X_tr = pd.DataFrame({"f1": range(20), "f2": range(20, 40)})
        y_tr = pd.Series([0] * 10 + [1] * 10)
        X_va = pd.DataFrame({"f1": [1, 2], "f2": [21, 22]})

        params = {"cat_depth": 6, "cat_lr": 0.01, "cat_rsm": 0.7}
        ensemble._train_cat_fold(X_tr, y_tr, X_va, 1, params)

        call_kwargs = mock_cat_cls.call_args.kwargs
        assert call_kwargs.get("early_stopping_rounds") == 100
