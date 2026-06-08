import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from models.stacked_ensemble import StackedEnsemble


def _make_binary_data(n: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "f3": rng.rand(n),
        }
    )
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

    def test_encode_cats_categorical_fillna(self):
        """Categorical列の_encode_catsで未知カテゴリが-1になる (TypeError回帰)"""
        ensemble = StackedEnsemble(cat_cols=["color"])
        ensemble._cat_codes = {"color": {"red": 0, "blue": 1, "green": 2}}

        X = pd.DataFrame(
            {
                "color": pd.Categorical(["red", "blue", "green", "red", "yellow"]),
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        result = ensemble._encode_cats(X)
        assert result["color"].tolist() == [0.0, 1.0, 2.0, 0.0, -1.0]

    def test_encode_cats_categorical_with_nan(self):
        """Categorical列のNaN値が-1になる"""
        ensemble = StackedEnsemble(cat_cols=["color"])
        ensemble._cat_codes = {"color": {"red": 0, "blue": 1}}

        X = pd.DataFrame(
            {
                "color": pd.Categorical(["red", None, "blue", None]),
                "f1": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = ensemble._encode_cats(X)
        assert result["color"].tolist() == [0.0, -1.0, 1.0, -1.0]

    def test_encode_cats_returns_input_when_no_categorical_columns(self):
        """数値列のみの場合は不要なDataFrameコピーを作成しない"""
        ensemble = StackedEnsemble(cat_cols=[])
        X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})

        result = ensemble._encode_cats(X)

        assert result is X


class TestFeatureNameImportanceCompat:
    """feature_name() / feature_importance() の lgb.Booster 互換インターフェース"""

    def test_feature_name_returns_list(self):
        """train後、feature_name()が文字列リストを返す"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2)
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        names = ensemble.feature_name()
        assert isinstance(names, list)
        assert set(names) == {"f1", "f2", "f3"}

    def test_feature_name_before_train_returns_empty(self):
        """train前、feature_name()が空リストを返す"""
        ensemble = StackedEnsemble(cat_cols=[])
        assert ensemble.feature_name() == []

    def test_feature_importance_returns_ndarray(self):
        """train後、feature_importance()がndarrayを返す"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2)
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        imp = ensemble.feature_importance(importance_type="gain")
        assert isinstance(imp, np.ndarray)
        assert len(imp) == 3  # 3 features

    def test_feature_importance_before_train_returns_empty(self):
        """train前、feature_importance()が空配列を返す"""
        ensemble = StackedEnsemble(cat_cols=[])
        result = ensemble.feature_importance()
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_feature_importance_normalized_and_averaged(self):
        """feature_importance()は正規化平均のため全要素の合計が概ね1.0"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2)
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        imp = ensemble.feature_importance(importance_type="gain")
        # 3モデルの正規化平均なので合計は約1.0
        assert abs(imp.sum() - 1.0) < 0.01

    def test_extract_feature_ranking_with_stacked_ensemble(self):
        """extract_feature_ranking() が StackedEnsemble で動作する"""
        from models.walk_forward_cv import extract_feature_ranking

        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2)
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

        ranking, top_features = extract_feature_ranking(ensemble, top_n=3)
        assert isinstance(ranking, dict)
        assert isinstance(top_features, list)
        assert len(top_features) <= 3
        # top_featuresの要素が全て学習特徴量に含まれる
        for f in top_features:
            assert f in {"f1", "f2", "f3"}


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
        callbacks = (
            call_kwargs[1].get("callbacks")
            if call_kwargs[1]
            else call_kwargs.kwargs.get("callbacks")
        )
        assert callbacks is not None
        # early_stopping callbackのstopping_rounds属性を確認
        assert any(hasattr(cb, "stopping_rounds") and cb.stopping_rounds == 100 for cb in callbacks)

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
        assert any(hasattr(cb, "stopping_rounds") and cb.stopping_rounds == 100 for cb in callbacks)

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

        # LightGBM: objective内でsuggest結果をキャプチャ
        captured_lgb: dict = {}

        def obj_lgb(t):
            captured_lgb.update(ensemble._suggest_lgbm_params(t))
            return 0.0

        study_lgb = optuna.create_study(direction="maximize")
        study_lgb.optimize(obj_lgb, n_trials=1)
        assert captured_lgb["lgb_num_leaves"] <= 63

        # XGBoost
        captured_xgb: dict = {}

        def obj_xgb(t):
            captured_xgb.update(ensemble._suggest_xgb_params(t))
            return 0.0

        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(obj_xgb, n_trials=1)
        assert captured_xgb["xgb_max_depth"] <= 8

        # CatBoost
        captured_cat: dict = {}

        def obj_cat(t):
            captured_cat.update(ensemble._suggest_cat_params(t))
            return 0.0

        study_cat = optuna.create_study(direction="maximize")
        study_cat.optimize(obj_cat, n_trials=1)
        assert captured_cat["cat_depth"] <= 10

        # 複雑度順序(上限): LGB(63) < XGB(2^8=256) < CAT(2^10=1024)
        assert 63 < 2**8  # LGB max < XGB max
        assert 2**8 <= 2**10  # XGB max <= CAT max

    @patch("xgboost.train")
    def test_xgb_early_stopping_in_fold(self, mock_xgb_train):
        """_train_xgb_foldがearly_stopping_rounds=100をxgb.trainに渡す"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.5])
        mock_xgb_train.return_value = mock_model

        ensemble = StackedEnsemble(cat_cols=[])
        X_tr = pd.DataFrame({"f1": range(20), "f2": range(20, 40)})
        y_tr = pd.Series([0] * 10 + [1] * 10)
        X_va = pd.DataFrame({"f1": [1, 2], "f2": [21, 22]})

        params = {"xgb_max_depth": 6, "xgb_lr": 0.03, "xgb_col_sample": 0.7}
        ensemble._train_xgb_fold(X_tr, y_tr, X_va, 1, params)

        call_kwargs = mock_xgb_train.call_args.kwargs
        assert call_kwargs.get("early_stopping_rounds") == 100

    @patch("xgboost.DMatrix")
    @patch("xgboost.train")
    def test_eval_xgb_reuses_cached_dmatrix(self, mock_xgb_train, mock_dmatrix):
        """Optuna trial間でキャッシュ済みDMatrixを再利用する"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.2, 0.8])
        mock_xgb_train.return_value = mock_model
        cached_train = MagicMock()
        cached_valid = MagicMock()

        ensemble = StackedEnsemble(cat_cols=[])
        X_t = pd.DataFrame({"f1": [1.0, 2.0]})
        y_t = pd.Series([0, 1])
        X_v = pd.DataFrame({"f1": [3.0, 4.0]})
        y_v = pd.Series([0, 1])
        trial = MagicMock()

        score = ensemble._eval_xgb(
            trial,
            lambda _: {
                "xgb_max_depth": 6,
                "xgb_lr": 0.03,
                "xgb_col_sample": 0.7,
            },
            X_t,
            y_t,
            X_v,
            y_v,
            1,
            dtrain=cached_train,
            dvalid=cached_valid,
        )

        assert score == 0.99
        mock_dmatrix.assert_not_called()
        assert mock_xgb_train.call_args.args[1] is cached_train
        assert mock_model.predict.call_args.args[0] is cached_valid

    def test_race_group_split_does_not_split_race(self):
        """時系列分割位置が同一レースの途中に入らない。"""
        race_ids = pd.Series(["r1"] * 3 + ["r2"] * 2 + ["r3"] * 4 + ["r4"] * 2)

        split = StackedEnsemble.race_group_split_index(race_ids, ratio=0.5)

        assert split == 5
        assert race_ids.iloc[:split].nunique() == 2
        assert set(race_ids.iloc[:split]).isdisjoint(set(race_ids.iloc[split:]))

    def test_race_group_split_rejects_noncontiguous_races(self):
        """同一レースが離れた位置にある入力を黙って分割しない。"""
        race_ids = pd.Series(["r1", "r2", "r1", "r2"])

        with np.testing.assert_raises_regex(ValueError, "contiguous"):
            StackedEnsemble.race_group_split_index(race_ids)

    def test_optuna_validation_is_disjoint_from_oof_validation(self):
        """Optuna専用検証レースがスタッキングOOF検証に混入しない。"""
        race_ids = pd.Series(np.repeat([f"r{i:02d}" for i in range(20)], 3))
        groups = StackedEnsemble._normalize_groups(race_ids, len(race_ids))
        first_train_idx, _ = StackedEnsemble._expanding_group_splits(groups, n_folds=3)[0]
        tune_groups = groups.iloc[first_train_idx].reset_index(drop=True)
        tune_split = StackedEnsemble._group_boundary(tune_groups, 0.8)
        tune_valid_idx = first_train_idx[tune_split:]
        oof_valid_indices = np.concatenate(
            [
                valid_idx
                for _, valid_idx in StackedEnsemble._expanding_group_splits(
                    groups,
                    n_folds=3,
                )
            ]
        )

        assert set(tune_valid_idx).isdisjoint(set(oof_valid_indices))

    def test_predict_xgb_uses_best_iteration(self):
        """XGBoost予測はearly stoppingの最良反復までに限定する。"""
        model = MagicMock()
        model.best_iteration = 17
        model.predict.return_value = np.array([0.4, 0.6])
        data = MagicMock()

        result = StackedEnsemble._predict_xgb_best(model, data)

        assert result.tolist() == [0.4, 0.6]
        model.predict.assert_called_once_with(data, iteration_range=(0, 18))

    def test_probability_objective_handles_single_class(self):
        """単一クラスの短い検証期間でもOptuna目的値が有限になる。"""
        y = pd.Series([0, 0, 0])
        preds = np.array([0.1, 0.2, 0.3])

        score = StackedEnsemble._probability_objective(y, preds)

        assert np.isfinite(score)
        assert score < 0.5

    @patch("catboost.CatBoostClassifier")
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


class TestDiversityCheck:
    """Task 2: 多様性検証メソッド + テスト拡張"""

    def test_check_diversity_logs_pairwise_correlation(self, caplog):
        """_check_diversityがOOF予測の3ペア相関をINFOでログ出力"""
        ensemble = StackedEnsemble(cat_cols=[])
        # 異なる予測値 — 相関は低い
        oof_preds = np.array(
            [
                [0.1, 0.3, 0.7],
                [0.2, 0.5, 0.8],
                [0.3, 0.1, 0.2],
                [0.4, 0.9, 0.6],
                [0.5, 0.2, 0.4],
            ]
        )
        y = pd.Series([0, 1, 0, 1, 0])
        importances = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 1.0, 2.0]),
            np.array([2.0, 3.0, 1.0]),
        ]
        feature_names = ["f1", "f2", "f3"]

        with caplog.at_level(logging.INFO, logger="models.stacked_ensemble"):
            ensemble._check_diversity(oof_preds, y, importances, feature_names)

        corr_logs = [r for r in caplog.records if "OOF prediction correlation" in r.message]
        assert len(corr_logs) == 3  # LGB-XGB, LGB-CAT, XGB-CAT

    def test_check_diversity_warns_high_correlation(self, caplog):
        """相関>=0.95の場合、WARNINGログが出力される"""
        ensemble = StackedEnsemble(cat_cols=[])
        # 全列が同じ → 相関1.0
        col = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        oof_preds = np.column_stack([col, col, col])
        y = pd.Series([0, 1, 0, 1, 0])
        importances = [np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0])]
        feature_names = ["f1", "f2"]

        with caplog.at_level(logging.WARNING, logger="models.stacked_ensemble"):
            ensemble._check_diversity(oof_preds, y, importances, feature_names)

        warn_logs = [r for r in caplog.records if "High prediction correlation" in r.message]
        assert len(warn_logs) == 3  # 全ペアが高相関

    def test_check_diversity_logs_importance_correlation(self, caplog):
        """_check_diversityがfeature importanceのSpearman順位相関をログ出力"""
        ensemble = StackedEnsemble(cat_cols=[])
        oof_preds = np.array(
            [
                [0.1, 0.3, 0.7],
                [0.2, 0.5, 0.8],
                [0.3, 0.1, 0.2],
            ]
        )
        y = pd.Series([0, 1, 0])
        importances = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 1.0, 2.0]),
            np.array([2.0, 3.0, 1.0]),
        ]
        feature_names = ["f1", "f2", "f3"]

        with caplog.at_level(logging.INFO, logger="models.stacked_ensemble"):
            ensemble._check_diversity(oof_preds, y, importances, feature_names)

        imp_logs = [r for r in caplog.records if "Feature importance rank correlation" in r.message]
        assert len(imp_logs) == 3  # LGB-XGB, LGB-CAT, XGB-CAT

    def test_check_diversity_warns_high_importance_correlation(self, caplog):
        """importance順位相関>0.8の場合、WARNINGログが出力される"""
        ensemble = StackedEnsemble(cat_cols=[])
        oof_preds = np.array(
            [
                [0.1, 0.3, 0.7],
                [0.2, 0.5, 0.8],
                [0.3, 0.1, 0.2],
            ]
        )
        y = pd.Series([0, 1, 0])
        # 同一importance → Spearman相関1.0
        imp = np.array([1.0, 2.0, 3.0])
        importances = [imp.copy(), imp.copy(), imp.copy()]
        feature_names = ["f1", "f2", "f3"]

        with caplog.at_level(logging.WARNING, logger="models.stacked_ensemble"):
            ensemble._check_diversity(oof_preds, y, importances, feature_names)

        warn_logs = [r for r in caplog.records if "High importance correlation" in r.message]
        assert len(warn_logs) == 3  # 全ペアが高相関

    def test_full_ensemble_with_optuna(self):
        """500行ダミーデータでtrain→predictが完走"""
        X, y = _make_binary_data(n=500, seed=20)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3)
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        preds = ensemble.predict(X.iloc[split:])
        assert len(preds) == len(X) - split
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_base_models_have_different_hp(self):
        """train後の3モデルのlearning_rateが全て異なる"""
        X, y = _make_binary_data(n=300, seed=21)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3)
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        lrs = {
            ensemble.best_params["lgbm"]["lgb_lr"],
            ensemble.best_params["xgb"]["xgb_lr"],
            ensemble.best_params["cat"]["cat_lr"],
        }
        # 全て異なる値であることを確認(3つのユニーク値)
        assert len(lrs) == 3


class TestPredictionOrthogonalization:
    """Level-1予測特徴量の直交化テスト"""

    def test_fit_prediction_orthogonalizer_reduces_lgb_xgb_corr(self):
        rng = np.random.RandomState(123)
        lgb_pred = np.linspace(0.05, 0.95, 200)
        xgb_pred = lgb_pred * 0.98 + rng.normal(0.0, 0.005, 200)
        cat_pred = rng.uniform(0.05, 0.95, 200)
        raw = np.column_stack([lgb_pred, xgb_pred, cat_pred])

        # Default: partial orthogonalization (strength=0.5)
        ensemble = StackedEnsemble(cat_cols=[], orthogonalize_threshold=0.95)
        transformed = ensemble._fit_prediction_orthogonalizer(raw)

        raw_corr = np.corrcoef(raw[:, 0], raw[:, 1])[0, 1]
        transformed_corr = np.corrcoef(transformed[:, 0], transformed[:, 1])[0, 1]
        assert raw_corr > 0.95
        assert transformed_corr < raw_corr  # reduced but not zero
        assert ensemble._orthogonalization[1]["enabled"] is True

        # Full orthogonalization (strength=1.0): near-zero correlation
        ensemble_full = StackedEnsemble(
            cat_cols=[], orthogonalize_threshold=0.95, orthogonalize_strength=1.0
        )
        transformed_full = ensemble_full._fit_prediction_orthogonalizer(raw)
        full_corr = np.corrcoef(transformed_full[:, 0], transformed_full[:, 1])[0, 1]
        assert abs(full_corr) < 0.05

    def test_apply_prediction_orthogonalizer_matches_fit_transform(self):
        rng = np.random.RandomState(456)
        lgb_pred = np.linspace(0.05, 0.95, 150)
        xgb_pred = lgb_pred * 0.99 + rng.normal(0.0, 0.003, 150)
        cat_pred = rng.uniform(0.05, 0.95, 150)
        raw = np.column_stack([lgb_pred, xgb_pred, cat_pred])

        ensemble = StackedEnsemble(cat_cols=[], orthogonalize_threshold=0.95)
        transformed = ensemble._fit_prediction_orthogonalizer(raw)
        reapplied = ensemble._apply_prediction_orthogonalizer(raw)

        assert np.allclose(transformed, reapplied)
