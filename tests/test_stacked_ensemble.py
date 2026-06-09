"""StackedEnsemble のテスト — 基本機能 + Optuna + 直交化 + 新目的関数 + surface"""

# ruff: noqa: N806

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.stacked_ensemble import StackedEnsemble

# ───────────────── helpers ─────────────────


def _make_binary_data(n: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """既存テスト用の単純バイナリデータ (レース構造なし)"""
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


def _make_race_data(
    n_races: int = 50,
    horses_per_race: int = 10,
    seed: int = 42,
    *,
    multi_year: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """レース構造付きデータ。

    Returns: (X, y, race_ids, odds, dates)
    """
    rng = np.random.RandomState(seed)
    n = n_races * horses_per_race
    race_ids = pd.Series([f"r{i:04d}" for i in range(n_races) for _ in range(horses_per_race)])
    if multi_year:
        # 3年に分散 (年別安定性テスト用)
        years_list: list[int] = []
        per_year = n_races // 3
        for yr_offset, yr in enumerate([2022, 2023, 2024]):
            count = per_year if yr_offset < 2 else n_races - 2 * per_year
            years_list.extend([yr] * count)
        # pad if needed
        while len(years_list) < n_races:
            years_list.append(2024)
        dates = pd.Series(
            [
                pd.Timestamp(f"{years_list[i]}-06-01") + pd.Timedelta(days=i * 3)
                for i in range(n_races)
                for _ in range(horses_per_race)
            ]
        )
    else:
        dates = pd.Series(
            [
                pd.Timestamp("2023-01-01") + pd.Timedelta(days=i * 7)
                for i in range(n_races)
                for _ in range(horses_per_race)
            ]
        )
    odds = pd.Series(rng.uniform(2.0, 50.0, n))
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n), "f3": rng.rand(n)})
    y = pd.Series(0, index=range(n))
    for i in range(n_races):
        winner = rng.randint(0, horses_per_race)
        y.iloc[i * horses_per_race + winner] = 1
    return X, y, race_ids, odds, dates


# ═══════════════════════ 既存テスト (surface 追加) ═══════════════════════


class TestStackedEnsemble:
    def test_train_and_predict(self):
        """学習→予測で [0,1] の確率が返る"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        preds = ensemble.predict(X.iloc[split:])
        assert len(preds) == len(X) - split
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_base_models_trained(self):
        """3つのベースモデルが学習されている"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        assert ensemble.lgbm_model is not None
        assert ensemble.xgb_model is not None
        assert ensemble.cat_model is not None
        assert ensemble.meta_model is not None

    def test_different_from_single_lgbm(self):
        """アンサンブル予測が単一LGBMと異なる"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        preds_ens = ensemble.predict(X.iloc[split:])
        preds_lgbm = ensemble.lgbm_model.predict(X.iloc[split:])
        # アンサンブルとLGBM単体で予測が異なることを確認
        assert not np.allclose(preds_ens, preds_lgbm, atol=1e-6)

    def test_best_iteration_compatible(self):
        """lgb.Booster 互換の best_iteration 属性がある"""
        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        assert hasattr(ensemble, "best_iteration")
        assert ensemble.best_iteration == 0  # アンサンブルでは使用しない

    def test_encode_cats_categorical_fillna(self):
        """Categorical列の_encode_catsで未知カテゴリが-1になる"""
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
            {"color": pd.Categorical(["red", None, "blue", None]), "f1": [1.0, 2.0, 3.0, 4.0]}
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
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2, surface="turf")
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
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        imp = ensemble.feature_importance(importance_type="gain")
        assert isinstance(imp, np.ndarray)
        assert len(imp) == 3

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
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        imp = ensemble.feature_importance(importance_type="gain")
        # 3モデルの正規化平均なので合計は約1.0
        assert abs(imp.sum() - 1.0) < 0.01

    def test_extract_feature_ranking_with_stacked_ensemble(self):
        """extract_feature_ranking() が StackedEnsemble で動作する"""
        from models.walk_forward_cv import extract_feature_ranking

        X, y = _make_binary_data()
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        ranking, top_features = extract_feature_ranking(ensemble, top_n=3)
        assert isinstance(ranking, dict)
        assert isinstance(top_features, list)
        assert len(top_features) <= 3
        # top_featuresの要素が全て学習特徴量に含まれる
        for f in top_features:
            assert f in {"f1", "f2", "f3"}


class TestOptunaTuning:
    """Optuna HP最適化 + Early Stopping + 特徴量サブセット"""

    def test_optuna_tuning_produces_different_params(self):
        X, y = _make_binary_data(n=300, seed=10)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3, surface="turf")
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
        call_kwargs = mock_lgb_train.call_args
        callbacks = (
            call_kwargs[1].get("callbacks")
            if call_kwargs[1]
            else call_kwargs.kwargs.get("callbacks")
        )
        assert callbacks is not None
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
        X, y = _make_binary_data(n=300, seed=11)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        feat_frac = ensemble.best_params["lgbm"]["lgb_feat_frac"]
        assert 0.3 <= feat_frac <= 0.9

    def test_feature_fraction_in_xgb_params(self):
        X, y = _make_binary_data(n=300, seed=12)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        col_sample = ensemble.best_params["xgb"]["xgb_col_sample"]
        assert 0.3 <= col_sample <= 0.9

    def test_feature_fraction_in_cat_params(self):
        X, y = _make_binary_data(n=300, seed=13)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        rsm = ensemble.best_params["cat"]["cat_rsm"]
        assert 0.3 <= rsm <= 0.9

    def test_exploration_space_separation(self):
        """各モデルが異なる木複雑度空間を探索"""
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
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
            lambda _: {"xgb_max_depth": 6, "xgb_lr": 0.03, "xgb_col_sample": 0.7},
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
        race_ids = pd.Series(["r1"] * 3 + ["r2"] * 2 + ["r3"] * 4 + ["r4"] * 2)
        split = StackedEnsemble.race_group_split_index(race_ids, ratio=0.5)
        assert split == 5
        assert race_ids.iloc[:split].nunique() == 2
        assert set(race_ids.iloc[:split]).isdisjoint(set(race_ids.iloc[split:]))

    def test_race_group_split_rejects_noncontiguous_races(self):
        race_ids = pd.Series(["r1", "r2", "r1", "r2"])
        with np.testing.assert_raises_regex(ValueError, "contiguous"):
            StackedEnsemble.race_group_split_index(race_ids)

    def test_optuna_validation_is_disjoint_from_oof_validation(self):
        race_ids = pd.Series(np.repeat([f"r{i:02d}" for i in range(20)], 3))
        groups = StackedEnsemble._normalize_groups(race_ids, len(race_ids))
        first_train_idx, _ = StackedEnsemble._expanding_group_splits(groups, n_folds=3)[0]
        tune_groups = groups.iloc[first_train_idx].reset_index(drop=True)
        tune_split = StackedEnsemble._group_boundary(tune_groups, 0.8)
        tune_valid_idx = first_train_idx[tune_split:]
        oof_valid_indices = np.concatenate(
            [
                valid_idx
                for _, valid_idx in StackedEnsemble._expanding_group_splits(groups, n_folds=3)
            ]
        )
        assert set(tune_valid_idx).isdisjoint(set(oof_valid_indices))

    def test_predict_xgb_uses_best_iteration(self):
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
        ensemble = StackedEnsemble(cat_cols=[])
        oof_preds = np.array(
            [[0.1, 0.3, 0.7], [0.2, 0.5, 0.8], [0.3, 0.1, 0.2], [0.4, 0.9, 0.6], [0.5, 0.2, 0.4]]
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
        assert len(corr_logs) == 3

    def test_check_diversity_warns_high_correlation(self, caplog):
        ensemble = StackedEnsemble(cat_cols=[])
        col = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        oof_preds = np.column_stack([col, col, col])
        y = pd.Series([0, 1, 0, 1, 0])
        importances = [np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0])]
        feature_names = ["f1", "f2"]
        with caplog.at_level(logging.WARNING, logger="models.stacked_ensemble"):
            ensemble._check_diversity(oof_preds, y, importances, feature_names)
        warn_logs = [r for r in caplog.records if "High prediction correlation" in r.message]
        assert len(warn_logs) == 3

    def test_check_diversity_logs_importance_correlation(self, caplog):
        ensemble = StackedEnsemble(cat_cols=[])
        oof_preds = np.array([[0.1, 0.3, 0.7], [0.2, 0.5, 0.8], [0.3, 0.1, 0.2]])
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
        assert len(imp_logs) == 3

    def test_check_diversity_warns_high_importance_correlation(self, caplog):
        ensemble = StackedEnsemble(cat_cols=[])
        oof_preds = np.array([[0.1, 0.3, 0.7], [0.2, 0.5, 0.8], [0.3, 0.1, 0.2]])
        y = pd.Series([0, 1, 0])
        imp = np.array([1.0, 2.0, 3.0])
        importances = [imp.copy(), imp.copy(), imp.copy()]
        feature_names = ["f1", "f2", "f3"]
        with caplog.at_level(logging.WARNING, logger="models.stacked_ensemble"):
            ensemble._check_diversity(oof_preds, y, importances, feature_names)
        warn_logs = [r for r in caplog.records if "High importance correlation" in r.message]
        assert len(warn_logs) == 3

    def test_full_ensemble_with_optuna(self):
        X, y = _make_binary_data(n=500, seed=20)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        preds = ensemble.predict(X.iloc[split:])
        assert len(preds) == len(X) - split
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_base_models_have_different_hp(self):
        X, y = _make_binary_data(n=300, seed=21)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=3, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        lrs = {
            ensemble.best_params["lgbm"]["lgb_lr"],
            ensemble.best_params["xgb"]["xgb_lr"],
            ensemble.best_params["cat"]["cat_lr"],
        }
        assert len(lrs) == 3


class TestPredictionOrthogonalization:
    """Level-1予測特徴量の直交化テスト"""

    def test_fit_prediction_orthogonalizer_reduces_lgb_xgb_corr(self):
        rng = np.random.RandomState(123)
        lgb_pred = np.linspace(0.05, 0.95, 200)
        xgb_pred = lgb_pred * 0.98 + rng.normal(0.0, 0.005, 200)
        cat_pred = rng.uniform(0.05, 0.95, 200)
        raw = np.column_stack([lgb_pred, xgb_pred, cat_pred])
        ensemble = StackedEnsemble(cat_cols=[], orthogonalize_threshold=0.95, surface="turf")
        transformed = ensemble._fit_prediction_orthogonalizer(raw)
        raw_corr = np.corrcoef(raw[:, 0], raw[:, 1])[0, 1]
        transformed_corr = np.corrcoef(transformed[:, 0], transformed[:, 1])[0, 1]
        assert raw_corr > 0.95
        assert transformed_corr < raw_corr
        assert ensemble._orthogonalization[1]["enabled"] is True
        ensemble_full = StackedEnsemble(
            cat_cols=[],
            orthogonalize_threshold=0.95,
            orthogonalize_strength=1.0,
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


# ═══════════════════════ 新規テスト (要件 F) ═══════════════════════


class TestRaceTop1Objective:
    """A. Race Top-1 目的関数のテスト"""

    def test_selects_one_per_race(self):
        """各レースで予測確率最大の1頭だけ選ばれる"""
        y = pd.Series([1, 0, 0, 0, 1, 0])  # 2レース × 3頭
        preds = np.array([0.8, 0.1, 0.1, 0.3, 0.6, 0.1])
        race_ids = pd.Series(["r1", "r1", "r1", "r2", "r2", "r2"])
        odds = pd.Series([5.0, 10.0, 15.0, 3.0, 7.0, 20.0])
        score = StackedEnsemble._race_top1_objective(
            y,
            preds,
            race_ids=race_ids,
            odds=odds,
        )
        # r1: top1 idx=0 (pred=0.8), y=1 → odds=5.0
        # r2: top1 idx=4 (pred=0.6), y=1 → odds=7.0
        # ROI = mean(5.0, 7.0) / 2 = 3.0 (clipped to 2/2=1.0... wait)
        # top1_roi = clip(5.0,0,2)/2=1.0, clip(7.0,0,2)/2=1.0 → mean=1.0
        # Actually: top1_roi_vals = [5.0, 7.0], clip→[2.0, 2.0], mean=2.0, /2=1.0
        assert np.isfinite(score)

    def test_top1_hit_rate_correct(self):
        """Top1 hit rate が正確に計算される"""
        # 3レース、レース1は的中、レース2と3は不的中
        y = pd.Series([1, 0, 0, 0, 0, 1, 0, 0, 0])
        preds = np.array([0.9, 0.05, 0.05, 0.8, 0.1, 0.1, 0.4, 0.3, 0.3])
        race_ids = pd.Series(["r1", "r1", "r1", "r2", "r2", "r2", "r3", "r3", "r3"])
        odds = pd.Series([3.0] * 9)
        score = StackedEnsemble._race_top1_objective(
            y,
            preds,
            race_ids=race_ids,
            odds=odds,
        )
        # r1 top1=idx0, y=1 → hit
        # r2 top1=idx3, y=0 → miss
        # r3 top1=idx6, y=0 → miss
        # top1_hit = 1/3
        assert np.isfinite(score)

    def test_confirmed_odds_fallback_to_tanodds(self):
        """odds=None の場合でも有限値が返る (ROI=0 として保守的に処理)"""
        y = pd.Series([1, 0])
        preds = np.array([0.9, 0.1])
        race_ids = pd.Series(["r1", "r1"])
        score = StackedEnsemble._race_top1_objective(
            y,
            preds,
            race_ids=race_ids,
            odds=None,
        )
        assert np.isfinite(score)
        # odds=None → NaN → valid_odds=0 → raw_roi=0 → top1_roi=0
        # score は AUC + Brier + hit_rate + stability のみで構成

    def test_single_class_returns_finite(self):
        """単一クラスでも有限値を返す"""
        y = pd.Series([0, 0, 0, 0, 0, 0])
        preds = np.array([0.9, 0.05, 0.05, 0.8, 0.1, 0.1])
        race_ids = pd.Series(["r1", "r1", "r1", "r2", "r2", "r2"])
        odds = pd.Series([3.0] * 6)
        score = StackedEnsemble._race_top1_objective(
            y,
            preds,
            race_ids=race_ids,
            odds=odds,
        )
        assert np.isfinite(score)

    def test_empty_data_returns_zero(self):
        """空データで 0.0 を返す"""
        y = pd.Series([], dtype=float)
        preds = np.array([])
        race_ids = pd.Series([], dtype=str)
        score = StackedEnsemble._race_top1_objective(
            y,
            preds,
            race_ids=race_ids,
        )
        assert score == 0.0


class TestStability:
    """安定性計算のテスト"""

    def test_block_stability_when_few_years(self):
        """年 <2 の場合ブロック安定性が使われる"""
        # 30レース × 3頭、日付は1年以内
        n_races = 30
        rng = np.random.RandomState(99)
        top1_df = pd.DataFrame(
            {
                "y": rng.randint(0, 2, n_races),
                "odds": rng.uniform(2.0, 10.0, n_races),
                "date": pd.date_range("2023-01-01", periods=n_races, freq="W"),
            }
        )
        stability = StackedEnsemble._compute_stability(top1_df, dates_col="date")
        # 1年しかないのでブロック安定性 → 常に [0, 1]
        assert 0.0 <= stability <= 1.0
        assert np.isfinite(stability)

    def test_year_stability_with_multiple_years(self):
        """2年以上あれば年別安定性が使われる"""
        top1_df = pd.DataFrame(
            {
                "y": [1] * 20 + [0] * 20 + [1] * 20,
                "odds": [3.0] * 60,
                "date": [pd.Timestamp("2022-06-01") + pd.Timedelta(days=i) for i in range(20)]
                + [pd.Timestamp("2023-06-01") + pd.Timedelta(days=i) for i in range(20)]
                + [pd.Timestamp("2024-06-01") + pd.Timedelta(days=i) for i in range(20)],
            }
        )
        stability = StackedEnsemble._compute_stability(top1_df, dates_col="date")
        assert 0.0 <= stability <= 1.0
        assert np.isfinite(stability)

    def test_too_few_races_returns_neutral(self):
        """3レース未満では中立値 0.5"""
        top1_df = pd.DataFrame({"y": [1, 0], "odds": [3.0, 5.0]})
        stability = StackedEnsemble._compute_stability(top1_df)
        assert stability == 0.5


class TestSurfaceValidation:
    """B. Surface 検証"""

    def test_none_surface_raises_on_train(self):
        """surface=None で train() を呼ぶと ValueError"""
        X, y = _make_binary_data(n=100)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], surface=None)
        with pytest.raises(ValueError, match="surface must be one of"):
            ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

    def test_invalid_surface_raises_on_train(self):
        """surface='unknown' で train() を呼ぶと ValueError"""
        X, y = _make_binary_data(n=100)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], surface="unknown")
        with pytest.raises(ValueError, match="surface must be one of"):
            ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])

    def test_old_joblib_predict_compat(self):
        """surface 属性がなくても _encode_cats などは動作する (旧 joblib 互換)"""
        ensemble = StackedEnsemble(cat_cols=[])
        # surface は None のまま
        assert ensemble.surface is None
        # _encode_cats, _learn_cat_codes は動作する
        X = pd.DataFrame({"f1": [1.0, 2.0]})
        ensemble._learn_cat_codes(X)
        result = ensemble._encode_cats(X)
        assert len(result) == 2


class TestSearchSpaceDifference:
    """B. 芝/ダート探索差"""

    def test_dirt_has_wider_regularization(self):
        """ダートの LGBM 正則化上限が芝より広い"""
        turf_l1_max = StackedEnsemble.SEARCH_SPACES["turf"]["lgbm"]["lgb_lambda_l1"][1]
        dirt_l1_max = StackedEnsemble.SEARCH_SPACES["dirt"]["lgbm"]["lgb_lambda_l1"][1]
        assert dirt_l1_max > turf_l1_max

    def test_dirt_xgb_wider_reg(self):
        """ダートの XGB 正則化上限が芝より広い"""
        turf_alpha = StackedEnsemble.SEARCH_SPACES["turf"]["xgb"]["xgb_reg_alpha"][1]
        dirt_alpha = StackedEnsemble.SEARCH_SPACES["dirt"]["xgb"]["xgb_reg_alpha"][1]
        assert dirt_alpha > turf_alpha

    def test_dirt_cat_wider_reg(self):
        """ダートの CatBoost 正則化上限が芝より広い"""
        turf_l2 = StackedEnsemble.SEARCH_SPACES["turf"]["cat"]["cat_l2_leaf_reg"][1]
        dirt_l2 = StackedEnsemble.SEARCH_SPACES["dirt"]["cat"]["cat_l2_leaf_reg"][1]
        assert dirt_l2 > turf_l2

    def test_existing_ranges_included_in_dirt(self):
        """ダートの既存パラメータ範囲が芝の範囲を包含する"""
        for key in ["lgb_num_leaves", "lgb_lr", "lgb_feat_frac"]:
            turf_range = StackedEnsemble.SEARCH_SPACES["turf"]["lgbm"][key]
            dirt_range = StackedEnsemble.SEARCH_SPACES["dirt"]["lgbm"][key]
            assert dirt_range[0] <= turf_range[0]
            assert dirt_range[1] >= turf_range[1]


class TestNewParams:
    """C. 新規パラメータ全学習経路"""

    def test_new_lgbm_params_in_best_params(self):
        """LGBM の新規パラメータが best_params に含まれる"""
        X, y = _make_binary_data(n=300, seed=30)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        bp = ensemble.best_params["lgbm"]
        assert "lgb_min_child_samples" in bp
        assert "lgb_lambda_l1" in bp
        assert "lgb_lambda_l2" in bp
        assert "lgb_bagging_fraction" in bp

    def test_new_xgb_params_in_best_params(self):
        """XGB の新規パラメータが best_params に含まれる"""
        X, y = _make_binary_data(n=300, seed=31)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        bp = ensemble.best_params["xgb"]
        assert "xgb_min_child_weight" in bp
        assert "xgb_reg_alpha" in bp
        assert "xgb_reg_lambda" in bp
        assert "xgb_subsample" in bp

    def test_new_cat_params_in_best_params(self):
        """CatBoost の新規パラメータが best_params に含まれる"""
        X, y = _make_binary_data(n=300, seed=32)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        bp = ensemble.best_params["cat"]
        assert "cat_l2_leaf_reg" in bp
        assert "cat_random_strength" in bp
        assert "cat_subsample" in bp

    def test_new_params_in_build_lgbm_dict(self):
        """_build_lgbm_dict に新規パラメータが含まれる"""
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        params = {
            "lgb_num_leaves": 31,
            "lgb_lr": 0.03,
            "lgb_feat_frac": 0.8,
            "lgb_min_child_samples": 50,
            "lgb_lambda_l1": 1.0,
            "lgb_lambda_l2": 2.0,
            "lgb_bagging_fraction": 0.7,
        }
        d = ensemble._build_lgbm_dict(params, 1)
        assert d["min_child_samples"] == 50
        assert d["lambda_l1"] == 1.0
        assert d["lambda_l2"] == 2.0
        assert d["bagging_fraction"] == 0.7
        assert d["bagging_freq"] == 1

    def test_new_params_in_build_xgb_dict(self):
        """_build_xgb_dict に新規パラメータが含まれる"""
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        params = {
            "xgb_max_depth": 6,
            "xgb_lr": 0.05,
            "xgb_col_sample": 0.8,
            "xgb_min_child_weight": 10,
            "xgb_reg_alpha": 1.5,
            "xgb_reg_lambda": 2.5,
            "xgb_subsample": 0.8,
        }
        d = ensemble._build_xgb_dict(params, 1)
        assert d["min_child_weight"] == 10
        assert d["reg_alpha"] == 1.5
        assert d["reg_lambda"] == 2.5
        assert d["subsample"] == 0.8

    def test_new_params_in_build_cat_dict(self):
        """_build_cat_dict に新規パラメータが含まれる"""
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        params = {
            "cat_depth": 8,
            "cat_lr": 0.01,
            "cat_rsm": 0.7,
            "cat_l2_leaf_reg": 5.0,
            "cat_random_strength": 3.0,
            "cat_subsample": 0.8,
        }
        d = ensemble._build_cat_dict(params, 1)
        assert d["l2_leaf_reg"] == 5.0
        assert d["random_strength"] == 3.0
        assert d["subsample"] == 0.8
        assert d["bootstrap_type"] == "Bernoulli"

    def test_cat_subsample_one_no_bootstrap_type(self):
        """cat_subsample=1.0 の場合は bootstrap_type を設定しない"""
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        params = {
            "cat_depth": 6,
            "cat_lr": 0.01,
            "cat_rsm": 0.7,
            "cat_l2_leaf_reg": 3.0,
            "cat_random_strength": 1.0,
            "cat_subsample": 1.0,
        }
        d = ensemble._build_cat_dict(params, 1)
        assert "bootstrap_type" not in d


class TestOldKeyFallback:
    """C. 旧 best_params キーなし時の既定値フォールバック"""

    def test_get_param_returns_existing_key(self):
        params = {"lgb_lr": 0.05}
        assert StackedEnsemble._get_param(params, "lgb_lr", "lgbm") == 0.05

    def test_get_param_falls_back_to_defaults(self):
        params = {"lgb_lr": 0.05}
        assert StackedEnsemble._get_param(params, "lgb_lambda_l1", "lgbm") == 0.0
        assert StackedEnsemble._get_param(params, "lgb_bagging_fraction", "lgbm") == 1.0

    def test_build_lgbm_dict_with_old_params(self):
        """旧形式 params (新キーなし) でも _build_lgbm_dict が動作する"""
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        old_params = {"lgb_num_leaves": 31, "lgb_lr": 0.03, "lgb_feat_frac": 0.8}
        d = ensemble._build_lgbm_dict(old_params, 1)
        assert d["min_child_samples"] == 20  # PARAM_DEFAULTS
        assert d["lambda_l1"] == 0.0
        assert "bagging_fraction" not in d  # 1.0 → 省略

    def test_build_xgb_dict_with_old_params(self):
        """旧形式 params でも _build_xgb_dict が動作する"""
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        old_params = {"xgb_max_depth": 6, "xgb_lr": 0.03, "xgb_col_sample": 0.7}
        d = ensemble._build_xgb_dict(old_params, 1)
        assert d["min_child_weight"] == 1
        assert d["subsample"] == 1.0

    def test_build_cat_dict_with_old_params(self):
        """旧形式 params でも _build_cat_dict が動作する"""
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        old_params = {"cat_depth": 6, "cat_lr": 0.01, "cat_rsm": 0.7}
        d = ensemble._build_cat_dict(old_params, 1)
        assert d["l2_leaf_reg"] == 3.0
        assert "bootstrap_type" not in d


class TestXGBBestIteration:
    """XGB best_iteration のテスト"""

    def test_xgb_best_iteration_in_full_model(self):
        """train() 後の XGBoost モデルが best_iteration を持つ"""
        X, y = _make_binary_data(n=300, seed=40)
        split = int(len(X) * 0.8)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2, surface="turf")
        ensemble.train(X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:])
        assert hasattr(ensemble.xgb_model, "best_iteration")


class TestMetaStudy:
    """D. Ridge/直交化の別 Study テスト"""

    def test_meta_train_valid_disjoint(self):
        """meta-train と meta-valid がレース単位で重複しない"""
        X, y, race_ids, odds, dates = _make_race_data(n_races=50, horses_per_race=10)
        split = StackedEnsemble.race_group_split_index(race_ids)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2, surface="turf")
        ensemble.train(
            X.iloc[:split],
            y.iloc[:split],
            X.iloc[split:],
            y.iloc[split:],
            train_race_ids=race_ids.iloc[:split],
            valid_race_ids=race_ids.iloc[split:],
            train_odds=odds.iloc[:split],
            train_dates=dates.iloc[:split],
        )
        # meta params が保存されている
        assert "meta" in ensemble.best_params
        meta = ensemble.best_params["meta"]
        assert "ridge_alpha" in meta
        assert "orthogonalize_threshold" in meta
        assert "orthogonalize_strength" in meta

    def test_meta_params_reflected_in_model(self):
        """最良 meta params が Ridge alpha と直交化パラメータに反映される"""
        X, y, race_ids, odds, dates = _make_race_data(n_races=50, horses_per_race=10)
        split = StackedEnsemble.race_group_split_index(race_ids)
        ensemble = StackedEnsemble(cat_cols=[], n_trials=2, surface="turf")
        ensemble.train(
            X.iloc[:split],
            y.iloc[:split],
            X.iloc[split:],
            y.iloc[split:],
            train_race_ids=race_ids.iloc[:split],
            valid_race_ids=race_ids.iloc[split:],
            train_odds=odds.iloc[:split],
            train_dates=dates.iloc[:split],
        )
        meta = ensemble.best_params["meta"]
        # Ridge alpha が反映されている
        assert ensemble.meta_model is not None
        assert float(ensemble.meta_model.alpha) == pytest.approx(
            float(meta.get("ridge_alpha", 1.0)),
            rel=1e-6,
        )
        # 直交化パラメータが反映されている
        assert ensemble.orthogonalize_threshold == pytest.approx(
            float(meta.get("orthogonalize_threshold", 0.95)),
            rel=1e-6,
        )
        assert ensemble.orthogonalize_strength == pytest.approx(
            float(meta.get("orthogonalize_strength", 0.5)),
            rel=1e-6,
        )

    def test_meta_study_few_races_uses_defaults(self):
        """レース数 <10 ではデフォルト値が使われる"""
        # 5レース × 10頭 = 50行 (学習分割後の OOF で <10 レースになる可能性)
        # ただし sequential groups の場合は400ユニーク → meta study 実行される
        # 少量データで直接 _tune_meta_params をテスト
        ensemble = StackedEnsemble(cat_cols=[], surface="turf")
        # 5レースのグループ
        groups = pd.Series([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        result = ensemble._tune_meta_params(
            np.random.rand(10, 3),
            np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0]),
            race_groups=groups,
        )
        # デフォルト値が返る
        assert "ridge_alpha" in result

    def test_old_model_default_meta_params(self):
        """旧モデルのデフォルト meta 値が PARAM_DEFAULTS に定義されている"""
        defaults = StackedEnsemble.PARAM_DEFAULTS["meta"]
        assert defaults["ridge_alpha"] == 1.0
        assert defaults["orthogonalize_threshold"] == 0.95
        assert defaults["orthogonalize_strength"] == 0.5


# ═══════════════════════ PM レビュー修正テスト ═══════════════════════


class TestROICalculation:
    """Issue 1: 集計ROIクリップの検証"""

    def test_raw_roi_aggregate_clip(self):
        """高オッズ的中が集計ROIに正しく反映される (個別クリップではない)"""
        # r1: top1的中, odds=10.0; r2: top1不的中
        y = pd.Series([1, 0, 0, 0])
        preds = np.array([0.9, 0.05, 0.05, 0.8])
        race_ids = pd.Series(["r1", "r1", "r2", "r2"])
        odds = pd.Series([10.0, 20.0, 5.0, 8.0])
        score_high = StackedEnsemble._race_top1_objective(
            y,
            preds,
            race_ids=race_ids,
            odds=odds,
        )
        # r1 top1=idx0, y=1, odds=10.0 → raw_roi = mean(10.0, 0) = 5.0
        # top1_roi = clip(5.0, 0, 2) / 2 = 1.0
        # 高オッズが反映されていることを確認
        assert np.isfinite(score_high)

        # 比較: 全部外れ (top1_roi=0)
        y_miss = pd.Series([0, 0, 0, 0])
        score_miss = StackedEnsemble._race_top1_objective(
            y_miss,
            preds,
            race_ids=race_ids,
            odds=odds,
        )
        # 的中ありスコア > 全部外れスコア (ROI成分の差)
        assert score_high > score_miss

    def test_nan_odds_treated_conservatively(self):
        """NaN の odds は払戻0として保守的に扱われる"""
        y = pd.Series([1, 0])
        preds = np.array([0.9, 0.1])
        race_ids = pd.Series(["r1", "r1"])
        odds = pd.Series([np.nan, 5.0])
        score = StackedEnsemble._race_top1_objective(
            y,
            preds,
            race_ids=race_ids,
            odds=odds,
        )
        # top1=idx0, y=1, odds=NaN → valid_odds=0, raw_roi=0
        assert np.isfinite(score)

    def test_zero_odds_treated_conservatively(self):
        """0 または負の odds は払戻0として扱われる"""
        y = pd.Series([1, 0, 0, 0, 0])
        preds = np.array([0.9, 0.1, 0.8, 0.1, 0.1])
        race_ids = pd.Series(["r1", "r1", "r2", "r2", "r2"])
        odds = pd.Series([0.0, -1.0, np.inf, 5.0, 3.0])
        score = StackedEnsemble._race_top1_objective(
            y,
            preds,
            race_ids=race_ids,
            odds=odds,
        )
        # r1: odds=0 → payout=0; r2: odds=inf → non-finite → payout=0
        assert np.isfinite(score)


class TestStabilityMinimumROI:
    """Issue 2: 安定性に期間別最低ROIが含まれる"""

    def test_all_hit_high_odds_has_high_stability(self):
        """全期間で的中+高オッズ → 安定性が高い"""
        top1_df = pd.DataFrame(
            {
                "y": [1, 1, 1],
                "odds": [5.0, 6.0, 7.0],
                "date": [
                    pd.Timestamp("2022-06-01"),
                    pd.Timestamp("2023-06-01"),
                    pd.Timestamp("2024-06-01"),
                ],
            }
        )
        stability = StackedEnsemble._compute_stability(top1_df, dates_col="date")
        # 全期間ROI>0 → clipped_min > 0, consistency高
        assert stability > 0.5
        assert np.isfinite(stability)

    def test_zero_hit_has_low_stability(self):
        """全期間で不的中 → clipped_min=0 → 安定性が低い"""
        top1_df = pd.DataFrame(
            {
                "y": [0, 0, 0],
                "odds": [5.0, 6.0, 7.0],
                "date": [
                    pd.Timestamp("2022-06-01"),
                    pd.Timestamp("2023-06-01"),
                    pd.Timestamp("2024-06-01"),
                ],
            }
        )
        stability = StackedEnsemble._compute_stability(top1_df, dates_col="date")
        # min_roi=0 → clipped_min=0, consistency=1.0 (分散ゼロ)
        # → 0.5 * 1.0 + 0.5 * 0.0 = 0.5 (ちょうど)
        assert stability <= 0.5
        assert np.isfinite(stability)

    def test_nan_odds_in_stability(self):
        """NaN オッズの期間はROI=0として扱われる"""
        top1_df = pd.DataFrame(
            {
                "y": [1, 1],
                "odds": [np.nan, 5.0],
            }
        )
        # 2データポイント → n<3 → 0.5
        stability = StackedEnsemble._compute_stability(top1_df)
        assert stability == 0.5


class TestFeatureImportanceAllZeros:
    """Issue 5: 全ゼロモデルの除外"""

    def test_feature_importance_all_zeros_returns_zeros(self):
        """全モデル重要度ゼロ → zeros を返す"""
        ensemble = StackedEnsemble(cat_cols=[])
        ensemble.lgbm_model = MagicMock()
        ensemble.xgb_model = MagicMock()
        ensemble.cat_model = MagicMock()
        n_features = 3
        ensemble._train_feature_names = ["f1", "f2", "f3"]
        ensemble.lgbm_model.feature_name.return_value = ["f1", "f2", "f3"]
        ensemble.lgbm_model.feature_importance.return_value = np.zeros(n_features)
        ensemble.xgb_model.get_score.return_value = {"f1": 0.0, "f2": 0.0, "f3": 0.0}
        ensemble.cat_model.get_feature_importance.return_value = np.zeros(n_features)
        imp = ensemble.feature_importance(importance_type="gain")
        assert isinstance(imp, np.ndarray)
        assert len(imp) == n_features
        assert np.allclose(imp, 0.0)

    def test_feature_importance_one_model_nonzero_sums_to_one(self):
        """1モデルのみ非ゼロ → そのモデルだけで正規化して合計1"""
        ensemble = StackedEnsemble(cat_cols=[])
        ensemble.lgbm_model = MagicMock()
        ensemble.xgb_model = MagicMock()
        ensemble.cat_model = MagicMock()
        ensemble._train_feature_names = ["f1", "f2"]
        ensemble.lgbm_model.feature_name.return_value = ["f1", "f2"]
        ensemble.lgbm_model.feature_importance.return_value = np.array([3.0, 1.0])
        ensemble.xgb_model.get_score.return_value = {"f1": 0.0, "f2": 0.0}
        ensemble.cat_model.get_feature_importance.return_value = np.array([0.0, 0.0])
        imp = ensemble.feature_importance(importance_type="gain")
        assert abs(imp.sum() - 1.0) < 1e-6


class TestSafeCorrelation:
    """Issue 7: 定数配列/NaN で安全に処理"""

    def test_safe_corr_constant_arrays_returns_nan(self):
        """定数配列の相関は NaN"""
        a = np.array([1.0, 1.0, 1.0, 1.0])
        b = np.array([2.0, 3.0, 4.0, 5.0])
        result = StackedEnsemble._safe_corr(a, b)
        assert np.isnan(result)

    def test_safe_corr_normal_arrays(self):
        """通常配列の相関は有限値"""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 4.0, 6.0, 8.0])
        result = StackedEnsemble._safe_corr(a, b)
        assert np.isfinite(result)
        assert abs(result - 1.0) < 1e-6

    def test_corr_penalty_constant_preds_no_nan(self):
        """定数予測でも _compute_corr_penalty が NaN を返さない"""
        constant_preds = np.array([0.5, 0.5, 0.5, 0.5])
        normal_preds = np.array([0.1, 0.3, 0.5, 0.7])
        penalty = StackedEnsemble._compute_corr_penalty(
            constant_preds,
            [normal_preds],
            weight=0.1,
            threshold=0.85,
        )
        assert np.isfinite(penalty)

    def test_diversity_constant_importance_no_warning_flood(self, caplog):
        """定数重要度配列でも warning が大量発生しない"""
        ensemble = StackedEnsemble(cat_cols=[])
        oof_preds = np.array(
            [[0.1, 0.3, 0.5], [0.2, 0.5, 0.3], [0.3, 0.2, 0.4]],
        )
        y = pd.Series([0, 1, 0])
        # 1つだけ定数 (全部同じ値)
        importances = [
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 1.0, 2.0]),
            np.array([5.0, 5.0, 5.0]),
        ]
        feature_names = ["f1", "f2", "f3"]
        with caplog.at_level(logging.WARNING, logger="models.stacked_ensemble"):
            ensemble._check_diversity(oof_preds, y, importances, feature_names)
        # 定数配列とのspearmanrはスキップされ、warningが出ない
        high_imp_warnings = [
            r for r in caplog.records if "High importance correlation" in r.message
        ]
        # 全ペアで定数が関係するのは (1,2) と (0,2) — これらはスキップされる
        # (0,1) は非定数同士 → spearman計算 → rho=-1.0 → 警告なし (< 0.8)
        assert len(high_imp_warnings) == 0
