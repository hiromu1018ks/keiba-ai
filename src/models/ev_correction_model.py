"""EV補正モデル -- P/E分解で独立性破綻を解決 (C-5)"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd


class EVCorrectionModel:
    """
    2段階モデルの「独立性破綻」を補正するモデル。

    v5.4: P補正モデルとE補正モデルに分解
    v5.5: P補正に init_score = logit(p_pred) を設定 (再学習化の防止)
          E補正の weight を 1/√p に変更 (ノイズ過剰適合の防止)

    P補正: 全サンプルで binary classification (init_score付き)
    E補正: 1着馬のみで log residual を 1/√p 重み付き回帰
    最終:  EV_corrected = P_corrected × E_corrected
    """

    E_CLIP_FLOOR: float = 1.0

    FEATURE_COLS: list[str] = [
        # 2段階モデルの出力 (v5.5: p_win_pred を除外 → init_scoreで代替)
        "e_return_win_pred",
        # 交互作用特徴量
        "p_x_e_interaction",
        "p_minus_e_gap",
        # 市場歪み
        "signed_log_error_win",
        "abs_log_error_win",
        "market_entropy",
        "popularity_rank",
        # FLB slope (市場集中度)
        "implied_prob_hhi",
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "field_size",
        # 騎手コンテキスト (Group C, Stage2)
        "jockey_wr_overall",
        "jockey_wr_distance",
        "jockey_wr_venue",
        "jockey_prize_log",
        # 調教師コンテキスト (Group D, Stage2)
        "trainer_wr_overall",
        "trainer_wr_distance",
        "trainer_wr_venue",
        "trainer_prize_log",
        # 騎手-調教師コンビ (B4, Stage2)
        "jt_combo_wr",
        "jt_combo_place_rate",
        "jt_combo_starts",
        "jt_combo_prize_log",
    ]

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """交互作用特徴量を追加"""
        df["p_x_e_interaction"] = df["p_win_pred"] * df["e_return_win_pred"]
        df["p_minus_e_gap"] = np.abs(
            np.log(df["p_win_pred"] + 1e-8) - np.log(df["e_return_win_pred"] + 1e-8)
        )
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量データフレームを準備する"""
        features = df[self.FEATURE_COLS].copy()
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin"]:
            if col in features.columns:
                features[col] = features[col].astype("category")
        return features

    def train(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        """P補正モデルとE補正モデルをそれぞれ学習"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        df = df.copy()
        assert "ev_win" in df.columns, (
            "ev_win が必要です。先に WinTwoStageModel.predict_ev() を実行してください"
        )

        df = self._add_interaction_features(df)
        features = self._prepare_features(df)

        # ── Model P: P補正 (全サンプル・binary classification) ──
        y_p = (df["kakuteijyuni"] == 1).astype(int)
        p_pred_clipped = np.clip(df["p_win_pred"], 1e-4, 1 - 1e-4)
        init_score = np.log(p_pred_clipped / (1 - p_pred_clipped))

        # P correction: train/valid split (80/20) with init_score (時系列分割)
        n_p = len(features)
        split_p = int(n_p * 0.8)
        train_idx_p = np.arange(split_p)
        valid_idx_p = np.arange(split_p, n_p)

        train_data_p = lgb.Dataset(
            features.iloc[train_idx_p],
            label=y_p.iloc[train_idx_p],
            init_score=init_score[train_idx_p],
        )
        valid_data_p = lgb.Dataset(
            features.iloc[valid_idx_p],
            label=y_p.iloc[valid_idx_p],
            init_score=init_score[valid_idx_p],
            reference=train_data_p,
        )

        self.p_correction_model = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "is_unbalance": True,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data_p,
            num_boost_round=300,
            valid_sets=[valid_data_p],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        # ── Model E: E補正 (1着馬のみ・1/√p 重み付き回帰) ──
        winners = df[df["kakuteijyuni"] == 1].copy()
        e_pred_clipped = np.clip(winners["e_return_win_pred"], self.E_CLIP_FLOOR, None)
        winners["log_e_correction"] = np.log(
            winners["confirmed_odds"].clip(lower=self.E_CLIP_FLOOR)
        ) - np.log(e_pred_clipped)
        winners["_e_sample_weight"] = 1.0 / np.sqrt(np.clip(winners["p_win_pred"], 0.01, None))

        features_e = self._prepare_features(winners)
        e_weight = winners["_e_sample_weight"].values

        # E correction: train/valid split (80/20) with weight (時系列分割)
        n_e = len(features_e)
        split_e = int(n_e * 0.8)
        train_idx_e = np.arange(split_e)
        valid_idx_e = np.arange(split_e, n_e)

        train_data_e = lgb.Dataset(
            features_e.iloc[train_idx_e],
            label=winners["log_e_correction"].iloc[train_idx_e],
            weight=e_weight[train_idx_e],
        )
        valid_data_e = lgb.Dataset(
            features_e.iloc[valid_idx_e],
            label=winners["log_e_correction"].iloc[valid_idx_e],
            weight=e_weight[valid_idx_e],
            reference=train_data_e,
        )

        self.e_correction_model = lgb.train(
            {
                "objective": "regression_l1",
                "metric": "mae",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data_e,
            num_boost_round=300,
            valid_sets=[valid_data_e],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

    def correct_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全馬のEVをP補正×E補正で補正する。
        P_corrected = sigmoid(logit(P_pred) + correction_logit)  ← [0,1] に制約
        E_corrected = e_return_win_pred × exp(log_e_correction)
        EV_corrected = P_corrected × E_corrected
        """
        df = df.copy()
        df = self._add_interaction_features(df)
        features = self._prepare_features(df)

        # P補正の適用 (binary出力 → sigmoid で [0,1] に制約)
        p_pred_clipped = np.clip(df["p_win_pred"], 1e-4, 1 - 1e-4)
        init_score = np.log(p_pred_clipped / (1 - p_pred_clipped))
        p_best = (
            self.p_correction_model.best_iteration
            if self.p_correction_model.best_iteration > 0
            else None
        )
        p_correction_logit = (
            self.p_correction_model.predict(features, num_iteration=p_best) + init_score
        )
        df["p_win_corrected"] = 1.0 / (1.0 + np.exp(-p_correction_logit))

        # E補正の適用
        e_best = (
            self.e_correction_model.best_iteration
            if self.e_correction_model.best_iteration > 0
            else None
        )
        log_e_corr = self.e_correction_model.predict(features, num_iteration=e_best)
        df["e_return_win_corrected"] = df["e_return_win_pred"] * np.exp(log_e_corr)

        # 最終補正EV
        df["ev_win_corrected"] = df["p_win_corrected"] * df["e_return_win_corrected"]
        return df


class PlaceEVCorrectionModel:
    """
    複勝用 EV 補正モデル — P/E 分解パターンの Place 版。

    Win 版 (EVCorrectionModel) と同じ構造だが、以下が異なる:
    - P-target: kakuteijyuni <= 3 (複勝的中)
    - P-init_score: logit(p_place_pred)
    - E-target: log(fukuoddslow) - log(e_return_place_pred)
    - E-filter: placed horses only (kakuteijyuni <= 3)
    - E-weight: 1/sqrt(p_place_pred)
    - 出力列: p_place_corrected, e_return_place_corrected, ev_place_corrected
    """

    E_CLIP_FLOOR: float = 1.0

    FEATURE_COLS: list[str] = [
        # 2段階モデルの出力
        "e_return_place_pred",
        # 市場歪み
        "signed_log_error_win",
        "abs_log_error_win",
        "market_entropy",
        "popularity_rank",
        # FLB slope (市場集中度)
        "implied_prob_hhi",
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "field_size",
        # 騎手コンテキスト
        "jockey_wr_overall",
        "jockey_wr_distance",
        "jockey_wr_venue",
        "jockey_prize_log",
        # 調教師コンテキスト
        "trainer_wr_overall",
        "trainer_wr_distance",
        "trainer_wr_venue",
        "trainer_prize_log",
        # 騎手-調教師コンビ
        "jt_combo_wr",
        "jt_combo_place_rate",
        "jt_combo_starts",
        "jt_combo_prize_log",
    ]

    def __init__(self) -> None:
        self.p_correction_model: lgb.Booster | None = None
        self.e_correction_model: lgb.Booster | None = None
        self._trained: bool = False

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """交互作用特徴量を追加"""
        df["p_x_e_interaction_place"] = df["p_place_pred"] * df["e_return_place_pred"]
        df["p_minus_e_gap_place"] = np.abs(
            np.log(df["p_place_pred"] + 1e-8) - np.log(df["e_return_place_pred"] + 1e-8)
        )
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量データフレームを準備する"""
        all_cols = self.FEATURE_COLS + ["p_x_e_interaction_place", "p_minus_e_gap_place"]
        features = df[all_cols].copy()
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin"]:
            if col in features.columns:
                features[col] = features[col].astype("category")
        return features

    def train(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        """P補正モデルとE補正モデルをそれぞれ学習"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        df = df.copy()
        assert "ev_place" in df.columns, (
            "ev_place が必要です。先に PlaceTwoStageModel.predict_ev() を実行してください"
        )

        df = self._add_interaction_features(df)
        features = self._prepare_features(df)

        # ── Model P: P補正 (全サンプル・binary classification) ──
        y_p = (df["kakuteijyuni"] <= 3).astype(int)
        p_pred_clipped = np.clip(df["p_place_pred"], 1e-4, 1 - 1e-4)
        init_score = np.log(p_pred_clipped / (1 - p_pred_clipped))

        # P correction: train/valid split (80/20) with init_score (時系列分割)
        n_p = len(features)
        split_p = int(n_p * 0.8)
        train_idx_p = np.arange(split_p)
        valid_idx_p = np.arange(split_p, n_p)

        train_data_p = lgb.Dataset(
            features.iloc[train_idx_p],
            label=y_p.iloc[train_idx_p],
            init_score=init_score[train_idx_p],
        )
        valid_data_p = lgb.Dataset(
            features.iloc[valid_idx_p],
            label=y_p.iloc[valid_idx_p],
            init_score=init_score[valid_idx_p],
            reference=train_data_p,
        )

        self.p_correction_model = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "is_unbalance": True,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data_p,
            num_boost_round=300,
            valid_sets=[valid_data_p],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        # ── Model E: E補正 (複勝的中馬のみ・1/√p 重み付き回帰) ──
        placed = df[df["kakuteijyuni"] <= 3].copy()
        e_pred_clipped = np.clip(placed["e_return_place_pred"], self.E_CLIP_FLOOR, None)
        placed["log_e_correction"] = np.log(
            placed["fukuoddslow"].clip(lower=self.E_CLIP_FLOOR)
        ) - np.log(e_pred_clipped)
        placed["_e_sample_weight"] = 1.0 / np.sqrt(np.clip(placed["p_place_pred"], 0.01, None))

        features_e = self._prepare_features(placed)
        e_weight = placed["_e_sample_weight"].values

        # E correction: train/valid split (80/20) with weight (時系列分割)
        n_e = len(features_e)
        split_e = int(n_e * 0.8)
        train_idx_e = np.arange(split_e)
        valid_idx_e = np.arange(split_e, n_e)

        train_data_e = lgb.Dataset(
            features_e.iloc[train_idx_e],
            label=placed["log_e_correction"].iloc[train_idx_e],
            weight=e_weight[train_idx_e],
        )
        valid_data_e = lgb.Dataset(
            features_e.iloc[valid_idx_e],
            label=placed["log_e_correction"].iloc[valid_idx_e],
            weight=e_weight[valid_idx_e],
            reference=train_data_e,
        )

        callbacks: list = [lgb.early_stopping(100, verbose=False)]
        if n_e < 10:
            # サンプル数不足で早期停止をスキップ
            callbacks = []

        self.e_correction_model = lgb.train(
            {
                "objective": "regression_l1",
                "metric": "mae",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data_e,
            num_boost_round=300,
            valid_sets=[valid_data_e],
            callbacks=callbacks,
        )

        self._trained = True

    def correct_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全馬の複勝EVをP補正×E補正で補正する。
        P_corrected = sigmoid(logit(P_pred) + correction_logit)  ← [0,1] に制約
        E_corrected = e_return_place_pred × exp(log_e_correction)
        EV_corrected = P_corrected × E_corrected

        未学習時 (_trained=False): ev_place をそのまま ev_place_corrected として出力。
        """
        if not self._trained:
            df = df.copy()
            if "ev_place" not in df.columns:
                df["ev_place"] = df["p_place_pred"] * df["e_return_place_pred"]
            df["ev_place_corrected"] = df["ev_place"]
            return df

        df = df.copy()
        df = self._add_interaction_features(df)
        features = self._prepare_features(df)

        # P補正の適用 (binary出力 → sigmoid で [0,1] に制約)
        p_pred_clipped = np.clip(df["p_place_pred"], 1e-4, 1 - 1e-4)
        init_score = np.log(p_pred_clipped / (1 - p_pred_clipped))
        p_best = (
            self.p_correction_model.best_iteration  # type: ignore[union-attr]
            if self.p_correction_model is not None and self.p_correction_model.best_iteration > 0
            else None
        )
        p_correction_logit = (
            self.p_correction_model.predict(features, num_iteration=p_best)  # type: ignore[union-attr]
            + init_score
        )
        df["p_place_corrected"] = 1.0 / (1.0 + np.exp(-p_correction_logit))

        # E補正の適用
        e_best = (
            self.e_correction_model.best_iteration  # type: ignore[union-attr]
            if self.e_correction_model is not None and self.e_correction_model.best_iteration > 0
            else None
        )
        log_e_corr = self.e_correction_model.predict(features, num_iteration=e_best)  # type: ignore[union-attr]
        df["e_return_place_corrected"] = df["e_return_place_pred"] * np.exp(log_e_corr)

        # 最終補正EV
        df["ev_place_corrected"] = df["p_place_corrected"] * df["e_return_place_corrected"]
        return df
