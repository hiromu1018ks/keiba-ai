"""EV補正モデル -- P/E分解で独立性破綻を解決 (C-5)"""

from __future__ import annotations

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
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "field_size",
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
        for col in ["surface", "distance_bin"]:
            if col in features.columns:
                features[col] = features[col].astype("category")
        return features

    def train(self, df: pd.DataFrame) -> None:
        """P補正モデルとE補正モデルをそれぞれ学習"""
        df = df.copy()
        assert "ev_win" in df.columns, (
            "ev_win が必要です。先に WinTwoStageModel.predict_ev() を実行してください"
        )

        df = self._add_interaction_features(df)
        features = self._prepare_features(df)

        # ── Model P: P補正 (全サンプル・binary classification) ──
        y_p = (df["finish_pos"] == 1).astype(int)
        p_pred_clipped = np.clip(df["p_win_pred"], 1e-4, 1 - 1e-4)
        init_score = np.log(p_pred_clipped / (1 - p_pred_clipped))

        self.p_correction_model = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "is_unbalance": True,
                "feature_fraction": 0.7,
                "verbose": -1,
            },
            lgb.Dataset(features, label=y_p, init_score=init_score),
            num_boost_round=300,
        )

        # ── Model E: E補正 (1着馬のみ・1/√p 重み付き回帰) ──
        winners = df[df["finish_pos"] == 1].copy()
        e_pred_clipped = np.clip(winners["e_return_win_pred"], self.E_CLIP_FLOOR, None)
        winners["log_e_correction"] = np.log(
            winners["win_odds_actual"].clip(lower=self.E_CLIP_FLOOR)
        ) - np.log(e_pred_clipped)
        winners["_e_sample_weight"] = 1.0 / np.sqrt(np.clip(winners["p_win_pred"], 0.01, None))

        features_e = self._prepare_features(winners)

        self.e_correction_model = lgb.train(
            {
                "objective": "regression_l1",
                "metric": "mae",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "feature_fraction": 0.7,
                "verbose": -1,
            },
            lgb.Dataset(
                features_e,
                label=winners["log_e_correction"],
                weight=winners["_e_sample_weight"].values,
            ),
            num_boost_round=300,
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
        p_correction_logit = self.p_correction_model.predict(features) + init_score
        df["p_win_corrected"] = 1.0 / (1.0 + np.exp(-p_correction_logit))

        # E補正の適用
        log_e_corr = self.e_correction_model.predict(features)
        df["e_return_win_corrected"] = df["e_return_win_pred"] * np.exp(log_e_corr)

        # 最終補正EV
        df["ev_win_corrected"] = df["p_win_corrected"] * df["e_return_win_corrected"]
        return df
