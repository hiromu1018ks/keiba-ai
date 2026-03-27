"""Stage1 能力モデル -- LightGBM Ranker, オッズ不入力 (Rule 1)"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from models.submodel_manager import SubModelManager

logger = logging.getLogger(__name__)


class AbilityModel:
    """
    馬の基本能力を評価するStage1モデル。
    LightGBM Ranker (lambdarank) で芝/ダート別に学習する。
    オッズ特徴量は一切使用しない (Rule 1)。

    出力:
      p_ability_win:  レース内相対確率 (softmax変換)
      p_ability_place: 複勝的中確率 (rankを3着内/それ以外に変換)
    """

    FEATURE_COLS: list[str] = [
        # 既存 (7)
        "surface", "distance_bin", "track_condition_code",
        "grade_code", "field_size",
        "weight_diff_from_mean", "difficulty_score",
        # Phase 1: 馬の過去成績 (3)
        "norm_finish_logit_avg", "jockey_surprise", "haron_time_zscore_avg",
        # Phase 1: レース内z-score (3)
        "norm_finish_logit_avg_race_z",
        "jockey_surprise_race_z",
        "haron_time_zscore_avg_race_z",
        # Phase 1: レース内pct (3)
        "norm_finish_logit_avg_race_pct",
        "jockey_surprise_race_pct",
        "haron_time_zscore_avg_race_pct",
    ]

    def __init__(self) -> None:
        self.models: dict[str, lgb.Booster] = {}
        self._submodel_mgr = SubModelManager()

    def train(self, df: pd.DataFrame) -> None:
        """芝/ダート別に LightGBM Ranker を学習"""
        for key in SubModelManager.VALID_KEYS:
            key_df = df[df["surface"] == key].copy()
            if len(key_df) == 0:
                logger.warning(f"SubModel '{key}' に学習データなし")
                continue

            key_df = key_df.sort_values("race_id")
            features = key_df[self.FEATURE_COLS].copy()
            for col in ["surface", "distance_bin", "grade_code"]:
                if col in features.columns:
                    features[col] = features[col].astype("category")

            # ラベル: 1着=3, 2着=2, 3着=1, 4着以降=0
            y = key_df["finish_pos"].apply(lambda x: max(0, 4 - x) if x > 0 else 0)
            groups = key_df.groupby("race_id").size().values

            self.models[key] = lgb.train(
                {
                    "objective": "lambdarank",
                    "metric": "ndcg",
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "feature_fraction": 0.7,
                    "verbose": -1,
                },
                lgb.Dataset(features, label=y, group=groups),
                num_boost_round=500,
            )
            logger.info(f"SubModel '{key}' 学習完了: {len(key_df)} samples")

    def add_ability_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ranker の出力をレース内 softmax で確率に変換して追加する。
        p_ability_win: 単勝的中確率 (softmax)
        p_ability_place: 複勝的中確率 (3着内ランクの確率)
        """
        df = df.copy()

        for key in SubModelManager.VALID_KEYS:
            if key not in self.models:
                continue
            mask = df["surface"] == key
            if not mask.any():
                continue

            features = df.loc[mask, self.FEATURE_COLS].copy()
            for col in ["surface", "distance_bin", "grade_code"]:
                if col in features.columns:
                    features[col] = features[col].astype("category")

            raw_scores = self.models[key].predict(features)

            # レース内 softmax (log-sum-exp trick で数値安定化) -> p_ability_win
            df.loc[mask, "_raw_score"] = raw_scores
            log_sum_exp = (
                df.loc[mask, "_raw_score"]
                .groupby(df.loc[mask, "race_id"])
                .transform(lambda s: np.log(np.exp(s - s.max()).sum()) + s.max())
            )
            df.loc[mask, "p_ability_win"] = np.exp(df.loc[mask, "_raw_score"] - log_sum_exp)

        df = df.drop(columns=["_raw_score"], errors="ignore")

        return df
