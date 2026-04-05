"""Stage1 能力モデル -- LightGBM Ranker, オッズ不入力 (Rule 1)"""

from __future__ import annotations

import logging
import os

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
      (p_ability_place は PlaceAbilityModel で別途計算)
    """

    FEATURE_COLS: list[str] = [
        # レース条件 (7)
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",
        "field_size",
        "weight_diff_from_mean",
        "difficulty_score",
        # 過去成績 (8)
        "norm_finish_logit_avg",
        "harontimel3_avg",
        "harontimel3_zscore",
        "timediff_avg",
        "jyuni1c_avg",
        "jyuni4c_avg",
        "closing_index_avg",
        "kyakusitukubun_cd",
        # 血統 (6)
        "blood_surface_wr",
        "blood_distance_wr",
        "blood_condition_wr",
        "blood_total_wr",
        "blood_prize_log",
        "blood_keito_cd",
        # 交互作用 (3)
        "kyakusitu_x_distance",
        "kyakusitu_x_surface",
        "weight_x_distance",
        # レース内正規化 (5) — race_rank
        "norm_finish_logit_avg_race_rank",
        "harontimel3_avg_race_rank",
        "timediff_avg_race_rank",
        "jyuni1c_avg_race_rank",
        "closing_index_avg_race_rank",
        # 馬体 (1)
        "weight_absolute",
    ]

    def __init__(self) -> None:
        self.models: dict[str, lgb.Booster] = {}
        self._submodel_mgr = SubModelManager()

    def train(
        self, df: pd.DataFrame, *, early_stopping: bool = False, num_threads: int = 0
    ) -> None:
        """芝/ダート別に LightGBM Ranker を学習。

        Args:
            df: 学習データ。
            early_stopping: True の場合、80/20 で train/valid を分割し
                            early_stopping(50) を適用する。OOF fold では使用しない。
            num_threads: LightGBM スレッド数。0 の場合は自動計算。
        """
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        params: dict = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "feature_fraction": 0.7,
            "num_threads": num_threads,
            "verbose": -1,
        }

        # DataFrame内に実際に存在するsurfaceのみ処理
        surfaces_in_data = set(df["surface"].unique()) & set(SubModelManager.VALID_KEYS)
        for key in surfaces_in_data:
            key_df = df[df["surface"] == key].copy()
            key_df = key_df.sort_values("race_id")
            features = key_df[self.FEATURE_COLS].copy()
            for col in features.columns:
                if pd.api.types.is_integer_dtype(features[col]):
                    features[col] = features[col].astype(float)
            for col in [
                "surface",
                "distance_bin",
                "grade_code",
                "kyakusitukubun_cd",
                "blood_keito_cd",
                "kyakusitu_x_distance",
                "kyakusitu_x_surface",
            ]:
                if col in features.columns:
                    features[col] = features[col].astype("category")

            # ラベル: 1着=3, 2着=2, 3着=1, 4着以降=0
            y = key_df["kakuteijyuni"].apply(lambda x: max(0, 4 - x) if x > 0 else 0)
            groups = key_df.groupby("race_id").size().values
            n_groups = len(groups)

            if early_stopping and n_groups >= 2:
                # レース単位で train/valid を分割する
                # group 配列: [4, 4, 3] = レース1に4頭, レース2に4頭, レース3に3頭
                race_perm = np.random.RandomState(42).permutation(n_groups)
                race_split = int(n_groups * 0.8)

                # 行レベルのインデックスに変換
                race_ids_per_row = np.repeat(np.arange(n_groups), groups)
                train_race_ids = set(race_perm[:race_split].tolist())
                train_mask = np.array([rid in train_race_ids for rid in race_ids_per_row])
                valid_mask = ~train_mask

                train_idx = np.where(train_mask)[0]
                valid_idx = np.where(valid_mask)[0]

                train_groups = groups[np.sort(race_perm[:race_split])]
                valid_groups = groups[np.sort(race_perm[race_split:])]

                train_data = lgb.Dataset(
                    features.iloc[train_idx],
                    label=y.iloc[train_idx],
                    group=train_groups,
                )
                valid_data = lgb.Dataset(
                    features.iloc[valid_idx],
                    label=y.iloc[valid_idx],
                    group=valid_groups,
                    reference=train_data,
                )

                self.models[key] = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
                )
            else:
                self.models[key] = lgb.train(
                    params,
                    lgb.Dataset(features, label=y, group=groups),
                    num_boost_round=500,
                )
            logger.info(f"SubModel '{key}' 学習完了: {len(key_df)} samples")

    def add_ability_probs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ranker の出力をレース内 softmax で確率に変換して追加する。
        p_ability_win: 単勝的中確率 (softmax)
        """
        df = df.copy()

        for key in SubModelManager.VALID_KEYS:
            if key not in self.models:
                continue
            mask = df["surface"] == key
            if not mask.any():
                continue

            features = df.loc[mask, self.FEATURE_COLS].copy()
            for col in [
                "surface",
                "distance_bin",
                "grade_code",
                "kyakusitukubun_cd",
                "blood_keito_cd",
                "kyakusitu_x_distance",
                "kyakusitu_x_surface",
            ]:
                if col in features.columns:
                    features[col] = features[col].astype("category")

            booster = self.models[key]
            best_iter = booster.best_iteration
            raw_scores = booster.predict(
                features,
                num_iteration=best_iter if best_iter > 0 else None,
            )

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

    def train_oof(
        self, df: pd.DataFrame, n_folds: int = 3, *, num_threads: int = 0
    ) -> pd.DataFrame:
        """K-fold expanding window で OOF p_ability_win を生成する。

        レース日を時系列順にソートし、expanding window で fold を分割する。
        各 fold のテスト期間データに対して OOF (out-of-fold) 予測を行う。
        最後に全データで最終モデルを学習し、推論に備える。
        """
        df = df.copy()
        # race_date が文字列の場合に datetime に変換
        if df["race_date"].dtype == object:
            df["race_date"] = pd.to_datetime(df["race_date"])
        df = df.sort_values("race_date").reset_index(drop=True)
        oof_preds = pd.Series(np.nan, index=df.index, dtype=np.float64)

        dates = sorted(df["race_date"].unique())
        n_dates = len(dates)

        # データ不足時はフォールバック
        if n_dates < n_folds + 1:
            self.train(df, num_threads=num_threads)
            return self.add_ability_probs(df)

        # fold 境界: n_folds+1 個の等分割点
        boundaries = [dates[n_dates * (i + 1) // (n_folds + 1)] for i in range(n_folds)]

        for i in range(n_folds):
            train_end = boundaries[i]
            test_end = boundaries[i + 1] if i + 1 < n_folds else dates[-1] + pd.Timedelta(days=1)

            train_mask = df["race_date"] < train_end
            test_mask = (df["race_date"] >= train_end) & (df["race_date"] < test_end)

            train_df = df.loc[train_mask].copy()
            test_df = df.loc[test_mask].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                continue

            fold_model = AbilityModel()
            fold_model.train(train_df, num_threads=num_threads)
            test_df = fold_model.add_ability_probs(test_df)

            oof_preds.loc[test_mask] = test_df["p_ability_win"].values

        # 最終モデルを全データで学習（推論用、early stopping あり）
        self.train(df, early_stopping=True, num_threads=num_threads)

        # OOF 予測を設定
        df["p_ability_win"] = oof_preds
        return df
