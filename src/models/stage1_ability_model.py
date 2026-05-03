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
        "harontimel5_avg",
        "harontimel5_zscore",
        "harontime_late_trend",
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
        "harontimel5_avg_race_rank",
        "timediff_avg_race_rank",
        "jyuni1c_avg_race_rank",
        "closing_index_avg_race_rank",
        # 馬体 (3)
        "weight_absolute",
        "weight_zscore",
        "weight_change_zone",
        # 休養期間 (2)
        "days_since_last_race",
        "rest_category",
        # フォームサイクル (3) — B3
        "form_trend",
        "form_consistency",
        "form_peak_flag",
        # 種牡馬産駎 (5)
        "sire_wr",
        "sire_surface_wr",
        "sire_distance_wr",
        "sire_prize_avg",
        "bms_wr",
        # ペース適性 (3)
        "pace_aptitude",
        "front_pace_wr",
        "closing_pace_wr",
        # コース適性 (2)
        "course_wr",
        "course_distance_wr",
        # 追加改善特徴量
        "draw_ratio",
        "class_move",
        "blinker_change",
        "is_nar_transfer",
        "nar_recent_ratio",
        "track_condition_delta",
        "pace_pressure",
        "pace_scenario_fit",
        # TSER-02: クラス調整フォーメトリック
        "class_adj_formetric",
        # TSER-03: z-score改善トラジェクトリ
        "haron_zscore_trend",
        # PACE-01: ペースフィグアサブ特徴量
        "pace_corner_stability",
        "pace_closing_power",
        "pace_position_consistency",
        # PACE-02: 実績ベースのペース適性
        "actual_pace_fit",
    ]

    def __init__(self) -> None:
        self.models: dict[str, lgb.Booster] = {}
        self._submodel_mgr = SubModelManager()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        features = df[available_cols].copy()
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
        return features

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
            features = self._prepare_features(key_df)

            # ラベル: 1着=3, 2着=2, 3着=1, 4着以降=0
            y = key_df["kakuteijyuni"].apply(lambda x: max(0, 4 - x) if x > 0 else 0)
            groups = key_df.groupby("race_id").size().values
            n_groups = len(groups)

            if early_stopping and n_groups >= 2:
                # 時系列分割: groupはrace_id順に既にソートされている前提
                # 前半80%をtrain、後半20%をvalidとする (リーク防止)
                race_split = int(n_groups * 0.8)

                train_groups = groups[:race_split]
                valid_groups = groups[race_split:]

                # 行レベルのインデックスに変換
                race_ids_per_row = np.repeat(np.arange(n_groups), groups)
                train_race_ids = set(range(race_split))
                train_mask = np.array([rid in train_race_ids for rid in race_ids_per_row])
                valid_mask = ~train_mask

                train_idx = np.where(train_mask)[0]
                valid_idx = np.where(valid_mask)[0]

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

            features = self._prepare_features(df.loc[mask])

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
