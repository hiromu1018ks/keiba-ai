"""複勝能力モデル — LGBMClassifier + Isotonic校正 + 温度スケーリング"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

TEMPERATURE: float = 1.0  # v5: 1.0 (no scaling) — T<1 causes overconfidence


class PlaceAbilityModel:
    """複勝的中確率を直接推定するbinaryモデル。
    p_ability_place = p_ability_win * 3.0 の粗い近似を置き換える。
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
        # Stage1 output
        "p_ability_win",
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
        # v5: レースコンテキスト特徴量
        "race_mean_fuku_odds",
        "race_std_fuku_odds",
        "odds_popularity_gap",
        "surface_track_interaction",
        # 市場構造 (D-06: 市場集中度・歪度)
        "implied_prob_hhi",
        "odds_skewness",
        # 市場クロス整合性 (MCF-07)
        "rl_favorite_in_wide_top1",
        "rl_trio_overlap",
        "rl_market_consistency",
        "rl_trio_odds_ratio",
        "rl_wide_harville_ratio",
    ]

    def __init__(self) -> None:
        self._model: lgb.LGBMClassifier | None = None
        self._calibrated: CalibratedClassifierCV | None = None

    def train(self, df: pd.DataFrame, *, n_jobs: int = 0) -> None:
        """学習 + Isotonic校正（時系列分割）"""
        if n_jobs <= 0:
            n_jobs = max(1, (os.cpu_count() or 4) // 2)
        assert "race_date" in df.columns, "race_date が必要"
        assert "kakuteijyuni" in df.columns, "kakuteijyuni が必要"

        df = df.copy()
        y = (df["kakuteijyuni"] <= 3).astype(int)
        # v5: 新規特徴量はテストデータに存在しない場合があるため、存在する列のみ使用
        available_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[available_cols].copy()  # noqa: N806
        for col in [
            "surface",
            "distance_bin",
            "grade_code",
            "kyakusitukubun_cd",
            "blood_keito_cd",
            "kyakusitu_x_distance",
            "kyakusitu_x_surface",
        ]:
            if col in X.columns:
                X[col] = X[col].astype("category")

        # 時系列分割: 80% train, 20% calibrate
        dates = sorted(df["race_date"].unique())
        split_date = dates[int(len(dates) * 0.8)]
        train_mask = df["race_date"] < split_date
        calib_mask = df["race_date"] >= split_date

        X_train, y_train = X[train_mask], y[train_mask]  # noqa: N806
        X_calib, y_calib = X[calib_mask], y[calib_mask]  # noqa: N806

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)

        self._model = lgb.LGBMClassifier(
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            num_leaves=15,  # v5: 31→15 過学習抑制
            max_depth=-1,
            min_data_in_leaf=200,  # v5: 100→200 過学習抑制
            feature_fraction=0.7,
            reg_lambda=2.0,  # v5: 1.0→2.0 正則化強化
            learning_rate=0.03,
            n_estimators=300,  # v5: 500→300 過学習抑制
            n_jobs=n_jobs,
            verbose=-1,
        )
        self._model.fit(X_train, y_train)

        if len(X_calib) >= 50:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.frozen import FrozenEstimator

            self._calibrated = CalibratedClassifierCV(
                estimator=FrozenEstimator(self._model),
                method="isotonic",
            )
            self._calibrated.fit(X_calib, y_calib)
        else:
            self._calibrated = None
            logger.warning("Insufficient calibration data (%d), skipping isotonic", len(X_calib))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """p_ability_place を設定して df を返す"""
        df = df.copy()
        available_cols = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[available_cols].copy()  # noqa: N806
        for col in [
            "surface",
            "distance_bin",
            "grade_code",
            "kyakusitukubun_cd",
            "blood_keito_cd",
            "kyakusitu_x_distance",
            "kyakusitu_x_surface",
        ]:
            if col in X.columns:
                X[col] = X[col].astype("category")

        if self._calibrated is not None:
            raw_p = self._calibrated.predict_proba(X)[:, 1]
        elif self._model is not None:
            raw_p = self._model.predict_proba(X)[:, 1]
        else:
            # 未学習時: p_ability_place を NaN で設定し、後段モデルがフォールバックする
            logger.warning("PlaceAbilityModel not trained, setting p_ability_place to NaN")
            df["p_ability_place_raw"] = np.nan
            df["p_ability_place"] = np.nan
            return df

        df["p_ability_place_raw"] = raw_p

        # 温度スケーリング
        scaled_series = pd.Series(raw_p ** (1 / TEMPERATURE), index=df.index)

        # レース内正規化: sum(p_place) ≈ 3
        race_sum = scaled_series.groupby(df["race_id"], observed=True).transform("sum")
        df["p_ability_place"] = scaled_series * (3.0 / race_sum.clip(lower=1e-6))

        # 整合性制約: p_place >= p_win
        if "p_ability_win" in df.columns:
            df["p_ability_place"] = np.maximum(df["p_ability_place"], df["p_ability_win"])
            race_sum = df.groupby("race_id", observed=True)["p_ability_place"].transform("sum")
            df["p_ability_place"] = df["p_ability_place"] * (3.0 / race_sum.clip(lower=1e-6))

        # 確率の上限: 1.0 を超えないようクリップ
        df["p_ability_place"] = df["p_ability_place"].clip(upper=1.0)

        return df
