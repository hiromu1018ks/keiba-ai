"""CQR (Conformalized Quantile Regression) によるEV予測区間推定.

Romano et al., 2019 "Conformalized Quantile Regression" に基づく。
LightGBM quantile regression で alpha/2 と 1-alpha/2 の分位点を学習し、
CQR非適合スコアでCP補正を適用する。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from domain.types import POST_RACE_COLS

logger = logging.getLogger(__name__)

# モデル出力を含めると actual_ev_win との過学習により Q_90=0 になる
_MODEL_OUTPUT_COLS: set[str] = {
    # MarketModel
    "signed_log_error_win", "abs_log_error_win",
    "market_log_error_win", "market_pred_error_win",
    "market_error_rank_in_race",
    # AbilityModel (Stage1) + derived
    "p_ability_win", "odds_to_ability_ratio",
    "deviation_rank", "deviation_zscore",
    # PlaceAbilityModel
    "p_ability_place_raw", "p_ability_place",
    # WinTwoStageModel
    "p_win_pred", "e_return_win_pred", "ev_win",
    # EVCorrectionModel (win)
    "p_x_e_interaction", "p_minus_e_gap",
    "p_win_corrected", "e_return_win_corrected",
    "ev_win_corrected", "ev_win_calibrated",
    # PlaceTwoStageModel
    "p_place_pred", "e_return_place_pred", "ev_place",
    # PlaceEVCorrectionModel
    "p_x_e_interaction_place", "p_minus_e_gap_place",
    "p_place_corrected", "place_bucket_multiplier",
    "e_return_place_corrected", "ev_place_corrected",
    # ConformalEVModel (CQR itself)
    "EV_lower_win_corrected", "EV_upper_win_corrected",
    "conformal_confidence_score",
    "EV_lower_place", "EV_upper_place",
    # SelectionGates
    "win_selection_ev", "win_selection_edge", "win_selection_prob",
    "place_selection_ev", "place_selection_edge", "place_selection_prob",
    # BenterGate
    "p_win_combined", "p_win_final", "edge_win", "p_win_oof",
    # CQR target
    "actual_ev_win", "actual_ev_place",
}
# DEPRECATED: _NON_FEATURE_COLS is kept for reference only.
# SAFE-01: Use ConformalEVModel.FEATURE_COLS whitelist instead of blacklist exclusion.
_NON_FEATURE_COLS: set[str] = {
    # IDs / metadata
    "race_id",
    "umaban",
    "race_date",
    "surface",
    "kettonum",
    # Object dtype (LightGBM cannot handle)
    "distance_bin", "grade_code", "track_condition_code",
} | set(POST_RACE_COLS) | _MODEL_OUTPUT_COLS


class ConformalEVModel:
    """CQR (Conformalized Quantile Regression) によるEV予測区間推定.

    Romano et al., 2019 "Conformalized Quantile Regression" に基づく。
    LightGBM quantile regression で分位点を学習し、CQR非適合スコアでCP補正する。
    """

    # ★ SAFE-01: Whitelist of feature columns for CQR training/prediction.
    # Derived from the union of raw feature columns used by upstream models.
    # POST_RACE_COLS and model output columns are explicitly excluded.
    FEATURE_COLS: list[str] = [
        # --- AbilityModel raw features ---
        "surface", "distance_bin", "track_condition_code", "grade_code",
        "field_size", "weight_diff_from_mean", "difficulty_score",
        "norm_finish_logit_avg", "harontimel5_avg", "harontimel5_zscore",
        "harontime_late_trend", "timediff_avg", "jyuni1c_avg", "jyuni4c_avg",
        "closing_index_avg", "kyakusitukubun_cd",
        "blood_surface_wr", "blood_distance_wr", "blood_condition_wr",
        "blood_total_wr", "blood_prize_log", "blood_keito_cd",
        "kyakusitu_x_distance", "kyakusitu_x_surface", "weight_x_distance",
        "norm_finish_logit_avg_race_rank", "harontimel5_avg_race_rank",
        "timediff_avg_race_rank", "jyuni1c_avg_race_rank",
        "closing_index_avg_race_rank",
        "weight_absolute", "weight_zscore", "weight_change_zone",
        "days_since_last_race", "rest_category",
        "form_trend", "form_consistency", "form_peak_flag",
        "sire_wr", "sire_surface_wr", "sire_distance_wr",
        "sire_prize_avg", "bms_wr",
        "pace_aptitude", "front_pace_wr", "closing_pace_wr",
        "course_wr", "course_distance_wr",
        "draw_ratio", "class_move", "blinker_change",
        "is_nar_transfer", "nar_recent_ratio",
        "track_condition_delta", "pace_pressure", "pace_scenario_fit",
        "class_adj_formetric", "haron_zscore_trend",
        "pace_corner_stability", "pace_closing_power", "pace_position_consistency",
        "actual_pace_fit",
        "class_promotions", "class_demotions", "class_net_change",
        "class_max_level", "class_level_std",
        "v_recovery_flag", "v_recovery_duration",
        "time_improvement_rate", "position_improvement_rate",
        "dist_change_avg_pos", "dist_change_win_rate", "dist_change_exp_count",
        "surf_change_avg_pos", "surf_change_win_rate", "surf_change_exp_count",
        "cond_change_avg_pos", "cond_change_win_rate", "cond_change_exp_count",
        "frame_number", "blinker_on", "weight_change_ratio",
        "popularity_rank", "popularity_rank_fallback_used",
        # --- Odds dynamics features ---
        "odds_drop_rate_60_10", "odds_drop_rate_30_10",
        "odds_velocity", "odds_volatility",
        "popularity_change_30_10",
        "odds_acceleration", "odds_direction_consistency",
        # --- Market bias / intra-race features ---
        "market_entropy", "overround",
        "odds_skewness", "implied_prob_hhi",
        "tanodds", "fukuoddslow", "tanninki",
        "odds", "race_mean_fuku_odds", "race_std_fuku_odds",
        "odds_gap_fav12", "odds_popularity_gap",
        "surface_track_interaction",
        # --- EMA / ROI EMA features ---
        "overround_ema", "entropy_ema",
        # --- Distance change / surface change / win dominance ---
        "distance_change", "surface_change",
        "class_drop_bounce", "freshness_score",
        # --- Jockey context ---
        "jockey_wr_overall", "jockey_wr_distance",
        "jockey_wr_venue", "jockey_prize_log",
        # --- Trainer context ---
        "trainer_wr_overall", "trainer_wr_distance",
        "trainer_wr_venue", "trainer_prize_log",
        # --- JT combo ---
        "jt_combo_wr", "jt_combo_place_rate",
        "jt_combo_starts", "jt_combo_prize_log",
        # --- Wide odds features ---
        "odds_to_ability_ratio",
        "deviation_rank", "deviation_zscore",
        # --- Bataijyu ---
        "bataijyu", "zogen_sa",
        # --- Interaction features ---
        "kyori", "trackcd",
        # --- 市場クロス整合性 (MCF-07) ---
        "rl_favorite_in_wide_top1",
        "rl_trio_overlap",
        "rl_market_consistency",
        "rl_trio_odds_ratio",
        "rl_wide_harville_ratio",
        # レースレベル集約 (RLF-01~06)
        "rl_log_odds_entropy",
        "rl_odds_dispersion",
        "rl_top3_odds_gap",
        "rl_top1_odds",
        "rl_favorite_rank_gap",
        "rl_n_horses",
        # TRF-01/02/03 + INT-01/02/03/04: Phase 36
        "form_trend_race_rank",
        "blood_total_wr_race_rank",
        "blood_surface_wr_race_rank",
        "weighted_recent_form_finish",
        "weighted_recent_form_time",
        "grade_x_form_trend",
        "distance_x_closing_index",
        "grade_x_blood_prize_log",
        # HLF-01/02/03: Phase 36 HaronTime L4 + LapTime pace features
        "closing_speed_ratio_avg",
        "closing_speed_ratio_zscore",
        "closing_speed_ratio_trend",
        "harontime_last3f_avg",
        "harontime_last3f_zscore",
        "harontime_last3f_trend",
        # D-02: haron_race_gap (Phase 36.1 new)
        "haron_race_gap_avg",
        "haron_race_gap_zscore",
        "haron_race_gap_trend",
        # D-03: pace_adj_finish (Phase 36.1 new)
        "pace_ratio_avg",
        "pace_early_avg",
        "pace_mid_avg",
        "pace_late_avg",
        # HLF-02: HaronTime race-rank
        "closing_speed_ratio_avg_race_rank",
        "harontime_last3f_avg_race_rank",
    ]

    def __init__(self, alpha: float = 0.1, feature_cols: list[str] | None = None) -> None:
        """Args:
            alpha: 有意水準 (デフォルト 0.1 = 90%信頼区間)
            feature_cols: 特徴量列名。Noneの場合はtrain()で自動抽出。
        """
        self.alpha = alpha
        self.feature_cols = feature_cols
        self._calibrated = False
        # LightGBM quantile models
        self.q_low_model: lgb.Booster | None = None
        self.q_high_model: lgb.Booster | None = None
        # CQR calibration quantiles (2-alpha構成)
        self._calibration_quantile_90: float = 0.0  # alpha=0.1用
        self._calibration_quantile_80: float = 0.0  # alpha=0.2用
        # Center residual floors. Pure CQR scores can be exactly zero when the
        # quantile models already cover a smoothed target; horse-race EV still
        # needs irreducible uncertainty for lower-bound ranking.
        self._residual_quantile_90: float = 0.0
        self._residual_quantile_80: float = 0.0

    def calibrate(
        self,
        win_df: pd.DataFrame,
        place_df: pd.DataFrame,
    ) -> None:
        """Backward compat: RobustConfidenceEstimator.calibrate() の互換shim.

        Phase 21 Plan 02で完全なCQR学習に置き換えられるまで、
        古いパイプラインコードが動作するよう残存させる。
        """
        logger.warning(
            "ConformalEVModel.calibrate() is a backward-compat shim. "
            "Full CQR training via train() will be integrated in Plan 02."
        )
        self._calibrated = True

    def predict_lower_bound(
        self,
        win_df: pd.DataFrame,
        place_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Backward compat: predict_interval()のラッパー (RobustConfidenceEstimator互換)."""
        win_result, place_result = self.predict_interval(win_df, place_df)
        win_result = win_result.drop(
            columns=["EV_upper_win_corrected", "conformal_confidence_score"],
            errors="ignore",
        )
        place_result = place_result.drop(
            columns=["EV_upper_place"],
            errors="ignore",
        )
        return win_result, place_result

    def train(
        self,
        df_calib: pd.DataFrame,
        *,
        num_threads: int = 0,
        lgb_params: dict | None = None,
        train_ratio: float = 0.8,
    ) -> None:
        """CQRモデルを学習・キャリブレーション.

        Args:
            df_calib: ev_win_calibrated, actual_ev_win, 特徴量列を含むDataFrame
            num_threads: LightGBMスレッド数
            lgb_params: LightGBM追加パラメータ
            train_ratio: 学習/キャリブレーション分割比。1.0の場合は全データを学習+キャリブレーションに使用
        """
        # ★ SAFE-01: Whitelist-based feature selection (blacklist _NON_FEATURE_COLS deprecated)
        if self.feature_cols is None:
            self.feature_cols = [
                c for c in self.FEATURE_COLS
                if c in df_calib.columns and pd.api.types.is_numeric_dtype(df_calib[c])
            ]

        # train_ratio=1.0の場合は全データを学習に使用し、同じデータでキャリブレーション
        if train_ratio >= 1.0:
            logger.warning(
                "train_ratio=%.1f: using same data for training and calibration. "
                "CQR intervals will be overfitted and may not achieve target coverage.",
                train_ratio,
            )
            df_train = df_calib
            df_val = df_calib
        else:
            n_total = len(df_calib)
            split_idx = int(n_total * train_ratio)
            df_train = df_calib.iloc[:split_idx]
            df_val = df_calib.iloc[split_idx:]

        # ターゲット抽出
        if "actual_ev_win" in df_train.columns:
            target_col = "actual_ev_win"
        else:
            target_col = "ev_win_calibrated"

        X_train = df_train[self.feature_cols]
        y_train = pd.to_numeric(df_train[target_col], errors="coerce")
        mask = y_train.notna()
        X_train = X_train[mask]
        y_train = y_train[mask]

        if len(y_train) < 200:
            logger.warning(
                "Insufficient samples for CQR training (%d < 200). "
                "Skipping CQR model training.",
                len(y_train),
            )
            return

        if y_train.std() < 1e-6:
            logger.warning(
                "Target variance too low for CQR training (std=%.8f). "
                "Skipping — would produce degenerate zero-width intervals.",
                y_train.std(),
            )
            return

        # LightGBM データセット
        train_set = lgb.Dataset(X_train, label=y_train.values)

        # デフォルトパラメータ
        default_params: dict = {
            "objective": "quantile",
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "verbose": -1,
            "num_threads": num_threads,
        }

        # q_low_model: alpha/2分位点 (例: alpha=0.1 -> quantile=0.05)
        params_low = {**default_params, "alpha": self.alpha / 2}
        if lgb_params:
            params_low.update(lgb_params)
        self.q_low_model = lgb.train(
            params_low,
            train_set,
            num_boost_round=200,
            valid_sets=[train_set],
            callbacks=[lgb.log_evaluation(0)],
        )

        # q_high_model: 1-alpha/2分位点 (例: alpha=0.1 -> quantile=0.95)
        params_high = {**default_params, "alpha": 1 - self.alpha / 2}
        if lgb_params:
            params_high.update(lgb_params)
        self.q_high_model = lgb.train(
            params_high,
            train_set,
            num_boost_round=200,
            valid_sets=[train_set],
            callbacks=[lgb.log_evaluation(0)],
        )

        # キャリブレーションセットで非適合スコア計算
        X_val = df_val[self.feature_cols]
        y_val = pd.to_numeric(df_val[target_col], errors="coerce")
        val_mask = y_val.notna()
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]

        if len(y_val) < 10:
            logger.warning(
                "Insufficient calibration samples (%d < 10). "
                "Skipping CQR calibration.",
                len(y_val),
            )
            return

        q_low_pred = self.q_low_model.predict(X_val)
        q_high_pred = self.q_high_model.predict(X_val)

        # CQR非適合スコア: max(q_low - y, y - q_high)
        # 標準CQRでは負値も許容されるが、ここではEV下限を選別シグナルにも使うため
        # 0未満を切り上げ、さらに中心残差から不確実性の下限を付与する。
        nonconformity_scores = np.maximum(
            np.maximum(q_low_pred - y_val.values, y_val.values - q_high_pred),
            0.0,
        )
        center_pred = 0.5 * (q_low_pred + q_high_pred)
        center_residual = np.abs(y_val.values - center_pred)

        # 有限サンプル補正付き補正量子 (Romano et al., 2019)
        n = len(nonconformity_scores)
        # 90%区間用 (alpha=0.1)
        q_90_level = min((1 - 0.1) * (1 + 1 / n), 1.0)
        q_90 = float(np.quantile(nonconformity_scores, q_90_level))
        # 80%区間用 (alpha=0.2)
        q_80_level = min((1 - 0.2) * (1 + 1 / n), 1.0)
        q_80 = float(np.quantile(nonconformity_scores, q_80_level))

        target_scale = max(float(np.nanmedian(np.abs(y_val.values))), 1e-6)
        residual_90 = float(np.quantile(center_residual, q_90_level))
        residual_80 = float(np.quantile(center_residual, q_80_level))
        floor_90 = max(residual_90 * 0.10, target_scale * 0.01)
        floor_80 = max(residual_80 * 0.10, target_scale * 0.01)
        self._residual_quantile_90 = floor_90
        self._residual_quantile_80 = floor_80
        self._calibration_quantile_90 = max(q_90, floor_90)
        self._calibration_quantile_80 = max(q_80, floor_80)

        self._calibrated = True
        logger.info(
            "CQR calibrated: Q_90=%.4f, Q_80=%.4f, residual_floor_90=%.4f, "
            "residual_floor_80=%.4f, n_calib=%d",
            self._calibration_quantile_90,
            self._calibration_quantile_80,
            self._residual_quantile_90,
            self._residual_quantile_80,
            n,
        )

    def predict_interval(
        self,
        win_df: pd.DataFrame,
        place_df: pd.DataFrame,
        alphas: tuple[float, ...] = (0.1, 0.2),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """EVの信頼区間(上下)を複数水準で推定.

        RobustConfidenceEstimator.predict_interval()と同じシグネチャ・出力互換。

        Note: CQR is applied to win EV only. Place intervals are identity
        pass-through (EV_lower_place = EV_upper_place = place_ev), providing
        zero-width intervals with no uncertainty quantification.
        TODO: implement CQR for place EV if needed.

        Args:
            win_df: ev_win_calibrated と特徴量列を含むDataFrame
            place_df: ev_place_corrected を含むDataFrame
            alphas: 信頼水準のタプル。0.1=90%区間、0.2=80%区間

        Returns:
            (win_df, place_df) with EV_lower/upper columns and conformal_confidence_score
        """
        win_df = win_df.copy()
        place_df = place_df.copy()

        # 未キャリブレーション時のフォールバック
        if not self._calibrated or self.q_low_model is None or self.q_high_model is None:
            logger.warning("ConformalEVModel not calibrated, using fallback values")
            if "ev_win_calibrated" in win_df.columns:
                win_ev = pd.to_numeric(win_df["ev_win_calibrated"], errors="coerce").fillna(0.0)
            elif "ev_win_corrected" in win_df.columns:
                win_ev = pd.to_numeric(win_df["ev_win_corrected"], errors="coerce").fillna(0.0)
            else:
                win_ev = pd.Series(0.0, index=win_df.index)
            win_df["EV_lower_win_corrected"] = win_ev
            win_df["EV_upper_win_corrected"] = win_ev
            win_df["conformal_confidence_score"] = 0.0

            if "ev_place_corrected" in place_df.columns:
                place_ev = pd.to_numeric(
                    place_df["ev_place_corrected"], errors="coerce"
                ).fillna(0.0)
            else:
                place_ev = pd.Series(0.0, index=place_df.index)
            place_df["EV_lower_place"] = place_ev
            place_df["EV_upper_place"] = place_ev
            return win_df, place_df

        # ベースEV取得
        if "ev_win_calibrated" in win_df.columns:
            win_ev = pd.to_numeric(win_df["ev_win_calibrated"], errors="coerce").fillna(0.0)
        elif "ev_win_corrected" in win_df.columns:
            win_ev = pd.to_numeric(win_df["ev_win_corrected"], errors="coerce").fillna(0.0)
        else:
            win_ev = pd.Series(0.0, index=win_df.index)

        # CQR予測
        if self.feature_cols is None:
            # ★ SAFE-01: Whitelist-based fallback (matches train() logic)
            feature_cols = [
                c for c in self.FEATURE_COLS
                if c in win_df.columns and pd.api.types.is_numeric_dtype(win_df[c])
            ]
        else:
            feature_cols = self.feature_cols

        # 推論コンテキストで一部特徴量が欠落する場合（キャリブレーションBT等）、
        # 利用可能な列のみを使用し欠落列は0で埋める
        available = set(win_df.columns)
        missing = [c for c in feature_cols if c not in available]
        if missing:
            logger.warning(
                "ConformalEV: %d/%d feature_cols missing, filling with 0: %s",
                len(missing), len(feature_cols), missing[:5],
            )
            missing_df = pd.DataFrame(0.0, index=win_df.index, columns=missing)
            win_df = pd.concat([win_df, missing_df], axis=1)

        X_win = win_df[feature_cols].copy()
        # object型列を数値に変換（学習時と推論時でdtypeが異なる場合）
        for c in X_win.columns:
            if X_win[c].dtype == object:
                X_win[c] = pd.to_numeric(X_win[c], errors="coerce").fillna(0.0)
        X_win = X_win.fillna(0.0)
        q_low = self.q_low_model.predict(X_win)
        q_high = self.q_high_model.predict(X_win)

        # Monotonicity guarantee: q_low <= q_high
        q_low = np.minimum(q_low, q_high)

        # 90%区間 (第一alpha, 通常alpha=0.1)
        lower_90 = np.maximum(q_low - self._calibration_quantile_90, 0.0)
        upper_90 = q_high + self._calibration_quantile_90
        if self._residual_quantile_90 > 0:
            base_lower_90 = np.maximum(
                win_ev.to_numpy(dtype=float) - self._residual_quantile_90,
                0.0,
            )
            lower_90 = (0.70 * lower_90) + (0.30 * base_lower_90)
            lower_90 = np.minimum(lower_90, upper_90)

        # 出力
        win_df["EV_lower_win_corrected"] = lower_90
        win_df["EV_upper_win_corrected"] = upper_90

        # 80%区間 (第二alpha, 通常alpha=0.2) - confidence_score用
        if len(alphas) > 1:
            lower_80 = np.maximum(q_low - self._calibration_quantile_80, 0.0)
            if self._residual_quantile_80 > 0:
                base_lower_80 = np.maximum(
                    win_ev.to_numpy(dtype=float) - self._residual_quantile_80,
                    0.0,
                )
                lower_80 = (0.70 * lower_80) + (0.30 * base_lower_80)
                lower_80 = np.minimum(lower_80, upper_90)
        else:
            lower_80 = lower_90

        # conformal_confidence_score: lower_80 * (1 - normalized_width)
        interval_width = pd.Series(upper_90 - lower_90, index=win_df.index).clip(lower=1e-6)
        if "race_id" in win_df.columns:
            max_width = (
                interval_width.groupby(win_df["race_id"], observed=True)
                .transform("max")
                .clip(lower=1e-6)
            )
            normalized_width = (interval_width / max_width).clip(0.0, 1.0)
        else:
            max_width = max(float(interval_width.max()), 1e-6)
            normalized_width = (interval_width / max_width).clip(0.0, 1.0)

        win_df["conformal_confidence_score"] = (lower_80 * (1.0 - normalized_width)).fillna(0.0)

        # Place側: 簡易処理 (CQR適用なし)
        if "ev_place_corrected" in place_df.columns:
            place_ev = pd.to_numeric(place_df["ev_place_corrected"], errors="coerce").fillna(0.0)
        else:
            place_ev = pd.Series(0.0, index=place_df.index)
        place_df["EV_lower_place"] = place_ev
        place_df["EV_upper_place"] = place_ev

        return win_df, place_df

    def save(self, models_dir: Path, surface: str) -> None:
        """CQRモデルをディスクに保存.

        Args:
            models_dir: 保存先ディレクトリ
            surface: サーフェス名 (turf/dirt)
        """
        if not self._calibrated or self.q_low_model is None or self.q_high_model is None:
            logger.warning("Cannot save uncalibrated ConformalEVModel for %s", surface)
            return

        models_dir.mkdir(parents=True, exist_ok=True)

        # LightGBMモデル保存
        low_path = models_dir / f"cqr_quantile_low_{surface}.lgb"
        high_path = models_dir / f"cqr_quantile_high_{surface}.lgb"
        self.q_low_model.save_model(str(low_path))
        self.q_high_model.save_model(str(high_path))

        # キャリブレーションパラメータ保存
        params_path = models_dir / f"cqr_params_{surface}.json"
        params = {
            "alpha": self.alpha,
            "calibration_quantile_90": self._calibration_quantile_90,
            "calibration_quantile_80": self._calibration_quantile_80,
            "residual_quantile_90": self._residual_quantile_90,
            "residual_quantile_80": self._residual_quantile_80,
            "feature_cols": self.feature_cols,
            "_calibrated": self._calibrated,
        }
        params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info("Saved ConformalEVModel for %s to %s", surface, models_dir)

    @classmethod
    def load(cls, models_dir: Path, surface: str) -> ConformalEVModel | None:
        """ディスクからCQRモデルを読み込み.

        Args:
            models_dir: モデルディレクトリ
            surface: サーフェス名 (turf/dirt)

        Returns:
            ConformalEVModel or None (ファイルが揃っていない場合)
        """
        low_path = models_dir / f"cqr_quantile_low_{surface}.lgb"
        high_path = models_dir / f"cqr_quantile_high_{surface}.lgb"
        params_path = models_dir / f"cqr_params_{surface}.json"

        if not (low_path.is_file() and high_path.is_file() and params_path.is_file()):
            return None

        model = cls()
        model.q_low_model = lgb.Booster(model_file=str(low_path))
        model.q_high_model = lgb.Booster(model_file=str(high_path))

        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)
        model.alpha = params["alpha"]
        model._calibration_quantile_90 = params["calibration_quantile_90"]
        model._calibration_quantile_80 = params["calibration_quantile_80"]
        model._residual_quantile_90 = params.get("residual_quantile_90", 0.0)
        model._residual_quantile_80 = params.get("residual_quantile_80", 0.0)
        model.feature_cols = params.get("feature_cols")
        model._calibrated = params.get("_calibrated", True)

        return model
