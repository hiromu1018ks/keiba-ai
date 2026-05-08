"""信頼区間推定 -- Conformal Prediction + Rolling Quantile min (Rule 4)"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RobustConfidenceEstimator:
    """
    EV の信頼区間下限を推定する。

    Rule 4: min(Conformal Prediction, Rolling Quantile) を採用。
    より保守的な (低い) 方の下限を使用することで、
    過信を防ぎ安全性を確保する。

    Conformal Prediction: 分布フリーの非適合スコアに基づく信頼区間
    Rolling Quantile: 時系列残差の分位数に基づく信頼区間
    """

    def __init__(self, alpha: float = 0.1, rolling_window: int = 200) -> None:
        """
        Args:
            alpha: 有意水準 (デフォルト 0.1 = 90%信頼区間)
            rolling_window: Rolling Quantile のウィンドウサイズ
        """
        self.alpha = alpha
        self.rolling_window = rolling_window
        self._calibrated = False

        # CP キャリブレーション結果
        self._win_cp_quantile: float = 0.0
        self._place_cp_quantile: float = 0.0
        # Rolling Quantile キャリブレーション結果
        self._win_rolling_quantile: float = 0.0
        self._place_rolling_quantile: float = 0.0
        # Race-condition-dependent CP quantile (SELC-02)
        self._win_cp_quantile_by_condition: dict[str, float] = {}

    def calibrate(
        self,
        win_df: pd.DataFrame,
        place_df: pd.DataFrame,
    ) -> None:
        """
        キャリブレーションデータで非適合スコアを計算。

        Args:
            win_df: ev_win_corrected, actual_ev_win を含む
            place_df: ev_place_corrected, actual_ev_place を含む
        """
        win_pred = pd.to_numeric(win_df["ev_win_corrected"], errors="coerce")
        win_actual = pd.to_numeric(win_df["actual_ev_win"], errors="coerce")
        win_mask = win_pred.notna() & win_actual.notna()

        # Win CP: 非適合スコア = |actual - predicted|
        win_residuals = (win_actual[win_mask] - win_pred[win_mask]).abs()
        if win_residuals.empty:
            logger.warning("No valid win residuals for confidence calibration")
            self._win_cp_quantile = 0.0
            self._win_rolling_quantile = 0.0
        else:
            self._win_cp_quantile = float(np.quantile(win_residuals.values, 1 - self.alpha))
            self._win_rolling_quantile = float(np.std(win_residuals.values) * 1.5)

            # Race-condition-dependent CP quantile (SELC-02)
            if "surface" in win_df.columns and "distance_bin" in win_df.columns:
                for (surf, dist), group in win_df[win_mask].groupby(["surface", "distance_bin"], observed=True):
                    if len(group) >= 30:
                        group_residuals = (win_actual.loc[group.index] - win_pred.loc[group.index]).abs()
                        self._win_cp_quantile_by_condition[f"{surf}_{dist}"] = float(
                            np.quantile(group_residuals.values, 1 - self.alpha)
                        )

        # Place CP
        place_pred = pd.to_numeric(place_df["ev_place_corrected"], errors="coerce")
        place_actual = pd.to_numeric(place_df["actual_ev_place"], errors="coerce")
        place_mask = place_pred.notna() & place_actual.notna()
        place_residuals = (place_actual[place_mask] - place_pred[place_mask]).abs()
        if place_residuals.empty:
            logger.warning("No valid place residuals for confidence calibration")
            self._place_cp_quantile = 0.0
            self._place_rolling_quantile = 0.0
        else:
            self._place_cp_quantile = float(np.quantile(place_residuals.values, 1 - self.alpha))
            # Rolling Quantile: 残差の標準偏差ベース
            # キャリブレーション時は全体のstdを使用し、推論時はrollingに切り替え
            self._place_rolling_quantile = float(np.std(place_residuals.values) * 1.5)

        self._calibrated = True

    def predict_interval(
        self,
        win_df: pd.DataFrame,
        place_df: pd.DataFrame,
        alphas: tuple[float, ...] = (0.1, 0.2),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """EVの信頼区間(上下)を複数水準で推定。min(CP, Rolling_Quantile)を採用 (Rule 4)。

        Args:
            win_df: ev_win_corrected を含むDataFrame
            place_df: ev_place_corrected を含むDataFrame
            alphas: 信頼水準のタプル。0.1=90%区間、0.2=80%区間

        Returns:
            win_df, place_df with EV_lower/upper columns and conformal_confidence_score
        """
        if not self._calibrated:
            logger.warning("RobustConfidenceEstimator not calibrated, using EV as bounds")
            win_df = win_df.copy()
            place_df = place_df.copy()
            win_df["EV_lower_win_corrected"] = pd.to_numeric(
                win_df.get("ev_win_corrected", pd.Series(0.0, index=win_df.index)),
                errors="coerce",
            ).fillna(0.0)
            win_df["EV_upper_win_corrected"] = win_df["EV_lower_win_corrected"]
            win_df["conformal_confidence_score"] = 0.0
            place_df["EV_lower_place"] = pd.to_numeric(
                place_df.get("ev_place_corrected", pd.Series(0.0, index=place_df.index)),
                errors="coerce",
            ).fillna(0.0)
            place_df["EV_upper_place"] = place_df["EV_lower_place"]
            return win_df, place_df

        win_df = win_df.copy()
        place_df = place_df.copy()
        win_ev = pd.to_numeric(win_df["ev_win_corrected"], errors="coerce").fillna(0.0)

        # Race-condition-dependent CP quantile (same as predict_lower_bound)
        cp_quantile_per_row: pd.Series | float
        if self._win_cp_quantile_by_condition:
            cond_keys = (
                win_df["surface"].astype(str) + "_" + win_df["distance_bin"].astype(str)
                if "surface" in win_df.columns and "distance_bin" in win_df.columns
                else pd.Series("", index=win_df.index)
            )
            cond_quantiles = cond_keys.map(self._win_cp_quantile_by_condition)
            cp_quantile_per_row = cond_quantiles.fillna(self._win_cp_quantile)
        else:
            cp_quantile_per_row = self._win_cp_quantile

        # Use first alpha for primary lower/upper bounds
        primary_alpha = alphas[0]

        # Recalculate CP quantile for primary alpha if different from self.alpha
        if abs(primary_alpha - self.alpha) < 1e-9:
            primary_cp_quantile = cp_quantile_per_row
            primary_rolling_quantile = self._win_rolling_quantile
        else:
            # Reuse calibration data: recompute quantile at different alpha
            # Higher alpha -> narrower interval -> smaller quantile
            # scale = sqrt(calibrated_alpha / requested_alpha)
            scale = np.sqrt(self.alpha / primary_alpha)
            logger.warning(
                "Scaling CP quantile by sqrt(%.3f/%.3f) -- Gaussian approximation "
                "may be invalid for non-Gaussian residual distributions",
                self.alpha, primary_alpha,
            )
            primary_cp_quantile = (
                cp_quantile_per_row * scale
                if isinstance(cp_quantile_per_row, pd.Series)
                else float(cp_quantile_per_row) * scale
            )
            primary_rolling_quantile = self._win_rolling_quantile * scale

        # Primary interval (widest, e.g. 90%)
        cp_lower = win_ev - primary_cp_quantile
        rolling_lower = win_ev - primary_rolling_quantile
        lower = np.maximum(np.minimum(cp_lower, rolling_lower), 0.0)

        cp_upper = win_ev + primary_cp_quantile
        rolling_upper = win_ev + primary_rolling_quantile
        upper = np.minimum(cp_upper, rolling_upper)  # conservative: narrower interval

        win_df["EV_lower_win_corrected"] = lower
        win_df["EV_upper_win_corrected"] = upper

        # Use second alpha for confidence scoring (narrower, e.g. 80%)
        if len(alphas) > 1:
            secondary_alpha = alphas[1]
            if abs(secondary_alpha - self.alpha) < 1e-9:
                secondary_cp_quantile = cp_quantile_per_row
            else:
                scale = np.sqrt(self.alpha / secondary_alpha)
                logger.warning(
                    "Scaling CP quantile by sqrt(%.3f/%.3f) -- Gaussian approximation "
                    "may be invalid for non-Gaussian residual distributions",
                    self.alpha, secondary_alpha,
                )
                secondary_cp_quantile = (
                    cp_quantile_per_row * scale
                    if isinstance(cp_quantile_per_row, pd.Series)
                    else float(cp_quantile_per_row) * scale
                )
            secondary_lower = np.maximum(win_ev - secondary_cp_quantile, 0.0)
            win_df["_ev_lower_secondary"] = secondary_lower
        else:
            win_df["_ev_lower_secondary"] = lower

        # conformal_confidence_score per D-06, D-08:
        # Higher score = more confident bet
        # Score = EV_lower_80 * (1 - normalized_width)
        interval_width = (upper - lower).clip(lower=1e-6)
        if "race_id" in win_df.columns:
            max_width = interval_width.groupby(win_df["race_id"], observed=True).transform("max").clip(lower=1e-6)
            normalized_width = (interval_width / max_width).clip(0.0, 1.0)
        else:
            max_width = interval_width.max()
            normalized_width = (interval_width / max(max_width, 1e-6)).clip(0.0, 1.0)

        ev_lower_for_score = win_df["_ev_lower_secondary"]
        win_df["conformal_confidence_score"] = (
            ev_lower_for_score * (1.0 - normalized_width)
        ).fillna(0.0)

        # Clean up temporary column
        win_df.drop(columns=["_ev_lower_secondary"], inplace=True, errors="ignore")

        # Place bounds (simpler: no confidence score needed for place)
        place_ev = pd.to_numeric(place_df["ev_place_corrected"], errors="coerce").fillna(0.0)
        cp_lower_place = place_ev - self._place_cp_quantile
        rolling_lower_place = place_ev - self._place_rolling_quantile
        place_df["EV_lower_place"] = np.maximum(
            np.minimum(cp_lower_place, rolling_lower_place), 0.0
        )
        cp_upper_place = place_ev + self._place_cp_quantile
        rolling_upper_place = place_ev + self._place_rolling_quantile
        place_df["EV_upper_place"] = np.minimum(cp_upper_place, rolling_upper_place)  # conservative

        return win_df, place_df

    def predict_lower_bound(
        self,
        win_df: pd.DataFrame,
        place_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """EVの信頼区間下限を推定。predict_interval()のラッパー (後方互換)。"""
        win_result, place_result = self.predict_interval(win_df, place_df)
        # Remove upper bound and confidence columns for backward compatibility
        win_result = win_result.drop(
            columns=["EV_upper_win_corrected", "conformal_confidence_score"],
            errors="ignore",
        )
        place_result = place_result.drop(
            columns=["EV_upper_place"],
            errors="ignore",
        )
        return win_result, place_result
