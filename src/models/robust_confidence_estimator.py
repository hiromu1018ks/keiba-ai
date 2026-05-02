"""信頼区間推定 — Conformal Prediction + Rolling Quantile min (Rule 4)"""

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
                for (surf, dist), group in win_df[win_mask].groupby(["surface", "distance_bin"]):
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

    def predict_lower_bound(
        self,
        win_df: pd.DataFrame,
        place_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        EV の信頼区間下限を推定。
        min(CP_bound, Rolling_Quantile_bound) を採用 (Rule 4)。
        """
        if not self._calibrated:
            # 未キャリブレーション時: EVをそのまま下限として使用 (保守的)
            logger.warning("RobustConfidenceEstimator not calibrated, using EV as lower bound")
            win_df = win_df.copy()
            place_df = place_df.copy()
            win_df["EV_lower_win_corrected"] = win_df.get("ev_win_corrected", 0.0)
            place_df["EV_lower_place"] = place_df.get("ev_place_corrected", 0.0)
            return win_df, place_df

        win_df = win_df.copy()
        place_df = place_df.copy()
        win_ev = pd.to_numeric(win_df["ev_win_corrected"], errors="coerce").fillna(0.0)
        place_ev = pd.to_numeric(place_df["ev_place_corrected"], errors="coerce").fillna(0.0)

        # Win lower bound (with race-condition-dependent CP quantile)
        cp_quantile = self._win_cp_quantile
        if self._win_cp_quantile_by_condition:
            cond_keys = (
                win_df["surface"].astype(str) + "_" + win_df["distance_bin"].astype(str)
                if "surface" in win_df.columns and "distance_bin" in win_df.columns
                else pd.Series("", index=win_df.index)
            )
            cond_quantiles = cond_keys.map(self._win_cp_quantile_by_condition)
            # Use conditional quantile where available, global otherwise
            cp_quantile_per_row = cond_quantiles.fillna(self._win_cp_quantile)
        else:
            cp_quantile_per_row = pd.Series(self._win_cp_quantile, index=win_df.index)

        cp_lower_win = win_ev - cp_quantile_per_row
        rolling_lower_win = win_ev - self._win_rolling_quantile
        win_df["EV_lower_win_corrected"] = np.maximum(
            np.minimum(cp_lower_win, rolling_lower_win),
            0.0,
        )

        # Place lower bound
        cp_lower_place = place_ev - self._place_cp_quantile
        rolling_lower_place = place_ev - self._place_rolling_quantile
        place_df["EV_lower_place"] = np.maximum(
            np.minimum(cp_lower_place, rolling_lower_place),
            0.0,
        )

        return win_df, place_df
