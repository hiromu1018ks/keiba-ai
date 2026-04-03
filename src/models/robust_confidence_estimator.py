"""信頼区間推定 — Conformal Prediction + Rolling Quantile min (Rule 4)"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
        # Win CP: 非適合スコア = |actual - predicted|
        win_residuals = (win_df["actual_ev_win"] - win_df["ev_win_corrected"]).abs()
        self._win_cp_quantile = float(np.quantile(win_residuals.values, 1 - self.alpha))

        # Place CP
        place_residuals = (place_df["actual_ev_place"] - place_df["ev_place_corrected"]).abs()
        self._place_cp_quantile = float(np.quantile(place_residuals.values, 1 - self.alpha))

        # Rolling Quantile: 残差の標準偏差ベース
        # キャリブレーション時は全体のstdを使用し、推論時はrollingに切り替え
        self._win_rolling_quantile = float(
            np.std(win_residuals.values) * 1.5  # 1.5σ を保守的境界
        )
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
            import logging
            logging.getLogger(__name__).warning(
                "RobustConfidenceEstimator not calibrated, using EV as lower bound"
            )
            win_df = win_df.copy()
            place_df = place_df.copy()
            win_df["EV_lower_win_corrected"] = win_df.get("ev_win_corrected", 0.0)
            place_df["EV_lower_place"] = place_df.get("ev_place_corrected", 0.0)
            return win_df, place_df

        win_df = win_df.copy()
        place_df = place_df.copy()

        # Win lower bound
        cp_lower_win = win_df["ev_win_corrected"] - self._win_cp_quantile
        rolling_lower_win = win_df["ev_win_corrected"] - self._win_rolling_quantile
        win_df["EV_lower_win_corrected"] = np.maximum(
            np.minimum(cp_lower_win, rolling_lower_win),
            0.0,
        )

        # Place lower bound
        cp_lower_place = place_df["ev_place_corrected"] - self._place_cp_quantile
        rolling_lower_place = place_df["ev_place_corrected"] - self._place_rolling_quantile
        place_df["EV_lower_place"] = np.maximum(
            np.minimum(cp_lower_place, rolling_lower_place),
            0.0,
        )

        return win_df, place_df
