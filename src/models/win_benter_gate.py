"""Win Benter Gate -- 単勝確率に市場確率をブレンドするパイプライン。

Benter (1994) ロジット合成を単勝に適用。
内部で既存BenterCombinationを利用し、Win固有の前処理(tanodds)と
後処理(レース正規化)を統合する (D-11)。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from models.benter_combination import BenterCombination

if TYPE_CHECKING:
    from models.benter_combination import TemperatureScaling
    from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class WinBenterGate:
    """Win-specific Benter combination gate.

    Encapsulates: market probability extraction from tanodds, Benter blending,
    optional calibration, optional temperature scaling, and race normalization.
    """

    def __init__(
        self,
        benter: BenterCombination,
        calibrator: IsotonicRegression | None = None,
        temp_scaler: TemperatureScaling | None = None,
    ) -> None:
        self.benter = benter
        self.calibrator = calibrator
        self.temp_scaler = temp_scaler

    @staticmethod
    def extract_market_probability(tanodds: np.ndarray) -> np.ndarray:
        """Convert tanodds to implied probability with clipping (D-03).

        CRITICAL: Use 'tanodds' column, NOT 'tanoddslow'.
        """
        raw = np.where(np.isfinite(tanodds) & (tanodds > 0), 1.0 / tanodds, np.nan)
        # NaN (zero/negative odds) を中央値0.5で埋めてからクリップ
        result = np.where(np.isnan(raw), 0.5, raw)
        return np.clip(result, 0.01, 0.99)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full Win Benter pipeline to a DataFrame.

        Pipeline: p_win_corrected + tanodds -> Benter combine -> (calibrate)
        -> (temp_scale) -> race normalize -> edge calculation
        """
        df = df.copy()
        p_fund = df["p_win_corrected"].values  # After EV correction (D-01)
        p_market = self.extract_market_probability(df["tanodds"].values)

        # Step 1: Benter combination
        p_combined = self.benter.combine(p_fund, p_market)

        # Step 2: Calibration (Beta or Isotonic) -- set in Plan 02
        if self.calibrator is not None:
            p_combined = self.calibrator.transform(p_combined)

        # Step 3: Temperature scaling (optional, D-06)
        if self.temp_scaler is not None:
            p_combined = self.temp_scaler.transform(p_combined)

        df["p_win_combined"] = p_combined

        # Step 4: Race normalization (D-09, D-10)
        race_sums = df.groupby("race_id")["p_win_combined"].transform("sum")
        df["p_win_final"] = df["p_win_combined"] / race_sums

        # Edge calculation
        df["edge_win"] = df["p_win_final"] * df["tanodds"] - 1.0
        return df


def generate_win_oof_predictions(
    df: pd.DataFrame,
    win_model_cls: type,
    ev_corrector: object,
    n_splits: int = 5,
    num_threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate OOF predictions for Win Benter fitting (D-04).

    Uses KFold CV on training data to generate unbiased predictions.
    Independent of use_ensemble -- best practice.

    Args:
        df: Training DataFrame sorted by race_date.
        win_model_cls: WinTwoStageModel class (imported lazily to avoid circular deps).
        ev_corrector: EVCorrectionModel with trained correct_ev() method.
        n_splits: Number of CV folds.
        num_threads: LightGBM thread count.

    Returns:
        (oof_p_fund, oof_p_market, oof_y) -- aligned arrays for Benter.fit()
    """
    df = df.sort_values("race_date").reset_index(drop=True)
    kfold = KFold(n_splits=n_splits, shuffle=False)  # Time-series: no shuffle
    oof_preds = np.full(len(df), np.nan)

    for train_idx, val_idx in kfold.split(df):
        fold_model = win_model_cls()
        fold_train = df.iloc[train_idx]
        fold_model.train_hit_model(fold_train, num_threads=num_threads)
        fold_val = df.iloc[val_idx].copy()
        fold_val = fold_model.predict_ev(fold_val)
        oof_preds[val_idx] = fold_val["p_win_pred"].values

    # Apply EV correction to OOF predictions for consistency with D-01
    df = df.copy()
    df["p_win_oof"] = oof_preds
    df = ev_corrector.correct_ev(df)

    p_fund = df["p_win_corrected"].values
    p_market = np.clip(
        np.where(df["tanodds"] > 0, 1.0 / df["tanodds"].values, np.nan),
        0.01,
        0.99,
    )
    y = (df["kakuteijyuni"] == 1).astype(int).values

    # Drop NaN entries
    valid = ~(np.isnan(p_fund) | np.isnan(p_market))
    logger.info(
        "Win OOF: %d valid / %d total samples", int(valid.sum()), len(valid)
    )
    return p_fund[valid], p_market[valid], y[valid]
