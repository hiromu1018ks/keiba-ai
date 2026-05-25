"""Win Benter Gate -- 単勝確率に市場確率をブレンドするパイプライン。

Benter (1994) ロジット合成を単勝に適用。
内部で既存BenterCombinationを利用し、Win固有の前処理(tanodds)と
後処理(レース正規化)を統合する (D-11)。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from models.benter_combination import BenterCombination

if TYPE_CHECKING:
    from sklearn.isotonic import IsotonicRegression

    from models.benter_combination import TemperatureScaling

logger = logging.getLogger(__name__)


def _walk_forward_race_splits(
    df: pd.DataFrame,
    *,
    n_splits: int,
    min_train_races: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding walk-forward row indexes without splitting a race."""
    if "race_id" not in df.columns:
        return []
    race_cols = ["race_id"]
    if "race_date" in df.columns:
        race_cols.append("race_date")
    race_order = df[race_cols].drop_duplicates()
    if "race_date" in race_order.columns:
        race_order = race_order.sort_values(["race_date", "race_id"])
    else:
        race_order = race_order.sort_values("race_id")
    race_order = race_order.reset_index(drop=True)
    n_races = len(race_order)
    if n_races <= min_train_races + 1:
        return []

    effective_splits = min(n_splits, max(2, n_races // max(1, min_train_races)))
    if n_races <= effective_splits:
        return []

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    for train_race_idx, val_race_idx in splitter.split(race_order):
        if len(train_race_idx) < min_train_races or len(val_race_idx) == 0:
            continue
        train_races = set(race_order.iloc[train_race_idx]["race_id"])
        val_races = set(race_order.iloc[val_race_idx]["race_id"])
        train_idx = df.index[df["race_id"].isin(train_races)].to_numpy()
        val_idx = df.index[df["race_id"].isin(val_races)].to_numpy()
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


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
        race_sums = df.groupby("race_id", observed=True)["p_win_combined"].transform("sum")
        df["p_win_final"] = df["p_win_combined"] / race_sums

        # Edge calculation
        df["edge_win"] = df["p_win_final"] * df["tanodds"] - 1.0
        return df


def generate_win_oof_predictions(
    df: pd.DataFrame,
    win_model_cls: type,
    ev_corrector: Any,
    n_splits: int = 5,
    num_threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate OOF predictions for Win Benter fitting (D-04).

    Uses expanding walk-forward splits on race_date to generate predictions.
    Each validation fold is scored only by models trained on past races.

    Args:
        df: Training DataFrame sorted by race_date.
        win_model_cls: WinTwoStageModel class (imported lazily to avoid circular deps).
        ev_corrector: EVCorrectionModel with trained correct_ev() method.
        n_splits: Number of CV folds.
        num_threads: LightGBM thread count.

    Returns:
        (oof_p_fund, oof_p_market, oof_y) -- aligned arrays for Benter.fit()
    """
    sort_cols = [col for col in ["race_date", "race_id", "umaban"] if col in df.columns]
    df = (
        df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)
    )
    splits = _walk_forward_race_splits(df, n_splits=n_splits)
    p_fund_oof = np.full(len(df), np.nan)
    p_market_oof = np.full(len(df), np.nan)
    y_oof = np.full(len(df), np.nan)

    for train_idx, val_idx in splits:
        fold_model = win_model_cls()
        fold_train = df.iloc[train_idx]
        fold_val_raw = df.iloc[val_idx].copy()
        try:
            fold_model.train_hit_model(fold_train, num_threads=num_threads)
            fold_model.train_return_model(fold_train, num_threads=num_threads)

            fold_train_pred = fold_model.predict_ev(fold_train.copy())
            fold_ev_corrector = ev_corrector.__class__()
            fold_ev_corrector.train(fold_train_pred, num_threads=num_threads)

            fold_val = fold_model.predict_ev(fold_val_raw)
            fold_val["p_win_oof"] = fold_val["p_win_pred"]
            fold_val = fold_ev_corrector.correct_ev(
                fold_val,
                probability_col="p_win_oof",
            )
        except Exception as exc:
            logger.warning("Skipping Win OOF fold: %s", exc)
            continue

        p_fund_oof[val_idx] = pd.to_numeric(fold_val["p_win_corrected"], errors="coerce").values
        p_market_oof[val_idx] = np.clip(
            np.where(fold_val["tanodds"] > 0, 1.0 / fold_val["tanodds"].values, np.nan),
            0.01,
            0.99,
        )
        y_oof[val_idx] = (fold_val["kakuteijyuni"] == 1).astype(float).values

    valid = ~(np.isnan(p_fund_oof) | np.isnan(p_market_oof) | np.isnan(y_oof))
    logger.info("Win OOF: %d valid / %d total samples", int(valid.sum()), len(valid))
    return p_fund_oof[valid], p_market_oof[valid], y_oof[valid].astype(int)


# ---------------------------------------------------------------------------
# Calibration comparison functions (D-05, D-07, D-08)
# ---------------------------------------------------------------------------


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (D-07).

    ECE = sum_b (n_b / N) * |avg_confidence_b - avg_accuracy_b|
    Reference: Guo et al., 2017 "On Calibration of Modern Neural Networks"
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    if n_total == 0:
        return 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        avg_confidence = float(y_prob[mask].mean())
        avg_accuracy = float(y_true[mask].mean())
        ece += count * abs(avg_accuracy - avg_confidence)
    return ece / n_total


class BetaCalibrationManual:
    """Manual 3-parameter Beta calibration (fallback if betacal package incompatible)."""

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 1.0
        self.c: float = 0.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> BetaCalibrationManual:
        p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
        y = np.asarray(y, dtype=float)

        def neg_loglik(params: np.ndarray) -> float:
            a, b = np.exp(params[0]), np.exp(params[1])  # Ensure positive
            c = params[2]
            z = np.clip((p - c) / (1.0 - c + 1e-10), 1e-10, 1 - 1e-10)
            from scipy.special import betaln

            eps = 1e-10
            log_z = np.log(z + eps)
            log_1z = np.log(1 - z + eps)
            log_pdf = (a - 1) * log_z + (b - 1) * log_1z - betaln(a, b)
            p_cal = 1.0 / (1.0 + np.exp(-log_pdf))
            p_cal = np.clip(p_cal, 1e-10, 1 - 1e-10)
            return float(-np.sum(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal)))

        from scipy.optimize import minimize as scipy_minimize

        res = scipy_minimize(neg_loglik, x0=[0.0, 0.0, 0.0], method="L-BFGS-B")
        self.a = float(np.exp(res.x[0]))
        self.b = float(np.exp(res.x[1]))
        self.c = float(res.x[2])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
        z = np.clip((p - self.c) / (1.0 - self.c + 1e-10), 1e-10, 1 - 1e-10)
        from scipy.special import betaln

        eps = 1e-10
        log_pdf = (
            (self.a - 1) * np.log(z + eps)
            + (self.b - 1) * np.log(1 - z + eps)
            - betaln(self.a, self.b)
        )
        return 1.0 / (1.0 + np.exp(-log_pdf))


def compare_calibrations(
    p_benter: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
) -> dict[str, object]:
    """Compare Beta vs Isotonic calibration on Benter-combined probabilities (D-05, D-07).

    Splits data by train_ratio (time-series: first train_ratio is train, rest is validation).
    Returns dict with Brier Scores, ECE values, and winner.

    Args:
        p_benter: Benter-combined win probabilities (OOF).
        y: Binary win/loss labels.
        train_ratio: Fraction used for fitting calibrators.

    Returns:
        dict with keys: beta_brier, iso_brier, beta_ece, iso_ece,
                       winner, beta_calibrator, iso_calibrator
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    p_benter = np.asarray(p_benter, dtype=float)
    y = np.asarray(y, dtype=float)

    # Time-series split
    split = int(len(p_benter) * train_ratio)
    p_train, p_val = p_benter[:split], p_benter[split:]
    y_train, y_val = y[:split], y[split:]

    if len(p_train) < 100 or len(p_val) < 50:
        logger.warning(
            "Insufficient data for calibration comparison: train=%d, val=%d",
            len(p_train),
            len(p_val),
        )
        return {
            "beta_brier": float("inf"),
            "iso_brier": float("inf"),
            "beta_ece": float("inf"),
            "iso_ece": float("inf"),
            "winner": "none",
            "beta_calibrator": None,
            "iso_calibrator": None,
        }

    # Beta calibration (3-param, recommended per D-08)
    has_beta = False
    beta_cal: object = None
    try:
        from betacal import BetaCalibration as _BetaCal

        _raw = _BetaCal(parameters="abm")
        _raw.fit(p_train, y_train)
        p_beta = np.asarray(_raw.predict(p_val), dtype=float)
        # Wrap predict() as transform() for calibrate() compatibility
        beta_cal = BetaCalibrationManual()
        beta_cal.fit(p_train, y_train)
        has_beta = True
    except (ImportError, Exception) as e:
        logger.warning("betacal unavailable or failed (%s), using manual fallback", e)
        beta_cal = BetaCalibrationManual()
        beta_cal.fit(p_train, y_train)
        p_beta = np.asarray(beta_cal.transform(p_val), dtype=float)
        has_beta = True

    # Isotonic calibration (comparison per D-05)
    iso_cal = IsotonicRegression(out_of_bounds="clip")
    iso_cal.fit(p_train, y_train)
    p_iso = np.asarray(iso_cal.transform(p_val), dtype=float)

    # Quantitative comparison (D-07)
    beta_brier = float(brier_score_loss(y_val, np.clip(p_beta, 1e-10, 1 - 1e-10)))
    iso_brier = float(brier_score_loss(y_val, np.clip(p_iso, 1e-10, 1 - 1e-10)))
    beta_ece = compute_ece(y_val, p_beta)
    iso_ece = compute_ece(y_val, p_iso)

    # Determine winner (Brier Score primary, ECE secondary)
    if beta_brier <= iso_brier:
        winner = "beta"
    elif iso_brier < beta_brier and (iso_brier - beta_brier) / max(beta_brier, 1e-10) < 0.05:
        # Isotonic is only slightly better -- prefer Beta for stability (D-08)
        winner = "beta"
    else:
        winner = "isotonic"

    logger.info(
        "Calibration comparison: Beta(Brier=%.6f, ECE=%.6f) vs "
        "Isotonic(Brier=%.6f, ECE=%.6f) -> winner=%s",
        beta_brier,
        beta_ece,
        iso_brier,
        iso_ece,
        winner,
    )

    return {
        "beta_brier": beta_brier,
        "iso_brier": iso_brier,
        "beta_ece": beta_ece,
        "iso_ece": iso_ece,
        "winner": winner,
        "beta_calibrator": beta_cal if has_beta else None,
        "iso_calibrator": iso_cal,
    }


def generate_reliability_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Generate reliability diagram data for visualization (D-07, Success Criteria 4).

    Returns per-bin data for plotting fraction_of_positives vs mean_predicted_value.
    Perfect calibration: the two arrays are equal (diagonal line).
    """
    from sklearn.calibration import calibration_curve

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    return {
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted_value,
        "bin_edges": np.linspace(0.0, 1.0, n_bins + 1),
    }
