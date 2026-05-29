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
from utils.wf_splits import walk_forward_race_splits as _walk_forward_race_splits

from models.benter_combination import BenterCombination

if TYPE_CHECKING:
    from sklearn.isotonic import IsotonicRegression

    from models.benter_combination import TemperatureScaling

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Comprehensive column audit for IFF build_frame(mode="train") compatibility.
#
# The OOF result DataFrame must contain every column listed in
# schema_registry.py train_sources for REQUIRED specs, plus all columns
# that derived features reference (e.g. race_id for groupby).
#
# Columns are split into two groups:
#   FOLD_GENERATED -- produced by predict_ev() or correct_ev() during each
#                     fold and must be captured per-fold into OOF Series.
#   STATIC_PASSTHROUGH -- already present in the source df and simply copied.
# ---------------------------------------------------------------------------

# Columns generated inside each fold by predict_ev / correct_ev.
# These must be captured into OOF Series because fold models differ.
_FOLD_GENERATED_COLS: tuple[str, ...] = (
    "ev_win_corrected",
    "p_x_e_interaction",
    "p_minus_e_gap",
)

# Static columns that IFF build_frame(mode="train") may resolve.
# Covers every train_sources entry in schema_registry.py FEATURE_SPECS
# plus auxiliary columns needed by derived feature computations.
_STATIC_PASSTHROUGH_COLS: tuple[str, ...] = (
    # Market / odds (already present, extend list)
    "tanodds",
    "popularity_rank",
    "field_size",
    "race_id",
    "race_date",
    "umaban",
    "surface",
    # IFF required: model_prob sources
    "p_ability_win",
    "p_ability_place",
    # IFF required: market_prob sources
    "p_market_win_adj",
    "overround",
    "market_entropy",
    "odds_skewness",
    "implied_prob_hhi",
    # IFF required: model_market_gap sources
    "signed_log_error_win",
    "abs_log_error_win",
    "deviation_rank",
    "deviation_zscore",
    "odds_to_ability_ratio",
    "market_error_rank_in_race",
    # IFF required: race_relative sources
    "rl_n_horses",
    "form_trend_race_rank",
    "blood_total_wr_race_rank",
    "closing_index_avg",
    # IFF required: odds_band sources (tanodds already listed above)
    # IFF required: late_odds sources
    "odds_drop_rate_60_10",
    "odds_drop_rate_30_10",
    "odds_velocity",
    "odds_volatility",
    "odds_acceleration",
    "odds_direction_consistency",
    "popularity_change_30_10",
    # IFF required: ability_form sources
    "norm_finish_logit_avg",
    "harontimel5_zscore",
    "form_trend",
    "form_consistency",
    "blood_surface_wr",
    "blood_total_wr",
    "sire_wr",
    "jockey_wr_overall",
    "trainer_wr_overall",
    "jt_combo_wr",
    "class_level_current",
    "weighted_recent_form_finish",
    "grade_x_form_trend",
    "distance_x_closing_index",
    "dm_time_rank",
    "class_move",
    # IFF required: course_pace sources
    "closing_speed_ratio_avg",
    "haron_race_gap_avg",
    "pace_ratio_avg",
    "distance_bin",
    "grade_code",
    "track_condition_code",
    "course_wr",
    "pace_aptitude",
    "haron_zscore_trend",
    "pace_early_avg",
    "pace_late_avg",
    "closing_speed_ratio_avg_race_rank",
    # IFF required: uncertainty sources
    "EV_lower_win_corrected",
    "EV_upper_win_corrected",
    "conformal_confidence_score",
    "market_log_error_win",
    "isotonic_residual_win",
)

# ---------------------------------------------------------------------------
# String-to-numeric encoding for categorical columns that IFF expects as float.
#
# The training pipeline stores surface/distance_bin/grade_code as strings
# (LightGBM handles them as categorical). IFF schema_registry declares these
# as dtype="float64" and RaceLevelRanker expects numeric values.
# This encoding is applied after passthrough copy to ensure numeric output.
# ---------------------------------------------------------------------------
_STRING_COL_ENCODINGS: dict[str, dict[str, float]] = {
    # surface: 0=turf, 1=dirt (matches RaceLevelRanker convention at line 525)
    "surface": {"turf": 0, "dirt": 1},
    # distance_bin: ordinal encoding by distance range
    "distance_bin": {"sprint": 0, "mile": 1, "intermediate": 2, "long": 3, "unknown": -1},
    # grade_code: JRA gradecd letter codes (A=G1, B=G2, C=G3, etc.)
    # Scale matches features/race_class.py GRADE_LEVEL_MAP.
    "grade_code": {
        "X": 0.0,   # ungraded
        "H": 5.0,   # other
        "E": 5.0,   # Open/special
        "D": 5.5,   # non-graded stakes
        "G": 5.5,   # jump graded
        "L": 5.5,   # Listed
        "C": 6.0,   # G3
        "B": 7.0,   # G2
        "A": 8.0,   # G1
        "": 0.0,    # empty/missing
    },
}


def _encode_string_columns(result: pd.DataFrame) -> None:
    """Encode string categorical columns to numeric in-place.

    For columns listed in _STRING_COL_ENCODINGS, if the column contains
    string values, map them to numeric. Already-numeric columns are left
    unchanged (passthrough when the source df already converted them).
    """
    for col, mapping in _STRING_COL_ENCODINGS.items():
        if col not in result.columns:
            continue
        series = result[col]
        # Skip if already numeric (e.g. test data or pre-converted pipeline)
        if pd.api.types.is_numeric_dtype(series):
            continue
        # Map string values to numeric; unmapped values get NaN
        result[col] = series.map(mapping).astype(float)
        unmapped_count = result[col].isna().sum()
        if unmapped_count > 0:
            logger.warning(
                "String column '%s' had %d unmapped values after encoding. "
                "Mapping: %s. Sample unmapped: %s",
                col,
                unmapped_count,
                mapping,
                series.dropna().unique()[:5] if series.notna().any() else [],
            )


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
) -> pd.DataFrame:
    """Generate OOF predictions for MarketAwareWinCalibrator training (D-04, D-21).

    Uses expanding walk-forward splits on race_date to generate predictions.
    Each validation fold is scored only by models trained on past races.

    Args:
        df: Training DataFrame sorted by race_date.
        win_model_cls: WinTwoStageModel class (imported lazily to avoid circular deps).
        ev_corrector: EVCorrectionModel with trained correct_ev() method.
        n_splits: Number of CV folds.
        num_threads: LightGBM thread count.

    Returns:
        DataFrame with OOF predictions plus all columns needed by
        IFF build_frame(mode="train") and MarketAwareWinCalibrator.
        Rows with NaN in core columns are dropped.
    """
    sort_cols = [col for col in ["race_date", "race_id", "umaban"] if col in df.columns]
    df = (
        df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df.reset_index(drop=True)
    )
    splits = _walk_forward_race_splits(df, n_splits=n_splits)

    # Initialize OOF Series for fold-generated columns
    oof_p_win_corrected = pd.Series(np.nan, index=df.index, dtype=float)
    oof_p_market_norm = pd.Series(np.nan, index=df.index, dtype=float)
    oof_kakuteijyuni = pd.Series(np.nan, index=df.index, dtype=float)
    oof_p_win_oof = pd.Series(np.nan, index=df.index, dtype=float)
    # Phase 40: OOF-safe calibrated EV for ranker value target (D-09, D-12)
    oof_calibrated_ev = pd.Series(np.nan, index=df.index, dtype=float)
    # OOF e_return_win_pred for IFF build_frame (if_e_return requires it in train mode)
    oof_e_return_win_pred = pd.Series(np.nan, index=df.index, dtype=float)
    # Additional fold-generated columns for IFF compatibility
    oof_fold_cols: dict[str, pd.Series] = {
        col: pd.Series(np.nan, index=df.index, dtype=float)
        for col in _FOLD_GENERATED_COLS
    }

    n_failed = 0
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
        except (ValueError, RuntimeError) as exc:
            n_failed += 1
            logger.warning("Skipping Win OOF fold: %s", exc)
            if n_failed >= len(splits):
                raise RuntimeError(
                    f"All {len(splits)} Win OOF folds failed; "
                    "check input data quality"
                ) from exc
            continue

        oof_p_win_corrected.iloc[val_idx] = pd.to_numeric(
            fold_val["p_win_corrected"], errors="coerce"
        ).values
        oof_p_market_norm.iloc[val_idx] = np.clip(
            np.where(fold_val["tanodds"] > 0, 1.0 / fold_val["tanodds"].values, np.nan),
            0.01,
            0.99,
        )
        oof_kakuteijyuni.iloc[val_idx] = (
            fold_val["kakuteijyuni"].astype(float).values
        )
        oof_p_win_oof.iloc[val_idx] = pd.to_numeric(
            fold_val["p_win_oof"], errors="coerce"
        ).values
        # Phase 40: capture fold-level EV correction for ranker value target (D-12)
        if "ev_win_corrected" in fold_val.columns:
            oof_calibrated_ev.iloc[val_idx] = pd.to_numeric(
                fold_val["ev_win_corrected"], errors="coerce"
            ).values
        # Capture e_return_win_pred for IFF train-mode resolution
        if "e_return_win_pred" in fold_val.columns:
            oof_e_return_win_pred.iloc[val_idx] = pd.to_numeric(
                fold_val["e_return_win_pred"], errors="coerce"
            ).values
        # Capture additional fold-generated columns for IFF compatibility
        for col in _FOLD_GENERATED_COLS:
            if col in fold_val.columns:
                oof_fold_cols[col].iloc[val_idx] = pd.to_numeric(
                    fold_val[col], errors="coerce"
                ).values

    # Build result DataFrame with all columns needed by downstream consumers
    result = df[[]].copy()
    result["p_win_corrected"] = oof_p_win_corrected
    result["p_win_oof"] = oof_p_win_oof
    result["p_market_norm"] = oof_p_market_norm
    result["kakuteijyuni"] = oof_kakuteijyuni
    # Phase 40: OOF-safe calibrated EV for ranker value target (D-09, D-12)
    result["calibrated_ev_oof"] = oof_calibrated_ev
    # e_return_win_pred for IFF build_frame train mode
    result["e_return_win_pred"] = oof_e_return_win_pred
    # Additional fold-generated columns
    for col, series in oof_fold_cols.items():
        result[col] = series

    # Copy all static/passthrough columns from source df
    for col in _STATIC_PASSTHROUGH_COLS:
        if col in df.columns:
            result[col] = df[col].values

    # Encode string categorical columns to numeric for IFF / RaceLevelRanker.
    # The training pipeline stores surface/distance_bin/grade_code as strings
    # (LightGBM handles them as categorical), but IFF expects float64.
    _encode_string_columns(result)

    # Compute p_win_race_rank_pct from OOF predictions (D-19)
    valid_oof_mask = result["p_win_oof"].notna()
    result["p_win_race_rank_pct"] = np.nan
    if valid_oof_mask.any() and "race_id" in result.columns:
        result.loc[valid_oof_mask, "p_win_race_rank_pct"] = (
            result.loc[valid_oof_mask]
            .groupby("race_id", observed=True)["p_win_oof"]
            .rank(pct=True, method="min", ascending=False)
            .values
        )

    # Drop rows with NaN in core columns
    core_cols = ["p_win_oof", "p_market_norm", "kakuteijyuni"]
    valid = result[core_cols].notna().all(axis=1)
    result = result[valid].reset_index(drop=True)

    logger.info("Win OOF: %d valid / %d total samples", len(result), len(df))
    return result


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

    # Beta calibration (3-param manual, D-08)
    has_beta = True
    beta_cal = BetaCalibrationManual()
    beta_cal.fit(p_train, y_train)
    p_beta = np.asarray(beta_cal.transform(p_val), dtype=float)

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
