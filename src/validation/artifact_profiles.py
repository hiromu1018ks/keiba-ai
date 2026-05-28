"""OOF artifact health profiles for Phase 39/40 components.

Provides CalibratorArtifactProfile (MarketAwareWinCalibrator) and
RankerArtifactProfile (RaceLevelRanker) for validating OOF artifact
integrity without modifying the OOFHealthValidator core.

SAF-02: Plugin pattern per D-06 -- OOFHealthValidator imports PROFILES
registry to discover Phase 39/40 artifact checks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CalibratorArtifactProfile:
    """Artifact profile for MarketAwareWinCalibrator OOF output.

    Validates:
    - Required columns present (race_id, p_win_combined, p_win_final, fold)
    - Forbidden columns absent (p_win_pred)
    - Probability columns: no NaN, no inf, values in [0, 1]
    - Sum-to-1.0 per race_id for p_win_combined
    - Fold column present
    """

    def __init__(
        self,
        required_columns: tuple[str, ...] = (
            "race_id", "p_win_combined", "p_win_final", "fold",
        ),
        forbidden_columns: tuple[str, ...] = ("p_win_pred",),
        probability_columns: tuple[str, ...] = ("p_win_combined", "p_win_final"),
        fold_col: str = "ability_oof_fold",
        sum_to_one_tolerance: float = 1e-6,
    ) -> None:
        self.required_columns = required_columns
        self.forbidden_columns = forbidden_columns
        self.probability_columns = probability_columns
        self.fold_col = fold_col
        self.sum_to_one_tolerance = sum_to_one_tolerance

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Run all calibrator artifact checks, return failure strings.

        Per D-07: NaN/inf detection, [0,1] range, sum-to-1.0,
        p_win_pred forbidden, fold required.
        """
        failures: list[str] = []

        # (a) Check required columns
        for col in self.required_columns:
            if col not in df.columns:
                failures.append(f"Required column '{col}' missing (required)")

        # Early exit if race_id missing (can't do per-race checks)
        has_race_id = "race_id" in df.columns

        # (c) Check fold_col present
        if self.fold_col not in df.columns:
            # Also check if default 'fold' is in required_columns (short alias)
            if "fold" in df.columns:
                pass  # 'fold' column present, acceptable
            else:
                failures.append(
                    f"Fold column '{self.fold_col}' missing (required)"
                )

        # (b) Check forbidden columns
        for col in self.forbidden_columns:
            if col in df.columns:
                failures.append(
                    f"Forbidden column '{col}' present -- "
                    f"train-mode prediction detected (forbidden)"
                )

        # (d) Check probability columns for NaN, inf, range
        for col in self.probability_columns:
            if col not in df.columns:
                continue  # Already reported as missing required column

            values = df[col]

            # NaN check
            nan_count = int(values.isna().sum())
            if nan_count > 0:
                failures.append(
                    f"Column '{col}' contains {nan_count} NaN value(s)"
                )

            # inf check
            inf_count = int(np.isinf(values.dropna()).sum())
            if inf_count > 0:
                failures.append(
                    f"Column '{col}' contains {inf_count} inf value(s)"
                )

            # Range check [0, 1]
            valid_vals = values.dropna()
            if len(valid_vals) > 0:
                above = int((valid_vals > 1.0).sum())
                below = int((valid_vals < 0.0).sum())
                if above > 0 or below > 0:
                    out_of_range = above + below
                    failures.append(
                        f"Column '{col}' has {out_of_range} value(s) "
                        f"outside [0, 1] range"
                    )

        # (e) Sum-to-1.0 per race_id
        if has_race_id and "p_win_combined" in df.columns:
            race_sums = df.groupby("race_id", observed=True)["p_win_combined"].sum()
            violations = race_sums[
                (race_sums - 1.0).abs() > self.sum_to_one_tolerance
            ]
            if len(violations) > 0:
                failures.append(
                    f"Sum-to-1.0 violation in {len(violations)} race(s) -- "
                    f"p_win_combined does not sum to 1.0 within tolerance "
                    f"(sum-to-1 check)"
                )

        return failures


class RankerArtifactProfile:
    """Artifact profile for RaceLevelRanker OOF output.

    Validates:
    - Required columns present (race_id, investment_score, fold)
    - Score columns: no NaN, no inf (investment_score, relevance_score, value_score)
    - Race-level rank determinism (warning for ties)
    - Fold column present
    """

    def __init__(
        self,
        required_columns: tuple[str, ...] = (
            "race_id", "investment_score", "fold",
        ),
        score_columns: tuple[str, ...] = (
            "investment_score", "relevance_score", "value_score",
        ),
        fold_col: str = "ability_oof_fold",
    ) -> None:
        self.required_columns = required_columns
        self.score_columns = score_columns
        self.fold_col = fold_col

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Run all ranker artifact checks, return failure strings.

        Per D-08: NaN/inf in scores, race-level rank determinism,
        fold required.
        """
        failures: list[str] = []

        # (a) Check required columns
        for col in self.required_columns:
            if col not in df.columns:
                failures.append(f"Required column '{col}' missing (required)")

        # (b) Check fold_col present
        if self.fold_col not in df.columns:
            if "fold" in df.columns:
                pass  # 'fold' column present, acceptable
            else:
                failures.append(
                    f"Fold column '{self.fold_col}' missing (required)"
                )

        # (c) Check score columns for NaN, inf
        for col in self.score_columns:
            if col not in df.columns:
                continue  # May not be required, skip

            values = df[col]

            # NaN check
            nan_count = int(values.isna().sum())
            if nan_count > 0:
                failures.append(
                    f"Column '{col}' contains {nan_count} NaN value(s)"
                )

            # inf check
            inf_count = int(np.isinf(values.dropna()).sum())
            if inf_count > 0:
                failures.append(
                    f"Column '{col}' contains {inf_count} inf value(s)"
                )

        # (d) Race-level rank determinism check
        has_race_id = "race_id" in df.columns
        has_investment = "investment_score" in df.columns

        if has_race_id and has_investment:
            for race_id, race_df in df.groupby("race_id", observed=True):
                scores = race_df["investment_score"]
                # Check for duplicate scores
                if scores.duplicated().any():
                    # Identical scores exist -- rank method "first" gives
                    # different ranks to identical scores, which is technically
                    # deterministic but not stable across implementations.
                    n_dup = int(scores.duplicated().sum())
                    failures.append(
                        f"WARNING: Race '{race_id}' has {n_dup} duplicated "
                        f"investment_score(s) -- rank determinism may vary "
                        f"(rank determinism)"
                    )

        return failures


# Module-level default instances for convenience
DEFAULT_CALIBRATOR_PROFILE = CalibratorArtifactProfile()
DEFAULT_RANKER_PROFILE = RankerArtifactProfile()

# PROFILES registry: OOFHealthValidator plugin discovery (D-06)
PROFILES: dict[str, type] = {
    "calibrator": CalibratorArtifactProfile,
    "ranker": RankerArtifactProfile,
}
