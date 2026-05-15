"""Target Encoding for high-cardinality categorical variables.

OOF-safe implementation using 3-fold expanding window (same as AbilityModel.train_oof).
Prevents target leakage by ensuring each fold's test data only uses
training data from earlier time periods.

Target encoding columns:
- te_blood_keito_cd: blood lineage code TE (Stage1 + Stage2)
- te_kisyucode: jockey code TE (Stage2 only, per Phase 25 D-02)
- te_chokyosicode: trainer code TE (Stage2 only, per Phase 25 D-02)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Stage1用: blood_keito_cdのみ (安全性: TE target == Stage1 targetのため、
# Stage1 OOF内でリークする可能性がある -> 実際はStage1には追加せず Stage2のみ)
TE_FEATURE_COLS: list[str] = ["te_blood_keito_cd"]

# Stage2用: 全3カテゴリ (Win/Place FEATURE_COLSに追加)
TE_STAGE2_FEATURE_COLS: list[str] = [
    "te_blood_keito_cd",
    "te_kisyucode",
    "te_chokyosicode",
]


class TargetEncoder:
    """OOF-safe target encoder using expanding window folds.

    Uses the same fold boundaries as AbilityModel.train_oof():
    - n_folds=3 expanding window by race_date
    - boundaries = [dates[n_dates * (i+1) // (n_folds+1)] for i in range(n_folds)]

    Smoothing uses Beta(1, smoothing) pattern from jockey_context_features.py:
    te_value = (cat_sum + smoothing * global_mean) / (cat_count + smoothing)
    """

    def __init__(
        self,
        cat_cols: list[str],
        target_col: str = "kakuteijyuni",
        n_folds: int = 3,
        smoothing: int = 10,
        min_samples: int = 5,
    ) -> None:
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.min_samples = min_samples

        # Learned state (populated by fit_transform_oof)
        self.encoding_maps_: dict[str, dict[float, float]] = {}
        self.global_mean_: float = 0.0

    def fit_transform_oof(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute OOF-safe target encodings using expanding window folds.

        Args:
            df: DataFrame with race_date, cat_cols, and target_col.
                Need not be sorted -- sorted internally.

        Returns:
            DataFrame with te_{cat_col} columns added.
        """
        df = df.copy()

        # Ensure race_date is datetime and sorted
        if df["race_date"].dtype == object:
            df["race_date"] = pd.to_datetime(df["race_date"])
        df = df.sort_values("race_date").reset_index(drop=True)

        # Binary target: kakuteijyuni == 1 -> 1, else -> 0
        target_binary = (df[self.target_col] == 1).astype(int)
        global_mean = float(target_binary.mean())
        self.global_mean_ = global_mean

        # Compute fold boundaries (same as AbilityModel.train_oof)
        dates = sorted(df["race_date"].unique())
        n_dates = len(dates)

        # Initialize TE columns with NaN
        for cat_col in self.cat_cols:
            te_col = f"te_{cat_col}"
            df[te_col] = np.nan

        if n_dates < self.n_folds + 1:
            # Not enough dates for folds -- fill with global mean
            for cat_col in self.cat_cols:
                te_col = f"te_{cat_col}"
                df[te_col] = global_mean
            # Still build encoding maps from all data
            for cat_col in self.cat_cols:
                self.encoding_maps_[cat_col] = self._compute_cat_stats(
                    df, cat_col, target_binary, global_mean,
                )
            return df

        boundaries = [
            dates[n_dates * (i + 1) // (self.n_folds + 1)]
            for i in range(self.n_folds)
        ]

        # Process each fold
        for i in range(self.n_folds):
            train_end = boundaries[i]
            test_end = (
                boundaries[i + 1]
                if i + 1 < self.n_folds
                else dates[-1] + pd.Timedelta(days=1)
            )

            train_mask = df["race_date"] < train_end
            test_mask = (df["race_date"] >= train_end) & (df["race_date"] < test_end)

            train_target = target_binary.loc[train_mask]

            if len(train_target) == 0:
                # No training data for this fold (first fold of expanding window)
                # Fill test rows with global mean
                for cat_col in self.cat_cols:
                    te_col = f"te_{cat_col}"
                    df.loc[test_mask, te_col] = global_mean
                continue

            fold_global_mean = float(train_target.mean())

            for cat_col in self.cat_cols:
                te_col = f"te_{cat_col}"

                # Compute per-category stats from training data
                train_cat = df.loc[train_mask, cat_col]
                cat_stats = train_target.groupby(train_cat, observed=True).agg(
                    ["sum", "count"]
                )

                # Smoothed TE: (sum + smoothing * fold_global_mean) / (count + smoothing)
                smoothed = (
                    cat_stats["sum"] + self.smoothing * fold_global_mean
                ) / (cat_stats["count"] + self.smoothing)

                # Map to test data
                test_cats = df.loc[test_mask, cat_col]
                te_values = test_cats.map(smoothed)

                # Fill unknown categories with fold global mean
                te_values = te_values.fillna(fold_global_mean)

                df.loc[test_mask, te_col] = te_values.values

        # Build final encoding maps from all data (for transform())
        for cat_col in self.cat_cols:
            self.encoding_maps_[cat_col] = self._compute_cat_stats(
                df, cat_col, target_binary, global_mean,
            )

        # Fill remaining NaN (rows in first fold's train-only portion)
        # These rows are never in any test fold, so they have no OOF TE value.
        # Use the full-data encoding map for a reasonable (non-OOF) estimate.
        for cat_col in self.cat_cols:
            te_col = f"te_{cat_col}"
            nan_mask = df[te_col].isna()
            if nan_mask.any():
                mapping = self.encoding_maps_[cat_col]
                df.loc[nan_mask, te_col] = (
                    df.loc[nan_mask, cat_col].map(mapping).fillna(global_mean)
                )

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned TE encodings to new data (inference time).

        Args:
            df: DataFrame with cat_cols.

        Returns:
            DataFrame with te_{cat_col} columns added.
        """
        df = df.copy()

        for cat_col in self.cat_cols:
            te_col = f"te_{cat_col}"
            if cat_col in self.encoding_maps_:
                mapping = self.encoding_maps_[cat_col]
                df[te_col] = df[cat_col].map(mapping).fillna(self.global_mean_)
            else:
                df[te_col] = self.global_mean_

        return df

    def _compute_cat_stats(
        self,
        df: pd.DataFrame,
        cat_col: str,
        target_binary: pd.Series,
        global_mean: float,
    ) -> dict[float, float]:
        """Compute smoothed TE values for all categories in data."""
        cat_stats = target_binary.groupby(df[cat_col], observed=True).agg(
            ["sum", "count"]
        )
        smoothed = (
            cat_stats["sum"] + self.smoothing * global_mean
        ) / (cat_stats["count"] + self.smoothing)
        return smoothed.to_dict()
