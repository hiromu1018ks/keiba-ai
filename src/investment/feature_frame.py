"""Investment feature frame builder.

Provides InvestmentFeatureFrameBuilder with build_frame(df, mode) API for
dual-mode (train/infer) investment feature generation per D-10, D-11,
IFF-01, IFF-02, IFF-03.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from investment.leakage import validate_no_post_race_leakage
from investment.schema_registry import (
    CATEGORY_ORDER,
    FEATURE_SPECS,
    InvestmentFeatureSpec,
)

__all__ = [
    "InvestmentFeatureFrameBuilder",
    "build_frame",
]

# Constant for builder version
BUILDERS_VERSION = "1.0.0"

# Logit helper
_CLIP_EPS = 1e-15


def _logit(p: pd.Series | np.ndarray) -> np.ndarray:
    """Compute logit(p) with edge case clipping."""
    p_arr = np.asarray(p, dtype=np.float64)
    p_clipped = np.clip(p_arr, _CLIP_EPS, 1.0 - _CLIP_EPS)
    return np.log(p_clipped / (1.0 - p_clipped))


class InvestmentFeatureFrameBuilder:
    """Builder for investment feature frames (D-10, IFF-01).

    Generates 90-130 'if_*' columns using schema-driven dual-mode resolution.
    Train mode uses OOF-safe sources; infer mode uses production sources.
    Output schema is identical across modes (D-12).
    """

    BUILDERS_VERSION: str = BUILDERS_VERSION

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_source(
        df: pd.DataFrame, spec: InvestmentFeatureSpec, mode: str
    ) -> pd.Series:
        """Resolve source column for a spec based on mode (D-18, D-19).

        Args:
            df: Input DataFrame with source columns.
            spec: Feature specification.
            mode: "train" or "infer".

        Returns:
            pd.Series with resolved values.

        Raises:
            ValueError: If required source column is not found.
        """
        sources = spec.train_sources if mode == "train" else spec.infer_sources

        for source_col in sources:
            if source_col in df.columns:
                return df[source_col].copy()

        # Source not found
        if spec.required:
            raise ValueError(
                f"Required feature '{spec.name}': no source column found "
                f"for mode={mode}. Expected one of: {sources}"
            )

        # Optional: return default_value series
        return pd.Series(
            spec.default_value, index=df.index, dtype="float64"
        )

    @staticmethod
    def _compute_derived(
        name: str, result: pd.DataFrame, df: pd.DataFrame
    ) -> pd.Series | None:
        """Compute a derived feature (empty train/infer_sources in spec).

        Args:
            name: Feature name (e.g. "if_ev_raw").
            result: Partially-built output DataFrame (identity cols + resolved features).
            df: Original input DataFrame.

        Returns:
            pd.Series with computed values, or None if name is not a derived feature.
        """
        if name == "if_ev_raw":
            return result["if_p_win"] * result["if_e_return"]

        if name == "if_edge_win":
            p_final = result["if_p_win_final"]
            odds = result["if_odds"]
            return p_final * odds - 1.0

        if name == "if_logit_gap":
            p_model = result["if_p_win"].astype(np.float64)
            p_market = result["if_implied_prob"].astype(np.float64)
            return pd.Series(_logit(p_model) - _logit(p_market), index=result.index)

        if name == "if_abs_logit_gap":
            return result["if_logit_gap"].abs()

        if name == "if_edge_rank_in_race":
            # Rank of logit_gap within each race (pct, descending)
            return result.groupby("race_id")["if_logit_gap"].rank(
                pct=True, method="min", ascending=False
            )

        if name == "if_edge_zscore_in_race":
            # Z-score of logit_gap within each race
            grouped = result.groupby("race_id")["if_logit_gap"]
            mean = grouped.transform("mean")
            std = grouped.transform("std")
            return (result["if_logit_gap"] - mean) / std.replace(0, np.nan)

        if name == "if_top3_gap":
            # Gap between each horse's logit_gap and the top3 mean in race
            def _top3_gap(group: pd.Series) -> pd.Series:
                top3_mean = group.nlargest(3).mean()
                return group - top3_mean
            return result.groupby("race_id")["if_logit_gap"].transform(_top3_gap)

        if name == "if_field_ev_dispersion":
            # Std of if_ev_corrected within each race
            return result.groupby("race_id")["if_ev_corrected"].transform("std")

        if name == "if_p_win_race_rank":
            return result.groupby("race_id")["if_p_win"].rank(
                pct=True, method="min", ascending=False
            )

        if name == "if_ev_race_rank":
            return result.groupby("race_id")["if_ev_corrected"].rank(
                pct=True, method="min", ascending=False
            )

        if name == "if_ev_top1_gap":
            # Gap to top1 EV in race
            def _top1_gap(group: pd.Series) -> pd.Series:
                top1 = group.max()
                return group - top1
            return result.groupby("race_id")["if_ev_corrected"].transform(_top1_gap)

        if name == "if_ev_top3_indicator":
            # 1 if ev_race_rank <= 3, else 0
            rank = result.groupby("race_id")["if_ev_corrected"].rank(
                method="min", ascending=False
            )
            return (rank <= 3).astype("float64")

        if name == "if_p_win_gap_to_fav":
            # Gap to favorite's p_win (lowest popularity_rank = favorite)
            if "if_popularity_rank" not in result.columns:
                return pd.Series(np.nan, index=result.index)
            idx_min = result.groupby("race_id")["if_popularity_rank"].idxmin()
            fav_p_win_map = result.loc[idx_min].set_index("race_id")["if_p_win"]
            return result["if_p_win"] - result["race_id"].map(fav_p_win_map)

        if name == "if_odds_log":
            return np.log(result["if_odds"].clip(lower=_CLIP_EPS))

        if name == "if_odds_band_median_ev":
            # Median of if_ev_corrected within each odds band in race
            return result.groupby(["race_id", "if_odds_band_id"])[
                "if_ev_corrected"
            ].transform("median")

        if name == "if_odds_band_count":
            # Count within each odds band in race
            return result.groupby(["race_id", "if_odds_band_id"])[
                "if_odds_band_id"
            ].transform("count").astype("float64")

        if name == "if_odds_band_ev_rank":
            # Rank of if_ev_corrected within odds band in race
            return result.groupby(["race_id", "if_odds_band_id"])[
                "if_ev_corrected"
            ].rank(pct=True, method="min", ascending=False)

        if name == "if_late_money_ratio":
            # odds_drop_30_10 / odds_drop_60_10
            drop_30 = result.get("if_odds_drop_30_10")
            drop_60 = result.get("if_odds_drop_60_10")
            if drop_30 is not None and drop_60 is not None:
                return drop_30 / drop_60.replace(0, np.nan)
            return pd.Series(np.nan, index=result.index)

        if name == "if_conformal_width":
            upper = result["if_conformal_upper"]
            lower = result["if_conformal_lower"]
            return upper - lower

        if name == "if_ev_uncertainty_ratio":
            width = result["if_conformal_width"]
            ev = result["if_ev_corrected"]
            return width / ev.replace(0, np.nan)

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_frame(
        self,
        df: pd.DataFrame,
        mode: Literal["train", "infer"],
    ) -> pd.DataFrame:
        """Build investment feature frame (D-10, IFF-01).

        Generates 90-130 'if_*' columns using schema-driven dual-mode resolution.
        Train mode uses OOF-safe sources; infer mode uses production sources.
        Output schema is identical across modes (D-12).

        Args:
            df: Input DataFrame with source columns.
            mode: "train" (OOF-safe) or "infer" (production).

        Returns:
            Investment feature DataFrame with identity cols + if_* cols + missing indicators.

        Raises:
            ValueError: If mode is invalid or required source is missing.
        """
        # (a) Mode validation (D-11)
        if mode not in ("train", "infer"):
            raise ValueError(
                f"mode must be 'train' or 'infer', got '{mode}'"
            )

        # Build columns into a dict to avoid DataFrame fragmentation
        columns: dict[str, np.ndarray | pd.Series] = {}

        # (b) Identity columns
        for col in ("race_id", "umaban"):
            if col in df.columns:
                columns[col] = df[col].values

        # (c) Pass 1: resolve all source-based features first (non-derived)
        # This ensures derived features can reference columns from any category.
        missing_indicators: list[str] = []
        derived_specs: list[InvestmentFeatureSpec] = []

        for cat in CATEGORY_ORDER:
            for spec in FEATURE_SPECS.values():
                if spec.category != cat:
                    continue

                is_derived = (
                    len(spec.train_sources) == 0
                    and len(spec.infer_sources) == 0
                )

                if is_derived:
                    derived_specs.append(spec)
                    continue

                # Resolve from source columns
                series = self._resolve_source(df, spec, mode)
                columns[spec.name] = series.astype(spec.dtype).values

                if spec.missing_indicator is not None:
                    columns[spec.missing_indicator] = series.isna().astype("int8").values
                    missing_indicators.append(spec.missing_indicator)

        # Build initial result DataFrame from dict (avoids fragmentation)
        result = pd.DataFrame(columns, index=df.index)

        # (c cont.) Pass 2: compute derived features
        # Each derived column is added to result immediately so subsequent
        # derived features can reference it (e.g. if_abs_logit_gap needs if_logit_gap).
        for spec in derived_specs:
            series = self._compute_derived(spec.name, result, df)
            if series is None:
                if spec.required:
                    raise ValueError(
                        f"Required derived feature '{spec.name}' "
                        f"has no computation defined"
                    )
                series = pd.Series(
                    spec.default_value, index=df.index, dtype="float64"
                )

            result[spec.name] = series.astype(spec.dtype).values

            if spec.missing_indicator is not None:
                result[spec.missing_indicator] = series.isna().astype("int8").values
                missing_indicators.append(spec.missing_indicator)

        # (f) Leakage validation
        validate_no_post_race_leakage(result.columns.tolist())

        # (g) Fix column order: identity + CATEGORY_ORDER + missing indicators
        ordered_cols: list[str] = []
        for col in ("race_id", "umaban"):
            if col in result.columns:
                ordered_cols.append(col)
        # Add specs in CATEGORY_ORDER then definition order
        for cat in CATEGORY_ORDER:
            for spec in FEATURE_SPECS.values():
                if spec.category == cat and spec.name in result.columns:
                    if spec.name not in ordered_cols:
                        ordered_cols.append(spec.name)
        # Add missing indicators last
        for mi in missing_indicators:
            if mi not in ordered_cols:
                ordered_cols.append(mi)

        result = result[ordered_cols]

        # (h) Sort by race_id, umaban for determinism (D-27)
        result = result.sort_values(
            ["race_id", "umaban"], kind="mergesort"
        ).reset_index(drop=True)

        return result

    def build_train_frame(
        self, df: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Build investment feature frame in train mode (D-10).

        Convenience wrapper for build_frame(df, mode="train").
        """
        return self.build_frame(df, mode="train", **kwargs)

    def build_inference_frame(
        self, df: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Build investment feature frame in infer mode (D-10).

        Convenience wrapper for build_frame(df, mode="infer").
        """
        return self.build_frame(df, mode="infer", **kwargs)


# Module-level convenience function
def build_frame(
    df: pd.DataFrame,
    mode: Literal["train", "infer"],
) -> pd.DataFrame:
    """Module-level convenience function for building investment feature frame.

    Creates a default InvestmentFeatureFrameBuilder and delegates to build_frame().
    """
    builder = InvestmentFeatureFrameBuilder()
    return builder.build_frame(df, mode)
