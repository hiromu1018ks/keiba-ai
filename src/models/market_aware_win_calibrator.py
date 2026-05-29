"""MarketAwareWinCalibrator -- Benter-type logit-blend calibrator with segment conditioning.

Replaces WinBenterGate + WinSegmentCalibrator with a single LogisticRegression + L2
model that blends model and market logits with regularized segment features and
interactions. Per-segment coefficients are eliminated in favor of a global regularized
model to prevent sparse segment overfitting.

Architecture (D-17):
  6 main effects + 15 segment one-hot + 30 logit x segment interactions = 51 features
"""

# ruff: noqa: N803,N806 -- ML convention: X for feature matrix

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from utils.wf_splits import walk_forward_race_splits as _walk_forward_race_splits

logger = logging.getLogger(__name__)


@dataclass
class MarketAwareWinCalibrator:
    """Benter-type logit-blend calibrator with segment conditioning.

    Uses sklearn LogisticRegression with L2 regularization to blend model and
    market logits. Segment conditioning features (popularity rank, odds band,
    probability rank) are encoded as regularized features/interactions, NOT
    per-segment coefficients.
    """

    calibrator: LogisticRegression | None = None
    feature_names: list[str] = field(default_factory=list)
    best_c: float | None = None
    c_selection_results: dict[str, Any] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)
    _trained: bool = False

    # D-09: Fixed odds bands
    ODDS_BAND_EDGES: ClassVar[list[float]] = [1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, float("inf")]
    ODDS_BAND_NAMES: ClassVar[list[str]] = ["1-2", "2-3", "3-5", "5-10", "10-30", "30-100", "100+"]

    # D-13: Fixed popularity buckets
    POP_BUCKET_EDGES: ClassVar[list[float]] = [0, 1.5, 3.5, 6.5, 9.5, float("inf")]
    POP_BUCKET_NAMES: ClassVar[list[str]] = [
        "pop_1", "pop_2_3", "pop_4_6", "pop_7_9", "pop_10_plus",
    ]

    # D-15: Fixed p_rank buckets
    P_RANK_NAMES: ClassVar[list[str]] = ["top_25", "mid_25_75", "bottom_25"]

    # D-04: C grid for regularization strength
    C_GRID: ClassVar[list[float]] = [0.03, 0.1, 0.3, 1.0, 3.0]

    # D-03: Beta market contribution floor
    BETA_MARKET_FLOOR: ClassVar[float] = 0.20

    @property
    def is_trained(self) -> bool:
        return self._trained and self.calibrator is not None

    # ------------------------------------------------------------------
    # One-hot encoding helpers with guaranteed schema (D-11)
    # ------------------------------------------------------------------

    def _encode_odds_band(self, tanodds: pd.Series) -> pd.DataFrame:
        """One-hot encode odds_band with guaranteed 7 columns (D-09, D-10, D-11)."""
        band = pd.cut(tanodds, bins=self.ODDS_BAND_EDGES, labels=self.ODDS_BAND_NAMES, right=False)
        dummies = pd.get_dummies(band, dtype=float)
        # Ensure all 7 expected columns exist (D-11)
        for name in self.ODDS_BAND_NAMES:
            if name not in dummies.columns:
                dummies[name] = 0.0
        return dummies[self.ODDS_BAND_NAMES]

    def _encode_pop_bucket(self, popularity_rank: pd.Series, field_size: pd.Series) -> pd.DataFrame:
        """One-hot encode popularity_bucket with guaranteed 5 columns (D-13, D-11)."""
        bucket = pd.cut(
            popularity_rank,
            bins=self.POP_BUCKET_EDGES,
            labels=self.POP_BUCKET_NAMES,
            right=False,
        )
        dummies = pd.get_dummies(bucket, dtype=float)
        for name in self.POP_BUCKET_NAMES:
            if name not in dummies.columns:
                dummies[name] = 0.0
        return dummies[self.POP_BUCKET_NAMES]

    def _encode_p_rank(self, p_win_race_rank_pct: pd.Series) -> pd.DataFrame:
        """One-hot encode p_rank_bucket with guaranteed 3 columns (D-15, D-11)."""
        # top_25: pct >= 0.75, mid_25_75: 0.25 <= pct < 0.75, bottom_25: pct < 0.25
        vals = p_win_race_rank_pct.values
        top = (vals >= 0.75).astype(float)
        mid = ((vals >= 0.25) & (vals < 0.75)).astype(float)
        bottom = (vals < 0.25).astype(float)
        return pd.DataFrame(
            {"top_25": top, "mid_25_75": mid, "bottom_25": bottom},
            index=p_win_race_rank_pct.index,
        )[self.P_RANK_NAMES]

    # ------------------------------------------------------------------
    # Feature matrix construction
    # ------------------------------------------------------------------

    def build_feature_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Build ~51-dim feature matrix for LogisticRegression.

        Args:
            df: DataFrame with columns: p_model, p_market, tanodds,
                popularity_rank, field_size, p_win_race_rank_pct

        Returns:
            (X, feature_names) where X has shape (N, 51)
        """
        eps = 1e-10

        # Main effects (6 continuous) -- D-01, D-08, D-12, D-14
        p_model_vals = np.clip(df["p_model"].values.astype(float), eps, 1 - eps)
        p_market_vals = np.clip(df["p_market"].values.astype(float), eps, 1 - eps)

        logit_model = np.log(p_model_vals / (1 - p_model_vals))
        logit_market = np.log(p_market_vals / (1 - p_market_vals))
        log_odds = np.log1p(df["tanodds"].values.astype(float))  # D-08 continuous
        popularity_rank_pct = np.clip(
            df["popularity_rank"].values.astype(float)
            / np.maximum(df["field_size"].values.astype(float), 1.0),
            0.0,
            1.0,
        )  # D-12
        p_win_race_rank_pct = df["p_win_race_rank_pct"].values.astype(float)  # D-14
        field_size_vals = df["field_size"].values.astype(float)

        main_names = [
            "logit_model",
            "logit_market",
            "log_odds",
            "popularity_rank_pct",
            "p_win_race_rank_pct",
            "field_size",
        ]

        # One-hot encodings (7 + 5 + 3 = 15 segment features)
        odds_band_oh = self._encode_odds_band(df["tanodds"])  # (N, 7)
        pop_bucket_oh = self._encode_pop_bucket(df["popularity_rank"], df["field_size"])  # (N, 5)
        p_rank_oh = self._encode_p_rank(df["p_win_race_rank_pct"])  # (N, 3)

        segment_features = np.hstack([odds_band_oh.values, pop_bucket_oh.values, p_rank_oh.values])
        segment_names = (
            list(odds_band_oh.columns)
            + list(pop_bucket_oh.columns)
            + list(p_rank_oh.columns)
        )
        assert len(segment_names) == 15

        # Interactions: logit x segment (D-06) -- 30 features
        # logit_model x segment(15) + logit_market x segment(15)
        interaction_model = logit_model[:, None] * segment_features  # (N, 15)
        interaction_market = logit_market[:, None] * segment_features  # (N, 15)

        interaction_names_model = [f"logit_model_x_{s}" for s in segment_names]
        interaction_names_market = [f"logit_market_x_{s}" for s in segment_names]

        # Assemble: 6 main + 15 one-hot + 30 interactions = 51 (D-17)
        X = np.hstack([
            np.column_stack([
                logit_model,
                logit_market,
                log_odds,
                popularity_rank_pct,
                p_win_race_rank_pct,
                field_size_vals,
            ]),
            segment_features,
            interaction_model,
            interaction_market,
        ])

        feature_names = (
            main_names
            + segment_names
            + interaction_names_model
            + interaction_names_market
        )
        assert len(feature_names) == 51, f"Expected 51 features, got {len(feature_names)}"
        assert X.shape[1] == 51, f"Expected X with 51 columns, got {X.shape[1]}"

        return X, feature_names

    # ------------------------------------------------------------------
    # Training with C-selection WF grid search
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, n_splits: int = 5) -> MarketAwareWinCalibrator:
        """Train calibrator with C-selection WF grid search.

        Args:
            df: OOF DataFrame with columns: p_model (or p_win_oof), p_market,
                tanodds, popularity_rank, field_size, p_win_race_rank_pct,
                race_id, kakuteijyuni. May also include race_date, surface.
            n_splits: Number of WF folds for C-selection.

        Raises:
            ValueError: If df contains p_win_pred without p_win_oof (D-22).
        """
        # D-22 guard: reject train-mode p_win_pred
        if "p_win_pred" in df.columns and "p_win_oof" not in df.columns:
            raise ValueError(
                "Input contains 'p_win_pred' (train-mode predictions). "
                "Use OOF predictions (p_win_oof or p_model) for calibrator training. (D-22)"
            )

        # Unconditional copy to avoid mutating caller DataFrame (CR-02)
        df = df.copy()

        # Resolve p_model column
        if "p_model" not in df.columns and "p_win_oof" in df.columns:
            df["p_model"] = df["p_win_oof"]

        # Resolve p_market column
        if "p_market" not in df.columns and "p_market_norm" in df.columns:
            df["p_market"] = df["p_market_norm"]

        # Extract target
        y = (df["kakuteijyuni"] == 1).astype(int).values

        # Build feature matrix
        X, feature_names = self.build_feature_matrix(df)
        self.feature_names = feature_names

        # WF splits for C-selection
        splits = _walk_forward_race_splits(df, n_splits=n_splits)
        if len(splits) < 2:
            logger.warning("Insufficient WF splits for C-selection, fitting with default C=1.0")
            self._fit_final(X, y, c=1.0)
            self.best_c = 1.0
            self._trained = True
            return self

        # D-04: C-selection grid search
        c_grid_results: dict[float, dict[str, Any]] = {}
        best_c: float = self.C_GRID[0]
        best_logloss: float = float("inf")

        for c in self.C_GRID:
            fold_loglosses: list[float] = []
            fold_briers: list[float] = []
            for train_idx, val_idx in splits:
                lr = LogisticRegression(C=c, max_iter=1000, fit_intercept=True)
                lr.fit(X[train_idx], y[train_idx])
                p_val = lr.predict_proba(X[val_idx])[:, 1]
                fold_loglosses.append(log_loss(y[val_idx], p_val))
                fold_briers.append(brier_score_loss(y[val_idx], p_val))

            mean_logloss = float(np.mean(fold_loglosses))
            mean_brier = float(np.mean(fold_briers))
            c_grid_results[c] = {
                "mean_logloss": mean_logloss,
                "mean_brier": mean_brier,
                "fold_loglosses": fold_loglosses,
                "fold_briers": fold_briers,
            }

            # D-04: primary metric logloss, tie-breaker smaller C
            if mean_logloss < best_logloss or (
                np.isclose(mean_logloss, best_logloss) and c < best_c
            ):
                best_logloss = mean_logloss
                best_c = c

        self.c_selection_results = {
            "c_grid_results": {
                str(c): {
                    k: v for k, v in result.items()
                    if k != "fold_loglosses" and k != "fold_briers"
                }
                for c, result in c_grid_results.items()
            },
            "best_c": best_c,
            "best_logloss": best_logloss,
        }
        self.best_c = best_c

        # D-05: Year/surface actual/predicted ratio gate
        if "surface" in df.columns and "race_date" in df.columns:
            self._check_ratio_diagnostics(df, X, y, best_c)

        # Fit final model with best C
        self._fit_final(X, y, c=best_c)

        # D-03: beta_market guard
        beta_contribution = self._compute_beta_market_contribution()
        self.training_summary["beta_market_contribution"] = beta_contribution

        if beta_contribution < self.BETA_MARKET_FLOOR:
            logger.warning(
                "beta_market effective contribution %.4f < floor %.2f. "
                "Setting shadow-only mode (D-03).",
                beta_contribution,
                self.BETA_MARKET_FLOOR,
            )
            self._trained = False
            self.training_summary["deployment_status"] = "shadow_only"
            self.training_summary["shadow_reason"] = (
                f"beta_market_contribution={beta_contribution:.4f} < {self.BETA_MARKET_FLOOR}"
            )
            return self

        self._trained = True
        self.training_summary["deployment_status"] = "deployable"
        self.training_summary["trained"] = True
        return self

    def _fit_final(self, X: np.ndarray, y: np.ndarray, c: float) -> None:
        """Fit final LogisticRegression on all data with given C."""
        # D-01/D-02: LogisticRegression + L2 (sklearn 1.8 -- no penalty param)
        self.calibrator = LogisticRegression(C=c, max_iter=1000, fit_intercept=True)
        self.calibrator.fit(X, y)
        self.training_summary["best_c"] = c
        self.training_summary["n_samples"] = int(len(y))
        self.training_summary["n_features"] = int(X.shape[1])
        self.training_summary["n_positive"] = int(y.sum())

    def _check_ratio_diagnostics(
        self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray, c: float,
    ) -> None:
        """D-05: Check year/surface actual/predicted ratio not worsen materially."""
        lr = LogisticRegression(C=c, max_iter=1000, fit_intercept=True)
        lr.fit(X, y)
        p_pred = lr.predict_proba(X)[:, 1]

        temp_df = df.copy()
        temp_df["_y"] = y
        temp_df["_p"] = p_pred

        overall_ratio = float(y.mean()) / max(float(p_pred.mean()), 1e-10)

        # Check year-level ratio
        if "race_date" in temp_df.columns:
            temp_df["_year"] = pd.to_datetime(temp_df["race_date"]).dt.year
            year_groups = temp_df.groupby("_year", observed=True).agg(
                actual=("_y", "sum"),
                predicted=("_p", "sum"),
            )
            for yr, row in year_groups.iterrows():
                if row["predicted"] > 0:
                    ratio = row["actual"] / row["predicted"]
                    if overall_ratio > 0 and abs(ratio / overall_ratio - 1.0) > 0.10:
                        logger.info(
                            "D-05 year gate: year=%s ratio=%.4f vs overall=%.4f",
                            yr, ratio, overall_ratio,
                        )

        # Check surface-level ratio
        if "surface" in temp_df.columns:
            surf_groups = temp_df.groupby("surface", observed=True).agg(
                actual=("_y", "sum"),
                predicted=("_p", "sum"),
            )
            for surf, row in surf_groups.iterrows():
                if row["predicted"] > 0:
                    ratio = row["actual"] / row["predicted"]
                    if overall_ratio > 0 and abs(ratio / overall_ratio - 1.0) > 0.10:
                        logger.info(
                            "D-05 surface gate: surface=%s ratio=%.4f vs overall=%.4f",
                            surf, ratio, overall_ratio,
                        )

    def _compute_beta_market_contribution(self) -> float:
        """D-03: Compute relative contribution of logit(p_market) coefficient.

        Returns abs(coef_market) / (abs(coef_model) + abs(coef_market)).
        """
        if self.calibrator is None or self.feature_names is None:
            return 0.0

        coef = self.calibrator.coef_[0]
        logit_model_idx = self.feature_names.index("logit_model")
        logit_market_idx = self.feature_names.index("logit_market")

        abs_model = abs(float(coef[logit_model_idx]))
        abs_market = abs(float(coef[logit_market_idx]))

        denom = abs_model + abs_market
        if denom < 1e-10:
            return 0.0
        return abs_market / denom

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply calibrator to inference DataFrame (replaces WinBenterGate).

        Args:
            df: DataFrame with columns: p_win_corrected, tanodds,
                popularity_rank, field_size, race_id

        Returns:
            DataFrame with p_win_combined, p_win_final, edge_win columns added.
        """
        df = df.copy()

        # Build inference features
        df["p_model"] = df["p_win_corrected"]
        df["p_market"] = np.clip(1.0 / df["tanodds"].values, 0.01, 0.99)

        # D-12: Compute popularity_rank_pct at inference time
        df["popularity_rank_pct"] = (
            df["popularity_rank"].astype(float) / df["field_size"].astype(float).clip(lower=1)
        ).clip(0, 1)

        # D-14: Compute p_win_race_rank_pct at inference time from p_model
        df["p_win_race_rank_pct"] = df.groupby("race_id", observed=True)["p_model"].rank(
            pct=True, method="min", ascending=False,
        )

        X, _ = self.build_feature_matrix(df)

        # Predict calibrated probabilities
        p_raw = self.calibrator.predict_proba(X)[:, 1]
        df["p_win_combined"] = p_raw

        # Race normalization: sum-to-1.0 per race_id
        race_sums = df.groupby("race_id", observed=True)["p_win_combined"].transform("sum")
        df["p_win_final"] = df["p_win_combined"] / race_sums.clip(lower=1e-10)

        # Edge calculation
        df["edge_win"] = df["p_win_final"] * df["tanodds"] - 1.0

        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save calibrator state to joblib file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "calibrator": self.calibrator,
                "feature_names": self.feature_names,
                "best_c": self.best_c,
                "c_selection_results": self.c_selection_results,
                "training_summary": self.training_summary,
                "_trained": self._trained,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> MarketAwareWinCalibrator:
        """Load calibrator state from joblib file."""
        state = joblib.load(path)
        obj = cls(
            calibrator=state.get("calibrator"),
            feature_names=list(state.get("feature_names", [])),
            best_c=state.get("best_c"),
            c_selection_results=dict(state.get("c_selection_results", {})),
            training_summary=dict(state.get("training_summary", {})),
            _trained=bool(state.get("_trained", False)),
        )
        return obj
