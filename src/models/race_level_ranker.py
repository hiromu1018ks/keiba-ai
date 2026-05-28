"""RaceLevelRanker -- learned ranker with per-surface Ridge models.

Combines two independent Ridge regression models per surface (relevance_scorer
+ value_scorer) into a single investment_score per horse via pre-declared
fixed weights on race-level robust percentile ranks.

Architecture (D-01):
  4 Ridge models: relevance_scorer_turf/dirt, value_scorer_turf/dirt
  Combination: 0.35*rel_pct + 0.35*val_pct + 0.20*log_ev_pct - 0.10*uncertainty_pct

Shadow mode pattern follows MarketAwareWinCalibrator exactly (D-16).
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
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import ndcg_score

from utils.wf_splits import walk_forward_race_splits as _walk_forward_race_splits

logger = logging.getLogger(__name__)


@dataclass
class RaceLevelRanker:
    """Learned ranker combining relevance and value Ridge models per surface.

    Uses sklearn Ridge with L2 regularization to learn relevance (graded
    finishing position) and value (mispricing/EV) scores. Per-surface
    independent models stored as optional fields.
    """

    # Per-surface Ridge models (4 total)
    relevance_scorer_turf: Ridge | None = None
    relevance_scorer_dirt: Ridge | None = None
    value_scorer_turf: Ridge | None = None
    value_scorer_dirt: Ridge | None = None

    # Feature names for each scorer type
    relevance_feature_names: list[str] = field(default_factory=list)
    value_feature_names: list[str] = field(default_factory=list)

    # Shadow mode state
    _trained: bool = False
    training_summary: dict[str, Any] = field(default_factory=dict)

    # D-06: Alpha grid for regularization strength
    ALPHA_GRID: ClassVar[list[float]] = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    # D-23: Relevance scorer features (canonical IFF names, verified against schema_registry)
    RELEVANCE_FEATURES: ClassVar[list[str]] = [
        "if_p_win_final",
        "if_p_win_race_rank",
        "if_p_ability_win",
        "rel_p_ability_win_rank",
        "if_norm_finish_avg",
        "if_closing_index",
        "if_weighted_recent_form",
        "if_jockey_wr",
        "if_trainer_wr",
        "if_blood_surface_wr",
        "if_class_level",
        "if_surface",
        "if_distance_bin",
        "if_grade_code",
        "if_n_horses",
    ]

    # D-24: Value scorer features (canonical IFF names, verified against schema_registry)
    VALUE_FEATURES: ClassVar[list[str]] = [
        "if_logit_gap",
        "if_edge_win",
        "if_ev_calibrated",
        "if_odds_log",
        "if_odds_band_id",
        "if_odds_drop_60_10",
        "if_odds_drop_30_10",
        "if_overround",
        "if_market_entropy",
        "if_conformal_width",
        "if_ev_uncertainty_ratio",
        "if_p_win_race_rank",
        "if_n_horses",
    ]

    # Derived value features (computed at training/scoring time)
    DERIVED_VALUE_FEATURES: ClassVar[list[str]] = [
        "if_odds_rank",
        "if_abs_logit_gap",
    ]

    @property
    def is_trained(self) -> bool:
        """Shadow mode: True only when trained and primary model exists."""
        return self._trained and self.relevance_scorer_turf is not None

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    def _compute_relevance_target(self, kakuteijyuni: pd.Series) -> np.ndarray:
        """D-08: Graded relevance by finishing position.

        Maps kakuteijyuni to: 1.00 (1st), 0.55 (2nd), 0.30 (3rd),
        0.10 (4th-5th), 0.00 (otherwise).
        """
        pos = pd.to_numeric(kakuteijyuni, errors="coerce").values
        return np.select(
            [pos == 1, pos == 2, pos == 3, np.isin(pos, [4, 5])],
            [1.00, 0.55, 0.30, 0.10],
            default=0.00,
        ).astype(float)

    def _compute_value_target(self, df: pd.DataFrame) -> np.ndarray:
        """D-09: Composite value target (OOF-safe).

        value_target = clipped_log_ev + mispricing_bonus - uncertainty_penalty
        """
        # clipped_log_ev = log(calibrated_ev) clipped to [-1, 1]
        calibrated_ev = pd.to_numeric(
            df.get(
                "calibrated_ev_oof",
                df.get("ev_win_corrected", pd.Series(np.nan, index=df.index)),
            ),
            errors="coerce",
        )
        clipped_log_ev = np.log(calibrated_ev.clip(lower=1e-6)).clip(-1.0, 1.0)

        # mispricing_bonus = clipped(logit(p_model) - logit(p_market))
        p_model = pd.to_numeric(
            df.get("p_win_oof", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        ).clip(1e-10, 1 - 1e-10)
        p_market = pd.to_numeric(
            df.get("p_market_norm", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        ).clip(1e-10, 1 - 1e-10)
        logit_model = np.log(p_model / (1 - p_model))
        logit_market = np.log(p_market / (1 - p_market))
        mispricing_bonus = (logit_model - logit_market).clip(-1.0, 1.0)

        # uncertainty_penalty from conformal width
        uncertainty = pd.to_numeric(
            df.get("if_conformal_width", pd.Series(0.0, index=df.index)),
            errors="coerce",
        )
        uncertainty_penalty = uncertainty.fillna(0.0) * 0.1

        result = clipped_log_ev + mispricing_bonus - uncertainty_penalty
        return np.asarray(result.values, dtype=float)

    # ------------------------------------------------------------------
    # Feature matrix construction
    # ------------------------------------------------------------------

    def _build_relevance_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Build relevance feature matrix from curated features (D-23)."""
        df_work = df.copy()

        # Compute rel_p_ability_win_rank from if_p_ability_win via groupby rank
        has_rel_rank = "rel_p_ability_win_rank" not in df_work.columns
        has_ability = "if_p_ability_win" in df_work.columns
        if has_rel_rank and has_ability:
            df_work["rel_p_ability_win_rank"] = df_work.groupby("race_id", observed=True)[
                "if_p_ability_win"
            ].rank(pct=True, method="average")

        feature_names = []
        arrays: list[np.ndarray] = []

        for feat in self.RELEVANCE_FEATURES:
            if feat in df_work.columns:
                vals = pd.to_numeric(df_work[feat], errors="coerce").fillna(0.0).values
            else:
                logger.debug("Relevance feature '%s' not found, using zeros", feat)
                vals = np.zeros(len(df_work))
            arrays.append(vals)
            feature_names.append(feat)

        X = np.column_stack(arrays)
        return X, feature_names

    def _build_value_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Build value feature matrix from curated features (D-24)."""
        df_work = df.copy()

        # Derive if_odds_rank from if_odds_log groupby rank
        if "if_odds_rank" not in df_work.columns and "if_odds_log" in df_work.columns:
            df_work["if_odds_rank"] = df_work.groupby("race_id", observed=True)[
                "if_odds_log"
            ].rank(pct=True, method="average")

        # Derive if_abs_logit_gap from if_logit_gap
        if "if_abs_logit_gap" not in df_work.columns and "if_logit_gap" in df_work.columns:
            df_work["if_abs_logit_gap"] = pd.to_numeric(
                df_work["if_logit_gap"], errors="coerce"
            ).abs()

        feature_names: list[str] = []
        arrays: list[np.ndarray] = []

        # Core value features
        for feat in self.VALUE_FEATURES:
            if feat in df_work.columns:
                vals = pd.to_numeric(df_work[feat], errors="coerce").fillna(0.0).values
            else:
                logger.debug("Value feature '%s' not found, using zeros", feat)
                vals = np.zeros(len(df_work))
            arrays.append(vals)
            feature_names.append(feat)

        # Derived value features
        for feat in self.DERIVED_VALUE_FEATURES:
            if feat in df_work.columns:
                vals = pd.to_numeric(df_work[feat], errors="coerce").fillna(0.0).values
            else:
                logger.debug("Derived value feature '%s' not found, using zeros", feat)
                vals = np.zeros(len(df_work))
            arrays.append(vals)
            feature_names.append(feat)

        X = np.column_stack(arrays)
        return X, feature_names

    # ------------------------------------------------------------------
    # Race-level robust percentile rank (D-27)
    # ------------------------------------------------------------------

    def _race_pct_rank(self, values: pd.Series, race_id: pd.Series) -> pd.Series:
        """D-27: Race-level robust percentile rank with deterministic tie handling."""
        return values.groupby(race_id, observed=True).rank(
            pct=True, method="average", ascending=True,
        )

    # ------------------------------------------------------------------
    # Training with alpha grid selection
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, n_splits: int = 5) -> RaceLevelRanker:
        """Train per-surface Ridge models with alpha grid selection.

        Args:
            df: OOF DataFrame with columns required by RELEVANCE_FEATURES,
                VALUE_FEATURES, plus race_id, surface, kakuteijyuni,
                calibrated_ev_oof, p_win_oof, p_market_norm.
            n_splits: Number of WF folds for alpha selection.

        Returns:
            self (trained ranker).
        """
        df = df.copy()

        surfaces: list[tuple[str, int]] = [("turf", 0), ("dirt", 1)]
        for surface_name, surface_val in surfaces:
            mask = df["surface"] == surface_val
            df_surf = df.loc[mask].copy()
            # Reset index so positional indexing matches numpy array indices
            df_surf = df_surf.reset_index(drop=True)

            if len(df_surf) < 20:
                logger.warning(
                    "Insufficient data for %s (%d rows), skipping",
                    surface_name, len(df_surf),
                )
                continue

            # --- Relevance scorer ---
            rel_X, rel_features = self._build_relevance_features(df_surf)
            rel_y = self._compute_relevance_target(df_surf["kakuteijyuni"])

            best_alpha_rel = self._select_alpha_relevance(
                df_surf, rel_X, rel_y, n_splits,
            )

            ridge_rel = Ridge(alpha=best_alpha_rel)
            ridge_rel.fit(rel_X, rel_y)

            if surface_name == "turf":
                self.relevance_scorer_turf = ridge_rel
            else:
                self.relevance_scorer_dirt = ridge_rel

            self.relevance_feature_names = rel_features
            self.training_summary[f"relevance_best_alpha_{surface_name}"] = best_alpha_rel

            # --- Value scorer ---
            val_X, val_features = self._build_value_features(df_surf)
            val_y = self._compute_value_target(df_surf)

            best_alpha_val = self._select_alpha_value(
                df_surf, val_X, val_y, n_splits,
            )

            ridge_val = Ridge(alpha=best_alpha_val)
            ridge_val.fit(val_X, val_y)

            if surface_name == "turf":
                self.value_scorer_turf = ridge_val
            else:
                self.value_scorer_dirt = ridge_val

            self.value_feature_names = val_features
            self.training_summary[f"value_best_alpha_{surface_name}"] = best_alpha_val

            # --- D-11 diagnostics ---
            self._compute_diagnostics(
                df_surf, surface_name, ridge_rel, ridge_val,
                rel_X, rel_y, val_X, val_y,
            )

        self._trained = True
        self.training_summary["deployment_status"] = "shadow_only"
        self.training_summary["component_names"] = [
            "relevance_score", "value_score",
            "relevance_score_pct", "value_score_pct",
            "calibrated_log_ev_pct", "uncertainty_penalty_pct",
            "investment_score",
        ]
        self.training_summary["trained"] = True
        self.training_summary["n_samples"] = len(df)

        return self

    def _select_alpha_relevance(
        self,
        df_surf: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
    ) -> float:
        """D-06: Select best alpha for relevance scorer by NDCG@3.

        Tie-breaker: larger alpha (stronger regularization).
        """
        splits = _walk_forward_race_splits(df_surf, n_splits=n_splits)
        if len(splits) < 2:
            return self.ALPHA_GRID[-1]  # strongest regularization as default

        best_alpha = self.ALPHA_GRID[-1]
        best_metric = -np.inf

        for alpha in self.ALPHA_GRID:
            fold_ndcgs: list[float] = []
            for train_idx, val_idx in splits:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X[train_idx], y[train_idx])
                pred = ridge.predict(X[val_idx])

                # NDCG@3 evaluation per race
                ndcg_vals: list[float] = []
                df_val = df_surf.iloc[val_idx]
                for race_id in df_val["race_id"].unique():
                    race_mask = df_val["race_id"].values == race_id
                    if race_mask.sum() < 3:
                        continue
                    true_rel = y[val_idx][race_mask]
                    pred_rel = pred[race_mask]
                    # ndcg_score expects shape (1, n) or (n,)
                    if len(true_rel) >= 2:
                        ndcg_vals.append(
                            ndcg_score([true_rel], [pred_rel], k=3),
                        )
                if ndcg_vals:
                    fold_ndcgs.append(float(np.mean(ndcg_vals)))

            if not fold_ndcgs:
                continue
            mean_ndcg = float(np.mean(fold_ndcgs))

            # D-06: primary metric NDCG@3, tie-breaker larger alpha
            if mean_ndcg > best_metric or (
                np.isclose(mean_ndcg, best_metric) and alpha > best_alpha
            ):
                best_metric = mean_ndcg
                best_alpha = alpha

        return best_alpha

    def _select_alpha_value(
        self,
        df_surf: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
    ) -> float:
        """D-06: Select best alpha for value scorer by Spearman rank correlation.

        Tie-breaker: larger alpha (stronger regularization).
        """
        splits = _walk_forward_race_splits(df_surf, n_splits=n_splits)
        if len(splits) < 2:
            return self.ALPHA_GRID[-1]

        best_alpha = self.ALPHA_GRID[-1]
        best_metric = -np.inf

        for alpha in self.ALPHA_GRID:
            fold_corrs: list[float] = []
            for train_idx, val_idx in splits:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X[train_idx], y[train_idx])
                pred = ridge.predict(X[val_idx])

                true_val = y[val_idx]
                if len(true_val) >= 3 and np.std(true_val) > 1e-10 and np.std(pred) > 1e-10:
                    corr, _ = spearmanr(true_val, pred)
                    if np.isfinite(corr):
                        fold_corrs.append(abs(corr))

            if not fold_corrs:
                continue
            mean_corr = float(np.mean(fold_corrs))

            if mean_corr > best_metric or (
                np.isclose(mean_corr, best_metric) and alpha > best_alpha
            ):
                best_metric = mean_corr
                best_alpha = alpha

        return best_alpha

    def _compute_diagnostics(
        self,
        df_surf: pd.DataFrame,
        surface_name: str,
        ridge_rel: Ridge,
        ridge_val: Ridge,
        rel_X: np.ndarray,
        rel_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
    ) -> None:
        """D-11: Compute binary is_win diagnostics per surface."""
        # Predict relevance scores
        rel_scores = ridge_rel.predict(rel_X)

        df_diag = df_surf.copy()
        df_diag["_rel_score"] = rel_scores

        # Rank by relevance_score within each race (descending = best first)
        df_diag["_rel_rank"] = df_diag.groupby("race_id", observed=True)["_rel_score"].rank(
            ascending=False, method="first",
        )

        # top1_win_rate: fraction of races where ranker top-1 horse actually won
        top1 = df_diag[df_diag["_rel_rank"] == 1.0]
        top1_win_rate = float((top1["kakuteijyuni"] == 1).mean())

        # ndcg_at_3: using relevance_target as true relevance
        ndcg_vals: list[float] = []
        for race_id in df_diag["race_id"].unique():
            race_mask = df_diag["race_id"] == race_id
            race_df = df_diag.loc[race_mask]
            if len(race_df) < 3:
                continue
            true_rel = rel_y[race_mask.values]
            pred_rel = rel_scores[race_mask.values]
            if len(true_rel) >= 2:
                ndcg_vals.append(ndcg_score([true_rel], [pred_rel], k=3))
        ndcg_at_3 = float(np.mean(ndcg_vals)) if ndcg_vals else 0.0

        # rank_of_actual_winner: mean rank (1-based) of kakuteijyuni==1 horse
        actual_winners = df_diag[df_diag["kakuteijyuni"] == 1]
        rank_of_actual_winner = float(actual_winners["_rel_rank"].mean())

        # top3_contains_winner: fraction where actual winner is in top-3
        top3_contains = 0.0
        for race_id in df_diag["race_id"].unique():
            race_mask = df_diag["race_id"] == race_id
            race_df = df_diag.loc[race_mask]
            winner_rank = race_df.loc[race_df["kakuteijyuni"] == 1, "_rel_rank"]
            if len(winner_rank) > 0 and winner_rank.iloc[0] <= 3.0:
                top3_contains += 1.0
        n_races = df_diag["race_id"].nunique()
        top3_contains_winner = top3_contains / max(n_races, 1)

        self.training_summary[f"{surface_name}_diagnostics"] = {
            "top1_win_rate": top1_win_rate,
            "ndcg_at_3": ndcg_at_3,
            "rank_of_actual_winner": rank_of_actual_winner,
            "top3_contains_winner": top3_contains_winner,
        }

    # ------------------------------------------------------------------
    # Scoring (inference)
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all runners in race(s), adding investment_score columns.

        Shadow mode: if not trained, returns df unchanged (D-16).

        Args:
            df: DataFrame with IFF features and race_id column.

        Returns:
            DataFrame with investment_score columns added.
        """
        # Shadow mode guard (D-16)
        if not self.is_trained:
            return df

        df = df.copy()

        # Determine surface for each row (0=turf, 1=dirt)
        surfaces = df["if_surface"].unique()

        # Initialize output columns
        df["relevance_score"] = np.nan
        df["value_score"] = np.nan

        for surface_val in surfaces:
            surface_val_f = float(surface_val)
            surface_name = "turf" if surface_val_f == 0 else "dirt"
            mask = df["if_surface"] == surface_val

            # Select appropriate Ridge models
            ridge_rel = getattr(self, f"relevance_scorer_{surface_name}", None)
            ridge_val = getattr(self, f"value_scorer_{surface_name}", None)

            if ridge_rel is None or ridge_val is None:
                logger.warning(
                    "No %s models available, skipping scoring for %d rows",
                    surface_name, mask.sum(),
                )
                continue

            df_surf = df.loc[mask]

            # Relevance scoring
            rel_X, _ = self._build_relevance_features(df_surf)
            df.loc[mask, "relevance_score"] = ridge_rel.predict(rel_X)

            # Value scoring
            val_X, _ = self._build_value_features(df_surf)
            df.loc[mask, "value_score"] = ridge_val.predict(val_X)

        # Compute calibrated_log_ev from if_ev_calibrated
        ev_cal = pd.to_numeric(
            df.get("if_ev_calibrated", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )
        calibrated_log_ev = np.log(ev_cal.clip(lower=1e-6))

        # Uncertainty from if_conformal_width
        uncertainty = pd.to_numeric(
            df.get("if_conformal_width", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0.0)

        # D-27: Race-level robust percentile ranks
        df["relevance_score_pct"] = self._race_pct_rank(
            df["relevance_score"].fillna(0.0), df["race_id"],
        )
        df["value_score_pct"] = self._race_pct_rank(
            df["value_score"].fillna(0.0), df["race_id"],
        )
        df["calibrated_log_ev_pct"] = self._race_pct_rank(
            calibrated_log_ev.fillna(0.0), df["race_id"],
        )
        df["uncertainty_penalty_pct"] = self._race_pct_rank(
            uncertainty, df["race_id"],
        )

        # D-03: Fixed weight combination
        df["investment_score"] = (
            0.35 * df["relevance_score_pct"]
            + 0.35 * df["value_score_pct"]
            + 0.20 * df["calibrated_log_ev_pct"]
            - 0.10 * df["uncertainty_penalty_pct"]
        )

        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save ranker state to joblib file (follows MAWC pattern)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "relevance_scorer_turf": self.relevance_scorer_turf,
                "relevance_scorer_dirt": self.relevance_scorer_dirt,
                "value_scorer_turf": self.value_scorer_turf,
                "value_scorer_dirt": self.value_scorer_dirt,
                "relevance_feature_names": self.relevance_feature_names,
                "value_feature_names": self.value_feature_names,
                "training_summary": self.training_summary,
                "_trained": self._trained,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> RaceLevelRanker:
        """Load ranker state from joblib file."""
        state = joblib.load(path)
        obj = cls(
            relevance_scorer_turf=state.get("relevance_scorer_turf"),
            relevance_scorer_dirt=state.get("relevance_scorer_dirt"),
            value_scorer_turf=state.get("value_scorer_turf"),
            value_scorer_dirt=state.get("value_scorer_dirt"),
            relevance_feature_names=list(state.get("relevance_feature_names", [])),
            value_feature_names=list(state.get("value_feature_names", [])),
            training_summary=dict(state.get("training_summary", {})),
            _trained=bool(state.get("_trained", False)),
        )
        return obj
