"""Final win-bet selection scoring policy.

The policy keeps the production invariant of one win bet per race. It does not
decide whether to bet; it only ranks runners within a race after the prediction
stack has produced win EV/edge columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from models.win_selection_gate import ensure_win_selection_columns

DEFAULT_LATE_ODDS_DROP_WEIGHT: float = 0.06
DEFAULT_LOG_ODDS_PENALTY: float = 0.05
DEFAULT_PROB_RANK_BONUS: float = 0.02
MAX_DEPLOYED_LATE_ODDS_DROP_WEIGHT: float = 0.12
MAX_DEPLOYED_LOG_ODDS_PENALTY: float = 0.08
MAX_DEPLOYED_PROB_RANK_BONUS: float = 0.05
MIN_POLICY_MEAN_ROI_IMPROVEMENT: float = 0.03
MAX_POLICY_YEAR_ROI_REGRESSION: float = 0.02
DEFAULT_CANDIDATE_WEIGHTS: tuple[float, ...] = (
    0.0,
    0.03,
    0.06,
    0.08,
    0.10,
    0.12,
)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _race_key(df: pd.DataFrame) -> pd.Series:
    if "race_id" in df.columns:
        return df["race_id"]
    return pd.Series("_race", index=df.index, dtype=object)


def race_zscore(values: pd.Series, race_key: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not numeric.notna().any():
        return pd.Series(0.0, index=values.index, dtype=float)
    grouped = numeric.groupby(race_key, observed=True)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((numeric - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _year_series(df: pd.DataFrame) -> pd.Series:
    if "race_date" in df.columns:
        years = pd.to_datetime(df["race_date"], errors="coerce").dt.year
        if years.notna().any():
            return years.astype("Int64")
    if "race_id" in df.columns:
        years = pd.to_numeric(df["race_id"].astype(str).str[:4], errors="coerce")
        return years.astype("Int64")
    return pd.Series(pd.NA, index=df.index, dtype="Int64")


def _top_one_by_score(df: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    scored = df.copy()
    scored["_policy_score"] = pd.to_numeric(score, errors="coerce").fillna(float("-inf"))
    if "race_id" not in scored.columns:
        return scored.sort_values("_policy_score", ascending=False).head(1)
    return (
        scored.sort_values(["race_id", "_policy_score"], ascending=[True, False])
        .groupby("race_id", as_index=False, observed=True)
        .head(1)
    )


def _roi_for_score(df: pd.DataFrame, score: pd.Series) -> float:
    if df.empty:
        return float("nan")
    selected = _top_one_by_score(df, score)
    if selected.empty:
        return float("nan")
    odds = _numeric(selected, "tanodds")
    if not odds.notna().any():
        odds = _numeric(selected, "confirmed_odds")
    if odds.dropna().quantile(0.75) > 100.0:
        odds = odds / 100.0
    returns = np.where(
        _numeric(selected, "kakuteijyuni").eq(1),
        odds.clip(lower=0.0).fillna(0.0) * 100.0,
        0.0,
    )
    return float(np.sum(returns) / (len(selected) * 100.0))


def sanitize_late_odds_drop_weight(value: Any) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return DEFAULT_LATE_ODDS_DROP_WEIGHT
    if not np.isfinite(weight):
        return DEFAULT_LATE_ODDS_DROP_WEIGHT
    if weight < 0.0 or weight > MAX_DEPLOYED_LATE_ODDS_DROP_WEIGHT:
        return DEFAULT_LATE_ODDS_DROP_WEIGHT
    return weight


def sanitize_log_odds_penalty(value: Any) -> float:
    try:
        penalty = float(value)
    except (TypeError, ValueError):
        return DEFAULT_LOG_ODDS_PENALTY
    if not np.isfinite(penalty):
        return DEFAULT_LOG_ODDS_PENALTY
    if penalty < 0.0 or penalty > MAX_DEPLOYED_LOG_ODDS_PENALTY:
        return DEFAULT_LOG_ODDS_PENALTY
    return penalty


def sanitize_prob_rank_bonus(value: Any) -> float:
    try:
        bonus = float(value)
    except (TypeError, ValueError):
        return DEFAULT_PROB_RANK_BONUS
    if not np.isfinite(bonus):
        return DEFAULT_PROB_RANK_BONUS
    if bonus < 0.0 or bonus > MAX_DEPLOYED_PROB_RANK_BONUS:
        return DEFAULT_PROB_RANK_BONUS
    return bonus


def deployed_policy_params(policy: Any | None) -> dict[str, float]:
    defaults = {
        "late_odds_drop_weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
        "log_odds_penalty": DEFAULT_LOG_ODDS_PENALTY,
        "prob_rank_bonus": DEFAULT_PROB_RANK_BONUS,
    }
    if policy is None:
        return defaults
    summary = getattr(policy, "training_summary", {}) or {}
    if not isinstance(summary, dict) or summary.get("deployable") is not True:
        return defaults
    return {
        "late_odds_drop_weight": sanitize_late_odds_drop_weight(
            getattr(policy, "late_odds_drop_weight", None)
        ),
        "log_odds_penalty": sanitize_log_odds_penalty(
            getattr(policy, "log_odds_penalty", None)
        ),
        "prob_rank_bonus": sanitize_prob_rank_bonus(getattr(policy, "prob_rank_bonus", None)),
    }


def deployed_late_odds_drop_weight(policy: Any | None) -> float:
    return deployed_policy_params(policy)["late_odds_drop_weight"]


@dataclass
class WinSelectionPolicy:
    """Race-level scoring policy for final win-bet selection.

    A positive `odds_drop_rate_30_10` means odds shortened from 30 minutes to
    10 minutes before post. The deployed policy may adjust the penalty magnitude,
    but it must not reverse the sign and reward visible late steam. Realized ROI
    tuning on sparse racing outcomes is too noisy to justify that reversal.
    """

    late_odds_drop_weight: float = DEFAULT_LATE_ODDS_DROP_WEIGHT
    log_odds_penalty: float = DEFAULT_LOG_ODDS_PENALTY
    prob_rank_bonus: float = DEFAULT_PROB_RANK_BONUS
    candidate_weights: tuple[float, ...] = DEFAULT_CANDIDATE_WEIGHTS
    is_trained: bool = False
    training_summary: dict[str, Any] = field(default_factory=dict)

    def _base_edge(self, df: pd.DataFrame) -> pd.Series:
        prepared = ensure_win_selection_columns(df)
        edge = _numeric(prepared, "win_selection_edge")
        if edge.notna().any():
            return edge
        return _numeric(prepared, "win_selection_ev") - 1.0

    def score(
        self,
        df: pd.DataFrame,
        *,
        base_edge: pd.Series | None = None,
        race_key: pd.Series | None = None,
    ) -> pd.Series:
        key = race_key if race_key is not None else _race_key(df)
        if base_edge is not None:
            base = pd.to_numeric(base_edge, errors="coerce")
        else:
            base = self._base_edge(df)
        late_drop = _numeric(df, "odds_drop_rate_30_10")
        late_drop_z = race_zscore(late_drop, key)
        odds = _numeric(df, "tanodds").clip(lower=1.0)
        log_odds = np.log1p(odds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prob = _numeric(df, "p_win_final").where(
            _numeric(df, "p_win_final").notna(),
            _numeric(df, "win_selection_prob"),
        )
        prob_rank = prob.groupby(key, observed=True).rank(
            pct=True,
            method="average",
            ascending=True,
        ).fillna(0.5)
        return (
            base
            - float(self.late_odds_drop_weight) * late_drop_z
            - float(self.log_odds_penalty) * log_odds
            + float(self.prob_rank_bonus) * prob_rank
        ).astype(float)

    def apply(
        self,
        df: pd.DataFrame,
        *,
        score_col: str = "win_market_selection_score",
    ) -> pd.DataFrame:
        prepared = ensure_win_selection_columns(df)
        key = _race_key(prepared)
        late_drop_z = race_zscore(_numeric(prepared, "odds_drop_rate_30_10"), key)
        prepared["win_late_odds_drop_z"] = late_drop_z
        odds = _numeric(prepared, "tanodds").clip(lower=1.0)
        prepared["win_log_odds"] = np.log1p(odds).replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0.0)
        prob = _numeric(prepared, "p_win_final").where(
            _numeric(prepared, "p_win_final").notna(),
            _numeric(prepared, "win_selection_prob"),
        )
        prepared["win_model_prob_rank"] = prob.groupby(key, observed=True).rank(
            pct=True,
            method="average",
            ascending=True,
        ).fillna(0.5)
        prepared["win_log_odds_penalty"] = self.log_odds_penalty
        prepared["win_prob_rank_bonus"] = self.prob_rank_bonus
        prepared[score_col] = self.score(prepared, race_key=key)
        if "race_id" in prepared.columns:
            prepared["selected_rank_by_win_market_score"] = prepared[score_col].groupby(
                prepared["race_id"], observed=True
            ).rank(method="first", ascending=False)
        else:
            prepared["selected_rank_by_win_market_score"] = prepared[score_col].rank(
                method="first", ascending=False
            )
        return prepared

    def train(self, df: pd.DataFrame) -> WinSelectionPolicy:
        prepared = ensure_win_selection_columns(df)
        required = {"race_id", "kakuteijyuni", "tanodds", "win_selection_edge"}
        if not required.issubset(prepared.columns):
            self.is_trained = False
            self.training_summary = {"reason": "missing_required_columns"}
            return self

        years = _year_series(prepared)
        valid = years.notna()
        prepared = prepared.loc[valid].copy()
        years = years.loc[valid].astype(int)
        if prepared.empty:
            self.is_trained = False
            self.training_summary = {"reason": "no_valid_years"}
            return self

        key = _race_key(prepared)
        base = self._base_edge(prepared)
        late_drop_z = race_zscore(_numeric(prepared, "odds_drop_rate_30_10"), key)
        odds = _numeric(prepared, "tanodds").clip(lower=1.0)
        log_odds = np.log1p(odds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prob = _numeric(prepared, "p_win_final").where(
            _numeric(prepared, "p_win_final").notna(),
            _numeric(prepared, "win_selection_prob"),
        )
        prob_rank = prob.groupby(key, observed=True).rank(
            pct=True,
            method="average",
            ascending=True,
        ).fillna(0.5)
        log_penalty = sanitize_log_odds_penalty(self.log_odds_penalty)
        prob_bonus = sanitize_prob_rank_bonus(self.prob_rank_bonus)

        rows: list[dict[str, Any]] = []
        for weight in sorted({sanitize_late_odds_drop_weight(w) for w in self.candidate_weights}):
            score = (
                base
                - float(weight) * late_drop_z
                - log_penalty * log_odds
                + prob_bonus * prob_rank
            )
            roi_all = _roi_for_score(prepared, score)
            year_rois: list[float] = []
            year_roi_by_year: dict[str, float] = {}
            for year in sorted(years.dropna().unique().tolist()):
                mask = years.eq(int(year))
                if not mask.any():
                    continue
                year_roi = _roi_for_score(prepared.loc[mask], score.loc[mask])
                year_rois.append(year_roi)
                year_roi_by_year[str(int(year))] = year_roi
            clean_year_rois = [r for r in year_rois if np.isfinite(r)]
            rows.append(
                {
                    "weight": float(weight),
                    "roi_all": roi_all,
                    "roi_mean_by_year": float(np.nanmean(clean_year_rois))
                    if clean_year_rois
                    else float("nan"),
                    "roi_min_by_year": float(np.nanmin(clean_year_rois))
                    if clean_year_rois
                    else float("nan"),
                    "roi_std_by_year": float(np.nanstd(clean_year_rois))
                    if clean_year_rois
                    else float("nan"),
                    "n_years": int(len(clean_year_rois)),
                    "year_rois": year_roi_by_year,
                }
            )

        result = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(
            subset=["roi_all", "roi_mean_by_year"]
        )
        if result.empty:
            self.is_trained = False
            self.training_summary = {"reason": "no_valid_candidates"}
            return self

        default = result.loc[result["weight"].eq(DEFAULT_LATE_ODDS_DROP_WEIGHT)]
        if default.empty:
            default_score = (
                base
                - DEFAULT_LATE_ODDS_DROP_WEIGHT * late_drop_z
                - log_penalty * log_odds
                + prob_bonus * prob_rank
            )
            default_year_rois: dict[str, float] = {}
            for year in sorted(years.dropna().unique().tolist()):
                mask = years.eq(int(year))
                default_year_rois[str(int(year))] = _roi_for_score(
                    prepared.loc[mask],
                    default_score.loc[mask],
                )
            default_row = {
                "weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
                "roi_all": _roi_for_score(prepared, default_score),
                "roi_mean_by_year": float(np.nanmean(list(default_year_rois.values()))),
                "roi_min_by_year": float(np.nanmin(list(default_year_rois.values()))),
                "roi_std_by_year": float(np.nanstd(list(default_year_rois.values()))),
                "n_years": len(default_year_rois),
                "year_rois": default_year_rois,
            }
            result = pd.concat([result, pd.DataFrame([default_row])], ignore_index=True)
            default = result.loc[result["weight"].eq(DEFAULT_LATE_ODDS_DROP_WEIGHT)]
        default_row = default.iloc[0]
        default_roi = float(default_row["roi_mean_by_year"])
        result["objective"] = (
            result["roi_mean_by_year"]
            - 0.15 * result["roi_std_by_year"].fillna(0.0)
            + 0.10 * result["roi_min_by_year"].fillna(result["roi_mean_by_year"])
            - 0.02 * result["weight"].abs()
        )
        best = result.sort_values(["objective", "roi_mean_by_year"], ascending=False).iloc[0]
        default_year_rois = dict(default_row["year_rois"])
        best_year_rois = dict(best["year_rois"])
        year_deltas = {
            year: float(best_year_rois.get(year, np.nan) - default_roi)
            for year, default_roi in default_year_rois.items()
            if np.isfinite(best_year_rois.get(year, np.nan)) and np.isfinite(default_roi)
        }
        mean_delta = float(best["roi_mean_by_year"] - default_roi)
        min_year_delta = min(year_deltas.values()) if year_deltas else float("-inf")
        deployable = (
            float(best["weight"]) != DEFAULT_LATE_ODDS_DROP_WEIGHT
            and int(best["n_years"]) >= 3
            and mean_delta >= MIN_POLICY_MEAN_ROI_IMPROVEMENT
            and min_year_delta >= -MAX_POLICY_YEAR_ROI_REGRESSION
        )
        fallback_reason = None
        if not deployable:
            fallback_reason = (
                "use_default_weight_until_positive_penalty_beats_default_across_years"
            )
            best = default_row

        self.late_odds_drop_weight = sanitize_late_odds_drop_weight(best["weight"])
        self.is_trained = True
        self.training_summary = {
            "selected_weight": self.late_odds_drop_weight,
            "default_weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
            "default_log_odds_penalty": log_penalty,
            "default_prob_rank_bonus": prob_bonus,
            "default_roi_mean_by_year": default_roi,
            "selected_roi_mean_by_year": float(best["roi_mean_by_year"]),
            "selected_roi_min_by_year": float(best["roi_min_by_year"]),
            "selected_roi_all": float(best["roi_all"]),
            "candidate_best_weight": float(
                result.sort_values(["objective", "roi_mean_by_year"], ascending=False).iloc[0][
                    "weight"
                ]
            ),
            "candidate_best_mean_delta_vs_default": mean_delta,
            "candidate_best_min_year_delta_vs_default": float(min_year_delta),
            "deployable": deployable,
            "fallback_reason": fallback_reason,
            "n_years": int(best["n_years"]),
            "candidates": result.sort_values("objective", ascending=False)
            .head(10)
            .to_dict(orient="records"),
        }
        return self

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "late_odds_drop_weight": self.late_odds_drop_weight,
                "log_odds_penalty": self.log_odds_penalty,
                "prob_rank_bonus": self.prob_rank_bonus,
                "candidate_weights": tuple(self.candidate_weights),
                "is_trained": self.is_trained,
                "training_summary": self.training_summary,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> WinSelectionPolicy:
        state = joblib.load(path)
        return cls(
            late_odds_drop_weight=sanitize_late_odds_drop_weight(
                state.get("late_odds_drop_weight", DEFAULT_LATE_ODDS_DROP_WEIGHT)
            ),
            log_odds_penalty=sanitize_log_odds_penalty(
                state.get("log_odds_penalty", DEFAULT_LOG_ODDS_PENALTY)
            ),
            prob_rank_bonus=sanitize_prob_rank_bonus(
                state.get("prob_rank_bonus", DEFAULT_PROB_RANK_BONUS)
            ),
            candidate_weights=tuple(state.get("candidate_weights", DEFAULT_CANDIDATE_WEIGHTS)),
            is_trained=bool(state.get("is_trained", True)),
            training_summary=dict(state.get("training_summary", {})),
        )
