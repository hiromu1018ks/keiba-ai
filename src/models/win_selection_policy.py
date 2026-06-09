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

DEFAULT_LATE_ODDS_DROP_WEIGHT: float = 0.09
DEFAULT_LOG_ODDS_PENALTY: float = 0.08
DEFAULT_PROB_RANK_BONUS: float = 0.01
DEFAULT_EV_TAIL_PENALTY_WEIGHT: float = 0.0
DEFAULT_EV_TAIL_THRESHOLD: float = 1.2
DEFAULT_MARKET_RISK_PENALTY_WEIGHT: float = 0.10
DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT: float = 0.06
DEFAULT_DIRT_LOG_ODDS_PENALTY: float = 0.05
DEFAULT_DIRT_PROB_RANK_BONUS: float = 0.02
DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT: float = 0.0
MAX_DEPLOYED_LATE_ODDS_DROP_WEIGHT: float = 0.12
MAX_DEPLOYED_LOG_ODDS_PENALTY: float = 0.08
MAX_DEPLOYED_PROB_RANK_BONUS: float = 0.05
MAX_DEPLOYED_EV_TAIL_PENALTY_WEIGHT: float = 1.2
MAX_DEPLOYED_MARKET_RISK_PENALTY_WEIGHT: float = 0.15
MIN_POLICY_MEAN_ROI_IMPROVEMENT: float = 0.05
# The yearly guard below rejects any candidate that loses to the default in an
# OOF year, so a smaller mean edge is acceptable for deployment. This keeps
# stable tail shrinkage available without fitting to a single validation year.
MIN_TAIL_SHRINKAGE_MEAN_ROI_IMPROVEMENT: float = 0.005
MIN_TAIL_SHRINKAGE_YEAR_ROI: float = 0.80
MAX_POLICY_YEAR_ROI_REGRESSION: float = 0.0
MIN_POLICY_DEPLOY_ROI_ALL: float = 0.85
DEFAULT_CANDIDATE_WEIGHTS: tuple[float, ...] = (
    0.0,
    0.03,
    0.06,
    0.08,
    0.09,
    0.10,
    0.12,
)
DEFAULT_CANDIDATE_EV_TAIL_PENALTIES: tuple[float, ...] = (
    0.0,
    0.15,
    0.30,
    0.50,
    0.75,
    1.00,
)
DEFAULT_CANDIDATE_LOG_ODDS_PENALTIES: tuple[float, ...] = (
    0.0,
    0.03,
    0.05,
    0.06,
    0.08,
)
DEFAULT_CANDIDATE_PROB_RANK_BONUSES: tuple[float, ...] = (
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
)
DEFAULT_CANDIDATE_MARKET_RISK_WEIGHTS: tuple[float, ...] = (
    0.0,
    0.05,
    0.08,
    0.10,
)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _race_key(df: pd.DataFrame) -> pd.Series:
    if "race_id" in df.columns:
        return df["race_id"]
    return pd.Series("_race", index=df.index, dtype=object)


def _surface_series(df: pd.DataFrame) -> pd.Series:
    if "surface" not in df.columns:
        return pd.Series("turf", index=df.index, dtype=object)
    return df["surface"].astype(str).str.lower()


def _dirt_mask(df: pd.DataFrame) -> pd.Series:
    return _surface_series(df).eq("dirt")


def _detect_surface(df: pd.DataFrame) -> str:
    """Detect which surface the training data belongs to (majority rule)."""
    if "surface" not in df.columns:
        return "turf"
    vals = df["surface"].astype(str).str.lower()
    if vals.eq("dirt").mean() >= 0.5:
        return "dirt"
    return "turf"


def _surface_defaults(df: pd.DataFrame) -> dict[str, float]:
    is_dirt = bool(_dirt_mask(df).mean() >= 0.5)
    if is_dirt:
        return {
            "late_odds_drop_weight": DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
            "log_odds_penalty": DEFAULT_DIRT_LOG_ODDS_PENALTY,
            "prob_rank_bonus": DEFAULT_DIRT_PROB_RANK_BONUS,
            "market_risk_penalty_weight": DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
        }
    return {
        "late_odds_drop_weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
        "log_odds_penalty": DEFAULT_LOG_ODDS_PENALTY,
        "prob_rank_bonus": DEFAULT_PROB_RANK_BONUS,
        "market_risk_penalty_weight": DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
    }


def surface_param_series(
    df: pd.DataFrame,
    *,
    turf_value: float,
    dirt_value: float,
) -> pd.Series:
    return pd.Series(float(turf_value), index=df.index, dtype=float).mask(
        _dirt_mask(df),
        float(dirt_value),
    )


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
    if "win_return_unit" in selected.columns:
        unit_return = _numeric(selected, "win_return_unit")
        if unit_return.notna().any():
            return float(unit_return.clip(lower=0.0).fillna(0.0).sum() / len(selected))
    if "win_return" in selected.columns:
        yen_return = _numeric(selected, "win_return")
        if yen_return.notna().any():
            return float(yen_return.clip(lower=0.0).fillna(0.0).sum() / (len(selected) * 100.0))

    odds = _numeric(selected, "confirmed_odds")
    if not odds.notna().any():
        odds = _numeric(selected, "tanodds")
    if odds.dropna().quantile(0.75) > 100.0:
        odds = odds / 100.0
    returns = np.where(
        _numeric(selected, "kakuteijyuni").eq(1),
        odds.clip(lower=0.0).fillna(0.0),
        0.0,
    )
    return float(np.sum(returns) / len(selected))


def _candidate_year_deltas(
    candidate_year_rois: Any,
    default_year_rois: dict[str, float],
) -> dict[str, float]:
    if not isinstance(candidate_year_rois, dict):
        return {}

    deltas: dict[str, float] = {}
    for year, default_roi in default_year_rois.items():
        try:
            candidate_roi = float(candidate_year_rois.get(year, np.nan))
            default_value = float(default_roi)
        except (TypeError, ValueError):
            continue
        if np.isfinite(candidate_roi) and np.isfinite(default_value):
            deltas[str(year)] = candidate_roi - default_value
    return deltas


def _annotate_policy_deployability(
    result: pd.DataFrame,
    *,
    default_row: pd.Series,
    default_late_weight: float,
    default_log_odds: float = DEFAULT_LOG_ODDS_PENALTY,
    default_prob_rank: float = DEFAULT_PROB_RANK_BONUS,
    default_market_risk: float = DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
) -> pd.DataFrame:
    """Add deployment diagnostics and reject candidates with any yearly regression."""
    annotated = result.copy()
    default_year_rois = dict(default_row["year_rois"])
    default_roi = float(default_row["roi_mean_by_year"])

    mean_deltas: list[float] = []
    min_year_deltas: list[float] = []
    year_delta_maps: list[dict[str, float]] = []
    for _, row in annotated.iterrows():
        year_deltas = _candidate_year_deltas(row.get("year_rois", {}), default_year_rois)
        year_delta_maps.append(year_deltas)
        mean_deltas.append(float(row["roi_mean_by_year"] - default_roi))
        min_year_deltas.append(min(year_deltas.values()) if year_deltas else float("-inf"))

    annotated["mean_delta_vs_default"] = mean_deltas
    annotated["min_year_delta_vs_default"] = min_year_deltas
    annotated["year_delta_vs_default"] = year_delta_maps

    # Detect whether any coefficient changed from the surface default.
    changed = annotated["weight"].astype(float).ne(float(default_late_weight)) | annotated[
        "ev_tail_penalty_weight"
    ].astype(float).ne(DEFAULT_EV_TAIL_PENALTY_WEIGHT)
    if "w_log" in annotated.columns:
        changed = changed | annotated["w_log"].astype(float).ne(float(default_log_odds))
    if "w_prob" in annotated.columns:
        changed = changed | annotated["w_prob"].astype(float).ne(float(default_prob_rank))
    if "w_risk" in annotated.columns:
        changed = changed | annotated["w_risk"].astype(float).ne(float(default_market_risk))
    annotated["candidate_changed"] = changed

    annotated["changes_late_weight"] = (
        annotated["weight"].astype(float).ne(float(default_late_weight))
    )
    annotated["uses_tail_shrinkage"] = (
        annotated["ev_tail_penalty_weight"].astype(float).gt(DEFAULT_EV_TAIL_PENALTY_WEIGHT)
    )
    annotated["roi_floor_met"] = annotated["roi_all"].astype(float).ge(MIN_POLICY_DEPLOY_ROI_ALL)
    annotated["stable_tail_shrinkage_met"] = (
        annotated["uses_tail_shrinkage"]
        & annotated["mean_delta_vs_default"].ge(MIN_TAIL_SHRINKAGE_MEAN_ROI_IMPROVEMENT)
        & annotated["min_year_delta_vs_default"].ge(-MAX_POLICY_YEAR_ROI_REGRESSION)
        & annotated["roi_min_by_year"].astype(float).ge(MIN_TAIL_SHRINKAGE_YEAR_ROI)
        & annotated["roi_floor_met"]
        & (
            ~annotated["changes_late_weight"]
            | annotated["mean_delta_vs_default"].ge(MIN_POLICY_MEAN_ROI_IMPROVEMENT)
        )
    )
    non_tail_deployable = (
        annotated["candidate_changed"]
        & ~annotated["uses_tail_shrinkage"]
        & annotated["n_years"].astype(int).ge(3)
        & annotated["mean_delta_vs_default"].ge(MIN_POLICY_MEAN_ROI_IMPROVEMENT)
        & annotated["min_year_delta_vs_default"].ge(-MAX_POLICY_YEAR_ROI_REGRESSION)
        & annotated["roi_floor_met"]
    )
    tail_deployable = (
        annotated["candidate_changed"]
        & annotated["uses_tail_shrinkage"]
        & annotated["n_years"].astype(int).ge(3)
        & annotated["stable_tail_shrinkage_met"]
    )
    annotated["deployable_candidate"] = non_tail_deployable | tail_deployable
    return annotated


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


def sanitize_ev_tail_penalty_weight(value: Any) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return DEFAULT_EV_TAIL_PENALTY_WEIGHT
    if not np.isfinite(weight):
        return DEFAULT_EV_TAIL_PENALTY_WEIGHT
    if weight < 0.0 or weight > MAX_DEPLOYED_EV_TAIL_PENALTY_WEIGHT:
        return DEFAULT_EV_TAIL_PENALTY_WEIGHT
    return weight


def sanitize_market_risk_penalty_weight(value: Any) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MARKET_RISK_PENALTY_WEIGHT
    if not np.isfinite(weight):
        return DEFAULT_MARKET_RISK_PENALTY_WEIGHT
    if weight < 0.0 or weight > MAX_DEPLOYED_MARKET_RISK_PENALTY_WEIGHT:
        return DEFAULT_MARKET_RISK_PENALTY_WEIGHT
    return weight


def ev_tail_pressure(
    df: pd.DataFrame,
    *,
    threshold: float = DEFAULT_EV_TAIL_THRESHOLD,
) -> pd.Series:
    """Continuous OOF-learned overconfidence pressure for high-EV tails.

    The pressure is a ranking shrinkage feature, not a filter. It is strongest
    when EV is high while the implied hit probability is low and odds are long.
    """
    prepared = ensure_win_selection_columns(df)
    ev = _numeric(prepared, "win_selection_ev")
    if not ev.notna().any():
        ev = _numeric(prepared, "win_selection_edge") + 1.0
    ev_excess = (ev - float(threshold)).clip(lower=0.0, upper=4.0)

    odds = _numeric(prepared, "tanodds").clip(lower=1.0)
    odds_tail = (
        (np.log1p(odds) - np.log1p(30.0)) / max(1e-9, np.log1p(100.0) - np.log1p(30.0))
    ).clip(lower=0.0, upper=1.0)

    prob = _numeric(prepared, "p_win_final").where(
        _numeric(prepared, "p_win_final").notna(),
        _numeric(prepared, "win_selection_prob"),
    )
    low_prob_tail = ((0.08 - prob) / 0.08).clip(lower=0.0, upper=1.0)

    pressure = np.log1p(ev_excess) * (1.0 + 0.50 * odds_tail + 0.25 * low_prob_tail)
    return pd.to_numeric(pressure, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def market_residual_score(
    df: pd.DataFrame,
    *,
    race_key: pd.Series | None = None,
) -> pd.Series:
    """Model win-probability edge over the normalized public market."""
    existing = _numeric(df, "win_market_residual")
    prepared = ensure_win_selection_columns(df)
    key = race_key if race_key is not None else _race_key(prepared)
    prob = _numeric(prepared, "p_win_final").where(
        _numeric(prepared, "p_win_final").notna(),
        _numeric(prepared, "p_win_final_oof"),
    )
    prob = prob.where(prob.notna(), _numeric(prepared, "win_selection_prob"))
    odds = _numeric(prepared, "tanodds")
    market_raw = pd.Series(np.nan, index=prepared.index, dtype=float)
    valid_odds = odds.gt(0.0) & odds.notna()
    market_raw.loc[valid_odds] = (1.0 / odds.loc[valid_odds]).clip(0.01, 0.99)
    market_sum = market_raw.groupby(key, observed=True).transform("sum")
    market_norm = market_raw / market_sum.replace(0.0, np.nan)
    residual = prob - market_norm
    if residual.notna().any():
        return residual.replace([np.inf, -np.inf], np.nan).astype(float)
    if existing.notna().any():
        return existing.replace([np.inf, -np.inf], np.nan).astype(float)
    return _numeric(prepared, "win_market_logit_edge").replace([np.inf, -np.inf], np.nan)


def market_risk_penalty(
    df: pd.DataFrame,
    *,
    race_key: pd.Series | None = None,
    tail_ev: pd.Series | None = None,
) -> pd.Series:
    """Smooth ranking penalty for unstable high-odds/low-probability EV tails."""
    prepared = ensure_win_selection_columns(df)
    existing = _numeric(prepared, "win_market_risk_penalty")
    if existing.notna().any():
        return existing.fillna(0.0).astype(float)

    key = race_key if race_key is not None else _race_key(prepared)
    odds = _numeric(prepared, "tanodds")
    ev = _numeric(prepared, "win_selection_ev")
    prob = _numeric(prepared, "p_win_final").where(
        _numeric(prepared, "p_win_final").notna(),
        _numeric(prepared, "p_win_final_oof"),
    )
    prob = prob.where(prob.notna(), _numeric(prepared, "win_selection_prob"))
    prob_rank = prob.groupby(key, observed=True).rank(method="first", ascending=False)
    tail_source = tail_ev if tail_ev is not None else ev
    penalty = (
        odds.ge(30.0).fillna(False).astype(float) * 0.20
        + ev.ge(5.0).fillna(False).astype(float) * 0.30
        + pd.to_numeric(tail_source, errors="coerce").ge(1.5).fillna(False).astype(float) * 0.25
        + (odds.ge(10.0).fillna(False) & prob.lt(0.05).fillna(False)).astype(float) * 0.20
        + prob_rank.gt(8.0).fillna(False).astype(float) * 0.10
        + prob.lt(0.03).fillna(False).astype(float) * 0.10
    )
    return pd.to_numeric(penalty, errors="coerce").fillna(0.0).astype(float)


def surface_aware_selection_base(
    df: pd.DataFrame,
    *,
    race_key: pd.Series | None = None,
    fallback_edge: pd.Series | None = None,
) -> pd.Series:
    """Use market residual for turf and model edge for dirt."""
    prepared = ensure_win_selection_columns(df)
    residual = market_residual_score(prepared, race_key=race_key)
    if fallback_edge is not None:
        edge = pd.to_numeric(fallback_edge, errors="coerce")
    else:
        edge = _numeric(prepared, "win_selection_edge")
        edge = edge.where(edge.notna(), _numeric(prepared, "win_selection_ev") - 1.0)
    if not residual.notna().any():
        return edge
    base = residual.where(residual.notna(), edge)
    return base.where(~_dirt_mask(prepared), edge).astype(float)


def deployed_policy_params(policy: Any | None) -> dict[str, float]:
    defaults = {
        "late_odds_drop_weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
        "log_odds_penalty": DEFAULT_LOG_ODDS_PENALTY,
        "prob_rank_bonus": DEFAULT_PROB_RANK_BONUS,
        "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
        "ev_tail_threshold": DEFAULT_EV_TAIL_THRESHOLD,
        "market_risk_penalty_weight": DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
        "dirt_late_odds_drop_weight": DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
        "dirt_log_odds_penalty": DEFAULT_DIRT_LOG_ODDS_PENALTY,
        "dirt_prob_rank_bonus": DEFAULT_DIRT_PROB_RANK_BONUS,
        "dirt_market_risk_penalty_weight": DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
    }
    if policy is None:
        return defaults
    summary = getattr(policy, "training_summary", {}) or {}
    if not isinstance(summary, dict) or summary.get("deployable") is not True:
        return defaults
    surface = getattr(policy, "surface", "turf")
    if surface == "dirt":
        return {
            "late_odds_drop_weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
            "log_odds_penalty": DEFAULT_LOG_ODDS_PENALTY,
            "prob_rank_bonus": DEFAULT_PROB_RANK_BONUS,
            "ev_tail_penalty_weight": sanitize_ev_tail_penalty_weight(
                getattr(policy, "ev_tail_penalty_weight", None)
            ),
            "ev_tail_threshold": DEFAULT_EV_TAIL_THRESHOLD,
            "market_risk_penalty_weight": DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
            "dirt_late_odds_drop_weight": sanitize_late_odds_drop_weight(
                getattr(policy, "late_odds_drop_weight", None)
            ),
            "dirt_log_odds_penalty": sanitize_log_odds_penalty(
                getattr(policy, "log_odds_penalty", None)
            ),
            "dirt_prob_rank_bonus": sanitize_prob_rank_bonus(
                getattr(policy, "prob_rank_bonus", None)
            ),
            "dirt_market_risk_penalty_weight": sanitize_market_risk_penalty_weight(
                getattr(policy, "market_risk_penalty_weight", None)
            ),
        }
    return {
        "late_odds_drop_weight": sanitize_late_odds_drop_weight(
            getattr(policy, "late_odds_drop_weight", None)
        ),
        "log_odds_penalty": sanitize_log_odds_penalty(getattr(policy, "log_odds_penalty", None)),
        "prob_rank_bonus": sanitize_prob_rank_bonus(getattr(policy, "prob_rank_bonus", None)),
        "ev_tail_penalty_weight": sanitize_ev_tail_penalty_weight(
            getattr(policy, "ev_tail_penalty_weight", None)
        ),
        "ev_tail_threshold": DEFAULT_EV_TAIL_THRESHOLD,
        "market_risk_penalty_weight": sanitize_market_risk_penalty_weight(
            getattr(policy, "market_risk_penalty_weight", None)
        ),
        "dirt_late_odds_drop_weight": DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
        "dirt_log_odds_penalty": DEFAULT_DIRT_LOG_ODDS_PENALTY,
        "dirt_prob_rank_bonus": DEFAULT_DIRT_PROB_RANK_BONUS,
        "dirt_market_risk_penalty_weight": DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
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

    The ``surface`` attribute records which surface this policy was trained on.
    Learned coefficients are mapped to the correct surface slot during
    persistence and deployment; old models without ``surface`` default to "turf".
    """

    surface: str = "turf"
    late_odds_drop_weight: float = DEFAULT_LATE_ODDS_DROP_WEIGHT
    log_odds_penalty: float = DEFAULT_LOG_ODDS_PENALTY
    prob_rank_bonus: float = DEFAULT_PROB_RANK_BONUS
    ev_tail_penalty_weight: float = DEFAULT_EV_TAIL_PENALTY_WEIGHT
    market_risk_penalty_weight: float = DEFAULT_MARKET_RISK_PENALTY_WEIGHT
    candidate_weights: tuple[float, ...] = DEFAULT_CANDIDATE_WEIGHTS
    candidate_ev_tail_penalties: tuple[float, ...] = DEFAULT_CANDIDATE_EV_TAIL_PENALTIES
    candidate_log_odds_penalties: tuple[float, ...] = DEFAULT_CANDIDATE_LOG_ODDS_PENALTIES
    candidate_prob_rank_bonuses: tuple[float, ...] = DEFAULT_CANDIDATE_PROB_RANK_BONUSES
    candidate_market_risk_weights: tuple[float, ...] = DEFAULT_CANDIDATE_MARKET_RISK_WEIGHTS
    is_trained: bool = False
    training_summary: dict[str, Any] = field(default_factory=dict)

    def _param_series(
        self,
        df: pd.DataFrame,
        *,
        learned: float,
        turf_default: float,
        dirt_default: float,
    ) -> pd.Series:
        """Apply learned coefficient to the policy's surface, defaults to the other.

        For a turf-trained policy, ``learned`` goes to turf rows and ``dirt_default``
        to dirt rows.  For a dirt-trained policy the mapping is reversed.
        """
        if self.surface == "dirt":
            return surface_param_series(df, turf_value=turf_default, dirt_value=learned)
        return surface_param_series(df, turf_value=learned, dirt_value=dirt_default)

    def _base_edge(self, df: pd.DataFrame) -> pd.Series:
        prepared = ensure_win_selection_columns(df)
        return surface_aware_selection_base(prepared)

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
        prob_rank = (
            prob.groupby(key, observed=True)
            .rank(
                pct=True,
                method="average",
                ascending=True,
            )
            .fillna(0.5)
        )
        risk_penalty = market_risk_penalty(df, race_key=key)
        late_weight = self._param_series(
            df,
            learned=self.late_odds_drop_weight,
            turf_default=DEFAULT_LATE_ODDS_DROP_WEIGHT,
            dirt_default=DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
        )
        log_penalty = self._param_series(
            df,
            learned=self.log_odds_penalty,
            turf_default=DEFAULT_LOG_ODDS_PENALTY,
            dirt_default=DEFAULT_DIRT_LOG_ODDS_PENALTY,
        )
        prob_bonus = self._param_series(
            df,
            learned=self.prob_rank_bonus,
            turf_default=DEFAULT_PROB_RANK_BONUS,
            dirt_default=DEFAULT_DIRT_PROB_RANK_BONUS,
        )
        risk_weight = self._param_series(
            df,
            learned=self.market_risk_penalty_weight,
            turf_default=DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
            dirt_default=DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
        )
        return (
            base
            - late_weight * late_drop_z
            - log_penalty * log_odds
            + prob_bonus * prob_rank
            - float(self.ev_tail_penalty_weight) * ev_tail_pressure(df)
            - risk_weight * risk_penalty
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
        prepared["win_log_odds"] = (
            np.log1p(odds)
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )
        prob = _numeric(prepared, "p_win_final").where(
            _numeric(prepared, "p_win_final").notna(),
            _numeric(prepared, "win_selection_prob"),
        )
        prepared["win_model_prob_rank"] = (
            prob.groupby(key, observed=True)
            .rank(
                pct=True,
                method="average",
                ascending=True,
            )
            .fillna(0.5)
        )
        prepared["win_late_odds_drop_weight"] = self._param_series(
            prepared,
            learned=self.late_odds_drop_weight,
            turf_default=DEFAULT_LATE_ODDS_DROP_WEIGHT,
            dirt_default=DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
        )
        prepared["win_log_odds_penalty"] = self._param_series(
            prepared,
            learned=self.log_odds_penalty,
            turf_default=DEFAULT_LOG_ODDS_PENALTY,
            dirt_default=DEFAULT_DIRT_LOG_ODDS_PENALTY,
        )
        prepared["win_prob_rank_bonus"] = self._param_series(
            prepared,
            learned=self.prob_rank_bonus,
            turf_default=DEFAULT_PROB_RANK_BONUS,
            dirt_default=DEFAULT_DIRT_PROB_RANK_BONUS,
        )
        prepared["win_ev_tail_pressure"] = ev_tail_pressure(prepared)
        prepared["win_ev_tail_penalty_weight"] = self.ev_tail_penalty_weight
        prepared["win_market_risk_penalty"] = market_risk_penalty(prepared, race_key=key)
        prepared["win_market_risk_penalty_weight"] = self._param_series(
            prepared,
            learned=self.market_risk_penalty_weight,
            turf_default=DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
            dirt_default=DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
        )
        prepared[score_col] = self.score(prepared, race_key=key)
        if "race_id" in prepared.columns:
            prepared["selected_rank_by_win_market_score"] = (
                prepared[score_col]
                .groupby(prepared["race_id"], observed=True)
                .rank(method="first", ascending=False)
            )
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

        self.surface = _detect_surface(prepared)
        key = _race_key(prepared)
        base = self._base_edge(prepared)
        late_drop_z = race_zscore(_numeric(prepared, "odds_drop_rate_30_10"), key)
        odds = _numeric(prepared, "tanodds").clip(lower=1.0)
        log_odds = np.log1p(odds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prob = _numeric(prepared, "p_win_final").where(
            _numeric(prepared, "p_win_final").notna(),
            _numeric(prepared, "win_selection_prob"),
        )
        prob_rank = (
            prob.groupby(key, observed=True)
            .rank(
                pct=True,
                method="average",
                ascending=True,
            )
            .fillna(0.5)
        )
        defaults = _surface_defaults(prepared)
        default_late = defaults["late_odds_drop_weight"]
        default_log = defaults["log_odds_penalty"]
        default_prob = defaults["prob_rank_bonus"]
        default_risk = defaults["market_risk_penalty_weight"]
        ev_tail = ev_tail_pressure(prepared)
        risk_penalty = market_risk_penalty(prepared, race_key=key)

        def _eval(
            w_late: float, w_log: float, w_prob: float, w_tail: float, w_risk: float
        ) -> dict[str, Any]:
            score = (
                base
                - w_late * late_drop_z
                - w_log * log_odds
                + w_prob * prob_rank
                - w_tail * ev_tail
                - w_risk * risk_penalty
            )
            roi_all = _roi_for_score(prepared, score)
            year_rois: dict[str, float] = {}
            for year in sorted(years.dropna().unique().tolist()):
                mask = years.eq(int(year))
                if mask.any():
                    year_rois[str(int(year))] = _roi_for_score(prepared.loc[mask], score.loc[mask])
            clean_year_rois = [r for r in year_rois.values() if np.isfinite(r)]
            return {
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
                "n_years": len(clean_year_rois),
                "year_rois": year_rois,
            }

        def _objective(row: dict[str, Any] | pd.Series) -> float:
            roi_mean = float(row["roi_mean_by_year"])
            roi_min = float(row.get("roi_min_by_year", roi_mean))
            roi_std = float(row.get("roi_std_by_year", 0.0) or 0.0)
            return (
                roi_mean
                - 0.15 * roi_std
                + 0.10 * roi_min
                - 0.02 * abs(float(row.get("weight", 0.0)))
                - 0.01 * abs(float(row.get("ev_tail_penalty_weight", 0.0)))
                - 0.005 * abs(float(row.get("w_log", 0.0)))
                - 0.005 * abs(float(row.get("w_prob", 0.0)))
                - 0.005 * abs(float(row.get("w_risk", 0.0)))
            )

        # ── Stage 1: 2-D grid (w_late × w_tail) with other params at defaults ──
        late_weights = sorted({sanitize_late_odds_drop_weight(w) for w in self.candidate_weights})
        tail_weights = sorted(
            {sanitize_ev_tail_penalty_weight(w) for w in self.candidate_ev_tail_penalties}
        )

        all_rows: list[dict[str, Any]] = []
        for w_late in late_weights:
            for w_tail in tail_weights:
                m = _eval(w_late, default_log, default_prob, w_tail, default_risk)
                all_rows.append(
                    {
                        "weight": w_late,
                        "ev_tail_penalty_weight": w_tail,
                        "w_log": default_log,
                        "w_prob": default_prob,
                        "w_risk": default_risk,
                        **m,
                    }
                )

        # Pick best stage-1 by objective
        best_s1_obj = -np.inf
        best_s1 = {
            "w_late": default_late,
            "w_tail": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
        }
        for row in all_rows:
            obj = _objective(row)
            if obj > best_s1_obj:
                best_s1_obj = obj
                best_s1 = {
                    "w_late": float(row["weight"]),
                    "w_tail": float(row["ev_tail_penalty_weight"]),
                }

        # ── Stage 2: coordinate descent for w_log, w_prob, w_risk ──
        log_cands = sorted(
            {sanitize_log_odds_penalty(w) for w in self.candidate_log_odds_penalties}
        )
        prob_cands = sorted({sanitize_prob_rank_bonus(w) for w in self.candidate_prob_rank_bonuses})
        risk_cands = sorted(
            {sanitize_market_risk_penalty_weight(w) for w in self.candidate_market_risk_weights}
        )

        current_best: dict[str, float] = {
            "w_late": best_s1["w_late"],
            "w_tail": best_s1["w_tail"],
            "w_log": default_log,
            "w_prob": default_prob,
            "w_risk": default_risk,
        }

        for coord_name, coord_cands in [
            ("w_log", log_cands),
            ("w_prob", prob_cands),
            ("w_risk", risk_cands),
        ]:
            best_val = current_best[coord_name]
            best_obj = -np.inf
            for val in coord_cands:
                trial = {**current_best, coord_name: val}
                m = _eval(
                    trial["w_late"],
                    trial["w_log"],
                    trial["w_prob"],
                    trial["w_tail"],
                    trial["w_risk"],
                )
                candidate_row: dict[str, Any] = {
                    "weight": trial["w_late"],
                    "ev_tail_penalty_weight": trial["w_tail"],
                    **trial,
                    **m,
                }
                all_rows.append(candidate_row)
                obj = _objective(candidate_row)
                if obj > best_obj:
                    best_obj = obj
                    best_val = val
            current_best[coord_name] = best_val

        # ── Build result DataFrame ──
        result = (
            pd.DataFrame(all_rows)
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["roi_all", "roi_mean_by_year"])
        )
        if result.empty:
            self.is_trained = False
            self.training_summary = {"reason": "no_valid_candidates"}
            return self

        result["objective"] = result.apply(_objective, axis=1)

        # Ensure the all-defaults candidate exists for guard comparison
        default_metrics = _eval(
            default_late,
            default_log,
            default_prob,
            DEFAULT_EV_TAIL_PENALTY_WEIGHT,
            default_risk,
        )
        default_row_data: dict[str, Any] = {
            "weight": default_late,
            "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
            "w_log": default_log,
            "w_prob": default_prob,
            "w_risk": default_risk,
            **default_metrics,
        }
        has_default = (
            result["weight"].eq(default_late)
            & result["ev_tail_penalty_weight"].eq(DEFAULT_EV_TAIL_PENALTY_WEIGHT)
            & result["w_log"].eq(default_log)
            & result["w_prob"].eq(default_prob)
            & result["w_risk"].eq(default_risk)
        )
        if not has_default.any():
            result = pd.concat([result, pd.DataFrame([default_row_data])], ignore_index=True)

        default_row = result.loc[
            result["weight"].eq(default_late)
            & result["ev_tail_penalty_weight"].eq(DEFAULT_EV_TAIL_PENALTY_WEIGHT)
            & result["w_log"].eq(default_log)
            & result["w_prob"].eq(default_prob)
            & result["w_risk"].eq(default_risk)
        ].iloc[0]

        result = _annotate_policy_deployability(
            result,
            default_row=default_row,
            default_late_weight=default_late,
            default_log_odds=default_log,
            default_prob_rank=default_prob,
            default_market_risk=default_risk,
        )
        candidate_best = result.sort_values(
            ["objective", "roi_mean_by_year"],
            ascending=False,
        ).iloc[0]
        deployable_candidates = result.loc[result["deployable_candidate"].fillna(False)].copy()
        deployable = not deployable_candidates.empty
        fallback_reason: str | None = None
        if deployable:
            best = deployable_candidates.sort_values(
                ["objective", "roi_mean_by_year"],
                ascending=False,
            ).iloc[0]
        else:
            if not bool(candidate_best["roi_floor_met"]) and not bool(
                candidate_best["stable_tail_shrinkage_met"]
            ):
                fallback_reason = "use_default_policy_until_oof_roi_floor_is_met"
            else:
                fallback_reason = "use_default_policy_until_candidate_beats_default_in_every_year"
            best = default_row

        self.late_odds_drop_weight = sanitize_late_odds_drop_weight(best["weight"])
        self.log_odds_penalty = sanitize_log_odds_penalty(best.get("w_log", default_log))
        self.prob_rank_bonus = sanitize_prob_rank_bonus(best.get("w_prob", default_prob))
        self.ev_tail_penalty_weight = sanitize_ev_tail_penalty_weight(
            best["ev_tail_penalty_weight"]
        )
        self.market_risk_penalty_weight = sanitize_market_risk_penalty_weight(
            best.get("w_risk", default_risk)
        )
        self.is_trained = True

        year_deltas = _candidate_year_deltas(
            best.get("year_rois", {}), dict(default_row.get("year_rois", {}))
        )

        self.training_summary = {
            "surface": self.surface,
            # ── selected parameters ──
            "selected_weight": self.late_odds_drop_weight,
            "selected_log_odds_penalty": self.log_odds_penalty,
            "selected_prob_rank_bonus": self.prob_rank_bonus,
            "selected_ev_tail_penalty_weight": self.ev_tail_penalty_weight,
            "selected_market_risk_penalty_weight": self.market_risk_penalty_weight,
            # ── default parameters ──
            "default_weight": default_late,
            "default_log_odds_penalty": default_log,
            "default_prob_rank_bonus": default_prob,
            "default_ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
            "default_market_risk_penalty_weight": default_risk,
            "ev_tail_threshold": DEFAULT_EV_TAIL_THRESHOLD,
            # ── guard thresholds ──
            "min_policy_deploy_roi_all": MIN_POLICY_DEPLOY_ROI_ALL,
            "min_tail_shrinkage_mean_roi_improvement": MIN_TAIL_SHRINKAGE_MEAN_ROI_IMPROVEMENT,
            "min_tail_shrinkage_year_roi": MIN_TAIL_SHRINKAGE_YEAR_ROI,
            # ── default ROI ──
            "default_roi_mean_by_year": float(default_row["roi_mean_by_year"]),
            "default_roi_all": float(default_row["roi_all"]),
            "default_year_rois": dict(default_row.get("year_rois", {})),
            # ── selected ROI ──
            "selected_roi_mean_by_year": float(best["roi_mean_by_year"]),
            "selected_roi_min_by_year": float(best["roi_min_by_year"]),
            "selected_roi_all": float(best["roi_all"]),
            "selected_year_rois": dict(best.get("year_rois", {})),
            # ── per-year deltas ──
            "selected_mean_delta_vs_default": float(best.get("mean_delta_vs_default", 0.0)),
            "selected_min_year_delta_vs_default": float(best.get("min_year_delta_vs_default", 0.0)),
            "selected_year_deltas_vs_default": year_deltas,
            # ── candidate-best (best objective regardless of deployability) ──
            "candidate_best_weight": float(candidate_best["weight"]),
            "candidate_best_ev_tail_penalty_weight": float(
                candidate_best["ev_tail_penalty_weight"]
            ),
            "candidate_best_mean_delta_vs_default": float(candidate_best["mean_delta_vs_default"]),
            "candidate_best_min_year_delta_vs_default": float(
                candidate_best["min_year_delta_vs_default"]
            ),
            "roi_floor_met": bool(candidate_best["roi_floor_met"]),
            "stable_tail_shrinkage_met": bool(candidate_best["stable_tail_shrinkage_met"]),
            "deployable": deployable,
            "deployable_reason": (
                "candidate_beats_default_in_all_oof_years"
                if deployable
                else (fallback_reason or "unknown")
            ),
            "fallback_reason": fallback_reason,
            "n_years": int(best["n_years"]),
            "candidate_count": len(result),
            "candidates": result.sort_values("objective", ascending=False)
            .head(10)
            .to_dict(orient="records"),
        }
        return self

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "surface": self.surface,
                "late_odds_drop_weight": self.late_odds_drop_weight,
                "log_odds_penalty": self.log_odds_penalty,
                "prob_rank_bonus": self.prob_rank_bonus,
                "ev_tail_penalty_weight": self.ev_tail_penalty_weight,
                "market_risk_penalty_weight": self.market_risk_penalty_weight,
                "candidate_weights": tuple(self.candidate_weights),
                "candidate_ev_tail_penalties": tuple(self.candidate_ev_tail_penalties),
                "candidate_log_odds_penalties": tuple(self.candidate_log_odds_penalties),
                "candidate_prob_rank_bonuses": tuple(self.candidate_prob_rank_bonuses),
                "candidate_market_risk_weights": tuple(self.candidate_market_risk_weights),
                "is_trained": self.is_trained,
                "training_summary": self.training_summary,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> WinSelectionPolicy:
        state = joblib.load(path)
        return cls(
            surface=str(state.get("surface", "turf")),
            late_odds_drop_weight=sanitize_late_odds_drop_weight(
                state.get("late_odds_drop_weight", DEFAULT_LATE_ODDS_DROP_WEIGHT)
            ),
            log_odds_penalty=sanitize_log_odds_penalty(
                state.get("log_odds_penalty", DEFAULT_LOG_ODDS_PENALTY)
            ),
            prob_rank_bonus=sanitize_prob_rank_bonus(
                state.get("prob_rank_bonus", DEFAULT_PROB_RANK_BONUS)
            ),
            ev_tail_penalty_weight=sanitize_ev_tail_penalty_weight(
                state.get("ev_tail_penalty_weight", DEFAULT_EV_TAIL_PENALTY_WEIGHT)
            ),
            market_risk_penalty_weight=sanitize_market_risk_penalty_weight(
                state.get("market_risk_penalty_weight", DEFAULT_MARKET_RISK_PENALTY_WEIGHT)
            ),
            candidate_weights=tuple(state.get("candidate_weights", DEFAULT_CANDIDATE_WEIGHTS)),
            candidate_ev_tail_penalties=tuple(
                state.get("candidate_ev_tail_penalties", DEFAULT_CANDIDATE_EV_TAIL_PENALTIES)
            ),
            candidate_log_odds_penalties=tuple(
                state.get("candidate_log_odds_penalties", DEFAULT_CANDIDATE_LOG_ODDS_PENALTIES)
            ),
            candidate_prob_rank_bonuses=tuple(
                state.get("candidate_prob_rank_bonuses", DEFAULT_CANDIDATE_PROB_RANK_BONUSES)
            ),
            candidate_market_risk_weights=tuple(
                state.get("candidate_market_risk_weights", DEFAULT_CANDIDATE_MARKET_RISK_WEIGHTS)
            ),
            is_trained=bool(state.get("is_trained", True)),
            training_summary=dict(state.get("training_summary", {})),
        )
