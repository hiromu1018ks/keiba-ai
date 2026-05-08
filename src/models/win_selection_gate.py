"""Learned gate for win bet selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from logging import getLogger

logger = getLogger(__name__)


def _numeric_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def build_win_selection_ev(df: pd.DataFrame) -> pd.Series:
    lower_ev = _numeric_or_nan(df, "EV_lower_win_corrected")
    corrected_ev = _numeric_or_nan(df, "ev_win_corrected")
    direct_ev = _numeric_or_nan(df, "ev_win")

    if corrected_ev.notna().any():
        selection_ev = lower_ev.where(lower_ev.notna(), corrected_ev)
        safety_floor = corrected_ev * 0.85
        return pd.concat([selection_ev, safety_floor], axis=1).max(axis=1).astype(float)
    if lower_ev.notna().any():
        return lower_ev.astype(float)
    return direct_ev.astype(float)


def ensure_win_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "win_selection_ev" not in prepared.columns:
        if "EV_lower_win_corrected" in prepared.columns or "ev_win_corrected" in prepared.columns:
            prepared["win_selection_ev"] = build_win_selection_ev(prepared)
        elif "edge_win" in prepared.columns:
            prepared["win_selection_ev"] = _numeric_or_nan(prepared, "edge_win") + 1.0
        else:
            prepared["win_selection_ev"] = _numeric_or_nan(prepared, "ev_win")

    if "win_selection_edge" not in prepared.columns:
        prepared["win_selection_edge"] = _numeric_or_nan(prepared, "win_selection_ev") - 1.0

    if "win_selection_prob" not in prepared.columns:
        if "p_win_final" in prepared.columns:
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_final")
        elif "p_win_combined" in prepared.columns:
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_combined")
        else:
            prepared["win_selection_prob"] = _numeric_or_nan(prepared, "p_win_corrected")

    return prepared


def _quantile_edges(series: pd.Series, n_bins: int) -> list[float]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return [-np.inf, np.inf]

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    values = np.unique(clean.quantile(quantiles).to_numpy(dtype=float))
    if len(values) <= 1:
        value = float(values[0]) if len(values) == 1 else 0.0
        return [-np.inf, value, np.inf]

    edges = values.tolist()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bucketize(series: pd.Series, edges: list[float]) -> pd.Series:
    if len(edges) < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=edges,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    ).astype("Float64")


def _smoothed_score(
    mean_value: float,
    sample_count: float,
    global_value: float,
    prior_weight: float,
) -> float:
    numerator = mean_value * sample_count + global_value * prior_weight
    denominator = sample_count + prior_weight
    return float(numerator / denominator)


class WinSelectionGateModel:
    """OOF-learned gate and reranker for final win bet selection."""

    SCORE_COL = "win_gate_score"
    PASS_COL = "win_gate_pass"
    RANK_COL = "win_gate_rank"
    GAP_COL = "win_gate_score_gap"
    MARKET_CONDITION_COL = "market_condition_score"
    AGGRESSIVE_STRENGTH_COL = "aggressive_strength"
    AGGRESSIVE_TIER_COL = "aggressive_tier"
    RUNNER_UP_SCORE_COL = "runner_up_gate_score"
    RUNNER_UP_GAP_COL = "runner_up_gate_score_gap"
    RUNNER_UP_PROB_COL = "runner_up_win_selection_prob"
    RUNNER_UP_EDGE_COL = "runner_up_win_selection_edge"
    RUNNER_UP_ODDS_COL = "runner_up_tanoddslow"
    SOFT_PROB_BUFFER = 0.01
    SOFT_EDGE_BUFFER = 0.02
    SOFT_ODDS_BUFFER = 1.0

    def __init__(
        self,
        *,
        n_bins: int = 6,
        prior_weight: float = 24.0,
        min_train_races: int = 200,
        min_fold_races: int = 80,
        max_folds: int = 4,
    ) -> None:
        self.n_bins = n_bins
        self.prior_weight = prior_weight
        self.min_train_races = min_train_races
        self.min_fold_races = min_fold_races
        self.max_folds = max_folds

        self.threshold: float = 0.0
        self.global_score: float = 1.0
        self.prob_edges: list[float] = []
        self.edge_edges: list[float] = []
        self.odds_edges: list[float] = []
        self.combo_scores: dict[tuple[int, int, int], float] = {}
        self.pair_scores: dict[tuple[str, int, int], float] = {}
        self.single_scores: dict[tuple[str, int], float] = {}
        # ODDS-03: conformal confidence edges for pair scoring
        self._confidence_edges: list[float] = []
        self.min_prob = 0.0
        self.min_edge = 0.0
        self.max_odds = float("inf")
        self.add_second_score_min = float("inf")
        self.add_second_score_gap_max = float("inf")
        self.add_second_min_prob = 0.0
        self.add_second_min_edge = 0.0
        self.add_second_max_odds = float("inf")
        self.add_second_min_market_condition = 0.0
        self.strong_aggressive_threshold = 1.0
        self.add_second_enabled = False
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    @staticmethod
    def _favorite_implied_prob(df: pd.DataFrame) -> pd.Series:
        if "race_id" not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        if "popularity_rank" not in df.columns or "odds" not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)

        popularity_rank = _numeric_or_nan(df, "popularity_rank")
        odds = _numeric_or_nan(df, "odds")
        favorite_odds = odds.where(popularity_rank.eq(1))
        favorite_odds = favorite_odds.groupby(df["race_id"], observed=True).transform("min")
        favorite_prob = pd.Series(np.nan, index=df.index, dtype=float)
        valid = favorite_odds.notna() & favorite_odds.gt(0.0)
        favorite_prob.loc[valid] = 1.0 / favorite_odds.loc[valid]
        return favorite_prob

    @classmethod
    def _compute_market_condition_score(cls, df: pd.DataFrame) -> pd.Series:
        favorite_prob = cls._favorite_implied_prob(df)
        overround = _numeric_or_nan(df, "overround")
        overround_adj = 1.0 - np.clip(overround - 0.20, 0.0, 0.15) / 0.15
        return favorite_prob * overround_adj

    def _prepare_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = ensure_win_selection_columns(df)
        required_cols = [
            "race_id",
            "race_date",
            "kakuteijyuni",
            "tanoddslow",
            "win_selection_prob",
            "win_selection_edge",
        ]
        optional_cols = [
            "surface",
            "market_entropy",
            "overround",
            "popularity_rank",
            "odds",
        ]
        missing = [col for col in required_cols if col not in prepared.columns]
        if missing:
            return pd.DataFrame(columns=required_cols)

        columns = required_cols + [col for col in optional_cols if col in prepared.columns]
        prepared = prepared[columns].copy()
        prepared["race_date"] = pd.to_datetime(prepared["race_date"], errors="coerce")
        prepared["tanoddslow"] = _numeric_or_nan(prepared, "tanoddslow")
        prepared["win_selection_prob"] = _numeric_or_nan(prepared, "win_selection_prob")
        prepared["win_selection_edge"] = _numeric_or_nan(prepared, "win_selection_edge")
        prepared["kakuteijyuni"] = _numeric_or_nan(prepared, "kakuteijyuni")
        prepared["market_entropy"] = _numeric_or_nan(prepared, "market_entropy")
        prepared["overround"] = _numeric_or_nan(prepared, "overround")
        prepared["popularity_rank"] = _numeric_or_nan(prepared, "popularity_rank")
        prepared["odds"] = _numeric_or_nan(prepared, "odds")
        if "surface" in prepared.columns:
            prepared["surface"] = prepared["surface"].fillna("unknown").astype(str)
        else:
            prepared["surface"] = "unknown"

        prepared = prepared.dropna(
            subset=[
                "race_id",
                "race_date",
                "tanoddslow",
                "win_selection_prob",
                "win_selection_edge",
                "kakuteijyuni",
            ]
        )
        if prepared.empty:
            return prepared

        prepared = prepared[prepared["tanoddslow"] > 0].copy()
        prepared["log_win_odds"] = np.log1p(prepared["tanoddslow"])
        prepared["realized_win_roi"] = np.where(
            prepared["kakuteijyuni"] == 1,
            prepared["tanoddslow"],
            0.0,
        )
        prepared[self.MARKET_CONDITION_COL] = self._compute_market_condition_score(prepared)
        return prepared.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    def _build_score_tables(self, df: pd.DataFrame) -> dict[str, Any]:
        work = df.copy()
        prob_edges = _quantile_edges(work["win_selection_prob"], self.n_bins)
        edge_edges = _quantile_edges(work["win_selection_edge"], self.n_bins)
        odds_edges = _quantile_edges(work["log_win_odds"], self.n_bins)

        work["_prob_bin"] = _bucketize(work["win_selection_prob"], prob_edges)
        work["_edge_bin"] = _bucketize(work["win_selection_edge"], edge_edges)
        work["_odds_bin"] = _bucketize(work["log_win_odds"], odds_edges)

        global_score = float(work["realized_win_roi"].mean())

        combo_scores: dict[tuple[int, int, int], float] = {}
        grouped = (
            work.groupby(
                ["_prob_bin", "_edge_bin", "_odds_bin"],
                observed=True,
            )["realized_win_roi"]
            .agg(["mean", "count"])
            .reset_index()
        )
        for _, data in grouped.iterrows():
            if (
                pd.isna(data["_prob_bin"])
                or pd.isna(data["_edge_bin"])
                or pd.isna(data["_odds_bin"])
            ):
                continue
            key = (int(data["_prob_bin"]), int(data["_edge_bin"]), int(data["_odds_bin"]))
            combo_scores[key] = _smoothed_score(
                float(data["mean"]),
                float(data["count"]),
                global_score,
                self.prior_weight,
            )

        pair_specs = {
            "prob_edge": ["_prob_bin", "_edge_bin"],
            "prob_odds": ["_prob_bin", "_odds_bin"],
            "edge_odds": ["_edge_bin", "_odds_bin"],
        }
        pair_scores: dict[tuple[str, int, int], float] = {}
        for prefix, cols in pair_specs.items():
            grouped = (
                work.groupby(cols, observed=True)["realized_win_roi"]
                .agg(["mean", "count"])
                .reset_index()
            )
            for _, data in grouped.iterrows():
                if pd.isna(data[cols[0]]) or pd.isna(data[cols[1]]):
                    continue
                key = (prefix, int(data[cols[0]]), int(data[cols[1]]))
                pair_scores[key] = _smoothed_score(
                    float(data["mean"]),
                    float(data["count"]),
                    global_score,
                    self.prior_weight,
                )

        single_specs = {
            "prob": "_prob_bin",
            "edge": "_edge_bin",
            "odds": "_odds_bin",
        }
        single_scores: dict[tuple[str, int], float] = {}
        for prefix, col in single_specs.items():
            grouped = (
                work.groupby(col, observed=True)["realized_win_roi"]
                .agg(["mean", "count"])
                .reset_index()
            )
            for _, data in grouped.iterrows():
                if pd.isna(data[col]):
                    continue
                key = (prefix, int(data[col]))
                single_scores[key] = _smoothed_score(
                    float(data["mean"]),
                    float(data["count"]),
                    global_score,
                    self.prior_weight,
                )

        # ODDS-03: conformal confidence pair scores (D-08)
        confidence_col = work.get("conformal_confidence_score")
        if confidence_col is not None:
            confidence_edges = _quantile_edges(
                confidence_col.fillna(0.0),
                max(3, self.n_bins // 2),  # Fewer bins to avoid sparsity
            )
            work["_confidence_bin"] = _bucketize(
                confidence_col.fillna(0.0),
                confidence_edges,
            )

            confidence_pair_specs = {
                "confidence_prob": ["_confidence_bin", "_prob_bin"],
                "confidence_edge": ["_confidence_bin", "_edge_bin"],
            }
            for prefix, cols in confidence_pair_specs.items():
                grouped = (
                    work.groupby(cols, observed=True)["realized_win_roi"]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                for _, data in grouped.iterrows():
                    if pd.isna(data[cols[0]]) or pd.isna(data[cols[1]]):
                        continue
                    if data["count"] < 5:
                        continue
                    key = (prefix, int(data[cols[0]]), int(data[cols[1]]))
                    pair_scores[key] = _smoothed_score(
                        float(data["mean"]),
                        float(data["count"]),
                        global_score,
                        self.prior_weight,
                    )
        else:
            confidence_edges = []

        return {
            "prob_edges": prob_edges,
            "edge_edges": edge_edges,
            "odds_edges": odds_edges,
            "confidence_edges": confidence_edges,
            "global_score": global_score,
            "combo_scores": combo_scores,
            "pair_scores": pair_scores,
            "single_scores": single_scores,
        }

    @classmethod
    def _score_row_from_tables(
        cls,
        prob_bin: float,
        edge_bin: float,
        odds_bin: float,
        *,
        global_score: float,
        combo_scores: dict[tuple[int, int, int], float],
        pair_scores: dict[tuple[str, int, int], float],
        single_scores: dict[tuple[str, int], float],
        confidence_bin: float | None = None,
    ) -> float:
        if pd.isna(prob_bin) or pd.isna(edge_bin) or pd.isna(odds_bin):
            return global_score

        prob_key = int(prob_bin)
        edge_key = int(edge_bin)
        odds_key = int(odds_bin)

        combo_key = (prob_key, edge_key, odds_key)
        if combo_key in combo_scores:
            return combo_scores[combo_key]

        pair_values = [
            pair_scores[key]
            for key in [
                ("prob_edge", prob_key, edge_key),
                ("prob_odds", prob_key, odds_key),
                ("edge_odds", edge_key, odds_key),
            ]
            if key in pair_scores
        ]

        # ODDS-03: confidence pair fallback
        if confidence_bin is not None and not pd.isna(confidence_bin):
            conf_key = int(confidence_bin)
            for key in [
                ("confidence_prob", conf_key, prob_key),
                ("confidence_edge", conf_key, edge_key),
            ]:
                if key in pair_scores:
                    pair_values.append(pair_scores[key])

        if pair_values:
            return float(np.mean(pair_values))

        single_values = [
            single_scores[key]
            for key in [
                ("prob", prob_key),
                ("edge", edge_key),
                ("odds", odds_key),
            ]
            if key in single_scores
        ]
        if single_values:
            return float(np.mean(single_values))

        return global_score

    @classmethod
    def _score_frame_from_tables(
        cls,
        df: pd.DataFrame,
        tables: dict[str, Any],
    ) -> pd.Series:
        prepared = ensure_win_selection_columns(df)
        prob_bins = _bucketize(prepared["win_selection_prob"], list(tables["prob_edges"]))
        edge_bins = _bucketize(prepared["win_selection_edge"], list(tables["edge_edges"]))
        odds_values = np.log1p(_numeric_or_nan(prepared, "tanoddslow").clip(lower=0.0))
        odds_bins = _bucketize(odds_values, list(tables["odds_edges"]))

        # ODDS-03: confidence bin for pair scoring
        confidence_edges = tables.get("confidence_edges", [])
        if confidence_edges and "conformal_confidence_score" in prepared.columns:
            confidence_bins = _bucketize(
                _numeric_or_nan(prepared, "conformal_confidence_score").fillna(0.0),
                list(confidence_edges),
            )
        else:
            confidence_bins = pd.Series(np.nan, index=prepared.index, dtype=float)

        scores = [
            cls._score_row_from_tables(
                prob_bin,
                edge_bin,
                odds_bin,
                global_score=float(tables["global_score"]),
                combo_scores=dict(tables["combo_scores"]),
                pair_scores=dict(tables["pair_scores"]),
                single_scores=dict(tables["single_scores"]),
                confidence_bin=conf_bin,
            )
            for prob_bin, edge_bin, odds_bin, conf_bin in zip(
                prob_bins, edge_bins, odds_bins, confidence_bins, strict=False
            )
        ]
        return pd.Series(scores, index=prepared.index, dtype=float)

    def _score_frame(self, df: pd.DataFrame) -> pd.Series:
        tables = {
            "prob_edges": self.prob_edges,
            "edge_edges": self.edge_edges,
            "odds_edges": self.odds_edges,
            "confidence_edges": self._confidence_edges,
            "global_score": self.global_score,
            "combo_scores": self.combo_scores,
            "pair_scores": self.pair_scores,
            "single_scores": self.single_scores,
        }
        return self._score_frame_from_tables(df, tables)

    def _build_walk_forward_folds(self, n_races: int) -> list[tuple[int, int]]:
        if n_races < self.min_train_races + self.min_fold_races:
            return []

        remaining = n_races - self.min_train_races
        fold_size = max(self.min_fold_races, remaining // max(self.max_folds, 1))
        folds: list[tuple[int, int]] = []
        train_end = self.min_train_races
        while train_end < n_races and len(folds) < self.max_folds:
            test_end = min(train_end + fold_size, n_races)
            if test_end - train_end < max(20, self.min_fold_races // 2):
                break
            folds.append((train_end, test_end))
            train_end = test_end
        return folds

    def _build_threshold_grid(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[float], list[float], list[float]]:
        prob_values = sorted(
            {
                0.08,
                0.10,
                0.12,
                0.15,
                0.18,
                *(
                    float(df["win_selection_prob"].quantile(q))
                    for q in [0.75, 0.80, 0.85, 0.90, 0.93, 0.95]
                ),
            }
        )
        edge_values = sorted(
            {
                0.0,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                *(
                    float(df["win_selection_edge"].quantile(q))
                    for q in [0.90, 0.92, 0.94, 0.96, 0.97]
                ),
            }
        )
        odds_values = sorted(
            {
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
                15.0,
                18.0,
                *(
                    float(df["tanoddslow"].quantile(q))
                    for q in [0.50, 0.60, 0.70, 0.80, 0.90]
                ),
            }
        )
        return prob_values, edge_values, odds_values

    def _surface_score(
        self,
        df: pd.DataFrame,
        min_prob: float,
        min_edge: float,
        max_odds: float,
    ) -> pd.Series:
        prob = pd.to_numeric(df["win_selection_prob"], errors="coerce")
        edge = pd.to_numeric(df["win_selection_edge"], errors="coerce")
        odds = pd.to_numeric(df["tanoddslow"], errors="coerce")

        prob_scale = max(min_prob, 0.05)
        edge_scale = max(abs(min_edge), 0.05)
        odds_scale = max(max_odds, 1.0)
        return (
            prob / prob_scale
            + (edge - min_edge) / edge_scale
            + (max_odds - odds) / odds_scale
        ).astype(float)

    def _simulate_threshold_surface(
        self,
        df: pd.DataFrame,
        min_prob: float,
        min_edge: float,
        max_odds: float,
    ) -> dict[str, float]:
        scored = df.copy()
        scored[self.SCORE_COL] = self._surface_score(scored, min_prob, min_edge, max_odds)
        candidates = scored.loc[self._pass_mask(scored, min_prob, min_edge, max_odds)].copy()
        if candidates.empty:
            return {"roi": 0.0, "profit": 0.0, "max_drawdown": float("inf"), "bets": 0.0}

        candidates = candidates.sort_values(
            ["race_id", self.SCORE_COL, "win_selection_edge", "win_selection_prob"],
            ascending=[True, False, False, False],
        )
        candidates = candidates.groupby(
            "race_id", as_index=False, sort=False, observed=True
        ).head(1)
        if candidates.empty:
            return {"roi": 0.0, "profit": 0.0, "max_drawdown": float("inf"), "bets": 0.0}

        candidates = candidates.sort_values(["race_date", "race_id"]).reset_index(drop=True)
        profit_units = candidates["realized_win_roi"] - 1.0
        equity = profit_units.cumsum()
        drawdown = (equity.cummax() - equity).fillna(0.0)
        return {
            "roi": float(candidates["realized_win_roi"].mean()),
            "profit": float(profit_units.sum()),
            "max_drawdown": float(drawdown.max()),
            "bets": float(len(candidates)),
        }

    @staticmethod
    def _pass_mask(
        df: pd.DataFrame,
        min_prob: float,
        min_edge: float,
        max_odds: float,
    ) -> pd.Series:
        prob = _numeric_or_nan(df, "win_selection_prob")
        edge = _numeric_or_nan(df, "win_selection_edge")
        odds = _numeric_or_nan(df, "tanoddslow")
        return (
            prob.ge(min_prob)
            & edge.ge(min_edge)
            & odds.gt(0.0)
            & odds.le(max_odds)
        )

    def _build_oof_scores(
        self,
        prepared: pd.DataFrame,
        race_order: pd.DataFrame,
        folds: list[tuple[int, int]],
    ) -> pd.DataFrame:
        fold_frames: list[pd.DataFrame] = []
        for train_end, test_end in folds:
            train_races = set(race_order.iloc[:train_end]["race_id"])
            test_races = set(race_order.iloc[train_end:test_end]["race_id"])
            fold_train = prepared[prepared["race_id"].isin(train_races)].copy()
            fold_test = prepared[prepared["race_id"].isin(test_races)].copy()
            if fold_train.empty or fold_test.empty:
                continue
            tables = self._build_score_tables(fold_train)
            fold_test[self.SCORE_COL] = self._score_frame_from_tables(fold_test, tables)
            fold_frames.append(fold_test)

        if not fold_frames:
            return prepared.iloc[0:0].copy()

        return (
            pd.concat(fold_frames, ignore_index=True)
            .sort_values(["race_date", "race_id"])
            .reset_index(drop=True)
        )

    def _with_rank_context(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        if self.SCORE_COL not in prepared.columns:
            prepared[self.SCORE_COL] = self._score_frame(prepared)

        scores = _numeric_or_nan(prepared, self.SCORE_COL)
        ranks = scores.groupby(prepared["race_id"], observed=True).rank(
            method="first", ascending=False
        )
        gaps = scores.groupby(prepared["race_id"], observed=True).transform("max") - scores
        prob = _numeric_or_nan(prepared, "win_selection_prob")
        edge = _numeric_or_nan(prepared, "win_selection_edge")
        odds = _numeric_or_nan(prepared, "tanoddslow")

        prepared[self.RANK_COL] = ranks
        prepared[self.GAP_COL] = gaps
        prepared[self.RUNNER_UP_SCORE_COL] = scores.where(ranks.eq(2)).groupby(
            prepared["race_id"], observed=True
        ).transform("max")
        prepared[self.RUNNER_UP_GAP_COL] = gaps.where(ranks.eq(2)).groupby(
            prepared["race_id"], observed=True
        ).transform("min")
        prepared[self.RUNNER_UP_PROB_COL] = prob.where(ranks.eq(2)).groupby(
            prepared["race_id"], observed=True
        ).transform("max")
        prepared[self.RUNNER_UP_EDGE_COL] = edge.where(ranks.eq(2)).groupby(
            prepared["race_id"], observed=True
        ).transform("max")
        prepared[self.RUNNER_UP_ODDS_COL] = odds.where(ranks.eq(2)).groupby(
            prepared["race_id"], observed=True
        ).transform("max")
        prepared[self.MARKET_CONDITION_COL] = self._compute_market_condition_score(prepared)
        return prepared

    def _primary_selection_mask(self, df: pd.DataFrame) -> pd.Series:
        prepared = df.copy()
        if self.PASS_COL not in prepared.columns:
            prepared[self.PASS_COL] = self._pass_mask(
                prepared,
                self.min_prob,
                self.min_edge,
                self.max_odds,
            )

        hard_mask = prepared[self.PASS_COL].fillna(False).astype(bool)
        soft_mask = self.soft_pass_mask(
            prepared,
            edge_floor=0.0,
            min_prob=self.min_prob,
            max_odds=self.max_odds,
            max_per_race=1,
        )
        eligible = prepared.loc[hard_mask | soft_mask].copy()
        if eligible.empty:
            return pd.Series(False, index=prepared.index, dtype=bool)

        eligible = eligible.sort_values(
            ["race_id", self.SCORE_COL, "win_selection_edge", "win_selection_prob"],
            ascending=[True, False, False, False],
        )
        selected = eligible.groupby("race_id", as_index=False, sort=False, observed=True).head(1)
        return prepared.index.isin(selected.index)

    def _fit_add_second_reranker(self, oof_scored: pd.DataFrame) -> None:
        self.add_second_enabled = False
        if oof_scored.empty:
            return

        scored = self._with_rank_context(oof_scored)
        scored[self.PASS_COL] = self._pass_mask(
            scored,
            self.min_prob,
            self.min_edge,
            self.max_odds,
        )
        primary_mask = self._primary_selection_mask(scored)
        selected_races = scored["race_id"].isin(scored.loc[primary_mask, "race_id"])
        candidates = scored.loc[selected_races & scored[self.RANK_COL].eq(2)].copy()
        if candidates.empty:
            return

        score_values = sorted(
            set(
                float(candidates[self.SCORE_COL].quantile(q))
                for q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
            )
        )
        gap_values = sorted(
            set(
                float(candidates[self.RUNNER_UP_GAP_COL].quantile(q))
                for q in [0.20, 0.30, 0.40, 0.50, 0.60]
            )
        )
        odds_values = sorted(
            set(
                float(candidates[self.RUNNER_UP_ODDS_COL].quantile(q))
                for q in [0.50, 0.60, 0.70, 0.80]
            )
            | {6.0, 8.0, 10.0, 12.0}
        )
        market_values = sorted(
            set(
                float(candidates[self.MARKET_CONDITION_COL].quantile(q))
                for q in [0.20, 0.40, 0.60]
            )
            | {0.0}
        )
        prob_values = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
        edge_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

        best_params: tuple[float, float, float, float, float, float] | None = None
        best_key = (False, -np.inf, -np.inf, -np.inf)
        min_candidates = max(20, int(len(candidates) * 0.04))

        for score_min in score_values:
            for gap_max in gap_values:
                for prob_min in prob_values:
                    for edge_min in edge_values:
                        for odds_max in odds_values:
                            for market_min in market_values:
                                mask = (
                                    candidates[self.SCORE_COL].ge(score_min)
                                    & candidates[self.RUNNER_UP_GAP_COL].le(gap_max)
                                    & candidates[self.RUNNER_UP_PROB_COL].ge(prob_min)
                                    & candidates[self.RUNNER_UP_EDGE_COL].ge(edge_min)
                                    & candidates[self.RUNNER_UP_ODDS_COL].gt(0.0)
                                    & candidates[self.RUNNER_UP_ODDS_COL].le(odds_max)
                                    & candidates[self.MARKET_CONDITION_COL].ge(market_min)
                                )
                                selected = candidates.loc[mask].copy()
                                if len(selected) < min_candidates:
                                    continue

                                profit = float((selected["realized_win_roi"] - 1.0).sum())
                                roi = float(selected["realized_win_roi"].mean())
                                key = (profit > 0, profit, roi, float(len(selected)))
                                if key > best_key:
                                    best_key = key
                                    best_params = (
                                        score_min,
                                        gap_max,
                                        prob_min,
                                        edge_min,
                                        odds_max,
                                        market_min,
                                    )

        if best_params is None:
            return

        (
            self.add_second_score_min,
            self.add_second_score_gap_max,
            self.add_second_min_prob,
            self.add_second_min_edge,
            self.add_second_max_odds,
            self.add_second_min_market_condition,
        ) = best_params
        self.strong_aggressive_threshold = 1.0
        self.add_second_enabled = True

    def train(self, df: pd.DataFrame) -> None:
        prepared = self._prepare_training_frame(df)
        if prepared.empty:
            logger.debug("WinSelectionGate: no data after preparation, skipping")
            return

        race_order = (
            prepared[["race_id", "race_date"]]
            .drop_duplicates()
            .sort_values(["race_date", "race_id"])
            .reset_index(drop=True)
        )
        folds = self._build_walk_forward_folds(len(race_order))
        if not folds:
            logger.debug(
                "WinSelectionGate: only %d races (min_train_races=%d), skipping",
                len(race_order),
                self.min_train_races,
            )
            return

        prob_grid, edge_grid, odds_grid = self._build_threshold_grid(prepared)
        total_eval_races = sum(test_end - train_end for train_end, test_end in folds)
        min_bets = max(40, int(total_eval_races * 0.015))
        best_params = None
        best_key = (False, -np.inf, -np.inf, np.inf, np.inf)

        for min_prob in prob_grid:
            for min_edge in edge_grid:
                for max_odds in odds_grid:
                    total_profit = 0.0
                    total_bets = 0.0
                    max_drawdown = 0.0
                    total_return = 0.0
                    for train_end, test_end in folds:
                        test_races = set(race_order.iloc[train_end:test_end]["race_id"])
                        fold_test = prepared[prepared["race_id"].isin(test_races)].copy()
                        if fold_test.empty:
                            continue
                        metrics = self._simulate_threshold_surface(
                            fold_test,
                            min_prob,
                            min_edge,
                            max_odds,
                        )
                        total_profit += metrics["profit"]
                        total_bets += metrics["bets"]
                        total_return += metrics["roi"] * metrics["bets"]
                        max_drawdown = max(max_drawdown, metrics["max_drawdown"])
                    if total_bets < min_bets:
                        continue

                    roi = total_return / total_bets if total_bets > 0 else 0.0
                    key = (
                        total_profit > 0,
                        total_profit,
                        roi,
                        -max_drawdown,
                        total_bets,
                    )
                    if key > best_key:
                        best_key = key
                        best_params = (min_prob, min_edge, max_odds)

        if best_params is None:
            logger.debug("WinSelectionGate: no profitable threshold combination found, skipping")
            return

        self.min_prob, self.min_edge, self.max_odds = best_params
        score_tables = self._build_score_tables(prepared)
        self.prob_edges = list(score_tables["prob_edges"])
        self.edge_edges = list(score_tables["edge_edges"])
        self.odds_edges = list(score_tables["odds_edges"])
        self._confidence_edges = list(score_tables["confidence_edges"])
        self.global_score = float(score_tables["global_score"])
        self.combo_scores = dict(score_tables["combo_scores"])
        self.pair_scores = dict(score_tables["pair_scores"])
        self.single_scores = dict(score_tables["single_scores"])
        self._trained = True

        oof_scored = self._build_oof_scores(prepared, race_order, folds)
        self._fit_add_second_reranker(oof_scored)

    def _runner_up_strength(self, df: pd.DataFrame) -> pd.Series:
        if not self.add_second_enabled:
            return pd.Series(0.0, index=df.index, dtype=float)

        components: list[pd.Series] = []
        runner_up_score = _numeric_or_nan(df, self.RUNNER_UP_SCORE_COL)
        runner_up_gap = _numeric_or_nan(df, self.RUNNER_UP_GAP_COL)
        runner_up_prob = _numeric_or_nan(df, self.RUNNER_UP_PROB_COL)
        runner_up_edge = _numeric_or_nan(df, self.RUNNER_UP_EDGE_COL)
        runner_up_odds = _numeric_or_nan(df, self.RUNNER_UP_ODDS_COL)
        market_condition = _numeric_or_nan(df, self.MARKET_CONDITION_COL)

        if np.isfinite(self.add_second_score_min) and self.add_second_score_min > 0:
            components.append(
                (runner_up_score / self.add_second_score_min).clip(lower=0.0, upper=2.0)
            )
        if np.isfinite(self.add_second_score_gap_max) and self.add_second_score_gap_max > 0:
            components.append(
                (1.0 - (runner_up_gap / self.add_second_score_gap_max)).clip(lower=0.0, upper=1.0)
            )
        if self.add_second_min_prob > 0:
            components.append(
                (runner_up_prob / self.add_second_min_prob).clip(lower=0.0, upper=2.0)
            )
        if self.add_second_min_edge > 0:
            components.append(
                (runner_up_edge / self.add_second_min_edge).clip(lower=0.0, upper=2.0)
            )
        if self.add_second_min_market_condition > 0:
            components.append(
                (market_condition / self.add_second_min_market_condition).clip(lower=0.0, upper=2.0)
            )
        if np.isfinite(self.add_second_max_odds) and self.add_second_max_odds > 0:
            odds_ratio = 1.0 - ((runner_up_odds / self.add_second_max_odds) - 1.0).clip(
                lower=0.0,
                upper=1.0,
            )
            components.append(odds_ratio.clip(lower=0.0, upper=1.0))

        if not components:
            return pd.Series(0.0, index=df.index, dtype=float)

        return pd.concat(components, axis=1).mean(axis=1).astype(float)

    def annotate_race_context(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self._with_rank_context(df)
        prepared[self.AGGRESSIVE_STRENGTH_COL] = self._runner_up_strength(prepared)

        strong_runner_up = self._runner_up_hard_mask(prepared, max_odds=self.add_second_max_odds)
        strong_race = strong_runner_up.groupby(prepared["race_id"], observed=True).transform("max")
        prepared[self.AGGRESSIVE_TIER_COL] = np.where(strong_race, "strong", "weak")
        return prepared

    def _runner_up_hard_mask(
        self,
        df: pd.DataFrame,
        *,
        max_odds: float,
    ) -> pd.Series:
        if not self.add_second_enabled or "race_id" not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)

        prepared = df.copy()
        if self.RANK_COL not in prepared.columns:
            prepared = self.annotate_race_context(prepared)

        score = _numeric_or_nan(prepared, self.SCORE_COL)
        gap = _numeric_or_nan(prepared, self.RUNNER_UP_GAP_COL)
        prob = _numeric_or_nan(prepared, self.RUNNER_UP_PROB_COL)
        edge = _numeric_or_nan(prepared, self.RUNNER_UP_EDGE_COL)
        odds = _numeric_or_nan(prepared, self.RUNNER_UP_ODDS_COL)
        market_condition = _numeric_or_nan(prepared, self.MARKET_CONDITION_COL)
        odds_cap = min(max_odds, self.add_second_max_odds)

        return (
            _numeric_or_nan(prepared, self.RANK_COL).eq(2)
            & score.ge(self.add_second_score_min)
            & gap.le(self.add_second_score_gap_max)
            & prob.ge(self.add_second_min_prob)
            & edge.ge(self.add_second_min_edge)
            & odds.gt(0.0)
            & odds.le(odds_cap)
            & market_condition.ge(self.add_second_min_market_condition)
        )

    def runner_up_candidate_reason(
        self,
        df: pd.DataFrame,
        *,
        selected_races: pd.Series,
        max_odds: float,
    ) -> pd.Series:
        prepared = self.annotate_race_context(df)
        if len(selected_races) != len(prepared):
            raise ValueError("selected_races must align with df index")

        eligible = self._runner_up_hard_mask(prepared, max_odds=max_odds)
        reasons = pd.Series("", index=prepared.index, dtype=object)
        reasons.loc[eligible & selected_races] = "add_second"
        reasons.loc[eligible & ~selected_races] = "rescue"
        return reasons

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = ensure_win_selection_columns(df)
        if not self._trained:
            prepared[self.SCORE_COL] = np.nan
            prepared[self.PASS_COL] = False
            return prepared

        if self.prob_edges and self.edge_edges and self.odds_edges:
            scores = self._score_frame(prepared)
        else:
            scores = self._surface_score(prepared, self.min_prob, self.min_edge, self.max_odds)
        odds = _numeric_or_nan(prepared, "tanoddslow")
        prepared[self.SCORE_COL] = scores
        prepared[self.PASS_COL] = (
            (pd.to_numeric(prepared["win_selection_prob"], errors="coerce") >= self.min_prob)
            & (pd.to_numeric(prepared["win_selection_edge"], errors="coerce") >= self.min_edge)
            & (odds > 0)
            & (odds <= self.max_odds)
        )
        return prepared

    def soft_pass_mask(
        self,
        df: pd.DataFrame,
        *,
        edge_floor: float = 0.0,
        min_prob: float = 0.0,
        max_odds: float = float("inf"),
        max_per_race: int = 1,
    ) -> pd.Series:
        prepared = df.copy()
        if not self._trained or max_per_race <= 0 or "race_id" not in prepared.columns:
            return pd.Series(False, index=prepared.index, dtype=bool)
        if self.SCORE_COL not in prepared.columns or self.PASS_COL not in prepared.columns:
            prepared = self.score(prepared)

        prob = pd.to_numeric(prepared["win_selection_prob"], errors="coerce")
        edge = pd.to_numeric(prepared["win_selection_edge"], errors="coerce")
        odds = _numeric_or_nan(prepared, "tanoddslow")
        hard_mask = prepared[self.PASS_COL].fillna(False).astype(bool)
        outer_mask = (
            (edge >= edge_floor)
            & (prob >= min_prob)
            & (odds > 0)
            & (odds <= max_odds)
        )
        near_mask = (
            (prob >= self.min_prob - self.SOFT_PROB_BUFFER)
            & (edge >= self.min_edge - self.SOFT_EDGE_BUFFER)
            & (odds <= min(max_odds, self.max_odds + self.SOFT_ODDS_BUFFER))
        )

        no_hard_pass = ~prepared["race_id"].isin(prepared.loc[hard_mask, "race_id"])
        eligible = prepared.loc[no_hard_pass & outer_mask & near_mask & ~hard_mask].copy()
        if eligible.empty:
            return pd.Series(False, index=prepared.index, dtype=bool)

        eligible = eligible.sort_values(
            ["race_id", self.SCORE_COL, "win_selection_edge", "win_selection_prob"],
            ascending=[True, False, False, False],
        )
        selected = eligible.groupby(
            "race_id", as_index=False, sort=False, observed=True
        ).head(max_per_race)
        return prepared.index.isin(selected.index)

    def save(self, path: Path) -> None:
        state = {
            "n_bins": self.n_bins,
            "prior_weight": self.prior_weight,
            "min_train_races": self.min_train_races,
            "min_fold_races": self.min_fold_races,
            "max_folds": self.max_folds,
            "threshold": self.threshold,
            "global_score": self.global_score,
            "prob_edges": self.prob_edges,
            "edge_edges": self.edge_edges,
            "odds_edges": self.odds_edges,
            "confidence_edges": self._confidence_edges,
            "combo_scores": self.combo_scores,
            "pair_scores": self.pair_scores,
            "single_scores": self.single_scores,
            "min_prob": self.min_prob,
            "min_edge": self.min_edge,
            "max_odds": self.max_odds,
            "add_second_score_min": self.add_second_score_min,
            "add_second_score_gap_max": self.add_second_score_gap_max,
            "add_second_min_prob": self.add_second_min_prob,
            "add_second_min_edge": self.add_second_min_edge,
            "add_second_max_odds": self.add_second_max_odds,
            "add_second_min_market_condition": self.add_second_min_market_condition,
            "strong_aggressive_threshold": self.strong_aggressive_threshold,
            "add_second_enabled": self.add_second_enabled,
            "_trained": self._trained,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: Path) -> WinSelectionGateModel:
        state = joblib.load(path)
        model = cls(
            n_bins=int(state["n_bins"]),
            prior_weight=float(state["prior_weight"]),
            min_train_races=int(state["min_train_races"]),
            min_fold_races=int(state["min_fold_races"]),
            max_folds=int(state["max_folds"]),
        )
        model.threshold = float(state["threshold"])
        model.global_score = float(state["global_score"])
        model.prob_edges = list(state["prob_edges"])
        model.edge_edges = list(state["edge_edges"])
        model.odds_edges = list(state["odds_edges"])
        model._confidence_edges = list(state.get("confidence_edges", []))
        model.combo_scores = dict(state["combo_scores"])
        model.pair_scores = dict(state["pair_scores"])
        model.single_scores = dict(state["single_scores"])
        model.min_prob = float(state.get("min_prob", 0.0))
        model.min_edge = float(state.get("min_edge", 0.0))
        model.max_odds = float(state.get("max_odds", float("inf")))
        model.add_second_score_min = float(state.get("add_second_score_min", float("inf")))
        model.add_second_score_gap_max = float(
            state.get("add_second_score_gap_max", float("inf"))
        )
        model.add_second_min_prob = float(state.get("add_second_min_prob", 0.0))
        model.add_second_min_edge = float(state.get("add_second_min_edge", 0.0))
        model.add_second_max_odds = float(state.get("add_second_max_odds", float("inf")))
        model.add_second_min_market_condition = float(
            state.get("add_second_min_market_condition", 0.0)
        )
        model.strong_aggressive_threshold = float(state.get("strong_aggressive_threshold", 1.0))
        model.add_second_enabled = bool(state.get("add_second_enabled", False))
        model._trained = bool(state["_trained"])
        return model
