"""Learned gate for place bet selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _numeric_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def build_place_selection_ev(df: pd.DataFrame) -> pd.Series:
    lower_ev = _numeric_or_nan(df, "EV_lower_place")
    corrected_ev = _numeric_or_nan(df, "ev_place_corrected")
    direct_ev = _numeric_or_nan(df, "ev_place_direct")

    if corrected_ev.notna().any():
        selection_ev = lower_ev.where(lower_ev.notna(), corrected_ev)
        safety_floor = corrected_ev * 0.85
        return pd.concat([selection_ev, safety_floor], axis=1).max(axis=1).astype(float)
    if lower_ev.notna().any():
        return lower_ev.astype(float)
    return direct_ev.astype(float)


def ensure_place_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "place_selection_ev" not in prepared.columns:
        if "EV_lower_place" in prepared.columns or "ev_place_corrected" in prepared.columns:
            prepared["place_selection_ev"] = build_place_selection_ev(prepared)
        elif "edge_place" in prepared.columns:
            prepared["place_selection_ev"] = _numeric_or_nan(prepared, "edge_place") + 1.0
        else:
            prepared["place_selection_ev"] = _numeric_or_nan(prepared, "ev_place_direct")

    if "place_selection_edge" not in prepared.columns:
        prepared["place_selection_edge"] = _numeric_or_nan(prepared, "place_selection_ev") - 1.0

    if "place_selection_prob" not in prepared.columns:
        if "p_place_corrected" in prepared.columns:
            prepared["place_selection_prob"] = _numeric_or_nan(prepared, "p_place_corrected")
        elif "p_place_combined" in prepared.columns:
            prepared["place_selection_prob"] = _numeric_or_nan(prepared, "p_place_combined")
        else:
            prepared["place_selection_prob"] = _numeric_or_nan(prepared, "p_place_pred")

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


class PlaceSelectionGateModel:
    """OOF-learned gate for final place bet selection."""

    SCORE_COL = "place_gate_score"
    PASS_COL = "place_gate_pass"

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

        self.threshold: float = 1.0
        self.global_score: float = 1.0
        self.prob_edges: list[float] = []
        self.edge_edges: list[float] = []
        self.odds_edges: list[float] = []
        self.combo_scores: dict[tuple[int, int, int], float] = {}
        self.pair_scores: dict[tuple[str, int, int], float] = {}
        self.single_scores: dict[tuple[str, int], float] = {}
        self.min_prob = 0.0
        self.min_edge = 0.0
        self.max_odds = float("inf")
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _prepare_training_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = ensure_place_selection_columns(df)
        required_cols = [
            "race_id",
            "race_date",
            "kakuteijyuni",
            "fukuoddslow",
            "place_selection_prob",
            "place_selection_edge",
        ]
        missing = [col for col in required_cols if col not in prepared.columns]
        if missing:
            return pd.DataFrame(columns=required_cols)

        prepared = prepared[required_cols].copy()
        prepared["race_date"] = pd.to_datetime(prepared["race_date"], errors="coerce")
        prepared["fukuoddslow"] = _numeric_or_nan(prepared, "fukuoddslow")
        prepared["place_selection_prob"] = _numeric_or_nan(prepared, "place_selection_prob")
        prepared["place_selection_edge"] = _numeric_or_nan(prepared, "place_selection_edge")
        prepared["kakuteijyuni"] = _numeric_or_nan(prepared, "kakuteijyuni")
        prepared = prepared.dropna(
            subset=[
                "race_id",
                "race_date",
                "fukuoddslow",
                "place_selection_prob",
                "place_selection_edge",
                "kakuteijyuni",
            ]
        )
        if prepared.empty:
            return prepared

        prepared = prepared[prepared["fukuoddslow"] > 0].copy()
        prepared["log_place_odds"] = np.log1p(prepared["fukuoddslow"])
        prepared["realized_place_roi"] = np.where(
            prepared["kakuteijyuni"] <= 3,
            prepared["fukuoddslow"],
            0.0,
        )
        return prepared.sort_values(["race_date", "race_id"]).reset_index(drop=True)

    def _build_score_tables(self, df: pd.DataFrame) -> dict[str, Any]:
        work = df.copy()
        prob_edges = _quantile_edges(work["place_selection_prob"], self.n_bins)
        edge_edges = _quantile_edges(work["place_selection_edge"], self.n_bins)
        odds_edges = _quantile_edges(work["log_place_odds"], self.n_bins)

        work["_prob_bin"] = _bucketize(work["place_selection_prob"], prob_edges)
        work["_edge_bin"] = _bucketize(work["place_selection_edge"], edge_edges)
        work["_odds_bin"] = _bucketize(work["log_place_odds"], odds_edges)

        global_score = float(work["realized_place_roi"].mean())

        combo_scores: dict[tuple[int, int, int], float] = {}
        grouped = (
            work.groupby(
                ["_prob_bin", "_edge_bin", "_odds_bin"],
                observed=True,
            )["realized_place_roi"]
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
                work.groupby(cols, observed=True)["realized_place_roi"]
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
                work.groupby(col, observed=True)["realized_place_roi"]
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

        return {
            "prob_edges": prob_edges,
            "edge_edges": edge_edges,
            "odds_edges": odds_edges,
            "global_score": global_score,
            "combo_scores": combo_scores,
            "pair_scores": pair_scores,
            "single_scores": single_scores,
        }

    def _score_row(self, prob_bin: float, edge_bin: float, odds_bin: float) -> float:
        if pd.isna(prob_bin) or pd.isna(edge_bin) or pd.isna(odds_bin):
            return self.global_score

        prob_key = int(prob_bin)
        edge_key = int(edge_bin)
        odds_key = int(odds_bin)

        combo_key = (prob_key, edge_key, odds_key)
        if combo_key in self.combo_scores:
            return self.combo_scores[combo_key]

        pair_values = [
            self.pair_scores[key]
            for key in [
                ("prob_edge", prob_key, edge_key),
                ("prob_odds", prob_key, odds_key),
                ("edge_odds", edge_key, odds_key),
            ]
            if key in self.pair_scores
        ]
        if pair_values:
            return float(np.mean(pair_values))

        single_values = [
            self.single_scores[key]
            for key in [
                ("prob", prob_key),
                ("edge", edge_key),
                ("odds", odds_key),
            ]
            if key in self.single_scores
        ]
        if single_values:
            return float(np.mean(single_values))

        return self.global_score

    def _score_frame(self, df: pd.DataFrame) -> pd.Series:
        prepared = ensure_place_selection_columns(df)
        prob_bins = _bucketize(prepared["place_selection_prob"], self.prob_edges)
        edge_bins = _bucketize(prepared["place_selection_edge"], self.edge_edges)
        odds_values = np.log1p(_numeric_or_nan(prepared, "fukuoddslow").clip(lower=0.0))
        odds_bins = _bucketize(odds_values, self.odds_edges)

        scores = [
            self._score_row(prob_bin, edge_bin, odds_bin)
            for prob_bin, edge_bin, odds_bin in zip(prob_bins, edge_bins, odds_bins, strict=False)
        ]
        return pd.Series(scores, index=prepared.index, dtype=float)

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

    def _simulate_threshold(self, df: pd.DataFrame, threshold: float) -> dict[str, float]:
        candidates = df[df[self.SCORE_COL] >= threshold].copy()
        if candidates.empty:
            return {"roi": 0.0, "profit": 0.0, "max_drawdown": float("inf"), "bets": 0.0}

        candidates = candidates.sort_values(
            ["race_id", self.SCORE_COL, "place_selection_edge", "place_selection_prob"],
            ascending=[True, False, False, False],
        )
        candidates = candidates.groupby("race_id", as_index=False, sort=False).head(1)
        if candidates.empty:
            return {"roi": 0.0, "profit": 0.0, "max_drawdown": float("inf"), "bets": 0.0}

        candidates = candidates.sort_values(["race_date", "race_id"]).reset_index(drop=True)
        profit_units = candidates["realized_place_roi"] - 1.0
        equity = profit_units.cumsum()
        drawdown = (equity.cummax() - equity).fillna(0.0)

        return {
            "roi": float(candidates["realized_place_roi"].mean()),
            "profit": float(profit_units.sum()),
            "max_drawdown": float(drawdown.max()),
            "bets": float(len(candidates)),
        }

    def _build_threshold_grid(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[float], list[float], list[float]]:
        prob_values = sorted(
            set(
                float(df["place_selection_prob"].quantile(q))
                for q in [0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98]
            )
        )
        edge_values = sorted(
            set(
                float(df["place_selection_edge"].quantile(q))
                for q in [0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99]
            )
        )
        odds_values = sorted(
            set(
                float(df["fukuoddslow"].quantile(q))
                for q in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
            )
        )
        return prob_values, edge_values, odds_values

    def _surface_score(
        self,
        df: pd.DataFrame,
        min_prob: float,
        min_edge: float,
        max_odds: float,
    ) -> pd.Series:
        prob = pd.to_numeric(df["place_selection_prob"], errors="coerce")
        edge = pd.to_numeric(df["place_selection_edge"], errors="coerce")
        odds = pd.to_numeric(df["fukuoddslow"], errors="coerce")

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
        mask = (
            (pd.to_numeric(scored["place_selection_prob"], errors="coerce") >= min_prob)
            & (pd.to_numeric(scored["place_selection_edge"], errors="coerce") >= min_edge)
            & (pd.to_numeric(scored["fukuoddslow"], errors="coerce") > 0)
            & (pd.to_numeric(scored["fukuoddslow"], errors="coerce") <= max_odds)
        )
        candidates = scored.loc[mask].copy()
        if candidates.empty:
            return {"roi": 0.0, "profit": 0.0, "max_drawdown": float("inf"), "bets": 0.0}

        candidates = candidates.sort_values(
            ["race_id", self.SCORE_COL, "place_selection_edge", "place_selection_prob"],
            ascending=[True, False, False, False],
        )
        candidates = candidates.groupby("race_id", as_index=False, sort=False).head(1)
        if candidates.empty:
            return {"roi": 0.0, "profit": 0.0, "max_drawdown": float("inf"), "bets": 0.0}

        candidates = candidates.sort_values(["race_date", "race_id"]).reset_index(drop=True)
        profit_units = candidates["realized_place_roi"] - 1.0
        equity = profit_units.cumsum()
        drawdown = (equity.cummax() - equity).fillna(0.0)
        return {
            "roi": float(candidates["realized_place_roi"].mean()),
            "profit": float(profit_units.sum()),
            "max_drawdown": float(drawdown.max()),
            "bets": float(len(candidates)),
        }

    def train(self, df: pd.DataFrame) -> None:
        prepared = self._prepare_training_frame(df)
        if prepared.empty:
            return

        race_order = (
            prepared[["race_id", "race_date"]]
            .drop_duplicates()
            .sort_values(["race_date", "race_id"])
            .reset_index(drop=True)
        )
        folds = self._build_walk_forward_folds(len(race_order))
        if not folds:
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
                        -total_bets,
                    )
                    if key > best_key:
                        best_key = key
                        best_params = (min_prob, min_edge, max_odds)

        if best_params is None:
            return

        self.min_prob, self.min_edge, self.max_odds = best_params
        self.threshold = 0.0
        self._trained = True

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = ensure_place_selection_columns(df)
        if not self._trained:
            prepared[self.SCORE_COL] = np.nan
            prepared[self.PASS_COL] = False
            return prepared

        scores = self._surface_score(prepared, self.min_prob, self.min_edge, self.max_odds)
        odds = _numeric_or_nan(prepared, "fukuoddslow")
        prepared[self.SCORE_COL] = scores
        prepared[self.PASS_COL] = (
            (pd.to_numeric(prepared["place_selection_prob"], errors="coerce") >= self.min_prob)
            & (pd.to_numeric(prepared["place_selection_edge"], errors="coerce") >= self.min_edge)
            & (odds > 0)
            & (odds <= self.max_odds)
        )
        return prepared

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
            "combo_scores": self.combo_scores,
            "pair_scores": self.pair_scores,
            "single_scores": self.single_scores,
            "min_prob": self.min_prob,
            "min_edge": self.min_edge,
            "max_odds": self.max_odds,
            "_trained": self._trained,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: Path) -> PlaceSelectionGateModel:
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
        model.combo_scores = dict(state["combo_scores"])
        model.pair_scores = dict(state["pair_scores"])
        model.single_scores = dict(state["single_scores"])
        model.min_prob = float(state.get("min_prob", 0.0))
        model.min_edge = float(state.get("min_edge", 0.0))
        model.max_odds = float(state.get("max_odds", float("inf")))
        model._trained = bool(state["_trained"])
        return model
