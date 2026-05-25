"""Profit-oriented candidate selector for win betting.

This selector is intentionally separate from WinSelectionGate.  The gate scores
calibration slices; this model decides how many win candidates in a race are
worth staking from an OOF profit objective with an explicit volume floor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from models.win_selection_gate import ensure_win_selection_columns

PROFIT_SCORE_COL = "win_profit_score"
PROFIT_PASS_COL = "win_profit_selector_pass"
PROFIT_RANK_COL = "win_profit_rank"
PROFIT_STAKE_SCALE_COL = "win_profit_stake_scale"
PROFIT_REASON_COL = "win_profit_reason"


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _race_key(df: pd.DataFrame) -> pd.Series:
    if "race_id" in df.columns:
        return df["race_id"]
    return pd.Series("_race", index=df.index, dtype=object)


def _rank_desc(values: pd.Series, race_key: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(values, errors="coerce")
        .groupby(race_key, observed=True)
        .rank(
            method="first",
            ascending=False,
        )
    )


def _profit_score(df: pd.DataFrame) -> pd.Series:
    market_score = _numeric(df, "win_market_selection_score")
    if market_score.notna().any():
        return market_score.fillna(float("-inf")).astype(float)

    edge = _numeric(df, "win_selection_edge")
    odds = _numeric(df, "tanodds").clip(lower=1.0)
    log_odds = np.log1p(odds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prob = _numeric(df, "win_selection_prob")
    key = _race_key(df)
    prob_rank = (
        prob.groupby(key, observed=True)
        .rank(
            pct=True,
            method="average",
            ascending=True,
        )
        .fillna(0.5)
    )
    return (edge - 0.05 * log_odds + 0.02 * prob_rank).fillna(float("-inf")).astype(float)


def _safe_quantiles(series: pd.Series, quantiles: list[float]) -> list[float]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return []
    return sorted({float(clean.quantile(q)) for q in quantiles if 0.0 <= q <= 1.0})


@dataclass
class WinProfitSelectorParams:
    """Deployable rule learned by walk-forward profit simulation."""

    rank_limit: int = 1
    min_score: float = float("-inf")
    min_edge: float = float("-inf")
    min_prob: float = 0.0
    min_odds: float = 1.0
    max_odds: float = float("inf")


@dataclass
class WinProfitSelector:
    """Select 0-N win candidates per race from a profit objective.

    The objective maximizes unit-stake profit while penalizing low coverage, so
    the learned policy can skip weak races or add second/third candidates without
    degenerating into tiny-bet-count cherry picking.
    """

    min_train_races: int = 200
    min_fold_races: int = 80
    max_folds: int = 4
    max_rank_limit: int = 3
    min_bets_per_eval_race: float = 0.35
    low_volume_penalty: float = 1.5
    drawdown_penalty: float = 0.02
    max_stake_scale: float = 2.0
    params: WinProfitSelectorParams = field(default_factory=WinProfitSelectorParams)
    training_summary: dict[str, Any] = field(default_factory=dict)
    _trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def max_per_race(self) -> int:
        return max(1, int(self.params.rank_limit))

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = ensure_win_selection_columns(df)
        prepared = prepared.copy()
        prepared[PROFIT_SCORE_COL] = _profit_score(prepared)
        key = _race_key(prepared)
        prepared[PROFIT_RANK_COL] = _rank_desc(prepared[PROFIT_SCORE_COL], key)
        if "race_date" not in prepared.columns and "race_id" in prepared.columns:
            race_year = prepared["race_id"].astype(str).str[:4]
            prepared["race_date"] = pd.to_datetime(race_year + "-01-01", errors="coerce")
        return prepared

    def _build_folds(self, race_order: pd.DataFrame) -> list[tuple[int, int]]:
        n_races = len(race_order)
        if n_races < self.min_train_races + self.min_fold_races:
            return []

        remaining = n_races - self.min_train_races
        fold_size = max(self.min_fold_races, remaining // max(1, self.max_folds))
        folds: list[tuple[int, int]] = []
        train_end = self.min_train_races
        while train_end < n_races and len(folds) < self.max_folds:
            test_end = min(n_races, train_end + fold_size)
            if test_end - train_end >= self.min_fold_races:
                folds.append((train_end, test_end))
            train_end = test_end
        return folds

    def _candidate_params(self, prepared: pd.DataFrame) -> list[WinProfitSelectorParams]:
        score_values = [float("-inf")]
        top_ranked = prepared[prepared[PROFIT_RANK_COL].le(self.max_rank_limit)]
        score_values.extend(_safe_quantiles(top_ranked[PROFIT_SCORE_COL], [0.05, 0.15, 0.30]))

        edge_values = [float("-inf"), -0.10, 0.0, 0.05, 0.10, 0.20]
        prob_values = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
        odds_ranges = [
            (1.0, float("inf")),
            (1.0, 100.0),
            (2.0, 30.0),
            (2.0, 50.0),
            (5.0, 50.0),
            (10.0, 100.0),
        ]

        params: list[WinProfitSelectorParams] = []
        for rank_limit in range(1, self.max_rank_limit + 1):
            for min_score in score_values:
                for min_edge in edge_values:
                    for min_prob in prob_values:
                        for min_odds, max_odds in odds_ranges:
                            params.append(
                                WinProfitSelectorParams(
                                    rank_limit=rank_limit,
                                    min_score=min_score,
                                    min_edge=min_edge,
                                    min_prob=min_prob,
                                    min_odds=min_odds,
                                    max_odds=max_odds,
                                )
                            )
        return params

    def _mask(self, prepared: pd.DataFrame, params: WinProfitSelectorParams) -> pd.Series:
        odds = _numeric(prepared, "tanodds")
        edge = _numeric(prepared, "win_selection_edge")
        prob = _numeric(prepared, "win_selection_prob")
        return (
            prepared[PROFIT_RANK_COL].le(float(params.rank_limit))
            & prepared[PROFIT_SCORE_COL].ge(params.min_score)
            & edge.ge(params.min_edge)
            & prob.ge(params.min_prob)
            & odds.ge(params.min_odds)
            & odds.lt(params.max_odds)
        ).fillna(False)

    def _simulate(
        self,
        prepared: pd.DataFrame,
        params: WinProfitSelectorParams,
    ) -> dict[str, float]:
        selected = prepared.loc[self._mask(prepared, params)].copy()
        bets = len(selected)
        if bets == 0:
            return {"bets": 0.0, "return": 0.0, "profit": 0.0, "roi": 0.0, "max_dd": 0.0}

        odds = _numeric(selected, "tanodds").clip(lower=0.0).fillna(0.0)
        hit = _numeric(selected, "kakuteijyuni").eq(1)
        returns = pd.Series(np.where(hit, odds, 0.0), index=selected.index, dtype=float)
        profit_units = returns - 1.0

        order_cols = [col for col in ["race_date", "race_id"] if col in selected.columns]
        if order_cols:
            profit_units = profit_units.loc[selected.sort_values(order_cols).index]
        cumulative = profit_units.cumsum()
        running_max = cumulative.cummax().clip(lower=0.0)
        max_dd = float((running_max - cumulative).max()) if not cumulative.empty else 0.0
        total_return = float(returns.sum())
        return {
            "bets": float(bets),
            "return": total_return,
            "profit": float(profit_units.sum()),
            "roi": total_return / bets,
            "max_dd": max_dd,
        }

    def train(self, df: pd.DataFrame) -> WinProfitSelector:
        prepared = self._prepare(df)
        required = {"race_id", "race_date", "kakuteijyuni", "tanodds"}
        if prepared.empty or not required.issubset(prepared.columns):
            self.training_summary = {"reason": "missing_required_columns"}
            self._trained = False
            return self

        race_order = (
            prepared[["race_id", "race_date"]]
            .drop_duplicates()
            .sort_values(["race_date", "race_id"])
            .reset_index(drop=True)
        )
        folds = self._build_folds(race_order)
        if not folds:
            self.training_summary = {
                "reason": "insufficient_races",
                "n_races": int(len(race_order)),
            }
            self._trained = False
            return self

        total_eval_races = sum(test_end - train_end for train_end, test_end in folds)
        min_eval_bets = max(40.0, total_eval_races * self.min_bets_per_eval_race)
        best_params: WinProfitSelectorParams | None = None
        best_key = (False, float("-inf"), float("-inf"), float("-inf"))
        candidate_rows: list[dict[str, Any]] = []

        for params in self._candidate_params(prepared):
            aggregate = {"bets": 0.0, "return": 0.0, "profit": 0.0, "max_dd": 0.0}
            yearly_roi: dict[str, float] = {}
            for train_end, test_end in folds:
                test_races = set(race_order.iloc[train_end:test_end]["race_id"])
                fold_test = prepared[prepared["race_id"].isin(test_races)].copy()
                metrics = self._simulate(fold_test, params)
                aggregate["bets"] += metrics["bets"]
                aggregate["return"] += metrics["return"]
                aggregate["profit"] += metrics["profit"]
                aggregate["max_dd"] = max(aggregate["max_dd"], metrics["max_dd"])
                years = pd.to_datetime(fold_test["race_date"], errors="coerce").dt.year
                for year in sorted(years.dropna().unique().tolist()):
                    year_metrics = self._simulate(fold_test.loc[years.eq(year)], params)
                    if year_metrics["bets"] > 0:
                        yearly_roi[str(int(year))] = year_metrics["roi"]

            bets = aggregate["bets"]
            if bets <= 0:
                continue
            roi = aggregate["return"] / bets
            volume_deficit = max(0.0, min_eval_bets - bets)
            objective = (
                aggregate["profit"]
                - self.low_volume_penalty * volume_deficit
                - self.drawdown_penalty * aggregate["max_dd"]
            )
            volume_ok = bets >= min_eval_bets
            key = (volume_ok, objective, aggregate["profit"], bets)
            row = {
                "params": asdict(params),
                "bets": bets,
                "roi": roi,
                "profit": aggregate["profit"],
                "max_dd": aggregate["max_dd"],
                "objective": objective,
                "volume_ok": volume_ok,
                "yearly_roi": yearly_roi,
            }
            candidate_rows.append(row)
            if key > best_key:
                best_key = key
                best_params = params

        if best_params is None:
            self.training_summary = {"reason": "no_valid_candidates"}
            self._trained = False
            return self

        self.params = best_params
        self._trained = True
        best_metrics = self._simulate(prepared, best_params)
        top_candidates = sorted(
            candidate_rows,
            key=lambda row: (bool(row["volume_ok"]), float(row["objective"])),
            reverse=True,
        )[:10]
        self.training_summary = {
            "selected_params": asdict(best_params),
            "train_bets": best_metrics["bets"],
            "train_roi": best_metrics["roi"],
            "train_profit": best_metrics["profit"],
            "min_eval_bets": min_eval_bets,
            "n_eval_races": int(total_eval_races),
            "n_train_races": int(len(race_order)),
            "candidates": top_candidates,
        }
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare(df)
        if not self._trained:
            prepared[PROFIT_PASS_COL] = False
            prepared[PROFIT_STAKE_SCALE_COL] = 1.0
            prepared[PROFIT_REASON_COL] = "untrained"
            return prepared

        pass_mask = self._mask(prepared, self.params)
        prepared[PROFIT_PASS_COL] = pass_mask
        selected_score = prepared[PROFIT_SCORE_COL].where(pass_mask)
        if selected_score.notna().sum() >= 2:
            score_min = float(selected_score.min())
            score_max = float(selected_score.max())
            denom = max(1e-9, score_max - score_min)
            scale = 1.0 + (selected_score - score_min) / denom
        else:
            scale = pd.Series(1.0, index=prepared.index, dtype=float)
        prepared[PROFIT_STAKE_SCALE_COL] = (
            scale.fillna(1.0).clip(lower=1.0, upper=self.max_stake_scale).astype(float)
        )
        prepared[PROFIT_REASON_COL] = np.where(pass_mask, "profit_selector", "profit_filtered")
        return prepared

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "min_train_races": self.min_train_races,
                "min_fold_races": self.min_fold_races,
                "max_folds": self.max_folds,
                "max_rank_limit": self.max_rank_limit,
                "min_bets_per_eval_race": self.min_bets_per_eval_race,
                "low_volume_penalty": self.low_volume_penalty,
                "drawdown_penalty": self.drawdown_penalty,
                "max_stake_scale": self.max_stake_scale,
                "params": asdict(self.params),
                "training_summary": self.training_summary,
                "_trained": self._trained,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> WinProfitSelector:
        state = joblib.load(path)
        model = cls(
            min_train_races=int(state.get("min_train_races", 200)),
            min_fold_races=int(state.get("min_fold_races", 80)),
            max_folds=int(state.get("max_folds", 4)),
            max_rank_limit=int(state.get("max_rank_limit", 3)),
            min_bets_per_eval_race=float(state.get("min_bets_per_eval_race", 0.35)),
            low_volume_penalty=float(state.get("low_volume_penalty", 1.5)),
            drawdown_penalty=float(state.get("drawdown_penalty", 0.02)),
            max_stake_scale=float(state.get("max_stake_scale", 2.0)),
        )
        params = state.get("params", {})
        model.params = WinProfitSelectorParams(
            rank_limit=int(params.get("rank_limit", 1)),
            min_score=float(params.get("min_score", float("-inf"))),
            min_edge=float(params.get("min_edge", float("-inf"))),
            min_prob=float(params.get("min_prob", 0.0)),
            min_odds=float(params.get("min_odds", 1.0)),
            max_odds=float(params.get("max_odds", float("inf"))),
        )
        model.training_summary = dict(state.get("training_summary", {}))
        model._trained = bool(state.get("_trained", False))
        return model
