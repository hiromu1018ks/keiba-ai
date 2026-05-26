"""OOF-based segment calibration for final win selection.

The calibrator is intentionally conservative: it only shrinks over-confident
win probabilities in turf segments where OOF outcomes underperformed the model.
It does not filter races or reduce bet count; it changes race-internal ranking
through the probability residual used by ``WinSelectionPolicy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd

from models.win_selection_gate import ensure_win_selection_columns


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
        return pd.Series("unknown", index=df.index, dtype=object)
    return df["surface"].astype(str).str.lower()


@dataclass
class WinSegmentCalibrator:
    """Shrink over-confident win probabilities by OOF reliability segments."""

    target_surface: str = "turf"
    prior_strength: float = 500.0
    min_segment_rows: int = 120
    min_segment_wins: int = 3
    min_factor: float = 0.85
    max_factor: float = 1.0
    apply_ev_factor: bool = False
    segment_table: dict[str, dict[str, float]] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)
    _trained: bool = False

    ODDS_BINS: ClassVar[tuple[float, ...]] = (1.0, 2.0, 5.0, 10.0, 30.0, 100.0, np.inf)
    ODDS_LABELS: ClassVar[tuple[str, ...]] = ("1-2", "2-5", "5-10", "10-30", "30-100", "100+")
    EV_BINS: ClassVar[tuple[float, ...]] = (-np.inf, 0.8, 1.0, 1.2, 1.5, 2.0, np.inf)
    EV_LABELS: ClassVar[tuple[str, ...]] = (
        "<0.8",
        "0.8-1.0",
        "1.0-1.2",
        "1.2-1.5",
        "1.5-2.0",
        "2.0+",
    )
    RANK_BINS: ClassVar[tuple[float, ...]] = (0.0, 1.0, 3.0, 6.0, 8.0, np.inf)
    RANK_LABELS: ClassVar[tuple[str, ...]] = ("1", "2-3", "4-6", "7-8", "9+")

    @property
    def is_trained(self) -> bool:
        return self._trained and bool(self.segment_table)

    def _prob_source(self, df: pd.DataFrame) -> pd.Series:
        prob = _numeric(df, "p_win_final_oof")
        prob = prob.where(prob.notna(), _numeric(df, "p_win_final"))
        prob = prob.where(prob.notna(), _numeric(df, "win_selection_prob"))
        prob = prob.where(prob.notna(), _numeric(df, "p_win_oof"))
        return prob

    def _ev_source(self, df: pd.DataFrame) -> pd.Series:
        ev = _numeric(df, "win_selection_ev_tail_calibrated")
        ev = ev.where(ev.notna(), _numeric(df, "win_selection_ev"))
        ev = ev.where(ev.notna(), _numeric(df, "win_selection_edge") + 1.0)
        return ev

    def _segment_keys(self, df: pd.DataFrame) -> pd.Series:
        prepared = ensure_win_selection_columns(df)
        prob = self._prob_source(prepared)
        ev = self._ev_source(prepared)
        rank = prob.groupby(_race_key(prepared), observed=True).rank(
            method="first",
            ascending=False,
        )
        odds_band = (
            pd.cut(
                _numeric(prepared, "tanodds"),
                bins=self.ODDS_BINS,
                labels=self.ODDS_LABELS,
                right=False,
            )
            .astype("string")
            .fillna("unknown")
        )
        ev_band = (
            pd.cut(ev, bins=self.EV_BINS, labels=self.EV_LABELS, right=False)
            .astype("string")
            .fillna("unknown")
        )
        rank_band = (
            pd.cut(
                rank,
                bins=self.RANK_BINS,
                labels=self.RANK_LABELS,
                right=True,
                include_lowest=True,
            )
            .astype("string")
            .fillna("unknown")
        )
        return (
            _surface_series(prepared)
            + "|"
            + odds_band.astype(str)
            + "|"
            + rank_band.astype(str)
            + "|"
            + ev_band.astype(str)
        )

    def train(self, df: pd.DataFrame) -> WinSegmentCalibrator:
        prepared = ensure_win_selection_columns(df)
        required = {"race_id", "kakuteijyuni", "tanodds", "win_selection_prob"}
        if prepared.empty or not required.issubset(prepared.columns):
            self._trained = False
            self.training_summary = {"reason": "missing_required_columns"}
            return self

        surface = _surface_series(prepared)
        target_mask = surface.eq(self.target_surface)
        if int(target_mask.sum()) < self.min_segment_rows:
            self._trained = False
            self.training_summary = {
                "reason": "insufficient_target_surface_rows",
                "target_surface": self.target_surface,
                "rows": int(target_mask.sum()),
            }
            return self

        train_df = prepared.loc[target_mask].copy()
        prob = self._prob_source(train_df)
        ev = self._ev_source(train_df)
        odds = _numeric(train_df, "confirmed_odds").where(
            _numeric(train_df, "confirmed_odds").notna(),
            _numeric(train_df, "tanodds"),
        )
        if odds.dropna().quantile(0.75) > 100.0:
            odds = odds / 100.0
        is_win = _numeric(train_df, "kakuteijyuni").eq(1)
        actual_return_unit = odds.clip(lower=0.0).where(is_win, 0.0)
        train_df["_segment_key"] = self._segment_keys(train_df)
        train_df["_prob"] = prob
        train_df["_ev"] = ev
        train_df["_actual_return_unit"] = actual_return_unit
        train_df["_is_win"] = is_win.astype(float)

        grouped = train_df.groupby("_segment_key", observed=True).agg(
            n=("_segment_key", "size"),
            wins=("_is_win", "sum"),
            pred_prob_sum=("_prob", "sum"),
            pred_prob_mean=("_prob", "mean"),
            pred_ev_sum=("_ev", "sum"),
            pred_ev_mean=("_ev", "mean"),
            actual_return_sum=("_actual_return_unit", "sum"),
        )
        grouped = grouped[
            grouped["n"].ge(self.min_segment_rows)
            & grouped["wins"].ge(self.min_segment_wins)
            & grouped["pred_prob_sum"].gt(0.0)
            & grouped["pred_prob_mean"].gt(0.0)
            & grouped["pred_ev_sum"].gt(0.0)
            & grouped["pred_ev_mean"].gt(0.0)
        ].copy()
        if grouped.empty:
            self._trained = False
            self.training_summary = {
                "reason": "no_reliable_segments",
                "target_surface": self.target_surface,
            }
            return self

        grouped["p_factor"] = (
            (grouped["wins"] + self.prior_strength * grouped["pred_prob_mean"])
            / (grouped["pred_prob_sum"] + self.prior_strength * grouped["pred_prob_mean"])
        ).clip(lower=self.min_factor, upper=self.max_factor)
        grouped["ev_factor"] = (
            (grouped["actual_return_sum"] + self.prior_strength * grouped["pred_ev_mean"])
            / (grouped["pred_ev_sum"] + self.prior_strength * grouped["pred_ev_mean"])
        ).clip(lower=self.min_factor, upper=self.max_factor)
        deployable = grouped[
            grouped["p_factor"].lt(0.995)
            | (self.apply_ev_factor and grouped["ev_factor"].lt(0.995))
        ].copy()
        self.segment_table = {
            str(key): {
                "p_factor": float(row["p_factor"]),
                "ev_factor": float(row["ev_factor"]),
                "n": float(row["n"]),
                "wins": float(row["wins"]),
            }
            for key, row in deployable.iterrows()
        }
        self._trained = bool(self.segment_table)
        self.training_summary = {
            "target_surface": self.target_surface,
            "prior_strength": self.prior_strength,
            "min_segment_rows": self.min_segment_rows,
            "min_segment_wins": self.min_segment_wins,
            "min_factor": self.min_factor,
            "max_factor": self.max_factor,
            "apply_ev_factor": self.apply_ev_factor,
            "n_segments": int(len(grouped)),
            "n_deployed_segments": int(len(self.segment_table)),
            "mean_p_factor": float(deployable["p_factor"].mean()) if not deployable.empty else 1.0,
            "min_p_factor": float(deployable["p_factor"].min()) if not deployable.empty else 1.0,
            "trained": self._trained,
        }
        return self

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        if not self.is_trained:
            prepared["win_segment_prob_factor"] = 1.0
            prepared["win_segment_ev_factor"] = 1.0
            prepared["win_segment_key"] = pd.NA
            return prepared

        keys = self._segment_keys(prepared)
        p_factor_map = {key: value["p_factor"] for key, value in self.segment_table.items()}
        ev_factor_map = {key: value["ev_factor"] for key, value in self.segment_table.items()}
        target_mask = _surface_series(prepared).eq(self.target_surface)
        p_factor = keys.map(p_factor_map).fillna(1.0).astype(float).where(target_mask, 1.0)
        ev_factor = keys.map(ev_factor_map).fillna(1.0).astype(float).where(target_mask, 1.0)
        if not self.apply_ev_factor:
            ev_factor = pd.Series(1.0, index=prepared.index, dtype=float)

        prepared["win_segment_key"] = keys
        prepared["win_segment_prob_factor"] = p_factor
        prepared["win_segment_ev_factor"] = ev_factor
        for col in ["p_win_final", "p_win_final_oof", "win_selection_prob"]:
            if col in prepared.columns:
                prepared[col] = (pd.to_numeric(prepared[col], errors="coerce") * p_factor).clip(
                    lower=1e-9, upper=0.99
                )

        if self.apply_ev_factor:
            ev_col = (
                "win_selection_ev_tail_calibrated"
                if "win_selection_ev_tail_calibrated" in prepared.columns
                else "win_selection_ev"
            )
            ev_adjusted = (pd.to_numeric(prepared[ev_col], errors="coerce") * ev_factor).clip(
                lower=0.0,
            )
            prepared[ev_col] = ev_adjusted
            if "win_selection_ev" in prepared.columns:
                prepared["win_selection_ev"] = ev_adjusted
            if "win_selection_edge" in prepared.columns:
                prepared["win_selection_edge"] = ev_adjusted - 1.0
        return prepared

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "target_surface": self.target_surface,
                "prior_strength": self.prior_strength,
                "min_segment_rows": self.min_segment_rows,
                "min_segment_wins": self.min_segment_wins,
                "min_factor": self.min_factor,
                "max_factor": self.max_factor,
                "apply_ev_factor": self.apply_ev_factor,
                "segment_table": self.segment_table,
                "training_summary": self.training_summary,
                "_trained": self._trained,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> WinSegmentCalibrator:
        state = joblib.load(path)
        obj = cls(
            target_surface=str(state.get("target_surface", "turf")),
            prior_strength=float(state.get("prior_strength", 500.0)),
            min_segment_rows=int(state.get("min_segment_rows", 120)),
            min_segment_wins=int(state.get("min_segment_wins", 3)),
            min_factor=float(state.get("min_factor", 0.85)),
            max_factor=float(state.get("max_factor", 1.0)),
            apply_ev_factor=bool(state.get("apply_ev_factor", False)),
        )
        obj.segment_table = dict(state.get("segment_table", {}))
        obj.training_summary = dict(state.get("training_summary", {}))
        obj._trained = bool(state.get("_trained", bool(obj.segment_table)))
        return obj
