"""WinTop1 スコアブレンダー -- current score と market residual の混合比学習.

WinSelectionPolicy が生成した win_market_selection_score (current) と
市場残差 (market residual) を race 内 percentile rank に変換し、
OOF walk-forward で最適混合比を学習する。

混合式: blended = (1 - w) * current_rank + w * market_rank
  - current_rank: race 内 win_market_selection_score の percentile rank
  - market_rank: race 内 market_residual の percentile rank
  - w=0 が baseline (従来スコアのみ)。厳密改善のみ採用。

dirt の EV 依存を弱める意図。surface ごとに別インスタンスで学習。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from models.win_selection_policy import market_residual_score

logger = logging.getLogger(__name__)

CANDIDATE_WEIGHTS: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

BLENDER_WEIGHT_COL = "blender_weight"
BLENDER_CURRENT_RANK_COL = "blender_current_rank"
BLENDER_MARKET_RANK_COL = "blender_market_rank"
BLENDER_SCORE_COL = "blender_score"
BLENDER_RAW_SCORE_COL = "win_market_selection_score_raw"

BLENDER_DIAGNOSTIC_COLS = [
    BLENDER_WEIGHT_COL,
    BLENDER_CURRENT_RANK_COL,
    BLENDER_MARKET_RANK_COL,
    BLENDER_SCORE_COL,
]


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _race_key(df: pd.DataFrame) -> pd.Series:
    if "race_id" in df.columns:
        return df["race_id"]
    return pd.Series("_race", index=df.index, dtype=object)


@dataclass
class WinTop1ScoreBlender:
    """current score と market residual の混合比を学習するモデル.

    各 surface (turf/dirt) の SubmodelSet に 1 個ずつ保持される。
    学習は win selection OOF frame 上で walk-forward 評価を行い、
    最適な weight を選択する。改善がない場合は w=0 (従来スコアのみ) となる。
    """

    candidate_weights: list[float] = field(default_factory=lambda: list(CANDIDATE_WEIGHTS))
    min_train_races: int = 200
    min_fold_races: int = 80
    max_folds: int = 4
    stability_penalty: float = 0.25
    min_roi_floor: float = 0.85
    min_roi_penalty: float = 0.5

    selected_weight: float = 0.0
    is_trained: bool = False
    training_summary: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_folds(self, race_order: pd.DataFrame) -> list[tuple[int, int]]:
        """展開ウィンドウ fold を構築 (WinTop1OddsReranker と同一パターン)."""
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

    def _simulate_weight(
        self,
        df: pd.DataFrame,
        weight: float,
        fold_race_ids: set,
    ) -> dict[str, Any]:
        """fold テスト区間の全レースについて、weight による top-1 選択をシミュレート.

        Vectorized: groupby/idxmax で Python loop を回避。
        払戻は confirmed_odds 優先、欠損時 tanodds フォールバック。
        スコア計算には confirmed_odds を使用しない。
        """
        fold_df = df[df["race_id"].isin(fold_race_ids)].copy()
        if fold_df.empty:
            return {"profit": 0.0, "bets": 0.0, "roi": 0.0, "n_fold_races": 0}

        key = _race_key(fold_df)

        # Current score rank (within race, percentile, higher = better)
        current_score = _numeric(fold_df, "win_market_selection_score").fillna(float("-inf"))
        current_rank = (
            current_score.groupby(key, observed=True)
            .rank(
                method="first",
                ascending=True,
                pct=True,
            )
            .fillna(0.0)
        )

        # Market residual rank (within race, percentile, higher = better)
        market_res = market_residual_score(fold_df)
        market_rank = (
            market_res.groupby(key, observed=True)
            .rank(
                method="first",
                ascending=True,
                pct=True,
            )
            .fillna(0.0)
        )

        # Blended score: (1-w)*current + w*market
        blended = (1.0 - weight) * current_rank + weight * market_rank

        # Payout odds: confirmed_odds preferred, tanodds fallback
        payout_odds = _numeric(fold_df, "confirmed_odds")
        if payout_odds.notna().any():
            payout_odds = payout_odds.fillna(_numeric(fold_df, "tanodds"))
        else:
            payout_odds = _numeric(fold_df, "tanodds")
        fold_df["_payout_odds"] = payout_odds
        fold_df["_hit"] = _numeric(fold_df, "kakuteijyuni").eq(1)
        fold_df["_blended"] = blended

        # Top-1 per race via idxmax
        best_idx = fold_df.groupby("race_id", observed=True)["_blended"].idxmax()

        # Compute profit from selected horses
        best_df = fold_df.loc[best_idx]
        hits = best_df["_hit"].values
        payouts = np.where(hits, best_df["_payout_odds"].values, 0.0)
        profit_per_bet = payouts - 1.0
        total_profit = float(profit_per_bet.sum())
        total_bets = len(best_idx)
        n_fold_races = fold_df["race_id"].nunique()

        return {
            "profit": total_profit,
            "bets": float(total_bets),
            "roi": (total_profit + total_bets) / total_bets if total_bets > 0 else 0.0,
            "n_fold_races": n_fold_races,
            "bets_equals_races": total_bets == n_fold_races,
        }

    def train(self, df: pd.DataFrame) -> WinTop1ScoreBlender:
        """win selection OOF frame から最適混合 weight を学習する.

        入力は WinSelectionPolicy.apply() 済みの DataFrame を想定。
        win_market_selection_score, p_win_final (or win_selection_prob),
        tanodds, kakuteijyuni, race_id, race_date が必要。
        confirmed_odds が存在すれば払戻計算に使用する。
        """
        required = {
            "race_id",
            "race_date",
            "kakuteijyuni",
            "tanodds",
            "win_market_selection_score",
        }
        if df.empty or not required.issubset(df.columns):
            self.training_summary = {"reason": "missing_required_columns"}
            self.is_trained = True
            self.selected_weight = 0.0
            return self

        # レースを時系列順にソート
        race_order = (
            df[["race_id", "race_date"]]
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
            self.is_trained = True
            self.selected_weight = 0.0
            return self

        # 全 fold のレース ID をキャッシュ
        fold_race_sets: list[set] = []
        for train_end, test_end in folds:
            fold_race_sets.append(set(race_order.iloc[train_end:test_end]["race_id"]))

        # 各 weight を評価
        weight_metrics: dict[float, dict[str, Any]] = {}
        for w in self.candidate_weights:
            fold_results: list[dict[str, Any]] = []
            for fold_rids in fold_race_sets:
                metrics = self._simulate_weight(df, w, fold_rids)
                if metrics["bets"] > 0:
                    fold_results.append(metrics)

            if not fold_results:
                continue

            agg_profit = sum(m["profit"] for m in fold_results)
            agg_bets = sum(m["bets"] for m in fold_results)
            fold_rois = [m["roi"] for m in fold_results]
            total_fold_races = sum(m["n_fold_races"] for m in fold_results)

            # Verify bet count matches race count (sanity check)
            bets_ok = all(m["bets_equals_races"] for m in fold_results)

            weight_metrics[w] = {
                "profit": agg_profit,
                "bets": agg_bets,
                "roi": (agg_profit + agg_bets) / agg_bets if agg_bets > 0 else 0.0,
                "min_fold_roi": min(fold_rois),
                "fold_roi_std": float(np.std(fold_rois)) if len(fold_rois) > 1 else 0.0,
                "n_folds": len(fold_results),
                "n_fold_races": total_fold_races,
                "bets_ok": bets_ok,
            }

        if not weight_metrics:
            self.training_summary = {"reason": "no_evaluable_folds"}
            self.is_trained = True
            self.selected_weight = 0.0
            return self

        # w=0 baseline を取得
        baseline = weight_metrics.get(0.0, None)

        # 目的関数: WinTop1OddsReranker と同一
        def _objective(m: dict[str, Any]) -> float:
            roi = float(m["roi"])
            fold_std = float(m["fold_roi_std"])
            min_fold = float(m["min_fold_roi"])
            return (
                (roi - 1.0)
                - self.stability_penalty * fold_std
                - self.min_roi_penalty * max(0.0, self.min_roi_floor - min_fold)
            )

        baseline_obj = _objective(baseline) if baseline is not None else float("-inf")

        # 最適 weight を探索 (同点なら w=0 を優先)
        best_weight = 0.0
        best_obj = baseline_obj
        for w, m in weight_metrics.items():
            obj = _objective(m)
            if obj > best_obj:
                best_obj = obj
                best_weight = w

        self.selected_weight = best_weight
        self.is_trained = True
        self.training_summary = {
            "selected_weight": self.selected_weight,
            "objective": best_obj,
            "baseline_weight": 0.0,
            "baseline_objective": baseline_obj,
            "baseline_metrics": baseline,
            "best_metrics": weight_metrics.get(best_weight),
            "all_weight_metrics": {str(k): v for k, v in weight_metrics.items()},
            "n_eval_races": int(len(race_order)),
            "n_folds": len(folds),
        }
        logger.info(
            "WinTop1ScoreBlender trained: selected_weight=%.2f, objective=%.4f, "
            "baseline_obj=%.4f, n_races=%d, n_folds=%d",
            self.selected_weight,
            best_obj,
            baseline_obj,
            len(race_order),
            len(folds),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """学習済み weight で current/market score をブレンドする.

        w=0 または未訓練時は no-op (df をそのまま返す)。
        常に全行を返す（行数・行順を維持）。
        win_market_selection_score を blended score で置換し、
        元スコアを win_market_selection_score_raw に保存する。
        """
        if not self.is_trained or self.selected_weight == 0.0:
            return df

        if df.empty:
            return df

        result = df.copy()

        # Save original score
        result[BLENDER_RAW_SCORE_COL] = pd.to_numeric(
            result["win_market_selection_score"],
            errors="coerce",
        )

        # Market residual
        market_res = market_residual_score(result)

        # Rank within race
        key = _race_key(result)
        current_score = pd.to_numeric(
            result["win_market_selection_score"],
            errors="coerce",
        ).fillna(float("-inf"))
        current_rank = (
            current_score.groupby(key, observed=True)
            .rank(
                method="first",
                ascending=True,
                pct=True,
            )
            .fillna(0.0)
        )
        market_rank = (
            market_res.groupby(key, observed=True)
            .rank(
                method="first",
                ascending=True,
                pct=True,
            )
            .fillna(0.0)
        )

        # Blend
        w = self.selected_weight
        blended = (1.0 - w) * current_rank + w * market_rank

        # Replace score
        result["win_market_selection_score"] = blended

        # Diagnostic columns
        result[BLENDER_WEIGHT_COL] = w
        result[BLENDER_CURRENT_RANK_COL] = current_rank
        result[BLENDER_MARKET_RANK_COL] = market_rank
        result[BLENDER_SCORE_COL] = blended

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "candidate_weights": self.candidate_weights,
                "min_train_races": self.min_train_races,
                "min_fold_races": self.min_fold_races,
                "max_folds": self.max_folds,
                "stability_penalty": self.stability_penalty,
                "min_roi_floor": self.min_roi_floor,
                "min_roi_penalty": self.min_roi_penalty,
                "selected_weight": self.selected_weight,
                "is_trained": self.is_trained,
                "training_summary": self.training_summary,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> WinTop1ScoreBlender:
        state = joblib.load(path)
        model = cls(
            candidate_weights=list(state.get("candidate_weights", CANDIDATE_WEIGHTS)),
            min_train_races=int(state.get("min_train_races", 200)),
            min_fold_races=int(state.get("min_fold_races", 80)),
            max_folds=int(state.get("max_folds", 4)),
            stability_penalty=float(state.get("stability_penalty", 0.25)),
            min_roi_floor=float(state.get("min_roi_floor", 0.85)),
            min_roi_penalty=float(state.get("min_roi_penalty", 0.5)),
        )
        model.selected_weight = float(state.get("selected_weight", 0.0))
        model.is_trained = bool(state.get("is_trained", False))
        model.training_summary = dict(state.get("training_summary", {}))
        return model
