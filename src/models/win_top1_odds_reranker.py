"""高オッズ暴走抑制リランカー -- odds cap による top-1 候補の再順位付け.

WinSelectionPolicy が生成した win_market_selection_score を使って
各レースの top-1 を決定する際、学習済み odds cap を超える候補を
順位対象外にする。cap 内候補が存在しない場合は元の全候補から選ぶ。
これにより高オッズの「暴走」選択を抑制しつつ、常に 1 頭/race を保証する。

学習は walk-forward fold で cap を評価し、1 レース当たり profit (ROI-1) を
主指標とする。fold ROI std と min fold ROI penalty を同程度の単位で組み合わせる。
改善がない/同点の場合は cap=inf (フィルタなし) を採用する。

払戻計算では confirmed_odds（確定オッズ）を優先し、欠損時は tanodds に
フォールバックする。confirmed_odds はスコア計算には一切使用しない。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANDIDATE_CAPS: list[float] = [20.0, 30.0, 40.0, 50.0, 75.0, 100.0, float("inf")]

RERANKER_APPLIED_COL = "reranker_applied"
RERANKER_CAP_COL = "reranker_cap"
RERANKER_ORIG_TOP1_COL = "reranker_original_top1_umaban"
RERANKER_FINAL_TOP1_COL = "reranker_final_top1_umaban"
RERANKER_SWITCH_REASON_COL = "reranker_switch_reason"

RERANKER_DIAGNOSTIC_COLS = [
    RERANKER_APPLIED_COL,
    RERANKER_CAP_COL,
    RERANKER_ORIG_TOP1_COL,
    RERANKER_FINAL_TOP1_COL,
    RERANKER_SWITCH_REASON_COL,
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
class WinTop1OddsReranker:
    """odds cap 学習による top-1 再順位付けモデル.

    各 surface (turf/dirt) の SubmodelSet に 1 個ずつ保持される。
    学習は win selection OOF frame 上で walk-forward 評価を行い、
    最適な odds cap を選択する。改善がない場合は cap=inf (無制限) となる。
    """

    candidate_caps: list[float] = field(default_factory=lambda: list(CANDIDATE_CAPS))
    min_train_races: int = 200
    min_fold_races: int = 80
    max_folds: int = 4
    stability_penalty: float = 0.25
    min_roi_floor: float = 0.85
    min_roi_penalty: float = 0.5

    selected_cap: float = float("inf")
    is_trained: bool = False
    training_summary: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_folds(self, race_order: pd.DataFrame) -> list[tuple[int, int]]:
        """展開ウィンドウ fold を構築 (WinProfitSelector と同一パターン)."""
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

    def _simulate_cap(
        self,
        df: pd.DataFrame,
        cap: float,
        fold_race_ids: set,
    ) -> dict[str, Any]:
        """fold テスト区間の全レースについて、cap による top-1 選択をシミュレート.

        Vectorized: groupby/idxmax で Python loop を回避。
        払戻は confirmed_odds 優先、欠損時 tanodds フォールバック。
        スコア計算には confirmed_odds を使用しない。
        """
        fold_df = df[df["race_id"].isin(fold_race_ids)].copy()
        if fold_df.empty:
            return {"profit": 0.0, "bets": 0.0, "roi": 0.0, "n_anomalous": 0}

        # Pre-compute numeric columns
        fold_df["_score_num"] = _numeric(fold_df, "win_market_selection_score").fillna(
            float("-inf")
        )
        fold_df["_odds_num"] = _numeric(fold_df, "tanodds").fillna(0.0)
        fold_df["_hit"] = _numeric(fold_df, "kakuteijyuni").eq(1)

        # Payout odds: confirmed_odds preferred, tanodds fallback
        payout_odds = _numeric(fold_df, "confirmed_odds")
        if payout_odds.notna().any():
            payout_odds = payout_odds.fillna(fold_df["_odds_num"])
        else:
            payout_odds = fold_df["_odds_num"]
        fold_df["_payout_odds"] = payout_odds

        # Cap eligibility
        cap_eligible = (fold_df["_odds_num"] > 0) & (fold_df["_odds_num"] <= cap)
        any_cap_eligible = cap_eligible.groupby(fold_df["race_id"], observed=True).transform("any")

        # Odds-valid fallback (tanodds > 0)
        odds_valid = fold_df["_odds_num"] > 0
        any_odds_valid = odds_valid.groupby(fold_df["race_id"], observed=True).transform("any")

        # Effective eligibility mask per row:
        # - If race has cap-eligible: use cap_eligible
        # - Elif race has odds-valid: use odds_valid
        # - Else (anomalous race, no valid odds): use all rows (fallback to score top1)
        effective_eligible = np.where(
            any_cap_eligible,
            cap_eligible,
            np.where(any_odds_valid, odds_valid, True),
        )
        fold_df["_effective_score"] = np.where(
            effective_eligible, fold_df["_score_num"], float("-inf")
        )

        # Count anomalous races (no valid tanodds at all)
        n_anomalous = int(
            (~any_odds_valid).groupby(fold_df["race_id"], observed=True).first().sum()
        )

        # Find best horse per race via idxmax
        best_idx = fold_df.groupby("race_id", observed=True)["_effective_score"].idxmax()

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
            "n_anomalous": n_anomalous,
            "n_fold_races": n_fold_races,
            "bets_equals_races": total_bets == n_fold_races,
        }

    def train(self, df: pd.DataFrame) -> WinTop1OddsReranker:
        """win selection OOF frame から最適 odds cap を学習する.

        入力は WinProfitSelector.score() 後の DataFrame を想定。
        win_market_selection_score, tanodds, kakuteijyuni, race_id, race_date が必要。
        confirmed_odds が存在すれば払戻計算に使用するが、スコア特徴には混ぜない。
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
            self.selected_cap = float("inf")
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
            self.selected_cap = float("inf")
            return self

        # 全 fold のレース ID をキャッシュ
        fold_race_sets: list[set] = []
        for train_end, test_end in folds:
            fold_race_sets.append(set(race_order.iloc[train_end:test_end]["race_id"]))

        # 各 cap を評価
        cap_metrics: dict[float, dict[str, Any]] = {}
        for cap in self.candidate_caps:
            fold_results: list[dict[str, Any]] = []
            for fold_rids in fold_race_sets:
                metrics = self._simulate_cap(df, cap, fold_rids)
                if metrics["bets"] > 0:
                    fold_results.append(metrics)

            if not fold_results:
                continue

            agg_profit = sum(m["profit"] for m in fold_results)
            agg_bets = sum(m["bets"] for m in fold_results)
            fold_rois = [m["roi"] for m in fold_results]
            total_anomalous = sum(m["n_anomalous"] for m in fold_results)
            total_fold_races = sum(m["n_fold_races"] for m in fold_results)

            # Verify bet count matches race count (sanity check)
            bets_ok = all(m["bets_equals_races"] for m in fold_results)

            cap_metrics[cap] = {
                "profit": agg_profit,
                "bets": agg_bets,
                "roi": (agg_profit + agg_bets) / agg_bets if agg_bets > 0 else 0.0,
                "min_fold_roi": min(fold_rois),
                "fold_roi_std": float(np.std(fold_rois)) if len(fold_rois) > 1 else 0.0,
                "n_folds": len(fold_results),
                "n_anomalous": total_anomalous,
                "n_fold_races": total_fold_races,
                "bets_ok": bets_ok,
            }

        if not cap_metrics:
            self.training_summary = {"reason": "no_evaluable_folds"}
            self.is_trained = True
            self.selected_cap = float("inf")
            return self

        # cap=inf の baseline を取得
        baseline = cap_metrics.get(float("inf"), None)

        # 目的関数: 1レース当たりprofit (ROI-1) を主指標にする
        # 安定性ペナルティ: fold ROI 標準偏差 + fold 最小 ROI 下限クリップ
        # 全て同程度の単位 (0〜0.5程度) で組み合わせる
        def _objective(m: dict[str, Any]) -> float:
            roi = float(m["roi"])
            fold_std = float(m["fold_roi_std"])
            min_fold = float(m["min_fold_roi"])
            return (
                (roi - 1.0)
                - self.stability_penalty * fold_std
                - self.min_roi_penalty * max(0.0, self.min_roi_floor - min_fold)
            )

        # Baseline (inf) の objective を明示的に計算
        baseline_obj = _objective(baseline) if baseline is not None else float("-inf")

        # 最適 cap を探索 (同点なら inf を優先 -- 小さい cap を選ばない)
        best_cap = float("inf")
        best_obj = baseline_obj
        for cap, m in cap_metrics.items():
            obj = _objective(m)
            if obj > best_obj:
                best_obj = obj
                best_cap = cap

        # 同点の場合は inf を採用 (改善がない/同点ならフィルタなし)
        # best_obj == baseline_obj の時点で best_cap は inf のまま

        self.selected_cap = best_cap
        self.is_trained = True
        self.training_summary = {
            "selected_cap": self.selected_cap,
            "objective": best_obj,
            "baseline_cap": float("inf"),
            "baseline_objective": baseline_obj,
            "baseline_metrics": baseline,
            "best_metrics": cap_metrics.get(best_cap),
            "all_cap_metrics": {str(k): v for k, v in cap_metrics.items()},
            "n_eval_races": int(len(race_order)),
            "n_folds": len(folds),
        }
        logger.info(
            "WinTop1OddsReranker trained: selected_cap=%s, objective=%.4f, "
            "baseline_obj=%.4f, n_races=%d, n_folds=%d",
            self.selected_cap,
            best_obj,
            baseline_obj,
            len(race_order),
            len(folds),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def apply(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """学習済み odds cap を候補に適用し、top-1 を再選択する.

        各 race_id ごとに:
          1. 全候補から score max の original top-1 を特定
          2. tanodds <= cap の候補が存在すれば、それらの中から score max を final top-1 とする
          3. cap 内候補が0なら original top-1 を維持
        常に各 race で 1 頭を返す。見送りは発生しない。

        戻り値は各 race につき 1 行 (final top-1) のみを含む。
        診断列を各行に追加する。

        switch_reason:
          - odds_cap_switch: cap内候補があり、元top1→cap内別馬に切り替わった
          - top1_within_cap: cap内候補があり、元top1もcap内で不変
          - no_eligible: cap内候補が0、元top1維持
          - inf_cap: cap=inf (フィルタなし)
          - untrained: 未学習
        """
        if candidates.empty:
            return self._annotate_empty(candidates)

        if not self.is_trained or self.selected_cap == float("inf"):
            reason = "untrained" if not self.is_trained else "inf_cap"
            return self._select_original_top1_per_race(candidates, reason=reason)

        cap = self.selected_cap
        # Determine key column for grouping
        key_col = "race_id" if "race_id" in candidates.columns else None
        key_series = (
            candidates["race_id"]
            if key_col
            else pd.Series("_race", index=candidates.index, dtype=object)
        )
        score = _numeric(candidates, "win_market_selection_score").fillna(float("-inf"))
        odds = _numeric(candidates, "tanodds").fillna(0.0)

        # Build per-race info
        result_indices: list[int] = []
        orig_top1_map: dict[Any, Any] = {}
        final_top1_map: dict[Any, Any] = {}
        applied_map: dict[Any, bool] = {}
        reason_map: dict[Any, str] = {}

        for rid, race_df in candidates.groupby(key_series, observed=True):
            race_score = score.loc[race_df.index]
            race_odds = odds.loc[race_df.index]

            # Original top-1 (score max, ties broken by row order via idxmax)
            orig_idx = race_score.idxmax()
            orig_umaban = race_df.loc[orig_idx, "umaban"] if "umaban" in race_df.columns else np.nan

            # Cap-eligible
            eligible_mask = (race_odds > 0) & (race_odds <= cap)
            has_eligible = eligible_mask.any()

            if has_eligible:
                elig_idx = race_score.loc[eligible_mask].idxmax()
                final_umaban = (
                    race_df.loc[elig_idx, "umaban"] if "umaban" in race_df.columns else np.nan
                )
                if orig_idx != elig_idx:
                    # Actual switch
                    result_indices.append(elig_idx)
                    applied_map[rid] = True
                    reason_map[rid] = "odds_cap_switch"
                else:
                    # Original top1 was within cap, no change
                    result_indices.append(orig_idx)
                    applied_map[rid] = False
                    reason_map[rid] = "top1_within_cap"
                final_top1_map[rid] = final_umaban
            else:
                # No eligible candidates, keep original
                result_indices.append(orig_idx)
                applied_map[rid] = False
                reason_map[rid] = "no_eligible"
                final_top1_map[rid] = orig_umaban

            orig_top1_map[rid] = orig_umaban

        if not result_indices:
            return self._annotate_empty(candidates)

        # Select only the final top-1 rows, preserving original index
        result = candidates.loc[result_indices].copy()

        # Add diagnostic columns via race key mapping
        if key_col:
            race_ids_for_map = result[key_col]
        else:
            race_ids_for_map = pd.Series("_race", index=result.index, dtype=object)
        result[RERANKER_APPLIED_COL] = race_ids_for_map.map(applied_map).fillna(False).astype(bool)
        result[RERANKER_CAP_COL] = cap
        result[RERANKER_ORIG_TOP1_COL] = race_ids_for_map.map(orig_top1_map)
        result[RERANKER_FINAL_TOP1_COL] = race_ids_for_map.map(final_top1_map)
        result[RERANKER_SWITCH_REASON_COL] = race_ids_for_map.map(reason_map).fillna("unknown")

        return result

    def _annotate_empty(self, df: pd.DataFrame) -> pd.DataFrame:
        """空 DataFrame に診断列を追加して返す."""
        df = df.copy()
        for col in RERANKER_DIAGNOSTIC_COLS:
            if col == RERANKER_APPLIED_COL:
                df[col] = pd.Series(dtype=bool, index=df.index)
            elif col == RERANKER_CAP_COL:
                df[col] = self.selected_cap
            elif col == RERANKER_SWITCH_REASON_COL:
                df[col] = "empty"
            else:
                df[col] = np.nan
        return df

    def _select_original_top1_per_race(
        self, candidates: pd.DataFrame, *, reason: str
    ) -> pd.DataFrame:
        """未学習/inf_cap 時に score top-1 をそのまま返す (1 行/race)."""
        key = _race_key(candidates)
        score = _numeric(candidates, "win_market_selection_score").fillna(float("-inf"))

        best_idx = score.groupby(key, observed=True).idxmax()
        result = candidates.loc[best_idx].copy()

        orig_top1 = (
            result["umaban"]
            if "umaban" in result.columns
            else pd.Series(np.nan, index=result.index)
        )
        result[RERANKER_APPLIED_COL] = False
        result[RERANKER_CAP_COL] = self.selected_cap
        result[RERANKER_ORIG_TOP1_COL] = orig_top1.values
        result[RERANKER_FINAL_TOP1_COL] = orig_top1.values
        result[RERANKER_SWITCH_REASON_COL] = reason
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "candidate_caps": self.candidate_caps,
                "min_train_races": self.min_train_races,
                "min_fold_races": self.min_fold_races,
                "max_folds": self.max_folds,
                "stability_penalty": self.stability_penalty,
                "min_roi_floor": self.min_roi_floor,
                "min_roi_penalty": self.min_roi_penalty,
                "selected_cap": self.selected_cap,
                "is_trained": self.is_trained,
                "training_summary": self.training_summary,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> WinTop1OddsReranker:
        state = joblib.load(path)
        model = cls(
            candidate_caps=list(state.get("candidate_caps", CANDIDATE_CAPS)),
            min_train_races=int(state.get("min_train_races", 200)),
            min_fold_races=int(state.get("min_fold_races", 80)),
            max_folds=int(state.get("max_folds", 4)),
            stability_penalty=float(state.get("stability_penalty", 0.25)),
            min_roi_floor=float(state.get("min_roi_floor", 0.85)),
            min_roi_penalty=float(state.get("min_roi_penalty", 0.5)),
        )
        model.selected_cap = float(state.get("selected_cap", float("inf")))
        model.is_trained = bool(state.get("is_trained", False))
        model.training_summary = dict(state.get("training_summary", {}))
        return model
