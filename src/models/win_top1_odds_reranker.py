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

# Gap reranker (dirt-only, second stage after odds cap)
GAP_RERANKER_DEPLOYED_COL = "gap_reranker_deployed"
GAP_RERANKER_APPLIED_COL = "gap_reranker_applied"
GAP_RERANKER_ORIG_TOP1_COL = "gap_reranker_original_top1_umaban"
GAP_RERANKER_FINAL_TOP1_COL = "gap_reranker_final_top1_umaban"
GAP_RERANKER_SCORE_GAP_COL = "gap_reranker_score_gap"
GAP_RERANKER_PROB_MARGIN_COL = "gap_reranker_prob_margin"
GAP_RERANKER_THRESHOLD_COL = "gap_reranker_threshold"
GAP_RERANKER_SWITCH_REASON_COL = "gap_reranker_switch_reason"

GAP_RERANKER_DIAGNOSTIC_COLS = [
    GAP_RERANKER_DEPLOYED_COL,
    GAP_RERANKER_APPLIED_COL,
    GAP_RERANKER_ORIG_TOP1_COL,
    GAP_RERANKER_FINAL_TOP1_COL,
    GAP_RERANKER_SCORE_GAP_COL,
    GAP_RERANKER_PROB_MARGIN_COL,
    GAP_RERANKER_THRESHOLD_COL,
    GAP_RERANKER_SWITCH_REASON_COL,
]

# Probability column priority for top2 detection (inference: p_win_final first)
# OOF columns are training-only artifacts and must not influence inference.
PROB_COLS_PRIORITY = ["p_win_final", "p_win_final_oof", "win_selection_prob", "p_win_oof"]

# Training simulation priority: OOF column first (more accurate for OOF evaluation)
OOF_PROB_COLS_PRIORITY = ["p_win_final_oof", "p_win_final", "win_selection_prob", "p_win_oof"]


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

    # Gap reranker fields (dirt-only, second stage after odds cap)
    gap_reranker_deployed: bool = False
    gap_reranker_surface: str = ""
    gap_reranker_max_change_rate: float = 0.10  # training/holdout deploy guard only
    gap_reranker_min_prob_margin: float = 0.01
    gap_reranker_score_gap_threshold: float = float("inf")
    gap_reranker_top2_odds_le_top1: bool = False
    gap_reranker_top2_max_odds: float = float("inf")
    gap_reranker_training_summary: dict[str, Any] = field(default_factory=dict)

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

    def _simulate_gap_reranker(
        self,
        df: pd.DataFrame,
        *,
        cap: float,
        gap_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Gap reranker の OOF シミュレーション (vectorized).

        cap 適用済み top-1 と top-2 を比較し、確率・gap・margin 条件で
        top-2 へ切替える場合の ROI/hit_rate/変更数を返す。
        gap_params に {"disabled": True} を渡すと cap-only ベースラインになる。

        戻り値:
            roi, hit_rate, profit, bets, n_changes, change_rate, yearly (年別 dict)
        """
        gap_disabled = gap_params.get("disabled", False)
        gap_threshold = gap_params.get("score_gap_threshold", float("inf"))
        min_margin = gap_params.get("min_prob_margin", 0.0)
        max_change_rate = gap_params.get("max_change_rate", 0.10)
        top2_odds_le = gap_params.get("top2_odds_le_top1", False)
        top2_max_odds = gap_params.get("top2_max_odds", float("inf"))

        sim = df.copy()
        sim["_score"] = _numeric(sim, "win_market_selection_score").fillna(float("-inf"))
        sim["_odds"] = _numeric(sim, "tanodds").fillna(0.0)
        sim["_hit"] = _numeric(sim, "kakuteijyuni").eq(1)

        # 払戻オッズ: confirmed_odds 優先、tanodds fallback
        payout = _numeric(sim, "confirmed_odds")
        if payout.notna().any():
            payout = payout.fillna(sim["_odds"])
        else:
            payout = sim["_odds"]
        sim["_payout"] = payout

        # 確率列を priority chain から取得 (simulation: OOF-first)
        prob_col: str | None = None
        for pc in OOF_PROB_COLS_PRIORITY:
            if pc in sim.columns and _numeric(sim, pc).notna().any():
                prob_col = pc
                break
        sim["_prob"] = _numeric(sim, prob_col) if prob_col else pd.Series(np.nan, index=sim.index)

        # Cap eligibility
        cap_mask = (sim["_odds"] > 0) & (sim["_odds"] <= cap)
        any_cap = cap_mask.groupby(sim["race_id"], observed=True).transform("any")
        valid = sim["_odds"] > 0
        any_valid = valid.groupby(sim["race_id"], observed=True).transform("any")
        eligible = np.where(any_cap, cap_mask, np.where(any_valid, valid, True))
        sim["_eff_score"] = np.where(eligible, sim["_score"], float("-inf"))

        # Per-race cap-eligible count (gap rerank不可 if < 2)
        cap_elig_per_race = cap_mask.groupby(sim["race_id"], observed=True).transform("sum")
        cap_elig_map = cap_elig_per_race.groupby(sim["race_id"], observed=True).first().to_dict()

        # Top1: cap-eligible score max
        top1_idx = sim.groupby("race_id", observed=True)["_eff_score"].idxmax()
        sim["_is_top1"] = False
        sim.loc[top1_idx.values, "_is_top1"] = True

        # Top2: top1 を除外した次点
        sim["_eff_score2"] = np.where(sim["_is_top1"], float("-inf"), sim["_eff_score"])
        top2_idx = sim.groupby("race_id", observed=True)["_eff_score2"].idxmax()

        # Top1 / Top2 情報を展開 (race_id merge — reset_index+concat の行順依存を回避)
        t1_needed = sim.loc[top1_idx.values][
            ["race_id", "_score", "_prob", "_payout", "_hit"]
        ].reset_index(drop=True)

        t2_needed = sim.loc[top2_idx.values][
            ["race_id", "_score", "_prob", "_payout", "_hit", "_odds"]
        ].reset_index(drop=True)

        t1_needed.columns = [
            "race_id",
            "t1_score",
            "t1_prob",
            "t1_payout",
            "t1_hit",
        ]
        t2_needed.columns = [
            "race_id",
            "t2_score",
            "t2_prob",
            "t2_payout",
            "t2_hit",
            "t2_odds",
        ]

        merged = t1_needed.merge(t2_needed, on="race_id", how="inner")
        merged["_cap_elig_count"] = merged["race_id"].map(cap_elig_map)

        # Score gap, prob margin
        merged["score_gap"] = merged["t1_score"] - merged["t2_score"]
        merged["prob_margin"] = merged["t2_prob"] - merged["t1_prob"]

        # 切替条件
        switch_cond = (
            (merged["t2_prob"] > merged["t1_prob"])
            & (merged["score_gap"] <= gap_threshold)
            & (merged["prob_margin"] >= min_margin)
        )
        if top2_odds_le and "t1_odds" not in merged.columns:
            # top1 odds は top1_idx 側にない場合: sim から取得
            t1_odds = sim.loc[top1_idx.values, "_odds"].values
            merged["t1_odds"] = t1_odds
        if top2_odds_le:
            switch_cond = switch_cond & (merged["t2_odds"] <= merged["t1_odds"])
        if top2_max_odds < float("inf"):
            switch_cond = switch_cond & (merged["t2_odds"] <= top2_max_odds)

        # Cap eligible < 2 → gap rerank不可
        switch_cond = switch_cond & (merged["_cap_elig_count"] >= 2)

        # Disabled baseline: cap-only top1 (no gap switching whatsoever)
        if gap_disabled:
            switch_cond = pd.Series(False, index=switch_cond.index)

        # 変更率ガード (training/holdout deploy guard only — NOT for per-race apply)
        raw_change_rate = float(switch_cond.mean())
        rate_guard_failed = False
        if not gap_disabled and raw_change_rate > max_change_rate:
            rate_guard_failed = True
            switch_cond = pd.Series(False, index=switch_cond.index)

        # 最終選択
        final_hit = np.where(switch_cond, merged["t2_hit"].values, merged["t1_hit"].values)
        n_changes = int(switch_cond.sum())
        n_bets = len(merged)

        # 年別メトリクス (race_date から year を取得)
        dates = sim.loc[top1_idx.values, "race_date"].values
        try:
            years = pd.to_datetime(dates).year
        except (ValueError, TypeError):
            years = np.full(n_bets, 0)

        merged["_year"] = years
        yearly: dict[str, dict[str, Any]] = {}
        for yr, grp in merged.groupby("_year", observed=True):
            nb = len(grp)
            # Recompute per-year final hit/payout (losing selection → payout 0)
            sw = switch_cond.loc[grp.index]
            fy_hit = np.where(sw, grp["t2_hit"].values, grp["t1_hit"].values)
            fp_sel = np.where(sw, grp["t2_payout"].values, grp["t1_payout"].values)
            fp = np.where(fy_hit, fp_sel, 0.0)
            prof = (fp - 1.0).sum()
            yearly[str(int(yr))] = {
                "roi": float((prof + nb) / nb) if nb > 0 else 0.0,
                "hit_rate": float(fy_hit.mean()),
                "profit": float(prof),
                "bets": int(nb),
                "n_changes": int(sw.sum()),
                "change_rate": float(sw.mean()),
            }

        payouts_all = np.where(
            final_hit,
            np.where(switch_cond, merged["t2_payout"].values, merged["t1_payout"].values),
            0.0,
        )
        profit = (payouts_all - 1.0).sum()
        return {
            "roi": float((profit + n_bets) / n_bets) if n_bets > 0 else 0.0,
            "hit_rate": float(final_hit.mean()),
            "profit": float(profit),
            "bets": int(n_bets),
            "n_changes": n_changes,
            "change_rate": float(n_changes / n_bets) if n_bets > 0 else 0.0,
            "raw_candidate_change_rate": raw_change_rate,
            "rate_guard_failed": rate_guard_failed,
            "yearly": yearly,
        }

    def train(self, df: pd.DataFrame, *, surface: str = "") -> WinTop1OddsReranker:
        """win selection OOF frame から最適 odds cap を学習する.

        入力は WinProfitSelector.score() 後の DataFrame を想定。
        win_market_selection_score, tanodds, kakuteijyuni, race_id, race_date が必要。
        confirmed_odds が存在すれば払戻計算に使用するが、スコア特徴には混ぜない。

        surface="dirt" の場合、cap学習後にダート限定の gap reranker 探索を行う。
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

        # --- Gap reranker exploration (dirt only) ---
        if surface == "dirt":
            self._train_gap_reranker(df)

        return self

    # ------------------------------------------------------------------
    # Gap Reranker Training (dirt-only, second stage)
    # ------------------------------------------------------------------

    def _train_gap_reranker(self, df: pd.DataFrame) -> None:
        """ダート限定の gap reranker パラメータ探索.

        1. 探索期間 (2022-2023) でスコア gap 分布から閾値候補を生成
        2. 全パラメータ候補を探索期間で評価 (年別 ROI delta, hit delta, 変更率)
        3. 最良候補を 2024 holdout で検証
        4. 条件を満たせば deploy、さもなくば disabled
        """
        if df.empty or "race_date" not in df.columns:
            self.gap_reranker_training_summary = {"reason": "empty_or_no_date"}
            return

        dirt_df = df[df["surface"] == "dirt"].copy() if "surface" in df.columns else df.copy()
        if dirt_df.empty:
            self.gap_reranker_training_summary = {"reason": "no_dirt_races"}
            return

        # --- 期間分割 ---
        dirt_df["_date"] = pd.to_datetime(dirt_df["race_date"], errors="coerce")
        explore_df = dirt_df[
            (dirt_df["_date"] >= "2022-01-01") & (dirt_df["_date"] < "2024-01-01")
        ].copy()
        holdout_df = dirt_df[
            (dirt_df["_date"] >= "2024-01-01") & (dirt_df["_date"] < "2025-01-01")
        ].copy()

        n_explore_races = explore_df["race_id"].nunique() if not explore_df.empty else 0
        n_holdout_races = holdout_df["race_id"].nunique() if not holdout_df.empty else 0

        if n_explore_races < 200:
            self.gap_reranker_training_summary = {
                "reason": "insufficient_explore_races",
                "n_explore_races": n_explore_races,
            }
            logger.info("Gap reranker skipped: insufficient explore races (%d)", n_explore_races)
            return
        if n_holdout_races < 100:
            self.gap_reranker_training_summary = {
                "reason": "insufficient_holdout_races",
                "n_holdout_races": n_holdout_races,
            }
            logger.info("Gap reranker skipped: insufficient holdout races (%d)", n_holdout_races)
            return

        # --- Baseline 計算 ---
        raw_baseline = self._simulate_gap_reranker(
            explore_df,
            cap=float("inf"),
            gap_params={"disabled": True},
        )
        cap_baseline = self._simulate_gap_reranker(
            explore_df,
            cap=self.selected_cap,
            gap_params={"disabled": True},
        )

        # --- Score gap 分布から閾値候補を生成 ---
        cap_for_explore = self.selected_cap
        _sim = explore_df.copy()
        _sim["_score"] = _numeric(_sim, "win_market_selection_score").fillna(float("-inf"))
        _sim["_odds"] = _numeric(_sim, "tanodds").fillna(0.0)
        cap_mask = (_sim["_odds"] > 0) & (_sim["_odds"] <= cap_for_explore)
        any_cap = cap_mask.groupby(_sim["race_id"], observed=True).transform("any")
        valid = _sim["_odds"] > 0
        any_valid = valid.groupby(_sim["race_id"], observed=True).transform("any")
        eligible = np.where(any_cap, cap_mask, np.where(any_valid, valid, True))
        _sim["_eff"] = np.where(eligible, _sim["_score"], float("-inf"))

        top1_idx = _sim.groupby("race_id", observed=True)["_eff"].idxmax()
        _sim["_is_top1"] = False
        _sim.loc[top1_idx.values, "_is_top1"] = True
        _sim["_eff2"] = np.where(_sim["_is_top1"], float("-inf"), _sim["_eff"])
        top2_idx = _sim.groupby("race_id", observed=True)["_eff2"].idxmax()

        t1_scores = _sim.loc[top1_idx.values, "_score"].values
        t2_scores = _sim.loc[top2_idx.values, "_score"].values
        score_gaps = t1_scores - t2_scores
        # top2が存在しない(nan) raceは除外
        valid_gaps = score_gaps[~np.isnan(score_gaps) & np.isfinite(score_gaps)]

        if len(valid_gaps) < 50:
            self.gap_reranker_training_summary = {
                "reason": "insufficient_valid_gaps",
                "n_valid_gaps": int(len(valid_gaps)),
            }
            logger.info("Gap reranker skipped: insufficient valid gaps (%d)", len(valid_gaps))
            return

        quantiles = np.quantile(valid_gaps, [0.25, 0.50, 0.75, 0.90])
        gap_thresholds = quantiles.tolist() + [float("inf")]

        # --- パラメータ候補生成 ---
        min_margins = [0.0, 0.01, 0.02, 0.03, 0.05]
        max_change_rates = [0.05, 0.10]
        candidates: list[dict[str, Any]] = []

        # disabled baseline
        candidates.append({"disabled": True})

        for gt in gap_thresholds:
            for mm in min_margins:
                for cr in max_change_rates:
                    candidates.append(
                        {
                            "disabled": False,
                            "score_gap_threshold": gt,
                            "min_prob_margin": mm,
                            "max_change_rate": cr,
                            "top2_odds_le_top1": False,
                            "top2_max_odds": float("inf"),
                        }
                    )

        # Near-miss candidates: top2_max_odds=10.0
        # 組合せ爆発回避: top2_odds_le_top1=False, max_change_rate=0.10 only
        for gt in gap_thresholds:
            for mm in min_margins:
                candidates.append(
                    {
                        "disabled": False,
                        "score_gap_threshold": gt,
                        "min_prob_margin": mm,
                        "max_change_rate": 0.10,
                        "top2_odds_le_top1": False,
                        "top2_max_odds": 10.0,
                    }
                )

        # --- 探索期間評価 ---
        best_candidate: dict[str, Any] | None = None
        best_avg_roi_delta = float("-inf")
        best_tiebreak: tuple[float, ...] = (
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        )
        cap_only_roi = cap_baseline["roi"]
        cap_only_hit = cap_baseline["hit_rate"]

        # 年別 cap-only ROI/hit を事前計算 (各候補の年別deltaの共通baseline)
        yearly_cap_only: dict[str, float] = {}
        yearly_cap_only_hit: dict[str, float] = {}
        for yr_key in sorted(explore_df["_date"].dt.year.dropna().unique()):
            yr = int(yr_key)
            yr_df = explore_df[explore_df["_date"].dt.year == yr]
            yr_cap = self._simulate_gap_reranker(
                yr_df,
                cap=self.selected_cap,
                gap_params={"disabled": True},
            )
            yearly_cap_only[str(yr)] = yr_cap["roi"]
            yearly_cap_only_hit[str(yr)] = yr_cap["hit_rate"]

        # 診断用: 条件に関係なく最良候補を追跡
        diag_best_cand: dict[str, Any] | None = None
        diag_best_roi_delta = float("-inf")

        for cand in candidates:
            sim_result = self._simulate_gap_reranker(
                explore_df,
                cap=self.selected_cap,
                gap_params=cand,
            )

            if cand["disabled"]:
                continue  # disabled は skip (baseline のみ)

            if sim_result.get("rate_guard_failed", False):
                continue  # 変更率超過 → candidate不適格

            yearly = sim_result.get("yearly", {})
            year_roi_deltas: list[float] = []
            all_conditions_met = True

            for yr, ym in yearly.items():
                y_cap_roi = yearly_cap_only.get(yr, cap_only_roi)
                y_cap_hit = yearly_cap_only_hit.get(yr, cap_only_hit)

                roi_delta = ym["roi"] - y_cap_roi
                hit_delta = ym["hit_rate"] - y_cap_hit
                cr = ym["change_rate"]

                year_roi_deltas.append(roi_delta)

                # 各年: ROI delta >= -0.01, 変更率 1-10%, hit delta >= -1pp
                if roi_delta < -0.01:
                    all_conditions_met = False
                if cr < 0.01 or cr > 0.10:
                    all_conditions_met = False
                if hit_delta < -0.01:
                    all_conditions_met = False

            avg_roi_delta = float(np.mean(year_roi_deltas)) if year_roi_deltas else 0.0

            # 診断用: 条件に関係なく最良候補を追跡
            if avg_roi_delta > diag_best_roi_delta:
                diag_best_roi_delta = avg_roi_delta
                diag_best_cand = cand

            # 選択条件: 2年平均ROI delta > 0, 各年delta >= -0.01,
            #           各年変更率1-10%, 各年hit delta >= -1pp
            # 同点時: 変更率が低い/閾値が厳しい候補優先
            if avg_roi_delta <= 0 or not all_conditions_met:
                continue

            # Tiebreaker: lower change_rate → lower threshold → higher margin → lower top2_max_odds
            gt = cand.get("score_gap_threshold", float("inf"))
            mm = cand.get("min_prob_margin", 0.0)
            t2mo = cand.get("top2_max_odds", float("inf"))
            cr = sim_result["change_rate"]
            tb = (
                cr,
                gt if np.isfinite(gt) else 1e9,
                -mm,
                t2mo if np.isfinite(t2mo) else 1e9,
            )

            if avg_roi_delta > best_avg_roi_delta + 1e-9:
                best_avg_roi_delta = avg_roi_delta
                best_candidate = cand
                best_tiebreak = tb
            elif abs(avg_roi_delta - best_avg_roi_delta) <= 1e-9 and tb < best_tiebreak:
                best_candidate = cand
                best_tiebreak = tb

        if best_candidate is None:
            # 診断用: 最良候補の年別詳細を計算 (条件を満たさなくても)
            diagnostic_yearly: dict[str, dict[str, Any]] = {}
            if diag_best_cand is not None:
                for yr_key in sorted(explore_df["_date"].dt.year.dropna().unique()):
                    yr = int(yr_key)
                    yr_df = explore_df[explore_df["_date"].dt.year == yr]
                    yr_cap = self._simulate_gap_reranker(
                        yr_df,
                        cap=self.selected_cap,
                        gap_params={"disabled": True},
                    )
                    yr_gap = self._simulate_gap_reranker(
                        yr_df,
                        cap=self.selected_cap,
                        gap_params=diag_best_cand,
                    )
                    diagnostic_yearly[str(yr)] = {
                        "cap_only_roi": yr_cap["roi"],
                        "gap_roi": yr_gap["roi"],
                        "roi_delta": yr_gap["roi"] - yr_cap["roi"],
                        "cap_only_hit_rate": yr_cap["hit_rate"],
                        "gap_hit_rate": yr_gap["hit_rate"],
                        "hit_delta": yr_gap["hit_rate"] - yr_cap["hit_rate"],
                        "n_changes": yr_gap["n_changes"],
                        "change_rate": yr_gap["change_rate"],
                    }
            self.gap_reranker_training_summary = {
                "reason": "no_candidate_meets_criteria",
                "raw_score_baseline": {
                    "roi": raw_baseline["roi"],
                    "hit_rate": raw_baseline["hit_rate"],
                    "bets": raw_baseline["bets"],
                },
                "cap_only_baseline": {
                    "roi": cap_only_roi,
                    "hit_rate": cap_baseline["hit_rate"],
                    "bets": cap_baseline["bets"],
                },
                "explore_yearly": diagnostic_yearly,
                "n_explore_races": n_explore_races,
                "n_holdout_races": n_holdout_races,
                "n_candidates_evaluated": len(candidates),
                "deployed": False,
            }
            logger.info("Gap reranker: no candidate meets criteria on explore period")
            return

        # --- 探索期間の年別詳細 (best_candidate が確定済み) ---
        explore_yearly = {}
        for yr_key in sorted(explore_df["_date"].dt.year.dropna().unique()):
            yr = int(yr_key)
            yr_df = explore_df[explore_df["_date"].dt.year == yr]
            yr_cap = self._simulate_gap_reranker(
                yr_df,
                cap=self.selected_cap,
                gap_params={"disabled": True},
            )
            yr_gap = self._simulate_gap_reranker(
                yr_df,
                cap=self.selected_cap,
                gap_params=best_candidate,
            )
            explore_yearly[str(yr)] = {
                "cap_only_roi": yr_cap["roi"],
                "gap_roi": yr_gap["roi"],
                "roi_delta": yr_gap["roi"] - yr_cap["roi"],
                "cap_only_hit_rate": yr_cap["hit_rate"],
                "gap_hit_rate": yr_gap["hit_rate"],
                "hit_delta": yr_gap["hit_rate"] - yr_cap["hit_rate"],
                "n_changes": yr_gap["n_changes"],
                "change_rate": yr_gap["change_rate"],
            }

        # --- 2024 holdout 検証 ---
        holdout_result = self._simulate_gap_reranker(
            holdout_df,
            cap=self.selected_cap,
            gap_params=best_candidate,
        )
        holdout_cap_only = self._simulate_gap_reranker(
            holdout_df,
            cap=self.selected_cap,
            gap_params={"disabled": True},
        )

        holdout_roi_delta = holdout_result["roi"] - holdout_cap_only["roi"]
        holdout_change_rate = holdout_result["change_rate"]
        holdout_n_changes = holdout_result["n_changes"]
        holdout_hit_delta = holdout_result["hit_rate"] - holdout_cap_only["hit_rate"]

        min_changes = max(30, int(0.01 * n_holdout_races))
        deploy_conditions = [
            ("roi_delta > 0", holdout_roi_delta > 0),
            ("change_rate 1-10%", 0.01 <= holdout_change_rate <= 0.10),
            (f"n_changes >= {min_changes}", holdout_n_changes >= min_changes),
            ("hit_delta >= -1pp", holdout_hit_delta >= -0.01),
        ]
        all_pass = all(cond[1] for cond in deploy_conditions)
        failures = [name for name, passed in deploy_conditions if not passed]

        if not all_pass:
            self.gap_reranker_training_summary = {
                "reason": "holdout_validation_failed",
                "failures": failures,
                "selected_candidate": best_candidate,
                "raw_score_baseline": {
                    "roi": raw_baseline["roi"],
                    "hit_rate": raw_baseline["hit_rate"],
                    "bets": raw_baseline["bets"],
                },
                "cap_only_baseline": {
                    "roi": cap_only_roi,
                    "hit_rate": cap_baseline["hit_rate"],
                    "bets": cap_baseline["bets"],
                },
                "explore_yearly": explore_yearly,
                "holdout_2024": {
                    "cap_only_roi": holdout_cap_only["roi"],
                    "cap_only_hit_rate": holdout_cap_only["hit_rate"],
                    "gap_roi": holdout_result["roi"],
                    "gap_hit_rate": holdout_result["hit_rate"],
                    "roi_delta": holdout_roi_delta,
                    "change_rate": holdout_change_rate,
                    "n_changes": holdout_n_changes,
                    "hit_delta": holdout_hit_delta,
                },
                "n_explore_races": n_explore_races,
                "n_holdout_races": n_holdout_races,
                "deployed": False,
            }
            logger.info("Gap reranker: holdout validation failed: %s", ", ".join(failures))
            return

        # --- Deploy ---
        self.gap_reranker_deployed = True
        self.gap_reranker_surface = "dirt"
        self.gap_reranker_score_gap_threshold = float(best_candidate["score_gap_threshold"])
        self.gap_reranker_min_prob_margin = float(best_candidate["min_prob_margin"])
        self.gap_reranker_max_change_rate = float(best_candidate["max_change_rate"])
        self.gap_reranker_top2_odds_le_top1 = bool(best_candidate.get("top2_odds_le_top1", False))
        self.gap_reranker_top2_max_odds = float(best_candidate.get("top2_max_odds", float("inf")))

        self.gap_reranker_training_summary = {
            "raw_score_baseline": {
                "roi": raw_baseline["roi"],
                "hit_rate": raw_baseline["hit_rate"],
                "bets": raw_baseline["bets"],
            },
            "cap_only_baseline": {
                "roi": cap_only_roi,
                "hit_rate": cap_baseline["hit_rate"],
                "bets": cap_baseline["bets"],
            },
            "explore_yearly": explore_yearly,
            "holdout_2024": {
                "cap_only_roi": holdout_cap_only["roi"],
                "gap_roi": holdout_result["roi"],
                "roi_delta": holdout_roi_delta,
                "cap_only_hit_rate": holdout_cap_only["hit_rate"],
                "gap_hit_rate": holdout_result["hit_rate"],
                "hit_delta": holdout_hit_delta,
                "n_changes": holdout_n_changes,
                "change_rate": holdout_change_rate,
            },
            "n_changes": holdout_n_changes,
            "change_rate": holdout_change_rate,
            "selected_params": best_candidate,
            "n_explore_races": n_explore_races,
            "n_holdout_races": n_holdout_races,
            "deployed": True,
        }
        logger.info(
            "Gap reranker DEPLOYED for dirt: threshold=%.4f, margin=%.3f, "
            "max_change_rate=%.2f, holdout_roi_delta=%.4f, n_changes=%d",
            self.gap_reranker_score_gap_threshold,
            self.gap_reranker_min_prob_margin,
            self.gap_reranker_max_change_rate,
            holdout_roi_delta,
            holdout_n_changes,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_surface(candidates: pd.DataFrame) -> str:
        """候補 DataFrame から surface を検出する。"""
        if "surface" in candidates.columns and not candidates.empty:
            val = candidates["surface"].iloc[0]
            if pd.notna(val):
                return str(val)
        return ""

    @staticmethod
    def _get_prob_value(row: pd.Series, df: pd.DataFrame, idx) -> float:
        """PROB_COLS_PRIORITY から利用可能な確率値を取得する。"""
        for pc in PROB_COLS_PRIORITY:
            if pc in df.columns:
                val = df.at[idx, pc]
                if pd.notna(val):
                    return float(val)
        return float("nan")

    def apply(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """学習済み odds cap → gap reranker を候補に適用し、top-1 を再選択する.

        各 race_id ごとに:
          1. odds cap: tanodds <= cap の候補から score max を top-1 とする
          2. gap reranker (dirt only): top2 の確率が top1 より高く、score gap が
             閾値以下、確率差が margin 以上なら top2 へ切替
        常に各 race で 1 頭を返す。見送りは発生しない。

        戻り値は各 race につき 1 行 (final top-1) のみを含む。
        reranker_* および gap_reranker_* 診断列を各行に追加する。
        """
        if candidates.empty:
            return self._annotate_empty_with_gap(candidates)

        # Stage 1: Odds cap
        cap_result = self._apply_odds_cap(candidates)

        # Stage 2: Gap reranker (dirt only, needs original candidates)
        surface = self._detect_surface(candidates)
        result = self._apply_gap_reranker(
            cap_result,
            all_candidates=candidates,
            surface=surface,
        )
        return result

    def _apply_odds_cap(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Odds cap による top-1 再選択 (1 行/race).

        既存の apply ロジックと同一。cap 適用済みの 1行/race DataFrame を返す。
        """
        if not self.is_trained or self.selected_cap == float("inf"):
            reason = "untrained" if not self.is_trained else "inf_cap"
            return self._select_original_top1_per_race(candidates, reason=reason)

        cap = self.selected_cap
        key_col = "race_id" if "race_id" in candidates.columns else None
        key_series = (
            candidates["race_id"]
            if key_col
            else pd.Series("_race", index=candidates.index, dtype=object)
        )
        score = _numeric(candidates, "win_market_selection_score").fillna(float("-inf"))
        odds = _numeric(candidates, "tanodds").fillna(0.0)

        result_indices: list[int] = []
        orig_top1_map: dict[Any, Any] = {}
        final_top1_map: dict[Any, Any] = {}
        applied_map: dict[Any, bool] = {}
        reason_map: dict[Any, str] = {}

        for rid, race_df in candidates.groupby(key_series, observed=True):
            race_score = score.loc[race_df.index]
            race_odds = odds.loc[race_df.index]

            orig_idx = race_score.idxmax()
            orig_umaban = race_df.loc[orig_idx, "umaban"] if "umaban" in race_df.columns else np.nan

            eligible_mask = (race_odds > 0) & (race_odds <= cap)
            has_eligible = eligible_mask.any()

            if has_eligible:
                elig_idx = race_score.loc[eligible_mask].idxmax()
                final_umaban = (
                    race_df.loc[elig_idx, "umaban"] if "umaban" in race_df.columns else np.nan
                )
                if orig_idx != elig_idx:
                    result_indices.append(elig_idx)
                    applied_map[rid] = True
                    reason_map[rid] = "odds_cap_switch"
                else:
                    result_indices.append(orig_idx)
                    applied_map[rid] = False
                    reason_map[rid] = "top1_within_cap"
                final_top1_map[rid] = final_umaban
            else:
                result_indices.append(orig_idx)
                applied_map[rid] = False
                reason_map[rid] = "no_eligible"
                final_top1_map[rid] = orig_umaban

            orig_top1_map[rid] = orig_umaban

        if not result_indices:
            return self._annotate_empty_with_gap(candidates)

        result = candidates.loc[result_indices].copy()

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

    def _apply_gap_reranker(
        self,
        cap_result: pd.DataFrame,
        *,
        all_candidates: pd.DataFrame,
        surface: str,
    ) -> pd.DataFrame:
        """Gap reranker を cap 適用済み結果に second stage として適用する.

        cap_result: _apply_odds_cap の出力 (1行/race, reranker_* 列付き)
        all_candidates: cap 適用前の全候補 (全馬)
        surface: "dirt" または ""
        """
        if cap_result.empty:
            return cap_result

        # gap reranker 未デプロイまたは非ダート → gap 診断列だけ付加
        if not self.gap_reranker_deployed or surface != "dirt":
            cap_result[GAP_RERANKER_DEPLOYED_COL] = False
            cap_result[GAP_RERANKER_APPLIED_COL] = False
            cap_result[GAP_RERANKER_ORIG_TOP1_COL] = np.nan
            cap_result[GAP_RERANKER_FINAL_TOP1_COL] = np.nan
            cap_result[GAP_RERANKER_SCORE_GAP_COL] = np.nan
            cap_result[GAP_RERANKER_PROB_MARGIN_COL] = np.nan
            cap_result[GAP_RERANKER_THRESHOLD_COL] = np.nan
            cap_result[GAP_RERANKER_SWITCH_REASON_COL] = "not_deployed"
            return cap_result

        # デプロイ済みダート → per-race gap 評価
        key_col = "race_id" if "race_id" in cap_result.columns else None
        score_all = _numeric(all_candidates, "win_market_selection_score").fillna(float("-inf"))
        odds_all = _numeric(all_candidates, "tanodds").fillna(0.0)
        cap = self.selected_cap

        # 全 race の gap 評価結果を収集
        final_rows: list[pd.Series] = []
        n_switched = 0

        for result_idx, result_row in cap_result.iterrows():
            rid = result_row[key_col] if key_col else "_race"

            # この race の全候補を取得
            if key_col and key_col in all_candidates.columns:
                race_mask = all_candidates[key_col] == rid
                race_cands = all_candidates.loc[race_mask]
            else:
                race_cands = all_candidates

            if race_cands.empty or len(race_cands) < 2:
                # 候補不足 → gap 診断だけ付加してそのまま
                row = result_row.copy()
                row[GAP_RERANKER_DEPLOYED_COL] = True
                row[GAP_RERANKER_APPLIED_COL] = False
                row[GAP_RERANKER_ORIG_TOP1_COL] = result_row.get(RERANKER_FINAL_TOP1_COL, np.nan)
                row[GAP_RERANKER_FINAL_TOP1_COL] = row[GAP_RERANKER_ORIG_TOP1_COL]
                row[GAP_RERANKER_SCORE_GAP_COL] = np.nan
                row[GAP_RERANKER_PROB_MARGIN_COL] = np.nan
                row[GAP_RERANKER_THRESHOLD_COL] = self.gap_reranker_score_gap_threshold
                row[GAP_RERANKER_SWITCH_REASON_COL] = "insufficient_candidates"
                final_rows.append(row)
                continue

            # top1 = cap 適用済み馬 (result_row)
            top1_score_val = float(result_row.get("win_market_selection_score", float("-inf")))
            top1_prob = self._get_prob_value(result_row, all_candidates, result_idx)

            # cap-eligible 馬から top2 を探す
            race_scores = score_all.loc[race_cands.index]
            race_odds = odds_all.loc[race_cands.index]
            cap_mask = (race_odds > 0) & (race_odds <= cap)

            if cap_mask.sum() < 2:
                row = result_row.copy()
                row[GAP_RERANKER_DEPLOYED_COL] = True
                row[GAP_RERANKER_APPLIED_COL] = False
                row[GAP_RERANKER_ORIG_TOP1_COL] = result_row.get(RERANKER_FINAL_TOP1_COL, np.nan)
                row[GAP_RERANKER_FINAL_TOP1_COL] = row[GAP_RERANKER_ORIG_TOP1_COL]
                row[GAP_RERANKER_SCORE_GAP_COL] = np.nan
                row[GAP_RERANKER_PROB_MARGIN_COL] = np.nan
                row[GAP_RERANKER_THRESHOLD_COL] = self.gap_reranker_score_gap_threshold
                row[GAP_RERANKER_SWITCH_REASON_COL] = (
                    "no_cap_eligible" if cap_mask.sum() == 0 else "single_cap_eligible"
                )
                final_rows.append(row)
                continue

            # top1 index を特定 (score 最大の cap-eligible 馬)
            top1_in_race_idx = race_scores.loc[cap_mask].idxmax()
            top2_candidates = race_cands.drop(top1_in_race_idx)
            if top2_candidates.empty:
                row = result_row.copy()
                row[GAP_RERANKER_DEPLOYED_COL] = True
                row[GAP_RERANKER_APPLIED_COL] = False
                row[GAP_RERANKER_ORIG_TOP1_COL] = result_row.get(RERANKER_FINAL_TOP1_COL, np.nan)
                row[GAP_RERANKER_FINAL_TOP1_COL] = row[GAP_RERANKER_ORIG_TOP1_COL]
                row[GAP_RERANKER_SCORE_GAP_COL] = np.nan
                row[GAP_RERANKER_PROB_MARGIN_COL] = np.nan
                row[GAP_RERANKER_THRESHOLD_COL] = self.gap_reranker_score_gap_threshold
                row[GAP_RERANKER_SWITCH_REASON_COL] = "single_candidate"
                final_rows.append(row)
                continue

            # top2 = 残りの cap-eligible で score 最大
            t2_scores = score_all.loc[top2_candidates.index]
            t2_odds_vals = odds_all.loc[top2_candidates.index]
            t2_cap_mask = (t2_odds_vals > 0) & (t2_odds_vals <= cap)
            if not t2_cap_mask.any():
                # cap内top2なし → gap rerank不可
                row = result_row.copy()
                row[GAP_RERANKER_DEPLOYED_COL] = True
                row[GAP_RERANKER_APPLIED_COL] = False
                row[GAP_RERANKER_ORIG_TOP1_COL] = result_row.get(RERANKER_FINAL_TOP1_COL, np.nan)
                row[GAP_RERANKER_FINAL_TOP1_COL] = row[GAP_RERANKER_ORIG_TOP1_COL]
                row[GAP_RERANKER_SCORE_GAP_COL] = np.nan
                row[GAP_RERANKER_PROB_MARGIN_COL] = np.nan
                row[GAP_RERANKER_THRESHOLD_COL] = self.gap_reranker_score_gap_threshold
                row[GAP_RERANKER_SWITCH_REASON_COL] = "no_cap_eligible_top2"
                final_rows.append(row)
                continue
            top2_idx = t2_scores.loc[t2_cap_mask].idxmax()
            top2_score_val = float(t2_scores.at[top2_idx])
            top2_prob = self._get_prob_value(
                pd.Series(),
                all_candidates,
                top2_idx,
            )

            score_gap = top1_score_val - top2_score_val
            prob_margin = top2_prob - top1_prob

            # 切替条件チェック
            top2_prob_higher = top2_prob > top1_prob
            gap_within = score_gap <= self.gap_reranker_score_gap_threshold
            margin_ok = prob_margin >= self.gap_reranker_min_prob_margin

            switch = top2_prob_higher and gap_within and margin_ok

            # 追加条件
            if switch and self.gap_reranker_top2_odds_le_top1:
                t2_odds = float(odds_all.at[top2_idx])
                t1_odds = float(
                    all_candidates.at[top1_in_race_idx, "tanodds"]
                    if "tanodds" in all_candidates.columns
                    else 0.0
                )
                if t2_odds > t1_odds:
                    switch = False
            if switch and self.gap_reranker_top2_max_odds < float("inf"):
                t2_odds = float(odds_all.at[top2_idx])
                if t2_odds > self.gap_reranker_top2_max_odds:
                    switch = False

            orig_umaban = result_row.get(RERANKER_FINAL_TOP1_COL, np.nan)

            if switch:
                # top2 行を取得し、reranker 診断列を引き継ぐ
                new_row = all_candidates.loc[top2_idx].copy()
                for rc in RERANKER_DIAGNOSTIC_COLS:
                    if rc in result_row.index:
                        new_row[rc] = result_row[rc]
                new_row[GAP_RERANKER_DEPLOYED_COL] = True
                new_row[GAP_RERANKER_APPLIED_COL] = True
                new_row[GAP_RERANKER_ORIG_TOP1_COL] = orig_umaban
                new_row[GAP_RERANKER_FINAL_TOP1_COL] = new_row.get("umaban", np.nan)
                new_row[GAP_RERANKER_SCORE_GAP_COL] = score_gap
                new_row[GAP_RERANKER_PROB_MARGIN_COL] = prob_margin
                new_row[GAP_RERANKER_THRESHOLD_COL] = self.gap_reranker_score_gap_threshold
                new_row[GAP_RERANKER_SWITCH_REASON_COL] = "prob_gap_switch"
                final_rows.append(new_row)
                n_switched += 1
            else:
                row = result_row.copy()
                row[GAP_RERANKER_DEPLOYED_COL] = True
                row[GAP_RERANKER_APPLIED_COL] = False
                row[GAP_RERANKER_ORIG_TOP1_COL] = orig_umaban
                row[GAP_RERANKER_FINAL_TOP1_COL] = orig_umaban
                row[GAP_RERANKER_SCORE_GAP_COL] = score_gap
                row[GAP_RERANKER_PROB_MARGIN_COL] = prob_margin
                row[GAP_RERANKER_THRESHOLD_COL] = self.gap_reranker_score_gap_threshold
                if not top2_prob_higher:
                    row[GAP_RERANKER_SWITCH_REASON_COL] = "top2_prob_lower"
                elif not gap_within:
                    row[GAP_RERANKER_SWITCH_REASON_COL] = "score_gap_exceeds"
                elif not margin_ok:
                    row[GAP_RERANKER_SWITCH_REASON_COL] = "margin_insufficient"
                else:
                    row[GAP_RERANKER_SWITCH_REASON_COL] = "odds_condition_failed"
                final_rows.append(row)

        # max_change_rate は学習/holdoutのdeploy guard専用。
        # per-race applyでは変更率guardを適用しない (1 race = 100% で全取消となるため)。
        # 変更率はdiagnosticで監視: gap_reranker_switch_reason で判断可能。

        result = pd.DataFrame(final_rows)
        return result

    def _annotate_empty_with_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        """空 DataFrame に reranker + gap 診断列を追加して返す."""
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
        for col in GAP_RERANKER_DIAGNOSTIC_COLS:
            if col == GAP_RERANKER_DEPLOYED_COL or col == GAP_RERANKER_APPLIED_COL:
                df[col] = pd.Series(dtype=bool, index=df.index)
            elif col == GAP_RERANKER_THRESHOLD_COL:
                df[col] = np.nan
            elif col == GAP_RERANKER_SWITCH_REASON_COL:
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
        # Gap reranker diagnostics (not deployed in this path)
        result[GAP_RERANKER_DEPLOYED_COL] = False
        result[GAP_RERANKER_APPLIED_COL] = False
        result[GAP_RERANKER_ORIG_TOP1_COL] = np.nan
        result[GAP_RERANKER_FINAL_TOP1_COL] = np.nan
        result[GAP_RERANKER_SCORE_GAP_COL] = np.nan
        result[GAP_RERANKER_PROB_MARGIN_COL] = np.nan
        result[GAP_RERANKER_THRESHOLD_COL] = np.nan
        result[GAP_RERANKER_SWITCH_REASON_COL] = "cap_not_applied"
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
                # Gap reranker fields
                "gap_reranker_deployed": self.gap_reranker_deployed,
                "gap_reranker_surface": self.gap_reranker_surface,
                "gap_reranker_max_change_rate": self.gap_reranker_max_change_rate,
                "gap_reranker_min_prob_margin": self.gap_reranker_min_prob_margin,
                "gap_reranker_score_gap_threshold": self.gap_reranker_score_gap_threshold,
                "gap_reranker_top2_odds_le_top1": self.gap_reranker_top2_odds_le_top1,
                "gap_reranker_top2_max_odds": self.gap_reranker_top2_max_odds,
                "gap_reranker_training_summary": self.gap_reranker_training_summary,
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
        # Gap reranker fields (backward compatible: missing keys → defaults)
        model.gap_reranker_deployed = bool(state.get("gap_reranker_deployed", False))
        model.gap_reranker_surface = str(state.get("gap_reranker_surface", ""))
        model.gap_reranker_max_change_rate = float(state.get("gap_reranker_max_change_rate", 0.10))
        model.gap_reranker_min_prob_margin = float(state.get("gap_reranker_min_prob_margin", 0.01))
        model.gap_reranker_score_gap_threshold = float(
            state.get("gap_reranker_score_gap_threshold", float("inf"))
        )
        model.gap_reranker_top2_odds_le_top1 = bool(
            state.get("gap_reranker_top2_odds_le_top1", False)
        )
        model.gap_reranker_top2_max_odds = float(
            state.get("gap_reranker_top2_max_odds", float("inf"))
        )
        model.gap_reranker_training_summary = dict(state.get("gap_reranker_training_summary", {}))
        return model
