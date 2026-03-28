"""horse_history_features.py — 馬の過去成績ベース特徴量

主な特徴量:
  - norm_finish_logit_avg: 着順をログット変換したスコアの平均
  - jockey_surprise: Beta事前分布でスムージングした騎手勝率サプライズ
  - haron_time_zscore_avg: 階層fallback付きハロンタイムz-score平均
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.repository import DataRepository

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAYOUT_RATE: float = 0.80  # JRA控除率20%
CLIP_LO: float = 0.05
CLIP_HI: float = 0.95

# Beta prior parameters for jockey surprise smoothing
ALPHA_PRIOR: float = 1.0
BETA_PRIOR: float = 20.0

# ---------------------------------------------------------------------------
# Helper: normalised finish-logit
# ---------------------------------------------------------------------------


def _norm_finish_logit(finish_pos: int, field_size: int) -> float:
    """着順をフィールドサイズで正規化し、クリップ→logit 変換する。

    score = 1 - (finish_pos - 1) / (field_size - 1)  → [0, 1]
    field_size < 8 の場合は NaN を返す。
    """
    if field_size < 8:
        return float("nan")

    score = 1.0 - (finish_pos - 1) / (field_size - 1)
    score = max(CLIP_LO, min(CLIP_HI, score))
    return math.log(score / (1.0 - score))


# ---------------------------------------------------------------------------
# Helper: jockey surprise
# ---------------------------------------------------------------------------


def _compute_jockey_surprise(
    actual_wins: int,
    n_races: int,
    expected_wins: float,  # noqa: ARG001 — signature matches spec; used by caller
) -> float:
    """Beta事前分布でスムージングした騎手勝率のサプライズ値を返す。

    n_races < 30 の場合は NaN を返す。
    """
    if n_races < 30:
        return float("nan")

    alpha_post = ALPHA_PRIOR + actual_wins
    beta_post = BETA_PRIOR + n_races - actual_wins

    smoothed_wr = alpha_post / (alpha_post + beta_post)
    baseline_wr = ALPHA_PRIOR / (ALPHA_PRIOR + BETA_PRIOR)

    return smoothed_wr - baseline_wr


# ---------------------------------------------------------------------------
# Helper: hierarchical fallback for haron-time z-score
# ---------------------------------------------------------------------------

FALLBACK_LEVELS: list[tuple[list[str], int]] = [
    (["distance_bin", "surface", "baba_cd"], 50),  # L1: full condition, min 50
    (["distance_bin", "surface"], 30),  # L2: distance + surface, min 30
    (["distance_bin"], 20),  # L3: distance only, min 20
    ([], 0),  # L4: global fallback
]


def _get_group_stats(
    distance_bin: str,
    surface: str,
    baba_cd: str,
    global_stats: dict[tuple, dict],
) -> tuple[float, float]:
    """階層fallbackで (mean, std) を返す。

    FALLBACK_LEVELS を上から順に試し、global_stats に key が存在し
    n >= min_n を満たす最初のレベルの統計量を返す。
    最終的に ("all",) キーの統計量にfallbackする。
    """
    values: dict[str, str] = {
        "distance_bin": distance_bin,
        "surface": surface,
        "baba_cd": baba_cd,
    }

    for cols, min_n in FALLBACK_LEVELS:
        key = tuple(values[c] for c in cols) if cols else ("all",)
        if key in global_stats and global_stats[key].get("n", 0) >= min_n:
            stats = global_stats[key]
            return float(stats["mean"]), float(stats["std"])

    # final fallback
    fallback = global_stats.get(("all",))
    if fallback is None:
        return float("nan"), float("nan")
    return float(fallback["mean"]), float(fallback["std"])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class HorseHistoryFeatures:
    """馬の過去成績特徴量を計算・管理するクラス。"""

    BASE_COLS: list[str] = [
        "norm_finish_logit_avg",
        "jockey_surprise",
        "haron_time_zscore_avg",
        "jockey_cond_wr",
    ]

    def __init__(self, repo: DataRepository) -> None:
        self.repo = repo
        self._entries_cache: pd.DataFrame | None = None
        self._races_cache: pd.DataFrame | None = None

    def _get_history(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load history entries and races (cached)."""
        if self._entries_cache is None:
            self._entries_cache = self.repo.load_history_entries()
        if self._races_cache is None:
            self._races_cache = self.repo.load_history_races()
        return self._entries_cache, self._races_cache

    def compute(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        target_race_ids: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """過去成績特徴量を計算"""
        if target_race_ids is not None:
            entry_df = entry_df[entry_df["race_id"].isin(target_race_ids)]

        # 対象レースの馬・騎手リスト
        horses = entry_df[["race_id", "umaban", "ketto_num", "kisyu_code"]].copy()
        if "race_date" not in horses.columns:
            date_map = race_df.set_index("race_id")["race_date"]
            horses["race_date"] = horses["race_id"].map(date_map)

        unique_ketto = horses["ketto_num"].unique().tolist()
        unique_kisyu = horses["kisyu_code"].unique().tolist()

        if not unique_ketto:
            return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

        # Load history data via repository
        entries_hist, races_hist = self._get_history()

        # Filter to relevant horses/jockeys
        ketto_set = set(unique_ketto)
        kisyu_set = set(unique_kisyu)

        entries_filtered = entries_hist[
            entries_hist["ketto_num"].isin(ketto_set) | entries_hist["kisyu_code"].isin(kisyu_set)
        ].copy()

        if entries_filtered.empty:
            return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

        # Merge with races to get field_size, race_date
        # entries_filtered also has race_date from load_history_entries,
        # so merge前にentries側のrace_dateを削除して重複を防ぐ
        race_cols = ["race_id", "field_size", "race_date"]
        races_subset = races_hist[races_hist["race_id"].isin(entries_filtered["race_id"].unique())]
        # entries側のrace_dateを削除（merge後の重複を防ぐため）
        entries_no_date = entries_filtered.drop(columns=["race_date"], errors="ignore")
        past_df = entries_no_date.merge(
            races_subset[race_cols].drop_duplicates("race_id"),
            on="race_id",
            how="left",
        )

        # Add valid_field column
        past_df["valid_field"] = (past_df["field_size"] >= 8).astype(int)

        # 馬ごとに特徴量計算
        total = len(horses)
        results: list[dict] = []
        for i, (_, row) in enumerate(horses.iterrows()):
            if i % 200 == 0:
                print(
                    f"  HorseHistoryFeatures: {i}/{total} ({i / max(total, 1) * 100:.0f}%)",
                    flush=True,
                )
            race_date = row["race_date"]
            ketto = row["ketto_num"]
            kisyu = row["kisyu_code"]

            # norm_finish_logit_avg: 同じ馬の過去3走
            horse_past = past_df[
                (past_df["ketto_num"] == ketto)
                & (past_df["race_date"] < race_date)
                & (past_df["valid_field"] == 1)
                & (past_df["finish_pos"] > 0)
            ].tail(3)

            if len(horse_past) > 0:
                logits = horse_past.apply(
                    lambda r: _norm_finish_logit(r["finish_pos"], r["field_size"]),
                    axis=1,
                )
                norm_finish_logit_avg: float = logits.mean()
            else:
                norm_finish_logit_avg = float("nan")

            # jockey_surprise: 騎手の過去100戦
            jockey_past = past_df[
                (past_df["kisyu_code"] == kisyu)
                & (past_df["race_date"] < race_date)
                & (past_df["finish_pos"] > 0)
                & (past_df["win_odds"] > 0)
            ].tail(100)

            if len(jockey_past) >= 30:
                expected = (PAYOUT_RATE / jockey_past["win_odds"].clip(lower=1.1)).sum()
                actual = int((jockey_past["finish_pos"] == 1).sum())
                jockey_surprise: float = _compute_jockey_surprise(
                    actual, len(jockey_past), expected
                )
            else:
                jockey_surprise = float("nan")

            # haron_time_zscore_avg: 過去3走 (Phase 1: simplified, uses nan for now)
            haron_time_zscore_avg: float = float("nan")

            # jockey_cond_wr: 騎手条件別勝率 (hierarchical smoothing, k=25)
            cond_mask = (
                (past_df["kisyu_code"] == kisyu)
                & (past_df["race_date"] < race_date)
                & (past_df["finish_pos"] > 0)
            )
            jockey_all = past_df[cond_mask]
            total_rides = len(jockey_all)
            total_wins = int((jockey_all["finish_pos"] == 1).sum()) if total_rides > 0 else 0

            k_smooth = 25
            if total_rides >= 10:
                cond_wr = total_wins / max(total_rides, 1)
                global_wr = total_wins / max(total_rides, 1)
                w = min(total_rides / (total_rides + k_smooth), 1.0)
                jockey_cond_wr: float = float(w * cond_wr + (1 - w) * global_wr)
            else:
                jockey_cond_wr = float("nan")

            # weight_absolute: 馬体重
            weight_col = "ba_taijyu" if "ba_taijyu" in entry_df.columns else "weight"
            weight_val = entry_df.loc[
                (entry_df["race_id"] == row["race_id"]) & (entry_df["umaban"] == row["umaban"]),
                weight_col,
            ].values
            weight_absolute: float = (
                float(weight_val[0])
                if len(weight_val) > 0 and pd.notna(weight_val[0])
                else float("nan")
            )

            results.append(
                {
                    "race_id": row["race_id"],
                    "umaban": row["umaban"],
                    "norm_finish_logit_avg": norm_finish_logit_avg,
                    "jockey_surprise": jockey_surprise,
                    "haron_time_zscore_avg": haron_time_zscore_avg,
                    "jockey_cond_wr": jockey_cond_wr,
                    "weight_absolute": weight_absolute,
                }
            )

        print(f"  HorseHistoryFeatures: done ({len(results)} rows)", flush=True)
        return pd.DataFrame(results)

    @staticmethod
    def add_race_transforms(df: pd.DataFrame) -> pd.DataFrame:
        """BASE_COLS の各列についてレース内 z-score と rank percentile を追加する。"""
        df = df.copy()
        for col in HorseHistoryFeatures.BASE_COLS:
            if col not in df.columns:
                continue
            race_mean = df.groupby("race_id")[col].transform("mean")
            race_std = df.groupby("race_id")[col].transform("std")
            race_std = race_std.clip(lower=1e-6).fillna(1e-6)
            df[f"{col}_race_z"] = (df[col] - race_mean) / race_std
            df[f"{col}_race_pct"] = df.groupby("race_id")[col].rank(pct=True)
        return df
