"""horse_history_features.py — 馬の過去成績ベース特徴量

主な特徴量:
  - norm_finish_logit_avg: 着順をログット変換したスコアの平均
  - harontimel3_avg: 直近3走のハロンタイム平均
  - harontimel3_zscore: 距離ビンz-scoreの直近3走平均
  - timediff_avg: 直近3走のタイム差平均
  - jyuni1c_avg: 直近3走の1コーナー通過位置平均
  - jyuni4c_avg: 直近3走の4コーナー通過位置平均
  - closing_index_avg: (4C正規化 - 着順正規化) の直近3走平均
  - kyakusitukubun_cd: 直近走の脚質コード
  - jockey_surprise: Beta事前分布でスムージングした騎手勝率サプライズ
  - jockey_cond_wr: 騎手条件別勝率
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

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


def _norm_finish_logit_vec(finish_pos: np.ndarray, field_size: np.ndarray) -> np.ndarray:
    """Vectorized version of _norm_finish_logit."""
    score = 1.0 - (finish_pos - 1) / np.maximum(field_size - 1, 1)
    score = np.clip(score, CLIP_LO, CLIP_HI)
    result = np.log(score / (1.0 - score))
    result[field_size < 8] = np.nan
    return result


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
        "harontimel3_avg",  # NEW - replaces haron_time_zscore_avg
        "harontimel3_zscore",  # NEW
        "timediff_avg",  # NEW
        "jyuni1c_avg",  # NEW
        "jyuni4c_avg",  # NEW
        "closing_index_avg",  # NEW
        "kyakusitukubun_cd",  # NEW (non-numeric, category)
        "jockey_surprise",  # existing (will move to Stage2 in Task 9)
        "jockey_cond_wr",  # existing (will move to Stage2 in Task 9)
    ]

    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._entries_cache: pd.DataFrame | None = None
        self._races_cache: pd.DataFrame | None = None

    def _get_history(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load history entries and races (cached)."""
        if self._entries_cache is None:
            from db.readers import load_history_entries

            self._entries_cache = load_history_entries(self.store)
        if self._races_cache is None:
            from db.readers import load_history_races

            self._races_cache = load_history_races(self.store)
        return self._entries_cache, self._races_cache

    def compute(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        target_race_ids: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """過去成績特徴量を計算（pre-indexed lookup + searchsorted 高速版）"""
        if target_race_ids is not None:
            entry_df = entry_df[entry_df["race_id"].isin(target_race_ids)]

        # 対象レースの馬・騎手リスト
        horses = entry_df[["race_id", "umaban", "kettonum", "kisyucode"]].copy()
        if "race_date" not in horses.columns:
            date_map = race_df.set_index("race_id")["race_date"]
            horses["race_date"] = horses["race_id"].map(date_map)

        unique_ketto = horses["kettonum"].unique().tolist()
        unique_kisyu = horses["kisyucode"].unique().tolist()

        if not unique_ketto:
            return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

        # Load history data via repository
        entries_hist, races_hist = self._get_history()

        # Filter to relevant horses/jockeys
        ketto_set = set(unique_ketto)
        kisyu_set = set(unique_kisyu)

        entries_filtered = entries_hist[
            entries_hist["kettonum"].isin(ketto_set) | entries_hist["kisyucode"].isin(kisyu_set)
        ].copy()

        if entries_filtered.empty:
            return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

        # Merge with races to get syussotosu (field_size), race_date, surface
        race_cols = ["race_id", "syussotosu", "race_date", "trackcd", "kyori", "surface"]
        races_subset = races_hist[races_hist["race_id"].isin(entries_filtered["race_id"].unique())]
        entries_no_date = entries_filtered.drop(columns=["race_date"], errors="ignore")
        past_df = entries_no_date.merge(
            races_subset[race_cols].drop_duplicates("race_id"),
            on="race_id",
            how="left",
        )

        # Add distance_bin (surface is computed by ETL from trackcd)
        if (
            "distance_bin" not in past_df.columns
            and "kyori" in past_df.columns
            and "surface" in past_df.columns
        ):
            is_turf = past_df["surface"] == "turf"
            dist = past_df["kyori"]
            past_df["distance_bin"] = "unknown"
            past_df.loc[is_turf & (dist > 2100), "distance_bin"] = "long"
            past_df.loc[is_turf & (dist <= 2100), "distance_bin"] = "intermediate"
            past_df.loc[is_turf & (dist <= 1700), "distance_bin"] = "mile"
            past_df.loc[is_turf & (dist <= 1400), "distance_bin"] = "sprint"
            past_df.loc[~is_turf & (dist > 1700), "distance_bin"] = "intermediate"
            past_df.loc[~is_turf & (dist <= 1700), "distance_bin"] = "mile"
            past_df.loc[~is_turf & (dist <= 1400), "distance_bin"] = "sprint"

        past_df["valid_field"] = (past_df["syussotosu"].fillna(-1) >= 8).astype(int)

        # Pre-index past data by kettonum (sorted by race_date)
        past_df_sorted = past_df.sort_values(["kettonum", "race_date"]).reset_index(drop=True)
        past_by_ketto: dict[str, pd.DataFrame] = {
            k: g.reset_index(drop=True) for k, g in past_df_sorted.groupby("kettonum")
        }

        # Pre-index past data by kisyucode for jockey_surprise (kakuteijyuni > 0 AND odds > 0)
        past_by_kisyu: dict[str, pd.DataFrame] = {
            k: g.reset_index(drop=True)
            for k, g in past_df_sorted[
                (past_df_sorted["kakuteijyuni"] > 0) & (past_df_sorted["odds"] > 0)
            ].groupby("kisyucode")
        }

        # Pre-index past data by kisyucode for jockey_cond_wr
        # (kakuteijyuni > 0 only, no odds filter)
        past_by_kisyu_all: dict[str, pd.DataFrame] = {
            k: g.reset_index(drop=True)
            for k, g in past_df_sorted[(past_df_sorted["kakuteijyuni"] > 0)].groupby("kisyucode")
        }

        # Weight column name
        weight_col = "bataijyu" if "bataijyu" in entry_df.columns else "weight"

        total = len(horses)
        results: list[dict] = []
        empty_past = pd.DataFrame()

        for i, (_, row) in enumerate(horses.iterrows()):
            if i % 200 == 0:
                print(
                    f"  HorseHistoryFeatures: {i}/{total} ({i / max(total, 1) * 100:.0f}%)",
                    flush=True,
                )
            race_date = row["race_date"]
            ketto = row["kettonum"]
            kisyu = row["kisyucode"]

            # --- Horse features: O(1) lookup + O(log m) searchsorted ---
            horse_past_all = past_by_ketto.get(ketto, empty_past)
            if len(horse_past_all) > 0:
                valid_past = horse_past_all[
                    (horse_past_all["valid_field"] == 1) & (horse_past_all["kakuteijyuni"] > 0)
                ]
                # searchsorted for date cutoff (ensure datetime64 types)
                if len(valid_past) > 0:
                    dates_np = valid_past["race_date"].values.astype("datetime64[ns]")
                    target_date_np = np.datetime64(race_date, "ns")
                    idx = dates_np.searchsorted(target_date_np, side="left")
                    horse_past = valid_past.iloc[max(0, idx - 3) : idx]
                else:
                    horse_past = valid_past
            else:
                horse_past = empty_past

            # norm_finish_logit_avg
            if len(horse_past) > 0:
                logits = _norm_finish_logit_vec(
                    horse_past["kakuteijyuni"].values.astype(float),
                    horse_past["syussotosu"].values.astype(float),
                )
                norm_finish_logit_avg: float = float(np.nanmean(logits))
            else:
                norm_finish_logit_avg = float("nan")

            # harontimel3_avg
            if "harontimel3" in horse_past.columns and len(horse_past) > 0:
                ht_vals = horse_past["harontimel3"].dropna()
                harontimel3_avg: float = (
                    float(ht_vals.tail(3).mean()) if len(ht_vals) > 0 else float("nan")
                )
            else:
                harontimel3_avg = float("nan")

            # harontimel3_zscore
            if (
                "harontimel3" in horse_past.columns
                and "distance_bin" in horse_past.columns
                and len(horse_past) > 0
            ):
                ht = horse_past["harontimel3"]
                db = horse_past["distance_bin"]
                valid = ht.notna() & db.notna()
                if valid.sum() > 0:
                    grp_stats = (
                        horse_past.loc[valid]
                        .groupby("distance_bin")["harontimel3"]
                        .agg(["mean", "std"])
                    )
                    zscores: list[float] = []
                    for _, r in horse_past.loc[valid].iterrows():
                        bin_key = r["distance_bin"]
                        if bin_key in grp_stats.index and grp_stats.loc[bin_key, "std"] > 0:
                            z = (r["harontimel3"] - grp_stats.loc[bin_key, "mean"]) / grp_stats.loc[
                                bin_key, "std"
                            ]
                            zscores.append(z)
                        else:
                            zscores.append(float("nan"))
                    harontimel3_zscore: float = (
                        float(pd.Series(zscores).tail(3).mean()) if zscores else float("nan")
                    )
                else:
                    harontimel3_zscore = float("nan")
            else:
                harontimel3_zscore = float("nan")

            # timediff_avg
            if "timediff" in horse_past.columns and len(horse_past) > 0:
                td_vals = horse_past["timediff"].dropna()
                timediff_avg: float = (
                    float(td_vals.tail(3).mean()) if len(td_vals) > 0 else float("nan")
                )
            else:
                timediff_avg = float("nan")

            # jyuni1c_avg
            if "jyuni1c" in horse_past.columns and len(horse_past) > 0:
                c1_vals = horse_past["jyuni1c"].dropna()
                jyuni1c_avg: float = (
                    float(c1_vals.tail(3).mean()) if len(c1_vals) > 0 else float("nan")
                )
            else:
                jyuni1c_avg = float("nan")

            # jyuni4c_avg
            if "jyuni4c" in horse_past.columns and len(horse_past) > 0:
                c4_vals = horse_past["jyuni4c"].dropna()
                jyuni4c_avg: float = (
                    float(c4_vals.tail(3).mean()) if len(c4_vals) > 0 else float("nan")
                )
            else:
                jyuni4c_avg = float("nan")

            # closing_index_avg
            if (
                all(c in horse_past.columns for c in ["jyuni4c", "kakuteijyuni", "syussotosu"])
                and len(horse_past) > 0
            ):
                valid_ci = horse_past.dropna(subset=["jyuni4c", "kakuteijyuni", "syussotosu"])
                valid_ci = valid_ci[valid_ci["syussotosu"] > 1]
                if len(valid_ci) > 0:
                    norm_4c = (valid_ci["jyuni4c"] - 1) / (valid_ci["syussotosu"] - 1)
                    norm_finish = (valid_ci["kakuteijyuni"] - 1) / (valid_ci["syussotosu"] - 1)
                    closing_indices = norm_4c - norm_finish
                    closing_index_avg: float = float(closing_indices.tail(3).mean())
                else:
                    closing_index_avg = float("nan")
            else:
                closing_index_avg = float("nan")

            # kyakusitukubun_cd
            if "kyakusitukubun" in horse_past.columns and len(horse_past) > 0:
                kt_vals = horse_past["kyakusitukubun"].dropna()
                kyakusitukubun_cd: float | int = (
                    int(kt_vals.iloc[-1]) if len(kt_vals) > 0 else float("nan")
                )
            else:
                kyakusitukubun_cd = float("nan")

            # --- Jockey features: O(1) lookup + O(log m) searchsorted ---
            jockey_past_all = past_by_kisyu.get(kisyu, empty_past)
            if len(jockey_past_all) > 0:
                dates_np = jockey_past_all["race_date"].values.astype("datetime64[ns]")
                target_date_np = np.datetime64(race_date, "ns")
                idx = dates_np.searchsorted(target_date_np, side="left")
                jockey_past = jockey_past_all.iloc[max(0, idx - 100) : idx]
            else:
                jockey_past = empty_past

            if len(jockey_past) >= 30:
                expected = (PAYOUT_RATE / jockey_past["odds"].clip(lower=1.1)).sum()
                actual = int((jockey_past["kakuteijyuni"] == 1).sum())
                jockey_surprise: float = _compute_jockey_surprise(
                    actual, len(jockey_past), expected
                )
            else:
                jockey_surprise = float("nan")

            # jockey_cond_wr — uses past_by_kisyu_all (kakuteijyuni > 0 only, no odds filter)
            jockey_all_past = past_by_kisyu_all.get(kisyu, empty_past)
            if len(jockey_all_past) > 0:
                dates_np = jockey_all_past["race_date"].values.astype("datetime64[ns]")
                target_date_np = np.datetime64(race_date, "ns")
                idx = dates_np.searchsorted(target_date_np, side="left")
                jockey_all = jockey_all_past.iloc[:idx]
                total_rides = len(jockey_all)
            else:
                jockey_all = empty_past
                total_rides = 0
            total_wins = int((jockey_all["kakuteijyuni"] == 1).sum()) if total_rides > 0 else 0

            k_smooth = 25
            if total_rides >= 10:
                cond_wr = total_wins / max(total_rides, 1)
                global_wr = total_wins / max(total_rides, 1)
                w = min(total_rides / (total_rides + k_smooth), 1.0)
                jockey_cond_wr: float = float(w * cond_wr + (1 - w) * global_wr)
            else:
                jockey_cond_wr = float("nan")

            # weight_absolute
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
                    "harontimel3_avg": harontimel3_avg,
                    "harontimel3_zscore": harontimel3_zscore,
                    "timediff_avg": timediff_avg,
                    "jyuni1c_avg": jyuni1c_avg,
                    "jyuni4c_avg": jyuni4c_avg,
                    "closing_index_avg": closing_index_avg,
                    "kyakusitukubun_cd": kyakusitukubun_cd,
                    "jockey_surprise": jockey_surprise,
                    "jockey_cond_wr": jockey_cond_wr,
                    "weight_absolute": weight_absolute,
                }
            )

        print(f"  HorseHistoryFeatures: done ({len(results)} rows)", flush=True)
        return pd.DataFrame(results)

    @staticmethod
    def add_race_transforms(df: pd.DataFrame) -> pd.DataFrame:
        """数値BASE_COLS の各列についてレース内 percentile rank を追加。
        カテゴリ列 (kyakusitukubun_cd 等) は除外。jockey系は Stage2 に移動後、
        race_rank を生成しない (Task 9 で BASE_COLS から除外)。
        """
        # race_rank を生成する列を明示 (数値のみ)
        race_rank_cols = [
            "norm_finish_logit_avg",
            "harontimel3_avg",
            "harontimel3_zscore",
            "timediff_avg",
            "jyuni1c_avg",
            "jyuni4c_avg",
            "closing_index_avg",
            # 注意: kyakusitukubun_cd, jockey_surprise, jockey_cond_wr は race_rank を生成しない
        ]
        df = df.copy()
        for col in race_rank_cols:
            if col not in df.columns:
                continue
            df[f"{col}_race_rank"] = df.groupby("race_id")[col].rank(pct=True, method="average")
        return df
