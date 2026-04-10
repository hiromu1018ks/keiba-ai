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

from features.form_cycle_features import compute_form_features

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


def _lookup_expanding_stats(
    target_date: np.datetime64,
    db_val: str,
    surf_val: str,
    baba_val: str,
    expanding_stats: dict[tuple, np.ndarray],
) -> tuple[float, float]:
    """expanding_stats から target_date 以前の最新の mean/std を取得。

    Returns (mean, std). 見つからない場合は (nan, nan).
    """
    for cols, _min_n in FALLBACK_LEVELS:
        if cols:
            col_map = {"distance_bin": db_val, "surface": surf_val, "baba_cd": baba_val}
            key = tuple(col_map[c] for c in cols)
        else:
            key = ("all",)

        arr = expanding_stats.get(key)
        if arr is None or len(arr) == 0:
            continue

        dates = arr[:, 0].astype("datetime64[ns]")
        # searchsorted for target_date: find last index where date < target_date
        idx = dates.searchsorted(target_date, side="left")
        if idx > 0:
            return float(arr[idx - 1, 1]), float(arr[idx - 1, 2])

    return float("nan"), float("nan")


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
        "weight_absolute",  # A2: 馬体重
        "weight_zscore",  # A2: 馬個体の体重分布に対する正規化
        "days_since_last_race",  # A3: 前走からの日数
        "rest_category",  # A3: 休養期間カテゴリ (1-5)
        # B3: フォームサイクル
        "form_trend",
        "form_consistency",
        "form_peak_flag",
    ]

    def __init__(self, store: ParquetStore, *, n_past: int = 5) -> None:
        self.store = store
        self.n_past = n_past
        self._n_past = n_past  # 内部参照用
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
        """過去成績特徴量を計算（itertuples + numpy array 高速版）"""
        if target_race_ids is not None:
            entry_df = entry_df[entry_df["race_id"].isin(target_race_ids)]

        # 対象レースの馬・騎手リスト
        horses = entry_df[["race_id", "umaban", "kettonum", "kisyucode"]].copy()
        # datakubun違いで同一(race_id, umaban)が複数行存在する場合は先頭を保持
        horses = horses.drop_duplicates(subset=["race_id", "umaban"], keep="first")
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

        # Merge with races to get syussotosu (field_size), race_date, surface, baba
        race_cols_all = [
            "race_id",
            "syussotosu",
            "race_date",
            "trackcd",
            "kyori",
            "surface",
            "track_condition_code",
        ]
        races_subset = races_hist[races_hist["race_id"].isin(entries_filtered["race_id"].unique())]
        race_cols = [c for c in race_cols_all if c in races_subset.columns]
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

        # Add baba_cd column for z-score hierarchical fallback
        if "baba_cd" not in past_df.columns and "track_condition_code" in past_df.columns:
            past_df["baba_cd"] = past_df["track_condition_code"].fillna(-1).astype(int).astype(str)
            past_df.loc[past_df["baba_cd"] == "-1", "baba_cd"] = ""

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

        # -------------------------------------------------------------------
        # Pre-convert past DataFrames to dict-of-numpy-arrays for fast access
        # -------------------------------------------------------------------
        cols_horse = [
            "race_date",
            "valid_field",
            "kakuteijyuni",
            "syussotosu",
            "harontimel3",
            "distance_bin",
            "surface",
            "baba_cd",
            "timediff",
            "jyuni1c",
            "jyuni4c",
            "kyakusitukubun",
            "bataijyu",
        ]
        cols_jockey = ["race_date", "kakuteijyuni", "odds"]
        cols_jockey_all = ["race_date", "kakuteijyuni"]

        past_by_ketto_arr: dict[str, dict[str, np.ndarray]] = {}
        for k, df in past_by_ketto.items():
            arrs: dict[str, np.ndarray] = {}
            for col in cols_horse:
                if col in df.columns:
                    arrs[col] = df[col].values
            arrs["_valid_mask"] = (arrs.get("valid_field", np.array([], dtype=bool)) == 1) & (
                arrs.get("kakuteijyuni", np.array([], dtype=float)) > 0
            )
            past_by_ketto_arr[k] = arrs

        past_by_kisyu_arr: dict[str, dict[str, np.ndarray]] = {}
        for k, df in past_by_kisyu.items():
            arrs_j: dict[str, np.ndarray] = {}
            for col in cols_jockey:
                if col in df.columns:
                    arrs_j[col] = df[col].values
            past_by_kisyu_arr[k] = arrs_j

        past_by_kisyu_all_arr: dict[str, dict[str, np.ndarray]] = {}
        for k, df in past_by_kisyu_all.items():
            arrs_ja: dict[str, np.ndarray] = {}
            for col in cols_jockey_all:
                if col in df.columns:
                    arrs_ja[col] = df[col].values
            past_by_kisyu_all_arr[k] = arrs_ja

        # -------------------------------------------------------------------
        # Pre-compute weight lookup for vectorized access
        # -------------------------------------------------------------------
        weight_col = "bataijyu" if "bataijyu" in entry_df.columns else "weight"
        if weight_col in entry_df.columns:
            _weight_map = (
                entry_df.drop_duplicates(subset=["race_id", "umaban"])
                .set_index(["race_id", "umaban"])[weight_col]
                .sort_index()
            )
        else:
            _weight_map = pd.Series(dtype=float)

        # -------------------------------------------------------------------
        # Pre-compute column availability flags (same for all horses)
        # -------------------------------------------------------------------
        _sample_arrs = next(iter(past_by_ketto_arr.values()), {})
        _has_harontimel3 = "harontimel3" in _sample_arrs
        _has_distance_bin = "distance_bin" in _sample_arrs
        _has_timediff = "timediff" in _sample_arrs
        _has_jyuni1c = "jyuni1c" in _sample_arrs
        _has_jyuni4c = "jyuni4c" in _sample_arrs
        _has_kyakusitukubun = "kyakusitukubun" in _sample_arrs
        _has_surface = "surface" in _sample_arrs
        _has_baba_cd = "baba_cd" in _sample_arrs

        # -------------------------------------------------------------------
        # Pre-compute EXPANDING stats for harontimel3 z-score (time-series, no leak)
        # Key: group_tuple -> np.array of (race_date, cumulative_mean, cumulative_std)
        # At query time, searchsorted for the date to get stats BEFORE that date.
        # -------------------------------------------------------------------
        expanding_stats: dict[tuple, np.ndarray] = {}
        if _has_harontimel3 and _has_distance_bin:
            valid_past_all = past_df_sorted[
                past_df_sorted["harontimel3"].notna() & past_df_sorted["distance_bin"].notna()
            ].copy()
            if len(valid_past_all) > 0:
                # Fill defaults
                if _has_surface:
                    valid_past_all["_surf"] = valid_past_all["surface"].fillna("")
                else:
                    valid_past_all["_surf"] = ""
                if _has_baba_cd:
                    valid_past_all["_baba"] = valid_past_all["baba_cd"].fillna("")
                else:
                    valid_past_all["_baba"] = ""
                valid_past_all["_db"] = valid_past_all["distance_bin"]
                valid_past_all["_ht"] = valid_past_all["harontimel3"].astype(float)
                valid_past_all["_rd"] = valid_past_all["race_date"]

                # Sort by race_date (should already be, but ensure)
                valid_past_all = valid_past_all.sort_values("_rd").reset_index(drop=True)

                # Compute expanding stats per FALLBACK_LEVELS group
                for cols, min_n in FALLBACK_LEVELS:
                    if cols:
                        col_map = {"distance_bin": "_db", "surface": "_surf", "baba_cd": "_baba"}
                        group_cols = [col_map[c] for c in cols]
                        grouped = valid_past_all.groupby(group_cols)
                        for key, grp_df in grouped:
                            if not isinstance(key, tuple):
                                key = (key,)
                            if len(grp_df) < min_n:
                                continue
                            ht_vals = grp_df["_ht"].values
                            dates_vals = grp_df["_rd"].values
                            # Expanding mean/std: cumulative from start
                            cum_count = np.arange(1, len(ht_vals) + 1)
                            cum_mean = np.cumsum(ht_vals) / cum_count
                            # Expanding std via online algorithm
                            cum_x2 = np.cumsum(ht_vals**2) / cum_count
                            cum_var = cum_x2 - cum_mean**2
                            # Bessel's correction: multiply by n/(n-1)
                            cum_var[1:] = cum_var[1:] * cum_count[1:] / (cum_count[1:] - 1)
                            cum_var[0] = 0.0
                            cum_std = np.sqrt(np.maximum(cum_var, 0.0))
                            # Store: [(date, mean, std), ...] sorted by date
                            arr = np.column_stack([dates_vals.astype(float), cum_mean, cum_std])
                            expanding_stats[key] = arr

                # Global fallback (L4): expanding stats over ALL data
                ht_all_vals = valid_past_all["_ht"].values
                dates_all_vals = valid_past_all["_rd"].values
                if len(ht_all_vals) > 0:
                    cum_count = np.arange(1, len(ht_all_vals) + 1)
                    cum_mean = np.cumsum(ht_all_vals) / cum_count
                    cum_x2 = np.cumsum(ht_all_vals**2) / cum_count
                    cum_var = cum_x2 - cum_mean**2
                    cum_var[1:] = cum_var[1:] * cum_count[1:] / (cum_count[1:] - 1)
                    cum_var[0] = 0.0
                    cum_std = np.sqrt(np.maximum(cum_var, 0.0))
                    arr = np.column_stack([dates_all_vals.astype(float), cum_mean, cum_std])
                    expanding_stats[("all",)] = arr

        total = len(horses)
        results: list[dict] = []

        for i, row in enumerate(horses.itertuples(index=False)):
            if i % 200 == 0:
                print(
                    f"  HorseHistoryFeatures: {i}/{total} ({i / max(total, 1) * 100:.0f}%)",
                    flush=True,
                )
            race_date = row.race_date
            ketto = row.kettonum
            kisyu = row.kisyucode

            # --- Horse features: O(1) lookup + O(log m) searchsorted ---
            horse_arrs = past_by_ketto_arr.get(ketto)
            if horse_arrs is not None and len(horse_arrs.get("race_date", [])) > 0:
                valid_mask = horse_arrs["_valid_mask"]
                if valid_mask.any():
                    dates_all = horse_arrs["race_date"].astype("datetime64[ns]")
                    target_date_np = np.datetime64(race_date, "ns")
                    # searchsorted on valid dates only
                    valid_dates = dates_all[valid_mask]
                    idx = valid_dates.searchsorted(target_date_np, side="left")
                    start = max(0, idx - self._n_past)
                    # Gather last-3 valid past race arrays
                    hp_kakuteijyuni = horse_arrs["kakuteijyuni"][valid_mask][start:idx]
                    hp_syussotosu = horse_arrs["syussotosu"][valid_mask][start:idx]
                    n_past = len(hp_kakuteijyuni)
                else:
                    n_past = 0
            else:
                n_past = 0

            # A3: days_since_last_race + rest_category
            if n_past > 0:
                last_race_date = horse_arrs["race_date"][valid_mask][idx - 1]
                if isinstance(last_race_date, np.datetime64):
                    days_since: float = float(
                        (np.datetime64(race_date, "ns") - last_race_date.astype("datetime64[ns]"))
                        / np.timedelta64(1, "D")
                    )
                else:
                    days_since = float("nan")
                # rest_category (数値エンコード、LightGBM用)
                if days_since <= 7:
                    rest_cat: float = 1.0  # consecutive
                elif days_since <= 30:
                    rest_cat = 2.0  # short
                elif days_since <= 90:
                    rest_cat = 3.0  # medium
                elif days_since <= 180:
                    rest_cat = 4.0  # long
                else:
                    rest_cat = 5.0  # return
            else:
                days_since = float("nan")
                rest_cat = float("nan")

            # norm_finish_logit_avg
            if n_past > 0:
                logits = _norm_finish_logit_vec(
                    hp_kakuteijyuni.astype(float),
                    hp_syussotosu.astype(float),
                )
                norm_finish_logit_avg: float = float(np.nanmean(logits))
            else:
                norm_finish_logit_avg = float("nan")

            # harontimel3_avg
            if _has_harontimel3 and n_past > 0:
                ht_raw = horse_arrs["harontimel3"][valid_mask][start:idx].astype(float)
                ht_valid = ht_raw[~np.isnan(ht_raw)]
                if len(ht_valid) > 0:
                    # tail(3) → last 3 non-NaN values
                    harontimel3_avg: float = float(ht_valid[-self._n_past:].mean())
                else:
                    harontimel3_avg = float("nan")
            else:
                harontimel3_avg = float("nan")

            # harontimel3_zscore — expanding hierarchical fallback z-score (no leak)
            if _has_harontimel3 and _has_distance_bin and n_past > 0 and expanding_stats:
                ht_raw = horse_arrs["harontimel3"][valid_mask][start:idx].astype(float)
                db_raw = horse_arrs["distance_bin"][valid_mask][start:idx]
                surf_raw = horse_arrs.get("surface", np.array([], dtype=object))
                if len(surf_raw) > 0:
                    surf_raw = surf_raw[valid_mask][start:idx]
                else:
                    surf_raw = np.array([""] * len(ht_raw))
                baba_raw = horse_arrs.get("baba_cd", np.array([], dtype=object))
                if len(baba_raw) > 0:
                    baba_raw = baba_raw[valid_mask][start:idx]
                else:
                    baba_raw = np.array([""] * len(ht_raw))

                valid_ht_db = ~np.isnan(ht_raw) & pd.notna(db_raw)
                if valid_ht_db.any():
                    ht_v = ht_raw[valid_ht_db]
                    db_v = db_raw[valid_ht_db]
                    surf_v = surf_raw[valid_ht_db]
                    baba_v = baba_raw[valid_ht_db]
                    # Get dates for each past race
                    dates_v = horse_arrs["race_date"][valid_mask][start:idx][valid_ht_db]
                    dates_v = dates_v.astype("datetime64[ns]")

                    zscores: list[float] = []
                    for j in range(len(ht_v)):
                        target_np = dates_v[j]
                        mean, std = _lookup_expanding_stats(
                            target_np,
                            str(db_v[j]),
                            str(surf_v[j]),
                            str(baba_v[j]),
                            expanding_stats,
                        )
                        if std > 0:
                            zscores.append(float((ht_v[j] - mean) / std))
                        else:
                            zscores.append(float("nan"))
                    if zscores:
                        z_arr = np.array(zscores)
                        # tail(3).mean() — last 3 values
                        harontimel3_zscore = float(
                            pd.Series(z_arr).tail(self._n_past).mean()
                        )
                    else:
                        harontimel3_zscore = float("nan")
                else:
                    harontimel3_zscore = float("nan")
            else:
                harontimel3_zscore = float("nan")

            # timediff_avg
            if _has_timediff and n_past > 0:
                td_raw = horse_arrs["timediff"][valid_mask][start:idx].astype(float)
                td_valid = td_raw[~np.isnan(td_raw)]
                if len(td_valid) > 0:
                    timediff_avg: float = float(td_valid[-self._n_past:].mean())
                else:
                    timediff_avg = float("nan")
            else:
                timediff_avg = float("nan")

            # jyuni1c_avg
            if _has_jyuni1c and n_past > 0:
                c1_raw = horse_arrs["jyuni1c"][valid_mask][start:idx].astype(float)
                c1_valid = c1_raw[~np.isnan(c1_raw)]
                if len(c1_valid) > 0:
                    jyuni1c_avg: float = float(c1_valid[-self._n_past:].mean())
                else:
                    jyuni1c_avg = float("nan")
            else:
                jyuni1c_avg = float("nan")

            # jyuni4c_avg
            if _has_jyuni4c and n_past > 0:
                c4_raw = horse_arrs["jyuni4c"][valid_mask][start:idx].astype(float)
                c4_valid = c4_raw[~np.isnan(c4_raw)]
                if len(c4_valid) > 0:
                    jyuni4c_avg: float = float(c4_valid[-self._n_past:].mean())
                else:
                    jyuni4c_avg = float("nan")
            else:
                jyuni4c_avg = float("nan")

            # closing_index_avg
            if _has_jyuni4c and n_past > 0:
                c4_raw = horse_arrs["jyuni4c"][valid_mask][start:idx].astype(float)
                kj_raw = horse_arrs["kakuteijyuni"][valid_mask][start:idx].astype(float)
                sy_raw = horse_arrs["syussotosu"][valid_mask][start:idx].astype(float)
                valid_ci = ~np.isnan(c4_raw) & ~np.isnan(kj_raw) & ~np.isnan(sy_raw) & (sy_raw > 1)
                if valid_ci.any():
                    norm_4c = (c4_raw[valid_ci] - 1) / (sy_raw[valid_ci] - 1)
                    norm_finish = (kj_raw[valid_ci] - 1) / (sy_raw[valid_ci] - 1)
                    closing_indices = norm_4c - norm_finish
                    closing_index_avg: float = float(closing_indices[-self._n_past:].mean())
                else:
                    closing_index_avg = float("nan")
            else:
                closing_index_avg = float("nan")

            # kyakusitukubun_cd
            if _has_kyakusitukubun and n_past > 0:
                kt_raw = horse_arrs["kyakusitukubun"][valid_mask][start:idx]
                kt_valid = kt_raw[~pd.isna(kt_raw)]
                if len(kt_valid) > 0:
                    kyakusitukubun_cd: float | int = int(kt_valid[-1])
                else:
                    kyakusitukubun_cd = float("nan")
            else:
                kyakusitukubun_cd = float("nan")

            # --- Jockey features: O(1) lookup + O(log m) searchsorted ---
            jockey_arrs = past_by_kisyu_arr.get(kisyu)
            if jockey_arrs is not None and len(jockey_arrs.get("race_date", [])) > 0:
                j_dates = jockey_arrs["race_date"].astype("datetime64[ns]")
                target_date_np = np.datetime64(race_date, "ns")
                idx_j = j_dates.searchsorted(target_date_np, side="left")
                j_start = max(0, idx_j - 100)
                j_kakuteijyuni = jockey_arrs["kakuteijyuni"][j_start:idx_j]
                j_odds = jockey_arrs["odds"][j_start:idx_j]
                n_jockey = len(j_kakuteijyuni)
            else:
                n_jockey = 0

            if n_jockey >= 30:
                expected = float((PAYOUT_RATE / np.clip(j_odds, 1.1, None)).sum())
                actual = int((j_kakuteijyuni == 1).sum())
                jockey_surprise: float = _compute_jockey_surprise(actual, n_jockey, expected)
            else:
                jockey_surprise = float("nan")

            # jockey_cond_wr — uses past_by_kisyu_all (kakuteijyuni > 0 only, no odds filter)
            jockey_all_arrs = past_by_kisyu_all_arr.get(kisyu)
            if jockey_all_arrs is not None and len(jockey_all_arrs.get("race_date", [])) > 0:
                ja_dates = jockey_all_arrs["race_date"].astype("datetime64[ns]")
                target_date_np = np.datetime64(race_date, "ns")
                idx_ja = ja_dates.searchsorted(target_date_np, side="left")
                ja_kakuteijyuni = jockey_all_arrs["kakuteijyuni"][:idx_ja]
                total_rides = len(ja_kakuteijyuni)
            else:
                total_rides = 0

            if total_rides > 0:
                total_wins = int((ja_kakuteijyuni == 1).sum())
            else:
                total_wins = 0

            k_smooth = 25
            if total_rides >= 10:
                cond_wr = total_wins / max(total_rides, 1)
                global_wr = total_wins / max(total_rides, 1)
                w = min(total_rides / (total_rides + k_smooth), 1.0)
                jockey_cond_wr: float = float(w * cond_wr + (1 - w) * global_wr)
            else:
                jockey_cond_wr = float("nan")

            # weight_absolute — vectorized lookup
            wkey = (row.race_id, row.umaban)
            if wkey in _weight_map.index:
                wval = _weight_map.loc[wkey]
                # .loc may return Series if duplicate keys exist — take first
                if isinstance(wval, pd.Series):
                    wval = wval.iloc[0]
                weight_absolute: float = float(wval) if pd.notna(wval) else float("nan")
            else:
                weight_absolute = float("nan")

            # A2: weight_zscore — 馬個体の体重分布に対する正規化
            if n_past > 0 and "bataijyu" in horse_arrs:
                past_weights = horse_arrs["bataijyu"][valid_mask][:idx].astype(float)
                past_valid_w = past_weights[~np.isnan(past_weights)]
                if len(past_valid_w) >= 2 and pd.notna(weight_absolute):
                    w_mean = float(past_valid_w.mean())
                    w_std = float(past_valid_w.std())
                    if w_std > 0:
                        weight_zscore: float = float((weight_absolute - w_mean) / w_std)
                    else:
                        weight_zscore = 0.0
                else:
                    weight_zscore = float("nan")
            else:
                weight_zscore = float("nan")

            # B3: フォームサイクル特徴量
            if n_past >= 2:
                _fc_kj = horse_arrs["kakuteijyuni"][valid_mask][start:idx].astype(float)
                _fc_ss = horse_arrs["syussotosu"][valid_mask][start:idx].astype(float)
                form_trend, form_consistency, form_peak_flag = compute_form_features(_fc_kj, _fc_ss)
            else:
                form_trend = float("nan")
                form_consistency = float("nan")
                form_peak_flag = float("nan")

            results.append(
                {
                    "race_id": row.race_id,
                    "umaban": row.umaban,
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
                    "weight_zscore": weight_zscore,
                    "days_since_last_race": days_since,
                    "rest_category": rest_cat,
                    "form_trend": form_trend,
                    "form_consistency": form_consistency,
                    "form_peak_flag": form_peak_flag,
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
