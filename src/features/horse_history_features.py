"""horse_history_features.py — 馬の過去成績ベース特徴量

主な特徴量:
  - norm_finish_logit_avg: 着順をログット変換したスコアの平均
  - harontimel5_avg: 直近5走のハロンタイム平均
  - harontimel5_zscore: 距離ビンz-scoreの直近5走平均
  - harontime_late_trend: 最後2走平均 - 最初3走平均 (負=改善傾向)
  - timediff_avg: 直近5走のタイム差平均
  - jyuni1c_avg: 直近5走の1コーナー通過位置平均
  - jyuni4c_avg: 直近5走の4コーナー通過位置平均
  - closing_index_avg: (4C正規化 - 着順正規化) の直近5走平均
  - kyakusitukubun_cd: 直近走の脚質コード
  - jockey_surprise: Beta事前分布でスムージングした騎手勝率サプライズ
  - jockey_cond_wr: 騎手条件別勝率
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from features.form_cycle_features import compute_form_features
from features.high_odds_features import (
    compute_class_trajectory,
    compute_env_adaptability,
    compute_form_improvement_rate,
)

logger = logging.getLogger(__name__)

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

_CLASS_LEVEL_MAP: dict[str, float] = {
    "A": 8.0,
    "B": 7.0,
    "C": 6.0,
    "D": 5.0,
    "E": 4.0,
}

# HLF-01: Distance threshold for harontime_last3f unified column (D-01)
DISTANCE_THRESHOLD: int = 2000  # >= 2000m: prefer L4, < 2000m: prefer L3


def _coerce_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _class_level_from_values(grade_code: object, jyoken_code: object) -> float:
    grade = str(grade_code).strip() if pd.notna(grade_code) else ""
    if grade in _CLASS_LEVEL_MAP:
        return _CLASS_LEVEL_MAP[grade]
    return _coerce_float(jyoken_code)


def _blinker_flag(value: object) -> float:
    flag = _coerce_float(value)
    if np.isnan(flag):
        return float("nan")
    return 1.0 if flag > 0 else 0.0


def _compute_distance_bin(kyori: object, surface: object) -> str:
    """kyori と surface から distance_bin を計算する"""
    try:
        dist = float(kyori)
    except (ValueError, TypeError):
        return "unknown"
    is_turf = str(surface).strip().lower() == "turf"
    if is_turf:
        if dist > 2100:
            return "long"
        elif dist <= 1400:
            return "sprint"
        elif dist <= 1700:
            return "mile"
        else:
            return "intermediate"
    else:
        if dist > 1700:
            return "intermediate"
        elif dist <= 1400:
            return "sprint"
        else:
            return "mile"

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
    expected_wins: float,
) -> float:
    """Beta事前分布でスムージングした騎手勝率のサプライズ値を返す。

    actual_wins をBeta事後分布で平滑化し、expected_wins（オッズ推定期待勝利数）
    を同様に平滑化したベースラインとの差分を返す。
    n_races < 30 の場合は NaN を返す。
    """
    if n_races < 30:
        return float("nan")

    alpha_post = ALPHA_PRIOR + actual_wins
    beta_post = BETA_PRIOR + n_races - actual_wins

    smoothed_wr = alpha_post / (alpha_post + beta_post)

    # オッズ推定期待勝利数を同じBeta事前分布で平滑化
    smoothed_expected_wr = (ALPHA_PRIOR + expected_wins) / (
        ALPHA_PRIOR + BETA_PRIOR + n_races
    )

    return smoothed_wr - smoothed_expected_wr


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
# Helper: haron stats (L5 + late trend)
# ---------------------------------------------------------------------------


def _compute_haron_stats(
    history: pd.DataFrame,
    target_date: pd.Timestamp,
) -> dict[str, float]:
    """直近5走のハロンタイム統計量を計算する (PIT: 当日以前のみ使用)。

    Args:
        history: race_date, harontimel3 列を持つ DataFrame
        target_date: 対象レース日付

    Returns:
        dict with keys:
          - harontimel5_avg: 直近5走のハロンタイム平均 (NaNスキップ、nanmean)
          - harontimel5_zscore: 5走に拡張したz-score (expanding_stats 使用)
          - harontime_late_trend: 最後2走平均 - 最初3走平均 (負=改善傾向)
    """
    # PIT CRITICAL: 必ず当日以前のみ使用
    past = history[history["race_date"] < target_date].copy()
    if past.empty or "harontimel3" not in past.columns:
        return {
            "harontimel5_avg": float("nan"),
            "harontimel5_zscore": float("nan"),
            "harontime_late_trend": float("nan"),
        }

    ht = past["harontimel3"].astype(float).values
    ht_valid = ht[~np.isnan(ht)]

    if len(ht_valid) == 0:
        return {
            "harontimel5_avg": float("nan"),
            "harontimel5_zscore": float("nan"),
            "harontime_late_trend": float("nan"),
        }

    # 直近5走の平均 (tail 5 of valid values)
    l5_avg = float(ht_valid[-5:].mean())

    # late_trend: 最後2走 vs 最初3走 (5走以上必要)
    if len(ht_valid) >= 5:
        last_2 = ht_valid[-2:].mean()
        first_3 = ht_valid[:3].mean()
        late_trend = float(last_2 - first_3)  # 負=改善
    else:
        late_trend = float("nan")

    # z-score は compute() 内で expanding_stats を使って別途計算されるため、
    # ここではプレースホルダーとして NaN を返す
    return {
        "harontimel5_avg": l5_avg,
        "harontimel5_zscore": float("nan"),  # compute()内で計算
        "harontime_late_trend": late_trend,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class HorseHistoryFeatures:
    """馬の過去成績特徴量を計算・管理するクラス。"""

    # P3: クラスレベルキャッシュ — 3回呼び出し間(芝submodel, ダートsubmodel, test BT)で
    # 前計算結果を共有 (~100s削減)
    _class_cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}

    @classmethod
    def clear_class_cache(cls) -> None:
        """テスト用/終了時: キャッシュをクリア"""
        cls._class_cache.clear()

    BASE_COLS: list[str] = [
        "norm_finish_logit_avg",
        "harontimel5_avg",  # EMA重み付けハロンタイム平均 (halflife=3)
        "harontimel5_zscore",  # 直近5走z-score
        "harontime_late_trend",  # 最後2走 vs 最初3走 (負=改善)
        "timediff_avg",
        "jyuni1c_avg",
        "jyuni4c_avg",
        "closing_index_avg",
        "kyakusitukubun_cd",  # non-numeric, category
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
        # 追加改善特徴量
        "class_move",
        "blinker_change",
        "is_nar_transfer",
        "nar_recent_ratio",
        "track_condition_delta",
        # FEAT-02: 単勝特化新特徴量
        "distance_change",
        "surface_change",
        "class_drop_bounce",
        "win_dominance",
        "freshness_score",
        # TSER-02: クラス調整フォーメトリック
        "class_adj_formetric",
        # TSER-03: z-score改善トラジェクトリ
        "haron_zscore_trend",
        # HODDS-02: クラストラジェクトリ (D-05, D-06, D-07)
        "class_promotions",
        "class_demotions",
        "class_net_change",
        "class_max_level",
        "class_level_std",
        "v_recovery_flag",
        "v_recovery_duration",
        # HODDS-03: フォーム改善率 (D-08, D-09)
        "time_improvement_rate",
        "position_improvement_rate",
        # HODDS-04: 環境変化適性 (D-10, D-11)
        "dist_change_avg_pos",
        "dist_change_win_rate",
        "dist_change_exp_count",
        "surf_change_avg_pos",
        "surf_change_win_rate",
        "surf_change_exp_count",
        "cond_change_avg_pos",
        "cond_change_win_rate",
        "cond_change_exp_count",
        # TRF-02: weighted_recent_form (EMA halflife=3, D-07/D-08)
        "weighted_recent_form_finish",
        "weighted_recent_form_time",
        # HLF-01: HaronTime L4 history stats (avg/zscore/trend)
        "harontimel4_avg",
        "harontimel4_zscore",
        "harontimel4_trend",
        # HLF-01: harontime_last3f unified column (distance-based auto-selection)
        "harontime_last3f_avg",
        "harontime_last3f_zscore",
        "harontime_last3f_trend",
        # HLF-03: LapTime pace features
        "pace_ratio_avg",
        "pace_ratio_zscore",
        "pace_ratio_trend",
        "pace_early_avg",
        "pace_mid_avg",
        "pace_late_avg",
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

    def _get_cached_horse_arrays(
        self, ketto: str, past_by_ketto_arr: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray] | None:
        """インスタンス間でキャッシュを共有 (P3: ~100s削減)

        3回呼び出し(芝submodel, ダートsubmodel, test BT)間で
        前計算結果を共有する。
        """
        cache_key = f"{id(past_by_ketto_arr)}_{ketto}"
        if cache_key in self._class_cache:
            _, cached = self._class_cache[cache_key]
            return cached
        result = past_by_ketto_arr.get(ketto)
        if result is not None:
            self._class_cache[cache_key] = (np.array([]), result)
        return result

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
        base_cols = ["race_id", "umaban", "kettonum", "kisyucode"]
        optional_entry_cols = [c for c in ["blinker", "jyocd"] if c in entry_df.columns]
        horses = entry_df[base_cols + optional_entry_cols].copy()
        # datakubun違いで同一(race_id, umaban)が複数行存在する場合は先頭を保持
        horses = horses.drop_duplicates(subset=["race_id", "umaban"], keep="first")
        race_context_cols = [
            c
            for c in [
                "race_id",
                "race_date",
                "surface",
                "track_condition_code",
                "gradecd",
                "jyokencd1",
                "distance_bin",
                "kyori",
            ]
            if c in race_df.columns
        ]
        if race_context_cols:
            race_context = race_df[race_context_cols].drop_duplicates(subset=["race_id"])
            horses = horses.merge(race_context, on="race_id", how="left")

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
            "gradecd",
            "jyokencd1",
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
            k: g.reset_index(drop=True)
            for k, g in past_df_sorted.groupby("kettonum", observed=True)
        }

        # Pre-index past data by kisyucode for jockey_surprise (kakuteijyuni > 0 AND odds > 0)
        past_by_kisyu: dict[str, pd.DataFrame] = {
            k: g.reset_index(drop=True)
            for k, g in past_df_sorted[
                (past_df_sorted["kakuteijyuni"] > 0) & (past_df_sorted["odds"] > 0)
            ].groupby("kisyucode", observed=True)
        }

        # Pre-index past data by kisyucode for jockey_cond_wr
        # (kakuteijyuni > 0 only, no odds filter)
        past_by_kisyu_all: dict[str, pd.DataFrame] = {
            k: g.reset_index(drop=True)
            for k, g in past_df_sorted[
                (past_df_sorted["kakuteijyuni"] > 0)
            ].groupby("kisyucode", observed=True)
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
            "harontimel4",  # HLF-01: L4 haron time (Phase 35 ETL)
            "distance_bin",
            "surface",
            "baba_cd",
            "timediff",
            "jyuni1c",
            "jyuni4c",
            "kyakusitukubun",
            "bataijyu",
            "track_condition_code",
            "gradecd",
            "jyokencd1",
            "blinker",
            "jyocd",
        ]
        cols_jockey = ["race_date", "kakuteijyuni", "odds"]
        cols_jockey_all = ["race_date", "kakuteijyuni", "surface"]

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
        _has_harontimel4 = "harontimel4" in _sample_arrs
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
                        grouped = valid_past_all.groupby(group_cols, observed=True)
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

        # -------------------------------------------------------------------
        # Pre-compute EXPANDING stats for harontimel4 z-score (HLF-01)
        # Same FALLBACK_LEVELS structure, separate dict to avoid key collision
        # -------------------------------------------------------------------
        expanding_stats_hl4: dict[tuple, np.ndarray] = {}
        if _has_harontimel4 and _has_distance_bin:
            valid_past_hl4 = past_df_sorted[
                past_df_sorted["harontimel4"].notna() & past_df_sorted["distance_bin"].notna()
            ].copy()
            if len(valid_past_hl4) > 0:
                if _has_surface:
                    valid_past_hl4["_surf"] = valid_past_hl4["surface"].fillna("")
                else:
                    valid_past_hl4["_surf"] = ""
                if _has_baba_cd:
                    valid_past_hl4["_baba"] = valid_past_hl4["baba_cd"].fillna("")
                else:
                    valid_past_hl4["_baba"] = ""
                valid_past_hl4["_db"] = valid_past_hl4["distance_bin"]
                valid_past_hl4["_ht4"] = valid_past_hl4["harontimel4"].astype(float)
                valid_past_hl4["_rd"] = valid_past_hl4["race_date"]

                valid_past_hl4 = valid_past_hl4.sort_values("_rd").reset_index(drop=True)

                for cols, min_n in FALLBACK_LEVELS:
                    if cols:
                        col_map = {"distance_bin": "_db", "surface": "_surf", "baba_cd": "_baba"}
                        group_cols = [col_map[c] for c in cols]
                        grouped = valid_past_hl4.groupby(group_cols, observed=True)
                        for key, grp_df in grouped:
                            if not isinstance(key, tuple):
                                key = (key,)
                            if len(grp_df) < min_n:
                                continue
                            ht4_vals = grp_df["_ht4"].values
                            dates_vals = grp_df["_rd"].values
                            cum_count = np.arange(1, len(ht4_vals) + 1)
                            cum_mean = np.cumsum(ht4_vals) / cum_count
                            cum_x2 = np.cumsum(ht4_vals**2) / cum_count
                            cum_var = cum_x2 - cum_mean**2
                            cum_var[1:] = cum_var[1:] * cum_count[1:] / (cum_count[1:] - 1)
                            cum_var[0] = 0.0
                            cum_std = np.sqrt(np.maximum(cum_var, 0.0))
                            arr = np.column_stack([dates_vals.astype(float), cum_mean, cum_std])
                            expanding_stats_hl4[key] = arr

                # Global fallback (L4 level)
                ht4_all_vals = valid_past_hl4["_ht4"].values
                dates_all_vals = valid_past_hl4["_rd"].values
                if len(ht4_all_vals) > 0:
                    cum_count = np.arange(1, len(ht4_all_vals) + 1)
                    cum_mean = np.cumsum(ht4_all_vals) / cum_count
                    cum_x2 = np.cumsum(ht4_all_vals**2) / cum_count
                    cum_var = cum_x2 - cum_mean**2
                    cum_var[1:] = cum_var[1:] * cum_count[1:] / (cum_count[1:] - 1)
                    cum_var[0] = 0.0
                    cum_std = np.sqrt(np.maximum(cum_var, 0.0))
                    arr = np.column_stack([dates_all_vals.astype(float), cum_mean, cum_std])
                    expanding_stats_hl4[("all",)] = arr

        # -------------------------------------------------------------------
        # HLF-03: Pre-compute LapTime pace_ratio per past race (PIT-safe)
        # LapTime1~25 are POST_RACE_COLS — only past-race data is used.
        # pace_ratio = late_avg / early_avg (D-05)
        # n_laps = kyori / 200 (D-04), segments via np.array_split
        #
        # Note: LapTime columns live in races_hist (not merged into past_df).
        # We build a per-race lookup from races_hist, then match to horses
        # via entries_hist during per-horse computation.
        # -------------------------------------------------------------------
        _has_laptime = any(c in races_hist.columns for c in ["laptime1", "laptime2"])
        _pace_lookup: dict[str, dict[str, np.ndarray]] = {}

        if _has_laptime and "kyori" in races_hist.columns:
            lap_cols = [f"laptime{i}" for i in range(1, 26)]
            available_lap_cols = [c for c in lap_cols if c in races_hist.columns]

            # Build per-race pace_ratio from races_hist
            _race_pace_lookup: dict[str, dict[str, float]] = {}
            for _, race_row in races_hist.iterrows():
                kyori_val = race_row.get("kyori")
                if pd.isna(kyori_val) or float(kyori_val) < 600:
                    continue
                n_laps = int(float(kyori_val) / 200)
                if n_laps < 3:
                    continue

                laptimes = np.array([
                    float(race_row[c]) if c in available_lap_cols and pd.notna(race_row.get(c))
                    else float("nan")
                    for c in [f"laptime{i}" for i in range(1, n_laps + 1)]
                ])
                if np.any(np.isnan(laptimes)):
                    continue

                segments = np.array_split(laptimes, 3)
                e_avg = float(np.nanmean(segments[0])) if len(segments[0]) > 0 else float("nan")
                m_avg = float(np.nanmean(segments[1])) if len(segments[1]) > 0 else float("nan")
                l_avg = float(np.nanmean(segments[2])) if len(segments[2]) > 0 else float("nan")

                if not np.isnan(e_avg) and e_avg > 0 and not np.isnan(l_avg):
                    _race_pace_lookup[str(race_row["race_id"])] = {
                        "pace_ratio": l_avg / e_avg,
                        "early_avg": e_avg,
                        "mid_avg": m_avg,
                        "late_avg": l_avg,
                    }

            # Build per-horse pace history from entries_hist + race_pace_lookup
            if _race_pace_lookup:
                # Group entries_hist by kettonum to find each horse's past races
                entries_hist_sorted = (
                    entries_filtered.sort_values(["kettonum", "race_date"]).reset_index(drop=True)
                    if "race_date" in entries_filtered.columns
                    else entries_filtered
                )

                for ketto_key, ketto_grp in entries_hist_sorted.groupby("kettonum", observed=True):
                    pace_ratios: list[float] = []
                    early_avgs: list[float] = []
                    mid_avgs: list[float] = []
                    late_avgs: list[float] = []
                    pace_dates: list[np.datetime64] = []

                    for _, ent_row in ketto_grp.iterrows():
                        rid = str(ent_row["race_id"])
                        pace_info = _race_pace_lookup.get(rid)
                        if pace_info is None:
                            continue
                        pace_ratios.append(pace_info["pace_ratio"])
                        early_avgs.append(pace_info["early_avg"])
                        mid_avgs.append(pace_info["mid_avg"])
                        late_avgs.append(pace_info["late_avg"])
                        rd = ent_row.get("race_date")
                        if pd.notna(rd):
                            pace_dates.append(np.datetime64(rd, "ns"))

                    if pace_ratios and len(pace_dates) == len(pace_ratios):
                        sort_idx = np.argsort(pace_dates)
                        _pace_lookup[str(ketto_key)] = {
                            "pace_ratios": np.array(pace_ratios)[sort_idx],
                            "early_avgs": np.array(early_avgs)[sort_idx],
                            "mid_avgs": np.array(mid_avgs)[sort_idx],
                            "late_avgs": np.array(late_avgs)[sort_idx],
                            "race_dates": np.array(pace_dates)[sort_idx],
                        }

        total = len(horses)
        results: list[dict] = []

        for i, row in enumerate(horses.itertuples(index=False)):
            if i % 200 == 0:
                logger.debug(
                    "HorseHistoryFeatures: %d/%d (%.0f%%)", i, total,
                    i / max(total, 1) * 100,
                )
            race_date = row.race_date
            ketto = row.kettonum
            kisyu = row.kisyucode

            # --- Horse features: O(1) lookup + O(log m) searchsorted ---
            horse_arrs = self._get_cached_horse_arrays(ketto, past_by_ketto_arr)
            target_date_np = np.datetime64(race_date, "ns")
            if horse_arrs is not None and len(horse_arrs.get("race_date", [])) > 0:
                dates_all = horse_arrs["race_date"].astype("datetime64[ns]")
                valid_mask = horse_arrs["_valid_mask"]
                if valid_mask.any():
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

            history_mask = (
                (
                    (horse_arrs["kakuteijyuni"] > 0)
                    & (
                        horse_arrs.get("valid_field", np.ones(len(horse_arrs["kakuteijyuni"]), dtype=int))
                        == 1
                    )
                )
                if horse_arrs is not None and "kakuteijyuni" in horse_arrs
                else np.array([], dtype=bool)
            )
            has_history = horse_arrs is not None and len(horse_arrs.get("race_date", [])) > 0
            if has_history and history_mask.any():
                history_dates = dates_all[history_mask]
                hist_idx = history_dates.searchsorted(target_date_np, side="left")
                hist_start = max(0, hist_idx - self._n_past)
            else:
                hist_idx = 0
                hist_start = 0

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

            # harontimel5_avg (EMA重み付けハロンタイム平均, halflife=3)
            if _has_harontimel3 and n_past > 0:
                ht_raw = horse_arrs["harontimel3"][valid_mask][start:idx].astype(float)
                ht_valid = ht_raw[~np.isnan(ht_raw)]
                if len(ht_valid) > 0:
                    # TSER-01: EMA重み付け (D-02: halflife=3, D-03: 全過去走使用)
                    halflife = 3
                    decay = np.log(2) / halflife  # ≈ 0.231
                    n_ht = len(ht_valid)
                    # Geometric decay: w[i] = (1 - ln(2)/halflife)^i, i=0 oldest
                    weights = (1 - decay) ** np.arange(n_ht)
                    # Reverse so index 0 = newest (highest weight)
                    weights = weights[::-1]
                    weights = weights / weights.sum()
                    harontimel5_avg: float = float(np.sum(ht_valid * weights))
                else:
                    harontimel5_avg = float("nan")
            else:
                harontimel5_avg = float("nan")

            # TSER-02: class_adj_formetric — クラス調整フォーメトリック (D-04, D-05, D-06)
            if n_past > 0:
                _ca_kj = hp_kakuteijyuni.astype(float)
                _ca_ss = hp_syussotosu.astype(float)
                _ca_valid_ss = _ca_ss > 1
                if _ca_valid_ss.any():
                    _ca_norm_finish = (_ca_kj[_ca_valid_ss] - 1) / (_ca_ss[_ca_valid_ss] - 1)
                    _ca_grade = horse_arrs.get("gradecd", np.array([], dtype=object))
                    _ca_jyoken = horse_arrs.get("jyokencd1", np.array([], dtype=object))
                    if len(_ca_grade) > 0 and len(_ca_jyoken) > 0:
                        _ca_grade_v = _ca_grade[valid_mask][start:idx][_ca_valid_ss]
                        _ca_jyoken_v = _ca_jyoken[valid_mask][start:idx][_ca_valid_ss]
                        _ca_levels = np.array([
                            _class_level_from_values(g, j)
                            for g, j in zip(_ca_grade_v, _ca_jyoken_v)
                        ])
                        _ca_valid = ~np.isnan(_ca_levels) & ~np.isnan(_ca_norm_finish)
                        if _ca_valid.any():
                            cl = _ca_levels[_ca_valid]
                            nf = _ca_norm_finish[_ca_valid]
                            class_adj_formetric: float = float(np.sum(nf * cl) / np.sum(cl))
                        else:
                            class_adj_formetric = float("nan")
                    else:
                        class_adj_formetric = float("nan")
                else:
                    class_adj_formetric = float("nan")
            else:
                class_adj_formetric = float("nan")

            # harontimel5_zscore — expanding hierarchical fallback z-score (no leak)
            # Also computes haron_zscore_trend (TSER-03)
            haron_zscore_trend: float = float("nan")
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
                        # tail(5).mean() — last 5 values
                        harontimel5_zscore: float = float(
                            pd.Series(z_arr).tail(self._n_past).mean()
                        )
                        # TSER-03: z-score改善トラジェクトリ (D-07, D-09)
                        valid_z = z_arr[~np.isnan(z_arr)]
                        if len(valid_z) >= 3:
                            x = np.arange(len(valid_z), dtype=float)
                            haron_zscore_trend = float(np.polyfit(x, valid_z, 1)[0])
                    else:
                        harontimel5_zscore = float("nan")
                else:
                    harontimel5_zscore = float("nan")
            else:
                harontimel5_zscore = float("nan")

            # -----------------------------------------------------------------
            # HLF-01: HaronTime L4 history stats (avg/zscore/trend)
            # -----------------------------------------------------------------
            harontimel4_avg: float = float("nan")
            harontimel4_zscore: float = float("nan")
            harontimel4_trend: float = float("nan")

            if _has_harontimel4 and n_past > 0:
                ht4_raw = horse_arrs["harontimel4"][valid_mask][start:idx].astype(float)
                ht4_valid = ht4_raw[~np.isnan(ht4_raw)]

                if len(ht4_valid) > 0:
                    # L4 avg: EMA(halflife=3) — same pattern as harontimel5_avg
                    halflife_ht4 = 3
                    decay_ht4 = np.log(2) / halflife_ht4
                    n_ht4 = len(ht4_valid)
                    weights_ht4 = (1 - decay_ht4) ** np.arange(n_ht4)
                    weights_ht4 = weights_ht4[::-1]
                    weights_ht4 = weights_ht4 / weights_ht4.sum()
                    harontimel4_avg = float(np.sum(ht4_valid * weights_ht4))

                    # L4 trend: linear regression slope of last 3 valid values
                    ht4_for_trend = ht4_valid[-3:] if len(ht4_valid) >= 3 else ht4_valid
                    if len(ht4_for_trend) >= 2:
                        x_trend = np.arange(len(ht4_for_trend), dtype=float)
                        harontimel4_trend = float(np.polyfit(x_trend, ht4_for_trend, 1)[0])

            # L4 zscore: hierarchical expanding_stats (separate dict for L4)
            if _has_harontimel4 and _has_distance_bin and n_past > 0 and expanding_stats_hl4:
                ht4_raw_z = horse_arrs["harontimel4"][valid_mask][start:idx].astype(float)
                db_raw = horse_arrs["distance_bin"][valid_mask][start:idx]
                surf_raw = horse_arrs.get("surface", np.array([], dtype=object))
                if len(surf_raw) > 0:
                    surf_raw = surf_raw[valid_mask][start:idx]
                else:
                    surf_raw = np.array([""] * len(ht4_raw_z))
                baba_raw = horse_arrs.get("baba_cd", np.array([], dtype=object))
                if len(baba_raw) > 0:
                    baba_raw = baba_raw[valid_mask][start:idx]
                else:
                    baba_raw = np.array([""] * len(ht4_raw_z))

                valid_ht4_db = ~np.isnan(ht4_raw_z) & pd.notna(db_raw)
                if valid_ht4_db.any():
                    ht4_v = ht4_raw_z[valid_ht4_db]
                    db_v = db_raw[valid_ht4_db]
                    surf_v = surf_raw[valid_ht4_db]
                    baba_v = baba_raw[valid_ht4_db]
                    dates_v = horse_arrs["race_date"][valid_mask][start:idx][valid_ht4_db]
                    dates_v = dates_v.astype("datetime64[ns]")

                    zscores_hl4: list[float] = []
                    for j in range(len(ht4_v)):
                        target_np = dates_v[j]
                        mean4, std4 = _lookup_expanding_stats(
                            target_np,
                            str(db_v[j]),
                            str(surf_v[j]),
                            str(baba_v[j]),
                            expanding_stats_hl4,
                        )
                        if std4 > 0:
                            zscores_hl4.append(float((ht4_v[j] - mean4) / std4))
                        else:
                            zscores_hl4.append(float("nan"))
                    if zscores_hl4:
                        z_arr_hl4 = np.array(zscores_hl4)
                        harontimel4_zscore = float(
                            pd.Series(z_arr_hl4).tail(self._n_past).mean()
                        )

            # -----------------------------------------------------------------
            # HLF-01: harontime_last3f unified column (D-01, D-02)
            # Distance-based auto-selection: >=2000m prefer L4, <2000m prefer L3
            # -----------------------------------------------------------------
            harontime_last3f_avg: float = float("nan")
            harontime_last3f_zscore: float = float("nan")
            harontime_last3f_trend: float = float("nan")
            unified_raw: np.ndarray | None = None

            current_kyori = float(getattr(row, "kyori", 0))

            if n_past > 0:
                # Build unified raw array based on distance preference
                l3_raw = (
                    horse_arrs["harontimel3"][valid_mask][start:idx].astype(float)
                    if _has_harontimel3
                    else np.full(idx - start, np.nan)
                )
                l4_raw = (
                    horse_arrs["harontimel4"][valid_mask][start:idx].astype(float)
                    if _has_harontimel4
                    else np.full(idx - start, np.nan)
                )

                # Per-race: select preferred, fallback to other
                if current_kyori >= DISTANCE_THRESHOLD:
                    # Long distance: prefer L4, fallback L3
                    primary, fallback = l4_raw, l3_raw
                else:
                    # Short/middle: prefer L3, fallback L4
                    primary, fallback = l3_raw, l4_raw

                # Build unified array: use primary where valid, else fallback
                unified_raw = np.where(~np.isnan(primary), primary, fallback)
                unified_valid = unified_raw[~np.isnan(unified_raw)]

                if len(unified_valid) > 0:
                    # Unified avg: EMA(halflife=3)
                    halflife_uni = 3
                    decay_uni = np.log(2) / halflife_uni
                    n_uni = len(unified_valid)
                    weights_uni = (1 - decay_uni) ** np.arange(n_uni)
                    weights_uni = weights_uni[::-1]
                    weights_uni = weights_uni / weights_uni.sum()
                    harontime_last3f_avg = float(np.sum(unified_valid * weights_uni))

                    # Unified trend: linear regression slope of last 3
                    uni_for_trend = unified_valid[-3:] if len(unified_valid) >= 3 else unified_valid
                    if len(uni_for_trend) >= 2:
                        x_uni = np.arange(len(uni_for_trend), dtype=float)
                        harontime_last3f_trend = float(np.polyfit(x_uni, uni_for_trend, 1)[0])

            # Unified zscore: use L3 expanding_stats as proxy (D-02: L3 has more coverage)
            if n_past > 0 and _has_distance_bin and expanding_stats:
                # Re-use unified_raw from above if available
                if unified_raw is None or np.all(np.isnan(unified_raw)):
                    pass  # no unified data
                else:
                    db_raw_u = horse_arrs["distance_bin"][valid_mask][start:idx]
                    surf_raw_u = horse_arrs.get("surface", np.array([], dtype=object))
                    if len(surf_raw_u) > 0:
                        surf_raw_u = surf_raw_u[valid_mask][start:idx]
                    else:
                        surf_raw_u = np.array([""] * len(unified_raw))
                    baba_raw_u = horse_arrs.get("baba_cd", np.array([], dtype=object))
                    if len(baba_raw_u) > 0:
                        baba_raw_u = baba_raw_u[valid_mask][start:idx]
                    else:
                        baba_raw_u = np.array([""] * len(unified_raw))

                    valid_uni_db = ~np.isnan(unified_raw) & pd.notna(db_raw_u)
                    if valid_uni_db.any():
                        uni_v = unified_raw[valid_uni_db]
                        db_u = db_raw_u[valid_uni_db]
                        surf_u = surf_raw_u[valid_uni_db]
                        baba_u = baba_raw_u[valid_uni_db]
                        dates_u = horse_arrs["race_date"][valid_mask][start:idx][valid_uni_db]
                        dates_u = dates_u.astype("datetime64[ns]")

                        zscores_uni: list[float] = []
                        for j in range(len(uni_v)):
                            target_np_u = dates_u[j]
                            mean_u, std_u = _lookup_expanding_stats(
                                target_np_u,
                                str(db_u[j]),
                                str(surf_u[j]),
                                str(baba_u[j]),
                                expanding_stats,  # L3 stats as proxy
                            )
                            if std_u > 0:
                                zscores_uni.append(float((uni_v[j] - mean_u) / std_u))
                            else:
                                zscores_uni.append(float("nan"))
                        if zscores_uni:
                            z_arr_uni = np.array(zscores_uni)
                            harontime_last3f_zscore = float(
                                pd.Series(z_arr_uni).tail(self._n_past).mean()
                            )

            # harontime_late_trend: 最後2走 vs 最初3走 (負=改善傾向)
            if _has_harontimel3 and n_past >= 5:
                ht_for_trend = horse_arrs["harontimel3"][valid_mask][start:idx].astype(float)
                ht_valid_trend = ht_for_trend[~np.isnan(ht_for_trend)]
                if len(ht_valid_trend) >= 5:
                    last_2 = ht_valid_trend[-2:].mean()
                    first_3 = ht_valid_trend[:3].mean()
                    harontime_late_trend: float = float(last_2 - first_3)  # 負=改善
                else:
                    harontime_late_trend = float("nan")
            else:
                harontime_late_trend = float("nan")

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

            # jockey_cond_wr — サーフェス条件別勝率 (hierarchical shrinkage)
            # cond_wr: 同サーフェスでの勝率, global_wr: 全体勝率
            jockey_all_arrs = past_by_kisyu_all_arr.get(kisyu)
            if jockey_all_arrs is not None and len(jockey_all_arrs.get("race_date", [])) > 0:
                ja_dates = jockey_all_arrs["race_date"].astype("datetime64[ns]")
                target_date_np = np.datetime64(race_date, "ns")
                idx_ja = ja_dates.searchsorted(target_date_np, side="left")
                ja_kakuteijyuni = jockey_all_arrs["kakuteijyuni"][:idx_ja]
                total_rides = len(ja_kakuteijyuni)
                # 同サーフェスの騎乗のみで条件別勝率を計算
                ja_surfaces = jockey_all_arrs.get("surface", np.array([], dtype=object))
                if len(ja_surfaces) > 0:
                    ja_surfaces = ja_surfaces[:idx_ja]
                current_surface = str(row.surface) if hasattr(row, "surface") else ""
                cond_mask = (
                    (ja_surfaces == current_surface)
                    if len(ja_surfaces) > 0
                    else np.array([], dtype=bool)
                )
            else:
                total_rides = 0
                cond_mask = np.array([], dtype=bool)

            if total_rides > 0:
                total_wins = int((ja_kakuteijyuni == 1).sum())
            else:
                total_wins = 0

            k_smooth = 25
            if total_rides >= 10:
                global_wr = total_wins / max(total_rides, 1)
                cond_rides = int(cond_mask.sum()) if len(cond_mask) > 0 else 0
                if cond_rides >= 5:
                    cond_wins = (
                        int((ja_kakuteijyuni[cond_mask] == 1).sum())
                        if len(cond_mask) > 0
                        else 0
                    )
                    cond_wr = cond_wins / max(cond_rides, 1)
                    w = min(cond_rides / (cond_rides + k_smooth), 1.0)
                else:
                    cond_wr = global_wr
                    w = 0.0
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

            # HODDS-02: クラストラジェクトリ (D-05, D-06, D-07)
            if n_past >= 2 and "gradecd" in horse_arrs and "jyokencd1" in horse_arrs:
                _ct_grade = horse_arrs["gradecd"][valid_mask][start:idx]
                _ct_jyoken = horse_arrs["jyokencd1"][valid_mask][start:idx]
                (
                    class_promotions, class_demotions, class_net_change,
                    class_max_level, class_level_std,
                    v_recovery_flag, v_recovery_duration,
                ) = compute_class_trajectory(_ct_grade, _ct_jyoken)
            else:
                class_promotions = float("nan")
                class_demotions = float("nan")
                class_net_change = float("nan")
                class_max_level = float("nan")
                class_level_std = float("nan")
                v_recovery_flag = float("nan")
                v_recovery_duration = float("nan")

            # HODDS-03: フォーム改善率 (D-08, D-09)
            if n_past >= 2 and "harontimel3" in horse_arrs:
                _fi_ht = horse_arrs["harontimel3"][valid_mask][start:idx].astype(float)
                _fi_kj = horse_arrs["kakuteijyuni"][valid_mask][start:idx].astype(float)
                _fi_ss = horse_arrs["syussotosu"][valid_mask][start:idx].astype(float)
                time_improvement_rate, position_improvement_rate = (
                    compute_form_improvement_rate(_fi_ht, _fi_kj, _fi_ss)
                )
            else:
                time_improvement_rate = float("nan")
                position_improvement_rate = float("nan")

            # class_move: 現在クラス - 前走クラス (正=昇級, 負=降級)
            current_class_level = _class_level_from_values(
                getattr(row, "gradecd", float("nan")),
                getattr(row, "jyokencd1", float("nan")),
            )
            if hist_idx > 0 and horse_arrs is not None:
                last_grade = (
                    horse_arrs["gradecd"][history_mask][hist_start:hist_idx][-1]
                    if "gradecd" in horse_arrs
                    else float("nan")
                )
                last_jyoken = (
                    horse_arrs["jyokencd1"][history_mask][hist_start:hist_idx][-1]
                    if "jyokencd1" in horse_arrs
                    else float("nan")
                )
                last_class_level = _class_level_from_values(last_grade, last_jyoken)
                class_move = (
                    current_class_level - last_class_level
                    if not np.isnan(current_class_level) and not np.isnan(last_class_level)
                    else float("nan")
                )
            else:
                class_move = float("nan")

            # equipment / transfer / track-condition deltas
            current_blinker = _blinker_flag(getattr(row, "blinker", float("nan")))
            current_jyocd = _coerce_float(getattr(row, "jyocd", float("nan")))
            current_track_condition = _coerce_float(
                getattr(row, "track_condition_code", float("nan"))
            )
            if hist_idx > 0 and horse_arrs is not None:
                last_blinker = (
                    _blinker_flag(horse_arrs["blinker"][history_mask][hist_start:hist_idx][-1])
                    if "blinker" in horse_arrs
                    else float("nan")
                )
                blinker_change = (
                    current_blinker - last_blinker
                    if not np.isnan(current_blinker) and not np.isnan(last_blinker)
                    else float("nan")
                )

                last_jyocd = (
                    _coerce_float(horse_arrs["jyocd"][history_mask][hist_start:hist_idx][-1])
                    if "jyocd" in horse_arrs
                    else float("nan")
                )
                is_nar_transfer = (
                    1.0
                    if not np.isnan(current_jyocd)
                    and not np.isnan(last_jyocd)
                    and current_jyocd <= 10
                    and last_jyocd > 10
                    else 0.0
                    if not np.isnan(current_jyocd) and not np.isnan(last_jyocd)
                    else float("nan")
                )

                recent_jyocd = (
                    pd.to_numeric(
                        pd.Series(horse_arrs["jyocd"][history_mask][hist_start:hist_idx]),
                        errors="coerce",
                    )
                    if "jyocd" in horse_arrs
                    else pd.Series(dtype=float)
                )
                nar_recent_ratio = (
                    float((recent_jyocd > 10).mean())
                    if not recent_jyocd.empty and recent_jyocd.notna().any()
                    else float("nan")
                )

                last_track_condition = (
                    _coerce_float(
                        horse_arrs["track_condition_code"][history_mask][hist_start:hist_idx][-1]
                    )
                    if "track_condition_code" in horse_arrs
                    else float("nan")
                )
                track_condition_delta = (
                    current_track_condition - last_track_condition
                    if not np.isnan(current_track_condition) and not np.isnan(last_track_condition)
                    else float("nan")
                )
            else:
                blinker_change = float("nan")
                is_nar_transfer = float("nan")
                nar_recent_ratio = float("nan")
                track_condition_delta = float("nan")

            # -----------------------------------------------------------------
            # FEAT-02: 単勝特化新特徴量 (5 features)
            # -----------------------------------------------------------------

            # current_distance_bin: 現在レースの距離ビン (distance_change/env_adaptabilityで共用)
            if hasattr(row, "distance_bin") and not pd.isna(getattr(row, "distance_bin", None)):
                current_db = str(getattr(row, "distance_bin"))
            else:
                current_db = _compute_distance_bin(
                    getattr(row, "kyori", None), getattr(row, "surface", "")
                )

            # distance_change: 距離変更要検知 (current distance_bin != last race distance_bin)
            if hist_idx > 0 and horse_arrs is not None and "distance_bin" in horse_arrs:
                last_db = str(horse_arrs["distance_bin"][history_mask][hist_start:hist_idx][-1])
                distance_change: float = 1.0 if current_db != last_db else 0.0
            else:
                distance_change = float("nan")

            # surface_change: 芝ダート変更要検知 (current surface != last race surface)
            if hist_idx > 0 and horse_arrs is not None and "surface" in horse_arrs:
                current_surf = str(getattr(row, "surface", ""))
                last_surf = str(horse_arrs["surface"][history_mask][hist_start:hist_idx][-1])
                surface_change: float = 1.0 if current_surf != last_surf else 0.0
            else:
                surface_change = float("nan")

            # HODDS-04: 環境変化適性 (D-10, D-11)
            _env_nan_keys = [
                "dist_change_avg_pos", "dist_change_win_rate", "dist_change_exp_count",
                "surf_change_avg_pos", "surf_change_win_rate", "surf_change_exp_count",
                "cond_change_avg_pos", "cond_change_win_rate", "cond_change_exp_count",
            ]
            if (
                hist_idx > 0
                and horse_arrs is not None
                and "distance_bin" in horse_arrs
                and "surface" in horse_arrs
                and "track_condition_code" in horse_arrs
            ):
                _ea_kj = horse_arrs["kakuteijyuni"][history_mask][hist_start:hist_idx].astype(float)
                _ea_ss = horse_arrs["syussotosu"][history_mask][hist_start:hist_idx].astype(float)
                _ea_db = horse_arrs["distance_bin"][history_mask][hist_start:hist_idx]
                _ea_surf = horse_arrs["surface"][history_mask][hist_start:hist_idx]
                _ea_cond = horse_arrs["track_condition_code"][history_mask][hist_start:hist_idx].astype(float)
                env_stats = compute_env_adaptability(
                    _ea_kj, _ea_ss, _ea_db, _ea_surf, _ea_cond,
                    current_distance_bin=current_db,
                    current_surface=str(getattr(row, "surface", "")),
                    current_track_condition=_coerce_float(
                        getattr(row, "track_condition_code", float("nan"))
                    ),
                )
            else:
                env_stats = {k: float("nan") for k in _env_nan_keys}

            # class_drop_bounce: クラス落リバウンド (降級かつ直近成績悪化時に高い値)
            # norm_recent_b = (kj - 1) / (ss - 1): 0=1着, 1=最下位 の正規化着順
            # avg_recent_b > 0.5 は直近レースで後半着順 (悪いフォーム) を意味する
            # avg が高いほどフォームが悪く、バウンスシグナルが強い
            if hist_idx >= 2 and not np.isnan(class_move) and class_move < -0.5:
                recent_kj_b = hp_kakuteijyuni[-2:].astype(float)
                recent_ss_b = hp_syussotosu[-2:].astype(float)
                valid_recent_b = recent_ss_b > 1
                if valid_recent_b.any():
                    norm_recent_b = (recent_kj_b[valid_recent_b] - 1) / (recent_ss_b[valid_recent_b] - 1)
                    avg_recent_b = float(np.nanmean(norm_recent_b))
                    class_drop_bounce: float = (
                        min(float(abs(class_move)) * avg_recent_b, 10.0)
                        if avg_recent_b > 0.5
                        else 0.0
                    )
                else:
                    class_drop_bounce = float("nan")
            elif not np.isnan(class_move):
                class_drop_bounce = 0.0
            else:
                class_drop_bounce = float("nan")

            # win_dominance: 勝利dominance (勝利時の平均フィールドサイズ)
            if n_past > 0:
                win_mask = hp_kakuteijyuni == 1
                if win_mask.any():
                    win_sizes = hp_syussotosu[win_mask].astype(float)
                    valid_sizes = win_sizes[~np.isnan(win_sizes) & (win_sizes > 0)]
                    win_dominance: float = (
                        float(np.mean(valid_sizes)) if len(valid_sizes) > 0 else float("nan")
                    )
                else:
                    # 走歴はあるが勝利なし -- NaNで「勝利情報なし」を表現
                    # (no-history case と同じNaNなので、LightGBMが同一扱い可能)
                    win_dominance = float("nan")
            else:
                win_dominance = float("nan")

            # freshness_score: フレッシュネス (休息品質 x 直近フォーム品質)
            if not np.isnan(days_since) and n_past >= 3:
                if days_since <= 7:
                    rest_score = 0.3
                elif days_since <= 30:
                    rest_score = 0.7
                elif days_since <= 60:
                    rest_score = 1.0
                elif days_since <= 90:
                    rest_score = 0.8
                else:
                    rest_score = 0.4
                recent_kj_f = hp_kakuteijyuni[-3:].astype(float)
                recent_ss_f = hp_syussotosu[-3:].astype(float)
                valid_recent_f = recent_ss_f > 1
                if valid_recent_f.any():
                    norm_pos = (recent_kj_f[valid_recent_f] - 1) / (recent_ss_f[valid_recent_f] - 1)
                    form_score = 1.0 - float(np.nanmean(norm_pos))
                    freshness_score: float = max(rest_score * max(form_score, 0.0), 0.0)
                else:
                    freshness_score = float("nan")
            else:
                freshness_score = float("nan")

            # TRF-02: weighted_recent_form (EMA halflife=3, D-07/D-08)
            if n_past > 0:
                # weighted_recent_form_finish: EMA(norm_finish_logit, halflife=3)
                _wrf_logits = _norm_finish_logit_vec(
                    hp_kakuteijyuni.astype(float),
                    hp_syussotosu.astype(float),
                )
                _wrf_valid_logits = _wrf_logits[~np.isnan(_wrf_logits)]
                if len(_wrf_valid_logits) > 0:
                    halflife_wrf = 3
                    decay_wrf = np.log(2) / halflife_wrf
                    n_wrf = len(_wrf_valid_logits)
                    weights_wrf = (1 - decay_wrf) ** np.arange(n_wrf)
                    weights_wrf = weights_wrf[::-1]
                    weights_wrf = weights_wrf / weights_wrf.sum()
                    weighted_recent_form_finish: float = float(
                        np.sum(_wrf_valid_logits * weights_wrf)
                    )
                else:
                    weighted_recent_form_finish = float("nan")

                # weighted_recent_form_time: EMA(timediff, halflife=3)
                if _has_timediff:
                    _wrf_td = horse_arrs["timediff"][valid_mask][start:idx].astype(float)
                    _wrf_valid_td = _wrf_td[~np.isnan(_wrf_td)]
                    if len(_wrf_valid_td) > 0:
                        halflife_wrf2 = 3
                        decay_wrf2 = np.log(2) / halflife_wrf2
                        n_wrf2 = len(_wrf_valid_td)
                        weights_wrf2 = (1 - decay_wrf2) ** np.arange(n_wrf2)
                        weights_wrf2 = weights_wrf2[::-1]
                        weights_wrf2 = weights_wrf2 / weights_wrf2.sum()
                        weighted_recent_form_time: float = float(
                            np.sum(_wrf_valid_td * weights_wrf2)
                        )
                    else:
                        weighted_recent_form_time = float("nan")
                else:
                    weighted_recent_form_time = float("nan")
            else:
                weighted_recent_form_finish = float("nan")
                weighted_recent_form_time = float("nan")

            # -----------------------------------------------------------------
            # HLF-03: LapTime pace features from past races (PIT-safe)
            # -----------------------------------------------------------------
            pace_ratio_avg: float = float("nan")
            pace_ratio_zscore: float = float("nan")
            pace_ratio_trend: float = float("nan")
            pace_early_avg: float = float("nan")
            pace_mid_avg: float = float("nan")
            pace_late_avg: float = float("nan")

            if _has_laptime:
                _pace_data = _pace_lookup.get(ketto)
                if _pace_data is not None and len(_pace_data["pace_ratios"]) > 0:
                    # PIT-safe: searchsorted on pace_dates
                    pace_dates_arr = _pace_data["race_dates"].astype("datetime64[ns]")
                    pace_idx = pace_dates_arr.searchsorted(target_date_np, side="left")
                    if pace_idx > 0:
                        past_pr = _pace_data["pace_ratios"][:pace_idx]
                        past_ea = _pace_data["early_avgs"][:pace_idx]
                        past_ma = _pace_data["mid_avgs"][:pace_idx]
                        past_la = _pace_data["late_avgs"][:pace_idx]

                        if len(past_pr) > 0:
                            # pace_ratio_avg: EMA(halflife=3)
                            pr_valid = past_pr[~np.isnan(past_pr)]
                            if len(pr_valid) > 0:
                                halflife_pr = 3
                                decay_pr = np.log(2) / halflife_pr
                                n_pr = len(pr_valid)
                                weights_pr = (1 - decay_pr) ** np.arange(n_pr)
                                weights_pr = weights_pr[::-1]
                                weights_pr = weights_pr / weights_pr.sum()
                                pace_ratio_avg = float(np.sum(pr_valid * weights_pr))

                                # pace_ratio_trend: linear regression of last 3
                                pr_trend = pr_valid[-3:] if len(pr_valid) >= 3 else pr_valid
                                if len(pr_trend) >= 2:
                                    x_pr = np.arange(len(pr_trend), dtype=float)
                                    pace_ratio_trend = float(np.polyfit(x_pr, pr_trend, 1)[0])

                                # pace_ratio_zscore: global z-score from horse's own history
                                if len(pr_valid) >= 3:
                                    pr_mean = float(np.mean(pr_valid))
                                    pr_std = float(np.std(pr_valid, ddof=1))
                                    if pr_std > 0:
                                        pace_ratio_zscore = float(
                                            (pace_ratio_avg - pr_mean) / pr_std
                                        )

                            # Segment averages: simple mean of past values
                            ea_valid = past_ea[~np.isnan(past_ea)]
                            if len(ea_valid) > 0:
                                pace_early_avg = float(np.mean(ea_valid))

                            ma_valid = past_ma[~np.isnan(past_ma)]
                            if len(ma_valid) > 0:
                                pace_mid_avg = float(np.mean(ma_valid))

                            la_valid = past_la[~np.isnan(past_la)]
                            if len(la_valid) > 0:
                                pace_late_avg = float(np.mean(la_valid))

            results.append(
                {
                    "race_id": row.race_id,
                    "umaban": row.umaban,
                    "norm_finish_logit_avg": norm_finish_logit_avg,
                    "harontimel5_avg": harontimel5_avg,
                    "harontimel5_zscore": harontimel5_zscore,
                    "harontime_late_trend": harontime_late_trend,
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
                    "class_move": class_move,
                    "blinker_change": blinker_change,
                    "is_nar_transfer": is_nar_transfer,
                    "nar_recent_ratio": nar_recent_ratio,
                    "track_condition_delta": track_condition_delta,
                    # FEAT-02: 単勝特化新特徴量
                    "distance_change": distance_change,
                    "surface_change": surface_change,
                    "class_drop_bounce": class_drop_bounce,
                    "win_dominance": win_dominance,
                    "freshness_score": freshness_score,
                    # TSER-02: クラス調整フォーメトリック
                    "class_adj_formetric": class_adj_formetric,
                    # TSER-03: z-score改善トラジェクトリ
                    "haron_zscore_trend": haron_zscore_trend,
                    # HODDS-02: クラストラジェクトリ
                    "class_promotions": class_promotions,
                    "class_demotions": class_demotions,
                    "class_net_change": class_net_change,
                    "class_max_level": class_max_level,
                    "class_level_std": class_level_std,
                    "v_recovery_flag": v_recovery_flag,
                    "v_recovery_duration": v_recovery_duration,
                    # HODDS-03: フォーム改善率
                    "time_improvement_rate": time_improvement_rate,
                    "position_improvement_rate": position_improvement_rate,
                    # HODDS-04: 環境変化適性
                    "dist_change_avg_pos": env_stats["dist_change_avg_pos"],
                    "dist_change_win_rate": env_stats["dist_change_win_rate"],
                    "dist_change_exp_count": env_stats["dist_change_exp_count"],
                    "surf_change_avg_pos": env_stats["surf_change_avg_pos"],
                    "surf_change_win_rate": env_stats["surf_change_win_rate"],
                    "surf_change_exp_count": env_stats["surf_change_exp_count"],
                    "cond_change_avg_pos": env_stats["cond_change_avg_pos"],
                    "cond_change_win_rate": env_stats["cond_change_win_rate"],
                    "cond_change_exp_count": env_stats["cond_change_exp_count"],
                    # TRF-02: weighted_recent_form (EMA halflife=3, D-07/D-08)
                    "weighted_recent_form_finish": weighted_recent_form_finish,
                    "weighted_recent_form_time": weighted_recent_form_time,
                    # HLF-01: HaronTime L4 history stats
                    "harontimel4_avg": harontimel4_avg,
                    "harontimel4_zscore": harontimel4_zscore,
                    "harontimel4_trend": harontimel4_trend,
                    # HLF-01: harontime_last3f unified column
                    "harontime_last3f_avg": harontime_last3f_avg,
                    "harontime_last3f_zscore": harontime_last3f_zscore,
                    "harontime_last3f_trend": harontime_last3f_trend,
                    # HLF-03: LapTime pace features (placeholder until Task 2)
                    "pace_ratio_avg": pace_ratio_avg,
                    "pace_ratio_zscore": pace_ratio_zscore,
                    "pace_ratio_trend": pace_ratio_trend,
                    "pace_early_avg": pace_early_avg,
                    "pace_mid_avg": pace_mid_avg,
                    "pace_late_avg": pace_late_avg,
                }
            )

        logger.debug("HorseHistoryFeatures: done (%d rows)", len(results))
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
            "harontimel5_avg",
            "harontimel5_zscore",
            "timediff_avg",
            "jyuni1c_avg",
            "jyuni4c_avg",
            "closing_index_avg",
            # TRF-01: 新規race-rank列 (D-13)
            "form_trend",
            "blood_total_wr",
            "blood_surface_wr",
            # HLF-02: HaronTime L4 + unified race-rank
            "harontimel4_avg",
            "harontime_last3f_avg",
            # 注意: kyakusitukubun_cd, jockey系, harontime_late_trend は
            # race_rank を生成しない
        ]
        df = df.copy()
        for col in race_rank_cols:
            if col not in df.columns:
                continue
            df[f"{col}_race_rank"] = (
                df.groupby("race_id", observed=True)[col]
                .rank(pct=True, method="average")
            )
        return df
