"""mining_features.py -- Group F: n_mining 予想走破タイム特徴量 (DATA-04)

JRA-VAN DataLab の n_mining テーブル(82列)から DataKubun=3(直前予想)のレコードを使用し、
wide format (1レース1行, 18頭分の列) を long format に変換して特徴量を生成する。

主な特徴量:
  - dm_time_rank: レース内予想タイムランク (1=最速予想)
  - dm_time_zscore: レース内予想タイム z-score
  - dm_confidence_range: 予想誤差(信頼度)の幅 (GosaP + GosaM)
  - dm_time_margin_to_fav: レース最速予想とのタイム差

PIT安全性:
  全82列がPRE (レース前予想データ) であることを PIT監査で確認済み。
  DMTime はJRA-VANの予想走破タイムであり、実際の走破タイムではない。
  DataKubun=3 は馬体重発表後の最終予想であり、最も情報量が多い。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "dm_time_rank",
    "dm_time_zscore",
    "dm_confidence_range",
    "dm_time_margin_to_fav",
]


# ---------------------------------------------------------------------------
# Wide-to-long pivot
# ---------------------------------------------------------------------------


def _pivot_mining_to_long(mining_df: pd.DataFrame) -> pd.DataFrame:
    """n_mining wide format (18 horses per row) を long format に変換する。

    各レース1行 (Umaban1..18, DMTime1..18, DMGosaP1..18, DMGosaM1..18) を
    (race_id, umaban, dm_time, dm_gosa_plus, dm_gosa_minus) のlong形式に変換。

    "sp" (初期値) のスロットは除外される。
    """
    subsets: list[pd.DataFrame] = []

    for i in range(1, 19):
        rename_map: dict[str, str] = {
            f"umaban{i}": "umaban",
            f"dmtime{i}": "dm_time",
            f"dmgosap{i}": "dm_gosa_plus",
            f"dmgosam{i}": "dm_gosa_minus",
        }

        # 必要な列が存在する場合のみ処理
        available_cols = [c for c in rename_map if c in mining_df.columns]
        if len(available_cols) < 4:
            continue

        cols_needed = ["race_id"] + list(rename_map.keys())
        subset = mining_df[cols_needed].rename(columns=rename_map).copy()

        # "sp" (初期値) と NaN を除外
        mask = subset["umaban"].notna() & (subset["umaban"] != "sp")
        subset = subset[mask].copy()

        # 型変換
        subset["umaban"] = pd.to_numeric(subset["umaban"], errors="coerce").astype(
            "Int64"
        )
        subset["dm_time"] = pd.to_numeric(subset["dm_time"], errors="coerce")
        subset["dm_gosa_plus"] = pd.to_numeric(subset["dm_gosa_plus"], errors="coerce")
        subset["dm_gosa_minus"] = pd.to_numeric(
            subset["dm_gosa_minus"], errors="coerce"
        )

        # 数値変換後 NaN を除外 (invalid な dm_time)
        subset = subset[subset["dm_time"].notna() & subset["umaban"].notna()]

        subsets.append(subset)

    if not subsets:
        return pd.DataFrame(columns=["race_id", "umaban", "dm_time", "dm_gosa_plus", "dm_gosa_minus"])

    result = pd.concat(subsets, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# MiningFeatures
# ---------------------------------------------------------------------------


class MiningFeatures:
    """n_mining テーブルから予想走破タイム特徴量を生成。

    DataKubun=3 (直前予想、馬体重発表後) のレコードを使用。
    利用不可の場合は DataKubun=2 にフォールバック。
    """

    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._mining_cache: pd.DataFrame | None = None

    def _load_mining(self) -> pd.DataFrame:
        """mining.parquet を読み込む (キャッシュ付き)。存在しない場合は空DataFrame。"""
        if self._mining_cache is None:
            if self.store.exists("raw", "mining"):
                self._mining_cache = self.store.read("raw", "mining")
            else:
                self._mining_cache = pd.DataFrame()
        return self._mining_cache

    def _ensure_race_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """race_id列がなければ year+monthday+jyocd+kaiji+nichiji+racenum から計算。"""
        if "race_id" not in df.columns:
            required = ["year", "monthday", "jyocd", "kaiji", "nichiji", "racenum"]
            if all(c in df.columns for c in required):
                df = df.copy()
                df["race_id"] = (
                    df["year"].astype(str).str.zfill(4)
                    + df["monthday"].astype(str).str.zfill(4)
                    + df["jyocd"].astype(str).str.zfill(2)
                    + df["kaiji"].astype(str).str.zfill(2)
                    + df["nichiji"].astype(str).str.zfill(2)
                    + df["racenum"].astype(str).str.zfill(2)
                )
        return df

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """entry_df (race_id, umaban) から mining特徴量を計算して返す。

        Returns DataFrame with columns: race_id, umaban, + FEATURE_COLS
        """
        result = entry_df[["race_id", "umaban"]].copy()
        for col in FEATURE_COLS:
            result[col] = np.nan

        mining_df = self._load_mining()
        if mining_df.empty:
            return result

        # race_id を保証
        mining_df = self._ensure_race_id(mining_df)
        if "race_id" not in mining_df.columns:
            return result

        # DataKubun フィルタ: 3 (直前予想) を優先、なければ 2 にフォールバック
        kubun_col = "datakubun"
        if kubun_col in mining_df.columns:
            mining_df[kubun_col] = mining_df[kubun_col].astype(str)
            kubun3 = mining_df[mining_df[kubun_col] == "3"]
            if not kubun3.empty:
                mining_df = kubun3
            else:
                kubun2 = mining_df[mining_df[kubun_col] == "2"]
                if not kubun2.empty:
                    mining_df = kubun2

        # wide -> long 変換
        long_df = _pivot_mining_to_long(mining_df)
        if long_df.empty:
            return result

        # (race_id, umaban) でマージ
        long_df["umaban"] = long_df["umaban"].astype("Int64")
        result["umaban"] = result["umaban"].astype("Int64")

        merged = result[["race_id", "umaban"]].merge(
            long_df[["race_id", "umaban", "dm_time", "dm_gosa_plus", "dm_gosa_minus"]],
            on=["race_id", "umaban"],
            how="left",
        )

        if merged.empty:
            return result

        # --- dm_time_rank: レース内予想タイムランク (小さい=速い=良い) ---
        merged["dm_time_rank"] = merged.groupby("race_id", observed=True)[
            "dm_time"
        ].rank(method="min", ascending=True)

        # --- dm_time_zscore: レース内予想タイム z-score ---
        grp = merged.groupby("race_id", observed=True)["dm_time"]
        mean = grp.transform("mean")
        std = grp.transform("std")
        merged["dm_time_zscore"] = (merged["dm_time"] - mean) / std.replace(0, np.nan)
        merged["dm_time_zscore"] = merged["dm_time_zscore"].fillna(0.0)

        # --- dm_confidence_range: 予想誤差幅 (GosaP + GosaM) ---
        merged["dm_confidence_range"] = (
            merged["dm_gosa_plus"].fillna(0) + merged["dm_gosa_minus"].fillna(0)
        )

        # --- dm_time_margin_to_fav: レース最速予想とのタイム差 ---
        min_time = merged.groupby("race_id", observed=True)["dm_time"].transform("min")
        merged["dm_time_margin_to_fav"] = merged["dm_time"] - min_time

        # 結果を反映
        for col in FEATURE_COLS:
            result[col] = merged[col].values

        return result[["race_id", "umaban"] + FEATURE_COLS]

    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """df (race_id, umaban) から mining特徴量を計算 (compute() へのエイリアス)。

        _train_submodel() 統合用。
        """
        return self.compute(df)
