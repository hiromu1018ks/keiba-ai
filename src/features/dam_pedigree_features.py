"""dam_pedigree_features.py -- Group B-2: 繁殖牝馬産駒成績特徴量 (DATA-01)

n_sanku (産駒血統マスタ) から繁殖牝馬の産駒成績を集計し、特徴量を生成する。

主な特徴量:
  - dam_wr: 繁殖牝馬の産駒勝率 (Beta平滑化)
  - dam_surface_wr: 繁殖牝馬の産駒芝勝率 (Beta平滑化)
  - dam_prize_log: 繁殖牝馬の産駒平均賞金 (log変換)
  - breeder_strength: 繁殖牝馬の産駒を生産した生産者数 (log変換)

PIT安全性:
  n_sanku 全26列がPRE (静的マスターデータ)。レース結果は一切含まれない。
  horse_career_stats は各レース時点での事前累積値 (PIT安全)。

Cross-reference chain:
  entry_df.kettonum -> sanku (find MNum=dam) -> sanku again (find all offspring)
  -> career_stats (aggregate wins/starts per dam)
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

# Beta prior parameters for win-rate smoothing
ALPHA_PRIOR: int = 1
BETA_PRIOR: int = 10
TOTAL_OFFSET: int = ALPHA_PRIOR + BETA_PRIOR  # = 11

FEATURE_COLS: list[str] = [
    "dam_wr",
    "dam_surface_wr",
    "dam_prize_log",
    "breeder_strength",
]


class DamPedigreeFeatures:
    """繁殖牝馬産駒成績特徴量を生成。

    n_sanku から繁殖牝馬(MNum)を特定し、その産駒のキャリア統計を集計する。
    """

    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._sanku_cache: pd.DataFrame | None = None
        self._career_cache: pd.DataFrame | None = None

    def _load_sanku(self) -> pd.DataFrame:
        """sanku.parquet を読み込みキャッシュする"""
        if self._sanku_cache is None:
            if self.store.exists("raw", "sanku"):
                self._sanku_cache = self.store.read("raw", "sanku")
            else:
                self._sanku_cache = pd.DataFrame()
        return self._sanku_cache

    def _load_career_stats(self) -> pd.DataFrame:
        """horse_career_stats を読み込みキャッシュする"""
        if self._career_cache is None:
            from db.readers import load_career_stats

            self._career_cache = load_career_stats(self.store)
        return self._career_cache

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """entry_df (race_id, umaban, kettonum) -> 繁殖牝馬特徴量 DataFrame。

        Args:
            entry_df: race_id, umaban, kettonum 列を持つ DataFrame

        Returns:
            race_id, umaban + FEATURE_COLS を持つ DataFrame
        """
        # 同一(race_id, umaban)の重複を排除
        entry_df = entry_df.drop_duplicates(subset=["race_id", "umaban"], keep="first")

        sanku = self._load_sanku()
        career = self._load_career_stats()

        if (
            sanku.empty
            or career.empty
            or "kettonum" not in entry_df.columns
        ):
            return entry_df[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        # 列名正規化 (小文字)
        sanku_cols = sanku.columns.str.lower()
        sanku = sanku.copy()
        sanku.columns = sanku_cols

        # kettonum -> MNum (母の血統登録番号) のマッピング
        if "kettonum" not in sanku.columns or "mnum" not in sanku.columns:
            return entry_df[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        # エントリー馬の kettonum から dam MNum を特定
        kettonum_to_mnum = sanku.set_index("kettonum")["mnum"]
        entry_mnums = entry_df["kettonum"].map(kettonum_to_mnum)

        # 各 dam MNum について産駎を特定
        unique_mnums = entry_mnums.dropna().unique()

        if len(unique_mnums) == 0:
            return entry_df[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        # dam 列の正規化 (breedercode or breeder_code)
        breeder_col = "breedercode" if "breedercode" in sanku.columns else "breeder_code"

        # 各 dam ごとに産駒情報を集計
        dam_features: dict[str, dict[str, float]] = {}

        # career stats の最新行を各馬ごとに取得
        if "race_date" in career.columns:
            career_latest = career.sort_values("race_date").groupby("kettonum").last()
        else:
            career_latest = career.groupby("kettonum").last()

        for mnum in unique_mnums:
            # この dam の産駎一覧
            offspring = sanku[sanku["mnum"] == mnum]
            offspring_kettonums = offspring["kettonum"].values

            # 産駎のキャリア統計を取得
            offspring_career = career_latest[
                career_latest.index.isin(offspring_kettonums)
            ]

            if offspring_career.empty:
                dam_features[mnum] = {
                    "dam_wr": np.nan,
                    "dam_surface_wr": np.nan,
                    "dam_prize_log": np.nan,
                    "breeder_strength": np.nan,
                }
                continue

            # 産駒全体の勝率
            total_wins = offspring_career["cum_wins"].sum()
            total_starts = offspring_career["cum_starts"].sum()
            dam_wr = (total_wins + ALPHA_PRIOR) / (total_starts + TOTAL_OFFSET)

            # 産駒の芝勝率
            total_turf_wins = offspring_career.get("cum_turf_wins", pd.Series(0)).sum()
            total_turf_starts = offspring_career.get("cum_turf_starts", pd.Series(0)).sum()
            if total_turf_starts > 0:
                dam_surface_wr = (total_turf_wins + ALPHA_PRIOR) / (
                    total_turf_starts + TOTAL_OFFSET
                )
            else:
                dam_surface_wr = np.nan

            # 産駒の平均賞金 (log変換)
            mean_prize = offspring_career.get("cum_prize", pd.Series([0])).mean()
            dam_prize_log = np.log1p(mean_prize) if mean_prize > 0 else np.nan

            # breeder_strength: log(1 + unique breeder count)
            if breeder_col in offspring.columns:
                unique_breeders = offspring[breeder_col].dropna().nunique()
                breeder_strength = np.log1p(unique_breeders)
            else:
                breeder_strength = np.nan

            dam_features[mnum] = {
                "dam_wr": dam_wr,
                "dam_surface_wr": dam_surface_wr,
                "dam_prize_log": dam_prize_log,
                "breeder_strength": breeder_strength,
            }

        # 結果を entry_df にマップ
        result = entry_df[["race_id", "umaban", "kettonum"]].copy()
        for col in FEATURE_COLS:
            result[col] = entry_mnums.map(
                lambda m, c=col: dam_features.get(m, {}).get(c, np.nan)
                if pd.notna(m)
                else np.nan
            )

        return result[["race_id", "umaban"] + FEATURE_COLS]
