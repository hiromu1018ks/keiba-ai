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

  career stats を (race_id, kettonum) ごとに保持するため、各エントリの
  race_id に対応する race_date を基準に、その日以前の最新累積値を産駒ごとに
  取得する (merge_asof による PIT-safe 取得)。

Cross-reference chain:
  entry_df.kettonum -> sanku (find MNum=dam) -> sanku again (find all offspring)
  -> career_stats (aggregate wins/starts per dam, PIT-safe by race_date)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

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

        PIT安全: 各エントリの race_id に対応する race_date を基準に、
        その日以前の最新累積値を産駒ごとに取得する。

        Args:
            entry_df: race_id, umaban, kettonum 列を持つ DataFrame

        Returns:
            race_id, umaban + FEATURE_COLS を持つ DataFrame
        """
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

        # --- PIT-safe career lookup ---
        # career に race_date / race_id がなければ PIT-safe な取得が不可。
        if "race_date" not in career.columns or "race_id" not in career.columns:
            # フォールバック: 従来の last() (PIT違反あり、ログ警告)
            logger.warning(
                "DamPedigreeFeatures: career stats に race_date/race_id がなく "
                "PIT-safe 取得不可。フォールバック (last()) を使用。"
            )
            return self._compute_fallback(entry_df, sanku, career)

        # エントリー馬の kettonum から dam MNum を特定
        kettonum_to_mnum = sanku.set_index("kettonum")["mnum"]
        result = entry_df[["race_id", "umaban", "kettonum"]].copy()
        result["mnum"] = result["kettonum"].map(kettonum_to_mnum)

        # mnum が取れないエントリの早期チェック
        if result["mnum"].notna().sum() == 0:
            return result[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        # dam 列の正規化 (breedercode or breeder_code)
        breeder_col = "breedercode" if "breedercode" in sanku.columns else "breeder_code"

        # 各エントリの race_date を取得
        if "race_date" in entry_df.columns:
            entry_race_dates = entry_df[["race_id", "race_date"]].drop_duplicates("race_id")
        else:
            entry_race_dates = career[["race_id", "race_date"]].drop_duplicates("race_id")
        result = result.merge(entry_race_dates, on="race_id", how="left")

        # merge() は index を再構築するため、対応する mask も再計算する。
        valid = result["mnum"].notna()

        # 全 dam の産駒 kettonum を収集
        mnums_with_entries = result.loc[valid, "mnum"].unique()
        offspring_per_mnum: dict[str, list[str]] = {}
        all_offspring_kettonums: set[str] = set()
        for mnum in mnums_with_entries:
            offspring = sanku.loc[sanku["mnum"] == mnum, "kettonum"].tolist()
            offspring_per_mnum[mnum] = offspring
            all_offspring_kettonums.update(offspring)

        # career を産駒に絞り込み
        career_offspring = career[career["kettonum"].isin(all_offspring_kettonums)].copy()

        if career_offspring.empty:
            return result[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        # 重複 (race_id, kettonum) を排除し、ソート
        career_offspring = career_offspring.sort_values(
            ["kettonum", "race_date", "race_id"]
        ).drop_duplicates(subset=["kettonum", "race_id"], keep="last")

        # 必要な累積列
        cum_cols = ["cum_wins", "cum_starts", "cum_turf_wins", "cum_turf_starts", "cum_prize"]

        # 各エントリ (race_id, mnum) ごとに産駒の PIT キャリアを集計
        # (mnum, target_date) の組でキャッシュ
        dam_features_cache: dict[tuple[str, object], dict[str, float]] = {}

        for idx in result.index:
            mnum = result.loc[idx, "mnum"]
            if pd.isna(mnum):
                for c in FEATURE_COLS:
                    result.loc[idx, c] = np.nan
                continue

            target_date = result.loc[idx, "race_date"]
            if pd.isna(target_date):
                for c in FEATURE_COLS:
                    result.loc[idx, c] = np.nan
                continue

            cache_key = (mnum, target_date)
            if cache_key in dam_features_cache:
                feats = dam_features_cache[cache_key]
            else:
                offspring_kettonums = offspring_per_mnum.get(mnum, [])
                if len(offspring_kettonums) == 0:
                    feats = {c: np.nan for c in FEATURE_COLS}
                else:
                    oc = career_offspring[
                        career_offspring["kettonum"].isin(offspring_kettonums)
                    ]

                    if oc.empty:
                        feats = {c: np.nan for c in FEATURE_COLS}
                    else:
                        # merge_asof: target_date 以前の最新行を kettonum ごとに取得
                        # left は race_date でソート, right も race_date でソート必須
                        oc_sorted = oc.sort_values("race_date").copy()
                        oc_sorted["kettonum"] = oc_sorted["kettonum"].astype(str)
                        left = pd.DataFrame({
                            "kettonum": [str(k) for k in offspring_kettonums],
                            "race_date": target_date,
                        })
                        left = left.sort_values("race_date")

                        merged_asof = pd.merge_asof(
                            left,
                            oc_sorted[["kettonum", "race_date"] + cum_cols],
                            on="race_date",
                            by="kettonum",
                            direction="backward",
                        )

                        # 産駒全体の勝率 (Beta 平滑化)
                        total_wins = merged_asof["cum_wins"].fillna(0).sum()
                        total_starts = merged_asof["cum_starts"].fillna(0).sum()
                        if total_starts > 0:
                            dam_wr = (total_wins + ALPHA_PRIOR) / (
                                total_starts + TOTAL_OFFSET
                            )
                        else:
                            dam_wr = np.nan

                        # 産駒の芝勝率
                        total_turf_wins = merged_asof["cum_turf_wins"].fillna(0).sum()
                        total_turf_starts = merged_asof["cum_turf_starts"].fillna(0).sum()
                        if total_turf_starts > 0:
                            dam_surface_wr = (
                                total_turf_wins + ALPHA_PRIOR
                            ) / (total_turf_starts + TOTAL_OFFSET)
                        else:
                            dam_surface_wr = np.nan

                        # 産駒の平均賞金 (log変換)
                        mean_prize = merged_asof["cum_prize"].fillna(0).mean()
                        dam_prize_log = np.log1p(mean_prize) if mean_prize > 0 else np.nan

                        # breeder_strength: log(1 + unique breeder count)
                        offspring_rows = sanku[sanku["mnum"] == mnum]
                        if breeder_col in offspring_rows.columns:
                            unique_breeders = offspring_rows[breeder_col].dropna().nunique()
                            breeder_strength = np.log1p(unique_breeders)
                        else:
                            breeder_strength = np.nan

                        feats = {
                            "dam_wr": dam_wr,
                            "dam_surface_wr": dam_surface_wr,
                            "dam_prize_log": dam_prize_log,
                            "breeder_strength": breeder_strength,
                        }

                dam_features_cache[cache_key] = feats

            for c in FEATURE_COLS:
                result.loc[idx, c] = feats.get(c, np.nan)

        return result[["race_id", "umaban"] + FEATURE_COLS]

    def _compute_fallback(
        self,
        entry_df: pd.DataFrame,
        sanku: pd.DataFrame,
        career: pd.DataFrame,
    ) -> pd.DataFrame:
        """PIT-safe でないフォールバック (race_date/race_id が無い場合)。

        従来の last() ベースの集計。PIT 違反あり。
        """
        kettonum_to_mnum = sanku.set_index("kettonum")["mnum"]
        entry_mnums = entry_df["kettonum"].map(kettonum_to_mnum)
        unique_mnums = entry_mnums.dropna().unique()

        if len(unique_mnums) == 0:
            return entry_df[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        breeder_col = "breedercode" if "breedercode" in sanku.columns else "breeder_code"
        dam_features: dict[str, dict[str, float]] = {}
        career_latest = career.groupby("kettonum", observed=True).last()

        for mnum in unique_mnums:
            offspring = sanku[sanku["mnum"] == mnum]
            offspring_kettonums = offspring["kettonum"].values
            offspring_career = career_latest[
                career_latest.index.isin(offspring_kettonums)
            ]

            if offspring_career.empty:
                dam_features[mnum] = {c: np.nan for c in FEATURE_COLS}
                continue

            total_wins = offspring_career["cum_wins"].sum()
            total_starts = offspring_career["cum_starts"].sum()
            dam_wr = (total_wins + ALPHA_PRIOR) / (total_starts + TOTAL_OFFSET)

            total_turf_wins = offspring_career.get("cum_turf_wins", pd.Series(0)).sum()
            total_turf_starts = offspring_career.get("cum_turf_starts", pd.Series(0)).sum()
            dam_surface_wr = (
                (total_turf_wins + ALPHA_PRIOR) / (total_turf_starts + TOTAL_OFFSET)
                if total_turf_starts > 0
                else np.nan
            )

            mean_prize = offspring_career.get("cum_prize", pd.Series([0])).mean()
            dam_prize_log = np.log1p(mean_prize) if mean_prize > 0 else np.nan

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

        result = entry_df[["race_id", "umaban", "kettonum"]].copy()
        for col in FEATURE_COLS:
            result[col] = entry_mnums.map(
                lambda m, c=col: dam_features.get(m, {}).get(c, np.nan)
                if pd.notna(m)
                else np.nan
            )

        return result[["race_id", "umaban"] + FEATURE_COLS]
