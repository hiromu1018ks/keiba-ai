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

        sanku["kettonum"] = sanku["kettonum"].astype(str)
        sanku["mnum"] = sanku["mnum"].astype(str)

        # エントリー馬の kettonum から dam MNum を特定
        kettonum_to_mnum = (
            sanku.drop_duplicates("kettonum", keep="last").set_index("kettonum")["mnum"]
        )
        result = entry_df[["race_id", "umaban", "kettonum"]].copy()
        result["mnum"] = result["kettonum"].astype(str).map(kettonum_to_mnum)

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

        return self._compute_pit_vectorized(result, sanku, career, breeder_col)

    @staticmethod
    def _compute_pit_vectorized(
        result: pd.DataFrame,
        sanku: pd.DataFrame,
        career: pd.DataFrame,
        breeder_col: str,
    ) -> pd.DataFrame:
        """母ごとの産駒累積値を差分集計し、全対象日へ一括でasof結合する。"""
        output_cols = ["race_id", "umaban"] + FEATURE_COLS
        valid_mnums = result["mnum"].dropna().unique()
        offspring = sanku.loc[
            sanku["mnum"].isin(valid_mnums), ["kettonum", "mnum"]
        ].drop_duplicates("kettonum", keep="last")

        career_work = career.copy()
        career_work["kettonum"] = career_work["kettonum"].astype(str)
        career_work["race_date"] = pd.to_datetime(career_work["race_date"], errors="coerce")
        career_work = career_work.merge(offspring, on="kettonum", how="inner")
        if career_work.empty:
            return result[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        cum_cols = [
            "cum_wins",
            "cum_starts",
            "cum_turf_wins",
            "cum_turf_starts",
            "cum_prize",
        ]
        for col in cum_cols:
            career_work[col] = pd.to_numeric(career_work[col], errors="coerce").fillna(0.0)

        # 同一馬・同一日の最終スナップショットを採用し、累積値を日次差分へ変換する。
        career_work = (
            career_work.sort_values(["kettonum", "race_date", "race_id"])
            .drop_duplicates(["kettonum", "race_id"], keep="last")
            .drop_duplicates(["kettonum", "race_date"], keep="last")
            .sort_values(["kettonum", "race_date"])
        )
        daily_deltas = career_work.groupby("kettonum", observed=True)[cum_cols].diff()
        first_for_horse = ~career_work["kettonum"].duplicated()
        daily_deltas.loc[first_for_horse, cum_cols] = career_work.loc[
            first_for_horse, cum_cols
        ].to_numpy()
        daily_deltas[["mnum", "race_date"]] = career_work[["mnum", "race_date"]]

        dam_history = (
            daily_deltas.groupby(["mnum", "race_date"], observed=True)[cum_cols]
            .sum()
            .sort_index()
            .groupby(level="mnum", observed=True)
            .cumsum()
            .reset_index()
        )

        target_pairs = (
            result.loc[
                result["mnum"].notna() & result["race_date"].notna(),
                ["mnum", "race_date"],
            ]
            .drop_duplicates()
            .copy()
        )
        target_pairs["race_date"] = pd.to_datetime(target_pairs["race_date"], errors="coerce")
        target_pairs = target_pairs.dropna(subset=["race_date"])
        if target_pairs.empty:
            return result[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        # merge_asof は結合キーの全体昇順を要求する。
        target_pairs = target_pairs.sort_values(["race_date", "mnum"])
        dam_history = dam_history.sort_values(["race_date", "mnum"])
        features = pd.merge_asof(
            target_pairs,
            dam_history,
            on="race_date",
            by="mnum",
            direction="backward",
        )

        offspring_count = offspring.groupby("mnum", observed=True)["kettonum"].nunique()
        features["offspring_count"] = features["mnum"].map(offspring_count)
        features["dam_wr"] = np.where(
            features["cum_starts"] > 0,
            (features["cum_wins"] + ALPHA_PRIOR)
            / (features["cum_starts"] + TOTAL_OFFSET),
            np.nan,
        )
        features["dam_surface_wr"] = np.where(
            features["cum_turf_starts"] > 0,
            (features["cum_turf_wins"] + ALPHA_PRIOR)
            / (features["cum_turf_starts"] + TOTAL_OFFSET),
            np.nan,
        )
        mean_prize = features["cum_prize"] / features["offspring_count"]
        features["dam_prize_log"] = np.where(
            mean_prize > 0, np.log1p(mean_prize), np.nan
        )

        if breeder_col in sanku.columns:
            breeder_counts = sanku.groupby("mnum", observed=True)[breeder_col].nunique()
            features["breeder_strength"] = np.log1p(features["mnum"].map(breeder_counts))
        else:
            features["breeder_strength"] = np.nan

        result = result.merge(
            features[["mnum", "race_date"] + FEATURE_COLS],
            on=["mnum", "race_date"],
            how="left",
        )
        return result[output_cols]

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
