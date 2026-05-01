"""特徴量エンジン v5.3 — メインオーケストレータ

カテゴリ:
  A: 馬の能力 (Stage1出力、本モジュールでは計算しない)
  B: レース内相対値 (intra_race_features.py)
  C: オッズ変化率 (odds_dynamics_features.py)
  D: 市場歪み (market_bias_features.py)
  E: 情報非対称性 (info_asymmetry_features.py, race_difficulty_model.py)
  F: 距離帯・馬場 one-hot (SubModelManager が担当)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.models import Entry, Race
from utils.timing import TimingContext

_GRADE_LEVEL_MAP: dict[str, float] = {
    "A": 8.0,
    "B": 7.0,
    "C": 6.0,
    "D": 5.0,
    "E": 4.0,
}


def _compute_class_level(
    grade_code: pd.Series | None,
    jyoken_code: pd.Series | None,
) -> pd.Series:
    if grade_code is None and jyoken_code is None:
        return pd.Series(dtype=float)

    if grade_code is not None:
        grade_series = grade_code.fillna("").astype(str).str.strip()
        grade_level = grade_series.map(_GRADE_LEVEL_MAP)
    else:
        index = jyoken_code.index if jyoken_code is not None else None
        grade_level = pd.Series(np.nan, index=index)

    if jyoken_code is not None:
        jyoken_level = pd.to_numeric(jyoken_code, errors="coerce")
    else:
        jyoken_level = pd.Series(np.nan, index=grade_level.index)

    return grade_level.fillna(jyoken_level)


def _compute_popularity_rank_from_tanodds(df: pd.DataFrame) -> pd.Series:
    if "tanodds" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    tanodds = pd.to_numeric(df["tanodds"], errors="coerce")
    valid_mask = tanodds.notna() & (tanodds > 0)
    popularity_rank = pd.Series(np.nan, index=df.index, dtype=float)
    if not valid_mask.any():
        return popularity_rank

    if "race_id" in df.columns:
        popularity_rank.loc[valid_mask] = tanodds.loc[valid_mask].groupby(
            df.loc[valid_mask, "race_id"],
            observed=True,
        ).rank(method="first", ascending=True)
    else:
        popularity_rank.loc[valid_mask] = tanodds.loc[valid_mask].rank(
            method="first",
            ascending=True,
        )
    return popularity_rank


class FeatureEngine:
    """特徴量エンジンのメインオーケストレータ

    build_all(): バッチ学習用 — 3つのDataFrameをマージして全特徴量を計算
    build_features(): 推論用 — Race + list[Entry] から単レース特徴量を計算
    """

    def __init__(self, exclude_steeple: bool = True) -> None:
        self._exclude_steeple = exclude_steeple

    def build_all(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        odds_ts_df: pd.DataFrame | None = None,
        store: object | None = None,
    ) -> pd.DataFrame:
        """バッチ特徴量生成（TrainingPipelineV5 から呼ばれる）

        Args:
            race_df: レースメタデータ (load_races() の出力)
            entry_df: 出走馬データ (load_entries_with_results() の出力)
            odds_df: オッズスナップショット (load_odds_snapshots() の出力)
            odds_ts_df: オッズ時系列データ (省略可、B-3 で compute_odds_dynamics() に渡す)
            store: ParquetStore (省略可、血統特徴量で使用)

        Returns:
            全馬の特徴量を含むDataFrame (1行 = 1馬)
        """
        # 1. race + entry を race_id で結合
        #    race_df は同一 race_id が複数行ある場合があるため dedup
        #    entries 側の共有列を除外して _x/_y サフィックスを防止
        #    (race_date, year 等の識別列は race_df から取得;
        #     harontimel3/4 は HorseHistoryFeatures が self._entry_df から直接参照)
        _race_entry_shared = [
            "datakubun",
            "harontimel3",
            "harontimel4",
            "jyocd",
            "kaiji",
            "makedate",
            "monthday",
            "nichiji",
            "race_date",
            "racenum",
            "recordspec",
            "recordupkubun",
            "year",
        ]
        entry_subset = entry_df.drop(
            columns=[c for c in _race_entry_shared if c in entry_df.columns]
        )
        race_dedup = race_df.drop_duplicates(subset=["race_id"])
        df = pd.merge(race_dedup, entry_subset, on="race_id", how="inner")

        # 2. odds を (race_id, umaban) で結合
        #    entries と odds_tanpuku は year, monthday, race_date 等の共有列があるため、
        #    不要な列を事前に除外して _x/_y サフィックスの発生を防止する
        odds_cols = ["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]
        df = pd.merge(df, odds_df[odds_cols], on=["race_id", "umaban"], how="left")

        # LEAK修正: entries.odds は確定オッズ (レース後)。特徴量計算では
        # tanodds (発走前スナップショット) を優先使用する。
        # 確定オッズは confirmed_odds に保存 (学習ターゲット用)
        if "odds" in df.columns:
            df["confirmed_odds"] = df["odds"].copy()
        if "odds" in df.columns and "tanodds" in df.columns:
            mask = (df["tanodds"] > 0) & df["tanodds"].notna()
            df.loc[mask, "odds"] = df.loc[mask, "tanodds"]

        # 3. 障害レース除外
        if self._exclude_steeple:
            df = df[df["trackcd"] < 51]

        # 4. 基本特徴量のマッピング
        df = self._map_basic_features(df)

        # 5. サブモジュールの特徴量計算
        from features.intra_race_features import compute_intra_race_features

        with TimingContext("build_all/intra_race"):
            df = compute_intra_race_features(df)

        from features.odds_dynamics_features import compute_odds_dynamics

        with TimingContext("build_all/odds_dynamics"):
            df = compute_odds_dynamics(df, odds_ts_df)

        from features.market_bias_features import compute_market_bias

        with TimingContext("build_all/market_bias"):
            df = compute_market_bias(df)

        from features.market_bias_features import compute_flb_slope

        with TimingContext("build_all/flb_slope"):
            flb_result = compute_flb_slope(df)
            df = pd.concat([df, flb_result], axis=1)

        from features.race_difficulty_model import compute_difficulty_score

        with TimingContext("build_all/difficulty"):
            df = compute_difficulty_score(df)

        # Group B: 血統特徴量
        if store is not None:
            with TimingContext("build_all/bloodline"):
                from features.bloodline_features import BloodlineFeatures

                bloodline = BloodlineFeatures(store)
                bloodline_df = bloodline.compute(df)
                df = pd.merge(df, bloodline_df, on=["race_id", "umaban"], how="left")

        # NOTE: Group C (ペース適性特徴量) と Group D (コース別適性特徴量) は
        # TrainingPipeline._train_submodel() で計算されるため、ここではプレースホルダーなし
        # pace_aptitude, front_pace_wr, closing_pace_wr, course_wr, course_distance_wr

        # NOTE: Group E (interaction features) は HorseHistoryFeatures 後に呼ぶこと。
        # kyakusitu_cd が必要なため、build_all では実行しない。
        # _train_submodel / BacktestEngine で hist_df merge 後に呼び出す。

        return df

    def build_features(
        self,
        race: Race,
        entries: list[Entry],
        odds_snapshot: pd.DataFrame | None = None,
        odds_ts: pd.DataFrame | None = None,
        snap_minutes: int | None = None,
    ) -> pd.DataFrame:
        """単レース推論用特徴量生成（BettingOrchestrator から呼ばれる）

        設計書 §12 呼び出し: self.feat_engine.build_features(race, entries, snap_minutes=10)

        Args:
            race: レース情報ドメインモデル
            entries: 出走馬ドメインモデルのリスト
            odds_snapshot: 現在のオッズスナップショット
            odds_ts: オッズ時系列データ (省略可、B-3 で使用)
            snap_minutes: オッズスナップショットの取得分前 (省略可、B-3 で使用)

        Returns:
            全馬の特徴量を含むDataFrame (1行 = 1馬)
        """
        # 1. Race → DataFrame (生カラム名)
        race_data = {
            "race_id": race.race_id,
            "trackcd": race.track_cd,
            "kyori": race.distance,
            "gradecd": race.grade_cd,
            "syussotosu": race.field_size,
            "tenkocd": race.tenko_cd,
            "syubetucd": race.syubetu_cd,
            "jyokencd1": race.jyoken_cd,
            "track_condition_code": race.baba_cd,
        }
        race_row = pd.DataFrame([race_data])

        # 2. list[Entry] → DataFrame (生カラム名)
        entry_rows = []
        for e in entries:
            entry_rows.append(
                {
                    "race_id": race.race_id,
                    "umaban": e.umaban,
                    "kettonum": e.ketto_num,
                    "kakuteijyuni": e.finish_pos,
                    "odds": e.win_odds_actual,
                    "ninki": e.popularity_rank,
                    "bataijyu": e.ba_taijyu,
                    "kisyucode": e.kisyu_code,
                    "chokyosicode": e.chokyosi_code,
                }
            )
        entry_df = pd.DataFrame(entry_rows)

        # 3. 結合
        df = pd.merge(race_row, entry_df, on="race_id", how="inner")

        # 4. オッズ結合
        if odds_snapshot is not None:
            df = pd.merge(df, odds_snapshot, on=["race_id", "umaban"], how="left")
            # 推論パス: entries.odds は 0 (未発走)、tanodds を使用
            if "odds" in df.columns and "tanodds" in df.columns:
                mask = (df["tanodds"] > 0) & df["tanodds"].notna()
                df.loc[mask, "odds"] = df.loc[mask, "tanodds"]

        # 5. 基本特徴量マッピング
        df = self._map_basic_features(df)

        # 6. サブモジュールの特徴量計算（推論用 — hist特徴量は除く）

        return df

    def _map_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生カラム名 → ML特徴量名へのマッピング

        ETLが型変換・surface・track_condition_codeを処理済みのため、
        ML固有の別名のみをここで設定する。
        """
        # distance_bin: kyori + surface から計算 (ETLには含まれない)
        if "distance_bin" not in df.columns and "kyori" in df.columns and "surface" in df.columns:
            is_turf = df["surface"] == "turf"
            dist = df["kyori"]
            df["distance_bin"] = "unknown"
            # Turf: sprint(<=1400), mile(<=1700), intermediate(<=2100), long(>2100)
            df.loc[is_turf & (dist > 2100), "distance_bin"] = "long"
            df.loc[is_turf & (dist <= 2100), "distance_bin"] = "intermediate"
            df.loc[is_turf & (dist <= 1700), "distance_bin"] = "mile"
            df.loc[is_turf & (dist <= 1400), "distance_bin"] = "sprint"
            # Dirt: sprint(<=1400), mile(<=1700), intermediate(>1700)
            df.loc[~is_turf & (dist > 1700), "distance_bin"] = "intermediate"
            df.loc[~is_turf & (dist <= 1700), "distance_bin"] = "mile"
            df.loc[~is_turf & (dist <= 1400), "distance_bin"] = "sprint"

        # track_condition_code: ETLが計算済み。推論パス用のガードのみ
        # (build_features() では race.baba_cd から直接渡される)

        # grade_code: gradecd → grade_code コピー
        if "gradecd" in df.columns and "grade_code" not in df.columns:
            df["grade_code"] = df["gradecd"].replace("", "X")  # X=未格付け
        if "class_level_current" not in df.columns:
            df["class_level_current"] = _compute_class_level(
                df["grade_code"] if "grade_code" in df.columns else df.get("gradecd"),
                df["jyokencd1"] if "jyokencd1" in df.columns else None,
            )

        # field_size: syussotosu → field_size コピー
        # 未発走レースでは syussotosu=0 のため、race_id ごとの行数で補完
        if "syussotosu" in df.columns and "field_size" not in df.columns:
            df["field_size"] = df["syussotosu"]
            if (df["field_size"] == 0).any():
                actual = df.groupby("race_id").size()
                df["field_size"] = (
                    df["race_id"].map(actual).fillna(df["field_size"]).astype("Int64")
                )

        # popularity_rank は pre-post tanodds から再計算して train/test の定義を揃える。
        # tanodds が欠損した馬のみ tanninki、最後に ninki へフォールバックする。
        if "popularity_rank" not in df.columns:
            df["popularity_rank_fallback_used"] = 0.0
            df["popularity_rank"] = _compute_popularity_rank_from_tanodds(df)
            invalid_mask = (df["popularity_rank"] == 0) | df["popularity_rank"].isna()
            if invalid_mask.any():
                import logging

                fallback_mask = invalid_mask.copy()
                if "tanninki" in df.columns:
                    tanninki_values = pd.to_numeric(df["tanninki"], errors="coerce")
                    usable_tanninki = (
                        fallback_mask & tanninki_values.notna() & (tanninki_values > 0)
                    )
                    df.loc[usable_tanninki, "popularity_rank"] = tanninki_values.loc[
                        usable_tanninki
                    ]
                    fallback_mask = fallback_mask & ~usable_tanninki
                if "ninki" in df.columns:
                    fallback_values = pd.to_numeric(df["ninki"], errors="coerce")
                    usable_fallback = (
                        fallback_mask
                        & fallback_values.notna()
                        & (fallback_values > 0)
                    )
                    if usable_fallback.any():
                        logging.getLogger(__name__).warning(
                            "popularity_rank fell back to ninki for %d horses",
                            int(usable_fallback.sum()),
                        )
                    df.loc[usable_fallback, "popularity_rank"] = fallback_values.loc[
                        usable_fallback
                    ]
                    df.loc[usable_fallback, "popularity_rank_fallback_used"] = 1.0
                    fallback_mask = fallback_mask & ~usable_fallback
                if fallback_mask.any():
                    logging.getLogger(__name__).warning(
                        "popularity_rank missing for %d horses after "
                        "tanodds/tanninki/ninki fallback",
                        int(fallback_mask.sum()),
                    )
                    df.loc[fallback_mask, "popularity_rank"] = float("nan")
        elif "popularity_rank_fallback_used" not in df.columns:
            df["popularity_rank_fallback_used"] = 0.0

        # draw / post-position 特徴量
        if "umaban" in df.columns and "field_size" in df.columns:
            field_size = pd.to_numeric(df["field_size"], errors="coerce")
            umaban = pd.to_numeric(df["umaban"], errors="coerce")
            df["draw_ratio"] = np.where(
                field_size > 1,
                (umaban - 1.0) / (field_size - 1.0),
                float("nan"),
            )
        else:
            df["draw_ratio"] = float("nan")

        if "wakuban" in df.columns:
            df["frame_number"] = pd.to_numeric(df["wakuban"], errors="coerce")
        else:
            df["frame_number"] = float("nan")

        if "blinker" in df.columns:
            blinker = pd.to_numeric(df["blinker"], errors="coerce")
            df["blinker_on"] = np.where(blinker.fillna(0) > 0, 1.0, 0.0)
        else:
            df["blinker_on"] = float("nan")

        # A2: weight_change_zone — 体重変化カテゴリ (zogen_sa ベース、数値エンコード)
        if "zogen_sa" in df.columns:
            zogen = df["zogen_sa"].astype(float)
            zone = pd.Series(1, index=df.index)  # default: stable (-4 ~ +4)
            zone[(zogen >= 4) & (zogen <= 12)] = 2  # golden
            zone[(zogen >= -14) & (zogen < -4)] = 0  # caution (下側)
            zone[(zogen > 12) & (zogen <= 14)] = 0  # caution (上側)
            zone[(zogen < -14) | (zogen > 14)] = -1  # danger
            df["weight_change_zone"] = zone.astype(float)
        else:
            df["weight_change_zone"] = float("nan")

        # A3: weight_change_ratio — 体重変化率 (zogen_sa / bataijyu)
        if "zogen_sa" in df.columns and "bataijyu" in df.columns:
            weight = df["bataijyu"].astype(float)
            zogen = df["zogen_sa"].astype(float)
            # 変化率 = 増減差 / 馬体重（パーセンテージ）
            # 馬体重が0またはNaNの場合はNaNにする
            df["weight_change_ratio"] = np.where(
                (weight > 0) & weight.notna(),
                zogen / weight * 100,  # パーセンテージに変換
                float("nan")
            )
        else:
            df["weight_change_ratio"] = float("nan")

        return df
