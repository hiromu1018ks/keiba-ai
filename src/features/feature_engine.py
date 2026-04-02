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

import pandas as pd

from domain.models import Entry, Race


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
            "datakubun", "harontimel3", "harontimel4", "jyocd", "kaiji",
            "makedate", "monthday", "nichiji", "race_date", "racenum",
            "recordspec", "recordupkubun", "year",
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

        # 3. 障害レース除外
        if self._exclude_steeple:
            df = df[df["trackcd"] < 51]

        # 4. 基本特徴量のマッピング
        df = self._map_basic_features(df)

        # 5. サブモジュールの特徴量計算
        from features.intra_race_features import compute_intra_race_features

        df = compute_intra_race_features(df)

        from features.odds_dynamics_features import compute_odds_dynamics

        df = compute_odds_dynamics(df, odds_ts_df)

        from features.market_bias_features import compute_market_bias

        df = compute_market_bias(df)

        from features.race_difficulty_model import compute_difficulty_score

        df = compute_difficulty_score(df)

        # Group B: 血統特徴量
        if store is not None:
            from features.bloodline_features import BloodlineFeatures

            bloodline = BloodlineFeatures(store)
            bloodline_df = bloodline.compute(df)
            df = pd.merge(df, bloodline_df, on=["race_id", "umaban"], how="left")

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
            df["grade_code"] = df["gradecd"]

        # field_size: syussotosu → field_size コピー
        if "syussotosu" in df.columns and "field_size" not in df.columns:
            df["field_size"] = df["syussotosu"]

        # popularity_rank: ninki → popularity_rank コピー
        if "ninki" in df.columns and "popularity_rank" not in df.columns:
            df["popularity_rank"] = df["ninki"]

        # running_style: kyakusitukubun → running_style (int変換)
        if "kyakusitukubun" in df.columns:
            df["running_style"] = df["kyakusitukubun"].fillna(0).astype(int)

        return df
