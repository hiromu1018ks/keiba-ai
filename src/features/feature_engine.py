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
    ) -> pd.DataFrame:
        """バッチ特徴量生成（TrainingPipelineV5 から呼ばれる）

        Args:
            race_df: レースメタデータ (load_races() の出力)
            entry_df: 出走馬データ (load_entries_with_results() の出力)
            odds_df: オッズスナップショット (load_odds_snapshots() の出力)
            odds_ts_df: オッズ時系列データ (省略可、B-3 で compute_odds_dynamics() に渡す)

        Returns:
            全馬の特徴量を含むDataFrame (1行 = 1馬)
        """
        # 1. race + entry を race_id で結合
        #    race_df は同一 race_id が複数行ある場合があるため dedup
        race_dedup = race_df.drop_duplicates(subset=["race_id"])
        df = pd.merge(race_dedup, entry_df, on="race_id", how="inner")

        # 2. odds を (race_id, umaban) で結合
        df = pd.merge(df, odds_df, on=["race_id", "umaban"], how="left")

        # 3. 障害レース除外
        if self._exclude_steeple:
            df = df[df["track_cd"] < 51]

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
        # 1. Race → DataFrame
        race_data = {
            "race_id": race.race_id,
            "surface": race.surface.value,
            "distance_band": race.distance_band,
            "track_cd": race.track_cd,
            "distance": race.distance,
            "baba_cd": race.baba_cd,
            "grade_cd": race.grade_cd,
            "field_size": race.field_size,
            "tenko_cd": race.tenko_cd,
            "syubetu_cd": race.syubetu_cd,
            "jyoken_cd": race.jyoken_cd,
        }
        race_row = pd.DataFrame([race_data])

        # 2. list[Entry] → DataFrame
        entry_rows = []
        for e in entries:
            entry_rows.append(
                {
                    "race_id": race.race_id,
                    "umaban": e.umaban,
                    "ketto_num": e.ketto_num,
                    "finish_pos": e.finish_pos,
                    "win_odds": e.win_odds_actual,
                    "ninki": e.popularity_rank,
                    "ba_taijyu": e.ba_taijyu,
                    "zogen_fugo": e.zogen_fugo,
                    "zogen_sa": e.zogen_sa,
                    "kisyu_code": e.kisyu_code,
                    "chokyosi_code": e.chokyosi_code,
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
        """DB列名 → FEATURE_COLS 名へのマッピング

        rename: distance_band→distance_bin, baba_cd→track_condition_code, grade_cd→grade_code
        copy: ninki→popularity_rank, surface→surface_key (両方を保持)
        """
        rename_map: dict[str, str] = {}
        if "distance_band" in df.columns:
            rename_map["distance_band"] = "distance_bin"
        if "baba_cd" in df.columns:
            rename_map["baba_cd"] = "track_condition_code"
        if "grade_cd" in df.columns:
            rename_map["grade_cd"] = "grade_code"

        df = df.rename(columns=rename_map)

        # ninki → popularity_rank (ninki は別用途でも使うためコピー)
        if "ninki" in df.columns:
            df["popularity_rank"] = df["ninki"]

        # surface_key (downstream SubModelManager フィルタ用)
        if "surface" in df.columns:
            df["surface_key"] = df["surface"]

        return df
