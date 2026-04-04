"""芝/ダート2分割モデル管理 + 距離帯one-hot特徴量生成 (§6)"""

from __future__ import annotations

import pandas as pd

from domain.models import Race


class SubModelManager:
    """
    v5.0: 芝/ダートの2分割に縮小。
    距離帯・馬場状態はモデル分割ではなく特徴量として対応する。
    """

    VALID_KEYS: list[str] = ["turf", "dirt"]
    MIN_SAMPLES: int = 20_000

    def get_key(self, race: Race) -> str:
        """Raceオブジェクトからサブモデルキーを取得"""
        return race.surface.value

    def get_models(self, race: Race, models: dict[str, object]) -> object:
        """指定レースのサブモデルを返す"""
        key = self.get_key(race)
        return models[key]

    def should_split_further(
        self,
        key: str,
        condition: str,
        sample_count: int,
    ) -> bool:
        """将来的にサブモデルを追加するかどうかの判定"""
        return sample_count >= self.MIN_SAMPLES

    def add_distance_band_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        距離帯をone-hot特徴量として追加する。
        モデル分割ではなく特徴量で吸収するため、少サンプルでも適切に扱える。
        """
        df = df.copy()
        # 芝距離帯
        df["is_turf_sprint"] = ((df["surface"] == "turf") & (df["kyori"] <= 1400)).astype(int)
        df["is_turf_mile"] = ((df["surface"] == "turf") & (df["kyori"].between(1401, 1700))).astype(
            int
        )
        df["is_turf_intermediate"] = (
            (df["surface"] == "turf") & (df["kyori"].between(1701, 2100))
        ).astype(int)
        df["is_turf_long"] = ((df["surface"] == "turf") & (df["kyori"] >= 2101)).astype(int)
        # ダート距離帯
        df["is_dirt_sprint"] = ((df["surface"] == "dirt") & (df["kyori"] <= 1400)).astype(int)
        df["is_dirt_mile"] = ((df["surface"] == "dirt") & (df["kyori"].between(1401, 1700))).astype(
            int
        )
        df["is_dirt_intermediate"] = ((df["surface"] == "dirt") & (df["kyori"] >= 1701)).astype(int)
        # 馬場状態 (track_condition_code: 1=良, 2=稍重, 3=重, 4=不良)
        df["is_good_track"] = df["track_condition_code"].isin([1, 2]).astype(int)
        df["is_soft_track"] = df["track_condition_code"].isin([3, 4]).astype(int)
        return df
