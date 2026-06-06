"""特徴量マニフェスト・状態・ビルド結果の dataclass 定義 (D-03).

FeatureManifest: モデル入力列の名前・順序・dtype・バージョンの SHA256 ハッシュ。
FeatureState: 推論時に必要な学習期間統計 (track_stats, track_month_stats)。
FeatureBuildResult: build_for_training/build_for_inference の戻り値。

ハッシュ対象はモデル入力列のみ。race_id / ターゲット列 / POST_RACE / 構築日時 / データ値は除外。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from domain.types import POST_RACE_COLS

# マニフェストハッシュ対象外の列 (race_id, ターゲット, POST_RACE)
_EXCLUDED_FROM_MANIFEST: frozenset[str] = frozenset(
    {"race_id", "kakuteijyuni", "confirmed_odds"} | set(POST_RACE_COLS)
)


@dataclass(frozen=True)
class FeatureManifest:
    """特徴量カラムの名前・順序・dtype・バージョンを保持する不変 dataclass。

    compute_hash() はモデル入力列のみを対象とし、race_id / ターゲット /
    POST_RACE / 構築日時を除外する。
    """

    column_names: tuple[str, ...]
    column_dtypes: tuple[str, ...]
    feature_version: str

    def compute_hash(self) -> str:
        """SHA256 ハッシュを計算。

        Returns:
            64文字の16進数文字列。
        """
        payload = json.dumps(
            {
                "columns": list(self.column_names),
                "dtypes": list(self.column_dtypes),
                "version": self.feature_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, version: str) -> FeatureManifest:
        """DataFrame からマニフェストを生成。

        POST_RACE_COLS / race_id / kakuteijyuni / confirmed_odds を除外し、
        カラム名をアルファベット順にソートして決定性を担保する。

        Args:
            df: 特徴量 DataFrame。
            version: 特徴量定義バージョン。

        Returns:
            FeatureManifest インスタンス。
        """
        model_cols = sorted(
            c for c in df.columns if c not in _EXCLUDED_FROM_MANIFEST
        )
        dtypes = tuple(str(df[c].dtype) for c in model_cols)
        return cls(
            column_names=tuple(model_cols),
            column_dtypes=dtypes,
            feature_version=version,
        )


@dataclass(frozen=True)
class FeatureState:
    """推論時に必要な学習期間統計を保持する不変 dataclass (D-04)。

    from_submodel_set() で SubmodelSet から生成。track_stats が None の場合は fail-fast。
    """

    track_stats: dict[str, dict[str, float]]
    track_month_stats: dict[str, dict[str, float]]
    feature_version: str

    @classmethod
    def from_submodel_set(cls, submodel: Any, version: str) -> FeatureState:
        """SubmodelSet から FeatureState を生成 (D-04)。

        Args:
            submodel: SubmodelSet インスタンス。track_stats / track_month_stats を持つ。
            version: 特徴量定義バージョン。

        Returns:
            FeatureState インスタンス。

        Raises:
            ValueError: submodel.track_stats が None の場合 (Phase 51 TRN-04 要件)。
        """
        if submodel.track_stats is None:
            raise ValueError(
                "SubmodelSet.track_stats is None — "
                "train with Phase 51 TRN-04 track_stats persistence enabled"
            )
        return cls(
            track_stats=submodel.track_stats,
            track_month_stats=submodel.track_month_stats or {},
            feature_version=version,
        )

    def compute_hash(self) -> str:
        """SHA256 ハッシュを計算。"""
        payload = json.dumps(
            {
                "track_stats": self.track_stats,
                "track_month_stats": self.track_month_stats,
                "feature_version": self.feature_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureBuildResult:
    """build_for_training / build_for_inference の戻り値 (D-03)。

    frame は __eq__ から除外し manifest でのみ等価性を判定する。
    """

    frame: pd.DataFrame
    manifest: FeatureManifest

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureBuildResult):
            return NotImplemented
        return self.manifest == other.manifest

    def __hash__(self) -> int:
        return hash(self.manifest)
