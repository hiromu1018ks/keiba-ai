"""FeatureManifest / FeatureState / FeatureBuildResult / PITModuleRegistry の単体テスト。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pandas as pd
import pytest

from domain.types import POST_RACE_COLS
from features.feature_manifest import (
    FeatureBuildResult,
    FeatureManifest,
    FeatureState,
)
from features.pit_registry import PITModuleRegistry


# ---------------------------------------------------------------------------
# FeatureManifest
# ---------------------------------------------------------------------------


class TestFeatureManifest:
    """FeatureManifest.compute_hash() と from_dataframe() のテスト。"""

    def test_compute_hash_deterministic(self) -> None:
        """同じ入力に対して同じハッシュを返す。"""
        manifest = FeatureManifest(
            column_names=("a", "b", "c"),
            column_dtypes=("float64", "int64", "float64"),
            feature_version="1.0",
        )
        assert manifest.compute_hash() == manifest.compute_hash()
        assert len(manifest.compute_hash()) == 64

    def test_compute_hash_differs_on_column_order(self) -> None:
        """カラム順序が異なるとハッシュが変わる (sort_keys=True だが
        column_names は tuple 順でシリアライズされる)。"""
        m1 = FeatureManifest(
            column_names=("a", "b"),
            column_dtypes=("float64", "int64"),
            feature_version="1.0",
        )
        m2 = FeatureManifest(
            column_names=("b", "a"),
            column_dtypes=("int64", "float64"),
            feature_version="1.0",
        )
        assert m1.compute_hash() != m2.compute_hash()

    def test_compute_hash_differs_on_version(self) -> None:
        """バージョンが異なるとハッシュが変わる。"""
        m1 = FeatureManifest(
            column_names=("a",),
            column_dtypes=("float64",),
            feature_version="1.0",
        )
        m2 = FeatureManifest(
            column_names=("a",),
            column_dtypes=("float64",),
            feature_version="2.0",
        )
        assert m1.compute_hash() != m2.compute_hash()

    def test_from_dataframe_excludes_post_race_and_id(self) -> None:
        """from_dataframe が POST_RACE_COLS / race_id / ターゲット列を除外する。"""
        df = pd.DataFrame(
            {
                "race_id": ["r1", "r2"],
                "kakuteijyuni": [1, 2],
                "confirmed_odds": [3.0, 5.0],
                "feature_a": [0.1, 0.2],
                "feature_b": [10, 20],
                **{col: [0, 0] for col in POST_RACE_COLS[:3]},  # 代表的な POST_RACE 列
            }
        )
        manifest = FeatureManifest.from_dataframe(df, version="1.0")
        assert "race_id" not in manifest.column_names
        assert "kakuteijyuni" not in manifest.column_names
        assert "confirmed_odds" not in manifest.column_names
        for col in POST_RACE_COLS[:3]:
            assert col not in manifest.column_names
        # モデル入力列のみ含む
        assert "feature_a" in manifest.column_names
        assert "feature_b" in manifest.column_names

    def test_from_dataframe_sorts_columns(self) -> None:
        """from_dataframe がカラム名をアルファベット順にソートする。"""
        df = pd.DataFrame({"z_col": [1], "a_col": [2], "m_col": [3]})
        manifest = FeatureManifest.from_dataframe(df, version="1.0")
        assert manifest.column_names == ("a_col", "m_col", "z_col")


# ---------------------------------------------------------------------------
# FeatureState
# ---------------------------------------------------------------------------


class TestFeatureState:
    """FeatureState.from_submodel_set() のテスト。"""

    def test_raises_on_none_track_stats(self) -> None:
        """track_stats が None の場合に ValueError を送出する。"""
        submodel = MagicMock()
        submodel.track_stats = None
        with pytest.raises(ValueError, match="TRN-04"):
            FeatureState.from_submodel_set(submodel, version="1.0")

    def test_succeeds_with_track_stats(self) -> None:
        """track_stats が存在する場合に正常に生成する。"""
        submodel = MagicMock()
        submodel.track_stats = {"track_01": {"mean": 1.0, "std": 0.5}}
        submodel.track_month_stats = {"track_01_06": {"mean": 1.2}}
        state = FeatureState.from_submodel_set(submodel, version="1.0")
        assert state.track_stats == {"track_01": {"mean": 1.0, "std": 0.5}}
        assert state.track_month_stats == {"track_01_06": {"mean": 1.2}}
        assert state.feature_version == "1.0"

    def test_default_empty_track_month_stats(self) -> None:
        """track_month_stats が None の場合に空 dict を設定する。"""
        submodel = MagicMock()
        submodel.track_stats = {"t": {"m": 1.0}}
        submodel.track_month_stats = None
        state = FeatureState.from_submodel_set(submodel, version="1.0")
        assert state.track_month_stats == {}

    def test_compute_hash_deterministic(self) -> None:
        """同じ入力に対して同じハッシュを返す。"""
        state = FeatureState(
            track_stats={"t": {"m": 1.0}},
            track_month_stats={},
            feature_version="1.0",
        )
        assert state.compute_hash() == state.compute_hash()
        assert len(state.compute_hash()) == 64


# ---------------------------------------------------------------------------
# FeatureBuildResult
# ---------------------------------------------------------------------------


class TestFeatureBuildResult:
    """FeatureBuildResult の凍結不変性テスト。"""

    def test_frozen_immutability(self) -> None:
        """frozen dataclass への代入が FrozenInstanceError を送出する。"""
        manifest = FeatureManifest(
            column_names=("a",),
            column_dtypes=("float64",),
            feature_version="1.0",
        )
        result = FeatureBuildResult(
            frame=pd.DataFrame({"a": [1]}),
            manifest=manifest,
        )
        with pytest.raises(FrozenInstanceError):
            result.frame = pd.DataFrame()  # type: ignore[misc]

    def test_creation(self) -> None:
        """正常な生成とプロパティアクセス。"""
        manifest = FeatureManifest(
            column_names=("a", "b"),
            column_dtypes=("float64", "int64"),
            feature_version="1.0",
        )
        df = pd.DataFrame({"a": [1.0], "b": [2]})
        result = FeatureBuildResult(frame=df, manifest=manifest)
        assert len(result.frame) == 1
        assert result.manifest.feature_version == "1.0"


# ---------------------------------------------------------------------------
# PITModuleRegistry
# ---------------------------------------------------------------------------


class TestPITModuleRegistry:
    """PITModuleRegistry.verify_pit_compliance() のテスト。"""

    def test_returns_violations_for_future_dates(self) -> None:
        """未来日のデータに対して違反を検出する。"""
        registry = PITModuleRegistry()
        df = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2024-12-31", "2025-06-15"]),
            }
        )
        prediction_date = pd.Timestamp("2025-01-01")
        violations = registry.verify_pit_compliance(df, prediction_date)
        assert len(violations) > 0
        # HorseHistoryFeatures が違反に含まれる (race_date 列を使用)
        assert any("HorseHistoryFeatures" in v for v in violations)

    def test_returns_empty_for_compliant_dates(self) -> None:
        """全データが予測日より前の場合に違反なし。"""
        registry = PITModuleRegistry()
        df = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2024-01-01", "2024-06-30"]),
            }
        )
        prediction_date = pd.Timestamp("2025-01-01")
        violations = registry.verify_pit_compliance(df, prediction_date)
        assert violations == []

    def test_none_date_column_skipped(self) -> None:
        """max_date_column=None のモジュールは検証をスキップする。"""
        registry = PITModuleRegistry()
        # SireFeatures は max_date_column=None
        sire_contract = registry.contracts.get("SireFeatures")
        assert sire_contract is not None
        assert sire_contract.max_date_column is None

    def test_default_modules_registered(self) -> None:
        """13 の標準モジュールが事前登録されている。"""
        registry = PITModuleRegistry()
        expected_modules = [
            "HorseHistoryFeatures",
            "PaceAptitudeFeatures",
            "CourseFeatures",
            "SireFeatures",
            "DamPedigreeFeatures",
            "RecordFeatures",
            "TrackConditionFeatures",
            "InteractionFeatures",
            "MiningFeatures",
            "RelativeFeatures",
            "JockeyContextFeatures",
            "TrainerContextFeatures",
            "JockeyTrainerComboFeatures",
        ]
        for name in expected_modules:
            assert name in registry.contracts, f"{name} not registered"
        assert len(registry.contracts) == 13
