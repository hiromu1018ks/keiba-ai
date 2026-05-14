"""DamPedigreeFeatures の mock-based テスト (DB不要)"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from features.dam_pedigree_features import FEATURE_COLS, DamPedigreeFeatures


def _make_store(
    sanku: pd.DataFrame | None = None,
    career_stats: pd.DataFrame | None = None,
) -> MagicMock:
    """ParquetStore mock を構築"""
    store = MagicMock()

    def exists(category: str, name: str) -> bool:
        if category == "raw" and name == "sanku":
            return sanku is not None and not sanku.empty
        return False

    store.exists.side_effect = exists

    def read(category: str, name: str, **kwargs):  # type: ignore[misc]
        if category == "raw" and name == "sanku":
            return sanku if sanku is not None else pd.DataFrame()
        return pd.DataFrame()

    store.read.side_effect = read
    return store


def _make_career_stats_mock(career_stats: pd.DataFrame | None) -> MagicMock:
    """load_career_stats の戻り値をモックするパッチ用データ"""
    return career_stats


@pytest.fixture
def basic_sanku() -> pd.DataFrame:
    """基本sankuデータ: 3頭, 2頭が同じ母"""
    return pd.DataFrame(
        {
            "kettonum": ["20201001", "20202001", "20203001"],
            "mnum": ["D001", "D001", "D002"],
            "breedercode": ["BR01", "BR02", "BR01"],
        }
    )


@pytest.fixture
def basic_career() -> pd.DataFrame:
    """基本キャリア統計: 3頭の累積成績"""
    return pd.DataFrame(
        {
            "kettonum": ["20201001", "20202001", "20203001"],
            "cum_wins": [5, 3, 2],
            "cum_starts": [30, 20, 15],
            "cum_turf_wins": [3, 2, 1],
            "cum_turf_starts": [20, 15, 10],
            "cum_prize": [50000000, 30000000, 10000000],
        }
    )


@pytest.fixture
def basic_entry() -> pd.DataFrame:
    """基本エントリーデータ: 3頭"""
    return pd.DataFrame(
        {
            "race_id": ["R001", "R001", "R001"],
            "umaban": [1, 2, 3],
            "kettonum": ["20201001", "20202001", "20203001"],
        }
    )


class TestDamPedigreeFeatures:
    """DamPedigreeFeatures テストスイート"""

    def test_compute_returns_all_feature_columns(
        self, basic_sanku: pd.DataFrame, basic_career: pd.DataFrame, basic_entry: pd.DataFrame
    ) -> None:
        """Test 1: compute() は dam_wr, dam_surface_wr, dam_prize_log, breeder_strength を返す"""
        store = _make_store(sanku=basic_sanku)
        feat = DamPedigreeFeatures(store)
        feat._career_cache = basic_career

        result = feat.compute(basic_entry)

        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_dam_wr_uses_beta_smoothing(
        self, basic_sanku: pd.DataFrame, basic_career: pd.DataFrame, basic_entry: pd.DataFrame
    ) -> None:
        """Test 2: dam_wr は Beta(1,10) 平滑化を使用"""
        store = _make_store(sanku=basic_sanku)
        feat = DamPedigreeFeatures(store)
        feat._career_cache = basic_career

        result = feat.compute(basic_entry)

        # Dam D001: offspring A (5 wins/30 starts) + B (3 wins/20 starts)
        # total_wins = 8, total_starts = 50
        # Beta(1,10) smoothed = (8 + 1) / (50 + 11) = 9/61 ≈ 0.1475
        expected_d001 = (8 + 1) / (50 + 11)
        # Horse A (idx 0) と Horse B (idx 1) は同じ dam D001
        np.testing.assert_allclose(
            result.loc[[result.index[0], result.index[1]], "dam_wr"].values,
            expected_d001,
            rtol=1e-4,
        )

    def test_no_dam_data_returns_nan(
        self, basic_entry: pd.DataFrame
    ) -> None:
        """Test 3: 母の産駒データがない場合は NaN"""
        # sanku に存在しない kettonum のエントリ
        store = _make_store()  # empty store
        feat = DamPedigreeFeatures(store)
        feat._career_cache = pd.DataFrame()

        result = feat.compute(basic_entry)

        for col in FEATURE_COLS:
            assert result[col].isna().all(), f"{col} should be all NaN"

    def test_feature_cols_has_4_entries_no_duplicates(self) -> None:
        """Test 4: FEATURE_COLS はちょうど4要素で重複なし"""
        assert len(FEATURE_COLS) == 4
        assert len(set(FEATURE_COLS)) == 4
        expected = {"dam_wr", "dam_surface_wr", "dam_prize_log", "breeder_strength"}
        assert set(FEATURE_COLS) == expected

    def test_empty_store_returns_nan(
        self, basic_entry: pd.DataFrame
    ) -> None:
        """Test 5: 空 store は全特徴量 NaN"""
        store = _make_store()  # sanku=None → empty
        feat = DamPedigreeFeatures(store)
        feat._career_cache = pd.DataFrame()

        result = feat.compute(basic_entry)

        for col in FEATURE_COLS:
            assert result[col].isna().all()

    def test_breeder_strength_log_unique_breeders(
        self, basic_sanku: pd.DataFrame, basic_career: pd.DataFrame, basic_entry: pd.DataFrame
    ) -> None:
        """Test 6: breeder_strength = log(1 + unique breeder count)"""
        store = _make_store(sanku=basic_sanku)
        feat = DamPedigreeFeatures(store)
        feat._career_cache = basic_career

        result = feat.compute(basic_entry)

        # Dam D001: BR01, BR02 → 2 unique breeders → log(1+2) = log(3)
        # Dam D002: BR01 → 1 unique breeder → log(1+1) = log(2)
        # result は race_id + umaban + FEATURE_COLS。entry_dfと同じ行順。
        # Horse A (idx 0) と Horse B (idx 1) は dam D001 → log(3)
        # Horse C (idx 2) は dam D002 → log(2)
        np.testing.assert_allclose(
            result.loc[result.index[0], "breeder_strength"],
            np.log(3),
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            result.loc[result.index[2], "breeder_strength"],
            np.log(2),
            rtol=1e-4,
        )

    def test_more_breeders_means_higher_strength(
        self, basic_sanku: pd.DataFrame, basic_career: pd.DataFrame, basic_entry: pd.DataFrame
    ) -> None:
        """Test 7: より多いユニーク生産者 = より高い breeder_strength"""
        store = _make_store(sanku=basic_sanku)
        feat = DamPedigreeFeatures(store)
        feat._career_cache = basic_career

        result = feat.compute(basic_entry)

        # Horse A (idx 0, dam D001, 2 breeders) > Horse C (idx 2, dam D002, 1 breeder)
        d001_val = result.loc[result.index[0], "breeder_strength"]
        d002_val = result.loc[result.index[2], "breeder_strength"]
        assert d001_val > d002_val
