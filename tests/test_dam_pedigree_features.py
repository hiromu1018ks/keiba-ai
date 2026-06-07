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


class TestPITSafeBehavior:
    """Tests for PIT-safe career lookup using merge_asof."""

    def test_pit_safe_different_dates_get_different_features(self) -> None:
        """Entries from different dates get different dam features."""
        sanku = pd.DataFrame({
            "kettonum": ["A1", "A2"],
            "mnum": ["D1", "D1"],
            "breedercode": ["BR1", "BR1"],
        })

        # Career stats with race_date for PIT-safe lookup
        # Horse A1: early career (2020) had 1 win in 10 starts,
        #           later career (2023) had 5 wins in 30 starts
        career = pd.DataFrame({
            "kettonum": ["A1", "A1", "A2", "A2"],
            "race_id": ["R001", "R003", "R002", "R004"],
            "race_date": pd.to_datetime(["2020-06-01", "2023-06-01", "2020-06-01", "2023-06-01"]),
            "cum_wins": [1, 5, 2, 4],
            "cum_starts": [10, 30, 10, 20],
            "cum_turf_wins": [0, 3, 1, 2],
            "cum_turf_starts": [5, 20, 5, 10],
            "cum_prize": [1000000, 50000000, 2000000, 40000000],
        })

        store = _make_store(sanku=sanku)
        feat = DamPedigreeFeatures(store)
        feat._career_cache = career

        # Entry at early date (R001 = 2020-06-01)
        entry_early = pd.DataFrame({
            "race_id": ["R001"],
            "umaban": [1],
            "kettonum": ["A1"],
        })
        # Entry at late date (R003 = 2023-06-01)
        entry_late = pd.DataFrame({
            "race_id": ["R003"],
            "umaban": [1],
            "kettonum": ["A1"],
        })

        result_early = feat.compute(entry_early)
        result_late = feat.compute(entry_late)

        # Early entry: target_date=2020-06-01
        # merge_asof backward: A1 at 2020-06-01 -> cum_wins=1, cum_starts=10
        #                      A2 at 2020-06-01 -> cum_wins=2, cum_starts=10
        # total_wins=3, total_starts=20, dam_wr = (3+1)/(20+11) = 4/31
        expected_early = (3 + 1) / (20 + 11)
        np.testing.assert_allclose(
            result_early.loc[result_early.index[0], "dam_wr"],
            expected_early,
            rtol=1e-4,
        )

        # Late entry: target_date=2023-06-01
        # merge_asof backward: A1 at 2023-06-01 -> cum_wins=5, cum_starts=30
        #                      A2 at 2023-06-01 -> cum_wins=4, cum_starts=20
        # total_wins=9, total_starts=50, dam_wr = (9+1)/(50+11) = 10/61
        expected_late = (9 + 1) / (50 + 11)
        np.testing.assert_allclose(
            result_late.loc[result_late.index[0], "dam_wr"],
            expected_late,
            rtol=1e-4,
        )

        # Verify they are different (PIT-safe ensures this)
        assert (
            result_early.loc[result_early.index[0], "dam_wr"]
            != result_late.loc[result_late.index[0], "dam_wr"]
        )

    def test_pit_safe_no_future_leak(self) -> None:
        """Verify that future career data does not leak into earlier entries."""
        sanku = pd.DataFrame({
            "kettonum": ["H1"],
            "mnum": ["D1"],
            "breedercode": ["BR1"],
        })

        career = pd.DataFrame({
            "kettonum": ["H1", "H1"],
            "race_id": ["R_early", "R_late"],
            "race_date": pd.to_datetime(["2020-01-01", "2025-01-01"]),
            "cum_wins": [2, 10],
            "cum_starts": [10, 50],
            "cum_turf_wins": [1, 5],
            "cum_turf_starts": [5, 25],
            "cum_prize": [5000000, 50000000],
        })

        # Entry at the early date - should NOT see the late career stats
        entry_early = pd.DataFrame({
            "race_id": ["R_early"],
            "umaban": [1],
            "kettonum": ["H1"],
        })

        store = _make_store(sanku=sanku)
        feat = DamPedigreeFeatures(store)
        feat._career_cache = career

        result = feat.compute(entry_early)

        # target_date=2020-01-01, backward merge_asof finds the 2020-01-01 row
        # H1: cum_wins=2, cum_starts=10
        # dam_wr=(2+1)/(10+11)=3/21
        expected = (2 + 1) / (10 + 11)
        np.testing.assert_allclose(
            result.loc[result.index[0], "dam_wr"],
            expected,
            rtol=1e-4,
        )

        # Verify it is NOT the late value: (10+1)/(50+11)=11/61
        late_expected = (10 + 1) / (50 + 11)
        assert abs(result.loc[result.index[0], "dam_wr"] - late_expected) > 0.01

    def test_pit_safe_offspring_not_in_target_race(self) -> None:
        """Offspring not running in target race still get their latest prior stats."""
        sanku = pd.DataFrame({
            "kettonum": ["H1", "H2"],
            "mnum": ["D1", "D1"],
            "breedercode": ["BR1", "BR1"],
        })

        # H1 has races at 2020 and 2022
        # H2 only has a race at 2019 (does NOT run in 2022)
        career = pd.DataFrame({
            "kettonum": ["H1", "H1", "H2"],
            "race_id": ["R1", "R2", "R3"],
            "race_date": pd.to_datetime(["2020-01-01", "2022-01-01", "2019-01-01"]),
            "cum_wins": [3, 7, 1],
            "cum_starts": [20, 40, 5],
            "cum_turf_wins": [2, 4, 0],
            "cum_turf_starts": [10, 20, 0],
            "cum_prize": [10000000, 30000000, 1000000],
        })

        # Entry for H1 at R2 (2022-01-01)
        # H2 did not run in R2, but merge_asof backward finds H2 2019 row
        entry = pd.DataFrame({
            "race_id": ["R2"],
            "umaban": [1],
            "kettonum": ["H1"],
        })

        store = _make_store(sanku=sanku)
        feat = DamPedigreeFeatures(store)
        feat._career_cache = career

        result = feat.compute(entry)

        # target_date=2022-01-01
        # H1: backward merge -> 2022 row: cum_wins=7, cum_starts=40
        # H2: backward merge -> 2019 row: cum_wins=1, cum_starts=5
        # total_wins=8, total_starts=45, dam_wr=(8+1)/(45+11)=9/56
        expected = (8 + 1) / (45 + 11)
        np.testing.assert_allclose(
            result.loc[result.index[0], "dam_wr"],
            expected,
            rtol=1e-4,
        )

    def test_pit_safe_handles_non_default_entry_index_and_categorical_key(self) -> None:
        """merge後のindex変更とCategoricalなkettonumを同時に処理できる。"""
        sanku = pd.DataFrame({
            "kettonum": ["H1", "H2"],
            "mnum": ["D1", "D1"],
            "breedercode": ["BR1", "BR1"],
        })
        career = pd.DataFrame({
            "kettonum": pd.Categorical(["H1", "H2"]),
            "race_id": ["R1", "R0"],
            "race_date": pd.to_datetime(["2022-01-01", "2021-01-01"]),
            "cum_wins": [3, 1],
            "cum_starts": [20, 5],
            "cum_turf_wins": [2, 0],
            "cum_turf_starts": [10, 0],
            "cum_prize": [10_000_000, 1_000_000],
        })
        entry = pd.DataFrame(
            {"race_id": ["R1"], "umaban": [1], "kettonum": ["H1"]},
            index=[99],
        )

        feat = DamPedigreeFeatures(_make_store(sanku=sanku))
        feat._career_cache = career
        result = feat.compute(entry)

        assert result["dam_wr"].iloc[0] == pytest.approx((4 + 1) / (25 + 11))

    def test_vectorized_pit_handles_multiple_dams_and_dates(self) -> None:
        """複数母・複数日付を一括計算しても各母の履歴が混ざらない。"""
        sanku = pd.DataFrame({
            "kettonum": ["A1", "A2", "B1"],
            "mnum": ["DA", "DA", "DB"],
            "breedercode": ["BR1", "BR2", "BR3"],
        })
        career = pd.DataFrame({
            "kettonum": ["A1", "A2", "B1", "A1", "B1"],
            "race_id": ["A_E", "A2_E", "B_E", "A_L", "B_L"],
            "race_date": pd.to_datetime([
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2022-01-01",
                "2022-01-01",
            ]),
            "cum_wins": [1, 2, 4, 5, 8],
            "cum_starts": [10, 10, 20, 30, 40],
            "cum_turf_wins": [1, 0, 2, 3, 4],
            "cum_turf_starts": [5, 5, 10, 20, 20],
            "cum_prize": [1_000_000, 2_000_000, 4_000_000, 9_000_000, 8_000_000],
        })
        entry = pd.DataFrame({
            "race_id": ["A_E", "B_E", "A_L", "B_L"],
            "umaban": [1, 2, 3, 4],
            "kettonum": ["A1", "B1", "A1", "B1"],
        })

        feat = DamPedigreeFeatures(_make_store(sanku=sanku))
        feat._career_cache = career
        result = feat.compute(entry)

        assert result["dam_wr"].tolist() == pytest.approx([
            (3 + 1) / (20 + 11),
            (4 + 1) / (20 + 11),
            (7 + 1) / (40 + 11),
            (8 + 1) / (40 + 11),
        ])

    def test_fallback_used_when_no_race_date(self) -> None:
        """When career has no race_date/race_id, fallback path is used."""
        sanku = pd.DataFrame({
            "kettonum": ["H1"],
            "mnum": ["D1"],
            "breedercode": ["BR1"],
        })

        # Career without race_date or race_id columns
        career = pd.DataFrame({
            "kettonum": ["H1"],
            "cum_wins": [5],
            "cum_starts": [30],
            "cum_turf_wins": [3],
            "cum_turf_starts": [20],
            "cum_prize": [50000000],
        })

        entry = pd.DataFrame({
            "race_id": ["R001"],
            "umaban": [1],
            "kettonum": ["H1"],
        })

        store = _make_store(sanku=sanku)
        feat = DamPedigreeFeatures(store)
        feat._career_cache = career

        result = feat.compute(entry)

        # Fallback: last() -> cum_wins=5, cum_starts=30
        # dam_wr = (5+1)/(30+11) = 6/41
        expected = (5 + 1) / (30 + 11)
        np.testing.assert_allclose(
            result.loc[result.index[0], "dam_wr"],
            expected,
            rtol=1e-4,
        )
