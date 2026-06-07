"""test_bloodline_features.py — BloodlineFeatures (PIT 版) の単体テスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from features.bloodline_features import ALPHA_PRIOR, FEATURE_COLS, TOTAL_OFFSET, BloodlineFeatures


def _make_store(career_df: pd.DataFrame) -> MagicMock:
    """PIT 版モック: load_career_stats 経由で career_df を返す。"""
    store = MagicMock()
    store.exists.return_value = not career_df.empty
    store.read.return_value = career_df
    return store


def _make_entry(n: int = 1, ketto_nums: list[str] | None = None) -> pd.DataFrame:
    if ketto_nums is None:
        ketto_nums = ["K001"] * n
    return pd.DataFrame(
        {
            "race_id": ["r1"] * n,
            "umaban": list(range(1, n + 1)),
            "kettonum": ketto_nums,
        }
    )


def _make_career_row(
    kettonum: str = "K001",
    cum_starts: int = 80,
    cum_wins: int = 10,
    cum_prize: float = 50000.0,
    cum_turf_starts: int = 50,
    cum_turf_wins: int = 5,
    cum_dirt_starts: int = 30,
    cum_dirt_wins: int = 5,
    cum_short_starts: int = 30,
    cum_short_wins: int = 3,
) -> dict:
    """Build one row of career stats with sensible defaults."""
    return {
        "race_id": "r1",
        "kettonum": kettonum,
        "race_date": pd.Timestamp("2025-01-01"),
        "cum_starts": cum_starts,
        "cum_wins": cum_wins,
        "cum_prize": cum_prize,
        "cum_turf_starts": cum_turf_starts,
        "cum_turf_wins": cum_turf_wins,
        "cum_dirt_starts": cum_dirt_starts,
        "cum_dirt_wins": cum_dirt_wins,
        "cum_short_starts": cum_short_starts,
        "cum_short_wins": cum_short_wins,
    }


# === Tests ===


class TestBloodTotalWr:
    def test_blood_total_wr(self):
        """cum_wins=10, cum_starts=80 -> (10+1)/(80+11)"""
        career = pd.DataFrame([_make_career_row(cum_wins=10, cum_starts=80)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        expected = (10 + ALPHA_PRIOR) / (80 + TOTAL_OFFSET)
        assert abs(result["blood_total_wr"].iloc[0] - expected) < 1e-10

    def test_debut_horse_nan(self):
        """cum_starts=0 -> NaN"""
        career = pd.DataFrame([_make_career_row(cum_starts=0, cum_wins=0)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert np.isnan(result["blood_total_wr"].iloc[0])


class TestBloodSurfaceWr:
    def test_blood_surface_wr(self):
        """cum_turf_wins=5, cum_turf_starts=50 -> (5+1)/(50+11)"""
        career = pd.DataFrame([_make_career_row(cum_turf_wins=5, cum_turf_starts=50)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        expected = (5 + ALPHA_PRIOR) / (50 + TOTAL_OFFSET)
        assert abs(result["blood_surface_wr"].iloc[0] - expected) < 1e-10

    def test_no_turf_starts_is_nan(self):
        """cum_turf_starts=0 -> NaN"""
        career = pd.DataFrame([_make_career_row(cum_turf_starts=0, cum_turf_wins=0)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert np.isnan(result["blood_surface_wr"].iloc[0])


class TestBloodDistanceWr:
    def test_blood_distance_wr(self):
        """cum_short_wins=3, cum_short_starts=30 -> (3+1)/(30+11)"""
        career = pd.DataFrame([_make_career_row(cum_short_wins=3, cum_short_starts=30)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        expected = (3 + ALPHA_PRIOR) / (30 + TOTAL_OFFSET)
        assert abs(result["blood_distance_wr"].iloc[0] - expected) < 1e-10

    def test_no_short_starts_is_nan(self):
        """cum_short_starts=0 -> NaN"""
        career = pd.DataFrame([_make_career_row(cum_short_starts=0, cum_short_wins=0)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert np.isnan(result["blood_distance_wr"].iloc[0])


class TestBloodPrizeLog:
    def test_blood_prize_log(self):
        career = pd.DataFrame([_make_career_row(cum_prize=50000.0)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert abs(result["blood_prize_log"].iloc[0] - np.log1p(50000)) < 1e-6

    def test_blood_prize_log_zero(self):
        career = pd.DataFrame([_make_career_row(cum_prize=0.0)])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert np.isnan(result["blood_prize_log"].iloc[0])


class TestEdgeCases:
    def test_missing_horse(self):
        """kettonum not in career -> all NaN (except blood_keito_cd = 'unknown')"""
        career = pd.DataFrame([_make_career_row(kettonum="K999")])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry(ketto_nums=["K001"]))
        for col in FEATURE_COLS:
            if col in ("blood_condition_wr", "blood_keito_cd"):
                continue
            assert np.isnan(result[col].iloc[0]), f"Expected NaN for {col}"

    def test_empty_entry(self):
        career = pd.DataFrame([_make_career_row()])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(
            pd.DataFrame(columns=["race_id", "umaban", "kettonum"])
        )
        assert len(result) == 0

    def test_empty_career(self):
        """career が空 -> all NaN"""
        store = _make_store(pd.DataFrame())
        result = BloodlineFeatures(store).compute(_make_entry())
        for col in FEATURE_COLS:
            assert np.isnan(result[col].iloc[0]), f"Expected NaN for {col}"

    def test_phase2_columns_are_nan(self):
        career = pd.DataFrame([_make_career_row()])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        assert np.isnan(result["blood_condition_wr"].iloc[0])
        # blood_keito_cd は keito データが利用不可の場合 'unknown'
        assert result["blood_keito_cd"].iloc[0] == "unknown"

    def test_multiple_horses(self):
        career = pd.DataFrame(
            [
                _make_career_row(kettonum="K001", cum_turf_wins=5, cum_turf_starts=50),
                _make_career_row(kettonum="K002", cum_turf_wins=10, cum_turf_starts=100),
            ]
        )
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry(n=2, ketto_nums=["K001", "K002"]))
        assert len(result) == 2
        assert abs(result["blood_surface_wr"].iloc[0] - (5 + 1) / (50 + 11)) < 1e-10
        assert abs(result["blood_surface_wr"].iloc[1] - (10 + 1) / (100 + 11)) < 1e-10

    def test_result_columns(self):
        """結果に race_id, umaban + FEATURE_COLS が含まれる"""
        career = pd.DataFrame([_make_career_row()])
        store = _make_store(career)
        result = BloodlineFeatures(store).compute(_make_entry())
        expected_cols = ["race_id", "umaban"] + FEATURE_COLS
        assert list(result.columns) == expected_cols


class TestBloodKeitoCd:
    """blood_keito_cd: entry -> sire -> keito master."""

    def test_blood_keito_cd_from_sire(self):
        """blood_keito_cd が種牡馬の系統コードを返す"""
        from unittest.mock import patch

        store = MagicMock()
        feat = BloodlineFeatures.__new__(BloodlineFeatures)
        feat.store = store
        feat._career_cache = pd.DataFrame(
            {
                "race_id": ["R1"],
                "kettonum": ["001"],
                "race_date": [pd.Timestamp("2024-06-01")],
                "cum_starts": [5],
                "cum_wins": [0],
                "cum_prize": [0],
                "cum_turf_starts": [3],
                "cum_turf_wins": [0],
                "cum_dirt_starts": [2],
                "cum_dirt_wins": [0],
                "cum_short_starts": [2],
                "cum_short_wins": [0],
            }
        )

        # horses: kettonum -> sire_id (ketto3infohansyokunum1)
        mock_horses = pd.DataFrame(
            {"kettonum": ["001"], "ketto3infohansyokunum1": ["SIRE_X"]}
        )
        # keito: keitoucode -> keitousystemcd
        mock_keito = pd.DataFrame(
            {"keitoucode": ["SIRE_X"], "keitousystemcd": ["SS"]}
        )

        with patch.object(
            store,
            "read",
            side_effect=lambda cat, name: {
                ("raw", "horses"): mock_horses,
                ("raw", "keito"): mock_keito,
            }.get((cat, name), pd.DataFrame()),
        ):
            feat._keito_cache = None  # キャッシュクリア
            entry_df = pd.DataFrame(
                {"race_id": ["R1"], "umaban": [1], "kettonum": ["001"]}
            )
            result = feat.compute(entry_df)
            assert result["blood_keito_cd"].iloc[0] == "SS"

    def test_blood_keito_cd_from_current_etl_schema(self):
        """現行ETLの hansyokunum/keitoname 形式から系統名を返す。"""
        store = MagicMock()
        store.exists.return_value = True
        store.read.side_effect = lambda cat, name: {
            ("raw", "horses"): pd.DataFrame(
                {"kettonum": ["001"], "ketto3infohansyokunum1": ["1140004481"]}
            ),
            ("raw", "keito"): pd.DataFrame(
                {"hansyokunum": ["1140004481"], "keitoname": ["サンデーサイレンス"]}
            ),
        }.get((cat, name), pd.DataFrame())

        feat = BloodlineFeatures(store)
        assert feat._load_keito_map()["1140004481"] == "サンデーサイレンス"

    def test_blood_keito_cd_unknown_for_missing_sire(self):
        """未知の種牡馬は 'unknown' を返す"""
        from unittest.mock import patch

        store = MagicMock()
        feat = BloodlineFeatures.__new__(BloodlineFeatures)
        feat.store = store
        feat._career_cache = pd.DataFrame(
            {
                "race_id": ["R1"],
                "kettonum": ["001"],
                "race_date": [pd.Timestamp("2024-06-01")],
                "cum_starts": [5],
                "cum_wins": [0],
                "cum_prize": [0],
                "cum_turf_starts": [3],
                "cum_turf_wins": [0],
                "cum_dirt_starts": [2],
                "cum_dirt_wins": [0],
                "cum_short_starts": [2],
                "cum_short_wins": [0],
            }
        )

        # horses に kettonum=001 がない場合
        mock_horses = pd.DataFrame(
            {"kettonum": ["999"], "ketto3infohansyokunum1": ["OTHER_SIRE"]}
        )
        mock_keito = pd.DataFrame()

        with patch.object(
            store,
            "read",
            side_effect=lambda cat, name: {
                ("raw", "horses"): mock_horses,
                ("raw", "keito"): mock_keito,
            }.get((cat, name), pd.DataFrame()),
        ):
            feat._keito_cache = None
            entry_df = pd.DataFrame(
                {"race_id": ["R1"], "umaban": [1], "kettonum": ["001"]}
            )
            result = feat.compute(entry_df)
            assert result["blood_keito_cd"].iloc[0] == "unknown"


class TestBloodConditionWr:
    """blood_condition_wr: 馬場状態別勝率 Beta平滑化"""

    @staticmethod
    def _make_career_with_condition() -> pd.DataFrame:
        return pd.DataFrame({
            "race_id": ["R1"],
            "kettonum": ["001"],
            "race_date": [pd.Timestamp("2024-06-01")],
            "cum_starts": [20],
            "cum_wins": [3],
            "cum_prize": [5000],
            "cum_turf_starts": [10],
            "cum_turf_wins": [2],
            "cum_dirt_starts": [10],
            "cum_dirt_wins": [1],
            "cum_short_starts": [8],
            "cum_short_wins": [2],
            "cum_turf_good_starts": [8],
            "cum_turf_good_wins": [2],
            "cum_turf_heavy_starts": [2],
            "cum_turf_heavy_wins": [0],
            "cum_dirt_good_starts": [6],
            "cum_dirt_good_wins": [1],
            "cum_dirt_heavy_starts": [4],
            "cum_dirt_heavy_wins": [0],
        })

    def test_blood_condition_wr_good_turf(self):
        """blood_condition_wr が芝良馬場の勝率を返す"""
        store = MagicMock()
        feat = BloodlineFeatures.__new__(BloodlineFeatures)
        feat.store = store
        feat._career_cache = self._make_career_with_condition()
        feat._keito_cache = {}

        entry_df = pd.DataFrame({
            "race_id": ["R1"],
            "umaban": [1],
            "kettonum": ["001"],
            "surface": ["turf"],
            "track_condition_code": [1],  # good
        })
        result = feat.compute(entry_df)
        # turf + good -> cum_turf_good: 2/8 -> Beta(1+2, 10+8) = 3/19
        expected = (2 + ALPHA_PRIOR) / (8 + TOTAL_OFFSET)
        assert abs(result["blood_condition_wr"].iloc[0] - expected) < 0.001

    def test_blood_condition_wr_heavy_dirt(self):
        """blood_condition_wr がダート不良馬場の勝率を返す"""
        store = MagicMock()
        feat = BloodlineFeatures.__new__(BloodlineFeatures)
        feat.store = store
        feat._career_cache = self._make_career_with_condition()
        feat._keito_cache = {}

        entry_df = pd.DataFrame({
            "race_id": ["R1"],
            "umaban": [1],
            "kettonum": ["001"],
            "surface": ["dirt"],
            "track_condition_code": [4],  # heavy (不良)
        })
        result = feat.compute(entry_df)
        # dirt + heavy -> cum_dirt_heavy: 0/4 -> Beta(1+0, 10+4-0) = 1/15
        expected = 1 / 15
        assert abs(result["blood_condition_wr"].iloc[0] - expected) < 0.001

    def test_blood_condition_wr_no_data_returns_nan(self):
        """累積データがない場合はNaNを返す"""
        store = MagicMock()
        feat = BloodlineFeatures.__new__(BloodlineFeatures)
        feat.store = store
        feat._keito_cache = {}
        # cum_turf_good_starts=0 -> starts>0 check fails -> NaN
        career = self._make_career_with_condition().copy()
        for col in [
            "cum_turf_good_starts", "cum_turf_good_wins",
            "cum_turf_heavy_starts", "cum_turf_heavy_wins",
            "cum_dirt_good_starts", "cum_dirt_good_wins",
            "cum_dirt_heavy_starts", "cum_dirt_heavy_wins",
        ]:
            career[col] = 0
        feat._career_cache = career

        entry_df = pd.DataFrame({
            "race_id": ["R1"],
            "umaban": [1],
            "kettonum": ["001"],
            "surface": ["turf"],
            "track_condition_code": [1],
        })
        result = feat.compute(entry_df)
        assert pd.isna(result["blood_condition_wr"].iloc[0])
