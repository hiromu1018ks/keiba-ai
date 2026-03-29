"""test_bloodline_features.py — BloodlineFeatures の単体テスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from features.bloodline_features import ALPHA_PRIOR, BETA_PRIOR, BloodlineFeatures, TOTAL_OFFSET


# ---------------------------------------------------------------------------
# Helper: build a mock DataRepository with a horses DataFrame
# ---------------------------------------------------------------------------


def _make_repo(horses_df: pd.DataFrame) -> MagicMock:
    repo = MagicMock()
    repo.load_horses.return_value = horses_df
    return repo


def _make_entry(n: int = 1, ketto_nums: list[str] | None = None) -> pd.DataFrame:
    if ketto_nums is None:
        ketto_nums = ["K001"] * n
    return pd.DataFrame(
        {
            "race_id": ["r1"] * n,
            "umaban": list(range(1, n + 1)),
            "ketto_num": ketto_nums,
        }
    )


def _make_horses_row(
    kettonum: str = "K001",
    ba1chakukaisu1: int = 5,
    ba1_total: int = 50,
    kyori1chakukaisu1: int = 3,
    kyori1_total: int = 30,
    chuochakukaisu1: int = 10,
    chuo_total: int = 80,
    ruikeihonsyoheiti: float = 50000.0,
) -> dict:
    """Build one row of x_UMA data with sensible defaults.

    chakukaisu columns: 1=1着, 2=2着, 3=3着, 4=4着, 5=5着, 6=着外
    total = sum(1..6) means total starts for that category.
    """
    # Distribute non-win finishes across places 2-6
    ba1_rest = ba1_total - ba1chakukaisu1
    ky1_rest = kyori1_total - kyori1chakukaisu1
    chuo_rest = chuo_total - chuochakukaisu1

    return {
        "kettonum": kettonum,
        "ba1chakukaisu1": ba1chakukaisu1,
        "ba1chakukaisu2": ba1_rest // 5,
        "ba1chakukaisu3": ba1_rest // 5,
        "ba1chakukaisu4": ba1_rest // 5,
        "ba1chakukaisu5": ba1_rest // 5,
        "ba1chakukaisu6": ba1_rest - 4 * (ba1_rest // 5),
        "kyori1chakukaisu1": kyori1chakukaisu1,
        "kyori1chakukaisu2": ky1_rest // 5,
        "kyori1chakukaisu3": ky1_rest // 5,
        "kyori1chakukaisu4": ky1_rest // 5,
        "kyori1chakukaisu5": ky1_rest // 5,
        "kyori1chakukaisu6": ky1_rest - 4 * (ky1_rest // 5),
        "chuochakukaisu1": chuochakukaisu1,
        "chuochakukaisu2": chuo_rest // 5,
        "chuochakukaisu3": chuo_rest // 5,
        "chuochakukaisu4": chuo_rest // 5,
        "chuochakukaisu5": chuo_rest // 5,
        "chuochakukaisu6": chuo_rest - 4 * (chuo_rest // 5),
        "ruikeihonsyoheiti": ruikeihonsyoheiti,
    }


# ===========================================================================
# Tests: _smoothed_wr
# ===========================================================================


class TestSmoothedWr:
    """Beta(α,β) 平滑化勝率: (wins+1)/(total+11)"""

    def test_smoothed_wr_basic(self):
        """wins=5, total=50 -> (5+1)/(50+11) = 6/61"""
        result = BloodlineFeatures._smoothed_wr(5, 50)
        expected = (5 + ALPHA_PRIOR) / (50 + TOTAL_OFFSET)
        assert abs(result - expected) < 1e-10
        assert abs(result - 6 / 61) < 1e-10

    def test_smoothed_wr_zero_total(self):
        """total=0 -> NaN"""
        result = BloodlineFeatures._smoothed_wr(3, 0)
        assert np.isnan(result)

    def test_smoothed_wr_zero_wins(self):
        """wins=0, total=100 -> (0+1)/(100+11) = 1/111"""
        result = BloodlineFeatures._smoothed_wr(0, 100)
        expected = 1 / 111
        assert abs(result - expected) < 1e-10

    def test_smoothed_wr_all_wins(self):
        """wins=100, total=100 -> (100+1)/(100+11) = 101/111"""
        result = BloodlineFeatures._smoothed_wr(100, 100)
        expected = 101 / 111
        assert abs(result - expected) < 1e-10


# ===========================================================================
# Tests: compute — 血統馬場別勝率
# ===========================================================================


class TestBloodSurfaceWr:
    """blood_surface_wr: 芝 (ba1) の平滑化勝率"""

    def test_blood_surface_wr(self):
        """ba1chakukaisu1=5, total=50 -> (5+1)/(50+11)"""
        horses = pd.DataFrame([_make_horses_row(ba1chakukaisu1=5, ba1_total=50)])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = (5 + ALPHA_PRIOR) / (50 + TOTAL_OFFSET)
        assert abs(result["blood_surface_wr"].iloc[0] - expected) < 1e-10

    def test_blood_surface_wr_no_turf_starts(self):
        """ba1 全部 0 -> total=0 -> NaN"""
        horses = pd.DataFrame([_make_horses_row(ba1chakukaisu1=0, ba1_total=0)])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        assert np.isnan(result["blood_surface_wr"].iloc[0])


# ===========================================================================
# Tests: compute — 総合成績勝率
# ===========================================================================


class TestBloodTotalWr:
    """blood_total_wr: 中央 (chuo) 総合成績の平滑化勝率"""

    def test_blood_total_wr(self):
        """chuo: wins=10, total=80 -> (10+1)/(80+11)"""
        horses = pd.DataFrame([_make_horses_row(chuochakukaisu1=10, chuo_total=80)])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = (10 + ALPHA_PRIOR) / (80 + TOTAL_OFFSET)
        assert abs(result["blood_total_wr"].iloc[0] - expected) < 1e-10


# ===========================================================================
# Tests: compute — 累計賞金 log 変換
# ===========================================================================


class TestBloodPrizeLog:
    """blood_prize_log: log(1 + prize) 変換"""

    def test_blood_prize_log(self):
        """ruikeihonsyoheiti=50000 -> log1p(50000)"""
        horses = pd.DataFrame([_make_horses_row(ruikeihonsyoheiti=50000.0)])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = float(np.log1p(50000.0))
        assert abs(result["blood_prize_log"].iloc[0] - expected) < 1e-10

    def test_blood_prize_log_zero(self):
        """prize=0 -> NaN"""
        horses = pd.DataFrame([_make_horses_row(ruikeihonsyoheiti=0.0)])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        assert np.isnan(result["blood_prize_log"].iloc[0])

    def test_blood_prize_log_nan(self):
        """prize=NaN -> NaN"""
        horses = pd.DataFrame([_make_horses_row(ruikeihonsyoheiti=float("nan"))])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        assert np.isnan(result["blood_prize_log"].iloc[0])


# ===========================================================================
# Tests: compute — 距離別勝率
# ===========================================================================


class TestBloodDistanceWr:
    """blood_distance_wr: 短距離 (kyori1) の平滑化勝率"""

    def test_blood_distance_wr(self):
        """kyori1: wins=3, total=30 -> (3+1)/(30+11) = 4/41"""
        horses = pd.DataFrame([_make_horses_row(kyori1chakukaisu1=3, kyori1_total=30)])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = (3 + ALPHA_PRIOR) / (30 + TOTAL_OFFSET)
        assert abs(result["blood_distance_wr"].iloc[0] - expected) < 1e-10


# ===========================================================================
# Tests: edge cases
# ===========================================================================


class TestEdgeCases:
    """欠損・空データのエッジケース"""

    def test_missing_horse(self):
        """ketto_num not in horses_df -> all feature columns NaN"""
        horses = pd.DataFrame([_make_horses_row(kettonum="K999")])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry(ketto_nums=["K001"])  # K001 not in horses
        result = feat.compute(entry)
        from features.bloodline_features import FEATURE_COLS

        for col in FEATURE_COLS:
            if col in ("blood_condition_wr", "blood_keito_cd"):
                # Phase 2 placeholders are always NaN
                continue
            assert np.isnan(result[col].iloc[0]), f"Expected NaN for {col}"

    def test_empty_entry(self):
        """空の entry_df -> 空の結果 (race_id, umaban + NaN columns)"""
        horses = pd.DataFrame([_make_horses_row()])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = pd.DataFrame(columns=["race_id", "umaban", "ketto_num"])
        result = feat.compute(entry)
        # Should have correct columns but be empty
        assert "race_id" in result.columns
        assert "umaban" in result.columns
        assert len(result) == 0

    def test_multiple_horses(self):
        """複数頭のエントリーでそれぞれ正しい値が返る"""
        horses = pd.DataFrame([
            _make_horses_row(kettonum="K001", ba1chakukaisu1=5, ba1_total=50),
            _make_horses_row(kettonum="K002", ba1chakukaisu1=10, ba1_total=100),
        ])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry(n=2, ketto_nums=["K001", "K002"])
        result = feat.compute(entry)
        assert len(result) == 2
        expected_k1 = (5 + ALPHA_PRIOR) / (50 + TOTAL_OFFSET)
        expected_k2 = (10 + ALPHA_PRIOR) / (100 + TOTAL_OFFSET)
        assert abs(result["blood_surface_wr"].iloc[0] - expected_k1) < 1e-10
        assert abs(result["blood_surface_wr"].iloc[1] - expected_k2) < 1e-10

    def test_phase2_columns_are_nan(self):
        """Phase 2 プレースホルダー (blood_condition_wr, blood_keito_cd) は NaN"""
        horses = pd.DataFrame([_make_horses_row()])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        assert np.isnan(result["blood_condition_wr"].iloc[0])
        assert np.isnan(result["blood_keito_cd"].iloc[0])

    def test_empty_horses_df(self):
        """horses_df が空 -> 全列 NaN"""
        horses = pd.DataFrame()
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        from features.bloodline_features import FEATURE_COLS

        for col in FEATURE_COLS:
            assert np.isnan(result[col].iloc[0]), f"Expected NaN for {col}"

    def test_result_columns(self):
        """結果に race_id, umaban + FEATURE_COLS が含まれる"""
        horses = pd.DataFrame([_make_horses_row()])
        repo = _make_repo(horses)
        feat = BloodlineFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        expected_cols = ["race_id", "umaban"] + [
            "blood_surface_wr",
            "blood_distance_wr",
            "blood_condition_wr",
            "blood_total_wr",
            "blood_prize_log",
            "blood_keito_cd",
        ]
        assert list(result.columns) == expected_cols
