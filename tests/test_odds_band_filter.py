"""src/betting/odds_band_filter.py のテスト"""

from __future__ import annotations

import pandas as pd
import pytest

from betting.odds_band_filter import OddsBandFilter


class TestOddsBandFilter:
    """OddsBandFilter の単体テスト"""

    def test_calibrate_empty_list_excluded_bands_empty(self) -> None:
        """Test 1: calibrate() に空リストを渡すと excluded_bands が空であること"""
        obf = OddsBandFilter()
        obf.calibrate([])
        assert obf.excluded_bands == {}

    def test_calibrate_roi_below_100_excludes_band(self) -> None:
        """Test 2: ROI < 100% のバンドのみ → そのバンドが excluded"""
        obf = OddsBandFilter()
        # バンド 1.0-3.0 のみで ROI = 50/100 = 50% < 100%
        bet_history = [
            {"odds": 2.0, "stake": 100, "result": 50},
        ]
        obf.calibrate(bet_history)
        assert "1.0-3.0" in obf.excluded_bands

    def test_calibrate_roi_at_or_above_100_keeps_band(self) -> None:
        """Test 3: ROI >= 100% のバンドは excluded にならない"""
        obf = OddsBandFilter()
        # バンド 1.0-3.0: ROI = 200/100 = 200%
        bet_history = [
            {"odds": 2.0, "stake": 100, "result": 200},
        ]
        obf.calibrate(bet_history)
        assert "1.0-3.0" not in obf.excluded_bands

    def test_filter_excludes_candidates_in_excluded_bands(self) -> None:
        """Test 4: filter() は除外バンドの候補を除外し、それ以外は保持"""
        obf = OddsBandFilter()
        # バンド 1.0-3.0 のみ ROI < 100%
        bet_history = [
            {"odds": 2.0, "stake": 100, "result": 50},   # 1.0-3.0, ROI 50%
            {"odds": 5.0, "stake": 100, "result": 200},  # 3.0-10.0, ROI 200%
        ]
        obf.calibrate(bet_history)

        candidates = pd.DataFrame({
            "umaban": [1, 2, 3],
            "tanodds": [2.0, 5.0, 15.0],  # 1.0-3.0, 3.0-10.0, 10.0-30.0
        })
        result = obf.filter(candidates)
        assert len(result) == 2
        assert 1 not in result["umaban"].values  # 除外された
        assert 2 in result["umaban"].values
        assert 3 in result["umaban"].values

    def test_filter_without_calibrate_returns_all(self) -> None:
        """Test 5: calibrate() 未呼び出しで filter() を呼んでもエラーにならず全候補を返す"""
        obf = OddsBandFilter()
        candidates = pd.DataFrame({
            "umaban": [1, 2],
            "tanodds": [2.0, 5.0],
        })
        result = obf.filter(candidates)
        assert len(result) == 2

    def test_excluded_bands_property_contains_roi_and_count(self) -> None:
        """Test 6: excluded_bands が band_name -> {roi, count} を含む"""
        obf = OddsBandFilter()
        bet_history = [
            {"odds": 2.0, "stake": 100, "result": 50},   # 1.0-3.0, ROI 50%, count 1
            {"odds": 5.0, "stake": 100, "result": 200},  # 3.0-10.0, ROI 200%
            {"odds": 2.5, "stake": 100, "result": 30},   # 1.0-3.0, ROI 30%, count 2
        ]
        obf.calibrate(bet_history)
        excluded = obf.excluded_bands
        assert "1.0-3.0" in excluded
        assert excluded["1.0-3.0"]["roi"] == pytest.approx(0.4)  # (50+30)/(100+100)
        assert excluded["1.0-3.0"]["count"] == 2

    def test_calibrate_custom_roi_threshold_keeps_band(self) -> None:
        """roi_threshold=0.9 のとき ROI=0.95 のバンドは除外されない"""
        obf = OddsBandFilter(roi_threshold=0.9)
        # バンド 1.0-3.0: ROI = 95/100 = 0.95 > 0.9 → KEEP
        bet_history = [
            {"odds": 2.0, "stake": 100, "result": 95},
        ]
        obf.calibrate(bet_history)
        assert "1.0-3.0" not in obf.excluded_bands

    def test_calibrate_custom_roi_threshold_excludes_band(self) -> None:
        """roi_threshold=1.1 のとき ROI=1.05 のバンドは除外される"""
        obf = OddsBandFilter(roi_threshold=1.1)
        # バンド 1.0-3.0: ROI = 105/100 = 1.05 < 1.1 → EXCLUDE
        bet_history = [
            {"odds": 2.0, "stake": 100, "result": 105},
        ]
        obf.calibrate(bet_history)
        assert "1.0-3.0" in obf.excluded_bands
