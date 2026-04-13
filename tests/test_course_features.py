"""test_course_features.py — CourseFeatures の単体テスト"""

import numpy as np
import pandas as pd

from features.course_features import CourseFeatures, _beta_smooth


class TestBetaSmooth:
    def test_prior_only(self):
        """0戦0勝 → Beta(1,11) ≈ 0.0909"""
        assert abs(_beta_smooth(0, 0) - 1 / 11) < 0.001

    def test_50_percent_win_rate(self):
        """5戦5勝 → (1+5)/(1+10+5) = 6/16 = 0.375"""
        assert abs(_beta_smooth(5, 5) - 6 / 16) < 0.001


class TestCourseWr:
    def test_course_wr_beta_smoothed(self):
        """course_wr が競馬場別のBeta平滑化勝率を返す"""
        history = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                "jyocd": ["01", "01", "05"],
                "kakuteijyuni": [1, 3, 2],
                "distance_bin": ["sprint", "sprint", "mile"],
                "syussotosu": [10, 12, 8],
            }
        )
        feat = CourseFeatures()
        result = feat.compute(history, jyocd="01", distance_bin="sprint", target_date="2024-04-01")
        # 01競馬場: 1着1回/2出走 → (1+1)/(1+10+2) = 2/13 ≈ 0.154
        assert abs(result["course_wr"] - 2 / 13) < 0.001

    def test_course_distance_wr(self):
        """course_distance_wr が競馬場×距離帯別のBeta平滑化勝率を返す"""
        history = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                "jyocd": ["01", "01", "05"],
                "kakuteijyuni": [1, 3, 2],
                "distance_bin": ["sprint", "sprint", "mile"],
                "syussotosu": [10, 12, 8],
            }
        )
        feat = CourseFeatures()
        result = feat.compute(history, jyocd="01", distance_bin="sprint", target_date="2024-04-01")
        # 01+sprint: 1着1回/2出走 → (1+1)/(1+10+2) = 2/13
        assert abs(result["course_distance_wr"] - 2 / 13) < 0.001

    def test_no_venue_history_returns_prior(self):
        """該当競馬場の履歴なし → 事前分布 Beta(1,11)"""
        history = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "jyocd": ["05", "05"],  # 中山のみ
                "kakuteijyuni": [1, 2],
                "distance_bin": ["mile", "mile"],
                "syussotosu": [10, 12],
            }
        )
        feat = CourseFeatures()
        result = feat.compute(history, jyocd="01", distance_bin="sprint", target_date="2024-03-01")
        assert abs(result["course_wr"] - 1 / 11) < 0.001
        assert abs(result["course_distance_wr"] - 1 / 11) < 0.001

    def test_empty_history_returns_nan(self):
        """空履歴 → NaN (not prior, because no past data at all)"""
        feat = CourseFeatures()
        result = feat.compute(
            pd.DataFrame(), jyocd="01", distance_bin="sprint", target_date="2024-06-01"
        )
        assert np.isnan(result["course_wr"])
        assert np.isnan(result["course_distance_wr"])

    def test_pit_future_excluded(self):
        """当日レースは除外される (PIT)"""
        history = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2024-05-15", "2024-06-01"]),
                "jyocd": ["01", "01"],
                "kakuteijyuni": [1, 1],  # 当日も勝っているが除外されるべき
                "distance_bin": ["sprint", "sprint"],
                "syussotosu": [10, 10],
            }
        )
        feat = CourseFeatures()
        result = feat.compute(history, jyocd="01", distance_bin="sprint", target_date="2024-06-01")
        # 6/1は除外 → 5/15のみ: 1着1回/1出走 → (1+1)/(1+10+1) = 2/12 = 1/6
        assert abs(result["course_wr"] - 2 / 12) < 0.001

    def test_different_distance_bin_filters_correctly(self):
        """異なるdistance_binはフィルタリングされる"""
        history = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                "jyocd": ["01", "01", "01"],
                "kakuteijyuni": [1, 1, 5],
                "distance_bin": ["sprint", "mile", "intermediate"],
                "syussotosu": [10, 12, 14],
            }
        )
        feat = CourseFeatures()
        # sprint指定 → sprintの1件のみカウント
        result = feat.compute(history, jyocd="01", distance_bin="sprint", target_date="2024-04-01")
        # 01全体: 2勝/3出 → (1+2)/(1+10+3) = 3/14; 01+sprint: 1勝/1出 → (1+1)/(1+10+1) = 2/12
        assert abs(result["course_wr"] - 3 / 14) < 0.001
        assert abs(result["course_distance_wr"] - 2 / 12) < 0.001
