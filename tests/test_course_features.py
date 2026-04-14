"""test_course_features.py — CourseFeatures の単体テスト"""

from unittest.mock import MagicMock

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


def _mock_store() -> MagicMock:
    """compute_batch を使わないテスト用の mock store"""
    return MagicMock()


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
        feat = CourseFeatures(_mock_store())
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
        feat = CourseFeatures(_mock_store())
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
        feat = CourseFeatures(_mock_store())
        result = feat.compute(history, jyocd="01", distance_bin="sprint", target_date="2024-03-01")
        assert abs(result["course_wr"] - 1 / 11) < 0.001
        assert abs(result["course_distance_wr"] - 1 / 11) < 0.001

    def test_empty_history_returns_nan(self):
        """空履歴 → NaN (not prior, because no past data at all)"""
        feat = CourseFeatures(_mock_store())
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
        feat = CourseFeatures(_mock_store())
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
        feat = CourseFeatures(_mock_store())
        # sprint指定 → sprintの1件のみカウント
        result = feat.compute(history, jyocd="01", distance_bin="sprint", target_date="2024-04-01")
        # 01全体: 2勝/3出 → (1+2)/(1+10+3) = 3/14; 01+sprint: 1勝/1出 → (1+1)/(1+10+1) = 2/12
        assert abs(result["course_wr"] - 3 / 14) < 0.001
        assert abs(result["course_distance_wr"] - 2 / 12) < 0.001


class TestCourseFeaturesComputeBatch:
    """CourseFeatures.compute_batch() のテスト"""

    def _make_store(self, entries_df: pd.DataFrame, races_df: pd.DataFrame):
        """テスト用の mock ParquetStore を作成"""
        store = MagicMock()
        store.read.side_effect = lambda cat, name, **kw: (
            entries_df.copy() if name == "entries" else races_df.copy()
        )
        return store

    def test_compute_batch_returns_two_columns(self):
        """compute_batch が course_wr, course_distance_wr を返す"""
        entries = pd.DataFrame(
            {
                "kettonum": ["K1", "K1", "K2"],
                "race_id": ["H1", "H2", "H3"],
                "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                "kakuteijyuni": [1, 3, 2],
                "syussotosu": [10, 12, 8],
                "jyocd": ["01", "05", "02"],
            }
        )
        races = pd.DataFrame(
            {
                "race_id": ["H1", "H2", "H3"],
                "trackcd": [10, 10, 23],
                "kyori": [1600, 2000, 1400],
                "surface": ["turf", "turf", "dirt"],
                "track_condition_code": [1, 2, 3],
                "syussotosu": [16, 12, 8],
                "jyocd": ["01", "05", "02"],
            }
        )
        store = self._make_store(entries, races)

        df = pd.DataFrame(
            {
                "kettonum": ["K1", "K1", "K2"],
                "race_id": ["R1", "R2", "R1"],
                "race_date": pd.to_datetime(["2024-06-01", "2024-06-15", "2024-06-15"]),
                "surface": ["turf", "turf", "dirt"],
                "distance_bin": ["mile", "sprint", "sprint"],
                "jyocd": ["01", "01", "02"],
            }
        )

        feat = CourseFeatures(store)
        result = feat.compute_batch(df)

        assert "course_wr" in result.columns
        assert "course_distance_wr" in result.columns
        assert len(result) == 3

    def test_compute_batch_with_no_history(self):
        """過去走データがない場合、NaN が返る"""
        entries = pd.DataFrame(
            {
                "kettonum": ["K1"],
                "race_id": ["H1"],
                "race_date": pd.to_datetime(["2024-01-01"]),
                "kakuteijyuni": [1],
                "syussotosu": [10],
                "jyocd": ["01"],
            }
        )
        races = pd.DataFrame(
            {
                "race_id": ["H1"],
                "trackcd": [10],
                "kyori": [1600],
                "surface": ["turf"],
                "track_condition_code": [1],
                "syussotosu": [10],
                "jyocd": ["01"],
            }
        )
        store = self._make_store(entries, races)

        df = pd.DataFrame(
            {
                "kettonum": ["KX"],
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-06-01"]),
                "surface": ["turf"],
                "distance_bin": ["sprint"],
                "jyocd": ["01"],
            }
        )

        feat = CourseFeatures(store)
        result = feat.compute_batch(df)

        assert len(result) == 1
        assert np.isnan(result["course_wr"].iloc[0])
        assert np.isnan(result["course_distance_wr"].iloc[0])

    def test_compute_batch_filters_by_jyocd(self):
        """jyocd ごとの正しいフィルタリング"""
        entries = pd.DataFrame(
            {
                "kettonum": ["K1", "K1"],
                "race_id": ["H1", "H2"],
                "race_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "kakuteijyuni": [1, 1],
                "syussotosu": [10, 10],
                "jyocd": ["01", "05"],
            }
        )
        races = pd.DataFrame(
            {
                "race_id": ["H1", "H2"],
                "trackcd": [10, 10],
                "kyori": [1600, 1600],
                "surface": ["turf", "turf"],
                "track_condition_code": [1, 1],
                "syussotosu": [10, 10],
                "jyocd": ["01", "05"],
            }
        )
        store = self._make_store(entries, races)

        df = pd.DataFrame(
            {
                "kettonum": ["K1"],
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-06-01"]),
                "surface": ["turf"],
                "distance_bin": ["sprint"],
                "jyocd": ["01"],
            }
        )

        feat = CourseFeatures(store)
        result = feat.compute_batch(df)

        assert len(result) == 1
        # K1 has 1 race at jyocd 01 (H1) → (1+1)/(1+10+1) = 2/12
        assert abs(result["course_wr"].iloc[0] - 2 / 12) < 0.001

    def test_compute_batch_pit_no_future_leak(self):
        """当日以降のデータは使われない (PIT)"""
        entries = pd.DataFrame(
            {
                "kettonum": ["K1", "K1"],
                "race_id": ["H1", "H2"],
                "race_date": pd.to_datetime(["2024-05-15", "2024-07-01"]),
                "kakuteijyuni": [1, 1],
                "syussotosu": [10, 10],
                "jyocd": ["01", "01"],
            }
        )
        races = pd.DataFrame(
            {
                "race_id": ["H1", "H2"],
                "trackcd": [10, 10],
                "kyori": [1600, 1600],
                "surface": ["turf", "turf"],
                "track_condition_code": [1, 1],
                "syussotosu": [10, 10],
                "jyocd": ["01", "01"],
            }
        )
        store = self._make_store(entries, races)

        df = pd.DataFrame(
            {
                "kettonum": ["K1"],
                "race_id": ["R1"],
                "race_date": pd.to_datetime(["2024-06-01"]),
                "surface": ["turf"],
                "distance_bin": ["sprint"],
                "jyocd": ["01"],
            }
        )

        feat = CourseFeatures(store)
        result = feat.compute_batch(df)

        assert len(result) == 1
        # H2 (2024-07-01) は R1 (2024-06-01) より未来 → 除外
        # H1 (2024-05-15) のみ: 1着1回/1出走 → (1+1)/(1+10+1) = 2/12
        assert abs(result["course_wr"].iloc[0] - 2 / 12) < 0.001
