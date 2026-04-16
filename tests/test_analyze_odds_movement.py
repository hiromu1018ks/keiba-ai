"""tests/test_analyze_odds_movement.py — オッズ変動分析のユニットテスト"""

import pandas as pd
import pytest

from scripts.analyze_odds_movement import compute_movement_features


@pytest.fixture
def sample_time_series() -> pd.DataFrame:
    """3頭×5時点のサンプル時系列データ"""
    rows = []
    base_time = "01051200"  # MMDDHHmm 形式

    # 馬1: 急落パターン (50→40→30→20→10)
    for i, odds in enumerate([50.0, 40.0, 30.0, 20.0, 10.0]):
        rows.append(
            {
                "race_id": "20240101010101",
                "umaban": "1",
                "happyotime": base_time,
                "tanodds": odds,
                "tanninki": 10 - i,
                "race_date": pd.Timestamp("2024-01-01"),
                "year": 2024,
                "jyocd": 10,
            }
        )
        base_time = f"0105{1200 + i * 10:04d}"  # 10分刻み

    # 馬2: 安定パターン (5.0→4.8→5.0→4.9→5.1)
    base_time2 = "01051200"
    for i, odds in enumerate([5.0, 4.8, 5.0, 4.9, 5.1]):
        rows.append(
            {
                "race_id": "20240101010101",
                "umaban": "2",
                "happyotime": base_time2,
                "tanodds": odds,
                "tanninki": 2,
                "race_date": pd.Timestamp("2024-01-01"),
                "year": 2024,
                "jyocd": 10,
            }
        )
        base_time2 = f"0105{1200 + i * 10:04d}"

    # 馬3: 急騰パターン (3.0→4.0→6.0→8.0→12.0)
    base_time3 = "01051200"
    for i, odds in enumerate([3.0, 4.0, 6.0, 8.0, 12.0]):
        rows.append(
            {
                "race_id": "20240101010101",
                "umaban": "3",
                "happyotime": base_time3,
                "tanodds": odds,
                "tanninki": 1,
                "race_date": pd.Timestamp("2024-01-01"),
                "year": 2024,
                "jyocd": 10,
            }
        )
        base_time3 = f"0105{1200 + i * 10:04d}"

    return pd.DataFrame(rows)


class TestComputeMovementFeatures:
    def test_returns_correct_shape(self, sample_time_series: pd.DataFrame) -> None:
        result = compute_movement_features(sample_time_series)
        assert result.shape[0] == 3  # 3頭
        assert "odds_drop_60_10" in result.columns
        assert "odds_drop_30_10" in result.columns
        assert "n_points" in result.columns

    def test_steamer_detection(self, sample_time_series: pd.DataFrame) -> None:
        """馬1: 50→10 で80%急落"""
        result = compute_movement_features(sample_time_series)
        horse1 = result[result["umaban"] == "1"].iloc[0]
        assert horse1["odds_drop_60_10"] > 0.5  # 50%以上低下
        assert horse1["final_odds"] == 10.0

    def test_stable_horse(self, sample_time_series: pd.DataFrame) -> None:
        """馬2: ほぼ変動なし"""
        result = compute_movement_features(sample_time_series)
        horse2 = result[result["umaban"] == "2"].iloc[0]
        assert abs(horse2["odds_drop_60_10"]) < 0.1  # 10%未満

    def test_drifter_detection(self, sample_time_series: pd.DataFrame) -> None:
        """馬3: 3→12 で急騰（オッズ上昇 = dropが負）"""
        result = compute_movement_features(sample_time_series)
        horse3 = result[result["umaban"] == "3"].iloc[0]
        assert horse3["odds_drop_60_10"] < -0.5  # 50%以上上昇（負のdrop）

    def test_n_points_count(self, sample_time_series: pd.DataFrame) -> None:
        result = compute_movement_features(sample_time_series)
        assert (result["n_points"] == 5).all()

    def test_excludes_nan_odds(self) -> None:
        """NaNオッズの行は除外される"""
        df = pd.DataFrame(
            [
                {
                    "race_id": "r1",
                    "umaban": "1",
                    "happyotime": "01051200",
                    "tanodds": float("nan"),
                    "tanninki": 1,
                    "race_date": pd.Timestamp("2024-01-01"),
                    "year": 2024,
                },
                {
                    "race_id": "r1",
                    "umaban": "1",
                    "happyotime": "01051300",
                    "tanodds": 5.0,
                    "tanninki": 1,
                    "race_date": pd.Timestamp("2024-01-01"),
                    "year": 2024,
                },
            ]
        )
        result = compute_movement_features(df)
        assert len(result) == 1
        assert result.iloc[0]["n_points"] == 1

    def test_excludes_zero_odds(self) -> None:
        """ゼロオッズの行は除外される"""
        df = pd.DataFrame(
            [
                {
                    "race_id": "r1",
                    "umaban": "1",
                    "happyotime": "01051200",
                    "tanodds": 0.0,
                    "tanninki": 1,
                    "race_date": pd.Timestamp("2024-01-01"),
                    "year": 2024,
                },
                {
                    "race_id": "r1",
                    "umaban": "1",
                    "happyotime": "01051300",
                    "tanodds": 5.0,
                    "tanninki": 1,
                    "race_date": pd.Timestamp("2024-01-01"),
                    "year": 2024,
                },
            ]
        )
        result = compute_movement_features(df)
        assert result.iloc[0]["n_points"] == 1

    def test_excludes_nar_races(self) -> None:
        """NARレース(jyocd>=30)は除外"""
        df = pd.DataFrame(
            [
                {
                    "race_id": "r1",
                    "umaban": "1",
                    "happyotime": "01051200",
                    "tanodds": 5.0,
                    "tanninki": 1,
                    "race_date": pd.Timestamp("2024-01-01"),
                    "year": 2024,
                    "jyocd": 35,
                },
                {
                    "race_id": "r1",
                    "umaban": "1",
                    "happyotime": "01051300",
                    "tanodds": 4.0,
                    "tanninki": 1,
                    "race_date": pd.Timestamp("2024-01-01"),
                    "year": 2024,
                    "jyocd": 35,
                },
            ]
        )
        result = compute_movement_features(df)
        assert len(result) == 0
