"""tests/test_analyze_odds_movement.py — オッズ変動分析のユニットテスト"""

import pandas as pd
import pytest

from scripts.analyze_odds_movement import classify_movement, compute_movement_features, join_results


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


class TestClassifyMovement:
    def test_steamer_classification(self):
        df = pd.DataFrame(
            {
                "race_id": ["r1"],
                "umaban": ["1"],
                "odds_drop_60_10": [0.7],
                "odds_drop_30_10": [0.5],
                "odds_drop_10_final": [0.2],
                "pop_change_30_10": [3],
                "n_points": [10],
                "early_odds": [50.0],
                "final_odds": [15.0],
            }
        )
        result = classify_movement(df, threshold=0.20)
        assert result.iloc[0]["movement_class"] == "steamer"
        assert result.iloc[0]["movement_bucket"] == "strong_drop"

    def test_drifter_classification(self):
        df = pd.DataFrame(
            {
                "race_id": ["r1"],
                "umaban": ["1"],
                "odds_drop_60_10": [-0.7],
                "odds_drop_30_10": [-0.5],
                "odds_drop_10_final": [-0.2],
                "pop_change_30_10": [-3],
                "n_points": [10],
                "early_odds": [5.0],
                "final_odds": [15.0],
            }
        )
        result = classify_movement(df, threshold=0.20)
        assert result.iloc[0]["movement_class"] == "drifter"
        assert result.iloc[0]["movement_bucket"] == "strong_rise"

    def test_stable_classification(self):
        df = pd.DataFrame(
            {
                "race_id": ["r1"],
                "umaban": ["1"],
                "odds_drop_60_10": [0.05],
                "odds_drop_30_10": [0.03],
                "odds_drop_10_final": [0.02],
                "pop_change_30_10": [0],
                "n_points": [10],
                "early_odds": [5.0],
                "final_odds": [4.9],
            }
        )
        result = classify_movement(df, threshold=0.20)
        assert result.iloc[0]["movement_class"] == "stable"
        assert result.iloc[0]["movement_bucket"] == "stable"

    def test_custom_threshold(self):
        df = pd.DataFrame(
            {
                "race_id": ["r1"],
                "umaban": ["1"],
                "odds_drop_60_10": [0.18],
                "odds_drop_30_10": [0.18],
                "odds_drop_10_final": [0.05],
                "pop_change_30_10": [1],
                "n_points": [10],
                "early_odds": [10.0],
                "final_odds": [8.2],
            }
        )
        # threshold=0.20 → stable
        result_loose = classify_movement(df, threshold=0.20)
        assert result_loose.iloc[0]["movement_class"] == "stable"
        # threshold=0.15 → steamer
        result_tight = classify_movement(df, threshold=0.15)
        assert result_tight.iloc[0]["movement_class"] == "steamer"


class TestJoinResults:
    @pytest.fixture
    def sample_joined_data(self):
        """join_results 用のモックデータ"""
        movement = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r2"],
                "umaban": ["1", "2", "1"],
                "odds_drop_30_10": [0.3, -0.1, 0.5],
                "n_points": [10, 10, 10],
                "final_odds": [15.0, 5.0, 8.0],
                "movement_class": ["steamer", "stable", "steamer"],
                "movement_bucket": ["moderate_drop", "stable", "strong_drop"],
            }
        )

        entries = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r2"],
                "umaban": [1, 2, 1],
                "kakuteijyuni": [1, 4, 3],
                "ninki": [1, 5, 3],
                "kisyucode": ["00001", "00002", "00001"],
                "chokyosicode": ["A001", "A002", "A001"],
            }
        )

        races = pd.DataFrame(
            {
                "race_id": ["r1", "r2"],
                "kyori": [1800, 1200],
                "syussotosu": [16, 10],
                "trackcd": [10, 23],  # 10=芝(turf), 23=ダート(dirt)
            }
        )

        payouts = pd.DataFrame(
            {
                "race_id": ["r1", "r2"],
                "payfukusyoumaban1": [1, 1],
                "payfukusyoumaban2": [3, 2],
                "payfukusyoumaban3": [pd.NA, pd.NA],
                "payfukusyopay1": [120.0, 80.0],
                "payfukusyopay2": [40.0, 30.0],
                "payfukusyopay3": [pd.NA, pd.NA],
            }
        )

        return movement, entries, races, payouts

    def test_place_detection_win(self, sample_joined_data):
        mov, ent, rac, pay = sample_joined_data
        result = join_results(mov, ent, rac, pay)
        # r1-umaban1: 1着 → 複勝的中
        r1_h1 = result[(result["race_id"] == "r1") & (result["umaban"] == "1")].iloc[0]
        assert r1_h1["is_place"] == 1
        assert r1_h1["place_payout"] == 120.0

    def test_place_detection_third(self, sample_joined_data):
        mov, ent, rac, pay = sample_joined_data
        result = join_results(mov, ent, rac, pay)
        # r2-umaban1: 3着 → 複勝的中
        r2_h1 = result[(result["race_id"] == "r2") & (result["umaban"] == "1")].iloc[0]
        assert r2_h1["is_place"] == 1
        assert r2_h1["place_payout"] == 80.0

    def test_no_place_fourth(self, sample_joined_data):
        mov, ent, rac, pay = sample_joined_data
        result = join_results(mov, ent, rac, pay)
        # r1-umaban2: 4着 → 複勝外
        r1_h2 = result[(result["race_id"] == "r1") & (result["umaban"] == "2")].iloc[0]
        assert r1_h2["is_place"] == 0
        assert r1_h2["place_payout"] == 0.0

    def test_surface_mapping(self, sample_joined_data):
        mov, ent, rac, pay = sample_joined_data
        result = join_results(mov, ent, rac, pay)
        r1 = result[result["race_id"] == "r1"].iloc[0]  # trackcd=10 → turf
        r2 = result[result["race_id"] == "r2"].iloc[0]  # trackcd=23 → dirt
        assert r1["surface"] == "turf"
        assert r2["surface"] == "dirt"

    def test_min_points_filter(self, sample_joined_data):
        mov, ent, rac, pay = sample_joined_data
        # 1頭だけ n_points=3 にしてフィルタされるか確認
        mov.loc[mov.index[0], "n_points"] = 3
        result = join_results(mov, ent, rac, pay, min_points=5)
        assert len(result) == 2  # 3 pointsの馬が除外
