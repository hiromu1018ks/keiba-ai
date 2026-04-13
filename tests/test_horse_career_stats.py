"""horse_career_stats: point-in-time 累積成績のテスト"""

import pandas as pd
import pytest

from features.horse_career_stats import precompute_career_stats


@pytest.fixture
def sample_data():
    """3頭の馬 × 数レースのテストデータ"""
    entries = pd.DataFrame(
        {
            "race_id": ["20250101A01", "20250115A01", "20250201A01", "20250101A02", "20250101A03"],
            "kettonum": ["H001", "H001", "H001", "H002", "H003"],
            "kakuteijyuni": pd.array([1, 3, 2, 5, pd.NA], dtype="Int64"),
            "honsyokin": pd.array([50000, 10000, 20000, 0, pd.NA], dtype="Int64"),
            "race_date": pd.to_datetime(
                ["2025-01-01", "2025-01-15", "2025-02-01", "2025-01-01", "2025-01-01"]
            ),
            "jyocd": pd.array([5, 5, 5, 5, 5], dtype="Int64"),
        }
    )
    races = pd.DataFrame(
        {
            "race_id": ["20250101A01", "20250115A01", "20250201A01", "20250101A02", "20250101A03"],
            "trackcd": pd.array([17, 17, 24, 17, 24], dtype="Int64"),
            "kyori": pd.array([1600, 1200, 1800, 1400, 1200], dtype="Int64"),
        }
    )
    return entries, races


def test_total_stats_cumulative(sample_data):
    """累積勝利数・出走数が正しいことを確認"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    h001 = result[result["kettonum"] == "H001"].sort_values("race_date")

    # 1レース目: デビュー → cum_starts=0, cum_wins=0
    first = h001.iloc[0]
    assert first["cum_starts"] == 0
    assert first["cum_wins"] == 0

    # 2レース目: 1戦1勝 → cum_starts=1, cum_wins=1
    second = h001.iloc[1]
    assert second["cum_starts"] == 1
    assert second["cum_wins"] == 1

    # 3レース目: 2戦1勝(1着1回,3着1回) → cum_starts=2, cum_wins=1
    third = h001.iloc[2]
    assert third["cum_starts"] == 2
    assert third["cum_wins"] == 1


def test_debut_horse_zero_starts(sample_data):
    """デビュー馬は cum_starts=0 であること"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    h003 = result[result["kettonum"] == "H003"]
    assert len(h003) == 1
    assert h003.iloc[0]["cum_starts"] == 0
    assert h003.iloc[0]["cum_wins"] == 0


def test_prize_cumulative(sample_data):
    """累積賞金が正しいことを確認"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    h001 = result[result["kettonum"] == "H001"].sort_values("race_date")

    # 1レース目: デビュー → cum_prize=0
    assert h001.iloc[0]["cum_prize"] == 0

    # 2レース目: 前走で50000円獲得 → cum_prize=50000
    assert h001.iloc[1]["cum_prize"] == 50000

    # 3レース目: 前走までに60000円獲得 → cum_prize=60000
    assert h001.iloc[2]["cum_prize"] == 60000


def test_surface_specific_stats(sample_data):
    """芝/ダート別の累積成績が正しいこと"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    h001 = result[result["kettonum"] == "H001"].sort_values("race_date")

    # 1レース目: 芝(trackcd=17) → cum_turf_starts=0, cum_dirt_starts=0
    first = h001.iloc[0]
    assert first["cum_turf_starts"] == 0

    # 3レース目: ダート(trackcd=24), 前に芝2戦 → cum_turf_starts=2, cum_dirt_starts=0
    third = h001.iloc[2]
    assert third["cum_turf_starts"] == 2
    assert third["cum_dirt_starts"] == 0


def test_distance_specific_stats(sample_data):
    """芝1600以下の累積成績が正しいこと (kyori1 近似)"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    h001 = result[result["kettonum"] == "H001"].sort_values("race_date")

    # 1レース目: 芝1600m → 条件該当, でもデビューなので cum_short_starts=0
    first = h001.iloc[0]
    assert first["cum_short_starts"] == 0

    # 2レース目: 芝1200m, 前走は芝1600m(該当) → cum_short_starts=1, cum_short_wins=1
    second = h001.iloc[1]
    assert second["cum_short_starts"] == 1
    assert second["cum_short_wins"] == 1

    # 3レース目: ダート1800m, 前走は芝1200m(該当) → cum_short_starts=2, cum_short_wins=1
    third = h001.iloc[2]
    assert third["cum_short_starts"] == 2
    assert third["cum_short_wins"] == 1


def test_condition_columns_in_precompute():
    """baba_cd別の累積成績が正しく計算される"""
    # entries と races は別々のDataFrame
    entries = pd.DataFrame({
        "kettonum": ["001", "001", "001"],
        "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "race_id": ["R1", "R2", "R3"],
        "kakuteijyuni": [1, 3, 2],
        "jyocd": [5, 5, 5],
        "honsyokin": [1000, 0, 500],
    })
    # races に track_condition_code を含める
    races = pd.DataFrame({
        "race_id": ["R1", "R2", "R3"],
        "trackcd": [11, 11, 11],
        "kyori": [1600, 1800, 1600],
        "track_condition_code": [1, 3, 2],  # good, heavy, good
    })

    result = precompute_career_stats(entries, races)

    # 条件別累積列が存在する
    for col in ["cum_turf_good_starts", "cum_turf_good_wins",
                "cum_turf_heavy_starts", "cum_turf_heavy_wins",
                "cum_dirt_good_starts", "cum_dirt_good_wins",
                "cum_dirt_heavy_starts", "cum_dirt_heavy_wins"]:
        assert col in result.columns, f"Missing column: {col}"
    # PIT: shift(1)→cumsum により当日の結果は含まれない
    row0 = result.iloc[0]
    assert row0["cum_turf_good_starts"] == 0  # 最初のレース前は0
    # 2走目: R1はturf+good で1回出走している → cum_turf_good_starts=1
    row1 = result.iloc[1]
    assert row1["cum_turf_good_starts"] == 1


def test_output_columns(sample_data):
    """出力に必要なカラムが全て含まれること"""
    entries, races = sample_data
    result = precompute_career_stats(entries, races)

    expected_cols = [
        "race_id",
        "kettonum",
        "race_date",
        "cum_starts",
        "cum_wins",
        "cum_prize",
        "cum_turf_starts",
        "cum_turf_wins",
        "cum_dirt_starts",
        "cum_dirt_wins",
        "cum_short_starts",
        "cum_short_wins",
        "cum_turf_good_starts",
        "cum_turf_good_wins",
        "cum_turf_heavy_starts",
        "cum_turf_heavy_wins",
        "cum_dirt_good_starts",
        "cum_dirt_good_wins",
        "cum_dirt_heavy_starts",
        "cum_dirt_heavy_wins",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"
