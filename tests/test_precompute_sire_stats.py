"""precompute_sire_stats: 種牡馬産駒累積統計のテスト"""

import sys
from pathlib import Path

import pandas as pd

# scripts/ を import path に追加
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from precompute_sire_stats import compute_sire_stats  # noqa: E402


def test_precompute_sire_stats_creates_parquet(tmp_path):
    """種牡馬産駒累積統計が正しく計算される"""
    # モックデータ: 2頭の馬、同じ種牡馬
    entries = pd.DataFrame(
        {
            "kettonum": ["001", "002", "001", "002"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"]),
            "race_id": ["R1", "R1", "R2", "R2"],
            "kakuteijyuni": [1, 2, 3, 1],
            "kyori": [1600, 1600, 1800, 1800],
            "trackcd": [11, 11, 11, 11],
            "track_condition_code": [1, 1, 2, 2],
            "honsyokin": [1000, 0, 0, 500],
        }
    )
    horses = pd.DataFrame(
        {
            "kettonum": ["001", "002"],
            "ketto3infohansyokunum1": ["SIRE_A", "SIRE_A"],
        }
    )

    result = compute_sire_stats(entries, horses)

    # 出力列の確認
    assert "sire_id" in result.columns
    assert "sire_starts" in result.columns
    assert "sire_wins" in result.columns
    assert "sire_turf_starts" in result.columns
    assert "sire_short_starts" in result.columns
    assert "sire_prize_total" in result.columns

    # PIT: shift(1).cumsum() — 当日の結果を含まない (horse_career_stats と同じパターン)
    # 1/1: daily_starts=2, daily_wins=1 → shift(1)=0 → cum=0
    row_jan = result[result["race_date"] == "2024-01-01"]
    assert row_jan["sire_starts"].iloc[0] == 0  # 最初のレース前は0
    # 2/1: shift(1)=2 → cum=2
    row_feb = result[result["race_date"] == "2024-02-01"]
    assert row_feb["sire_starts"].iloc[0] == 2  # 1/1の2件
    assert row_feb["sire_wins"].iloc[0] == 1  # 1/1のkettonum=001が1着


def test_sire_stats_multiple_sires():
    """複数の種牡馬が正しく分離されること"""
    entries = pd.DataFrame(
        {
            "kettonum": ["001", "002", "003"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01"]),
            "race_id": ["R1", "R1", "R2"],
            "kakuteijyuni": [1, 2, 1],
            "kyori": [1600, 1600, 1600],
            "trackcd": [11, 11, 11],
            "track_condition_code": [1, 1, 1],
            "honsyokin": [1000, 0, 500],
        }
    )
    horses = pd.DataFrame(
        {
            "kettonum": ["001", "002", "003"],
            "ketto3infohansyokunum1": ["SIRE_A", "SIRE_B", "SIRE_A"],
        }
    )

    result = compute_sire_stats(entries, horses)

    # R2 (2024-02-01) の SIRE_A: R1 で kettonum=001 が1着 → cum_starts=1, cum_wins=1
    row_r2 = result[result["race_date"] == "2024-02-01"]
    assert row_r2["sire_starts"].iloc[0] == 1
    assert row_r2["sire_wins"].iloc[0] == 1


def test_sire_stats_turf_dirt_separation():
    """芝/ダート別統計が正しく計算されること"""
    entries = pd.DataFrame(
        {
            "kettonum": ["001", "001", "001"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "race_id": ["R1", "R2", "R3"],
            "kakuteijyuni": [1, 1, 1],
            "kyori": [1600, 1600, 1600],
            "trackcd": [11, 24, 11],  # turf, dirt, turf
            "track_condition_code": [1, 1, 1],
            "honsyokin": [1000, 2000, 3000],
        }
    )
    horses = pd.DataFrame(
        {
            "kettonum": ["001"],
            "ketto3infohansyokunum1": ["SIRE_A"],
        }
    )

    result = compute_sire_stats(entries, horses)
    result = result.sort_values("race_date")

    # R3 (3/1): 2走前までに turf=1, dirt=1
    row_r3 = result.iloc[2]
    assert row_r3["sire_turf_starts"] == 1
    assert row_r3["sire_dirt_starts"] == 1
    assert row_r3["sire_turf_wins"] == 1
    assert row_r3["sire_dirt_wins"] == 1


def test_sire_stats_prize_cumulative():
    """累積賞金が正しく計算されること"""
    entries = pd.DataFrame(
        {
            "kettonum": ["001", "001", "001"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "race_id": ["R1", "R2", "R3"],
            "kakuteijyuni": [1, 2, 1],
            "kyori": [1600, 1600, 1600],
            "trackcd": [11, 11, 11],
            "track_condition_code": [1, 1, 1],
            "honsyokin": [1000, 200, 500],
        }
    )
    horses = pd.DataFrame(
        {
            "kettonum": ["001"],
            "ketto3infohansyokunum1": ["SIRE_A"],
        }
    )

    result = compute_sire_stats(entries, horses)
    result = result.sort_values("race_date")

    # R1: デビュー → prize=0
    assert result.iloc[0]["sire_prize_total"] == 0
    # R2: R1で1000円 → prize=1000
    assert result.iloc[1]["sire_prize_total"] == 1000
    # R3: R1+R2で1200円 → prize=1200
    assert result.iloc[2]["sire_prize_total"] == 1200


def test_sire_stats_output_columns():
    """出力カラムが全て含まれること"""
    entries = pd.DataFrame(
        {
            "kettonum": ["001"],
            "race_date": pd.to_datetime(["2024-01-01"]),
            "race_id": ["R1"],
            "kakuteijyuni": [1],
            "kyori": [1600],
            "trackcd": [11],
            "track_condition_code": [1],
            "honsyokin": [1000],
        }
    )
    horses = pd.DataFrame(
        {
            "kettonum": ["001"],
            "ketto3infohansyokunum1": ["SIRE_A"],
        }
    )

    result = compute_sire_stats(entries, horses)

    expected_cols = [
        "sire_id",
        "race_date",
        "sire_starts",
        "sire_wins",
        "sire_places",
        "sire_turf_starts",
        "sire_turf_wins",
        "sire_dirt_starts",
        "sire_dirt_wins",
        "sire_short_starts",
        "sire_short_wins",
        "sire_long_starts",
        "sire_long_wins",
        "sire_prize_total",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_sire_stats_nan_sire_excluded():
    """sire_id が NaN の行は除外されること"""
    entries = pd.DataFrame(
        {
            "kettonum": ["001", "002"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "race_id": ["R1", "R1"],
            "kakuteijyuni": [1, 2],
            "kyori": [1600, 1600],
            "trackcd": [11, 11],
            "track_condition_code": [1, 1],
            "honsyokin": [1000, 0],
        }
    )
    horses = pd.DataFrame(
        {
            "kettonum": ["001"],
            "ketto3infohansyokunum1": ["SIRE_A"],
        }
    )

    result = compute_sire_stats(entries, horses)

    # kettonum=002 は sire_id 不明 → 除外
    assert len(result) == 1
