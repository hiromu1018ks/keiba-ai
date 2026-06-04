# tests/test_track_condition_features.py
import pandas as pd
import pytest

from features.track_condition_features import TRACK_CONDITION_COLS, compute_track_condition_features

# ---------------------------------------------------------------------------
# T1-01: dirt_moisture_x_kyakusitu (数値積)
# ---------------------------------------------------------------------------


def test_dirt_moisture_x_kyakusitu():
    """含水率 × 脚質コードの数値積 (ダート用)"""
    df = pd.DataFrame(
        {
            "dirt_moisture": [5.0, 10.0, 15.0],
            "kyakusitukubun_cd": [1.0, 2.0, 3.0],
        }
    )
    result = compute_track_condition_features(df)
    assert "dirt_moisture_x_kyakusitu" in result.columns
    assert result["dirt_moisture_x_kyakusitu"].iloc[0] == pytest.approx(5.0 * 1.0)
    assert result["dirt_moisture_x_kyakusitu"].iloc[1] == pytest.approx(10.0 * 2.0)
    assert result["dirt_moisture_x_kyakusitu"].iloc[2] == pytest.approx(15.0 * 3.0)


def test_dirt_moisture_x_kyakusitu_nan():
    """NaN伝播: いずれかがNaNなら結果もNaN"""
    df = pd.DataFrame(
        {
            "dirt_moisture": [5.0, float("nan"), 15.0],
            "kyakusitukubun_cd": [1.0, 2.0, float("nan")],
        }
    )
    result = compute_track_condition_features(df)
    assert "dirt_moisture_x_kyakusitu" in result.columns
    assert result["dirt_moisture_x_kyakusitu"].iloc[0] == pytest.approx(5.0)
    assert pd.isna(result["dirt_moisture_x_kyakusitu"].iloc[1])
    assert pd.isna(result["dirt_moisture_x_kyakusitu"].iloc[2])


# ---------------------------------------------------------------------------
# T1-02: turf_cushion_track_relative / turf_cushion_track_zscore
# ---------------------------------------------------------------------------


def test_turf_cushion_track_relative():
    """芝クッション値の競馬場別相対値 (track_statsベース)"""
    df = pd.DataFrame(
        {
            "turf_cushion": [9.5, 8.0, 10.0],
            "trackcd": ["05", "05", "09"],
        }
    )
    track_stats = {
        "05": {"mean": 9.0, "std": 0.8},
        "09": {"mean": 10.0, "std": 1.0},
    }
    result = compute_track_condition_features(df, track_stats=track_stats)
    assert "turf_cushion_track_relative" in result.columns
    assert result["turf_cushion_track_relative"].iloc[0] == pytest.approx(9.5 - 9.0)
    assert result["turf_cushion_track_relative"].iloc[1] == pytest.approx(8.0 - 9.0)
    assert result["turf_cushion_track_relative"].iloc[2] == pytest.approx(10.0 - 10.0)


def test_turf_cushion_track_zscore():
    """芝クッション値の競馬場別zscore"""
    df = pd.DataFrame(
        {
            "turf_cushion": [9.5, 8.0, 10.0],
            "trackcd": ["05", "05", "09"],
        }
    )
    track_stats = {
        "05": {"mean": 9.0, "std": 0.8},
        "09": {"mean": 10.0, "std": 1.0},
    }
    result = compute_track_condition_features(df, track_stats=track_stats)
    assert "turf_cushion_track_zscore" in result.columns
    assert result["turf_cushion_track_zscore"].iloc[0] == pytest.approx(0.5 / 0.8)
    assert result["turf_cushion_track_zscore"].iloc[1] == pytest.approx(-1.0 / 0.8)
    assert result["turf_cushion_track_zscore"].iloc[2] == pytest.approx(0.0 / 1.0)


def test_turf_cushion_track_zscore_std_zero():
    """std==0 の場合 zscore は NaN"""
    df = pd.DataFrame(
        {
            "turf_cushion": [9.0, 9.0],
            "trackcd": ["05", "05"],
        }
    )
    track_stats = {"05": {"mean": 9.0, "std": 0.0}}
    result = compute_track_condition_features(df, track_stats=track_stats)
    assert "turf_cushion_track_zscore" in result.columns
    assert pd.isna(result["turf_cushion_track_zscore"].iloc[0])


def test_turf_cushion_features_without_track_stats():
    """track_stats が None の場合、T1-02特徴量は生成されない"""
    df = pd.DataFrame(
        {
            "turf_cushion": [9.5, 8.0],
            "trackcd": ["05", "05"],
        }
    )
    result = compute_track_condition_features(df, track_stats=None)
    assert "turf_cushion_track_relative" not in result.columns
    assert "turf_cushion_track_zscore" not in result.columns


def test_turf_cushion_features_nan_propagation():
    """turf_cushion が NaN の場合、relative/zscore も NaN"""
    df = pd.DataFrame(
        {
            "turf_cushion": [float("nan"), 8.0],
            "trackcd": ["05", "05"],
        }
    )
    track_stats = {"05": {"mean": 9.0, "std": 0.8}}
    result = compute_track_condition_features(df, track_stats=track_stats)
    assert "turf_cushion_track_relative" in result.columns
    assert pd.isna(result["turf_cushion_track_relative"].iloc[0])
    assert result["turf_cushion_track_relative"].iloc[1] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# T2-01: dirt_moisture_x_barrier_pos + flags
# ---------------------------------------------------------------------------


def test_dirt_moisture_x_barrier_pos():
    """含水率 × 枠番の数値積"""
    df = pd.DataFrame(
        {
            "dirt_moisture": [5.0, 10.0],
            "frame_number": [1.0, 8.0],
        }
    )
    result = compute_track_condition_features(df)
    assert "dirt_moisture_x_barrier_pos" in result.columns
    assert result["dirt_moisture_x_barrier_pos"].iloc[0] == pytest.approx(5.0 * 1.0)
    assert result["dirt_moisture_x_barrier_pos"].iloc[1] == pytest.approx(10.0 * 8.0)


def test_dirt_moisture_x_barrier_pos_nan():
    """NaN伝播: いずれかがNaNならNaN"""
    df = pd.DataFrame(
        {
            "dirt_moisture": [5.0, float("nan")],
            "frame_number": [1.0, 8.0],
        }
    )
    result = compute_track_condition_features(df)
    assert pd.isna(result["dirt_moisture_x_barrier_pos"].iloc[1])


def test_dirt_moisture_high_flag():
    """含水率 > 12 で high_flag = 1.0"""
    df = pd.DataFrame(
        {
            "dirt_moisture": [5.0, 15.0, float("nan")],
        }
    )
    result = compute_track_condition_features(df)
    assert "dirt_moisture_high_flag" in result.columns
    assert result["dirt_moisture_high_flag"].iloc[0] == 0.0
    assert result["dirt_moisture_high_flag"].iloc[1] == 1.0
    assert pd.isna(result["dirt_moisture_high_flag"].iloc[2])


def test_dirt_moisture_dry_flag():
    """含水率 < 3 で dry_flag = 1.0"""
    df = pd.DataFrame(
        {
            "dirt_moisture": [2.0, 5.0, float("nan")],
        }
    )
    result = compute_track_condition_features(df)
    assert "dirt_moisture_dry_flag" in result.columns
    assert result["dirt_moisture_dry_flag"].iloc[0] == 1.0
    assert result["dirt_moisture_dry_flag"].iloc[1] == 0.0
    assert pd.isna(result["dirt_moisture_dry_flag"].iloc[2])


# ---------------------------------------------------------------------------
# T2-02: turf_cushion_x_kyakusitu (数値積)
# ---------------------------------------------------------------------------


def test_turf_cushion_x_kyakusitu():
    """芝クッション値 × 脚質コードの数値積"""
    df = pd.DataFrame(
        {
            "turf_cushion": [9.0, 8.5, 10.0],
            "kyakusitukubun_cd": [1.0, 2.0, 3.0],
        }
    )
    result = compute_track_condition_features(df)
    assert "turf_cushion_x_kyakusitu" in result.columns
    assert result["turf_cushion_x_kyakusitu"].iloc[0] == pytest.approx(9.0 * 1.0)
    assert result["turf_cushion_x_kyakusitu"].iloc[1] == pytest.approx(8.5 * 2.0)


def test_turf_cushion_x_kyakusitu_nan():
    """NaN伝播: いずれかがNaNならNaN"""
    df = pd.DataFrame(
        {
            "turf_cushion": [9.0, float("nan")],
            "kyakusitukubun_cd": [1.0, 2.0],
        }
    )
    result = compute_track_condition_features(df)
    assert pd.isna(result["turf_cushion_x_kyakusitu"].iloc[1])


# ---------------------------------------------------------------------------
# T2-03: sire_x_cushion_band (カテゴリ積)
# ---------------------------------------------------------------------------


def test_sire_x_cushion_band():
    """種牡馬 × 5段階クッションビン → category型"""
    df = pd.DataFrame(
        {
            "sire_id": ["S001", "S002", "S003"],
            "turf_cushion": [6.5, 8.5, 10.5],
        }
    )
    result = compute_track_condition_features(df)
    assert "sire_x_cushion_band" in result.columns
    assert result["sire_x_cushion_band"].dtype.name == "category"
    assert result["sire_x_cushion_band"].iloc[0] == "S001_very_soft"
    assert result["sire_x_cushion_band"].iloc[1] == "S002_standard"
    assert result["sire_x_cushion_band"].iloc[2] == "S003_very_firm"


def test_sire_x_cushion_band_nan_cushion():
    """turf_cushion NaN → cushion_band NaN → interaction も NaN"""
    df = pd.DataFrame(
        {
            "sire_id": ["S001", "S002"],
            "turf_cushion": [8.5, float("nan")],
        }
    )
    result = compute_track_condition_features(df)
    assert "sire_x_cushion_band" in result.columns
    # NaN cushion → NaN interaction
    assert pd.isna(result["sire_x_cushion_band"].iloc[1])


def test_sire_x_cushion_band_bin_boundaries():
    """ビン境界値: [0,7]=very_soft, (7,8]=soft, (8,9]=standard, (9,10]=firm, (10,inf]=very_firm"""
    df = pd.DataFrame(
        {
            "sire_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "turf_cushion": [0.1, 7.0, 8.0, 9.0, 10.0, 11.0],
        }
    )
    result = compute_track_condition_features(df)
    cats = result["sire_x_cushion_band"].tolist()
    # 7.0 → (0,7] → very_soft (right=True)
    assert cats[0] == "S1_very_soft"
    # pd.cut right=True: bins=[0,7,8,9,10,inf] means (0,7], (7,8], (8,9], (9,10], (10,inf]
    assert cats[0] == "S1_very_soft"   # 0.1 in (0,7]
    assert cats[1] == "S2_very_soft"   # 7.0 in (0,7] (right edge included)
    assert cats[2] == "S3_soft"        # 8.0 in (7,8]
    assert cats[3] == "S4_standard"    # 9.0 in (8,9]
    assert cats[4] == "S5_firm"        # 10.0 in (9,10]
    assert cats[5] == "S6_very_firm"   # 11.0 in (10,inf]


# ---------------------------------------------------------------------------
# 汎用テスト: 欠損列ガード / NaN伝播 / 定数カウント
# ---------------------------------------------------------------------------


def test_missing_columns():
    """必要列がない場合は特徴量を追加しない"""
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    result = compute_track_condition_features(df)
    for col in TRACK_CONDITION_COLS:
        assert col not in result.columns


def test_track_condition_cols_constant():
    """TRACK_CONDITION_COLS定数が8個の特徴量名を含む"""
    assert len(TRACK_CONDITION_COLS) == 8
    expected = [
        "dirt_moisture_x_kyakusitu",
        "turf_cushion_track_relative",
        "turf_cushion_track_zscore",
        "dirt_moisture_x_barrier_pos",
        "dirt_moisture_high_flag",
        "dirt_moisture_dry_flag",
        "turf_cushion_x_kyakusitu",
        "sire_x_cushion_band",
    ]
    for feat in expected:
        assert feat in TRACK_CONDITION_COLS, f"Missing: {feat}"


def test_does_not_modify_input():
    """元のDataFrameは変更されない"""
    df = pd.DataFrame(
        {
            "dirt_moisture": [5.0, 10.0],
            "kyakusitukubun_cd": [1.0, 2.0],
        }
    )
    original_cols = list(df.columns)
    _ = compute_track_condition_features(df)
    assert list(df.columns) == original_cols


def test_compute_track_stats():
    """_compute_track_stats: trackcd別のmean/stdを計算"""
    from features.track_condition_features import _compute_track_stats

    df = pd.DataFrame(
        {
            "turf_cushion": [9.0, 9.5, 8.5, 10.0, 10.5],
            "trackcd": ["05", "05", "05", "09", "09"],
        }
    )
    stats = _compute_track_stats(df)
    assert "05" in stats
    assert "09" in stats
    assert abs(stats["05"]["mean"] - 9.0) < 1e-6
    assert abs(stats["09"]["mean"] - 10.25) < 1e-6
    assert stats["05"]["std"] > 0
