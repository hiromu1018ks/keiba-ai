# tests/test_track_condition_features.py
import pandas as pd
import pytest

from features.track_condition_features import (
    RACE_CONDITION_COLS,
    TRACK_CONDITION_COLS,
    TRACK_DERIVED_COLS,
    compute_race_condition_features,
    compute_track_condition_features,
)

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


# ---------------------------------------------------------------------------
# 外科的ルーティング検証 (Phase 48, D-04/D-05)
# ---------------------------------------------------------------------------


def test_surgical_routing_included_models_have_track_condition_features():
    """外科的ルーティング: 対象モデルのFEATURE_COLSに8個のトラック条件特徴量が含まれる (D-04)"""
    from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel
    from models.place_ability_model import PlaceAbilityModel
    from models.stage1_ability_model import AbilityModel
    from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
    from models.wide_two_stage_model import WideTwoStageModel

    included_models = {
        "AbilityModel": AbilityModel.FEATURE_COLS,
        "WinTwoStageModel": WinTwoStageModel.FEATURE_COLS,
        "PlaceTwoStageModel.HIT": PlaceTwoStageModel.HIT_FEATURE_COLS,
        "PlaceTwoStageModel.RETURN": PlaceTwoStageModel.RETURN_FEATURE_COLS,
        "PlaceTwoStageModel.FEATURE": PlaceTwoStageModel.FEATURE_COLS,
        "EVCorrectionModel": EVCorrectionModel.FEATURE_COLS,
        "PlaceEVCorrectionModel": PlaceEVCorrectionModel.FEATURE_COLS,
        "PlaceAbilityModel": PlaceAbilityModel.FEATURE_COLS,
        "WideTwoStageModel.SHARED": WideTwoStageModel.SHARED_FEATURE_COLS,
    }
    for model_name, cols in included_models.items():
        for feat in TRACK_CONDITION_COLS:
            assert feat in cols, f"{model_name} missing track condition feature: {feat}"


def test_surgical_routing_excluded_models():
    """外科的ルーティング: 除外モデルにトラック条件特徴量は含まれない (D-05)"""
    from models.conformal_ev_model import ConformalEVModel
    from models.market_model import MarketModel
    from models.race_quality_screener import RaceQualityScreener
    from models.regime_detector import RegimeDetector

    excluded_models = {
        "MarketModel": MarketModel.FEATURE_COLS,
        "RaceQualityScreener": RaceQualityScreener.FEATURE_COLS,
        "RegimeDetector": RegimeDetector.FEATURE_COLS,
        "ConformalEVModel": ConformalEVModel.FEATURE_COLS,
    }
    for model_name, cols in excluded_models.items():
        for feat in TRACK_CONDITION_COLS:
            assert feat not in cols, f"{model_name} should NOT have track condition feature: {feat}"


# ---------------------------------------------------------------------------
# T4-01: track_front_bias_score / kickback_risk_score / expected_pace_class
# ---------------------------------------------------------------------------


def test_dirt_front_bias_high_moisture():
    """ダート含水率15 -> front_bias = clip((15-3)/(12-3), 0, 1) = 1.0"""
    df = pd.DataFrame({"dirt_moisture": [15.0], "trackcd": ["23"]})
    result = compute_track_condition_features(df)
    assert "track_front_bias_score" in result.columns
    assert result["track_front_bias_score"].iloc[0] == pytest.approx(1.0)


def test_dirt_front_bias_low_moisture():
    """ダート含水率1 -> front_bias = clip((1-3)/9, 0, 1) = 0.0"""
    df = pd.DataFrame({"dirt_moisture": [1.0], "trackcd": ["23"]})
    result = compute_track_condition_features(df)
    assert result["track_front_bias_score"].iloc[0] == pytest.approx(0.0)


def test_turf_front_bias_mid_cushion():
    """芝クッション9 -> front_bias = clip((9-8)/(10-8), 0, 1) = 0.5"""
    df = pd.DataFrame({"turf_cushion": [9.0], "trackcd": ["05"]})
    result = compute_track_condition_features(df)
    assert result["track_front_bias_score"].iloc[0] == pytest.approx(0.5)


def test_front_bias_nan_propagation():
    """NaN moisture -> NaN front_bias"""
    df = pd.DataFrame({"dirt_moisture": [float("nan")], "trackcd": ["23"]})
    result = compute_track_condition_features(df)
    assert pd.isna(result["track_front_bias_score"].iloc[0])


def test_dirt_kickback_dry():
    """ダート含水率1 -> kickback = clip((12-1)/9, 0, 1) = 1.22 -> clip = 1.0"""
    df = pd.DataFrame({"dirt_moisture": [1.0], "trackcd": ["23"]})
    result = compute_track_condition_features(df)
    assert "kickback_risk_score" in result.columns
    assert result["kickback_risk_score"].iloc[0] == pytest.approx(1.0)


def test_dirt_kickback_wet():
    """ダート含水率15 -> kickback = clip((12-15)/9, 0, 1) = 0.0"""
    df = pd.DataFrame({"dirt_moisture": [15.0], "trackcd": ["23"]})
    result = compute_track_condition_features(df)
    assert result["kickback_risk_score"].iloc[0] == pytest.approx(0.0)


def test_kickback_nan_propagation():
    """NaN moisture -> NaN kickback"""
    df = pd.DataFrame({"dirt_moisture": [float("nan")], "trackcd": ["23"]})
    result = compute_track_condition_features(df)
    assert pd.isna(result["kickback_risk_score"].iloc[0])


def test_pace_class_slow():
    """high front_bias + low kickback -> slow (0)"""
    df = pd.DataFrame({"dirt_moisture": [15.0], "trackcd": ["23"]})  # bias=1.0, kickback=0.0
    result = compute_track_condition_features(df)
    assert "expected_pace_class" in result.columns
    assert result["expected_pace_class"].iloc[0] == pytest.approx(0.0)


def test_pace_class_fast():
    """low front_bias + high kickback -> fast (2)"""
    df = pd.DataFrame({"dirt_moisture": [1.0], "trackcd": ["23"]})  # bias=0.0, kickback=1.0
    result = compute_track_condition_features(df)
    assert result["expected_pace_class"].iloc[0] == pytest.approx(2.0)


def test_pace_class_neutral():
    """middle values -> neutral (1)"""
    df = pd.DataFrame(
        {"dirt_moisture": [7.5], "trackcd": ["23"]},
    )  # bias=0.5, kickback=0.5
    result = compute_track_condition_features(df)
    assert result["expected_pace_class"].iloc[0] == pytest.approx(1.0)


def test_pace_class_nan():
    """NaN moisture -> NaN pace_class"""
    df = pd.DataFrame({"dirt_moisture": [float("nan")], "trackcd": ["23"]})
    result = compute_track_condition_features(df)
    assert pd.isna(result["expected_pace_class"].iloc[0])


# ---------------------------------------------------------------------------
# T3-04: season deviation (_compute_track_month_stats)
# ---------------------------------------------------------------------------


def test_compute_track_month_stats_basic():
    """_compute_track_month_stats: trackcd x month level stats"""
    from features.track_condition_features import _compute_track_month_stats

    df = pd.DataFrame(
        {
            "turf_cushion": [9.0, 9.5, 8.5, 10.0, 10.5, 11.0],
            "dirt_moisture": [5.0, 6.0, 4.0, 8.0, 7.0, 9.0],
            "trackcd": ["05", "05", "05", "09", "09", "09"],
            "race_date": pd.to_datetime(
                ["2024-01-15", "2024-01-20", "2024-02-10", "2024-01-05", "2024-01-25", "2024-02-15"]
            ),
        }
    )
    stats = _compute_track_month_stats(df)
    # 05_1 has 2 values (9.0, 9.5)
    assert "05_1" in stats
    assert abs(stats["05_1"]["cushion_mean"] - 9.25) < 1e-6
    assert stats["05_1"]["cushion_std"] > 0
    assert abs(stats["05_1"]["moisture_mean"] - 5.5) < 1e-6
    # 05_2 has only 1 value -> should not be in stats (requires >= 2)
    assert "05_2" not in stats or stats.get("05_2", {}).get("cushion_std", 1) == 0


def test_cushion_season_deviation():
    """cushion_season_deviation = (cushion - track_month_mean) / track_month_std"""
    df = pd.DataFrame(
        {
            "turf_cushion": [10.0],
            "trackcd": ["05"],
            "race_date": pd.to_datetime(["2024-01-15"]),
        }
    )
    track_month_stats = {
        "05_1": {
            "cushion_mean": 9.0, "cushion_std": 0.5,
            "moisture_mean": 5.0, "moisture_std": 1.0,
        },
    }
    result = compute_track_condition_features(df, track_month_stats=track_month_stats)
    assert "cushion_season_deviation" in result.columns
    assert result["cushion_season_deviation"].iloc[0] == pytest.approx((10.0 - 9.0) / 0.5)


def test_season_deviation_std_zero():
    """std==0 -> season_deviation is NaN"""
    df = pd.DataFrame(
        {
            "turf_cushion": [9.0],
            "trackcd": ["05"],
            "race_date": pd.to_datetime(["2024-01-15"]),
        }
    )
    track_month_stats = {
        "05_1": {
            "cushion_mean": 9.0, "cushion_std": 0.0,
            "moisture_mean": 5.0, "moisture_std": 1.0,
        },
    }
    result = compute_track_condition_features(df, track_month_stats=track_month_stats)
    assert pd.isna(result["cushion_season_deviation"].iloc[0])


def test_season_deviation_no_stats():
    """track_month_stats is None -> no season deviation features"""
    df = pd.DataFrame(
        {
            "turf_cushion": [9.0],
            "trackcd": ["05"],
            "race_date": pd.to_datetime(["2024-01-15"]),
        }
    )
    result = compute_track_condition_features(df, track_month_stats=None)
    assert "cushion_season_deviation" not in result.columns
    assert "moisture_season_deviation" not in result.columns


# ---------------------------------------------------------------------------
# T4-03: anomaly flags
# ---------------------------------------------------------------------------


def test_cushion_anomaly_flag_triggered():
    """|season_deviation| > 2 -> flag = 1.0"""
    df = pd.DataFrame(
        {
            "turf_cushion": [12.0],
            "trackcd": ["05"],
            "race_date": pd.to_datetime(["2024-01-15"]),
        }
    )
    track_month_stats = {
        "05_1": {
            "cushion_mean": 9.0, "cushion_std": 1.0,
            "moisture_mean": 5.0, "moisture_std": 1.0,
        },
    }
    result = compute_track_condition_features(df, track_month_stats=track_month_stats)
    assert "cushion_anomaly_flag" in result.columns
    # deviation = (12-9)/1 = 3.0, |3.0| > 2 -> 1.0
    assert result["cushion_anomaly_flag"].iloc[0] == pytest.approx(1.0)


def test_cushion_anomaly_flag_normal():
    """|season_deviation| <= 2 -> flag = 0.0"""
    df = pd.DataFrame(
        {
            "turf_cushion": [10.0],
            "trackcd": ["05"],
            "race_date": pd.to_datetime(["2024-01-15"]),
        }
    )
    track_month_stats = {
        "05_1": {
            "cushion_mean": 9.0, "cushion_std": 1.0,
            "moisture_mean": 5.0, "moisture_std": 1.0,
        },
    }
    result = compute_track_condition_features(df, track_month_stats=track_month_stats)
    # deviation = (10-9)/1 = 1.0, |1.0| <= 2 -> 0.0
    assert result["cushion_anomaly_flag"].iloc[0] == pytest.approx(0.0)


def test_anomaly_flag_nan_propagation():
    """NaN season_deviation -> NaN flag"""
    df = pd.DataFrame(
        {
            "turf_cushion": [float("nan")],
            "trackcd": ["05"],
            "race_date": pd.to_datetime(["2024-01-15"]),
        }
    )
    track_month_stats = {
        "05_1": {
            "cushion_mean": 9.0, "cushion_std": 1.0,
            "moisture_mean": 5.0, "moisture_std": 1.0,
        },
    }
    result = compute_track_condition_features(df, track_month_stats=track_month_stats)
    assert pd.isna(result["cushion_anomaly_flag"].iloc[0])


# ---------------------------------------------------------------------------
# T4-04: interactions (3 numeric products + surface_condition_transition)
# ---------------------------------------------------------------------------


def test_cushion_x_distance():
    """cushion_x_distance = turf_cushion * kyori"""
    df = pd.DataFrame({"turf_cushion": [9.0], "kyori": [1600.0], "trackcd": ["05"]})
    result = compute_track_condition_features(df)
    assert "cushion_x_distance" in result.columns
    assert result["cushion_x_distance"].iloc[0] == pytest.approx(9.0 * 1600.0)


def test_cushion_x_distance_nan():
    """NaN propagation for cushion_x_distance"""
    df = pd.DataFrame({"turf_cushion": [float("nan")], "kyori": [1600.0], "trackcd": ["05"]})
    result = compute_track_condition_features(df)
    assert pd.isna(result["cushion_x_distance"].iloc[0])


def test_moisture_x_weight():
    """moisture_x_weight = dirt_moisture * bataijyu"""
    df = pd.DataFrame({"dirt_moisture": [10.0], "bataijyu": [500.0], "trackcd": ["23"]})
    result = compute_track_condition_features(df)
    assert "moisture_x_weight" in result.columns
    assert result["moisture_x_weight"].iloc[0] == pytest.approx(10.0 * 500.0)


def test_moisture_x_weight_nan():
    """NaN propagation for moisture_x_weight"""
    df = pd.DataFrame({"dirt_moisture": [10.0], "bataijyu": [float("nan")], "trackcd": ["23"]})
    result = compute_track_condition_features(df)
    assert pd.isna(result["moisture_x_weight"].iloc[0])


def test_cushion_x_age():
    """cushion_x_age = turf_cushion * barei"""
    df = pd.DataFrame({"turf_cushion": [9.0], "barei": [5.0], "trackcd": ["05"]})
    result = compute_track_condition_features(df)
    assert "cushion_x_age" in result.columns
    assert result["cushion_x_age"].iloc[0] == pytest.approx(9.0 * 5.0)


def test_cushion_x_age_nan():
    """NaN propagation for cushion_x_age"""
    df = pd.DataFrame({"turf_cushion": [9.0], "barei": [float("nan")], "trackcd": ["05"]})
    result = compute_track_condition_features(df)
    assert pd.isna(result["cushion_x_age"].iloc[0])


def test_surface_condition_transition_dirt():
    """dirt: surface_condition_transition = dirt_moisture - prev_dirt_moisture"""
    df = pd.DataFrame(
        {
            "dirt_moisture": [10.0],
            "prev_dirt_moisture": [7.0],
            "trackcd": ["23"],
        }
    )
    result = compute_track_condition_features(df)
    assert "surface_condition_transition" in result.columns
    assert result["surface_condition_transition"].iloc[0] == pytest.approx(3.0)


def test_surface_condition_transition_turf():
    """turf: surface_condition_transition = turf_cushion - prev_turf_cushion"""
    df = pd.DataFrame(
        {
            "turf_cushion": [10.0],
            "prev_turf_cushion": [8.5],
            "trackcd": ["05"],
        }
    )
    result = compute_track_condition_features(df)
    assert result["surface_condition_transition"].iloc[0] == pytest.approx(1.5)


def test_surface_condition_transition_nan_prev():
    """NaN prev -> NaN transition"""
    df = pd.DataFrame(
        {
            "dirt_moisture": [10.0],
            "prev_dirt_moisture": [float("nan")],
            "trackcd": ["23"],
        }
    )
    result = compute_track_condition_features(df)
    assert pd.isna(result["surface_condition_transition"].iloc[0])


# ---------------------------------------------------------------------------
# T4-02: race-level features (compute_race_condition_features)
# ---------------------------------------------------------------------------


def test_race_condition_match_score():
    """race_condition_match_score = mean of matching aptitude rate"""
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "dirt_moisture": [15.0, 15.0, 15.0],  # wet dirt (>= 12)
            "turf_cushion": [float("nan")] * 3,
            "horse_dirt_wet_hit_rate": [0.4, 0.5, 0.3],
            "horse_dirt_dry_hit_rate": [0.2, 0.1, 0.2],
            "horse_dirt_wet_starts_count": [5.0, 4.0, 3.0],
            "horse_dirt_dry_starts_count": [3.0, 2.0, 3.0],
            "kyakusitukubun_cd": [1.0, 2.0, 3.0],
        }
    )
    result = compute_race_condition_features(df)
    assert "race_condition_match_score" in result.columns
    # wet dirt -> use horse_dirt_wet_hit_rate mean
    assert result["race_condition_match_score"].iloc[0] == pytest.approx((0.4 + 0.5 + 0.3) / 3)


def test_race_condition_match_max():
    """race_condition_match_max = max of matching aptitude rate"""
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "dirt_moisture": [15.0, 15.0],  # wet dirt
            "turf_cushion": [float("nan")] * 2,
            "horse_dirt_wet_hit_rate": [0.4, 0.6],
            "horse_dirt_dry_hit_rate": [0.1, 0.2],
            "horse_dirt_wet_starts_count": [5.0, 4.0],
            "horse_dirt_dry_starts_count": [3.0, 2.0],
            "kyakusitukubun_cd": [1.0, 2.0],
        }
    )
    result = compute_race_condition_features(df)
    assert result["race_condition_match_max"].iloc[0] == pytest.approx(0.6)


def test_race_condition_match_ratio():
    """race_condition_match_ratio = count(rate >= 0.3 AND starts >= 3) / valid entries"""
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "dirt_moisture": [15.0, 15.0, 15.0],  # wet dirt
            "turf_cushion": [float("nan")] * 3,
            "horse_dirt_wet_hit_rate": [0.4, 0.2, 0.5],
            "horse_dirt_dry_hit_rate": [0.1, 0.2, 0.1],
            "horse_dirt_wet_starts_count": [5.0, 4.0, 2.0],  # 3rd has < 3 starts
            "horse_dirt_dry_starts_count": [3.0, 2.0, 3.0],
            "kyakusitukubun_cd": [1.0, 2.0, 3.0],
        }
    )
    result = compute_race_condition_features(df)
    assert "race_condition_match_ratio" in result.columns
    # Only 1st (0.4 >= 0.3, 5 >= 3) qualifies. 2nd: 0.2 < 0.3. 3rd: 2 < 3 starts.
    assert result["race_condition_match_ratio"].iloc[0] == pytest.approx(1.0 / 3)


def test_race_field_front_bias():
    """race_field_front_bias = front_runner_ratio * track_front_bias_score"""
    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "dirt_moisture": [10.0, 10.0, 10.0],
            "kyakusitukubun_cd": [1.0, 2.0, 3.0],  # 1=逃, 2=先 -> 2 front runners
            "turf_cushion": [float("nan")] * 3,
        }
    )
    # First compute track features to get track_front_bias_score
    df_with_bias = compute_track_condition_features(df)
    result = compute_race_condition_features(df_with_bias)
    assert "race_field_front_bias" in result.columns
    front_runner_ratio = 2.0 / 3.0
    bias = df_with_bias["track_front_bias_score"].iloc[0]
    expected = front_runner_ratio * bias
    assert result["race_field_front_bias"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Constants: TRACK_DERIVED_COLS / RACE_CONDITION_COLS
# ---------------------------------------------------------------------------


def test_track_derived_cols_constant():
    """TRACK_DERIVED_COLS has 11 features"""
    assert len(TRACK_DERIVED_COLS) == 11
    expected = [
        "track_front_bias_score",
        "kickback_risk_score",
        "expected_pace_class",
        "cushion_season_deviation",
        "moisture_season_deviation",
        "cushion_anomaly_flag",
        "moisture_extreme_flag",
        "cushion_x_distance",
        "moisture_x_weight",
        "cushion_x_age",
        "surface_condition_transition",
    ]
    for feat in expected:
        assert feat in TRACK_DERIVED_COLS, f"Missing: {feat}"


def test_race_condition_cols_constant():
    """RACE_CONDITION_COLS has 4 features"""
    assert len(RACE_CONDITION_COLS) == 4
    expected = [
        "race_condition_match_score",
        "race_condition_match_max",
        "race_condition_match_ratio",
        "race_field_front_bias",
    ]
    for feat in expected:
        assert feat in RACE_CONDITION_COLS, f"Missing: {feat}"
