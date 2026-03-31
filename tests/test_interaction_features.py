# tests/test_interaction_features.py
import pandas as pd

from features.interaction_features import compute_interaction_features


def test_kyakusitu_x_distance():
    """脚質×距離bin の文字列結合"""
    df = pd.DataFrame(
        {
            "kyakusitu_cd": [1.0, 2.0, 3.0],
            "distance_bin": ["sprint", "mile", "intermediate"],
        }
    )
    result = compute_interaction_features(df)
    assert "kyakusitu_x_distance" in result.columns
    assert result["kyakusitu_x_distance"].tolist() == ["1.0_sprint", "2.0_mile", "3.0_intermediate"]


def test_kyakusitu_x_surface():
    """脚質×馬場 の文字列結合"""
    df = pd.DataFrame(
        {
            "kyakusitu_cd": [1.0, 2.0],
            "surface": ["turf", "dirt"],
        }
    )
    result = compute_interaction_features(df)
    assert "kyakusitu_x_surface" in result.columns
    assert result["kyakusitu_x_surface"].tolist() == ["1.0_turf", "2.0_dirt"]


def test_weight_x_distance():
    """馬体重×距離の数値積"""
    df = pd.DataFrame(
        {
            "weight_absolute": [450.0, 500.0],
            "kyori": [1200, 2400],
        }
    )
    result = compute_interaction_features(df)
    assert "weight_x_distance" in result.columns
    assert result["weight_x_distance"].tolist() == [450.0 * 1200, 500.0 * 2400]


def test_weight_x_distance_nan():
    """NaN伝播: いずれかがNaNなら結果もNaN"""
    df = pd.DataFrame(
        {
            "weight_absolute": [450.0, float("nan"), 500.0],
            "kyori": [1200, 2400, float("nan")],
        }
    )
    result = compute_interaction_features(df)
    assert result["weight_x_distance"].iloc[0] == 450.0 * 1200
    assert pd.isna(result["weight_x_distance"].iloc[1])
    assert pd.isna(result["weight_x_distance"].iloc[2])


def test_kyakusitu_fallback():
    """kyakusitu_cd がなければ脚質交互作用は生成しない (リーク防止)"""
    df = pd.DataFrame(
        {
            "kyakusitu": [1.0, 2.0],
            "distance_bin": ["sprint", "mile"],
            "surface": ["turf", "dirt"],
        }
    )
    result = compute_interaction_features(df)
    # kyakusitu_cd がない場合、ポストレースkyakusituへのフォールバックは禁止
    assert "kyakusitu_x_distance" not in result.columns
    assert "kyakusitu_x_surface" not in result.columns


def test_ba_taijyu_fallback():
    """weight_absolute がなく bataijyu があればそちらを使う"""
    df = pd.DataFrame(
        {
            "bataijyu": [450.0, 500.0],
            "kyori": [1200, 2400],
        }
    )
    result = compute_interaction_features(df)
    assert "weight_x_distance" in result.columns
    assert result["weight_x_distance"].tolist() == [450.0 * 1200, 500.0 * 2400]


def test_missing_columns():
    """必要列がない場合は追加しない"""
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    result = compute_interaction_features(df)
    assert "kyakusitu_x_distance" not in result.columns
    assert "weight_x_distance" not in result.columns
