# tests/test_interaction_features.py
import pandas as pd
import pytest

from features.interaction_features import compute_interaction_features


def test_kyakusitu_x_distance():
    """脚質×距離bin の文字列結合"""
    df = pd.DataFrame(
        {
            "kyakusitukubun_cd": [1.0, 2.0, 3.0],
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
            "kyakusitukubun_cd": [1.0, 2.0],
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
    """kyakusitukubun_cd がなければ脚質交互作用は生成しない (リーク防止)"""
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


def test_pace_projection_features():
    """履歴脚質から race-level のペース投影特徴量を生成する"""
    df = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "field_size": [4] * 4,
            "kyakusitukubun_cd": [1, 2, 3, 4],
        }
    )
    result = compute_interaction_features(df)
    assert "pace_pressure" in result.columns
    assert "closer_share" in result.columns
    assert "pace_scenario_fit" in result.columns
    assert result["pace_pressure"].iloc[0] == 0.5
    assert result["closer_share"].iloc[0] == 0.5
    assert result["pace_scenario_fit"].iloc[0] < 0  # 逃げ先行はハイペース不利
    assert result["pace_scenario_fit"].iloc[-1] > 0  # 差し追込はハイペース有利


# ---------------------------------------------------------------------------
# PACE-02: actual_pace_fit
# ---------------------------------------------------------------------------


def test_actual_pace_fit_front_runner():
    """脚質1(逃げ)の馬に actual_pace_fit == front_pace_wr"""
    df = pd.DataFrame(
        {
            "race_id": ["R1"],
            "kyakusitukubun_cd": [1],  # 逃げ
            "front_pace_wr": [0.3],
            "closing_pace_wr": [0.1],
        }
    )
    result = compute_interaction_features(df)
    assert "actual_pace_fit" in result.columns
    assert abs(result["actual_pace_fit"].iloc[0] - 0.3) < 1e-6


def test_actual_pace_fit_closer():
    """脚質4(追込)の馬に actual_pace_fit == closing_pace_wr"""
    df = pd.DataFrame(
        {
            "race_id": ["R1"],
            "kyakusitukubun_cd": [4],  # 追込
            "front_pace_wr": [0.3],
            "closing_pace_wr": [0.2],
        }
    )
    result = compute_interaction_features(df)
    assert abs(result["actual_pace_fit"].iloc[0] - 0.2) < 1e-6


def test_actual_pace_fit_nan_when_unknown_style():
    """脚質NaNの馬に actual_pace_fit が NaN"""
    df = pd.DataFrame(
        {
            "race_id": ["R1"],
            "kyakusitukubun_cd": [float("nan")],
            "front_pace_wr": [0.3],
            "closing_pace_wr": [0.2],
        }
    )
    result = compute_interaction_features(df)
    assert "actual_pace_fit" in result.columns
    assert pd.isna(result["actual_pace_fit"].iloc[0])


def test_actual_pace_fit_missing_pace_wr():
    """front_pace_wr/closing_pace_wrがない場合は actual_pace_fit が生成されない"""
    df = pd.DataFrame(
        {
            "race_id": ["R1"],
            "kyakusitukubun_cd": [1],
        }
    )
    result = compute_interaction_features(df)
    assert "actual_pace_fit" not in result.columns


# ---------------------------------------------------------------------------
# INTER-02: 新規ドメイン知識交互作用項 (9個)
# ---------------------------------------------------------------------------


def test_surface_x_distance_bin():
    """surface列とdistance_bin列がある場合、surface_x_distance_binがcategory型で生成される"""
    df = pd.DataFrame(
        {
            "surface": ["turf", "dirt", "turf"],
            "distance_bin": ["short", "long", "middle"],
        }
    )
    result = compute_interaction_features(df)
    assert "surface_x_distance_bin" in result.columns
    assert result["surface_x_distance_bin"].tolist() == ["turf_short", "dirt_long", "turf_middle"]
    assert result["surface_x_distance_bin"].dtype.name == "category"


def test_blood_keito_x_surface():
    """blood_keito_cd列とsurface列がある場合、blood_keito_x_surfaceがcategory型で生成される"""
    df = pd.DataFrame(
        {
            "blood_keito_cd": [1.0, 2.0, 3.0],
            "surface": ["turf", "dirt", "turf"],
        }
    )
    result = compute_interaction_features(df)
    assert "blood_keito_x_surface" in result.columns
    assert result["blood_keito_x_surface"].tolist() == ["1.0_turf", "2.0_dirt", "3.0_turf"]
    assert result["blood_keito_x_surface"].dtype.name == "category"


def test_grade_code_x_distance_bin():
    """grade_code列とdistance_bin列がある場合、grade_code_x_distance_binがcategory型で生成される"""
    df = pd.DataFrame(
        {
            "grade_code": ["G1", "G2", "G3"],
            "distance_bin": ["long", "middle", "short"],
        }
    )
    result = compute_interaction_features(df)
    assert "grade_code_x_distance_bin" in result.columns
    assert result["grade_code_x_distance_bin"].tolist() == ["G1_long", "G2_middle", "G3_short"]
    assert result["grade_code_x_distance_bin"].dtype.name == "category"


def test_sire_wr_x_distance():
    """sire_wr列とkyori列がある場合、sire_wr_x_distanceがNaN安全な数値積で生成される"""
    df = pd.DataFrame(
        {
            "sire_wr": [0.15, 0.20],
            "kyori": [1200, 2400],
        }
    )
    result = compute_interaction_features(df)
    assert "sire_wr_x_distance" in result.columns
    assert result["sire_wr_x_distance"].iloc[0] == pytest.approx(0.15 * 1200)
    assert result["sire_wr_x_distance"].iloc[1] == pytest.approx(0.20 * 2400)


def test_blood_surface_wr_x_condition():
    """blood_surface_wr列とtrack_condition_code列がある場合、blood_surface_wr_x_conditionがNaN安全な数値積"""
    df = pd.DataFrame(
        {
            "blood_surface_wr": [0.18, 0.12],
            "track_condition_code": [1, 2],
        }
    )
    result = compute_interaction_features(df)
    assert "blood_surface_wr_x_condition" in result.columns
    assert result["blood_surface_wr_x_condition"].iloc[0] == pytest.approx(0.18 * 1)
    assert result["blood_surface_wr_x_condition"].iloc[1] == pytest.approx(0.12 * 2)


def test_pace_pressure_x_closing_index():
    """pace_pressure列とclosing_index_avg列がある場合、NaN安全な数値積"""
    df = pd.DataFrame(
        {
            "pace_pressure": [0.5, 0.3],
            "closing_index_avg": [0.4, 0.6],
        }
    )
    result = compute_interaction_features(df)
    assert "pace_pressure_x_closing_index" in result.columns
    assert result["pace_pressure_x_closing_index"].iloc[0] == pytest.approx(0.5 * 0.4)
    assert result["pace_pressure_x_closing_index"].iloc[1] == pytest.approx(0.3 * 0.6)


def test_haron_x_distance():
    """harontimel5_avg列とkyori列がある場合、NaN安全な数値積"""
    df = pd.DataFrame(
        {
            "harontimel5_avg": [35.5, 37.2],
            "kyori": [1200, 2400],
        }
    )
    result = compute_interaction_features(df)
    assert "haron_x_distance" in result.columns
    assert result["haron_x_distance"].iloc[0] == pytest.approx(35.5 * 1200)
    assert result["haron_x_distance"].iloc[1] == pytest.approx(37.2 * 2400)


def test_surface_x_past_perf():
    """norm_finish_logit_avg列とsurface列がある場合、NaN安全な数値積"""
    df = pd.DataFrame(
        {
            "norm_finish_logit_avg": [0.5, -0.3],
            "surface": ["turf", "dirt"],
        }
    )
    result = compute_interaction_features(df)
    assert "surface_x_past_perf" in result.columns
    # turf=1, dirt=2
    assert result["surface_x_past_perf"].iloc[0] == pytest.approx(0.5 * 1)
    assert result["surface_x_past_perf"].iloc[1] == pytest.approx(-0.3 * 2)


def test_weight_x_class():
    """weight_absolute/bataijyu列とgrade_code列がある場合、NaN安全な数値積"""
    df = pd.DataFrame(
        {
            "weight_absolute": [450.0, 500.0],
            "grade_code": ["G1", "G3"],
        }
    )
    result = compute_interaction_features(df)
    assert "weight_x_class" in result.columns
    # G1=5, G3=3
    assert result["weight_x_class"].iloc[0] == pytest.approx(450.0 * 5)
    assert result["weight_x_class"].iloc[1] == pytest.approx(500.0 * 3)


def test_interaction_nan_safety():
    """NaNを含む数値積がNaN安全であること"""
    df = pd.DataFrame(
        {
            "sire_wr": [0.15, float("nan"), 0.20],
            "kyori": [1200, 2400, float("nan")],
        }
    )
    result = compute_interaction_features(df)
    assert "sire_wr_x_distance" in result.columns
    assert result["sire_wr_x_distance"].iloc[0] == pytest.approx(0.15 * 1200)
    assert pd.isna(result["sire_wr_x_distance"].iloc[1])
    assert pd.isna(result["sire_wr_x_distance"].iloc[2])


def test_missing_interaction_columns_skipped():
    """欠損列がある交互作用項はエラーなくスキップされ、他の交互作用項は正常に生成される"""
    df = pd.DataFrame(
        {
            "surface": ["turf", "dirt"],
            "distance_bin": ["short", "long"],
            # blood_keito_cdなし
            # grade_codeなし
        }
    )
    result = compute_interaction_features(df)
    assert "surface_x_distance_bin" in result.columns
    assert "blood_keito_x_surface" not in result.columns
    assert "grade_code_x_distance_bin" not in result.columns


def test_interaction_cols_constant():
    """INTERACTION_COLS定数が12個の交互作用名を含む"""
    from features.interaction_features import INTERACTION_COLS
    assert len(INTERACTION_COLS) == 12
    assert "surface_x_distance_bin" in INTERACTION_COLS
    assert "sire_wr_x_distance" in INTERACTION_COLS


def test_weight_x_class_bataijyu_fallback():
    """weight_absoluteがなくbataijyuがあればweight_x_classはbataijyuを使用"""
    df = pd.DataFrame(
        {
            "bataijyu": [460.0, 510.0],
            "grade_code": ["G1", "G3"],
        }
    )
    result = compute_interaction_features(df)
    assert "weight_x_class" in result.columns
    assert result["weight_x_class"].iloc[0] == pytest.approx(460.0 * 5)
    assert result["weight_x_class"].iloc[1] == pytest.approx(510.0 * 3)
