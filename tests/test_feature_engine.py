"""src/features/feature_engine.py のテスト"""

import pandas as pd
import pytest

from domain.models import Entry, Race
from features.feature_engine import FeatureEngine


@pytest.fixture
def sample_race_df() -> pd.DataFrame:
    """1レース分の race データ（18頭立て）— 生カラム名"""
    return pd.DataFrame(
        {
            "race_id": ["2024032405030208"] * 18,
            "race_date": [pd.Timestamp("2024-03-24")] * 18,
            "trackcd": [11] * 18,
            "kyori": [1600] * 18,
            "tenkocd": [1] * 18,
            "track_condition_code": [1] * 18,
            "syubetucd": ["13"] * 18,
            "jyokencd1": ["999"] * 18,
            "gradecd": ["_"] * 18,
            "syussotosu": [18] * 18,
            "surface": ["turf"] * 18,
        }
    )


@pytest.fixture
def sample_entry_df() -> pd.DataFrame:
    """18頭の出走馬データ — 生カラム名"""
    umaban = list(range(1, 19))
    odds = [
        1.5,
        2.3,
        3.1,
        5.0,
        8.2,
        12.5,
        18.0,
        25.0,
        35.0,
        45.0,
        55.0,
        68.0,
        80.0,
        95.0,
        110.0,
        130.0,
        150.0,
        200.0,
    ]
    return pd.DataFrame(
        {
            "race_id": ["2024032405030208"] * 18,
            "umaban": umaban,
            "kettonum": [f"000{i:07d}" for i in range(1, 19)],
            "kakuteijyuni": [1, 2, 3, 4, 5, 0, 7, 8, 0, 10, 11, 12, 13, 14, 15, 16, 0, 18],
            "odds": odds,
            "ninki": list(range(1, 19)),
            "bataijyu": [
                480,
                472,
                488,
                464,
                496,
                458,
                500,
                484,
                468,
                492,
                476,
                504,
                460,
                482,
                498,
                470,
                486,
                454,
            ],
            "zogen_fugo": [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3],
            "zogen_sa": [-4, 2, 0, -6, 4, 0, -2, 6, 0, -8, 2, 0, -4, 8, 0, -2, 4, 0],
            "kisyucode": [f"010{i:02d}" for i in range(1, 19)],
            "chokyosicode": [f"010{i:02d}" for i in range(1, 19)],
        }
    )


@pytest.fixture
def sample_odds_df() -> pd.DataFrame:
    """18頭のオッズスナップショット — 生カラム名"""
    umaban = list(range(1, 19))
    tanodds = [
        1.5,
        2.3,
        3.1,
        5.0,
        8.2,
        12.5,
        18.0,
        25.0,
        35.0,
        45.0,
        55.0,
        68.0,
        80.0,
        95.0,
        110.0,
        130.0,
        150.0,
        200.0,
    ]
    return pd.DataFrame(
        {
            "race_id": ["2024032405030208"] * 18,
            "umaban": umaban,
            "tanodds": tanodds,
            "tanninki": list(range(1, 19)),
            "fukuoddslow": [
                1.1,
                1.2,
                1.3,
                1.5,
                1.8,
                2.1,
                2.5,
                2.9,
                3.3,
                3.7,
                4.1,
                4.5,
                4.9,
                5.3,
                5.7,
                6.1,
                6.5,
                7.0,
            ],
        }
    )


class TestFeatureEngineBuildAll:
    def test_merge_produces_correct_shape(self, sample_race_df, sample_entry_df, sample_odds_df):
        """race_df + entry_df + odds_df をマージして18行のDataFrameを返す"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert result.shape[0] == 18
        assert "race_id" in result.columns
        assert "umaban" in result.columns

    def test_output_has_basic_features(self, sample_race_df, sample_entry_df, sample_odds_df):
        """基本特徴量列が存在する"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        expected_cols = [
            "surface",
            "distance_bin",
            "track_condition_code",
            "grade_code",
            "field_size",
            "popularity_rank",
        ]
        for col in expected_cols:
            assert col in result.columns, f"列 '{col}' が不足"

    def test_distance_bin_computed_from_kyori(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ):
        """kyori + surface から distance_bin が計算される"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert "distance_bin" in result.columns
        # 全行が "mile" であることを確認 (kyori=1600, surface=turf)
        assert (result["distance_bin"] == "mile").all()

    def test_track_condition_code_from_raw(self, sample_race_df, sample_entry_df, sample_odds_df):
        """track_condition_code が生データから引き継がれる"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert "track_condition_code" in result.columns
        assert (result["track_condition_code"] == 1).all()

    def test_grade_code_from_gradecd(self, sample_race_df, sample_entry_df, sample_odds_df):
        """gradecd から grade_code にコピーされる"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert "grade_code" in result.columns
        assert (result["grade_code"] == "_").all()

    def test_field_size_from_syussotosu(self, sample_race_df, sample_entry_df, sample_odds_df):
        """syussotosu から field_size にコピーされる"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert "field_size" in result.columns
        assert (result["field_size"] == 18).all()

    def test_popularity_rank_from_tanninki(self, sample_race_df, sample_entry_df, sample_odds_df):
        """tanninki (発走前人気) から popularity_rank にコピーされる"""
        engine = FeatureEngine()
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert "popularity_rank" in result.columns

    def test_exclude_steeple(self, sample_race_df, sample_entry_df, sample_odds_df):
        """障害レース(trackcd >= 51)を除外"""
        steeple_race = sample_race_df.copy()
        steeple_race["trackcd"] = 51
        steeple_race["surface"] = "exclude"
        steeple_race["race_id"] = "1999010101010101"
        steeple_entry = sample_entry_df.copy()
        steeple_entry["race_id"] = "1999010101010101"
        steeple_odds = sample_odds_df.copy()
        steeple_odds["race_id"] = "1999010101010101"

        combined_races = pd.concat([sample_race_df, steeple_race], ignore_index=True)
        combined_entries = pd.concat([sample_entry_df, steeple_entry], ignore_index=True)
        combined_odds = pd.concat([sample_odds_df, steeple_odds], ignore_index=True)

        engine = FeatureEngine(exclude_steeple=True)
        result = engine.build_all(combined_races, combined_entries, combined_odds)
        assert result.shape[0] == 18  # 障害レースの18頭は除外

    def test_no_exclude_steeple(self, sample_race_df, sample_entry_df, sample_odds_df):
        """exclude_steeple=False では障害レースも含む"""
        steeple_race = sample_race_df.copy()
        steeple_race["trackcd"] = 51
        steeple_race["surface"] = "exclude"
        steeple_race["race_id"] = "1999010101010101"
        steeple_entry = sample_entry_df.copy()
        steeple_entry["race_id"] = "1999010101010101"
        steeple_odds = sample_odds_df.copy()
        steeple_odds["race_id"] = "1999010101010101"

        combined_races = pd.concat([sample_race_df, steeple_race], ignore_index=True)
        combined_entries = pd.concat([sample_entry_df, steeple_entry], ignore_index=True)
        combined_odds = pd.concat([sample_odds_df, steeple_odds], ignore_index=True)

        engine = FeatureEngine(exclude_steeple=False)
        result = engine.build_all(combined_races, combined_entries, combined_odds)
        assert result.shape[0] == 36


@pytest.fixture
def sample_race() -> Race:
    return Race(
        year=2024,
        month_day="0324",
        jyo_cd="05",
        kaiji="03",
        nichiji="02",
        race_num="08",
        track_cd=11,
        distance=1600,
        tenko_cd=1,
        baba_cd=1,
        syubetu_cd="13",
        jyoken_cd="999",
        grade_cd="_",
        field_size=18,
    )


@pytest.fixture
def sample_entries() -> list[Entry]:
    entries = []
    for i in range(1, 4):
        entries.append(
            Entry(
                race_id="2024032405030208",
                umaban=i,
                ketto_num=f"0000000{i}",
                finish_pos=i,
                win_odds_actual=float(i + 1),
                popularity_rank=i,
                running_style=i,
                ba_taijyu=480.0,
                zogen_fugo=2,
                zogen_sa=-2.0,
                kisyu_code="01001",
                chokyosi_code="01001",
            )
        )
    return entries


class TestFeatureEngineBuildFeatures:
    def test_build_features_returns_dataframe(self, sample_race, sample_entries):
        """Race + list[Entry] からDataFrameを生成"""
        engine = FeatureEngine()
        result = engine.build_features(sample_race, sample_entries)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3

    def test_build_features_has_basic_columns(self, sample_race, sample_entries):
        """推論結果に基本特徴量列が含まれる"""
        engine = FeatureEngine()
        result = engine.build_features(sample_race, sample_entries)
        assert "track_condition_code" in result.columns
        assert "grade_code" in result.columns
        assert "field_size" in result.columns
        assert "popularity_rank" in result.columns

    def test_build_features_with_odds_snapshot(self, sample_race, sample_entries):
        """オッズスナップショットを結合できる"""
        odds_df = pd.DataFrame(
            {
                "race_id": ["2024032405030208"] * 3,
                "umaban": [1, 2, 3],
                "tanodds": [2.0, 3.0, 4.0],
                "fukuoddslow": [1.1, 1.3, 1.5],
            }
        )
        engine = FeatureEngine()
        result = engine.build_features(sample_race, sample_entries, odds_snapshot=odds_df)
        assert "tanodds" in result.columns
        assert result["tanodds"].tolist() == [2.0, 3.0, 4.0]


class TestWeightChangeZone:
    """A2: weight_change_zone のユニットテスト (_map_basic_features内)"""

    def _make_df_with_zogen(self, zogen_values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "race_id": ["R1"] * len(zogen_values),
                "umaban": list(range(1, len(zogen_values) + 1)),
                "surface": ["turf"] * len(zogen_values),
                "kyori": [1600] * len(zogen_values),
                "gradecd": [0] * len(zogen_values),
                "syussotosu": [10] * len(zogen_values),
                "ninki": list(range(1, len(zogen_values) + 1)),
                "kyakusitukubun": [1] * len(zogen_values),
                "zogen_sa": zogen_values,
            }
        )

    def test_golden_zone(self) -> None:
        from features.feature_engine import FeatureEngine

        fe = FeatureEngine()
        df = self._make_df_with_zogen([5.0, 8.0, 12.0])
        result = fe._map_basic_features(df)
        assert "weight_change_zone" in result.columns
        assert (result["weight_change_zone"] == 2).all()

    def test_stable_zone(self) -> None:
        from features.feature_engine import FeatureEngine

        fe = FeatureEngine()
        df = self._make_df_with_zogen([0.0, -3.0, 3.0, 4.0])
        result = fe._map_basic_features(df)
        assert result["weight_change_zone"].iloc[0] == 1  # 0.0 -> stable
        assert result["weight_change_zone"].iloc[1] == 1  # -3.0 -> stable
        assert result["weight_change_zone"].iloc[2] == 1  # 3.0 -> stable

    def test_caution_zone(self) -> None:
        from features.feature_engine import FeatureEngine

        fe = FeatureEngine()
        df = self._make_df_with_zogen([-5.0, 13.0])
        result = fe._map_basic_features(df)
        assert (result["weight_change_zone"] == 0).all()

    def test_danger_zone(self) -> None:
        from features.feature_engine import FeatureEngine

        fe = FeatureEngine()
        df = self._make_df_with_zogen([15.0, -15.0])
        result = fe._map_basic_features(df)
        assert (result["weight_change_zone"] == -1).all()

    def test_missing_zogen_sa(self) -> None:
        from features.feature_engine import FeatureEngine

        fe = FeatureEngine()
        df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1600],
                "gradecd": [0],
                "syussotosu": [10],
                "ninki": [1],
                "kyakusitukubun": [1],
            }
        )
        result = fe._map_basic_features(df)
        assert "weight_change_zone" in result.columns
        assert result["weight_change_zone"].isna().all()

    def test_boundary_values(self) -> None:
        """境界値テスト: zogen=4 は golden, zogen=-4 は stable"""
        from features.feature_engine import FeatureEngine

        fe = FeatureEngine()
        df = self._make_df_with_zogen([4.0, -4.0, 14.0, -14.0])
        result = fe._map_basic_features(df)
        assert result["weight_change_zone"].iloc[0] == 2  # 4 -> golden (4<=4<=12)
        assert result["weight_change_zone"].iloc[1] == 1  # -4 -> stable (-4 ~ 4)
        assert result["weight_change_zone"].iloc[2] == 0  # 14 -> caution (wait, 14 > 12 and <=14)
        assert result["weight_change_zone"].iloc[3] == 0  # -14 -> caution (-14 <= -4 boundary)


class TestLeakPrevention:
    """データリーク修正の検証テスト"""

    def _make_race_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "race_id": ["R001"] * 3,
                "trackcd": [11] * 3,
                "kyori": [1600] * 3,
                "syussotosu": [3] * 3,
                "surface": ["turf"] * 3,
                "gradecd": ["_"] * 3,
            }
        )

    def _make_entry_df(self, odds: list[float], ninki: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "race_id": ["R001"] * 3,
                "umaban": [1, 2, 3],
                "odds": odds,
                "ninki": ninki,
                "bataijyu": [480.0, 470.0, 490.0],
            }
        )

    def _make_odds_df(self, tanodds: list[float], tanninki: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "race_id": ["R001"] * 3,
                "umaban": [1, 2, 3],
                "tanodds": tanodds,
                "fukuoddslow": [1.1, 1.3, 1.5],
                "tanninki": tanninki,
            }
        )

    def test_popularity_rank_uses_tanninki_not_ninki(self) -> None:
        """tanninki と ninki が異なる場合、popularity_rank は tanninki を使用"""
        engine = FeatureEngine()
        race_df = self._make_race_df()
        # ninki は確定人気 (1,2,3) — リーク源
        entry_df = self._make_entry_df(odds=[3.0, 5.0, 8.0], ninki=[1, 2, 3])
        # tanninki は発走前人気 (3,1,2) — 正しい値
        odds_df = self._make_odds_df(tanodds=[3.0, 5.0, 8.0], tanninki=[3, 1, 2])

        result = engine.build_all(race_df, entry_df, odds_df)
        assert result["popularity_rank"].tolist() == [3, 1, 2]

    def test_popularity_rank_fallback_when_tanninki_zero(self) -> None:
        """tanninki が全て 0 の場合、ninki をフォールバックとして使用"""
        engine = FeatureEngine()
        race_df = self._make_race_df()
        entry_df = self._make_entry_df(odds=[3.0, 5.0, 8.0], ninki=[1, 2, 3])
        odds_df = self._make_odds_df(tanodds=[3.0, 5.0, 8.0], tanninki=[0, 0, 0])

        result = engine.build_all(race_df, entry_df, odds_df)
        # tanninki=0 → ninki フォールバック
        assert result["popularity_rank"].tolist() == [1, 2, 3]

    def test_odds_replaced_by_tanodds(self) -> None:
        """odds (確定) が tanodds (発走前) で上書きされる"""
        engine = FeatureEngine()
        race_df = self._make_race_df()
        # entries.odds は確定オッズ (低い = 強い馬)
        entry_df = self._make_entry_df(odds=[2.0, 4.0, 7.0], ninki=[1, 2, 3])
        # tanodds は発走前オッズ (異なる値)
        odds_df = self._make_odds_df(tanodds=[3.5, 5.0, 9.0], tanninki=[1, 2, 3])

        result = engine.build_all(race_df, entry_df, odds_df)
        # odds 列は tanodds の値に上書きされているはず
        assert result["odds"].tolist() == [3.5, 5.0, 9.0]

    def test_confirmed_odds_preserved(self) -> None:
        """confirmed_odds 列に元の確定オッズが保存される"""
        engine = FeatureEngine()
        race_df = self._make_race_df()
        entry_df = self._make_entry_df(odds=[2.0, 4.0, 7.0], ninki=[1, 2, 3])
        odds_df = self._make_odds_df(tanodds=[3.5, 5.0, 9.0], tanninki=[1, 2, 3])

        result = engine.build_all(race_df, entry_df, odds_df)
        assert "confirmed_odds" in result.columns
        assert result["confirmed_odds"].tolist() == [2.0, 4.0, 7.0]

    def test_odds_fallback_when_tanodds_missing(self) -> None:
        """tanodds が 0/NaN の場合、entries.odds をフォールバックとして保持"""
        engine = FeatureEngine()
        race_df = self._make_race_df()
        entry_df = self._make_entry_df(odds=[2.0, 4.0, 7.0], ninki=[1, 2, 3])
        odds_df = pd.DataFrame(
            {
                "race_id": ["R001"] * 3,
                "umaban": [1, 2, 3],
                "tanodds": [0.0, float("nan"), 9.0],
                "fukuoddslow": [1.1, 1.3, 1.5],
                "tanninki": [1, 2, 3],
            }
        )

        result = engine.build_all(race_df, entry_df, odds_df)
        # tanodds=0 と NaN はフォールバック、9.0 は上書き
        assert result["odds"].iloc[0] == 2.0  # tanodds=0 → entries.odds
        assert result["odds"].iloc[1] == 4.0  # tanodds=NaN → entries.odds
        assert result["odds"].iloc[2] == 9.0  # tanodds=9.0 → 上書き
