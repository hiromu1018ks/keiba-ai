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
            "zogensa": [-4, 2, 0, -6, 4, 0, -2, 6, 0, -8, 2, 0, -4, 8, 0, -2, 4, 0],
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
                "zogensa": zogen_values,
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

    def test_popularity_rank_uses_tanodds_rank_first(self) -> None:
        """tanodds がある場合、popularity_rank は tanodds から再計算される"""
        engine = FeatureEngine()
        race_df = self._make_race_df()
        entry_df = self._make_entry_df(odds=[3.0, 5.0, 8.0], ninki=[1, 2, 3])
        # tanninki が不正でも tanodds の順位を優先する
        odds_df = self._make_odds_df(tanodds=[3.0, 5.0, 8.0], tanninki=[3, 1, 2])

        result = engine.build_all(race_df, entry_df, odds_df)
        assert result["popularity_rank"].tolist() == [1.0, 2.0, 3.0]

    def test_popularity_rank_falls_back_to_tanninki_when_tanodds_missing(self) -> None:
        """tanodds が使えない場合は tanninki を利用する"""
        engine = FeatureEngine()
        race_df = self._make_race_df()
        entry_df = self._make_entry_df(odds=[3.0, 5.0, 8.0], ninki=[1, 2, 3])
        odds_df = self._make_odds_df(tanodds=[0.0, float("nan"), 0.0], tanninki=[3, 1, 2])

        result = engine.build_all(race_df, entry_df, odds_df)
        assert result["popularity_rank"].tolist() == [3.0, 1.0, 2.0]

    def test_popularity_rank_fallback_when_tanninki_zero(self) -> None:
        """tanodds/tanninki が使えない場合、popularity_rank は NaN になる (ninkiフォールバック廃止)"""
        engine = FeatureEngine()
        race_df = self._make_race_df()
        entry_df = self._make_entry_df(odds=[3.0, 5.0, 8.0], ninki=[1, 2, 3])
        odds_df = self._make_odds_df(tanodds=[0.0, float("nan"), 0.0], tanninki=[0, 0, 0])

        result = engine.build_all(race_df, entry_df, odds_df)
        # SAFE-01: ninki fallback is removed; all horses get NaN popularity_rank
        assert all(pd.isna(v) for v in result["popularity_rank"].tolist())

    def test_popularity_rank_warns_on_missing_after_tanodds_tanninki(self, caplog: object) -> None:
        """tanodds/tanninki が使えない場合に missing 警告ログを出力する (ninkiフォールバック廃止)"""
        import logging

        engine = FeatureEngine()
        race_df = self._make_race_df()
        entry_df = self._make_entry_df(odds=[3.0, 5.0, 8.0], ninki=[1, 2, 3])
        odds_df = self._make_odds_df(tanodds=[0.0, float("nan"), 0.0], tanninki=[0, 0, 0])
        with caplog.at_level(logging.WARNING, logger="features.feature_engine"):  # type: ignore[attr-defined]
            result = engine.build_all(race_df, entry_df, odds_df)  # noqa: F841
        assert any(
            "popularity_rank" in rec.message and "tanodds/tanninki" in rec.message
            for rec in caplog.records  # type: ignore[attr-defined]
        )

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

    def test_confirmed_odds_dropped_by_safe01(self) -> None:
        """SAFE-01: confirmed_odds (POST_RACE列) は build_all() 出力から除外される"""
        engine = FeatureEngine()
        race_df = self._make_race_df()
        entry_df = self._make_entry_df(odds=[2.0, 4.0, 7.0], ninki=[1, 2, 3])
        odds_df = self._make_odds_df(tanodds=[3.5, 5.0, 9.0], tanninki=[1, 2, 3])

        result = engine.build_all(race_df, entry_df, odds_df)
        # SAFE-01: confirmed_odds is a POST_RACE column and must not appear in output
        assert "confirmed_odds" not in result.columns

    def test_running_style_not_created_from_kyakusitukubun(self) -> None:
        """kyakusitukubun が入力にあっても running_style 列は生成されない"""
        engine = FeatureEngine()
        race_df = pd.DataFrame(
            {
                "race_id": ["R001"] * 3,
                "trackcd": [11] * 3,
                "kyori": [1600] * 3,
                "syussotosu": [3] * 3,
                "surface": ["turf"] * 3,
                "gradecd": ["_"] * 3,
            }
        )
        entry_df = pd.DataFrame(
            {
                "race_id": ["R001"] * 3,
                "umaban": [1, 2, 3],
                "odds": [3.0, 5.0, 8.0],
                "ninki": [1, 2, 3],
                "bataijyu": [480.0, 470.0, 490.0],
                "kyakusitukubun": [1, 2, 3],
            }
        )
        odds_df = pd.DataFrame(
            {
                "race_id": ["R001"] * 3,
                "umaban": [1, 2, 3],
                "tanodds": [3.0, 5.0, 8.0],
                "fukuoddslow": [1.1, 1.3, 1.5],
                "tanninki": [1, 2, 3],
            }
        )
        result = engine.build_all(race_df, entry_df, odds_df)
        assert "running_style" not in result.columns

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


class TestFLBSlopeFeatures:
    """compute_flb_slope の wiring テスト — odds_skewness, implied_prob_hhi"""

    def test_build_all_includes_odds_skewness(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ) -> None:
        """build_all の結果に odds_skewness が含まれる"""
        engine = FeatureEngine(exclude_steeple=False)
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert "odds_skewness" in result.columns, "odds_skewness should be in output"

    def test_build_all_includes_implied_prob_hhi(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ) -> None:
        """build_all の結果に implied_prob_hhi が含まれる"""
        engine = FeatureEngine(exclude_steeple=False)
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert "implied_prob_hhi" in result.columns, "implied_prob_hhi should be in output"

    def test_odds_skewness_values_from_18_horse_race(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ) -> None:
        """18頭立てのレースで odds_skewness が正の値（右偏り）になる"""
        engine = FeatureEngine(exclude_steeple=False)
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        # 全馬同じレースなので race_id ごとに1つの値
        skew_vals = result["odds_skewness"].dropna().unique()
        assert len(skew_vals) == 1
        # 18頭で 1.5〜200 のオッズ分布は右に歪む → 正の歪度
        assert skew_vals[0] > 0

    def test_implied_prob_hhi_range(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ) -> None:
        """implied_prob_hhi が 0 < HHI <= 1 の範囲にある"""
        engine = FeatureEngine(exclude_steeple=False)
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        hhi_vals = result["implied_prob_hhi"].dropna().unique()
        assert len(hhi_vals) == 1
        assert 0 < hhi_vals[0] <= 1.0


class TestFeatureCache:
    """FeatureEngine Parquetキャッシュのユニットテスト"""

    def test_cache_key_deterministic(self) -> None:
        """同じ入力で同じキャッシュキーが生成される"""
        from pathlib import Path

        from features.feature_engine import compute_cache_key

        paths = [Path("data/raw/races.parquet")]
        key1 = compute_cache_key(paths, ("2020-01-01", "2023-12-31"), "build_all")
        key2 = compute_cache_key(paths, ("2020-01-01", "2023-12-31"), "build_all")
        assert key1 == key2
        assert len(key1) == 16

    def test_cache_key_different_inputs(self) -> None:
        """異なる日付範囲で異なるキャッシュキーが生成される"""
        from pathlib import Path

        from features.feature_engine import compute_cache_key

        paths = [Path("data/raw/races.parquet")]
        key1 = compute_cache_key(paths, ("2020-01-01", "2023-12-31"), "build_all")
        key2 = compute_cache_key(paths, ("2021-01-01", "2024-12-31"), "build_all")
        assert key1 != key2

    def test_cache_key_different_type(self) -> None:
        """異なる特徴量種別で異なるキャッシュキーが生成される"""
        from pathlib import Path

        from features.feature_engine import compute_cache_key

        paths = [Path("data/raw/races.parquet")]
        key1 = compute_cache_key(paths, ("2020-01-01", "2023-12-31"), "build_all")
        key2 = compute_cache_key(paths, ("2020-01-01", "2023-12-31"), "intra_race")
        assert key1 != key2

    def test_is_cache_valid_no_cache_file(self, tmp_path: object) -> None:
        """キャッシュファイルが存在しない場合はFalse"""
        import pathlib

        from features.feature_engine import is_cache_valid

        cache_path = pathlib.Path(str(tmp_path)) / "nonexistent.parquet"
        source_path = pathlib.Path(str(tmp_path)) / "source.parquet"
        source_path.touch()
        assert is_cache_valid(cache_path, [source_path]) is False

    def test_is_cache_valid_stale_source(self, tmp_path: object) -> None:
        """ソースファイルがキャッシュより新しい場合はFalse"""
        import pathlib
        import time

        from features.feature_engine import is_cache_valid

        tp = pathlib.Path(str(tmp_path))
        cache_path = tp / "cache.parquet"
        source_path = tp / "source.parquet"
        cache_path.touch()
        time.sleep(0.05)
        source_path.touch()
        assert is_cache_valid(cache_path, [source_path]) is False

    def test_is_cache_valid_fresh(self, tmp_path: object) -> None:
        """キャッシュがソースより新しい場合はTrue"""
        import pathlib
        import time

        from features.feature_engine import is_cache_valid

        tp = pathlib.Path(str(tmp_path))
        source_path = tp / "source.parquet"
        source_path.touch()
        time.sleep(0.05)
        cache_path = tp / "cache.parquet"
        cache_path.touch()
        assert is_cache_valid(cache_path, [source_path]) is True

    def test_build_all_uses_cache_on_hit(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ) -> None:
        """キャッシュHIT時にキャッシュされたDataFrameを返す"""
        from unittest.mock import MagicMock, patch

        from db.parquet_store import ParquetStore

        cached_df = pd.DataFrame({"race_id": ["cached"], "col": [1.0]})
        mock_store = MagicMock(spec=ParquetStore)
        mock_store.data_dir = "data"
        mock_store.read.return_value = cached_df
        # exists() はパスチェック用にTrueを返す
        mock_store.exists.return_value = True

        engine = FeatureEngine(exclude_steeple=False, use_cache=True)

        with patch("features.feature_engine.is_cache_valid", return_value=True):
            result = engine.build_all(
                sample_race_df, sample_entry_df, sample_odds_df, store=mock_store
            )

        assert len(result) == 1
        assert result["race_id"].iloc[0] == "cached"

    def test_build_all_writes_cache_on_miss(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ) -> None:
        """キャッシュMISS時にstore.write()が呼ばれる"""
        from unittest.mock import MagicMock, patch

        from db.parquet_store import ParquetStore

        mock_store = MagicMock(spec=ParquetStore)
        mock_store.data_dir = "data"
        mock_store.exists.return_value = True

        engine = FeatureEngine(exclude_steeple=False, use_cache=True)

        with patch("features.feature_engine.is_cache_valid", return_value=False):
            result = engine.build_all(
                sample_race_df, sample_entry_df, sample_odds_df, store=mock_store
            )

        # store.write() should have been called for cache save
        assert mock_store.write.called
        # The result should be a real computation result (18 rows)
        assert len(result) == 18
        # Check the cache_dir argument
        call_args = mock_store.write.call_args
        assert call_args[0][0] == "features/cache"

    def test_build_all_skip_cache_when_no_store(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ) -> None:
        """store=Noneのときキャッシュ処理をスキップして正常動作する"""
        engine = FeatureEngine(exclude_steeple=False, use_cache=True)
        result = engine.build_all(sample_race_df, sample_entry_df, sample_odds_df)
        assert len(result) == 18

    def test_build_all_disabled_cache(
        self, sample_race_df, sample_entry_df, sample_odds_df
    ) -> None:
        """use_cache=Falseのときキャッシュ書き込みが呼ばれない"""
        from unittest.mock import MagicMock

        mock_store = MagicMock()
        mock_store.data_dir = "data"

        engine = FeatureEngine(exclude_steeple=False, use_cache=False)
        result = engine.build_all(
            sample_race_df, sample_entry_df, sample_odds_df, store=mock_store
        )
        assert len(result) == 18
        # store.write() should not be called for cache save when cache is disabled
        mock_store.write.assert_not_called()


class TestCodeHash:
    """compute_code_hash() と compute_cache_key() 拡張のテスト"""

    def test_compute_code_hash_returns_hex_string(self, tmp_path: object) -> None:
        """正常系で16文字hex文字列が返る"""
        import pathlib

        from features.feature_engine import compute_code_hash

        features_dir = pathlib.Path(str(tmp_path)) / "features"
        features_dir.mkdir()
        (features_dir / "engine.py").write_text("x = 1", encoding="utf-8")
        (features_dir / "helper.py").write_text("y = 2", encoding="utf-8")

        result = compute_code_hash(str(features_dir))
        assert isinstance(result, str)
        assert len(result) == 16
        # hex文字列であることを確認
        assert all(c in "0123456789abcdef" for c in result)

    def test_compute_code_hash_changes_on_file_change(self, tmp_path: object) -> None:
        """tmp_path配下の.pyファイル変更でハッシュが変わる"""
        import pathlib

        from features.feature_engine import compute_code_hash

        features_dir = pathlib.Path(str(tmp_path)) / "features"
        features_dir.mkdir()
        (features_dir / "engine.py").write_text("x = 1", encoding="utf-8")

        hash1 = compute_code_hash(str(features_dir))
        (features_dir / "engine.py").write_text("x = 2", encoding="utf-8")
        hash2 = compute_code_hash(str(features_dir))

        assert hash1 != hash2

    def test_compute_code_hash_empty_dir(self, tmp_path: object) -> None:
        """.pyファイルがない場合は空文字"""
        import pathlib

        from features.feature_engine import compute_code_hash

        features_dir = pathlib.Path(str(tmp_path)) / "features"
        features_dir.mkdir()

        result = compute_code_hash(str(features_dir))
        assert result == ""

    def test_compute_cache_key_with_code_hash(self) -> None:
        """code_hash引数付きでハッシュが変わる"""
        from pathlib import Path

        from features.feature_engine import compute_cache_key

        paths = [Path("data/raw/races.parquet")]
        key_no_hash = compute_cache_key(paths, ("2020-01-01", "2023-12-31"), "build_all")
        key_with_hash = compute_cache_key(
            paths, ("2020-01-01", "2023-12-31"), "build_all",
            code_hash="abc123",
        )
        key_other_hash = compute_cache_key(
            paths, ("2020-01-01", "2023-12-31"), "build_all",
            code_hash="def456",
        )
        assert key_no_hash != key_with_hash
        assert key_with_hash != key_other_hash

    def test_compute_cache_key_backward_compatible(self) -> None:
        """code_hash=Noneで従来と同じキーが返る (後方互換)"""
        from pathlib import Path

        from features.feature_engine import compute_cache_key

        paths = [Path("data/raw/races.parquet")]
        # 従来の3引数呼び出し (code_hashなし)
        key_old_style = compute_cache_key(paths, ("2020-01-01", "2023-12-31"), "build_all")
        # 新しい呼び出し (code_hash=None)
        key_new_style = compute_cache_key(
            paths, ("2020-01-01", "2023-12-31"), "build_all",
            code_hash=None,
        )
        # code_hash="" と明示的に指定した場合も同じ
        key_explicit_empty = compute_cache_key(
            paths, ("2020-01-01", "2023-12-31"), "build_all",
            code_hash="",
        )
        assert key_old_style == key_new_style
        assert key_old_style == key_explicit_empty


class TestCleanupStaleCache:
    """_cleanup_stale_cache() のテスト"""

    def test_cleanup_stale_cache_removes_old_files(self, tmp_path: object) -> None:
        """古いfeat_*.parquetが削除され新しいものは残る"""
        import pathlib

        cache_dir = pathlib.Path(str(tmp_path)) / "cache"
        cache_dir.mkdir()
        current_name = "feat_abc123"

        # 現在のキャッシュファイル
        (cache_dir / f"{current_name}.parquet").write_bytes(b"current")
        # 古いキャッシュファイル
        (cache_dir / "feat_old001.parquet").write_bytes(b"old1")
        (cache_dir / "feat_old002.parquet").write_bytes(b"old2")
        # 関係ないファイル
        (cache_dir / "other.txt").write_text("keep", encoding="utf-8")

        engine = FeatureEngine()
        engine._cleanup_stale_cache(cache_dir, current_name)

        # 現在のキャッシュは残る
        assert (cache_dir / f"{current_name}.parquet").exists()
        # 古いキャッシュは削除される
        assert not (cache_dir / "feat_old001.parquet").exists()
        assert not (cache_dir / "feat_old002.parquet").exists()
        # 関係ないファイルは残る
        assert (cache_dir / "other.txt").exists()
