"""FeatureBuilder クラスの単体テスト。

全13エンリッチメントモジュールをモックし、正しい順序で呼び出されることを検証する。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from features.feature_builder import FeatureBuilder
from features.feature_manifest import FeatureBuildResult, FeatureManifest, FeatureState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_race_df(n: int = 3) -> pd.DataFrame:
    """最小限の race_df を生成。"""
    return pd.DataFrame({
        "race_id": [f"r{i}" for i in range(n)],
        "race_date": pd.to_datetime(["2024-01-01"] * n),
        "trackcd": [1] * n,
        "kyori": [1600] * n,
        "jyocd": [1] * n,
    })


def _make_entry_df(n: int = 3) -> pd.DataFrame:
    """最小限の entry_df を生成。"""
    return pd.DataFrame({
        "race_id": [f"r{i}" for i in range(n)],
        "umaban": [1, 2, 3][:n],
        "kettonum": [100, 200, 300][:n],
        "kisyucode": [10, 20, 30][:n],
        "chokyosicode": [40, 50, 60][:n],
        "race_date": pd.to_datetime(["2024-01-01"] * n),
    })


def _make_odds_df(n: int = 3) -> pd.DataFrame:
    """最小限の odds_df を生成。"""
    return pd.DataFrame({
        "race_id": [f"r{i}" for i in range(n)],
        "umaban": [1, 2, 3][:n],
        "tanodds": [3.0, 5.0, 10.0][:n],
    })


def _make_feat_df(n: int = 3) -> pd.DataFrame:
    """FeatureEngine.build_all() のモック戻り値。"""
    df = pd.DataFrame({
        "race_id": [f"r{i}" for i in range(n)],
        "umaban": [1, 2, 3][:n],
        "kettonum": [100, 200, 300][:n],
        "surface": ["turf"] * n,
        "race_date": pd.to_datetime(["2024-01-01"] * n),
        "trackcd": [1] * n,
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n),
        "kakuteijyuni": [1, 2, 3][:n],
        "confirmed_odds": [3.0, 5.0, 10.0][:n],
    })
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeatureBuilderTraining:
    """build_for_training のテスト。"""

    @patch("features.feature_builder.SubModelManager")
    @patch("features.feature_builder.FeatureEngine")
    def test_returns_feature_build_result(
        self, mock_fe_cls: MagicMock, mock_sm_cls: MagicMock
    ) -> None:
        """build_for_training が FeatureBuildResult を返す。"""
        mock_store = MagicMock()
        mock_fe = MagicMock()
        mock_fe_cls.return_value = mock_fe
        mock_fe.build_all.return_value = _make_feat_df()
        mock_sm = MagicMock()
        mock_sm_cls.return_value = mock_sm
        mock_sm.add_distance_band_features.side_effect = lambda df: df

        # 全エンリッチメントモジュールをモック
        with patch("features.horse_history_features.HorseHistoryFeatures") as mock_hist, \
             patch("features.horse_history_features.HorseHistoryFeatures.add_race_transforms", staticmethod(lambda df: df)), \
             patch("features.pace_aptitude_features.PaceAptitudeFeatures") as mock_pace, \
             patch("features.course_features.CourseFeatures") as mock_course, \
             patch("features.sire_features.SireFeatures") as mock_sire, \
             patch("features.dam_pedigree_features.DamPedigreeFeatures") as mock_dam, \
             patch("features.record_features.RecordFeatures") as mock_record, \
             patch("features.track_condition_features.compute_track_condition_features", side_effect=lambda df, **kw: df), \
             patch("features.track_condition_features.compute_race_condition_features", side_effect=lambda df, **kw: df), \
             patch("features.interaction_features.compute_interaction_features", side_effect=lambda df: df), \
             patch("features.mining_features.MiningFeatures") as mock_mining, \
             patch("features.relative_features.compute_relative_features", side_effect=lambda df: df), \
             patch("features.jockey_context_features.JockeyContextFeatures") as mock_jockey, \
             patch("features.trainer_context_features.TrainerContextFeatures") as mock_trainer, \
             patch("features.jockey_trainer_combo.JockeyTrainerComboFeatures") as mock_jt, \
             patch("db.readers.load_sire_stats", return_value=pd.DataFrame()), \
             patch("db.readers.load_horses", return_value=pd.DataFrame()):

            mock_hist_inst = MagicMock()
            mock_hist.return_value = mock_hist_inst
            mock_hist_inst.compute.return_value = pd.DataFrame({
                "race_id": ["r0", "r1", "r2"],
                "umaban": [1, 2, 3],
                "hist_feat": [0.1, 0.2, 0.3],
            })

            for mod in [mock_pace, mock_course]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute_batch.return_value = pd.DataFrame()

            mock_sire_inst = MagicMock()
            mock_sire.return_value = mock_sire_inst
            mock_sire_inst.compute_batch.return_value = pd.DataFrame()

            mock_dam_inst = MagicMock()
            mock_dam.return_value = mock_dam_inst
            mock_dam_inst.compute.return_value = pd.DataFrame()

            mock_record_inst = MagicMock()
            mock_record.return_value = mock_record_inst
            mock_record_inst.compute.return_value = pd.DataFrame()

            mock_mining_inst = MagicMock()
            mock_mining.return_value = mock_mining_inst
            mock_mining_inst.compute.return_value = pd.DataFrame()

            for mod in [mock_jockey, mock_trainer, mock_jt]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute.return_value = pd.DataFrame({
                    "race_id": ["r0", "r1", "r2"],
                    "umaban": [1, 2, 3],
                })

            builder = FeatureBuilder(store=mock_store)
            result = builder.build_for_training(
                _make_race_df(), _make_entry_df(), _make_odds_df()
            )

            assert isinstance(result, FeatureBuildResult)
            assert not result.frame.empty
            assert isinstance(result.manifest, FeatureManifest)
            assert len(result.manifest.compute_hash()) == 64


class TestFeatureBuilderInference:
    """build_for_inference のテスト。"""

    def test_raises_on_none_feature_state(self) -> None:
        """feature_state=None で ValueError を送出する。"""
        mock_store = MagicMock()
        builder = FeatureBuilder(store=mock_store)
        with pytest.raises(ValueError, match="feature_state is required"):
            builder.build_for_inference(
                _make_race_df(), _make_entry_df(), _make_odds_df(),
                feature_state=None,  # type: ignore[arg-type]
            )

    @patch("features.feature_builder.SubModelManager")
    @patch("features.feature_builder.FeatureEngine")
    def test_removes_post_race_cols(
        self, mock_fe_cls: MagicMock, mock_sm_cls: MagicMock
    ) -> None:
        """build_for_inference が POST_RACE 列を除去する。"""
        mock_store = MagicMock()
        mock_fe = MagicMock()
        mock_fe_cls.return_value = mock_fe
        feat_df = _make_feat_df()
        feat_df["time"] = [110.0, 112.0, 115.0]
        feat_df["ninki"] = [1, 2, 3]
        mock_fe.build_all.return_value = feat_df
        mock_sm = MagicMock()
        mock_sm_cls.return_value = mock_sm
        mock_sm.add_distance_band_features.side_effect = lambda df: df

        state = FeatureState(
            track_stats={"track_01": {"mean": 1.0}},
            track_month_stats={},
            feature_version="1.0",
        )

        with patch("features.horse_history_features.HorseHistoryFeatures") as mock_hist, \
             patch("features.horse_history_features.HorseHistoryFeatures.add_race_transforms", staticmethod(lambda df: df)), \
             patch("features.pace_aptitude_features.PaceAptitudeFeatures") as mock_pace, \
             patch("features.course_features.CourseFeatures") as mock_course, \
             patch("features.sire_features.SireFeatures") as mock_sire, \
             patch("features.dam_pedigree_features.DamPedigreeFeatures") as mock_dam, \
             patch("features.record_features.RecordFeatures") as mock_record, \
             patch("features.track_condition_features.compute_track_condition_features", side_effect=lambda df, **kw: df), \
             patch("features.track_condition_features.compute_race_condition_features", side_effect=lambda df, **kw: df), \
             patch("features.interaction_features.compute_interaction_features", side_effect=lambda df: df), \
             patch("features.mining_features.MiningFeatures") as mock_mining, \
             patch("features.relative_features.compute_relative_features", side_effect=lambda df: df), \
             patch("features.jockey_context_features.JockeyContextFeatures") as mock_jockey, \
             patch("features.trainer_context_features.TrainerContextFeatures") as mock_trainer, \
             patch("features.jockey_trainer_combo.JockeyTrainerComboFeatures") as mock_jt, \
             patch("db.readers.load_sire_stats", return_value=pd.DataFrame()), \
             patch("db.readers.load_horses", return_value=pd.DataFrame()):

            mock_hist_inst = MagicMock()
            mock_hist.return_value = mock_hist_inst
            mock_hist_inst.compute.return_value = pd.DataFrame({
                "race_id": ["r0", "r1", "r2"],
                "umaban": [1, 2, 3],
                "hist_feat": [0.1, 0.2, 0.3],
            })

            for mod in [mock_pace, mock_course]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute_batch.return_value = pd.DataFrame()

            mock_sire_inst = MagicMock()
            mock_sire.return_value = mock_sire_inst
            mock_sire_inst.compute_batch.return_value = pd.DataFrame()

            for mod in [mock_dam, mock_record, mock_mining]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute.return_value = pd.DataFrame()

            for mod in [mock_jockey, mock_trainer, mock_jt]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute.return_value = pd.DataFrame()

            builder = FeatureBuilder(store=mock_store)
            result = builder.build_for_inference(
                _make_race_df(), _make_entry_df(), _make_odds_df(),
                feature_state=state,
            )

            assert isinstance(result, FeatureBuildResult)
            for col in ["time", "ninki"]:
                assert col not in result.frame.columns, (
                    f"POST_RACE column '{col}' should be removed"
                )


class TestFeatureBuilderEnrichmentOrder:
    """13 エンリッチメントモジュールの呼び出し順序テスト。"""

    @patch("features.feature_builder.SubModelManager")
    @patch("features.feature_builder.FeatureEngine")
    def test_enrichment_modules_called_in_correct_order(
        self, mock_fe_cls: MagicMock, mock_sm_cls: MagicMock
    ) -> None:
        """エンリッチメントモジュールが正しい順序で呼び出される。"""
        mock_store = MagicMock()
        mock_fe = MagicMock()
        mock_fe_cls.return_value = mock_fe
        mock_fe.build_all.return_value = _make_feat_df()
        mock_sm = MagicMock()
        mock_sm_cls.return_value = mock_sm
        mock_sm.add_distance_band_features.side_effect = lambda df: df

        call_order: list[str] = []

        def _track_fn(name: str):
            """呼び出し順序を記録するラッパー。"""
            def _fn(*args, **kwargs):
                call_order.append(name)
                if args and isinstance(args[0], pd.DataFrame):
                    return args[0]
                return pd.DataFrame()
            return _fn

        with patch("features.horse_history_features.HorseHistoryFeatures") as mock_hist, \
             patch("features.horse_history_features.HorseHistoryFeatures.add_race_transforms") as mock_rt, \
             patch("features.pace_aptitude_features.PaceAptitudeFeatures") as mock_pace, \
             patch("features.course_features.CourseFeatures") as mock_course, \
             patch("features.sire_features.SireFeatures") as mock_sire, \
             patch("features.dam_pedigree_features.DamPedigreeFeatures") as mock_dam, \
             patch("features.record_features.RecordFeatures") as mock_record, \
             patch("features.track_condition_features.compute_track_condition_features") as mock_tcf, \
             patch("features.track_condition_features.compute_race_condition_features") as mock_rcf, \
             patch("features.interaction_features.compute_interaction_features") as mock_inter, \
             patch("features.mining_features.MiningFeatures") as mock_mining, \
             patch("features.relative_features.compute_relative_features") as mock_rel, \
             patch("features.jockey_context_features.JockeyContextFeatures") as mock_jockey, \
             patch("features.trainer_context_features.TrainerContextFeatures") as mock_trainer, \
             patch("features.jockey_trainer_combo.JockeyTrainerComboFeatures") as mock_jt, \
             patch("db.readers.load_sire_stats", return_value=pd.DataFrame({"sire_id": ["s1"]})), \
             patch("db.readers.load_horses", return_value=pd.DataFrame({
                 "kettonum": [100, 200, 300],
                 "ketto3infohansyokunum1": ["s1", "s2", "s3"],
                 "ketto3infohansyokunum5": ["b1", "b2", "b3"],
             })):

            mock_hist_inst = MagicMock()
            mock_hist.return_value = mock_hist_inst
            mock_hist_inst.compute.return_value = pd.DataFrame({
                "race_id": ["r0", "r1", "r2"], "umaban": [1, 2, 3],
                "h": [1, 2, 3],
            })
            mock_rt.side_effect = lambda df: (call_order.append("race_transforms"), df)[1]

            mock_pace_inst = MagicMock()
            mock_pace.return_value = mock_pace_inst
            mock_pace_inst.compute_batch.side_effect = lambda df: (call_order.append("pace"), pd.DataFrame())[1]

            mock_course_inst = MagicMock()
            mock_course.return_value = mock_course_inst
            mock_course_inst.compute_batch.side_effect = lambda df: (call_order.append("course"), pd.DataFrame())[1]

            mock_sire_inst = MagicMock()
            mock_sire.return_value = mock_sire_inst
            mock_sire_inst.compute_batch.side_effect = lambda df: (call_order.append("sire"), pd.DataFrame())[1]

            mock_dam_inst = MagicMock()
            mock_dam.return_value = mock_dam_inst
            mock_dam_inst.compute.side_effect = lambda df: (call_order.append("dam"), pd.DataFrame())[1]

            mock_record_inst = MagicMock()
            mock_record.return_value = mock_record_inst
            mock_record_inst.compute.side_effect = lambda df: (call_order.append("record"), pd.DataFrame())[1]

            mock_tcf.side_effect = lambda df, **kw: (call_order.append("track_cond"), df)[1]
            mock_rcf.side_effect = lambda df, **kw: (call_order.append("race_cond"), df)[1]
            mock_inter.side_effect = lambda df: (call_order.append("interaction"), df)[1]

            mock_mining_inst = MagicMock()
            mock_mining.return_value = mock_mining_inst
            mock_mining_inst.compute.side_effect = lambda df: (call_order.append("mining"), pd.DataFrame())[1]

            mock_rel.side_effect = lambda df: (call_order.append("relative"), df)[1]

            for mod, name in [
                (mock_jockey, "jockey"),
                (mock_trainer, "trainer"),
                (mock_jt, "jt_combo"),
            ]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute.side_effect = lambda df, _n=name: (
                    call_order.append(_n),
                    pd.DataFrame({
                        "race_id": df["race_id"].values[:len(df)],
                        "umaban": [1, 2, 3][:len(df)],
                    }),
                )[1]

            builder = FeatureBuilder(store=mock_store)
            builder.build_for_training(
                _make_race_df(), _make_entry_df(), _make_odds_df()
            )

            expected_order = [
                "race_transforms",
                "pace",
                "course",
                "sire",
                "dam",
                "record",
                "track_cond",
                "race_cond",
                "interaction",
                "mining",
                "relative",
                "jockey",
                "trainer",
                "jt_combo",
            ]
            assert call_order == expected_order, (
                f"Enrichment order mismatch:\n  got:      {call_order}\n  expected: {expected_order}"
            )


class TestFeatureBuilderManifestHash:
    """manifest ハッシュの決定性テスト。"""

    @patch("features.feature_builder.SubModelManager")
    @patch("features.feature_builder.FeatureEngine")
    def test_manifest_hash_deterministic(
        self, mock_fe_cls: MagicMock, mock_sm_cls: MagicMock
    ) -> None:
        """同じ入力に対して同じ manifest ハッシュを返す。"""
        mock_store = MagicMock()
        mock_fe = MagicMock()
        mock_fe_cls.return_value = mock_fe
        mock_fe.build_all.return_value = _make_feat_df()
        mock_sm = MagicMock()
        mock_sm_cls.return_value = mock_sm
        mock_sm.add_distance_band_features.side_effect = lambda df: df

        with patch("features.horse_history_features.HorseHistoryFeatures") as mock_hist, \
             patch("features.horse_history_features.HorseHistoryFeatures.add_race_transforms", staticmethod(lambda df: df)), \
             patch("features.pace_aptitude_features.PaceAptitudeFeatures") as mock_pace, \
             patch("features.course_features.CourseFeatures") as mock_course, \
             patch("features.sire_features.SireFeatures") as mock_sire, \
             patch("features.dam_pedigree_features.DamPedigreeFeatures") as mock_dam, \
             patch("features.record_features.RecordFeatures") as mock_record, \
             patch("features.track_condition_features.compute_track_condition_features", side_effect=lambda df, **kw: df), \
             patch("features.track_condition_features.compute_race_condition_features", side_effect=lambda df, **kw: df), \
             patch("features.interaction_features.compute_interaction_features", side_effect=lambda df: df), \
             patch("features.mining_features.MiningFeatures") as mock_mining, \
             patch("features.relative_features.compute_relative_features", side_effect=lambda df: df), \
             patch("features.jockey_context_features.JockeyContextFeatures") as mock_jockey, \
             patch("features.trainer_context_features.TrainerContextFeatures") as mock_trainer, \
             patch("features.jockey_trainer_combo.JockeyTrainerComboFeatures") as mock_jt, \
             patch("db.readers.load_sire_stats", return_value=pd.DataFrame()), \
             patch("db.readers.load_horses", return_value=pd.DataFrame()):

            mock_hist_inst = MagicMock()
            mock_hist.return_value = mock_hist_inst
            mock_hist_inst.compute.return_value = pd.DataFrame({
                "race_id": ["r0", "r1", "r2"], "umaban": [1, 2, 3],
            })

            for mod in [mock_pace, mock_course, mock_sire]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute_batch.return_value = pd.DataFrame()

            for mod in [mock_dam, mock_record, mock_mining]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute.return_value = pd.DataFrame()

            for mod in [mock_jockey, mock_trainer, mock_jt]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute.return_value = pd.DataFrame()

            builder = FeatureBuilder(store=mock_store)
            r1 = builder.build_for_training(
                _make_race_df(), _make_entry_df(), _make_odds_df()
            )
            r2 = builder.build_for_training(
                _make_race_df(), _make_entry_df(), _make_odds_df()
            )
            assert r1.manifest.compute_hash() == r2.manifest.compute_hash()


class TestFeatureBuilderTrackConditionState:
    """FeatureState と TrackConditionFeatures の統合テスト。"""

    @patch("features.feature_builder.SubModelManager")
    @patch("features.feature_builder.FeatureEngine")
    def test_inference_uses_feature_state_track_stats(
        self, mock_fe_cls: MagicMock, mock_sm_cls: MagicMock
    ) -> None:
        """推論時に FeatureState の track_stats が使用される。"""
        mock_store = MagicMock()
        mock_fe = MagicMock()
        mock_fe_cls.return_value = mock_fe
        mock_fe.build_all.return_value = _make_feat_df()
        mock_sm = MagicMock()
        mock_sm_cls.return_value = mock_sm
        mock_sm.add_distance_band_features.side_effect = lambda df: df

        state = FeatureState(
            track_stats={"track_01": {"mean": 1.0, "std": 0.5}},
            track_month_stats={"track_01_06": {"mean": 1.2}},
            feature_version="1.0",
        )

        with patch("features.horse_history_features.HorseHistoryFeatures") as mock_hist, \
             patch("features.horse_history_features.HorseHistoryFeatures.add_race_transforms", staticmethod(lambda df: df)), \
             patch("features.pace_aptitude_features.PaceAptitudeFeatures") as mock_pace, \
             patch("features.course_features.CourseFeatures") as mock_course, \
             patch("features.sire_features.SireFeatures") as mock_sire, \
             patch("features.dam_pedigree_features.DamPedigreeFeatures") as mock_dam, \
             patch("features.record_features.RecordFeatures") as mock_record, \
             patch("features.track_condition_features.compute_track_condition_features") as mock_tcf, \
             patch("features.track_condition_features.compute_race_condition_features", side_effect=lambda df, **kw: df), \
             patch("features.interaction_features.compute_interaction_features", side_effect=lambda df: df), \
             patch("features.mining_features.MiningFeatures") as mock_mining, \
             patch("features.relative_features.compute_relative_features", side_effect=lambda df: df), \
             patch("features.jockey_context_features.JockeyContextFeatures") as mock_jockey, \
             patch("features.trainer_context_features.TrainerContextFeatures") as mock_trainer, \
             patch("features.jockey_trainer_combo.JockeyTrainerComboFeatures") as mock_jt, \
             patch("db.readers.load_sire_stats", return_value=pd.DataFrame()), \
             patch("db.readers.load_horses", return_value=pd.DataFrame()):

            mock_hist_inst = MagicMock()
            mock_hist.return_value = mock_hist_inst
            mock_hist_inst.compute.return_value = pd.DataFrame({
                "race_id": ["r0", "r1", "r2"], "umaban": [1, 2, 3],
            })

            for mod in [mock_pace, mock_course, mock_sire]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute_batch.return_value = pd.DataFrame()

            for mod in [mock_dam, mock_record, mock_mining]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute.return_value = pd.DataFrame()

            for mod in [mock_jockey, mock_trainer, mock_jt]:
                inst = MagicMock()
                mod.return_value = inst
                inst.compute.return_value = pd.DataFrame()

            builder = FeatureBuilder(store=mock_store)
            builder.build_for_inference(
                _make_race_df(), _make_entry_df(), _make_odds_df(),
                feature_state=state,
            )

            # compute_track_condition_features が FeatureState の統計を受け取ったか確認
            mock_tcf.assert_called_once()
            call_kwargs = mock_tcf.call_args[1]
            assert call_kwargs.get("track_stats") == state.track_stats
            assert call_kwargs.get("track_month_stats") == state.track_month_stats
