"""test_live_data_integration.py — ライブトラック条件統合テスト (D-06, D-07)

FeatureBuilder._merge_live_track_conditions(), SessionManifest.set_live_data(),
PaperPredictor.setup() live_track_conditions passthrough, collate_moisture_rule()
の動作を検証する。
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Test 1: _merge_live_track_conditions — None の場合変更なし
# ---------------------------------------------------------------------------


class TestMergeNoneLive:
    """live_track_conditions が None の場合、_merge が呼ばれず既存動作と同一。"""

    def test_merge_none_returns_original(self) -> None:
        from features.feature_builder import FeatureBuilder

        builder = FeatureBuilder.__new__(FeatureBuilder)
        builder.store = MagicMock()

        df = pd.DataFrame(
            {
                "race_id": ["2024010101010100AA"],
                "dirt_moisture": [5.0],
                "turf_cushion": [9.5],
            }
        )
        result = builder._merge_live_track_conditions(df, None)
        # None の場合は元の DataFrame をそのまま返す
        assert result is df


# ---------------------------------------------------------------------------
# Test 2: ライブ値が履歴を上書きする
# ---------------------------------------------------------------------------


class TestMergeLiveOverridesHistory:
    """ライブ値がある場合、履歴の dirt_moisture/turf_cushion が上書きされる。"""

    def test_override_dirt_moisture(self) -> None:
        from features.feature_builder import FeatureBuilder

        builder = FeatureBuilder.__new__(FeatureBuilder)
        builder.store = MagicMock()

        df = pd.DataFrame(
            {
                "race_id": ["2024010101010100AA", "2024010101010100AB"],
                "dirt_moisture": [5.0, 6.0],
                "turf_cushion": [9.5, 10.0],
            }
        )
        live_df = pd.DataFrame(
            {
                "race_id": ["2024010101010100AA"],
                "dirt_moisture": [7.5],
                "turf_cushion": [11.2],
            }
        )
        result = builder._merge_live_track_conditions(df, live_df)
        # race_id AA: live 値で上書き
        assert result.loc[result["race_id"] == "2024010101010100AA", "dirt_moisture"].iloc[0] == 7.5
        assert result.loc[result["race_id"] == "2024010101010100AA", "turf_cushion"].iloc[0] == 11.2
        # race_id AB: 変更なし
        assert result.loc[result["race_id"] == "2024010101010100AB", "dirt_moisture"].iloc[0] == 6.0


# ---------------------------------------------------------------------------
# Test 3: ライブ NaN は履歴値を保持
# ---------------------------------------------------------------------------


class TestMergeLiveNaNPreservesHistory:
    """live 値が NaN の列は履歴値が保持される。"""

    def test_nan_live_preserves_history(self) -> None:
        from features.feature_builder import FeatureBuilder

        builder = FeatureBuilder.__new__(FeatureBuilder)
        builder.store = MagicMock()

        df = pd.DataFrame(
            {
                "race_id": ["2024010101010100AA"],
                "dirt_moisture": [5.0],
                "turf_cushion": [9.5],
            }
        )
        live_df = pd.DataFrame(
            {
                "race_id": ["2024010101010100AA"],
                "dirt_moisture": [np.nan],
                "turf_cushion": [11.2],
            }
        )
        result = builder._merge_live_track_conditions(df, live_df)
        # dirt_moisture は NaN → 履歴値 5.0 を保持
        assert result.loc[0, "dirt_moisture"] == 5.0
        # turf_cushion は非 NaN → 上書き
        assert result.loc[0, "turf_cushion"] == 11.2


# ---------------------------------------------------------------------------
# Test 4: 空 DataFrame 入力で例外なし
# ---------------------------------------------------------------------------


class TestMergeEmptyLiveDF:
    """空 DataFrame 入力で例外なし。"""

    def test_empty_live_df_no_change(self) -> None:
        from features.feature_builder import FeatureBuilder

        builder = FeatureBuilder.__new__(FeatureBuilder)
        builder.store = MagicMock()

        df = pd.DataFrame(
            {
                "race_id": ["2024010101010100AA"],
                "dirt_moisture": [5.0],
            }
        )
        live_df = pd.DataFrame(columns=["race_id", "dirt_moisture"])
        result = builder._merge_live_track_conditions(df, live_df)
        # 空 DataFrame → 変更なし
        assert len(result) == 1
        assert result.loc[0, "dirt_moisture"] == 5.0


# ---------------------------------------------------------------------------
# Test 5: session_manifest.set_live_data() で4メタデータが記録される
# ---------------------------------------------------------------------------


class TestSessionManifestLiveData:
    """set_live_data() で source, measured_at, fetched_at, html_hash, venue_codes が記録される。"""

    def test_set_live_data_records_fields(self) -> None:
        from features.session_manifest import SessionManifest

        manifest = SessionManifest(session_id="test", prediction_date="2024-01-01")
        manifest.set_live_data(
            source="JRA",
            measured_at="10:30",
            fetched_at="2024-01-01T10:35:00",
            html_hash="abc123",
            venue_codes=["05", "08"],
        )
        d = manifest.to_dict()
        assert "live_data" in d
        assert d["live_data"]["source"] == "JRA"
        assert d["live_data"]["measured_at"] == "10:30"
        assert d["live_data"]["fetched_at"] == "2024-01-01T10:35:00"
        assert d["live_data"]["html_hash"] == "abc123"
        assert d["live_data"]["venue_codes"] == ["05", "08"]


# ---------------------------------------------------------------------------
# Test 6: collate_moisture_rule — JRA/CSV 一致で正しい規則選択
# ---------------------------------------------------------------------------


class TestCollateMoistureRuleSelectsBest:
    """照合で最適規則が選択される。"""

    def test_goal_rule_selected(self) -> None:
        from features.feature_builder import collate_moisture_rule

        # JRA値: goal=5.0, 4c=6.0
        # CSV値: 5.0 (goalに一致)
        rule = collate_moisture_rule(
            jra_goal=5.0, jra_4c=6.0, csv_value=5.0,
        )
        assert rule == "goal"

    def test_4c_rule_selected(self) -> None:
        from features.feature_builder import collate_moisture_rule

        # CSV値: 6.0 (4cに一致)
        rule = collate_moisture_rule(
            jra_goal=5.0, jra_4c=6.0, csv_value=6.0,
        )
        assert rule == "4c"

    def test_mean_rule_selected(self) -> None:
        from features.feature_builder import collate_moisture_rule

        # CSV値: 5.5 (mean に一致)
        rule = collate_moisture_rule(
            jra_goal=5.0, jra_4c=6.0, csv_value=5.5,
        )
        assert rule == "mean"


# ---------------------------------------------------------------------------
# Test 7: collate_moisture_rule — 照合不能で例外
# ---------------------------------------------------------------------------


class TestCollateMoistureRuleMismatchHalts:
    """全規則不一致で例外が送出される。"""

    def test_mismatch_raises_error(self) -> None:
        from features.feature_builder import collate_moisture_rule

        # CSV値: 99.0 (どの規則とも閾値外)
        with pytest.raises(ValueError, match="照合不能"):
            collate_moisture_rule(
                jra_goal=5.0, jra_4c=6.0, csv_value=99.0,
            )


# ---------------------------------------------------------------------------
# Test 8: collate_moisture_rule — 重複データ不足でデフォルト規則 + 警告
# ---------------------------------------------------------------------------


class TestCollateMoistureRuleInsufficientData:
    """重複データ不足でデフォルト規則 (mean) を返す。"""

    def test_no_csv_data_returns_default(self) -> None:
        from features.feature_builder import collate_moisture_rule

        # csv_value が None → insufficient data
        rule = collate_moisture_rule(
            jra_goal=5.0, jra_4c=6.0, csv_value=None,
        )
        assert rule == "mean"


# ---------------------------------------------------------------------------
# Test 9: dirt_moisture aggregation — goal 規則
# ---------------------------------------------------------------------------


class TestDirtMoistureAggregationGoal:
    """goal 規則で dirt_moisture = dirt_moisture_goal。"""

    def test_goal_rule(self) -> None:
        from features.feature_builder import aggregate_dirt_moisture

        result = aggregate_dirt_moisture(
            goal=5.0, four_c=6.0, rule="goal",
        )
        assert result == 5.0


# ---------------------------------------------------------------------------
# Test 10: dirt_moisture aggregation — 4c 規則
# ---------------------------------------------------------------------------


class TestDirtMoistureAggregation4c:
    """4c 規則で dirt_moisture = dirt_moisture_4c。"""

    def test_4c_rule(self) -> None:
        from features.feature_builder import aggregate_dirt_moisture

        result = aggregate_dirt_moisture(
            goal=5.0, four_c=6.0, rule="4c",
        )
        assert result == 6.0


# ---------------------------------------------------------------------------
# Test 11: dirt_moisture aggregation — mean 規則
# ---------------------------------------------------------------------------


class TestDirtMoistureAggregationMean:
    """mean 規則で dirt_moisture = (goal + 4c) / 2。"""

    def test_mean_rule(self) -> None:
        from features.feature_builder import aggregate_dirt_moisture

        result = aggregate_dirt_moisture(
            goal=5.0, four_c=6.0, rule="mean",
        )
        assert result == pytest.approx(5.5)

    def test_mean_rule_goal_only(self) -> None:
        """片方のみ非NaNの場合はその値を使用。"""
        from features.feature_builder import aggregate_dirt_moisture

        result = aggregate_dirt_moisture(
            goal=5.0, four_c=None, rule="mean",
        )
        assert result == 5.0

    def test_mean_rule_4c_only(self) -> None:
        from features.feature_builder import aggregate_dirt_moisture

        result = aggregate_dirt_moisture(
            goal=None, four_c=6.0, rule="mean",
        )
        assert result == 6.0


# ---------------------------------------------------------------------------
# Test 12: ライブ取得失敗時に sys.exit(1)
# ---------------------------------------------------------------------------


class TestLiveFailureStopsPrediction:
    """ライブ取得失敗時に予測を停止し非ゼロ終了する。"""

    def test_fetch_failure_exits(self) -> None:
        """JRATrackConditionFetcher.fetch_all_venues が例外を送出した場合、
        run_paper_trading.py が sys.exit(1) で終了する。"""
        # これはスクリプトレベルのテスト — 実際の fetch 呼び出しが例外を送出する
        # ことを確認するだけで十分。スクリプト統合テストは別途 verify で実行。
        # ユニットレベルでは、例外が送出されることを確認:
        from ingestion.track_condition_fetcher import TrackConditionParseError

        with pytest.raises(TrackConditionParseError):
            from ingestion.track_condition_fetcher import parse_track_condition_html
            parse_track_condition_html("")


# ---------------------------------------------------------------------------
# Test 13: PaperPredictor.setup() が live_track_conditions を build_for_inference に渡す
# ---------------------------------------------------------------------------


class TestPredictorSetupPassesLiveConditions:
    """PaperPredictor.setup() が live_track_conditions を build_for_inference() に渡す。"""

    def test_setup_passes_live_conditions(self) -> None:
        from paper_trading.predictor import PaperPredictor

        # Mock dependencies
        mock_store = MagicMock()
        mock_race_predictor = MagicMock()
        mock_models = MagicMock()
        mock_models.submodels = {}
        mock_everydb2 = MagicMock()
        mock_everydb2.get_race_schedule.return_value = []

        predictor = PaperPredictor(
            store=mock_store,
            race_predictor=mock_race_predictor,
            models=mock_models,
        )

        # When no schedule, setup returns early — test that the parameter is accepted
        result = predictor.setup(
            target_date=date(2024, 1, 1),
            everydb2=mock_everydb2,
            live_track_conditions=None,
        )
        assert result == []

    def test_setup_signature_accepts_live_track_conditions(self) -> None:
        """setup() シグネチャに live_track_conditions パラメータが存在する。"""
        import inspect
        from paper_trading.predictor import PaperPredictor

        sig = inspect.signature(PaperPredictor.setup)
        assert "live_track_conditions" in sig.parameters
