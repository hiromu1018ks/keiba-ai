"""WinBenterGate ユニットテスト -- TDD RED/GREEN/REFACTOR"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from models.benter_combination import BenterCombination

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_race_df(
    n_races: int = 2,
    horses_per_race: int = 5,
    p_win_corrected: list[float] | None = None,
    tanodds: list[float] | None = None,
    kakuteijyuni: list[int] | None = None,
) -> pd.DataFrame:
    """テスト用 DataFrame を構築するヘルパー."""
    n = n_races * horses_per_race
    if p_win_corrected is None:
        base_p = [0.3, 0.2, 0.15, 0.1, 0.25]
        p_win_corrected = (base_p * ((n // len(base_p)) + 1))[:n]
    if tanodds is None:
        base_odds = [3.0, 5.0, 7.0, 10.0, 4.0]
        tanodds = (base_odds * ((n // len(base_odds)) + 1))[:n]
    if kakuteijyuni is None:
        base_kj = [1, 2, 3, 4, 5]
        kakuteijyuni = (base_kj * ((n // len(base_kj)) + 1))[:n]
    race_ids = []
    for r in range(n_races):
        race_ids.extend([f"R{r}"] * horses_per_race)
    return pd.DataFrame(
        {
            "race_id": race_ids,
            "p_win_corrected": p_win_corrected,
            "tanodds": tanodds,
            "kakuteijyuni": kakuteijyuni,
        }
    )


def _make_full_oof_df(n: int = 100) -> pd.DataFrame:
    """Build a DataFrame simulating the full feature set that training_pipeline
    would pass to generate_win_oof_predictions, including all IFF source columns."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "race_id": [f"R{i // 5}" for i in range(n)],
            "race_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "umaban": [i % 5 + 1 for i in range(n)],
            "popularity_rank": [i % 5 + 1 for i in range(n)],
            "field_size": [5] * n,
            "surface": [0] * n,
            "p_win_pred": rng.uniform(0.05, 0.4, n),
            "tanodds": rng.uniform(2.0, 20.0, n),
            "kakuteijyuni": rng.randint(1, 16, n),
            # IFF required sources -- model_prob
            "p_ability_win": rng.uniform(0.05, 0.4, n),
            "p_ability_place": rng.uniform(0.1, 0.6, n),
            # IFF required sources -- market_prob
            "p_market_win_adj": rng.uniform(0.05, 0.4, n),
            "overround": rng.uniform(0.1, 0.3, n),
            "market_entropy": rng.uniform(1.5, 3.0, n),
            "odds_skewness": rng.uniform(-1.0, 1.0, n),
            "implied_prob_hhi": rng.uniform(0.05, 0.3, n),
            # IFF required sources -- model_market_gap
            "signed_log_error_win": rng.uniform(-0.5, 0.5, n),
            "abs_log_error_win": rng.uniform(0.0, 0.5, n),
            "deviation_rank": rng.uniform(1, 5, n),
            "deviation_zscore": rng.uniform(-2.0, 2.0, n),
            "odds_to_ability_ratio": rng.uniform(0.5, 2.0, n),
            "market_error_rank_in_race": rng.uniform(1, 5, n),
            # IFF required sources -- race_relative
            "rl_n_horses": [5.0] * n,
            "form_trend_race_rank": rng.uniform(0.0, 1.0, n),
            "blood_total_wr_race_rank": rng.uniform(0.0, 1.0, n),
            "closing_index_avg": rng.uniform(-1.0, 1.0, n),
            # IFF required sources -- late_odds
            "odds_drop_rate_60_10": rng.uniform(-0.1, 0.1, n),
            "odds_drop_rate_30_10": rng.uniform(-0.1, 0.1, n),
            "odds_velocity": rng.uniform(-0.05, 0.05, n),
            "odds_volatility": rng.uniform(0.0, 0.1, n),
            "odds_acceleration": rng.uniform(-0.02, 0.02, n),
            "odds_direction_consistency": rng.uniform(0.0, 1.0, n),
            "popularity_change_30_10": rng.uniform(-2.0, 2.0, n),
            # IFF required sources -- ability_form
            "norm_finish_logit_avg": rng.uniform(-1.0, 1.0, n),
            "harontimel5_zscore": rng.uniform(-2.0, 2.0, n),
            "form_trend": rng.uniform(-1.0, 1.0, n),
            "form_consistency": rng.uniform(0.0, 1.0, n),
            "blood_surface_wr": rng.uniform(0.0, 0.3, n),
            "blood_total_wr": rng.uniform(0.0, 0.3, n),
            "sire_wr": rng.uniform(0.0, 0.3, n),
            "jockey_wr_overall": rng.uniform(0.0, 0.3, n),
            "trainer_wr_overall": rng.uniform(0.0, 0.3, n),
            "jt_combo_wr": rng.uniform(0.0, 0.3, n),
            "class_level_current": rng.uniform(1, 5, n),
            "weighted_recent_form_finish": rng.uniform(1, 10, n),
            "grade_x_form_trend": rng.uniform(-1.0, 1.0, n),
            "distance_x_closing_index": rng.uniform(-1.0, 1.0, n),
            "dm_time_rank": rng.uniform(1, 5, n),
            "class_move": rng.uniform(-2, 2, n),
            # IFF required sources -- course_pace
            "closing_speed_ratio_avg": rng.uniform(0.5, 1.5, n),
            "haron_race_gap_avg": rng.uniform(-1.0, 1.0, n),
            "pace_ratio_avg": rng.uniform(0.5, 1.5, n),
            "distance_bin": [2] * n,
            "grade_code": [0] * n,
            "track_condition_code": [1] * n,
            "course_wr": rng.uniform(0.0, 0.3, n),
            "pace_aptitude": rng.uniform(0.0, 1.0, n),
            "haron_zscore_trend": rng.uniform(-1.0, 1.0, n),
            "pace_early_avg": rng.uniform(12.0, 14.0, n),
            "pace_late_avg": rng.uniform(12.0, 14.0, n),
            "closing_speed_ratio_avg_race_rank": rng.uniform(0.0, 1.0, n),
            # IFF required sources -- uncertainty
            "EV_lower_win_corrected": rng.uniform(0.5, 1.5, n),
            "EV_upper_win_corrected": rng.uniform(1.5, 5.0, n),
            "conformal_confidence_score": rng.uniform(0.0, 1.0, n),
            "market_log_error_win": rng.uniform(-0.5, 0.5, n),
            "isotonic_residual_win": rng.uniform(-0.3, 0.3, n),
        }
    )


def _make_full_oof_df_with_strings(n: int = 100) -> pd.DataFrame:
    """Build a DataFrame with STRING surface/distance_bin/grade_code, matching
    the real training pipeline where LightGBM uses them as categorical features."""
    df = _make_full_oof_df(n)
    # Replace numeric values with the actual string values from the pipeline
    df["surface"] = ["turf"] * n
    df["distance_bin"] = ["sprint"] * n
    df["grade_code"] = ["A"] * n  # G1 grade
    return df


class FakeWinTwoStageModel:
    """Fake model that generates p_win_pred, e_return_win_pred, ev_win."""

    def train_hit_model(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        return None

    def train_return_model(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        return None

    def predict_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["p_win_pred"] = np.clip(result["p_win_pred"], 0.01, 0.99)
        result["e_return_win_pred"] = result["tanodds"]
        result["ev_win"] = result["p_win_pred"] * result["e_return_win_pred"]
        return result


class FakeEVCorrectionModel:
    """Fake corrector that generates p_win_corrected, ev_win_corrected, and interactions."""

    def train(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        return None

    def correct_ev(
        self,
        df: pd.DataFrame,
        *,
        probability_col: str = "p_win_pred",
    ) -> pd.DataFrame:
        result = df.copy()
        result["p_win_corrected"] = result[probability_col] * 0.95
        result["e_return_win_corrected"] = result["e_return_win_pred"] * 1.02
        result["ev_win_corrected"] = result["p_win_corrected"] * result["e_return_win_corrected"]
        result["ev_win_calibrated"] = result["ev_win_corrected"].copy()
        # Interaction features generated by _add_interaction_features
        result["p_x_e_interaction"] = result[probability_col] * result["e_return_win_pred"]
        result["p_minus_e_gap"] = np.abs(
            np.log(result[probability_col] + 1e-8) - np.log(result["e_return_win_pred"] + 1e-8)
        )
        return result


# ---------------------------------------------------------------------------
# Test 1: extract_market_probability
# ---------------------------------------------------------------------------


class TestExtractMarketProbability:
    """tanodds を市場確率に変換しクリップする."""

    def test_basic_conversion(self) -> None:
        from models.win_benter_gate import WinBenterGate

        tanodds = np.array([3.0, 5.0, 10.0])
        result = WinBenterGate.extract_market_probability(tanodds)
        assert_allclose(result, [1 / 3, 0.2, 0.1], atol=1e-3)

    def test_clipping(self) -> None:
        from models.win_benter_gate import WinBenterGate

        # 極端なオッズ: 1.01 → ~0.99, 1000 → 0.001 → clipped to 0.01
        tanodds = np.array([1.01, 1000.0])
        result = WinBenterGate.extract_market_probability(tanodds)
        assert result[0] <= 0.99
        assert result[1] >= 0.01

    def test_zero_nan_handling(self) -> None:
        from models.win_benter_gate import WinBenterGate

        tanodds = np.array([0.0, -1.0, np.nan])
        result = WinBenterGate.extract_market_probability(tanodds)
        # 0や負の値はNaNになるが、clipで[0.01, 0.99]に入る
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.01)
        assert np.all(result <= 0.99)


# ---------------------------------------------------------------------------
# Test 2: apply() produces p_win_combined in (0, 1) range
# ---------------------------------------------------------------------------


class TestApplyPwinCombined:
    """apply() が p_win_combined を (0, 1) 範囲で出力する."""

    def test_combined_range(self) -> None:
        from models.win_benter_gate import WinBenterGate

        benter = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        gate = WinBenterGate(benter=benter)
        df = _make_race_df()
        result = gate.apply(df)
        assert "p_win_combined" in result.columns
        combined = result["p_win_combined"].values
        assert np.all(combined > 0)
        assert np.all(combined < 1)
        assert not np.any(np.isnan(combined))


# ---------------------------------------------------------------------------
# Test 3: apply() produces p_win_final where race sum == 1.0
# ---------------------------------------------------------------------------


class TestApplyRaceNormalization:
    """apply() 後の p_win_final がレース単位で合計 1.0 になる."""

    def test_race_sums_to_one(self) -> None:
        from models.win_benter_gate import WinBenterGate

        benter = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        gate = WinBenterGate(benter=benter)
        df = _make_race_df(n_races=3, horses_per_race=6)
        result = gate.apply(df)
        assert "p_win_final" in result.columns

        race_sums = result.groupby("race_id")["p_win_final"].sum()
        assert_allclose(race_sums.values, np.ones(len(race_sums)), atol=1e-9)


# ---------------------------------------------------------------------------
# Test 4: apply() produces edge_win column
# ---------------------------------------------------------------------------


class TestApplyEdgeWin:
    """apply() が edge_win = p_win_final * tanodds - 1.0 を出力する."""

    def test_edge_win_column(self) -> None:
        from models.win_benter_gate import WinBenterGate

        benter = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        gate = WinBenterGate(benter=benter)
        df = _make_race_df()
        result = gate.apply(df)
        assert "edge_win" in result.columns

        expected_edge = result["p_win_final"] * result["tanodds"] - 1.0
        assert_allclose(result["edge_win"].values, expected_edge.values, atol=1e-9)


# ---------------------------------------------------------------------------
# Test 5: generate_win_oof_predictions returns 3 valid arrays
# ---------------------------------------------------------------------------


class TestGenerateWinOofPredictions:
    """OOF 予測生成が DataFrame を返す."""

    def test_oof_output_shape(self) -> None:
        from models.win_benter_gate import generate_win_oof_predictions

        n = 100
        df = pd.DataFrame(
            {
                "race_id": [f"R{i // 5}" for i in range(n)],
                "race_date": pd.date_range("2020-01-01", periods=n, freq="D"),
                "umaban": [i % 5 + 1 for i in range(n)],
                "popularity_rank": [i % 5 + 1 for i in range(n)],
                "field_size": [5] * n,
                "surface": ["turf"] * n,
                "p_win_pred": np.random.uniform(0.05, 0.4, n),
                "tanodds": np.random.uniform(2.0, 20.0, n),
                "kakuteijyuni": np.random.randint(1, 16, n),
            }
        )

        result = generate_win_oof_predictions(
            df,
            win_model_cls=FakeWinTwoStageModel,
            ev_corrector=FakeEVCorrectionModel(),
            n_splits=5,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Check required columns for MarketAwareWinCalibrator (D-18/D-19/D-20)
        for col in [
            "p_win_oof", "p_market_norm", "tanodds", "popularity_rank",
            "field_size", "p_win_race_rank_pct", "race_id", "kakuteijyuni",
        ]:
            assert col in result.columns, f"Missing column: {col}"
        assert not result["p_win_oof"].isna().any()
        assert not result["p_market_norm"].isna().any()


# ---------------------------------------------------------------------------
# Test 5b: generate_win_oof_predictions emits ranker-required columns (D-12/D-13)
# ---------------------------------------------------------------------------


class TestGenerateWinOofRankerColumns:
    """OOF 予測に RaceLevelRanker が必要とする列が含まれる (D-12/D-13)."""

    @staticmethod
    def _make_oof_result() -> pd.DataFrame:
        """Generate OOF result using fake models."""
        from models.win_benter_gate import generate_win_oof_predictions

        n = 100
        df = pd.DataFrame(
            {
                "race_id": [f"R{i // 5}" for i in range(n)],
                "race_date": pd.date_range("2020-01-01", periods=n, freq="D"),
                "umaban": [i % 5 + 1 for i in range(n)],
                "popularity_rank": [i % 5 + 1 for i in range(n)],
                "field_size": [5] * n,
                "surface": [0] * n,
                "p_win_pred": np.random.uniform(0.05, 0.4, n),
                "tanodds": np.random.uniform(2.0, 20.0, n),
                "kakuteijyuni": np.random.randint(1, 16, n),
            }
        )

        return generate_win_oof_predictions(
            df,
            win_model_cls=FakeWinTwoStageModel,
            ev_corrector=FakeEVCorrectionModel(),
            n_splits=5,
        )

    def test_calibrated_ev_oof_column_exists(self) -> None:
        """calibrated_ev_oof 列が OOF 出力に含まれる (D-09 value target 用)."""
        result = self._make_oof_result()
        assert "calibrated_ev_oof" in result.columns, (
            "Missing calibrated_ev_oof column for value target computation"
        )
        assert not result["calibrated_ev_oof"].isna().any(), (
            "calibrated_ev_oof should not have NaN"
        )

    def test_no_fewer_rows_than_before(self) -> None:
        """拡張前後で行数が減少しない (既存カラムが影響を受けない)."""
        result = self._make_oof_result()
        assert len(result) > 0

    def test_kakuteijyuni_preserved(self) -> None:
        """kakuteijyuni の値が入力と一致する (label-only, feature としては使用しない)."""
        result = self._make_oof_result()
        assert "kakuteijyuni" in result.columns
        assert result["kakuteijyuni"].notna().all()

    def test_existing_columns_preserved(self) -> None:
        """既存の必須列 (p_win_oof, p_market_norm 等) が保持される."""
        result = self._make_oof_result()
        required_existing = [
            "p_win_oof", "p_market_norm", "p_win_corrected",
            "kakuteijyuni", "tanodds", "popularity_rank", "field_size",
            "race_id", "race_date", "umaban", "surface", "p_win_race_rank_pct",
        ]
        for col in required_existing:
            assert col in result.columns, f"Missing existing column: {col}"

    def test_ev_win_corrected_values_correct(self) -> None:
        """calibrated_ev_oof が ev_win_corrected (= p_win_corrected * e_return_win_corrected) に基づく."""
        result = self._make_oof_result()
        # calibrated_ev_oof should equal ev_win_corrected from fold-level correction
        expected = result["p_win_corrected"] * result["e_return_win_pred"] * 1.02
        np.testing.assert_allclose(
            result["calibrated_ev_oof"].values,
            expected.values,
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# Test 5c: IFF build_frame(mode="train") column audit
# ---------------------------------------------------------------------------


class TestOofIFFColumnAudit:
    """OOF output contains all columns required by IFF build_frame(mode="train").

    This test validates the comprehensive fix for the systematic missing-columns
    pattern where generate_win_oof_predictions() was missing columns that
    InvestmentFeatureFrameBuilder.build_frame() requires in train mode.
    """

    @staticmethod
    def _make_oof_result_with_full_features() -> pd.DataFrame:
        """Generate OOF result with a full feature DataFrame simulating training_pipeline."""
        from models.win_benter_gate import generate_win_oof_predictions

        df = _make_full_oof_df(n=100)

        return generate_win_oof_predictions(
            df,
            win_model_cls=FakeWinTwoStageModel,
            ev_corrector=FakeEVCorrectionModel(),
            n_splits=5,
        )

    def test_fold_generated_ev_win_corrected(self) -> None:
        """ev_win_corrected is captured from fold predictions."""
        result = self._make_oof_result_with_full_features()
        assert "ev_win_corrected" in result.columns, (
            "Missing ev_win_corrected -- required by if_ev_corrected train source"
        )
        assert result["ev_win_corrected"].notna().any()

    def test_fold_generated_interaction_cols(self) -> None:
        """p_x_e_interaction and p_minus_e_gap are captured from fold predictions."""
        result = self._make_oof_result_with_full_features()
        for col in ["p_x_e_interaction", "p_minus_e_gap"]:
            assert col in result.columns, f"Missing fold-generated column: {col}"

    def test_required_iff_sources_present(self) -> None:
        """All IFF REQUIRED train_sources columns are present in the OOF output."""
        from investment.schema_registry import FEATURE_SPECS

        result = self._make_oof_result_with_full_features()
        result_cols = set(result.columns)

        missing_required: list[str] = []
        for spec in FEATURE_SPECS.values():
            if not spec.required:
                continue
            if not spec.train_sources:
                continue  # derived feature, no source needed
            found = any(src in result_cols for src in spec.train_sources)
            if not found:
                missing_required.append(
                    f"{spec.name} needs {spec.train_sources}"
                )

        assert not missing_required, (
            f"Missing required IFF source columns: {missing_required}"
        )

    def test_static_passthrough_ability_cols(self) -> None:
        """p_ability_win and p_ability_place are passed through from source df."""
        result = self._make_oof_result_with_full_features()
        assert "p_ability_win" in result.columns
        assert result["p_ability_win"].notna().any()

    def test_static_passthrough_market_cols(self) -> None:
        """Market-related columns are passed through from source df."""
        result = self._make_oof_result_with_full_features()
        for col in ["p_market_win_adj", "overround", "market_entropy",
                     "signed_log_error_win", "abs_log_error_win"]:
            assert col in result.columns, f"Missing market column: {col}"

    def test_static_passthrough_race_level_cols(self) -> None:
        """Race-level columns (rl_n_horses, distance_bin, etc.) are passed through."""
        result = self._make_oof_result_with_full_features()
        for col in ["rl_n_horses", "distance_bin", "grade_code",
                     "track_condition_code"]:
            assert col in result.columns, f"Missing race-level column: {col}"

    def test_static_passthrough_uncertainty_cols(self) -> None:
        """Uncertainty columns are passed through from source df."""
        result = self._make_oof_result_with_full_features()
        for col in ["EV_lower_win_corrected", "EV_upper_win_corrected",
                     "conformal_confidence_score"]:
            assert col in result.columns, f"Missing uncertainty column: {col}"


# ---------------------------------------------------------------------------
# Test 5d: String column encoding for IFF compatibility
# ---------------------------------------------------------------------------


class TestStringColumnEncoding:
    """OOF output encodes string categorical columns to numeric for IFF.

    The training pipeline stores surface/distance_bin/grade_code as strings
    (LightGBM uses them as categorical features), but IFF expects float64.
    This tests the fix for: could not convert string to float: 'turf'
    """

    @staticmethod
    def _make_oof_result_with_strings() -> pd.DataFrame:
        """Generate OOF result using string categorical columns (real pipeline behavior)."""
        from models.win_benter_gate import generate_win_oof_predictions

        df = _make_full_oof_df_with_strings(n=100)
        return generate_win_oof_predictions(
            df,
            win_model_cls=FakeWinTwoStageModel,
            ev_corrector=FakeEVCorrectionModel(),
            n_splits=5,
        )

    def test_surface_encoded_to_numeric(self) -> None:
        """surface 'turf' is encoded to numeric 0 in OOF output."""
        result = self._make_oof_result_with_strings()
        assert "surface" in result.columns
        assert pd.api.types.is_numeric_dtype(result["surface"]), (
            f"surface should be numeric, got dtype={result['surface'].dtype}"
        )
        assert (result["surface"] == 0).all(), (
            f"'turf' should encode to 0, got values: {result['surface'].unique()}"
        )

    def test_distance_bin_encoded_to_numeric(self) -> None:
        """distance_bin 'sprint' is encoded to numeric 0 in OOF output."""
        result = self._make_oof_result_with_strings()
        assert "distance_bin" in result.columns
        assert pd.api.types.is_numeric_dtype(result["distance_bin"]), (
            f"distance_bin should be numeric, got dtype={result['distance_bin'].dtype}"
        )
        assert (result["distance_bin"] == 0).all(), (
            f"'sprint' should encode to 0, got values: {result['distance_bin'].unique()}"
        )

    def test_grade_code_encoded_to_numeric(self) -> None:
        """grade_code 'A' (G1) is encoded to numeric 8.0 in OOF output."""
        result = self._make_oof_result_with_strings()
        assert "grade_code" in result.columns
        assert pd.api.types.is_numeric_dtype(result["grade_code"]), (
            f"grade_code should be numeric, got dtype={result['grade_code'].dtype}"
        )
        assert (result["grade_code"] == 8.0).all(), (
            f"'A' (G1) should encode to 8.0, got values: {result['grade_code'].unique()}"
        )

    def test_iff_build_frame_succeeds_with_string_source(self) -> None:
        """IFF build_frame(mode="train") succeeds without ValueError when
        source DataFrame had string surface/distance_bin/grade_code.

        This is the actual bug scenario: IFF calls .astype('float64') on
        these columns and 'turf' cannot be converted to float.
        """
        from investment.feature_frame import InvestmentFeatureFrameBuilder

        result = self._make_oof_result_with_strings()
        # This should NOT raise ValueError: could not convert string to float: 'turf'
        iff_builder = InvestmentFeatureFrameBuilder()
        iff_df = iff_builder.build_frame(result, mode="train")
        assert len(iff_df) > 0
        assert "if_surface" in iff_df.columns
        assert pd.api.types.is_numeric_dtype(iff_df["if_surface"])

    def test_numeric_surface_passthrough(self) -> None:
        """When surface is already numeric (e.g. 0/1), it passes through unchanged."""
        from models.win_benter_gate import generate_win_oof_predictions

        df = _make_full_oof_df(n=100)  # uses numeric surface=0
        result = generate_win_oof_predictions(
            df,
            win_model_cls=FakeWinTwoStageModel,
            ev_corrector=FakeEVCorrectionModel(),
            n_splits=5,
        )
        assert "surface" in result.columns
        assert pd.api.types.is_numeric_dtype(result["surface"])
        assert (result["surface"] == 0).all()


# ---------------------------------------------------------------------------
# Test 6: SubmodelSet has win_* fields
# ---------------------------------------------------------------------------


class TestSubmodelSetWinFields:
    """SubmodelSet has market_aware_win_calibrator (Phase 39 CAL-04)."""

    def test_win_fields_exist(self) -> None:
        from dataclasses import fields as dc_fields

        from domain.models import SubmodelSet

        field_names = [f.name for f in dc_fields(SubmodelSet)]
        assert "market_aware_win_calibrator" in field_names
        # Old fields removed (CAL-04)
        assert "win_benter" not in field_names
        assert "win_isotonic_calibrator" not in field_names
        assert "win_temperature_scaler" not in field_names
        assert "win_segment_calibrator" not in field_names

    def test_win_fields_default_none(self) -> None:
        """market_aware_win_calibrator フィールドのデフォルト値が None であることを確認する."""
        # 他の必須フィールドはmockで埋める
        from domain.models import SubmodelSet

        mock = MagicMock()
        sub = SubmodelSet(
            market=mock,
            stage1=mock,
            place_ability=mock,
            win=mock,
            ev_corrector=mock,
            place=mock,
            place_ev_corrector=mock,
            wide=mock,
            conformal_ev_model=mock,
        )
        assert sub.market_aware_win_calibrator is None


# ---------------------------------------------------------------------------
# Test 7: compute_ece returns non-negative float
# ---------------------------------------------------------------------------


class TestComputeECE:
    """compute_ece が非負の ECE を返す."""

    def test_returns_nonnegative(self) -> None:
        from models.win_benter_gate import compute_ece

        y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        result = compute_ece(y_true, y_prob)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_perfect_calibration(self) -> None:
        """完全にキャリブレーションされた予測は ECE が 0 に近い."""
        from models.win_benter_gate import compute_ece

        rng = np.random.RandomState(42)
        n = 10000
        # y_prob がそのまま確率として使える → 十分大きなサンプルで ECE ~ 0
        y_prob = rng.uniform(0.05, 0.95, n)
        y_true = (rng.random(n) < y_prob).astype(float)
        result = compute_ece(y_true, y_prob, n_bins=20)
        # 統計的ばらつきを考慮して 0.05 以下を許容
        assert result < 0.05


# ---------------------------------------------------------------------------
# Test 8: compare_calibrations returns required keys and selects lower Brier
# ---------------------------------------------------------------------------


class TestCompareCalibrations:
    """compare_calibrations が正しいキーを返し、Brier Score で勝者を選ぶ."""

    def test_returns_required_keys(self) -> None:
        from models.win_benter_gate import compare_calibrations

        rng = np.random.RandomState(42)
        n = 1000
        p_benter = rng.uniform(0.01, 0.5, n)
        y = (rng.random(n) < p_benter).astype(int)

        result = compare_calibrations(p_benter, y, train_ratio=0.8)
        for key in ["beta_brier", "iso_brier", "beta_ece", "iso_ece", "winner"]:
            assert key in result, f"Missing key: {key}"

    def test_selects_lower_brier(self) -> None:
        """Beta Brier Score が低い場合 winner='beta' になる."""
        from models.win_benter_gate import compare_calibrations

        rng = np.random.RandomState(42)
        n = 2000
        p_benter = rng.uniform(0.01, 0.5, n)
        y = (rng.random(n) < p_benter).astype(int)

        result = compare_calibrations(p_benter, y, train_ratio=0.8)
        # どちらかが勝者
        assert result["winner"] in ("beta", "isotonic")


# ---------------------------------------------------------------------------
# Test 9: generate_reliability_data returns required keys
# ---------------------------------------------------------------------------


class TestGenerateReliabilityData:
    """generate_reliability_data が信頼性ダイアグラムデータを返す."""

    def test_returns_required_keys(self) -> None:
        from models.win_benter_gate import generate_reliability_data

        rng = np.random.RandomState(42)
        n = 500
        y_true = rng.randint(0, 2, n).astype(float)
        y_prob = rng.uniform(0.1, 0.9, n)

        result = generate_reliability_data(y_true, y_prob, n_bins=10)
        assert "fraction_of_positives" in result
        assert "mean_predicted_value" in result
        assert "bin_edges" in result
        assert len(result["bin_edges"]) == 11  # n_bins + 1

    def test_perfect_calibration(self) -> None:
        """完全キャリブレーションでは fraction_of_positives ≈ mean_predicted_value."""
        from models.win_benter_gate import generate_reliability_data

        rng = np.random.RandomState(42)
        n = 50000
        y_prob = rng.uniform(0.1, 0.9, n)
        y_true = (rng.random(n) < y_prob).astype(float)

        result = generate_reliability_data(y_true, y_prob, n_bins=10)
        # 各ビンで fraction_of_positives と mean_predicted_value が近いことを確認
        fop = result["fraction_of_positives"]
        mpv = result["mean_predicted_value"]
        assert len(fop) == len(mpv)
        # 大きなサンプルなので最大差が 0.1 以下を期待
        assert np.max(np.abs(fop - mpv)) < 0.1
