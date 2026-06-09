"""Tests for RaceLevelRanker -- learned ranker with Ridge models.

TDD RED phase: all tests MUST fail before implementation.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge


def _make_oof_df(
    *,
    n_races: int = 30,
    n_horses_per_race: int = 10,
    surface_value: int | None = None,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Helper to create a synthetic OOF DataFrame for training.

    When surface_value is provided, all rows use that surface.
    When None, surfaces alternate between 0 (turf) and 1 (dirt).
    """
    rng = np.random.RandomState(rng_seed)
    rows: list[dict] = []
    for r in range(n_races):
        race_id = f"R{r:04d}"
        if surface_value is not None:
            surface = surface_value
        else:
            surface = 0 if r % 2 == 0 else 1  # 0=turf, 1=dirt
        for h in range(n_horses_per_race):
            umaban = h + 1
            kakuteijyuni = h + 1
            if h == 0:
                kakuteijyuni = 1
            else:
                kakuteijyuni = rng.randint(2, n_horses_per_race + 1)

            p_win = rng.uniform(0.02, 0.25)
            p_ability_win = rng.uniform(0.02, 0.30)
            p_market = 1.0 / rng.uniform(2.0, 30.0)
            ev = p_win * rng.uniform(2.0, 30.0)

            rows.append(
                {
                    "race_id": race_id,
                    "umaban": umaban,
                    "race_date": f"2023-{(r // 5) + 1:02d}-01",
                    "surface": surface,
                    "kakuteijyuni": kakuteijyuni,
                    # Model prob features
                    "if_p_win_final": p_win,
                    "if_p_win_race_rank": rng.uniform(0.0, 1.0),
                    "if_p_ability_win": p_ability_win,
                    "if_norm_finish_avg": rng.uniform(-1.0, 1.0),
                    "if_closing_index": rng.uniform(0.0, 1.0),
                    "if_weighted_recent_form": rng.uniform(-1.0, 1.0),
                    "if_jockey_wr": rng.uniform(0.05, 0.20),
                    "if_trainer_wr": rng.uniform(0.05, 0.20),
                    "if_blood_surface_wr": rng.uniform(0.05, 0.20),
                    "if_class_level": rng.uniform(1.0, 5.0),
                    "if_surface": float(surface),
                    "if_distance_bin": rng.uniform(0.0, 3.0),
                    "if_grade_code": rng.uniform(0.0, 5.0),
                    "if_n_horses": float(n_horses_per_race),
                    # Value features
                    "if_logit_gap": rng.uniform(-2.0, 2.0),
                    "if_edge_win": rng.uniform(-0.5, 0.5),
                    "if_ev_calibrated": ev,
                    "if_odds_log": rng.uniform(0.5, 4.0),
                    "if_odds_band_id": rng.uniform(1.0, 7.0),
                    "if_odds_drop_60_10": rng.uniform(-0.1, 0.1),
                    "if_odds_drop_30_10": rng.uniform(-0.1, 0.1),
                    "if_overround": rng.uniform(0.15, 0.30),
                    "if_market_entropy": rng.uniform(0.5, 2.0),
                    "if_conformal_width": rng.uniform(0.1, 2.0),
                    "if_ev_uncertainty_ratio": rng.uniform(0.0, 1.0),
                    # Target construction columns
                    "calibrated_ev_oof": ev,
                    "p_win_oof": p_win,
                    "p_market_norm": p_market,
                }
            )
    df = pd.DataFrame(rows)
    for race_id in df["race_id"].unique():
        mask = df["race_id"] == race_id
        n = mask.sum()
        df.loc[mask, "kakuteijyuni"] = np.arange(1, n + 1)
    return df


class TestRaceLevelRankerInit:
    """RaceLevelRanker initializes with all None models, _trained=False."""

    def test_init_defaults(self) -> None:
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        assert rlr.relevance_scorer_turf is None
        assert rlr.relevance_scorer_dirt is None
        assert rlr.value_scorer_turf is None
        assert rlr.value_scorer_dirt is None
        assert rlr._trained is False  # noqa: SLF001
        assert rlr.training_summary == {}

    def test_is_trained_false_initially(self) -> None:
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        assert rlr.is_trained is False


class TestRaceLevelRankerTrain:
    """train() fits per-surface Ridge models with alpha grid selection."""

    @pytest.fixture()
    def sample_oof_df(self) -> pd.DataFrame:
        """Create a synthetic OOF DataFrame for training (both surfaces)."""
        return _make_oof_df()

    def test_train_sets_trained_true(self, sample_oof_df: pd.DataFrame) -> None:
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        rlr.train(sample_oof_df)
        assert rlr.is_trained is True

    def test_train_populates_ridge_models(self, sample_oof_df: pd.DataFrame) -> None:
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        rlr.train(sample_oof_df)
        assert isinstance(rlr.relevance_scorer_turf, Ridge)
        assert isinstance(rlr.relevance_scorer_dirt, Ridge)
        assert isinstance(rlr.value_scorer_turf, Ridge)
        assert isinstance(rlr.value_scorer_dirt, Ridge)

    def test_train_alpha_grid_selection(self, sample_oof_df: pd.DataFrame) -> None:
        """train() selects best alpha from ALPHA_GRID."""
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        rlr.train(sample_oof_df)
        summary = rlr.training_summary
        # Should contain best_alpha per surface for both scorers
        for surface in ["turf", "dirt"]:
            assert f"relevance_best_alpha_{surface}" in summary
            assert f"value_best_alpha_{surface}" in summary
            assert summary[f"relevance_best_alpha_{surface}"] in RaceLevelRanker.ALPHA_GRID
            assert summary[f"value_best_alpha_{surface}"] in RaceLevelRanker.ALPHA_GRID

    def test_train_single_surface_numeric_no_spurious_warning(self) -> None:
        """train() with only numeric surface=0 (turf) must not emit
        'Insufficient data for dirt' warning (bug from hardcoded surface list).

        This reproduces the pipeline scenario where _train_submodel() trains
        on a single surface and passes the resulting oof_cal_df (with numeric
        surface) to RaceLevelRanker.train().
        """
        from models.race_level_ranker import RaceLevelRanker

        # All turf (surface=0, numeric as produced by _encode_string_columns)
        df_turf = _make_oof_df(surface_value=0)

        # Capture log warnings to verify no spurious "Insufficient data" messages
        import io

        log_buffer = io.StringIO()
        handler = logging.Handler()
        handler.emit = lambda record: log_buffer.emit(record.getMessage())  # type: ignore[attr-defined]
        test_logger = logging.getLogger("models.race_level_ranker")
        test_logger.addHandler(handler)
        old_level = test_logger.level
        test_logger.setLevel(logging.WARNING)

        try:
            rlr = RaceLevelRanker()
            rlr.train(df_turf)
        finally:
            test_logger.removeHandler(handler)
            test_logger.setLevel(old_level)

        log_output = log_buffer.getvalue()
        assert "Insufficient data" not in log_output, (
            f"Spurious 'Insufficient data' warning in log: {log_output}"
        )

        # Only turf models should be trained
        assert isinstance(rlr.relevance_scorer_turf, Ridge)
        assert isinstance(rlr.value_scorer_turf, Ridge)
        assert rlr.relevance_scorer_dirt is None
        assert rlr.value_scorer_dirt is None

    def test_train_single_surface_string_no_spurious_warning(self) -> None:
        """train() with string surface='dirt' works correctly (dual-format)."""
        from models.race_level_ranker import RaceLevelRanker

        # All dirt (string, as would be before _encode_string_columns)
        df_dirt = _make_oof_df(surface_value=1)
        # Override with string values to test both code paths
        df_dirt["surface"] = "dirt"

        import io

        log_buffer = io.StringIO()
        handler = logging.Handler()
        handler.emit = lambda record: log_buffer.emit(record.getMessage())  # type: ignore[attr-defined]
        test_logger = logging.getLogger("models.race_level_ranker")
        test_logger.addHandler(handler)
        old_level = test_logger.level
        test_logger.setLevel(logging.WARNING)

        try:
            rlr = RaceLevelRanker()
            rlr.train(df_dirt)
        finally:
            test_logger.removeHandler(handler)
            test_logger.setLevel(old_level)

        log_output = log_buffer.getvalue()
        assert "Insufficient data" not in log_output, (
            f"Spurious 'Insufficient data' warning in log: {log_output}"
        )

        # Only dirt models should be trained
        assert rlr.relevance_scorer_turf is None
        assert isinstance(rlr.relevance_scorer_dirt, Ridge)
        assert rlr.value_scorer_turf is None
        assert isinstance(rlr.value_scorer_dirt, Ridge)


class TestRaceLevelRankerScore:
    """score() produces investment_score columns."""

    @pytest.fixture()
    def trained_ranker(self) -> "RaceLevelRanker":
        from models.race_level_ranker import RaceLevelRanker

        df = _make_oof_df()
        rlr = RaceLevelRanker()
        rlr.train(df)
        return rlr

    def test_score_adds_investment_columns(
        self,
        trained_ranker: "RaceLevelRanker",
    ) -> None:
        """score() adds all required investment score columns."""
        rng = np.random.RandomState(99)
        rows: list[dict] = []
        for h in range(10):
            rows.append(
                {
                    "race_id": "R_SCORE_01",
                    "umaban": h + 1,
                    "if_surface": 0.0,
                    "if_p_win_final": rng.uniform(0.02, 0.25),
                    "if_p_win_race_rank": rng.uniform(0.0, 1.0),
                    "if_p_ability_win": rng.uniform(0.02, 0.30),
                    "if_norm_finish_avg": rng.uniform(-1.0, 1.0),
                    "if_closing_index": rng.uniform(0.0, 1.0),
                    "if_weighted_recent_form": rng.uniform(-1.0, 1.0),
                    "if_jockey_wr": rng.uniform(0.05, 0.20),
                    "if_trainer_wr": rng.uniform(0.05, 0.20),
                    "if_blood_surface_wr": rng.uniform(0.05, 0.20),
                    "if_class_level": rng.uniform(1.0, 5.0),
                    "if_distance_bin": rng.uniform(0.0, 3.0),
                    "if_grade_code": rng.uniform(0.0, 5.0),
                    "if_n_horses": 10.0,
                    "if_logit_gap": rng.uniform(-2.0, 2.0),
                    "if_edge_win": rng.uniform(-0.5, 0.5),
                    "if_ev_calibrated": rng.uniform(0.5, 3.0),
                    "if_odds_log": rng.uniform(0.5, 4.0),
                    "if_odds_band_id": rng.uniform(1.0, 7.0),
                    "if_odds_drop_60_10": rng.uniform(-0.1, 0.1),
                    "if_odds_drop_30_10": rng.uniform(-0.1, 0.1),
                    "if_overround": rng.uniform(0.15, 0.30),
                    "if_market_entropy": rng.uniform(0.5, 2.0),
                    "if_conformal_width": rng.uniform(0.1, 2.0),
                    "if_ev_uncertainty_ratio": rng.uniform(0.0, 1.0),
                }
            )
        df = pd.DataFrame(rows)

        result = trained_ranker.score(df)

        expected_cols = [
            "relevance_score",
            "value_score",
            "relevance_score_pct",
            "value_score_pct",
            "calibrated_log_ev_pct",
            "uncertainty_penalty_pct",
            "investment_score",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_investment_score_formula(self, trained_ranker: "RaceLevelRanker") -> None:
        """investment_score fixed-weight formula verification."""
        rng = np.random.RandomState(99)
        rows: list[dict] = []
        for h in range(10):
            rows.append(
                {
                    "race_id": "R_FORMULA_01",
                    "umaban": h + 1,
                    "if_surface": 0.0,
                    "if_p_win_final": rng.uniform(0.02, 0.25),
                    "if_p_win_race_rank": rng.uniform(0.0, 1.0),
                    "if_p_ability_win": rng.uniform(0.02, 0.30),
                    "if_norm_finish_avg": rng.uniform(-1.0, 1.0),
                    "if_closing_index": rng.uniform(0.0, 1.0),
                    "if_weighted_recent_form": rng.uniform(-1.0, 1.0),
                    "if_jockey_wr": rng.uniform(0.05, 0.20),
                    "if_trainer_wr": rng.uniform(0.05, 0.20),
                    "if_blood_surface_wr": rng.uniform(0.05, 0.20),
                    "if_class_level": rng.uniform(1.0, 5.0),
                    "if_distance_bin": rng.uniform(0.0, 3.0),
                    "if_grade_code": rng.uniform(0.0, 5.0),
                    "if_n_horses": 10.0,
                    "if_logit_gap": rng.uniform(-2.0, 2.0),
                    "if_edge_win": rng.uniform(-0.5, 0.5),
                    "if_ev_calibrated": rng.uniform(0.5, 3.0),
                    "if_odds_log": rng.uniform(0.5, 4.0),
                    "if_odds_band_id": rng.uniform(1.0, 7.0),
                    "if_odds_drop_60_10": rng.uniform(-0.1, 0.1),
                    "if_odds_drop_30_10": rng.uniform(-0.1, 0.1),
                    "if_overround": rng.uniform(0.15, 0.30),
                    "if_market_entropy": rng.uniform(0.5, 2.0),
                    "if_conformal_width": rng.uniform(0.1, 2.0),
                    "if_ev_uncertainty_ratio": rng.uniform(0.0, 1.0),
                }
            )
        df = pd.DataFrame(rows)

        result = trained_ranker.score(df)

        expected = (
            0.35 * result["relevance_score_pct"]
            + 0.35 * result["value_score_pct"]
            + 0.20 * result["calibrated_log_ev_pct"]
            - 0.10 * result["uncertainty_penalty_pct"]
        )
        np.testing.assert_allclose(result["investment_score"].values, expected.values, rtol=1e-10)

    def test_score_shadow_mode_guard(self) -> None:
        """score() when _trained=False returns DataFrame unchanged."""
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = rlr.score(df)
        assert list(result.columns) == ["a"]
        assert len(result) == 3


class TestRaceLevelRankerPersistence:
    """save()/load() round-trip preserves all models and state."""

    @pytest.fixture()
    def trained_ranker(self) -> "RaceLevelRanker":
        from models.race_level_ranker import RaceLevelRanker

        df = _make_oof_df()
        rlr = RaceLevelRanker()
        rlr.train(df)
        return rlr

    def test_save_load_roundtrip(self, trained_ranker: "RaceLevelRanker") -> None:
        from models.race_level_ranker import RaceLevelRanker as _RLR

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "race_level_ranker.joblib"
            trained_ranker.save(path)

            loaded = _RLR.load(path)
            assert loaded.is_trained is True
            assert loaded.relevance_scorer_turf is not None
            assert loaded.relevance_scorer_dirt is not None
            assert loaded.value_scorer_turf is not None
            assert loaded.value_scorer_dirt is not None
            assert loaded.training_summary == trained_ranker.training_summary


class TestComputeRelevanceTarget:
    """_compute_relevance_target maps kakuteijyuni to graded relevance."""

    def test_relevance_target_mapping(self) -> None:
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        kakuteijyuni = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = rlr._compute_relevance_target(kakuteijyuni)  # noqa: SLF001

        expected = np.array([1.00, 0.55, 0.30, 0.10, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestComputeValueTarget:
    """_compute_value_target produces composite OOF-safe target."""

    def test_value_target_composite(self) -> None:
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        df = pd.DataFrame(
            {
                "calibrated_ev_oof": [0.5, 1.0, 2.0, 3.0],
                "p_win_oof": [0.05, 0.10, 0.20, 0.30],
                "p_market_norm": [0.10, 0.10, 0.15, 0.25],
                "if_conformal_width": [0.2, 0.5, 1.0, 2.0],
            }
        )
        result = rlr._compute_value_target(df)  # noqa: SLF001

        assert len(result) == 4
        assert result.dtype == np.float64
        # Values should be finite (no inf/nan from log of positive values)
        assert np.all(np.isfinite(result))


class TestRacePctRank:
    """_race_pct_rank uses groupby(race_id).rank(pct=True, method='average')."""

    def test_race_pct_rank(self) -> None:
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        values = pd.Series([0.3, 0.5, 0.1, 0.9, 0.7, 0.2])
        race_id = pd.Series(["R1", "R1", "R1", "R2", "R2", "R2"])
        result = rlr._race_pct_rank(values, race_id)  # noqa: SLF001

        assert len(result) == 6
        assert result.dtype == np.float64
        # All values should be in [0, 1]
        assert (result >= 0.0).all() and (result <= 1.0).all()


class TestD11Diagnostics:
    """train() populates D-11 diagnostics per surface."""

    @pytest.fixture()
    def trained_ranker_with_diagnostics(self) -> "RaceLevelRanker":
        from models.race_level_ranker import RaceLevelRanker

        df = _make_oof_df()
        rlr = RaceLevelRanker()
        rlr.train(df)
        return rlr

    def test_diagnostics_per_surface(
        self,
        trained_ranker_with_diagnostics: "RaceLevelRanker",
    ) -> None:
        summary = trained_ranker_with_diagnostics.training_summary
        for surface in ["turf", "dirt"]:
            key = f"{surface}_diagnostics"
            assert key in summary, f"Missing {key} in training_summary"
            diag = summary[key]
            assert "top1_win_rate" in diag
            assert "ndcg_at_3" in diag
            assert "rank_of_actual_winner" in diag
            assert "top3_contains_winner" in diag
            # top1_win_rate should be between 0 and 1
            assert 0.0 <= diag["top1_win_rate"] <= 1.0
            # ndcg_at_3 should be between 0 and 1
            assert 0.0 <= diag["ndcg_at_3"] <= 1.0
            # rank_of_actual_winner should be >= 1
            assert diag["rank_of_actual_winner"] >= 1.0
            # top3_contains_winner should be between 0 and 1
            assert 0.0 <= diag["top3_contains_winner"] <= 1.0


class TestRaceLevelRankerIsTrainedFix:
    """is_trained returns True for single-surface trained models (bug fix)."""

    def test_is_trained_turf_only(self) -> None:
        """Ranker trained on turf-only data must report is_trained=True."""
        from models.race_level_ranker import RaceLevelRanker

        df_turf = _make_oof_df(surface_value=0)
        rlr = RaceLevelRanker()
        rlr.train(df_turf)
        assert rlr.is_trained is True
        assert rlr.relevance_scorer_turf is not None
        assert rlr.relevance_scorer_dirt is None

    def test_is_trained_dirt_only(self) -> None:
        """Ranker trained on dirt-only data must report is_trained=True."""
        from models.race_level_ranker import RaceLevelRanker

        df_dirt = _make_oof_df(surface_value=1)
        rlr = RaceLevelRanker()
        rlr.train(df_dirt)
        assert rlr.is_trained is True
        assert rlr.relevance_scorer_dirt is not None
        assert rlr.relevance_scorer_turf is None

    def test_is_trained_neither_surface(self) -> None:
        """Fresh ranker with no training must report is_trained=False."""
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        assert rlr.is_trained is False


class TestRaceLevelRankerDeployment:
    """Deployment validation and persistence."""

    def test_deployed_surfaces_default_empty(self) -> None:
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        assert rlr.deployed_surfaces == frozenset()

    def test_is_surface_deployed_false_default(self) -> None:
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        assert rlr.is_surface_deployed("turf") is False
        assert rlr.is_surface_deployed("dirt") is False
        assert rlr.is_surface_deployed(0) is False
        assert rlr.is_surface_deployed(1) is False

    def test_is_surface_deployed_numeric_and_string(self) -> None:
        """is_surface_deployed handles both numeric and string surface values."""
        from models.race_level_ranker import RaceLevelRanker

        rlr = RaceLevelRanker()
        rlr._deployed_surfaces = {"turf"}  # noqa: SLF001
        assert rlr.is_surface_deployed("turf") is True
        assert rlr.is_surface_deployed(0) is True
        assert rlr.is_surface_deployed("dirt") is False

    def test_validate_oof_insufficient_data_skips(self) -> None:
        """OOF validation skips when data has < 100 rows."""
        from models.race_level_ranker import RaceLevelRanker

        df_small = _make_oof_df(n_races=3, n_horses_per_race=5)
        rlr = RaceLevelRanker()
        rlr.train(df_small)
        rlr.validate_oof_deployment(df_small)
        assert rlr.deployed_surfaces == frozenset()
        assert "turf_oof_validation" not in rlr.training_summary

    def test_validate_oof_missing_baseline_skips(self) -> None:
        """OOF validation skips when baseline column is missing."""
        from models.race_level_ranker import RaceLevelRanker

        df = _make_oof_df(n_races=40, n_horses_per_race=8)
        rlr = RaceLevelRanker()
        rlr.validate_oof_deployment(df)
        # No win_market_selection_score → should skip
        assert rlr.deployed_surfaces == frozenset()

    def test_validate_oof_records_metrics(self) -> None:
        """OOF validation records per-surface metrics in training_summary."""
        from models.race_level_ranker import RaceLevelRanker

        df = _make_oof_df(n_races=40, n_horses_per_race=8)
        # Simulate operational baseline score (normally from WinSelectionPolicy)
        df["win_market_selection_score"] = df["p_win_oof"]
        df["tanodds"] = 1.0 / df["p_win_oof"].clip(lower=0.01)  # simulate odds
        rlr = RaceLevelRanker()
        rlr.train(df)
        rlr.validate_oof_deployment(df)
        for surface in ["turf", "dirt"]:
            key = f"{surface}_oof_validation"
            assert key in rlr.training_summary, f"Missing {key}"
            metrics = rlr.training_summary[key]
            assert "ranker_top1_hit_rate" in metrics
            assert "baseline_top1_hit_rate" in metrics
            assert "hit_rate_improvement" in metrics
            assert "ranker_top1_roi" in metrics
            assert "baseline_top1_roi" in metrics
            assert "roi_improvement" in metrics
            assert "ranker_ndcg_at_3" in metrics
            assert "baseline_ndcg_at_3" in metrics
            assert "ndcg_improvement" in metrics
            assert "n_folds_evaluated" in metrics
            assert "total_races" in metrics
            assert metrics["total_races"] >= 10
            assert metrics["n_folds_evaluated"] >= 1

    def test_validate_uses_custom_baseline_col(self) -> None:
        """validate_oof_deployment accepts custom baseline_col parameter."""
        from models.race_level_ranker import RaceLevelRanker

        df = _make_oof_df(n_races=40, n_horses_per_race=8)
        df["tanodds"] = 1.0 / df["p_win_oof"].clip(lower=0.01)
        rlr = RaceLevelRanker()
        rlr.train(df)
        # Use p_win_oof as baseline (backward-compatible path)
        rlr.validate_oof_deployment(df, baseline_col="p_win_oof")
        assert "turf_oof_validation" in rlr.training_summary

    def test_save_load_preserves_deployed_surfaces(self) -> None:
        """save/load round-trip preserves _deployed_surfaces."""
        from models.race_level_ranker import RaceLevelRanker

        df = _make_oof_df()
        rlr = RaceLevelRanker()
        rlr.train(df)
        rlr._deployed_surfaces = {"turf"}  # noqa: SLF001

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ranker.joblib"
            rlr.save(path)
            loaded = RaceLevelRanker.load(path)
            assert loaded.is_trained is True
            assert loaded.deployed_surfaces == frozenset({"turf"})
            assert loaded.is_surface_deployed("turf") is True
            assert loaded.is_surface_deployed("dirt") is False

    def test_save_load_backward_compatible_no_deploy_key(self) -> None:
        """Load old joblib without _deployed_surfaces key defaults to empty."""
        import joblib

        from models.race_level_ranker import RaceLevelRanker

        df = _make_oof_df()
        rlr = RaceLevelRanker()
        rlr.train(df)
        state = {
            "relevance_scorer_turf": rlr.relevance_scorer_turf,
            "relevance_scorer_dirt": rlr.relevance_scorer_dirt,
            "value_scorer_turf": rlr.value_scorer_turf,
            "value_scorer_dirt": rlr.value_scorer_dirt,
            "relevance_feature_names": rlr.relevance_feature_names,
            "value_feature_names": rlr.value_feature_names,
            "training_summary": rlr.training_summary,
            "_trained": True,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old_ranker.joblib"
            joblib.dump(state, path)
            loaded = RaceLevelRanker.load(path)
            assert loaded.is_trained is True
            assert loaded.deployed_surfaces == frozenset()


class TestResolvePayoutOdds:
    """_resolve_payout_odds: confirmed_odds preferred, tanodds fallback."""

    def test_confirmed_odds_priority(self) -> None:
        """confirmed_odds=10, tanodds=5 → returns 10."""
        from models.race_level_ranker import _resolve_payout_odds

        row = pd.Series({"confirmed_odds": 10.0, "tanodds": 5.0})
        assert _resolve_payout_odds(row) == 10.0

    def test_tanodds_fallback(self) -> None:
        """No confirmed_odds → returns tanodds."""
        from models.race_level_ranker import _resolve_payout_odds

        row = pd.Series({"tanodds": 5.0})
        assert _resolve_payout_odds(row) == 5.0

    def test_both_missing_returns_nan(self) -> None:
        """Neither column → returns NaN."""
        from models.race_level_ranker import _resolve_payout_odds

        row = pd.Series({"confirmed_odds": pd.NA, "tanodds": pd.NA})
        assert pd.isna(_resolve_payout_odds(row))

    def test_confirmed_nan_uses_tanodds(self) -> None:
        """confirmed_odds=NaN → falls back to tanodds."""
        from models.race_level_ranker import _resolve_payout_odds

        row = pd.Series({"confirmed_odds": np.nan, "tanodds": 5.0})
        assert _resolve_payout_odds(row) == 5.0


class TestValidateOofDeploymentPayout:
    """validate_oof_deployment uses confirmed_odds for ROI via _resolve_payout_odds."""

    def test_confirmed_odds_preferred_in_training_summary(self) -> None:
        """confirmed_odds=10, tanodds=5 → baseline_top1_roi reflects confirmed_odds."""
        from models.race_level_ranker import RaceLevelRanker

        df = _make_oof_df(n_races=40, n_horses_per_race=8, surface_value=0)
        # Horse 1 (umaban=1) always wins: kakuteijyuni reassigned 1..8
        # Set baseline so horse 1 is always top1
        df["win_market_selection_score"] = 0.0
        df.loc[df["umaban"] == 1, "win_market_selection_score"] = 100.0
        # tanodds=5, confirmed_odds=10 for horse 1 (the winner baseline picks)
        df["tanodds"] = 10.0
        df["confirmed_odds"] = 20.0
        df.loc[df["umaban"] == 1, "tanodds"] = 5.0
        df.loc[df["umaban"] == 1, "confirmed_odds"] = 10.0

        rlr = RaceLevelRanker()
        rlr.train(df)
        rlr.validate_oof_deployment(df)

        metrics = rlr.training_summary["turf_oof_validation"]
        # Baseline always picks horse 1 (winner), confirmed_odds=10
        assert metrics["baseline_top1_roi"] == pytest.approx(10.0, abs=0.5)

    def test_tanodds_used_when_no_confirmed_odds(self) -> None:
        """Without confirmed_odds column, tanodds is used for ROI."""
        from models.race_level_ranker import RaceLevelRanker

        df = _make_oof_df(n_races=40, n_horses_per_race=8, surface_value=0)
        df["win_market_selection_score"] = 0.0
        df.loc[df["umaban"] == 1, "win_market_selection_score"] = 100.0
        df["tanodds"] = 10.0
        df.loc[df["umaban"] == 1, "tanodds"] = 7.0
        # No confirmed_odds column at all

        rlr = RaceLevelRanker()
        rlr.train(df)
        rlr.validate_oof_deployment(df)

        metrics = rlr.training_summary["turf_oof_validation"]
        assert metrics["baseline_top1_roi"] == pytest.approx(7.0, abs=0.5)
