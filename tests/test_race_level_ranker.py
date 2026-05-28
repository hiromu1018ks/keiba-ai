"""Tests for RaceLevelRanker -- learned ranker with Ridge models.

TDD RED phase: all tests MUST fail before implementation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge


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
        """Create a synthetic OOF DataFrame for training."""
        rng = np.random.RandomState(42)
        n_races = 30
        n_horses_per_race = 10
        rows: list[dict] = []
        for r in range(n_races):
            race_id = f"R{r:04d}"
            surface = 0 if r % 2 == 0 else 1  # 0=turf, 1=dirt
            for h in range(n_horses_per_race):
                umaban = h + 1
                kakuteijyuni = h + 1  # deterministic finishing position
                # Shuffle to make it non-trivial
                if h == 0:
                    kakuteijyuni = 1
                else:
                    kakuteijyuni = rng.randint(2, n_horses_per_race + 1)

                p_win = rng.uniform(0.02, 0.25)
                p_ability_win = rng.uniform(0.02, 0.30)
                p_market = 1.0 / rng.uniform(2.0, 30.0)
                ev = p_win * rng.uniform(2.0, 30.0)

                rows.append({
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
                })
        df = pd.DataFrame(rows)
        # Ensure unique kakuteijyuni per race
        for race_id in df["race_id"].unique():
            mask = df["race_id"] == race_id
            n = mask.sum()
            df.loc[mask, "kakuteijyuni"] = np.arange(1, n + 1)
        return df

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


class TestRaceLevelRankerScore:
    """score() produces investment_score columns."""

    @pytest.fixture()
    def trained_ranker(self) -> "RaceLevelRanker":
        from models.race_level_ranker import RaceLevelRanker

        rng = np.random.RandomState(42)
        n_races = 30
        n_horses_per_race = 10
        rows: list[dict] = []
        for r in range(n_races):
            race_id = f"R{r:04d}"
            surface = 0 if r % 2 == 0 else 1
            for h in range(n_horses_per_race):
                umaban = h + 1
                kakuteijyuni = h + 1
                p_win = rng.uniform(0.02, 0.25)
                p_ability_win = rng.uniform(0.02, 0.30)
                p_market = 1.0 / rng.uniform(2.0, 30.0)
                ev = p_win * rng.uniform(2.0, 30.0)

                rows.append({
                    "race_id": race_id,
                    "umaban": umaban,
                    "race_date": f"2023-{(r // 5) + 1:02d}-01",
                    "surface": surface,
                    "kakuteijyuni": kakuteijyuni,
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
                    "calibrated_ev_oof": ev,
                    "p_win_oof": p_win,
                    "p_market_norm": p_market,
                })
        df = pd.DataFrame(rows)
        for race_id in df["race_id"].unique():
            mask = df["race_id"] == race_id
            n = mask.sum()
            df.loc[mask, "kakuteijyuni"] = np.arange(1, n + 1)

        rlr = RaceLevelRanker()
        rlr.train(df)
        return rlr

    def test_score_adds_investment_columns(
        self, trained_ranker: "RaceLevelRanker",
    ) -> None:
        """score() adds all required investment score columns."""
        rng = np.random.RandomState(99)
        rows: list[dict] = []
        for h in range(10):
            rows.append({
                "race_id": "R_SCORE_01",
                "umaban": h + 1,
                "surface": 0.0,
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
            })
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
        """investment_score = 0.35*rel_pct + 0.35*val_pct + 0.20*log_ev_pct - 0.10*uncertainty_pct."""
        rng = np.random.RandomState(99)
        rows: list[dict] = []
        for h in range(10):
            rows.append({
                "race_id": "R_FORMULA_01",
                "umaban": h + 1,
                "surface": 0.0,
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
            })
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

        rng = np.random.RandomState(42)
        n_races = 30
        n_horses_per_race = 10
        rows: list[dict] = []
        for r in range(n_races):
            race_id = f"R{r:04d}"
            surface = 0 if r % 2 == 0 else 1
            for h in range(n_horses_per_race):
                umaban = h + 1
                kakuteijyuni = h + 1
                p_win = rng.uniform(0.02, 0.25)
                p_ability_win = rng.uniform(0.02, 0.30)
                p_market = 1.0 / rng.uniform(2.0, 30.0)
                ev = p_win * rng.uniform(2.0, 30.0)

                rows.append({
                    "race_id": race_id,
                    "umaban": umaban,
                    "race_date": f"2023-{(r // 5) + 1:02d}-01",
                    "surface": surface,
                    "kakuteijyuni": kakuteijyuni,
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
                    "calibrated_ev_oof": ev,
                    "p_win_oof": p_win,
                    "p_market_norm": p_market,
                })
        df = pd.DataFrame(rows)
        for race_id in df["race_id"].unique():
            mask = df["race_id"] == race_id
            n = mask.sum()
            df.loc[mask, "kakuteijyuni"] = np.arange(1, n + 1)

        rlr = RaceLevelRanker()
        rlr.train(df)
        return rlr

    def test_save_load_roundtrip(self, trained_ranker: "RaceLevelRanker") -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "race_level_ranker.joblib"
            trained_ranker.save(path)

            loaded = RaceLevelRanker.load(path)
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
        df = pd.DataFrame({
            "calibrated_ev_oof": [0.5, 1.0, 2.0, 3.0],
            "p_win_oof": [0.05, 0.10, 0.20, 0.30],
            "p_market_norm": [0.10, 0.10, 0.15, 0.25],
            "if_conformal_width": [0.2, 0.5, 1.0, 2.0],
        })
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

        rng = np.random.RandomState(42)
        n_races = 30
        n_horses_per_race = 10
        rows: list[dict] = []
        for r in range(n_races):
            race_id = f"R{r:04d}"
            surface = 0 if r % 2 == 0 else 1
            for h in range(n_horses_per_race):
                umaban = h + 1
                kakuteijyuni = h + 1
                p_win = rng.uniform(0.02, 0.25)
                p_ability_win = rng.uniform(0.02, 0.30)
                p_market = 1.0 / rng.uniform(2.0, 30.0)
                ev = p_win * rng.uniform(2.0, 30.0)

                rows.append({
                    "race_id": race_id,
                    "umaban": umaban,
                    "race_date": f"2023-{(r // 5) + 1:02d}-01",
                    "surface": surface,
                    "kakuteijyuni": kakuteijyuni,
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
                    "calibrated_ev_oof": ev,
                    "p_win_oof": p_win,
                    "p_market_norm": p_market,
                })
        df = pd.DataFrame(rows)
        for race_id in df["race_id"].unique():
            mask = df["race_id"] == race_id
            n = mask.sum()
            df.loc[mask, "kakuteijyuni"] = np.arange(1, n + 1)

        rlr = RaceLevelRanker()
        rlr.train(df)
        return rlr

    def test_diagnostics_per_surface(self, trained_ranker_with_diagnostics: "RaceLevelRanker") -> None:
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
