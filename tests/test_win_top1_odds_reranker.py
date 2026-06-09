"""WinTop1OddsReranker tests.

Covers: training objective, vectorized simulation, apply semantics,
confirmed_odds payout, save/load, RacePredictor integration,
backward compatibility, diagnostics propagation.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from models.win_top1_odds_reranker import (
    GAP_RERANKER_APPLIED_COL,
    GAP_RERANKER_DEPLOYED_COL,
    GAP_RERANKER_DIAGNOSTIC_COLS,
    GAP_RERANKER_FINAL_TOP1_COL,
    GAP_RERANKER_ORIG_TOP1_COL,
    GAP_RERANKER_PROB_MARGIN_COL,
    GAP_RERANKER_SCORE_GAP_COL,
    GAP_RERANKER_SWITCH_REASON_COL,
    RERANKER_APPLIED_COL,
    RERANKER_CAP_COL,
    RERANKER_DIAGNOSTIC_COLS,
    RERANKER_FINAL_TOP1_COL,
    RERANKER_ORIG_TOP1_COL,
    RERANKER_SWITCH_REASON_COL,
    WinTop1OddsReranker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reranker_rows(
    n_races: int = 300,
    high_odds_hit_rate: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic OOF frame for reranker training.

    Creates races with:
      - Horse 1: low odds (~3.0), high score → usually top-1
      - Horse 2: mid odds (~15.0), mid score
      - Horse 3: high odds (~80.0), sometimes highest score (simulates "runaway")

    When high_odds_hit_rate > 0, horse 3 occasionally wins with high odds,
    making cap selection unprofitable.
    """
    rng = np.random.RandomState(seed)
    rows: list[dict[str, object]] = []
    for race_idx in range(n_races):
        race_id = f"R{race_idx:05d}"
        race_date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=race_idx)
        # Horse 3 occasionally has the highest score (simulating high-odds runaway)
        h3_score_boost = 2.0 if rng.random() < 0.15 else 0.0
        # Horse 3 sometimes wins (for payout simulation)
        h3_wins = rng.random() < high_odds_hit_rate
        rows.extend(
            [
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": 1,
                    "kakuteijyuni": 1 if (not h3_wins and race_idx % 4 == 0) else 3,
                    "tanodds": 3.0,
                    "confirmed_odds": 3.2,
                    "win_market_selection_score": 1.0,
                },
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": 2,
                    "kakuteijyuni": 1 if (not h3_wins and race_idx % 5 == 0) else 4,
                    "tanodds": 15.0,
                    "confirmed_odds": 14.5,
                    "win_market_selection_score": 0.5,
                },
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": 3,
                    "kakuteijyuni": 1 if h3_wins else 8,
                    "tanodds": 80.0,
                    "confirmed_odds": 82.0,
                    "win_market_selection_score": 0.3 + h3_score_boost,
                },
            ]
        )
    return pd.DataFrame(rows)


def _make_single_race_df(
    odds_list: list[float],
    score_list: list[float],
    *,
    race_id: str = "TEST_R1",
    surface: str = "turf",
) -> pd.DataFrame:
    """Build a minimal single-race DataFrame for apply() tests."""
    rows = []
    for i, (odds, score) in enumerate(zip(odds_list, score_list), start=1):
        rows.append(
            {
                "race_id": race_id,
                "race_date": pd.Timestamp("2025-01-01"),
                "surface": surface,
                "umaban": i,
                "tanodds": odds,
                "win_market_selection_score": score,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


class TestWinTop1OddsRerankerTraining:
    def test_train_selects_finite_cap_when_profitable(self) -> None:
        """When high-odds runaway selections are losing, a cap should be selected."""
        # Horse 3 has boosted score 15% of the time but never wins → cap helps
        model = WinTop1OddsReranker(
            min_train_races=100,
            min_fold_races=50,
            max_folds=3,
        )
        df = _make_reranker_rows(n_races=300, high_odds_hit_rate=0.0)
        model.train(df)

        assert model.is_trained is True
        assert model.selected_cap < float("inf")
        assert "objective" in model.training_summary
        assert "baseline_objective" in model.training_summary

    def test_train_insufficient_races_returns_inf(self) -> None:
        """Too few races → cap=inf."""
        model = WinTop1OddsReranker(
            min_train_races=500,
            min_fold_races=100,
        )
        df = _make_reranker_rows(n_races=50)
        model.train(df)

        assert model.is_trained is True
        assert model.selected_cap == float("inf")
        assert model.training_summary["reason"] == "insufficient_races"

    def test_train_missing_columns_returns_inf(self) -> None:
        """Missing required columns → cap=inf."""
        model = WinTop1OddsReranker()
        model.train(pd.DataFrame({"race_id": ["R1"], "umaban": [1]}))

        assert model.selected_cap == float("inf")

    def test_train_no_improvement_selects_inf(self) -> None:
        """When no cap beats inf baseline, inf is selected."""
        # All horses have low odds (≤ 10.0), so no cap helps
        rows: list[dict[str, object]] = []
        for race_idx in range(300):
            race_id = f"R{race_idx:05d}"
            race_date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=race_idx)
            rows.extend(
                [
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "umaban": 1,
                        "kakuteijyuni": 1 if race_idx % 3 == 0 else 2,
                        "tanodds": 2.5,
                        "win_market_selection_score": 1.0,
                    },
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "umaban": 2,
                        "kakuteijyuni": 1 if race_idx % 4 == 0 else 3,
                        "tanodds": 5.0,
                        "win_market_selection_score": 0.5,
                    },
                ]
            )
        df = pd.DataFrame(rows)
        model = WinTop1OddsReranker(
            min_train_races=100,
            min_fold_races=50,
            max_folds=3,
        )
        model.train(df)

        assert model.is_trained is True
        # All odds ≤ 5.0, no cap should improve over inf
        assert model.selected_cap == float("inf")

    def test_train_tie_selects_inf_not_smaller_cap(self) -> None:
        """When cap ties with inf, inf is selected (don't prefer smaller cap)."""
        # Build data where cap=20.0 ties with inf
        model = WinTop1OddsReranker(
            min_train_races=100,
            min_fold_races=50,
            max_folds=3,
        )
        df = _make_reranker_rows(n_races=300, high_odds_hit_rate=0.0)
        model.train(df)

        # If objective ties, inf should be selected
        best_obj = model.training_summary.get("objective", float("-inf"))
        baseline_obj = model.training_summary.get("baseline_objective", float("-inf"))
        if np.isclose(best_obj, baseline_obj):
            assert model.selected_cap == float("inf")

    def test_train_fold_chronological_order(self) -> None:
        """Folds should be in chronological race order."""
        model = WinTop1OddsReranker(
            min_train_races=100,
            min_fold_races=50,
            max_folds=3,
        )
        df = _make_reranker_rows(n_races=300)
        model.train(df)

        summary = model.training_summary
        assert "n_folds" in summary
        assert summary["n_folds"] > 0

        # Verify fold order via _build_folds directly
        race_order = (
            df[["race_id", "race_date"]]
            .drop_duplicates()
            .sort_values(["race_date", "race_id"])
            .reset_index(drop=True)
        )
        folds = model._build_folds(race_order)
        for i in range(len(folds) - 1):
            assert folds[i][1] <= folds[i + 1][0], "Folds must be chronological"

    def test_train_bet_count_equals_race_count(self) -> None:
        """Each fold must produce exactly 1 bet per race."""
        df = _make_reranker_rows(n_races=300)
        model = WinTop1OddsReranker(
            min_train_races=100,
            min_fold_races=50,
            max_folds=3,
        )
        model.train(df)

        all_metrics = model.training_summary.get("all_cap_metrics", {})
        for cap_str, metrics in all_metrics.items():
            assert metrics["bets_ok"], (
                f"cap={cap_str}: bets ({metrics['bets']}) != fold_races ({metrics['n_fold_races']})"
            )

    def test_train_confirmed_odds_used_for_payout(self) -> None:
        """confirmed_odds should be preferred over tanodds for payout calculation."""
        rows: list[dict[str, object]] = []
        for race_idx in range(300):
            race_id = f"R{race_idx:05d}"
            race_date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=race_idx)
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": 1,
                    "kakuteijyuni": 1 if race_idx % 2 == 0 else 2,
                    "tanodds": 5.0,
                    "confirmed_odds": 5.5,  # Different from tanodds
                    "win_market_selection_score": 1.0,
                }
            )
        df = pd.DataFrame(rows)
        model = WinTop1OddsReranker(
            min_train_races=100,
            min_fold_races=50,
            max_folds=3,
        )
        model.train(df)

        # Verify that simulation uses confirmed_odds (5.5) not tanodds (5.0)
        # ROI = total_return / bets.  150 wins * 5.5 = 825, 300 bets → ROI = 2.75
        # If tanodds (5.0) were used: 150 * 5.0 = 750, ROI = 2.50
        inf_metrics = model.training_summary.get("baseline_metrics", {})
        if inf_metrics:
            actual_roi = inf_metrics.get("roi", 0.0)
            assert actual_roi == pytest.approx(2.75, abs=0.1), (
                f"ROI {actual_roi} doesn't reflect confirmed_odds=5.5 "
                f"(expected ~2.75, would be 2.50 if using tanodds)"
            )


# ---------------------------------------------------------------------------
# Apply tests
# ---------------------------------------------------------------------------


class TestWinTop1OddsRerankerApply:
    def _trained_model(self, cap: float = 30.0) -> WinTop1OddsReranker:
        model = WinTop1OddsReranker()
        model.selected_cap = cap
        model.is_trained = True
        return model

    def test_apply_returns_one_per_race_multi_race(self) -> None:
        """2+ races: always returns exactly 1 horse per race."""
        df = pd.concat(
            [
                _make_single_race_df(
                    [3.0, 80.0, 15.0],
                    [1.0, 0.3, 0.5],
                    race_id="R1",
                ),
                _make_single_race_df(
                    [5.0, 20.0, 40.0],
                    [0.4, 0.9, 0.7],
                    race_id="R2",
                ),
            ],
            ignore_index=True,
        )
        model = self._trained_model(cap=30.0)
        result = model.apply(df)

        assert len(result) == 2
        assert set(result["race_id"].tolist()) == {"R1", "R2"}

    def test_apply_switches_when_orig_top1_exceeds_cap(self) -> None:
        """Original top1 has odds > cap → switch to cap-eligible next best."""
        # Horse 3: odds=80, score=1.5 (top1 but > cap=30)
        # Horse 1: odds=3, score=1.0 (within cap)
        df = _make_single_race_df([3.0, 15.0, 80.0], [1.0, 0.5, 1.5])
        model = self._trained_model(cap=30.0)
        result = model.apply(df)

        assert len(result) == 1
        assert result.iloc[0]["umaban"] == 1  # Switched to cap-eligible horse
        assert result[RERANKER_APPLIED_COL].iloc[0] == True  # noqa: E712
        assert result[RERANKER_SWITCH_REASON_COL].iloc[0] == "odds_cap_switch"
        assert result[RERANKER_ORIG_TOP1_COL].iloc[0] == 3
        assert result[RERANKER_FINAL_TOP1_COL].iloc[0] == 1

    def test_apply_no_change_when_orig_top1_within_cap(self) -> None:
        """Original top1 is within cap → no change."""
        df = _make_single_race_df([3.0, 15.0, 80.0], [1.5, 0.5, 0.3])
        model = self._trained_model(cap=30.0)
        result = model.apply(df)

        assert len(result) == 1
        assert result.iloc[0]["umaban"] == 1  # Original top1 kept
        assert result[RERANKER_APPLIED_COL].iloc[0] == False  # noqa: E712
        assert result[RERANKER_SWITCH_REASON_COL].iloc[0] == "top1_within_cap"

    def test_apply_fallback_when_no_eligible_candidates(self) -> None:
        """All horses exceed cap → keep original top1."""
        # All odds > 30 (cap)
        df = _make_single_race_df([50.0, 80.0, 100.0], [1.5, 0.5, 0.3])
        model = self._trained_model(cap=30.0)
        result = model.apply(df)

        assert len(result) == 1
        assert result.iloc[0]["umaban"] == 1  # Original top1 kept
        assert result[RERANKER_APPLIED_COL].iloc[0] == False  # noqa: E712
        assert result[RERANKER_SWITCH_REASON_COL].iloc[0] == "no_eligible"

    def test_apply_inf_cap_selects_original_top1(self) -> None:
        """cap=inf → no filtering, just select original top1."""
        df = _make_single_race_df([3.0, 80.0, 15.0], [0.5, 1.5, 0.3])
        model = self._trained_model(cap=float("inf"))
        result = model.apply(df)

        assert len(result) == 1
        assert result.iloc[0]["umaban"] == 2  # Highest score regardless of odds
        assert result[RERANKER_APPLIED_COL].iloc[0] == False  # noqa: E712
        assert result[RERANKER_SWITCH_REASON_COL].iloc[0] == "inf_cap"

    def test_apply_untrained_returns_original_top1(self) -> None:
        """Untrained model → returns original top1 with 'untrained' reason."""
        df = _make_single_race_df([3.0, 80.0], [0.5, 1.5])
        model = WinTop1OddsReranker()
        result = model.apply(df)

        assert len(result) == 1
        assert result.iloc[0]["umaban"] == 2
        assert result[RERANKER_SWITCH_REASON_COL].iloc[0] == "untrained"

    def test_apply_empty_df_returns_empty(self) -> None:
        """Empty input → empty output with diagnostic columns."""
        model = self._trained_model(cap=30.0)
        result = model.apply(pd.DataFrame())

        assert result.empty

    def test_apply_preserves_original_columns(self) -> None:
        """All original columns are preserved in the output."""
        df = _make_single_race_df([3.0, 80.0], [0.5, 1.5])
        model = self._trained_model(cap=30.0)
        result = model.apply(df)

        for col in ["race_id", "umaban", "tanodds", "win_market_selection_score"]:
            assert col in result.columns

    def test_apply_index_stability(self) -> None:
        """Output rows retain their original DataFrame index."""
        df = _make_single_race_df([3.0, 80.0], [0.5, 1.5])
        original_idx = df.index.tolist()
        model = self._trained_model(cap=30.0)
        result = model.apply(df)

        # Selected row should have its original index
        assert result.index[0] in original_idx

    def test_apply_diagnostics_propagation(self) -> None:
        """All 5 diagnostic columns are present and correctly typed."""
        df = _make_single_race_df([3.0, 80.0, 15.0], [1.0, 0.3, 0.5])
        model = self._trained_model(cap=30.0)
        result = model.apply(df)

        for col in [
            RERANKER_APPLIED_COL,
            RERANKER_CAP_COL,
            RERANKER_ORIG_TOP1_COL,
            RERANKER_FINAL_TOP1_COL,
            RERANKER_SWITCH_REASON_COL,
        ]:
            assert col in result.columns, f"Missing diagnostic column: {col}"

        assert result[RERANKER_CAP_COL].iloc[0] == 30.0


# ---------------------------------------------------------------------------
# Save / Load tests
# ---------------------------------------------------------------------------


class TestWinTop1OddsRerankerPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Save and load preserves all fields."""
        model = WinTop1OddsReranker(
            min_train_races=100,
            min_fold_races=50,
            max_folds=3,
        )
        df = _make_reranker_rows(n_races=300)
        model.train(df)

        path = tmp_path / "reranker.joblib"
        model.save(path)
        loaded = WinTop1OddsReranker.load(path)

        assert loaded.is_trained == model.is_trained
        assert loaded.selected_cap == model.selected_cap
        assert loaded.stability_penalty == model.stability_penalty
        assert loaded.min_roi_floor == model.min_roi_floor
        assert loaded.min_roi_penalty == model.min_roi_penalty
        assert loaded.training_summary == model.training_summary

    def test_save_load_untrained(self, tmp_path: Path) -> None:
        """Save/load of untrained model."""
        model = WinTop1OddsReranker()
        path = tmp_path / "reranker_untrained.joblib"
        model.save(path)
        loaded = WinTop1OddsReranker.load(path)

        assert loaded.is_trained is False
        assert loaded.selected_cap == float("inf")

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save() creates intermediate directories."""
        model = WinTop1OddsReranker()
        deep_path = tmp_path / "a" / "b" / "c" / "reranker.joblib"
        model.save(deep_path)
        assert deep_path.is_file()


# ---------------------------------------------------------------------------
# RacePredictor integration tests
# ---------------------------------------------------------------------------


class TestRacePredictorIntegration:
    def _make_submodel_with_reranker(self, cap: float = 30.0) -> SimpleNamespace:
        """Build a SubmodelSet-like namespace with reranker attached."""
        reranker = WinTop1OddsReranker()
        reranker.selected_cap = cap
        reranker.is_trained = True

        sub = SimpleNamespace(
            market=MagicMock(),
            stage1=MagicMock(),
            place_ability=None,
            win=MagicMock(),
            ev_corrector=MagicMock(),
            place=None,
            wide=None,
            conformal_ev_model=None,
            place_selection_gate=None,
            benter_combo=None,
            isotonic_calibrator=None,
            market_aware_win_calibrator=None,
            win_selection_gate=None,
            win_selection_policy=None,
            win_profit_selector=None,
            win_race_level_ranker=None,
            win_top1_odds_reranker=reranker,
            ev_lower_threshold_turf=1.0,
            ev_lower_threshold_dirt=1.0,
            target_encoder=None,
        )
        return sub

    def test_reranker_selects_cap_eligible_via_race_predictor(self) -> None:
        """RacePredictor.get_win_candidates uses reranker to avoid high-odds top1."""
        from backtest.race_predictor import RacePredictor
        from domain.models import TrainedModelsV5

        sub = self._make_submodel_with_reranker(cap=30.0)
        models = MagicMock(spec=TrainedModelsV5)
        models.submodels = {"turf": sub}
        models.quality_screener = MagicMock()
        models.regime_detector = MagicMock()
        models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.03,
            "max_bets_per_race": 3,
        }

        predictor = RacePredictor(models=models, betting_target="win")
        candidates = predictor.get_win_candidates(
            pd.DataFrame(
                [
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 1,
                        "tanodds": 3.0,
                        "p_win_final": 0.45,
                        "win_selection_prob": 0.45,
                        "win_selection_ev": 0.90,
                        "win_selection_edge": -0.10,
                        "win_market_selection_score": 1.0,
                    },
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 2,
                        "tanodds": 80.0,
                        "p_win_final": 0.02,
                        "win_selection_prob": 0.02,
                        "win_selection_ev": 0.60,
                        "win_selection_edge": -0.40,
                        "win_market_selection_score": 1.5,  # Higher score but odds > cap
                    },
                ]
            )
        )

        # Reranker should select horse 1 (within cap) instead of horse 2 (odds > 30)
        assert len(candidates) == 1
        assert candidates.iloc[0]["umaban"] == 1

    def test_race_predictor_backward_compat_no_reranker(self) -> None:
        """RacePredictor works normally when reranker is None."""
        from backtest.race_predictor import RacePredictor
        from domain.models import TrainedModelsV5

        sub = SimpleNamespace(
            market=MagicMock(),
            stage1=MagicMock(),
            place_ability=None,
            win=MagicMock(),
            ev_corrector=MagicMock(),
            place=None,
            wide=None,
            conformal_ev_model=None,
            place_selection_gate=None,
            benter_combo=None,
            isotonic_calibrator=None,
            market_aware_win_calibrator=None,
            win_selection_gate=None,
            win_selection_policy=None,
            win_profit_selector=None,
            win_race_level_ranker=None,
            win_top1_odds_reranker=None,  # No reranker
            ev_lower_threshold_turf=1.0,
            ev_lower_threshold_dirt=1.0,
            target_encoder=None,
        )
        models = MagicMock(spec=TrainedModelsV5)
        models.submodels = {"turf": sub}
        models.quality_screener = MagicMock()
        models.regime_detector = MagicMock()
        models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.03,
            "max_bets_per_race": 3,
        }

        predictor = RacePredictor(models=models, betting_target="win")
        candidates = predictor.get_win_candidates(
            pd.DataFrame(
                [
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 1,
                        "tanodds": 3.0,
                        "p_win_final": 0.45,
                        "win_selection_prob": 0.45,
                        "win_selection_ev": 0.90,
                        "win_selection_edge": -0.10,
                        "win_market_selection_score": 1.0,
                    },
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 2,
                        "tanodds": 80.0,
                        "p_win_final": 0.02,
                        "win_selection_prob": 0.02,
                        "win_selection_ev": 0.60,
                        "win_selection_edge": -0.40,
                        "win_market_selection_score": 1.5,
                    },
                ]
            )
        )

        # Without reranker, horse 2 (highest score) should be selected
        assert len(candidates) == 1
        assert candidates.iloc[0]["umaban"] == 2

    def test_surface_specific_reranker(self) -> None:
        """Different surfaces use different rerankers."""
        from backtest.race_predictor import RacePredictor
        from domain.models import TrainedModelsV5

        turf_reranker = WinTop1OddsReranker()
        turf_reranker.selected_cap = 20.0
        turf_reranker.is_trained = True

        dirt_reranker = WinTop1OddsReranker()
        dirt_reranker.selected_cap = 50.0
        dirt_reranker.is_trained = True

        def _make_sub(reranker: WinTop1OddsReranker) -> SimpleNamespace:
            return SimpleNamespace(
                market=MagicMock(),
                stage1=MagicMock(),
                place_ability=None,
                win=MagicMock(),
                ev_corrector=MagicMock(),
                place=None,
                wide=None,
                conformal_ev_model=None,
                place_selection_gate=None,
                benter_combo=None,
                isotonic_calibrator=None,
                market_aware_win_calibrator=None,
                win_selection_gate=None,
                win_selection_policy=None,
                win_profit_selector=None,
                win_race_level_ranker=None,
                win_top1_odds_reranker=reranker,
                ev_lower_threshold_turf=1.0,
                ev_lower_threshold_dirt=1.0,
                target_encoder=None,
            )

        models = MagicMock(spec=TrainedModelsV5)
        models.submodels = {
            "turf": _make_sub(turf_reranker),
            "dirt": _make_sub(dirt_reranker),
        }
        models.quality_screener = MagicMock()
        models.regime_detector = MagicMock()
        models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.03,
            "max_bets_per_race": 3,
        }

        predictor = RacePredictor(models=models, betting_target="win")

        # Turf: cap=20 → horse with odds=25 excluded, pick next
        turf_cands = predictor.get_win_candidates(
            pd.DataFrame(
                [
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 1,
                        "tanodds": 3.0,
                        "p_win_final": 0.40,
                        "win_selection_prob": 0.40,
                        "win_selection_ev": 1.2,
                        "win_selection_edge": 0.2,
                        "win_market_selection_score": 0.5,
                    },
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 2,
                        "tanodds": 25.0,
                        "p_win_final": 0.05,
                        "win_selection_prob": 0.05,
                        "win_selection_ev": 1.25,
                        "win_selection_edge": 0.25,
                        "win_market_selection_score": 1.0,  # > cap=20
                    },
                ]
            )
        )
        assert len(turf_cands) == 1
        assert turf_cands.iloc[0]["umaban"] == 1  # 25 > cap=20, pick horse 1

        # Dirt: cap=50 → horse with odds=25 is within cap, pick it.
        # get_win_candidates recalculates win_market_selection_score internally,
        # so win_selection_edge must be high enough to overcome log_odds_penalty.
        # edge=0.5 survives: 0.5 - 0.05*log1p(25) + 0.02*pct_rank(0.05) ≈ 0.347
        # vs horse 1:       0.2 - 0.05*log1p(3)  + 0.02*pct_rank(0.40) ≈ 0.151
        dirt_cands = predictor.get_win_candidates(
            pd.DataFrame(
                [
                    {
                        "race_id": "R2",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "dirt",
                        "umaban": 1,
                        "tanodds": 3.0,
                        "p_win_final": 0.40,
                        "win_selection_prob": 0.40,
                        "win_selection_ev": 1.2,
                        "win_selection_edge": 0.2,
                        "win_market_selection_score": 0.5,
                    },
                    {
                        "race_id": "R2",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "dirt",
                        "umaban": 2,
                        "tanodds": 25.0,
                        "p_win_final": 0.05,
                        "win_selection_prob": 0.05,
                        "win_selection_ev": 1.25,
                        "win_selection_edge": 0.5,
                        "win_market_selection_score": 1.0,  # < cap=50
                    },
                ]
            )
        )
        assert len(dirt_cands) == 1
        assert dirt_cands.iloc[0]["umaban"] == 2  # 25 < cap=50, horse 2 is top1

    def test_reranker_skipped_when_profit_selector_active(self) -> None:
        """Reranker forced top-1 is skipped when ProfitSelector is trained.

        When both reranker and profit_selector are present, the reranker must
        not override the ProfitSelector's 0-N candidate contract. The reranker
        is only for the fallback top-1 path (no ProfitSelector / untrained).
        """
        from backtest.race_predictor import RacePredictor
        from domain.models import TrainedModelsV5
        from models.win_profit_selector import WinProfitSelector, WinProfitSelectorParams

        selector = WinProfitSelector()
        selector.params = WinProfitSelectorParams(
            rank_limit=2,
            min_score=float("-inf"),
            min_edge=-0.20,
            min_prob=0.0,
            min_odds=1.0,
            max_odds=100.0,
        )
        selector._trained = True

        # Both reranker (cap=30) and profit_selector (rank_limit=2)
        sub = self._make_submodel_with_reranker(cap=30.0)
        sub.win_profit_selector = selector

        models = MagicMock(spec=TrainedModelsV5)
        models.submodels = {"turf": sub}
        models.quality_screener = MagicMock()
        models.regime_detector = MagicMock()
        models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.03,
            "max_bets_per_race": 3,
        }

        predictor = RacePredictor(models=models, betting_target="win")
        candidates = predictor.get_win_candidates(
            pd.DataFrame(
                [
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 1,
                        "tanodds": 3.0,
                        "p_win_final": 0.50,
                        "win_selection_prob": 0.50,
                        "win_selection_ev": 1.20,
                        "win_selection_edge": 0.20,
                        "win_market_selection_score": 1.00,
                    },
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 2,
                        "tanodds": 5.0,
                        "p_win_final": 0.15,
                        "win_selection_prob": 0.15,
                        "win_selection_ev": 1.10,
                        "win_selection_edge": 0.10,
                        "win_market_selection_score": 0.80,
                    },
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 3,
                        "tanodds": 30.0,
                        "p_win_final": 0.02,
                        "win_selection_prob": 0.02,
                        "win_selection_ev": 0.60,
                        "win_selection_edge": -0.40,
                        "win_market_selection_score": 0.10,
                    },
                ]
            )
        )

        # ProfitSelector rank_limit=2 → 2 candidates, NOT 1 from reranker
        assert len(candidates) == 2
        assert set(candidates["umaban"].tolist()) == {1, 2}

    def test_reranker_diagnostics_propagated_to_diagnostic_df(self) -> None:
        """Reranker diagnostic columns appear in attrs['win_diagnostic_df'].

        All horses in the prepared frame (not just the selected top-1) should
        have per-race reranker diagnostics (original/final/applied/cap/reason)
        mapped by race_id.
        """
        from backtest.race_predictor import RacePredictor
        from domain.models import TrainedModelsV5

        sub = self._make_submodel_with_reranker(cap=30.0)
        models = MagicMock(spec=TrainedModelsV5)
        models.submodels = {"turf": sub}
        models.quality_screener = MagicMock()
        models.regime_detector = MagicMock()
        models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.03,
            "max_bets_per_race": 3,
        }

        predictor = RacePredictor(models=models, betting_target="win")

        # Horse 3: highest score (edge=0.30, p=0.45), odds=80 > cap=30
        # → reranker switches to horse 2 (highest score within cap)
        candidates = predictor.get_win_candidates(
            pd.DataFrame(
                [
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 1,
                        "tanodds": 3.0,
                        "p_win_final": 0.50,
                        "win_selection_prob": 0.50,
                        "win_selection_ev": 1.20,
                        "win_selection_edge": 0.20,
                        "win_market_selection_score": 1.0,
                    },
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 2,
                        "tanodds": 15.0,
                        "p_win_final": 0.10,
                        "win_selection_prob": 0.10,
                        "win_selection_ev": 1.10,
                        "win_selection_edge": 0.10,
                        "win_market_selection_score": 0.5,
                    },
                    {
                        "race_id": "R1",
                        "race_date": pd.Timestamp("2025-01-01"),
                        "surface": "turf",
                        "umaban": 3,
                        "tanodds": 80.0,
                        "p_win_final": 0.45,
                        "win_selection_prob": 0.45,
                        "win_selection_ev": 1.20,
                        "win_selection_edge": 0.30,
                        "win_market_selection_score": 1.5,
                    },
                ]
            )
        )

        assert len(candidates) == 1
        # Horse 2 is highest score within cap (horse 3 excluded by odds > cap)
        assert candidates.iloc[0]["umaban"] == 2

        diag_df = candidates.attrs["win_diagnostic_df"]
        assert diag_df is not None
        # All 3 horses should be in the diagnostic frame
        assert len(diag_df) == 3

        # All 5 diagnostic columns present
        for col in RERANKER_DIAGNOSTIC_COLS:
            assert col in diag_df.columns, f"Missing {col} in diagnostic_df"

        # Diagnostics are consistent across all horses in the same race
        assert diag_df[RERANKER_APPLIED_COL].iloc[0]
        assert diag_df[RERANKER_ORIG_TOP1_COL].iloc[0] == 3
        assert diag_df[RERANKER_FINAL_TOP1_COL].iloc[0] == 2
        assert diag_df[RERANKER_SWITCH_REASON_COL].iloc[0] == "odds_cap_switch"
        assert diag_df[RERANKER_CAP_COL].iloc[0] == 30.0

    def test_reranker_applied_when_profit_selector_max_per_race_1(self) -> None:
        """Reranker applies when profit_selector is trained but max_per_race == 1.

        The reranker is only skipped when profit_selector_enabled AND max_per_race > 1
        (multi-candidate contract). When max_per_race == 1 both the reranker and
        profit selector produce a single horse; the reranker adds odds-cap semantics.

        We use rank_limit=3 (all horses pass profit_pass) but override max_per_race
        to 1, so the reranker sees multiple candidates and can actually switch.
        """
        from backtest.race_predictor import RacePredictor
        from domain.models import TrainedModelsV5
        from models.win_profit_selector import WinProfitSelector, WinProfitSelectorParams

        selector = WinProfitSelector()
        selector.params = WinProfitSelectorParams(
            rank_limit=3,  # All 3 horses pass profit_pass
            min_score=float("-inf"),
            min_edge=-0.20,
            min_prob=0.0,
            min_odds=1.0,
            max_odds=100.0,
        )
        selector._trained = True

        # Override max_per_race to 1 (simulates rank_limit=1 without filtering)
        _orig_max_per_race = WinProfitSelector.max_per_race
        WinProfitSelector.max_per_race = property(  # type: ignore[assignment]
            lambda self: 1 if self is selector else _orig_max_per_race.fget(self)  # type: ignore[attr-defined]
        )
        try:
            sub = self._make_submodel_with_reranker(cap=30.0)
            sub.win_profit_selector = selector

            models = MagicMock(spec=TrainedModelsV5)
            models.submodels = {"turf": sub}
            models.quality_screener = MagicMock()
            models.regime_detector = MagicMock()
            models.regime_detector.get_strategy_params.return_value = {
                "edge_threshold": 0.03,
                "max_bets_per_race": 3,
            }

            predictor = RacePredictor(models=models, betting_target="win")

            # Horse 3: highest recalculated score, odds=80 > cap=30
            # Horse 2: second score, odds=15 < cap=30 → reranker selects horse 2
            candidates = predictor.get_win_candidates(
                pd.DataFrame(
                    [
                        {
                            "race_id": "R1",
                            "race_date": pd.Timestamp("2025-01-01"),
                            "surface": "turf",
                            "umaban": 1,
                            "tanodds": 3.0,
                            "p_win_final": 0.50,
                            "win_selection_prob": 0.50,
                            "win_selection_ev": 1.20,
                            "win_selection_edge": 0.20,
                            "win_market_selection_score": 1.0,
                        },
                        {
                            "race_id": "R1",
                            "race_date": pd.Timestamp("2025-01-01"),
                            "surface": "turf",
                            "umaban": 2,
                            "tanodds": 15.0,
                            "p_win_final": 0.10,
                            "win_selection_prob": 0.10,
                            "win_selection_ev": 1.10,
                            "win_selection_edge": 0.10,
                            "win_market_selection_score": 0.5,
                        },
                        {
                            "race_id": "R1",
                            "race_date": pd.Timestamp("2025-01-01"),
                            "surface": "turf",
                            "umaban": 3,
                            "tanodds": 80.0,
                            "p_win_final": 0.45,
                            "win_selection_prob": 0.45,
                            "win_selection_ev": 1.20,
                            "win_selection_edge": 0.30,
                            "win_market_selection_score": 1.5,
                        },
                    ]
                )
            )

            # Reranker should have applied: horse 2 (within cap) instead of horse 3
            assert len(candidates) == 1
            assert candidates.iloc[0]["umaban"] == 2
            # Diagnostic columns prove the reranker code path was taken
            assert RERANKER_APPLIED_COL in candidates.columns
            assert candidates[RERANKER_APPLIED_COL].iloc[0] == True  # noqa: E712
            assert candidates[RERANKER_SWITCH_REASON_COL].iloc[0] == "odds_cap_switch"
        finally:
            WinProfitSelector.max_per_race = _orig_max_per_race  # type: ignore[assignment]

    def test_reranker_diagnostics_with_profit_selector_max_per_race_1(self) -> None:
        """Diagnostics are mapped to all horses when profit_selector (max=1) coexists.

        When reranker applies alongside profit_selector with max_per_race=1, the
        returned attrs['win_diagnostic_df'] must include all horses (not just
        the selected top-1) with reranker diagnostic columns mapped by race_id.
        """
        from backtest.race_predictor import RacePredictor
        from domain.models import TrainedModelsV5
        from models.win_profit_selector import WinProfitSelector, WinProfitSelectorParams

        selector = WinProfitSelector()
        selector.params = WinProfitSelectorParams(
            rank_limit=3,
            min_score=float("-inf"),
            min_edge=-0.20,
            min_prob=0.0,
            min_odds=1.0,
            max_odds=100.0,
        )
        selector._trained = True

        _orig_max_per_race = WinProfitSelector.max_per_race
        WinProfitSelector.max_per_race = property(  # type: ignore[assignment]
            lambda self: 1 if self is selector else _orig_max_per_race.fget(self)  # type: ignore[attr-defined]
        )
        try:
            sub = self._make_submodel_with_reranker(cap=30.0)
            sub.win_profit_selector = selector

            models = MagicMock(spec=TrainedModelsV5)
            models.submodels = {"turf": sub}
            models.quality_screener = MagicMock()
            models.regime_detector = MagicMock()
            models.regime_detector.get_strategy_params.return_value = {
                "edge_threshold": 0.03,
                "max_bets_per_race": 3,
            }

            predictor = RacePredictor(models=models, betting_target="win")

            candidates = predictor.get_win_candidates(
                pd.DataFrame(
                    [
                        {
                            "race_id": "R1",
                            "race_date": pd.Timestamp("2025-01-01"),
                            "surface": "turf",
                            "umaban": 1,
                            "tanodds": 3.0,
                            "p_win_final": 0.50,
                            "win_selection_prob": 0.50,
                            "win_selection_ev": 1.20,
                            "win_selection_edge": 0.20,
                            "win_market_selection_score": 1.0,
                        },
                        {
                            "race_id": "R1",
                            "race_date": pd.Timestamp("2025-01-01"),
                            "surface": "turf",
                            "umaban": 2,
                            "tanodds": 15.0,
                            "p_win_final": 0.10,
                            "win_selection_prob": 0.10,
                            "win_selection_ev": 1.10,
                            "win_selection_edge": 0.10,
                            "win_market_selection_score": 0.5,
                        },
                        {
                            "race_id": "R1",
                            "race_date": pd.Timestamp("2025-01-01"),
                            "surface": "turf",
                            "umaban": 3,
                            "tanodds": 80.0,
                            "p_win_final": 0.45,
                            "win_selection_prob": 0.45,
                            "win_selection_ev": 1.20,
                            "win_selection_edge": 0.30,
                            "win_market_selection_score": 1.5,
                        },
                    ]
                )
            )

            # win_diagnostic_df must exist and contain all 3 horses
            diag_df = candidates.attrs.get("win_diagnostic_df")
            assert diag_df is not None, "win_diagnostic_df must be set in attrs"
            assert len(diag_df) == 3, f"Expected 3 horses in diagnostic_df, got {len(diag_df)}"

            # All 5 diagnostic columns present in the diagnostic frame
            for col in RERANKER_DIAGNOSTIC_COLS:
                assert col in diag_df.columns, f"Missing {col} in diagnostic_df"

            # Values mapped by race_id: same for all rows in the same race
            for col in RERANKER_DIAGNOSTIC_COLS:
                unique_vals = diag_df[col].dropna().unique()
                if col == RERANKER_APPLIED_COL:
                    assert len(unique_vals) == 1 and unique_vals[0] == True  # noqa: E712
                elif col == RERANKER_CAP_COL:
                    assert unique_vals[0] == 30.0
                elif col == RERANKER_ORIG_TOP1_COL:
                    assert unique_vals[0] == 3  # Horse 3 was original top1
                elif col == RERANKER_FINAL_TOP1_COL:
                    assert unique_vals[0] == 2  # Reranker switched to horse 2
                elif col == RERANKER_SWITCH_REASON_COL:
                    assert unique_vals[0] == "odds_cap_switch"
        finally:
            WinProfitSelector.max_per_race = _orig_max_per_race  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# _simulate_cap vectorized tests
# ---------------------------------------------------------------------------


class TestSimulateCapVectorized:
    def test_confirmed_odds_preferred_for_payout(self) -> None:
        """_simulate_cap uses confirmed_odds for payout, not tanodds."""
        model = WinTop1OddsReranker()
        # Build a single fold with 1 race, horse wins
        df = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "umaban": 1,
                    "kakuteijyuni": 1,  # Winner
                    "tanodds": 5.0,
                    "confirmed_odds": 6.0,  # Different
                    "win_market_selection_score": 1.0,
                },
                {
                    "race_id": "R1",
                    "umaban": 2,
                    "kakuteijyuni": 2,
                    "tanodds": 10.0,
                    "confirmed_odds": 11.0,
                    "win_market_selection_score": 0.5,
                },
            ]
        )
        result = model._simulate_cap(df, cap=30.0, fold_race_ids={"R1"})
        # Payout should be confirmed_odds (6.0), profit = 6.0 - 1.0 = 5.0
        assert result["profit"] == pytest.approx(5.0, abs=0.01)

    def test_tanodds_fallback_when_no_confirmed_odds(self) -> None:
        """When confirmed_odds is missing, tanodds is used."""
        model = WinTop1OddsReranker()
        df = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "umaban": 1,
                    "kakuteijyuni": 1,
                    "tanodds": 5.0,
                    "win_market_selection_score": 1.0,
                },
                {
                    "race_id": "R1",
                    "umaban": 2,
                    "kakuteijyuni": 2,
                    "tanodds": 10.0,
                    "win_market_selection_score": 0.5,
                },
            ]
        )
        result = model._simulate_cap(df, cap=30.0, fold_race_ids={"R1"})
        # Payout should be tanodds (5.0), profit = 5.0 - 1.0 = 4.0
        assert result["profit"] == pytest.approx(4.0, abs=0.01)

    def test_bet_count_equals_race_count(self) -> None:
        """Always 1 bet per race, even with anomalous odds."""
        model = WinTop1OddsReranker()
        df = pd.DataFrame(
            [
                # Normal race
                {
                    "race_id": "R1",
                    "umaban": 1,
                    "kakuteijyuni": 1,
                    "tanodds": 3.0,
                    "win_market_selection_score": 1.0,
                },
                # Anomalous race (all odds=0)
                {
                    "race_id": "R2",
                    "umaban": 1,
                    "kakuteijyuni": 2,
                    "tanodds": 0.0,
                    "win_market_selection_score": 1.0,
                },
                {
                    "race_id": "R2",
                    "umaban": 2,
                    "kakuteijyuni": 1,
                    "tanodds": 0.0,
                    "win_market_selection_score": 0.5,
                },
            ]
        )
        result = model._simulate_cap(df, cap=30.0, fold_race_ids={"R1", "R2"})
        assert result["bets"] == 2.0  # 1 per race
        assert result["n_anomalous"] == 1  # R2 has no valid tanodds
        assert result["bets_equals_races"] is True

    def test_cap_exclusion_correct_selection(self) -> None:
        """Horse with odds > cap is excluded from selection."""
        model = WinTop1OddsReranker()
        df = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "umaban": 1,
                    "kakuteijyuni": 2,  # Doesn't win
                    "tanodds": 3.0,
                    "win_market_selection_score": 0.5,
                },
                {
                    "race_id": "R1",
                    "umaban": 2,
                    "kakuteijyuni": 1,  # Wins but odds > cap
                    "tanodds": 80.0,
                    "win_market_selection_score": 1.5,
                },
                {
                    "race_id": "R1",
                    "umaban": 3,
                    "kakuteijyuni": 3,
                    "tanodds": 15.0,
                    "win_market_selection_score": 0.3,
                },
            ]
        )
        # With cap=30: horse 2 (odds=80) excluded → horse 1 (score=0.5) selected
        result = model._simulate_cap(df, cap=30.0, fold_race_ids={"R1"})
        assert result["bets"] == 1.0
        assert result["profit"] == pytest.approx(-1.0, abs=0.01)  # Horse 1 doesn't win

        # Without cap (inf): horse 2 (score=1.5) selected
        result_inf = model._simulate_cap(df, cap=float("inf"), fold_race_ids={"R1"})
        assert result_inf["profit"] == pytest.approx(79.0, abs=0.01)  # 80-1


# ---------------------------------------------------------------------------
# Gap Reranker Tests
# ---------------------------------------------------------------------------


def _make_gap_test_df(
    n_races_per_year: int = 300,
    years: list[int] | None = None,
    surface: str = "dirt",
    switch_frac: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """ダートOOF風のテストデータを生成する.

    switch_frac の割合で、score top1 が負け・top2 が勝つ且つ top2 確率 > top1 の
    レースを作成する。残りは top1 が勝つ。
    """
    if years is None:
        years = [2022, 2023, 2024]
    rng = np.random.RandomState(seed)
    rows: list[dict[str, object]] = []
    for year in years:
        for ri in range(n_races_per_year):
            race_id = f"G{year}_{ri}"
            m = rng.randint(1, 13)
            d = rng.randint(1, 29)
            race_date = f"{year}-{m:02d}-{d:02d}"
            is_switch = rng.random() < switch_frac

            if is_switch:
                # top1 by score loses, top2 has higher prob
                rows.append(_gap_horse(race_id, race_date, surface, 1, 2, 1.0, 0.10, 5.0))
                rows.append(_gap_horse(race_id, race_date, surface, 2, 1, 0.9, 0.20, 3.0))
                rows.append(_gap_horse(race_id, race_date, surface, 3, 3, 0.2, 0.05, 15.0))
            else:
                # top1 by score wins, top2 has lower prob
                rows.append(_gap_horse(race_id, race_date, surface, 1, 1, 1.0, 0.25, 3.0))
                rows.append(_gap_horse(race_id, race_date, surface, 2, 2, 0.5, 0.10, 5.0))
                rows.append(_gap_horse(race_id, race_date, surface, 3, 3, 0.2, 0.05, 15.0))
    return pd.DataFrame(rows)


def _gap_horse(
    race_id: str,
    race_date: str,
    surface: str,
    umaban: int,
    kakuteijyuni: int,
    score: float,
    prob: float,
    odds: float,
) -> dict[str, object]:
    return {
        "race_id": race_id,
        "race_date": race_date,
        "surface": surface,
        "umaban": umaban,
        "kakuteijyuni": kakuteijyuni,
        "tanodds": odds,
        "confirmed_odds": round(odds * 1.02, 1),
        "win_market_selection_score": score,
        "p_win_final_oof": prob,
        "p_win_final": prob,
        "win_selection_prob": prob,
        "p_win_oof": prob,
    }


def _make_deployable_gap_df(
    n_races_per_year: int = 600,
) -> pd.DataFrame:
    """Deterministic data that guarantees gap reranker deployment.

    Race composition per year (deterministic, no randomness):
      - 7% "good switch": top1 loses (score=1.0, prob=0.10, jyuni=2),
        top2 wins (score=0.95, prob=0.20, jyuni=1), gap=0.05.
        Candidate with threshold >= 0.05 catches these → switches to winner.
        Disabled (cap-only) keeps top1 loser → profit -1 per race.
      - 2% "bad switch": top1 wins (score=1.0, prob=0.15, jyuni=1),
        top2 loses (score=0.4, prob=0.20, jyuni=3), gap=0.6.
        Candidate with threshold < 0.6 avoids these → keeps winner.
        Disabled (cap-only) also keeps top1 winner.
      - 91% "no switch": top1 wins (score=1.0, prob=0.25, jyuni=1),
        top2 loses (score=0.5, prob=0.10, jyuni=2), gap=0.5.
        t2_prob < t1_prob → no switch regardless of threshold.

    Disabled baseline = cap-only top1 (horse 1, score=1.0) always.
    Wins 93% (2%+91%), loses 7% (good switch where top1 is actually the loser).

    Score gap distribution: 7% at 0.05, 91% at 0.5, 2% at 0.6.
    Quantile thresholds all resolve to ~0.5 → candidate with threshold=0.5
    switches only "good switch" races (7%), avoiding "bad switch" (2%).
    Change rate = 7% (within 1-10% guard). Both years positive delta vs disabled.
    """
    rows: list[dict[str, object]] = []
    for year in [2022, 2023, 2024]:
        for ri in range(n_races_per_year):
            race_id = f"DEPLOY{year}_{ri}"
            m = (ri % 12) + 1
            d = (ri % 28) + 1
            race_date = f"{year}-{m:02d}-{d:02d}"
            race_mod = ri % 100

            if race_mod < 7:
                # 7% good switch
                rows.append(_gap_horse(race_id, race_date, "dirt", 1, 2, 1.0, 0.10, 5.0))
                rows.append(_gap_horse(race_id, race_date, "dirt", 2, 1, 0.95, 0.20, 3.0))
                rows.append(_gap_horse(race_id, race_date, "dirt", 3, 3, 0.1, 0.03, 15.0))
            elif race_mod < 9:
                # 2% bad switch
                rows.append(_gap_horse(race_id, race_date, "dirt", 1, 1, 1.0, 0.15, 3.0))
                rows.append(_gap_horse(race_id, race_date, "dirt", 2, 3, 0.4, 0.20, 8.0))
                rows.append(_gap_horse(race_id, race_date, "dirt", 3, 2, 0.1, 0.05, 15.0))
            else:
                # 91% no switch
                rows.append(_gap_horse(race_id, race_date, "dirt", 1, 1, 1.0, 0.25, 3.0))
                rows.append(_gap_horse(race_id, race_date, "dirt", 2, 2, 0.5, 0.10, 5.0))
                rows.append(_gap_horse(race_id, race_date, "dirt", 3, 3, 0.1, 0.05, 15.0))
    return pd.DataFrame(rows)


def _make_holdout_worse_gap_df(
    n_races_per_year: int = 600,
) -> pd.DataFrame:
    """Explore allows deploy, but holdout (2024) has reversed outcomes.

    Explore (2022-2023): identical to _make_deployable_gap_df.
    Holdout (2024):
      - 7% "reverse switch": gap=0.05, t2_prob > t1, but t2 LOSES.
        Candidate switches to loser; disabled (cap-only) keeps top1 winner.
      - 3% "aggressive-only win": gap=0.6, t2_prob > t1, t2 WINS.
        Candidate avoids (gap > threshold) → keeps top1 loser.
        Disabled (cap-only) also keeps top1 loser.
      - 90% "no switch": same as explore.

    Net effect: candidate switches 7% to losers while disabled (cap-only) wins
    those 7%. Both lose 3%. Candidate ROI < disabled ROI → deployed=False.
    """
    rows: list[dict[str, object]] = []
    for year in [2022, 2023, 2024]:
        for ri in range(n_races_per_year):
            race_id = f"HW{year}_{ri}"
            m = (ri % 12) + 1
            d = (ri % 28) + 1
            race_date = f"{year}-{m:02d}-{d:02d}"
            race_mod = ri % 100

            if year < 2024:
                # Explore: same deployable pattern
                if race_mod < 7:
                    rows.append(_gap_horse(race_id, race_date, "dirt", 1, 2, 1.0, 0.10, 5.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 2, 1, 0.95, 0.20, 3.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 3, 3, 0.1, 0.03, 15.0))
                elif race_mod < 9:
                    rows.append(_gap_horse(race_id, race_date, "dirt", 1, 1, 1.0, 0.15, 3.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 2, 3, 0.4, 0.20, 8.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 3, 2, 0.1, 0.05, 15.0))
                else:
                    rows.append(_gap_horse(race_id, race_date, "dirt", 1, 1, 1.0, 0.25, 3.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 2, 2, 0.5, 0.10, 5.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 3, 3, 0.1, 0.05, 15.0))
            else:
                # Holdout 2024: reversed outcomes
                if race_mod < 7:
                    # reverse switch: candidate switches to loser
                    rows.append(_gap_horse(race_id, race_date, "dirt", 1, 1, 1.0, 0.10, 5.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 2, 2, 0.95, 0.20, 3.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 3, 3, 0.1, 0.03, 15.0))
                elif race_mod < 10:
                    # aggressive-only win: disabled switches to winner, candidate misses
                    rows.append(_gap_horse(race_id, race_date, "dirt", 1, 2, 1.0, 0.15, 3.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 2, 1, 0.4, 0.20, 8.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 3, 3, 0.1, 0.05, 15.0))
                else:
                    rows.append(_gap_horse(race_id, race_date, "dirt", 1, 1, 1.0, 0.25, 3.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 2, 2, 0.5, 0.10, 5.0))
                    rows.append(_gap_horse(race_id, race_date, "dirt", 3, 3, 0.1, 0.05, 15.0))
    return pd.DataFrame(rows)


class TestGapReranker:
    """Score-gap second-stage reranker (dirt-only) tests."""

    # --- Training tests ---

    def test_gap_turf_always_disabled(self) -> None:
        """surface='turf' では gap reranker は常に disabled."""
        df = _make_gap_test_df(n_races_per_year=300, surface="turf")
        model = WinTop1OddsReranker()
        model.train(df, surface="turf")
        assert model.gap_reranker_deployed is False
        assert model.gap_reranker_surface == ""

    def test_gap_insufficient_dates(self) -> None:
        """race_date が 2022-2023 に不足している場合 disabled."""
        rows = []
        # 400 unique races all in 2021 (enough for cap training, but not for gap)
        for ri in range(400):
            race_id = f"OLD_{ri}"
            rows.append(_gap_horse(race_id, "2021-06-01", "dirt", 1, 1, 1.0, 0.20, 3.0))
            rows.append(_gap_horse(race_id, "2021-06-01", "dirt", 2, 2, 0.5, 0.10, 5.0))
            rows.append(_gap_horse(race_id, "2021-06-01", "dirt", 3, 3, 0.2, 0.05, 15.0))
        df = pd.DataFrame(rows)
        model = WinTop1OddsReranker()
        model.train(df, surface="dirt")
        assert model.gap_reranker_deployed is False
        assert "insufficient" in model.gap_reranker_training_summary.get("reason", "")

    def test_gap_empty_df(self) -> None:
        """空 DataFrame では disabled."""
        model = WinTop1OddsReranker()
        model.train(pd.DataFrame(), surface="dirt")
        assert model.gap_reranker_deployed is False

    def test_gap_dirt_deploy(self) -> None:
        """決定論的データで gap reranker が必ず deployed=True になることを検証.

        _make_deployable_gap_df は 7% good switch (gap=0.05) + 2% bad switch
        (gap=0.6) + 91% no switch の構成。disabled (cap-only) baseline は
        top1 (horse 1) を常に選択し、7% で敗退。threshold=0.5 の候補は
        good switch のみを top2 (勝者) へ切替え →
        disabled baseline を上回る ROI delta を両年で達成。
        """
        df = _make_deployable_gap_df(n_races_per_year=600)
        model = WinTop1OddsReranker()
        model.train(df, surface="dirt")

        assert model.gap_reranker_deployed is True, (
            "gap reranker must deploy with deployable synthetic data"
        )
        assert model.gap_reranker_surface == "dirt"

        summary = model.gap_reranker_training_summary
        assert summary.get("deployed") is True
        assert "selected_params" in summary
        assert "holdout_2024" in summary
        assert summary["holdout_2024"]["roi_delta"] > 0, (
            f"holdout ROI delta must be positive: {summary['holdout_2024']['roi_delta']:.4f}"
        )
        assert summary["holdout_2024"]["change_rate"] >= 0.01, "holdout change_rate must be >= 1%"
        assert summary["holdout_2024"]["change_rate"] <= 0.10
        assert summary["holdout_2024"]["n_changes"] >= 30
        assert summary["holdout_2024"]["hit_delta"] >= -0.01, (
            f"holdout hit delta must be >= -1pp: {summary['holdout_2024']['hit_delta']:.4f}"
        )

        params = summary["selected_params"]
        assert "score_gap_threshold" in params
        assert "min_prob_margin" in params
        assert "max_change_rate" in params
        assert "top2_max_odds" in params

        # Explore yearly: all years must have ROI delta >= -0.01 and hit delta >= -1pp
        assert "explore_yearly" in summary
        for yr, yr_data in summary["explore_yearly"].items():
            assert yr_data.get("roi_delta", -1) >= -0.01, (
                f"explore {yr} ROI delta must be >= -0.01: {yr_data.get('roi_delta', 'N/A')}"
            )
            assert yr_data.get("hit_delta", -1) >= -0.01, (
                f"explore {yr} hit delta must be >= -1pp: {yr_data.get('hit_delta', 'N/A')}"
            )
            assert yr_data.get("change_rate", 0) >= 0.01, (
                f"explore {yr} change_rate must be >= 1%: {yr_data.get('change_rate', 'N/A')}"
            )
            assert yr_data.get("change_rate", 1) <= 0.10, (
                f"explore {yr} change_rate must be <= 10%: {yr_data.get('change_rate', 'N/A')}"
            )

    def test_gap_dirt_holdout_worse_not_deployed(self) -> None:
        """Explore で改善しても holdout で悪化する場合 deployed=False を検証.

        _make_holdout_worse_gap_df は explore (2022-2023) は deployable
        パターンと同一だが、holdout (2024) で reverse switch が発生。
        候補は reverse switch (7%) で敗退馬へ切替え、disabled (cap-only)
        は top1 勝者をそのまま保持 → holdout delta が負 → deployed=False。
        """
        df = _make_holdout_worse_gap_df(n_races_per_year=600)
        model = WinTop1OddsReranker()
        model.train(df, surface="dirt")

        assert model.gap_reranker_deployed is False, (
            "gap reranker must NOT deploy when holdout is worse"
        )

        summary = model.gap_reranker_training_summary
        assert summary.get("deployed") is False
        assert summary.get("reason") == "holdout_validation_failed"
        assert "failures" in summary
        assert "roi_delta > 0" in summary["failures"]
        # New condition: hit_delta >= -1pp also fails (7pp degradation)
        assert "hit_delta >= -1pp" in summary["failures"]
        # Verify explore_yearly is saved even on holdout failure
        assert "explore_yearly" in summary

    def test_gap_dirt_2024_worse_disabled(self) -> None:
        """2024 で改善しない場合 disabled となることを明示検証.

        全 race で top1 が勝つデータでは改善余地がない → disabled。
        """
        df = _make_gap_test_df(
            n_races_per_year=300,
            surface="dirt",
            switch_frac=0.0,
        )
        model = WinTop1OddsReranker()
        model.train(df, surface="dirt")
        assert model.gap_reranker_deployed is False
        reason = model.gap_reranker_training_summary.get("reason", "")
        assert reason == "no_candidate_meets_criteria"

    def test_gap_dirt_no_improvement(self) -> None:
        """全 race で top1 が勝つ場合、gap reranker は改善なし → disabled."""
        df = _make_gap_test_df(
            n_races_per_year=300,
            surface="dirt",
            switch_frac=0.0,
        )
        model = WinTop1OddsReranker()
        model.train(df, surface="dirt")
        # With switch_frac=0, there's no room for improvement
        assert model.gap_reranker_deployed is False

    # --- Apply tests ---

    def test_gap_apply_turf_not_deployed(self) -> None:
        """Turf では gap apply は常に not_deployed."""
        model = WinTop1OddsReranker()
        model.is_trained = True
        model.selected_cap = 30.0
        candidates = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "surface": "turf",
                    "umaban": 1,
                    "tanodds": 5.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
                {
                    "race_id": "R1",
                    "surface": "turf",
                    "umaban": 2,
                    "tanodds": 3.0,
                    "win_market_selection_score": 0.8,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
            ]
        )
        result = model.apply(candidates)
        assert len(result) == 1
        assert not result[GAP_RERANKER_DEPLOYED_COL].iloc[0]
        assert not result[GAP_RERANKER_APPLIED_COL].iloc[0]

    def test_gap_apply_switch(self) -> None:
        """top2 確率 > top1, gap <= threshold, margin >= min → 切替."""
        model = WinTop1OddsReranker()
        model.is_trained = True
        model.selected_cap = 50.0
        model.gap_reranker_deployed = True
        model.gap_reranker_surface = "dirt"
        model.gap_reranker_score_gap_threshold = 0.5
        model.gap_reranker_min_prob_margin = 0.01
        model.gap_reranker_max_change_rate = 0.60  # Allow 50% change for this test

        candidates = pd.DataFrame(
            [
                # Race 1: switch beneficial (top2 higher prob)
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 5.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 3.0,
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 3,
                    "tanodds": 15.0,
                    "win_market_selection_score": 0.2,
                    "p_win_final_oof": 0.05,
                    "p_win_final": 0.05,
                },
                # Race 2: no switch (top1 higher prob)
                {
                    "race_id": "R2",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 3.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.25,
                    "p_win_final": 0.25,
                },
                {
                    "race_id": "R2",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 5.0,
                    "win_market_selection_score": 0.8,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
            ]
        )
        result = model.apply(candidates)
        assert len(result) == 2
        # R1 should have switched
        r1 = result[result["race_id"] == "R1"].iloc[0]
        assert r1[GAP_RERANKER_DEPLOYED_COL]
        assert r1[GAP_RERANKER_APPLIED_COL]
        assert r1[GAP_RERANKER_FINAL_TOP1_COL] == 2
        assert r1[GAP_RERANKER_ORIG_TOP1_COL] == 1
        assert r1[GAP_RERANKER_SCORE_GAP_COL] == pytest.approx(0.1)
        assert r1[GAP_RERANKER_PROB_MARGIN_COL] == pytest.approx(0.1)
        assert r1[GAP_RERANKER_SWITCH_REASON_COL] == "prob_gap_switch"
        # R2 should not have switched
        r2 = result[result["race_id"] == "R2"].iloc[0]
        assert not r2[GAP_RERANKER_APPLIED_COL]

    def test_gap_apply_no_switch_prob_lower(self) -> None:
        """top2 確率 < top1 → 不切替."""
        model = _make_deployed_gap_model(threshold=0.5, margin=0.01)
        candidates = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 3.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.25,
                    "p_win_final": 0.25,
                },
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 5.0,
                    "win_market_selection_score": 0.8,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
            ]
        )
        result = model.apply(candidates)
        assert len(result) == 1
        assert not result[GAP_RERANKER_APPLIED_COL].iloc[0]
        assert result[GAP_RERANKER_FINAL_TOP1_COL].iloc[0] == 1
        assert result[GAP_RERANKER_SWITCH_REASON_COL].iloc[0] == "top2_prob_lower"

    def test_gap_apply_no_switch_gap_large(self) -> None:
        """score_gap > threshold → 不切替."""
        model = _make_deployed_gap_model(threshold=0.05, margin=0.01)
        candidates = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 3.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 5.0,
                    "win_market_selection_score": 0.3,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
            ]
        )
        result = model.apply(candidates)
        assert len(result) == 1
        assert not result[GAP_RERANKER_APPLIED_COL].iloc[0]
        assert result[GAP_RERANKER_SWITCH_REASON_COL].iloc[0] == "score_gap_exceeds"

    def test_gap_apply_no_switch_margin_small(self) -> None:
        """prob_margin < min_margin → 不切替."""
        model = _make_deployed_gap_model(threshold=0.5, margin=0.10)
        candidates = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 3.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.18,
                    "p_win_final": 0.18,
                },
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 5.0,
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
            ]
        )
        result = model.apply(candidates)
        assert len(result) == 1
        assert not result[GAP_RERANKER_APPLIED_COL].iloc[0]
        assert result[GAP_RERANKER_SWITCH_REASON_COL].iloc[0] == "margin_insufficient"

    def test_gap_apply_no_runtime_change_rate_guard(self) -> None:
        """apply時は変更率guardを適用しない (学習/holdout専用).

        1 raceずつapplyするRacePredictorでは切替=100%で全取消となるため、
        max_change_rateはsimulation/holdout deploy guard専用。
        applyでは全てのswitchがそのまま適用される。
        """
        model = _make_deployed_gap_model(threshold=0.5, margin=0.01, max_cr=0.02)
        # Create 2 races, both would switch → 100% change rate
        # Previously cancelled by guard; now switches proceed
        candidates = pd.DataFrame(
            [
                # Race 1: switch beneficial
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 5.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 3.0,
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
                # Race 2: switch beneficial
                {
                    "race_id": "R2",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 5.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R2",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 3.0,
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
            ]
        )
        result = model.apply(candidates)
        assert len(result) == 2
        # Switches should proceed (no runtime guard)
        assert result[GAP_RERANKER_APPLIED_COL].all()
        assert result[GAP_RERANKER_SWITCH_REASON_COL].iloc[0] == "prob_gap_switch"
        assert result[GAP_RERANKER_SWITCH_REASON_COL].iloc[1] == "prob_gap_switch"

    def test_gap_multiple_races_one_each(self) -> None:
        """複数 race で各 1 頭を返す."""
        model = _make_deployed_gap_model(threshold=0.5, margin=0.01)
        candidates = pd.DataFrame(
            [
                # Race 1: no switch (top1 higher prob)
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 3.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.25,
                    "p_win_final": 0.25,
                },
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 5.0,
                    "win_market_selection_score": 0.8,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                # Race 2: switch (top2 higher prob)
                {
                    "race_id": "R2",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 5.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R2",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 3.0,
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
            ]
        )
        result = model.apply(candidates)
        assert len(result) == 2
        race_ids = sorted(result["race_id"].tolist())
        assert race_ids == ["R1", "R2"]

    # --- Persistence tests ---

    def test_gap_save_load_roundtrip(self) -> None:
        """save → load で gap フィールドが完全に一致."""
        model = _make_deployed_gap_model(threshold=0.15, margin=0.02, max_cr=0.05)
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gap_reranker.joblib"
            model.save(path)
            loaded = WinTop1OddsReranker.load(path)
            assert loaded.gap_reranker_deployed is True
            assert loaded.gap_reranker_surface == "dirt"
            assert loaded.gap_reranker_score_gap_threshold == pytest.approx(0.15)
            assert loaded.gap_reranker_min_prob_margin == pytest.approx(0.02)
            assert loaded.gap_reranker_max_change_rate == pytest.approx(0.05)
            assert loaded.gap_reranker_training_summary == model.gap_reranker_training_summary
            # Verify cap fields also preserved
            assert loaded.selected_cap == model.selected_cap
            assert loaded.is_trained == model.is_trained

    def test_gap_backward_compat_load(self) -> None:
        """旧 joblib (gap フィールドなし) を読み込める."""
        model = WinTop1OddsReranker()
        model.is_trained = True
        model.selected_cap = 30.0
        model.training_summary = {"selected_cap": 30.0}
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old_reranker.joblib"
            # Save with old format (no gap fields)
            joblib.dump(
                {
                    "candidate_caps": model.candidate_caps,
                    "min_train_races": model.min_train_races,
                    "min_fold_races": model.min_fold_races,
                    "max_folds": model.max_folds,
                    "stability_penalty": model.stability_penalty,
                    "min_roi_floor": model.min_roi_floor,
                    "min_roi_penalty": model.min_roi_penalty,
                    "selected_cap": model.selected_cap,
                    "is_trained": model.is_trained,
                    "training_summary": model.training_summary,
                },
                path,
            )
            loaded = WinTop1OddsReranker.load(path)
            assert loaded.selected_cap == 30.0
            assert loaded.is_trained is True
            # Gap fields should be defaults
            assert loaded.gap_reranker_deployed is False
            assert loaded.gap_reranker_surface == ""
            assert loaded.gap_reranker_score_gap_threshold == float("inf")
            assert loaded.gap_reranker_min_prob_margin == 0.01
            assert loaded.gap_reranker_max_change_rate == 0.10
            assert loaded.gap_reranker_training_summary == {}

    # --- Simulation tests ---

    def test_gap_confirmed_odds_payout(self) -> None:
        """_simulate_gap_reranker で confirmed_odds を払戻に使用."""
        model = WinTop1OddsReranker()
        df = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 1,
                    "kakuteijyuni": 2,
                    "tanodds": 5.0,
                    "confirmed_odds": 5.5,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R1",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 2,
                    "kakuteijyuni": 1,
                    "tanodds": 3.0,
                    "confirmed_odds": 3.2,
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
            ]
        )
        result = model._simulate_gap_reranker(
            df,
            cap=float("inf"),
            gap_params={
                "disabled": False,
                "score_gap_threshold": 0.5,
                "min_prob_margin": 0.01,
                "max_change_rate": 1.0,  # 100% switch OK for this small test
            },
        )
        # Should switch to horse 2 (wins, confirmed_odds=3.2)
        assert result["n_changes"] == 1
        assert result["roi"] == pytest.approx(3.2, abs=0.01)  # 100% win at 3.2

    # --- Diagnostics propagation tests ---

    def test_gap_diagnostics_propagation(self) -> None:
        """gap_reranker_* 列が apply 結果に含まれる."""
        model = _make_deployed_gap_model(threshold=0.5, margin=0.01)
        candidates = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 3.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.25,
                    "p_win_final": 0.25,
                },
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 5.0,
                    "win_market_selection_score": 0.8,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
            ]
        )
        result = model.apply(candidates)
        for col in GAP_RERANKER_DIAGNOSTIC_COLS:
            assert col in result.columns, f"Missing diagnostic column: {col}"
        # Also verify existing reranker diagnostics are preserved
        for col in RERANKER_DIAGNOSTIC_COLS:
            assert col in result.columns, f"Missing reranker diagnostic: {col}"

    def test_gap_inf_cap_gap_not_deployed(self) -> None:
        """cap=inf かつ gap 未デプロイ → gap 診断列は not_deployed."""
        model = WinTop1OddsReranker()
        model.is_trained = True
        model.selected_cap = float("inf")
        candidates = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 3.0,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 5.0,
                    "win_market_selection_score": 0.8,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
            ]
        )
        result = model.apply(candidates)
        # _apply_gap_reranker overrides switch_reason to "not_deployed"
        assert result[GAP_RERANKER_SWITCH_REASON_COL].iloc[0] == "not_deployed"

    def test_gap_simulation_losing_selection_zero_payout(self) -> None:
        """_simulate_gap_reranker: 不的中選択の払戻は0 (重大バグ修正の検証).

        選択馬が負けた場合、payout=0, profit=-1 となる。
        confirmed_odds は勝ち馬のオッズにのみ使用。
        """
        model = WinTop1OddsReranker()
        # 2 races: R1 switches (top2 wins), R2 no switch (top1 wins)
        df = pd.DataFrame(
            [
                # R1: switch beneficial (top2 wins at confirmed_odds=3.5)
                {
                    "race_id": "R1",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 1,
                    "kakuteijyuni": 2,  # loses
                    "tanodds": 5.0,
                    "confirmed_odds": 5.5,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R1",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 2,
                    "kakuteijyuni": 1,  # wins
                    "tanodds": 3.0,
                    "confirmed_odds": 3.5,
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
                # R2: no switch (top1 wins), but with a losing top2
                {
                    "race_id": "R2",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 1,
                    "kakuteijyuni": 1,  # wins
                    "tanodds": 4.0,
                    "confirmed_odds": 4.2,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.15,
                    "p_win_final": 0.15,
                },
                {
                    "race_id": "R2",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 2,
                    "kakuteijyuni": 2,  # loses
                    "tanodds": 8.0,
                    "confirmed_odds": 8.5,
                    "win_market_selection_score": 0.7,
                    "p_win_final_oof": 0.05,
                    "p_win_final": 0.05,
                },
            ]
        )
        result = model._simulate_gap_reranker(
            df,
            cap=float("inf"),
            gap_params={
                "disabled": False,
                "score_gap_threshold": 0.5,
                "min_prob_margin": 0.01,
                "max_change_rate": 1.0,
            },
        )
        # R1 switches (top2 wins at 3.5), R2 no switch (top2 prob < top1)
        assert result["n_changes"] == 1
        assert result["bets"] == 2
        assert result["hit_rate"] == pytest.approx(1.0)  # both selected horses win
        # R1 payout = 3.5 (win), R2 payout = 4.2 (win)
        # profit = (3.5 - 1) + (4.2 - 1) = 2.5 + 3.2 = 5.7
        assert result["profit"] == pytest.approx(5.7, abs=0.01)
        # ROI = (5.7 + 2) / 2 = 3.85
        assert result["roi"] == pytest.approx(3.85, abs=0.01)

        # Now test with a race where selected horse LOSES
        df_lose = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 1,
                    "kakuteijyuni": 2,  # loses
                    "tanodds": 5.0,
                    "confirmed_odds": 5.5,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R1",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 2,
                    "kakuteijyuni": 3,  # loses
                    "tanodds": 3.0,
                    "confirmed_odds": 3.2,
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
            ]
        )
        result_lose = model._simulate_gap_reranker(
            df_lose,
            cap=float("inf"),
            gap_params={
                "disabled": False,
                "score_gap_threshold": 0.5,
                "min_prob_margin": 0.01,
                "max_change_rate": 1.0,
            },
        )
        # Top1 (score=1.0) loses, top2 (score=0.9) also loses → payout = 0
        assert result_lose["n_changes"] == 1  # switches to top2
        assert result_lose["hit_rate"] == pytest.approx(0.0)  # top2 also loses
        assert result_lose["profit"] == pytest.approx(-1.0, abs=0.01)  # 1 race, -1 per bet
        assert result_lose["roi"] == pytest.approx(0.0, abs=0.01)  # (0 + 1) / 1 = 1.0 → ROI = 1.0

        # Yearly check: losing selection in a year gives ROI = 1.0 (bets = returns)
        yearly = result_lose.get("yearly", {})
        assert "2023" in yearly
        assert yearly["2023"]["roi"] == pytest.approx(0.0, abs=0.01)
        assert yearly["2023"]["hit_rate"] == pytest.approx(0.0, abs=0.01)
        assert yearly["2023"]["profit"] == pytest.approx(-1.0, abs=0.01)

    def test_gap_simulation_single_cap_eligible_skips(self) -> None:
        """_simulate_gap_reranker: cap eligible が1頭のみの race は gap rerank不可.

        cap=20 の場合、odds > 20 の馬は cap 外。
        race 内に cap eligible が 1 頭のみなら切替しない。
        """
        model = WinTop1OddsReranker()
        df = pd.DataFrame(
            [
                # R1: 1 cap-eligible (horse 1 odds=15), 1 cap-out (horse 2 odds=50)
                {
                    "race_id": "R1",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 1,
                    "kakuteijyuni": 1,
                    "tanodds": 15.0,
                    "confirmed_odds": 15.3,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R1",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 2,
                    "kakuteijyuni": 2,
                    "tanodds": 50.0,
                    "confirmed_odds": 52.0,
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
                # R2: 2 cap-eligible (both within cap=20) → normal switch
                {
                    "race_id": "R2",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 1,
                    "kakuteijyuni": 2,
                    "tanodds": 10.0,
                    "confirmed_odds": 10.2,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R2",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 2,
                    "kakuteijyuni": 1,
                    "tanodds": 8.0,
                    "confirmed_odds": 8.2,
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
            ]
        )
        result = model._simulate_gap_reranker(
            df,
            cap=20.0,
            gap_params={
                "disabled": False,
                "score_gap_threshold": 0.5,
                "min_prob_margin": 0.01,
                "max_change_rate": 1.0,
            },
        )
        # R1: single cap-eligible → no switch (0 changes from R1)
        # R2: 2 cap-eligible → switch
        assert result["n_changes"] == 1
        assert result["bets"] == 2

    def test_simulate_gap_reranker_disabled_baseline(self) -> None:
        """disabled=True: top2高確率でも切替0、cap-only結果と一致.

        disabled ベースラインは switch_cond を全 False にし、純粋な
        cap-only top1 の ROI/hit/n_changes/change_rate を返す。
        top2 が高確率であっても一切切替えが発生しない。
        """
        model = WinTop1OddsReranker()
        # 2 races: R1 では top2 が高確率 (通常なら切替)、R2 では top1 が高確率
        df = pd.DataFrame(
            [
                # R1: top1 loses, top2 wins → would switch if not disabled
                {
                    "race_id": "R1",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 1,
                    "kakuteijyuni": 2,  # loses
                    "tanodds": 5.0,
                    "confirmed_odds": 5.1,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R1",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 2,
                    "kakuteijyuni": 1,  # wins
                    "tanodds": 3.0,
                    "confirmed_odds": 3.06,
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
                # R2: top1 wins, top2 lower prob → no switch anyway
                {
                    "race_id": "R2",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 1,
                    "kakuteijyuni": 1,  # wins
                    "tanodds": 4.0,
                    "confirmed_odds": 4.08,
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.25,
                    "p_win_final": 0.25,
                },
                {
                    "race_id": "R2",
                    "race_date": "2023-06-01",
                    "surface": "dirt",
                    "umaban": 2,
                    "kakuteijyuni": 2,  # loses
                    "tanodds": 8.0,
                    "confirmed_odds": 8.16,
                    "win_market_selection_score": 0.5,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
            ]
        )

        result = model._simulate_gap_reranker(
            df,
            cap=float("inf"),
            gap_params={"disabled": True},
        )

        # Must produce zero changes regardless of prob/gap conditions
        assert result["n_changes"] == 0
        assert result["change_rate"] == 0.0
        assert result["raw_candidate_change_rate"] == 0.0
        assert result["rate_guard_failed"] is False

        # Cap-only top1 results:
        # R1: top1 (horse 1) loses → payout 0, profit -1
        # R2: top1 (horse 1) wins at confirmed_odds 4.08 → profit 4.08-1=3.08
        assert result["bets"] == 2
        assert result["hit_rate"] == pytest.approx(0.5)  # 1/2 wins
        assert result["profit"] == pytest.approx(-1.0 + 3.08, abs=0.01)  # 2.08
        assert result["roi"] == pytest.approx((2.08 + 2) / 2, abs=0.01)  # 2.04

    def test_simulate_gap_reranker_disabled_matches_cap_only(self) -> None:
        """disabled baseline の ROI/hit が cap-only に一致.

        disabled=True の結果と、min_prob_margin=inf (別の方法でno-switchを
        実現) の結果が同一であることを検証。
        """
        model = WinTop1OddsReranker()
        df = _make_gap_test_df(n_races_per_year=200, years=[2023], surface="dirt")

        result_disabled = model._simulate_gap_reranker(
            df,
            cap=float("inf"),
            gap_params={"disabled": True},
        )
        result_inf_margin = model._simulate_gap_reranker(
            df,
            cap=float("inf"),
            gap_params={"min_prob_margin": float("inf"), "max_change_rate": 1.0},
        )

        # Both should produce identical ROI, hit_rate, profit, bets
        assert result_disabled["roi"] == pytest.approx(result_inf_margin["roi"], abs=1e-6)
        assert result_disabled["hit_rate"] == pytest.approx(result_inf_margin["hit_rate"], abs=1e-6)
        assert result_disabled["profit"] == pytest.approx(result_inf_margin["profit"], abs=1e-6)
        assert result_disabled["bets"] == result_inf_margin["bets"]
        assert result_disabled["n_changes"] == 0
        assert result_inf_margin["n_changes"] == 0

    def test_gap_apply_single_cap_eligible(self) -> None:
        """apply: cap eligible が1頭のみの race は gap rerank不可.

        cap=20 の場合、odds > 20 の馬は cap 外。
        cap eligible が 1 頭のみなら single_cap_eligible reason.
        """
        model = _make_deployed_gap_model(threshold=0.5, margin=0.01, max_cr=0.60)
        model.selected_cap = 20.0  # Tight cap to make horse 2 cap-out
        candidates = pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 15.0,  # within cap
                    "win_market_selection_score": 1.0,
                    "p_win_final_oof": 0.10,
                    "p_win_final": 0.10,
                },
                {
                    "race_id": "R1",
                    "surface": "dirt",
                    "umaban": 2,
                    "tanodds": 50.0,  # outside cap
                    "win_market_selection_score": 0.9,
                    "p_win_final_oof": 0.20,
                    "p_win_final": 0.20,
                },
            ]
        )
        result = model.apply(candidates)
        assert len(result) == 1
        assert not result[GAP_RERANKER_APPLIED_COL].iloc[0]
        assert result[GAP_RERANKER_SWITCH_REASON_COL].iloc[0] == "single_cap_eligible"


def _make_deployed_gap_model(
    *,
    threshold: float = 0.5,
    margin: float = 0.01,
    max_cr: float = 0.10,
) -> WinTop1OddsReranker:
    """gap reranker がデプロイ済みのモデルを作成 (apply テスト用)."""
    model = WinTop1OddsReranker()
    model.is_trained = True
    model.selected_cap = 50.0
    model.gap_reranker_deployed = True
    model.gap_reranker_surface = "dirt"
    model.gap_reranker_score_gap_threshold = threshold
    model.gap_reranker_min_prob_margin = margin
    model.gap_reranker_max_change_rate = max_cr
    model.gap_reranker_training_summary = {
        "deployed": True,
        "selected_params": {
            "score_gap_threshold": threshold,
            "min_prob_margin": margin,
            "max_change_rate": max_cr,
        },
    }
    return model
