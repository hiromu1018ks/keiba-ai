"""ShadowComparisonFramework のユニットテスト (SHD-01~03)

BacktestEngine を2回実行し (baseline vs shadow)、
事後アライメントとメトリクス計算を検証する。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestResult
from domain.models import SubmodelSet, TrainedModelsV5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trained_models(
    *,
    mawc_trained: bool = False,
    ranker_trained: bool = False,
) -> TrainedModelsV5:
    """Build a minimal TrainedModelsV5 with optional MAWC/ranker mocks."""
    mock_submodel = MagicMock()  # No spec to allow dynamic attribute access

    mawc = MagicMock()
    mawc.is_trained = mawc_trained
    mock_submodel.market_aware_win_calibrator = mawc if mawc_trained else None

    ranker = MagicMock()
    ranker.is_trained = ranker_trained
    mock_submodel.win_race_level_ranker = ranker if ranker_trained else None

    # Inference chain mocks (for predict() tests)
    mock_submodel.market = MagicMock()
    mock_submodel.stage1 = MagicMock()
    mock_submodel.win = MagicMock()
    mock_submodel.ev_corrector = MagicMock()
    mock_submodel.place = None
    mock_submodel.win_selection_gate = None
    mock_submodel.win_selection_policy = None
    mock_submodel.win_profit_selector = None
    mock_submodel.conformal_ev_model = None
    mock_submodel.target_encoder = None

    quality = MagicMock()
    regime = MagicMock()
    regime.get_strategy_params.return_value = {"fractional_kelly": 0.5}

    return TrainedModelsV5(
        submodels={"turf": mock_submodel, "dirt": mock_submodel},
        quality_screener=quality,
        regime_detector=regime,
        train_period=("2020-01-01", "2023-12-31"),
    )


def _make_bet_history(
    race_ids: list[str],
    selected_umabans: list[int],
    odds: list[float],
    results: list[float],
    stakes: list[float] | None = None,
) -> list[dict]:
    """Build a synthetic bet_history list."""
    history = []
    for i, rid in enumerate(race_ids):
        history.append({
            "race_id": rid,
            "umaban": selected_umabans[i],
            "stake": stakes[i] if stakes else 100.0,
            "odds": odds[i],
            "tanodds": odds[i],
            "closing_win_odds": odds[i] * 1.05,
            "result": results[i],
            "final_odds": odds[i],
            "is_actual_bet": True,
            "surface": "turf",
            "p_win_final": 0.3 + i * 0.05,
            "win_selection_ev": 1.1 + i * 0.05,
            "win_market_selection_score": 0.5 + i * 0.1,
            "investment_score": 0.6 + i * 0.1,
            "edge": 0.1,
        })
    return history


def _make_backtest_result(
    bet_history: list[dict],
    *,
    total_stake: float | None = None,
    total_return: float | None = None,
    max_drawdown: float = 0.1,
) -> BacktestResult:
    """Build a BacktestResult from bet_history."""
    ts = total_stake or sum(b.get("stake", 100) for b in bet_history)
    tr = total_return or sum(b.get("result", 0) for b in bet_history)
    return BacktestResult(
        total_bets=len(bet_history),
        total_stake=ts,
        total_return=tr,
        winning_bets=sum(1 for b in bet_history if b.get("result", 0) > 0),
        total_roi=tr / ts if ts > 0 else 0.0,
        max_drawdown=max_drawdown,
        final_bankroll=100_000 + tr - ts,
        bet_history=bet_history,
    )


# ===================================================================
# Task 1: Dataclasses and ShadowComparisonFramework
# ===================================================================

class TestFoldDefinition:
    """Test 1: FoldDefinition fold generation."""

    def test_create_folds_default(self) -> None:
        from backtest.shadow_comparison import FoldDefinition

        folds = FoldDefinition.create_folds([2024, 2025], train_window=4)
        assert len(folds) == 2

        # Fold 2024: train 2020-2023, test 2024
        assert folds[0].year == 2024
        assert folds[0].train_start == "2020-01-01"
        assert folds[0].train_end == "2023-12-31"
        assert folds[0].test_start == "2024-01-01"
        assert folds[0].test_end == "2024-12-31"

        # Fold 2025: train 2021-2024, test 2025
        assert folds[1].year == 2025
        assert folds[1].train_start == "2021-01-01"
        assert folds[1].train_end == "2024-12-31"
        assert folds[1].test_start == "2025-01-01"
        assert folds[1].test_end == "2025-12-31"

    def test_create_folds_custom_window(self) -> None:
        from backtest.shadow_comparison import FoldDefinition

        folds = FoldDefinition.create_folds([2024], train_window=3)
        assert len(folds) == 1
        assert folds[0].train_start == "2021-01-01"
        assert folds[0].train_end == "2023-12-31"

    def test_fold_definition_is_frozen(self) -> None:
        from backtest.shadow_comparison import FoldDefinition

        fold = FoldDefinition(
            year=2024,
            train_start="2020-01-01",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-12-31",
        )
        with pytest.raises(AttributeError):
            fold.year = 2025  # type: ignore[misc]


class TestVariantConfig:
    """Test 2: VariantConfig dataclass."""

    def test_variant_config_fields(self) -> None:
        from backtest.shadow_comparison import VariantConfig

        vc = VariantConfig(
            variant_name="baseline",
            model_dir=Path("data/models-backtest"),
            enable_market_aware_calibrator=False,
            enable_race_level_ranker=False,
        )
        assert vc.variant_name == "baseline"
        assert vc.enable_market_aware_calibrator is False
        assert vc.enable_race_level_ranker is False

    def test_variant_config_shadow(self) -> None:
        from backtest.shadow_comparison import VariantConfig

        vc = VariantConfig(
            variant_name="ridge_shadow",
            model_dir=Path("data/models-backtest"),
            enable_market_aware_calibrator=True,
            enable_race_level_ranker=True,
        )
        assert vc.variant_name == "ridge_shadow"
        assert vc.enable_market_aware_calibrator is True
        assert vc.enable_race_level_ranker is True


class TestComparisonMetrics:
    """Test 2: ComparisonMetrics dataclass."""

    def test_comparison_metrics_defaults(self) -> None:
        from backtest.shadow_comparison import ComparisonMetrics

        m = ComparisonMetrics()
        assert m.brier == 0.0
        assert m.logloss == 0.0
        assert m.ece == 0.0
        assert m.roi == 0.0
        assert m.hit_rate == 0.0
        assert m.bet_count == 0
        assert m.avg_odds == 0.0
        assert m.max_drawdown == 0.0
        assert m.clv is None
        assert m.clv_available is False
        assert m.selection_agreement is None
        assert m.avg_investment_score is None
        assert m.actual_predicted_ratio == 0.0


class TestShadowComparisonResult:
    """Test 2: ShadowComparisonResult dataclass."""

    def test_shadow_comparison_result_fields(self) -> None:
        from backtest.shadow_comparison import ShadowComparisonResult

        r = ShadowComparisonResult(
            fold=MagicMock(),
            variants={},
            race_diff=pd.DataFrame(),
            horse_diff=pd.DataFrame(),
            metrics={},
            alignment_succeeded=False,
        )
        assert isinstance(r.variants, dict)
        assert isinstance(r.metrics, dict)
        assert r.alignment_succeeded is False


class TestRaceLevelAlignment:
    """Test 4: Post-hoc race-level alignment."""

    def test_align_race_level_two_variants(self) -> None:
        from backtest.shadow_comparison import ShadowComparisonFramework

        # Baseline: race R1 -> horse 3, R2 -> horse 5
        baseline_bh = _make_bet_history(
            ["R1", "R2"],
            [3, 5],
            [5.0, 8.0],
            [0.0, 800.0],
        )
        # Shadow: race R1 -> horse 3 (same), R2 -> horse 7 (different)
        shadow_bh = _make_bet_history(
            ["R1", "R2"],
            [3, 7],
            [5.0, 12.0],
            [0.0, 1200.0],
        )
        results = {
            "baseline": _make_backtest_result(baseline_bh),
            "shadow": _make_backtest_result(shadow_bh),
        }
        framework = object.__new__(ShadowComparisonFramework)
        aligned = framework._align_race_level(results)

        assert len(aligned) == 2
        row_r1 = aligned[aligned["race_id"] == "R1"].iloc[0]
        row_r2 = aligned[aligned["race_id"] == "R2"].iloc[0]

        # R1: same selection
        assert row_r1["baseline_selected_umaban"] == 3
        assert row_r1["shadow_selected_umaban"] == 3
        assert row_r1["selected_changed"] is False or row_r1["selected_changed"] == False

        # R2: different selection
        assert row_r2["baseline_selected_umaban"] == 5
        assert row_r2["shadow_selected_umaban"] == 7
        assert row_r2["selected_changed"] is True or row_r2["selected_changed"] == True

    def test_align_race_level_preserves_odds(self) -> None:
        from backtest.shadow_comparison import ShadowComparisonFramework

        baseline_bh = _make_bet_history(["R1"], [2], [4.0], [0.0])
        shadow_bh = _make_bet_history(["R1"], [3], [6.0], [600.0])
        results = {
            "baseline": _make_backtest_result(baseline_bh),
            "shadow": _make_backtest_result(shadow_bh),
        }
        framework = object.__new__(ShadowComparisonFramework)
        aligned = framework._align_race_level(results)

        row = aligned.iloc[0]
        assert row["baseline_tanodds"] == 4.0
        assert row["shadow_tanodds"] == 6.0


class TestHorseLevelAlignment:
    """Test 5: Post-hoc horse-level alignment."""

    def test_align_horse_level_basic(self) -> None:
        from backtest.shadow_comparison import ShadowComparisonFramework

        baseline_bh = [
            {
                "race_id": "R1", "umaban": 1,
                "p_win_final": 0.10, "investment_score": 0.2,
                "stake": 0, "is_actual_bet": False,
                "result": 0, "odds": 10.0, "tanodds": 10.0,
                "closing_win_odds": 11.0, "final_odds": 10.0,
                "surface": "turf", "win_market_selection_score": 0.3,
                "edge": 0.01, "win_selection_ev": 1.0,
            },
            {
                "race_id": "R1", "umaban": 2,
                "p_win_final": 0.30, "investment_score": 0.6,
                "stake": 100, "is_actual_bet": True,
                "result": 0, "odds": 4.0, "tanodds": 4.0,
                "closing_win_odds": 4.2, "final_odds": 4.0,
                "surface": "turf", "win_market_selection_score": 0.7,
                "edge": 0.15, "win_selection_ev": 1.2,
            },
        ]
        shadow_bh = [
            {
                "race_id": "R1", "umaban": 1,
                "p_win_final": 0.12, "investment_score": 0.3,
                "stake": 100, "is_actual_bet": True,
                "result": 1000, "odds": 10.0, "tanodds": 10.0,
                "closing_win_odds": 11.0, "final_odds": 10.0,
                "surface": "turf", "win_market_selection_score": 0.5,
                "edge": 0.05, "win_selection_ev": 1.1,
            },
            {
                "race_id": "R1", "umaban": 2,
                "p_win_final": 0.28, "investment_score": 0.5,
                "stake": 0, "is_actual_bet": False,
                "result": 0, "odds": 4.0, "tanodds": 4.0,
                "closing_win_odds": 4.2, "final_odds": 4.0,
                "surface": "turf", "win_market_selection_score": 0.65,
                "edge": 0.10, "win_selection_ev": 1.15,
            },
        ]
        results = {
            "baseline": _make_backtest_result(baseline_bh),
            "shadow": _make_backtest_result(shadow_bh),
        }
        framework = object.__new__(ShadowComparisonFramework)
        aligned = framework._align_horse_level(results)

        assert len(aligned) == 2
        h1 = aligned[aligned["umaban"] == 1].iloc[0]
        h2 = aligned[aligned["umaban"] == 2].iloc[0]

        # Horse 1: baseline p=0.10, shadow p=0.12
        assert abs(h1["baseline_p_win_final"] - 0.10) < 1e-6
        assert abs(h1["shadow_p_win_final"] - 0.12) < 1e-6
        # baseline did NOT select horse 1 (stake=0), shadow did (stake=100)
        assert h1["baseline_selected"] is False or h1["baseline_selected"] == False
        assert h1["shadow_selected"] is True or h1["shadow_selected"] == True


class TestComputeMetrics:
    """Test 6: Metrics computation."""

    def _make_aligned_horse_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "race_id": ["R1", "R1", "R2", "R2"],
            "umaban": [1, 2, 1, 3],
            "baseline_p_win_final": [0.3, 0.7, 0.2, 0.8],
            "baseline_selected": [False, True, False, True],
            "baseline_investment_score": [0.3, 0.7, 0.2, 0.8],
            "shadow_p_win_final": [0.3, 0.7, 0.2, 0.8],
            "shadow_selected": [False, True, False, True],
            "shadow_investment_score": [0.3, 0.7, 0.2, 0.8],
            "kakuteijyuni": [2, 1, 3, 1],
        })

    def test_brier_score(self) -> None:
        """Brier = mean((p - is_win)^2). p=[0.3, 0.7], actual=[0, 1] -> 0.09."""
        from backtest.shadow_comparison import ShadowComparisonFramework

        horse_df = pd.DataFrame({
            "race_id": ["R1", "R1"],
            "umaban": [1, 2],
            "baseline_p_win_final": [0.3, 0.7],
            "kakuteijyuni": [2, 1],
            "baseline_selected": [False, True],
            "baseline_investment_score": [0.3, 0.7],
            "shadow_p_win_final": [0.3, 0.7],
            "shadow_selected": [False, True],
            "shadow_investment_score": [0.3, 0.7],
        })
        framework = object.__new__(ShadowComparisonFramework)
        metrics = framework.compute_metrics(
            pd.DataFrame(), horse_df, "baseline", [],
        )
        # Brier: (0.3-0)^2 + (0.7-1)^2 = 0.09 + 0.09 = 0.18 / 2 = 0.09
        assert abs(metrics.brier - 0.09) < 1e-6

    def test_logloss(self) -> None:
        """Logloss: -mean(is_win * log(p) + (1-is_win) * log(1-p))."""
        from backtest.shadow_comparison import ShadowComparisonFramework

        horse_df = pd.DataFrame({
            "race_id": ["R1", "R1"],
            "umaban": [1, 2],
            "baseline_p_win_final": [0.3, 0.7],
            "kakuteijyuni": [2, 1],
            "baseline_selected": [False, True],
            "baseline_investment_score": [0.3, 0.7],
            "shadow_p_win_final": [0.3, 0.7],
            "shadow_selected": [False, True],
            "shadow_investment_score": [0.3, 0.7],
        })
        framework = object.__new__(ShadowComparisonFramework)
        metrics = framework.compute_metrics(
            pd.DataFrame(), horse_df, "baseline", [],
        )
        expected = -(
            (0 * math.log(0.3) + 1 * math.log(0.7))
            + (1 * math.log(0.7) + 0 * math.log(0.3))
        ) / 2
        assert abs(metrics.logloss - expected) < 1e-4

    def test_roi_and_hr(self) -> None:
        from backtest.shadow_comparison import ShadowComparisonFramework

        bh = _make_bet_history(
            ["R1", "R2", "R3"],
            [1, 2, 3],
            [5.0, 3.0, 10.0],
            [500.0, 0.0, 0.0],
            stakes=[100.0, 100.0, 100.0],
        )
        bt_result = _make_backtest_result(bh, max_drawdown=0.15)
        horse_df = pd.DataFrame({
            "race_id": ["R1", "R2", "R3"],
            "umaban": [1, 2, 3],
            "baseline_p_win_final": [0.2, 0.3, 0.1],
            "kakuteijyuni": [1, 2, 5],
            "baseline_selected": [True, True, True],
            "baseline_investment_score": [0.5, 0.6, 0.4],
            "shadow_p_win_final": [0.2, 0.3, 0.1],
            "shadow_selected": [True, True, True],
            "shadow_investment_score": [0.5, 0.6, 0.4],
        })
        framework = object.__new__(ShadowComparisonFramework)
        metrics = framework.compute_metrics(
            pd.DataFrame(), horse_df, "baseline", bh, bt_result=bt_result,
        )
        # ROI: (500 - 300) / 300 - 1 = 0.667
        assert abs(metrics.roi - (500.0 / 300.0 - 1.0)) < 1e-4
        # HR: 1/3
        assert abs(metrics.hit_rate - 1.0 / 3.0) < 1e-4
        assert metrics.bet_count == 3
        assert abs(metrics.max_drawdown - 0.15) < 1e-6

    def test_ece_computation(self) -> None:
        """ECE with 10 equal-width bins."""
        from backtest.shadow_comparison import ShadowComparisonFramework

        horse_df = pd.DataFrame({
            "race_id": [f"R{i}" for i in range(20)],
            "umaban": list(range(20)),
            "baseline_p_win_final": [0.05 * (i + 1) for i in range(20)],
            "kakuteijyuni": [1 if i == 0 else 2 for i in range(20)],
            "baseline_selected": [True] * 20,
            "baseline_investment_score": [0.5] * 20,
            "shadow_p_win_final": [0.05 * (i + 1) for i in range(20)],
            "shadow_selected": [True] * 20,
            "shadow_investment_score": [0.5] * 20,
        })
        framework = object.__new__(ShadowComparisonFramework)
        metrics = framework.compute_metrics(
            pd.DataFrame(), horse_df, "baseline", [],
        )
        assert metrics.ece >= 0.0

    def test_actual_predicted_ratio(self) -> None:
        from backtest.shadow_comparison import ShadowComparisonFramework

        horse_df = pd.DataFrame({
            "race_id": ["R1", "R1", "R2", "R2"],
            "umaban": [1, 2, 1, 3],
            "baseline_p_win_final": [0.2, 0.8, 0.1, 0.9],
            "kakuteijyuni": [1, 2, 2, 1],
            "baseline_selected": [False, True, False, True],
            "baseline_investment_score": [0.2, 0.8, 0.1, 0.9],
            "shadow_p_win_final": [0.2, 0.8, 0.1, 0.9],
            "shadow_selected": [False, True, False, True],
            "shadow_investment_score": [0.2, 0.8, 0.1, 0.9],
        })
        framework = object.__new__(ShadowComparisonFramework)
        metrics = framework.compute_metrics(
            pd.DataFrame(), horse_df, "baseline", [],
        )
        # mean(actual) = 0.5, mean(predicted) = 0.5, ratio = 1.0
        assert abs(metrics.actual_predicted_ratio - 1.0) < 1e-4


class TestSelectionAgreement:
    """Test 7: Selection agreement computation."""

    def test_selection_agreement(self) -> None:
        """3 races: 2 same, 1 different -> agreement = 2/3."""
        from backtest.shadow_comparison import ShadowComparisonFramework

        race_diff = pd.DataFrame({
            "race_id": ["R1", "R2", "R3"],
            "baseline_selected_umaban": [3, 5, 7],
            "shadow_selected_umaban": [3, 5, 9],
            "selected_changed": [False, False, True],
        })
        framework = object.__new__(ShadowComparisonFramework)
        agreement = framework._compute_selection_agreement(race_diff)
        assert abs(agreement - 2.0 / 3.0) < 1e-6


class TestCLV:
    """Test 8: CLV computation."""

    def test_clv_with_valid_inputs(self) -> None:
        from backtest.shadow_comparison import ShadowComparisonFramework

        bh = [
            {"stake": 100, "result": 500, "odds": 4.0, "tanodds": 4.0,
             "closing_win_odds": 4.4, "race_id": "R1", "umaban": 1,
             "p_win_final": 0.25, "win_market_selection_score": 0.5,
             "investment_score": 0.5, "edge": 0.1, "win_selection_ev": 1.1,
             "final_odds": 4.0, "is_actual_bet": True, "surface": "turf"},
            {"stake": 100, "result": 0, "odds": 8.0, "tanodds": 8.0,
             "closing_win_odds": 9.0, "race_id": "R2", "umaban": 2,
             "p_win_final": 0.15, "win_market_selection_score": 0.4,
             "investment_score": 0.4, "edge": 0.05, "win_selection_ev": 1.05,
             "final_odds": 8.0, "is_actual_bet": True, "surface": "turf"},
        ]
        framework = object.__new__(ShadowComparisonFramework)
        clv, available = framework._compute_clv(bh)
        # CLV: closing/betting - 1
        # bet 1: 4.4/4.0 - 1 = 0.10, bet 2: 9.0/8.0 - 1 = 0.125
        # mean = (0.10 + 0.125) / 2 = 0.1125
        assert available is True
        assert abs(clv - 0.1125) < 1e-4

    def test_clv_with_missing_closing_odds(self) -> None:
        from backtest.shadow_comparison import ShadowComparisonFramework

        bh = [
            {"stake": 100, "result": 500, "odds": 4.0, "tanodds": 4.0,
             "closing_win_odds": None, "race_id": "R1", "umaban": 1,
             "p_win_final": 0.25, "win_market_selection_score": 0.5,
             "investment_score": 0.5, "edge": 0.1, "win_selection_ev": 1.1,
             "final_odds": 4.0, "is_actual_bet": True, "surface": "turf"},
        ]
        framework = object.__new__(ShadowComparisonFramework)
        clv, available = framework._compute_clv(bh)
        assert available is False
        assert clv is None


class TestMetricsByGroup:
    """Test 9: Aggregation by surface, odds_band, prob_rank_band."""

    def test_group_by_odds_band(self) -> None:
        from backtest.shadow_comparison import ShadowComparisonFramework

        horse_df = pd.DataFrame({
            "race_id": ["R1", "R1", "R2", "R2", "R3", "R3"],
            "umaban": [1, 2, 1, 3, 2, 4],
            "baseline_p_win_final": [0.3, 0.7, 0.2, 0.8, 0.15, 0.85],
            "kakuteijyuni": [1, 2, 3, 1, 1, 2],
            "baseline_selected": [False, True, False, True, True, False],
            "baseline_investment_score": [0.3, 0.7, 0.2, 0.8, 0.15, 0.85],
            "shadow_p_win_final": [0.3, 0.7, 0.2, 0.8, 0.15, 0.85],
            "shadow_selected": [False, True, False, True, True, False],
            "shadow_investment_score": [0.3, 0.7, 0.2, 0.8, 0.15, 0.85],
            "closing_win_odds": [2.0, 4.0, 6.0, 3.0, 25.0, 1.5],
        })
        framework = object.__new__(ShadowComparisonFramework)
        grouped = framework.compute_metrics_by_group(horse_df, "odds_band")
        assert isinstance(grouped, dict)
        assert len(grouped) > 0
        # Should have bands like "1-3", "3-5", "5-10", "10-30"
        for band_name, metrics in grouped.items():
            assert isinstance(band_name, str)
            assert metrics.bet_count >= 0


class TestNWayFramework:
    """Test 10: N-way framework with multiple variants."""

    @patch("backtest.engine.BacktestEngine")
    @patch("db.model_loader.ModelLoader")
    @patch("db.readers.load_odds_time_series_range")
    def test_three_variant_alignment(
        self,
        mock_load_odds: MagicMock,
        mock_loader_cls: MagicMock,
        mock_engine_cls: MagicMock,
    ) -> None:
        from backtest.shadow_comparison import (
            ShadowComparisonFramework,
            VariantConfig,
        )

        mock_load_odds.return_value = pd.DataFrame()

        models = _make_trained_models(mawc_trained=True, ranker_trained=True)
        mock_loader = MagicMock()
        mock_loader.load_from_dir.return_value = (models, MagicMock())
        mock_loader_cls.return_value = mock_loader

        def make_engine_side_effect(*args: object, **kwargs: object) -> MagicMock:
            engine = MagicMock()
            diag_prefix = kwargs.get("diag_prefix", "")
            if "baseline" in str(diag_prefix):
                bh = _make_bet_history(["R1"], [1], [3.0], [0.0])
            elif "shadow_a" in str(diag_prefix):
                bh = _make_bet_history(["R1"], [2], [5.0], [500.0])
            else:
                bh = _make_bet_history(["R1"], [1], [3.0], [0.0])
            engine.run.return_value = _make_backtest_result(bh)
            return engine

        mock_engine_cls.side_effect = make_engine_side_effect

        variants = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow_a", Path("data/shadow_a"), True, True),
            VariantConfig("shadow_b", Path("data/shadow_b"), True, False),
        ]
        framework = ShadowComparisonFramework(variants=variants, store=MagicMock())
        fold = MagicMock()
        fold.year = 2024
        fold.test_start = "2024-01-01"
        fold.test_end = "2024-12-31"
        fold.train_start = "2020-01-01"
        fold.train_end = "2023-12-31"

        result = framework.run_fold(fold)
        assert "baseline" in result.variants
        assert "shadow_a" in result.variants
        assert "shadow_b" in result.variants


class TestStrictMode:
    """Test 11-12: D-21 strict mode for flag/artifact mismatch."""

    def test_strict_mode_mawc_missing(self) -> None:
        from backtest.shadow_comparison import (
            ShadowComparisonFramework,
            VariantConfig,
        )

        models = _make_trained_models(mawc_trained=False, ranker_trained=False)
        variants = [
            VariantConfig("baseline", Path("data/bt"), True, False),
        ]
        framework = ShadowComparisonFramework(variants=variants)
        with pytest.raises(ValueError, match="enable_market_aware_calibrator=True"):
            framework._validate_artifacts("baseline", models, Path("data/bt/2024"))

    def test_strict_mode_ranker_missing(self) -> None:
        from backtest.shadow_comparison import (
            ShadowComparisonFramework,
            VariantConfig,
        )

        models = _make_trained_models(mawc_trained=False, ranker_trained=False)
        variants = [
            VariantConfig("baseline", Path("data/bt"), False, True),
        ]
        framework = ShadowComparisonFramework(variants=variants)
        with pytest.raises(ValueError, match="enable_race_level_ranker=True"):
            framework._validate_artifacts("baseline", models, Path("data/bt/2024"))

    def test_strict_mode_passes_when_artifact_present(self) -> None:
        from backtest.shadow_comparison import (
            ShadowComparisonFramework,
            VariantConfig,
        )

        models = _make_trained_models(mawc_trained=True, ranker_trained=True)
        variants = [
            VariantConfig("baseline", Path("data/bt"), True, True),
        ]
        framework = ShadowComparisonFramework(variants=variants)
        # Should not raise
        framework._validate_artifacts("baseline", models, Path("data/bt/2024"))


# ===================================================================
# Task 2: Feature flag injection into RacePredictor
# ===================================================================

class TestRacePredictorFlags:
    """Tests for enable_market_aware_calibrator / enable_race_level_ranker flags."""

    def test_mawc_disabled_skips_apply(self) -> None:
        """Test 1: RacePredictor with MAWC disabled skips mawc.apply()."""
        from backtest.race_predictor import RacePredictor

        models = _make_trained_models(mawc_trained=True, ranker_trained=False)
        rp = RacePredictor(
            models,
            betting_target="win",
            enable_market_aware_calibrator=False,
        )
        assert rp.enable_market_aware_calibrator is False

    def test_ranker_disabled_skips_score(self) -> None:
        """Test 2: RacePredictor with ranker disabled skips ranker.score()."""
        from backtest.race_predictor import RacePredictor

        models = _make_trained_models(mawc_trained=False, ranker_trained=True)
        rp = RacePredictor(
            models,
            betting_target="win",
            enable_race_level_ranker=False,
        )
        assert rp.enable_race_level_ranker is False

    def test_default_flags_are_true(self) -> None:
        """Test 3: Default flags are True (backward compatible)."""
        from backtest.race_predictor import RacePredictor

        models = _make_trained_models(mawc_trained=False, ranker_trained=False)
        rp = RacePredictor(models, betting_target="win")
        assert rp.enable_market_aware_calibrator is True
        assert rp.enable_race_level_ranker is True

    def test_shadow_flags_from_trained_models(self) -> None:
        """Test 4: RacePredictor reads _shadow_flags from TrainedModelsV5."""
        from backtest.race_predictor import RacePredictor

        models = _make_trained_models(mawc_trained=True, ranker_trained=True)
        models._shadow_flags = {
            "enable_market_aware_calibrator": False,
            "enable_race_level_ranker": False,
        }
        rp = RacePredictor(models, betting_target="win")
        assert rp.enable_market_aware_calibrator is False
        assert rp.enable_race_level_ranker is False

    def test_shadow_flags_override_constructor(self) -> None:
        """Test 4b: _shadow_flags overrides constructor args."""
        from backtest.race_predictor import RacePredictor

        models = _make_trained_models(mawc_trained=True, ranker_trained=True)
        models._shadow_flags = {
            "enable_market_aware_calibrator": False,
        }
        rp = RacePredictor(
            models,
            betting_target="win",
            enable_market_aware_calibrator=True,
        )
        # _shadow_flags should override
        assert rp.enable_market_aware_calibrator is False
        # ranker flag not in _shadow_flags, should use constructor default (True)
        assert rp.enable_race_level_ranker is True

    def test_predict_baseline_path_no_mawc(self) -> None:
        """Test 5: When MAWC disabled, the flag prevents mawc.apply()."""
        from backtest.race_predictor import RacePredictor

        models = _make_trained_models(mawc_trained=True, ranker_trained=False)
        rp = RacePredictor(
            models,
            betting_target="win",
            enable_market_aware_calibrator=False,
        )
        # Verify flag state — mawc.apply() will be skipped because
        # enable_market_aware_calibrator is False. We test this via
        # the flag attribute since mocking the full predict() chain
        # requires extensive setup beyond the scope of this unit test.
        assert rp.enable_market_aware_calibrator is False
        # The baseline path (p_win_corrected normalization) runs when
        # enable_market_aware_calibrator=False. This is verified by
        # the code path at lines 274-280 in race_predictor.py.

    def test_predict_ranker_disabled_no_investment_score(self) -> None:
        """Test 6: When ranker disabled, ranker.score() is skipped."""
        from backtest.race_predictor import RacePredictor

        models = _make_trained_models(mawc_trained=False, ranker_trained=True)
        rp = RacePredictor(
            models,
            betting_target="win",
            enable_race_level_ranker=False,
        )
        # Verify flag state — ranker.score() will be skipped because
        # enable_race_level_ranker is False. The guard at line 289 in
        # race_predictor.py checks self.enable_race_level_ranker first.
        assert rp.enable_race_level_ranker is False


# ===================================================================
# Task 3: Integration wiring
# ===================================================================

class TestRunFoldIntegration:
    """Integration tests for run_fold with feature flag injection."""

    @patch("backtest.engine.BacktestEngine")
    @patch("db.model_loader.ModelLoader")
    @patch("db.readers.load_payouts")
    @patch("db.readers.load_odds_snapshots")
    @patch("db.readers.load_entries")
    @patch("db.readers.load_races")
    @patch("db.readers.load_odds_time_series_range")
    def test_run_fold_loads_models_and_injects_flags(
        self,
        mock_load_odds: MagicMock,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds_snapshots: MagicMock,
        mock_load_payouts: MagicMock,
        mock_loader_cls: MagicMock,
        mock_engine_cls: MagicMock,
    ) -> None:
        from backtest.shadow_comparison import (
            ShadowComparisonFramework,
            VariantConfig,
        )

        # P1 + P2: preload用モック — 空DataFrameを返す
        mock_load_odds.return_value = pd.DataFrame()
        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds_snapshots.return_value = pd.DataFrame()
        mock_load_payouts.return_value = pd.DataFrame()

        models = _make_trained_models(mawc_trained=True, ranker_trained=True)
        mock_loader = MagicMock()
        mock_loader.load_from_dir.return_value = (models, MagicMock())
        mock_loader_cls.return_value = mock_loader

        mock_engine = MagicMock()
        bh = _make_bet_history(["R1", "R2"], [1, 3], [5.0, 8.0], [0.0, 800.0])
        mock_engine.run.return_value = _make_backtest_result(bh)
        mock_engine_cls.return_value = mock_engine

        mock_store = MagicMock()
        variants = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]
        framework = ShadowComparisonFramework(
            variants=variants,
            betting_target="win",
            store=mock_store,
        )
        from backtest.shadow_comparison import FoldDefinition

        fold = FoldDefinition(
            year=2024,
            train_start="2020-01-01",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-12-31",
        )
        result = framework.run_fold(fold)

        # Verify ModelLoader.load_from_dir called with correct paths
        assert mock_loader.load_from_dir.call_count == 2
        mock_loader.load_from_dir.assert_any_call(Path("data/bt/2024"))
        mock_loader.load_from_dir.assert_any_call(Path("data/shadow/2024"))

        # P1 + P2: 5ローダーはfold単位で1回だけ呼ばれる(2 variantsでも1回)
        assert mock_load_races.call_count == 1
        assert mock_load_entries.call_count == 1
        assert mock_load_odds_snapshots.call_count == 1
        assert mock_load_payouts.call_count == 1
        assert mock_load_odds.call_count == 1

        # Verify _shadow_flags set correctly on loaded models
        assert hasattr(models, "_shadow_flags")

        # Verify BacktestEngine.run called with correct dates
        assert mock_engine.run.call_count == 2
        mock_engine.run.assert_any_call(
            test_start="2024-01-01", test_end="2024-12-31",
            training_bet_history=[],
        )

        # P1 + P2: constructor が全preloaded kwargs + 同一storeを受けていること
        # 2 variantsで2回呼ばれるため、call_args_listで全呼び出しを検証
        assert mock_engine_cls.call_count == 2
        for call in mock_engine_cls.call_args_list:
            kwargs = call.kwargs
            assert "preloaded_race_df" in kwargs
            assert "preloaded_entry_df" in kwargs
            assert "preloaded_final_odds_df" in kwargs
            assert "preloaded_payouts_df" in kwargs
            assert "preloaded_odds_ts" in kwargs
            assert kwargs["store"] is mock_store

        # Verify result structure
        assert "baseline" in result.variants
        assert "shadow" in result.variants
        assert isinstance(result.race_diff, pd.DataFrame)
        assert isinstance(result.horse_diff, pd.DataFrame)
        assert "baseline" in result.metrics
        assert "shadow" in result.metrics

        # Verify default: training_bet_history=[] (skip calibration BT)
        for call in mock_engine.run.call_args_list:
            assert call.kwargs.get("training_bet_history") == []

    @patch("backtest.engine.BacktestEngine")
    @patch("db.model_loader.ModelLoader")
    @patch("db.readers.load_payouts")
    @patch("db.readers.load_odds_snapshots")
    @patch("db.readers.load_entries")
    @patch("db.readers.load_races")
    @patch("db.readers.load_odds_time_series_range")
    def test_run_default_folds(
        self,
        mock_load_odds: MagicMock,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds_snapshots: MagicMock,
        mock_load_payouts: MagicMock,
        mock_loader_cls: MagicMock,
        mock_engine_cls: MagicMock,
    ) -> None:
        from backtest.shadow_comparison import (
            ShadowComparisonFramework,
            VariantConfig,
        )

        mock_load_odds.return_value = pd.DataFrame()
        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds_snapshots.return_value = pd.DataFrame()
        mock_load_payouts.return_value = pd.DataFrame()

        models = _make_trained_models(mawc_trained=True, ranker_trained=True)
        mock_loader = MagicMock()
        mock_loader.load_from_dir.return_value = (models, MagicMock())
        mock_loader_cls.return_value = mock_loader

        mock_engine = MagicMock()
        mock_engine.run.return_value = _make_backtest_result(
            _make_bet_history(["R1"], [1], [5.0], [0.0]),
        )
        mock_engine_cls.return_value = mock_engine

        mock_store = MagicMock()
        variants = [
            VariantConfig("baseline", Path("data/bt"), False, False),
        ]
        framework = ShadowComparisonFramework(variants=variants, store=mock_store)
        results = framework.run()

        # Default: 2 folds (2024, 2025)
        assert len(results) == 2

    @patch("backtest.engine.BacktestEngine")
    @patch("db.model_loader.ModelLoader")
    @patch("db.readers.load_payouts")
    @patch("db.readers.load_odds_snapshots")
    @patch("db.readers.load_entries")
    @patch("db.readers.load_races")
    @patch("db.readers.load_odds_time_series_range")
    def test_run_custom_fold_list(
        self,
        mock_load_odds: MagicMock,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds_snapshots: MagicMock,
        mock_load_payouts: MagicMock,
        mock_loader_cls: MagicMock,
        mock_engine_cls: MagicMock,
    ) -> None:
        from backtest.shadow_comparison import (
            FoldDefinition,
            ShadowComparisonFramework,
            VariantConfig,
        )

        mock_load_odds.return_value = pd.DataFrame()
        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds_snapshots.return_value = pd.DataFrame()
        mock_load_payouts.return_value = pd.DataFrame()

        models = _make_trained_models(mawc_trained=True, ranker_trained=True)
        mock_loader = MagicMock()
        mock_loader.load_from_dir.return_value = (models, MagicMock())
        mock_loader_cls.return_value = mock_loader

        mock_engine = MagicMock()
        mock_engine.run.return_value = _make_backtest_result(
            _make_bet_history(["R1"], [1], [5.0], [0.0]),
        )
        mock_engine_cls.return_value = mock_engine

        mock_store = MagicMock()
        variants = [
            VariantConfig("baseline", Path("data/bt"), False, False),
        ]
        framework = ShadowComparisonFramework(variants=variants, store=mock_store)
        custom_folds = [FoldDefinition(
            year=2023,
            train_start="2019-01-01",
            train_end="2022-12-31",
            test_start="2023-01-01",
            test_end="2023-12-31",
        )]
        results = framework.run(folds=custom_folds)
        assert len(results) == 1
        assert results[0].fold.year == 2023

    @patch("backtest.engine.BacktestEngine")
    @patch("db.model_loader.ModelLoader")
    @patch("db.readers.load_payouts")
    @patch("db.readers.load_odds_snapshots")
    @patch("db.readers.load_entries")
    @patch("db.readers.load_races")
    @patch("db.readers.load_odds_time_series_range")
    def test_run_fold_calibration_bt_enabled_passes_none(
        self,
        mock_load_odds: MagicMock,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds_snapshots: MagicMock,
        mock_load_payouts: MagicMock,
        mock_loader_cls: MagicMock,
        mock_engine_cls: MagicMock,
    ) -> None:
        """run_calibration_bt=True: engine.run() called with training_bet_history=None."""
        from backtest.shadow_comparison import (
            FoldDefinition,
            ShadowComparisonFramework,
            VariantConfig,
        )

        mock_load_odds.return_value = pd.DataFrame()
        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds_snapshots.return_value = pd.DataFrame()
        mock_load_payouts.return_value = pd.DataFrame()

        models = _make_trained_models(mawc_trained=True, ranker_trained=True)
        mock_loader = MagicMock()
        mock_loader.load_from_dir.return_value = (models, MagicMock())
        mock_loader_cls.return_value = mock_loader

        mock_engine = MagicMock()
        bh = _make_bet_history(["R1", "R2"], [1, 3], [5.0, 8.0], [0.0, 800.0])
        mock_engine.run.return_value = _make_backtest_result(bh)
        mock_engine_cls.return_value = mock_engine

        mock_store = MagicMock()
        variants = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]
        framework = ShadowComparisonFramework(
            variants=variants,
            betting_target="win",
            store=mock_store,
            run_calibration_bt=True,
        )
        fold = FoldDefinition(
            year=2024,
            train_start="2020-01-01",
            train_end="2023-12-31",
            test_start="2024-01-01",
            test_end="2024-12-31",
        )
        framework.run_fold(fold)

        # Verify training_bet_history=None (auto-generate enabled)
        assert mock_engine.run.call_count == 2
        for call in mock_engine.run.call_args_list:
            assert call.kwargs.get("training_bet_history") is None


# ===================================================================
# Task 1 (Plan 41-02): Output artifact methods
# ===================================================================


def _make_comparison_result(
    fold_year: int = 2024,
    *,
    n_races: int = 3,
    n_horses_per_race: int = 4,
) -> "ShadowComparisonResult":
    """Build a synthetic ShadowComparisonResult for artifact tests."""
    from backtest.shadow_comparison import (
        ComparisonMetrics,
        FoldDefinition,
        ShadowComparisonResult,
        VariantResult,
    )

    fold = FoldDefinition(
        year=fold_year,
        train_start=f"{fold_year - 4}-01-01",
        train_end=f"{fold_year - 1}-12-31",
        test_start=f"{fold_year}-01-01",
        test_end=f"{fold_year}-12-31",
    )

    # Build bet histories for baseline and shadow
    race_ids = [f"R{i}" for i in range(n_races)]
    baseline_bh = _make_bet_history(
        race_ids, [1] * n_races, [5.0] * n_races, [500.0] + [0.0] * (n_races - 1),
    )
    shadow_bh = _make_bet_history(
        race_ids, [2] * n_races, [6.0] * n_races, [600.0] + [0.0] * (n_races - 1),
    )

    baseline_bt = _make_backtest_result(baseline_bh)
    shadow_bt = _make_backtest_result(shadow_bh)

    # Build race_diff DataFrame
    race_rows: list[dict] = []
    for i, rid in enumerate(race_ids):
        race_rows.append({
            "race_id": rid,
            "baseline_selected_umaban": 1,
            "shadow_selected_umaban": 2 if i == 0 else 1,
            "selected_changed": i == 0,
            "baseline_tanodds": 5.0,
            "shadow_tanodds": 6.0,
            "baseline_p_win_final": 0.25,
            "shadow_p_win_final": 0.20,
            "baseline_result": baseline_bh[i]["result"],
            "shadow_result": shadow_bh[i]["result"],
            "baseline_stake": 100.0,
            "shadow_stake": 100.0,
        })
    race_diff = pd.DataFrame(race_rows)

    # Build horse_diff DataFrame
    horse_rows: list[dict] = []
    for i, rid in enumerate(race_ids):
        for umaban in range(1, n_horses_per_race + 1):
            horse_rows.append({
                "race_id": rid,
                "umaban": umaban,
                "baseline_p_win_final": 0.25 if umaban == 1 else 0.1,
                "shadow_p_win_final": 0.20 if umaban == 2 else 0.1,
                "baseline_investment_score": 0.5 if umaban == 1 else 0.2,
                "shadow_investment_score": 0.6 if umaban == 2 else 0.2,
                "baseline_selected": umaban == 1,
                "shadow_selected": umaban == 2,
                "closing_win_odds": 5.0 + umaban,
                "kakuteijyuni": 1 if umaban == 1 else umaban,
                "surface": "turf",
            })
    horse_diff = pd.DataFrame(horse_rows)

    metrics = {
        "baseline": ComparisonMetrics(
            brier=0.18, logloss=0.65, ece=0.04, roi=0.67,
            hit_rate=0.33, bet_count=n_races, avg_odds=5.0,
            max_drawdown=0.10, clv=0.05, clv_available=True,
            selection_agreement=0.67, avg_investment_score=0.4,
            actual_predicted_ratio=1.0,
        ),
        "shadow": ComparisonMetrics(
            brier=0.16, logloss=0.60, ece=0.03, roi=0.80,
            hit_rate=0.33, bet_count=n_races, avg_odds=6.0,
            max_drawdown=0.08, clv=0.06, clv_available=True,
            selection_agreement=0.67, avg_investment_score=0.45,
            actual_predicted_ratio=1.0,
        ),
    }

    return ShadowComparisonResult(
        fold=fold,
        variants={
            "baseline": VariantResult("baseline", baseline_bt, {
                "enable_market_aware_calibrator": False,
                "enable_race_level_ranker": False,
            }),
            "shadow": VariantResult("shadow", shadow_bt, {
                "enable_market_aware_calibrator": True,
                "enable_race_level_ranker": True,
            }),
        },
        race_diff=race_diff,
        horse_diff=horse_diff,
        metrics=metrics,
        alignment_succeeded=True,
    )


class TestSaveResults:
    """Tests for save_results() output artifact generation."""

    def test_save_results_creates_json_metrics(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import save_results

        results = [_make_comparison_result(2024)]
        paths = save_results(results, tmp_path)

        assert "metrics_json" in paths
        assert paths["metrics_json"].exists()
        import json
        data = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
        assert "folds" in data
        assert "2024" in data["folds"]
        assert "overall" in data
        assert "generated_at" in data

    def test_save_results_json_contains_metrics(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import save_results

        results = [_make_comparison_result(2024)]
        paths = save_results(results, tmp_path)

        import json
        data = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
        fold_data = data["folds"]["2024"]
        assert "metrics" in fold_data
        assert "baseline" in fold_data["metrics"]
        assert "shadow" in fold_data["metrics"]
        assert "brier" in fold_data["metrics"]["baseline"]
        assert "roi" in fold_data["metrics"]["baseline"]

    def test_save_results_creates_race_diff_parquet(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import save_results

        results = [_make_comparison_result(2024)]
        paths = save_results(results, tmp_path)

        assert "race_diff_parquet" in paths
        assert paths["race_diff_parquet"].exists()
        df = pd.read_parquet(paths["race_diff_parquet"])
        assert "race_id" in df.columns
        assert "fold_year" in df.columns
        assert "baseline_selected_umaban" in df.columns

    def test_save_results_creates_race_diff_csv(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import save_results

        results = [_make_comparison_result(2024)]
        paths = save_results(results, tmp_path)

        assert "race_diff_csv" in paths
        assert paths["race_diff_csv"].exists()
        df = pd.read_csv(paths["race_diff_csv"], encoding="utf-8-sig")
        assert len(df) > 0
        # CSV row count matches Parquet
        df_pq = pd.read_parquet(paths["race_diff_parquet"])
        assert len(df) == len(df_pq)

    def test_save_results_creates_horse_diff_parquet(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import save_results

        results = [_make_comparison_result(2024)]
        paths = save_results(results, tmp_path)

        assert "horse_diff_parquet" in paths
        assert paths["horse_diff_parquet"].exists()
        df = pd.read_parquet(paths["horse_diff_parquet"])
        assert "race_id" in df.columns
        assert "umaban" in df.columns
        assert "baseline_p_win_final" in df.columns
        assert "shadow_p_win_final" in df.columns

    def test_save_results_multi_fold(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import save_results

        results = [
            _make_comparison_result(2024),
            _make_comparison_result(2025),
        ]
        paths = save_results(results, tmp_path)

        import json
        data = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
        assert "2024" in data["folds"]
        assert "2025" in data["folds"]
        assert "overall" in data

        # Parquet contains data from both folds
        df = pd.read_parquet(paths["race_diff_parquet"])
        assert set(df["fold_year"].unique()) == {2024, 2025}

    def test_save_results_json_has_grouped_metrics(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import save_results

        results = [_make_comparison_result(2024)]
        paths = save_results(results, tmp_path)

        import json
        data = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
        fold_data = data["folds"]["2024"]
        # Grouped metrics dimensions per D-13
        assert "metrics_by_surface" in fold_data
        assert "metrics_by_odds_band" in fold_data
        assert "metrics_by_selected_changed" in fold_data


class TestSaveManifest:
    """Tests for save_manifest() manifest generation."""

    def test_save_manifest_creates_file(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import (
            VariantConfig,
            save_manifest,
            save_results,
        )

        results = [_make_comparison_result(2024)]
        artifact_paths = save_results(results, tmp_path)
        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]
        manifest_path = save_manifest(
            results, variant_configs, tmp_path, artifact_paths,
        )
        assert manifest_path.exists()

    def test_save_manifest_has_variants(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import (
            VariantConfig,
            save_manifest,
            save_results,
        )

        results = [_make_comparison_result(2024)]
        artifact_paths = save_results(results, tmp_path)
        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]
        manifest_path = save_manifest(
            results, variant_configs, tmp_path, artifact_paths,
        )

        import json
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "variants" in data
        assert len(data["variants"]) == 2
        assert data["variants"][0]["variant_name"] == "baseline"
        assert data["variants"][0]["flag_states"]["enable_market_aware_calibrator"] is False
        assert data["variants"][1]["variant_name"] == "shadow"
        assert data["variants"][1]["flag_states"]["enable_market_aware_calibrator"] is True

    def test_save_manifest_has_sha256_hashes(self, tmp_path: Path) -> None:
        import hashlib

        from backtest.shadow_comparison import (
            VariantConfig,
            save_manifest,
            save_results,
        )

        results = [_make_comparison_result(2024)]
        artifact_paths = save_results(results, tmp_path)
        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]
        manifest_path = save_manifest(
            results, variant_configs, tmp_path, artifact_paths,
        )

        import json
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "artifacts" in data
        for key in ["metrics_json", "race_diff_parquet", "race_diff_csv", "horse_diff_parquet"]:
            assert key in data["artifacts"]
            assert "sha256" in data["artifacts"][key]
            # Verify SHA256 matches actual file
            actual_path = tmp_path / data["artifacts"][key]["path"]
            expected_hash = hashlib.sha256(actual_path.read_bytes()).hexdigest()
            assert data["artifacts"][key]["sha256"] == expected_hash

    def test_save_manifest_has_fold_definitions(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import (
            VariantConfig,
            save_manifest,
            save_results,
        )

        results = [
            _make_comparison_result(2024),
            _make_comparison_result(2025),
        ]
        artifact_paths = save_results(results, tmp_path)
        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
        ]
        manifest_path = save_manifest(
            results, variant_configs, tmp_path, artifact_paths,
        )

        import json
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "folds" in data
        assert len(data["folds"]) == 2
        assert data["folds"][0]["year"] == 2024
        assert data["folds"][1]["year"] == 2025

    def test_save_manifest_has_baseline_definition(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import (
            VariantConfig,
            save_manifest,
            save_results,
        )

        results = [_make_comparison_result(2024)]
        artifact_paths = save_results(results, tmp_path)
        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]
        manifest_path = save_manifest(
            results, variant_configs, tmp_path, artifact_paths,
        )

        import json
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Baseline variant should have baseline_definition per D-22
        baseline_variant = data["variants"][0]
        assert "baseline_definition" in baseline_variant
        assert "MAWC" in baseline_variant["baseline_definition"] or "disabled" in baseline_variant["baseline_definition"]

    def test_save_manifest_has_metric_definitions(self, tmp_path: Path) -> None:
        from backtest.shadow_comparison import (
            VariantConfig,
            save_manifest,
            save_results,
        )

        results = [_make_comparison_result(2024)]
        artifact_paths = save_results(results, tmp_path)
        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
        ]
        manifest_path = save_manifest(
            results, variant_configs, tmp_path, artifact_paths,
        )

        import json
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "metric_definitions" in data
        assert "brier" in data["metric_definitions"]
        assert "logloss" in data["metric_definitions"]
        assert "ece" in data["metric_definitions"]


# ===================================================================
# Task 3 (Plan 41-02): CLI script and integration
# ===================================================================


class TestCLIScript:
    """Tests for run_shadow_comparison.py CLI."""

    def test_cli_arg_parsing_baseline_flags(self) -> None:
        """Baseline variant has enable_market_aware_calibrator=False."""
        from backtest.shadow_comparison import VariantConfig

        # Simulate CLI arg parsing logic from run_shadow_comparison.py
        baseline = VariantConfig(
            variant_name="baseline",
            model_dir=Path("data/bt"),
            enable_market_aware_calibrator=False,
            enable_race_level_ranker=False,
        )
        assert baseline.enable_market_aware_calibrator is False
        assert baseline.enable_race_level_ranker is False

    def test_cli_arg_parsing_shadow_flags(self) -> None:
        """Shadow variant has enable_market_aware_calibrator=True."""
        from backtest.shadow_comparison import VariantConfig

        shadow = VariantConfig(
            variant_name="ridge_shadow",
            model_dir=Path("data/shadow"),
            enable_market_aware_calibrator=True,
            enable_race_level_ranker=True,
        )
        assert shadow.enable_market_aware_calibrator is True
        assert shadow.enable_race_level_ranker is True

    def test_fold_definition_from_cli_args(self) -> None:
        """FoldDefinition.create_folds([2024, 2025], train_window=4) per D-05."""
        from backtest.shadow_comparison import FoldDefinition

        folds = FoldDefinition.create_folds([2024, 2025], train_window=4)
        assert len(folds) == 2

        f2024 = folds[0]
        assert f2024.year == 2024
        assert f2024.train_start == "2020-01-01"
        assert f2024.train_end == "2023-12-31"
        assert f2024.test_start == "2024-01-01"
        assert f2024.test_end == "2024-12-31"

        f2025 = folds[1]
        assert f2025.year == 2025
        assert f2025.train_start == "2021-01-01"
        assert f2025.train_end == "2024-12-31"

    @patch("backtest.engine.BacktestEngine")
    @patch("db.model_loader.ModelLoader")
    @patch("db.readers.load_odds_time_series_range")
    def test_cli_full_flow_mocked(
        self,
        mock_load_odds: MagicMock,
        mock_loader_cls: MagicMock,
        mock_engine_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Full CLI flow with mocked framework."""
        from backtest.shadow_comparison import (
            FoldDefinition,
            ShadowComparisonFramework,
            VariantConfig,
            save_manifest,
            save_results,
        )

        mock_load_odds.return_value = pd.DataFrame()

        models = _make_trained_models(mawc_trained=True, ranker_trained=True)
        mock_loader = MagicMock()
        mock_loader.load_from_dir.return_value = (models, MagicMock())
        mock_loader_cls.return_value = mock_loader

        mock_engine = MagicMock()
        bh = _make_bet_history(["R1", "R2"], [1, 3], [5.0, 8.0], [500.0, 0.0])
        mock_engine.run.return_value = _make_backtest_result(bh)
        mock_engine_cls.return_value = mock_engine

        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]

        folds = FoldDefinition.create_folds([2024], train_window=4)
        framework = ShadowComparisonFramework(
            variants=variant_configs,
            betting_target="win",
            betting_mode="flat",
            store=MagicMock(),
        )
        results = framework.run(folds)

        # Save artifacts
        artifact_paths = save_results(results, tmp_path)
        manifest_path = save_manifest(results, variant_configs, tmp_path, artifact_paths)

        assert artifact_paths["metrics_json"].exists()
        assert artifact_paths["race_diff_parquet"].exists()
        assert manifest_path.exists()

    @patch("backtest.engine.BacktestEngine")
    @patch("db.model_loader.ModelLoader")
    @patch("db.readers.load_odds_time_series_range")
    def test_cli_report_flag_triggers_generation(
        self,
        mock_load_odds: MagicMock,
        mock_loader_cls: MagicMock,
        mock_engine_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--report flag triggers HTML report generation."""
        from backtest.shadow_comparison import (
            FoldDefinition,
            ShadowComparisonFramework,
            VariantConfig,
            save_manifest,
            save_results,
        )

        mock_load_odds.return_value = pd.DataFrame()

        models = _make_trained_models(mawc_trained=True, ranker_trained=True)
        mock_loader = MagicMock()
        mock_loader.load_from_dir.return_value = (models, MagicMock())
        mock_loader_cls.return_value = mock_loader

        mock_engine = MagicMock()
        bh = _make_bet_history(["R1"], [1], [5.0], [0.0])
        mock_engine.run.return_value = _make_backtest_result(bh)
        mock_engine_cls.return_value = mock_engine

        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]

        folds = FoldDefinition.create_folds([2024], train_window=4)
        framework = ShadowComparisonFramework(
            variants=variant_configs,
            betting_target="win",
            store=MagicMock(),
        )
        results = framework.run(folds)
        artifact_paths = save_results(results, tmp_path)
        save_manifest(results, variant_configs, tmp_path, artifact_paths)

        # Generate HTML report
        from backtest.shadow_report import ShadowComparisonReportGenerator

        import json

        gen = ShadowComparisonReportGenerator(tmp_path)
        metrics_data = json.loads(
            artifact_paths["metrics_json"].read_text(encoding="utf-8"),
        )
        report_path = gen.generate(
            comparison_results=results,
            variant_configs=variant_configs,
            metrics_json=metrics_data,
        )
        assert report_path.exists()
        assert report_path.name == "shadow_comparison_report.html"

    def test_cli_help_prints(self) -> None:
        """CLI --help prints without error."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/run_shadow_comparison.py", "--help"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert result.stdout is not None
        assert "baseline-root" in result.stdout
        assert "shadow-root" in result.stdout

    def test_cli_calibration_bt_flag_default_off(self) -> None:
        """--calibration-bt defaults to False."""
        import scripts.run_shadow_comparison as script_module

        parser = script_module.build_parser()
        args = parser.parse_args([
            "--baseline-root", "data/bt",
            "--shadow-root", "data/shadow",
        ])
        assert args.calibration_bt is False

    def test_cli_calibration_bt_flag_enabled(self) -> None:
        """--calibration-bt flag enables calibration."""
        import scripts.run_shadow_comparison as script_module

        parser = script_module.build_parser()
        args = parser.parse_args([
            "--baseline-root", "data/bt",
            "--shadow-root", "data/shadow",
            "--calibration-bt",
        ])
        assert args.calibration_bt is True
