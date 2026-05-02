"""WinBenterGate ユニットテスト -- TDD RED/GREEN/REFACTOR"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from unittest.mock import MagicMock, patch

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
    return pd.DataFrame({
        "race_id": race_ids,
        "p_win_corrected": p_win_corrected,
        "tanodds": tanodds,
        "kakuteijyuni": kakuteijyuni,
    })


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
    """OOF 予測生成が3つの整合配列を返す."""

    def test_oof_output_shape(self) -> None:
        from models.win_benter_gate import generate_win_oof_predictions

        n = 100
        df = pd.DataFrame({
            "race_id": [f"R{i // 5}" for i in range(n)],
            "race_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "p_win_pred": np.random.uniform(0.05, 0.4, n),
            "tanodds": np.random.uniform(2.0, 20.0, n),
            "kakuteijyuni": np.random.randint(1, 16, n),
        })

        # Mock WinTwoStageModel
        mock_model_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.predict_ev.side_effect = lambda d: d.assign(
            p_win_pred=d.get("p_win_pred", np.random.uniform(0.05, 0.4, len(d)))
        )
        mock_instance.train_hit_model.return_value = None
        mock_model_cls.return_value = mock_instance

        # Mock EVCorrectionModel
        mock_ev = MagicMock()
        mock_ev.correct_ev.side_effect = lambda d: d.assign(
            p_win_corrected=d.get("p_win_pred", np.random.uniform(0.05, 0.4, len(d)))
            * 0.95
        )

        p_fund, p_market, y = generate_win_oof_predictions(
            df,
            win_model_cls=mock_model_cls,
            ev_corrector=mock_ev,
            n_splits=5,
        )

        assert len(p_fund) == len(p_market) == len(y)
        assert len(p_fund) > 0
        assert not np.any(np.isnan(p_fund))
        assert not np.any(np.isnan(p_market))
        # y should be binary (0 or 1)
        assert set(np.unique(y)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Test 6: SubmodelSet has win_* fields
# ---------------------------------------------------------------------------

class TestSubmodelSetWinFields:
    """SubmodelSet に win_benter, win_isotonic_calibrator, win_temperature_scaler がある."""

    def test_win_fields_exist(self) -> None:
        from dataclasses import fields as dc_fields

        from domain.models import SubmodelSet

        field_names = [f.name for f in dc_fields(SubmodelSet)]
        assert "win_benter" in field_names
        assert "win_isotonic_calibrator" in field_names
        assert "win_temperature_scaler" in field_names

    def test_win_fields_default_none(self) -> None:
        """win_* フィールドのデフォルト値が None であることを確認する."""
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
            confidence=mock,
        )
        assert sub.win_benter is None
        assert sub.win_isotonic_calibrator is None
        assert sub.win_temperature_scaler is None
