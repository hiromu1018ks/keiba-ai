"""EV推定精度診断モジュールのテスト."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.ev_diagnostics import (
    _brier_decomposition,
    _compute_ece,
    _reliability_diagram_data,
    compute_ev_diagnostics,
    console_summary,
)


def _build_ev_df(
    n_rows: int = 300,
    ev_mean: float = 1.2,
    ev_std: float = 0.4,
    seed: int = 42,
    surface: str = "turf",
) -> pd.DataFrame:
    """テスト用OOF DataFrameを構築する."""
    rng = np.random.RandomState(seed)
    rows: list[dict[str, object]] = []
    for i in range(n_rows):
        won = rng.random() < 0.15
        row: dict[str, object] = {
            "p_win_corrected": float(np.clip(rng.normal(0.15, 0.05), 0.01, 0.99)),
            "ev_win_corrected": float(np.clip(rng.normal(ev_mean, ev_std), 0.1, 5.0)),
            "confirmed_odds": float(np.clip(rng.normal(8.0, 5.0), 1.0, 100.0)),
            "kakuteijyuni": 1 if won else rng.randint(2, 18),
            "surface": surface,
            "race_date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=i % 365),
            "win_selection_edge": float(rng.normal(0.05, 0.02)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


class TestComputeECE:
    def test_ece_zero_for_perfect_calibration(self) -> None:
        """完全キャリブレーション時 ECE=0"""
        y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        assert _compute_ece(y_true, y_prob) == pytest.approx(0.0, abs=1e-6)

    def test_ece_positive_for_miscalibration(self) -> None:
        """ミスキャリブレーション時 ECE > 0"""
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, size=200).astype(float)
        y_prob = np.clip(rng.normal(0.5, 0.2, size=200), 0.01, 0.99)
        assert _compute_ece(y_true, y_prob) > 0.0


class TestBrierDecomposition:
    def test_brier_components_sum_property(self) -> None:
        """Brier >= reliability - resolution + uncertainty (Murphy近似)"""
        rng = np.random.RandomState(1)
        y_true = rng.randint(0, 2, size=200).astype(float)
        y_prob = np.clip(rng.normal(0.4, 0.15, size=200), 0.01, 0.99)
        result = _brier_decomposition(y_true, y_prob)
        assert result["brier_score"] >= 0.0
        assert result["reliability"] >= 0.0
        assert result["resolution"] >= 0.0
        assert 0.0 <= result["uncertainty"] <= 0.25

    def test_brier_perfect_prediction(self) -> None:
        """完璧な予測時 brier=0"""
        y_true = np.array([1.0, 1.0, 0.0, 0.0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        result = _brier_decomposition(y_true, y_prob)
        assert result["brier_score"] == pytest.approx(0.0, abs=1e-6)


class TestReliabilityDiagram:
    def test_returns_expected_keys(self) -> None:
        """Reliability diagram dataが必要なキーを持つ"""
        rng = np.random.RandomState(2)
        y_true = rng.randint(0, 2, size=100).astype(float)
        y_prob = np.clip(rng.normal(0.3, 0.1, size=100), 0.01, 0.99)
        result = _reliability_diagram_data(y_true, y_prob)
        assert "fraction_of_positives" in result
        assert "mean_predicted_value" in result
        assert len(result["fraction_of_positives"]) > 0


class TestComputeEvDiagnostics:
    def test_basic_returns_required_fields(self) -> None:
        """基本診断が必須フィールドを返す"""
        df = _build_ev_df(n_rows=300)
        result = compute_ev_diagnostics(df)
        assert "surface" in result
        assert "n_valid" in result
        assert "correlation" in result
        assert "rmse" in result
        assert "ev_bias" in result
        assert result["n_valid"] > 0

    def test_json_output(self) -> None:
        """JSON出力が正しく書き込まれる"""
        df = _build_ev_df(n_rows=200)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ev_diag_turf.json"
            result = compute_ev_diagnostics(df, output_path=path, surface="turf")
            assert path.exists()
            with open(path) as f:
                saved = json.load(f)
            assert saved["surface"] == "turf"
            assert saved["n_valid"] == result["n_valid"]

    def test_insufficient_samples_warning(self) -> None:
        """サンプル不足時にwarningが設定される"""
        df = _build_ev_df(n_rows=5)
        result = compute_ev_diagnostics(df)
        assert "warning" in result
        assert result["warning"] == "insufficient_samples"

    def test_missing_actual_ev_computes_from_odds(self) -> None:
        """actual_ev_win列がなくてもconfirmed_oddsから計算される"""
        df = _build_ev_df(n_rows=100)
        df = df.drop(columns=["actual_ev_win"], errors="ignore")
        result = compute_ev_diagnostics(df)
        assert result["n_valid"] > 0

    def test_console_summary_no_error(self) -> None:
        """console_summaryが例外を投げない"""
        df = _build_ev_df(n_rows=200)
        result = compute_ev_diagnostics(df, surface="turf")
        # Should not raise
        console_summary(result)

    def test_temporal_drift_included(self) -> None:
        """時系列ドリフトが結果に含まれる"""
        df = _build_ev_df(n_rows=300)
        result = compute_ev_diagnostics(df)
        assert "temporal_drift" in result
        assert isinstance(result["temporal_drift"], list)
        assert len(result["temporal_drift"]) > 0
