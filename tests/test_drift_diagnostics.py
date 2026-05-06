"""ドリフト診断モジュールのテスト."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.drift_diagnostics import DRIFT_COLUMNS, compute_drift_diagnostics, console_summary


def _build_oof_df(
    n_rows: int = 300,
    prob_mean: float = 0.20,
    prob_std: float = 0.05,
    seed: int = 0,
    surface: str | None = "turf",
) -> pd.DataFrame:
    """テスト用OOF DataFrameを構築する."""
    rng = np.random.RandomState(seed)
    rows: list[dict[str, object]] = []
    for i in range(n_rows):
        row: dict[str, object] = {
            "p_win_pred": float(np.clip(rng.normal(prob_mean, prob_std), 0.01, 0.99)),
            "ev_win": float(np.clip(rng.normal(1.2, 0.3), 0.1, 5.0)),
            "p_win_corrected": float(np.clip(rng.normal(prob_mean + 0.02, prob_std), 0.01, 0.99)),
            "ev_win_corrected": float(np.clip(rng.normal(1.3, 0.4), 0.1, 5.0)),
            "win_selection_prob": float(np.clip(rng.normal(prob_mean, prob_std), 0.01, 0.99)),
            "win_selection_edge": float(np.clip(rng.normal(0.05, 0.02), -0.5, 1.0)),
            "win_selection_ev": float(np.clip(rng.normal(1.15, 0.3), 0.1, 5.0)),
        }
        if surface is not None:
            row["surface"] = surface if i < n_rows // 2 else "dirt"
        row["race_date"] = pd.Timestamp("2022-01-01") + pd.Timedelta(days=i % 365)
        rows.append(row)
    return pd.DataFrame(rows)


def test_compute_drift_diagnostics_basic() -> None:
    """Test 1: df_oof単体でstats辞書を返す (characterization mode)."""
    df = _build_oof_df(n_rows=300, surface=None)
    result = compute_drift_diagnostics(df)

    assert "columns" in result
    assert "drift_detected" in result
    assert result["drift_detected"] is False
    assert "surface" not in result.get("surfaces", {})

    # DRIFT_COLUMNSのうちDataFrameに存在する列が全てstatsを持つ
    for col in DRIFT_COLUMNS:
        if col in df.columns:
            assert col in result["columns"], f"Missing stats for {col}"
            stats = result["columns"][col]["stats"]
            assert "mean" in stats
            assert "std" in stats
            assert "q25" in stats
            assert "q50" in stats
            assert "q75" in stats
            assert "n" in stats
            assert stats["n"] == 300


def test_drift_detection_warning() -> None:
    """Test 2: df_baselineと比較してdrift_detected=trueを返す."""
    rng = np.random.RandomState(0)
    df_oof = _build_oof_df(n_rows=300, prob_mean=0.20, prob_std=0.05, seed=0, surface=None)
    # 明らかに異なる分布のbaseline
    df_baseline = _build_oof_df(n_rows=300, prob_mean=0.50, prob_std=0.15, seed=1, surface=None)

    result = compute_drift_diagnostics(df_oof, df_baseline=df_baseline)

    assert result["drift_detected"] is True
    # 少なくとも1列がKS p-value < 0.05
    drift_cols = []
    for col, data in result["columns"].items():
        if "comparison" in data:
            if data["comparison"]["ks_pvalue"] < 0.05:
                drift_cols.append(col)
    assert len(drift_cols) > 0, "No drift detected despite very different distributions"

    # recommendationsが含まれる
    assert "recommendations" in result
    assert len(result["recommendations"]) > 0


def test_surface_split_diagnostics() -> None:
    """Test 3: surface列がある場合にサーフェス別breakdownを返す."""
    df = _build_oof_df(n_rows=300, surface="turf")
    result = compute_drift_diagnostics(df)

    assert "surfaces" in result
    assert "turf" in result["surfaces"]
    assert "dirt" in result["surfaces"]

    for surf in ["turf", "dirt"]:
        surf_result = result["surfaces"][surf]
        assert "columns" in surf_result
        assert "drift_detected" in surf_result
        # 各サーフェスにもstatsがある
        for col in DRIFT_COLUMNS:
            if col in df.columns:
                assert col in surf_result["columns"]


def test_year_split_diagnostics() -> None:
    """Test 4: race_date列がある場合に年度別breakdownを返す."""
    df = _build_oof_df(n_rows=300, surface=None)
    # 複数年度を含む
    rng = np.random.RandomState(42)
    df["race_date"] = pd.to_datetime(df["race_date"])
    # 半分を2022年、半分を2023年に設定
    df.loc[:149, "race_date"] = pd.Timestamp("2022-06-01") + pd.to_timedelta(
        rng.randint(0, 180, 150), unit="D"
    )
    df.loc[150:, "race_date"] = pd.Timestamp("2023-06-01") + pd.to_timedelta(
        rng.randint(0, 180, 150), unit="D"
    )

    result = compute_drift_diagnostics(df)

    assert "years" in result
    assert "2022" in result["years"]
    assert "2023" in result["years"]

    for year in ["2022", "2023"]:
        year_result = result["years"][year]
        assert "columns" in year_result
        assert "drift_detected" in year_result


def test_json_output(tmp_path: Path) -> None:
    """Test 5: output_path指定時にJSON出力する."""
    df = _build_oof_df(n_rows=200, surface=None)
    output_path = tmp_path / "drift_test.json"
    result = compute_drift_diagnostics(df, output_path=output_path)

    assert output_path.exists()
    with open(output_path, encoding="utf-8") as f:
        loaded = json.load(f)

    assert "columns" in loaded
    assert "drift_detected" in loaded
    assert loaded["drift_detected"] is False


def test_recommendations_on_drift() -> None:
    """Test 6: drift_detected時にWARNINGログとrecommendationsを含む."""
    df_oof = _build_oof_df(n_rows=300, prob_mean=0.20, prob_std=0.05, seed=0, surface=None)
    df_baseline = _build_oof_df(n_rows=300, prob_mean=0.50, prob_std=0.15, seed=1, surface=None)

    with pytest.warns(None) as _:  # ログが例外を起こさないこと
        result = compute_drift_diagnostics(df_oof, df_baseline=df_baseline)

    assert result["drift_detected"] is True
    assert "recommendations" in result
    assert any("再学習" in r or "retrain" in r.lower() for r in result["recommendations"])


def test_missing_and_nan_columns() -> None:
    """Test 7: 欠搋列・全NaN列・<30行の列をgracefullyにskipする."""
    df = pd.DataFrame(
        {
            "p_win_pred": [0.1] * 50,
            "ev_win": [np.nan] * 50,  # 全NaN → skip
            "p_win_corrected": [0.15] * 25 + [np.nan] * 25,  # 25 non-NaN < 30 → skip
            "win_selection_prob": [0.2] * 50,
            "win_selection_edge": [0.05] * 50,
            "win_selection_ev": [1.1] * 50,
            "race_date": [pd.Timestamp("2023-01-01")] * 50,
        }
    )

    result = compute_drift_diagnostics(df)

    # p_win_pred, win_selection_prob/edge/ev はstatsを持つ
    assert "p_win_pred" in result["columns"]
    assert "win_selection_prob" in result["columns"]
    # ev_win (全NaN) と p_win_corrected (< 30 non-NaN) はskip
    assert "ev_win" not in result["columns"]
    assert "p_win_corrected" not in result["columns"]


def test_console_summary_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Test 8: console_summary()がINFOログでフォーマット済みサマリを出力する."""
    df = _build_oof_df(n_rows=200, surface=None)
    result = compute_drift_diagnostics(df)

    with caplog.at_level(logging.INFO, logger="models.drift_diagnostics"):
        console_summary(result)

    # 何らかのINFOログが出力されていること
    assert len(caplog.records) > 0
    log_messages = " ".join(r.getMessage() for r in caplog.records)
    # 分布stats or ドリフト情報が含まれていること
    assert "p_win_pred" in log_messages or "DRIFT" in log_messages or "mean" in log_messages
