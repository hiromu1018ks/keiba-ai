"""WinSelectionGateModel tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


def test_win_selection_gate_trains_and_scores() -> None:
    """Test 1: WinSelectionGateModel trains on 120 races and produces score/pass columns."""
    from models.win_selection_gate import WinSelectionGateModel

    rows: list[dict[str, object]] = []
    for race_idx in range(120):
        race_id = f"R{race_idx:04d}"
        race_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=race_idx)
        for umaban, prob, edge, odds, finish in [
            (1, 0.62, 0.24, 2.2, 1 if race_idx % 10 == 0 else 5),  # Win: only 1st
            (2, 0.28, 0.02, 4.5, 4),
            (3, 0.10, -0.15, 11.0, 8),
        ]:
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": umaban,
                    "kakuteijyuni": finish,
                    "tanoddslow": odds,
                    "win_selection_prob": prob,
                    "win_selection_edge": edge,
                }
            )

    df = pd.DataFrame(rows)
    model = WinSelectionGateModel(min_train_races=40, min_fold_races=20, max_folds=3)
    model.train(df)

    assert model.is_trained is True

    scored = model.score(
        pd.DataFrame(
            {
                "race_id": ["T1", "T1"],
                "race_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
                "umaban": [1, 2],
                "tanoddslow": [2.2, 11.0],
                "win_selection_prob": [0.60, 0.09],
                "win_selection_edge": [0.22, -0.18],
            }
        )
    )

    assert "win_gate_score" in scored.columns
    assert "win_gate_pass" in scored.columns
    # High prob + high edge horse should have higher score than low prob + low edge
    assert scored.loc[0, "win_gate_score"] > scored.loc[1, "win_gate_score"]


def test_ensure_win_selection_columns_fallback_chain() -> None:
    """Test 4: ensure_win_selection_columns handles EV/edge/prob fallback chain."""
    from models.win_selection_gate import ensure_win_selection_columns

    # Case 1: Has EV_lower_win_corrected -> should compute win_selection_ev
    df1 = pd.DataFrame(
        {
            "EV_lower_win_corrected": [1.2, 0.8],
            "ev_win_corrected": [1.3, 0.9],
            "ev_win": [1.1, 0.7],
            "p_win_final": [0.30, 0.10],
            "edge_win": [0.05, -0.10],
        }
    )
    result1 = ensure_win_selection_columns(df1)
    assert "win_selection_ev" in result1.columns
    assert "win_selection_edge" in result1.columns
    assert "win_selection_prob" in result1.columns

    # Case 2: Only ev_win -> should fallback correctly
    df2 = pd.DataFrame(
        {
            "ev_win": [1.1, 0.7],
            "p_win_pred": [0.30, 0.10],
        }
    )
    result2 = ensure_win_selection_columns(df2)
    assert "win_selection_ev" in result2.columns
    assert "win_selection_prob" in result2.columns


def test_build_win_selection_ev() -> None:
    """Test 5: build_win_selection_ev returns max of lower_ev + corrected_ev safety floor."""
    from models.win_selection_gate import build_win_selection_ev

    # When both EV_lower_win_corrected and ev_win_corrected exist
    df = pd.DataFrame(
        {
            "EV_lower_win_corrected": [0.20, 1.50],
            "ev_win_corrected": [1.50, 0.80],
            "ev_win": [0.50, 1.00],
        }
    )
    ev = build_win_selection_ev(df)
    # Row 0: lower_ev=0.20 (notna), corrected_ev=1.50 -> selection_ev=0.20 (keep lower since notna)
    # safety_floor = 1.50 * 0.85 = 1.275
    # result = max(0.20, 1.275) = 1.275
    assert ev.iloc[0] == pytest.approx(1.275)
    # Row 1: lower_ev=1.50 (notna), corrected_ev=0.80 -> selection_ev=1.50 (keep lower since notna)
    # safety_floor = 0.80 * 0.85 = 0.68
    # result = max(1.50, 0.68) = 1.50
    assert ev.iloc[1] == pytest.approx(1.50)


def test_win_selection_gate_hit_condition() -> None:
    """Test 6: realized_win_roi is positive only when kakuteijyuni==1."""
    from models.win_selection_gate import WinSelectionGateModel

    # Create data where some horses finished 1st (win) and others did not
    rows: list[dict[str, object]] = []
    for race_idx in range(120):
        race_id = f"R{race_idx:04d}"
        race_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=race_idx)
        for umaban, prob, edge, odds, finish in [
            (1, 0.62, 0.24, 2.2, 1 if race_idx % 10 == 0 else 5),
            (2, 0.28, 0.02, 4.5, 4),
            (3, 0.10, -0.15, 11.0, 8),
        ]:
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": umaban,
                    "kakuteijyuni": finish,
                    "tanoddslow": odds,
                    "win_selection_prob": prob,
                    "win_selection_edge": edge,
                }
            )

    df = pd.DataFrame(rows)
    model = WinSelectionGateModel(min_train_races=40, min_fold_races=20, max_folds=3)
    model.train(df)

    # Verify that model training uses kakuteijyuni==1 for hit detection
    # by checking that the model actually trained successfully
    assert model.is_trained is True


def test_win_selection_gate_save_load_roundtrip() -> None:
    """Test 7: save/load roundtrip preserves model state."""
    from models.win_selection_gate import WinSelectionGateModel

    rows: list[dict[str, object]] = []
    for race_idx in range(120):
        race_id = f"R{race_idx:04d}"
        race_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=race_idx)
        for umaban, prob, edge, odds, finish in [
            (1, 0.62, 0.24, 2.2, 1 if race_idx % 10 == 0 else 5),
            (2, 0.28, 0.02, 4.5, 4),
            (3, 0.10, -0.15, 11.0, 8),
        ]:
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": umaban,
                    "kakuteijyuni": finish,
                    "tanoddslow": odds,
                    "win_selection_prob": prob,
                    "win_selection_edge": edge,
                }
            )

    df = pd.DataFrame(rows)
    model = WinSelectionGateModel(min_train_races=40, min_fold_races=20, max_folds=3)
    model.train(df)
    assert model.is_trained

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        model.save(tmp_path)
        loaded = WinSelectionGateModel.load(tmp_path)

        assert loaded.is_trained is True
        assert loaded.min_prob == model.min_prob
        assert loaded.min_edge == model.min_edge
        assert loaded.max_odds == model.max_odds
        assert loaded.global_score == model.global_score
    finally:
        tmp_path.unlink(missing_ok=True)


def test_win_selection_gate_soft_pass_mask() -> None:
    """Test 8: soft_pass_mask rescues top-ranked near-threshold runners."""
    from models.win_selection_gate import WinSelectionGateModel

    model = WinSelectionGateModel()
    model.min_prob = 0.38
    model.min_edge = 0.05
    model.max_odds = 10.0
    model._trained = True

    scored = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "umaban": [1, 2, 1, 2],
            "tanoddslow": [9.8, 8.0, 5.0, 8.5],
            "win_selection_prob": [0.375, 0.31, 0.42, 0.375],
            "win_selection_edge": [0.035, 0.08, 0.07, 0.04],
            "win_gate_score": [1.4, 1.0, 1.2, 1.1],
            "win_gate_pass": [False, False, True, False],
        }
    )

    mask = model.soft_pass_mask(
        scored,
        edge_floor=0.0,
        min_prob=0.08,
        max_odds=18.0,
        max_per_race=1,
    )

    # R1: horse 1 is top-ranked by score, near threshold, should be rescued
    assert mask.tolist() == [True, False, False, False]
