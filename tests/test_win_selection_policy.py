"""WinSelectionPolicy tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import pandas as pd
import pytest

from models.win_selection_policy import (
    DEFAULT_LATE_ODDS_DROP_WEIGHT,
    DEFAULT_LOG_ODDS_PENALTY,
    DEFAULT_PROB_RANK_BONUS,
    WinSelectionPolicy,
    deployed_late_odds_drop_weight,
    deployed_policy_params,
)


def _training_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    race_no = 0
    for year in [2022, 2023]:
        for month in range(1, 4):
            race_no += 1
            race_id = f"{year}{month:02d}{race_no:08d}"
            race_date = pd.Timestamp(year=year, month=month, day=1)
            rows.extend(
                [
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "umaban": 1,
                        "kakuteijyuni": 3,
                        "tanodds": 10.0,
                        "win_selection_prob": 0.13,
                        "win_selection_ev": 1.30,
                        "win_selection_edge": 0.30,
                        "odds_drop_rate_30_10": 1.0,
                    },
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "umaban": 2,
                        "kakuteijyuni": 1,
                        "tanodds": 4.0,
                        "win_selection_prob": 0.28,
                        "win_selection_ev": 1.28,
                        "win_selection_edge": 0.28,
                        "odds_drop_rate_30_10": -1.0,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_train_selects_late_odds_drop_penalty_from_historical_roi() -> None:
    policy = WinSelectionPolicy()

    policy.train(_training_rows())
    scored = policy.apply(
        pd.DataFrame(
            {
                "race_id": ["202401010001", "202401010001"],
                "race_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
                "umaban": [1, 2],
                "kakuteijyuni": [2, 1],
                "tanodds": [10.0, 4.0],
                "win_selection_prob": [0.13, 0.28],
                "win_selection_ev": [1.30, 1.28],
                "win_selection_edge": [0.30, 0.28],
                "odds_drop_rate_30_10": [1.0, -1.0],
            }
        )
    )

    assert policy.is_trained is True
    assert policy.late_odds_drop_weight > 0.0
    assert scored.sort_values("selected_rank_by_win_market_score").iloc[0]["umaban"] == 2
    assert scored["win_late_odds_drop_z"].tolist() == pytest.approx([0.70710678, -0.70710678])


def test_win_selection_policy_save_load_roundtrip() -> None:
    policy = WinSelectionPolicy()
    policy.train(_training_rows())

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "policy.joblib"
        policy.save(path)
        loaded = WinSelectionPolicy.load(path)

    assert loaded.is_trained is True
    assert loaded.late_odds_drop_weight == pytest.approx(policy.late_odds_drop_weight)
    assert loaded.training_summary["selected_weight"] == pytest.approx(
        policy.training_summary["selected_weight"]
    )


def test_negative_saved_policy_weight_is_not_deployed() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bad_policy.joblib"
        joblib.dump(
            {
                "late_odds_drop_weight": -0.12,
                "candidate_weights": (-0.12, 0.06),
                "is_trained": True,
                "training_summary": {"deployable": True},
            },
            path,
        )
        loaded = WinSelectionPolicy.load(path)

    assert loaded.late_odds_drop_weight == pytest.approx(DEFAULT_LATE_ODDS_DROP_WEIGHT)
    assert deployed_late_odds_drop_weight(loaded) == pytest.approx(
        DEFAULT_LATE_ODDS_DROP_WEIGHT
    )
    params = deployed_policy_params(loaded)
    assert params["log_odds_penalty"] == pytest.approx(DEFAULT_LOG_ODDS_PENALTY)
    assert params["prob_rank_bonus"] == pytest.approx(DEFAULT_PROB_RANK_BONUS)


def test_non_deployable_policy_uses_default_weight() -> None:
    policy = WinSelectionPolicy(late_odds_drop_weight=0.12, is_trained=True)
    policy.training_summary = {"deployable": False}

    assert deployed_late_odds_drop_weight(policy) == pytest.approx(
        DEFAULT_LATE_ODDS_DROP_WEIGHT
    )
    assert deployed_policy_params(policy) == {
        "late_odds_drop_weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
        "log_odds_penalty": DEFAULT_LOG_ODDS_PENALTY,
        "prob_rank_bonus": DEFAULT_PROB_RANK_BONUS,
    }
