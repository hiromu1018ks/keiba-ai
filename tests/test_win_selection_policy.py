"""WinSelectionPolicy tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import pandas as pd
import pytest

from models.win_selection_policy import (
    DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
    DEFAULT_DIRT_LOG_ODDS_PENALTY,
    DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
    DEFAULT_DIRT_PROB_RANK_BONUS,
    DEFAULT_EV_TAIL_PENALTY_WEIGHT,
    DEFAULT_LATE_ODDS_DROP_WEIGHT,
    DEFAULT_LOG_ODDS_PENALTY,
    DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
    DEFAULT_PROB_RANK_BONUS,
    MAX_DEPLOYED_MARKET_RISK_PENALTY_WEIGHT,
    MIN_DIRT_JOINT_MEAN_ROI_IMPROVEMENT,
    MIN_DIRT_JOINT_YEAR_ROI_FLOOR,
    WinSelectionPolicy,
    _annotate_policy_deployability,
    _roi_for_score,
    _roi_for_score_with_cap,
    _roi_from_selected,
    deployed_late_odds_drop_weight,
    deployed_policy_params,
    sanitize_market_risk_penalty_weight,
)


def _training_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    race_no = 0
    for year in [2021, 2022, 2023]:
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
                        "tanodds": 5.0,
                        "p_win_final": 0.60,
                        "win_selection_prob": 0.60,
                        "win_selection_ev": 1.30,
                        "win_selection_edge": 0.30,
                        "odds_drop_rate_30_10": 1.0,
                    },
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "umaban": 2,
                        "kakuteijyuni": 1,
                        "tanodds": 5.0,
                        "p_win_final": 0.59,
                        "win_selection_prob": 0.59,
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
                "tanodds": [5.0, 5.0],
                "p_win_final": [0.60, 0.59],
                "win_selection_prob": [0.60, 0.59],
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


def test_roi_for_score_prefers_realized_win_return_over_snapshot_odds() -> None:
    rows = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "umaban": [1, 2, 1, 2],
            "kakuteijyuni": [1, 2, 1, 2],
            "tanodds": [50.0, 2.0, 40.0, 2.0],
            "confirmed_odds": [3.0, 2.0, 4.0, 2.0],
            "win_return_unit": [3.0, 0.0, 4.0, 0.0],
        }
    )
    score = pd.Series([1.0, 0.0, 1.0, 0.0], index=rows.index)

    assert _roi_for_score(rows, score) == pytest.approx(3.5)


def test_win_selection_policy_save_load_roundtrip() -> None:
    policy = WinSelectionPolicy()
    policy.train(_training_rows())

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "policy.joblib"
        policy.save(path)
        loaded = WinSelectionPolicy.load(path)

    assert loaded.is_trained is True
    assert loaded.surface == policy.surface
    assert loaded.late_odds_drop_weight == pytest.approx(policy.late_odds_drop_weight)
    assert loaded.log_odds_penalty == pytest.approx(policy.log_odds_penalty)
    assert loaded.market_risk_penalty_weight == pytest.approx(policy.market_risk_penalty_weight)
    assert loaded.training_summary["selected_weight"] == pytest.approx(
        policy.training_summary["selected_weight"]
    )
    assert loaded.training_summary["surface"] == policy.training_summary["surface"]
    assert loaded.ev_tail_penalty_weight == pytest.approx(policy.ev_tail_penalty_weight)


def _tail_shrinkage_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    race_no = 0
    for year in [2021, 2022, 2023]:
        for month in range(1, 8):
            race_no += 1
            race_id = f"{year}{month:02d}{race_no:08d}"
            race_date = pd.Timestamp(year=year, month=month, day=1)
            rows.extend(
                [
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "umaban": 1,
                        "kakuteijyuni": 2,
                        "tanodds": 20.0,
                        "win_selection_prob": 0.07,
                        "win_selection_ev": 2.00,
                        "win_selection_edge": 1.00,
                        "win_market_residual": 0.40,
                        "odds_drop_rate_30_10": 0.0,
                    },
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "umaban": 2,
                        "kakuteijyuni": 1,
                        "tanodds": 4.0,
                        "win_selection_prob": 0.25,
                        "win_selection_ev": 1.15,
                        "win_selection_edge": 0.15,
                        "win_market_residual": 0.15,
                        "odds_drop_rate_30_10": 0.0,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_train_can_deploy_oof_ev_tail_shrinkage_without_filtering() -> None:
    policy = WinSelectionPolicy()
    rows = _tail_shrinkage_rows()
    policy.train(rows)
    scored = policy.apply(rows.head(2))

    assert policy.training_summary["deployable"] is True
    assert policy.ev_tail_penalty_weight > 0.0
    assert scored.sort_values("selected_rank_by_win_market_score").iloc[0]["umaban"] == 2


def test_tail_shrinkage_uses_stricter_deploy_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "models.win_selection_policy.MIN_TAIL_SHRINKAGE_MEAN_ROI_IMPROVEMENT",
        10.0,
    )
    policy = WinSelectionPolicy()
    policy.train(_tail_shrinkage_rows())

    assert policy.training_summary["deployable"] is False
    assert policy.training_summary["stable_tail_shrinkage_met"] is False
    assert policy.ev_tail_penalty_weight == pytest.approx(DEFAULT_EV_TAIL_PENALTY_WEIGHT)


def test_policy_deployability_rejects_any_yearly_regression() -> None:
    result = pd.DataFrame(
        [
            {
                "weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "roi_all": 0.90,
                "roi_mean_by_year": 0.90,
                "roi_min_by_year": 0.90,
                "n_years": 3,
                "year_rois": {"2021": 0.90, "2022": 0.90, "2023": 0.90},
            },
            {
                "weight": 0.12,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "roi_all": 0.96,
                "roi_mean_by_year": 0.96,
                "roi_min_by_year": 0.89,
                "n_years": 3,
                "year_rois": {"2021": 1.05, "2022": 0.94, "2023": 0.89},
            },
            {
                "weight": 0.08,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "roi_all": 0.96,
                "roi_mean_by_year": 0.96,
                "roi_min_by_year": 0.95,
                "n_years": 3,
                "year_rois": {"2021": 0.97, "2022": 0.96, "2023": 0.95},
            },
            {
                "weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
                "ev_tail_penalty_weight": 0.30,
                "roi_all": 0.91,
                "roi_mean_by_year": 0.91,
                "roi_min_by_year": 0.905,
                "n_years": 3,
                "year_rois": {"2021": 0.915, "2022": 0.91, "2023": 0.905},
            },
            {
                "weight": 0.10,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "roi_all": 0.93,
                "roi_mean_by_year": 0.93,
                "roi_min_by_year": 0.92,
                "n_years": 3,
                "year_rois": {"2021": 0.94, "2022": 0.93, "2023": 0.92},
            },
        ]
    )

    annotated = _annotate_policy_deployability(
        result,
        default_row=result.iloc[0],
        default_late_weight=DEFAULT_LATE_ODDS_DROP_WEIGHT,
    )

    risky = annotated.loc[annotated["weight"].eq(0.12)].iloc[0]
    stable = annotated.loc[annotated["weight"].eq(0.08)].iloc[0]
    stable_tail = annotated.loc[annotated["ev_tail_penalty_weight"].eq(0.30)].iloc[0]
    weak_late_change = annotated.loc[annotated["weight"].eq(0.10)].iloc[0]
    assert risky["min_year_delta_vs_default"] == pytest.approx(-0.01)
    assert bool(risky["deployable_candidate"]) is False
    assert stable["min_year_delta_vs_default"] == pytest.approx(0.05)
    assert bool(stable["deployable_candidate"]) is True
    assert stable_tail["mean_delta_vs_default"] == pytest.approx(0.01)
    assert bool(stable_tail["deployable_candidate"]) is True
    assert weak_late_change["mean_delta_vs_default"] == pytest.approx(0.03)
    assert bool(weak_late_change["deployable_candidate"]) is False


def test_dirt_policy_keeps_edge_base_and_dirt_defaults() -> None:
    policy = WinSelectionPolicy()
    scored = policy.apply(
        pd.DataFrame(
            {
                "race_id": ["202401010001", "202401010001"],
                "race_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
                "surface": ["dirt", "dirt"],
                "umaban": [1, 2],
                "kakuteijyuni": [2, 1],
                "tanodds": [2.0, 12.0],
                "p_win_final": [0.60, 0.08],
                "win_selection_prob": [0.60, 0.08],
                "win_selection_ev": [1.05, 1.30],
                "win_selection_edge": [0.05, 0.30],
                "odds_drop_rate_30_10": [0.0, 0.0],
            }
        )
    )

    selected = scored.sort_values("selected_rank_by_win_market_score").iloc[0]
    assert selected["umaban"] == 2
    assert selected["win_late_odds_drop_weight"] == pytest.approx(
        DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT
    )
    assert selected["win_log_odds_penalty"] == pytest.approx(DEFAULT_DIRT_LOG_ODDS_PENALTY)
    assert selected["win_prob_rank_bonus"] == pytest.approx(DEFAULT_DIRT_PROB_RANK_BONUS)
    assert selected["win_market_risk_penalty_weight"] == pytest.approx(
        DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT
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
    assert deployed_late_odds_drop_weight(loaded) == pytest.approx(DEFAULT_LATE_ODDS_DROP_WEIGHT)
    params = deployed_policy_params(loaded)
    assert params["log_odds_penalty"] == pytest.approx(DEFAULT_LOG_ODDS_PENALTY)
    assert params["prob_rank_bonus"] == pytest.approx(DEFAULT_PROB_RANK_BONUS)


def test_non_deployable_policy_uses_default_weight() -> None:
    policy = WinSelectionPolicy(late_odds_drop_weight=0.12, is_trained=True)
    policy.training_summary = {"deployable": False}

    assert deployed_late_odds_drop_weight(policy) == pytest.approx(DEFAULT_LATE_ODDS_DROP_WEIGHT)
    params = deployed_policy_params(policy)
    assert params["late_odds_drop_weight"] == pytest.approx(DEFAULT_LATE_ODDS_DROP_WEIGHT)
    assert params["log_odds_penalty"] == pytest.approx(DEFAULT_LOG_ODDS_PENALTY)
    assert params["prob_rank_bonus"] == pytest.approx(DEFAULT_PROB_RANK_BONUS)
    assert params["recommended_odds_cap"] is None


# ── Surface-aware deployment tests ──────────────────────────────────


def test_dirt_trained_policy_maps_learned_params_to_dirt_slots() -> None:
    """deployed_policy_params maps learned values to dirt slots for dirt policy."""
    policy = WinSelectionPolicy(surface="dirt", is_trained=True)
    policy.late_odds_drop_weight = 0.08
    policy.log_odds_penalty = 0.04
    policy.prob_rank_bonus = 0.03
    policy.market_risk_penalty_weight = 0.05
    policy.training_summary = {"deployable": True}

    params = deployed_policy_params(policy)

    assert params["dirt_late_odds_drop_weight"] == pytest.approx(0.08)
    assert params["dirt_log_odds_penalty"] == pytest.approx(0.04)
    assert params["dirt_prob_rank_bonus"] == pytest.approx(0.03)
    assert params["dirt_market_risk_penalty_weight"] == pytest.approx(0.05)
    # turf slots stay at turf defaults
    assert params["late_odds_drop_weight"] == pytest.approx(DEFAULT_LATE_ODDS_DROP_WEIGHT)
    assert params["log_odds_penalty"] == pytest.approx(DEFAULT_LOG_ODDS_PENALTY)
    assert params["prob_rank_bonus"] == pytest.approx(DEFAULT_PROB_RANK_BONUS)
    assert params["market_risk_penalty_weight"] == pytest.approx(DEFAULT_MARKET_RISK_PENALTY_WEIGHT)


def test_turf_trained_policy_maps_learned_params_to_turf_slots() -> None:
    """deployed_policy_params maps learned values to turf slots for turf policy."""
    policy = WinSelectionPolicy(surface="turf", is_trained=True)
    policy.late_odds_drop_weight = 0.10
    policy.log_odds_penalty = 0.06
    policy.prob_rank_bonus = 0.02
    policy.market_risk_penalty_weight = 0.08
    policy.training_summary = {"deployable": True}

    params = deployed_policy_params(policy)

    assert params["late_odds_drop_weight"] == pytest.approx(0.10)
    assert params["log_odds_penalty"] == pytest.approx(0.06)
    assert params["prob_rank_bonus"] == pytest.approx(0.02)
    assert params["market_risk_penalty_weight"] == pytest.approx(0.08)
    # dirt slots stay at dirt defaults
    assert params["dirt_late_odds_drop_weight"] == pytest.approx(DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT)
    assert params["dirt_log_odds_penalty"] == pytest.approx(DEFAULT_DIRT_LOG_ODDS_PENALTY)
    assert params["dirt_prob_rank_bonus"] == pytest.approx(DEFAULT_DIRT_PROB_RANK_BONUS)
    assert params["dirt_market_risk_penalty_weight"] == pytest.approx(
        DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT
    )


def test_old_joblib_load_defaults_surface_to_turf() -> None:
    """Old joblib without surface or market_risk_penalty_weight defaults safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "old_policy.joblib"
        joblib.dump(
            {
                "late_odds_drop_weight": 0.09,
                "log_odds_penalty": 0.08,
                "prob_rank_bonus": 0.01,
                "ev_tail_penalty_weight": 0.0,
                "is_trained": True,
                "training_summary": {"deployable": True},
            },
            path,
        )
        loaded = WinSelectionPolicy.load(path)

    assert loaded.surface == "turf"
    assert loaded.market_risk_penalty_weight == pytest.approx(DEFAULT_MARKET_RISK_PENALTY_WEIGHT)
    assert loaded.is_trained is True
    params = deployed_policy_params(loaded)
    assert params["late_odds_drop_weight"] == pytest.approx(0.09)


def test_dirt_policy_score_uses_learned_dirt_values() -> None:
    """Dirt-trained policy applies learned values to dirt rows in score/apply."""
    policy = WinSelectionPolicy(surface="dirt")
    policy.late_odds_drop_weight = 0.10
    policy.market_risk_penalty_weight = 0.08

    scored = policy.apply(
        pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "race_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
                "surface": ["dirt", "dirt"],
                "umaban": [1, 2],
                "kakuteijyuni": [2, 1],
                "tanodds": [2.0, 12.0],
                "p_win_final": [0.60, 0.08],
                "win_selection_prob": [0.60, 0.08],
                "win_selection_ev": [1.05, 1.30],
                "win_selection_edge": [0.05, 0.30],
                "odds_drop_rate_30_10": [0.0, 0.0],
            }
        )
    )

    assert scored["win_late_odds_drop_weight"].iloc[0] == pytest.approx(0.10)
    assert scored["win_market_risk_penalty_weight"].iloc[0] == pytest.approx(0.08)


def test_yearly_regression_rejects_log_odds_candidate() -> None:
    """A log_odds_penalty candidate that regresses in any OOF year is rejected."""
    result = pd.DataFrame(
        [
            {
                "weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "w_log": DEFAULT_LOG_ODDS_PENALTY,
                "w_prob": DEFAULT_PROB_RANK_BONUS,
                "w_risk": DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
                "roi_all": 0.92,
                "roi_mean_by_year": 0.92,
                "roi_min_by_year": 0.92,
                "n_years": 3,
                "year_rois": {"2021": 0.92, "2022": 0.92, "2023": 0.92},
            },
            {
                "weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "w_log": 0.06,
                "w_prob": DEFAULT_PROB_RANK_BONUS,
                "w_risk": DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
                "roi_all": 0.96,
                "roi_mean_by_year": 0.96,
                "roi_min_by_year": 0.91,
                "n_years": 3,
                "year_rois": {"2021": 1.01, "2022": 0.96, "2023": 0.91},
            },
            {
                "weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "w_log": DEFAULT_LOG_ODDS_PENALTY,
                "w_prob": 0.02,
                "w_risk": DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
                "roi_all": 0.98,
                "roi_mean_by_year": 0.98,
                "roi_min_by_year": 0.97,
                "n_years": 3,
                "year_rois": {"2021": 0.99, "2022": 0.97, "2023": 0.98},
            },
        ]
    )

    annotated = _annotate_policy_deployability(
        result,
        default_row=result.iloc[0],
        default_late_weight=DEFAULT_LATE_ODDS_DROP_WEIGHT,
        default_log_odds=DEFAULT_LOG_ODDS_PENALTY,
        default_prob_rank=DEFAULT_PROB_RANK_BONUS,
        default_market_risk=DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
    )

    bad_log = annotated.loc[annotated["w_log"].eq(0.06)].iloc[0]
    assert bad_log["min_year_delta_vs_default"] == pytest.approx(-0.01)
    assert bool(bad_log["deployable_candidate"]) is False

    good_stable = annotated.loc[annotated["w_prob"].eq(0.02)].iloc[0]
    assert good_stable["min_year_delta_vs_default"] == pytest.approx(0.05)
    assert bool(good_stable["deployable_candidate"]) is True


def test_one_per_race_invariant_with_learned_coefficients() -> None:
    """Policy always selects exactly 1 top-1 per race, regardless of coefficients."""
    policy = WinSelectionPolicy()
    rows: list[dict[str, object]] = []
    for race_id in ["R1", "R2", "R3"]:
        for umaban in range(1, 6):
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": pd.Timestamp("2022-01-01"),
                    "umaban": umaban,
                    "kakuteijyuni": 6 - umaban,
                    "tanodds": float(umaban * 2),
                    "p_win_final": 0.3,
                    "win_selection_prob": 0.3,
                    "win_selection_ev": 1.2,
                    "win_selection_edge": 0.2,
                    "odds_drop_rate_30_10": 0.0,
                }
            )
    scored = policy.apply(pd.DataFrame(rows))

    for race_id in ["R1", "R2", "R3"]:
        race_scores = scored.loc[scored["race_id"] == race_id]
        assert int((race_scores["selected_rank_by_win_market_score"] == 1).sum()) == 1


def test_score_ignores_confirmed_odds() -> None:
    """confirmed_odds must not affect the ranking score (prevents leakage)."""
    policy = WinSelectionPolicy()
    base_rows = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "race_date": [
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2022-01-01"),
            ],
            "umaban": [1, 2],
            "tanodds": [5.0, 3.0],
            "p_win_final": [0.20, 0.30],
            "win_selection_prob": [0.20, 0.30],
            "win_selection_ev": [1.2, 1.1],
            "win_selection_edge": [0.2, 0.1],
            "odds_drop_rate_30_10": [0.0, 0.0],
        }
    )

    rows_with_confirmed = base_rows.copy()
    rows_with_confirmed["confirmed_odds"] = [999.0, 1.0]

    score_base = policy.score(base_rows)
    score_confirmed = policy.score(rows_with_confirmed)

    pd.testing.assert_series_equal(score_base, score_confirmed)


def test_sanitize_market_risk_penalty_weight_clamps_oob() -> None:
    """Out-of-range market_risk_penalty_weight falls back to default."""
    assert sanitize_market_risk_penalty_weight(0.05) == pytest.approx(0.05)
    assert sanitize_market_risk_penalty_weight(-1.0) == pytest.approx(
        DEFAULT_MARKET_RISK_PENALTY_WEIGHT
    )
    assert sanitize_market_risk_penalty_weight(
        MAX_DEPLOYED_MARKET_RISK_PENALTY_WEIGHT + 1.0
    ) == pytest.approx(DEFAULT_MARKET_RISK_PENALTY_WEIGHT)
    assert sanitize_market_risk_penalty_weight(None) == pytest.approx(
        DEFAULT_MARKET_RISK_PENALTY_WEIGHT
    )


# ── Dirt joint cap evaluation tests ───────────────────────────────────


def _dirt_joint_rows() -> pd.DataFrame:
    """Dirt training data with 3 OOF years where cap-aware selection helps.

    Pattern: each race has a high-score high-odds horse (kakuteijyuni != 1)
    and a lower-score moderate-odds winner. Without cap, the high-odds
    horse is selected and loses. With cap, the winner is selected.
    """
    rows: list[dict[str, object]] = []
    race_no = 0
    for year in [2021, 2022, 2023]:
        for month in range(1, 7):
            race_no += 1
            race_id = f"{year}{month:02d}{race_no:08d}"
            race_date = pd.Timestamp(year=year, month=month, day=1)
            rows.extend(
                [
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "surface": "dirt",
                        "umaban": 1,
                        "kakuteijyuni": 2,
                        "tanodds": 60.0,  # high odds, selected without cap
                        "p_win_final": 0.04,
                        "win_selection_prob": 0.04,
                        "win_selection_ev": 2.40,
                        "win_selection_edge": 1.40,
                        "odds_drop_rate_30_10": 0.0,
                        "win_return_unit": 0.0,
                    },
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "surface": "dirt",
                        "umaban": 2,
                        "kakuteijyuni": 1,
                        "tanodds": 5.0,  # moderate odds, within cap, winner
                        "p_win_final": 0.20,
                        "win_selection_prob": 0.20,
                        "win_selection_ev": 1.00,
                        "win_selection_edge": 0.00,
                        "odds_drop_rate_30_10": 0.0,
                        "win_return_unit": 5.0,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_dirt_joint_candidate_selection() -> None:
    """Dirt training activates joint (coefficient, cap) evaluation.

    Verifies: joint evaluation activated, baseline cap is finite, candidate
    count includes cap dimension, and non-deployable ⇒ recommended cap is None.
    Deployment at +0.02 is covered by dedicated annotation tests.
    """
    policy = WinSelectionPolicy()
    policy.train(_dirt_joint_rows())

    assert policy.is_trained is True
    assert policy.surface == "dirt"
    summary = policy.training_summary
    assert summary["is_dirt_joint"] is True
    # Baseline cap is always set in dirt joint mode
    assert summary["baseline_odds_cap"] is not None
    assert summary["baseline_odds_cap"] != float("inf")
    # Candidates count includes cap dimension
    assert summary["candidate_count"] >= 7  # at least one coeff set × 7 caps
    # Synthetic data may not produce a deployable changed candidate;
    # when deployable is False, recommended cap must be None.
    if not summary["deployable"]:
        assert summary["recommended_odds_cap"] is None


def test_dirt_joint_rejects_yearly_regression() -> None:
    """Dirt joint evaluation rejects (coeff, cap) pairs that regress in any OOF year.

    A candidate that loses to baseline in any year should not be deployable,
    even with a large mean improvement. In dirt joint mode, the tail shrinkage
    path uses the same 0.02 threshold, so a 0.01 mean candidate is also rejected.
    """
    result = pd.DataFrame(
        [
            {
                "weight": DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "w_log": DEFAULT_DIRT_LOG_ODDS_PENALTY,
                "w_prob": DEFAULT_DIRT_PROB_RANK_BONUS,
                "w_risk": DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
                "odds_cap": 50.0,
                "roi_all": 0.90,
                "roi_mean_by_year": 0.90,
                "roi_min_by_year": 0.90,
                "n_years": 3,
                "year_rois": {"2021": 0.90, "2022": 0.90, "2023": 0.90},
            },
            {
                "weight": 0.12,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "w_log": DEFAULT_DIRT_LOG_ODDS_PENALTY,
                "w_prob": DEFAULT_DIRT_PROB_RANK_BONUS,
                "w_risk": DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
                "odds_cap": 30.0,
                "roi_all": 0.96,
                "roi_mean_by_year": 0.96,
                "roi_min_by_year": 0.89,
                "n_years": 3,
                "year_rois": {"2021": 1.05, "2022": 0.94, "2023": 0.89},
            },
            {
                "weight": DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
                "ev_tail_penalty_weight": 0.30,
                "w_log": DEFAULT_DIRT_LOG_ODDS_PENALTY,
                "w_prob": DEFAULT_DIRT_PROB_RANK_BONUS,
                "w_risk": DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
                "odds_cap": 50.0,
                "roi_all": 0.91,
                "roi_mean_by_year": 0.91,
                "roi_min_by_year": 0.905,
                "n_years": 3,
                "year_rois": {"2021": 0.915, "2022": 0.91, "2023": 0.905},
            },
        ]
    )

    annotated = _annotate_policy_deployability(
        result,
        default_row=result.iloc[0],
        default_late_weight=DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
        default_log_odds=DEFAULT_DIRT_LOG_ODDS_PENALTY,
        default_prob_rank=DEFAULT_DIRT_PROB_RANK_BONUS,
        default_market_risk=DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
        default_odds_cap=50.0,
        is_dirt_joint=True,
    )

    # Candidate with year regression (2023: 0.89 < 0.90) should be rejected
    risky = annotated.loc[annotated["odds_cap"].eq(30.0)].iloc[0]
    assert bool(risky["deployable_candidate"]) is False

    # Tail shrinkage candidate with mean_delta=0.01 < 0.02 is NOT deployable
    # in dirt joint mode (unified 0.02 threshold)
    stable_tail = annotated.loc[annotated["ev_tail_penalty_weight"].eq(0.30)].iloc[0]
    assert stable_tail["mean_delta_vs_default"] == pytest.approx(0.01)
    assert bool(stable_tail["deployable_candidate"]) is False
    # Rejection is from mean threshold, not the year ROI floor
    assert stable_tail["roi_min_by_year"] >= MIN_DIRT_JOINT_YEAR_ROI_FLOOR


def test_recommended_cap_persistence_backward_compat() -> None:
    """Save/load preserves recommended_odds_cap; old models default to None."""
    # New model with cap
    policy = WinSelectionPolicy(surface="dirt", is_trained=True)
    policy.late_odds_drop_weight = 0.03
    policy.training_summary = {
        "surface": "dirt",
        "deployable": True,
        "recommended_odds_cap": 50.0,
        "baseline_odds_cap": 75.0,
        "is_dirt_joint": True,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dirt_policy.joblib"
        policy.save(path)
        loaded = WinSelectionPolicy.load(path)

    assert loaded.surface == "dirt"
    assert loaded.training_summary["recommended_odds_cap"] == 50.0
    assert loaded.training_summary["baseline_odds_cap"] == 75.0
    params = deployed_policy_params(loaded)
    assert params["recommended_odds_cap"] == 50.0

    # Old model without recommended_odds_cap
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "old_policy.joblib"
        joblib.dump(
            {
                "surface": "turf",
                "late_odds_drop_weight": 0.09,
                "log_odds_penalty": 0.08,
                "prob_rank_bonus": 0.01,
                "ev_tail_penalty_weight": 0.0,
                "market_risk_penalty_weight": 0.10,
                "is_trained": True,
                "training_summary": {"deployable": True, "surface": "turf"},
            },
            path,
        )
        loaded = WinSelectionPolicy.load(path)

    assert loaded.training_summary.get("recommended_odds_cap") is None
    params = deployed_policy_params(loaded)
    assert params["recommended_odds_cap"] is None


def test_pipeline_reranker_uses_policy_cap() -> None:
    """For dirt, policy's recommended_odds_cap overrides reranker's cap when deployable."""
    from models.win_top1_odds_reranker import WinTop1OddsReranker

    policy = WinSelectionPolicy(surface="dirt", is_trained=True)
    policy.training_summary = {
        "deployable": True,
        "is_dirt_joint": True,
        "recommended_odds_cap": 50.0,
        "baseline_odds_cap": 75.0,
    }

    reranker = WinTop1OddsReranker()
    reranker.selected_cap = 100.0
    reranker.is_trained = True
    reranker.training_summary = {"selected_cap": 100.0}

    # Simulate pipeline integration (matches training_pipeline.py pattern)
    original_reranker_cap = reranker.selected_cap
    policy_summary = policy.training_summary
    policy_cap = policy_summary.get("recommended_odds_cap")
    if (
        policy_cap is not None
        and policy_summary.get("deployable") is True
        and policy_summary.get("is_dirt_joint") is True
    ):
        reranker.selected_cap = float(policy_cap)
        reranker.training_summary["cap_source"] = "joint_oof_policy_selection"
        reranker.training_summary["policy_recommended_odds_cap"] = float(policy_cap)
        reranker.training_summary["reranker_selected_cap"] = original_reranker_cap

    assert reranker.selected_cap == 50.0
    assert reranker.training_summary["cap_source"] == "joint_oof_policy_selection"
    assert reranker.training_summary["reranker_selected_cap"] == 100.0


def test_one_per_race_with_cap_fallback() -> None:
    """select_top1_with_cap always selects exactly one horse per race.

    Tests: cap-eligible, no cap-eligible (fallback), anomalous (no valid odds).
    """
    from models.win_top1_odds_reranker import select_top1_with_cap_indices

    df = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2", "R3", "R3", "R4", "R4"],
            "umaban": [1, 2, 1, 2, 1, 2, 1, 2],
            "tanodds": [3.0, 8.0, 60.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    score = pd.Series([10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0], index=df.index)

    best_idx = select_top1_with_cap_indices(
        df, cap=10.0, score=score, odds=df["tanodds"], race_key=df["race_id"]
    )

    # R1: both within cap, picks score 10 (umaban=1)
    assert df.loc[best_idx["R1"], "umaban"] == 1
    # R2: only umaban=2 within cap, picks it
    assert df.loc[best_idx["R2"], "umaban"] == 2
    # R3: no valid odds, fallback to all → score 10 (umaban=1)
    assert df.loc[best_idx["R3"], "umaban"] == 1
    # R4: no valid odds, same
    assert df.loc[best_idx["R4"], "umaban"] == 1
    assert len(best_idx) == 4


def test_turf_training_no_joint_cap() -> None:
    """Turf training never activates joint cap evaluation."""
    policy = WinSelectionPolicy()
    policy.train(_training_rows())

    assert policy.surface == "turf"
    summary = policy.training_summary
    assert summary.get("is_dirt_joint") is False
    assert summary.get("recommended_odds_cap") is None
    assert summary.get("baseline_odds_cap") is None


def test_confirmed_odds_no_score_leakage_with_cap() -> None:
    """confirmed_odds must not affect cap-aware ROI selection or score."""
    from models.win_top1_odds_reranker import select_top1_with_cap_indices

    base_rows = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "umaban": [1, 2],
            "tanodds": [5.0, 3.0],
            "kakuteijyuni": [2, 1],
        }
    )
    score = pd.Series([1.0, 0.5], index=base_rows.index)

    # Without confirmed_odds
    best_idx_a = select_top1_with_cap_indices(
        base_rows,
        cap=10.0,
        score=score,
        odds=base_rows["tanodds"],
        race_key=base_rows["race_id"],
    )

    # With confirmed_odds (should not change selection)
    rows_with_confirmed = base_rows.copy()
    rows_with_confirmed["confirmed_odds"] = [999.0, 1.0]
    best_idx_b = select_top1_with_cap_indices(
        rows_with_confirmed,
        cap=10.0,
        score=score,
        odds=rows_with_confirmed["tanodds"],
        race_key=rows_with_confirmed["race_id"],
    )

    pd.testing.assert_series_equal(best_idx_a, best_idx_b)

    # Also verify score() is unaffected
    policy = WinSelectionPolicy()
    rows_for_score = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "race_date": [pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-01")],
            "surface": ["turf", "turf"],
            "umaban": [1, 2],
            "tanodds": [5.0, 3.0],
            "p_win_final": [0.20, 0.30],
            "win_selection_prob": [0.20, 0.30],
            "win_selection_ev": [1.2, 1.1],
            "win_selection_edge": [0.2, 0.1],
            "odds_drop_rate_30_10": [0.0, 0.0],
        }
    )
    score_base = policy.score(rows_for_score)

    rows_for_score["confirmed_odds"] = [999.0, 1.0]
    score_confirmed = policy.score(rows_for_score)

    pd.testing.assert_series_equal(score_base, score_confirmed)


def test_roi_from_selected_prefers_win_return_unit() -> None:
    """_roi_from_selected uses win_return_unit priority correctly."""
    selected = pd.DataFrame(
        {
            "race_id": ["R1", "R2"],
            "umaban": [1, 1],
            "kakuteijyuni": [1, 1],
            "tanodds": [50.0, 40.0],
            "confirmed_odds": [3.0, 4.0],
            "win_return_unit": [3.0, 4.0],
        }
    )
    assert _roi_from_selected(selected) == pytest.approx(3.5)


def test_roi_for_score_with_cap_selects_within_cap() -> None:
    """_roi_for_score_with_cap selects winner within cap, not highest score."""
    rows = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "umaban": [1, 2, 1, 2],
            "kakuteijyuni": [2, 1, 2, 1],
            "tanodds": [60.0, 5.0, 80.0, 4.0],
            "win_return_unit": [0.0, 5.0, 0.0, 4.0],
        }
    )
    score = pd.Series([10.0, 5.0, 10.0, 5.0], index=rows.index)

    # Without cap: selects high-score high-odds losers → ROI 0
    roi_no_cap = _roi_for_score(rows, score)
    assert roi_no_cap == pytest.approx(0.0)

    # With cap=10: selects moderate-odds winners → positive ROI
    roi_with_cap = _roi_for_score_with_cap(rows, score, cap=10.0)
    assert roi_with_cap > 0.0
    assert roi_with_cap == pytest.approx(4.5)  # (5.0 + 4.0) / 2


# ── Dirt joint deployability threshold tests (0.02) ──────────────────────


def _dirt_joint_deploy_result(
    *,
    candidate_mean: float,
    candidate_year_rois: dict[str, float],
    candidate_weight: float = 0.12,
    candidate_odds_cap: float = 30.0,
    baseline_mean: float = 0.90,
    baseline_year_rois: dict[str, float] | None = None,
    candidate_roi_all: float = 0.92,
    candidate_roi_min: float = 0.91,
) -> pd.DataFrame:
    """Helper: build a 2-row joint result (baseline + candidate)."""
    if baseline_year_rois is None:
        baseline_year_rois = {"2021": 0.90, "2022": 0.90, "2023": 0.90}
    return pd.DataFrame(
        [
            {
                "weight": DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "w_log": DEFAULT_DIRT_LOG_ODDS_PENALTY,
                "w_prob": DEFAULT_DIRT_PROB_RANK_BONUS,
                "w_risk": DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
                "odds_cap": 50.0,
                "roi_all": baseline_mean,
                "roi_mean_by_year": baseline_mean,
                "roi_min_by_year": baseline_mean,
                "n_years": 3,
                "year_rois": baseline_year_rois,
            },
            {
                "weight": candidate_weight,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "w_log": DEFAULT_DIRT_LOG_ODDS_PENALTY,
                "w_prob": DEFAULT_DIRT_PROB_RANK_BONUS,
                "w_risk": DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
                "odds_cap": candidate_odds_cap,
                "roi_all": candidate_roi_all,
                "roi_mean_by_year": candidate_mean,
                "roi_min_by_year": candidate_roi_min,
                "n_years": 3,
                "year_rois": candidate_year_rois,
            },
        ]
    )


def _dirt_joint_annotate(result: pd.DataFrame) -> pd.DataFrame:
    return _annotate_policy_deployability(
        result,
        default_row=result.iloc[0],
        default_late_weight=DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
        default_log_odds=DEFAULT_DIRT_LOG_ODDS_PENALTY,
        default_prob_rank=DEFAULT_DIRT_PROB_RANK_BONUS,
        default_market_risk=DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
        default_odds_cap=50.0,
        is_dirt_joint=True,
    )


def test_dirt_joint_deployable_mean_0_02_all_years_positive() -> None:
    """Dirt joint candidate with mean +0.02 and all years nonnegative deploys."""
    result = _dirt_joint_deploy_result(
        candidate_mean=0.92,
        candidate_year_rois={"2021": 0.93, "2022": 0.92, "2023": 0.91},
    )
    annotated = _dirt_joint_annotate(result)
    candidate = annotated.iloc[1]
    assert candidate["mean_delta_vs_default"] == pytest.approx(0.02)
    assert candidate["min_year_delta_vs_default"] == pytest.approx(0.01)
    assert bool(candidate["deployable_candidate"]) is True


def test_dirt_joint_not_deployable_mean_0_019() -> None:
    """Dirt joint candidate with mean +0.019 does NOT deploy (needs >= 0.02)."""
    result = _dirt_joint_deploy_result(
        candidate_mean=0.919,
        candidate_year_rois={"2021": 0.928, "2022": 0.919, "2023": 0.91},
    )
    annotated = _dirt_joint_annotate(result)
    candidate = annotated.iloc[1]
    assert candidate["mean_delta_vs_default"] == pytest.approx(0.019)
    assert bool(candidate["deployable_candidate"]) is False


def test_dirt_joint_not_deployable_negative_year_delta() -> None:
    """Dirt joint candidate that regresses in any year does NOT deploy.

    Even with a large mean improvement (0.04), a single negative year delta
    is rejected by the no-year-regression guard.
    """
    result = _dirt_joint_deploy_result(
        candidate_mean=0.94,
        candidate_year_rois={"2021": 1.00, "2022": 0.94, "2023": 0.88},
    )
    annotated = _dirt_joint_annotate(result)
    candidate = annotated.iloc[1]
    assert candidate["mean_delta_vs_default"] == pytest.approx(0.04)
    assert candidate["min_year_delta_vs_default"] == pytest.approx(-0.02)
    assert bool(candidate["deployable_candidate"]) is False


def test_dirt_joint_deployable_with_tail_shrinkage() -> None:
    """Dirt joint candidate using tail shrinkage with mean >= 0.02 deploys.

    Unlike legacy mode where tail shrinkage uses 0.005, dirt joint uses
    the same 0.02 threshold for all candidates.
    """
    result = pd.DataFrame(
        [
            {
                "weight": DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
                "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
                "w_log": DEFAULT_DIRT_LOG_ODDS_PENALTY,
                "w_prob": DEFAULT_DIRT_PROB_RANK_BONUS,
                "w_risk": DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
                "odds_cap": 50.0,
                "roi_all": 0.90,
                "roi_mean_by_year": 0.90,
                "roi_min_by_year": 0.90,
                "n_years": 3,
                "year_rois": {"2021": 0.90, "2022": 0.90, "2023": 0.90},
            },
            {
                "weight": DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
                "ev_tail_penalty_weight": 0.30,
                "w_log": DEFAULT_DIRT_LOG_ODDS_PENALTY,
                "w_prob": DEFAULT_DIRT_PROB_RANK_BONUS,
                "w_risk": DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
                "odds_cap": 50.0,
                "roi_all": 0.93,
                "roi_mean_by_year": 0.93,
                "roi_min_by_year": 0.92,
                "n_years": 3,
                "year_rois": {"2021": 0.94, "2022": 0.93, "2023": 0.92},
            },
        ]
    )
    annotated = _dirt_joint_annotate(result)
    tail_candidate = annotated.iloc[1]
    assert tail_candidate["mean_delta_vs_default"] == pytest.approx(0.03)
    assert bool(tail_candidate["deployable_candidate"]) is True


def test_non_deployable_dirt_joint_policy_recommended_cap_is_none() -> None:
    """Non-deployable dirt joint policy returns None for recommended_odds_cap.

    deployed_policy_params falls back to defaults (including cap=None) when
    deployable is False.
    """
    policy = WinSelectionPolicy(surface="dirt", is_trained=True)
    policy.training_summary = {
        "deployable": False,
        "is_dirt_joint": True,
        "recommended_odds_cap": None,
        "baseline_odds_cap": 50.0,
    }
    params = deployed_policy_params(policy)
    assert params["recommended_odds_cap"] is None


def test_pipeline_reranker_no_override_when_not_deployable() -> None:
    """Pipeline does NOT override reranker cap when policy is not deployable.

    Even if is_dirt_joint is True, a non-deployable policy must not push
    a cap override to the reranker.
    """
    from models.win_top1_odds_reranker import WinTop1OddsReranker

    policy = WinSelectionPolicy(surface="dirt", is_trained=True)
    policy.training_summary = {
        "deployable": False,
        "is_dirt_joint": True,
        "recommended_odds_cap": None,
        "baseline_odds_cap": 50.0,
    }

    reranker = WinTop1OddsReranker()
    reranker.selected_cap = 100.0
    reranker.is_trained = True
    reranker.training_summary = {"selected_cap": 100.0}

    # Simulate pipeline integration (matches training_pipeline.py pattern)
    policy_summary = policy.training_summary
    policy_cap = policy_summary.get("recommended_odds_cap")
    if (
        policy_cap is not None
        and policy_summary.get("deployable") is True
        and policy_summary.get("is_dirt_joint") is True
    ):
        reranker.selected_cap = float(policy_cap)
        reranker.training_summary["cap_source"] = "joint_oof_policy_selection"

    # Should NOT have been overridden
    assert reranker.selected_cap == 100.0
    assert "cap_source" not in reranker.training_summary


def test_pipeline_reranker_no_override_when_not_dirt_joint() -> None:
    """Pipeline does NOT override when is_dirt_joint is False (e.g. turf)."""
    from models.win_top1_odds_reranker import WinTop1OddsReranker

    policy = WinSelectionPolicy(surface="turf", is_trained=True)
    policy.training_summary = {
        "deployable": True,
        "is_dirt_joint": False,
        "recommended_odds_cap": None,
    }

    reranker = WinTop1OddsReranker()
    reranker.selected_cap = 75.0
    reranker.is_trained = True
    reranker.training_summary = {"selected_cap": 75.0}

    policy_summary = policy.training_summary
    policy_cap = policy_summary.get("recommended_odds_cap")
    if (
        policy_cap is not None
        and policy_summary.get("deployable") is True
        and policy_summary.get("is_dirt_joint") is True
    ):
        reranker.selected_cap = float(policy_cap)

    assert reranker.selected_cap == 75.0


def test_dirt_joint_train_records_deploy_threshold() -> None:
    """Dirt joint training records deploy_mean_roi_threshold in training_summary."""
    policy = WinSelectionPolicy()
    policy.train(_dirt_joint_rows())

    assert policy.training_summary.get("deploy_mean_roi_threshold") == pytest.approx(
        MIN_DIRT_JOINT_MEAN_ROI_IMPROVEMENT
    )
    assert policy.training_summary["is_dirt_joint"] is True


def test_dirt_joint_full_grid_cross_combination_coverage() -> None:
    """Non-default w_log + non-default w_tail combinations are evaluable.

    The staged search (Stage 1: w_late×w_tail, Stage 2: coordinate descent
    over w_log/w_prob/w_risk) only pairs alternate w_log with the single
    Stage-1 winner, missing cross-combinations where non-default w_log AND
    non-default w_tail co-occur.  The grid supplement (7×6×5=210 base combos)
    ensures these are all evaluated at each cap, not just the coordinates
    independently explored by the greedy descent.
    """
    policy = WinSelectionPolicy()
    policy.train(_dirt_joint_rows())
    summary = policy.training_summary

    assert summary["is_dirt_joint"] is True

    # Full grid: 7 late_weights × 6 tail_weights × 5 log_cands = 210
    n_expected_grid = (
        len((0.0, 0.03, 0.06, 0.08, 0.09, 0.10, 0.12))  # late
        * len((0.0, 0.15, 0.30, 0.50, 0.75, 1.00))  # tail
        * len((0.0, 0.03, 0.05, 0.06, 0.08))  # log
    )
    assert n_expected_grid == 210

    # Coefficient count must include the full grid supplement plus staged
    # additions (which may introduce different w_prob/w_risk values).
    assert summary["joint_coefficient_count"] is not None
    assert summary["joint_coefficient_count"] >= n_expected_grid

    # Total (coefficient, cap) pairs must cover all grid combos × caps.
    n_caps = 7  # 20, 30, 40, 50, 75, 100, inf
    assert summary["joint_total_pair_count"] is not None
    assert summary["joint_total_pair_count"] >= n_expected_grid * n_caps
