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
    DEFAULT_EV_TAIL_THRESHOLD,
    DEFAULT_LATE_ODDS_DROP_WEIGHT,
    DEFAULT_LOG_ODDS_PENALTY,
    DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
    DEFAULT_PROB_RANK_BONUS,
    MAX_DEPLOYED_MARKET_RISK_PENALTY_WEIGHT,
    WinSelectionPolicy,
    _annotate_policy_deployability,
    _roi_for_score,
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
    assert deployed_policy_params(policy) == {
        "late_odds_drop_weight": DEFAULT_LATE_ODDS_DROP_WEIGHT,
        "log_odds_penalty": DEFAULT_LOG_ODDS_PENALTY,
        "prob_rank_bonus": DEFAULT_PROB_RANK_BONUS,
        "ev_tail_penalty_weight": DEFAULT_EV_TAIL_PENALTY_WEIGHT,
        "ev_tail_threshold": DEFAULT_EV_TAIL_THRESHOLD,
        "market_risk_penalty_weight": DEFAULT_MARKET_RISK_PENALTY_WEIGHT,
        "dirt_late_odds_drop_weight": DEFAULT_DIRT_LATE_ODDS_DROP_WEIGHT,
        "dirt_log_odds_penalty": DEFAULT_DIRT_LOG_ODDS_PENALTY,
        "dirt_prob_rank_bonus": DEFAULT_DIRT_PROB_RANK_BONUS,
        "dirt_market_risk_penalty_weight": DEFAULT_DIRT_MARKET_RISK_PENALTY_WEIGHT,
    }


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
