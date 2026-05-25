"""WinProfitSelector tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _make_profit_rows(n_races: int = 180) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for race_idx in range(n_races):
        race_id = f"R{race_idx:04d}"
        race_date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=race_idx)
        rows.extend(
            [
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": 1,
                    "kakuteijyuni": 1 if race_idx % 5 == 0 else 3,
                    "tanodds": 2.0,
                    "win_selection_prob": 0.45,
                    "win_selection_ev": 0.90,
                    "win_selection_edge": -0.10,
                    "win_market_selection_score": 1.00,
                },
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": 2,
                    "kakuteijyuni": 1 if race_idx % 4 == 0 else 4,
                    "tanodds": 12.0,
                    "win_selection_prob": 0.10,
                    "win_selection_ev": 1.20,
                    "win_selection_edge": 0.20,
                    "win_market_selection_score": 0.80,
                },
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": 3,
                    "kakuteijyuni": 6,
                    "tanodds": 30.0,
                    "win_selection_prob": 0.02,
                    "win_selection_ev": 0.60,
                    "win_selection_edge": -0.40,
                    "win_market_selection_score": 0.10,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_win_profit_selector_trains_and_scores_multiple_candidate_policy() -> None:
    from models.win_profit_selector import PROFIT_PASS_COL, PROFIT_SCORE_COL, WinProfitSelector

    model = WinProfitSelector(
        min_train_races=60,
        min_fold_races=30,
        max_folds=3,
        min_bets_per_eval_race=0.25,
    )
    model.train(_make_profit_rows())

    assert model.is_trained is True
    assert model.training_summary["min_eval_bets"] > 0

    scored = model.score(
        pd.DataFrame(
            [
                {
                    "race_id": "T1",
                    "race_date": pd.Timestamp("2025-01-01"),
                    "umaban": 1,
                    "tanodds": 2.0,
                    "win_selection_prob": 0.45,
                    "win_selection_ev": 0.90,
                    "win_selection_edge": -0.10,
                    "win_market_selection_score": 1.00,
                },
                {
                    "race_id": "T1",
                    "race_date": pd.Timestamp("2025-01-01"),
                    "umaban": 2,
                    "tanodds": 12.0,
                    "win_selection_prob": 0.10,
                    "win_selection_ev": 1.20,
                    "win_selection_edge": 0.20,
                    "win_market_selection_score": 0.80,
                },
            ]
        )
    )

    assert PROFIT_SCORE_COL in scored.columns
    assert PROFIT_PASS_COL in scored.columns
    assert scored[PROFIT_PASS_COL].any()


def test_win_profit_selector_save_load_roundtrip(tmp_path: Path) -> None:
    from models.win_profit_selector import WinProfitSelector

    model = WinProfitSelector(min_train_races=60, min_fold_races=30, max_folds=3)
    model.train(_make_profit_rows())
    assert model.is_trained

    path = tmp_path / "win_profit_selector.joblib"
    model.save(path)
    loaded = WinProfitSelector.load(path)

    assert loaded.is_trained is True
    assert loaded.params == model.params
    assert loaded.training_summary["selected_params"] == model.training_summary["selected_params"]


def test_win_profit_selector_untrained_marks_no_pass() -> None:
    from models.win_profit_selector import PROFIT_PASS_COL, WinProfitSelector

    scored = WinProfitSelector().score(
        pd.DataFrame(
            {
                "race_id": ["R1"],
                "tanodds": [3.0],
                "win_selection_prob": [0.2],
                "win_selection_ev": [1.1],
            }
        )
    )

    assert scored[PROFIT_PASS_COL].tolist() == [False]


def test_win_profit_selector_requires_volume() -> None:
    from models.win_profit_selector import WinProfitSelector

    model = WinProfitSelector(min_train_races=500, min_fold_races=100)
    model.train(_make_profit_rows(50))

    assert model.is_trained is False
    assert model.training_summary["reason"] == "insufficient_races"


def test_win_profit_selector_rejects_unprofitable_policy() -> None:
    from models.win_profit_selector import WinProfitSelector

    rows: list[dict[str, object]] = []
    for race_idx in range(180):
        race_id = f"R{race_idx:04d}"
        race_date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=race_idx)
        for umaban in range(1, 4):
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": umaban,
                    "kakuteijyuni": 2 + umaban,
                    "tanodds": 4.0 + umaban,
                    "win_selection_prob": 0.20,
                    "win_selection_ev": 0.90,
                    "win_selection_edge": 0.10,
                    "win_market_selection_score": 1.0 / umaban,
                }
            )

    model = WinProfitSelector(min_train_races=60, min_fold_races=30, max_folds=3)
    model.train(pd.DataFrame(rows))

    assert model.is_trained is False
    assert model.training_summary["reason"] == "no_deployable_profit_policy"


def test_race_predictor_uses_profit_selector_candidate_set() -> None:
    from types import SimpleNamespace

    from backtest.race_predictor import RacePredictor
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
    models = SimpleNamespace(submodels={"turf": SimpleNamespace(win_profit_selector=selector)})
    predictor = RacePredictor(models)

    candidates = predictor.get_win_candidates(
        pd.DataFrame(
            [
                {
                    "race_id": "R1",
                    "race_date": pd.Timestamp("2025-01-01"),
                    "surface": "turf",
                    "umaban": 1,
                    "tanodds": 2.0,
                    "p_win_final": 0.45,
                    "win_selection_prob": 0.45,
                    "win_selection_ev": 0.90,
                    "win_selection_edge": -0.10,
                    "win_market_selection_score": 1.00,
                },
                {
                    "race_id": "R1",
                    "race_date": pd.Timestamp("2025-01-01"),
                    "surface": "turf",
                    "umaban": 2,
                    "tanodds": 12.0,
                    "p_win_final": 0.10,
                    "win_selection_prob": 0.10,
                    "win_selection_ev": 1.20,
                    "win_selection_edge": 0.20,
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

    assert set(candidates["umaban"].tolist()) == {1, 2}
    assert candidates["win_profit_selector_pass"].all()
