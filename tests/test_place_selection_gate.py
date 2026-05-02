"""PlaceSelectionGateModel tests."""

from __future__ import annotations

import pandas as pd


def test_place_selection_gate_trains_and_scores() -> None:
    from models.place_selection_gate import PlaceSelectionGateModel

    rows: list[dict[str, object]] = []
    for race_idx in range(120):
        race_id = f"R{race_idx:04d}"
        race_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=race_idx)
        for umaban, prob, edge, odds, finish in [
            (1, 0.62, 0.24, 2.2, 2 if race_idx % 3 != 0 else 5),
            (2, 0.28, 0.02, 4.5, 4),
            (3, 0.10, -0.15, 11.0, 8),
        ]:
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "umaban": umaban,
                    "kakuteijyuni": finish,
                    "fukuoddslow": odds,
                    "place_selection_prob": prob,
                    "place_selection_edge": edge,
                }
            )

    df = pd.DataFrame(rows)
    model = PlaceSelectionGateModel(min_train_races=40, min_fold_races=20, max_folds=3)
    model.train(df)

    assert model.is_trained is True

    scored = model.score(
        pd.DataFrame(
            {
                "race_id": ["T1", "T1"],
                "race_date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
                "umaban": [1, 2],
                "fukuoddslow": [2.2, 11.0],
                "place_selection_prob": [0.60, 0.09],
                "place_selection_edge": [0.22, -0.18],
            }
        )
    )

    assert "place_gate_score" in scored.columns
    assert "place_gate_pass" in scored.columns
    assert scored.loc[0, "place_gate_score"] > scored.loc[1, "place_gate_score"]


def test_place_selection_gate_soft_pass_rescues_top_ranked_near_threshold_runner() -> None:
    from models.place_selection_gate import PlaceSelectionGateModel

    model = PlaceSelectionGateModel()
    model.min_prob = 0.38
    model.min_edge = 0.05
    model.max_odds = 10.0
    model._trained = True

    scored = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "umaban": [1, 2, 1, 2],
            "fukuoddslow": [9.8, 8.0, 5.0, 8.5],
            "place_selection_prob": [0.375, 0.31, 0.42, 0.375],
            "place_selection_edge": [0.035, 0.08, 0.07, 0.04],
            "place_gate_score": [1.4, 1.0, 1.2, 1.1],
            "place_gate_pass": [False, False, True, False],
        }
    )

    mask = model.soft_pass_mask(
        scored,
        edge_floor=0.0,
        min_prob=0.08,
        max_odds=18.0,
        max_per_race=1,
    )

    assert mask.tolist() == [True, False, False, False]

