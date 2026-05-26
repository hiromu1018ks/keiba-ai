"""WinSegmentCalibrator tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _make_segment_rows(n_races: int = 160) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for race_idx in range(n_races):
        rows.append(
            {
                "race_id": f"R{race_idx:04d}",
                "race_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=race_idx),
                "surface": "turf",
                "umaban": 1,
                "kakuteijyuni": 1 if race_idx % 5 == 0 else 2,
                "tanodds": 5.0,
                "confirmed_odds": 5.0,
                "p_win_final_oof": 0.40,
                "p_win_final": 0.40,
                "win_selection_prob": 0.40,
                "win_selection_ev": 1.20,
                "win_selection_edge": 0.20,
            }
        )
    return pd.DataFrame(rows)


def test_win_segment_calibrator_trains_turf_shrink_only() -> None:
    from models.win_segment_calibrator import WinSegmentCalibrator

    model = WinSegmentCalibrator()
    model.train(_make_segment_rows())

    assert model.is_trained is True
    assert model.training_summary["target_surface"] == "turf"
    assert model.training_summary["n_deployed_segments"] > 0
    assert model.training_summary["min_p_factor"] < 1.0

    scored = model.apply(
        pd.DataFrame(
            [
                {
                    "race_id": "T1",
                    "surface": "turf",
                    "umaban": 1,
                    "tanodds": 5.0,
                    "p_win_final": 0.40,
                    "win_selection_prob": 0.40,
                    "win_selection_ev": 1.20,
                    "win_selection_edge": 0.20,
                },
                {
                    "race_id": "D1",
                    "surface": "dirt",
                    "umaban": 1,
                    "tanodds": 5.0,
                    "p_win_final": 0.40,
                    "win_selection_prob": 0.40,
                    "win_selection_ev": 1.20,
                    "win_selection_edge": 0.20,
                },
            ]
        )
    )

    turf_row = scored.loc[scored["surface"].eq("turf")].iloc[0]
    dirt_row = scored.loc[scored["surface"].eq("dirt")].iloc[0]
    assert 0.85 <= turf_row["win_segment_prob_factor"] < 1.0
    assert turf_row["p_win_final"] < 0.40
    assert turf_row["win_selection_prob"] < 0.40
    assert turf_row["win_selection_ev"] == 1.20
    assert turf_row["win_segment_ev_factor"] == 1.0
    assert dirt_row["win_segment_prob_factor"] == 1.0
    assert dirt_row["p_win_final"] == 0.40


def test_win_segment_calibrator_save_load_roundtrip(tmp_path: Path) -> None:
    from models.win_segment_calibrator import WinSegmentCalibrator

    model = WinSegmentCalibrator()
    model.train(_make_segment_rows())
    assert model.is_trained

    path = tmp_path / "win_segment_calibrator.joblib"
    model.save(path)
    loaded = WinSegmentCalibrator.load(path)

    assert loaded.is_trained is True
    assert loaded.segment_table == model.segment_table
    assert loaded.training_summary == model.training_summary
