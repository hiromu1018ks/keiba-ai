"""DiagnosticLogger のテスト"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from backtest.diagnostic_logger import DiagnosticLogger


class TestDiagnosticLogger:
    def test_log_race_adds_race_record(self):
        logger = DiagnosticLogger()
        logger.log_race(
            race_id="20240101010111",
            regime="AGGRESSIVE",
            ev_threshold=1.10,
            quality_passed=True,
            quality_score=0.65,
            n_candidates=3,
            n_bets=2,
        )
        assert len(logger.race_records) == 1
        rec = logger.race_records[0]
        assert rec.race_id == "20240101010111"
        assert rec.regime == "AGGRESSIVE"
        assert rec.ev_threshold == 1.10
        assert rec.quality_passed is True
        assert rec.n_candidates == 3
        assert rec.n_bets == 2

    def test_log_horse_adds_horse_record(self):
        logger = DiagnosticLogger()
        logger.log_horse(
            race_id="20240101010111",
            umaban=5,
            p_place_pred=0.35,
            e_return_place_pred=4.5,
            ev_place=1.575,
            fukuoddslow=4.2,
            is_bet=True,
            p_place_corrected=0.31,
            ev_place_corrected=1.42,
            ev_lower_place=1.18,
        )
        assert len(logger.horse_records) == 1
        rec = logger.horse_records[0]
        assert rec.umaban == 5
        assert rec.ev_place == pytest.approx(1.575)
        assert rec.is_bet is True
        assert rec.p_place_corrected == pytest.approx(0.31)
        assert rec.ev_lower_place == pytest.approx(1.18)

    def test_log_horse_records_win_diagnostics(self):
        logger = DiagnosticLogger()
        logger.log_horse(
            race_id="20240101010111",
            umaban=5,
            p_place_pred=0.0,
            e_return_place_pred=0.0,
            ev_place=0.0,
            fukuoddslow=0.0,
            is_bet=True,
            is_actual_bet=True,
            p_win_pred=0.12,
            p_win_corrected=0.10,
            p_win_final=0.09,
            e_return_win_pred=12.0,
            e_return_win_corrected=10.0,
            win_selection_ev=1.8,
            win_selection_ev_tail_calibrated=1.26,
            win_selection_edge=0.26,
            win_selection_prob=0.09,
            win_gate_score=1.2,
            win_gate_pass=True,
            win_gate_odds_score=1.08,
            win_gate_prob_score=0.92,
            win_gate_edge_score=1.03,
            win_gate_edge_odds_score=1.11,
            p_market_win_raw=0.071,
            p_market_win_norm=0.08,
            win_market_residual=0.01,
            win_market_logit_edge=0.13,
            win_market_prob_ratio=1.12,
            win_market_value_ratio=1.26,
            win_market_selection_score=0.82,
            win_late_odds_drop_z=1.5,
            win_late_odds_drop_weight=0.06,
            win_log_odds=2.708,
            win_log_odds_penalty=0.05,
            win_model_prob_rank=0.75,
            win_prob_rank_bonus=0.02,
            win_market_risk_penalty=0.20,
            risk_flags="longshot_low_probability",
            tanodds=14.0,
            closing_win_odds=13.5,
            clv=-0.0357,
            final_odds=13.8,
            stake=100.0,
            result=0.0,
            excluded_reason=None,
            filter_pass_flags="edge=True;odds=True",
            candidate_count_before_filter=3,
            candidate_count_after_filter=1,
            selected_rank_by_p_win_final=2.0,
            selected_rank_by_win_selection_ev=1.0,
            selected_rank_by_win_market_logit_edge=1.0,
            selected_rank_by_win_market_score=1.0,
        )

        rec = logger.horse_records[0]
        assert rec.is_actual_bet is True
        assert rec.p_win_final == pytest.approx(0.09)
        assert rec.win_selection_ev_tail_calibrated == pytest.approx(1.26)
        assert rec.win_gate_odds_score == pytest.approx(1.08)
        assert rec.win_market_logit_edge == pytest.approx(0.13)
        assert rec.win_late_odds_drop_z == pytest.approx(1.5)
        assert rec.win_late_odds_drop_weight == pytest.approx(0.06)
        assert rec.win_log_odds_penalty == pytest.approx(0.05)
        assert rec.win_model_prob_rank == pytest.approx(0.75)
        assert rec.risk_flags == "longshot_low_probability"
        assert rec.clv == pytest.approx(-0.0357)
        assert rec.stake == pytest.approx(100.0)

    def test_save_creates_two_csv_files(self):
        logger = DiagnosticLogger()
        logger.log_race("20240101010111", "CONSERVATIVE", 1.30, True, 0.5, 2, 1)
        logger.log_horse("20240101010111", 5, 0.35, 4.5, 1.575, 4.2, True)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            logger.save(outdir, prefix="test")

            race_path = outdir / "test_race_diagnostics.csv"
            horse_path = outdir / "test_horse_diagnostics.csv"
            assert race_path.exists()
            assert horse_path.exists()

            race_df = pd.read_csv(race_path)
            assert len(race_df) == 1
            assert "regime" in race_df.columns
            assert "ev_threshold" in race_df.columns

            horse_df = pd.read_csv(horse_path)
            assert len(horse_df) == 1
            assert "p_place_pred" in horse_df.columns
            assert "ev_place" in horse_df.columns
            assert "ev_lower_place" in horse_df.columns

    def test_save_empty_logger_creates_no_files(self):
        logger = DiagnosticLogger()
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            logger.save(outdir, prefix="empty")
            assert not (outdir / "empty_race_diagnostics.csv").exists()
            assert not (outdir / "empty_horse_diagnostics.csv").exists()

    def test_log_horse_features_adds_record(self):
        logger = DiagnosticLogger()
        logger.log_horse_features({"race_id": "20240101010111", "umaban": 5, "ev_place": 1.5})
        assert len(logger.feature_records) == 1
        assert logger.feature_records[0]["race_id"] == "20240101010111"
        assert logger.feature_records[0]["umaban"] == 5

    def test_save_creates_parquet_when_features_logged(self):
        logger = DiagnosticLogger()
        logger.log_horse_features({
            "race_id": "20240101010111",
            "umaban": 5,
            "ev_place": 1.5,
            "surface": "turf",
        })
        logger.log_horse_features({
            "race_id": "20240101010111",
            "umaban": 8,
            "ev_place": 0.9,
            "surface": "dirt",
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            logger.save(outdir, prefix="test")

            parquet_path = outdir / "test_horse_features.parquet"
            assert parquet_path.exists()

            df = pd.read_parquet(parquet_path)
            assert len(df) == 2
            assert "race_id" in df.columns
            assert "umaban" in df.columns
            assert "ev_place" in df.columns
            assert "surface" in df.columns

    def test_save_creates_no_parquet_when_no_features(self):
        logger = DiagnosticLogger()
        logger.log_race("20240101010111", "AGGRESSIVE", 1.10, True, 0.6, 3, 2)
        logger.log_horse("20240101010111", 5, 0.35, 4.5, 1.575, 4.2, True)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            logger.save(outdir, prefix="test")
            assert not (outdir / "test_horse_features.parquet").exists()

    def test_parquet_excludes_nested_types(self):
        """list/dict 値を持つ列が parquet に出力されないことを確認。"""
        logger = DiagnosticLogger()
        logger.log_horse_features({
            "race_id": "20240101010111",
            "umaban": 5,
            "top3_finishers": [{"umaban": 1}, {"umaban": 2}],
            "nested": {"key": "value"},
            "ev_place": 1.5,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            logger.save(outdir, prefix="test")

            df = pd.read_parquet(outdir / "test_horse_features.parquet")
            assert "top3_finishers" not in df.columns
            assert "nested" not in df.columns
            assert "ev_place" in df.columns
            assert "race_id" in df.columns

    def test_multiple_races_and_horses(self):
        logger = DiagnosticLogger()
        # 2 races, 3 horses each
        for i in range(2):
            rid = f"2024010101011{i}"
            logger.log_race(rid, "AGGRESSIVE", 1.10, True, 0.6, 3, 2)
            for umaban in [1, 5, 8]:
                logger.log_horse(rid, umaban, 0.3, 4.0, 1.2, 3.8, umaban == 5)

        assert len(logger.race_records) == 2
        assert len(logger.horse_records) == 6

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.save(Path(tmpdir), prefix="multi")
            race_df = pd.read_csv(Path(tmpdir) / "multi_race_diagnostics.csv")
            horse_df = pd.read_csv(Path(tmpdir) / "multi_horse_diagnostics.csv")
            assert len(race_df) == 2
            assert len(horse_df) == 6
            assert horse_df["is_bet"].sum() == 2  # 1 per race
