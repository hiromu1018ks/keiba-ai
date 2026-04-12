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
        )
        assert len(logger.horse_records) == 1
        rec = logger.horse_records[0]
        assert rec.umaban == 5
        assert rec.ev_place == pytest.approx(1.575)
        assert rec.is_bet is True

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
