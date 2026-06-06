"""Strategy alignment integration tests (Phase 53-01).

Tests for:
- parse_args() rejects wide betting target
- strategy_config from manifest via build_strategy_config_from_params()
- 3-way target mismatch fail-fast
- session_manifest strategy fields
- RaceQualityScreener integration in RacePredictor.should_bet() (STR-05)
- OddsBandFilter injection for win-target (Task 2)
- No OddsBandFilter for place-target (Task 2)
- OddsBandFilter 4 metadata fields in session_manifest (Task 2)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure src/ is on sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ── Task 1 Tests ──────────────────────────────────────────


class TestParseArgsRejectsWide:
    """--betting-target wide は argparse choices=["win","place"] で拒否される"""

    def test_rejects_wide_target(self) -> None:
        from scripts.run_paper_trading import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--mode", "predict", "--date", "2026-01-01",
                        "--betting-target", "wide", "--betting-mode", "flat"])

    def test_accepts_win_target(self) -> None:
        from scripts.run_paper_trading import parse_args

        args = parse_args(["--mode", "predict", "--date", "2026-01-01",
                           "--betting-target", "win", "--betting-mode", "flat"])
        assert args.betting_target == "win"

    def test_accepts_place_target(self) -> None:
        from scripts.run_paper_trading import parse_args

        args = parse_args(["--mode", "predict", "--date", "2026-01-01",
                           "--betting-target", "place", "--betting-mode", "kelly"])
        assert args.betting_target == "place"


class TestStrategyConfigFromManifest:
    """strategy_manifest_path が存在すれば JSON 読込→build_strategy_config_from_params()"""

    def test_manifest_loads_and_builds_config(self, tmp_path: Path) -> None:
        from betting.default_strategy import build_strategy_config_from_params

        manifest_data = {
            "fk_aggressive": 0.4,
            "ev_aggressive": 1.15,
            "target_ev": 1.10,
            "roi_threshold": 1.05,
        }
        manifest_path = tmp_path / "strategy_manifest.json"
        manifest_path.write_text(json.dumps(manifest_data))

        data = json.loads(manifest_path.read_text())
        config = build_strategy_config_from_params(data)
        assert config["fractional_kelly"] == 0.4
        assert config["target_ev"] == 1.10
        assert config["roi_threshold"] == 1.05


class TestThreeWayTargetMismatch:
    """model/manifest/CLI の3者で betting_target が不一致なら sys.exit(1)"""

    def test_model_cli_mismatch_exits(self) -> None:
        """model_target="win", cli_target="place" → fail-fast"""
        from scripts.run_paper_trading import _validate_betting_target_alignment

        models = MagicMock()
        models.meta = {"betting_target": "win"}

        with pytest.raises(SystemExit) as exc_info:
            _validate_betting_target_alignment(
                models=models,
                manifest_target=None,
                cli_target="place",
            )
        assert exc_info.value.code == 1

    def test_all_match_passes(self) -> None:
        """All three targets match → no error"""
        from scripts.run_paper_trading import _validate_betting_target_alignment

        models = MagicMock()
        models.meta = {"betting_target": "win"}

        _validate_betting_target_alignment(
            models=models,
            manifest_target="win",
            cli_target="win",
        )
        # Should not raise

    def test_manifest_cli_mismatch_exits(self) -> None:
        """manifest_target="place", cli_target="win" → fail-fast"""
        from scripts.run_paper_trading import _validate_betting_target_alignment

        models = MagicMock()
        models.meta = {"betting_target": "win"}

        with pytest.raises(SystemExit) as exc_info:
            _validate_betting_target_alignment(
                models=models,
                manifest_target="place",
                cli_target="win",
            )
        assert exc_info.value.code == 1


class TestSessionManifestStrategyFields:
    """session_manifest.set_strategy_params() で4フィールドが記録される"""

    def test_set_strategy_params_records_fields(self) -> None:
        from features.session_manifest import SessionManifest

        manifest = SessionManifest(
            session_id="test123",
            prediction_date="2026-06-01",
        )
        manifest.set_strategy_params(
            betting_target="win",
            betting_mode="flat",
            strategy_manifest_path="/data/strategy_manifest.json",
            strategy_manifest_sha256="abc123",
        )

        data = manifest.to_dict()
        assert data["betting_target"] == "win"
        assert data["betting_mode"] == "flat"
        assert data["strategy_manifest_path"] == "/data/strategy_manifest.json"
        assert data["strategy_manifest_sha256"] == "abc123"


class TestRaceQualityScreenerIntegrated:
    """RacePredictor.should_bet() が内部で RaceQualityScreener.should_bet() を呼び出す (STR-05)"""

    def test_should_bet_calls_screener(self) -> None:
        from backtest.race_predictor import RacePredictor

        mock_screener = MagicMock()
        mock_screener.should_bet.return_value = True

        mock_models = MagicMock()
        mock_models.quality_screener = mock_screener
        mock_models.submodels = {"turf": MagicMock()}
        mock_models.regime_detector = MagicMock()

        predictor = RacePredictor(mock_models, betting_target="place")
        race_df = pd.DataFrame({
            "race_id": ["20260601A0010011"],
            "umaban": [1],
            "surface": ["turf"],
        })

        result = predictor.should_bet(race_df)

        assert result is True
        mock_screener.should_bet.assert_called_once()


# ── Task 2 Tests ──────────────────────────────────────────


class TestRacePredictorOddsBandFilterWin:
    """OddsBandFilter 注入で select_bets() がフィルタを適用する"""

    def test_odds_band_filter_applied_in_select_bets(self) -> None:
        from backtest.race_predictor import RacePredictor
        from betting.odds_band_filter import OddsBandFilter

        mock_models = MagicMock()
        mock_models.submodels = {"turf": MagicMock()}
        mock_models.regime_detector = MagicMock()
        mock_models.regime_detector.get_strategy_params.return_value = {
            "max_bets_per_race": 1,
        }
        mock_models.quality_screener = MagicMock()
        mock_models.quality_screener.should_bet.return_value = True

        obf = MagicMock(spec=OddsBandFilter)
        obf.filter.side_effect = lambda df, **kwargs: df

        predictor = RacePredictor(
            mock_models,
            betting_target="win",
            odds_band_filter=obf,
        )

        # Provide a candidates DataFrame with required columns
        candidates = pd.DataFrame({
            "race_id": ["20260601A0010011"],
            "umaban": [1],
            "tanodds": [5.0],
            "win_selection_ev": [1.5],
            "win_selection_edge": [0.5],
            "win_selection_ev_tail_calibrated": [1.5],
        })
        # Mock get_win_candidates to return our test candidates
        with patch.object(predictor, "get_win_candidates", return_value=candidates):
            predictor.select_bets(
                pd.DataFrame({"race_id": ["20260601A0010011"], "umaban": [1], "surface": ["turf"]}),
                bankroll=100000,
                betting_target="win",
            )

        # OddsBandFilter.filter() should have been called
        obf.filter.assert_called_once()


class TestRacePredictorNoOddsBandFilterPlace:
    """place-target では OddsBandFilter が None で filter が呼ばれない"""

    def test_no_odds_band_filter_for_place(self) -> None:
        from backtest.race_predictor import RacePredictor
        from betting.odds_band_filter import OddsBandFilter

        mock_models = MagicMock()
        mock_models.submodels = {"turf": MagicMock()}
        mock_models.regime_detector = MagicMock()
        mock_models.regime_detector.get_strategy_params.return_value = {
            "max_bets_per_race": 3,
        }
        mock_models.quality_screener = MagicMock()

        obf = MagicMock(spec=OddsBandFilter)

        predictor = RacePredictor(
            mock_models,
            betting_target="place",
            odds_band_filter=obf,
        )

        # place candidates — the OddsBandFilter should NOT be called
        candidates = pd.DataFrame({
            "race_id": ["20260601A0010011"],
            "umaban": [1],
            "fukuoddslow": [2.5],
            "place_selection_ev": [1.2],
            "place_selection_edge": [0.2],
            "place_selection_prob": [0.5],
            "place_selection_reason": ["threshold"],
        })
        with patch.object(predictor, "get_place_candidates", return_value=candidates):
            predictor.select_bets(
                pd.DataFrame({"race_id": ["20260601A0010011"], "umaban": [1], "surface": ["turf"]}),
                bankroll=100000,
                betting_target="place",
            )

        obf.filter.assert_not_called()


class TestOBFMetadataFourFields:
    """set_obf_metadata() で4フィールドが session_manifest に記録される"""

    def test_obf_metadata_fields(self) -> None:
        from features.session_manifest import SessionManifest

        manifest = SessionManifest(
            session_id="test456",
            prediction_date="2026-06-01",
        )
        manifest.set_obf_metadata(
            calibration_data_end_date="2024-12-31",
            roi_threshold=1.05,
            excluded_bands={"10.0-30.0", "30.0+"},
            config_hash="sha256hash123",
        )

        data = manifest.to_dict()
        obf = data["odds_band_filter_metadata"]
        assert obf["calibration_data_end_date"] == "2024-12-31"
        assert obf["roi_threshold"] == 1.05
        assert obf["excluded_bands"] == ["10.0-30.0", "30.0+"]
        assert obf["config_hash"] == "sha256hash123"
