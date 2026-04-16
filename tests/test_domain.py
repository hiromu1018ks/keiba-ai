"""src/domain モジュールのテスト"""

import numpy as np
import pytest

from domain.models import (
    Bet,
    DDState,
    Entry,
    OddsSnapshot,
    Race,
    RegimeConfig,
    SubmodelSet,
    TwoStageConfig,
)
from domain.types import BetType, RecoveryState, RegimeState, Surface


class TestEnums:
    def test_surface_values(self):
        assert Surface.TURF.value == "turf"
        assert Surface.DIRT.value == "dirt"

    def test_bet_type_values(self):
        assert BetType.WIN.value == "win"
        assert BetType.PLACE.value == "place"
        assert BetType.WIDE.value == "wide"

    def test_recovery_state_values(self):
        assert RecoveryState.NORMAL.value == "normal"
        assert RecoveryState.REDUCED.value == "reduced"
        assert RecoveryState.RECOVERING.value == "recovering"

    def test_regime_state_values(self):
        assert RegimeState.AGGRESSIVE.value == "aggressive"
        assert RegimeState.CONSERVATIVE.value == "conservative"
        assert RegimeState.COLLAPSED.value == "collapsed"


class TestRace:
    def test_create_race_minimal(self):
        race = Race(
            year=2024,
            month_day="0324",
            jyo_cd="05",
            kaiji="03",
            nichiji="02",
            race_num="08",
            track_cd=11,
            distance=1600,
            tenko_cd=1,
            baba_cd=1,
            syubetu_cd=13,
            jyoken_cd=999,
            grade_cd="_",
            field_size=18,
        )
        assert race.surface == Surface.TURF
        assert race.distance == 1600
        assert race.distance_band == "mile"

    def test_surface_dirt(self):
        race = Race(
            year=2024,
            month_day="0324",
            jyo_cd="05",
            kaiji="03",
            nichiji="02",
            race_num="08",
            track_cd=23,
            distance=1200,
            tenko_cd=1,
            baba_cd=1,
            syubetu_cd=13,
            jyoken_cd=999,
            grade_cd="_",
            field_size=14,
        )
        assert race.surface == Surface.DIRT

    def test_race_id_format(self):
        race = Race(
            year=2024,
            month_day="0324",
            jyo_cd="05",
            kaiji="03",
            nichiji="02",
            race_num="08",
            track_cd=11,
            distance=1600,
            tenko_cd=1,
            baba_cd=1,
            syubetu_cd=13,
            jyoken_cd=999,
            grade_cd="_",
            field_size=18,
        )
        assert race.race_id == "2024032405030208"

    def test_is_good_track(self):
        race = Race(
            year=2024,
            month_day="0324",
            jyo_cd="05",
            kaiji="03",
            nichiji="02",
            race_num="08",
            track_cd=11,
            distance=1600,
            tenko_cd=1,
            baba_cd=1,
            syubetu_cd=13,
            jyoken_cd=999,
            grade_cd="_",
            field_size=18,
        )
        assert race.is_good_track is True

    def test_is_soft_track(self):
        race = Race(
            year=2024,
            month_day="0324",
            jyo_cd="05",
            kaiji="03",
            nichiji="02",
            race_num="08",
            track_cd=11,
            distance=1600,
            tenko_cd=1,
            baba_cd=3,
            syubetu_cd=13,
            jyoken_cd=999,
            grade_cd="_",
            field_size=18,
        )
        assert race.is_good_track is False
        assert race.is_soft_track is True


class TestEntry:
    def test_create_entry(self):
        entry = Entry(
            race_id="2024032405030208",
            umaban=5,
            ketto_num="0001234567",
            finish_pos=1,
            win_odds_actual=3.2,
            popularity_rank=2,
            running_style=2,
            ba_taijyu=480,
            zogen_fugo=2,
            zogen_sa=-4,
            kisyu_code="01056",
            chokyosi_code="01023",
        )
        assert entry.is_winner is True
        assert entry.is_place is True

    def test_entry_not_winner(self):
        entry = Entry(
            race_id="2024032405030208",
            umaban=5,
            ketto_num="0001234567",
            finish_pos=4,
            win_odds_actual=15.8,
            popularity_rank=8,
            running_style=4,
            ba_taijyu=476,
            zogen_fugo=1,
            zogen_sa=2,
            kisyu_code="01056",
            chokyosi_code="01023",
        )
        assert entry.is_winner is False
        assert entry.is_place is False

    def test_entry_cancelled(self):
        entry = Entry(
            race_id="2024032405030208",
            umaban=5,
            ketto_num="0001234567",
            finish_pos=0,
            win_odds_actual=0.0,
            popularity_rank=0,
            running_style=0,
            ba_taijyu=0,
            zogen_fugo=0,
            zogen_sa=0,
            kisyu_code="",
            chokyosi_code="",
        )
        assert entry.is_winner is False
        assert entry.is_cancelled is True


class TestBet:
    def test_create_bet(self):
        bet = Bet(
            race_id="2024032405030208",
            umaban=5,
            bet_type=BetType.WIN,
            odds=3.2,
            ev_lower_corrected=1.15,
            stake=200,
        )
        assert bet.bet_type == BetType.WIN
        assert bet.stake == 200

    def test_bet_minimum_stake(self):
        bet = Bet(
            race_id="2024032405030208",
            umaban=5,
            bet_type=BetType.WIN,
            odds=3.2,
            ev_lower_corrected=1.05,
            stake=50,
        )
        assert bet.stake < 100
        assert bet.is_valid is False

    def test_bet_has_edge_field(self):
        """Bet dataclass should have an edge field for Value Betting."""
        bet = Bet(
            race_id="20240101T11R01",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=1.5,
            ev_lower_corrected=0.0,
            stake=100.0,
            edge=0.033,
        )
        assert bet.edge == pytest.approx(0.033)

    def test_bet_edge_defaults_to_zero(self):
        """Bet edge should default to 0.0 for backward compatibility."""
        bet = Bet(
            race_id="20240101T11R01",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=1.5,
            ev_lower_corrected=1.2,
            stake=100.0,
        )
        assert bet.edge == 0.0


class TestOddsSnapshot:
    def test_create_snapshot(self):
        snapshot = OddsSnapshot(
            race_id="2024032405030208",
            happyo_time="03241505",
            umaban=5,
            tan_odds=3.2,
            fuku_odds=1.4,
        )
        assert snapshot.umaban == 5
        assert snapshot.tan_odds == 3.2


class TestDDState:
    def test_create_dd_state(self):
        state = DDState(
            current_dd=0.08,
            rolling_roi=1.05,
            n_bets_eval=150,
            recovery_state=RecoveryState.NORMAL,
        )
        assert state.recovery_state == RecoveryState.NORMAL


class TestTwoStageConfig:
    def test_defaults(self):
        config = TwoStageConfig()
        assert config.hit_metric == "auc"
        assert config.hit_rounds == 500
        assert config.return_rounds == 300
        assert config.min_hit_samples == 200


class TestRegimeConfig:
    def test_defaults(self):
        config = RegimeConfig()
        assert config.window == 200
        assert config.min_samples == 100
        assert config.fav_rate_aggressive == 0.28


class TestSubmodelSet:
    def test_submodel_set_accepts_benter_combo(self) -> None:
        """SubmodelSet が benter_combo + isotonic_calibrator フィールドを受け入れること"""
        from unittest.mock import MagicMock

        from models.benter_combination import BenterCombination
        from sklearn.isotonic import IsotonicRegression

        combo = BenterCombination(alpha=0.4, beta=0.6, gamma=-0.1)
        calibrator = IsotonicRegression(out_of_bounds="clip")

        sub = SubmodelSet(
            market=MagicMock(),
            stage1=MagicMock(),
            place_ability=MagicMock(),
            win=MagicMock(),
            ev_corrector=MagicMock(),
            place=MagicMock(),
            place_ev_corrector=MagicMock(),
            wide=MagicMock(),
            confidence=MagicMock(),
            use_ensemble=False,
            benter_combo=combo,
            isotonic_calibrator=calibrator,
        )
        assert sub.benter_combo is not None
        assert sub.benter_combo is combo
        assert sub.isotonic_calibrator is not None
        assert sub.isotonic_calibrator is calibrator

    def test_submodel_set_benter_combo_default_none(self) -> None:
        """SubmodelSet の benter_combo + isotonic_calibrator デフォルトが None であること"""
        from unittest.mock import MagicMock

        sub = SubmodelSet(
            market=MagicMock(),
            stage1=MagicMock(),
            place_ability=MagicMock(),
            win=MagicMock(),
            ev_corrector=MagicMock(),
            place=MagicMock(),
            place_ev_corrector=MagicMock(),
            wide=MagicMock(),
            confidence=MagicMock(),
        )
        assert sub.benter_combo is None
        assert sub.isotonic_calibrator is None
