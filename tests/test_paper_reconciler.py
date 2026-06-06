"""PaperReconciler のテスト — 3-column state model (D-03), schema_version=2"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from paper_trading.reconciler import PaperReconciler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bet_row(
    *,
    race_id: str = "2026040510010101",
    bet_type: str = "place",
    umaban: int = 3,
    stake: float = 100.0,
    odds: float = 2.4,
    session_id: str = "testsess00000001",
    settlement_status: str = "pending",
    outcome: object = None,
    payout: object = None,
    race_date: str = "2026-04-05",
    extra: dict | None = None,
) -> dict:
    """Create a single bet row with all required schema_version=2 columns."""
    row: dict = {
        "bet_id": PaperReconciler.compute_bet_id(session_id, race_id, bet_type, umaban),
        "session_id": session_id,
        "schema_version": 2,
        "settlement_status": settlement_status,
        "outcome": outcome,
        "payout": payout,
        "race_id": race_id,
        "bet_type": bet_type,
        "umaban": umaban,
        "stake": stake,
        "odds": odds,
        "ev": 1.5,
        "surface": "turf",
        "distance": 1200,
        "bankroll_after": 99900.0,
        "race_date": pd.Timestamp(race_date),
        "post_time": "15:30",
        "is_paper": True,
        "predicted_at": "2026-04-05T15:00:00",
        "horse_name": "テスト馬",
    }
    if extra:
        row.update(extra)
    return row


def _make_payouts_df(
    *,
    race_id: str = "2026040510010101",
    win_umaban: int | None = None,
    win_pay: float | None = None,
    place_entries: list[tuple[int, float]] | None = None,
) -> pd.DataFrame:
    """Create a minimal payouts DataFrame matching EveryDB2 schema."""
    row: dict = {"race_id": [race_id]}

    # Win columns
    if win_umaban is not None and win_pay is not None:
        row["paytansyoumaban1"] = [float(win_umaban)]
        row["paytansyopay1"] = [float(win_pay)]
    else:
        row["paytansyoumaban1"] = [float("nan")]
        row["paytansyopay1"] = [float("nan")]

    # Place columns
    for i in range(1, 6):
        if place_entries and i <= len(place_entries):
            row[f"payfukusyoumaban{i}"] = [float(place_entries[i - 1][0])]
            row[f"payfukusyopay{i}"] = [float(place_entries[i - 1][1])]
        else:
            row[f"payfukusyoumaban{i}"] = [float("nan")]
            row[f"payfukusyopay{i}"] = [float("nan")]

    return pd.DataFrame(row)


# ---------------------------------------------------------------------------
# compute_bet_id
# ---------------------------------------------------------------------------

class TestComputeBetId:
    def test_deterministic(self) -> None:
        id1 = PaperReconciler.compute_bet_id("sess1", "race1", "win", 5)
        id2 = PaperReconciler.compute_bet_id("sess1", "race1", "win", 5)
        assert id1 == id2

    def test_length_32(self) -> None:
        bet_id = PaperReconciler.compute_bet_id("s", "r", "win", 1)
        assert len(bet_id) == 32

    def test_different_inputs_different_ids(self) -> None:
        id1 = PaperReconciler.compute_bet_id("sess1", "race1", "win", 5)
        id2 = PaperReconciler.compute_bet_id("sess1", "race1", "place", 5)
        assert id1 != id2


# ---------------------------------------------------------------------------
# _validate_bet_schema
# ---------------------------------------------------------------------------

class TestValidateBetSchema:
    def test_valid_pending(self) -> None:
        df = pd.DataFrame([_make_bet_row()])
        errors = PaperReconciler._validate_bet_schema(df)
        assert errors == []

    def test_old_schema_rejected(self) -> None:
        df = pd.DataFrame([{"result": 0.0, "stake": 100}])
        errors = PaperReconciler._validate_bet_schema(df)
        assert len(errors) == 1
        assert "Old schema" in errors[0]

    def test_pending_with_outcome_fails(self) -> None:
        row = _make_bet_row(outcome="won")
        df = pd.DataFrame([row])
        errors = PaperReconciler._validate_bet_schema(df)
        assert any("Pending" in e for e in errors)

    def test_pending_with_payout_fails(self) -> None:
        row = _make_bet_row(payout=100.0)
        df = pd.DataFrame([row])
        errors = PaperReconciler._validate_bet_schema(df)
        assert any("Pending" in e for e in errors)

    def test_settled_with_null_outcome_fails(self) -> None:
        row = _make_bet_row(settlement_status="settled", outcome=None, payout=0.0)
        df = pd.DataFrame([row])
        errors = PaperReconciler._validate_bet_schema(df)
        assert any("Settled" in e for e in errors)

    def test_settled_negative_payout_fails(self) -> None:
        row = _make_bet_row(settlement_status="settled", outcome="lost", payout=-10.0)
        df = pd.DataFrame([row])
        errors = PaperReconciler._validate_bet_schema(df)
        assert any("payout" in e.lower() for e in errors)

    def test_duplicate_bet_id_fails(self) -> None:
        row1 = _make_bet_row()
        row2 = _make_bet_row(extra={"stake": 200.0})
        df = pd.DataFrame([row1, row2])
        errors = PaperReconciler._validate_bet_schema(df)
        assert any("unique" in e.lower() for e in errors)

    def test_zero_stake_fails(self) -> None:
        row = _make_bet_row(stake=0.0)
        df = pd.DataFrame([row])
        errors = PaperReconciler._validate_bet_schema(df)
        assert any("stake" in e.lower() for e in errors)

    def test_wrong_schema_version_fails(self) -> None:
        row = _make_bet_row()
        row["schema_version"] = 1
        df = pd.DataFrame([row])
        errors = PaperReconciler._validate_bet_schema(df)
        assert any("schema_version" in e for e in errors)


# ---------------------------------------------------------------------------
# Settlement: Win bets
# ---------------------------------------------------------------------------

class TestWinSettlement:
    def test_win_bet_won(self, tmp_path: Path) -> None:
        mock_everydb2 = MagicMock()
        mock_everydb2.get_payouts.return_value = _make_payouts_df(
            race_id="2026040510010101",
            win_umaban=3,
            win_pay=350.0,  # 100 yen -> 350 yen = 3.5x multiplier
        )

        bets_path = tmp_path / "bets.parquet"
        pd.DataFrame([_make_bet_row(bet_type="win", umaban=3)]).to_parquet(bets_path, index=False)

        reconciler = PaperReconciler(
            store=MagicMock(), bets_path=bets_path, everydb2=mock_everydb2,
        )
        result = reconciler.reconcile(date(2026, 4, 5))

        assert result["n_settled"] == 1
        assert result["n_new_wins"] == 1

        bets_df = pd.read_parquet(bets_path)
        assert bets_df.iloc[0]["outcome"] == "won"
        assert bets_df.iloc[0]["payout"] == pytest.approx(350.0)  # 100 * 3.5
        assert bets_df.iloc[0]["settlement_status"] == "settled"

    def test_win_bet_lost(self, tmp_path: Path) -> None:
        mock_everydb2 = MagicMock()
        mock_everydb2.get_payouts.return_value = _make_payouts_df(
            race_id="2026040510010101",
            win_umaban=5,  # different horse won
            win_pay=200.0,
        )

        bets_path = tmp_path / "bets.parquet"
        pd.DataFrame([_make_bet_row(bet_type="win", umaban=3)]).to_parquet(bets_path, index=False)

        reconciler = PaperReconciler(
            store=MagicMock(), bets_path=bets_path, everydb2=mock_everydb2,
        )
        result = reconciler.reconcile(date(2026, 4, 5))

        assert result["n_settled"] == 1
        assert result["n_new_wins"] == 0

        bets_df = pd.read_parquet(bets_path)
        assert bets_df.iloc[0]["outcome"] == "lost"
        assert bets_df.iloc[0]["payout"] == 0.0
        assert bets_df.iloc[0]["settlement_status"] == "settled"


# ---------------------------------------------------------------------------
# Settlement: Place bets
# ---------------------------------------------------------------------------

class TestPlaceSettlement:
    def test_place_bet_won(self, tmp_path: Path) -> None:
        mock_everydb2 = MagicMock()
        mock_everydb2.get_payouts.return_value = _make_payouts_df(
            race_id="2026040510010101",
            place_entries=[(3, 240.0), (7, 180.0), (1, 150.0)],
        )

        bets_path = tmp_path / "bets.parquet"
        pd.DataFrame([_make_bet_row(bet_type="place", umaban=3)]).to_parquet(bets_path, index=False)

        reconciler = PaperReconciler(
            store=MagicMock(), bets_path=bets_path, everydb2=mock_everydb2,
        )
        result = reconciler.reconcile(date(2026, 4, 5))

        assert result["n_settled"] == 1
        assert result["n_new_wins"] == 1

        bets_df = pd.read_parquet(bets_path)
        assert bets_df.iloc[0]["outcome"] == "won"
        assert bets_df.iloc[0]["payout"] == pytest.approx(240.0)  # 100 * 2.4

    def test_place_bet_lost(self, tmp_path: Path) -> None:
        mock_everydb2 = MagicMock()
        mock_everydb2.get_payouts.return_value = _make_payouts_df(
            race_id="2026040510010101",
            place_entries=[(5, 200.0), (7, 180.0), (1, 150.0)],
        )

        bets_path = tmp_path / "bets.parquet"
        pd.DataFrame([_make_bet_row(bet_type="place", umaban=3)]).to_parquet(bets_path, index=False)

        reconciler = PaperReconciler(
            store=MagicMock(), bets_path=bets_path, everydb2=mock_everydb2,
        )
        result = reconciler.reconcile(date(2026, 4, 5))

        assert result["n_settled"] == 1
        assert result["n_new_wins"] == 0

        bets_df = pd.read_parquet(bets_path)
        assert bets_df.iloc[0]["outcome"] == "lost"
        assert bets_df.iloc[0]["payout"] == 0.0


# ---------------------------------------------------------------------------
# ROI calculation (D-05)
# ---------------------------------------------------------------------------

class TestROI:
    def test_effective_stake_excludes_refunded(self, tmp_path: Path) -> None:
        """ROI effective_stake should exclude refunded bets."""
        mock_everydb2 = MagicMock()
        mock_everydb2.get_payouts.return_value = _make_payouts_df(
            race_id="2026040510010101",
            place_entries=[(3, 240.0)],
        )

        rows = [
            _make_bet_row(umaban=3, stake=100.0),  # will be won
            _make_bet_row(
                umaban=5,
                stake=100.0,
                settlement_status="settled",
                outcome="refunded",
                payout=100.0,
            ),
        ]
        bets_path = tmp_path / "bets.parquet"
        pd.DataFrame(rows).to_parquet(bets_path, index=False)

        reconciler = PaperReconciler(
            store=MagicMock(), bets_path=bets_path, everydb2=mock_everydb2,
        )
        result = reconciler.reconcile(date(2026, 4, 5))

        # effective_stake = won(100) + lost(0) = 100 (refunded excluded)
        assert result["effective_stake"] == pytest.approx(100.0)
        # total_return = payout of won = 240.0
        assert result["total_return"] == pytest.approx(240.0)
        # ROI = 240 / 100 = 2.4
        assert result["cumulative_roi"] == pytest.approx(2.4)

    def test_roi_with_losses(self, tmp_path: Path) -> None:
        """ROI includes losses in effective_stake."""
        mock_everydb2 = MagicMock()
        mock_everydb2.get_payouts.return_value = _make_payouts_df(
            race_id="2026040510010101",
            place_entries=[(3, 240.0)],
        )

        rows = [
            _make_bet_row(umaban=3, stake=100.0),  # will be won (payout 240)
            _make_bet_row(
                umaban=7,
                stake=100.0,
                settlement_status="settled",
                outcome="lost",
                payout=0.0,
            ),
        ]
        bets_path = tmp_path / "bets.parquet"
        pd.DataFrame(rows).to_parquet(bets_path, index=False)

        reconciler = PaperReconciler(
            store=MagicMock(), bets_path=bets_path, everydb2=mock_everydb2,
        )
        result = reconciler.reconcile(date(2026, 4, 5))

        # effective_stake = won(100) + lost(100) = 200
        assert result["effective_stake"] == pytest.approx(200.0)
        # total_return = 240
        assert result["total_return"] == pytest.approx(240.0)
        # ROI = 240 / 200 = 1.2
        assert result["cumulative_roi"] == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# Old schema rejection (D-18)
# ---------------------------------------------------------------------------

class TestOldSchemaRejection:
    def test_old_schema_in_bets_raises(self, tmp_path: Path) -> None:
        """bets.parquet with old 'result' column and no 'payout' raises ValueError."""
        mock_everydb2 = MagicMock()
        old_bet = {
            "race_id": "2026040510010101",
            "bet_type": "place",
            "umaban": 3,
            "stake": 100.0,
            "result": 0.0,
        }
        bets_path = tmp_path / "bets.parquet"
        pd.DataFrame([old_bet]).to_parquet(bets_path, index=False)

        reconciler = PaperReconciler(
            store=MagicMock(), bets_path=bets_path, everydb2=mock_everydb2,
        )
        with pytest.raises(ValueError, match="Old schema"):
            reconciler.reconcile(date(2026, 4, 5))


# ---------------------------------------------------------------------------
# Invalid payout keeps pending (D-11 item 6)
# ---------------------------------------------------------------------------

class TestInvalidPayout:
    def test_zero_multiplier_keeps_pending(self, tmp_path: Path) -> None:
        """Invalid payout multiplier (0) keeps bet as pending."""
        mock_everydb2 = MagicMock()
        mock_everydb2.get_payouts.return_value = _make_payouts_df(
            race_id="2026040510010101",
            win_umaban=3,
            win_pay=0.0,  # Invalid: 0 pay
        )

        bets_path = tmp_path / "bets.parquet"
        pd.DataFrame([_make_bet_row(bet_type="win", umaban=3)]).to_parquet(bets_path, index=False)

        reconciler = PaperReconciler(
            store=MagicMock(), bets_path=bets_path, everydb2=mock_everydb2,
        )
        result = reconciler.reconcile(date(2026, 4, 5))

        # Should not settle
        assert result["n_settled"] == 0

        bets_df = pd.read_parquet(bets_path)
        assert bets_df.iloc[0]["settlement_status"] == "pending"


# ---------------------------------------------------------------------------
# Idempotent reconciliation
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_already_settled_skipped(self, tmp_path: Path) -> None:
        """Already-settled bets are not reprocessed."""
        mock_everydb2 = MagicMock()
        mock_everydb2.get_payouts.return_value = _make_payouts_df(
            race_id="2026040510010101",
            place_entries=[(3, 240.0)],
        )

        row = _make_bet_row(
            settlement_status="settled",
            outcome="won",
            payout=240.0,
        )
        bets_path = tmp_path / "bets.parquet"
        pd.DataFrame([row]).to_parquet(bets_path, index=False)

        reconciler = PaperReconciler(
            store=MagicMock(), bets_path=bets_path, everydb2=mock_everydb2,
        )
        result = reconciler.reconcile(date(2026, 4, 5))
        assert result["n_settled"] == 0


# ---------------------------------------------------------------------------
# No bets / no payout data
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_bets_file(self, tmp_path: Path) -> None:
        """No bets.parquet returns empty result."""
        reconciler = PaperReconciler(
            store=MagicMock(),
            bets_path=tmp_path / "nonexistent.parquet",
            everydb2=MagicMock(),
        )
        result = reconciler.reconcile(date(2026, 4, 5))
        assert result["n_bets"] == 0

    def test_no_pending_bets(self, tmp_path: Path) -> None:
        """All bets already settled returns ROI summary."""
        row = _make_bet_row(settlement_status="settled", outcome="lost", payout=0.0)
        bets_path = tmp_path / "bets.parquet"
        pd.DataFrame([row]).to_parquet(bets_path, index=False)

        reconciler = PaperReconciler(
            store=MagicMock(), bets_path=bets_path, everydb2=MagicMock(),
        )
        result = reconciler.reconcile(date(2026, 4, 5))
        assert result["n_settled"] == 0
        assert result["cumulative_roi"] == 0.0

    def test_no_payout_data(self, tmp_path: Path) -> None:
        """No payout data available keeps bets pending."""
        mock_everydb2 = MagicMock()
        mock_everydb2.get_payouts.return_value = pd.DataFrame()

        bets_path = tmp_path / "bets.parquet"
        pd.DataFrame([_make_bet_row()]).to_parquet(bets_path, index=False)

        reconciler = PaperReconciler(
            store=MagicMock(), bets_path=bets_path, everydb2=mock_everydb2,
        )
        result = reconciler.reconcile(date(2026, 4, 5))
        assert result["n_settled"] == 0

        bets_df = pd.read_parquet(bets_path)
        assert bets_df.iloc[0]["settlement_status"] == "pending"


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_atomic_write_creates_file(self, tmp_path: Path) -> None:
        target = tmp_path / "subdir" / "test.parquet"
        df = pd.DataFrame([{"a": 1}])
        PaperReconciler._atomic_write_parquet(df, target)
        assert target.exists()
        result = pd.read_parquet(target)
        assert len(result) == 1
