"""検証結果JSON生成モジュールのテスト (Phase 18, VAL-01/VAL-02)

TestValidationReport: evaluate_validation / generate_validation_report / generate_cause_analysis
の3公開関数のテスト。
"""

from __future__ import annotations

from typing import Any

import pytest

from backtest.engine import BacktestResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bet_history() -> list[dict[str, Any]]:
    """オッズバンド/レジーム/芝ダートが含まれるbet_history fixture (10件)"""
    return [
        {
            "race_id": "20240101010101", "bet_type": "win", "umaban": 1,
            "stake": 100, "odds": 1.8, "fuku_odds_low": 1.9, "result": 190,
            "surface": "turf", "kyori": 1600, "ev": 1.2, "edge": 0.05,
            "popularity": 2, "bankroll_after": 100090, "race_date": "2024-01-01",
            "regime": "AGGRESSIVE",
        },
        {
            "race_id": "20240102010101", "bet_type": "win", "umaban": 3,
            "stake": 100, "odds": 3.5, "fuku_odds_low": 3.2, "result": 0,
            "surface": "turf", "kyori": 2000, "ev": 1.1, "edge": 0.03,
            "popularity": 5, "bankroll_after": 99990, "race_date": "2024-02-15",
            "regime": "CONSERVATIVE",
        },
        {
            "race_id": "20240301010101", "bet_type": "win", "umaban": 7,
            "stake": 100, "odds": 8.0, "fuku_odds_low": 7.5, "result": 750,
            "surface": "dirt", "kyori": 1200, "ev": 1.5, "edge": 0.08,
            "popularity": 8, "bankroll_after": 100640, "race_date": "2024-03-01",
            "regime": "AGGRESSIVE",
        },
        {
            "race_id": "20240401010101", "bet_type": "win", "umaban": 2,
            "stake": 100, "odds": 2.5, "fuku_odds_low": 2.3, "result": 0,
            "surface": "turf", "kyori": 1800, "ev": 0.9, "edge": -0.02,
            "popularity": 3, "bankroll_after": 100540, "race_date": "2024-04-01",
            "regime": "CONSERVATIVE",
        },
        {
            "race_id": "20240501010101", "bet_type": "win", "umaban": 5,
            "stake": 100, "odds": 12.0, "fuku_odds_low": 11.0, "result": 0,
            "surface": "dirt", "kyori": 1400, "ev": 1.3, "edge": 0.06,
            "popularity": 10, "bankroll_after": 100440, "race_date": "2024-05-01",
            "regime": "CONSERVATIVE",
        },
        {
            "race_id": "20240601010101", "bet_type": "win", "umaban": 4,
            "stake": 100, "odds": 4.5, "fuku_odds_low": 4.8, "result": 480,
            "surface": "turf", "kyori": 1600, "ev": 1.4, "edge": 0.07,
            "popularity": 6, "bankroll_after": 100820, "race_date": "2024-06-01",
            "regime": "AGGRESSIVE",
        },
        {
            "race_id": "20240701010101", "bet_type": "win", "umaban": 6,
            "stake": 100, "odds": 6.0, "fuku_odds_low": 5.5, "result": 0,
            "surface": "turf", "kyori": 2000, "ev": 1.1, "edge": 0.03,
            "popularity": 7, "bankroll_after": 100720, "race_date": "2024-07-01",
            "regime": "CONSERVATIVE",
        },
        {
            "race_id": "20240801010101", "bet_type": "win", "umaban": 1,
            "stake": 100, "odds": 1.5, "fuku_odds_low": 1.6, "result": 160,
            "surface": "dirt", "kyori": 1000, "ev": 1.0, "edge": 0.01,
            "popularity": 1, "bankroll_after": 100780, "race_date": "2024-08-01",
            "regime": "CONSERVATIVE",
        },
        {
            "race_id": "20240901010101", "bet_type": "win", "umaban": 9,
            "stake": 100, "odds": 15.0, "fuku_odds_low": 14.0, "result": 0,
            "surface": "turf", "kyori": 2400, "ev": 0.8, "edge": -0.05,
            "popularity": 12, "bankroll_after": 100680, "race_date": "2024-09-01",
            "regime": "CONSERVATIVE",
        },
        {
            "race_id": "20250101010101", "bet_type": "win", "umaban": 2,
            "stake": 100, "odds": 3.0, "fuku_odds_low": 3.5, "result": 350,
            "surface": "turf", "kyori": 1600, "ev": 1.2, "edge": 0.04,
            "popularity": 4, "bankroll_after": 100930, "race_date": "2025-01-01",
            "regime": "AGGRESSIVE",
        },
    ]


@pytest.fixture
def pass_result(sample_bet_history: list[dict[str, Any]]) -> BacktestResult:
    """ROI>100% のBacktestResult (テスト用)"""
    total_stake = sum(b["stake"] for b in sample_bet_history)
    total_return = sum(b["result"] for b in sample_bet_history)
    return BacktestResult(
        total_bets=len(sample_bet_history),
        total_stake=total_stake,
        total_return=total_return,
        winning_bets=sum(1 for b in sample_bet_history if b["result"] > 0),
        total_roi=total_return / total_stake if total_stake > 0 else 0.0,
        bet_history=sample_bet_history,
    )


@pytest.fixture
def fail_result() -> BacktestResult:
    """ROI<100% のBacktestResult (テスト用)"""
    bet_history = [
        {
            "race_id": "20240101010101", "stake": 100, "odds": 3.0,
            "fuku_odds_low": 3.0, "result": 0, "surface": "turf",
            "race_date": "2024-06-01", "regime": "CONSERVATIVE", "ev": 1.2,
        },
        {
            "race_id": "20240102010101", "stake": 100, "odds": 5.0,
            "fuku_odds_low": 5.0, "result": 200, "surface": "dirt",
            "race_date": "2024-06-15", "regime": "AGGRESSIVE", "ev": 0.8,
        },
    ]
    return BacktestResult(
        total_bets=2,
        total_stake=200,
        total_return=200,
        winning_bets=1,
        total_roi=1.0,  # ROI = 100% -> FAIL (not > 1.0)
        bet_history=bet_history,
    )


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------

class TestValidationReport:
    """VAL-01/VAL-02 検証レポート生成テスト"""

    def test_evaluate_validation_pass(self) -> None:
        """Test 1: ROI>1.0 and bets>=100 -> PASS"""
        from backtest.validation_report import evaluate_validation

        assert evaluate_validation(roi=1.05, total_bets=200) == "PASS"

    def test_evaluate_validation_fail_roi(self) -> None:
        """Test 2: ROI<=1.0 -> FAIL"""
        from backtest.validation_report import evaluate_validation

        assert evaluate_validation(roi=0.89, total_bets=200) == "FAIL"

    def test_evaluate_validation_fail_bet_count(self) -> None:
        """Test 3: ROI>1.0 but bets<100 -> FAIL"""
        from backtest.validation_report import evaluate_validation

        assert evaluate_validation(roi=1.05, total_bets=50) == "FAIL"

    def test_generate_validation_report_structure(
        self, pass_result: BacktestResult
    ) -> None:
        """Test 4: generate_validation_report()が正しいJSON構造を返す"""
        from backtest.validation_report import generate_validation_report

        report = generate_validation_report(
            result=pass_result,
            test_start="2024-01-01",
            test_end="2025-12-31",
            train_start="2020-01-01",
            train_end="2023-12-31",
            manifest_path=None,
            pfp_result=None,
        )

        # トップレベルキーの存在確認
        assert "validation_timestamp" in report
        assert "test_period" in report
        assert report["test_period"] == ["2024-01-01", "2025-12-31"]
        assert "train_period" in report
        assert report["train_period"] == ["2020-01-01", "2023-12-31"]
        assert "manifest" in report
        assert "pfp_verification" in report
        assert "roi" in report
        assert "yearly_breakdown" in report
        assert "validation_result" in report

        # manifest (manifest_path=None)
        assert report["manifest"]["path"] is None
        assert report["manifest"]["sha256_verified"] is None

        # pfp_verification (pfp_result=None)
        assert report["pfp_verification"]["passed"] is None
        assert "message" in report["pfp_verification"]

        # roi
        assert report["roi"]["total_roi"] == pass_result.total_roi
        assert report["roi"]["total_bets"] == pass_result.total_bets
        assert report["roi"]["total_stake"] == pass_result.total_stake
        assert report["roi"]["total_return"] == pass_result.total_return
        assert report["roi"]["target_roi"] == 1.0
        assert report["roi"]["target_bets"] == 100
        assert isinstance(report["roi"]["passed"], bool)

        # yearly_breakdown に年が含まれる
        assert "2024" in report["yearly_breakdown"]
        assert "2025" in report["yearly_breakdown"]
        for year_data in report["yearly_breakdown"].values():
            assert "roi" in year_data
            assert "bets" in year_data
            assert "stake" in year_data
            assert "return" in year_data

        # ROI>100% & 10 bets -> FAIL (bets < 100)
        assert report["validation_result"] == "FAIL"

    def test_generate_cause_analysis_odds_bands(
        self, sample_bet_history: list[dict[str, Any]]
    ) -> None:
        """Test 5: generate_cause_analysis()がオッズバンド別ROI(4バンド)を含む"""
        from backtest.validation_report import generate_cause_analysis

        analysis = generate_cause_analysis(sample_bet_history)

        assert "odds_band_roi" in analysis
        bands = analysis["odds_band_roi"]
        # 4バンドが存在することを確認
        expected_bands = {"1.0-2.0", "2.0-5.0", "5.0-10.0", "10.0+"}
        assert set(bands.keys()) == expected_bands
        for band_data in bands.values():
            assert "roi" in band_data
            assert "bets" in band_data
            assert "stake" in band_data
            assert "return" in band_data

    def test_generate_cause_analysis_regime_roi(
        self, sample_bet_history: list[dict[str, Any]]
    ) -> None:
        """Test 6: generate_cause_analysis()がレジーム別ROIを含む"""
        from backtest.validation_report import generate_cause_analysis

        analysis = generate_cause_analysis(sample_bet_history)

        assert "regime_roi" in analysis
        regimes = analysis["regime_roi"]
        # fixtureにはAGGRESSIVEとCONSERVATIVEが含まれる
        assert "AGGRESSIVE" in regimes or "CONSERVATIVE" in regimes
        for regime_data in regimes.values():
            assert "roi" in regime_data
            assert "bets" in regime_data
            assert "stake" in regime_data
            assert "return" in regime_data

    def test_generate_cause_analysis_empty_history(self) -> None:
        """Test 7: 空bet_historyでerrorキーを含むdictを返す"""
        from backtest.validation_report import generate_cause_analysis

        analysis = generate_cause_analysis([])
        assert "error" in analysis

    def test_generate_cause_analysis_missing_fields(self) -> None:
        """Test 8: regime/surface欠損時も.get()で安全に処理する (KeyErrorが発生しない)"""
        from backtest.validation_report import generate_cause_analysis

        # regime/surface/final_odds/oddsが欠損したbet_history
        incomplete_history: list[dict[str, Any]] = [
            {"stake": 100, "result": 150, "race_date": "2024-01-01", "ev": 1.2},
            {"stake": 100, "result": 0, "race_date": "2024-02-01", "ev": 0.8},
        ]
        # KeyErrorが発生しないことを確認
        analysis = generate_cause_analysis(incomplete_history)
        assert "odds_band_roi" in analysis
        assert "regime_roi" in analysis
        assert "surface_roi" in analysis
        assert "ev_diagnosis" in analysis
        assert "bet_count_sufficiency" in analysis

    def test_generate_cause_analysis_tail_segments_use_actual_stakes(self) -> None:
        """EV帯/単勝オッズ尾部/人気帯は実ベット行だけで集計する"""
        from backtest.validation_report import generate_cause_analysis

        history: list[dict[str, Any]] = [
            {
                "race_id": "R1",
                "stake": 100,
                "result": 0,
                "final_odds": 55.0,
                "win_selection_ev_tail_calibrated": 3.2,
                "popularity": 9,
                "race_date": "2024-01-01",
                "is_actual_bet": True,
            },
            {
                "race_id": "R1",
                "stake": None,
                "result": 9999,
                "final_odds": 120.0,
                "win_selection_ev_tail_calibrated": 6.0,
                "popularity": 12,
                "race_date": "2024-01-01",
                "is_actual_bet": False,
            },
        ]

        analysis = generate_cause_analysis(history)

        assert "ev_band_roi" in analysis
        assert "win_odds_band_roi" in analysis
        assert "popularity_band_roi" in analysis
        assert "tail_flag_roi" in analysis
        assert analysis["tail_flag_roi"]["ev>=3"]["bets"] == 1
        assert analysis["tail_flag_roi"]["ev>=5"]["bets"] == 0
        assert analysis["tail_flag_roi"]["odds>=50"]["bets"] == 1
        assert analysis["tail_flag_roi"]["odds>=100"]["bets"] == 0
        assert analysis["popularity_band_roi"]["9-12"]["bets"] == 1
