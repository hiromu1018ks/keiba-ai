"""ShadowComparisonReportGenerator のユニットテスト (D-16, D-17).

HTML レポート生成の検証 — self-contained で side-by-side 比較を含む。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtest.engine import BacktestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bet_history(n: int = 5) -> list[dict]:
    """Build synthetic bet history for report tests."""
    history = []
    for i in range(n):
        history.append({
            "race_id": f"2024010{i}",
            "umaban": i + 1,
            "stake": 100.0,
            "odds": 5.0 + i,
            "tanodds": 5.0 + i,
            "closing_win_odds": 5.5 + i,
            "result": 500.0 if i == 0 else 0.0,
            "final_odds": 5.0 + i,
            "is_actual_bet": True,
            "surface": "turf" if i % 2 == 0 else "dirt",
            "p_win_final": 0.2 + i * 0.05,
            "win_selection_ev": 1.1 + i * 0.05,
            "win_market_selection_score": 0.5 + i * 0.1,
            "investment_score": 0.6 + i * 0.1,
            "edge": 0.1,
        })
    return history


def _make_backtest_result(bet_history: list[dict]) -> BacktestResult:
    ts = sum(b.get("stake", 100) for b in bet_history)
    tr = sum(b.get("result", 0) for b in bet_history)
    return BacktestResult(
        total_bets=len(bet_history),
        total_stake=ts,
        total_return=tr,
        winning_bets=sum(1 for b in bet_history if b.get("result", 0) > 0),
        total_roi=tr / ts if ts > 0 else 0.0,
        max_drawdown=0.1,
        final_bankroll=100_000 + tr - ts,
        bet_history=bet_history,
    )


def _make_comparison_result() -> "ShadowComparisonResult":
    """Build synthetic ShadowComparisonResult for report tests."""
    from backtest.shadow_comparison import (
        ComparisonMetrics,
        FoldDefinition,
        ShadowComparisonResult,
        VariantResult,
    )

    fold = FoldDefinition(
        year=2024,
        train_start="2020-01-01",
        train_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2024-12-31",
    )

    baseline_bh = _make_bet_history(5)
    shadow_bh = _make_bet_history(5)

    # Build race_diff
    race_rows = []
    for i in range(5):
        race_rows.append({
            "race_id": f"2024010{i}",
            "baseline_selected_umaban": 1,
            "shadow_selected_umaban": 2 if i < 2 else 1,
            "selected_changed": i < 2,
            "baseline_tanodds": 5.0 + i,
            "shadow_tanodds": 6.0 + i,
            "baseline_result": baseline_bh[i]["result"],
            "shadow_result": shadow_bh[i]["result"],
            "baseline_stake": 100.0,
            "shadow_stake": 100.0,
            "baseline_p_win_final": 0.2 + i * 0.05,
            "shadow_p_win_final": 0.18 + i * 0.05,
            "baseline_investment_score": 0.5 + i * 0.1,
            "shadow_investment_score": 0.55 + i * 0.1,
        })
    race_diff = pd.DataFrame(race_rows)

    metrics = {
        "baseline": ComparisonMetrics(
            brier=0.18, logloss=0.65, ece=0.04, roi=0.67,
            hit_rate=0.20, bet_count=5, avg_odds=7.0,
            max_drawdown=0.10, clv=0.05, clv_available=True,
            selection_agreement=0.60, avg_investment_score=0.60,
            actual_predicted_ratio=1.0,
        ),
        "shadow": ComparisonMetrics(
            brier=0.16, logloss=0.60, ece=0.03, roi=0.80,
            hit_rate=0.20, bet_count=5, avg_odds=7.0,
            max_drawdown=0.08, clv=0.06, clv_available=True,
            selection_agreement=0.60, avg_investment_score=0.65,
            actual_predicted_ratio=1.0,
        ),
    }

    return ShadowComparisonResult(
        fold=fold,
        variants={
            "baseline": VariantResult(
                "baseline", _make_backtest_result(baseline_bh),
                {"enable_market_aware_calibrator": False, "enable_race_level_ranker": False},
            ),
            "shadow": VariantResult(
                "shadow", _make_backtest_result(shadow_bh),
                {"enable_market_aware_calibrator": True, "enable_race_level_ranker": True},
            ),
        },
        race_diff=race_diff,
        horse_diff=pd.DataFrame(),
        metrics=metrics,
        alignment_succeeded=True,
    )


# ===================================================================
# Tests
# ===================================================================


class TestShadowComparisonReportGenerator:
    """Tests for ShadowComparisonReportGenerator."""

    def test_init_creates_output_dir(self, tmp_path: Path) -> None:
        from backtest.shadow_report import ShadowComparisonReportGenerator

        out = tmp_path / "reports"
        gen = ShadowComparisonReportGenerator(out)
        assert out.exists()

    def test_generate_produces_html_file(self, tmp_path: Path) -> None:
        from backtest.shadow_report import ShadowComparisonReportGenerator

        gen = ShadowComparisonReportGenerator(tmp_path)
        cr = _make_comparison_result()

        from backtest.shadow_comparison import VariantConfig

        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]
        metrics_json = {
            "folds": {
                "2024": {
                    "metrics": {
                        "baseline": {"roi": 0.67, "brier": 0.18},
                        "shadow": {"roi": 0.80, "brier": 0.16},
                    },
                    "metrics_by_surface": {},
                    "metrics_by_odds_band": {},
                    "selection_agreement": 0.60,
                },
            },
            "overall": {
                "metrics": {
                    "baseline": {"roi": 0.67, "brier": 0.18},
                    "shadow": {"roi": 0.80, "brier": 0.16},
                },
            },
        }
        report_path = gen.generate(
            comparison_results=[cr],
            variant_configs=variant_configs,
            metrics_json=metrics_json,
        )
        assert report_path.exists()
        assert report_path.stat().st_size > 0

    def test_generate_html_contains_overall_summary(self, tmp_path: Path) -> None:
        from backtest.shadow_report import ShadowComparisonReportGenerator

        gen = ShadowComparisonReportGenerator(tmp_path)
        cr = _make_comparison_result()

        from backtest.shadow_comparison import VariantConfig

        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]
        metrics_json = {
            "folds": {
                "2024": {
                    "metrics": {
                        "baseline": {"roi": 0.67, "brier": 0.18},
                        "shadow": {"roi": 0.80, "brier": 0.16},
                    },
                    "selection_agreement": 0.60,
                },
            },
            "overall": {
                "metrics": {
                    "baseline": {"roi": 0.67, "brier": 0.18},
                    "shadow": {"roi": 0.80, "brier": 0.16},
                },
            },
        }
        report_path = gen.generate(
            comparison_results=[cr],
            variant_configs=variant_configs,
            metrics_json=metrics_json,
        )
        html = report_path.read_text(encoding="utf-8")
        assert "Overall Summary" in html or "Overall" in html

    def test_generate_html_contains_selection_agreement(self, tmp_path: Path) -> None:
        from backtest.shadow_report import ShadowComparisonReportGenerator

        gen = ShadowComparisonReportGenerator(tmp_path)
        cr = _make_comparison_result()

        from backtest.shadow_comparison import VariantConfig

        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]
        metrics_json = {
            "folds": {
                "2024": {
                    "metrics": {
                        "baseline": {"roi": 0.67},
                        "shadow": {"roi": 0.80},
                    },
                    "selection_agreement": 0.60,
                },
            },
            "overall": {
                "metrics": {
                    "baseline": {"roi": 0.67},
                    "shadow": {"roi": 0.80},
                },
            },
        }
        report_path = gen.generate(
            comparison_results=[cr],
            variant_configs=variant_configs,
            metrics_json=metrics_json,
        )
        html = report_path.read_text(encoding="utf-8")
        assert "Selection Agreement" in html or "selection" in html.lower()

    def test_generate_html_contains_calibration(self, tmp_path: Path) -> None:
        from backtest.shadow_report import ShadowComparisonReportGenerator

        gen = ShadowComparisonReportGenerator(tmp_path)
        cr = _make_comparison_result()

        from backtest.shadow_comparison import VariantConfig

        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]
        metrics_json = {
            "folds": {
                "2024": {
                    "metrics": {
                        "baseline": {"roi": 0.67},
                        "shadow": {"roi": 0.80},
                    },
                    "selection_agreement": 0.60,
                },
            },
            "overall": {
                "metrics": {
                    "baseline": {"roi": 0.67},
                    "shadow": {"roi": 0.80},
                },
            },
        }
        report_path = gen.generate(
            comparison_results=[cr],
            variant_configs=variant_configs,
            metrics_json=metrics_json,
        )
        html = report_path.read_text(encoding="utf-8")
        assert "Calibration" in html or "Brier" in html

    def test_generate_html_contains_variant_names(self, tmp_path: Path) -> None:
        from backtest.shadow_report import ShadowComparisonReportGenerator

        gen = ShadowComparisonReportGenerator(tmp_path)
        cr = _make_comparison_result()

        from backtest.shadow_comparison import VariantConfig

        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
            VariantConfig("shadow", Path("data/shadow"), True, True),
        ]
        metrics_json = {
            "folds": {"2024": {"metrics": {}}},
            "overall": {"metrics": {}},
        }
        report_path = gen.generate(
            comparison_results=[cr],
            variant_configs=variant_configs,
            metrics_json=metrics_json,
        )
        html = report_path.read_text(encoding="utf-8")
        assert "baseline" in html
        assert "shadow" in html

    def test_generate_html_is_self_contained(self, tmp_path: Path) -> None:
        """HTML should have no external CSS/JS links (self-contained per D-16)."""
        from backtest.shadow_report import ShadowComparisonReportGenerator

        gen = ShadowComparisonReportGenerator(tmp_path)
        cr = _make_comparison_result()

        from backtest.shadow_comparison import VariantConfig

        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
        ]
        metrics_json = {
            "folds": {"2024": {"metrics": {}}},
            "overall": {"metrics": {}},
        }
        report_path = gen.generate(
            comparison_results=[cr],
            variant_configs=variant_configs,
            metrics_json=metrics_json,
        )
        html = report_path.read_text(encoding="utf-8")
        # No external CDN links
        assert "cdn.jsdelivr.net" not in html
        assert "code.jquery.com" not in html
        # Has inline style tag
        assert "<style>" in html

    def test_generate_html_contains_fold_breakdown(self, tmp_path: Path) -> None:
        from backtest.shadow_report import ShadowComparisonReportGenerator

        gen = ShadowComparisonReportGenerator(tmp_path)
        cr = _make_comparison_result()

        from backtest.shadow_comparison import VariantConfig

        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
        ]
        metrics_json = {
            "folds": {
                "2024": {
                    "metrics": {
                        "baseline": {"roi": 0.67},
                    },
                    "selection_agreement": 0.60,
                },
            },
            "overall": {"metrics": {}},
        }
        report_path = gen.generate(
            comparison_results=[cr],
            variant_configs=variant_configs,
            metrics_json=metrics_json,
        )
        html = report_path.read_text(encoding="utf-8")
        assert "2024" in html
        assert "Fold" in html or "fold" in html

    def test_generate_html_has_source_of_truth_footer(self, tmp_path: Path) -> None:
        """Footer must note JSON/Parquet are source of truth per D-17."""
        from backtest.shadow_report import ShadowComparisonReportGenerator

        gen = ShadowComparisonReportGenerator(tmp_path)
        cr = _make_comparison_result()

        from backtest.shadow_comparison import VariantConfig

        variant_configs = [
            VariantConfig("baseline", Path("data/bt"), False, False),
        ]
        metrics_json = {
            "folds": {"2024": {"metrics": {}}},
            "overall": {"metrics": {}},
        }
        report_path = gen.generate(
            comparison_results=[cr],
            variant_configs=variant_configs,
            metrics_json=metrics_json,
        )
        html = report_path.read_text(encoding="utf-8")
        assert "source of truth" in html.lower() or "JSON" in html
