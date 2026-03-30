"""バックテストHTMLレポート生成器"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from backtest.engine import BacktestResult


class BacktestReportGenerator:
    """バックテスト結果から自己完結型HTMLレポートを生成"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        result: BacktestResult,
        bet_history: list[dict[str, Any]],
        train_period: str = "",
        test_period: str = "",
    ) -> Path:
        """HTMLレポートを生成し、ファイルパスを返す"""
        bets = self._derive_fields(bet_history)
        monthly = self._compute_monthly_stats(bets)
        conditions = self._compute_condition_stats(bets)
        bankroll = self._compute_bankroll_series(bets)

        summary = {
            "roi": result.total_roi,
            "win_rate": result.winning_bets / result.total_bets if result.total_bets > 0 else 0.0,
            "profit": result.profit,
            "max_dd": result.max_drawdown,
            "final_bankroll": result.final_bankroll,
            "total_bets": result.total_bets,
            "total_stake": result.total_stake,
            "total_return": result.total_return,
            "test_period": test_period,
            "train_period": train_period,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
        template = env.get_template("report.html")

        try:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit_hash = "unknown"

        footer_info = f"commit: {commit_hash}"

        html = template.render(
            summary=summary,
            bankroll_series=bankroll,
            monthly_stats=monthly,
            condition_stats=conditions,
            bet_details=bets,
            footer_info=footer_info,
        )

        outpath = self.output_dir / "backtest_report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath

    def _derive_fields(self, bet_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """race_date, profit, is_win を派生フィールドとして追加"""
        if not bet_history:
            return []
        enriched = []
        for bet in bet_history:
            d = dict(bet)
            d["race_date"] = f"{bet['race_id'][:4]}-{bet['race_id'][4:6]}-{bet['race_id'][6:8]}"
            d["profit"] = bet["result"] - bet["stake"]
            d["is_win"] = bet["result"] > 0
            enriched.append(d)
        return enriched

    def _compute_monthly_stats(self, bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """月次集計: ROI, 的中率, ベット数, 投資額, 払戻額"""
        if not bets:
            return []
        monthly: dict[str, dict[str, float]] = defaultdict(
            lambda: {"bets": 0, "wins": 0, "stake": 0.0, "total_return": 0.0}
        )
        for b in bets:
            month = b["race_date"][:7]  # "YYYY-MM"
            monthly[month]["bets"] += 1
            monthly[month]["stake"] += b["stake"]
            if b["result"] > 0:
                monthly[month]["wins"] += 1
                monthly[month]["total_return"] += b["result"]

        result = []
        for month, stats in sorted(monthly.items()):
            bets_count = stats["bets"]
            result.append(
                {
                    "month": month,
                    "bets": bets_count,
                    "wins": int(stats["wins"]),
                    "win_rate": stats["wins"] / bets_count if bets_count > 0 else 0.0,
                    "stake": stats["stake"],
                    "total_return": stats["total_return"],
                    "roi": stats["total_return"] / stats["stake"] if stats["stake"] > 0 else 0.0,
                }
            )
        return result

    @staticmethod
    def _distance_band(surface: str, distance: int) -> str:
        """surface + distance → distance_band (FeatureEngine と同じロジック)"""
        if surface == "turf":
            if distance <= 1400:
                return "sprint"
            if distance <= 1700:
                return "mile"
            if distance <= 2100:
                return "intermediate"
            return "long"
        else:
            if distance <= 1400:
                return "sprint"
            if distance <= 1700:
                return "mile"
            return "intermediate"

    def _compute_condition_stats(self, bets: list[dict[str, Any]]) -> dict[str, Any]:
        """路面×距離帯、人気帯、EV帯の集計"""
        if not bets:
            return {"surface_distance": [], "popularity_bands": [], "ev_bands": []}

        # --- Surface × Distance Band ---
        sd_groups: dict[str, dict[str, float]] = defaultdict(
            lambda: {"bets": 0, "wins": 0, "stake": 0.0, "total_return": 0.0}
        )
        for b in bets:
            band = self._distance_band(b["surface"], b["distance"])
            key = f"{b['surface']}|{band}"
            sd_groups[key]["bets"] += 1
            sd_groups[key]["stake"] += b["stake"]
            if b["result"] > 0:
                sd_groups[key]["wins"] += 1
                sd_groups[key]["total_return"] += b["result"]

        surface_distance = []
        for key, s in sorted(sd_groups.items()):
            surface, band = key.split("|")
            n = s["bets"]
            surface_distance.append(
                {
                    "surface": surface,
                    "distance_band": band,
                    "bets": n,
                    "wins": int(s["wins"]),
                    "win_rate": s["wins"] / n if n > 0 else 0.0,
                    "roi": s["total_return"] / s["stake"] if s["stake"] > 0 else 0.0,
                }
            )

        # --- Helper for banded stats ---
        def _band_stats(
            bets_list: list[dict[str, Any]],
            key_fn: Any,
            band_order: list[str],
        ) -> list[dict[str, Any]]:
            groups: dict[str, dict[str, float]] = defaultdict(
                lambda: {"bets": 0, "wins": 0, "total_payout": 0.0}
            )
            for b in bets_list:
                band = key_fn(b)
                groups[band]["bets"] += 1
                if b["result"] > 0:
                    groups[band]["wins"] += 1
                    groups[band]["total_payout"] += b["result"]

            result = []
            for band in band_order:
                if band not in groups:
                    continue
                g = groups[band]
                n = g["bets"]
                result.append(
                    {
                        "band": band,
                        "bets": n,
                        "wins": int(g["wins"]),
                        "win_rate": g["wins"] / n if n > 0 else 0.0,
                        "avg_payout": g["total_payout"] / g["wins"] if g["wins"] > 0 else 0.0,
                        "roi": g["total_payout"] / (n * 100.0) if n > 0 else 0.0,
                    }
                )
            return result

        popularity_bands = _band_stats(
            bets,
            lambda b: "1-3" if b["popularity"] <= 3 else "4-6" if b["popularity"] <= 6 else "7+",
            ["1-3", "4-6", "7+"],
        )
        ev_bands = _band_stats(
            bets,
            lambda b: (
                "<1.0"
                if b["ev"] < 1.0
                else "1.0-1.2"
                if b["ev"] < 1.2
                else "1.2-1.5"
                if b["ev"] < 1.5
                else "1.5+"
            ),
            ["<1.0", "1.0-1.2", "1.2-1.5", "1.5+"],
        )

        return {
            "surface_distance": surface_distance,
            "popularity_bands": popularity_bands,
            "ev_bands": ev_bands,
        }

    def _compute_bankroll_series(self, bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """日付ごとの資金推移とドローダウンを抽出"""
        if not bets:
            return []
        series = []
        peak = 0.0
        for b in bets:
            bal = b["bankroll_after"]
            peak = max(peak, bal)
            dd = (peak - bal) / peak if peak > 0 else 0.0
            series.append(
                {
                    "date": b["race_date"],
                    "bankroll": bal,
                    "drawdown": dd,
                }
            )
        return series
