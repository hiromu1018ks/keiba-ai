"""Paper Trading 用 HTML レポート生成器"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment


class PaperTradingReport:
    """Paper Trading 累積レポートを生成。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        bets: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> Path:
        """HTML レポートを生成"""
        enriched = self._derive_fields(bets)
        monthly = self._compute_monthly_stats(enriched)
        bankroll_series = self._compute_bankroll_series(enriched)

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

        html = self._render_html(enriched, monthly, bankroll_series, summary, commit_hash)

        outpath = self.output_dir / "report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath

    @staticmethod
    def _derive_fields(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not bets:
            return []
        return [
            {
                **b,
                "profit": b["result"] - b["stake"],
                "is_win": b["result"] > 0,
            }
            for b in bets
        ]

    @staticmethod
    def _compute_monthly_stats(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not bets:
            return []
        from collections import defaultdict

        monthly: dict[str, dict[str, float]] = defaultdict(
            lambda: {"bets": 0, "wins": 0, "stake": 0.0, "total_return": 0.0}
        )
        for b in bets:
            month = b["race_date"][:7]
            monthly[month]["bets"] += 1
            monthly[month]["stake"] += b["stake"]
            if b["result"] > 0:
                monthly[month]["wins"] += 1
                monthly[month]["total_return"] += b["result"]

        return [
            {
                "month": m,
                "bets": s["bets"],
                "wins": int(s["wins"]),
                "roi": s["total_return"] / s["stake"] if s["stake"] > 0 else 0.0,
            }
            for m, s in sorted(monthly.items())
        ]

    @staticmethod
    def _compute_bankroll_series(bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not bets:
            return []
        peak = 0.0
        series: list[dict[str, Any]] = []
        for b in bets:
            bal = b["bankroll_after"]
            peak = max(peak, bal)
            dd = (peak - bal) / peak if peak > 0 else 0.0
            series.append({"date": b["race_date"], "bankroll": bal, "drawdown": dd})
        return series

    @staticmethod
    def _render_html(
        bets: list[dict[str, Any]],
        monthly: list[dict[str, Any]],
        bankroll_series: list[dict[str, Any]],
        summary: dict[str, Any],
        commit_hash: str,
    ) -> str:
        """シンプルな HTML レポートを生成 (Jinja2 テンプレート)"""
        env = Environment(loader=BaseLoader(), autoescape=True)
        env.filters["pct"] = lambda x: f"{x:.1%}"
        env.filters["yen"] = lambda x: f"¥{x:,.0f}"

        template_str = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Paper Trading Report</title>
<style>
body{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px}
h1{color:#333}.kpi{display:flex;gap:20px;margin:20px 0}
.kpi-card{background:#f5f5f5;padding:15px;border-radius:8px;flex:1;text-align:center}
.kpi-card .value{font-size:24px;font-weight:bold}
.kpi-card .label{color:#666;font-size:14px}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ddd;padding:8px;text-align:right}
th{background:#f0f0f0}
.win{color:green;font-weight:bold}.lose{color:red}
</style></head><body>
<h1>Paper Trading Report</h1>
<div class="kpi">
<div class="kpi-card"><div class="value">{{ summary.cumulative_roi|pct }}</div>\
<div class="label">Cumulative ROI</div></div>
<div class="kpi-card"><div class="value">{{ summary.n_bets }}</div>\
<div class="label">Total Bets</div></div>
<div class="kpi-card"><div class="value">{{ summary.max_dd|pct }}</div>\
<div class="label">Max Drawdown</div></div>
<div class="kpi-card"><div class="value">{{ summary.bankroll|yen }}</div>\
<div class="label">Bankroll</div></div>
</div>
<h2>Bet History</h2>
<table><tr><th>Date</th><th>Race</th><th>Horse</th><th>Uma</th><th>Odds</th><th>EV</th><th>Result</th><th>P/L</th></tr>
{% for b in bets %}
<tr><td>{{ b.race_date }}</td><td>{{ b.race_id }}</td><td>{{ b.horse_name }}</td>
<td>{{ b.umaban }}</td><td>{{ b.odds }}</td><td>{{ b.ev }}</td>
<td>{{ b.result }}</td><td class="{{ 'win' if b.is_win else 'lose' }}">{{ b.profit }}</td></tr>
{% endfor %}
</table>
<p style="color:#999;font-size:12px">commit: {{ commit_hash }} | generated: {{ generated_at }}</p>
</body></html>"""

        template = env.from_string(template_str)
        return template.render(
            bets=bets,
            monthly=monthly,
            bankroll_series=bankroll_series,
            summary=summary,
            commit_hash=commit_hash,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
