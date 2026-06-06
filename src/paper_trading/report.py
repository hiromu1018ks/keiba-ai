"""PaperTradingReport: pure HTML renderer consuming Aggregator results (D-12)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment


class PaperTradingReport:
    """Paper Trading レポートを Aggregator 集計結果から HTML 描画 (D-12).

    旧 _derive_fields() / _compute_monthly_stats() は削除済み。
    集計ロジックは PaperTradingReportAggregator が担当 (D-11)。
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        aggregate_results: dict[str, Any],
        bets: list[dict[str, Any]] | None = None,
    ) -> Path:
        """HTML レポートを Aggregator 集計結果から生成 (D-12).

        Args:
            aggregate_results: PaperTradingReportAggregator.aggregate_all() の出力。
                "daily", "weekly", "target" キーを含む。
            bets: ベット履歴テーブル用のデータ (bets.parquet から直接読み込み、D-10)。
                None の場合は空リストとして扱う。

        Returns:
            生成された report.html のパス。
        """
        if bets is None:
            bets = []

        daily = aggregate_results.get("daily", {})
        target_data = aggregate_results.get("target", {})

        # KPI は Aggregator daily stats から取得
        summary: dict[str, Any] = {
            "cumulative_roi": daily.get("roi", 0.0),
            "n_bets": daily.get("total_bets", 0),
            "max_dd": self._compute_max_dd(bets),
            "bankroll": self._compute_bankroll_from_daily(daily, bets),
        }

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

        html = self._render_html(
            bets=bets,
            summary=summary,
            daily=daily,
            target_data=target_data,
            commit_hash=commit_hash,
        )

        outpath = self.output_dir / "report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath

    @staticmethod
    def _compute_max_dd(bets: list[dict[str, Any]]) -> float:
        """ベットリストから最大ドローダウンを計算。

        payout - stake を累積してピークからの最大下落率を算出。
        """
        if not bets:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for b in bets:
            pnl = float(b.get("payout", 0) or 0) - float(b.get("stake", 0) or 0)
            cumulative += pnl
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def _compute_bankroll_from_daily(
        daily: dict[str, Any],
        bets: list[dict[str, Any]],
    ) -> float:
        """bankroll を daily stats から計算。

        total_return を初期バンクロール (100,000) に加算して概算。
        """
        total_return = daily.get("total_return", 0.0)
        effective_stake = daily.get("effective_stake", 0.0)
        # net_profit = total_return - effective_stake
        return 100000.0 + (total_return - effective_stake)

    @staticmethod
    def _render_html(
        bets: list[dict[str, Any]],
        summary: dict[str, Any],
        daily: dict[str, Any],
        target_data: dict[str, Any],
        commit_hash: str,
    ) -> str:
        """HTML レポートを Jinja2 テンプレートで生成。"""
        env = Environment(loader=BaseLoader(), autoescape=True)
        env.filters["pct"] = lambda x: f"{x:.1%}"
        env.filters["yen"] = lambda x: f"¥{x:,.0f}"

        # pending 情報
        pending_count = daily.get("pending_count", 0)
        unsettled_stake = daily.get("unsettled_stake", 0.0)

        # target 別内訳
        targets = target_data.get("targets", {})

        # model identity (D-19)
        model_identity = daily.get("model_identity", {})

        template_str = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Paper Trading Report</title>
<style>
body{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px}
h1{color:#333}.kpi{display:flex;gap:20px;margin:20px 0;flex-wrap:wrap}
.kpi-card{background:#f5f5f5;padding:15px;border-radius:8px;flex:1;text-align:center;min-width:150px}
.kpi-card .value{font-size:24px;font-weight:bold}
.kpi-card .label{color:#666;font-size:14px}
.kpi-card.pending{background:#fff3cd;border:1px solid #ffc107}
.kpi-card.target{background:#e8f4f8;border:1px solid #17a2b8}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ddd;padding:8px;text-align:right}
th{background:#f0f0f0}
.badge{padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold}
.badge-settled{background:#d4edda;color:#155724}
.badge-pending{background:#fff3cd;color:#856404}
.badge-won{background:#d4edda;color:#155724}
.badge-lost{background:#f8d7da;color:#721c24}
.win{color:green;font-weight:bold}.lose{color:red}
.footer{margin-top:30px;padding:15px;background:#f8f9fa;border-radius:8px;font-size:12px;color:#666}
.footer h3{margin-top:0;font-size:14px;color:#333}
.footer dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:0}
.footer dt{font-weight:bold}
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
{% if daily.hit_rate is defined %}
<div class="kpi">
<div class="kpi-card"><div class="value">{{ daily.hit_rate|pct }}</div>\
<div class="label">Hit Rate</div></div>
<div class="kpi-card"><div class="value">{{ daily.total_return|yen }}</div>\
<div class="label">Total Return</div></div>
<div class="kpi-card"><div class="value">{{ daily.effective_stake|yen }}</div>\
<div class="label">Effective Stake</div></div>
</div>
{% endif %}
{% if pending_count > 0 %}
<div class="kpi">
<div class="kpi-card pending"><div class="value">{{ pending_count }}</div>\
<div class="label">Pending</div></div>
<div class="kpi-card pending"><div class="value">{{ unsettled_stake|yen }}</div>\
<div class="label">Unsettled Stake</div></div>
</div>
{% endif %}
{% if targets %}
<h2>Target Breakdown</h2>
<div class="kpi">
{% for bt, t in targets.items() %}
<div class="kpi-card target"><div class="value">{{ bt|upper }}: {{ t.roi|pct }}</div>\
<div class="label">{{ t.total_bets }} bets, {{ t.hit_rate|pct }} hit rate</div></div>
{% endfor %}
</div>
{% endif %}
<h2>Bet History</h2>
<table><tr><th>Date</th><th>Race</th><th>Horse</th><th>Uma</th><th>Odds</th>\
<th>EV</th><th>Status</th><th>Outcome</th><th>P/L</th></tr>
{% for b in bets %}
<tr><td>{{ b.race_date }}</td><td>{{ b.race_id }}</td><td>{{ b.horse_name }}</td>
<td>{{ b.umaban }}</td><td>{{ b.odds }}</td><td>{{ b.ev }}</td>
<td><span class="badge badge-{{ b.settlement_status }}">{{ b.settlement_status }}</span></td>
<td>{% if b.outcome == 'won' %}<span class="badge badge-won">won</span>\
{% elif b.outcome == 'lost' %}<span class="badge badge-lost">lost</span>\
{% else %}{{ b.outcome }}{% endif %}</td>
<td class="{{ 'win' if b.payout > 0 else 'lose' }}">{{ b.payout - b.stake }}</td></tr>
{% endfor %}
</table>
<div class="footer">
<h3>Model Identity</h3>
{% if model_identity %}
<dl>
<dt>MLflow Run ID</dt><dd>{{ model_identity.model_run_id or 'N/A' }}</dd>
<dt>Training Period</dt>
<dd>{{ model_identity.training_start or 'N/A' }} ~ {{ model_identity.training_end or 'N/A' }}</dd>
<dt>Manifest Hash</dt><dd>{{ model_identity.manifest_hash or 'N/A' }}</dd>
</dl>
{% else %}
<p>No model identity available.</p>
{% endif %}
<p>commit: {{ commit_hash }} | generated: {{ generated_at }}</p>
</div>
</body></html>"""

        template = env.from_string(template_str)
        return template.render(
            bets=bets,
            summary=summary,
            daily=daily,
            pending_count=pending_count,
            unsettled_stake=unsettled_stake,
            targets=targets,
            model_identity=model_identity,
            commit_hash=commit_hash,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
