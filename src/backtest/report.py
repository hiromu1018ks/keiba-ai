"""バックテストHTMLレポート生成器"""

from __future__ import annotations

import json
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
        betting_target: str = "place",
    ) -> Path:
        """HTMLレポートを生成し、ファイルパスを返す"""
        bets = self._derive_fields(bet_history)
        monthly = self._compute_monthly_stats(bets)
        conditions = self._compute_condition_stats(bets, betting_target=betting_target)
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
            "exclusion_stats": {
                "collapsed_skipped": result.n_collapsed_skipped,
                "ev_excluded": result.n_ev_excluded,
                "odds_band_excluded": result.n_odds_band_excluded,
                "win_ev_odds_excluded": result.n_win_ev_odds_excluded,
                "win_stake_increased": result.n_win_stake_increased,
                "total_win_stake_increased": result.total_win_stake_increased,
                "odds_band_filter_excluded": result.exclusion_stats.get(
                    "odds_band_filter_excluded", {}
                ),
            },
        }

        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
        env.filters["format_number"] = lambda x: f"{x:,.0f}"
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
            betting_target=betting_target,
        )

        outpath = self.output_dir / "backtest_report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath

    def save_bet_history(self, bet_history: list[dict[str, Any]]) -> Path:
        """bet_history を JSON に保存"""
        path = self.output_dir / "bet_history.json"
        path.write_text(json.dumps(bet_history, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def save_ai_diagnostics(
        self,
        bets: list[dict[str, Any]],
        result: BacktestResult,
        betting_target: str = "place",
    ) -> Path | None:
        """AI分析用診断JSONを保存 (D-06, D-08)"""
        if betting_target != "win" or not bets:
            return None

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

        monthly = self._compute_monthly_stats(bets)
        regime = self._compute_regime_stats(bets)
        conditions = self._compute_condition_stats(bets, betting_target="win")

        # Highlights: best/worst band 自動特定 (D-08)
        all_bands: list[dict[str, Any]] = []
        for band in conditions.get("popularity_bands", []):
            all_bands.append({
                "name": f"人気{band['band']}", "roi": band["roi"], "bets": band["bets"],
            })
        for band in conditions.get("odds_multiplier_bands", []):
            all_bands.append({
                "name": f"オッズ{band['band']}", "roi": band["roi"], "bets": band["bets"],
            })
        for band in conditions.get("ev_bands", []):
            all_bands.append({
                "name": f"EV{band['band']}", "roi": band["roi"], "bets": band["bets"],
            })
        for r in regime:
            all_bands.append({
                "name": f"Regime:{r['regime']}", "roi": r["roi"], "bets": r["bets"],
            })

        significant_bands = [b for b in all_bands if b["bets"] >= 5]
        best_band = max(significant_bands, key=lambda x: x["roi"]) if significant_bands else None
        worst_band = min(significant_bands, key=lambda x: x["roi"]) if significant_bands else None

        # 月別トレンド (D-08)
        monthly_trend = "stable"
        if len(monthly) >= 3:
            first_half = sum(m["roi"] for m in monthly[: len(monthly) // 2]) / (len(monthly) // 2)
            second_half = sum(m["roi"] for m in monthly[len(monthly) // 2 :]) / (
                len(monthly) - len(monthly) // 2
            )
            if second_half > first_half * 1.1:
                monthly_trend = "improving"
            elif second_half < first_half * 0.9:
                monthly_trend = "declining"

        edges = [b.get("edge", 0) for b in bets if "edge" in b]
        avg_edge = sum(edges) / len(edges) if edges else 0.0

        diagnostic = {
            "meta": {
                "betting_target": "win",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "commit": commit_hash,
            },
            "summary": {
                "roi": result.total_roi,
                "win_rate": result.winning_bets / result.total_bets
                if result.total_bets > 0
                else 0.0,
                "total_bets": result.total_bets,
                "total_stake": result.total_stake,
                "total_return": result.total_return,
                "avg_edge": avg_edge,
                "profit": result.profit,
            },
            "monthly_trend": monthly,
            "regime_breakdown": regime,
            "odds_multiplier_bands": conditions.get("odds_multiplier_bands", []),
            "popularity_bands": conditions.get("popularity_bands", []),
            "ev_bands": conditions.get("ev_bands", []),
            "surface_distance": conditions.get("surface_distance", []),
            "highlights": {
                "best_band": best_band,
                "worst_band": worst_band,
                "monthly_trend": monthly_trend,
                "overperforming_conditions": [
                    b for b in significant_bands
                    if b["roi"] > result.total_roi and b["bets"] >= 5
                ],
                "underperforming_conditions": [
                    b for b in significant_bands
                    if b["roi"] < result.total_roi and b["bets"] >= 5
                ],
            },
            "exclusion": {
                "collapsed_skipped": result.n_collapsed_skipped,
                "ev_excluded": result.n_ev_excluded,
                "odds_band_excluded": result.n_odds_band_excluded,
                "win_ev_odds_excluded": result.n_win_ev_odds_excluded,
                "win_stake_increased": result.n_win_stake_increased,
                "total_win_stake_increased": result.total_win_stake_increased,
                "excluded_odds_bands": result.exclusion_stats.get(
                    "odds_band_filter_excluded", {}
                ),
                "total_candidates_evaluated": result.exclusion_stats.get(
                    "total_candidates_evaluated", 0
                ),
            },
        }

        path = self.output_dir / "ai_diagnostics.json"
        path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_bet_history(self, path: Path) -> list[dict[str, Any]]:
        """bet_history JSON を読み込み"""
        data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def _derive_fields(self, bet_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """race_date, profit, is_win を派生フィールドとして追加"""
        if not bet_history:
            return []
        enriched = []
        for bet in bet_history:
            d = dict(bet)
            rid = str(bet.get("race_id", ""))
            if len(rid) >= 8:
                d["race_date"] = f"{rid[:4]}-{rid[4:6]}-{rid[6:8]}"
            else:
                d["race_date"] = ""
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

    def _compute_regime_stats(self, bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Regime別集計: ROI, 的中率, ベット数"""
        if not bets:
            return []
        regime_data: dict[str, dict[str, float]] = defaultdict(
            lambda: {"bets": 0, "wins": 0, "stake": 0.0, "total_return": 0.0}
        )
        for b in bets:
            regime = b.get("regime", "unknown")
            regime_data[regime]["bets"] += 1
            regime_data[regime]["stake"] += b["stake"]
            if b["result"] > 0:
                regime_data[regime]["wins"] += 1
                regime_data[regime]["total_return"] += b["result"]
        result: list[dict[str, Any]] = []
        for regime in ["aggressive", "conservative", "collapsed"]:
            if regime not in regime_data:
                continue
            s = regime_data[regime]
            n = s["bets"]
            result.append({
                "regime": regime,
                "bets": n,
                "wins": int(s["wins"]),
                "win_rate": s["wins"] / n if n > 0 else 0.0,
                "roi": s["total_return"] / s["stake"] if s["stake"] > 0 else 0.0,
            })
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

    def _compute_condition_stats(
        self,
        bets: list[dict[str, Any]],
        betting_target: str = "place",
    ) -> dict[str, Any]:
        """路面×距離帯、人気帯、EV帯の集計。win 時はオッズ倍率帯とregime帯も追加"""
        if not bets:
            return {
                "surface_distance": [],
                "popularity_bands": [],
                "ev_bands": [],
                "odds_multiplier_bands": [],
                "regime_bands": [],
            }

        # --- Surface × Distance Band ---
        sd_groups: dict[str, dict[str, float]] = defaultdict(
            lambda: {"bets": 0, "wins": 0, "stake": 0.0, "total_return": 0.0}
        )
        for b in bets:
            band = self._distance_band(b["surface"], b["kyori"])
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
                lambda: {"bets": 0, "wins": 0, "total_payout": 0.0, "total_stake": 0.0}
            )
            for b in bets_list:
                band = key_fn(b)
                groups[band]["bets"] += 1
                groups[band]["total_stake"] += b["stake"]
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
                        "roi": (
                            g["total_payout"] / g["total_stake"] if g["total_stake"] > 0 else 0.0
                        ),
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

        # --- Win-specific bands (D-01, D-03, RPT-03) ---
        odds_multiplier_bands: list[dict[str, Any]] = []
        regime_bands: list[dict[str, Any]] = []
        if betting_target == "win":
            odds_multiplier_bands = _band_stats(
                bets,
                lambda b: (
                    "1.0-3.0" if b.get("odds", 0) < 3.0
                    else "3.0-10.0" if b.get("odds", 0) < 10.0
                    else "10.0-30.0" if b.get("odds", 0) < 30.0
                    else "30.0+"
                ),
                ["1.0-3.0", "3.0-10.0", "10.0-30.0", "30.0+"],
            )
            regime_bands = self._compute_regime_stats(bets)

        return {
            "surface_distance": surface_distance,
            "popularity_bands": popularity_bands,
            "ev_bands": ev_bands,
            "odds_multiplier_bands": odds_multiplier_bands,
            "regime_bands": regime_bands,
        }

    def _compute_daily_stats(self, bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """日次集計: 的中率, 回収率, ベット数, 投資額, 払戻額"""
        if not bets:
            return []
        daily: dict[str, dict[str, float]] = defaultdict(
            lambda: {"bets": 0, "wins": 0, "stake": 0.0, "total_return": 0.0}
        )
        for b in bets:
            day = b["race_date"][:10]  # "YYYY-MM-DD"
            daily[day]["bets"] += 1
            daily[day]["stake"] += b["stake"]
            if b["result"] > 0:
                daily[day]["wins"] += 1
                daily[day]["total_return"] += b["result"]

        result = []
        for day, stats in sorted(daily.items()):
            bets_count = stats["bets"]
            result.append(
                {
                    "date": day,
                    "bets": bets_count,
                    "wins": int(stats["wins"]),
                    "win_rate": stats["wins"] / bets_count if bets_count > 0 else 0.0,
                    "stake": stats["stake"],
                    "total_return": stats["total_return"],
                    "roi": stats["total_return"] / stats["stake"] if stats["stake"] > 0 else 0.0,
                }
            )
        return result

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


class MultiYearReportGenerator:
    """マルチ年度バックテスト結果から自己完結型HTMLレポートを生成"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._single_gen = BacktestReportGenerator(output_dir)

    def generate(
        self,
        results: dict[int, BacktestResult],
        metadata: dict[int, dict[str, str]],
        betting_target: str = "place",
    ) -> Path:
        """マルチ年度HTMLレポートを生成"""
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
        env.filters["format_number"] = lambda x: f"{x:,.0f}"
        template = env.get_template("multi_year_report.html")

        # 年度別データを組み立て
        year_data: dict[int, dict[str, Any]] = {}
        for year, result in sorted(results.items()):
            enriched = self._single_gen._derive_fields(result.bet_history)
            meta = metadata.get(year, {})
            year_data[year] = {
                "summary": {
                    "roi": result.total_roi,
                    "win_rate": (
                        result.winning_bets / result.total_bets if result.total_bets > 0 else 0.0
                    ),
                    "profit": result.profit,
                    "max_dd": result.max_drawdown,
                    "final_bankroll": result.final_bankroll,
                    "total_bets": result.total_bets,
                    "total_stake": result.total_stake,
                    "total_return": result.total_return,
                    "total_wins": result.winning_bets,
                    "train_period": f"{meta.get('train_start', '')} ~ {meta.get('train_end', '')}",
                    "test_period": f"{meta.get('test_start', '')} ~ {meta.get('test_end', '')}",
                    "train_seconds": int(float(meta.get("train_seconds", "0"))),
                    "test_seconds": int(float(meta.get("test_seconds", "0"))),
                },
                "monthly_stats": self._single_gen._compute_monthly_stats(enriched),
                "daily_stats": self._single_gen._compute_daily_stats(enriched),
                "condition_stats": self._single_gen._compute_condition_stats(
                    enriched, betting_target=betting_target
                ),
                "bankroll_series": self._single_gen._compute_bankroll_series(enriched),
                "bet_details": enriched,
            }

        # 全体サマリー計算
        all_bets = sum(r.total_bets for r in results.values()) if results else 0
        all_stake = sum(r.total_stake for r in results.values()) if results else 0.0
        all_return = sum(r.total_return for r in results.values()) if results else 0.0
        best_year: int = max(results, key=lambda y: results[y].total_roi) if results else 0
        worst_year: int = min(results, key=lambda y: results[y].total_roi) if results else 0
        overall: dict[str, Any] = {
            "total_bets": all_bets,
            "total_stake": all_stake,
            "total_return": all_return,
            "profit": all_return - all_stake,
            "roi": all_return / all_stake if all_stake > 0 else 0.0,
            "best_year": best_year,
            "worst_year": worst_year,
            "best_roi": results[best_year].total_roi if results else 0.0,
            "worst_roi": results[worst_year].total_roi if results else 0.0,
        }

        html = template.render(
            year_data=year_data,
            overall=overall,
            years=sorted(results.keys()),
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        outpath = self.output_dir / "multi_year_report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath
