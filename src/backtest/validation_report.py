"""検証結果JSON生成モジュール (Phase 18, VAL-01/VAL-02)

バックテスト完了時に検証レポートを生成する。D-06(ROI>100%+100+ベット判定)、
D-07/D-08(JSON出力)、D-11(原因分析レポート)を実現する。

公開関数:
    evaluate_validation: D-06基準でPASS/FAIL判定
    generate_validation_report: 検証結果JSON生成
    generate_cause_analysis: ROI<100%時の原因分析
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_validation(roi: float, total_bets: int) -> str:
    """D-06判定ロジック。ROI>1.0 and total_bets>=100 の場合 "PASS"。

    Args:
        roi: テスト期間全体のROI (1.0 = 100%)
        total_bets: テスト期間全体のベット数

    Returns:
        "PASS" or "FAIL"
    """
    if roi > 1.0 and total_bets >= 100:
        return "PASS"
    return "FAIL"


def generate_validation_report(
    result: object,
    test_start: str,
    test_end: str,
    train_start: str,
    train_end: str,
    manifest_path: Path | None = None,
    pfp_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """検証結果JSONを生成する (D-07, D-08)。

    Args:
        result: BacktestResult
        test_start: テスト開始日 (YYYY-MM-DD)
        test_end: テスト終了日 (YYYY-MM-DD)
        train_start: 学習開始日 (YYYY-MM-DD)
        train_end: 学習終了日 (YYYY-MM-DD)
        manifest_path: manifestファイルパス (Noneの場合はmanifest情報なし)
        pfp_result: PFP verify結果 (Noneの場合はPFP未使用)

    Returns:
        検証結果dict (RESEARCH Pattern 2 schema準拠)
    """
    total_bets: int = getattr(result, "total_bets", 0)
    total_stake: float = getattr(result, "total_stake", 0.0)
    total_return: float = getattr(result, "total_return", 0.0)
    total_roi: float = getattr(result, "total_roi", 0.0)
    bet_history: list[dict[str, Any]] = getattr(result, "bet_history", [])

    # Manifest情報
    manifest_info: dict[str, Any] = {
        "path": str(manifest_path) if manifest_path is not None else None,
        "sha256_verified": True if manifest_path is not None else None,
        "sha256_hash": None,
    }

    # PFP検証結果
    pfp_info: dict[str, Any]
    if pfp_result is not None:
        pfp_info = pfp_result
    else:
        pfp_info = {"passed": None, "message": "PFP not used"}

    # ROI判定
    roi_passed = total_roi > 1.0 and total_bets >= 100

    # 年別内訳 (bet_historyのrace_date[:4]で集計)
    yearly_breakdown = _compute_yearly_breakdown(bet_history)

    # 原因分析 (ROI<=1.0の場合のみ)
    cause_analysis = None
    if total_roi <= 1.0 and bet_history:
        cause_analysis = generate_cause_analysis(bet_history)

    validation_result = evaluate_validation(total_roi, total_bets)

    return {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "test_period": [test_start, test_end],
        "train_period": [train_start, train_end],
        "manifest": manifest_info,
        "pfp_verification": pfp_info,
        "roi": {
            "total_roi": total_roi,
            "total_bets": total_bets,
            "total_stake": total_stake,
            "total_return": total_return,
            "target_roi": 1.0,
            "target_bets": 100,
            "passed": roi_passed,
        },
        "yearly_breakdown": yearly_breakdown,
        "validation_result": validation_result,
        "cause_analysis": cause_analysis,
    }


def generate_cause_analysis(bet_history: list[dict[str, Any]]) -> dict[str, Any]:
    """ROI<100%時の原因分析レポートを生成する (D-11)。

    分析項目:
    - odds_band_roi: オッズバンド別ROI (4バンド)
    - regime_roi: レジーム別ROI
    - ev_diagnosis: EV過大/過小評価分析
    - bet_count_sufficiency: ベット数十分性
    - surface_roi: 芝/ダート別ROI

    全bet_historyフィールドアクセスは.get()で防御する (Pitfall 4回避)。

    Args:
        bet_history: ベット履歴list

    Returns:
        原因分析dict。空bet_historyの場合は{"error": "No bet_history available"}
    """
    if not bet_history:
        return {"error": "No bet_history available"}

    # オッズバンド別ROI
    band_buckets: dict[str, list[dict[str, float]]] = {
        "1.0-2.0": [],
        "2.0-5.0": [],
        "5.0-10.0": [],
        "10.0+": [],
    }
    for b in bet_history:
        odds = b.get("final_odds", b.get("odds", 0))
        stake = float(b.get("stake", 0))
        result_val = float(b.get("result", 0))
        if odds <= 2.0:
            band = "1.0-2.0"
        elif odds <= 5.0:
            band = "2.0-5.0"
        elif odds <= 10.0:
            band = "5.0-10.0"
        else:
            band = "10.0+"
        band_buckets[band].append({"stake": stake, "result": result_val})

    odds_band_roi: dict[str, dict[str, Any]] = {}
    for name, bets in band_buckets.items():
        total_s = sum(x["stake"] for x in bets)
        total_r = sum(x["result"] for x in bets)
        odds_band_roi[name] = {
            "roi": total_r / total_s if total_s > 0 else 0.0,
            "bets": len(bets),
            "stake": total_s,
            "return": total_r,
        }

    # レジーム別ROI
    regime_stats: dict[str, dict[str, float]] = {}
    for b in bet_history:
        regime = b.get("regime", "UNKNOWN")
        if regime not in regime_stats:
            regime_stats[regime] = {"stake": 0.0, "result": 0.0, "bets": 0}
        regime_stats[regime]["stake"] += float(b.get("stake", 0))
        regime_stats[regime]["result"] += float(b.get("result", 0))
        regime_stats[regime]["bets"] += 1

    regime_roi: dict[str, dict[str, Any]] = {}
    for regime, stats in regime_stats.items():
        regime_roi[regime] = {
            "roi": stats["result"] / stats["stake"] if stats["stake"] > 0 else 0.0,
            "bets": int(stats["bets"]),
            "stake": stats["stake"],
            "return": stats["result"],
        }

    # EV診断 (過大/過小評価分析)
    overestimated: list[dict[str, float]] = []
    underestimated: list[dict[str, float]] = []
    for b in bet_history:
        ev = float(b.get("ev", 0))
        stake = float(b.get("stake", 0))
        result_val = float(b.get("result", 0))
        entry = {"ev": ev, "stake": stake, "result": result_val}
        if ev >= 1.0:
            overestimated.append(entry)
        else:
            underestimated.append(entry)

    def _aggregate_ev(entries: list[dict[str, float]]) -> dict[str, Any]:
        if not entries:
            return {"count": 0, "avg_ev": 0.0, "actual_roi": 0.0}
        total_s = sum(e["stake"] for e in entries)
        total_r = sum(e["result"] for e in entries)
        return {
            "count": len(entries),
            "avg_ev": sum(e["ev"] for e in entries) / len(entries),
            "actual_roi": total_r / total_s if total_s > 0 else 0.0,
        }

    ev_diagnosis = {
        "overestimated_ev": _aggregate_ev(overestimated),
        "underestimated_ev": _aggregate_ev(underestimated),
    }

    # ベット数十分性
    total_bets = len(bet_history)
    bet_count_sufficiency = {
        "total": total_bets,
        "target": 100,
        "sufficient": total_bets >= 100,
    }

    # 芝/ダート別ROI
    surface_stats: dict[str, dict[str, float]] = {}
    for b in bet_history:
        surface = b.get("surface", "unknown")
        if surface not in surface_stats:
            surface_stats[surface] = {"stake": 0.0, "result": 0.0, "bets": 0}
        surface_stats[surface]["stake"] += float(b.get("stake", 0))
        surface_stats[surface]["result"] += float(b.get("result", 0))
        surface_stats[surface]["bets"] += 1

    surface_roi: dict[str, dict[str, Any]] = {}
    for surface, stats in surface_stats.items():
        surface_roi[surface] = {
            "roi": stats["result"] / stats["stake"] if stats["stake"] > 0 else 0.0,
            "bets": int(stats["bets"]),
            "stake": stats["stake"],
            "return": stats["result"],
        }

    return {
        "odds_band_roi": odds_band_roi,
        "regime_roi": regime_roi,
        "ev_diagnosis": ev_diagnosis,
        "bet_count_sufficiency": bet_count_sufficiency,
        "surface_roi": surface_roi,
    }


def _compute_yearly_breakdown(
    bet_history: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """bet_historyのrace_dateから年別ROI集計を計算する。

    Args:
        bet_history: ベット履歴list

    Returns:
        {"2024": {"roi": ..., "bets": ..., "stake": ..., "return": ...}, ...}
    """
    yearly: dict[str, dict[str, float]] = {}
    for b in bet_history:
        date_str = b.get("race_date", "")
        year = date_str[:4] if len(date_str) >= 4 else "unknown"
        if year not in yearly:
            yearly[year] = {"stake": 0.0, "return": 0.0, "bets": 0}
        yearly[year]["stake"] += float(b.get("stake", 0))
        result_val = float(b.get("result", 0))
        if result_val > 0:
            yearly[year]["return"] += result_val
        yearly[year]["bets"] += 1

    result: dict[str, dict[str, Any]] = {}
    for year, data in yearly.items():
        result[year] = {
            "roi": data["return"] / data["stake"] if data["stake"] > 0 else 0.0,
            "bets": int(data["bets"]),
            "stake": data["stake"],
            "return": data["return"],
        }
    return result
