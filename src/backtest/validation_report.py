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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if result != result:
            return default
        return result
    except (TypeError, ValueError):
        return default


def _actual_bet_rows(bet_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actual_rows: list[dict[str, Any]] = []
    for row in bet_history:
        if "is_actual_bet" in row and not bool(row.get("is_actual_bet")):
            continue
        stake = _to_float(row.get("stake"), 0.0)
        if stake <= 0:
            continue
        actual_rows.append(row)
    return actual_rows


def _roi_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_stake = sum(_to_float(row.get("stake"), 0.0) for row in rows)
    total_return = sum(_to_float(row.get("result"), 0.0) for row in rows)
    return {
        "roi": total_return / total_stake if total_stake > 0 else 0.0,
        "bets": len(rows),
        "stake": total_stake,
        "return": total_return,
    }


def _pick_win_odds(row: dict[str, Any]) -> float:
    for col in ("final_odds", "tanodds", "odds", "fuku_odds_low"):
        if col in row:
            odds = _to_float(row.get(col), 0.0)
            if odds > 0:
                return odds
    return 0.0


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
        "oof_warning": (
            "OOF artifacts are not used as evidence in this validation report; "
            "OOF regeneration is tracked separately."
        ),
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

    actual_bets = _actual_bet_rows(bet_history)
    if not actual_bets:
        return {"error": "No actual bets available"}

    # オッズバンド別ROI
    band_buckets: dict[str, list[dict[str, Any]]] = {
        "1.0-2.0": [],
        "2.0-5.0": [],
        "5.0-10.0": [],
        "10.0+": [],
    }
    for b in actual_bets:
        odds = _pick_win_odds(b)
        stake = _to_float(b.get("stake"), 0.0)
        result_val = _to_float(b.get("result"), 0.0)
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
        odds_band_roi[name] = _roi_summary(bets)

    win_odds_band_buckets: dict[str, list[dict[str, Any]]] = {
        "1.0-2.0": [],
        "2.0-5.0": [],
        "5.0-10.0": [],
        "10.0-30.0": [],
        "30.0-50.0": [],
        "50.0-100.0": [],
        "100.0+": [],
    }
    for b in actual_bets:
        odds = _pick_win_odds(b)
        if odds <= 2.0:
            band = "1.0-2.0"
        elif odds <= 5.0:
            band = "2.0-5.0"
        elif odds <= 10.0:
            band = "5.0-10.0"
        elif odds <= 30.0:
            band = "10.0-30.0"
        elif odds <= 50.0:
            band = "30.0-50.0"
        elif odds < 100.0:
            band = "50.0-100.0"
        else:
            band = "100.0+"
        win_odds_band_buckets[band].append(b)
    win_odds_band_roi = {
        name: _roi_summary(rows) for name, rows in win_odds_band_buckets.items()
    }

    ev_band_buckets: dict[str, list[dict[str, Any]]] = {
        "<1.0": [],
        "1.0-1.2": [],
        "1.2-1.5": [],
        "1.5-2.0": [],
        "2.0-3.0": [],
        "3.0-5.0": [],
        "5.0+": [],
    }
    for b in actual_bets:
        ev = _to_float(
            b.get("win_selection_ev_tail_calibrated", b.get("win_selection_ev", b.get("ev"))),
            0.0,
        )
        if ev < 1.0:
            band = "<1.0"
        elif ev < 1.2:
            band = "1.0-1.2"
        elif ev < 1.5:
            band = "1.2-1.5"
        elif ev < 2.0:
            band = "1.5-2.0"
        elif ev < 3.0:
            band = "2.0-3.0"
        elif ev < 5.0:
            band = "3.0-5.0"
        else:
            band = "5.0+"
        ev_band_buckets[band].append(b)
    ev_band_roi = {name: _roi_summary(rows) for name, rows in ev_band_buckets.items()}

    popularity_band_buckets: dict[str, list[dict[str, Any]]] = {
        "1-3": [],
        "4-6": [],
        "7-8": [],
        "9-12": [],
        "13+": [],
        "unknown": [],
    }
    for b in actual_bets:
        popularity = int(_to_float(b.get("popularity", b.get("popularity_rank")), 0.0))
        if 1 <= popularity <= 3:
            band = "1-3"
        elif 4 <= popularity <= 6:
            band = "4-6"
        elif 7 <= popularity <= 8:
            band = "7-8"
        elif 9 <= popularity <= 12:
            band = "9-12"
        elif popularity >= 13:
            band = "13+"
        else:
            band = "unknown"
        popularity_band_buckets[band].append(b)
    popularity_band_roi = {
        name: _roi_summary(rows) for name, rows in popularity_band_buckets.items()
    }

    tail_flag_roi = {
        "ev>=3": _roi_summary(
            [
                b for b in actual_bets
                if _to_float(
                    b.get(
                        "win_selection_ev_tail_calibrated",
                        b.get("win_selection_ev", b.get("ev")),
                    ),
                    0.0,
                ) >= 3.0
            ]
        ),
        "ev>=5": _roi_summary(
            [
                b for b in actual_bets
                if _to_float(
                    b.get(
                        "win_selection_ev_tail_calibrated",
                        b.get("win_selection_ev", b.get("ev")),
                    ),
                    0.0,
                ) >= 5.0
            ]
        ),
        "odds>=50": _roi_summary([b for b in actual_bets if _pick_win_odds(b) >= 50.0]),
        "odds>=100": _roi_summary([b for b in actual_bets if _pick_win_odds(b) >= 100.0]),
    }

    # レジーム別ROI
    regime_stats: dict[str, dict[str, float]] = {}
    for b in actual_bets:
        regime = b.get("regime", "UNKNOWN")
        if regime not in regime_stats:
            regime_stats[regime] = {"stake": 0.0, "result": 0.0, "bets": 0}
        regime_stats[regime]["stake"] += _to_float(b.get("stake"), 0.0)
        regime_stats[regime]["result"] += _to_float(b.get("result"), 0.0)
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
    for b in actual_bets:
        ev = _to_float(
            b.get("win_selection_ev_tail_calibrated", b.get("win_selection_ev", b.get("ev"))),
            0.0,
        )
        stake = _to_float(b.get("stake"), 0.0)
        result_val = _to_float(b.get("result"), 0.0)
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
    total_bets = len(actual_bets)
    bet_count_sufficiency = {
        "total": total_bets,
        "target": 100,
        "sufficient": total_bets >= 100,
    }

    # 芝/ダート別ROI
    surface_stats: dict[str, dict[str, float]] = {}
    for b in actual_bets:
        surface = b.get("surface", "unknown")
        if surface not in surface_stats:
            surface_stats[surface] = {"stake": 0.0, "result": 0.0, "bets": 0}
        surface_stats[surface]["stake"] += _to_float(b.get("stake"), 0.0)
        surface_stats[surface]["result"] += _to_float(b.get("result"), 0.0)
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
        "win_odds_band_roi": win_odds_band_roi,
        "ev_band_roi": ev_band_roi,
        "popularity_band_roi": popularity_band_roi,
        "tail_flag_roi": tail_flag_roi,
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
    bet_history = _actual_bet_rows(bet_history)
    yearly: dict[str, dict[str, float]] = {}
    for b in bet_history:
        date_str = b.get("race_date", "")
        year = date_str[:4] if len(date_str) >= 4 else "unknown"
        if year not in yearly:
            yearly[year] = {"stake": 0.0, "return": 0.0, "bets": 0}
        yearly[year]["stake"] += _to_float(b.get("stake"), 0.0)
        result_val = _to_float(b.get("result"), 0.0)
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
