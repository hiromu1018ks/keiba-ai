"""Spike 003: キャリブレーションBT期間短縮の精度影響を検証する。

使い方:
  python .planning/spikes/003-calibration-shorten/verify_calibration_shorten.py

既存のbet_history.jsonを使って、異なる期間窓での
OddsBandFilterキャリブレーション結果を比較する。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from betting.odds_band_filter import OddsBandFilter

BANDS = [(1.0, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, float("inf"))]
BAND_NAMES = ["1.0-3.0", "3.0-10.0", "10.0-30.0", "30.0+"]


def _get_band_name(odds: float) -> str:
    for (lo, hi), name in zip(BANDS, BAND_NAMES):
        if lo <= odds < hi:
            return name
    return "30.0+"


def load_bet_history(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_band_stats(bets: list[dict]) -> dict:
    """OddsBandFilter.calibrate() と同じ統計を計算。"""
    band_data = {
        name: {"total_stake": 0.0, "total_return": 0.0, "count": 0}
        for name in BAND_NAMES
    }

    for bet in bets:
        odds = bet.get("odds", 0)
        if odds < 1.0:
            continue
        band = _get_band_name(odds)
        band_data[band]["total_stake"] += bet.get("stake", 0)
        band_data[band]["total_return"] += bet.get("result", 0)
        band_data[band]["count"] += 1

    result = {}
    for name, data in band_data.items():
        roi = data["total_return"] / data["total_stake"] if data["total_stake"] > 0 else 0
        result[name] = {
            "roi": round(roi, 4),
            "count": data["count"],
            "total_stake": round(data["total_stake"], 0),
            "total_return": round(data["total_return"], 0),
            "excluded": roi < 1.0,
        }
    return result


def filter_bets_by_window(bets: list[dict], months: int) -> list[dict]:
    """直近Nヶ月のベットのみを抽出。"""
    if months <= 0:
        return bets

    dates = [b.get("race_date", "") for b in bets]
    max_date = max(dates)
    cutoff = datetime.strptime(max_date, "%Y-%m-%d") - timedelta(days=months * 30)

    return [b for b in bets if datetime.strptime(b.get("race_date", ""), "%Y-%m-%d") >= cutoff]


def compare_windows(bets: list[dict]) -> list[dict]:
    """異なる期間窓でバンド統計を比較。"""
    windows = [
        ("full", 0),
        ("24mo", 24),
        ("12mo", 12),
        ("9mo", 9),
        ("6mo", 6),
        ("3mo", 3),
    ]

    results = []
    for label, months in windows:
        filtered = filter_bets_by_window(bets, months)
        stats = compute_band_stats(filtered)
        excluded_bands = [name for name, data in stats.items() if data["excluded"]]

        results.append({
            "window": label,
            "months": months,
            "total_bets": len(filtered),
            "excluded_bands": excluded_bands,
            "band_stats": stats,
        })

    return results


def compute_roi_series(bets: list[dict], period_months: int = 1) -> list[dict]:
    """期間ごとのROI推移を計算（安定性確認用）。"""
    if not bets:
        return []

    dates = sorted(set(b.get("race_date", "")[:7] for b in bets))  # YYYY-MM
    series = []

    for ym in dates:
        period_bets = [b for b in bets if b.get("race_date", "").startswith(ym)]
        stats = compute_band_stats(period_bets)
        series.append({
            "period": ym,
            "bets": len(period_bets),
            "band_roi": {name: data["roi"] for name, data in stats.items() if data["count"] > 0},
        })

    return series


def main() -> None:
    print("=" * 70)
    print("  Spike 003: Calibration Period Shortening Verification")
    print("=" * 70)

    # bet_history.json をロード（テスト期間 = 2024年）
    bet_paths = [
        ROOT / "data" / "backtest" / "bet_history.json",
        ROOT / "data" / "backtest" / "multi_year_bet_history.json",
    ]

    all_bets: list[dict] = []
    for path in bet_paths:
        if path.exists():
            bets = load_bet_history(str(path))
            dates = [b.get("race_date", "") for b in bets if b.get("race_date")]
            print(f"\n  Loaded: {path.name}")
            print(f"    Bets: {len(bets)}, Range: {min(dates)} ~ {max(dates)}")
            all_bets.extend(bets)

    if not all_bets:
        print("  ERROR: No bet_history found")
        return

    # 1. フル期間のベースライン統計
    print("\n" + "-" * 70)
    print("  [1] Full Period Band Statistics (Baseline)")
    print("-" * 70)
    full_stats = compute_band_stats(all_bets)
    for name, data in full_stats.items():
        status = "EXCLUDED" if data["excluded"] else "OK"
        print(f"    {name:>10s}: ROI={data['roi']:.4f}, n={data['count']:>5d}, "
              f"stake={data['total_stake']:>10.0f}, return={data['total_return']:>10.0f}  [{status}]")

    # 2. 期間窓比較
    print("\n" + "-" * 70)
    print("  [2] Window Comparison")
    print("-" * 70)
    comparisons = compare_windows(all_bets)

    full_excluded = comparisons[0]["excluded_bands"]
    print(f"\n    {'Window':>6s} | {'Bets':>6s} | {'Excluded Bands':>20s} | Match? | Details")
    print(f"    {'-'*6}-+-{'-'*6}-+-{'-'*20}-+-{'-'*6}-+-{'-'*40}")

    for comp in comparisons:
        excluded = comp["excluded_bands"]
        match = excluded == full_excluded
        details = []
        for band in BAND_NAMES:
            roi = comp["band_stats"][band]["roi"]
            n = comp["band_stats"][band]["count"]
            details.append(f"{band}={roi:.3f}(n={n})")
        detail_str = ", ".join(details)

        print(f"    {comp['window']:>6s} | {comp['total_bets']:>6d} | "
              f"{','.join(excluded) if excluded else '(none)':>20s} | "
              f"{'OK' if match else 'DIFF':>6s} | {detail_str}")

    # 3. 月次ROI推移（安定性確認）
    print("\n" + "-" * 70)
    print("  [3] Monthly ROI Stability (Per-Band)")
    print("-" * 70)
    series = compute_roi_series(all_bets)
    print(f"\n    {'Month':>8s}", end="")
    for band in BAND_NAMES:
        print(f" | {band:>10s}", end="")
    print()
    print(f"    {'-'*8}", end="")
    for _ in BAND_NAMES:
        print(f"-+-{'-'*10}", end="")
    print()

    for entry in series:
        print(f"    {entry['period']:>8s}", end="")
        for band in BAND_NAMES:
            roi = entry["band_roi"].get(band, None)
            if roi is not None:
                marker = " X" if roi < 1.0 else ""
                print(f" | {roi:>8.3f}{marker:>2s}", end="")
            else:
                print(f" | {'--':>10s}", end="")
        print()

    # 4. 統計的有意性チェック
    print("\n" + "-" * 70)
    print("  [4] Statistical Significance: Min Bets for Stable Band Decision")
    print("-" * 70)

    full_baseline = comparisons[0]
    for band in BAND_NAMES:
        n_full = full_baseline["band_stats"][band]["count"]
        roi_full = full_baseline["band_stats"][band]["roi"]
        print(f"\n    {band}:")
        print(f"      Full period: ROI={roi_full:.4f}, n={n_full}")

        for comp in comparisons[1:]:
            n_win = comp["band_stats"][band]["count"]
            roi_win = comp["band_stats"][band]["roi"]
            same_decision = (roi_win < 1.0) == (roi_full < 1.0)
            print(f"      {comp['window']:>6s}: ROI={roi_win:.4f}, n={n_win:>4d}, "
                  f"same_decision={'YES' if same_decision else 'NO':>3s}")

    # 5. 結論
    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)

    stable_windows = []
    for comp in comparisons[1:]:
        if comp["excluded_bands"] == full_excluded:
            stable_windows.append(comp["window"])

    if stable_windows:
        best = stable_windows[-1]
        best_comp = next(c for c in comparisons if c["window"] == best)
        savings_pct = (1 - best_comp["total_bets"] / comparisons[0]["total_bets"]) * 100
        print(f"\n  Shortest stable window: {best} ({best_comp['total_bets']} bets)")
        print(f"  Same exclusion bands as full period: {full_excluded}")
        print(f"  Estimated bet reduction: {savings_pct:.0f}%")
        print(f"\n  RECOMMENDATION: Calibration BT can be shortened to {best}")
        print(f"  without changing OddsBandFilter exclusion decisions.")
    else:
        print("\n  WARNING: No shorter window produces identical exclusion bands.")
        print("  Calibration period shortening may affect accuracy.")

    # 結果をJSONに保存
    output = {
        "full_stats": full_stats,
        "comparisons": comparisons,
        "monthly_series": series,
        "stable_windows": stable_windows,
    }
    output_path = Path(__file__).parent / "calibration_analysis.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
