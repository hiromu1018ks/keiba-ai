#!/usr/bin/env python
"""Market Excess OOF lightweight experiment.

Target variable: market_excess_diff = is_win - p_market_win_norm
- Learns "how much a horse beats/falls short of market expectation"
- NOT a new P-model; a meta-model verification using existing pipeline outputs

Train: 2024 backtest data / Test: 2025 backtest data
ROI primary: confirmed_odds (payout-based) / reference: tanodds

Output:
  data/analysis/win_market_excess_oof.json
  data/analysis/win_market_excess_oof.md
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtest"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
YEARS = [2024, 2025]

QUANTILE_CUTS = [0.0, 0.20, 0.40, 0.60, 0.80, 0.95, 0.99, 1.0]
QUANTILE_LABELS = [
    "0-20%", "20-40%", "40-60%", "60-80%", "80-95%", "95-99%", "99-100%",
]

NEED_COLS = [
    "race_id", "umaban", "kakuteijyuni", "surface", "tanodds", "confirmed_odds",
    "p_win_pred", "p_win_corrected", "p_win_final", "p_market_win_norm",
    "ev_win", "ev_win_corrected", "ev_win_calibrated",
    "popularity_rank", "field_size", "p_ability_win",
    "market_entropy", "overround", "odds_skewness",
    "win_market_logit_edge", "win_market_prob_ratio", "win_market_value_ratio",
    "track_condition_code", "distance_bin", "grade_code",
]

# Features for the meta-model (model output + market diff only)
FEATURE_COLS = [
    "p_win_pred", "p_win_corrected", "p_win_final", "p_market_win_norm",
    "edge_diff_pred", "edge_diff_final",
    "edge_ratio_pred", "edge_ratio_final",
    "ev_win", "ev_win_corrected", "ev_win_calibrated",
    "tanodds", "popularity_rank",
    "surface_enc", "field_size",
    "p_ability_win", "market_entropy", "overround",
    # Optional features (may be useful)
    "odds_skewness", "win_market_logit_edge",
    "win_market_prob_ratio", "win_market_value_ratio",
    "track_condition_code",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load backtest horse features for 2024/2025."""
    frames: list[pd.DataFrame] = []
    for year in YEARS:
        path = BACKTEST_DIR / f"bt_{year}_horse_features.parquet"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        # Read full file, then select needed columns
        df = pd.read_parquet(path)
        available = df.columns.tolist()
        cols_to_keep = [c for c in NEED_COLS if c in available]
        df = df[cols_to_keep].copy()
        df["year"] = year
        frames.append(df)
        print(f"  Loaded {path.name}: {len(df)} rows, {len(cols_to_keep)} cols")

    if not frames:
        print("ERROR: No data loaded")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)

    # Basic filtering
    df = df[df["tanodds"] > 0].copy()
    df["is_win"] = (df["kakuteijyuni"] == 1).astype(int)
    df["race_id"] = df["race_id"].astype(str)

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features for the meta-model."""
    df = df.copy()

    # Edge features (model vs market)
    p_mkt = df["p_market_win_norm"].clip(1e-6, 1 - 1e-6)
    df["edge_diff_pred"] = df["p_win_pred"] - p_mkt
    df["edge_diff_final"] = df["p_win_final"] - p_mkt
    df["edge_ratio_pred"] = (df["p_win_pred"] / p_mkt).clip(0.01, 100.0)
    df["edge_ratio_final"] = (df["p_win_final"] / p_mkt).clip(0.01, 100.0)

    # Surface encoding
    df["surface_enc"] = (df["surface"] == "dirt").astype(int)

    return df


def compute_target(df: pd.DataFrame) -> pd.Series:
    """Compute market excess target: is_win - p_market."""
    return df["is_win"] - df["p_market_win_norm"]


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------
def roi_summary(
    df: pd.DataFrame,
    odds_col: str = "confirmed_odds",
    win_col: str = "is_win",
) -> dict:
    """Compute ROI assuming 100 yen flat bet per horse."""
    n = len(df)
    if n == 0:
        return {
            "count": 0, "hit_rate": 0.0, "avg_odds": 0.0,
            "roi": 0.0, "profit": 0,
        }
    total_payout = (df[odds_col] * df[win_col]).sum() * 100
    total_bet = n * 100
    roi = total_payout / total_bet
    return {
        "count": n,
        "hit_rate": float(df[win_col].mean()),
        "avg_odds": float(df[odds_col].mean()),
        "roi": float(roi),
        "profit": float(total_payout - total_bet),
    }


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.Series, pd.Series, dict]:
    """Train LightGBM regression on 2024, predict on 2025.

    Returns (train_pred, test_pred, model_info).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not available, falling back to sklearn HistGradientBoosting")
        return _train_sklearn(train_df, test_df, feature_cols)

    target_train = compute_target(train_df)
    target_test = compute_target(test_df)

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = target_train.values

    params = {
        "objective": "regression_l1",
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    train_pred = pd.Series(model.predict(X_train), index=train_df.index)
    test_pred = pd.Series(model.predict(X_test), index=test_df.index)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))

    # Train metrics
    train_mae = float(np.mean(np.abs(y_train - train_pred.values)))
    train_corr = float(np.corrcoef(y_train, train_pred.values)[0, 1])

    model_info = {
        "library": "lightgbm",
        "params": {k: v for k, v in params.items() if k != "verbose"},
        "train_mae": train_mae,
        "train_corr": train_corr,
        "feature_importance": importance,
        "train_n": len(train_df),
        "test_n": len(test_df),
    }

    return train_pred, test_pred, model_info


def _train_sklearn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.Series, pd.Series, dict]:
    """Fallback: sklearn HistGradientBoostingRegressor."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    target_train = compute_target(train_df)

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = target_train.values

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=300,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=50,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_pred = pd.Series(model.predict(X_train), index=train_df.index)
    test_pred = pd.Series(model.predict(X_test), index=test_df.index)

    train_mae = float(np.mean(np.abs(y_train - train_pred.values)))
    train_corr = float(np.corrcoef(y_train, train_pred.values)[0, 1])

    model_info = {
        "library": "sklearn_histgb",
        "train_mae": train_mae,
        "train_corr": train_corr,
        "train_n": len(train_df),
        "test_n": len(test_df),
    }

    return train_pred, test_pred, model_info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_quantile_roi(
    df: pd.DataFrame,
    pred_col: str,
) -> list[dict]:
    """Quantile-based ROI analysis (7 bins)."""
    df = df.copy()
    df["_pred"] = df[pred_col]
    # Compute quantile bin edges from the data
    bin_edges = df["_pred"].quantile(QUANTILE_CUTS).tolist()
    # Ensure edges are unique (round to avoid duplicate edges)
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1e-9
    df["_quantile"] = pd.cut(
        df["_pred"],
        bins=bin_edges,
        labels=QUANTILE_LABELS,
        include_lowest=True,
    )

    results = []
    for label in QUANTILE_LABELS:
        sub = df[df["_quantile"] == label]
        payout_roi = roi_summary(sub, "confirmed_odds")
        tanodds_roi = roi_summary(sub, "tanodds")
        results.append({
            "quantile": label,
            "count": payout_roi["count"],
            "hit_rate": payout_roi["hit_rate"],
            "avg_confirmed_odds": payout_roi["avg_odds"],
            "avg_tanodds": tanodds_roi["avg_odds"],
            "roi_payout": payout_roi["roi"],
            "roi_tanodds": tanodds_roi["roi"],
            "profit_payout": payout_roi["profit"],
            "avg_excess_actual": float((sub["is_win"] - sub["p_market_win_norm"]).mean())
                if len(sub) > 0 else 0.0,
        })

    return results


def compute_correlations(df: pd.DataFrame, pred_col: str) -> dict:
    """Correlation analysis for market_excess_pred."""
    pred = df[pred_col]
    actual_excess = df["is_win"] - df["p_market_win_norm"]

    def _corr(a: pd.Series, b: pd.Series) -> float:
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(a.corr(b))

    return {
        "pred_vs_actual_excess": _corr(pred, actual_excess),
        "pred_vs_is_win": _corr(pred, df["is_win"]),
        "pred_vs_p_market": _corr(pred, df["p_market_win_norm"]),
        "pred_vs_tanodds": _corr(pred, df["tanodds"]),
        "pred_vs_ev_win": _corr(pred, df["ev_win"]),
        "pred_vs_edge_ratio_final": _corr(
            pred,
            (df["p_win_final"] / df["p_market_win_norm"].clip(1e-6)).clip(0.01, 100),
        ),
    }


def compare_baselines(df: pd.DataFrame, pred_col: str) -> list[dict]:
    """Compare market_excess_pred top-K% ROI vs existing metrics."""
    df = df.copy()

    # Baseline scoring columns
    baselines = [
        ("market_excess_pred", pred_col),
        ("edge_ratio_final", "edge_ratio_final"),
        ("pred_ev_final", None),  # computed inline
        ("ev_win", "ev_win"),
        ("win_market_value_ratio", "win_market_value_ratio"),
        ("win_market_logit_edge", "win_market_logit_edge"),
    ]

    # Compute pred_ev_final
    df["pred_ev_final"] = df["p_win_final"] * df["tanodds"]

    results = []
    for name, col in baselines:
        if col is None:
            col = name
        if col not in df.columns:
            continue

        top_pcts = [0.01, 0.05, 0.10, 0.20]
        entry = {"name": name, "col": col}
        for pct in top_pcts:
            threshold = df[col].quantile(1 - pct)
            top = df[df[col] >= threshold]
            payout_r = roi_summary(top, "confirmed_odds")
            tanodds_r = roi_summary(top, "tanodds")
            entry[f"top_{int(pct*100)}pct"] = {
                "count": payout_r["count"],
                "roi_payout": payout_r["roi"],
                "roi_tanodds": tanodds_r["roi"],
                "avg_confirmed_odds": payout_r["avg_odds"],
                "hit_rate": payout_r["hit_rate"],
            }
        results.append(entry)

    return results


def evaluate_by_surface(df: pd.DataFrame, pred_col: str) -> dict:
    """Year x Surface breakdown."""
    results = {}
    for year in sorted(df["year"].unique()):
        for surface in ["turf", "dirt"]:
            sub = df[(df["year"] == year) & (df["surface"] == surface)]
            if len(sub) < 10:
                continue
            key = f"{year}_{surface}"

            # Top 10% ROI
            threshold = sub[pred_col].quantile(0.90)
            top10 = sub[sub[pred_col] >= threshold]
            payout = roi_summary(top10, "confirmed_odds")
            tanodds = roi_summary(top10, "tanodds")

            results[key] = {
                "n_total": len(sub),
                "n_top10": len(top10),
                "top10_roi_payout": payout["roi"],
                "top10_roi_tanodds": tanodds["roi"],
                "top10_hit_rate": payout["hit_rate"],
                "top10_avg_confirmed_odds": payout["avg_odds"],
            }
    return results


def sanity_check(
    quantile_results: list[dict],
    correlations: dict,
    baseline_results: list[dict],
) -> dict:
    """Run sanity checks and return pass/fail."""
    checks = {}

    # 1. Top quantile ROI improvement
    top_roi = quantile_results[-1]["roi_payout"]  # 99-100% bin
    mid_roi = quantile_results[3]["roi_payout"]  # 60-80% bin
    checks["top_improves"] = {
        "pass": top_roi > mid_roi,
        "detail": f"top={top_roi:.3f} vs mid={mid_roi:.3f}",
    }

    # 2. Not dominated by few high-odds hits
    top_count = quantile_results[-1]["count"]
    top_avg_odds = quantile_results[-1]["avg_confirmed_odds"]
    checks["not_few_high_odds"] = {
        "pass": top_count >= 10 and top_avg_odds < 100,
        "detail": f"count={top_count}, avg_odds={top_avg_odds:.1f}",
    }

    # 3. Not just copying p_market or tanodds
    corr_market = abs(correlations["pred_vs_p_market"])
    corr_odds = abs(correlations["pred_vs_tanodds"])
    checks["not_copying_market"] = {
        "pass": corr_market < 0.9,
        "detail": f"corr_p_market={corr_market:.3f}",
    }
    checks["not_copying_odds"] = {
        "pass": corr_odds < 0.9,
        "detail": f"corr_tanodds={corr_odds:.3f}",
    }

    # 4. Compare with best baseline (top 5%)
    excess_top5 = None
    best_baseline_top5 = 0.0
    best_baseline_name = ""
    for b in baseline_results:
        roi = b.get("top_5pct", {}).get("roi_payout", 0)
        if b["name"] == "market_excess_pred":
            excess_top5 = roi
        elif roi > best_baseline_top5:
            best_baseline_top5 = roi
            best_baseline_name = b["name"]
    if excess_top5 is not None:
        checks["beats_baselines"] = {
            "pass": excess_top5 > best_baseline_top5,
            "detail": (
                f"excess_top5={excess_top5:.3f} "
                f"vs {best_baseline_name}={best_baseline_top5:.3f}"
            ),
        }

    all_pass = all(c["pass"] for c in checks.values())
    return {"all_pass": all_pass, "checks": checks}


def determine_verdict(
    sanity: dict,
    quantile_results: list[dict],
    test_df: pd.DataFrame,
    pred_col: str,
) -> str:
    """Determine final verdict: A/B/C/D."""
    if not sanity["checks"]:
        return "D: insufficient data"

    # D: data issues
    if len(test_df) < 100:
        return "D: insufficient test data"

    top_bin = quantile_results[-1]
    top5_roi = None
    for i, q in enumerate(quantile_results):
        # 80-95% + 95-99% + 99-100% = top ~20%, approximate top 5% as 95-100%
        if q["quantile"] in ("95-99%", "99-100%"):
            if top5_roi is None:
                top5_roi = q["roi_payout"]

    # C: dangerous (few high-odds hits)
    if not sanity["checks"]["not_few_high_odds"]["pass"]:
        return "C: dangerous (few high-odds hits)"

    # B: weak (doesn't beat baselines)
    if not sanity["checks"].get("beats_baselines", {}).get("pass", False):
        return "B: weak (doesn't beat existing metrics)"

    # A: promising (beats baselines + passes sanity)
    if sanity["all_pass"]:
        return "A: promising (beats baselines, passes all sanity checks)"

    # Partial pass
    if sanity["checks"].get("beats_baselines", {}).get("pass", False):
        return "B: weak (beats baselines but fails some sanity checks)"

    return "B: weak"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    results: dict,
) -> str:
    """Generate Markdown report."""
    lines: list[str] = []

    lines.append("# Market Excess OOF Lightweight Experiment")
    lines.append("")
    lines.append(f"Generated: {results['meta']['timestamp']}")
    lines.append("")

    # 1. Conclusion
    lines.append("## 1. Conclusion")
    lines.append("")
    verdict = results["verdict"]
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")

    # Caveat
    lines.append("> **Note:** This is a lightweight out-of-sample verification (2024→2025),")
    lines.append("> not a strict production OOF. Features include existing pipeline outputs")
    lines.append("> (p_win_final, edge_ratio, etc.) which already contain market-diff/MAWC/correction.")
    lines.append("> Good results do not guarantee independent new learning.")
    lines.append("")

    # 2. Target variable
    lines.append("## 2. Target Variable")
    lines.append("")
    lines.append("`market_excess_diff = is_win - p_market_win_norm`")
    lines.append("")
    lines.append("| Component | Description |")
    lines.append("|-----------|-------------|")
    lines.append("| is_win | `(kakuteijyuni == 1)` — actual win (1) or loss (0) |")
    lines.append("| p_market_win_norm | Market-implied probability, normalized within race |")
    lines.append("| market_excess_diff | Positive = beat market, Negative = fell short |")
    lines.append("")

    # 3. Data
    lines.append("## 3. Data")
    lines.append("")
    mi = results["model_info"]
    lines.append(f"- Train: 2024 ({mi['train_n']:,} rows)")
    lines.append(f"- Test: 2025 ({mi['test_n']:,} rows)")
    lines.append(f"- Features: {len(results['feature_importance'])} dimensions")
    lines.append(f"- Model: {mi['library']} (MAE objective)")
    lines.append(f"- Train MAE: {mi['train_mae']:.4f}")
    lines.append(f"- Train correlation: {mi['train_corr']:.4f}")
    lines.append("")

    # 4. Feature importance
    lines.append("## 4. Feature Importance (Top 10)")
    lines.append("")
    imp = results["feature_importance"]
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
    lines.append("| Rank | Feature | Importance |")
    lines.append("|------|---------|------------|")
    for i, (feat, val) in enumerate(sorted_imp, 1):
        lines.append(f"| {i} | {feat} | {val:.1f} |")
    lines.append("")

    # 5. Quantile ROI
    lines.append("## 5. market_excess_pred Quantile ROI (Test: 2025)")
    lines.append("")
    lines.append("| Quantile | N | Hit% | ROI(payout) | ROI(tanodds) | Avg CO | Avg Excess |")
    lines.append("|----------|---|------|-------------|--------------|--------|------------|")
    for q in results["quantile_roi"]:
        lines.append(
            f"| {q['quantile']} | {q['count']} | {q['hit_rate']:.1%} "
            f"| {q['roi_payout']:.1%} | {q['roi_tanodds']:.1%} "
            f"| {q['avg_confirmed_odds']:.1f} | {q['avg_excess_actual']:+.4f} |"
        )
    lines.append("")
    lines.append("> ROI(payout) = confirmed_odds-based (primary). ROI(tanodds) = reference.")
    lines.append("")

    # 6. Baseline comparison
    lines.append("## 6. Baseline Comparison (Test: 2025, Top 5%)")
    lines.append("")
    lines.append("| Metric | N | ROI(payout) | ROI(tanodds) | Hit% | Avg CO |")
    lines.append("|--------|---|-------------|--------------|------|--------|")
    for b in results["baselines"]:
        t5 = b.get("top_5pct", {})
        lines.append(
            f"| {b['name']} | {t5.get('count', 0)} "
            f"| {t5.get('roi_payout', 0):.1%} "
            f"| {t5.get('roi_tanodds', 0):.1%} "
            f"| {t5.get('hit_rate', 0):.1%} "
            f"| {t5.get('avg_confirmed_odds', 0):.1f} |"
        )
    lines.append("")

    # 7. Correlations
    lines.append("## 7. Correlation Analysis (Test: 2025)")
    lines.append("")
    corr = results["correlations"]
    lines.append("| Pair | Correlation |")
    lines.append("|------|------------|")
    for pair, val in corr.items():
        warn = " ⚠️" if abs(val) > 0.9 and "p_market" in pair or "tanodds" in pair else ""
        lines.append(f"| {pair} | {val:.4f}{warn} |")
    lines.append("")

    # 8. Year/Surface breakdown
    lines.append("## 8. Year × Surface Breakdown (Top 10%)")
    lines.append("")
    lines.append("| Group | N_total | N_top10 | ROI(payout) | ROI(tanodds) | Hit% | Avg CO |")
    lines.append("|-------|---------|---------|-------------|--------------|------|--------|")
    for key in sorted(results["surface_breakdown"].keys()):
        s = results["surface_breakdown"][key]
        lines.append(
            f"| {key} | {s['n_total']} | {s['n_top10']} "
            f"| {s['top10_roi_payout']:.1%} "
            f"| {s['top10_roi_tanodds']:.1%} "
            f"| {s['top10_hit_rate']:.1%} "
            f"| {s['top10_avg_confirmed_odds']:.1f} |"
        )
    lines.append("")

    # 9. Sanity check
    lines.append("## 9. Sanity Check")
    lines.append("")
    for name, check in results["sanity"]["checks"].items():
        status = "✅ PASS" if check["pass"] else "❌ FAIL"
        lines.append(f"- **{name}**: {status} — {check['detail']}")
    lines.append("")
    all_pass = results["sanity"]["all_pass"]
    lines.append(f"**Overall: {'ALL PASS ✅' if all_pass else 'SOME FAILURES ❌'}**")
    lines.append("")

    # 10. Next steps
    lines.append("## 10. Next Steps")
    lines.append("")
    if verdict.startswith("A"):
        lines.append("- market_excess target is promising → proceed to small-scale production OOF")
        lines.append("- Design strict walk-forward validation with proper OOF predictions")
        lines.append("- Consider training a standalone model (not using existing pipeline outputs)")
    elif verdict.startswith("B"):
        lines.append("- Existing features alone are insufficient for market excess target")
        lines.append("- Consider adding fundamental features (form, trainer, jockey stats)")
        lines.append("- May need to go back to feature/data source improvements")
    elif verdict.startswith("C"):
        lines.append("- Results depend on few high-odds hits — statistically unreliable")
        lines.append("- Need more data or different approach")
    else:
        lines.append("- Insufficient data or model training failed")
        lines.append("- Check data availability and model configuration")
    lines.append("")

    # 11. Execution command
    lines.append("## 11. Execution Command")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/analyze_win_market_excess_oof.py")
    lines.append("```")
    lines.append("")

    # 12. Generated files
    lines.append("## 12. Generated Files")
    lines.append("")
    lines.append("- `data/analysis/win_market_excess_oof.json`")
    lines.append("- `data/analysis/win_market_excess_oof.md`")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Market Excess OOF Lightweight Experiment")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"  Total: {len(df)} rows ({df['year'].value_counts().to_dict()})")

    # 2. Compute features
    print("\n[2/6] Computing features...")
    df = compute_features(df)

    # Resolve available features
    avail_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        print(f"  WARNING: Missing features: {missing_features}")
        print(f"  Using {len(avail_features)} available features")

    nan_count = df[avail_features].isna().sum().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in features, filling with 0")
        df[avail_features] = df[avail_features].fillna(0)

    # 3. Split
    train_df = df[df["year"] == 2024].copy()
    test_df = df[df["year"] == 2025].copy()
    print(f"  Train: {len(train_df)} / Test: {len(test_df)}")

    # 4. Train model
    print("\n[3/6] Training LightGBM regression...")
    train_pred, test_pred, model_info = train_and_predict(
        train_df, test_df, avail_features,
    )
    print(f"  Train MAE: {model_info['train_mae']:.4f}")
    print(f"  Train corr: {model_info['train_corr']:.4f}")

    # Add predictions to dataframes
    test_df["market_excess_pred"] = test_pred
    train_df["market_excess_pred"] = train_pred

    # 5. Evaluate
    print("\n[4/6] Evaluating...")

    quantile_roi = evaluate_quantile_roi(test_df, "market_excess_pred")
    print("  Quantile ROI (test 2025, payout-based):")
    for q in quantile_roi:
        print(
            f"    {q['quantile']:>8s}: N={q['count']:5d} "
            f"ROI(payout)={q['roi_payout']:.1%} "
            f"ROI(tanodds)={q['roi_tanodds']:.1%} "
            f"avg_CO={q['avg_confirmed_odds']:.1f}"
        )

    correlations = compute_correlations(test_df, "market_excess_pred")
    print("\n  Correlations (test 2025):")
    for pair, val in correlations.items():
        print(f"    {pair}: {val:.4f}")

    print("\n  Comparing baselines (test 2025)...")
    baselines = compare_baselines(test_df, "market_excess_pred")
    for b in baselines:
        t5 = b.get("top_5pct", {})
        print(
            f"    {b['name']:>30s}: top5% N={t5.get('count', 0):4d} "
            f"ROI(payout)={t5.get('roi_payout', 0):.1%}"
        )

    surface_breakdown = evaluate_by_surface(test_df, "market_excess_pred")

    # 6. Sanity check & verdict
    print("\n[5/6] Running sanity checks...")
    sanity = sanity_check(quantile_roi, correlations, baselines)
    for name, check in sanity["checks"].items():
        status = "PASS" if check["pass"] else "FAIL"
        print(f"  {status}: {name} - {check['detail']}")

    verdict = determine_verdict(sanity, quantile_roi, test_df, "market_excess_pred")
    print(f"\n  VERDICT: {verdict}")

    # 7. Save results
    print("\n[6/6] Saving results...")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "meta": {
            "script": "analyze_win_market_excess_oof.py",
            "timestamp": datetime.now().isoformat(),
            "description": (
                "Market excess meta-model lightweight OOS experiment. "
                "Target: is_win - p_market_win_norm. "
                "NOT strict production OOF; lightweight 2024→2025 out-of-sample."
            ),
        },
        "model_info": model_info,
        "feature_importance": model_info.get("feature_importance", {}),
        "quantile_roi": quantile_roi,
        "correlations": correlations,
        "baselines": baselines,
        "surface_breakdown": surface_breakdown,
        "sanity": sanity,
        "verdict": verdict,
    }

    json_path = ANALYSIS_DIR / "win_market_excess_oof.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {json_path}")

    report = generate_report(results)
    md_path = ANALYSIS_DIR / "win_market_excess_oof.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {md_path}")

    print("\n" + "=" * 60)
    print(f"DONE. Verdict: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
