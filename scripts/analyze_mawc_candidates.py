"""analyze_mawc_candidates.py -- OOF-only lightweight MAWC candidate comparison.

2x2 factorial design to separate C-selection vs interaction-removal effects:
  A: 51-dim + baseline C selection  (logloss-based, reference)
  B: 42-dim + baseline C selection  (selective interactions)
  C: 51-dim + ECE-weighted C selection
  D: 42-dim + ECE-weighted C selection

Constraints:
  - Uses only existing OOF data, joblib models, and parquet files.
  - No run_train, run_backtest, run_wf_validation.
  - No model saving.
  - Output: data/analysis/mawc_candidate_comparison.{json,md}

Usage:
  python scripts/analyze_mawc_candidates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.market_aware_win_calibrator import MarketAwareWinCalibrator  # noqa: E402
from utils.wf_splits import walk_forward_race_splits  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 2x2 factorial candidate definitions
CANDIDATES = {
    "A": {"dim": 51, "c_selection": "logloss",      "label": "51-dim + logloss C"},
    "B": {"dim": 42, "c_selection": "logloss",      "label": "42-dim + logloss C"},
    "C": {"dim": 51, "c_selection": "ece_weighted",  "label": "51-dim + ECE-weighted C"},
    "D": {"dim": 42, "c_selection": "ece_weighted",  "label": "42-dim + ECE-weighted C"},
}

BASELINE_C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
EXTENDED_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]

# 6 stable/significant logit_model_x_* interactions to KEEP in 42-dim
KEPT_MODEL_INTERACTIONS = [
    "logit_model_x_100+",
    "logit_model_x_top_25",
    "logit_model_x_pop_1",
    "logit_model_x_pop_4_6",
    "logit_model_x_2-3",
    "logit_model_x_10-30",
]

# 9 noisy/unstable logit_model_x_* interactions to REMOVE in 42-dim
REMOVED_MODEL_INTERACTIONS = [
    "logit_model_x_1-2",
    "logit_model_x_3-5",
    "logit_model_x_5-10",
    "logit_model_x_30-100",
    "logit_model_x_pop_2_3",
    "logit_model_x_pop_7_9",
    "logit_model_x_pop_10_plus",
    "logit_model_x_mid_25_75",
    "logit_model_x_bottom_25",
]


# ---------------------------------------------------------------------------
# ECE computation (mirrors ShadowComparisonFramework)
# ---------------------------------------------------------------------------

def _compute_ece(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width binning."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_pred)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        mask = (
            (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
            if i == n_bins - 1
            else (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        )
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        ece += abs(y_pred[mask].mean() - y_true[mask].mean()) * (n_in_bin / total)
    return float(ece)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_oof_data(oof_path: Path) -> pd.DataFrame:
    """Load OOF data and derive required columns.

    Uses win_selection_oof.parquet, which is the lightweight persisted OOF artifact
    closest to the MAWC/RaceLevelRanker training frame. p_win_oof is the fold-safe
    model probability; p_win_pred is identical in this artifact.
    """
    df = pd.read_parquet(oof_path)
    if "p_win_oof" not in df.columns:
        raise ValueError(f"{oof_path} must contain p_win_oof")

    df["p_model"] = df["p_win_oof"]
    df["p_market"] = np.clip(1.0 / df["tanodds"].values, 0.01, 0.99)

    if "field_size" not in df.columns:
        df["field_size"] = df.groupby("race_id", observed=True)["race_id"].transform("size")

    if "popularity_rank" not in df.columns:
        df["popularity_rank"] = (
            df.groupby("race_id", observed=True)["tanodds"]
            .rank(method="min", ascending=True)
            .astype(float)
        )

    df["p_win_race_rank_pct"] = (
        df.groupby("race_id", observed=True)["p_model"]
        .rank(pct=True, method="min", ascending=False)
    )
    df["popularity_rank_pct"] = (
        df["popularity_rank"].astype(float)
        / df["field_size"].astype(float).clip(lower=1)
    ).clip(0, 1)
    required = [
        "p_model", "p_market", "tanodds", "popularity_rank",
        "field_size", "kakuteijyuni", "race_id", "surface",
    ]
    df = df.dropna(subset=required).copy()
    return df


# ---------------------------------------------------------------------------
# Feature matrix builders
# ---------------------------------------------------------------------------

def build_51dim(
    df: pd.DataFrame, helper: MarketAwareWinCalibrator,
) -> tuple[np.ndarray, list[str]]:
    """Full 51-dim feature matrix (baseline)."""
    return helper.build_feature_matrix(df)


def build_42dim(
    df: pd.DataFrame, helper: MarketAwareWinCalibrator,
) -> tuple[np.ndarray, list[str]]:
    """42-dim feature matrix (9 noisy logit_model_x_* removed).

    6 main + 15 segment + 6 kept model interactions + 15 market interactions = 42.
    """
    x_51, names_51 = helper.build_feature_matrix(df)
    kept_indices: list[int] = []
    kept_names: list[str] = []
    for i, name in enumerate(names_51):
        # Skip removed model interactions
        if name.startswith("logit_model_x_") and name not in KEPT_MODEL_INTERACTIONS:
            continue
        kept_indices.append(i)
        kept_names.append(name)
    return x_51[:, kept_indices], kept_names


# ---------------------------------------------------------------------------
# C-selection strategies
# ---------------------------------------------------------------------------

def select_c_logloss(c_grid_results: dict[float, dict]) -> float:
    """Baseline C selection: lowest mean_logloss, tie-break smaller C."""
    best_c: float | None = None
    best_logloss = float("inf")
    for c, m in c_grid_results.items():
        if m["mean_logloss"] < best_logloss or (
            np.isclose(m["mean_logloss"], best_logloss) and c < (best_c or float("inf"))
        ):
            best_logloss = m["mean_logloss"]
            best_c = c
    return best_c  # type: ignore[return-value]


def select_c_ece_weighted(c_grid_results: dict[float, dict]) -> float:
    """ECE-weighted C selection: composite rank (0.5*logloss + 0.5*ECE)."""
    c_values = list(c_grid_results.keys())
    loglosses = [c_grid_results[c]["mean_logloss"] for c in c_values]
    eces = [c_grid_results[c]["mean_ece"] for c in c_values]
    logloss_ranks = pd.Series(loglosses).rank().values
    ece_ranks = pd.Series(eces).rank().values
    composite = 0.5 * logloss_ranks + 0.5 * ece_ranks
    best_idx = int(np.argmin(composite))
    # Tie-break: smaller C
    best_val = composite[best_idx]
    ties = [i for i in range(len(c_values)) if np.isclose(composite[i], best_val)]
    if len(ties) > 1:
        return min(c_values[i] for i in ties)
    return c_values[best_idx]


# ---------------------------------------------------------------------------
# Segment metrics
# ---------------------------------------------------------------------------

def compute_segment_metrics(
    df: pd.DataFrame, y: np.ndarray, p_pred: np.ndarray,
) -> dict[str, dict]:
    """ECE and APR per key segment."""
    segments = {
        "pop_1": df["popularity_rank"].values == 1,
        "pop_4_6": (df["popularity_rank"].values >= 4) & (df["popularity_rank"].values <= 6),
        "odds_1-3": (df["tanodds"].values >= 1.0) & (df["tanodds"].values < 3.0),
        "odds_10-30": (df["tanodds"].values >= 10.0) & (df["tanodds"].values < 30.0),
    }
    result: dict[str, dict] = {}
    for name, mask in segments.items():
        n = int(mask.sum())
        if n > 0:
            result[name] = {
                "n": n,
                "ece": _compute_ece(p_pred[mask], y[mask]),
                "apr": float(y[mask].mean()) / max(float(p_pred[mask].mean()), 1e-10),
            }
    return result


# ---------------------------------------------------------------------------
# Single candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate(
    df: pd.DataFrame,
    surface: str,
    cand_key: str,
    helper: MarketAwareWinCalibrator,
) -> dict:
    """Evaluate one candidate on one surface using WF 5-fold."""
    config = CANDIDATES[cand_key]
    dim = config["dim"]
    c_selection = config["c_selection"]

    # Build feature matrix
    if dim == 51:
        x_matrix, feature_names = build_51dim(df, helper)
    else:
        x_matrix, feature_names = build_42dim(df, helper)

    y = (df["kakuteijyuni"] == 1).astype(int).values

    # C grid
    c_grid = EXTENDED_C_GRID if c_selection == "ece_weighted" else BASELINE_C_GRID

    # WF 5-fold C selection
    splits = walk_forward_race_splits(df, n_splits=5)
    c_grid_results: dict[float, dict] = {}
    for c in c_grid:
        fold_loglosses: list[float] = []
        fold_briers: list[float] = []
        fold_eces: list[float] = []
        for train_idx, val_idx in splits:
            lr = LogisticRegression(C=c, max_iter=1000, fit_intercept=True)
            lr.fit(x_matrix[train_idx], y[train_idx])
            p_val = lr.predict_proba(x_matrix[val_idx])[:, 1]
            fold_loglosses.append(float(log_loss(y[val_idx], p_val)))
            fold_briers.append(float(brier_score_loss(y[val_idx], p_val)))
            fold_eces.append(_compute_ece(p_val, y[val_idx]))

        c_grid_results[c] = {
            "mean_logloss": float(np.mean(fold_loglosses)),
            "mean_brier": float(np.mean(fold_briers)),
            "mean_ece": float(np.mean(fold_eces)),
        }

    # Select best C
    best_c = (
        select_c_logloss(c_grid_results)
        if c_selection == "logloss"
        else select_c_ece_weighted(c_grid_results)
    )

    # Fit final model on all data
    lr_final = LogisticRegression(C=best_c, max_iter=1000, fit_intercept=True)
    lr_final.fit(x_matrix, y)
    p_pred = lr_final.predict_proba(x_matrix)[:, 1]

    # OOS metrics (from WF fold averages)
    oos = c_grid_results[best_c]

    # In-sample metrics (full data)
    is_brier = float(brier_score_loss(y, p_pred))
    is_logloss = float(log_loss(y, p_pred))
    is_ece = _compute_ece(p_pred, y)

    # APR and pred_sum
    apr = float(y.mean()) / max(float(p_pred.mean()), 1e-10)
    pred_sum = float(p_pred.sum())

    # Coefficients
    coef = lr_final.coef_[0]
    idx_model = feature_names.index("logit_model")
    idx_market = feature_names.index("logit_market")
    coef_model = float(coef[idx_model])
    coef_market = float(coef[idx_market])
    abs_m, abs_mk = abs(coef_model), abs(coef_market)
    beta_market = abs_mk / (abs_m + abs_mk) if (abs_m + abs_mk) > 1e-10 else 0.0

    # Segment metrics
    segment_metrics = compute_segment_metrics(df, y, p_pred)

    # Top 10 coefficients by absolute value
    abs_coef = np.abs(coef)
    top10_idx = np.argsort(abs_coef)[::-1][:10]
    top10_coefficients = [
        {"feature": feature_names[i], "coef": float(coef[i]), "abs_coef": float(abs_coef[i])}
        for i in top10_idx
    ]

    return {
        "candidate": cand_key,
        "label": config["label"],
        "surface": surface,
        "n_rows": int(len(df)),
        "dim": dim,
        "c_selection": c_selection,
        "best_c": best_c,
        "oos_brier": oos["mean_brier"],
        "oos_logloss": oos["mean_logloss"],
        "oos_ece": oos["mean_ece"],
        "is_brier": is_brier,
        "is_logloss": is_logloss,
        "is_ece": is_ece,
        "apr": apr,
        "pred_sum": pred_sum,
        "coef_model": coef_model,
        "coef_market": coef_market,
        "beta_market": beta_market,
        "segment_metrics": segment_metrics,
        "top10_coefficients": top10_coefficients,
        "c_grid_results": {
            str(c): {k: v for k, v in r.items()} for c, r in c_grid_results.items()
        },
        "feature_names": feature_names,
        "n_features": len(feature_names),
    }


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------

def generate_markdown(results: list[dict], baseline_metadata: dict) -> str:
    """Generate comparison Markdown report."""
    lines: list[str] = [
        "# MAWC Candidate Comparison (OOF Analysis)",
        "",
        "## Important: OOF Source",
        "",
        "This analysis uses **`data/oof/win_selection_oof.parquet`** and sets",
        "`p_model = p_win_oof`. That artifact is the persisted OOF frame closest",
        "to the MAWC/RaceLevelRanker training path. When `popularity_rank` and",
        "`field_size` are absent, they are reconstructed from odds rank and race size",
        "for analysis only.",
        "",
        "## 2x2 Factorial Design",
        "",
        "|  | Baseline C Selection (logloss) | ECE-weighted C Selection |",
        "|--|---|---|",
        "| **51-dim (all interactions)** | **A** (baseline reference) | **C** |",
        "| **42-dim (selective interactions)** | **B** | **D** |",
        "",
        "### C Grid",
        "",
        "- Baseline C selection: `[0.03, 0.1, 0.3, 1.0, 3.0]` → pick lowest logloss",
        "- ECE-weighted C selection: `[0.01, 0.03, 0.1, 0.3, 1.0]`",
        "  → composite rank (0.5*logloss + 0.5*ECE)",
        "",
        "### Removed Interactions (42-dim)",
        "",
        "Kept 6 stable/significant `logit_model_x_*`:",
    ]
    for name in KEPT_MODEL_INTERACTIONS:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("Removed 9 noisy/unstable `logit_model_x_*`:")
    for name in REMOVED_MODEL_INTERACTIONS:
        lines.append(f"- `{name}`")
    lines.append("")

    # Baseline model reference
    lines.append("## Baseline Model Reference (from joblib)")
    lines.append("")
    lines.append("| Surface | Year | C | coef_model | coef_market | beta_market |")
    lines.append("|---------|------|---|------------|-------------|-------------|")
    for key, meta in baseline_metadata.items():
        lines.append(
            f"| {meta['surface']} | {meta['year']} | {meta['best_c']} | "
            f"{meta['coef_model']:+.4f} | {meta['coef_market']:+.4f} | "
            f"{meta['beta_market']:.4f} |"
        )
    lines.append("")

    # Main comparison tables per surface
    for surface in ["turf", "dirt"]:
        surf_results = [r for r in results if r["surface"] == surface]
        n_rows = surf_results[0]["n_rows"] if surf_results else 0
        lines.append(f"## Surface: {surface} ({n_rows} rows)")
        lines.append("")

        # Main metrics table
        lines.append("### OOS Metrics (WF 5-fold average)")
        lines.append("")
        lines.append(
            "| Cand | Dim | C Sel | Best C | OOS Brier | OOS Logloss | OOS ECE "
            "| APR | beta_mkt | coef_model | coef_market | Reject? |"
        )
        lines.append(
            "|------|-----|-------|--------|-----------|-------------|--------"
            "|-----|----------|------------|-------------|---------|"
        )
        for r in surf_results:
            reject = ""
            if r["coef_market"] < 0:
                reject = "**REJECT** (market<0)"
            elif r["beta_market"] < 0.20:
                reject = f"**REJECT** (beta={r['beta_market']:.3f}<0.20)"
            lines.append(
                f"| **{r['candidate']}** | {r['dim']} | {r['c_selection']} | "
                f"{r['best_c']:.4f} | {r['oos_brier']:.6f} | {r['oos_logloss']:.6f} | "
                f"{r['oos_ece']:.6f} | {r['apr']:.4f} | {r['beta_market']:.4f} | "
                f"{r['coef_model']:+.4f} | {r['coef_market']:+.4f} | {reject} |"
            )
        lines.append("")

        # Segment metrics
        lines.append(f"### Segment Metrics ({surface})")
        lines.append("")
        lines.append(
            "| Cand | pop_1 ECE | pop_4_6 ECE | odds_1-3 ECE | odds_10-30 ECE "
            "| pred_sum |"
        )
        lines.append(
            "|------|-----------|-------------|--------------|----------------"
            "|----------|"
        )
        for r in surf_results:
            seg = r["segment_metrics"]
            pop1 = seg.get("pop_1", {}).get("ece", float("nan"))
            pop46 = seg.get("pop_4_6", {}).get("ece", float("nan"))
            o13 = seg.get("odds_1-3", {}).get("ece", float("nan"))
            o1030 = seg.get("odds_10-30", {}).get("ece", float("nan"))
            lines.append(
                f"| **{r['candidate']}** | {pop1:.4f} | {pop46:.4f} | "
                f"{o13:.4f} | {o1030:.4f} | {r['pred_sum']:.1f} |"
            )
        lines.append("")

        # C grid details
        for r in surf_results:
            lines.append(
                f"#### C Grid Detail: Candidate {r['candidate']} "
                f"({r['label']}, {surface})"
            )
            lines.append("")
            lines.append("| C | OOS Logloss | OOS Brier | OOS ECE |")
            lines.append("|---|-------------|-----------|---------|")
            for c_str, m in sorted(r["c_grid_results"].items(), key=lambda x: float(x[0])):
                marker = " **← best**" if float(c_str) == r["best_c"] else ""
                lines.append(
                    f"| {float(c_str):.4f} | {m['mean_logloss']:.6f} | "
                    f"{m['mean_brier']:.6f} | {m['mean_ece']:.6f} |{marker}"
                )
            lines.append("")

        # Top 10 coefficients
        for r in surf_results:
            lines.append(f"#### Top 10 Coefficients: Candidate {r['candidate']} ({surface})")
            lines.append("")
            lines.append("| Feature | Coefficient | |Coef| |")
            lines.append("|---------|-------------|-------|")
            for entry in r["top10_coefficients"]:
                lines.append(
                    f"| `{entry['feature']}` | {entry['coef']:+.6f} | "
                    f"{entry['abs_coef']:.6f} |"
                )
            lines.append("")

    # Verdict section
    lines.append("## Adoption Verdict")
    lines.append("")
    lines.append("### Instant Rejection Criteria")
    lines.append("")
    lines.append("- `logit_market < 0`: market odds used inversely (failure pattern from v2.2)")
    lines.append("- `beta_market < 0.20`: market contribution too low")
    lines.append("")
    lines.append("### Comparison Criteria (Candidate A as baseline)")
    lines.append("")
    lines.append("- OOS ECE must not degrade by > 10% relative")
    lines.append("- APR must not degrade by > 5% relative")
    lines.append("- pred_sum must not increase by > 10% relative")
    lines.append("- `logit_market` must remain positive")
    lines.append("")

    for surface in ["turf", "dirt"]:
        surf_map = {r["candidate"]: r for r in results if r["surface"] == surface}
        baseline = surf_map["A"]
        lines.append(f"### {surface}")
        lines.append("")
        for key in ["B", "C", "D"]:
            r = surf_map[key]
            if r["coef_market"] < 0:
                verdict = f"**INSTANT REJECT**: logit_market = {r['coef_market']:+.4f} < 0"
            elif r["beta_market"] < 0.20:
                verdict = f"**INSTANT REJECT**: beta_market = {r['beta_market']:.4f} < 0.20"
            else:
                ece_ratio = r["oos_ece"] / max(baseline["oos_ece"], 1e-10)
                apr_ratio = r["apr"] / max(baseline["apr"], 1e-10)
                pred_ratio = r["pred_sum"] / max(baseline["pred_sum"], 1e-10)
                issues: list[str] = []
                if ece_ratio > 1.10:
                    issues.append(f"ECE {ece_ratio:.2f}x baseline")
                if apr_ratio < 0.95:
                    issues.append(f"APR {apr_ratio:.2f}x baseline")
                if pred_ratio > 1.10:
                    issues.append(f"pred_sum {pred_ratio:.2f}x baseline")
                if issues:
                    verdict = "FLAGGED: " + ", ".join(issues)
                else:
                    verdict = "PASSES all adoption criteria"
            lines.append(f"- **Candidate {key}** ({r['label']}): {verdict}")
        lines.append("")

    # Factorial decomposition
    lines.append("## Factorial Decomposition")
    lines.append("")
    lines.append("Comparing candidates to isolate the effect of each factor:")
    lines.append("")

    for surface in ["turf", "dirt"]:
        surf_map = {r["candidate"]: r for r in results if r["surface"] == surface}
        a = surf_map["A"]
        b = surf_map["B"]
        c = surf_map["C"]
        d = surf_map["D"]
        lines.append(f"### {surface}")
        lines.append("")
        lines.append("| Comparison | What changed | OOS ECE delta | OOS Brier delta | APR delta |")
        lines.append("|------------|-------------|---------------|-----------------|-----------|")
        comparisons = [
            ("A → B", "Remove 9 interactions", a, b),
            ("A → C", "Change C selection", a, c),
            ("B → D", "Change C selection (42-dim)", b, d),
            ("C → D", "Remove 9 interactions (ECE C)", c, d),
        ]
        for label, desc, before, after in comparisons:
            ece_d = after["oos_ece"] - before["oos_ece"]
            brier_d = after["oos_brier"] - before["oos_brier"]
            apr_d = after["apr"] - before["apr"]
            lines.append(
                f"| {label} | {desc} | {ece_d:+.6f} | {brier_d:+.6f} | {apr_d:+.4f} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("*Generated by `scripts/analyze_mawc_candidates.py`*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    oof_path = PROJECT_ROOT / "data" / "oof" / "win_selection_oof.parquet"
    output_dir = PROJECT_ROOT / "data" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load OOF data
    print("Loading OOF data...")
    df_all = prepare_oof_data(oof_path)
    print(f"  Total: {len(df_all)} rows")

    # Load baseline model metadata for reference
    baseline_metadata: dict[str, dict] = {}
    helper = MarketAwareWinCalibrator()
    for year in [2024, 2025]:
        for surface in ["turf", "dirt"]:
            p = (
                PROJECT_ROOT
                / "data"
                / "models-backtest"
                / str(year)
                / f"market_aware_win_calibrator_{surface}.joblib"
            )
            if p.is_file():
                import joblib
                state = joblib.load(p)
                fn = state.get("feature_names", [])
                coef = state.get("calibrator").coef_[0]
                idx_m = fn.index("logit_model")
                idx_mk = fn.index("logit_market")
                abs_m = abs(float(coef[idx_m]))
                abs_mk = abs(float(coef[idx_mk]))
                baseline_metadata[f"{year}_{surface}"] = {
                    "year": year,
                    "surface": surface,
                    "best_c": state.get("best_c"),
                    "coef_model": float(coef[idx_m]),
                    "coef_market": float(coef[idx_mk]),
                    "beta_market": abs_mk / (abs_m + abs_mk) if (abs_m + abs_mk) > 0 else 0,
                }

    # Evaluate all candidates
    all_results: list[dict] = []
    for surface in ["turf", "dirt"]:
        # reset_index(drop=True) is CRITICAL: walk_forward_race_splits returns
        # df.index values, which must be contiguous 0..N-1 to match numpy arrays.
        df_surf = df_all[df_all["surface"] == surface].copy().reset_index(drop=True)
        print(f"\n{'='*60}")
        print(f"Surface: {surface} ({len(df_surf)} rows)")
        print(f"{'='*60}")

        for cand_key in ["A", "B", "C", "D"]:
            config = CANDIDATES[cand_key]
            print(f"\n  Candidate {cand_key}: {config['label']}")
            result = evaluate_candidate(df_surf, surface, cand_key, helper)
            all_results.append(result)

            # Instant rejection check
            if result["coef_market"] < 0:
                print(f"    INSTANT REJECT: logit_market = {result['coef_market']:+.4f} < 0")
            elif result["beta_market"] < 0.20:
                print(f"    INSTANT REJECT: beta_market = {result['beta_market']:.4f} < 0.20")
            else:
                print(f"    OK: C={result['best_c']:.4f}  OOS_ECE={result['oos_ece']:.6f}  "
                      f"OOS_Brier={result['oos_brier']:.6f}  APR={result['apr']:.4f}  "
                      f"beta_mkt={result['beta_market']:.4f}")
                print(f"         coef_model={result['coef_model']:+.4f}  "
                      f"coef_market={result['coef_market']:+.4f}  "
                      f"pred_sum={result['pred_sum']:.1f}")

    # Save JSON
    json_path = output_dir / "mawc_candidate_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"baseline_metadata": baseline_metadata, "results": all_results},
            f, indent=2, ensure_ascii=False, default=str,
        )
    print(f"\nJSON saved: {json_path}")

    # Save Markdown
    md_text = generate_markdown(all_results, baseline_metadata)
    md_path = output_dir / "mawc_candidate_comparison.md"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Markdown saved: {md_path}")

    # Quick summary
    print(f"\n{'='*60}")
    print("QUICK SUMMARY")
    print(f"{'='*60}")
    for surface in ["turf", "dirt"]:
        print(f"\n  {surface.upper()}:")
        for r in all_results:
            if r["surface"] != surface:
                continue
            reject = ""
            if r["coef_market"] < 0:
                reject = " [REJECT: market<0]"
            elif r["beta_market"] < 0.20:
                reject = f" [REJECT: beta={r['beta_market']:.3f}]"
            print(f"    {r['candidate']}: ECE={r['oos_ece']:.6f} Brier={r['oos_brier']:.6f} "
                  f"APR={r['apr']:.4f} C={r['best_c']:.4f}{reject}")


if __name__ == "__main__":
    main()
