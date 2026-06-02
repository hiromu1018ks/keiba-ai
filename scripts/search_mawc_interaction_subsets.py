"""Search lightweight MAWC selective-interaction subsets.

This script is a pre-shadow diagnostic. It tries all subsets of the six
candidate `logit_model_x_*` interactions identified by v2.3 analysis, using
the baseline C=0.03, and compares each subset against the saved baseline MAWC
on the persisted win-selection OOF artifact.

It does not run training pipelines, backtests, or shadow comparison.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.market_aware_win_calibrator import MarketAwareWinCalibrator  # noqa: E402


SEARCH_INTERACTIONS: list[str] = [
    "logit_model_x_100+",
    "logit_model_x_top_25",
    "logit_model_x_pop_1",
    "logit_model_x_pop_4_6",
    "logit_model_x_2-3",
    "logit_model_x_10-30",
]

KEY_SEGMENTS: dict[str, Any] = {
    "pop_1": lambda df: df["popularity_rank"] == 1,
    "pop_4_6": lambda df: (df["popularity_rank"] >= 4) & (df["popularity_rank"] <= 6),
    "odds_1_3": lambda df: (df["tanodds"] >= 1.0) & (df["tanodds"] < 3.0),
    "odds_10_30": lambda df: (df["tanodds"] >= 10.0) & (df["tanodds"] < 30.0),
}


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_pred)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= edges[i]) & (y_pred <= edges[i + 1])
        else:
            mask = (y_pred >= edges[i]) & (y_pred < edges[i + 1])
        if not mask.any():
            continue
        ece += abs(float(y_pred[mask].mean()) - float(y_true[mask].mean())) * mask.mean()
    return float(ece)


def prepare_oof(oof_path: Path) -> pd.DataFrame:
    """Load and augment persisted win-selection OOF."""
    df = pd.read_parquet(oof_path).copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["field_size"] = df.groupby("race_id", observed=True)["race_id"].transform("size")
    df["popularity_rank"] = (
        df.groupby("race_id", observed=True)["tanodds"]
        .rank(method="min", ascending=True)
        .astype(float)
    )
    required = [
        "race_id",
        "umaban",
        "race_date",
        "surface",
        "tanodds",
        "p_win_oof",
        "p_win_corrected",
        "kakuteijyuni",
        "field_size",
        "popularity_rank",
    ]
    return df.dropna(subset=required).sort_values(["race_date", "race_id", "umaban"]).reset_index(
        drop=True
    )


def add_mawc_columns(df: pd.DataFrame, p_model_col: str) -> pd.DataFrame:
    """Create columns expected by MarketAwareWinCalibrator.build_feature_matrix."""
    work = df.copy()
    work["p_model"] = pd.to_numeric(work[p_model_col], errors="coerce")
    work["p_market"] = np.clip(1.0 / pd.to_numeric(work["tanodds"], errors="coerce"), 0.01, 0.99)
    work["p_win_race_rank_pct"] = (
        work.groupby("race_id", observed=True)["p_model"]
        .rank(pct=True, method="min", ascending=False)
    )
    return work


def feature_matrix_for_names(
    df: pd.DataFrame,
    feature_names: list[str],
    helper: MarketAwareWinCalibrator,
) -> np.ndarray:
    """Build the full feature matrix and select requested names."""
    x_full, names_full = helper.build_feature_matrix(df)
    indices = [names_full.index(name) for name in feature_names]
    return x_full[:, indices]


def names_for_subset(
    subset: set[str],
    helper: MarketAwareWinCalibrator,
    df: pd.DataFrame,
) -> list[str]:
    """Return feature names for a selective subset."""
    _, names_full = helper.build_feature_matrix(df.head(1).copy())
    names: list[str] = []
    for name in names_full:
        if name.startswith("logit_model_x_") and name not in subset:
            continue
        names.append(name)
    return names


def normalize_by_race(df: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """Normalize raw probabilities to sum to one per race."""
    temp = pd.DataFrame({"race_id": df["race_id"].values, "p_raw": raw})
    sums = temp.groupby("race_id", observed=True)["p_raw"].transform("sum").clip(lower=1e-12)
    return (temp["p_raw"] / sums).to_numpy()


def metric_block(df: pd.DataFrame, p: np.ndarray) -> dict[str, float]:
    """Overall metric block."""
    y = (df["kakuteijyuni"] == 1).astype(int).to_numpy()
    pred_sum = float(p.sum())
    return {
        "ece": compute_ece(y, p),
        "apr": float(y.sum() / max(pred_sum, 1e-12)),
        "pred_sum": pred_sum,
    }


def segment_blocks(df: pd.DataFrame, p: np.ndarray) -> dict[str, dict[str, float]]:
    """Key segment metrics."""
    result: dict[str, dict[str, float]] = {}
    for name, mask_fn in KEY_SEGMENTS.items():
        mask = mask_fn(df).to_numpy()
        if not mask.any():
            continue
        result[name] = metric_block(df.iloc[np.flatnonzero(mask)], p[mask])
    return result


def apply_baseline(df: pd.DataFrame, model_path: Path) -> np.ndarray:
    """Apply saved baseline MAWC."""
    mawc = MarketAwareWinCalibrator.load(model_path)
    applied = mawc.apply(df)
    return applied["p_win_final"].to_numpy()


def evaluate_subset(
    *,
    df_train: pd.DataFrame,
    df_apply: pd.DataFrame,
    feature_names: list[str],
    helper: MarketAwareWinCalibrator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit one subset on p_win_oof and apply using p_win_corrected."""
    train = add_mawc_columns(df_train, "p_win_oof")
    apply = add_mawc_columns(df_apply, "p_win_corrected")
    x_train = feature_matrix_for_names(train, feature_names, helper)
    y_train = (train["kakuteijyuni"] == 1).astype(int).to_numpy()
    lr = LogisticRegression(C=0.03, max_iter=1000)
    lr.fit(x_train, y_train)

    x_apply = feature_matrix_for_names(apply, feature_names, helper)
    p_raw = lr.predict_proba(x_apply)[:, 1]
    p_final = normalize_by_race(apply, p_raw)

    coef = lr.coef_[0]
    idx_model = feature_names.index("logit_model")
    idx_market = feature_names.index("logit_market")
    abs_model = abs(float(coef[idx_model]))
    abs_market = abs(float(coef[idx_market]))
    beta_market = abs_market / (abs_model + abs_market) if abs_model + abs_market > 0 else 0.0
    return p_final, {
        "coef_logit_model": float(coef[idx_model]),
        "coef_logit_market": float(coef[idx_market]),
        "beta_market": beta_market,
    }


def score_context(
    base: dict[str, float],
    cand: dict[str, float],
    base_segments: dict[str, dict[str, float]],
    cand_segments: dict[str, dict[str, float]],
) -> tuple[float, list[str]]:
    """Score one year/surface context. Lower is better."""
    issues: list[str] = []
    score = 0.0

    ece_delta = cand["ece"] - base["ece"]
    apr_delta = cand["apr"] - base["apr"]
    pred_ratio = cand["pred_sum"] / max(base["pred_sum"], 1e-12)

    score += max(0.0, ece_delta) * 100.0
    score += max(0.0, -apr_delta) * 10.0
    score += abs(pred_ratio - 1.0) * 2.0

    if ece_delta > 0.002:
        issues.append(f"overall_ece_delta={ece_delta:+.4f}")
    if apr_delta < -0.02:
        issues.append(f"overall_apr_delta={apr_delta:+.4f}")
    if abs(pred_ratio - 1.0) > 0.03:
        issues.append(f"pred_ratio={pred_ratio:.3f}")

    for seg_name, base_seg in base_segments.items():
        cand_seg = cand_segments.get(seg_name)
        if cand_seg is None:
            continue
        seg_ece_delta = cand_seg["ece"] - base_seg["ece"]
        seg_apr_delta = cand_seg["apr"] - base_seg["apr"]
        seg_pred_ratio = cand_seg["pred_sum"] / max(base_seg["pred_sum"], 1e-12)
        score += max(0.0, seg_ece_delta) * 25.0
        score += max(0.0, -seg_apr_delta) * 2.0
        score += abs(seg_pred_ratio - 1.0) * 0.5
        if seg_ece_delta > 0.01:
            issues.append(f"{seg_name}_ece_delta={seg_ece_delta:+.4f}")
        if seg_apr_delta < -0.04:
            issues.append(f"{seg_name}_apr_delta={seg_apr_delta:+.4f}")

    return score, issues


def subset_label(subset: set[str]) -> str:
    """Short subset label."""
    if not subset:
        return "none"
    return "+".join(name.replace("logit_model_x_", "") for name in sorted(subset))


def main() -> None:
    """Run subset search."""
    helper = MarketAwareWinCalibrator()
    df_all = prepare_oof(ROOT / "data/oof/win_selection_oof.parquet")

    contexts: list[dict[str, Any]] = []
    for year in [2024, 2025]:
        meta = json.loads((ROOT / "data/models-backtest" / str(year) / "meta.json").read_text())
        train_start = pd.to_datetime(meta["train_start"])
        train_end = pd.to_datetime(meta["train_end"])
        df_year = df_all[(df_all["race_date"] >= train_start) & (df_all["race_date"] <= train_end)]
        for surface in ["turf", "dirt"]:
            df_surface = (
                df_year[df_year["surface"] == surface]
                .copy()
                .sort_values(["race_date", "race_id", "umaban"])
                .reset_index(drop=True)
            )
            base_p = apply_baseline(
                df_surface,
                ROOT
                / "data/models-backtest"
                / str(year)
                / f"market_aware_win_calibrator_{surface}.joblib",
            )
            contexts.append(
                {
                    "year": year,
                    "surface": surface,
                    "df": df_surface,
                    "base": metric_block(df_surface, base_p),
                    "base_segments": segment_blocks(df_surface, base_p),
                }
            )

    subset_results: list[dict[str, Any]] = []
    all_subsets = [
        set(combo)
        for n_items in range(len(SEARCH_INTERACTIONS) + 1)
        for combo in itertools.combinations(SEARCH_INTERACTIONS, n_items)
    ]
    for subset in all_subsets:
        total_score = 0.0
        all_issues: list[str] = []
        per_context: dict[str, Any] = {}
        coef_market_values: list[float] = []
        beta_values: list[float] = []

        for context in contexts:
            df_context = context["df"]
            train = add_mawc_columns(df_context, "p_win_oof")
            feature_names = names_for_subset(subset, helper, train)
            p_cand, coef = evaluate_subset(
                df_train=df_context,
                df_apply=df_context,
                feature_names=feature_names,
                helper=helper,
            )
            cand = metric_block(df_context, p_cand)
            cand_segments = segment_blocks(df_context, p_cand)
            score, issues = score_context(
                context["base"],
                cand,
                context["base_segments"],
                cand_segments,
            )
            key = f"{context['year']}_{context['surface']}"
            per_context[key] = {
                "base": context["base"],
                "candidate": cand,
                "segments": cand_segments,
                "score": score,
                "issues": issues,
                "coef": coef,
            }
            total_score += score
            all_issues.extend(f"{key}:{issue}" for issue in issues)
            coef_market_values.append(coef["coef_logit_market"])
            beta_values.append(coef["beta_market"])

        if any(value < 0 for value in coef_market_values):
            all_issues.append("coef_market_negative")
            total_score += 100.0
        if min(beta_values) < 0.20:
            all_issues.append("beta_market_below_0.20")
            total_score += 100.0

        subset_results.append(
            {
                "label": subset_label(subset),
                "kept_interactions": sorted(subset),
                "n_kept": len(subset),
                "n_features": 36 + len(subset),
                "score": total_score,
                "issue_count": len(all_issues),
                "issues": all_issues,
                "min_coef_market": min(coef_market_values),
                "min_beta_market": min(beta_values),
                "per_context": per_context,
            }
        )

    subset_results.sort(key=lambda item: (item["issue_count"], item["score"], item["n_kept"]))

    output_dir = ROOT / "data/analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mawc_interaction_subset_search.json"
    json_path.write_text(
        json.dumps({"results": subset_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# MAWC Interaction Subset Search",
        "",
        "Search space: all 64 subsets of the six v2.3 candidate model interactions.",
        "C is fixed at baseline-compatible `0.03`.",
        "",
        "## Top 15",
        "",
        "| Rank | Kept | Features | Issues | Score | Min beta | Min coef_market |",
        "|------|------|----------|--------|-------|----------|-----------------|",
    ]
    for idx, result in enumerate(subset_results[:15], start=1):
        kept = ", ".join(name.replace("logit_model_x_", "") for name in result["kept_interactions"])
        if not kept:
            kept = "(none)"
        lines.append(
            f"| {idx} | {kept} | {result['n_features']} | {result['issue_count']} | "
            f"{result['score']:.4f} | {result['min_beta_market']:.4f} | "
            f"{result['min_coef_market']:+.4f} |"
        )

    best = subset_results[0]
    lines.extend(
        [
            "",
            "## Best Details",
            "",
            f"- Kept interactions: {', '.join(best['kept_interactions']) or '(none)'}",
            f"- Feature dim: {best['n_features']}",
            f"- Issue count: {best['issue_count']}",
            f"- Score: {best['score']:.4f}",
            "",
            "| Context | Base ECE | Cand ECE | Base APR | Cand APR | Pred ratio | Issues |",
            "|---------|----------|----------|----------|----------|------------|--------|",
        ]
    )
    for key, detail in best["per_context"].items():
        base = detail["base"]
        cand = detail["candidate"]
        pred_ratio = cand["pred_sum"] / max(base["pred_sum"], 1e-12)
        issues = ", ".join(detail["issues"])
        lines.append(
            f"| {key} | {base['ece']:.6f} | {cand['ece']:.6f} | "
            f"{base['apr']:.4f} | {cand['apr']:.4f} | {pred_ratio:.4f} | {issues} |"
        )

    md_path = output_dir / "mawc_interaction_subset_search.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(md_path)
    print(json_path)
    print("best:", best["label"], "features=", best["n_features"], "issues=", best["issue_count"])


if __name__ == "__main__":
    main()
