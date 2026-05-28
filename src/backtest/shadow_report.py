"""Shadow Comparison Report Generator (D-16, D-17).

ShadowComparisonFramework の結果から自己完結型HTMLレポートを生成する。
JSON/Parquet が信頼できる情報源。HTML は人間によるレビュー用 (D-17)。
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from backtest.shadow_comparison import (
    ShadowComparisonResult,
    VariantConfig,
)

logger = logging.getLogger(__name__)


class ShadowComparisonReportGenerator:
    """Shadow Comparison Report Generator (D-16).

    BacktestReportGenerator パターンに従い、Jinja2 + HTML で
    side-by-side 比較レポートを生成する。
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir = Path(__file__).parent / "templates"

    def generate(
        self,
        comparison_results: list[ShadowComparisonResult],
        variant_configs: list[VariantConfig],
        metrics_json: dict[str, Any],
    ) -> Path:
        """HTMLレポートを生成し、ファイルパスを返す."""
        env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
        )
        template = env.get_template("shadow_comparison_report.html")

        # --- Build context ---
        variants_info: list[dict[str, Any]] = []
        for vc in variant_configs:
            variants_info.append({
                "variant_name": vc.variant_name,
                "model_dir": str(vc.model_dir),
                "flag_states": {
                    "enable_market_aware_calibrator": vc.enable_market_aware_calibrator,
                    "enable_race_level_ranker": vc.enable_race_level_ranker,
                },
                "is_baseline": vc.variant_name == "baseline",
            })

        # Overall summary from metrics_json
        overall_section = metrics_json.get("overall", {})
        overall_metrics = overall_section.get("metrics", {})

        # Per-fold sections
        folds_section = metrics_json.get("folds", {})

        # Selection agreement section
        selection_examples: list[dict[str, Any]] = []
        total_races = 0
        changed_races = 0
        overall_agreement: float | None = None

        for cr in comparison_results:
            if not cr.race_diff.empty and "selected_changed" in cr.race_diff.columns:
                total_races += len(cr.race_diff)
                changed_races += int(cr.race_diff["selected_changed"].sum())

                # Top selection-change examples (max 20)
                changed_df = cr.race_diff[cr.race_diff["selected_changed"]]
                for _, row in changed_df.head(20).iterrows():
                    example: dict[str, Any] = {
                        "race_id": str(row.get("race_id", "")),
                        "baseline_umaban": row.get("baseline_selected_umaban", ""),
                        "shadow_umaban": row.get("shadow_selected_umaban", ""),
                        "baseline_odds": row.get("baseline_tanodds", ""),
                        "shadow_odds": row.get("shadow_tanodds", ""),
                        "baseline_result": row.get("baseline_result", 0),
                        "shadow_result": row.get("shadow_result", 0),
                        "fold_year": cr.fold.year,
                    }
                    selection_examples.append(example)

        if total_races > 0:
            overall_agreement = 1.0 - changed_races / total_races

        # Calibration section
        calibration_data: dict[str, dict[str, Any]] = {}
        for vname, vmetrics in overall_metrics.items():
            calibration_data[vname] = {
                "brier": vmetrics.get("brier", 0),
                "logloss": vmetrics.get("logloss", 0),
                "ece": vmetrics.get("ece", 0),
            }

        # Get git commit hash
        try:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit_hash = "unknown"

        from datetime import datetime, timezone

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        html = template.render(
            variants_info=variants_info,
            overall_metrics=overall_metrics,
            folds_section=folds_section,
            selection_agreement=overall_agreement,
            total_races=total_races,
            changed_races=changed_races,
            selection_examples=selection_examples,
            calibration_data=calibration_data,
            generated_at=generated_at,
            commit_hash=commit_hash,
        )

        outpath = self.output_dir / "shadow_comparison_report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath
