"""MawcConservativeReportGenerator -- HTML report for conservative MAWC retrain.

Extracted from mawc_conservative_retrainer.py to reduce coupling.
The retraining engine and presentation layer are independent concerns.

Uses Jinja2 FileSystemLoader with template at src/models/templates/.
Imports dataclasses for type hinting only (no logic dependency).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class MawcConservativeReportGenerator:
    """HTML report generator for MAWC conservative retrain results.

    Generates a self-contained HTML report with 5 sections:
    1. Configuration
    2. Per-Surface Results
    3. Quality Gate Comparison
    4. Favorite Band Guard (Odds 1-3)
    5. C Grid Candidates

    Args:
        output_dir: Directory where the HTML report will be written.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir = Path(__file__).parent / "templates"

    def generate(
        self,
        manifest: dict,
        retrain_results: list[Any],
    ) -> Path:
        """Generate HTML report and write to output_dir.

        Args:
            manifest: Manifest dict from MawcConservativeRetrainer.generate_manifest().
            retrain_results: List of ConservativeRetrainResult objects.

        Returns:
            Path to the generated HTML file.
        """
        env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
        )
        template = env.get_template("mawc_conservative_report.html")

        # Build context for template
        context: dict[str, Any] = {
            "manifest": manifest,
            "retrain_results": retrain_results,
            "per_surface": manifest.get("per_year_surface", {}),
        }

        html = template.render(**context)

        outpath = self.output_dir / "mawc_conservative_report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath
