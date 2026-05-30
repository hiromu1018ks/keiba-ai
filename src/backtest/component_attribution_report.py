"""ComponentAttributionReportGenerator -- HTML report for component attribution.

Extracted from component_attribution.py to reduce coupling.
The analysis engine (component_attribution.py) and presentation layer
(this module) are independent concerns.

Uses Jinja2 FileSystemLoader with template at src/backtest/templates/.
Imports ComponentAttributionResult for type hinting only (no logic dependency).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from backtest.component_attribution import ComponentAttributionResult
from backtest.historical_bisect import HistoricalBisectResult

logger = logging.getLogger(__name__)


class ComponentAttributionReportGenerator:
    """HTML report generator for component attribution results.

    Generates a self-contained HTML report with 4 attribution sections
    plus a recommendations footer.

    Args:
        output_dir: Directory where the HTML report will be written.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir = Path(__file__).parent / "templates"

    def generate(
        self,
        attribution_result: ComponentAttributionResult,
        historical_result: HistoricalBisectResult | None = None,
    ) -> Path:
        """Generate HTML report and write to output_dir.

        Args:
            attribution_result: ComponentAttribution.run_full_attribution() result.
            historical_result: Optional HistoricalBisect result for context.

        Returns:
            Path to the generated HTML file.
        """
        env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
        )
        template = env.get_template("component_attribution_report.html")

        # Build context for template
        context: dict[str, Any] = {
            "ece_attribution": attribution_result.ece_attribution,
            "apr_attribution": attribution_result.apr_attribution,
            "bet_count_attribution": attribution_result.bet_count_attribution,
            "coefficient_analysis": attribution_result.coefficient_analysis,
            "upstream_anomaly_check": attribution_result.upstream_anomaly_check,
            "recommendations": attribution_result.recommendations,
            "historical_result": historical_result,
        }

        html = template.render(**context)

        outpath = self.output_dir / "component_attribution_report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath
