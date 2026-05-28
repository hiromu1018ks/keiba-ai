"""Feature Routing Audit CLI script (SAF-01).

Produces JSON + Markdown audit reports verifying that calibrator and ranker
features are not leaking into critical target models.

使い方:
  python scripts/run_feature_routing_audit.py
  python scripts/run_feature_routing_audit.py --output-dir data/audit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows cp932 環境でエンコーディング問題を回避
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from audit.feature_routing_registry import (  # noqa: E402
    REGISTRY_VERSION,
    run_feature_audit,
)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Feature Routing Audit — verify no forbidden feature leakage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/audit"),
        help="Output directory for audit reports (default: data/audit)",
    )
    parser.add_argument(
        "--registry-version",
        type=str,
        default=None,
        help="Expected registry version for verification (optional)",
    )
    return parser


def run_audit(output_dir: Path) -> dict:
    """Run audit, write JSON + Markdown reports, return results dict.

    Returns the results dict from run_feature_audit() with generated_at
    and output paths appended.
    """
    results = run_feature_audit()
    generated_at = datetime.now(timezone.utc).isoformat()

    results["generated_at"] = generated_at

    output_dir.mkdir(parents=True, exist_ok=True)

    # -- JSON report --
    json_path = output_dir / "feature_routing_audit.json"
    json_report = {
        "registry_version": REGISTRY_VERSION,
        "generated_at": generated_at,
        "overall_status": results["overall_status"],
        "models": results["critical_models"] + results["advisory_models"],
    }
    json_path.write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # -- Markdown report --
    md_path = output_dir / "feature_routing_audit.md"
    md_lines = _build_markdown_report(
        registry_version=REGISTRY_VERSION,
        generated_at=generated_at,
        overall_status=results["overall_status"],
        critical_models=results["critical_models"],
        advisory_models=results["advisory_models"],
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    results["json_path"] = str(json_path)
    results["md_path"] = str(md_path)

    return results


def _build_markdown_report(
    *,
    registry_version: str,
    generated_at: str,
    overall_status: str,
    critical_models: list[dict],
    advisory_models: list[dict],
) -> list[str]:
    """Build Markdown report lines."""
    lines: list[str] = [
        "# Feature Routing Audit Report",
        "",
        f"- **Registry Version:** {registry_version}",
        f"- **Generated At:** {generated_at}",
        f"- **Overall Status:** {overall_status}",
        "",
        "## Fail-Fast Models (Critical)",
        "",
        "| Model | Status | Features Checked | Forbidden Intersections |",
        "|-------|--------|-----------------|------------------------|",
    ]

    for m in critical_models:
        intersections = ", ".join(m["forbidden_intersections"]) or "---"
        status_str = m["status"]
        lines.append(
            f"| {m['model_name']} | {status_str} "
            f"| {m['checked_feature_count']} | {intersections} |"
        )

    lines.extend([
        "",
        "## Advisory Models",
        "",
        "| Model | Status | Features Checked | Warning Intersections |",
        "|-------|--------|-----------------|----------------------|",
    ])

    for m in advisory_models:
        intersections = ", ".join(m["warning_intersections"]) or "---"
        status_str = m["status"]
        lines.append(
            f"| {m['model_name']} | {status_str} "
            f"| {m['checked_feature_count']} | {intersections} |"
        )

    # Summary
    total_models = len(critical_models) + len(advisory_models)
    total_critical = len(critical_models)
    lines.extend([
        "",
        "## Summary",
        "",
        f"- **Total models checked:** {total_models}",
        f"- **Critical models:** {total_critical}",
        f"- **Advisory models:** {len(advisory_models)}",
        f"- **Overall verdict:** {overall_status}",
        "",
    ])

    return lines


def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    print("Feature Routing Audit (SAF-01)")
    print("=" * 40)

    if args.registry_version and args.registry_version != REGISTRY_VERSION:
        print(
            f"ERROR: Registry version mismatch: "
            f"expected={args.registry_version}, actual={REGISTRY_VERSION}",
        )
        sys.exit(1)

    results = run_audit(args.output_dir)

    # Print summary to stdout
    print(f"Registry version: {REGISTRY_VERSION}")
    print(f"Overall status: {results['overall_status']}")
    print()

    print("Critical Models:")
    for m in results["critical_models"]:
        status_marker = "PASS" if m["status"] == "PASS" else "FAIL"
        intersections = m["forbidden_intersections"]
        extra = f" intersections={intersections}" if intersections else ""
        print(
            f"  [{status_marker}] {m['model_name']} "
            f"({m['checked_feature_count']} features){extra}"
        )

    print()
    print("Advisory Models:")
    for m in results["advisory_models"]:
        status_marker = m["status"]
        intersections = m["warning_intersections"]
        extra = f" intersections={intersections}" if intersections else ""
        print(
            f"  [{status_marker}] {m['model_name']} "
            f"({m['checked_feature_count']} features){extra}"
        )

    print()
    print(f"JSON report: {results['json_path']}")
    print(f"Markdown report: {results['md_path']}")

    if results["overall_status"] == "FAIL":
        print()
        print("AUDIT FAILED: forbidden feature intersections detected")
        sys.exit(1)
    else:
        print()
        print("AUDIT PASSED: no forbidden feature intersections")
        sys.exit(0)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
