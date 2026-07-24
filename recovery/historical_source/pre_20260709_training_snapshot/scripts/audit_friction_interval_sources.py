from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from friction_affordance.ontology import (  # noqa: E402
    FRICTION_INTERVAL_BENCHMARKS,
    FRICTION_INTERVAL_REFERENCE_SOURCES,
    weak_mu_interval_from_state,
)


DEFAULT_OUT = Path("reports/paper_protocol_summary/friction_interval_source_audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT.with_suffix(".json"))
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT.with_suffix(".md"))
    args = parser.parse_args()

    report = build_report()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report() -> dict[str, Any]:
    rows = []
    for item in FRICTION_INTERVAL_BENCHMARKS:
        low, high = weak_mu_interval_from_state(
            friction=item["mapped_state"],
            wetness=item["mapped_state"],
            snow=item["mapped_state"],
            material=item.get("mapped_material"),
        )
        reference_low = float(item["reference_low"])
        reference_high = float(item["reference_high"])
        contains = (
            low is not None
            and high is not None
            and float(low) <= reference_low
            and float(high) >= reference_high
        )
        width = None if low is None or high is None else float(high) - float(low)
        reference_width = reference_high - reference_low
        rows.append(
            {
                **item,
                "ontology_low": low,
                "ontology_high": high,
                "ontology_width": width,
                "reference_width": reference_width,
                "contains_reference": contains,
                "status": "pass" if contains else "fail",
            }
        )
    verdict = "pass" if all(row["contains_reference"] for row in rows) else "fail"
    return {
        "verdict": verdict,
        "task_framing": (
            "The project estimates a visual-evidence-conditioned friction affordance "
            "interval. Public RSCD/RoadSaW/RoadSC labels are not synchronized "
            "friction-meter or tire-dynamics ground truth."
        ),
        "sources": FRICTION_INTERVAL_REFERENCE_SOURCES,
        "rows": rows,
        "policy": (
            "Ontology intervals should be conservative envelopes around public "
            "road-condition/TRFC references. Narrow point estimates from vehicle-"
            "specific tests are not used as exact labels because tire, speed, "
            "temperature, water depth, and load are unobserved in the public "
            "image datasets."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Friction Interval Source Audit",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        report["task_framing"],
        "",
        "## Public Reference Sources",
        "",
        "| Source key | DOI / URL | Role |",
        "|---|---|---|",
    ]
    for key, item in report["sources"].items():
        lines.append(f"| `{key}` | {item['doi']} | {item['role']} |")
    lines.extend(
        [
            "",
            "## Ontology Coverage Check",
            "",
            "| Anchor | mapped state | reference interval | ontology interval | status | source |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| {anchor} | {state} | [{rlow:.2f}, {rhigh:.2f}] | {ontology} | {status} | {source} |".format(
                anchor=row["anchor"],
                state=row["mapped_state"],
                rlow=float(row["reference_low"]),
                rhigh=float(row["reference_high"]),
                ontology=_fmt_interval(row["ontology_low"], row["ontology_high"]),
                status=row["status"],
                source=row["source"],
            )
        )
    lines.extend(["", "## Policy", "", report["policy"], ""])
    return "\n".join(lines)


def _fmt_interval(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "-"
    return f"[{float(low):.2f}, {float(high):.2f}]"


if __name__ == "__main__":
    main()
