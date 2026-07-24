from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_OUT = Path("reports/paper_protocol_audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = sorted([path for path in args.root.glob("*") if path.is_dir()])
    rows = []
    for out_dir in outputs:
        audit_json = args.out_dir / f"{out_dir.name}_audit.json"
        audit_md = args.out_dir / f"{out_dir.name}_audit.md"
        subprocess.run(
            [
                sys.executable,
                "scripts/audit_topvenue_results.py",
                "--output-dir",
                str(out_dir),
                "--out-md",
                str(audit_md),
                "--out-json",
                str(audit_json),
            ],
            check=True,
        )
        report = json.loads(audit_json.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": out_dir.name,
                "verdict": report.get("verdict"),
                "num_blocks": sum(1 for item in report.get("checks", []) if item.get("level") == "block"),
                "num_warnings": sum(1 for item in report.get("checks", []) if item.get("level") == "warn"),
                "blockers": [
                    item.get("name")
                    for item in report.get("checks", [])
                    if item.get("level") == "block"
                ],
                "metrics": report.get("metrics", {}),
            }
        )
    payload: dict[str, Any] = {"root": str(args.root), "runs": rows}
    (args.out_dir / "paper_protocol_audit_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md = render_markdown(payload)
    (args.out_dir / "paper_protocol_audit_summary.md").write_text(md, encoding="utf-8")
    print(md)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Paper Protocol Audit Summary", "", f"Root: `{payload['root']}`", ""]
    lines.append("| Run | Verdict | Blocks | Warnings | risk F1 | risk CI | calibrated coverage | coverage CI | dataset-ID bal acc | Blockers |")
    lines.append("|---|---|---:|---:|---:|---|---:|---|---:|---|")
    for row in payload["runs"]:
        metrics = row.get("metrics", {})
        lines.append(
            "| {name} | {verdict} | {blocks} | {warnings} | {risk} | {risk_ci} | {cov} | {cov_ci} | {dataset} | {blockers} |".format(
                name=row["name"],
                verdict=row["verdict"],
                blocks=row["num_blocks"],
                warnings=row["num_warnings"],
                risk=fmt(metrics.get("risk_macro_f1")),
                risk_ci=metrics.get("risk_macro_f1_ci") or "-",
                cov=fmt(metrics.get("calibrated_test_coverage")),
                cov_ci=metrics.get("calibrated_test_coverage_ci") or "-",
                dataset=fmt(metrics.get("dataset_id_balanced_accuracy")),
                blockers=", ".join(row.get("blockers", [])) or "-",
            )
        )
    lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


if __name__ == "__main__":
    main()
