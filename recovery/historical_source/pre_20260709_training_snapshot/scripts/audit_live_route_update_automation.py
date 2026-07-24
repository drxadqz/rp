from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _file_row(path: Path, *, required: bool = True) -> dict[str, Any]:
    exists = path.exists()
    row: dict[str, Any] = {
        "path": str(path),
        "required": required,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
    }
    if exists:
        row["mtime"] = path.stat().st_mtime
    return row


def build_report(project_root: Path, summary_dir: Path) -> dict[str, Any]:
    script = project_root / "scripts" / "write_live_research_route_update.py"
    audit_script = project_root / "scripts" / "audit_live_route_update_automation.py"
    postprocess = project_root / "scripts" / "postprocess_protocol_outputs.py"
    runner = project_root / "scripts" / "run_paper_protocol_direct.py"
    md = summary_dir / "live_research_route_update.md"
    js = summary_dir / "live_research_route_update.json"

    postprocess_text = _read(postprocess)
    runner_text = _read(runner)
    script_call = "scripts/write_live_research_route_update.py"
    audit_call = "scripts/audit_live_route_update_automation.py"
    active_training_call = "scripts/write_active_live_training_reports.py"
    objective_audit_call = "scripts/write_objective_completion_audit.py"
    checks = [
        {
            "name": "generator_script_exists",
            "level": "pass" if script.exists() else "block",
            "message": "Live route generator script exists." if script.exists() else "Live route generator script is missing.",
        },
        {
            "name": "markdown_report_exists",
            "level": "pass" if md.exists() and md.stat().st_size > 0 else "block",
            "message": "Live route markdown report exists." if md.exists() and md.stat().st_size > 0 else "Live route markdown report is missing or empty.",
        },
        {
            "name": "json_report_exists",
            "level": "pass" if js.exists() and js.stat().st_size > 0 else "block",
            "message": "Live route JSON report exists." if js.exists() and js.stat().st_size > 0 else "Live route JSON report is missing or empty.",
        },
        {
            "name": "postprocess_invokes_generator",
            "level": "pass" if script_call in postprocess_text else "block",
            "message": "Full postprocess invokes live route generator." if script_call in postprocess_text else "Full postprocess does not invoke live route generator.",
        },
        {
            "name": "lightweight_refresh_invokes_generator",
            "level": "pass" if script_call in runner_text else "block",
            "message": "Lightweight queue refresh invokes live route generator." if script_call in runner_text else "Lightweight queue refresh does not invoke live route generator.",
        },
        {
            "name": "postprocess_invokes_audit",
            "level": "pass" if audit_call in postprocess_text else "warn",
            "message": "Full postprocess invokes live route automation audit." if audit_call in postprocess_text else "Full postprocess does not yet invoke live route automation audit.",
        },
        {
            "name": "lightweight_refresh_invokes_audit",
            "level": "pass" if audit_call in runner_text else "warn",
            "message": "Lightweight queue refresh invokes live route automation audit." if audit_call in runner_text else "Lightweight queue refresh does not yet invoke live route automation audit.",
        },
        {
            "name": "postprocess_invokes_active_training_reports",
            "level": "pass" if active_training_call in postprocess_text else "warn",
            "message": "Full postprocess refreshes active-run training trend and diagnosis." if active_training_call in postprocess_text else "Full postprocess does not refresh active-run training trend and diagnosis.",
        },
        {
            "name": "lightweight_refresh_invokes_active_training_reports",
            "level": "pass" if active_training_call in runner_text else "warn",
            "message": "Lightweight queue refresh refreshes active-run training trend and diagnosis." if active_training_call in runner_text else "Lightweight queue refresh does not refresh active-run training trend and diagnosis.",
        },
        {
            "name": "postprocess_invokes_objective_completion_audit",
            "level": "pass" if objective_audit_call in postprocess_text else "warn",
            "message": "Full postprocess refreshes the strict objective completion audit." if objective_audit_call in postprocess_text else "Full postprocess does not refresh the strict objective completion audit.",
        },
        {
            "name": "lightweight_refresh_invokes_objective_completion_audit",
            "level": "pass" if objective_audit_call in runner_text else "warn",
            "message": "Lightweight queue refresh refreshes the strict objective completion audit." if objective_audit_call in runner_text else "Lightweight queue refresh does not refresh the strict objective completion audit.",
        },
    ]
    levels = [row["level"] for row in checks]
    verdict = "block" if "block" in levels else ("warn" if "warn" in levels else "pass")
    return {
        "verdict": verdict,
        "checks": checks,
        "files": [
            _file_row(script),
            _file_row(audit_script),
            _file_row(postprocess),
            _file_row(runner),
            _file_row(md),
            _file_row(js),
        ],
        "policy": "The live route and active training reports must be regenerated by both full postprocess and lightweight queue refresh so long-running experiments keep the research route synchronized with current evidence.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Route Update Automation Audit",
        "",
        f"Verdict: `{report.get('verdict', '-')}`",
        "",
        report.get("policy", ""),
        "",
        "## Checks",
        "",
        "| Level | Check | Message |",
        "|---|---|---|",
    ]
    for row in report.get("checks", []):
        lines.append(f"| {row.get('level')} | `{row.get('name')}` | {row.get('message')} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "| Exists | Size | Path |",
            "|---:|---:|---|",
        ]
    )
    for row in report.get("files", []):
        lines.append(f"| {row.get('exists')} | {row.get('size_bytes')} | `{row.get('path')}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--summary-dir", type=Path, default=Path("reports/paper_protocol_summary"))
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    summary_dir = args.summary_dir
    report = build_report(project_root, summary_dir)
    out_md = args.out_md or summary_dir / "live_route_update_automation_audit.md"
    out_json = args.out_json or summary_dir / "live_route_update_automation_audit.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
