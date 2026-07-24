from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from write_queue_recovery_report import _annotate_processes, _decode_encoded_command, _process_snapshot


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")

GPU_BLOCKING_KINDS = {
    "active_training",
    "active_evaluation",
    "active_postprocess",
    "queue_orchestrator",
    "fast_screen_followup",
    "direct_visual_followup",
    "rscd27_followup",
    "v17_candidate_followup",
    "fail_fast_candidate_gate",
}

WATCHER_KINDS = {
    "fast_screen_followup",
    "direct_visual_followup",
    "rscd27_followup",
    "v17_candidate_followup",
    "postprocess_followup",
    "fail_fast_candidate_gate",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report whether it is safe to manually launch another GPU-heavy experiment."
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "gpu_scheduling_guard_report.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "gpu_scheduling_guard_report.json",
    )
    args = parser.parse_args()

    report = build_report()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report() -> dict[str, Any]:
    processes = _annotate_processes(_process_snapshot())
    blockers = [
        _summarize_process(proc)
        for proc in processes
        if str(proc.get("kind") or "") in GPU_BLOCKING_KINDS
    ]
    watchers = [
        _summarize_process(proc)
        for proc in processes
        if str(proc.get("kind") or "") in WATCHER_KINDS
    ]
    verdict = "busy" if blockers else "idle"
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": verdict,
        "manual_launch_allowed": verdict == "idle",
        "blockers": blockers,
        "watchers": watchers,
        "policy": [
            "Do not manually start RSCD per-day, foundation probes, or another formal queue while blockers exist.",
            "Watcher processes count as blockers because they may launch GPU jobs after their wait condition is satisfied.",
            "Manual launch is allowed only when this report says idle and no training/evaluation/postprocess process is visible.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GPU Scheduling Guard Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        f"Manual launch allowed: `{report['manual_launch_allowed']}`",
        "",
        "## Policy",
        "",
    ]
    for item in report.get("policy", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Blocking Processes", ""])
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(["| Kind | PID | Parent | Command |", "|---|---:|---:|---|"])
        for proc in blockers:
            lines.append(
                "| {kind} | {pid} | {parent} | `{cmd}` |".format(
                    kind=proc.get("kind"),
                    pid=proc.get("pid"),
                    parent=proc.get("parent_pid"),
                    cmd=_escape(proc.get("command_short")),
                )
            )
    else:
        lines.append("No GPU-blocking processes were found.")
    lines.extend(["", "## Watchers", ""])
    watchers = report.get("watchers") or []
    if watchers:
        lines.extend(["| Kind | PID | Parent | Command |", "|---|---:|---:|---|"])
        for proc in watchers:
            lines.append(
                "| {kind} | {pid} | {parent} | `{cmd}` |".format(
                    kind=proc.get("kind"),
                    pid=proc.get("pid"),
                    parent=proc.get("parent_pid"),
                    cmd=_escape(proc.get("command_short")),
                )
            )
    else:
        lines.append("No follow-up watcher processes were found.")
    lines.append("")
    return "\n".join(lines)


def _summarize_process(proc: dict[str, Any]) -> dict[str, Any]:
    cmd = str(proc.get("CommandLine") or proc.get("command") or "")
    decoded = _decode_encoded_command(cmd)
    display = decoded or cmd
    return {
        "kind": proc.get("kind"),
        "pid": proc.get("ProcessId") or proc.get("pid"),
        "parent_pid": proc.get("ParentProcessId") or proc.get("parent_pid"),
        "command_short": " ".join(str(display).split())[:260],
    }


def _escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|")


if __name__ == "__main__":
    main()
