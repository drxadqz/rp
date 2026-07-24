from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from write_queue_recovery_report import _annotate_processes, _process_snapshot


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
WATCHER_KINDS = {
    "fast_screen_followup",
    "direct_visual_followup",
    "rscd27_followup",
    "v17_candidate_followup",
    "postprocess_followup",
    "waiting_queue",
}
WAIT_RE = re.compile(r"Get-Process\s+-Id\s+(\d+)", re.IGNORECASE)
SCRIPT_RE = re.compile(r"scripts[/\\][\w_]+\.(?:py|ps1)[^\r\n]*", re.IGNORECASE)
MESSAGE_RE = re.compile(r"-Value\s+\"([^\"]+)\"", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode queued follow-up PowerShell watchers into a readable execution-chain report."
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "followup_watcher_report.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "followup_watcher_report.json",
    )
    args = parser.parse_args()

    report = build_report()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report() -> dict[str, Any]:
    processes = _annotate_processes(_process_snapshot())
    watchers = [_summarize_watcher(proc) for proc in processes if proc.get("kind") in WATCHER_KINDS]
    watchers = sorted(watchers, key=lambda row: (str(row.get("wait_pid") or ""), str(row.get("pid") or "")))
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_watchers": len(watchers),
        "watchers": watchers,
        "notes": [
            "This report only decodes already-running watcher processes.",
            "It does not start, stop, or reprioritize any GPU job.",
            "A watcher is a GPU blocker if its decoded commands can launch training or evaluation after its wait PID exits.",
        ],
    }


def _summarize_watcher(proc: dict[str, Any]) -> dict[str, Any]:
    decoded = str(proc.get("decoded_command") or "")
    raw = str(proc.get("CommandLine") or "")
    text = decoded or raw
    wait_pids = WAIT_RE.findall(text)
    scripts = _compact_unique(SCRIPT_RE.findall(text), limit=12, width=220)
    messages = _compact_unique(MESSAGE_RE.findall(text), limit=10, width=180)
    return {
        "kind": proc.get("kind"),
        "pid": proc.get("ProcessId") or proc.get("pid"),
        "parent_pid": proc.get("ParentProcessId") or proc.get("parent_pid"),
        "wait_pid": wait_pids[0] if wait_pids else proc.get("wait_pid"),
        "wait_pids": wait_pids,
        "messages": messages,
        "script_commands": scripts,
        "decoded_chars": len(decoded),
        "command_short": " ".join(text.split())[:360],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Follow-up Watcher Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Watchers: `{report['num_watchers']}`",
        "",
        "## Notes",
        "",
    ]
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    lines.extend(["", "## Watcher Chain", ""])
    watchers = report.get("watchers") or []
    if not watchers:
        lines.append("No follow-up watcher processes were found.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(["| Kind | PID | Parent | Wait PID | Planned scripts |", "|---|---:|---:|---:|---|"])
    for watcher in watchers:
        planned = "<br>".join(f"`{_escape(cmd)}`" for cmd in watcher.get("script_commands", [])[:5])
        if len(watcher.get("script_commands", [])) > 5:
            planned += f"<br>... +{len(watcher.get('script_commands', [])) - 5} more"
        lines.append(
            "| {kind} | {pid} | {parent} | {wait_pid} | {planned} |".format(
                kind=watcher.get("kind"),
                pid=watcher.get("pid"),
                parent=watcher.get("parent_pid"),
                wait_pid=watcher.get("wait_pid") or "-",
                planned=planned or "-",
            )
        )

    lines.extend(["", "## Watcher Messages", ""])
    for watcher in watchers:
        lines.append(f"### PID {watcher.get('pid')} `{watcher.get('kind')}`")
        messages = watcher.get("messages") or []
        if messages:
            for message in messages:
                lines.append(f"- {_escape(message)}")
        else:
            lines.append("- No status messages parsed.")
        lines.append("")
    return "\n".join(lines)


def _compact_unique(values: list[str], *, limit: int, width: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = " ".join(str(value).split())
        if len(compact) > width:
            compact = compact[: width - 3] + "..."
        if compact and compact not in seen:
            seen.add(compact)
            out.append(compact)
        if len(out) >= limit:
            break
    return out


def _escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|")


if __name__ == "__main__":
    main()
