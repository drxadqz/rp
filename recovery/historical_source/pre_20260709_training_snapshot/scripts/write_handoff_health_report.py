from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from write_queue_recovery_report import DEFAULT_PYTHON, build_report as build_queue_recovery_report


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")

REQUIRED_ARTIFACTS = [
    "best.pt",
    "evaluate_test.json",
    "detailed_test.json",
    "interval_calibration_90.json",
    "bootstrap_metrics.json",
    "topvenue_result_audit.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "handoff_health_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "handoff_health_report.json")
    args = parser.parse_args()

    report = build_report(args.root, args.summary_dir, args.log_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(root: Path, summary_dir: Path, log_dir: Path) -> dict[str, Any]:
    queue_recovery = _fresh_queue_recovery(root, summary_dir, log_dir) or _load_json(summary_dir / "queue_recovery_report.json") or {}
    processes = _process_snapshot()
    latest_handoff = _latest_handoff_log(log_dir)
    v5 = _artifact_status(root / "v5_full_faf")
    roadsaw = _artifact_status(root / "lodo_roadsaw_full_faf")
    rscd = _artifact_status(root / "lodo_rscd_full_faf")
    queue_order = [
        row.get("run") or row.get("name")
        for row in queue_recovery.get("queue_order", [])
        if isinstance(row, dict)
    ]
    roadsaw_priority = _before(queue_order, "lodo_roadsaw_full_faf", "lodo_rscd_full_faf")
    active_rows = [
        row
        for row in queue_recovery.get("active_rows", [])
        if isinstance(row, dict)
    ]
    active_names = [str(row.get("name") or row.get("run")) for row in active_rows]
    roadsaw_delayed_by_active_queue = (
        bool(active_names)
        and not roadsaw["complete"]
        and "lodo_roadsaw_full_faf" not in active_names
        and any(name.startswith("lodo_") for name in active_names)
    )
    handoff_processes = [
        proc for proc in processes if _is_powershell_file_process(proc, "handoff_to_roadsaw_priority_after_v5.ps1")
    ]
    priority_watcher_processes = [
        proc for proc in processes if _is_powershell_file_process(proc, "watch_roadsaw_priority_after_current_lodo.ps1")
    ]
    followup_processes = [
        proc for proc in processes if _is_powershell_file_process(proc, "run_paper_protocol_after_pid.ps1")
    ]
    priority_fast_screen_followup = any(
        "runfastscreenbeforefollowup" in _norm_cmd(proc).lower()
        for proc in [*priority_watcher_processes, *followup_processes]
    )
    queue_processes = [
        proc for proc in processes if _is_python_process(proc) and "run_paper_protocol_direct.py" in _norm_cmd(proc)
    ]
    train_processes = [
        proc for proc in processes if _is_python_process(proc) and "scripts/train.py" in _norm_cmd(proc)
    ]
    if roadsaw["complete"]:
        verdict = "handoff_no_longer_needed_roadsaw_complete"
    elif not roadsaw_priority:
        verdict = "priority_order_failure"
    elif not v5["complete"] and queue_processes and train_processes and not handoff_processes:
        verdict = "handoff_not_required_priority_queue_active"
    elif v5["complete"] and not handoff_processes and not priority_watcher_processes and not queue_processes:
        verdict = "needs_manual_resume_after_v5"
    elif roadsaw_delayed_by_active_queue and queue_processes:
        verdict = "post_v5_queue_active_roadsaw_delayed"
    elif v5["complete"] and queue_processes:
        verdict = "post_v5_queue_active"
    elif handoff_processes:
        verdict = "watching_v5"
    else:
        verdict = "watcher_missing_before_v5_complete"
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "log_dir": str(log_dir),
        "verdict": verdict,
        "roadsaw_priority_order": roadsaw_priority,
        "roadsaw_delayed_by_active_queue": roadsaw_delayed_by_active_queue,
        "active_rows": active_rows,
        "next_incomplete": _compact_next_incomplete(queue_recovery.get("next_incomplete")),
        "v5_full_faf": v5,
        "lodo_roadsaw_full_faf": roadsaw,
        "lodo_rscd_full_faf": rscd,
        "handoff_processes": handoff_processes,
        "priority_watcher_processes": priority_watcher_processes,
        "followup_processes": followup_processes,
        "priority_fast_screen_followup": priority_fast_screen_followup,
        "queue_processes": queue_processes,
        "train_processes": train_processes,
        "latest_handoff_log": latest_handoff,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Handoff Health Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Priority",
        "",
        f"- RoadSaW priority order: `{report.get('roadsaw_priority_order')}`.",
        f"- RoadSaW delayed by active queue: `{report.get('roadsaw_delayed_by_active_queue')}`.",
        f"- Next incomplete: `{_format_next_incomplete(report.get('next_incomplete'))}`.",
        "",
        "## Artifacts",
        "",
        "| Run | Complete | Missing |",
        "|---|---:|---|",
    ]
    for key in ["v5_full_faf", "lodo_roadsaw_full_faf", "lodo_rscd_full_faf"]:
        row = report.get(key, {})
        missing = ", ".join(row.get("missing", [])) or "-"
        lines.append(f"| `{key}` | `{row.get('complete')}` | {missing} |")
    lines.extend(["", "## Processes", ""])
    lines.append(f"- Handoff watchers: `{len(report.get('handoff_processes', []))}`.")
    lines.append(f"- RoadSaW priority watchers: `{len(report.get('priority_watcher_processes', []))}`.")
    lines.append(f"- Follow-up watchers: `{len(report.get('followup_processes', []))}`.")
    lines.append(f"- Fast-screen + promotion follow-up enabled: `{report.get('priority_fast_screen_followup')}`.")
    lines.append(f"- Queue processes: `{len(report.get('queue_processes', []))}`.")
    lines.append(f"- Train processes: `{len(report.get('train_processes', []))}`.")
    for row in report.get("active_rows", []):
        lines.append(f"- Active queue row: `{_format_next_incomplete(row)}`.")
    log = report.get("latest_handoff_log") or {}
    if log:
        lines.extend(["", "## Latest Handoff Log", ""])
        lines.append(f"- Path: `{log.get('path')}`.")
        for item in log.get("tail", []):
            lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def _artifact_status(path: Path) -> dict[str, Any]:
    present = [name for name in REQUIRED_ARTIFACTS if (path / name).exists()]
    missing = [name for name in REQUIRED_ARTIFACTS if name not in present]
    return {
        "path": str(path),
        "complete": not missing,
        "present": present,
        "missing": missing,
        "has_last_checkpoint": (path / "last.pt").exists(),
    }


def _format_next_incomplete(item: Any) -> str:
    if not isinstance(item, dict) or not item:
        return "-"
    name = item.get("name") or "-"
    status = item.get("status") or "-"
    epoch = item.get("active_epoch")
    epochs = item.get("active_epochs")
    step = item.get("active_step")
    steps = item.get("active_steps")
    if epoch and epochs and step and steps:
        return f"{name} {status} epoch {epoch}/{epochs} step {step}/{steps}"
    return f"{name} {status}"


def _latest_handoff_log(log_dir: Path) -> dict[str, Any]:
    candidates = sorted(
        list(log_dir.glob("roadsaw_priority_after_*.log")) + list(log_dir.glob("roadsaw_priority_handoff_*.log")),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {}
    path = candidates[0]
    tail = [_clean_text(line) for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[-12:]]
    return {
        "path": str(path),
        "last_modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "tail": tail,
    }


def _process_snapshot() -> list[dict[str, Any]]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.ProcessId -ne $PID -and ($_.CommandLine -like '*run_paper_protocol*' -or "
        "$_.CommandLine -like '*handoff_to_roadsaw_priority_after_v5*' -or "
        "$_.CommandLine -like '*watch_roadsaw_priority_after_current_lodo*' -or "
        "$_.CommandLine -like '*scripts\\\\train.py*' -or "
        "$_.CommandLine -like '*scripts/train.py*') } | "
        "Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine | ConvertTo-Json -Depth 3"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except Exception as exc:
        return [{"error": str(exc)}]
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return [{"error": proc.stdout.strip()[:500]}]
    if isinstance(data, dict):
        data = [data]
    return [
        {
            "pid": item.get("ProcessId"),
            "parent_pid": item.get("ParentProcessId"),
            "name": item.get("Name"),
            "executable_path": item.get("ExecutablePath"),
            "command_line": item.get("CommandLine") or "",
        }
        for item in data
        if isinstance(item, dict)
    ]


def _norm_cmd(proc: dict[str, Any]) -> str:
    return str(proc.get("command_line", "")).replace("\\", "/")


def _is_python_process(proc: dict[str, Any]) -> bool:
    name = str(proc.get("name", "")).lower()
    exe = str(proc.get("executable_path", "")).replace("\\", "/").lower()
    return name == "python.exe" or exe.endswith("/python.exe")


def _is_powershell_file_process(proc: dict[str, Any], script_name: str) -> bool:
    name = str(proc.get("name", "")).lower()
    cmd = _norm_cmd(proc).lower()
    return name in {"powershell.exe", "pwsh.exe"} and " -file " in f" {cmd} " and script_name.lower() in cmd


def _clean_text(value: Any) -> str:
    return str(value).replace("\ufeff", "").strip()


def _compact_next_incomplete(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {
        "name": item.get("name") or item.get("run"),
        "status": item.get("status"),
        "active_epoch": item.get("active_epoch"),
        "active_epochs": item.get("active_epochs"),
        "active_step": item.get("active_step"),
        "active_steps": item.get("active_steps"),
    }


def _before(items: list[Any], first: str, second: str) -> bool:
    try:
        return items.index(first) < items.index(second)
    except ValueError:
        return False


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fresh_queue_recovery(root: Path, summary_dir: Path, log_dir: Path) -> dict[str, Any] | None:
    try:
        return build_queue_recovery_report(root, summary_dir, log_dir, DEFAULT_PYTHON)
    except Exception:
        return None


if __name__ == "__main__":
    main()
