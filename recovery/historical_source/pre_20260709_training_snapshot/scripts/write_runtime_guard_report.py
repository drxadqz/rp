from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")
STALE_LOG_SECONDS = 600
MIN_FREE_GB = 10.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "runtime_guard_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "runtime_guard_report.json")
    args = parser.parse_args()

    report = build_report(args.root, args.summary_dir, args.log_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(root: Path, summary_dir: Path, log_dir: Path) -> dict[str, Any]:
    queue = _load_json(summary_dir / "queue_recovery_report.json") or {}
    handoff = _load_json(summary_dir / "handoff_health_report.json") or {}
    dashboard = _load_json(summary_dir / "experiment_status_dashboard.json") or {}
    active_watch = _load_json(summary_dir / "active_training_watch_report.json") or {}
    processes = _process_snapshot()
    gpu = _gpu_snapshot()
    disks = _disk_snapshot(["C:\\", "D:\\", "E:\\"])
    active = _active_row(queue, dashboard, active_watch)
    log_health = _log_health(active)
    log_errors = _recent_log_errors(log_dir)
    checks: list[dict[str, Any]] = []

    queue_count = sum(
        1 for proc in processes if _is_python_process(proc) and "run_paper_protocol_direct.py" in _norm_cmd(proc)
    )
    train_count = sum(
        1 for proc in processes if _is_python_process(proc) and "scripts/train.py" in _norm_cmd(proc)
    )
    handoff_count = sum(
        1 for proc in processes if _is_powershell_file_process(proc, "handoff_to_roadsaw_priority_after_v5.ps1")
    )
    priority_watcher_count = sum(
        1 for proc in processes if _is_powershell_file_process(proc, "watch_roadsaw_priority_after_current_lodo.ps1")
    )
    followup_processes = [proc for proc in processes if _is_followup_waiter_process(proc)]
    followup_count = len(followup_processes)

    _add_check(checks, "queue_process", "pass" if queue_count == 1 else "block", f"Queue process count is {queue_count}.")
    if active:
        active_status = str(active.get("status", ""))
        if "running" in active_status or "partial" in active_status:
            _add_check(checks, "active_worker", "pass" if train_count >= 1 else "block", f"Train process count is {train_count}.")
    handoff_verdict = handoff.get("verdict")
    handoff_optional = handoff_verdict in {
        "handoff_not_required_priority_queue_active",
        "post_v5_queue_active",
        "post_v5_queue_active_roadsaw_delayed",
        "handoff_no_longer_needed_roadsaw_complete",
    }
    _add_check(
        checks,
        "handoff_process",
        "pass" if handoff_count + priority_watcher_count >= 1 or handoff_optional else "warn",
        f"Handoff watcher count is {handoff_count}; RoadSaW priority watcher count is {priority_watcher_count}.",
    )
    _add_check(
        checks,
        "followup_waiters",
        "pass" if followup_count <= 8 else "warn",
        f"Follow-up waiter count is {followup_count}.",
    )
    if followup_processes:
        normalized_followups = [_norm_cmd(proc).lower() for proc in followup_processes]
        grace_enabled = all("prioritywatchergraceseconds" in cmd for cmd in normalized_followups)
        roadsaw_priority_done = handoff_verdict == "handoff_no_longer_needed_roadsaw_complete"
        _add_check(
            checks,
            "followup_priority_grace",
            "pass" if grace_enabled or roadsaw_priority_done else "warn",
            (
                "RoadSaW priority handoff is complete; follow-up priority grace is no longer required."
                if roadsaw_priority_done
                else (
                    "Follow-up watcher has RoadSaW priority grace enabled."
                    if grace_enabled
                    else "Follow-up watcher is missing RoadSaW priority grace."
                )
            ),
        )
        fast_screen_enabled = any(
            "runfastscreenbeforefollowup" in cmd
            or "run_fast_screen_protocol.py" in cmd
            or "fast_screen" in cmd
            for cmd in normalized_followups
        )
        _add_check(
            checks,
            "followup_fast_screen_promotion",
            "pass" if fast_screen_enabled else "warn",
            (
                "Follow-up watcher will run fast-screen and promotion selection before the main queue."
                if fast_screen_enabled
                else "Follow-up watcher will skip fast-screen/promotion selection."
            ),
        )

    stale = log_health.get("age_seconds")
    if stale is None:
        _add_check(checks, "active_log", "warn", "No active log age could be read.")
    else:
        level = "pass" if float(stale) <= STALE_LOG_SECONDS else "block"
        _add_check(checks, "active_log", level, f"Active log age is {stale:.1f}s.")

    acceptable_handoff = {
        "watching_v5",
        "handoff_not_required_priority_queue_active",
        "post_v5_queue_active",
        "handoff_no_longer_needed_roadsaw_complete",
    }
    roadsaw_handoff_level = "pass" if handoff_verdict in acceptable_handoff else "block"
    if handoff_verdict == "post_v5_queue_active_roadsaw_delayed":
        roadsaw_handoff_level = "warn"
    _add_check(
        checks,
        "roadsaw_handoff",
        roadsaw_handoff_level,
        f"Handoff verdict is {handoff_verdict}.",
    )

    if gpu.get("available"):
        _add_check(checks, "gpu_visible", "pass", f"GPU {gpu.get('name')} visible, memory {gpu.get('memory_used_mb')}/{gpu.get('memory_total_mb')} MB.")
    else:
        _add_check(checks, "gpu_visible", "block", f"GPU unavailable: {gpu.get('error')}")

    low_disks = [disk for disk in disks if disk.get("free_gb", 0.0) < MIN_FREE_GB]
    _add_check(
        checks,
        "disk_free",
        "pass" if not low_disks else "warn",
        "All tracked disks have enough free space." if not low_disks else f"Low space: {[disk.get('path') for disk in low_disks]}",
    )

    if log_errors:
        _add_check(checks, "recent_log_errors", "block", f"Found {len(log_errors)} recent error markers.")
    else:
        _add_check(checks, "recent_log_errors", "pass", "No recent Traceback/OOM/error markers in queue logs.")

    verdict = "pass"
    if any(item["level"] == "block" for item in checks):
        verdict = "block"
    elif any(item["level"] == "warn" for item in checks):
        verdict = "warn"

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "log_dir": str(log_dir),
        "verdict": verdict,
        "active": active,
        "log_health": log_health,
        "processes": processes,
        "gpu": gpu,
        "disks": disks,
        "checks": checks,
        "recent_log_errors": log_errors[:10],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Runtime Guard Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Checks",
        "",
        "| Level | Check | Message |",
        "|---|---|---|",
    ]
    for item in report.get("checks", []):
        lines.append(f"| {item.get('level')} | `{item.get('name')}` | {item.get('message')} |")
    active = report.get("active") or {}
    if active:
        lines.extend(["", "## Active Row", ""])
        phase = active.get("active_phase") or "step"
        lines.append(
            "- `{name}` `{status}` epoch `{epoch}/{epochs}` {phase} `{step}/{steps}`.".format(
                name=active.get("name"),
                status=active.get("status"),
                epoch=active.get("active_epoch", "-"),
                epochs=active.get("active_epochs", "-"),
                phase=phase,
                step=active.get("active_step", "-"),
                steps=active.get("active_steps", "-"),
            )
        )
    errors = report.get("recent_log_errors", [])
    if errors:
        lines.extend(["", "## Recent Error Markers", ""])
        for item in errors[:10]:
            lines.append(f"- `{item.get('path')}`: `{item.get('line')}`")
    lines.append("")
    return "\n".join(lines)


def _active_row(queue: dict[str, Any], dashboard: dict[str, Any], active_watch: dict[str, Any]) -> dict[str, Any]:
    dashboard_active = {}
    active_rows = dashboard.get("active_rows", [])
    if active_rows and isinstance(active_rows[0], dict):
        dashboard_active = active_rows[0]

    watch_active = _active_from_watch(active_watch)
    for row in queue.get("queue_order", []):
        if not isinstance(row, dict):
            continue
        if row.get("status") in {"running_or_partial", "partial_ci_missing"}:
            row = _merge_active_watch(row, watch_active)
            if dashboard_active and dashboard_active.get("name") == row.get("name"):
                dashboard_epoch = _int(dashboard_active.get("active_epoch"))
                queue_epoch = _int(row.get("active_epoch"))
                if dashboard_epoch is not None and queue_epoch is not None:
                    if dashboard_epoch < queue_epoch:
                        return _merge_active_watch(row, watch_active)
                    if dashboard_epoch > queue_epoch:
                        merged = dict(row)
                        for key, value in dashboard_active.items():
                            if value is not None:
                                merged[key] = value
                        return _merge_active_watch(merged, watch_active)
                dashboard_step = _int(dashboard_active.get("active_step"))
                queue_step = _int(row.get("active_step"))
                if dashboard_step is not None and (queue_step is None or dashboard_step >= queue_step):
                    merged = dict(row)
                    for key, value in dashboard_active.items():
                        if value is not None:
                            merged[key] = value
                    return _merge_active_watch(merged, watch_active)
            return _merge_active_watch(row, watch_active)
    if dashboard_active:
        return _merge_active_watch(dashboard_active, watch_active)
    return watch_active


def _active_from_watch(report: dict[str, Any]) -> dict[str, Any]:
    active = report.get("active")
    if not isinstance(active, dict) or not active.get("name"):
        return {}
    out: dict[str, Any] = {
        "name": active.get("name"),
        "status": active.get("status"),
        "active_phase": active.get("phase"),
        "active_epoch": active.get("epoch"),
        "active_epochs": active.get("epochs"),
        "active_step": active.get("step"),
        "active_steps": active.get("steps"),
        "active_tqdm_eta": active.get("eta"),
        "active_tqdm_rate": active.get("rate"),
    }
    if active.get("err_log"):
        out["active_log"] = active.get("err_log")
    elif active.get("out_log"):
        out["active_log"] = active.get("out_log")
    return {key: value for key, value in out.items() if value is not None}


def _merge_active_watch(base: dict[str, Any], watch: dict[str, Any]) -> dict[str, Any]:
    if not watch or watch.get("name") != base.get("name"):
        return base
    base_phase = str(base.get("active_phase") or "")
    watch_phase = str(watch.get("active_phase") or "")
    base_epoch = _int(base.get("active_epoch"))
    watch_epoch = _int(watch.get("active_epoch"))
    base_step = _int(base.get("active_step"))
    watch_step = _int(watch.get("active_step"))
    use_watch = False
    if watch_phase and watch_phase != base_phase:
        use_watch = True
    if watch_epoch is not None and base_epoch is not None:
        use_watch = use_watch or watch_epoch > base_epoch or (watch_epoch == base_epoch and (base_step is None or (watch_step or -1) >= base_step))
    elif watch_step is not None and (base_step is None or watch_step >= base_step):
        use_watch = True
    if not use_watch:
        return base
    merged = dict(base)
    for key, value in watch.items():
        if value is not None:
            merged[key] = value
    return merged


def _log_health(active: dict[str, Any]) -> dict[str, Any]:
    if not active:
        return {}
    path = active.get("active_log")
    age = active.get("active_log_age_seconds")
    if age is None and path:
        try:
            age = datetime.now().timestamp() - Path(path).stat().st_mtime
        except OSError:
            age = None
    return {
        "path": path,
        "age_seconds": _num(age),
        "stale": bool(active.get("active_log_stale")),
    }


def _recent_log_errors(log_dir: Path) -> list[dict[str, str]]:
    if not log_dir.exists():
        return []
    markers = ["Traceback", "CUDA out of memory", "RuntimeError", "Exception"]
    logs = sorted(
        list(log_dir.glob("*.err.log")) + list(log_dir.glob("*.out.log")) + list(log_dir.glob("*.log")),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[:8]
    hits: list[dict[str, str]] = []
    for path in logs:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-200:]
        except OSError:
            continue
        for line in lines:
            clean = line.replace("\ufeff", "").strip()
            if any(marker in clean for marker in markers):
                hits.append({"path": str(path), "line": clean[:240]})
    return hits


def _process_snapshot() -> list[dict[str, Any]]:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.ProcessId -ne $PID -and ($_.CommandLine -like '*run_paper_protocol*' -or "
        "$_.CommandLine -like '*handoff_to_roadsaw_priority_after_v5*' -or "
        "$_.CommandLine -like '*watch_roadsaw_priority_after_current_lodo*' -or "
        "$_.CommandLine -like '*scripts\\\\train.py*' -or "
        "$_.CommandLine -like '*scripts/train.py*' -or "
        "$_.CommandLine -like '*EncodedCommand*') } | "
        "Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine | ConvertTo-Json -Depth 3"
    )
    try:
        proc = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, timeout=15)
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
            "decoded_command": _decode_encoded_command(item.get("CommandLine") or ""),
        }
        for item in data
        if isinstance(item, dict)
    ]


def _norm_cmd(proc: dict[str, Any]) -> str:
    raw = str(proc.get("command_line", ""))
    decoded = str(proc.get("decoded_command") or "")
    return f"{raw}\n{decoded}".replace("\\", "/")


def _is_python_process(proc: dict[str, Any]) -> bool:
    name = str(proc.get("name", "")).lower()
    exe = str(proc.get("executable_path", "")).replace("\\", "/").lower()
    return name == "python.exe" or exe.endswith("/python.exe")


def _is_powershell_file_process(proc: dict[str, Any], script_name: str) -> bool:
    name = str(proc.get("name", "")).lower()
    cmd = _norm_cmd(proc).lower()
    return name in {"powershell.exe", "pwsh.exe"} and script_name.lower() in cmd


def _is_followup_waiter_process(proc: dict[str, Any]) -> bool:
    name = str(proc.get("name", "")).lower()
    if name not in {"powershell.exe", "pwsh.exe"}:
        return False
    cmd = _norm_cmd(proc).lower()
    if "get-ciminstance win32_process" in cmd:
        return False
    keywords = [
        "run_paper_protocol_after_pid.ps1",
        "v17_lean_quality_physics_safety",
        "v17 formal",
        "run_fast_screen_protocol.py",
        "fast_screen",
        "direct_visual_friction",
        "extremeroad",
        "extreme_road",
        "run_rscd_surface_classification.py",
        "rscd-27",
        "rscd27",
        "postprocess_protocol_outputs.py",
    ]
    return any(keyword in cmd for keyword in keywords)


def _decode_encoded_command(command: str) -> str | None:
    match = re.search(r"-(?:EncodedCommand|Enc)\s+([A-Za-z0-9+/=]+)", command, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return base64.b64decode(match.group(1)).decode("utf-16le", errors="replace")
    except Exception:
        return None


def _gpu_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"available": False, "error": proc.stderr.strip()}
    first = [part.strip() for part in proc.stdout.strip().splitlines()[0].split(",")]
    keys = ["name", "temperature_c", "utilization_percent", "memory_used_mb", "memory_total_mb", "power_w"]
    return {"available": True, **dict(zip(keys, first))}


def _disk_snapshot(paths: list[str]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            out.append({"path": path, "error": str(exc)})
            continue
        out.append(
            {
                "path": path,
                "free_gb": round(usage.free / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "total_gb": round(usage.total / (1024**3), 2),
            }
        )
    return out


def _count_matching(processes: list[dict[str, Any]], needle: str) -> int:
    return sum(1 for proc in processes if needle in proc.get("command_line", ""))


def _add_check(checks: list[dict[str, Any]], name: str, level: str, message: str) -> None:
    checks.append({"name": name, "level": level, "message": message})


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
