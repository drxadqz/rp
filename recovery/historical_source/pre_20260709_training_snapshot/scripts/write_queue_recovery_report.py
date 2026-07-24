from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")
DEFAULT_PYTHON = Path(r"D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe")
TQDM_RE = re.compile(
    r"(?P<phase>train|eval):\s*(?P<pct>\d+)%.*\|\s*"
    r"(?P<step>\d+)\s*/\s*(?P<steps>\d+)\s*"
    r"\[(?P<elapsed>[^<\]]+)<(?P<eta>[^,\]]+),\s*(?P<rate>[^\]]+)\]"
)

QUEUE_ORDER = [
    "v0_global_only",
    "v1_physics_texture",
    "v2_friction_set",
    "v3_dg_losses",
    "v4_evidence_aux",
    "v5_full_faf",
    "lodo_roadsaw_full_faf",
    "lodo_rscd_full_faf",
    "lodo_roadsc_full_faf",
    "single_roadsaw_full_faf",
    "single_rscd_full_faf",
    "single_roadsc_full_faf",
    "baseline_single_roadsaw_global_convnext",
    "baseline_single_rscd_global_convnext",
    "baseline_single_roadsc_global_convnext",
    "v6_full_faf_fourier",
    "v7_full_faf_fourier_dann",
    "v8_full_faf_fourier_roadprior",
    "v9_full_faf_roadsaw_hard_sampling",
    "v10_full_faf_consistency",
    "v11_full_faf_domain_adapter",
    "v12_full_faf_roi_interval_safety",
    "v13_lean_physics_evidence",
    "v14_lean_road_roi_safety",
    "v15_lean_bottom_square_style_safety",
    "v16_lean_bottom_square_color_constancy_safety",
    "v17_lean_quality_physics_safety",
    "v18_lean_mixstyle_quality_safety",
    "v19_lean_state_contrast_quality_safety",
    "v20_lean_interval_order_quality_safety",
    "v21_lean_quality_uncertainty_safety",
    "v22_lean_quality_order_contrast_safety",
    "v23_lean_region_mixture_evidence_safety",
    "v24_lean_multi_query_region_evidence_safety",
    "v25_lean_masked_query_consistency_safety",
    "final_lodo_roadsaw_lean_road_roi_safety",
    "final_lodo_rscd_lean_road_roi_safety",
    "final_lodo_roadsc_lean_road_roi_safety",
    "final_single_roadsaw_lean_road_roi_safety",
    "final_single_rscd_lean_road_roi_safety",
    "final_single_roadsc_lean_road_roi_safety",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "queue_recovery_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "queue_recovery_report.json")
    args = parser.parse_args()

    report = build_report(args.root, args.summary_dir, args.log_dir, args.python)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(root: Path, summary_dir: Path, log_dir: Path, python: Path) -> dict[str, Any]:
    cached_progress = _load_json(summary_dir / "paper_protocol_progress.json") or []
    disk_progress = _load_progress_from_dirs(root)
    cached_by_name = {row.get("name"): row for row in cached_progress if isinstance(row, dict)}
    progress = []
    for disk_row in disk_progress:
        cached_row = cached_by_name.get(disk_row.get("name"), {})
        # Disk artifacts are the authority for completion/running state. Cached
        # progress can lag behind after post-processing finishes.
        progress.append({**cached_row, **disk_row})
    by_name = {row.get("name"): row for row in progress}
    ordered = [_queue_row(name, by_name.get(name, {}), log_dir, root / name) for name in QUEUE_ORDER]
    active = [row for row in ordered if row["status"] in {"running_or_partial", "partial_ci_missing"}]
    next_missing = next((row for row in ordered if row["status"] != "complete"), None)
    complete = [row for row in ordered if row["status"] == "complete"]
    missing = [row for row in ordered if row["status"] == "missing"]
    partial = [row for row in ordered if row["status"] not in {"complete", "missing"}]
    process_snapshot = _annotate_processes(_process_snapshot())
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "summary_dir": str(summary_dir),
        "log_dir": str(log_dir),
        "queue_order": ordered,
        "num_total": len(ordered),
        "num_complete": len(complete),
        "num_missing": len(missing),
        "num_partial": len(partial),
        "active_rows": active,
        "next_incomplete": next_missing,
        "process_snapshot": process_snapshot,
        "tail": _queue_log_tail(log_dir),
        "recovery_commands": _recovery_commands(python, root, log_dir),
    }


def _queue_row(name: str, row: dict[str, Any], log_dir: Path, run_dir: Path) -> dict[str, Any]:
    status = row.get("status", "missing")
    state = _load_json(run_dir / "training_state.json") or {}
    epoch = state.get("epoch", row.get("epoch"))
    epochs = state.get("epochs", row.get("epochs"))
    active = _active_log_progress(name, log_dir) if status != "complete" else {}
    if status != "complete":
        epoch = state.get("epoch", epoch)
        epochs = state.get("epochs", epochs)
    has_active_log_epoch = active.get("active_epoch") is not None
    return {
        "name": name,
        "status": status,
        "epoch": epoch,
        "epochs": epochs,
        "active_epoch": None if status == "complete" else active.get("active_epoch", row.get("active_epoch")),
        "active_epochs": None if status == "complete" else active.get("active_epochs", row.get("active_epochs")),
        "active_step": None if status == "complete" else active.get("active_step") if has_active_log_epoch else row.get("active_step"),
        "active_steps": None if status == "complete" else active.get("active_steps") if has_active_log_epoch else row.get("active_steps"),
        "active_phase": None if status == "complete" else active.get("active_phase", row.get("active_phase")),
        "active_log": None if status == "complete" else active.get("active_log", row.get("active_log")),
        "active_log_mtime": None if status == "complete" else active.get("active_log_mtime", row.get("active_log_mtime")),
        "active_log_age_seconds": None if status == "complete" else active.get("active_log_age_seconds", row.get("active_log_age_seconds")),
        "active_log_stale": None if status == "complete" else active.get("active_log_stale", row.get("active_log_stale")),
        "last_update": _latest_mtime(run_dir) or row.get("last_update"),
        "artifacts": row.get("artifacts", []),
    }


def _load_progress_from_dirs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for name in QUEUE_ORDER:
        path = root / name
        status = "missing"
        if path.exists():
            required = [
                "best.pt",
                "evaluate_test.json",
                "detailed_test.json",
                "interval_calibration_90.json",
                "bootstrap_metrics.json",
                "topvenue_result_audit.json",
            ]
            status = "complete" if all((path / item).exists() for item in required) else "running_or_partial"
        rows.append({"name": name, "status": status})
    return rows


def _process_snapshot() -> list[dict[str, str]]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*run_paper_protocol*' -or $_.CommandLine -like '*enforce_fail_fast_candidate_gate.py*' -or $_.CommandLine -like '*scripts\\\\train.py*' -or $_.CommandLine -like '*scripts/train.py*' -or $_.CommandLine -like '*evaluate.py*' -or $_.CommandLine -like '*evaluate_detailed.py*' -or $_.CommandLine -like '*calibrate_intervals.py*' -or $_.CommandLine -like '*bootstrap_metrics.py*' -or $_.CommandLine -like '*dataset_id_diagnostic.py*' -or $_.CommandLine -like '*export_evidence_maps.py*' -or $_.CommandLine -like '*analyze_evidence_field.py*' -or $_.CommandLine -like '*summarize_confusions.py*' -or $_.CommandLine -like '*audit_topvenue_results.py*' -or $_.CommandLine -like '*EncodedCommand*' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
    except Exception as exc:
        return [{"error": str(exc)}]
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return [{"raw": proc.stdout.strip()}]
    if isinstance(payload, dict):
        payload = [payload]
    return payload if isinstance(payload, list) else []


def _annotate_processes(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for proc in processes:
        row = dict(proc)
        cmd = str(row.get("CommandLine") or row.get("raw") or "")
        decoded = _decode_encoded_command(cmd)
        searchable = f"{cmd}\n{decoded or ''}"
        cmd_lower = cmd.lower()
        lower = searchable.lower()
        if "get-ciminstance win32_process" in lower:
            continue
        if "long-lived powershell ast parser" in lower or "command-safety layer" in lower:
            continue
        if "enforce_fail_fast_candidate_gate.py" in lower:
            kind = "fail_fast_candidate_gate"
        elif "scripts/train.py" in cmd_lower or "scripts\\train.py" in cmd_lower:
            kind = "active_training"
        elif "evaluate.py" in cmd_lower or "evaluate_detailed.py" in cmd_lower:
            kind = "active_evaluation"
        elif (
            "calibrate_intervals.py" in cmd_lower
            or "bootstrap_metrics.py" in cmd_lower
            or "dataset_id_diagnostic.py" in cmd_lower
            or "export_evidence_maps.py" in cmd_lower
            or "analyze_evidence_field.py" in cmd_lower
            or "summarize_confusions.py" in cmd_lower
            or "audit_topvenue_results.py" in cmd_lower
        ):
            kind = "active_postprocess"
        elif "v17_lean_quality_physics_safety" in lower or "v17 formal" in lower:
            kind = "v17_candidate_followup"
        elif "run_paper_protocol_after_pid.ps1" in lower:
            kind = "fast_screen_followup"
        elif "run_fast_screen_protocol.py" in lower or "fast_screen" in lower:
            kind = "fast_screen_followup"
        elif "direct_visual_friction" in lower or "extremeroad" in lower or "extreme_road" in lower:
            kind = "direct_visual_followup"
        elif "run_rscd_surface_classification.py" in lower or "rscd-27" in lower or "rscd27" in lower:
            kind = "rscd27_followup"
        elif "postprocess_protocol_outputs.py" in lower:
            kind = "postprocess_followup"
        elif "run_paper_protocol_direct.py" in lower and "--wait-pid" in lower:
            kind = "waiting_queue"
        elif "run_paper_protocol_direct.py" in lower:
            kind = "queue_orchestrator"
        else:
            kind = "other"
        row["kind"] = kind
        row["decoded_command"] = decoded
        row["phase"] = _cmd_arg(searchable, "--phase")
        row["wait_pid"] = _cmd_arg(searchable, "--wait-pid") or _cmd_arg(searchable, "-WaitPid")
        out.append(row)
    return out


def _decode_encoded_command(command: str) -> str | None:
    match = re.search(r"-(?:EncodedCommand|Enc)\s+([A-Za-z0-9+/=]+)", command, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return base64.b64decode(match.group(1)).decode("utf-16le", errors="replace")
    except Exception:
        return None


def _cmd_arg(command: str, flag: str) -> str | None:
    pattern = re.compile(rf"{re.escape(flag)}\s+([^\s]+)")
    match = pattern.search(command)
    return match.group(1) if match else None


def _active_log_progress(run_name: str, log_dir: Path) -> dict[str, Any]:
    logs = sorted(
        log_dir.glob(f"{run_name}_train_*.out.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not logs:
        return {}
    path = logs[0]
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    mtime = path.stat().st_mtime
    age_seconds = max(0.0, datetime.now().timestamp() - mtime)
    progress: dict[str, Any] = {
        "active_log": str(path),
        "active_log_mtime": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "active_log_age_seconds": round(age_seconds, 1),
        # Training logs print every configured log interval, while eval/postprocess
        # can be quieter. Treat this as a conservative warning, not a failure.
        "active_log_stale": age_seconds > 30 * 60,
    }
    epoch_re = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
    step_re = re.compile(r"train step\s+(\d+)\s*/\s*(\d+)")
    for line in lines[-250:]:
        epoch_match = epoch_re.search(line)
        if epoch_match:
            progress["active_epoch"] = int(epoch_match.group(1))
            progress["active_epochs"] = int(epoch_match.group(2))
            progress.pop("active_step", None)
            progress.pop("active_steps", None)
        step_match = step_re.search(line)
        if step_match:
            progress["active_step"] = int(step_match.group(1))
            progress["active_steps"] = int(step_match.group(2))
    _merge_tqdm_progress(progress, log_dir, run_name)
    return progress


def _merge_tqdm_progress(progress: dict[str, Any], log_dir: Path, run_name: str) -> None:
    tqdm = _tqdm_snapshot(log_dir, run_name)
    if not tqdm:
        return
    progress["active_phase"] = tqdm.get("phase")
    step = _as_int(tqdm.get("step"))
    steps = _as_int(tqdm.get("steps"))
    if step is not None and steps is not None:
        current_step = _as_int(progress.get("active_step"))
        progress["active_step"] = step
        progress["active_steps"] = steps
        progress["active_tqdm_percent"] = tqdm.get("percent")
        progress["active_tqdm_eta"] = tqdm.get("eta")
        progress["active_tqdm_rate"] = tqdm.get("rate")
    age = _as_float(tqdm.get("age_seconds"))
    if age is not None and age < _as_float(progress.get("active_log_age_seconds"), default=age + 1.0):
        progress["active_log"] = tqdm.get("log")
        progress["active_log_mtime"] = tqdm.get("mtime")
        progress["active_log_age_seconds"] = round(age, 1)
        progress["active_log_stale"] = age > 30 * 60


def _tqdm_snapshot(log_dir: Path, run_name: str) -> dict[str, Any] | None:
    candidates = sorted(
        log_dir.glob(f"*{run_name}*.err.log"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    path = candidates[0]
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in reversed(lines[-500:]):
        match = TQDM_RE.search(line)
        if not match:
            continue
        mtime = path.stat().st_mtime
        return {
            "log": str(path),
            "mtime": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "age_seconds": max(0.0, datetime.now().timestamp() - mtime),
            "phase": match.group("phase"),
            "percent": int(match.group("pct")),
            "step": int(match.group("step")),
            "steps": int(match.group("steps")),
            "elapsed": match.group("elapsed").strip(),
            "eta": match.group("eta").strip(),
            "rate": match.group("rate").strip(),
        }
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _queue_log_tail(log_dir: Path, lines: int = 20) -> list[str]:
    path = log_dir / "run_paper_protocol_direct.log"
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def _latest_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    latest = max((item.stat().st_mtime for item in path.rglob("*") if item.is_file()), default=None)
    if latest is None:
        return None
    return datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S")


def _recovery_commands(python: Path, root: Path, log_dir: Path) -> dict[str, str]:
    return {
        "resume_all": (
            f"{python} scripts\\run_paper_protocol_direct.py --phase all "
            f"--python {python} --root {root} --log-dir {log_dir} --postprocess-each"
        ),
        "refresh_postprocess": (
            f"{python} scripts\\postprocess_protocol_outputs.py --root {root} "
            "--summary-dir reports\\paper_protocol_summary"
        ),
        "status_dashboard": f"{python} scripts\\write_experiment_dashboard.py",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Queue Recovery Report",
        "",
        f"Generated at: {report['generated_at']}",
        f"Root: `{report['root']}`",
        f"Log dir: `{report['log_dir']}`",
        "",
        "## Summary",
        "",
        f"- Total queued runs: `{report['num_total']}`.",
        f"- Complete: `{report['num_complete']}`; partial/running: `{report['num_partial']}`; missing: `{report['num_missing']}`.",
    ]
    active = report.get("active_rows", [])
    if active:
        lines.append("- Active: " + ", ".join(_run_text(row) for row in active) + ".")
    else:
        lines.append("- Active: none detected.")
    next_item = report.get("next_incomplete")
    if next_item:
        lines.append(f"- Next incomplete in queue order: `{next_item.get('name')}` (`{next_item.get('status')}`).")
    lines.extend(["", "## Queue Order", ""])
    lines.append("| # | Run | Status | Progress | Log age | Last update |")
    lines.append("|---:|---|---|---|---:|---|")
    for idx, row in enumerate(report.get("queue_order", []), start=1):
        lines.append(
            f"| {idx} | `{row.get('name')}` | `{row.get('status')}` | {_progress_text(row)} | {_log_age_text(row)} | {row.get('last_update') or '-'} |"
        )
    lines.extend(["", "## Process Snapshot", ""])
    processes = report.get("process_snapshot", [])
    if not processes:
        lines.append("- No queue/train processes found.")
    for proc in processes:
        cmd = str(proc.get("decoded_command") or proc.get("CommandLine") or proc.get("raw") or proc.get("error") or "-")
        if len(cmd) > 240:
            cmd = cmd[:237] + "..."
        detail = []
        if proc.get("phase"):
            detail.append(f"phase={proc.get('phase')}")
        if proc.get("wait_pid"):
            detail.append(f"wait_pid={proc.get('wait_pid')}")
        suffix = f" ({', '.join(detail)})" if detail else ""
        lines.append(
            f"- `{proc.get('kind', 'other')}` PID `{proc.get('ProcessId', '-')}` "
            f"parent `{proc.get('ParentProcessId', '-')}`{suffix}: `{cmd}`"
        )
    lines.extend(["", "## Log Tail", ""])
    for line in report.get("tail", []):
        lines.append(f"- `{line}`")
    lines.extend(["", "## Recovery Commands", ""])
    for name, cmd in report.get("recovery_commands", {}).items():
        lines.append(f"- `{name}`: `{cmd}`")
    lines.append("")
    return "\n".join(lines)


def _run_text(row: dict[str, Any]) -> str:
    return f"{row.get('name')} {_progress_text(row)}"


def _progress_text(row: dict[str, Any]) -> str:
    epoch = row.get("active_epoch") or row.get("epoch")
    epochs = row.get("active_epochs") or row.get("epochs")
    step = row.get("active_step")
    steps = row.get("active_steps")
    phase = row.get("active_phase") or "step"
    if epoch is None:
        return "-"
    if step is not None and steps is not None:
        return f"epoch {epoch}/{epochs}, {phase} {step}/{steps}"
    return f"epoch {epoch}/{epochs}"


def _log_age_text(row: dict[str, Any]) -> str:
    age = row.get("active_log_age_seconds")
    if age is None:
        return "-"
    try:
        seconds = float(age)
    except (TypeError, ValueError):
        return "-"
    suffix = " stale" if row.get("active_log_stale") else ""
    if seconds < 60:
        return f"{seconds:.0f}s{suffix}"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m{suffix}"
    return f"{seconds / 3600:.1f}h{suffix}"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    main()
