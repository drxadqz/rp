from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from extract_training_history_from_log import parse_log as parse_training_log


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")
TRAIN_STEP_RE = re.compile(r"train step\s+(\d+)/(\d+)\s+loss=([-+0-9.eE]+)")
EPOCH_RE = re.compile(r"^Epoch\s+(\d+)/(\d+)")
TQDM_RE = re.compile(
    r"(?P<phase>train|eval):\s*(?P<pct>\d+)%.*\|\s*"
    r"(?P<step>\d+)\s*/\s*(?P<steps>\d+)\s*"
    r"\[(?P<elapsed>[^<\]]+)<(?P<eta>[^,\]]+),\s*(?P<rate>[^\]]+)\]"
)
ERROR_RE = re.compile(r"Traceback|CUDA out of memory|out of memory|RuntimeError|Error", re.IGNORECASE)
POSTPROCESS_STAGES = {
    "calibrate_intervals.py": "calibrate_intervals",
    "bootstrap_metrics.py": "bootstrap_metrics",
    "dataset_id_diagnostic.py": "dataset_id_diagnostic",
    "export_evidence_maps.py": "evidence_maps",
    "analyze_evidence_field.py": "evidence_field_audit",
    "audit_topvenue_results.py": "audit",
    "slim_best_checkpoints.py": "slim_best_checkpoints",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "active_training_watch_report.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "active_training_watch_report.json",
    )
    args = parser.parse_args()

    report = build_report(args.root, args.summary_dir, args.log_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(root: Path, summary_dir: Path, log_dir: Path) -> dict[str, Any]:
    dashboard = _load_json(summary_dir / "experiment_status_dashboard.json") or {}
    queue = _load_json(summary_dir / "queue_recovery_report.json") or {}
    runtime = _load_json(summary_dir / "runtime_guard_report.json") or {}
    gpu_guard = _load_json(summary_dir / "gpu_scheduling_guard_report.json") or {}
    active = _active_row(dashboard, queue)
    process_active = _active_training_from_processes()
    gpu_active = _active_training_for_gpu_guard(gpu_guard)
    if process_active:
        active = process_active
    elif gpu_active:
        active = gpu_active
    if not active:
        return {
            "generated_at": _now(),
            "verdict": "idle",
            "active": None,
            "message": "No active training row detected.",
        }

    name = str(active.get("name"))
    run_dir = root / name
    train_out_log = _find_log(name, log_dir, "out", active.get("active_log"))
    out_log = train_out_log
    err_log = _find_log(name, log_dir, "err", active.get("active_log"))
    eval_err_log = _find_eval_log(name, log_dir, "err")
    active_postprocess = _active_postprocess_for_name(name, gpu_guard)
    postprocess_stage = active_postprocess.get("stage")
    if postprocess_stage:
        stage_out_log = _find_stage_log(name, log_dir, postprocess_stage, "out")
        stage_err_log = _find_stage_log(name, log_dir, postprocess_stage, "err")
        out_log = stage_out_log or out_log
        err_log = stage_err_log or err_log
    history = _parse_history(train_out_log)
    completed = [row for row in history if row.get("val_metrics")]
    latest = completed[-1] if completed else {}
    previous = completed[-2] if len(completed) >= 2 else {}
    recent_steps = _recent_train_steps(train_out_log)
    recent_same_epoch_steps = _same_epoch_tail(recent_steps)
    latest_epoch_marker = _latest_epoch_marker(train_out_log)
    tqdm = _latest_tqdm(err_log)
    eval_tqdm = _latest_tqdm(eval_err_log)
    if not postprocess_stage and _prefer_eval_tqdm(eval_tqdm, tqdm, eval_err_log, err_log):
        tqdm = eval_tqdm
        err_log = eval_err_log
    active_errors = _active_errors([out_log, err_log])
    checkpoints = {
        "best": (run_dir / "best.pt").exists(),
        "best_safety": (run_dir / "best_safety.pt").exists(),
        "last": (run_dir / "last.pt").exists(),
    }

    latest_train_epoch = recent_steps[-1].get("epoch") if recent_steps else None
    active_epoch = _prefer_int(
        tqdm.get("epoch"),
        latest_epoch_marker.get("epoch"),
        active.get("active_epoch"),
        latest_train_epoch,
    )
    active_epochs = _prefer_int(
        latest_epoch_marker.get("epochs"),
        active.get("active_epochs"),
        active.get("epochs"),
    )
    active_step = _prefer_int(tqdm.get("step"), active.get("active_step"))
    active_steps = _prefer_int(tqdm.get("steps"), active.get("active_steps"))
    if postprocess_stage and not tqdm:
        active_step = None
        active_steps = None
    phase = postprocess_stage or tqdm.get("phase") or active.get("active_phase")

    verdict = "running"
    warnings: list[str] = []
    if active_errors:
        verdict = "attention"
        warnings.append("Active logs contain error markers.")
    if not checkpoints["best"] and completed:
        verdict = "attention"
        warnings.append("Completed validation exists but best.pt is missing.")
    if runtime.get("verdict") == "block":
        verdict = "attention"
        warnings.append("Runtime guard is blocking.")

    return {
        "generated_at": _now(),
        "verdict": verdict,
        "warnings": warnings,
        "active": {
            "name": name,
            "status": active.get("status"),
            "phase": phase,
            "epoch": active_epoch,
            "epochs": active_epochs,
            "step": active_step,
            "steps": active_steps,
            "eta": tqdm.get("eta"),
            "rate": tqdm.get("rate"),
            "pid": active_postprocess.get("pid") or active.get("pid"),
            "out_log": str(out_log) if out_log else None,
            "err_log": str(err_log) if err_log else None,
        },
        "latest_completed_epoch": _compact_epoch(latest),
        "previous_completed_epoch": _compact_epoch(previous),
        "delta_latest_vs_previous": _epoch_delta(latest, previous),
        "recent_train_steps": (recent_same_epoch_steps or recent_steps)[-8:],
        "recent_loss_slope": _loss_slope((recent_same_epoch_steps or recent_steps)[-8:]),
        "recent_loss_scope": "current_epoch" if recent_same_epoch_steps else "all_recent",
        "checkpoints": checkpoints,
        "runtime_guard": {
            "verdict": runtime.get("verdict"),
            "checks": runtime.get("checks", []),
        },
        "active_errors": active_errors[:10],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Active Training Watch Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report.get('verdict')}`",
        "",
    ]
    active = report.get("active") or {}
    if not active:
        lines.append(report.get("message", "No active run."))
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "## Active Run",
            "",
            "- `{name}` `{status}` phase `{phase}`, epoch `{epoch}/{epochs}`, step `{step}/{steps}`, ETA `{eta}`, rate `{rate}`.".format(
                name=active.get("name"),
                status=active.get("status"),
                phase=active.get("phase") or "-",
                epoch=active.get("epoch") or "-",
                epochs=active.get("epochs") or "-",
                step=active.get("step") or "-",
                steps=active.get("steps") or "-",
                eta=active.get("eta") or "-",
                rate=active.get("rate") or "-",
            ),
            f"- Out log: `{active.get('out_log')}`",
            f"- Err log: `{active.get('err_log')}`",
            "",
        ]
    )

    latest = report.get("latest_completed_epoch") or {}
    previous = report.get("previous_completed_epoch") or {}
    delta = report.get("delta_latest_vs_previous") or {}
    lines.extend(["## Latest Completed Epoch", ""])
    if latest:
        lines.append(
            "- Epoch `{epoch}`: val loss `{loss}`, risk acc `{risk}`, friction acc `{friction}`, raw coverage `{coverage}`, raw width `{width}`.".format(
                epoch=latest.get("epoch"),
                loss=_fmt(latest.get("val_loss")),
                risk=_fmt_pct(latest.get("val_acc_risk")),
                friction=_fmt_pct(latest.get("val_acc_friction")),
                coverage=_fmt_pct(latest.get("val_mu_interval_coverage")),
                width=_fmt(latest.get("val_mu_interval_width")),
            )
        )
    else:
        lines.append("- No completed validation epoch parsed yet.")
    if previous:
        lines.append(
            "- Delta vs previous: val loss `{loss}`, risk acc `{risk}`, raw coverage `{coverage}`.".format(
                loss=_fmt_delta(delta.get("val_loss")),
                risk=_fmt_delta_pct(delta.get("val_acc_risk")),
                coverage=_fmt_delta_pct(delta.get("val_mu_interval_coverage")),
            )
        )
    lines.append("")

    lines.extend(["## Recent Train Loss", ""])
    steps = report.get("recent_train_steps") or []
    if steps:
        lines.append("| Epoch | Step | Loss |")
        lines.append("|---:|---:|---:|")
        for row in steps:
            lines.append(f"| {row.get('epoch')} | {row.get('step')}/{row.get('steps')} | {_fmt(row.get('loss'))} |")
        lines.append("")
        scope = "current epoch" if report.get("recent_loss_scope") == "current_epoch" else "displayed points"
        lines.append(f"- Recent loss slope over {scope}: `{_fmt_delta(report.get('recent_loss_slope'))}`.")
    else:
        lines.append("- No train step loss lines parsed yet.")
    lines.append("")

    checkpoints = report.get("checkpoints") or {}
    lines.extend(["## Artifacts", ""])
    lines.append(
        "- Checkpoints: best `{best}`, best_safety `{best_safety}`, last `{last}`.".format(
            best=checkpoints.get("best"),
            best_safety=checkpoints.get("best_safety"),
            last=checkpoints.get("last"),
        )
    )
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for item in warnings:
            lines.append(f"- {item}")
    errors = report.get("active_errors") or []
    if errors:
        lines.extend(["", "## Active Error Markers", ""])
        for item in errors:
            lines.append(f"- `{item.get('path')}` line `{item.get('line_number')}`: `{item.get('line')}`")
    lines.append("")
    return "\n".join(lines)


def _active_row(dashboard: dict[str, Any], queue: dict[str, Any]) -> dict[str, Any]:
    dashboard_active = {}
    active_rows = dashboard.get("active_rows") or []
    if active_rows and isinstance(active_rows[0], dict):
        dashboard_active = active_rows[0]
    queue_active = {}
    for row in queue.get("queue_order", []):
        if isinstance(row, dict) and row.get("status") in {"running_or_partial", "partial_ci_missing"}:
            queue_active = row
            break
    return _prefer_active_row(dashboard_active, queue_active)


def _prefer_active_row(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if not left:
        return right
    if not right:
        return left
    if left.get("name") != right.get("name"):
        left_age = _num(left.get("active_log_age_seconds"))
        right_age = _num(right.get("active_log_age_seconds"))
        if left_age is not None and right_age is not None and left_age != right_age:
            return left if left_age < right_age else right
        if left_age is None and right_age is not None:
            return right
        if right_age is None and left_age is not None:
            return left
    left_epoch = _prefer_int(left.get("active_epoch"), left.get("epoch"))
    right_epoch = _prefer_int(right.get("active_epoch"), right.get("epoch"))
    if left_epoch is not None and right_epoch is not None and left_epoch != right_epoch:
        return left if left_epoch > right_epoch else right
    left_step = _prefer_int(left.get("active_step"))
    right_step = _prefer_int(right.get("active_step"))
    if left_step is not None and right_step is not None and left_step != right_step:
        return left if left_step > right_step else right
    if right.get("active_log_age_seconds") is not None and left.get("active_log_age_seconds") is not None:
        return right if float(right["active_log_age_seconds"]) <= float(left["active_log_age_seconds"]) else left
    return right


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _active_postprocess_for_name(name: str, gpu_guard: dict[str, Any]) -> dict[str, Any]:
    for row in gpu_guard.get("blockers", []):
        if not isinstance(row, dict) or row.get("kind") != "active_postprocess":
            continue
        command = str(row.get("command_short") or "")
        if name not in command:
            continue
        for marker, stage in POSTPROCESS_STAGES.items():
            if marker in command:
                return {
                    "pid": row.get("pid"),
                    "stage": stage,
                    "command_short": command,
                }
        return {
            "pid": row.get("pid"),
            "stage": "postprocess",
            "command_short": command,
        }
    return {}


def _active_training_for_gpu_guard(gpu_guard: dict[str, Any]) -> dict[str, Any]:
    for row in gpu_guard.get("blockers", []):
        if not isinstance(row, dict) or row.get("kind") != "active_training":
            continue
        command = str(row.get("command_short") or "")
        name = _run_name_from_train_command(command)
        if not name:
            continue
        return {
            "name": name,
            "status": "running_or_partial",
            "active_phase": "train",
            "pid": row.get("pid"),
        }
    return {}


def _active_training_from_processes() -> dict[str, Any]:
    rows = _windows_process_rows()
    for row in rows:
        command = str(row.get("CommandLine") or row.get("commandline") or "")
        normalized = command.replace("\\", "/")
        if "scripts/train.py" not in normalized:
            continue
        name = _run_name_from_train_command(command)
        if not name:
            continue
        return {
            "name": name,
            "status": "running_or_partial",
            "active_phase": "train",
            "pid": row.get("ProcessId") or row.get("processid"),
        }
    return {}


def _windows_process_rows() -> list[dict[str, Any]]:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -like '*scripts/train.py*' -or $_.CommandLine -like '*scripts\\train.py*' } | "
            "ForEach-Object { [string]$_.ProcessId + \"`t\" + [string]$_.CommandLine }"
        ),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    rows: list[dict[str, Any]] = []
    for raw in proc.stdout.splitlines():
        if "\t" not in raw:
            continue
        pid, command_line = raw.split("\t", 1)
        rows.append({"ProcessId": pid.strip(), "CommandLine": command_line.strip()})
    return rows


def _run_name_from_train_command(command: str) -> str | None:
    normalized = command.replace("\\", "/")
    marker = "configs/experiments/paper_protocol/"
    if marker not in normalized:
        return None
    tail = normalized.split(marker, 1)[1].split()[0].strip("'\"")
    return Path(tail).stem or None


def _find_log(name: str, log_dir: Path, kind: str, active_log: Any = None) -> Path | None:
    suffix = f".{kind}.log"
    candidates: list[Path] = []
    if active_log:
        path = Path(str(active_log))
        if path.name.endswith(suffix):
            candidates.extend([path, Path.cwd() / path])
        stem = path.name.rsplit(".", 2)[0]
        candidates.append(path.with_name(f"{stem}.{kind}.log"))
        candidates.append(log_dir / f"{stem}.{kind}.log")
    candidates.extend(sorted(log_dir.glob(f"{name}_*.{kind}.log"), key=lambda p: p.stat().st_mtime, reverse=True))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_stage_log(name: str, log_dir: Path, stage: str, kind: str) -> Path | None:
    candidates = sorted(
        log_dir.glob(f"{name}_{stage}_*.{kind}.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _find_eval_log(name: str, log_dir: Path, kind: str) -> Path | None:
    candidates = sorted(
        log_dir.glob(f"{name}_evaluate_*.{kind}.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _prefer_eval_tqdm(
    eval_tqdm: dict[str, Any],
    current_tqdm: dict[str, Any],
    eval_log: Path | None,
    current_log: Path | None,
) -> bool:
    if not eval_tqdm or eval_tqdm.get("phase") != "eval" or not eval_log:
        return False
    if _prefer_int(eval_tqdm.get("step")) and _prefer_int(eval_tqdm.get("steps")):
        step = int(eval_tqdm["step"])
        steps = int(eval_tqdm["steps"])
        if step < steps:
            return True
    if not current_tqdm:
        return True
    if not current_log:
        return True
    try:
        return eval_log.stat().st_mtime >= current_log.stat().st_mtime
    except OSError:
        return True


def _parse_history(path: Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    try:
        return parse_training_log(path)
    except (OSError, UnicodeDecodeError):
        return []


def _recent_train_steps(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    current_epoch = None
    current_epochs = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for raw in lines:
        line = raw.strip()
        epoch_match = EPOCH_RE.match(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            current_epochs = int(epoch_match.group(2))
            continue
        step_match = TRAIN_STEP_RE.search(line)
        if not step_match:
            continue
        rows.append(
            {
                "epoch": current_epoch,
                "epochs": current_epochs,
                "step": int(step_match.group(1)),
                "steps": int(step_match.group(2)),
                "loss": float(step_match.group(3)),
            }
        )
    return rows


def _same_epoch_tail(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    latest_epoch = rows[-1].get("epoch")
    if latest_epoch is None:
        return []
    return [row for row in rows if row.get("epoch") == latest_epoch]


def _latest_epoch_marker(path: Path | None) -> dict[str, int]:
    if not path or not path.exists():
        return {}
    latest: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for raw in lines:
        match = EPOCH_RE.match(raw.strip())
        if match:
            latest = {"epoch": int(match.group(1)), "epochs": int(match.group(2))}
    return latest


def _latest_tqdm(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    latest: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in lines[-300:]:
        match = TQDM_RE.search(line.replace("\r", "\n"))
        if not match:
            continue
        latest = {
            "phase": match.group("phase"),
            "percent": int(match.group("pct")),
            "step": int(match.group("step")),
            "steps": int(match.group("steps")),
            "elapsed": match.group("elapsed").strip(),
            "eta": match.group("eta").strip(),
            "rate": match.group("rate").strip(),
        }
    return latest


def _active_errors(paths: list[Path | None]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        if not path or not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines[-500:], start=max(1, len(lines) - 499)):
            if ERROR_RE.search(line):
                out.append({"path": str(path), "line_number": idx, "line": line.strip()[:240]})
    return out


def _compact_epoch(row: dict[str, Any]) -> dict[str, Any]:
    if not row:
        return {}
    val = row.get("val_metrics") or {}
    train = row.get("train_metrics") or {}
    return {
        "epoch": row.get("epoch"),
        "epochs": row.get("epochs"),
        "saved_best": row.get("saved_best"),
        "train_loss": train.get("loss"),
        "train_acc_risk": train.get("acc_risk"),
        "train_mu_interval_coverage": train.get("mu_interval_coverage"),
        "train_mu_interval_width": train.get("mu_interval_width"),
        "val_loss": val.get("loss"),
        "val_acc_risk": val.get("acc_risk"),
        "val_acc_friction": val.get("acc_friction"),
        "val_acc_wetness": val.get("acc_wetness"),
        "val_mu_interval_coverage": val.get("mu_interval_coverage"),
        "val_mu_interval_width": val.get("mu_interval_width"),
    }


def _epoch_delta(latest: dict[str, Any], previous: dict[str, Any]) -> dict[str, float | None]:
    latest_c = _compact_epoch(latest)
    previous_c = _compact_epoch(previous)
    keys = [
        "val_loss",
        "val_acc_risk",
        "val_acc_friction",
        "val_acc_wetness",
        "val_mu_interval_coverage",
        "val_mu_interval_width",
    ]
    return {
        key: _float_or_none(latest_c.get(key)) - _float_or_none(previous_c.get(key))
        if _float_or_none(latest_c.get(key)) is not None and _float_or_none(previous_c.get(key)) is not None
        else None
        for key in keys
    }


def _loss_slope(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2:
        return None
    first = _float_or_none(rows[0].get("loss"))
    last = _float_or_none(rows[-1].get("loss"))
    if first is None or last is None:
        return None
    return last - first


def _prefer_int(*values: Any) -> int | None:
    for value in values:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    return "-" if number is None else f"{number:.4f}"


def _fmt_pct(value: Any) -> str:
    number = _float_or_none(value)
    return "-" if number is None else f"{number * 100:.2f}%"


def _fmt_delta(value: Any) -> str:
    number = _float_or_none(value)
    return "-" if number is None else f"{number:+.4f}"


def _fmt_delta_pct(value: Any) -> str:
    number = _float_or_none(value)
    return "-" if number is None else f"{number * 100:+.2f}%"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
