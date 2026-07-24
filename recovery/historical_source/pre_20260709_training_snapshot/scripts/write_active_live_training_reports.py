from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--run", default=None)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "active_live_training_reports.json")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "active_live_training_reports.md")
    args = parser.parse_args()

    report = build_report(args.summary_dir, args.log_dir, requested_run=args.run)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(summary_dir: Path, log_dir: Path, *, requested_run: str | None = None) -> dict[str, Any]:
    active = _find_active(summary_dir, requested_run)
    if not active.get("name"):
        return {
            "verdict": "idle",
            "active": active,
            "generated": [],
            "message": "No active run was found.",
        }

    name = str(active["name"])
    out_log = _resolve_log(log_dir, name, "out", active.get("out_log"))
    err_log = _resolve_log(log_dir, name, "err", active.get("err_log"))
    if out_log is None:
        return {
            "verdict": "missing_log",
            "active": active,
            "generated": [],
            "message": f"No out log was found for active run {name}.",
        }

    generated: list[str] = []
    trend_json = summary_dir / f"{name}_live_training_trend.json"
    trend_md = summary_dir / f"{name}_live_training_trend.md"
    trend_cmd = [
        sys.executable,
        "scripts/extract_training_log_metrics.py",
        "--log",
        str(out_log),
        "--run",
        name,
        "--out-json",
        str(trend_json),
        "--out-md",
        str(trend_md),
    ]
    if err_log is not None:
        trend_cmd.extend(["--err-log", str(err_log)])
    subprocess.run(trend_cmd, check=True)
    generated.extend([str(trend_json), str(trend_md)])
    trend = _load_json(trend_json) or {}
    active = _prefer_newer_active(active, trend.get("active_progress"))

    diagnosis_json = summary_dir / f"{name}_training_diagnosis.json"
    diagnosis_md = summary_dir / f"{name}_training_diagnosis.md"
    subprocess.run(
        [
            sys.executable,
            "scripts/write_live_training_diagnosis.py",
            "--run",
            name,
            "--summary-dir",
            str(summary_dir),
            "--out-json",
            str(diagnosis_json),
            "--out-md",
            str(diagnosis_md),
        ],
        check=True,
    )
    generated.extend([str(diagnosis_json), str(diagnosis_md)])

    return {
        "verdict": "updated",
        "active": {**active, "out_log": str(out_log), "err_log": str(err_log) if err_log else None},
        "generated": generated,
    }


def render_markdown(report: dict[str, Any]) -> str:
    active = report.get("active") or {}
    lines = [
        "# Active Live Training Reports",
        "",
        f"Verdict: `{report.get('verdict')}`",
        f"Active run: `{active.get('name') or '-'}`",
    ]
    if active.get("epoch") or active.get("step"):
        lines.append(
            "Progress: epoch `{epoch}/{epochs}`, step `{step}/{steps}`.".format(
                epoch=active.get("epoch") or "-",
                epochs=active.get("epochs") or "-",
                step=active.get("step") or "-",
                steps=active.get("steps") or "-",
            )
        )
    if active.get("phase") == "eval":
        lines.append(
            "Validation: step `{step}/{steps}`, ETA `{eta}`, rate `{rate}`.".format(
                step=active.get("eval_step") or "-",
                steps=active.get("eval_steps") or "-",
                eta=active.get("eval_tqdm_eta") or "-",
                rate=active.get("eval_tqdm_rate") or "-",
            )
        )
    if report.get("message"):
        lines.append(f"Message: {report['message']}")
    generated = report.get("generated") or []
    if generated:
        lines.extend(["", "## Generated", ""])
        for path in generated:
            lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def _find_active(summary_dir: Path, requested_run: str | None) -> dict[str, Any]:
    if requested_run:
        return {"name": requested_run}

    process_active = _active_training_from_processes()
    if process_active:
        return process_active

    watch = _load_json(summary_dir / "active_training_watch_report.json") or {}
    watch_active = watch.get("active") if isinstance(watch, dict) else {}
    if isinstance(watch_active, dict) and watch_active.get("name"):
        return {
            "name": watch_active.get("name"),
            "epoch": watch_active.get("epoch"),
            "epochs": watch_active.get("epochs"),
            "step": watch_active.get("step"),
            "steps": watch_active.get("steps"),
            "out_log": watch_active.get("out_log"),
            "err_log": watch_active.get("err_log"),
        }

    queue = _load_json(summary_dir / "queue_recovery_report.json") or {}
    for row in queue.get("active_rows", []) or []:
        if row.get("name"):
            return {
                "name": row.get("name"),
                "epoch": row.get("active_epoch"),
                "epochs": row.get("active_epochs"),
                "step": row.get("active_step"),
                "steps": row.get("active_steps"),
                "out_log": row.get("active_log"),
            }

    dashboard = _load_json(summary_dir / "experiment_status_dashboard.json") or {}
    for row in dashboard.get("active_rows", []) or []:
        if row.get("name"):
            return {
                "name": row.get("name"),
                "epoch": row.get("active_epoch"),
                "epochs": row.get("active_epochs"),
                "step": row.get("active_step"),
                "steps": row.get("active_steps"),
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
            "phase": "train",
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


def _resolve_log(log_dir: Path, run_name: str, suffix: str, preferred: Any) -> Path | None:
    if preferred:
        path = Path(str(preferred))
        if path.exists() and path.name.endswith(f".{suffix}.log"):
            return path
        if path.exists() and suffix == "out":
            return path
    candidates = sorted(
        log_dir.glob(f"{run_name}_*.{suffix}.log"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _prefer_newer_active(current: dict[str, Any], parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return current
    out = dict(current)
    current_epoch = _as_int(out.get("epoch"))
    parsed_epoch = _as_int(parsed.get("epoch"))
    current_step = _as_int(out.get("step"))
    parsed_step = _as_int(parsed.get("step"))
    if (
        current_epoch is None
        or (parsed_epoch is not None and parsed_epoch > current_epoch)
        or (parsed_epoch == current_epoch and parsed_step is not None and (current_step is None or parsed_step >= current_step))
    ):
        for key in [
            "epoch",
            "epochs",
            "step",
            "steps",
            "phase",
            "tqdm_eta",
            "tqdm_rate",
            "tqdm_percent",
            "eval_step",
            "eval_steps",
            "eval_tqdm_eta",
            "eval_tqdm_rate",
            "eval_tqdm_percent",
        ]:
            if key in parsed:
                out[key] = parsed.get(key)
    return out


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
