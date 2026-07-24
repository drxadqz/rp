from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("E:/perception_outputs/rscd_surface_classification")
DEFAULTS = {
    "S7_full": ROOT / "c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709",
    "S96_cap250": ROOT / "c3_farnet_screen_s96_wc_pair_relative_boundary_20260712",
    "S133c_full": ROOT / "c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715",
    "S135c_screen": ROOT / "c3_farnet_screen_s135c_s96_wc_moderate_film_rough_focus_stem_20260715",
    "S135c_full": ROOT / "c3_farnet_formal_fullmanifest_s135c_s96_wc_moderate_film_rough_focus_stem_20260715",
}
S133C_STDERR = DEFAULTS["S133c_full"] / "train_stderr_20260715_182019.log"
S135C_HANDOFF = DEFAULTS["S135c_screen"] / "handoff_after_s133c.log"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _tail(path: Path, n: int = 20) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def _parse_progress(lines: list[str]) -> dict[str, Any]:
    pattern = re.compile(
        r"train:\s*(?P<pct>\d+)%.*?\|\s*(?P<step>\d+)\s*/\s*(?P<total>\d+)\s*"
        r"\[(?P<elapsed>[^<\]]+)<(?P<eta>[^,\]]+),\s*(?P<speed>[^\]]+)\]"
    )
    for line in reversed(lines):
        match = pattern.search(line)
        if not match:
            continue
        step = int(match.group("step"))
        total = int(match.group("total"))
        return {
            "line": line,
            "percent": int(match.group("pct")),
            "step": step,
            "total": total,
            "fraction": step / max(total, 1),
            "elapsed": match.group("elapsed").strip(),
            "eta": match.group("eta").strip(),
            "speed": match.group("speed").strip(),
        }
    return {"line": lines[-1] if lines else "", "percent": None, "step": None, "total": None}


def _file_freshness(path: Path, *, stale_after_seconds: int = 600) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "modified": None,
            "age_seconds": None,
            "stale_after_seconds": stale_after_seconds,
            "is_stale": True,
        }
    modified_ts = path.stat().st_mtime
    modified = datetime.fromtimestamp(modified_ts)
    age_seconds = max(0.0, (datetime.now() - modified).total_seconds())
    return {
        "path": str(path),
        "exists": True,
        "modified": modified.isoformat(timespec="seconds"),
        "age_seconds": round(age_seconds, 1),
        "stale_after_seconds": stale_after_seconds,
        "is_stale": age_seconds > stale_after_seconds,
    }


def _powershell(command: str) -> str:
    try:
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _process_status() -> dict[str, Any]:
    python_cmd = (
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*train.py*' -and $_.CommandLine -like '*c3_farnet*' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    watcher_cmd = (
        "Get-CimInstance Win32_Process -Filter \"name='powershell.exe'\" | "
        "Where-Object { $_.CommandLine -like '*run_s135_after_s133c.ps1*' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    return {
        "training_processes_json": _powershell(python_cmd),
        "watcher_processes_json": _powershell(watcher_cmd),
    }


def _gpu_status() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


def _disk_status() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for drive in ["C:/", "D:/", "E:/"]:
        try:
            usage = shutil.disk_usage(drive)
            out[drive[0]] = {
                "free_gb": round(usage.free / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "total_gb": round(usage.total / (1024**3), 2),
            }
        except Exception:
            continue
    return out


def _run_status(name: str, path: Path) -> dict[str, Any]:
    metrics = _read_json(path / "test_metrics.json")
    summary = (metrics or {}).get("summary", {}) if metrics else {}
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "has_test_metrics": metrics is not None,
        "top1": summary.get("top1"),
        "macro_f1": summary.get("macro_f1"),
        "num_samples": summary.get("num_samples"),
        "num_errors": summary.get("num_errors"),
        "has_best_checkpoint": (path / "best_checkpoint.pth").exists(),
        "has_predictions": (path / "predictions_test.csv").exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a live RSCD experiment pipeline status snapshot.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "comparison_live_20260715")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress = _parse_progress(_tail(S133C_STDERR, 30))
    log_freshness = _file_freshness(S133C_STDERR)
    status = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runs": [_run_status(name, path) for name, path in DEFAULTS.items()],
        "s133c_progress": progress,
        "s133c_log_freshness": log_freshness,
        "s133c_log_tail": _tail(S133C_STDERR, 8),
        "handoff_tail": _tail(S135C_HANDOFF, 12),
        "processes": _process_status(),
        "gpu": _gpu_status(),
        "disk": _disk_status(),
    }
    json_path = args.output_dir / "pipeline_status.json"
    json_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# RSCD Pipeline Status",
        "",
        f"- Time: `{status['timestamp']}`",
        f"- S133c progress: `{progress.get('percent')}%`, step `{progress.get('step')}/{progress.get('total')}`, ETA `{progress.get('eta')}`",
        f"- S133c log age: `{log_freshness.get('age_seconds')}s`, stale: `{log_freshness.get('is_stale')}`",
        f"- GPU: `{status['gpu']}`",
        "",
        "## Runs",
        "",
        "| Run | Metrics | Samples | Top-1 | Macro-F1 | Best ckpt | Predictions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in status["runs"]:
        top1 = "-" if run["top1"] is None else f"{100.0 * float(run['top1']):.3f}%"
        macro = "-" if run["macro_f1"] is None else f"{100.0 * float(run['macro_f1']):.3f}%"
        md.append(
            f"| {run['name']} | {run['has_test_metrics']} | {run['num_samples'] or '-'} | "
            f"{top1} | {macro} | {run['has_best_checkpoint']} | {run['has_predictions']} |"
        )
    md.extend(["", "## Disk", "", "| Drive | Free GB | Used GB | Total GB |", "|---|---:|---:|---:|"])
    for name, item in status["disk"].items():
        md.append(f"| {name} | {item['free_gb']} | {item['used_gb']} | {item['total_gb']} |")
    md.extend(["", "## Recent S133c Log", ""])
    md.extend(f"- `{line}`" for line in status["s133c_log_tail"])
    md.extend(["", "## Handoff Log", ""])
    md.extend(f"- `{line}`" for line in status["handoff_tail"])

    md_path = args.output_dir / "pipeline_status.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
