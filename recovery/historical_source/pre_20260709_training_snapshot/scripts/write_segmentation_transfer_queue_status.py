from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize background segmentation-transfer audit/promotion watchers."
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_SUMMARY / "segmentation_transfer_queue_status.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_SUMMARY / "segmentation_transfer_queue_status.md",
    )
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(summary_dir: Path) -> dict[str, Any]:
    after_queue = _load_json(
        summary_dir / "external_segmentation_after_queue_clipseg" / "external_segmentation_after_queue.json"
    ) or {}
    promotion = _load_json(
        summary_dir / "segmentation_transfer_promotion" / "promotion_clipseg_log.json"
    ) or {}
    after_wait = after_queue.get("wait_pids") or []
    promotion_wait = promotion.get("wait_pid")
    pids = [int(pid) for pid in after_wait if str(pid).isdigit()]
    if promotion_wait and str(promotion_wait).isdigit():
        pids.append(int(promotion_wait))
    process_rows = [_process_row(pid) for pid in pids]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "claim_boundary": (
            "This report tracks background automation only. It is not performance "
            "evidence and does not imply that external segmentation masks are useful."
        ),
        "after_queue": {
            "backend": after_queue.get("backend"),
            "device": after_queue.get("device"),
            "wait_pids": after_wait,
            "last_step": _last_step(after_queue),
        },
        "promotion": {
            "backend": promotion.get("backend"),
            "wait_pid": promotion_wait,
            "audit_verdict": promotion.get("audit_verdict"),
            "last_step": _last_step(promotion),
        },
        "process_rows": process_rows,
        "safety_contract": [
            "External segmentation audit waits for the full formal queue orchestrator before installing transformers.",
            "Audit device is CPU, so it should not compete for RTX 3050 VRAM.",
            "Promotion creates only bounded candidate configs/caches and does not enter the formal paper queue automatically.",
        ],
    }


def _process_row(pid: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                f"Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue | "
                "Select-Object Id,ProcessName,CPU,WorkingSet64,StartTime | ConvertTo-Json -Compress"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    text = result.stdout.strip()
    if not text:
        return {"pid": pid, "alive": False}
    try:
        data = json.loads(text)
        if isinstance(data, list):
            data = data[0] if data else {}
    except json.JSONDecodeError:
        data = {}
    return {
        "pid": pid,
        "alive": bool(data),
        "process_name": data.get("ProcessName"),
        "cpu": data.get("CPU"),
        "working_set_mb": round(float(data.get("WorkingSet64", 0) or 0) / (1024 * 1024), 2),
        "start_time": data.get("StartTime"),
    }


def _last_step(report: dict[str, Any]) -> dict[str, Any]:
    steps = report.get("steps", []) if isinstance(report.get("steps"), list) else []
    return steps[-1] if steps else {}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Segmentation Transfer Queue Status",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Watchers",
        "",
        "| Watcher | Backend | Wait PID(s) | Last step |",
        "|---|---|---|---|",
        "| External mask audit | `{backend}` | `{wait}` | `{last}` |".format(
            backend=report["after_queue"].get("backend"),
            wait=report["after_queue"].get("wait_pids"),
            last=report["after_queue"].get("last_step"),
        ),
        "| Promotion | `{backend}` | `{wait}` | `{last}` |".format(
            backend=report["promotion"].get("backend"),
            wait=report["promotion"].get("wait_pid"),
            last=report["promotion"].get("last_step"),
        ),
        "",
        "## Processes",
        "",
        "| PID | Alive | Process | CPU | Working set MB |",
        "|---:|---|---|---:|---:|",
    ]
    for row in report.get("process_rows", []):
        lines.append(
            "| {pid} | {alive} | {process} | {cpu} | {mem} |".format(
                pid=row.get("pid"),
                alive=row.get("alive"),
                process=row.get("process_name", "-"),
                cpu=_fmt(row.get("cpu")),
                mem=_fmt(row.get("working_set_mb")),
            )
        )
    lines.extend(["", "## Safety Contract", ""])
    lines.extend(f"- {item}" for item in report.get("safety_contract", []))
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    main()
