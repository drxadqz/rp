from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


RUN_DIR = Path(
    "E:/perception_outputs/rscd_surface_classification/"
    "c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709"
)
PID = 13808


PROGRESS_RE = re.compile(
    r"(?P<stage>eval|train):\s+(?P<pct>\d+)%\|.*?\|\s+"
    r"(?P<done>\d+)/(?P<total>\d+)\s+\[(?P<elapsed>[^<\]]+)(?:<(?P<eta>[^,\]]+))?"
)


def last_progress() -> dict[str, object] | None:
    stderr_files = sorted(RUN_DIR.glob("train_stderr_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not stderr_files:
        return None
    text = stderr_files[0].read_text(encoding="utf-8", errors="ignore")
    matches = list(PROGRESS_RE.finditer(text.replace("\r", "\n")))
    if not matches:
        return None
    m = matches[-1]
    done = int(m.group("done"))
    total = int(m.group("total"))
    return {
        "stage": m.group("stage"),
        "pct": int(m.group("pct")),
        "done": done,
        "total": total,
        "fraction": done / total if total else 0.0,
        "elapsed": m.group("elapsed"),
        "eta": m.group("eta") or "",
        "stderr": str(stderr_files[0]),
    }


def process_alive() -> bool:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {PID} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return str(PID) in result.stdout
    except Exception:
        return False


def gpu_status() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main() -> None:
    status = {
        "pid": PID,
        "alive": process_alive(),
        "gpu": gpu_status(),
        "run_dir": str(RUN_DIR),
        "history_exists": (RUN_DIR / "history.json").exists(),
        "best_checkpoint_exists": (RUN_DIR / "best_checkpoint.pth").exists(),
        "metrics_exists": (RUN_DIR / "metrics.json").exists(),
        "progress": last_progress(),
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
