from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, default=None)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    log = args.log or run_dir / "safety_checkpoint_watcher.log"
    state_path = run_dir / "safety_checkpoint_watcher_state.json"
    best_path = run_dir / "best_safety.pt"
    last_path = run_dir / "last.pt"
    best_proxy = _checkpoint_proxy(best_path)
    last_mtime = None
    _log(log, f"watcher started run_dir={run_dir} wait_pid={args.wait_pid} best_proxy={best_proxy}")

    while True:
        if args.wait_pid is not None and not _pid_alive(args.wait_pid):
            _log(log, f"wait pid {args.wait_pid} exited; watcher stopping")
            break
        if last_path.exists():
            mtime = last_path.stat().st_mtime
            if last_mtime is None or mtime > last_mtime:
                last_mtime = mtime
                try:
                    info = _checkpoint_info(last_path)
                except Exception as exc:  # noqa: BLE001 - checkpoint may be mid-write.
                    _log(log, f"skip unreadable last checkpoint: {exc}")
                    time.sleep(min(max(int(args.poll_seconds), 10), 120))
                    continue
                proxy = float(info["safety_proxy"])
                if best_proxy is None or proxy > best_proxy:
                    tmp_path = best_path.with_suffix(".pt.tmp")
                    shutil.copy2(last_path, tmp_path)
                    tmp_path.replace(best_path)
                    best_proxy = proxy
                    info["source"] = str(last_path)
                    info["best_safety_path"] = str(best_path)
                    info["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    state_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
                    _log(log, f"updated best_safety epoch={info.get('epoch')} proxy={proxy:.6f}")
                else:
                    _log(log, f"checked last epoch={info.get('epoch')} proxy={proxy:.6f} best={best_proxy:.6f}")
        time.sleep(max(int(args.poll_seconds), 10))


def _checkpoint_proxy(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(_checkpoint_info(path)["safety_proxy"])
    except Exception:  # noqa: BLE001
        return None


def _checkpoint_info(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    metrics = ckpt.get("metrics", {})
    risk = float(metrics.get("acc_risk", 0.0) or 0.0)
    friction = float(metrics.get("acc_friction", 0.0) or 0.0)
    coverage = float(metrics.get("mu_interval_coverage", 0.0) or 0.0)
    width = float(metrics.get("mu_interval_width", 0.0) or 0.0)
    proxy = risk + 0.5 * friction + 0.5 * coverage - 0.1 * width
    return {
        "checkpoint": str(path),
        "epoch": ckpt.get("epoch"),
        "safety_proxy": proxy,
        "loss": metrics.get("loss"),
        "acc_risk": metrics.get("acc_risk"),
        "acc_friction": metrics.get("acc_friction"),
        "mu_interval_coverage": metrics.get("mu_interval_coverage"),
        "mu_interval_width": metrics.get("mu_interval_width"),
    }


def _pid_alive(pid: int) -> bool:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(0x1000, False, int(pid))
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return ctypes.get_last_error() == 5
        except Exception:
            return True
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
