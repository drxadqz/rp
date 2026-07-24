from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_REPORT_DIR = Path("reports/paper_protocol_summary/external_segmentation_after_queue")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for active formal-training processes to finish, then run a "
            "small external-segmentation audit and optional road-mask cache job. "
            "This keeps SAM/CLIPSeg/SegFormer work out of the GPU queue until "
            "the current paper protocol is idle."
        )
    )
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--backend", choices=["opencv", "clipseg", "segformer", "sam"], default="clipseg")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--samples-per-dataset", type=int, default=34)
    parser.add_argument("--samples-per-manifest", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--install-transformers", action="store_true")
    parser.add_argument("--install-segment-anything", action="store_true")
    parser.add_argument("--run-cache", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    log: dict = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wait_pids": args.wait_pid,
        "backend": args.backend,
        "device": args.device,
        "steps": [],
    }
    _write_log(log, args.report_dir)

    _wait_for_pids(args.wait_pid, int(args.poll_seconds), log, args.report_dir)
    if args.install_transformers:
        _run([str(args.python), "-m", "pip", "install", "transformers", "accelerate", "safetensors"], log)
    if args.install_segment_anything:
        _run([str(args.python), "-m", "pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"], log)

    audit_cmd = [
        str(args.python),
        "scripts/audit_external_segmentation_masks.py",
        "--backend",
        args.backend,
        "--samples-per-dataset",
        str(int(args.samples_per_dataset)),
        "--overlays-per-dataset",
        "8",
        "--image-size",
        str(int(args.image_size)),
        "--device",
        args.device,
        "--out-dir",
        str(Path("reports/paper_protocol_summary/external_segmentation_masks") / args.backend),
        "--out-json",
        str(Path("reports/paper_protocol_summary/external_segmentation_masks") / f"external_segmentation_mask_audit_{args.backend}.json"),
        "--out-md",
        str(Path("reports/paper_protocol_summary/external_segmentation_masks") / f"external_segmentation_mask_audit_{args.backend}.md"),
    ]
    if args.local_files_only:
        audit_cmd.append("--local-files-only")
    if args.sam_checkpoint is not None:
        audit_cmd.extend(["--sam-checkpoint", str(args.sam_checkpoint)])
    _run(audit_cmd, log)

    if args.run_cache:
        cache_cmd = [
            str(args.python),
            "scripts/cache_external_road_masks.py",
            "--backend",
            args.backend,
            "--samples-per-manifest",
            str(int(args.samples_per_manifest)),
            "--image-size",
            str(int(args.image_size)),
            "--resize-mode",
            "bottom_square",
            "--device",
            args.device,
        ]
        if args.local_files_only:
            cache_cmd.append("--local-files-only")
        if args.sam_checkpoint is not None:
            cache_cmd.extend(["--sam-checkpoint", str(args.sam_checkpoint)])
        _run(cache_cmd, log)

    _run([str(args.python), "scripts/write_cv_transfer_decision_report.py"], log)
    _run([str(args.python), "scripts/topvenue_readiness_gate.py"], log)
    log["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_log(log, args.report_dir)


def _wait_for_pids(pids: list[int], poll_seconds: int, log: dict, report_dir: Path) -> None:
    remaining = [pid for pid in pids if pid > 0]
    if not remaining:
        log["steps"].append({"step": "wait", "status": "skipped_no_pids"})
        _write_log(log, report_dir)
        return
    while remaining:
        alive = [pid for pid in remaining if _pid_alive(pid)]
        log["steps"].append(
            {
                "step": "wait",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "alive": alive,
            }
        )
        _write_log(log, report_dir)
        if not alive:
            return
        remaining = alive
        time.sleep(max(int(poll_seconds), 5))


def _pid_alive(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {int(pid)}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return str(int(pid)) in result.stdout


def _run(cmd: list[str], log: dict) -> None:
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log["steps"].append(
        {
            "step": "run",
            "started_at": started,
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "returncode": int(result.returncode),
            "command": cmd,
            "output_tail": result.stdout[-4000:],
        }
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _write_log(log: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "external_segmentation_after_queue.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
