from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_LOG = Path("reports/paper_protocol_summary/segmentation_transfer_promotion/promotion_log.json")
DEFAULT_RUN_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\segmentation_transfer")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "After an external segmentation audit finishes, promote a promising "
            "backend into a small cached-mask EvidenceField candidate config. "
            "This creates only a bounded candidate, not a formal paper-queue run."
        )
    )
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--backend", type=str, default="clipseg")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--samples-per-manifest", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-mode", type=str, default="bottom_square")
    parser.add_argument("--candidate-prefix", type=str, default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT / "road_mask_cache")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_RUN_ROOT / "road_mask_cache" / "manifests")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/experiments/segmentation_transfer"))
    parser.add_argument("--report-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    log = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wait_pid": args.wait_pid,
        "backend": args.backend,
        "steps": [],
    }
    _write_log(log, args.report_log)
    if args.wait_pid:
        _wait_for_pid(args.wait_pid, int(args.poll_seconds), log, args.report_log)

    audit = _load_json(args.audit_json)
    verdict = str((audit or {}).get("verdict", "missing"))
    log["audit_verdict"] = verdict
    if verdict in {"missing", "no_samples", "external_masks_unstable_do_not_full_preprocess", "blocked_missing_dependency"}:
        log["steps"].append(
            {
                "step": "skip_promotion",
                "reason": f"audit verdict `{verdict}` is not promotion-worthy",
            }
        )
        _write_log(log, args.report_log)
        return

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
        str(args.resize_mode),
        "--device",
        str(args.device),
        "--out-root",
        str(args.out_root),
        "--out-manifest-dir",
        str(args.manifest_dir),
        "--report-dir",
        str(Path("reports/paper_protocol_summary/external_road_mask_cache") / args.backend),
    ]
    if args.local_files_only:
        cache_cmd.append("--local-files-only")
    _run(cache_cmd, log)

    prefix = args.candidate_prefix or f"{args.backend}_mask_supervised_evidence_screen"
    make_cmd = [
        str(args.python),
        "scripts/make_mask_supervised_candidate_configs.py",
        "--cached-manifest-dir",
        str(args.manifest_dir),
        "--backend",
        args.backend,
        "--resize-mode",
        str(args.resize_mode),
        "--image-size",
        str(int(args.image_size)),
        "--prefix",
        prefix,
        "--out-dir",
        str(args.config_dir),
        "--run-root",
        str(DEFAULT_RUN_ROOT),
        "--max-train-samples",
        str(max(int(args.samples_per_manifest) * 3, 12)),
        "--max-val-samples",
        str(max(int(args.samples_per_manifest), 6)),
        "--max-test-samples",
        str(max(int(args.samples_per_manifest), 6)),
    ]
    _run(make_cmd, log)

    config_path = args.config_dir / f"{prefix}.yaml"
    audit_cmd = [
        str(args.python),
        "scripts/audit_segmentation_transfer_config.py",
        "--config",
        str(config_path),
        "--out-json",
        str(Path("reports/paper_protocol_summary") / f"segmentation_transfer_config_audit_{args.backend}.json"),
        "--out-md",
        str(Path("reports/paper_protocol_summary") / f"segmentation_transfer_config_audit_{args.backend}.md"),
    ]
    _run(audit_cmd, log)
    _run([str(args.python), "scripts/write_cv_transfer_decision_report.py"], log)
    _run([str(args.python), "scripts/topvenue_readiness_gate.py"], log)
    log["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_log(log, args.report_log)


def _wait_for_pid(pid: int, poll_seconds: int, log: dict, report_log: Path) -> None:
    while _pid_alive(pid):
        log["steps"].append(
            {
                "step": "wait",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "alive": [pid],
            }
        )
        _write_log(log, report_log)
        time.sleep(max(int(poll_seconds), 5))
    log["steps"].append(
        {
            "step": "wait",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "alive": [],
        }
    )
    _write_log(log, report_log)


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


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_log(log: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
