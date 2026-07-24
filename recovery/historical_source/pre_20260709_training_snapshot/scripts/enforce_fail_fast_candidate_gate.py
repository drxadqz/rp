from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(r"D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe")
DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "reports" / "paper_protocol_summary"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "paper_protocol_queue"
DEFAULT_FAST_LOG_DIR = PROJECT_ROOT / "outputs" / "fast_screen_queue"

CANDIDATE_STEMS = {
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
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--fast-log-dir", type=Path, default=DEFAULT_FAST_LOG_DIR)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.fast_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "fail_fast_candidate_gate.log"
    _log(log_path, f"starting fail-fast candidate gate for pid={args.watch_pid}")

    while True:
        allowed = _allowed_formal_candidates(args.summary_dir)
        excluded = sorted(CANDIDATE_STEMS - allowed)
        procs = _process_snapshot()
        active_candidate = _active_candidate_train(procs)
        old_alive = any(proc.get("pid") == args.watch_pid for proc in procs)
        if active_candidate:
            stem = active_candidate["stem"]
            pid = int(active_candidate["pid"])
            _log(log_path, f"observed candidate train stem={stem} pid={pid}; allowed={sorted(allowed)}")
            if stem in excluded:
                _log(log_path, f"gating excluded candidate {stem}; switching to fast-screen route")
                if not args.dry_run:
                    _stop_pid(pid, log_path)
                    _stop_pid(args.watch_pid, log_path)
                    _wait_until_gone(pid, log_path)
                    _wait_until_gone(args.watch_pid, log_path)
                    _run_fail_fast_route(args, log_path)
                else:
                    _log(log_path, "dry-run: would stop excluded candidate and old orchestrator")
                return
        if not old_alive:
            _log(log_path, "watched orchestrator is gone; no gate action needed")
            return
        if args.once:
            _log(log_path, "once mode complete; no gate action taken")
            return
        time.sleep(max(1.0, float(args.poll_seconds)))


def _allowed_formal_candidates(summary_dir: Path) -> set[str]:
    report = _load_json(summary_dir / "fail_fast_exploration_report.json") or {}
    policy = report.get("formal_policy") if isinstance(report, dict) else {}
    selected = (policy or {}).get("promoted_or_fallback")
    if not isinstance(selected, list):
        return set()
    out = {str(item) for item in selected if str(item) in CANDIDATE_STEMS}
    return out


def _active_candidate_train(procs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for proc in procs:
        cmd = str(proc.get("command") or "")
        if "scripts/train.py" not in cmd.replace("\\", "/"):
            continue
        for stem in CANDIDATE_STEMS:
            if f"{stem}.yaml" in cmd:
                return {**proc, "stem": stem}
    return None


def _process_snapshot() -> list[dict[str, Any]]:
    if sys.platform != "win32":
        return []
    command = r"""
$rows = Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'run_paper_protocol_direct.py|scripts/train.py|scripts\\train.py|run_fast_screen_protocol.py' } |
  ForEach-Object {
    [PSCustomObject]@{
      pid = $_.ProcessId
      parent = $_.ParentProcessId
      name = $_.Name
      command = $_.CommandLine
    }
  }
$rows | ConvertTo-Json -Compress
"""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "pid": int(row.get("pid")),
                "parent": row.get("parent"),
                "name": row.get("name"),
                "command": row.get("command") or "",
            }
        )
    return out


def _stop_pid(pid: int, log_path: Path) -> None:
    _log(log_path, f"stopping pid={pid}")
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {int(pid)} -Force -ErrorAction SilentlyContinue"],
        cwd=PROJECT_ROOT,
        check=False,
    )


def _wait_until_gone(pid: int, log_path: Path, *, timeout_s: float = 120.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if not any(proc.get("pid") == pid for proc in _process_snapshot()):
            _log(log_path, f"pid={pid} is gone")
            return
        time.sleep(2.0)
    _log(log_path, f"timeout waiting for pid={pid} to exit")


def _run_fail_fast_route(args: argparse.Namespace, log_path: Path) -> None:
    _run(
        [
            str(args.python),
            "scripts/postprocess_protocol_outputs.py",
            "--root",
            str(args.root),
            "--summary-dir",
            str(args.summary_dir),
        ],
        args.log_dir,
        "gate_postprocess_before_fast_screen",
        log_path,
    )
    _run(
        [
            str(args.python),
            "scripts/run_fast_screen_protocol.py",
            "--scope",
            "candidates",
            "--log-dir",
            str(args.fast_log_dir),
            "--lean-first-wave",
            "--bootstrap-samples",
            "100",
            "--dataset-diagnostic-samples",
            "2000",
            "--evidence-map-samples",
            "12",
            "--evidence-audit-samples",
            "1000",
        ],
        args.fast_log_dir,
        "gate_fast_screen_candidates",
        log_path,
    )
    _run(
        [
            str(args.python),
            "scripts/write_fast_screen_status_report.py",
            "--log-dir",
            str(args.fast_log_dir),
        ],
        args.fast_log_dir,
        "gate_write_fast_screen_status",
        log_path,
    )
    _run(
        [
            str(args.python),
            "scripts/write_fast_to_formal_promotion_report.py",
            "--summary-dir",
            str(args.summary_dir),
        ],
        args.log_dir,
        "gate_write_fast_to_formal_promotion",
        log_path,
    )
    _run(
        [
            str(args.python),
            "scripts/write_fail_fast_exploration_report.py",
            "--summary-dir",
            str(args.summary_dir),
        ],
        args.log_dir,
        "gate_write_fail_fast_report",
        log_path,
    )
    _run(
        [
            str(args.python),
            "scripts/run_paper_protocol_direct.py",
            "--phase",
            "candidates",
            "--candidate-policy",
            "fail_fast",
            "--python",
            str(args.python),
            "--root",
            str(args.root),
            "--log-dir",
            str(args.log_dir),
            "--postprocess-each",
        ],
        args.log_dir,
        "gate_formal_fail_fast_candidates",
        log_path,
    )


def _run(cmd: list[str], log_dir: Path, name: str, gate_log: Path) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = log_dir / f"{name}_{stamp}.out.log"
    err = log_dir / f"{name}_{stamp}.err.log"
    _log(gate_log, f"RUN {name}: {' '.join(cmd)}")
    with out.open("w", encoding="utf-8", errors="replace") as fout, err.open(
        "w", encoding="utf-8", errors="replace"
    ) as ferr:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fout, stderr=ferr)
    if proc.returncode != 0:
        _log(gate_log, f"{name} failed exit={proc.returncode} out={out} err={err}")
        raise RuntimeError(f"{name} failed exit={proc.returncode}")
    _log(gate_log, f"{name} completed")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return raw if isinstance(raw, dict) else None


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
