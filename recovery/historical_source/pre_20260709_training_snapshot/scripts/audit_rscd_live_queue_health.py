from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMPARISON_DIR = Path(r"E:\perception_outputs\rscd_surface_classification\comparison_live_20260715")
DEFAULT_LIVE_STATUS_DIR = DEFAULT_COMPARISON_DIR / "live_route_status_20260715"
DEFAULT_OUTPUT_DIR = DEFAULT_COMPARISON_DIR / "queue_health_20260716"
DEFAULT_EXPERIMENT_BOARD_DIR = DEFAULT_COMPARISON_DIR / "experiment_board_20260716"
DEFAULT_SCREEN_PLATEAU_DIR = DEFAULT_COMPARISON_DIR / "screen_plateau_diagnosis_20260716"
DEFAULT_PROMOTION_GATE_DIR = DEFAULT_COMPARISON_DIR / "promotion_gate_regression_20260716"
DEFAULT_SOTA_COMPLETION_DIR = DEFAULT_COMPARISON_DIR / "sota_completion_evidence_20260716"
DEFAULT_S133C_LOG = Path(
    r"E:\perception_outputs\rscd_surface_classification"
    r"\c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715"
    r"\train_stderr_20260715_182019.log"
)

EXPECTED_WATCHERS = {
    "S133c_resume_watchdog": "watch_s133c_resume_if_needed.ps1",
    "S135_after_S133c": "run_s135_after_s133c.ps1",
    "S136_after_S135": "run_s136_after_s135_if_needed.ps1",
    "S136d_after_S136": "run_s136d_after_s136_if_needed.ps1",
    "S137_after_current_queue": "run_s137_after_current_queue_if_needed.ps1",
    "S138_after_S137": "run_s138_after_s137_if_needed.ps1",
}

READINESS_REPORTS = {
    "S136_after_rgb_fix": DEFAULT_COMPARISON_DIR
    / "S136_queue_readiness_after_rgb_fix_20260715"
    / "s136_queue_readiness.json",
    "S136d_after_rgb_fix": DEFAULT_COMPARISON_DIR
    / "S136d_queue_readiness_after_rgb_fix_20260715"
    / "s136d_queue_readiness.json",
    "S137_after_rgb_fix": DEFAULT_COMPARISON_DIR
    / "S137_queue_readiness_after_rgb_fix_20260715"
    / "s137_queue_readiness.json",
    "S138_dual_film_texture": DEFAULT_COMPARISON_DIR
    / "S138_queue_readiness_20260716"
    / "s138_queue_readiness.json",
}

REQUIRED_READINESS_CHECKS = {
    "S136_after_rgb_fix": [
        "full_manifest_rows_complete",
        "control_protocol_matches_screen",
        "control_disables_adaptive_gate",
    ],
    "S136d_after_rgb_fix": [
        "full_manifest_rows_complete",
        "teacher_full_manifest_rows_complete",
        "nodistill_control_protocol_matches_screen",
    ],
    "S137_after_rgb_fix": [
        "full_manifest_rows_complete",
        "screen_control_same_param_count",
        "watcher_has_fair_gates",
    ],
    "S138_dual_film_texture": [
        "full_manifest_rows_complete",
        "screen_control_same_param_count",
        "screen_control_only_intended_diffs",
    ],
}


@dataclass
class Check:
    name: str
    severity: str
    passed: bool
    message: str
    details: dict[str, Any]


def add_check(
    checks: list[Check],
    name: str,
    severity: str,
    passed: bool,
    message: str,
    **details: Any,
) -> None:
    checks.append(Check(name=name, severity=severity, passed=bool(passed), message=message, details=details))


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_powershell_json(command: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return [{"_error": completed.stderr.strip(), "_returncode": completed.returncode}]
    text = completed.stdout.strip()
    if not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, dict):
        return [payload]
    return list(payload)


def get_training_processes() -> list[dict[str, Any]]:
    command = (
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*train.py*' -or "
        "$_.CommandLine -like '*train_coupled_factor_backbone*' -or "
        "$_.CommandLine -like '*cache_teacher_logits*' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 3"
    )
    return run_powershell_json(command)


def get_watcher_processes() -> list[dict[str, Any]]:
    watcher_filter = " -or ".join(
        f"$_.CommandLine -like '*{script}*'" for script in EXPECTED_WATCHERS.values()
    )
    command = (
        "Get-CimInstance Win32_Process -Filter \"name='powershell.exe'\" | "
        f"Where-Object {{ ({watcher_filter}) -and $_.CommandLine -notlike '*Get-CimInstance*' }} | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 3"
    )
    return run_powershell_json(command)


def get_heavy_analysis_processes() -> list[dict[str, Any]]:
    command = (
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*analyze_high_error_feature_values.py*' -or "
        "$_.CommandLine -like '*analyze_rscd_physics_cues.py*' -or "
        "$_.CommandLine -like '*analyze_rscd_feature_graph_patterns.py*' -or "
        "$_.CommandLine -like '*analyze_rscd_complete_graph_patterns.py*' } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 3"
    )
    return run_powershell_json(command)


def parse_log_progress(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"available": False, "log": str(log_path)}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"train:\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)\s+\[([^\]]+)\]", text)
    age_seconds = max(datetime.now().timestamp() - log_path.stat().st_mtime, 0.0)
    if not matches:
        return {"available": True, "progress_found": False, "log": str(log_path), "age_seconds": age_seconds}
    percent_s, step_s, total_s, elapsed = matches[-1]
    step = int(step_s)
    total = int(total_s)
    return {
        "available": True,
        "progress_found": True,
        "log": str(log_path),
        "percent": int(percent_s),
        "step": step,
        "total_steps": total,
        "remaining_steps": max(total - step, 0),
        "elapsed": elapsed,
        "age_seconds": age_seconds,
        "last_write": datetime.fromtimestamp(log_path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def inspect_step_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - defensive runtime audit
        return {
            "exists": True,
            "path": str(path),
            "readable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "size_bytes": path.stat().st_size,
            "last_write": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        }
    model = state.get("model")
    optimizer = state.get("optimizer")
    scaler = state.get("scaler")
    train_partial = state.get("train_partial")
    return {
        "exists": True,
        "path": str(path),
        "readable": True,
        "size_bytes": path.stat().st_size,
        "last_write": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "epoch": state.get("epoch"),
        "step": state.get("step"),
        "total_steps": state.get("total_steps"),
        "model_items": len(model) if isinstance(model, dict) else None,
        "has_optimizer": isinstance(optimizer, dict) and bool(optimizer.get("param_groups")),
        "has_scaler": isinstance(scaler, dict) and "scale" in scaler,
        "has_train_partial": isinstance(train_partial, dict),
        "config_output_dir": (state.get("config") or {}).get("output_dir") if isinstance(state.get("config"), dict) else None,
    }


def refresh_live_status(output_dir: Path, s133c_log: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/write_rscd_live_route_status.py",
            "--output-dir",
            str(output_dir),
            "--s133c-log",
            str(s133c_log),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "json_path": str(output_dir / "rscd_live_route_status.json"),
        "md_path": str(output_dir / "rscd_live_route_status.md"),
    }


def refresh_experiment_board(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_rscd_experiment_board.py",
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "json_path": str(output_dir / "rscd_experiment_board.json"),
        "md_path": str(output_dir / "rscd_experiment_board.md"),
    }


def refresh_screen_plateau_diagnosis(board_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_rscd_screen_plateaus.py",
            "--board-csv",
            str(board_dir / "rscd_experiment_board.csv"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "json_path": str(output_dir / "screen_plateau_diagnosis.json"),
        "md_path": str(output_dir / "screen_plateau_diagnosis.md"),
    }


def refresh_promotion_gate_regression(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_promotion_gate_regression.py",
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "json_path": str(output_dir / "promotion_gate_regression.json"),
        "md_path": str(output_dir / "promotion_gate_regression.md"),
    }


def refresh_sota_completion_evidence(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/write_rscd_sota_completion_evidence.py",
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    payload = read_json(output_dir / "rscd_sota_completion_evidence.json") or {}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "verdict": payload.get("verdict"),
        "json_path": str(output_dir / "rscd_sota_completion_evidence.json"),
        "md_path": str(output_dir / "rscd_sota_completion_evidence.md"),
    }


def disk_snapshot(paths: list[Path]) -> dict[str, dict[str, float]]:
    roots: dict[str, dict[str, float]] = {}
    candidates = list(paths)
    if sys.platform.startswith("win"):
        candidates.extend(Path(f"{letter}:\\") for letter in "CDEFG")
    for path in candidates:
        root = Path(path.anchor)
        if not root.anchor:
            continue
        try:
            usage = shutil.disk_usage(str(root))
        except FileNotFoundError:
            continue
        roots[root.anchor] = {
            "free_gb": round(usage.free / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "total_gb": round(usage.total / (1024**3), 2),
        }
    return roots


def readiness_ok(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    if "ok" in payload:
        return bool(payload["ok"])
    if "overall" in payload:
        return str(payload["overall"]).lower() == "pass"
    return False


def readiness_check_names(payload: dict[str, Any] | None) -> set[str]:
    if payload is None:
        return set()
    checks = payload.get("checks") or []
    names = set()
    for item in checks:
        if isinstance(item, dict) and item.get("name"):
            names.add(str(item["name"]))
    return names


def inspect_watcher_script(script: Path, required_tokens: list[str]) -> dict[str, Any]:
    if not script.exists():
        return {"exists": False, "missing": required_tokens}
    text = script.read_text(encoding="utf-8", errors="ignore")
    missing = [token for token in required_tokens if token not in text]
    return {"exists": True, "missing": missing}


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# RSCD Live Queue Health Audit",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Overall: `{payload['overall']}`",
        f"- Live decision: `{payload.get('live_decision', {}).get('action', '-')}`",
        "",
        "## Checks",
        "",
        "| Severity | Pass | Check | Message |",
        "|---|---:|---|---|",
    ]
    for item in payload["checks"]:
        lines.append(f"| {item['severity']} | {item['passed']} | `{item['name']}` | {item['message']} |")

    lines.extend(["", "## Active Training", ""])
    for proc in payload["training_processes"]:
        lines.append(f"- PID `{proc.get('ProcessId', '-')}`: `{proc.get('CommandLine', '')}`")
    progress = payload["s133c_progress"]
    if progress.get("progress_found"):
        lines.append(
            "- S133c progress: `{step}/{total}` (`{percent}%`), log age `{age:.1f}s`.".format(
                step=progress["step"],
                total=progress["total_steps"],
                percent=progress["percent"],
                age=progress.get("age_seconds", -1.0),
            )
        )
    step_checkpoint = payload.get("s133c_step_checkpoint") or {}
    if step_checkpoint.get("exists"):
        lines.append(
            "- Step checkpoint: step `{step}/{total}`, readable `{readable}`, size `{size}` bytes, written `{written}`.".format(
                step=step_checkpoint.get("step", "-"),
                total=step_checkpoint.get("total_steps", "-"),
                readable=step_checkpoint.get("readable", False),
                size=step_checkpoint.get("size_bytes", "-"),
                written=step_checkpoint.get("last_write", "-"),
            )
        )
    else:
        lines.append(f"- Step checkpoint: missing at `{step_checkpoint.get('path', '-')}`.")

    lines.extend(["", "## Watchers", ""])
    for proc in payload["watcher_processes"]:
        lines.append(f"- PID `{proc.get('ProcessId', '-')}`: `{proc.get('CommandLine', '')}`")

    lines.extend(["", "## Disk", "", "| Root | Free GB | Used GB | Total GB |", "|---|---:|---:|---:|"])
    for root, stats in payload["disk"].items():
        lines.append(f"| `{root}` | {stats['free_gb']} | {stats['used_gb']} | {stats['total_gb']} |")

    lines.extend(["", "## Readiness Reports", ""])
    for name, report in payload["readiness_reports"].items():
        lines.append(f"- `{name}`: ok=`{report['ok']}`, path=`{report['path']}`")

    board = payload.get("experiment_board_refresh", {})
    lines.extend(
        [
            "",
            "## Experiment Board",
            "",
            f"- Refresh returncode: `{board.get('returncode', '-')}`",
            f"- Report: `{board.get('md_path', '-')}`",
        ]
    )
    plateau = payload.get("screen_plateau_refresh", {})
    lines.extend(
        [
            "",
            "## Screen Plateau Diagnosis",
            "",
            f"- Refresh returncode: `{plateau.get('returncode', '-')}`",
            f"- Report: `{plateau.get('md_path', '-')}`",
        ]
    )
    promotion = payload.get("promotion_gate_regression", {})
    lines.extend(
        [
            "",
            "## Promotion Gate Regression",
            "",
            f"- Refresh returncode: `{promotion.get('returncode', '-')}`",
            f"- Report: `{promotion.get('md_path', '-')}`",
        ]
    )
    completion = payload.get("sota_completion_evidence", {})
    lines.extend(
        [
            "",
            "## SOTA Completion Evidence",
            "",
            f"- Refresh returncode: `{completion.get('returncode', '-')}`",
            f"- Verdict: `{completion.get('verdict', '-')}`",
            f"- Report: `{completion.get('md_path', '-')}`",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the live RSCD formal queue without starting new experiments.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--live-status-dir", type=Path, default=DEFAULT_LIVE_STATUS_DIR)
    parser.add_argument("--experiment-board-dir", type=Path, default=DEFAULT_EXPERIMENT_BOARD_DIR)
    parser.add_argument("--screen-plateau-dir", type=Path, default=DEFAULT_SCREEN_PLATEAU_DIR)
    parser.add_argument("--promotion-gate-dir", type=Path, default=DEFAULT_PROMOTION_GATE_DIR)
    parser.add_argument("--sota-completion-dir", type=Path, default=DEFAULT_SOTA_COMPLETION_DIR)
    parser.add_argument("--s133c-log", type=Path, default=DEFAULT_S133C_LOG)
    parser.add_argument("--max-log-age-seconds", type=float, default=900.0)
    parser.add_argument("--min-output-free-gb", type=float, default=8.0)
    parser.add_argument("--step-checkpoint-every", type=int, default=2000)
    parser.add_argument("--skip-live-refresh", action="store_true")
    args = parser.parse_args()

    checks: list[Check] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    live_refresh = {"skipped": True}
    if not args.skip_live_refresh:
        live_refresh = refresh_live_status(args.live_status_dir, args.s133c_log)
    add_check(
        checks,
        "live_status_refresh",
        "block",
        bool(args.skip_live_refresh or live_refresh.get("returncode") == 0),
        "Live route status should refresh before queue health is judged.",
        **live_refresh,
    )

    board_refresh = refresh_experiment_board(args.experiment_board_dir)
    add_check(
        checks,
        "experiment_board_refresh",
        "warn",
        board_refresh.get("returncode") == 0,
        "Experiment evidence board should refresh so full/screen/smoke results remain separated.",
        **board_refresh,
    )
    plateau_refresh = refresh_screen_plateau_diagnosis(args.experiment_board_dir, args.screen_plateau_dir)
    add_check(
        checks,
        "screen_plateau_diagnosis_refresh",
        "warn",
        plateau_refresh.get("returncode") == 0,
        "Screen plateau diagnosis should refresh so repeated non-improving route families are explicit.",
        **plateau_refresh,
    )
    promotion_gate = refresh_promotion_gate_regression(args.promotion_gate_dir)
    add_check(
        checks,
        "promotion_gate_regression",
        "block",
        promotion_gate.get("returncode") == 0,
        "Promotion audit fairness gates should block protocol mismatches and enforce full SOTA checks.",
        **promotion_gate,
    )
    sota_completion = refresh_sota_completion_evidence(args.sota_completion_dir)
    add_check(
        checks,
        "sota_completion_evidence_refresh",
        "warn",
        sota_completion.get("returncode") == 0,
        "Strict RSCD SOTA completion evidence should refresh so final-goal status is explicit.",
        **sota_completion,
    )

    live_json_path = args.live_status_dir / "rscd_live_route_status.json"
    live_status = read_json(live_json_path)
    live_decision = (live_status or {}).get("decision") or {}
    add_check(
        checks,
        "live_status_json_exists",
        "block",
        live_status is not None,
        f"Live route JSON must exist: {live_json_path}",
        path=str(live_json_path),
    )
    add_check(
        checks,
        "decision_waits_for_s133c",
        "warn",
        live_decision.get("action") == "wait_for_s133c_full",
        "While S133c is the active full-data candidate, the safe decision is wait_for_s133c_full.",
        decision=live_decision,
    )

    training = get_training_processes()
    heavy_analysis = get_heavy_analysis_processes()
    add_check(
        checks,
        "single_active_training_process",
        "block",
        len(training) == 1,
        "There should be exactly one active RSCD GPU-heavy Python job during S133c.",
        processes=training,
    )
    active_cmd = str(training[0].get("CommandLine", "")) if len(training) == 1 else ""
    add_check(
        checks,
        "active_training_is_s133c",
        "block",
        "s133c_s96_boundary_earlyphysics" in active_cmd,
        "The active job should be the S133c full-data candidate, not a duplicate or downstream route.",
        command=active_cmd,
    )
    add_check(
        checks,
        "no_stray_heavy_analysis_process",
        "warn",
        len(heavy_analysis) == 0,
        "Long CPU analysis jobs should not be left running while the formal GPU queue is active.",
        processes=heavy_analysis,
    )

    progress = parse_log_progress(args.s133c_log)
    add_check(
        checks,
        "s133c_log_progress_parse",
        "block",
        bool(progress.get("progress_found")),
        "S133c train log should contain parseable tqdm progress.",
        progress=progress,
    )
    add_check(
        checks,
        "s133c_log_fresh",
        "block",
        bool(progress.get("progress_found")) and float(progress.get("age_seconds", 10**9)) <= args.max_log_age_seconds,
        "S133c train log should be fresh; stale logs indicate a hung run or wrong process.",
        max_age_seconds=args.max_log_age_seconds,
        progress=progress,
    )
    step_checkpoint = inspect_step_checkpoint(args.s133c_log.parent / "last_step_checkpoint.pth")
    progress_step = int(progress.get("step") or 0)
    checkpoint_expected = bool(progress.get("progress_found")) and progress_step >= int(args.step_checkpoint_every)
    checkpoint_step = int(step_checkpoint.get("step") or 0)
    checkpoint_valid = (
        bool(step_checkpoint.get("readable"))
        and checkpoint_step >= int(args.step_checkpoint_every)
        and int(step_checkpoint.get("total_steps") or 0) >= progress_step
        and int(step_checkpoint.get("model_items") or 0) > 0
        and bool(step_checkpoint.get("has_optimizer"))
        and bool(step_checkpoint.get("has_scaler"))
    )
    add_check(
        checks,
        "s133c_step_checkpoint_recoverable",
        "block" if checkpoint_expected else "warn",
        (not checkpoint_expected) or checkpoint_valid,
        "Once S133c passes the first save interval, last_step_checkpoint.pth should be readable and contain model/optimizer/scaler state.",
        checkpoint_expected=checkpoint_expected,
        step_checkpoint_every=int(args.step_checkpoint_every),
        progress_step=progress_step,
        checkpoint=step_checkpoint,
    )

    watchers = get_watcher_processes()
    for watcher_name, script_name in EXPECTED_WATCHERS.items():
        matches = [proc for proc in watchers if script_name in str(proc.get("CommandLine", ""))]
        add_check(
            checks,
            f"watcher_active_{watcher_name}",
            "warn",
            len(matches) == 1,
            f"Expected exactly one active watcher for {script_name}.",
            matches=matches,
        )

    watcher_requirements = {
        "scripts/watch_s133c_resume_if_needed.ps1": [
            "last_step_checkpoint.pth",
            "RestartCooldownSeconds",
            "Get-ActiveS133cTraining",
            "train.py",
        ],
        "scripts/run_s135_after_s133c.ps1": [
            "audit_s135c_queue_readiness.py",
            "strict_screen_promotion_audit_vs_s96",
            "strict_promotion_audit_vs_s7_full",
        ],
        "scripts/run_s136_after_s135_if_needed.ps1": [
            "audit_s136_queue_readiness.py",
            "S136_screen_promotion_audit_vs_S96",
            "S133c full already passed strict SOTA audit. S136 fallback not needed. Exiting.",
            "S135 screen promoted to full. Waiting for S135 full metrics before deciding on S136 fallback.",
            "S135 full strict SOTA audit failed. S136 fallback may start.",
            "S135 full metrics exist; waiting for strict SOTA audit before deciding on S136 fallback.",
        ],
        "scripts/run_s136d_after_s136_if_needed.ps1": [
            "audit_s136d_queue_readiness.py",
            "cache_teacher_logits.py",
            "S136d_full_promotion_audit_vs_S7",
            "S133c full already passed strict SOTA audit. S136d fallback not needed. Exiting.",
            "S135 full already passed strict SOTA audit. S136d fallback not needed. Exiting.",
            "S135 full strict audit failed but S136 watcher is still active; waiting for S136 decision first.",
        ],
        "scripts/run_s137_after_current_queue_if_needed.ps1": [
            "audit_s137_queue_readiness.py",
            "S137_learned_scale_space_vs_off_control",
            "S137_full_promotion_audit_vs_S7",
            "$S133StrictAuditDir",
        ],
        "scripts/run_s138_after_s137_if_needed.ps1": [
            "audit_s138_queue_readiness.py",
            "S138_dual_film_texture_vs_off_control",
            "S138_full_promotion_audit_vs_S7",
            "$S133StrictAuditDir",
        ],
    }
    watcher_script_audit = {}
    for script_text, tokens in watcher_requirements.items():
        if script_text not in {
            "scripts/run_s135_after_s133c.ps1",
            "scripts/watch_s133c_resume_if_needed.ps1",
        }:
            tokens = list(tokens) + ["FeatureDiagnosis", "--feature-diagnosis"]
        audit = inspect_watcher_script(ROOT / script_text, tokens)
        watcher_script_audit[script_text] = audit
        add_check(
            checks,
            f"watcher_script_gates_{Path(script_text).stem}",
            "block",
            audit["exists"] and not audit["missing"],
            f"Watcher {script_text} should contain its required readiness, promotion, and diagnosis gates.",
            audit=audit,
        )

    readiness_payloads: dict[str, dict[str, Any]] = {}
    for name, path in READINESS_REPORTS.items():
        payload = read_json(path)
        ok = readiness_ok(payload)
        required_checks = REQUIRED_READINESS_CHECKS.get(name, [])
        present_checks = readiness_check_names(payload)
        missing_required_checks = [
            check_name for check_name in required_checks if check_name not in present_checks
        ]
        readiness_payloads[name] = {
            "path": str(path),
            "exists": payload is not None,
            "ok": ok,
            "required_checks": required_checks,
            "missing_required_checks": missing_required_checks,
        }
        add_check(
            checks,
            f"readiness_passed_{name}",
            "block",
            ok,
            f"Latest post-RGB-fix readiness report should pass for {name}.",
            path=str(path),
        )
        add_check(
            checks,
            f"readiness_hardened_{name}",
            "block",
            not missing_required_checks,
            f"Readiness report for {name} should include full-manifest and same-budget hard gates.",
            path=str(path),
            required_checks=required_checks,
            missing=missing_required_checks,
        )

    disk = disk_snapshot([DEFAULT_COMPARISON_DIR, DEFAULT_S133C_LOG, ROOT])
    output_drive = Path(str(DEFAULT_COMPARISON_DIR)).anchor
    output_free = disk.get(output_drive, {}).get("free_gb", -1.0)
    add_check(
        checks,
        "output_disk_free",
        "warn",
        output_free >= args.min_output_free_gb,
        "Output drive should have enough free space for the queued screen/full artifacts.",
        output_drive=output_drive,
        free_gb=output_free,
        min_output_free_gb=args.min_output_free_gb,
    )

    full_sota_runs = []
    for run in (live_status or {}).get("runs", []):
        if run.get("fair_full_test") and run.get("beats_public_sota"):
            full_sota_runs.append(run.get("name"))
    add_check(
        checks,
        "no_full_sota_pass_yet",
        "info",
        not full_sota_runs,
        "No queued full-data run has already cleared both public SOTA thresholds; keep the goal active.",
        full_sota_runs=full_sota_runs,
    )

    block_failures = [check for check in checks if check.severity == "block" and not check.passed]
    warn_failures = [check for check in checks if check.severity == "warn" and not check.passed]
    overall = "pass" if not block_failures else "fail"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall": overall,
        "num_block_failures": len(block_failures),
        "num_warn_failures": len(warn_failures),
        "checks": [asdict(check) for check in checks],
        "live_refresh": live_refresh,
        "experiment_board_refresh": board_refresh,
        "screen_plateau_refresh": plateau_refresh,
        "promotion_gate_regression": promotion_gate,
        "sota_completion_evidence": sota_completion,
        "live_decision": live_decision,
        "s133c_progress": progress,
        "s133c_step_checkpoint": step_checkpoint,
        "training_processes": training,
        "heavy_analysis_processes": heavy_analysis,
        "watcher_processes": watchers,
        "watcher_script_audit": watcher_script_audit,
        "readiness_reports": readiness_payloads,
        "disk": disk,
    }

    json_path = args.output_dir / "rscd_live_queue_health.json"
    md_path = args.output_dir / "rscd_live_queue_health.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, md_path)
    print(md_path)
    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
