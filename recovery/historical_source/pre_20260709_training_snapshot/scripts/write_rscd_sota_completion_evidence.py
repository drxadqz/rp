from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_COMPARISON_DIR = Path(r"E:\perception_outputs\rscd_surface_classification\comparison_live_20260715")
DEFAULT_LIVE_JSON = DEFAULT_COMPARISON_DIR / "live_route_status_20260715" / "rscd_live_route_status.json"
DEFAULT_BOARD_JSON = DEFAULT_COMPARISON_DIR / "experiment_board_20260716" / "rscd_experiment_board.json"
DEFAULT_HEALTH_JSON = DEFAULT_COMPARISON_DIR / "queue_health_20260716" / "rscd_live_queue_health.json"
DEFAULT_OUTPUT_DIR = DEFAULT_COMPARISON_DIR / "sota_completion_evidence_20260716"
FULL_TRAIN_SAMPLES = 958_941
FULL_VAL_SAMPLES = 19_860
FULL_TEST_SAMPLES = 49_500


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def is_empty(value: Any) -> bool:
    return value is None or value == ""


def count_manifest_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{100.0 * float(value):.3f}%"
    except (TypeError, ValueError):
        return "-"


def pp(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{100.0 * float(value):+.3f} pp"
    except (TypeError, ValueError):
        return "-"


def is_baseline(name: str) -> bool:
    lowered = name.lower()
    return "baseline" in lowered or name in {"S7_full_baseline", "S96_cap250_baseline"}


def full_candidate_rows(live: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in live.get("runs", []) or []:
        if not isinstance(row, dict):
            continue
        if is_baseline(str(row.get("name", ""))):
            continue
        if bool(row.get("fair_full_test")):
            rows.append(row)
    return rows


def best_board_full(board: dict[str, Any]) -> dict[str, Any] | None:
    rows = [
        row
        for row in board.get("runs", []) or []
        if isinstance(row, dict) and row.get("protocol") == "full" and int(row.get("num_samples") or 0) == 49_500
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: (float(row.get("macro_f1") or 0.0), float(row.get("top1") or 0.0)))


def full_training_protocol_evidence(row: dict[str, Any], full_samples: int) -> dict[str, Any]:
    run_dir = Path(str(row.get("path") or ""))
    config_path = run_dir / "config_resolved.yaml"
    cfg = read_yaml(config_path)
    if cfg is None:
        return {
            "passed": False,
            "config_path": str(config_path),
            "reason": "config_resolved.yaml missing or unreadable",
            "checks": {"config_exists": False},
        }

    data = cfg.get("data") or {}
    train_cfg = cfg.get("train") or {}
    eval_cfg = cfg.get("eval") or {}
    manifests = {
        "train": Path(str(data.get("train_manifest") or "")),
        "val": Path(str(data.get("val_manifest") or "")),
        "test": Path(str(data.get("test_manifest") or "")),
    }
    counts = {split: count_manifest_rows(path) for split, path in manifests.items()}
    checks = {
        "config_exists": True,
        "train_manifest_full": counts["train"] == FULL_TRAIN_SAMPLES,
        "val_manifest_full": counts["val"] == FULL_VAL_SAMPLES,
        "test_manifest_full": counts["test"] == full_samples,
        "train_samples_per_epoch_uncapped": int(train_cfg.get("samples_per_epoch") or 0) == 0,
        "train_max_samples_per_class_uncapped": is_empty(train_cfg.get("max_train_samples_per_class")),
        "train_max_samples_uncapped": is_empty(train_cfg.get("max_train_samples")),
        "eval_val_uncapped": is_empty(eval_cfg.get("max_val_samples_per_class")),
        "eval_test_uncapped": is_empty(eval_cfg.get("max_test_samples_per_class")),
    }
    return {
        "passed": all(bool(value) for value in checks.values()),
        "config_path": str(config_path),
        "manifest_paths": {split: str(path) for split, path in manifests.items()},
        "manifest_counts": counts,
        "checks": checks,
        "train_protocol": {
            "samples_per_epoch": train_cfg.get("samples_per_epoch"),
            "max_train_samples_per_class": train_cfg.get("max_train_samples_per_class"),
            "max_train_samples": train_cfg.get("max_train_samples"),
        },
        "eval_protocol": {
            "max_val_samples_per_class": eval_cfg.get("max_val_samples_per_class"),
            "max_test_samples_per_class": eval_cfg.get("max_test_samples_per_class"),
        },
    }


def build_report(
    live: dict[str, Any],
    board: dict[str, Any],
    health: dict[str, Any],
    *,
    live_json: Path = DEFAULT_LIVE_JSON,
    board_json: Path = DEFAULT_BOARD_JSON,
    health_json: Path = DEFAULT_HEALTH_JSON,
) -> dict[str, Any]:
    thresholds = live.get("thresholds") or board.get("thresholds") or {}
    sota_top1 = float(thresholds.get("sota_top1") or 0.9286)
    sota_macro = float(thresholds.get("sota_macro_f1") or 0.8949)
    full_samples = int(thresholds.get("full_test_samples") or 49_500)
    full_rows = full_candidate_rows(live)
    protocol_evidence = {
        str(row.get("name")): full_training_protocol_evidence(row, full_samples)
        for row in full_rows
    }
    protocol_passed = [
        row for row in full_rows if bool(protocol_evidence.get(str(row.get("name")), {}).get("passed"))
    ]
    strict_audit_passed = [
        row
        for row in full_rows
        if bool(row.get("beats_public_sota"))
        and bool((row.get("strict_promotion_audit") or {}).get("passed"))
        and int(row.get("num_samples") or 0) == full_samples
    ]
    strict_passed = [
        row
        for row in strict_audit_passed
        if bool(protocol_evidence.get(str(row.get("name")), {}).get("passed"))
    ]
    threshold_only = [
        row
        for row in full_rows
        if bool(row.get("beats_public_sota")) and int(row.get("num_samples") or 0) == full_samples
    ]
    missing_strict = [row for row in threshold_only if row.get("strict_promotion_audit") is None]
    failed_strict = [
        row
        for row in threshold_only
        if row.get("strict_promotion_audit") is not None
        and not bool((row.get("strict_promotion_audit") or {}).get("passed"))
    ]
    active_progress = live.get("active_progress") or {}
    if strict_passed:
        verdict = "complete"
        next_action = "Finalize the SOTA claim from the strict audit and per-class evidence."
    elif strict_audit_passed:
        verdict = "strict_audit_pass_but_full_training_protocol_failed"
        next_action = "Do not claim completion until the candidate is proven to use complete train/val/test manifests without caps."
    elif missing_strict:
        verdict = "awaiting_strict_promotion_audit"
        next_action = "Wait for strict paired promotion audit before claiming SOTA."
    elif failed_strict:
        verdict = "threshold_pass_but_strict_audit_failed"
        next_action = "Do not claim SOTA; inspect paired regressions and weak classes."
    elif active_progress.get("progress_found"):
        verdict = "awaiting_active_full_run"
        next_action = "Let the active full-data run finish; do not start another GPU route."
    elif full_rows:
        verdict = "not_yet_sota"
        next_action = "Use the weakest classes and paired regressions to choose the next task-adapted mechanism."
    else:
        verdict = "no_completed_candidate_full_run"
        next_action = "Wait for the queued full run or launch the next gated route only after the current queue is idle."

    blockers = []
    if not strict_passed:
        blockers.append(
            "No non-baseline run currently proves all four hard conditions: complete train/val/test protocol, "
            "49,500 test samples, public SOTA thresholds, and strict paired audit pass."
        )
    if active_progress.get("progress_found"):
        blockers.append(
            "Active run still training: step {}/{} ({}%).".format(
                active_progress.get("step", "-"),
                active_progress.get("total_steps", "-"),
                active_progress.get("percent", "-"),
            )
        )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "verdict": verdict,
        "next_action": next_action,
        "thresholds": {
            "sota_top1": sota_top1,
            "sota_macro_f1": sota_macro,
            "full_test_samples": full_samples,
        },
        "hard_requirements": {
            "non_baseline_full_test_49500": bool(full_rows),
            "complete_train_val_test_protocol": bool(protocol_passed),
            "beats_public_top1_and_macro_f1": bool(threshold_only),
            "strict_paired_promotion_audit_passed": bool(strict_audit_passed),
            "all_completion_evidence_passed": bool(strict_passed),
        },
        "active_progress": active_progress,
        "queue_health": {
            "overall": health.get("overall"),
            "generated_at": health.get("generated_at"),
            "live_decision": health.get("live_decision"),
            "advisory_only": True,
            "report": str(health_json.with_suffix(".md")),
        },
        "completed_full_candidates": [
            {
                "name": row.get("name"),
                "path": row.get("path"),
                "top1": row.get("top1"),
                "macro_f1": row.get("macro_f1"),
                "num_samples": row.get("num_samples"),
                "beats_public_sota": row.get("beats_public_sota"),
                "strict_promotion_audit": row.get("strict_promotion_audit"),
                "full_training_protocol": protocol_evidence.get(str(row.get("name"))),
                "worst_class": row.get("worst_class"),
                "worst_f1": row.get("worst_f1"),
            }
            for row in full_rows
        ],
        "strict_passed_candidates": [row.get("name") for row in strict_passed],
        "threshold_only_candidates": [row.get("name") for row in threshold_only],
        "missing_strict_candidates": [row.get("name") for row in missing_strict],
        "failed_strict_candidates": [row.get("name") for row in failed_strict],
        "best_full_board_row": best_board_full(board),
        "blockers": blockers,
        "source_files": {
            "live_status": str(live_json),
            "experiment_board": str(board_json),
            "queue_health": str(health_json),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    thresholds = report["thresholds"]
    best = report.get("best_full_board_row") or {}
    lines = [
        "# RSCD SOTA Completion Evidence",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Verdict: `{report['verdict']}`",
        f"- Next action: {report['next_action']}",
        f"- Required full-test samples: `{thresholds['full_test_samples']}`",
        f"- Public SOTA Top-1 / Macro-F1: `{pct(thresholds['sota_top1'])}` / `{pct(thresholds['sota_macro_f1'])}`",
        "",
        "## Hard Requirements",
        "",
        "| Requirement | Satisfied |",
        "|---|---:|",
    ]
    for key, value in report["hard_requirements"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Current Best Full Evidence", ""])
    if best:
        lines.extend(
            [
                f"- Board best full run: `{best.get('run')}`",
                f"- Samples: `{best.get('num_samples')}`",
                f"- Top-1 / Macro-F1: `{pct(best.get('top1'))}` / `{pct(best.get('macro_f1'))}`",
                f"- Gaps to SOTA: `{pp(best.get('top1_gap_to_sota'))}` / `{pp(best.get('macro_f1_gap_to_sota'))}`",
                f"- Extra correct needed for Top-1 SOTA: `{best.get('extra_correct_to_top1_sota')}`",
                f"- Worst class: `{best.get('worst_class')}` ({pct(best.get('worst_f1'))})",
            ]
        )
    else:
        lines.append("- No full board row is available.")

    lines.extend(
        [
            "",
            "## Completed Candidate Full Runs",
            "",
            "| Run | Samples | Full train protocol | Top-1 | Macro-F1 | Beats public SOTA | Strict audit | Worst class | Worst F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    rows = report.get("completed_full_candidates") or []
    if rows:
        for row in rows:
            audit = row.get("strict_promotion_audit") or {}
            protocol = row.get("full_training_protocol") or {}
            lines.append(
                f"| {row.get('name')} | {row.get('num_samples')} | {protocol.get('passed')} | "
                f"{pct(row.get('top1'))} | {pct(row.get('macro_f1'))} | "
                f"{row.get('beats_public_sota')} | {audit.get('passed') if audit else '-'} | "
                f"{row.get('worst_class') or '-'} | {pct(row.get('worst_f1'))} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - |")

    protocol_rows = [row for row in rows if row.get("full_training_protocol")]
    if protocol_rows:
        lines.extend(
            [
                "",
                "## Full Training Protocol Evidence",
                "",
                "| Run | Train rows | Val rows | Test rows | samples_per_epoch | train cap/class | val cap | test cap | Passed |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in protocol_rows:
            protocol = row.get("full_training_protocol") or {}
            counts = protocol.get("manifest_counts") or {}
            train_protocol = protocol.get("train_protocol") or {}
            eval_protocol = protocol.get("eval_protocol") or {}
            lines.append(
                f"| {row.get('name')} | {counts.get('train')} | {counts.get('val')} | {counts.get('test')} | "
                f"{train_protocol.get('samples_per_epoch')} | {train_protocol.get('max_train_samples_per_class')} | "
                f"{eval_protocol.get('max_val_samples_per_class')} | {eval_protocol.get('max_test_samples_per_class')} | "
                f"{protocol.get('passed')} |"
            )

    lines.extend(["", "## Blockers", ""])
    if report.get("blockers"):
        lines.extend(f"- {item}" for item in report["blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "## Source Files", ""])
    for key, value in report.get("source_files", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write strict evidence for whether the RSCD SOTA objective is complete.")
    parser.add_argument("--live-json", type=Path, default=DEFAULT_LIVE_JSON)
    parser.add_argument("--board-json", type=Path, default=DEFAULT_BOARD_JSON)
    parser.add_argument("--health-json", type=Path, default=DEFAULT_HEALTH_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    live = read_json(args.live_json) or {}
    board = read_json(args.board_json) or {}
    health = read_json(args.health_json) or {}
    report = build_report(live, board, health, live_json=args.live_json, board_json=args.board_json, health_json=args.health_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "rscd_sota_completion_evidence.json"
    md_path = args.output_dir / "rscd_sota_completion_evidence.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "report": str(md_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
