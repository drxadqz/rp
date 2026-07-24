from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SOTA_TOP1 = 0.9286
SOTA_MACRO_F1 = 0.8949
FULL_TEST_SAMPLES = 49_500
DEFAULT_COMPARISON_DIR = Path(r"E:\perception_outputs\rscd_surface_classification\comparison_live_20260715")
KEY_CLASSES = (
    "water_concrete_slight",
    "wet_concrete_slight",
    "water_concrete_severe",
    "wet_concrete_severe",
    "dry_concrete_slight",
    "dry_concrete_severe",
    "water_asphalt_slight",
)


DEFAULT_S133C_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification"
    r"\c3_farnet_formal_fullmanifest_s133c_s96_boundary_earlyphysics_b16_20260715"
)
DEFAULT_S135_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification"
    r"\c3_farnet_screen_s135c_s96_wc_moderate_film_rough_focus_stem_20260715"
)
DEFAULT_S135_FULL_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification"
    r"\c3_farnet_formal_fullmanifest_s135c_s96_wc_moderate_film_rough_focus_stem_20260715"
)
DEFAULT_S136_DIR = Path(r"E:\perception_outputs\rscd_surface_classification\s136_coupled_factor_backbone_screen_20260715")
DEFAULT_S136D_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification\s136d_coupled_factor_backbone_safe_distill_screen_20260715"
)
DEFAULT_S136D_FULL_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification\s136d_coupled_factor_backbone_safe_distill_full_20260715"
)
DEFAULT_S137_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification\s137_concrete_roughness_scalespace_screen_20260715"
)
DEFAULT_S137_CONTROL_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification\s137_concrete_roughness_scalespace_control_20260715"
)
DEFAULT_S137_FULL_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification\s137_concrete_roughness_scalespace_full_20260715"
)
DEFAULT_S138_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification\s138_dual_film_texture_roughness_screen_20260716"
)
DEFAULT_S138_CONTROL_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification\s138_dual_film_texture_roughness_control_20260716"
)
DEFAULT_S138_FULL_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification\s138_dual_film_texture_roughness_full_20260716"
)
DEFAULT_S7_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification"
    r"\c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709"
)
DEFAULT_S96_DIR = Path(
    r"E:\perception_outputs\rscd_surface_classification\c3_farnet_screen_s96_wc_pair_relative_boundary_20260712"
)


@dataclass
class RunStatus:
    name: str
    path: str
    exists: bool
    complete: bool
    fair_full_test: bool
    top1: float | None
    macro_f1: float | None
    mean_precision: float | None
    mean_recall: float | None
    weighted_f1: float | None
    num_samples: int | None
    param_count: int | None
    worst_class: str | None
    worst_f1: float | None
    key_class_f1: dict[str, float]
    beats_public_sota: bool | None
    strict_promotion_audit: dict[str, Any] | None
    next_mechanism: dict[str, Any] | None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_summary(run_dir: Path) -> dict[str, Any] | None:
    payload = read_json(run_dir / "test_metrics.json")
    if payload is None:
        return None
    return dict(payload.get("summary", payload))


def optional_int(payload: dict[str, Any], key: str) -> int | None:
    if key not in payload:
        return None
    value = payload.get(key)
    if value is None or value == "":
        return None
    return int(float(value))


def read_per_class(run_dir: Path) -> dict[str, dict[str, float]]:
    path = run_dir / "per_class_metrics.csv"
    if not path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("class") or row.get("\ufeffclass") or "")
            if not name:
                continue
            rows[name] = {
                "precision": float(row.get("precision") or 0.0),
                "recall": float(row.get("recall") or 0.0),
                "f1": float(row.get("f1") or 0.0),
                "support": float(row.get("support") or 0.0),
            }
    return rows


def read_next_mechanism(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "next_mechanism_decision" / "next_mechanism_decision.json"
    payload = read_json(path)
    if payload is None:
        return None
    decision = payload.get("decision") or {}
    return {
        "action": decision.get("action"),
        "mechanism_target": decision.get("mechanism_target"),
        "promote_full": decision.get("promote_full"),
        "full_sota_pass": decision.get("full_sota_pass"),
        "reason": decision.get("reason"),
        "status": payload.get("status", "complete" if payload.get("ok") else "unknown"),
        "report": str(path.with_suffix(".md")),
    }


def read_promotion_audit_payload(path: Path) -> dict[str, Any] | None:
    payload = read_json(path)
    if payload is None:
        return None
    decision = payload.get("decision") or {}
    checks = decision.get("checks") or {}
    paired = decision.get("paired_improvement") or {}
    transfer = payload.get("prediction_transfer") or {}
    delta = decision.get("summary_delta") or {}
    return {
        "path": str(path),
        "report": str(path.with_name("promotion_audit.md")),
        "ok": payload.get("ok"),
        "passed": decision.get("passed"),
        "requires_sota": any(key.endswith("_beats_sota") or key.endswith("_full_test_samples") for key in checks),
        "candidate_samples": decision.get("candidate_samples"),
        "baseline_samples": decision.get("baseline_samples"),
        "common_predictions": decision.get("common_predictions"),
        "top1_delta": delta.get("top1"),
        "macro_f1_delta": delta.get("macro_f1"),
        "top1_beats_sota": checks.get("top1_beats_sota"),
        "macro_f1_beats_sota": checks.get("macro_f1_beats_sota"),
        "fixed": paired.get("fixed", transfer.get("fixed")),
        "worsened": paired.get("worsened", transfer.get("worsened")),
        "net_fixed": paired.get("net_fixed", transfer.get("net_fixed")),
        "paired_sign_test_p_one_sided": paired.get("paired_sign_test_p_one_sided"),
        "mcnemar_p_two_sided_approx": paired.get("mcnemar_p_two_sided_approx"),
        "paired_improvement_significant": paired.get("paired_improvement_significant"),
    }


def promotion_audit_candidates(name: str, run_dir: Path) -> list[Path]:
    if "baseline" in name:
        return []
    candidates: list[Path] = []
    if "screen" in name:
        candidates.append(run_dir / "strict_screen_promotion_audit_vs_s96" / "promotion_audit.json")
    if "full" in name:
        candidates.append(run_dir / "strict_promotion_audit_vs_s7_full" / "promotion_audit.json")

    comparison_prefix = {
        "S136_no_distill_screen": "S136_screen_promotion_audit_vs_S96",
        "S136d_safe_distill_screen": "S136d_screen_promotion_audit_vs_S96",
        "S136d_full": "S136d_full_promotion_audit_vs_S7",
        "S137_concrete_roughness_screen": "S137_screen_promotion_audit_vs_S96",
        "S137_full": "S137_full_promotion_audit_vs_S7",
        "S138_dual_film_texture_screen": "S138_screen_promotion_audit_vs_S96",
        "S138_full": "S138_full_promotion_audit_vs_S7",
    }.get(name)
    if comparison_prefix:
        candidates.append(DEFAULT_COMPARISON_DIR / comparison_prefix / "promotion_audit.json")
    return candidates


def read_strict_promotion_audit(name: str, run_dir: Path) -> dict[str, Any] | None:
    for path in promotion_audit_candidates(name, run_dir):
        audit = read_promotion_audit_payload(path)
        if audit is not None:
            return audit
    return None


def run_status(name: str, run_dir: Path) -> RunStatus:
    exists = run_dir.exists()
    summary = read_summary(run_dir) if exists else None
    per_class = read_per_class(run_dir) if exists else {}
    complete = summary is not None and (run_dir / "per_class_metrics.csv").exists() and (run_dir / "predictions_test.csv").exists()
    num_samples = int(float(summary.get("num_samples", 0) or 0)) if summary else None
    fair_full = bool(num_samples == FULL_TEST_SAMPLES)
    worst_class = None
    worst_f1 = None
    if per_class:
        worst_class, worst_payload = min(per_class.items(), key=lambda item: item[1].get("f1", 0.0))
        worst_f1 = float(worst_payload.get("f1", 0.0))
    top1 = float(summary.get("top1", 0.0)) if summary else None
    macro_f1 = float(summary.get("macro_f1", 0.0)) if summary else None
    beats_public_sota = None
    if fair_full and top1 is not None and macro_f1 is not None:
        beats_public_sota = top1 >= SOTA_TOP1 and macro_f1 >= SOTA_MACRO_F1
    return RunStatus(
        name=name,
        path=str(run_dir),
        exists=exists,
        complete=complete,
        fair_full_test=fair_full,
        top1=top1,
        macro_f1=macro_f1,
        mean_precision=float(summary.get("mean_precision", 0.0)) if summary else None,
        mean_recall=float(summary.get("mean_recall", 0.0)) if summary else None,
        weighted_f1=float(summary.get("weighted_f1", 0.0)) if summary else None,
        num_samples=num_samples,
        param_count=optional_int(summary, "param_count") if summary else None,
        worst_class=worst_class,
        worst_f1=worst_f1,
        key_class_f1={key: float(per_class.get(key, {}).get("f1", 0.0)) for key in KEY_CLASSES if key in per_class},
        beats_public_sota=beats_public_sota,
        strict_promotion_audit=read_strict_promotion_audit(name, run_dir) if exists else None,
        next_mechanism=read_next_mechanism(run_dir) if exists else None,
    )


def parse_progress(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"log": str(log_path), "available": False}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"train:\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)\s+\[([^\]]+)\]", text)
    if not matches:
        return {"log": str(log_path), "available": True, "progress_found": False}
    pct_s, step_s, total_s, elapsed = matches[-1]
    step = int(step_s)
    total = int(total_s)
    return {
        "log": str(log_path),
        "available": True,
        "progress_found": True,
        "percent": int(pct_s),
        "step": step,
        "total_steps": total,
        "remaining_steps": max(total - step, 0),
        "elapsed": elapsed,
    }


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.3f}%"


def pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:+.3f} pp"


def compact_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def status_label(status: RunStatus) -> str:
    if not status.exists:
        return "missing"
    if not status.complete:
        return "running_or_pending"
    if status.fair_full_test:
        return "complete_full"
    return "complete_screen_or_smoke"


def decide_next(statuses: list[RunStatus], diagnosis: dict[str, Any] | None) -> dict[str, str]:
    full_candidates = [
        s for s in statuses if "baseline" not in s.name and s.fair_full_test and s.beats_public_sota
    ]
    strict_passed = [
        s for s in full_candidates if (s.strict_promotion_audit or {}).get("passed") is True
    ]
    if strict_passed:
        best = max(strict_passed, key=lambda item: (item.macro_f1 or 0.0, item.top1 or 0.0))
        return {
            "action": "final_verify_and_write",
            "reason": f"{best.name} clears both public SOTA thresholds and passes strict paired promotion audit.",
        }
    if full_candidates and any(s.strict_promotion_audit is None for s in full_candidates):
        names = ", ".join(s.name for s in full_candidates if s.strict_promotion_audit is None)
        return {
            "action": "wait_for_strict_promotion_audit",
            "reason": f"{names} clears public thresholds but strict paired promotion audit is not available yet.",
        }
    active = next((s for s in statuses if s.name == "S133c_full_candidate" and s.exists and not s.complete), None)
    if active:
        return {
            "action": "wait_for_s133c_full",
            "reason": "The current full-data candidate is still training; do not start another GPU-heavy route.",
        }
    if diagnosis:
        decision = diagnosis.get("decision") or {}
        action = str(decision.get("route_action") or "unknown")
        reason = str(decision.get("reason") or "diagnosis did not include a reason")
        return {"action": action, "reason": reason}
    return {
        "action": "inspect_queue_manually",
        "reason": "No full SOTA pass and no diagnosis file was available.",
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# RSCD Live Route Status",
        "",
        f"- Public SOTA Top-1 threshold: `{pct(SOTA_TOP1)}`",
        f"- Public SOTA Macro-F1 threshold: `{pct(SOTA_MACRO_F1)}`",
        f"- Full-test sample requirement: `{FULL_TEST_SAMPLES}`",
        f"- Decision: `{payload['decision']['action']}`",
        f"- Reason: {payload['decision']['reason']}",
        "",
        "## Active Training Progress",
        "",
    ]
    progress = payload["active_progress"]
    if progress.get("progress_found"):
        lines.extend(
            [
                f"- Log: `{progress['log']}`",
                f"- Progress: `{progress['percent']}%`",
                f"- Step: `{progress['step']}/{progress['total_steps']}`",
                f"- Remaining steps: `{progress['remaining_steps']}`",
                f"- Elapsed: `{progress['elapsed']}`",
            ]
        )
    else:
        lines.append(f"- Progress unavailable from `{progress.get('log')}`")

    lines.extend(
        [
            "",
            "## Run Status",
            "",
            "| Run | State | Samples | Top-1 | Macro-F1 | Worst class | Worst F1 | Params | Beats SOTA | Strict audit |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for item in payload["runs"]:
        strict = item.get("strict_promotion_audit") or {}
        lines.append(
            f"| {item['name']} | {status_label(RunStatus(**item))} | "
            f"{item['num_samples'] if item['num_samples'] is not None else '-'} | "
            f"{pct(item['top1'])} | {pct(item['macro_f1'])} | "
            f"{item['worst_class'] or '-'} | {pct(item['worst_f1'])} | "
            f"{item['param_count'] if item['param_count'] is not None else '-'} | "
            f"{item['beats_public_sota'] if item['beats_public_sota'] is not None else '-'} | "
            f"{strict.get('passed') if strict else '-'} |"
        )

    lines.extend(["", "## Key Class F1", "", "| Run | " + " | ".join(KEY_CLASSES) + " |", "|---" + "|---:" * len(KEY_CLASSES) + "|"])
    for item in payload["runs"]:
        values = [pct(item["key_class_f1"].get(key)) for key in KEY_CLASSES]
        lines.append(f"| {item['name']} | " + " | ".join(values) + " |")

    audit_rows = [item for item in payload["runs"] if item.get("strict_promotion_audit")]
    if audit_rows:
        lines.extend(
            [
                "",
                "## Strict Promotion Audits",
                "",
                "| Run | Passed | Samples | Common rows | Top-1 delta | Macro-F1 delta | Fixed | Worsened | Net | Sign-test p | Paired significant |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in audit_rows:
            audit = item["strict_promotion_audit"]
            lines.append(
                f"| {item['name']} | {audit.get('passed')} | "
                f"{compact_number(audit.get('candidate_samples'))}/{compact_number(audit.get('baseline_samples'))} | "
                f"{compact_number(audit.get('common_predictions'))} | "
                f"{pp(audit.get('top1_delta'))} | {pp(audit.get('macro_f1_delta'))} | "
                f"{compact_number(audit.get('fixed'))} | {compact_number(audit.get('worsened'))} | "
                f"{compact_number(audit.get('net_fixed'))} | "
                f"{compact_number(audit.get('paired_sign_test_p_one_sided'))} | "
                f"{audit.get('paired_improvement_significant')} |"
            )

    lines.extend(
        [
            "",
            "## Next Mechanism Decisions",
            "",
            "| Run | Action | Mechanism target | Promote full | Full SOTA pass | Status |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for item in payload["runs"]:
        decision = item.get("next_mechanism")
        if not decision:
            lines.append(f"| {item['name']} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {item['name']} | {decision.get('action') or '-'} | "
            f"{decision.get('mechanism_target') or '-'} | "
            f"{decision.get('promote_full') if decision.get('promote_full') is not None else '-'} | "
            f"{decision.get('full_sota_pass') if decision.get('full_sota_pass') is not None else '-'} | "
            f"{decision.get('status') or '-'} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a live status report for the RSCD SOTA route.")
    parser.add_argument("--s133c-dir", type=Path, default=DEFAULT_S133C_DIR)
    parser.add_argument("--s135-dir", type=Path, default=DEFAULT_S135_DIR)
    parser.add_argument("--s135-full-dir", type=Path, default=DEFAULT_S135_FULL_DIR)
    parser.add_argument("--s136-dir", type=Path, default=DEFAULT_S136_DIR)
    parser.add_argument("--s136d-dir", type=Path, default=DEFAULT_S136D_DIR)
    parser.add_argument("--s136d-full-dir", type=Path, default=DEFAULT_S136D_FULL_DIR)
    parser.add_argument("--s137-dir", type=Path, default=DEFAULT_S137_DIR)
    parser.add_argument("--s137-control-dir", type=Path, default=DEFAULT_S137_CONTROL_DIR)
    parser.add_argument("--s137-full-dir", type=Path, default=DEFAULT_S137_FULL_DIR)
    parser.add_argument("--s138-dir", type=Path, default=DEFAULT_S138_DIR)
    parser.add_argument("--s138-control-dir", type=Path, default=DEFAULT_S138_CONTROL_DIR)
    parser.add_argument("--s138-full-dir", type=Path, default=DEFAULT_S138_FULL_DIR)
    parser.add_argument("--s7-dir", type=Path, default=DEFAULT_S7_DIR)
    parser.add_argument("--s96-dir", type=Path, default=DEFAULT_S96_DIR)
    parser.add_argument("--s133c-log", type=Path, default=DEFAULT_S133C_DIR / "train_stderr_20260715_182019.log")
    parser.add_argument(
        "--diagnosis-json",
        type=Path,
        default=Path(
            r"E:\perception_outputs\rscd_surface_classification\comparison_live_20260715"
            r"\S136d_mechanism_diagnosis_latest\s136d_mechanism_diagnosis.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    statuses = [
        run_status("S133c_full_candidate", args.s133c_dir),
        run_status("S135c_screen_candidate", args.s135_dir),
        run_status("S135c_full_candidate", args.s135_full_dir),
        run_status("S136_no_distill_screen", args.s136_dir),
        run_status("S136d_safe_distill_screen", args.s136d_dir),
        run_status("S136d_full", args.s136d_full_dir),
        run_status("S137_concrete_roughness_screen", args.s137_dir),
        run_status("S137_off_control_screen", args.s137_control_dir),
        run_status("S137_full", args.s137_full_dir),
        run_status("S138_dual_film_texture_screen", args.s138_dir),
        run_status("S138_off_control_screen", args.s138_control_dir),
        run_status("S138_full", args.s138_full_dir),
        run_status("S7_full_baseline", args.s7_dir),
        run_status("S96_cap250_baseline", args.s96_dir),
    ]
    diagnosis = read_json(args.diagnosis_json)
    payload = {
        "active_progress": parse_progress(args.s133c_log),
        "decision": decide_next(statuses, diagnosis),
        "runs": [asdict(status) for status in statuses],
        "thresholds": {
            "sota_top1": SOTA_TOP1,
            "sota_macro_f1": SOTA_MACRO_F1,
            "full_test_samples": FULL_TEST_SAMPLES,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rscd_live_route_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(payload, args.output_dir / "rscd_live_route_status.md")
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "report": str(args.output_dir / "rscd_live_route_status.md"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
