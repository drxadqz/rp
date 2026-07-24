from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_PYTHON = Path(r"D:\NMI_SPWFM_datasets\conda_envs\faf_paper\python.exe")
DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")


LEAN_FAST_WAVE = [
    "v1_physics_texture",
    "v14_lean_road_roi_safety",
    "v17_lean_quality_physics_safety",
    "v18_lean_mixstyle_quality_safety",
    "v19_lean_state_contrast_quality_safety",
    "v20_lean_interval_order_quality_safety",
    "v21_lean_quality_uncertainty_safety",
    "v22_lean_quality_order_contrast_safety",
    "v23_lean_region_mixture_evidence_safety",
    "v24_lean_multi_query_region_evidence_safety",
    "v25_lean_masked_query_consistency_safety",
    "v16_lean_bottom_square_color_constancy_safety",
    "v15_lean_bottom_square_style_safety",
    "v13_lean_physics_evidence",
]

FORMAL_REFERENCE_CANDIDATES = [
    "v14_lean_road_roi_safety",
    "v17_lean_quality_physics_safety",
    "v18_lean_mixstyle_quality_safety",
    "v19_lean_state_contrast_quality_safety",
    "v20_lean_interval_order_quality_safety",
    "v21_lean_quality_uncertainty_safety",
    "v22_lean_quality_order_contrast_safety",
    "v23_lean_region_mixture_evidence_safety",
    "v24_lean_multi_query_region_evidence_safety",
    "v25_lean_masked_query_consistency_safety",
]

FULL_STACK_HELD = [
    "v6_full_faf_fourier",
    "v7_full_faf_fourier_dann",
    "v8_full_faf_fourier_roadprior",
    "v9_full_faf_roadsaw_hard_sampling",
    "v10_full_faf_consistency",
    "v11_full_faf_domain_adapter",
    "v12_full_faf_roi_interval_safety",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "rapid_exploration_decision_report.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_SUMMARY_DIR / "rapid_exploration_decision_report.json",
    )
    args = parser.parse_args()

    report = build_report(
        args.summary_dir,
        python=args.python,
        root=args.root,
        log_dir=args.log_dir,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(
    summary_dir: Path,
    *,
    python: Path,
    root: Path,
    log_dir: Path,
) -> dict[str, Any]:
    fail_fast = _load_json(summary_dir / "fail_fast_exploration_report.json") or {}
    fast_to_formal = _load_json(summary_dir / "fast_to_formal_promotion_report.json") or {}
    guard = _load_json(summary_dir / "gpu_scheduling_guard_report.json") or {}
    active = (
        _load_json(summary_dir / "active_training_watch_report.json")
        or _load_json(summary_dir / "active_live_training_reports.json")
        or {}
    )
    p0 = _load_json(summary_dir / "paper_p0_ablation_table.json") or {}
    lodo = _load_json(summary_dir / "lodo_generalization_report.json") or {}
    external = _load_json(summary_dir / "external_benchmark_report.json") or {}
    direct_benchmark = _load_json(summary_dir / "direct_friction_public_benchmark_audit.json") or {}
    quality = _load_json(summary_dir / "quality_domain_diagnostic_report.json") or {}

    killed = _kill_rows(fail_fast, p0)
    protected = _protect_rows(fail_fast)
    active_state = _active_state(active, guard)
    training_signal = _training_signal(summary_dir, active_state)
    next_wave = _next_wave(fail_fast, fast_to_formal, guard, python, root, log_dir)
    hard_problems = _hard_problems(lodo, external, direct_benchmark, quality)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": _verdict(guard, fast_to_formal, fail_fast),
        "claim_boundary": (
            "This report controls exploration speed and pruning. It does not turn a "
            "fast-screen result into a paper claim; full matched baselines, LODO, "
            "calibration, bootstrap, and claim audits remain mandatory."
        ),
        "active_state": active_state,
        "current_training_signal": training_signal,
        "kill_now": killed,
        "protect_or_keep": protected,
        "hard_problems": hard_problems,
        "next_wave": next_wave,
        "rapid_validation_rules": _rapid_validation_rules(),
        "route_if_next_wave_fails": _route_if_next_wave_fails(),
    }


def _verdict(
    guard: dict[str, Any],
    fast_to_formal: dict[str, Any],
    fail_fast: dict[str, Any],
) -> str:
    if guard.get("manual_launch_allowed") is False:
        return "gpu_busy_keep_monitoring_no_new_launch"
    promotion_verdict = fast_to_formal.get("verdict")
    if promotion_verdict == "promotion_ready":
        return "gpu_idle_run_promoted_formal"
    if promotion_verdict == "no_candidate_clearly_promoted":
        return "gpu_idle_redesign_before_formal"
    policy = (fail_fast.get("formal_policy") or {}).get("verdict")
    if policy == "screen_first":
        return "gpu_idle_run_lean_fast_screen_first"
    return "gpu_idle_refresh_reports_then_screen"


def _active_state(active: dict[str, Any], guard: dict[str, Any]) -> dict[str, Any]:
    current = active.get("active") or {}
    return {
        "gpu_guard_verdict": guard.get("verdict"),
        "manual_launch_allowed": guard.get("manual_launch_allowed"),
        "active_run": current.get("name"),
        "epoch": current.get("epoch"),
        "epochs": current.get("epochs"),
        "step": current.get("step"),
        "steps": current.get("steps"),
        "phase": current.get("phase"),
        "eta": current.get("eta") or current.get("tqdm_eta"),
        "rate": current.get("rate") or current.get("tqdm_rate"),
        "blockers": guard.get("blockers") or [],
    }


def _training_signal(summary_dir: Path, active_state: dict[str, Any]) -> dict[str, Any]:
    run = active_state.get("active_run")
    if not run:
        return {"status": "no_active_run"}
    diagnosis = _load_json(summary_dir / f"{run}_training_diagnosis.json") or {}
    if diagnosis:
        active_progress = diagnosis.get("active_progress") or {}
    else:
        active_progress = {}
    latest = diagnosis.get("latest_epoch") or {}
    best_loss = diagnosis.get("best_val_loss_epoch") or {}
    best_safety = diagnosis.get("best_safety_proxy_epoch") or {}
    signals = diagnosis.get("signals") or {}
    if not latest:
        if diagnosis:
            return {
                "status": "active_no_completed_epoch",
                "run": run,
                "active_progress": active_progress,
                "message": "diagnosis exists, but no validation epoch has completed yet",
            }
        return {"status": "missing_diagnosis", "run": run}
    loss_epoch = best_loss.get("epoch")
    safety_epoch = best_safety.get("epoch")
    checkpoint_policy = "selected_checkpoint_required"
    if loss_epoch and safety_epoch and loss_epoch != safety_epoch:
        checkpoint_policy = "do_not_blindly_use_best_loss_or_latest"
    return {
        "status": "available",
        "run": run,
        "latest_epoch": latest.get("epoch"),
        "latest_val_loss": latest.get("val_loss"),
        "latest_risk_acc": latest.get("val_risk_acc"),
        "latest_friction_acc": latest.get("val_friction_acc"),
        "latest_raw_coverage": latest.get("raw_coverage"),
        "latest_raw_width": latest.get("raw_width"),
        "best_val_loss_epoch": loss_epoch,
        "best_safety_epoch": safety_epoch,
        "val_loss_delta_vs_previous": signals.get("val_loss_delta_vs_previous"),
        "risk_acc_delta_vs_previous": signals.get("risk_acc_delta_vs_previous"),
        "raw_coverage_delta_vs_previous": signals.get("raw_coverage_delta_vs_previous"),
        "checkpoint_policy": checkpoint_policy,
    }


def _kill_rows(fail_fast: dict[str, Any], p0: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for row in fail_fast.get("kill_or_downgrade") or []:
        rows.append(
            {
                "item": str(row.get("item") or "-"),
                "decision": str(row.get("decision") or "-"),
                "evidence": _format_evidence(row.get("evidence")),
                "action": str(row.get("action") or "-"),
            }
        )
    if rows:
        return rows

    p0_rows = p0.get("rows") or []
    for row in p0_rows:
        method = str(row.get("method") or "")
        if method in {"+ DG losses", "Full model"}:
            rows.append(
                {
                    "item": method,
                    "decision": "kill_or_rework_if_no_later_rescue",
                    "evidence": (
                        f"risk F1={_pct(row.get('risk_macro_f1'))}, "
                        f"low recall={_pct(row.get('low_friction_recall'))}, "
                        f"worst dataset F1={_pct(row.get('worst_dataset_f1'))}"
                    ),
                    "action": "Do not spend formal GPU time before fast-screen evidence.",
                }
            )
    return rows


def _protect_rows(fail_fast: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for row in fail_fast.get("protect_or_conditional_keep") or []:
        rows.append(
            {
                "item": str(row.get("item") or "-"),
                "decision": str(row.get("decision") or "-"),
                "evidence": _format_evidence(row.get("evidence")),
                "action": str(row.get("action") or "-"),
            }
        )
    if rows:
        return rows
    return [
        {
            "item": "PhysicsTexture",
            "decision": "protect_as_lean_core",
            "evidence": "P0 ablation improved risk, friction, low-friction recall, and worst-dataset F1.",
            "action": "Use it as the first principle for new candidates.",
        }
    ]


def _hard_problems(
    lodo: dict[str, Any],
    external: dict[str, Any],
    direct_benchmark: dict[str, Any],
    quality: dict[str, Any],
) -> list[dict[str, str]]:
    problems = [
        {
            "problem": "cross_dataset_transfer",
            "status": "failed_in_base_lodo",
            "evidence": _lodo_evidence(lodo),
            "fast_test": "Only trust LODO as failure evidence until final lean LODO rows improve.",
        },
        {
            "problem": "fair_numeric_claim",
            "status": "not_ready",
            "evidence": _external_completion(external),
            "fast_test": "Use matched single-dataset FAF vs ConvNeXt rows before any superiority claim.",
        },
        {
            "problem": "direct_measured_friction_benchmark",
            "status": "context_only_currently",
            "evidence": _direct_benchmark_evidence(direct_benchmark),
            "fast_test": "Do not spend GPU time reproducing SIWNet/WCamNet until a protocol-equivalent measured-friction dataset is locally available.",
        },
        {
            "problem": "roadsaw_quality_and_wetness",
            "status": "watchlist",
            "evidence": _quality_evidence(quality),
            "fast_test": "v17 must improve quality-slice or wetness/risk behavior without widening intervals too much.",
        },
    ]
    return problems


def _direct_benchmark_evidence(report: dict[str, Any]) -> str:
    if not report:
        return "Direct-friction public-benchmark audit is missing; keep external friction-regression papers out of numeric claims."
    counts = report.get("counts") or {}
    verdict = report.get("verdict")
    main = report.get("current_main_comparison") or ""
    direct = counts.get("direct_context_sources")
    proxy = counts.get("fair_proxy_sources")
    return _compact_text(
        f"verdict={verdict}; direct_context_sources={direct}; fair_proxy_sources={proxy}; {main}",
        300,
    )


def _lodo_evidence(lodo: dict[str, Any]) -> str:
    rows = lodo.get("rows") or lodo.get("heldout_rows") or []
    if not rows:
        return "LODO report unavailable in JSON; keep base LODO failure from current summaries as the route boundary."
    parts = []
    for row in rows[:3]:
        name = row.get("held_out") or row.get("heldout_dataset") or row.get("run") or row.get("dataset") or "-"
        risk = row.get("risk_f1") or row.get("risk_macro_f1")
        friction = row.get("friction_f1") or row.get("friction_macro_f1")
        parts.append(f"{name}: risk F1 {_pct(risk)}, friction F1 {_pct(friction)}")
    return "; ".join(parts)


def _external_completion(external: dict[str, Any]) -> str:
    completion = external.get("completion_status") or {}
    if completion:
        missing = completion.get("fair_single_dataset_complete")
        final = completion.get("final_method_complete")
        return f"fair_single_dataset_complete={missing}; final_method_complete={final}"
    return "External benchmark report says matched single-dataset ConvNeXt rows are still mandatory."


def _quality_evidence(quality: dict[str, Any]) -> str:
    diagnosis = quality.get("diagnosis") or []
    if diagnosis:
        return _compact_text(str(diagnosis[0].get("evidence") or diagnosis[0].get("finding") or ""), 260)
    return "RoadSaW near-white and low-contrast slices must remain explicit stress slices."


def _next_wave(
    fail_fast: dict[str, Any],
    fast_to_formal: dict[str, Any],
    guard: dict[str, Any],
    python: Path,
    root: Path,
    log_dir: Path,
) -> dict[str, Any]:
    policy = fail_fast.get("formal_policy") or {}
    formal_command = fast_to_formal.get("formal_command") or policy.get("formal_promoted_command")
    fast_command = policy.get("fast_screen_command") or (
        f"{python} scripts\\run_fast_screen_protocol.py --scope candidates --lean-first-wave "
        "--bootstrap-samples 100 --dataset-diagnostic-samples 2000 "
        "--evidence-map-samples 12 --evidence-audit-samples 1000"
    )

    if guard.get("manual_launch_allowed") is False:
        command = None
        mode = "wait_active_gpu"
        reason = "GPU guard is busy; do not manually launch a duplicate worker."
    elif formal_command and fast_to_formal.get("verdict") == "promotion_ready":
        command = formal_command
        mode = "formal_promoted"
        reason = "Fast-screen evidence has chosen a small formal set."
    else:
        command = fast_command
        mode = "lean_fast_screen"
        reason = "Run a cheap candidate screen before any full-stack formal route."

    return {
        "mode": mode,
        "reason": reason,
        "command": command,
        "first_wave": (policy.get("fast_screen_first_wave") or LEAN_FAST_WAVE),
        "formal_reference_candidates": (policy.get("promoted_or_fallback") or FORMAL_REFERENCE_CANDIDATES),
        "held_full_stack": (policy.get("full_stack_held_until_screen") or FULL_STACK_HELD),
        "formal_command_template": (
            f"{python} scripts\\run_paper_protocol_direct.py --phase candidates "
            f"--candidate-policy fail_fast --python {python} --root {root} "
            f"--log-dir {log_dir} --postprocess-each"
        ),
    }


def _rapid_validation_rules() -> list[str]:
    return [
        "Stop a route if risk F1 or low-friction recall drops by at least 2 percentage points without a clear worst-domain or interval gain.",
        "Stop a route if calibrated coverage is below 88% after calibration, unless it is only a diagnostic run.",
        "Do not promote a route that expands interval width by more than 0.04-0.05 without fixing a safety-critical slice.",
        "Treat dataset-ID balanced accuracy above 85% as shortcut risk; require either lower shortcut or better held-out/worst-domain safety.",
        "Promote at most two candidates from a fast screen to full formal runs.",
        "Keep all external-paper claims contextual unless split, labels, and metrics match exactly.",
        "Discard direct measured-friction numeric comparisons unless the same sensor target, public split, and metric are reproduced locally.",
    ]


def _route_if_next_wave_fails() -> list[dict[str, str]]:
    return [
        {
            "trigger": "v14, v17, v18, v19, v20, v21, v22, v23, and v24 all fail",
            "route": "abandon full-stack FAF as final method",
            "next_move": "Use PhysicsTexture + matched ConvNeXt as the core paper, with EvidenceField only as failure-analysis/interpretability.",
        },
        {
            "trigger": "all lean candidates pass in-domain but fail LODO",
            "route": "single-dataset paper route",
            "next_move": "Frame multi-dataset results as domain-gap evidence; add RSCD per-day or ExtremeRoad as a separate benchmark only after audit.",
        },
        {
            "trigger": "dataset shortcut stays high after v15/v16/v18/v19/v20/v21/v22/v23/v24",
            "route": "foundation-feature or pseudo-road-mask probe",
            "next_move": "Try a frozen DINOv2/road-mask constrained probe only after current formal queue is idle.",
        },
        {
            "trigger": "RoadSaW wet/near-white remains poor",
            "route": "quality-aware wet-physics branch",
            "next_move": "Add a minimal specular/contrast/water-film RGB proxy head and test only on wetness/quality slices first.",
        },
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Rapid Exploration Decision Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Boundary: {report['claim_boundary']}",
        "",
        "## Active State",
        "",
    ]
    active = report["active_state"]
    lines.append(
        "- GPU guard `{guard}`, manual launch `{manual}`, active `{run}`, epoch `{epoch}/{epochs}`, "
        "step `{step}/{steps}`, ETA `{eta}`, rate `{rate}`.".format(
            guard=active.get("gpu_guard_verdict"),
            manual=active.get("manual_launch_allowed"),
            run=active.get("active_run") or "-",
            epoch=active.get("epoch") or "-",
            epochs=active.get("epochs") or "-",
            step=active.get("step") or "-",
            steps=active.get("steps") or "-",
            eta=active.get("eta") or "-",
            rate=active.get("rate") or "-",
        )
    )
    signal = report.get("current_training_signal") or {}
    lines.extend(["", "## Current Training Signal", ""])
    if signal.get("status") == "available":
        lines.append(
            "- Latest completed epoch `{epoch}`: val loss `{loss}`, risk acc `{risk}%`, friction acc `{friction}%`, "
            "raw coverage `{coverage}%`, raw width `{width}`.".format(
                epoch=signal.get("latest_epoch") or "-",
                loss=_fmt(signal.get("latest_val_loss")),
                risk=_pct_plain(signal.get("latest_risk_acc")),
                friction=_pct_plain(signal.get("latest_friction_acc")),
                coverage=_pct_plain(signal.get("latest_raw_coverage")),
                width=_fmt(signal.get("latest_raw_width")),
            )
        )
        lines.append(
            "- Previous-epoch deltas: val loss `{loss}`, risk acc `{risk}`, raw coverage `{coverage}`.".format(
                loss=_fmt(signal.get("val_loss_delta_vs_previous")),
                risk=_fmt_signed_pct(signal.get("risk_acc_delta_vs_previous")),
                coverage=_fmt_signed_pct(signal.get("raw_coverage_delta_vs_previous")),
            )
        )
        lines.append(
            "- Checkpoint policy: `{policy}`; best val-loss epoch `{loss_epoch}`, best safety epoch `{safety_epoch}`.".format(
                policy=signal.get("checkpoint_policy"),
                loss_epoch=signal.get("best_val_loss_epoch") or "-",
                safety_epoch=signal.get("best_safety_epoch") or "-",
            )
        )
    else:
        lines.append(f"- Training diagnosis status: `{signal.get('status', 'missing')}`.")
    lines.extend(["", "## Kill Now", ""])
    lines.extend(_table(report["kill_now"], ["item", "decision", "evidence", "action"]))
    lines.extend(["", "## Protect Or Keep", ""])
    lines.extend(_table(report["protect_or_keep"], ["item", "decision", "evidence", "action"]))
    lines.extend(["", "## Hard Problems", ""])
    lines.extend(_table(report["hard_problems"], ["problem", "status", "evidence", "fast_test"]))
    wave = report["next_wave"]
    lines.extend(
        [
            "",
            "## Next Wave",
            "",
            f"- Mode: `{wave['mode']}`.",
            f"- Reason: {wave['reason']}",
            f"- First wave: {_code_list(wave.get('first_wave'))}.",
            f"- Formal reference candidates: {_code_list(wave.get('formal_reference_candidates'))}.",
            f"- Held full stack: {_code_list(wave.get('held_full_stack'))}.",
        ]
    )
    if wave.get("command"):
        lines.extend(["", "Command when guard permits:", "", f"`{wave['command']}`"])
    lines.extend(["", "## Rapid Validation Rules", ""])
    for item in report["rapid_validation_rules"]:
        lines.append(f"- {item}")
    lines.extend(["", "## If The Next Wave Fails", ""])
    lines.extend(_table(report["route_if_next_wave_fails"], ["trigger", "route", "next_move"]))
    lines.append("")
    return "\n".join(lines)


def _table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["- No rows."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        vals = [_md_cell(row.get(col, "-")) for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _compact_text(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_evidence(value: Any) -> str:
    if isinstance(value, list):
        return _compact_text("; ".join(str(item) for item in value), 260)
    return _compact_text(str(value or "-"), 260)


def _md_cell(value: Any) -> str:
    return _compact_text(str(value).replace("|", "\\|"), 320)


def _pct(value: Any) -> str:
    try:
        if value is None:
            return "-"
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def _pct_plain(value: Any) -> str:
    try:
        if value is None:
            return "-"
        return f"{100.0 * float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_signed_pct(value: Any) -> str:
    try:
        if value is None:
            return "-"
        return f"{100.0 * float(value):+.2f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt(value: Any) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _code_list(items: Any) -> str:
    if not items:
        return "-"
    return ", ".join(f"`{item}`" for item in items)


if __name__ == "__main__":
    main()
