"""Write live remaining-experiment reports from the dashboard JSON.

This script is intentionally read-only with respect to experiments: it only
summarizes already generated protocol evidence. It keeps the human-facing
remaining-work reports synchronized with the authoritative dashboard.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _pct(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    try:
        return f"{float(value) * 100:.2f}"
    except (TypeError, ValueError):
        return default


def _num(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return default


def _list_or_dash(items: list[str] | None) -> str:
    if not items:
        return "-"
    return ", ".join(f"`{x}`" for x in items)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _watcher_lines(summary_dir: Path) -> list[str]:
    report = _load_json(summary_dir / "followup_watcher_report.json") or {}
    watchers = report.get("watchers") or []
    guard = _load_json(summary_dir / "gpu_scheduling_guard_report.json") or {}
    guard_watchers = guard.get("watchers") or []
    if not watchers:
        if guard_watchers:
            lines = [
                f"- GPU guard watcher/blocker processes visible: `{len(guard_watchers)}`.",
                "- Treat these as active blockers because they may launch or stop queued GPU work after their wait condition changes.",
                "",
                "| Kind | PID | Parent PID | Command |",
                "|---|---:|---:|---|",
            ]
            for watcher in guard_watchers:
                lines.append(
                    "| {kind} | {pid} | {parent} | `{cmd}` |".format(
                        kind=watcher.get("kind", "-"),
                        pid=watcher.get("pid", "-"),
                        parent=watcher.get("parent_pid", "-"),
                        cmd=_shorten(watcher.get("command_short"), 130),
                    )
                )
            return lines
        return [
            "- No follow-up watcher chain is currently visible.",
            "- Manual launch remains governed by the GPU scheduling guard.",
        ]

    lines = [
        f"- Follow-up watchers visible: `{len(watchers)}`.",
        "- These watchers may launch queued experiments after their wait PID exits, so manual GPU launches should remain disabled while the GPU guard is busy.",
        "",
        "| Kind | PID | Wait PID | Planned scripts |",
        "|---|---:|---:|---|",
    ]
    for watcher in watchers:
        scripts = watcher.get("script_commands") or []
        planned = "<br>".join(f"`{_shorten(script, 130)}`" for script in scripts[:3])
        if len(scripts) > 3:
            planned += f"<br>... +{len(scripts) - 3} more"
        lines.append(
            "| {kind} | {pid} | {wait_pid} | {planned} |".format(
                kind=watcher.get("kind", "-"),
                pid=watcher.get("pid", "-"),
                wait_pid=watcher.get("wait_pid") or "-",
                planned=planned or "-",
            )
        )
    return lines


def _shorten(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _active_line(dash: dict[str, Any]) -> str:
    active_rows = dash.get("active_rows") or []
    if not active_rows:
        return "No active training row is visible."
    row = active_rows[0]
    name = row.get("name", "unknown")
    epoch = row.get("active_epoch") or row.get("epoch")
    epochs = row.get("active_epochs") or row.get("epochs")
    step = row.get("active_step")
    steps = row.get("active_steps")
    tqdm = (dash.get("active_tqdm") or {}).get(name, {})
    eta = tqdm.get("eta")
    rate = tqdm.get("rate")
    phase = tqdm.get("phase")
    parts = [f"`{name}`"]
    if epoch and epochs:
        parts.append(f"epoch `{epoch}/{epochs}`")
    if step and steps:
        parts.append(f"step `{step}/{steps}`")
    if phase == "eval":
        eval_step = tqdm.get("eval_step")
        eval_steps = tqdm.get("eval_steps")
        if eval_step and eval_steps:
            parts.append(f"validation `{eval_step}/{eval_steps}`")
    if eta:
        parts.append(f"ETA `{eta}`")
    if rate:
        parts.append(f"rate `{rate}`")
    return ", ".join(parts) + "."


def _overlay_live_active_row(dash: dict[str, Any], summary_dir: Path) -> dict[str, Any]:
    active_report = (
        _load_json(summary_dir / "active_training_watch_report.json")
        or _load_json(summary_dir / "active_live_training_reports.json")
        or {}
    )
    active = active_report.get("active") or {}
    name = active.get("name")
    if not name:
        return dash

    out = dict(dash)
    rows = list(out.get("active_rows") or [])
    base = next((row for row in rows if row.get("name") == name), rows[0] if rows else {})
    merged = {
        **base,
        "name": name,
        "active_epoch": active.get("epoch") or base.get("active_epoch") or base.get("epoch"),
        "active_epochs": active.get("epochs") or base.get("active_epochs") or base.get("epochs"),
        "active_step": active.get("step") or base.get("active_step"),
        "active_steps": active.get("steps") or base.get("active_steps"),
    }
    out["active_rows"] = [merged] + [row for row in rows if row.get("name") != name]
    active_tqdm = dict(out.get("active_tqdm") or {})
    active_tqdm[name] = {
        **(active_tqdm.get(name) or {}),
        "phase": active.get("phase") or (active_tqdm.get(name) or {}).get("phase"),
        "eta": active.get("eval_tqdm_eta") or active.get("eta") or active.get("tqdm_eta") or (active_tqdm.get(name) or {}).get("eta"),
        "rate": active.get("eval_tqdm_rate") or active.get("rate") or active.get("tqdm_rate") or (active_tqdm.get(name) or {}).get("rate"),
        "eval_step": active.get("eval_step") or (active_tqdm.get(name) or {}).get("eval_step"),
        "eval_steps": active.get("eval_steps") or (active_tqdm.get(name) or {}).get("eval_steps"),
    }
    out["active_tqdm"] = active_tqdm
    out["report_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out


def _overlay_live_training_metrics(dash: dict[str, Any], summary_dir: Path) -> dict[str, Any]:
    active_report = (
        _load_json(summary_dir / "active_training_watch_report.json")
        or _load_json(summary_dir / "active_live_training_reports.json")
        or {}
    )
    active = active_report.get("active") or {}
    name = active.get("name")
    if not name:
        return dash
    diagnosis = _load_json(summary_dir / f"{name}_training_diagnosis.json") or {}
    latest = diagnosis.get("latest_epoch") or {}
    signals = diagnosis.get("signals") or {}
    if not latest:
        return dash
    out = dict(dash)
    out["live_training"] = {
        **(out.get("live_training") or {}),
        "latest_completed_epoch": latest.get("epoch"),
        "latest_val_loss": latest.get("val_loss"),
        "latest_val_risk_acc": latest.get("val_risk_acc"),
        "latest_val_friction_acc": latest.get("val_friction_acc"),
        "latest_raw_coverage": latest.get("raw_coverage"),
        "latest_raw_width": latest.get("raw_width"),
        "previous_delta_val_loss": signals.get("val_loss_delta_vs_previous"),
        "previous_delta_val_risk_acc": signals.get("risk_acc_delta_vs_previous"),
        "previous_delta_raw_coverage": signals.get("raw_coverage_delta_vs_previous"),
        "best_val_loss_epoch": (diagnosis.get("best_val_loss_epoch") or {}).get("epoch"),
        "best_safety_proxy_epoch": (diagnosis.get("best_safety_proxy_epoch") or {}).get("epoch"),
    }
    return out


def _overlay_fresh_contract_and_completeness(dash: dict[str, Any], summary_dir: Path) -> dict[str, Any]:
    """Prefer freshly generated contract/completeness JSON over dashboard cache."""
    out = dict(dash)

    artifact = _load_json(summary_dir / "artifact_contract_report.json") or {}
    if isinstance(artifact, dict) and artifact:
        out["artifact_contract"] = {
            **(out.get("artifact_contract") or {}),
            "verdict": artifact.get("verdict"),
            "num_runs": artifact.get("num_runs"),
            "num_contract_complete": artifact.get("num_contract_complete"),
            "num_contract_incomplete": artifact.get("num_contract_incomplete"),
            "num_invalid_complete_like": artifact.get("num_invalid_complete_like"),
            "num_stale_rows": artifact.get("num_stale_rows"),
        }
        rows = artifact.get("rows") or []
        if rows:
            out["progress_counts"] = {
                "complete": sum(1 for row in rows if row.get("progress_status") == "complete"),
                "running_or_partial": sum(1 for row in rows if row.get("progress_status") == "running_or_partial"),
                "missing": sum(1 for row in rows if row.get("progress_status") == "missing"),
            }

    completeness = _load_json(summary_dir / "protocol_completeness.json") or {}
    if isinstance(completeness, dict) and completeness.get("group_status"):
        out["group_status"] = completeness["group_status"]

    return out


def _gpu_line(dash: dict[str, Any]) -> str:
    gpu = (dash.get("system") or {}).get("gpu") or {}
    if not gpu:
        return "GPU status is unavailable."
    return (
        f"{gpu.get('name', 'GPU')}, util `{gpu.get('utilization_percent', '-') }%`, "
        f"memory `{gpu.get('memory_used_mb', '-')}/{gpu.get('memory_total_mb', '-')} MB`, "
        f"temp `{gpu.get('temperature_c', '-')} C`."
    )


def _p0_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| Method | friction F1 | risk F1 | low-friction recall | calibrated coverage | worst dataset F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        out.append(
            "| {method} | {friction} | {risk} | {low} | {coverage} | {worst} |".format(
                method=row.get("method", "-"),
                friction=_pct(row.get("friction_f1")),
                risk=_pct(row.get("risk_f1")),
                low=_pct(row.get("low_friction_recall")),
                coverage=_pct(row.get("calibrated_coverage")),
                worst=_pct(row.get("worst_dataset_f1")),
            )
        )
    return out


def _missing_group_lines(dash: dict[str, Any]) -> list[str]:
    groups = dash.get("group_status") or {}
    order = [
        ("single_dataset_faf", "Single-dataset FAF"),
        ("single_dataset_baselines", "Matched ConvNeXt baselines"),
        ("p1_candidates", "P1/P2/P3 candidates"),
        ("final_method_lodo", "Final lean LODO"),
        ("final_method_single_dataset", "Final lean single-dataset"),
    ]
    lines: list[str] = []
    for key, title in order:
        item = groups.get(key) or {}
        complete = item.get("complete")
        missing = item.get("missing_runs") or []
        status = "complete" if complete else "incomplete"
        lines.append(f"- {title}: `{status}`; missing {_list_or_dash(missing)}.")
    return lines


def _group_missing(dash: dict[str, Any], keys: list[str]) -> list[str]:
    groups = dash.get("group_status") or {}
    missing: list[str] = []
    for key in keys:
        item = groups.get(key) or {}
        missing.extend(item.get("missing_runs") or [])
    return missing


def _gate_rows(dash: dict[str, Any]) -> list[str]:
    rows = [
        "| Gate | Current evidence | Paper-use status | Next action |",
        "|---|---|---|---|",
    ]
    readiness = dash.get("readiness") or {}
    blocks = readiness.get("blocking_gates") or []
    warnings = readiness.get("warning_gates") or []
    gate_map = {g.get("name"): g for g in blocks + warnings}

    def add(
        name: str,
        evidence: str,
        status: str,
        action: str,
        missing_override: list[str] | None = None,
    ) -> None:
        gate = gate_map.get(name) or {}
        missing = missing_override if missing_override is not None else gate.get("missing")
        if missing:
            evidence = f"{evidence} Missing: {_list_or_dash(missing)}."
        rows.append(f"| `{name}` | {evidence} | {status} | {action} |")

    add(
        "fair_single_dataset_complete",
        "Matched FAF vs ConvNeXt evidence is incomplete.",
        "No SOTA-style numerical claim yet.",
        "Finish current priority queue.",
        _group_missing(dash, ["single_dataset_faf", "single_dataset_baselines"]),
    )
    add(
        "dataset_shortcut",
        "Dataset-ID probes remain high for completed rows.",
        "Shortcut mitigation is not yet supported.",
        "Run Fourier/style/color/ROI/domain-adapter candidates.",
    )
    add(
        "heldout_roadsaw_risk_f1",
        "Base LODO RoadSaW is a severe OOD failure.",
        "Use as failure analysis only.",
        "Run final lean LODO after candidate selection.",
    )
    add(
        "conditional_interval_report",
        "Conditional undercoverage cells remain on the watchlist.",
        "Intervals must be reported with coverage and width.",
        "Run v12/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24 and conditional calibration.",
    )
    add(
        "final_method_complete",
        "Final lean route has not completed matched evidence.",
        "Do not freeze final paper method.",
        "Select by predeclared safety/generalization score.",
        _group_missing(dash, ["final_method_lodo", "final_method_single_dataset"]),
    )
    return rows


def _fail_fast_lines(summary_dir: Path) -> list[str]:
    report = _load_json(summary_dir / "fail_fast_exploration_report.json") or {}
    if not report:
        return [
            "- Fail-fast report is not generated yet.",
            "- Run `scripts/write_fail_fast_exploration_report.py` after the current summaries are refreshed.",
        ]
    policy = report.get("formal_policy") or {}
    kill = report.get("kill_or_downgrade") or []
    keep = report.get("protect_or_conditional_keep") or []
    lines = [
        f"- Verdict: `{report.get('verdict', '-')}`.",
        f"- Policy: `{policy.get('verdict', '-')}`.",
        f"- Formal promoted/fallback candidates: {_list_or_dash(policy.get('promoted_or_fallback') or [])}.",
        f"- Fast-screen first wave: {_list_or_dash(policy.get('fast_screen_first_wave') or [])}.",
        f"- Held until screen evidence: {_list_or_dash(policy.get('held_until_screen') or [])}.",
        f"- Full-stack routes held until screen evidence: {_list_or_dash(policy.get('full_stack_held_until_screen') or [])}.",
    ]
    if kill:
        killed = [f"{row.get('item', '-')}: `{row.get('decision', '-')}`" for row in kill]
        lines.append("- Kill/downgrade now: " + "; ".join(killed) + ".")
    if keep:
        kept = [f"{row.get('item', '-')}: `{row.get('decision', '-')}`" for row in keep]
        lines.append("- Protect/conditional keep: " + "; ".join(kept) + ".")
    lines.append(
        "- Future candidate scheduling should use fail-fast defaults or explicit `--only`; do not run every v6-v24 route unless reproducing the exhaustive protocol."
    )
    return lines


def write_remaining_report(dash: dict[str, Any], out_path: Path) -> None:
    counts = dash.get("progress_counts") or {}
    live = dash.get("live_training") or {}
    lodo = dash.get("lodo_generalization") or {}
    artifact = dash.get("artifact_contract") or {}
    runtime = dash.get("runtime_guard") or {}
    p0_rows = dash.get("core_ablation") or []
    final_sel = dash.get("final_method_selection") or {}

    lines = [
        "# Current Remaining Experiments And Next Actions",
        "",
        f"Generated: {dash.get('report_generated_at') or dash.get('generated_at', '-')}",
        "",
        "## Current Execution",
        "",
        f"- Progress counts: complete `{counts.get('complete', 0)}`, running/partial `{counts.get('running_or_partial', 0)}`, missing `{counts.get('missing', 0)}`.",
        f"- Active row: {_active_line(dash)}",
        f"- GPU: {_gpu_line(dash)}",
        f"- Runtime guard: `{runtime.get('verdict', '-')}`.",
        f"- Artifact contract: `{artifact.get('verdict', '-')}`, contract-complete `{artifact.get('num_contract_complete', '-')}/{artifact.get('num_runs', '-')}`.",
    ]

    if live:
        lines.extend(
            [
                f"- Latest completed epoch: `{live.get('latest_completed_epoch', '-')}` with val loss `{_num(live.get('latest_val_loss'))}`, risk acc `{_pct(live.get('latest_val_risk_acc'))}%`, friction acc `{_pct(live.get('latest_val_friction_acc'))}%`, raw coverage `{_pct(live.get('latest_raw_coverage'))}%`, raw width `{_num(live.get('latest_raw_width'))}`.",
                f"- Previous epoch deltas: val loss `{_num(live.get('previous_delta_val_loss'))}`, risk acc `{_pct(live.get('previous_delta_val_risk_acc'))}%`, raw coverage `{_pct(live.get('previous_delta_raw_coverage'))}%`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Completed Evidence",
            "",
            "- P0 ablation is complete.",
            "- Base LODO is complete, but it shows severe cross-dataset transfer failure.",
            "- Dataset integrity, image-style audit, public friction-anchor audit, and config-to-code trace have passed.",
            "",
        ]
    )
    lines.extend(_p0_table(p0_rows))

    lines.extend(
        [
            "",
            "## LODO Readout",
            "",
            f"- Overall verdict: `{lodo.get('verdict', '-')}`.",
            f"- Held-out RoadSaW verdict: `{lodo.get('roadsaw_verdict', '-')}`.",
            f"- Held-out RoadSaW risk F1 `{_pct(lodo.get('roadsaw_risk_f1'))}%`, friction F1 `{_pct(lodo.get('roadsaw_friction_f1'))}%`, calibrated coverage `{_pct(lodo.get('roadsaw_calibrated_coverage'))}%`.",
            "- Claim boundary: LODO currently supports failure analysis and algorithm motivation, not an OOD-success claim.",
            "",
            "## Remaining Experiments",
            "",
        ]
    )
    lines.extend(_missing_group_lines(dash))

    lines.extend(
        [
            "",
            "## Automatic Follow-up Chain",
            "",
        ]
    )
    lines.extend(_watcher_lines(out_path.parent))

    lines.extend(
        [
            "",
            "## Main Problems",
            "",
            "- The current Full FAF stack is not the final method candidate; P0 supports the lean PhysicsTexture-centered route.",
            "- Dataset shortcut remains high, so the method cannot yet claim that it has learned dataset-invariant friction cues.",
            "- Held-out RoadSaW transfer is very weak, so cross-dataset generalization is not supported.",
            "- RoadSaW damp/wet/very-wet behavior and low-friction recall remain key stress cases.",
            "- Conditional interval undercoverage remains serious; coverage must be reported together with interval width.",
            "- Matched ConvNeXt baselines are incomplete, so fair SOTA-style numerical claims are not yet allowed.",
            "",
            "## Current Method Decision",
            "",
            f"- Final selection verdict: `{final_sel.get('verdict', '-')}`.",
            "- Keep: `PhysicsTexture`.",
            "- Keep provisionally: `EvidenceField`, pending ROI/failure-map evidence.",
            "- Rework or remove unless rescued: `FrictionSet`, `DG losses`, `Full fusion`.",
            "",
            "## Fail-Fast Exploration Decision",
            "",
        ]
    )
    lines.extend(_fail_fast_lines(out_path.parent))
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "1. Let the active official queue continue; do not start a duplicate GPU worker.",
            "2. Finish remaining single-dataset FAF and matched ConvNeXt rows.",
            "3. Refresh artifact contract, protocol completeness, bootstrap, interval, and claim-evidence reports after each completed run.",
            "4. Run fast-screen or fail-fast-promoted shortcut, wetness, ROI, quality, and interval candidates when the GPU scheduling guard permits.",
            "5. Freeze the final lean road-ROI safety method only after candidate, LODO, and matched-baseline evidence exists.",
            "6. Run RSCD per-day and DINOv2/foundation probes only after the official queue and watcher chain are idle.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_gate_matrix(dash: dict[str, Any], out_path: Path) -> None:
    watcher_report = _load_json(out_path.parent / "followup_watcher_report.json") or {}
    watchers = watcher_report.get("watchers") or []
    guard_report = _load_json(out_path.parent / "gpu_scheduling_guard_report.json") or {}
    guard_watchers = guard_report.get("watchers") or []
    visible_watchers = watchers or guard_watchers
    lines = [
        "# Remaining Experiment Gate Matrix",
        "",
        f"Generated: {dash.get('report_generated_at') or dash.get('generated_at', '-')}",
        "",
        "## Current Runtime State",
        "",
        f"- Active row: {_active_line(dash)}",
        f"- GPU: {_gpu_line(dash)}",
        f"- Runtime guard: `{(dash.get('runtime_guard') or {}).get('verdict', '-')}`.",
        f"- Readiness verdict: `{(dash.get('readiness') or {}).get('verdict', '-')}`.",
        f"- Follow-up/GPU-guard watchers visible: `{len(visible_watchers)}`.",
        "",
        "## Evidence Already Strong",
        "",
        "| Requirement | Current evidence | Paper-use status |",
        "|---|---|---|",
        "| Dataset integrity | Local RSCD/RoadSaW/RoadSC manifests and image paths are audited. | Usable as dataset integrity evidence. |",
        "| Dataset-style mismatch | RSCD, RoadSaW, and RoadSC native image styles differ. | Must be reported as shortcut risk. |",
        "| P0 ablation | Six core rows are complete. | Usable as module-retention evidence. |",
        "| Base LODO | Three held-out datasets are complete and weak. | Failure analysis only, not OOD success. |",
        "| Friction anchors | Public weak friction interval anchors pass audit. | Supports weak friction-affordance intervals, not measured friction. |",
        "",
        "## Gates Not Yet Passed",
        "",
    ]
    lines.extend(_gate_rows(dash))
    lines.extend(
        [
            "",
            "## Immediate Execution Order",
            "",
            "1. Let the active official training job finish.",
            "2. Complete remaining single-dataset FAF rows.",
            "3. Complete matched ConvNeXt baselines.",
            "4. Refresh all postprocess and claim-evidence reports.",
            "5. Run fast-screen/candidate rows only when the GPU guard is idle.",
            "6. Run final lean LODO and matched single-dataset rows after candidate selection.",
            "7. Run RSCD per-day and foundation probes as separate benchmark packages after the watcher chain is idle.",
            "",
            "## Claim Wording Boundary",
            "",
            "The current project estimates visual-evidence-conditioned friction-affordance intervals from public weak labels. It must not be described as direct measured tire-road friction coefficient estimation unless synchronized friction-meter or vehicle-dynamics ground truth is available.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", default="reports/paper_protocol_summary")
    parser.add_argument("--dashboard-json", default=None)
    parser.add_argument("--remaining-md", default="current_remaining_experiments_and_next_actions.md")
    parser.add_argument("--gate-md", default=None)
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    dash_path = Path(args.dashboard_json) if args.dashboard_json else summary_dir / "experiment_status_dashboard.json"
    dash = json.loads(dash_path.read_text(encoding="utf-8"))
    dash = _overlay_fresh_contract_and_completeness(dash, summary_dir)
    dash = _overlay_live_active_row(dash, summary_dir)
    dash = _overlay_live_training_metrics(dash, summary_dir)

    generated = str(dash.get("generated_at", ""))
    date_token = generated[:10].replace("-", "") if generated else "current"
    remaining_path = summary_dir / args.remaining_md
    gate_name = args.gate_md or f"remaining_experiment_gate_matrix_{date_token}.md"
    gate_path = summary_dir / gate_name

    write_remaining_report(dash, remaining_path)
    write_gate_matrix(dash, gate_path)
    print(f"Wrote {remaining_path}")
    print(f"Wrote {gate_path}")


if __name__ == "__main__":
    main()
