from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "objective_completion_audit.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "objective_completion_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    md = render_markdown(report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)


def build_report(summary_dir: Path) -> dict[str, Any]:
    completeness = _load_json(summary_dir / "protocol_completeness.json") or {}
    queue = _load_json(summary_dir / "queue_recovery_report.json") or {}
    dashboard = _load_json(summary_dir / "experiment_status_dashboard.json") or {}
    active_live = (
        _load_json(summary_dir / "active_training_watch_report.json")
        or _load_json(summary_dir / "active_live_training_reports.json")
        or {}
    )
    gate = _load_json(summary_dir / "topvenue_readiness_gate.json") or {}
    dataset_integrity = _load_json(summary_dir / "dataset_integrity_view_audit.json") or {}
    dataset_style = _load_json(summary_dir / "dataset_image_style_audit.json") or {}
    dataset_view = _load_text(summary_dir / "dataset_view_source_evidence_report.md") or ""
    route_decision = _load_text(summary_dir / "dataset_route_decision_and_fair_benchmark_plan.md") or ""
    live_route = _load_json(summary_dir / "live_research_route_update.json") or {}
    interval_sources = _load_json(summary_dir / "friction_interval_source_audit.json") or {}
    direct_visual = _load_json(summary_dir / "direct_visual_friction_report.json") or {}
    p0_claim = _load_json(summary_dir / "p0_claim_report.json") or {}
    module_retention = _load_json(summary_dir / "module_retention_report.json") or {}
    external_benchmark = _load_json(summary_dir / "external_benchmark_report.json") or {}
    gpu_protocol = _load_json(summary_dir / "gpu_protocol_audit.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    segmentation_transfer_config = _load_json(summary_dir / "segmentation_transfer_config_audit.json") or {}

    requirements = {row.get("name"): row for row in completeness.get("requirements", []) if isinstance(row, dict)}
    rows = _requirement_rows(
        requirements=requirements,
        queue=queue,
        gate=gate,
        dataset_integrity=dataset_integrity,
        dataset_style=dataset_style,
        dataset_view=dataset_view,
        route_decision=route_decision,
        live_route=live_route,
        interval_sources=interval_sources,
        direct_visual=direct_visual,
        p0_claim=p0_claim,
        module_retention=module_retention,
        external_benchmark=external_benchmark,
        gpu_protocol=gpu_protocol,
        shortcut=shortcut,
        segmentation_transfer_config=segmentation_transfer_config,
    )
    counts = {
        "complete": sum(1 for row in rows if row["status"] == "complete"),
        "partial": sum(1 for row in rows if row["status"] == "partial"),
        "incomplete": sum(1 for row in rows if row["status"] == "incomplete"),
        "configured": sum(1 for row in rows if row["status"] == "configured"),
        "not_applicable_now": sum(1 for row in rows if row["status"] == "not_applicable_now"),
    }
    current = _current_execution(queue, dashboard, gate, active_live)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "scope": "visual tire-road friction-affordance research objective under public/reusable datasets",
        "claim_boundary": (
            "The project estimates weak visual friction-affordance intervals from public road-condition labels; "
            "it does not yet prove synchronized measured tire-road friction accuracy."
        ),
        "current_execution": current,
        "readiness": {
            "verdict": gate.get("verdict", "missing"),
            "blocks": gate.get("num_blocks"),
            "warnings": gate.get("num_warnings"),
        },
        "counts": counts,
        "requirements": rows,
        "allowed_claims": _allowed_claims(rows, gate),
        "disallowed_claims": _disallowed_claims(rows, gate),
        "next_order": _next_order(requirements, queue, rows),
    }


def _requirement_rows(
    *,
    requirements: dict[str, dict[str, Any]],
    queue: dict[str, Any],
    gate: dict[str, Any],
    dataset_integrity: dict[str, Any],
    dataset_style: dict[str, Any],
    dataset_view: str,
    route_decision: str,
    live_route: dict[str, Any],
    interval_sources: dict[str, Any],
    direct_visual: dict[str, Any],
    p0_claim: dict[str, Any],
    module_retention: dict[str, Any],
    external_benchmark: dict[str, Any],
    gpu_protocol: dict[str, Any],
    shortcut: dict[str, Any],
    segmentation_transfer_config: dict[str, Any],
) -> list[dict[str, Any]]:
    integrity = dataset_integrity.get("path_checks", {}) if isinstance(dataset_integrity, dict) else {}
    dataset_rows = dataset_integrity.get("dataset_rows", {}) if isinstance(dataset_integrity, dict) else {}
    cross_dataset = dataset_integrity.get("cross_dataset", {}) if isinstance(dataset_integrity, dict) else {}
    roadsaw = dataset_rows.get("roadsaw", {}) if isinstance(dataset_rows.get("roadsaw", {}), dict) else {}
    roadsc = dataset_rows.get("roadsc", {}) if isinstance(dataset_rows.get("roadsc", {}), dict) else {}
    rscd = dataset_rows.get("rscd", {}) if isinstance(dataset_rows.get("rscd", {}), dict) else {}
    style_signals = dataset_style.get("cross_dataset_signals", {}) if isinstance(dataset_style, dict) else {}
    view_lower = dataset_view.lower()
    route_lower = route_decision.lower()
    live_route_text = json.dumps(live_route, ensure_ascii=False).lower() if isinstance(live_route, dict) else ""
    gates = {item.get("name"): item for item in gate.get("gates", []) if isinstance(item, dict)}
    rows: list[dict[str, Any]] = []
    rows.append(
        _row(
            "Verify local datasets are complete and readable",
            "complete"
            if int(integrity.get("missing_checked_paths", 1) or 0) == 0
            and int(integrity.get("checked_unique_paths", 0) or 0) > 0
            else "incomplete",
            (
                "{existing}/{checked} audited unique paths exist; missing checked paths {missing}; "
                "total manifest unique paths {total}; mode {mode}."
            ).format(
                existing=integrity.get("existing_checked_paths", "-"),
                checked=integrity.get("checked_unique_paths", "-"),
                missing=integrity.get("missing_checked_paths", "-"),
                total=integrity.get("total_unique_paths", "-"),
                mode=integrity.get("check_mode", "-"),
            ),
            "Regenerate integrity audit if manifests change.",
        )
    )
    rows.append(
        _row(
            "Explain why RoadSaW contains near-white images",
            "complete" if (roadsaw.get("near_white") or {}).get("count", 0) > 0 and roadsaw.get("decode_errors", 1) == 0 else "incomplete",
            "RoadSaW near-white count {count} ({rate:.2%}); top classes include {classes}; decode errors {decode_errors}.".format(
                count=(roadsaw.get("near_white") or {}).get("count", 0),
                rate=float((roadsaw.get("near_white") or {}).get("rate", 0.0) or 0.0),
                classes=", ".join(list(((roadsaw.get("near_white") or {}).get("by_class") or {}).keys())[:4]) or "-",
                decode_errors=roadsaw.get("decode_errors", "-"),
            ),
            "Keep near-white vs normal-quality RoadSaW as a quality-slice metric.",
        )
    )
    rows.append(
        _row(
            "Decide whether RSCD is left/right wheel-front imagery",
            "complete" if (rscd.get("view_inference") or {}).get("inference") == "local_patch_or_narrow_forward_crop" else "partial",
            "{inference}; {caution}".format(
                inference=(rscd.get("view_inference") or {}).get("inference", "missing"),
                caution=(rscd.get("view_inference") or {}).get("caution", "No wheel-specific source proof was found."),
            ),
            "Use only road-surface patch/crop wording in papers and slides.",
        )
    )
    rows.append(
        _row(
            "Explain why RSCD differs from RoadSaW/RoadSC",
            "complete" if rscd.get("dimension_top") and roadsaw.get("dimension_top") and roadsc.get("dimension_top") else "partial",
            "Dominant sizes: RSCD {rscd_dims}, RoadSaW {roadsaw_dims}, RoadSC {roadsc_dims}; aspect span {aspect_span}; width span {width_span}.".format(
                rscd_dims=rscd.get("dimension_top", {}),
                roadsaw_dims=roadsaw.get("dimension_top", {}),
                roadsc_dims=roadsc.get("dimension_top", {}),
                aspect_span=(cross_dataset.get("aspect_span") or (style_signals.get("aspect_median_range") or {}).get("span") or "-"),
                width_span=(cross_dataset.get("width_span") or (style_signals.get("width_median_range") or {}).get("span") or "-"),
            ),
            "Use the visual gap as domain-generalization motivation.",
        )
    )
    rows.append(
        _row(
            "Decide whether to use RSCD only, other datasets only, or all datasets",
            "complete" if (
                "hierarchical" in route_lower
                or "not naive pooling" in route_lower
                or "do not treat rscd, roadsaw, and roadsc as one homogeneous benchmark" in live_route_text
            ) else "partial",
            "Route decision report selects single-dataset fair tables plus LODO stress tests, not a homogeneous pooled SOTA table.",
            "Revisit only after matched baselines and LODO complete.",
        )
    )
    rows.append(
        _row(
            "Find public/reasonable friction interval anchors",
            "complete" if interval_sources.get("verdict") == "pass" else "incomplete",
            f"Friction source audit verdict {interval_sources.get('verdict', 'missing')}; anchors {len(interval_sources.get('rows', [])) if isinstance(interval_sources.get('rows'), list) else '-'}.\n",
            "Keep weak-affordance wording; do not call anchors measured image-level friction labels.",
        )
    )
    direct_status = direct_visual.get("status") or "missing"
    rows.append(
        _row(
            "Find direct image-to-friction benchmark papers/datasets",
            "partial" if external_benchmark or direct_status in {"pending", "complete"} else "incomplete",
            "External benchmark report maps {sources} public sources and {comparisons} comparability rows; ExtremeRoad direct route status is {direct_status}.".format(
                sources=len(external_benchmark.get("public_sources", []) or []) if isinstance(external_benchmark, dict) else "-",
                comparisons=len(external_benchmark.get("comparability_matrix", []) or []) if isinstance(external_benchmark, dict) else "-",
                direct_status=direct_status,
            ),
            "Finish ExtremeRoad runs; use WCamNet only as context unless reproducible data/splits become available.",
        )
    )
    rows.append(
        _row(
            "Download needed public datasets",
            "partial" if int(integrity.get("missing_checked_paths", 1) or 0) == 0 else "incomplete",
            "RSCD/RoadSaW/RoadSC are locally audited ({rows} rows, {missing} missing); optional direct-friction datasets remain separate.".format(
                rows=integrity.get("total_rows", "-"),
                missing=integrity.get("missing_checked_paths", "-"),
            ),
            "Only add future datasets after license, files, splits, and labels are reproducible.",
        )
    )
    rows.append(
        _row(
            "Create/use Conda GPU experiment environment",
            "complete" if gpu_protocol.get("verdict") == "pass" else "partial",
            f"GPU protocol verdict {gpu_protocol.get('verdict', 'missing')}; CUDA {gpu_protocol.get('torch', {}).get('cuda_available', '-')}.",
            "Continue monitoring VRAM/OOM while queue runs.",
        )
    )
    rows.append(_requirement_row("P0 full ablation table", requirements.get("p0_ablation_complete"), "P0 table is the core innovation evidence."))
    rows.append(
        _row(
            "P0 module-retention decision",
            "partial" if module_retention.get("verdict") else "incomplete",
            f"Module retention verdict {module_retention.get('verdict', 'missing')}; hard context {module_retention.get('missing_context', {})}.",
            "Freeze keep/remove/merge only after LODO, candidates, and fair baselines complete.",
        )
    )
    rows.append(
        _requirement_row(
            "LODO experiments",
            requirements.get("lodo_complete"),
            "Use completed LODO as failure-analysis/generalization evidence; rerun only if manifests, labels, or final method change.",
        )
    )
    rows.append(
        _row(
            "Re-run audit after LODO",
            "complete" if requirements.get("lodo_complete", {}).get("status") == "complete" and gate.get("verdict") else "incomplete",
            f"Readiness verdict {gate.get('verdict', 'missing')}; blocks {gate.get('num_blocks', '-')}; LODO requirement {requirements.get('lodo_complete', {}).get('status', 'missing')}.",
            "The audit has been rerun, but remaining blocks must be cleared before final claims.",
        )
    )
    rows.append(_fair_single_dataset_row(queue, requirements.get("fair_single_dataset_complete")))
    rows.append(
        _row(
            "RSCD SOTA-style fair comparison",
            "configured" if Path("scripts/run_rscd_surface_classification.py").exists() else "incomplete",
            "RSCD-27 original class-label protocol script exists; formal results are pending.",
            "Run fast/formal RSCD-27 rows and compare only under matching labels/splits/metrics.",
        )
    )
    rows.append(_requirement_row("P1 shortcut/domain-robust candidates", requirements.get("candidate_path_complete"), "Needed to test shortcut mitigation candidates v6-v25."))
    rows.append(
        _row(
            "P2 EvidenceField improvement",
            "partial"
            if (
                gates.get("evidence_failure_report", {}).get("level") == "pass"
                or segmentation_transfer_config.get("verdict") == "pass"
            )
            else "incomplete",
            (
                "Evidence/failure reports exist for completed rows; "
                "mask-supervised segmentation-transfer config verdict {seg_verdict}, pseudo-road loss {pseudo_loss}. "
                "ROI/final evidence candidate metrics are still pending."
            ).format(
                seg_verdict=segmentation_transfer_config.get("verdict", "missing"),
                pseudo_loss=(
                    (segmentation_transfer_config.get("batch_report") or {}).get(
                        "loss_evidence_attention_pseudo_road",
                        "-",
                    )
                    if isinstance(segmentation_transfer_config.get("batch_report"), dict)
                    else "-"
                ),
            ),
            "Run v8/v10/v12/v14-v25, CLIPSeg/SAM mask-supervised candidates, and final rows; select success/failure evidence maps.",
        )
    )
    rows.append(
        _row(
            "P3 interval-quality improvement",
            "partial" if gates.get("conditional_interval_report", {}).get("level") in {"pass", "warn"} else "incomplete",
            "Conditional interval reports exist, but raw/conditional coverage remains a risk.",
            "Run coverage-aware candidates and report coverage-width tradeoffs.",
        )
    )
    rows.append(_requirement_row("Final lean method", requirements.get("final_method_complete"), "Final paper method needs LODO and matched single-dataset evidence."))
    rows.append(
        _row(
            "Demonstrate learned road/friction evidence rather than dataset identity",
            "incomplete" if shortcut.get("verdict") == "warn" else "partial",
            f"Dataset shortcut verdict {shortcut.get('verdict', 'missing')}; high-shortcut rows {shortcut.get('num_high_shortcut', '-')}.",
            "Use v6-v25/final rows to reduce dataset-ID predictability while preserving safety metrics.",
        )
    )
    rows.append(
        _row(
            "Produce top-venue-ready final algorithm and results",
            "incomplete" if gate.get("verdict") != "ready_for_strict_paper_claims" else "complete",
            f"Readiness verdict {gate.get('verdict', 'missing')}; blocks {gate.get('num_blocks', '-')}; warnings {gate.get('num_warnings', '-')}.",
            "Complete experiments, prune modules, rerun audits, then write final claim-limited paper evidence.",
        )
    )
    return rows


def _requirement_row(title: str, req: dict[str, Any] | None, remaining: str) -> dict[str, Any]:
    status = (req or {}).get("status", "missing")
    mapped = "complete" if status == "complete" else "incomplete"
    missing = ", ".join((req or {}).get("missing", []) or []) or "-"
    return _row(title, mapped, f"Protocol completeness status {status}; missing {missing}.", remaining)


def _fair_single_dataset_row(queue: dict[str, Any], req: dict[str, Any] | None) -> dict[str, Any]:
    required = [
        "single_roadsaw_full_faf",
        "single_rscd_full_faf",
        "single_roadsc_full_faf",
        "baseline_single_roadsaw_global_convnext",
        "baseline_single_rscd_global_convnext",
        "baseline_single_roadsc_global_convnext",
    ]
    statuses = _queue_status_map(queue)
    missing_or_incomplete = [run for run in required if statuses.get(run) != "complete"]
    if not statuses:
        status = (req or {}).get("status", "missing")
        missing_or_incomplete = list((req or {}).get("missing", []) or [])
        mapped = "complete" if status == "complete" else "incomplete"
        evidence = "Protocol completeness status {status}; missing {missing}.".format(
            status=status,
            missing=", ".join(missing_or_incomplete) or "-",
        )
    else:
        mapped = "complete" if not missing_or_incomplete else "incomplete"
        evidence = "Latest queue statuses: {statuses}. Missing/incomplete: {missing}.".format(
            statuses=", ".join(f"{run}={statuses.get(run, 'missing')}" for run in required),
            missing=", ".join(missing_or_incomplete) or "-",
        )
    return _row(
        "Matched single-dataset FAF vs ConvNeXt baselines",
        mapped,
        evidence,
        "Needed for fair numerical paper comparison.",
    )


def _queue_status_map(queue: dict[str, Any]) -> dict[str, str]:
    rows = queue.get("queue_order") if isinstance(queue, dict) else None
    if not isinstance(rows, list):
        return {}
    out: dict[str, str] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("name"):
            out[str(row["name"])] = str(row.get("status") or "missing")
    return out


def _row(title: str, status: str, evidence: str, remaining: str) -> dict[str, str]:
    return {"requirement": title, "status": status, "evidence": evidence, "remaining_work": remaining}


def _current_execution(
    queue: dict[str, Any],
    dashboard: dict[str, Any],
    gate: dict[str, Any],
    active_live: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_report = (active_live or {}).get("active") or {}
    dashboard_active = dashboard.get("active_rows") if isinstance(dashboard.get("active_rows"), list) else []
    queue_active = _queue_active_rows(queue)
    active = dashboard_active[0] if dashboard_active else (queue_active[0] if queue_active else queue.get("next_incomplete", {}))
    if queue_active:
        active = {**active, **queue_active[0]}
    if active_report.get("name"):
        same_run = not active.get("name") or str(active.get("name")) == str(active_report.get("name"))
        if same_run and not queue_active:
            active = {
                **active,
                "name": active_report.get("name"),
                "active_epoch": active_report.get("epoch") or active.get("active_epoch") or active.get("epoch"),
                "active_epochs": active_report.get("epochs") or active.get("active_epochs") or active.get("epochs"),
                "active_step": active_report.get("step") or active.get("active_step"),
                "active_steps": active_report.get("steps") or active.get("active_steps"),
                "status": active.get("status") or "running_or_partial",
            }
    queue_summary = queue.get("summary") or {}
    return {
        "active_run": active.get("name"),
        "status": active.get("status"),
        "epoch": active.get("active_epoch") or active.get("epoch"),
        "epochs": active.get("active_epochs") or active.get("epochs"),
        "step": active.get("active_step"),
        "steps": active.get("active_steps"),
        "queue_complete": queue.get("num_complete") or queue_summary.get("complete"),
        "queue_total": queue.get("num_total") or queue_summary.get("total"),
        "readiness": gate.get("verdict", "missing"),
    }


def _queue_active_rows(queue: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(queue, dict):
        return []
    active = queue.get("active_rows") if isinstance(queue.get("active_rows"), list) else []
    if active:
        return active
    rows = queue.get("queue_order") if isinstance(queue.get("queue_order"), list) else []
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and (
            row.get("active_epoch") is not None
            or row.get("active_step") is not None
            or row.get("status") in {"running", "partial", "running_or_partial"}
        )
    ]


def _allowed_claims(rows: list[dict[str, str]], gate: dict[str, Any]) -> list[str]:
    complete = {row["requirement"] for row in rows if row["status"] == "complete"}
    allowed = []
    if "Verify local datasets are complete and readable" in complete:
        allowed.append("Local dataset files/manifests have been audited for presence and geometry.")
    if "P0 full ablation table" in complete:
        allowed.append("P0 ablation supports conservative module-level observations, especially the PhysicsTexture signal.")
    allowed.append("The current task is weak visual friction-affordance interval estimation, not measured friction regression.")
    if gate.get("verdict") == "not_ready":
        allowed.append("Current negative LODO and shortcut evidence may be reported as failure analysis, not as broad OOD success.")
    return allowed


def _disallowed_claims(rows: list[dict[str, str]], gate: dict[str, Any]) -> list[str]:
    disallowed = [
        "Do not claim measured tire-road friction accuracy.",
        "Do not compare numerically to external papers unless labels, splits, preprocessing, and metrics match.",
    ]
    if gate.get("verdict") != "ready_for_strict_paper_claims":
        disallowed.append("Do not claim the final algorithm is top-venue-ready yet.")
    if any(row["requirement"] == "LODO experiments" and row["status"] != "complete" for row in rows):
        disallowed.append("Do not claim cross-dataset generalization yet.")
    if any(row["requirement"] == "Matched single-dataset FAF vs ConvNeXt baselines" and row["status"] != "complete" for row in rows):
        disallowed.append("Do not claim superiority over a strong same-split visual baseline yet.")
    return disallowed


def _next_order(requirements: dict[str, dict[str, Any]], queue: dict[str, Any], rows: list[dict[str, str]]) -> list[str]:
    active = _current_execution(queue, {}, {}, {})
    actions = []
    if active.get("active_run"):
        actions.append(f"Let `{active['active_run']}` finish without launching competing GPU work.")
    if requirements.get("lodo_complete", {}).get("status") != "complete":
        actions.append("After LODO finishes, rerun full postprocess, readiness gate, claim ledger, reviewer checklist, and artifact contract.")
    if requirements.get("fair_single_dataset_complete", {}).get("status") != "complete":
        actions.append("Run matched single-dataset FAF and ConvNeXt baselines for RSCD, RoadSaW, and RoadSC.")
    if requirements.get("candidate_path_complete", {}).get("status") != "complete":
        actions.append("Run v6-v25 P1/P2/P3 candidates and rank by shortcut, safety, evidence, and coverage-width metrics.")
    if requirements.get("final_method_complete", {}).get("status") != "complete":
        actions.append("Freeze and run the lean final method only after candidate evidence exists.")
    incomplete = [row["requirement"] for row in rows if row["status"] == "incomplete"]
    if incomplete:
        actions.append(f"Keep objective open; incomplete high-level requirements remain: {', '.join(incomplete[:5])}.")
    return actions


def render_markdown(report: dict[str, Any]) -> str:
    current = report["current_execution"]
    lines = [
        "# Objective Completion Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Scope: {report['scope']}.",
        "",
        report["claim_boundary"],
        "",
        "This audit is intentionally strict. A requirement is marked complete only when current artifacts prove it.",
        "",
        "## Current Execution Snapshot",
        "",
        "| Item | Current evidence |",
        "|---|---|",
        f"| Active official GPU run | `{current.get('active_run')}` |",
        f"| Status | `{current.get('status')}` |",
        f"| Latest queue progress | epoch `{current.get('epoch')}/{current.get('epochs')}`, step `{current.get('step')}/{current.get('steps')}` |",
        f"| Queue progress | `{current.get('queue_complete')}/{current.get('queue_total')}` complete |",
        f"| Overall readiness | `{current.get('readiness')}` |",
        "",
        "## Requirement-Level Audit",
        "",
        "| Requirement from objective | Status | Evidence proving current status | Remaining work |",
        "|---|---|---|---|",
    ]
    for row in report["requirements"]:
        lines.append(
            "| {requirement} | {status} | {evidence} | {remaining_work} |".format(
                requirement=_escape(row["requirement"]),
                status=f"`{row['status']}`",
                evidence=_escape(row["evidence"]),
                remaining_work=_escape(row["remaining_work"]),
            )
        )
    lines.extend(["", "## Status Counts", ""])
    lines.append("| Status | Count |")
    lines.append("|---|---:|")
    for key, value in report["counts"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Allowed Claims Now", ""])
    for item in report["allowed_claims"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Not Allowed Yet", ""])
    for item in report["disallowed_claims"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Required Evidence Order", ""])
    for idx, action in enumerate(report["next_order"], start=1):
        lines.append(f"{idx}. {action}")
    lines.append("")
    return "\n".join(lines)


def _escape(value: Any) -> str:
    text = str(value).replace("\n", "<br>")
    return text.replace("|", "\\|")


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


if __name__ == "__main__":
    main()
