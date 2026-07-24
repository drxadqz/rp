from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "reviewer_evidence_checklist.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "reviewer_evidence_checklist.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def build_report(summary_dir: Path) -> dict[str, Any]:
    dashboard = _load_json(summary_dir / "experiment_status_dashboard.json") or {}
    readiness = _load_json(summary_dir / "topvenue_readiness_gate.json") or {}
    completeness = _load_json(summary_dir / "protocol_completeness.json") or {}
    paper_p0 = _load_json(summary_dir / "paper_p0_ablation_table.json") or {}
    p0_claim = _load_json(summary_dir / "p0_claim_report.json") or {}
    lodo = _load_json(summary_dir / "lodo_generalization_report.json") or {}
    external = _load_json(summary_dir / "external_benchmark_report.json") or {}
    fair_protocol = _load_json(summary_dir / "fair_comparison_protocol_audit.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    input_style = _load_json(summary_dir / "input_canonicalization_style_audit.json") or {}
    wetness = _load_json(summary_dir / "wetness_state_report.json") or {}
    interval = _load_json(summary_dir / "interval_quality_report.json") or {}
    interval_sources = _load_json(summary_dir / "friction_interval_source_audit.json") or {}
    evidence = _load_json(summary_dir / "evidence_failure_report.json") or {}
    queue_path = summary_dir / "queue_recovery_report.json"
    watch_path = summary_dir / "active_training_watch_report.json"
    queue = _load_json(queue_path) or {}
    watch = _load_json(watch_path) or {}
    if _is_older(watch_path, queue_path):
        watch = {}
    active_live = watch or _load_json(summary_dir / "active_live_training_reports.json") or {}

    requirements = {item.get("name"): item for item in completeness.get("requirements", []) if isinstance(item, dict)}
    gates = {item.get("name"): item for item in readiness.get("gates", []) if isinstance(item, dict)}
    claims = [
        _claim_p0(paper_p0, p0_claim),
        _claim_lodo(lodo, requirements, gates),
        _claim_fair_baseline(external, fair_protocol, requirements, gates),
        _claim_shortcut(shortcut, input_style),
        _claim_wetness(wetness, lodo),
        _claim_interval(interval, interval_sources),
        _claim_evidence(evidence),
        _claim_final_method(requirements, gates),
    ]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "readiness_verdict": readiness.get("verdict") or (dashboard.get("readiness") or {}).get("verdict"),
        "active_run": _active_run(queue, watch, dashboard, active_live),
        "claims": claims,
        "strict_rules": [
            "Do not claim measured tire-road friction; claim visual-evidence-conditioned friction-affordance intervals from public weak labels.",
            "Do not compare numeric results to external papers unless split, label space, preprocessing, and metric definition match.",
            "Use matched local ConvNeXt rows as the primary numerical baseline.",
            "Use LODO, especially held-out RoadSaW, as the main cross-dataset generalization evidence.",
            "Keep or remove modules by predeclared safety/generalization metrics, not by pooled accuracy alone.",
        ],
        "next_milestones": [
            "Let the active training queue finish without launching a duplicate GPU worker.",
            "Analyze the completed LODO failures, especially held-out RoadSaW, as the shortcut/wetness failure mode.",
            "Run matched single-dataset FAF and ConvNeXt baselines.",
            "Run v6-v25 candidates and rank by dataset-ID drop, RoadSaW wetness, quality-slice robustness, low-friction recall, worst-dataset F1, and coverage-width tradeoff.",
            "Freeze the final lean route only after candidate, LODO, and fair-baseline evidence is complete.",
        ],
    }


def _claim_p0(paper_p0: dict[str, Any], p0_claim: dict[str, Any]) -> dict[str, Any]:
    complete = paper_p0.get("status") == "complete" and p0_claim.get("core_status") == "complete"
    best = paper_p0.get("best_by_metric") or {}
    return {
        "claim": "Core P0 ablation identifies which modules earn retention.",
        "status": "supported" if complete else "missing",
        "evidence": [
            "paper_p0_ablation_table.md/csv/tex",
            "p0_claim_report.md/json",
            f"best friction: {(best.get('friction_macro_f1') or {}).get('method')}",
            f"best worst-dataset: {(best.get('worst_dataset_f1') or {}).get('method')}",
        ],
        "missing": [] if complete else ["complete P0 rows and refreshed P0 reports"],
        "allowed_wording": (
            "P0 supports keeping PhysicsTexture and treating FrictionSet/DG/full fusion as provisional remove-or-rework."
            if complete
            else "P0 is not yet complete."
        ),
        "next_action": "Use the table as the core ablation evidence; do not promote Full fusion as final method.",
    }


def _claim_lodo(lodo: dict[str, Any], requirements: dict[str, dict[str, Any]], gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    req = requirements.get("lodo_complete", {})
    complete = req.get("status") == "complete"
    verdict = str(lodo.get("verdict") or "")
    roadsaw_verdict = str((lodo.get("roadsaw_readout") or {}).get("verdict") or "")
    failed = "failure" in verdict or roadsaw_verdict.endswith("_failure")
    status = "not_supported" if complete and failed else ("supported" if complete else "not_supported_yet")
    roadsaw_status = _roadsaw_lodo_status(lodo)
    return {
        "claim": "The method generalizes across public road-condition datasets.",
        "status": status,
        "evidence": [
            "lodo_generalization_report.md/json",
            f"LODO verdict: {verdict or '-'}",
            f"RoadSaW LODO status: {roadsaw_status}",
            f"readiness gate: {(gates.get('lodo_complete') or {}).get('level')}",
        ],
        "missing": [] if complete else (req.get("missing") or ["lodo_roadsaw_full_faf", "lodo_rscd_full_faf", "lodo_roadsc_full_faf"]),
        "allowed_wording": (
            "LODO evidence supports cross-dataset generalization."
            if status == "supported"
            else (
                "LODO is complete and shows severe cross-dataset transfer failure; use it as failure analysis, not as an OOD success claim."
                if complete
                else "Do not make an OOD claim before held-out RoadSaW and the remaining LODO rows finish."
            )
        ),
        "next_action": (
            "Use the completed held-out RoadSaW failure to drive shortcut, wetness, and interval candidates."
            if complete
            else "Finish LODO, then use held-out RoadSaW as the decisive wetness-domain stress test."
        ),
    }


def _roadsaw_lodo_status(lodo: dict[str, Any]) -> str:
    readout = lodo.get("roadsaw_readout") or {}
    if readout.get("status"):
        return str(readout.get("status"))
    for row in lodo.get("rows", []) or []:
        if str(row.get("held_out") or "").lower() == "roadsaw":
            return str(row.get("status") or "missing")
    return "missing"


def _claim_fair_baseline(
    external: dict[str, Any],
    fair_protocol: dict[str, Any],
    requirements: dict[str, dict[str, Any]],
    gates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    req = requirements.get("fair_single_dataset_complete", {})
    protocol_ok = str(fair_protocol.get("verdict", "")).startswith("pass")
    complete = req.get("status") == "complete"
    return {
        "claim": "The method improves over a strong same-split ConvNeXt visual baseline.",
        "status": "supported" if complete and protocol_ok else "not_supported_yet",
        "evidence": [
            "external_benchmark_report.md/json",
            "fair_comparison_protocol_audit.md/json",
            f"protocol verdict: {fair_protocol.get('verdict')}",
            f"primary baseline level: {_external_primary_level(external)}",
            f"readiness gate: {(gates.get('fair_single_dataset_complete') or {}).get('level')}",
        ],
        "missing": req.get("missing") or [
            "single_roadsaw_full_faf",
            "single_rscd_full_faf",
            "single_roadsc_full_faf",
            "baseline_single_roadsaw_global_convnext",
            "baseline_single_rscd_global_convnext",
            "baseline_single_roadsc_global_convnext",
        ],
        "allowed_wording": (
            "Matched local FAF-vs-ConvNeXt rows are the primary fair numerical comparison."
            if complete
            else "External numbers are context only; wait for matched ConvNeXt rows."
        ),
        "next_action": "Run same-split FAF and ConvNeXt baselines after LODO priority evidence.",
    }


def _claim_shortcut(shortcut: dict[str, Any], input_style: dict[str, Any]) -> dict[str, Any]:
    high = int(shortcut.get("num_high_shortcut", 0) or 0)
    complete = int(shortcut.get("num_complete", 0) or 0)
    best_style = _best_input_style(input_style)
    return {
        "claim": "The representation reduces dataset-style shortcut learning.",
        "status": "partial" if best_style else "not_supported_yet",
        "evidence": [
            "dataset_shortcut_report.md/json",
            "input_canonicalization_style_audit.md/json",
            f"completed rows over shortcut threshold: {high}/{complete}",
            f"best input style diagnostic: {best_style.get('run')} ({_fmt_pct(best_style.get('relative'))} of baseline)",
        ],
        "missing": ["task-metric and dataset-ID results for v6/v7/v11/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24/v25"],
        "allowed_wording": "Current completed rows still encode dataset identity; v16 only shows preprocessing-level style-gap reduction.",
        "next_action": "Run shortcut candidates and keep only those that lower dataset-ID probes without hurting task metrics.",
    }


def _claim_wetness(wetness: dict[str, Any], lodo: dict[str, Any]) -> dict[str, Any]:
    watchlist = int(wetness.get("num_watchlist", 0) or 0)
    complete_rows = [row for row in wetness.get("rows", []) if row.get("status") == "complete"]
    latest = wetness.get("latest_complete") or (complete_rows[-1] if complete_rows else {})
    roadsaw_lodo_status = _roadsaw_lodo_status(lodo)
    roadsaw_lodo_done = roadsaw_lodo_status == "complete"
    missing = ["v9/v10/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24/v25 wetness stress results"]
    if roadsaw_lodo_done:
        missing.append("completed RoadSaW LODO is a wetness-domain failure signal, not a positive wetness claim")
    else:
        missing.insert(0, "held-out RoadSaW LODO")
    return {
        "claim": "The method handles RoadSaW damp/wet/very-wet states.",
        "status": "partial" if watchlist else "supported",
        "evidence": [
            "wetness_state_report.md/json",
            f"watchlist rows: {watchlist}",
            f"latest complete: {latest.get('run')}",
            f"latest RoadSaW wetness macro-F1: {_fmt_pct(latest.get('roadsaw_wetness_macro_f1'))}",
            f"RoadSaW LODO status: {roadsaw_lodo_status}",
        ],
        "missing": missing,
        "allowed_wording": "RoadSaW wetness is a current weakness and should be presented as a stress-test target.",
        "next_action": "Use wet-state hard sampling, ordinal wetness loss, consistency, and ROI candidates.",
    }


def _claim_interval(interval: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    num_watch = int(interval.get("num_watchlist_items", 0) or 0)
    source_ok = sources.get("verdict") == "pass"
    return {
        "claim": "The model outputs calibrated weak friction-affordance intervals with meaningful width.",
        "status": "partial" if source_ok else "not_supported_yet",
        "evidence": [
            "friction_interval_source_audit.md/json",
            "interval_quality_report.md/json",
            f"public anchor verdict: {sources.get('verdict')}",
            f"conditional undercoverage watchlist: {num_watch}",
        ],
        "missing": ["v12/v14/v15/v16/v17/v18/v19/v20/v21/v22/v23/v24/v25/final conditional coverage-width evidence"],
        "allowed_wording": "Public anchors support the weak interval framing, but conditional undercoverage must be reported and improved.",
        "next_action": "Rank P3 candidates by conditional coverage and width together.",
    }


def _claim_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    runs = evidence.get("runs") or []
    examples = evidence.get("examples") or []
    return {
        "claim": "EvidenceField provides interpretable road-region evidence.",
        "status": "partial" if runs else "not_supported_yet",
        "evidence": [
            "evidence_failure_report.md/json",
            f"audited EvidenceField runs: {len(runs)}",
            f"sampled examples: {len(examples)}",
        ],
        "missing": ["candidate ROI/pseudo-road-mask results", "success-vs-failure evidence map analysis for final method"],
        "allowed_wording": "EvidenceField is promising but not yet final explanatory evidence.",
        "next_action": "Use ROI constraints and failure-map audits to prove attention stays on plausible road evidence.",
    }


def _claim_final_method(requirements: dict[str, dict[str, Any]], gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    req = requirements.get("final_method_complete", {})
    return {
        "claim": "The final lean road-ROI safety method is ready as the paper method.",
        "status": "supported" if req.get("status") == "complete" else "not_supported_yet",
        "evidence": [
            "final_method_selection_report.md/json",
            "final lean LODO rows",
            "final lean single-dataset rows",
            f"readiness gate: {(gates.get('final_method_complete') or {}).get('level')}",
        ],
        "missing": req.get("missing") or [
            "final_lodo_roadsaw_lean_road_roi_safety",
            "final_lodo_rscd_lean_road_roi_safety",
            "final_lodo_roadsc_lean_road_roi_safety",
            "final_single_roadsaw_lean_road_roi_safety",
            "final_single_rscd_lean_road_roi_safety",
            "final_single_roadsc_lean_road_roi_safety",
        ],
        "allowed_wording": "Keep final architecture provisional until final LODO and matched baseline evidence finish.",
        "next_action": "Freeze final method only after v6-v25 candidate evidence.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    active = report.get("active_run") or {}
    lines = [
        "# Reviewer Evidence Checklist",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Readiness verdict: `{report.get('readiness_verdict')}`",
        "",
        "## Active Execution",
        "",
        "- Active run: `{name}` `{status}` epoch `{epoch}/{epochs}` step `{step}/{steps}`.".format(
            name=active.get("name") or "-",
            status=active.get("status") or "-",
            epoch=active.get("epoch") or "-",
            epochs=active.get("epochs") or "-",
            step=active.get("step") or "-",
            steps=active.get("steps") or "-",
        ),
        "",
        "## Claim Checklist",
        "",
        "| Claim | Status | Evidence | Missing | Allowed wording | Next action |",
        "|---|---|---|---|---|---|",
    ]
    for row in report.get("claims", []):
        lines.append(
            "| {claim} | `{status}` | {evidence} | {missing} | {wording} | {action} |".format(
                claim=_clean(row.get("claim")),
                status=row.get("status"),
                evidence=_fmt_list(row.get("evidence")),
                missing=_fmt_list(row.get("missing")),
                wording=_clean(row.get("allowed_wording")),
                action=_clean(row.get("next_action")),
            )
        )
    lines.extend(["", "## Strict Rules", ""])
    for item in report.get("strict_rules", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Next Milestones", ""])
    for item in report.get("next_milestones", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _active_run(
    queue: dict[str, Any],
    watch: dict[str, Any],
    dashboard: dict[str, Any],
    active_live: dict[str, Any],
) -> dict[str, Any]:
    active = (watch.get("active") if isinstance(watch, dict) else None) or {}
    if active.get("name"):
        out = _normalize_active(active)
        out["status"] = active.get("status") or "running_or_partial"
        return out
    for row in queue.get("queue_order", []) if isinstance(queue, dict) else []:
        if row.get("status") in {"running_or_partial", "partial_ci_missing"}:
            return _normalize_active(row)
    rows = (queue.get("active_rows") if isinstance(queue, dict) else None) or []
    if rows:
        return _normalize_active(rows[0])
    rows = (dashboard.get("active_rows") if isinstance(dashboard, dict) else None) or []
    if rows:
        return _normalize_active(rows[0])
    active = (active_live.get("active") if isinstance(active_live, dict) else None) or {}
    if active.get("name"):
        out = _normalize_active(active)
        out["status"] = "running_or_partial"
        return out
    return {}


def _normalize_active(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("name") or row.get("run"),
        "status": row.get("status"),
        "epoch": row.get("active_epoch") or row.get("epoch"),
        "epochs": row.get("active_epochs") or row.get("epochs"),
        "step": row.get("active_step") or row.get("step"),
        "steps": row.get("active_steps") or row.get("steps"),
        "phase": row.get("active_phase") or row.get("phase"),
    }


def _external_primary_level(external: dict[str, Any]) -> str:
    for row in external.get("comparability_matrix", []) or external.get("external_comparability_matrix", []) or []:
        if row.get("level") == "primary_numeric_baseline":
            return str(row.get("source") or "matched local ConvNeXt")
    return "matched local ConvNeXt"


def _best_input_style(report: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for item in report.get("configs") or []:
        cross = item.get("cross_dataset_signals") or {}
        rel = item.get("relative_to_first_config") or {}
        score = cross.get("style_gap_score")
        relative = ((rel.get("style_gap_score") or {}).get("relative") if isinstance(rel, dict) else None)
        if isinstance(score, (int, float)):
            rows.append({"run": item.get("run"), "score": score, "relative": relative})
    return min(rows, key=lambda row: row["score"]) if rows else {}


def _fmt_list(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        return "<br>".join(_clean(item) for item in value) if value else "-"
    return _clean(value)


def _fmt_pct(value: Any) -> str:
    try:
        if value is None:
            return "-"
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def _clean(value: Any) -> str:
    if value is None:
        return "-"
    return str(value).replace("\n", " ").replace("|", "/")


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _is_older(candidate: Path, reference: Path) -> bool:
    if not candidate.exists() or not reference.exists():
        return False
    try:
        return candidate.stat().st_mtime < reference.stat().st_mtime
    except OSError:
        return False


if __name__ == "__main__":
    main()
