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
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "claim_evidence_ledger.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "claim_evidence_ledger.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_report(summary_dir: Path) -> dict[str, Any]:
    readiness = _load_json(summary_dir / "topvenue_readiness_gate.json") or {}
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    completeness = _load_json(summary_dir / "protocol_completeness.json") or {}
    contract = _load_json(summary_dir / "artifact_contract_report.json") or {}
    p0 = _load_json(summary_dir / "p0_claim_report.json") or {}
    lodo = _load_json(summary_dir / "lodo_generalization_report.json") or {}
    fair = _load_json(summary_dir / "fair_comparison_report.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    interval = _load_json(summary_dir / "interval_quality_report.json") or {}
    wetness = _load_json(summary_dir / "wetness_state_report.json") or {}
    evidence_failure = _load_json(summary_dir / "evidence_failure_report.json") or {}
    final_selection = _load_json(summary_dir / "final_method_selection_report.json") or {}
    module_retention = _load_json(summary_dir / "module_retention_report.json") or {}
    source_audit = _load_json(summary_dir / "friction_interval_source_audit.json") or {}
    dataset_inventory = _load_json(summary_dir / "dataset_inventory_report.json") or {}
    external = _load_json(summary_dir / "external_benchmark_report.json") or {}

    requirements = _requirements(completeness)
    gates = _gates(readiness)
    rows = [
        _claim_p0(requirements, contract, p0, module_retention),
        _claim_lodo(requirements, lodo),
        _claim_fair_baseline(requirements, fair, external),
        _claim_shortcut(requirements, shortcut, summary),
        _claim_interval(source_audit, interval),
        _claim_wet_roadsaw(requirements, wetness, lodo),
        _claim_interpretability(summary, evidence_failure, contract),
        _claim_final_method(requirements, final_selection),
        _claim_dataset_and_label_validity(dataset_inventory, source_audit, external),
    ]
    verdict_counts: dict[str, int] = {}
    for row in rows:
        verdict_counts[row["status"]] = verdict_counts.get(row["status"], 0) + 1
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "readiness_verdict": readiness.get("verdict"),
        "readiness_blocks": readiness.get("num_blocks"),
        "readiness_warnings": readiness.get("num_warnings"),
        "claim_rows": rows,
        "status_counts": verdict_counts,
        "blocking_gates": [
            gate.get("name")
            for gate in gates
            if gate.get("level") == "block"
        ],
        "claim_rules": [
            "Only `supported` claims can appear as main paper claims.",
            "`partial` claims can appear as preliminary evidence with explicit limitations.",
            "`not_supported` claims must not be made as positive claims; use them only as failure analysis.",
            "`not_supported_yet` claims must remain future work or next-step discussion.",
            "Never describe RSCD/RoadSaW/RoadSC weak labels as synchronized measured tire-road friction coefficients.",
            "Published external numbers are contextual unless split, labels, preprocessing, and metrics match the local protocol.",
        ],
    }


def _claim_p0(
    requirements: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    p0: dict[str, Any],
    module_retention: dict[str, Any],
) -> dict[str, Any]:
    req = requirements.get("p0_ablation_complete", {})
    hard = (contract.get("hard_status") or {}).get("p0_ablation", {})
    module_actions = [
        {
            "module": row.get("module") or row.get("method"),
            "decision": row.get("current_decision") or row.get("claim_recommendation"),
        }
        for row in (module_retention.get("rows") or p0.get("adjacent_deltas") or [])
    ][:6]
    complete = req.get("status") == "complete" and hard.get("complete") is True
    return {
        "claim": "Core modules improve visual friction-affordance estimation.",
        "status": "supported" if complete else "not_supported_yet",
        "evidence": [
            f"P0 requirement: {req.get('status', 'missing')}",
            f"P0 artifact contract: {hard.get('num_complete', 0)}/{hard.get('num_runs', 0)} complete",
            f"P0 claim report: {p0.get('core_status') or p0.get('status')}",
        ],
        "missing_or_risk": req.get("missing", []),
        "allowed_wording": (
            "Full P0 ablation supports module-level claims."
            if complete
            else "Only completed rows can be discussed; Full model is still pending."
        ),
        "module_actions": module_actions,
    }


def _claim_lodo(requirements: dict[str, dict[str, Any]], lodo: dict[str, Any]) -> dict[str, Any]:
    req = requirements.get("lodo_complete", {})
    complete = req.get("status") == "complete"
    roadsaw = lodo.get("roadsaw_readout") if isinstance(lodo.get("roadsaw_readout"), dict) else {}
    roadsaw_status = roadsaw.get("status") or lodo.get("roadsaw_status") or "missing"
    verdict = str(lodo.get("verdict") or "")
    roadsaw_verdict = str(roadsaw.get("verdict") or "")
    failed = "failure" in verdict or roadsaw_verdict.endswith("_failure")
    status = "not_supported" if complete and failed else ("supported" if complete else "not_supported_yet")
    risks = []
    if complete and failed:
        for row in lodo.get("rows", []) if isinstance(lodo.get("rows"), list) else []:
            held = row.get("held_out") or "unknown"
            risk_f1 = _pct(row.get("risk_f1"))
            risks.append(f"{held}_transfer_failure_risk_f1_{risk_f1}")
    return {
        "claim": "The method generalizes across public road-condition datasets.",
        "status": status,
        "evidence": [
            f"LODO requirement: {req.get('status', 'missing')}",
            f"LODO verdict: {verdict or '-'}",
            f"Held-out RoadSaW status: {roadsaw_status}",
        ],
        "missing_or_risk": risks if complete else req.get("missing", []),
        "allowed_wording": (
            "LODO evidence supports cross-dataset generalization."
            if status == "supported"
            else (
                "LODO is complete and shows severe transfer failure; present it as failure analysis and motivation for the next method."
                if complete
                else "Do not make an OOD/generalization claim before held-out RoadSaW and other LODO rows finish."
            )
        ),
    }


def _claim_fair_baseline(
    requirements: dict[str, dict[str, Any]],
    fair: dict[str, Any],
    external: dict[str, Any],
) -> dict[str, Any]:
    req = requirements.get("fair_single_dataset_complete", {})
    complete = req.get("status") == "complete"
    return {
        "claim": "FAF improves over a strong same-split ConvNeXt visual baseline.",
        "status": "supported" if complete else "not_supported_yet",
        "evidence": [
            f"Fair single-dataset requirement: {req.get('status', 'missing')}",
            f"Fair report rows: {len(fair.get('rows', []) or fair.get('fair_single_dataset_deltas', []) or [])}",
            external.get("strict_comparison_rule", "strict comparison rule missing"),
        ],
        "missing_or_risk": req.get("missing", []),
        "allowed_wording": (
            "Matched FAF-vs-ConvNeXt rows support fair baseline claims."
            if complete
            else "Published numbers are context only; wait for matched ConvNeXt rows."
        ),
    }


def _claim_shortcut(
    requirements: dict[str, dict[str, Any]],
    shortcut: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    high = int(shortcut.get("num_high_shortcut", 0) or 0)
    complete = int(shortcut.get("num_complete", 0) or 0)
    p1 = requirements.get("candidate_path_complete", {})
    p1_done = p1.get("status") == "complete"
    supported = p1_done and high == 0 and complete > 0
    status = "supported" if supported else ("partial" if complete else "not_supported_yet")
    best = shortcut.get("best_completed_by_core_state_probe") or {}
    return {
        "claim": "The representation reduces dataset-style shortcut learning.",
        "status": status,
        "evidence": [
            f"Dataset shortcut verdict: {shortcut.get('verdict')}",
            f"High-shortcut completed rows: {high}/{complete}",
            f"P1 candidate requirement: {p1.get('status', 'missing')}",
            f"Best completed probe row: {best.get('run', '-')}",
        ],
        "missing_or_risk": p1.get("missing", []) or [row.get("method") for row in summary.get("core_ablation", []) if row.get("dataset_id_balanced_accuracy")],
        "allowed_wording": (
            "Shortcut mitigation is supported by completed P1 diagnostics."
            if supported
            else "Current completed rows still encode dataset identity strongly; present this as a failure mode and P1 target."
        ),
    }


def _claim_interval(source_audit: dict[str, Any], interval: dict[str, Any]) -> dict[str, Any]:
    sources_ok = source_audit.get("verdict") == "pass"
    watch = int(interval.get("num_watchlist_items", 0) or 0)
    rows = int(interval.get("num_runs", 0) or 0)
    supported = sources_ok and rows > 0 and watch == 0
    status = "supported" if supported else ("partial" if sources_ok and rows > 0 else "not_supported_yet")
    return {
        "claim": "The model outputs calibrated weak friction-affordance intervals with meaningful width.",
        "status": status,
        "evidence": [
            f"Friction interval source audit: {source_audit.get('verdict')}",
            f"Interval audited runs: {rows}",
            f"Conditional undercoverage watchlist: {watch}",
        ],
        "missing_or_risk": [
            item.get("group")
            for item in (interval.get("watchlist") or [])[:8]
        ],
        "allowed_wording": (
            "Calibration and conditional interval quality support the interval claim."
            if supported
            else "Source anchors are valid, but conditional undercoverage must be reported and improved."
        ),
    }


def _claim_wet_roadsaw(
    requirements: dict[str, dict[str, Any]],
    wetness: dict[str, Any],
    lodo: dict[str, Any],
) -> dict[str, Any]:
    lodo_done = requirements.get("lodo_complete", {}).get("status") == "complete"
    watch = int(wetness.get("num_watchlist", 0) or 0)
    complete = int(wetness.get("num_complete", 0) or 0)
    supported = lodo_done and complete > 0 and watch == 0
    status = "supported" if supported else ("partial" if complete else "not_supported_yet")
    return {
        "claim": "The method handles RoadSaW damp/wet/very-wet states.",
        "status": status,
        "evidence": [
            f"Wetness audited rows: {complete}",
            f"Wetness watchlist rows: {watch}",
            f"RoadSaW LODO verdict: {lodo.get('roadsaw_verdict') or lodo.get('verdict')}",
        ],
        "missing_or_risk": [
            item.get("run")
            for item in (wetness.get("watchlist") or [])[:8]
        ],
        "allowed_wording": (
            "RoadSaW wet-state handling is supported."
            if supported
            else "RoadSaW wetness is currently a weakness; use it as the main stress-test narrative."
        ),
    }


def _claim_interpretability(
    summary: dict[str, Any],
    evidence_failure: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    runs = int(evidence_failure.get("num_evidence_runs", 0) or 0)
    examples = len(evidence_failure.get("examples", []) or [])
    hard = (contract.get("hard_status") or {}).get("p0_ablation", {})
    supported = runs > 0 and examples > 0
    return {
        "claim": "EvidenceField provides interpretable road-region evidence.",
        "status": "partial" if supported and not hard.get("complete") else ("supported" if supported else "not_supported_yet"),
        "evidence": [
            f"Evidence audit runs: {runs}",
            f"Evidence examples: {examples}",
            f"P0 artifact contract complete: {hard.get('complete')}",
        ],
        "missing_or_risk": [],
        "allowed_wording": (
            "Evidence maps can be used for qualitative and failure analysis; final interpretability claim waits for v5/final rows."
            if supported
            else "Do not make interpretability claims until evidence maps and audits exist."
        ),
    }


def _claim_final_method(
    requirements: dict[str, dict[str, Any]],
    final_selection: dict[str, Any],
) -> dict[str, Any]:
    req = requirements.get("final_method_complete", {})
    ready = final_selection.get("verdict") == "ready_to_select_final_method"
    supported = req.get("status") == "complete" and ready
    return {
        "claim": "The final lean road-ROI safety method is the paper method.",
        "status": "supported" if supported else "not_supported_yet",
        "evidence": [
            f"Final method requirement: {req.get('status', 'missing')}",
            f"Final selection verdict: {final_selection.get('verdict')}",
        ],
        "missing_or_risk": req.get("missing", []),
        "allowed_wording": (
            "Final method can be frozen."
            if supported
            else "Keep final architecture provisional until final LODO and matched single-dataset evidence finish."
        ),
    }


def _claim_dataset_and_label_validity(
    dataset_inventory: dict[str, Any],
    source_audit: dict[str, Any],
    external: dict[str, Any],
) -> dict[str, Any]:
    datasets_ok = dataset_inventory.get("verdict") == "pass"
    sources_ok = source_audit.get("verdict") == "pass"
    external_sources = len(external.get("public_sources", []) or [])
    supported = datasets_ok and sources_ok and external_sources >= 10
    return {
        "claim": "The benchmark uses public datasets and public friction-interval anchors under a weak-label framing.",
        "status": "supported" if supported else "partial",
        "evidence": [
            f"Dataset inventory: {dataset_inventory.get('verdict')}",
            f"Friction interval source audit: {source_audit.get('verdict')}",
            f"External/public sources mapped: {external_sources}",
        ],
        "missing_or_risk": [],
        "allowed_wording": (
            "Supported: public visual labels are mapped to conservative weak friction-affordance intervals."
            if supported
            else "Dataset/source framing needs more audit evidence."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Claim Evidence Ledger",
        "",
        f"Generated at: {report['generated_at']}",
        f"Summary dir: `{report['summary_dir']}`",
        f"Readiness: `{report['readiness_verdict']}` ({report['readiness_blocks']} blocks, {report['readiness_warnings']} warnings)",
        "",
        "## Claim Status",
        "",
        "| Claim | Status | Evidence | Missing / risk | Allowed wording |",
        "|---|---|---|---|---|",
    ]
    for row in report["claim_rows"]:
        evidence = "<br>".join(str(item) for item in row.get("evidence", []) if item)
        missing = ", ".join(f"`{item}`" for item in row.get("missing_or_risk", []) if item) or "-"
        lines.append(
            "| {claim} | `{status}` | {evidence} | {missing} | {wording} |".format(
                claim=row["claim"],
                status=row["status"],
                evidence=evidence or "-",
                missing=missing,
                wording=row["allowed_wording"],
            )
        )
    lines.extend(["", "## Blocking Gates", ""])
    for item in report.get("blocking_gates", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Claim Rules", ""])
    for item in report["claim_rules"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _requirements(completeness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("name")): item
        for item in completeness.get("requirements", [])
        if isinstance(item, dict) and item.get("name")
    }


def _gates(readiness: dict[str, Any]) -> list[dict[str, Any]]:
    gates = readiness.get("gates", [])
    return gates if isinstance(gates, list) else []


def _pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}pct"
    except (TypeError, ValueError):
        return "missing"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    main()
