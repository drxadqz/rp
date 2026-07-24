from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "final_freeze_audit.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "final_freeze_audit.json"

FINAL_ROWS = [
    "final_lodo_roadsaw_lean_road_roi_safety",
    "final_lodo_rscd_lean_road_roi_safety",
    "final_lodo_roadsc_lean_road_roi_safety",
    "final_single_roadsaw_lean_road_roi_safety",
    "final_single_rscd_lean_road_roi_safety",
    "final_single_roadsc_lean_road_roi_safety",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether the final paper method is allowed to be frozen."
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out_md)
    print(args.out_json)


def build_report(summary_dir: Path) -> dict[str, Any]:
    artifact = _load_json(summary_dir / "artifact_contract_report.json") or {}
    candidate_pruning = _load_json(summary_dir / "candidate_pruning_report.json") or {}
    fair_priority = _load_json(summary_dir / "fair_comparison_execution_priority.json") or {}
    cv_protocol = _load_json(summary_dir / "cv_transfer_experiment_protocol.json") or {}
    final_selection = _load_json(summary_dir / "final_method_selection_report.json") or {}
    module_retention = _load_json(summary_dir / "module_retention_report.json") or {}
    algorithm_audit = _load_json(summary_dir / "algorithm_module_audit.json") or {}

    hard = artifact.get("hard_status", {}) if isinstance(artifact.get("hard_status"), dict) else {}
    fair_status = hard.get("single_dataset_fair", {}) if isinstance(hard.get("single_dataset_fair"), dict) else {}
    candidate_status = hard.get("p1_candidates", {}) if isinstance(hard.get("p1_candidates"), dict) else {}
    final_status = hard.get("final_method", {}) if isinstance(hard.get("final_method"), dict) else {}
    pruning_counts = candidate_pruning.get("counts", {}) if isinstance(candidate_pruning.get("counts"), dict) else {}
    cv_counts = cv_protocol.get("counts", {}) if isinstance(cv_protocol.get("counts"), dict) else {}

    dependencies = [
        {
            "name": "fair_single_dataset_complete",
            "complete": bool(fair_status.get("complete")),
            "missing": fair_status.get("missing", []),
            "why": "Needed before claiming FAF beats a same-split ConvNeXt baseline.",
        },
        {
            "name": "candidate_metrics_complete",
            "complete": bool(candidate_status.get("complete")),
            "missing": candidate_status.get("missing", []),
            "why": "Needed before deciding which CV-transfer modules survive into the final method.",
        },
        {
            "name": "candidate_pruning_decision_ready",
            "complete": _candidate_pruning_ready(candidate_pruning),
            "missing": _candidate_pruning_gap(candidate_pruning),
            "why": "Needed to keep, merge, or delete modules by predeclared rules.",
        },
        {
            "name": "cv_transfer_protocol_ready",
            "complete": not bool(cv_protocol.get("blocks")),
            "missing": cv_protocol.get("blocks", []),
            "why": "Needed to prove segmentation/domain/material transfer routes are executable before judging metrics.",
        },
        {
            "name": "final_runs_complete",
            "complete": bool(final_status.get("complete")),
            "missing": final_status.get("missing", []),
            "why": "Needed before the selected final method can be used for paper claims.",
        },
    ]
    blocking = [row for row in dependencies if not row["complete"]]
    final_rows = _final_rows(algorithm_audit)
    final_risky_modules = _final_risky_modules(final_rows)
    provisional_decision = _provisional_decision(final_selection, module_retention)
    verdict = "frozen_ready" if not blocking and not final_risky_modules else "not_frozen"
    if not blocking and final_risky_modules:
        verdict = "not_frozen_final_configs_need_recheck"
    elif blocking:
        verdict = "not_frozen_waiting_for_fair_candidate_and_final_evidence"

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "claim_boundary": (
            "The final method is a provisional lean route until fair baselines, CV-transfer candidates, "
            "candidate pruning, and final-method runs are complete."
        ),
        "dependencies": dependencies,
        "blocking_dependencies": [row["name"] for row in blocking],
        "final_rows": final_rows,
        "final_risky_modules": final_risky_modules,
        "candidate_pruning": {
            "verdict": candidate_pruning.get("verdict"),
            "pending": pruning_counts.get("pending"),
            "keep": pruning_counts.get("complete_keep"),
            "prune_or_rework": pruning_counts.get("complete_pruned"),
            "rescue": pruning_counts.get("complete_rescue"),
        },
        "cv_transfer": {
            "verdict": cv_protocol.get("verdict"),
            "routes": cv_counts.get("routes"),
            "implementation_ready": cv_counts.get("implementation_ready"),
            "metric_pending": cv_counts.get("metric_pending"),
        },
        "fair_execution": {
            "verdict": fair_priority.get("verdict"),
            "first_incomplete_stage": _first_incomplete_stage(fair_priority),
        },
        "provisional_decision": provisional_decision,
        "policy": [
            "Do not freeze the final architecture until fair ConvNeXt baselines are complete.",
            "Do not keep FrictionSet, generic DG losses, or full fusion unless later candidate metrics rescue them.",
            "Do not add wet optical, region-mixture, MixStyle, or mask-supervised routes to final claims until their candidate rows beat predeclared criteria.",
            "Final rows are paper evidence only after checkpoint, detailed test metrics, calibration, bootstrap, and audits exist.",
        ],
    }


def _candidate_pruning_ready(report: dict[str, Any]) -> bool:
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    pending = int(counts.get("pending", 0) or 0)
    completed_decisions = sum(
        int(counts.get(key, 0) or 0)
        for key in ["complete_keep", "complete_pruned", "complete_rescue", "complete_neutral"]
    )
    return pending == 0 and completed_decisions > 0


def _candidate_pruning_gap(report: dict[str, Any]) -> list[str]:
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    pending = int(counts.get("pending", 0) or 0)
    if pending:
        return [f"{pending} candidate/module decisions pending"]
    return []


def _final_rows(algorithm_audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = algorithm_audit.get("rows", []) if isinstance(algorithm_audit.get("rows"), list) else []
    out = []
    for row in rows:
        run = row.get("run")
        if run not in FINAL_ROWS:
            continue
        modules = row.get("modules", {}) if isinstance(row.get("modules"), dict) else {}
        out.append(
            {
                "run": run,
                "modules": {
                    key: bool(modules.get(key))
                    for key in [
                        "physics_texture",
                        "physics_quality_cues",
                        "wet_optical_quality_cues",
                        "friction_set",
                        "dg_losses",
                        "road_likelihood_prior",
                        "roi_attention_constraint",
                        "weak_view_consistency",
                        "mask_aware_consistency",
                        "coverage_aware_training",
                        "safety_weighted_coverage",
                        "visual_quality_weighted_coverage",
                    ]
                },
            }
        )
    return out


def _final_risky_modules(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risky_names = ["friction_set", "dg_losses"]
    findings = []
    for row in final_rows:
        modules = row.get("modules", {})
        active = [name for name in risky_names if modules.get(name)]
        if active:
            findings.append({"run": row.get("run"), "active_risky_modules": active})
    return findings


def _provisional_decision(final_selection: dict[str, Any], module_retention: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_selection_verdict": final_selection.get("verdict"),
        "module_retention_verdict": module_retention.get("verdict"),
        "top_completed": [
            row.get("method")
            for row in final_selection.get("provisional_top_completed", [])[:3]
            if isinstance(row, dict)
        ]
        if isinstance(final_selection.get("provisional_top_completed"), list)
        else [],
    }


def _first_incomplete_stage(report: dict[str, Any]) -> dict[str, Any] | None:
    stages = report.get("stages", []) if isinstance(report.get("stages"), list) else []
    return next((stage for stage in stages if stage.get("status") != "complete"), None)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Final Freeze Audit",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Dependencies",
        "",
        "| Dependency | Complete | Missing | Why |",
        "|---|---:|---|---|",
    ]
    for row in report["dependencies"]:
        lines.append(
            f"| {row['name']} | {row['complete']} | {_compact(row['missing'])} | {row['why']} |"
        )
    lines.extend(["", "## Final Config Snapshot", ""])
    lines.append("| Run | Active Modules |")
    lines.append("|---|---|")
    for row in report["final_rows"]:
        active = [name for name, enabled in row["modules"].items() if enabled]
        lines.append(f"| {row['run']} | {_compact(active)} |")
    lines.extend(["", "## Policy", ""])
    lines.extend(f"- {item}" for item in report["policy"])
    lines.append("")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _compact(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    if isinstance(value, dict):
        return "; ".join(f"{key}={val}" for key, val in value.items())
    return str(value)


if __name__ == "__main__":
    main()
