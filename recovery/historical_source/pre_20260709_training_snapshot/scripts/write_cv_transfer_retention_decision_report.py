from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_OUT_MD = DEFAULT_SUMMARY_DIR / "cv_transfer_retention_decision_report.md"
DEFAULT_OUT_JSON = DEFAULT_SUMMARY_DIR / "cv_transfer_retention_decision_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize result-driven retain/prune decisions for CV-transfer candidate groups."
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
    priority = _load_json(summary_dir / "cv_transfer_candidate_priority_report.json") or {}
    pruning = _load_json(summary_dir / "candidate_pruning_report.json") or {}
    artifact = _load_json(summary_dir / "artifact_contract_report.json") or {}
    fair = _load_json(summary_dir / "fair_comparison_execution_priority.json") or {}

    prune_by_run = {
        str(row.get("run")): row
        for row in pruning.get("rows", [])
        if isinstance(row, dict) and row.get("run")
    }
    artifact_by_run = {
        str(row.get("run") or row.get("name")): row
        for row in artifact.get("rows", artifact.get("run_contract", []))
        if isinstance(row, dict) and (row.get("run") or row.get("name"))
    }
    rows = []
    for group in priority.get("rows", []) if isinstance(priority.get("rows"), list) else []:
        if group.get("name") == "fair_comparison_before_claims":
            rows.append(_fair_row(group, artifact, fair))
            continue
        rows.append(_group_row(group, prune_by_run, artifact_by_run))

    blocks = [
        row["group"]
        for row in rows
        if row.get("level") == "block"
    ]
    verdict = "waiting_for_candidate_metrics"
    if blocks:
        verdict = "retention_decision_blocks"
    elif any(row.get("decision") in {"keep_route", "prune_route", "merge_only"} for row in rows):
        verdict = "retention_decisions_available"

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "claim_boundary": (
            "This report applies predeclared CV-transfer retain/prune logic to completed candidate metrics. "
            "Pending groups remain hypotheses, not final method claims."
        ),
        "rows": rows,
        "blocks": blocks,
        "dependencies": {
            "priority_verdict": priority.get("verdict"),
            "candidate_pruning_verdict": pruning.get("verdict"),
            "artifact_verdict": artifact.get("verdict"),
            "fair_execution_verdict": fair.get("verdict"),
        },
        "decision_scale": {
            "keep_route": "At least one completed candidate shows supported safety/generalization/coverage or audited interpretability evidence.",
            "rescue_route": "Some signal is promising but requires paired fair/final confirmation.",
            "merge_only": "No clear standalone gain; keep only if it simplifies or explains final method behavior.",
            "prune_route": "Completed candidates violate safety/coverage/width or shortcut rules.",
            "pending_metrics": "No completed candidate metrics yet.",
        },
    }


def _fair_row(group: dict[str, Any], artifact: dict[str, Any], fair: dict[str, Any]) -> dict[str, Any]:
    hard_groups = artifact.get("hard_groups", {}) if isinstance(artifact.get("hard_groups"), dict) else {}
    fair_group = hard_groups.get("single_dataset_fair", {}) if isinstance(hard_groups.get("single_dataset_fair"), dict) else {}
    runs = [str(run) for run in group.get("runs", [])]
    artifact_rows = artifact.get("rows", []) if isinstance(artifact.get("rows"), list) else []
    completed = [
        str(row.get("run") or row.get("name"))
        for row in artifact_rows
        if (row.get("run") or row.get("name")) in runs
        and row.get("contract_status") == "complete"
    ]
    if not completed:
        completed = fair_group.get("complete_runs") or group.get("complete_runs") or []
    missing = [run for run in runs if run not in set(completed)]
    if fair_group.get("missing") and len(completed) < len(runs):
        missing = fair_group.get("missing")
    active = [row.get("run") for row in fair.get("active_runs", []) if isinstance(row, dict) and row.get("run")]
    complete = len(completed) == len(runs)
    return {
        "priority": group.get("priority"),
        "group": group.get("name"),
        "level": "pass" if complete else "warn",
        "decision": "fair_claims_unlocked" if complete else "wait_for_fair_baselines",
        "complete_runs": completed,
        "active_runs": active,
        "missing_runs": missing,
        "keep_or_drop_basis": group.get("keep_rule"),
        "reason": (
            "Matched single-dataset FAF/ConvNeXt evidence is complete."
            if complete
            else "Fair baseline claims remain locked until all matched single-dataset rows complete."
        ),
    }


def _group_row(
    group: dict[str, Any],
    prune_by_run: dict[str, dict[str, Any]],
    artifact_by_run: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    runs = [str(run) for run in group.get("runs", [])]
    run_rows = []
    for run in runs:
        prune = prune_by_run.get(run, {})
        artifact = artifact_by_run.get(run, {})
        decision = prune.get("decision") or "pending_result"
        status = prune.get("progress_status") or artifact.get("contract_status") or artifact.get("progress_status") or "missing"
        run_rows.append(
            {
                "run": run,
                "status": status,
                "decision": decision,
                "reason": prune.get("reason"),
                "metrics": prune.get("metrics", {}),
                "deltas_vs_reference": prune.get("deltas_vs_reference", {}),
            }
        )

    complete_rows = [row for row in run_rows if row["status"] == "complete"]
    decisions = [row["decision"] for row in complete_rows]
    if not complete_rows:
        decision = "pending_metrics"
        level = "warn"
        reason = "No completed candidate metrics yet; retain only as a queued hypothesis."
    elif any(item == "keep_candidate" for item in decisions):
        decision = "keep_route"
        level = "pass"
        reason = "At least one completed candidate passed the predeclared candidate-pruning criteria."
    elif any(item == "rescue_candidate" for item in decisions):
        decision = "rescue_route"
        level = "warn"
        reason = "A completed candidate has a partial signal but still needs fair/final confirmation."
    elif all(item == "prune_or_rework" for item in decisions):
        decision = "prune_route"
        level = "block"
        reason = "All completed candidates in this group failed safety, coverage, width, or shortcut criteria."
    elif complete_rows and all(item in {"neutral_or_merge", "prune_or_rework"} for item in decisions):
        decision = "merge_only"
        level = "warn"
        reason = "Completed candidates do not justify a standalone module; merge only if it simplifies the final method."
    else:
        decision = "pending_mixed"
        level = "warn"
        reason = "Some results exist but the group does not yet have a clear retain/prune decision."

    return {
        "priority": group.get("priority"),
        "group": group.get("name"),
        "cv_source": group.get("cv_source"),
        "level": level,
        "decision": decision,
        "reason": reason,
        "complete_runs": [row["run"] for row in complete_rows],
        "missing_runs": [row["run"] for row in run_rows if row["status"] != "complete"],
        "primary_metrics": group.get("primary_metrics", []),
        "keep_rule": group.get("keep_rule"),
        "drop_rule": group.get("drop_rule"),
        "rapid_prune_trigger": group.get("rapid_prune_trigger"),
        "run_rows": run_rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CV Transfer Retention Decision Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Verdict: `{report['verdict']}`",
        "",
        f"Claim boundary: {report['claim_boundary']}",
        "",
        "## Group Decisions",
        "",
        "| Priority | Group | Decision | Level | Complete | Missing | Reason |",
        "|---:|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {priority} | {group} | `{decision}` | `{level}` | {complete} | {missing} | {reason} |".format(
                priority=row.get("priority"),
                group=row.get("group"),
                decision=row.get("decision"),
                level=row.get("level"),
                complete=_join(row.get("complete_runs", [])),
                missing=_join(row.get("missing_runs", [])),
                reason=row.get("reason", "-"),
            )
        )
    lines.extend(["", "## Run-Level Decisions", ""])
    for row in report["rows"]:
        lines.append(f"### {row.get('group')}")
        lines.append(f"- CV source: {row.get('cv_source', '-')}")
        lines.append(f"- Keep rule: {row.get('keep_rule') or row.get('keep_or_drop_basis') or '-'}")
        lines.append(f"- Drop rule: {row.get('drop_rule') or '-'}")
        lines.append(f"- Rapid prune trigger: {row.get('rapid_prune_trigger') or '-'}")
        run_rows = row.get("run_rows", [])
        if run_rows:
            lines.append("")
            lines.append("| Run | Status | Decision | Reason |")
            lines.append("|---|---|---|---|")
            for run in run_rows:
                lines.append(
                    f"| `{run.get('run')}` | `{run.get('status')}` | `{run.get('decision')}` | {run.get('reason') or '-'} |"
                )
        lines.append("")
    lines.extend(["## Decision Scale", ""])
    for key, value in report["decision_scale"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _join(items: list[Any]) -> str:
    cleaned = [str(item) for item in items if item]
    return ", ".join(cleaned) if cleaned else "-"


if __name__ == "__main__":
    main()
