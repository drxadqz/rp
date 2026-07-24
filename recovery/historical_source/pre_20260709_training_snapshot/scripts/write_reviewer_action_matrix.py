from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")


GROUP_PURPOSES = {
    "p0_ablation_complete": (
        "Complete the core ablation table.",
        "Proves which proposed modules are useful enough to keep.",
    ),
    "lodo_complete": (
        "Run leave-one-dataset-out generalization.",
        "Tests whether the model transfers across public datasets, especially held-out RoadSaW.",
    ),
    "fair_single_dataset_complete": (
        "Run matched single-dataset FAF and ConvNeXt baselines.",
        "Provides fair same-split comparison against a strong public visual baseline.",
    ),
    "candidate_path_complete": (
        "Run P1/P2/P3 robustness candidates.",
        "Searches for a publishable route to reduce dataset shortcut, RoadSaW wetness failures, and interval undercoverage.",
    ),
    "final_method_complete": (
        "Run the selected final lean method.",
        "Produces the final paper tables after module pruning and candidate selection.",
    ),
}


RUN_ROLES = {
    "v5_full_faf": "Full P0 model row; closes the main ablation table.",
    "lodo_roadsaw_full_faf": "Most important held-out-domain stress test because RoadSaW wetness is the current failure mode.",
    "v6_full_faf_fourier": "Tests low-frequency style perturbation against dataset shortcut.",
    "v7_full_faf_fourier_dann": "Tests domain-adversarial shortcut suppression after style augmentation.",
    "v8_full_faf_fourier_roadprior": "Tests whether road-likelihood/bottom-road priors ground local evidence.",
    "v9_full_faf_roadsaw_hard_sampling": "Targets damp/wet/very_wet confusion and RoadSaW rare states.",
    "v10_full_faf_consistency": "Tests weak-view prediction and attention consistency.",
    "v11_full_faf_domain_adapter": "Allows small dataset-specific style adapters while sharing friction semantics.",
    "v12_full_faf_roi_interval_safety": "Combines road ROI grounding with safety-weighted interval coverage.",
    "v13_lean_physics_evidence": "Prunes unstable FrictionSet/DG stack while keeping physics texture and evidence.",
    "v14_lean_road_roi_safety": "Likely final candidate: lean physics/evidence with ROI, wetness ordinal, and safety interval losses.",
    "v15_lean_bottom_square_style_safety": "Tests bottom-square road input canonicalization to reduce native size/aspect shortcut while keeping the lean safety route.",
    "v16_lean_bottom_square_color_constancy_safety": "Tests soft Gray-World color constancy on top of bottom-square canonicalization to reduce camera/color dataset shortcut.",
    "v17_lean_quality_physics_safety": "Tests explicit near-white, low-texture, and wet-road regional quality cues inside the PhysicsTexture branch.",
    "v18_lean_mixstyle_quality_safety": "Tests feature-statistics mixing as a lightweight dataset-style shortcut mitigation probe.",
    "v19_lean_state_contrast_quality_safety": "Tests cross-dataset same-state contrastive alignment to reduce dataset shortcut while preserving road-state separation.",
    "v20_lean_interval_order_quality_safety": "Tests pairwise non-overlapping weak friction-interval ordering as a physics regularizer from public proxy labels.",
    "v21_lean_quality_uncertainty_safety": "Tests image-derived near-white, low-texture, and specular-highlight coverage weighting for conservative weak-friction intervals on visually ambiguous road states.",
    "v22_lean_quality_order_contrast_safety": "Tests whether visual-quality uncertainty, weak interval ordering, and small same-state contrast jointly improve ambiguous wet/snow slices without extra shortcut leakage.",
    "v23_lean_region_mixture_evidence_safety": "Tests whether segmentation-style local region-mixture evidence improves wet/snow/low-texture interval coverage without pixel-level labels.",
    "v24_lean_multi_query_region_evidence_safety": "Tests whether mask-query-style multi-region evidence separates wet/snow/glare patches and expands friction intervals only when local evidence disagrees.",
    "v25_lean_masked_query_consistency_safety": "Tests whether MIC-style masked weak-view consistency makes multi-query road evidence stable under local occlusion and camera-style perturbations.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "reviewer_action_matrix.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "reviewer_action_matrix.json")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_report(summary_dir: Path) -> dict[str, Any]:
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    completeness = _load_json(summary_dir / "protocol_completeness.json") or {}
    gate = _load_json(summary_dir / "topvenue_readiness_gate.json") or {}
    p0 = _load_json(summary_dir / "p0_claim_report.json") or {}
    selection = _load_json(summary_dir / "final_method_selection_report.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    wetness = _load_json(summary_dir / "wetness_state_report.json") or {}
    intervals = _load_json(summary_dir / "interval_quality_report.json") or {}
    open_source = _load_json(summary_dir / "open_source_reproducibility_plan.json") or {}
    queue_path = summary_dir / "queue_recovery_report.json"
    watch_path = summary_dir / "active_training_watch_report.json"
    queue = _load_json(queue_path) or {}
    watch = _load_json(watch_path) or {}
    if _is_older(watch_path, queue_path):
        watch = {}
    live = _load_active_live_trend(summary_dir, queue, watch)

    missing_runs = _missing_by_requirement(completeness)
    completed = _completed_p0(summary)
    module_actions = _module_actions(p0)
    risks = _risk_items(gate, shortcut, wetness, intervals)
    source_roles = _source_roles(open_source)
    active_row = _active_row(queue, watch)
    active_state = _active_training_state(queue, active_row)

    queue_counts = _queue_counts(queue)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "readiness": {
            "verdict": gate.get("verdict"),
            "num_blocks": gate.get("num_blocks"),
            "num_warnings": gate.get("num_warnings"),
        },
        "queue": {
            "summary": queue.get("summary", {}) or queue_counts,
            "active": _active_run(queue, watch),
            "live": _live_compact(live, active_row, watch, active_state),
        },
        "missing_by_requirement": missing_runs,
        "p0_completed": completed,
        "module_actions": module_actions,
        "risks": risks,
        "source_roles": source_roles,
        "acceptance_tests": _acceptance_tests(),
        "next_actions": _next_actions(missing_runs, risks, module_actions),
    }


def render_markdown(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    queue = report["queue"]
    live = queue.get("live") or {}
    lines = [
        "# Reviewer Action Matrix",
        "",
        f"Generated at: {report['generated_at']}",
        "",
        "This is a reviewer-facing control sheet for the weak-supervised visual friction-affordance project. "
        "It separates evidence that already exists from evidence that still has to be produced.",
        "",
        "## Current Verdict",
        "",
        f"- Readiness: `{readiness.get('verdict')}` with `{readiness.get('num_blocks')}` blocks and `{readiness.get('num_warnings')}` warnings.",
        f"- Active run: `{queue.get('active') or '-'}`.",
        f"- Live progress: `{live.get('progress') or '-'}`.",
        f"- Latest completed epoch: `{live.get('latest_epoch') or '-'}`; val risk acc `{_fmt_pct(live.get('risk_acc'))}`; raw interval coverage `{_fmt_pct(live.get('raw_coverage'))}`.",
        "",
        "## Missing Evidence",
        "",
        "| Requirement | Why It Matters | Missing Runs |",
        "|---|---|---|",
    ]
    for item in report["missing_by_requirement"]:
        role = GROUP_PURPOSES.get(item["requirement"], ("Complete this group.", "Required for the paper protocol."))
        lines.append(
            "| {req} | {why} | {missing} |".format(
                req=item["requirement"],
                why=role[1],
                missing=", ".join(f"`{run}`" for run in item["missing"]) or "-",
            )
        )
    lines.extend(["", "## Critical Runs", "", "| Run | Role |", "|---|---|"])
    for item in report["missing_by_requirement"]:
        for run in item["missing"]:
            if run in RUN_ROLES:
                lines.append(f"| `{run}` | {RUN_ROLES[run]} |")
    lines.extend(["", "## P0 Module Decisions", "", "| Module | Current Action | Reason |", "|---|---|---|"])
    for item in report["module_actions"]:
        lines.append(f"| {item['module']} | `{item['action']}` | {item['reason']} |")
    lines.extend(["", "## Completed P0 Snapshot", "", "| Method | friction F1 | risk F1 | low recall | calibrated coverage | worst dataset F1 |", "|---|---:|---:|---:|---:|---:|"])
    for row in report["p0_completed"]:
        lines.append(
            "| {method} | {friction} | {risk} | {low} | {cov} | {worst} |".format(
                method=row.get("method"),
                friction=_fmt_pct(row.get("friction_macro_f1") or row.get("friction_f1")),
                risk=_fmt_pct(row.get("risk_macro_f1") or row.get("risk_f1")),
                low=_fmt_pct(row.get("low_friction_recall")),
                cov=_fmt_pct(row.get("calibrated_coverage")),
                worst=_fmt_pct(row.get("worst_dataset_f1")),
            )
        )
    lines.extend(["", "## Current Failure Modes", ""])
    for risk in report["risks"]:
        lines.append(f"- `{risk['name']}`: {risk['message']}")
    lines.extend(["", "## External Source Roles", "", "| Source | Role | Status |", "|---|---|---|"])
    for item in report["source_roles"]:
        lines.append(f"| {item['name']} | {item['role']} | `{item['status']}` |")
    lines.extend(["", "## Acceptance Tests", ""])
    for item in report["acceptance_tests"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in report["next_actions"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _missing_by_requirement(completeness: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for req in completeness.get("requirements", []):
        missing = req.get("missing") or []
        if missing:
            out.append({"requirement": req.get("name"), "missing": list(missing)})
    return out


def _completed_p0(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("core_ablation") or summary.get("ablation") or []
    return [row for row in rows if row.get("status") == "complete"][:6]


def _module_actions(p0: dict[str, Any]) -> list[dict[str, str]]:
    actions = []
    for row in p0.get("adjacent_deltas", []):
        module = str(row.get("module") or "-")
        rec = str(row.get("claim_recommendation") or row.get("recommendation") or "pending")
        reason = _module_reason(module, rec)
        actions.append({"module": module, "action": rec, "reason": reason})
    if not actions:
        actions.append({"module": "Full fusion", "action": "pending", "reason": "Waiting for full model test/bootstrap/audit artifacts."})
    return actions


def _module_reason(module: str, recommendation: str) -> str:
    if "PhysicsTexture" in module and recommendation == "keep":
        return "Currently improves safety and generalization metrics in the completed P0 rows."
    if "Evidence" in module and recommendation == "keep":
        return "Keeps the interpretable local-evidence story, but final retention depends on LODO and evidence-map audits."
    if recommendation in {"rework_or_remove", "remove"}:
        return "The module has not yet earned its complexity under the safety/generalization rule."
    return "Decision remains conditional on the full P0 and LODO results."


def _risk_items(
    gate: dict[str, Any],
    shortcut: dict[str, Any],
    wetness: dict[str, Any],
    intervals: dict[str, Any],
) -> list[dict[str, str]]:
    out = []
    for item in gate.get("gates", []):
        if item.get("level") == "block":
            out.append({"name": str(item.get("name")), "message": str(item.get("message"))})
    if shortcut.get("num_high_shortcut"):
        out.append(
            {
                "name": "dataset_shortcut_high",
                "message": f"{shortcut.get('num_high_shortcut')} completed rows exceed the dataset-ID probe threshold.",
            }
        )
    if wetness.get("num_watchlist"):
        out.append(
            {
                "name": "roadsaw_wetness_weak",
                "message": f"{wetness.get('num_watchlist')} completed rows are on the RoadSaW wetness watchlist.",
            }
        )
    if intervals.get("num_watchlist_items"):
        out.append(
            {
                "name": "conditional_interval_undercoverage",
                "message": f"{intervals.get('num_watchlist_items')} conditional interval cells are under the coverage watch threshold.",
            }
        )
    return out


def _source_roles(open_source: dict[str, Any]) -> list[dict[str, str]]:
    rows = open_source.get("rows") or open_source.get("sources") or open_source.get("source_map") or []
    out = []
    for row in rows:
        out.append(
            {
                "name": str(row.get("name") or row.get("source") or "-"),
                "role": str(row.get("venue_or_role") or row.get("role") or row.get("project_role") or "-"),
                "status": str(row.get("integration_status") or row.get("status") or "-"),
            }
        )
    return out


def _active_run(queue: dict[str, Any], watch: dict[str, Any] | None = None) -> str | None:
    row = _active_row(queue, watch)
    if row:
        return str(row.get("name") or row.get("run") or "")
    return None


def _active_row(queue: dict[str, Any], watch: dict[str, Any] | None = None) -> dict[str, Any] | None:
    watch_active = (watch or {}).get("active") or {}
    if watch_active.get("name"):
        return {
            "name": watch_active.get("name"),
            "status": watch_active.get("status"),
            "phase": watch_active.get("phase"),
            "active_epoch": watch_active.get("epoch"),
            "active_epochs": watch_active.get("epochs"),
            "active_step": watch_active.get("step"),
            "active_steps": watch_active.get("steps"),
        }
    rows = queue.get("active_rows") or []
    if rows:
        return rows[0]
    rows = queue.get("queue_order") or []
    for row in rows:
        if row.get("status") == "running_or_partial":
            return row
    next_incomplete = queue.get("next_incomplete") or {}
    if next_incomplete.get("status") == "running_or_partial":
        return next_incomplete
    return None


def _queue_counts(queue: dict[str, Any]) -> dict[str, int]:
    rows = queue.get("queue_order") or []
    if not rows:
        return {
            "total": int(queue.get("num_total") or 0),
            "complete": int(queue.get("num_complete") or 0),
            "partial_or_running": int(queue.get("num_partial") or 0),
            "missing": int(queue.get("num_missing") or 0),
        }
    return {
        "total": len(rows),
        "complete": sum(1 for row in rows if row.get("status") == "complete"),
        "partial_or_running": sum(1 for row in rows if row.get("status") == "running_or_partial"),
        "missing": sum(1 for row in rows if row.get("status") == "missing"),
    }


def _acceptance_tests() -> list[str]:
    return [
        "P0 table must include Global-only through Full model with bootstrap intervals and module recommendations.",
        "Held-out RoadSaW LODO must be reported before any cross-dataset generalization claim.",
        "Single-dataset FAF and ConvNeXt must use identical splits, labels, metrics, and calibration protocol.",
        "A module is kept only if it improves safety/generalization or interpretability without a major worst-dataset or shortcut regression.",
        "Coverage must be reported with interval width; weak friction intervals must not be described as measured tire-road friction.",
        "Dataset-ID shortcut must either be reduced by P1/final candidates or explicitly framed as a limitation.",
    ]


def _next_actions(
    missing: list[dict[str, Any]],
    risks: list[dict[str, str]],
    module_actions: list[dict[str, str]],
) -> list[str]:
    actions = []
    if any("v5_full_faf" in item["missing"] for item in missing):
        actions.append("Let `v5_full_faf` finish, then run postprocess to close the P0 ablation table.")
    if any(item["requirement"] == "lodo_complete" for item in missing):
        actions.append("Finish the remaining LODO row and use the completed held-out RoadSaW failure as the main shortcut/wetness stress test.")
    if any(item["requirement"] == "fair_single_dataset_complete" for item in missing):
        actions.append("Run matched single-dataset FAF vs ConvNeXt to create fair public-dataset comparison tables.")
    if any("dataset_shortcut" in risk["name"] for risk in risks):
        actions.append("Prioritize Fourier style jitter, wetness-conditioned alignment, DANN/adapters, and road ROI candidates.")
    if any(item["action"] == "rework_or_remove" for item in module_actions):
        actions.append("Treat FrictionSet/DG as provisional; remove or merge them unless candidate/final rows recover safety and worst-dataset metrics.")
    actions.append("Select the final method only after P1/P2/P3 candidates are ranked by risk F1, low-friction recall, worst-dataset F1, coverage-width, and dataset-ID probe.")
    return actions


def _live_compact(
    live: dict[str, Any],
    active_row: dict[str, Any] | None = None,
    watch: dict[str, Any] | None = None,
    active_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (watch or {}).get("active"):
        latest = (watch or {}).get("latest_completed_epoch") or {}
        state_metrics = _training_state_metrics(active_state)
        return {
            "progress": _row_progress(active_row),
            "latest_epoch": state_metrics.get("epoch") or latest.get("epoch"),
            "risk_acc": state_metrics.get("risk_acc") or latest.get("val_acc_risk"),
            "raw_coverage": state_metrics.get("raw_coverage") or latest.get("val_mu_interval_coverage"),
        }
    active_name = str(active_row.get("name") or active_row.get("run") or "") if active_row else ""
    live_name = str(live.get("run") or "")
    if active_name and live_name and active_name != live_name:
        state_metrics = _training_state_metrics(active_state)
        return {
            "progress": _row_progress(active_row),
            "latest_epoch": state_metrics.get("epoch") or (active_row.get("epoch") if active_row else None),
            "risk_acc": state_metrics.get("risk_acc"),
            "raw_coverage": state_metrics.get("raw_coverage"),
        }
    active = live.get("active_progress") or {}
    latest = live.get("latest_completed_epoch") or {}
    val = latest.get("val") or {}
    progress = None
    if active:
        epoch = active.get("epoch")
        epochs = active.get("epochs")
        step = active.get("step")
        steps = active.get("steps")
        if step is not None and steps is not None:
            progress = f"epoch {epoch}/{epochs}, step {step}/{steps}"
        else:
            progress = f"epoch {epoch}/{epochs}"
    return {
        "progress": progress or _row_progress(active_row),
        "latest_epoch": latest.get("epoch") or _training_state_metrics(active_state).get("epoch"),
        "risk_acc": val.get("acc_risk") or _training_state_metrics(active_state).get("risk_acc"),
        "raw_coverage": val.get("mu_interval_coverage") or _training_state_metrics(active_state).get("raw_coverage"),
    }


def _active_training_state(queue: dict[str, Any], active_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not active_row:
        return None
    root = queue.get("root")
    name = active_row.get("name") or active_row.get("run")
    if not root or not name:
        return None
    return _load_json(Path(root) / str(name) / "training_state.json")


def _training_state_metrics(state: dict[str, Any] | None) -> dict[str, Any]:
    if not state:
        return {}
    val = state.get("val_metrics") or {}
    return {
        "epoch": state.get("epoch"),
        "risk_acc": val.get("acc_risk"),
        "raw_coverage": val.get("mu_interval_coverage"),
    }


def _row_progress(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    epoch = row.get("active_epoch") or row.get("epoch")
    epochs = row.get("active_epochs") or row.get("epochs")
    step = row.get("active_step")
    steps = row.get("active_steps")
    if epoch is None:
        return None
    if step is not None and steps is not None:
        return f"epoch {epoch}/{epochs}, step {step}/{steps}"
    return f"epoch {epoch}/{epochs}"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_active_live_trend(summary_dir: Path, queue: dict[str, Any], watch: dict[str, Any]) -> dict[str, Any]:
    candidates: list[str] = []
    active = watch.get("active") if isinstance(watch, dict) else {}
    if isinstance(active, dict) and active.get("name"):
        candidates.append(str(active["name"]))
    if isinstance(queue, dict):
        for row in queue.get("active_rows", []) or []:
            if row.get("name"):
                candidates.append(str(row["name"]))
    candidates.append("v5_full_faf")

    seen: set[str] = set()
    for run_name in candidates:
        if run_name in seen:
            continue
        seen.add(run_name)
        trend = _load_json(summary_dir / f"{run_name}_live_training_trend.json")
        if isinstance(trend, dict) and trend:
            return trend
    return {}


def _is_older(candidate: Path, reference: Path) -> bool:
    if not candidate.exists() or not reference.exists():
        return False
    try:
        return candidate.stat().st_mtime < reference.stat().st_mtime
    except OSError:
        return False


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


if __name__ == "__main__":
    main()
