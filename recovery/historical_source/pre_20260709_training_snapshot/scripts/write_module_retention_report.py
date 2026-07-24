from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")

PRIMARY_METRICS = [
    "friction_macro_f1_delta",
    "risk_macro_f1_delta",
    "low_friction_recall_delta",
    "calibrated_coverage_delta",
    "worst_dataset_f1_delta",
]

MODULE_RESCUE_TESTS = {
    "PhysicsTexture": [
        "Full model and LODO rows must preserve the current worst-dataset and low-friction gains.",
        "Matched single-dataset rows must not show that the gain is only from pooled multi-dataset training.",
    ],
    "FrictionSet": [
        "Must recover worst-dataset F1 or held-out RoadSaW while preserving its interval-coverage benefit.",
        "Must not widen intervals without improving conditional coverage-width tradeoff.",
    ],
    "DG losses": [
        "A redesigned DG component must reduce dataset-ID balanced accuracy without harming risk F1 or low-friction recall.",
        "Held-out RoadSaW must improve relative to the non-DG counterpart before any DG claim is kept.",
    ],
    "EvidenceField aux": [
        "Evidence maps must stay road/bottom-ROI focused and be supported by quantitative success/failure audit.",
        "LODO or final lean rows must show that interpretability is not bought by worse RoadSaW performance.",
    ],
    "Full fusion": [
        "Full model must complete test, calibration, bootstrap, and audit artifacts.",
        "Full model must beat or match the best lean/simple row on safety/generalization score.",
    ],
    "Fourier style jitter": [
        "Dataset-ID probe should drop while risk F1, low-friction recall, and worst-dataset F1 remain competitive.",
        "Held-out RoadSaW must not collapse after style perturbation.",
    ],
    "Domain-adversarial training": [
        "Domain head should reduce dataset predictability without erasing physical wet/snow/risk distinctions.",
        "Compare v7 against v6 to isolate the adversarial component.",
    ],
    "Road prior": [
        "Attention/evidence should shift toward plausible road pixels without hurting low-friction recall.",
        "Compare v8 against v6 to isolate the road-prior contribution.",
    ],
    "Wet-state hard sampling": [
        "RoadSaW damp/wet/very_wet macro-F1 and ordinal MAE should improve without overfitting RoadSaW.",
        "Compare v9 against v8/v6 and inspect held-out RoadSaW.",
    ],
    "Weak-view consistency": [
        "Prediction and evidence consistency should improve robustness without narrowing intervals unsafely.",
        "Compare v10 against the nearest Fourier/road-prior counterpart.",
    ],
    "Domain-specific adapter": [
        "Adapters may absorb camera style, but shared friction semantics must still transfer in LODO.",
        "Keep only if single-dataset and LODO both improve or shortcut drops cleanly.",
    ],
    "ROI interval safety": [
        "Conditional coverage must improve with reasonable width, especially RoadSaW very_wet and RoadSC snow cells.",
        "Keep only if it improves coverage-width tradeoff, not merely coverage by broadening intervals.",
    ],
    "External road-mask supervision": [
        "CLIPSeg/SAM/SegFormer pseudo masks must pass a small cross-dataset audit before any full preprocessing.",
        "Mask-supervised EvidenceField must improve attention-on-road, RoadSaW wet/white slices, or interval coverage-width without hurting low-friction recall.",
        "Compare against no-mask EvidenceField, heuristic road prior, and v23 region-mixture evidence before retaining it.",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "module_retention_report.md")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "module_retention_report.json")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_SUMMARY_DIR / "module_retention_report.csv")
    args = parser.parse_args()

    report = build_report(args.summary_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    _write_csv(args.out_csv, report["rows"])
    print(render_markdown(report))


def build_report(summary_dir: Path) -> dict[str, Any]:
    p0 = _load_json(summary_dir / "p0_claim_report.json") or {}
    final_selection = _load_json(summary_dir / "final_method_selection_report.json") or {}
    completeness = _load_json(summary_dir / "protocol_completeness.json") or {}
    shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    wetness = _load_json(summary_dir / "wetness_state_report.json") or {}
    interval = _load_json(summary_dir / "interval_quality_report.json") or {}
    segmentation_transfer = _load_json(summary_dir / "segmentation_transfer_config_audit.json") or {}
    segmentation_queue = _load_json(summary_dir / "segmentation_transfer_queue_status.json") or {}

    requirements = {row.get("name"): row for row in completeness.get("requirements", [])}
    missing_context = {
        "p0_complete": _is_complete(requirements.get("p0_ablation_complete")),
        "lodo_complete": _is_complete(requirements.get("lodo_complete")),
        "fair_single_dataset_complete": _is_complete(requirements.get("fair_single_dataset_complete")),
        "candidate_path_complete": _is_complete(requirements.get("candidate_path_complete")),
        "final_method_complete": _is_complete(requirements.get("final_method_complete")),
    }

    p0_rows_by_method = {
        str(row.get("method")): row
        for row in p0.get("rows", [])
        if isinstance(row, dict) and row.get("method")
    }
    best_completed_p0 = _best_completed_p0_row(p0_rows_by_method, final_selection)

    rows = []
    for delta in p0.get("adjacent_deltas", []):
        rows.append(_row_from_delta(delta, missing_context, p0_rows_by_method, best_completed_p0))

    seen = {row["module"] for row in rows}
    for item in final_selection.get("module_decisions", []):
        module = str(item.get("module") or "")
        if not module or module in seen:
            continue
        rows.append(_row_from_candidate(item, missing_context))
        seen.add(module)
    if "External road-mask supervision" not in seen:
        rows.append(_row_from_segmentation_transfer(segmentation_transfer, segmentation_queue, missing_context))
        seen.add("External road-mask supervision")

    risk_context = _risk_context(shortcut, wetness, interval, segmentation_transfer, segmentation_queue)
    verdict = _verdict(rows, missing_context)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_dir": str(summary_dir),
        "verdict": verdict,
        "hard_rule": (
            "A module is final-keep only after P0, LODO, fair single-dataset comparisons, "
            "and final-method evidence show a safety/generalization or interpretability gain "
            "without major worst-domain, low-friction, shortcut, or interval-width regression."
        ),
        "missing_context": missing_context,
        "risk_context": risk_context,
        "rows": rows,
        "recommended_final_route_now": _recommended_route(rows, missing_context),
    }


def _row_from_delta(
    delta: dict[str, Any],
    missing_context: dict[str, bool],
    p0_rows_by_method: dict[str, dict[str, Any]],
    best_completed_p0: dict[str, Any] | None,
) -> dict[str, Any]:
    module = str(delta.get("module") or "-")
    rec = str(delta.get("claim_recommendation") or "pending")
    evidence = _delta_evidence(delta)
    action = _action(module, rec, evidence, missing_context)
    current_method = str(delta.get("current") or "")
    absolute = _absolute_context(current_method, p0_rows_by_method, best_completed_p0)
    summary = action["reason"]
    if absolute.get("summary"):
        summary = f"{summary} Absolute P0 check: {absolute['summary']}"
    return {
        "module": module,
        "current_method": current_method,
        "current_decision": action["decision"],
        "base_recommendation": rec,
        "evidence_summary": summary,
        "absolute_context_summary": absolute.get("summary"),
        "best_completed_p0_method": absolute.get("best_method"),
        "selection_score_gap_to_best": absolute.get("score_gap"),
        "rescue_or_confirmation_tests": MODULE_RESCUE_TESTS.get(module, []),
        "next_required_evidence": action["next_required_evidence"],
        "delta_friction_f1": _num(delta.get("friction_macro_f1_delta")),
        "delta_risk_f1": _num(delta.get("risk_macro_f1_delta")),
        "delta_low_recall": _num(delta.get("low_friction_recall_delta")),
        "delta_calibrated_coverage": _num(delta.get("calibrated_coverage_delta")),
        "delta_calibrated_width": _num(delta.get("calibrated_width_delta")),
        "delta_worst_dataset_f1": _num(delta.get("worst_dataset_f1_delta")),
        "delta_dataset_id_bal_acc": _num(delta.get("dataset_id_balanced_accuracy_delta")),
    }


def _row_from_candidate(item: dict[str, Any], missing_context: dict[str, bool]) -> dict[str, Any]:
    module = str(item.get("module") or "-")
    rec = str(item.get("decision") or "pending")
    action = _action(module, rec, {}, missing_context)
    return {
        "module": module,
        "current_method": str(item.get("method") or ""),
        "current_decision": action["decision"],
        "base_recommendation": rec,
        "evidence_summary": action["reason"],
        "absolute_context_summary": None,
        "best_completed_p0_method": None,
        "selection_score_gap_to_best": None,
        "rescue_or_confirmation_tests": MODULE_RESCUE_TESTS.get(module, []),
        "next_required_evidence": action["next_required_evidence"],
        "delta_friction_f1": _num(item.get("delta_friction_macro_f1")),
        "delta_risk_f1": _num(item.get("delta_risk_macro_f1")),
        "delta_low_recall": _num(item.get("delta_low_friction_recall")),
        "delta_calibrated_coverage": _num(item.get("delta_calibrated_coverage")),
        "delta_calibrated_width": _num(item.get("delta_calibrated_width")),
        "delta_worst_dataset_f1": _num(item.get("delta_worst_dataset_risk_f1")),
        "delta_dataset_id_bal_acc": _num(item.get("delta_dataset_id_balanced_accuracy")),
    }


def _row_from_segmentation_transfer(
    segmentation_transfer: dict[str, Any],
    segmentation_queue: dict[str, Any],
    missing_context: dict[str, bool],
) -> dict[str, Any]:
    batch = segmentation_transfer.get("batch_report", {}) if isinstance(segmentation_transfer.get("batch_report"), dict) else {}
    pseudo_loss = _num(batch.get("loss_evidence_attention_pseudo_road"))
    road_mass = _num(batch.get("attention_pseudo_road_mass"))
    queue_after = segmentation_queue.get("after_queue", {}) if isinstance(segmentation_queue.get("after_queue"), dict) else {}
    promotion = segmentation_queue.get("promotion", {}) if isinstance(segmentation_queue.get("promotion"), dict) else {}
    verdict = segmentation_transfer.get("verdict", "missing")
    if verdict == "pass":
        decision = "configured_pending_metrics"
        summary = (
            "Engineering path is smoke-tested: cached road_mask_path manifests load into the dataloader "
            f"and activate pseudo-road attention loss ({pseudo_loss if pseudo_loss is not None else '-'}). "
            "This is not performance evidence; external CLIPSeg/SAM mask quality and candidate metrics are still pending."
        )
    else:
        decision = "pending"
        summary = "Mask-supervised EvidenceField path is not yet smoke-tested."
    next_required = [
        "Wait for the full formal GPU queue to finish before running CLIPSeg/SAM dependency installation and audit.",
        "Run external segmentation mask audit across RSCD/RoadSaW/RoadSC and inspect overlays.",
        "Generate bounded cached-mask candidate and compare against v23/no-mask EvidenceField on RoadSaW wet/white slices and interval coverage-width.",
        "Promote to formal queue only if the bounded candidate beats lightweight ROI/region-mixture evidence.",
    ]
    if not missing_context.get("final_method_complete"):
        next_required.append("Verify any retained mask-supervised component in final LODO and matched single-dataset rows.")
    return {
        "module": "External road-mask supervision",
        "current_method": str(segmentation_transfer.get("config") or ""),
        "current_decision": decision,
        "base_recommendation": "configured" if verdict == "pass" else "pending",
        "evidence_summary": summary,
        "absolute_context_summary": None,
        "best_completed_p0_method": None,
        "selection_score_gap_to_best": None,
        "rescue_or_confirmation_tests": MODULE_RESCUE_TESTS["External road-mask supervision"],
        "next_required_evidence": next_required,
        "delta_friction_f1": None,
        "delta_risk_f1": None,
        "delta_low_recall": None,
        "delta_calibrated_coverage": None,
        "delta_calibrated_width": None,
        "delta_worst_dataset_f1": None,
        "delta_dataset_id_bal_acc": None,
        "segmentation_pseudo_loss": pseudo_loss,
        "segmentation_attention_road_mass": road_mass,
        "segmentation_audit_wait_pids": queue_after.get("wait_pids"),
        "segmentation_promotion_wait_pid": promotion.get("wait_pid"),
    }


def _action(
    module: str,
    recommendation: str,
    evidence: dict[str, Any],
    missing_context: dict[str, bool],
) -> dict[str, Any]:
    missing_evidence = [name for name, complete in missing_context.items() if not complete]
    next_required = _next_required(module, missing_context)
    if recommendation == "keep":
        if missing_evidence:
            return {
                "decision": "provisional_keep",
                "reason": "Positive P0 signal, but final retention still needs LODO, fair baseline, and final-method evidence.",
                "next_required_evidence": next_required,
            }
        return {
            "decision": "final_keep_candidate",
            "reason": "All hard evidence groups are complete and the module has positive retained evidence.",
            "next_required_evidence": [],
        }
    if recommendation in {"rework_or_remove", "remove"}:
        return {
            "decision": "provisional_remove_or_merge",
            "reason": _remove_reason(evidence),
            "next_required_evidence": next_required,
        }
    return {
        "decision": "pending",
        "reason": "The required adjacent or candidate rows are incomplete.",
        "next_required_evidence": next_required,
    }


def _delta_evidence(delta: dict[str, Any]) -> dict[str, Any]:
    values = {key: _num(delta.get(key)) for key in PRIMARY_METRICS}
    return values


def _remove_reason(evidence: dict[str, Any]) -> str:
    reasons = []
    if _lt(evidence.get("friction_macro_f1_delta"), -0.02):
        reasons.append("friction F1 drops by more than 2 points")
    if _lt(evidence.get("risk_macro_f1_delta"), -0.02):
        reasons.append("risk F1 drops by more than 2 points")
    if _lt(evidence.get("low_friction_recall_delta"), -0.02):
        reasons.append("low-friction recall drops by more than 2 points")
    if _lt(evidence.get("worst_dataset_f1_delta"), -0.02):
        reasons.append("worst-dataset F1 drops")
    if not reasons:
        reasons.append("current P0 evidence does not justify final complexity")
    return "; ".join(reasons) + "."


def _next_required(module: str, missing_context: dict[str, bool]) -> list[str]:
    out = []
    if not missing_context.get("p0_complete"):
        out.append("Finish full P0 row and adjacent deltas.")
    if module in {"PhysicsTexture", "FrictionSet", "DG losses", "EvidenceField aux", "Full fusion"}:
        if not missing_context.get("lodo_complete"):
            out.append("Check held-out RoadSaW/RSCD/RoadSC LODO.")
        if not missing_context.get("fair_single_dataset_complete"):
            out.append("Check same-split FAF vs ConvNeXt single-dataset comparisons.")
    if module not in {"PhysicsTexture", "FrictionSet", "DG losses", "EvidenceField aux", "Full fusion"}:
        if not missing_context.get("candidate_path_complete"):
            out.append("Run the matching P1/P2/P3 candidate row.")
    if not missing_context.get("final_method_complete"):
        out.append("Verify final lean method LODO and matched single-dataset evidence.")
    return out


def _risk_context(
    shortcut: dict[str, Any],
    wetness: dict[str, Any],
    interval: dict[str, Any],
    segmentation_transfer: dict[str, Any],
    segmentation_queue: dict[str, Any],
) -> dict[str, Any]:
    batch = segmentation_transfer.get("batch_report", {}) if isinstance(segmentation_transfer.get("batch_report"), dict) else {}
    return {
        "dataset_shortcut_verdict": shortcut.get("verdict"),
        "high_shortcut_rows": shortcut.get("num_high_shortcut"),
        "roadsaw_wetness_watchlist": wetness.get("num_watchlist"),
        "conditional_interval_watchlist": interval.get("num_watchlist_items"),
        "segmentation_transfer_config": segmentation_transfer.get("verdict"),
        "segmentation_pseudo_road_loss": batch.get("loss_evidence_attention_pseudo_road"),
        "segmentation_audit_wait": (segmentation_queue.get("after_queue") or {}).get("wait_pids")
        if isinstance(segmentation_queue.get("after_queue"), dict)
        else None,
    }


def _best_completed_p0_row(
    p0_rows_by_method: dict[str, dict[str, Any]],
    final_selection: dict[str, Any],
) -> dict[str, Any] | None:
    top = final_selection.get("provisional_top_completed") or []
    if top and isinstance(top[0], dict):
        method = str(top[0].get("method") or "")
        row = dict(p0_rows_by_method.get(method, {}))
        if row:
            return row
    completed = [
        row for row in p0_rows_by_method.values()
        if row.get("status") == "complete"
    ]
    if not completed:
        return None
    return max(completed, key=_selection_score)


def _absolute_context(
    current_method: str,
    p0_rows_by_method: dict[str, dict[str, Any]],
    best_completed_p0: dict[str, Any] | None,
) -> dict[str, Any]:
    current = p0_rows_by_method.get(current_method)
    if not current or current.get("status") != "complete" or not best_completed_p0:
        return {"summary": None, "best_method": None, "score_gap": None}
    best_method = str(best_completed_p0.get("method") or "")
    current_score = _selection_score(current)
    best_score = _selection_score(best_completed_p0)
    score_gap = current_score - best_score
    if current_method == best_method or abs(score_gap) < 1e-9:
        return {
            "summary": "current row is the best completed P0 safety/generalization row.",
            "best_method": best_method,
            "score_gap": score_gap,
        }

    gaps = []
    for key, label, percent, better_high in [
        ("risk_macro_f1", "risk F1", True, True),
        ("low_friction_recall", "low recall", True, True),
        ("worst_dataset_f1", "worst dataset F1", True, True),
        ("raw_interval_coverage", "raw coverage", True, True),
        ("calibrated_width", "calibrated width", False, False),
        ("dataset_id_balanced_accuracy", "dataset-ID", True, False),
    ]:
        cur = _num(current.get(key))
        best = _num(best_completed_p0.get(key))
        if cur is None or best is None:
            continue
        diff = cur - best
        if (better_high and diff < -0.005) or (not better_high and diff > 0.005):
            gaps.append(f"{label} {fmt_gap(diff, percent)}")
    gap_text = "; ".join(gaps[:4]) if gaps else f"selection score {score_gap:+.4f}"
    return {
        "summary": f"current row trails `{best_method}` ({gap_text}).",
        "best_method": best_method,
        "score_gap": score_gap,
    }


def _selection_score(row: dict[str, Any]) -> float:
    existing = _num(row.get("selection_score"))
    if existing is not None:
        return existing
    risk = _num(row.get("risk_macro_f1")) or 0.0
    low = _num(row.get("low_friction_recall")) or 0.0
    worst = _num(row.get("worst_dataset_f1")) or 0.0
    raw_cov = _num(row.get("raw_interval_coverage")) or 0.0
    cal_cov = _num(row.get("calibrated_coverage")) or 0.0
    width = _num(row.get("calibrated_width")) or 1.0
    shortcut = _num(row.get("dataset_id_balanced_accuracy")) or 1.0
    coverage_quality = max(0.0, 1.0 - abs(cal_cov - 0.90))
    width_quality = max(0.0, 1.0 - width)
    shortcut_quality = max(0.0, 1.0 - shortcut)
    return (
        0.25 * risk
        + 0.20 * low
        + 0.20 * worst
        + 0.15 * raw_cov
        + 0.10 * coverage_quality
        + 0.05 * width_quality
        + 0.05 * shortcut_quality
    )


def _verdict(rows: list[dict[str, Any]], missing_context: dict[str, bool]) -> str:
    if any(not complete for complete in missing_context.values()):
        return "pending_hard_evidence"
    if any(row["current_decision"] == "provisional_remove_or_merge" for row in rows):
        return "ready_with_pruned_modules"
    return "ready_to_freeze_modules"


def _recommended_route(rows: list[dict[str, Any]], missing_context: dict[str, bool]) -> list[str]:
    keep = [row["module"] for row in rows if row["current_decision"] in {"provisional_keep", "final_keep_candidate"}]
    remove = [row["module"] for row in rows if row["current_decision"] == "provisional_remove_or_merge"]
    configured = [row["module"] for row in rows if row["current_decision"] == "configured_pending_metrics"]
    route = []
    if keep:
        route.append("Current lean core to protect: " + ", ".join(keep) + ".")
    if remove:
        route.append("Keep out of the final route unless rescued by later evidence: " + ", ".join(remove) + ".")
    if configured:
        route.append("Configured but not retained yet: " + ", ".join(configured) + ".")
    if not missing_context.get("lodo_complete"):
        route.append("Do not freeze the final architecture before held-out RoadSaW LODO.")
    if not missing_context.get("fair_single_dataset_complete"):
        route.append("Do not claim superiority before matched ConvNeXt comparisons.")
    return route


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Module Retention Report",
        "",
        f"Generated at: {report['generated_at']}",
        f"Verdict: `{report['verdict']}`",
        "",
        report["hard_rule"],
        "",
        "## Current Risk Context",
        "",
    ]
    for key, value in report["risk_context"].items():
        lines.append(f"- `{key}`: `{value}`.")
    lines.extend(["", "## Module Decisions", ""])
    lines.append("| Module | Decision | Evidence summary | Next required evidence |")
    lines.append("|---|---|---|---|")
    for row in report["rows"]:
        lines.append(
            "| {module} | `{decision}` | {summary} | {next} |".format(
                module=row["module"],
                decision=row["current_decision"],
                summary=row["evidence_summary"],
                next="<br>".join(row["next_required_evidence"]) or "-",
            )
        )
    lines.extend(["", "## Rescue Or Confirmation Tests", ""])
    for row in report["rows"]:
        tests = row.get("rescue_or_confirmation_tests") or []
        if not tests:
            continue
        lines.append(f"### {row['module']}")
        for item in tests:
            lines.append(f"- {item}")
        lines.append("")
    lines.extend(["## Recommended Route Now", ""])
    for item in report["recommended_final_route_now"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "module",
        "current_method",
        "current_decision",
        "base_recommendation",
        "evidence_summary",
        "absolute_context_summary",
        "best_completed_p0_method",
        "selection_score_gap_to_best",
        "delta_friction_f1",
        "delta_risk_f1",
        "delta_low_recall",
        "delta_calibrated_coverage",
        "delta_calibrated_width",
        "delta_worst_dataset_f1",
        "delta_dataset_id_bal_acc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _is_complete(requirement: dict[str, Any] | None) -> bool:
    return bool(requirement) and requirement.get("status") == "complete"


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lt(value: Any, threshold: float) -> bool:
    num = _num(value)
    return num is not None and num < threshold


def fmt_gap(value: float, percent: bool) -> str:
    if percent:
        return f"{100.0 * value:+.2f}"
    return f"{value:+.4f}"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    main()
