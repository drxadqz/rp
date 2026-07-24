from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_OUT = Path("reports/paper_protocol_summary")


ABLATION_ROWS = [
    ("Global-only", "v0_global_only"),
    ("+ PhysicsTexture", "v1_physics_texture"),
    ("+ FrictionSet", "v2_friction_set"),
    ("+ DG losses", "v3_dg_losses"),
    ("+ EvidenceField aux", "v4_evidence_aux"),
    ("Full model", "v5_full_faf"),
    ("Full + Fourier candidate", "v6_full_faf_fourier"),
    ("Full + Fourier + DANN candidate", "v7_full_faf_fourier_dann"),
    ("Full + Fourier + road prior candidate", "v8_full_faf_fourier_roadprior"),
    ("Full + wet-state hard-sampling candidate", "v9_full_faf_roadsaw_hard_sampling"),
    ("Full + consistency candidate", "v10_full_faf_consistency"),
    ("Full + domain adapter candidate", "v11_full_faf_domain_adapter"),
    ("Full + ROI interval-safety candidate", "v12_full_faf_roi_interval_safety"),
    ("Lean Physics+Evidence candidate", "v13_lean_physics_evidence"),
    ("Lean road-ROI safety candidate", "v14_lean_road_roi_safety"),
]

CORE_ABLATION_ROWS = ABLATION_ROWS[:6]

LODO_ROWS = [
    ("held-out RoadSaW", "lodo_roadsaw_full_faf"),
    ("held-out RSCD", "lodo_rscd_full_faf"),
    ("held-out RoadSC", "lodo_roadsc_full_faf"),
]

SINGLE_ROWS = [
    ("RoadSaW only", "single_roadsaw_full_faf"),
    ("RSCD only", "single_rscd_full_faf"),
    ("RoadSC only", "single_roadsc_full_faf"),
]

FAIR_BASELINE_ROWS = [
    ("RoadSaW global ConvNeXt", "baseline_single_roadsaw_global_convnext"),
    ("RSCD global ConvNeXt", "baseline_single_rscd_global_convnext"),
    ("RoadSC global ConvNeXt", "baseline_single_roadsc_global_convnext"),
]

FINAL_LODO_ROWS = [
    ("final held-out RoadSaW", "final_lodo_roadsaw_lean_road_roi_safety"),
    ("final held-out RSCD", "final_lodo_rscd_lean_road_roi_safety"),
    ("final held-out RoadSC", "final_lodo_roadsc_lean_road_roi_safety"),
]

FINAL_SINGLE_ROWS = [
    ("RoadSaW final lean safety", "final_single_roadsaw_lean_road_roi_safety"),
    ("RSCD final lean safety", "final_single_rscd_lean_road_roi_safety"),
    ("RoadSC final lean safety", "final_single_roadsc_lean_road_roi_safety"),
]

ALL_RESULT_ROWS = ABLATION_ROWS + LODO_ROWS + SINGLE_ROWS + FAIR_BASELINE_ROWS + FINAL_LODO_ROWS + FINAL_SINGLE_ROWS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ablation = build_table(args.root, ABLATION_ROWS)
    core_ablation = build_core_ablation_table(ablation)
    lodo = build_table(args.root, LODO_ROWS)
    single = build_table(args.root, SINGLE_ROWS)
    fair_baselines = build_table(args.root, FAIR_BASELINE_ROWS)
    fair_deltas = build_fair_deltas(single, fair_baselines)
    final_lodo = build_table(args.root, FINAL_LODO_ROWS)
    final_single = build_table(args.root, FINAL_SINGLE_ROWS)
    final_fair_deltas = build_fair_deltas(final_single, fair_baselines)
    rule_baselines = build_rule_baseline_table(args.out_dir)
    recommendations = build_recommendations(ablation)
    dataset_breakdown = build_dataset_breakdown(args.root, ALL_RESULT_ROWS)
    class_breakdown = build_class_breakdown(args.root, ALL_RESULT_ROWS)

    write_csv(args.out_dir / "ablation_table.csv", ablation)
    write_csv(args.out_dir / "core_ablation_table.csv", core_ablation)
    (args.out_dir / "core_ablation_table.md").write_text(
        render_core_ablation_markdown(core_ablation),
        encoding="utf-8",
    )
    write_csv(args.out_dir / "lodo_table.csv", lodo)
    write_csv(args.out_dir / "single_dataset_table.csv", single)
    write_csv(args.out_dir / "fair_baseline_table.csv", fair_baselines)
    write_csv(args.out_dir / "fair_single_dataset_deltas.csv", fair_deltas)
    write_csv(args.out_dir / "final_lodo_table.csv", final_lodo)
    write_csv(args.out_dir / "final_single_dataset_table.csv", final_single)
    write_csv(args.out_dir / "final_fair_single_dataset_deltas.csv", final_fair_deltas)
    write_csv(args.out_dir / "rule_baseline_table.csv", rule_baselines)
    write_csv(args.out_dir / "module_recommendations.csv", recommendations)
    write_csv(args.out_dir / "dataset_breakdown_table.csv", dataset_breakdown)
    write_csv(args.out_dir / "class_f1_breakdown.csv", class_breakdown)
    payload = {
        "root": str(args.root),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ablation": ablation,
        "core_ablation": core_ablation,
        "lodo": lodo,
        "single_dataset": single,
        "fair_baselines": fair_baselines,
        "fair_single_dataset_deltas": fair_deltas,
        "final_lodo": final_lodo,
        "final_single_dataset": final_single,
        "final_fair_single_dataset_deltas": final_fair_deltas,
        "rule_baselines": rule_baselines,
        "module_recommendations": recommendations,
        "dataset_breakdown": dataset_breakdown,
        "class_f1_breakdown": class_breakdown,
    }
    (args.out_dir / "paper_protocol_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md = render_markdown(payload)
    (args.out_dir / "paper_protocol_summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote: {args.out_dir}")


def build_table(root: Path, rows: list[tuple[str, str]]) -> list[dict[str, Any]]:
    table = []
    for label, dirname in rows:
        out_dir = root / dirname
        metrics = collect_metrics(out_dir)
        table.append({"method": label, "output_dir": str(out_dir), **metrics})
    return table


def build_core_ablation_table(ablation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {label for label, _ in CORE_ABLATION_ROWS}
    rows = []
    for row in ablation:
        if row.get("method") not in wanted:
            continue
        rows.append(
            {
                "method": row.get("method"),
                "status": row.get("status"),
                "friction_f1": row.get("friction_macro_f1"),
                "risk_f1": row.get("risk_macro_f1"),
                "low_friction_recall": row.get("low_friction_recall"),
                "calibrated_coverage": row.get("calibrated_coverage"),
                "calibrated_width": row.get("calibrated_width"),
                "worst_dataset_f1": row.get("worst_dataset_f1"),
                "dataset_id_balanced_accuracy": row.get("dataset_id_balanced_accuracy"),
                "audit_verdict": row.get("audit_verdict"),
            }
        )
    return rows


def collect_metrics(out_dir: Path) -> dict[str, Any]:
    result_state = inspect_result_state(out_dir)
    base = empty_metric_row(result_state["status"])
    base.update(
        {
            "epoch": result_state.get("epoch"),
            "epochs": result_state.get("epochs"),
            "stale_reason": result_state.get("stale_reason"),
        }
    )
    if result_state["ignore_metrics"]:
        return base

    detailed = load_json(out_dir / "detailed_test.json")
    calib = load_json(out_dir / "interval_calibration_90.json")
    bootstrap = load_json(out_dir / "bootstrap_metrics.json")
    diag = load_json(out_dir / "dataset_id_diagnostic.json")
    audit = load_json(out_dir / "topvenue_result_audit.json")
    if detailed is None:
        return base

    friction = detailed.get("tasks", {}).get("friction", {})
    risk = detailed.get("tasks", {}).get("risk", {})
    mu = detailed.get("mu_interval", {})
    low = _low_friction_info(detailed)
    test_calib = (calib or {}).get("test_split", {})
    dataset_conditional = (calib or {}).get("dataset_conditional_test", {}).get("_pooled", {})
    dataset_core_conditional = (calib or {}).get("dataset_core_conditional_test", {}).get("_pooled", {})
    risk_conditional = (calib or {}).get("risk_conditional_test", {}).get("_pooled", {})
    hierarchical_conditional = (calib or {}).get("hierarchical_conditional_test", {}).get("pooled", {})
    worst_risk = dig(risk, ["by_dataset", "_worst_macro_f1", "value"])
    worst_friction = dig(friction, ["by_dataset", "_worst_macro_f1", "value"])
    missing_completion = missing_required_artifacts(out_dir)
    if not missing_completion:
        status = "complete"
    elif calib is not None and bootstrap is None:
        status = "partial_ci_missing"
    else:
        status = "partial"
    low_applicable = low.get("applicable")
    bootstrap_fields = bootstrap_metric_fields(bootstrap)
    if low_applicable is False:
        bootstrap_fields["low_friction_recall_ci_low"] = None
        bootstrap_fields["low_friction_recall_ci_high"] = None
        bootstrap_fields["low_friction_recall_bootstrap_std"] = None
    return {
        "epoch": result_state.get("epoch"),
        "epochs": result_state.get("epochs"),
        "stale_reason": result_state.get("stale_reason"),
        "missing_completion_artifacts": ";".join(missing_completion),
        "status": status,
        "num_samples": detailed.get("num_samples_seen"),
        "friction_macro_f1": friction.get("macro_f1"),
        "friction_accuracy": friction.get("accuracy"),
        "risk_macro_f1": risk.get("macro_f1"),
        "risk_accuracy": risk.get("accuracy"),
        "low_friction_recall": None if low_applicable is False else low.get("recall"),
        "low_friction_precision": None if low_applicable is False else low.get("precision"),
        "low_friction_f1": None if low_applicable is False else low.get("f1"),
        "low_friction_recall_applicable": low_applicable,
        "low_friction_positive_count": low.get("num_positive"),
        "raw_interval_coverage": mu.get("coverage"),
        "raw_interval_width": mu.get("width_mean"),
        "calibrated_coverage": test_calib.get("calibrated_coverage"),
        "calibrated_width": test_calib.get("calibrated_width"),
        "dataset_conditional_calibrated_coverage": dataset_conditional.get("calibrated_coverage"),
        "dataset_conditional_calibrated_width": dataset_conditional.get("calibrated_width"),
        "dataset_core_conditional_calibrated_coverage": dataset_core_conditional.get("calibrated_coverage"),
        "dataset_core_conditional_calibrated_width": dataset_core_conditional.get("calibrated_width"),
        "risk_conditional_calibrated_coverage": risk_conditional.get("calibrated_coverage"),
        "risk_conditional_calibrated_width": risk_conditional.get("calibrated_width"),
        "hierarchical_conditional_calibrated_coverage": hierarchical_conditional.get("calibrated_coverage"),
        "hierarchical_conditional_calibrated_width": hierarchical_conditional.get("calibrated_width"),
        "hierarchical_conditional_mean_radius": hierarchical_conditional.get("mean_radius"),
        "worst_dataset_risk_f1": worst_risk,
        "worst_dataset_friction_f1": worst_friction,
        "worst_dataset_f1": min_defined(worst_risk, worst_friction),
        "dataset_id_balanced_accuracy": (diag or {}).get("overall_dataset_id_balanced_accuracy"),
        "audit_verdict": (audit or {}).get("verdict"),
        **bootstrap_fields,
    }


def empty_metric_row(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "friction_macro_f1": None,
        "risk_macro_f1": None,
        "low_friction_recall": None,
        "raw_interval_coverage": None,
        "raw_interval_width": None,
        "calibrated_coverage": None,
        "calibrated_width": None,
        "hierarchical_conditional_calibrated_coverage": None,
        "hierarchical_conditional_calibrated_width": None,
        "worst_dataset_risk_f1": None,
        "worst_dataset_friction_f1": None,
        "worst_dataset_f1": None,
    }


def inspect_result_state(out_dir: Path) -> dict[str, Any]:
    state = load_json(out_dir / "training_state.json")
    epoch = state.get("epoch") if isinstance(state, dict) else None
    epochs = state.get("epochs") if isinstance(state, dict) else None
    has_training_artifact = any(
        (out_dir / name).exists()
        for name in ["training_state.json", "best.pt", "last.pt"]
    )
    detailed = out_dir / "detailed_test.json"
    if not detailed.exists():
        return {
            "status": "running" if has_training_artifact else "missing",
            "ignore_metrics": True,
            "epoch": epoch,
            "epochs": epochs,
            "stale_reason": None,
        }

    stale_reason = stale_result_reason(out_dir)
    if stale_reason:
        return {
            "status": "stale",
            "ignore_metrics": True,
            "epoch": epoch,
            "epochs": epochs,
            "stale_reason": stale_reason,
        }
    return {
        "status": "has_fresh_detailed",
        "ignore_metrics": False,
        "epoch": epoch,
        "epochs": epochs,
        "stale_reason": None,
    }


def missing_required_artifacts(out_dir: Path) -> list[str]:
    required = [
        "best.pt",
        "evaluate_test.json",
        "detailed_test.json",
        "interval_calibration_90.json",
        "bootstrap_metrics.json",
        "topvenue_result_audit.json",
    ]
    missing = [name for name in required if not (out_dir / name).exists()]
    if not (
        out_dir.name.startswith("single_")
        or out_dir.name.startswith("baseline_single_")
        or out_dir.name.startswith("final_single_")
    ):
        if not (out_dir / "dataset_id_diagnostic.json").exists():
            missing.append("dataset_id_diagnostic.json")
    config = load_json(out_dir / "config.json")
    if isinstance(config, dict) and config.get("model", {}).get("use_evidence_field"):
        if not (out_dir / "evidence_maps").exists():
            missing.append("evidence_maps")
        for name in ["evidence_field_audit.json", "evidence_field_audit.md"]:
            if not (out_dir / name).exists():
                missing.append(name)
    return missing


def stale_result_reason(out_dir: Path) -> str | None:
    detailed = out_dir / "detailed_test.json"
    if not detailed.exists():
        return None
    detailed_mtime = detailed.stat().st_mtime
    best = out_dir / "best.pt"
    if best.exists() and best.stat().st_mtime > detailed_mtime + 1:
        return "best_checkpoint_newer_than_detailed_test"
    last = out_dir / "last.pt"
    state = load_json(out_dir / "training_state.json")
    if last.exists() and isinstance(state, dict):
        epoch = state.get("epoch")
        epochs = state.get("epochs")
        if epoch is None or epochs is None or int(epoch) < int(epochs):
            return "training_or_resume_checkpoint_present_after_detailed_test"
    return None


def bootstrap_metric_fields(bootstrap: Any) -> dict[str, Any]:
    if not isinstance(bootstrap, dict):
        return {}
    out: dict[str, Any] = {}
    mapping = {
        "friction_macro_f1": ("classification", "friction", "macro_f1"),
        "friction_worst_dataset_macro_f1": ("classification", "friction", "worst_dataset_macro_f1"),
        "risk_macro_f1": ("classification", "risk", "macro_f1"),
        "risk_worst_dataset_macro_f1": ("classification", "risk", "worst_dataset_macro_f1"),
        "low_friction_recall": ("low_friction_detection", "recall"),
        "raw_interval_coverage": ("mu_interval", "raw_coverage"),
        "raw_interval_width": ("mu_interval", "raw_width"),
        "calibrated_coverage": ("mu_interval", "calibrated_coverage"),
        "calibrated_width": ("mu_interval", "calibrated_width"),
        "hierarchical_conditional_calibrated_coverage": ("mu_interval", "hierarchical_calibrated_coverage"),
        "hierarchical_conditional_calibrated_width": ("mu_interval", "hierarchical_calibrated_width"),
        "hierarchical_worst_dataset_core_coverage": ("mu_interval", "hierarchical_worst_dataset_core_coverage"),
    }
    for name, path in mapping.items():
        item = dig(bootstrap, list(path))
        if not isinstance(item, dict):
            continue
        out[f"{name}_ci_low"] = item.get("ci_low")
        out[f"{name}_ci_high"] = item.get("ci_high")
        out[f"{name}_bootstrap_std"] = item.get("std")
    return out


def build_recommendations(ablation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete = [row for row in ablation if row.get("status") == "complete"]
    if len(complete) < 2:
        return [
            {
                "module": "all",
                "decision": "pending",
                "evidence": "Ablation suite is incomplete; no keep/remove decision is statistically defensible yet.",
            }
        ]
    by_method = {row["method"]: row for row in complete}
    steps = [
        ("PhysicsTexture", "Global-only", "+ PhysicsTexture"),
        ("FrictionSet", "+ PhysicsTexture", "+ FrictionSet"),
        ("DG losses", "+ FrictionSet", "+ DG losses"),
        ("EvidenceField", "+ DG losses", "+ EvidenceField aux"),
        ("Final fusion", "+ EvidenceField aux", "Full model"),
        ("Fourier style jitter", "Full model", "Full + Fourier candidate"),
        ("Road-likelihood attention prior", "Full + Fourier candidate", "Full + Fourier + road prior candidate"),
        ("Wet-state hard sampling", "Full + Fourier + road prior candidate", "Full + wet-state hard-sampling candidate"),
        ("Weak-view consistency", "Full + wet-state hard-sampling candidate", "Full + consistency candidate"),
        ("Domain-specific adapter", "Full + consistency candidate", "Full + domain adapter candidate"),
        ("ROI interval safety", "Full + consistency candidate", "Full + ROI interval-safety candidate"),
        ("Lean simplification", "Full model", "Lean Physics+Evidence candidate"),
        ("Lean road-ROI safety", "Lean Physics+Evidence candidate", "Lean road-ROI safety candidate"),
    ]
    recs = []
    for module, prev_name, cur_name in steps:
        prev = by_method.get(prev_name)
        cur = by_method.get(cur_name)
        if not prev or not cur:
            recs.append({"module": module, "decision": "pending", "evidence": "Required adjacent rows missing."})
            continue
        delta_risk = safe_delta(cur, prev, "risk_macro_f1")
        delta_low = safe_delta(cur, prev, "low_friction_recall")
        delta_worst = safe_delta(cur, prev, "worst_dataset_risk_f1")
        delta_friction = safe_delta(cur, prev, "friction_macro_f1")
        delta_raw_cov = safe_delta(cur, prev, "raw_interval_coverage")
        delta_calib_cov = safe_delta(cur, prev, "calibrated_coverage")
        delta_calib_width = safe_delta(cur, prev, "calibrated_width")
        delta_dataset_id = safe_delta(cur, prev, "dataset_id_balanced_accuracy")
        primary = [delta_risk, delta_low, delta_worst, delta_friction]
        improved = sum(1 for value in primary if value is not None and value >= 0.005)
        harmed = sum(1 for value in primary if value is not None and value <= -0.02)
        interval_help = (delta_raw_cov is not None and delta_raw_cov >= 0.03) or (
            delta_calib_width is not None
            and delta_calib_width <= -0.03
            and (delta_calib_cov is None or delta_calib_cov >= -0.01)
        )
        shortcut_help = delta_dataset_id is not None and delta_dataset_id <= -0.02
        width_hurt = delta_calib_width is not None and delta_calib_width >= 0.05 and not interval_help
        if harmed or width_hurt:
            decision = "remove_or_rework"
        elif improved >= 2 or interval_help or shortcut_help:
            decision = "keep"
        else:
            decision = "merge_or_simplify"
        recs.append(
            {
                "module": module,
                "decision": decision,
                "delta_risk_macro_f1": delta_risk,
                "delta_low_friction_recall": delta_low,
                "delta_worst_dataset_risk_f1": delta_worst,
                "delta_friction_macro_f1": delta_friction,
                "delta_raw_interval_coverage": delta_raw_cov,
                "delta_calibrated_coverage": delta_calib_cov,
                "delta_calibrated_width": delta_calib_width,
                "delta_dataset_id_balanced_accuracy": delta_dataset_id,
            }
        )
    return recs


def build_fair_deltas(
    single: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    dataset_names = ["RSCD", "RoadSaW", "RoadSC"]
    rows = []
    for dataset in dataset_names:
        faf = next((row for row in single if row["method"].lower().startswith(dataset.lower())), None)
        base = next((row for row in baselines if row["method"].lower().startswith(dataset.lower())), None)
        if not faf or not base or faf.get("status") != "complete" or base.get("status") != "complete":
            rows.append({"dataset": dataset, "status": "pending"})
            continue
        rows.append(
            {
                "dataset": dataset,
                "status": "complete",
                "delta_friction_macro_f1": safe_delta(faf, base, "friction_macro_f1"),
                "delta_risk_macro_f1": safe_delta(faf, base, "risk_macro_f1"),
                "delta_low_friction_recall": safe_delta(faf, base, "low_friction_recall"),
                "delta_calibrated_coverage": safe_delta(faf, base, "calibrated_coverage"),
                "delta_worst_dataset_risk_f1": safe_delta(faf, base, "worst_dataset_risk_f1"),
            }
        )
    return rows


def build_rule_baseline_table(summary_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in ["rscd", "roadsaw", "roadsc"]:
        path = summary_dir / f"rule_baseline_{dataset}_test.json"
        payload = load_json(path)
        if not isinstance(payload, dict):
            rows.append({"dataset": dataset, "status": "missing", "path": str(path)})
            continue
        eval_metrics = payload.get("eval", {})
        rows.append(
            {
                "dataset": dataset,
                "status": "complete",
                "path": str(path),
                "num_samples": eval_metrics.get("num_samples"),
                "coverage": eval_metrics.get("coverage"),
                "avg_width": eval_metrics.get("avg_width"),
                "median_width": eval_metrics.get("median_width"),
                "mid_mae": eval_metrics.get("mid_mae"),
                "note": payload.get("note"),
            }
        )
    return rows


def build_dataset_breakdown(root: Path, rows: list[tuple[str, str]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for method, dirname in rows:
        out_dir = root / dirname
        result_state = inspect_result_state(out_dir)
        if result_state["ignore_metrics"]:
            table.append(
                {
                    "method": method,
                    "run": dirname,
                    "status": result_state["status"],
                    "epoch": result_state.get("epoch"),
                    "epochs": result_state.get("epochs"),
                    "stale_reason": result_state.get("stale_reason"),
                }
            )
            continue
        detailed = load_json(out_dir / "detailed_test.json")
        calib = load_json(out_dir / "interval_calibration_90.json")
        if detailed is None:
            table.append({"method": method, "run": dirname, "status": "missing"})
            continue
        datasets = _datasets_in_detail(detailed)
        if not datasets:
            table.append({"method": method, "run": dirname, "status": "partial", "dataset": "unknown"})
            continue
        for dataset in datasets:
            row = {
                "method": method,
                "run": dirname,
                "status": "complete" if calib is not None else "partial",
                "dataset": dataset,
            }
            for task in ["friction", "risk", "wetness", "snow", "material", "unevenness"]:
                task_summary = detailed.get("tasks", {}).get(task, {})
                dataset_summary = task_summary.get("by_dataset", {}).get(dataset, {})
                if dataset_summary:
                    row[f"{task}_num_samples"] = dataset_summary.get("num_samples")
                    row[f"{task}_macro_f1"] = dataset_summary.get("macro_f1")
                    row[f"{task}_accuracy"] = dataset_summary.get("accuracy")
                    row[f"{task}_balanced_accuracy"] = dataset_summary.get("balanced_accuracy")
            risk_summary = detailed.get("tasks", {}).get("risk", {}).get("by_dataset", {}).get(dataset, {})
            row["low_friction_recall"] = _risk_low_recall(risk_summary)
            mu_summary = detailed.get("mu_interval", {}).get("by_dataset", {}).get(dataset, {})
            if mu_summary:
                row["raw_interval_coverage"] = mu_summary.get("coverage")
                row["raw_interval_width"] = mu_summary.get("width_mean")
                row["mu_mean_mae_to_interval_mid"] = mu_summary.get("mean_mae_to_interval_mid")
            calib_summary = (calib or {}).get("dataset_conditional_test", {}).get(dataset, {})
            if calib_summary:
                row["calibrated_coverage"] = calib_summary.get("calibrated_coverage")
                row["calibrated_width"] = calib_summary.get("calibrated_width")
                row["used_group_radius"] = calib_summary.get("used_group_radius")
            table.append(row)
    return table


def build_class_breakdown(root: Path, rows: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method, dirname in rows:
        if inspect_result_state(root / dirname)["ignore_metrics"]:
            continue
        detailed = load_json(root / dirname / "detailed_test.json")
        if detailed is None:
            continue
        for task in ["friction", "risk", "wetness", "snow", "material", "unevenness"]:
            task_summary = detailed.get("tasks", {}).get(task)
            if not task_summary:
                continue
            out.extend(_class_rows(method, dirname, task, "overall", task_summary))
            by_dataset = task_summary.get("by_dataset", {})
            for dataset, dataset_summary in by_dataset.items():
                if dataset.startswith("_"):
                    continue
                out.extend(_class_rows(method, dirname, task, dataset, dataset_summary))
    return out


def _datasets_in_detail(detailed: dict[str, Any]) -> list[str]:
    datasets: set[str] = set()
    for task_summary in detailed.get("tasks", {}).values():
        for name in task_summary.get("by_dataset", {}):
            if not str(name).startswith("_"):
                datasets.add(str(name))
    for name in detailed.get("mu_interval", {}).get("by_dataset", {}):
        if not str(name).startswith("_"):
            datasets.add(str(name))
    return sorted(datasets)


def _risk_low_recall(risk_summary: dict[str, Any]) -> float | None:
    labels = risk_summary.get("confusion_matrix_labels")
    matrix = risk_summary.get("confusion_matrix")
    if not labels or not matrix:
        return None
    positives = [idx for idx, label in enumerate(labels) if label in {"high", "very_high"}]
    if not positives:
        return None
    true_positive_support = sum(sum(matrix[idx]) for idx in positives)
    if true_positive_support <= 0:
        return None
    recalled = sum(sum(matrix[idx][pred_idx] for pred_idx in positives) for idx in positives)
    return float(recalled) / float(true_positive_support)


def _low_friction_info(detailed: dict[str, Any]) -> dict[str, Any]:
    info = dict(detailed.get("low_friction_detection") or {})
    if info.get("applicable") is not None and info.get("num_positive") is not None:
        return info
    risk_summary = detailed.get("tasks", {}).get("risk", {})
    labels = risk_summary.get("confusion_matrix_labels")
    matrix = risk_summary.get("confusion_matrix")
    if not labels or not matrix:
        return info
    positives = [idx for idx, label in enumerate(labels) if label in {"high", "very_high"}]
    if not positives:
        return info
    num_positive = int(sum(sum(matrix[idx]) for idx in positives))
    info["num_positive"] = num_positive
    info["applicable"] = num_positive > 0
    if num_positive <= 0:
        return info
    info.setdefault("recall", _risk_low_recall(risk_summary))
    return info


def _class_rows(
    method: str,
    run: str,
    task: str,
    scope: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = summary.get("confusion_matrix_labels", [])
    matrix = summary.get("confusion_matrix", [])
    per_class = summary.get("per_class_f1", {})
    rows = []
    for idx, label in enumerate(labels):
        support = None
        if idx < len(matrix):
            support = int(sum(matrix[idx]))
        rows.append(
            {
                "method": method,
                "run": run,
                "task": task,
                "scope": scope,
                "class": label,
                "support": support,
                "f1": per_class.get(label),
            }
        )
    return rows


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Paper Protocol Summary",
        "",
        f"Root: `{payload['root']}`",
        f"Generated at: {payload.get('generated_at', '-')}",
        "",
        "Rows marked `running` or `stale` are intentionally excluded from metric reporting to avoid mixing old evaluations with the active training run.",
        "",
    ]
    lines.extend(["## Core P0 Ablation Table", ""])
    lines.append(
        "| Method | Status | friction F1 | risk F1 | low-friction recall | calibrated coverage | calibrated width | worst dataset F1 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in payload.get("core_ablation", []):
        lines.append(
            "| {method} | {status} | {friction} | {risk} | {low} | {cov} | {width} | {worst} |".format(
                method=row.get("method"),
                status=row.get("status"),
                friction=fmt(row.get("friction_f1")),
                risk=fmt(row.get("risk_f1")),
                low=fmt(row.get("low_friction_recall")),
                cov=fmt(row.get("calibrated_coverage")),
                width=fmt_abs(row.get("calibrated_width")),
                worst=fmt(row.get("worst_dataset_f1")),
            )
        )
    lines.append("")
    for title, key in [
        ("Ablation Table", "ablation"),
        ("Leave-One-Dataset-Out Table", "lodo"),
        ("Single-Dataset Fair Comparison", "single_dataset"),
        ("Fair Visual Baselines", "fair_baselines"),
        ("Final-Method LODO Table", "final_lodo"),
        ("Final-Method Single-Dataset Table", "final_single_dataset"),
    ]:
        rows = payload[key]
        lines.extend([f"## {title}", ""])
        lines.append(
            "| Method | Status | friction F1 | risk F1 | low recall | raw cov | raw width | calib cov | calib width | worst F1 | dataset-ID bal acc |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                "| {method} | {status} | {friction} | {risk} | {low} | {raw_cov} | {raw_width} | {cov} | {calib_width} | {worst} | {dataset_id} |".format(
                    method=row["method"],
                    status=row.get("status"),
                    friction=fmt(row.get("friction_macro_f1")),
                    risk=fmt(row.get("risk_macro_f1")),
                    low=fmt(row.get("low_friction_recall")),
                    raw_cov=fmt(row.get("raw_interval_coverage")),
                    raw_width=fmt_abs(row.get("raw_interval_width")),
                    cov=fmt(row.get("calibrated_coverage")),
                    calib_width=fmt_abs(row.get("calibrated_width")),
                    worst=fmt(row.get("worst_dataset_f1")),
                    dataset_id=fmt(row.get("dataset_id_balanced_accuracy")),
                )
            )
    lines.append("")
    lines.extend(["## Conditional Calibration Overview", ""])
    lines.append(
        "Pooled calibration uses one conformal radius. Conditional calibration uses group-specific radii when the validation split has enough samples, otherwise it falls back to the pooled radius."
    )
    lines.append("")
    lines.append(
        "| Method | Status | pooled cov/width | dataset cov/width | dataset::core cov/width | risk cov/width | hierarchy cov/width |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in payload.get("ablation", []):
        lines.append(
            "| {method} | {status} | {pooled} | {dataset} | {core} | {risk} | {hierarchy} |".format(
                method=row["method"],
                status=row.get("status"),
                pooled=fmt_cov_width(row.get("calibrated_coverage"), row.get("calibrated_width")),
                dataset=fmt_cov_width(
                    row.get("dataset_conditional_calibrated_coverage"),
                    row.get("dataset_conditional_calibrated_width"),
                ),
                core=fmt_cov_width(
                    row.get("dataset_core_conditional_calibrated_coverage"),
                    row.get("dataset_core_conditional_calibrated_width"),
                ),
                risk=fmt_cov_width(
                    row.get("risk_conditional_calibrated_coverage"),
                    row.get("risk_conditional_calibrated_width"),
                ),
                hierarchy=fmt_cov_width(
                    row.get("hierarchical_conditional_calibrated_coverage"),
                    row.get("hierarchical_conditional_calibrated_width"),
                ),
            )
        )
    lines.append("")

    lines.extend(["## Module Recommendations", ""])
    lines.append("| Module | Decision | d risk F1 | d low recall | d worst risk F1 | d raw cov | d calib width | d dataset-ID bal acc |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["module_recommendations"]:
        lines.append(
            "| {module} | {decision} | {d1} | {d2} | {d3} | {d4} | {d5} | {d6} |".format(
                module=row["module"],
                decision=row["decision"],
                d1=fmt(row.get("delta_risk_macro_f1")),
                d2=fmt(row.get("delta_low_friction_recall")),
                d3=fmt(row.get("delta_worst_dataset_risk_f1")),
                d4=fmt_delta(row.get("delta_raw_interval_coverage")),
                d5=fmt_abs_delta(row.get("delta_calibrated_width")),
                d6=fmt_delta(row.get("delta_dataset_id_balanced_accuracy")),
            )
        )
    lines.append("")
    lines.extend(["## Single-Dataset FAF vs Global ConvNeXt", ""])
    lines.append("| Dataset | Status | delta friction F1 | delta risk F1 | delta low recall | delta calibrated coverage |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in payload["fair_single_dataset_deltas"]:
        lines.append(
            "| {dataset} | {status} | {df} | {dr} | {dl} | {dc} |".format(
                dataset=row["dataset"],
                status=row["status"],
                df=fmt_delta(row.get("delta_friction_macro_f1")),
                dr=fmt_delta(row.get("delta_risk_macro_f1")),
                dl=fmt_delta(row.get("delta_low_friction_recall")),
                dc=fmt_delta(row.get("delta_calibrated_coverage")),
            )
        )
    lines.append("")
    lines.extend(["## Final Lean Safety vs Global ConvNeXt", ""])
    lines.append("| Dataset | Status | delta friction F1 | delta risk F1 | delta low recall | delta calibrated coverage |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in payload["final_fair_single_dataset_deltas"]:
        lines.append(
            "| {dataset} | {status} | {df} | {dr} | {dl} | {dc} |".format(
                dataset=row["dataset"],
                status=row["status"],
                df=fmt_delta(row.get("delta_friction_macro_f1")),
                dr=fmt_delta(row.get("delta_risk_macro_f1")),
                dl=fmt_delta(row.get("delta_low_friction_recall")),
                dc=fmt_delta(row.get("delta_calibrated_coverage")),
            )
        )
    lines.append("")
    lines.extend(["## Non-Visual Rule Interval Baseline", ""])
    lines.append(
        "This baseline fits `class_label -> median weak friction interval` on train manifests and evaluates on held-out test manifests. It is a label-prior sanity check, not a fair visual model."
    )
    lines.append("")
    lines.append("| Dataset | Status | samples | coverage | avg width | median width | mid MAE |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in payload.get("rule_baselines", []):
        lines.append(
            "| {dataset} | {status} | {n} | {coverage} | {avg_width} | {median_width} | {mae} |".format(
                dataset=row.get("dataset", "-"),
                status=row.get("status", "-"),
                n=row.get("num_samples") if row.get("num_samples") is not None else "-",
                coverage=fmt(row.get("coverage")),
                avg_width=fmt_abs(row.get("avg_width")),
                median_width=fmt_abs(row.get("median_width")),
                mae=fmt_abs(row.get("mid_mae")),
            )
        )
    lines.append("")
    completed_dataset_rows = [
        row for row in payload.get("dataset_breakdown", [])
        if row.get("status") in {"complete", "partial"} and row.get("dataset")
    ][:36]
    lines.extend(["## Dataset Breakdown Preview", ""])
    if not completed_dataset_rows:
        lines.append("- No completed dataset-level rows yet.")
    else:
        lines.append("| Method | Dataset | friction F1 | risk F1 | low recall | raw cov | calib cov |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in completed_dataset_rows:
            lines.append(
                "| {method} | {dataset} | {friction} | {risk} | {low} | {raw_cov} | {calib_cov} |".format(
                    method=row["method"],
                    dataset=row["dataset"],
                    friction=fmt(row.get("friction_macro_f1")),
                    risk=fmt(row.get("risk_macro_f1")),
                    low=fmt(row.get("low_friction_recall")),
                    raw_cov=fmt(row.get("raw_interval_coverage")),
                    calib_cov=fmt(row.get("calibrated_coverage")),
                )
            )
    lines.append("")
    lines.append(
        "Full dataset/class breakdowns are written to `dataset_breakdown_table.csv` and `class_f1_breakdown.csv`."
    )
    lines.append("")
    return "\n".join(lines)


def render_core_ablation_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Core P0 Ablation Table",
        "",
        "| Method | Status | friction F1 | risk F1 | low-friction recall | calibrated coverage | calibrated width | worst dataset F1 | dataset-ID bal acc | audit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {status} | {friction} | {risk} | {low} | {cov} | {width} | {worst} | {dataset_id} | {audit} |".format(
                method=row.get("method"),
                status=row.get("status"),
                friction=fmt(row.get("friction_f1")),
                risk=fmt(row.get("risk_f1")),
                low=fmt(row.get("low_friction_recall")),
                cov=fmt(row.get("calibrated_coverage")),
                width=fmt_abs(row.get("calibrated_width")),
                worst=fmt(row.get("worst_dataset_f1")),
                dataset_id=fmt(row.get("dataset_id_balanced_accuracy")),
                audit=row.get("audit_verdict") or "-",
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def dig(items: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = items
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def safe_delta(cur: dict[str, Any], prev: dict[str, Any], key: str) -> float | None:
    if cur.get(key) is None or prev.get(key) is None:
        return None
    return float(cur[key]) - float(prev[key])


def min_defined(*values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return min(present)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}"


def fmt_delta(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):+.2f}"


def fmt_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def fmt_abs_delta(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.4f}"


def fmt_cov_width(coverage: Any, width: Any) -> str:
    if coverage is None or width is None:
        return "-"
    return f"{fmt(coverage)} / {fmt_abs(width)}"


if __name__ == "__main__":
    main()
