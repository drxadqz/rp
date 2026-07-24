from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_protocol_progress import ROWS, inspect_run
from extract_training_history_from_log import parse_log as parse_training_log
from topvenue_readiness_gate import build_report as build_readiness_report


DEFAULT_ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\paper_protocol")
DEFAULT_SUMMARY_DIR = Path("reports/paper_protocol_summary")
DEFAULT_LOG_DIR = Path("outputs/paper_protocol_queue")
TQDM_RE = re.compile(
    r"(?P<phase>train|eval):\s*(?P<pct>\d+)%.*\|\s*"
    r"(?P<step>\d+)\s*/\s*(?P<steps>\d+)\s*"
    r"\[(?P<elapsed>[^<\]]+)<(?P<eta>[^,\]]+),\s*(?P<rate>[^\]]+)\]"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_SUMMARY_DIR / "experiment_status_dashboard.json")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_SUMMARY_DIR / "experiment_status_dashboard.md")
    args = parser.parse_args()

    report = build_dashboard(args.root, args.summary_dir, args.log_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)


def build_dashboard(root: Path, summary_dir: Path, log_dir: Path) -> dict[str, Any]:
    rows = [inspect_run(root / name, log_dir=log_dir) for name in ROWS]
    completeness = _load_json(summary_dir / "protocol_completeness.json") or {}
    gate = _build_live_readiness(summary_dir)
    default_trend = _load_json(summary_dir / "v5_full_faf_live_training_trend.json") or {}
    summary = _load_json(summary_dir / "paper_protocol_summary.json") or {}
    algorithm_audit = _load_json(summary_dir / "algorithm_module_audit.json") or {}
    interval_quality = _load_json(summary_dir / "interval_quality_report.json") or {}
    wetness_state = _load_json(summary_dir / "wetness_state_report.json") or {}
    dataset_shortcut = _load_json(summary_dir / "dataset_shortcut_report.json") or {}
    dataset_image_style = _load_json(summary_dir / "dataset_image_style_audit.json") or {}
    input_canonicalization_style = _load_json(summary_dir / "input_canonicalization_style_audit.json") or {}
    evidence_failure = _load_json(summary_dir / "evidence_failure_report.json") or {}
    p0_claim = _load_json(summary_dir / "p0_claim_report.json") or {}
    paper_p0_table = _load_json(summary_dir / "paper_p0_ablation_table.json") or {}
    final_selection = _load_json(summary_dir / "final_method_selection_report.json") or {}
    safety_selection = _load_json(summary_dir / "safety_selection_report.json") or {}
    checkpoint_policy = _load_json(summary_dir / "checkpoint_policy_report.json") or {}
    config_to_code_trace = _load_json(summary_dir / "config_to_code_trace_report.json") or {}
    goal_evidence = _load_json(summary_dir / "goal_evidence_audit.json") or {}
    artifact_contract = _load_json(summary_dir / "artifact_contract_report.json") or {}
    claim_ledger = _load_json(summary_dir / "claim_evidence_ledger.json") or {}
    reviewer_checklist = _load_json(summary_dir / "reviewer_evidence_checklist.json") or {}
    gap_analysis = _load_json(summary_dir / "current_algorithm_gap_analysis.json") or {}
    queue_recovery_path = summary_dir / "queue_recovery_report.json"
    runtime_guard_path = summary_dir / "runtime_guard_report.json"
    active_training_watch_path = summary_dir / "active_training_watch_report.json"
    queue_recovery = _load_json(queue_recovery_path) or {}
    external_benchmark = _load_json(summary_dir / "external_benchmark_report.json") or {}
    open_source_plan = _load_json(summary_dir / "open_source_reproducibility_plan.json") or {}
    innovation_roadmap = _load_json(summary_dir / "topvenue_innovation_roadmap.json") or {}
    candidate_hypothesis = _load_json(summary_dir / "candidate_hypothesis_matrix.json") or {}
    lodo_generalization = _load_json(summary_dir / "lodo_generalization_report.json") or {}
    friction_interval_sources = _load_json(summary_dir / "friction_interval_source_audit.json") or {}
    default_live_training_diagnosis = _load_json(summary_dir / "v5_full_faf_training_diagnosis.json") or {}
    config_batch_check = _load_json(summary_dir / "config_batch_check.json") or {}
    config_forward_check = _load_json(summary_dir / "config_forward_loss_smoke.json") or {}
    gpu_protocol_audit = _load_json(summary_dir / "gpu_protocol_audit.json") or {}
    handoff_health = _load_json(summary_dir / "handoff_health_report.json") or {}
    runtime_guard = _load_json(runtime_guard_path) or {}
    roadsaw_lodo_protocol = _load_json(summary_dir / "roadsaw_lodo_protocol_audit.json") or {}
    fair_comparison_protocol = _load_json(summary_dir / "fair_comparison_protocol_audit.json") or {}
    active_training_watch = _load_json(active_training_watch_path) or {}
    if _is_older(active_training_watch_path, queue_recovery_path):
        active_training_watch = {}
    if _is_older(runtime_guard_path, queue_recovery_path):
        runtime_guard = {}
    fast_screen = _load_json(summary_dir / "fast_screen_status_report.json") or {}
    fast_to_formal = _load_json(summary_dir / "fast_to_formal_promotion_report.json") or {}

    active_rows_raw = [row for row in rows if row.get("status") in {"running_or_partial", "partial_ci_missing"}]
    tqdm = {
        row["name"]: _tqdm_snapshot(log_dir, row["name"])
        for row in active_rows_raw
    }
    active_rows = _active_rows_with_tqdm(active_rows_raw, tqdm)
    active_run_name = active_rows[0].get("name") if active_rows else "v5_full_faf"
    trend = _load_json(summary_dir / f"{active_run_name}_live_training_trend.json") or default_trend
    live_training_diagnosis = (
        _load_json(summary_dir / f"{active_run_name}_training_diagnosis.json")
        or default_live_training_diagnosis
    )
    live_training = _compact_trend(trend)
    _merge_active_row_progress(live_training, active_rows)
    _merge_tqdm_into_live_training(live_training, tqdm, active_rows)
    _merge_active_log_completed_metrics(live_training, active_rows, log_dir)
    group_status = _group_status_from_artifact_contract(
        completeness.get("group_status", {}),
        artifact_contract,
    )
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "summary_dir": str(summary_dir),
        "active_rows": active_rows,
        "active_tqdm": tqdm,
        "progress_counts": _progress_counts(rows),
        "group_status": group_status,
        "requirements": completeness.get("requirements", []),
        "readiness": {
            "verdict": gate.get("verdict"),
            "num_blocks": gate.get("num_blocks"),
            "num_warnings": gate.get("num_warnings"),
            "blocking_gates": [item for item in gate.get("gates", []) if item.get("level") == "block"],
            "warning_gates": [item for item in gate.get("gates", []) if item.get("level") == "warn"],
            "pass_gates": [item for item in gate.get("gates", []) if item.get("level") == "pass"],
        },
        "live_training": live_training,
        "core_ablation": summary.get("core_ablation", []),
        "p0_claim": _compact_p0_claim(p0_claim),
        "paper_p0_ablation_table": _compact_paper_p0_table(paper_p0_table),
        "p3_ready_runs": _p3_ready_runs(algorithm_audit),
        "final_route_sanity": _compact_final_route_sanity(algorithm_audit),
        "interval_quality": _compact_interval_quality(interval_quality),
        "wetness_state": _compact_wetness_state(wetness_state),
        "dataset_shortcut": _compact_dataset_shortcut(dataset_shortcut),
        "dataset_image_style": _compact_dataset_image_style(dataset_image_style),
        "input_canonicalization_style": _compact_input_canonicalization_style(input_canonicalization_style),
        "evidence_failure": _compact_evidence_failure(evidence_failure),
        "final_method_selection": _compact_final_method_selection(final_selection),
        "safety_selection": _compact_safety_selection(safety_selection),
        "checkpoint_policy": _compact_checkpoint_policy(checkpoint_policy),
        "config_to_code_trace": _compact_config_to_code_trace(config_to_code_trace),
        "goal_evidence": _compact_goal_evidence(goal_evidence),
        "artifact_contract": _compact_artifact_contract(artifact_contract),
        "claim_evidence": _compact_claim_evidence(claim_ledger),
        "reviewer_evidence_checklist": _compact_reviewer_evidence_checklist(reviewer_checklist),
        "gap_analysis": _compact_gap_analysis(gap_analysis),
        "queue_recovery": _compact_queue_recovery(queue_recovery, active_rows),
        "external_benchmark": _compact_external_benchmark(external_benchmark),
        "open_source_reproducibility": _compact_open_source_plan(open_source_plan),
        "topvenue_innovation_roadmap": _compact_innovation_roadmap(innovation_roadmap),
        "candidate_hypothesis_matrix": _compact_candidate_hypothesis_matrix(candidate_hypothesis),
        "lodo_generalization": _compact_lodo_generalization(lodo_generalization),
        "friction_interval_sources": _compact_friction_interval_sources(friction_interval_sources),
        "live_training_diagnosis": _compact_live_training_diagnosis(
            live_training_diagnosis,
            current_run=live_training.get("run"),
            live_training=live_training,
            active_watch=active_training_watch,
        ),
        "config_batch_check": _compact_config_batch_check(config_batch_check),
        "config_forward_loss_smoke": _compact_forward_loss_smoke(config_forward_check),
        "gpu_protocol_audit": _compact_gpu_protocol_audit(gpu_protocol_audit),
        "handoff_health": _compact_handoff_health(handoff_health, active_rows),
        "runtime_guard": _compact_runtime_guard(runtime_guard),
        "active_training_watch": _compact_active_training_watch(active_training_watch),
        "roadsaw_lodo_protocol": _compact_roadsaw_lodo_protocol(roadsaw_lodo_protocol),
        "fair_comparison_protocol": _compact_fair_comparison_protocol(fair_comparison_protocol),
        "fast_screen": _compact_fast_screen(fast_screen),
        "fast_to_formal": _compact_fast_to_formal(fast_to_formal),
        "system": {
            "gpu": _gpu_snapshot(),
            "disks": _disk_snapshot(["C:\\", "D:\\", "E:\\"]),
        },
        "next_milestones": _next_milestones(completeness, gate),
    }


def _progress_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        out[status] = out.get(status, 0) + 1
    return out


def _build_live_readiness(summary_dir: Path) -> dict[str, Any]:
    try:
        return build_readiness_report(summary_dir)
    except Exception as exc:
        gate = _load_json(summary_dir / "topvenue_readiness_gate.json") or {}
        if isinstance(gate, dict):
            gate = dict(gate)
            gate["dashboard_live_recompute_error"] = str(exc)
            return gate
        return {
            "verdict": "not_ready",
            "num_blocks": 1,
            "num_warnings": 0,
            "gates": [
                {
                    "level": "block",
                    "name": "topvenue_readiness_gate_recompute_failed",
                    "message": str(exc),
                }
            ],
        }


def _group_status_from_artifact_contract(
    fallback: dict[str, Any],
    artifact_contract: dict[str, Any],
) -> dict[str, Any]:
    out = dict(fallback or {})
    hard = artifact_contract.get("hard_status", {}) if isinstance(artifact_contract, dict) else {}
    mapping = {
        "p0_ablation": ["p0_ablation"],
        "lodo": ["lodo"],
        "single_dataset_fair": ["single_dataset_faf", "single_dataset_baselines"],
        "p1_candidates": ["p1_candidates"],
        "final_method": ["final_method_lodo", "final_method_single_dataset"],
    }
    for source, targets in mapping.items():
        row = hard.get(source)
        if not isinstance(row, dict):
            continue
        missing = list(row.get("missing", []) or [])
        for target in targets:
            target_missing = missing
            if source == "single_dataset_fair":
                if target == "single_dataset_faf":
                    target_missing = [name for name in missing if str(name).startswith("single_")]
                else:
                    target_missing = [name for name in missing if str(name).startswith("baseline_single_")]
            elif source == "final_method":
                if target == "final_method_lodo":
                    target_missing = [name for name in missing if str(name).startswith("final_lodo_")]
                else:
                    target_missing = [name for name in missing if str(name).startswith("final_single_")]
            previous = fallback.get(target, {}) if isinstance(fallback, dict) else {}
            target_runs = int(previous.get("num_runs", 0) or 0)
            if target_runs <= 0:
                target_runs = int(row.get("num_runs", 0) or 0)
            if target in {"single_dataset_faf", "single_dataset_baselines", "final_method_lodo", "final_method_single_dataset"}:
                target_runs = max(target_runs, len(target_missing))
            target_complete = max(0, target_runs - len(target_missing))
            out[target] = {
                "complete": not target_missing,
                "num_runs": target_runs,
                "num_complete": target_complete,
                "missing_runs": target_missing,
                "source": "artifact_contract",
            }
    return out


def _compact_trend(trend: dict[str, Any]) -> dict[str, Any]:
    latest = trend.get("latest_completed_epoch") or {}
    val = latest.get("val") or {}
    active = trend.get("active_progress")
    delta = (trend.get("trend") or {}).get("from_previous_to_latest", {})
    return {
        "run": trend.get("run"),
        "active_progress": active,
        "latest_completed_epoch": latest.get("epoch"),
        "latest_val_loss": val.get("loss"),
        "latest_val_risk_acc": val.get("acc_risk"),
        "latest_val_friction_acc": val.get("acc_friction"),
        "latest_raw_coverage": val.get("mu_interval_coverage"),
        "latest_raw_width": val.get("mu_interval_width"),
        "previous_delta_val_loss": delta.get("val_loss"),
        "previous_delta_val_risk_acc": delta.get("val_acc_risk"),
        "previous_delta_raw_coverage": delta.get("val_mu_interval_coverage"),
    }


def _merge_active_row_progress(live: dict[str, Any], active_rows: list[dict[str, Any]]) -> None:
    if not active_rows:
        return
    row = active_rows[0]
    row_progress = {
        "epoch": row.get("active_epoch"),
        "epochs": row.get("active_epochs"),
        "step": row.get("active_step"),
        "steps": row.get("active_steps"),
    }
    if row_progress["epoch"] is None:
        return
    current = live.get("active_progress") or {}
    current_step = _as_int(current.get("step"))
    row_step = _as_int(row_progress.get("step"))
    if not current or row_step is None or current_step is None or row_step >= current_step:
        live["active_progress"] = row_progress
        live["run"] = row.get("name") or live.get("run")


def _merge_tqdm_into_live_training(
    live: dict[str, Any],
    tqdm: dict[str, dict[str, Any] | None],
    active_rows: list[dict[str, Any]],
) -> None:
    for row in active_rows:
        snapshot = tqdm.get(str(row.get("name")))
        if not snapshot or snapshot.get("phase") != "train":
            continue
        epoch = row.get("active_epoch")
        epochs = row.get("active_epochs")
        if epoch is None:
            continue
        tqdm_step = _as_int(snapshot.get("step"))
        tqdm_steps = _as_int(snapshot.get("steps"))
        if tqdm_step is None or tqdm_steps is None:
            continue
        current = live.get("active_progress") or {}
        current_step = _as_int(current.get("step"))
        if current_step is None or tqdm_step >= current_step:
            live["active_progress"] = {
                "epoch": epoch,
                "epochs": epochs,
                "step": tqdm_step,
                "steps": tqdm_steps,
            }
            live["run"] = row.get("name") or live.get("run")
        return


def _merge_active_log_completed_metrics(
    live: dict[str, Any],
    active_rows: list[dict[str, Any]],
    log_dir: Path,
) -> None:
    if not active_rows:
        return
    row = active_rows[0]
    log_path = _resolve_log_path(row.get("active_log"), log_dir)
    if log_path is None:
        return
    try:
        history = parse_training_log(log_path)
    except (OSError, UnicodeDecodeError):
        return
    completed = [item for item in history if item.get("val_metrics")]
    if not completed:
        return
    latest = completed[-1]
    val = latest.get("val_metrics") or {}
    prev_val = (completed[-2].get("val_metrics") if len(completed) >= 2 else None) or {}
    live["run"] = row.get("name") or live.get("run")
    live["latest_completed_epoch"] = latest.get("epoch")
    live["latest_val_loss"] = val.get("loss")
    live["latest_val_risk_acc"] = val.get("acc_risk")
    live["latest_val_friction_acc"] = val.get("acc_friction")
    live["latest_raw_coverage"] = val.get("mu_interval_coverage")
    live["latest_raw_width"] = val.get("mu_interval_width")
    live["previous_delta_val_loss"] = _diff_float(
        _as_float(val.get("loss")), _as_float(prev_val.get("loss"))
    )
    live["previous_delta_val_risk_acc"] = _diff_float(
        _as_float(val.get("acc_risk")), _as_float(prev_val.get("acc_risk"))
    )
    live["previous_delta_raw_coverage"] = _diff_float(
        _as_float(val.get("mu_interval_coverage")), _as_float(prev_val.get("mu_interval_coverage"))
    )
    live["latest_completed_epoch_source"] = str(log_path)


def _resolve_log_path(value: Any, log_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        candidates.append(log_dir / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _p3_ready_runs(algorithm_audit: dict[str, Any]) -> list[str]:
    rows = algorithm_audit.get("rows", []) if isinstance(algorithm_audit, dict) else []
    ready = []
    for row in rows:
        modules = row.get("modules", {})
        if modules.get("coverage_aware_training") and modules.get("safety_weighted_coverage"):
            ready.append(str(row.get("run")))
    return ready


def _compact_final_route_sanity(algorithm_audit: dict[str, Any]) -> dict[str, Any]:
    rows = algorithm_audit.get("rows", []) if isinstance(algorithm_audit, dict) else []
    final_rows = [row for row in rows if str(row.get("run", "")).startswith("final_")]
    if not final_rows:
        return {}

    unstable_present: list[str] = []
    missing_core: list[str] = []
    missing_safety: list[str] = []
    for row in final_rows:
        run = str(row.get("run"))
        modules = row.get("modules", {})
        if modules.get("friction_set") or modules.get("dg_losses"):
            unstable_present.append(run)
        if not modules.get("physics_texture") or not modules.get("evidence_field"):
            missing_core.append(run)
        required_safety = [
            "fourier_style_jitter",
            "road_likelihood_prior",
            "pseudo_road_mask_supervision",
            "weak_view_consistency",
            "roi_attention_constraint",
            "coverage_aware_training",
            "safety_weighted_coverage",
            "semantic_conditional_alignment",
            "wetness_ordinal_loss",
        ]
        if any(not modules.get(name) for name in required_safety):
            missing_safety.append(run)

    verdict = "pass"
    if unstable_present or missing_core or missing_safety:
        verdict = "warn"
    return {
        "verdict": verdict,
        "num_final_runs": len(final_rows),
        "unstable_present": unstable_present,
        "missing_core": missing_core,
        "missing_safety": missing_safety,
        "policy": (
            "Final-method configs should prune currently unstable FrictionSet/DG losses "
            "while keeping the lean PhysicsTexture + EvidenceField + road-ROI safety route "
            "with small state-conditioned semantic alignment."
        ),
    }


def _compact_interval_quality(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("rows"):
        return {}
    return {
        "num_runs": report.get("num_runs"),
        "num_cells": report.get("num_cells"),
        "num_watchlist_items": report.get("num_watchlist_items"),
        "top_watchlist": report.get("watchlist", [])[:5],
    }


def _compact_wetness_state(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("rows"):
        return {}
    complete = [row for row in report.get("rows", []) if row.get("status") == "complete"]
    complete.sort(key=lambda row: row.get("run", ""))
    return {
        "num_complete": report.get("num_complete"),
        "num_watchlist": report.get("num_watchlist"),
        "watchlist": report.get("watchlist", [])[:5],
        "latest_complete": complete[-1] if complete else None,
    }


def _compact_dataset_shortcut(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("rows"):
        return {}
    rows = [row for row in report.get("rows", []) if row.get("status") == "complete"]
    rows.sort(key=lambda row: float(row.get("core_state_conditioned_balanced_accuracy") or 1.0))
    return {
        "verdict": report.get("verdict"),
        "threshold": report.get("threshold"),
        "num_complete": report.get("num_complete"),
        "num_high_shortcut": report.get("num_high_shortcut"),
        "best_completed": report.get("best_completed_by_core_state_probe"),
        "top_rows": rows[:5],
    }


def _compact_dataset_image_style(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("datasets"):
        return {}
    pipeline = report.get("config_image_pipeline") or {}
    cross = report.get("cross_dataset_signals") or {}
    rows = []
    for name, row in sorted((report.get("datasets") or {}).items()):
        rows.append(
            {
                "dataset": name,
                "num_samples": row.get("num_samples"),
                "width_median": _nested_value(row, "width", "median"),
                "height_median": _nested_value(row, "height", "median"),
                "aspect_median": _nested_value(row, "aspect", "median"),
                "brightness_mean": _nested_value(row, "brightness", "mean"),
                "contrast_mean": _nested_value(row, "contrast", "mean"),
                "saturation_mean": _nested_value(row, "saturation", "mean"),
                "suffixes": row.get("suffixes", {}),
                "modes": row.get("modes", {}),
            }
        )
    return {
        "generated_at": report.get("generated_at"),
        "claim_boundary": report.get("claim_boundary"),
        "image_size": pipeline.get("image_size"),
        "resize_mode": pipeline.get("resize_mode", "stretch"),
        "aspect_span": _cross_span(cross, "aspect_median_range"),
        "width_span": _cross_span(cross, "width_median_range"),
        "height_span": _cross_span(cross, "height_median_range"),
        "brightness_span": _cross_span(cross, "brightness_range"),
        "saturation_span": _cross_span(cross, "saturation_range"),
        "datasets": rows,
        "recommendations": list(report.get("recommendations") or [])[:5],
    }


def _compact_input_canonicalization_style(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("configs"):
        return {}
    rows = []
    for item in report.get("configs") or []:
        transform = item.get("transform") or {}
        cross = item.get("cross_dataset_signals") or {}
        relative = item.get("relative_to_first_config") or {}
        rgb_spans = [
            _cross_span(cross, "red_mean_span"),
            _cross_span(cross, "green_mean_span"),
            _cross_span(cross, "blue_mean_span"),
        ]
        rgb_spans_numeric = [float(value) for value in rgb_spans if isinstance(value, (int, float))]
        rows.append(
            {
                "run": item.get("run"),
                "resize_mode": transform.get("resize_mode"),
                "gray_world_alpha": transform.get("gray_world_alpha"),
                "fourier_low_freq_jitter_p": transform.get("fourier_low_freq_jitter_p"),
                "random_resized_crop": transform.get("random_resized_crop"),
                "style_gap_score": _cross_signal_value(cross, "style_gap_score"),
                "style_gap_relative": _nested_value(relative, "style_gap_score", "relative"),
                "brightness_span": _cross_span(cross, "brightness_span"),
                "contrast_span": _cross_span(cross, "contrast_span"),
                "saturation_span": _cross_span(cross, "saturation_span"),
                "rgb_span_max": max(rgb_spans_numeric) if rgb_spans_numeric else None,
                "channel_mean_spread_span": _cross_span(cross, "channel_mean_spread_span"),
            }
        )
    valid = [row for row in rows if isinstance(row.get("style_gap_score"), (int, float))]
    best = min(valid, key=lambda row: float(row["style_gap_score"])) if valid else {}
    baseline = rows[0] if rows else {}
    return {
        "generated_at": report.get("generated_at"),
        "claim_boundary": report.get("claim_boundary"),
        "max_samples_per_dataset": report.get("max_samples_per_dataset"),
        "baseline_run": baseline.get("run"),
        "best_run": best.get("run"),
        "best_style_gap_score": best.get("style_gap_score"),
        "best_style_gap_relative": best.get("style_gap_relative"),
        "rows": rows,
        "recommendations": list(report.get("recommendations") or [])[:5],
    }


def _nested_value(row: dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _cross_span(cross: dict[str, Any], key: str) -> Any:
    value = cross.get(key) if isinstance(cross, dict) else None
    if not isinstance(value, dict):
        return None
    return value.get("span")


def _cross_signal_value(cross: dict[str, Any], key: str) -> Any:
    value = cross.get(key) if isinstance(cross, dict) else None
    if isinstance(value, dict):
        return value.get("span")
    return value


def _compact_evidence_failure(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("runs"):
        return {}
    examples = report.get("examples", [])
    roadsaw_examples = [
        item for item in examples
        if str(item.get("dataset", "")).lower() == "roadsaw"
        and "failure" in str(item.get("tag", "")).lower()
    ]
    return {
        "num_evidence_runs": report.get("num_evidence_runs"),
        "num_examples": len(examples),
        "num_roadsaw_failure_examples": len(roadsaw_examples),
        "runs": report.get("runs", [])[:5],
    }


def _compact_p0_claim(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("adjacent_deltas"):
        return {}
    return {
        "core_status": report.get("core_status"),
        "recommendations": [
            {
                "module": row.get("module"),
                "status": row.get("status"),
                "recommendation": row.get("claim_recommendation"),
            }
            for row in report.get("adjacent_deltas", [])
        ],
    }


def _compact_paper_p0_table(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("rows"):
        return {}
    return {
        "status": report.get("status"),
        "claim_boundary": report.get("claim_boundary"),
        "best_by_metric": report.get("best_by_metric") or {},
        "rows": report.get("rows", []),
        "artifacts": {
            "md": "reports/paper_protocol_summary/paper_p0_ablation_table.md",
            "csv": "reports/paper_protocol_summary/paper_p0_ablation_table.csv",
            "tex": "reports/paper_protocol_summary/paper_p0_ablation_table.tex",
        },
    }


def _compact_final_method_selection(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("selection_rule"):
        return {}
    return {
        "verdict": report.get("verdict"),
        "top_completed": report.get("provisional_top_completed", [])[:5],
        "risk_register": report.get("risk_register", [])[:8],
        "recommended_action": report.get("recommended_action", [])[:6],
    }


def _compact_safety_selection(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("rows"):
        return {}
    rows = report.get("rows", [])
    complete = [row for row in rows if row.get("status") == "complete"]
    helpful = [
        row
        for row in complete
        if (_as_float(row.get("delta_low_friction_recall")) or 0.0) > 0
        or (_as_float(row.get("delta_raw_interval_coverage")) or 0.0) > 0
        or (_as_float(row.get("delta_risk_f1")) or 0.0) > 0
    ]
    ranked = sorted(
        complete,
        key=lambda row: (
            _as_float(row.get("delta_low_friction_recall")) or 0.0,
            _as_float(row.get("delta_raw_interval_coverage")) or 0.0,
            _as_float(row.get("delta_risk_f1")) or 0.0,
        ),
        reverse=True,
    )
    return {
        "verdict": report.get("verdict"),
        "rule": report.get("rule"),
        "num_rows": len(rows),
        "num_complete": len(complete),
        "num_helpful": len(helpful),
        "top_rows": ranked[:5],
    }


def _compact_checkpoint_policy(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("policy"):
        return {}
    policy = report.get("policy", {})
    core = report.get("core_status", {})
    safety = report.get("safety_status", {})
    live = report.get("live_checkpoint") or report.get("live_v5", {})
    return {
        "policy": policy,
        "audit_rules": report.get("audit_rules", [])[:4],
        "p0_complete": len(core.get("complete_methods", []) or []),
        "p0_incomplete": core.get("incomplete_methods", []),
        "safety_verdict": safety.get("verdict"),
        "safety_complete": safety.get("num_complete"),
        "safety_rows": safety.get("num_rows"),
        "live_checkpoint": live,
    }


def _compact_config_to_code_trace(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("rows"):
        return {}
    failures = [
        row.get("name")
        for row in report.get("rows", [])
        if row.get("num_configured_runs", 0) > 0 and not row.get("source_ok")
    ]
    return {
        "verdict": report.get("verdict"),
        "num_rows": report.get("num_rows"),
        "num_blocks": report.get("num_blocks"),
        "num_warnings": report.get("num_warnings"),
        "failures": failures,
    }


def _compact_goal_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {}
    current = report.get("current_execution") if isinstance(report.get("current_execution"), dict) else {}
    return {
        "generated_at": report.get("generated_at"),
        "current_run": current.get("name"),
        "current_status": current.get("status"),
        "num_requirements": report.get("num_requirements"),
        "num_incomplete_requirements": report.get("num_incomplete_requirements"),
        "incomplete_requirements": report.get("incomplete_requirements", [])[:8],
        "claim_boundary": report.get("claim_boundary"),
    }


def _compact_artifact_contract(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {}
    hard = report.get("hard_status", {}) if isinstance(report.get("hard_status"), dict) else {}
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    incomplete = [
        {
            "name": row.get("name"),
            "scope": row.get("scope"),
            "progress_status": row.get("progress_status"),
            "contract_status": row.get("contract_status"),
            "missing": row.get("missing_required_artifacts", [])[:6],
            "next_action": row.get("next_action"),
        }
        for row in rows
        if row.get("contract_status") != "complete"
    ]
    return {
        "verdict": report.get("verdict"),
        "num_runs": report.get("num_runs"),
        "num_contract_complete": report.get("num_contract_complete"),
        "num_contract_incomplete": report.get("num_contract_incomplete"),
        "num_invalid_complete_like": report.get("num_invalid_complete_like"),
        "num_stale_rows": report.get("num_stale_rows"),
        "hard_status": hard,
        "top_incomplete": incomplete[:8],
    }


def _compact_claim_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {}
    rows = report.get("claim_rows", []) if isinstance(report.get("claim_rows"), list) else []
    return {
        "generated_at": report.get("generated_at"),
        "status_counts": report.get("status_counts", {}),
        "top_not_supported": [
            {
                "claim": row.get("claim"),
                "status": row.get("status"),
                "missing_or_risk": row.get("missing_or_risk", [])[:5],
                "allowed_wording": row.get("allowed_wording"),
            }
            for row in rows
            if row.get("status") in {"not_supported", "not_supported_yet"}
        ][:5],
        "partial": [
            {
                "claim": row.get("claim"),
                "missing_or_risk": row.get("missing_or_risk", [])[:5],
                "allowed_wording": row.get("allowed_wording"),
            }
            for row in rows
            if row.get("status") == "partial"
        ][:5],
    }


def _compact_reviewer_evidence_checklist(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("claims"):
        return {}
    claims = [row for row in report.get("claims", []) if isinstance(row, dict)]
    counts: dict[str, int] = {}
    for row in claims:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "generated_at": report.get("generated_at"),
        "readiness_verdict": report.get("readiness_verdict"),
        "status_counts": counts,
        "supported": [row for row in claims if row.get("status") == "supported"][:5],
        "partial": [row for row in claims if row.get("status") == "partial"][:5],
        "not_supported": [
            row for row in claims if row.get("status") in {"not_supported", "not_supported_yet"}
        ][:5],
        "strict_rules": list(report.get("strict_rules") or [])[:5],
        "next_milestones": list(report.get("next_milestones") or [])[:6],
        "artifact": "reports/paper_protocol_summary/reviewer_evidence_checklist.md",
    }


def _compact_gap_analysis(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("candidate_implementation"):
        return {}
    return {
        "generated_at": report.get("generated_at"),
        "p0_status": report.get("p0_status"),
        "pending_p0": report.get("pending_p0", []),
        "key_failures": report.get("key_failures", [])[:5],
        "candidate_implementation": report.get("candidate_implementation", [])[:4],
        "hard_claim_rules": report.get("hard_claim_rules", [])[:4],
    }


def _compact_queue_recovery(report: dict[str, Any], active_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("queue_order"):
        return {}
    report_active_by_name = {
        row.get("name"): row
        for row in report.get("active_rows", [])
        if row.get("name")
    }
    fresh_active = []
    for row in active_rows or []:
        report_row = report_active_by_name.get(row.get("name")) or {}
        source = _newer_progress_row(row, report_row)
        fresh_active.append(
            {
                "name": source.get("name") or row.get("name"),
                "status": source.get("status") or row.get("status"),
                "active_epoch": source.get("active_epoch"),
                "active_epochs": source.get("active_epochs"),
                "active_step": source.get("active_step"),
                "active_steps": source.get("active_steps"),
                "active_phase": source.get("active_phase"),
                "epoch": source.get("epoch"),
                "epochs": source.get("epochs"),
                "active_log_age_seconds": source.get("active_log_age_seconds")
                if source.get("active_log_age_seconds") is not None
                else report_row.get("active_log_age_seconds"),
                "active_log_stale": source.get("active_log_stale")
                if source.get("active_log_stale") is not None
                else report_row.get("active_log_stale"),
            }
        )
    return {
        "generated_at": report.get("generated_at"),
        "num_total": report.get("num_total"),
        "num_complete": report.get("num_complete"),
        "num_partial": report.get("num_partial"),
        "num_missing": report.get("num_missing"),
        "active_rows": fresh_active[:3] if fresh_active else report.get("active_rows", [])[:3],
        "next_incomplete": report.get("next_incomplete"),
        "process_snapshot": _compact_queue_process_snapshot(report),
        "recovery_commands": report.get("recovery_commands", {}),
    }


def _compact_queue_process_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    processes = report.get("process_snapshot", [])
    if not isinstance(processes, list):
        return {}
    counts: dict[str, int] = {}
    followups: list[dict[str, Any]] = []
    followup_kinds = {
        "fast_screen_followup",
        "rscd27_followup",
        "direct_visual_followup",
        "postprocess_followup",
        "v17_candidate_followup",
        "waiting_queue",
    }
    for proc in processes:
        if not isinstance(proc, dict):
            continue
        kind = str(proc.get("kind") or "other")
        counts[kind] = counts.get(kind, 0) + 1
        if kind not in followup_kinds:
            continue
        preview = str(proc.get("decoded_command") or proc.get("CommandLine") or "")
        preview = " ".join(preview.replace("\r", " ").replace("\n", " ").split())
        if len(preview) > 180:
            preview = preview[:177] + "..."
        followups.append(
            {
                "kind": kind,
                "pid": proc.get("ProcessId"),
                "parent": proc.get("ParentProcessId"),
                "phase": proc.get("phase"),
                "wait_pid": proc.get("wait_pid"),
                "preview": preview,
            }
        )
    return {"counts": counts, "followups": followups[:8]}


def _newer_progress_row(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_epoch = _as_int(left.get("active_epoch") or left.get("epoch"))
    right_epoch = _as_int(right.get("active_epoch") or right.get("epoch"))
    if left_epoch is not None and right_epoch is not None and left_epoch != right_epoch:
        return left if left_epoch > right_epoch else right
    left_step = _as_int(left.get("active_step"))
    right_step = _as_int(right.get("active_step"))
    if left_step is not None and right_step is not None and left_step != right_step:
        return left if left_step > right_step else right
    if right.get("active_phase") and right.get("active_log_age_seconds") is not None:
        return right
    if right_step is not None and left_step is None:
        return right
    return left


def _active_rows_with_tqdm(
    active_rows: list[dict[str, Any]],
    tqdm: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in active_rows:
        out = dict(row)
        snapshot = tqdm.get(row.get("name"))
        if snapshot:
            out["active_phase"] = snapshot.get("phase") or out.get("active_phase")
            out["active_step"] = snapshot.get("step")
            out["active_steps"] = snapshot.get("steps")
        merged.append(out)
    return merged


def _compact_open_source_plan(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("rows"):
        return {}
    rows = report.get("rows", [])
    return {
        "num_sources": report.get("num_sources"),
        "num_implemented_or_configured": report.get("num_implemented_or_configured"),
        "num_future_only": report.get("num_future_only"),
        "claim_policy": report.get("claim_policy"),
        "strict_claim_rules": report.get("strict_claim_rules", [])[:5],
        "implemented_rows": [
            {
                "name": row.get("name"),
                "status": row.get("integration_status"),
                "project_use": row.get("project_use"),
            }
            for row in rows
            if str(row.get("integration_status", "")).startswith("implemented")
            or str(row.get("integration_status", "")).startswith("partly")
            or str(row.get("integration_status", "")).startswith("protocol")
        ],
    }


def _compact_external_benchmark(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("public_sources"):
        return {}
    sources = report.get("public_sources", [])
    alignment = report.get("dataset_alignment", [])
    requirements = report.get("completion_relevant_requirements", {})
    return {
        "num_sources": len(sources),
        "datasets": [row.get("dataset") for row in alignment],
        "strict_comparison_rule": report.get("strict_comparison_rule"),
        "completion_relevant_requirements": {
            name: item.get("status", "missing") for name, item in requirements.items()
        },
        "source_names": [item.get("name") for item in sources],
    }


def _compact_innovation_roadmap(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("pattern_rows"):
        return {}
    return {
        "num_patterns": len(report.get("pattern_rows", [])),
        "num_sources": len(report.get("source_rows", [])),
        "readiness": report.get("readiness", {}),
        "current_evidence": report.get("current_evidence", {}),
        "next_decisions": report.get("next_decisions", [])[:5],
        "strict_claim_rules": report.get("strict_claim_rules", [])[:5],
    }


def _compact_candidate_hypothesis_matrix(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("rows"):
        return {}
    coverage = report.get("coverage", {}) if isinstance(report.get("coverage"), dict) else {}
    signals = report.get("current_failure_signals", []) if isinstance(report.get("current_failure_signals"), list) else []
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    next_rows = [
        row
        for row in rows
        if row.get("contract_status") != "complete"
    ][:8]
    return {
        "verdict": report.get("verdict"),
        "num_rows": report.get("num_rows"),
        "phase_counts": coverage.get("phase_counts", {}),
        "contract_status_counts": coverage.get("contract_status_counts", {}),
        "candidate_runs": coverage.get("candidate_runs", []),
        "final_runs": coverage.get("final_runs", []),
        "fair_baseline_runs": coverage.get("fair_baseline_runs", []),
        "lodo_runs": coverage.get("lodo_runs", []),
        "failure_signals": signals[:5],
        "next_rows": next_rows,
        "decision_policy": report.get("decision_policy", [])[:5],
    }


def _compact_lodo_generalization(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("roadsaw_readout"):
        return {}
    readout = report.get("roadsaw_readout") or {}
    metrics = readout.get("heldout_metrics") or {}
    deltas = readout.get("deltas_vs_mixed_reference") or {}
    preliminary = []
    for row in report.get("rows", []):
        if not isinstance(row, dict) or not row.get("preliminary_evaluate_test"):
            continue
        metrics_pre = row.get("preliminary_evaluate_test") or {}
        preliminary.append(
            {
                "held_out": row.get("held_out"),
                "status": row.get("status"),
                "acc_friction": metrics_pre.get("acc_friction"),
                "acc_risk": metrics_pre.get("acc_risk"),
                "acc_snow": metrics_pre.get("acc_snow"),
                "raw_coverage": metrics_pre.get("raw_coverage"),
                "raw_width": metrics_pre.get("raw_width"),
                "loss": metrics_pre.get("loss"),
            }
        )
    return {
        "verdict": report.get("verdict"),
        "roadsaw_verdict": readout.get("verdict"),
        "roadsaw_status": readout.get("status"),
        "claim_boundary": readout.get("claim_boundary"),
        "roadsaw_risk_f1": metrics.get("risk_f1"),
        "roadsaw_friction_f1": metrics.get("friction_f1"),
        "roadsaw_calibrated_coverage": metrics.get("calibrated_coverage"),
        "delta_risk_f1_vs_mixed": deltas.get("risk_f1"),
        "delta_friction_f1_vs_mixed": deltas.get("friction_f1"),
        "preliminary_evaluate_test": preliminary,
        "next_actions": readout.get("next_actions", [])[:4],
    }


def _compact_roadsaw_lodo_protocol(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("verdict"):
        return {}
    splits = report.get("splits", {}) if isinstance(report.get("splits"), dict) else {}
    out_splits: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        summary = splits.get(split, {}) if isinstance(splits.get(split), dict) else {}
        out_splits[split] = {
            "rows": summary.get("num_rows"),
            "datasets": summary.get("datasets", {}),
            "split_values": summary.get("split_values", {}),
            "domains": summary.get("domains", {}),
            "mu_low": summary.get("min_mu_low"),
            "mu_high": summary.get("max_mu_high"),
        }
    checks = report.get("checks", []) if isinstance(report.get("checks"), list) else []
    return {
        "verdict": report.get("verdict"),
        "config": report.get("config"),
        "output_dir": report.get("output_dir"),
        "splits": out_splits,
        "num_blocks": sum(1 for item in checks if item.get("level") == "block"),
        "num_warnings": sum(1 for item in checks if item.get("level") == "warn"),
        "checks": checks[:5],
        "policy": report.get("policy", [])[:4],
    }


def _compact_fair_comparison_protocol(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("verdict"):
        return {}
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    return {
        "verdict": report.get("verdict"),
        "num_pairs": report.get("num_pairs"),
        "num_blocks": report.get("num_blocks"),
        "num_warnings": report.get("num_warnings"),
        "strict_single_pairs_pass": sum(
            1
            for row in rows
            if row.get("scope") == "single_dataset_full_faf_vs_convnext"
            and row.get("status") == "pass"
        ),
        "final_pairs_documented": sum(
            1
            for row in rows
            if row.get("scope") == "final_lean_vs_convnext"
            and row.get("status") in {"pass", "pass_with_warnings"}
        ),
        "rows": rows[:6],
        "non_pass_checks": [
            item for item in report.get("checks", []) if item.get("level") != "pass"
        ][:6],
        "policy": report.get("policy", [])[:4],
    }


def _compact_fast_screen(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("rows"):
        return {}
    rows = report.get("rows", []) if isinstance(report.get("rows"), list) else []
    complete = [row for row in rows if row.get("status") == "complete"]
    complete.sort(key=lambda row: _as_float(row.get("screen_score")) or -1.0, reverse=True)
    return {
        "verdict": report.get("verdict"),
        "counts": report.get("counts", {}),
        "top_complete": complete[:5],
        "next_actions": report.get("next_actions", [])[:4],
        "decision_rules": report.get("decision_rules", [])[:5],
        "claim_boundary": report.get("claim_boundary"),
    }


def _compact_fast_to_formal(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("verdict"):
        return {}
    return {
        "verdict": report.get("verdict"),
        "promoted": report.get("promoted", [])[:5],
        "fallback_sources": report.get("fallback_sources", [])[:5],
        "formal_command": report.get("formal_command"),
        "fast_screen_command": report.get("fast_screen_command"),
        "next_actions": report.get("next_actions", [])[:4],
        "rules": report.get("rules", [])[:5],
    }


def _compact_friction_interval_sources(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("rows"):
        return {}
    rows = report.get("rows", [])
    failures = [row for row in rows if row.get("status") != "pass"]
    sources = report.get("sources", {})
    return {
        "verdict": report.get("verdict"),
        "num_sources": len(sources) if isinstance(sources, dict) else None,
        "num_anchors": len(rows),
        "num_failures": len(failures),
        "task_framing": report.get("task_framing"),
        "policy": report.get("policy"),
        "source_keys": list(sources.keys())[:8] if isinstance(sources, dict) else [],
        "failures": failures[:5],
    }


def _compact_live_training_diagnosis(
    report: dict[str, Any],
    *,
    current_run: str | None = None,
    live_training: dict[str, Any] | None = None,
    active_watch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    watch_active = (active_watch or {}).get("active") or {}
    watch_latest = (active_watch or {}).get("latest_completed_epoch") or {}
    watch_previous = (active_watch or {}).get("previous_completed_epoch") or {}
    watch_matches = bool(
        watch_active.get("name")
        and (not current_run or str(watch_active.get("name")) == str(current_run))
    )
    if (not isinstance(report, dict) or not report.get("signals")) and not watch_matches:
        return {}
    if current_run and report.get("run") and str(report.get("run")) != str(current_run):
        if not watch_matches:
            return {}
    latest = report.get("latest_epoch") or {}
    best = report.get("best_val_loss_epoch") or {}
    signals = report.get("signals") or {}
    out = {
        "run": report.get("run"),
        "active_progress": report.get("active_progress"),
        "latest_epoch": latest.get("epoch"),
        "best_epoch": best.get("epoch"),
        "latest_val_loss": latest.get("val_loss"),
        "best_val_loss": best.get("val_loss"),
        "val_loss_delta_vs_best": signals.get("val_loss_delta_vs_best"),
        "raw_coverage_delta_vs_first": signals.get("raw_coverage_delta_vs_first"),
        "validation_degradation_flag": signals.get("validation_degradation_flag"),
        "coverage_degradation_flag": signals.get("coverage_degradation_flag"),
        "recommendation": report.get("recommendation", [])[:5],
    }
    if watch_matches and watch_latest:
        previous_loss = _as_float(watch_previous.get("val_loss"))
        latest_loss = _as_float(watch_latest.get("val_loss"))
        best_loss = _as_float(out.get("best_val_loss"))
        previous_cov = _as_float(watch_previous.get("val_mu_interval_coverage"))
        latest_cov = _as_float(watch_latest.get("val_mu_interval_coverage"))
        out.update(
            {
                "run": watch_active.get("name") or out.get("run") or current_run,
                "active_progress": {
                    "phase": watch_active.get("phase"),
                    "epoch": watch_active.get("epoch"),
                    "epochs": watch_active.get("epochs"),
                    "step": watch_active.get("step"),
                    "steps": watch_active.get("steps"),
                },
                "latest_epoch": watch_latest.get("epoch"),
                "latest_val_loss": latest_loss,
                "latest_val_risk_acc": watch_latest.get("val_acc_risk"),
                "latest_val_friction_acc": watch_latest.get("val_acc_friction"),
                "latest_raw_coverage": latest_cov,
                "raw_coverage_delta_vs_previous": _diff_float(latest_cov, previous_cov),
            }
        )
        if best_loss is not None and latest_loss is not None:
            out["val_loss_delta_vs_best"] = latest_loss - best_loss
            out["validation_degradation_flag"] = latest_loss > best_loss + 0.02
        else:
            live_loss = _as_float((live_training or {}).get("latest_val_loss"))
            if live_loss is not None and latest_loss is not None:
                out["val_loss_delta_vs_best"] = _diff_float(latest_loss, live_loss)
        if previous_cov is not None and latest_cov is not None:
            out["coverage_degradation_flag"] = latest_cov < previous_cov - 0.03
            out["raw_coverage_delta_vs_first"] = latest_cov - previous_cov
        if out.get("best_epoch") is not None and out.get("latest_epoch") != out.get("best_epoch"):
            recs = list(out.get("recommendation") or [])
            recs.insert(
                0,
                f"Latest active-watch epoch is {out.get('latest_epoch')}; keep using the selected best checkpoint epoch {out.get('best_epoch')} unless the final protocol changes.",
            )
            out["recommendation"] = recs[:5]
    return out


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _diff_float(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _compact_active_training_watch(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("verdict"):
        return {}
    active = report.get("active") or {}
    latest = report.get("latest_completed_epoch") or {}
    checkpoints = report.get("checkpoints") or {}
    return {
        "verdict": report.get("verdict"),
        "warnings": report.get("warnings", [])[:5],
        "active": active,
        "latest_completed_epoch": latest,
        "recent_train_steps": report.get("recent_train_steps", [])[-6:],
        "recent_loss_slope": report.get("recent_loss_slope"),
        "recent_loss_scope": report.get("recent_loss_scope"),
        "checkpoints": checkpoints,
        "num_active_error_markers": len(report.get("active_errors", []) or []),
    }


def _compact_config_batch_check(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("checks"):
        return {}
    failures = report.get("failures", [])
    return {
        "config_count": report.get("config_count"),
        "split_count": report.get("split_count"),
        "checks": report.get("checks"),
        "num_failures": len(failures),
        "failures": failures[:5],
    }


def _compact_forward_loss_smoke(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("checks"):
        return {}
    failures = report.get("failures", [])
    return {
        "checks": report.get("checks"),
        "config_count": report.get("config_count"),
        "num_failures": len(failures),
        "rows": report.get("rows", [])[:8],
        "note": report.get("note"),
    }


def _compact_gpu_protocol_audit(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("verdict"):
        return {}
    rows = report.get("rows", [])
    failures = report.get("failures", [])
    warnings = report.get("warnings", [])
    torch_info = report.get("torch") or {}
    effective_batches = sorted(
        {
            row.get("effective_batch")
            for row in rows
            if isinstance(row, dict) and row.get("effective_batch") is not None
        }
    )
    batch_sizes = sorted(
        {
            row.get("batch_size")
            for row in rows
            if isinstance(row, dict) and row.get("batch_size") is not None
        }
    )
    image_sizes = sorted(
        {
            row.get("image_size")
            for row in rows
            if isinstance(row, dict) and row.get("image_size") is not None
        }
    )
    return {
        "verdict": report.get("verdict"),
        "num_configs": report.get("num_configs"),
        "num_failures": len(failures),
        "num_warnings": len(warnings),
        "queue_python_exists": report.get("queue_python_exists"),
        "queue_python": report.get("queue_python"),
        "torch_version": torch_info.get("version"),
        "cuda_available": torch_info.get("cuda_available"),
        "cuda_build": torch_info.get("cuda_build"),
        "device_name": torch_info.get("device_name"),
        "effective_batches": effective_batches,
        "batch_sizes": batch_sizes,
        "image_sizes": image_sizes,
        "amp_rows": sum(1 for row in rows if isinstance(row, dict) and row.get("amp") is True),
        "failures": failures[:5],
        "warnings": warnings[:5],
    }


def _compact_handoff_health(
    report: dict[str, Any],
    active_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("verdict"):
        return {}
    latest_tail = (report.get("latest_handoff_log") or {}).get("tail", [])[-3:]
    next_incomplete = _merge_next_incomplete_with_active(report.get("next_incomplete"), active_rows)
    if report.get("verdict") == "handoff_no_longer_needed_roadsaw_complete" and active_rows:
        next_incomplete = active_rows[0]
    merged_active_rows = _merge_handoff_active_rows(report.get("active_rows", []), active_rows)
    roadsaw_delayed = report.get("roadsaw_delayed_by_active_queue")
    if report.get("verdict") == "handoff_no_longer_needed_roadsaw_complete":
        roadsaw_delayed = False
    return {
        "verdict": report.get("verdict"),
        "roadsaw_priority_order": report.get("roadsaw_priority_order"),
        "roadsaw_delayed_by_active_queue": roadsaw_delayed,
        "next_incomplete": _compact_next_incomplete_text(next_incomplete),
        "active_rows": [
            _compact_next_incomplete_text(row)
            for row in merged_active_rows[:3]
            if isinstance(row, dict)
        ],
        "v5_complete": (report.get("v5_full_faf") or {}).get("complete"),
        "roadsaw_complete": (report.get("lodo_roadsaw_full_faf") or {}).get("complete"),
        "handoff_watchers": len(report.get("handoff_processes", [])),
        "priority_watchers": len(report.get("priority_watcher_processes", [])),
        "priority_fast_screen_followup": report.get("priority_fast_screen_followup"),
        "queue_processes": len(report.get("queue_processes", [])),
        "train_processes": len(report.get("train_processes", [])),
        "latest_log": (report.get("latest_handoff_log") or {}).get("path"),
        "latest_log_tail": [str(item).replace("\ufeff", "").strip() for item in latest_tail],
    }


def _merge_next_incomplete_with_active(
    item: Any,
    active_rows: list[dict[str, Any]] | None,
) -> Any:
    if not isinstance(item, dict):
        return item
    name = item.get("name") or item.get("run")
    if not name:
        return item
    for row in active_rows or []:
        if row.get("name") != name:
            continue
        merged = dict(item)
        for key in ("active_epoch", "active_epochs", "active_step", "active_steps", "active_phase", "epoch", "epochs"):
            if row.get(key) is not None:
                merged[key] = row.get(key)
        return merged
    return item


def _merge_handoff_active_rows(
    report_rows: Any,
    active_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    stale_rows = [row for row in report_rows if isinstance(row, dict)]
    fresh_by_name = {
        row.get("name"): row
        for row in active_rows or []
        if isinstance(row, dict) and row.get("name")
    }
    merged: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for row in stale_rows:
        name = row.get("name") or row.get("run")
        fresh = fresh_by_name.get(name)
        merged.append(_newer_progress_row(fresh, row) if fresh else row)
        seen.add(name)
    for name, row in fresh_by_name.items():
        if name not in seen:
            merged.append(row)
    return merged


def _compact_runtime_guard(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("verdict"):
        return {}
    checks = report.get("checks", [])
    return {
        "verdict": report.get("verdict"),
        "num_blocks": sum(1 for item in checks if item.get("level") == "block"),
        "num_warnings": sum(1 for item in checks if item.get("level") == "warn"),
        "checks": checks,
        "active": report.get("active") or {},
        "log_health": report.get("log_health") or {},
        "recent_log_errors": report.get("recent_log_errors", [])[:5],
    }


def _compact_next_incomplete_text(item: Any) -> str:
    if not isinstance(item, dict) or not item:
        return "-"
    name = item.get("name") or item.get("run") or "-"
    status = item.get("status") or "-"
    epoch = item.get("active_epoch")
    epochs = item.get("active_epochs")
    step = item.get("active_step")
    steps = item.get("active_steps")
    if epoch and epochs and step and steps:
        return f"{name} {status} epoch {epoch}/{epochs} step {step}/{steps}"
    return f"{name} {status}"


def _gpu_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"available": False, "error": proc.stderr.strip()}
    first = [part.strip() for part in proc.stdout.strip().splitlines()[0].split(",")]
    keys = ["name", "temperature_c", "utilization_percent", "memory_used_mb", "memory_total_mb", "power_w"]
    return {"available": True, **dict(zip(keys, first))}


def _tqdm_snapshot(log_dir: Path, run_name: str) -> dict[str, Any] | None:
    if not log_dir.exists():
        return None
    candidates = sorted(
        log_dir.glob(f"*{run_name}*.err.log"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    path = candidates[0]
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in reversed(lines[-500:]):
        match = TQDM_RE.search(line)
        if not match:
            continue
        out: dict[str, Any] = {
            "log": str(path),
            "phase": match.group("phase"),
            "percent": int(match.group("pct")),
            "step": int(match.group("step")),
            "steps": int(match.group("steps")),
            "elapsed": match.group("elapsed").strip(),
            "eta": match.group("eta").strip(),
            "rate": match.group("rate").strip(),
        }
        return out
    return None


def _disk_snapshot(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            rows.append({"path": path, "available": False, "error": str(exc)})
            continue
        rows.append(
            {
                "path": path,
                "available": True,
                "free_gb": round(usage.free / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "total_gb": round(usage.total / (1024**3), 2),
            }
        )
    return rows


def _next_milestones(completeness: dict[str, Any], gate: dict[str, Any]) -> list[str]:
    requirements = {item.get("name"): item for item in completeness.get("requirements", [])}
    gates = {item.get("name"): item for item in gate.get("gates", [])}
    milestones: list[str] = []
    if requirements.get("p0_ablation_complete", {}).get("status") != "complete":
        milestones.append("Finish v5_full_faf training and postprocess test/calibration/bootstrap artifacts.")
    if requirements.get("lodo_complete", {}).get("status") != "complete":
        milestones.append("Finish the remaining LODO row and analyze the completed held-out RoadSaW failure before making OOD claims.")
    if gates.get("dataset_shortcut", {}).get("level") in {"warn", "block"}:
        milestones.append("Use P1 candidates to reduce dataset shortcut evidence.")
    if requirements.get("fair_single_dataset_complete", {}).get("status") != "complete":
        milestones.append("Run single-dataset FAF and matched ConvNeXt baselines.")
    if requirements.get("final_method_complete", {}).get("status") != "complete":
        milestones.append("Run final lean road-ROI safety LODO and single-dataset comparisons.")
    return milestones


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Experiment Status Dashboard",
        "",
        f"Generated at: {report['generated_at']}",
        f"Root: `{report['root']}`",
        "",
        "## Current State",
        "",
        f"- Readiness verdict: `{report['readiness'].get('verdict')}` "
        f"({report['readiness'].get('num_blocks')} blocks, {report['readiness'].get('num_warnings')} warnings).",
        f"- Progress counts: `{json.dumps(report['progress_counts'], ensure_ascii=False, sort_keys=True)}`.",
    ]
    active = report.get("active_rows", [])
    if active:
        active_tqdm = report.get("active_tqdm", {})
        lines.append(
            "- Active rows: "
            + ", ".join(_active_row_text(row, active_tqdm.get(row.get("name"))) for row in active)
            + "."
        )
    else:
        lines.append("- Active rows: none.")

    gpu = report.get("system", {}).get("gpu", {})
    if gpu.get("available"):
        lines.append(
            "- GPU: {name}, {util}% util, {used}/{total} MB, {temp} C.".format(
                name=gpu.get("name"),
                util=gpu.get("utilization_percent"),
                used=gpu.get("memory_used_mb"),
                total=gpu.get("memory_total_mb"),
                temp=gpu.get("temperature_c"),
            )
        )
    batch_check = report.get("config_batch_check", {})
    if batch_check:
        lines.append(
            "- Config batch precheck: `{checks}` checks across `{configs}` configs, `{failures}` failures.".format(
                checks=batch_check.get("checks"),
                configs=batch_check.get("config_count"),
                failures=batch_check.get("num_failures"),
            )
        )
    forward_check = report.get("config_forward_loss_smoke", {})
    if forward_check:
        lines.append(
            "- Forward/loss smoke precheck: `{checks}` high-risk configs, `{failures}` failures.".format(
                checks=forward_check.get("checks"),
                failures=forward_check.get("num_failures"),
            )
        )
    gpu_protocol = report.get("gpu_protocol_audit", {})
    if gpu_protocol:
        lines.append(
            "- GPU protocol audit: `{verdict}`, `{configs}` configs, CUDA `{cuda}`, AMP rows `{amp}`.".format(
                verdict=gpu_protocol.get("verdict"),
                configs=gpu_protocol.get("num_configs"),
                cuda=gpu_protocol.get("cuda_available"),
                amp=gpu_protocol.get("amp_rows"),
            )
        )
    handoff = report.get("handoff_health", {})
    if handoff:
        lines.append(
            "- RoadSaW handoff: `{verdict}`, priority `{priority}`, watchers `{watchers}`.".format(
                verdict=handoff.get("verdict"),
                priority=handoff.get("roadsaw_priority_order"),
                watchers=handoff.get("handoff_watchers"),
            )
        )
        if handoff.get("priority_watchers") is not None:
            lines.append(f"- RoadSaW priority watcher processes: `{handoff.get('priority_watchers')}`.")
        fast_followup_count = (
            ((report.get("queue_recovery") or {}).get("process_snapshot") or {})
            .get("counts", {})
            .get("fast_screen_followup", 0)
        )
        if handoff.get("priority_fast_screen_followup") is not None and not (
            handoff.get("priority_fast_screen_followup") is False and fast_followup_count
        ):
            lines.append(f"- RoadSaW fast-screen + promotion follow-up enabled: `{handoff.get('priority_fast_screen_followup')}`.")
        if handoff.get("roadsaw_delayed_by_active_queue"):
            active_text = ", ".join(str(item) for item in handoff.get("active_rows", []) if item) or "-"
            lines.append(f"- RoadSaW is priority in the refreshed queue, but the active older queue is running: {active_text}.")
    runtime_guard = report.get("runtime_guard", {})
    if runtime_guard:
        lines.append(
            "- Runtime guard: `{verdict}` ({blocks} blocks, {warnings} warnings).".format(
                verdict=runtime_guard.get("verdict"),
                blocks=runtime_guard.get("num_blocks"),
                warnings=runtime_guard.get("num_warnings"),
            )
        )
    watch = report.get("active_training_watch", {})
    if watch:
        checkpoints = watch.get("checkpoints") or {}
        active_watch = watch.get("active") or {}
        loss_scope = _loss_scope_label(watch.get("recent_loss_scope"))
        lines.append(
            "- Active training watch: `{verdict}` for `{run}`, recent loss slope ({scope}) `{slope}`, checkpoints best `{best}`/safety `{safety}`.".format(
                verdict=watch.get("verdict"),
                run=active_watch.get("name"),
                scope=loss_scope,
                slope=_fmt_signed_abs(watch.get("recent_loss_slope")),
                best=checkpoints.get("best"),
                safety=checkpoints.get("best_safety"),
            )
        )
    lodo = report.get("lodo_generalization", {})
    if lodo:
        lines.append(
            "- RoadSaW LODO readout: `{verdict}` (`{status}`).".format(
                verdict=lodo.get("roadsaw_verdict"),
                status=lodo.get("roadsaw_status"),
            )
        )
    roadsaw_protocol = report.get("roadsaw_lodo_protocol", {})
    if roadsaw_protocol:
        splits = roadsaw_protocol.get("splits", {})
        train = splits.get("train", {})
        test = splits.get("test", {})
        lines.append(
            "- RoadSaW LODO protocol audit: `{verdict}`; train datasets `{train}`, test datasets `{test}`.".format(
                verdict=roadsaw_protocol.get("verdict"),
                train=_dict_text(train.get("datasets", {})),
                test=_dict_text(test.get("datasets", {})),
            )
        )
    fair_protocol = report.get("fair_comparison_protocol", {})
    if fair_protocol:
        lines.append(
            "- Fair comparison protocol audit: `{verdict}`; `{strict}` strict single-dataset pairs pass, `{final}` final pairs documented.".format(
                verdict=fair_protocol.get("verdict"),
                strict=fair_protocol.get("strict_single_pairs_pass"),
                final=fair_protocol.get("final_pairs_documented"),
            )
        )
    friction_sources = report.get("friction_interval_sources", {})
    if friction_sources:
        lines.append(
            "- Friction interval source audit: `{verdict}` with `{anchors}` public anchors from `{sources}` source groups.".format(
                verdict=friction_sources.get("verdict"),
                anchors=friction_sources.get("num_anchors"),
                sources=friction_sources.get("num_sources"),
            )
        )
    image_style = report.get("dataset_image_style", {})
    if image_style:
        lines.append(
            "- Dataset image style audit: resize `{resize}`, aspect span `{aspect}`, width span `{width}`, brightness span `{brightness}`.".format(
                resize=image_style.get("resize_mode"),
                aspect=_fmt_abs(image_style.get("aspect_span")),
                width=_fmt_abs(image_style.get("width_span")),
                brightness=_fmt_abs(image_style.get("brightness_span")),
            )
        )
    input_style = report.get("input_canonicalization_style", {})
    if input_style:
        lines.append(
            "- Input canonicalization style audit: best `{run}`, style-gap `{score}`, relative to `{base}` `{relative}`.".format(
                run=input_style.get("best_run"),
                score=_fmt_abs(input_style.get("best_style_gap_score")),
                base=input_style.get("baseline_run"),
                relative=_fmt_pct(input_style.get("best_style_gap_relative")),
            )
        )
    trace = report.get("config_to_code_trace", {})
    if trace:
        lines.append(
            "- Config-to-code trace: `{verdict}` across `{rows}` innovation modules, `{blocks}` blocks.".format(
                verdict=trace.get("verdict"),
                rows=trace.get("num_rows"),
                blocks=trace.get("num_blocks"),
            )
        )
    goal = report.get("goal_evidence", {})
    if goal:
        lines.append(
            "- Goal evidence audit: current `{run}`, `{incomplete}/{total}` hard requirements incomplete.".format(
                run=goal.get("current_run"),
                incomplete=goal.get("num_incomplete_requirements"),
                total=goal.get("num_requirements"),
            )
        )
    artifact_contract = report.get("artifact_contract", {})
    if artifact_contract:
        lines.append(
            "- Artifact contract: `{verdict}`, `{complete}/{total}` runs evidence-complete, stale rows `{stale}`.".format(
                verdict=artifact_contract.get("verdict"),
                complete=artifact_contract.get("num_contract_complete"),
                total=artifact_contract.get("num_runs"),
                stale=artifact_contract.get("num_stale_rows"),
            )
        )
    claim_evidence = report.get("claim_evidence", {})
    if claim_evidence:
        counts = claim_evidence.get("status_counts", {})
        lines.append(
            "- Claim evidence: supported `{supported}`, partial `{partial}`, not supported `{hard_not_supported}`, not yet supported `{not_supported}`.".format(
                supported=counts.get("supported", 0),
                partial=counts.get("partial", 0),
                hard_not_supported=counts.get("not_supported", 0),
                not_supported=counts.get("not_supported_yet", 0),
            )
        )
    reviewer_checklist = report.get("reviewer_evidence_checklist", {})
    if reviewer_checklist:
        counts = reviewer_checklist.get("status_counts", {})
        lines.append(
            "- Reviewer evidence checklist: supported `{supported}`, partial `{partial}`, not supported `{hard_not_supported}`, not yet supported `{not_supported}`.".format(
                supported=counts.get("supported", 0),
                partial=counts.get("partial", 0),
                hard_not_supported=counts.get("not_supported", 0),
                not_supported=counts.get("not_supported_yet", 0),
            )
        )
    hypothesis = report.get("candidate_hypothesis_matrix", {})
    if hypothesis:
        status_counts = hypothesis.get("contract_status_counts", {})
        lines.append(
            "- Candidate hypothesis matrix: `{verdict}`, `{rows}` rows, contract statuses `{statuses}`.".format(
                verdict=hypothesis.get("verdict"),
                rows=hypothesis.get("num_rows"),
                statuses=json.dumps(status_counts, ensure_ascii=False, sort_keys=True),
            )
        )
    paper_p0 = report.get("paper_p0_ablation_table", {})
    if paper_p0:
        best = paper_p0.get("best_by_metric", {})
        lines.append(
            "- Paper P0 table: `{status}`, best friction `{friction}`, best worst-dataset `{worst}`.".format(
                status=paper_p0.get("status"),
                friction=(best.get("friction_macro_f1") or {}).get("method"),
                worst=(best.get("worst_dataset_f1") or {}).get("method"),
            )
        )
    fast_screen = report.get("fast_screen", {})
    if fast_screen:
        lines.append(
            "- Fast-screen protocol: `{verdict}`, counts `{counts}`.".format(
                verdict=fast_screen.get("verdict"),
                counts=json.dumps(fast_screen.get("counts", {}), ensure_ascii=False, sort_keys=True),
            )
        )
    fast_to_formal = report.get("fast_to_formal", {})
    if fast_to_formal:
        lines.append(
            "- Fast-to-formal promotion: `{verdict}`, promoted `{count}`.".format(
                verdict=fast_to_formal.get("verdict"),
                count=len(fast_to_formal.get("promoted") or []),
            )
        )
    lines.append("")

    fast_screen = report.get("fast_screen", {})
    if fast_screen:
        lines.extend(["## Fast-Screen Protocol", ""])
        lines.append(f"- Verdict: `{fast_screen.get('verdict')}`.")
        lines.append(f"- Counts: `{json.dumps(fast_screen.get('counts', {}), ensure_ascii=False, sort_keys=True)}`.")
        if fast_screen.get("claim_boundary"):
            lines.append(f"- Claim boundary: {fast_screen.get('claim_boundary')}")
        top = fast_screen.get("top_complete") or []
        if top:
            lines.append("| Run | Source | Score | risk F1 | low recall | cal cov | width | dataset-ID |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
            for row in top:
                lines.append(
                    "| {run} | {source} | {score} | {risk} | {low} | {cov} | {width} | {shortcut} |".format(
                        run=row.get("run"),
                        source=row.get("source_run"),
                        score=_fmt_abs(row.get("screen_score")),
                        risk=_fmt_pct(row.get("risk_f1")),
                        low=_fmt_pct(row.get("low_friction_recall")),
                        cov=_fmt_pct(row.get("calibrated_coverage")),
                        width=_fmt_abs(row.get("calibrated_width")),
                        shortcut=_fmt_pct(row.get("dataset_id_bal_acc")),
                    )
                )
        actions = fast_screen.get("next_actions") or []
        if actions:
            lines.append("Next fast-screen actions:")
            for item in actions:
                lines.append(f"- {item}")
        lines.append("")

    fast_to_formal = report.get("fast_to_formal", {})
    if fast_to_formal:
        lines.extend(["## Fast-To-Formal Promotion", ""])
        lines.append(f"- Verdict: `{fast_to_formal.get('verdict')}`.")
        promoted = fast_to_formal.get("promoted") or []
        fallback = fast_to_formal.get("fallback_sources") or []
        if promoted:
            lines.append("Promoted candidates:")
            for row in promoted:
                lines.append(f"- `{row.get('source_run')}` score `{_fmt_abs(row.get('promotion_score'))}`")
        elif fallback:
            lines.append("Fallback formal candidates: " + ", ".join(f"`{item}`" for item in fallback) + ".")
        command = fast_to_formal.get("formal_command")
        if command:
            lines.append(f"- Formal command: `{command}`")
        actions = fast_to_formal.get("next_actions") or []
        if actions:
            lines.append("Next promotion actions:")
            for item in actions:
                lines.append(f"- {item}")
        lines.append("")

    trend = report.get("live_training", {})
    if trend:
        lines.extend(["## Live Training", ""])
        active_progress = trend.get("active_progress") or {}
        lines.append(
            "- `{run}` active: epoch {epoch}/{epochs}, step {step}/{steps}.".format(
                run=trend.get("run"),
                epoch=active_progress.get("epoch", "-"),
                epochs=active_progress.get("epochs", "-"),
                step=active_progress.get("step", "-"),
                steps=active_progress.get("steps", "-"),
            )
        )
        lines.append(
            "- Latest completed epoch {epoch}: val loss `{loss}`, risk acc `{risk}`, raw coverage `{cov}`, raw width `{width}`.".format(
                epoch=trend.get("latest_completed_epoch"),
                loss=_fmt_abs(trend.get("latest_val_loss")),
                risk=_fmt_pct(trend.get("latest_val_risk_acc")),
                cov=_fmt_pct(trend.get("latest_raw_coverage")),
                width=_fmt_abs(trend.get("latest_raw_width")),
            )
        )
        lines.append(
            "- Previous delta: val loss `{loss}`, risk acc `{risk}`, raw coverage `{cov}`.".format(
                loss=_fmt_signed_abs(trend.get("previous_delta_val_loss")),
                risk=_fmt_signed_pct(trend.get("previous_delta_val_risk_acc")),
                cov=_fmt_signed_pct(trend.get("previous_delta_raw_coverage")),
            )
        )
        lines.append("")

    watch = report.get("active_training_watch", {})
    if watch:
        lines.extend(["## Active Training Watch", ""])
        active_watch = watch.get("active") or {}
        latest = watch.get("latest_completed_epoch") or {}
        lines.append(
            "- `{run}` phase `{phase}`, epoch `{epoch}/{epochs}`, step `{step}/{steps}`, ETA `{eta}`, rate `{rate}`.".format(
                run=active_watch.get("name"),
                phase=active_watch.get("phase") or "-",
                epoch=active_watch.get("epoch") or "-",
                epochs=active_watch.get("epochs") or "-",
                step=active_watch.get("step") or "-",
                steps=active_watch.get("steps") or "-",
                eta=active_watch.get("eta") or "-",
                rate=active_watch.get("rate") or "-",
            )
        )
        if latest:
            lines.append(
                "- Watch latest completed epoch `{epoch}`: val loss `{loss}`, risk acc `{risk}`, friction acc `{friction}`, raw coverage `{coverage}`.".format(
                    epoch=latest.get("epoch"),
                    loss=_fmt_abs(latest.get("val_loss")),
                    risk=_fmt_pct(latest.get("val_acc_risk")),
                    friction=_fmt_pct(latest.get("val_acc_friction")),
                    coverage=_fmt_pct(latest.get("val_mu_interval_coverage")),
                )
            )
        recent = watch.get("recent_train_steps") or []
        if recent:
            lines.append("| Epoch | Step | Loss |")
            lines.append("|---:|---:|---:|")
            for row in recent:
                lines.append(
                    "| {epoch} | {step}/{steps} | {loss} |".format(
                        epoch=row.get("epoch"),
                        step=row.get("step"),
                        steps=row.get("steps"),
                        loss=_fmt_abs(row.get("loss")),
                    )
                )
            loss_scope = _loss_scope_label(watch.get("recent_loss_scope"))
            lines.append(
                f"- Recent displayed loss slope ({loss_scope}): `{_fmt_signed_abs(watch.get('recent_loss_slope'))}`."
            )
        if watch.get("warnings"):
            for item in watch.get("warnings", [])[:3]:
                lines.append(f"- Watch warning: {item}")
        if watch.get("num_active_error_markers"):
            lines.append(f"- Active error markers: `{watch.get('num_active_error_markers')}`.")
        lines.append("")

    diagnosis = report.get("live_training_diagnosis", {})
    if diagnosis:
        lines.extend(["## Live Training Diagnosis", ""])
        lines.append(
            "- `{run}` best epoch `{best}` vs latest completed epoch `{latest}`; val-loss delta vs best `{delta}`.".format(
                run=diagnosis.get("run"),
                best=diagnosis.get("best_epoch"),
                latest=diagnosis.get("latest_epoch"),
                delta=_fmt_signed_abs(diagnosis.get("val_loss_delta_vs_best")),
            )
        )
        lines.append(
            "- Flags: validation degradation `{val_flag}`, coverage degradation `{cov_flag}`, raw coverage delta vs {cov_ref} `{cov_delta}`.".format(
                val_flag=diagnosis.get("validation_degradation_flag"),
                cov_flag=diagnosis.get("coverage_degradation_flag"),
                cov_ref="previous" if diagnosis.get("raw_coverage_delta_vs_previous") is not None else "first",
                cov_delta=_fmt_signed_pct(
                    diagnosis.get("raw_coverage_delta_vs_previous")
                    if diagnosis.get("raw_coverage_delta_vs_previous") is not None
                    else diagnosis.get("raw_coverage_delta_vs_first")
                ),
            )
        )
        for item in diagnosis.get("recommendation", [])[:3]:
            lines.append(f"- {item}")
        lines.append("")

    handoff = report.get("handoff_health", {})
    if handoff:
        lines.extend(["## RoadSaW Handoff", ""])
        lines.append(
            "- Verdict `{verdict}`; priority order `{priority}`; next incomplete `{next}`.".format(
                verdict=handoff.get("verdict"),
                priority=handoff.get("roadsaw_priority_order"),
                next=handoff.get("next_incomplete"),
            )
        )
        if handoff.get("roadsaw_delayed_by_active_queue"):
            active_text = ", ".join(str(item) for item in handoff.get("active_rows", []) if item) or "-"
            lines.append(f"- RoadSaW delayed by active older queue: {active_text}.")
        lines.append(
            "- Processes: handoff watchers `{watchers}`, queue `{queue}`, train `{train}`.".format(
                watchers=handoff.get("handoff_watchers"),
                queue=handoff.get("queue_processes"),
                train=handoff.get("train_processes"),
            )
        )
        if handoff.get("priority_watchers") is not None:
            lines.append(f"- RoadSaW priority watcher processes: `{handoff.get('priority_watchers')}`.")
        fast_followup_count = (
            ((report.get("queue_recovery") or {}).get("process_snapshot") or {})
            .get("counts", {})
            .get("fast_screen_followup", 0)
        )
        if handoff.get("priority_fast_screen_followup") is not None and not (
            handoff.get("priority_fast_screen_followup") is False and fast_followup_count
        ):
            lines.append(f"- Fast-screen + promotion follow-up enabled: `{handoff.get('priority_fast_screen_followup')}`.")
        lines.append(
            "- Artifacts: v5 complete `{v5}`, held-out RoadSaW complete `{roadsaw}`.".format(
                v5=handoff.get("v5_complete"),
                roadsaw=handoff.get("roadsaw_complete"),
            )
        )
        for item in handoff.get("latest_log_tail", []):
            lines.append(f"- Handoff log: `{item}`")
        lines.append("")

    roadsaw_protocol = report.get("roadsaw_lodo_protocol", {})
    if roadsaw_protocol:
        lines.extend(["## RoadSaW LODO Protocol", ""])
        lines.append(
            "- Verdict `{verdict}` with `{blocks}` blocks and `{warnings}` warnings.".format(
                verdict=roadsaw_protocol.get("verdict"),
                blocks=roadsaw_protocol.get("num_blocks"),
                warnings=roadsaw_protocol.get("num_warnings"),
            )
        )
        lines.append(f"- Config: `{roadsaw_protocol.get('config')}`.")
        for split, summary in (roadsaw_protocol.get("splits") or {}).items():
            lines.append(
                "- `{split}`: rows `{rows}`, datasets `{datasets}`, split values `{split_values}`, mu range `{low}-{high}`.".format(
                    split=split,
                    rows=summary.get("rows"),
                    datasets=_dict_text(summary.get("datasets", {})),
                    split_values=_dict_text(summary.get("split_values", {})),
                    low=_fmt_abs(summary.get("mu_low")),
                    high=_fmt_abs(summary.get("mu_high")),
                )
            )
        for item in roadsaw_protocol.get("policy", [])[:2]:
            lines.append(f"- Policy: {item}")
        lines.append("")

    fair_protocol = report.get("fair_comparison_protocol", {})
    if fair_protocol:
        lines.extend(["## Fair Comparison Protocol", ""])
        lines.append(
            "- Verdict `{verdict}` across `{pairs}` FAF/ConvNeXt pairs with `{blocks}` blocks and `{warnings}` documented warnings.".format(
                verdict=fair_protocol.get("verdict"),
                pairs=fair_protocol.get("num_pairs"),
                blocks=fair_protocol.get("num_blocks"),
                warnings=fair_protocol.get("num_warnings"),
            )
        )
        lines.append(
            "- Strict full-FAF single-dataset pairs passing: `{count}/3`; final-method pairs documented: `{final}/3`.".format(
                count=fair_protocol.get("strict_single_pairs_pass"),
                final=fair_protocol.get("final_pairs_documented"),
            )
        )
        lines.append("| Scope | Dataset | Pair | Status | Effective batch | Rows train/val/test |")
        lines.append("|---|---|---|---|---:|---:|")
        for row in fair_protocol.get("rows", []):
            manifest = row.get("manifest_summary", {})
            counts = "{}/{}/{}".format(
                (manifest.get("train") or {}).get("num_rows", "-"),
                (manifest.get("val") or {}).get("num_rows", "-"),
                (manifest.get("test") or {}).get("num_rows", "-"),
            )
            lines.append(
                "| {scope} | {dataset} | `{pair}` | `{status}` | {batch} | {counts} |".format(
                    scope=row.get("scope"),
                    dataset=row.get("dataset"),
                    pair=row.get("pair"),
                    status=row.get("status"),
                    batch=row.get("effective_batch_faf"),
                    counts=counts,
                )
            )
        for item in fair_protocol.get("non_pass_checks", [])[:3]:
            lines.append(f"- `{item.get('level')}` `{item.get('name')}` on `{item.get('pair')}`: {item.get('message')}")
        for item in fair_protocol.get("policy", [])[:2]:
            lines.append(f"- Policy: {item}")
        lines.append("")

    runtime_guard = report.get("runtime_guard", {})
    if runtime_guard:
        lines.extend(["## Runtime Guard", ""])
        lines.append(
            "- Verdict `{verdict}` with `{blocks}` blocks and `{warnings}` warnings.".format(
                verdict=runtime_guard.get("verdict"),
                blocks=runtime_guard.get("num_blocks"),
                warnings=runtime_guard.get("num_warnings"),
            )
        )
        active = runtime_guard.get("active") or {}
        log_health = runtime_guard.get("log_health") or {}
        if active:
            lines.append(
                "- Active guard row `{name}` `{status}`; log age `{age}` seconds.".format(
                    name=active.get("name"),
                    status=active.get("status"),
                    age=_fmt_abs(log_health.get("age_seconds")),
                )
            )
        for item in runtime_guard.get("checks", []):
            if item.get("level") != "pass":
                lines.append(f"- `{item.get('level')}` `{item.get('name')}`: {item.get('message')}")
        for item in runtime_guard.get("recent_log_errors", [])[:3]:
            lines.append(f"- Recent error marker `{item.get('path')}`: `{item.get('line')}`")
        lines.append("")

    lodo = report.get("lodo_generalization", {})
    if lodo:
        lines.extend(["## LODO Readout", ""])
        lines.append(
            "- RoadSaW verdict `{verdict}`; status `{status}`; overall LODO verdict `{overall}`.".format(
                verdict=lodo.get("roadsaw_verdict"),
                status=lodo.get("roadsaw_status"),
                overall=lodo.get("verdict"),
            )
        )
        lines.append(f"- Claim boundary: {lodo.get('claim_boundary')}")
        lines.append(
            "- Held-out RoadSaW metrics: risk F1 `{risk}`, friction F1 `{friction}`, calibrated coverage `{coverage}`.".format(
                risk=_fmt_pct(lodo.get("roadsaw_risk_f1")),
                friction=_fmt_pct(lodo.get("roadsaw_friction_f1")),
                coverage=_fmt_pct(lodo.get("roadsaw_calibrated_coverage")),
            )
        )
        lines.append(
            "- Delta vs mixed-test reference: risk F1 `{risk}`, friction F1 `{friction}`.".format(
                risk=_fmt_signed_pct(lodo.get("delta_risk_f1_vs_mixed")),
                friction=_fmt_signed_pct(lodo.get("delta_friction_f1_vs_mixed")),
            )
        )
        prelim = lodo.get("preliminary_evaluate_test") or []
        for row in prelim:
            lines.append(
                "- Preliminary held-out `{held}` evaluate_test: risk acc `{risk}`, friction acc `{friction}`, raw coverage `{coverage}`, width `{width}`.".format(
                    held=row.get("held_out"),
                    risk=_fmt_pct(row.get("acc_risk")),
                    friction=_fmt_pct(row.get("acc_friction")),
                    coverage=_fmt_pct(row.get("raw_coverage")),
                    width=_fmt_abs(row.get("raw_width")),
                )
            )
        for item in lodo.get("next_actions", [])[:3]:
            lines.append(f"- {item}")
        lines.append("")

    lines.extend(["## Blocking Gates", ""])
    for gate in report["readiness"].get("blocking_gates", []):
        lines.append(f"- `{gate.get('name')}`: {gate.get('message')}")
    lines.append("")

    p0 = report.get("p0_claim", {})
    if p0:
        lines.extend(["## P0 Module Claims", ""])
        lines.append(f"- P0 claim status: `{p0.get('core_status')}`.")
        for item in p0.get("recommendations", []):
            lines.append(
                f"- `{item.get('module')}`: `{item.get('recommendation') or item.get('status')}`."
            )
        paper_p0 = report.get("paper_p0_ablation_table", {})
        if paper_p0:
            best = paper_p0.get("best_by_metric", {})
            artifacts = paper_p0.get("artifacts", {})
            lines.append("")
            lines.append(
                "- Paper-ready P0 table: `{status}`; Markdown `{md}`, CSV `{csv}`, LaTeX `{tex}`.".format(
                    status=paper_p0.get("status"),
                    md=artifacts.get("md"),
                    csv=artifacts.get("csv"),
                    tex=artifacts.get("tex"),
                )
            )
            for key, item in best.items():
                lines.append(
                    "- Best `{metric}` owner: `{method}` (`{value}`).".format(
                        metric=item.get("label") or key,
                        method=item.get("method"),
                        value=_fmt_pct(item.get("value")),
                    )
                )
        lines.append("")

    interval = report.get("interval_quality", {})
    if interval:
        lines.extend(["## Interval Watchlist", ""])
        lines.append(
            f"- Audited `{interval.get('num_runs')}` calibrated runs, `{interval.get('num_cells')}` conditional cells, "
            f"`{interval.get('num_watchlist_items')}` undercoverage watchlist items."
        )
        for item in interval.get("top_watchlist", []):
            coverage = item.get("raw_coverage") if item.get("reason") == "raw_undercoverage" else item.get("calibrated_coverage")
            lines.append(
                "- `{run}` {scope}:{group} {reason}, coverage `{coverage}`, n={n}.".format(
                    run=item.get("run"),
                    scope=item.get("scope"),
                    group=item.get("group_label"),
                    reason=item.get("reason"),
                    coverage=_fmt_pct(coverage),
                    n=item.get("num_samples"),
                )
            )
        lines.append("")

    wetness = report.get("wetness_state", {})
    if wetness:
        lines.extend(["## Wetness State Watchlist", ""])
        lines.append(
            f"- Audited `{wetness.get('num_complete')}` completed rows; `{wetness.get('num_watchlist')}` rows cross RoadSaW wetness watch thresholds."
        )
        latest = wetness.get("latest_complete") or {}
        if latest:
            lines.append(
                "- Latest complete `{run}`: RoadSaW wetness F1 `{macro}`, ordinal MAE `{mae}`, severe misorder `{severe}`.".format(
                    run=latest.get("run"),
                    macro=_fmt_pct(latest.get("roadsaw_wetness_macro_f1")),
                    mae=_fmt_abs(latest.get("roadsaw_ordinal_mae")),
                    severe=_fmt_pct(latest.get("roadsaw_severe_misorder_rate")),
                )
            )
        for item in wetness.get("watchlist", []):
            lines.append(
                "- `{run}`: RoadSaW wet F1 `{macro}`, ordinal MAE `{mae}`, severe misorder `{severe}`.".format(
                    run=item.get("run"),
                    macro=_fmt_pct(item.get("roadsaw_wetness_macro_f1")),
                    mae=_fmt_abs(item.get("roadsaw_ordinal_mae")),
                    severe=_fmt_pct(item.get("roadsaw_severe_misorder_rate")),
                )
            )
        lines.append("")

    shortcut = report.get("dataset_shortcut", {})
    if shortcut:
        lines.extend(["## Dataset Shortcut", ""])
        best = shortcut.get("best_completed") or {}
        lines.append(
            f"- Verdict `{shortcut.get('verdict')}`; `{shortcut.get('num_high_shortcut')}` of `{shortcut.get('num_complete')}` completed rows exceed the shortcut threshold."
        )
        lines.append(
            "- Best completed by core-state probe `{run}`: core-state bal acc `{core}`, overall bal acc `{overall}`.".format(
                run=best.get("run"),
                core=_fmt_pct(best.get("core_state_conditioned_balanced_accuracy")),
                overall=_fmt_pct(best.get("overall_balanced_accuracy")),
            )
        )
        for row in shortcut.get("top_rows", []):
            lines.append(
                "- `{run}` overall `{overall}`, risk-cond `{risk}`, core-state-cond `{core}`.".format(
                    run=row.get("run"),
                    overall=_fmt_pct(row.get("overall_balanced_accuracy")),
                    risk=_fmt_pct(row.get("risk_conditioned_balanced_accuracy")),
                    core=_fmt_pct(row.get("core_state_conditioned_balanced_accuracy")),
                )
            )
        lines.append("")

    image_style = report.get("dataset_image_style", {})
    if image_style:
        lines.extend(["## Dataset Image Style Audit", ""])
        lines.append(
            "- Input policy: image size `{size}`, resize mode `{resize}`; aspect span `{aspect}`, width span `{width}`, height span `{height}`, saturation span `{sat}`.".format(
                size=image_style.get("image_size"),
                resize=image_style.get("resize_mode"),
                aspect=_fmt_abs(image_style.get("aspect_span")),
                width=_fmt_abs(image_style.get("width_span")),
                height=_fmt_abs(image_style.get("height_span")),
                sat=_fmt_abs(image_style.get("saturation_span")),
            )
        )
        if image_style.get("claim_boundary"):
            lines.append(f"- Claim boundary: {image_style.get('claim_boundary')}")
        lines.append("| Dataset | samples | native median size | aspect | brightness | contrast | saturation | suffixes | modes |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
        for row in image_style.get("datasets", []):
            lines.append(
                "| {dataset} | {n} | {w}x{h} | {aspect} | {brightness} | {contrast} | {saturation} | {suffixes} | {modes} |".format(
                    dataset=row.get("dataset"),
                    n=row.get("num_samples"),
                    w=_fmt_abs(row.get("width_median")),
                    h=_fmt_abs(row.get("height_median")),
                    aspect=_fmt_abs(row.get("aspect_median")),
                    brightness=_fmt_abs(row.get("brightness_mean")),
                    contrast=_fmt_abs(row.get("contrast_mean")),
                    saturation=_fmt_abs(row.get("saturation_mean")),
                    suffixes=_dict_text(row.get("suffixes")),
                    modes=_dict_text(row.get("modes")),
                )
            )
        if image_style.get("recommendations"):
            lines.append("")
            for item in image_style.get("recommendations", []):
                lines.append(f"- {item}")
        lines.append("")

    input_style = report.get("input_canonicalization_style", {})
    if input_style:
        lines.extend(["## Input Canonicalization Style Audit", ""])
        lines.append(
            "- Best deterministic style canonicalization: `{run}` with style-gap `{score}` (`{relative}` of `{base}`).".format(
                run=input_style.get("best_run"),
                score=_fmt_abs(input_style.get("best_style_gap_score")),
                relative=_fmt_pct(input_style.get("best_style_gap_relative")),
                base=input_style.get("baseline_run"),
            )
        )
        if input_style.get("claim_boundary"):
            lines.append(f"- Claim boundary: {input_style.get('claim_boundary')}")
        lines.append(
            "| Run | resize | GrayWorld | Fourier jitter | style gap | rel. to baseline | brightness span | saturation span | RGB max span | channel spread span |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in input_style.get("rows", []):
            lines.append(
                "| {run} | {resize} | {gray} | {fourier} | {score} | {relative} | {brightness} | {saturation} | {rgb} | {channel} |".format(
                    run=row.get("run"),
                    resize=row.get("resize_mode"),
                    gray=_fmt_abs(row.get("gray_world_alpha")),
                    fourier=_fmt_abs(row.get("fourier_low_freq_jitter_p")),
                    score=_fmt_abs(row.get("style_gap_score")),
                    relative=_fmt_pct(row.get("style_gap_relative")),
                    brightness=_fmt_abs(row.get("brightness_span")),
                    saturation=_fmt_abs(row.get("saturation_span")),
                    rgb=_fmt_abs(row.get("rgb_span_max")),
                    channel=_fmt_abs(row.get("channel_mean_spread_span")),
                )
            )
        if input_style.get("recommendations"):
            lines.append("")
            for item in input_style.get("recommendations", []):
                lines.append(f"- {item}")
        lines.append("")

    evidence = report.get("evidence_failure", {})
    if evidence:
        lines.extend(["## Evidence Failure Audit", ""])
        lines.append(
            f"- Audited `{evidence.get('num_evidence_runs')}` EvidenceField runs, "
            f"`{evidence.get('num_examples')}` candidate examples, "
            f"`{evidence.get('num_roadsaw_failure_examples')}` RoadSaW failure examples."
        )
        for row in evidence.get("runs", []):
            lines.append(
                "- `{run}` sampled risk acc `{risk}`, RoadSaW acc `{roadsaw}`, "
                "risk failures `{fail}`, low-friction failures `{lowfail}`.".format(
                    run=row.get("run"),
                    risk=_fmt_pct(row.get("risk_accuracy_sampled")),
                    roadsaw=_fmt_pct(row.get("roadsaw_risk_accuracy_sampled")),
                    fail=row.get("risk_failure_count"),
                    lowfail=row.get("low_friction_failure_count"),
                )
            )
        lines.append("")

    final_selection = report.get("final_method_selection", {})
    if final_selection:
        lines.extend(["## Final Method Selection", ""])
        lines.append(f"- Verdict: `{final_selection.get('verdict')}`.")
        lines.append("| Rank | Group | Method | Score | risk F1 | low recall | raw cov | width | dataset-ID |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
        for idx, row in enumerate(final_selection.get("top_completed", []), start=1):
            lines.append(
                "| {idx} | {group} | {method} | {score} | {risk} | {low} | {raw} | {width} | {domain} |".format(
                    idx=idx,
                    group=row.get("group"),
                    method=row.get("method"),
                    score=_fmt_abs(row.get("selection_score")),
                    risk=_fmt_pct(row.get("risk_f1")),
                    low=_fmt_pct(row.get("low_friction_recall")),
                    raw=_fmt_pct(row.get("raw_interval_coverage")),
                    width=_fmt_abs(row.get("calibrated_width")),
                    domain=_fmt_pct(row.get("dataset_id_balanced_accuracy")),
                )
            )
        lines.append("")
        lines.append("Risk register:")
        for item in final_selection.get("risk_register", [])[:5]:
            lines.append(f"- `{item.get('risk')}` `{item.get('level')}`: {item.get('action')}")
        lines.append("")
        lines.append("Selection actions:")
        for item in final_selection.get("recommended_action", [])[:5]:
            lines.append(f"- {item}")
        lines.append("")

    gap = report.get("gap_analysis", {})
    if gap:
        lines.extend(["## Algorithm Gap And Candidate Coverage", ""])
        lines.append(
            "- P0 status `{status}`; pending P0 rows: {pending}.".format(
                status=gap.get("p0_status"),
                pending=", ".join(f"`{item}`" for item in gap.get("pending_p0", [])) or "-",
            )
        )
        if gap.get("key_failures"):
            lines.append("Key failure signals:")
            for item in gap.get("key_failures", [])[:4]:
                lines.append(f"- `{item.get('issue')}`: {item.get('evidence')}")
        lines.append("")
        lines.append("| Phase | Status | Runs | Missing modules |")
        lines.append("|---|---|---|---|")
        for item in gap.get("candidate_implementation", []):
            lines.append(
                "| {phase} | `{status}` | {runs} | {missing} |".format(
                    phase=item.get("phase"),
                    status=item.get("status"),
                    runs=", ".join(f"`{run}`" for run in item.get("present_runs", [])) or "-",
                    missing=", ".join(f"`{module}`" for module in item.get("missing_modules", [])) or "-",
                )
            )
        lines.append("")

    safety = report.get("safety_selection", {})
    if safety:
        lines.extend(["## Safety Checkpoint Selection", ""])
        lines.append(
            "- Verdict `{verdict}`; complete rows `{complete}/{rows}`, helpful rows `{helpful}`.".format(
                verdict=safety.get("verdict"),
                complete=safety.get("num_complete"),
                rows=safety.get("num_rows"),
                helpful=safety.get("num_helpful"),
            )
        )
        if safety.get("rule"):
            lines.append(f"- Rule: {safety.get('rule')}")
        lines.append("| Method | d risk F1 | d low recall | d raw cov | d calib cov | d calib width |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in safety.get("top_rows", []):
            lines.append(
                "| {method} | {risk} | {low} | {raw} | {cov} | {width} |".format(
                    method=row.get("method"),
                    risk=_fmt_signed_pct(row.get("delta_risk_f1")),
                    low=_fmt_signed_pct(row.get("delta_low_friction_recall")),
                    raw=_fmt_signed_pct(row.get("delta_raw_interval_coverage")),
                    cov=_fmt_signed_pct(row.get("delta_calibrated_coverage")),
                    width=_fmt_signed_abs(row.get("delta_calibrated_width")),
                )
            )
        lines.append("")

    checkpoint_policy = report.get("checkpoint_policy", {})
    if checkpoint_policy:
        lines.extend(["## Checkpoint Policy", ""])
        lines.append(
            "- P0 status: `{complete}` complete methods; incomplete: {missing}.".format(
                complete=checkpoint_policy.get("p0_complete"),
                missing=", ".join(f"`{item}`" for item in checkpoint_policy.get("p0_incomplete", [])) or "-",
            )
        )
        safety_rows = checkpoint_policy.get("safety_rows")
        if safety_rows is not None:
            lines.append(
                "- Safety checkpoint status: `{verdict}`, complete `{complete}/{rows}`.".format(
                    verdict=checkpoint_policy.get("safety_verdict"),
                    complete=checkpoint_policy.get("safety_complete"),
                    rows=safety_rows,
                )
            )
        policy = checkpoint_policy.get("policy", {})
        for key in [
            "main_ablation_table",
            "supplemental_safety_analysis",
            "final_method_selection",
            "claim_boundary",
        ]:
            if policy.get(key):
                lines.append(f"- `{key}`: {policy.get(key)}")
        live = checkpoint_policy.get("live_checkpoint", {})
        if live:
            lines.append(
                "- Live checkpoint split `{run}`: latest epoch `{latest}`, best val-loss epoch `{best}`, best safety-proxy epoch `{safety}`.".format(
                    run=live.get("run") or "-",
                    latest=live.get("latest_epoch"),
                    best=live.get("best_val_loss_epoch"),
                    safety=live.get("best_safety_proxy_epoch"),
                )
            )
        lines.append("")

    trace = report.get("config_to_code_trace", {})
    if trace:
        lines.extend(["## Config-To-Code Trace", ""])
        lines.append(
            "- Verdict `{verdict}` with `{blocks}` blocks and `{warnings}` warnings across `{rows}` traced modules.".format(
                verdict=trace.get("verdict"),
                blocks=trace.get("num_blocks"),
                warnings=trace.get("num_warnings"),
                rows=trace.get("num_rows"),
            )
        )
        failures = trace.get("failures") or []
        if failures:
            lines.append("- Missing traces: " + ", ".join(f"`{item}`" for item in failures) + ".")
        else:
            lines.append("- All configured innovation modules have source-code traces.")
        lines.append("")

    artifact_contract = report.get("artifact_contract", {})
    if artifact_contract:
        lines.extend(["## Artifact Contract", ""])
        lines.append(
            "- Verdict `{verdict}`; contract-complete `{complete}/{total}`, invalid complete-looking `{invalid}`, stale rows `{stale}`.".format(
                verdict=artifact_contract.get("verdict"),
                complete=artifact_contract.get("num_contract_complete"),
                total=artifact_contract.get("num_runs"),
                invalid=artifact_contract.get("num_invalid_complete_like"),
                stale=artifact_contract.get("num_stale_rows"),
            )
        )
        hard = artifact_contract.get("hard_status", {})
        for name, item in hard.items():
            lines.append(
                "- `{name}`: `{complete}` ({done}/{total} complete).".format(
                    name=name,
                    complete=item.get("complete"),
                    done=item.get("num_complete"),
                    total=item.get("num_runs"),
                )
            )
        for row in artifact_contract.get("top_incomplete", [])[:5]:
            missing = ", ".join(f"`{item}`" for item in row.get("missing", [])) or "-"
            lines.append(
                "- `{run}` `{status}`: {missing}; next `{action}`.".format(
                    run=row.get("name"),
                    status=row.get("contract_status"),
                    missing=missing,
                    action=row.get("next_action"),
                )
            )
        lines.append("")

    claim_evidence = report.get("claim_evidence", {})
    if claim_evidence:
        lines.extend(["## Claim Evidence", ""])
        counts = claim_evidence.get("status_counts", {})
        lines.append(
            "- Supported `{supported}`, partial `{partial}`, not supported `{hard_not_supported}`, not yet supported `{not_supported}`.".format(
                supported=counts.get("supported", 0),
                partial=counts.get("partial", 0),
                hard_not_supported=counts.get("not_supported", 0),
                not_supported=counts.get("not_supported_yet", 0),
            )
        )
        for row in claim_evidence.get("top_not_supported", [])[:5]:
            missing = ", ".join(f"`{item}`" for item in row.get("missing_or_risk", []) if item) or "-"
            label = "Not supported" if row.get("status") == "not_supported" else "Not yet supported"
            lines.append(
                "- {label}: {claim}; missing/risk {missing}; wording: {wording}".format(
                    label=label,
                    claim=row.get("claim"),
                    missing=missing,
                    wording=row.get("allowed_wording"),
                )
            )
        for row in claim_evidence.get("partial", [])[:3]:
            missing = ", ".join(f"`{item}`" for item in row.get("missing_or_risk", []) if item) or "-"
            lines.append(
                "- Partial: {claim}; risk {missing}; wording: {wording}".format(
                    claim=row.get("claim"),
                    missing=missing,
                    wording=row.get("allowed_wording"),
                )
            )
        lines.append("")

    reviewer_checklist = report.get("reviewer_evidence_checklist", {})
    if reviewer_checklist:
        lines.extend(["## Reviewer Evidence Checklist", ""])
        counts = reviewer_checklist.get("status_counts", {})
        lines.append(
            "- Checklist artifact: `{artifact}`; supported `{supported}`, partial `{partial}`, not supported `{hard_not_supported}`, not yet supported `{not_supported}`.".format(
                artifact=reviewer_checklist.get("artifact"),
                supported=counts.get("supported", 0),
                partial=counts.get("partial", 0),
                hard_not_supported=counts.get("not_supported", 0),
                not_supported=counts.get("not_supported_yet", 0),
            )
        )
        for row in reviewer_checklist.get("supported", [])[:3]:
            lines.append(f"- Supported: {row.get('claim')} -> {row.get('allowed_wording')}")
        for row in reviewer_checklist.get("not_supported", [])[:5]:
            missing = ", ".join(f"`{item}`" for item in row.get("missing", []) if item) or "-"
            label = "Not supported" if row.get("status") == "not_supported" else "Not supported yet"
            lines.append(
                "- {label}: {claim}; missing {missing}; wording: {wording}".format(
                    label=label,
                    claim=row.get("claim"),
                    missing=missing,
                    wording=row.get("allowed_wording"),
                )
            )
        for row in reviewer_checklist.get("partial", [])[:4]:
            missing = ", ".join(f"`{item}`" for item in row.get("missing", []) if item) or "-"
            lines.append(
                "- Partial: {claim}; missing/risk {missing}; wording: {wording}".format(
                    claim=row.get("claim"),
                    missing=missing,
                    wording=row.get("allowed_wording"),
                )
            )
        if reviewer_checklist.get("strict_rules"):
            lines.append("")
            lines.append("Strict rules:")
            for item in reviewer_checklist.get("strict_rules", []):
                lines.append(f"- {item}")
        lines.append("")

    external = report.get("external_benchmark", {})
    if external:
        lines.extend(["## External Benchmark Alignment", ""])
        lines.append(
            "- Sources mapped: `{sources}`; datasets aligned: {datasets}.".format(
                sources=external.get("num_sources"),
                datasets=", ".join(f"`{item}`" for item in external.get("datasets", []) if item),
            )
        )
        statuses = external.get("completion_relevant_requirements", {})
        if statuses:
            lines.append(
                "- Evidence status: "
                + ", ".join(f"`{name}`={status}" for name, status in statuses.items())
                + "."
            )
        if external.get("strict_comparison_rule"):
            lines.append(f"- Rule: {external.get('strict_comparison_rule')}")
        lines.append("")

    open_source = report.get("open_source_reproducibility", {})
    if open_source:
        lines.extend(["## Open-Source Reproducibility", ""])
        lines.append(
            "- Sources mapped: `{sources}`; implemented/configured: `{implemented}`; future-only: `{future}`.".format(
                sources=open_source.get("num_sources"),
                implemented=open_source.get("num_implemented_or_configured"),
                future=open_source.get("num_future_only"),
            )
        )
        for row in open_source.get("implemented_rows", [])[:5]:
            lines.append(
                "- `{name}` `{status}`: {use}".format(
                    name=row.get("name"),
                    status=row.get("status"),
                    use=row.get("project_use"),
                )
            )
        lines.append("")
        lines.append("Strict claim rules:")
        for item in open_source.get("strict_claim_rules", [])[:3]:
            lines.append(f"- {item}")
        lines.append("")

    roadmap = report.get("topvenue_innovation_roadmap", {})
    if roadmap:
        lines.extend(["## Top-Venue Innovation Roadmap", ""])
        lines.append(
            "- Patterns mapped: `{patterns}`; sources mapped: `{sources}`.".format(
                patterns=roadmap.get("num_patterns"),
                sources=roadmap.get("num_sources"),
            )
        )
        evidence = roadmap.get("current_evidence", {})
        if evidence:
            best_risk = evidence.get("p0_best_completed_by_risk_method") or evidence.get("best_completed_p0_method")
            best_safety = (
                evidence.get("p0_best_completed_by_safety_score_method")
                or evidence.get("best_completed_p0_method")
            )
            lines.append(
                "- Current evidence: P0 rows `{p0}/{total}`, risk-F1 best `{best_risk}`, safety/generalization best `{best_safety}`, shortcut verdict `{shortcut}`.".format(
                    p0=evidence.get("p0_complete_rows"),
                    total=evidence.get("p0_total_rows"),
                    best_risk=best_risk,
                    best_safety=best_safety,
                    shortcut=evidence.get("dataset_shortcut_verdict"),
                )
            )
        for item in roadmap.get("next_decisions", [])[:4]:
            lines.append(
                "- `{decision}` {action}".format(
                    decision=item.get("decision"),
                    action=item.get("action"),
                )
            )
        lines.append("")

    hypothesis = report.get("candidate_hypothesis_matrix", {})
    if hypothesis:
        lines.extend(["## Candidate Hypothesis Matrix", ""])
        lines.append(
            "- Verdict `{verdict}` across `{rows}` predeclared experiment hypotheses.".format(
                verdict=hypothesis.get("verdict"),
                rows=hypothesis.get("num_rows"),
            )
        )
        lines.append(
            "- Phase counts: `{phases}`; contract statuses: `{statuses}`.".format(
                phases=json.dumps(hypothesis.get("phase_counts", {}), ensure_ascii=False, sort_keys=True),
                statuses=json.dumps(hypothesis.get("contract_status_counts", {}), ensure_ascii=False, sort_keys=True),
            )
        )
        if hypothesis.get("failure_signals"):
            lines.append("Failure signals driving the candidate queue:")
            for item in hypothesis.get("failure_signals", [])[:4]:
                lines.append(f"- `{item.get('signal')}`: {item.get('candidate_response')}")
        lines.append("")
        lines.append("| Next run | Phase | Status | Success rule | Failure action |")
        lines.append("|---|---|---|---|---|")
        for row in hypothesis.get("next_rows", [])[:6]:
            lines.append(
                "| `{run}` | {phase} | `{status}` | {success} | {failure} |".format(
                    run=row.get("run"),
                    phase=row.get("phase"),
                    status=row.get("contract_status"),
                    success=row.get("success_criteria"),
                    failure=row.get("failure_action"),
                )
            )
        lines.append("")

    friction_sources = report.get("friction_interval_sources", {})
    if friction_sources:
        lines.extend(["## Friction Interval Sources", ""])
        lines.append(
            "- Verdict `{verdict}`: `{anchors}` public reference anchors, `{failures}` failures.".format(
                verdict=friction_sources.get("verdict"),
                anchors=friction_sources.get("num_anchors"),
                failures=friction_sources.get("num_failures"),
            )
        )
        if friction_sources.get("source_keys"):
            lines.append(
                "- Source groups: "
                + ", ".join(f"`{item}`" for item in friction_sources.get("source_keys", []))
                + "."
            )
        if friction_sources.get("task_framing"):
            lines.append(f"- Framing: {friction_sources.get('task_framing')}")
        lines.append("")

    queue = report.get("queue_recovery", {})
    if queue:
        lines.extend(["## Queue Recovery", ""])
        lines.append(
            "- Queue snapshot `{stamp}`: `{complete}/{total}` complete, `{partial}` partial/running, `{missing}` missing.".format(
                stamp=queue.get("generated_at", "-"),
                complete=queue.get("num_complete"),
                total=queue.get("num_total"),
                partial=queue.get("num_partial"),
                missing=queue.get("num_missing"),
            )
        )
        next_item = queue.get("next_incomplete") or {}
        if next_item:
            lines.append(
                f"- Next incomplete by execution order: `{next_item.get('name')}` (`{next_item.get('status')}`)."
            )
        for row in queue.get("active_rows", []):
            lines.append(
                "- Active queue row `{name}`: {progress}; log age {age}.".format(
                    name=row.get("name"),
                    progress=_queue_progress_text(row),
                    age=_queue_log_age_text(row),
                )
            )
        proc_snapshot = queue.get("process_snapshot") or {}
        if proc_snapshot:
            lines.append(
                "- Process kinds: `{counts}`.".format(
                    counts=json.dumps(proc_snapshot.get("counts", {}), ensure_ascii=False, sort_keys=True)
                )
            )
            for proc in proc_snapshot.get("followups", []):
                detail = []
                if proc.get("phase"):
                    detail.append(f"phase={proc.get('phase')}")
                if proc.get("wait_pid"):
                    detail.append(f"wait_pid={proc.get('wait_pid')}")
                suffix = f" ({', '.join(detail)})" if detail else ""
                lines.append(
                    "- Follow-up `{kind}` PID `{pid}` parent `{parent}`{suffix}: `{preview}`".format(
                        kind=proc.get("kind"),
                        pid=proc.get("pid"),
                        parent=proc.get("parent"),
                        preview=proc.get("preview") or "-",
                        suffix=suffix,
                    )
                )
        commands = queue.get("recovery_commands", {})
        if commands:
            lines.append(f"- Recovery command: `{commands.get('resume_all')}`")
        lines.append("")

    lines.extend(["## P3 Prepared Runs", ""])
    ready = report.get("p3_ready_runs", [])
    lines.append(", ".join(f"`{item}`" for item in ready) if ready else "- None.")
    lines.append("")

    final_route = report.get("final_route_sanity", {})
    if final_route:
        lines.extend(["## Final Route Sanity", ""])
        lines.append(
            "- Verdict `{verdict}` across `{runs}` final configs.".format(
                verdict=final_route.get("verdict"),
                runs=final_route.get("num_final_runs"),
            )
        )
        lines.append(f"- Policy: {final_route.get('policy')}")
        if final_route.get("unstable_present"):
            lines.append(
                "- Unstable modules still present: "
                + ", ".join(f"`{item}`" for item in final_route.get("unstable_present", []))
                + "."
            )
        if final_route.get("missing_core"):
            lines.append(
                "- Missing lean core modules: "
                + ", ".join(f"`{item}`" for item in final_route.get("missing_core", []))
                + "."
            )
        if final_route.get("missing_safety"):
            lines.append(
                "- Missing final safety mechanisms: "
                + ", ".join(f"`{item}`" for item in final_route.get("missing_safety", []))
                + "."
            )
        lines.append("")

    lines.extend(["## Disk", ""])
    lines.append("| Disk | Free GB | Used GB | Total GB |")
    lines.append("|---|---:|---:|---:|")
    for disk in report.get("system", {}).get("disks", []):
        if disk.get("available"):
            lines.append(f"| {disk['path']} | {disk['free_gb']} | {disk['used_gb']} | {disk['total_gb']} |")
    lines.append("")

    lines.extend(["## Next Milestones", ""])
    for item in report.get("next_milestones", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _active_row_text(row: dict[str, Any], tqdm: dict[str, Any] | None = None) -> str:
    epoch = row.get("active_epoch")
    epochs = row.get("active_epochs")
    step = row.get("active_step")
    steps = row.get("active_steps")
    phase = row.get("active_phase") or "step"
    suffix = ""
    if tqdm:
        phase = tqdm.get("phase") or phase
        suffix = f" ({phase} {tqdm.get('percent')}%, ETA {tqdm.get('eta')}, {tqdm.get('rate')})"
        tqdm_step = _as_int(tqdm.get("step"))
        row_step = _as_int(step)
        if tqdm_step is not None and (phase != "train" or row_step is None or tqdm_step >= row_step):
            step = tqdm_step
            steps = tqdm.get("steps")
    if epoch is not None and step is not None:
        return f"{row.get('name')} epoch {epoch}/{epochs} {phase} {step}/{steps}{suffix}"
    if epoch is not None:
        return f"{row.get('name')} epoch {epoch}/{epochs}{suffix}"
    return str(row.get("name")) + suffix


def _queue_progress_text(row: dict[str, Any]) -> str:
    epoch = row.get("active_epoch") or row.get("epoch")
    epochs = row.get("active_epochs") or row.get("epochs")
    step = row.get("active_step")
    steps = row.get("active_steps")
    phase = row.get("active_phase") or "step"
    if epoch is None:
        return "-"
    if step is not None and steps is not None:
        return f"epoch {epoch}/{epochs}, {phase} {step}/{steps}"
    return f"epoch {epoch}/{epochs}"


def _queue_log_age_text(row: dict[str, Any]) -> str:
    age = row.get("active_log_age_seconds")
    if age is None:
        return "-"
    try:
        seconds = float(age)
    except (TypeError, ValueError):
        return "-"
    suffix = " stale" if row.get("active_log_stale") else ""
    if seconds < 60:
        return f"{seconds:.0f}s{suffix}"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m{suffix}"
    return f"{seconds / 3600:.1f}h{suffix}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def _fmt_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_signed_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):+.2f}%"


def _fmt_signed_abs(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.4f}"


def _loss_scope_label(value: Any) -> str:
    if value == "current_epoch":
        return "current epoch"
    return "displayed points"


def _dict_text(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    return ", ".join(f"{key}:{val}" for key, val in value.items())


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
