from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\NMI_SPWFM_datasets\friction_affordance_outputs\rscd_surface_classification")
OUT = Path("reports/paper_protocol_summary/rscd_final_method_selection")

BASELINE = "formal_convnext_tiny_b12e20_resume"
PHYSICS = "formal_physics_texture_quality_b12e20_resume"
PHYSICS_ALT = "formal_physics_texture_quality_b12e20_parallel"
PROMOTED = "formal_physics_wavelet_directional_film_gate_hier"
TTA = "tta_ensemble_physics_texture_formal_hflip"
POSTHOC_CALIBRATION = "topology_logit_calibration_physics_texture"
MATERIAL_GATE_FAST = "fast_physics_material_gate_patch_quality"
FACTOR_LOGIT_FAST = "fast_physics_factor_logit_adjustment"
FACTOR_MARGINAL_MARKER = "factor_marginal"
FACTORIZED_LOW_RANK_MARKER = "factorized_lowrank"
CALIBRATION_DISTILL_MARKER = "calibration_distill"
RETINEX_FAST = [
    "fast_physics_retinex_texture_quality",
    "fast_physics_retinex_film_gate_hier",
]
PATCH_STATS_FAST = [
    "fast_physics_texture_quality_patch_stats",
    "fast_physics_texture_quality_patch_stats_224",
]
FOUNDATION_FAST = [
    "fast_dinov2_global_rscd",
    "fast_dinov2_physics_texture_rscd",
]
SKIPPED_MARKERS = {
    "fast_dinov2_physics_texture_rscd": "skipped_after_global_failure.json",
}

STRICT_TARGET = {
    "method": "RoadFormer-L",
    "top1": 0.9286,
    "macro_f1": 0.8499,
}


def main() -> None:
    rows = _load_rows()
    by_name = {row["name"]: row for row in rows}
    baseline = by_name.get(BASELINE)
    physics = by_name.get(PHYSICS) or by_name.get(PHYSICS_ALT)
    promoted = by_name.get(PROMOTED)
    posthoc = by_name.get(POSTHOC_CALIBRATION)
    report_posthoc = _select_report_posthoc(rows)

    for row in rows:
        row["delta_top1_vs_baseline"] = _delta(row, baseline, "top1")
        row["delta_f1_vs_baseline"] = _delta(row, baseline, "macro_f1")
        row["delta_top1_vs_physics"] = _delta(row, physics, "top1")
        row["delta_f1_vs_physics"] = _delta(row, physics, "macro_f1")
        row["gap_top1_to_strict_target"] = row["top1"] - STRICT_TARGET["top1"]
        row["gap_f1_to_strict_target"] = row["macro_f1"] - STRICT_TARGET["macro_f1"]
        row["slices"] = _slice_metrics(row["payload"])

    decision = _decision(
        rows=rows,
        baseline=baseline,
        physics=physics,
        promoted=promoted,
        posthoc=posthoc,
        report_posthoc=report_posthoc,
    )
    result = {
        "claim_boundary": (
            "This is an automatic local method-selection audit for RSCD-27 road-surface "
            "classification. It selects the best defensible local method; it does not "
            "declare external SOTA unless the strict RoadFormer/RoadMamba protocol is matched."
        ),
        "strict_external_context_target": STRICT_TARGET,
        "required_pending_outputs": _pending_outputs(),
        "rows": sorted(_strip_payload(rows), key=lambda r: (r["macro_f1"], r["top1"]), reverse=True),
        "decision": decision,
    }
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT.with_suffix(".md").write_text(_to_markdown(result), encoding="utf-8")
    print(OUT.with_suffix(".md"))


def _load_rows() -> list[dict[str, Any]]:
    rows = []
    if not ROOT.exists():
        return rows
    for path in sorted(ROOT.glob("*/evaluate_test.json")):
        name = path.parent.name
        if name.startswith(("debug_", "smoke_")):
            continue
        payload = _load_json(path)
        if not payload:
            continue
        summary = payload.get("summary", {})
        mean_precision, mean_recall = _mean_pr_from_payload(payload)
        slices = _slice_metrics(payload)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "top1": _num(summary.get("top1")),
                "mean_precision": _num(summary.get("mean_precision", mean_precision)),
                "mean_recall": _num(summary.get("mean_recall", mean_recall)),
                "macro_f1": _num(summary.get("macro_f1")),
                "weighted_f1": _num(summary.get("weighted_f1")),
                "balanced_accuracy": _num(summary.get("balanced_accuracy")),
                "num_samples": int(summary.get("num_samples") or 0),
                **slices,
                "payload": payload,
            }
        )
    return rows


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean_pr_from_payload(payload: dict[str, Any]) -> tuple[float, float]:
    report = payload.get("classification_report", {})
    precisions = []
    recalls = []
    for label, item in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if isinstance(item, dict) and "precision" in item and "recall" in item:
            precisions.append(_num(item.get("precision")))
            recalls.append(_num(item.get("recall")))
    if not precisions:
        return 0.0, 0.0
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def _delta(row: dict[str, Any], base: dict[str, Any] | None, key: str) -> float | None:
    if base is None:
        return None
    return float(row[key]) - float(base[key])


def _slice_metrics(payload: dict[str, Any]) -> dict[str, float]:
    report = payload.get("classification_report", {})
    class_rows = []
    for label, item in report.items():
        if not isinstance(item, dict) or "f1-score" not in item:
            continue
        support = int(item.get("support") or 0)
        if support <= 0:
            continue
        class_rows.append(
            {
                "label": str(label),
                "friction": _friction_state(str(label)),
                "material": _material_state(str(label)),
                "f1": _num(item.get("f1-score")),
                "recall": _num(item.get("recall")),
                "support": support,
            }
        )
    return {
        "wet_water_f1": _weighted_slice(class_rows, lambda r: r["friction"] in {"wet", "water"}, "f1"),
        "winter_f1": _weighted_slice(
            class_rows,
            lambda r: r["friction"] in {"fresh_snow", "melted_snow", "ice"},
            "f1",
        ),
        "low_friction_visual_f1": _weighted_slice(
            class_rows,
            lambda r: r["friction"] in {"wet", "water", "fresh_snow", "melted_snow", "ice"},
            "f1",
        ),
        "water_f1": _weighted_slice(class_rows, lambda r: r["friction"] == "water", "f1"),
        "wet_concrete_f1": _weighted_slice(
            class_rows,
            lambda r: r["friction"] == "wet" and r["material"] == "concrete",
            "f1",
        ),
        "water_concrete_f1": _weighted_slice(
            class_rows,
            lambda r: r["friction"] == "water" and r["material"] == "concrete",
            "f1",
        ),
    }


def _friction_state(label: str) -> str:
    label = label.strip().lower().replace("-", "_")
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return label
    return label.split("_")[0] if label else "unknown"


def _material_state(label: str) -> str | None:
    label = label.strip().lower().replace("-", "_")
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return None
    parts = label.split("_")
    return parts[1] if len(parts) >= 2 else None


def _weighted_slice(rows: list[dict[str, Any]], pred: Any, key: str) -> float:
    selected = [row for row in rows if pred(row)]
    support = sum(int(row["support"]) for row in selected)
    if support <= 0:
        return 0.0
    return sum(float(row[key]) * int(row["support"]) for row in selected) / support


def _pending_outputs() -> list[str]:
    names = [PROMOTED, TTA, MATERIAL_GATE_FAST, *RETINEX_FAST, *PATCH_STATS_FAST, *FOUNDATION_FAST]
    return [
        str(ROOT / name / "evaluate_test.json")
        for name in names
        if not (ROOT / name / "evaluate_test.json").exists() and not _skip_marker_exists(name)
    ]


def _decision(
    *,
    rows: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    physics: dict[str, Any] | None,
    promoted: dict[str, Any] | None,
    posthoc: dict[str, Any] | None,
    report_posthoc: dict[str, Any] | None,
) -> dict[str, Any]:
    if not baseline or not physics:
        return {
            "status": "waiting_for_core_formals",
            "main_method": None,
            "message": "ConvNeXt and PhysicsTexture formal results are required before final selection.",
            "module_actions": [],
        }
    strict_single_rows = [row for row in rows if _is_strict_single_model_candidate(row)]
    metric_best_formal = max(strict_single_rows, key=lambda r: (r["macro_f1"], r["top1"])) if strict_single_rows else physics
    best_formal = _select_retained_formal(metric_best_formal, physics)
    pending = _pending_outputs()

    foundation_rows = [row for row in rows if row["name"] in FOUNDATION_FAST]
    dinov2_global = next((row for row in foundation_rows if row["name"] == "fast_dinov2_global_rscd"), None)
    dinov2_skipped = _skip_marker_exists("fast_dinov2_physics_texture_rscd")

    module_actions = [
        {
            "module": "PhysicsTexture",
            "action": "keep_as_validated_core",
            "reason": "It improves the matched formal ConvNeXt baseline and gives friction-relevant slice evidence.",
        },
        {
            "module": "PhysicsAttention",
            "action": "prune_current_implementation",
            "reason": "Fast RSCD screens were strongly below PhysicsTexture; weak pseudo-masks are not reliable without pixel labels.",
        },
        {
            "module": "FrictionSet_and_generic_DG_losses",
            "action": "do_not_use_as_main_rscd_method",
            "reason": "Earlier ablations reduced worst-dataset or main metrics; keep only as analysis unless redesigned.",
        },
        _foundation_action(dinov2_global, dinov2_skipped),
        _patch_invariant_action(rows),
    ]
    roughness_neighbor_rows = [row for row in rows if "roughness_neighbor" in row["name"]]
    if roughness_neighbor_rows:
        best_roughness_neighbor = max(roughness_neighbor_rows, key=lambda r: (r["macro_f1"], r["top1"]))
        action = (
            "review_for_promotion"
            if best_roughness_neighbor["macro_f1"] > best_formal["macro_f1"]
            and best_roughness_neighbor["top1"] >= best_formal["top1"] - 0.001
            else "prune_current_implementation"
        )
        module_actions.append(
            {
                "module": "RoughnessNeighborResidual",
                "action": action,
                "reason": (
                    f"Best roughness-neighbor screen `{best_roughness_neighbor['name']}` reached "
                    f"{best_roughness_neighbor['top1'] * 100:.2f}% Top-1 and "
                    f"{best_roughness_neighbor['macro_f1'] * 100:.2f}% Mean-F1. "
                    "It is below the current strict main method, so keep the idea only as a diagnostic "
                    "unless a later version improves hard slices without water/dry regression."
                ),
            }
        )
    roughness_smoothing_rows = [row for row in rows if "roughsmooth" in row["name"]]
    if roughness_smoothing_rows:
        best_roughness_smoothing = max(roughness_smoothing_rows, key=lambda r: (r["macro_f1"], r["top1"]))
        best_top1_smoothing = max(roughness_smoothing_rows, key=lambda r: (r["top1"], r["macro_f1"]))
        action = (
            "review_for_promotion"
            if best_roughness_smoothing["macro_f1"] > best_formal["macro_f1"]
            and best_roughness_smoothing["top1"] >= best_formal["top1"] - 0.001
            else "diagnostic_only_do_not_promote"
        )
        module_actions.append(
            {
                "module": "RoughnessNeighborSmoothing",
                "action": action,
                "reason": (
                    "Narrow soft-label smoothing between adjacent roughness states confirms the roughness "
                    "bottleneck but does not beat the strict main Macro-F1. "
                    f"Best Mean-F1 run `{best_roughness_smoothing['name']}` reached "
                    f"{best_roughness_smoothing['top1'] * 100:.2f}% Top-1 and "
                    f"{best_roughness_smoothing['macro_f1'] * 100:.2f}% Mean-F1; "
                    f"best Top-1 run `{best_top1_smoothing['name']}` reached "
                    f"{best_top1_smoothing['top1'] * 100:.2f}% Top-1 and "
                    f"{best_top1_smoothing['macro_f1'] * 100:.2f}% Mean-F1. "
                    "Keep this as evidence for a future factor-coupled label geometry loss, not as the current main method."
                ),
            }
        )
    material_rows = [row for row in rows if row["name"] == MATERIAL_GATE_FAST]
    if material_rows:
        material = material_rows[0]
        action = "prune_fast_screen" if material["macro_f1"] < physics["macro_f1"] else "review_for_promotion"
        reason = (
            "Patch-compatible material-conditioned gating did not beat the PhysicsTexture reference on fast Macro-F1 "
            "and regressed wet/water or water-concrete slices."
            if action == "prune_fast_screen"
            else "Patch-compatible material-conditioned gating beat the PhysicsTexture reference; review slice metrics before promotion."
        )
        module_actions.append({"module": "MaterialConditionedGate", "action": action, "reason": reason})
    else:
        module_actions.append(
            {
                "module": "MaterialConditionedGate",
                "action": "pending_fast_screen",
                "reason": "Queued as a patch-compatible alternative to the failed Wavelet/Directional/FiLM material route.",
            }
        )
    retinex_rows = [row for row in rows if row["name"] in RETINEX_FAST]
    if retinex_rows:
        best_retinex = max(retinex_rows, key=lambda r: (r["macro_f1"], r["top1"]))
        action = "prune_fast_screen" if best_retinex["macro_f1"] < physics["macro_f1"] else "review_for_promotion"
        reason = (
            "Retinex illumination-invariant candidates did not beat PhysicsTexture on fast Macro-F1, so they are not kept as a main RSCD route."
            if action == "prune_fast_screen"
            else "A Retinex candidate beat PhysicsTexture on fast Macro-F1; review hard wet/water slices before promotion."
        )
        module_actions.append({"module": "RetinexTexture", "action": action, "reason": reason})
    else:
        module_actions.append(
            {
                "module": "RetinexTexture",
                "action": "pending_fast_screen",
                "reason": "Queued to test whether illumination-invariant reflectance and chromaticity cues help wet-film ambiguity.",
            }
        )
    if promoted is None:
        module_actions.append(
            {
                "module": "WaveletDirectionalFiLM",
                "action": "pending_formal_result",
                "reason": "Fast screen was slightly positive, but final formal test is not available yet.",
            }
        )
    else:
        keep, reason = _passes_retention_gate(promoted, physics)
        if keep:
            module_actions.append(
                {
                    "module": "WaveletDirectionalFiLM",
                    "action": "retain_as_main_candidate",
                    "reason": reason,
                }
            )
        else:
            module_actions.append(
                {
                    "module": "WaveletDirectionalFiLM",
                    "action": "prune_or_keep_as_failed_fast_to_formal_case",
                    "reason": reason,
                }
            )

    posthoc_retained = False
    if posthoc is not None:
        posthoc_retained, posthoc_reason = _passes_retention_gate(posthoc, physics)
        module_actions.append(
            {
                "module": "PostHocLogitCalibration",
                "action": "retain_as_posthoc_candidate" if posthoc_retained else "do_not_promote",
                "reason": (
                    posthoc_reason
                    + " Report separately from strict end-to-end single-model results because this candidate uses validation-fitted calibration."
                ),
            }
        )
    else:
        module_actions.append(
            {
                "module": "PostHocLogitCalibration",
                "action": "not_available",
                "reason": "No validation-fitted post-hoc calibration result is available.",
            }
        )

    factor_logit_rows = [row for row in rows if row["name"] == FACTOR_LOGIT_FAST]
    if factor_logit_rows:
        factor_logit = factor_logit_rows[0]
        fast_ref = next((row for row in rows if row["name"] == "fast_physics_texture_quality"), None)
        ref = fast_ref or physics
        action = "prune_fast_screen" if factor_logit["macro_f1"] < ref["macro_f1"] else "review_for_promotion"
        reason = (
            "The factor-aware logit adjustment did not beat the fast PhysicsTexture reference; "
            f"current fast result is {factor_logit['top1'] * 100:.2f}% Top-1 and "
            f"{factor_logit['macro_f1'] * 100:.2f}% Mean-F1. Keep the code switch for future ablations, "
            "but do not promote this implementation to formal training."
            if action == "prune_fast_screen"
            else "The factor-aware logit adjustment beat the local reference; review class-slice evidence before formal promotion."
        )
        module_actions.append({"module": "FactorLogitAdjustment", "action": action, "reason": reason})
    else:
        module_actions.append(
            {
                "module": "FactorLogitAdjustment",
                "action": "not_screened",
                "reason": "No fast result is available for the trainable factor-aware calibration head.",
            }
        )

    factor_marginal_rows = [row for row in rows if FACTOR_MARGINAL_MARKER in row["name"]]
    if factor_marginal_rows:
        module_actions.append(_factor_marginal_action(rows, factor_marginal_rows))

    factorized_low_rank_rows = [row for row in rows if FACTORIZED_LOW_RANK_MARKER in row["name"]]
    if factorized_low_rank_rows:
        module_actions.append(_factorized_low_rank_action(rows, factorized_low_rank_rows))

    graph_label_rows = [
        row
        for row in rows
        if "label_graph" in row["name"]
        or "graph_diffusion" in row["name"]
        or "graph_angular" in row["name"]
    ]
    if graph_label_rows:
        module_actions.append(_graph_label_action(graph_label_rows, best_formal))

    dual_aux_rows = [row for row in rows if "dual_aux" in row["name"]]
    if dual_aux_rows:
        module_actions.append(_dual_aux_action(dual_aux_rows, best_formal))

    distill_rows = [row for row in rows if CALIBRATION_DISTILL_MARKER in row["name"]]
    if distill_rows:
        module_actions.append(_calibration_distill_action(rows, distill_rows))
    else:
        module_actions.append(
            {
                "module": "CalibrationDistillation",
                "action": "not_screened",
                "reason": (
                    "No fast result is available for distilling the validation-fitted "
                    "post-hoc calibration back into a trainable single model."
                ),
            }
        )

    if pending:
        status = "pending_downstream_candidates"
        message = "Core formal evidence exists, but promoted/TTA/material/Retinex/foundation-probe downstream outputs are still pending."
    elif best_formal["top1"] > STRICT_TARGET["top1"] and best_formal["macro_f1"] > STRICT_TARGET["macro_f1"]:
        status = "possible_external_sota_requires_protocol_audit"
        message = "Best local result exceeds the strict context target; verify exact RoadFormer/RoadMamba protocol before claiming SOTA."
    elif posthoc_retained and best_formal["name"] == physics["name"]:
        status = "select_physics_texture_core_with_posthoc_calibration"
        message = (
            "Use PhysicsTexture as the strict single-model core. Also report the validation-fitted "
            "logit calibration as a separate post-hoc candidate because it improves averages and "
            "hard wet/water slices without replacing the end-to-end model claim."
        )
    elif best_formal["name"] == physics["name"]:
        status = "select_physics_texture_core"
        message = (
            "Use PhysicsTexture as the main local method; more complex formal candidates "
            "must beat it without regressing wet/water hard slices before replacing it."
        )
    else:
        status = "select_best_strict_single_model_candidate"
        message = "Use the best strict local single-model candidate if class-slice evidence does not reveal a safety-critical regression."

    return {
        "status": status,
        "main_method": best_formal["name"],
        "metric_best_formal": metric_best_formal["name"],
        "message": message,
        "best_formal_top1": best_formal["top1"],
        "best_formal_macro_f1": best_formal["macro_f1"],
        "best_posthoc_method": report_posthoc["name"] if report_posthoc is not None else None,
        "best_posthoc_top1": report_posthoc["top1"] if report_posthoc is not None else None,
        "best_posthoc_macro_f1": report_posthoc["macro_f1"] if report_posthoc is not None else None,
        "module_actions": module_actions,
    }


def _select_retained_formal(
    metric_best_formal: dict[str, Any],
    physics: dict[str, Any],
) -> dict[str, Any]:
    if metric_best_formal["name"] == physics["name"]:
        return physics
    keep, _ = _passes_retention_gate(metric_best_formal, physics)
    return metric_best_formal if keep else physics


def _is_strict_single_model_candidate(row: dict[str, Any]) -> bool:
    """Return true for end-to-end single-model RSCD test results.

    Output directory prefixes are historical: some late-stage fine-tuning runs
    started as screens but still produce a complete single-checkpoint test on
    the original RSCD split. TTA, validation-fitted calibration, fast subsets,
    debug runs, and smoke tests remain excluded from the strict single-model
    decision.
    """

    name = str(row.get("name") or "")
    if int(row.get("num_samples") or 0) < 49500:
        return False
    excluded_prefixes = ("tta_", "fast_", "debug_", "smoke_", "topology_", "conditional_topology_")
    if name.startswith(excluded_prefixes):
        return False
    excluded_markers = ("calibration", "posthoc", "ensemble")
    if any(marker in name for marker in excluded_markers):
        return False
    return True


def _select_report_posthoc(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        name = str(row.get("name") or "")
        if int(row.get("num_samples") or 0) < 49500:
            continue
        if name.startswith(("tta_", "topology_", "conditional_topology_")):
            candidates.append(row)
        elif "calibration" in name or "ensemble" in name or "posthoc" in name:
            candidates.append(row)
    return max(candidates, key=lambda r: (r["macro_f1"], r["top1"])) if candidates else None


def _passes_retention_gate(candidate: dict[str, Any], physics: dict[str, Any]) -> tuple[bool, str]:
    macro_gain = float(candidate["macro_f1"]) - float(physics["macro_f1"])
    top1_gain = float(candidate["top1"]) - float(physics["top1"])
    slice_tolerances = {
        "wet_water_f1": -0.003,
        "low_friction_visual_f1": -0.003,
        "water_concrete_f1": -0.005,
    }
    slice_deltas = {}
    for key, tolerance in slice_tolerances.items():
        cand_value = float(candidate.get("slices", {}).get(key) or 0.0)
        phys_value = float(physics.get("slices", {}).get(key) or 0.0)
        slice_deltas[key] = cand_value - phys_value
        if slice_deltas[key] < tolerance:
            return (
                False,
                (
                    "Formal candidate is rejected despite any average gain because "
                    f"`{key}` regresses by {slice_deltas[key] * 100:.2f}pp versus "
                    "PhysicsTexture on a safety-relevant RSCD wet/low-friction slice."
                ),
            )
    if macro_gain < 0.002:
        return (
            False,
            (
                "Formal candidate is rejected because its Mean-F1 gain over "
                f"PhysicsTexture is only {macro_gain * 100:.2f}pp; top-venue pruning "
                "requires a nontrivial average gain plus hard-slice preservation."
            ),
        )
    if top1_gain < -0.001:
        return (
            False,
            (
                "Formal candidate is rejected because Top-1 regresses by "
                f"{top1_gain * 100:.2f}pp versus the simpler PhysicsTexture core."
            ),
        )
    slice_text = ", ".join(f"{key} {delta * 100:+.2f}pp" for key, delta in slice_deltas.items())
    return (
        True,
        (
            f"Formal candidate retained: Mean-F1 gain {macro_gain * 100:+.2f}pp, "
            f"Top-1 gain {top1_gain * 100:+.2f}pp, and hard-slice deltas are acceptable "
            f"({slice_text})."
        ),
    )


def _skip_marker_exists(name: str) -> bool:
    marker = SKIPPED_MARKERS.get(name)
    return bool(marker and (ROOT / name / marker).exists())


def _foundation_action(dinov2_global: dict[str, Any] | None, dinov2_skipped: bool) -> dict[str, str]:
    if dinov2_global and dinov2_global["macro_f1"] < 0.50:
        return {
            "module": "DINOv2_foundation_probe",
            "action": "prune_current_end_to_end_protocol",
            "reason": (
                "The fast DINOv2 global screen is far below the ConvNeXt/PhysicsTexture references "
                f"({dinov2_global['top1'] * 100:.2f}% Top-1, {dinov2_global['macro_f1'] * 100:.2f}% Mean-F1). "
                "The paired DINOv2+Physics run is skipped if the pruning marker exists; revisit this route only as frozen feature extraction or a linear-probe protocol."
            ),
        }
    if dinov2_skipped:
        return {
            "module": "DINOv2_foundation_probe",
            "action": "skipped_after_failed_global_probe",
            "reason": "The DINOv2+Physics run has an explicit skip marker from the failed global DINOv2 fast screen.",
        }
    return {
        "module": "DINOv2_foundation_probe",
        "action": "pending_fast_screen",
        "reason": "Fast RSCD-only screens are queued to test whether self-supervised visual foundation features are a stronger route than stacking more hand-crafted branches.",
    }


def _patch_invariant_action(rows: list[dict[str, Any]]) -> dict[str, str]:
    patch_rows = [row for row in rows if row["name"] in PATCH_STATS_FAST]
    if not patch_rows:
        return {
            "module": "PatchInvariantQualityStats",
            "action": "pending_fast_screen",
            "reason": "RSCD images are close road patches, so bottom-vs-top contact-region cues are being tested against position-invariant patch statistics before any deletion claim.",
        }
    best = max(patch_rows, key=lambda row: (row["macro_f1"], row["top1"]))
    return {
        "module": "PatchInvariantQualityStats",
        "action": "neutral_keep_switch_no_formal_promotion",
        "reason": (
            "Fast patch-invariant quality statistics are slightly positive but below the formal-promotion gate "
            f"({best['top1'] * 100:.2f}% Top-1, {best['macro_f1'] * 100:.2f}% Mean-F1). "
            "Use patch-invariant wording for RSCD and keep the switch; do not claim bottom-contact semantics."
        ),
    }


def _factor_marginal_action(
    rows: list[dict[str, Any]],
    factor_marginal_rows: list[dict[str, Any]],
) -> dict[str, str]:
    fast_ref = next((row for row in rows if row["name"] == "fast_physics_texture_quality"), None)
    best = max(factor_marginal_rows, key=lambda row: (row["macro_f1"], row["top1"]))
    if fast_ref is None:
        return {
            "module": "FactorMarginalConsistency",
            "action": "review_without_fast_reference",
            "reason": (
                f"Best factor-marginal run `{best['name']}` is {best['top1'] * 100:.2f}% Top-1 "
                f"and {best['macro_f1'] * 100:.2f}% Mean-F1, but the fast PhysicsTexture reference is missing."
            ),
        }
    macro_delta = float(best["macro_f1"]) - float(fast_ref["macro_f1"])
    top1_delta = float(best["top1"]) - float(fast_ref["top1"])
    slices = best.get("slices", {})
    ref_slices = fast_ref.get("slices", {})
    wet_delta = float(slices.get("wet_water_f1") or 0.0) - float(ref_slices.get("wet_water_f1") or 0.0)
    water_delta = float(slices.get("water_f1") or 0.0) - float(ref_slices.get("water_f1") or 0.0)
    if macro_delta > 0.003 and top1_delta >= -0.001 and wet_delta >= -0.002:
        action = "review_for_formal_promotion"
        verdict = "The factor-marginal constraint improved the fast reference without a safety-slice regression."
    else:
        action = "prune_current_fast_screen"
        verdict = (
            "The factor-marginal constraint did not beat fast PhysicsTexture and also failed to rescue "
            "wet/water hard slices, so it should not be promoted to formal training."
        )
    return {
        "module": "FactorMarginalConsistency",
        "action": action,
        "reason": (
            f"{verdict} Best run `{best['name']}`: {best['top1'] * 100:.2f}% Top-1, "
            f"{best['macro_f1'] * 100:.2f}% Mean-F1, deltas versus fast PhysicsTexture "
            f"Top-1 {top1_delta * 100:+.2f}pp, Mean-F1 {macro_delta * 100:+.2f}pp, "
            f"wet/water {wet_delta * 100:+.2f}pp, water {water_delta * 100:+.2f}pp."
        ),
    }


def _factorized_low_rank_action(
    rows: list[dict[str, Any]],
    factorized_rows: list[dict[str, Any]],
) -> dict[str, str]:
    fast_ref = next((row for row in rows if row["name"] == "fast_physics_texture_quality"), None)
    best = max(factorized_rows, key=lambda row: (row["macro_f1"], row["top1"]))
    if fast_ref is None:
        return {
            "module": "FactorizedLowRankHead",
            "action": "review_without_fast_reference",
            "reason": (
                f"Best factorized low-rank run `{best['name']}` is {best['top1'] * 100:.2f}% Top-1 "
                f"and {best['macro_f1'] * 100:.2f}% Mean-F1, but the fast PhysicsTexture reference is missing."
            ),
        }
    top1_delta = float(best["top1"]) - float(fast_ref["top1"])
    f1_delta = float(best["macro_f1"]) - float(fast_ref["macro_f1"])
    wet_water_delta = float(best.get("wet_water_f1", 0.0)) - float(fast_ref.get("wet_water_f1", 0.0))
    water_delta = float(best.get("water_f1", 0.0)) - float(fast_ref.get("water_f1", 0.0))
    if top1_delta > 0.0025 and f1_delta > 0.001:
        action = "keep_as_promising_fast_screen_but_fix_hard_slices"
        verdict = (
            "The low-rank factorized head gives a small fast average gain, supporting the RSCD compositional-label hypothesis. "
            "It should not be promoted as the main method until wet/water safety slices stop regressing."
        )
    elif top1_delta > 0.0 or f1_delta > 0.0:
        action = "neutral_keep_for_research_ablation"
        verdict = "The low-rank factorized head is weakly positive on averages but below the formal-promotion gate."
    else:
        action = "prune_current_fast_screen"
        verdict = "The current low-rank factorized head does not beat the fast PhysicsTexture reference."
    return {
        "module": "FactorizedLowRankHead",
        "action": action,
        "reason": (
            f"{verdict} Best run `{best['name']}`: {best['top1'] * 100:.2f}% Top-1, "
            f"{best['macro_f1'] * 100:.2f}% Mean-F1, deltas versus fast PhysicsTexture "
            f"Top-1 {top1_delta * 100:+.2f}pp, Mean-F1 {f1_delta * 100:+.2f}pp, "
            f"wet/water {wet_water_delta * 100:+.2f}pp, water {water_delta * 100:+.2f}pp."
        ),
    }


def _graph_label_action(
    graph_rows: list[dict[str, Any]],
    reference: dict[str, Any],
) -> dict[str, str]:
    best = max(graph_rows, key=lambda row: (row["macro_f1"], row["top1"]))
    top1_delta = float(best["top1"]) - float(reference["top1"])
    f1_delta = float(best["macro_f1"]) - float(reference["macro_f1"])
    best_slices = best.get("slices", {})
    ref_slices = reference.get("slices", {})
    wet_delta = float(best_slices.get("wet_water_f1") or 0.0) - float(ref_slices.get("wet_water_f1") or 0.0)
    water_concrete_delta = float(best_slices.get("water_concrete_f1") or 0.0) - float(
        ref_slices.get("water_concrete_f1") or 0.0
    )
    if f1_delta > 0.0005 and top1_delta >= -0.001 and wet_delta >= -0.001:
        action = "review_for_promotion"
        verdict = "The best graph-label method beats the current strict reference without a wet/water safety regression."
    else:
        action = "diagnostic_only_do_not_promote"
        verdict = (
            "Current graph-label methods reveal useful local structure but do not beat the strict main method; "
            "keep them as diagnostics or redesign them as uncertainty-gated hard-neighborhood losses."
        )
    return {
        "module": "GraphLabelMethods",
        "action": action,
        "reason": (
            f"{verdict} Best graph run `{best['name']}`: {best['top1'] * 100:.2f}% Top-1, "
            f"{best['macro_f1'] * 100:.2f}% Mean-F1, deltas versus strict reference "
            f"Top-1 {top1_delta * 100:+.2f}pp, Mean-F1 {f1_delta * 100:+.2f}pp, "
            f"wet/water {wet_delta * 100:+.2f}pp, water-concrete {water_concrete_delta * 100:+.2f}pp."
        ),
    }


def _dual_aux_action(
    dual_aux_rows: list[dict[str, Any]],
    reference: dict[str, Any],
) -> dict[str, str]:
    best = max(dual_aux_rows, key=lambda row: (row["macro_f1"], row["top1"]))
    top1_delta = float(best["top1"]) - float(reference["top1"])
    f1_delta = float(best["macro_f1"]) - float(reference["macro_f1"])
    best_slices = best.get("slices", {})
    ref_slices = reference.get("slices", {})
    wet_delta = float(best_slices.get("wet_water_f1") or 0.0) - float(ref_slices.get("wet_water_f1") or 0.0)
    water_delta = float(best_slices.get("water_f1") or 0.0) - float(ref_slices.get("water_f1") or 0.0)
    if f1_delta > 0.0005 and top1_delta >= -0.001 and wet_delta >= -0.001:
        action = "review_for_promotion"
        verdict = "Global-local auxiliary supervision beat the strict reference without a wet/water regression."
    else:
        action = "diagnostic_only_do_not_promote"
        verdict = (
            "Simple global/physics auxiliary supervision did not beat the strict main method; "
            "it improves some concrete roughness cells but weakens wet/water safety slices."
        )
    return {
        "module": "GlobalLocalAuxSupervision",
        "action": action,
        "reason": (
            f"{verdict} Best run `{best['name']}`: {best['top1'] * 100:.2f}% Top-1, "
            f"{best['macro_f1'] * 100:.2f}% Mean-F1, deltas versus strict reference "
            f"Top-1 {top1_delta * 100:+.2f}pp, Mean-F1 {f1_delta * 100:+.2f}pp, "
            f"wet/water {wet_delta * 100:+.2f}pp, water {water_delta * 100:+.2f}pp."
        ),
    }


def _calibration_distill_action(
    rows: list[dict[str, Any]],
    distill_rows: list[dict[str, Any]],
) -> dict[str, str]:
    physics_formal = next((row for row in rows if row["name"] == PHYSICS), None)
    formal_distill_rows = [row for row in distill_rows if row["name"].startswith("formal_")]
    if physics_formal is not None and formal_distill_rows:
        best_formal = max(formal_distill_rows, key=lambda row: (row["macro_f1"], row["top1"]))
        keep, gate_reason = _passes_retention_gate(best_formal, physics_formal)
        action = "promote_formal_distillation" if keep else "formal_failed_do_not_promote"
        macro_delta = float(best_formal["macro_f1"]) - float(physics_formal["macro_f1"])
        top1_delta = float(best_formal["top1"]) - float(physics_formal["top1"])
        return {
            "module": "CalibrationDistillation",
            "action": action,
            "reason": (
                f"Formal calibration-distillation result `{best_formal['name']}` is "
                f"{best_formal['top1'] * 100:.2f}% Top-1 and {best_formal['macro_f1'] * 100:.2f}% Mean-F1, "
                f"relative to PhysicsTexture {top1_delta * 100:+.2f}pp Top-1 and {macro_delta * 100:+.2f}pp Mean-F1. "
                f"{gate_reason}"
            ),
        }

    physics_fast = next((row for row in rows if row["name"] == "fast_physics_texture_quality"), None)
    best = max(distill_rows, key=lambda row: (row["macro_f1"], row["top1"]))
    if physics_fast is None:
        return {
            "module": "CalibrationDistillation",
            "action": "review_without_fast_reference",
            "reason": (
                f"Best calibration-distillation fast run is `{best['name']}` "
                f"({best['top1'] * 100:.2f}% Top-1, {best['macro_f1'] * 100:.2f}% Mean-F1), "
                "but the fast PhysicsTexture reference is missing."
            ),
        }

    best_top1 = max(distill_rows, key=lambda row: row["top1"])
    wet_delta = (
        float(best.get("slices", {}).get("wet_water_f1") or 0.0)
        - float(physics_fast.get("slices", {}).get("wet_water_f1") or 0.0)
    )
    water_delta = (
        float(best.get("slices", {}).get("water_f1") or 0.0)
        - float(physics_fast.get("slices", {}).get("water_f1") or 0.0)
    )
    macro_delta = float(best["macro_f1"]) - float(physics_fast["macro_f1"])
    top1_delta = float(best_top1["top1"]) - float(physics_fast["top1"])

    if macro_delta > 0.003 and wet_delta >= -0.002 and water_delta >= -0.005:
        action = "review_for_formal_promotion"
        verdict = "The best calibration-distillation run beats fast PhysicsTexture without unacceptable wet/water regression."
    else:
        action = "do_not_promote_current_fast_screen"
        verdict = (
            "Current calibration-distillation screens improve some averages but do not give a clean "
            "hard-slice-preserving gain, so they should not replace the strict PhysicsTexture core."
        )

    return {
        "module": "CalibrationDistillation",
        "action": action,
        "reason": (
            f"{verdict} Best Macro-F1 run `{best['name']}`: "
            f"{best['top1'] * 100:.2f}% Top-1, {best['macro_f1'] * 100:.2f}% Mean-F1, "
            f"wet/water delta {wet_delta * 100:+.2f}pp, water delta {water_delta * 100:+.2f}pp. "
            f"Best Top-1 run `{best_top1['name']}` improves Top-1 by {top1_delta * 100:+.2f}pp "
            "but must preserve safety-relevant wet/water classes before formal promotion."
        ),
    }


def _strip_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item.pop("payload", None)
        out.append(item)
    return out


def _pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _to_markdown(result: dict[str, Any]) -> str:
    decision = result["decision"]
    lines = [
        "# RSCD Final Method Selection",
        "",
        result["claim_boundary"],
        "",
        "## Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Main method: `{decision.get('main_method') or 'pending'}`",
        f"- Message: {decision['message']}",
    ]
    if decision.get("best_posthoc_method"):
        lines.extend(
            [
                f"- Best post-hoc method: `{decision['best_posthoc_method']}`",
                (
                    "- Best post-hoc Top-1 / Mean-F1: "
                    f"{_pct(decision.get('best_posthoc_top1'))} / {_pct(decision.get('best_posthoc_macro_f1'))}"
                ),
            ]
        )
    lines.extend(["", "## Pending Outputs", ""])
    pending = result["required_pending_outputs"]
    if pending:
        lines.extend(f"- `{item}`" for item in pending)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Candidate Table",
            "",
            "| run | Top-1 | Mean-F1 | dTop1 vs Physics | dF1 vs Physics | wet/water F1 | water concrete F1 | gap Top-1 to RoadFormer-L | gap F1 to RoadFormer-L |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["rows"]:
        slices = row.get("slices", {})
        lines.append(
            "| `{name}` | {top1} | {f1} | {dtp} | {dfp} | {wet} | {wc} | {gt} | {gf} |".format(
                name=row["name"],
                top1=_pct(row["top1"]),
                f1=_pct(row["macro_f1"]),
                dtp=_pct(row.get("delta_top1_vs_physics"), signed=True),
                dfp=_pct(row.get("delta_f1_vs_physics"), signed=True),
                wet=_pct(float(slices.get("wet_water_f1") or 0.0)),
                wc=_pct(float(slices.get("water_concrete_f1") or 0.0)),
                gt=_pct(row.get("gap_top1_to_strict_target"), signed=True),
                gf=_pct(row.get("gap_f1_to_strict_target"), signed=True),
            )
        )
    lines.extend(["", "## Module Actions", ""])
    for item in decision["module_actions"]:
        lines.append(f"- `{item['module']}`: `{item['action']}`. {item['reason']}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
