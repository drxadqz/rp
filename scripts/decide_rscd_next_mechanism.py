from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FULL_TEST_SAMPLES = 49_500
SOTA_TOP1 = 0.9286
SOTA_MACRO_F1 = 0.8949
KEY_CLASSES = {
    "water_concrete_slight",
    "water_concrete_severe",
    "water_concrete_smooth",
    "wet_concrete_slight",
    "wet_concrete_severe",
    "wet_concrete_smooth",
    "dry_concrete_slight",
    "dry_concrete_severe",
    "water_asphalt_slight",
}


@dataclass
class Decision:
    action: str
    reason: str
    mechanism_target: str
    promote_full: bool
    full_sota_pass: bool


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_diagnosis(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = read_json(path)
    if payload is None:
        return None
    model_keys = ["logistic_raw", "logistic_value_augmented", "random_forest_values", "hist_gradient_values"]
    model_scores = []
    for key in model_keys:
        item = payload.get(key)
        if not isinstance(item, dict):
            continue
        model_scores.append(
            {
                "model": key,
                "target_true_accuracy": float(item.get("target_true_accuracy", 0.0)),
                "target_true_macro_f1": float(item.get("target_true_macro_f1", 0.0)),
            }
        )
    best_model = max(model_scores, key=lambda row: row["target_true_macro_f1"], default=None)
    reranks = list(payload.get("rerank_thresholds") or [])
    best_rerank = max(reranks, key=lambda row: float(row.get("full_macro_f1", 0.0)), default=None)
    base_macro = float(payload.get("rerank_base_full_macro_f1", 0.0) or 0.0)
    base_top1 = float(payload.get("rerank_base_full_accuracy", 0.0) or 0.0)
    return {
        "path": str(path),
        "num_target_classes": len(payload.get("target_classes") or []),
        "selected_features": list(payload.get("selected_augmented_features") or []),
        "model_scores": model_scores,
        "best_target_model": best_model,
        "best_target_macro_f1": float(best_model["target_true_macro_f1"]) if best_model else None,
        "rerank_base_full_accuracy": base_top1,
        "rerank_base_full_macro_f1": base_macro,
        "best_rerank": best_rerank,
        "late_rerank_hurts_macro_f1": bool(best_rerank and float(best_rerank.get("full_macro_f1", 0.0)) < base_macro),
    }


def read_summary(run_dir: Path) -> dict[str, Any] | None:
    payload = read_json(run_dir / "test_metrics.json")
    if payload is None:
        return None
    return dict(payload.get("summary", payload))


def read_per_class(run_dir: Path) -> dict[str, dict[str, float]]:
    path = run_dir / "per_class_metrics.csv"
    if not path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = [str(name).lstrip("\ufeff") for name in (reader.fieldnames or [])]
        class_field = "class" if "class" in fields else (fields[0] if fields else "class")
        for row in reader:
            if class_field not in row and f"\ufeff{class_field}" in row:
                class_field = f"\ufeff{class_field}"
            label = str(row.get(class_field, ""))
            if not label:
                continue
            out[label] = {
                "precision": float(row.get("precision") or 0.0),
                "recall": float(row.get("recall") or 0.0),
                "f1": float(row.get("f1") or 0.0),
                "support": float(row.get("support") or 0.0),
            }
    return out


def read_predictions(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "predictions_test.csv"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = str(row.get("true_label", ""))
            pred_label = str(row.get("pred_label", ""))
            out.append(
                {
                    "image_path": str(row.get("image_path", "")),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": float(row.get("confidence") or 0.0),
                    "correct": true_label == pred_label,
                    "true_factors": parse_label(true_label),
                    "pred_factors": parse_label(pred_label),
                }
            )
    return out


def parse_label(label: str) -> dict[str, str]:
    parts = label.split("_")
    if len(parts) == 3 and parts[0] in {"dry", "wet", "water"}:
        friction, material, roughness = parts
    elif len(parts) == 2 and parts[0] in {"dry", "wet", "water"}:
        friction, material = parts
        roughness = "nonparam"
    elif label == "fresh_snow":
        friction, material, roughness = "snow_ice", "snow", "fresh"
    elif label == "melted_snow":
        friction, material, roughness = "snow_ice", "snow", "melted"
    elif label == "ice":
        friction, material, roughness = "snow_ice", "ice", "ice"
    else:
        friction, material, roughness = "unknown", label, "unknown"
    return {
        "friction": friction,
        "material": material,
        "roughness": roughness,
        "friction_material": f"{friction}_{material}",
        "material_roughness": f"{material}_{roughness}",
    }


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.3f}%"


def pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:+.3f} pp"


def metric_value(payload: dict[str, Any] | None, key: str) -> float | None:
    if not payload or key not in payload or payload.get(key) is None:
        return None
    return float(payload[key])


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, float]:
    if baseline is None:
        return {}
    return {
        key: float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0))
        for key in ["top1", "macro_f1", "mean_precision", "mean_recall", "weighted_f1"]
    }


def top_class_deltas(
    candidate: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(set(candidate) | set(baseline)):
        c = candidate.get(name, {})
        b = baseline.get(name, {})
        rows.append(
            {
                "class": name,
                "candidate_f1": float(c.get("f1", 0.0)),
                "baseline_f1": float(b.get("f1", 0.0)),
                "delta_f1": float(c.get("f1", 0.0)) - float(b.get("f1", 0.0)),
                "support": float(c.get("support", b.get("support", 0.0))),
                "key": name in KEY_CLASSES,
                **parse_label(name),
            }
        )
    return rows


def factor_pressure(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    pressure: dict[str, Any] = {}
    errors = [row for row in predictions if not row["correct"]]
    for axis in ["friction", "material", "roughness", "friction_material", "material_roughness"]:
        wrong: Counter[str] = Counter()
        confusion: Counter[tuple[str, str]] = Counter()
        for row in errors:
            true_value = row["true_factors"][axis]
            pred_value = row["pred_factors"][axis]
            if true_value != pred_value:
                wrong[true_value] += 1
                confusion[(true_value, pred_value)] += 1
        pressure[axis] = {
            "wrong_by_true": [{"factor": k, "count": int(v)} for k, v in wrong.most_common(10)],
            "top_confusions": [
                {"true": k[0], "pred": k[1], "count": int(v)}
                for k, v in confusion.most_common(12)
            ],
        }
    return pressure


def transfer_counts(candidate_dir: Path, baseline_dir: Path | None) -> dict[str, Any]:
    if baseline_dir is None:
        return {}
    cand = {row["image_path"]: row for row in read_predictions(candidate_dir)}
    base = {row["image_path"]: row for row in read_predictions(baseline_dir)}
    fixed = Counter()
    worsened = Counter()
    for path in sorted(set(cand) & set(base)):
        c = cand[path]
        b = base[path]
        if c["true_label"] != b["true_label"]:
            continue
        c_ok = bool(c["correct"])
        b_ok = bool(b["correct"])
        if c_ok and not b_ok:
            fixed[c["true_label"]] += 1
        elif b_ok and not c_ok:
            worsened[c["true_label"]] += 1
    return {
        "fixed": int(sum(fixed.values())),
        "worsened": int(sum(worsened.values())),
        "net_fixed": int(sum(fixed.values()) - sum(worsened.values())),
        "fixed_by_class": [{"class": k, "count": int(v)} for k, v in fixed.most_common(12)],
        "worsened_by_class": [{"class": k, "count": int(v)} for k, v in worsened.most_common(12)],
    }


def top1_error_budget(summary: dict[str, Any], target: float) -> dict[str, Any]:
    samples = int(float(summary.get("num_samples", 0) or 0))
    top1 = float(summary.get("top1", 0.0))
    current_correct = int(round(top1 * samples))
    target_correct = int(math.ceil(target * samples))
    return {
        "samples": samples,
        "current_top1": top1,
        "target_top1": target,
        "current_correct_est": current_correct,
        "target_correct_min": target_correct,
        "required_extra_correct": max(0, target_correct - current_correct),
    }


def mechanism_blueprint(target: str, feature_diagnosis: dict[str, Any] | None = None) -> dict[str, Any]:
    common_gates = {
        "screen_success": [
            "Top-1 must not decrease versus the same-budget screen baseline.",
            "Macro-F1 must not decrease versus the same-budget screen baseline.",
            "No key coupled class may drop by more than 0.5 pp F1.",
            "No non-key class may drop by more than 1.5 pp F1.",
        ],
        "full_success": [
            "Full protocol must use the complete train/val/test manifests.",
            "Full-test Top-1 must reach at least 92.86%.",
            "Full-test Macro-F1 must reach at least 89.49%.",
            "Worst-class F1 and water_concrete_slight F1 must be reported explicitly.",
        ],
        "same_budget_control": [
            "Keep the same manifest caps, epochs, image size, batch size, and optimizer.",
            "Ablate only the claimed mechanism.",
            "Use a fixed or identity gate control when the mechanism is a learned gate.",
        ],
    }
    if target == "early_concrete_roughness_scale_space_expert":
        return {
            "target": target,
            "rscc_factor_target": [
                "material_roughness: concrete_slight and concrete_severe",
                "roughness axis: slight versus severe versus smooth",
                "secondary guard: water/wet concrete film should not be damaged",
            ],
            "first_principle": (
                "The dominant measured failure is not generic texture recognition; it is a concrete-conditioned "
                "roughness boundary. Concrete hides small height/texture changes under low-contrast wet/water film, "
                "so the backbone should sense multi-scale roughness before global pooling and condition that evidence "
                "on concrete likelihood."
            ),
            "early_mechanism": [
                "Compute gray/texture evidence at multiple scales inside the stem or stage-1 feature flow.",
                "Use gradient, Laplacian, and local-contrast energies after small/medium smoothing scales.",
                "Create a concrete-conditioned roughness gate from concrete proxy, wet/water film proxy, and scale-space texture contrast.",
                "Inject the gate as early FiLM/depthwise modulation into low-level feature maps, not as a late classifier head.",
                "Use separate gates for dry-concrete and wet/water-concrete because their visual coupling is different.",
            ],
            "candidate_name_hint": "S137_early_concrete_roughness_scale_space_expert",
            "diagnostic_pairs": [
                "dry_concrete_slight -> dry_concrete_severe",
                "dry_concrete_severe -> dry_concrete_slight",
                "water_concrete_slight -> water_concrete_severe",
                "wet_concrete_severe -> wet_concrete_slight",
            ],
            **common_gates,
        }
    if target == "early_dual_film_texture_roughness_coupling_backbone":
        selected_features = []
        if feature_diagnosis:
            selected_features = list(feature_diagnosis.get("selected_augmented_features") or [])[:12]
        return {
            "target": target,
            "rscc_factor_target": [
                "roughness axis: concrete_smooth / concrete_slight / concrete_severe",
                "friction-material axis: wet_concrete and water_concrete",
                "coupling form: dry concrete roughness is visible texture, but wet/water concrete roughness is texture hidden under film",
                "guard: asphalt water/wet classes must not be pulled into the concrete specialist",
            ],
            "first_principle": (
                "The S7 feature-value diagnosis shows that the separable evidence is real but too weak for a late reranker: "
                "hand-crafted target-class classifiers reached only about 52% Macro-F1 and hurt full-test Top-1/Macro-F1 when used after the classifier. "
                "Therefore the next valid route is an early backbone mechanism that learns two different coupling laws: "
                "visible meso/gradient roughness for dry concrete, and film-erased texture plus wet-connectedness for wet/water concrete."
            ),
            "early_mechanism": [
                "Split the stem into two task-conditioned early experts rather than a single uniform roughness gate.",
                "Expert A handles dry-concrete roughness using meso-scale contrast, gradient dispersion, and Laplacian texture.",
                "Expert B handles wet/water-concrete coupling using film-erasure, wet-connectedness, specular/dark-water evidence, and texture-under-film residuals.",
                "A learned factor router chooses the expert mixture from concrete likelihood and film evidence at stage 0/1.",
                "The two expert outputs are recombined with no-spill residual bounds so reliable asphalt/snow/ice classes keep their existing feature path.",
                "The module must modulate early feature maps by FiLM/depthwise gates; it must not be implemented as a late feature-value reranker.",
            ],
            "candidate_name_hint": "S138_early_dual_film_texture_roughness_coupling_backbone",
            "diagnostic_pairs": [
                "dry_concrete_smooth <-> dry_concrete_slight",
                "dry_concrete_slight <-> dry_concrete_severe",
                "water_concrete_slight <-> water_concrete_severe",
                "wet_concrete_slight <-> wet_concrete_severe",
                "water_concrete_slight <-> wet_concrete_slight",
            ],
            "feature_diagnosis_support": selected_features,
            **common_gates,
        }
    if target == "early_wet_water_film_concrete_coupling_expert":
        return {
            "target": target,
            "rscc_factor_target": [
                "friction_material: wet_concrete and water_concrete",
                "friction axis: wet versus water",
                "secondary guard: roughness slight/severe should not collapse",
            ],
            "first_principle": (
                "Wet and water labels differ by film thickness, specular continuity, and texture occlusion. "
                "This is a physics-coupled visual state, not a simple class boundary. The model should compare "
                "specular/dark-film evidence against visible microtexture before the global semantic decision."
            ),
            "early_mechanism": [
                "Estimate film evidence from dark-water, specular, saturation, and texture-under-film cues.",
                "Condition early features on concrete likelihood because concrete changes wet/water reflectance differently from asphalt.",
                "Use an opponent wet-vs-water gate that enhances features where film evidence suppresses texture.",
                "Keep roughness evidence in a parallel branch so film cues do not erase slight/severe distinctions.",
            ],
            "candidate_name_hint": "S137_early_wet_water_film_concrete_coupling_expert",
            "diagnostic_pairs": [
                "wet_concrete_smooth -> water_concrete_smooth",
                "water_concrete_smooth -> wet_concrete_smooth",
                "water_concrete_slight -> wet_concrete_slight",
                "wet_concrete_severe -> water_concrete_severe",
            ],
            **common_gates,
        }
    if target == "top1_no_spill":
        return {
            "target": target,
            "rscc_factor_target": [
                "all already reliable classes",
                "key coupled weak classes remain protected",
            ],
            "first_principle": (
                "A route that lifts low-F1 classes can still fail the paper-level Top-1 target if it spills errors into "
                "large-support easy classes. The mechanism should explicitly preserve high-confidence correct decisions."
            ),
            "early_mechanism": [
                "Use classwise no-regression gates from the baseline teacher only on high-confidence non-focus examples.",
                "Do not freeze all logits; protect easy classes while allowing weak concrete/wet/water boundaries to move.",
                "Audit fixed-versus-worsened predictions per class before considering full training.",
            ],
            "candidate_name_hint": "S137_classwise_no_spill_guard",
            "diagnostic_pairs": [
                "fixed/worsened prediction transfer by class",
                "high-confidence errors introduced into dry/asphalt/snow classes",
            ],
            **common_gates,
        }
    if target == "weak_class_macro_f1":
        return {
            "target": target,
            "rscc_factor_target": [
                "lowest-F1 coupled classes",
                "water_concrete_slight, wet_concrete_slight, water_concrete_severe",
            ],
            "first_principle": (
                "Top-1 can improve while Macro-F1 worsens when common classes dominate the loss. "
                "The mechanism should rebalance factor-coupled evidence without changing the full-data protocol."
            ),
            "early_mechanism": [
                "Use factor-aware sampling or loss weighting only around low-F1 coupled boundaries.",
                "Tie the weights to measured per-class F1 and pair confusions, not manual cherry-picking.",
                "Keep a no-spill audit against the baseline to prevent easy-class damage.",
            ],
            "candidate_name_hint": "S137_factor_balanced_weak_class_guard",
            "diagnostic_pairs": [
                "water_concrete_slight boundary family",
                "wet_concrete_slight boundary family",
            ],
            **common_gates,
        }
    return {
        "target": target,
        "rscc_factor_target": [
            "friction/material/roughness coupling remains broad",
            "no single factor dominates the measured failure",
        ],
        "first_principle": (
            "If no single factor family dominates, the next architecture should revise early multi-factor evidence "
            "rather than adding another late classifier correction."
        ),
        "early_mechanism": [
            "Keep PhysicsTexture and LocalPhysicsField evidence explicit.",
            "Add early/mid feature conditioning only where factor evidence is measured to be unstable.",
            "Use same-budget controls and per-factor confusion deltas.",
        ],
        "candidate_name_hint": "S137_early_multifactor_evidence_experts",
        "diagnostic_pairs": [
            "top factor confusions from next_mechanism_decision.json",
            "top class drops and gains from same-budget comparison",
        ],
        **common_gates,
    }


def low_class_pressure(per_class: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, stats in per_class.items():
        rows.append(
            {
                "class": name,
                "f1": float(stats.get("f1", 0.0)),
                "precision": float(stats.get("precision", 0.0)),
                "recall": float(stats.get("recall", 0.0)),
                "support": float(stats.get("support", 0.0)),
                "key": name in KEY_CLASSES,
                **parse_label(name),
            }
        )
    return sorted(rows, key=lambda row: row["f1"])


def decide(
    *,
    candidate_name: str,
    candidate_dir: Path,
    protocol: str,
    candidate_summary: dict[str, Any],
    baseline_summary: dict[str, Any] | None,
    candidate_classes: dict[str, dict[str, float]],
    baseline_classes: dict[str, dict[str, float]],
    predictions: list[dict[str, Any]],
    transfer: dict[str, Any],
) -> Decision:
    top1 = float(candidate_summary.get("top1", 0.0))
    macro_f1 = float(candidate_summary.get("macro_f1", 0.0))
    samples = int(float(candidate_summary.get("num_samples", 0) or 0))
    full_sota_pass = samples == FULL_TEST_SAMPLES and top1 >= SOTA_TOP1 and macro_f1 >= SOTA_MACRO_F1
    if full_sota_pass:
        return Decision(
            action="final_verify_and_write",
            reason="Full-test candidate clears both public Top-1 and Macro-F1 SOTA thresholds.",
            mechanism_target="completed_candidate",
            promote_full=False,
            full_sota_pass=True,
        )

    delta = metric_delta(candidate_summary, baseline_summary)
    class_delta = top_class_deltas(candidate_classes, baseline_classes) if baseline_classes else []
    key_drops = [row for row in class_delta if row["key"] and row["delta_f1"] < -0.005]
    large_drops = [row for row in class_delta if row["delta_f1"] < -0.015]
    net_fixed = int(transfer.get("net_fixed", 0)) if transfer else 0
    no_spill_ok = not key_drops and not large_drops and net_fixed >= 0

    if protocol == "screen" and baseline_summary is not None:
        if delta.get("top1", -1.0) >= 0.0 and delta.get("macro_f1", -1.0) >= 0.0 and no_spill_ok:
            return Decision(
                action="promote_screen_to_full",
                reason="Screen candidate improves Top-1 and Macro-F1 over its screen baseline without key-class or no-spill regressions.",
                mechanism_target="active_candidate",
                promote_full=True,
                full_sota_pass=False,
            )
        if delta.get("macro_f1", 0.0) > 0.0 and delta.get("top1", 0.0) < 0.0:
            return Decision(
                action="add_no_spill_or_calibration_before_full",
                reason="Screen improves balanced F1 but loses Top-1, indicating weak-class repair is spilling into reliable classes.",
                mechanism_target="top1_no_spill",
                promote_full=False,
                full_sota_pass=False,
            )
        if delta.get("top1", 0.0) > 0.0 and delta.get("macro_f1", 0.0) < 0.0:
            return Decision(
                action="rebalance_weak_coupled_classes",
                reason="Screen improves global accuracy but hurts Macro-F1, so the mechanism is protecting easy classes while weakening low-F1 coupled classes.",
                mechanism_target="weak_class_macro_f1",
                promote_full=False,
                full_sota_pass=False,
            )

    pressure = factor_pressure(predictions)
    rough_count = sum(item["count"] for item in pressure["roughness"]["wrong_by_true"][:2])
    friction_count = sum(item["count"] for item in pressure["friction"]["wrong_by_true"][:2])
    material_roughness_top = {item["factor"]: item["count"] for item in pressure["material_roughness"]["wrong_by_true"]}
    low_classes = low_class_pressure(candidate_classes)
    bottom_names = {row["class"] for row in low_classes[:6]}
    is_s137 = "s137" in candidate_name.lower() or "s137_" in str(candidate_dir).lower()

    if is_s137 and not full_sota_pass:
        return Decision(
            action="design_next_custom_backbone",
            reason=(
                "S137 was already the single concrete roughness scale-space route. If it fails promotion/SOTA gates, "
                "the next route should not repeat the same mechanism; use a dual early expert that separately learns "
                "dry-concrete visible roughness and wet/water-concrete film-texture coupling."
            ),
            mechanism_target="early_dual_film_texture_roughness_coupling_backbone",
            promote_full=False,
            full_sota_pass=False,
        )

    if material_roughness_top.get("concrete_slight", 0) + material_roughness_top.get("concrete_severe", 0) >= max(1, rough_count // 2):
        return Decision(
            action="design_next_custom_backbone",
            reason="Dominant pressure is concrete slight/severe roughness, matching the measured RSCD failure mode.",
            mechanism_target="early_concrete_roughness_scale_space_expert",
            promote_full=False,
            full_sota_pass=False,
        )
    if {"water_concrete_slight", "wet_concrete_slight", "water_concrete_severe"} & bottom_names or friction_count >= rough_count:
        return Decision(
            action="design_next_custom_backbone",
            reason="Dominant pressure is wet/water concrete film ambiguity and low-F1 water/wet concrete coupled labels.",
            mechanism_target="early_wet_water_film_concrete_coupling_expert",
            promote_full=False,
            full_sota_pass=False,
        )
    return Decision(
        action="revise_early_evidence_experts",
        reason="Candidate does not clear promotion/SOTA gates; revise task-specific early evidence extraction rather than adding a late head.",
        mechanism_target="early_multifactor_evidence_experts",
        promote_full=False,
        full_sota_pass=False,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    decision = payload["decision"]
    candidate = payload["candidate_summary"]
    baseline = payload.get("baseline_summary") or {}
    delta = payload.get("summary_delta") or {}
    lines = [
        "# RSCD Next Mechanism Decision",
        "",
        f"- Candidate: `{payload['candidate_name']}`",
        f"- Candidate dir: `{payload['candidate_dir']}`",
        f"- Protocol: `{payload['protocol']}`",
        f"- Action: `{decision['action']}`",
        f"- Mechanism target: `{decision['mechanism_target']}`",
        f"- Promote full: `{decision['promote_full']}`",
        f"- Full SOTA pass: `{decision['full_sota_pass']}`",
        f"- Reason: {decision['reason']}",
        "",
        "## Metrics",
        "",
        "| Metric | Candidate | Baseline | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in ["top1", "macro_f1", "mean_precision", "mean_recall", "weighted_f1"]:
        lines.append(
            f"| {key} | {pct(metric_value(candidate, key))} | "
            f"{pct(metric_value(baseline, key))} | "
            f"{pp(metric_value(delta, key))} |"
        )
    budget = payload.get("top1_budget") or {}
    if budget:
        lines.extend(
            [
                "",
                "## Top-1 SOTA Budget",
                "",
                f"- Full-test samples: `{budget['samples']}`",
                f"- Required extra correct predictions: `{budget['required_extra_correct']}`",
            ]
        )
    feature = payload.get("feature_diagnosis") or {}
    if feature:
        lines.extend(["", "## Feature-Value Diagnosis Evidence", ""])
        lines.append(f"- Source: `{feature.get('path', '-')}`")
        lines.append(f"- Target classes: `{feature.get('num_target_classes', '-')}`")
        lines.append(f"- Best target-class Macro-F1 from hand-crafted values: `{pct(feature.get('best_target_macro_f1'))}`")
        lines.append(f"- Base full-test Macro-F1 before rerank: `{pct(feature.get('rerank_base_full_macro_f1'))}`")
        best_rerank = feature.get("best_rerank") or {}
        if best_rerank:
            lines.append(
                "- Best rerank threshold still gives Top-1 `{}` and Macro-F1 `{}`; this is negative evidence for late reranking.".format(
                    pct(best_rerank.get("full_accuracy")),
                    pct(best_rerank.get("full_macro_f1")),
                )
            )
        top_features = feature.get("selected_features") or []
        if top_features:
            lines.append(f"- Top diagnosis features: `{', '.join(top_features[:12])}`")
    lines.extend(["", "## Lowest-F1 Classes", "", "| Class | F1 | P | R | Factor |", "|---|---:|---:|---:|---|"])
    for row in payload["lowest_classes"][:10]:
        lines.append(
            f"| {row['class']} | {pct(row['f1'])} | {pct(row['precision'])} | {pct(row['recall'])} | "
            f"{row['friction']} + {row['material']} + {row['roughness']} |"
        )
    lines.extend(["", "## Factor Pressure", ""])
    pressure = payload["factor_pressure"]
    for axis in ["roughness", "friction", "material_roughness", "friction_material"]:
        lines.extend([f"### {axis}", "", "| True factor | Errors |", "|---|---:|"])
        for row in pressure[axis]["wrong_by_true"][:8]:
            lines.append(f"| {row['factor']} | {row['count']} |")
        lines.append("")
    blueprint = payload.get("blueprint") or {}
    if blueprint:
        lines.extend(
            [
                "## Task-Adapted Mechanism Blueprint",
                "",
                f"- Candidate name hint: `{blueprint.get('candidate_name_hint', '-')}`",
                f"- First-principle rationale: {blueprint.get('first_principle', '-')}",
                "",
                "### Targeted RSCD Factors",
                "",
            ]
        )
        for item in blueprint.get("rscc_factor_target", []):
            lines.append(f"- {item}")
        lines.extend(["", "### Early/Mid Mechanism", ""])
        for item in blueprint.get("early_mechanism", []):
            lines.append(f"- {item}")
        lines.extend(["", "### Diagnostic Pairs", ""])
        for item in blueprint.get("diagnostic_pairs", []):
            lines.append(f"- `{item}`")
        lines.extend(["", "### Same-Budget Control", ""])
        for item in blueprint.get("same_budget_control", []):
            lines.append(f"- {item}")
        lines.extend(["", "### Screen Success Gate", ""])
        for item in blueprint.get("screen_success", []):
            lines.append(f"- {item}")
        lines.extend(["", "### Full Success Gate", ""])
        for item in blueprint.get("full_success", []):
            lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide the next RSCD mechanism after a candidate run lands.")
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--protocol", choices=["screen", "full"], default="screen")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--feature-diagnosis", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_summary = read_summary(args.candidate_dir)
    candidate_classes = read_per_class(args.candidate_dir)
    candidate_predictions = read_predictions(args.candidate_dir)
    baseline_summary = read_summary(args.baseline_dir) if args.baseline_dir else None
    baseline_classes = read_per_class(args.baseline_dir) if args.baseline_dir else {}

    missing: list[str] = []
    if candidate_summary is None:
        missing.append(str(args.candidate_dir / "test_metrics.json"))
    if not candidate_classes:
        missing.append(str(args.candidate_dir / "per_class_metrics.csv"))
    if not candidate_predictions:
        missing.append(str(args.candidate_dir / "predictions_test.csv"))
    if args.baseline_dir and baseline_summary is None:
        missing.append(str(args.baseline_dir / "test_metrics.json"))
    if missing:
        payload = {
            "ok": False,
            "status": "pending",
            "candidate_name": args.candidate_name,
            "candidate_dir": str(args.candidate_dir),
            "missing": missing,
            "decision": asdict(
                Decision(
                    action="wait_for_candidate_outputs",
                    reason="Candidate or baseline artifacts are not complete yet.",
                    mechanism_target="pending",
                    promote_full=False,
                    full_sota_pass=False,
                )
            ),
        }
        (args.output_dir / "next_mechanism_decision.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_markdown(
            {
                **payload,
                "protocol": args.protocol,
                "candidate_summary": {},
                "baseline_summary": {},
                "summary_delta": {},
                "top1_budget": {},
                "lowest_classes": [],
                "factor_pressure": {
                    "roughness": {"wrong_by_true": []},
                    "friction": {"wrong_by_true": []},
                    "material_roughness": {"wrong_by_true": []},
                    "friction_material": {"wrong_by_true": []},
                },
            },
            args.output_dir / "next_mechanism_decision.md",
        )
        print(json.dumps({"ok": False, "status": "pending", "report": str(args.output_dir / "next_mechanism_decision.md")}, ensure_ascii=False))
        return 0

    transfer = transfer_counts(args.candidate_dir, args.baseline_dir) if args.baseline_dir else {}
    feature_diagnosis_payload = load_feature_diagnosis(args.feature_diagnosis)
    decision = decide(
        candidate_name=args.candidate_name,
        candidate_dir=args.candidate_dir,
        protocol=args.protocol,
        candidate_summary=candidate_summary,
        baseline_summary=baseline_summary,
        candidate_classes=candidate_classes,
        baseline_classes=baseline_classes,
        predictions=candidate_predictions,
        transfer=transfer,
    )
    class_deltas = top_class_deltas(candidate_classes, baseline_classes) if baseline_classes else []
    lowest = low_class_pressure(candidate_classes)
    pressure = factor_pressure(candidate_predictions)
    payload = {
        "ok": True,
        "candidate_name": args.candidate_name,
        "candidate_dir": str(args.candidate_dir),
        "baseline_name": args.baseline_name if args.baseline_dir else None,
        "baseline_dir": str(args.baseline_dir) if args.baseline_dir else None,
        "protocol": args.protocol,
        "decision": asdict(decision),
        "blueprint": mechanism_blueprint(decision.mechanism_target, feature_diagnosis_payload),
        "feature_diagnosis": feature_diagnosis_payload,
        "candidate_summary": candidate_summary,
        "baseline_summary": baseline_summary,
        "summary_delta": metric_delta(candidate_summary, baseline_summary),
        "top1_budget": top1_error_budget(candidate_summary, SOTA_TOP1),
        "transfer": transfer,
        "lowest_classes": lowest,
        "factor_pressure": pressure,
        "top_class_drops": sorted(class_deltas, key=lambda row: row["delta_f1"])[:15],
        "top_class_gains": sorted(class_deltas, key=lambda row: row["delta_f1"], reverse=True)[:15],
    }
    (args.output_dir / "next_mechanism_decision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "lowest_classes.csv", lowest)
    write_csv(args.output_dir / "top_class_drops.csv", payload["top_class_drops"])
    write_csv(args.output_dir / "top_class_gains.csv", payload["top_class_gains"])
    write_markdown(payload, args.output_dir / "next_mechanism_decision.md")
    print(
        json.dumps(
            {
                "ok": True,
                "action": decision.action,
                "mechanism_target": decision.mechanism_target,
                "report": str(args.output_dir / "next_mechanism_decision.md"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
