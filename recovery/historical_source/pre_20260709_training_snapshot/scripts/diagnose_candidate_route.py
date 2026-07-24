from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


KEY_CLASSES = [
    "water_concrete_slight",
    "water_concrete_severe",
    "water_concrete_smooth",
    "wet_concrete_slight",
    "wet_concrete_severe",
    "dry_concrete_slight",
]

SOTA_TARGETS = {
    "top1": 0.9286,
    "macro_f1": 0.8949,
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summary(run_dir: Path) -> dict[str, Any] | None:
    payload = _read_json(run_dir / "test_metrics.json")
    if payload is None:
        return None
    return dict(payload.get("summary", payload))


def _read_per_class(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
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
            rows[label] = {
                "precision": float(row.get("precision") or 0.0),
                "recall": float(row.get("recall") or 0.0),
                "f1": float(row.get("f1") or 0.0),
                "support": float(row.get("support") or 0.0),
            }
    return rows


def _parse_label(label: str) -> dict[str, str]:
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
        "friction_roughness": f"{friction}_{roughness}",
        "material_roughness": f"{material}_{roughness}",
    }


def _pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.3f}%"


def _pp(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:+.3f} pp"


def _float(summary: dict[str, Any] | None, key: str) -> float | None:
    if not summary or summary.get(key) is None:
        return None
    try:
        return float(summary[key])
    except (TypeError, ValueError):
        return None


def _class_deltas(candidate: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in sorted(set(candidate) | set(baseline)):
        c = candidate.get(label, {})
        b = baseline.get(label, {})
        row = {
            "class": label,
            "candidate_f1": float(c.get("f1", 0.0)),
            "baseline_f1": float(b.get("f1", 0.0)),
            "delta_f1": float(c.get("f1", 0.0)) - float(b.get("f1", 0.0)),
            "candidate_precision": float(c.get("precision", 0.0)),
            "baseline_precision": float(b.get("precision", 0.0)),
            "candidate_recall": float(c.get("recall", 0.0)),
            "baseline_recall": float(b.get("recall", 0.0)),
            "support": float(c.get("support", b.get("support", 0.0))),
            **_parse_label(label),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["delta_f1"])
    return rows


def _factor_damage(rows: list[dict[str, Any]], factor_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(factor_name, "unknown"))].append(float(row["delta_f1"]))
    out = [
        {
            "factor": factor,
            "mean_delta_f1": sum(values) / len(values),
            "min_delta_f1": min(values),
            "num_classes": len(values),
        }
        for factor, values in grouped.items()
    ]
    return sorted(out, key=lambda row: row["mean_delta_f1"])


def _stable_cue_top(stable_path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if stable_path is None or not stable_path.exists():
        return {}
    payload = _read_json(stable_path)
    if not payload:
        return {}
    top_by_pair = payload.get("top_by_pair", {})
    out: dict[str, list[dict[str, Any]]] = {}
    for pair, rows in top_by_pair.items():
        out[pair] = [
            {
                "feature": row.get("feature"),
                "family": row.get("family"),
                "mean_abs_cohen_d": row.get("mean_abs_cohen_d"),
                "mean_signed_cohen_d": row.get("mean_signed_cohen_d"),
                "sign_consistency": row.get("sign_consistency"),
            }
            for row in rows[:4]
        ]
    return out


def _load_comparison(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if path.is_dir():
        path = path / "run_comparison.json"
    return _read_json(path)


def _load_sota_gap(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if path.is_dir():
        path = path / "sota_gap_budget.json"
    return _read_json(path)


def _failed_checks(decision: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not decision:
        return []
    return [dict(item) for item in decision.get("checks", []) if not bool(item.get("pass"))]


def _route_status(
    candidate_summary: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    full_protocol_required: bool,
) -> str:
    if candidate_summary is None:
        return "pending"
    if decision and bool(decision.get("promote_to_full")):
        return "promote_to_full"
    if decision and not bool(decision.get("promote_to_full")):
        return "reject_or_redesign"
    samples = _float(candidate_summary, "num_samples")
    top1 = _float(candidate_summary, "top1")
    macro = _float(candidate_summary, "macro_f1")
    if full_protocol_required and (samples is None or samples < 40000):
        return "screen_only_waiting_for_decision"
    if top1 is not None and macro is not None and top1 >= SOTA_TARGETS["top1"] and macro >= SOTA_TARGETS["macro_f1"]:
        return "strict_sota_candidate"
    return "needs_decision"


def _next_route(
    status: str,
    failed: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
    sota_gap: dict[str, Any] | None,
) -> dict[str, Any]:
    top1_budget = (sota_gap or {}).get("top1_budget", {}) if sota_gap else {}
    macro_uplift = (sota_gap or {}).get("macro_uplift", {}) if sota_gap else {}
    required_extra = top1_budget.get("required_extra_correct")
    required_error_share = top1_budget.get("required_error_reduction_share")
    uplift_plan = macro_uplift.get("plan", []) if isinstance(macro_uplift, dict) else []
    pressure_text = ""
    if required_extra is not None:
        pressure_text = (
            f" SOTA pressure: this route still needs about {int(required_extra)} extra correct predictions"
            + (
                f" ({100.0 * float(required_error_share):.1f}% of current errors) without adding new errors."
                if required_error_share is not None
                else "."
            )
        )
    weakest_targets = [str(item.get("class")) for item in uplift_plan[:3] if item.get("class")]
    if status == "pending":
        return {
            "decision": "wait",
            "reason": "Candidate metrics are not available yet.",
            "single_next_route": None,
        }
    if status == "promote_to_full":
        return {
            "decision": "promote",
            "reason": "The screen passed conservative promotion gates; full RSCD is the next fair test." + pressure_text,
            "single_next_route": "Run the existing full config only; do not branch until full metrics are known.",
        }
    if status == "strict_sota_candidate":
        return {
            "decision": "verify_full_protocol",
            "reason": "Metrics meet strict SOTA targets; verify protocol, predictions, and per-class outputs before claiming.",
            "single_next_route": "Run fair SOTA audit and reproducibility snapshot, then freeze the result.",
        }

    failed_names = {str(item.get("name", "")) for item in failed}
    key_rows = {row["class"]: row for row in class_rows}
    wcs_delta = key_rows.get("water_concrete_slight", {}).get("delta_f1")
    wcsev_delta = key_rows.get("water_concrete_severe", {}).get("delta_f1")
    macro_failed = "macro_f1_gain" in failed_names
    top1_failed = "top1_no_material_drop" in failed_names
    hard_failed = "hard_class_mean_no_spill" in failed_names
    wcs_failed = "water_concrete_slight_f1_gain" in failed_names
    spill_failed = "water_concrete_severe_no_spill" in failed_names or hard_failed or top1_failed

    top_drops = class_rows[:5]
    top_gains = sorted(class_rows, key=lambda row: row["delta_f1"], reverse=True)[:5]
    net_fixed = None
    if comparison:
        net_fixed = comparison.get("net_fixed")

    if wcs_failed and spill_failed:
        route = (
            "Discard this exact visibility-gate strength. Next single route should be an early factor-conditioned "
            "custom backbone block: first estimate material confidence and water-film strength from PhysicsTexture, "
            "then let separate roughness filters operate only inside water+concrete, wet+concrete, and dry+concrete "
            "subspaces. The key change is decoupling the coupling form instead of applying one water-concrete gate to all hard pairs. "
            "Add a no-spill objective over the highest-support error-pressure classes so the Top-1 budget is improved, not merely redistributed."
        )
        reason = (
            "The target class did not clear its required gain and protection gates were also hit, so simply amplifying "
            "S135c would likely trade one class for another." + pressure_text
        )
    elif wcs_failed:
        route = (
            "Do not continue tuning the current contrast-visibility stem alone. Next single route should add an early "
            "ordinal roughness comparator inside the same water+concrete branch: compare smooth/slight/severe tokens "
            "by signed local contrast, dark-film quantile, and saturation micro-variation before the first major downsampling. "
            "Protect high-support concrete-slight/concrete-severe classes so the weak-class gain also reduces the Top-1 error count."
        )
        reason = (
            "The intended weak class did not gain enough, so the mechanism needs a sharper roughness-order comparison rather than a late correction."
            + pressure_text
        )
    elif spill_failed:
        route = (
            "Keep the useful target mechanism but redesign its router. Next single route should be a no-spill early "
            "mixture-of-couplings stem: one expert for water+concrete roughness, one for wet+concrete film texture, "
            "one identity/protection expert for unaffected classes, with a PhysicsTexture-derived gate and anchor non-regression. "
            "Gate updates by confidence and factor-family so gains on weak classes cannot erase correct predictions from easy/high-support classes."
        )
        reason = (
            "The target likely improved, but collateral drops block promotion. The next innovation should separate class families instead of increasing feature capacity."
            + pressure_text
        )
    elif macro_failed:
        route = (
            "Next single route should broaden the early coupling block from water+concrete to the concrete-slight/severe family, "
            "sharing stable gray_std/sat_std filters but using family-specific gates for dry, wet, and water friction states. "
            "The acceptance criterion must include net corrected samples, not Macro-F1 alone."
        )
        reason = "The local target may be acceptable, but macro-F1 did not improve enough across classes." + pressure_text
    else:
        route = (
            "Run a deeper audit before designing another route: inspect top drops, high-confidence errors, and stable cues. "
            "Do not start full training without a passed promotion decision."
        )
        reason = "The failure pattern is not specific enough from the available gates." + pressure_text

    return {
        "decision": "reject_or_redesign",
        "reason": reason,
        "single_next_route": route,
        "sota_pressure": {
            "required_extra_correct": required_extra,
            "required_error_reduction_share": required_error_share,
            "weakest_macro_targets": weakest_targets,
        },
        "observed_target_deltas": {
            "water_concrete_slight_delta_f1": wcs_delta,
            "water_concrete_severe_delta_f1": wcsev_delta,
            "net_fixed": net_fixed,
        },
        "top_f1_drops": top_drops,
        "top_f1_gains": top_gains,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Candidate Route Diagnosis")
    lines.append("")
    lines.append(f"- Candidate: `{payload['candidate_name']}`")
    lines.append(f"- Candidate dir: `{payload['candidate_dir']}`")
    lines.append(f"- Baseline dir: `{payload['baseline_dir']}`")
    lines.append(f"- Status: **{payload['status']}**")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Candidate | Baseline | Delta | SOTA target |")
    lines.append("|---|---:|---:|---:|---:|")
    for key in ["top1", "macro_f1", "hard_class_mean_f1"]:
        cand = payload["summary_delta"].get(key, {}).get("candidate")
        base = payload["summary_delta"].get(key, {}).get("baseline")
        delta = payload["summary_delta"].get(key, {}).get("delta")
        target = SOTA_TARGETS.get(key)
        lines.append(f"| {key} | {_pct(cand)} | {_pct(base)} | {_pp(delta)} | {_pct(target)} |")
    lines.append("")
    lines.append("## Promotion Gate Failures")
    if payload["failed_checks"]:
        lines.append("")
        lines.append("| Check | Candidate | Baseline | Delta | Threshold |")
        lines.append("|---|---:|---:|---:|---:|")
        for item in payload["failed_checks"]:
            lines.append(
                f"| {item.get('name')} | {_pct(item.get('candidate'))} | {_pct(item.get('baseline'))} | "
                f"{_pp(item.get('delta'))} | {_pp(item.get('threshold'))} |"
            )
    else:
        lines.append("")
        lines.append("- No failed promotion checks were available.")
    lines.append("")
    lines.append("## Key Class Deltas")
    lines.append("")
    lines.append("| Class | Candidate F1 | Baseline F1 | Delta | Targeted factor meaning |")
    lines.append("|---|---:|---:|---:|---|")
    for row in payload["key_class_deltas"]:
        factors = f"{row['friction']} + {row['material']} + {row['roughness']}"
        lines.append(
            f"| {row['class']} | {_pct(row['candidate_f1'])} | {_pct(row['baseline_f1'])} | {_pp(row['delta_f1'])} | {factors} |"
        )
    lines.append("")
    lines.append("## Factor-Level Damage")
    lines.append("")
    for factor_name, rows in payload["factor_damage"].items():
        lines.append(f"### {factor_name}")
        lines.append("")
        lines.append("| Factor | Mean F1 delta | Worst class delta | Classes |")
        lines.append("|---|---:|---:|---:|")
        for row in rows[:6]:
            lines.append(
                f"| {row['factor']} | {_pp(row['mean_delta_f1'])} | {_pp(row['min_delta_f1'])} | {row['num_classes']} |"
            )
        lines.append("")
    lines.append("## Stable Physics Cues To Respect")
    lines.append("")
    for pair, rows in list(payload["stable_cues"].items())[:6]:
        lines.append(f"- `{pair}`: " + ", ".join(f"{r['feature']} ({r['family']})" for r in rows[:3]))
    lines.append("")
    lines.append("## Route Decision")
    route = payload["next_route"]
    lines.append("")
    lines.append(f"- Decision: **{route['decision']}**")
    lines.append(f"- Reason: {route['reason']}")
    if route.get("sota_pressure"):
        pressure = route["sota_pressure"]
        lines.append(
            f"- SOTA pressure: required extra correct predictions `{pressure.get('required_extra_correct')}`, "
            f"required error reduction share `{_pct(pressure.get('required_error_reduction_share'))}`, "
            f"weakest Macro-F1 targets `{', '.join(pressure.get('weakest_macro_targets') or [])}`"
        )
    if route.get("single_next_route"):
        lines.append(f"- Next single route: {route['single_next_route']}")
    lines.append("")
    lines.append("## Implementation Constraint")
    lines.append("")
    lines.append(
        "Any next experiment must be implemented as a task-adapted RSCD/FAF mechanism: target friction, material, roughness, "
        "or their coupling explicitly; use PhysicsTexture/LocalPhysicsField/SemanticPhysicsAttention-style evidence where relevant; "
        "prefer early or mid-level conditioning; and compare under the same budget before full promotion."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose one RSCD candidate route and recommend exactly one next action.")
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--decision-json", type=Path, default=None)
    parser.add_argument("--comparison-json", type=Path, default=None)
    parser.add_argument("--stable-cue-json", type=Path, default=None)
    parser.add_argument("--sota-gap-json", type=Path, default=None)
    parser.add_argument("--full-protocol-required", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cand_summary = _summary(args.candidate_dir)
    base_summary = _summary(args.baseline_dir)
    decision = _read_json(args.decision_json) if args.decision_json else None
    comparison = _load_comparison(args.comparison_json)
    sota_gap = _load_sota_gap(args.sota_gap_json)
    status = _route_status(cand_summary, decision, args.full_protocol_required)

    cand_class = _read_per_class(args.candidate_dir / "per_class_metrics.csv")
    base_class = _read_per_class(args.baseline_dir / "per_class_metrics.csv")
    class_rows = _class_deltas(cand_class, base_class) if cand_class and base_class else []
    key_rows = [row for row in class_rows if row["class"] in KEY_CLASSES]
    key_rows.sort(key=lambda row: KEY_CLASSES.index(row["class"]) if row["class"] in KEY_CLASSES else 999)
    failed = _failed_checks(decision)
    stable = _stable_cue_top(args.stable_cue_json)

    def metric_delta(key: str) -> dict[str, float | None]:
        cand = _float(cand_summary, key)
        base = _float(base_summary, key)
        return {
            "candidate": cand,
            "baseline": base,
            "delta": (cand - base) if cand is not None and base is not None else None,
        }

    payload = {
        "ok": True,
        "candidate_name": args.candidate_name,
        "candidate_dir": str(args.candidate_dir),
        "baseline_dir": str(args.baseline_dir),
        "status": status,
        "candidate_summary_available": cand_summary is not None,
        "baseline_summary_available": base_summary is not None,
        "summary_delta": {
            "top1": metric_delta("top1"),
            "macro_f1": metric_delta("macro_f1"),
            "hard_class_mean_f1": metric_delta("hard_class_mean_f1"),
        },
        "failed_checks": failed,
        "key_class_deltas": key_rows,
        "factor_damage": {
            "friction": _factor_damage(class_rows, "friction") if class_rows else [],
            "material": _factor_damage(class_rows, "material") if class_rows else [],
            "roughness": _factor_damage(class_rows, "roughness") if class_rows else [],
            "friction_material": _factor_damage(class_rows, "friction_material") if class_rows else [],
            "material_roughness": _factor_damage(class_rows, "material_roughness") if class_rows else [],
        },
        "stable_cues": stable,
        "sota_gap": {
            "available": sota_gap is not None,
            "top1_budget": (sota_gap or {}).get("top1_budget"),
            "macro_uplift": (sota_gap or {}).get("macro_uplift"),
        },
        "next_route": _next_route(status, failed, class_rows, comparison, sota_gap),
    }
    _write_csv(args.output_dir / "class_deltas.csv", class_rows)
    (args.output_dir / "candidate_route_diagnosis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(payload, args.output_dir / "candidate_route_diagnosis.md")
    print(json.dumps({"status": status, "report": str(args.output_dir / "candidate_route_diagnosis.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
