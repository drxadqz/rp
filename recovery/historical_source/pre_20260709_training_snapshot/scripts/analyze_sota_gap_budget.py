from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOTA = {
    "best_top1": {"method": "RoadFormer-L", "value": 0.9286},
    "best_macro_f1": {"method": "RSPNet-L", "value": 0.8949},
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(run_dir: Path) -> dict[str, Any] | None:
    payload = _read_json(run_dir / "test_metrics.json")
    if payload is None:
        return None
    return dict(payload.get("summary", payload))


def _read_per_class(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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
            rows.append(
                {
                    "class": label,
                    "precision": float(row.get("precision") or 0.0),
                    "recall": float(row.get("recall") or 0.0),
                    "f1": float(row.get("f1") or 0.0),
                    "support": int(float(row.get("support") or 0.0)),
                    **_parse_label(label),
                }
            )
    return rows


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = str(row.get("true_label", ""))
            pred_label = str(row.get("pred_label", ""))
            rows.append(
                {
                    "image_path": str(row.get("image_path", "")),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": float(row.get("confidence") or 0.0),
                    "correct": true_label == pred_label,
                    "true_factors": _parse_label(true_label),
                    "pred_factors": _parse_label(pred_label),
                }
            )
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
    if summary is None or summary.get(key) is None:
        return None
    try:
        return float(summary[key])
    except (TypeError, ValueError):
        return None


def _macro_uplift_plan(per_class: list[dict[str, Any]], target_macro: float) -> dict[str, Any]:
    if not per_class:
        return {"required_sum_f1": None, "plan": []}
    macro = sum(float(row["f1"]) for row in per_class) / len(per_class)
    required_sum = max(0.0, target_macro - macro) * len(per_class)
    remaining = required_sum
    plan: list[dict[str, Any]] = []
    for row in sorted(per_class, key=lambda item: float(item["f1"])):
        if remaining <= 1e-12:
            break
        # Do not pretend one class should be pushed beyond the current target macro
        # unless the target cannot be met by lifting low-F1 classes to that level.
        natural_room = max(0.0, target_macro - float(row["f1"]))
        room = natural_room if natural_room > 0 else max(0.0, 1.0 - float(row["f1"]))
        take = min(room, remaining)
        if take <= 0:
            continue
        plan.append(
            {
                "class": row["class"],
                "current_f1": float(row["f1"]),
                "suggested_f1_gain": take,
                "suggested_target_f1": float(row["f1"]) + take,
                "support": row["support"],
                "friction": row["friction"],
                "material": row["material"],
                "roughness": row["roughness"],
            }
        )
        remaining -= take
    return {
        "current_macro_f1": macro,
        "target_macro_f1": target_macro,
        "required_sum_f1": required_sum,
        "unallocated_sum_f1": max(0.0, remaining),
        "plan": plan,
    }


def _error_budget(summary: dict[str, Any], predictions: list[dict[str, Any]], target_top1: float) -> dict[str, Any]:
    num_samples = int(float(summary.get("num_samples", len(predictions)) or len(predictions)))
    top1 = float(summary.get("top1", 0.0))
    correct = int(round(top1 * num_samples))
    target_correct = int(math.ceil(target_top1 * num_samples))
    required_extra = max(0, target_correct - correct)
    errors = [row for row in predictions if not row["correct"]]
    return {
        "num_samples": num_samples,
        "current_top1": top1,
        "target_top1": target_top1,
        "current_correct_est": correct,
        "target_correct_min": target_correct,
        "current_error_est": max(0, num_samples - correct),
        "required_extra_correct": required_extra,
        "required_error_reduction_share": (required_extra / len(errors)) if errors else None,
    }


def _factor_error_tables(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for factor_name in ["friction", "material", "roughness", "friction_material", "material_roughness"]:
        total = 0
        correct = 0
        confusion: Counter[tuple[str, str]] = Counter()
        per_true_total: Counter[str] = Counter()
        per_true_wrong: Counter[str] = Counter()
        for row in predictions:
            true_factor = row["true_factors"][factor_name]
            pred_factor = row["pred_factors"][factor_name]
            total += 1
            per_true_total[true_factor] += 1
            if true_factor == pred_factor:
                correct += 1
            else:
                per_true_wrong[true_factor] += 1
                confusion[(true_factor, pred_factor)] += 1
        per_true = []
        for factor, count in per_true_total.items():
            wrong = per_true_wrong[factor]
            per_true.append(
                {
                    "factor": factor,
                    "support": int(count),
                    "wrong": int(wrong),
                    "accuracy": float(1.0 - wrong / count) if count else 0.0,
                }
            )
        per_true.sort(key=lambda item: item["accuracy"])
        out[factor_name] = {
            "accuracy": float(correct / total) if total else 0.0,
            "per_true": per_true,
            "top_confusions": [
                {"true": key[0], "pred": key[1], "count": int(count)}
                for key, count in confusion.most_common(12)
            ],
        }
    return out


def _class_error_pressure(predictions: list[dict[str, Any]], per_class: list[dict[str, Any]]) -> list[dict[str, Any]]:
    error_by_true: Counter[str] = Counter()
    high_conf_error_by_true: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    for row in predictions:
        if row["correct"]:
            continue
        true_label = row["true_label"]
        pred_label = row["pred_label"]
        error_by_true[true_label] += 1
        if float(row["confidence"]) >= 0.75:
            high_conf_error_by_true[true_label] += 1
        confusion[(true_label, pred_label)] += 1
    class_lookup = {row["class"]: row for row in per_class}
    rows: list[dict[str, Any]] = []
    for label, count in error_by_true.items():
        metrics = class_lookup.get(label, {})
        rows.append(
            {
                "class": label,
                "errors": int(count),
                "high_conf_errors": int(high_conf_error_by_true[label]),
                "f1": float(metrics.get("f1", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "precision": float(metrics.get("precision", 0.0)),
                "support": int(metrics.get("support", 0)),
                **_parse_label(label),
            }
        )
    rows.sort(key=lambda item: (item["errors"], -item["f1"]), reverse=True)
    return rows


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
    lines.append("# RSCD SOTA Gap Budget")
    lines.append("")
    lines.append(f"- Run: `{payload['run_name']}`")
    lines.append(f"- Run dir: `{payload['run_dir']}`")
    lines.append(f"- Full protocol: `{payload['full_protocol']}`")
    lines.append("")
    lines.append("## Hard Targets")
    lines.append("")
    eb = payload["top1_budget"]
    mu = payload["macro_uplift"]
    lines.append(
        f"- Top-1 target: `{payload['targets']['top1_method']}` at {_pct(payload['targets']['top1'])}; "
        f"current {_pct(eb['current_top1'])}; needs **{eb['required_extra_correct']}** additional correct predictions."
    )
    lines.append(
        f"- Macro-F1 target: `{payload['targets']['macro_method']}` at {_pct(payload['targets']['macro_f1'])}; "
        f"current {_pct(mu.get('current_macro_f1'))}; needs total class-F1 uplift **{mu.get('required_sum_f1', 0.0):.4f}**."
    )
    if eb.get("required_error_reduction_share") is not None:
        lines.append(
            f"- This equals correcting about {_pct(eb['required_error_reduction_share'])} of current errors without adding new errors."
        )
    lines.append("")
    lines.append("## Macro-F1 Minimum Uplift Plan")
    lines.append("")
    lines.append("| Class | Current F1 | Suggested gain | Suggested target | Factor |")
    lines.append("|---|---:|---:|---:|---|")
    for row in mu.get("plan", [])[:12]:
        factor = f"{row['friction']} + {row['material']} + {row['roughness']}"
        lines.append(
            f"| {row['class']} | {_pct(row['current_f1'])} | {_pp(row['suggested_f1_gain'])} | "
            f"{_pct(row['suggested_target_f1'])} | {factor} |"
        )
    lines.append("")
    lines.append("## Error Pressure By Class")
    lines.append("")
    lines.append("| Class | Errors | High-conf errors | F1 | Recall | Factor |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in payload["class_error_pressure"][:12]:
        factor = f"{row['friction']} + {row['material']} + {row['roughness']}"
        lines.append(
            f"| {row['class']} | {row['errors']} | {row['high_conf_errors']} | {_pct(row['f1'])} | {_pct(row['recall'])} | {factor} |"
        )
    lines.append("")
    lines.append("## Weak Factor Accuracies")
    lines.append("")
    for factor_name, table in payload["factor_errors"].items():
        lines.append(f"### {factor_name}")
        lines.append("")
        lines.append("| Factor | Accuracy | Wrong | Support |")
        lines.append("|---|---:|---:|---:|")
        for row in table["per_true"][:6]:
            lines.append(f"| {row['factor']} | {_pct(row['accuracy'])} | {row['wrong']} | {row['support']} |")
        lines.append("")
    lines.append("## Mechanism Consequence")
    lines.append("")
    lines.append(
        "A route that only improves Macro-F1 but loses many easy examples cannot beat the Top-1 SOTA. "
        "The next mechanism must therefore combine a low-class uplift objective with a no-spill constraint on already reliable factors."
    )
    lines.append(
        "For RSCD, the strongest current pressure remains coupled labels involving water/wet film, concrete material, and slight/severe roughness. "
        "This supports early factor-conditioned coupling rather than another late classifier correction."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantify how far an RSCD run is from public SOTA targets.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top1-target", type=float, default=DEFAULT_SOTA["best_top1"]["value"])
    parser.add_argument("--top1-method", default=DEFAULT_SOTA["best_top1"]["method"])
    parser.add_argument("--macro-f1-target", type=float, default=DEFAULT_SOTA["best_macro_f1"]["value"])
    parser.add_argument("--macro-method", default=DEFAULT_SOTA["best_macro_f1"]["method"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.run_dir.name
    summary = _summary(args.run_dir)
    per_class = _read_per_class(args.run_dir / "per_class_metrics.csv")
    predictions = _read_predictions(args.run_dir / "predictions_test.csv")
    if summary is None or not per_class:
        payload = {
            "ok": False,
            "run_name": run_name,
            "run_dir": str(args.run_dir),
            "missing": [
                str(path)
                for path in [args.run_dir / "test_metrics.json", args.run_dir / "per_class_metrics.csv"]
                if not path.exists()
            ],
        }
        (args.output_dir / "sota_gap_budget.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    top1_budget = _error_budget(summary, predictions, args.top1_target)
    macro_uplift = _macro_uplift_plan(per_class, args.macro_f1_target)
    factor_errors = _factor_error_tables(predictions)
    class_pressure = _class_error_pressure(predictions, per_class)
    full_protocol = int(float(summary.get("num_samples", 0) or 0)) >= 40000
    payload = {
        "ok": True,
        "run_name": run_name,
        "run_dir": str(args.run_dir),
        "full_protocol": full_protocol,
        "targets": {
            "top1": args.top1_target,
            "top1_method": args.top1_method,
            "macro_f1": args.macro_f1_target,
            "macro_method": args.macro_method,
        },
        "summary": summary,
        "top1_budget": top1_budget,
        "macro_uplift": macro_uplift,
        "factor_errors": factor_errors,
        "class_error_pressure": class_pressure,
    }
    _write_csv(args.output_dir / "class_error_pressure.csv", class_pressure)
    _write_csv(args.output_dir / "macro_f1_uplift_plan.csv", macro_uplift.get("plan", []))
    for factor_name, table in factor_errors.items():
        _write_csv(args.output_dir / f"{factor_name}_factor_accuracy.csv", table["per_true"])
        _write_csv(args.output_dir / f"{factor_name}_factor_confusions.csv", table["top_confusions"])
    (args.output_dir / "sota_gap_budget.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(payload, args.output_dir / "sota_gap_budget.md")
    print(json.dumps({"ok": True, "report": str(args.output_dir / "sota_gap_budget.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
