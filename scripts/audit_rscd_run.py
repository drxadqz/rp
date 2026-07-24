from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOTA = {
    "RSPNet-L": {"top1": 0.9201, "macro_f1": 0.8949},
    "RoadFormer-L": {"top1": 0.9286, "macro_f1": 0.8499},
    "RoadMamba-B": {"top1": 0.9281, "macro_f1": 0.8479},
}

FULL_PROTOCOL_MIN_SAMPLES = 40000


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_per_class(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = [str(name).lstrip("\ufeff") for name in (reader.fieldnames or [])]
        class_field = "class" if "class" in fields else (fields[0] if fields else "class")
        for row in reader:
            if class_field not in row and f"\ufeff{class_field}" in row:
                class_field = f"\ufeff{class_field}"
            name = str(row.get(class_field, ""))
            if not name:
                continue
            out[name] = {
                "precision": float(row.get("precision") or 0.0),
                "recall": float(row.get("recall") or 0.0),
                "f1": float(row.get("f1") or 0.0),
                "support": float(row.get("support") or 0.0),
            }
    return out


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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "image_path": str(row.get("image_path", "")),
                    "true_label": str(row.get("true_label", "")),
                    "pred_label": str(row.get("pred_label", "")),
                    "confidence": float(row.get("confidence") or 0.0),
                }
            )
    return rows


def _counter_table(counter: Counter, fields: list[str], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, count in counter.most_common(top_k):
        values = list(key) if isinstance(key, tuple) else [key]
        rows.append({field: values[idx] for idx, field in enumerate(fields)} | {"count": int(count)})
    return rows


def _group_class_f1(per_class: dict[str, dict[str, float]], factor_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    supports: Counter[str] = Counter()
    for label, metrics in per_class.items():
        factor = _parse_label(label)[factor_name]
        grouped[factor].append(float(metrics.get("f1", 0.0)))
        supports[factor] += int(float(metrics.get("support", 0.0)))
    rows = []
    for factor, values in grouped.items():
        rows.append(
            {
                "factor": factor,
                "macro_f1": float(_mean(values) or 0.0),
                "num_classes": len(values),
                "support": int(supports[factor]),
            }
        )
    return sorted(rows, key=lambda row: row["macro_f1"])


def _factor_accuracy(predictions: list[dict[str, Any]], factor_name: str) -> dict[str, Any]:
    correct = 0
    total = 0
    confusions: Counter[tuple[str, str]] = Counter()
    per_true_total: Counter[str] = Counter()
    per_true_correct: Counter[str] = Counter()
    for row in predictions:
        true_factor = _parse_label(str(row["true_label"]))[factor_name]
        pred_factor = _parse_label(str(row["pred_label"]))[factor_name]
        total += 1
        per_true_total[true_factor] += 1
        if true_factor == pred_factor:
            correct += 1
            per_true_correct[true_factor] += 1
        else:
            confusions[(true_factor, pred_factor)] += 1
    per_true = []
    for factor, count in per_true_total.items():
        per_true.append(
            {
                "factor": factor,
                "accuracy": float(per_true_correct[factor] / count) if count else 0.0,
                "support": int(count),
            }
        )
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "total": int(total),
        "per_true": sorted(per_true, key=lambda row: row["accuracy"]),
        "top_confusions": _counter_table(confusions, ["true_factor", "pred_factor"], 20),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit one RSCD run for fair SOTA comparison and failure localization.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    run_dir = args.run_dir
    run_name = args.run_name or run_dir.name
    output_dir = args.output_dir or (run_dir / "fair_sota_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [run_dir / "test_metrics.json", run_dir / "per_class_metrics.csv", run_dir / "predictions_test.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"ok": False, "missing": missing, "run_dir": str(run_dir)}
        (output_dir / "fair_sota_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    metrics = _read_json(run_dir / "test_metrics.json")
    summary = dict(metrics.get("summary", metrics))
    per_class = _read_per_class(run_dir / "per_class_metrics.csv")
    predictions = _read_predictions(run_dir / "predictions_test.csv")

    num_samples = int(summary.get("num_samples", len(predictions)) or 0)
    top1 = float(summary.get("top1", 0.0))
    macro_f1 = float(summary.get("macro_f1", 0.0))
    full_protocol = num_samples >= FULL_PROTOCOL_MIN_SAMPLES
    best_top1 = max(item["top1"] for item in DEFAULT_SOTA.values())
    best_macro = max(item["macro_f1"] for item in DEFAULT_SOTA.values())

    class_rows = []
    for label, item in per_class.items():
        factors = _parse_label(label)
        class_rows.append(
            {
                "class": label,
                "precision": float(item.get("precision", 0.0)),
                "recall": float(item.get("recall", 0.0)),
                "f1": float(item.get("f1", 0.0)),
                "support": int(float(item.get("support", 0.0))),
                **factors,
            }
        )
    class_rows.sort(key=lambda row: row["f1"])
    _write_csv(output_dir / "classes_by_f1.csv", class_rows)

    label_confusions: Counter[tuple[str, str]] = Counter()
    confidence_wrong: list[dict[str, Any]] = []
    for row in predictions:
        true_label = str(row["true_label"])
        pred_label = str(row["pred_label"])
        if true_label != pred_label:
            label_confusions[(true_label, pred_label)] += 1
            confidence_wrong.append(row)
    confidence_wrong.sort(key=lambda row: float(row["confidence"]), reverse=True)
    top_wrong_rows = confidence_wrong[: args.top_k]
    _write_csv(output_dir / "top_high_confidence_errors.csv", top_wrong_rows)

    factor_names = [
        "friction",
        "material",
        "roughness",
        "friction_material",
        "friction_roughness",
        "material_roughness",
    ]
    factor_accuracy = {name: _factor_accuracy(predictions, name) for name in factor_names}
    grouped_f1 = {name: _group_class_f1(per_class, name) for name in factor_names}

    payload = {
        "ok": True,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "num_samples": num_samples,
        "full_protocol": full_protocol,
        "summary": {
            "top1": top1,
            "macro_f1": macro_f1,
            "mean_precision": summary.get("mean_precision"),
            "mean_recall": summary.get("mean_recall"),
            "weighted_f1": summary.get("weighted_f1"),
            "num_errors": summary.get("num_errors"),
            "hard_class_mean_f1": summary.get("hard_class_mean_f1"),
        },
        "sota_gates": {
            "top1_target": best_top1,
            "macro_f1_target": best_macro,
            "top1_delta_to_best": top1 - best_top1,
            "macro_f1_delta_to_best": macro_f1 - best_macro,
            "strict_sota_pass": bool(full_protocol and top1 > best_top1 and macro_f1 > best_macro),
        },
        "bottom_classes": class_rows[: args.top_k],
        "top_label_confusions": _counter_table(label_confusions, ["true_label", "pred_label"], args.top_k),
        "factor_accuracy": factor_accuracy,
        "grouped_class_f1": grouped_f1,
    }
    (output_dir / "fair_sota_audit.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        f"# RSCD fair SOTA audit: {run_name}",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Full protocol: **{full_protocol}** ({num_samples} samples; threshold {FULL_PROTOCOL_MIN_SAMPLES})",
        f"- Top-1: **{_pct(top1)}** ({_pp(top1 - best_top1)} vs best public Top-1 reference)",
        f"- Macro-F1: **{_pct(macro_f1)}** ({_pp(macro_f1 - best_macro)} vs best public Macro-F1 reference)",
        f"- Strict pass over current references: **{payload['sota_gates']['strict_sota_pass']}**",
        "",
        "## Public Reference Gates",
        "",
        "| Method | Top-1 | Macro-F1/F1 | Top-1 delta | Macro-F1 delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, vals in DEFAULT_SOTA.items():
        md.append(
            f"| {name} | {_pct(vals['top1'])} | {_pct(vals['macro_f1'])} | "
            f"{_pp(top1 - vals['top1'])} | {_pp(macro_f1 - vals['macro_f1'])} |"
        )

    md.extend(["", "## Bottom Classes", "", "| Class | F1 | Precision | Recall | Support | Factors |", "|---|---:|---:|---:|---:|---|"])
    for row in class_rows[: args.top_k]:
        md.append(
            f"| {row['class']} | {_pct(row['f1'])} | {_pct(row['precision'])} | {_pct(row['recall'])} | "
            f"{row['support']} | {row['friction']} / {row['material']} / {row['roughness']} |"
        )

    md.extend(["", "## Factor-Level Accuracy", "", "| Factor view | Accuracy | Weakest true factor | Weakest factor accuracy |", "|---|---:|---|---:|"])
    for name in factor_names:
        item = factor_accuracy[name]
        weakest = item["per_true"][0] if item["per_true"] else {"factor": "-", "accuracy": None}
        md.append(f"| {name} | {_pct(item['accuracy'])} | {weakest['factor']} | {_pct(weakest['accuracy'])} |")

    md.extend(["", "## Worst Factor Groups by Class Macro-F1", ""])
    for name in ["friction", "material", "roughness", "friction_material", "material_roughness"]:
        md.extend([f"### {name}", "", "| Group | Macro-F1 | Classes | Support |", "|---|---:|---:|---:|"])
        for row in grouped_f1[name][: min(args.top_k, 8)]:
            md.append(f"| {row['factor']} | {_pct(row['macro_f1'])} | {row['num_classes']} | {row['support']} |")
        md.append("")

    md.extend(["## Top Label Confusions", "", "| True label | Pred label | Count |", "|---|---|---:|"])
    for row in payload["top_label_confusions"]:
        md.append(f"| {row['true_label']} | {row['pred_label']} | {row['count']} |")

    md.extend(["", "## Diagnosis for Next Mechanism", ""])
    weakest_factor = min(
        ((name, factor_accuracy[name]["accuracy"]) for name in ["friction", "material", "roughness"]),
        key=lambda item: item[1],
    )
    worst_class = class_rows[0] if class_rows else None
    if worst_class:
        md.append(
            f"- The weakest class is `{worst_class['class']}` with F1 {_pct(worst_class['f1'])}; "
            f"its factor form is {worst_class['friction']} / {worst_class['material']} / {worst_class['roughness']}."
        )
    md.append(
        f"- Among the three semantic factors, `{weakest_factor[0]}` has the lowest prediction consistency "
        f"({_pct(weakest_factor[1])}), so the next RSCD-adapted mechanism should target this factor or its coupling."
    )
    md.append(
        "- This audit is read-only and does not validate a new algorithm by itself; it only provides evidence for choosing the next single experiment."
    )

    (output_dir / "fair_sota_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(output_dir / "fair_sota_audit.md"), "ok": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
