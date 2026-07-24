from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


KEY_CLASSES = {
    "water_concrete_slight",
    "water_concrete_severe",
    "water_concrete_smooth",
    "wet_concrete_slight",
    "wet_concrete_severe",
    "dry_concrete_slight",
}


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


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = str(row["image_path"])
            out[image_path] = {
                "true_label": str(row["true_label"]),
                "pred_label": str(row["pred_label"]),
                "confidence": float(row.get("confidence") or 0.0),
            }
    return out


def _pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def _pp(value: float) -> str:
    return f"{100.0 * value:+.3f} pp"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two RSCD run directories with per-class and prediction deltas.")
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    output_dir = args.output_dir or args.candidate_dir / "compare_to_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    cand_metrics_path = args.candidate_dir / "test_metrics.json"
    base_metrics_path = args.baseline_dir / "test_metrics.json"
    cand_pred_path = args.candidate_dir / "predictions_test.csv"
    base_pred_path = args.baseline_dir / "predictions_test.csv"
    cand_class_path = args.candidate_dir / "per_class_metrics.csv"
    base_class_path = args.baseline_dir / "per_class_metrics.csv"
    missing = [
        str(path)
        for path in [cand_metrics_path, base_metrics_path, cand_pred_path, base_pred_path, cand_class_path, base_class_path]
        if not path.exists()
    ]
    if missing:
        payload = {"ok": False, "missing": missing}
        (output_dir / "run_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    cand_summary = _read_json(cand_metrics_path).get("summary", {})
    base_summary = _read_json(base_metrics_path).get("summary", {})
    cand_class = _read_per_class(cand_class_path)
    base_class = _read_per_class(base_class_path)
    cand_pred = _read_predictions(cand_pred_path)
    base_pred = _read_predictions(base_pred_path)
    common_paths = sorted(set(cand_pred) & set(base_pred))

    corrected: Counter[str] = Counter()
    worsened: Counter[str] = Counter()
    unchanged_correct: Counter[str] = Counter()
    unchanged_wrong: Counter[str] = Counter()
    changed_transitions: Counter[tuple[str, str, str, str]] = Counter()
    fixed_transitions: Counter[tuple[str, str]] = Counter()
    broken_transitions: Counter[tuple[str, str]] = Counter()
    changed_rows: list[dict[str, Any]] = []
    for path in common_paths:
        c = cand_pred[path]
        b = base_pred[path]
        true_label = str(c["true_label"])
        if true_label != str(b["true_label"]):
            continue
        cand_ok = c["pred_label"] == true_label
        base_ok = b["pred_label"] == true_label
        if cand_ok and not base_ok:
            corrected[true_label] += 1
            fixed_transitions[(true_label, str(b["pred_label"]))] += 1
        elif base_ok and not cand_ok:
            worsened[true_label] += 1
            broken_transitions[(true_label, str(c["pred_label"]))] += 1
        elif cand_ok and base_ok:
            unchanged_correct[true_label] += 1
        else:
            unchanged_wrong[true_label] += 1
        if c["pred_label"] != b["pred_label"]:
            changed_transitions[(true_label, str(b["pred_label"]), str(c["pred_label"]), "changed")] += 1
            changed_rows.append(
                {
                    "image_path": path,
                    "true_label": true_label,
                    "baseline_pred": b["pred_label"],
                    "candidate_pred": c["pred_label"],
                    "baseline_confidence": b["confidence"],
                    "candidate_confidence": c["confidence"],
                    "status": "fixed" if cand_ok and not base_ok else ("worsened" if base_ok and not cand_ok else "changed_wrong"),
                }
            )

    class_names = sorted(set(cand_class) | set(base_class))
    class_rows: list[dict[str, Any]] = []
    for name in class_names:
        c = cand_class.get(name, {})
        b = base_class.get(name, {})
        row = {
            "class": name,
            "candidate_f1": float(c.get("f1", 0.0)),
            "baseline_f1": float(b.get("f1", 0.0)),
            "delta_f1": float(c.get("f1", 0.0)) - float(b.get("f1", 0.0)),
            "candidate_precision": float(c.get("precision", 0.0)),
            "baseline_precision": float(b.get("precision", 0.0)),
            "delta_precision": float(c.get("precision", 0.0)) - float(b.get("precision", 0.0)),
            "candidate_recall": float(c.get("recall", 0.0)),
            "baseline_recall": float(b.get("recall", 0.0)),
            "delta_recall": float(c.get("recall", 0.0)) - float(b.get("recall", 0.0)),
            "support": float(c.get("support", b.get("support", 0.0))),
            "fixed_count": int(corrected[name]),
            "worsened_count": int(worsened[name]),
            "net_fixed": int(corrected[name] - worsened[name]),
            "key_class": name in KEY_CLASSES,
        }
        class_rows.append(row)
    class_rows.sort(key=lambda row: row["delta_f1"])

    with (output_dir / "per_class_delta.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()) if class_rows else ["class"])
        writer.writeheader()
        writer.writerows(class_rows)
    with (output_dir / "changed_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "image_path",
            "true_label",
            "baseline_pred",
            "candidate_pred",
            "baseline_confidence",
            "candidate_confidence",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changed_rows)

    def counter_rows(counter: Counter, names: list[str]) -> list[dict[str, Any]]:
        rows = []
        for key, count in counter.most_common(args.top_k):
            values = list(key) if isinstance(key, tuple) else [key]
            rows.append({**{name: values[i] for i, name in enumerate(names)}, "count": int(count)})
        return rows

    payload = {
        "ok": True,
        "candidate_name": args.candidate_name,
        "baseline_name": args.baseline_name,
        "candidate_dir": str(args.candidate_dir),
        "baseline_dir": str(args.baseline_dir),
        "common_predictions": len(common_paths),
        "candidate_summary": {
            key: cand_summary.get(key)
            for key in ["top1", "macro_f1", "mean_precision", "mean_recall", "hard_class_mean_f1", "num_errors", "num_samples"]
        },
        "baseline_summary": {
            key: base_summary.get(key)
            for key in ["top1", "macro_f1", "mean_precision", "mean_recall", "hard_class_mean_f1", "num_errors", "num_samples"]
        },
        "summary_delta": {
            key: float(cand_summary.get(key, 0.0)) - float(base_summary.get(key, 0.0))
            for key in ["top1", "macro_f1", "mean_precision", "mean_recall", "hard_class_mean_f1"]
        },
        "total_fixed": int(sum(corrected.values())),
        "total_worsened": int(sum(worsened.values())),
        "net_fixed": int(sum(corrected.values()) - sum(worsened.values())),
        "top_f1_gains": sorted(class_rows, key=lambda row: row["delta_f1"], reverse=True)[: args.top_k],
        "top_f1_drops": class_rows[: args.top_k],
        "top_fixed_transitions": counter_rows(fixed_transitions, ["true_label", "baseline_wrong_pred"]),
        "top_broken_transitions": counter_rows(broken_transitions, ["true_label", "candidate_wrong_pred"]),
    }
    (output_dir / "run_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        f"# {args.candidate_name} vs {args.baseline_name}",
        "",
        f"- Candidate: `{args.candidate_dir}`",
        f"- Baseline: `{args.baseline_dir}`",
        f"- Common prediction rows: {len(common_paths)}",
        f"- Fixed / worsened / net: {payload['total_fixed']} / {payload['total_worsened']} / {payload['net_fixed']}",
        "",
        "## Summary Delta",
        "",
        "| Metric | Candidate | Baseline | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in ["top1", "macro_f1", "mean_precision", "mean_recall", "hard_class_mean_f1"]:
        md.append(
            f"| {key} | {_pct(float(cand_summary.get(key, 0.0)))} | "
            f"{_pct(float(base_summary.get(key, 0.0)))} | {_pp(payload['summary_delta'][key])} |"
        )
    md.extend(["", "## Key Class Delta", "", "| Class | F1 delta | Candidate F1 | Baseline F1 | Fixed | Worsened | Net |", "|---|---:|---:|---:|---:|---:|---:|"])
    key_rows = [row for row in class_rows if row["key_class"]]
    for row in sorted(key_rows, key=lambda item: item["class"]):
        md.append(
            f"| {row['class']} | {_pp(row['delta_f1'])} | {_pct(row['candidate_f1'])} | "
            f"{_pct(row['baseline_f1'])} | {row['fixed_count']} | {row['worsened_count']} | {row['net_fixed']} |"
        )
    md.extend(["", "## Largest F1 Gains", "", "| Class | Delta | Candidate F1 | Baseline F1 | Net fixed |", "|---|---:|---:|---:|---:|"])
    for row in payload["top_f1_gains"][:10]:
        md.append(f"| {row['class']} | {_pp(row['delta_f1'])} | {_pct(row['candidate_f1'])} | {_pct(row['baseline_f1'])} | {row['net_fixed']} |")
    md.extend(["", "## Largest F1 Drops", "", "| Class | Delta | Candidate F1 | Baseline F1 | Net fixed |", "|---|---:|---:|---:|---:|"])
    for row in payload["top_f1_drops"][:10]:
        md.append(f"| {row['class']} | {_pp(row['delta_f1'])} | {_pct(row['candidate_f1'])} | {_pct(row['baseline_f1'])} | {row['net_fixed']} |")
    (output_dir / "run_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": str(output_dir / "run_comparison.md"), "ok": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
