from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ANCHOR_DIR = Path(
    "E:/perception_outputs/rscd_surface_classification/"
    "c3_farnet_official_anchor_source_reliable_router_s7_fulltest_20260708/fast_test"
)
DEFAULT_OUT_DIR = Path("reports/paper_protocol_summary/s7_anchor_error_diagnosis_20260709")


def parse_paved_label(label: str) -> tuple[str, str, str] | None:
    parts = label.split("_")
    if len(parts) != 3:
        return None
    friction, material, roughness = parts
    if friction not in {"dry", "wet", "water"}:
        return None
    if material not in {"asphalt", "concrete"}:
        return None
    if roughness not in {"smooth", "slight", "severe"}:
        return None
    return friction, material, roughness


def read_per_class(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            f1_key = "f1-score" if "f1-score" in row else "f1"
            rows.append(
                {
                    "class": row["class"],
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row[f1_key]),
                    "support": float(row["support"]),
                }
            )
    return rows


def read_confusion(path: Path) -> tuple[list[str], list[list[int]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = header[1:]
        matrix: list[list[int]] = []
        row_labels: list[str] = []
        for row in reader:
            row_labels.append(row[0])
            matrix.append([int(float(v)) for v in row[1:]])
    if labels != row_labels:
        raise ValueError("confusion matrix row/column labels do not match")
    return labels, matrix


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def diagnose(anchor_dir: Path, out_dir: Path) -> int:
    metrics = json.loads((anchor_dir / "metrics.json").read_text(encoding="utf-8"))
    per_class = read_per_class(anchor_dir / "per_class_metrics.csv")
    labels, matrix = read_confusion(anchor_dir / "confusion_matrix.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    bottom = sorted(per_class, key=lambda row: row["f1"])[:10]
    write_csv(
        out_dir / "bottom_10_classes.csv",
        [
            {
                "class": row["class"],
                "precision_%": 100.0 * row["precision"],
                "recall_%": 100.0 * row["recall"],
                "f1_%": 100.0 * row["f1"],
                "support": row["support"],
            }
            for row in bottom
        ],
        ["class", "precision_%", "recall_%", "f1_%", "support"],
    )

    pair_rows: list[dict[str, Any]] = []
    factor_error_counts: dict[str, int] = defaultdict(int)
    paved_error_count = 0
    for i, true_label in enumerate(labels):
        true_factor = parse_paved_label(true_label)
        for j, pred_label in enumerate(labels):
            if i == j:
                continue
            count = int(matrix[i][j])
            if count <= 0:
                continue
            pred_factor = parse_paved_label(pred_label)
            factor_diff: list[str] = []
            if true_factor is not None and pred_factor is not None:
                for axis_name, a, b in zip(("friction", "material", "roughness"), true_factor, pred_factor, strict=True):
                    if a != b:
                        factor_diff.append(axis_name)
                        factor_error_counts[axis_name] += count
                paved_error_count += count
            pair_rows.append(
                {
                    "true": true_label,
                    "pred": pred_label,
                    "count": count,
                    "factor_diff": "+".join(factor_diff) if factor_diff else "non_paved_or_same_factor",
                }
            )
    pair_rows.sort(key=lambda row: int(row["count"]), reverse=True)
    write_csv(out_dir / "top_confusion_pairs.csv", pair_rows[:60], ["true", "pred", "count", "factor_diff"])

    weakness = {row["class"]: row for row in per_class}
    route_notes: list[str] = []
    for true_name, pred_name in [
        ("water_concrete_slight", "water_concrete_severe"),
        ("water_concrete_slight", "wet_concrete_slight"),
        ("dry_concrete_slight", "dry_concrete_severe"),
        ("dry_concrete_severe", "dry_concrete_slight"),
        ("wet_concrete_slight", "water_concrete_slight"),
        ("wet_concrete_severe", "water_concrete_severe"),
    ]:
        if true_name in labels and pred_name in labels:
            route_notes.append(f"- `{true_name} -> {pred_name}`: {matrix[labels.index(true_name)][labels.index(pred_name)]} errors")

    total_errors = int(metrics["summary"]["num_errors"])
    factor_lines = []
    for axis in ("friction", "material", "roughness"):
        count = factor_error_counts.get(axis, 0)
        share = count / max(1, paved_error_count)
        factor_lines.append(f"- {axis}: {count} paved-error hits, {100.0 * share:.2f}% of paved factor-error hits")

    bottom_lines = [
        f"- `{row['class']}`: F1={pct(float(row['f1']))}, P={pct(float(row['precision']))}, R={pct(float(row['recall']))}, support={int(row['support'])}"
        for row in bottom
    ]
    pair_lines = [
        f"- `{row['true']} -> {row['pred']}`: {row['count']} errors ({row['factor_diff']})"
        for row in pair_rows[:15]
    ]

    md = [
        "# S7 Anchor Error Diagnosis",
        "",
        f"- Anchor dir: `{anchor_dir}`",
        f"- Top-1: {pct(float(metrics['summary']['top1']))}",
        f"- Macro-F1: {pct(float(metrics['summary']['macro_f1']))}",
        f"- Total errors: {total_errors}",
        "",
        "## Bottom Classes",
        "",
        *bottom_lines,
        "",
        "## Top Confusion Pairs",
        "",
        *pair_lines,
        "",
        "## Paved Factor Error Shares",
        "",
        *factor_lines,
        "",
        "## Route-Relevant Counts",
        "",
        *route_notes,
        "",
        "## Interpretation",
        "",
        "- The next architecture change should still be pair-local and factor-aware: most useful fixes are boundaries where the label differs by one paved factor, especially roughness within concrete or wet/water concrete.",
        "- The prepared S8 WCS incoming route is justified only if it reduces `water_concrete_slight` misses without increasing severe-to-slight false positives on wet/water concrete.",
        "- Any later early/mid backbone change should report these same top pairs before and after training, because aggregate Top-1 can hide roughness-family damage.",
    ]
    (out_dir / "s7_anchor_error_diagnosis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "out_dir": str(out_dir)}, ensure_ascii=False))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose S7 anchor per-class and confusion failures.")
    parser.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    raise SystemExit(diagnose(args.anchor_dir, args.out_dir))


if __name__ == "__main__":
    main()
