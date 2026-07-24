from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", type=Path, required=True)
    parser.add_argument("--task", default="friction")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    detailed = json.loads(args.detailed.read_text(encoding="utf-8"))
    task = detailed.get("tasks", {}).get(args.task)
    if not task:
        raise SystemExit(f"Task not found in detailed file: {args.task}")
    node = task
    title_suffix = "overall"
    if args.dataset:
        node = task.get("by_dataset", {}).get(args.dataset)
        title_suffix = f"dataset={args.dataset}"
        if not node:
            raise SystemExit(f"Dataset not found for task {args.task}: {args.dataset}")

    rows = build_rows(node, args.task, title_suffix, args.top_k)
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["task", "scope"])
            writer.writeheader()
            writer.writerows(rows)
    md = render_markdown(rows, args.task, title_suffix)
    print(md)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")


def build_rows(node: dict[str, Any], task: str, scope: str, top_k: int) -> list[dict[str, Any]]:
    labels = node.get("confusion_matrix_labels", [])
    matrix = node.get("confusion_matrix", [])
    per_class = node.get("per_class_f1", {})
    rows: list[dict[str, Any]] = []
    for i, true_label in enumerate(labels):
        support = sum(matrix[i]) if i < len(matrix) else 0
        rows.append(
            {
                "kind": "per_class",
                "task": task,
                "scope": scope,
                "true_label": true_label,
                "pred_label": "",
                "count": support,
                "class_f1": per_class.get(true_label),
            }
        )
    confusions = []
    for i, true_label in enumerate(labels):
        if i >= len(matrix):
            continue
        for j, pred_label in enumerate(labels):
            if i == j or j >= len(matrix[i]):
                continue
            count = int(matrix[i][j])
            if count > 0:
                confusions.append(
                    {
                        "kind": "confusion",
                        "task": task,
                        "scope": scope,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": count,
                        "class_f1": "",
                    }
                )
    rows.extend(sorted(confusions, key=lambda row: row["count"], reverse=True)[:top_k])
    return rows


def render_markdown(rows: list[dict[str, Any]], task: str, scope: str) -> str:
    lines = [f"# Confusion Summary: {task} ({scope})", ""]
    lines.append("## Per-Class F1")
    lines.append("")
    lines.append("| class | support | F1 |")
    lines.append("|---|---:|---:|")
    for row in [item for item in rows if item["kind"] == "per_class"]:
        lines.append(f"| {row['true_label']} | {row['count']} | {fmt(row['class_f1'])} |")
    lines.append("")
    lines.append("## Top Confusions")
    lines.append("")
    lines.append("| true | predicted | count |")
    lines.append("|---|---|---:|")
    for row in [item for item in rows if item["kind"] == "confusion"]:
        lines.append(f"| {row['true_label']} | {row['pred_label']} | {row['count']} |")
    lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value in {None, ""}:
        return "-"
    return f"{100.0 * float(value):.2f}"


if __name__ == "__main__":
    main()
